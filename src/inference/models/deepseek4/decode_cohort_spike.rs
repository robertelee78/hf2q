//! Test-only B=4 decode-body performance falsifier.

use anyhow::{Context, Result};
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::{DType, GraphExecutor, MlxBuffer};

use super::cache::Deepseek4Cache;
use super::forward_support::{alloc, begin_decode_pool_token, end_decode_pool_token};
use super::submission::{drain, retained_reference_pipeline_enabled, SubmissionChain};
use super::Deepseek4Model;

pub(super) fn publish_verifier_cohort_after_gate(
    caches: &mut [&mut Deepseek4Cache; 4],
    positions: [usize; 4],
    commit_gate: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let tickets = positions
        .iter()
        .zip(caches.iter())
        .map(|(&position, cache)| {
            cache
                .validate_step_commit(position)
                .context("prevalidate DeepSeek-V4 B=4 decode publication")
        })
        .collect::<Result<Vec<_>>>();
    let tickets = match tickets {
        Ok(tickets) => tickets,
        Err(error) => {
            for cache in caches.iter_mut() {
                cache.poison();
            }
            return Err(error);
        }
    };
    if let Err(error) = commit_gate() {
        for cache in caches.iter_mut() {
            cache.poison();
        }
        return Err(error).context("DeepSeek-V4 B=4 decode rejected before cache publication");
    }
    for (cache, ticket) in caches.iter_mut().zip(tickets) {
        cache.publish_step_end(ticket);
    }
    Ok(())
}

impl Deepseek4Model {
    /// Execute four independent cache attentions in a common retained
    /// command-buffer chain, then pack their rows for one batched FFN graph.
    pub(super) fn forward_verifier_decode_cohort(
        &mut self,
        token_ids: [u32; 4],
        caches: &mut [&mut Deepseek4Cache; 4],
    ) -> Result<MlxBuffer> {
        anyhow::ensure!(
            retained_reference_pipeline_enabled(),
            "DeepSeek-V4 B=4 decode requires retained Metal command-buffer references"
        );
        let positions = std::array::from_fn(|lane| caches[lane].position());
        anyhow::ensure!(
            positions.iter().all(|&position| position == positions[0]),
            "DeepSeek-V4 B=4 decode requires equal cache positions, got {positions:?}"
        );
        anyhow::ensure!(
            caches.iter().all(|cache| cache.plan == caches[0].plan),
            "DeepSeek-V4 B=4 decode requires identical cache plans"
        );
        for cache in caches.iter() {
            cache
                .plan_next_step()
                .context("plan DeepSeek-V4 B=4 decode cache transaction")?;
        }

        begin_decode_pool_token();
        let (result, submitted_any) =
            self.forward_verifier_decode_cohort_uncommitted(&token_ids, caches);
        end_decode_pool_token();
        match result {
            Ok(state) => {
                publish_verifier_cohort_after_gate(caches, positions, || Ok(()))?;
                Ok(state)
            }
            Err(error) => {
                if submitted_any {
                    for cache in caches.iter_mut() {
                        cache.poison();
                    }
                }
                Err(error).context("DeepSeek-V4 B=4 decode transaction failed")
            }
        }
    }

    fn forward_verifier_decode_cohort_uncommitted(
        &mut self,
        token_ids: &[u32; 4],
        caches: &mut [&mut Deepseek4Cache; 4],
    ) -> (Result<MlxBuffer>, bool) {
        const LAYERS_PER_COMMAND_BUFFER: usize = 2;

        let mut submitted_any = false;
        let result = (|| -> Result<MlxBuffer> {
            let layers = self.cfg.num_hidden_layers as usize;
            let hc = self.cfg.hyper_connection_count as usize;
            let hidden = self.cfg.hidden_size as usize;
            let row_elements = hc
                .checked_mul(hidden)
                .context("DeepSeek-V4 B=4 decode row width overflow")?;
            let combined_elements = 4_usize
                .checked_mul(row_elements)
                .context("DeepSeek-V4 B=4 decode state size overflow")?;
            let device = self.ctx.device().clone();
            let executor = GraphExecutor::new(device.clone());
            let layer_states = (0..layers)
                .map(|layer| {
                    alloc(
                        &device,
                        DType::F32,
                        vec![4, hc, hidden],
                        &format!("B=4 decode layer {layer} state"),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let combined_attention = alloc(
                &device,
                DType::F32,
                vec![4, hc, hidden],
                "B=4 decode attention",
            )?;
            let mut state = None;
            let mut in_flight =
                SubmissionChain::with_capacity(layers.div_ceil(LAYERS_PER_COMMAND_BUFFER));

            let encode_result = (|| -> Result<()> {
                for start in (0..layers).step_by(LAYERS_PER_COMMAND_BUFFER) {
                    let end = (start + LAYERS_PER_COMMAND_BUFFER).min(layers);
                    let mut session = executor.begin().with_context(|| {
                        format!("begin DeepSeek-V4 B=4 decode layers {start}..{end}")
                    })?;
                    for layer in start..end {
                        let lane_states = if let Some(previous) = state.as_ref() {
                            session.barrier();
                            let lane_states = (0..4)
                                .map(|lane| {
                                    let lane_state = alloc(
                                        &device,
                                        DType::F32,
                                        vec![1, hc, hidden],
                                        "B=4 decode lane state",
                                    )?;
                                    dispatch_copy_f32(
                                        session.encoder_mut(),
                                        &mut self.ctx.registry,
                                        device.metal_device(),
                                        previous,
                                        &lane_state,
                                        lane * row_elements,
                                        0,
                                        row_elements,
                                    )
                                    .with_context(|| {
                                        format!(
                                            "unpack DeepSeek-V4 B=4 layer-{layer} lane-{lane} state"
                                        )
                                    })?;
                                    Ok(lane_state)
                                })
                                .collect::<Result<Vec<_>>>()?;
                            session.barrier();
                            Some(lane_states)
                        } else {
                            None
                        };

                        let mut attentions = Vec::with_capacity(4);
                        for lane in 0..4 {
                            let attention = if layer == 0 {
                                anyhow::ensure!(
                                    lane_states.is_none(),
                                    "DeepSeek-V4 B=4 layer 0 must embed each lane"
                                );
                                self.forward_uncompressed_attention_one(
                                    None,
                                    token_ids[lane],
                                    layer,
                                    caches[lane],
                                    false,
                                    None,
                                    Some(&mut session),
                                )
                            } else if self.cfg.compress_ratios[layer] == 0 {
                                self.forward_uncompressed_attention_one(
                                    Some(&lane_states.as_ref().expect("lane states")[lane]),
                                    token_ids[lane],
                                    layer,
                                    caches[lane],
                                    false,
                                    None,
                                    Some(&mut session),
                                )
                            } else {
                                self.forward_compressed_attention_one(
                                    &lane_states.as_ref().expect("lane states")[lane],
                                    layer,
                                    caches[lane],
                                    false,
                                    None,
                                    Some(&mut session),
                                )
                            }
                            .with_context(|| {
                                format!(
                                    "encode DeepSeek-V4 B=4 layer-{layer} lane-{lane} attention"
                                )
                            })?;
                            attentions.push(attention);
                        }
                        session.barrier();
                        for (lane, attention) in attentions.iter().enumerate() {
                            dispatch_copy_f32(
                                session.encoder_mut(),
                                &mut self.ctx.registry,
                                device.metal_device(),
                                attention,
                                &combined_attention,
                                0,
                                lane * row_elements,
                                row_elements,
                            )
                            .with_context(|| {
                                format!("pack DeepSeek-V4 B=4 layer-{layer} lane-{lane} attention")
                            })?;
                        }
                        session.barrier();
                        state = Some(self.forward_ffn_rows(
                            &combined_attention,
                            token_ids,
                            layer,
                            None,
                            Some(&mut session),
                            Some(layer_states[layer].clone()),
                            None,
                        )?);
                    }
                    submitted_any = true;
                    in_flight.push((
                        format!("execute DeepSeek-V4 B=4 decode layers {start}..{end}"),
                        session.commit(),
                    ));
                }
                Ok(())
            })();
            let drained = drain(&in_flight).context("drain DeepSeek-V4 B=4 decode pipeline");
            drop(in_flight);
            match (encode_result, drained) {
                (Ok(()), Ok(())) => {
                    let state = state.context("DeepSeek-V4 B=4 decode encoded zero layers")?;
                    anyhow::ensure!(
                        state.element_count() == combined_elements,
                        "DeepSeek-V4 B=4 decode output size drift"
                    );
                    Ok(state)
                }
                (Err(error), Ok(())) => Err(error),
                (Ok(()), Err(error)) => Err(error),
                (Err(error), Err(drain_error)) => Err(error).context(format!(
                    "B=4 decode pipeline drain also failed: {drain_error:#}"
                )),
            }
        })();
        (result, submitted_any)
    }
}
