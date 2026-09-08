//! Exact four-lane decode transaction with independent lane arithmetic.

use anyhow::{Context, Result};
use mlx_native::ops::copy::dispatch_copy_f32;
use mlx_native::{DType, GraphExecutor, MlxBuffer};

use super::cache::Deepseek4Cache;
use super::forward_support::{alloc_persistent, begin_decode_pool_token, end_decode_pool_token};
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
    /// Execute four independent one-row attention/FFN graphs in one retained
    /// command-buffer chain, then pack only their final states.
    #[cfg(test)]
    pub(super) fn forward_verifier_decode_cohort(
        &mut self,
        token_ids: [u32; 4],
        caches: &mut [&mut Deepseek4Cache; 4],
    ) -> Result<MlxBuffer> {
        self.forward_verifier_decode_cohort_with_commit_gate(token_ids, caches, || Ok(()))
    }

    pub(crate) fn forward_verifier_decode_cohort_with_commit_gate(
        &mut self,
        token_ids: [u32; 4],
        caches: &mut [&mut Deepseek4Cache; 4],
        commit_gate: impl FnOnce() -> Result<()>,
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
                publish_verifier_cohort_after_gate(caches, positions, commit_gate)?;
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
            anyhow::ensure!(layers > 0, "DeepSeek-V4 B=4 decode encoded zero layers");
            let row_elements = hc
                .checked_mul(hidden)
                .context("DeepSeek-V4 B=4 decode row width overflow")?;
            let combined_elements = 4_usize
                .checked_mul(row_elements)
                .context("DeepSeek-V4 B=4 decode state size overflow")?;
            let device = self.ctx.device().clone();
            let executor = GraphExecutor::new(device.clone());
            let combined_state = alloc_persistent(
                &device,
                DType::F32,
                vec![4, hc, hidden],
                "B=4 decode final state",
            )?;
            let mut lane_states: [Option<MlxBuffer>; 4] = std::array::from_fn(|_| None);
            let mut in_flight =
                SubmissionChain::with_capacity(layers.div_ceil(LAYERS_PER_COMMAND_BUFFER));

            let encode_result = (|| -> Result<()> {
                for start in (0..layers).step_by(LAYERS_PER_COMMAND_BUFFER) {
                    let end = (start + LAYERS_PER_COMMAND_BUFFER).min(layers);
                    let mut session = executor.begin().with_context(|| {
                        format!("begin DeepSeek-V4 B=4 decode layers {start}..{end}")
                    })?;
                    for layer in start..end {
                        let mut next_states = Vec::with_capacity(4);
                        for lane in 0..4 {
                            let attention = if layer == 0 {
                                anyhow::ensure!(
                                    lane_states[lane].is_none(),
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
                                    Some(lane_states[lane].as_ref().expect("lane state")),
                                    token_ids[lane],
                                    layer,
                                    caches[lane],
                                    false,
                                    None,
                                    Some(&mut session),
                                )
                            } else {
                                self.forward_compressed_attention_one(
                                    lane_states[lane].as_ref().expect("lane state"),
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
                            next_states.push(Some(
                                self.forward_ffn_one(
                                    &attention,
                                    token_ids[lane],
                                    layer,
                                    Some(&mut in_flight),
                                    Some(&mut session),
                                )
                                .with_context(|| {
                                    format!("encode DeepSeek-V4 B=4 layer-{layer} lane-{lane} FFN")
                                })?,
                            ));
                        }
                        session.barrier();
                        lane_states = next_states.try_into().map_err(|states: Vec<_>| {
                            anyhow::anyhow!(
                                "DeepSeek-V4 B=4 decode produced {} lane states",
                                states.len()
                            )
                        })?;
                        if layer + 1 == layers {
                            for (lane, state) in lane_states.iter().enumerate() {
                                dispatch_copy_f32(
                                    session.encoder_mut(),
                                    &mut self.ctx.registry,
                                    device.metal_device(),
                                    state.as_ref().expect("final lane state"),
                                    &combined_state,
                                    0,
                                    lane * row_elements,
                                    row_elements,
                                )
                                .with_context(|| {
                                    format!("pack DeepSeek-V4 B=4 final lane-{lane} state")
                                })?;
                            }
                        }
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
                    anyhow::ensure!(
                        combined_state.element_count() == combined_elements,
                        "DeepSeek-V4 B=4 decode output size drift"
                    );
                    Ok(combined_state)
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
