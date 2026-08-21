//! Test-only, non-authoritative source-BF16 forward-call substrate.
//!
//! A call updates the owned base-text cache and returns one terminally
//! completed full-vocabulary row. It does not traverse a prediction plan,
//! write/publish a target, or mint teacher authority.

use anyhow::{anyhow, bail, ensure, Context, Result};
use mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32;
use mlx_native::{DType, KernelRegistry};

use crate::inference::models::qwen35::delta_net::DeltaNetLayerShape;
use crate::inference::models::qwen35::execution_dispatch::{
    source_teacher_graph_policy_sha256, with_source_teacher_graph_scope,
};
use crate::inference::models::qwen35::ffn::DenseFfnShape;
use crate::inference::models::qwen35::full_attn::FullAttnShape;
use crate::inference::models::qwen35::gpu_delta_net::build_delta_net_layer;
use crate::inference::models::qwen35::gpu_ffn::build_dense_ffn_layer_gpu;
use crate::inference::models::qwen35::gpu_full_attn::build_gated_attn_layer;
use crate::inference::models::qwen35::kv_cache::{
    HybridKvCache, LayerSlot, PreparedQwen35BaseTextCacheV1,
};
use crate::inference::models::qwen35::Qwen35LayerKind;
use crate::serve::multi_seq_kv::SlotId;

use super::runner_io::{gather_bf16_embedding_rows, source_bf16_output_head_last, text_positions};
use super::{PreparedQwen35SourceAttentionV1, PreparedQwen35SourceTeacherV1};

pub(super) struct SourceTeacherCallResult {
    pub(super) logits: Vec<f32>,
    pub(super) graph_policy_sha256: String,
}

pub(super) struct PrivateSourceTeacherParitySessionV1 {
    teacher: PreparedQwen35SourceTeacherV1,
    cache: HybridKvCache,
    registry: KernelRegistry,
    next_position: u32,
    poisoned: bool,
}

struct CallPreflight {
    end_position: u32,
    linear_parity: bool,
}

impl PrivateSourceTeacherParitySessionV1 {
    pub(super) fn new(
        teacher: PreparedQwen35SourceTeacherV1,
        prepared_cache: PreparedQwen35BaseTextCacheV1,
    ) -> Result<Self> {
        let cache = prepared_cache.into_cache();
        preflight_source_teacher_state(&teacher, &cache, 0)?;
        Ok(Self {
            teacher,
            cache,
            registry: KernelRegistry::new(),
            next_position: 0,
            poisoned: false,
        })
    }

    pub(super) fn run_call(&mut self, token_ids: &[u32]) -> Result<SourceTeacherCallResult> {
        ensure!(!self.poisoned, "source teacher parity session is poisoned");
        let preflight = preflight_source_teacher_call(
            &self.teacher,
            &self.cache,
            token_ids,
            self.next_position,
        )?;
        let graph_policy_sha256 = source_teacher_graph_policy_sha256()?;
        self.poisoned = true;
        let result = with_source_teacher_graph_scope(|| {
            run_source_teacher_call_scoped(
                &self.teacher,
                &mut self.cache,
                &mut self.registry,
                token_ids,
                self.next_position,
            )
        });
        match result {
            Ok(logits) => {
                // The output-head wait above drained every prior command
                // buffer on the same queue, so pooled scratch can now be
                // recycled safely.
                if let Err(error) = postflight_source_teacher_call(&self.cache, &preflight) {
                    crate::inference::models::qwen35::decode_pool::reset_decode_pool();
                    return Err(error);
                }
                crate::inference::models::qwen35::decode_pool::reset_decode_pool();
                self.next_position = preflight.end_position;
                self.poisoned = false;
                Ok(SourceTeacherCallResult {
                    logits,
                    graph_policy_sha256,
                })
            }
            Err(error) => match terminal_drain(&self.teacher) {
                Ok(()) => {
                    crate::inference::models::qwen35::decode_pool::reset_decode_pool();
                    Err(error)
                }
                Err(drain_error) => Err(anyhow!(
                    "source teacher call failed: {error:#}; terminal drain also failed: {drain_error:#}"
                )),
            },
        }
    }
}

fn preflight_source_teacher_state(
    teacher: &PreparedQwen35SourceTeacherV1,
    cache: &HybridKvCache,
    expected_position: u32,
) -> Result<bool> {
    let config = &teacher.config;
    ensure!(
        teacher.layers.len() == config.layer_types.len()
            && cache
                .slot_index_for_layer(u32::try_from(config.layer_types.len())?)
                .is_none()
            && cache.n_seqs == 1
            && cache.mtp_slot.is_none()
            && !cache.tq_kv_active
            && teacher.device.registry_id() == cache_device_registry_id(cache)?,
        "source teacher graph/cache profile differs from prepared config"
    );
    let mut full_rank = 0usize;
    let mut linear_rank = 0usize;
    let mut linear_parity = None;
    for (layer_index, kind) in config.layer_types.iter().copied().enumerate() {
        match (kind, &teacher.layers[layer_index].attention) {
            (Qwen35LayerKind::FullAttention, PreparedQwen35SourceAttentionV1::Full(_)) => {
                ensure!(
                    cache.slot_index_for_layer(u32::try_from(layer_index)?)
                        == Some(LayerSlot::Full(full_rank as u32)),
                    "source teacher full-attention cache rank differs at layer {layer_index}"
                );
                let slot = &cache.full_attn[full_rank];
                ensure!(
                    slot.tq.is_none()
                        && slot.k.is_some()
                        && slot.v.is_some()
                        && slot.current_len.as_slice() == [expected_position],
                    "source teacher full-attention cursor/profile differs at layer {layer_index}"
                );
                full_rank += 1;
            }
            (Qwen35LayerKind::LinearAttention, PreparedQwen35SourceAttentionV1::Linear(_)) => {
                ensure!(
                    cache.slot_index_for_layer(u32::try_from(layer_index)?)
                        == Some(LayerSlot::Linear(linear_rank as u32)),
                    "source teacher Delta cache rank differs at layer {layer_index}"
                );
                let slot = &cache.linear_attn[linear_rank];
                ensure!(
                    slot.capture_states.is_none()
                        && slot.conv_capture_states.is_none()
                        && slot.pp_flipped.len() == 1,
                    "source teacher Delta cache profile differs at layer {layer_index}"
                );
                let parity = slot.pp_flipped[0];
                ensure!(
                    linear_parity.is_none_or(|expected| expected == parity),
                    "source teacher Delta cache parity is desynchronized"
                );
                linear_parity = Some(parity);
                linear_rank += 1;
            }
            _ => bail!("source teacher layer schedule differs at layer {layer_index}"),
        }
    }
    ensure!(
        full_rank == cache.full_attn.len() && linear_rank == cache.linear_attn.len(),
        "source teacher cache cardinality differs from layer schedule"
    );
    linear_parity.context("source teacher cache lacks Delta state")
}

fn preflight_source_teacher_call(
    teacher: &PreparedQwen35SourceTeacherV1,
    cache: &HybridKvCache,
    token_ids: &[u32],
    first_position: u32,
) -> Result<CallPreflight> {
    let linear_parity = preflight_source_teacher_state(teacher, cache, first_position)?;
    let sequence_length = u32::try_from(token_ids.len())?;
    ensure!(
        if first_position == 0 {
            sequence_length >= 16
        } else {
            sequence_length == 1
        },
        "source teacher v1 admits one >=16-token fresh prefill followed by one-token continuations"
    );
    ensure!(
        teacher.config.vocab_size > 0
            && teacher.config.vocab_size % 2 == 0
            && token_ids
                .iter()
                .all(|token_id| *token_id < teacher.config.vocab_size),
        "source teacher tokens/output vocabulary differ from the v1 output route"
    );
    let end_position = first_position
        .checked_add(sequence_length)
        .context("source teacher call position overflow")?;
    ensure!(
        end_position <= cache.max_seq_len,
        "source teacher call exceeds cache capacity"
    );
    Ok(CallPreflight {
        end_position,
        linear_parity,
    })
}

fn postflight_source_teacher_call(cache: &HybridKvCache, preflight: &CallPreflight) -> Result<()> {
    ensure!(
        cache
            .full_attn
            .iter()
            .all(|slot| slot.current_len.as_slice() == [preflight.end_position])
            && cache.linear_attn.iter().all(|slot| {
                slot.pp_flipped.len() == 1 && slot.pp_flipped[0] != preflight.linear_parity
            }),
        "source teacher cache did not advance exactly once"
    );
    Ok(())
}

fn run_source_teacher_call_scoped(
    teacher: &PreparedQwen35SourceTeacherV1,
    cache: &mut HybridKvCache,
    registry: &mut KernelRegistry,
    token_ids: &[u32],
    first_position: u32,
) -> Result<Vec<f32>> {
    let config = &teacher.config;
    ensure!(
        teacher.device.registry_id() == cache_device_registry_id(cache)?
            && cache.n_seqs == 1
            && cache.mtp_slot.is_none()
            && !cache.tq_kv_active,
        "source teacher call cache differs from prepared device/profile"
    );
    let sequence_length = u32::try_from(token_ids.len())?;
    ensure!(
        sequence_length > 0
            && first_position
                .checked_add(sequence_length)
                .is_some_and(|end| end <= cache.max_seq_len),
        "source teacher call exceeds cache capacity"
    );
    let positions = text_positions(&teacher.device, first_position, sequence_length)?;
    let mut hidden = gather_bf16_embedding_rows(
        &teacher.device,
        &teacher.embedding,
        token_ids,
        config.vocab_size,
        config.hidden_size,
    )?;
    let slot_id = SlotId(0);
    let mut full_rank = 0usize;
    let mut linear_rank = 0usize;
    for (layer_index, layer) in teacher.layers.iter().enumerate() {
        let next_hidden = {
            let attention_output = match &layer.attention {
                PreparedQwen35SourceAttentionV1::Full(weights) => {
                    let shape = FullAttnShape::from_config(config);
                    let slot = cache.full_attn.get_mut(full_rank).with_context(|| {
                        format!("source teacher lacks full cache layer {layer_index}")
                    })?;
                    full_rank += 1;
                    build_gated_attn_layer(
                        &teacher.device,
                        registry,
                        &hidden,
                        &positions,
                        weights,
                        Some(slot),
                        cache.max_seq_len,
                        sequence_length,
                        shape.hidden_size,
                        shape.n_head,
                        shape.n_kv,
                        shape.head_dim,
                        shape.rotary_dim,
                        shape.rope_theta,
                        shape.mrope_section,
                        shape.rms_norm_eps,
                        None,
                        None,
                        None,
                        None,
                        slot_id,
                    )
                    .with_context(|| format!("source teacher full attention layer {layer_index}"))?
                }
                PreparedQwen35SourceAttentionV1::Linear(weights) => {
                    let shape = DeltaNetLayerShape::from_config(config);
                    let slot = cache.linear_attn.get_mut(linear_rank).with_context(|| {
                        format!("source teacher lacks Delta cache layer {layer_index}")
                    })?;
                    linear_rank += 1;
                    let (conv_in, conv_out) = slot.conv_bufs_for_slot(slot_id);
                    let (state_in, state_out) = slot.recurrent_bufs_for_slot(slot_id);
                    let output = build_delta_net_layer(
                        &teacher.device,
                        registry,
                        &hidden,
                        weights,
                        conv_in,
                        conv_out,
                        state_in,
                        state_out,
                        sequence_length,
                        shape.hidden_size,
                        shape.n_k_heads,
                        shape.n_v_heads,
                        shape.d_k,
                        shape.d_v,
                        shape.conv_kernel,
                        shape.rms_norm_eps,
                        None,
                        None,
                        slot_id,
                    )
                    .with_context(|| format!("source teacher Delta layer {layer_index}"))?;
                    slot.swap_for_slot(slot_id);
                    output
                }
            };

            let element_count = usize::try_from(sequence_length)?
                .checked_mul(usize::try_from(config.hidden_size)?)
                .context("source teacher layer boundary size overflow")?;
            let ffn_input = teacher
                .device
                .alloc_buffer(
                    element_count
                        .checked_mul(4)
                        .context("source teacher FFN input bytes overflow")?,
                    DType::F32,
                    vec![sequence_length as usize, config.hidden_size as usize],
                )
                .context("allocate source teacher FFN input")?;
            let ffn_residual = teacher
                .device
                .alloc_buffer(
                    element_count
                        .checked_mul(4)
                        .context("source teacher FFN residual bytes overflow")?,
                    DType::F32,
                    vec![sequence_length as usize, config.hidden_size as usize],
                )
                .context("allocate source teacher FFN residual")?;
            let mut boundary = teacher
                .device
                .command_encoder()
                .context("create source teacher layer-boundary encoder")?;
            let post_attention_norm = match &layer.attention {
                PreparedQwen35SourceAttentionV1::Full(weights) => &weights.post_attn_norm,
                PreparedQwen35SourceAttentionV1::Linear(weights) => &weights.post_attn_norm,
            };
            dispatch_fused_residual_norm_f32(
                &mut boundary,
                registry,
                teacher.device.metal_device(),
                &hidden,
                &attention_output,
                post_attention_norm,
                &ffn_input,
                Some(&ffn_residual),
                sequence_length,
                config.hidden_size,
                config.rms_norm_eps,
            )
            .with_context(|| format!("source teacher layer boundary {layer_index}"))?;
            boundary.commit();
            let shape = DenseFfnShape {
                hidden_size: config.hidden_size,
                intermediate_size: config
                    .intermediate_size
                    .context("source teacher lacks dense FFN size")?,
            };
            build_dense_ffn_layer_gpu(
                &teacher.device,
                registry,
                &ffn_input,
                &layer.ffn,
                shape,
                Some(&ffn_residual),
            )
            .with_context(|| format!("source teacher dense FFN layer {layer_index}"))?
        };
        hidden = next_hidden;
        if sequence_length > 1 {
            // The dense FFN prefill path terminally waited. All pooled layer
            // scratch is dead here; retain only the device-owned residual.
            crate::inference::models::qwen35::decode_pool::reset_for_prefill_chunk();
        }
    }
    ensure!(
        full_rank == cache.full_attn.len() && linear_rank == cache.linear_attn.len(),
        "source teacher layer schedule differs from base cache"
    );
    source_bf16_output_head_last(
        &teacher.device,
        registry,
        &hidden,
        &teacher.output_norm,
        &teacher.output,
        sequence_length,
        config.hidden_size,
        config.vocab_size,
        config.rms_norm_eps,
    )
}

fn cache_device_registry_id(cache: &HybridKvCache) -> Result<u64> {
    if let Some(slot) = cache.full_attn.first() {
        return Ok(slot
            .k
            .as_ref()
            .context("source teacher full cache K is absent")?
            .metal_buffer()
            .device()
            .registry_id());
    }
    let slot = cache
        .linear_attn
        .first()
        .context("source teacher cache has no layer state")?;
    Ok(slot.conv_state.metal_buffer().device().registry_id())
}

fn terminal_drain(teacher: &PreparedQwen35SourceTeacherV1) -> Result<()> {
    let mut encoder = teacher
        .device
        .command_encoder()
        .context("create source teacher terminal-drain encoder")?;
    encoder
        .commit_and_wait_labeled("qwen35.source_teacher.failure_drain")
        .context("drain source teacher queue after failure")
}

#[cfg(test)]
mod tests;
