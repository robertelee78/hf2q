//! Private source-BF16 forward-call substrate for the one-shot family runner.
//!
//! A call updates the owned base-text cache and returns one terminally
//! completed full-vocabulary row when requested. This module exposes no raw
//! model or cache; only the consuming run-input worker may mint authority.

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::fused_norm_add::dispatch_fused_residual_norm_f32;
use mlx_native::{DType, KernelRegistry};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::inference::dense_bf16_activation::activate_native_bf16_dense;
use crate::inference::models::qwen35::delta_net::DeltaNetLayerShape;
use crate::inference::models::qwen35::execution_dispatch::{
    source_teacher_graph_policy_sha256, SourceTeacherGraphScope,
};
use crate::inference::models::qwen35::ffn::DenseFfnShape;
use crate::inference::models::qwen35::full_attn::FullAttnShape;
use crate::inference::models::qwen35::gpu_delta_net::build_delta_net_layer;
use crate::inference::models::qwen35::gpu_ffn::build_dense_ffn_layer_gpu;
use crate::inference::models::qwen35::gpu_full_attn::build_gated_attn_layer;
use crate::inference::models::qwen35::kv_cache::{HybridKvCache, PreparedQwen35BaseTextCacheV1};
use crate::serve::multi_seq_kv::SlotId;

use super::runner_io::{gather_bf16_embedding_rows, source_bf16_output_head_last, text_positions};
use super::{PreparedQwen35SourceAttentionV1, PreparedQwen35SourceTeacherV1};

mod state;

use state::{
    cache_device_registry_id, postflight_source_teacher_call, preflight_source_teacher_call,
    preflight_source_teacher_state,
};

pub(super) struct SourceTeacherCallResult {
    pub(super) logits: Vec<f32>,
    pub(super) graph_policy_sha256: String,
}

/// Unforgeable one-shot authorization for moving the prepared cache into this
/// private runner. The constructor and fields never leave this module.
pub(in crate::inference::models::qwen35) struct SourceTeacherCacheAuthorization<'scope> {
    _scope: &'scope SourceTeacherGraphScope,
    _private: (),
}

struct SourceTeacherExecutionCacheV1<'scope> {
    cache: HybridKvCache,
    _scope: &'scope SourceTeacherGraphScope,
}

pub(super) struct SourceTeacherSessionV1<'scope> {
    teacher: PreparedQwen35SourceTeacherV1,
    cache: SourceTeacherExecutionCacheV1<'scope>,
    registry: KernelRegistry,
    next_position: u32,
    poisoned: bool,
    _scope: &'scope SourceTeacherGraphScope,
}

impl<'scope> SourceTeacherSessionV1<'scope> {
    pub(super) fn new(
        scope: &'scope SourceTeacherGraphScope,
        teacher: PreparedQwen35SourceTeacherV1,
        prepared_cache: PreparedQwen35BaseTextCacheV1,
    ) -> Result<Self> {
        let authorization = SourceTeacherCacheAuthorization {
            _scope: scope,
            _private: (),
        };
        let cache = SourceTeacherExecutionCacheV1 {
            cache: prepared_cache.into_source_teacher_cache(authorization),
            _scope: scope,
        };
        preflight_source_teacher_state(&teacher, &cache.cache, 0)?;
        static NEXT_SOURCE_TEACHER_ACTIVATION_EPOCH: AtomicU64 = AtomicU64::new(1);
        let activation_epoch = NEXT_SOURCE_TEACHER_ACTIVATION_EPOCH.fetch_add(1, Ordering::Relaxed);
        ensure!(
            activation_epoch != 0,
            "source teacher activation epoch exhausted"
        );
        let mut registry = KernelRegistry::new();
        let native_bf16 = teacher
            .native_bf16_matrices()
            .context("inventory source-teacher BF16 projections")?;
        ensure!(
            activate_native_bf16_dense(
                &mut registry,
                &teacher.device,
                activation_epoch,
                &native_bf16,
            )
            .context("activate source-teacher BF16 routes")?
            .is_some(),
            "source teacher contains no BF16 projection to activate"
        );
        Ok(Self {
            teacher,
            cache,
            registry,
            next_position: 0,
            poisoned: false,
            _scope: scope,
        })
    }

    pub(super) fn run_call(
        &mut self,
        token_ids: &[u32],
        evaluate_output_head: bool,
    ) -> Result<Option<SourceTeacherCallResult>> {
        ensure!(!self.poisoned, "source teacher session is poisoned");
        let preflight = preflight_source_teacher_call(
            &self.teacher,
            &self.cache.cache,
            token_ids,
            self.next_position,
        )?;
        let graph_policy_sha256 = source_teacher_graph_policy_sha256()?;
        self.poisoned = true;
        let result = run_source_teacher_call_scoped(
            &self.teacher,
            &mut self.cache.cache,
            &mut self.registry,
            token_ids,
            self.next_position,
            evaluate_output_head,
        );
        match result {
            Ok(logits) => {
                // The output-head wait above drained every prior command
                // buffer on the same queue, so pooled scratch can now be
                // recycled safely.
                if let Err(error) =
                    postflight_source_teacher_call(&self.cache.cache, &preflight)
                {
                    crate::inference::models::qwen35::decode_pool::reset_decode_pool();
                    return Err(error);
                }
                crate::inference::models::qwen35::decode_pool::reset_decode_pool();
                self.next_position = preflight.end_position;
                self.poisoned = false;
                Ok(logits.map(|logits| SourceTeacherCallResult {
                    logits,
                    graph_policy_sha256,
                }))
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

    pub(super) fn reset_example(&mut self) -> Result<()> {
        ensure!(!self.poisoned, "source teacher session is poisoned");
        self.cache.cache.reset_for_slot(SlotId(0))?;
        self.next_position = 0;
        preflight_source_teacher_state(&self.teacher, &self.cache.cache, 0)?;
        Ok(())
    }

    pub(super) fn terminal_drain_after_panic(&mut self) -> Result<()> {
        self.poisoned = true;
        terminal_drain(&self.teacher)?;
        crate::inference::models::qwen35::decode_pool::reset_decode_pool();
        Ok(())
    }

    pub(super) fn terminal_drain_for_completion(&mut self) -> Result<()> {
        ensure!(!self.poisoned, "source teacher session is poisoned");
        terminal_drain(&self.teacher)?;
        crate::inference::models::qwen35::decode_pool::reset_decode_pool();
        Ok(())
    }

    pub(super) fn device_name(&self) -> String {
        self.teacher.device.name()
    }

    pub(super) fn device_registry_id(&self) -> u64 {
        self.teacher.device.registry_id()
    }

    pub(super) fn next_position(&self) -> u32 {
        self.next_position
    }

    pub(super) fn finish_source_lineage(
        self,
    ) -> Result<
        crate::inference::models::qwen35::source_precision::snapshot::VerifiedQwenSourceSnapshot,
    > {
        self.teacher.snapshot.rehash_retained_files()?;
        let super::PreparedQwen35SourceTeacherV1 { snapshot, .. } = self.teacher;
        Ok(snapshot)
    }
}

fn run_source_teacher_call_scoped(
    teacher: &PreparedQwen35SourceTeacherV1,
    cache: &mut HybridKvCache,
    registry: &mut KernelRegistry,
    token_ids: &[u32],
    first_position: u32,
    evaluate_output_head: bool,
) -> Result<Option<Vec<f32>>> {
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
            if crate::inference::models::qwen35::execution_dispatch::source_teacher_scope_active() {
                boundary
                    .commit_and_wait_labeled("source_teacher.layer.boundary")
                    .with_context(|| {
                        format!("complete source teacher layer boundary {layer_index}")
                    })?;
            } else {
                boundary.commit();
            }
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
    if evaluate_output_head {
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
        .map(Some)
    } else {
        terminal_drain(teacher)?;
        Ok(None)
    }
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

#[cfg(test)]
pub(super) use tests::{cpu_model, h256_fixture, last_cpu_logits};
