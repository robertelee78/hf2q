//! Checked preparation limits and lower-bound runtime accounting.

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use crate::inference::models::qwen35::kv_cache::plan_qwen35_base_text_cache;
use crate::inference::models::qwen35::source_precision::upload_plan::{
    QwenSourceMetalCapacityV1, QwenSourceMetalUploadLimits,
};
use crate::inference::models::qwen35::{Qwen35Config, Qwen35LayerKind};

const HARD_MAX_SEQUENCE_TOKENS: u32 = 4_096;
const HARD_MAX_TARGET_ROWS: u32 = 16_384;
const HARD_MAX_TARGET_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const HARD_MAX_CPU_CONTROL_MIRROR_BYTES: u64 = 256 * 1024 * 1024;
const UPLOAD_SCRATCH_BYTES: u64 = 4 * 1024 * 1024;

/// Caller-selected work and reserve envelope for the later completed runner.
/// The reserve is an allowance for builder scratch and allocator bookkeeping,
/// not an observed peak or a reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct Qwen35SourceTeacherLimitsV1 {
    pub max_sequence_tokens: u32,
    pub max_target_rows: u32,
    pub max_cpu_control_mirror_bytes: u64,
    pub unmeasured_runtime_reserve_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Qwen35SourceTeacherRuntimeEnvelopeV1 {
    pub(super) max_sequence_tokens: u32,
    pub(super) max_target_rows: u32,
    pub(super) base_full_attention_cache_bytes: u64,
    pub(super) base_linear_attention_state_bytes: u64,
    pub(super) max_input_activation_bytes: u64,
    pub(super) one_logit_row_bytes: u64,
    pub(super) target_payload_upper_bound_bytes: u64,
    pub(super) cpu_control_mirror_bytes: u64,
    pub(super) accounted_runtime_payload_bytes: u64,
    pub(super) unmeasured_runtime_reserve_bytes: u64,
}

pub(super) fn runtime_envelope(
    config: &Qwen35Config,
    limits: Qwen35SourceTeacherLimitsV1,
) -> Result<Qwen35SourceTeacherRuntimeEnvelopeV1> {
    ensure!(
        limits.max_sequence_tokens > 0
            && limits.max_sequence_tokens <= HARD_MAX_SEQUENCE_TOKENS
            && limits.max_sequence_tokens <= config.max_position_embeddings,
        "source teacher sequence bound exceeds the v1/config limit"
    );
    ensure!(
        limits.max_target_rows > 0 && limits.max_target_rows <= HARD_MAX_TARGET_ROWS,
        "source teacher target-row bound exceeds the v1 limit"
    );
    ensure!(
        limits.max_cpu_control_mirror_bytes <= HARD_MAX_CPU_CONTROL_MIRROR_BYTES,
        "source teacher CPU-control mirror bound exceeds the v1 limit"
    );

    let cache_plan = plan_qwen35_base_text_cache(config, limits.max_sequence_tokens)?;
    let base_full_attention_cache_bytes = cache_plan.base_full_attention_cache_bytes();
    let base_linear_attention_state_bytes = cache_plan.base_linear_attention_state_bytes();
    let linear_layers = config
        .layer_types
        .iter()
        .filter(|kind| **kind == Qwen35LayerKind::LinearAttention)
        .count() as u64;
    let max_input_activation_bytes = checked_product(&[
        limits.max_sequence_tokens as u64,
        config.hidden_size as u64,
        4,
    ])?;
    let one_logit_row_bytes = checked_product(&[config.vocab_size as u64, 4])?;
    let target_payload_upper_bound_bytes = one_logit_row_bytes
        .checked_mul(limits.max_target_rows as u64)
        .context("source teacher target payload bytes overflow")?;
    ensure!(
        target_payload_upper_bound_bytes <= HARD_MAX_TARGET_BYTES,
        "source teacher target payload exceeds the v1 limit"
    );
    let cpu_control_mirror_bytes = checked_product(&[
        linear_layers,
        (2_u64)
            .checked_mul(config.linear_num_value_heads as u64)
            .and_then(|value| value.checked_add(config.linear_value_head_dim as u64))
            .context("source teacher CPU-control mirror elements overflow")?,
        4,
    ])?;
    ensure!(
        cpu_control_mirror_bytes <= limits.max_cpu_control_mirror_bytes,
        "source teacher CPU-control mirrors exceed the configured bound"
    );
    let accounted_runtime_payload_bytes = [
        base_full_attention_cache_bytes,
        base_linear_attention_state_bytes,
        max_input_activation_bytes,
        one_logit_row_bytes,
        cpu_control_mirror_bytes,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| total.checked_add(bytes))
    .context("source teacher accounted runtime bytes overflow")?;
    Ok(Qwen35SourceTeacherRuntimeEnvelopeV1 {
        max_sequence_tokens: limits.max_sequence_tokens,
        max_target_rows: limits.max_target_rows,
        base_full_attention_cache_bytes,
        base_linear_attention_state_bytes,
        max_input_activation_bytes,
        one_logit_row_bytes,
        target_payload_upper_bound_bytes,
        cpu_control_mirror_bytes,
        accounted_runtime_payload_bytes,
        unmeasured_runtime_reserve_bytes: limits.unmeasured_runtime_reserve_bytes,
    })
}

pub(super) fn validate_combined_capacity(
    planned_weight_bytes: u64,
    runtime: &Qwen35SourceTeacherRuntimeEnvelopeV1,
    upload_limits: QwenSourceMetalUploadLimits,
    capacity: QwenSourceMetalCapacityV1,
) -> Result<()> {
    let accounted = planned_weight_bytes
        .checked_add(UPLOAD_SCRATCH_BYTES)
        .and_then(|value| value.checked_add(runtime.accounted_runtime_payload_bytes))
        .and_then(|value| value.checked_add(runtime.unmeasured_runtime_reserve_bytes))
        .context("source teacher combined capacity requirement overflow")?;
    let host_required = accounted
        .checked_add(upload_limits.host_reserve_bytes)
        .context("source teacher combined host capacity requirement overflow")?;
    let metal_required = accounted
        .checked_add(upload_limits.metal_reserve_bytes)
        .context("source teacher combined Metal capacity requirement overflow")?;
    let metal_available = capacity
        .metal_recommended_working_set_bytes
        .checked_sub(capacity.metal_current_allocated_bytes)
        .context("source teacher Metal working-set observation is already exhausted")?;
    ensure!(
        host_required <= capacity.host_available_bytes && metal_required <= metal_available,
        "source teacher combined weight/runtime requirement exceeds observed capacity"
    );
    Ok(())
}

pub(super) fn validate_incremental_capacity(
    runtime: &Qwen35SourceTeacherRuntimeEnvelopeV1,
    upload_limits: QwenSourceMetalUploadLimits,
    capacity: QwenSourceMetalCapacityV1,
) -> Result<()> {
    let accounted = runtime
        .accounted_runtime_payload_bytes
        .checked_add(runtime.unmeasured_runtime_reserve_bytes)
        .context("source teacher incremental capacity requirement overflow")?;
    let host_required = accounted
        .checked_add(upload_limits.host_reserve_bytes)
        .context("source teacher incremental host capacity requirement overflow")?;
    let metal_required = accounted
        .checked_add(upload_limits.metal_reserve_bytes)
        .context("source teacher incremental Metal capacity requirement overflow")?;
    let metal_available = capacity
        .metal_recommended_working_set_bytes
        .checked_sub(capacity.metal_current_allocated_bytes)
        .context("source teacher Metal working-set observation is already exhausted")?;
    ensure!(
        host_required <= capacity.host_available_bytes && metal_required <= metal_available,
        "source teacher incremental runtime requirement exceeds observed capacity"
    );
    Ok(())
}

fn checked_product(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(1_u64, |product, value| product.checked_mul(*value))
        .context("source teacher byte calculation overflow")
}
