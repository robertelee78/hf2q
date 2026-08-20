//! Checked, base-text-only cache preparation for the source-BF16 teacher.
//!
//! This module deliberately stops before graph execution. It plans and owns
//! one fresh F32 cache with one sequence, no MTP slot, no TQ buffers, and no
//! speculative DeltaNet capture. No raw cache or Metal buffer escapes.

use anyhow::{ensure, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{HybridKvCache, LayerSlot};
use crate::inference::models::qwen35::{Qwen35Config, Qwen35LayerKind, Qwen35Variant};
use crate::serve::multi_seq_kv::SlotId;

#[cfg(test)]
#[path = "source_teacher_tests.rs"]
mod tests;

const CACHE_SCHEMA_VERSION: u32 = 1;
const CACHE_PROFILE: &str = "dense_qwen35_source_teacher_base_text_f32_cache_v1";
const HARD_MAX_CACHE_LAYERS: u32 = 256;
const HARD_MAX_CACHE_SEQUENCE_TOKENS: u32 = 4_096;
const HARD_MAX_CACHE_PAYLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct Qwen35BaseTextCacheBufferPlanV1 {
    role: String,
    shape: Vec<usize>,
    dtype: &'static str,
    byte_len: u64,
}

#[derive(Serialize)]
struct CachePlanHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    max_sequence_tokens: u32,
    n_sequences: u32,
    full_attention_slots: usize,
    linear_attention_slots: usize,
    buffer_records: &'a [Qwen35BaseTextCacheBufferPlanV1],
    base_full_attention_cache_bytes: u64,
    base_linear_attention_state_bytes: u64,
    total_payload_bytes: u64,
    mtp_slot_allocated: bool,
    tq_kv_active: bool,
    linear_capture_allocated: bool,
}

/// Stable checked layout shared by B3a preflight and actual cache creation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::inference::models::qwen35) struct Qwen35BaseTextCachePlanV1 {
    schema_version: u32,
    profile: &'static str,
    max_sequence_tokens: u32,
    n_sequences: u32,
    full_attention_slots: usize,
    linear_attention_slots: usize,
    buffer_records: Vec<Qwen35BaseTextCacheBufferPlanV1>,
    base_full_attention_cache_bytes: u64,
    base_linear_attention_state_bytes: u64,
    total_payload_bytes: u64,
    mtp_slot_allocated: bool,
    tq_kv_active: bool,
    linear_capture_allocated: bool,
    layout_sha256: String,
}

impl Qwen35BaseTextCachePlanV1 {
    pub(in crate::inference::models::qwen35) fn base_full_attention_cache_bytes(&self) -> u64 {
        self.base_full_attention_cache_bytes
    }

    pub(in crate::inference::models::qwen35) fn base_linear_attention_state_bytes(&self) -> u64 {
        self.base_linear_attention_state_bytes
    }
}

#[derive(Serialize)]
struct CacheReceiptHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    layout_sha256: &'a str,
    device_name: &'a str,
    device_registry_id: u64,
    actual_payload_bytes: u64,
    fresh_semantic_state: bool,
}

/// Process-local proof of a fresh host-visible Metal cache allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(in crate::inference::models::qwen35) struct Qwen35BaseTextCacheReceiptV1 {
    schema_version: u32,
    profile: &'static str,
    plan: Qwen35BaseTextCachePlanV1,
    device_name: String,
    device_registry_id: u64,
    actual_payload_bytes: u64,
    fresh_semantic_state: bool,
    receipt_sha256: String,
}

/// Opaque owner of the fresh cache. A later runner slice will add the only
/// consuming execution transition; this slice exposes no cache or buffer.
pub(in crate::inference::models::qwen35) struct PreparedQwen35BaseTextCacheV1 {
    cache: HybridKvCache,
    receipt: Qwen35BaseTextCacheReceiptV1,
}

impl PreparedQwen35BaseTextCacheV1 {
    #[allow(dead_code)] // consumed by the source-teacher runner slice
    pub(in crate::inference::models::qwen35) fn receipt(&self) -> &Qwen35BaseTextCacheReceiptV1 {
        &self.receipt
    }
}

/// Compute the exact F32 cache payload without allocating model or Metal
/// state. The authenticated config remains immutable and its declared MTP is
/// intentionally absent from this base-text profile.
pub(in crate::inference::models::qwen35) fn plan_qwen35_base_text_cache(
    config: &Qwen35Config,
    max_sequence_tokens: u32,
) -> Result<Qwen35BaseTextCachePlanV1> {
    ensure!(
        max_sequence_tokens > 0
            && max_sequence_tokens <= HARD_MAX_CACHE_SEQUENCE_TOKENS
            && max_sequence_tokens <= config.max_position_embeddings,
        "source teacher cache sequence bound is outside the authenticated config"
    );
    ensure!(
        config.variant == Qwen35Variant::Dense
            && config.intermediate_size.is_some()
            && config.moe.is_none(),
        "source teacher cache profile requires dense Qwen"
    );
    ensure!(
        config.num_hidden_layers > 0
            && config.num_hidden_layers <= HARD_MAX_CACHE_LAYERS
            && usize::try_from(config.num_hidden_layers)? == config.layer_types.len(),
        "source teacher cache layer schedule length differs from config"
    );
    ensure!(
        config.num_key_value_heads > 0
            && config.head_dim > 0
            && config.linear_num_key_heads > 0
            && config.linear_num_value_heads > 0
            && config.linear_key_head_dim > 0
            && config.linear_value_head_dim > 0
            && config.linear_conv_kernel_dim > 0
            && config.linear_num_value_heads >= config.linear_num_key_heads
            && config.linear_num_value_heads % config.linear_num_key_heads == 0,
        "source teacher cache dimensions must be nonzero"
    );

    let conv_channels = 2_u64
        .checked_mul(u64::from(config.linear_num_key_heads))
        .and_then(|value| value.checked_mul(u64::from(config.linear_key_head_dim)))
        .and_then(|value| {
            value.checked_add(
                u64::from(config.linear_num_value_heads)
                    .checked_mul(u64::from(config.linear_value_head_dim))?,
            )
        })
        .context("source teacher cache Delta channels overflow")?;
    let conv_channels = usize::try_from(conv_channels)
        .context("source teacher cache Delta channels exceed usize")?;
    let conv_history = usize::try_from(config.linear_conv_kernel_dim.saturating_sub(1).max(1))?;

    let mut full_rank = 0usize;
    let mut linear_rank = 0usize;
    let mut records = Vec::new();
    let mut full_bytes = 0_u64;
    let mut linear_bytes = 0_u64;
    for (layer_index, kind) in config.layer_types.iter().copied().enumerate() {
        match kind {
            Qwen35LayerKind::FullAttention => {
                let shape = vec![
                    1,
                    usize::try_from(config.num_key_value_heads)?,
                    usize::try_from(max_sequence_tokens)?,
                    usize::try_from(config.head_dim)?,
                ];
                for suffix in ["k", "v"] {
                    let record = planned_buffer(
                        format!("blk.{layer_index}.base_text_cache.{suffix}"),
                        shape.clone(),
                    )?;
                    full_bytes = full_bytes
                        .checked_add(record.byte_len)
                        .context("source teacher full-attention cache bytes overflow")?;
                    records.push(record);
                }
                full_rank += 1;
            }
            Qwen35LayerKind::LinearAttention => {
                let conv_shape = vec![conv_channels, conv_history, 1];
                let recurrent_shape = vec![
                    usize::try_from(config.linear_key_head_dim)?,
                    usize::try_from(config.linear_value_head_dim)?,
                    usize::try_from(config.linear_num_value_heads)?,
                    1,
                ];
                for (suffix, shape) in [
                    ("conv_state", conv_shape.clone()),
                    ("conv_state_scratch", conv_shape),
                    ("recurrent", recurrent_shape.clone()),
                    ("recurrent_scratch", recurrent_shape),
                ] {
                    let record = planned_buffer(
                        format!("blk.{layer_index}.base_text_cache.{suffix}"),
                        shape,
                    )?;
                    linear_bytes = linear_bytes
                        .checked_add(record.byte_len)
                        .context("source teacher linear-attention state bytes overflow")?;
                    records.push(record);
                }
                linear_rank += 1;
            }
        }
    }
    ensure!(
        records.len()
            == full_rank
                .checked_mul(2)
                .and_then(|value| value.checked_add(linear_rank.checked_mul(4)?))
                .context("source teacher cache record cardinality overflow")?,
        "source teacher cache record cardinality differs from its schedule"
    );
    ensure!(
        full_rank > 0 && linear_rank > 0,
        "source teacher cache profile requires both Qwen layer kinds"
    );
    let total_payload_bytes = full_bytes
        .checked_add(linear_bytes)
        .context("source teacher cache total bytes overflow")?;
    ensure!(
        total_payload_bytes <= HARD_MAX_CACHE_PAYLOAD_BYTES,
        "source teacher cache payload exceeds the v1 hard limit"
    );
    let mut plan = Qwen35BaseTextCachePlanV1 {
        schema_version: CACHE_SCHEMA_VERSION,
        profile: CACHE_PROFILE,
        max_sequence_tokens,
        n_sequences: 1,
        full_attention_slots: full_rank,
        linear_attention_slots: linear_rank,
        buffer_records: records,
        base_full_attention_cache_bytes: full_bytes,
        base_linear_attention_state_bytes: linear_bytes,
        total_payload_bytes,
        mtp_slot_allocated: false,
        tq_kv_active: false,
        linear_capture_allocated: false,
        layout_sha256: String::new(),
    };
    plan.layout_sha256 = plan_sha256(&plan)?;
    Ok(plan)
}

/// Allocate and validate the exact base-text cache. Failure drops all partial
/// Metal ownership and returns no prepared type.
#[allow(dead_code)] // consumed by the source-teacher runner slice
pub(in crate::inference::models::qwen35) fn prepare_qwen35_base_text_cache(
    config: &Qwen35Config,
    device: &MlxDevice,
    max_sequence_tokens: u32,
) -> Result<PreparedQwen35BaseTextCacheV1> {
    let plan = plan_qwen35_base_text_cache(config, max_sequence_tokens)?;
    ensure!(
        plan.buffer_records
            .iter()
            .all(|record| record.byte_len <= device.metal_device().max_buffer_length() as u64),
        "source teacher cache buffer exceeds the Metal device limit"
    );
    let mut cache =
        HybridKvCache::allocate_with_profile(config, device, max_sequence_tokens, 1, false, false)?;
    cache
        .reset_for_slot(SlotId(0))
        .context("initialize source teacher base-text cache")?;
    let mut receipt = Qwen35BaseTextCacheReceiptV1 {
        schema_version: CACHE_SCHEMA_VERSION,
        profile: CACHE_PROFILE,
        plan,
        device_name: device.name(),
        device_registry_id: device.registry_id(),
        actual_payload_bytes: 0,
        fresh_semantic_state: true,
        receipt_sha256: String::new(),
    };
    receipt.actual_payload_bytes = receipt.plan.total_payload_bytes;
    receipt.receipt_sha256 = receipt_sha256(&receipt)?;
    validate_fresh_cache(&cache, config, device, &receipt)?;
    Ok(PreparedQwen35BaseTextCacheV1 { cache, receipt })
}

fn validate_fresh_cache(
    cache: &HybridKvCache,
    config: &Qwen35Config,
    device: &MlxDevice,
    receipt: &Qwen35BaseTextCacheReceiptV1,
) -> Result<()> {
    ensure!(
        receipt.schema_version == CACHE_SCHEMA_VERSION
            && receipt.profile == CACHE_PROFILE
            && receipt.plan == plan_qwen35_base_text_cache(config, cache.max_seq_len)?
            && receipt.device_name == device.name()
            && receipt.device_registry_id == device.registry_id()
            && receipt.actual_payload_bytes == receipt.plan.total_payload_bytes
            && receipt.fresh_semantic_state
            && receipt.receipt_sha256 == receipt_sha256(receipt)?,
        "source teacher base-text cache receipt does not reproduce"
    );
    ensure!(
        cache.max_seq_len == receipt.plan.max_sequence_tokens
            && cache.n_seqs == 1
            && !cache.tq_kv_active
            && cache.mtp_slot.is_none()
            && cache.la_capture_active_tokens.is_none()
            && cache.full_attn.len() == receipt.plan.full_attention_slots
            && cache.linear_attn.len() == receipt.plan.linear_attention_slots
            && cache.per_layer_slot.len() == config.layer_types.len(),
        "source teacher base-text cache profile differs from its receipt"
    );

    let mut actual_records = Vec::with_capacity(receipt.plan.buffer_records.len());
    let mut full_rank = 0usize;
    let mut linear_rank = 0usize;
    for (layer_index, kind) in config.layer_types.iter().copied().enumerate() {
        match kind {
            Qwen35LayerKind::FullAttention => {
                ensure!(
                    cache.per_layer_slot[layer_index] == LayerSlot::Full(full_rank as u32),
                    "source teacher full-attention cache rank differs from schedule"
                );
                let slot = &cache.full_attn[full_rank];
                ensure!(
                    slot.tq.is_none() && slot.current_len.as_slice() == [0],
                    "source teacher full-attention cache is not fresh F32-only state"
                );
                actual_records.push(actual_buffer(
                    format!("blk.{layer_index}.base_text_cache.k"),
                    slot.k
                        .as_ref()
                        .context("source teacher cache K is absent")?,
                    receipt.device_registry_id,
                )?);
                actual_records.push(actual_buffer(
                    format!("blk.{layer_index}.base_text_cache.v"),
                    slot.v
                        .as_ref()
                        .context("source teacher cache V is absent")?,
                    receipt.device_registry_id,
                )?);
                full_rank += 1;
            }
            Qwen35LayerKind::LinearAttention => {
                ensure!(
                    cache.per_layer_slot[layer_index] == LayerSlot::Linear(linear_rank as u32),
                    "source teacher linear-attention cache rank differs from schedule"
                );
                let slot = &cache.linear_attn[linear_rank];
                ensure!(
                    slot.capture_states.is_none()
                        && slot.conv_capture_states.is_none()
                        && slot.pp_flipped.as_slice() == [false],
                    "source teacher linear-attention cache is not fresh base-only state"
                );
                for (suffix, buffer) in [
                    ("conv_state", &slot.conv_state),
                    ("conv_state_scratch", &slot.conv_state_scratch),
                    ("recurrent", &slot.recurrent),
                    ("recurrent_scratch", &slot.recurrent_scratch),
                ] {
                    actual_records.push(actual_buffer(
                        format!("blk.{layer_index}.base_text_cache.{suffix}"),
                        buffer,
                        receipt.device_registry_id,
                    )?);
                }
                linear_rank += 1;
            }
        }
    }
    ensure!(
        actual_records == receipt.plan.buffer_records,
        "source teacher base-text cache buffers differ from the checked plan"
    );
    let actual_payload_bytes = actual_records
        .iter()
        .try_fold(0_u64, |total, record| total.checked_add(record.byte_len));
    ensure!(
        actual_payload_bytes == Some(receipt.actual_payload_bytes)
            && u64::try_from(cache.total_bytes())? == receipt.actual_payload_bytes,
        "source teacher base-text cache payload accounting differs"
    );
    Ok(())
}

fn planned_buffer(role: String, shape: Vec<usize>) -> Result<Qwen35BaseTextCacheBufferPlanV1> {
    ensure!(
        !shape.is_empty() && shape.iter().all(|dimension| *dimension > 0),
        "source teacher cache buffer shape is empty or zero"
    );
    let elements = shape.iter().try_fold(1_u64, |product, dimension| {
        product.checked_mul(*dimension as u64)
    });
    let byte_len = elements
        .and_then(|elements| elements.checked_mul(4))
        .context("source teacher cache buffer bytes overflow")?;
    ensure!(
        byte_len <= usize::MAX as u64,
        "source teacher cache buffer exceeds usize"
    );
    Ok(Qwen35BaseTextCacheBufferPlanV1 {
        role,
        shape,
        dtype: "f32",
        byte_len,
    })
}

fn actual_buffer(
    role: String,
    buffer: &MlxBuffer,
    device_registry_id: u64,
) -> Result<Qwen35BaseTextCacheBufferPlanV1> {
    ensure!(
        buffer.dtype() == DType::F32
            && buffer.byte_len() == buffer.data_byte_len()
            && buffer.byte_offset() == 0
            && !buffer.is_file_backed()
            && buffer.is_cpu_writable()
            && buffer.metal_buffer().device().registry_id() == device_registry_id,
        "source teacher cache Metal buffer metadata differs from profile"
    );
    let record = planned_buffer(role, buffer.shape().to_vec())?;
    ensure!(
        u64::try_from(buffer.byte_len())? == record.byte_len,
        "source teacher cache Metal buffer bytes differ from shape"
    );
    Ok(record)
}

fn plan_sha256(plan: &Qwen35BaseTextCachePlanV1) -> Result<String> {
    let view = CachePlanHashView {
        schema_version: plan.schema_version,
        profile: plan.profile,
        max_sequence_tokens: plan.max_sequence_tokens,
        n_sequences: plan.n_sequences,
        full_attention_slots: plan.full_attention_slots,
        linear_attention_slots: plan.linear_attention_slots,
        buffer_records: &plan.buffer_records,
        base_full_attention_cache_bytes: plan.base_full_attention_cache_bytes,
        base_linear_attention_state_bytes: plan.base_linear_attention_state_bytes,
        total_payload_bytes: plan.total_payload_bytes,
        mtp_slot_allocated: plan.mtp_slot_allocated,
        tq_kv_active: plan.tq_kv_active,
        linear_capture_allocated: plan.linear_capture_allocated,
    };
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&view)?)))
}

fn receipt_sha256(receipt: &Qwen35BaseTextCacheReceiptV1) -> Result<String> {
    let view = CacheReceiptHashView {
        schema_version: receipt.schema_version,
        profile: receipt.profile,
        layout_sha256: &receipt.plan.layout_sha256,
        device_name: &receipt.device_name,
        device_registry_id: receipt.device_registry_id,
        actual_payload_bytes: receipt.actual_payload_bytes,
        fresh_semantic_state: receipt.fresh_semantic_state,
    };
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&view)?)))
}
