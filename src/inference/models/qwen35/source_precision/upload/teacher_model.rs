//! Family-owned preparation boundary for the source-precision Qwen teacher.
//!
//! Preparation consumes the exact B2a topology, performs a combined static
//! upload plus bounded-runtime capacity check before the first Metal weight
//! allocation, uploads through B2b, and drains every uploaded node into the
//! exact dense-Qwen layer slots. The resulting type is intentionally inert:
//! it has no buffer accessor and no forward/session constructor yet.

#[cfg(test)]
use anyhow::ensure;
use anyhow::{Context, Result};
use mlx_native::MlxDevice;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::inference::models::qwen35::gpu_delta_net::DeltaNetWeightsGpu;
use crate::inference::models::qwen35::gpu_ffn::DenseFfnWeightsGpu;
use crate::inference::models::qwen35::gpu_full_attn::FullAttnWeightsGpu;
use crate::inference::models::qwen35::{Qwen35Config, Qwen35LayerKind};

use super::{observe_capacity, upload_with_capacity, VerifiedQwen35Bf16MetalUploadV1};
use crate::inference::models::qwen35::source_precision::topology::VerifiedQwen35Bf16TopologyV1;
use crate::inference::models::qwen35::source_precision::upload_plan::QwenSourceMetalCapacityV1;
use crate::inference::models::qwen35::source_precision::upload_plan::QwenSourceMetalUploadLimits;

mod assemble;
mod layers;
mod preflight;
mod run_inputs;

#[cfg(test)]
mod tests;

const PREPARED_SCHEMA_VERSION: u32 = 1;
const PREPARED_PROFILE: &str = "dense_qwen35_source_bf16_prepared_text_graph_v1";
pub(crate) use preflight::Qwen35SourceTeacherLimitsV1;
use preflight::{
    runtime_envelope, validate_combined_capacity, validate_incremental_capacity,
    Qwen35SourceTeacherRuntimeEnvelopeV1,
};
pub(crate) use run_inputs::{
    prepare_qwen35_source_teacher_run_inputs, PreparedQwen35SourceTeacherRunInputsV1,
    Qwen35SourceTeacherPreparationPolicyV1,
};

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Qwen35SourceTeacherConfigV1 {
    hidden_size: u32,
    intermediate_size: u32,
    vocabulary_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    full_attention_interval: u32,
    layer_types: Vec<&'static str>,
    partial_rotary_factor_bits: u32,
    rope_theta_bits: u64,
    rotary_dim: u32,
    mrope_section: [u32; 4],
    mrope_interleaved: bool,
    rms_norm_eps_bits: u32,
    max_position_embeddings: u32,
    attn_output_gate: bool,
    mtp_num_hidden_layers: u32,
    mtp_use_dedicated_embeddings: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PreparedWeightSlotV1 {
    role: String,
    source_name: String,
    node_id: String,
    shape: Vec<usize>,
    dtype: super::super::topology::Qwen35FutureDType,
    transform: super::super::topology::Qwen35SourceTransformV1,
    byte_len: u64,
    buffer_byte_sha256: String,
}

#[derive(Serialize)]
struct PreparedGraphHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    topology_sha256: &'a str,
    source_snapshot_catalog_sha256: &'a str,
    projected_execution_config_sha256: &'a str,
    weight_slots: &'a [PreparedWeightSlotV1],
    bf16_tensor_count: usize,
    f32_tensor_count: usize,
    bf16_bytes: u64,
    f32_bytes: u64,
    authenticated_nonexecuted_mtp_sources: usize,
    excluded_vision_sources: usize,
    weight_precision: &'static str,
    q4_repack: bool,
    dwq: bool,
    tq: bool,
    mtp_executed: bool,
    graph_executed: bool,
}

#[derive(Serialize)]
struct PreparedReceiptHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    graph_catalog_sha256: &'a str,
    upload_catalog_sha256: &'a str,
    upload_receipt_sha256: &'a str,
    device_name: &'a str,
    device_registry_id: u64,
    runtime: &'a Qwen35SourceTeacherRuntimeEnvelopeV1,
    runtime_liveness_proven: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PreparedQwen35SourceTeacherReceiptV1 {
    schema_version: u32,
    profile: &'static str,
    topology_sha256: String,
    upload_catalog_sha256: String,
    upload_receipt_sha256: String,
    source_snapshot_catalog_sha256: String,
    device_name: String,
    device_registry_id: u64,
    projected_execution_config_sha256: String,
    runtime: Qwen35SourceTeacherRuntimeEnvelopeV1,
    weight_slots: Vec<PreparedWeightSlotV1>,
    bf16_tensor_count: usize,
    f32_tensor_count: usize,
    bf16_bytes: u64,
    f32_bytes: u64,
    authenticated_nonexecuted_mtp_sources: usize,
    excluded_vision_sources: usize,
    weight_precision: &'static str,
    q4_repack: bool,
    dwq: bool,
    tq: bool,
    mtp_executed: bool,
    graph_executed: bool,
    runtime_liveness_proven: bool,
    graph_catalog_sha256: String,
    preparation_receipt_sha256: String,
}

enum PreparedQwen35SourceAttentionV1 {
    Full(FullAttnWeightsGpu),
    Linear(DeltaNetWeightsGpu),
}

struct PreparedQwen35SourceLayerV1 {
    attention: PreparedQwen35SourceAttentionV1,
    ffn: DenseFfnWeightsGpu,
}

/// Opaque, non-executable family-owned source-teacher graph preparation.
///
/// This proves exact config/slot assembly of the B2b buffers. It proves no
/// command encoding, completion, numerical result, target, sensitivity, cost,
/// Dynamic admission, or selector authority.
pub(crate) struct PreparedQwen35SourceTeacherV1 {
    snapshot: super::super::snapshot::VerifiedQwenSourceSnapshot,
    device: MlxDevice,
    config: Qwen35Config,
    embedding: mlx_native::MlxBuffer,
    output_norm: mlx_native::MlxBuffer,
    output: mlx_native::MlxBuffer,
    layers: Vec<PreparedQwen35SourceLayerV1>,
    upload_limits: QwenSourceMetalUploadLimits,
    receipt: PreparedQwen35SourceTeacherReceiptV1,
}

impl PreparedQwen35SourceTeacherV1 {
    pub(crate) fn graph_catalog_sha256(&self) -> &str {
        &self.receipt.graph_catalog_sha256
    }

    pub(crate) fn preparation_receipt_sha256(&self) -> &str {
        &self.receipt.preparation_receipt_sha256
    }

    pub(crate) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn accounted_runtime_payload_bytes(&self) -> u64 {
        self.receipt.runtime.accounted_runtime_payload_bytes
    }

    #[cfg(test)]
    pub(super) fn validate_for_test(&self) -> Result<()> {
        ensure!(self.snapshot.catalog_sha256() == self.receipt.source_snapshot_catalog_sha256);
        ensure!(self.device.registry_id() == self.receipt.device_registry_id);
        ensure!(self.config.num_hidden_layers as usize == self.layers.len());
        ensure!(self.embedding.dtype() == mlx_native::DType::BF16);
        ensure!(self.output_norm.dtype() == mlx_native::DType::F32);
        ensure!(self.output.dtype() == mlx_native::DType::BF16);
        for (kind, layer) in self.config.layer_types.iter().zip(&self.layers) {
            ensure!(matches!(
                (kind, &layer.attention),
                (
                    Qwen35LayerKind::FullAttention,
                    PreparedQwen35SourceAttentionV1::Full(_)
                ) | (
                    Qwen35LayerKind::LinearAttention,
                    PreparedQwen35SourceAttentionV1::Linear(_)
                )
            ));
            ensure!(layer.ffn.gate.dtype() == mlx_native::DType::BF16);
            match &layer.attention {
                PreparedQwen35SourceAttentionV1::Full(weights) => {
                    ensure!(weights.wq.dtype() == mlx_native::DType::BF16);
                }
                PreparedQwen35SourceAttentionV1::Linear(weights) => {
                    ensure!(weights.attn_qkv.dtype() == mlx_native::DType::BF16);
                    ensure!(!weights.ssm_a_cpu.is_empty());
                }
            }
        }
        let graph = hex::encode(Sha256::digest(serde_json::to_vec(
            &PreparedGraphHashView {
                schema_version: self.receipt.schema_version,
                profile: self.receipt.profile,
                topology_sha256: &self.receipt.topology_sha256,
                source_snapshot_catalog_sha256: &self.receipt.source_snapshot_catalog_sha256,
                projected_execution_config_sha256: &self.receipt.projected_execution_config_sha256,
                weight_slots: &self.receipt.weight_slots,
                bf16_tensor_count: self.receipt.bf16_tensor_count,
                f32_tensor_count: self.receipt.f32_tensor_count,
                bf16_bytes: self.receipt.bf16_bytes,
                f32_bytes: self.receipt.f32_bytes,
                authenticated_nonexecuted_mtp_sources: self
                    .receipt
                    .authenticated_nonexecuted_mtp_sources,
                excluded_vision_sources: self.receipt.excluded_vision_sources,
                weight_precision: self.receipt.weight_precision,
                q4_repack: self.receipt.q4_repack,
                dwq: self.receipt.dwq,
                tq: self.receipt.tq,
                mtp_executed: self.receipt.mtp_executed,
                graph_executed: self.receipt.graph_executed,
            },
        )?));
        ensure!(graph == self.receipt.graph_catalog_sha256);
        let receipt = hex::encode(Sha256::digest(serde_json::to_vec(
            &PreparedReceiptHashView {
                schema_version: self.receipt.schema_version,
                profile: self.receipt.profile,
                graph_catalog_sha256: &self.receipt.graph_catalog_sha256,
                upload_catalog_sha256: &self.receipt.upload_catalog_sha256,
                upload_receipt_sha256: &self.receipt.upload_receipt_sha256,
                device_name: &self.receipt.device_name,
                device_registry_id: self.receipt.device_registry_id,
                runtime: &self.receipt.runtime,
                runtime_liveness_proven: self.receipt.runtime_liveness_proven,
            },
        )?));
        ensure!(receipt == self.receipt.preparation_receipt_sha256);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn receipt_json_for_test(&self) -> serde_json::Value {
        serde_json::to_value(&self.receipt).expect("prepared teacher receipt must serialize")
    }
}

/// Perform the combined B2b+B3a transition. Runtime capacity is checked
/// before any Metal weight allocation, then the verified upload is consumed
/// into the inert family-owned graph preparation.
pub(crate) fn prepare_qwen35_source_teacher(
    topology: VerifiedQwen35Bf16TopologyV1,
    device: &MlxDevice,
    upload_limits: QwenSourceMetalUploadLimits,
    teacher_limits: Qwen35SourceTeacherLimitsV1,
) -> Result<PreparedQwen35SourceTeacherV1> {
    let capacity = observe_capacity(device);
    prepare_with_capacity(
        topology,
        device,
        upload_limits,
        teacher_limits,
        capacity,
        |bytes, dtype, shape| Ok(device.alloc_buffer(bytes, dtype, shape)?),
    )
}

/// Consume a previously completed B2b upload. This preserves the standalone
/// B2b type-state as a promotable path, but its capacity check can cover only
/// the incremental runtime envelope because the weight allocation already
/// exists. New callers should prefer [`prepare_qwen35_source_teacher`].
pub(crate) fn prepare_uploaded_qwen35_source_teacher(
    upload: VerifiedQwen35Bf16MetalUploadV1,
    teacher_limits: Qwen35SourceTeacherLimitsV1,
) -> Result<PreparedQwen35SourceTeacherV1> {
    let capacity = observe_capacity(&upload._device);
    prepare_uploaded_with_capacity(upload, teacher_limits, capacity)
}

fn config_hash(config: &Qwen35Config) -> Result<String> {
    let view = Qwen35SourceTeacherConfigV1 {
        hidden_size: config.hidden_size,
        intermediate_size: config
            .intermediate_size
            .context("dense teacher lacks FFN size")?,
        vocabulary_size: config.vocab_size,
        num_hidden_layers: config.num_hidden_layers,
        num_attention_heads: config.num_attention_heads,
        num_key_value_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        linear_num_key_heads: config.linear_num_key_heads,
        linear_num_value_heads: config.linear_num_value_heads,
        linear_key_head_dim: config.linear_key_head_dim,
        linear_value_head_dim: config.linear_value_head_dim,
        linear_conv_kernel_dim: config.linear_conv_kernel_dim,
        full_attention_interval: config.full_attention_interval,
        layer_types: config
            .layer_types
            .iter()
            .map(|kind| match kind {
                Qwen35LayerKind::LinearAttention => "linear_attention",
                Qwen35LayerKind::FullAttention => "full_attention",
            })
            .collect(),
        partial_rotary_factor_bits: config.partial_rotary_factor.to_bits(),
        rope_theta_bits: config.rope_theta.to_bits(),
        rotary_dim: config.rotary_dim,
        mrope_section: config.mrope_section,
        mrope_interleaved: config.mrope_interleaved,
        rms_norm_eps_bits: config.rms_norm_eps.to_bits(),
        max_position_embeddings: config.max_position_embeddings,
        attn_output_gate: config.attn_output_gate,
        mtp_num_hidden_layers: config.mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings: config.mtp_use_dedicated_embeddings,
    };
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&view)?)))
}

fn prepare_with_capacity<A>(
    topology: VerifiedQwen35Bf16TopologyV1,
    device: &MlxDevice,
    upload_limits: QwenSourceMetalUploadLimits,
    teacher_limits: Qwen35SourceTeacherLimitsV1,
    capacity: QwenSourceMetalCapacityV1,
    allocate: A,
) -> Result<PreparedQwen35SourceTeacherV1>
where
    A: FnMut(usize, mlx_native::DType, Vec<usize>) -> Result<mlx_native::MlxBuffer>,
{
    let config = topology.projected_config_for_teacher()?;
    let planned_weight_bytes = topology.planned_output_bytes()?;
    let runtime = runtime_envelope(&config, teacher_limits)?;
    validate_combined_capacity(planned_weight_bytes, &runtime, upload_limits, capacity)?;
    let upload = upload_with_capacity(topology, device, upload_limits, capacity, allocate)?;
    assemble::assemble(upload, config, runtime)
}

fn prepare_uploaded_with_capacity(
    upload: VerifiedQwen35Bf16MetalUploadV1,
    teacher_limits: Qwen35SourceTeacherLimitsV1,
    capacity: QwenSourceMetalCapacityV1,
) -> Result<PreparedQwen35SourceTeacherV1> {
    let config =
        crate::inference::models::qwen35::source_config::qwen35_config_from_authenticated_source(
            upload._snapshot.config(),
        )?;
    let runtime = runtime_envelope(&config, teacher_limits)?;
    validate_incremental_capacity(&runtime, upload.receipt.limits, capacity)?;
    assemble::assemble(upload, config, runtime)
}

#[cfg(test)]
pub(super) fn prepare_with_capacity_for_test<A>(
    topology: VerifiedQwen35Bf16TopologyV1,
    device: &MlxDevice,
    upload_limits: QwenSourceMetalUploadLimits,
    teacher_limits: Qwen35SourceTeacherLimitsV1,
    capacity: QwenSourceMetalCapacityV1,
    allocate: A,
) -> Result<PreparedQwen35SourceTeacherV1>
where
    A: FnMut(usize, mlx_native::DType, Vec<usize>) -> Result<mlx_native::MlxBuffer>,
{
    prepare_with_capacity(
        topology,
        device,
        upload_limits,
        teacher_limits,
        capacity,
        allocate,
    )
}

#[cfg(test)]
pub(super) fn prepare_uploaded_with_capacity_for_test(
    upload: VerifiedQwen35Bf16MetalUploadV1,
    teacher_limits: Qwen35SourceTeacherLimitsV1,
    capacity: QwenSourceMetalCapacityV1,
) -> Result<PreparedQwen35SourceTeacherV1> {
    prepare_uploaded_with_capacity(upload, teacher_limits, capacity)
}
