use std::collections::BTreeSet;

use anyhow::{ensure, Context, Result};
use serde::Serialize;

use super::snapshot::VerifiedQwenSourceSnapshot;
use super::topology::{
    Qwen35FutureDType, Qwen35QGateBranch, Qwen35SourceTopologyRecord, Qwen35SourceTransformV1,
    Qwen35SourceUse,
};
use super::types::{SourcePrecisionDType, SOURCE_READ_CHUNK_BYTES};

const MAX_UPLOAD_OUTPUT_TENSORS: usize = 4_096;
const MAX_UPLOAD_TOTAL_OUTPUT_BYTES: u64 = 128 * 1024 * 1024 * 1024;
const MAX_UPLOAD_SINGLE_BUFFER_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_UPLOAD_RESERVE_BYTES: u64 = 128 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct QwenSourceMetalUploadLimits {
    pub max_output_tensors: usize,
    pub max_total_output_bytes: u64,
    pub max_single_buffer_bytes: u64,
    pub host_reserve_bytes: u64,
    pub metal_reserve_bytes: u64,
}

impl QwenSourceMetalUploadLimits {
    pub(crate) fn validate(self) -> Result<()> {
        ensure!(
            self.max_output_tensors > 0
                && self.max_output_tensors <= MAX_UPLOAD_OUTPUT_TENSORS
                && self.max_total_output_bytes > 0
                && self.max_total_output_bytes <= MAX_UPLOAD_TOTAL_OUTPUT_BYTES
                && self.max_single_buffer_bytes > 0
                && self.max_single_buffer_bytes <= MAX_UPLOAD_SINGLE_BUFFER_BYTES
                && self.host_reserve_bytes <= MAX_UPLOAD_RESERVE_BYTES
                && self.metal_reserve_bytes <= MAX_UPLOAD_RESERVE_BYTES,
            "source Metal upload limits exceed the hard v1 envelope"
        );
        Ok(())
    }
}

impl Default for QwenSourceMetalUploadLimits {
    fn default() -> Self {
        Self {
            max_output_tensors: MAX_UPLOAD_OUTPUT_TENSORS,
            max_total_output_bytes: MAX_UPLOAD_TOTAL_OUTPUT_BYTES,
            max_single_buffer_bytes: MAX_UPLOAD_SINGLE_BUFFER_BYTES,
            host_reserve_bytes: 8 * 1024 * 1024 * 1024,
            metal_reserve_bytes: 8 * 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct QwenSourceMetalCapacityV1 {
    pub(super) host_available_bytes: u64,
    pub(super) metal_recommended_working_set_bytes: u64,
    pub(super) metal_current_allocated_bytes: u64,
    pub(super) metal_max_buffer_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(super) struct QwenSourceMetalUploadPreflightV1 {
    pub(super) source_count: usize,
    pub(super) output_tensor_count: usize,
    pub(super) bf16_tensor_count: usize,
    pub(super) f32_tensor_count: usize,
    pub(super) total_output_bytes: u64,
    pub(super) max_single_buffer_bytes: u64,
    pub(super) loader_scratch_bytes: u64,
    /// Logical output payload plus the retained-source read scratch. The
    /// caller-provided reserves are an allowance for unmeasured Rust/Metal
    /// bookkeeping and allocator page granularity; this is not a measured
    /// process peak.
    pub(super) accounted_payload_and_scratch_bytes: u64,
}

pub(super) fn preflight_upload(
    snapshot: &VerifiedQwenSourceSnapshot,
    records: &[Qwen35SourceTopologyRecord],
    expected_bf16_tensors: usize,
    expected_f32_tensors: usize,
    limits: QwenSourceMetalUploadLimits,
    capacity: QwenSourceMetalCapacityV1,
) -> Result<QwenSourceMetalUploadPreflightV1> {
    limits.validate()?;
    ensure!(
        capacity.host_available_bytes > 0
            && capacity.metal_recommended_working_set_bytes > 0
            && capacity.metal_max_buffer_bytes > 0,
        "source Metal upload capacity observation is incomplete"
    );
    ensure!(
        snapshot.tensor_count() == records.len(),
        "retained source/topology cardinality differs before Metal upload"
    );

    let mut seen_nodes = BTreeSet::new();
    let mut output_tensor_count = 0_usize;
    let mut bf16_tensor_count = 0_usize;
    let mut f32_tensor_count = 0_usize;
    let mut total_output_bytes = 0_u64;
    let mut max_single_buffer_bytes = 0_u64;
    for source in records {
        let retained = snapshot
            .tensor_record(&source.source_name)
            .with_context(|| format!("retained source {} is absent", source.source_name))?;
        ensure!(
            retained.shape == source.source_shape
                && retained.byte_sha256 == source.source_byte_sha256
                && retained.disposition == source.disposition,
            "retained source {} differs from the B2a topology",
            source.source_name
        );
        let retained_elements = checked_elements(&retained.shape)?;
        ensure!(
            retained.byte_len
                == retained_elements
                    .checked_mul(2)
                    .context("retained source BF16 byte count overflow")?
                && (source.source_use == Qwen35SourceUse::ExcludedVision
                    || retained.dtype == SourcePrecisionDType::Bf16),
            "retained source {} has an invalid dtype or byte count for B2b",
            source.source_name
        );
        match source.source_use {
            Qwen35SourceUse::FutureExecution => ensure!(
                !source.outputs.is_empty(),
                "future-execution source {} has no output",
                source.source_name
            ),
            Qwen35SourceUse::AuthenticatedNonExecutedMtp | Qwen35SourceUse::ExcludedVision => {
                ensure!(
                    source.outputs.is_empty(),
                    "nonexecuted source {} unexpectedly has an output",
                    source.source_name
                )
            }
        }
        validate_source_transform(source)?;
        for output in &source.outputs {
            ensure!(
                seen_nodes.insert(output.node_id.as_str()),
                "source Metal output node {} is duplicated",
                output.node_id
            );
            ensure!(
                !output.shape.is_empty() && output.shape.iter().all(|dimension| *dimension > 0),
                "source Metal output {} has an empty or zero dimension",
                output.node_id
            );
            let elements = output.shape.iter().try_fold(1_u64, |product, dimension| {
                product.checked_mul(u64::try_from(*dimension).ok()?)
            });
            let elements = elements.context("source Metal output element count overflow")?;
            let element_bytes = match output.dtype {
                Qwen35FutureDType::Bf16 => {
                    bf16_tensor_count = bf16_tensor_count
                        .checked_add(1)
                        .context("source Metal BF16 count overflow")?;
                    2_u64
                }
                Qwen35FutureDType::F32 => {
                    f32_tensor_count = f32_tensor_count
                        .checked_add(1)
                        .context("source Metal F32 count overflow")?;
                    4_u64
                }
            };
            let bytes = elements
                .checked_mul(element_bytes)
                .context("source Metal output byte count overflow")?;
            ensure!(
                bytes <= limits.max_single_buffer_bytes && bytes <= capacity.metal_max_buffer_bytes,
                "source Metal output {} requires {bytes} bytes, exceeding the single-buffer bound",
                output.node_id
            );
            output_tensor_count = output_tensor_count
                .checked_add(1)
                .context("source Metal output count overflow")?;
            total_output_bytes = total_output_bytes
                .checked_add(bytes)
                .context("source Metal total output bytes overflow")?;
            max_single_buffer_bytes = max_single_buffer_bytes.max(bytes);
        }
    }
    ensure!(
        output_tensor_count <= limits.max_output_tensors
            && total_output_bytes <= limits.max_total_output_bytes,
        "source Metal upload exceeds the requested output bound"
    );
    ensure!(
        bf16_tensor_count == expected_bf16_tensors && f32_tensor_count == expected_f32_tensors,
        "source Metal upload output dtype counts differ from B2a"
    );

    let loader_scratch_bytes = u64::try_from(SOURCE_READ_CHUNK_BYTES)?;
    let accounted_payload_and_scratch_bytes = total_output_bytes
        .checked_add(loader_scratch_bytes)
        .context("source Metal upload peak byte count overflow")?;
    let host_capacity = capacity
        .host_available_bytes
        .checked_sub(limits.host_reserve_bytes)
        .context("source Metal host reserve exceeds available memory")?;
    let metal_capacity = capacity
        .metal_recommended_working_set_bytes
        .checked_sub(capacity.metal_current_allocated_bytes)
        .and_then(|bytes| bytes.checked_sub(limits.metal_reserve_bytes))
        .context("source Metal reserve exceeds the remaining recommended working set")?;
    ensure!(
        accounted_payload_and_scratch_bytes <= host_capacity
            && accounted_payload_and_scratch_bytes <= metal_capacity,
        "source Metal accounted payload and scratch {accounted_payload_and_scratch_bytes} exceeds host/Metal capacity ({host_capacity}/{metal_capacity})"
    );

    Ok(QwenSourceMetalUploadPreflightV1 {
        source_count: records.len(),
        output_tensor_count,
        bf16_tensor_count,
        f32_tensor_count,
        total_output_bytes,
        max_single_buffer_bytes,
        loader_scratch_bytes,
        accounted_payload_and_scratch_bytes,
    })
}

fn checked_elements(shape: &[usize]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |product, dimension| {
        product
            .checked_mul(u64::try_from(*dimension)?)
            .context("source Metal transform element count overflow")
    })
}

fn validate_source_transform(source: &Qwen35SourceTopologyRecord) -> Result<()> {
    if source.outputs.is_empty() {
        return Ok(());
    }
    let source_elements = checked_elements(&source.source_shape)?;
    if source.outputs.len() == 2 {
        return validate_q_gate_split(source, source_elements);
    }
    ensure!(
        source.outputs.len() == 1,
        "source {} has unsupported output fanout {}",
        source.source_name,
        source.outputs.len()
    );
    let output = &source.outputs[0];
    let output_elements = checked_elements(&output.shape)?;
    ensure!(
        output_elements == source_elements,
        "source {} transform changes element count unexpectedly",
        source.source_name
    );
    match &output.transform {
        Qwen35SourceTransformV1::Identity => {}
        Qwen35SourceTransformV1::AddOneF32 => ensure!(
            output.dtype == Qwen35FutureDType::F32,
            "source {} AddOne output must be F32",
            source.source_name
        ),
        Qwen35SourceTransformV1::ReorderVHeads {
            num_key_heads,
            num_values_per_key,
            block_elements,
            slice_start,
            slice_end,
        } => {
            let reordered =
                checked_product(&[*num_key_heads, *num_values_per_key, *block_elements])?;
            let (start, end) = match (*slice_start, *slice_end) {
                (None, None) => (0_u64, source_elements),
                (Some(start), Some(end)) => (u64::try_from(start)?, u64::try_from(end)?),
                _ => anyhow::bail!(
                    "source {} reorder has a partial slice descriptor",
                    source.source_name
                ),
            };
            ensure!(
                start <= end && end <= source_elements && end - start == reordered,
                "source {} reorder slice geometry is invalid",
                source.source_name
            );
        }
        Qwen35SourceTransformV1::ReorderVHeadsThenNegExpF32 {
            num_key_heads,
            num_values_per_key,
        } => ensure!(
            output.dtype == Qwen35FutureDType::F32
                && checked_product(&[*num_key_heads, *num_values_per_key])? == source_elements,
            "source {} reorder/NegExp geometry is invalid",
            source.source_name
        ),
        Qwen35SourceTransformV1::SqueezeAxis1ThenReorderVSlice {
            num_key_heads,
            num_values_per_key,
            value_head_dim,
            kernel_width,
            slice_start,
            slice_end,
        } => {
            ensure!(
                output.dtype == Qwen35FutureDType::F32
                    && source.source_shape.len() == 3
                    && source.source_shape[1] == 1
                    && output.shape.len() == 2
                    && source.source_shape[0] == output.shape[0]
                    && source.source_shape[2] == output.shape[1],
                "source {} squeeze geometry is invalid",
                source.source_name
            );
            let reordered = checked_product(&[
                *num_key_heads,
                *num_values_per_key,
                *value_head_dim,
                *kernel_width,
            ])?;
            ensure!(
                *slice_start <= *slice_end
                    && u64::try_from(*slice_end)? <= source_elements
                    && u64::try_from(*slice_end - *slice_start)? == reordered,
                "source {} squeezed reorder slice is invalid",
                source.source_name
            );
        }
        Qwen35SourceTransformV1::ReorderVHeadsPerRow {
            row_count,
            num_key_heads,
            num_values_per_key,
            value_head_dim,
        } => ensure!(
            checked_product(&[
                *row_count,
                *num_key_heads,
                *num_values_per_key,
                *value_head_dim,
            ])? == source_elements,
            "source {} per-row reorder geometry is invalid",
            source.source_name
        ),
        Qwen35SourceTransformV1::SplitInterleavedQGate { .. } => anyhow::bail!(
            "source {} has an incomplete Q/gate fanout",
            source.source_name
        ),
    }
    Ok(())
}

fn validate_q_gate_split(source: &Qwen35SourceTopologyRecord, source_elements: u64) -> Result<()> {
    let [query, gate] = source.outputs.as_slice() else {
        unreachable!();
    };
    let split = |output: &super::topology::Qwen35FutureTensorRecord| match &output.transform {
        Qwen35SourceTransformV1::SplitInterleavedQGate {
            branch,
            num_query_heads,
            head_dim,
            hidden_size,
        } => Some((*branch, *num_query_heads, *head_dim, *hidden_size)),
        _ => None,
    };
    let query_split = split(query).context("Q/gate query output lacks a split transform")?;
    let gate_split = split(gate).context("Q/gate gate output lacks a split transform")?;
    ensure!(
        query.dtype == Qwen35FutureDType::Bf16
            && gate.dtype == Qwen35FutureDType::Bf16
            && query_split.0 == Qwen35QGateBranch::Query
            && gate_split.0 == Qwen35QGateBranch::Gate
            && (query_split.1, query_split.2, query_split.3)
                == (gate_split.1, gate_split.2, gate_split.3),
        "source {} Q/gate split roles or parameters differ",
        source.source_name
    );
    let output_elements = checked_product(&[query_split.1, query_split.2, query_split.3])?;
    ensure!(
        checked_elements(&query.shape)? == output_elements
            && checked_elements(&gate.shape)? == output_elements
            && output_elements.checked_mul(2) == Some(source_elements),
        "source {} Q/gate split element counts are invalid",
        source.source_name
    );
    Ok(())
}

fn checked_product(values: &[usize]) -> Result<u64> {
    values.iter().try_fold(1_u64, |product, value| {
        ensure!(*value > 0, "source Metal transform dimension is zero");
        product
            .checked_mul(u64::try_from(*value)?)
            .context("source Metal transform product overflow")
    })
}
