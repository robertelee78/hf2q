use anyhow::{bail, ensure, Context, Result};
use half::bf16;
use mlx_native::{DType, MlxBuffer};
use sha2::{Digest, Sha256};

use super::snapshot::VerifiedQwenSourceSnapshot;
use super::topology::{
    Qwen35FutureDType, Qwen35FutureTensorRecord, Qwen35QGateBranch, Qwen35SourceTopologyRecord,
    Qwen35SourceTransformV1,
};

pub(super) struct UploadedTensorBuffer {
    pub(super) node_id: String,
    pub(super) shape: Vec<usize>,
    pub(super) dtype: Qwen35FutureDType,
    pub(super) byte_len: u64,
    pub(super) buffer_byte_sha256: String,
    pub(super) buffer: MlxBuffer,
}

pub(super) fn upload_source<A>(
    snapshot: &VerifiedQwenSourceSnapshot,
    source: &Qwen35SourceTopologyRecord,
    expected_device_registry_id: u64,
    scratch: &mut Vec<u8>,
    allocate: &mut A,
) -> Result<Vec<UploadedTensorBuffer>>
where
    A: FnMut(usize, DType, Vec<usize>) -> Result<MlxBuffer>,
{
    if source.outputs.is_empty() {
        return Ok(Vec::new());
    }
    let mut uploaded = source
        .outputs
        .iter()
        .map(|output| allocate_output(output, expected_device_registry_id, allocate))
        .collect::<Result<Vec<_>>>()?;
    if uploaded.len() == 2 {
        fill_q_gate(snapshot, source, scratch, &mut uploaded)?;
    } else {
        ensure!(uploaded.len() == 1, "unsupported source Metal fanout");
        fill_single(snapshot, source, scratch, &mut uploaded[0])?;
    }
    for output in &mut uploaded {
        verify_and_hash_output(output)?;
    }
    Ok(uploaded)
}

fn allocate_output<A>(
    output: &Qwen35FutureTensorRecord,
    expected_device_registry_id: u64,
    allocate: &mut A,
) -> Result<UploadedTensorBuffer>
where
    A: FnMut(usize, DType, Vec<usize>) -> Result<MlxBuffer>,
{
    let elements = checked_elements(&output.shape)?;
    let (dtype, element_bytes) = match output.dtype {
        Qwen35FutureDType::Bf16 => (DType::BF16, 2_usize),
        Qwen35FutureDType::F32 => (DType::F32, 4_usize),
    };
    let byte_len = elements
        .checked_mul(element_bytes)
        .context("source Metal output byte count overflow")?;
    let buffer = allocate(byte_len, dtype, output.shape.clone())?;
    ensure!(
        buffer.dtype() == dtype
            && buffer.shape() == output.shape
            && buffer.byte_len() == byte_len
            && buffer.data_byte_len() == byte_len
            && buffer.byte_offset() == 0
            && !buffer.is_file_backed()
            && buffer.is_cpu_writable()
            && buffer.metal_buffer().device().registry_id() == expected_device_registry_id,
        "allocated source Metal buffer metadata differs from its plan"
    );
    Ok(UploadedTensorBuffer {
        node_id: output.node_id.clone(),
        shape: output.shape.clone(),
        dtype: output.dtype,
        byte_len: u64::try_from(byte_len)?,
        buffer_byte_sha256: String::new(),
        buffer,
    })
}

fn fill_single(
    snapshot: &VerifiedQwenSourceSnapshot,
    source: &Qwen35SourceTopologyRecord,
    scratch: &mut Vec<u8>,
    uploaded: &mut UploadedTensorBuffer,
) -> Result<()> {
    let output = &source.outputs[0];
    let expected_elements = checked_elements(&output.shape)?;
    let mut written = 0_usize;
    match output.dtype {
        Qwen35FutureDType::Bf16 => {
            let destination = uploaded.buffer.as_mut_slice::<u16>()?;
            if matches!(output.transform, Qwen35SourceTransformV1::Identity)
                && cfg!(target_endian = "little")
            {
                let destination_bytes: &mut [u8] = bytemuck::cast_slice_mut(destination);
                snapshot.visit_tensor_le_bytes(
                    &source.source_name,
                    scratch,
                    |element_offset, bytes| {
                        let byte_offset = element_offset
                            .checked_mul(2)
                            .context("source Metal identity byte offset overflow")?;
                        let byte_end = byte_offset
                            .checked_add(bytes.len())
                            .context("source Metal identity byte end overflow")?;
                        destination_bytes
                            .get_mut(byte_offset..byte_end)
                            .context("source Metal identity copy exceeds output")?
                            .copy_from_slice(bytes);
                        written = written
                            .checked_add(bytes.len() / 2)
                            .context("source Metal write count overflow")?;
                        Ok(())
                    },
                )?;
                ensure!(
                    written == expected_elements,
                    "source {} wrote {written} values, expected {expected_elements}",
                    source.source_name
                );
                return Ok(());
            }
            snapshot.visit_tensor_le_bytes(
                &source.source_name,
                scratch,
                |element_offset, bytes| {
                    for (local, pair) in bytes.chunks_exact(2).enumerate() {
                        let source_index = element_offset
                            .checked_add(local)
                            .context("source Metal input index overflow")?;
                        let destination_index =
                            map_single_index(source_index, expected_elements, &output.transform)?;
                        destination[destination_index] = u16::from_le_bytes([pair[0], pair[1]]);
                        written = written
                            .checked_add(1)
                            .context("source Metal write count overflow")?;
                    }
                    Ok(())
                },
            )?;
        }
        Qwen35FutureDType::F32 => {
            let destination = uploaded.buffer.as_mut_slice::<f32>()?;
            snapshot.visit_tensor_le_bytes(
                &source.source_name,
                scratch,
                |element_offset, bytes| {
                    for (local, pair) in bytes.chunks_exact(2).enumerate() {
                        let source_index = element_offset
                            .checked_add(local)
                            .context("source Metal input index overflow")?;
                        let destination_index =
                            map_single_index(source_index, expected_elements, &output.transform)?;
                        let value =
                            bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32();
                        destination[destination_index] = transform_f32(value, &output.transform)?;
                        written = written
                            .checked_add(1)
                            .context("source Metal write count overflow")?;
                    }
                    Ok(())
                },
            )?;
        }
    }
    ensure!(
        written == expected_elements,
        "source {} wrote {written} values, expected {expected_elements}",
        source.source_name
    );
    Ok(())
}

fn fill_q_gate(
    snapshot: &VerifiedQwenSourceSnapshot,
    source: &Qwen35SourceTopologyRecord,
    scratch: &mut Vec<u8>,
    uploaded: &mut [UploadedTensorBuffer],
) -> Result<()> {
    let (first, second) = uploaded.split_at_mut(1);
    let query = &mut first[0];
    let gate = &mut second[0];
    let query_words = query.buffer.as_mut_slice::<u16>()?;
    let gate_words = gate.buffer.as_mut_slice::<u16>()?;
    let (num_query_heads, head_dim, hidden_size) = split_parameters(&source.outputs[0])?;
    ensure!(
        split_parameters(&source.outputs[1])? == (num_query_heads, head_dim, hidden_size),
        "Q/gate split parameters differ after preflight"
    );
    let source_rows = num_query_heads
        .checked_mul(2)
        .and_then(|value| value.checked_mul(head_dim))
        .context("Q/gate source row count overflow")?;
    let source_elements = source_rows
        .checked_mul(hidden_size)
        .context("Q/gate source element count overflow")?;
    let mut query_writes = 0_usize;
    let mut gate_writes = 0_usize;
    snapshot.visit_tensor_le_bytes(&source.source_name, scratch, |element_offset, bytes| {
        for (local, pair) in bytes.chunks_exact(2).enumerate() {
            let source_index = element_offset
                .checked_add(local)
                .context("Q/gate source index overflow")?;
            ensure!(
                source_index < source_elements,
                "Q/gate source index exceeds plan"
            );
            let row = source_index / hidden_size;
            let column = source_index % hidden_size;
            let head = row / (2 * head_dim);
            let within_head = row % (2 * head_dim);
            let (branch, branch_row) = if within_head < head_dim {
                (Qwen35QGateBranch::Query, head * head_dim + within_head)
            } else {
                (
                    Qwen35QGateBranch::Gate,
                    head * head_dim + within_head - head_dim,
                )
            };
            let destination_index = branch_row
                .checked_mul(hidden_size)
                .and_then(|value| value.checked_add(column))
                .context("Q/gate destination index overflow")?;
            let word = u16::from_le_bytes([pair[0], pair[1]]);
            match branch {
                Qwen35QGateBranch::Query => {
                    query_words[destination_index] = word;
                    query_writes += 1;
                }
                Qwen35QGateBranch::Gate => {
                    gate_words[destination_index] = word;
                    gate_writes += 1;
                }
            }
        }
        Ok(())
    })?;
    ensure!(
        query_writes == query_words.len() && gate_writes == gate_words.len(),
        "Q/gate split did not initialize every output value"
    );
    Ok(())
}

fn split_parameters(output: &Qwen35FutureTensorRecord) -> Result<(usize, usize, usize)> {
    match output.transform {
        Qwen35SourceTransformV1::SplitInterleavedQGate {
            num_query_heads,
            head_dim,
            hidden_size,
            ..
        } => Ok((num_query_heads, head_dim, hidden_size)),
        _ => bail!("output {} is not a Q/gate split", output.node_id),
    }
}

fn map_single_index(
    source_index: usize,
    element_count: usize,
    transform: &Qwen35SourceTransformV1,
) -> Result<usize> {
    ensure!(
        source_index < element_count,
        "source transform index exceeds plan"
    );
    match transform {
        Qwen35SourceTransformV1::Identity | Qwen35SourceTransformV1::AddOneF32 => Ok(source_index),
        Qwen35SourceTransformV1::ReorderVHeads {
            num_key_heads,
            num_values_per_key,
            block_elements,
            slice_start,
            slice_end,
        } => map_reorder_index(
            source_index,
            element_count,
            *num_key_heads,
            *num_values_per_key,
            *block_elements,
            *slice_start,
            *slice_end,
        ),
        Qwen35SourceTransformV1::ReorderVHeadsThenNegExpF32 {
            num_key_heads,
            num_values_per_key,
        } => map_reorder_index(
            source_index,
            element_count,
            *num_key_heads,
            *num_values_per_key,
            1,
            None,
            None,
        ),
        Qwen35SourceTransformV1::SqueezeAxis1ThenReorderVSlice {
            num_key_heads,
            num_values_per_key,
            value_head_dim,
            kernel_width,
            slice_start,
            slice_end,
        } => map_reorder_index(
            source_index,
            element_count,
            *num_key_heads,
            *num_values_per_key,
            value_head_dim
                .checked_mul(*kernel_width)
                .context("squeezed reorder block overflow")?,
            Some(*slice_start),
            Some(*slice_end),
        ),
        Qwen35SourceTransformV1::ReorderVHeadsPerRow {
            row_count,
            num_key_heads,
            num_values_per_key,
            value_head_dim,
        } => {
            let row_elements = num_key_heads
                .checked_mul(*num_values_per_key)
                .and_then(|value| value.checked_mul(*value_head_dim))
                .context("per-row reorder width overflow")?;
            ensure!(
                row_count.checked_mul(row_elements) == Some(element_count),
                "per-row reorder geometry differs from output"
            );
            let row = source_index / row_elements;
            let within_row = source_index % row_elements;
            Ok(row * row_elements
                + map_reorder_local(
                    within_row,
                    *num_key_heads,
                    *num_values_per_key,
                    *value_head_dim,
                )?)
        }
        Qwen35SourceTransformV1::SplitInterleavedQGate { .. } => {
            bail!("split transform reached the single-output path")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn map_reorder_index(
    source_index: usize,
    element_count: usize,
    num_key_heads: usize,
    num_values_per_key: usize,
    block_elements: usize,
    slice_start: Option<usize>,
    slice_end: Option<usize>,
) -> Result<usize> {
    let (start, end) = match (slice_start, slice_end) {
        (None, None) => (0, element_count),
        (Some(start), Some(end)) => (start, end),
        _ => bail!("reorder has a partial slice descriptor"),
    };
    if source_index < start || source_index >= end {
        return Ok(source_index);
    }
    Ok(start
        + map_reorder_local(
            source_index - start,
            num_key_heads,
            num_values_per_key,
            block_elements,
        )?)
}

fn map_reorder_local(
    source_index: usize,
    num_key_heads: usize,
    num_values_per_key: usize,
    block_elements: usize,
) -> Result<usize> {
    let per_key = num_values_per_key
        .checked_mul(block_elements)
        .context("reorder key width overflow")?;
    let key = source_index / per_key;
    let within_key = source_index % per_key;
    let value = within_key / block_elements;
    let within_block = within_key % block_elements;
    ensure!(key < num_key_heads, "reorder key index exceeds plan");
    value
        .checked_mul(num_key_heads)
        .and_then(|value| value.checked_add(key))
        .and_then(|value| value.checked_mul(block_elements))
        .and_then(|value| value.checked_add(within_block))
        .context("reorder destination index overflow")
}

fn transform_f32(value: f32, transform: &Qwen35SourceTransformV1) -> Result<f32> {
    match transform {
        Qwen35SourceTransformV1::Identity
        | Qwen35SourceTransformV1::ReorderVHeads { .. }
        | Qwen35SourceTransformV1::SqueezeAxis1ThenReorderVSlice { .. }
        | Qwen35SourceTransformV1::ReorderVHeadsPerRow { .. } => Ok(value),
        Qwen35SourceTransformV1::AddOneF32 => Ok(value + 1.0),
        Qwen35SourceTransformV1::ReorderVHeadsThenNegExpF32 { .. } => {
            Ok(-crate::convert::sleef_expf::sleef_expf(value))
        }
        Qwen35SourceTransformV1::SplitInterleavedQGate { .. } => {
            bail!("split transform reached the F32 path")
        }
    }
}

fn verify_and_hash_output(output: &mut UploadedTensorBuffer) -> Result<()> {
    let expected_dtype = match output.dtype {
        Qwen35FutureDType::Bf16 => DType::BF16,
        Qwen35FutureDType::F32 => DType::F32,
    };
    ensure!(
        output.buffer.dtype() == expected_dtype
            && output.buffer.shape() == output.shape
            && u64::try_from(output.buffer.byte_len())? == output.byte_len
            && u64::try_from(output.buffer.data_byte_len())? == output.byte_len
            && output.buffer.byte_offset() == 0
            && !output.buffer.is_file_backed(),
        "uploaded buffer {} metadata changed",
        output.node_id
    );
    let bytes: &[u8] = match output.dtype {
        Qwen35FutureDType::Bf16 => bytemuck::cast_slice(output.buffer.as_slice::<u16>()?),
        Qwen35FutureDType::F32 => bytemuck::cast_slice(output.buffer.as_slice::<f32>()?),
    };
    ensure!(
        u64::try_from(bytes.len())? == output.byte_len,
        "uploaded buffer {} byte view differs from plan",
        output.node_id
    );
    output.buffer_byte_sha256 = hex::encode(Sha256::digest(bytes));
    Ok(())
}

fn checked_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1_usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .context("source Metal output element count overflow")
    })
}
