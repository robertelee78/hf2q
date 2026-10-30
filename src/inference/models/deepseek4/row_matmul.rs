//! Row semantics for DeepSeek-V4 matrix products.
//!
//! Prompt graphs pack consecutive tokens of one request and may route an
//! `m = rows` product through any matrix kernel. Joint decode packs one token
//! from each of several independent requests and must reproduce every
//! request's one-row decode bit for bit. Block-quantized weights keep a single
//! `m = rows` dispatch: up to `MM_ROUTING_THRESHOLD` rows the canonical GGUF
//! route is the matvec family whose per-row arithmetic does not depend on `m`
//! (one input row per threadgroup row, unchanged accumulation order). Dense
//! float weights switch from a matvec at `m = 1` to tiled tensor kernels at
//! `m > 1`, so independent rows apply them one row at a time.

use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::quantized_matmul_ggml::{GgmlType, MM_ROUTING_THRESHOLD};
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};

use super::forward_support::raw_matmul;
use super::residency::RawMatrixRef;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RowMatmul {
    /// Rows are consecutive tokens of one sequence; any matrix route is valid.
    Matrix,
    /// Rows are independent decode requests; each row must equal `m = 1`.
    IndependentRows,
}

/// Whether independent rows must be applied one at a time for this storage.
pub(super) fn splits_independent_rows(ggml_type: GgmlType, rows: usize) -> bool {
    rows > 1 && matches!(ggml_type, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// Zero-copy view of row `row` of a row-major buffer, reshaped to `shape`.
pub(super) fn row_view(buffer: &MlxBuffer, row: usize, shape: Vec<usize>) -> Result<MlxBuffer> {
    let row_elements = shape
        .iter()
        .try_fold(1_usize, |count, &dim| count.checked_mul(dim))
        .context("DeepSeek-V4 row view shape overflow")?;
    let end = row
        .checked_add(1)
        .and_then(|rows| rows.checked_mul(row_elements))
        .context("DeepSeek-V4 row view end overflow")?;
    ensure!(
        end <= buffer.element_count(),
        "DeepSeek-V4 row {row} of {row_elements} elements exceeds buffer {:?}",
        buffer.shape()
    );
    let byte_offset = (row * row_elements)
        .checked_mul(buffer.dtype().size_of())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("DeepSeek-V4 row view byte offset overflow")?;
    buffer
        .slice_view(byte_offset, row_elements)
        .with_shape(shape)
        .context("reshape DeepSeek-V4 row view")
}

/// `raw_matmul` with explicit row semantics.
#[allow(clippy::too_many_arguments)]
pub(super) fn row_matmul(
    mode: RowMatmul,
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &RawMatrixRef<'_>,
    output: &MlxBuffer,
    m: usize,
    n: usize,
    k: usize,
    label: &str,
) -> Result<()> {
    if mode == RowMatmul::Matrix {
        return raw_matmul(
            session, registry, device, input, weight, output, m, n, k, label,
        );
    }
    if !splits_independent_rows(weight.ggml_type, m) {
        ensure!(
            m <= MM_ROUTING_THRESHOLD as usize,
            "DeepSeek-V4 {label} independent rows {m} exceed the per-row matvec route"
        );
        return raw_matmul(
            session, registry, device, input, weight, output, m, n, k, label,
        );
    }
    for row in 0..m {
        let input_row = row_view(input, row, vec![1, k])?;
        let output_row = row_view(output, row, vec![1, n])?;
        raw_matmul(
            session,
            registry,
            device,
            &input_row,
            weight,
            &output_row,
            1,
            n,
            k,
            label,
        )?;
    }
    Ok(())
}
