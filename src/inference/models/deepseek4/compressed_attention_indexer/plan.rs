//! Checked requested payload for one existing ratio-four indexer arena.
//! Pool bucket bytes are a separate projection, never total physical memory.
use super::selection::SelectionRequest;
use anyhow::{ensure, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::super::forward_support::alloc;
use super::super::Deepseek4Config;

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;

pub(crate) struct IndexerBufferPlan {
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
    pub(crate) label: &'static str,
    pub(crate) requested_bytes: usize,
    pub(crate) pooled_bucket_bytes: usize,
}

impl IndexerBufferPlan {
    fn new(dtype: DType, shape: Vec<usize>, label: &'static str) -> Result<Self> {
        let elements = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .with_context(|| format!("DeepSeek-V4 {label} shape overflow"))?;
        let requested_bytes = elements
            .checked_mul(dtype.size_of())
            .with_context(|| format!("DeepSeek-V4 {label} byte size overflow"))?;
        ensure!(requested_bytes > 0, "DeepSeek-V4 {label} has zero payload");
        // Published mlx-native 0.15.2 legacy pool bucket_size: next power of
        // two, minimum one. Inactive phases still allocate exact bytes.
        let pooled_bucket_bytes = requested_bytes
            .checked_next_power_of_two()
            .with_context(|| format!("DeepSeek-V4 {label} pool bucket overflow"))?;
        Ok(Self {
            dtype,
            shape,
            label,
            requested_bytes,
            pooled_bucket_bytes,
        })
    }

    pub(crate) fn allocate(self, device: &MlxDevice) -> Result<MlxBuffer> {
        self.allocate_with(|dtype, shape, label| alloc(device, dtype, shape, label))
    }
    pub(crate) fn allocate_with<T>(
        self,
        allocate: impl FnOnce(DType, Vec<usize>, &'static str) -> Result<T>,
    ) -> Result<T> {
        debug_assert!(self.requested_bytes <= self.pooled_bucket_bytes);
        allocate(self.dtype, self.shape, self.label)
    }
}

pub(crate) struct RatioFourIndexerPlan {
    pub(crate) buffers: [IndexerBufferPlan; 10],
    pub(crate) requested_bytes: usize,
    pub(crate) pooled_bucket_bytes: usize,
}

fn totals(buffers: &[IndexerBufferPlan]) -> Result<(usize, usize)> {
    let mut requested = 0usize;
    let mut buckets = 0usize;
    for buffer in buffers {
        requested = requested
            .checked_add(buffer.requested_bytes)
            .context("DeepSeek-V4 indexer requested payload sum overflow")?;
        buckets = buckets
            .checked_add(buffer.pooled_bucket_bytes)
            .context("DeepSeek-V4 indexer pool bucket sum overflow")?;
    }
    Ok((requested, buckets))
}

impl RatioFourIndexerPlan {
    /// Checked shapes for the actual immutable selection request; no context
    /// admission policy or smaller transaction is introduced.
    pub(crate) fn new(
        cfg: &Deepseek4Config,
        rows: usize,
        compressed_output_count: usize,
        selection: &SelectionRequest,
    ) -> Result<Self> {
        let heads = cfg.index_num_heads as usize;
        let dim = cfg.index_head_dim as usize;
        ensure!(
            rows > 0 && heads > 0 && dim > 0,
            "DeepSeek-V4 indexer dimensions must be nonzero"
        );
        let projected = dim
            .checked_mul(2)
            .context("DeepSeek-V4 indexer projected width overflow")?;
        let output_rows = compressed_output_count.max(1);
        ensure!(
            selection.params.query_len as usize == rows
                && selection.params.heads as usize == heads
                && selection.params.head_dim as usize == dim,
            "DeepSeek-V4 indexer plan/request geometry mismatch"
        );
        let [score, topk] = workspace_buffers(selection)?;
        let buffers = [
            IndexerBufferPlan::new(DType::F32, vec![1, rows, heads, dim], "index query f32")?,
            IndexerBufferPlan::new(DType::BF16, vec![1, rows, heads, dim], "index query bf16")?,
            IndexerBufferPlan::new(DType::F32, vec![1, rows, heads], "index weights")?,
            IndexerBufferPlan::new(DType::F32, vec![1, rows, heads], "scaled index weights")?,
            IndexerBufferPlan::new(DType::F32, vec![1, rows, projected], "index compressor KV")?,
            IndexerBufferPlan::new(
                DType::F32,
                vec![1, rows, projected],
                "index compressor score",
            )?,
            IndexerBufferPlan::new(
                DType::BF16,
                vec![1, output_rows, dim],
                "index compressor output",
            )?,
            IndexerBufferPlan::new(
                DType::BF16,
                vec![1, output_rows, 1, dim],
                "index compressor rotated output",
            )?,
            score,
            topk,
        ];
        let (requested_bytes, pooled_bucket_bytes) = totals(&buffers)?;
        Ok(Self {
            buffers,
            requested_bytes,
            pooled_bucket_bytes,
        })
    }

    pub(crate) fn allocate_with<T>(
        self,
        mut allocate: impl FnMut(IndexerBufferPlan) -> Result<T>,
    ) -> Result<[T; 10]> {
        debug_assert!(self.requested_bytes <= self.pooled_bucket_bytes);
        let [a, b, c, d, e, f, g, h, i, j] = self.buffers;
        Ok([
            allocate(a)?,
            allocate(b)?,
            allocate(c)?,
            allocate(d)?,
            allocate(e)?,
            allocate(f)?,
            allocate(g)?,
            allocate(h)?,
            allocate(i)?,
            allocate(j)?,
        ])
    }
}

pub(crate) fn workspace_buffers(request: &SelectionRequest) -> Result<[IndexerBufferPlan; 2]> {
    let (score, topk) = request.workspace_shapes()?;
    Ok([
        IndexerBufferPlan::new(DType::F32, score, "index score scratch")?,
        IndexerBufferPlan::new(DType::I32, topk, "index top-k scratch")?,
    ])
}
