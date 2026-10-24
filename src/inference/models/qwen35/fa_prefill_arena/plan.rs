//! Requested bytes for the existing seven direct FA bridge allocations.
//! This is a constructor plan, not a physical memory bound or admission cap.
use anyhow::{anyhow, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;

pub(crate) struct FaBufferPlan {
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
    pub(crate) label: &'static str,
    pub(crate) requested_bytes: usize,
}

impl FaBufferPlan {
    fn new(shape: Vec<usize>, label: &'static str) -> Result<Self> {
        let elements = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .with_context(|| format!("FaPrefillArena::new {label} shape overflow"))?;
        let requested_bytes = elements
            .checked_mul(DType::BF16.size_of())
            .with_context(|| format!("FaPrefillArena::new {label} byte size overflow"))?;
        Ok(Self {
            dtype: DType::BF16,
            shape,
            label,
            requested_bytes,
        })
    }

    pub(crate) fn allocate(self, device: &MlxDevice) -> Result<MlxBuffer> {
        device
            .alloc_buffer(self.requested_bytes, self.dtype, self.shape)
            .map_err(|e| anyhow!("FaPrefillArena::new alloc {}: {e}", self.label))
    }
}

pub(crate) struct FaPrefillPlan {
    pub(crate) buffers: [FaBufferPlan; 7],
    pub(crate) requested_bytes: usize,
}

impl FaPrefillPlan {
    /// Preserve the constructor's zero-dimension then GQA validation order.
    /// No new chunk, context, alignment or head-width policy is selected.
    pub(crate) fn new(
        seq_capacity: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<Self> {
        if seq_capacity == 0 || n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
            return Err(anyhow!(
                "FaPrefillArena::new: zero dim \
                 seq_capacity={} n_heads={} n_kv_heads={} head_dim={}",
                seq_capacity,
                n_heads,
                n_kv_heads,
                head_dim
            ));
        }
        if n_heads % n_kv_heads != 0 {
            return Err(anyhow!(
                "FaPrefillArena::new: n_heads ({}) must be divisible by n_kv_heads ({})",
                n_heads,
                n_kv_heads
            ));
        }
        let seq = seq_capacity as usize;
        let nh = n_heads as usize;
        let nkv = n_kv_heads as usize;
        let d = head_dim as usize;
        let buffers = [
            FaBufferPlan::new(vec![seq, nh, d], "q_bf16_seq")?,
            FaBufferPlan::new(vec![1, nh, seq, d], "q_bf16_hm")?,
            FaBufferPlan::new(vec![seq, nkv, d], "k_bf16_seq")?,
            FaBufferPlan::new(vec![1, nkv, seq, d], "k_bf16_hm")?,
            FaBufferPlan::new(vec![seq, nkv, d], "v_bf16_seq")?,
            FaBufferPlan::new(vec![1, nkv, seq, d], "v_bf16_hm")?,
            FaBufferPlan::new(vec![1, nh, seq, d], "out_bf16_hm")?,
        ];
        let requested_bytes = buffers
            .iter()
            .try_fold(0usize, |sum, buffer| {
                sum.checked_add(buffer.requested_bytes)
            })
            .context("FaPrefillArena::new requested payload sum overflow")?;
        Ok(Self {
            buffers,
            requested_bytes,
        })
    }

    pub(crate) fn into_buffers(self) -> [FaBufferPlan; 7] {
        debug_assert!(self.requested_bytes > 0);
        self.buffers
    }
}
