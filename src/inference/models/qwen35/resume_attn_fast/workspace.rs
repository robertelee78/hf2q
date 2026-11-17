//! Forward-owned, queue-ordered storage. Views use tight logical head strides.
use anyhow::{anyhow, ensure, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

#[derive(Clone)]
pub(crate) struct ResumeTqWorkspace {
    rows: u32,
    history: u32,
    heads: u32,
    kv_heads: u32,
    dim: u32,
    staging: MlxBuffer,
    k_bf16: MlxBuffer,
    v_bf16: MlxBuffer,
    out_seq: MlxBuffer,
}

impl ResumeTqWorkspace {
    pub(crate) fn new(
        device: &MlxDevice,
        rows: u32,
        history: u32,
        heads: u32,
        kv_heads: u32,
        dim: u32,
        output: Option<&MlxBuffer>,
    ) -> Result<Self> {
        ensure!(rows > 0 && history > 0 && heads > 0 && kv_heads > 0 && dim > 0,
            "resumed attention workspace needs nonzero dimensions");
        ensure!(heads % kv_heads == 0, "resumed attention workspace has invalid GQA");
        let elements = |a: u32, b: u32, c: u32| {
            (a as usize).checked_mul(b as usize).and_then(|n| n.checked_mul(c as usize))
                .ok_or_else(|| anyhow!("resumed attention workspace shape overflow"))
        };
        let kv = elements(history, kv_heads, dim)?;
        let q = elements(rows, heads, dim)?;
        if let Some(output) = output {
            ensure!(
                output.dtype() == DType::F32
                    && output.shape() == &[rows as usize, heads as usize, dim as usize],
                "resumed attention workspace output shape or dtype mismatch"
            );
        }
        let alloc = |elements: usize, dtype: DType, shape, label| {
            let bytes = elements.checked_mul(dtype.size_of())
                .ok_or_else(|| anyhow!("resumed attention workspace byte overflow"))?;
            device.alloc_buffer(bytes, dtype, shape)
                .map_err(|e| anyhow!("resumed attention workspace {label}: {e}"))
        };
        Ok(Self {
            rows, history, heads, kv_heads, dim,
            staging: alloc(kv, DType::F32, vec![kv_heads as usize, history as usize, dim as usize], "staging")?,
            k_bf16: alloc(kv, DType::BF16, vec![1, kv_heads as usize, history as usize, dim as usize], "K")?,
            v_bf16: alloc(kv, DType::BF16, vec![1, kv_heads as usize, history as usize, dim as usize], "V")?,
            out_seq: match output {
                Some(output) => output.clone(),
                None => alloc(q, DType::F32, vec![rows as usize, heads as usize, dim as usize], "output")?,
            },
        })
    }
}

/// Segment-specific shapes, backed by either private or forward-shared roots.
pub(crate) struct ResumeTqScratch {
    pub(super) seq_len: u32,
    pub(super) kv_seq_len: u32,
    pub(super) staging: MlxBuffer,
    pub(super) k_bf16: MlxBuffer,
    pub(super) v_bf16: MlxBuffer,
    pub(super) out_seq: MlxBuffer,
}

impl ResumeTqScratch {
    pub(super) fn new(
        device: &MlxDevice,
        seq_len: u32,
        kv_seq_len: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        shared: Option<&ResumeTqWorkspace>,
    ) -> Result<Self> {
        let private;
        let workspace = match shared {
            Some(workspace) => workspace,
            None => {
                private = ResumeTqWorkspace::new(device, seq_len, kv_seq_len, n_heads, n_kv_heads, head_dim, None)?;
                &private
            }
        };
        ensure!(seq_len <= workspace.rows && kv_seq_len <= workspace.history
            && n_heads == workspace.heads && n_kv_heads == workspace.kv_heads && head_dim == workspace.dim,
            "resume shared workspace does not cover the segment shape");
        let (r, k, nh, nkv, d) = (seq_len as usize, kv_seq_len as usize,
            n_heads as usize, n_kv_heads as usize, head_dim as usize);
        let view = |buffer: &MlxBuffer, shape: Vec<usize>| {
            buffer.slice_view(0, shape.iter().product()).with_shape(shape)
                .map_err(|e| anyhow!("resumed attention workspace view: {e}"))
        };
        Ok(Self {
            seq_len, kv_seq_len,
            staging: view(&workspace.staging, vec![nkv, k, d])?,
            k_bf16: view(&workspace.k_bf16, vec![1, nkv, k, d])?,
            v_bf16: view(&workspace.v_bf16, vec![1, nkv, k, d])?,
            out_seq: view(&workspace.out_seq, vec![r, nh, d])?,
        })
    }
}
