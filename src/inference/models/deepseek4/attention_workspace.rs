//! Three sequential F32 roles, one root; never simultaneous kernel operands.
//! HC projection finishes reading norm before query B writes q; query RMS
//! finishes reading q before the epilogue casts into attention_f32. Existing
//! GraphSession WAR/WAW annotations protect these aliases, including reorder.
use super::forward_support::alloc;
use anyhow::{ensure, Context, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};
#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    pub rows: usize,
    pub hc_hidden: usize,
    pub heads: usize,
    pub dim: usize,
    pub root_elements: usize,
    pub norm_elements: usize,
    pub query_elements: usize,
}
impl Plan {
    pub fn new(rows: usize, hc_hidden: usize, heads: usize, dim: usize) -> Result<Self> {
        ensure!(
            rows > 0 && hc_hidden > 0 && heads > 0 && dim > 0,
            "positive attention workspace geometry"
        );
        let width = heads
            .checked_mul(dim)
            .context("attention query width overflow")?;
        let norm_elements = rows
            .checked_mul(hc_hidden)
            .context("HC norm extent overflow")?;
        let query_elements = rows.checked_mul(width).context("query extent overflow")?;
        let root_elements = norm_elements.max(query_elements);
        root_elements
            .checked_mul(4)
            .context("attention workspace byte overflow")?;
        Ok(Self {
            rows,
            hc_hidden,
            heads,
            dim,
            root_elements,
            norm_elements,
            query_elements,
        })
    }
    #[cfg(test)]
    fn separate_bytes(self) -> Result<usize> {
        self.query_elements
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.norm_elements))
            .and_then(|n| n.checked_mul(4))
            .context("separate workspace sum overflow")
    }
}
pub(super) struct AttentionWorkspace {
    pub state_norm: MlxBuffer,
    pub q: MlxBuffer,
    pub attention_f32: MlxBuffer,
}
impl AttentionWorkspace {
    pub fn new(
        device: &MlxDevice,
        rows: usize,
        hc_hidden: usize,
        heads: usize,
        dim: usize,
    ) -> Result<Self> {
        let p = Plan::new(rows, hc_hidden, heads, dim)?;
        let root = alloc(
            device,
            DType::F32,
            vec![p.root_elements],
            "sequential HC/query/attention workspace",
        )?;
        Self::from_root(root, p)
    }
    fn from_root(root: MlxBuffer, p: Plan) -> Result<Self> {
        ensure!(
            root.dtype() == DType::F32 && root.data_byte_len() == p.root_elements * 4,
            "exact F32 workspace root extent"
        );
        let state_norm = root
            .slice_view(0, p.norm_elements)
            .with_shape(vec![p.rows, p.hc_hidden])?;
        let q = root
            .slice_view(0, p.query_elements)
            .with_shape(vec![p.rows, p.heads, p.dim])?;
        let attention_f32 = q.clone();
        ensure!(
            state_norm.contents_ptr() == q.contents_ptr()
                && q.contents_ptr() == attention_f32.contents_ptr(),
            "same-parent zero-relative-offset workspace views"
        );
        Ok(Self {
            state_norm,
            q,
            attention_f32,
        })
    }
}
#[cfg(test)]
#[path = "attention_workspace_controls.rs"]
mod controls;

#[cfg(test)]
#[path = "attention_workspace_gpu.rs"]
mod gpu;
