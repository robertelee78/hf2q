//! Actual ratio-four selection request and postprojection production boundary.
use super::super::Deepseek4Config;
use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_indexer::{
    deepseek_indexer_query_tile_plan, deepseek_indexer_topk_scratch_elements,
    dispatch_deepseek_indexer_into_mma_tiled, DeepSeekIndexerParams, DeepSeekIndexerQueryTilePlan,
};
use mlx_native::{KernelRegistry, MlxBuffer, MlxDevice};

pub(crate) struct SelectionRequest {
    pub(crate) params: DeepSeekIndexerParams,
    pub(crate) stride: usize,
    pub(crate) column: usize,
    pub(crate) native: Option<DeepSeekIndexerQueryTilePlan>,
}
impl SelectionRequest {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        cfg: &Deepseek4Config,
        rows: usize,
        position: usize,
        valid: usize,
        bias: usize,
        stride: usize,
        column: usize,
    ) -> Result<Self> {
        let params = DeepSeekIndexerParams {
            batch: 1,
            query_len: u32::try_from(rows).context("DeepSeek-V4 indexer rows exceed u32")?,
            kv_len: u32::try_from(valid).context("DeepSeek-V4 indexer keys exceed u32")?,
            start_pos: u32::try_from(position)
                .context("DeepSeek-V4 indexer position exceeds u32")?,
            ratio: 4,
            heads: cfg.index_num_heads,
            head_dim: cfg.index_head_dim,
            top_k: cfg.index_top_k,
            offset: i32::try_from(bias)
                .context("DeepSeek-V4 indexer attention offset exceeds i32")?,
        };
        ensure!(
            rows > 0 && params.heads == 64 && params.head_dim == 128 && params.top_k == 512,
            "DeepSeek-V4 unsupported indexer selection geometry"
        );
        params
            .start_pos
            .checked_add(params.query_len)
            .context("DeepSeek-V4 indexer end exceeds u32")?;
        ensure!(
            column.checked_add(512).is_some_and(|end| end <= stride)
                && u32::try_from(stride).is_ok()
                && u32::try_from(column).is_ok(),
            "DeepSeek-V4 invalid indexer output layout"
        );
        let native = if valid == 0 {
            None
        } else {
            Some(deepseek_indexer_query_tile_plan(&params, stride, column)?)
        };
        Ok(Self {
            params,
            stride,
            column,
            native,
        })
    }
    pub(crate) fn workspace_shapes(&self) -> Result<(Vec<usize>, Vec<usize>)> {
        let rows = self.params.query_len as usize;
        match &self.native {
            Some(p) => Ok((
                vec![1, p.workspace_rows, self.params.kv_len as usize],
                vec![1, p.workspace_rows, p.topk_row_elements],
            )),
            // Preserve the previous unused minimum storage, without a fake dispatch.
            None => {
                let n = deepseek_indexer_topk_scratch_elements(1, rows, 1)?;
                Ok((vec![1, rows, 1], vec![1, rows, n / rows]))
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_selection(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    request: &SelectionRequest,
    query: &MlxBuffer,
    cache: &MlxBuffer,
    weights: &MlxBuffer,
    scores: &MlxBuffer,
    topk: &MlxBuffer,
    output: &MlxBuffer,
) -> Result<()> {
    let (score_shape, topk_shape) = request.workspace_shapes()?;
    ensure!(
        scores.shape() == score_shape && topk.shape() == topk_shape,
        "DeepSeek-V4 indexer allocated workspace/request mismatch"
    );
    if request.native.is_none() {
        return Ok(());
    }
    let valid = request.params.kv_len as usize;
    let cache_elements = valid
        .checked_mul(request.params.head_dim as usize)
        .context("DeepSeek-V4 indexer cache view overflow")?;
    ensure!(
        cache.data_byte_len()
            >= cache_elements
                .checked_mul(2)
                .context("indexer cache bytes overflow")?,
        "DeepSeek-V4 indexer valid cache exceeds backing"
    );
    let cache_view = cache.slice_view(0, cache_elements).with_shape(vec![
        1,
        valid,
        request.params.head_dim as usize,
    ])?;
    session.barrier_between(&[query, &cache_view, weights], &[scores, output]);
    dispatch_deepseek_indexer_into_mma_tiled(
        session.encoder_mut(),
        registry,
        device,
        query,
        &cache_view,
        weights,
        scores,
        topk,
        output,
        request.stride,
        request.column,
        &request.params,
    )?;
    Ok(())
}
