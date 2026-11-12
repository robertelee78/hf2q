//! The live-tile MoE FFN layer (`HF2Q_QWEN_MOE_LIVE_TILES=1`, on top of
//! `HF2Q_QWEN_MOE_BATCHED=1` and `HF2Q_QWEN_MOE_STAGED_M8=1`) on the fused
//! expert stages of `mlx_native::moe_tiles_fused`.
//!
//! One MoE FFN layer over `rows = 1..=32` decode or DFlash2 target rows:
//!
//! | phase | dispatch | kernel |
//! |---|---|---|
//! | A | router logits, shared gate logit: every full block of eight rows in one dispatch each | `dense_matvec_f32_rows` |
//! | B | softmax + top-k + renorm | `moe_softmax_topk_simd_f32` |
//! | C | DFlash2 verify-row mask, one schedule of routed and shared tiles, pass grids | `moe_tiles_fused_schedule` |
//! | D | gather of every tile's rows | `moe_tiles_fused_gather_f32` |
//! | E | gate and up in one pass, routed and shared tiles in one grid | `moe_tiles_fused_gate_up_<q>` |
//! | F | down over those rows | `moe_tiles_fused_down_<q>` |
//! | G | weighted reduce + sigmoid(shared gate) * shared + residual | `moe_weighted_reduce[_masked]_f32` |
//!
//! Eight dispatches and six barriers at a width that is a multiple of eight
//! rows. A row the verify-row mask removed joins no routed tile; the router
//! and the shared expert run over every row of the step.
//!
//! Every dispatch computes a row from that row alone with the arithmetic of
//! the stage-by-stage form in [`super::moe_live_tiles`]: the row-exact router
//! kernel per block of eight rows, the tile arithmetic and register-path
//! rule of the `moe_tiles_*` stages. The layer's output, expert ids and
//! routing weights are bitwise that form's, and with them its row
//! independence (`moe_live_tiles_tests`).
//!
//! The fused stages need the routed and the shared expert's planes of each
//! projection in one format and one expert width, and at most 256 experts
//! and 256 routes a step. A layer outside that runs the stage-by-stage form.

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::dense_matvec_f32_rows::{
    dispatch_dense_matvec_f32_row_blocks, dispatch_dense_matvec_f32_rows,
};
use mlx_native::ops::moe_softmax_topk::dispatch_moe_softmax_topk_simd;
use mlx_native::ops::moe_weighted_reduce::{
    dispatch_moe_weighted_reduce, dispatch_moe_weighted_reduce_masked,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{
    moe_tiles_fused, moe_tiles_fused_admits, moe_tiles_fused_grouped_rows,
    moe_tiles_fused_output_rows, moe_tiles_fused_schedule_words, CommandEncoder, DType,
    KernelRegistry, MlxBuffer, MlxDevice, MoeTilesFusedBuffers, MoeTilesFusedMask,
    MoeTilesFusedParams, MoeTilesFusedStage, MoeTilesFusedWeights, MOE_TILES_FUSED_GRID_WORDS,
};

use super::ffn::MoeFfnShape;
use super::gpu_ffn::MoeFfnWeightsGpuQ;
use super::moe_batched::{
    pooled, DFlash2MoeRouteMask, MoeBatchedRouting, MoeDenseRole, QWEN_MOE_LIVE_TILES_ENV,
};
use super::moe_live_tiles::{build_moe_ffn_layer_tile_stages_into, row_block, BLOCK_ROWS};

/// Encode `output[rows, out] = input[rows, in] @ W^T` for the F32 weight of
/// `role` on the row-exact kernel: the full blocks of [`BLOCK_ROWS`] rows in
/// one dispatch, then the rows left over.
#[allow(clippy::too_many_arguments)]
fn encode_row_exact_blocks(
    enc: &mut CommandEncoder,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weights: &MoeFfnWeightsGpuQ,
    role: MoeDenseRole,
    output: &MlxBuffer,
    rows: u32,
    in_features: u32,
    out_features: u32,
) -> Result<()> {
    let weight = role.weight(weights);
    ensure!(
        weight.dtype() == DType::F32,
        "{role:?}: native router weights must be F32, got {:?}",
        weight.dtype()
    );
    let blocks = rows / BLOCK_ROWS;
    if blocks > 0 {
        dispatch_dense_matvec_f32_row_blocks(
            enc,
            registry,
            device,
            input,
            weight,
            output,
            BLOCK_ROWS,
            blocks,
            out_features,
            in_features,
        )
        .with_context(|| format!("{role:?} dense_matvec_f32_rows blocks"))?;
    }
    let rest = rows % BLOCK_ROWS;
    if rest > 0 {
        let start = blocks * BLOCK_ROWS;
        dispatch_dense_matvec_f32_rows(
            enc,
            registry,
            device,
            &row_block(input, start, rest, in_features),
            weight,
            &row_block(output, start, rest, out_features),
            rest,
            out_features,
            in_features,
        )
        .with_context(|| format!("{role:?} dense_matvec_f32_rows"))?;
    }
    Ok(())
}

/// Encode one MoE FFN layer over `rows` packed rows into `out`
/// (`[rows, hidden]`), adding `residual` when given. The caller validated
/// the row count and the layer's native storage. Does not commit.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_moe_ffn_layer_live_tiles_into(
    enc: &mut CommandEncoder,
    device: &MlxDevice,
    registry: &mut KernelRegistry,
    x: &MlxBuffer,
    weights: &MoeFfnWeightsGpuQ,
    shape: MoeFfnShape,
    residual: Option<&MlxBuffer>,
    out: &mut MlxBuffer,
    rows: u32,
    layer_idx: usize,
    route_mask: Option<DFlash2MoeRouteMask<'_>>,
) -> Result<MoeBatchedRouting> {
    let quantized = |t: GgmlType| matches!(t, GgmlType::Q4_K | GgmlType::Q6_K);
    ensure!(
        matches!(shape.num_experts_per_tok, 1 | 6 | 8)
            && shape.hidden_size % 256 == 0
            && shape.moe_intermediate_size % 256 == 0
            && shape.shared_intermediate_size % 256 == 0
            && quantized(weights.ggml_type_gate_up)
            && quantized(weights.ggml_type_down)
            && !crate::quantize::imatrix::is_active(),
        "{QWEN_MOE_LIVE_TILES_ENV}=1 requires top-k 1/6/8, expert widths that are multiples of \
         256, Q4_K/Q6_K native expert stacks and no imatrix collector"
    );
    let prepared = weights
        .prepared_m8
        .as_ref()
        .context("live-tile expert planes were not prepared at model upload")?;
    let live = prepared
        .live_tiles
        .as_ref()
        .context("the shared expert's live-tile planes were not prepared at model upload")?;
    let params = MoeTilesFusedParams {
        n_tokens: rows,
        top_k: shape.num_experts_per_tok,
        n_experts: shape.num_experts,
        hidden: shape.hidden_size,
        intermediate: shape.moe_intermediate_size,
        column_tile: live.column_tile,
    };
    let planes = MoeTilesFusedWeights {
        gate: &prepared.gate,
        up: &prepared.up,
        down: &prepared.down,
        shared_gate: &live.shared[0],
        shared_up: &live.shared[1],
        shared_down: &live.shared[2],
    };
    if moe_tiles_fused_admits(&planes, &params).is_err() {
        return build_moe_ffn_layer_tile_stages_into(
            enc, device, registry, x, weights, shape, residual, out, rows, layer_idx, route_mask,
        );
    }
    let context = || format!("live-tile MoE layer {layer_idx}");
    let r = rows as usize;
    let h = shape.hidden_size as usize;
    let m = shape.moe_intermediate_size as usize;
    let top_k = shape.num_experts_per_tok as usize;
    let ne = shape.num_experts;
    ensure!(
        x.element_count() >= r * h,
        "live-tile MoE layer {layer_idx}: input has {} values, need {rows} x {h}",
        x.element_count()
    );
    if let Some(mask) = route_mask {
        ensure!(
            mask.params.width == rows
                && mask.params.top_k == shape.num_experts_per_tok
                && mask.params.n_experts == ne,
            "{}: the route mask covers {} rows of {} routes over {} experts",
            context(),
            mask.params.width,
            mask.params.top_k,
            mask.params.n_experts
        );
    }
    let pairs = r * top_k;
    let tile_error = |e: mlx_native::MlxError| anyhow!("{} tiles: {e}", context());
    let grouped = moe_tiles_fused_grouped_rows(&params).map_err(tile_error)?;
    let schedule_words = moe_tiles_fused_schedule_words(&params).map_err(tile_error)?;
    let output_rows = moe_tiles_fused_output_rows(&params).map_err(tile_error)?;

    let logits = pooled(
        device,
        r * ne as usize,
        DType::F32,
        vec![r, ne as usize],
        "router logits",
    )?;
    let sh_logit = pooled(device, r, DType::F32, vec![r, 1], "shared gate logit")?;
    let ids = pooled(device, pairs, DType::U32, vec![pairs], "expert ids")?;
    let routing_weights = pooled(device, pairs, DType::F32, vec![pairs], "routing weights")?;
    // Every route's down row, then the shared expert's row of every token.
    let y = pooled(
        device,
        output_rows * h,
        DType::F32,
        vec![output_rows, h],
        "expert and shared down",
    )?;
    let y_shared = y.slice_view((pairs * h * DType::F32.size_of()) as u64, r * h);
    let schedule = pooled(
        device,
        schedule_words,
        DType::U32,
        vec![schedule_words],
        "tile schedule",
    )?;
    let routes = pooled(device, grouped, DType::U32, vec![grouped], "tile routes")?;
    let grids = pooled(
        device,
        MOE_TILES_FUSED_GRID_WORDS,
        DType::U32,
        vec![MOE_TILES_FUSED_GRID_WORDS],
        "tile grids",
    )?;
    let gathered = pooled(
        device,
        grouped * h,
        DType::BF16,
        vec![grouped, h],
        "tile rows",
    )?;
    let intermediate = pooled(
        device,
        grouped * m,
        DType::BF16,
        vec![grouped, m],
        "tile SwiGLU rows",
    )?;
    let dummy_residual;
    let residual_ref = match residual {
        Some(buffer) => buffer,
        None => {
            dummy_residual = pooled(device, 1, DType::F32, vec![1], "dummy residual")?;
            &dummy_residual
        }
    };
    let h32 = shape.hidden_size;

    // ---- A: router, shared gate logit ----
    for (role, output, out_features) in [
        (MoeDenseRole::Router, &logits, ne),
        (MoeDenseRole::SharedGateLogit, &sh_logit, 1),
    ] {
        encode_row_exact_blocks(
            enc,
            registry,
            device,
            x,
            weights,
            role,
            output,
            rows,
            h32,
            out_features,
        )
        .with_context(context)?;
    }
    enc.memory_barrier();

    // ---- B: routing over every row ----
    dispatch_moe_softmax_topk_simd(
        enc,
        registry,
        device,
        &logits,
        &ids,
        &routing_weights,
        rows,
        ne,
        shape.num_experts_per_tok,
    )
    .map_err(|e| anyhow!("{} top-k: {e}", context()))?;
    enc.memory_barrier();

    // ---- C-F: mask and schedule, gather, gate/up, down ----
    let mask = route_mask.map(|mask| MoeTilesFusedMask {
        verify_rows: mask.verify_rows,
        physical_lane: mask.params.physical_lane,
        lanes: mask.params.lanes,
        rows_per_lane: mask.params.rows_per_lane,
    });
    moe_tiles_fused(
        enc,
        registry,
        device,
        x,
        &ids,
        &routing_weights,
        mask.as_ref(),
        &planes,
        &MoeTilesFusedBuffers {
            schedule: &schedule,
            routes: &routes,
            grids: &grids,
            gathered: &gathered,
            intermediate: &intermediate,
        },
        &y,
        &params,
        MoeTilesFusedStage::Down,
    )
    .map_err(tile_error)?;
    enc.memory_barrier();

    // ---- G: weighted reduce + sigmoid(shared gate) * shared + residual ----
    // The down pass leaves masked routes' rows unwritten.
    let weighted_reduce = if route_mask.is_some() {
        dispatch_moe_weighted_reduce_masked
    } else {
        dispatch_moe_weighted_reduce
    };
    weighted_reduce(
        enc,
        registry,
        device,
        &routing_weights,
        &y,
        &sh_logit,
        &y_shared,
        residual_ref,
        out,
        rows,
        shape.num_experts_per_tok,
        h32,
        residual.is_some(),
    )
    .map_err(|e| anyhow!("{} weighted reduce: {e}", context()))?;

    #[cfg(test)]
    super::moe_batched::retain_routing_for_tests(
        &logits,
        &ids,
        &routing_weights,
        r,
        ne as usize,
        top_k,
    )?;

    Ok(MoeBatchedRouting {
        ids,
        weights: routing_weights,
    })
}
