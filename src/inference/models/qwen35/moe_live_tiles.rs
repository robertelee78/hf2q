//! The live-tile MoE FFN layer (`HF2Q_QWEN_MOE_LIVE_TILES=1`, on top of
//! `HF2Q_QWEN_MOE_BATCHED=1` and `HF2Q_QWEN_MOE_STAGED_M8=1`) stage by stage:
//! the form [`super::moe_live_tiles_fused`] falls back to for a layer its
//! fused expert stages do not admit, and the reference its bitwise tests
//! compare it with.
//!
//! One MoE FFN layer over `rows = 1..=32` decode or DFlash2 target rows:
//!
//! | phase | dispatch | kernel |
//! |---|---|---|
//! | A | router logits, shared gate logit, per block of up to eight rows | `dense_matvec_f32_rows` |
//! | A | shared expert's schedule (a tile per block of eight rows) | `kernel_mul_mm_id_map0c_ne20_1` |
//! | B | softmax + top-k + renorm; shared expert's gather | `moe_softmax_topk_simd_f32`, `moe_tiles_gather_f32` |
//! | B | DFlash2 verify-row mask (when given) | `dflash2_mask_moe_routes` |
//! | C | route schedule; gather of the scheduled rows and the pass grids | `kernel_mul_mm_id_map0c_ne20_<k>`, `moe_tiles_gather_f32` |
//! | D | gate and up in one pass, BF16 SwiGLU left in the schedule's rows: routed experts, shared expert | `moe_tiles_gate_up_<q>` |
//! | E | down over those rows: routed experts, shared expert | `moe_tiles_down_<q>` |
//! | F | weighted reduce + sigmoid(shared gate) * shared + residual | `moe_weighted_reduce[_masked]_f32` |
//!
//! The routed passes launch the schedule's live tiles alone (an indirect
//! grid), so a row the verify-row mask removed costs no routed-expert tile,
//! and neither its input nor its expert rows are read by a live row. The
//! router and the shared expert run over every row of the step, as before.
//!
//! Row independence. Every dispatch computes a row from that row alone, in an
//! order that does not depend on the step: the router and shared-logit
//! kernel is the row-exact one decode runs at up to eight rows, so a wider
//! step runs it once per block of eight; the expert tiles, routed and shared,
//! are row-independent by construction (`mlx_native::moe_tiles_*`). A row of
//! a width-32 verify step is therefore bitwise the row this route decodes
//! alone, whichever other rows and lanes share its step. The one exception
//! is an expert product that underflows FP32, which the register path and
//! the matmul tile may round differently (results of magnitude at most
//! 4e-31 in review).

use anyhow::{anyhow, ensure, Context, Result};
use mlx_native::ops::dense_matvec_f32_rows::DENSE_MATVEC_F32_ROWS_MAX_ROWS;
use mlx_native::ops::dflash2_attention_pack::dispatch_dflash2_mask_moe_routes;
use mlx_native::ops::moe_softmax_topk::dispatch_moe_softmax_topk_simd;
use mlx_native::ops::moe_weighted_reduce::{
    dispatch_moe_weighted_reduce, dispatch_moe_weighted_reduce_masked,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{
    moe_tiles_down, moe_tiles_gate_up, moe_tiles_gather, moe_tiles_grouped_rows,
    moe_tiles_schedule, CommandEncoder, DType, KernelRegistry, MlxBuffer, MlxDevice,
    MoeTilesBuffers, MoeTilesParams, MOE_TILES_GRID_WORDS,
};

use super::decode_pool::{with_id_mm_scratch, MmIdSlot};
use super::ffn::MoeFfnShape;
use super::gpu_ffn::MoeFfnWeightsGpuQ;
use super::moe_batched::{
    encode_native_dense_proj, pooled, DFlash2MoeRouteMask, MoeBatchedRouting, MoeDenseRole,
    QWEN_MOE_LIVE_TILES_ENV,
};

/// Rows of one router dispatch (the widest step of the row-exact kernel) and
/// of one shared-expert tile.
pub(super) const BLOCK_ROWS: u32 = DENSE_MATVEC_F32_ROWS_MAX_ROWS;

/// One tile step's scratch from the decode pool.
struct TileScratch {
    gathered: MlxBuffer,
    intermediate: MlxBuffer,
    grids: MlxBuffer,
}

impl TileScratch {
    fn pooled(device: &MlxDevice, params: &MoeTilesParams, label: &str) -> Result<Self> {
        let rows = moe_tiles_grouped_rows(params.n_tokens, params.top_k, params.n_experts)
            .map_err(|e| anyhow!("{label} tile rows: {e}"))?;
        let (h, m) = (params.hidden as usize, params.intermediate as usize);
        Ok(Self {
            gathered: pooled(device, rows * h, DType::BF16, vec![rows, h], label)?,
            intermediate: pooled(device, rows * m, DType::BF16, vec![rows, m], label)?,
            grids: pooled(
                device,
                MOE_TILES_GRID_WORDS,
                DType::U32,
                vec![MOE_TILES_GRID_WORDS],
                label,
            )?,
        })
    }

    fn buffers(&self) -> MoeTilesBuffers<'_> {
        MoeTilesBuffers {
            gathered: &self.gathered,
            intermediate: &self.intermediate,
            grids: &self.grids,
        }
    }
}

/// Rows `start..start + count` of a `[rows, width]` F32 buffer.
pub(super) fn row_block(buffer: &MlxBuffer, start: u32, count: u32, width: u32) -> MlxBuffer {
    buffer.slice_view(
        u64::from(start) * u64::from(width) * DType::F32.size_of() as u64,
        count as usize * width as usize,
    )
}

/// Encode `output[rows, out] = input[rows, in] @ W^T` for `role`, one
/// dispatch per block of [`BLOCK_ROWS`] rows.
#[allow(clippy::too_many_arguments)]
fn encode_row_exact_proj(
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
    for start in (0..rows).step_by(BLOCK_ROWS as usize) {
        let count = BLOCK_ROWS.min(rows - start);
        encode_native_dense_proj(
            enc,
            registry,
            device,
            &row_block(input, start, count, in_features),
            weights,
            role,
            &row_block(output, start, count, out_features),
            count,
            in_features,
            out_features,
        )?;
    }
    Ok(())
}

/// Encode one MoE FFN layer over `rows` packed rows into `out`
/// (`[rows, hidden]`), adding `residual` when given. The caller validated
/// the row count and the layer's native storage. Does not commit.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_moe_ffn_layer_tile_stages_into(
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
    let context = || format!("live-tile MoE layer {layer_idx}");
    let r = rows as usize;
    let h = shape.hidden_size as usize;
    let top_k = shape.num_experts_per_tok as usize;
    let ne = shape.num_experts;
    ensure!(
        x.element_count() >= r * h,
        "live-tile MoE layer {layer_idx}: input has {} values, need {rows} x {h}",
        x.element_count()
    );
    let pairs = r * top_k;
    let routed = MoeTilesParams {
        n_tokens: rows,
        top_k: shape.num_experts_per_tok,
        n_experts: ne,
        hidden: shape.hidden_size,
        intermediate: shape.moe_intermediate_size,
        masked_routes: route_mask.is_some(),
        column_tile: live.column_tile,
        shared_planes: false,
    };
    // The shared expert as its own step: row t routes to "expert" t / 8, so
    // each block of eight rows is one tile over the one set of planes.
    let shared_blocks = rows.div_ceil(BLOCK_ROWS);
    let shared = MoeTilesParams {
        n_tokens: rows,
        top_k: 1,
        n_experts: shared_blocks,
        hidden: shape.hidden_size,
        intermediate: shape.shared_intermediate_size,
        masked_routes: false,
        column_tile: live.column_tile,
        shared_planes: true,
    };

    let logits = pooled(
        device,
        r * ne as usize,
        DType::F32,
        vec![r, ne as usize],
        "router logits",
    )?;
    let sh_logit = pooled(device, r, DType::F32, vec![r, 1], "shared gate logit")?;
    let y_s = pooled(device, r * h, DType::F32, vec![r, h], "shared down")?;
    let ids = pooled(device, pairs, DType::U32, vec![pairs], "expert ids")?;
    let routing_weights = pooled(device, pairs, DType::F32, vec![pairs], "routing weights")?;
    let y_all = pooled(device, pairs * h, DType::F32, vec![pairs, h], "expert down")?;
    let routed_scratch = TileScratch::pooled(device, &routed, "routed expert tiles")?;
    let shared_scratch = TileScratch::pooled(device, &shared, "shared expert tiles")?;
    let mut shared_ids = pooled(device, r, DType::U32, vec![r], "shared expert blocks")?;
    for (row, block) in shared_ids
        .as_mut_slice::<u32>()
        .map_err(|e| anyhow!("{}: shared expert blocks: {e}", context()))?[..r]
        .iter_mut()
        .enumerate()
    {
        *block = row as u32 / BLOCK_ROWS;
    }
    let dummy_residual;
    let residual_ref = match residual {
        Some(buffer) => buffer,
        None => {
            dummy_residual = pooled(device, 1, DType::F32, vec![1], "dummy residual")?;
            &dummy_residual
        }
    };
    let h32 = shape.hidden_size;
    let tile_error = |stage: &str, e: mlx_native::MlxError| anyhow!("{} {stage}: {e}", context());
    // The routed and the shared schedule live in different scratch slots.
    let routed_slot =
        |f: &mut dyn FnMut(&mut mlx_native::IdMmScratch) -> mlx_native::Result<()>| {
            with_id_mm_scratch(
                MmIdSlot::Gate,
                device,
                ne,
                rows * shape.num_experts_per_tok,
                |s| f(s),
            )
        };
    let shared_slot =
        |f: &mut dyn FnMut(&mut mlx_native::IdMmScratch) -> mlx_native::Result<()>| {
            with_id_mm_scratch(MmIdSlot::Up, device, shared_blocks, rows, |s| f(s))
        };

    // ---- A: router, shared gate logit, shared expert's schedule ----
    for (role, output, out_features) in [
        (MoeDenseRole::Router, &logits, ne),
        (MoeDenseRole::SharedGateLogit, &sh_logit, 1),
    ] {
        encode_row_exact_proj(
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
    // The schedule's own barrier closes the phase.
    shared_slot(&mut |scratch| {
        moe_tiles_schedule(enc, registry, device, &shared_ids, scratch, &shared)
    })
    .map_err(|e| tile_error("shared schedule", e))?;

    // ---- B: routing over every row, shared expert's gather ----
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
    shared_slot(&mut |scratch| {
        moe_tiles_gather(
            enc,
            registry,
            device,
            x,
            &shared_ids,
            &shared_scratch.buffers(),
            scratch,
            &shared,
        )
    })
    .map_err(|e| tile_error("shared gather", e))?;
    enc.memory_barrier();
    // The mask rewrites the selector's outputs: the barrier above orders it
    // after the selector and the one below before the schedule reads the ids.
    if let Some(mask) = route_mask {
        dispatch_dflash2_mask_moe_routes(
            enc,
            registry,
            device,
            mask.verify_rows,
            &ids,
            &routing_weights,
            mask.params,
        )
        .map_err(|e| anyhow!("{} DFlash2 route mask: {e}", context()))?;
        enc.memory_barrier();
    }

    // ---- C: one route schedule, the scheduled rows of x, the pass grids ----
    routed_slot(&mut |scratch| {
        moe_tiles_schedule(enc, registry, device, &ids, scratch, &routed)?;
        moe_tiles_gather(
            enc,
            registry,
            device,
            x,
            &ids,
            &routed_scratch.buffers(),
            scratch,
            &routed,
        )
    })
    .map_err(|e| tile_error("route schedule", e))?;
    enc.memory_barrier();

    // ---- D: gate/up in one pass, routed experts and shared expert ----
    routed_slot(&mut |scratch| {
        moe_tiles_gate_up(
            enc,
            registry,
            device,
            &prepared.gate,
            &prepared.up,
            &ids,
            &routed_scratch.buffers(),
            scratch,
            &routed,
        )
    })
    .map_err(|e| tile_error("expert gate/up", e))?;
    shared_slot(&mut |scratch| {
        moe_tiles_gate_up(
            enc,
            registry,
            device,
            &live.shared[0],
            &live.shared[1],
            &shared_ids,
            &shared_scratch.buffers(),
            scratch,
            &shared,
        )
    })
    .map_err(|e| tile_error("shared gate/up", e))?;
    enc.memory_barrier();

    // ---- E: down over the schedules' SwiGLU rows ----
    routed_slot(&mut |scratch| {
        moe_tiles_down(
            enc,
            registry,
            device,
            &prepared.down,
            &ids,
            &routed_scratch.buffers(),
            scratch,
            &y_all,
            &routed,
        )
    })
    .map_err(|e| tile_error("expert down", e))?;
    shared_slot(&mut |scratch| {
        moe_tiles_down(
            enc,
            registry,
            device,
            &live.shared[2],
            &shared_ids,
            &shared_scratch.buffers(),
            scratch,
            &y_s,
            &shared,
        )
    })
    .map_err(|e| tile_error("shared down", e))?;
    enc.memory_barrier();

    // ---- F: weighted reduce + sigmoid(shared gate) * shared + residual ----
    // The routed down pass leaves masked routes' rows unwritten.
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
        &y_all,
        &sh_logit,
        &y_s,
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
