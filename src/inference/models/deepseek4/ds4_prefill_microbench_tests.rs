//! DeepSeek-V4 gap program P2: hf2q prefill per-op microbenchmarks at the ds4 0731 file's shapes.
//!
//! Model-free: synthetic weights (in-range block scales, random F16) and synthetic caches. Every cell
//! times hf2q's own dispatch entry point exactly as the matrix prefill path issues it for a chunk of
//! `T` rows (`RowMatmul::Matrix`, paired MoE gate/up with `IdMmScratch`, slotted down projection,
//! gathered sparse flash prefill, tiled indexer selection, the stateful compressor).
//!
//! Shapes: `T` = 256, 512, 1024, 2048 for the per-row ops (hf2q runs 512 per cooperative lane,
//! 2048 aggregate cooperative FFN rows, 256-token mixed slices; the peer's ubatch is <= 1024).
//! Attention, indexer selection and compressors run at `T` = 512 / 1024 and cache start `s0` = 0 / 8192,
//! the peer's prefill FLASH_ATTN_EXT / LIGHTNING_INDEXER cases.
//!
//! Timing: `ds4_bench_support::time_op_reps` (GPU command-buffer timestamps, >= 2 ms of GPU work per
//! session, median of 9) with a minimum of 2 copies per session. Weights use one copy: at `T >= 256`
//! every weight is reused across all rows of the chunk, and the routed-expert tensors (1.1-1.8 GB per
//! layer) exceed the system cache anyway.
//!
//! Small chunks (fix-1 design: recovery tail and short suffixes): `HF2Q_DS4_PREFILL_BENCH_ROWS` (comma list)
//! replaces both row lists and `HF2Q_DS4_PREFILL_BENCH_STARTS` the cache starts. Below `COLD_BELOW_ROWS` the
//! dense and expert cells rotate weight copies (cold, as a chunk of a real prompt reads each layer's weights
//! once), and routed experts take the production small-chunk route: separate gate and up plus down through
//! `ExpertMatmulRoute::Auto` without scratch, i.e. `mv_id` at `T <= MM_ID_ROUTING_THRESHOLD`.
//!
//! Run (standing grant, model-free, <= 15 min):
//! ```text
//! HF2Q_DS4_PREFILL_BENCH_OUT=<cells.jsonl> cargo test --release --bin hf2q ds4_prefill_microbench -- --ignored --nocapture
//! ```

use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_hadamard_mxfp4_bf16, dispatch_deepseek_mxfp8_fake_quant_bf16,
    DeepSeekMxfp8Params,
};
use mlx_native::ops::deepseek_compressor::{
    dispatch_deepseek_compressor, DeepSeekCompressorParams,
};
use mlx_native::ops::deepseek_hyper_connection::{
    dispatch_hc_post, dispatch_hc_pre, dispatch_hc_split_sinkhorn,
};
use mlx_native::ops::deepseek_indexer::{
    deepseek_indexer_query_tile_plan, dispatch_deepseek_indexer_into_mma_tiled,
    DeepSeekIndexerParams,
};
use mlx_native::ops::deepseek_moe_activation::{
    dispatch_deepseek_moe_swiglu, dispatch_deepseek_moe_weighted_reduce,
};
use mlx_native::ops::deepseek_moe_routing::{
    dispatch_deepseek_moe_sanitize_indices, dispatch_deepseek_moe_score_route,
};
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, dispatch_deepseek_tail_rope_f32_to_bf16,
    DeepSeekTailRopeParams,
};
use mlx_native::ops::elementwise::dispatch_cast_bf16_to_f32_with_encoder;
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, IdMmScratch, KernelRegistry, MlxBuffer, MlxDevice, MM_ID_ROUTING_THRESHOLD};
use serde_json::{json, Value};

use super::attention::{append_prefill_window_indices, window_indices};
use super::compressed_attention_indexer::layout::prepare_layout;
use super::ds4_bench_support::{
    copies_for, f16_matrix, f32_buf, floor_cells, gpu_warmup, time_op_reps, Out, Rng, Timing,
};
use super::ds4_op_microbench_tests::block_weight;
use super::forward_support::{
    alloc, expert_matmul, expert_matmul_pair, grouped_output_a_batched, raw_matmul, rms_params,
    BatchedGroupedOutputArena, ExpertMatmulRoute,
};
use super::prefill_annotations;
use super::prefill_flash_attention::{
    encode_deepseek_sparse_flash_prefill, DeepseekSparsePrefillFlashArena,
};
use super::residency::RawMatrixRef;
use crate::serve::gpu::GpuContext;

const ROWS: [usize; 4] = [256, 512, 1024, 2048];
const ATTN_ROWS: [usize; 2] = [512, 1024];
const STARTS: [usize; 2] = [0, 8192];
/// Chunks below this many rows read cold (rotated) weight copies.
const COLD_BELOW_ROWS: usize = 256;
const MIN_REPS: usize = 2;
const HIDDEN: usize = 4096;
const HEADS: usize = 64;
const DIM: usize = 512;
const ROPE: usize = 64;
const WINDOW: usize = 128;
const EXPERTS: usize = 256;
const TOP_K: usize = 6;
const INTER: usize = 2048;
const HC: usize = 4;
const IDX_HEADS: usize = 64;
const IDX_DIM: usize = 128;

fn time(
    ctx: &mut GpuContext,
    encode: impl FnMut(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice, usize) -> Result<()>,
) -> Timing {
    time_op_reps(ctx, MIN_REPS, encode)
}

/// `name` as a comma list of usize, or `default`.
fn env_list(name: &str, default: &[usize]) -> Vec<usize> {
    match std::env::var(name) {
        Ok(v) => v.split(',').map(|s| s.trim().parse().unwrap()).collect(),
        Err(_) => default.to_vec(),
    }
}

fn rows_list(default: &[usize]) -> Vec<usize> {
    env_list("HF2Q_DS4_PREFILL_BENCH_ROWS", default)
}

fn starts_list() -> Vec<usize> {
    env_list("HF2Q_DS4_PREFILL_BENCH_STARTS", &STARTS)
}

/// Weight copies for a T-row cell: one when warm-equivalent (T >= COLD_BELOW_ROWS), else enough to go cold.
fn weight_copies(rows: usize, bytes: usize) -> usize {
    copies_for(bytes, rows >= COLD_BELOW_ROWS)
}

fn bf16_buf(device: &MlxDevice, shape: Vec<usize>, seed: u64, scale: f32) -> MlxBuffer {
    let mut buf = alloc(device, DType::BF16, shape, "prefill bf16").unwrap();
    let mut rng = Rng(seed);
    for v in buf.as_mut_slice::<u16>().unwrap() {
        *v = half::bf16::from_f32(rng.unit() * scale).to_bits();
    }
    buf
}

fn zeros(device: &MlxDevice, dtype: DType, shape: Vec<usize>) -> MlxBuffer {
    let mut buf = alloc(device, dtype, shape, "prefill zeros").unwrap();
    buf.as_mut_slice::<u8>().unwrap().fill(0);
    buf
}

fn ones(device: &MlxDevice, n: usize) -> MlxBuffer {
    let mut buf = alloc(device, DType::F32, vec![n], "prefill ones").unwrap();
    buf.as_mut_slice::<f32>().unwrap().fill(1.0);
    buf
}

fn positions(device: &MlxDevice, start: usize, rows: usize) -> MlxBuffer {
    let mut buf = alloc(device, DType::U32, vec![rows], "prefill positions").unwrap();
    for (i, v) in buf.as_mut_slice::<u32>().unwrap().iter_mut().enumerate() {
        *v = (start + i) as u32;
    }
    buf
}

/// Distinct random experts per token (U32, as the sanitized routing output).
fn expert_ids(device: &MlxDevice, tokens: usize, seed: u64) -> MlxBuffer {
    let mut ids = alloc(device, DType::U32, vec![tokens, TOP_K], "prefill ids").unwrap();
    let mut rng = Rng(seed);
    for row in ids.as_mut_slice::<u32>().unwrap().chunks_mut(TOP_K) {
        let mut chosen: Vec<u32> = Vec::with_capacity(TOP_K);
        while chosen.len() < TOP_K {
            let e = (rng.next() % EXPERTS as u64) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        row.copy_from_slice(&chosen);
    }
    ids
}

/// Positive RoPE inverse frequencies `10000^(-2i/64)` (the kernel rejects non-positive ones).
fn rope_freqs(device: &MlxDevice) -> MlxBuffer {
    let mut buf = alloc(device, DType::F32, vec![ROPE / 2], "prefill rope freqs").unwrap();
    for (i, v) in buf.as_mut_slice::<f32>().unwrap().iter_mut().enumerate() {
        *v = 10000f32.powf(-2.0 * i as f32 / ROPE as f32);
    }
    buf
}

fn emit(out: &mut Out, base: Value, t: &Timing) {
    out.cell(&base, json!({}), t);
}

/// Dense block-quantized projections at m = T (`raw_matmul`, the prefill route).
fn dense_q8_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let ops: [(&str, usize, usize); 6] = [
        ("attn_q_a", 1024, 4096),
        ("attn_kv", 512, 4096),
        ("attn_q_b", 32768, 1024),
        ("attn_output_b", 4096, 8192),
        ("shexp_gate_up", 2048, 4096),
        ("shexp_down", 4096, 2048),
    ];
    let rows_set = rows_list(&ROWS);
    for (name, n, k) in ops {
        let bytes = n * k / 32 * 34;
        let most = rows_set.iter().map(|&t| weight_copies(t, bytes)).max().unwrap();
        let all: Vec<MlxBuffer> = (0..most)
            .map(|c| block_weight(&device, GgmlType::Q8_0, vec![n, k], 1 + 1000 * c as u64))
            .collect();
        let shape = [n, k];
        for &rows in &rows_set {
            let ws = &all[..weight_copies(rows, bytes)];
            let x = f32_buf(&device, vec![rows, k], 2, 1.0);
            let y = alloc(&device, DType::F32, vec![rows, n], "y").unwrap();
            let t = time(ctx, |s, r, d, rep| {
                let wr = RawMatrixRef {
                    buffer: &ws[rep % ws.len()],
                    ggml_type: GgmlType::Q8_0,
                    shape: &shape,
                };
                raw_matmul(s, r, d, &x, &wr, &y, rows, n, k, name)
            });
            emit(
                out,
                json!({"family": "dense_q8_0", "op": name, "n": n, "k": k, "t": rows, "copies": ws.len()}),
                &t,
            );
        }
    }
    // grouped output_a: 8 groups of [1024, 4096] over token-major attention
    let (groups, rank) = (8usize, 1024usize);
    let gw = HEADS * DIM / groups;
    let bytes = groups * rank * gw / 32 * 34;
    let most = rows_set.iter().map(|&t| weight_copies(t, bytes)).max().unwrap();
    let all: Vec<MlxBuffer> = (0..most)
        .map(|c| block_weight(&device, GgmlType::Q8_0, vec![groups * rank, gw], 3 + 1000 * c as u64))
        .collect();
    let shape = [groups * rank, gw];
    for &rows in &rows_set {
        let ws = &all[..weight_copies(rows, bytes)];
        let x = f32_buf(&device, vec![rows, HEADS * DIM], 4, 1.0);
        let y = alloc(&device, DType::F32, vec![rows, groups * rank], "y").unwrap();
        let arena = BatchedGroupedOutputArena::new(&device, rows, groups, gw, rank).unwrap();
        let t = time(ctx, |s, r, d, rep| {
            let wr = RawMatrixRef {
                buffer: &ws[rep % ws.len()],
                ggml_type: GgmlType::Q8_0,
                shape: &shape,
            };
            grouped_output_a_batched(s, r, d, &x, &wr, &y, &arena, rows, groups, rank, HEADS, DIM)
        });
        emit(
            out,
            json!({"family": "dense_q8_0", "op": "attn_output_a_grouped", "n": groups * rank, "k": gw, "t": rows, "copies": ws.len()}),
            &t,
        );
    }
}

/// F16 products at m = T (`row_matmul(Matrix)` = `raw_matmul`).
fn dense_f16_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let ops: [(&str, usize, usize); 7] = [
        ("hc_fn", 24, 16384),
        ("router", 256, 4096),
        ("compressor_r4", 1024, 4096),
        ("compressor_r128", 512, 4096),
        ("indexer_q_b", 8192, 1024),
        ("indexer_proj", 64, 4096),
        ("indexer_compressor", 256, 4096),
    ];
    let rows_set = rows_list(&ROWS);
    for (name, n, k) in ops {
        let bytes = n * k * 2;
        let most = rows_set.iter().map(|&t| weight_copies(t, bytes)).max().unwrap();
        let all: Vec<MlxBuffer> = (0..most)
            .map(|c| f16_matrix(&device, n, k, 10 + 1000 * c as u64))
            .collect();
        let shape = [n, k];
        for &rows in &rows_set {
            let ws = &all[..weight_copies(rows, bytes)];
            let x = f32_buf(&device, vec![rows, k], 11, 1.0);
            let y = alloc(&device, DType::F32, vec![rows, n], "y").unwrap();
            let t = time(ctx, |s, r, d, rep| {
                let wr = RawMatrixRef {
                    buffer: &ws[rep % ws.len()],
                    ggml_type: GgmlType::F16,
                    shape: &shape,
                };
                raw_matmul(s, r, d, &x, &wr, &y, rows, n, k, name)
            });
            emit(
                out,
                json!({"family": "dense_f16", "op": name, "n": n, "k": k, "t": rows, "copies": ws.len()}),
                &t,
            );
        }
    }
}

/// Routed experts: paired IQ2_XXS gate/up (`expert_matmul_pair`) and slotted Q2_K down; at
/// `T <= MM_ID_ROUTING_THRESHOLD` the production small-chunk route (separate gate, up and down, `Auto`
/// without scratch). Small chunks rotate expert-id sets so each copy reads different experts (cold).
fn expert_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let gate = block_weight(&device, GgmlType::IQ2_XXS, vec![EXPERTS, INTER, HIDDEN], 20);
    let up = block_weight(&device, GgmlType::IQ2_XXS, vec![EXPERTS, INTER, HIDDEN], 21);
    let down = block_weight(&device, GgmlType::Q2_K, vec![EXPERTS, HIDDEN, INTER], 22);
    let gu_shape = [EXPERTS, INTER, HIDDEN];
    let d_shape = [EXPERTS, HIDDEN, INTER];
    let iq2_expert = INTER * HIDDEN / 256 * 66;
    for rows in rows_list(&ROWS) {
        let id_sets: Vec<MlxBuffer> = (0..weight_copies(rows, iq2_expert * rows * TOP_K))
            .map(|c| expert_ids(&device, rows, rows as u64 + 1000 * c as u64))
            .collect();
        let x = f32_buf(&device, vec![rows, HIDDEN], 23, 1.0);
        let g = alloc(&device, DType::F32, vec![rows * TOP_K, INTER], "g").unwrap();
        let u = alloc(&device, DType::F32, vec![rows * TOP_K, INTER], "u").unwrap();
        let act = f32_buf(&device, vec![rows * TOP_K, INTER], 24, 1.0);
        let y = alloc(&device, DType::F32, vec![rows * TOP_K, HIDDEN], "y").unwrap();
        if rows <= MM_ID_ROUTING_THRESHOLD as usize {
            let t = time(ctx, |s, r, d, rep| {
                let ids = &id_sets[rep % id_sets.len()];
                for (w, o, label) in [(&gate, &g, "routed gate"), (&up, &u, "routed up")] {
                    let wr = RawMatrixRef {
                        buffer: w,
                        ggml_type: GgmlType::IQ2_XXS,
                        shape: &gu_shape,
                    };
                    expert_matmul(
                        s,
                        r,
                        d,
                        &x,
                        &wr,
                        ids,
                        o,
                        rows,
                        TOP_K,
                        EXPERTS,
                        INTER,
                        HIDDEN,
                        ExpertMatmulRoute::Auto,
                        None,
                        label,
                    )?;
                }
                Ok(())
            });
            emit(
                out,
                json!({"family": "experts", "op": "gate_up_pair_iq2_xxs", "n": INTER, "k": HIDDEN, "t": rows, "route": "auto_mv_id_separate", "copies": id_sets.len()}),
                &t,
            );
            let t = time(ctx, |s, r, d, rep| {
                let dw = RawMatrixRef {
                    buffer: &down,
                    ggml_type: GgmlType::Q2_K,
                    shape: &d_shape,
                };
                expert_matmul(
                    s,
                    r,
                    d,
                    &act,
                    &dw,
                    &id_sets[rep % id_sets.len()],
                    &y,
                    rows,
                    TOP_K,
                    EXPERTS,
                    HIDDEN,
                    INTER,
                    ExpertMatmulRoute::Auto,
                    None,
                    "routed down",
                )
            });
            emit(
                out,
                json!({"family": "experts", "op": "down_q2_k", "n": HIDDEN, "k": INTER, "t": rows, "route": "auto_mv_id", "copies": id_sets.len()}),
                &t,
            );
            continue;
        }
        let ids = &id_sets[0];
        let mut scratch = IdMmScratch::alloc(&device, EXPERTS as u32, rows as u32).unwrap();
        let t = time(ctx, |s, r, d, _| {
            let gw = RawMatrixRef {
                buffer: &gate,
                ggml_type: GgmlType::IQ2_XXS,
                shape: &gu_shape,
            };
            let uw = RawMatrixRef {
                buffer: &up,
                ggml_type: GgmlType::IQ2_XXS,
                shape: &gu_shape,
            };
            expert_matmul_pair(
                s,
                r,
                d,
                &x,
                &gw,
                &uw,
                ids,
                &g,
                &u,
                rows,
                TOP_K,
                EXPERTS,
                INTER,
                HIDDEN,
                &mut scratch,
                "paired routed gate/up",
            )
        });
        emit(
            out,
            json!({"family": "experts", "op": "gate_up_pair_iq2_xxs", "n": INTER, "k": HIDDEN, "t": rows, "route": "paired_mm_id", "copies": 1}),
            &t,
        );
        let t = time(ctx, |s, r, d, _| {
            let dw = RawMatrixRef {
                buffer: &down,
                ggml_type: GgmlType::Q2_K,
                shape: &d_shape,
            };
            expert_matmul(
                s,
                r,
                d,
                &act,
                &dw,
                ids,
                &y,
                rows,
                TOP_K,
                EXPERTS,
                HIDDEN,
                INTER,
                ExpertMatmulRoute::SlottedMm,
                Some(&mut scratch),
                "routed down",
            )
        });
        emit(
            out,
            json!({"family": "experts", "op": "down_q2_k", "n": HIDDEN, "k": INTER, "t": rows, "route": "slotted_mm", "copies": 1}),
            &t,
        );
    }
}

/// MoE routing, activations and reduce at T rows.
fn moe_misc_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let bias = f32_buf(&device, vec![EXPERTS], 30, 0.1);
    for rows in rows_list(&ROWS) {
        let logits = f32_buf(&device, vec![rows, EXPERTS], 31, 2.0);
        let indices = zeros(&device, DType::I32, vec![rows, TOP_K]);
        let weights = zeros(&device, DType::F32, vec![rows, TOP_K]);
        let safe = zeros(&device, DType::U32, vec![rows, TOP_K]);
        let status = zeros(&device, DType::U32, vec![1]);
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&logits, &bias], &[&indices, &weights]);
            dispatch_deepseek_moe_score_route(
                s.encoder_mut(),
                r,
                d,
                &logits,
                &bias,
                &indices,
                &weights,
                rows,
            )?;
            s.barrier_between(&[&indices], &[&safe]);
            dispatch_deepseek_moe_sanitize_indices(
                s.encoder_mut(),
                r,
                d,
                &indices,
                &safe,
                &status,
                rows,
            )?;
            Ok(())
        });
        emit(
            out,
            json!({"family": "moe_misc", "op": "route_sanitize", "t": rows}),
            &t,
        );
        for (name, act_rows) in [("swiglu_routed", rows * TOP_K), ("swiglu_shared", rows)] {
            let g = f32_buf(&device, vec![act_rows, INTER], 32, 1.0);
            let u = f32_buf(&device, vec![act_rows, INTER], 33, 1.0);
            let o = alloc(&device, DType::F32, vec![act_rows, INTER], "o").unwrap();
            let t = time(ctx, |s, r, d, _| {
                s.barrier_between(&[&g, &u], &[&o]);
                dispatch_deepseek_moe_swiglu(
                    s.encoder_mut(),
                    r,
                    d,
                    &g,
                    &u,
                    None,
                    &o,
                    &status,
                    act_rows,
                )?;
                Ok(())
            });
            emit(
                out,
                json!({"family": "moe_misc", "op": name, "t": rows}),
                &t,
            );
        }
        let idx = {
            let ids = expert_ids(&device, rows, 34);
            let mut i = alloc(&device, DType::I32, vec![rows, TOP_K], "i").unwrap();
            for (dst, src) in i
                .as_mut_slice::<i32>()
                .unwrap()
                .iter_mut()
                .zip(ids.as_slice::<u32>().unwrap())
            {
                *dst = *src as i32;
            }
            i
        };
        let w = f32_buf(&device, vec![rows, TOP_K], 35, 0.2);
        let routed = f32_buf(&device, vec![rows, TOP_K, HIDDEN], 36, 1.0);
        let shared = f32_buf(&device, vec![rows, HIDDEN], 37, 1.0);
        let o = alloc(&device, DType::F32, vec![rows, HIDDEN], "o").unwrap();
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&idx, &w, &routed, &shared], &[&o]);
            dispatch_deepseek_moe_weighted_reduce(
                s.encoder_mut(),
                r,
                d,
                &idx,
                &w,
                &routed,
                &shared,
                &o,
                &status,
                rows,
            )?;
            Ok(())
        });
        emit(
            out,
            json!({"family": "moe_misc", "op": "weighted_reduce", "t": rows}),
            &t,
        );
    }
}

/// Hyper-connection ops, norms, RoPE, casts and quantizers at T rows.
fn rowwise_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let hc_hidden = HC * HIDDEN;
    let mix = (2 + HC) * HC;
    let hc_scale = f32_buf(&device, vec![3], 40, 0.5);
    let hc_base = f32_buf(&device, vec![mix], 41, 0.5);
    let hc_params = rms_params(&device, 1e-6, hc_hidden, "hc").unwrap();
    let hidden_params = rms_params(&device, 1e-6, HIDDEN, "hidden").unwrap();
    let rank_params = rms_params(&device, 1e-6, 1024, "rank").unwrap();
    let head_params = rms_params(&device, 1e-6, DIM, "head").unwrap();
    let norm_w = ones(&device, HIDDEN);
    let rank_w = ones(&device, 1024);
    let head_w = ones(&device, DIM);
    let freqs = rope_freqs(&device);
    for rows in rows_list(&ROWS) {
        let r32 = rows as u32;
        let base = |op: &str| json!({"family": "rowwise", "op": op, "t": rows});
        let state = f32_buf(&device, vec![rows, HC, HIDDEN], 43, 1.0);
        let state_flat = state.with_shape(vec![rows, hc_hidden]).unwrap();
        let state_norm = alloc(&device, DType::F32, vec![rows, hc_hidden], "sn").unwrap();
        let mixes = f32_buf(&device, vec![rows, mix], 44, 0.5);
        let pre = zeros(&device, DType::F32, vec![rows, HC]);
        let post = zeros(&device, DType::F32, vec![rows, HC]);
        let comb = zeros(&device, DType::F32, vec![rows, HC, HC]);
        let attn_in = alloc(&device, DType::F32, vec![rows, HIDDEN], "ai").unwrap();
        let x = f32_buf(&device, vec![rows, HIDDEN], 45, 1.0);
        let hc_out = alloc(&device, DType::F32, vec![rows, HC, HIDDEN], "ho").unwrap();
        let t = time(ctx, |s, r, d, _| {
            Ok(prefill_annotations::rms_norm_no_scale_f32(
                s,
                r,
                d.metal_device(),
                &state_flat,
                &state_norm,
                &hc_params,
                r32,
                hc_hidden as u32,
            )?)
        });
        emit(out, base("hc_norm"), &t);
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&mixes, &hc_scale, &hc_base], &[&pre, &post, &comb]);
            dispatch_hc_split_sinkhorn(
                s.encoder_mut(),
                r,
                d,
                &mixes,
                &hc_scale,
                &hc_base,
                &pre,
                &post,
                &comb,
                r32,
            )?;
            Ok(())
        });
        emit(out, base("hc_split_sinkhorn"), &t);
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&state, &pre], &[&attn_in]);
            dispatch_hc_pre(
                s.encoder_mut(),
                r,
                d,
                &state,
                &pre,
                &attn_in,
                r32,
                HIDDEN as u32,
            )?;
            Ok(())
        });
        emit(out, base("hc_pre"), &t);
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&x, &state, &post, &comb], &[&hc_out]);
            dispatch_hc_post(
                s.encoder_mut(),
                r,
                d,
                &x,
                &state,
                &post,
                &comb,
                &hc_out,
                r32,
                HIDDEN as u32,
            )?;
            Ok(())
        });
        emit(out, base("hc_post"), &t);
        let y = alloc(&device, DType::F32, vec![rows, HIDDEN], "y").unwrap();
        let t = time(ctx, |s, r, d, _| {
            Ok(prefill_annotations::rms_norm(
                s,
                r,
                d.metal_device(),
                &x,
                &norm_w,
                &y,
                &hidden_params,
                r32,
                HIDDEN as u32,
            )?)
        });
        emit(out, base("rms_norm_4096"), &t);
        let xr = f32_buf(&device, vec![rows, 1024], 46, 1.0);
        let yr = alloc(&device, DType::F32, vec![rows, 1024], "yr").unwrap();
        let t = time(ctx, |s, r, d, _| {
            Ok(prefill_annotations::rms_norm(
                s,
                r,
                d.metal_device(),
                &xr,
                &rank_w,
                &yr,
                &rank_params,
                r32,
                1024,
            )?)
        });
        emit(out, base("rms_norm_1024"), &t);
        let q = f32_buf(&device, vec![rows, HEADS, DIM], 47, 1.0);
        let qn = alloc(&device, DType::F32, vec![1, rows, HEADS, DIM], "qn").unwrap();
        let t = time(ctx, |s, r, d, _| {
            Ok(prefill_annotations::rms_norm_no_scale_f32(
                s,
                r,
                d.metal_device(),
                &q,
                &qn,
                &head_params,
                (rows * HEADS) as u32,
                DIM as u32,
            )?)
        });
        emit(out, base("q_norm_512x64"), &t);
        let kv = f32_buf(&device, vec![rows, DIM], 48, 1.0);
        let kvn = alloc(&device, DType::F32, vec![1, rows, 1, DIM], "kvn").unwrap();
        let t = time(ctx, |s, r, d, _| {
            Ok(prefill_annotations::rms_norm(
                s,
                r,
                d.metal_device(),
                &kv,
                &head_w,
                &kvn,
                &head_params,
                r32,
                DIM as u32,
            )?)
        });
        emit(out, base("kv_norm_512"), &t);
        let pos = positions(&device, 1000, rows);
        let rope = DeepSeekTailRopeParams {
            batch: 1,
            seq_len: r32,
            heads: HEADS as u32,
            head_dim: DIM as u32,
            rope_dim: ROPE as u32,
            inverse: 0,
        };
        let q_rope = alloc(&device, DType::BF16, vec![1, rows, HEADS, DIM], "qr").unwrap();
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&qn, &pos, &freqs], &[&q_rope]);
            dispatch_deepseek_tail_rope_f32_to_bf16(
                s.encoder_mut(),
                r,
                d,
                &qn,
                &pos,
                &freqs,
                &q_rope,
                &rope,
            )?;
            Ok(())
        });
        emit(out, base("q_rope"), &t);
        let kv_rope = alloc(&device, DType::BF16, vec![1, rows, 1, DIM], "kr").unwrap();
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&kvn, &pos, &freqs], &[&kv_rope]);
            dispatch_deepseek_tail_rope_f32_to_bf16(
                s.encoder_mut(),
                r,
                d,
                &kvn,
                &pos,
                &freqs,
                &kv_rope,
                &DeepSeekTailRopeParams { heads: 1, ..rope },
            )?;
            s.barrier_between(&[&kv_rope], &[&kv_rope]);
            dispatch_deepseek_mxfp8_fake_quant_bf16(
                s.encoder_mut(),
                r,
                d,
                &kv_rope,
                &DeepSeekMxfp8Params {
                    rows: r32,
                    row_width: DIM as u32,
                    quantized_width: (DIM - ROPE) as u32,
                    block_size: 64,
                },
            )?;
            Ok(())
        });
        emit(out, base("kv_rope_fake_quant"), &t);
        let attn = bf16_buf(&device, vec![1, rows, HEADS, DIM], 49, 1.0);
        let unrot = alloc(&device, DType::BF16, vec![1, rows, HEADS, DIM], "un").unwrap();
        let attn_f32 = alloc(&device, DType::F32, vec![rows, HEADS, DIM], "af").unwrap();
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&attn, &pos, &freqs], &[&unrot]);
            dispatch_deepseek_tail_rope_bf16(
                s.encoder_mut(),
                r,
                d,
                &attn,
                &pos,
                &freqs,
                &unrot,
                &DeepSeekTailRopeParams { inverse: 1, ..rope },
            )?;
            s.barrier_between(&[&unrot], &[&attn_f32]);
            dispatch_cast_bf16_to_f32_with_encoder(
                s.encoder_mut(),
                r,
                d.metal_device(),
                &unrot,
                &attn_f32,
                (rows * HEADS * DIM) as u32,
            )?;
            Ok(())
        });
        emit(out, base("inverse_rope_cast"), &t);
        let iq = bf16_buf(&device, vec![1, rows, IDX_HEADS, IDX_DIM], 50, 1.0);
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&iq], &[&iq]);
            dispatch_deepseek_hadamard_mxfp4_bf16(
                s.encoder_mut(),
                r,
                d,
                &iq,
                (rows * IDX_HEADS) as u32,
            )?;
            Ok(())
        });
        emit(out, base("indexer_q_hadamard"), &t);
        // Window write: the last min(T, 128) rows into the 128-row ring.
        let window = zeros(&device, DType::BF16, vec![WINDOW, DIM]);
        let n_new = rows.min(WINDOW);
        let src = kv_rope.slice_view(((rows - n_new) * DIM * 2) as u64, n_new * DIM);
        let t = time(ctx, |s, r, d, _| {
            s.barrier_between(&[&src], &[&window]);
            dispatch_kv_cache_copy(
                s.encoder_mut(),
                r,
                d.metal_device(),
                &src,
                &window,
                0,
                DIM as u32,
                n_new as u32,
                WINDOW as u32,
                true,
            )?;
            Ok(())
        });
        emit(out, base("window_write"), &t);
    }
}

/// Distinct sorted random picks from `0..valid` (at most `k`).
fn picks(rng: &mut Rng, valid: usize, k: usize) -> Vec<i32> {
    if valid <= k {
        return (0..valid as i32).collect();
    }
    let mut set = std::collections::BTreeSet::new();
    while set.len() < k {
        set.insert((rng.next() % valid as u64) as i32);
    }
    set.into_iter().collect()
}

/// Gathered sparse flash prefill for window, HCA and CSA layers.
fn attention_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let sinks = f32_buf(&device, vec![HEADS], 60, 1.0);
    for rows in rows_list(&ATTN_ROWS) {
        let q = bf16_buf(&device, vec![1, rows, HEADS, DIM], 61, 1.0);
        let kv = bf16_buf(&device, vec![1, rows, 1, DIM], 62, 0.5);
        let o = alloc(&device, DType::BF16, vec![1, rows, HEADS, DIM], "o").unwrap();
        for s0 in starts_list() {
            let prefix = (s0 > 0).then(|| bf16_buf(&device, vec![1, WINDOW, 1, DIM], 63, 0.5));
            for (kind, ratio) in [("window", 0u32), ("hca", 128), ("csa", 4)] {
                let valid = if ratio == 0 {
                    0
                } else {
                    (s0 + rows) / ratio as usize
                };
                // (indices, width, raw_prefix_len, raw_kv_len, kv_len), as the production prefill builds them.
                let (index, width, raw_prefix_len, raw_kv_len, kv_len) = if ratio == 0 {
                    // Window layers: `attention_forward` window indices over [raw prefix ++ chunk].
                    let window = if s0 > 0 {
                        append_prefill_window_indices(WINDOW, rows, s0).unwrap()
                    } else {
                        window_indices(WINDOW, rows, s0).unwrap()
                    };
                    let width = window[0].len();
                    let prefix_len = if s0 > 0 { WINDOW } else { 0 };
                    let flat: Vec<i32> = window.into_iter().flatten().collect();
                    (
                        flat,
                        width,
                        prefix_len,
                        prefix_len + rows,
                        prefix_len + rows,
                    )
                } else {
                    let layout =
                        prepare_layout(ratio, rows, s0, 0, WINDOW, valid, 512, true).unwrap();
                    let mut index = layout.index_storage.clone();
                    if ratio == 4 {
                        // Indexer output region: each row's causal top-512 over its visible compressed rows.
                        let column = layout.indexer_output_offset.unwrap();
                        let mut rng = Rng(64);
                        for row in 0..rows {
                            let visible = ((s0 + row + 1) / 4).min(valid);
                            for (slot, pick) in
                                picks(&mut rng, visible, 512).into_iter().enumerate()
                            {
                                index[row * layout.storage_stride + column + slot] =
                                    pick + layout.compressed_attention_offset as i32;
                            }
                        }
                    }
                    (
                        index,
                        layout.attention_width,
                        layout.raw_prefix_len,
                        layout.compact_raw_kv_len,
                        layout.compact_prefill_kv_len,
                    )
                };
                let mut indices = alloc(&device, DType::I32, vec![1, rows, width], "idx").unwrap();
                indices
                    .as_mut_slice::<i32>()
                    .unwrap()
                    .copy_from_slice(&index);
                let comp = bf16_buf(&device, vec![valid.max(1), DIM], 65, 0.5);
                let arena =
                    DeepseekSparsePrefillFlashArena::new(&device, rows, HEADS, DIM, kv_len, width)
                        .unwrap();
                let t = time(ctx, |s, r, d, _| {
                    encode_deepseek_sparse_flash_prefill(
                        s,
                        r,
                        d,
                        &q,
                        prefix.as_ref(),
                        &kv,
                        &comp,
                        &sinks,
                        &indices,
                        &o,
                        &arena,
                        rows,
                        raw_prefix_len,
                        raw_kv_len,
                        kv_len,
                        width,
                        HEADS,
                        DIM,
                        1.0 / (DIM as f32).sqrt(),
                    )
                });
                emit(
                    out,
                    json!({"family": "attention", "op": format!("{kind}_flash_prefill_gathered"), "t": rows,
                                 "s0": s0, "width": width, "kv_len": kv_len, "valid_compressed": valid}),
                    &t,
                );
            }
        }
    }
}

/// Ratio-4 indexer selection (score + top-k) over a T-row query tile at cache start s0.
fn indexer_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let (stride, column) = (640usize, 128usize);
    for rows in rows_list(&ATTN_ROWS) {
        let q = bf16_buf(&device, vec![1, rows, IDX_HEADS, IDX_DIM], 70, 1.0);
        let w = f32_buf(
            &device,
            vec![1, rows, IDX_HEADS],
            71,
            1.0 / ((IDX_DIM * IDX_HEADS) as f32).sqrt(),
        );
        let o = zeros(&device, DType::I32, vec![1, rows, stride]);
        for s0 in starts_list() {
            let keys = (s0 + rows) / 4;
            let kv = bf16_buf(&device, vec![1, keys, IDX_DIM], 72, 0.5);
            let params = DeepSeekIndexerParams {
                batch: 1,
                query_len: rows as u32,
                kv_len: keys as u32,
                start_pos: s0 as u32,
                ratio: 4,
                heads: IDX_HEADS as u32,
                head_dim: IDX_DIM as u32,
                top_k: 512,
                offset: 128,
            };
            let plan = deepseek_indexer_query_tile_plan(&params, stride, column).unwrap();
            let scratch = zeros(&device, DType::F32, vec![1, plan.workspace_rows, keys]);
            let topk = zeros(
                &device,
                DType::I32,
                vec![1, plan.workspace_rows, plan.topk_row_elements],
            );
            let t = time(ctx, |s, r, d, _| {
                s.barrier_between(&[&q, &kv, &w], &[&scratch, &topk, &o]);
                dispatch_deepseek_indexer_into_mma_tiled(
                    s.encoder_mut(),
                    r,
                    d,
                    &q,
                    &kv,
                    &w,
                    &scratch,
                    &topk,
                    &o,
                    stride,
                    column,
                    &params,
                )?;
                Ok(())
            });
            emit(
                out,
                json!({"family": "indexer", "op": "select_score_topk", "t": rows, "s0": s0, "keys": keys}),
                &t,
            );
        }
    }
}

/// Stateful compressor over a T-row chunk (main r4 / r128, indexer r4), plus RoPE of the emitted blocks.
fn compressor_cells(ctx: &mut GpuContext, out: &mut Out) {
    let device = ctx.device().clone();
    let freqs = rope_freqs(&device);
    for (name, ratio, dim) in [
        ("csa_main", 4usize, 512usize),
        ("csa_indexer", 4, 128),
        ("hca_main", 128, 512),
    ] {
        let coff = if ratio == 4 { 2 } else { 1 };
        let projected = coff * dim;
        let ape = f32_buf(&device, vec![ratio, projected], 81, 0.1);
        let norm = ones(&device, dim);
        for rows in rows_list(&ATTN_ROWS) {
            for s0 in starts_list() {
                let params = DeepSeekCompressorParams {
                    batch: 1,
                    seq_len: rows as u32,
                    start_pos: s0 as u32,
                    ratio: ratio as u32,
                    head_dim: dim as u32,
                    cache_len: ((s0 + rows) / ratio + 1) as u32,
                    epsilon: 1e-6,
                    write_cache: 0,
                };
                let kv = f32_buf(&device, vec![1, rows, projected], 82, 1.0);
                let score = f32_buf(&device, vec![1, rows, projected], 83, 1.0);
                let ks = f32_buf(&device, vec![1, coff * ratio, projected], 84, 1.0);
                let ss = f32_buf(&device, vec![1, coff * ratio, projected], 85, 1.0);
                let slots = params.output_slots();
                let o = alloc(&device, DType::BF16, vec![1, slots, dim], "o").unwrap();
                let blocks = params.output_count();
                let cpos = positions(&device, s0, blocks.max(1));
                let rope_out =
                    alloc(&device, DType::BF16, vec![1, blocks.max(1), 1, dim], "ro").unwrap();
                let t = time(ctx, |s, r, d, _| {
                    s.barrier_between(&[&kv, &score, &ape, &norm, &ks, &ss], &[&ks, &ss, &o]);
                    dispatch_deepseek_compressor(
                        s.encoder_mut(),
                        r,
                        d,
                        &kv,
                        &score,
                        &ape,
                        &norm,
                        &ks,
                        &ss,
                        &o,
                        &o,
                        &params,
                    )?;
                    if blocks > 0 {
                        let rope_in = o
                            .slice_view(0, blocks * dim)
                            .with_shape(vec![1, blocks, 1, dim])?;
                        s.barrier_between(&[&rope_in, &cpos, &freqs], &[&rope_out]);
                        dispatch_deepseek_tail_rope_bf16(
                            s.encoder_mut(),
                            r,
                            d,
                            &rope_in,
                            &cpos,
                            &freqs,
                            &rope_out,
                            &DeepSeekTailRopeParams {
                                batch: 1,
                                seq_len: blocks as u32,
                                heads: 1,
                                head_dim: dim as u32,
                                rope_dim: ROPE as u32,
                                inverse: 0,
                            },
                        )?;
                    }
                    Ok(())
                });
                emit(
                    out,
                    json!({"family": "compressor", "op": name, "ratio": ratio, "dim": dim, "t": rows, "s0": s0, "blocks": blocks}),
                    &t,
                );
            }
        }
    }
}

#[test]
#[ignore = "model-free GPU microbenchmarks (gap program P2; standing grant, <= 15 min)"]
fn ds4_prefill_microbench() {
    let out_path =
        std::env::var("HF2Q_DS4_PREFILL_BENCH_OUT").expect("set HF2Q_DS4_PREFILL_BENCH_OUT");
    let mut out = Out(std::fs::File::create(&out_path).unwrap());
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    gpu_warmup(&mut ctx);
    floor_cells(&mut ctx, &mut out);
    attention_cells(&mut ctx, &mut out);
    indexer_cells(&mut ctx, &mut out);
    compressor_cells(&mut ctx, &mut out);
    rowwise_cells(&mut ctx, &mut out);
    moe_misc_cells(&mut ctx, &mut out);
    dense_q8_cells(&mut ctx, &mut out);
    dense_f16_cells(&mut ctx, &mut out);
    expert_cells(&mut ctx, &mut out);
    eprintln!("cells: {out_path}");
}
