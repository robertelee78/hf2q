//! DeepSeek-V4 gap program M2c (Fix B stage 0): what one dispatch over B rows buys with TODAY's kernels.
//!
//! The per-row cache-dependent kernels already accept a batch dimension when the B rows' data is
//! contiguous. This model-free bench times, at the ds4 file's decode shapes and B = 1..4:
//! - `per_row`: B dispatches at batch 1 on row views (as joint decode issues them, one barrier per row);
//! - `batched`: one dispatch at batch B on the same contiguous buffers.
//! It also checks that the batched outputs (and compressor states) are byte-equal to the per-row ones.
//!
//! Ops:
//! - window / HCA sparse attention: `dispatch_deepseek_sparse_attention_with_scratch` (score + reduce);
//! - CSA sparse attention: `dispatch_deepseek_sparse_attention_flash_prefill` (validate, gather, flash, sanitize);
//! - CSA indexer select: `dispatch_deepseek_indexer_into_mma_tiled` with `query_len = B` over one indexer cache
//!   (the kernel has no per-query cache; rows share it here, which can only make `batched` look better);
//! - compressor tails: `dispatch_deepseek_compressor` (CSA 512, indexer 128, HCA 512; steady and emit steps).
//!
//! Timing: `ds4_bench_support` (GPU timestamps, >= 2 ms per session, median of 9). Cold cells rotate over
//! input copies exceeding 256 MiB (at most 64).
//!
//! Run (standing grant, model-free, <= 15 min):
//! ```text
//! HF2Q_DS4_BATCH_PROBE_OUT=<cells.jsonl> cargo test --release --bin hf2q ds4_batch_probe -- --ignored --nocapture
//! ```

use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_compressor::{
    dispatch_deepseek_compressor, DeepSeekCompressorParams,
};
use mlx_native::ops::deepseek_indexer::{
    deepseek_indexer_query_tile_plan, dispatch_deepseek_indexer_into_mma_tiled,
    DeepSeekIndexerParams,
};
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention_flash_prefill,
    dispatch_deepseek_sparse_attention_with_scratch, DeepSeekSparseAttentionParams,
};
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use serde_json::{json, Value};

use super::ds4_bench_support::{
    copies_for, f32_buf, floor_cells, gpu_warmup, run_once, time_op, Out, Rng,
};
use super::forward_support::alloc;
use super::row_matmul::row_view;
use crate::serve::gpu::GpuContext;

const HEADS: usize = 64;
const DIM: usize = 512;
const MAX_ROWS: usize = 4;

fn bf16_buf(device: &MlxDevice, shape: Vec<usize>, seed: u64, scale: f32) -> MlxBuffer {
    let mut buf = alloc(device, DType::BF16, shape, "probe bf16").unwrap();
    let mut rng = Rng(seed);
    for v in buf.as_mut_slice::<u16>().unwrap() {
        *v = half::bf16::from_f32(rng.unit() * scale).to_bits();
    }
    buf
}

fn zeros(device: &MlxDevice, dtype: DType, shape: Vec<usize>) -> MlxBuffer {
    let mut buf = alloc(device, dtype, shape, "probe zeros").unwrap();
    buf.as_mut_slice::<u8>().unwrap().fill(0);
    buf
}

fn i32_buf(device: &MlxDevice, shape: Vec<usize>, values: &[i32]) -> MlxBuffer {
    let mut buf = alloc(device, DType::I32, shape, "probe i32").unwrap();
    buf.as_mut_slice::<i32>().unwrap().copy_from_slice(values);
    buf
}

fn bytes(buf: &MlxBuffer) -> Vec<u8> {
    buf.as_slice::<u8>().unwrap().to_vec()
}

fn copy_into(dst: &mut MlxBuffer, src: &[u8]) {
    dst.as_mut_slice::<u8>().unwrap().copy_from_slice(src);
}

/// Row `row` of a buffer whose leading dimension is the batch, reshaped to `shape` (leading 1).
fn rv(buf: &MlxBuffer, row: usize, shape: Vec<usize>) -> MlxBuffer {
    row_view(buf, row, shape).unwrap()
}

/// Time `per_row` and `batched` encodes and emit both cells.
fn probe(
    ctx: &mut GpuContext,
    out: &mut Out,
    base: &Value,
    equal: bool,
    mut per_row: impl FnMut(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice, usize) -> Result<()>,
    mut batched: impl FnMut(&mut GraphSession<'_>, &mut KernelRegistry, &MlxDevice, usize) -> Result<()>,
) {
    let t = time_op(ctx, |s, r, d, c| per_row(s, r, d, c));
    out.cell(base, json!({"mode": "per_row", "bitwise_equal": equal}), &t);
    let t = time_op(ctx, |s, r, d, c| batched(s, r, d, c));
    out.cell(base, json!({"mode": "batched", "bitwise_equal": equal}), &t);
}

/// Distinct sorted indices: the window `0..min(window, width)` then `width - window` compressed rows.
fn indices_for(rng: &mut Rng, kv_len: usize, width: usize, window: usize) -> Vec<i32> {
    let mut row: Vec<i32> = (0..window.min(width) as i32).collect();
    let mut chosen = std::collections::BTreeSet::new();
    while row.len() + chosen.len() < width {
        chosen.insert(window + (rng.next() as usize) % (kv_len - window));
    }
    row.extend(chosen.into_iter().map(|i| i as i32));
    row
}

struct AttnInputs {
    q: MlxBuffer,
    kv: Vec<MlxBuffer>,
    indices: MlxBuffer,
    sinks: MlxBuffer,
}

fn attn_inputs(
    device: &MlxDevice,
    b: usize,
    kv_len: usize,
    width: usize,
    copies: usize,
) -> AttnInputs {
    let mut rng = Rng(7 + b as u64);
    let idx: Vec<i32> = (0..b)
        .flat_map(|_| indices_for(&mut rng, kv_len, width, 128.min(kv_len)))
        .collect();
    AttnInputs {
        q: bf16_buf(device, vec![b, 1, HEADS, DIM], 11, 1.0),
        kv: (0..copies)
            .map(|c| bf16_buf(device, vec![b, kv_len, DIM], 100 + c as u64, 0.5))
            .collect(),
        indices: i32_buf(device, vec![b, 1, width], &idx),
        sinks: f32_buf(device, vec![HEADS], 13, 1.0),
    }
}

fn sparse_params(batch: usize, kv_len: usize, width: usize) -> DeepSeekSparseAttentionParams {
    DeepSeekSparseAttentionParams {
        batch: batch as u32,
        query_len: 1,
        kv_len: kv_len as u32,
        top_k: width as u32,
        heads: HEADS as u32,
        head_dim: DIM as u32,
        scale: 1.0 / (DIM as f32).sqrt(),
    }
}

/// Window / HCA layers: score + reduce.
fn score_reduce_cells(
    ctx: &mut GpuContext,
    out: &mut Out,
    kind: &str,
    kv_len: usize,
    width: usize,
) {
    let device = ctx.device().clone();
    for b in 1..=MAX_ROWS {
        let copies = copies_for(b * kv_len * DIM * 2, false);
        let x = attn_inputs(&device, b, kv_len, width, copies);
        let scratch_rows = zeros(&device, DType::F32, vec![b, 1, HEADS, width]);
        let scratch_batch = zeros(&device, DType::F32, vec![b, 1, HEADS, width]);
        let out_rows = zeros(&device, DType::BF16, vec![b, 1, HEADS, DIM]);
        let out_batch = zeros(&device, DType::BF16, vec![b, 1, HEADS, DIM]);
        let per_row = |s: &mut GraphSession<'_>,
                       r: &mut KernelRegistry,
                       d: &MlxDevice,
                       c: usize|
         -> Result<()> {
            let kv = &x.kv[c % x.kv.len()];
            for row in 0..b {
                let q = rv(&x.q, row, vec![1, 1, HEADS, DIM]);
                let k = rv(kv, row, vec![1, kv_len, DIM]);
                let i = rv(&x.indices, row, vec![1, 1, width]);
                let sc = rv(&scratch_rows, row, vec![1, 1, HEADS, width]);
                let o = rv(&out_rows, row, vec![1, 1, HEADS, DIM]);
                s.barrier_between(&[&q, &k, &x.sinks, &i], &[&sc, &o]);
                dispatch_deepseek_sparse_attention_with_scratch(
                    s.encoder_mut(),
                    r,
                    d,
                    &q,
                    &k,
                    &x.sinks,
                    &i,
                    &sc,
                    &o,
                    &sparse_params(1, kv_len, width),
                )?;
            }
            Ok(())
        };
        let batched = |s: &mut GraphSession<'_>,
                       r: &mut KernelRegistry,
                       d: &MlxDevice,
                       c: usize|
         -> Result<()> {
            let kv = &x.kv[c % x.kv.len()];
            s.barrier_between(
                &[&x.q, kv, &x.sinks, &x.indices],
                &[&scratch_batch, &out_batch],
            );
            dispatch_deepseek_sparse_attention_with_scratch(
                s.encoder_mut(),
                r,
                d,
                &x.q,
                kv,
                &x.sinks,
                &x.indices,
                &scratch_batch,
                &out_batch,
                &sparse_params(b, kv_len, width),
            )?;
            Ok(())
        };
        let pr = per_row;
        let ba = batched;
        run_once(ctx, |s, r, d| pr(s, r, d, 0));
        run_once(ctx, |s, r, d| ba(s, r, d, 0));
        let equal = bytes(&out_rows) == bytes(&out_batch);
        let base = json!({"family": "batch_probe", "op": format!("{kind}_attention"), "kernel": "sparse_score_reduce",
                          "b": b, "kv_len": kv_len, "width": width, "copies": copies, "warm": false});
        probe(ctx, out, &base, equal, pr, ba);
    }
}

/// CSA layers: validate + gather + flash + sanitize.
fn flash_cells(ctx: &mut GpuContext, out: &mut Out, kv_len: usize, width: usize) {
    let device = ctx.device().clone();
    for b in 1..=MAX_ROWS {
        let copies = copies_for(b * kv_len * DIM * 2, false);
        let x = attn_inputs(&device, b, kv_len, width, copies);
        let mk = |device: &MlxDevice| {
            (
                zeros(device, DType::BF16, vec![b, 1, width, DIM]),
                zeros(device, DType::BF16, vec![b, 1, width]),
                zeros(device, DType::U32, vec![b, 1]),
                zeros(device, DType::U32, vec![b, 1, HEADS]),
                zeros(device, DType::BF16, vec![b, 1, HEADS, DIM]),
            )
        };
        let (g_rows, m_rows, ig_rows, ih_rows, out_rows) = mk(&device);
        let (g_b, m_b, ig_b, ih_b, out_b) = mk(&device);
        let per_row = |s: &mut GraphSession<'_>,
                       r: &mut KernelRegistry,
                       d: &MlxDevice,
                       c: usize|
         -> Result<()> {
            let kv = &x.kv[c % x.kv.len()];
            for row in 0..b {
                let q = rv(&x.q, row, vec![1, 1, HEADS, DIM]);
                let k = rv(kv, row, vec![1, kv_len, DIM]);
                let i = rv(&x.indices, row, vec![1, 1, width]);
                let g = rv(&g_rows, row, vec![1, 1, width, DIM]);
                let m = rv(&m_rows, row, vec![1, 1, width]);
                let ig = rv(&ig_rows, row, vec![1, 1]);
                let ih = rv(&ih_rows, row, vec![1, 1, HEADS]);
                let o = rv(&out_rows, row, vec![1, 1, HEADS, DIM]);
                s.barrier_between(&[&q, &k, &x.sinks, &i], &[&g, &m, &ig, &ih, &o]);
                dispatch_deepseek_sparse_attention_flash_prefill(
                    s.encoder_mut(),
                    r,
                    d,
                    &q,
                    &k,
                    &x.sinks,
                    &i,
                    &g,
                    &m,
                    &ig,
                    &ih,
                    &o,
                    &sparse_params(1, kv_len, width),
                )?;
            }
            Ok(())
        };
        let batched = |s: &mut GraphSession<'_>,
                       r: &mut KernelRegistry,
                       d: &MlxDevice,
                       c: usize|
         -> Result<()> {
            let kv = &x.kv[c % x.kv.len()];
            s.barrier_between(
                &[&x.q, kv, &x.sinks, &x.indices],
                &[&g_b, &m_b, &ig_b, &ih_b, &out_b],
            );
            dispatch_deepseek_sparse_attention_flash_prefill(
                s.encoder_mut(),
                r,
                d,
                &x.q,
                kv,
                &x.sinks,
                &x.indices,
                &g_b,
                &m_b,
                &ig_b,
                &ih_b,
                &out_b,
                &sparse_params(b, kv_len, width),
            )?;
            Ok(())
        };
        let pr = per_row;
        let ba = batched;
        run_once(ctx, |s, r, d| pr(s, r, d, 0));
        run_once(ctx, |s, r, d| ba(s, r, d, 0));
        let equal = bytes(&out_rows) == bytes(&out_b);
        let base = json!({"family": "batch_probe", "op": "csa_attention", "kernel": "sparse_flash",
                          "b": b, "kv_len": kv_len, "width": width, "copies": copies, "warm": false});
        probe(ctx, out, &base, equal, pr, ba);
    }
}

fn indexer_params(queries: usize, keys: usize, start_pos: usize) -> DeepSeekIndexerParams {
    DeepSeekIndexerParams {
        batch: 1,
        query_len: queries as u32,
        kv_len: keys as u32,
        start_pos: start_pos as u32,
        ratio: 4,
        heads: 64,
        head_dim: 128,
        top_k: 512,
        offset: 128,
    }
}

/// CSA indexer select: score (MMA) + block sort + merges + finalize.
fn indexer_cells(ctx: &mut GpuContext, out: &mut Out, keys: usize) {
    let device = ctx.device().clone();
    let (stride, column) = (640usize, 128usize);
    // Row r sits at position start + r and sees min(keys, (start + r + 1) / 4) = keys - 1 or keys compressed rows;
    // the kernel requires (start + query_len) / 4 <= keys for the whole batch.
    let start = 4 * keys - MAX_ROWS;
    for b in 1..=MAX_ROWS {
        let copies = copies_for(keys * 128 * 2, false);
        let kvs: Vec<MlxBuffer> = (0..copies)
            .map(|c| bf16_buf(&device, vec![1, keys, 128], 300 + c as u64, 0.5))
            .collect();
        let q = bf16_buf(&device, vec![1, b, 64, 128], 21, 1.0);
        let w = f32_buf(&device, vec![1, b, 64], 22, 1.0 / (64.0f32 * 128.0).sqrt());
        let scratch = |rows: usize| {
            let plan = deepseek_indexer_query_tile_plan(
                &indexer_params(rows, keys, start),
                stride,
                column,
            )
            .unwrap();
            (
                zeros(&device, DType::F32, vec![1, plan.workspace_rows, keys]),
                zeros(
                    &device,
                    DType::I32,
                    vec![1, plan.workspace_rows, plan.topk_row_elements],
                ),
            )
        };
        let row_scratch: Vec<(MlxBuffer, MlxBuffer)> = (0..b).map(|_| scratch(1)).collect();
        let (sc_b, tk_b) = scratch(b);
        let out_rows = zeros(&device, DType::I32, vec![1, b, stride]);
        let out_b = zeros(&device, DType::I32, vec![1, b, stride]);
        let per_row = |s: &mut GraphSession<'_>,
                       r: &mut KernelRegistry,
                       d: &MlxDevice,
                       c: usize|
         -> Result<()> {
            let kv = &kvs[c % kvs.len()];
            for row in 0..b {
                let qr = q
                    .slice_view((row * 64 * 128 * 2) as u64, 64 * 128)
                    .with_shape(vec![1, 1, 64, 128])?;
                let wr = w
                    .slice_view((row * 64 * 4) as u64, 64)
                    .with_shape(vec![1, 1, 64])?;
                let orow = out_rows
                    .slice_view((row * stride * 4) as u64, stride)
                    .with_shape(vec![1, 1, stride])?;
                let (sc, tk) = &row_scratch[row];
                s.barrier_between(&[&qr, kv, &wr], &[sc, tk, &orow]);
                dispatch_deepseek_indexer_into_mma_tiled(
                    s.encoder_mut(),
                    r,
                    d,
                    &qr,
                    kv,
                    &wr,
                    sc,
                    tk,
                    &orow,
                    stride,
                    column,
                    &indexer_params(1, keys, start + row),
                )?;
            }
            Ok(())
        };
        let batched = |s: &mut GraphSession<'_>,
                       r: &mut KernelRegistry,
                       d: &MlxDevice,
                       c: usize|
         -> Result<()> {
            let kv = &kvs[c % kvs.len()];
            s.barrier_between(&[&q, kv, &w], &[&sc_b, &tk_b, &out_b]);
            dispatch_deepseek_indexer_into_mma_tiled(
                s.encoder_mut(),
                r,
                d,
                &q,
                kv,
                &w,
                &sc_b,
                &tk_b,
                &out_b,
                stride,
                column,
                &indexer_params(b, keys, start),
            )?;
            Ok(())
        };
        let pr = per_row;
        let ba = batched;
        run_once(ctx, |s, r, d| pr(s, r, d, 0));
        run_once(ctx, |s, r, d| ba(s, r, d, 0));
        let equal = bytes(&out_rows) == bytes(&out_b);
        let base = json!({"family": "batch_probe", "op": "csa_indexer_select", "kernel": "indexer_into_mma_tiled",
                          "b": b, "keys": keys, "copies": copies, "warm": false,
                          "note": "rows share one indexer cache (kernel has no per-query cache)"});
        probe(ctx, out, &base, equal, pr, ba);
    }
}

/// Compressor kernel (decode, seq_len 1): `emit` = the step completes a block.
fn compressor_cells(ctx: &mut GpuContext, out: &mut Out, name: &str, ratio: usize, dim: usize) {
    let device = ctx.device().clone();
    let coff = if ratio == 4 { 2 } else { 1 };
    let projected = coff * dim;
    for emit in [false, true] {
        // start_pos = position of this token; it emits when (pos + 1) % ratio == 0.
        let pos = if emit { 4096 - 1 } else { 4096 - 2 };
        for b in 1..=MAX_ROWS {
            let params = |batch: usize| DeepSeekCompressorParams {
                batch: batch as u32,
                seq_len: 1,
                start_pos: pos as u32,
                ratio: ratio as u32,
                head_dim: dim as u32,
                cache_len: 8192,
                epsilon: 1e-6,
                write_cache: 0,
            };
            let slots = params(b).output_slots();
            let state_shape = vec![b, coff * ratio, projected];
            let kv = f32_buf(&device, vec![b, 1, projected], 31, 1.0);
            let score = f32_buf(&device, vec![b, 1, projected], 32, 1.0);
            let ape = f32_buf(&device, vec![ratio, projected], 33, 0.1);
            let mut norm = alloc(&device, DType::F32, vec![dim], "probe norm").unwrap();
            norm.as_mut_slice::<f32>().unwrap().fill(1.0);
            let init_kv = f32_buf(&device, state_shape.clone(), 34, 1.0);
            let init_score = f32_buf(&device, state_shape.clone(), 35, 1.0);
            let (init_kv, init_score) = (bytes(&init_kv), bytes(&init_score));
            let st = |init: &[u8]| {
                let mut buf =
                    alloc(&device, DType::F32, state_shape.clone(), "probe state").unwrap();
                copy_into(&mut buf, init);
                buf
            };
            let (kv_rows, score_rows) = (st(&init_kv), st(&init_score));
            let (kv_b, score_b) = (st(&init_kv), st(&init_score));
            let out_rows = zeros(&device, DType::BF16, vec![b, slots, dim]);
            let out_b = zeros(&device, DType::BF16, vec![b, slots, dim]);
            let per_row = |s: &mut GraphSession<'_>,
                           r: &mut KernelRegistry,
                           d: &MlxDevice,
                           _c: usize|
             -> Result<()> {
                for row in 0..b {
                    let k = rv(&kv, row, vec![1, 1, projected]);
                    let sc = rv(&score, row, vec![1, 1, projected]);
                    let ks = rv(&kv_rows, row, vec![1, coff * ratio, projected]);
                    let ss = rv(&score_rows, row, vec![1, coff * ratio, projected]);
                    let o = rv(&out_rows, row, vec![1, slots, dim]);
                    s.barrier_between(&[&k, &sc, &ape, &norm, &ks, &ss], &[&ks, &ss, &o]);
                    dispatch_deepseek_compressor(
                        s.encoder_mut(),
                        r,
                        d,
                        &k,
                        &sc,
                        &ape,
                        &norm,
                        &ks,
                        &ss,
                        &o,
                        &o,
                        &params(1),
                    )?;
                }
                Ok(())
            };
            let batched = |s: &mut GraphSession<'_>,
                           r: &mut KernelRegistry,
                           d: &MlxDevice,
                           _c: usize|
             -> Result<()> {
                s.barrier_between(
                    &[&kv, &score, &ape, &norm, &kv_b, &score_b],
                    &[&kv_b, &score_b, &out_b],
                );
                dispatch_deepseek_compressor(
                    s.encoder_mut(),
                    r,
                    d,
                    &kv,
                    &score,
                    &ape,
                    &norm,
                    &kv_b,
                    &score_b,
                    &out_b,
                    &out_b,
                    &params(b),
                )?;
                Ok(())
            };
            let pr = per_row;
            let ba = batched;
            // One step from the same initial state: outputs and updated states must match.
            run_once(ctx, |s, r, d| pr(s, r, d, 0));
            run_once(ctx, |s, r, d| ba(s, r, d, 0));
            let equal = bytes(&out_rows) == bytes(&out_b)
                && bytes(&kv_rows) == bytes(&kv_b)
                && bytes(&score_rows) == bytes(&score_b);
            let base = json!({"family": "batch_probe", "op": format!("{name}_compressor"), "kernel": "deepseek_compressor",
                              "b": b, "ratio": ratio, "head_dim": dim, "emit": emit, "position": pos,
                              "copies": 1, "warm": true});
            probe(ctx, out, &base, equal, pr, ba);
        }
    }
}

#[test]
#[ignore = "model-free GPU microbenchmarks (gap program M2c; standing grant, <= 15 min)"]
fn ds4_batch_probe() {
    let out_path = std::env::var("HF2Q_DS4_BATCH_PROBE_OUT").expect("set HF2Q_DS4_BATCH_PROBE_OUT");
    let mut out = Out(std::fs::File::create(&out_path).unwrap());
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();

    gpu_warmup(&mut ctx);
    floor_cells(&mut ctx, &mut out);
    // Shapes of M2b's context-2048 and 8192 cases.
    score_reduce_cells(&mut ctx, &mut out, "window", 128, 128);
    score_reduce_cells(&mut ctx, &mut out, "hca", 128 + 16, 144);
    score_reduce_cells(&mut ctx, &mut out, "hca", 128 + 64, 192);
    flash_cells(&mut ctx, &mut out, 128 + 512, 640);
    flash_cells(&mut ctx, &mut out, 128 + 2048, 640);
    indexer_cells(&mut ctx, &mut out, 512);
    indexer_cells(&mut ctx, &mut out, 2048);
    compressor_cells(&mut ctx, &mut out, "csa_main", 4, 512);
    compressor_cells(&mut ctx, &mut out, "csa_indexer", 4, 128);
    compressor_cells(&mut ctx, &mut out, "hca_main", 128, 512);
    eprintln!("cells: {out_path}");
}
