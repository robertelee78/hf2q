//! DeepSeek-V4 gap program M2a: hf2q per-op microbenchmarks at the ds4 0731 file's decode shapes, B = 1..4 rows.
//!
//! Model-free: synthetic weights with in-range scales, no model runtime. Each cell times hf2q's own dispatch
//! wrappers (`raw_matmul`, `row_matmul(IndependentRows)`, `grouped_output_a_batched`, `expert_matmul` forced
//! `mv_id`) exactly as joint decode issues them. One trial encodes `REPS` back-to-back copies of the op in one
//! session and times `finish()` (commit + wait); the cell is the median of `TRIALS` trials divided by `REPS`.
//! Weights rotate over enough copies to exceed 256 MiB, so each repetition reads cold weights from DRAM,
//! as one layer of a decode step does; `warm` cells reuse one copy.
//!
//! Run (standing grant, model-free, <= 15 min):
//! ```text
//! HF2Q_DS4_MICROBENCH_OUT=<cells.jsonl> cargo test --release --bin hf2q ds4_op_microbench -- --ignored --nocapture
//! ```

use std::io::Write;
use std::time::Instant;

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, MlxBuffer, MlxDevice};
use serde_json::json;

use super::forward_support::{
    alloc, expert_matmul, grouped_output_a_batched, raw_matmul, BatchedGroupedOutputArena,
    ExpertMatmulRoute,
};
use super::residency::RawMatrixRef;
use super::row_matmul::{row_matmul, RowMatmul};
use crate::serve::gpu::GpuContext;

const TRIALS: usize = 9;
const REPS: usize = 16;
const COLD_BYTES: usize = 256 << 20;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Random block bytes with every block's f16 scale fields set to small normal values.
fn block_weight(
    device: &MlxDevice,
    ggml_type: GgmlType,
    shape: Vec<usize>,
    seed: u64,
) -> MlxBuffer {
    let k = *shape.last().unwrap();
    let rows: usize = shape[..shape.len() - 1].iter().product();
    let bb = ggml_type.block_bytes() as usize;
    let bytes = rows * k / ggml_type.block_values() as usize * bb;
    let mut buf = device.alloc_buffer(bytes, DType::U8, shape).unwrap();
    let data = buf.as_mut_slice::<u8>().unwrap();
    let mut rng = Rng(seed);
    for chunk in data.chunks_mut(8) {
        let w = rng.next().to_le_bytes();
        chunk.copy_from_slice(&w[..chunk.len()]);
    }
    let scale = half::f16::from_f32(1.0 / 512.0).to_le_bytes();
    for block in data.chunks_exact_mut(bb) {
        match ggml_type {
            GgmlType::Q8_0 | GgmlType::IQ2_XXS => block[..2].copy_from_slice(&scale),
            GgmlType::Q2_K => {
                block[80..82].copy_from_slice(&scale);
                block[82..84].copy_from_slice(&scale);
            }
            other => panic!("unsupported bench type {other:?}"),
        }
    }
    buf
}

fn f16_weight(device: &MlxDevice, n: usize, k: usize, seed: u64) -> MlxBuffer {
    let mut buf = device
        .alloc_buffer(n * k * 2, DType::F16, vec![n, k])
        .unwrap();
    let mut rng = Rng(seed);
    for v in buf.as_mut_slice::<u16>().unwrap() {
        *v = half::f16::from_f32(((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05)
            .to_bits();
    }
    buf
}

fn input(device: &MlxDevice, shape: Vec<usize>, seed: u64) -> MlxBuffer {
    let mut buf = alloc(device, DType::F32, shape, "bench input").unwrap();
    let mut rng = Rng(seed);
    for v in buf.as_mut_slice::<f32>().unwrap() {
        *v = ((rng.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
    }
    buf
}

fn copies_for(bytes: usize, warm: bool) -> usize {
    if warm {
        1
    } else {
        COLD_BYTES.div_ceil(bytes).clamp(1, 64)
    }
}

/// Median seconds per op over TRIALS trials of REPS encodes each.
fn time_op(
    ctx: &mut GpuContext,
    mut encode: impl FnMut(
        &mut mlx_native::graph::GraphSession<'_>,
        &mut mlx_native::KernelRegistry,
        &MlxDevice,
        usize,
    ),
) -> f64 {
    let device = ctx.device().clone();
    let mut samples = Vec::with_capacity(TRIALS + 1);
    for trial in 0..=TRIALS {
        let (executor, registry) = ctx.split();
        let mut session = executor.begin().unwrap();
        for rep in 0..REPS {
            encode(&mut session, registry, &device, rep);
        }
        let t0 = Instant::now();
        session.finish().unwrap();
        let dt = t0.elapsed().as_secs_f64();
        if trial > 0 {
            samples.push(dt / REPS as f64); // trial 0 warms pipelines
        }
    }
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

struct Out(std::fs::File);
impl Out {
    fn cell(&mut self, cell: serde_json::Value) {
        eprintln!("{cell}");
        writeln!(self.0, "{cell}").unwrap();
    }
}

#[test]
#[ignore = "model-free GPU microbenchmarks (gap program M2a; standing grant, <= 15 min)"]
fn ds4_op_microbench() {
    let out_path = std::env::var("HF2Q_DS4_MICROBENCH_OUT").expect("set HF2Q_DS4_MICROBENCH_OUT");
    let mut out = Out(std::fs::File::create(&out_path).unwrap());
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();

    // ── dense Q8_0 projections: batched raw_matmul at m = B, vs B one-row calls ─────────────
    let dense_q8: [(&str, usize, usize); 6] = [
        ("attn_q_a", 1024, 4096),
        ("attn_kv", 512, 4096),
        ("attn_q_b", 32768, 1024),
        ("attn_output_b", 4096, 8192),
        ("shexp_gate_up", 2048, 4096),
        ("shexp_down", 4096, 2048),
    ];
    for (name, n, k) in dense_q8 {
        for warm in [false, true] {
            let bytes = n * k / 32 * 34;
            let copies = copies_for(bytes, warm);
            let weights: Vec<MlxBuffer> = (0..copies)
                .map(|c| block_weight(&device, GgmlType::Q8_0, vec![n, k], c as u64))
                .collect();
            let shape = [n, k];
            for b in 1..=4usize {
                let x = input(&device, vec![b, k], 7);
                let y = alloc(&device, DType::F32, vec![b, n], "bench out").unwrap();
                for mode in ["batched", "per_row"] {
                    let secs = time_op(&mut ctx, |s, r, d, rep| {
                        let w = RawMatrixRef {
                            buffer: &weights[rep % copies],
                            ggml_type: GgmlType::Q8_0,
                            shape: &shape,
                        };
                        if mode == "batched" {
                            raw_matmul(s, r, d, &x, &w, &y, b, n, k, name).unwrap();
                        } else {
                            for row in 0..b {
                                let xr = super::row_matmul::row_view(&x, row, vec![1, k]).unwrap();
                                let yr = super::row_matmul::row_view(&y, row, vec![1, n]).unwrap();
                                raw_matmul(s, r, d, &xr, &w, &yr, 1, n, k, name).unwrap();
                            }
                        }
                    });
                    out.cell(json!({"family": "dense_q8_0", "op": name, "n": n, "k": k, "b": b, "mode": mode,
                                    "warm": warm, "us": secs * 1e6, "weight_bytes": bytes,
                                    "gbps": bytes as f64 / secs / 1e9}));
                }
            }
        }
    }

    // ── output head Q8_0 [129280, 4096] (forward_logits raw_matmul at m = B) ─────────────────────────
    {
        let (n, k) = (129_280usize, 4096usize);
        let w = block_weight(&device, GgmlType::Q8_0, vec![n, k], 99);
        let shape = [n, k];
        for b in 1..=4usize {
            let x = input(&device, vec![b, k], 3);
            let y = alloc(&device, DType::F32, vec![b, n], "bench out").unwrap();
            let secs = time_op(&mut ctx, |s, r, d, _| {
                let wr = RawMatrixRef {
                    buffer: &w,
                    ggml_type: GgmlType::Q8_0,
                    shape: &shape,
                };
                raw_matmul(s, r, d, &x, &wr, &y, b, n, k, "output head").unwrap();
            });
            out.cell(json!({"family": "dense_q8_0", "op": "output_head", "n": n, "k": k, "b": b, "mode": "batched",
                            "warm": false, "us": secs * 1e6, "weight_bytes": n * k / 32 * 34}));
        }
    }

    // ── grouped output_a Q8_0: 8 groups of [1024, 4096] (grouped_output_a_batched at rows = B) ─────────
    {
        let (groups, rank, heads, head_dim) = (8usize, 1024usize, 64usize, 512usize);
        let gw = heads * head_dim / groups;
        let bytes = groups * rank * gw / 32 * 34;
        let copies = copies_for(bytes, false);
        let weights: Vec<MlxBuffer> = (0..copies)
            .map(|c| {
                block_weight(
                    &device,
                    GgmlType::Q8_0,
                    vec![groups * rank, gw],
                    50 + c as u64,
                )
            })
            .collect();
        let shape = [groups * rank, gw];
        for b in 1..=4usize {
            let x = input(&device, vec![b, heads * head_dim], 5);
            let y = alloc(&device, DType::F32, vec![b, groups * rank], "bench out").unwrap();
            let arena = BatchedGroupedOutputArena::new(&device, b, groups, gw, rank).unwrap();
            let secs = time_op(&mut ctx, |s, r, d, rep| {
                let w = RawMatrixRef {
                    buffer: &weights[rep % copies],
                    ggml_type: GgmlType::Q8_0,
                    shape: &shape,
                };
                grouped_output_a_batched(
                    s, r, d, &x, &w, &y, &arena, b, groups, rank, heads, head_dim,
                )
                .unwrap();
            });
            out.cell(json!({"family": "dense_q8_0", "op": "attn_output_a_grouped", "n": groups * rank, "k": gw,
                            "b": b, "mode": "batched", "warm": false, "us": secs * 1e6, "weight_bytes": bytes,
                            "gbps": bytes as f64 / secs / 1e9}));
        }
    }

    // ── F16 products issued per row in joint decode (row_matmul IndependentRows; m = 1 is the L9 matvec) ──
    let f16_rows: [(&str, usize, usize); 7] = [
        ("hc_fn", 24, 16384),
        ("output_hc_fn", 4, 16384),
        ("router", 256, 4096),
        ("compressor_r4", 1024, 4096),
        ("compressor_r128", 512, 4096),
        ("indexer_q_b", 8192, 1024),
        ("indexer_proj", 64, 4096),
    ];
    for (name, n, k) in f16_rows {
        let bytes = n * k * 2;
        let copies = copies_for(bytes, false);
        let weights: Vec<MlxBuffer> = (0..copies)
            .map(|c| f16_weight(&device, n, k, 200 + c as u64))
            .collect();
        let shape = [n, k];
        for b in 1..=4usize {
            let x = input(&device, vec![b, k], 11);
            let y = alloc(&device, DType::F32, vec![b, n], "bench out").unwrap();
            let secs = time_op(&mut ctx, |s, r, d, rep| {
                let w = RawMatrixRef {
                    buffer: &weights[rep % copies],
                    ggml_type: GgmlType::F16,
                    shape: &shape,
                };
                row_matmul(
                    RowMatmul::IndependentRows,
                    s,
                    r,
                    d,
                    &x,
                    &w,
                    &y,
                    b,
                    n,
                    k,
                    name,
                )
                .unwrap();
            });
            out.cell(json!({"family": "f16_row", "op": name, "n": n, "k": k, "b": b, "mode": "independent_rows",
                            "warm": false, "us": secs * 1e6, "weight_bytes": bytes,
                            "gbps": (bytes * b) as f64 / secs / 1e9}));
        }
    }

    // ── routed experts, forced mv_id as joint decode issues them (gate/up: n_tokens = B, top-6; down: 6B rows) ──
    let experts: [(&str, GgmlType, usize, usize, bool); 2] = [
        (
            "expert_gate_or_up_iq2_xxs",
            GgmlType::IQ2_XXS,
            2048,
            4096,
            true,
        ),
        ("expert_down_q2_k", GgmlType::Q2_K, 4096, 2048, false),
    ];
    for (name, ty, n, k, broadcast) in experts {
        let w = block_weight(&device, ty, vec![256, n, k], 300);
        let shape = [256usize, n, k];
        for b in 1..=4usize {
            let mut ids = alloc(&device, DType::U32, vec![b * 6], "bench ids").unwrap();
            let mut rng = Rng(b as u64);
            let idv = ids.as_mut_slice::<u32>().unwrap();
            for t in 0..b {
                let mut chosen = Vec::new();
                while chosen.len() < 6 {
                    let e = (rng.next() % 256) as u32;
                    if !chosen.contains(&e) {
                        chosen.push(e);
                    }
                }
                idv[t * 6..t * 6 + 6].copy_from_slice(&chosen);
            }
            let (n_tokens, top_k, x) = if broadcast {
                (b, 6usize, input(&device, vec![b, k], 13))
            } else {
                (b * 6, 1usize, input(&device, vec![b * 6, k], 13))
            };
            let y = alloc(&device, DType::F32, vec![b * 6, n], "bench out").unwrap();
            let secs = time_op(&mut ctx, |s, r, d, _| {
                let wr = RawMatrixRef {
                    buffer: &w,
                    ggml_type: ty,
                    shape: &shape,
                };
                expert_matmul(
                    s,
                    r,
                    d,
                    &x,
                    &wr,
                    &ids,
                    &y,
                    n_tokens,
                    top_k,
                    256,
                    n,
                    k,
                    ExpertMatmulRoute::ForceMv,
                    None,
                    name,
                )
                .unwrap();
            });
            let expert_bytes = n * k / 256 * ty.block_bytes() as usize;
            out.cell(
                json!({"family": "expert_mv_id", "op": name, "n": n, "k": k, "b": b, "rows": b * 6,
                            "mode": "force_mv", "warm": false, "us": secs * 1e6,
                            "weight_bytes_read_max": expert_bytes * b * 6}),
            );
        }
    }
    eprintln!("cells: {out_path}");
}
