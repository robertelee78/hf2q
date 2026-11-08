//! DeepSeek-V4 gap program M2a: hf2q per-op microbenchmarks at the ds4 0731 file's decode shapes, B = 1..4 rows.
//!
//! Model-free: synthetic weights with in-range scales, no model runtime. Each cell times hf2q's own dispatch
//! wrappers (`raw_matmul`, `row_matmul(IndependentRows)`, `grouped_output_a_batched`, `expert_matmul` forced
//! `mv_id`) exactly as joint decode issues them. Timing is `ds4_bench_support` (shared with M2b): GPU
//! command-buffer timestamps, copies per session calibrated for >= 2 ms of GPU work, median of 9 trials per
//! copy, `wall_us` and `encode_us` per copy alongside, a 2 s warm-up first and >= 50 ms of untimed sessions
//! before every cell; the `floor` cells record the method's floor in every run.
//!
//! Every family is timed warm (one weight copy, or for routed experts one fixed expert set) and cold:
//! dense weights rotate as `copies[rep % copies]` over enough copies to exceed 256 MiB (at most 64, so every
//! copy is read in every session); routed experts rotate over as many distinct expert-id sets. Tensors
//! larger than 256 MiB (output head) have one copy, so their warm and cold cells read the same bytes.
//!
//! `HF2Q_DS4_OP_BENCH_ROWS` (comma list) replaces B = 1..4 (the tail rows step uses `1,8`).
//!
//! Run (standing grant, model-free, <= 15 min):
//! ```text
//! HF2Q_DS4_MICROBENCH_OUT=<cells.jsonl> cargo test --release --bin hf2q ds4_op_microbench -- --ignored --nocapture
//! ```

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, MlxBuffer, MlxDevice};
use serde_json::json;

use super::ds4_bench_support::{
    bench_matvec, copies_for, f16_matrix, f32_buf, floor_cells, gpu_warmup, time_op, Out, Rng,
    COLD_BYTES, MIN_REPS,
};
use super::forward_support::{
    alloc, expert_matmul, grouped_output_a_batched, raw_matmul, BatchedGroupedOutputArena,
    ExpertMatmulRoute,
};
use super::residency::RawMatrixRef;
use super::row_matmul::{row_matmul, row_view, RowMatmul};
use crate::serve::gpu::GpuContext;

/// Random block bytes with every block's f16 scale fields set to small normal values.
pub(super) fn block_weight(
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

fn input(device: &MlxDevice, shape: Vec<usize>, seed: u64) -> MlxBuffer {
    f32_buf(device, shape, seed, 1.0)
}

/// `top_k` distinct expert ids per token for `tokens` tokens.
fn expert_ids(device: &MlxDevice, tokens: usize, top_k: usize, seed: u64) -> MlxBuffer {
    let mut ids = alloc(device, DType::U32, vec![tokens * top_k], "bench ids").unwrap();
    let mut rng = Rng(seed);
    let idv = ids.as_mut_slice::<u32>().unwrap();
    for t in 0..tokens {
        let mut chosen = Vec::with_capacity(top_k);
        while chosen.len() < top_k {
            let e = (rng.next() % 256) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        idv[t * top_k..(t + 1) * top_k].copy_from_slice(&chosen);
    }
    ids
}

/// Row counts B: `HF2Q_DS4_OP_BENCH_ROWS` (comma list, e.g. `1,8` for the tail rows step), else 1..=4.
fn op_rows() -> Vec<usize> {
    match std::env::var("HF2Q_DS4_OP_BENCH_ROWS") {
        Ok(list) => list
            .split(',')
            .map(|b| b.trim().parse().expect("HF2Q_DS4_OP_BENCH_ROWS: comma list of row counts"))
            .collect(),
        Err(_) => (1..=4).collect(),
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

    gpu_warmup(&mut ctx);
    floor_cells(&mut ctx, &mut out);

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
        let bytes = n * k / 32 * 34;
        let all: Vec<MlxBuffer> = (0..copies_for(bytes, false))
            .map(|c| block_weight(&device, GgmlType::Q8_0, vec![n, k], c as u64))
            .collect();
        let shape = [n, k];
        for warm in [false, true] {
            let weights = &all[..copies_for(bytes, warm)];
            for b in op_rows() {
                let x = input(&device, vec![b, k], 7);
                let y = alloc(&device, DType::F32, vec![b, n], "bench out").unwrap();
                for mode in ["batched", "per_row"] {
                    let t = time_op(&mut ctx, |s, r, d, rep| {
                        let w = RawMatrixRef {
                            buffer: &weights[rep % weights.len()],
                            ggml_type: GgmlType::Q8_0,
                            shape: &shape,
                            matvec: bench_matvec(),
                        };
                        if mode == "batched" {
                            raw_matmul(s, r, d, &x, &w, &y, b, n, k, name)
                        } else {
                            for row in 0..b {
                                let xr = row_view(&x, row, vec![1, k])?;
                                let yr = row_view(&y, row, vec![1, n])?;
                                raw_matmul(s, r, d, &xr, &w, &yr, 1, n, k, name)?;
                            }
                            Ok(())
                        }
                    });
                    out.cell(
                        &json!({"family": "dense_q8_0", "op": name, "n": n, "k": k, "b": b, "mode": mode,
                                "warm": warm, "copies": weights.len(), "weight_bytes": bytes}),
                        json!({"gbps": bytes as f64 / t.us / 1e3}),
                        &t,
                    );
                }
            }
        }
    }

    // ── output head Q8_0 [129280, 4096] (forward_logits raw_matmul at m = B); one copy > COLD_BYTES ──
    {
        let (n, k) = (129_280usize, 4096usize);
        let bytes = n * k / 32 * 34;
        let w = block_weight(&device, GgmlType::Q8_0, vec![n, k], 99);
        let shape = [n, k];
        for warm in [false, true] {
            for b in op_rows() {
                let x = input(&device, vec![b, k], 3);
                let y = alloc(&device, DType::F32, vec![b, n], "bench out").unwrap();
                let t = time_op(&mut ctx, |s, r, d, _| {
                    let wr = RawMatrixRef {
                        buffer: &w,
                        ggml_type: GgmlType::Q8_0,
                        shape: &shape,
                        matvec: bench_matvec(),
                    };
                    raw_matmul(s, r, d, &x, &wr, &y, b, n, k, "output head")
                });
                out.cell(
                    &json!({"family": "dense_q8_0", "op": "output_head", "n": n, "k": k, "b": b,
                            "mode": "batched", "warm": warm, "copies": 1, "weight_bytes": bytes,
                            "note": "one copy exceeds COLD_BYTES; warm and cold read the same bytes"}),
                    json!({"gbps": bytes as f64 / t.us / 1e3}),
                    &t,
                );
            }
        }
    }

    // ── grouped output_a Q8_0: 8 groups of [1024, 4096] (grouped_output_a_batched at rows = B) ─────────
    {
        let (groups, rank, heads, head_dim) = (8usize, 1024usize, 64usize, 512usize);
        let gw = heads * head_dim / groups;
        let bytes = groups * rank * gw / 32 * 34;
        let all: Vec<MlxBuffer> = (0..copies_for(bytes, false))
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
        for warm in [false, true] {
            let weights = &all[..copies_for(bytes, warm)];
            for b in op_rows() {
                let x = input(&device, vec![b, heads * head_dim], 5);
                let y = alloc(&device, DType::F32, vec![b, groups * rank], "bench out").unwrap();
                let arena = BatchedGroupedOutputArena::new(&device, b, groups, gw, rank).unwrap();
                let t = time_op(&mut ctx, |s, r, d, rep| {
                    let w = RawMatrixRef {
                        buffer: &weights[rep % weights.len()],
                        ggml_type: GgmlType::Q8_0,
                        shape: &shape,
                        matvec: bench_matvec(),
                    };
                    grouped_output_a_batched(
                        s, r, d, &x, &w, &y, &arena, b, groups, rank, heads, head_dim,
                    )
                });
                out.cell(
                    &json!({"family": "dense_q8_0", "op": "attn_output_a_grouped", "n": groups * rank,
                            "k": gw, "b": b, "mode": "batched", "warm": warm, "copies": weights.len(),
                            "weight_bytes": bytes}),
                    json!({"gbps": bytes as f64 / t.us / 1e3}),
                    &t,
                );
            }
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
        let all: Vec<MlxBuffer> = (0..copies_for(bytes, false))
            .map(|c| f16_matrix(&device, n, k, 200 + c as u64))
            .collect();
        let shape = [n, k];
        for warm in [false, true] {
            let weights = &all[..copies_for(bytes, warm)];
            for b in op_rows() {
                let x = input(&device, vec![b, k], 11);
                let y = alloc(&device, DType::F32, vec![b, n], "bench out").unwrap();
                let t = time_op(&mut ctx, |s, r, d, rep| {
                    let w = RawMatrixRef {
                        buffer: &weights[rep % weights.len()],
                        ggml_type: GgmlType::F16,
                        shape: &shape,
                        matvec: bench_matvec(),
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
                });
                out.cell(
                    &json!({"family": "f16_row", "op": name, "n": n, "k": k, "b": b,
                            "mode": "independent_rows", "warm": warm, "copies": weights.len(),
                            "weight_bytes": bytes}),
                    json!({"gbps": (bytes * b) as f64 / t.us / 1e3}),
                    &t,
                );
            }
        }
    }

    // ── routed experts, forced mv_id as joint decode issues them (gate/up: n_tokens = B, top-6; down: 6B rows) ──
    // Warm: one fixed expert set per token (re-read every copy). Cold: rotate over distinct random expert sets
    // until the experts read per session exceed COLD_BYTES.
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
        let expert_bytes = n * k / ty.block_values() as usize * ty.block_bytes() as usize;
        for b in op_rows() {
            let per_copy = expert_bytes * b * 6;
            let all: Vec<MlxBuffer> = (0..copies_for(per_copy, false))
                .map(|c| expert_ids(&device, b, 6, (b * MIN_REPS + c) as u64))
                .collect();
            let (n_tokens, top_k, x) = if broadcast {
                (b, 6usize, input(&device, vec![b, k], 13))
            } else {
                (b * 6, 1usize, input(&device, vec![b * 6, k], 13))
            };
            let y = alloc(&device, DType::F32, vec![b * 6, n], "bench out").unwrap();
            for warm in [false, true] {
                let ids = &all[..copies_for(per_copy, warm)];
                let t = time_op(&mut ctx, |s, r, d, rep| {
                    let wr = RawMatrixRef {
                        buffer: &w,
                        ggml_type: ty,
                        shape: &shape,
                        matvec: bench_matvec(),
                    };
                    expert_matmul(
                        s,
                        r,
                        d,
                        &x,
                        &wr,
                        &ids[rep % ids.len()],
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
                });
                out.cell(
                    &json!({"family": "expert_mv_id", "op": name, "n": n, "k": k, "b": b, "rows": b * 6,
                            "mode": "force_mv", "warm": warm, "copies": ids.len(),
                            "weight_bytes_read_max": per_copy, "cold_bytes": COLD_BYTES}),
                    json!({}),
                    &t,
                );
            }
        }
    }
    eprintln!("cells: {out_path}");
}
