//! Fix-1 (tail rows step) microbench: the cache-dependent cores and HC row ops of 8 token-wise
//! prompt tokens as 8 one-token steps vs one rows step, at the ds4 0731 file's shapes.
//!
//! Model-free (synthetic caches and weights, `ds4_attn_microbench_tests` helpers). Core cells time the
//! production `plan_row_cores` / `encode_row_cores`:
//! - `serial_x8`: 8 rows, each encoded alone (`RowCaches::Independent`, one row) at its own position,
//!   a full barrier between tokens (the GPU work of 8 one-token steps; their command-buffer sync gaps
//!   are not in the GPU time and are added from the floor cells by the aggregator);
//! - `rows_step`: the same 8 positions as one `RowCaches::Sequential` step on one cache.
//!
//! Every copy rotates over all layers of the kind (cold weights, as a step reads each layer once).
//! HC row ops come from `hc_cells` at B = 1 and 8. The products (Q8_0, F16, experts) come from the M2a
//! bench with `HF2Q_DS4_OP_BENCH_ROWS=1,8`. Both benches follow `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC`.
//!
//! Run (standing grant, model-free, <= 15 min):
//! ```text
//! HF2Q_DS4_TAIL_ROWS_BENCH_OUT=<cells.jsonl> cargo test --release --bin hf2q ds4_tail_rows_microbench -- --ignored --nocapture
//! ```

use mlx_native::{DType, MlxBuffer, MlxDevice};
use serde_json::json;

use super::cache::{CacheStep, Deepseek4Cache, Deepseek4CachePlan};
use super::cache_tests::config;
use super::ds4_attn_microbench_tests::{
    bf16_buf, ds4_ratios, hc_cells, host_table, synthetic_cache, LayerWeights, Unused,
};
use super::ds4_bench_support::{bench_matvec, f32_buf, floor_cells, gpu_warmup, time_op, Out};
use super::forward_support::{alloc, alloc_host_input};
use super::joint_decode::MAX_JOINT_DECODE_ROWS;
use super::joint_decode_core::{
    encode_row_cores, plan_row_cores, JointRowCore, JointRowInputs, RowCaches,
};
use super::rope::yarn_frequencies;
use super::row_matmul::row_view;
use super::Deepseek4Config;
use crate::serve::gpu::GpuContext;

const CONTEXT: usize = 4096;
/// First tail position: the run's prompts put the 8-token tail at 749..1919.
const START: usize = 1000;
const ROWS: usize = MAX_JOINT_DECODE_ROWS;

struct RowInputs {
    _batched: Vec<MlxBuffer>,
    rows: Vec<JointRowInputs>,
}

fn row_inputs(device: &MlxDevice, cfg: &Deepseek4Config) -> RowInputs {
    let heads = cfg.num_attention_heads as usize;
    let head_dim = cfg.head_dim as usize;
    let hidden = cfg.hidden_size as usize;
    let q_rank = cfg.q_lora_rank as usize;
    let q_rope = bf16_buf(device, vec![1, ROWS, heads, head_dim], 71, 1.0);
    let kv_rope = bf16_buf(device, vec![1, ROWS, 1, head_dim], 72, 0.5);
    let attention = alloc(device, DType::BF16, vec![1, ROWS, heads, head_dim], "out").unwrap();
    let attn_norm = f32_buf(device, vec![ROWS, hidden], 73, 1.0);
    let q_a_norm = f32_buf(device, vec![ROWS, q_rank], 74, 1.0);
    let mut positions = alloc_host_input(device, DType::U32, vec![ROWS], "positions").unwrap();
    for (row, value) in positions
        .as_logical_mut_slice::<u32>()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *value = (START + row) as u32;
    }
    let rows = (0..ROWS)
        .map(|row| JointRowInputs {
            q_rope: row_view(&q_rope, row, vec![1, 1, heads, head_dim]).unwrap(),
            kv_rope: row_view(&kv_rope, row, vec![1, 1, 1, head_dim]).unwrap(),
            attention: row_view(&attention, row, vec![1, 1, heads, head_dim]).unwrap(),
            attn_norm: row_view(&attn_norm, row, vec![1, hidden]).unwrap(),
            q_a_norm: row_view(&q_a_norm, row, vec![1, q_rank]).unwrap(),
            position: row_view(&positions, row, vec![1]).unwrap(),
        })
        .collect();
    RowInputs {
        _batched: vec![q_rope, kv_rope, attention, attn_norm, q_a_norm, positions],
        rows,
    }
}

#[allow(clippy::too_many_arguments)]
fn core_cells(
    ctx: &mut GpuContext,
    out: &mut Out,
    cfg: &Deepseek4Config,
    cache: &Deepseek4Cache,
    steps: &[CacheStep],
    weights: &[LayerWeights],
    unused: &Unused,
    frequencies: (&MlxBuffer, &MlxBuffer),
    inputs: &RowInputs,
) {
    let device = ctx.device().clone();
    for (kind, ratio) in [("window", 0u32), ("hca", 128), ("csa", 4)] {
        let layers = cfg
            .compress_ratios
            .iter()
            .enumerate()
            .filter(|(_, &r)| r == ratio)
            .map(|(layer, _)| layer)
            .collect::<Vec<_>>();
        let n = layers.len();
        let frequency = if ratio == 0 { frequencies.0 } else { frequencies.1 };
        let refs = layers
            .iter()
            .map(|&layer| (ratio != 0).then(|| weights[layer].compressed(unused)))
            .collect::<Vec<_>>();
        let caches = vec![cache; ROWS];
        // Per layer: 8 one-row plans (serial) and one 8-row plan (rows step).
        let serial: Vec<Vec<Vec<JointRowCore>>> = layers
            .iter()
            .map(|&layer| {
                (0..ROWS)
                    .map(|row| {
                        plan_row_cores(&device, cfg, &[cache], &steps[row..=row], layer, ratio)
                            .unwrap()
                    })
                    .collect()
            })
            .collect();
        let stepped: Vec<Vec<JointRowCore>> = layers
            .iter()
            .map(|&layer| plan_row_cores(&device, cfg, &caches, steps, layer, ratio).unwrap())
            .collect();
        let base = json!({"family": "tail_core", "kind": kind, "ratio": ratio, "rows": ROWS,
                          "start": START, "layers": n, "matvec": format!("{:?}", bench_matvec())});

        let t = time_op(ctx, |s, r, d, rep| {
            let i = rep % n;
            for row in 0..ROWS {
                s.barrier();
                encode_row_cores(
                    s,
                    r,
                    d,
                    cfg,
                    &[cache],
                    &steps[row..=row],
                    layers[i],
                    &serial[i][row],
                    std::slice::from_ref(&inputs.rows[row]),
                    &weights[layers[i]].sinks,
                    refs[i].as_ref(),
                    frequency,
                    RowCaches::Independent,
                )?;
            }
            Ok(())
        });
        out.cell(&base, json!({"mode": "serial_x8"}), &t);

        let t = time_op(ctx, |s, r, d, rep| {
            let i = rep % n;
            s.barrier();
            encode_row_cores(
                s,
                r,
                d,
                cfg,
                &caches,
                steps,
                layers[i],
                &stepped[i],
                &inputs.rows,
                &weights[layers[i]].sinks,
                refs[i].as_ref(),
                frequency,
                RowCaches::Sequential,
            )
        });
        out.cell(&base, json!({"mode": "rows_step"}), &t);
    }
}

#[test]
#[ignore = "model-free GPU microbenchmarks (gap program fix 1; standing grant, <= 15 min)"]
fn ds4_tail_rows_microbench() {
    let out_path =
        std::env::var("HF2Q_DS4_TAIL_ROWS_BENCH_OUT").expect("set HF2Q_DS4_TAIL_ROWS_BENCH_OUT");
    let mut out = Out(std::fs::File::create(&out_path).unwrap());
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let cfg = config(ds4_ratios());

    gpu_warmup(&mut ctx);
    floor_cells(&mut ctx, &mut out);
    hc_cells(&mut ctx, &mut out, &cfg, &[1, ROWS]);

    let plan = Deepseek4CachePlan::for_context(&cfg, CONTEXT).unwrap();
    let mut cache = synthetic_cache(&device, &plan, 1);
    for position in 0..START {
        cache.commit_step(position).unwrap();
    }
    let steps = (START..START + ROWS)
        .map(|position| cache.plan_step_at(position).unwrap())
        .collect::<Vec<_>>();
    let weights = cfg
        .compress_ratios
        .iter()
        .enumerate()
        .map(|(layer, &ratio)| LayerWeights::new(&device, &cfg, ratio, layer as u64 + 1))
        .collect::<Vec<_>>();
    let unused = Unused {
        buffer: alloc(&device, DType::F32, vec![1], "unused").unwrap(),
        shape: [1, 1],
    };
    let rope = cfg.rope_head_dim as usize;
    let window = host_table(
        &device,
        yarn_frequencies(rope, 0, cfg.rope_theta, 1.0, cfg.yarn_beta_fast, cfg.yarn_beta_slow)
            .unwrap(),
    );
    let compressed = host_table(
        &device,
        yarn_frequencies(
            rope,
            cfg.original_context_length as usize,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.yarn_beta_fast,
            cfg.yarn_beta_slow,
        )
        .unwrap(),
    );
    let inputs = row_inputs(&device, &cfg);
    core_cells(
        &mut ctx,
        &mut out,
        &cfg,
        &cache,
        &steps,
        &weights,
        &unused,
        (&window, &compressed),
        &inputs,
    );
    eprintln!("cells: {out_path}");
}
