//! Model-free: the batched joint-decode cores (`joint_decode_rows`) equal the
//! per-row cores (`joint_decode_core`) bitwise.
//!
//! Each case builds two identical sets of synthetic per-row caches (the M2b
//! builders: production `Deepseek4Cache` layout, random BF16 rows, random
//! compressor states) and synthetic F16 layer weights, then advances every
//! row by the same steps through `plan_*_row`/`encode_*_row` one row at a
//! time on one set and through `plan_*_layer`/encode once for all rows on the
//! other. Rows sit at unequal positions (window widths below 128, CSA widths
//! below and above the flash threshold, one and several top-k blocks, masked
//! top-k slots, emit and non-emit compressor steps). After every step the
//! attention output, the attention index storage, and every cache buffer
//! (window/compressed KV, indexer KV, main and indexer compressor state)
//! must be byte-equal. The F16 projections take the process's matvec family
//! (`HF2Q_DEEPSEEK_ROW_EXACT_MATVEC`), so run once per family.

use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::cache::{Deepseek4Cache, Deepseek4CachePlan, LayerCache, LayerCacheStep};
use super::ds4_attn_microbench_tests::{
    bf16_buf, ds4_ratios, host_table, synthetic_cache, LayerWeights, Unused, CONTEXT,
};
use super::ds4_bench_support::{bench_matvec, f32_buf, run_once, Rng};
use super::forward_support::{alloc, alloc_host_input};
use super::joint_decode_core::{
    encode_compressed_row, encode_window_row, plan_compressed_row, plan_window_row, JointRowInputs,
};
use super::joint_decode_rows::{
    batched_cores_admit, encode_window_layer, plan_compressed_layer, plan_window_layer,
    projection_takes_single_dispatch, AttnCores, JointBatchInputs,
};
use super::rope::yarn_frequencies;
use super::row_matmul::row_view;
use super::Deepseek4Config;
use crate::serve::gpu::GpuContext;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;

fn fill_states(cache: &mut Deepseek4Cache, seed: u64) {
    let mut rng = Rng(seed);
    for lc in cache.layers_mut() {
        for state in [
            lc.main_kv_state.as_mut(),
            lc.main_score_state.as_mut(),
            lc.indexer_kv_state.as_mut(),
            lc.indexer_score_state.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            for v in state.as_mut_slice::<f32>().unwrap() {
                *v = rng.unit();
            }
        }
    }
}

fn cache_bytes(lc: &LayerCache) -> Vec<Vec<u8>> {
    [
        Some(&lc.attention_kv),
        lc.indexer_kv.as_ref(),
        lc.main_kv_state.as_ref(),
        lc.main_score_state.as_ref(),
        lc.indexer_kv_state.as_ref(),
        lc.indexer_score_state.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(|buffer| buffer.as_slice::<u8>().unwrap().to_vec())
    .collect()
}

/// One joint step's batched inputs (as `encode_joint_attention_layer` builds them).
struct StepInputs {
    q_rope: MlxBuffer,
    kv_rope: MlxBuffer,
    attn_norm: MlxBuffer,
    q_a_norm: MlxBuffer,
    positions: MlxBuffer,
}

impl StepInputs {
    fn new(device: &MlxDevice, cfg: &Deepseek4Config, positions: &[usize], seed: u64) -> Self {
        let b = positions.len();
        let heads = cfg.num_attention_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let mut buffer = alloc_host_input(device, DType::U32, vec![b], "test positions").unwrap();
        for (dst, &p) in buffer
            .as_logical_mut_slice::<u32>()
            .unwrap()
            .iter_mut()
            .zip(positions)
        {
            *dst = p as u32;
        }
        Self {
            q_rope: bf16_buf(device, vec![1, b, heads, head_dim], seed, 1.0),
            kv_rope: bf16_buf(device, vec![1, b, 1, head_dim], seed + 1, 0.5),
            attn_norm: f32_buf(device, vec![b, cfg.hidden_size as usize], seed + 2, 1.0),
            q_a_norm: f32_buf(device, vec![b, cfg.q_lora_rank as usize], seed + 3, 1.0),
            positions: buffer,
        }
    }

    fn output(&self, device: &MlxDevice, cfg: &Deepseek4Config) -> MlxBuffer {
        let b = self.positions.element_count();
        let mut out = alloc(
            device,
            DType::BF16,
            vec![
                1,
                b,
                cfg.num_attention_heads as usize,
                cfg.head_dim as usize,
            ],
            "test attention",
        )
        .unwrap();
        out.as_mut_slice::<u16>().unwrap().fill(0x7f7f);
        out
    }

    fn row(&self, cfg: &Deepseek4Config, attention: &MlxBuffer, row: usize) -> JointRowInputs {
        let heads = cfg.num_attention_heads as usize;
        let head_dim = cfg.head_dim as usize;
        JointRowInputs {
            q_rope: row_view(&self.q_rope, row, vec![1, 1, heads, head_dim]).unwrap(),
            kv_rope: row_view(&self.kv_rope, row, vec![1, 1, 1, head_dim]).unwrap(),
            attention: row_view(attention, row, vec![1, 1, heads, head_dim]).unwrap(),
            attn_norm: row_view(&self.attn_norm, row, vec![1, cfg.hidden_size as usize]).unwrap(),
            q_a_norm: row_view(&self.q_a_norm, row, vec![1, cfg.q_lora_rank as usize]).unwrap(),
            position: row_view(&self.positions, row, vec![1]).unwrap(),
        }
    }

    fn batch<'a>(&'a self, attention: &'a MlxBuffer) -> JointBatchInputs<'a> {
        JointBatchInputs {
            q_rope: &self.q_rope,
            kv_rope: &self.kv_rope,
            attention,
            attn_norm: &self.attn_norm,
            q_a_norm: &self.q_a_norm,
            positions: &self.positions,
        }
    }
}

/// Run `steps` joint steps from `start` positions on one layer of `ratio`,
/// per-row vs batched, and require byte equality after each.
fn check_layer(
    ctx: &mut GpuContext,
    cfg: &Deepseek4Config,
    ratio: u32,
    start: &[usize],
    steps: usize,
) {
    check_layer_with(ctx, cfg, ratio, start, steps, GgmlType::F16);
}

/// [`check_layer`] with the compressor and indexer projections stored as
/// `storage` (F16 as in the ds4 file, Q8_0 as in hf2q's own artifact).
fn check_layer_with(
    ctx: &mut GpuContext,
    cfg: &Deepseek4Config,
    ratio: u32,
    start: &[usize],
    steps: usize,
    storage: GgmlType,
) {
    let device = ctx.device().clone();
    let layer = (0..cfg.compress_ratios.len())
        .find(|&l| cfg.compress_ratios[l] == ratio)
        .unwrap();
    let b = start.len();
    assert!(batched_cores_admit(ratio, start));
    let plan = Deepseek4CachePlan::for_context(cfg, CONTEXT).unwrap();
    let build = || -> Vec<Deepseek4Cache> {
        (0..b)
            .map(|row| {
                let mut cache = synthetic_cache(&device, &plan, 100 + row as u64);
                fill_states(&mut cache, 200 + row as u64);
                cache
            })
            .collect()
    };
    let mut per_row = build();
    let mut batched = build();
    let weights = LayerWeights::with_storage(&device, cfg, ratio, layer as u64, storage);
    let unused = Unused {
        buffer: alloc(&device, DType::F32, vec![1], "test unused").unwrap(),
        shape: [1, 1],
    };
    let rope = cfg.rope_head_dim as usize;
    let frequencies = if ratio == 0 {
        host_table(
            &device,
            yarn_frequencies(
                rope,
                0,
                cfg.rope_theta,
                1.0,
                cfg.yarn_beta_fast,
                cfg.yarn_beta_slow,
            )
            .unwrap(),
        )
    } else {
        host_table(
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
        )
    };
    for step in 0..steps {
        let positions: Vec<usize> = start.iter().map(|&p| p + step).collect();
        for caches in [&mut per_row, &mut batched] {
            for (cache, &position) in caches.iter_mut().zip(&positions) {
                cache.next_position = position;
            }
        }
        let plans: Vec<_> = per_row
            .iter()
            .map(|c| c.plan_next_step().unwrap())
            .collect();
        assert_eq!(
            plans,
            batched
                .iter()
                .map(|c| c.plan_next_step().unwrap())
                .collect::<Vec<_>>()
        );
        let layer_steps: Vec<&LayerCacheStep> = plans.iter().map(|s| &s.layers[layer]).collect();
        let inputs = StepInputs::new(&device, cfg, &positions, 1000 + step as u64);
        let out_row = inputs.output(&device, cfg);
        let out_batched = inputs.output(&device, cfg);
        let label = format!(
            "ratio {ratio} positions {positions:?} emit {:?} matvec {:?}",
            layer_steps
                .iter()
                .map(|s| s.compressed_write_slot.is_some())
                .collect::<Vec<_>>(),
            bench_matvec()
        );
        let refs = (ratio != 0).then(|| weights.compressed(&unused));
        let mut row_indices = Vec::new();
        let mut batched_indices = Vec::new();
        if ratio == 0 {
            let cores: Vec<_> = (0..b)
                .map(|row| {
                    plan_window_row(
                        &device,
                        &per_row[row],
                        layer,
                        positions[row],
                        cfg.head_dim as usize,
                    )
                    .unwrap()
                })
                .collect();
            run_once(ctx, |s, r, d| {
                for row in 0..b {
                    encode_window_row(
                        s,
                        r,
                        d,
                        cfg,
                        &per_row[row].layers()[layer],
                        layer_steps[row],
                        &cores[row],
                        &weights.sinks,
                        &inputs.row(cfg, &out_row, row),
                    )?;
                }
                Ok(())
            });
            let layer_caches: Vec<&LayerCache> =
                batched.iter().map(|c| &c.layers()[layer]).collect();
            let window =
                plan_window_layer(&device, &layer_caches, &positions, cfg.head_dim as usize)
                    .unwrap();
            run_once(ctx, |s, r, d| {
                encode_window_layer(
                    s,
                    r,
                    d,
                    cfg,
                    &layer_caches,
                    &layer_steps,
                    &window,
                    &weights.sinks,
                    &inputs.batch(&out_batched),
                )
            });
        } else {
            let refs = refs.as_ref().unwrap();
            let cores: Vec<_> = (0..b)
                .map(|row| {
                    plan_compressed_row(
                        &device,
                        cfg,
                        &per_row[row],
                        layer,
                        ratio,
                        positions[row],
                        layer_steps[row],
                    )
                    .unwrap()
                })
                .collect();
            run_once(ctx, |s, r, d| {
                for row in 0..b {
                    encode_compressed_row(
                        s,
                        r,
                        d,
                        cfg,
                        &per_row[row].layers()[layer],
                        layer_steps[row],
                        &cores[row],
                        refs,
                        &frequencies,
                        &inputs.row(cfg, &out_row, row),
                    )?;
                }
                Ok(())
            });
            row_indices = cores
                .iter()
                .map(|core| {
                    core.index_storage()
                        .as_logical_slice::<i32>()
                        .unwrap()
                        .to_vec()
                })
                .collect();
            let layer_caches: Vec<&LayerCache> =
                batched.iter().map(|c| &c.layers()[layer]).collect();
            let grafts: Vec<usize> = batched.iter().map(|c| c.graft_slots()).collect();
            let compressed = plan_compressed_layer(
                &device,
                cfg,
                &layer_caches,
                &grafts,
                ratio,
                &positions,
                &layer_steps,
            )
            .unwrap();
            run_once(ctx, |s, r, d| {
                compressed.encode(
                    s,
                    r,
                    d,
                    cfg,
                    &layer_caches,
                    &layer_steps,
                    refs,
                    &frequencies,
                    &inputs.batch(&out_batched),
                )
            });
            batched_indices = (0..b)
                .map(|row| compressed.row_index_storage(row))
                .collect();
        }
        let heads_bytes = cfg.num_attention_heads as usize * cfg.head_dim as usize * 2;
        let a = out_row.as_slice::<u8>().unwrap();
        let c = out_batched.as_slice::<u8>().unwrap();
        for row in 0..b {
            assert!(
                a[row * heads_bytes..(row + 1) * heads_bytes]
                    == c[row * heads_bytes..(row + 1) * heads_bytes],
                "{label}: row {row} attention output differs"
            );
            if ratio != 0 {
                assert_eq!(
                    row_indices[row], batched_indices[row],
                    "{label}: row {row} indices"
                );
            }
            let lhs = cache_bytes(&per_row[row].layers()[layer]);
            let rhs = cache_bytes(&batched[row].layers()[layer]);
            for (buffer, (l, r)) in lhs.iter().zip(&rhs).enumerate() {
                assert!(l == r, "{label}: row {row} cache buffer {buffer} differs");
            }
        }
        // A batched row must not depend on its peers: the output is not all guard.
        assert!(c.chunks_exact(2).any(|v| v != [0x7f, 0x7f]));
    }
}

fn gpu() -> (
    std::sync::MutexGuard<'static, ()>,
    GpuContext,
    Deepseek4Config,
) {
    let lock = crate::inference::hf2q_gpu_test_lock();
    (
        lock,
        GpuContext::new().unwrap(),
        super::cache_tests::config(ds4_ratios()),
    )
}

#[test]
fn batched_window_cores_equal_per_row_cores() {
    let (_lock, mut ctx, cfg) = gpu();
    for start in [
        &[2047usize][..],
        &[2047, 50],
        &[127, 5000, 1],
        &[8000, 2047, 64, 300],
    ] {
        check_layer(&mut ctx, &cfg, 0, start, 2);
    }
}

#[test]
fn batched_hca_cores_equal_per_row_cores() {
    let (_lock, mut ctx, cfg) = gpu();
    // 2047 and 1279 emit at the first step, 2046 at the second.
    for start in [
        &[2047usize][..],
        &[2046, 2047],
        &[1279, 3000, 127],
        &[2047, 8100, 400, 2046],
    ] {
        check_layer(&mut ctx, &cfg, 128, start, 2);
    }
}

#[test]
fn batched_csa_cores_equal_per_row_cores() {
    let (_lock, mut ctx, cfg) = gpu();
    for start in [
        // One flash width (640): contiguous flash batch, 1 and 2 top-k blocks.
        &[2100usize, 8187][..],
        // Mixed widths (flash permuted, logits rows), masked top-k slots
        // (fewer than 512 valid keys), emit and non-emit rows.
        &[2046, 757, 2333, 1700],
        &[383, 2047, 1],
        &[2047],
    ] {
        check_layer(&mut ctx, &cfg, 4, start, 3);
    }
}

/// hf2q's own artifact stores the compressor and indexer projections as
/// Q8_0: the batched cores issue them as one `m = B` dispatch (stock Q8_0
/// matvec, or the stock-order row-exact kernel), which must still equal the
/// per-row cores' `m = 1` products bitwise.
#[test]
fn batched_cores_with_q8_0_projections_equal_per_row_cores() {
    let (_lock, mut ctx, cfg) = gpu();
    check_layer_with(
        &mut ctx,
        &cfg,
        4,
        &[2046, 757, 2333, 1700],
        3,
        GgmlType::Q8_0,
    );
    check_layer_with(&mut ctx, &cfg, 128, &[2047, 8100, 400], 2, GgmlType::Q8_0);
}

/// The batched projections take one `m = B` dispatch only where joint-decode
/// admission proves `m = B` rows equal to `m = 1` (or row-exact, or one
/// row); every other storage runs per row, so admission needs no entry for
/// them.
#[test]
fn batched_projections_dispatch_only_proven_routes_at_m_b() {
    for rows in 2..=8 {
        // Unproven block storage (a K-quant): per row.
        for unproven in [GgmlType::Q4_K, GgmlType::Q6_K] {
            assert!(!projection_takes_single_dispatch(false, unproven, rows));
        }
        // Proven block storage: one dispatch.
        assert!(projection_takes_single_dispatch(
            false,
            GgmlType::Q8_0,
            rows
        ));
        // Dense float on the stock family splits per row.
        assert!(!projection_takes_single_dispatch(
            false,
            GgmlType::F16,
            rows
        ));
        // The stock-order row-exact route is one dispatch.
        assert!(projection_takes_single_dispatch(true, GgmlType::F16, rows));
    }
    for storage in [GgmlType::Q4_K, GgmlType::F16, GgmlType::Q8_0] {
        assert!(projection_takes_single_dispatch(false, storage, 1));
    }
}

#[test]
fn batched_cores_switch_resolution() {
    use std::ffi::OsStr;
    assert_eq!(AttnCores::resolve(None).unwrap(), AttnCores::PerRow);
    assert_eq!(
        AttnCores::resolve(Some(OsStr::new("0"))).unwrap(),
        AttnCores::PerRow
    );
    assert_eq!(
        AttnCores::resolve(Some(OsStr::new("1"))).unwrap(),
        AttnCores::Batched
    );
    assert!(AttnCores::resolve(Some(OsStr::new("yes"))).is_err());
    assert_eq!(AttnCores::default(), AttnCores::PerRow);
    // Position-zero rows have no recurrent compressor state: per-row cores.
    assert!(!batched_cores_admit(4, &[0, 5]));
    assert!(batched_cores_admit(0, &[0, 5]));
    assert!(!batched_cores_admit(128, &[5; 9]));
}
