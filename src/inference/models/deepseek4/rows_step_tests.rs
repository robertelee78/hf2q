//! Model-free exactness of the rows step's cache-dependent cores.
//!
//! The rows step reuses joint decode's per-row products (row-exact or per-row
//! stock, proven by `row_matmul_tests` / `row_exact_matvec_tests` and the
//! joint-decode real-artifact test) and its `mv_id` expert route. What is new
//! is that its rows are consecutive positions of ONE cache: row `i + 1`
//! overwrites a window slot row `i` reads and continues row `i`'s compressor
//! and indexer recurrences. These tests run the production core encoders at
//! the ds4 file's attention shapes (window, ratio-4 and ratio-128 layers) on
//! synthetic caches and weights, and require the 8-row `RowCaches::Sequential`
//! step to leave every cache byte and every row's attention output bitwise
//! equal to 8 one-token steps, each in its own command buffer with the cache
//! committed between them.

use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::cache::{CacheStep, Deepseek4Cache, Deepseek4CachePlan};
use super::cache_tests::config;
use super::ds4_attn_microbench_tests::{
    bf16_buf, host_table, synthetic_cache, LayerWeights, Unused,
};
use super::ds4_bench_support::{f32_buf, run_once, Rng};
use super::forward_support::{alloc, alloc_host_input};
use super::joint_decode::MAX_JOINT_DECODE_ROWS;
use super::joint_decode_core::{encode_row_cores, plan_row_cores, JointRowInputs, RowCaches};
use super::rope::yarn_frequencies;
use super::row_matmul::row_view;
use super::Deepseek4Config;
use crate::serve::gpu::GpuContext;

const CONTEXT: usize = 1024;
/// Context of the indexer top-k case: positions past 2048, where a ratio-4
/// layer holds more than `index_top_k` (512) compressed rows.
const TOP_K_CONTEXT: usize = 4096;
const RATIOS: [u32; 3] = [0, 4, 128];

fn fill_f32(buffer: &mut MlxBuffer, seed: u64) {
    let mut rng = Rng(seed);
    for value in buffer.as_mut_slice::<f32>().unwrap() {
        *value = rng.unit();
    }
}

/// Synthetic cache with random window/compressed/indexer rows and random
/// recurrent compressor states, advanced to `start`.
fn seeded_cache(device: &MlxDevice, plan: &Deepseek4CachePlan, start: usize) -> Deepseek4Cache {
    let mut cache = synthetic_cache(device, plan, 7);
    for (layer, lc) in cache.layers_mut().iter_mut().enumerate() {
        let seed = 9_000 + layer as u64 * 8;
        for (offset, state) in [
            lc.main_kv_state.as_mut(),
            lc.main_score_state.as_mut(),
            lc.indexer_kv_state.as_mut(),
            lc.indexer_score_state.as_mut(),
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(state) = state {
                fill_f32(state, seed + offset as u64);
            }
        }
    }
    for position in 0..start {
        cache.commit_step(position).unwrap();
    }
    cache
}

/// Every cache buffer's bytes, labelled, for bitwise comparison.
fn cache_bytes(cache: &Deepseek4Cache) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for (layer, lc) in cache.layers().iter().enumerate() {
        let mut push = |name: &str, buffer: Option<&MlxBuffer>| {
            if let Some(buffer) = buffer {
                out.push((
                    format!("layer {layer} {name}"),
                    buffer.as_slice::<u8>().unwrap().to_vec(),
                ));
            }
        };
        push("attention_kv", Some(&lc.attention_kv));
        push("indexer_kv", lc.indexer_kv.as_ref());
        push("main_kv_state", lc.main_kv_state.as_ref());
        push("main_score_state", lc.main_score_state.as_ref());
        push("indexer_kv_state", lc.indexer_kv_state.as_ref());
        push("indexer_score_state", lc.indexer_score_state.as_ref());
    }
    out
}

/// Batched per-row inputs of one step (as the joint graph builds them before
/// the per-row cores) and one attention output buffer per layer.
struct Rows {
    q_rope: MlxBuffer,
    kv_rope: Vec<MlxBuffer>,
    attn_norm: MlxBuffer,
    q_a_norm: MlxBuffer,
    positions: MlxBuffer,
    attention: Vec<MlxBuffer>,
}

impl Rows {
    fn new(device: &MlxDevice, cfg: &Deepseek4Config, start: usize, rows: usize) -> Self {
        let heads = cfg.num_attention_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let mut positions = alloc_host_input(device, DType::U32, vec![rows], "positions").unwrap();
        for (row, value) in positions
            .as_logical_mut_slice::<u32>()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *value = (start + row) as u32;
        }
        Self {
            q_rope: bf16_buf(device, vec![1, rows, heads, head_dim], 41, 1.0),
            // One KV row per layer, as each layer projects its own.
            kv_rope: (0..RATIOS.len())
                .map(|layer| bf16_buf(device, vec![1, rows, 1, head_dim], 42 + layer as u64, 0.5))
                .collect(),
            attn_norm: f32_buf(device, vec![rows, cfg.hidden_size as usize], 43, 1.0),
            q_a_norm: f32_buf(device, vec![rows, cfg.q_lora_rank as usize], 44, 1.0),
            positions,
            attention: (0..RATIOS.len())
                .map(|_| {
                    let mut buffer =
                        alloc(device, DType::BF16, vec![1, rows, heads, head_dim], "attention")
                            .unwrap();
                    buffer.as_mut_slice::<u8>().unwrap().fill(0);
                    buffer
                })
                .collect(),
        }
    }

    fn inputs(&self, cfg: &Deepseek4Config, layer: usize, row: usize) -> JointRowInputs {
        let heads = cfg.num_attention_heads as usize;
        let head_dim = cfg.head_dim as usize;
        JointRowInputs {
            q_rope: row_view(&self.q_rope, row, vec![1, 1, heads, head_dim]).unwrap(),
            kv_rope: row_view(&self.kv_rope[layer], row, vec![1, 1, 1, head_dim]).unwrap(),
            attention: row_view(&self.attention[layer], row, vec![1, 1, heads, head_dim]).unwrap(),
            attn_norm: row_view(&self.attn_norm, row, vec![1, cfg.hidden_size as usize]).unwrap(),
            q_a_norm: row_view(&self.q_a_norm, row, vec![1, cfg.q_lora_rank as usize]).unwrap(),
            position: row_view(&self.positions, row, vec![1]).unwrap(),
        }
    }
}

struct Fixture {
    cfg: Deepseek4Config,
    plan: Deepseek4CachePlan,
    weights: Vec<LayerWeights>,
    unused: Unused,
    window_frequencies: MlxBuffer,
    compressed_frequencies: MlxBuffer,
}

impl Fixture {
    fn new(device: &MlxDevice, context: usize) -> Self {
        let cfg = config(RATIOS.to_vec());
        let rope = cfg.rope_head_dim as usize;
        Self {
            plan: Deepseek4CachePlan::for_context(&cfg, context).unwrap(),
            weights: RATIOS
                .iter()
                .enumerate()
                .map(|(layer, &ratio)| LayerWeights::new(device, &cfg, ratio, layer as u64 + 1))
                .collect(),
            unused: Unused {
                buffer: alloc(device, DType::F32, vec![1], "unused").unwrap(),
                shape: [1, 1],
            },
            window_frequencies: host_table(
                device,
                yarn_frequencies(rope, 0, cfg.rope_theta, 1.0, cfg.yarn_beta_fast, cfg.yarn_beta_slow)
                    .unwrap(),
            ),
            compressed_frequencies: host_table(
                device,
                yarn_frequencies(
                    rope,
                    cfg.original_context_length as usize,
                    cfg.compress_rope_theta,
                    cfg.rope_factor,
                    cfg.yarn_beta_fast,
                    cfg.yarn_beta_slow,
                )
                .unwrap(),
            ),
            cfg,
        }
    }

    /// Encode every layer's cores for `steps` (rows `first..first + steps.len()`
    /// of `rows`) in one command buffer, all rows on `cache`.
    fn run(
        &self,
        ctx: &mut GpuContext,
        cache: &Deepseek4Cache,
        steps: &[CacheStep],
        rows: &Rows,
        first: usize,
        order: RowCaches,
    ) {
        let caches = vec![cache; steps.len()];
        run_once(ctx, |session, registry, device| {
            for (layer, &ratio) in RATIOS.iter().enumerate() {
                let cores = plan_row_cores(device, &self.cfg, &caches, steps, layer, ratio)?;
                let inputs = (0..steps.len())
                    .map(|row| rows.inputs(&self.cfg, layer, first + row))
                    .collect::<Vec<_>>();
                let weights = &self.weights[layer];
                let compressed = (ratio != 0).then(|| weights.compressed(&self.unused));
                let frequencies = if ratio == 0 {
                    &self.window_frequencies
                } else {
                    &self.compressed_frequencies
                };
                session.barrier();
                encode_row_cores(
                    session,
                    registry,
                    device,
                    &self.cfg,
                    &caches,
                    steps,
                    layer,
                    &cores,
                    &inputs,
                    &weights.sinks,
                    compressed.as_ref(),
                    frequencies,
                    order,
                )?;
            }
            Ok(())
        });
    }
}

/// 8 rows from `start` as one sequential rows step vs 8 one-token steps; the first differing
/// buffer, or `None` when every cache byte and attention output is equal. `stale_plans` plans every
/// row of the rows step at `start` (negative control: the comparison must see it).
fn rows_step_mismatch(start: usize, stale_plans: bool) -> Option<String> {
    rows_step_mismatch_in(CONTEXT, start, stale_plans)
}

fn rows_step_mismatch_in(context: usize, start: usize, stale_plans: bool) -> Option<String> {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let fixture = Fixture::new(&device, context);
    let rows = MAX_JOINT_DECODE_ROWS;

    // Reference: one-token steps, each in its own command buffer, committed.
    let mut serial = seeded_cache(&device, &fixture.plan, start);
    let serial_rows = Rows::new(&device, &fixture.cfg, start, rows);
    for row in 0..rows {
        let step = serial.plan_next_step().unwrap();
        assert_eq!(step.position, start + row);
        fixture.run(
            &mut ctx,
            &serial,
            &[step],
            &serial_rows,
            row,
            RowCaches::Independent,
        );
        serial.commit_step(start + row).unwrap();
    }

    // Rows step: every row planned up front, one command buffer, one publication.
    let mut stepped = seeded_cache(&device, &fixture.plan, start);
    let stepped_rows = Rows::new(&device, &fixture.cfg, start, rows);
    let steps = (start..start + rows)
        .map(|position| {
            stepped
                .plan_step_at(if stale_plans { start } else { position })
                .unwrap()
        })
        .collect::<Vec<_>>();
    fixture.run(
        &mut ctx,
        &stepped,
        &steps,
        &stepped_rows,
        0,
        RowCaches::Sequential,
    );
    let ticket = stepped.validate_prefill_commit(start, rows).unwrap();
    stepped.publish_prefill_end(ticket);

    assert_eq!(stepped.position(), serial.position());
    for layer in 0..RATIOS.len() {
        assert!(
            serial_rows.attention[layer]
                .as_slice::<u16>()
                .unwrap()
                .iter()
                .any(|&bits| bits != 0),
            "start {start}: layer {layer} attention never ran"
        );
    }
    for ((name, left), (_, right)) in cache_bytes(&serial).iter().zip(cache_bytes(&stepped)) {
        if left != &right {
            return Some(format!("start {start}: {name} differs"));
        }
    }
    (0..RATIOS.len()).find_map(|layer| {
        (serial_rows.attention[layer].as_slice::<u8>().unwrap()
            != stepped_rows.attention[layer].as_slice::<u8>().unwrap())
        .then(|| format!("start {start}: layer {layer} attention outputs differ"))
    })
}

#[test]
fn rows_step_cores_match_one_token_steps_across_window_wrap_and_csa_emits() {
    // Positions 123..=130: the window ring wraps at 128 (the row at 128
    // overwrites the slot of position 0, which the rows at 123..=127 attend
    // to), and ratio-4 blocks complete at positions 123 and 127.
    assert_eq!(rows_step_mismatch(123, false), None);
}

#[test]
fn rows_step_cores_match_one_token_steps_across_an_hca_block() {
    // Positions 251..=258: the ratio-128 block completes at position 255.
    assert_eq!(rows_step_mismatch(251, false), None);
}

#[test]
fn rows_step_cores_match_one_token_steps_while_the_indexer_drops_rows() {
    // Positions 2045..=2052: the ratio-4 layer holds 511 compressed rows at
    // 2045 and emits at 2047 and 2051, so rows 2051 and 2052 select 512 of
    // 513 rows (the indexer's top-k drops one) under the sequential order.
    {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let cfg = config(RATIOS.to_vec());
        let plan = Deepseek4CachePlan::for_context(&cfg, TOP_K_CONTEXT).unwrap();
        let mut cache = Deepseek4Cache::allocate(&plan, MlxDevice::new().unwrap()).unwrap();
        for position in 0..2045 {
            cache.commit_step(position).unwrap();
        }
        let valid =
            |position| cache.plan_step_at(position).unwrap().layers[1].compressed_valid_after;
        assert_eq!(valid(2045), 511);
        assert!(valid(2051) > cfg.index_top_k as usize);
    }
    assert_eq!(rows_step_mismatch_in(TOP_K_CONTEXT, 2045, false), None);
}

#[test]
fn rows_step_comparison_detects_rows_planned_at_the_wrong_position() {
    assert!(rows_step_mismatch(123, true).is_some());
}

#[test]
fn rows_step_plans_equal_committed_one_token_plans() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let cfg = config(RATIOS.to_vec());
    let plan = Deepseek4CachePlan::for_context(&cfg, CONTEXT).unwrap();
    let mut cache = Deepseek4Cache::allocate(&plan, device).unwrap();
    for position in 0..251 {
        cache.commit_step(position).unwrap();
    }
    let ahead = (251..259)
        .map(|position| cache.plan_step_at(position).unwrap())
        .collect::<Vec<_>>();
    assert!(cache.plan_step_at(250).is_err(), "positions behind the cursor are refused");
    assert!(cache.plan_step_at(CONTEXT).is_err(), "positions past the context are refused");
    for (offset, planned) in ahead.iter().enumerate() {
        assert_eq!(&cache.plan_next_step().unwrap(), planned);
        cache.commit_step(251 + offset).unwrap();
    }
}

#[test]
fn rows_step_rejects_a_poisoned_cache() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let plan = Deepseek4CachePlan::for_context(&config(RATIOS.to_vec()), CONTEXT).unwrap();
    let mut cache = Deepseek4Cache::allocate(&plan, device).unwrap();
    cache.poison();
    assert!(cache.plan_step_at(0).is_err());
}

/// Model-free reproduction of the 2026-11-07 model-gate setup failure: a cache smaller than one
/// sliding window gets a ring of `capacity` rows, and every compressed-attention path (here the
/// one-token core the rows step uses; the matrix prefill has the same check) refuses it as
/// "compressed attention cache shape drift". Real caches start at one full window or more.
#[test]
fn caches_below_one_window_are_refused_by_the_compressed_cores() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().unwrap();
    let cfg = config(RATIOS.to_vec());
    let window = cfg.sliding_window as usize;
    for (capacity, refused) in [(18, true), (window, false), (CONTEXT, false)] {
        let plan = Deepseek4CachePlan::for_context(&cfg, capacity).unwrap();
        assert_eq!(plan.layers[1].window_kv.shape[0], capacity.min(window));
        let mut cache = Deepseek4Cache::allocate(&plan, device.clone()).unwrap();
        for position in 0..3 {
            cache.commit_step(position).unwrap();
        }
        let step = cache.plan_next_step().unwrap();
        let planned = plan_row_cores(&device, &cfg, &[&cache], &[step], 1, RATIOS[1]);
        match planned {
            Err(error) => {
                assert!(refused, "capacity {capacity}: {error:#}");
                assert!(format!("{error:#}").contains("compressed attention cache shape drift"));
            }
            Ok(_) => assert!(!refused, "capacity {capacity} must be refused"),
        }
    }
}
