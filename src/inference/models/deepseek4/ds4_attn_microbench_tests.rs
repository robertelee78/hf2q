//! DeepSeek-V4 gap program M2b: hf2q's per-row, cache-dependent decode attention cores and the
//! hyper-connection ops at the ds4 0731 file's shapes, B = 1..4 independent rows.
//!
//! Model-free: synthetic caches (random BF16 rows, production `Deepseek4Cache` layout for the
//! 43-layer ratio pattern 0,0,{4,128}...) and synthetic F16 compressor/indexer weights. Each row
//! owns its own cache, as in joint decode. Cells encode exactly the per-row calls joint decode
//! issues (`joint_decode.rs` row loop):
//! - `core`: `plan_window_row`/`encode_window_row` or `plan_compressed_row`/`encode_compressed_row`
//!   for each of the B rows (window write, compressor, indexer + top-k, sparse attention);
//! - sub-ops through the same entry points those functions call: `window_write`,
//!   `main_compressor` (2 F16 matvecs + tail), `main_compressor_tail`, `indexer` (all of
//!   `encode_ratio_four_indexer`), `indexer_compressor_tail`, `indexer_select` (score + top-k,
//!   `encode_selection`), `attention` (the sparse/flash decode kernel alone);
//! - HC ops: `rms_norm_no_scale_f32` over 4x4096, F16 `hc_fn` per row, `dispatch_hc_split_sinkhorn`,
//!   `dispatch_hc_pre`, `dispatch_hc_post`, and the whole pre site.
//!
//! Timing: `ds4_bench_support` (shared with M2a): GPU command-buffer timestamps, `reps` per
//! session calibrated for >= 2 ms of GPU work, median of 9 trials per copy; `wall_us` (commit + wait)
//! and `encode_us` (host) per copy are recorded too. `plan_us` is the host time of `plan_*_row` for
//! one layer's B rows inside a decode pool cycle. The `floor` cells (1 or 16 dependent 1-element
//! dispatches per copy) record the method's floor in every run. A 2 s F16 matvec warm-up runs
//! first, and every cell runs >= 50 ms of untimed sessions before its timed ones.
//! Cold cells rotate over distinct layers (own weights and own cache rows) until the weights
//! exceed 256 MiB; `warm` cells reuse one layer. Compressed layers are timed at a step that emits a
//! compressed row (`emit`) and one that does not; amortize as `(ratio-1)/ratio * steady + emit/ratio`.
//! At context 2048 the CSA steady step sees 511 valid compressed rows, so one of its 512 top-k
//! slots is masked (as at every context below 2048); its emit step and context 8192 are unmasked.
//!
//! Positions match the peer's decode FLASH_ATTN_EXT cases: context 2048 (raw kv 1280, HCA
//! 1280+256, CSA 1280+512 with n_kv_max 640) and 8192 (CSA 1280+2048).
//!
//! Run (standing grant, model-free, <= 15 min):
//! ```text
//! HF2Q_DS4_ATTN_BENCH_OUT=<cells.jsonl> cargo test --release --bin hf2q ds4_attn_microbench -- --ignored --nocapture
//! ```

use std::time::Instant;

use anyhow::Result;
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_hyper_connection::{
    dispatch_hc_post, dispatch_hc_pre, dispatch_hc_split_sinkhorn,
};
use mlx_native::ops::deepseek_sparse_attention::{
    dispatch_deepseek_sparse_attention, dispatch_deepseek_sparse_attention_flash_decode,
    dispatch_deepseek_sparse_attention_with_scratch, DeepSeekSparseAttentionParams,
};
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};
use serde_json::{json, Value};

use super::attention::window_indices;
use super::attention_weights::AttentionWeightsRef;
use super::cache::{CacheStep, Deepseek4Cache, Deepseek4CachePlan, LayerCache, LayerCacheStep};
use super::cache_window_write::encode_window_write;
use super::compressed_attention::{
    sparse_attention_scratch_shape, SparseFlashDecodeArena, SPARSE_FLASH_MIN_TOP_K,
};
use super::compressed_attention_indexer::layout::prepare_layout;
use super::compressed_attention_indexer::selection::{encode_selection, SelectionRequest};
use super::compressed_attention_indexer::{encode_ratio_four_indexer, RatioFourIndexerArena};
use super::compressed_attention_main::{encode_main_compressor, MainCompressorArena};
use super::compressed_attention_weights::{
    CompressedAttentionWeightsRef, CompressorWeightsRef, IndexerWeightsRef,
};
use super::compressor_tail::{encode_indexer_tail, encode_main_tail, CompressorTail};
use super::ds4_bench_support::{
    bench_matvec, copies_for, f16_matrix, f32_buf, floor_cells, gpu_warmup, median, run_once,
    time_op, Out, Rng, COLD_BYTES, TRIALS,
};
use super::forward_support::{
    alloc, alloc_host_input, begin_decode_pool_token, end_decode_pool_token, rms_params,
};
use super::joint_decode_core::{
    encode_compressed_row, encode_window_row, plan_compressed_row, plan_window_row, JointRowInputs,
};
use super::prefill_annotations;
use super::residency::RawMatrixRef;
use super::rope::yarn_frequencies;
use super::row_matmul::{row_matmul, row_view, RowMatmul};
use super::Deepseek4Config;
use crate::serve::gpu::GpuContext;

const CONTEXT: usize = 8192;
const MAX_ROWS: usize = 4;
const LAYERS: usize = 43;

pub(super) fn ds4_ratios() -> Vec<u32> {
    (0..LAYERS)
        .map(|layer| match layer {
            0 | 1 => 0,
            l if l % 2 == 0 => 4,
            _ => 128,
        })
        .collect()
}

pub(super) fn fill_bf16(buf: &mut MlxBuffer, seed: u64, scale: f32) {
    let mut rng = Rng(seed);
    for v in buf.as_mut_slice::<u16>().unwrap() {
        *v = half::bf16::from_f32(rng.unit() * scale).to_bits();
    }
}

pub(super) fn bf16_buf(device: &MlxDevice, shape: Vec<usize>, seed: u64, scale: f32) -> MlxBuffer {
    let mut buf = alloc(device, DType::BF16, shape, "bench bf16").unwrap();
    fill_bf16(&mut buf, seed, scale);
    buf
}

pub(super) fn host_table(device: &MlxDevice, values: Vec<f32>) -> MlxBuffer {
    let mut buf = alloc_host_input(device, DType::F32, vec![values.len()], "bench table").unwrap();
    buf.as_logical_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&values);
    buf
}

/// Host µs to plan one layer's B rows, median of TRIALS decode-pool cycles (joint decode plans inside
/// `begin_decode_pool_token`), plus one plan made outside the pool for the GPU cells.
fn plan_rows<T>(layers: usize, mut plan: impl FnMut() -> Vec<T>) -> (Vec<T>, f64) {
    let mut samples = Vec::with_capacity(TRIALS);
    for _ in 0..=TRIALS {
        begin_decode_pool_token();
        let t0 = Instant::now();
        let planned = plan();
        samples.push(t0.elapsed().as_secs_f64() / layers as f64);
        drop(planned);
        end_decode_pool_token();
    }
    samples.remove(0); // cycle 0 fills the pool
    (plan(), median(samples) * 1e6)
}

struct CompressorWeights {
    kv: MlxBuffer,
    gate: MlxBuffer,
    ape: MlxBuffer,
    norm: MlxBuffer,
    shape: [usize; 2],
}

impl CompressorWeights {
    fn new(device: &MlxDevice, ratio: usize, dim: usize, hidden: usize, seed: u64) -> Self {
        let projected = (usize::from(ratio == 4) + 1) * dim;
        let mut norm = alloc(device, DType::F32, vec![dim], "bench compressor norm").unwrap();
        norm.as_mut_slice::<f32>().unwrap().fill(1.0);
        Self {
            kv: f16_matrix(device, projected, hidden, seed),
            gate: f16_matrix(device, projected, hidden, seed + 1),
            ape: f32_buf(device, vec![ratio, projected], seed + 2, 0.1),
            norm,
            shape: [projected, hidden],
        }
    }

    fn bytes(&self) -> usize {
        2 * self.shape[0] * self.shape[1] * 2
    }

    fn get(&self) -> CompressorWeightsRef<'_> {
        CompressorWeightsRef {
            kv: RawMatrixRef {
                buffer: &self.kv,
                ggml_type: GgmlType::F16,
                shape: &self.shape,
                matvec: bench_matvec(),
            },
            gate: RawMatrixRef {
                buffer: &self.gate,
                ggml_type: GgmlType::F16,
                shape: &self.shape,
                matvec: bench_matvec(),
            },
            ape: &self.ape,
            norm: &self.norm,
        }
    }
}

struct IndexerWeights {
    q_b: MlxBuffer,
    q_b_shape: [usize; 2],
    projection: MlxBuffer,
    projection_shape: [usize; 2],
    compressor: CompressorWeights,
}

/// Synthetic F16 weights of one layer as stored in the ds4 file; only what the per-row core reads.
pub(super) struct LayerWeights {
    pub(super) sinks: MlxBuffer,
    compressor: Option<CompressorWeights>,
    indexer: Option<IndexerWeights>,
}

impl LayerWeights {
    pub(super) fn new(device: &MlxDevice, cfg: &Deepseek4Config, ratio: u32, seed: u64) -> Self {
        let hidden = cfg.hidden_size as usize;
        let seed = seed * 64;
        let compressor = (ratio != 0).then(|| {
            CompressorWeights::new(device, ratio as usize, cfg.head_dim as usize, hidden, seed)
        });
        let indexer = (ratio == 4).then(|| {
            let heads = cfg.index_num_heads as usize;
            let dim = cfg.index_head_dim as usize;
            let q_rank = cfg.q_lora_rank as usize;
            IndexerWeights {
                q_b: f16_matrix(device, heads * dim, q_rank, seed + 10),
                q_b_shape: [heads * dim, q_rank],
                projection: f16_matrix(device, heads, hidden, seed + 11),
                projection_shape: [heads, hidden],
                compressor: CompressorWeights::new(device, 4, dim, hidden, seed + 12),
            }
        });
        Self {
            sinks: f32_buf(
                device,
                vec![cfg.num_attention_heads as usize],
                seed + 20,
                1.0,
            ),
            compressor,
            indexer,
        }
    }

    fn bytes(&self) -> usize {
        self.compressor.as_ref().map_or(0, CompressorWeights::bytes)
            + self.indexer.as_ref().map_or(0, |i| {
                (i.q_b_shape[0] * i.q_b_shape[1] + i.projection_shape[0] * i.projection_shape[1])
                    * 2
                    + i.compressor.bytes()
            })
    }

    pub(super) fn compressed<'a>(&'a self, unused: &'a Unused) -> CompressedAttentionWeightsRef<'a> {
        let m = || RawMatrixRef {
            buffer: &unused.buffer,
            ggml_type: GgmlType::F16,
            shape: &unused.shape,
            matvec: bench_matvec(),
        };
        let u = &unused.buffer;
        CompressedAttentionWeightsRef {
            // The per-row core reads only the sinks; the projections run outside it.
            attention: AttentionWeightsRef {
                hc_fn: m(),
                hc_base: u,
                hc_scale: u,
                attn_norm: u,
                q_a: m(),
                q_a_norm: u,
                q_b: m(),
                kv: m(),
                kv_norm: u,
                sinks: &self.sinks,
                output_a: m(),
                output_b: m(),
            },
            compressor: self.compressor.as_ref().unwrap().get(),
            indexer: self.indexer.as_ref().map(|i| IndexerWeightsRef {
                q_b: RawMatrixRef {
                    buffer: &i.q_b,
                    ggml_type: GgmlType::F16,
                    shape: &i.q_b_shape,
                    matvec: bench_matvec(),
                },
                projection: RawMatrixRef {
                    buffer: &i.projection,
                    ggml_type: GgmlType::F16,
                    shape: &i.projection_shape,
                    matvec: bench_matvec(),
                },
                compressor: i.compressor.get(),
            }),
        }
    }
}

/// Stand-in for the projection tensors that the per-row core never reads.
pub(super) struct Unused {
    pub(super) buffer: MlxBuffer,
    pub(super) shape: [usize; 2],
}

/// Batched projection outputs of one joint step and their per-row views (as joint decode builds them).
struct RowBatch {
    rows: Vec<JointRowInputs>,
    _batched: [MlxBuffer; 6],
}

impl RowBatch {
    fn new(device: &MlxDevice, cfg: &Deepseek4Config, b: usize, position: usize) -> Self {
        let heads = cfg.num_attention_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let hidden = cfg.hidden_size as usize;
        let q_rank = cfg.q_lora_rank as usize;
        let q_rope = bf16_buf(device, vec![1, b, heads, head_dim], 31, 1.0);
        let kv_rope = bf16_buf(device, vec![1, b, 1, head_dim], 32, 0.5);
        let attention = alloc(
            device,
            DType::BF16,
            vec![1, b, heads, head_dim],
            "bench out",
        )
        .unwrap();
        let attn_norm = f32_buf(device, vec![b, hidden], 33, 1.0);
        let q_a_norm = f32_buf(device, vec![b, q_rank], 34, 1.0);
        let mut positions =
            alloc_host_input(device, DType::U32, vec![b], "bench positions").unwrap();
        positions
            .as_logical_mut_slice::<u32>()
            .unwrap()
            .fill(position as u32);
        let rows = (0..b)
            .map(|row| JointRowInputs {
                q_rope: row_view(&q_rope, row, vec![1, 1, heads, head_dim]).unwrap(),
                kv_rope: row_view(&kv_rope, row, vec![1, 1, 1, head_dim]).unwrap(),
                attention: row_view(&attention, row, vec![1, 1, heads, head_dim]).unwrap(),
                attn_norm: row_view(&attn_norm, row, vec![1, hidden]).unwrap(),
                q_a_norm: row_view(&q_a_norm, row, vec![1, q_rank]).unwrap(),
                position: row_view(&positions, row, vec![1]).unwrap(),
            })
            .collect();
        Self {
            rows,
            _batched: [q_rope, kv_rope, attention, attn_norm, q_a_norm, positions],
        }
    }
}

pub(super) fn synthetic_cache(device: &MlxDevice, plan: &Deepseek4CachePlan, seed: u64) -> Deepseek4Cache {
    let mut cache = Deepseek4Cache::allocate(plan, device.clone()).unwrap();
    for (layer, lc) in cache.layers_mut().iter_mut().enumerate() {
        let seed = seed * 1000 + layer as u64 * 2;
        fill_bf16(&mut lc.attention_kv, seed, 0.5);
        if let Some(indexer) = lc.indexer_kv.as_mut() {
            fill_bf16(indexer, seed + 1, 0.5);
        }
    }
    cache
}

struct IndexerSub {
    selection: SelectionRequest,
    arena: RatioFourIndexerArena,
    tail: [MlxBuffer; 4],
    query: MlxBuffer,
    weights: MlxBuffer,
    scores: MlxBuffer,
    topk: MlxBuffer,
}

enum Sparse {
    Flash(SparseFlashDecodeArena),
    Logits(MlxBuffer),
}

/// The host plan of `plan_compressed_row`, kept open so each sub-op can be encoded alone.
struct SubPlan {
    compressed_count: usize,
    compressed_write_start: usize,
    indexer_write_start: usize,
    window_capacity: usize,
    physical_kv_len: usize,
    attention_width: usize,
    compressed_positions: MlxBuffer,
    main: MainCompressorArena,
    main_tail: [MlxBuffer; 4],
    indexer: Option<IndexerSub>,
    indices: MlxBuffer,
    attention_indices: MlxBuffer,
    sparse: Sparse,
}

fn tail_buffers(
    device: &MlxDevice,
    projected: usize,
    dim: usize,
    count: usize,
    seed: u64,
) -> [MlxBuffer; 4] {
    let rows = count.max(1);
    [
        f32_buf(device, vec![1, 1, projected], seed, 1.0),
        f32_buf(device, vec![1, 1, projected], seed + 1, 1.0),
        alloc(device, DType::BF16, vec![1, rows, dim], "bench tail output").unwrap(),
        alloc(
            device,
            DType::BF16,
            vec![1, rows, 1, dim],
            "bench tail rope",
        )
        .unwrap(),
    ]
}

fn plan_sub(
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    cache: &Deepseek4Cache,
    layer: usize,
    ratio: u32,
    position: usize,
    step: &LayerCacheStep,
) -> SubPlan {
    let head_dim = cfg.head_dim as usize;
    let graft = cache.graft_slots();
    let lc = &cache.layers()[layer];
    let window_capacity = lc.window_kv.shape()[0];
    let compressed_capacity = lc.compressed_kv.as_ref().unwrap().shape()[0];
    let compressed_count = usize::from(step.compressed_write_slot.is_some());
    let valid = step.compressed_valid_after;
    let layout = prepare_layout(
        ratio,
        1,
        position,
        graft,
        window_capacity,
        valid,
        cfg.index_top_k as usize,
        false,
    )
    .unwrap();
    let mut compressed_positions = alloc_host_input(
        device,
        DType::U32,
        vec![compressed_count.max(1)],
        "bench cpos",
    )
    .unwrap();
    if compressed_count == 1 {
        compressed_positions.as_logical_mut_slice::<u32>().unwrap()[0] =
            (position + 1 - ratio as usize) as u32;
    }
    let mut indices = alloc_host_input(
        device,
        DType::I32,
        vec![1, 1, layout.storage_stride],
        "bench idx",
    )
    .unwrap();
    indices
        .as_logical_mut_slice::<i32>()
        .unwrap()
        .copy_from_slice(&layout.index_storage);
    let attention_width = layout.attention_width;
    let attention_indices = if layout.storage_stride == attention_width {
        indices.with_shape(vec![1, 1, attention_width]).unwrap()
    } else {
        indices
            .slice_view(0, attention_width)
            .with_shape(vec![1, 1, attention_width])
            .unwrap()
    };
    let indexer = (ratio == 4).then(|| {
        let selection = SelectionRequest::new(
            cfg,
            1,
            position,
            valid,
            layout.compressed_attention_offset + graft,
            layout.storage_stride,
            layout.indexer_output_offset.unwrap(),
        )
        .unwrap();
        let arena =
            RatioFourIndexerArena::new(device, cfg, 1, compressed_count, &selection).unwrap();
        let (score_shape, topk_shape) = selection.workspace_shapes().unwrap();
        let heads = cfg.index_num_heads as usize;
        let dim = cfg.index_head_dim as usize;
        IndexerSub {
            tail: tail_buffers(device, 2 * dim, dim, compressed_count, 41),
            query: bf16_buf(device, vec![1, 1, heads, dim], 42, 1.0),
            weights: f32_buf(
                device,
                vec![1, 1, heads],
                43,
                1.0 / ((dim * heads) as f32).sqrt(),
            ),
            scores: alloc(device, DType::F32, score_shape, "bench scores").unwrap(),
            topk: alloc(device, DType::I32, topk_shape, "bench topk").unwrap(),
            selection,
            arena,
        }
    });
    let sparse = if ratio == 4 && attention_width >= SPARSE_FLASH_MIN_TOP_K {
        Sparse::Flash(SparseFlashDecodeArena::new(device, attention_width, head_dim).unwrap())
    } else {
        Sparse::Logits(
            alloc(
                device,
                DType::F32,
                sparse_attention_scratch_shape(1, attention_width),
                "bench logits",
            )
            .unwrap(),
        )
    };
    let projected = (usize::from(ratio == 4) + 1) * head_dim;
    SubPlan {
        compressed_count,
        compressed_write_start: step.compressed_write_slot.unwrap_or(0) + graft,
        indexer_write_start: step.indexer_write_slot.unwrap_or(0),
        window_capacity,
        physical_kv_len: window_capacity + compressed_capacity,
        attention_width,
        compressed_positions,
        main: MainCompressorArena::new(device, cfg, ratio, 1, compressed_count).unwrap(),
        main_tail: tail_buffers(device, projected, head_dim, compressed_count, 40),
        indexer,
        indices,
        attention_indices,
        sparse,
    }
}

fn attention_params(
    cfg: &Deepseek4Config,
    kv_len: usize,
    top_k: usize,
) -> DeepSeekSparseAttentionParams {
    let head_dim = cfg.head_dim as usize;
    DeepSeekSparseAttentionParams {
        batch: 1,
        query_len: 1,
        kv_len: kv_len as u32,
        top_k: top_k as u32,
        heads: cfg.num_attention_heads,
        head_dim: head_dim as u32,
        scale: 1.0 / (head_dim as f32).sqrt(),
    }
}

/// The attention dispatch of `encode_compressed_row`, alone.
fn encode_compressed_attention(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    lc: &LayerCache,
    sp: &SubPlan,
    sinks: &MlxBuffer,
    inputs: &JointRowInputs,
) -> Result<()> {
    let cache_view =
        lc.attention_kv
            .with_shape(vec![1, sp.physical_kv_len, cfg.head_dim as usize])?;
    let params = attention_params(cfg, sp.physical_kv_len, sp.attention_width);
    match &sp.sparse {
        Sparse::Flash(flash) => {
            session.barrier_between(
                &[&inputs.q_rope, &cache_view, sinks, &sp.attention_indices],
                &[
                    &flash.gathered_kv,
                    &flash.mask,
                    &flash.invalid_global,
                    &flash.invalid_heads,
                    &inputs.attention,
                ],
            );
            dispatch_deepseek_sparse_attention_flash_decode(
                session.encoder_mut(),
                registry,
                device,
                &inputs.q_rope,
                &cache_view,
                sinks,
                &sp.attention_indices,
                &flash.gathered_kv,
                &flash.mask,
                &flash.invalid_global,
                &flash.invalid_heads,
                &inputs.attention,
                &params,
            )?;
        }
        Sparse::Logits(logits) => {
            session.barrier_between(
                &[&inputs.q_rope, &cache_view, sinks, &sp.attention_indices],
                &[&inputs.attention],
            );
            dispatch_deepseek_sparse_attention_with_scratch(
                session.encoder_mut(),
                registry,
                device,
                &inputs.q_rope,
                &cache_view,
                sinks,
                &sp.attention_indices,
                logits,
                &inputs.attention,
                &params,
            )?;
        }
    }
    Ok(())
}

fn encode_select(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    lc: &LayerCache,
    sp: &SubPlan,
) -> Result<()> {
    let ix = sp.indexer.as_ref().unwrap();
    encode_selection(
        session,
        registry,
        device,
        &ix.selection,
        &ix.query,
        lc.indexer_kv.as_ref().unwrap(),
        &ix.weights,
        &ix.scores,
        &ix.topk,
        &sp.indices,
    )
}

/// Everything one (layer kind, context, position, B) case shares.
struct Case<'a> {
    cfg: &'a Deepseek4Config,
    caches: &'a [Deepseek4Cache],
    steps: &'a [CacheStep],
    weights: &'a [Option<LayerWeights>],
    unused: &'a Unused,
    frequencies: &'a MlxBuffer,
    batch: &'a RowBatch,
    ratio: u32,
    position: usize,
    b: usize,
}

impl Case<'_> {
    fn lc(&self, row: usize, layer: usize) -> &LayerCache {
        &self.caches[row].layers()[layer]
    }
    fn step(&self, row: usize, layer: usize) -> &LayerCacheStep {
        &self.steps[row].layers[layer]
    }
}

fn window_cells(
    ctx: &mut GpuContext,
    out: &mut Out,
    base: &Value,
    case: &Case<'_>,
    layers: &[usize],
) {
    let cfg = case.cfg;
    let head_dim = cfg.head_dim as usize;
    let device = ctx.device().clone();
    let n = layers.len();
    let b = case.b;
    let (cores, plan_us) = plan_rows(n, || {
        layers
            .iter()
            .flat_map(|&layer| (0..b).map(move |row| (layer, row)))
            .map(|(layer, row)| {
                plan_window_row(&device, &case.caches[row], layer, case.position, head_dim).unwrap()
            })
            .collect()
    });
    let t = time_op(ctx, |s, r, d, rep| {
        let layer = layers[rep % n];
        let sinks = &case.weights[layer].as_ref().unwrap().sinks;
        for row in 0..b {
            encode_window_row(
                s,
                r,
                d,
                cfg,
                case.lc(row, layer),
                case.step(row, layer),
                &cores[(rep % n) * b + row],
                sinks,
                &case.batch.rows[row],
            )?;
        }
        Ok(())
    });
    out.cell(base, json!({"op": "core", "plan_us": plan_us}), &t);

    let t = time_op(ctx, |s, r, d, rep| {
        let layer = layers[rep % n];
        for row in 0..b {
            let lc = case.lc(row, layer);
            let inputs = &case.batch.rows[row];
            encode_window_write(
                s,
                r,
                d,
                lc,
                &inputs.kv_rope,
                &[&inputs.kv_rope],
                None,
                Some(case.step(row, layer)),
                1,
                lc.window_kv.shape()[0],
                head_dim,
            )?;
        }
        Ok(())
    });
    out.cell(base, json!({"op": "window_write"}), &t);

    let window_capacity = case.lc(0, layers[0]).window_kv.shape()[0];
    let index_row = window_indices(window_capacity, 1, case.position)
        .unwrap()
        .remove(0);
    let width = index_row.len();
    let mut indices =
        alloc_host_input(&device, DType::I32, vec![1, 1, width], "bench idx").unwrap();
    indices
        .as_logical_mut_slice::<i32>()
        .unwrap()
        .copy_from_slice(&index_row);
    let t = time_op(ctx, |s, r, d, rep| {
        let layer = layers[rep % n];
        let sinks = &case.weights[layer].as_ref().unwrap().sinks;
        for row in 0..b {
            let inputs = &case.batch.rows[row];
            let cache_view =
                case.lc(row, layer)
                    .window_kv
                    .with_shape(vec![1, window_capacity, head_dim])?;
            s.barrier_between(
                &[&inputs.q_rope, &cache_view, sinks, &indices],
                &[&inputs.attention],
            );
            dispatch_deepseek_sparse_attention(
                s.encoder_mut(),
                r,
                d,
                &inputs.q_rope,
                &cache_view,
                sinks,
                &indices,
                &inputs.attention,
                &attention_params(cfg, window_capacity, width),
            )?;
        }
        Ok(())
    });
    out.cell(
        base,
        json!({"op": "attention", "kernel": "sparse_score_reduce", "attention_width": width}),
        &t,
    );
}

fn compressed_cells(
    ctx: &mut GpuContext,
    out: &mut Out,
    base: &Value,
    case: &Case<'_>,
    layers: &[usize],
) {
    let cfg = case.cfg;
    let head_dim = cfg.head_dim as usize;
    let device = ctx.device().clone();
    let n = layers.len();
    let b = case.b;
    let ratio = case.ratio;
    let position = case.position;
    let refs: Vec<CompressedAttentionWeightsRef<'_>> = layers
        .iter()
        .map(|&layer| {
            case.weights[layer]
                .as_ref()
                .unwrap()
                .compressed(case.unused)
        })
        .collect();
    let slot = |rep: usize, row: usize| (rep % n) * b + row;

    // (a) the whole per-row core, exactly as joint decode plans and encodes it.
    let (cores, plan_us) = plan_rows(n, || {
        layers
            .iter()
            .flat_map(|&layer| (0..b).map(move |row| (layer, row)))
            .map(|(layer, row)| {
                plan_compressed_row(
                    &device,
                    cfg,
                    &case.caches[row],
                    layer,
                    ratio,
                    position,
                    case.step(row, layer),
                )
                .unwrap()
            })
            .collect()
    });
    let t = time_op(ctx, |s, r, d, rep| {
        let layer = layers[rep % n];
        for row in 0..b {
            encode_compressed_row(
                s,
                r,
                d,
                cfg,
                case.lc(row, layer),
                case.step(row, layer),
                &cores[slot(rep, row)],
                &refs[rep % n],
                case.frequencies,
                &case.batch.rows[row],
            )?;
        }
        Ok(())
    });
    out.cell(base, json!({"op": "core", "plan_us": plan_us}), &t);
    drop(cores);

    // (b) sub-ops through the entry points `encode_compressed_row` calls.
    let subs: Vec<SubPlan> = layers
        .iter()
        .flat_map(|&layer| (0..b).map(move |row| (layer, row)))
        .map(|(layer, row)| {
            plan_sub(
                &device,
                cfg,
                &case.caches[row],
                layer,
                ratio,
                position,
                case.step(row, layer),
            )
        })
        .collect();
    let width = subs[0].attention_width;
    let valid = case.step(0, layers[0]).compressed_valid_after;

    let t = time_op(ctx, |s, r, d, rep| {
        let layer = layers[rep % n];
        for row in 0..b {
            let inputs = &case.batch.rows[row];
            encode_window_write(
                s,
                r,
                d,
                case.lc(row, layer),
                &inputs.kv_rope,
                &[&inputs.kv_rope],
                None,
                Some(case.step(row, layer)),
                1,
                subs[slot(rep, row)].window_capacity,
                head_dim,
            )?;
        }
        Ok(())
    });
    out.cell(base, json!({"op": "window_write"}), &t);

    let t = time_op(ctx, |s, r, d, rep| {
        let layer = layers[rep % n];
        for row in 0..b {
            let sp = &subs[slot(rep, row)];
            encode_main_compressor(
                s,
                r,
                d,
                cfg,
                ratio,
                1,
                position,
                sp.compressed_write_start,
                sp.compressed_count,
                case.lc(row, layer),
                &case.batch.rows[row].attn_norm,
                &sp.compressed_positions,
                case.frequencies,
                &refs[rep % n].compressor,
                &sp.main,
            )?;
        }
        Ok(())
    });
    out.cell(base, json!({"op": "main_compressor"}), &t);

    let t = time_op(ctx, |s, r, d, rep| {
        let layer = layers[rep % n];
        for row in 0..b {
            let sp = &subs[slot(rep, row)];
            let lc = case.lc(row, layer);
            let cache = lc.compressed_kv.as_ref().unwrap();
            let [kv, score, output, rope] = &sp.main_tail;
            encode_main_tail(
                s,
                r,
                d,
                cfg,
                ratio,
                1,
                position,
                sp.compressed_write_start,
                sp.compressed_count,
                cache,
                cache.shape()[0],
                lc.main_kv_state.as_ref().unwrap(),
                lc.main_score_state.as_ref().unwrap(),
                &CompressorTail {
                    kv,
                    score,
                    output,
                    rope,
                    ape: refs[rep % n].compressor.ape,
                    norm: refs[rep % n].compressor.norm,
                },
                &sp.compressed_positions,
                case.frequencies,
            )?;
        }
        Ok(())
    });
    out.cell(base, json!({"op": "main_compressor_tail"}), &t);

    if ratio == 4 {
        let t = time_op(ctx, |s, r, d, rep| {
            let layer = layers[rep % n];
            for row in 0..b {
                let sp = &subs[slot(rep, row)];
                let ix = sp.indexer.as_ref().unwrap();
                let inputs = &case.batch.rows[row];
                encode_ratio_four_indexer(
                    s,
                    r,
                    d,
                    cfg,
                    1,
                    position,
                    sp.indexer_write_start,
                    sp.compressed_count,
                    &ix.selection,
                    case.lc(row, layer),
                    &inputs.attn_norm,
                    &inputs.q_a_norm,
                    &inputs.position,
                    &sp.compressed_positions,
                    case.frequencies,
                    refs[rep % n].indexer.as_ref().unwrap(),
                    &ix.arena,
                    &sp.indices,
                )?;
            }
            Ok(())
        });
        out.cell(base, json!({"op": "indexer"}), &t);

        let t = time_op(ctx, |s, r, d, rep| {
            let layer = layers[rep % n];
            for row in 0..b {
                let sp = &subs[slot(rep, row)];
                let ix = sp.indexer.as_ref().unwrap();
                let lc = case.lc(row, layer);
                let cache = lc.indexer_kv.as_ref().unwrap();
                let compressor = &refs[rep % n].indexer.as_ref().unwrap().compressor;
                let [kv, score, output, rope] = &ix.tail;
                encode_indexer_tail(
                    s,
                    r,
                    d,
                    cfg,
                    1,
                    position,
                    sp.indexer_write_start,
                    sp.compressed_count,
                    cache,
                    cache.shape()[0],
                    lc.indexer_kv_state.as_ref().unwrap(),
                    lc.indexer_score_state.as_ref().unwrap(),
                    &CompressorTail {
                        kv,
                        score,
                        output,
                        rope,
                        ape: compressor.ape,
                        norm: compressor.norm,
                    },
                    &sp.compressed_positions,
                    case.frequencies,
                )?;
            }
            Ok(())
        });
        out.cell(base, json!({"op": "indexer_compressor_tail"}), &t);

        let t = time_op(ctx, |s, r, d, rep| {
            let layer = layers[rep % n];
            for row in 0..b {
                encode_select(s, r, d, case.lc(row, layer), &subs[slot(rep, row)])?;
            }
            Ok(())
        });
        out.cell(
            base,
            json!({"op": "indexer_select", "kernel": "indexer_into_mma_tiled", "index_keys": valid,
                   "top_k": cfg.index_top_k}),
            &t,
        );

        // The attention-only cell must read real top-k indices, not the -1 placeholders.
        run_once(ctx, |s, r, d| {
            for (i, sp) in subs.iter().enumerate() {
                encode_select(s, r, d, case.lc(i % b, layers[i / b]), sp)?;
            }
            Ok(())
        });
    }

    let t = time_op(ctx, |s, r, d, rep| {
        let layer = layers[rep % n];
        for row in 0..b {
            encode_compressed_attention(
                s,
                r,
                d,
                cfg,
                case.lc(row, layer),
                &subs[slot(rep, row)],
                refs[rep % n].attention.sinks,
                &case.batch.rows[row],
            )?;
        }
        Ok(())
    });
    let kernel = match subs[0].sparse {
        Sparse::Flash(_) => "sparse_flash_decode",
        Sparse::Logits(_) => "sparse_score_reduce",
    };
    out.cell(
        base,
        json!({"op": "attention", "kernel": kernel, "attention_width": width,
               "physical_kv_len": subs[0].physical_kv_len}),
        &t,
    );
}

/// One HC site's buffers at B rows, encoded with the calls of `encode_compressed_attention_prelude`
/// (norm, hc_fn, split + Sinkhorn, pre) and `encode_compressed_attention_epilogue` (post).
struct HcRows<'a> {
    rows: usize,
    hidden: usize,
    hc_hidden: usize,
    mix: usize,
    hc_fn: &'a [MlxBuffer],
    fn_shape: &'a [usize; 2],
    hc_scale: &'a MlxBuffer,
    hc_base: &'a MlxBuffer,
    params: &'a MlxBuffer,
    state: MlxBuffer,
    state_flat: MlxBuffer,
    state_norm: MlxBuffer,
    mixes: MlxBuffer,
    pre: MlxBuffer,
    post: MlxBuffer,
    comb: MlxBuffer,
    attn_input: MlxBuffer,
    x: MlxBuffer,
    output: MlxBuffer,
}

impl HcRows<'_> {
    fn norm(&self, s: &mut GraphSession<'_>, r: &mut KernelRegistry, d: &MlxDevice) -> Result<()> {
        prefill_annotations::rms_norm_no_scale_f32(
            s,
            r,
            d.metal_device(),
            &self.state_flat,
            &self.state_norm,
            self.params,
            self.rows as u32,
            self.hc_hidden as u32,
        )?;
        Ok(())
    }

    fn mixes(
        &self,
        s: &mut GraphSession<'_>,
        r: &mut KernelRegistry,
        d: &MlxDevice,
        rep: usize,
    ) -> Result<()> {
        let w = RawMatrixRef {
            // `rep % copies` with copies = len (cold) or 1 (warm), set by the caller's slice.
            buffer: &self.hc_fn[rep % self.hc_fn.len()],
            ggml_type: GgmlType::F16,
            shape: self.fn_shape,
            matvec: bench_matvec(),
        };
        row_matmul(
            RowMatmul::IndependentRows,
            s,
            r,
            d,
            &self.state_norm,
            &w,
            &self.mixes,
            self.rows,
            self.mix,
            self.hc_hidden,
            "bench HC function",
        )
    }

    fn sinkhorn(
        &self,
        s: &mut GraphSession<'_>,
        r: &mut KernelRegistry,
        d: &MlxDevice,
    ) -> Result<()> {
        s.barrier_between(
            &[&self.mixes, self.hc_scale, self.hc_base],
            &[&self.pre, &self.post, &self.comb],
        );
        dispatch_hc_split_sinkhorn(
            s.encoder_mut(),
            r,
            d,
            &self.mixes,
            self.hc_scale,
            self.hc_base,
            &self.pre,
            &self.post,
            &self.comb,
            self.rows as u32,
        )?;
        Ok(())
    }

    fn pre(&self, s: &mut GraphSession<'_>, r: &mut KernelRegistry, d: &MlxDevice) -> Result<()> {
        s.barrier_between(&[&self.state, &self.pre], &[&self.attn_input]);
        dispatch_hc_pre(
            s.encoder_mut(),
            r,
            d,
            &self.state,
            &self.pre,
            &self.attn_input,
            self.rows as u32,
            self.hidden as u32,
        )?;
        Ok(())
    }

    fn post(&self, s: &mut GraphSession<'_>, r: &mut KernelRegistry, d: &MlxDevice) -> Result<()> {
        s.barrier_between(
            &[&self.x, &self.state, &self.post, &self.comb],
            &[&self.output],
        );
        dispatch_hc_post(
            s.encoder_mut(),
            r,
            d,
            &self.x,
            &self.state,
            &self.post,
            &self.comb,
            &self.output,
            self.rows as u32,
            self.hidden as u32,
        )?;
        Ok(())
    }
}

pub(super) fn hc_cells(ctx: &mut GpuContext, out: &mut Out, cfg: &Deepseek4Config, rows: &[usize]) {
    let device = ctx.device().clone();
    let hidden = cfg.hidden_size as usize;
    let hc = cfg.hyper_connection_count as usize;
    let hc_hidden = hc * hidden;
    let mix = (2 + hc) * hc;
    let fn_shape = [mix, hc_hidden];
    let fn_bytes = mix * hc_hidden * 2;
    let copies = copies_for(fn_bytes, false);
    let hc_fn: Vec<MlxBuffer> = (0..copies)
        .map(|c| f16_matrix(&device, mix, hc_hidden, 500 + c as u64))
        .collect();
    let hc_scale = f32_buf(&device, vec![3], 61, 0.5);
    let hc_base = f32_buf(&device, vec![mix], 62, 0.5);
    let params = rms_params(&device, cfg.rms_norm_eps, hc_hidden, "bench hc params").unwrap();
    for &b in rows {
        let state = f32_buf(&device, vec![b, hc, hidden], 63, 1.0);
        let mut h = HcRows {
            rows: b,
            hidden,
            hc_hidden,
            mix,
            hc_fn: &hc_fn,
            fn_shape: &fn_shape,
            hc_scale: &hc_scale,
            hc_base: &hc_base,
            params: &params,
            state_flat: state.with_shape(vec![b, hc_hidden]).unwrap(),
            state,
            state_norm: alloc(&device, DType::F32, vec![b, hc_hidden], "bench").unwrap(),
            mixes: f32_buf(&device, vec![b, mix], 64, 0.5),
            pre: alloc(&device, DType::F32, vec![b, hc], "bench").unwrap(),
            post: alloc(&device, DType::F32, vec![b, hc], "bench").unwrap(),
            comb: alloc(&device, DType::F32, vec![b, hc, hc], "bench").unwrap(),
            attn_input: alloc(&device, DType::F32, vec![b, hidden], "bench").unwrap(),
            x: f32_buf(&device, vec![b, hidden], 65, 1.0),
            output: alloc(&device, DType::F32, vec![b, hc, hidden], "bench").unwrap(),
        };
        // Populate pre/post/comb once so the pre/post cells read Sinkhorn outputs.
        run_once(ctx, |s, r, d| h.sinkhorn(s, r, d));
        let base = json!({"family": "hc", "b": b, "hc": hc, "hidden": hidden,
                          "sinkhorn_iterations": cfg.hyper_connection_sinkhorn_iterations});
        // Weightless ops: one set of activations, warm by construction.
        let t = time_op(ctx, |s, r, d, _| h.norm(s, r, d));
        out.cell(&base, json!({"op": "hc_norm", "warm": true}), &t);
        let t = time_op(ctx, |s, r, d, _| h.sinkhorn(s, r, d));
        out.cell(&base, json!({"op": "hc_split_sinkhorn", "warm": true}), &t);
        let t = time_op(ctx, |s, r, d, _| h.pre(s, r, d));
        out.cell(&base, json!({"op": "hc_pre", "warm": true}), &t);
        let t = time_op(ctx, |s, r, d, _| h.post(s, r, d));
        out.cell(&base, json!({"op": "hc_post", "warm": true}), &t);
        for warm in [false, true] {
            let n = if warm { 1 } else { copies };
            h.hc_fn = &hc_fn[..n];
            let t = time_op(ctx, |s, r, d, rep| h.mixes(s, r, d, rep));
            out.cell(
                &base,
                json!({"op": "hc_fn", "warm": warm, "weight_bytes": fn_bytes, "copies": n}),
                &t,
            );
            let t = time_op(ctx, |s, r, d, rep| {
                h.norm(s, r, d)?;
                h.mixes(s, r, d, rep)?;
                h.sinkhorn(s, r, d)?;
                h.pre(s, r, d)
            });
            out.cell(
                &base,
                json!({"op": "hc_pre_site", "parts": "norm+hc_fn+split_sinkhorn+pre", "warm": warm,
                       "copies": n}),
                &t,
            );
        }
    }
}

#[test]
#[ignore = "model-free GPU microbenchmarks (gap program M2b; standing grant, <= 15 min)"]
fn ds4_attn_microbench() {
    let out_path = std::env::var("HF2Q_DS4_ATTN_BENCH_OUT").expect("set HF2Q_DS4_ATTN_BENCH_OUT");
    let mut out = Out(std::fs::File::create(&out_path).unwrap());
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut ctx = GpuContext::new().unwrap();
    let device = ctx.device().clone();
    let cfg = super::cache_tests::config(ds4_ratios());

    gpu_warmup(&mut ctx);
    floor_cells(&mut ctx, &mut out);
    hc_cells(&mut ctx, &mut out, &cfg, &(1..=MAX_ROWS).collect::<Vec<_>>());

    let plan = Deepseek4CachePlan::for_context(&cfg, CONTEXT).unwrap();
    let mut caches: Vec<Deepseek4Cache> = (0..MAX_ROWS)
        .map(|row| synthetic_cache(&device, &plan, row as u64 + 1))
        .collect();
    let unused = Unused {
        buffer: alloc(&device, DType::F32, vec![1], "bench unused").unwrap(),
        shape: [1, 1],
    };
    let rope = cfg.rope_head_dim as usize;
    let window_frequencies = host_table(
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
    );
    let compressed_frequencies = host_table(
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

    // Distinct layers to rotate over for cold cells: enough weight bytes to exceed COLD_BYTES.
    let mut weights: Vec<Option<LayerWeights>> = (0..LAYERS).map(|_| None).collect();
    let mut layer_sets: Vec<(&str, u32, Vec<usize>)> = Vec::new();
    for (kind, ratio) in [("window", 0u32), ("hca", 128), ("csa", 4)] {
        let all: Vec<usize> = (0..LAYERS)
            .filter(|&l| cfg.compress_ratios[l] == ratio)
            .collect();
        let mut chosen = Vec::new();
        let mut bytes = 0usize;
        for &layer in &all {
            let w = LayerWeights::new(&device, &cfg, ratio, layer as u64);
            bytes += w.bytes();
            weights[layer] = Some(w);
            chosen.push(layer);
            if bytes >= COLD_BYTES {
                break;
            }
        }
        eprintln!(
            "{kind}: {} layers, {} MiB weights",
            chosen.len(),
            bytes >> 20
        );
        layer_sets.push((kind, ratio, chosen));
    }

    for (kind, ratio, layers) in &layer_sets {
        let (kind, ratio) = (*kind, *ratio);
        // (context, the peer's decode FLASH_ATTN_EXT kv, its n_kv_max)
        let contexts: &[(usize, usize, usize)] = match ratio {
            0 => &[(2048, 1280, 0)],
            128 => &[(2048, 1536, 0), (8192, 1536, 0)],
            _ => &[(2048, 1792, 640), (8192, 3328, 640)],
        };
        for &(context, peer_kv, peer_n_kv_max) in contexts {
            let positions: &[(usize, bool)] = if ratio == 0 {
                &[(context - 1, false)]
            } else {
                &[(context - 2, false), (context - 1, true)]
            };
            for &(position, emit) in positions {
                for cache in &mut caches {
                    cache.next_position = position;
                }
                let steps: Vec<CacheStep> =
                    caches.iter().map(|c| c.plan_next_step().unwrap()).collect();
                let valid = steps[0].layers[layers[0]].compressed_valid_after;
                if ratio != 0 {
                    assert_eq!(
                        steps[0].layers[layers[0]].compressed_write_slot.is_some(),
                        emit,
                        "{kind} position {position} emit drift"
                    );
                }
                let frequencies = if ratio == 0 {
                    &window_frequencies
                } else {
                    &compressed_frequencies
                };
                for b in 1..=MAX_ROWS {
                    let batch = RowBatch::new(&device, &cfg, b, position);
                    let case = Case {
                        cfg: &cfg,
                        caches: &caches,
                        steps: &steps,
                        weights: &weights,
                        unused: &unused,
                        frequencies,
                        batch: &batch,
                        ratio,
                        position,
                        b,
                    };
                    for warm in [false, true] {
                        let set: &[usize] = if warm { &layers[..1] } else { layers };
                        let base = json!({"family": "attn", "kind": kind, "ratio": ratio, "context": context,
                                          "position": position, "emit": emit, "valid_compressed": valid,
                                          "b": b, "warm": warm, "layers_rotated": set.len(),
                                          "peer_kv": peer_kv, "peer_n_kv_max": peer_n_kv_max});
                        if ratio == 0 {
                            window_cells(&mut ctx, &mut out, &base, &case, set);
                        } else {
                            compressed_cells(&mut ctx, &mut out, &base, &case, set);
                        }
                    }
                }
            }
        }
    }
    eprintln!("cells: {out_path}");
}
