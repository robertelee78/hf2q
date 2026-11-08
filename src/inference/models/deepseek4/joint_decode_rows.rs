//! Phase-major, batched cache-dependent attention cores for joint decode
//! (`HF2Q_DEEPSEEK_BATCHED_ATTN_CORES=1`, ADR-063 gap program fix B stage 1).
//!
//! `joint_decode_core` encodes one row at a time: window write, compressor,
//! indexer and sparse attention for row 0, then row 1, and so on. Here each
//! phase runs once for all rows of a layer through mlx-native's rows
//! kernels (`ops::deepseek_rows`), which run the one-row kernels' bodies for
//! up to eight rows, each against its own cache through a fixed binding
//! slot. Every row still reads only its own cache at its own position, so
//! the order across rows is free and each row's arithmetic is the one-row
//! graph's:
//! - window writes stay one `kv_cache_copy` per row;
//! - compressor and indexer F16 projections: one `m = B` stock-order
//!   dispatch (`row_matmul` -> `row_exact_matvec_stock`, every row bitwise the
//!   stock one-row matvec) under `HF2Q_DEEPSEEK_ROW_EXACT_MATVEC=1`; on the stock family the
//!   per-row one-row matvecs, each into its own buffer (no write hazard
//!   between rows), then one exact row gather (`project_rows`);
//! - compressor recurrences, indexer score + top-k and sparse attention are
//!   one rows dispatch each (the flash path once per distinct width);
//! - the indexer query RoPE, Hadamard and weight scale and the compressed-row
//!   RoPE, quantization and Hadamard are row-independent elementwise kernels
//!   issued once over all rows; each emitting row then copies its row into
//!   its own cache.
//!
//! Joint rows therefore stay bitwise equal to `forward_verifier_one` on the
//! same cache (`joint_decode_rows_tests.rs`). Layers with a row at position
//! zero (no recurrent state yet) keep the per-row cores.

use anyhow::{ensure, Context, Result};
use mlx_native::graph::GraphSession;
use mlx_native::ops::deepseek_activation_quant::{
    dispatch_deepseek_hadamard_mxfp4_bf16, dispatch_deepseek_mxfp8_fake_quant_bf16,
    DeepSeekMxfp8Params,
};
use mlx_native::ops::deepseek_rows::{
    deepseek_indexer_rows_workspace_elements, deepseek_sparse_attention_rows_scratch_elements,
    deepseek_sparse_flash_rows_contiguous, dispatch_deepseek_compressor_rows,
    dispatch_deepseek_indexer_rows, dispatch_deepseek_rows_gather_f32,
    dispatch_deepseek_sparse_attention_flash_rows, dispatch_deepseek_sparse_attention_rows,
    DeepSeekCompressorRow, DeepSeekIndexerRow, DeepSeekSparseAttentionRow,
    DeepSeekSparseFlashRowsWorkspace, DEEPSEEK_ROWS_MAX,
};
use mlx_native::ops::deepseek_sparse_attention::DEEPSEEK_SPARSE_HEADS;
use mlx_native::ops::deepseek_tail_rope::{
    dispatch_deepseek_tail_rope_bf16, DeepSeekTailRopeParams,
};
use mlx_native::ops::elementwise::scalar_mul_f32;
use mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, KernelRegistry, MlxBuffer, MlxDevice};

use super::attention::window_indices;
use super::cache::{LayerCache, LayerCacheStep};
use super::cache_window_write::encode_window_write;
use super::compressed_attention::SPARSE_FLASH_MIN_TOP_K;
use super::compressed_attention_indexer::layout::prepare_layout;
use super::compressed_attention_weights::{
    CompressedAttentionWeightsRef, CompressorWeightsRef, IndexerWeightsRef,
};
use super::forward_support::{
    alloc, alloc_host_input, raw_matmul, row_exact_storage, ROW_EXACT_MAX_ROWS,
};
use super::joint_decode_routes::{joint_route_is_exact, JointProduct};
use super::prefill_annotations;
use super::residency::RawMatrixRef;
use super::row_matmul::{row_matmul, row_view, splits_independent_rows, RowMatmul};
use super::Deepseek4Config;

/// How joint decode encodes the per-row, cache-dependent attention cores.
/// Chosen once at model load.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AttnCores {
    /// One row at a time (`joint_decode_core`).
    #[default]
    PerRow,
    /// Phase-major rows dispatches over all rows of a layer (this module).
    Batched,
}

impl AttnCores {
    /// Process switch read by [`Self::from_env`].
    pub const ENV: &'static str = "HF2Q_DEEPSEEK_BATCHED_ATTN_CORES";

    /// `HF2Q_DEEPSEEK_BATCHED_ATTN_CORES`: `1` selects [`Self::Batched`],
    /// `0` or unset [`Self::PerRow`]; anything else is an error.
    pub fn from_env() -> Result<Self> {
        Self::resolve(std::env::var_os(Self::ENV).as_deref())
    }

    pub fn resolve(value: Option<&std::ffi::OsStr>) -> Result<Self> {
        match value.map(|value| value.to_str()) {
            None | Some(Some("0")) => Ok(Self::PerRow),
            Some(Some("1")) => Ok(Self::Batched),
            Some(_) => anyhow::bail!("{} must be 0 or 1 (got {value:?})", Self::ENV),
        }
    }
}

/// The joint step's batched buffers, row `r` of each belonging to row `r`.
pub(super) struct JointBatchInputs<'a> {
    /// BF16 `[1, B, 64, 512]`.
    pub q_rope: &'a MlxBuffer,
    /// BF16 `[1, B, 1, 512]`.
    pub kv_rope: &'a MlxBuffer,
    /// BF16 `[1, B, 64, 512]`, written.
    pub attention: &'a MlxBuffer,
    /// F32 `[B, hidden]`.
    pub attn_norm: &'a MlxBuffer,
    /// F32 `[B, q_lora_rank]`.
    pub q_a_norm: &'a MlxBuffer,
    /// U32 `[B]` token positions.
    pub positions: &'a MlxBuffer,
}

/// Whether a layer of `ratio` can take the batched cores at these positions.
pub(super) fn batched_cores_admit(ratio: u32, positions: &[usize]) -> bool {
    (1..=DEEPSEEK_ROWS_MAX).contains(&positions.len())
        && (ratio == 0 || positions.iter().all(|&position| position > 0))
}

fn host_i32(device: &MlxDevice, values: &[i32], label: &str) -> Result<MlxBuffer> {
    let mut buffer = alloc_host_input(device, DType::I32, vec![values.len().max(1)], label)?;
    buffer.as_logical_mut_slice::<i32>()?[..values.len()].copy_from_slice(values);
    Ok(buffer)
}

fn host_zero_u32(device: &MlxDevice, len: usize, label: &str) -> Result<MlxBuffer> {
    let mut buffer = alloc_host_input(device, DType::U32, vec![len.max(1)], label)?;
    buffer.as_logical_mut_slice::<u32>()?.fill(0);
    Ok(buffer)
}

fn sparse_scale(cfg: &Deepseek4Config) -> f32 {
    1.0 / (cfg.head_dim as f32).sqrt()
}

/// Whether a batched compressor/indexer projection over `rows` rows may run
/// as one `m = rows` dispatch. Joint-decode admission
/// (`joint_decode_routes::joint_product_names`) assumes these projections
/// run per row (`m = 1`, exact for any storage), so a single dispatch is
/// allowed only where its rows are proven equal to `m = 1`: the stock-order
/// row-exact route, one row, or a storage whose `m = rows` route joint
/// decode already proves (`joint_route_is_exact(RowMatmul, _)`) and that
/// `row_matmul` does not split. Everything else runs per row.
pub(super) fn projection_takes_single_dispatch(
    row_exact: bool,
    ggml_type: GgmlType,
    rows: usize,
) -> bool {
    row_exact
        || rows == 1
        || (!splits_independent_rows(ggml_type, rows)
            && joint_route_is_exact(JointProduct::RowMatmul, ggml_type))
}

/// A compressor/indexer projection of `input [B, k]` into `output [B, n]`.
/// When [`projection_takes_single_dispatch`] holds, one
/// `row_matmul(IndependentRows)`; otherwise one-row products (the per-row
/// cores' `m = 1` route, exact for any storage), each into its own buffer so
/// the B products carry no write hazard on each other and overlap, then one
/// exact row gather into the contiguous operand.
#[allow(clippy::too_many_arguments)]
fn project_rows(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    input: &MlxBuffer,
    weight: &RawMatrixRef<'_>,
    output: &MlxBuffer,
    rows: usize,
    n: usize,
    k: usize,
    label: &str,
) -> Result<()> {
    // Same row-exact decision as `row_matmul` at these rows.
    let row_exact = row_exact_storage(registry, weight, rows.min(ROW_EXACT_MAX_ROWS)).is_some();
    if projection_takes_single_dispatch(row_exact, weight.ggml_type, rows) {
        return row_matmul(
            RowMatmul::IndependentRows,
            session,
            registry,
            device,
            input,
            weight,
            output,
            rows,
            n,
            k,
            label,
        );
    }
    let mut row_outputs = Vec::with_capacity(rows);
    for row in 0..rows {
        let row_output = alloc(device, DType::F32, vec![1, n], label)?;
        raw_matmul(
            session,
            registry,
            device,
            &row_view(input, row, vec![1, k])?,
            weight,
            &row_output,
            1,
            n,
            k,
            label,
        )?;
        row_outputs.push(row_output);
    }
    let sources: Vec<&MlxBuffer> = row_outputs.iter().collect();
    session.barrier_between(&sources, &[output]);
    dispatch_deepseek_rows_gather_f32(
        session.encoder_mut(),
        registry,
        device,
        &sources,
        output,
        n,
    )?;
    Ok(())
}

/// Every row's window write (one `kv_cache_copy` per row, as the per-row cores).
pub(super) fn encode_window_writes(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    layer_caches: &[&LayerCache],
    steps: &[&LayerCacheStep],
    kv_rope: &MlxBuffer,
) -> Result<()> {
    let head_dim = cfg.head_dim as usize;
    for (row, (layer_cache, step)) in layer_caches.iter().zip(steps).enumerate() {
        let input = row_view(kv_rope, row, vec![1, 1, 1, head_dim])?;
        encode_window_write(
            session,
            registry,
            device,
            layer_cache,
            &input,
            &[&input],
            None,
            Some(step),
            1,
            layer_cache.window_kv.shape()[0],
            head_dim,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sliding-window layers
// ---------------------------------------------------------------------------

/// Host-prepared state of one uncompressed layer for every joint row.
pub(super) struct JointWindowLayer {
    window_capacity: usize,
    widths: Vec<usize>,
    offsets: Vec<usize>,
    indices: MlxBuffer,
    scratch: MlxBuffer,
}

pub(super) fn plan_window_layer(
    device: &MlxDevice,
    layer_caches: &[&LayerCache],
    positions: &[usize],
    head_dim: usize,
) -> Result<JointWindowLayer> {
    ensure!(
        batched_cores_admit(0, positions) && layer_caches.len() == positions.len(),
        "DeepSeek-V4 batched window cores need 1..={DEEPSEEK_ROWS_MAX} aligned rows"
    );
    let window_capacity = layer_caches[0].window_kv.shape()[0];
    let mut packed = Vec::new();
    let mut widths = Vec::with_capacity(positions.len());
    let mut offsets = Vec::with_capacity(positions.len());
    for (layer_cache, &position) in layer_caches.iter().zip(positions) {
        ensure!(
            layer_cache.window_kv.shape() == [window_capacity, head_dim],
            "DeepSeek-V4 window cache shape drift across joint rows"
        );
        let row = window_indices(window_capacity, 1, position)?
            .into_iter()
            .next()
            .context("DeepSeek-V4 window index row missing")?;
        offsets.push(packed.len());
        widths.push(row.len());
        packed.extend(row);
    }
    let max_width = widths.iter().copied().max().unwrap_or(1);
    let scratch_elements =
        deepseek_sparse_attention_rows_scratch_elements(positions.len(), max_width);
    Ok(JointWindowLayer {
        window_capacity,
        widths,
        offsets,
        indices: host_i32(device, &packed, "joint rows window indices")?,
        scratch: alloc(
            device,
            DType::F32,
            vec![scratch_elements],
            "joint rows window logits",
        )?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_window_layer(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    layer_caches: &[&LayerCache],
    steps: &[&LayerCacheStep],
    plan: &JointWindowLayer,
    sinks: &MlxBuffer,
    inputs: &JointBatchInputs<'_>,
) -> Result<()> {
    encode_window_writes(
        session,
        registry,
        device,
        cfg,
        layer_caches,
        steps,
        inputs.kv_rope,
    )?;
    plan.encode_attention(session, registry, device, cfg, layer_caches, sinks, inputs)
}

impl JointWindowLayer {
    /// Per-row attention widths.
    #[cfg(test)]
    pub(super) fn widths(&self) -> &[usize] {
        &self.widths
    }

    /// Sparse attention over every row's window (after the window writes).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_attention(
        &self,
        session: &mut GraphSession<'_>,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        cfg: &Deepseek4Config,
        layer_caches: &[&LayerCache],
        sinks: &MlxBuffer,
        inputs: &JointBatchInputs<'_>,
    ) -> Result<()> {
        let plan = self;
        let rows: Vec<DeepSeekSparseAttentionRow<'_>> = layer_caches
            .iter()
            .enumerate()
            .map(|(row, layer_cache)| DeepSeekSparseAttentionRow {
                kv: &layer_cache.window_kv,
                kv_len: plan.window_capacity,
                width: plan.widths[row],
                index_offset: plan.offsets[row],
                q_row: row,
            })
            .collect();
        let mut reads: Vec<&MlxBuffer> = vec![inputs.q_rope, sinks, &plan.indices];
        reads.extend(layer_caches.iter().map(|cache| &cache.window_kv));
        session.barrier_between(&reads, &[inputs.attention, &plan.scratch]);
        dispatch_deepseek_sparse_attention_rows(
            session.encoder_mut(),
            registry,
            device,
            inputs.q_rope,
            sinks,
            &plan.indices,
            &plan.scratch,
            inputs.attention,
            &rows,
            sparse_scale(cfg),
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compressed (ratio-4 and ratio-128) layers
// ---------------------------------------------------------------------------

/// One row's host plan; mirrors `joint_decode_core::plan_compressed_row`.
struct CompressedRowPlan {
    position: usize,
    emit: bool,
    compressed_write_start: usize,
    indexer_write_start: usize,
    physical_kv_len: usize,
    attention_width: usize,
    index_offset: usize,
    /// `(valid keys, index bias, output column)` when the row selects.
    selection: Option<(usize, i32, usize)>,
}

/// Batched compressor buffers, row `r` at row `r`.
struct RowsCompressorArena {
    kv: MlxBuffer,
    score: MlxBuffer,
    output: MlxBuffer,
    rope: MlxBuffer,
}

impl RowsCompressorArena {
    fn new(device: &MlxDevice, rows: usize, projected: usize, dim: usize) -> Result<Self> {
        Ok(Self {
            kv: alloc(
                device,
                DType::F32,
                vec![rows, projected],
                "joint rows compressor KV",
            )?,
            score: alloc(
                device,
                DType::F32,
                vec![rows, projected],
                "joint rows compressor score",
            )?,
            output: alloc(
                device,
                DType::BF16,
                vec![rows, 1, dim],
                "joint rows compressor output",
            )?,
            rope: alloc(
                device,
                DType::BF16,
                vec![1, rows, 1, dim],
                "joint rows compressor rotated output",
            )?,
        })
    }
}

struct RowsIndexerArena {
    query_f32: MlxBuffer,
    query_bf16: MlxBuffer,
    weights: MlxBuffer,
    scaled_weights: MlxBuffer,
    compressor: RowsCompressorArena,
    scores: MlxBuffer,
    topk: MlxBuffer,
}

struct RowsFlashArena {
    rows: Vec<usize>,
    gathered_kv: MlxBuffer,
    mask: MlxBuffer,
    invalid_global: MlxBuffer,
    invalid_heads: MlxBuffer,
    q_scratch: MlxBuffer,
    output_scratch: MlxBuffer,
}

struct RowsLogitsArena {
    rows: Vec<usize>,
    scratch: MlxBuffer,
}

/// Host-prepared state of one compressed layer for every joint row.
pub(super) struct JointCompressedLayer {
    ratio: u32,
    rows: Vec<CompressedRowPlan>,
    indices: MlxBuffer,
    compressed_positions: MlxBuffer,
    main: RowsCompressorArena,
    indexer: Option<RowsIndexerArena>,
    flash: Option<RowsFlashArena>,
    logits: Option<RowsLogitsArena>,
}

pub(super) fn plan_compressed_layer(
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    layer_caches: &[&LayerCache],
    grafts: &[usize],
    ratio: u32,
    positions: &[usize],
    steps: &[&LayerCacheStep],
) -> Result<JointCompressedLayer> {
    ensure!(
        matches!(ratio, 4 | 128),
        "DeepSeek-V4 batched compressed cores need ratio 4 or 128, got {ratio}"
    );
    ensure!(
        batched_cores_admit(ratio, positions)
            && layer_caches.len() == positions.len()
            && grafts.len() == positions.len()
            && steps.len() == positions.len(),
        "DeepSeek-V4 batched compressed cores need 1..={DEEPSEEK_ROWS_MAX} aligned rows at nonzero positions"
    );
    let b = positions.len();
    let head_dim = cfg.head_dim as usize;
    let mut rows = Vec::with_capacity(b);
    let mut packed = Vec::new();
    let mut compressed_positions = alloc_host_input(
        device,
        DType::U32,
        vec![b],
        "joint rows compressed block positions",
    )?;
    let block_positions = compressed_positions.as_logical_mut_slice::<u32>()?;
    block_positions.fill(0);
    for (row, ((layer_cache, &position), step)) in
        layer_caches.iter().zip(positions).zip(steps).enumerate()
    {
        let graft = grafts[row];
        let window_capacity = layer_cache.window_kv.shape()[0];
        let compressed_capacity = layer_cache
            .compressed_kv
            .as_ref()
            .context("compressed KV cache is missing")?
            .shape()[0];
        ensure!(
            window_capacity == cfg.sliding_window as usize
                && layer_cache.attention_kv.shape()
                    == [window_capacity + compressed_capacity, head_dim],
            "DeepSeek-V4 joint row {row} compressed attention cache shape drift"
        );
        let emit = step.compressed_write_slot.is_some();
        let layout = prepare_layout(
            ratio,
            1,
            position,
            graft,
            window_capacity,
            step.compressed_valid_after,
            cfg.index_top_k as usize,
            false,
        )?;
        let selection = if ratio == 4 {
            let bias = i32::try_from(
                layout
                    .compressed_attention_offset
                    .checked_add(graft)
                    .context("DeepSeek-V4 indexer offset overflow")?,
            )
            .context("DeepSeek-V4 indexer attention offset exceeds i32")?;
            let column = layout
                .indexer_output_offset
                .context("ratio-four output offset missing")?;
            ensure!(
                column + cfg.index_top_k as usize <= layout.storage_stride,
                "DeepSeek-V4 invalid indexer output layout"
            );
            (step.compressed_valid_after > 0).then_some((step.compressed_valid_after, bias, column))
        } else {
            None
        };
        if emit {
            block_positions[row] = u32::try_from(position + 1 - ratio as usize)
                .context("DeepSeek-V4 compressed block position exceeds u32")?;
        }
        rows.push(CompressedRowPlan {
            position,
            emit,
            compressed_write_start: step.compressed_write_slot.unwrap_or(0) + graft,
            indexer_write_start: step.indexer_write_slot.unwrap_or(0),
            physical_kv_len: window_capacity + compressed_capacity,
            attention_width: layout.attention_width,
            index_offset: packed.len(),
            selection,
        });
        packed.extend_from_slice(&layout.index_storage);
    }
    let coefficient = usize::from(ratio == 4) + 1;
    let main = RowsCompressorArena::new(device, b, coefficient * head_dim, head_dim)?;
    let indexer = (ratio == 4)
        .then(|| -> Result<RowsIndexerArena> {
            let heads = cfg.index_num_heads as usize;
            let dim = cfg.index_head_dim as usize;
            let selecting: Vec<usize> = rows
                .iter()
                .filter_map(|row| row.selection.map(|(valid, _, _)| valid))
                .collect();
            let max_keys = selecting.iter().copied().max().unwrap_or(1);
            let (score_elements, topk_elements) =
                deepseek_indexer_rows_workspace_elements(selecting.len().max(1), max_keys);
            Ok(RowsIndexerArena {
                query_f32: alloc(
                    device,
                    DType::F32,
                    vec![1, b, heads, dim],
                    "joint rows index query f32",
                )?,
                query_bf16: alloc(
                    device,
                    DType::BF16,
                    vec![1, b, heads, dim],
                    "joint rows index query bf16",
                )?,
                weights: alloc(
                    device,
                    DType::F32,
                    vec![1, b, heads],
                    "joint rows index weights",
                )?,
                scaled_weights: alloc(
                    device,
                    DType::F32,
                    vec![1, b, heads],
                    "joint rows scaled index weights",
                )?,
                compressor: RowsCompressorArena::new(device, b, 2 * dim, dim)?,
                scores: alloc(
                    device,
                    DType::F32,
                    vec![score_elements],
                    "joint rows index scores",
                )?,
                topk: alloc(
                    device,
                    DType::I32,
                    vec![topk_elements],
                    "joint rows index top-k",
                )?,
            })
        })
        .transpose()?;
    let (flash_rows, logits_rows): (Vec<usize>, Vec<usize>) =
        (0..b).partition(|&row| ratio == 4 && rows[row].attention_width >= SPARSE_FLASH_MIN_TOP_K);
    let flash = (!flash_rows.is_empty())
        .then(|| -> Result<RowsFlashArena> {
            let n = flash_rows.len();
            let total_width: usize = flash_rows
                .iter()
                .map(|&row| rows[row].attention_width)
                .sum();
            let row_elements = DEEPSEEK_SPARSE_HEADS * head_dim;
            Ok(RowsFlashArena {
                gathered_kv: alloc(
                    device,
                    DType::BF16,
                    vec![total_width * head_dim],
                    "joint rows gathered flash KV",
                )?,
                mask: alloc(
                    device,
                    DType::BF16,
                    vec![total_width],
                    "joint rows flash mask",
                )?,
                invalid_global: host_zero_u32(device, n, "joint rows flash global validity")?,
                invalid_heads: host_zero_u32(
                    device,
                    n * DEEPSEEK_SPARSE_HEADS,
                    "joint rows flash head validity",
                )?,
                q_scratch: alloc(
                    device,
                    DType::BF16,
                    vec![n * row_elements],
                    "joint rows flash q",
                )?,
                output_scratch: alloc(
                    device,
                    DType::BF16,
                    vec![n * row_elements],
                    "joint rows flash output",
                )?,
                rows: flash_rows,
            })
        })
        .transpose()?;
    let logits = (!logits_rows.is_empty())
        .then(|| -> Result<RowsLogitsArena> {
            let max_width = logits_rows
                .iter()
                .map(|&row| rows[row].attention_width)
                .max()
                .unwrap_or(1);
            let elements =
                deepseek_sparse_attention_rows_scratch_elements(logits_rows.len(), max_width);
            Ok(RowsLogitsArena {
                scratch: alloc(
                    device,
                    DType::F32,
                    vec![elements],
                    "joint rows sparse logits",
                )?,
                rows: logits_rows,
            })
        })
        .transpose()?;
    Ok(JointCompressedLayer {
        ratio,
        rows,
        indices: host_i32(device, &packed, "joint rows compressed attention indices")?,
        compressed_positions,
        main,
        indexer,
        flash,
        logits,
    })
}

/// Projections + one compressor rows dispatch + the emitting rows' tail.
#[allow(clippy::too_many_arguments)]
fn encode_rows_compressor(
    session: &mut GraphSession<'_>,
    registry: &mut KernelRegistry,
    device: &MlxDevice,
    cfg: &Deepseek4Config,
    plan: &JointCompressedLayer,
    ratio: u32,
    dim: usize,
    attn_norm: &MlxBuffer,
    weights: &CompressorWeightsRef<'_>,
    arena: &RowsCompressorArena,
    states: &[(&MlxBuffer, &MlxBuffer)],
    caches: &[&MlxBuffer],
    indexer: bool,
    frequencies: &MlxBuffer,
) -> Result<()> {
    let b = plan.rows.len();
    let hidden = cfg.hidden_size as usize;
    let projected = (usize::from(ratio == 4) + 1) * dim;
    let (kv_label, gate_label) = if indexer {
        ("index compressor KV", "index compressor gate")
    } else {
        ("main compressor KV", "main compressor gate")
    };
    project_rows(
        session,
        registry,
        device,
        attn_norm,
        &weights.kv,
        &arena.kv,
        b,
        projected,
        hidden,
        kv_label,
    )?;
    project_rows(
        session,
        registry,
        device,
        attn_norm,
        &weights.gate,
        &arena.score,
        b,
        projected,
        hidden,
        gate_label,
    )?;
    let mut reads: Vec<&MlxBuffer> = vec![&arena.kv, &arena.score, weights.ape, weights.norm];
    let mut writes: Vec<&MlxBuffer> = vec![&arena.output];
    for (kv_state, score_state) in states {
        reads.extend([*kv_state, *score_state]);
        writes.extend([*kv_state, *score_state]);
    }
    session.barrier_between(&reads, &writes);
    let rows: Vec<DeepSeekCompressorRow<'_>> = states
        .iter()
        .zip(&plan.rows)
        .map(|(&(kv_state, score_state), row)| {
            Ok(DeepSeekCompressorRow {
                kv_state,
                score_state,
                start_pos: u32::try_from(row.position)
                    .context("DeepSeek-V4 compressor position exceeds u32")?,
            })
        })
        .collect::<Result<_>>()?;
    dispatch_deepseek_compressor_rows(
        session.encoder_mut(),
        registry,
        device,
        &arena.kv,
        &arena.score,
        weights.ape,
        weights.norm,
        &arena.output,
        &rows,
        ratio,
        dim as u32,
        cfg.rms_norm_eps,
    )?;
    if !plan.rows.iter().any(|row| row.emit) {
        return Ok(());
    }
    // Row-independent RoPE (and quantization / Hadamard) over every row; only
    // emitting rows' results are copied into their caches.
    let rope_input = arena.output.with_shape(vec![1, b, 1, dim])?;
    session.barrier_between(
        &[&rope_input, &plan.compressed_positions, frequencies],
        &[&arena.rope],
    );
    dispatch_deepseek_tail_rope_bf16(
        session.encoder_mut(),
        registry,
        device,
        &rope_input,
        &plan.compressed_positions,
        frequencies,
        &arena.rope,
        &DeepSeekTailRopeParams {
            batch: 1,
            seq_len: b as u32,
            heads: 1,
            head_dim: dim as u32,
            rope_dim: cfg.rope_head_dim,
            inverse: 0,
        },
    )?;
    session.barrier_between(&[&arena.rope], &[&arena.rope]);
    if indexer {
        dispatch_deepseek_hadamard_mxfp4_bf16(
            session.encoder_mut(),
            registry,
            device,
            &arena.rope,
            b as u32,
        )?;
    } else {
        dispatch_deepseek_mxfp8_fake_quant_bf16(
            session.encoder_mut(),
            registry,
            device,
            &arena.rope,
            &DeepSeekMxfp8Params {
                rows: b as u32,
                row_width: dim as u32,
                quantized_width: (dim - cfg.rope_head_dim as usize) as u32,
                block_size: 64,
            },
        )?;
    }
    for (row, (plan_row, cache)) in plan.rows.iter().zip(caches).enumerate() {
        if !plan_row.emit {
            continue;
        }
        let source = row_view(&arena.rope, row, vec![1, 1, 1, dim])?;
        let write_start = if indexer {
            plan_row.indexer_write_start
        } else {
            plan_row.compressed_write_start
        };
        session.barrier_between(&[&source], &[*cache]);
        dispatch_kv_cache_copy(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &source,
            cache,
            write_start as u32,
            dim as u32,
            1,
            cache.shape()[0] as u32,
            false,
        )?;
    }
    Ok(())
}

impl JointCompressedLayer {
    /// Row `row`'s `(attention width, physical KV length, flash path)`.
    #[cfg(test)]
    pub(super) fn row_shape(&self, row: usize) -> (usize, usize, bool) {
        let plan = &self.rows[row];
        let flash = self
            .flash
            .as_ref()
            .is_some_and(|flash| flash.rows.contains(&row));
        (plan.attention_width, plan.physical_kv_len, flash)
    }

    /// Row `row`'s attention index storage (window, graft and indexer slots).
    #[cfg(test)]
    pub(super) fn row_index_storage(&self, row: usize) -> Vec<i32> {
        let packed = self
            .indices
            .as_logical_slice::<i32>()
            .expect("packed indices");
        let end = self
            .rows
            .get(row + 1)
            .map_or(packed.len(), |next| next.index_offset);
        packed[self.rows[row].index_offset..end].to_vec()
    }

    /// Every row's window write, main compressor, indexer and sparse
    /// attention, phase by phase.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        session: &mut GraphSession<'_>,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        cfg: &Deepseek4Config,
        layer_caches: &[&LayerCache],
        steps: &[&LayerCacheStep],
        weights: &CompressedAttentionWeightsRef<'_>,
        frequencies: &MlxBuffer,
        inputs: &JointBatchInputs<'_>,
    ) -> Result<()> {
        encode_window_writes(
            session,
            registry,
            device,
            cfg,
            layer_caches,
            steps,
            inputs.kv_rope,
        )?;
        self.encode_main_compressor(
            session,
            registry,
            device,
            cfg,
            layer_caches,
            &weights.compressor,
            frequencies,
            inputs.attn_norm,
        )?;
        if let Some(indexer_weights) = weights.indexer.as_ref() {
            self.encode_indexer(
                session,
                registry,
                device,
                cfg,
                layer_caches,
                indexer_weights,
                frequencies,
                inputs,
            )?;
        }
        self.encode_attention(
            session,
            registry,
            device,
            cfg,
            layer_caches,
            weights.attention.sinks,
            inputs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_main_compressor(
        &self,
        session: &mut GraphSession<'_>,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        cfg: &Deepseek4Config,
        layer_caches: &[&LayerCache],
        weights: &CompressorWeightsRef<'_>,
        frequencies: &MlxBuffer,
        attn_norm: &MlxBuffer,
    ) -> Result<()> {
        let mut states = Vec::with_capacity(layer_caches.len());
        let mut caches = Vec::with_capacity(layer_caches.len());
        for layer_cache in layer_caches {
            states.push((
                layer_cache
                    .main_kv_state
                    .as_ref()
                    .context("main compressor KV state is missing")?,
                layer_cache
                    .main_score_state
                    .as_ref()
                    .context("main compressor score state is missing")?,
            ));
            caches.push(
                layer_cache
                    .compressed_kv
                    .as_ref()
                    .context("compressed attention cache is missing")?,
            );
        }
        encode_rows_compressor(
            session,
            registry,
            device,
            cfg,
            self,
            self.ratio,
            cfg.head_dim as usize,
            attn_norm,
            weights,
            &self.main,
            &states,
            &caches,
            false,
            frequencies,
        )
    }

    /// The ratio-4 indexer: projections, query RoPE / Hadamard / weight
    /// scale, the indexer compressor, then score + top-k into the rows'
    /// attention indices.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_indexer(
        &self,
        session: &mut GraphSession<'_>,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        cfg: &Deepseek4Config,
        layer_caches: &[&LayerCache],
        weights: &IndexerWeightsRef<'_>,
        frequencies: &MlxBuffer,
        inputs: &JointBatchInputs<'_>,
    ) -> Result<()> {
        let arena = self
            .indexer
            .as_ref()
            .context("DeepSeek-V4 batched indexer arena missing")?;
        let b = self.rows.len();
        let hidden = cfg.hidden_size as usize;
        let q_rank = cfg.q_lora_rank as usize;
        let heads = cfg.index_num_heads as usize;
        let dim = cfg.index_head_dim as usize;
        project_rows(
            session,
            registry,
            device,
            inputs.q_a_norm,
            &weights.q_b,
            &arena.query_f32,
            b,
            heads * dim,
            q_rank,
            "index query",
        )?;
        project_rows(
            session,
            registry,
            device,
            inputs.attn_norm,
            &weights.projection,
            &arena.weights,
            b,
            heads,
            hidden,
            "index head weights",
        )?;
        let mut states = Vec::with_capacity(b);
        let mut caches = Vec::with_capacity(b);
        for layer_cache in layer_caches {
            states.push((
                layer_cache
                    .indexer_kv_state
                    .as_ref()
                    .context("ratio-four indexer KV state is missing")?,
                layer_cache
                    .indexer_score_state
                    .as_ref()
                    .context("ratio-four indexer score state is missing")?,
            ));
            caches.push(
                layer_cache
                    .indexer_kv
                    .as_ref()
                    .context("ratio-four indexer cache is missing")?,
            );
        }
        prefill_annotations::indexer_query_rope(
            session,
            registry,
            device,
            &arena.query_f32,
            inputs.positions,
            frequencies,
            &arena.query_bf16,
            &DeepSeekTailRopeParams {
                batch: 1,
                seq_len: b as u32,
                heads: heads as u32,
                head_dim: dim as u32,
                rope_dim: cfg.rope_head_dim,
                inverse: 0,
            },
        )?;
        session.barrier_between(&[&arena.query_bf16], &[&arena.query_bf16]);
        dispatch_deepseek_hadamard_mxfp4_bf16(
            session.encoder_mut(),
            registry,
            device,
            &arena.query_bf16,
            (b * heads) as u32,
        )?;
        session.barrier_between(&[&arena.weights], &[&arena.scaled_weights]);
        scalar_mul_f32(
            session.encoder_mut(),
            registry,
            device.metal_device(),
            &arena.weights,
            &arena.scaled_weights,
            b * heads,
            1.0 / ((dim * heads) as f32).sqrt(),
        )?;
        encode_rows_compressor(
            session,
            registry,
            device,
            cfg,
            self,
            4,
            dim,
            inputs.attn_norm,
            &weights.compressor,
            &arena.compressor,
            &states,
            &caches,
            true,
            frequencies,
        )?;
        self.encode_selection(session, registry, device, layer_caches)
    }

    /// Indexer score + top-k for every selecting row (the one-row
    /// `encode_selection`), writing each row's attention index storage.
    pub(super) fn encode_selection(
        &self,
        session: &mut GraphSession<'_>,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        layer_caches: &[&LayerCache],
    ) -> Result<()> {
        let arena = self
            .indexer
            .as_ref()
            .context("DeepSeek-V4 batched indexer arena missing")?;
        let mut views = Vec::new();
        let mut meta = Vec::new();
        for (row, (plan, layer_cache)) in self.rows.iter().zip(layer_caches).enumerate() {
            let Some((valid, bias, column)) = plan.selection else {
                continue;
            };
            let cache = layer_cache
                .indexer_kv
                .as_ref()
                .context("ratio-four indexer cache is missing")?;
            let elements = valid
                .checked_mul(128)
                .context("DeepSeek-V4 indexer cache view overflow")?;
            ensure!(
                cache.data_byte_len() >= elements * 2,
                "DeepSeek-V4 indexer valid cache exceeds backing"
            );
            views.push(cache.slice_view(0, elements));
            meta.push((row, valid, bias, plan.index_offset + column));
        }
        if views.is_empty() {
            return Ok(());
        }
        let rows: Vec<DeepSeekIndexerRow<'_>> = views
            .iter()
            .zip(&meta)
            .map(|(kv, &(row, valid, bias, output_offset))| {
                Ok(DeepSeekIndexerRow {
                    kv,
                    kv_len: valid,
                    start_pos: u32::try_from(self.rows[row].position)
                        .context("DeepSeek-V4 indexer position exceeds u32")?,
                    offset: bias,
                    q_row: row,
                    output_offset,
                })
            })
            .collect::<Result<_>>()?;
        let mut reads: Vec<&MlxBuffer> = vec![&arena.query_bf16, &arena.scaled_weights];
        reads.extend(views.iter());
        session.barrier_between(&reads, &[&arena.scores, &arena.topk, &self.indices]);
        dispatch_deepseek_indexer_rows(
            session.encoder_mut(),
            registry,
            device,
            &arena.query_bf16,
            &arena.scaled_weights,
            &arena.scores,
            &arena.topk,
            &self.indices,
            &rows,
        )?;
        Ok(())
    }

    /// Sparse attention for every row: the flash rows path for ratio-4 rows
    /// at width >= `SPARSE_FLASH_MIN_TOP_K`, score + reduce for the others.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_attention(
        &self,
        session: &mut GraphSession<'_>,
        registry: &mut KernelRegistry,
        device: &MlxDevice,
        cfg: &Deepseek4Config,
        layer_caches: &[&LayerCache],
        sinks: &MlxBuffer,
        inputs: &JointBatchInputs<'_>,
    ) -> Result<()> {
        let head_dim = cfg.head_dim as usize;
        let views = layer_caches
            .iter()
            .zip(&self.rows)
            .map(|(layer_cache, row)| {
                layer_cache
                    .attention_kv
                    .with_shape(vec![row.physical_kv_len, head_dim])
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        let spec = |row: usize| DeepSeekSparseAttentionRow {
            kv: &views[row],
            kv_len: self.rows[row].physical_kv_len,
            width: self.rows[row].attention_width,
            index_offset: self.rows[row].index_offset,
            q_row: row,
        };
        let mut reads: Vec<&MlxBuffer> = vec![inputs.q_rope, sinks, &self.indices];
        reads.extend(layer_caches.iter().map(|cache| &cache.attention_kv));
        if let Some(flash) = &self.flash {
            let rows: Vec<_> = flash.rows.iter().map(|&row| spec(row)).collect();
            let mut writes: Vec<&MlxBuffer> = vec![
                &flash.gathered_kv,
                &flash.mask,
                &flash.invalid_global,
                &flash.invalid_heads,
                inputs.attention,
            ];
            if !deepseek_sparse_flash_rows_contiguous(&rows) {
                writes.extend([&flash.q_scratch, &flash.output_scratch]);
            }
            session.barrier_between(&reads, &writes);
            dispatch_deepseek_sparse_attention_flash_rows(
                session.encoder_mut(),
                registry,
                device,
                inputs.q_rope,
                sinks,
                &self.indices,
                &DeepSeekSparseFlashRowsWorkspace {
                    gathered_kv: &flash.gathered_kv,
                    mask: &flash.mask,
                    invalid_global: &flash.invalid_global,
                    invalid_heads: &flash.invalid_heads,
                    q_scratch: &flash.q_scratch,
                    output_scratch: &flash.output_scratch,
                },
                inputs.attention,
                &rows,
                sparse_scale(cfg),
            )?;
        }
        if let Some(logits) = &self.logits {
            let rows: Vec<_> = logits.rows.iter().map(|&row| spec(row)).collect();
            session.barrier_between(&reads, &[inputs.attention, &logits.scratch]);
            dispatch_deepseek_sparse_attention_rows(
                session.encoder_mut(),
                registry,
                device,
                inputs.q_rope,
                sinks,
                &self.indices,
                &logits.scratch,
                inputs.attention,
                &rows,
                sparse_scale(cfg),
            )?;
        }
        Ok(())
    }
}
