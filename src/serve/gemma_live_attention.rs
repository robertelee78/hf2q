//! Gemma 4 live-resume (warm suffix) attention: the one per-sequence function
//! used by a single-slot warm prefill and by every lane of a resumed
//! multi-sequence prefill (ADR-063 Stage 2).
//!
//! A live resume appends `rows` suffix rows to a slot that already holds
//! `start` valid tokens. Row-parallel work (projections, norms, FFN/MoE, head)
//! runs once over the concatenated suffix stream in
//! `forward_prefill_batched`; this module runs attention for each sequence
//! against that sequence's own KV view, addressed as slot 0:
//!
//! * global layers append the whole suffix first (append-only storage), then
//!   attend `[0, graft + start + chunk_end)`;
//! * sliding layers stage the ring's chronological history plus the chunk's
//!   own rows, attend, and only then commit the chunk into the ring, so later
//!   suffix rows never overwrite history an earlier query reads;
//! * TQ-HB V is quantized exactly as decode stores it, and the FWHT sign undo
//!   runs once over the layer's output rows.
//!
//! Two kernels implement the attention step:
//!
//! * [`GemmaLiveKernel::TiledF16`]: compact K to dense F16, dequantize (or
//!   compact) the active V range to F16, then tiled flash attention: d256
//!   with an SWA mask and tile-skip `blk` for sliding layers (1024-query
//!   chunks), `flash_attn_prefill_f16_d512_resume` for global layers
//!   (4096-query chunks). One `batch = 1` dispatch per (sequence, chunk):
//!   no block-diagonal mask is involved.
//! * [`GemmaLiveKernel::VecHybrid`]: `flash_attn_vec_hybrid_batched`, one
//!   decode-style threadgroup per (query, head), in 256-query chunks split
//!   at the 512/1024 cache-length buckets.
//!
//! Everything that reaches a kernel is a function of the sequence alone
//! (its rows, start, KV view and Q/K/V rows), so a lane of a joint cohort is
//! bitwise equal to the same sequence prefilled alone whenever both resolve
//! the same kernel. [`LiveRoute`] guarantees that: a single-slot resume and a
//! [`GemmaLiveAttention::PerSlotExact`] cohort resolve the identical route.
use anyhow::Result;
use mlx_native::{DType, GraphSession, KernelRegistry, MlxBuffer, MlxDevice};

use crate::inference::models::gemma4::{GemmaLiveAttention, GemmaLiveKernel, HybridKvBuffers};
use crate::serve::forward_prefill::GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS;

/// Vec-hybrid query chunk: bounds the attention scratch while keeping every
/// dispatch in a single hybrid-kernel scheduling bucket.
pub(crate) const VEC_QUERY_CHUNK: usize = 256;
/// Tiled sliding-layer query chunk: one sliding window of queries, so the
/// ring stage (`cap - 1` history rows plus the chunk) holds every key any
/// chunk query can see.
pub(crate) const TILED_SLIDING_QUERY_CHUNK: usize = 1024;
/// Tiled global-layer query chunk. It equals the Gemma 4 transaction row
/// cap: slot-aware warm prefills and installed lanes never exceed it, and
/// SerialFifo rejects Gemma prompts above it
/// (`validate_serial_prefill_family`), so in serving every sequence takes one
/// global chunk. The split only guards direct callers of the live resume
/// against unbounded F16 Q/O scratch.
pub(crate) const TILED_GLOBAL_QUERY_CHUNK: usize = 4096;
/// Warm suffixes shorter than this keep the vec-hybrid kernel under
/// [`GemmaLiveKernel::TiledF16`]: below the 32-token batched-prefill
/// containment boundary (ADR-040, 2026-08-14 tiny-prefill fault) the tiled
/// route has no serving evidence, and the joint cohort never admits such a
/// lane.
pub(crate) const WARM_TILED_MIN_ROWS: usize = GEMMA_SLOT_BATCHED_PREFILL_MIN_TOKENS;

pub(crate) const WARM_TILED_ATTENTION_ENV: &str = "HF2Q_GEMMA_WARM_TILED_ATTENTION";
/// On by default. Revert rule: if the root's real-artifact gates or the
/// paired quality campaign fail, set this to `false` before merging.
const WARM_TILED_ATTENTION_DEFAULT: bool = true;

fn parse_warm_tiled_switch(raw: Option<&str>) -> std::result::Result<bool, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(WARM_TILED_ATTENTION_DEFAULT);
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(true),
        "0" | "off" | "false" => Ok(false),
        _ => Err(format!(
            "{WARM_TILED_ATTENTION_ENV} must be 1/on/true or 0/off/false, got {raw:?}"
        )),
    }
}

fn warm_kernel_for_switch(raw: Option<&str>) -> GemmaLiveKernel {
    let tiled = parse_warm_tiled_switch(raw).unwrap_or_else(|error| {
        tracing::warn!(
            error,
            fallback = WARM_TILED_ATTENTION_DEFAULT,
            "invalid Gemma warm tiled attention switch"
        );
        WARM_TILED_ATTENTION_DEFAULT
    });
    if tiled {
        GemmaLiveKernel::TiledF16
    } else {
        GemmaLiveKernel::VecHybrid
    }
}

/// Kernel for warm live-resume attention (single-slot resumes and joint warm
/// cohorts), resolved once per model load from
/// `HF2Q_GEMMA_WARM_TILED_ATTENTION`.
pub(crate) fn warm_live_kernel_from_env() -> GemmaLiveKernel {
    let raw = std::env::var(WARM_TILED_ATTENTION_ENV).ok();
    let kernel = warm_kernel_for_switch(raw.as_deref());
    tracing::info!(?kernel, "Gemma warm live attention kernel");
    kernel
}

/// `HF2Q_GEMMA_TILED_LIVE`: installed long-prompt multi-sequence lanes and
/// the stable cue batch ([`GemmaLiveAttention::Tiled`]) only.
fn tiled_live_enabled() -> bool {
    !matches!(
        std::env::var("HF2Q_GEMMA_TILED_LIVE").as_deref(),
        Ok("0") | Ok("false") | Ok("off")
    )
}

/// Attention route of one live-resume forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LiveRoute {
    kernel: GemmaLiveKernel,
    /// Under `TiledF16`, sequences with fewer rows keep `VecHybrid`.
    min_tiled_rows: usize,
}

impl LiveRoute {
    /// `attention` is the multi-sequence descriptor's route, `None` for a
    /// single-slot resume.
    pub(crate) fn resolve(
        attention: Option<GemmaLiveAttention>,
        warm_kernel: GemmaLiveKernel,
    ) -> Self {
        Self::select(attention, warm_kernel, tiled_live_enabled())
    }

    fn select(
        attention: Option<GemmaLiveAttention>,
        warm_kernel: GemmaLiveKernel,
        tiled_live: bool,
    ) -> Self {
        match attention {
            Some(GemmaLiveAttention::Tiled) => Self {
                kernel: if tiled_live {
                    GemmaLiveKernel::TiledF16
                } else {
                    GemmaLiveKernel::VecHybrid
                },
                min_tiled_rows: 0,
            },
            None | Some(GemmaLiveAttention::PerSlotExact) => Self {
                kernel: warm_kernel,
                min_tiled_rows: WARM_TILED_MIN_ROWS,
            },
        }
    }

    pub(crate) fn seq_kernel(self, rows: usize) -> GemmaLiveKernel {
        match self.kernel {
            GemmaLiveKernel::TiledF16 if rows >= self.min_tiled_rows => GemmaLiveKernel::TiledF16,
            _ => GemmaLiveKernel::VecHybrid,
        }
    }

    pub(crate) fn uses_vec(self, spans: &[LiveSpan]) -> bool {
        spans
            .iter()
            .any(|span| self.seq_kernel(span.rows) == GemmaLiveKernel::VecHybrid)
    }

    /// Prefill banner fragment.
    pub(crate) fn label(self) -> &'static str {
        match (self.kernel, self.min_tiled_rows) {
            (GemmaLiveKernel::VecHybrid, _) => "vec-hybrid",
            (GemmaLiveKernel::TiledF16, 0) => "tiled-f16",
            (GemmaLiveKernel::TiledF16, _) => "tiled-f16 (vec-hybrid below 32 rows)",
        }
    }
}

/// Rows of one sequence inside the concatenated suffix stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LiveSpan {
    /// First row in the concatenated stream.
    pub(crate) offset: usize,
    pub(crate) rows: usize,
    /// Prompt-indexed position of the first suffix row (cached tokens).
    pub(crate) start: usize,
}

/// One sequence's KV view of the current layer plus its stream rows.
pub(crate) struct LiveSeq<'a> {
    pub(crate) kv: &'a HybridKvBuffers,
    pub(crate) span: LiveSpan,
}

fn vec_query_chunk_len(start_pos: usize, remaining: usize) -> usize {
    let mut len = remaining.min(VEC_QUERY_CHUNK);
    for boundary in [512usize, 1024usize] {
        if start_pos < boundary && start_pos.saturating_add(len) > boundary {
            len = boundary - start_pos;
            break;
        }
    }
    len.max(1)
}

fn chunk_len(kernel: GemmaLiveKernel, is_ring: bool, start_pos: usize, remaining: usize) -> usize {
    match kernel {
        GemmaLiveKernel::VecHybrid => vec_query_chunk_len(start_pos, remaining),
        GemmaLiveKernel::TiledF16 if is_ring => remaining.min(TILED_SLIDING_QUERY_CHUNK),
        GemmaLiveKernel::TiledF16 => remaining.min(TILED_GLOBAL_QUERY_CHUNK),
    }
}

/// `(local_start, len)` query chunks of a sequence that starts at absolute
/// prompt position `start`.
fn chunks(
    kernel: GemmaLiveKernel,
    is_ring: bool,
    start: usize,
    rows: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut local = 0usize;
    std::iter::from_fn(move || {
        if local >= rows {
            return None;
        }
        let len = chunk_len(kernel, is_ring, start + local, rows - local);
        let chunk = (local, len);
        local += len;
        Some(chunk)
    })
}

/// Query rows a sliding-layer stage holds after its `cap - 1` history rows.
fn stage_query_rows(kernel: GemmaLiveKernel, rows: usize) -> usize {
    rows.min(match kernel {
        GemmaLiveKernel::VecHybrid => VEC_QUERY_CHUNK,
        GemmaLiveKernel::TiledF16 => TILED_SLIDING_QUERY_CHUNK,
    })
}

/// Logical query offset inside a chronological sliding-cache stage. The
/// stage copies up to `cache_capacity - 1` history rows before the chunk;
/// capping at the model sliding-window size instead would point long
/// linear-resume queries at the wrong staged rows.
fn stage_history_len(absolute_start: usize, cache_capacity: usize) -> usize {
    absolute_start.min(cache_capacity.saturating_sub(1))
}

/// Keys a chunk attends and the absolute key index of its first query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ChunkKeys {
    kv_seq_len: usize,
    query_offset: usize,
}

fn chunk_keys(
    is_ring: bool,
    cache_capacity: usize,
    graft_len: usize,
    absolute_start: usize,
    len: usize,
) -> ChunkKeys {
    // Sliding rings are prompt-indexed and never hold graft rows; global
    // rows are absolute, graft rows first (ADR-059).
    let query_offset = if is_ring {
        stage_history_len(absolute_start, cache_capacity)
    } else {
        graft_len + absolute_start
    };
    ChunkKeys {
        kv_seq_len: query_offset + len,
        query_offset,
    }
}

/// Fill the vec-hybrid sliding-stage query positions of `span`.
fn fill_vec_stage_positions(values: &mut [u32], span: LiveSpan, cache_capacity: usize) {
    for (local_start, len) in chunks(GemmaLiveKernel::VecHybrid, true, span.start, span.rows) {
        let history = stage_history_len(span.start + local_start, cache_capacity);
        let base = span.offset + local_start;
        for (local, value) in values[base..base + len].iter_mut().enumerate() {
            *value = (history + local) as u32;
        }
    }
}

/// Per-forward inputs of the vec-hybrid kernel: per-row slot ids (always
/// view slot 0), per-row sliding-stage positions and the split-K scratch.
pub(crate) struct LiveVecScratch {
    slot_ids: MlxBuffer,
    stage_positions: MlxBuffer,
    tmp: MlxBuffer,
}

impl LiveVecScratch {
    /// `None` when no sequence of this forward takes the vec-hybrid kernel.
    pub(crate) fn alloc(
        dev: &MlxDevice,
        route: LiveRoute,
        spans: &[LiveSpan],
        sliding_capacity: usize,
        total_rows: usize,
        nh: usize,
        max_hd: usize,
    ) -> Result<Option<Self>> {
        if !route.uses_vec(spans) {
            return Ok(None);
        }
        let alloc_u32 = |name: &str| -> Result<MlxBuffer> {
            dev.alloc_buffer(total_rows * 4, DType::U32, vec![total_rows])
                .map_err(|e| anyhow::anyhow!("batched live {name}: {e}"))
        };
        let mut slot_ids = alloc_u32("slot ids")?;
        slot_ids
            .as_mut_slice::<u32>()
            .map_err(|e| anyhow::anyhow!("live slot ids write: {e}"))?
            .fill(0);
        let mut stage_positions = alloc_u32("stage positions")?;
        {
            let values = stage_positions
                .as_mut_slice::<u32>()
                .map_err(|e| anyhow::anyhow!("live stage positions write: {e}"))?;
            for &span in spans {
                fill_vec_stage_positions(values, span, sliding_capacity);
            }
        }
        let chunk_queries = total_rows.min(VEC_QUERY_CHUNK);
        let tmp_bytes = mlx_native::ops::flash_attn_vec_tq_hb::tmp_buffer_bytes(
            (chunk_queries * nh) as u32,
            max_hd as u32,
        );
        let tmp = dev
            .alloc_buffer(tmp_bytes, DType::F32, vec![tmp_bytes / 4])
            .map_err(|e| anyhow::anyhow!("batched live sdpa tmp: {e}"))?;
        Ok(Some(Self {
            slot_ids,
            stage_positions,
            tmp,
        }))
    }
}

/// Layer-wide inputs of the live attention.
pub(crate) struct LiveLayer<'a> {
    pub(crate) layer_idx: usize,
    pub(crate) nh: usize,
    pub(crate) nkv: usize,
    pub(crate) hd: usize,
    pub(crate) sliding_window: u32,
    pub(crate) graft_len: usize,
    pub(crate) tq_scale_factor_d512: f32,
    pub(crate) tq_codebook_bits: u32,
    /// Normed, RoPE'd Q of every stream row, `[rows, nh, hd]` F32.
    pub(crate) q: &'a MlxBuffer,
    /// Normed K / V of every stream row, `[rows, nkv, hd]` F32.
    pub(crate) k: &'a MlxBuffer,
    pub(crate) v: &'a MlxBuffer,
    /// Attention output, `[rows, nh, hd]` F32.
    pub(crate) out: &'a MlxBuffer,
    /// Absolute (graft-shifted) RoPE position of every stream row.
    pub(crate) positions: &'a MlxBuffer,
    pub(crate) vec: Option<&'a LiveVecScratch>,
    /// Rows of the whole stream (the FWHT undo extent).
    pub(crate) total_rows: usize,
}

/// Encode one layer's live attention for every sequence, then the FWHT sign
/// undo. A single-slot resume passes one sequence; a multi-sequence resume
/// passes each lane's own slot view.
pub(crate) fn encode_live_layer_attention<'a>(
    s: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    route: LiveRoute,
    seqs: impl IntoIterator<Item = LiveSeq<'a>>,
) -> Result<()> {
    let mut packed_v: Option<bool> = None;
    for seq in seqs {
        let seq_packed = seq.kv.v_packed.dtype() != DType::F16;
        anyhow::ensure!(
            *packed_v.get_or_insert(seq_packed) == seq_packed,
            "live attention L{}: sequences disagree on the V storage dtype",
            layer.layer_idx
        );
        encode_live_seq(s, reg, dev, layer, &seq, route.seq_kernel(seq.span.rows))?;
    }
    if packed_v == Some(true) {
        s.barrier_between(&[layer.out], &[layer.out]);
        mlx_native::ops::fwht_standalone::dispatch_fwht_sign_undo_f32(
            s.encoder_mut(),
            reg,
            dev.metal_device(),
            layer.out,
            (layer.total_rows * layer.nh) as u32,
            layer.hd as u32,
        )
        .map_err(|e| anyhow::anyhow!("live hybrid FWHT undo L{}: {e}", layer.layer_idx))?;
    }
    Ok(())
}

/// Chronological sliding stage: up to `cap - 1` history rows plus one chunk.
struct RingStage {
    k: MlxBuffer,
    v: MlxBuffer,
    norms: MlxBuffer,
    capacity: usize,
}

impl RingStage {
    fn alloc(
        dev: &MlxDevice,
        kv: &HybridKvBuffers,
        nkv: usize,
        hd: usize,
        query_rows: usize,
        layer_idx: usize,
    ) -> Result<Self> {
        let capacity = kv.capacity.saturating_sub(1) + query_rows;
        let norms_per_pos = kv.norms_per_pos;
        anyhow::ensure!(
            capacity > 0 && norms_per_pos > 0,
            "live ring stage L{layer_idx} has invalid shape"
        );
        let v_dtype = kv.v_packed.dtype();
        // SAFETY: every chunk initializes the history + suffix prefix it
        // reads before any attention dispatch; tail rows are never read.
        // F16-V attention never reads the norms.
        let k = unsafe {
            dev.alloc_buffer_for_overwrite(
                nkv * capacity * hd * DType::F16.size_of(),
                DType::F16,
                vec![nkv, capacity, hd],
            )
        }
        .map_err(|e| anyhow::anyhow!("live ring stage K L{layer_idx}: {e}"))?;
        let v = unsafe {
            dev.alloc_buffer_for_overwrite(
                nkv * capacity * hd * v_dtype.size_of(),
                v_dtype,
                vec![nkv, capacity, hd],
            )
        }
        .map_err(|e| anyhow::anyhow!("live ring stage V L{layer_idx}: {e}"))?;
        let norms = unsafe {
            dev.alloc_buffer_for_overwrite(
                nkv * capacity * norms_per_pos * DType::F32.size_of(),
                DType::F32,
                vec![nkv, capacity, norms_per_pos],
            )
        }
        .map_err(|e| anyhow::anyhow!("live ring stage norms L{layer_idx}: {e}"))?;
        Ok(Self {
            k,
            v,
            norms,
            capacity,
        })
    }
}

/// Write `len` stream rows starting at `src_row` into `dst` rows
/// `[dst_pos, dst_pos + len)` of a `[nkv, capacity, hd]` cache: K as F16, V
/// as F16 or TQ-HB (`ring` selects the ring-wrapping quantize write).
#[allow(clippy::too_many_arguments)]
fn write_kv_rows(
    s: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    (k, v, norms): (&MlxBuffer, &MlxBuffer, &MlxBuffer),
    capacity: usize,
    dst_pos: usize,
    len: usize,
    src_row: usize,
    ring: bool,
    what: &str,
) -> Result<()> {
    let metal_dev = dev.metal_device();
    let (nkv, hd, layer_idx) = (layer.nkv as u32, layer.hd as u32, layer.layer_idx);
    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
        s.encoder_mut(),
        reg,
        metal_dev,
        layer.k,
        k,
        nkv,
        hd,
        capacity as u32,
        dst_pos as u32,
        len as u32,
        src_row as u32,
    )
    .map_err(|e| anyhow::anyhow!("live {what} K L{layer_idx}: {e}"))?;
    if v.dtype() == DType::F16 {
        mlx_native::ops::kv_cache_copy::dispatch_kv_cache_copy_seq_f32_to_f16(
            s.encoder_mut(),
            reg,
            metal_dev,
            layer.v,
            v,
            nkv,
            hd,
            capacity as u32,
            dst_pos as u32,
            len as u32,
            src_row as u32,
        )
        .map_err(|e| anyhow::anyhow!("live {what} V L{layer_idx}: {e}"))?;
    } else {
        mlx_native::ops::hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq(
            s.encoder_mut(),
            reg,
            metal_dev,
            layer.v,
            v,
            norms,
            nkv,
            hd,
            capacity as u32,
            dst_pos as u32,
            len as u32,
            src_row as u32,
            ring,
            layer.tq_scale_factor_d512,
            layer.tq_codebook_bits,
        )
        .map_err(|e| anyhow::anyhow!("live {what} FWHT V L{layer_idx}: {e}"))?;
    }
    Ok(())
}

/// Copy the `history` rows ending at logical position `logical_end` of a
/// head-major `[nkv, src_capacity, row]` ring into rows `[0, history)` of a
/// head-major `[nkv, dst_capacity, row]` buffer, chronologically.
#[allow(clippy::too_many_arguments)]
fn linearize_rows(
    s: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    (src, dst): (&MlxBuffer, &MlxBuffer),
    nkv: usize,
    (src_capacity, dst_capacity): (usize, usize),
    (history, logical_end): (usize, usize),
    row_bytes: usize,
    layer_idx: usize,
    what: &str,
) -> Result<()> {
    mlx_native::ops::kv_cache_copy::dispatch_kv_cache_linearize_ring_bytes(
        s.encoder_mut(),
        reg,
        dev.metal_device(),
        src,
        dst,
        nkv as u32,
        src_capacity as u32,
        dst_capacity as u32,
        history as u32,
        logical_end as u32,
        row_bytes as u32,
    )
    .map_err(|e| anyhow::anyhow!("live {what} L{layer_idx}: {e}"))
}

fn encode_live_seq(
    s: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    seq: &LiveSeq<'_>,
    kernel: GemmaLiveKernel,
) -> Result<()> {
    let kv = seq.kv;
    let span = seq.span;
    let (cap, is_ring, layer_idx) = (kv.capacity, kv.is_sliding, layer.layer_idx);
    let (nkv, hd) = (layer.nkv, layer.hd);
    let row_elems = layer.nh * hd;
    let kv_bufs = (&kv.k, &kv.v_packed, &kv.v_norms);

    // Global storage is append-only: the whole suffix lands before its
    // queries run. Its write position is absolute (graft rows first).
    if !is_ring {
        let write_start = layer.graft_len + span.start;
        anyhow::ensure!(
            write_start + span.rows <= cap,
            "live global KV overflow L{layer_idx}: start={write_start} len={} cap={cap}",
            span.rows
        );
        s.barrier_between(&[layer.k, layer.v], &[&kv.k, &kv.v_packed, &kv.v_norms]);
        write_kv_rows(
            s,
            reg,
            dev,
            layer,
            kv_bufs,
            cap,
            write_start,
            span.rows,
            span.offset,
            false,
            "global append",
        )?;
    }

    let stage = if is_ring {
        Some(RingStage::alloc(
            dev,
            kv,
            nkv,
            hd,
            stage_query_rows(kernel, span.rows),
            layer_idx,
        )?)
    } else {
        None
    };
    for (local_start, len) in chunks(kernel, is_ring, span.start, span.rows) {
        let stream_row = span.offset + local_start;
        let absolute_start = span.start + local_start;
        let keys = chunk_keys(is_ring, cap, layer.graft_len, absolute_start, len);
        let q = layer
            .q
            .slice_view((stream_row * row_elems * 4) as u64, len * row_elems);
        let out = layer
            .out
            .slice_view((stream_row * row_elems * 4) as u64, len * row_elems);
        let (attn_k, attn_v, attn_norms, kv_capacity) = if let Some(stage) = stage.as_ref() {
            let history = keys.query_offset;
            // The stage copies this chunk's suffix rows out of the normed
            // K/V stream, so declare those reads too: the head norm/RoPE that
            // writes them may still be in the current concurrent group.
            s.barrier_between(
                &[&kv.k, &kv.v_packed, &kv.v_norms, layer.k, layer.v],
                &[&stage.k, &stage.v, &stage.norms],
            );
            let rings = (cap, stage.capacity);
            let range = (history, absolute_start);
            let f16_row = hd * DType::F16.size_of();
            let v_row = hd * kv.v_packed.dtype().size_of();
            let (k_pair, v_pair) = ((&kv.k, &stage.k), (&kv.v_packed, &stage.v));
            linearize_rows(
                s, reg, dev, k_pair, nkv, rings, range, f16_row, layer_idx, "stage K",
            )?;
            linearize_rows(
                s, reg, dev, v_pair, nkv, rings, range, v_row, layer_idx, "stage V",
            )?;
            if kv.v_packed.dtype() != DType::F16 {
                let norm_row = kv.norms_per_pos * DType::F32.size_of();
                let norms = (&kv.v_norms, &stage.norms);
                linearize_rows(
                    s,
                    reg,
                    dev,
                    norms,
                    nkv,
                    rings,
                    range,
                    norm_row,
                    layer_idx,
                    "stage norms",
                )?;
            }
            write_kv_rows(
                s,
                reg,
                dev,
                layer,
                (&stage.k, &stage.v, &stage.norms),
                stage.capacity,
                history,
                len,
                stream_row,
                false,
                "ring stage suffix",
            )?;
            (&stage.k, &stage.v, &stage.norms, stage.capacity)
        } else {
            (&kv.k, &kv.v_packed, &kv.v_norms, cap)
        };

        match kernel {
            GemmaLiveKernel::TiledF16 => encode_tiled_chunk(
                s,
                reg,
                dev,
                layer,
                (attn_k, attn_v, attn_norms),
                kv_capacity,
                is_ring,
                keys,
                len,
                &q,
                &out,
            )?,
            GemmaLiveKernel::VecHybrid => {
                let scratch = layer.vec.ok_or_else(|| {
                    anyhow::anyhow!("live vec-hybrid attention L{layer_idx} has no scratch")
                })?;
                let slot_chunk = scratch.slot_ids.slice_view(0, len);
                let pos_source = if is_ring {
                    &scratch.stage_positions
                } else {
                    layer.positions
                };
                let pos_chunk = pos_source.slice_view((stream_row * 4) as u64, len);
                let kv_seq_len = keys.kv_seq_len as u32;
                let params = mlx_native::ops::flash_attn_vec_hybrid::FlashAttnVecTqHbParams {
                    num_heads: layer.nh as u32,
                    num_kv_heads: nkv as u32,
                    head_dim: hd as u32,
                    kv_seq_len,
                    kv_capacity: kv_capacity as u32,
                    scale: 1.0,
                    mask_type: if is_ring { 3 } else { 1 },
                    sliding_window: if is_ring { layer.sliding_window } else { 0 },
                    softcap: 0.0,
                    ring_start: 0,
                    scale_factor_d512: layer.tq_scale_factor_d512,
                    codebook_bits: layer.tq_codebook_bits,
                    fuse_fwht_pre: 0,
                    nsg: mlx_native::ops::flash_attn_vec_tq_hb::compute_nsg(kv_seq_len),
                };
                s.barrier_between(
                    &[
                        &q,
                        attn_k,
                        attn_v,
                        attn_norms,
                        &scratch.tmp,
                        &slot_chunk,
                        &pos_chunk,
                    ],
                    &[&out, &scratch.tmp],
                );
                mlx_native::ops::flash_attn_vec_hybrid::flash_attn_vec_hybrid_batched(
                    s.encoder_mut(),
                    reg,
                    dev,
                    len as u32,
                    &q,
                    attn_k,
                    attn_v,
                    attn_norms,
                    &out,
                    &scratch.tmp,
                    &slot_chunk,
                    &pos_chunk,
                    &params,
                )
                .map_err(|e| anyhow::anyhow!("live hybrid SDPA L{layer_idx}: {e}"))?;
            }
        }

        // The chunk's history has been read; commit its rows into the ring.
        if is_ring {
            s.barrier_between(&[layer.k, layer.v], &[&kv.k, &kv.v_packed, &kv.v_norms]);
            write_kv_rows(
                s,
                reg,
                dev,
                layer,
                kv_bufs,
                cap,
                absolute_start,
                len,
                stream_row,
                true,
                "ring commit",
            )?;
        }
    }
    Ok(())
}

/// Tiled F16 flash attention for one chunk of one sequence: compact K,
/// expand V to F16, attend with `batch = 1`, and write F32 rows back.
#[allow(clippy::too_many_arguments)]
fn encode_tiled_chunk(
    s: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    (src_k, src_v, src_norms): (&MlxBuffer, &MlxBuffer, &MlxBuffer),
    src_capacity: usize,
    is_ring: bool,
    keys: ChunkKeys,
    len: usize,
    q: &MlxBuffer,
    out: &MlxBuffer,
) -> Result<()> {
    let metal_dev = dev.metal_device();
    let (nh, nkv, hd, layer_idx) = (layer.nh, layer.nkv, layer.hd, layer.layer_idx);
    let row_elems = nh * hd;
    let kv_seq_len = keys.kv_seq_len;
    let scratch = |elements: usize, dtype: DType, shape: Vec<usize>, what: &str| {
        // SAFETY: the compaction / dequantization / permute dispatches below
        // initialize every element the attention reads; O is fully written
        // by the attention kernel before the cast reads it.
        let buffer =
            unsafe { dev.alloc_buffer_for_overwrite(elements * dtype.size_of(), dtype, shape) };
        buffer.map_err(|e| anyhow::anyhow!("live tiled {what} scratch L{layer_idx}: {e}"))
    };
    let dense_k = scratch(
        nkv * kv_seq_len * hd,
        DType::F16,
        vec![nkv, kv_seq_len, hd],
        "K",
    )?;
    let dense_v = scratch(
        nkv * kv_seq_len * hd,
        DType::F16,
        vec![nkv, kv_seq_len, hd],
        "V",
    )?;
    let q_f16 = scratch(len * row_elems, DType::F16, vec![nh, len, hd], "Q")?;
    let out_f16 = scratch(len * row_elems, DType::F16, vec![nh, len, hd], "O")?;
    let out_f32_head = scratch(len * row_elems, DType::F32, vec![nh, len, hd], "O32")?;

    s.barrier_between(&[src_k, src_v, src_norms, q], &[&dense_k, &dense_v, &q_f16]);
    // `history = logical_end = kv_seq_len` copies rows `[0, kv_seq_len)`:
    // the stage is already chronological and global rows are linear.
    let rings = (src_capacity, kv_seq_len);
    let range = (kv_seq_len, kv_seq_len);
    let f16_row = hd * DType::F16.size_of();
    let k_pair = (src_k, &dense_k);
    linearize_rows(
        s, reg, dev, k_pair, nkv, rings, range, f16_row, layer_idx, "tiled K",
    )?;
    if src_v.dtype() == DType::F16 {
        let v_pair = (src_v, &dense_v);
        linearize_rows(
            s, reg, dev, v_pair, nkv, rings, range, f16_row, layer_idx, "tiled V",
        )?;
    } else {
        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq_f16(
            s.encoder_mut(),
            reg,
            metal_dev,
            src_v,
            src_norms,
            &dense_v,
            nkv as u32,
            hd as u32,
            src_capacity as u32,
            0,
            kv_seq_len as u32,
            layer.tq_scale_factor_d512,
            layer.tq_codebook_bits,
        )
        .map_err(|e| anyhow::anyhow!("live tiled V expand L{layer_idx}: {e}"))?;
    }
    mlx_native::ops::transpose::permute_021_f32_to_f16(
        s.encoder_mut(),
        reg,
        metal_dev,
        q,
        &q_f16,
        len,
        nh,
        hd,
    )
    .map_err(|e| anyhow::anyhow!("live tiled Q permute L{layer_idx}: {e}"))?;

    if is_ring {
        let mask = mlx_native::ops::flash_attn_prefill_mask::build_sdpa_mask_f16(
            dev,
            reg,
            s.encoder_mut(),
            &mlx_native::ops::flash_attn_prefill_mask::SdpaMaskParams {
                seq_len_q: len as u32,
                seq_len_k: kv_seq_len as u32,
                window_size: Some(layer.sliding_window),
                causal: true,
                q_abs_offset: keys.query_offset as u32,
            },
        )
        .map_err(|e| anyhow::anyhow!("live tiled mask L{layer_idx}: {e}"))?;
        // The mask buffer is allocated inside its builder, so its write was
        // never declared to the conflict tracker: without this fence the
        // `blk` classifier below runs concurrently with the mask fill and
        // can classify a partly written (zero = attended) mask, letting the
        // tile-skip path attend future or out-of-window keys. The fence also
        // completes the K/V compaction and Q permute above.
        s.barrier();
        let blk_params = mlx_native::ops::flash_attn_prefill_blk::BlkParams {
            seq_len_q: len as u32,
            seq_len_k: kv_seq_len as u32,
            bq: 32,
            bk: 16,
        };
        let blk = mlx_native::ops::flash_attn_prefill_blk::alloc_blk_buffer(dev, &blk_params)
            .map_err(|e| anyhow::anyhow!("live tiled blk alloc L{layer_idx}: {e}"))?;
        s.barrier_between(&[&mask], &[&blk]);
        mlx_native::ops::flash_attn_prefill_blk::dispatch_flash_attn_prefill_blk_f16(
            s.encoder_mut(),
            dev,
            reg,
            &mask,
            &blk,
            &blk_params,
        )
        .map_err(|e| anyhow::anyhow!("live tiled blk L{layer_idx}: {e}"))?;
        s.barrier_between(&[&q_f16, &dense_k, &dense_v, &mask, &blk], &[&out_f16]);
        mlx_native::ops::flash_attn_prefill::dispatch_flash_attn_prefill_f16_d256_with_blk(
            s.encoder_mut(),
            dev,
            reg,
            &q_f16,
            &dense_k,
            &dense_v,
            Some(&mask),
            Some(&blk),
            &out_f16,
            &mlx_native::ops::flash_attn_prefill::FlashAttnPrefillParams {
                n_heads: nh as u32,
                n_kv_heads: nkv as u32,
                head_dim: hd as u32,
                seq_len_q: len as u32,
                seq_len_k: kv_seq_len as u32,
                batch: 1,
                scale: 1.0,
                do_causal: false,
            },
        )
        .map_err(|e| anyhow::anyhow!("live tiled sliding FA L{layer_idx}: {e}"))?;
    } else {
        anyhow::ensure!(
            hd == 512,
            "live tiled global layer L{layer_idx} has head_dim={hd}"
        );
        s.barrier_between(&[&q_f16, &dense_k, &dense_v], &[&out_f16]);
        mlx_native::ops::flash_attn_prefill_d512::dispatch_flash_attn_prefill_f16_d512_resume(
            s.encoder_mut(),
            dev,
            reg,
            &q_f16,
            &dense_k,
            &dense_v,
            &out_f16,
            &mlx_native::ops::flash_attn_prefill::FlashAttnPrefillResumeParams {
                n_heads: nh as u32,
                n_kv_heads: nkv as u32,
                head_dim: hd as u32,
                seq_len_q: len as u32,
                seq_len_k: kv_seq_len as u32,
                batch: 1,
                scale: 1.0,
                do_causal: true,
                q_offset_in_k: keys.query_offset as u32,
                kv_capacity: kv_seq_len as u32,
            },
        )
        .map_err(|e| anyhow::anyhow!("live tiled global FA L{layer_idx}: {e}"))?;
    }
    s.barrier_between(&[&out_f16], &[&out_f32_head]);
    mlx_native::ops::elementwise::cast(
        s.encoder_mut(),
        reg,
        metal_dev,
        &out_f16,
        &out_f32_head,
        len * row_elems,
        mlx_native::ops::elementwise::CastDirection::F16ToF32,
    )
    .map_err(|e| anyhow::anyhow!("live tiled O cast L{layer_idx}: {e}"))?;
    s.barrier_between(&[&out_f32_head], &[out]);
    mlx_native::ops::transpose::permute_021_f32(
        s.encoder_mut(),
        reg,
        metal_dev,
        &out_f32_head,
        out,
        nh,
        len,
        hd,
    )
    .map_err(|e| anyhow::anyhow!("live tiled O permute L{layer_idx}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::gemma4::GemmaLiveKernel::{TiledF16, VecHybrid};

    fn chunk_list(kernel: GemmaLiveKernel, is_ring: bool, start: usize, rows: usize) -> Vec<usize> {
        chunks(kernel, is_ring, start, rows)
            .map(|(_, len)| len)
            .collect()
    }

    #[test]
    fn warm_switch_defaults_on_and_zero_selects_the_vec_hybrid_route() {
        for raw in [
            None,
            Some(""),
            Some("  "),
            Some("1"),
            Some("on"),
            Some("TRUE"),
        ] {
            assert_eq!(warm_kernel_for_switch(raw), TiledF16, "{raw:?}");
        }
        for raw in ["0", "off", "False", " 0 "] {
            assert_eq!(warm_kernel_for_switch(Some(raw)), VecHybrid, "{raw:?}");
        }
        assert!(parse_warm_tiled_switch(Some("yes")).is_err());
        // An invalid value keeps the default rather than silently reverting.
        assert_eq!(warm_kernel_for_switch(Some("yes")), TiledF16);
    }

    #[test]
    fn single_slot_and_joint_cohort_always_resolve_the_same_route() {
        for warm in [TiledF16, VecHybrid] {
            for tiled_live in [true, false] {
                let single = LiveRoute::select(None, warm, tiled_live);
                let joint =
                    LiveRoute::select(Some(GemmaLiveAttention::PerSlotExact), warm, tiled_live);
                assert_eq!(single, joint, "warm={warm:?} tiled_live={tiled_live}");
                for rows in [1, 31, 32, 33, 56, 385, 4096] {
                    assert_eq!(single.seq_kernel(rows), joint.seq_kernel(rows));
                }
            }
        }
    }

    #[test]
    fn warm_route_follows_its_switch_and_keeps_tiny_suffixes_on_vec_hybrid() {
        // `HF2Q_GEMMA_TILED_LIVE` never moves the warm route.
        for tiled_live in [true, false] {
            let tiled = LiveRoute::select(None, TiledF16, tiled_live);
            assert_eq!(tiled.seq_kernel(1), VecHybrid);
            assert_eq!(tiled.seq_kernel(31), VecHybrid);
            assert_eq!(tiled.seq_kernel(32), TiledF16);
            assert_eq!(tiled.seq_kernel(385), TiledF16);
            let vec = LiveRoute::select(None, VecHybrid, tiled_live);
            assert_eq!(vec.seq_kernel(385), VecHybrid);
        }
    }

    #[test]
    fn installed_lanes_keep_their_own_switch_and_tile_every_length() {
        // `HF2Q_GEMMA_WARM_TILED_ATTENTION` never moves installed lanes.
        for warm in [TiledF16, VecHybrid] {
            let on = LiveRoute::select(Some(GemmaLiveAttention::Tiled), warm, true);
            assert_eq!(on.seq_kernel(1), TiledF16, "a 4096+4096+1 tail stays tiled");
            assert_eq!(on.seq_kernel(4096), TiledF16);
            let off = LiveRoute::select(Some(GemmaLiveAttention::Tiled), warm, false);
            assert_eq!(off.seq_kernel(4096), VecHybrid);
        }
    }

    #[test]
    fn vec_scratch_is_needed_only_when_some_sequence_takes_vec_hybrid() {
        let span = |rows| LiveSpan {
            offset: 0,
            rows,
            start: 465,
        };
        let tiled = LiveRoute::select(None, TiledF16, true);
        assert!(!tiled.uses_vec(&[span(56), span(385)]));
        assert!(tiled.uses_vec(&[span(56), span(5)]));
        assert!(LiveRoute::select(None, VecHybrid, true).uses_vec(&[span(56)]));
    }

    #[test]
    fn vec_chunks_are_bounded_and_do_not_cross_kernel_buckets() {
        assert_eq!(vec_query_chunk_len(0, 900), 256);
        assert_eq!(vec_query_chunk_len(500, 900), 12);
        assert_eq!(vec_query_chunk_len(512, 900), 256);
        assert_eq!(vec_query_chunk_len(1000, 900), 24);
        assert_eq!(vec_query_chunk_len(1024, 900), 256);
        assert_eq!(vec_query_chunk_len(7_000, 17), 17);
        // Gate shape: history 465 + suffix 385 splits at 512.
        assert_eq!(chunk_list(VecHybrid, true, 465, 385), [47, 256, 82]);
    }

    #[test]
    fn tiled_chunks_cover_each_sequence_in_window_and_transaction_sized_steps() {
        // Every gate lane is one chunk on both layer kinds.
        for (start, rows) in [(465, 56), (714, 200), (943, 311), (1254, 385)] {
            assert_eq!(chunk_list(TiledF16, true, start, rows), [rows]);
            assert_eq!(chunk_list(TiledF16, false, start, rows), [rows]);
        }
        assert_eq!(chunk_list(TiledF16, true, 0, 2_500), [1024, 1024, 452]);
        assert_eq!(chunk_list(TiledF16, false, 7, 4_096), [4096]);
        assert_eq!(chunk_list(TiledF16, false, 7, 9_000), [4096, 4096, 808]);
        // Chunks are contiguous and cover the sequence exactly.
        for kernel in [TiledF16, VecHybrid] {
            for is_ring in [true, false] {
                let mut next = 0;
                for (local, len) in chunks(kernel, is_ring, 1_000, 5_000) {
                    assert_eq!(local, next);
                    assert!(len > 0);
                    next += len;
                }
                assert_eq!(next, 5_000);
            }
        }
    }

    #[test]
    fn ring_stage_holds_the_window_history_plus_the_largest_chunk() {
        let cap = 1024;
        for kernel in [TiledF16, VecHybrid] {
            for (start, rows) in [(465, 56), (1254, 385), (0, 2_500), (40_000, 3_000)] {
                let stage = cap - 1 + stage_query_rows(kernel, rows);
                for (local, len) in chunks(kernel, true, start, rows) {
                    let keys = chunk_keys(true, cap, 0, start + local, len);
                    assert!(keys.kv_seq_len <= stage, "{kernel:?} {start}+{local}");
                    assert_eq!(keys.query_offset, (start + local).min(cap - 1));
                }
            }
        }
    }

    #[test]
    fn global_chunks_attend_graft_rows_with_an_exact_causal_offset() {
        // The resume kernel's causal row is `q_offset_in_k + i`; with a bound
        // graft the chunk's queries sit after the graft rows, and the last
        // query must see exactly the keys up to itself.
        for graft in [0usize, 37] {
            for (local, len) in chunks(TiledF16, false, 900, 5_000) {
                let keys = chunk_keys(false, 65_536, graft, 900 + local, len);
                assert_eq!(keys.query_offset, graft + 900 + local);
                assert_eq!(keys.query_offset + len, keys.kv_seq_len);
            }
        }
        // Sliding rings never hold graft rows.
        assert_eq!(chunk_keys(true, 1024, 37, 100, 50).query_offset, 100);
    }

    #[test]
    fn vec_stage_positions_are_the_same_for_a_lane_alone_or_in_a_cohort() {
        let cap = 1024;
        let lanes = [(465, 56), (714, 200), (943, 311), (1254, 385)];
        let total: usize = lanes.iter().map(|&(_, rows)| rows).sum();
        let mut joint = vec![u32::MAX; total];
        let mut offset = 0;
        let mut spans = Vec::new();
        for &(start, rows) in &lanes {
            let span = LiveSpan {
                offset,
                rows,
                start,
            };
            fill_vec_stage_positions(&mut joint, span, cap);
            spans.push(span);
            offset += rows;
        }
        for span in spans {
            let mut alone = vec![u32::MAX; span.rows];
            fill_vec_stage_positions(&mut alone, LiveSpan { offset: 0, ..span }, cap);
            assert_eq!(
                &joint[span.offset..span.offset + span.rows],
                alone.as_slice()
            );
        }
        // History 1254 saturates the 1023-row stage: each 256-query chunk
        // restarts at 1023.
        let last = &joint[total - 385..];
        assert_eq!(last[0], 1023);
        assert_eq!(last[255], 1023 + 255);
        assert_eq!(last[256], 1023);
        // History 465 is below the cap: positions are absolute.
        assert_eq!(joint[0], 465);
        assert!(joint.iter().all(|&value| value != u32::MAX));
    }

    #[test]
    fn stage_history_follows_the_actual_cache_history() {
        assert_eq!(stage_history_len(2_005, 18_404), 2_005);
        assert_eq!(stage_history_len(2_005, 1_024), 1_023);
        assert_eq!(stage_history_len(0, 0), 0);
    }
}
