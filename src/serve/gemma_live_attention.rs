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
//! Three kernels implement the attention step:
//!
//! * [`GemmaLiveKernel::TiledF16`]: compact K to dense F16, dequantize (or
//!   compact) the active V range to F16, then tiled flash attention: d256
//!   with an SWA mask and tile-skip `blk` for sliding layers (1024-query
//!   chunks), `flash_attn_prefill_f16_d512_resume` for global layers
//!   (4096-query chunks). One `batch = 1` dispatch per (sequence, chunk):
//!   no block-diagonal mask is involved.
//! * [`GemmaLiveKernel::TensorF16`] (ADR-063 gap program F1): the same
//!   compaction and F16 V expansion into a dense range whose 64-row key tiles
//!   are aligned to absolute positions, then mlx-native's tensor-API flash
//!   attention (`flash_attn_prefill_tensor_f16_d{256,512}`) reading the
//!   layer's F32 Q rows and writing its F32 output rows in place; same
//!   chunks as `TiledF16`.
//! * [`GemmaLiveKernel::VecHybrid`]: `flash_attn_vec_hybrid_batched`, one
//!   decode-style threadgroup per (query, head), in 256-query chunks split
//!   at the 512/1024 cache-length buckets.
//!
//! Transient scratch of the tensor route. Sliding layers stage at most one
//! window of history plus the chunk, so their scratch is bounded by the
//! suffix rows. Global layers attend the whole history: the dense F16 K and V
//! copies cost `4 · nkv · hd` bytes per key, the additive mask
//! `2 · rows · keys` bytes and its tile classification (`blk`) one byte per
//! tile of queries-per-threadgroup × 64 keys. [`LiveScratch`] keeps that history-proportional
//! part bounded for one forward: a sequence's dense K/V are built once per
//! global layer, its queries run in sub-chunks whose mask stays under
//! [`TENSOR_GLOBAL_MASK_BYTES_MAX`], and the forward waits for the GPU
//! before it would hold more than [`TENSOR_GLOBAL_SCRATCH_BUDGET_BYTES`] of
//! submitted global scratch. Every buffer the global path allocates is
//! accounted at its exact size: both dense copies, each mask and each `blk`
//! buffer ([`tensor_global_envelope_bytes`] is the resulting bound). Sub-chunks start on threadgroup boundaries and
//! the waits only order work, so every output bit equals the unsplit route.
//!
//! Everything that reaches a kernel is a function of the sequence alone
//! (its rows, start, KV view and Q/K/V rows), so a lane of a joint cohort is
//! bitwise equal to the same sequence prefilled alone whenever both resolve
//! the same kernel. [`LiveRoute`] guarantees that: a single-slot resume and
//! every multi-sequence lane resolve the one warm route.
use anyhow::Result;
use mlx_native::{DType, GraphExecutor, GraphSession, KernelRegistry, MlxBuffer, MlxDevice};

use crate::inference::models::gemma4::{GemmaLiveKernel, HybridKvBuffers};
use crate::inference::models::route_default::{is_kill_spelling, Resolved, RouteSource};
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

/// Largest additive mask of one tensor global-layer dispatch. A sequence's
/// queries split into sub-chunks under it; 64 MiB leaves a 4,096-row suffix
/// unsplit up to 8,192 keys.
pub(crate) const TENSOR_GLOBAL_MASK_BYTES_MAX: u64 = 64 << 20;
/// Submitted-but-unfinished global-layer tensor scratch one forward may hold
/// before it waits for the GPU. One sequence's dense K/V of one layer may
/// exceed it alone (4 KiB per key on Gemma 4 26B-A4B: 1 GiB at 262,144
/// keys); the forward then holds that one pair plus one mask.
pub(crate) const TENSOR_GLOBAL_SCRATCH_BUDGET_BYTES: u64 = 512 << 20;

/// Limits of the tensor route's history-proportional scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TensorScratchLimits {
    pub(crate) mask_bytes: u64,
    pub(crate) budget_bytes: u64,
}

impl TensorScratchLimits {
    pub(crate) const SERVING: Self = Self {
        mask_bytes: TENSOR_GLOBAL_MASK_BYTES_MAX,
        budget_bytes: TENSOR_GLOBAL_SCRATCH_BUDGET_BYTES,
    };
    /// One dispatch per chunk and no waits: the route as first shipped,
    /// which real-artifact gates compare the bounded route against.
    #[cfg(test)]
    pub(crate) const UNBOUNDED: Self = Self {
        mask_bytes: u64::MAX,
        budget_bytes: u64::MAX,
    };
}

#[cfg(test)]
thread_local! {
    static TEST_LIMITS: std::cell::Cell<Option<TensorScratchLimits>> =
        const { std::cell::Cell::new(None) };
    static LAST_FORWARD: std::cell::Cell<LiveScratchStats> =
        const { std::cell::Cell::new(LiveScratchStats { waits: 0, peak_bytes: 0 }) };
}

/// Limits of the calling thread's next forwards (tests only).
#[cfg(test)]
pub(crate) fn set_test_tensor_scratch_limits(limits: Option<TensorScratchLimits>) {
    TEST_LIMITS.with(|cell| cell.set(limits));
}

/// Scratch accounting of the calling thread's last live forward (tests only).
#[cfg(test)]
pub(crate) fn last_forward_scratch_stats() -> LiveScratchStats {
    LAST_FORWARD.with(std::cell::Cell::get)
}

/// What one forward's [`LiveScratch`] did.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct LiveScratchStats {
    /// GPU waits taken to release submitted scratch.
    pub(crate) waits: u32,
    /// Largest accounted global-layer tensor scratch alive at once.
    pub(crate) peak_bytes: u64,
}

/// Accounting of the tensor route's global-layer scratch across one forward.
///
/// `held` is scratch the encoder still owns (a sequence's dense K/V while
/// its sub-chunks run); `submitted` is scratch whose owner dropped it but
/// whose command buffers may not have finished, so Metal still holds it.
/// Before admitting more than the budget the forward finishes the session
/// (a full GPU wait) and continues in a fresh one, which releases everything
/// submitted. The accounting assumes no command buffer finishes on its own,
/// so it is an upper bound.
pub(crate) struct LiveScratch<'e> {
    exec: &'e GraphExecutor,
    limits: TensorScratchLimits,
    /// False while the layer's session cannot be split (recorded graph
    /// sessions of `HF2Q_GRAPH_OPT_PREFILL=1`): the route then accounts
    /// without waiting.
    can_wait: bool,
    held: u64,
    submitted: u64,
    stats: LiveScratchStats,
}

impl<'e> LiveScratch<'e> {
    pub(crate) fn new(exec: &'e GraphExecutor) -> Self {
        #[cfg(test)]
        let limits = TEST_LIMITS
            .with(std::cell::Cell::get)
            .unwrap_or(TensorScratchLimits::SERVING);
        #[cfg(not(test))]
        let limits = TensorScratchLimits::SERVING;
        Self {
            exec,
            limits,
            can_wait: true,
            held: 0,
            submitted: 0,
            stats: LiveScratchStats::default(),
        }
    }

    /// Whether the current layer's session may be finished and reopened.
    pub(crate) fn set_can_wait(&mut self, can_wait: bool) {
        self.can_wait = can_wait;
    }

    /// The layer boundary waited for the GPU: nothing submitted remains.
    pub(crate) fn layer_finished(&mut self) {
        self.submitted = 0;
    }

    fn admit(&mut self, s: &mut GraphSession<'e>, bytes: u64, held: bool) -> Result<()> {
        if self.can_wait
            && self.submitted > 0
            && self
                .held
                .saturating_add(self.submitted)
                .saturating_add(bytes)
                > self.limits.budget_bytes
        {
            if crate::serve::gemma_prefill_catsplit::enabled() {
                // Keep the warm-prefill split's attention time whole.
                crate::serve::gemma_prefill_catsplit::boundary(
                    s,
                    self.exec,
                    crate::serve::gemma_prefill_catsplit::Cat::Attention,
                )?;
            } else {
                let fresh = self
                    .exec
                    .begin()
                    .map(crate::inference::native_observation::session)
                    .map_err(|e| anyhow::anyhow!("live tensor scratch re-begin: {e}"))?;
                std::mem::replace(s, fresh)
                    .finish()
                    .map_err(|e| anyhow::anyhow!("live tensor scratch wait: {e}"))?;
            }
            self.submitted = 0;
            self.stats.waits += 1;
        }
        if held {
            self.held = self.held.saturating_add(bytes);
        } else {
            self.submitted = self.submitted.saturating_add(bytes);
        }
        self.stats.peak_bytes = self
            .stats
            .peak_bytes
            .max(self.held.saturating_add(self.submitted));
        Ok(())
    }

    /// The encoder dropped `bytes` of held scratch; Metal keeps it until
    /// the command buffers that use it finish.
    fn release(&mut self, bytes: u64) {
        self.held = self.held.saturating_sub(bytes);
        self.submitted = self.submitted.saturating_add(bytes);
    }
}

impl Drop for LiveScratch<'_> {
    fn drop(&mut self) {
        if self.stats.waits > 0 {
            tracing::debug!(
                waits = self.stats.waits,
                peak_bytes = self.stats.peak_bytes,
                "Gemma live tensor attention waited to bound its global-layer scratch"
            );
        }
        #[cfg(test)]
        LAST_FORWARD.with(|cell| cell.set(self.stats));
    }
}

/// Query rows per tensor global-layer dispatch of a sequence whose padded
/// key count is `kv_pad`: the largest multiple of the kernel's queries per
/// threadgroup (`bq`) whose `[rows, kv_pad]` F16 mask stays under
/// `mask_bytes`, at least `bq` and at most [`TILED_GLOBAL_QUERY_CHUNK`]. A
/// function of the sequence's own keys, so a lane splits exactly as the
/// same sequence alone.
fn tensor_global_query_rows(kv_pad: usize, bq: usize, mask_bytes: u64) -> usize {
    let row_bytes = (kv_pad as u64)
        .saturating_mul(DType::F16.size_of() as u64)
        .max(1);
    let rows = usize::try_from(mask_bytes / row_bytes).unwrap_or(usize::MAX);
    (rows / bq * bq).clamp(bq, TILED_GLOBAL_QUERY_CHUNK)
}

pub(crate) const WARM_TILED_ATTENTION_ENV: &str = "HF2Q_GEMMA_WARM_TILED_ATTENTION";
/// Unset default: the tensor-API route of the Gemma 4 release gate (ADR-063
/// Q4-GEMMA-ATTN, campaign `q4-gemma-release-gate-max40-v3`). The tiled F16
/// route stays opt-in: under a four-agent cohort it moved a lane's decision
/// logits enough to change its greedy choice at a 1.5-nat solo margin
/// (quality gate, 2026-11-01).
const WARM_ATTENTION_DEFAULT: GemmaLiveKernel = GemmaLiveKernel::TensorF16;
/// Kill-switch route (`0`/`off`/`false`), and the route an unset switch
/// falls back to on a device without the Metal tensor API.
const WARM_ATTENTION_STOCK: GemmaLiveKernel = GemmaLiveKernel::VecHybrid;

/// `tensor` selects the gap-program F1 route `TensorF16` (also the unset
/// default), `1`/`on`/`true` `TiledF16`, `0`/`off`/`false`/`stock`
/// vec-hybrid; anything else is an error.
fn parse_warm_tiled_switch(
    raw: Option<&str>,
) -> std::result::Result<Resolved<GemmaLiveKernel>, String> {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Ok(Resolved::default_value(WARM_ATTENTION_DEFAULT));
    };
    match raw.to_ascii_lowercase().as_str() {
        "1" | "on" | "true" => Ok(Resolved::environment(GemmaLiveKernel::TiledF16)),
        "tensor" => Ok(Resolved::environment(GemmaLiveKernel::TensorF16)),
        value if is_kill_spelling(value) => Ok(Resolved::environment(WARM_ATTENTION_STOCK)),
        _ => Err(format!(
            "{WARM_TILED_ATTENTION_ENV} must be tensor, 1/on/true or 0/off/false/stock, got {raw:?}"
        )),
    }
}

/// Applies the device probe to a resolved switch. The tensor route needs
/// the Metal tensor API: an unset switch whose default cannot build on this
/// device falls back to vec-hybrid (with the reason), and an explicit
/// `tensor` there is an error, so the load refuses it instead of failing at
/// the first warm resume. Other kernels are never probed.
fn admit_warm_kernel(
    resolved: Resolved<GemmaLiveKernel>,
    tensor_unavailable: impl FnOnce() -> Option<String>,
) -> std::result::Result<(Resolved<GemmaLiveKernel>, Option<String>), String> {
    if resolved.value != GemmaLiveKernel::TensorF16 {
        return Ok((resolved, None));
    }
    match (tensor_unavailable(), resolved.source) {
        (None, _) => Ok((resolved, None)),
        (Some(reason), RouteSource::Default) => {
            Ok((Resolved::fallback(WARM_ATTENTION_STOCK), Some(reason)))
        }
        (Some(reason), _) => Err(format!(
            "{WARM_TILED_ATTENTION_ENV}=tensor cannot run on this host: {reason}. Unset it for \
             the automatic route (vec-hybrid here) or set 0 for vec-hybrid"
        )),
    }
}

/// Why the tensor-API warm attention pipelines cannot run on `device`
/// (`None` when both head dims build).
fn tensor_route_unavailable(registry: &mut KernelRegistry, device: &MlxDevice) -> Option<String> {
    use mlx_native::ops::flash_attn_prefill_tensor::flash_attn_prefill_tensor_available;
    for head_dim in [256, 512] {
        match flash_attn_prefill_tensor_available(registry, device, head_dim) {
            Ok(true) => {}
            Ok(false) => {
                return Some(format!(
                "the Metal tensor API is unavailable for the d{head_dim} warm attention pipeline"
            ))
            }
            Err(error) => {
                return Some(format!(
                    "the d{head_dim} tensor warm attention pipeline does not build: {error}"
                ))
            }
        }
    }
    None
}

/// Kernel for warm live-resume attention (single-slot resumes, joint warm
/// cohorts and installed long-prompt lanes), resolved once per model load
/// from `HF2Q_GEMMA_WARM_TILED_ATTENTION`. Unset selects the tensor-API
/// route where the device builds it and vec-hybrid otherwise. An invalid
/// value refuses the load: a mistyped kill switch must never serve the
/// route it was meant to turn off.
pub(crate) fn warm_live_kernel_from_env(
    registry: &mut KernelRegistry,
    device: &MlxDevice,
) -> Result<Resolved<GemmaLiveKernel>> {
    let raw = std::env::var(WARM_TILED_ATTENTION_ENV).ok();
    let resolved = parse_warm_tiled_switch(raw.as_deref()).map_err(anyhow::Error::msg)?;
    let (kernel, fallback_reason) =
        admit_warm_kernel(resolved, || tensor_route_unavailable(registry, device))
            .map_err(anyhow::Error::msg)?;
    if let Some(reason) = &fallback_reason {
        tracing::warn!(
            reason,
            kernel = ?kernel.value,
            "Gemma warm live attention default is unavailable; using the stock kernel"
        );
    }
    tracing::info!(
        kernel = ?kernel.value,
        source = kernel.source.as_str(),
        "Gemma warm live attention kernel"
    );
    Ok(kernel)
}

/// Warm kernel with `HF2Q_GEMMA_WARM_TILED_ATTENTION` unset on a device with
/// the Metal tensor API (the shipped default); ADR-063 Q1 harnesses pin
/// their expected route against it.
#[cfg(test)]
pub(crate) fn default_warm_live_kernel() -> GemmaLiveKernel {
    WARM_ATTENTION_DEFAULT
}

/// Attention route of one live-resume forward: the model's warm kernel, with
/// suffixes under [`WARM_TILED_MIN_ROWS`] on `VecHybrid`. Single-slot
/// resumes and every multi-sequence lane take this one route.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LiveRoute {
    kernel: GemmaLiveKernel,
}

impl LiveRoute {
    pub(crate) fn resolve(warm_kernel: GemmaLiveKernel) -> Self {
        Self {
            kernel: warm_kernel,
        }
    }

    pub(crate) fn seq_kernel(self, rows: usize) -> GemmaLiveKernel {
        match self.kernel {
            GemmaLiveKernel::TiledF16 if rows >= WARM_TILED_MIN_ROWS => GemmaLiveKernel::TiledF16,
            GemmaLiveKernel::TensorF16 if rows >= WARM_TILED_MIN_ROWS => GemmaLiveKernel::TensorF16,
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
        match self.kernel {
            GemmaLiveKernel::VecHybrid => "vec-hybrid",
            GemmaLiveKernel::TiledF16 => "tiled-f16 (vec-hybrid below 32 rows)",
            GemmaLiveKernel::TensorF16 => "tensor-f16 (vec-hybrid below 32 rows)",
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
        GemmaLiveKernel::TiledF16 | GemmaLiveKernel::TensorF16 if is_ring => {
            remaining.min(TILED_SLIDING_QUERY_CHUNK)
        }
        GemmaLiveKernel::TiledF16 | GemmaLiveKernel::TensorF16 => {
            remaining.min(TILED_GLOBAL_QUERY_CHUNK)
        }
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
        GemmaLiveKernel::TiledF16 | GemmaLiveKernel::TensorF16 => TILED_SLIDING_QUERY_CHUNK,
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
pub(crate) fn encode_live_layer_attention<'a, 'e>(
    s: &mut GraphSession<'e>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    route: LiveRoute,
    seqs: impl IntoIterator<Item = LiveSeq<'a>>,
    scratch: &mut LiveScratch<'e>,
) -> Result<()> {
    let mut packed_v: Option<bool> = None;
    for seq in seqs {
        let seq_packed = seq.kv.v_packed.dtype() != DType::F16;
        anyhow::ensure!(
            *packed_v.get_or_insert(seq_packed) == seq_packed,
            "live attention L{}: sequences disagree on the V storage dtype",
            layer.layer_idx
        );
        encode_live_seq(
            s,
            reg,
            dev,
            layer,
            &seq,
            route.seq_kernel(seq.span.rows),
            scratch,
        )?;
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

fn encode_live_seq<'e>(
    s: &mut GraphSession<'e>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    seq: &LiveSeq<'_>,
    kernel: GemmaLiveKernel,
    scratch: &mut LiveScratch<'e>,
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

    if !is_ring && kernel == GemmaLiveKernel::TensorF16 {
        return encode_tensor_global_seq(s, reg, dev, layer, seq, scratch);
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
            GemmaLiveKernel::TensorF16 => {
                // Absolute position of dense row 0 (the stage starts at the
                // oldest staged history row; global rows are absolute).
                let first_key_abs = if is_ring {
                    absolute_start - keys.query_offset
                } else {
                    0
                };
                encode_tensor_chunk(
                    s,
                    reg,
                    dev,
                    layer,
                    (attn_k, attn_v, attn_norms),
                    kv_capacity,
                    is_ring,
                    keys,
                    first_key_abs,
                    len,
                    &q,
                    &out,
                )?
            }
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

/// Dense F16 K/V of one sequence for the tensor-API kernel (gap program
/// F1). Dense row `r` holds absolute position `first_key_abs - lead + r`
/// with `lead = first_key_abs % 64`: the kernel's 64-row key tiles then
/// start at absolute multiples of 64, so a row's result does not depend on
/// where its chunk or stage begins (design §3). The `lead` rows before the
/// first key and the rows past the last key up to the padded length are zero
/// and masked `-inf`.
///
/// Layout of each buffer: `lead · hd` zeros, then the compacted
/// `[nkv, rows, hd]` rows, then `(kv_pad − lead − rows) · hd` zeros. Head
/// `h`, dense row `r` is element `h · rows · hd + r · hd`, so rows outside a
/// head's own range read neighbouring finite rows (another head's, a later
/// key of the same head, or the zero lead/tail), all masked.
struct TensorDenseKv {
    k: MlxBuffer,
    v: MlxBuffer,
    lead: usize,
    /// Key rows compacted per head.
    rows: usize,
}

/// Bytes of both dense buffers for `rows` keys per head starting at
/// absolute position `first_key_abs`.
fn tensor_dense_kv_bytes(nkv: usize, hd: usize, rows: usize, first_key_abs: usize) -> u64 {
    let step = mlx_native::ops::flash_attn_prefill_tensor::CACHE_ROWS_PER_STEP as usize;
    let lead = first_key_abs % step;
    let pad = (lead + rows).div_ceil(step) * step;
    let elems = (pad - rows) * hd + nkv * rows * hd;
    2 * (elems * DType::F16.size_of()) as u64
}

/// Compact (K) and expand or compact (V) the first `rows` chronological
/// rows of a `[nkv, src_capacity, hd]` cache into fresh dense buffers.
#[allow(clippy::too_many_arguments)]
fn build_tensor_dense_kv(
    s: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    (src_k, src_v, src_norms): (&MlxBuffer, &MlxBuffer, &MlxBuffer),
    src_capacity: usize,
    rows: usize,
    first_key_abs: usize,
) -> Result<TensorDenseKv> {
    use mlx_native::ops::flash_attn_prefill_tensor as tensor;
    let (nkv, hd, layer_idx) = (layer.nkv, layer.hd, layer.layer_idx);
    let step = tensor::CACHE_ROWS_PER_STEP as usize;
    let lead = first_key_abs % step;
    let kv_pad = tensor::kv_pad((lead + rows) as u32) as usize;
    let tail = kv_pad - lead - rows;
    let dense_elems = (lead + tail) * hd + nkv * rows * hd;
    let dense = |what: &str| -> Result<MlxBuffer> {
        let mut buffer = dev
            .alloc_buffer(dense_elems * 2, DType::F16, vec![dense_elems])
            .map_err(|e| anyhow::anyhow!("live tensor {what} scratch L{layer_idx}: {e}"))?;
        // The lead and tail rows are read by the kernel (masked) and must be
        // finite; the compaction below writes everything between them.
        let halves = buffer
            .as_mut_slice::<u16>()
            .map_err(|e| anyhow::anyhow!("live tensor {what} zero fill L{layer_idx}: {e}"))?;
        halves[..lead * hd].fill(0);
        halves[dense_elems - tail * hd..].fill(0);
        Ok(buffer)
    };
    let dense_k = dense("K")?;
    let dense_v = dense("V")?;
    let body_bytes = (lead * hd * 2) as u64;
    let body_elems = nkv * rows * hd;
    let k_body = dense_k.slice_view(body_bytes, body_elems);
    let v_body = dense_v.slice_view(body_bytes, body_elems);

    s.barrier_between(&[src_k, src_v, src_norms], &[&dense_k, &dense_v]);
    let rings = (src_capacity, rows);
    let range = (rows, rows);
    let f16_row = hd * DType::F16.size_of();
    linearize_rows(
        s,
        reg,
        dev,
        (src_k, &k_body),
        nkv,
        rings,
        range,
        f16_row,
        layer_idx,
        "tensor K",
    )?;
    if src_v.dtype() == DType::F16 {
        linearize_rows(
            s,
            reg,
            dev,
            (src_v, &v_body),
            nkv,
            rings,
            range,
            f16_row,
            layer_idx,
            "tensor V",
        )?;
    } else {
        mlx_native::ops::tq_dequantize_kv::dispatch_tq_dequantize_hb_kv_seq_f16(
            s.encoder_mut(),
            reg,
            dev.metal_device(),
            src_v,
            src_norms,
            &v_body,
            nkv as u32,
            hd as u32,
            src_capacity as u32,
            0,
            rows as u32,
            layer.tq_scale_factor_d512,
            layer.tq_codebook_bits,
        )
        .map_err(|e| anyhow::anyhow!("live tensor V expand L{layer_idx}: {e}"))?;
    }
    Ok(TensorDenseKv {
        k: dense_k,
        v: dense_v,
        lead,
        rows,
    })
}

/// Padded key count of a tensor dispatch whose last key is dense row
/// `lead + kv_seq_len - 1`.
fn tensor_kv_pad(lead: usize, kv_seq_len: usize) -> usize {
    mlx_native::ops::flash_attn_prefill_tensor::kv_pad((lead + kv_seq_len) as u32) as usize
}

/// Bytes of the additive mask of one `len`-query dispatch over `kv_pad` keys.
fn tensor_mask_bytes(len: usize, kv_pad: usize) -> u64 {
    (len * kv_pad * DType::F16.size_of()) as u64
}

/// Tile classification of one `len`-query dispatch over `kv_pad` keys with
/// `bq` queries per threadgroup: the parameters [`encode_tensor_attend`]
/// allocates and dispatches with.
fn tensor_blk_params(
    len: usize,
    kv_pad: usize,
    bq: usize,
) -> mlx_native::ops::flash_attn_prefill_blk::BlkParams {
    mlx_native::ops::flash_attn_prefill_blk::BlkParams {
        seq_len_q: len as u32,
        seq_len_k: kv_pad as u32,
        bq: bq as u32,
        bk: mlx_native::ops::flash_attn_prefill_tensor::CACHE_ROWS_PER_STEP,
    }
}

/// Bytes one tensor dispatch allocates beside the dense K/V: its additive
/// mask and its `blk` buffer, the latter at mlx-native's own size
/// (`blk_buffer_byte_len`: one byte per tile, padded to 32).
fn tensor_dispatch_scratch_bytes(len: usize, kv_pad: usize, bq: usize) -> Result<u64> {
    let blk = mlx_native::ops::flash_attn_prefill_blk::blk_buffer_byte_len(&tensor_blk_params(
        len, kv_pad, bq,
    ))
    .map_err(|e| anyhow::anyhow!("live tensor blk size: {e}"))?;
    Ok(tensor_mask_bytes(len, kv_pad) + blk as u64)
}

/// Upper bound of the accounted global-layer scratch of one forward whose
/// deepest sequence attends `total_keys` keys on layers of `nkv` KV heads
/// and head dim `hd`, under `limits`: the larger of the budget and that
/// sequence's dense K/V of one layer, plus the mask and `blk` buffer of its
/// largest sub-chunk. Holds for any number of lanes and layers, because a
/// forward holds one sequence's dense K/V at a time and waits before
/// submitted scratch would pass the budget.
pub(crate) fn tensor_global_envelope_bytes(
    nkv: usize,
    hd: usize,
    total_keys: usize,
    limits: TensorScratchLimits,
) -> Result<u64> {
    let bq = mlx_native::ops::flash_attn_prefill_tensor::queries_per_threadgroup(hd as u32)
        .map_err(|e| anyhow::anyhow!("live tensor envelope: {e}"))? as usize;
    let kv_pad = tensor_kv_pad(0, total_keys);
    let rows = tensor_global_query_rows(kv_pad, bq, limits.mask_bytes);
    Ok(
        tensor_dense_kv_bytes(nkv, hd, total_keys, 0).max(limits.budget_bytes)
            + tensor_dispatch_scratch_bytes(rows, kv_pad, bq)?,
    )
}

/// One tensor-API dispatch: `len` queries of `q` against the first
/// `keys.kv_seq_len` rows of `dense`, written to `out`.
#[allow(clippy::too_many_arguments)]
fn encode_tensor_attend(
    s: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    dense: &TensorDenseKv,
    is_ring: bool,
    keys: ChunkKeys,
    len: usize,
    q: &MlxBuffer,
    out: &MlxBuffer,
) -> Result<()> {
    use mlx_native::ops::flash_attn_prefill_tensor as tensor;
    let (nh, nkv, hd, layer_idx) = (layer.nh, layer.nkv, layer.hd, layer.layer_idx);
    anyhow::ensure!(
        keys.kv_seq_len <= dense.rows,
        "live tensor L{layer_idx}: {} keys exceed the {} dense rows",
        keys.kv_seq_len,
        dense.rows
    );
    let kv_pad = tensor_kv_pad(dense.lead, keys.kv_seq_len);
    let mask = mlx_native::ops::flash_attn_prefill_mask::build_sdpa_mask_f16(
        dev,
        reg,
        s.encoder_mut(),
        &mlx_native::ops::flash_attn_prefill_mask::SdpaMaskParams {
            seq_len_q: len as u32,
            seq_len_k: kv_pad as u32,
            window_size: is_ring.then_some(layer.sliding_window),
            causal: true,
            q_abs_offset: (dense.lead + keys.query_offset) as u32,
        },
    )
    .map_err(|e| anyhow::anyhow!("live tensor mask L{layer_idx}: {e}"))?;
    // The mask is allocated inside its builder, so its write is undeclared:
    // fence before the blk classifier reads it (see `encode_tiled_chunk`).
    s.barrier();
    let bq = tensor::queries_per_threadgroup(hd as u32)
        .map_err(|e| anyhow::anyhow!("live tensor L{layer_idx}: {e}"))?;
    let blk_params = tensor_blk_params(len, kv_pad, bq as usize);
    let blk = mlx_native::ops::flash_attn_prefill_blk::alloc_blk_buffer(dev, &blk_params)
        .map_err(|e| anyhow::anyhow!("live tensor blk alloc L{layer_idx}: {e}"))?;
    s.barrier_between(&[&mask], &[&blk]);
    mlx_native::ops::flash_attn_prefill_blk::dispatch_flash_attn_prefill_blk_f16(
        s.encoder_mut(),
        dev,
        reg,
        &mask,
        &blk,
        &blk_params,
    )
    .map_err(|e| anyhow::anyhow!("live tensor blk L{layer_idx}: {e}"))?;
    s.barrier_between(&[q, &dense.k, &dense.v, &mask, &blk], &[out]);
    tensor::dispatch_flash_attn_prefill_tensor_f16(
        s.encoder_mut(),
        reg,
        dev,
        q,
        &dense.k,
        &dense.v,
        &mask,
        &blk,
        out,
        &tensor::FlashAttnPrefillTensorParams {
            rows: len as u32,
            n_heads: nh as u32,
            n_kv_heads: nkv as u32,
            head_dim: hd as u32,
            kv_pad: kv_pad as u32,
            q_row_stride: (nh * hd) as u32,
            q_head_stride: hd as u32,
            k_row_stride: hd as u32,
            k_head_stride: (dense.rows * hd) as u32,
            v_row_stride: hd as u32,
            v_head_stride: (dense.rows * hd) as u32,
            scale: 1.0,
        },
    )
    .map_err(|e| anyhow::anyhow!("live tensor FA L{layer_idx}: {e}"))?;
    Ok(())
}

/// Tensor-API flash attention for one sliding-layer chunk of one sequence:
/// its stage (at most one window of history plus the chunk) is compacted
/// and attended in one dispatch.
#[allow(clippy::too_many_arguments)]
fn encode_tensor_chunk(
    s: &mut GraphSession<'_>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    src: (&MlxBuffer, &MlxBuffer, &MlxBuffer),
    src_capacity: usize,
    is_ring: bool,
    keys: ChunkKeys,
    first_key_abs: usize,
    len: usize,
    q: &MlxBuffer,
    out: &MlxBuffer,
) -> Result<()> {
    let dense = build_tensor_dense_kv(
        s,
        reg,
        dev,
        layer,
        src,
        src_capacity,
        keys.kv_seq_len,
        first_key_abs,
    )?;
    encode_tensor_attend(s, reg, dev, layer, &dense, is_ring, keys, len, q, out)
}

/// Tensor-API flash attention of one sequence on a global layer, with
/// history-proportional scratch bounded by `scratch` (module doc). The
/// suffix is already appended, so one dense K/V pair over
/// `graft + start + rows` keys serves every sub-chunk; sub-chunk `c` attends
/// its first `graft + start + end_c` rows and masks the rest, which is what
/// a dispatch over exactly those rows computes.
fn encode_tensor_global_seq<'e>(
    s: &mut GraphSession<'e>,
    reg: &mut KernelRegistry,
    dev: &MlxDevice,
    layer: &LiveLayer<'_>,
    seq: &LiveSeq<'_>,
    scratch: &mut LiveScratch<'e>,
) -> Result<()> {
    use mlx_native::ops::flash_attn_prefill_tensor as tensor;
    let (kv, span) = (seq.kv, seq.span);
    let (nkv, hd, layer_idx) = (layer.nkv, layer.hd, layer.layer_idx);
    let row_elems = layer.nh * hd;
    let total_keys = layer.graft_len + span.start + span.rows;
    let bq = tensor::queries_per_threadgroup(hd as u32)
        .map_err(|e| anyhow::anyhow!("live tensor L{layer_idx}: {e}"))? as usize;
    let sub_rows =
        tensor_global_query_rows(tensor_kv_pad(0, total_keys), bq, scratch.limits.mask_bytes);

    let dense_bytes = tensor_dense_kv_bytes(nkv, hd, total_keys, 0);
    scratch.admit(s, dense_bytes, true)?;
    let dense = build_tensor_dense_kv(
        s,
        reg,
        dev,
        layer,
        (&kv.k, &kv.v_packed, &kv.v_norms),
        kv.capacity,
        total_keys,
        0,
    )?;
    let mut local = 0usize;
    while local < span.rows {
        let len = sub_rows.min(span.rows - local);
        let keys = chunk_keys(false, kv.capacity, layer.graft_len, span.start + local, len);
        // The dispatch's mask and `blk` buffer, both alive until its
        // command buffer finishes.
        scratch.admit(
            s,
            tensor_dispatch_scratch_bytes(len, tensor_kv_pad(dense.lead, keys.kv_seq_len), bq)?,
            false,
        )?;
        let stream_row = span.offset + local;
        let q = layer
            .q
            .slice_view((stream_row * row_elems * 4) as u64, len * row_elems);
        let out = layer
            .out
            .slice_view((stream_row * row_elems * 4) as u64, len * row_elems);
        encode_tensor_attend(s, reg, dev, layer, &dense, false, keys, len, &q, &out)?;
        local += len;
    }
    drop(dense);
    scratch.release(dense_bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::gemma4::GemmaLiveKernel::{TensorF16, TiledF16, VecHybrid};

    fn chunk_list(kernel: GemmaLiveKernel, is_ring: bool, start: usize, rows: usize) -> Vec<usize> {
        chunks(kernel, is_ring, start, rows)
            .map(|(_, len)| len)
            .collect()
    }

    fn warm(raw: &str) -> Resolved<GemmaLiveKernel> {
        parse_warm_tiled_switch(Some(raw)).unwrap()
    }

    #[test]
    fn warm_switch_defaults_to_the_release_gate_tensor_route() {
        for raw in [None, Some(""), Some("  ")] {
            assert_eq!(
                parse_warm_tiled_switch(raw),
                Ok(Resolved::default_value(TensorF16)),
                "{raw:?}"
            );
        }
        assert_eq!(default_warm_live_kernel(), TensorF16);
        for raw in ["0", "off", "false", "stock", "False", "STOCK", " 0 "] {
            assert_eq!(
                warm(raw),
                Resolved::environment(VecHybrid),
                "kill switch {raw:?}"
            );
        }
        for raw in ["1", "on", "TRUE"] {
            assert_eq!(warm(raw), Resolved::environment(TiledF16), "{raw:?}");
        }
        for raw in ["tensor", " Tensor "] {
            assert_eq!(warm(raw), Resolved::environment(TensorF16), "{raw:?}");
        }
        assert_eq!(warm("0").source.as_str(), "env");
    }

    /// A set value that is not a route refuses the load; it never resolves
    /// to the default.
    #[test]
    fn warm_switch_invalid_value_is_refused() {
        for raw in ["yes", "of", "no", "vec", "tiled", "2", "tensr"] {
            let error = parse_warm_tiled_switch(Some(raw)).unwrap_err();
            assert!(error.contains(WARM_TILED_ATTENTION_ENV), "{raw:?}: {error}");
        }
    }

    // ---- bounded global-layer scratch of the tensor route -----------------

    #[test]
    fn tensor_global_sub_chunks_are_whole_threadgroups_under_the_mask_limit() {
        let limit = TENSOR_GLOBAL_MASK_BYTES_MAX;
        // Shallow histories keep one dispatch per 4,096-row chunk.
        assert_eq!(tensor_global_query_rows(8_192, 16, limit), 4_096);
        assert_eq!(tensor_global_query_rows(1_344, 16, limit), 4_096);
        // Deep histories split; every sub-chunk is a multiple of the
        // kernel's queries per threadgroup and its mask fits the limit.
        for kv_pad in [16_384usize, 102_400, 131_072, 262_144, 1_048_576] {
            for bq in [16usize, 32] {
                let rows = tensor_global_query_rows(kv_pad, bq, limit);
                assert!(rows >= bq && rows % bq == 0, "{kv_pad} {bq}: {rows}");
                assert!(
                    tensor_mask_bytes(rows, kv_pad) <= limit || rows == bq,
                    "{kv_pad} {bq}: {rows}"
                );
            }
        }
        assert_eq!(tensor_global_query_rows(262_144, 16, limit), 128);
        // A limit below one threadgroup's mask still makes progress.
        assert_eq!(tensor_global_query_rows(262_144, 16, 1), 16);
        // No limit: the route as first shipped (one dispatch per chunk).
        assert_eq!(
            tensor_global_query_rows(262_144, 16, TensorScratchLimits::UNBOUNDED.mask_bytes),
            TILED_GLOBAL_QUERY_CHUNK
        );
    }

    /// The accounted scratch of a dispatch is its mask plus mlx-native's
    /// own `blk` size, and the envelope adds exactly that to the larger of
    /// the budget and one dense K/V pair.
    #[test]
    fn tensor_dispatch_scratch_counts_the_mask_and_the_blk_buffer_exactly() {
        use mlx_native::ops::flash_attn_prefill_blk::blk_buffer_byte_len;
        // 128 queries (8 threadgroups of 16) over 262,144 keys (4,096 key
        // tiles): a 64 MiB mask and 32 KiB of tile classification.
        assert_eq!(tensor_mask_bytes(128, 262_144), 64 << 20);
        assert_eq!(
            tensor_dispatch_scratch_bytes(128, 262_144, 16).unwrap(),
            (64 << 20) + (32 << 10)
        );
        for (len, kv_pad, bq) in [
            (1, 64, 16),
            (150, 384, 16),
            (4_096, 8_192, 16),
            (1_024, 2_048, 32),
        ] {
            let blk = blk_buffer_byte_len(&tensor_blk_params(len, kv_pad, bq)).unwrap() as u64;
            assert!(blk >= (len.div_ceil(bq) * (kv_pad / 64)) as u64 && blk % 32 == 0);
            assert_eq!(
                tensor_dispatch_scratch_bytes(len, kv_pad, bq).unwrap(),
                tensor_mask_bytes(len, kv_pad) + blk,
                "{len} {kv_pad} {bq}"
            );
        }
        let serving = TensorScratchLimits::SERVING;
        // 262,144 keys: 1 GiB of dense K/V, one 64 MiB mask, 32 KiB of blk.
        assert_eq!(
            tensor_global_envelope_bytes(2, 512, 262_144, serving).unwrap(),
            (1 << 30) + (64 << 20) + (32 << 10)
        );
        // Shallow: the budget dominates; a 4,096-row dispatch over 2,048
        // keys has a 16 MiB mask and 256 x 32 tiles.
        assert_eq!(
            tensor_global_envelope_bytes(2, 512, 2_048, serving).unwrap(),
            (512 << 20) + (16 << 20) + 256 * 32
        );
    }

    #[test]
    fn tensor_dense_kv_bytes_follow_the_documented_formula() {
        // Gemma 4 26B-A4B global layer: 2 KV heads, head dim 512, F16 K and
        // V: 4 KiB per key (plus the padding rows of one tile).
        let (nkv, hd) = (2usize, 512usize);
        assert_eq!(tensor_dense_kv_bytes(nkv, hd, 262_144, 0), 262_144 * 4_096);
        assert_eq!(tensor_dense_kv_bytes(nkv, hd, 102_400, 0), 102_400 * 4_096);
        // 100 keys pad to 128 rows: 28 padding rows of one head each.
        assert_eq!(
            tensor_dense_kv_bytes(nkv, hd, 100, 0),
            2 * 2 * (100 * nkv * hd + 28 * hd) as u64
        );
    }

    /// One synthetic global layer (2 KV heads, 4 heads, head dim 512, the
    /// production TQ-HB V or F16 V) with per-lane append-only caches.
    struct GlobalFixture {
        ctx: crate::serve::gpu::GpuContext,
        v_dtype: DType,
    }

    const FX_NH: usize = 4;
    const FX_NKV: usize = 2;
    const FX_HD: usize = 512;
    const FX_CAP: usize = 1_024;

    fn fx_values(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    impl GlobalFixture {
        fn new(v_dtype: DType) -> Self {
            Self {
                ctx: crate::serve::gpu::GpuContext::new().expect("Metal device"),
                v_dtype,
            }
        }

        fn kv(&self) -> HybridKvBuffers {
            let dev = self.ctx.device();
            let elems = FX_NKV * FX_CAP * FX_HD;
            let zeroed = |bytes: usize, dtype: DType, shape: Vec<usize>| {
                dev.alloc_buffer(bytes, dtype, shape)
                    .expect("fixture cache")
            };
            HybridKvBuffers {
                k: zeroed(elems * 2, DType::F16, vec![FX_NKV, FX_CAP, FX_HD]),
                v_packed: zeroed(
                    elems * self.v_dtype.size_of(),
                    self.v_dtype,
                    vec![FX_NKV, FX_CAP, FX_HD],
                ),
                v_norms: zeroed(FX_NKV * FX_CAP * 2 * 4, DType::F32, vec![FX_NKV, FX_CAP, 2]),
                capacity: FX_CAP,
                is_sliding: false,
                norms_per_pos: 2,
                bf16_xlen_k: None,
                bf16_xlen_v: None,
            }
        }

        fn stream(&self, seed: u64, rows: usize, heads: usize) -> MlxBuffer {
            let elems = rows * heads * FX_HD;
            let mut buffer = self
                .ctx
                .device()
                .alloc_buffer(elems * 4, DType::F32, vec![rows, heads, FX_HD])
                .expect("fixture stream");
            buffer
                .as_mut_slice::<f32>()
                .unwrap()
                .copy_from_slice(&fx_values(seed, elems));
            buffer
        }

        /// One live forward of `lanes` (`(cache, start, rows)` each) on the
        /// tensor route under `limits`; returns the output bits per lane and
        /// the forward's scratch statistics.
        fn forward(
            &mut self,
            seed: u64,
            lanes: &[(&HybridKvBuffers, usize, usize)],
            limits: TensorScratchLimits,
        ) -> (Vec<Vec<u32>>, LiveScratchStats) {
            let total: usize = lanes.iter().map(|lane| lane.2).sum();
            let q = self.stream(seed, total, FX_NH);
            let k = self.stream(seed ^ 0x5151, total, FX_NKV);
            let v = self.stream(seed ^ 0xa3a3, total, FX_NKV);
            let out = self.stream(0, total, FX_NH);
            let positions = self
                .ctx
                .device()
                .alloc_buffer(total * 4, DType::U32, vec![total])
                .unwrap();
            let layer = LiveLayer {
                layer_idx: 5,
                nh: FX_NH,
                nkv: FX_NKV,
                hd: FX_HD,
                sliding_window: 1_024,
                graft_len: 0,
                tq_scale_factor_d512: 1.0,
                tq_codebook_bits: 8,
                q: &q,
                k: &k,
                v: &v,
                out: &out,
                positions: &positions,
                vec: None,
                total_rows: total,
            };
            let mut offset = 0;
            let seqs: Vec<LiveSeq<'_>> = lanes
                .iter()
                .map(|&(kv, start, rows)| {
                    let span = LiveSpan {
                        offset,
                        rows,
                        start,
                    };
                    offset += rows;
                    LiveSeq { kv, span }
                })
                .collect();
            set_test_tensor_scratch_limits(Some(limits));
            {
                let (exec, reg) = self.ctx.split();
                let dev = exec.device();
                let mut scratch = LiveScratch::new(exec);
                let mut session = exec.begin().unwrap();
                encode_live_layer_attention(
                    &mut session,
                    reg,
                    dev,
                    &layer,
                    LiveRoute::resolve(TensorF16),
                    seqs,
                    &mut scratch,
                )
                .unwrap();
                session.finish().unwrap();
            }
            set_test_tensor_scratch_limits(None);
            let bits: Vec<u32> = out
                .as_slice::<f32>()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect();
            let row = FX_NH * FX_HD;
            let mut offset = 0;
            let per_lane = lanes
                .iter()
                .map(|&(_, _, rows)| {
                    let lane = bits[offset * row..(offset + rows) * row].to_vec();
                    offset += rows;
                    lane
                })
                .collect();
            (per_lane, last_forward_scratch_stats())
        }
    }

    fn cache_bytes(kv: &HybridKvBuffers) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let bytes = |buffer: &MlxBuffer| buffer.as_slice::<u8>().unwrap().to_vec();
        (bytes(&kv.k), bytes(&kv.v_packed), bytes(&kv.v_norms))
    }

    /// Bounding the global-layer scratch changes no bit. On each V storage:
    /// a history is appended, then a suffix resumes on it, once as the
    /// route first shipped (one dispatch, no waits) and once with a mask
    /// limit of two threadgroups and a one-byte budget (a sub-chunk per 32
    /// queries and a GPU wait before every admission). Outputs and caches
    /// are equal, and a two-lane cohort under the tight limits gives each
    /// lane the bits it has alone.
    #[test]
    fn bounded_tensor_global_scratch_is_bitwise_equal_to_the_unsplit_route() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        const HISTORY: [usize; 2] = [200, 77];
        const SUFFIX: [usize; 2] = [150, 96];
        for v_dtype in [DType::U8, DType::F16] {
            let mut fx = GlobalFixture::new(v_dtype);
            // Two threadgroups (16 queries each at head dim 512) of the
            // deepest dispatch: 350 keys pad to 384.
            let tight = TensorScratchLimits {
                mask_bytes: tensor_mask_bytes(32, 384),
                budget_bytes: 1,
            };
            let mut alone = Vec::new();
            for limits in [TensorScratchLimits::UNBOUNDED, tight] {
                let mut arm = Vec::new();
                for lane in 0..2 {
                    let kv = fx.kv();
                    let (_, built) =
                        fx.forward(11 + lane as u64, &[(&kv, 0, HISTORY[lane])], limits);
                    let (out, resumed) = fx.forward(
                        23 + lane as u64,
                        &[(&kv, HISTORY[lane], SUFFIX[lane])],
                        limits,
                    );
                    if limits == TensorScratchLimits::UNBOUNDED {
                        assert_eq!((built.waits, resumed.waits), (0, 0), "{v_dtype:?}");
                    } else {
                        // Lane 0: 350 keys pad to 384, so 32 queries per
                        // sub-chunk and five masks for 150 rows. Lane 1: 173
                        // keys pad to 192, 64 queries per sub-chunk, two
                        // masks. Every mask after the first waits.
                        assert_eq!(resumed.waits, [4, 1][lane], "{v_dtype:?} lane {lane}");
                    }
                    assert!(resumed.peak_bytes > 0);
                    arm.push((out.into_iter().next().unwrap(), cache_bytes(&kv)));
                }
                alone.push(arm);
            }
            assert!(
                alone[0] == alone[1],
                "{v_dtype:?}: bounded scratch changed an output or cache bit"
            );
            assert!(
                alone[0][0].0.iter().any(|&bits| bits != 0),
                "{v_dtype:?}: empty output"
            );

            // The same two suffixes as one cohort (one concatenated stream).
            // A lane's stream rows differ from its solo forward's, so compare
            // the cohort under both limits, and lane 0 (whose rows lead the
            // stream) against its solo bits.
            let cohort: Vec<_> = [TensorScratchLimits::UNBOUNDED, tight]
                .into_iter()
                .map(|limits| {
                    let caches = [fx.kv(), fx.kv()];
                    for lane in 0..2 {
                        fx.forward(
                            11 + lane as u64,
                            &[(&caches[lane], 0, HISTORY[lane])],
                            limits,
                        );
                    }
                    let lanes: Vec<_> = (0..2)
                        .map(|lane| (&caches[lane], HISTORY[lane], SUFFIX[lane]))
                        .collect();
                    let (out, stats) = fx.forward(23, &lanes, limits);
                    (out, cache_bytes(&caches[0]), stats)
                })
                .collect();
            assert!(
                cohort[0].0 == cohort[1].0,
                "{v_dtype:?}: cohort bits differ"
            );
            assert!(
                cohort[0].1 == cohort[1].1,
                "{v_dtype:?}: cohort cache differs"
            );
            assert_eq!(cohort[0].2.waits, 0);
            assert!(cohort[1].2.waits > 0);
            assert!(
                cohort[1].0[0] == alone[0][0].0,
                "{v_dtype:?}: lane 0 in a cohort differs from lane 0 alone"
            );
        }
    }

    /// The accounting holds a sequence's dense K/V while its sub-chunks run,
    /// counts dropped scratch as submitted until a wait, and waits only when
    /// the budget would be exceeded and something can be released.
    #[test]
    fn live_scratch_waits_only_to_release_submitted_scratch() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let ctx = crate::serve::gpu::GpuContext::new().expect("Metal device");
        let exec = &ctx.executor;
        set_test_tensor_scratch_limits(Some(TensorScratchLimits {
            mask_bytes: 64,
            budget_bytes: 1_000,
        }));
        let mut scratch = LiveScratch::new(exec);
        set_test_tensor_scratch_limits(None);
        let mut session = exec.begin().unwrap();
        // One oversized pair is admitted without a wait: nothing to release.
        scratch.admit(&mut session, 5_000, true).unwrap();
        assert_eq!(
            (scratch.stats.waits, scratch.held, scratch.submitted),
            (0, 5_000, 0)
        );
        // Its first mask too; the second finds a submitted mask and waits.
        scratch.admit(&mut session, 300, false).unwrap();
        assert_eq!((scratch.stats.waits, scratch.submitted), (0, 300));
        scratch.admit(&mut session, 300, false).unwrap();
        assert_eq!((scratch.stats.waits, scratch.submitted), (1, 300));
        // Dropping the pair moves it to submitted; the next layer's pair
        // waits for it.
        scratch.release(5_000);
        assert_eq!((scratch.held, scratch.submitted), (0, 5_300));
        scratch.admit(&mut session, 5_000, true).unwrap();
        assert_eq!(
            (scratch.stats.waits, scratch.held, scratch.submitted),
            (2, 5_000, 0)
        );
        // Peak: the pair plus one mask; the waits keep it from adding up.
        assert_eq!(scratch.stats.peak_bytes, 5_300);
        // A recorded session cannot be split: account, never wait.
        scratch.set_can_wait(false);
        scratch.release(5_000);
        scratch.admit(&mut session, 5_000, true).unwrap();
        assert_eq!(scratch.stats.waits, 2);
        // A layer boundary that waited releases everything submitted.
        scratch.layer_finished();
        assert_eq!(scratch.submitted, 0);
        session.finish().unwrap();
    }

    #[test]
    fn warm_default_falls_back_to_vec_hybrid_without_the_tensor_api() {
        let unavailable = || Some("no tensor API".to_string());
        // Unset: the default falls back and reports why.
        let (kernel, reason) =
            admit_warm_kernel(Resolved::default_value(TensorF16), unavailable).unwrap();
        assert_eq!(kernel, Resolved::fallback(VecHybrid));
        assert_eq!(reason.as_deref(), Some("no tensor API"));
        // Unset on a capable device: the default serves.
        let (kernel, reason) =
            admit_warm_kernel(Resolved::default_value(TensorF16), || None).unwrap();
        assert_eq!(kernel, Resolved::default_value(TensorF16));
        assert_eq!(reason, None);
        // Explicit `tensor`: kept on a capable device, refused at load (not
        // at the first warm resume) on a host without the tensor API, with a
        // message naming the switch, the reason and the way out.
        let (kernel, reason) =
            admit_warm_kernel(Resolved::environment(TensorF16), || None).unwrap();
        assert_eq!(kernel, Resolved::environment(TensorF16));
        assert_eq!(reason, None);
        let refusal = admit_warm_kernel(Resolved::environment(TensorF16), unavailable).unwrap_err();
        for needle in [
            WARM_TILED_ATTENTION_ENV,
            "no tensor API",
            "Unset it",
            "set 0",
        ] {
            assert!(refusal.contains(needle), "{refusal}");
        }
        // The other explicit kernels need no tensor API and are not probed.
        for explicit in [TiledF16, VecHybrid] {
            let (kernel, reason) = admit_warm_kernel(Resolved::environment(explicit), || {
                panic!("a kernel without tensor pipelines must not be probed")
            })
            .unwrap();
            assert_eq!(kernel, Resolved::environment(explicit));
            assert_eq!(reason, None);
        }
    }

    /// The real probe on this device, through a production context: it
    /// agrees with mlx-native's own availability answer for both head dims,
    /// a capable device builds both pipelines (their compiled entries exist
    /// afterwards) and resolves the tensor default, and an incapable one
    /// reports the reason and falls back. Either way the frozen routing
    /// policy is untouched.
    #[test]
    fn warm_default_probe_builds_both_tensor_pipelines_or_reports_why() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        use mlx_native::ops::flash_attn_prefill_tensor as tensor;
        let mut ctx = crate::serve::gpu::GpuContext::new().expect("Metal device");
        let policy = *ctx.ggml_routing_policy();
        let (exec, registry) = ctx.split();
        let device = exec.device();
        let native: Vec<bool> = [256, 512]
            .into_iter()
            .map(|hd| tensor::flash_attn_prefill_tensor_available(registry, device, hd).unwrap())
            .collect();
        let reason = tensor_route_unavailable(registry, device);
        assert_eq!(
            reason.is_none(),
            native == [true, true],
            "{reason:?} {native:?}"
        );
        let (resolved, why) =
            admit_warm_kernel(Resolved::default_value(TensorF16), || reason.clone()).unwrap();
        if native == [true, true] {
            assert_eq!(resolved, Resolved::default_value(TensorF16));
            assert_eq!(why, None);
            for name in [tensor::K_TENSOR_F16_D256, tensor::K_TENSOR_F16_D512] {
                assert!(
                    registry.get_pipeline(name, device.metal_device()).is_ok(),
                    "{name} must build after a successful probe"
                );
            }
        } else {
            let reason = reason.expect("an unavailable route names its reason");
            assert!(reason.contains("Metal tensor API"), "{reason}");
            assert_eq!(resolved, Resolved::fallback(VecHybrid));
            assert_eq!(why.as_deref(), Some(reason.as_str()));
            eprintln!("RECORDED SKIP: no Metal tensor API on this device ({reason})");
        }
        assert_eq!(registry.ggml_routing_policy(), Some(&policy));
    }

    #[test]
    fn warm_route_follows_its_switch_and_keeps_tiny_suffixes_on_vec_hybrid() {
        let tiled = LiveRoute::resolve(TiledF16);
        assert_eq!(tiled.seq_kernel(1), VecHybrid);
        assert_eq!(tiled.seq_kernel(31), VecHybrid);
        assert_eq!(tiled.seq_kernel(32), TiledF16);
        assert_eq!(tiled.seq_kernel(385), TiledF16);
        let tensor = LiveRoute::resolve(TensorF16);
        assert_eq!(tensor.seq_kernel(1), VecHybrid);
        assert_eq!(tensor.seq_kernel(31), VecHybrid);
        assert_eq!(tensor.seq_kernel(32), TensorF16);
        assert_eq!(tensor.seq_kernel(385), TensorF16);
        let vec = LiveRoute::resolve(VecHybrid);
        assert_eq!(vec.seq_kernel(385), VecHybrid);
    }

    #[test]
    fn vec_scratch_is_needed_only_when_some_sequence_takes_vec_hybrid() {
        let span = |rows| LiveSpan {
            offset: 0,
            rows,
            start: 465,
        };
        let tiled = LiveRoute::resolve(TiledF16);
        assert!(!tiled.uses_vec(&[span(56), span(385)]));
        assert!(tiled.uses_vec(&[span(56), span(5)]));
        assert!(LiveRoute::resolve(VecHybrid).uses_vec(&[span(56)]));
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

    #[test]
    fn tensor_route_takes_the_tiled_chunks_and_keeps_tiny_suffixes_on_vec_hybrid() {
        let route = LiveRoute::resolve(TensorF16);
        assert_eq!(route.seq_kernel(31), VecHybrid);
        assert_eq!(route.seq_kernel(32), TensorF16);
        assert_eq!(route.label(), "tensor-f16 (vec-hybrid below 32 rows)");
        for (is_ring, start, rows) in
            [(true, 0, 2_500), (false, 7, 9_000), (true, 465, 385)]
        {
            assert_eq!(
                chunk_list(TensorF16, is_ring, start, rows),
                chunk_list(TiledF16, is_ring, start, rows)
            );
        }
        assert_eq!(
            stage_query_rows(TensorF16, 2_000),
            stage_query_rows(TiledF16, 2_000)
        );
    }

    #[test]
    fn gemma_live_kernel_wire_names_are_the_variant_names() {
        // `/hf2q/v1/runtime` and the Q5 readiness check use the variant names.
        for kernel in [TiledF16, VecHybrid, TensorF16] {
            assert_eq!(kernel.wire_name(), format!("{kernel:?}"));
        }
    }
}
