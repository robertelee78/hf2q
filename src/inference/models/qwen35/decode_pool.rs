//! Thread-local arena buffer pool for the qwen35 decode hot path.
//!
//! # Purpose
//!
//! ADR-012 §Optimize / Task #15 — close the MoE dwq46 0.90× decode parity gap
//! vs the peer.  Diagnostic localization (`HF2Q_DECODE_PROFILE=1`) showed
//! the gap is fixed-cost-per-layer not per-byte, and the per-layer cost is
//! dominated by ~1750 `MlxDevice::alloc_buffer()` calls per decode token
//! across the three dispatch helpers (`gpu_delta_net::build_delta_net_layer`,
//! `gpu_ffn::build_moe_ffn_layer_gpu_q`, `gpu_full_attn::build_gated_attn_layer`).
//! Each direct alloc hits Metal's `newBuffer` allocator (5-30 µs each); a
//! per-token arena pool reuses the underlying `metal::Buffer` objects across
//! token boundaries so steady-state allocation cost amortizes to near zero.
//!
//! # Why a thread-local
//!
//! The dispatch helpers are deeply nested (`build_delta_net_layer` calls
//! `apply_pre_norm` calls `dispatch_rms_norm` etc.).  Threading a
//! `&mut MlxBufferPool` parameter through every helper would touch ~62
//! call sites + every signature.  A thread-local pool gives a clean
//! single-line replacement at each `device.alloc_buffer(...)` call site
//! (`pooled_alloc_buffer(device, ...)`) with zero signature thrash.
//!
//! Decode is single-threaded per `feedback_oom_prevention` (one model-loading
//! inference at a time on the M5 Max), so a thread-local is sufficient.
//!
//! # Lifecycle
//!
//! * [`reset_decode_pool`] is called at the top of each
//!   `Qwen35Model::forward_gpu_greedy` call (per token).
//! * Layer dispatches inside the forward call use [`pooled_alloc_buffer`]
//!   in place of `device.alloc_buffer`.
//! * Locally-bound `MlxBuffer` values fall out of scope at function exit;
//!   the pool's ARC clones keep the underlying Metal storage alive.
//! * The next token's `reset_decode_pool` moves all in-use clones back to
//!   the free list, ready for reuse by subsequent allocations.
//!
//! # Caller contract
//!
//! No `MlxBuffer` returned from `pooled_alloc_buffer` may outlive a
//! [`reset_decode_pool`] call from the same thread.  In Rust's ownership
//! model, locally-bound buffers fall out of scope at the end of their
//! lexical block, making the per-decode-token pattern safe by construction
//! provided allocations stay inside `forward_gpu_greedy`'s call tree.

pub mod profile;

use std::cell::RefCell;

use mlx_native::ops::quantized_matmul_id_ggml::IdMmScratch;
use mlx_native::{DType, MlxBuffer, MlxBufferPool, MlxDevice};

thread_local! {
    /// Per-thread arena pool.  Initialized lazily by the first
    /// [`pooled_alloc_buffer`] call; reset between decode tokens.
    static DECODE_POOL: RefCell<MlxBufferPool> = RefCell::new(MlxBufferPool::new());

    /// Per-thread cached `IdMmScratch` instances for the three MoE FFN
    /// `quantized_matmul_id_ggml` call slots (W-5b.24).  Each FFN call
    /// site (gate, up, down) gets its own scratch so the three calls
    /// inside the gate+up Phase C concurrent block do NOT race on
    /// htpe/hids writes — the down call lives in Phase E behind a
    /// barrier so any of the three could share with down, but keeping
    /// three distinct scratches mirrors the call structure 1:1 and is
    /// trivially correct.
    ///
    /// Grown on demand via `with_id_mm_scratch_*` helpers.  The first
    /// FFN call of a prefill allocates 6 device buffers (2 per scratch
    /// × 3 scratches); every subsequent FFN call (47 layers × 3 calls
    /// = 141 calls) reuses them.  Net: 286 of 288 per-prefill device
    /// allocs eliminated, matching the W-5b.23 audit's recovery target.
    static MM_ID_SCRATCH_GATE: RefCell<Option<IdMmScratch>> = const { RefCell::new(None) };
    static MM_ID_SCRATCH_UP: RefCell<Option<IdMmScratch>> = const { RefCell::new(None) };
    static MM_ID_SCRATCH_DOWN: RefCell<Option<IdMmScratch>> = const { RefCell::new(None) };
}

/// Owning-worker only. The serving publisher verifies its worker registration first.
/// This does not reset, allocate Metal buffers, wait, or mutate profile health.
pub(crate) fn worker_memory_snapshot() -> Result<(&'static str, mlx_native::PoolMemorySnapshot), String> {
    if let Some(result) = profile::memory_snapshot_if_selected() {
        return result.map(|snapshot| ("growth_limited", snapshot));
    }
    DECODE_POOL.try_with(|cell| {
        let pool = cell.try_borrow().map_err(|_| "legacy pool borrow busy".to_string())?;
        pool.memory_snapshot().map(|snapshot| ("legacy", snapshot)).map_err(|error| error.to_string())
    }).map_err(|_| "legacy worker TLS unavailable".to_string()).and_then(|r| r)
}

/// Allocate from the thread-local decode pool.
///
/// API-compatible with `MlxDevice::alloc_buffer` so call sites in the
/// dispatch helpers can be edited mechanically.
#[inline]
#[cfg_attr(test, track_caller)]
pub fn pooled_alloc_buffer(
    device: &MlxDevice,
    byte_len: usize,
    dtype: DType,
    shape: Vec<usize>,
) -> std::result::Result<MlxBuffer, mlx_native::MlxError> {
    if let Some(result) = profile::alloc(device, byte_len, dtype, &shape) { return result; }
    // The pool hands out the power-of-two bucket the request rounds up to.
    let bucket = byte_len.max(1).next_power_of_two();
    ISSUED_BYTES.with(|issued| issued.set(issued.get() + bucket));
    ISSUED_TOTAL.with(|issued| issued.set(issued.get() + bucket as u64));
    #[cfg(test)]
    {
        let site = std::panic::Location::caller();
        SCRATCH_TRACE.with(|trace| {
            if let Some(trace) = trace.borrow_mut().as_mut() {
                trace.push((site.file(), site.line(), byte_len));
            }
        });
    }
    if let Some(caps) = CAPS.with(std::cell::Cell::get) {
        #[cfg(test)]
        CAPPED_FREE_PEAK.with(|peak| {
            peak.set(peak.get().max(DECODE_POOL.with(|cell| cell.borrow().free_bytes())))
        });
        // Before the allocation: what the pool held free at its last reset
        // plus what it has handed out since bounds what it holds now (a
        // reused buffer counts as if it were new). A request that could
        // take the pool over its cap is served outside it.
        let held = FREE_AT_RESET.with(std::cell::Cell::get)
            + POOLED_SINCE_RESET.with(std::cell::Cell::get);
        if held.saturating_add(bucket) > caps.pool {
            let buffer = device.alloc_buffer(byte_len, dtype, shape)?;
            OVERFLOW.with(|overflow| overflow.borrow_mut().push(buffer.clone()));
            return Ok(buffer);
        }
    }
    let buffer = DECODE_POOL.with(|cell| cell.borrow_mut().alloc(device, byte_len, dtype, shape))?;
    POOLED_SINCE_RESET.with(|pooled| pooled.set(pooled.get() + bucket));
    Ok(buffer)
}

/// The caps a forward puts on this thread's pool ([`cap_pool`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoolCaps {
    /// What the pool may hold at any time: checked before every allocation.
    pub(crate) pool: usize,
    /// Free bytes above which a reset releases the pool's free buffers.
    pub(crate) retained: usize,
}

thread_local! {
    /// Bucket bytes handed out since the last reset, by the pool or outside
    /// it.
    static ISSUED_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Bucket bytes handed out by this thread, ever (never reset).
    static ISSUED_TOTAL: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// The largest scratch one layer has requested since the last
    /// [`take_layer_scratch_peak`], and whether a layer over its bound was
    /// reported already.
    static LAYER_SCRATCH_PEAK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static LAYER_SCRATCH_REPORTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The pool's part of [`ISSUED_BYTES`].
    static POOLED_SINCE_RESET: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Free bytes the pool held after its last reset or, later, when caps
    /// were put on it.
    static FREE_AT_RESET: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The caps in force, and whether they end with the next reset.
    static CAPS: std::cell::Cell<Option<PoolCaps>> = const { std::cell::Cell::new(None) };
    static CAPS_END_AT_RESET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Buffers served outside the pool since the last reset. They are held
    /// like pooled ones, so a command buffer still in flight keeps its
    /// operands, and are released (not recycled) at the reset.
    static OVERFLOW: RefCell<Vec<MlxBuffer>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
thread_local! {
    /// Test trace of the scratch requests: (file, line, bytes requested).
    static SCRATCH_TRACE: RefCell<Option<Vec<(&'static str, u32, usize)>>> =
        const { RefCell::new(None) };
}

/// Start (`true`) or stop a trace of this thread's scratch requests and
/// return what was traced so far.
#[cfg(test)]
pub(crate) fn trace_scratch(on: bool) -> Vec<(&'static str, u32, usize)> {
    SCRATCH_TRACE.with(|trace| trace.replace(on.then(Vec::new)).unwrap_or_default())
}

/// Bucket bytes of scratch this thread has out since its pool's last reset:
/// handed out by the pool or served outside it.
pub(crate) fn issued_bytes() -> usize {
    ISSUED_BYTES.with(std::cell::Cell::get)
}

/// Bucket bytes of scratch this thread has requested so far (monotone).
pub(crate) fn issued_total() -> u64 {
    ISSUED_TOTAL.with(std::cell::Cell::get)
}

/// Record that one layer of a forward requested `bytes` of scratch, against
/// the bound derived for the forward's shape (`None`: no bound applies). A
/// layer over its bound ran on a route the derivation does not cover: it is
/// reported once per thread, the forward is not affected.
pub(crate) fn note_layer_scratch(bytes: u64, bound: Option<u64>) {
    LAYER_SCRATCH_PEAK.with(|peak| peak.set(peak.get().max(bytes)));
    if bound.is_some_and(|bound| bytes > bound) && !LAYER_SCRATCH_REPORTED.with(|cell| cell.replace(true)) {
        tracing::warn!(
            layer_scratch_bytes = bytes,
            bound_bytes = bound.unwrap_or_default(),
            "Qwen prefill: a layer requested more scratch than the bound derived for its shape"
        );
    }
}

/// The largest scratch one layer has requested since the last call.
#[cfg(test)]
pub(crate) fn take_layer_scratch_peak() -> u64 {
    LAYER_SCRATCH_PEAK.with(|peak| peak.replace(0))
}

/// Put `caps` on this thread's pool from now on (`None`: no caps, the
/// default).
///
/// The pool reuses a free buffer only in the exact bucket a request rounds
/// up to and never shrinks on its own, so forwards of varying sizes, and
/// forwards that skip host waits, grow it without bound. Under caps:
/// - before every allocation, a request that could take the pool over
///   `caps.pool` is served by a plain device buffer instead, held until the
///   next reset and released there, so the pool never holds more than
///   `caps.pool` bytes;
/// - a reset releases the pool's free buffers when they exceed
///   `caps.retained`. A reset follows a drained queue and the last use of
///   every issued buffer, so the free buffers have no user.
/// Released and outside buffers are allocated again when needed: that costs
/// allocation time, never memory.
///
/// Putting caps on a pool that earlier, uncapped forwards left warm trims
/// it at once: its free buffers are released when they exceed
/// `caps.retained`, before the first capped allocation, so the hard cap
/// holds from the first window on. The free buffers are those the
/// forward's own initial reset took back (a reset follows a drained queue
/// and the last use of every issued buffer); what the forward has issued
/// since (its embedding) is in use and is not touched. The release takes
/// effect immediately (the pool removes them from the residency set and
/// commits the set); no command buffer is needed for it.
pub(crate) fn cap_pool(caps: Option<PoolCaps>) {
    CAPS.with(|cell| cell.set(caps));
    CAPS_END_AT_RESET.with(|cell| cell.set(false));
    if let Some(caps) = caps {
        if !default_pool() {
            return;
        }
        DECODE_POOL.with(|cell| {
            let mut pool = cell.borrow_mut();
            if pool.free_bytes() > caps.retained {
                #[cfg(test)]
                let before = (pool.free_bytes(), trim_probe());
                pool.clear();
                #[cfg(test)]
                if let (freed, Some(before)) = before {
                    let after = trim_probe().unwrap_or(before);
                    TRIM_SAMPLE.with(|sample| sample.set(Some((freed, before, after))));
                }
            }
            // What the pool holds now is this plus what is in use, which is
            // exactly what it has handed out since its reset.
            FREE_AT_RESET.with(|free| free.set(pool.free_bytes()));
            #[cfg(test)]
            CAPPED_FREE_PEAK.with(|peak| peak.set(peak.get().max(pool.free_bytes())));
        });
    }
}

#[cfg(test)]
thread_local! {
    /// The most free bytes the pool held at a capped allocation or at a cap
    /// activation since the last [`take_capped_free_peak`].
    static CAPPED_FREE_PEAK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// A test's reading of the device's allocated bytes.
    static TRIM_PROBE: std::cell::RefCell<Option<Box<dyn Fn() -> usize>>> =
        const { std::cell::RefCell::new(None) };
    /// `(free bytes released, probe before, probe after)` of the last trim
    /// at a cap activation with a probe set.
    static TRIM_SAMPLE: std::cell::Cell<Option<(usize, usize, usize)>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn trim_probe() -> Option<usize> {
    TRIM_PROBE.with(|probe| probe.borrow().as_ref().map(|read| read()))
}

/// Sample `probe` immediately before and after the release of a warm pool
/// at a cap activation on this thread.
#[cfg(test)]
pub(crate) fn set_trim_probe(probe: Option<Box<dyn Fn() -> usize>>) {
    TRIM_PROBE.with(|cell| *cell.borrow_mut() = probe);
}

/// `(free bytes released, probe before, probe after)` of the last trim at a
/// cap activation since the last call.
#[cfg(test)]
pub(crate) fn take_trim_sample() -> Option<(usize, usize, usize)> {
    TRIM_SAMPLE.with(std::cell::Cell::take)
}

/// The most free bytes this thread's pool held while caps were in force
/// (sampled when the caps were put on and at every capped allocation) since
/// the last call.
#[cfg(test)]
pub(crate) fn take_capped_free_peak() -> usize {
    CAPPED_FREE_PEAK.with(|peak| peak.replace(0))
}

/// The caps of [`cap_pool`] apply through one more reset and then end. A
/// forward calls this when it returns: the reset that takes its last
/// buffers back is the next forward's or the next decode step's.
pub(crate) fn end_caps_at_next_reset() {
    CAPS_END_AT_RESET.with(|cell| cell.set(CAPS.with(std::cell::Cell::get).is_some()));
}

/// Release the buffers served outside the pool. The caller has drained the
/// device queue: nothing in flight uses them.
pub(crate) fn release_overflow() {
    OVERFLOW.with(|overflow| overflow.borrow_mut().clear());
}

/// Whether this thread's pool is the default one (not the growth-limited
/// investigation profile, which has its own quota and no release).
pub(crate) fn default_pool() -> bool {
    !profile::selected()
}

/// Free bytes this thread's default pool holds.
#[cfg(test)]
pub(crate) fn free_bytes() -> usize {
    DECODE_POOL.with(|cell| cell.borrow().free_bytes())
}

thread_local! {
    /// Whether multi-row layer outputs come from this pool (see
    /// [`StepPooledOutputs`]).
    static STEP_POOLED_OUTPUTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, the multi-row (`seq > 1`) layer outputs that otherwise use
/// `MlxDevice::alloc_buffer` (dense FFN `down_out` / residual sum, attention
/// projection and Q/gate outputs) come from this pool instead.
///
/// Those outputs are fresh device allocations because a prefill chunk resets
/// this pool between layers while a layer's output becomes the next layer's
/// input. A joint decode step resets the pool only at its start, after the
/// previous step drained, so inside its layer loop pooled outputs live long
/// enough. Every such output is fully written by its producing kernel, so the
/// arithmetic is unchanged (Class A); only the per-step Metal allocation,
/// zero-fill and residency-set churn go away.
pub(crate) struct StepPooledOutputs {
    previous: bool,
}

impl StepPooledOutputs {
    /// Open the scope when `enabled`; a disabled scope changes nothing.
    pub(crate) fn enter(enabled: bool) -> Self {
        Self {
            previous: STEP_POOLED_OUTPUTS.with(|cell| cell.replace(enabled)),
        }
    }
}

impl Drop for StepPooledOutputs {
    fn drop(&mut self) {
        STEP_POOLED_OUTPUTS.with(|cell| cell.set(self.previous));
    }
}

/// Whether a [`StepPooledOutputs`] scope is open on this thread.
pub(crate) fn step_pooled_outputs() -> bool {
    STEP_POOLED_OUTPUTS.with(std::cell::Cell::get)
}

/// Reset the thread-local decode pool — moves every buffer handed out
/// since the last reset back to the free list.
///
/// **Caller contract:** no `MlxBuffer` returned by [`pooled_alloc_buffer`]
/// since the previous reset may still be in scope on this thread.  Calling
/// `reset_decode_pool` while a buffer is still referenced is a soundness
/// hole — the pool may re-issue the same Metal storage to a future
/// allocation, causing aliasing.
pub fn reset_decode_pool() {
    // Legacy/test compatibility entry. Production uses the checked form.
    if let Err(_error) = checked_reset_decode_pool() { profile::quarantine(); }
}

/// Caller guarantees no future issued-buffer access AND a successful drain.
pub fn checked_reset_decode_pool() -> std::result::Result<(), mlx_native::MlxError> {
    if let Some(result) = profile::reset() { return result; }
    let caps = CAPS.with(std::cell::Cell::get);
    DECODE_POOL.with(|cell| {
        let mut pool = cell.borrow_mut();
        pool.reset();
        if caps.is_some_and(|caps| pool.free_bytes() > caps.retained) {
            pool.clear();
        }
        FREE_AT_RESET.with(|free| free.set(pool.free_bytes()));
    });
    release_overflow();
    ISSUED_BYTES.with(|issued| issued.set(0));
    POOLED_SINCE_RESET.with(|pooled| pooled.set(0));
    if CAPS_END_AT_RESET.with(|cell| cell.replace(false)) {
        CAPS.with(|cell| cell.set(None));
    }
    Ok(())
}

/// Reset the thread-local pool at a per-prefill-layer boundary.
///
/// Bytewise identical to [`reset_decode_pool`]; the separate name documents
/// a distinct lifecycle: called between prefill layer iterations in
/// `forward_gpu_impl` after every layer has issued its
/// [`mlx_native::CommandEncoder::commit_and_wait`] (so all in-flight Metal
/// work referencing the layer-scoped scratches has drained), recycling the
/// dense-Q FFN scratches, attention pre-norm/per-head-norm/imrope scratches,
/// and DeltaNet apply_proj scratches before the next layer's allocations.
///
/// # Caller contract (W-5b.15)
///
/// At call time **no pool-allocated `MlxBuffer` may have an outstanding ARC
/// clone whose underlying Metal storage matches a power-of-two bucket the
/// next layer's [`pooled_alloc_buffer`] calls will request**.  In particular:
///
/// * The cross-layer `hidden` buffer (the residual stream consumed by the
///   next layer's attention) **must not** be pool-allocated.  Production
///   path: `embed_tokens_gpu` returns a `device.alloc_buffer` for the first
///   layer, and the dense-Q FFN's pooled `_into` variant (W-5b.15) writes
///   its FINAL output (the buffer that becomes the next `hidden`) to a
///   `device.alloc_buffer` — internal scratches stay pooled.
/// * Same-layer locals (`attn_out`, `q_normed`, FFN gate/up/hidden scratches,
///   etc.) are bound only inside the loop body and are dropped at the closing
///   brace before this reset fires from the *next* iteration's top.
///
/// At chunk-prefill working set (Qwen3.6-27B, seq_len=4096, h=5120, m=17408)
/// without per-layer reset, the dense-Q FFN's 5 pooled scratches alone
/// accumulate ~1 GB / dense layer × 33 dense layers ≈ 33 GB cumulative
/// before layer 33, overrunning Metal's residency-set quota and producing
/// "GPU command buffer completed with error status" — the W-5b.14
/// architectural-limit failure.  The per-layer reset closes this lifecycle
/// gap so dense-Q `_into` can use the pool unconditionally and capture the
/// W-5b.13 audit's projected ~30–40% allocation-churn savings.
pub fn reset_for_prefill_chunk() {
    if let Err(_error) = checked_reset_for_prefill_chunk() { profile::quarantine(); }
}

/// Successful K drain plus no future accesses to issued scratch are required.
pub fn checked_reset_for_prefill_chunk() -> std::result::Result<(), mlx_native::MlxError> {
    checked_reset_decode_pool()
}

/// Diagnostic accessor: number of buffers currently in-use (alloc'd but
/// not yet reset).  Surfaced for the `HF2Q_DECODE_PROFILE` instrumentation.
#[allow(dead_code)]
pub fn decode_pool_in_use_count() -> usize {
    if let Some((issued, _)) = profile::counts() { return issued; }
    DECODE_POOL.with(|cell| cell.borrow().in_use_count())
}

/// Diagnostic accessor: number of buffers currently in the free list.
#[allow(dead_code)]
pub fn decode_pool_free_count() -> usize {
    if let Some((_, free)) = profile::counts() { return free; }
    DECODE_POOL.with(|cell| cell.borrow().free_count())
}

/// Slot identifier for the three MoE FFN `quantized_matmul_id_ggml` call
/// sites: gate, up, and down.  Each slot gets its own thread-local
/// `IdMmScratch` so concurrent dispatches (gate+up in Phase C) do not
/// race on the scratch's htpe/hids buffers.
#[derive(Debug, Clone, Copy)]
pub enum MmIdSlot {
    Gate,
    Up,
    Down,
}

/// Run a closure with a mutable reference to the slot's thread-local
/// `IdMmScratch`, lazily growing the cached scratch if its capacity is
/// less than the requested `(n_experts, max_n_tokens)` pair.
///
/// W-5b.24 wire-up support — replaces 6 per-FFN-call device allocations
/// with 0 (after the first call) by amortising scratch ownership across
/// every FFN call in the prefill.
///
/// On capacity miss the cached scratch is dropped (its underlying Metal
/// buffers freed) and a new larger one is allocated; subsequent calls
/// at the same or smaller size hit the cache.
///
/// # Errors
///
/// Returns `MlxError` if `IdMmScratch::alloc` fails on cache miss.
pub fn with_id_mm_scratch<F, R>(
    slot: MmIdSlot,
    device: &MlxDevice,
    n_experts: u32,
    max_n_tokens: u32,
    f: F,
) -> std::result::Result<R, mlx_native::MlxError>
where
    F: FnOnce(&mut IdMmScratch) -> std::result::Result<R, mlx_native::MlxError>,
{
    let cell = match slot {
        MmIdSlot::Gate => &MM_ID_SCRATCH_GATE,
        MmIdSlot::Up => &MM_ID_SCRATCH_UP,
        MmIdSlot::Down => &MM_ID_SCRATCH_DOWN,
    };
    cell.with(|cell| {
        let mut guard = cell.borrow_mut();
        // Reallocate when the cached scratch's constructor bounds (its
        // capacity accessors) do not cover this call.
        let needs_realloc = match guard.as_ref() {
            None => true,
            Some(scratch) => {
                n_experts > scratch.capacity_n_experts()
                    || max_n_tokens > scratch.capacity_n_tokens()
            }
        };
        if needs_realloc {
            *guard = Some(IdMmScratch::alloc(device, n_experts, max_n_tokens)?);
        }
        let scratch = guard.as_mut().expect("just allocated");
        f(scratch)
    })
}

/// Drop all cached `IdMmScratch` slots for the current thread.
///
/// Used by the `forensic A/B` test path to force a re-allocation
/// between LEGACY and NEW runs so neither path benefits from a warm
/// scratch left behind by the other.  Production paths do not call
/// this; the scratches survive across prefills and decode tokens, with
/// per-allocation cost amortising to zero.
#[allow(dead_code)]
pub fn drop_id_mm_scratch() {
    MM_ID_SCRATCH_GATE.with(|cell| *cell.borrow_mut() = None);
    MM_ID_SCRATCH_UP.with(|cell| *cell.borrow_mut() = None);
    MM_ID_SCRATCH_DOWN.with(|cell| *cell.borrow_mut() = None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_resets_recycle_metal_buffers() {
        // Skip the GPU-touching part of this test if no Metal device is
        // available (CI builders, headless Linux).  The test merely
        // exercises pool lifecycle; the device is still required because
        // the first alloc must successfully create a Metal buffer.
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };

        // Cycle 1: alloc, locals drop, reset.
        let ptr_a = {
            let buf =
                pooled_alloc_buffer(&device, 1024, DType::F32, vec![256]).expect("cycle 1 alloc");
            buf.contents_ptr()
        };
        assert!(decode_pool_in_use_count() >= 1);

        reset_decode_pool();
        assert_eq!(decode_pool_in_use_count(), 0);

        // Cycle 2: same bucket size must reuse the cycle-1 metal buffer.
        let buf = pooled_alloc_buffer(&device, 1024, DType::F32, vec![256]).expect("cycle 2 alloc");
        let ptr_b = buf.contents_ptr();
        assert_eq!(
            ptr_b, ptr_a,
            "thread-local pool must reuse Metal buffer across reset"
        );

        reset_decode_pool();
    }
}
