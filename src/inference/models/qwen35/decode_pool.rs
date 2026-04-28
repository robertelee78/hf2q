//! Thread-local arena buffer pool for the qwen35 decode hot path.
//!
//! # Purpose
//!
//! ADR-012 §Optimize / Task #15 — close the MoE dwq46 0.90× decode parity gap
//! vs llama.cpp.  Diagnostic localization (`HF2Q_DECODE_PROFILE=1`) showed
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

/// Allocate from the thread-local decode pool.
///
/// API-compatible with `MlxDevice::alloc_buffer` so call sites in the
/// dispatch helpers can be edited mechanically.
#[inline]
pub fn pooled_alloc_buffer(
    device: &MlxDevice,
    byte_len: usize,
    dtype: DType,
    shape: Vec<usize>,
) -> std::result::Result<MlxBuffer, mlx_native::MlxError> {
    DECODE_POOL.with(|cell| {
        cell.borrow_mut().alloc(device, byte_len, dtype, shape)
    })
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
    DECODE_POOL.with(|cell| cell.borrow_mut().reset());
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
    DECODE_POOL.with(|cell| cell.borrow_mut().reset());
}

/// Diagnostic accessor: number of buffers currently in-use (alloc'd but
/// not yet reset).  Surfaced for the `HF2Q_DECODE_PROFILE` instrumentation.
#[allow(dead_code)]
pub fn decode_pool_in_use_count() -> usize {
    DECODE_POOL.with(|cell| cell.borrow().in_use_count())
}

/// Diagnostic accessor: number of buffers currently in the free list.
#[allow(dead_code)]
pub fn decode_pool_free_count() -> usize {
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
        // Capacity check: the inner scratch's caps are private, so we
        // proxy via `IdMmScratch::alloc`'s error-returning check_capacity
        // path indirectly — easier to just track the caps we requested
        // and grow when we exceed them.  Using the option's None state
        // for first-time allocation, plus a stored cap pair on Some.
        //
        // We tag the cap pair onto a side-table of static AtomicU32s
        // keyed off slot — but a simpler approach: stash the caps in
        // adjacent thread_local cells.  Inline that via the helper
        // closure here: any check_capacity error reallocates.
        let needs_realloc = match guard.as_ref() {
            None => true,
            Some(scratch) => {
                // Probe the public api: scratch.htpe.element_count is
                // the n_experts cap; scratch.hids.element_count() ==
                // n_experts_cap × n_tokens_cap.  Both fields pub.
                let cap_n_experts = scratch.htpe.element_count() as u32;
                let cap_total = scratch.hids.element_count() as u64;
                let cap_n_tokens = if cap_n_experts == 0 {
                    0
                } else {
                    (cap_total / cap_n_experts as u64) as u32
                };
                n_experts > cap_n_experts || max_n_tokens > cap_n_tokens
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

// ================================================================
// W-5b.26: FfnOutputCache — lift FFN output buffers to prefill scope
// ================================================================
//
// Mirrors the W-5b.24 `IdMmScratch` lifecycle pattern at the next layer
// up: instead of caching internal scratches of a single MoE-FFN call
// (gate/up/down `htpe`/`hids` slots), this caches the 9 output / scratch
// buffers `build_moe_ffn_layer_gpu_q_into` allocates per layer
// (`ids_buf` / `weights_buf` / `gate_all` / `up_all` / `h_all` / `y_all`
// / `h_s` / `out` / `silu_params`).  These 9 × 48 layers = 432 per-prefill
// `pooled_alloc_buffer` calls become 9 first-call device allocs + 423
// cache hits (zero device-alloc work per cached slot).
//
// # Lifetime / safety contract
//
// Every cached buffer's last GPU consumer is gated behind one of:
//   - `enc.commit_and_wait()` at the end of every prefill MoE layer
//     (`forward_gpu.rs:1671`, `seq_len > 1` branch), which drains all
//     pending dispatches before the next layer's encoder begins.
//   - `enc.commit()` at the end of every decode MoE layer
//     (`forward_gpu.rs:1669`, `seq_len == 1` branch).  No wait, but
//     Metal's command queue is in-order per-queue, so the next layer's
//     encoder cannot read until the previous layer's writes complete on
//     GPU.  The buffer's *next read* (next layer's `fused_residual_norm`
//     reading from `hidden`, where `hidden = previous out_buf clone`)
//     happens before that next layer's encoder issues its own writes
//     into the same `out` slot — the GPU enforces the ordering even
//     without a CPU-side wait.
//
// The cross-layer hidden contract documented at `reset_for_prefill_chunk`
// (W-5b.15) for the dense-Q FFN's output buffer requires `device.alloc_buffer`
// to survive the per-layer `reset_for_prefill_chunk` call, because the pool's
// reset hands the buffer's bucket back to the free list while a clone is
// still live as `hidden`.  This `FfnOutputCache` is a *separate* cache from
// the pool — it does NOT participate in `reset_for_prefill_chunk`'s reset
// (the cached MlxBuffer's storage Arc keeps the Metal buffer alive across
// all per-layer resets).  Therefore caching `out_buf` here is sound: there
// is no reset-and-re-issue race.
//
// # Grow-on-demand semantics
//
// All 9 buffer shapes scale with `seq_len` (`total_rows = seq * top_k`,
// `n_h_all = total_rows * m_moe`, etc.).  Within a single forward pass
// `seq_len` is fixed per chunk, but a prefill→decode transition (or a
// final ragged chunk) can shrink it.  We grow on size-up, never shrink:
// a cached prefill-sized buffer happily serves smaller decode requests
// because every kernel checks `byte_len() <` (less-than, allowing
// over-allocation), and the kernel's own `n` / `count` parameters drive
// the actual access bounds.  See the `proj_pooled` doc-comment for the
// same byte_len-as-upper-bound invariant.

/// Slot identifier for the 9 cached FFN output buffers in
/// `build_moe_ffn_layer_gpu_q_into`.  See the W-5b.26 module-level
/// comment for the lifetime contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnOutputSlot {
    /// `[total_rows]` U32 — expert IDs from `dispatch_moe_softmax_topk`.
    Ids,
    /// `[total_rows]` F32 — expert weights from `dispatch_moe_softmax_topk`.
    Weights,
    /// `[total_rows, m_moe]` F32 — gate proj output across all routed experts.
    GateAll,
    /// `[total_rows, m_moe]` F32 — up proj output across all routed experts.
    UpAll,
    /// `[total_rows, m_moe]` F32 — `silu(gate) * up` activations.
    HAll,
    /// `[total_rows, h]` F32 — down proj output across all routed experts.
    YAll,
    /// `[seq, m_sh]` F32 — `silu(a_s) * b_s` shared-expert activations.
    HS,
    /// `[seq, h]` F32 — final FFN output (cross-layer `hidden`).
    Out,
    /// `[1]` U32 — silu_mul element count for `h_all` dispatch.
    SiluParams,
}

impl FfnOutputSlot {
    const COUNT: usize = 9;
    const fn idx(self) -> usize {
        match self {
            FfnOutputSlot::Ids => 0,
            FfnOutputSlot::Weights => 1,
            FfnOutputSlot::GateAll => 2,
            FfnOutputSlot::UpAll => 3,
            FfnOutputSlot::HAll => 4,
            FfnOutputSlot::YAll => 5,
            FfnOutputSlot::HS => 6,
            FfnOutputSlot::Out => 7,
            FfnOutputSlot::SiluParams => 8,
        }
    }
}

/// Per-thread cache of the 9 `build_moe_ffn_layer_gpu_q_into` output /
/// scratch buffers (W-5b.26).  Each slot stores `Some(buf)` once the
/// first FFN call of a prefill has allocated it; subsequent calls in
/// the same prefill (and across prefills, until a capacity miss forces
/// realloc) return clones of the cached buffer.
pub struct FfnOutputCache {
    slots: [Option<MlxBuffer>; FfnOutputSlot::COUNT],
}

impl FfnOutputCache {
    const fn new() -> Self {
        Self {
            slots: [const { None }; FfnOutputSlot::COUNT],
        }
    }

    /// Get a clone of the slot's cached buffer, or allocate (and cache)
    /// a new one of at least `byte_len` bytes if the slot is empty or
    /// the cached buffer is too small.
    ///
    /// Grows on capacity miss; never shrinks.  Returns a `MlxBuffer::clone`
    /// of the cached slot — the clone shares the residency-set guard
    /// with the cached buffer (no double-registration, no orphaning if
    /// the cache realloc's the slot later).  Kernel byte-len checks
    /// use `<` (less-than), so the cache's max-size buffer happily serves
    /// smaller requests.  Kernel access bounds are driven by the explicit
    /// `n` / `m` / `count` parameters in the kernel ParamsRef, NOT from
    /// the buffer's `shape()` — see the audit comment in
    /// `quantized_matmul_id_ggml_pooled` and `proj_pooled` for the
    /// `byte_len`-as-upper-bound invariant.
    ///
    /// `byte_len` and `dtype` arguments drive the cache-miss alloc path
    /// only.  `shape` is unused on cache hit (the cached buffer's
    /// max-shape rides along on the clone) — kernels do not read
    /// `MlxBuffer::shape()` on any of the 9 lifted call sites.
    fn get_or_alloc(
        &mut self,
        slot: FfnOutputSlot,
        device: &MlxDevice,
        byte_len: usize,
        dtype: DType,
        shape: Vec<usize>,
    ) -> std::result::Result<MlxBuffer, mlx_native::MlxError> {
        let i = slot.idx();
        let needs_realloc = match self.slots[i].as_ref() {
            None => true,
            Some(buf) => buf.byte_len() < byte_len || buf.dtype() != dtype,
        };
        if needs_realloc {
            // Round up to a power-of-two bucket like `MlxBufferPool`
            // does, so subsequent slightly-larger requests inside the
            // same bucket also hit the cache.  Mirrors the pool's
            // `bucket_size` rounding semantics exactly.
            let bucket = byte_len.next_power_of_two().max(64);
            let new_buf = device.alloc_buffer(bucket, dtype, shape)?;
            self.slots[i] = Some(new_buf);
        }
        // Clone the cached buffer — Arc bump on storage, shared
        // residency-set guard, no Metal-side work.  See the
        // `impl Clone for MlxBuffer` doc-comment in
        // `mlx-native/src/buffer.rs:82-99` for the safety contract.
        Ok(self.slots[i].as_ref().expect("just allocated").clone())
    }
}

thread_local! {
    /// Per-thread cached FFN output buffers (W-5b.26).
    static FFN_OUTPUT_CACHE: RefCell<FfnOutputCache> = const {
        RefCell::new(FfnOutputCache::new())
    };
}

/// Run a closure with a mutable reference to the thread-local
/// `FfnOutputCache`.  See [`with_id_mm_scratch`] for the analogous
/// pattern on a different cache.
///
/// W-5b.26 wire-up support — replaces 9 per-FFN-call
/// `pooled_alloc_buffer` calls (per layer × 48 layers = 432 per
/// prefill) with cache hits after the first layer's allocations.
///
/// # Errors
///
/// Returns `MlxError` if `MlxDevice::alloc_buffer` fails on a
/// capacity-miss realloc.
pub fn with_ffn_output_cache<F, R>(f: F) -> std::result::Result<R, mlx_native::MlxError>
where
    F: FnOnce(&mut FfnOutputCache) -> std::result::Result<R, mlx_native::MlxError>,
{
    FFN_OUTPUT_CACHE.with(|cell| {
        let mut guard = cell.borrow_mut();
        f(&mut guard)
    })
}

/// Convenience wrapper around `FfnOutputCache::get_or_alloc` that runs
/// inside the thread-local borrow.  Mirrors the per-slot ergonomics of
/// `with_id_mm_scratch` while letting the caller keep allocation calls
/// flat at the call site (no nested closures around each slot).
pub fn ffn_output_get_or_alloc(
    slot: FfnOutputSlot,
    device: &MlxDevice,
    byte_len: usize,
    dtype: DType,
    shape: Vec<usize>,
) -> std::result::Result<MlxBuffer, mlx_native::MlxError> {
    FFN_OUTPUT_CACHE.with(|cell| {
        cell.borrow_mut()
            .get_or_alloc(slot, device, byte_len, dtype, shape)
    })
}

/// Drop all cached FFN output buffers for the current thread.
///
/// Mirrors [`drop_id_mm_scratch`] — used by forensic A/B test paths
/// to force re-allocation between LEGACY and NEW runs so neither
/// benefits from a warm cache left behind by the other.  Production
/// paths do not call this; the cached buffers survive across prefills
/// and decode tokens, with per-allocation cost amortising to zero.
#[allow(dead_code)]
pub fn drop_ffn_output_cache() {
    FFN_OUTPUT_CACHE.with(|cell| {
        let mut guard = cell.borrow_mut();
        for slot in &mut guard.slots {
            *slot = None;
        }
    });
}

/// Diagnostic accessor: number of slots currently populated (for tests
/// + the W-5b.26 self-test).
#[allow(dead_code)]
pub fn ffn_output_cache_populated_count() -> usize {
    FFN_OUTPUT_CACHE.with(|cell| {
        cell.borrow().slots.iter().filter(|s| s.is_some()).count()
    })
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
            let buf = pooled_alloc_buffer(&device, 1024, DType::F32, vec![256])
                .expect("cycle 1 alloc");
            buf.contents_ptr()
        };
        assert!(decode_pool_in_use_count() >= 1);

        reset_decode_pool();
        assert_eq!(decode_pool_in_use_count(), 0);

        // Cycle 2: same bucket size must reuse the cycle-1 metal buffer.
        let buf = pooled_alloc_buffer(&device, 1024, DType::F32, vec![256])
            .expect("cycle 2 alloc");
        let ptr_b = buf.contents_ptr();
        assert_eq!(ptr_b, ptr_a, "thread-local pool must reuse Metal buffer across reset");

        reset_decode_pool();
    }
}
