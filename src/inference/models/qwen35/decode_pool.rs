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
// W-5b.27 Phase B: DenseFfnOutputCache — lift dense FFN scratches to
// prefill scope (Qwen3.6-27B DWQ46 production path)
// ================================================================
//
// Mirrors the W-5b.26-reverted `FfnOutputCache` pattern (scaffolded for
// MoE-Q at commit 21ba91e, reverted at fb93a43 because the target
// `build_moe_ffn_layer_gpu_q_into` never executes for the bench model)
// but targets the function that DOES execute for every layer of the
// Qwen3.6-27B DWQ46 PP4106 walkbar bench:
// `build_dense_ffn_layer_gpu_q_into_pooled` at `gpu_ffn.rs:777-899`.
//
// Per the W-5b.27 pre-coding verification probe (HF2Q_W5B27_PROBE=1
// at hf2q HEAD `ad86d45`):
//   - DENSE_FFN_Q_INTO_POOLED_ENTRY fired 128× per single-token
//     generation (64 prefill + 64 decode) — every layer's prefill
//     touches it.
//   - FFN_VARIANT layer={0..63} variant=DenseQ — all 64 layers DenseQ,
//     none MoeQ. (n_expert=0 also confirmed via `gguf-dump`: model
//     has only `ffn_down/gate/up`, no `ffn_*_exps` tensors.)
//
// 4 pooled scratches × 64 layers = **256 per-prefill pool ops**
// eligible for the lift pattern (vs the W-5b.26 attempt's 9 × 48 = 432
// pool ops at the wrong-target MoE-Q function).
//
// # The 4 pooled scratches caught by this cache
//
// Per `gpu_ffn.rs:807-836`:
//
// 1. `gate_buf` — F32 [seq_len, m] (n_h × 4 bytes) — gate proj output
// 2. `up_buf`   — F32 [seq_len, m] — up proj output
// 3. `hidden_buf` — F32 [seq_len, m] — `silu(gate) * up` activation
// 4. `silu_params_buf` — U32 [1] (4 bytes) — silu_mul element count
//
// (The FINAL output `down_out` and the optional `sum_buf` are
// `device.alloc_buffer` at prefill — they cross the per-layer
// `reset_for_prefill_chunk` boundary as the next layer's `hidden`,
// so they MUST stay outside any reset-driven free list. They also
// must NOT be cached here because a single shared cache slot for the
// final output would alias the previous layer's `hidden` clone.)
//
// # Lifetime / safety contract
//
// The 4 lifted scratches are pure single-layer scratches: every kernel
// reading them lives behind an `enc.memory_barrier()` that drains
// before the layer's `commit_and_wait()` returns (prefill) or before
// the next layer's encoder begins (decode, in-order Metal queue).
// The cached `MlxBuffer`'s storage Arc keeps its underlying Metal
// buffer alive across the per-layer `reset_for_prefill_chunk` reset —
// the cache is a SEPARATE thread-local from `MlxBufferPool`, NOT
// touched by the pool's reset path.
//
// Unlike the (W-5b.26-reverted) MoE-Q `FfnOutputCache::Out` slot, this
// cache does NOT cache the cross-layer `hidden` (down_out / sum_buf
// is device-alloc'd in-function) — so the cross-layer aliasing hazard
// W-5b.15 documents for the dense-Q FFN's output buffer DOES NOT
// APPLY to the 4 internal scratches we DO cache.
//
// # Grow-on-demand semantics
//
// All 4 buffer shapes scale with `seq_len`:
//   gate/up/hidden bytes = seq_len × m × 4
//   silu_params bytes    = 4 (constant)
// Within a single forward pass `seq_len` is fixed per chunk, but a
// prefill→decode transition (or a final ragged chunk) shrinks it.
// We grow on size-up, never shrink — the cached prefill-sized buffer
// happily serves smaller decode requests (kernel byte-len checks
// use `<` less-than, allowing over-allocation; access bounds are
// driven by the explicit `n_h` / `n_out` parameters in the kernel
// ParamsRef, NOT from `MlxBuffer::shape()`).

/// Slot identifier for the 4 cached dense-Q FFN scratch buffers in
/// `build_dense_ffn_layer_gpu_q_into_pooled` (W-5b.27 Phase B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseFfnSlot {
    /// `[seq_len, m]` F32 — gate proj output.
    Gate,
    /// `[seq_len, m]` F32 — up proj output.
    Up,
    /// `[seq_len, m]` F32 — `silu(gate) * up` activation.
    Hidden,
    /// `[1]` U32 — silu_mul element count for `hidden` dispatch.
    SiluParams,
}

impl DenseFfnSlot {
    const COUNT: usize = 4;
    const fn idx(self) -> usize {
        match self {
            DenseFfnSlot::Gate => 0,
            DenseFfnSlot::Up => 1,
            DenseFfnSlot::Hidden => 2,
            DenseFfnSlot::SiluParams => 3,
        }
    }
}

/// Per-thread cache of the 4 dense-Q FFN internal scratch buffers
/// (W-5b.27 Phase B).
pub struct DenseFfnOutputCache {
    slots: [Option<MlxBuffer>; DenseFfnSlot::COUNT],
}

impl DenseFfnOutputCache {
    const fn new() -> Self {
        Self {
            slots: [const { None }; DenseFfnSlot::COUNT],
        }
    }

    /// Get a clone of the slot's cached buffer, or allocate (and cache)
    /// a new one of at least `byte_len` bytes if the slot is empty or
    /// the cached buffer is too small.
    ///
    /// Grows on capacity miss; never shrinks.  Returns an
    /// `MlxBuffer::clone` of the cached slot — the clone shares the
    /// residency-set guard with the cached buffer (no double-registration,
    /// no orphaning if the cache realloc's the slot later).  Kernel
    /// byte-len checks use `<` (less-than), so the cache's max-size
    /// buffer happily serves smaller requests.
    ///
    /// `byte_len` and `dtype` arguments drive the cache-miss alloc path
    /// only.  `shape` is stored on the alloc but the kernel access bounds
    /// are driven by explicit `n_h` / `n_out` parameters.
    fn get_or_alloc(
        &mut self,
        slot: DenseFfnSlot,
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
            // same bucket also hit the cache.
            let bucket = byte_len.next_power_of_two().max(64);
            let new_buf = device.alloc_buffer(bucket, dtype, shape)?;
            self.slots[i] = Some(new_buf);
        }
        Ok(self.slots[i].as_ref().expect("just allocated").clone())
    }
}

thread_local! {
    /// Per-thread cached dense-Q FFN scratch buffers (W-5b.27 Phase B).
    ///
    /// Lives across all 64 layers of a single prefill (and across
    /// decode tokens) until a capacity miss forces realloc.  First
    /// prefill pays 4 device allocs (one per slot, lazy on cache
    /// miss); every subsequent layer in the same prefill (and every
    /// subsequent prefill at smaller-or-equal seq_len) hits the cache
    /// with zero device-alloc work.
    static DENSE_FFN_OUTPUT_CACHE: RefCell<DenseFfnOutputCache> = const {
        RefCell::new(DenseFfnOutputCache::new())
    };
}

/// Run a closure with a mutable reference to the thread-local
/// `DenseFfnOutputCache`.  See [`with_id_mm_scratch`] for the analogous
/// pattern on a different cache.
///
/// W-5b.27 Phase B wire-up support — replaces 4 per-FFN-call
/// `pooled_alloc_buffer` calls (per layer × 64 layers = 256 per
/// prefill) with cache hits after the first layer's allocations.
///
/// # Errors
///
/// Returns `MlxError` if `MlxDevice::alloc_buffer` fails on a
/// capacity-miss realloc.
#[allow(dead_code)]
pub fn with_dense_ffn_output_cache<F, R>(f: F) -> std::result::Result<R, mlx_native::MlxError>
where
    F: FnOnce(&mut DenseFfnOutputCache) -> std::result::Result<R, mlx_native::MlxError>,
{
    DENSE_FFN_OUTPUT_CACHE.with(|cell| {
        let mut guard = cell.borrow_mut();
        f(&mut guard)
    })
}

/// Convenience wrapper around `DenseFfnOutputCache::get_or_alloc` that
/// runs inside the thread-local borrow.  Mirrors the per-slot ergonomics
/// of `ffn_output_get_or_alloc` (W-5b.26-reverted) while keeping
/// allocation calls flat at the call site.
pub fn dense_ffn_get_or_alloc(
    slot: DenseFfnSlot,
    device: &MlxDevice,
    byte_len: usize,
    dtype: DType,
    shape: Vec<usize>,
) -> std::result::Result<MlxBuffer, mlx_native::MlxError> {
    DENSE_FFN_OUTPUT_CACHE.with(|cell| {
        cell.borrow_mut()
            .get_or_alloc(slot, device, byte_len, dtype, shape)
    })
}

/// Drop all cached dense-Q FFN scratches for the current thread.
///
/// Mirrors [`drop_id_mm_scratch`] — used by forensic A/B test paths
/// to force re-allocation between LEGACY and NEW runs so neither
/// benefits from a warm cache left behind by the other.  Production
/// paths do not call this; the cached buffers survive across prefills
/// and decode tokens.
#[allow(dead_code)]
pub fn drop_dense_ffn_output_cache() {
    DENSE_FFN_OUTPUT_CACHE.with(|cell| {
        let mut guard = cell.borrow_mut();
        for slot in &mut guard.slots {
            *slot = None;
        }
    });
}

/// Diagnostic accessor: number of slots currently populated in the
/// thread-local `DenseFfnOutputCache`.  Used by forensic test paths
/// to confirm cache participation.
#[allow(dead_code)]
pub fn dense_ffn_output_cache_populated_count() -> usize {
    DENSE_FFN_OUTPUT_CACHE.with(|cell| {
        cell.borrow()
            .slots
            .iter()
            .filter(|s| s.is_some())
            .count()
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
