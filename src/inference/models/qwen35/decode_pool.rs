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

use std::cell::RefCell;

use mlx_native::ops::quantized_matmul_id_ggml::IdMmScratch;
use mlx_native::{DType, DenseMatmulIdScratch, MlxBuffer, MlxBufferPool, MlxDevice};

use crate::inference::dense_expert_activation::DenseExpertScratchCache;

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
    static MM_ID_SCRATCH_GATE: RefCell<LegacyExpertScratchCache> = RefCell::new(LegacyExpertScratchCache::default());
    static MM_ID_SCRATCH_UP: RefCell<LegacyExpertScratchCache> = RefCell::new(LegacyExpertScratchCache::default());
    static MM_ID_SCRATCH_DOWN: RefCell<LegacyExpertScratchCache> = RefCell::new(LegacyExpertScratchCache::default());
    static DENSE_ID_SCRATCH_GATE: RefCell<DenseExpertScratchCache> = RefCell::new(DenseExpertScratchCache::default());
    static DENSE_ID_SCRATCH_UP: RefCell<DenseExpertScratchCache> = RefCell::new(DenseExpertScratchCache::default());
    static DENSE_ID_SCRATCH_DOWN: RefCell<DenseExpertScratchCache> = RefCell::new(DenseExpertScratchCache::default());
}

struct LegacyExpertScratchEntry {
    activation_epoch: u64,
    device_registry_id: u64,
    scratch: IdMmScratch,
}

/// Model-bound cache for the legacy quantized expert-ID map scratch.
///
/// Capacity alone cannot authorize reuse: model swaps construct a fresh
/// `MlxDevice`/residency set even when the next artifact has identical expert
/// geometry. Retaining the old scratch would keep allocation-backed Metal
/// state alive and route the new generation through the prior residency
/// owner.
#[derive(Default)]
struct LegacyExpertScratchCache {
    entry: Option<LegacyExpertScratchEntry>,
    #[cfg(test)]
    allocation_generation: u64,
}

impl LegacyExpertScratchCache {
    fn owned_bytes(&self) -> u64 {
        self.entry.as_ref().map_or(0, |entry| {
            (entry.scratch.htpe.byte_len() + entry.scratch.hids.byte_len()) as u64
        })
    }

    fn with<R>(
        &mut self,
        activation_epoch: u64,
        device: &MlxDevice,
        n_experts: u32,
        max_n_tokens: u32,
        f: impl FnOnce(&mut IdMmScratch) -> std::result::Result<R, mlx_native::MlxError>,
    ) -> std::result::Result<R, mlx_native::MlxError> {
        if activation_epoch == 0 {
            return Err(mlx_native::MlxError::InvalidArgument(
                "legacy expert scratch activation epoch must be nonzero".into(),
            ));
        }
        let device_registry_id = device.registry_id();
        let must_allocate = self.entry.as_ref().is_none_or(|entry| {
            let cap_n_experts = entry.scratch.htpe.element_count() as u32;
            let cap_total = entry.scratch.hids.element_count() as u64;
            let cap_n_tokens = if cap_n_experts == 0 {
                0
            } else {
                (cap_total / u64::from(cap_n_experts)) as u32
            };
            entry.activation_epoch != activation_epoch
                || entry.device_registry_id != device_registry_id
                || n_experts > cap_n_experts
                || max_n_tokens > cap_n_tokens
        });
        if must_allocate {
            // Drop the prior activation/device owner before asking Metal for
            // replacement storage. A failed allocation therefore leaves no
            // stale scratch that a retry could accidentally reuse.
            self.entry.take();
            self.entry = Some(LegacyExpertScratchEntry {
                activation_epoch,
                device_registry_id,
                scratch: IdMmScratch::alloc(device, n_experts, max_n_tokens)?,
            });
            #[cfg(test)]
            {
                self.allocation_generation += 1;
            }
        }
        f(&mut self
            .entry
            .as_mut()
            .expect("legacy expert scratch cache allocated")
            .scratch)
    }
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
    DECODE_POOL.with(|cell| cell.borrow_mut().alloc(device, byte_len, dtype, shape))
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

/// Drop every allocation-backed thread-local resource before another model
/// activation is materialized on this thread.
///
/// This is deliberately stronger than the per-token reset: `reset` retains
/// power-of-two scratch buckets for reuse, while an A→B transition must not
/// preserve A's device/residency ownership or high-water allocation. The
/// caller must have already dropped the stale model-bound GPU cache and must
/// invoke this before allocating the replacement cache.
pub(super) fn clear_for_model_activation() {
    DECODE_POOL.with(|cell| {
        let mut pool = cell.borrow_mut();
        // `MlxBufferPool::clear` releases its free buffers but deliberately
        // retains the residency-set owner.  An activation boundary must drop
        // that owner too: the replacement model receives a fresh
        // residency-enabled `MlxDevice`, and retaining A's empty pool would
        // make B's first allocation fail closed as a mixed-device request.
        // Replacing the arena drops buffers, registrations, and owner as one
        // unit before the next activation allocates anything.
        *pool = MlxBufferPool::new();
    });
    drop_id_mm_scratch();
}

/// Release allocation-backed decode and expert scratch at a drained worker
/// boundary, returning the exact bytes owned by the registered buffers.
pub(super) fn idle_runtime_owned_bytes() -> std::result::Result<u64, &'static str> {
    let pool_bytes = DECODE_POOL.with(|cell| {
        let pool = cell.borrow();
        if pool.in_use_count() != 0 {
            return Err("Qwen decode pool still has in-use buffers");
        }
        Ok(pool.free_bytes() as u64)
    })?;
    let mut scratch_bytes = 0u64;
    for cell in [&MM_ID_SCRATCH_GATE, &MM_ID_SCRATCH_UP, &MM_ID_SCRATCH_DOWN] {
        cell.with(|cell| {
            scratch_bytes = scratch_bytes.saturating_add(cell.borrow().owned_bytes());
        });
    }
    for cell in [
        &DENSE_ID_SCRATCH_GATE,
        &DENSE_ID_SCRATCH_UP,
        &DENSE_ID_SCRATCH_DOWN,
    ] {
        cell.with(|cell| {
            scratch_bytes = scratch_bytes.saturating_add(cell.borrow().owned_bytes());
        });
    }
    pool_bytes
        .checked_add(scratch_bytes)
        .ok_or("Qwen idle scratch byte total overflow")
}

/// Release allocation-backed decode and expert scratch at a drained worker
/// boundary, returning the exact bytes owned by the registered buffers.
pub(super) fn release_idle_runtime_state() -> std::result::Result<u64, &'static str> {
    let expected = idle_runtime_owned_bytes()?;
    let pool_bytes = DECODE_POOL.with(|cell| {
        let mut pool = cell.borrow_mut();
        if pool.in_use_count() != 0 {
            return Err("Qwen decode pool still has in-use buffers");
        }
        let bytes = pool.free_bytes() as u64;
        // Keep the pool's model-bound residency set, but use its explicit
        // clear path so every free allocation is removed and committed
        // before the worker authenticates the park receipt.
        pool.clear();
        Ok(bytes)
    })?;
    let mut scratch_bytes = 0u64;
    for cell in [&MM_ID_SCRATCH_GATE, &MM_ID_SCRATCH_UP, &MM_ID_SCRATCH_DOWN] {
        cell.with(|cell| {
            let mut cache = cell.borrow_mut();
            scratch_bytes = scratch_bytes.saturating_add(cache.owned_bytes());
            *cache = LegacyExpertScratchCache::default();
        });
    }
    for cell in [
        &DENSE_ID_SCRATCH_GATE,
        &DENSE_ID_SCRATCH_UP,
        &DENSE_ID_SCRATCH_DOWN,
    ] {
        cell.with(|cell| {
            scratch_bytes = scratch_bytes.saturating_add(cell.borrow_mut().release_owned_bytes());
        });
    }
    let released = pool_bytes
        .checked_add(scratch_bytes)
        .ok_or("Qwen idle scratch byte total overflow")?;
    debug_assert_eq!(released, expected);
    Ok(released)
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

#[cfg(test)]
pub(super) fn cached_legacy_id_scratch_count() -> usize {
    MM_ID_SCRATCH_GATE.with(|cell| usize::from(cell.borrow().entry.is_some()))
        + MM_ID_SCRATCH_UP.with(|cell| usize::from(cell.borrow().entry.is_some()))
        + MM_ID_SCRATCH_DOWN.with(|cell| usize::from(cell.borrow().entry.is_some()))
}

#[cfg(test)]
pub(super) fn cached_dense_id_scratch_count() -> usize {
    DENSE_ID_SCRATCH_GATE.with(|cell| usize::from(cell.borrow().has_entry_for_test()))
        + DENSE_ID_SCRATCH_UP.with(|cell| usize::from(cell.borrow().has_entry_for_test()))
        + DENSE_ID_SCRATCH_DOWN.with(|cell| usize::from(cell.borrow().has_entry_for_test()))
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

/// Borrow one artifact-native scalar expert scratch. Gate/up/down have
/// distinct storage because their map stages may coexist in one encoder.
pub fn with_dense_id_scratch<F, R>(
    slot: MmIdSlot,
    activation_epoch: u64,
    device: &MlxDevice,
    n_experts: u32,
    max_n_tokens: u32,
    f: F,
) -> std::result::Result<R, mlx_native::MlxError>
where
    F: FnOnce(&DenseMatmulIdScratch) -> std::result::Result<R, mlx_native::MlxError>,
{
    let cell = match slot {
        MmIdSlot::Gate => &DENSE_ID_SCRATCH_GATE,
        MmIdSlot::Up => &DENSE_ID_SCRATCH_UP,
        MmIdSlot::Down => &DENSE_ID_SCRATCH_DOWN,
    };
    cell.with(|cell| {
        cell.borrow_mut()
            .with(activation_epoch, device, n_experts, max_n_tokens, f)
    })
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
    activation_epoch: u64,
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
        cell.borrow_mut()
            .with(activation_epoch, device, n_experts, max_n_tokens, f)
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
    MM_ID_SCRATCH_GATE.with(|cell| *cell.borrow_mut() = LegacyExpertScratchCache::default());
    MM_ID_SCRATCH_UP.with(|cell| *cell.borrow_mut() = LegacyExpertScratchCache::default());
    MM_ID_SCRATCH_DOWN.with(|cell| *cell.borrow_mut() = LegacyExpertScratchCache::default());
    DENSE_ID_SCRATCH_GATE.with(|cell| *cell.borrow_mut() = DenseExpertScratchCache::default());
    DENSE_ID_SCRATCH_UP.with(|cell| *cell.borrow_mut() = DenseExpertScratchCache::default());
    DENSE_ID_SCRATCH_DOWN.with(|cell| *cell.borrow_mut() = DenseExpertScratchCache::default());
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

    #[test]
    fn activation_clear_releases_decode_high_water_and_expert_scratch() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device_a = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        clear_for_model_activation();

        let pooled = pooled_alloc_buffer(&device_a, 1024, DType::F32, vec![256])
            .expect("allocate activation-A decode scratch");
        drop(pooled);
        reset_decode_pool();
        assert!(decode_pool_free_count() >= 1);

        with_id_mm_scratch(MmIdSlot::Gate, 11, &device_a, 3, 2, |_| Ok(()))
            .expect("allocate activation-A expert scratch");
        with_dense_id_scratch(MmIdSlot::Up, 11, &device_a, 3, 2, |_| Ok(()))
            .expect("allocate activation-A scalar expert scratch");
        assert_eq!(cached_legacy_id_scratch_count(), 1);
        assert_eq!(cached_dense_id_scratch_count(), 1);

        clear_for_model_activation();
        assert_eq!(decode_pool_in_use_count(), 0);
        assert_eq!(decode_pool_free_count(), 0);
        assert_eq!(cached_legacy_id_scratch_count(), 0);
        assert_eq!(cached_dense_id_scratch_count(), 0);

        let device_b = MlxDevice::new().expect("create activation-B device");
        let pooled_b = pooled_alloc_buffer(&device_b, 1024, DType::F32, vec![256])
            .expect("activation clear must release the residency owner before B allocates");
        drop(pooled_b);
        clear_for_model_activation();
    }

    #[test]
    fn legacy_expert_scratch_rebinds_across_a_b_a_epochs() {
        let _gpu = crate::inference::hf2q_gpu_test_lock();
        let device_a = match MlxDevice::new() {
            Ok(device) => device,
            Err(_) => return,
        };
        let device_b = MlxDevice::new().expect("create activation-B device");
        let mut cache = LegacyExpertScratchCache::default();
        for (epoch, device, expected_generation) in [
            (11, &device_a, 1),
            (11, &device_a, 1),
            (22, &device_b, 2),
            (33, &device_a, 3),
        ] {
            cache
                .with(epoch, device, 3, 2, |_| Ok(()))
                .expect("allocate epoch-bound legacy expert scratch");
            assert_eq!(cache.allocation_generation, expected_generation);
        }
    }
}
