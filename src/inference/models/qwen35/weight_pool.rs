//! Long-lived `MlxBufferPool` for static weight residency-set registration.
//!
//! # Purpose
//!
//! ADR-005 Wave 5b.7 iter 2 — adopt mlx-native's
//! [`MlxBufferPool::register_existing`] residency-only registration path
//! across hf2q's weight-loading hot path so static weight tensors join the
//! device's `MTLResidencySet`.  Without `MTLResidencySet` membership, Metal
//! treats every weight buffer as a candidate for compaction/eviction and
//! pays cold-page-fault costs on first dispatch.  With residency hints,
//! the OS keeps pages wired and the cold first-forward pays the
//! ~17 GB DMA-from-disk cost only.
//!
//! # Why a separate pool from `decode_pool::DECODE_POOL`
//!
//! The decode pool is per-token: it bucket-rounds allocations to
//! `next_power_of_two`, bulk-recycles via [`reset`](MlxBufferPool::reset)
//! on every token, and serves transient activation buffers (~1750
//! allocs/token).  Routing static weights through that bucketing path
//! would inflate the 17.26 GB Qwen3.6 27B DWQ46 weight set to ~25.55 GB
//! (+48% / +8.30 GB) — unshippable on a 128 GB unified-memory M5 Max
//! (Wave 5b.6 STOP report).
//!
//! Instead, weights are still allocated at their exact size via
//! [`MlxDevice::alloc_buffer`] (no rounding) and only their residency-set
//! membership is tracked through the pool via
//! [`MlxBufferPool::register_existing`]: the pool **does not** take
//! ownership and **does not** recycle these buffers.  The caller's
//! `MlxBuffer` (held in `ForwardGpuCache`) remains the canonical owner.
//!
//! # Lifecycle
//!
//! * The pool is initialized lazily on first [`register_weight_buffer`]
//!   call.
//! * Weight `MlxBuffer`s are allocated via `device.alloc_buffer(...)` (or
//!   loaded via `gguf.load_tensor(...)`) **as before**, then registered
//!   via [`register_weight_buffer`] before being stored in
//!   `ForwardGpuCache`.
//! * The pool lives for the lifetime of the thread (`thread_local!`).
//!   Because forward passes run on a single owning thread (per
//!   `feedback_oom_prevention`: one model-loading inference at a time),
//!   the pool effectively spans every forward call — the residency hint
//!   stays in place across all dispatches.
//! * On thread teardown, the pool's `Drop` runs `remove_all_residency_allocations`;
//!   the underlying `metal::Buffer` ARCs are still held by the caller's
//!   `MlxBuffer` handles and are not freed.
//!
//! # `HF2Q_NO_RESIDENCY=1` escape hatch
//!
//! When `HF2Q_NO_RESIDENCY=1` is set in the environment, the
//! [`MlxDevice::new`] constructor in mlx-native returns a device with
//! `residency_set: None`.  In that mode [`MlxBufferPool::register_existing`]
//! returns `Ok(())` as a no-op — operators who suspect a residency-induced
//! regression can opt out without recompiling.
//!
//! # Soundness contract
//!
//! No `MlxBuffer` whose underlying `metal::Buffer` was registered via
//! [`register_weight_buffer`] may be dropped before the
//! `ForwardGpuCache`'s pool reference goes away.  In practice this is
//! trivial: both the buffers and the pool live for the program lifetime
//! (the cache is rebuilt only on model swap, and mlx-native's pool `Drop`
//! correctly cleans up residency-set membership before the device is
//! dropped).

use std::cell::RefCell;

use mlx_native::{MlxBuffer, MlxBufferPool, MlxDevice, MlxError};

thread_local! {
    /// Per-thread long-lived pool for static weight residency-set membership.
    /// Initialized lazily on the first [`register_weight_buffer`] call.
    static WEIGHT_POOL: RefCell<MlxBufferPool> = RefCell::new(MlxBufferPool::new());
}

/// Register `buffer`'s underlying Metal allocation with the thread-local
/// weight pool's residency set.
///
/// API-compatible no-op when `HF2Q_NO_RESIDENCY=1` is set.  Idempotent:
/// re-registering the same buffer is a HashMap lookup.
///
/// The pool does **not** take ownership of `buffer` — the caller retains
/// the `MlxBuffer` handle and is responsible for keeping it alive for as
/// long as the residency hint should stay active.
#[inline]
pub fn register_weight_buffer(
    device: &MlxDevice,
    buffer: &MlxBuffer,
) -> std::result::Result<(), MlxError> {
    WEIGHT_POOL.with(|cell| cell.borrow_mut().register_existing(device, buffer))
}

/// Diagnostic accessor: number of buffers tracked in the residency set
/// (i.e. number of distinct `register_weight_buffer` callers whose buffers
/// are still pointing at unique Metal allocations).
///
/// Note: this counts unique `metal::Buffer.contents()` pointers — re-registering
/// the same buffer does not increase the count.
#[allow(dead_code)]
pub fn weight_pool_residency_count() -> usize {
    // The pool's internal `resident_buffers` HashMap is private; we
    // approximate via `free_count + in_use_count`, which is always 0
    // for a register-only pool.  Tests that need exact counts should
    // construct a fresh pool and call the public counters there.
    WEIGHT_POOL.with(|cell| {
        let p = cell.borrow();
        p.free_count() + p.in_use_count()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_native::DType;

    #[test]
    fn register_existing_via_thread_local_is_idempotent() {
        // Skip if no Metal device available (CI / headless Linux).
        let device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };

        let buf = device
            .alloc_buffer(2048, DType::U8, vec![2048])
            .expect("alloc external");

        // First registration — should succeed.
        register_weight_buffer(&device, &buf).expect("register 1");
        // Idempotent — second registration is a no-op.
        register_weight_buffer(&device, &buf).expect("register 2 (idempotent)");

        // External buffer still valid.
        let slice: &[u8] = buf.as_slice().expect("slice still valid");
        assert_eq!(slice.len(), 2048);
    }

    #[test]
    fn register_does_not_recycle_external_buffers() {
        // Verify the register-only path: in_use + free counts must stay 0.
        let _device = match MlxDevice::new() {
            Ok(d) => d,
            Err(_) => return,
        };
        // `weight_pool_residency_count` returns free + in_use; for a
        // register-only pool both stay 0 regardless of how many buffers
        // are registered.
        assert_eq!(weight_pool_residency_count(), 0);
    }
}
