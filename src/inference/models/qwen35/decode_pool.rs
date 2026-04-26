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

use mlx_native::{DType, MlxBuffer, MlxBufferPool, MlxDevice};

thread_local! {
    /// Per-thread arena pool.  Initialized lazily by the first
    /// [`pooled_alloc_buffer`] call; reset between decode tokens.
    static DECODE_POOL: RefCell<MlxBufferPool> = RefCell::new(MlxBufferPool::new());
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
