//! Process-level [`EncoderWorker`] singleton (ADR-028 iter-382).
//!
//! Provides a lazy-init persistent worker thread for parallel command-buffer
//! encoding.  Spawned ONCE per process on first access, kept alive until
//! process exit.  This is the hf2q-side wrapper around mlx-native's
//! [`mlx_native::encoder_worker::EncoderWorker`].
//!
//! # Why a singleton?
//!
//! `forward_decode` runs many times per process (once per decoded token).
//! Per-token thread spawn was empirically falsified at -43 tok/s in the
//! prior threading attempt (see `forward_mlx.rs:4592-4595`).  The singleton
//! amortizes the spawn cost over all tokens served.
//!
//! # Usage (iter-383+ scope, not yet wired)
//!
//! ```ignore
//! use crate::serve::encoder_worker_singleton::global_encoder_worker;
//!
//! let worker = global_encoder_worker();
//! let (done_tx, done_rx) = std::sync::mpsc::channel();
//! worker.submit(move || {
//!     // ... encode work into a fresh CommandEncoder using shared
//!     //     Arc<MlxDevice> + Arc<MlxBuffer> handles ...
//!     done_tx.send(()).ok();
//! }).expect("worker died");
//! done_rx.recv().expect("no completion signal");
//! ```
//!
//! # When to enable
//!
//! Per ADR-028 iter-380/381, the worker is validated end-to-end at the
//! mlx-native level (Metal-dispatch tests pass).  iter-382 (this module)
//! adds the hf2q-side singleton.  iter-383+ adds production wiring +
//! benchmarks.  Default-OFF until bench shows positive delta.

use mlx_native::encoder_worker::EncoderWorker;
use std::sync::{LazyLock, Mutex};

/// Global encoder-worker singleton, lazily spawned on first access.
///
/// Wrapped in `Mutex` because `EncoderWorker::shutdown` requires `&mut`.
/// Call sites take a `MutexGuard` only briefly to call `submit()`; the
/// submit itself is non-blocking (just enqueues to channel).
static GLOBAL_ENCODER_WORKER: LazyLock<Mutex<EncoderWorker>> =
    LazyLock::new(|| Mutex::new(EncoderWorker::spawn()));

/// Submit a closure to the global encoder worker.  Returns immediately;
/// closure runs asynchronously on the worker thread.  Caller must arrange
/// own completion signaling (e.g., a `(tx, rx)` channel pair captured by
/// the closure).
///
/// Returns `Err` if the worker has been shut down (rare; typically only
/// during process exit).
pub fn submit_to_global_worker<F>(f: F) -> Result<(), &'static str>
where
    F: FnOnce() + Send + 'static,
{
    let guard = GLOBAL_ENCODER_WORKER
        .lock()
        .map_err(|_| "global encoder worker mutex poisoned")?;
    guard.submit(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn singleton_can_run_closure() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);
        let (tx, rx) = std::sync::mpsc::channel();

        submit_to_global_worker(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            tx.send(()).ok();
        }).expect("submit");

        rx.recv().expect("worker did not signal");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn singleton_persists_across_calls() {
        // Two sequential submits should both run on the SAME worker thread
        // (singleton, not per-call spawn).
        let thread_ids = Arc::new(Mutex::new(Vec::new()));
        let mut signals = Vec::new();

        for _ in 0..3 {
            let (tx, rx) = std::sync::mpsc::channel();
            signals.push(rx);
            let ids_clone = Arc::clone(&thread_ids);
            submit_to_global_worker(move || {
                let id = std::thread::current().id();
                ids_clone.lock().expect("lock").push(id);
                tx.send(()).ok();
            }).expect("submit");
        }

        for rx in signals {
            rx.recv().expect("worker died");
        }

        let ids = thread_ids.lock().expect("lock");
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], ids[1], "submissions ran on different threads");
        assert_eq!(ids[1], ids[2], "submissions ran on different threads");
    }
}
