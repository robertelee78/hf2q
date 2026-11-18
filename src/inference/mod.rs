//! Per-model inference modules.
//!
//! This is the home for architecture-specific inference code that does not
//! fit inside `src/serve/forward_mlx.rs` (which is Gemma-4-shaped).
//!
//! Per `project_model_class_split.md`: all model-specific code lives in a
//! per-model file under `models/`, not in generic infra.
//!
//! Populated by ADR-013 (Qwen3.5 / Qwen3.5-MoE).

pub mod models;
pub(crate) mod device_settle;
pub(crate) mod native_observation;
pub(crate) mod route_host;
pub mod spec_decode;
pub(crate) mod stage_profile;
pub mod vision;

/// ADR-053 — GLP (GGUF Layer Projection) runtime steering.
pub mod glp;

/// ADR-059 — KV-cache grafts (context-space steering artifacts).
pub mod graft;

// C1 is a contract oracle, not the C2 production multi-anchor implementation.
#[cfg(test)]
pub(crate) mod anchor_contract;

/// ADR-005 iter-230 A2 — crate-wide serialization lock for GPU-executing
/// tests.
///
/// Concurrent Metal encoding from multiple `cargo test` threads corrupts
/// kernel results (NaN bit-patterns, non-finite outputs, parity blowups —
/// spike: `gpu_full_attn` family 5/65 FAILED parallel, 65/65 green
/// single-threaded). Production is NOT in this mode: serve routes GPU
/// encoding through the encoder-worker singleton
/// (`serve/encoder_worker_singleton.rs`). Concurrent multi-thread Metal
/// encoding is UNSUPPORTED crate-wide; this lock serializes the test-only
/// violations of that rule against each other, cross-module, while
/// leaving CPU tests fully parallel.
///
/// Every GPU-kernel-executing test must hold this guard before creating or
/// encoding Metal work. The initial four-family scope was insufficient:
/// unguarded tests in other modules kept corrupting guarded ones.
///
/// Acquire exactly once per test invocation. A private shared test-entry
/// helper may acquire as its first statement if its callers do not acquire;
/// the mutex is non-reentrant. Keep the guard alive through checked completion
/// of all submitted GPU work. CPU-only tests need no GPU guard.
///
/// Poison-tolerant by design: a panicked GPU test must not cascade
/// poison-unwrap failures through the rest of the suite. Limitation
/// (accepted, ADR iter-230): a panicking test could theoretically leave
/// in-flight GPU state; cascade failures are attributable via the
/// original panic. Runner caveat: `cargo-nextest` runs tests in separate
/// PROCESSES — an in-process mutex does not serialize it; GPU families
/// are supported under `cargo test` only.
#[cfg(test)]
pub(crate) fn hf2q_gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) mod expert_dispatch;
pub(crate) mod gguf_contract;
/// Issue #727 — slot-major KV row copies shared by every family's
/// reclaimed-slot spill.
pub(crate) mod kv_slot_rows;

// Pending native0.15.13 publication/pin. Diagnostic only; no serving activation.
#[cfg(test)]
pub mod backing_observation;
