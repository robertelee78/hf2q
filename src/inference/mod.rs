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
pub mod spec_decode;
pub mod vision;

/// ADR-053 — GLP (GGUF Layer Projection) runtime steering.
pub mod glp;

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
/// Guarded modules: ALL GPU-kernel-executing test modules — 40 files,
/// enumerated authoritatively in `iter230_a2_lock_discipline`
/// (models::qwen35::gpu_full_attn), which pins that every test in each
/// enumerated module acquires this lock first-statement. Extend BOTH
/// the guards and that list on any new GPU-family sweep failure (the
/// AC-A3 loop-until-dry rule). The initial 4-family scope was
/// insufficient — unguarded GPU tests in other modules (eagle3
/// forward, vit, forward_gpu) kept corrupting guarded ones.
///
/// Rules (pinned by `iter230_a2_lock_discipline` in qwen35::gpu_full_attn):
/// - acquire EXACTLY ONCE, as the FIRST statement of the `#[test]` fn:
///   `let _gpu = crate::inference::hf2q_gpu_test_lock();`
/// - helpers never acquire (the mutex is non-reentrant);
/// - guarded tests only submit synchronous GPU work (`commit_and_wait`),
///   so no work outlives the guard.
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
