//! Shared env-var serialization lock for integration tests.
//!
//! ## Why this module exists
//!
//! Integration tests in `tests/*` that mutate process-global environment
//! variables (`std::env::set_var` / `remove_var`) need a binary-wide
//! serialization mechanism. Within one cargo test binary process,
//! multiple test threads run by default; concurrent env mutations from
//! different tests race and produce non-deterministic state.
//!
//! Historical bug (2026-05-08, surfaced in v0.1 release-plan §2.1
//! baseline run): `tests/peer_parity_gates.rs` had `static ENV_LOCK:
//! Mutex<()>` and `tests/common/llama_cpp_runner.rs::tests` had its own
//! `static TEST_ENV_LOCK: Mutex<()>`. Both ran in the same
//! `peer_parity_gates` binary process and both mutated
//! `HF2Q_LLAMA_PERPLEXITY_BIN`, but they were DIFFERENT Mutex
//! instances — they did not serialize with each other. Racing test
//! ordering caused `peer_perplexity_wrapper_handles_missing_binary` to
//! observe an unset env var (after the sibling test's `remove_var`
//! beat its own `resolve_binary` read), `resolve_binary` fell back to
//! `$PATH`, found the real `llama-perplexity` binary, and the test
//! panicked on the missing-binary sentinel assertion. That panic
//! poisoned its `ENV_LOCK` and cascaded into 2 sibling tests that
//! `expect`ed un-poisoned acquisition.
//!
//! The fix: a single shared `ENV_LOCK` (this module) referenced from
//! every env-mutating test in any `tests/` integration binary. Plus a
//! `lock_env()` helper that recovers from `PoisonError` — the
//! protected data is `()` so there is nothing to corrupt; treating
//! poisoning as fatal cascades unrelated test failures for no benefit.
//!
//! ## Usage
//!
//! ```ignore
//! use crate::common::env_lock::lock_env;
//!
//! #[test]
//! fn my_env_mutating_test() {
//!     let _guard = lock_env();
//!     // SAFETY: lock_env() serialises every env-mutating test in this
//!     // binary; the unsafe set_var/remove_var calls are bounded to
//!     // this critical section.
//!     unsafe {
//!         std::env::set_var("MY_VAR", "value");
//!     }
//!     // … test body …
//!     unsafe {
//!         std::env::remove_var("MY_VAR");
//!     }
//!     // assertions go here; if they panic, the lock un-poisons cleanly
//!     // for the next test (see ENV_LOCK doc-comment).
//! }
//! ```

use std::sync::{Mutex, MutexGuard};

/// Process-wide env-mutation serialization lock.
///
/// Every test in this binary that calls `std::env::set_var`,
/// `std::env::remove_var`, or otherwise reads-then-mutates env state
/// MUST acquire this via [`lock_env`] before mutating, hold it for the
/// duration of the read+mutate window, and drop it after.
///
/// Stored value is `()` — the lock protects "the right to mutate the
/// process env table for a moment", not any in-Rust data. Poisoning
/// recovery is therefore always safe (see [`lock_env`]).
pub static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Acquires [`ENV_LOCK`] with poison-recovery.
///
/// `Mutex<()>` auto-poisons when a holder panics. For mutexes that
/// guard real data this is a safety mechanism (the data may be in a
/// half-mutated state). For our `Mutex<()>` it serves no purpose —
/// there is no data to corrupt. Treating poisoning as a hard failure
/// cascades unrelated test failures: a bug in one test poisons the
/// lock and every sibling test using the lock then fails on
/// `expect("ENV_LOCK poisoned")` regardless of whether the original
/// bug affects them.
///
/// `lock_env` recovers via `PoisonError::into_inner`, returning a
/// usable guard. Tests that panic still report the original cause
/// (the panic propagates), but the lock state is reset so siblings
/// run cleanly.
#[must_use]
pub fn lock_env() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
