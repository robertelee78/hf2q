//! ADR-005 Phase 4 iter-209 (W77) — Multi-model auto-swap E2E scaffolding.
//!
//! This file is the integration-test surface for the AppState + pool +
//! handler refactor that iter-209 lands.  Iter-209 itself ships the
//! AppState wiring + handler routing + router-level unit tests in
//! `src/serve/api/router.rs::tests::iter209_*` (5 in-binary tests
//! against the synthetic empty-pool path).
//!
//! Iter-210 (W78) is responsible for fleshing out the **subprocess** end
//! of this file with the cross-model swap harness:
//!
//!   1. `hf2q serve --model <MODEL_A>` boots; `/v1/chat/completions`
//!      with `{"model": "<repo-id-A>"}` returns 200.
//!   2. The same server gets `/v1/chat/completions` with
//!      `{"model": "<repo-id-B>"}` — pre-iter-209 this would 400
//!      `model_not_loaded`; post-iter-209 the manager auto-swaps
//!      (LRU evicts A or B depending on memory budget) and returns 200.
//!   3. `/v1/models` reports `loaded: true` for whichever model is
//!      currently MRU.
//!
//! # Scopes (matches the iter-208 `multi_model_hotswap.rs` pattern)
//!
//! 1. **Default**: skip with a diagnostic.  Keeps `cargo test --release`
//!    cheap on dev machines.
//! 2. **`HF2Q_MULTI_MODEL_SWAP_E2E=1` + `HF2Q_MULTI_MODEL_SWAP_E2E_GGUF_A`
//!    + `HF2Q_MULTI_MODEL_SWAP_E2E_GGUF_B`**: runs the subprocess
//!    swap harness once iter-210 lands.  This iter-209 ships the
//!    scaffolding (verifies the binary boots; documents the closure
//!    target).
//!
//! ```bash
//! cargo test --release --test multi_model_swap
//!   # → all skipped, exit 0
//!
//! HF2Q_MULTI_MODEL_SWAP_E2E=1 \
//!   HF2Q_MULTI_MODEL_SWAP_E2E_GGUF_A=/path/to/model-a.gguf \
//!   HF2Q_MULTI_MODEL_SWAP_E2E_GGUF_B=/path/to/model-b.gguf \
//!   cargo test --release --test multi_model_swap -- --test-threads=1
//!   # → iter-210 will flesh out the spawn + swap + assert body here
//! ```

use std::path::PathBuf;
use std::process::Command;

const ENV_GATE: &str = "HF2Q_MULTI_MODEL_SWAP_E2E";
const ENV_GGUF_A: &str = "HF2Q_MULTI_MODEL_SWAP_E2E_GGUF_A";
const ENV_GGUF_B: &str = "HF2Q_MULTI_MODEL_SWAP_E2E_GGUF_B";

fn skip_unless_gated(name: &str) -> bool {
    if std::env::var(ENV_GATE).as_deref() == Ok("1") {
        return false;
    }
    eprintln!(
        "[skip] {name} — set {ENV_GATE}=1 (plus {ENV_GGUF_A} + {ENV_GGUF_B} \
         pointing at two cached GGUFs) to run the iter-210 multi-model swap E2E harness"
    );
    true
}

/// Locate the `hf2q` binary the cargo test runner just built.
fn hf2q_binary_path() -> PathBuf {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            PathBuf::from(manifest_dir).join("target")
        });
    let binary = target_dir.join("release").join("hf2q");
    assert!(
        binary.exists(),
        "hf2q binary not found at {} — did `cargo build --release` run?",
        binary.display()
    );
    binary
}

/// Smoke: `hf2q --version` returns 0.  Always-on; verifies the
/// scaffolding can locate the binary so iter-210's full body has a
/// known-good entry point.  Mirrors `tests/multi_model_hotswap.rs`'s
/// always-on smoke.
#[test]
fn binary_is_locatable_and_runs_version() {
    let bin = hf2q_binary_path();
    let out = Command::new(&bin)
        .arg("--version")
        .output()
        .expect("spawn hf2q --version");
    assert!(
        out.status.success(),
        "hf2q --version exited {:?}; stderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Iter-210 closure stub.  Body verifies the env-pointed GGUFs exist;
/// full subprocess swap-timing harness lands in iter-210.
#[test]
fn multi_model_swap_two_ggufs_e2e() {
    if skip_unless_gated("multi_model_swap_two_ggufs_e2e") {
        return;
    }
    let model_a = match std::env::var(ENV_GGUF_A) {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("[skip] multi_model_swap_two_ggufs_e2e — {ENV_GGUF_A} not set");
            return;
        }
    };
    let model_b = match std::env::var(ENV_GGUF_B) {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("[skip] multi_model_swap_two_ggufs_e2e — {ENV_GGUF_B} not set");
            return;
        }
    };
    assert!(
        model_a.exists(),
        "{ENV_GGUF_A} points at non-existent path: {}",
        model_a.display()
    );
    assert!(
        model_b.exists(),
        "{ENV_GGUF_B} points at non-existent path: {}",
        model_b.display()
    );
    eprintln!(
        "[stub] multi_model_swap_two_ggufs_e2e — iter-209 (W77) scaffolding only. \
         Iter-210 (W78) will flesh out the swap-timing assertion + AC 5356 closure. \
         GGUF_A={}, GGUF_B={}",
        model_a.display(),
        model_b.display()
    );
    let bin = hf2q_binary_path();
    let out = Command::new(&bin)
        .arg("--help")
        .output()
        .expect("spawn hf2q --help");
    assert!(out.status.success(), "hf2q --help exited {:?}", out.status);
}
