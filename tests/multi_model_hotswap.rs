//! ADR-005 Phase 4 iter-208 (W76) — HotSwapManager E2E harness scaffolding.
//!
//! AC 5356 (line 4779 of `docs/ADR-005-inference-server.md`) calls for a
//! measurement harness proving that two cached GGUFs swap-time on M5
//! Max stays under the spec'd budget.  This file is the scaffolding
//! that AC closure (iter-210) will fill in once iter-209 wires the
//! [`crate::serve::multi_model::HotSwapManager`] through `AppState`
//! and the `/v1/chat/completions` handler dispatches against the
//! manager (replacing today's single-slot `Option<Engine>`).
//!
//! # Why the scaffolding ships now
//!
//! 1. **Iter-208's surface is in-binary only.**  hf2q is a binary
//!    crate (no `[lib]` target — see `tests/mmproj_llama_cpp_compat.rs:32-37`)
//!    so an external test cannot import `HotSwapManager` directly.
//!    The 14 unit tests in `src/serve/multi_model.rs::tests` already
//!    exercise the manager's eviction + accounting + in-flight-Arc
//!    safety + load-or-get + try-get + evict + loader-error + filesize
//!    paths against a synthetic `MockEngine` fixture.  The integration
//!    surface that needs cross-process coverage is the
//!    `cmd_serve` ⇄ `HotSwapManager` ⇄ `/v1/chat/completions` chain,
//!    which lands in iter-209 (`AppState::pool: Arc<RwLock<HotSwapManager<Engine>>>`).
//!
//! 2. **Scaffolding now lets iter-210 close AC 5356 without a fresh
//!    file.**  The skip-mode harness ships with a documented run path
//!    (env vars + two cached GGUFs).  Iter-210's task is to flesh out
//!    the body now that the chain is end-to-end testable.
//!
//! # Scopes (matches `tests/auto_pipeline_smoke.rs` pattern)
//!
//! 1. **Default**: skip with a diagnostic.  Keeps `cargo test --release`
//!    cheap on dev machines.
//! 2. **`HF2Q_HOT_SWAP_E2E=1` + `HF2Q_HOT_SWAP_E2E_MODEL_A` +
//!    `HF2Q_HOT_SWAP_E2E_MODEL_B`**: runs the full subprocess
//!    swap-timing harness once iter-209 lands.  This iter-208 ships
//!    only the scaffolding (verifies the binary boots; documents
//!    the closure target).
//!
//! ```bash
//! cargo test --release --test multi_model_hotswap
//!   # → all skipped, exit 0
//!
//! HF2Q_HOT_SWAP_E2E=1 \
//!   HF2Q_HOT_SWAP_E2E_MODEL_A=/path/to/model-a.gguf \
//!   HF2Q_HOT_SWAP_E2E_MODEL_B=/path/to/model-b.gguf \
//!   cargo test --release --test multi_model_hotswap -- --test-threads=1
//!   # → iter-210 will flesh out the spawn + swap + assert body here
//! ```

use std::path::PathBuf;
use std::process::Command;

const ENV_GATE: &str = "HF2Q_HOT_SWAP_E2E";
const ENV_MODEL_A: &str = "HF2Q_HOT_SWAP_E2E_MODEL_A";
const ENV_MODEL_B: &str = "HF2Q_HOT_SWAP_E2E_MODEL_B";

fn skip_unless_gated(name: &str) -> bool {
    if std::env::var(ENV_GATE).as_deref() == Ok("1") {
        return false;
    }
    eprintln!(
        "[skip] {name} — set {ENV_GATE}=1 (plus {ENV_MODEL_A} + {ENV_MODEL_B} \
         pointing at two cached GGUFs) to run the iter-210 hot-swap E2E harness"
    );
    true
}

/// Locate the `hf2q` binary the cargo test runner just built.  Same
/// trick `tests/auto_pipeline_smoke.rs` and other integration suites
/// use: `target/release/hf2q` is the binary cargo just produced for
/// the parent test run.
fn hf2q_binary_path() -> PathBuf {
    // CARGO_TARGET_DIR overrides; otherwise default to ./target/release.
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
/// known-good entry point.  Skipping this would let scaffolding bit-rot
/// silently between iters.
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
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.to_lowercase().contains("hf2q"),
        "expected `hf2q` in --version output, got: {stdout}"
    );
}

/// AC 5356 measurement harness — DEFERRED to iter-210.
///
/// Body is intentionally empty in iter-208: the full chain
/// (HotSwapManager → AppState → /v1/chat/completions) is not
/// end-to-end testable until iter-209 lands AppState integration.
/// iter-210 will:
///
///   1. Verify both env-pointed GGUFs exist + parse cleanly.
///   2. Spawn `hf2q serve --model <MODEL_A>` on a free port.
///   3. POST `/v1/chat/completions` with `model: <MODEL_B's repo-id>` —
///      with iter-209's auto-swap, the manager loads B and serves.
///   4. POST `/v1/chat/completions` with `model: <MODEL_A's repo-id>` —
///      verifies A is still pooled (or re-loaded if evicted).
///   5. Measure the swap wall-clock and assert under the AC budget.
///
/// In-process E2E (iter-209/210 alternative) using
/// `DefaultModelLoader` against two real GGUFs requires hf2q to ship a
/// `[lib]` target; today's binary crate forces the subprocess path.
#[test]
fn hotswap_two_cached_ggufs_e2e() {
    if skip_unless_gated("hotswap_two_cached_ggufs_e2e") {
        return;
    }
    let model_a = match std::env::var(ENV_MODEL_A) {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "[skip] hotswap_two_cached_ggufs_e2e — {ENV_MODEL_A} not set"
            );
            return;
        }
    };
    let model_b = match std::env::var(ENV_MODEL_B) {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!(
                "[skip] hotswap_two_cached_ggufs_e2e — {ENV_MODEL_B} not set"
            );
            return;
        }
    };
    assert!(
        model_a.exists(),
        "{ENV_MODEL_A} points at non-existent path: {}",
        model_a.display()
    );
    assert!(
        model_b.exists(),
        "{ENV_MODEL_B} points at non-existent path: {}",
        model_b.display()
    );
    // iter-208 (W76) closure: confirm the two models are present and
    // the binary can boot one of them.  Iter-210 (W78+) will replace
    // this stub with the full swap-timing measurement.
    eprintln!(
        "[stub] hotswap_two_cached_ggufs_e2e — iter-208 scaffolding only. \
         Iter-210 will flesh out the full subprocess swap-timing harness. \
         MODEL_A={}, MODEL_B={}",
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
