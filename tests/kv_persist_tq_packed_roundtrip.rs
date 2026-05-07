//! ADR-017 §B-tq.4 iter-3 + §B-tq.5 — TQ-packed end-to-end round-trip parity test.
//!
//! ## Scope
//!
//! Load-bearing acceptance test for B-tq.4 — verifies the
//! `TqPackedSpillFactory` registration in `cmd_serve` produces a
//! fully-wired `TqPackedSpill` on engine load, and that the
//! engine-bound `snapshot_block` / `restore_block` paths route
//! through `MlxModelWeights::tq_v2_snapshot_block` /
//! `tq_v2_restore_block` (commit `b7e975d`) and the K+V bundle
//! codec (iter-1, commit `62bb8b5`).
//!
//! ## Three layers of testing
//!
//! 1. **Always-on smokes** (run on every `cargo test`) — gate
//!    resolution + binary path resolution.
//! 2. **Env-gated E2E** — full subprocess round-trip via the
//!    shared `crate::common::serve_driver`: spawn A → POST →
//!    drain → wait-exit → spawn B against same cache_dir →
//!    POST same prompt → assert R-C1 byte-identical decode.
//! 3. **Tracing-log assertions** — verifies the cmd_serve startup
//!    line `factory=TqPackedSpillFactory` fires (proves
//!    iter-2's single-registration policy selected the TQ
//!    branch).
//!
//! ## Default-off
//!
//! E2E test fires only when:
//!
//!   * `HF2Q_KV_PERSIST_TQ_E2E=1` (master gate), AND
//!   * `HF2Q_KV_PERSIST_E2E_MODEL_PATH=/path/to/gemma4.gguf`.
//!
//! Without these, the test short-circuits with a diagnostic and
//! returns success (default-runner CI stays green).
//!
//! ## What ships with this iter (B-tq.5)
//!
//! `phase_d_driver` was extracted from `kv_persist_gemma4_roundtrip.rs`
//! to `tests/common/serve_driver.rs` so this test can drive the
//! live subprocess flow without copying ~700 LOC.  The gemma4
//! roundtrip's `mod phase_d_driver` is now a thin wrapper that
//! re-exports `crate::common::serve_driver::*` plus the
//! gemma4-specific peer-arm helpers — its 33 existing tests still
//! pass post-refactor.

#![allow(clippy::needless_range_loop)]

mod common;

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use common::serve_driver as driver;

// =========================================================================
// Env-isolation lock.
// =========================================================================
//
// Mirrors the gemma4 roundtrip's pattern.  All env-mutating tests in
// this binary acquire `ENV_LOCK` first to avoid concurrent set/remove
// races.
static ENV_LOCK: Mutex<()> = Mutex::new(());

// =========================================================================
// Env gates.
// =========================================================================

/// Master gate for the TQ-active E2E test.
const ENV_TQ_E2E_GATE: &str = "HF2Q_KV_PERSIST_TQ_E2E";

/// Model path for the E2E test.  Mirrors the gemma4 roundtrip's
/// env so a single operator config drives both binaries.
const ENV_MODEL_PATH: &str = "HF2Q_KV_PERSIST_E2E_MODEL_PATH";

/// Distinct port from gemma4's 52339 so concurrent test binaries
/// don't collide.
const TQ_E2E_PORT: u16 = 52341;

/// Deterministic prompt for R-C1.  Short + simple so the round-trip
/// completes in reasonable time on a cold M5 Max load (~120s
/// startup + ~5s decode).
const TQ_E2E_PROMPT: &str = "Reply with exactly: hello world";

/// Token budget.  Small — we want byte-identity, not long output.
const TQ_E2E_MAX_TOKENS: u32 = 16;

// =========================================================================
// Helpers.
// =========================================================================

fn resolve_e2e_model_path() -> Option<PathBuf> {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let gate = std::env::var(ENV_TQ_E2E_GATE).unwrap_or_default();
    if gate.trim() != "1" {
        return None;
    }
    let raw = std::env::var(ENV_MODEL_PATH).unwrap_or_else(|_| {
        panic!(
            "[B-tq.4 E2E] {ENV_TQ_E2E_GATE}=1 set but {ENV_MODEL_PATH} unset \
             (would silently skip — would defeat the load-bearing acceptance test)"
        )
    });
    let path = PathBuf::from(&raw);
    if !path.exists() {
        panic!("[B-tq.4 E2E] {ENV_MODEL_PATH}={raw} does not exist on disk");
    }
    Some(path)
}

fn hf2q_binary_path() -> PathBuf {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"));
    let release = workspace.join("target/release/hf2q");
    if release.exists() {
        return release;
    }
    workspace.join("target/debug/hf2q")
}

/// Recursively count `*.safetensors` block files in cache_dir.
/// `read_dir` is non-recursive — the cache_dir layout is
/// `$CACHE/models/<fp>/kv/<bucket>/<sha>.safetensors`, so a
/// top-level `read_dir` only sees `locks/` and `models/`.
fn walk_block_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                out.push(p);
            }
        }
    }
    out
}

fn cache_dir_for_run() -> PathBuf {
    std::env::temp_dir().join(format!(
        "hf2q-btq4-e2e-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

/// **Advisory** check that the cmd_serve startup tracing log carries
/// the `factory=TqPackedSpillFactory` line (proves iter-2 single-
/// registration policy selected the TQ branch).
///
/// **Why advisory not load-bearing**: the registration log fires at
/// INFO level inside `cmd_serve` at `mod.rs:3076+`.  Default
/// `tracing_subscriber` filter is WARN, so the line gets suppressed
/// unless the test sets `RUST_LOG=info` on the child env.  We pass
/// `RUST_LOG=info,hf2q=info` in the spawn args, but if that gets
/// filtered for any reason we don't want a test panic — the
/// load-bearing assertion is R-C1 byte-identity, not stderr-grep.
fn check_tq_factory_registered_advisory(server: &driver::ServerGuard) {
    let log = server.log_tail();
    let joined = log.join("\n");
    let saw_tq = joined.contains("factory=TqPackedSpillFactory")
        || joined.contains("\"TqPackedSpillFactory\"");
    let saw_dense =
        joined.contains("factory=Gemma4DenseSpillFactory") && !saw_tq;
    if saw_dense {
        // Dense factory registered when we asked for TQ — that's a
        // mis-substitute, fail loud.
        panic!(
            "[B-tq.4 E2E] startup log shows Gemma4DenseSpillFactory \
             but HF2Q_TQ_KV=1 — single-registration policy bug.\n\n\
             --- stderr_tail ({} lines) ---\n{}",
            log.len(),
            joined,
        );
    }
    if !saw_tq {
        // INFO line may be filtered; downgrade to advisory.
        eprintln!(
            "[B-tq.4 E2E] advisory: stderr_tail did not contain \
             `factory=TqPackedSpillFactory` — likely tracing-filter \
             suppression at INFO level.  R-C1 byte-identity assertion \
             remains the load-bearing gate.\n\n\
             --- stderr_tail ({} lines, last 20) ---\n{}",
            log.len(),
            log.iter().rev().take(20).rev().cloned().collect::<Vec<_>>().join("\n"),
        );
    } else {
        eprintln!("[B-tq.4 E2E] factory=TqPackedSpillFactory confirmed in stderr");
    }
}

// =========================================================================
// Always-on smokes.
// =========================================================================

#[test]
fn kv_persist_tq_packed_e2e_gate_unset_returns_none() {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let prior_gate = std::env::var(ENV_TQ_E2E_GATE).ok();
    let prior_path = std::env::var(ENV_MODEL_PATH).ok();
    // SAFETY: lock held.
    unsafe { std::env::remove_var(ENV_TQ_E2E_GATE) };
    unsafe { std::env::remove_var(ENV_MODEL_PATH) };
    drop(_guard);

    assert!(
        resolve_e2e_model_path().is_none(),
        "gate resolution must return None when master gate unset"
    );

    let _restore_guard = ENV_LOCK.lock().expect("env lock");
    if let Some(v) = prior_gate {
        unsafe { std::env::set_var(ENV_TQ_E2E_GATE, v) };
    }
    if let Some(v) = prior_path {
        unsafe { std::env::set_var(ENV_MODEL_PATH, v) };
    }
}

#[test]
fn kv_persist_tq_packed_binary_path_resolves() {
    let bin = hf2q_binary_path();
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"));
    assert!(bin.starts_with(workspace));
    let s = bin.to_string_lossy();
    assert!(
        s.ends_with("target/release/hf2q") || s.ends_with("target/debug/hf2q"),
        "binary path {} should end in target/release/hf2q or target/debug/hf2q",
        bin.display()
    );
}

#[test]
fn kv_persist_tq_packed_shared_driver_constants_resolve() {
    // Smoke: verify the shared driver's port constant is something
    // other than ours so cross-binary collisions don't happen.
    assert_ne!(driver::PORT_DEFAULT, TQ_E2E_PORT);
    // And the readyz budget is sensible.
    assert!(driver::READYZ_BUDGET_SECS >= 60);
}

// =========================================================================
// Env-gated E2E test (B-tq.4 iter-3).
// =========================================================================

/// **B-tq.4 E2E acceptance test** — load-bearing R-C1 byte-identity
/// across cross-process snapshot/restore.
///
/// Default-off: short-circuits unless `HF2Q_KV_PERSIST_TQ_E2E=1` AND
/// `HF2Q_KV_PERSIST_E2E_MODEL_PATH=PATH` are set.
///
/// Sequence (when active):
///   1. Spawn server A with `HF2Q_TQ_KV=1` + `--kv-persist=$CACHE`.
///   2. Wait `/readyz`.
///   3. Assert factory selection log shows `TqPackedSpillFactory`.
///   4. Fetch canonical model id, POST chat completion (round 1) at
///      `temperature=0`, capture decoded text.
///   5. POST `/shutdown` to drain KV blocks to `$CACHE`.
///   6. Wait for graceful exit.
///   7. Spawn server B against the SAME `$CACHE` + same model.
///   8. Wait `/readyz`.
///   9. POST the SAME prompt; capture decoded text.
///  10. Assert the two capture texts are byte-identical (R-C1).
///
/// **Falsifier**: any byte-diff between captures → R-C1 FAIL →
/// either factory wiring is wrong, or the codec isn't byte-stable
/// across snapshot/restore, or the bundle codec is dropping bytes.
#[test]
fn kv_persist_tq_packed_b_tq_4_e2e() {
    let model_path = match resolve_e2e_model_path() {
        Some(p) => p,
        None => {
            eprintln!(
                "[B-tq.4 E2E] {ENV_TQ_E2E_GATE} not set — short-circuit. \
                 Set {ENV_TQ_E2E_GATE}=1 + {ENV_MODEL_PATH}=PATH to run."
            );
            return;
        }
    };
    let bin = hf2q_binary_path();
    assert!(
        bin.exists(),
        "[B-tq.4 E2E] hf2q binary not found at {} — run `cargo build --release`",
        bin.display()
    );
    let cache_dir = cache_dir_for_run();
    std::fs::create_dir_all(&cache_dir).expect("mkdir cache_dir");
    eprintln!(
        "[B-tq.4 E2E] gates green: model={}, bin={}, cache={}",
        model_path.display(),
        bin.display(),
        cache_dir.display(),
    );

    let extra_env: &[(&str, &str)] = &[
        ("HF2Q_TQ_KV", "1"),
        // Pin codebook bits at production default for reproducibility.
        // **B-tq.7** — pin at 8-bit (production default per ADR-007
        // §1234).  At bits >= 5, `tq_v2_snapshot_block` reads from
        // `leg_hb_encoded[layer]` (the HB-packed cache that 8-bit
        // SDPA actually uses).  At bits == 4 it falls back to
        // `kv_caches[layer]` (the legacy 4-bit nibble cache).
        ("HF2Q_TQ_CODEBOOK_BITS", "8"),
        // Lift tracing filter to INFO so the cmd_serve registration
        // log line at mod.rs:3076+ is visible in stderr (see
        // `check_tq_factory_registered_advisory`).
        ("RUST_LOG", "info,hf2q=info"),
    ];

    // ----------------------------------------------------------------
    // ROUND 1 — server A.
    // ----------------------------------------------------------------
    let server_a = driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        driver::HOST,
        TQ_E2E_PORT,
        extra_env,
    )
    .expect("[B-tq.4 E2E] spawn server A");
    driver::wait_for_readyz(&server_a).unwrap_or_else(|e| {
        panic!(
            "[B-tq.4 E2E] server A /readyz did not return 200: {e}\n\
             --- stderr_tail ({} lines) ---\n{}",
            server_a.log_tail().len(),
            server_a.log_tail().join("\n"),
        )
    });
    check_tq_factory_registered_advisory(&server_a);
    let canonical = driver::fetch_canonical_model_id(&server_a)
        .expect("[B-tq.4 E2E] fetch canonical model id (A)");
    eprintln!(
        "[B-tq.4 E2E] server A ready on {}:{} model={}",
        driver::HOST,
        TQ_E2E_PORT,
        canonical,
    );

    let capture_a = driver::decode_full_text(
        &server_a,
        &canonical,
        TQ_E2E_PROMPT,
        TQ_E2E_MAX_TOKENS,
    )
    .expect("[B-tq.4 E2E] decode round 1");
    eprintln!(
        "[B-tq.4 E2E] capture_a: {} bytes ({} tokens, ttft={:.1}ms)",
        capture_a.text.len(),
        capture_a.total_tokens,
        capture_a.ttft_ms,
    );
    assert!(
        !capture_a.text.is_empty(),
        "[B-tq.4 E2E] server A returned 0-byte SSE stream"
    );

    // Drain + shutdown.
    let _ = driver::trigger_graceful_shutdown(&server_a)
        .expect("[B-tq.4 E2E] POST /shutdown on A");
    let mut server_a = server_a;
    let exit_status = driver::wait_for_graceful_exit(
        &mut server_a,
        Duration::from_secs(60),
    )
    .expect("[B-tq.4 E2E] server A graceful exit");
    eprintln!(
        "[B-tq.4 E2E] server A exited cleanly: status={:?}",
        exit_status
    );

    // ----------------------------------------------------------------
    // Verify cache_dir has block files (proves KV blocks were
    // flushed by drain_loaded_models_to_disk).
    // ----------------------------------------------------------------
    let cached_files: Vec<PathBuf> = walk_block_files(&cache_dir);
    assert!(
        !cached_files.is_empty(),
        "[B-tq.4 E2E] cache_dir empty after drain — KV blocks were not \
         flushed.  Either the spiller didn't fire (factory mis-substitute) \
         or pre_evict skipped this layer."
    );
    eprintln!(
        "[B-tq.4 E2E] cache_dir post-drain: {} entries",
        cached_files.len()
    );

    // ----------------------------------------------------------------
    // ROUND 2 — server B against the SAME cache_dir.
    // ----------------------------------------------------------------
    let server_b = driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        driver::HOST,
        TQ_E2E_PORT,
        extra_env,
    )
    .expect("[B-tq.4 E2E] spawn server B");
    driver::wait_for_readyz(&server_b).unwrap_or_else(|e| {
        panic!(
            "[B-tq.4 E2E] server B /readyz did not return 200: {e}\n\
             --- stderr_tail ({} lines) ---\n{}",
            server_b.log_tail().len(),
            server_b.log_tail().join("\n"),
        )
    });
    check_tq_factory_registered_advisory(&server_b);
    let canonical_b = driver::fetch_canonical_model_id(&server_b)
        .expect("[B-tq.4 E2E] fetch canonical model id (B)");
    assert_eq!(
        canonical, canonical_b,
        "[B-tq.4 E2E] canonical model id changed across processes — \
         sanity check failed before R-C1 comparison"
    );
    eprintln!("[B-tq.4 E2E] server B ready, replaying prompt");

    let capture_b = driver::decode_full_text(
        &server_b,
        &canonical_b,
        TQ_E2E_PROMPT,
        TQ_E2E_MAX_TOKENS,
    )
    .expect("[B-tq.4 E2E] decode round 2");
    eprintln!(
        "[B-tq.4 E2E] capture_b: {} bytes ({} tokens, ttft={:.1}ms)",
        capture_b.text.len(),
        capture_b.total_tokens,
        capture_b.ttft_ms,
    );

    // Drain server B before the assert so the cache_dir is left
    // clean even if the assert panics.  ServerGuard::Drop kills if
    // we don't drain; that's a backup, but graceful drain is
    // preferred.
    let _ = driver::trigger_graceful_shutdown(&server_b);
    let mut server_b = server_b;
    let _ = driver::wait_for_graceful_exit(&mut server_b, Duration::from_secs(60));

    // ----------------------------------------------------------------
    // R-C1 ASSERT — byte-identical decode across processes.
    // ----------------------------------------------------------------
    if capture_a.text != capture_b.text {
        let common = capture_a
            .text
            .as_bytes()
            .iter()
            .zip(capture_b.text.as_bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let snippet_a = capture_a
            .text
            .get(common..common.saturating_add(120))
            .unwrap_or("")
            .to_string();
        let snippet_b = capture_b
            .text
            .get(common..common.saturating_add(120))
            .unwrap_or("")
            .to_string();
        panic!(
            "[B-tq.4 E2E] R-C1 FAIL: capture_a != capture_b \
             (diverge at byte offset {}; A={} bytes, B={} bytes)\n\
             A @ {}: {:?}\nB @ {}: {:?}",
            common,
            capture_a.text.len(),
            capture_b.text.len(),
            common,
            snippet_a,
            common,
            snippet_b,
        );
    }
    eprintln!(
        "[B-tq.4 E2E] R-C1 PASS — capture_a ({} bytes) == capture_b ({} bytes) \
         BYTE-IDENTICAL across cross-process snapshot+restore",
        capture_a.text.len(),
        capture_b.text.len(),
    );

    // Cleanup cache_dir on success (panic path leaves it for debugging).
    let _ = std::fs::remove_dir_all(&cache_dir);
}

// =========================================================================
// B-tq.4 perf benchmark — long-prompt cross-process restore speedup.
// =========================================================================

const ENV_TQ_PERF_GATE: &str = "HF2Q_KV_PERSIST_TQ_PERF";

/// **B-tq.4 perf benchmark**.  Same flow as
/// `kv_persist_tq_packed_b_tq_4_e2e` but with a longer prompt
/// (~1000 tokens) to exercise multiple block boundaries + measure
/// the cross-process restore TTFT speedup.
///
/// Reports `TTFT_cold` (server A) vs `TTFT_warm` (server B) and the
/// ratio `TTFT_warm / TTFT_cold`.  R-P5 spec for the dense path is
/// `≤ 0.15` (6.7× speedup or better); the TQ-active path's
/// ratio is informational at this iter (operator-driven gate-
/// validation work).
///
/// Default-off: fires only with `HF2Q_KV_PERSIST_TQ_PERF=1` +
/// `HF2Q_KV_PERSIST_E2E_MODEL_PATH=PATH`.
#[test]
fn kv_persist_tq_packed_b_tq_4_long_prompt_perf() {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let perf_gate = std::env::var(ENV_TQ_PERF_GATE).unwrap_or_default();
    drop(_guard);
    if perf_gate.trim() != "1" {
        eprintln!(
            "[B-tq.4 perf] {ENV_TQ_PERF_GATE} not set — short-circuit. \
             Set {ENV_TQ_PERF_GATE}=1 + {ENV_MODEL_PATH}=PATH to run."
        );
        return;
    }
    let model_path = match resolve_e2e_model_path() {
        Some(p) => p,
        None => {
            // Master gate set but model path missing — defer to
            // resolve_e2e_model_path's panic.  This branch only fires
            // if HF2Q_KV_PERSIST_TQ_E2E was unset BUT
            // HF2Q_KV_PERSIST_TQ_PERF was set.  Honor the perf gate
            // independently.
            let _g = ENV_LOCK.lock().expect("env lock");
            let raw = std::env::var(ENV_MODEL_PATH).unwrap_or_else(|_| {
                panic!(
                    "[B-tq.4 perf] {ENV_TQ_PERF_GATE}=1 set but \
                     {ENV_MODEL_PATH} unset"
                )
            });
            let p = PathBuf::from(&raw);
            if !p.exists() {
                panic!("[B-tq.4 perf] {ENV_MODEL_PATH}={raw} does not exist");
            }
            p
        }
    };
    let bin = hf2q_binary_path();
    assert!(bin.exists(), "[B-tq.4 perf] hf2q binary missing");
    let cache_dir = cache_dir_for_run();
    std::fs::create_dir_all(&cache_dir).expect("mkdir cache_dir");

    // Build a long prompt — repeated paragraph that tokenizes to
    // ~1000+ tokens for typical Gemma 4 vocab.  The exact token
    // count depends on tokenizer; the spec is "long enough to span
    // multiple BLOCK_TOKENS=256 boundaries", which 1000+ tokens
    // achieves comfortably.
    let mut long_prompt = String::with_capacity(8192);
    long_prompt.push_str(
        "Below is a long technical passage that should produce \
         approximately one thousand tokens worth of context for the \
         model to process during prefill.  We repeat the same \
         paragraph to control variability and avoid tokenizer \
         vocabulary edge cases.  ",
    );
    for _ in 0..14 {
        long_prompt.push_str(
            "The TurboQuant codec applies a fixed orthogonal \
             transform via Walsh-Hadamard rotation followed by \
             Lloyd-Max quantization at eight bits per coordinate.  \
             Centroid tables are baked into the binary at compile \
             time; rotation is a deterministic O(d log d) operation.  \
             Per-block magnitudes are stored in F32 alongside the \
             packed indices to enable byte-exact decode.  "
        );
    }
    long_prompt.push_str("Please respond with a one-sentence summary.");

    let extra_env: &[(&str, &str)] = &[
        ("HF2Q_TQ_KV", "1"),
        // **B-tq.7** — pin at 8-bit (production default per ADR-007
        // §1234).  At bits >= 5, `tq_v2_snapshot_block` reads from
        // `leg_hb_encoded[layer]` (the HB-packed cache that 8-bit
        // SDPA actually uses).  At bits == 4 it falls back to
        // `kv_caches[layer]` (the legacy 4-bit nibble cache).
        ("HF2Q_TQ_CODEBOOK_BITS", "8"),
        ("RUST_LOG", "info,hf2q=info"),
    ];

    eprintln!(
        "[B-tq.4 perf] long_prompt = {} chars; spawning server A...",
        long_prompt.len()
    );

    // ROUND 1: cold load + prefill (no cache).
    let server_a = driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        driver::HOST,
        TQ_E2E_PORT,
        extra_env,
    )
    .expect("[B-tq.4 perf] spawn A");
    driver::wait_for_readyz(&server_a)
        .expect("[B-tq.4 perf] server A /readyz");
    check_tq_factory_registered_advisory(&server_a);
    let canonical = driver::fetch_canonical_model_id(&server_a)
        .expect("[B-tq.4 perf] fetch model id A");
    let cold = driver::decode_full_text(&server_a, &canonical, &long_prompt, 32)
        .expect("[B-tq.4 perf] decode round 1 (cold)");
    eprintln!(
        "[B-tq.4 perf] COLD: ttft_cold={:.1}ms, prompt_tokens={:?}, \
         output_bytes={}, output_tokens={}",
        cold.ttft_ms,
        cold.prompt_tokens,
        cold.text.len(),
        cold.total_tokens,
    );

    let _ = driver::trigger_graceful_shutdown(&server_a);
    let mut server_a = server_a;
    let _ = driver::wait_for_graceful_exit(&mut server_a, Duration::from_secs(60));

    // Verify cache_dir population.
    let cached_files: Vec<PathBuf> = walk_block_files(&cache_dir);
    eprintln!(
        "[B-tq.4 perf] cache_dir post-drain: {} block files",
        cached_files.len()
    );

    // ROUND 2: cold load + warm restore from cache.
    let server_b = driver::spawn_hf2q_serve_with_kv_persist(
        &bin,
        &model_path,
        &cache_dir,
        driver::HOST,
        TQ_E2E_PORT,
        extra_env,
    )
    .expect("[B-tq.4 perf] spawn B");
    driver::wait_for_readyz(&server_b)
        .expect("[B-tq.4 perf] server B /readyz");
    check_tq_factory_registered_advisory(&server_b);
    let canonical_b = driver::fetch_canonical_model_id(&server_b)
        .expect("[B-tq.4 perf] fetch model id B");
    let warm = driver::decode_full_text(&server_b, &canonical_b, &long_prompt, 32)
        .expect("[B-tq.4 perf] decode round 2 (warm)");
    eprintln!(
        "[B-tq.4 perf] WARM: ttft_warm={:.1}ms, prompt_tokens={:?}, \
         output_bytes={}, output_tokens={}",
        warm.ttft_ms,
        warm.prompt_tokens,
        warm.text.len(),
        warm.total_tokens,
    );

    let _ = driver::trigger_graceful_shutdown(&server_b);
    let mut server_b = server_b;
    let _ = driver::wait_for_graceful_exit(&mut server_b, Duration::from_secs(60));

    // Report ratio.
    let ratio = warm.ttft_ms / cold.ttft_ms;
    eprintln!("[B-tq.4 perf] TTFT_warm / TTFT_cold = {:.4}", ratio);
    eprintln!(
        "[B-tq.4 perf] R-P5 dense spec is ≤0.15 (6.7× speedup); \
         TQ-active is informational at this iter."
    );

    // R-C1 still applies — the byte-identity gate.  Even on a long
    // prompt, the decode at temperature=0 should produce identical
    // output across processes IF the cache restore is byte-stable.
    if cold.text != warm.text {
        let common = cold
            .text
            .as_bytes()
            .iter()
            .zip(warm.text.as_bytes())
            .take_while(|(a, b)| a == b)
            .count();
        panic!(
            "[B-tq.4 perf] R-C1 FAIL on long prompt: cold != warm \
             (diverge at byte offset {}; cold={} bytes, warm={} bytes)",
            common,
            cold.text.len(),
            warm.text.len()
        );
    }
    eprintln!(
        "[B-tq.4 perf] R-C1 PASS on long prompt — {} bytes byte-identical",
        cold.text.len()
    );

    let _ = std::fs::remove_dir_all(&cache_dir);
}
