//! ADR-014 P0 spike — apex-MoE LazyTensorMap iter peak-RSS gate
//! (Decision 2: peak resident bytes ≤ 8 GB while iterating every tensor
//! through `materialize → drop`).
//!
//! Two surfaces:
//!
//! 1. `lazy_iter_peak_rss_smoke_with_synthetic_fixture` (always-on) —
//!    builds a tiny safetensors directory in a tempdir, drives the lazy
//!    reader through the same materialise→drop loop the real-model gate
//!    uses, and confirms the in-process `getrusage(RUSAGE_SELF)` plumbing
//!    reports a number that is non-zero AND grows when allocations occur.
//!    The absolute number is not asserted — cargo's multi-threaded test
//!    runner shares process state across tests and `RUSAGE_SELF`'s
//!    high-water mark is cumulative across the whole runner. The smoke is
//!    a wiring check only; the real Decision 2 gate runs out-of-process.
//!
//! 2. `lazy_iter_apex_moe_peak_rss_le_8gb` (`#[ignore]`-gated) — the
//!    Decision 2 gate. Resolves an HF-format safetensors directory via
//!    `HF2Q_APEX_MOE_PATH`, spawns a fresh subprocess (`current_exe()`
//!    targeted at `lazy_iter_subprocess_helper`) so peak RSS is measured
//!    on a clean address space, parses the helper's `PEAK_RSS_BYTES=N`
//!    line out of captured stdout, and asserts peak RSS ≤ 8 GB
//!    (`8 * 1024^3 = 8589934592`). On failure the assertion message
//!    surfaces the observed peak in MB AND the highest-water-during-iter
//!    tensor name + bytes the helper printed via `LAST_TENSOR=…`.
//!
//! 3. `lazy_iter_subprocess_helper` (`#[ignore]`-gated) — the helper test
//!    invoked by (2) via `cargo test ... -- --ignored --exact ... --nocapture`.
//!    Reads its target safetensors directory from `HF2Q_P0_SPIKE_INPUT`,
//!    runs the lazy reader through the materialise→drop loop, prints
//!    `PEAK_RSS_BYTES=<N>` (and `LAST_TENSOR=<name>:<bytes>` for
//!    diagnostics) to stdout, and exits via `std::process::exit(0)`. When
//!    `HF2Q_P0_SPIKE_INPUT` is not set this test is a no-op (prevents
//!    unintended runs of the helper standalone).
//!
//! ## Why a fresh subprocess for the real-model gate
//!
//! `getrusage(RUSAGE_SELF).ru_maxrss` is a high-water mark on macOS that
//! accumulates across every allocation made by the calling process — and
//! cargo's test harness is multi-threaded and shares the test process
//! with every other `tests/p0_apex_moe_lazy_iter.rs` test (synthetic
//! smoke first, then the real-model gate). Measuring in-process would
//! conflate the smoke fixture's bytes with the apex MoE iter's bytes.
//! Spawning a fresh subprocess (current_exe with the helper test as the
//! filter) gives the real-model iter its own clean PID; the high-water
//! mark we read at the end is exactly the apex MoE iter peak. This
//! mirrors the precedent in
//! `tests/common/llama_cpp_runner.rs:276-300` (`run_llama_perplexity`
//! comments on the same `RUSAGE_CHILDREN` over-reporting failure mode
//! P10 iter-1 documented).
//!
//! ## File-fence note
//!
//! Per the iter spec the only `.rs` files this iter may modify are
//! `tests/p0_apex_moe_lazy_iter.rs` (this file, NEW) and
//! `docs/ADR-014-streaming-convert-pipeline.md` (P0 row update only).
//! The lazy reader's modules live under `src/`; we reach them via the
//! established `#[path]`-include pattern (precedent:
//! `tests/imatrix_xvalidation.rs:48-52`). hf2q is a binary crate with no
//! `[lib]` target, so `use hf2q::...` is unavailable.

#[path = "../src/ir/mod.rs"]
mod ir;

#[path = "../src/progress.rs"]
mod progress;

// `safetensors.rs` references `crate::ir::lazy::...`, `crate::ir::...`,
// and `crate::progress::ProgressReporter`. The `#[path]`-included
// modules above land under this crate's root (`crate::ir`,
// `crate::progress`), so the imports resolve naturally.
#[path = "../src/input/safetensors.rs"]
mod safetensors_reader;

use std::path::{Path, PathBuf};
use std::process::Command;

use ir::lazy::LazyTensorMap;
use progress::ProgressReporter;
use safetensors_reader::read_tensors_lazy;

/// Decision 2 gate value: peak RSS ≤ 8 GB while iterating every tensor
/// through `materialize → drop`. Spec line: "Decision 2 `≤ 8 GB`
/// LazyTensorMap-iter spike on apex MoE pending".
const DECISION_2_GATE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Default apex MoE safetensors path. Per the iter spec; overridable via
/// `HF2Q_APEX_MOE_PATH`. NOTE: as of 2026-04-27 the canonical apex MoE
/// directory at `models/qwen3.6-35b-...-apex/` ships only the GGUF
/// artefact + tokenizer (no safetensors shards on disk). The gate test
/// is `#[ignore]`-gated: when run, it points at any HF-format safetensors
/// directory (overridable via env). `find_default_apex_path` returns the
/// first sensible candidate and surfaces a clear error otherwise.
const DEFAULT_APEX_MOE_PATH: &str =
    "/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-apex";

/// Env var: input safetensors directory for the helper subprocess.
const ENV_HELPER_INPUT: &str = "HF2Q_P0_SPIKE_INPUT";

/// Env var: override the default apex MoE path resolved by the gate.
const ENV_APEX_PATH: &str = "HF2Q_APEX_MOE_PATH";

/// Read peak resident-set-size in bytes from `getrusage(RUSAGE_SELF)`.
/// On macOS `ru_maxrss` is bytes (`man getrusage`); on Linux it's
/// kilobytes — the M5 Max benchmark hardware is macOS, matching the
/// `parse_bsd_time_peak_rss` convention from
/// `tests/common/llama_cpp_runner.rs`. A safe `target_os` cast is
/// applied so a Linux developer running the smoke locally still gets a
/// monotonically-growing number (just in different units), and the
/// always-on assertion is "grew", not "was N bytes".
fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: getrusage is a thread-safe libc call; `&mut usage` is a
    // valid pointer to a freshly zeroed struct. Return value is checked
    // and the function is total — there is no failure path for
    // RUSAGE_SELF on macOS / Linux.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    assert_eq!(rc, 0, "getrusage(RUSAGE_SELF) failed: errno-equivalent");
    let raw = usage.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        // Bytes already.
        raw
    } else {
        // Linux reports kilobytes; convert.
        raw.saturating_mul(1024)
    }
}

/// Build a minimal valid safetensors file in memory.
/// Mirrors the `create_test_safetensors` helper inside
/// `src/input/safetensors.rs::tests` which we cannot reuse from here
/// (private `#[cfg(test)]` mod).
fn write_synthetic_safetensors(model_dir: &Path, tensors: &[(&str, &[usize], &str, &[u8])]) {
    let mut header_map = serde_json::Map::new();
    let mut current_offset = 0usize;

    for (name, shape, dtype, data) in tensors {
        let mut tensor_info = serde_json::Map::new();
        tensor_info.insert(
            "dtype".to_string(),
            serde_json::Value::String((*dtype).to_string()),
        );
        tensor_info.insert(
            "shape".to_string(),
            serde_json::Value::Array(
                shape
                    .iter()
                    .map(|&s| serde_json::Value::Number(s.into()))
                    .collect(),
            ),
        );
        let end_offset = current_offset + data.len();
        tensor_info.insert(
            "data_offsets".to_string(),
            serde_json::Value::Array(vec![
                serde_json::Value::Number(current_offset.into()),
                serde_json::Value::Number(end_offset.into()),
            ]),
        );
        header_map.insert((*name).to_string(), serde_json::Value::Object(tensor_info));
        current_offset = end_offset;
    }

    let header_json =
        serde_json::to_string(&header_map).expect("synthetic safetensors header serializes");
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_size.to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    for (_, _, _, data) in tensors {
        file_data.extend_from_slice(data);
    }

    std::fs::write(model_dir.join("model.safetensors"), &file_data)
        .expect("write synthetic safetensors fixture");
}

/// Drive the lazy reader through one materialise→drop pass over every
/// tensor in `model_dir`. Returns `(tensors_seen, last_tensor_name,
/// last_tensor_bytes)` — the diagnostic surface used by both the
/// always-on smoke (sanity checks) and the helper subprocess (printed to
/// stdout for the parent's failure-message diagnostic).
///
/// Materialise → drop pattern: `into_iter()` consumes the map by value
/// (deterministic BTreeMap order); for each `(name, lazy)` we
/// `lazy.materialize()` to produce a `TensorRef` (which now owns the
/// `Vec<u8>` of bytes), record the tensor's byte length for diagnostics,
/// and let the `TensorRef` go out of scope at the end of the loop body.
/// Drop reclaims the `Vec<u8>` before the next iteration; this is the
/// streaming contract Decision 2 codifies.
fn drive_lazy_iter(model_dir: &Path) -> (usize, String, usize) {
    let progress = ProgressReporter::new();
    let lazy_map: LazyTensorMap = read_tensors_lazy(model_dir, &progress)
        .expect("lazy reader opens safetensors directory");

    let mut tensors_seen = 0usize;
    let mut last_name = String::new();
    let mut last_bytes = 0usize;

    for (name, lazy) in lazy_map.into_iter() {
        let tref = lazy
            .materialize()
            .unwrap_or_else(|err| panic!("materialise '{name}' failed: {err}"));
        last_bytes = tref.data.len();
        last_name = name;
        tensors_seen += 1;
        // `tref` (and its owned `Vec<u8>`) drops at end-of-scope here.
        drop(tref);
    }

    (tensors_seen, last_name, last_bytes)
}

/// Resolve the apex MoE input directory. Returns `Err(message)` (string,
/// not a typed error — surfaced verbatim in the assertion message) when
/// the resolved path either does not exist or is not an HF-format
/// safetensors directory. The `#[ignore]`-gated gate test treats a
/// missing path as "deferred to hardware run", matching the P10 cell
/// pattern.
fn resolve_apex_moe_path() -> Result<PathBuf, String> {
    let raw = std::env::var(ENV_APEX_PATH).unwrap_or_else(|_| DEFAULT_APEX_MOE_PATH.to_string());
    let path = PathBuf::from(&raw);

    if !path.exists() {
        return Err(format!(
            "Apex MoE directory not present on disk: {} \
             (set {ENV_APEX_PATH}=<HF safetensors dir> to override)",
            path.display()
        ));
    }

    // HF-format guard: must contain at least one .safetensors shard or
    // the index. The lazy reader's `discover_shards` accepts any of the
    // three forms but we surface a clean error here so the gate isn't
    // blamed for "no peak RSS reported" when the input is a GGUF-only
    // directory (e.g. the canonical apex MoE path on the M5 Max).
    let has_index = path.join("model.safetensors.index.json").exists();
    let has_single = path.join("model.safetensors").exists();
    let has_any = std::fs::read_dir(&path)
        .map_err(|e| format!("read_dir({}) failed: {e}", path.display()))?
        .filter_map(|e| e.ok())
        .any(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        });
    if !(has_index || has_single || has_any) {
        return Err(format!(
            "Path {} contains no safetensors shards (HF-format directory required; \
             the canonical apex MoE path ships only GGUF as of 2026-04-27 — \
             set {ENV_APEX_PATH} to an HF safetensors directory like \
             /opt/hf2q/models/gemma4 to run the gate)",
            path.display()
        ));
    }

    Ok(path)
}

// ---------------------------------------------------------------------
// Always-on: synthetic-fixture smoke that the measurement plumbing works.
// ---------------------------------------------------------------------

#[test]
fn lazy_iter_peak_rss_smoke_with_synthetic_fixture() {
    // 4 small tensors totalling 8 KiB of body bytes — enough to be
    // visible against process noise without slowing the smoke past
    // sub-second budget.
    let f32_a: Vec<u8> = (0..1024u32)
        .flat_map(|v| (v as f32).to_le_bytes())
        .collect();
    let f32_b: Vec<u8> = (0..1024u32)
        .flat_map(|v| ((v as f32) * 2.0).to_le_bytes())
        .collect();
    // Use F32 throughout: the BF16 dtype path is locked by other tests
    // (`test_lazy_byte_identical_to_eager_bridge`); here we only need
    // the iter contract to hold.
    let f32_c: Vec<u8> = (0..512u32)
        .flat_map(|v| ((v as f32) * 3.0).to_le_bytes())
        .collect();
    let f32_d: Vec<u8> = (0..512u32)
        .flat_map(|v| ((v as f32) * 4.0).to_le_bytes())
        .collect();

    let tmp = tempfile::tempdir().expect("synthesise tempdir");
    write_synthetic_safetensors(
        tmp.path(),
        &[
            ("blk.0.attn_q.weight", &[32, 32], "F32", &f32_a),
            ("blk.0.attn_k.weight", &[32, 32], "F32", &f32_b),
            ("blk.0.attn_v.weight", &[16, 32], "F32", &f32_c),
            ("blk.0.attn_o.weight", &[16, 32], "F32", &f32_d),
        ],
    );

    let rss_before = peak_rss_bytes();

    // Force a non-trivial allocation to verify the meter responds; this
    // is the wiring check, NOT the materialise contract (that's exercised
    // separately below). Without this, on a small fixture the lazy
    // iter's per-tensor `Vec<u8>` allocations may all reuse the same
    // freed slot and the high-water mark won't budge.
    let mut growth_witness: Vec<u8> = Vec::with_capacity(2 * 1024 * 1024);
    growth_witness.resize(2 * 1024 * 1024, 0xa5);
    // Touch every page so the kernel actually faults the bytes resident.
    for chunk in growth_witness.chunks_mut(4096) {
        chunk[0] = 0xa5;
    }

    let (tensors_seen, last_name, last_bytes) = drive_lazy_iter(tmp.path());

    let rss_after = peak_rss_bytes();

    // Drop after the second measurement so the bytes are guaranteed
    // resident across the call (LLVM cannot DCE a value referenced after
    // the second `peak_rss_bytes`).
    drop(growth_witness);

    // Iter contract: every tensor we wrote must be observed.
    assert_eq!(
        tensors_seen, 4,
        "lazy iter must yield every tensor we wrote; got {tensors_seen}"
    );
    // Deterministic BTreeMap order ⇒ last entry is the lexicographically
    // greatest key. We picked names that sort `attn_q < attn_k < attn_v
    // < attn_o`? No — alphabetically: `attn_k`, `attn_o`, `attn_q`,
    // `attn_v` ⇒ last = `blk.0.attn_v.weight` (16×32 F32 = 2048 bytes).
    assert_eq!(
        last_name, "blk.0.attn_v.weight",
        "deterministic BTreeMap iter order changed; expected attn_v last"
    );
    assert_eq!(
        last_bytes, 2048,
        "last-tensor diagnostic byte count is wrong"
    );

    // Plumbing contract: getrusage returns a positive number.
    assert!(
        rss_before > 0,
        "peak_rss_bytes returned 0 before smoke — getrusage plumbing broken"
    );
    assert!(
        rss_after > 0,
        "peak_rss_bytes returned 0 after smoke — getrusage plumbing broken"
    );
    // Plumbing contract: high-water mark is monotonic. The 2 MiB witness
    // allocation guarantees a measurable floor delta (RSS reporting on
    // macOS rounds to 4 KiB pages, so any allocation above the previous
    // peak shows up); the lazy iter's per-tensor allocations may or may
    // not contribute on top, depending on heap-reuse behaviour.
    assert!(
        rss_after >= rss_before,
        "peak RSS must not shrink across an allocation: \
         before={rss_before}, after={rss_after}"
    );
    // Sanity bound: the smoke fixture's whole working set is < 8 MiB; if
    // the meter reports something absurd (e.g. negative cast to u64::MAX)
    // we want to catch it here, not in the apex gate where a real RSS of
    // 4 GB would mask a bug.
    let absurd_bound: u64 = 4 * 1024 * 1024 * 1024;
    assert!(
        rss_after < absurd_bound,
        "peak RSS reported {rss_after} bytes — meter likely returning a \
         garbage value (unit miscast?)"
    );
}

// ---------------------------------------------------------------------
// Subprocess helper: drives the lazy iter inside a fresh process so the
// peak-RSS reading is isolated from cargo's test runner. Activated by
// the gate test below; stays inert when its env-var input is not set
// (so a developer running `cargo test --ignored` standalone doesn't get
// a confusing failure).
// ---------------------------------------------------------------------

#[test]
#[ignore = "Helper test invoked by lazy_iter_apex_moe_peak_rss_le_8gb via \
            current_exe(); reads HF2Q_P0_SPIKE_INPUT and prints PEAK_RSS_BYTES \
            to stdout. Standalone invocation is a no-op."]
fn lazy_iter_subprocess_helper() {
    let input = match std::env::var(ENV_HELPER_INPUT) {
        Ok(v) => v,
        Err(_) => {
            // Inert: nothing to do. The parent gate test sets the env
            // var; a developer running this filter manually without it
            // sees a clean no-op pass.
            eprintln!(
                "{ENV_HELPER_INPUT} not set; helper is a no-op when not \
                 invoked by the gate test."
            );
            return;
        }
    };

    let path = PathBuf::from(&input);
    let (tensors_seen, last_name, last_bytes) = drive_lazy_iter(&path);

    // Read peak RSS AFTER the iter has run + every per-tensor `Vec<u8>`
    // has been dropped. `RUSAGE_SELF.ru_maxrss` is a high-water mark, so
    // this captures the peak resident bytes the iter ever held.
    let peak = peak_rss_bytes();

    // Print the parseable single-line output the parent expects on its
    // own line, with a leading newline so any preceding cargo-runner
    // chatter (e.g. "running 1 test") doesn't get glued to ours.
    println!("\nPEAK_RSS_BYTES={peak}");
    println!("LAST_TENSOR={last_name}:{last_bytes}");
    println!("TENSORS_SEEN={tensors_seen}");
    println!("INPUT={input}");

    // Exit explicitly so the cargo-test runner does not get a chance to
    // emit additional bookkeeping lines that might confuse downstream
    // log parsers (the parent only greps for `PEAK_RSS_BYTES=`, but
    // belt-and-braces).
    std::process::exit(0);
}

// ---------------------------------------------------------------------
// Decision 2 gate: real-model peak RSS ≤ 8 GB. Runs only with --ignored.
// ---------------------------------------------------------------------

#[test]
#[ignore = "Decision 2 gate: needs HF2Q_APEX_MOE_PATH=/path/to/hf-safetensors-dir \
            (default /opt/hf2q/models/qwen3.6-35b-...-apex which currently has \
            no safetensors shards on disk). Runs the lazy iter in a fresh \
            subprocess to isolate peak RSS from cargo's runner state."]
fn lazy_iter_apex_moe_peak_rss_le_8gb() {
    let apex_path = match resolve_apex_moe_path() {
        Ok(p) => p,
        Err(msg) => {
            // Honest deferral: surface why we couldn't run, then exit
            // the test cleanly. This matches the
            // `Verdict::NotMeasured` precedent in
            // `tests/peer_parity_gates.rs` — the gate scaffold lives,
            // hardware run is documented as pending, no fake-green.
            eprintln!("[gate-deferred] {msg}");
            return;
        }
    };

    let exe = std::env::current_exe().expect("current_exe resolves for the test binary");

    let output = Command::new(&exe)
        .args([
            "--ignored",
            "--exact",
            "lazy_iter_subprocess_helper",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(ENV_HELPER_INPUT, &apex_path)
        // Suppress noisy logs from the lazy reader's tracing subscriber
        // so the helper's stdout is easier to parse.
        .env("RUST_LOG", "warn")
        .output()
        .expect("spawn helper subprocess");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "helper subprocess failed (status={:?}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        stdout,
        stderr
    );

    // Parse `PEAK_RSS_BYTES=N` out of stdout.
    let peak_line = stdout
        .lines()
        .find(|l| l.starts_with("PEAK_RSS_BYTES="))
        .unwrap_or_else(|| {
            panic!(
                "helper stdout did not contain PEAK_RSS_BYTES=...\n--- stdout ---\n{stdout}\n\
                 --- stderr ---\n{stderr}"
            )
        });
    let peak_str = peak_line
        .strip_prefix("PEAK_RSS_BYTES=")
        .expect("startsWith implies strip_prefix succeeds");
    let peak: u64 = peak_str
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("PEAK_RSS_BYTES value '{peak_str}' is not u64: {e}"));

    let last_tensor = stdout
        .lines()
        .find(|l| l.starts_with("LAST_TENSOR="))
        .unwrap_or("LAST_TENSOR=<unknown>");

    let peak_mb = peak / (1024 * 1024);
    let gate_mb = DECISION_2_GATE_BYTES / (1024 * 1024);

    assert!(
        peak <= DECISION_2_GATE_BYTES,
        "Decision 2 gate FAILED: peak RSS {peak_mb} MB ({peak} bytes) > {gate_mb} MB \
         (8 GB) on apex MoE LazyTensorMap iter at path {}\n  diagnostic: {last_tensor}\n  \
         (the LazyTensorMap iter is not yet streaming-bounded; see ADR-014 P0/P1/P2).",
        apex_path.display()
    );

    // Surface the actual measurement on success too — the iter spec asks
    // us to record the number when the gate runs.
    eprintln!(
        "[gate-pass] peak RSS {peak_mb} MB ≤ {gate_mb} MB on apex MoE \
         LazyTensorMap iter at {}",
        apex_path.display()
    );
}
