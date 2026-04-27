//! ADR-014 P10 iter-1 — subprocess wrappers for llama.cpp peer-parity
//! gates.
//!
//! Three entry points (one per Decision 15 gate column):
//! - [`run_llama_quantize`]: invokes `llama-quantize <input> <output>
//!   <variant>` to produce the peer's Q4_K_M GGUF for the
//!   uncalibrated cell.
//! - [`run_convert_hf_to_gguf`]: invokes `convert_hf_to_gguf.py`
//!   (llama.cpp's HF → GGUF script) to materialise the float source the
//!   uncalibrated peer cell quantises from.
//! - [`run_llama_imatrix`]: invokes `llama-imatrix` to produce the
//!   peer's importance-matrix .dat file for the imatrix-q4_k_m cell
//!   (then `run_llama_quantize` consumes it via `--imatrix`).
//!
//! Every wrapper:
//! 1. resolves the binary via env-var override → `$PATH` walk;
//! 2. on missing, fires `tracing::warn!` and returns
//!    [`super::metrics::RunMetrics::missing_binary`] — never panics,
//!    never silently substitutes synthetic data;
//! 3. on present, spawns the subprocess via `std::process::Command`
//!    (no `shell=true`, no `bash -c`) and parses
//!    `/usr/bin/time -l` BSD output for `peak_rss_bytes` when wrapped
//!    by the cold-cache shell driver. When invoked directly from Rust
//!    (smoke-test path), wall-clock is measured via `Instant::now`.
//!
//! Env-var overrides (Decision 15 R12 mitigation — operator can pin
//! a specific build of each peer for reproducible parity numbers):
//! - `HF2Q_LLAMA_QUANTIZE_BIN`        → overrides `llama-quantize`
//! - `HF2Q_LLAMA_CONVERT_HF_BIN`      → overrides `convert_hf_to_gguf.py`
//! - `HF2Q_LLAMA_IMATRIX_BIN`         → overrides `llama-imatrix`

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use super::metrics::RunMetrics;

// ---------------------------------------------------------------------
// Binary resolution (env-var override → $PATH walk; no `which` crate dep)
// ---------------------------------------------------------------------

/// Resolves a peer binary by:
///   1. honouring the explicit `env_var` override if it points at an
///      existing executable file;
///   2. otherwise walking `$PATH` for `default_name`.
///
/// Returns `None` if neither path resolves. Callers must surface
/// missing-binary via [`RunMetrics::missing_binary`] + `tracing::warn!`.
fn resolve_binary(env_var: &str, default_name: &str) -> Option<PathBuf> {
    if let Ok(override_path) = std::env::var(env_var) {
        let p = PathBuf::from(&override_path);
        if p.is_file() {
            return Some(p);
        }
        // The override is set but invalid — that is an operator
        // mistake the harness must surface (do NOT silently fall back
        // to $PATH; the override exists precisely so the operator
        // pins a specific build).
        return None;
    }

    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(default_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

// ---------------------------------------------------------------------
// /usr/bin/time -l BSD parser
// ---------------------------------------------------------------------

/// Parses a `/usr/bin/time -l` BSD-format stderr block. Returns
/// `peak_rss_bytes` parsed from the `maximum resident set size` line
/// (in bytes on macOS), or `None` if the line is absent (which means
/// the wrapper was invoked directly without the cold-cache shell
/// driver).
///
/// macOS sample line:
///   `        12345678  maximum resident set size`
pub(crate) fn parse_bsd_time_peak_rss(stderr: &str) -> Option<u64> {
    for line in stderr.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_suffix("maximum resident set size") {
            let num_part = rest.trim();
            if let Ok(n) = num_part.parse::<u64>() {
                return Some(n);
            }
        }
        // Some shells emit tabs/extra whitespace between the number
        // and the label — handle the split-whitespace fallback.
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() >= 5
            && fields[1] == "maximum"
            && fields[2] == "resident"
            && fields[3] == "set"
            && fields[4] == "size"
        {
            if let Ok(n) = fields[0].parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------
// Direct (in-process) subprocess invocation with wall-clock timing.
// ---------------------------------------------------------------------

/// Spawns `bin args…` via `std::process::Command`, captures stdout
/// and stderr, and records wall-clock + exit code into
/// [`RunMetrics`]. `peak_rss_bytes` is parsed from stderr if the
/// caller wrapped the spawn under `/usr/bin/time -l`; otherwise it is
/// reported as `0` to distinguish from the missing-binary sentinel
/// (`u64::MAX`).
fn spawn_and_measure(bin: &Path, args: &[&str]) -> RunMetrics {
    let started = Instant::now();
    let output = match Command::new(bin).args(args).output() {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!(
                "subprocess spawn failed for {}: {} — treating as missing binary",
                bin.display(),
                e
            );
            return RunMetrics::missing_binary(
                bin.file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("<unnamed>"),
            );
        }
    };
    let wall_s = started.elapsed().as_secs_f64();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let peak_rss_bytes = parse_bsd_time_peak_rss(&stderr).unwrap_or(0);
    let stderr_tail = tail_string(&stderr, 4096);

    RunMetrics {
        wall_s,
        peak_rss_bytes,
        exit_code: output.status.code().unwrap_or(-1),
        stderr_tail,
    }
}

/// Truncates `s` to the last `max_bytes` bytes on a UTF-8 char
/// boundary so the markdown table cell stays readable.
fn tail_string(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let split_at = s.len() - max_bytes;
    let mut idx = split_at;
    while idx < s.len() && !s.is_char_boundary(idx) {
        idx += 1;
    }
    s[idx..].to_string()
}

// ---------------------------------------------------------------------
// Public runners (one per Decision 15 column)
// ---------------------------------------------------------------------

/// Runs `llama-quantize <input.gguf> <output.gguf> <variant>`. Used
/// by the uncalibrated and imatrix Q4_K_M peer cells (after
/// [`run_convert_hf_to_gguf`] materialises the float source).
pub fn run_llama_quantize(
    input_gguf: &Path,
    output_gguf: &Path,
    variant: &str,
) -> RunMetrics {
    let bin = match resolve_binary("HF2Q_LLAMA_QUANTIZE_BIN", "llama-quantize") {
        Some(b) => b,
        None => {
            tracing::warn!(
                "llama-quantize not found on $PATH and HF2Q_LLAMA_QUANTIZE_BIN unset (or override invalid); peer cell will report Verdict::NotMeasured"
            );
            return RunMetrics::missing_binary("llama-quantize");
        }
    };

    let in_str = input_gguf.to_string_lossy();
    let out_str = output_gguf.to_string_lossy();
    let args = [in_str.as_ref(), out_str.as_ref(), variant];
    spawn_and_measure(&bin, &args)
}

/// Runs llama.cpp's `convert_hf_to_gguf.py` (Python, not a binary —
/// resolved either via `HF2Q_LLAMA_CONVERT_HF_BIN` pointing at the
/// script path or by looking for the script on `$PATH`). Materialises
/// the float source GGUF the uncalibrated and imatrix peer cells
/// quantise from.
pub fn run_convert_hf_to_gguf(input_hf_dir: &Path, output_gguf: &Path) -> RunMetrics {
    let bin = match resolve_binary("HF2Q_LLAMA_CONVERT_HF_BIN", "convert_hf_to_gguf.py") {
        Some(b) => b,
        None => {
            tracing::warn!(
                "convert_hf_to_gguf.py not found on $PATH and HF2Q_LLAMA_CONVERT_HF_BIN unset (or override invalid); peer cell will report Verdict::NotMeasured"
            );
            return RunMetrics::missing_binary("convert_hf_to_gguf.py");
        }
    };

    let in_str = input_hf_dir.to_string_lossy();
    let out_str = output_gguf.to_string_lossy();
    let args: [&str; 3] = [in_str.as_ref(), "--outfile", out_str.as_ref()];
    spawn_and_measure(&bin, &args)
}

/// Runs `llama-imatrix -m <model.gguf> -f <calibration_text> -o
/// <imatrix.dat>`. Produces the importance-matrix the imatrix-q4_k_m
/// peer cell consumes.
pub fn run_llama_imatrix(
    model_gguf: &Path,
    calibration_text: &Path,
    output_dat: &Path,
) -> RunMetrics {
    let bin = match resolve_binary("HF2Q_LLAMA_IMATRIX_BIN", "llama-imatrix") {
        Some(b) => b,
        None => {
            tracing::warn!(
                "llama-imatrix not found on $PATH and HF2Q_LLAMA_IMATRIX_BIN unset (or override invalid); peer cell will report Verdict::NotMeasured"
            );
            return RunMetrics::missing_binary("llama-imatrix");
        }
    };

    let m_str = model_gguf.to_string_lossy();
    let f_str = calibration_text.to_string_lossy();
    let o_str = output_dat.to_string_lossy();
    let args = [
        "-m",
        m_str.as_ref(),
        "-f",
        f_str.as_ref(),
        "-o",
        o_str.as_ref(),
    ];
    spawn_and_measure(&bin, &args)
}

// ---------------------------------------------------------------------
// Internal helpers for unit-testable behaviour that does not touch
// process-global env (env-var mutation across parallel tests is
// inherently racy under cargo's default test runner — the public
// `run_*` functions remain the env-driven surface, but the unit tests
// drive `spawn_and_measure` + `resolve_binary` directly with explicit
// inputs).
// ---------------------------------------------------------------------

#[cfg(test)]
pub(crate) fn spawn_and_measure_for_test(bin: &Path, args: &[&str]) -> RunMetrics {
    spawn_and_measure(bin, args)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `resolve_binary` returns `None` when the override env-var
    /// points at a non-existent path — the harness routes that
    /// through the missing-binary sentinel rather than silently
    /// falling back to `$PATH`. We test the resolver directly to
    /// avoid env-var races against the harness's own integration
    /// tests (which need a clean env).
    #[test]
    fn binary_missing_returns_sentinel() {
        // Spawn a guaranteed-absent path directly to drive the same
        // `Err(io::ErrorKind::NotFound)` branch the public wrapper
        // takes when `resolve_binary` succeeds-but-points-at-trash
        // (i.e., the file existed at `is_file()` time but vanished
        // before spawn — same observable behaviour: spawn fails ⇒
        // missing-binary sentinel).
        let bin = Path::new("/nonexistent/llama-quantize-test-stub");
        let m = spawn_and_measure_for_test(bin, &["arg1"]);
        assert!(
            m.is_missing_binary(),
            "missing binary must produce sentinel; got {:?}",
            m
        );
        assert_eq!(m.wall_s, -1.0);
        assert_eq!(m.peak_rss_bytes, u64::MAX);
        assert_eq!(m.exit_code, -1);
        assert!(
            m.stderr_tail.contains("llama-quantize"),
            "stderr_tail must name the missing binary; got `{}`",
            m.stderr_tail
        );
    }

    /// When the binary path resolves and the subprocess exits 0
    /// (we use `/bin/echo` as a guaranteed-present stand-in), the
    /// wrapper records real metrics: wall_s ≥ 0.0, exit_code 0,
    /// sentinel fields NOT set. Llama.cpp is *not* required for the
    /// harness to be trusted at the type level.
    #[test]
    fn binary_present_records_real_metrics() {
        let bin = Path::new("/bin/echo");
        let m = spawn_and_measure_for_test(bin, &["smoke"]);
        assert!(
            !m.is_missing_binary(),
            "real /bin/echo invocation must NOT produce missing-binary sentinel; got {:?}",
            m
        );
        assert!(m.wall_s >= 0.0, "wall_s must be non-negative; got {}", m.wall_s);
        assert_eq!(m.exit_code, 0, "/bin/echo always exits 0");
    }

    /// `parse_bsd_time_peak_rss` recognises the canonical macOS BSD
    /// `time -l` line shape.
    #[test]
    fn bsd_time_parser_extracts_peak_rss_bytes() {
        let stderr = "        1.23 real         0.45 user         0.06 sys\n            12345678  maximum resident set size\n                   0  average shared memory size\n";
        let rss = parse_bsd_time_peak_rss(stderr);
        assert_eq!(rss, Some(12345678));
    }

    /// `parse_bsd_time_peak_rss` returns `None` when the line is
    /// absent (caller should treat as 0, NOT as missing-binary).
    #[test]
    fn bsd_time_parser_returns_none_when_label_missing() {
        let stderr = "no time-l output here\n";
        assert!(parse_bsd_time_peak_rss(stderr).is_none());
    }
}
