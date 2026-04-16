//! Integration tests for the `generate` UX and tracing migration
//! (docs/generate-ux-cleanup.md).
//!
//! Tests drive the real `hf2q generate` binary against a GGUF model, so
//! they require a model. Tests skip gracefully (printing a note and
//! returning `Ok(())`) when `HF2Q_TEST_MODEL` is not set — this keeps CI
//! green on machines that don't carry the 16 GB model artifact while
//! letting developers run the full suite locally with:
//!
//!     HF2Q_TEST_MODEL=models/gemma-4-26B-.../gemma-4-....gguf \
//!         cargo test --release --test serve_ux
//!
//! Each test uses --temperature 0 for deterministic token output. Timing
//! values drift run-to-run and are normalized via `normalize_for_compare`.

use assert_cmd::Command;
use std::path::PathBuf;

const MODEL_ENV: &str = "HF2Q_TEST_MODEL";
const MAX_TOKENS: &str = "6";

/// Returns the test model path if `HF2Q_TEST_MODEL` is set and the file exists.
/// When returning None, the caller should skip the test (log + return).
fn test_model() -> Option<PathBuf> {
    let raw = std::env::var(MODEL_ENV).ok()?;
    let p = PathBuf::from(&raw);
    if !p.exists() {
        eprintln!(
            "{MODEL_ENV}={raw} does not exist on disk — skipping test"
        );
        return None;
    }
    Some(p)
}

/// Normalize a single line for fixture comparison:
/// 1. Strip tracing-subscriber fmt prefix ` LEVEL target: `.
/// 2. Replace timing values with the literal `<T>` so run-to-run drift
///    doesn't break comparisons. Token counts, layer counts, tensor
///    counts, and MB sizes are left alone — those are deterministic
///    under a fixed prompt + deterministic GPU math.
fn normalize(line: &str) -> String {
    // Strip tracing prefix: ` LEVEL target: `. Tracing's default format
    // left-pads INFO/WARN with a leading space; DEBUG/TRACE/ERROR don't
    // pad. Match either.
    static LEVEL_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let level_re = LEVEL_RE.get_or_init(|| {
        regex::Regex::new(r"^\s*(?:INFO|DEBUG|TRACE|WARN|ERROR)\s+[\w:_]+:\s").unwrap()
    });
    let stripped = level_re.replace(line, "").into_owned();

    // Timing substitutions — order matters (longer units first).
    static T_RE: std::sync::OnceLock<[regex::Regex; 3]> = std::sync::OnceLock::new();
    let res = T_RE.get_or_init(|| {
        [
            // "N.N tok/s" or "N tok/s"
            regex::Regex::new(r"\d+(?:\.\d+)?\s*tok/s").unwrap(),
            // "N.N ms" or "N ms"
            regex::Regex::new(r"\d+(?:\.\d+)?\s*ms\b").unwrap(),
            // "N.Ns" or "Ns" (seconds) — must end on non-word char (\b handles it)
            regex::Regex::new(r"\d+\.\d+s\b").unwrap(),
        ]
    });
    let mut s = stripped;
    for re in res {
        s = re.replace_all(&s, "<T>").into_owned();
    }
    s
}

#[test]
fn default_stdout_has_three_header_lines_and_generation() {
    let Some(model) = test_model() else {
        eprintln!("skipping: set {MODEL_ENV} to run");
        return;
    };
    let out = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "generate",
            "--model",
        ])
        .arg(&model)
        .args([
            "--prompt",
            "Test.",
            "--max-tokens",
            MAX_TOKENS,
            "--temperature",
            "0",
        ])
        .output()
        .expect("run hf2q generate");

    assert!(out.status.success(), "hf2q generate failed: {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Stdout isn't a TTY in `Command::output()`, so header is plain.
    let lines: Vec<&str> = stdout.split_inclusive('\n').collect();
    assert!(
        lines.len() >= 5,
        "expected at least 3 header + blank + generation; got {}:\n{stdout}",
        lines.len()
    );

    let line0 = lines[0].trim_end();
    let line1 = lines[1].trim_end();
    let line2 = lines[2].trim_end();
    let line3 = lines[3].trim_end();

    // Header line 1: "hf2q · <chip> · <backend>"
    assert!(
        line0.starts_with("hf2q · ") && line0.ends_with(" · mlx-native"),
        "line0 shape: {line0:?}"
    );

    // Header line 2: "<model> · loaded in X.Xs · N layers · X.X GB"
    assert!(
        line1.contains(" · loaded in ")
            && line1.contains(" layers · ")
            && line1.ends_with(" GB"),
        "line1 shape: {line1:?}"
    );

    // Header line 3: "prefill: N tok in Xms (Y tok/s)"
    assert!(
        line2.starts_with("prefill: ")
            && line2.contains(" tok in ")
            && line2.contains("ms (")
            && line2.ends_with(" tok/s)"),
        "line2 shape: {line2:?}"
    );

    // Blank line between header and generation.
    assert!(line3.is_empty(), "line3 should be blank, got {line3:?}");

    // Generation: remaining bytes are non-empty.
    let gen: String = lines[4..].concat();
    assert!(!gen.trim().is_empty(), "expected generation text");

    // No ANSI escape sequences on piped stdout.
    assert!(
        !stdout.contains("\x1b["),
        "expected no ANSI on piped stdout; got:\n{stdout}"
    );
}

#[test]
fn default_stderr_is_only_dim_or_plain_trailer() {
    let Some(model) = test_model() else {
        eprintln!("skipping: set {MODEL_ENV} to run");
        return;
    };
    let out = Command::cargo_bin("hf2q")
        .unwrap()
        .args(["generate", "--model"])
        .arg(&model)
        .args([
            "--prompt",
            "Test.",
            "--max-tokens",
            MAX_TOKENS,
            "--temperature",
            "0",
        ])
        .output()
        .expect("run hf2q generate");

    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Strip ANSI if present (stderr may still be a TTY when run from some
    // harnesses). Then collect non-empty lines.
    let plain = strip_ansi(&stderr);
    let non_empty: Vec<&str> = plain.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(
        non_empty.len(),
        1,
        "expected 1 non-empty stderr line at default verbosity; got {}:\n{plain}",
        non_empty.len()
    );
    assert!(
        non_empty[0].starts_with("--- mlx-native: ")
            && non_empty[0].ends_with(" ---"),
        "expected trailer; got {:?}",
        non_empty[0]
    );
}

#[test]
fn vv_boot_log_matches_fixture() {
    let Some(model) = test_model() else {
        eprintln!("skipping: set {MODEL_ENV} to run");
        return;
    };
    let fixture_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests/fixtures/generate_boot_log.golden.txt"]
        .iter()
        .collect();
    let fixture = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("read fixture {}: {e}", fixture_path.display()));

    let out = Command::cargo_bin("hf2q")
        .unwrap()
        .args(["generate", "-vv", "--model"])
        .arg(&model)
        .args([
            "--prompt",
            "Test.",
            "--max-tokens",
            "8",
            "--temperature",
            "0",
        ])
        .output()
        .expect("run hf2q generate -vv");

    assert!(out.status.success());
    let actual = strip_ansi(&String::from_utf8_lossy(&out.stderr));

    let norm_fixture: Vec<String> = fixture.lines().map(normalize).collect();
    let norm_actual: Vec<String> = actual.lines().map(normalize).collect();

    if norm_fixture != norm_actual {
        let fl = norm_fixture.len();
        let al = norm_actual.len();
        let diff_idx = norm_fixture
            .iter()
            .zip(norm_actual.iter())
            .position(|(a, b)| a != b);
        panic!(
            "-vv boot log drifted from fixture.\n  \
             fixture lines: {fl}, actual lines: {al}\n  \
             first diff at index: {diff_idx:?}\n  \
             fixture[{diff_idx:?}] = {:?}\n  \
             actual[{diff_idx:?}]  = {:?}\n\
             --- regenerate with ---\n  \
             ./target/release/hf2q generate -vv --model {} --prompt \"Test.\" --max-tokens 8 --temperature 0 \\\n    \
             2>tests/fixtures/generate_boot_log.golden.txt",
            diff_idx.map(|i| norm_fixture.get(i)),
            diff_idx.map(|i| norm_actual.get(i)),
            model.display(),
        );
    }
}

/// Strip CSI sequences `\x1b[...m` from a UTF-8 string. Regex-based so
/// multi-byte UTF-8 chars (em dash, etc.) round-trip unchanged.
fn strip_ansi(s: &str) -> String {
    static ANSI_RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = ANSI_RE.get_or_init(|| regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap());
    re.replace_all(s, "").into_owned()
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn normalize_strips_tracing_prefix() {
        assert_eq!(
            normalize(" INFO hf2q::serve: Loading GGUF model"),
            "Loading GGUF model"
        );
        assert_eq!(
            normalize("DEBUG hf2q::serve::forward_mlx: Loading embed_weight"),
            "Loading embed_weight"
        );
    }

    #[test]
    fn normalize_leaves_plain_lines() {
        assert_eq!(normalize("Prefill: KV cache dtype = F32"), "Prefill: KV cache dtype = F32");
    }

    #[test]
    fn normalize_replaces_ms_tok_per_s_seconds() {
        let got = normalize(
            " INFO hf2q::serve: mlx-native weights loaded (30 layers) in 2.4s",
        );
        assert_eq!(got, "mlx-native weights loaded (30 layers) in <T>");

        let got = normalize(
            "DEBUG hf2q::serve::forward_prefill: Prefill complete (dense SDPA): 15 tokens in 269.0 ms (55.8 tok/s), first decode token = 2094",
        );
        assert_eq!(
            got,
            "Prefill complete (dense SDPA): 15 tokens in <T> (<T>), first decode token = 2094"
        );
    }

    #[test]
    fn normalize_trailer() {
        assert_eq!(
            normalize("--- mlx-native: 8 tokens in 0.07s (121.2 tok/s) ---"),
            "--- mlx-native: 8 tokens in <T> (<T>) ---"
        );
    }

    #[test]
    fn strip_ansi_removes_escapes() {
        assert_eq!(strip_ansi("\x1b[2mdim\x1b[0m"), "dim");
        assert_eq!(strip_ansi("plain"), "plain");
    }
}
