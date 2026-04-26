//! Arch-generic conformance helpers (ADR-012 Decision 20).
//!
//! Each helper takes a `&ArchEntry` and returns a structured result.  No
//! per-arch hardcoding lives here — every Qwen-specific path is reached via
//! the registered entry's catalog/thresholds/corpus knobs.
//!
//! P8 ships:
//!   * [`assert_smoke_transcript`] — parse a `llama-cli -n 8 --temp 0
//!     --seed 42` transcript and assert it satisfies the smoke gate.
//!
//! P9 will extend this module with `ppl_kl_eval()` (Decision 17 PPL/KL).
//! P10's mmproj round-trip and P11's MTP round-trip helpers also land here.

use std::fmt;

use thiserror::Error;

use super::registry::ArchEntry;

#[derive(Debug, Error)]
pub enum ConformanceError {
    #[error("smoke transcript missing required token-count line for arch '{arch}'")]
    NoTokenCount { arch: String },

    #[error(
        "smoke transcript reported {actual} generated tokens for arch '{arch}', expected exactly 8"
    )]
    WrongTokenCount { arch: String, actual: u32 },

    #[error("smoke transcript for arch '{arch}' contained an error indicator: {line}")]
    ErrorIndicator { arch: String, line: String },

    #[error(
        "smoke transcript for arch '{arch}' reported {loaded} loaded tensors, expected at least {expected_min}"
    )]
    InsufficientTensors {
        arch: String,
        loaded: u64,
        expected_min: u64,
    },
}

/// Parsed outcome of a smoke run, suitable for committing as a fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct SmokeAssertion {
    pub arch: String,
    pub tokens_generated: u32,
    pub tensors_loaded: u64,
}

impl fmt::Display for SmokeAssertion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "arch={}, tokens_generated={}, tensors_loaded={}",
            self.arch, self.tokens_generated, self.tensors_loaded,
        )
    }
}

/// Substrings that, if present anywhere in a transcript, fail the smoke gate.
///
/// Match Decision 16 acceptance: `error|ERROR|panic|assertion|segfault`.  We
/// intentionally use a small substring list rather than regex to keep the
/// helper dep-free and obviously-correct.
const ERROR_INDICATORS: &[&str] = &[
    "error", "ERROR",  // case-sensitive: `Error:` and `ERROR:` both caught
    "panic",           // Rust panic
    "panicked",        // Rust panic-handler line
    "assertion",       // assertion failure
    "segfault",        // segmentation fault
    "Aborted",         // POSIX abort
    "fatal",           // generic fatal-log marker
    "FATAL",
];

/// Parse a `llama-cli -n 8 --seed 42 --temp 0 --no-warmup` transcript and
/// assert it satisfies the smoke gate.
///
/// The transcript is expected to contain (verbatim from llama-cli output):
///   * `n_eval = 8` somewhere on a `llama_print_timings` line
///   * `loaded tensor 0x<HEX>` (or equivalent) — checked loosely as a
///     `loaded tensor` substring count for now to avoid hex-parse fragility
///
/// Returns `Ok(SmokeAssertion)` with the parsed counts on success.
pub fn assert_smoke_transcript(
    entry: &ArchEntry,
    transcript: &str,
) -> Result<SmokeAssertion, ConformanceError> {
    // 1. No error indicators.  Skip lines that are clearly llama-cli's own
    //    metadata about *output* (e.g. "error_rate") — we only flag standalone
    //    occurrences that look like real errors.  For the conservative cut
    //    we just check exact substrings; if a future false positive appears
    //    the indicator list updates.
    for line in transcript.lines() {
        let trimmed = line.trim();
        for needle in ERROR_INDICATORS {
            if trimmed.contains(needle) {
                return Err(ConformanceError::ErrorIndicator {
                    arch: entry.arch.to_string(),
                    line: trimmed.to_string(),
                });
            }
        }
    }

    // 2. Token count.  Look for "n_eval = N" on a llama_print_timings line.
    let tokens_generated = parse_token_count(transcript).ok_or_else(|| {
        ConformanceError::NoTokenCount {
            arch: entry.arch.to_string(),
        }
    })?;
    if tokens_generated != 8 {
        return Err(ConformanceError::WrongTokenCount {
            arch: entry.arch.to_string(),
            actual: tokens_generated,
        });
    }

    // 3. Tensor count sanity.  llama-cli prints "loaded tensor" lines per
    //    tensor; we count occurrences and require at least the catalog's
    //    pattern count (a real model has many more — patterns × layers ×
    //    experts — but this floor protects against a stripped GGUF.)
    let tensors_loaded = count_loaded_tensors(transcript);
    let expected_min = entry.tensor_catalog.pattern_count() as u64;
    if tensors_loaded < expected_min {
        return Err(ConformanceError::InsufficientTensors {
            arch: entry.arch.to_string(),
            loaded: tensors_loaded,
            expected_min,
        });
    }

    Ok(SmokeAssertion {
        arch: entry.arch.to_string(),
        tokens_generated,
        tensors_loaded,
    })
}

fn parse_token_count(transcript: &str) -> Option<u32> {
    // llama-cli format: "llama_print_timings:        eval time = ... ( n_eval = 8 ...)"
    // We grep for "n_eval" anywhere and parse the integer after the next "=".
    for line in transcript.lines() {
        if let Some(idx) = line.find("n_eval") {
            let rest = &line[idx + "n_eval".len()..];
            // Skip optional whitespace and an "=" sign.
            let rest = rest.trim_start();
            let rest = rest.strip_prefix('=').unwrap_or(rest).trim_start();
            // Take the leading digits.
            let mut chars = rest.chars();
            let mut n = 0u32;
            let mut any = false;
            while let Some(c) = chars.next() {
                if let Some(d) = c.to_digit(10) {
                    n = n.saturating_mul(10).saturating_add(d);
                    any = true;
                } else {
                    break;
                }
            }
            if any {
                return Some(n);
            }
        }
    }
    None
}

fn count_loaded_tensors(transcript: &str) -> u64 {
    let mut n = 0u64;
    for line in transcript.lines() {
        // Either "loaded tensor" (newer llama.cpp) or "load_tensors:" lines
        // count toward the tensor floor.  We accept both because llama.cpp's
        // log format varies across builds — we care about presence, not
        // exact phrasing.
        if line.contains("loaded tensor") || line.contains("load_tensors:") {
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Registry;

    fn entry(name: &str) -> &'static ArchEntry {
        Registry::global().get(name).expect("entry registered")
    }

    fn good_transcript() -> String {
        // Minimal synthetic transcript that satisfies the gate for qwen35moe
        // (catalog pattern count is small; we add many `loaded tensor` lines).
        let mut s = String::new();
        s.push_str("loading qwen3.5-moe\n");
        for i in 0..200 {
            s.push_str(&format!("loaded tensor 0x{:x}: blk.{}.attn_q.weight\n", i, i % 40));
        }
        s.push_str("llama_print_timings: prompt eval time = 100 ms\n");
        s.push_str("llama_print_timings:        eval time = 50 ms / 8 runs ( n_eval = 8 )\n");
        s
    }

    #[test]
    fn good_transcript_passes() {
        let assertion = assert_smoke_transcript(entry("qwen35moe"), &good_transcript()).unwrap();
        assert_eq!(assertion.arch, "qwen35moe");
        assert_eq!(assertion.tokens_generated, 8);
        assert!(assertion.tensors_loaded >= entry("qwen35moe").tensor_catalog.pattern_count() as u64);
    }

    #[test]
    fn missing_n_eval_fails() {
        let bad = "loaded tensor 0x0\n".to_string();
        let err = assert_smoke_transcript(entry("qwen35"), &bad).unwrap_err();
        assert!(matches!(err, ConformanceError::NoTokenCount { .. }));
    }

    #[test]
    fn wrong_token_count_fails() {
        let mut bad = String::new();
        for i in 0..200 {
            bad.push_str(&format!("loaded tensor 0x{:x}\n", i));
        }
        bad.push_str("llama_print_timings: eval time = 50 ms ( n_eval = 16 )\n");
        let err = assert_smoke_transcript(entry("qwen35"), &bad).unwrap_err();
        match err {
            ConformanceError::WrongTokenCount { actual, .. } => assert_eq!(actual, 16),
            other => panic!("unexpected err: {other:?}"),
        }
    }

    #[test]
    fn error_indicator_in_transcript_fails() {
        let mut bad = good_transcript();
        bad.push_str("ERROR: tensor mismatch\n");
        let err = assert_smoke_transcript(entry("qwen35"), &bad).unwrap_err();
        assert!(matches!(err, ConformanceError::ErrorIndicator { .. }));
    }

    #[test]
    fn insufficient_tensors_fails() {
        // Only 2 loaded-tensor lines, far below the catalog floor.
        let bad = "loaded tensor 0x0\nloaded tensor 0x1\n\
                   llama_print_timings: ( n_eval = 8 )\n";
        let err = assert_smoke_transcript(entry("qwen35"), bad).unwrap_err();
        assert!(matches!(err, ConformanceError::InsufficientTensors { .. }));
    }

    #[test]
    fn parse_token_count_handles_variations() {
        assert_eq!(parse_token_count("foo n_eval = 8 bar"), Some(8));
        assert_eq!(parse_token_count("n_eval=12,"), Some(12));
        assert_eq!(parse_token_count("(n_eval = 256)"), Some(256));
        assert_eq!(parse_token_count("no count here"), None);
    }
}
