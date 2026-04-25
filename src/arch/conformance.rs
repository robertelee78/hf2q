//! Arch-generic conformance helpers — ADR-012 Decision 20.
//!
//! These helpers are consumed by:
//!   - `smoke.rs` (Decision 16) to assert tensor count, scan stderr
//!   - `hf2q smoke --quant dwq-*` (Decision 17) to measure PPL + KL
//!   - Integration tests for P11 MTP round-trip
//!
//! All helpers take `&ArchEntry` rather than hard-coded qwen paths
//! so Ministral (ADR-015) and DeepSeek-V3 (ADR-016) reuse them.

use std::path::{Path, PathBuf};

use super::catalog::CatalogExpansion;
use super::registry::{ArchEntry, QualityThresholds};

/// Exit codes used by `hf2q smoke` per ADR-012 Decision 16.
pub const EXIT_OK: u8 = 0;
pub const EXIT_HF_TOKEN_MISSING: u8 = 2;
pub const EXIT_INSUFFICIENT_DISK: u8 = 3;
pub const EXIT_LLAMA_CLI_MISSING: u8 = 4;
pub const EXIT_HF2Q_BINARY_NOT_RELEASE: u8 = 5;
pub const EXIT_HF_REPO_UNRESOLVABLE: u8 = 6;
pub const EXIT_UNKNOWN_ARCH: u8 = 7;
pub const EXIT_SMOKE_ASSERTION_FAILED: u8 = 8;

/// Result of a PPL + KL quality pass. Populated by P9 once `RealActivationCapture`
/// and the F16 reference forward are landed. Pre-P9 callers emit a `None`
/// measurement with a `skipped_reason`.
#[derive(Debug, Clone)]
pub struct QualityReport {
    pub arch: &'static str,
    pub quant_label: String,
    pub f16_perplexity: Option<f64>,
    pub dwq_perplexity: Option<f64>,
    pub median_kl_nats: Option<f64>,
    pub skipped_reason: Option<String>,
}

impl QualityReport {
    /// Apply the per-arch thresholds. Returns `Ok(())` on pass,
    /// `Err(reason)` on fail. A `skipped_reason` report is treated as
    /// pass for P8 smoke at Q4_0 (quality ACs land with P9).
    pub fn check(&self, thresholds: QualityThresholds) -> Result<(), String> {
        if self.skipped_reason.is_some() {
            return Ok(());
        }
        let Some(f16) = self.f16_perplexity else {
            return Err("f16_perplexity missing on non-skipped report".into());
        };
        let Some(dwq) = self.dwq_perplexity else {
            return Err("dwq_perplexity missing on non-skipped report".into());
        };
        let ratio = dwq / f16;
        let max_ratio = if self.quant_label.contains("4-6") || self.quant_label.contains("46") {
            thresholds.ppl_ratio_dwq46
        } else if self.quant_label.contains("4-8") || self.quant_label.contains("48") {
            thresholds.ppl_ratio_dwq48
        } else {
            // Q4_0 baseline — no PPL threshold in P8, treat as pass.
            return Ok(());
        };
        if ratio > max_ratio {
            return Err(format!(
                "{} PPL ratio {:.4} > {:.4} threshold (F16={:.4}, DWQ={:.4})",
                self.quant_label, ratio, max_ratio, f16, dwq
            ));
        }
        if let Some(kl) = self.median_kl_nats {
            if kl > thresholds.max_median_kl {
                return Err(format!(
                    "median KL {:.4} nats > {:.4} threshold",
                    kl, thresholds.max_median_kl
                ));
            }
        }
        Ok(())
    }
}

/// Smoke transcript output path — `tests/fixtures/smoke-transcripts/{arch}-{quant}.txt`
pub fn smoke_transcript_path(
    fixtures_root: &Path,
    arch: &str,
    quant: &str,
) -> PathBuf {
    fixtures_root
        .join("smoke-transcripts")
        .join(format!("{}-{}.txt", arch, quant))
}

/// Scan llama-cli stderr for regression patterns per Decision 16 §4.
///
/// Returns `Ok(())` if no pattern matched, `Err(reason)` on match.
pub fn scan_llama_cli_stderr(stderr: &str) -> Result<(), String> {
    const PATTERNS: &[&str] = &["error", "ERROR", "panic", "assertion", "segfault"];
    for line in stderr.lines() {
        for pat in PATTERNS {
            if line.contains(pat) {
                return Err(format!(
                    "llama-cli stderr matched regression pattern {:?}: {}",
                    pat, line
                ));
            }
        }
    }
    Ok(())
}

/// Extract `n_eval` from llama-cli's timing block. Returns `None` if
/// no `llama_print_timings: n_eval = %d` line is present.
pub fn extract_n_eval(stderr: &str) -> Option<u32> {
    for line in stderr.lines() {
        if let Some(rest) = line.split_once("n_eval = ") {
            let num: String = rest
                .1
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            return num.parse().ok();
        }
    }
    None
}

/// Extract loaded-tensor count from llama-cli stderr.
/// Pattern: `llama_model_load: loaded tensor 0x%x`
pub fn extract_loaded_tensor_count(stderr: &str) -> Option<u64> {
    for line in stderr.lines() {
        if let Some(rest) = line.split_once("loaded tensor ") {
            // Accept both decimal and 0x-hex forms.
            let token = rest.1.split_whitespace().next()?;
            if let Some(hex) = token.strip_prefix("0x").or_else(|| token.strip_prefix("0X")) {
                return u64::from_str_radix(hex, 16).ok();
            }
            return token.parse().ok();
        }
    }
    None
}

/// Assert a smoke transcript against a catalog.
///
/// Given a model's concrete parameters, verify (a) the number of
/// generated tokens is exactly `expected_n_gen`, (b) no regression
/// pattern in stderr, (c) the loaded-tensor count matches the
/// catalog expansion.
pub fn assert_smoke_transcript(
    entry: &ArchEntry,
    exp: CatalogExpansion,
    stderr: &str,
    expected_n_gen: u32,
) -> Result<(), String> {
    scan_llama_cli_stderr(stderr)?;
    let n_eval = extract_n_eval(stderr).ok_or("missing `n_eval` line in llama-cli output")?;
    if n_eval != expected_n_gen {
        return Err(format!(
            "generated tokens: expected {}, got {}",
            expected_n_gen, n_eval
        ));
    }
    let expected = entry.expected_tensor_count(exp);
    let Some(actual) = extract_loaded_tensor_count(stderr) else {
        return Err("missing `loaded tensor` line in llama-cli output".into());
    };
    if actual != expected {
        return Err(format!(
            "{}: loaded {} tensors, catalog expects {} (arch={}, num_layers={}, experts={}, mtp={})",
            entry.arch,
            actual,
            expected,
            entry.arch,
            exp.num_hidden_layers,
            exp.num_experts,
            exp.mtp_num_hidden_layers
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::entries::qwen35moe;

    #[test]
    fn scan_stderr_passes_clean_output() {
        let s = "llama_model_load: loaded tensor 0x2ff\n\
                 llama_print_timings: n_eval = 8\n";
        scan_llama_cli_stderr(s).expect("no regression patterns");
    }

    #[test]
    fn scan_stderr_flags_panic_line() {
        let s = "some line\npanic: boom\nmore\n";
        let err = scan_llama_cli_stderr(s).unwrap_err();
        assert!(err.contains("panic"));
    }

    #[test]
    fn scan_stderr_flags_assertion_line() {
        let s = "assertion failed: nope\n";
        let err = scan_llama_cli_stderr(s).unwrap_err();
        assert!(err.contains("assertion"));
    }

    #[test]
    fn extract_n_eval_reads_timings_block() {
        let s = "llama_print_timings: n_eval = 8 runs\n";
        assert_eq!(extract_n_eval(s), Some(8));
    }

    #[test]
    fn extract_loaded_tensor_count_parses_hex() {
        let s = "llama_model_load: loaded tensor 0x2ff\n";
        assert_eq!(extract_loaded_tensor_count(s), Some(0x2ff));
    }

    #[test]
    fn extract_loaded_tensor_count_parses_decimal() {
        let s = "llama_model_load: loaded tensor 767\n";
        assert_eq!(extract_loaded_tensor_count(s), Some(767));
    }

    #[test]
    fn assert_smoke_transcript_accepts_matching_count() {
        let exp = CatalogExpansion {
            num_hidden_layers: 40,
            num_full_attention_layers: 10,
            num_linear_attention_layers: 30,
            num_experts: 256,
            has_shared_expert: true,
            mtp_num_hidden_layers: 1,
        };
        // qwen35moe expected: 737 tensors at these params (0x2e1).
        let stderr = "llama_model_load: loaded tensor 0x2e1\n\
                      llama_print_timings: n_eval = 8 runs\n";
        assert_smoke_transcript(&qwen35moe::ENTRY, exp, stderr, 8).expect("pass");
    }

    #[test]
    fn assert_smoke_transcript_rejects_wrong_tensor_count() {
        let exp = CatalogExpansion {
            num_hidden_layers: 40,
            num_full_attention_layers: 10,
            num_linear_attention_layers: 30,
            num_experts: 256,
            has_shared_expert: true,
            mtp_num_hidden_layers: 1,
        };
        // Off by one — 736 when catalog expects 737 (0x2e0 vs 0x2e1).
        let stderr = "llama_model_load: loaded tensor 0x2e0\n\
                      llama_print_timings: n_eval = 8 runs\n";
        let err =
            assert_smoke_transcript(&qwen35moe::ENTRY, exp, stderr, 8).unwrap_err();
        assert!(err.contains("737"), "err = {}", err);
        assert!(err.contains("736"), "err = {}", err);
    }

    #[test]
    fn assert_smoke_transcript_rejects_wrong_n_gen() {
        let exp = CatalogExpansion {
            num_hidden_layers: 40,
            num_full_attention_layers: 10,
            num_linear_attention_layers: 30,
            num_experts: 256,
            has_shared_expert: true,
            mtp_num_hidden_layers: 1,
        };
        let stderr = "llama_model_load: loaded tensor 0x2ff\n\
                      llama_print_timings: n_eval = 4 runs\n";
        let err =
            assert_smoke_transcript(&qwen35moe::ENTRY, exp, stderr, 8).unwrap_err();
        assert!(err.contains("expected 8"));
    }

    /// Missing `n_eval` line — `extract_n_eval` returns None and the
    /// validator surfaces the actionable "missing n_eval" error.
    /// Defends against an upstream llama-cli format change that drops
    /// the timings block, which would silently pass an old smoke
    /// transcript by reading nothing as "okay".
    #[test]
    fn assert_smoke_transcript_rejects_missing_n_eval_line() {
        let exp = CatalogExpansion {
            num_hidden_layers: 40,
            num_full_attention_layers: 10,
            num_linear_attention_layers: 30,
            num_experts: 256,
            has_shared_expert: true,
            mtp_num_hidden_layers: 1,
        };
        let stderr = "llama_model_load: loaded tensor 0x2e1\n";
        let err =
            assert_smoke_transcript(&qwen35moe::ENTRY, exp, stderr, 8).unwrap_err();
        assert!(
            err.contains("n_eval"),
            "missing-n_eval error must name the line, got: {err}"
        );
    }

    /// Missing `loaded tensor` line — `extract_loaded_tensor_count`
    /// returns None and the validator surfaces "missing `loaded tensor`".
    /// Defends against an upstream llama-cli format change that drops
    /// the model-load summary, which would silently pass.
    #[test]
    fn assert_smoke_transcript_rejects_missing_loaded_tensor_line() {
        let exp = CatalogExpansion {
            num_hidden_layers: 40,
            num_full_attention_layers: 10,
            num_linear_attention_layers: 30,
            num_experts: 256,
            has_shared_expert: true,
            mtp_num_hidden_layers: 1,
        };
        let stderr = "llama_print_timings: n_eval = 8 runs\n";
        let err =
            assert_smoke_transcript(&qwen35moe::ENTRY, exp, stderr, 8).unwrap_err();
        assert!(
            err.contains("loaded tensor"),
            "missing-loaded-tensor error must name the line, got: {err}"
        );
    }

    #[test]
    fn quality_report_q4_0_passes_without_thresholds() {
        let report = QualityReport {
            arch: "qwen35",
            quant_label: "q4_0".to_string(),
            f16_perplexity: Some(10.0),
            dwq_perplexity: Some(100.0), // Any ratio — no threshold for q4_0 in P8.
            median_kl_nats: None,
            skipped_reason: None,
        };
        report
            .check(QualityThresholds::ADR_012_DEFAULT)
            .expect("q4_0 has no PPL gate in P8");
    }

    #[test]
    fn quality_report_dwq46_enforces_110_percent_bound() {
        let under = QualityReport {
            arch: "qwen35",
            quant_label: "dwq46".to_string(),
            f16_perplexity: Some(10.0),
            dwq_perplexity: Some(10.9),
            median_kl_nats: Some(0.01),
            skipped_reason: None,
        };
        under
            .check(QualityThresholds::ADR_012_DEFAULT)
            .expect("10.9/10.0 = 1.09 < 1.10");

        let over = QualityReport {
            arch: "qwen35",
            quant_label: "dwq46".to_string(),
            f16_perplexity: Some(10.0),
            dwq_perplexity: Some(11.5),
            median_kl_nats: Some(0.01),
            skipped_reason: None,
        };
        let err = over.check(QualityThresholds::ADR_012_DEFAULT).unwrap_err();
        assert!(err.contains("1.1500") || err.contains("PPL ratio"));
    }

    #[test]
    fn quality_report_skipped_passes() {
        let r = QualityReport {
            arch: "qwen35",
            quant_label: "dwq48".to_string(),
            f16_perplexity: None,
            dwq_perplexity: None,
            median_kl_nats: None,
            skipped_reason: Some("P9 not yet shipped".into()),
        };
        r.check(QualityThresholds::ADR_012_DEFAULT)
            .expect("skipped report passes");
    }

    #[test]
    fn smoke_transcript_path_is_stable() {
        let root = PathBuf::from("tests/fixtures");
        let p = smoke_transcript_path(&root, "qwen35", "q4_0");
        assert_eq!(
            p,
            PathBuf::from("tests/fixtures/smoke-transcripts/qwen35-q4_0.txt")
        );
    }

    /// Decision 16 AC — "exit with the distinct non-zero code listed
    /// above (2/3/4/5/6)". Exit codes are the user's debugging handle
    /// when a preflight fails; two constants colliding on the same
    /// integer would silently conflate two different prerequisites.
    /// Load-bearing guard as future ADRs register new arches that may
    /// need new exit codes (Decision 20 §extensibility).
    #[test]
    fn exit_codes_are_all_distinct() {
        let codes: Vec<(&str, u8)> = vec![
            ("EXIT_OK", EXIT_OK),
            ("EXIT_HF_TOKEN_MISSING", EXIT_HF_TOKEN_MISSING),
            ("EXIT_INSUFFICIENT_DISK", EXIT_INSUFFICIENT_DISK),
            ("EXIT_LLAMA_CLI_MISSING", EXIT_LLAMA_CLI_MISSING),
            ("EXIT_HF2Q_BINARY_NOT_RELEASE", EXIT_HF2Q_BINARY_NOT_RELEASE),
            ("EXIT_HF_REPO_UNRESOLVABLE", EXIT_HF_REPO_UNRESOLVABLE),
            ("EXIT_UNKNOWN_ARCH", EXIT_UNKNOWN_ARCH),
            ("EXIT_SMOKE_ASSERTION_FAILED", EXIT_SMOKE_ASSERTION_FAILED),
        ];
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(
                    codes[i].1, codes[j].1,
                    "exit code collision: {} and {} both = {}",
                    codes[i].0, codes[j].0, codes[i].1
                );
            }
        }
        // Decision 16 §AC also implies all non-zero codes are non-zero —
        // EXIT_OK is the only zero; everything else is a real failure.
        for (name, c) in codes.iter().filter(|(n, _)| *n != "EXIT_OK") {
            assert_ne!(*c, 0, "{name} must be non-zero (it signals failure)");
        }
    }
}
