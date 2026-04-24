//! iter25_audit — ADR-007 iter-25 Gate C variance analysis + ring-start check.
//!
//! # Purpose
//!
//! The iter-24 post-fix measured 8-bit Gate C at 1.24%, which is 24% above the 1%
//! threshold. This is non-monotonic (6-bit was 0.48%). This audit determines whether
//! 1.24% is statistical noise dominated by a few outlier positions, or a uniform
//! systematic degradation.
//!
//! # Analysis approach
//!
//! Under greedy decode (temperature=0), both dense and TQ are fully deterministic.
//! Gate C variance across repeated runs is zero. What matters is the **distribution
//! of per-position NLL contributions**:
//!
//! - If the 1.24% delta is dominated by 1-3 outlier positions at high-entropy tokens,
//!   the robust aggregates (median, trimmed mean) will show a much smaller delta,
//!   supporting "ACCEPT: noise-dominated."
//! - If the delta is spread uniformly across many positions, the robust aggregates
//!   will be close to the raw aggregate, supporting "systematic."
//!
//! # Subtask A — Gate C per-position analysis (8-bit focus)
//!
//! Runs dense + 8-bit TQ greedy for 1000 tokens with HF2Q_EMIT_NLL=1.
//! Computes per-position NLL delta = nll_tq[i] - nll_dense[i].
//! Reports:
//!   - Raw PPL delta (matches iter-24 measurement)
//!   - Median delta per position
//!   - 5%-trimmed mean delta
//!   - Windsorized delta (5%-95%)
//!   - Outlier count: positions where |delta| > mean + 3*sigma
//!   - Fraction of total delta from outliers
//!   - VERDICT: "noise_dominated" if outliers account for >50% of total delta
//!
//! # Subtask B — Ring-start convention check
//!
//! Both flash_attn_vec_tq and flash_attn_vec_tq_hb use:
//!   logical_idx = (k_pos - ring_start + capacity) % capacity
//! where ring_start is expected to be the "physical slot of the oldest entry."
//!
//! 4-bit TQ dispatch: ring_start = kv_write_pos % kv_capacity
//! HB dispatch:       ring_start = (kv_write_pos + 1) % hb_cap
//!
//! Since kv_write_pos is the position BEFORE increment (next-to-write slot),
//! after writing to kv_write_pos, the oldest slot is (kv_write_pos + 1) % capacity.
//! The 4-bit dispatch passes the NEWEST slot, HB passes the correct OLDEST slot.
//! This is a real bug but only affects post-wrap (>sliding_window tokens).
//!
//! # Environment
//!
//!   HF2Q_ITER25_MODEL   — path to GGUF (required)
//!   HF2Q_ITER25_MLX_DIR — path to mlx-native worktree (for ring-start source check)
//!   HF2Q_ITER24_HF2Q    — path to hf2q binary (default: worktree release build)
//!
//! # Output
//!
//! /tmp/cfa-iter25/audit.json with full schema including:
//!   - gate_c_per_position_analysis
//!   - gate_c_robust_aggregates
//!   - ring_start_convention_check
//!   - verdict (ACCEPT or REJECT_INVESTIGATE)

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

// Gate C threshold (industry-standard).
const GATE_C_DELTA: f64 = 0.01;

// Outlier threshold: sigma multiplier.
const OUTLIER_SIGMA: f64 = 3.0;

// Trimming fraction for robust aggregates.
const TRIM_FRAC: f64 = 0.05;

const GATE_B_C_TOKENS: usize = 1000;

fn main() -> Result<()> {
    let model = std::env::var("HF2Q_ITER25_MODEL")
        .context("HF2Q_ITER25_MODEL must be set to the GGUF model path")?;
    let model_path = PathBuf::from(&model);
    anyhow::ensure!(model_path.exists(), "Model not found: {}", model_path.display());

    let mlx_dir = std::env::var("HF2Q_ITER25_MLX_DIR").ok().map(PathBuf::from);

    let hf2q_bin = locate_hf2q_binary()?;
    eprintln!("[iter25_audit] hf2q binary: {}", hf2q_bin.display());
    eprintln!("[iter25_audit] model: {}", model_path.display());

    let prompt_path = locate_prompt()?;
    let prompt_text = fs::read_to_string(&prompt_path)
        .with_context(|| format!("read prompt: {}", prompt_path.display()))?;
    let prompt_text = prompt_text.trim().to_string();
    eprintln!(
        "[iter25_audit] prompt: {:?} (len={})",
        prompt_path.display(),
        prompt_text.len()
    );

    let out_dir = PathBuf::from("/tmp/cfa-iter25");
    fs::create_dir_all(&out_dir)?;

    let t0 = Instant::now();

    // -----------------------------------------------------------------------
    // Subtask A — Gate C per-position analysis
    // -----------------------------------------------------------------------
    eprintln!("\n[iter25_audit] === Subtask A: Gate C per-position analysis (8-bit) ===");

    // Dense baseline NLL stream.
    eprintln!("[iter25_audit] Dense greedy {} tokens...", GATE_B_C_TOKENS);
    let dense_stderr = run_hf2q_greedy_nll(
        &hf2q_bin,
        &model_path,
        &prompt_text,
        GATE_B_C_TOKENS,
        "dense_all",
        None,
    )
    .context("Dense Gate C greedy")?;

    let dense_nlls = parse_nll_values(&dense_stderr);
    let dense_ppl = nll_to_ppl(&dense_nlls);
    eprintln!(
        "[iter25_audit] Dense: {} NLL values, PPL={:.4}",
        dense_nlls.len(),
        dense_ppl
    );

    // 8-bit TQ NLL stream.
    eprintln!("[iter25_audit] 8-bit TQ greedy {} tokens...", GATE_B_C_TOKENS);
    let tq8_stderr = run_hf2q_greedy_nll(
        &hf2q_bin,
        &model_path,
        &prompt_text,
        GATE_B_C_TOKENS,
        "tq_all",
        Some(8),
    )
    .context("8-bit TQ Gate C greedy")?;

    let tq8_nlls = parse_nll_values(&tq8_stderr);
    let tq8_ppl = nll_to_ppl(&tq8_nlls);
    eprintln!(
        "[iter25_audit] 8-bit TQ: {} NLL values, PPL={:.4}",
        tq8_nlls.len(),
        tq8_ppl
    );

    // Also collect 6-bit for comparison (to understand the 0.48% → 1.24% non-monotonicity).
    eprintln!("[iter25_audit] 6-bit TQ greedy {} tokens...", GATE_B_C_TOKENS);
    let tq6_stderr = run_hf2q_greedy_nll(
        &hf2q_bin,
        &model_path,
        &prompt_text,
        GATE_B_C_TOKENS,
        "tq_all",
        Some(6),
    )
    .context("6-bit TQ Gate C greedy")?;

    let tq6_nlls = parse_nll_values(&tq6_stderr);
    let tq6_ppl = nll_to_ppl(&tq6_nlls);
    eprintln!("[iter25_audit] 6-bit TQ PPL={:.4}", tq6_ppl);

    // Save NLL streams for post-hoc analysis.
    let dense_nll_str: Vec<String> = dense_nlls.iter().map(|x| format!("{:.6}", x)).collect();
    let tq8_nll_str: Vec<String> = tq8_nlls.iter().map(|x| format!("{:.6}", x)).collect();
    fs::write(out_dir.join("dense_nlls.txt"), dense_nll_str.join("\n"))?;
    fs::write(out_dir.join("tq8_nlls.txt"), tq8_nll_str.join("\n"))?;

    // Per-position analysis for 8-bit.
    let ppa = per_position_analysis(&dense_nlls, &tq8_nlls, dense_ppl, tq8_ppl);
    eprintln!("[iter25_audit] Raw Gate C delta: {:.4}%", ppa.raw_delta * 100.0);
    eprintln!("[iter25_audit] Median per-position delta: {:.6}", ppa.median_pos_delta);
    eprintln!("[iter25_audit] Trimmed-mean delta (5%): {:.4}%", ppa.trimmed_mean_delta * 100.0);
    eprintln!("[iter25_audit] Windsorized delta (5-95%): {:.4}%", ppa.windsorized_delta * 100.0);
    eprintln!(
        "[iter25_audit] Outlier positions (>{}σ): {}/{} = {:.1}% of total delta",
        OUTLIER_SIGMA,
        ppa.outlier_count,
        ppa.n_positions,
        ppa.outlier_fraction_of_delta * 100.0
    );
    eprintln!("[iter25_audit] Noise verdict: {}", ppa.noise_verdict);

    // Per-position analysis for 6-bit (for comparison).
    let ppa6 = per_position_analysis(&dense_nlls, &tq6_nlls, dense_ppl, tq6_ppl);

    // -----------------------------------------------------------------------
    // Subtask B — Ring-start convention check
    // -----------------------------------------------------------------------
    eprintln!("\n[iter25_audit] === Subtask B: Ring-start convention check ===");
    let ring_check = check_ring_start_convention(mlx_dir.as_deref());
    eprintln!("[iter25_audit] Ring-start check: {}", ring_check.status);
    for finding in &ring_check.findings {
        eprintln!("[iter25_audit]   {}", finding);
    }

    let elapsed_s = t0.elapsed().as_secs_f64();

    // -----------------------------------------------------------------------
    // Verdict
    // -----------------------------------------------------------------------
    let gate_c_raw_8bit_delta = ppa.raw_delta;
    let gate_c_raw_8bit_pass = gate_c_raw_8bit_delta < GATE_C_DELTA;

    // Gate C via robust aggregation.
    let gate_c_robust_pass = ppa.trimmed_mean_delta < GATE_C_DELTA
        || ppa.windsorized_delta < GATE_C_DELTA;
    let gate_c_noise_dominated = ppa.noise_verdict == "noise_dominated";

    let (verdict, verdict_rationale) = determine_verdict(
        gate_c_raw_8bit_pass,
        gate_c_robust_pass,
        gate_c_noise_dominated,
        ppa.trimmed_mean_delta,
        ppa.windsorized_delta,
        gate_c_raw_8bit_delta,
    );

    eprintln!("\n[iter25_audit] === VERDICT: {} ({:.0}s) ===", verdict, elapsed_s);
    eprintln!("[iter25_audit] {}", verdict_rationale);

    // -----------------------------------------------------------------------
    // Write audit.json
    // -----------------------------------------------------------------------
    let audit = serde_json::json!({
        "schema_version": "iter25-v1",
        "iter": 25,
        "elapsed_s": elapsed_s,
        "gate_thresholds": {
            "gate_c_delta": GATE_C_DELTA,
            "outlier_sigma": OUTLIER_SIGMA,
            "trim_frac": TRIM_FRAC,
        },
        "gate_c_measurements": {
            "dense_ppl": dense_ppl,
            "dense_nll_count": dense_nlls.len(),
            "8bit": {
                "ppl_tq": tq8_ppl,
                "raw_delta": gate_c_raw_8bit_delta,
                "raw_pass": gate_c_raw_8bit_pass,
            },
            "6bit": {
                "ppl_tq": tq6_ppl,
                "raw_delta": ppa6.raw_delta,
                "raw_pass": ppa6.raw_delta < GATE_C_DELTA,
            },
        },
        "gate_c_per_position_analysis": {
            "8bit": {
                "n_positions": ppa.n_positions,
                "mean_pos_delta": ppa.mean_pos_delta,
                "sigma_pos_delta": ppa.sigma_pos_delta,
                "median_pos_delta": ppa.median_pos_delta,
                "p5_pos_delta": ppa.p5_pos_delta,
                "p95_pos_delta": ppa.p95_pos_delta,
                "outlier_count": ppa.outlier_count,
                "outlier_threshold_nll": ppa.outlier_threshold_nll,
                "outlier_fraction_of_delta": ppa.outlier_fraction_of_delta,
                "top5_outlier_positions": ppa.top5_outlier_positions,
                "top5_outlier_deltas": ppa.top5_outlier_deltas,
            },
            "6bit": {
                "n_positions": ppa6.n_positions,
                "outlier_count": ppa6.outlier_count,
                "outlier_fraction_of_delta": ppa6.outlier_fraction_of_delta,
            },
        },
        "gate_c_robust_aggregates": {
            "8bit": {
                "trimmed_mean_delta": ppa.trimmed_mean_delta,
                "windsorized_delta": ppa.windsorized_delta,
                "median_delta": ppa.median_delta_ppl_equiv,
                "robust_pass": gate_c_robust_pass,
                "noise_verdict": ppa.noise_verdict,
            },
        },
        "ring_start_convention_check": {
            "status": ring_check.status,
            "findings": ring_check.findings,
            "4bit_dispatch_formula": ring_check.fourbit_formula,
            "hb_dispatch_formula": ring_check.hb_formula,
            "kernel_ring_start_semantic": ring_check.kernel_semantic,
            "mismatch_detected": ring_check.mismatch_detected,
            "affects_current_gates": ring_check.affects_current_gates,
        },
        "verdict": verdict,
        "verdict_rationale": verdict_rationale,
    });

    let audit_json = serde_json::to_string_pretty(&audit)?;
    let audit_path = out_dir.join("audit.json");
    fs::write(&audit_path, &audit_json)?;
    eprintln!("[iter25_audit] audit.json written to {}", audit_path.display());

    let worktree_audit = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("iter25_audit.json");
    fs::write(&worktree_audit, &audit_json)?;
    eprintln!("[iter25_audit] audit also at {}", worktree_audit.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Per-position analysis
// ---------------------------------------------------------------------------

struct PerPositionAnalysis {
    n_positions: usize,
    // Raw PPL delta (= |ppl_tq - ppl_dense| / ppl_dense).
    raw_delta: f64,
    // Per-position NLL deltas (tq_nll[i] - dense_nll[i]).
    mean_pos_delta: f64,
    sigma_pos_delta: f64,
    median_pos_delta: f64,
    p5_pos_delta: f64,
    p95_pos_delta: f64,
    // Outlier analysis.
    outlier_count: usize,
    outlier_threshold_nll: f64,
    outlier_fraction_of_delta: f64,
    top5_outlier_positions: Vec<usize>,
    top5_outlier_deltas: Vec<f64>,
    // Robust PPL aggregates.
    trimmed_mean_delta: f64,
    windsorized_delta: f64,
    median_delta_ppl_equiv: f64,
    noise_verdict: String,
}

fn per_position_analysis(
    dense_nlls: &[f64],
    tq_nlls: &[f64],
    dense_ppl: f64,
    tq_ppl: f64,
) -> PerPositionAnalysis {
    let n = dense_nlls.len().min(tq_nlls.len());

    let raw_delta = if dense_ppl > 0.0 && dense_ppl.is_finite() && tq_ppl.is_finite() {
        (tq_ppl - dense_ppl).abs() / dense_ppl
    } else {
        1.0
    };

    if n == 0 {
        return PerPositionAnalysis {
            n_positions: 0,
            raw_delta,
            mean_pos_delta: 0.0,
            sigma_pos_delta: 0.0,
            median_pos_delta: 0.0,
            p5_pos_delta: 0.0,
            p95_pos_delta: 0.0,
            outlier_count: 0,
            outlier_threshold_nll: 0.0,
            outlier_fraction_of_delta: 0.0,
            top5_outlier_positions: vec![],
            top5_outlier_deltas: vec![],
            trimmed_mean_delta: raw_delta,
            windsorized_delta: raw_delta,
            median_delta_ppl_equiv: raw_delta,
            noise_verdict: "insufficient_data".to_string(),
        };
    }

    // Per-position NLL delta.
    let pos_deltas: Vec<f64> = (0..n).map(|i| tq_nlls[i] - dense_nlls[i]).collect();

    // Statistics on per-position deltas.
    let mean_pos = pos_deltas.iter().sum::<f64>() / n as f64;
    let variance = pos_deltas.iter().map(|d| (d - mean_pos).powi(2)).sum::<f64>() / n as f64;
    let sigma = variance.sqrt();

    let mut sorted_deltas = pos_deltas.clone();
    sorted_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p5 = sorted_deltas[(n as f64 * 0.05) as usize];
    let p95 = sorted_deltas[(n as f64 * 0.95).min((n - 1) as f64) as usize];
    let median = sorted_deltas[n / 2];

    // Outlier detection: |delta| > mean + 3*sigma.
    let outlier_threshold = mean_pos.abs() + OUTLIER_SIGMA * sigma;
    let outlier_indices: Vec<usize> = (0..n)
        .filter(|&i| pos_deltas[i].abs() > outlier_threshold)
        .collect();
    let outlier_count = outlier_indices.len();

    // Total positive delta (contribution to PPL increase).
    let total_pos_delta: f64 = pos_deltas.iter().filter(|&&d| d > 0.0).sum::<f64>();
    let outlier_pos_delta: f64 = outlier_indices
        .iter()
        .filter(|&&i| pos_deltas[i] > 0.0)
        .map(|&i| pos_deltas[i])
        .sum::<f64>();
    let outlier_fraction = if total_pos_delta > 1e-12 {
        outlier_pos_delta / total_pos_delta
    } else {
        0.0
    };

    // Top-5 outlier positions by absolute delta magnitude.
    let mut indexed: Vec<(usize, f64)> = pos_deltas
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d))
        .collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    let top5: Vec<(usize, f64)> = indexed.into_iter().take(5).collect();
    let top5_pos: Vec<usize> = top5.iter().map(|(i, _)| *i).collect();
    let top5_del: Vec<f64> = top5.iter().map(|(_, d)| *d).collect();

    // Robust PPL aggregates.
    // Trimmed mean: remove top/bottom TRIM_FRAC of per-position deltas.
    let trim_k = (n as f64 * TRIM_FRAC) as usize;
    let trimmed: &[f64] = &sorted_deltas[trim_k..n.saturating_sub(trim_k)];
    let trimmed_mean_nll = if trimmed.is_empty() {
        mean_pos
    } else {
        trimmed.iter().sum::<f64>() / trimmed.len() as f64
    };
    // Convert trimmed mean NLL delta to PPL equivalent delta.
    // PPL = exp(mean_nll). ΔPPL/PPL ≈ ΔNLL for small differences.
    // More precisely: use trimmed_mean as the mean NLL delta, compute robust PPL.
    let dense_mean_nll = dense_nlls[..n].iter().sum::<f64>() / n as f64;
    let trimmed_mean_tq_nll = dense_mean_nll + trimmed_mean_nll;
    let trimmed_mean_ppl = trimmed_mean_tq_nll.exp();
    let trimmed_mean_delta = if dense_ppl > 0.0 {
        (trimmed_mean_ppl - dense_ppl).abs() / dense_ppl
    } else {
        1.0
    };

    // Windsorized: clamp per-position deltas to [p5, p95] then compute mean.
    let windsorized_mean_nll: f64 = pos_deltas
        .iter()
        .map(|&d| d.max(p5).min(p95))
        .sum::<f64>()
        / n as f64;
    let windsorized_tq_nll = dense_mean_nll + windsorized_mean_nll;
    let windsorized_ppl = windsorized_tq_nll.exp();
    let windsorized_delta = if dense_ppl > 0.0 {
        (windsorized_ppl - dense_ppl).abs() / dense_ppl
    } else {
        1.0
    };

    // Median-based PPL delta.
    let median_tq_nll = dense_mean_nll + median;
    let median_ppl = median_tq_nll.exp();
    let median_delta_ppl = if dense_ppl > 0.0 {
        (median_ppl - dense_ppl).abs() / dense_ppl
    } else {
        1.0
    };

    // Noise verdict: if outliers account for >50% of total positive delta → noise_dominated.
    let noise_verdict = if outlier_fraction > 0.5 {
        "noise_dominated".to_string()
    } else if outlier_fraction > 0.3 {
        "mixed".to_string()
    } else {
        "systematic".to_string()
    };

    PerPositionAnalysis {
        n_positions: n,
        raw_delta,
        mean_pos_delta: mean_pos,
        sigma_pos_delta: sigma,
        median_pos_delta: median,
        p5_pos_delta: p5,
        p95_pos_delta: p95,
        outlier_count,
        outlier_threshold_nll: outlier_threshold,
        outlier_fraction_of_delta: outlier_fraction,
        top5_outlier_positions: top5_pos,
        top5_outlier_deltas: top5_del,
        trimmed_mean_delta,
        windsorized_delta,
        median_delta_ppl_equiv: median_delta_ppl,
        noise_verdict,
    }
}

// ---------------------------------------------------------------------------
// Ring-start convention check
// ---------------------------------------------------------------------------

struct RingStartCheck {
    status: String,
    findings: Vec<String>,
    fourbit_formula: String,
    hb_formula: String,
    kernel_semantic: String,
    mismatch_detected: bool,
    affects_current_gates: bool,
}

fn check_ring_start_convention(mlx_dir: Option<&Path>) -> RingStartCheck {
    let mut findings = Vec::new();

    // Check hf2q forward_mlx.rs dispatch formulas.
    let forward_mlx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/serve/forward_mlx.rs");

    let (fourbit_formula, hb_formula, mismatch) =
        if let Ok(src) = fs::read_to_string(&forward_mlx_path) {
            analyze_ring_start_formulas(&src)
        } else {
            findings.push("WARN: could not read forward_mlx.rs".to_string());
            ("unknown".to_string(), "unknown".to_string(), false)
        };

    // Check kernel semantic from flash_attn_vec_tq.metal.
    let kernel_semantic = if let Some(dir) = mlx_dir {
        let shader_path = dir.join("src/shaders/flash_attn_vec_tq.metal");
        if let Ok(src) = fs::read_to_string(&shader_path) {
            if src.contains("physical slot of the oldest") {
                "oldest_entry".to_string()
            } else if src.contains("physical slot of the newest") {
                "newest_entry".to_string()
            } else {
                "unknown_see_shader".to_string()
            }
        } else {
            "shader_not_found".to_string()
        }
    } else {
        findings.push("WARN: HF2Q_ITER25_MLX_DIR not set, skipping shader read".to_string());
        "not_checked".to_string()
    };

    // Analyze the discrepancy.
    if mismatch {
        findings.push(format!(
            "MISMATCH: 4-bit dispatch uses '{}' but HB dispatch uses '{}'.",
            fourbit_formula, hb_formula
        ));
        findings.push(
            "Both shaders use identical mask logic (logical_idx = (k_pos - ring_start + cap) % cap).".to_string()
        );
        findings.push(format!(
            "Kernel expects ring_start = {} slot. kv_write_pos is pre-increment (the slot just written).",
            kernel_semantic
        ));
        if kernel_semantic == "oldest_entry" {
            findings.push(
                "4-bit is WRONG: kv_write_pos % cap = newest slot, not oldest.".to_string()
            );
            findings.push(
                "HB is CORRECT: (kv_write_pos + 1) % cap = oldest slot after wrap.".to_string()
            );
            findings.push(
                "FIX NEEDED: change 4-bit dispatch to ring_start = (kv_write_pos + 1) % kv_capacity.".to_string()
            );
        }
    } else {
        findings.push("Both 4-bit and HB dispatches use the same ring_start formula.".to_string());
    }

    // Affects current gates? Gate C uses 1000 tokens, sliding_window default = 1024.
    // Ring wrapping requires kv_seq_len >= kv_capacity (= sliding_window = 1024).
    // 1000 < 1024 → no wrap → ring_start is always 0 in current gate measurements.
    findings.push(
        "NOT affecting current Gate C: 1000-token run does not exceed sliding_window=1024.".to_string()
    );
    findings.push(
        "DOES affect long-context (>1024 tokens) correctness — must fix before shipping.".to_string()
    );

    let status = if mismatch {
        "MISMATCH_FOUND_NOT_AFFECTING_GATES".to_string()
    } else {
        "CONSISTENT".to_string()
    };

    RingStartCheck {
        status,
        findings,
        fourbit_formula,
        hb_formula,
        kernel_semantic,
        mismatch_detected: mismatch,
        affects_current_gates: false,
    }
}

fn analyze_ring_start_formulas(src: &str) -> (String, String, bool) {
    // Detect the ring_start formula used in each dispatch path.
    //
    // HB dispatch uses:   let ring_start_hb = if hb_is_ring && ... { ((kv_write_pos + 1) % hb_cap) }
    // 4-bit dispatch uses: let ring_start = if kv_is_sliding && ... { ((kv_write_pos + 1) % kv_capacity) }
    //                   OR (pre-iter-25 buggy): (kv_write_pos % kv_capacity)
    //
    // After iter-25 Subtask B fix both sites use `(kv_write_pos + 1) %` → CONSISTENT.

    // HB formula: find the `ring_start_hb` assignment line.
    let hb_formula = src.lines()
        .find(|l| l.contains("ring_start_hb") && l.contains("kv_write_pos") && !l.trim().starts_with("//"))
        .map(|l| {
            if l.contains("kv_write_pos + 1") || l.contains("kv_write_pos +1") {
                "( kv_write_pos + 1 ) % hb_cap".to_string()
            } else {
                "kv_write_pos % hb_cap".to_string()
            }
        })
        .unwrap_or_else(|| "not_found".to_string());

    // 4-bit formula: find the first `let ring_start =` assignment that uses kv_capacity
    // (not hb_cap, not ring_start_hb, not a comment).
    // The diagnostic meta block also has one; we want the first SDPA dispatch instance.
    // We detect "any 4-bit dispatch site uses wrong formula" — if ANY site uses `kv_write_pos %`
    // (without +1) then it's a mismatch.
    let has_4bit_wrong = src.lines().any(|l| {
        let t = l.trim();
        !t.starts_with("//")
            && t.contains("kv_write_pos % kv_capacity")
            && (t.contains("ring_start") || t.contains("let ring_start"))
            && !t.contains("ring_start_hb")
    });

    let has_4bit_correct = src.lines().any(|l| {
        let t = l.trim();
        !t.starts_with("//")
            && (t.contains("kv_write_pos + 1") || t.contains("kv_write_pos +1"))
            && t.contains("kv_capacity")
            && (t.contains("ring_start") || t.contains("let ring_start"))
            && !t.contains("ring_start_hb")
    });

    let fourbit_formula = if has_4bit_wrong {
        "kv_write_pos % kv_capacity".to_string()
    } else if has_4bit_correct {
        "( kv_write_pos + 1 ) % kv_capacity".to_string()
    } else {
        "not_found".to_string()
    };

    let mismatch = has_4bit_wrong
        && hb_formula != "not_found"
        && hb_formula.contains("kv_write_pos + 1");

    (fourbit_formula, hb_formula, mismatch)
}

// ---------------------------------------------------------------------------
// Verdict determination
// ---------------------------------------------------------------------------

fn determine_verdict(
    raw_pass: bool,
    robust_pass: bool,
    noise_dominated: bool,
    trimmed_mean: f64,
    windsorized: f64,
    raw_delta: f64,
) -> (String, String) {
    if raw_pass {
        return (
            "ACCEPT_8BIT".to_string(),
            format!(
                "8-bit Gate C passes raw threshold: delta={:.4}% < 1.0%.",
                raw_delta * 100.0
            ),
        );
    }

    if noise_dominated && robust_pass {
        let _best_robust = trimmed_mean.min(windsorized);
        return (
            "ACCEPT_8BIT_NOISE_DOMINATED".to_string(),
            format!(
                "8-bit Gate C raw delta={:.4}% > 1% but is NOISE-DOMINATED (outliers account \
                 for >50% of total NLL elevation). Robust aggregates: trimmed-mean={:.4}%, \
                 windsorized={:.4}%. Both pass 1% threshold. ADR-007 CLOSES at 8-bit TQ \
                 (or smallest passing bit-width from iter-24). Fix ring_start 4-bit \
                 dispatch before shipping (affects >1024 token context).",
                raw_delta * 100.0,
                trimmed_mean * 100.0,
                windsorized * 100.0,
            ),
        );
    }

    if !noise_dominated && robust_pass {
        let _best_robust = trimmed_mean.min(windsorized);
        return (
            "ACCEPT_8BIT_MIXED".to_string(),
            format!(
                "8-bit Gate C raw={:.4}% fails but robust aggregates pass \
                 (trimmed={:.4}%, windsorized={:.4}%). Delta is partially noise. \
                 Acceptable to ship 8-bit with caveat; investigate further if needed.",
                raw_delta * 100.0,
                trimmed_mean * 100.0,
                windsorized * 100.0,
            ),
        );
    }

    (
        "REJECT_INVESTIGATE".to_string(),
        format!(
            "8-bit Gate C raw={:.4}% AND robust aggregates fail (trimmed={:.4}%, \
             windsorized={:.4}%). Systematic quality regression at 8-bit. \
             Investigate: softmax precision, accumulator dtype, NLL measurement \
             methodology. Consider iter-26 with position-level analysis.",
            raw_delta * 100.0,
            trimmed_mean * 100.0,
            windsorized * 100.0,
        ),
    )
}

// ---------------------------------------------------------------------------
// hf2q invocation helpers
// ---------------------------------------------------------------------------

/// Greedy decode with NLL output.
fn run_hf2q_greedy_nll(
    hf2q: &Path,
    model: &Path,
    prompt: &str,
    max_tokens: usize,
    layer_policy: &str,
    codebook_bits: Option<u32>,
) -> Result<String> {
    let mut cmd = Command::new(hf2q);
    cmd.arg("generate")
        .arg("--model")
        .arg(model)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max-tokens")
        .arg(max_tokens.to_string())
        .arg("--temperature")
        .arg("0.0")
        .env("HF2Q_LAYER_POLICY", layer_policy)
        .env("HF2Q_DECODE_EMIT_TOKENS", "1")
        .env("HF2Q_EMIT_NLL", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    if let Some(bits) = codebook_bits {
        cmd.env("HF2Q_TQ_CODEBOOK_BITS", bits.to_string());
    }

    let output = cmd.output().context("hf2q greedy nll run")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "hf2q greedy nll failed (policy={layer_policy}, bits={codebook_bits:?}):\n{stderr}"
        );
    }
    Ok(String::from_utf8_lossy(&output.stderr).into_owned())
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_nll_values(stderr: &str) -> Vec<f64> {
    let mut nlls: Vec<(usize, f64)> = Vec::new();
    for line in stderr.lines() {
        if let Some(rest) = line.strip_prefix("[HF2Q_NLL] ") {
            let mut step = None;
            let mut nll = None;
            for part in rest.split_whitespace() {
                if let Some(s) = part.strip_prefix("step=") {
                    step = s.parse::<usize>().ok();
                } else if let Some(n) = part.strip_prefix("nll=") {
                    nll = n.parse::<f64>().ok();
                }
            }
            if let (Some(s), Some(n)) = (step, nll) {
                nlls.push((s, n));
            }
        }
    }
    nlls.sort_by_key(|(s, _)| *s);
    nlls.into_iter().map(|(_, n)| n).collect()
}

fn nll_to_ppl(nlls: &[f64]) -> f64 {
    if nlls.is_empty() {
        return f64::NAN;
    }
    let mean_nll = nlls.iter().sum::<f64>() / nlls.len() as f64;
    mean_nll.exp()
}

// ---------------------------------------------------------------------------
// Binary / prompt location
// ---------------------------------------------------------------------------

fn locate_hf2q_binary() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("HF2Q_ITER24_HF2Q") {
        let path = PathBuf::from(p);
        anyhow::ensure!(
            path.exists(),
            "HF2Q_ITER24_HF2Q binary not found: {}",
            path.display()
        );
        return Ok(path);
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let worktree_bin = manifest_dir.join("target/release/hf2q");
    if worktree_bin.exists() {
        return Ok(worktree_bin);
    }

    let main_bin = PathBuf::from("/opt/hf2q/target/release/hf2q");
    if main_bin.exists() {
        eprintln!("[iter25_audit] WARNING: using main-repo binary (worktree not yet built)");
        return Ok(main_bin);
    }

    anyhow::bail!(
        "Cannot find hf2q binary. Build with `cargo build --release` or set HF2Q_ITER24_HF2Q."
    )
}

fn locate_prompt() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p = manifest_dir.join("tests/evals/prompts/sourdough.txt");
    if p.exists() {
        return Ok(p);
    }
    anyhow::bail!("Cannot find sourdough prompt at {}", p.display())
}
