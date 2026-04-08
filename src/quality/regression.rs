//! Quality regression detection and before/after comparison.
//!
//! Compares current quality metrics against a stored baseline to detect
//! degradation. Provides terminal-formatted comparison tables and
//! CI-friendly quality gate structures.

use serde::{Deserialize, Serialize};

use super::QualityReport;

/// Severity level for regression warnings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegressionSeverity {
    /// Degradation within tolerance
    Info,
    /// Degradation approaching tolerance limit
    Warning,
    /// Degradation exceeds tolerance
    Error,
}

impl std::fmt::Display for RegressionSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// A warning about quality regression on a specific metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionWarning {
    /// Name of the metric (e.g., "kl_divergence")
    pub metric: String,
    /// The baseline (previous) value
    pub baseline_value: f64,
    /// The current value
    pub current_value: f64,
    /// Percentage of degradation (positive = worse)
    pub degradation_pct: f64,
    /// Severity of the regression
    pub severity: RegressionSeverity,
}

/// Quality gate result for CI consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGate {
    /// Whether the quality gate passed
    pub passed: bool,
    /// List of threshold violations
    pub violations: Vec<String>,
    /// The thresholds that were applied
    pub thresholds: QualityGateThresholds,
}

/// Thresholds used for the quality gate check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateThresholds {
    pub max_kl_divergence: f64,
    pub max_perplexity_delta: f64,
    pub min_cosine_similarity: f64,
}

/// Compare current quality against a stored baseline.
///
/// Returns regression warnings if quality degraded beyond tolerance.
/// The `tolerance` parameter specifies the allowed degradation fraction
/// (e.g., 0.1 = 10% degradation allowed before flagging as error).
pub fn detect_regression(
    current: &QualityReport,
    baseline: &QualityReport,
    tolerance: f64,
) -> Vec<RegressionWarning> {
    let mut warnings = Vec::new();

    // KL divergence: lower is better, so increase = degradation
    if let (Some(curr_kl), Some(base_kl)) = (current.kl_divergence, baseline.kl_divergence) {
        if base_kl > 0.0 {
            let pct_change = ((curr_kl - base_kl) / base_kl) * 100.0;
            if pct_change > 0.0 {
                let severity = classify_degradation(pct_change / 100.0, tolerance);
                warnings.push(RegressionWarning {
                    metric: "kl_divergence".to_string(),
                    baseline_value: base_kl,
                    current_value: curr_kl,
                    degradation_pct: pct_change,
                    severity,
                });
            }
        }
    }

    // Perplexity delta: lower is better, so increase = degradation
    if let (Some(curr_ppl), Some(base_ppl)) =
        (current.perplexity_delta, baseline.perplexity_delta)
    {
        if base_ppl.abs() > 1e-9 {
            let pct_change = ((curr_ppl - base_ppl) / base_ppl.abs()) * 100.0;
            if pct_change > 0.0 {
                let severity = classify_degradation(pct_change / 100.0, tolerance);
                warnings.push(RegressionWarning {
                    metric: "perplexity_delta".to_string(),
                    baseline_value: base_ppl,
                    current_value: curr_ppl,
                    degradation_pct: pct_change,
                    severity,
                });
            }
        }
    }

    // Cosine similarity: higher is better, so decrease = degradation
    if let (Some(curr_cos), Some(base_cos)) =
        (current.cosine_sim_average, baseline.cosine_sim_average)
    {
        if base_cos > 0.0 {
            let pct_change = ((base_cos - curr_cos) / base_cos) * 100.0;
            if pct_change > 0.0 {
                let severity = classify_degradation(pct_change / 100.0, tolerance);
                warnings.push(RegressionWarning {
                    metric: "cosine_similarity".to_string(),
                    baseline_value: base_cos,
                    current_value: curr_cos,
                    degradation_pct: pct_change,
                    severity,
                });
            }
        }
    }

    warnings
}

/// Classify degradation severity based on how far it exceeds the tolerance.
fn classify_degradation(degradation_fraction: f64, tolerance: f64) -> RegressionSeverity {
    if degradation_fraction > tolerance {
        RegressionSeverity::Error
    } else if degradation_fraction > tolerance * 0.5 {
        RegressionSeverity::Warning
    } else {
        RegressionSeverity::Info
    }
}

/// Print a before/after quality comparison table to stderr.
pub fn print_comparison(before: &QualityReport, after: &QualityReport) {
    use console::style;

    eprintln!();
    eprintln!("{}", style("Quality Comparison").bold().cyan());
    eprintln!(
        "{}",
        style("──────────────────────────────").dim()
    );
    eprintln!(
        "  {:<18} {:>10} {:>10} {:>10}",
        style("Metric").bold(),
        style("Before").bold(),
        style("After").bold(),
        style("Delta").bold(),
    );

    // KL Divergence
    if let (Some(b), Some(a)) = (before.kl_divergence, after.kl_divergence) {
        let delta_pct = if b > 0.0 {
            ((a - b) / b) * 100.0
        } else {
            0.0
        };
        let delta_str = format!("{:+.1}%", delta_pct);
        // For KL: decrease is good (lower is better)
        let indicator = if delta_pct <= 0.0 { "+" } else { "!" };
        let styled_delta = if delta_pct <= 0.0 {
            style(format!("{} {}", delta_str, indicator)).green()
        } else if delta_pct < 10.0 {
            style(format!("{} {}", delta_str, indicator)).yellow()
        } else {
            style(format!("{} {}", delta_str, indicator)).red()
        };
        eprintln!(
            "  {:<18} {:>10.6} {:>10.6} {:>10}",
            "KL Divergence", b, a, styled_delta
        );
    }

    // Perplexity delta
    if let (Some(b), Some(a)) = (before.perplexity_delta, after.perplexity_delta) {
        let delta_pct = if b.abs() > 1e-9 {
            ((a - b) / b.abs()) * 100.0
        } else {
            0.0
        };
        let delta_str = format!("{:+.1}%", delta_pct);
        // For ppl delta: decrease is good (lower is better)
        let indicator = if delta_pct <= 0.0 { "+" } else { "!" };
        let styled_delta = if delta_pct <= 0.0 {
            style(format!("{} {}", delta_str, indicator)).green()
        } else if delta_pct < 10.0 {
            style(format!("{} {}", delta_str, indicator)).yellow()
        } else {
            style(format!("{} {}", delta_str, indicator)).red()
        };
        eprintln!(
            "  {:<18} {:>10.2} {:>10.2} {:>10}",
            "Perplexity Delta", b, a, styled_delta
        );
    }

    // Cosine similarity
    if let (Some(b), Some(a)) = (before.cosine_sim_average, after.cosine_sim_average) {
        let delta_pct = if b > 0.0 {
            ((a - b) / b) * 100.0
        } else {
            0.0
        };
        let delta_str = format!("{:+.2}%", delta_pct);
        // For cosine: increase is good (higher is better)
        let indicator = if delta_pct >= 0.0 { "+" } else { "!" };
        let styled_delta = if delta_pct >= 0.0 {
            style(format!("{} {}", delta_str, indicator)).green()
        } else if delta_pct > -1.0 {
            style(format!("{} {}", delta_str, indicator)).yellow()
        } else {
            style(format!("{} {}", delta_str, indicator)).red()
        };
        eprintln!(
            "  {:<18} {:>10.4} {:>10.4} {:>10}",
            "Cosine Sim (avg)", b, a, styled_delta
        );
    }

    eprintln!();
}

/// Build a QualityGate from a quality report and thresholds.
pub fn build_quality_gate(
    report: &QualityReport,
    thresholds: &super::QualityThresholds,
) -> QualityGate {
    let violations = super::check_thresholds(report, thresholds);
    QualityGate {
        passed: violations.is_empty(),
        violations,
        thresholds: QualityGateThresholds {
            max_kl_divergence: thresholds.max_kl_divergence,
            max_perplexity_delta: thresholds.max_perplexity_delta,
            min_cosine_similarity: thresholds.min_cosine_similarity,
        },
    }
}

/// Convert a RuVector QualityMetrics to a QualityReport for regression comparison.
pub fn quality_metrics_to_report(
    metrics: &crate::intelligence::ruvector::QualityMetrics,
) -> QualityReport {
    QualityReport {
        kl_divergence: metrics.kl_divergence,
        per_layer_kl: None,
        perplexity_pre: None,
        perplexity_post: None,
        perplexity_delta: metrics.perplexity_delta,
        per_layer_cosine_sim: None,
        cosine_sim_average: metrics.cosine_similarity,
        cosine_sim_min: None,
        cosine_sim_min_layer: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality::QualityReport;

    fn make_report(kl: f64, ppl_delta: f64, cosine: f64) -> QualityReport {
        QualityReport {
            kl_divergence: Some(kl),
            per_layer_kl: None,
            perplexity_pre: Some(10.0),
            perplexity_post: Some(10.0 + ppl_delta),
            perplexity_delta: Some(ppl_delta),
            per_layer_cosine_sim: None,
            cosine_sim_average: Some(cosine),
            cosine_sim_min: None,
            cosine_sim_min_layer: None,
        }
    }

    #[test]
    fn test_no_regression_when_quality_improves() {
        let baseline = make_report(0.05, 1.0, 0.98);
        let current = make_report(0.03, 0.5, 0.99);
        let warnings = detect_regression(&current, &baseline, 0.1);
        assert!(warnings.is_empty(), "Expected no warnings for improved quality");
    }

    #[test]
    fn test_regression_detected_on_kl() {
        let baseline = make_report(0.01, 0.5, 0.99);
        let current = make_report(0.05, 0.5, 0.99); // KL got much worse
        let warnings = detect_regression(&current, &baseline, 0.1);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].metric, "kl_divergence");
        assert_eq!(warnings[0].severity, RegressionSeverity::Error);
    }

    #[test]
    fn test_regression_within_tolerance_is_info() {
        let baseline = make_report(0.05, 1.0, 0.98);
        let current = make_report(0.052, 1.0, 0.98); // 4% increase, within 10% tolerance
        let warnings = detect_regression(&current, &baseline, 0.1);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].metric, "kl_divergence");
        assert_eq!(warnings[0].severity, RegressionSeverity::Info);
    }

    #[test]
    fn test_regression_warning_severity() {
        let baseline = make_report(0.05, 1.0, 0.98);
        // 8% increase: > 50% of 10% tolerance = warning
        let current = make_report(0.054, 1.0, 0.98);
        let warnings = detect_regression(&current, &baseline, 0.1);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].severity, RegressionSeverity::Warning);
    }

    #[test]
    fn test_regression_on_cosine_similarity_decrease() {
        let baseline = make_report(0.05, 1.0, 0.99);
        let current = make_report(0.05, 1.0, 0.95); // Cosine dropped
        let warnings = detect_regression(&current, &baseline, 0.1);
        assert!(warnings.iter().any(|w| w.metric == "cosine_similarity"));
    }

    #[test]
    fn test_no_regression_with_empty_reports() {
        let baseline = QualityReport::empty();
        let current = QualityReport::empty();
        let warnings = detect_regression(&current, &baseline, 0.1);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_build_quality_gate_pass() {
        let report = make_report(0.05, 1.0, 0.98);
        let thresholds = crate::quality::QualityThresholds::default();
        let gate = build_quality_gate(&report, &thresholds);
        assert!(gate.passed);
        assert!(gate.violations.is_empty());
    }

    #[test]
    fn test_build_quality_gate_fail() {
        let report = make_report(0.5, 5.0, 0.80);
        let thresholds = crate::quality::QualityThresholds::default();
        let gate = build_quality_gate(&report, &thresholds);
        assert!(!gate.passed);
        assert_eq!(gate.violations.len(), 3);
    }

    #[test]
    fn test_quality_gate_thresholds_serialized() {
        let report = make_report(0.05, 1.0, 0.98);
        let thresholds = crate::quality::QualityThresholds::default();
        let gate = build_quality_gate(&report, &thresholds);
        let json = serde_json::to_string(&gate).unwrap();
        assert!(json.contains("max_kl_divergence"));
        assert!(json.contains("max_perplexity_delta"));
        assert!(json.contains("min_cosine_similarity"));
    }

    #[test]
    fn test_quality_metrics_to_report() {
        let metrics = crate::intelligence::ruvector::QualityMetrics {
            kl_divergence: Some(0.023),
            perplexity_delta: Some(0.15),
            cosine_similarity: Some(0.997),
        };
        let report = quality_metrics_to_report(&metrics);
        assert_eq!(report.kl_divergence, Some(0.023));
        assert_eq!(report.perplexity_delta, Some(0.15));
        assert_eq!(report.cosine_sim_average, Some(0.997));
        assert!(report.per_layer_kl.is_none());
    }

    #[test]
    fn test_regression_severity_display() {
        assert_eq!(RegressionSeverity::Info.to_string(), "info");
        assert_eq!(RegressionSeverity::Warning.to_string(), "warning");
        assert_eq!(RegressionSeverity::Error.to_string(), "error");
    }

    #[test]
    fn test_classify_degradation() {
        assert_eq!(classify_degradation(0.02, 0.1), RegressionSeverity::Info);
        assert_eq!(classify_degradation(0.06, 0.1), RegressionSeverity::Warning);
        assert_eq!(classify_degradation(0.15, 0.1), RegressionSeverity::Error);
    }
}
