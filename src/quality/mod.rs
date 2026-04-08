//! Quality measurement module.
//!
//! Phase 1: Weight-level cosine similarity (no inference needed).
//! Phase 2 (future): KL divergence and perplexity via Candle forward passes.

pub mod cosine_sim;
pub mod kl_divergence;
pub mod perplexity;

/// Quality measurement report — the complete results of quality analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityReport {
    /// Overall KL divergence
    pub kl_divergence: Option<f64>,
    /// Per-layer KL divergence
    pub per_layer_kl: Option<Vec<f64>>,
    /// Pre-quantization perplexity
    pub perplexity_pre: Option<f64>,
    /// Post-quantization perplexity
    pub perplexity_post: Option<f64>,
    /// Perplexity delta
    pub perplexity_delta: Option<f64>,
    /// Per-layer cosine similarity
    pub per_layer_cosine_sim: Option<Vec<f64>>,
    /// Average cosine similarity
    pub cosine_sim_average: Option<f64>,
    /// Minimum cosine similarity (worst layer)
    pub cosine_sim_min: Option<f64>,
    /// Index of worst layer by cosine similarity
    pub cosine_sim_min_layer: Option<usize>,
}

impl QualityReport {
    /// Create an empty report (when quality measurement is skipped).
    pub fn empty() -> Self {
        Self {
            kl_divergence: None,
            per_layer_kl: None,
            perplexity_pre: None,
            perplexity_post: None,
            perplexity_delta: None,
            per_layer_cosine_sim: None,
            cosine_sim_average: None,
            cosine_sim_min: None,
            cosine_sim_min_layer: None,
        }
    }

    /// Whether any metrics are populated.
    pub fn has_metrics(&self) -> bool {
        self.kl_divergence.is_some()
            || self.perplexity_pre.is_some()
            || self.cosine_sim_average.is_some()
    }
}

/// Print a terminal summary of quality metrics.
pub fn print_quality_summary(report: &QualityReport) {
    use console::style;

    if !report.has_metrics() {
        return;
    }

    eprintln!();
    eprintln!("{}", style("Quality Metrics").bold().cyan());
    eprintln!("{}", style("───────────────").dim());

    if let Some(kl) = report.kl_divergence {
        let kl_style = if kl < 0.01 {
            style(format!("{:.6}", kl)).green()
        } else if kl < 0.1 {
            style(format!("{:.6}", kl)).yellow()
        } else {
            style(format!("{:.6}", kl)).red()
        };
        eprintln!("  KL Divergence: {}", kl_style);
    }

    if let Some(pre) = report.perplexity_pre {
        eprintln!("  Perplexity (pre):  {:.2}", pre);
    }
    if let Some(post) = report.perplexity_post {
        eprintln!("  Perplexity (post): {:.2}", post);
    }
    if let Some(delta) = report.perplexity_delta {
        let delta_style = if delta.abs() < 0.5 {
            style(format!("{:+.2}", delta)).green()
        } else if delta.abs() < 2.0 {
            style(format!("{:+.2}", delta)).yellow()
        } else {
            style(format!("{:+.2}", delta)).red()
        };
        eprintln!("  Perplexity delta:  {}", delta_style);
    }

    if let Some(avg) = report.cosine_sim_average {
        let avg_style = if avg > 0.99 {
            style(format!("{:.4}", avg)).green()
        } else if avg > 0.95 {
            style(format!("{:.4}", avg)).yellow()
        } else {
            style(format!("{:.4}", avg)).red()
        };
        eprintln!("  Cosine sim (avg):  {}", avg_style);
    }

    if let (Some(min), Some(idx)) = (report.cosine_sim_min, report.cosine_sim_min_layer) {
        eprintln!("  Cosine sim (min):  {:.4} (layer {})", min, idx);
    }

    eprintln!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_report_empty() {
        let report = QualityReport::empty();
        assert!(!report.has_metrics());
        assert!(report.kl_divergence.is_none());
    }

    #[test]
    fn test_quality_report_has_metrics() {
        let mut report = QualityReport::empty();
        assert!(!report.has_metrics());

        report.kl_divergence = Some(0.05);
        assert!(report.has_metrics());
    }
}
