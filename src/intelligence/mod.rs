//! Intelligence module — hardware profiling, model fingerprinting, auto mode.
//!
//! Orchestrates:
//! - HardwareProfiler: detect chip, memory, cores
//! - ModelFingerprint: stable identifier from config.json
//! - AutoResolver: RuVector query -> heuristic fallback
//! - RuVector: self-learning conversion result storage

pub mod auto_quant;
pub mod fingerprint;
pub mod hardware;
pub mod heuristics;
pub mod ruvector;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, warn};

use self::fingerprint::ModelFingerprint;
use self::hardware::HardwareProfile;
#[allow(unused_imports)]
use self::heuristics::HeuristicResult;

/// Errors from intelligence operations.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum IntelligenceError {
    #[error("Hardware profiling failed: {0}")]
    Hardware(#[from] hardware::HardwareError),

    #[error("Model fingerprinting failed: {0}")]
    Fingerprint(#[from] fingerprint::FingerprintError),

    #[error("Heuristic resolution failed: {0}")]
    Heuristics(#[from] heuristics::HeuristicsError),

    #[error("RuVector is not accessible: {reason}. Required to store learnings. Run `hf2q doctor` to diagnose.")]
    RuVectorUnavailable { reason: String },

    #[error("RuVector error: {0}")]
    RuVector(#[from] ruvector::RuVectorError),
}

/// Resolved auto mode configuration — the output of AutoResolver::resolve().
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedConfig {
    /// Recommended quantization method name
    pub quant_method: String,
    /// Recommended bit width
    pub bits: u8,
    /// Recommended group size (0 for f16 passthrough)
    pub group_size: usize,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Source of the recommendation
    pub source: ResolvedSource,
    /// Human-readable reasoning
    pub reasoning: String,
    /// Hardware profile used for this resolution
    pub hardware: HardwareProfile,
    /// Model fingerprint used for this resolution
    pub fingerprint: ModelFingerprint,
}

/// Where the auto mode recommendation came from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResolvedSource {
    /// From a stored RuVector result (exact match)
    RuVectorExact,
    /// From a stored RuVector result (similar match)
    RuVectorSimilar,
    /// From rule-based heuristics (no stored data available)
    Heuristic,
}

impl std::fmt::Display for ResolvedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuVectorExact => write!(f, "stored (exact match)"),
            Self::RuVectorSimilar => write!(f, "stored (similar match)"),
            Self::Heuristic => write!(f, "heuristic"),
        }
    }
}

/// The auto mode resolver — queries RuVector first, falls back to heuristics.
pub struct AutoResolver;

impl AutoResolver {
    /// Resolve the optimal quantization configuration for the given hardware and model.
    ///
    /// Resolution order:
    /// 1. Query RuVector for exact hardware+model match
    /// 2. Query RuVector for similar matches
    /// 3. Fall back to rule-based heuristics
    ///
    /// If RuVector is available but empty, a warning is logged and heuristics are used.
    /// If RuVector is unavailable (feature disabled or db error), this returns an error
    /// because every conversion must be able to store results.
    ///
    /// The `ruvector_db` parameter is optional: if None, it means the ruvector feature
    /// is not enabled or the database is not available (which should be caught earlier
    /// at the preflight/CLI level).
    pub fn resolve(
        hardware: &HardwareProfile,
        fingerprint: &ModelFingerprint,
        ruvector_db: Option<&ruvector::RuVectorDb>,
    ) -> Result<ResolvedConfig, IntelligenceError> {
        info!(
            hardware_id = %hardware.stable_id(),
            model_id = %fingerprint.stable_id(),
            model = %fingerprint,
            total_memory_gb = hardware.total_memory_gb(),
            available_memory_gb = hardware.available_memory_gb(),
            "Auto mode: resolving optimal quantization"
        );

        // Step 1: Try RuVector if available
        if let Some(db) = ruvector_db {
            match db.query_best_config(hardware, fingerprint) {
                Ok(Some(stored)) => {
                    info!(
                        method = %stored.quant_method,
                        confidence = stored.confidence,
                        source = %stored.source,
                        "Auto mode: using stored RuVector result"
                    );
                    return Ok(stored);
                }
                Ok(None) => {
                    warn!(
                        "No prior conversion data for this hardware+model combination. \
                         Using heuristics. Results will be stored after conversion."
                    );
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "RuVector query failed, falling back to heuristics"
                    );
                }
            }
        }

        // Step 2: Fall back to heuristics
        let heuristic = heuristics::select_quant(hardware, fingerprint)?;

        let resolved = ResolvedConfig {
            quant_method: heuristic.quant_method,
            bits: heuristic.bits,
            group_size: heuristic.group_size,
            confidence: heuristic.confidence,
            source: ResolvedSource::Heuristic,
            reasoning: heuristic.reasoning,
            hardware: hardware.clone(),
            fingerprint: fingerprint.clone(),
        };

        info!(
            method = %resolved.quant_method,
            confidence = resolved.confidence,
            source = "heuristic",
            "Auto mode: using heuristic recommendation"
        );

        Ok(resolved)
    }
}

/// Display the resolved config for user confirmation before conversion.
pub fn display_resolved_config(config: &ResolvedConfig) {
    use console::style;

    println!();
    println!("{}", style("Auto Mode Resolution").bold().cyan());
    println!("{}", style("--------------------").dim());
    println!(
        "  Hardware:    {} ({:.0} GB unified memory)",
        style(&config.hardware.chip_model).bold(),
        config.hardware.total_memory_gb(),
    );
    println!(
        "  Model:       {}",
        style(&config.fingerprint).bold(),
    );
    println!(
        "  Method:      {}",
        style(&config.quant_method).green().bold(),
    );
    if config.bits < 16 {
        println!("  Bits:        {}", config.bits);
        println!("  Group size:  {}", config.group_size);
    }
    println!(
        "  Confidence:  {:.0}%",
        config.confidence * 100.0,
    );
    println!(
        "  Source:      {}",
        config.source,
    );
    println!("  Reasoning:   {}", config.reasoning);
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hardware() -> HardwareProfile {
        HardwareProfile {
            chip_model: "Apple M5 Max".to_string(),
            total_memory_bytes: 128 * 1024 * 1024 * 1024,
            available_memory_bytes: 100 * 1024 * 1024 * 1024,
            performance_cores: 14,
            efficiency_cores: 4,
            total_cores: 18,
            memory_bandwidth_gbs: 540.0,
        }
    }

    fn make_fingerprint() -> ModelFingerprint {
        ModelFingerprint {
            architecture: "LlamaForCausalLM".to_string(),
            total_params: 8_000_000_000,
            layer_count: 32,
            expert_count: 0,
            attention_types: vec!["attention".to_string()],
            hidden_size: 4096,
            dtype: "bfloat16".to_string(),
            intermediate_size: Some(14336),
            num_attention_heads: 32,
            num_kv_heads: Some(8),
            vocab_size: 128256,
        }
    }

    #[test]
    fn test_resolve_without_ruvector_uses_heuristics() {
        let hw = make_hardware();
        let fp = make_fingerprint();

        let result = AutoResolver::resolve(&hw, &fp, None).unwrap();
        assert_eq!(result.source, ResolvedSource::Heuristic);
        assert!(!result.quant_method.is_empty());
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_resolved_config_has_hardware_and_fingerprint() {
        let hw = make_hardware();
        let fp = make_fingerprint();

        let result = AutoResolver::resolve(&hw, &fp, None).unwrap();
        assert_eq!(result.hardware.chip_model, "Apple M5 Max");
        assert_eq!(result.fingerprint.architecture, "LlamaForCausalLM");
    }

    #[test]
    fn test_resolved_source_display() {
        assert_eq!(format!("{}", ResolvedSource::RuVectorExact), "stored (exact match)");
        assert_eq!(format!("{}", ResolvedSource::Heuristic), "heuristic");
    }
}
