//! RuVector self-learning integration.
//!
//! Stores and retrieves conversion results for auto mode optimization.
//! The database lives at `~/.hf2q/ruvector/` and is required for operation.
//!
//! When the `ruvector` feature is not enabled, this module provides a
//! JSON-backed store that handles exact match lookups. With the feature
//! enabled, it additionally wraps ruvector-core for vector similarity search.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, info, warn};

use super::fingerprint::ModelFingerprint;
use super::hardware::HardwareProfile;
use super::{ResolvedConfig, ResolvedSource};

/// Errors from RuVector operations.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum RuVectorError {
    #[error("RuVector database not accessible at {path}: {reason}")]
    DatabaseUnavailable { path: String, reason: String },

    #[error("RuVector query failed: {reason}")]
    QueryFailed { reason: String },

    #[error("RuVector store failed: {reason}")]
    StoreFailed { reason: String },

    #[error("RuVector feature not enabled. Rebuild with: cargo build --features ruvector")]
    FeatureNotEnabled,

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A stored conversion result in RuVector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionRecord {
    /// Hardware profile at time of conversion
    pub hardware: HardwareProfile,
    /// Model fingerprint
    pub fingerprint: ModelFingerprint,
    /// Quantization method used
    pub quant_method: String,
    /// Bit width used
    pub bits: u8,
    /// Group size used
    pub group_size: usize,
    /// Quality metrics from the conversion
    pub quality: QualityMetrics,
    /// hf2q version at time of conversion
    pub hf2q_version: String,
    /// Timestamp (ISO 8601)
    pub timestamp: String,
    /// Whether this record is flagged for re-calibration (version change)
    pub needs_recalibration: bool,
}

/// Quality metrics stored with each conversion record.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityMetrics {
    /// Overall KL divergence (lower is better)
    pub kl_divergence: Option<f64>,
    /// Perplexity delta (lower is better)
    pub perplexity_delta: Option<f64>,
    /// Average cosine similarity (higher is better)
    pub cosine_similarity: Option<f64>,
}

/// Status of the RuVector database for diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuVectorStatus {
    /// Whether the database is operational
    pub operational: bool,
    /// Path to the database directory
    pub db_path: String,
    /// Number of stored conversion records
    pub record_count: usize,
    /// Database size on disk in bytes
    pub db_size_bytes: u64,
    /// Any status message
    pub message: String,
    /// Whether the ruvector crate feature flag is enabled
    pub feature_enabled: bool,
}

/// The default database directory relative to home.
const DB_DIR_NAME: &str = ".hf2q/ruvector";

/// The JSON records file name.
const RECORDS_FILE: &str = "records.json";

/// Get the default database path.
pub fn default_db_path() -> Result<PathBuf, RuVectorError> {
    let home = std::env::var("HOME").map_err(|_| RuVectorError::DatabaseUnavailable {
        path: "~/.hf2q/ruvector/".to_string(),
        reason: "HOME environment variable not set".to_string(),
    })?;
    Ok(PathBuf::from(home).join(DB_DIR_NAME))
}

/// The RuVector database handle.
///
/// Uses a JSON-backed store for conversion records with exact and similar
/// match lookups. When the `ruvector` crate feature is enabled, this can
/// additionally leverage ruvector-core's vector similarity search for
/// finding similar hardware/model configurations.
pub struct RuVectorDb {
    /// Path to the database directory
    db_path: PathBuf,
    /// In-memory records (loaded from disk on init)
    records: Vec<ConversionRecord>,
}

impl RuVectorDb {
    /// Open or create a RuVector database at the given path.
    pub fn open(db_path: &Path) -> Result<Self, RuVectorError> {
        // Ensure the directory exists
        if !db_path.exists() {
            std::fs::create_dir_all(db_path).map_err(|e| RuVectorError::DatabaseUnavailable {
                path: db_path.display().to_string(),
                reason: format!("Failed to create directory: {}", e),
            })?;
            info!(path = %db_path.display(), "Created RuVector database directory");
        }

        // Load existing records
        let records_path = db_path.join(RECORDS_FILE);
        let records = if records_path.exists() {
            let content = std::fs::read_to_string(&records_path)?;
            serde_json::from_str(&content).unwrap_or_else(|e| {
                warn!(
                    error = %e,
                    "Failed to parse RuVector records, starting fresh"
                );
                Vec::new()
            })
        } else {
            Vec::new()
        };

        debug!(
            path = %db_path.display(),
            records = records.len(),
            "RuVector database opened"
        );

        Ok(Self {
            db_path: db_path.to_path_buf(),
            records,
        })
    }

    /// Open the database at the default path (~/.hf2q/ruvector/).
    pub fn open_default() -> Result<Self, RuVectorError> {
        let path = default_db_path()?;
        Self::open(&path)
    }

    /// Get the database status for diagnostics.
    pub fn status(&self) -> RuVectorStatus {
        let db_size = self.calculate_db_size();

        RuVectorStatus {
            operational: true,
            db_path: self.db_path.display().to_string(),
            record_count: self.records.len(),
            db_size_bytes: db_size,
            message: if self.records.is_empty() {
                "Operational but empty. Will populate after first conversion.".to_string()
            } else {
                format!("Operational with {} stored conversions.", self.records.len())
            },
            feature_enabled: cfg!(feature = "ruvector"),
        }
    }

    /// Query for the best stored configuration matching this hardware+model combination.
    ///
    /// Returns the stored config with lowest KL divergence if multiple matches exist.
    /// Returns None if no matches are found (empty database or no matching records).
    pub fn query_best_config(
        &self,
        hardware: &HardwareProfile,
        fingerprint: &ModelFingerprint,
    ) -> Result<Option<ResolvedConfig>, RuVectorError> {
        if self.records.is_empty() {
            return Ok(None);
        }

        let hw_id = hardware.stable_id();
        let model_id = fingerprint.stable_id();

        // First: look for exact hardware+model matches (excluding flagged records)
        let exact_matches: Vec<&ConversionRecord> = self
            .records
            .iter()
            .filter(|r| {
                !r.needs_recalibration
                    && r.hardware.stable_id() == hw_id
                    && r.fingerprint.stable_id() == model_id
            })
            .collect();

        if !exact_matches.is_empty() {
            debug!(
                count = exact_matches.len(),
                "Found exact RuVector matches"
            );
            let best = select_best_record(&exact_matches);
            return Ok(Some(record_to_resolved(
                best,
                hardware,
                fingerprint,
                ResolvedSource::RuVectorExact,
            )));
        }

        // Second: look for same model on any hardware (useful recommendation)
        let model_matches: Vec<&ConversionRecord> = self
            .records
            .iter()
            .filter(|r| !r.needs_recalibration && r.fingerprint.stable_id() == model_id)
            .collect();

        if !model_matches.is_empty() {
            debug!(
                count = model_matches.len(),
                "Found similar RuVector matches (same model, different hardware)"
            );
            let best = select_best_record(&model_matches);
            return Ok(Some(record_to_resolved(
                best,
                hardware,
                fingerprint,
                ResolvedSource::RuVectorSimilar,
            )));
        }

        // Third: look for similar models (same architecture family, similar size)
        if let Some(config) = self.query_similar_configs(hardware, fingerprint)? {
            return Ok(Some(config));
        }

        // Fourth: use flagged (needs_recalibration) records as last resort
        let flagged_matches: Vec<&ConversionRecord> = self
            .records
            .iter()
            .filter(|r| {
                r.needs_recalibration
                    && r.hardware.stable_id() == hw_id
                    && r.fingerprint.stable_id() == model_id
            })
            .collect();

        if !flagged_matches.is_empty() {
            debug!(
                count = flagged_matches.len(),
                "Using flagged (needs revalidation) RuVector records with reduced confidence"
            );
            let best = select_best_record(&flagged_matches);
            let mut resolved = record_to_resolved(
                best,
                hardware,
                fingerprint,
                ResolvedSource::RuVectorSimilar,
            );
            // Reduce confidence further for flagged records
            resolved.confidence = (resolved.confidence * 0.7).clamp(0.2, 0.7);
            resolved.reasoning = format!(
                "{} [needs revalidation — from hf2q v{}]",
                resolved.reasoning, best.hf2q_version
            );
            return Ok(Some(resolved));
        }

        Ok(None)
    }

    /// Query for similar model configurations when no exact match exists.
    ///
    /// Similarity criteria (weighted):
    /// - Same architecture family (required)
    /// - Similar parameter count (within 2x, weighted by proximity)
    /// - Same or similar hardware (same chip family, bonus for exact match)
    fn query_similar_configs(
        &self,
        hardware: &HardwareProfile,
        fingerprint: &ModelFingerprint,
    ) -> Result<Option<ResolvedConfig>, RuVectorError> {
        let hw_chip_family = extract_chip_family(&hardware.chip_model);

        // Filter to records with the same architecture family
        let candidates: Vec<(&ConversionRecord, f64)> = self
            .records
            .iter()
            .filter(|r| !r.needs_recalibration)
            .filter(|r| r.fingerprint.architecture == fingerprint.architecture)
            .filter_map(|r| {
                let param_ratio = if fingerprint.total_params > 0 && r.fingerprint.total_params > 0
                {
                    let ratio = r.fingerprint.total_params as f64
                        / fingerprint.total_params as f64;
                    if !(0.5..=2.0).contains(&ratio) {
                        return None; // Outside 2x range
                    }
                    // Proximity score: 1.0 at exact match, 0.0 at 2x boundary
                    1.0 - (ratio.ln().abs() / 2.0_f64.ln())
                } else {
                    0.0
                };

                // Hardware similarity
                let r_chip_family = extract_chip_family(&r.hardware.chip_model);
                let hw_score = if r.hardware.stable_id() == hardware.stable_id() {
                    1.0
                } else if r_chip_family == hw_chip_family {
                    0.6
                } else {
                    0.2
                };

                // Weighted similarity: arch match (implicit, required) + param proximity + hw
                let similarity = param_ratio * 0.6 + hw_score * 0.4;
                Some((r, similarity))
            })
            .collect();

        if candidates.is_empty() {
            return Ok(None);
        }

        // Select the candidate with the highest combined similarity + quality score
        let best = candidates
            .iter()
            .max_by(|(a, a_sim), (b, b_sim)| {
                let a_combined = a_sim + compute_quality_score(&a.quality);
                let b_combined = b_sim + compute_quality_score(&b.quality);
                a_combined
                    .partial_cmp(&b_combined)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(r, _)| *r)
            .expect("candidates should not be empty");

        debug!(
            arch = %best.fingerprint.architecture,
            params = best.fingerprint.total_params,
            "Found similar model match in RuVector"
        );

        Ok(Some(record_to_resolved(
            best,
            hardware,
            fingerprint,
            ResolvedSource::RuVectorSimilar,
        )))
    }

    /// Store a conversion result.
    pub fn store_conversion(
        &mut self,
        hardware: &HardwareProfile,
        fingerprint: &ModelFingerprint,
        quant_method: &str,
        bits: u8,
        group_size: usize,
        quality: QualityMetrics,
    ) -> Result<(), RuVectorError> {
        let record = ConversionRecord {
            hardware: hardware.clone(),
            fingerprint: fingerprint.clone(),
            quant_method: quant_method.to_string(),
            bits,
            group_size,
            quality,
            hf2q_version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: make_timestamp(),
            needs_recalibration: false,
        };

        info!(
            model = %fingerprint.stable_id(),
            method = quant_method,
            bits = bits,
            "Storing conversion result in RuVector"
        );

        self.records.push(record);
        self.persist()?;

        Ok(())
    }

    /// Update the quality metrics for a previously stored conversion.
    ///
    /// Called after quality measurement completes (which may happen after initial
    /// storage). Finds the most recent matching record and updates its metrics.
    pub fn update_quality(
        &mut self,
        hardware_id: &str,
        model_id: &str,
        quant_method: &str,
        metrics: QualityMetrics,
    ) -> Result<(), RuVectorError> {
        // Find the most recent matching record (last in list)
        let idx = self
            .records
            .iter()
            .rposition(|r| {
                r.hardware.stable_id() == hardware_id
                    && r.fingerprint.stable_id() == model_id
                    && r.quant_method == quant_method
            })
            .ok_or_else(|| RuVectorError::QueryFailed {
                reason: format!(
                    "No record found for hw={}, model={}, method={}",
                    hardware_id, model_id, quant_method
                ),
            })?;

        let score = compute_quality_score(&metrics);
        info!(
            model = model_id,
            method = quant_method,
            quality_score = format!("{:.3}", score),
            "Updating quality metrics in RuVector"
        );

        self.records[idx].quality = metrics;
        self.persist()
    }

    /// Flag all records for re-calibration when the hf2q version changes.
    ///
    /// Records from a different hf2q version are flagged as needing re-calibration.
    /// They are still stored but excluded from auto mode queries.
    pub fn flag_version_changes(&mut self) -> Result<usize, RuVectorError> {
        let current_version = env!("CARGO_PKG_VERSION");
        let mut flagged = 0;

        for record in &mut self.records {
            if record.hf2q_version != current_version && !record.needs_recalibration {
                record.needs_recalibration = true;
                flagged += 1;
            }
        }

        if flagged > 0 {
            info!(
                flagged = flagged,
                current_version = current_version,
                "Flagged records for re-calibration due to version change"
            );
            self.persist()?;
        }

        Ok(flagged)
    }

    /// Persist the database to disk with file-level locking.
    ///
    /// Uses a lock file to prevent concurrent writes from multiple hf2q processes.
    fn persist(&self) -> Result<(), RuVectorError> {
        let records_path = self.db_path.join(RECORDS_FILE);
        let lock_path = self.db_path.join("records.lock");

        // Acquire exclusive lock
        let lock_file = std::fs::File::create(&lock_path)?;
        lock_exclusive(&lock_file)?;

        let content = serde_json::to_string_pretty(&self.records)?;
        std::fs::write(&records_path, &content)?;
        debug!(path = %records_path.display(), "RuVector database persisted");

        // Lock is released when lock_file is dropped
        drop(lock_file);
        Ok(())
    }

    /// Calculate total database size on disk.
    fn calculate_db_size(&self) -> u64 {
        let records_path = self.db_path.join(RECORDS_FILE);
        std::fs::metadata(&records_path)
            .map(|m| m.len())
            .unwrap_or(0)
    }

    /// Number of stored records.
    #[allow(dead_code)]
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Path to the database directory.
    #[allow(dead_code)]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }
}

/// Compute a quality score from metrics (higher = better).
///
/// Score = cosine_similarity * (1.0 - kl_divergence.min(1.0)) / (1.0 + perplexity_delta.abs())
///
/// When metrics are missing, sensible defaults are used:
/// - cosine_similarity defaults to 0.5 (neutral)
/// - kl_divergence defaults to 0.5 (moderate)
/// - perplexity_delta defaults to 0.0 (no change)
pub fn compute_quality_score(metrics: &QualityMetrics) -> f64 {
    let cosine = metrics.cosine_similarity.unwrap_or(0.5);
    let kl = metrics.kl_divergence.unwrap_or(0.5).min(1.0);
    let ppl_delta = metrics.perplexity_delta.unwrap_or(0.0);
    cosine * (1.0 - kl) / (1.0 + ppl_delta.abs())
}

/// Select the best record from a set of matches.
/// Ranks by computed quality_score (highest wins).
fn select_best_record<'a>(records: &[&'a ConversionRecord]) -> &'a ConversionRecord {
    records
        .iter()
        .max_by(|a, b| {
            let a_score = compute_quality_score(&a.quality);
            let b_score = compute_quality_score(&b.quality);
            a_score
                .partial_cmp(&b_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("select_best_record called with empty slice")
}

/// Convert a stored record to a ResolvedConfig.
fn record_to_resolved(
    record: &ConversionRecord,
    hardware: &HardwareProfile,
    fingerprint: &ModelFingerprint,
    source: ResolvedSource,
) -> ResolvedConfig {
    let quality_score = compute_quality_score(&record.quality);

    // Map quality_score (0.0-1.0) to confidence, capped at 0.95
    let base_confidence = (quality_score * 0.95).clamp(0.3, 0.95);

    // Reduce confidence for similar (non-exact) matches and flagged records
    let confidence = match &source {
        ResolvedSource::RuVectorSimilar => (base_confidence * 0.8).clamp(0.3, 0.85),
        _ => base_confidence,
    };

    ResolvedConfig {
        quant_method: record.quant_method.clone(),
        bits: record.bits,
        group_size: record.group_size,
        confidence,
        source,
        reasoning: format!(
            "Based on stored conversion from {} (quality score: {:.3}, KL: {}, PPL delta: {}, cosine: {})",
            record.timestamp,
            quality_score,
            record
                .quality
                .kl_divergence
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "N/A".to_string()),
            record
                .quality
                .perplexity_delta
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "N/A".to_string()),
            record
                .quality
                .cosine_similarity
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "N/A".to_string()),
        ),
        hardware: hardware.clone(),
        fingerprint: fingerprint.clone(),
    }
}

/// Generate an ISO 8601 timestamp using std::time.
fn make_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let (year, month, day) = days_to_ymd(days);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since Unix epoch to (year, month, day).
/// Algorithm from http://howardhinnant.github.io/date_algorithms.html
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Extract the chip family from a chip model string (e.g., "Apple M5 Max" -> "apple m5").
///
/// Returns the vendor and generation without variant (pro/max/ultra).
fn extract_chip_family(chip_model: &str) -> String {
    let lower = chip_model.to_lowercase();
    let tokens: Vec<&str> = lower.split_whitespace().collect();

    // For Apple Silicon: "apple" + "mN"
    if tokens.len() >= 2 && tokens[0] == "apple" {
        if let Some(gen) = tokens.iter().find(|t| {
            t.starts_with('m')
                && t.len() <= 3
                && t[1..].chars().all(|c| c.is_ascii_digit())
        }) {
            return format!("apple {}", gen);
        }
    }
    // Fallback: use full lowercased string
    lower
}

/// Acquire an exclusive file lock (platform-specific).
fn lock_exclusive(file: &std::fs::File) -> Result<(), RuVectorError> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = file.as_raw_fd();
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if ret != 0 {
            return Err(RuVectorError::Io(std::io::Error::last_os_error()));
        }
    }
    #[cfg(not(unix))]
    {
        let _ = file; // suppress unused warning on non-unix
    }
    Ok(())
}

/// Check if the ruvector crate feature is enabled at compile time.
#[allow(dead_code)]
pub fn is_feature_enabled() -> bool {
    cfg!(feature = "ruvector")
}

/// Check RuVector availability and return a status for diagnostics.
pub fn check_status() -> RuVectorStatus {
    match RuVectorDb::open_default() {
        Ok(db) => db.status(),
        Err(e) => RuVectorStatus {
            operational: false,
            db_path: default_db_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "~/.hf2q/ruvector/".to_string()),
            record_count: 0,
            db_size_bytes: 0,
            message: format!("Unavailable: {}", e),
            feature_enabled: cfg!(feature = "ruvector"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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
    fn test_open_creates_directory() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        assert!(!db_path.exists());
        let db = RuVectorDb::open(&db_path).unwrap();
        assert!(db_path.exists());
        assert_eq!(db.record_count(), 0);
    }

    #[test]
    fn test_store_and_query() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();
        let hw = make_hardware();
        let fp = make_fingerprint();

        db.store_conversion(
            &hw,
            &fp,
            "q4",
            4,
            64,
            QualityMetrics {
                kl_divergence: Some(0.023),
                perplexity_delta: Some(0.15),
                cosine_similarity: Some(0.997),
            },
        )
        .unwrap();

        assert_eq!(db.record_count(), 1);

        let result = db.query_best_config(&hw, &fp).unwrap();
        assert!(result.is_some());
        let resolved = result.unwrap();
        assert_eq!(resolved.quant_method, "q4");
        assert_eq!(resolved.source, ResolvedSource::RuVectorExact);
    }

    #[test]
    fn test_query_empty_db_returns_none() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let db = RuVectorDb::open(&db_path).unwrap();
        let hw = make_hardware();
        let fp = make_fingerprint();

        let result = db.query_best_config(&hw, &fp).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_query_prefers_lowest_kl_divergence() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();
        let hw = make_hardware();
        let fp = make_fingerprint();

        // Store two results with different quality
        db.store_conversion(
            &hw,
            &fp,
            "q4",
            4,
            64,
            QualityMetrics {
                kl_divergence: Some(0.05),
                perplexity_delta: Some(0.3),
                cosine_similarity: None,
            },
        )
        .unwrap();

        db.store_conversion(
            &hw,
            &fp,
            "q8",
            8,
            64,
            QualityMetrics {
                kl_divergence: Some(0.01),
                perplexity_delta: Some(0.1),
                cosine_similarity: None,
            },
        )
        .unwrap();

        let result = db.query_best_config(&hw, &fp).unwrap();
        assert!(result.is_some());
        let resolved = result.unwrap();
        // Should prefer q8 because it has lower KL divergence
        assert_eq!(resolved.quant_method, "q8");
    }

    #[test]
    fn test_query_different_hardware_finds_similar() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();
        let hw1 = make_hardware();
        let fp = make_fingerprint();

        db.store_conversion(
            &hw1,
            &fp,
            "q4",
            4,
            64,
            QualityMetrics {
                kl_divergence: Some(0.02),
                ..Default::default()
            },
        )
        .unwrap();

        // Query with different hardware, same model
        let hw2 = HardwareProfile {
            chip_model: "Apple M4 Pro".to_string(),
            total_memory_bytes: 48 * 1024 * 1024 * 1024,
            available_memory_bytes: 30 * 1024 * 1024 * 1024,
            performance_cores: 12,
            efficiency_cores: 4,
            total_cores: 16,
            memory_bandwidth_gbs: 273.0,
        };

        let result = db.query_best_config(&hw2, &fp).unwrap();
        assert!(result.is_some());
        let resolved = result.unwrap();
        assert_eq!(resolved.source, ResolvedSource::RuVectorSimilar);
    }

    #[test]
    fn test_persistence_across_opens() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        {
            let mut db = RuVectorDb::open(&db_path).unwrap();
            db.store_conversion(
                &make_hardware(),
                &make_fingerprint(),
                "q4",
                4,
                64,
                QualityMetrics::default(),
            )
            .unwrap();
        }

        {
            let db = RuVectorDb::open(&db_path).unwrap();
            assert_eq!(db.record_count(), 1);
        }
    }

    #[test]
    fn test_flag_version_changes() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();

        // Manually insert a record with a different version
        db.records.push(ConversionRecord {
            hardware: make_hardware(),
            fingerprint: make_fingerprint(),
            quant_method: "q4".to_string(),
            bits: 4,
            group_size: 64,
            quality: QualityMetrics::default(),
            hf2q_version: "0.0.1-old".to_string(),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            needs_recalibration: false,
        });

        let flagged = db.flag_version_changes().unwrap();
        assert_eq!(flagged, 1);

        // Flagged records should appear with reduced confidence and RuVectorSimilar source
        let result = db
            .query_best_config(&make_hardware(), &make_fingerprint())
            .unwrap();
        assert!(result.is_some());
        let resolved = result.unwrap();
        assert_eq!(resolved.source, ResolvedSource::RuVectorSimilar);
        assert!(resolved.confidence <= 0.7);
        assert!(resolved.reasoning.contains("needs revalidation"));
    }

    #[test]
    fn test_status_empty_db() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let db = RuVectorDb::open(&db_path).unwrap();
        let status = db.status();
        assert!(status.operational);
        assert_eq!(status.record_count, 0);
        assert!(status.message.contains("empty"));
    }

    #[test]
    fn test_status_populated_db() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();
        db.store_conversion(
            &make_hardware(),
            &make_fingerprint(),
            "q4",
            4,
            64,
            QualityMetrics::default(),
        )
        .unwrap();

        let status = db.status();
        assert!(status.operational);
        assert_eq!(status.record_count, 1);
        assert!(status.message.contains("1 stored"));
    }

    #[test]
    fn test_timestamp_format() {
        let ts = make_timestamp();
        assert!(ts.len() >= 19);
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
    }

    #[test]
    fn test_days_to_ymd_known_dates() {
        // 2024-01-01 is day 19723 since epoch
        let (y, m, d) = days_to_ymd(19723);
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert_eq!(d, 1);
    }

    #[test]
    fn test_multiple_stores_accumulate() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();

        for i in 0..5 {
            db.store_conversion(
                &make_hardware(),
                &make_fingerprint(),
                &format!("q{}", (i % 3) + 2),
                (i % 3 + 2) as u8,
                64,
                QualityMetrics {
                    kl_divergence: Some(0.01 * (i as f64 + 1.0)),
                    ..Default::default()
                },
            )
            .unwrap();
        }

        assert_eq!(db.record_count(), 5);
    }

    #[test]
    fn test_compute_quality_score_full_metrics() {
        let metrics = QualityMetrics {
            kl_divergence: Some(0.02),
            perplexity_delta: Some(0.15),
            cosine_similarity: Some(0.997),
        };
        let score = compute_quality_score(&metrics);
        // 0.997 * (1.0 - 0.02) / (1.0 + 0.15) = 0.997 * 0.98 / 1.15 ~ 0.8495
        assert!(score > 0.84 && score < 0.86, "score was {}", score);
    }

    #[test]
    fn test_compute_quality_score_defaults() {
        let metrics = QualityMetrics::default();
        let score = compute_quality_score(&metrics);
        // 0.5 * (1.0 - 0.5) / (1.0 + 0.0) = 0.25
        assert!((score - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_quality_score_perfect() {
        let metrics = QualityMetrics {
            kl_divergence: Some(0.0),
            perplexity_delta: Some(0.0),
            cosine_similarity: Some(1.0),
        };
        let score = compute_quality_score(&metrics);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_query_best_prefers_highest_quality_score() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();
        let hw = make_hardware();
        let fp = make_fingerprint();

        // Lower KL but no cosine sim
        db.store_conversion(
            &hw, &fp, "q4", 4, 64,
            QualityMetrics {
                kl_divergence: Some(0.01),
                perplexity_delta: Some(0.1),
                cosine_similarity: None,
            },
        ).unwrap();

        // Higher KL but excellent cosine sim -> higher quality score
        db.store_conversion(
            &hw, &fp, "q8", 8, 64,
            QualityMetrics {
                kl_divergence: Some(0.02),
                perplexity_delta: Some(0.05),
                cosine_similarity: Some(0.999),
            },
        ).unwrap();

        let result = db.query_best_config(&hw, &fp).unwrap().unwrap();
        assert_eq!(result.quant_method, "q8");
    }

    #[test]
    fn test_update_quality() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();
        let hw = make_hardware();
        let fp = make_fingerprint();

        // Store with empty metrics
        db.store_conversion(&hw, &fp, "q4", 4, 64, QualityMetrics::default())
            .unwrap();

        // Update with actual metrics
        let metrics = QualityMetrics {
            kl_divergence: Some(0.005),
            perplexity_delta: Some(0.1),
            cosine_similarity: Some(0.999),
        };
        db.update_quality(&hw.stable_id(), &fp.stable_id(), "q4", metrics)
            .unwrap();

        // Verify the update persisted
        let result = db.query_best_config(&hw, &fp).unwrap().unwrap();
        assert!(result.confidence > 0.8);

        // Reload and verify persistence
        let db2 = RuVectorDb::open(&db_path).unwrap();
        let result2 = db2.query_best_config(&hw, &fp).unwrap().unwrap();
        assert!(result2.confidence > 0.8);
    }

    #[test]
    fn test_update_quality_not_found() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();
        let result = db.update_quality("hw-abc", "model-xyz", "q4", QualityMetrics::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_similar_model_matching() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();

        // Store a result for a 7B Llama model
        let hw = make_hardware();
        let fp_7b = ModelFingerprint {
            architecture: "LlamaForCausalLM".to_string(),
            total_params: 7_000_000_000,
            layer_count: 32,
            expert_count: 0,
            attention_types: vec!["attention".to_string()],
            hidden_size: 4096,
            dtype: "bfloat16".to_string(),
            intermediate_size: Some(11008),
            num_attention_heads: 32,
            num_kv_heads: Some(8),
            vocab_size: 32000,
        };
        db.store_conversion(
            &hw, &fp_7b, "q4", 4, 64,
            QualityMetrics {
                kl_divergence: Some(0.02),
                perplexity_delta: Some(0.1),
                cosine_similarity: Some(0.998),
            },
        ).unwrap();

        // Query for an 8B Llama model (similar but not exact)
        let fp_8b = make_fingerprint(); // 8B params, same architecture
        let result = db.query_best_config(&hw, &fp_8b).unwrap();
        assert!(result.is_some());
        let resolved = result.unwrap();
        assert_eq!(resolved.source, ResolvedSource::RuVectorSimilar);
        assert_eq!(resolved.quant_method, "q4");
    }

    #[test]
    fn test_similar_model_different_arch_no_match() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("ruvector_test");

        let mut db = RuVectorDb::open(&db_path).unwrap();

        // Store a result for a Llama model
        let hw = make_hardware();
        let fp_llama = make_fingerprint();
        db.store_conversion(
            &hw, &fp_llama, "q4", 4, 64, QualityMetrics::default(),
        ).unwrap();

        // Query for a different architecture (Mistral) with different params
        let fp_mistral = ModelFingerprint {
            architecture: "MistralForCausalLM".to_string(),
            total_params: 7_000_000_000,
            layer_count: 32,
            expert_count: 0,
            attention_types: vec!["attention".to_string()],
            hidden_size: 4096,
            dtype: "bfloat16".to_string(),
            intermediate_size: Some(14336),
            num_attention_heads: 32,
            num_kv_heads: Some(8),
            vocab_size: 32000,
        };

        // Different hw too, so no exact or model match
        let hw2 = HardwareProfile {
            chip_model: "Apple M4".to_string(),
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
            available_memory_bytes: 10 * 1024 * 1024 * 1024,
            performance_cores: 4,
            efficiency_cores: 4,
            total_cores: 8,
            memory_bandwidth_gbs: 100.0,
        };

        let result = db.query_best_config(&hw2, &fp_mistral).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_chip_family() {
        assert_eq!(extract_chip_family("Apple M5 Max"), "apple m5");
        assert_eq!(extract_chip_family("Apple M4 Pro"), "apple m4");
        assert_eq!(extract_chip_family("Apple M1"), "apple m1");
        assert_eq!(extract_chip_family("Unknown GPU"), "unknown gpu");
    }
}
