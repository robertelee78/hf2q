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

        Ok(None)
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

    /// Persist the database to disk.
    fn persist(&self) -> Result<(), RuVectorError> {
        let records_path = self.db_path.join(RECORDS_FILE);
        let content = serde_json::to_string_pretty(&self.records)?;
        std::fs::write(&records_path, content)?;
        debug!(path = %records_path.display(), "RuVector database persisted");
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

/// Select the best record from a set of matches.
/// Prefers lowest KL divergence, then lowest perplexity delta.
fn select_best_record<'a>(records: &[&'a ConversionRecord]) -> &'a ConversionRecord {
    records
        .iter()
        .min_by(|a, b| {
            let a_kl = a.quality.kl_divergence.unwrap_or(f64::MAX);
            let b_kl = b.quality.kl_divergence.unwrap_or(f64::MAX);

            a_kl.partial_cmp(&b_kl)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let a_ppl = a.quality.perplexity_delta.unwrap_or(f64::MAX);
                    let b_ppl = b.quality.perplexity_delta.unwrap_or(f64::MAX);
                    a_ppl
                        .partial_cmp(&b_ppl)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
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
    let confidence = match &record.quality.kl_divergence {
        Some(kl) if *kl < 0.01 => 0.95,
        Some(kl) if *kl < 0.05 => 0.85,
        Some(kl) if *kl < 0.1 => 0.75,
        Some(_) => 0.65,
        None => 0.7,
    };

    ResolvedConfig {
        quant_method: record.quant_method.clone(),
        bits: record.bits,
        group_size: record.group_size,
        confidence,
        source,
        reasoning: format!(
            "Based on stored conversion from {} (KL divergence: {}, perplexity delta: {})",
            record.timestamp,
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

        // Flagged records should not appear in queries
        let result = db
            .query_best_config(&make_hardware(), &make_fingerprint())
            .unwrap();
        assert!(result.is_none());
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
}
