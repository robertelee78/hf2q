//! RuVector self-learning integration (Epic 7).
//!
//! Stores and retrieves conversion results for auto mode optimization.
//! Not yet implemented — stub types provided for compilation.

use thiserror::Error;

use super::fingerprint::ModelFingerprint;
use super::hardware::HardwareProfile;
use super::ResolvedConfig;

/// Errors from RuVector operations.
#[derive(Error, Debug)]
pub enum RuVectorError {
    #[error("RuVector integration is not yet implemented (Epic 7)")]
    NotImplemented,

    #[error("RuVector database not accessible at {path}: {reason}")]
    DatabaseUnavailable { path: String, reason: String },

    #[error("RuVector query failed: {reason}")]
    QueryFailed { reason: String },

    #[error("RuVector store failed: {reason}")]
    StoreFailed { reason: String },
}

/// Stub RuVector database handle for compilation.
/// Full implementation in Epic 7.
#[allow(dead_code)]
pub struct RuVectorDb;

#[allow(dead_code)]
impl RuVectorDb {
    /// Query for the best stored conversion config for the given hardware+model.
    pub fn query_best_config(
        &self,
        _hardware: &HardwareProfile,
        _fingerprint: &ModelFingerprint,
    ) -> Result<Option<ResolvedConfig>, RuVectorError> {
        // Not yet implemented — always returns None (no stored data)
        Ok(None)
    }
}
