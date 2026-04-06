//! Intelligence module — hardware profiling, model fingerprinting, auto mode.
//!
//! Orchestrates:
//! - HardwareProfiler: detect chip, memory, cores
//! - ModelFingerprint: stable identifier from config.json
//! - AutoResolver: RuVector query → heuristic fallback
//! - RuVector: self-learning conversion result storage

pub mod fingerprint;
pub mod hardware;
pub mod heuristics;
pub mod ruvector;

use thiserror::Error;

/// Errors from intelligence operations.
#[derive(Error, Debug)]
pub enum IntelligenceError {
    #[error("Auto mode is not yet implemented (Epic 6)")]
    NotImplemented,

    #[error("Hardware profiling failed: {reason}")]
    HardwareDetectionFailed { reason: String },

    #[error("RuVector is not accessible: {reason}. Required to store learnings. Run `hf2q doctor` to diagnose.")]
    RuVectorUnavailable { reason: String },
}
