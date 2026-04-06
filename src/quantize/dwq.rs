//! Distilled Weight Quantization (DWQ) calibration engine (Epic 5).
//!
//! Uses InferenceRunner for forward passes during calibration.
//! Not yet implemented.

use thiserror::Error;

/// Errors from DWQ calibration.
#[derive(Error, Debug)]
pub enum DwqError {
    #[error("DWQ calibration is not yet implemented (Epic 5)")]
    NotImplemented,

    #[error("Calibration data not available: {reason}")]
    NoCalibrationData { reason: String },

    #[error("Inference runner error: {0}")]
    InferenceError(String),
}
