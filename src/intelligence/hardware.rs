//! Hardware profiler via sysinfo (Epic 6).
//!
//! Not yet implemented.

use thiserror::Error;

/// Errors from hardware profiling.
#[derive(Error, Debug)]
pub enum HardwareError {
    #[error("Hardware profiling is not yet implemented (Epic 6)")]
    NotImplemented,
}

/// Hardware profile for a machine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HardwareProfile {
    /// Chip model (e.g., "Apple M5 Max")
    pub chip_model: String,
    /// Total unified memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes
    pub available_memory_bytes: u64,
    /// Number of performance cores
    pub performance_cores: u32,
    /// Number of efficiency cores
    pub efficiency_cores: u32,
}
