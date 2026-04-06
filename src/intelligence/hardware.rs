//! Hardware profiler via sysinfo.
//!
//! Detects chip model, memory, and core counts for auto mode decisions.
//! Produces a stable, hashable hardware identifier for RuVector lookups.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from hardware profiling.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum HardwareError {
    #[error("Hardware profiling failed: {reason}")]
    DetectionFailed { reason: String },

    #[error("System information unavailable: {field}")]
    FieldUnavailable { field: String },
}

/// Hardware profile for a machine.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct HardwareProfile {
    /// Chip model (e.g., "Apple M5 Max")
    pub chip_model: String,
    /// Total unified memory in bytes
    pub total_memory_bytes: u64,
    /// Available memory in bytes at detection time
    pub available_memory_bytes: u64,
    /// Number of performance cores
    pub performance_cores: u32,
    /// Number of efficiency cores
    pub efficiency_cores: u32,
    /// Total core count (performance + efficiency)
    pub total_cores: u32,
}

impl HardwareProfile {
    /// Produce a stable, hashable identifier string for RuVector lookups.
    ///
    /// The ID is based on chip model and total memory — not available memory,
    /// which fluctuates. Two machines with the same chip and RAM produce the same ID.
    pub fn stable_id(&self) -> String {
        let mut hasher = DefaultHasher::new();
        self.chip_model.hash(&mut hasher);
        self.total_memory_bytes.hash(&mut hasher);
        self.total_cores.hash(&mut hasher);
        format!("hw-{:016x}", hasher.finish())
    }

    /// Total memory in gigabytes (for display and heuristic calculations).
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Available memory in gigabytes.
    pub fn available_memory_gb(&self) -> f64 {
        self.available_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Hardware profiler that detects the current machine's capabilities.
pub struct HardwareProfiler;

impl HardwareProfiler {
    /// Detect hardware profile of the current machine.
    ///
    /// Uses sysinfo for cross-platform memory/CPU info, and macOS-specific
    /// sysctls for Apple Silicon chip model identification.
    pub fn detect() -> Result<HardwareProfile, HardwareError> {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_all();

        let total_memory_bytes = sys.total_memory();
        let available_memory_bytes = sys.available_memory();

        // Get CPU info
        let cpus = sys.cpus();
        let total_cores = cpus.len() as u32;

        // Detect chip model — on macOS, use sysctl for Apple Silicon identification
        let chip_model = detect_chip_model(&sys)?;

        // On Apple Silicon, performance/efficiency core split is typically:
        // M1: 4P+4E, M1 Pro: 8P+2E or 6P+2E, M1 Max: 8P+2E, M1 Ultra: 16P+4E
        // M2: 4P+4E, M2 Pro: 8P+4E or 6P+4E, M2 Max: 8P+4E, M2 Ultra: 16P+8E
        // M3: 4P+4E, M3 Pro: 6P+6E or 5P+6E, M3 Max: 12P+4E or 10P+4E
        // M4: 4P+6E, M4 Pro: 10P+4E or 12P+4E, M4 Max: 12P+4E or 14P+4E
        // We use sysctl to get the actual counts on macOS.
        let (performance_cores, efficiency_cores) = detect_core_counts(total_cores);

        Ok(HardwareProfile {
            chip_model,
            total_memory_bytes,
            available_memory_bytes,
            performance_cores,
            efficiency_cores,
            total_cores,
        })
    }
}

/// Detect the chip model name.
///
/// On macOS, reads `machdep.cpu.brand_string` via sysctl for the CPU brand,
/// and attempts to identify Apple Silicon chip model from system info.
fn detect_chip_model(sys: &sysinfo::System) -> Result<String, HardwareError> {
    // Try macOS sysctl first for Apple Silicon detection
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
        {
            if output.status.success() {
                let brand = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !brand.is_empty() {
                    return Ok(brand);
                }
            }
        }
    }

    // Fallback: use sysinfo CPU brand
    let cpus = sys.cpus();
    if let Some(cpu) = cpus.first() {
        let brand = cpu.brand().to_string();
        if !brand.is_empty() {
            return Ok(brand);
        }
    }

    // Last resort: return a generic identifier
    Ok(format!("Unknown ({})", std::env::consts::ARCH))
}

/// Detect performance and efficiency core counts.
///
/// On macOS, uses sysctl to read hw.perflevel counts.
/// On other platforms, assumes all cores are performance cores.
fn detect_core_counts(total_cores: u32) -> (u32, u32) {
    #[cfg(target_os = "macos")]
    {
        // Try to read performance and efficiency core counts from sysctl
        let perf = read_sysctl_u32("hw.perflevel0.logicalcpu_max");
        let eff = read_sysctl_u32("hw.perflevel1.logicalcpu_max");

        if let (Some(p), Some(e)) = (perf, eff) {
            return (p, e);
        }
    }

    // Fallback: assume all cores are performance cores
    (total_cores, 0)
}

/// Read a u32 value from sysctl (macOS only).
#[cfg(target_os = "macos")]
fn read_sysctl_u32(key: &str) -> Option<u32> {
    let output = std::process::Command::new("sysctl")
        .arg("-n")
        .arg(key)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let value_str = String::from_utf8_lossy(&output.stdout);
    value_str.trim().parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_profile_stable_id_deterministic() {
        let profile = HardwareProfile {
            chip_model: "Apple M5 Max".to_string(),
            total_memory_bytes: 128 * 1024 * 1024 * 1024,
            available_memory_bytes: 64 * 1024 * 1024 * 1024,
            performance_cores: 14,
            efficiency_cores: 4,
            total_cores: 18,
        };

        let id1 = profile.stable_id();
        let id2 = profile.stable_id();
        assert_eq!(id1, id2);
        assert!(id1.starts_with("hw-"));
    }

    #[test]
    fn test_hardware_profile_stable_id_ignores_available_memory() {
        let profile1 = HardwareProfile {
            chip_model: "Apple M5 Max".to_string(),
            total_memory_bytes: 128 * 1024 * 1024 * 1024,
            available_memory_bytes: 64 * 1024 * 1024 * 1024,
            performance_cores: 14,
            efficiency_cores: 4,
            total_cores: 18,
        };
        let profile2 = HardwareProfile {
            chip_model: "Apple M5 Max".to_string(),
            total_memory_bytes: 128 * 1024 * 1024 * 1024,
            available_memory_bytes: 32 * 1024 * 1024 * 1024, // different
            performance_cores: 14,
            efficiency_cores: 4,
            total_cores: 18,
        };

        assert_eq!(profile1.stable_id(), profile2.stable_id());
    }

    #[test]
    fn test_hardware_profile_different_chips_different_ids() {
        let profile1 = HardwareProfile {
            chip_model: "Apple M5 Max".to_string(),
            total_memory_bytes: 128 * 1024 * 1024 * 1024,
            available_memory_bytes: 64 * 1024 * 1024 * 1024,
            performance_cores: 14,
            efficiency_cores: 4,
            total_cores: 18,
        };
        let profile2 = HardwareProfile {
            chip_model: "Apple M4 Pro".to_string(),
            total_memory_bytes: 48 * 1024 * 1024 * 1024,
            available_memory_bytes: 32 * 1024 * 1024 * 1024,
            performance_cores: 12,
            efficiency_cores: 4,
            total_cores: 16,
        };

        assert_ne!(profile1.stable_id(), profile2.stable_id());
    }

    #[test]
    fn test_memory_gb_conversion() {
        let profile = HardwareProfile {
            chip_model: "Test".to_string(),
            total_memory_bytes: 128 * 1024 * 1024 * 1024,
            available_memory_bytes: 64 * 1024 * 1024 * 1024,
            performance_cores: 8,
            efficiency_cores: 4,
            total_cores: 12,
        };

        assert!((profile.total_memory_gb() - 128.0).abs() < 0.01);
        assert!((profile.available_memory_gb() - 64.0).abs() < 0.01);
    }

    #[test]
    fn test_detect_runs_without_panic() {
        // This test just ensures detect() doesn't panic on the current machine.
        // We can't assert specific values since they depend on the hardware.
        let result = HardwareProfiler::detect();
        assert!(result.is_ok());
        let profile = result.unwrap();
        assert!(profile.total_memory_bytes > 0);
        assert!(profile.total_cores > 0);
        assert!(!profile.chip_model.is_empty());
    }
}
