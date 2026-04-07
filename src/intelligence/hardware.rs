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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Estimated memory bandwidth in GB/s (derived from chip model)
    pub memory_bandwidth_gbs: f64,
}

// Manual impls so that memory_bandwidth_gbs (derived, f64) does not
// participate in Eq/Hash, which would break stable_id determinism and
// the Eq requirement (f64 is not Eq).
impl PartialEq for HardwareProfile {
    fn eq(&self, other: &Self) -> bool {
        self.chip_model == other.chip_model
            && self.total_memory_bytes == other.total_memory_bytes
            && self.available_memory_bytes == other.available_memory_bytes
            && self.performance_cores == other.performance_cores
            && self.efficiency_cores == other.efficiency_cores
            && self.total_cores == other.total_cores
    }
}

impl Eq for HardwareProfile {}

impl std::hash::Hash for HardwareProfile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.chip_model.hash(state);
        self.total_memory_bytes.hash(state);
        self.available_memory_bytes.hash(state);
        self.performance_cores.hash(state);
        self.efficiency_cores.hash(state);
        self.total_cores.hash(state);
    }
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

    /// Estimate tokens per second for a model of the given size.
    ///
    /// Uses a simplified model: tok/s ~ bandwidth_bytes / model_size_bytes.
    /// Returns 0.0 if `model_size_bytes` is 0.
    pub fn estimated_tok_s(&self, model_size_bytes: u64) -> f64 {
        if model_size_bytes == 0 {
            return 0.0;
        }
        let bandwidth_bytes = self.memory_bandwidth_gbs * 1e9;
        bandwidth_bytes / model_size_bytes as f64
    }

    /// Maximum model size (in bytes) that can sustain the target tok/s.
    ///
    /// Returns 0 if `target_tok_s` is <= 0.
    pub fn max_model_bytes_for_tok_s(&self, target_tok_s: f64) -> u64 {
        if target_tok_s <= 0.0 {
            return 0;
        }
        let bandwidth_bytes = self.memory_bandwidth_gbs * 1e9;
        (bandwidth_bytes / target_tok_s) as u64
    }
}

/// Look up memory bandwidth (GB/s) for a chip model string.
///
/// Parses strings like "Apple M5 Max", "Apple M1", etc. to extract the
/// generation and variant, then returns the published bandwidth.
/// Falls back to 100 GB/s for unrecognised chips.
pub fn lookup_memory_bandwidth_gbs(chip_model: &str) -> f64 {
    // Normalise to lowercase for matching
    let lower = chip_model.to_lowercase();

    // Try to find an "mN" pattern (m1, m2, m3, m4, m5 ...)
    let mut gen: Option<&str> = None;
    let mut variant: Option<&str> = None;

    // Tokenise on whitespace
    let tokens: Vec<&str> = lower.split_whitespace().collect();
    for (i, tok) in tokens.iter().enumerate() {
        if tok.starts_with('m') && tok.len() <= 3 {
            // Could be "m1", "m2", "m3", "m4", "m5" etc.
            if tok[1..].chars().all(|c| c.is_ascii_digit()) {
                gen = Some(*tok);
                // Check for a variant token following the generation
                if i + 1 < tokens.len() {
                    let next = tokens[i + 1];
                    if next == "pro" || next == "max" || next == "ultra" {
                        variant = Some(next);
                    }
                }
                break;
            }
        }
    }

    match (gen, variant) {
        // M1 family
        (Some("m1"), None) => 68.0,
        (Some("m1"), Some("pro")) => 200.0,
        (Some("m1"), Some("max")) => 400.0,
        (Some("m1"), Some("ultra")) => 800.0,
        // M2 family
        (Some("m2"), None) => 100.0,
        (Some("m2"), Some("pro")) => 200.0,
        (Some("m2"), Some("max")) => 400.0,
        (Some("m2"), Some("ultra")) => 800.0,
        // M3 family
        (Some("m3"), None) => 100.0,
        (Some("m3"), Some("pro")) => 150.0,
        (Some("m3"), Some("max")) => 400.0,
        (Some("m3"), Some("ultra")) => 800.0,
        // M4 family
        (Some("m4"), None) => 120.0,
        (Some("m4"), Some("pro")) => 273.0,
        (Some("m4"), Some("max")) => 546.0,
        // M5 family
        (Some("m5"), Some("max")) => 540.0,
        // Unknown
        _ => 100.0,
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

        let memory_bandwidth_gbs = lookup_memory_bandwidth_gbs(&chip_model);

        Ok(HardwareProfile {
            chip_model,
            total_memory_bytes,
            available_memory_bytes,
            performance_cores,
            efficiency_cores,
            total_cores,
            memory_bandwidth_gbs,
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

    fn make_profile(chip: &str) -> HardwareProfile {
        HardwareProfile {
            chip_model: chip.to_string(),
            total_memory_bytes: 128 * 1024 * 1024 * 1024,
            available_memory_bytes: 64 * 1024 * 1024 * 1024,
            performance_cores: 14,
            efficiency_cores: 4,
            total_cores: 18,
            memory_bandwidth_gbs: lookup_memory_bandwidth_gbs(chip),
        }
    }

    #[test]
    fn test_hardware_profile_stable_id_deterministic() {
        let profile = make_profile("Apple M5 Max");
        let id1 = profile.stable_id();
        let id2 = profile.stable_id();
        assert_eq!(id1, id2);
        assert!(id1.starts_with("hw-"));
    }

    #[test]
    fn test_hardware_profile_stable_id_ignores_available_memory() {
        let profile1 = make_profile("Apple M5 Max");
        let mut profile2 = make_profile("Apple M5 Max");
        profile2.available_memory_bytes = 32 * 1024 * 1024 * 1024;

        assert_eq!(profile1.stable_id(), profile2.stable_id());
    }

    #[test]
    fn test_stable_id_ignores_bandwidth() {
        let mut p1 = make_profile("Apple M4 Max");
        let mut p2 = make_profile("Apple M4 Max");
        p1.memory_bandwidth_gbs = 546.0;
        p2.memory_bandwidth_gbs = 999.0; // artificially different
        assert_eq!(p1.stable_id(), p2.stable_id());
    }

    #[test]
    fn test_hardware_profile_different_chips_different_ids() {
        let profile1 = make_profile("Apple M5 Max");
        let mut profile2 = make_profile("Apple M4 Pro");
        profile2.total_memory_bytes = 48 * 1024 * 1024 * 1024;
        profile2.performance_cores = 12;
        profile2.total_cores = 16;

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
            memory_bandwidth_gbs: 100.0,
        };

        assert!((profile.total_memory_gb() - 128.0).abs() < 0.01);
        assert!((profile.available_memory_gb() - 64.0).abs() < 0.01);
    }

    #[test]
    fn test_detect_runs_without_panic() {
        let result = HardwareProfiler::detect();
        assert!(result.is_ok());
        let profile = result.unwrap();
        assert!(profile.total_memory_bytes > 0);
        assert!(profile.total_cores > 0);
        assert!(!profile.chip_model.is_empty());
        assert!(profile.memory_bandwidth_gbs > 0.0);
    }

    // ---- Bandwidth lookup tests ----

    #[test]
    fn test_bandwidth_known_chips() {
        assert!((lookup_memory_bandwidth_gbs("Apple M1") - 68.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M1 Pro") - 200.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M1 Max") - 400.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M1 Ultra") - 800.0).abs() < f64::EPSILON);

        assert!((lookup_memory_bandwidth_gbs("Apple M2") - 100.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M2 Pro") - 200.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M2 Max") - 400.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M2 Ultra") - 800.0).abs() < f64::EPSILON);

        assert!((lookup_memory_bandwidth_gbs("Apple M3") - 100.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M3 Pro") - 150.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M3 Max") - 400.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M3 Ultra") - 800.0).abs() < f64::EPSILON);

        assert!((lookup_memory_bandwidth_gbs("Apple M4") - 120.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M4 Pro") - 273.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Apple M4 Max") - 546.0).abs() < f64::EPSILON);

        assert!((lookup_memory_bandwidth_gbs("Apple M5 Max") - 540.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_unknown_chip_fallback() {
        assert!((lookup_memory_bandwidth_gbs("Intel Core i9-13900K") - 100.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("Unknown (arm64)") - 100.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bandwidth_case_insensitive() {
        assert!((lookup_memory_bandwidth_gbs("apple m4 max") - 546.0).abs() < f64::EPSILON);
        assert!((lookup_memory_bandwidth_gbs("APPLE M4 MAX") - 546.0).abs() < f64::EPSILON);
    }

    // ---- Convenience method tests ----

    #[test]
    fn test_estimated_tok_s() {
        let profile = make_profile("Apple M4 Max"); // 546 GB/s
        // 546 GB/s = 546e9 B/s, model = 10 GB = 10e9 B
        // tok/s = 546e9 / 10e9 = 54.6
        let tok_s = profile.estimated_tok_s(10_000_000_000);
        assert!((tok_s - 54.6).abs() < 0.01);

        // Zero model size returns 0
        assert!((profile.estimated_tok_s(0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_max_model_bytes_for_tok_s() {
        let profile = make_profile("Apple M4 Max"); // 546 GB/s
        // target 10 tok/s -> 546e9 / 10 = 54.6e9
        let max_bytes = profile.max_model_bytes_for_tok_s(10.0);
        assert_eq!(max_bytes, 54_600_000_000);

        // Zero / negative target returns 0
        assert_eq!(profile.max_model_bytes_for_tok_s(0.0), 0);
        assert_eq!(profile.max_model_bytes_for_tok_s(-1.0), 0);
    }
}
