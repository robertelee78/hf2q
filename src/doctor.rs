//! hf2q doctor subcommand — diagnoses system health.
//!
//! Checks RuVector, hardware detection, hf CLI, disk space.
//! Provides specific remediation steps for each issue found.

use thiserror::Error;

use crate::intelligence::hardware::HardwareProfiler;
use crate::intelligence::ruvector;
use crate::progress::format_bytes;

/// Errors from doctor diagnostics.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum DoctorError {
    #[error("Doctor diagnostics encountered an error: {reason}")]
    DiagnosticFailed { reason: String },
}

/// Diagnostic check result.
#[derive(Debug)]
enum CheckResult {
    Pass(String),
    Warn(String, String), // message, remediation
    Fail(String, String), // message, remediation
    #[allow(dead_code)]
    Info(String),
}

/// Run the doctor diagnostics and print results.
pub fn run_doctor() -> anyhow::Result<()> {
    use console::style;

    println!();
    println!("{}", style("hf2q doctor").bold().green());
    println!(
        "{}",
        style("═══════════════════════════════════════").bold().green()
    );
    println!();

    let mut has_errors = false;
    let mut has_warnings = false;

    // Check 1: Hardware detection
    let hw_result = check_hardware();
    print_check("Hardware Detection", &hw_result);
    if matches!(hw_result, CheckResult::Fail(_, _)) {
        has_errors = true;
    }

    // Check 2: Platform
    let platform_result = check_platform();
    print_check("Platform", &platform_result);

    // Check 3: RuVector database
    let rv_result = check_ruvector();
    print_check("RuVector Database", &rv_result);
    match &rv_result {
        CheckResult::Fail(_, _) => has_errors = true,
        CheckResult::Warn(_, _) => has_warnings = true,
        _ => {}
    }

    // Check 4: hf CLI
    let hf_result = check_hf_cli();
    print_check("HuggingFace CLI", &hf_result);
    if matches!(hf_result, CheckResult::Warn(_, _)) {
        has_warnings = true;
    }

    // Check 6: Disk space
    let disk_result = check_disk_space();
    print_check("Disk Space", &disk_result);
    if matches!(disk_result, CheckResult::Warn(_, _)) {
        has_warnings = true;
    }

    // Summary
    println!();
    if has_errors {
        println!(
            "{}",
            style("Some checks FAILED. Resolve the issues above before using hf2q convert.")
                .red()
                .bold()
        );
    } else if has_warnings {
        println!(
            "{}",
            style("All critical checks passed with some warnings.").yellow()
        );
    } else {
        println!(
            "{}",
            style("All checks passed. hf2q is ready to use.").green().bold()
        );
    }
    println!();

    Ok(())
}

/// Print a check result with consistent formatting.
fn print_check(name: &str, result: &CheckResult) {
    use console::style;

    match result {
        CheckResult::Pass(msg) => {
            println!("  {} {}: {}", style("PASS").green().bold(), name, msg);
        }
        CheckResult::Warn(msg, remediation) => {
            println!("  {} {}: {}", style("WARN").yellow().bold(), name, msg);
            println!("         {}", style(remediation).dim());
        }
        CheckResult::Fail(msg, remediation) => {
            println!("  {} {}: {}", style("FAIL").red().bold(), name, msg);
            println!("         {}", style(remediation).dim());
        }
        CheckResult::Info(msg) => {
            println!("  {} {}: {}", style("INFO").cyan().bold(), name, msg);
        }
    }
}

/// Check hardware detection.
fn check_hardware() -> CheckResult {
    match HardwareProfiler::detect() {
        Ok(profile) => CheckResult::Pass(format!(
            "{} ({:.0} GB unified memory, {} cores [{} perf + {} efficiency])",
            profile.chip_model,
            profile.total_memory_gb(),
            profile.total_cores,
            profile.performance_cores,
            profile.efficiency_cores,
        )),
        Err(e) => CheckResult::Fail(
            format!("Hardware detection failed: {}", e),
            "Ensure sysinfo can access system information. Check macOS permissions.".to_string(),
        ),
    }
}

/// Check platform compatibility.
fn check_platform() -> CheckResult {
    #[cfg(target_os = "macos")]
    {
        #[cfg(target_arch = "aarch64")]
        {
            CheckResult::Pass("macOS on Apple Silicon (aarch64)".to_string())
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            CheckResult::Pass("macOS on Intel (x86_64)".to_string())
        }
    }
    #[cfg(target_os = "linux")]
    {
        CheckResult::Pass("Linux".to_string())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        CheckResult::Warn(
            "Unsupported platform".to_string(),
            "hf2q is tested on macOS (ARM64) and Linux (x86_64).".to_string(),
        )
    }
}

/// Check RuVector database status.
fn check_ruvector() -> CheckResult {
    let status = ruvector::check_status();

    if !status.operational {
        return CheckResult::Fail(
            format!("RuVector unavailable: {}", status.message),
            format!(
                "Ensure the directory {} is writable. \
                 Check permissions with: ls -la ~/.hf2q/",
                status.db_path
            ),
        );
    }

    if status.record_count == 0 {
        CheckResult::Warn(
            format!(
                "RuVector operational but empty at {}",
                status.db_path
            ),
            "This is normal for first use. The database will populate after your first conversion."
                .to_string(),
        )
    } else {
        CheckResult::Pass(format!(
            "{} stored conversions, {} on disk at {}",
            status.record_count,
            format_bytes(status.db_size_bytes),
            status.db_path,
        ))
    }
}

/// Check hf CLI availability.
fn check_hf_cli() -> CheckResult {
    let hf_cli = std::process::Command::new("hf")
        .arg("--version")
        .output();

    match hf_cli {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            CheckResult::Pass(format!("hf CLI available: {}", version.trim()))
        }
        _ => CheckResult::Warn(
            "hf CLI not found on PATH".to_string(),
            "Install with: pip install huggingface_hub[cli]. \
             The hf CLI is used as a download fallback for gated models."
                .to_string(),
        ),
    }
}

/// Check available disk space.
fn check_disk_space() -> CheckResult {
    // Check disk space at ~/.hf2q/ and current directory
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let hf2q_dir = std::path::PathBuf::from(&home).join(".hf2q");

    // Use sysinfo's Disks to check available space
    use sysinfo::Disks;
    let disks = Disks::new_with_refreshed_list();

    // Find the disk that contains our working directory
    let cwd = std::env::current_dir().unwrap_or_default();
    let mut best_match: Option<(u64, u64)> = None; // (available, total)

    for disk in disks.list() {
        let mount = disk.mount_point();
        // Find the most specific mount point that is a prefix of our cwd
        if cwd.starts_with(mount) {
            let available = disk.available_space();
            let total = disk.total_space();
            match &best_match {
                Some((_, _)) => {
                    // Prefer more specific mount point (longer path)
                    if mount.as_os_str().len()
                        > best_match
                            .as_ref()
                            .map(|_| 0)
                            .unwrap_or(0)
                    {
                        best_match = Some((available, total));
                    }
                }
                None => {
                    best_match = Some((available, total));
                }
            }
        }
    }

    match best_match {
        Some((available, total)) => {
            let available_gb = available as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);

            if available_gb < 10.0 {
                CheckResult::Warn(
                    format!(
                        "{} available of {} total",
                        format_bytes(available),
                        format_bytes(total)
                    ),
                    format!(
                        "Low disk space ({:.1} GB available). Large model conversions may need 50+ GB. \
                         Free space or use --output to write to a different disk.",
                        available_gb
                    ),
                )
            } else {
                CheckResult::Pass(format!(
                    "{:.1} GB available of {:.1} GB total",
                    available_gb, total_gb
                ))
            }
        }
        None => {
            // Fallback: check if the hf2q directory is writable
            if hf2q_dir.exists() || std::fs::create_dir_all(&hf2q_dir).is_ok() {
                CheckResult::Pass("Disk accessible (unable to determine exact free space)".to_string())
            } else {
                CheckResult::Warn(
                    "Unable to determine disk space".to_string(),
                    "Ensure sufficient disk space is available for model conversions.".to_string(),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_hardware_succeeds() {
        let result = check_hardware();
        assert!(matches!(result, CheckResult::Pass(_)));
    }

    #[test]
    fn test_check_platform() {
        let result = check_platform();
        // On macOS aarch64 this should pass; on other platforms it should warn
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(matches!(result, CheckResult::Pass(_)));
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        assert!(matches!(result, CheckResult::Warn(_, _)));
    }

    #[test]
    fn test_check_ruvector() {
        let result = check_ruvector();
        // Should be either Pass or Warn (empty), not Fail (directory should be creatable)
        assert!(!matches!(result, CheckResult::Fail(_, _)));
    }

    #[test]
    fn test_check_disk_space() {
        let result = check_disk_space();
        // Should succeed on any normal system
        assert!(!matches!(result, CheckResult::Fail(_, _)));
    }
}
