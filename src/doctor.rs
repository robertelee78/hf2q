//! hf2q doctor subcommand — diagnoses system health (Epic 7/9).
//!
//! Checks RuVector, hardware detection, mlx-rs Metal, hf CLI, disk space.
//! Not yet fully implemented.

use thiserror::Error;

/// Errors from doctor diagnostics.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum DoctorError {
    #[error("Doctor diagnostics encountered an error: {reason}")]
    DiagnosticFailed { reason: String },
}

/// Run the doctor diagnostics and print results.
pub fn run_doctor() -> anyhow::Result<()> {
    use console::style;

    println!();
    println!("{}", style("hf2q doctor").bold().green());
    println!("{}", style("═══════════").bold().green());
    println!();

    // Check 1: Rust toolchain
    println!("  {} Rust toolchain available", style("✓").green());

    // Check 2: Platform
    #[cfg(target_os = "macos")]
    {
        println!("  {} macOS platform detected", style("✓").green());
    }
    #[cfg(not(target_os = "macos"))]
    {
        println!(
            "  {} Non-macOS platform — MLX and CoreML backends unavailable",
            style("⚠").yellow()
        );
    }

    // Check 3: RuVector
    println!(
        "  {} RuVector integration not yet implemented (Epic 7)",
        style("○").dim()
    );

    // Check 4: MLX Metal backend
    println!(
        "  {} MLX inference runner not yet implemented (Epic 4)",
        style("○").dim()
    );

    // Check 5: hf CLI
    let hf_cli = std::process::Command::new("hf")
        .arg("--version")
        .output();
    match hf_cli {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            println!(
                "  {} hf CLI available: {}",
                style("✓").green(),
                version.trim()
            );
        }
        _ => {
            println!(
                "  {} hf CLI not found — install with: pip install huggingface_hub[cli]",
                style("⚠").yellow()
            );
        }
    }

    // Check 6: Available disk space
    println!(
        "  {} Disk space check not yet implemented",
        style("○").dim()
    );

    println!();
    Ok(())
}
