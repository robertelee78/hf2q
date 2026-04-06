//! Progress reporting abstraction wrapping indicatif.
//!
//! All progress bar usage in the codebase goes through `ProgressReporter`.
//! No direct indicatif calls outside this module.

use std::time::Instant;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use thiserror::Error;

/// Errors from progress operations.
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum ProgressError {
    #[error("Failed to create progress bar: {0}")]
    Creation(String),
}

/// Central progress reporting abstraction.
/// Wraps indicatif's `MultiProgress` for concurrent phase tracking.
pub struct ProgressReporter {
    multi: MultiProgress,
    start_time: Instant,
}

impl ProgressReporter {
    /// Create a new progress reporter.
    pub fn new() -> Self {
        Self {
            multi: MultiProgress::new(),
            start_time: Instant::now(),
        }
    }

    /// Create a spinner with a message (for indeterminate-length operations).
    #[allow(dead_code)]
    pub fn spinner(&self, message: &str) -> ProgressBar {
        let pb = self.multi.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner()),
        );
        pb.set_message(message.to_string());
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        pb
    }

    /// Create a progress bar with a known total count.
    pub fn bar(&self, total: u64, message: &str) -> ProgressBar {
        let pb = self.multi.add(ProgressBar::new(total));
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})"
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=> "),
        );
        pb.set_message(message.to_string());
        pb
    }

    /// Create a byte-counting progress bar (for file I/O).
    #[allow(dead_code)]
    pub fn bytes_bar(&self, total_bytes: u64, message: &str) -> ProgressBar {
        let pb = self.multi.add(ProgressBar::new(total_bytes));
        pb.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})"
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=> "),
        );
        pb.set_message(message.to_string());
        pb
    }

    /// Elapsed time since the reporter was created.
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Format elapsed time for display.
    pub fn elapsed_display(&self) -> String {
        let elapsed = self.elapsed();
        let secs = elapsed.as_secs();
        if secs < 60 {
            format!("{:.1}s", elapsed.as_secs_f64())
        } else if secs < 3600 {
            format!("{}m {:02}s", secs / 60, secs % 60)
        } else {
            format!("{}h {:02}m {:02}s", secs / 3600, (secs % 3600) / 60, secs % 60)
        }
    }
}

impl Default for ProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Print a conversion summary to the terminal.
#[allow(clippy::too_many_arguments)]
pub fn print_summary(
    model_name: &str,
    architecture: &str,
    param_count: u64,
    quant_method: &str,
    input_size_bytes: u64,
    output_size_bytes: u64,
    output_dir: &str,
    elapsed: &str,
) {
    use console::style;

    let compression_ratio = if output_size_bytes > 0 {
        input_size_bytes as f64 / output_size_bytes as f64
    } else {
        0.0
    };

    println!();
    println!("{}", style("═══ Conversion Summary ═══").bold().green());
    println!("  Model:        {}", style(model_name).bold());
    println!("  Architecture: {}", architecture);
    println!("  Parameters:   {}", format_param_count(param_count));
    println!("  Quantization: {}", style(quant_method).cyan());
    println!(
        "  Input size:   {}",
        format_bytes(input_size_bytes)
    );
    println!(
        "  Output size:  {} ({:.1}x compression)",
        format_bytes(output_size_bytes),
        compression_ratio
    );
    println!("  Output:       {}", style(output_dir).underlined());
    println!("  Elapsed:      {}", elapsed);
    println!("{}", style("══════════════════════════").bold().green());
    println!();
}

/// Format a byte count for human display.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format a parameter count for human display.
pub fn format_param_count(count: u64) -> String {
    const BILLION: u64 = 1_000_000_000;
    const MILLION: u64 = 1_000_000;

    if count >= BILLION {
        format!("{:.1}B", count as f64 / BILLION as f64)
    } else if count >= MILLION {
        format!("{:.1}M", count as f64 / MILLION as f64)
    } else {
        format!("{}", count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.46 KB");
        assert_eq!(format_bytes(1_500_000), "1.43 MB");
        assert_eq!(format_bytes(1_500_000_000), "1.40 GB");
    }

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(25_805_933_872), "25.8B");
        assert_eq!(format_param_count(7_000_000_000), "7.0B");
        assert_eq!(format_param_count(350_000_000), "350.0M");
    }

    #[test]
    fn test_progress_reporter_elapsed() {
        let reporter = ProgressReporter::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(reporter.elapsed().as_millis() >= 10);
    }
}
