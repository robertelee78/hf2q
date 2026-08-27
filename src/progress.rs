//! Progress reporting abstraction wrapping indicatif.
//!
//! All progress bar usage in the codebase goes through `ProgressReporter`.
//! No direct indicatif calls outside this module.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
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
            ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
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
                "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} {percent:>3}% ({bytes_per_sec}, {eta})",
            )
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=> "),
        );
        pb.set_message(message.to_string());
        pb
    }

    /// Adapt hf-hub's concurrent HTTP/Xet events to one non-chatty byte bar.
    pub fn hub_download(&self, message: &str) -> hf_hub::progress::Progress {
        HubDownloadObserver::new().progress_with_bar(Some(self.bytes_bar(0, message)))
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
            format!(
                "{}h {:02}m {:02}s",
                secs / 3600,
                (secs % 3600) / 60,
                secs % 60
            )
        }
    }
}

/// Output-agnostic state from one hf-hub download operation.
///
/// The sequence changes after every accepted event so a foreground owner can
/// coalesce the native Xet poller's ~10 Hz stream without coupling transfer
/// tasks to terminal rendering or a potentially blocking IPC sink.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HubDownloadSnapshot {
    pub completed_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_second: Option<u64>,
    pub complete: bool,
    pub sequence: u64,
}

#[derive(Default)]
struct HubDownloadState {
    completed_bytes: AtomicU64,
    total_bytes: AtomicU64,
    bytes_per_second: AtomicU64,
    rate_known: AtomicBool,
    complete: AtomicBool,
    sequence: AtomicU64,
}

/// Cloneable bridge between hf-hub worker callbacks and the foreground
/// operation that owns user-visible presentation.
#[derive(Clone, Default)]
pub struct HubDownloadObserver {
    state: Arc<HubDownloadState>,
}

impl HubDownloadObserver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn progress(&self) -> hf_hub::progress::Progress {
        self.progress_with_bar(None)
    }

    pub fn snapshot(&self) -> HubDownloadSnapshot {
        HubDownloadSnapshot {
            completed_bytes: self.state.completed_bytes.load(Ordering::Acquire),
            total_bytes: self.state.total_bytes.load(Ordering::Acquire),
            bytes_per_second: self
                .state
                .rate_known
                .load(Ordering::Acquire)
                .then(|| self.state.bytes_per_second.load(Ordering::Acquire)),
            complete: self.state.complete.load(Ordering::Acquire),
            sequence: self.state.sequence.load(Ordering::Acquire),
        }
    }

    fn progress_with_bar(&self, bar: Option<ProgressBar>) -> hf_hub::progress::Progress {
        hf_hub::progress::Progress::new(HubDownloadProgress {
            bar,
            observer: self.clone(),
            files: Mutex::new(BTreeMap::new()),
            aggregate_bytes: AtomicU64::new(0),
        })
    }

    fn set_total(&self, total_bytes: u64) {
        self.state
            .total_bytes
            .fetch_max(total_bytes, Ordering::Release);
    }

    fn set_position(&self, completed_bytes: u64) {
        self.state
            .completed_bytes
            .fetch_max(completed_bytes, Ordering::Release);
    }

    fn set_rate(&self, bytes_per_second: Option<f64>) {
        if let Some(rate) = bytes_per_second.filter(|rate| rate.is_finite() && *rate > 0.0) {
            self.state.bytes_per_second.store(
                rate.round().clamp(1.0, u64::MAX as f64) as u64,
                Ordering::Release,
            );
            self.state.rate_known.store(true, Ordering::Release);
        }
    }

    fn publish_event(&self) {
        self.state.sequence.fetch_add(1, Ordering::AcqRel);
    }

    fn finish(&self) {
        let total = self.state.total_bytes.load(Ordering::Acquire);
        self.set_position(total);
        self.state.complete.store(true, Ordering::Release);
        self.publish_event();
    }
}

struct HubDownloadProgress {
    bar: Option<ProgressBar>,
    observer: HubDownloadObserver,
    files: Mutex<BTreeMap<String, u64>>,
    aggregate_bytes: AtomicU64,
}

impl hf_hub::progress::ProgressHandler for HubDownloadProgress {
    fn on_progress(&self, event: &hf_hub::progress::ProgressEvent) {
        use hf_hub::progress::{DownloadEvent, ProgressEvent};

        let ProgressEvent::Download(event) = event else {
            return;
        };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.observer.set_total(*total_bytes);
                if let Some(bar) = self.bar.as_ref() {
                    bar.set_length(*total_bytes);
                }
                self.observer.publish_event();
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                total_bytes,
                bytes_per_sec,
            } => {
                self.aggregate_bytes
                    .store(*bytes_completed, Ordering::Relaxed);
                self.observer.set_total(*total_bytes);
                self.observer.set_position(*bytes_completed);
                self.observer.set_rate(*bytes_per_sec);
                if let Some(bar) = self.bar.as_ref() {
                    bar.set_length(bar.length().unwrap_or(0).max(*total_bytes));
                    bar.set_position(bar.position().max(*bytes_completed));
                }
                self.observer.publish_event();
            }
            DownloadEvent::Progress { files } => {
                if let Ok(mut positions) = self.files.try_lock() {
                    for file in files {
                        positions.insert(file.filename.clone(), file.bytes_completed);
                    }
                    // Aggregate Xet events and per-file HTTP/completion events
                    // overlap, so summing the channels can double-count. Their
                    // maximum is a monotonic lower bound for mixed snapshots
                    // and reaches the full total when all files complete.
                    let file_bytes = positions.values().copied().sum::<u64>();
                    let aggregate_bytes = self.aggregate_bytes.load(Ordering::Relaxed);
                    let completed_bytes = file_bytes.max(aggregate_bytes);
                    self.observer.set_position(completed_bytes);
                    if let Some(bar) = self.bar.as_ref() {
                        bar.set_position(bar.position().max(completed_bytes));
                    }
                    self.observer.publish_event();
                }
            }
            DownloadEvent::Complete => {
                self.observer.finish();
                if let Some(bar) = self.bar.as_ref() {
                    bar.finish_with_message("Download complete");
                }
            }
        }
    }
}

impl Default for ProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod hub_download_progress_tests {
    use super::*;
    use hf_hub::progress::{
        DownloadEvent, FileProgress, FileStatus, ProgressEvent, ProgressHandler,
    };

    fn file(filename: &str, bytes_completed: u64, total_bytes: u64) -> FileProgress {
        FileProgress {
            filename: filename.to_owned(),
            bytes_completed,
            total_bytes,
            status: FileStatus::InProgress,
        }
    }

    #[test]
    fn mixed_hub_progress_never_regresses_or_double_counts() {
        let handler = HubDownloadProgress {
            bar: Some(ProgressBar::hidden()),
            observer: HubDownloadObserver::new(),
            files: Mutex::new(BTreeMap::new()),
            aggregate_bytes: AtomicU64::new(0),
        };
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 2,
            total_bytes: 300,
        }));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![file("ordinary.bin", 40, 100)],
        }));
        assert_eq!(handler.bar.as_ref().unwrap().position(), 40);
        assert_eq!(handler.observer.snapshot().completed_bytes, 40);
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: 120,
            total_bytes: 200,
            bytes_per_sec: Some(1.0),
        }));
        assert_eq!(handler.bar.as_ref().unwrap().position(), 120);
        assert_eq!(handler.observer.snapshot().bytes_per_second, Some(1));
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![file("ordinary.bin", 100, 100)],
        }));
        assert_eq!(handler.bar.as_ref().unwrap().position(), 120);
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::AggregateProgress {
            bytes_completed: 80,
            total_bytes: 200,
            bytes_per_sec: Some(1.0),
        }));
        assert_eq!(handler.bar.as_ref().unwrap().position(), 120);
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![file("xet.bin", 200, 200)],
        }));
        assert_eq!(handler.bar.as_ref().unwrap().position(), 300);
        handler.on_progress(&ProgressEvent::Download(DownloadEvent::Complete));
        assert_eq!(
            handler.observer.snapshot(),
            HubDownloadSnapshot {
                completed_bytes: 300,
                total_bytes: 300,
                bytes_per_second: Some(1),
                complete: true,
                sequence: 7,
            }
        );
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
    println!("  Input size:   {}", format_bytes(input_size_bytes));
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
