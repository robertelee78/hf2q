//! Diagnostic timeline of DFlash2 verify cycles (`HF2Q_DFLASH2_CYCLE_TIMELINE`).
//!
//! When the variable names a file, every cycle appends one JSON line: the
//! host marks of the scheduler step and, for each command buffer committed
//! since the previous line, its host encode start, commit and wait times and
//! the driver-scheduling and GPU-execution intervals Metal reports for it
//! (mlx-native `command_timeline`). Every time is seconds on one clock (mach
//! absolute time), so the host and the GPU can be laid on one axis.
//! `scripts/dflash2_cycle_timeline.py` reduces the file.
//!
//! Recording changes no commit into a wait and adds nothing to a command.
//! Command buffers are held until the cycle's line is written, after the
//! cycle's own timing has been taken.
//!
//! Limits of a development diagnostic:
//!
//! - The file must not exist: a capture is never truncated or appended to.
//!   The path must be absolute. Either refusal is logged once and leaves the
//!   timeline off.
//! - The recorder and the file are process-wide: one engine per process.
//! - A cycle that leaves a command executing on purpose (the next step's
//!   prefetched draft) does not write its line at its end, because taking
//!   the commands would wait for that one. The line is written inside the
//!   next cycle, once that cycle's output head has completed
//!   ([`settle_deferred`]), with the commands committed up to the deferred
//!   cycle's end; later ones are carried into the next line. The last line
//!   of a capture can be missing if no cycle follows a deferred one.
//! - A cycle that returns early still writes its line (`finished: false`)
//!   and releases the command buffers held for it. Command buffers
//!   committed outside DFlash2 cycles (prefill, serial decode) are held
//!   until the next cycle's line; mlx-native discards them beyond 4,096
//!   commands and counts what it discarded, and every line carries those
//!   counters (`dropped_commands`, `dropped_marks`; nonzero means a gap).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{LazyLock, Mutex};

use mlx_native::command_timeline::{self, CommandTimeline};

/// Output file of the diagnostic cycle timeline; unset or empty is off.
pub(super) const DFLASH2_CYCLE_TIMELINE_ENV: &str = "HF2Q_DFLASH2_CYCLE_TIMELINE";

/// Open `path` for a new capture: absolute, and not an existing file.
fn open_capture(path: &Path) -> Result<File, String> {
    if !path.is_absolute() {
        return Err("the path is not absolute".to_owned());
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())
}

static SINK: LazyLock<Option<Mutex<BufWriter<File>>>> = LazyLock::new(|| {
    let path = std::env::var(DFLASH2_CYCLE_TIMELINE_ENV).ok()?;
    if path.is_empty() {
        return None;
    }
    match open_capture(Path::new(&path)) {
        Ok(file) => {
            command_timeline::set_enabled(true);
            tracing::info!(path, "Qwen DFlash2 cycle timeline enabled");
            Some(Mutex::new(BufWriter::new(file)))
        }
        Err(reason) => {
            tracing::warn!(path, reason, "Qwen DFlash2 cycle timeline disabled");
            None
        }
    }
});

/// A cycle's line without its commands, and the time of its `end` mark.
struct DeferredLine {
    line: serde_json::Map<String, serde_json::Value>,
    end_s: f64,
}

static DEFERRED: Mutex<Option<DeferredLine>> = Mutex::new(None);
/// Commands taken from the recorder that belong to a later line.
static CARRIED: Mutex<Vec<CommandTimeline>> = Mutex::new(Vec::new());

fn write_line(
    sink: &Mutex<BufWriter<File>>,
    mut line: serde_json::Map<String, serde_json::Value>,
    commands: &[CommandTimeline],
) {
    line.insert(
        "commands".to_owned(),
        commands.iter().map(command_json).collect(),
    );
    let line = serde_json::Value::Object(line);
    if let Ok(mut sink) = sink.lock() {
        if writeln!(sink, "{line}")
            .and_then(|()| sink.flush())
            .is_err()
        {
            tracing::warn!("Qwen DFlash2 cycle timeline write failed");
        }
    }
}

/// Every command recorded so far, carried ones first. Waits for a command
/// that is still executing.
fn take_commands() -> Vec<CommandTimeline> {
    let mut commands = CARRIED
        .lock()
        .map(|mut carried| std::mem::take(&mut *carried))
        .unwrap_or_default();
    commands.extend(command_timeline::drain());
    commands
}

/// Write the line of a cycle that deferred it, if there is one. Call it
/// where every command committed so far has completed: the commands
/// committed after the deferred cycle's end stay for the next line.
pub(super) fn settle_deferred() {
    let Some(sink) = SINK.as_ref() else {
        return;
    };
    let Some(deferred) = DEFERRED.lock().ok().and_then(|mut slot| slot.take()) else {
        return;
    };
    let (own, later): (Vec<_>, Vec<_>) = take_commands()
        .into_iter()
        .partition(|command| command.commit_begin_s <= deferred.end_s);
    if let Ok(mut carried) = CARRIED.lock() {
        *carried = later;
    }
    write_line(sink, deferred.line, &own);
}

/// One scheduler step being recorded. Host marks are placed with
/// `mlx_native::command_timeline::mark`, here and in the forward it calls.
/// Dropping it writes the cycle's line, so a step that returns early does
/// not leave its command buffers held.
pub(super) struct Cycle {
    summary: Option<CycleSummary>,
    defer: bool,
}

/// Summary fields of a finished cycle.
pub(super) struct CycleSummary {
    pub(super) active_lanes: usize,
    pub(super) selected_rows: usize,
    pub(super) accepted_rows: usize,
    pub(super) gpu_acceptance: bool,
}

impl Cycle {
    /// Start a cycle's record, or `None` when the timeline is off.
    pub(super) fn begin() -> Option<Self> {
        SINK.as_ref()?;
        // Marks left by work outside a cycle (prefill forwards) belong to
        // no line.
        drop(command_timeline::drain_marks());
        command_timeline::mark("begin");
        Some(Self {
            summary: None,
            defer: false,
        })
    }

    /// The cycle leaves a command executing on purpose: write its line
    /// later ([`settle_deferred`]) instead of waiting for that command.
    pub(super) fn defer_line(&mut self) {
        self.defer = true;
    }

    /// Record that the cycle ran to its end; its line is written when the
    /// value drops.
    pub(super) fn finish(mut self, summary: CycleSummary) {
        self.summary = Some(summary);
    }
}

impl Drop for Cycle {
    /// Write the cycle's line. Commands committed before the cycle began
    /// (prefill, serial decode) are included; the reducer attributes a
    /// command to the cycle by its commit time.
    fn drop(&mut self) {
        command_timeline::mark("end");
        let Some(sink) = SINK.as_ref() else {
            return;
        };
        let marks = command_timeline::drain_marks();
        // An earlier deferred line that was never settled comes first.
        settle_deferred();
        let end_s = marks
            .iter()
            .rev()
            .find(|(name, _)| *name == "end")
            .map_or(f64::MAX, |(_, seconds)| *seconds);
        let summary = self.summary.as_ref();
        let serde_json::Value::Object(line) = serde_json::json!({
            "finished": summary.is_some(),
            "active_lanes": summary.map(|summary| summary.active_lanes),
            "selected_rows": summary.map(|summary| summary.selected_rows),
            "accepted_rows": summary.map(|summary| summary.accepted_rows),
            "gpu_acceptance": summary.map(|summary| summary.gpu_acceptance),
            "dropped_commands": command_timeline::dropped(),
            "dropped_marks": command_timeline::dropped_marks(),
            "marks": marks,
        }) else {
            return;
        };
        if self.defer && summary.is_some() {
            if let Ok(mut slot) = DEFERRED.lock() {
                *slot = Some(DeferredLine { line, end_s });
                return;
            }
        }
        write_line(sink, line, &take_commands());
    }
}

fn command_json(command: &CommandTimeline) -> serde_json::Value {
    serde_json::json!({
        "label": command.label,
        "encode_begin": command.encode_begin_s,
        "commit_begin": command.commit_begin_s,
        "commit_end": command.commit_end_s,
        "wait_begin": command.wait_begin_s,
        "wait_end": command.wait_end_s,
        "kernel_start": command.kernel_start_s,
        "kernel_end": command.kernel_end_s,
        "gpu_start": command.gpu_start_s,
        "gpu_end": command.gpu_end_s,
        "dispatches_at_commit": command.dispatches_at_commit,
        "barriers_at_commit": command.barriers_at_commit,
        "completed": command.completed,
        "waited_in_drain": command.waited_in_drain,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_capture_is_never_truncated_and_needs_an_absolute_path() {
        let directory =
            std::env::temp_dir().join(format!("hf2q-dflash2-timeline-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("capture.jsonl");
        let _ = std::fs::remove_file(&path);

        assert!(open_capture(Path::new("relative/capture.jsonl")).is_err());
        drop(open_capture(&path).expect("a new file opens"));
        std::fs::write(&path, b"earlier capture\n").unwrap();
        assert!(
            open_capture(&path).is_err(),
            "an existing capture was opened"
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"earlier capture\n");
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
