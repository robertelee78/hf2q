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

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{LazyLock, Mutex};

use mlx_native::command_timeline::{self, CommandTimeline};

/// Output file of the diagnostic cycle timeline; unset or empty is off.
pub(super) const DFLASH2_CYCLE_TIMELINE_ENV: &str = "HF2Q_DFLASH2_CYCLE_TIMELINE";

static SINK: LazyLock<Option<Mutex<BufWriter<File>>>> = LazyLock::new(|| {
    let path = std::env::var(DFLASH2_CYCLE_TIMELINE_ENV).ok()?;
    if path.is_empty() {
        return None;
    }
    match File::create(&path) {
        Ok(file) => {
            command_timeline::set_enabled(true);
            tracing::info!(path, "Qwen DFlash2 cycle timeline enabled");
            Some(Mutex::new(BufWriter::new(file)))
        }
        Err(error) => {
            tracing::warn!(path, %error, "Qwen DFlash2 cycle timeline disabled: cannot create file");
            None
        }
    }
});

/// One scheduler step being recorded. Host marks are placed with
/// `mlx_native::command_timeline::mark`, here and in the forward it calls.
pub(super) struct Cycle(());

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
        Some(Self(()))
    }

    /// Write the cycle's line. Commands committed before the cycle began
    /// (prefill, a previous cycle that returned early) are included; the
    /// reducer attributes a command to the cycle by its commit time.
    pub(super) fn finish(self, summary: CycleSummary) {
        command_timeline::mark("end");
        let Some(sink) = SINK.as_ref() else {
            return;
        };
        let marks = command_timeline::drain_marks();
        let commands = command_timeline::drain();
        let line = serde_json::json!({
            "active_lanes": summary.active_lanes,
            "selected_rows": summary.selected_rows,
            "accepted_rows": summary.accepted_rows,
            "gpu_acceptance": summary.gpu_acceptance,
            "marks": marks,
            "commands": commands.iter().map(command_json).collect::<Vec<_>>(),
        });
        if let Ok(mut sink) = sink.lock() {
            if writeln!(sink, "{line}")
                .and_then(|()| sink.flush())
                .is_err()
            {
                tracing::warn!("Qwen DFlash2 cycle timeline write failed");
            }
        }
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
