//! Process-wide observations for ordinary Qwen physical decode batching.
//!
//! Scheduler concurrency and physical GPU batching are deliberately separate
//! measurements. A scheduler step may contain N live handles while the model
//! still executes N scalar forwards. These counters make that distinction
//! observable without inferring it from aggregate throughput. Speculative
//! target verification is deliberately excluded so its within-request width
//! cannot be mistaken for cross-request physical batching.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Qwen35DecodeObservationSnapshot {
    pub scheduler_steps: u64,
    pub scheduler_handles: u64,
    pub scheduler_max_width: u64,
    pub target_forwards: u64,
    pub target_body_rows: u64,
    pub target_body_max_width: u64,
    pub target_head_rows: u64,
    pub target_head_max_width: u64,
    pub command_buffers_created: u64,
    pub command_buffer_submissions: u64,
}

struct Qwen35DecodeObservations {
    scheduler_steps: AtomicU64,
    scheduler_handles: AtomicU64,
    scheduler_max_width: AtomicU64,
    target_forwards: AtomicU64,
    target_body_rows: AtomicU64,
    target_body_max_width: AtomicU64,
    target_head_rows: AtomicU64,
    target_head_max_width: AtomicU64,
    command_buffers_created: AtomicU64,
    command_buffer_submissions: AtomicU64,
}

static OBSERVATIONS: Qwen35DecodeObservations = Qwen35DecodeObservations {
    scheduler_steps: AtomicU64::new(0),
    scheduler_handles: AtomicU64::new(0),
    scheduler_max_width: AtomicU64::new(0),
    target_forwards: AtomicU64::new(0),
    target_body_rows: AtomicU64::new(0),
    target_body_max_width: AtomicU64::new(0),
    target_head_rows: AtomicU64::new(0),
    target_head_max_width: AtomicU64::new(0),
    command_buffers_created: AtomicU64::new(0),
    command_buffer_submissions: AtomicU64::new(0),
};

#[inline]
pub(crate) fn observe_scheduler_step(handle_count: usize) {
    let width = handle_count as u64;
    OBSERVATIONS.scheduler_steps.fetch_add(1, Ordering::Relaxed);
    OBSERVATIONS
        .scheduler_handles
        .fetch_add(width, Ordering::Relaxed);
    OBSERVATIONS
        .scheduler_max_width
        .fetch_max(width, Ordering::Relaxed);
}

/// Record one target invocation after its terminal GPU wait succeeds.
///
/// `body_width` and `head_width` are the actual row counts encoded into the
/// target body and output projection. They are not the scheduler handle count.
/// `command_buffers_created` comes from mlx-native's encoder-construction
/// counter. `command_buffer_submissions` comes from the native commit primitive;
/// neither value is inferred from the other.
#[inline]
pub(crate) fn observe_target_forward(
    body_width: usize,
    head_width: usize,
    command_buffers_created: u64,
    command_buffer_submissions: u64,
) {
    let body_width = body_width as u64;
    let head_width = head_width as u64;
    OBSERVATIONS.target_forwards.fetch_add(1, Ordering::Relaxed);
    OBSERVATIONS
        .target_body_rows
        .fetch_add(body_width, Ordering::Relaxed);
    OBSERVATIONS
        .target_body_max_width
        .fetch_max(body_width, Ordering::Relaxed);
    OBSERVATIONS
        .target_head_rows
        .fetch_add(head_width, Ordering::Relaxed);
    OBSERVATIONS
        .target_head_max_width
        .fetch_max(head_width, Ordering::Relaxed);
    OBSERVATIONS
        .command_buffers_created
        .fetch_add(command_buffers_created, Ordering::Relaxed);
    OBSERVATIONS
        .command_buffer_submissions
        .fetch_add(command_buffer_submissions, Ordering::Relaxed);
}

pub(crate) fn snapshot() -> Qwen35DecodeObservationSnapshot {
    Qwen35DecodeObservationSnapshot {
        scheduler_steps: OBSERVATIONS.scheduler_steps.load(Ordering::Relaxed),
        scheduler_handles: OBSERVATIONS.scheduler_handles.load(Ordering::Relaxed),
        scheduler_max_width: OBSERVATIONS.scheduler_max_width.load(Ordering::Relaxed),
        target_forwards: OBSERVATIONS.target_forwards.load(Ordering::Relaxed),
        target_body_rows: OBSERVATIONS.target_body_rows.load(Ordering::Relaxed),
        target_body_max_width: OBSERVATIONS.target_body_max_width.load(Ordering::Relaxed),
        target_head_rows: OBSERVATIONS.target_head_rows.load(Ordering::Relaxed),
        target_head_max_width: OBSERVATIONS.target_head_max_width.load(Ordering::Relaxed),
        command_buffers_created: OBSERVATIONS.command_buffers_created.load(Ordering::Relaxed),
        command_buffer_submissions: OBSERVATIONS
            .command_buffer_submissions
            .load(Ordering::Relaxed),
    }
}

#[cfg(test)]
pub(crate) fn reset() {
    OBSERVATIONS.scheduler_steps.store(0, Ordering::Relaxed);
    OBSERVATIONS.scheduler_handles.store(0, Ordering::Relaxed);
    OBSERVATIONS.scheduler_max_width.store(0, Ordering::Relaxed);
    OBSERVATIONS.target_forwards.store(0, Ordering::Relaxed);
    OBSERVATIONS.target_body_rows.store(0, Ordering::Relaxed);
    OBSERVATIONS
        .target_body_max_width
        .store(0, Ordering::Relaxed);
    OBSERVATIONS.target_head_rows.store(0, Ordering::Relaxed);
    OBSERVATIONS
        .target_head_max_width
        .store(0, Ordering::Relaxed);
    OBSERVATIONS
        .command_buffers_created
        .store(0, Ordering::Relaxed);
    OBSERVATIONS
        .command_buffer_submissions
        .store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_and_physical_widths_remain_distinct() {
        reset();
        observe_scheduler_step(4);
        for _ in 0..4 {
            observe_target_forward(1, 1, 6, 2);
        }

        let observed = snapshot();
        assert_eq!(observed.scheduler_steps, 1);
        assert_eq!(observed.scheduler_handles, 4);
        assert_eq!(observed.scheduler_max_width, 4);
        assert_eq!(observed.target_forwards, 4);
        assert_eq!(observed.target_body_rows, 4);
        assert_eq!(observed.target_body_max_width, 1);
        assert_eq!(observed.target_head_max_width, 1);
        assert_eq!(observed.command_buffers_created, 24);
        assert_eq!(observed.command_buffer_submissions, 8);

        reset();
        observe_scheduler_step(4);
        observe_target_forward(4, 4, 6, 2);

        let observed = snapshot();
        assert_eq!(observed.scheduler_handles, 4);
        assert_eq!(observed.target_forwards, 1);
        assert_eq!(observed.target_body_rows, 4);
        assert_eq!(observed.target_body_max_width, 4);
        assert_eq!(observed.target_head_rows, 4);
        assert_eq!(observed.target_head_max_width, 4);
        assert_eq!(observed.command_buffers_created, 6);
        assert_eq!(observed.command_buffer_submissions, 2);
    }
}
