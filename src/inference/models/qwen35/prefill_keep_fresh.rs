//! Keep the weights that only the prompt-prefill route reads from ageing
//! while a worker decodes (`HF2Q_QWEN_PREFILL_KEEP_FRESH=1`).
//!
//! On Apple GPUs a buffer that no command buffer has used for about a second
//! costs time on its next use, roughly in proportion to its size (measured:
//! the Qwen3.6 output head, 0.39 GiB, 1.2 ms fresh and 4.4 ms after a
//! 1.5 s pause; a 157-row Qwen3.6 prefill forward 71 ms fresh and 287 ms
//! after 1.2 s). Any use by a command buffer keeps a buffer fresh, however
//! little of it the command reads.
//!
//! The decode routes of a served Qwen model run on prepared copies of some
//! weights (the MoE experts' tile planes, the shared output head planes),
//! while the prompt prefill reads the GGUF blocks of the same weights. During
//! a stretch of decode cycles those blocks are used by nothing, and the next
//! prompt prefill pays for all of them. With the switch on, the worker issues
//! one tiny command buffer at most every [`INTERVAL`] while it decodes: one
//! four-byte read of each such buffer into private scratch. It is committed
//! without a wait, changes no model state and no output bit.

use std::cell::RefCell;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use mlx_native::ops::copy::{dispatch_strided_copy_f32, StridedCopyParams};
use mlx_native::{DType, MlxBuffer};

use super::joint_step_options::switch_from_process;
use super::model::Qwen35Model;

/// Switch, default off: `1`/`on`/`true` keeps the prefill-only weights of a
/// decoding worker fresh.
pub(crate) const QWEN_PREFILL_KEEP_FRESH_ENV: &str = "HF2Q_QWEN_PREFILL_KEEP_FRESH";

/// Longest a prefill-only weight goes without a use while the worker
/// decodes: well inside the second after which an unused buffer ages.
pub(crate) const INTERVAL: Duration = Duration::from_millis(250);

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    switch_from_process(QWEN_PREFILL_KEEP_FRESH_ENV).unwrap_or_else(|error| {
        tracing::warn!(error, fallback = false, "invalid Qwen prefill keep-fresh switch");
        false
    })
});

/// Whether the operator switched the keep-alive on.
pub(crate) fn enabled() -> bool {
    *ENABLED
}

/// Whether a use is due at `now` after the previous one at `last`.
fn due(last: Option<Instant>, now: Instant) -> bool {
    last.is_none_or(|last| now.saturating_duration_since(last) >= INTERVAL)
}

struct State {
    /// The model the state below belongs to.
    model: *const (),
    /// One F32 element per listed buffer: where its four bytes are copied.
    /// Kept across uses, so a command committed without a wait never
    /// outlives what it writes.
    scratch: Option<MlxBuffer>,
    last: Option<Instant>,
    /// A failed use ends the keep-alive for this model: it never fails a
    /// request.
    failed: bool,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

impl Qwen35Model {
    /// Use every weight only the prompt-prefill route reads, if the last
    /// use is at least [`INTERVAL`] old. Returns whether a command buffer
    /// was issued. Call on the worker thread, between GPU transactions.
    pub(crate) fn keep_prefill_weights_fresh(&self) -> Result<bool> {
        self.keep_prefill_weights_fresh_at(Instant::now())
    }

    /// [`Self::keep_prefill_weights_fresh`] at an explicit time.
    pub(crate) fn keep_prefill_weights_fresh_at(&self, now: Instant) -> Result<bool> {
        let model = self as *const Self as *const ();
        STATE.with(|cell| -> Result<bool> {
            let mut slot = cell.borrow_mut();
            if slot.as_ref().is_none_or(|state| state.model != model) {
                *slot = Some(State {
                    model,
                    scratch: None,
                    last: None,
                    failed: false,
                });
            }
            let state = slot.as_mut().expect("state was just installed");
            if state.failed || !due(state.last, now) {
                return Ok(false);
            }
            state.last = Some(now);
            let used = self.use_prefill_only_weights(&mut state.scratch);
            if used.is_err() {
                state.failed = true;
            }
            used.context("keep the prefill-only weights fresh")
        })
    }

    /// One command buffer, committed without a wait, that reads four bytes
    /// of every prefill-only weight into `scratch`. `false`: the model has
    /// no such weight.
    fn use_prefill_only_weights(&self, scratch: &mut Option<MlxBuffer>) -> Result<bool> {
        let buffers = self.prefill_only_weights()?;
        if buffers.is_empty() {
            return Ok(false);
        }
        self.with_gpu_cache_mut(|device, registry| {
            if scratch
                .as_ref()
                .is_none_or(|scratch| scratch.element_count() < buffers.len())
            {
                *scratch = Some(
                    device
                        .alloc_buffer(
                            buffers.len() * DType::F32.size_of(),
                            DType::F32,
                            vec![buffers.len()],
                        )
                        .map_err(|error| anyhow!("alloc prefill keep-fresh scratch: {error}"))?,
                );
            }
            let scratch = scratch.as_ref().expect("scratch was just allocated");
            let mut enc = device
                .command_encoder()
                .map(crate::inference::native_observation::encoder)
                .map_err(|error| anyhow!("prefill keep-fresh encoder: {error}"))?;
            for (index, buffer) in buffers.iter().enumerate() {
                let element = scratch.slice_view((index * DType::F32.size_of()) as u64, 1);
                dispatch_strided_copy_f32(
                    &mut enc,
                    registry,
                    device.metal_device(),
                    buffer,
                    &element,
                    &StridedCopyParams {
                        rows: 1,
                        cols: 1,
                        stride_row: 1,
                        stride_col: 1,
                    },
                )
                .map_err(|error| anyhow!("prefill keep-fresh read {index}: {error}"))?;
            }
            enc.commit_labeled("qwen35.prefill.keep_fresh");
            Ok(true)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::models::qwen35::joint_step_options::parse_switch;

    #[test]
    fn switch_is_off_unless_set_on() {
        let parse = |raw| parse_switch(QWEN_PREFILL_KEEP_FRESH_ENV, raw);
        assert_eq!(parse(None), Ok(false));
        assert_eq!(parse(Some("1")), Ok(true));
        assert_eq!(parse(Some("off")), Ok(false));
        assert!(parse(Some("250")).is_err());
    }

    #[test]
    fn a_use_is_due_at_first_and_then_once_per_interval() {
        let start = Instant::now();
        assert!(due(None, start));
        assert!(!due(Some(start), start));
        assert!(!due(Some(start), start + INTERVAL - Duration::from_millis(1)));
        assert!(due(Some(start), start + INTERVAL));
        // A clock that reads earlier than the last use never panics.
        assert!(!due(Some(start + INTERVAL), start));
    }
}
