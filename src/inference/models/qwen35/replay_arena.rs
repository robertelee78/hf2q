//! Storage of a verify step's DeltaNet replay
//! (`HF2Q_QWEN_VERIFY_REPLAY_REUSE=1` retains it between steps).
//!
//! A verify step retains, per DeltaNet layer and multi-row window, the inputs
//! of the replay commit ([`super::joint_decode_core::DeltaNetReplay`]). At
//! four lanes of eight rows that is 188 MB on Qwen3.8-27B (48 layers) and
//! 91 MB on Qwen3.6-35B-A3B (30 layers). The stock route allocates it for
//! every step: a fresh shared Metal buffer, a host `memset` of the whole
//! length (`MlxDevice::alloc_buffer` zero-fills), a residency-set commit at
//! the step's first command and a release at its end. All of that runs on the
//! serving worker between the draft command's commit and the target's first
//! command, so the GPU idles once the draft finishes.
//!
//! With the switch on, the worker thread keeps the largest arena a step has
//! used and hands it to the next step as is. The bytes are not cleared:
//! every element the verify forward or the replay commit reads from the
//! arena is written earlier in the same step by a dispatch ordered before
//! the reader on the same queue (the conv rows, scaled Q, normed K, V and the
//! packed `g`/`beta` rows by the forward; the K-1 staged conv-state rows and
//! the output scratch by the replay itself). A step's results are therefore
//! the stock route's bits (Class A); `replay_arena_tests` holds that with
//! the arena poisoned before every step. A change to any dispatch that
//! writes this storage must keep that property, or clear what it no longer
//! writes.
//!
//! An arena is owned by exactly one step at a time: a step takes it out of
//! the thread's cell and gives it back when its replay storage drops. A
//! second live step allocates its own.
//!
//! The retained arena is not part of any memory admission or budget (the
//! per-step allocation it replaces was not either): it grows to the largest
//! step the worker has run, never shrinks, and is released when the worker
//! thread ends. A retained arena too small for a step is released before
//! the larger one is allocated, so a worker holds one arena at most while
//! no step is between forward and commit. The cell is per thread, not per
//! device; a worker owns one device for its lifetime.

use std::cell::RefCell;

use anyhow::{anyhow, Result};
use mlx_native::{DType, MlxBuffer, MlxDevice};

use crate::inference::spec_decode::dflash2::route_profile;

/// Opt-in switch of the retained verify replay storage; exact value `1`.
pub(crate) const QWEN_VERIFY_REPLAY_REUSE_ENV: &str = "HF2Q_QWEN_VERIFY_REPLAY_REUSE";

thread_local! {
    static RETAINED: RefCell<Option<MlxBuffer>> = const { RefCell::new(None) };
    #[cfg(test)]
    static TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    #[cfg(test)]
    static FRESH_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Whether this model's resolved routes retain the replay storage.
pub(crate) fn enabled() -> bool {
    const SWITCH: route_profile::Switch = route_profile::switch(QWEN_VERIFY_REPLAY_REUSE_ENV);
    #[cfg(test)]
    if let Some(forced) = TEST_OVERRIDE.get() {
        return forced;
    }
    route_profile::is_one(SWITCH)
}

/// One step's replay storage: `F32` elements, at least as many as asked for.
pub(crate) struct ReplayArena {
    buffer: MlxBuffer,
    retain: bool,
}

impl ReplayArena {
    /// Storage for `elements` F32 elements: the thread's retained arena when
    /// the switch is on and it is large enough, otherwise a fresh zero-filled
    /// allocation.
    pub(crate) fn acquire(device: &MlxDevice, elements: usize) -> Result<Self> {
        let retain = enabled();
        if retain {
            let kept = RETAINED.with(|cell| cell.borrow_mut().take());
            if let Some(buffer) = kept.filter(|buffer| buffer.element_count() >= elements) {
                return Ok(Self { buffer, retain });
            }
        }
        #[cfg(test)]
        FRESH_ALLOCATIONS.set(FRESH_ALLOCATIONS.get() + 1);
        let buffer = device
            .alloc_buffer(elements * 4, DType::F32, vec![elements])
            .map_err(|e| anyhow!("allocate Qwen joint verify replay storage: {e}"))?;
        Ok(Self { buffer, retain })
    }

    pub(crate) fn buffer(&self) -> &MlxBuffer {
        &self.buffer
    }
}

impl Drop for ReplayArena {
    fn drop(&mut self) {
        if !self.retain {
            return;
        }
        // `try_with`: a step dropped during thread teardown releases its
        // arena instead of touching a destroyed cell.
        let _ = RETAINED.try_with(|cell| {
            let mut kept = cell.borrow_mut();
            if kept
                .as_ref()
                .is_none_or(|kept| kept.element_count() < self.buffer.element_count())
            {
                *kept = Some(self.buffer.clone());
            }
        });
    }
}

/// Force the switch on this thread; `None` restores the environment's value.
#[cfg(test)]
pub(crate) fn set_test_override(forced: Option<bool>) {
    TEST_OVERRIDE.set(forced);
}

/// Fill this thread's retained arena, if any, with `value`. Returns whether
/// an arena was retained.
#[cfg(test)]
pub(crate) fn fill_retained(value: f32) -> bool {
    RETAINED.with(|cell| match cell.borrow_mut().as_mut() {
        Some(buffer) => {
            buffer
                .as_mut_slice::<f32>()
                .expect("retained replay arena is host-writable")
                .fill(value);
            true
        }
        None => false,
    })
}

/// Drop this thread's retained arena.
#[cfg(test)]
pub(crate) fn release_retained() {
    RETAINED.with(|cell| cell.borrow_mut().take());
}

/// Whether this thread holds a retained arena.
#[cfg(test)]
pub(crate) fn has_retained() -> bool {
    RETAINED.with(|cell| cell.borrow().is_some())
}

/// Fresh replay allocations made on this thread so far.
#[cfg(test)]
pub(crate) fn fresh_allocations() -> usize {
    FRESH_ALLOCATIONS.get()
}
