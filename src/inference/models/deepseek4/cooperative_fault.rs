//! Test-only fault injection for the DeepSeek-V4 cooperative prefill
//! transaction (ADR-063 S1).
//!
//! The hook fires after one layer's attention and FFN command buffers have
//! completed for every cohort lane, so the failing transaction has already
//! written window rows, compressed/indexer rows and recurrent compressor
//! state into every member cache. Non-test builds compile no hook.

use std::cell::Cell;

use anyhow::Result;

thread_local! {
    static FAIL_AFTER_LAYER: Cell<Option<usize>> = const { Cell::new(None) };
}

/// One armed fault on the current thread. Dropping it disarms an unfired
/// fault so a failed assertion cannot leak it into a later transaction.
pub(super) struct ArmedCooperativeFault(());

impl ArmedCooperativeFault {
    pub(super) fn fired(&self) -> bool {
        FAIL_AFTER_LAYER.with(|armed| armed.get().is_none())
    }
}

impl Drop for ArmedCooperativeFault {
    fn drop(&mut self) {
        FAIL_AFTER_LAYER.with(|armed| armed.set(None));
    }
}

/// Fail the next cooperative prefill on this thread after `layer` completes.
pub(super) fn arm_after_layer(layer: usize) -> ArmedCooperativeFault {
    FAIL_AFTER_LAYER.with(|armed| {
        assert!(
            armed.get().is_none(),
            "a cooperative prefill fault is already armed"
        );
        armed.set(Some(layer));
    });
    ArmedCooperativeFault(())
}

pub(super) fn check_after_layer(layer: usize) -> Result<()> {
    let fire = FAIL_AFTER_LAYER.with(|armed| {
        let fire = armed.get() == Some(layer);
        if fire {
            armed.set(None);
        }
        fire
    });
    anyhow::ensure!(
        !fire,
        "injected DeepSeek-V4 cooperative prefill fault after layer {layer}"
    );
    Ok(())
}
