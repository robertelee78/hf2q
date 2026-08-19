#[cfg(test)]
use std::sync::atomic::Ordering;

use super::*;

#[cfg(not(test))]
pub(super) fn prepared_barrier() -> Result<(), PreparedVersionError> {
    Ok(())
}

#[cfg(test)]
pub(super) fn prepared_barrier() -> Result<(), PreparedVersionError> {
    OBSERVED_PREPARED_BARRIERS.fetch_add(1, Ordering::SeqCst);
    let abort = ABORT_AFTER_PREPARED_BARRIER.load(Ordering::SeqCst);
    if abort != 0 && ABORT_AFTER_PREPARED_BARRIER.fetch_sub(1, Ordering::SeqCst) == 1 {
        std::process::abort();
    }
    let fail = FAIL_AFTER_PREPARED_BARRIER.load(Ordering::SeqCst);
    if fail != 0 && FAIL_AFTER_PREPARED_BARRIER.fetch_sub(1, Ordering::SeqCst) == 1 {
        return Err(PreparedVersionError::Integrity);
    }
    Ok(())
}

#[cfg(test)]
pub(in crate::distribution) fn abort_after_prepared_barrier(barrier: usize) {
    assert_ne!(barrier, 0, "prepared barrier is one-based");
    ABORT_AFTER_PREPARED_BARRIER.store(barrier, Ordering::SeqCst);
}

#[cfg(test)]
pub(in crate::distribution) fn fail_after_prepared_barrier(barrier: usize) {
    assert_ne!(barrier, 0, "prepared barrier is one-based");
    FAIL_AFTER_PREPARED_BARRIER.store(barrier, Ordering::SeqCst);
}

#[cfg(test)]
pub(in crate::distribution) fn reset_observed_prepared_barriers() {
    OBSERVED_PREPARED_BARRIERS.store(0, Ordering::SeqCst);
    FAIL_AFTER_PREPARED_BARRIER.store(0, Ordering::SeqCst);
    ABORT_AFTER_PREPARED_BARRIER.store(0, Ordering::SeqCst);
}

#[cfg(test)]
pub(in crate::distribution) fn observed_prepared_barriers() -> usize {
    OBSERVED_PREPARED_BARRIERS.load(Ordering::SeqCst)
}

#[cfg(test)]
pub(in crate::distribution) fn set_prepared_precommit_hook(hook: impl FnOnce() + 'static) {
    PREPARED_PRECOMMIT_HOOK.with(|slot| {
        assert!(slot.borrow_mut().replace(Box::new(hook)).is_none());
    });
}

#[cfg(test)]
pub(super) fn prepared_precommit_hook() {
    PREPARED_PRECOMMIT_HOOK.with(|slot| {
        if let Some(hook) = slot.borrow_mut().take() {
            hook();
        }
    });
}

#[cfg(not(test))]
pub(super) fn prepared_precommit_hook() {}
