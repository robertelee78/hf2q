//! Test-only override of the prepared-route switches on the calling thread.
//!
//! The switches (`HF2Q_REPACKED_GGUF_M32`, `_M32_PROJ`, `_M32_HEAD`,
//! `_ANY_WIDTH`, `_SHARED_HEAD`) are read once per process, so a test cannot
//! set them through the environment without changing every other test of
//! the binary. A test that needs a route holds an [`Override`] instead:
//! while it lives, the switch readers of this thread return its values and
//! never consult the environment. Joint steps are encoded on the calling
//! thread, so the override covers a whole step.

use std::cell::Cell;

/// One value per prepared-route switch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct PreparedSwitches {
    /// `HF2Q_REPACKED_GGUF_M32`: the dense FFN, and the gate of the others.
    pub ffn: bool,
    /// `HF2Q_REPACKED_GGUF_M32_PROJ`.
    pub projections: bool,
    /// `HF2Q_REPACKED_GGUF_M32_HEAD`.
    pub head: bool,
    /// `HF2Q_REPACKED_GGUF_ANY_WIDTH`.
    pub any_width: bool,
    /// `HF2Q_REPACKED_GGUF_SHARED_HEAD`.
    pub shared_head: bool,
}

impl PreparedSwitches {
    /// Every switch off: the stock kernels.
    pub(crate) const OFF: Self = Self {
        ffn: false,
        projections: false,
        head: false,
        any_width: false,
        shared_head: false,
    };
    /// The prepared FFN, projections and head at 32 rows only.
    pub(crate) const WIDTH_32: Self = Self {
        ffn: true,
        projections: true,
        head: true,
        any_width: false,
        shared_head: false,
    };
    /// Every switch on.
    pub(crate) const ALL: Self = Self {
        ffn: true,
        projections: true,
        head: true,
        any_width: true,
        shared_head: true,
    };
}

thread_local! {
    static OVERRIDE: Cell<Option<PreparedSwitches>> = const { Cell::new(None) };
}

/// The override of this thread, if a test holds one.
pub(crate) fn current() -> Option<PreparedSwitches> {
    OVERRIDE.with(Cell::get)
}

/// Holds the calling thread's switches at fixed values until dropped.
pub(crate) struct Override {
    previous: Option<PreparedSwitches>,
}

impl Override {
    pub(crate) fn set(switches: PreparedSwitches) -> Self {
        Self {
            previous: OVERRIDE.with(|cell| cell.replace(Some(switches))),
        }
    }
}

impl Drop for Override {
    fn drop(&mut self) {
        OVERRIDE.with(|cell| cell.set(self.previous));
    }
}

#[test]
fn the_override_is_scoped_to_its_guard_and_nests() {
    let before = current();
    {
        let _all = Override::set(PreparedSwitches::ALL);
        assert_eq!(current(), Some(PreparedSwitches::ALL));
        {
            let _off = Override::set(PreparedSwitches::OFF);
            assert_eq!(current(), Some(PreparedSwitches::OFF));
        }
        assert_eq!(current(), Some(PreparedSwitches::ALL));
    }
    assert_eq!(current(), before);
}
