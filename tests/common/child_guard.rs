//! RAII guard that reaps spawned child processes on drop.
//!
//! Tests that spawn `hf2q serve` (or any long-running child) must
//! `.wait()` the child to avoid leaving zombie processes — even when
//! the test ends via panic.  Without an explicit `wait()` call the
//! kernel keeps the child entry in the process table until the test
//! binary itself exits and init reaps it; clippy 1.94+ flags this as
//! `clippy::zombie_processes`.
//!
//! Wrapping the spawned child in [`ChildGuard`] makes cleanup
//! automatic — `Drop` fires on normal scope exit and during panic
//! unwinding (the default `panic = unwind` strategy), issuing a
//! `kill()` followed by a `wait()`.  `Drop` does **not** fire on
//! abort-style termination — `std::process::abort()`, `panic = abort`,
//! `SIGKILL`, OOM — but in those cases the test binary is dying
//! anyway; the OS reparents any live orphaned children to init,
//! which reaps them once they exit.
//!
//! Usage:
//! ```ignore
//! let mut child = ChildGuard::new(
//!     std::process::Command::new(bin)
//!         .args([...])
//!         .spawn()
//!         .expect("spawn hf2q serve"),
//! );
//! // … test work …
//! if !some_cond {
//!     child.kill();          // explicit early kill + wait, idempotent
//!     panic!("…");
//! }
//! // end-of-test: no explicit kill needed — Drop reaps.
//! ```

use std::process::Child;

pub struct ChildGuard {
    child: Child,
    reaped: bool,
}

impl ChildGuard {
    pub fn new(child: Child) -> Self {
        Self {
            child,
            reaped: false,
        }
    }

    /// Kill the child now and wait for it to exit.  Idempotent — the
    /// Drop reaper notices `reaped == true` and skips re-kill.
    /// Failures from either syscall are intentionally swallowed: the
    /// caller is usually about to `panic!`, and there is nothing
    /// useful to do with an `ESRCH` from `kill()` (child already
    /// dead) or a `wait()` error on an unparented process.
    pub fn kill(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.reaped = true;
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        self.kill();
    }
}
