//! What the engines of this process have committed of the host's memory, and
//! how a load is admitted against it.
//!
//! # The two bounds of an admission
//!
//! A load that selects memory-carrying routes (a DFlash2 drafter, prepared
//! planes, retained scratch) may take them only within both of these
//! (`dflash2_policy::memory_verdict`):
//!
//! - **the budget**: physical memory minus every commitment charged here,
//!   minus the incoming model's file and KV budget, minus a reserve. A
//!   commitment is an engine's whole claim (model file, KV budget, drafter,
//!   planes, scratch), allocated or not. What a running engine may still
//!   allocate (a KV cache it has not filled, planes it prepares on first
//!   use, scratch) is inside its charge, so the budget already withholds it
//!   from the incoming load.
//! - **the present**: the memory the host reports available at the decision
//!   (free, inactive and purgeable pages), minus the incoming model's file
//!   and the reserve. This is what nobody holds now, whatever the ledger
//!   says: it excludes memory of a torn-down engine that the kernel has not
//!   reclaimed yet, buffers a command still running holds, the allocator's
//!   cache, and every other process.
//!
//! The budget alone would hand out bytes that are promised to no engine but
//! not back from the kernel; the present alone would hand out bytes a
//! running engine has been promised and not yet taken. A load takes the
//! smaller of the two. Neither bound covers another process that grows
//! after the decision; the reserve is the only margin for that.
//!
//! # Why an admission cannot race
//!
//! - **Against another admission.** The decision and its reservation are
//!   one transaction ([`Ledger::admit`]): the commitment is reserved only if
//!   the charged total has not grown since the decision read it, else the
//!   decision is made again. Two loads, through one pool or through
//!   independent ones, are never both decided against the same total. The
//!   later of two loads sees the earlier one charged in full from the
//!   moment it was admitted, before it has allocated anything.
//! - **Against a teardown.** A reservation stays charged until its engine's
//!   teardown has completed, so a decision that overlaps a teardown counts
//!   the engine in full. An admission also waits for every teardown that is
//!   in progress before it decides ([`TeardownMarker`]), so that it reads
//!   the host's available memory after the engine has let go and not in the
//!   middle. A teardown that does not end does not block loads for good: the
//!   wait is bounded, and the engine stays charged.
//! - **Against a running engine that allocates.** Those allocations are
//!   inside the engine's charge: the budget withheld them before the engine
//!   made them.
//!
//! # Who releases a reservation
//!
//! A reservation is owned by whoever owns the model's buffers. While a load
//! is in progress that is the loading thread; `Engine::spawn*` then gives it
//! to the worker thread with the model, and it is released by the worker's
//! teardown alone (`WorkerTeardown` in `serve::api::engine_supervisor`):
//! after the model is dropped, the thread's GPU caches are dropped, the
//! thread has terminated and every queue the model's threads encoded on is
//! settled (`inference::device_settle`). Dropping the last `Engine` handle
//! does not release it: an evicted engine whose worker is still tearing
//! down, or is wedged inside Metal, still holds its memory and is still
//! charged, and a teardown that fails leaves it charged for good
//! ([`Reservation::strand`]).
//!
//! The kernel returns the pages of released buffers on its own, some time
//! after user space has let go. The ledger does not claim to know when: the
//! present bound is what keeps those bytes from being handed out.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Duration;

/// What one engine holds resident, by component.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Commitment {
    /// The GGUF's weights.
    pub(crate) model_file_bytes: u64,
    /// The configured aggregate KV budget; zero when none is configured.
    pub(crate) kv_budget_bytes: u64,
    /// The DFlash2 drafter checkpoint.
    pub(crate) drafter_bytes: u64,
    /// Prepared planes of the fast routes.
    pub(crate) plane_bytes: u64,
    /// Scratch the fast routes retain between steps.
    pub(crate) scratch_bytes: u64,
}

impl Commitment {
    /// Sum of every component.
    pub(crate) fn total(&self) -> u64 {
        [
            self.model_file_bytes,
            self.kv_budget_bytes,
            self.drafter_bytes,
            self.plane_bytes,
            self.scratch_bytes,
        ]
        .into_iter()
        .fold(0, u64::saturating_add)
    }
}

struct Entry {
    /// The model the engine was loaded from, for diagnostics.
    model: PathBuf,
    commitment: Commitment,
    /// The thread that owns the model has begun to tear it down.
    tearing_down: bool,
}

#[derive(Default)]
struct Entries {
    next_id: u64,
    reserved: BTreeMap<u64, Entry>,
    /// Reservations whose teardown is in progress.
    tearing_down: usize,
    /// Admissions waiting for those teardowns.
    #[cfg(test)]
    admissions_waiting: usize,
}

impl Entries {
    fn total(&self) -> u64 {
        self.reserved
            .values()
            .map(|entry| entry.commitment.total())
            .fold(0, u64::saturating_add)
    }

    fn insert(&mut self, model: &Path, commitment: Commitment) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.reserved.insert(
            id,
            Entry {
                model: model.to_path_buf(),
                commitment,
                tearing_down: false,
            },
        );
        id
    }

    /// The teardown of `id` is over, released or stranded. Returns whether
    /// one was in progress.
    fn end_teardown(&mut self, id: u64, release: bool) -> bool {
        let entry = if release {
            self.reserved.remove(&id)
        } else {
            None
        };
        let was_tearing_down = match (entry, self.reserved.get_mut(&id)) {
            (Some(entry), _) => entry.tearing_down,
            (None, Some(entry)) => std::mem::take(&mut entry.tearing_down),
            (None, None) => false,
        };
        if was_tearing_down {
            self.tearing_down -= 1;
        }
        was_tearing_down
    }
}

/// How long an admission waits for teardowns in progress before it decides
/// with their engines still charged.
const ADMISSION_TEARDOWN_WAIT: Duration = Duration::from_secs(30);

/// The commitments of the engines of this process.
pub(crate) struct Ledger {
    entries: Mutex<Entries>,
    teardown_ended: Condvar,
    teardown_wait: Duration,
}

impl Ledger {
    pub(crate) fn new() -> Arc<Self> {
        Self::with_teardown_wait(ADMISSION_TEARDOWN_WAIT)
    }

    /// A ledger whose admissions wait at most `teardown_wait` for teardowns
    /// in progress.
    pub(crate) fn with_teardown_wait(teardown_wait: Duration) -> Arc<Self> {
        Arc::new(Self {
            entries: Mutex::default(),
            teardown_ended: Condvar::new(),
            teardown_wait,
        })
    }

    fn entries(&self) -> std::sync::MutexGuard<'_, Entries> {
        // The guarded state is plain bookkeeping, valid after any panic.
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Sum of every reservation's commitment.
    #[cfg(test)]
    pub(crate) fn total(&self) -> u64 {
        self.entries().total()
    }

    /// Reserve `commitment` for the engine being loaded from `model`,
    /// whatever is resident: a route that is not subject to an admission
    /// rule.
    pub(crate) fn reserve(self: &Arc<Self>, model: &Path, commitment: Commitment) -> Reservation {
        Reservation {
            id: self.entries().insert(model, commitment),
            ledger: Arc::clone(self),
        }
    }

    /// Wait for the teardowns in progress to end, for at most the ledger's
    /// bound, and return the charged total.
    fn total_after_teardowns(&self, model: &Path) -> u64 {
        // Mutated only by the test-only count of waiting admissions.
        #[cfg_attr(not(test), allow(unused_mut))]
        let mut entries = self.entries();
        if entries.tearing_down > 0 {
            tracing::info!(
                model = %model.display(),
                teardowns = entries.tearing_down,
                "load waits for engine teardowns in progress"
            );
        }
        #[cfg(test)]
        {
            entries.admissions_waiting += 1;
        }
        #[cfg_attr(not(test), allow(unused_mut))]
        let (mut entries, waited) = self
            .teardown_ended
            .wait_timeout_while(entries, self.teardown_wait, |entries| {
                entries.tearing_down > 0
            })
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        #[cfg(test)]
        {
            entries.admissions_waiting -= 1;
        }
        if waited.timed_out() {
            tracing::warn!(
                model = %model.display(),
                teardowns = entries.tearing_down,
                waited_s = self.teardown_wait.as_secs(),
                "engine teardowns did not end; the load is decided with their engines charged"
            );
        }
        entries.total()
    }

    /// Decide a load against what is charged and reserve what it decided,
    /// as one transaction, after the teardowns in progress have ended.
    ///
    /// `decide` receives the charged total and returns its decision with
    /// the commitment that decision needs. The commitment is reserved only
    /// if the charged total has not grown since `decide` read it; if
    /// another load reserved in between, `decide` runs again against the new
    /// total. A total that shrank leaves the decision valid, only more
    /// cautious than it had to be. `decide` runs without the ledger locked,
    /// so a slow decision (one that downloads a drafter) blocks no other
    /// load and no release.
    pub(crate) fn admit<T, E>(
        self: &Arc<Self>,
        model: &Path,
        mut decide: impl FnMut(u64) -> Result<(T, Commitment), E>,
    ) -> Result<(T, Reservation), E> {
        let mut resident = self.total_after_teardowns(model);
        loop {
            let (decision, commitment) = decide(resident)?;
            let mut entries = self.entries();
            let now = entries.total();
            if now <= resident {
                let id = entries.insert(model, commitment);
                drop(entries);
                return Ok((
                    decision,
                    Reservation {
                        ledger: Arc::clone(self),
                        id,
                    },
                ));
            }
            resident = now;
        }
    }

    /// Admissions that are waiting for a teardown to end.
    #[cfg(test)]
    pub(crate) fn admissions_waiting(&self) -> usize {
        self.entries().admissions_waiting
    }
}

/// One engine's commitment; released when dropped.
pub(crate) struct Reservation {
    ledger: Arc<Ledger>,
    id: u64,
}

impl Reservation {
    /// What the thread that owns the engine's model uses to say that its
    /// teardown has begun.
    pub(crate) fn teardown_marker(&self) -> TeardownMarker {
        TeardownMarker {
            ledger: Arc::clone(&self.ledger),
            id: self.id,
        }
    }

    /// Keep the commitment charged for the rest of the process: the teardown
    /// of the engine it stands for failed, so its memory cannot be taken to
    /// be free. `reason` is logged with the model.
    pub(crate) fn strand(self, reason: &str) {
        let mut entries = self.ledger.entries();
        let entry = entries.reserved.get(&self.id);
        tracing::error!(
            model = %entry.map_or_else(String::new, |entry| entry.model.display().to_string()),
            bytes = entry.map_or(0, |entry| entry.commitment.total()),
            reason,
            "engine memory stays reserved until the process restarts"
        );
        let ended = entries.end_teardown(self.id, false);
        drop(entries);
        if ended {
            self.ledger.teardown_ended.notify_all();
        }
        std::mem::forget(self);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let ended = self.ledger.entries().end_teardown(self.id, true);
        if ended {
            self.ledger.teardown_ended.notify_all();
        }
    }
}

impl std::fmt::Debug for Reservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Reservation")
            .field("id", &self.id)
            .finish()
    }
}

/// Marks the start of an engine's teardown. From [`Self::begin`] until the
/// reservation is released or stranded, new admissions wait
/// ([`Ledger::admit`]).
pub(crate) struct TeardownMarker {
    ledger: Arc<Ledger>,
    id: u64,
}

impl TeardownMarker {
    /// The thread that owns the model has stopped serving and is giving the
    /// model's memory back.
    pub(crate) fn begin(self) {
        let mut entries = self.ledger.entries();
        if let Some(entry) = entries.reserved.get_mut(&self.id) {
            if !std::mem::replace(&mut entry.tearing_down, true) {
                entries.tearing_down += 1;
            }
        }
    }
}

/// The process's physical footprint: what the kernel charges the process
/// for, GPU buffers included.
pub(crate) fn process_footprint_bytes() -> anyhow::Result<u64> {
    // SAFETY: `rusage_info_v4` is plain data, and `proc_pid_rusage` fills
    // the buffer of the flavor it is asked for.
    let mut info = unsafe { std::mem::zeroed::<libc::rusage_info_v4>() };
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            (&mut info as *mut libc::rusage_info_v4).cast(),
        )
    };
    anyhow::ensure!(
        rc == 0,
        "proc_pid_rusage failed: {}",
        std::io::Error::last_os_error()
    );
    Ok(info.ri_phys_footprint)
}

/// How often the footprint is read while the kernel reclaims.
const RECLAIM_POLL: Duration = Duration::from_millis(5);
/// The footprint counts as quiet when it has not fallen for this many polls
/// in a row (100 ms).
const RECLAIM_QUIET_POLLS: u32 = 20;
/// A fall smaller than this between the start of a quiet stretch and a poll
/// is not a reclaim in progress.
const RECLAIM_TOLERANCE_BYTES: u64 = 64 << 20;
/// Polls after which the wait gives up (10 s).
const RECLAIM_LIMIT_POLLS: u32 = 2_000;

/// Give the kernel time to return the memory of buffers this process
/// released, before the loads that waited for a teardown read what the host
/// has available.
///
/// Releasing a Metal buffer is not the end of it: the kernel reclaims the
/// pages afterwards, on its own, and gives no handle to wait on. Measured on
/// the Qwen3.6 gate artifact, 15 to 28 GiB of a 40 GiB engine were still
/// charged to the process when every thread of the engine had terminated
/// and its queues were settled, and were gone 0.1 to 1 s later. This watches
/// the process footprint until [`RECLAIM_QUIET_POLLS`] consecutive polls are
/// none more than [`RECLAIM_TOLERANCE_BYTES`] below the first of them, and
/// reports whether it saw that within [`RECLAIM_LIMIT_POLLS`] polls.
///
/// It is a courtesy, not a proof, and nothing relies on it for safety. The
/// footprint is the whole process's: it cannot say that one engine's bytes
/// are back. A load that reads the host's available memory too early is
/// admitted to less than it could have had; what keeps it from being
/// admitted to more is the present bound of the admission (module
/// documentation).
pub(crate) fn await_kernel_reclaim() -> anyhow::Result<()> {
    quiet_footprint(process_footprint_bytes, std::thread::sleep)
}

fn quiet_footprint(
    mut footprint: impl FnMut() -> anyhow::Result<u64>,
    mut pause: impl FnMut(Duration),
) -> anyhow::Result<()> {
    let mut reference = footprint()?;
    let mut quiet = 0;
    for _ in 0..RECLAIM_LIMIT_POLLS {
        pause(RECLAIM_POLL);
        let now = footprint()?;
        if now.saturating_add(RECLAIM_TOLERANCE_BYTES) < reference {
            reference = now;
            quiet = 0;
        } else {
            quiet += 1;
            if quiet == RECLAIM_QUIET_POLLS {
                return Ok(());
            }
        }
    }
    anyhow::bail!(
        "the process footprint was still falling after {} s",
        (RECLAIM_POLL * RECLAIM_LIMIT_POLLS).as_secs()
    )
}

static PROCESS: LazyLock<Arc<Ledger>> = LazyLock::new(Ledger::new);

/// The ledger of this process.
pub(crate) fn process() -> &'static Arc<Ledger> {
    &PROCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn commitment(model_gib: u64, planes_gib: u64) -> Commitment {
        Commitment {
            model_file_bytes: model_gib * GIB,
            kv_budget_bytes: 4 * GIB,
            drafter_bytes: GIB,
            plane_bytes: planes_gib * GIB,
            scratch_bytes: 0,
        }
    }

    #[test]
    fn a_reservation_is_counted_until_it_is_dropped() {
        let ledger = Ledger::new();
        assert_eq!(ledger.total(), 0);
        let first = ledger.reserve(Path::new("/a.gguf"), commitment(16, 10));
        let second = ledger.reserve(Path::new("/b.gguf"), commitment(20, 0));
        assert_eq!(ledger.total(), (16 + 4 + 1 + 10 + 20 + 4 + 1) * GIB);
        // A load that fails, or an engine that is unloaded, drops its
        // reservation.
        drop(first);
        assert_eq!(ledger.total(), (20 + 4 + 1) * GIB);
        drop(second);
        assert_eq!(ledger.total(), 0);
    }

    #[test]
    fn a_decision_is_made_against_the_resident_total_and_reserved() {
        let ledger = Ledger::new();
        let _first = ledger.reserve(Path::new("/a.gguf"), commitment(16, 10));
        let (seen, reservation) = ledger
            .admit(Path::new("/b.gguf"), |resident| {
                Ok::<_, ()>((resident, commitment(20, 0)))
            })
            .expect("admitted");
        assert_eq!(seen, (16 + 4 + 1 + 10) * GIB);
        assert_eq!(ledger.total(), (16 + 4 + 1 + 10 + 20 + 4 + 1) * GIB);
        drop(reservation);
        assert_eq!(ledger.total(), (16 + 4 + 1 + 10) * GIB);
    }

    #[test]
    fn a_refused_decision_reserves_nothing() {
        let ledger = Ledger::new();
        let refused = ledger.admit(Path::new("/a.gguf"), |_| {
            Err::<((), Commitment), _>("required route does not fit")
        });
        assert_eq!(refused.err(), Some("required route does not fit"));
        assert_eq!(ledger.total(), 0);
    }

    #[test]
    fn a_reservation_that_lands_during_a_decision_makes_it_decide_again() {
        let ledger = Ledger::new();
        let mut other = None;
        let mut seen = Vec::new();
        let (planes, _reservation) = ledger
            .admit(Path::new("/a.gguf"), |resident| {
                seen.push(resident);
                // Another load, through another pool, reserves while this
                // one is still deciding against the total it read.
                if other.is_none() {
                    other = Some(ledger.reserve(Path::new("/b.gguf"), commitment(20, 10)));
                }
                // The decision a memory rule makes: planes only on a host
                // nothing else is resident on.
                let planes = if resident == 0 { 10 } else { 0 };
                Ok::<_, ()>((planes, commitment(16, planes)))
            })
            .expect("admitted");
        assert_eq!(seen, vec![0, (20 + 4 + 1 + 10) * GIB]);
        assert_eq!(planes, 0, "the stale decision was not reserved");
        assert_eq!(ledger.total(), (20 + 4 + 1 + 10 + 16 + 4 + 1) * GIB);
    }

    #[test]
    fn a_release_during_a_decision_does_not_repeat_it() {
        let ledger = Ledger::new();
        let mut other = Some(ledger.reserve(Path::new("/b.gguf"), commitment(20, 10)));
        let mut decisions = 0;
        let _reservation = ledger
            .admit(Path::new("/a.gguf"), |_| {
                decisions += 1;
                other.take();
                Ok::<_, ()>(((), commitment(16, 0)))
            })
            .expect("admitted");
        assert_eq!(decisions, 1);
        assert_eq!(ledger.total(), (16 + 4 + 1) * GIB);
    }

    #[test]
    fn concurrent_loads_never_both_take_what_fits_once() {
        // Each load takes the planes only if they fit beside what is
        // resident. They fit once. Whatever the interleaving, exactly one
        // load gets them.
        const LOADS: usize = 8;
        const BOUND: u64 = 40 * GIB;
        for _ in 0..50 {
            let ledger = Ledger::new();
            let start = Arc::new(std::sync::Barrier::new(LOADS));
            let loads: Vec<_> = (0..LOADS)
                .map(|_| {
                    let (ledger, start) = (Arc::clone(&ledger), Arc::clone(&start));
                    std::thread::spawn(move || {
                        start.wait();
                        ledger
                            .admit(Path::new("/a.gguf"), |resident| {
                                let planes = Commitment {
                                    plane_bytes: 30 * GIB,
                                    ..Commitment::default()
                                };
                                let fits = resident.saturating_add(planes.total()) <= BOUND;
                                std::thread::yield_now();
                                Ok::<_, ()>((
                                    fits,
                                    if fits { planes } else { Commitment::default() },
                                ))
                            })
                            .expect("admitted")
                    })
                })
                .collect();
            let admitted: Vec<_> = loads
                .into_iter()
                .map(|load| load.join().expect("load thread"))
                .collect();
            assert_eq!(admitted.iter().filter(|(fits, _)| *fits).count(), 1);
            assert_eq!(ledger.total(), 30 * GIB);
        }
    }

    #[test]
    fn a_reservation_is_counted_while_another_thread_owns_it() {
        // The worker thread owns its engine's reservation: the ledger
        // counts it until that thread lets go, whoever else is gone.
        let ledger = Ledger::new();
        let reservation = ledger.reserve(Path::new("/a.gguf"), commitment(16, 10));
        let (release, released) = std::sync::mpsc::channel::<()>();
        let worker = std::thread::spawn(move || {
            let _held = reservation;
            let _ = released.recv();
        });
        assert_eq!(ledger.total(), (16 + 4 + 1 + 10) * GIB);
        drop(release);
        worker.join().expect("worker");
        assert_eq!(ledger.total(), 0);
    }

    #[test]
    fn a_stranded_reservation_is_counted_for_good() {
        let ledger = Ledger::new();
        let failed = ledger.reserve(Path::new("/a.gguf"), commitment(16, 10));
        let live = ledger.reserve(Path::new("/b.gguf"), commitment(20, 0));
        failed.strand("its teardown failed");
        drop(live);
        assert_eq!(ledger.total(), (16 + 4 + 1 + 10) * GIB);
    }

    /// Runs [`quiet_footprint`] over a scripted footprint (the last value repeats)
    /// and returns its result with the number of polls it made.
    fn settled(script: &[u64]) -> (anyhow::Result<()>, usize) {
        let mut reads = 0;
        let mut pauses = 0;
        let result = quiet_footprint(
            || {
                let value = script[reads.min(script.len() - 1)];
                reads += 1;
                Ok(value)
            },
            |pause| {
                assert_eq!(pause, RECLAIM_POLL);
                pauses += 1;
            },
        );
        (result, pauses)
    }

    #[test]
    fn the_reclaim_wait_ends_when_the_footprint_has_stopped_falling() {
        let quiet = RECLAIM_QUIET_POLLS as usize;
        // Nothing to reclaim: one quiet stretch.
        let (result, polls) = settled(&[40 * GIB]);
        assert!(result.is_ok());
        assert_eq!(polls, quiet);
        // The kernel is still reclaiming: 28 GiB, then 26, 7, 0.2 and flat.
        // The quiet stretch starts at the last fall.
        let falling = [28 * GIB, 26 * GIB, 7 * GIB, GIB / 5];
        let (result, polls) = settled(&falling);
        assert!(result.is_ok());
        assert_eq!(polls, falling.len() - 1 + quiet);
        // A fall that resumes inside a quiet stretch restarts it.
        let mut resumed = vec![28 * GIB; quiet];
        resumed.push(GIB);
        let (result, polls) = settled(&resumed);
        assert!(result.is_ok());
        assert_eq!(polls, resumed.len() - 1 + quiet);
        // A footprint that rises (another engine serving) is not a reclaim.
        let (result, polls) = settled(&[GIB, 2 * GIB, 3 * GIB]);
        assert!(result.is_ok());
        assert_eq!(polls, quiet);
        // Neither is a fall within the tolerance.
        let (result, polls) = settled(&[GIB, GIB - RECLAIM_TOLERANCE_BYTES]);
        assert!(result.is_ok());
        assert_eq!(polls, quiet);
    }

    #[test]
    fn a_footprint_that_never_stops_falling_fails_the_wait() {
        let mut reads = 0_u64;
        let mut polls = 0;
        let result = quiet_footprint(
            || {
                reads += 1;
                Ok(u64::MAX - reads * 2 * RECLAIM_TOLERANCE_BYTES)
            },
            |_| polls += 1,
        );
        assert!(result.unwrap_err().to_string().contains("still falling"));
        assert_eq!(polls, RECLAIM_LIMIT_POLLS);
        // A footprint that cannot be read fails it at once.
        let unreadable = quiet_footprint(|| anyhow::bail!("no rusage"), |_| {});
        assert!(unreadable.is_err());
    }

    #[test]
    fn the_process_footprint_is_readable() {
        assert!(process_footprint_bytes().expect("footprint") > 0);
    }

    #[test]
    fn a_commitment_sums_its_components_without_wrapping() {
        let all = Commitment {
            model_file_bytes: 1,
            kv_budget_bytes: 2,
            drafter_bytes: 3,
            plane_bytes: 4,
            scratch_bytes: 5,
        };
        assert_eq!(all.total(), 15);
        let huge = Commitment {
            model_file_bytes: u64::MAX,
            ..all
        };
        assert_eq!(huge.total(), u64::MAX);
    }
}
