//! What the engines resident in this process have committed of the host's
//! memory.
//!
//! Every `serve::load_engine` call reserves its engine's commitment here
//! before it reads a weight. A load that selects memory-carrying routes (a
//! DFlash2 drafter, prepared planes, retained scratch) is admitted against
//! what the engines already resident have committed, not against the
//! incoming model alone, and the decision and its reservation are one
//! transaction ([`Ledger::admit`]): two loads that run at once, through one
//! pool or through independent ones, cannot both decide against the same
//! total.
//!
//! A reservation is owned by whoever owns the model's buffers. While a load
//! is in progress that is the loading thread; `Engine::spawn*` then gives it
//! to the worker thread with the model, and it is released by the worker's
//! teardown alone (`WorkerTeardown` in `serve::api::engine_supervisor`):
//! after the model is dropped, the thread's GPU caches are released, the
//! thread has terminated, the dropped buffers are flushed from the Metal
//! residency set and the kernel has reclaimed them
//! ([`wait_until_reclaimed`]). Dropping the last `Engine` handle does not release it: an
//! evicted engine whose worker is still tearing down, or is wedged inside
//! Metal, still holds its memory and is still counted, and a teardown that
//! fails leaves it counted for good ([`Reservation::strand`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
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
}

#[derive(Default)]
struct Entries {
    next_id: u64,
    reserved: BTreeMap<u64, Entry>,
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
            },
        );
        id
    }
}

/// The commitments of the resident engines.
#[derive(Default)]
pub(crate) struct Ledger {
    entries: Mutex<Entries>,
}

impl Ledger {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn entries(&self) -> std::sync::MutexGuard<'_, Entries> {
        // The guarded state is plain bookkeeping, valid after any panic.
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Sum of every reservation's commitment.
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

    /// Decide a load against what is resident and reserve what it decided,
    /// as one transaction.
    ///
    /// `decide` receives the resident total and returns its decision with
    /// the commitment that decision needs. The commitment is reserved only
    /// if the resident total has not grown since `decide` read it; if
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
        let mut resident = self.total();
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
}

/// One engine's commitment; released when dropped.
pub(crate) struct Reservation {
    ledger: Arc<Ledger>,
    id: u64,
}

impl Reservation {
    /// Keep the commitment counted for the rest of the process: the teardown
    /// of the engine it stands for failed, so its memory cannot be taken to
    /// be free. `reason` is logged with the model.
    pub(crate) fn strand(self, reason: &str) {
        let entries = self.ledger.entries();
        let entry = entries.reserved.get(&self.id);
        tracing::error!(
            model = %entry.map_or_else(String::new, |entry| entry.model.display().to_string()),
            bytes = entry.map_or(0, |entry| entry.commitment.total()),
            reason,
            "engine memory stays reserved until the process restarts"
        );
        drop(entries);
        std::mem::forget(self);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger.entries().reserved.remove(&self.id);
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
/// The footprint counts as settled when it has not fallen for this many
/// polls in a row (100 ms).
const RECLAIM_QUIET_POLLS: u32 = 20;
/// A fall smaller than this between the start of a quiet stretch and a poll
/// is not a reclaim in progress.
const RECLAIM_TOLERANCE_BYTES: u64 = 64 << 20;
/// Polls after which a footprint that is still falling is a failed teardown
/// (60 s).
const RECLAIM_LIMIT_POLLS: u32 = 12_000;

/// Wait until the kernel has returned the memory of buffers this process
/// released.
///
/// Releasing a Metal buffer is not the end of it: the kernel reclaims the
/// pages afterwards, on its own. Measured on the Qwen3.6 gate artifact, 15
/// to 28 GiB of a 40 GiB engine were still charged to the process when every
/// thread of the engine had terminated and its devices were flushed, and
/// were gone 0.1 to 1 s later. The kernel gives no handle to wait on, so
/// this watches the one thing that shows it, the process footprint, until it
/// has stopped falling: until no poll in [`RECLAIM_QUIET_POLLS`] consecutive
/// polls is more than [`RECLAIM_TOLERANCE_BYTES`] below the first of them.
/// A footprint that keeps falling for [`RECLAIM_LIMIT_POLLS`] polls is an
/// error.
///
/// The footprint is the whole process's. Memory another engine frees at the
/// same time only prolongs the wait; memory allocated elsewhere in the
/// process faster than the kernel reclaims (it reclaimed 100 GiB/s and more
/// in the measurement) would hide a reclaim that is still running.
pub(crate) fn wait_until_reclaimed() -> anyhow::Result<()> {
    settle(process_footprint_bytes, std::thread::sleep)
}

fn settle(
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

    /// Runs [`settle`] over a scripted footprint (the last value repeats)
    /// and returns its result with the number of polls it made.
    fn settled(script: &[u64]) -> (anyhow::Result<()>, usize) {
        let mut reads = 0;
        let mut pauses = 0;
        let result = settle(
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
        let result = settle(
            || {
                reads += 1;
                Ok(u64::MAX - reads * 2 * RECLAIM_TOLERANCE_BYTES)
            },
            |_| polls += 1,
        );
        assert!(result.unwrap_err().to_string().contains("still falling"));
        assert_eq!(polls, RECLAIM_LIMIT_POLLS);
        // A footprint that cannot be read fails it at once.
        let unreadable = settle(|| anyhow::bail!("no rusage"), |_| {});
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
