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
//! is in progress that is the loading thread; once the worker thread is
//! spawned it is the worker, which releases the reservation after it has
//! dropped the model (`EngineSupervisor::hold_worker_memory`). Dropping the
//! last `Engine` handle does not release it: an evicted engine whose worker
//! is still tearing down, or is wedged inside Metal, still holds its memory
//! and is still counted.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};

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
    #[allow(dead_code)]
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
