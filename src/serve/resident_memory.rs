//! What the engines resident in this process have committed of the host's
//! memory.
//!
//! Every `serve::load_engine` call reserves its engine's commitment here
//! before it reads a weight, and the engine holds the reservation until it is
//! unloaded. A load that selects memory-carrying routes (a DFlash2 drafter,
//! prepared planes, retained scratch) is admitted against what the engines
//! already resident have committed, not against the incoming model alone. A
//! failed load releases its reservation; a load retried on another route
//! amends it.
//!
//! The reservation also carries the route decision the engine was loaded
//! with, which `/hf2q/v1/runtime` publishes so that a gate receipt binds the
//! profile that actually ran.

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
    published: Option<serde_json::Value>,
}

/// The commitments of the resident engines.
#[derive(Default)]
pub(crate) struct Ledger {
    entries: Mutex<(u64, BTreeMap<u64, Entry>)>,
}

impl Ledger {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Sum of every reservation's commitment.
    pub(crate) fn total(&self) -> u64 {
        self.entries
            .lock()
            .expect("resident memory ledger")
            .1
            .values()
            .map(|entry| entry.commitment.total())
            .fold(0, u64::saturating_add)
    }

    /// Reserve `commitment` for the engine loaded from `model` until the
    /// returned reservation is dropped.
    pub(crate) fn reserve(self: &Arc<Self>, model: &Path, commitment: Commitment) -> Reservation {
        let mut entries = self.entries.lock().expect("resident memory ledger");
        let id = entries.0;
        entries.0 += 1;
        entries.1.insert(
            id,
            Entry {
                model: model.to_path_buf(),
                commitment,
                published: None,
            },
        );
        Reservation {
            ledger: Arc::clone(self),
            id,
        }
    }
}

/// One engine's commitment; released when dropped.
pub(crate) struct Reservation {
    ledger: Arc<Ledger>,
    id: u64,
}

impl Reservation {
    fn with_entry(&self, change: impl FnOnce(&mut Entry)) {
        if let Some(entry) = self
            .ledger
            .entries
            .lock()
            .expect("resident memory ledger")
            .1
            .get_mut(&self.id)
        {
            change(entry);
        }
    }

    /// Replace the commitment: the engine is being loaded on another route.
    pub(crate) fn amend(&self, commitment: Commitment) {
        self.with_entry(|entry| entry.commitment = commitment);
    }

    /// Publish the route decision the engine was loaded with.
    pub(crate) fn publish(&self, decision: serde_json::Value) {
        self.with_entry(|entry| entry.published = Some(decision));
    }

    /// The published route decision.
    pub(crate) fn published(&self) -> Option<serde_json::Value> {
        let mut published = None;
        self.with_entry(|entry| published = entry.published.clone());
        published
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger
            .entries
            .lock()
            .expect("resident memory ledger")
            .1
            .remove(&self.id);
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
    fn a_load_retried_on_another_route_amends_its_commitment() {
        let ledger = Ledger::new();
        let reservation = ledger.reserve(Path::new("/a.gguf"), commitment(16, 10));
        reservation.amend(Commitment {
            model_file_bytes: 16 * GIB,
            kv_budget_bytes: 4 * GIB,
            ..Commitment::default()
        });
        assert_eq!(ledger.total(), 20 * GIB);
    }

    #[test]
    fn a_reservation_carries_the_decision_its_engine_published() {
        let ledger = Ledger::new();
        let first = ledger.reserve(Path::new("/a.gguf"), commitment(16, 0));
        let second = ledger.reserve(Path::new("/a.gguf"), commitment(16, 10));
        assert_eq!(first.published(), None);
        first.publish(serde_json::json!({"selected": false}));
        second.publish(serde_json::json!({"selected": true}));
        assert_eq!(
            first.published(),
            Some(serde_json::json!({"selected": false}))
        );
        assert_eq!(
            second.published(),
            Some(serde_json::json!({"selected": true}))
        );
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
