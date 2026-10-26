//! Immutable, per-engine publication of worker-owned Qwen pool observations.
//! Never owns a Metal resource and never claims physical bytes or a high-water.
use serde_json::{json, Value};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

const STALE_AFTER: Duration = Duration::from_secs(60);
const EXCLUDED: &[&str] = &[
    "kv_tq_mtp_caches_and_anchors",
    "gpu_cache_output_rings_and_hidden_parents",
    "idmm_scratch",
    "weight_pool",
    "direct_native_allocations",
    "other_worker_tls",
    "other_families",
    "driver_and_allocator_metadata",
];

#[derive(Default)]
pub(super) struct Store {
    owner: Option<ThreadId>,
    sequence: u64,
    status: &'static str,
    last: Option<Capture>,
}
struct Capture {
    sequence: u64,
    at: Instant,
    worker: String,
    slot: usize,
    slot_generation: u64,
    mode: &'static str,
    pool: Value,
}
impl Store {
    pub(super) fn bind(&mut self, owner: ThreadId) -> bool {
        if self.owner.is_some() {
            return false;
        }
        self.owner = Some(owner);
        self.status = "NeverObserved";
        true
    }
    pub(super) fn owns(&self, thread: ThreadId) -> bool {
        self.owner == Some(thread)
    }
    pub(super) fn invalidate(&mut self, reason: &'static str) {
        self.status = reason;
    }
    pub(super) fn publish(
        &mut self,
        owner: ThreadId,
        at: Instant,
        slot: usize,
        generation: u64,
        mode: &'static str,
        pool: Value,
    ) -> bool {
        if !self.owns(owner) {
            return false;
        }
        let Some(sequence) = self.sequence.checked_add(1) else {
            self.status = "SequenceOverflow";
            return false;
        };
        self.sequence = sequence;
        self.last = Some(Capture {
            sequence,
            at,
            worker: format!("{owner:?}"),
            slot,
            slot_generation: generation,
            mode,
            pool,
        });
        self.status = "ObservedAfterSuccessfulRequestDrain";
        true
    }
    pub(super) fn record_result(
        &mut self,
        owner: ThreadId,
        at: Instant,
        slot: usize,
        generation: u64,
        result: Result<(&'static str, Value), String>,
    ) {
        if !self.owns(owner) {
            return;
        }
        match result {
            Ok((mode, pool)) => {
                self.publish(owner, at, slot, generation, mode, pool);
            }
            Err(_) => self.invalidate("NativeCensusUnavailableLastObservationNotCurrent"),
        }
    }
    pub(super) fn view(&self, now: Instant, healthy: bool) -> Value {
        let last = self.last.as_ref().map(|c| {
            let age = now.checked_duration_since(c.at).unwrap_or_default();
            json!({"sequence": c.sequence, "capture_age_ms": age.as_millis(),
                "freshness": if age > STALE_AFTER { "StaleLastObservation" } else { "LastObservation" },
                "worker_thread": c.worker, "slot": c.slot, "slot_generation": c.slot_generation,
                "phase": "successful_request_completion_after_terminal_forward_readback",
                "pools": [{"name":"qwen35_decode_scratch", "mode":c.mode, "capacity":c.pool}]})
        });
        json!({"schema":"hf2q.worker-pool-census.v1", "status": if healthy {
                    if self.owner.is_none() { "WorkerNotRegistered" } else { self.status }
                } else { "WorkerFailedLastObservationNotCurrent" },
            "last_observation":last, "excluded_domains":EXCLUDED,
            "cross_pool_total":"Unavailable:no_cross_pool_identity_proof",
            "physical_residency":"Unavailable", "owners_outside_pool":"Unavailable",
            "lifetime_high_water":"Unavailable"})
    }
}

pub(super) fn unavailable(reason: &str) -> Value {
    json!({"schema":"hf2q.worker-pool-census.v1", "status":reason,
        "last_observation":null, "excluded_domains":EXCLUDED,
        "cross_pool_total":"Unavailable", "physical_residency":"Unavailable",
        "owners_outside_pool":"Unavailable", "lifetime_high_water":"Unavailable"})
}

pub(super) fn project(snapshot: mlx_native::PoolMemorySnapshot) -> Value {
    let count = |n: mlx_native::PoolRootCapacity| json!({"roots":n.roots,"bytes":n.bytes});
    json!({"retained_union":count(snapshot.retained), "free":count(snapshot.free),
        "issued":count(snapshot.issued), "residency_hint":count(snapshot.residency_hint),
        "labels_overlap":snapshot.labels_overlap,
        "units":"complete_metal_object_buffer_length_capacity_bytes",
        "label_totals_are_not_additive":true,
        "physical_residency":"Unavailable", "owners_outside_pool":"Unavailable",
        "lifetime_high_water":"Unavailable"})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_census_owner_and_immutable_last_record() {
        let mut store = Store::default();
        let owner = std::thread::current().id();
        assert!(store.bind(owner));
        assert!(!store.bind(owner));
        let now = Instant::now();
        assert!(store.publish(
            owner,
            now,
            2,
            7,
            "legacy",
            json!({"retained_union":{"bytes":24}})
        ));
        let old = store.view(now, true);
        assert!(store.publish(
            owner,
            now,
            3,
            8,
            "growth_limited",
            json!({"retained_union":{"bytes":48}})
        ));
        assert_eq!(
            old["last_observation"]["pools"][0]["capacity"]["retained_union"]["bytes"],
            24
        );
        assert_eq!(store.view(now, true)["last_observation"]["sequence"], 2);
    }
    #[test]
    fn worker_census_unknown_failure_and_age_are_not_zero_or_peak() {
        let mut store = Store::default();
        let owner = std::thread::current().id();
        let now = Instant::now();
        assert!(store.view(now, true)["last_observation"].is_null());
        store.bind(owner);
        store.publish(
            owner,
            now,
            0,
            1,
            "legacy",
            json!({"retained_union":{"bytes":24}}),
        );
        store.record_result(
            owner,
            now,
            0,
            1,
            Err("injected fallible census error".to_string()),
        );
        assert_eq!(
            store.view(now, true)["status"],
            "NativeCensusUnavailableLastObservationNotCurrent"
        );
        assert_eq!(
            store.view(now, true)["last_observation"]["pools"][0]["capacity"]["retained_union"]
                ["bytes"],
            24
        );
        store.invalidate("RequestCancelledLastObservationNotCurrent");
        let view = store.view(now + Duration::from_secs(61), true);
        assert_eq!(view["status"], "RequestCancelledLastObservationNotCurrent");
        assert_eq!(
            view["last_observation"]["freshness"],
            "StaleLastObservation"
        );
        assert_eq!(view["physical_residency"], "Unavailable");
        assert_eq!(view["lifetime_high_water"], "Unavailable");
        assert_eq!(
            store.view(now, false)["status"],
            "WorkerFailedLastObservationNotCurrent"
        );
        assert!(Store::default().view(now, true)["last_observation"].is_null());
    }
    #[test]
    fn worker_census_sequence_overflow_preserves_old_as_last_known() {
        let mut store = Store::default();
        let owner = std::thread::current().id();
        store.bind(owner);
        store.sequence = u64::MAX;
        assert!(!store.publish(owner, Instant::now(), 0, 0, "legacy", json!({})));
        assert_eq!(
            store.view(Instant::now(), true)["status"],
            "SequenceOverflow"
        );
    }
    #[test]
    fn worker_census_wrong_worker_cannot_publish_or_reuse_another_engine() {
        let mut store = Store::default();
        let owner = std::thread::current().id();
        let other = std::thread::spawn(|| std::thread::current().id())
            .join()
            .unwrap();
        store.bind(owner);
        assert!(!store.publish(other, Instant::now(), 0, 0, "legacy", json!({"bytes":24})));
        assert!(store.view(Instant::now(), true)["last_observation"].is_null());
        assert!(Store::default().view(Instant::now(), true)["last_observation"].is_null());
        let disabled = unavailable("Disabled");
        assert_eq!(disabled["status"], "Disabled");
        assert!(disabled["last_observation"].is_null());
        assert!(disabled.get("total_bytes").is_none());
    }
    #[test]
    fn worker_census_native_overlap_is_projected_without_addition() {
        let n = mlx_native::PoolRootCapacity {
            roots: 1,
            bytes: 24,
        };
        let view = project(mlx_native::PoolMemorySnapshot {
            retained: n,
            free: n,
            issued: n,
            residency_hint: n,
            labels_overlap: true,
            physical_residency: mlx_native::PoolMemoryUnavailable::PhysicalResidency,
            owners_outside_pool: mlx_native::PoolMemoryUnavailable::OwnersOutsidePool,
            lifetime_high_water: mlx_native::PoolMemoryUnavailable::LifetimeHighWater,
        });
        assert_eq!(view["retained_union"]["bytes"], 24);
        assert_eq!(view["free"]["bytes"], 24);
        assert_eq!(view["issued"]["bytes"], 24);
        assert_eq!(view["labels_overlap"], true);
        assert!(view.get("total_bytes").is_none());
    }
}
