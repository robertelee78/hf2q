use super::*;

#[test]
fn views_pools_and_queued_references_retain_one_parent_until_last_release() {
    let mut ledger = Ownership::new(identity()).unwrap();
    ledger
        .apply(
            &identity(),
            &[
                allocation(1, 1, 80),
                Event::Retain {
                    source: HandleId(1),
                    handle: HandleId(2),
                }, // tiny view
                Event::Retain {
                    source: HandleId(1),
                    handle: HandleId(3),
                }, // pool residency
                Event::Retain {
                    source: HandleId(2),
                    handle: HandleId(4),
                }, // queued command
                Event::Release {
                    handle: HandleId(1),
                },
                Event::Release {
                    handle: HandleId(2),
                },
                Event::Release {
                    handle: HandleId(3),
                },
            ],
            &limits(),
        )
        .unwrap();
    assert_eq!(ledger.snapshot().current, bytes(80));
    assert_eq!(ledger.snapshot().allocated, bytes(80));
    ledger
        .apply(
            &identity(),
            &[
                Event::Retain {
                    source: HandleId(4),
                    handle: HandleId(5),
                }, // retained view
                Event::Complete {
                    handle: HandleId(4),
                },
            ],
            &limits(),
        )
        .unwrap();
    assert_eq!(ledger.snapshot().current, bytes(80));
    assert_eq!(ledger.snapshot().completed_retained, bytes(80));
    ledger
        .apply(
            &identity(),
            &[Event::Release {
                handle: HandleId(4),
            }],
            &limits(),
        )
        .unwrap();
    assert_eq!(ledger.snapshot().current, bytes(80));
    assert_eq!(ledger.snapshot().completed_retained, bytes(0));
    ledger
        .apply(
            &identity(),
            &[Event::Release {
                handle: HandleId(5),
            }],
            &limits(),
        )
        .unwrap();
    assert_eq!(ledger.snapshot().current, bytes(0));
    assert_eq!(ledger.snapshot().released, bytes(80));
    assert_eq!(ledger.snapshot().peak, bytes(80));
}

#[test]
fn intermediate_breach_rolls_back_even_when_final_release_would_fit() {
    let mut ledger = Ownership::new(identity()).unwrap();
    ledger
        .apply(&identity(), &[allocation(1, 1, 40)], &limits())
        .unwrap();
    ledger
        .reserve(&identity(), ReservationId(1), bytes(60), &limits())
        .unwrap();
    let before = serde_json::to_value(ledger.snapshot()).unwrap();
    assert!(ledger
        .commit(
            &identity(),
            ReservationId(1),
            &[
                allocation(2, 2, 61),
                Event::Release {
                    handle: HandleId(1)
                },
            ],
            &limits()
        )
        .is_err());
    assert_eq!(serde_json::to_value(ledger.snapshot()).unwrap(), before);
    assert!(ledger
        .apply(
            &identity(),
            &[Event::Complete {
                handle: HandleId(99)
            }],
            &limits()
        )
        .is_err());
    assert_eq!(serde_json::to_value(ledger.snapshot()).unwrap(), before);
}

#[test]
fn growth_keeps_old_and_new_roots_and_reservations_do_not_double_count() {
    let mut ledger = Ownership::new(identity()).unwrap();
    ledger
        .apply(&identity(), &[allocation(1, 1, 40)], &limits())
        .unwrap();
    ledger
        .reserve(&identity(), ReservationId(1), bytes(60), &limits())
        .unwrap();
    assert_eq!(ledger.snapshot().current, bytes(100));
    ledger
        .commit(
            &identity(),
            ReservationId(1),
            &[
                allocation(2, 2, 60),
                Event::Release {
                    handle: HandleId(1),
                },
            ],
            &limits(),
        )
        .unwrap();
    assert_eq!(ledger.snapshot().current, bytes(60));
    assert_eq!(ledger.snapshot().peak, bytes(100));
    assert_eq!(ledger.snapshot().reserved, bytes(0));
}

#[test]
fn four_contending_transitions_cannot_each_acquire_the_permit() {
    let mut ledger = Ownership::new(identity()).unwrap();
    ledger
        .reserve(&identity(), ReservationId(0), bytes(10), &limits())
        .unwrap();
    for slot in 1..4 {
        assert!(ledger
            .reserve(&identity(), ReservationId(slot), bytes(10), &limits())
            .is_err());
    }
    assert_eq!(ledger.snapshot().current, bytes(10));
    ledger.cancel(&identity(), ReservationId(0)).unwrap();
    ledger
        .reserve(&identity(), ReservationId(1), bytes(10), &limits())
        .unwrap();
}

#[test]
fn failed_transition_is_atomic_and_does_not_publish_partial_roots() {
    let mut ledger = Ownership::new(identity()).unwrap();
    ledger
        .apply(&identity(), &[allocation(1, 1, 40)], &limits())
        .unwrap();
    ledger
        .reserve(&identity(), ReservationId(1), bytes(60), &limits())
        .unwrap();
    let before = serde_json::to_value(ledger.snapshot()).unwrap();
    assert!(ledger
        .commit(
            &identity(),
            ReservationId(1),
            &[allocation(2, 2, 61)],
            &limits()
        )
        .is_err());
    assert_eq!(serde_json::to_value(ledger.snapshot()).unwrap(), before);
    ledger.cancel(&identity(), ReservationId(1)).unwrap();
    assert_eq!(ledger.snapshot().current, bytes(40));
    assert!(ledger
        .apply(&identity(), &[allocation(1, 3, 1)], &limits())
        .is_err());
    assert_eq!(ledger.snapshot().current, bytes(40));
}

#[test]
fn aggregate_competitors_and_phase_peak_cannot_be_hidden_by_final_release() {
    let mut caps = limits();
    caps.competition = bound(4 * GIB);
    assert!(caps.project(bytes(115 * GIB)).is_ok());
    // Three individually fitting competitors must still be charged together.
    caps.competition = bound(3 * 4 * GIB);
    assert!(caps.project(bytes(115 * GIB)).is_err());
    let frozen = shape(Family::Gemma4);
    let mut inventory = empty_declared(frozen.clone());
    inventory
        .declare(
            CategoryKey {
                owner: Owner::Process,
                kind: Category::WeightResources,
            },
            Entry::Bound {
                bytes: bytes(40),
                provenance: "toy capacity".into(),
                formula_revision: "synthetic-v1".into(),
            },
        )
        .unwrap();
    let mut ledger = Ownership::new(identity()).unwrap();
    ledger
        .apply(
            &identity(),
            &[
                allocation(1, 1, 41),
                Event::Release {
                    handle: HandleId(1),
                },
            ],
            &limits(),
        )
        .unwrap();
    assert!(Receipt::declared(&inventory, &frozen, &limits(), &ledger).is_err());
}

#[test]
fn receipt_is_explicitly_declared_projection_and_never_runtime_qualified() {
    let frozen = shape(Family::Qwen38);
    let inventory = empty_declared(frozen.clone());
    let ownership = Ownership::new(identity()).unwrap();
    let receipt = Receipt::declared(&inventory, &frozen, &limits(), &ownership).unwrap();
    let json = serde_json::to_value(receipt).unwrap();
    assert_eq!(json["runtime_qualified"], false);
    assert_eq!(
        json["evidence_scope"],
        "caller_declared_model_free_projection_only"
    );
    assert_eq!(
        json["identity"]["artifact_sha256"],
        identity().artifact_sha256
    );
    assert!(json["limits"]["competition"]["Bound"]["provenance"].is_string());
}
