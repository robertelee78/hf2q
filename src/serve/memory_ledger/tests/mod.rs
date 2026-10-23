use super::*;

const GIB: u64 = 1 << 30;

fn identity() -> FrozenIdentity {
    FrozenIdentity {
        artifact_sha256: "a".repeat(64),
        workload_sha256: "b".repeat(64),
        transition_plan_sha256: "c".repeat(64),
    }
}

fn shape(family: Family) -> FrozenShape {
    FrozenShape {
        identity: identity(),
        family,
        slots: 4,
        context_per_slot: 262_144,
        result_cap: 8192,
        dispatch_threads: 2,
        vision: false,
        drafter: false,
        graft: false,
        persistence: false,
    }
}

fn bound(bytes: u64) -> Ceiling {
    Ceiling::Bound {
        bytes,
        provenance: "synthetic declared control".into(),
    }
}

fn limits() -> Limits {
    Limits {
        physical: bound(128 * GIB),
        reserve: 12 * GIB,
        competition: bound(0),
        family: Ceiling::NoneWithInventory {
            provenance: "synthetic no extra ceiling".into(),
        },
        working_set: bound(116 * GIB),
    }
}

fn bytes(value: u64) -> DeclaredBytes {
    DeclaredBytes {
        owned: value,
        metal: value,
    }
}

fn allocation(root: u64, handle: u64, size: u64) -> Event {
    Event::Allocate {
        root: RootId(root),
        handle: HandleId(handle),
        bytes: bytes(size),
    }
}

fn empty_declared(shape: FrozenShape) -> Inventory {
    let entries = shape
        .required_keys()
        .unwrap()
        .into_iter()
        .map(|key| {
            (
                key,
                Entry::ProvenZero {
                    reason: "synthetic event test without model".into(),
                    configuration_evidence: "synthetic fixture configuration".into(),
                },
            )
        })
        .collect();
    Inventory::from_entries(shape, entries).unwrap()
}

#[test]
fn family_schema_contains_boot_slots_anchor_roles_and_all_thread_pools() {
    for family in [
        Family::Qwen38,
        Family::Qwen36,
        Family::Gemma4,
        Family::Deepseek4,
    ] {
        let keys = shape(family).required_keys().unwrap();
        for slot in 0..4 {
            for kind in [
                Category::LiveCache,
                Category::CommittedAnchor,
                Category::PendingAnchor,
                Category::RollbackAnchor,
                Category::AnchorStore,
                Category::LiveLogits,
            ] {
                assert!(keys.contains(&CategoryKey {
                    owner: Owner::Slot(slot),
                    kind
                }));
            }
        }
        assert_eq!(
            keys.contains(&CategoryKey {
                owner: Owner::Boot,
                kind: Category::LiveCache
            }),
            family == Family::Deepseek4
        );
        let pool = match family {
            Family::Deepseek4 => Category::SubmissionPool,
            Family::Gemma4 => Category::DirectScratch,
            _ => Category::WeightPool,
        };
        for thread in 0..2 {
            assert!(keys.contains(&CategoryKey {
                owner: Owner::Thread(thread),
                kind: pool
            }));
        }
        for kind in [
            Category::Projector,
            Category::VisionBuffers,
            Category::GraftBank,
            Category::PersistBusy,
            Category::PersistPending,
            Category::PersistConstructing,
            Category::PersistHydration,
            Category::Drafter,
            Category::QueuedResources,
        ] {
            assert!(keys.contains(&CategoryKey {
                owner: Owner::Process,
                kind
            }));
        }
    }
}

#[test]
fn unknown_real_packet_rejects_for_each_family() {
    for family in [
        Family::Qwen38,
        Family::Qwen36,
        Family::Gemma4,
        Family::Deepseek4,
    ] {
        let frozen = shape(family);
        let inventory = Inventory::unknown(frozen.clone()).unwrap();
        let ownership = Ownership::new(identity()).unwrap();
        assert!(matches!(
            Receipt::declared(&inventory, &frozen, &limits(), &ownership),
            Err(Error::Unknown(_))
        ));
    }
}

#[test]
fn deepseek_cannot_omit_or_raise_existing_family_ceiling() {
    let frozen = shape(Family::Deepseek4);
    let inventory = empty_declared(frozen.clone());
    let ownership = Ownership::new(identity()).unwrap();
    let mut caps = limits();
    assert!(Receipt::declared(&inventory, &frozen, &caps, &ownership).is_err());
    caps.family = bound(117 * GIB);
    assert!(Receipt::declared(&inventory, &frozen, &caps, &ownership).is_err());
    caps.family = bound(116 * GIB);
    assert!(Receipt::declared(&inventory, &frozen, &caps, &ownership).is_ok());
}

#[test]
fn anchor_and_persistor_overlap_are_distinct_charges() {
    let mut inventory = empty_declared(shape(Family::Gemma4));
    for kind in [
        Category::CommittedAnchor,
        Category::PendingAnchor,
        Category::RollbackAnchor,
    ] {
        inventory
            .declare(
                CategoryKey {
                    owner: Owner::Slot(0),
                    kind,
                },
                Entry::Bound {
                    bytes: DeclaredBytes {
                        owned: 30,
                        metal: 0,
                    },
                    provenance: "three independent host copies".into(),
                    formula_revision: "synthetic-v1".into(),
                },
            )
            .unwrap();
    }
    for kind in [
        Category::PersistBusy,
        Category::PersistPending,
        Category::PersistConstructing,
        Category::PersistHydration,
    ] {
        inventory
            .declare(
                CategoryKey {
                    owner: Owner::Process,
                    kind,
                },
                Entry::Bound {
                    bytes: DeclaredBytes {
                        owned: 100,
                        metal: 0,
                    },
                    provenance: "independent bounded host image".into(),
                    formula_revision: "synthetic-v1".into(),
                },
            )
            .unwrap();
    }
    assert_eq!(
        inventory.total().unwrap(),
        DeclaredBytes {
            owned: 490,
            metal: 0
        }
    );
}

#[test]
fn import_cannot_drop_duplicate_or_invent_categories() {
    let frozen = shape(Family::Deepseek4);
    let rows = empty_declared(frozen.clone()).entries();
    let mut missing = rows.clone();
    missing.retain(|(key, _)| key.owner != Owner::Boot);
    assert!(Inventory::from_entries(frozen.clone(), missing).is_err());
    let mut duplicated = rows.clone();
    duplicated.push(rows[0].clone());
    assert!(Inventory::from_entries(frozen.clone(), duplicated).is_err());
    let mut extra = rows;
    extra.push((
        CategoryKey {
            owner: Owner::Thread(9),
            kind: Category::DecodePool,
        },
        Entry::ProvenZero {
            reason: "not in topology".into(),
            configuration_evidence: "synthetic fixture configuration".into(),
        },
    ));
    assert!(Inventory::from_entries(frozen, extra).is_err());
}

#[test]
fn unknown_scratch_and_missing_provenance_are_not_zero() {
    let mut inventory = empty_declared(shape(Family::Deepseek4));
    let key = CategoryKey {
        owner: Owner::Thread(0),
        kind: Category::PrefillPool,
    };
    inventory
        .declare(
            key,
            Entry::Unknown {
                reason: "scratch bound pending".into(),
            },
        )
        .unwrap();
    assert!(matches!(inventory.total(), Err(Error::Unknown(_))));
    inventory
        .declare(
            key,
            Entry::Bound {
                bytes: bytes(10),
                provenance: " ".into(),
                formula_revision: "synthetic-v1".into(),
            },
        )
        .unwrap();
    assert!(matches!(inventory.total(), Err(Error::Category(_))));
    inventory
        .declare(
            key,
            Entry::Bound {
                bytes: bytes(10),
                provenance: "declared input".into(),
                formula_revision: " ".into(),
            },
        )
        .unwrap();
    assert!(matches!(inventory.total(), Err(Error::Category(_))));
    inventory
        .declare(
            key,
            Entry::ProvenZero {
                reason: "disabled".into(),
                configuration_evidence: " ".into(),
            },
        )
        .unwrap();
    assert!(matches!(inventory.total(), Err(Error::Category(_))));
}

#[test]
fn wire_entries_reject_unknown_fields_and_missing_evidence() {
    assert!(serde_json::from_str::<CategoryKey>(
        r#"{"owner":"Process","kind":"LiveCache","omitted_state":true}"#
    )
    .is_err());
    assert!(serde_json::from_str::<Ceiling>(
        r#"{"Bound":{"bytes":1,"provenance":"toy","verified":true}}"#
    )
    .is_err());
    assert!(serde_json::from_str::<Event>(r#"{"Complete":{"handle":1,"release":true}}"#).is_err());
    assert!(serde_json::from_str::<Entry>(
        r#"{"Bound":{"bytes":{"owned":1,"metal":0},"provenance":"toy"}}"#
    )
    .is_err());
    assert!(serde_json::from_str::<Entry>(r#"{"ProvenZero":{"reason":"disabled"}}"#).is_err());
}

#[test]
fn frozen_workload_and_transition_cannot_silently_shrink() {
    let frozen = shape(Family::Gemma4);
    let mut changed = frozen.clone();
    changed.context_per_slot /= 2;
    assert!(empty_declared(changed).require_shape(&frozen).is_err());
    changed = frozen.clone();
    changed.result_cap /= 2;
    assert!(empty_declared(changed).require_shape(&frozen).is_err());
    changed = frozen.clone();
    changed.dispatch_threads = 1;
    assert!(empty_declared(changed).require_shape(&frozen).is_err());
    let mut changed_id = identity();
    changed_id.transition_plan_sha256 = "d".repeat(64);
    let mut ownership = Ownership::new(identity()).unwrap();
    assert!(ownership
        .apply(&changed_id, &[allocation(1, 1, 10)], &limits())
        .is_err());
    assert_eq!(ownership.snapshot().current, DeclaredBytes::ZERO);
    changed = frozen;
    changed.slots = 3;
    assert!(Inventory::unknown(changed).is_err());
}

mod limits;
mod ownership;
