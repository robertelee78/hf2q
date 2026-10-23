use super::*;

#[test]
fn host_family_and_metal_caps_are_independent() {
    let mut caps = limits();
    caps.family = bound(100 * GIB);
    caps.competition = bound(10 * GIB);
    // Competition reduces the 116 GiB host bound to 106, not the 100 GiB family cap to 90.
    assert!(caps.project(bytes(100 * GIB)).is_ok());
    assert!(matches!(
        caps.project(bytes(101 * GIB)),
        Err(Error::Limit("family process ceiling"))
    ));
    caps.family = Ceiling::NoneWithInventory {
        provenance: "complete synthetic inventory".into(),
    };
    assert!(caps.project(bytes(106 * GIB)).is_ok());
    assert!(caps.project(bytes(106 * GIB + 1)).is_err());
    caps.working_set = bound(90 * GIB);
    assert!(caps
        .project(DeclaredBytes {
            owned: 100 * GIB,
            metal: 80 * GIB
        })
        .is_ok());
    assert!(matches!(
        caps.project(bytes(100 * GIB)),
        Err(Error::Limit("declared Metal working-set policy"))
    ));
}

#[test]
fn unknown_ceilings_competition_and_reserve_fail_closed() {
    let mut caps = limits();
    caps.family = Ceiling::Unknown {
        reason: "inventory not inspected".into(),
    };
    assert!(matches!(caps.project(bytes(1)), Err(Error::Unknown(_))));
    caps = limits();
    caps.competition = Ceiling::Unknown {
        reason: "372 inaccessible external processes".into(),
    };
    assert!(caps.project(bytes(1)).is_err());
    caps = limits();
    caps.working_set = Ceiling::NoneWithInventory {
        provenance: "not optional".into(),
    };
    assert!(caps.project(bytes(1)).is_err());
    caps = limits();
    caps.reserve -= 1;
    assert!(caps.project(bytes(1)).is_err());
}

#[test]
fn real_deepseek_known_resource_lower_bound_exceeds_observed_policy() {
    let mut caps = limits();
    caps.family = bound(116 * GIB);
    caps.working_set = bound(115_448_725_504);
    // Encoded weights + four full caches + retained eager boot cache, no scratch/anchors.
    let known = bytes(107_426_004_892 + 8_205_189_120);
    assert!(matches!(
        caps.project(known),
        Err(Error::Limit("declared Metal working-set policy"))
    ));
}

#[test]
fn checked_arithmetic_rejects_overflow_and_invalid_metal_subset() {
    assert_eq!(bytes(u64::MAX).checked_add(bytes(1)), Err(Error::Overflow));
    assert!(DeclaredBytes { owned: 1, metal: 2 }.validate().is_err());
    assert!(matches!(
        limits().project(bytes(u64::MAX)),
        Err(Error::Overflow)
    ));
    let mut inventory = empty_declared(shape(Family::Gemma4));
    for kind in [Category::WeightResources, Category::ExpandedWeights] {
        inventory
            .declare(
                CategoryKey {
                    owner: Owner::Process,
                    kind,
                },
                Entry::Bound {
                    bytes: bytes(u64::MAX),
                    provenance: "overflow control".into(),
                    formula_revision: "synthetic-v1".into(),
                },
            )
            .unwrap();
    }
    assert_eq!(inventory.total(), Err(Error::Overflow));
}
