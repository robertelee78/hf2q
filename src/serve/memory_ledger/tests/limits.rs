use super::*;

#[test]
fn host_family_and_metal_caps_are_independent() {
    let mut caps = limits();
    caps.family = bound(100 * GIB);
    caps.competition = bound(10 * GIB);
    // Competition reduces the 126 GiB host bound to 116, not the 100 GiB family cap to 90.
    assert!(caps.project(bytes(100 * GIB)).is_ok());
    assert!(matches!(
        caps.project(bytes(101 * GIB)),
        Err(Error::Limit("family process ceiling"))
    ));
    caps.family = Ceiling::NoneWithInventory {
        provenance: "complete synthetic inventory".into(),
    };
    assert!(caps.project(bytes(116 * GIB)).is_ok());
    assert!(caps.project(bytes(116 * GIB + 1)).is_err());
    caps.metal_policy = MetalPolicy::Enforced {
        bytes: 90 * GIB,
        provenance: "chosen synthetic constraint".into(),
    };
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
fn unknown_ceilings_competition_and_reaction_fail_closed() {
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
    caps.metal_policy = MetalPolicy::Unknown {
        reason: "selection not inventoried".into(),
    };
    assert!(caps.project(bytes(1)).is_err());
    caps = limits();
    caps.reaction.inflight_overshoot_bytes = 0;
    assert!(caps.project(bytes(1)).is_err());
}

#[test]
fn real_deepseek_known_resource_lower_bound_exceeds_observed_policy() {
    let mut caps = limits();
    caps.family = bound(116 * GIB);
    caps.metal_policy = MetalPolicy::Enforced {
        bytes: 115_448_725_504,
        provenance:
            "historical deliberately selected recommendation constraint, not hardware limit".into(),
    };
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

#[test]
fn current_policy_has_no_fixed_reserve_or_deepseek_ceiling() {
    let mut caps = limits();
    caps.require_family(Family::Deepseek4).unwrap();
    let p = caps.project(bytes(120 * GIB)).unwrap();
    assert_eq!(p.host_cap_after_competition, 126 * GIB);
    assert_eq!(p.reaction_bytes, 2 * GIB);
    assert_eq!(p.family_cap, None);
    caps.metal_policy = MetalPolicy::RecommendationOnly {
        bytes: GIB,
        provenance: "declared device recommendation, not chosen cap".into(),
    };
    assert!(caps.project(bytes(120 * GIB)).is_ok());
    caps.metal_policy = MetalPolicy::Enforced {
        bytes: GIB,
        provenance: "explicit selected experimental policy".into(),
    };
    assert!(caps.project(bytes(120 * GIB)).is_err());
}

#[test]
fn reaction_source_and_checked_available_arithmetic_are_required() {
    let mut caps = limits();
    caps.reaction.provenance = " ".into();
    assert!(matches!(caps.project(bytes(1)), Err(Error::Unknown(_))));
    caps = limits();
    caps.reaction.maximum_stop_latency_ns = 0;
    assert!(caps.project(bytes(1)).is_err());
    caps = limits();
    caps.reaction.observation_age_bytes = u64::MAX;
    assert!(matches!(caps.project(bytes(1)), Err(Error::Overflow)));
    caps = limits();
    caps.competition = bound(127 * GIB);
    assert!(caps.project(bytes(0)).is_err());
    // No cached-files subtraction field exists; unrecognized refund rejects.
    let mut wire = serde_json::to_value(limits()).unwrap();
    wire["cached_files_credit"] = serde_json::json!(GIB);
    assert!(serde_json::from_value::<Limits>(wire).is_err());
}
