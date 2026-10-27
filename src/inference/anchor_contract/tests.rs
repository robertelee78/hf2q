use super::*;

fn identity() -> Identity {
    Identity {
        artifact: [1; 32],
        family_schema: "typed-fixture-v1".into(),
        tokenizer: [2; 32],
        template: [3; 32],
        configuration: [4; 32],
        conversation: 17,
        slot_generation: 2,
        lineage_generation: 1,
        graft_generation: 3,
        graft_digest: [5; 32],
        graft_len: 1,
    }
}
fn component(role: &str, fill: u8) -> Component {
    Component {
        role: role.into(),
        shape: vec![2, 2],
        element_bytes: 1,
        representation: "U8/plain-v1".into(),
        cursor: 2,
        capacity: 8,
        bytes: vec![fill; 4],
    }
}
fn anchor(id: u64) -> Anchor {
    Anchor {
        id,
        request: id,
        identity: identity(),
        logical_tokens: id as usize,
        prefix_digest: [id as u8; 32],
        components: vec![
            component("full", id as u8),
            component("sidecar", id as u8 + 20),
        ],
        owners: vec![Owner {
            id: 100 + id,
            extent: 8,
        }],
        required_live_rows: vec![LiveRows {
            owner: 1,
            start: 0,
            end: id as usize,
        }],
    }
}
fn store(k: usize, quota: usize) -> Store {
    Store::new(identity(), k, quota, vec![Owner { id: 1, extent: 64 }]).unwrap()
}
fn commit(s: &mut Store, a: Anchor) {
    let request = a.request;
    s.stage(a).unwrap();
    s.complete(request, true).unwrap();
}
fn can_restore(s: &Store, a: &Anchor) -> bool {
    s.preflight(
        a.id,
        &a.identity,
        a.logical_tokens,
        &a.prefix_digest,
        &a.components,
    )
    .is_ok()
}
fn restore(s: &mut Store, a: &Anchor) {
    let ticket = s
        .preflight(
            a.id,
            &a.identity,
            a.logical_tokens,
            &a.prefix_digest,
            &a.components,
        )
        .unwrap();
    s.apply(ticket, &mut Vec::new()).unwrap();
}

#[test]
fn anchor_c1_preflight_all_components_before_any_cursor_or_byte_mutation() {
    let mut s = store(4, 512);
    let mut bad = anchor(1);
    bad.components[1].shape = vec![usize::MAX, 2];
    commit(&mut s, bad);
    let wanted = anchor(1);
    let mut live = vec![component("full", 90), component("sidecar", 91)];
    live[0].cursor = 7;
    let before = live.clone();
    let store_before = format!("{s:?}");
    assert!(s
        .preflight(1, &identity(), 1, &[1; 32], &wanted.components)
        .is_err());
    assert_eq!(
        format!("{s:?}"),
        store_before,
        "late refusal preserves the source state, not merely a separate output vector"
    );
    assert_eq!(
        live, before,
        "late sidecar refusal must not rewind earlier cursors"
    );
    let mut valid = store(4, 512);
    commit(&mut valid, wanted.clone());
    let ticket = valid
        .preflight(1, &identity(), 1, &[1; 32], &wanted.components)
        .unwrap();
    valid.apply(ticket, &mut live).unwrap();
    assert_eq!(
        live, wanted.components,
        "all distinct payload bytes restored"
    );
}

#[test]
fn anchor_c1_identity_prefix_and_coordinate_boundaries_refuse_stale_state() {
    let mut s = store(4, 512);
    let a = anchor(1);
    commit(&mut s, a.clone());
    assert!(can_restore(&s, &a));
    for field in 0..11 {
        let mut changed = identity();
        match field {
            0 => changed.artifact[0] ^= 1,
            1 => changed.family_schema.push('x'),
            2 => changed.tokenizer[0] ^= 1,
            3 => changed.template[0] ^= 1,
            4 => changed.configuration[0] ^= 1,
            5 => changed.conversation += 1,
            6 => changed.slot_generation += 1,
            7 => changed.graft_generation += 1,
            8 => changed.graft_digest[0] ^= 1,
            9 => changed.graft_len += 1,
            _ => changed.lineage_generation += 1,
        }
        assert!(s
            .preflight(1, &changed, 1, &[1; 32], &a.components)
            .is_err());
    }
    assert!(s
        .preflight(1, &identity(), 2, &[1; 32], &a.components)
        .is_err());
    assert!(s
        .preflight(1, &identity(), 1, &[9; 32], &a.components)
        .is_err());
    let mut wrong_cursor = a.clone();
    wrong_cursor.id = 2;
    wrong_cursor.request = 2;
    wrong_cursor.components[0].cursor = 3;
    commit(&mut s, wrong_cursor);
    assert!(s
        .preflight(2, &identity(), 1, &[1; 32], &a.components)
        .is_err());
    assert_eq!(physical_cursor(0, 0, 0), Ok(0));
    assert_eq!(physical_cursor(129, 7, 8), Ok(128));
    assert!(physical_cursor(usize::MAX, 1, 0).is_err());
    assert!(physical_cursor(0, 0, 8).is_err());
    let mut wrong = a.components.clone();
    wrong[0].cursor = 9;
    let mut invalid = a.clone();
    invalid.components = wrong;
    invalid.id = 3;
    invalid.request = 3;
    commit(&mut s, invalid.clone());
    assert!(!can_restore(&s, &invalid));
}

#[test]
fn anchor_c1_rewind_divergent_write_tombstones_old_branch_before_submission() {
    let mut s = store(4, 512);
    for id in 1..=3 {
        commit(&mut s, anchor(id));
    }
    let old_c = anchor(3);
    restore(&mut s, &anchor(1));
    s.before_writes(
        1,
        &[LiveRows {
            owner: 1,
            start: 3,
            end: 4,
        }],
    )
    .unwrap();
    assert!(!can_restore(&s, &old_c));
    assert!(can_restore(&s, &anchor(1)));
    s.cancel(99, &[], true); // Later cancellation cannot resurrect B/C.
    assert!(!can_restore(&s, &anchor(2)));
    let mut reuse = anchor(3);
    reuse.request = 100;
    assert!(s.stage(reuse).is_err(), "a tombstoned ID cannot be reused");
    commit(&mut s, anchor(4));
    assert!(can_restore(&s, &anchor(4)));
    assert!(
        s.before_writes(1, &[]).is_err(),
        "write authorization is consumed once"
    );
    restore(&mut s, &anchor(1));
    s.before_writes(
        1,
        &[LiveRows {
            owner: 1,
            start: 0,
            end: 1,
        }],
    )
    .unwrap();
    assert!(!can_restore(&s, &anchor(4)));
    assert!(
        !can_restore(&s, &anchor(1)),
        "overwriting an ancestor's live rows destroys validity too"
    );
}

#[test]
fn anchor_c1_cancel_at_each_capture_boundary_never_publishes_pending() {
    for boundary in 0..3 {
        let mut s = store(4, 512);
        let a = anchor(1);
        if boundary >= 1 {
            s.begin_capture(
                a.clone(),
                &[Owner {
                    id: 700,
                    extent: 16,
                }],
            )
            .unwrap();
        }
        if boundary == 2 {
            s.finish_capture(true).unwrap();
        }
        s.cancel(1, &[], true);
        assert!(!can_restore(&s, &a));
        assert_eq!(s.reserved(), Ok(64));
        assert!(s.complete(1, true).is_err());
        assert!(
            s.stage(a).is_err(),
            "late re-capture must not reintroduce cancelled capture"
        );
    }
    let mut s = store(4, 512);
    commit(&mut s, anchor(1));
    assert!(s.complete(1, true).is_err());
    assert_eq!(s.committed.len(), 1);
}

#[test]
fn anchor_c1_partial_or_unknown_mutation_failure_forbids_shallower_fallback() {
    for known in [true, false] {
        let mut s = store(4, 512);
        commit(&mut s, anchor(1));
        commit(&mut s, anchor(2));
        s.begin_capture(
            anchor(3),
            &[Owner {
                id: 701,
                extent: 16,
            }],
        )
        .unwrap();
        let before = s.reserved().unwrap();
        s.partial_failure(known);
        assert!(s.poisoned);
        assert!(!can_restore(&s, &anchor(1)));
        assert!(s.stage(anchor(4)).is_err());
        assert_eq!(
            s.reserved(),
            Ok(if known { 64 } else { before }),
            "unknown retirement grants no credit"
        );
    }
}

#[test]
fn anchor_c1_k_committed_plus_pending_construction_and_async_reader_ownership() {
    let mut s = store(4, 128);
    for id in 1..=4 {
        commit(&mut s, anchor(id));
    }
    assert_eq!(s.committed.len(), 4);
    assert_eq!(s.reserved(), Ok(96));
    assert_eq!(
        s.begin_capture(
            anchor(5),
            &[Owner {
                id: 800,
                extent: 24
            }]
        ),
        Ok(128)
    );
    assert_eq!(s.reserved(), Ok(128));
    assert_eq!(s.committed.len(), 4);
    s.finish_capture(true).unwrap();
    assert_eq!(s.reserved(), Ok(104));
    s.retain_reader(900, 1).unwrap();
    s.complete(5, true).unwrap();
    assert_eq!(s.reserved(), Ok(104));
    s.finish_reader(900).unwrap();
    assert_eq!(s.reserved(), Ok(96));
    let before = format!("{s:?}");
    assert!(s
        .begin_capture(
            anchor(6),
            &[Owner {
                id: 801,
                extent: 25
            }]
        )
        .is_err());
    assert_eq!(format!("{s:?}"), before, "quota refusal is pre-mutation");
}

#[test]
fn anchor_c1_parent_aliases_and_right_sized_payloads_do_not_hide_retained_extents() {
    let alias = [
        Owner {
            id: 10,
            extent: 80 << 20,
        },
        Owner {
            id: 10,
            extent: 80 << 20,
        },
    ];
    assert_eq!(unique_bytes(&alias), Ok(80 << 20));
    assert!(unique_bytes([
        &alias[0],
        &Owner {
            id: 10,
            extent: 20 << 10
        }
    ])
    .is_err());
    // A tiny view into a large parent cannot fit a dedicated 20KiB payload grant.
    assert!(Store::new(identity(), 4, 20 << 10, vec![alias[0].clone()]).is_err());
    let mut s = store(4, 64 + (20 << 10));
    let mut pinned = anchor(1);
    pinned.owners = vec![alias[0].clone()];
    let before = format!("{s:?}");
    assert!(s.stage(pinned).is_err());
    assert_eq!(format!("{s:?}"), before);
    let mut missing = anchor(1);
    missing.owners.clear();
    assert!(s.stage(missing).is_err());
    assert_eq!(
        unique_bytes([
            &alias[0],
            &Owner {
                id: 11,
                extent: 80 << 20
            }
        ]),
        Ok(160 << 20),
        "independent equal-valued copies are separate owners"
    );
    assert!(unique_bytes([
        &Owner {
            id: 1,
            extent: usize::MAX
        },
        &Owner { id: 2, extent: 1 }
    ])
    .is_err());
}

#[test]
fn anchor_c1_four_conversations_graft_generation_and_idle_conservation() {
    let mut stores: Vec<_> = (0..4).map(|_| store(4, 512)).collect();
    for (i, s) in stores.iter_mut().enumerate() {
        s.identity.conversation += i as u64;
        let mut a = anchor(1);
        a.identity = s.identity.clone();
        commit(s, a);
    }
    for s in &stores {
        let mut a = anchor(1);
        a.identity = s.identity.clone();
        assert!(can_restore(s, &a));
    }
    for i in [3, 1, 0, 2] {
        let mut other = anchor(1);
        other.identity = stores[(i + 1) % 4].identity.clone();
        assert!(!can_restore(&stores[i], &other));
        let mut replaced = stores[i].identity.clone();
        replaced.graft_digest[0] ^= 1;
        replaced.graft_generation += 1;
        stores[i].replace_identity(replaced, true).unwrap();
        assert_eq!(stores[i].reserved(), Ok(64));
        assert!(stores[i].pending.is_none());
        assert!(stores[i].committed.is_empty());
    }
}

#[test]
fn anchor_c1_stale_tickets_and_open_transactions_never_apply() {
    for change in 0..3 {
        let mut s = store(4, 512);
        let a = anchor(1);
        commit(&mut s, a.clone());
        let ticket = s
            .preflight(1, &identity(), 1, &[1; 32], &a.components)
            .unwrap();
        match change {
            0 => s.cancel(9, &[], true),
            1 => s.partial_failure(true),
            _ => {
                let mut id = identity();
                id.slot_generation += 1;
                s.replace_identity(id, true).unwrap();
            }
        }
        let mut live = vec![component("untouched", 99)];
        let before = live.clone();
        assert!(s.apply(ticket, &mut live).is_err());
        assert_eq!(live, before);
    }
    let mut s = store(4, 512);
    let a = anchor(1);
    commit(&mut s, a.clone());
    s.begin_capture(anchor(2), &[]).unwrap();
    assert!(!can_restore(&s, &a));
    s.finish_capture(true).unwrap();
    assert!(!can_restore(&s, &a));
    assert!(
        s.retain_reader(2, 2).is_err(),
        "pending payload cannot be promoted to disk"
    );
}

#[test]
fn anchor_c1_pending_overlap_cancelled_rows_and_quarantine_are_independent() {
    let mut s = store(4, 512);
    commit(&mut s, anchor(1));
    restore(&mut s, &anchor(1));
    s.stage(anchor(2)).unwrap();
    s.before_writes(
        1,
        &[LiveRows {
            owner: 1,
            start: 1,
            end: 3,
        }],
    )
    .unwrap();
    assert!(s.complete(2, true).is_err());
    assert!(!can_restore(&s, &anchor(2)));
    assert!(can_restore(&s, &anchor(1)));
    s.cancel(
        3,
        &[LiveRows {
            owner: 1,
            start: 0,
            end: 1,
        }],
        true,
    );
    assert!(!can_restore(&s, &anchor(1)));
    for failure in 0..3 {
        let mut q = store(4, 512);
        commit(&mut q, anchor(1));
        q.begin_capture(
            anchor(2),
            &[Owner {
                id: 700,
                extent: 16,
            }],
        )
        .unwrap();
        let before = q.reserved();
        match failure {
            0 => q.cancel(2, &[], false),
            1 => {
                assert!(q.finish_capture(false).is_err());
            }
            _ => {
                let mut id = identity();
                id.slot_generation += 1;
                assert!(q.replace_identity(id, false).is_err());
            }
        }
        assert_eq!(q.reserved(), before);
        assert!(q.finish_reader(u64::MAX).is_err());
        assert_eq!(q.reserved(), before);
        assert!(q.retire_quarantine(false).is_err());
        let mut id = identity();
        id.slot_generation += 1;
        assert!(q.reset(id.clone()).is_err());
        q.retire_quarantine(true).unwrap();
        q.reset(id).unwrap();
        assert_eq!(q.reserved(), Ok(64));
    }
}

#[test]
fn anchor_c1_cross_store_tickets_and_invalid_physical_dependencies_refuse() {
    let mut a = store(4, 512);
    let mut b = store(4, 512);
    let image = anchor(1);
    commit(&mut a, image.clone());
    commit(&mut b, image.clone());
    let ticket = a
        .preflight(1, &identity(), 1, &[1; 32], &image.components)
        .unwrap();
    let mut live = vec![component("unchanged", 77)];
    let before = live.clone();
    assert!(b.apply(ticket, &mut live).is_err());
    assert_eq!(live, before);
    assert!(can_restore(&a, &image));
    assert!(can_restore(&b, &image));
    for range in [
        LiveRows {
            owner: 999,
            start: 0,
            end: 1,
        },
        LiveRows {
            owner: 1,
            start: 1,
            end: 1,
        },
        LiveRows {
            owner: 1,
            start: 0,
            end: 65,
        },
    ] {
        let mut bad = anchor(2);
        bad.required_live_rows = vec![range];
        let before = format!("{a:?}");
        assert!(a.stage(bad).is_err());
        assert_eq!(format!("{a:?}"), before);
    }
}

#[test]
fn anchor_c1_last_component_dtype_presence_and_reader_namespace_are_checked() {
    let mut s = store(4, 512);
    let a = anchor(1);
    commit(&mut s, a.clone());
    assert!(can_restore(&s, &a));
    for defect in 0..4 {
        let mut required = a.components.clone();
        match defect {
            0 => {
                required.pop();
            }
            1 => required[1].element_bytes = 2,
            2 => required[1].role = "wrong-codec-sidecar".into(),
            _ => required[1].representation = "U8/other-codec".into(),
        }
        let before = format!("{s:?}");
        assert!(s.preflight(1, &identity(), 1, &[1; 32], &required).is_err());
        assert_eq!(format!("{s:?}"), before);
    }
    s.retain_reader(u64::MAX, 1).unwrap();
    s.begin_capture(
        anchor(2),
        &[Owner {
            id: 700,
            extent: 16,
        }],
    )
    .unwrap();
    s.partial_failure(false);
    let quarantined = s.reserved();
    s.finish_reader(u64::MAX).unwrap();
    assert_eq!(
        s.reserved(),
        quarantined,
        "ordinary reader release cannot release unknown GPU owners"
    );
}

#[test]
fn anchor_c1_generation_advance_never_allows_another_generation_to_regress() {
    let mut s = store(4, 512);
    commit(&mut s, anchor(1));
    let mut id = identity();
    id.slot_generation += 1;
    id.lineage_generation -= 1;
    let before = format!("{s:?}");
    assert!(s.replace_identity(id, true).is_err());
    assert_eq!(format!("{s:?}"), before);
    assert!(can_restore(&s, &anchor(1)));
}
