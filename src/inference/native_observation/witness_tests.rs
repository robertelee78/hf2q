//! Witness manifest, padding, claim and end-marker controls (no GPU).
use super::*;
#[test]
fn actual_witness_footprint_requires_all_rows_and_checked_view_extent() {
    assert_eq!(footprint(4, 4, 8, 32, 128, 64, 192, 4), Some(128));
    assert_eq!(footprint(4, 1, 8, 8, 32, 0, 32, 4), None);
    assert_eq!(footprint(4, 4, 8, 32, 127, 64, 192, 4), None);
    assert_eq!(footprint(4, 4, 8, 32, 128, 65, 192, 4), None);
    assert_eq!(footprint(4, 4, 8, 31, 128, 64, 192, 4), None);
    assert_eq!(footprint(4, 4, usize::MAX, 0, 0, 0, 0, 4), None);
    assert_eq!(footprint(4, 4, 8, 32, 128, u64::MAX, 192, 4), None);
}
#[test]
fn actual_source_layout_does_not_relabel_scalar_or_temporal_as_independent_quad() {
    let scalar = [InputSpan {
        slot: 0,
        start: 3,
        count: 1,
    }];
    assert!(source_layout("qwen35.greedy.actual_tokens", &scalar));
    let temporal = [InputSpan {
        slot: 0,
        start: 3,
        count: 4,
    }];
    assert!(!source_layout("qwen35.greedy.actual_tokens", &temporal));
    assert!(source_layout("qwen35.forward.actual_tokens", &temporal));
    assert!(!source_layout(
        "gemma4.decode.actual_token_slot_position_arrays",
        &temporal
    ));
    let independent = std::array::from_fn::<_, 4, _>(|slot| InputSpan {
        slot: slot as u32,
        start: 3,
        count: 1,
    });
    assert!(source_layout(
        "gemma4.decode.actual_token_slot_position_arrays",
        &independent
    ));
    assert!(!source_layout("qwen35.forward.actual_tokens", &independent));
    assert!(!source_layout(
        "gemma4.decode.actual_token_slot_position_arrays",
        &[scalar[0], scalar[0]]
    ));
}

const OTHER_SOURCES: [&str; 11] = [
    "qwen35.greedy.actual_tokens",
    "qwen35.forward.actual_tokens",
    "gemma4.scalar_forward",
    "gemma4.decode.actual_token_slot_position_arrays",
    "gemma4.prefill.actual_concat_descriptor",
    "gemma4.prefill.actual_single_input",
    "deepseek4.scalar_verifier",
    "deepseek4.prefill.actual_cache_span",
    "deepseek4.sorted_cache_lane",
    "deepseek4.prefill.actual_cooperative_lane",
    "deepseek4.prefill.actual_cooperative_pack",
];

/// One fed one-token row per slot `0..n`, at slot-specific positions.
fn lanes(n: usize) -> Vec<InputSpan> {
    (0..n)
        .map(|slot| InputSpan {
            slot: slot as u32,
            start: 100 + 10 * slot as u64,
            count: 1,
        })
        .collect()
}

fn decode_owners(n: u32) -> Vec<crate::serve::phase_trace::SlotState> {
    (0..n)
        .map(|slot| crate::serve::phase_trace::SlotState {
            request_id: 50 + u64::from(slot),
            slot_id: slot,
            generation: 7,
            phase: "decode",
            active: true,
            forward_eligibility: Some(true),
            runnable_forward: None,
            runnable_emit: None,
            inflight: None,
            blocked: None,
        })
        .collect()
}

#[test]
fn joint_source_layout_is_one_to_eight_unique_one_token_rows() {
    for rows in 1..=MAX_JOINT_DECODE_ROWS {
        assert!(source_layout(JOINT_SOURCE, &lanes(rows)), "{rows} rows");
    }
    assert!(!source_layout(JOINT_SOURCE, &[]));
    assert!(!source_layout(
        JOINT_SOURCE,
        &lanes(MAX_JOINT_DECODE_ROWS + 1)
    ));
    let mut duplicate = lanes(3);
    duplicate[2].slot = 0;
    assert!(!source_layout(JOINT_SOURCE, &duplicate));
    let mut temporal = lanes(2);
    temporal[1].count = 2;
    assert!(!source_layout(JOINT_SOURCE, &temporal));
}

#[test]
fn padding_rows_exist_only_for_joint_target_decode_rows() {
    let owners = decode_owners(4);
    let (rows, total) =
        manifest(JOINT_SOURCE, &lanes(3), 1, "target", &owners, &owners).expect("joint 3 + 1");
    assert_eq!(total, 4);
    assert_eq!(
        rows.iter()
            .map(|row| (row.request_id, row.slot_id, row.token_position, row.phase))
            .collect::<Vec<_>>(),
        [
            (50, 0, 100, "ordinary"),
            (51, 1, 110, "ordinary"),
            (52, 2, 120, "ordinary")
        ]
    );
    for source in OTHER_SOURCES {
        let spans = (1..=2)
            .map(lanes)
            .find(|spans| source_layout(source, spans))
            .expect("a layout the source accepts");
        assert!(
            manifest(source, &spans, 0, "target", &owners, &owners).is_some(),
            "{source} without padding"
        );
        assert!(
            manifest(source, &spans, 1, "target", &owners, &owners).is_none(),
            "{source} cannot declare padding"
        );
    }
    // A joint verify or a prefill owner is refused, never relabelled.
    assert!(manifest(
        JOINT_SOURCE,
        &lanes(3),
        1,
        "target_verify",
        &owners,
        &owners
    )
    .is_none());
    let mut prefill = owners.clone();
    prefill[1].phase = "prefill";
    assert!(manifest(JOINT_SOURCE, &lanes(3), 1, "target", &prefill, &prefill).is_none());
    // The padding trap: the RMS matrix is the physical [4, h] buffer, so
    // only the physical total passes `footprint`; the logical 3 does not.
    assert_eq!(footprint(total, 4, 8, 32, 128, 0, 128, 4), Some(128));
    assert_eq!(footprint(3, 4, 8, 32, 128, 0, 128, 4), None);
}

#[test]
fn unarmed_joint_input_is_inactive_and_never_polls_its_rows() {
    assert!(
        !ARMED.load(Ordering::Acquire),
        "unit tests never arm the observer"
    );
    take_hook_counts();
    let rows = std::iter::from_fn(|| -> Option<(u32, i32)> {
        panic!("unarmed joint manifest polled its rows")
    });
    let guard = joint_decode_input(rows, 4);
    assert!(!guard.active && guard.end.is_none() && guard.previous.is_none());
    INPUT.with(|value| assert!(value.borrow().is_none()));
    drop(guard);
    assert_eq!(take_hook_counts(), [0; 4]);
}

#[test]
fn joint_manifest_resolves_owners_selected_on_this_thread_only() {
    let owners = decode_owners(3);
    let previous = SELECTED.with(|value| value.replace(owners.clone()));
    let _invocation = Invocation {
        previous: Some(previous),
        previous_epoch: EPOCH.with(|value| value.replace(41)),
        _thread_bound: PhantomData,
    };
    let here = |live: &[crate::serve::phase_trace::SlotState]| {
        SELECTED.with(|selected| {
            manifest(
                JOINT_SOURCE,
                &lanes(3),
                1,
                "target",
                &selected.borrow(),
                live,
            )
        })
    };
    let (rows, _) = here(&owners).expect("same-thread invocation owners");
    assert_eq!(
        rows.iter()
            .map(|row| (row.request_id, row.slot_id, row.generation))
            .collect::<Vec<_>>(),
        [(50, 0, 7), (51, 1, 7), (52, 2, 7)]
    );
    let live = owners.clone();
    let elsewhere = std::thread::spawn(move || {
        SELECTED.with(|selected| {
            manifest(
                JOINT_SOURCE,
                &lanes(3),
                1,
                "target",
                &selected.borrow(),
                &live,
            )
            .is_none()
        })
    });
    assert!(
        elsewhere.join().unwrap(),
        "another thread has no selected owners"
    );
    let mut recycled = owners;
    recycled[1].generation += 1;
    assert!(here(&recycled).is_none(), "recycled generation refused");
}

#[test]
fn claimed_descriptor_silences_every_later_witness_call() {
    assert!(matches!(claim(None), Claim::Unmapped));
    let descriptor = Arc::new(Descriptor {
        id: 9,
        source: JOINT_SOURCE,
        rows: Vec::new(),
        total: 4,
        role: "target",
        witness_claimed: AtomicBool::new(false),
    });
    assert!(matches!(claim(Some(descriptor.clone())), Claim::Witness(_)));
    // The other 47 DeltaNet pre-norms of the forward publish nothing.
    for _ in 0..47 {
        assert!(matches!(claim(Some(descriptor.clone())), Claim::Silent));
    }
}

fn guard(active: bool, joint: bool) -> Input {
    Input {
        previous: None,
        active,
        end: joint.then_some(EndMarker {
            forward_id: 9,
            epoch: 8,
        }),
        _thread_bound: PhantomData,
    }
}

fn markers() -> u32 {
    take_hook_counts()[Hook::Marker as usize]
}

#[test]
fn joint_guard_closes_its_forward_exactly_once_on_every_exit() {
    take_hook_counts();
    drop(guard(true, true));
    assert_eq!(markers(), 1);
    let mut taken = guard(true, true);
    assert_eq!(
        taken.take_end(),
        Some(EndMarker {
            forward_id: 9,
            epoch: 8
        })
    );
    assert_eq!(taken.take_end(), None);
    drop(taken);
    assert_eq!(markers(), 0);
    fn encode(fail: bool) -> Result<(), ()> {
        let _input = guard(true, true);
        if fail {
            return Err(());
        }
        Ok(())
    }
    assert!(encode(true).is_err());
    assert_eq!(markers(), 1, "early return");
    assert!(encode(false).is_ok());
    assert_eq!(markers(), 1);
    let unwound = std::panic::catch_unwind(|| {
        let _input = guard(true, true);
        panic!("joint encode failed");
    });
    assert!(unwound.is_err());
    assert_eq!(markers(), 1, "unwinding");
    drop(guard(false, true));
    drop(guard(true, false));
    assert_eq!(markers(), 0, "inactive or non-joint guard");
}
