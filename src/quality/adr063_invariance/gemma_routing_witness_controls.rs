//! Host-only actual admission/row-join controls; no device or producer execution.
use super::*;
#[test]
fn finite_case_budget_covers_t5_and_cancelled_extra_namespaces() {
    assert_eq!(PAYLOAD_BYTES, 2_782_080);
    assert_eq!(ENTRY_CAP, 33_120);
    assert!(4 * (144 + 64) * LAYERS <= ENTRY_CAP);
    assert!(4 * (128 + 132 + 16) * LAYERS <= ENTRY_CAP);
    assert!(PAYLOAD_BYTES + METADATA_BYTES + CPU_LEDGER_BYTES < 96 * 1024 * 1024);
    assert!(validate_budget(None).is_ok());
    assert!(validate_budget(Some(&Budget {
        payload_bytes: PAYLOAD_BYTES - 1,
        metadata_bytes: METADATA_BYTES,
        cpu_ledger_bytes: CPU_LEDGER_BYTES
    }))
    .is_err());
    let value = json!({"payload_bytes":PAYLOAD_BYTES,"metadata_bytes":METADATA_BYTES,"cpu_ledger_bytes":CPU_LEDGER_BYTES,"fallback":true});
    assert!(serde_json::from_value::<Budget>(value).is_err());
}
#[test]
fn witness_native_pin_matches_the_built_native_so_exact_budgets_admit() {
    // The build stamp is the Cargo.toml exact pin (`build_native_pin` control).
    assert_eq!(WITNESS_NATIVE_VERSION, env!("HF2Q_MLX_NATIVE_VERSION"));
    validate_budget(Some(&Budget {
        payload_bytes: PAYLOAD_BYTES,
        metadata_bytes: METADATA_BYTES,
        cpu_ledger_bytes: CPU_LEDGER_BYTES,
    }))
    .unwrap();
}
#[test]
fn actual_row_map_refuses_duplicate_wrong_slot_position_and_capacity() {
    let row = Row {
        lane: 2,
        position: 130,
    };
    rows_ok(&[row]).unwrap();
    assert!(rows_ok(&[row, row]).is_err());
    assert!(rows_ok(&[Row {
        lane: 4,
        position: 130
    }])
    .is_err());
    assert!(rows_ok(&[Row {
        lane: 2,
        position: 144
    }])
    .is_err());
    let mut call = Call {
        id: 1,
        namespace: Namespace::T5Verify,
        rows: vec![row],
        seen: BTreeSet::new(),
    };
    let mut global = BTreeSet::new();
    check_rows(&call, &global, 0, 29, &[row], 0).unwrap();
    assert!(check_rows(&call, &global, 0, 30, &[row], 0).is_err());
    assert!(check_rows(
        &call,
        &global,
        0,
        29,
        &[Row {
            lane: 1,
            position: 130
        }],
        0
    )
    .is_err());
    assert!(check_rows(
        &call,
        &global,
        0,
        29,
        &[Row {
            lane: 2,
            position: 129
        }],
        0
    )
    .is_err());
    assert!(check_rows(&call, &global, ENTRY_CAP, 29, &[row], 0).is_err());
    call.seen.insert((29, 0));
    assert!(check_rows(&call, &global, 0, 29, &[row], 0).is_err());
    call.seen.clear();
    global.insert((Namespace::T5Verify, 2, 130, 29));
    assert!(check_rows(&call, &global, 0, 29, &[row], 0).is_err());
    // A different invocation namespace is retained separately, never overwrites.
    call.namespace = Namespace::T5Continuation;
    check_rows(&call, &global, 0, 29, &[row], 0).unwrap();
}
#[test]
fn packed_physical_rows_are_not_temporal_or_fabricated_slot_four() {
    let rows = (0..4)
        .map(|lane| Row {
            lane,
            position: 128,
        })
        .collect::<Vec<_>>();
    rows_ok(&rows).unwrap();
    let call = Call {
        id: 2,
        namespace: Namespace::Canonical,
        rows: rows.clone(),
        seen: BTreeSet::new(),
    };
    for i in 0..4 {
        check_rows(&call, &BTreeSet::new(), 0, 0, &rows[i..i + 1], i).unwrap();
    }
    assert!(check_rows(
        &call,
        &BTreeSet::new(),
        0,
        0,
        &[Row {
            lane: 0,
            position: 129
        }],
        1
    )
    .is_err());
    assert!(check_rows(&call, &BTreeSet::new(), 0, 0, &rows, 1).is_err());
}
#[test]
fn exact_namespace_spans_and_raw_selection_validation_are_actual_used_guards() {
    let rows = |start, end| {
        (start..end)
            .map(|position| Row { lane: 1, position })
            .collect::<Vec<_>>()
    };
    namespace_rows(Mode::Transaction, Namespace::T5Verify, &rows(0, 132)).unwrap();
    assert!(namespace_rows(Mode::Transaction, Namespace::T5Verify, &rows(0, 128)).is_err());
    assert!(namespace_rows(Mode::Ordinary, Namespace::T5Verify, &rows(0, 132)).is_err());
    namespace_rows(
        Mode::Cancelled,
        Namespace::UnpublishedCancelled,
        &rows(64, 128),
    )
    .unwrap();
    assert!(namespace_rows(
        Mode::Cancelled,
        Namespace::UnpublishedCancelled,
        &rows(0, 64)
    )
    .is_err());
    let ids = [0, 1, 2, 3, 4, 5, 6, 7];
    let mut weights = [0.1; 8];
    let boundary = [7, 8];
    assert!(output::selected_valid(
        &ids,
        &weights,
        &boundary,
        &[0.2, 0.1]
    ));
    assert!(!output::selected_valid(
        &ids,
        &weights,
        &[6, 8],
        &[0.2, 0.1]
    ));
    assert!(!output::selected_valid(
        &ids,
        &weights,
        &[7, 6],
        &[0.2, 0.1]
    ));
    weights[1] = f32::NAN;
    assert!(!output::selected_valid(
        &ids,
        &weights,
        &boundary,
        &[0.2, 0.1]
    ));
}

#[test]
fn actual_native_boundary_projection_preserves_all_domains_without_native_serde() {
    let scored=output::project_boundary(&SelectionBoundary::Scored{kth_id:7,kth_score:0.25,next_id:9,next_score:0.125});
    assert_eq!(scored,json!({"kind":"Scored","kth_id":7,"kth_score_f32_bits":0.25f32.to_bits(),"next_id":9,"next_score_f32_bits":0.125f32.to_bits()}));
    assert_eq!(output::project_boundary(&SelectionBoundary::AllExpertsSelected{kth_id:1,kth_score:f32::NAN}),json!({"kind":"AllExpertsSelected","kth_id":1,"kth_score_f32_bits":f32::NAN.to_bits()}));
    assert_eq!(output::project_boundary(&SelectionBoundary::Invalid),json!({"kind":"Invalid"}));
}
