use super::*;
fn receipt(t: Trajectory) -> Receipt {
    Receipt {
        identity: Identity { family: Family::Qwen38, source_commit: "a".repeat(40),
            binary_sha256: "b".repeat(64), native_version: env!("HF2Q_MLX_NATIVE_VERSION").into(),
            native_checksum: "c".repeat(64), model_sha256: "d".repeat(64),
            tokenizer_sha256: "e".repeat(64), template_sha256: "f".repeat(64),
            config_sha256: "1".repeat(64), token_vectors_sha256: "2".repeat(64) },
        trajectory: t, adapter: cell(Family::Qwen38, t).adapter,
        execution_receipt_sha256: "3".repeat(64), trace_sha256: vec!["4".repeat(64)],
        reference: if t == Trajectory::T5GenuineSpeculation {
            Reference::SeparateCoherentOrdinary { dataset_sha256: "2".repeat(64),
                receipt_sha256: "5".repeat(64), additional_logical_positions: 576 }
        } else { Reference::ColdT0 { dataset_sha256: "2".repeat(64) } },
        actual_logical_lanes: 4, actual_logit_rows_per_lane: 80,
        full_active_extent_captured: true,
        backend_finding: Disposition::ExplicitInconclusiveNumericPolicy,
        missing_capabilities: vec![],
    }
}
#[test]
fn catalogue_is_exactly_four_families_times_nine_without_false_unsupported() {
    let cells = catalogue();
    assert_eq!(cells.len(), 36);
    for family in [Family::Qwen38, Family::Deepseek4, Family::Qwen36, Family::Gemma4] {
        assert_eq!(cells.iter().filter(|c| c.family == family).count(), 9);
    }
    assert_eq!(cells.iter().filter(|c| c.adapter == Adapter::SourceRefusal).count(), 2);
    assert_eq!(cell(Family::Deepseek4, Trajectory::T5GenuineSpeculation).adapter, Adapter::Missing);
    assert_eq!(cell(Family::Deepseek4, Trajectory::T6CancelRestore).adapter, Adapter::Missing);
    assert_eq!(cells.iter().filter(|c| c.family==Family::Gemma4 && c.adapter==Adapter::GemmaOrdinary).count(),6);
    assert_eq!(cell(Family::Gemma4,Trajectory::T2ActualCohort).adapter,Adapter::GemmaPackedCohort);
    assert_eq!(cell(Family::Gemma4,Trajectory::T6CancelRestore).adapter,Adapter::GemmaOwningWorkerRecovery);
    let t5 = cell(Family::Gemma4, Trajectory::T5GenuineSpeculation);
    assert_eq!(t5.adapter, Adapter::GemmaNgramTransaction);
    assert_eq!(t5.scope, "gemma_ngram global OptionC injected K3 transaction vs same-route ordinary arm");
    assert_eq!(t5.gap, "not the slot-aware serving body");
    assert!(cells.iter().all(|c| c.family != Family::Gemma4 || c.adapter != Adapter::Missing));
}
#[test]
fn actual_generic_entry_guard_refuses_before_forward_and_names_real_separate_routes() {
    for family in [Family::Qwen38, Family::Qwen36] {
        assert!(ordinary_preflight(family, Trajectory::T0ColdSerial).is_ok());
        assert!(ordinary_preflight(family, Trajectory::T2ActualCohort).unwrap_err().starts_with("SourceCapabilityRefusal"));
        for t in [Trajectory::T5GenuineSpeculation, Trajectory::T6CancelRestore] {
            assert!(ordinary_preflight(family, t).unwrap_err().starts_with("SeparateActualAdapterRequired"));
        }
    }
    assert!(ordinary_preflight(Family::Deepseek4, Trajectory::T5GenuineSpeculation)
        .unwrap_err().starts_with("LocalizationIncomplete"));
    let gemma = ordinary_preflight(Family::Gemma4, Trajectory::T5GenuineSpeculation).unwrap_err();
    assert!(gemma.starts_with("SeparateActualAdapterRequired: gemma_ngram"), "{gemma}");
    // The slot-aware ordinary endpoint still refuses T5 itself.
    assert!(crate::adr063_invariance::gemma_actual::require_ordinary(Trajectory::T5GenuineSpeculation).is_err());
}
#[test]
fn scoped_continuation_and_one_lane_localizers_never_become_full_matrix_extent() {
    let mut r = receipt(Trajectory::T6CancelRestore);
    assert!(!r.full_matrix_cell_extent().unwrap());
    r.actual_logit_rows_per_lane = 144;
    assert!(r.full_matrix_cell_extent().unwrap());
    r.actual_logical_lanes = 1;
    assert!(!r.full_matrix_cell_extent().unwrap());
    r.actual_logical_lanes = 4;
    r.missing_capabilities.push("actual kth+1 selection-domain witness".into());
    assert!(!r.full_matrix_cell_extent().unwrap());
    assert_eq!(r.backend_finding, Disposition::ExplicitInconclusiveNumericPolicy);
}
#[test]
fn separate_t5_reference_rejects_t0_relabelling_dataset_mix_and_uncharged_positions() {
    let mut r = receipt(Trajectory::T5GenuineSpeculation);
    assert!(r.validate().is_ok());
    r.reference = Reference::ColdT0 { dataset_sha256: "2".repeat(64) };
    assert!(r.validate().is_err());
    r.reference = Reference::SeparateCoherentOrdinary { dataset_sha256: "6".repeat(64),
        receipt_sha256: "5".repeat(64), additional_logical_positions: 576 };
    assert!(r.validate().is_err());
    r.reference = Reference::SeparateCoherentOrdinary { dataset_sha256: "2".repeat(64),
        receipt_sha256: "5".repeat(64), additional_logical_positions: 0 };
    assert!(r.validate().is_err());
}
