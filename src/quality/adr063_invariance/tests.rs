//! Host-only controls of the actual store/comparison boundary; no Metal device.
use super::{compare::compare, receipt::*, store::Store};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
fn identity() -> Identity {
    Identity {
        family: Family::Qwen38,
        source_commit: "a".repeat(40),
        binary_sha256: "b".repeat(64),
        native_version: env!("HF2Q_MLX_NATIVE_VERSION").into(),
        native_checksum: "c".repeat(64),
        model_sha256: "d".repeat(64),
        tokenizer_sha256: "e".repeat(64),
        template_sha256: "f".repeat(64),
        config_sha256: "1".repeat(64),
        token_vectors_sha256: "2".repeat(64),
        device_registry_id: 1,
        queue_identity: 2,
        model_generation: 3,
    }
}
fn at(stage: Stage) -> Coordinate {
    Coordinate {
        logical_step: 0,
        layer: 0,
        stage,
        agent: "lane-0".into(),
        logical_row_start: 0,
        segment: 0,
    }
}
struct Root(PathBuf);
impl Root {
    fn new() -> Self {
        let parent = std::env::temp_dir().canonicalize().unwrap();
        Self(parent.join(format!(
            "hf2q-q1-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Root {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn trace(root: &Root, stage: Stage, scalar: Scalar, bytes: &[u8]) -> Trace {
    let mut s = Store::new(
        &root.0,
        identity(),
        "T0".into(),
        vec![ExpectedTensor {
            at: at(stage),
            scalar,
            shape: vec![bytes.len() / scalar.bytes()],
        }],
        bytes.len() as u64,
        1,
    )
    .unwrap();
    s.capture(
        at(stage),
        scalar,
        vec![bytes.len() / scalar.bytes()],
        1,
        bytes.len() as u64,
        0,
        None,
        "actual caller checked drain",
        bytes,
    )
    .unwrap();
    s.finish().unwrap()
}
#[test]
fn q1_first_difference_beyond_first_page_and_last_row_is_located() {
    let a = Root::new();
    let b = Root::new();
    let x = vec![0u8; super::store::PAGE_BYTES + 16];
    let mut y = x.clone();
    let last = y.len() - 1;
    y[last] = 7;
    let l = trace(&a, Stage::CacheState, Scalar::U8, &x);
    let r = trace(&b, Stage::CacheState, Scalar::U8, &y);
    let result = compare(&a.0, &l, &b.0, &r, &BTreeMap::new());
    assert_eq!(result.disposition, Disposition::DivergedExactInvariant);
    assert_eq!(
        result.first_difference.unwrap().logical_element,
        last as u64
    );
    assert_eq!(result.compared_bytes, x.len() as u64);
}
#[test]
fn q1_equal_float_without_frozen_policy_is_inconclusive() {
    let a = Root::new();
    let b = Root::new();
    let x = 1f32.to_le_bytes();
    let l = trace(&a, Stage::Logits, Scalar::F32, &x);
    let r = trace(&b, Stage::Logits, Scalar::F32, &x);
    let result = compare(&a.0, &l, &b.0, &r, &BTreeMap::new());
    assert_eq!(
        result.disposition,
        Disposition::ExplicitInconclusiveNumericPolicy
    );
    assert!(!result.captured_extent_qualified);
}
#[test]
fn q1_routing_id_swap_cannot_pass_under_float_tolerance() {
    let a = Root::new();
    let b = Root::new();
    let l = trace(&a, Stage::RouterIds, Scalar::U32, &[1, 0, 0, 0, 2, 0, 0, 0]);
    let r = trace(&b, Stage::RouterIds, Scalar::U32, &[2, 0, 0, 0, 1, 0, 0, 0]);
    let policy = BTreeMap::from([(
        Stage::RouterIds,
        NumericPolicy::SourceQualified {
            max_abs: 1e9,
            relative_rms: 1e9,
            source: "negative control".into(),
        },
    )]);
    assert_eq!(
        compare(&a.0, &l, &b.0, &r, &policy).disposition,
        Disposition::DivergedExactInvariant
    );
}
#[test]
fn q1_corrupt_payload_or_missing_coverage_is_not_numeric_divergence() {
    let a = Root::new();
    let b = Root::new();
    let l = trace(&a, Stage::CacheState, Scalar::U8, &[1]);
    let mut r = trace(&b, Stage::CacheState, Scalar::U8, &[1]);
    std::fs::write(b.0.join(&r.coverage.captured[0].file), [2]).unwrap();
    assert_eq!(
        compare(&a.0, &l, &b.0, &r, &BTreeMap::new()).disposition,
        Disposition::InvalidEvidence
    );
    r.coverage.captured.clear();
    assert_eq!(
        compare(&a.0, &l, &b.0, &r, &BTreeMap::new()).disposition,
        Disposition::InvalidEvidence
    );
}
#[test]
fn q1_store_refuses_padding_overflow_duplicate_and_unchecked_capture() {
    let root = Root::new();
    let mut s = Store::new(
        &root.0,
        identity(),
        "T0".into(),
        vec![ExpectedTensor {
            at: at(Stage::Input),
            scalar: Scalar::F32,
            shape: vec![1],
        }],
        4,
        1,
    )
    .unwrap();
    assert!(s
        .capture(
            at(Stage::Input),
            Scalar::F32,
            vec![1],
            1,
            4,
            0,
            None,
            "",
            &[0; 4]
        )
        .is_err());
    assert!(s
        .capture(
            at(Stage::Input),
            Scalar::F32,
            vec![1],
            1,
            4,
            1,
            None,
            "drained",
            &[0; 4]
        )
        .is_err());
    s.capture(
        at(Stage::Input),
        Scalar::F32,
        vec![1],
        1,
        4,
        0,
        None,
        "drained",
        &[0; 4],
    )
    .unwrap();
    assert!(s
        .capture(
            at(Stage::Input),
            Scalar::F32,
            vec![1],
            1,
            4,
            0,
            None,
            "drained",
            &[0; 4]
        )
        .is_err());
}
#[test]
fn q1_nine_trajectories_do_not_count_shared_baseline_as_new_execution() {
    use super::protocol::Trajectory;
    let labels: std::collections::BTreeSet<_> = Trajectory::ALL.iter().map(|t| t.label()).collect();
    assert_eq!(labels.len(), 9);
    assert_eq!(9 * 4 * 144, 5184);
}
#[test]
fn q1_cohort_and_speculation_need_actual_execution_evidence() {
    use super::protocol::{ExecutionEvidence, Trajectory};
    let mut evidence = ExecutionEvidence {
        trajectory: Trajectory::T2ActualCohort,
        actual_route: "four serial calls, not batched".into(),
        positions_per_lane: vec![144; 4],
        checked_terminal_calls: 16,
        restored_at: None,
        cancelled_at: None,
        accepted_draft_rows: None,
        rejected_draft_rows: None,
        physical_cohort_rows: None,
        missing_adapter_capabilities: vec![],
    };
    assert!(evidence.validate().is_err());
    evidence.trajectory = Trajectory::T5GenuineSpeculation;
    evidence.accepted_draft_rows = Some(8);
    evidence.rejected_draft_rows = Some(0);
    assert!(evidence.validate().is_err());
    evidence.rejected_draft_rows = Some(1);
    assert!(evidence.validate().is_ok());
}
#[test]
fn q1_last_row_slice_cannot_satisfy_full_expected_matrix() {
    let root = Root::new();
    let mut s = Store::new(
        &root.0,
        identity(),
        "T0".into(),
        vec![ExpectedTensor {
            at: at(Stage::LayerOutput),
            scalar: Scalar::F32,
            shape: vec![128, 4],
        }],
        2048,
        1,
    )
    .unwrap();
    assert!(s
        .capture(
            at(Stage::LayerOutput),
            Scalar::F32,
            vec![1, 4],
            1,
            2048,
            0,
            None,
            "drained",
            &[0; 16]
        )
        .is_err());
    assert!(!s.finish().unwrap().coverage.errors.is_empty());
}
#[test]
fn q1_source_named_score_sentinel_is_exact_but_nan_or_logit_infinity_is_invalid() {
    let a = Root::new();
    let b = Root::new();
    let bytes = f32::NEG_INFINITY.to_le_bytes();
    let l = trace(&a, Stage::CacheMainScore, Scalar::F32, &bytes);
    let r = trace(&b, Stage::CacheMainScore, Scalar::F32, &bytes);
    let policy = BTreeMap::from([(Stage::CacheMainScore, NumericPolicy::Exact)]);
    assert_eq!(
        compare(&a.0, &l, &b.0, &r, &policy).disposition,
        Disposition::ExactEqual
    );
    let c = Root::new();
    let d = Root::new();
    let l = trace(&c, Stage::Logits, Scalar::F32, &bytes);
    let r = trace(&d, Stage::Logits, Scalar::F32, &bytes);
    assert_eq!(
        compare(
            &c.0,
            &l,
            &d.0,
            &r,
            &BTreeMap::from([(Stage::Logits, NumericPolicy::Exact)])
        )
        .disposition,
        Disposition::InvalidEvidence
    );
}
#[test]
fn q1_first_difference_uses_frozen_token_layer_order_not_file_or_map_order() {
    let left = Root::new();
    let right = Root::new();
    let mut first = at(Stage::Norm);
    first.layer = 1;
    first.logical_row_start = 0;
    let mut later = at(Stage::Norm);
    later.layer = 0;
    later.logical_row_start = 1;
    let expected = vec![
        ExpectedTensor {
            at: first.clone(),
            scalar: Scalar::U8,
            shape: vec![1],
        },
        ExpectedTensor {
            at: later.clone(),
            scalar: Scalar::U8,
            shape: vec![1],
        },
    ];
    let build = |root: &Root, value: u8| {
        let mut store =
            Store::new(&root.0, identity(), "T0".into(), expected.clone(), 2, 2).unwrap();
        for coordinate in [&later, &first] {
            store
                .capture(
                    coordinate.clone(),
                    Scalar::U8,
                    vec![1],
                    1,
                    1,
                    0,
                    None,
                    "drained",
                    &[value],
                )
                .unwrap();
        }
        store.finish().unwrap()
    };
    let l = build(&left, 0);
    let r = build(&right, 1);
    let result = compare(&left.0, &l, &right.0, &r, &BTreeMap::new());
    assert_eq!(result.first_difference.unwrap().at, first);
}
#[test]
fn q1_float_bounds_are_predeclared_and_never_learned_from_candidate() {
    let a = Root::new();
    let b = Root::new();
    let l = trace(&a, Stage::Logits, Scalar::F32, &1f32.to_le_bytes());
    let r = trace(&b, Stage::Logits, Scalar::F32, &2f32.to_le_bytes());
    let policy = BTreeMap::from([(
        Stage::Logits,
        NumericPolicy::SourceQualified {
            max_abs: 0.1,
            relative_rms: 0.1,
            source: "fixed primitive fixture policy".into(),
        },
    )]);
    assert_eq!(
        compare(&a.0, &l, &b.0, &r, &policy).disposition,
        Disposition::DivergedSourceQualifiedPolicy
    );
    let invalid = BTreeMap::from([(
        Stage::Logits,
        NumericPolicy::SourceQualified {
            max_abs: f64::NAN,
            relative_rms: 0.1,
            source: "invalid".into(),
        },
    )]);
    assert_eq!(
        compare(&a.0, &l, &b.0, &r, &invalid).disposition,
        Disposition::InvalidEvidence
    );
}
