//! Actual lifecycle/offset boundary controls, isolated from numeric fixtures.
use super::*;
use crate::adr063_invariance::{capture, family, qwen};
#[test]
fn q1_family_tls_scope_cleans_up_on_panic() {
    let root = Root::new();
    let coordinate = at(Stage::CacheState);
    let store = Store::new(
        &root.0,
        identity(),
        "scope".into(),
        vec![ExpectedTensor {
            at: coordinate.clone(),
            scalar: Scalar::U8,
            shape: vec![1],
        }],
        1,
        1,
    )
    .unwrap();
    let mut targets = BTreeMap::new();
    targets.insert(
        (0, 0, Stage::CacheState),
        CaptureTarget {
            at: coordinate,
            normalize_rows: false,
        },
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = family::install(Family::Qwen38, store, targets).unwrap();
        assert!(family::enabled(Family::Qwen38));
        panic!("deliberate capture unwind");
    }));
    assert!(result.is_err());
    assert!(!family::enabled(Family::Qwen38));
    assert!(family::finish().is_err());
}

#[test]
#[ignore = "root-controlled tiny Shared CPU allocation; no GPU dispatch, one 32-byte root"]
fn q1_actual_offset_view_capture_control() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = mlx_native::MlxDevice::new().unwrap();
    let mut buffer = device
        .alloc_buffer(32, mlx_native::DType::F32, vec![8])
        .unwrap();
    buffer
        .as_logical_mut_slice::<f32>()
        .unwrap()
        .copy_from_slice(&[11., 12., 21., 22., 23., 24., 31., 32.]);
    let view = buffer.slice_view(8, 4).with_shape(vec![4]).unwrap();
    let root = Root::new();
    let coordinate = at(Stage::CacheKeyNorm);
    let mut store = Store::new(
        &root.0,
        identity(),
        "offset-control".into(),
        vec![ExpectedTensor {
            at: coordinate.clone(),
            scalar: Scalar::F32,
            shape: vec![4],
        }],
        16,
        1,
    )
    .unwrap();
    capture::after_checked_drain(
        &mut store,
        coordinate,
        &view,
        &[4],
        None,
        "CPU initialized; no GPU work submitted",
    )
    .unwrap();
    let trace = store.finish().unwrap();
    let tensor = &trace.coverage.captured[0];
    assert_eq!(tensor.parent_bytes, 32);
    assert_eq!(tensor.parent_offset, 8);
    assert_eq!(
        tensor.parent_identity,
        buffer.contents_ptr() as usize as u64
    );
    let captured = std::fs::read(root.0.join(&tensor.file)).unwrap();
    assert_eq!(
        captured,
        bytemuck::cast_slice::<f32, u8>(&[21., 22., 23., 24.])
    );
    assert_ne!(
        captured,
        bytemuck::cast_slice::<f32, u8>(&[11., 12., 21., 22.])
    );
    eprintln!("Q1_ACTUAL_OFFSET_VIEW_CAPTURE_COMPLETE root_bytes=32 captured_bytes=16 offset=8 gpu_dispatches=0");
}

#[test]
fn q1_same_acquisition_root_is_not_invariance_evidence() {
    let root = Root::new();
    let trace = trace(&root, Stage::CacheState, Scalar::U8, &[1, 2, 3]);
    let result = compare(&root.0, &trace, &root.0, &trace, &BTreeMap::new());
    assert_eq!(result.disposition, Disposition::InvalidEvidence);
    assert!(!result.oracle_qualified);
    assert!(!result.captured_extent_qualified);
}

#[test]
fn q1_qwen_tls_scope_cleans_up_on_panic() {
    let root = Root::new();
    let coordinate = at(Stage::CacheState);
    let store = Store::new(
        &root.0,
        identity(),
        "qwen-scope".into(),
        vec![ExpectedTensor {
            at: coordinate.clone(),
            scalar: Scalar::U8,
            shape: vec![1],
        }],
        1,
        1,
    )
    .unwrap();
    let mut targets = BTreeMap::new();
    targets.insert(
        (0, Some(0), "layer_out".into()),
        CaptureTarget {
            at: coordinate,
            normalize_rows: false,
        },
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = qwen::install(store, targets).unwrap();
        assert!(qwen::enabled());
        panic!("deliberate Qwen capture unwind");
    }));
    assert!(result.is_err());
    assert!(!qwen::enabled());
    assert!(qwen::finish().is_err());
}
