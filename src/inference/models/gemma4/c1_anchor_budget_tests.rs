//! Real Hybrid descriptors, CPU anchor and reservation Drop; no device or model load.
use super::super::{
    cache_scaffold_plan::{Modes, Phase, SlotPlan},
    GemmaHybridSlotAnchor,
};
use super::*;
use mlx_native::DType;

#[test]
fn anchor_c1_gemma_ring_sidecars_xlen_and_real_last_reader_quota() {
    for xlen in [false, true] {
        let modes = Modes {
            hybrid: true,
            dense: false,
            nibble: false,
            dense_dtype: DType::F32,
            full_f16_v: false,
            xlen,
        };
        let ring = SlotPlan::new(Phase::Hybrid, 0, 8, 256, 8, true, 4, modes, None).unwrap();
        let global = SlotPlan::new(Phase::Hybrid, 5, 2, 512, 64, false, 4, modes, None).unwrap();
        assert!(ring.ring);
        assert!(!global.ring);
        assert_eq!(ring.buffers.len(), if xlen { 5 } else { 3 });
        assert_eq!(ring.buffers[0].dtype, DType::F16);
        assert_eq!(ring.buffers[1].dtype, DType::U8);
        assert_eq!(ring.buffers[2].dtype, DType::F32);
        assert!(ring
            .buffers
            .iter()
            .all(|b| b.shape[0] == 4 && b.bytes % 4 == 0));
        let payload = ring.buffers.iter().map(|b| b.bytes / 4).sum::<usize>();
        let q = Arc::new(Quota {
            used: AtomicUsize::new(0),
            limit: payload,
        });
        let reservation = acquire(&q, payload).unwrap();
        let layer = GemmaHybridSlidingLayerAnchor {
            k: vec![11; ring.buffers[0].bytes / 4],
            v_packed: vec![22; ring.buffers[1].bytes / 4],
            v_norms: Some(vec![33; ring.buffers[2].bytes / 4]),
            bf16_xlen_k: xlen.then(|| vec![44; ring.buffers[3].bytes / 4]),
            bf16_xlen_v: xlen.then(|| vec![55; ring.buffers[4].bytes / 4]),
        };
        let anchor = Arc::new(GemmaHybridSlotAnchor {
            prompt_len: 8,
            layers: vec![Some(layer), None],
            _reservation: Some(reservation),
        });
        assert_eq!(anchor.total_bytes(), payload);
        assert!(
            acquire(&q, 1).is_err(),
            "a full local test quota cannot be overdrawn"
        );
        let disk_reader = Arc::clone(&anchor);
        drop(anchor);
        assert_eq!(q.used.load(Ordering::Acquire), payload);
        assert_eq!(disk_reader.layers[0].as_ref().unwrap().k[0], 11);
        drop(disk_reader);
        assert_eq!(q.used.load(Ordering::Acquire), 0);
        // This exercises actual Reservation/Drop with a local payload-only quota.
        // Production reserve() metadata/token sizing and OS residency are not tested.
        assert!(payload < 256 * 1024);
    }
}

#[test]
fn anchor_c1_gemma_full_f16_v_and_norm_dummy_are_not_tq_sidecar_payloads() {
    let p = SlotPlan::new(
        Phase::Hybrid,
        5,
        2,
        512,
        8,
        false,
        4,
        Modes {
            hybrid: true,
            dense: false,
            nibble: false,
            dense_dtype: DType::F32,
            full_f16_v: true,
            xlen: false,
        },
        None,
    )
    .unwrap();
    assert_eq!(p.buffers.len(), 3);
    assert_eq!(p.buffers[0].dtype, DType::F16);
    assert_eq!(p.buffers[1].dtype, DType::F16);
    assert_eq!(p.buffers[2].shape, [1]);
    assert_eq!(
        p.buffers[2].bytes, 4,
        "dummy norm is one shared scalar, not a per-slot TQ copy"
    );
    assert!(SlotPlan::new(
        Phase::Hybrid,
        0,
        8,
        256,
        8,
        true,
        0,
        Modes {
            hybrid: true,
            dense: false,
            nibble: false,
            dense_dtype: DType::F32,
            full_f16_v: true,
            xlen: false
        },
        None
    )
    .is_err());
}
