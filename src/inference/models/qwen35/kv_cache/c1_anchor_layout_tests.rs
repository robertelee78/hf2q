//! Allocation-axis evidence for the C1 contract.
use super::*;

#[test]
fn anchor_c1_qwen_distinct_sequence_and_capture_axes_are_not_head_axes() {
    for dense in [true, false] {
        let mut cfg = header_config(dense);
        cfg.num_hidden_layers = 4;
        cfg.layer_types = default_layer_types(4, 4);
        let plan = HybridCacheAllocationPlan::new(&cfg, 64, 3, true, true).unwrap();
        for slot in &plan.slots {
            match slot.slot {
                LayerSlot::Full(_) => assert!(slot.roots.iter().all(|r| r.shape[0] == 3)),
                LayerSlot::Linear(_) => {
                    assert_eq!(slot.roots[0].shape.last(), Some(&3));
                    assert_eq!(slot.roots[2].shape.last(), Some(&3));
                }
            }
        }
        let captures = CaptureAllocationPlan::new(&cfg, 3, 7, &[[None, None]; 3]).unwrap();
        for a in captures.actions {
            let root = a.allocation.unwrap();
            match a.role {
                CaptureRole::Recurrent => assert_eq!(&root.shape[root.shape.len() - 2..], &[7, 3]),
                CaptureRole::Conv => assert_eq!(&root.shape[..2], &[3, 7]),
            }
        }
    }
}

#[test]
fn anchor_c1_qwen_capture_growth_charges_old_parents_and_new_roots() {
    for dense in [true, false] {
        let mut cfg = header_config(dense);
        cfg.num_hidden_layers = 4;
        cfg.layer_types = default_layer_types(4, 4);
        let old = [[Some(ExistingCapture {
            elements: 1,
            backing_bytes: 80 << 20,
        }); 2]; 3];
        let plan = CaptureAllocationPlan::new(&cfg, 4, 2, &old).unwrap();
        assert_eq!(plan.old_backing_bytes, 6 * (80 << 20));
        assert!(plan.new_bytes > 0);
        assert_eq!(
            plan.old_plus_new_bytes,
            plan.old_backing_bytes + plan.new_bytes
        );
        for a in &plan.actions {
            let root = a.allocation.as_ref().unwrap();
            match a.role {
                CaptureRole::Recurrent => assert_eq!(&root.shape[root.shape.len() - 2..], &[2, 4]),
                CaptureRole::Conv => assert_eq!(&root.shape[..2], &[4, 2]),
            }
        }
        let without_mtp = HybridCacheAllocationPlan::new(&cfg, 64, 4, true, false).unwrap();
        assert!(without_mtp.slots.iter().all(|s| !s.mtp));
        assert!(HybridCacheAllocationPlan::new(&cfg, 64, 0, true, true).is_err());
    }
}
