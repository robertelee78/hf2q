//! Actual planner/CPU-anchor layout evidence, not a proof of production restore atomicity.
use super::*;
use crate::inference::models::qwen35::kv_cache::HybridKvSlotAnchor;

#[test]
fn anchor_c1_qwen_dense_moe_slot_axes_mtp_and_owned_anchor_payload() {
    for dense in [true, false] {
        let mut cfg = header_config(dense);
        // Keep the supported artifact's head/channel geometry; four legal layers
        // bound the CPU fixture, with one FULL and three recurrent layers.
        cfg.num_hidden_layers = 4;
        cfg.layer_types = default_layer_types(4, 4);
        let p = HybridCacheAllocationPlan::new(&cfg, 64, 4, true, true).unwrap();
        assert_eq!(p.slots.len(), 5);
        assert_eq!(p.slots.iter().filter(|s| s.mtp).count(), 1);
        let mut conv = Vec::new();
        let mut recurrent = Vec::new();
        let mut full = Vec::new();
        let mut owned = 0;
        for (i, slot) in p.slots.iter().enumerate() {
            match slot.slot {
                LayerSlot::Full(_) => {
                    assert_eq!(slot.roots.len(), 4, "packed K/V both need norm sidecars");
                    assert!(slot.roots.iter().all(|r| r.shape[0] == 4));
                    assert_eq!(slot.roots[1].dtype, DType::F32);
                    assert_eq!(slot.roots[3].dtype, DType::F32);
                    if !slot.mtp {
                        full.push(8);
                    }
                }
                LayerSlot::Linear(_) => {
                    assert_eq!(slot.roots.len(), 4);
                    assert_eq!(slot.roots[0].label, "alloc conv_state");
                    assert_eq!(slot.roots[2].label, "alloc recurrent");
                    assert_eq!(slot.roots[0].shape.last(), Some(&4));
                    assert_eq!(slot.roots[2].shape.last(), Some(&4));
                    let c = slot.roots[0].bytes / 4;
                    let r = slot.roots[2].bytes / 4;
                    owned += c + r;
                    conv.push(vec![i as u8 + 1; c]);
                    recurrent.push(vec![i as u8 + 21; r]);
                }
            }
        }
        let a = HybridKvSlotAnchor {
            prompt_len: 8,
            full_attn_current_len: full,
            mtp_current_len: Some(8),
            linear_conv: conv,
            linear_recurrent: recurrent,
        };
        assert_eq!(a.prompt_len(), 8);
        assert_eq!(a.total_bytes(), owned);
        assert_eq!(a.linear_conv.len(), 3);
        assert_eq!(a.linear_recurrent.len(), 3);
        assert!(
            owned < 16 * 1024 * 1024,
            "bounded actual-geometry CPU fixture"
        );
        assert!(a
            .linear_conv
            .iter()
            .zip(&a.linear_recurrent)
            .all(|(c, r)| c[0] != r[0]));
        // This constructs the actual CPU type; it does not execute snapshot/restore.
        // FULL live KV is a validity dependency, not a duplicated anchor payload.
        assert!(a.total_bytes() < p.total_bytes);
    }
}

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
