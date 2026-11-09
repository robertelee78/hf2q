//! Issue #727: a spilled slot restores bitwise into any slot index, peers
//! stay untouched, and mismatched geometry is refused before any write.

use mlx_native::{MlxBuffer, MlxDevice};

use super::super::HybridKvCache;
use crate::inference::models::qwen35::{default_layer_types, Qwen35Config, Qwen35Variant};
use crate::serve::multi_seq_kv::SlotId;

const SLOTS: u32 = 4;
const CAPACITY: u32 = 32;

/// Two full-attention layers, two DeltaNet layers and the MTP block.
fn cfg(mtp: bool) -> Qwen35Config {
    Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 32,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 8,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 2,
        layer_types: default_layer_types(4, 2),
        partial_rotary_factor: 0.25,
        rope_theta: 1e7,
        rotary_dim: 8,
        mrope_section: [2, 2, 2, 2],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 4096,
        vocab_size: 256,
        attn_output_gate: true,
        mtp_num_hidden_layers: u32::from(mtp),
        mtp_use_dedicated_embeddings: true,
        intermediate_size: Some(128),
        moe: None,
    }
}

fn cache(device: &MlxDevice, mtp: bool, tq: bool) -> HybridKvCache {
    let cache = HybridKvCache::new_with_options(&cfg(mtp), device, CAPACITY, SLOTS, tq)
        .expect("allocate four-slot cache");
    assert_eq!(cache.mtp_slot.is_some(), mtp);
    assert_eq!(cache.full_attn.len(), 2);
    assert_eq!(cache.linear_attn.len(), 2);
    cache
}

/// Every per-slot buffer of the cache, in a fixed order.
fn buffers(cache: &HybridKvCache) -> Vec<&MlxBuffer> {
    let mut out = Vec::new();
    for block in cache.full_attn.iter().chain(cache.mtp_slot.iter()) {
        if let Some(tq) = &block.tq {
            out.extend([&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms]);
        }
        out.extend(block.k.as_ref());
        out.extend(block.v.as_ref());
    }
    for linear in &cache.linear_attn {
        out.extend([
            &linear.conv_state,
            &linear.conv_state_scratch,
            &linear.recurrent,
            &linear.recurrent_scratch,
        ]);
    }
    out
}

fn buffers_mut(cache: &mut HybridKvCache) -> Vec<&mut MlxBuffer> {
    let mut out = Vec::new();
    for block in cache.full_attn.iter_mut().chain(cache.mtp_slot.iter_mut()) {
        if let Some(tq) = block.tq.as_mut() {
            out.extend([
                &mut tq.k_packed,
                &mut tq.k_norms,
                &mut tq.v_packed,
                &mut tq.v_norms,
            ]);
        }
        out.extend(block.k.as_mut());
        out.extend(block.v.as_mut());
    }
    for linear in &mut cache.linear_attn {
        out.extend([
            &mut linear.conv_state,
            &mut linear.conv_state_scratch,
            &mut linear.recurrent,
            &mut linear.recurrent_scratch,
        ]);
    }
    out
}

/// Fill `slot`'s whole region of every buffer with a seed- and
/// buffer-distinct byte pattern.
fn plant_slot(cache: &mut HybridKvCache, slot: SlotId, seed: usize) {
    for (index, buffer) in buffers_mut(cache).into_iter().enumerate() {
        let region = buffer.byte_len() / SLOTS as usize;
        let start = slot.0 as usize * region;
        let bytes = buffer.as_mut_slice::<u8>().expect("mutable cache bytes");
        for (offset, byte) in bytes[start..start + region].iter_mut().enumerate() {
            *byte = ((seed * 31 + index * 17 + offset) % 251) as u8;
        }
    }
}

fn slot_region(buffer: &MlxBuffer, slot: SlotId) -> Vec<u8> {
    let region = buffer.byte_len() / SLOTS as usize;
    let start = slot.0 as usize * region;
    buffer.as_slice::<u8>().expect("cache bytes")[start..start + region].to_vec()
}

/// Rows `0..rows` of every head of `slot` in a `[n_seqs, heads, capacity,
/// inner]` buffer.
fn valid_rows(buffer: &MlxBuffer, slot: SlotId, rows: u32) -> Vec<u8> {
    let shape = buffer.shape();
    assert_eq!(shape.len(), 4);
    let (heads, capacity, inner) = (shape[1], shape[2], shape[3]);
    let row = inner * buffer.dtype().size_of();
    let bytes = buffer.as_slice::<u8>().expect("cache bytes");
    (0..heads)
        .flat_map(|head| {
            let start = (slot.0 as usize * heads + head) * capacity * row;
            bytes[start..start + rows as usize * row].to_vec()
        })
        .collect()
}

/// Every observable byte of one slot: cursors and valid K/V rows per
/// attention block, parity and all four state regions per DeltaNet layer,
/// and the graft length.
#[derive(PartialEq, Eq)]
struct Observed {
    attention: Vec<(u32, Vec<Vec<u8>>)>,
    linear: Vec<(bool, Vec<Vec<u8>>)>,
    graft_len: u32,
}

fn observe(cache: &HybridKvCache, slot: SlotId) -> Observed {
    let index = slot.0 as usize;
    let attention = cache
        .full_attn
        .iter()
        .chain(cache.mtp_slot.iter())
        .map(|block| {
            let rows = block.current_len[index];
            let mut regions = Vec::new();
            if let Some(tq) = &block.tq {
                for buffer in [&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms] {
                    regions.push(valid_rows(buffer, slot, rows));
                }
            }
            for buffer in block.k.iter().chain(block.v.iter()) {
                regions.push(valid_rows(buffer, slot, rows));
            }
            (rows, regions)
        })
        .collect();
    let linear = cache
        .linear_attn
        .iter()
        .map(|layer| {
            (
                layer.pp_flipped[index],
                [
                    &layer.conv_state,
                    &layer.conv_state_scratch,
                    &layer.recurrent,
                    &layer.recurrent_scratch,
                ]
                .into_iter()
                .map(|buffer| slot_region(buffer, slot))
                .collect(),
            )
        })
        .collect();
    Observed {
        attention,
        linear,
        graft_len: cache.graft_len[index],
    }
}

/// Every byte and every scalar a peer slot owns, observable or not.
fn raw_slot(cache: &HybridKvCache, slot: SlotId) -> (Vec<Vec<u8>>, Vec<u32>, Vec<bool>, u32) {
    let index = slot.0 as usize;
    (
        buffers(cache)
            .into_iter()
            .map(|buffer| slot_region(buffer, slot))
            .collect(),
        cache
            .full_attn
            .iter()
            .chain(cache.mtp_slot.iter())
            .map(|block| block.current_len[index])
            .collect(),
        cache
            .linear_attn
            .iter()
            .map(|layer| layer.pp_flipped[index])
            .collect(),
        cache.graft_len[index],
    )
}

/// Source slot 1: per-layer cursors, an MTP cursor independent of the
/// target, mixed DeltaNet parity and a graft region. Peers hold other values.
fn arrange(cache: &mut HybridKvCache) {
    for slot in 0..SLOTS {
        plant_slot(cache, SlotId(slot), slot as usize + 1);
    }
    for (layer, block) in cache.full_attn.iter_mut().enumerate() {
        block.current_len = vec![3, 9 + layer as u32, 5, 7];
    }
    if let Some(mtp) = cache.mtp_slot.as_mut() {
        mtp.current_len = vec![2, 6, 4, 1];
    }
    for (layer, linear) in cache.linear_attn.iter_mut().enumerate() {
        linear.pp_flipped = vec![false, layer % 2 == 0, true, layer % 2 == 1];
    }
    cache.graft_len = vec![0, 2, 0, 1];
}

#[test]
fn spilled_slot_restores_bitwise_into_another_slot_and_leaves_peers_alone() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    for (mtp, tq) in [(true, true), (true, false), (false, true)] {
        let label = format!("mtp={mtp} tq={tq}");
        let mut cache = cache(&device, mtp, tq);
        arrange(&mut cache);
        let source = observe(&cache, SlotId(1));
        let spill = cache.spill_slot(SlotId(1)).expect("spill slot 1");
        assert_eq!(
            spill.total_bytes(),
            cache.slot_spill_bytes(SlotId(1)).expect("spill size"),
            "{label}: size estimate must equal the copied bytes"
        );
        assert!(
            observe(&cache, SlotId(1)) == source,
            "{label}: spilling must not modify the source slot"
        );

        // The reclaim wipes the source; the destination holds another
        // conversation's bytes until the restore.
        cache.reset_for_slot(SlotId(1)).expect("wipe source slot");
        plant_slot(&mut cache, SlotId(3), 9);
        let peers = [0, 2].map(|slot| raw_slot(&cache, SlotId(slot)));
        cache
            .restore_slot_spill(SlotId(3), &spill)
            .expect("restore into slot 3");
        assert!(
            observe(&cache, SlotId(3)) == source,
            "{label}: slot 3 must equal slot 1 at spill time"
        );
        for (slot, before) in [0, 2].into_iter().zip(&peers) {
            assert!(
                raw_slot(&cache, SlotId(slot)) == *before,
                "{label}: peer slot {slot} changed"
            );
        }

        // The same spill also restores into its own (wiped) index.
        cache
            .restore_slot_spill(SlotId(1), &spill)
            .expect("restore into slot 1");
        assert!(
            observe(&cache, SlotId(1)) == source,
            "{label}: same-index restore after the wipe"
        );
    }
}

#[test]
fn restore_refuses_mismatched_geometry_before_writing() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut source = cache(&device, true, true);
    arrange(&mut source);
    let spill = source.spill_slot(SlotId(1)).expect("spill slot 1");

    for (mtp, tq, why) in [(true, false, "TQ/F32 mode"), (false, true, "MTP presence")] {
        let mut target = cache(&device, mtp, tq);
        arrange(&mut target);
        let before = raw_slot(&target, SlotId(2));
        assert!(
            target.restore_slot_spill(SlotId(2), &spill).is_err(),
            "{why} mismatch must be refused"
        );
        assert!(
            raw_slot(&target, SlotId(2)) == before,
            "{why} mismatch must be refused before any write"
        );
    }
    assert!(
        source.restore_slot_spill(SlotId(SLOTS), &spill).is_err(),
        "out-of-range slot must be refused"
    );
}

#[test]
fn spill_and_restore_refuse_an_active_capture_transaction() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut cache = cache(&device, true, true);
    arrange(&mut cache);
    let spill = cache
        .spill_slot(SlotId(1))
        .expect("spill between transactions");
    cache
        .ensure_la_capture(&cfg(true), &device, 2)
        .expect("arm capture");
    assert!(cache.spill_slot(SlotId(1)).is_err());
    assert!(cache.restore_slot_spill(SlotId(2), &spill).is_err());
    cache.clear_la_capture();
    cache
        .restore_slot_spill(SlotId(2), &spill)
        .expect("restore between transactions");
}
