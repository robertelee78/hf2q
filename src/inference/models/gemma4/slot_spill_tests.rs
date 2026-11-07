//! Issue #727: a spilled Gemma slot restores bitwise into any slot, moves in
//! place, touches no peer, and refuses a foreign geometry before writing.

use mlx_native::{DType, MlxDevice};

use super::super::{
    alloc_hb_kv_for_layer, alloc_multi_seq_dense_kv_for_layer, alloc_multi_seq_hybrid_kv_for_layer,
    alloc_multi_seq_mlx_kv_for_layer, MultiSeqDenseKvBuffers, MultiSeqHbKvBuffers,
    MultiSeqHybridKvBuffers, MultiSeqMlxKvCache,
};
use super::{GemmaSlotKvRegime, GemmaSlotScaffolds, SlotLayer};
use crate::inference::kv_slot_rows::{read_rows, SlotRows};
use crate::serve::multi_seq_kv::SlotId;

const SLOTS: u32 = 4;
/// A global layer and a sliding layer whose ring wraps in these tests.
const GLOBAL_CAP: usize = 64;
const RING_CAP: usize = 8;
const NKV: usize = 2;
const HD: usize = 16;

struct Scaffolds {
    hb: Vec<MultiSeqHbKvBuffers>,
    hybrid: Option<Vec<MultiSeqHybridKvBuffers>>,
    dense: Option<Vec<MultiSeqDenseKvBuffers>>,
    nibble: Option<Vec<MultiSeqMlxKvCache>>,
}

impl Scaffolds {
    fn view(&mut self) -> GemmaSlotScaffolds<'_> {
        GemmaSlotScaffolds {
            hb: &mut self.hb,
            hybrid: self.hybrid.as_deref_mut(),
            dense: self.dense.as_deref_mut(),
            nibble: self.nibble.as_deref_mut(),
        }
    }
}

/// `[global, sliding]` (plus a second global layer when `extra_layer`) of
/// every regime.
fn scaffolds(device: &MlxDevice, extra_layer: bool) -> Scaffolds {
    let mut shapes = vec![(GLOBAL_CAP, false), (RING_CAP, true)];
    if extra_layer {
        shapes.push((GLOBAL_CAP, false));
    }
    let each = |layer: usize, &(cap, ring): &(usize, bool)| (layer, cap, ring);
    let shapes: Vec<_> = shapes.iter().enumerate().map(|(i, s)| each(i, s)).collect();
    Scaffolds {
        hb: shapes
            .iter()
            .map(|&(layer, cap, ring)| {
                alloc_hb_kv_for_layer(device, layer, NKV, HD, cap, ring, SLOTS).expect("HB layer")
            })
            .collect(),
        hybrid: Some(
            shapes
                .iter()
                .map(|&(layer, cap, ring)| {
                    alloc_multi_seq_hybrid_kv_for_layer(device, layer, NKV, HD, cap, ring, SLOTS)
                        .expect("hybrid layer")
                })
                .collect(),
        ),
        dense: Some(
            shapes
                .iter()
                .map(|&(layer, cap, ring)| {
                    alloc_multi_seq_dense_kv_for_layer(
                        device,
                        layer,
                        NKV,
                        HD,
                        cap,
                        ring,
                        DType::F32,
                        SLOTS,
                    )
                    .expect("dense layer")
                })
                .collect(),
        ),
        nibble: Some(
            shapes
                .iter()
                .map(|&(layer, cap, ring)| {
                    // norms_per_pos 1: the 3-D `[n, nkv, cap]` norm layout.
                    alloc_multi_seq_mlx_kv_for_layer(device, layer, NKV, HD, cap, ring, 1, SLOTS)
                        .expect("nibble layer")
                })
                .collect(),
        ),
    }
}

fn layers_of(scaffolds: &mut Scaffolds, regime: GemmaSlotKvRegime) -> Vec<&mut dyn SlotLayer> {
    fn erase<T: SlotLayer + 'static>(layers: &mut [T]) -> Vec<&mut dyn SlotLayer> {
        layers
            .iter_mut()
            .map(|layer| layer as &mut dyn SlotLayer)
            .collect()
    }
    match regime {
        GemmaSlotKvRegime::Hb => erase(&mut scaffolds.hb),
        GemmaSlotKvRegime::Hybrid => erase(scaffolds.hybrid.as_mut().expect("hybrid")),
        GemmaSlotKvRegime::Dense => erase(scaffolds.dense.as_mut().expect("dense")),
        GemmaSlotKvRegime::Nibble => erase(scaffolds.nibble.as_mut().expect("nibble")),
    }
}

const ALL: [GemmaSlotKvRegime; 4] = [
    GemmaSlotKvRegime::Hb,
    GemmaSlotKvRegime::Hybrid,
    GemmaSlotKvRegime::Dense,
    GemmaSlotKvRegime::Nibble,
];

/// Fill every per-slot buffer of every regime with a slot- and
/// buffer-distinct byte pattern, and give each slot its own cursor.
fn plant(scaffolds: &mut Scaffolds, cursors: [u32; SLOTS as usize]) {
    for (regime_index, regime) in ALL.into_iter().enumerate() {
        for (layer_index, layer) in layers_of(scaffolds, regime).into_iter().enumerate() {
            for (buffer_index, buffer) in layer.slot_buffers_mut().into_iter().enumerate() {
                let bytes = buffer.as_mut_slice::<u8>().expect("host-visible buffer");
                let per_slot = bytes.len() / SLOTS as usize;
                for (offset, byte) in bytes.iter_mut().enumerate() {
                    let slot = offset / per_slot;
                    *byte = (offset.wrapping_mul(31).wrapping_add(
                        slot * 101 + regime_index * 7 + layer_index * 13 + buffer_index * 17,
                    ) % 251) as u8;
                }
            }
            layer.seq_lens_mut().copy_from_slice(&cursors);
        }
    }
}

/// Every per-slot byte of `slot` in every buffer of every regime, plus its
/// cursors: what a misaddressed write could change.
fn whole_slot(scaffolds: &mut Scaffolds, slot: usize) -> Vec<Vec<u8>> {
    let mut image = Vec::new();
    for regime in ALL {
        for layer in layers_of(scaffolds, regime) {
            let n_seqs = layer.n_seqs();
            for buffer in layer.slot_buffers() {
                let capacity = SlotRows::of(buffer, n_seqs, "test").unwrap().capacity();
                image.push(read_rows(buffer, slot, n_seqs, capacity, "test").unwrap());
            }
            image.push(layer.seq_lens()[slot].to_le_bytes().to_vec());
        }
    }
    image
}

/// Rows `0..rows(layer)` of `slot` in the `regime` scaffold.
fn rows_of(
    scaffolds: &mut Scaffolds,
    regime: GemmaSlotKvRegime,
    slot: usize,
    rows: impl Fn(usize, usize) -> usize,
) -> Vec<Vec<u8>> {
    let mut image = Vec::new();
    for (index, layer) in layers_of(scaffolds, regime).into_iter().enumerate() {
        let n_seqs = layer.n_seqs();
        let rows = rows(index, layer.capacity());
        for buffer in layer.slot_buffers() {
            image.push(read_rows(buffer, slot, n_seqs, rows, "test").unwrap());
        }
    }
    image
}

fn cursors_of(scaffolds: &mut Scaffolds, slot: usize) -> Vec<u32> {
    ALL.into_iter()
        .flat_map(|regime| {
            layers_of(scaffolds, regime)
                .into_iter()
                .map(|layer| layer.seq_lens()[slot])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn buffer_count(scaffolds: &mut Scaffolds, regime: GemmaSlotKvRegime) -> usize {
    layers_of(scaffolds, regime)
        .into_iter()
        .map(|layer| layer.slot_buffers().len())
        .sum()
}

/// Rows `from..GLOBAL_CAP` of every head of every buffer of the global layer
/// (layer 0) of `slot`.
fn global_tail(
    scaffolds: &mut Scaffolds,
    regime: GemmaSlotKvRegime,
    slot: usize,
    from: usize,
) -> Vec<Vec<u8>> {
    let layers = layers_of(scaffolds, regime);
    let layer = &layers[0];
    assert_eq!(layer.capacity(), GLOBAL_CAP);
    let n_seqs = layer.n_seqs();
    layer
        .slot_buffers()
        .into_iter()
        .flat_map(|buffer| {
            let all = read_rows(buffer, slot, n_seqs, GLOBAL_CAP, "test").unwrap();
            let head_bytes = all.len() / NKV;
            let row_bytes = head_bytes / GLOBAL_CAP;
            (0..NKV)
                .map(|head| {
                    all[head * head_bytes + from * row_bytes..(head + 1) * head_bytes].to_vec()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn spilled_gemma_slot_restores_bitwise_into_another_slot_and_leaves_peers_alone() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    for regime in ALL {
        let mut scaffolds = scaffolds(&device, false);
        // Slot 1 has wrapped its 8-row ring; slot 3 holds another
        // conversation with a longer history.
        plant(&mut scaffolds, [5, 20, 0, 40]);
        let source_rows = rows_of(&mut scaffolds, regime, 1, |_, cap| 20usize.min(cap));
        let source_cursors = cursors_of(&mut scaffolds, 1);
        let peers_before: Vec<_> = [0, 2]
            .into_iter()
            .map(|slot| whole_slot(&mut scaffolds, slot))
            .collect();
        let dest_tail = global_tail(&mut scaffolds, regime, 3, 20);

        let spill = scaffolds.view().spill(regime, SlotId(1), 0).expect("spill");
        assert_eq!(spill.cursor(), Some(20), "{regime:?}");
        let expected_bytes: usize =
            source_rows.iter().map(Vec::len).sum::<usize>() + 4 * 2 * std::mem::size_of::<u32>();
        assert_eq!(spill.total_bytes(), expected_bytes, "{regime:?}");
        assert_eq!(
            scaffolds.view().spill_bytes(regime, SlotId(1), 0).unwrap(),
            expected_bytes,
            "{regime:?}: spill_bytes is the spill's size without copying"
        );
        assert_eq!(
            spill
                .layers
                .iter()
                .map(|layer| layer.buffers.len())
                .sum::<usize>(),
            buffer_count(&mut scaffolds, regime),
            "{regime:?}: every per-slot buffer of the live scaffold"
        );

        // The reclaim wipe of the source, then the restore into slot 3.
        for layer_regime in ALL {
            for layer in layers_of(&mut scaffolds, layer_regime) {
                layer.seq_lens_mut()[1] = 0;
            }
        }
        scaffolds
            .view()
            .restore(SlotId(3), &spill)
            .expect("restore");
        assert_eq!(
            rows_of(&mut scaffolds, regime, 3, |_, cap| 20usize.min(cap)),
            source_rows,
            "{regime:?}: restored rows"
        );
        assert_eq!(
            cursors_of(&mut scaffolds, 3),
            source_cursors,
            "{regime:?}: cursors"
        );
        // Global rows at or above the restored cursor are not written.
        assert_eq!(
            global_tail(&mut scaffolds, regime, 3, 20),
            dest_tail,
            "{regime:?}: global rows above the cursor untouched"
        );
        for (slot, before) in [0, 2].into_iter().zip(&peers_before) {
            assert_eq!(
                &whole_slot(&mut scaffolds, slot),
                before,
                "{regime:?}: peer {slot}"
            );
        }
        // A second restore of the same spill is idempotent.
        scaffolds
            .view()
            .restore(SlotId(3), &spill)
            .expect("second restore");
        assert_eq!(
            rows_of(&mut scaffolds, regime, 3, |_, cap| 20usize.min(cap)),
            source_rows
        );
    }
}

#[test]
fn rows_floor_extends_global_rows_to_the_anchor_boundary() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut scaffolds = scaffolds(&device, false);
    plant(&mut scaffolds, [0, 10, 0, 0]);
    let regime = GemmaSlotKvRegime::Hybrid;
    let spill = scaffolds
        .view()
        .spill(regime, SlotId(1), 16)
        .expect("spill");
    assert_eq!(
        spill
            .layers
            .iter()
            .map(|layer| layer.rows)
            .collect::<Vec<_>>(),
        vec![16, RING_CAP],
        "global rows reach the floor; the ring is bounded by its capacity"
    );
    assert_eq!(spill.cursor(), Some(10), "the floor never moves a cursor");
    let expected = rows_of(&mut scaffolds, regime, 1, |_, cap| 16usize.min(cap));
    scaffolds.view().restore(SlotId(2), &spill).unwrap();
    assert_eq!(
        rows_of(&mut scaffolds, regime, 2, |_, cap| 16usize.min(cap)),
        expected
    );
    assert_eq!(cursors_of(&mut scaffolds, 2), vec![10; 8]);
}

#[test]
fn copy_slot_moves_state_in_place_without_touching_source_or_peers() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut scaffolds = scaffolds(&device, false);
    plant(&mut scaffolds, [7, 30, 3, 12]);
    let regime = GemmaSlotKvRegime::Hybrid;
    let reference = scaffolds.view().spill(regime, SlotId(1), 0).unwrap();
    let source = whole_slot(&mut scaffolds, 1);
    let peers: Vec<_> = [0, 3]
        .into_iter()
        .map(|slot| whole_slot(&mut scaffolds, slot))
        .collect();
    scaffolds
        .view()
        .copy_slot(regime, SlotId(1), SlotId(2), 0)
        .expect("in-place move");
    assert!(
        scaffolds.view().spill(regime, SlotId(2), 0).unwrap() == reference,
        "the moved slot equals a spill of the source"
    );
    assert_eq!(cursors_of(&mut scaffolds, 2), cursors_of(&mut scaffolds, 1));
    assert_eq!(whole_slot(&mut scaffolds, 1), source, "source unchanged");
    for (slot, before) in [0, 3].into_iter().zip(&peers) {
        assert_eq!(&whole_slot(&mut scaffolds, slot), before, "peer {slot}");
    }
}

#[test]
fn restore_refuses_a_foreign_geometry_before_writing() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut two = scaffolds(&device, false);
    plant(&mut two, [0, 12, 0, 0]);
    let spill = two
        .view()
        .spill(GemmaSlotKvRegime::Hybrid, SlotId(1), 0)
        .unwrap();
    let mut three = scaffolds(&device, true);
    plant(&mut three, [1, 2, 3, 4]);
    let before = whole_slot(&mut three, 2);
    let error = three
        .view()
        .restore(SlotId(2), &spill)
        .expect_err("layer count differs");
    assert!(format!("{error:#}").contains("layers"), "{error:#}");
    assert_eq!(whole_slot(&mut three, 2), before, "nothing written");

    // A spill of a scaffold the cache no longer provisions is refused too.
    let mut without_dense = scaffolds(&device, false);
    plant(&mut without_dense, [0, 0, 0, 0]);
    without_dense.dense = None;
    let error = without_dense
        .view()
        .restore(SlotId(0), &spill)
        .expect_err("dense presence differs");
    assert!(format!("{error:#}").contains("presence"), "{error:#}");
    let error = without_dense
        .view()
        .spill(GemmaSlotKvRegime::Dense, SlotId(0), 0)
        .expect_err("live scaffold absent");
    assert!(
        format!("{error:#}").contains("not provisioned"),
        "{error:#}"
    );
}

#[test]
fn slot_rows_cover_every_scaffold_buffer_layout() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let mut scaffolds = scaffolds(&device, false);
    for regime in ALL {
        for layer in layers_of(&mut scaffolds, regime) {
            let n_seqs = layer.n_seqs();
            let capacity = layer.capacity();
            for buffer in layer.slot_buffers() {
                let rows = SlotRows::of(buffer, n_seqs, "layout").expect("slot-major buffer");
                assert_eq!(rows.capacity(), capacity, "{regime:?}");
                assert_eq!(
                    rows.spilled_len(capacity, "layout").unwrap() * n_seqs,
                    buffer.byte_len(),
                    "{regime:?}: a slot's rows tile its region exactly"
                );
            }
        }
    }
}
