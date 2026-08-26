//! Exact-artifact authority for Gemma live rectangular prefill.
//!
//! Gemma has no recurrent/SSM state. Its complete persistent state at these
//! boundaries is the hybrid attention cache: per-layer cursor and layout plus
//! every logically live K, packed-V, V-norm, and optional xlen K/V byte. The
//! selected-slot oracle below compares those bytes directly; SHA-256 values
//! are retained only for compact mismatch diagnostics. Physical selected-slot tails beyond
//! the cursor are overwrite-backed and are not logical state. An unselected
//! slot is instead seeded and checked across its complete physical regions so
//! an out-of-lane write cannot hide in such a tail.

use super::*;

use crate::inference::models::gemma4::kv_cache::{
    GemmaHybridSlotAnchor, MultiSeqHbKvBuffers, MultiSeqHybridKvBuffers,
};
use crate::inference::models::gemma4::model::MultiSeqPrefillOutput;
use sha2::{Digest, Sha256};

const TEST_NAME: &str =
    "gemma_stable_whole_route_rectangles_tail_cue_anchor_and_continuation_are_exact";

#[derive(Clone)]
struct BufferImage {
    dtype: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
    sha256: [u8; 32],
}

#[derive(Clone)]
struct LayerImage {
    layer: usize,
    cursor: u32,
    capacity: usize,
    is_sliding: bool,
    norms_per_pos: usize,
    k: BufferImage,
    v_packed: BufferImage,
    v_norms: BufferImage,
    bf16_xlen_k: Option<BufferImage>,
    bf16_xlen_v: Option<BufferImage>,
}

#[derive(Clone)]
struct SlotImage {
    slot: SlotId,
    layers: Vec<LayerImage>,
}

struct CanaryLayer {
    layer: usize,
    cursor: u32,
    k: Vec<u8>,
    v_packed: Vec<u8>,
    v_norms: Option<Vec<u8>>,
    bf16_xlen_k: Option<Vec<u8>>,
    bf16_xlen_v: Option<Vec<u8>>,
}

fn gated_artifact() -> Option<PathBuf> {
    if std::env::var("HF2Q_BYTE_EQUIV_E2E").as_deref() != Ok("1") {
        eprintln!(
            "[skip] {TEST_NAME} — set HF2Q_BYTE_EQUIV_E2E=1 and \
             HF2Q_BYTE_EQUIV_E2E_GGUF=<path/to/gemma4.gguf>"
        );
        return None;
    }
    for (name, expected) in [
        ("HF2Q_HYBRID_KV", "1"),
        ("HF2Q_USE_DENSE", "0"),
        ("HF2Q_TQ_CODEBOOK_BITS", "8"),
        ("HF2Q_DFLASH_XLEN_SDPA", "0"),
    ] {
        assert_eq!(
            std::env::var(name).as_deref(),
            Ok(expected),
            "{TEST_NAME} requires {name}={expected}"
        );
    }
    let path = PathBuf::from(
        std::env::var("HF2Q_BYTE_EQUIV_E2E_GGUF").expect("HF2Q_BYTE_EQUIV_E2E_GGUF is required"),
    );
    assert!(path.is_file(), "GGUF missing: {}", path.display());
    Some(path)
}

fn load_options(model_path: &Path) -> LoadOptions {
    LoadOptions {
        model_path: model_path.to_path_buf(),
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn slot_region<'a>(
    buffer: &'a mlx_native::MlxBuffer,
    slot: SlotId,
    n_seqs: usize,
    context: &str,
) -> &'a [u8] {
    assert!(n_seqs > 0, "{context}: zero slots");
    assert!((slot.0 as usize) < n_seqs, "{context}: slot out of range");
    let bytes = buffer
        .as_slice::<u8>()
        .unwrap_or_else(|error| panic!("{context}: raw view failed: {error}"));
    assert_eq!(
        bytes.len() % n_seqs,
        0,
        "{context}: allocation is not slot-divisible"
    );
    let per_slot = bytes.len() / n_seqs;
    let start = slot.0 as usize * per_slot;
    &bytes[start..start + per_slot]
}

fn slot_region_mut<'a>(
    buffer: &'a mut mlx_native::MlxBuffer,
    slot: SlotId,
    n_seqs: usize,
    context: &str,
) -> &'a mut [u8] {
    assert!(n_seqs > 0, "{context}: zero slots");
    assert!((slot.0 as usize) < n_seqs, "{context}: slot out of range");
    let bytes = buffer
        .as_mut_slice::<u8>()
        .unwrap_or_else(|error| panic!("{context}: mutable raw view failed: {error}"));
    assert_eq!(
        bytes.len() % n_seqs,
        0,
        "{context}: allocation is not slot-divisible"
    );
    let per_slot = bytes.len() / n_seqs;
    let start = slot.0 as usize * per_slot;
    &mut bytes[start..start + per_slot]
}

fn seed_canary(scaffold: &mut [MultiSeqHybridKvBuffers], slot: SlotId) -> Vec<CanaryLayer> {
    scaffold
        .iter_mut()
        .enumerate()
        .map(|(layer, buffer)| {
            let n_seqs = buffer.n_seqs as usize;
            let seed = |field: u8| {
                (layer as u8)
                    .wrapping_mul(29)
                    .wrapping_add(field.wrapping_mul(47))
                    .wrapping_add(slot.0 as u8)
            };
            slot_region_mut(&mut buffer.k, slot, n_seqs, "canary K").fill(seed(1));
            slot_region_mut(&mut buffer.v_packed, slot, n_seqs, "canary V").fill(seed(2));
            let v_norms = if buffer.v_norms.data_byte_len() == 4 {
                None
            } else {
                slot_region_mut(&mut buffer.v_norms, slot, n_seqs, "canary V norms").fill(seed(3));
                Some(slot_region(&buffer.v_norms, slot, n_seqs, "canary V norms snapshot").to_vec())
            };
            let bf16_xlen_k = buffer.bf16_xlen_k.as_mut().map(|xlen| {
                slot_region_mut(xlen, slot, n_seqs, "canary xlen K").fill(seed(4));
                slot_region(xlen, slot, n_seqs, "canary xlen K snapshot").to_vec()
            });
            let bf16_xlen_v = buffer.bf16_xlen_v.as_mut().map(|xlen| {
                slot_region_mut(xlen, slot, n_seqs, "canary xlen V").fill(seed(5));
                slot_region(xlen, slot, n_seqs, "canary xlen V snapshot").to_vec()
            });
            CanaryLayer {
                layer,
                cursor: buffer.seq_lens[slot.0 as usize],
                k: slot_region(&buffer.k, slot, n_seqs, "canary K snapshot").to_vec(),
                v_packed: slot_region(&buffer.v_packed, slot, n_seqs, "canary V snapshot").to_vec(),
                v_norms,
                bf16_xlen_k,
                bf16_xlen_v,
            }
        })
        .collect()
}

fn assert_canary_unchanged(
    scaffold: &[MultiSeqHybridKvBuffers],
    slot: SlotId,
    expected: &[CanaryLayer],
    stage: &str,
) {
    assert_eq!(scaffold.len(), expected.len(), "{stage}: layer count");
    for (buffer, saved) in scaffold.iter().zip(expected) {
        let n_seqs = buffer.n_seqs as usize;
        assert_eq!(saved.layer, expected[saved.layer].layer);
        assert_eq!(
            buffer.seq_lens[slot.0 as usize], saved.cursor,
            "{stage}: L{} unselected cursor changed",
            saved.layer
        );
        assert_eq!(
            slot_region(&buffer.k, slot, n_seqs, "canary K check"),
            saved.k,
            "{stage}: L{} unselected K changed",
            saved.layer
        );
        assert_eq!(
            slot_region(&buffer.v_packed, slot, n_seqs, "canary V check"),
            saved.v_packed,
            "{stage}: L{} unselected V changed",
            saved.layer
        );
        match (&saved.v_norms, buffer.v_norms.data_byte_len() == 4) {
            (None, true) => {}
            (Some(expected_bytes), false) => assert_eq!(
                slot_region(&buffer.v_norms, slot, n_seqs, "canary V norms check"),
                expected_bytes,
                "{stage}: L{} unselected V norms changed",
                saved.layer
            ),
            _ => panic!("{stage}: L{} V norms layout changed", saved.layer),
        }
        match (&saved.bf16_xlen_k, &buffer.bf16_xlen_k) {
            (None, None) => {}
            (Some(expected_bytes), Some(current)) => assert_eq!(
                slot_region(current, slot, n_seqs, "canary xlen K check"),
                expected_bytes,
                "{stage}: L{} unselected xlen K changed",
                saved.layer
            ),
            _ => panic!("{stage}: L{} xlen K layout changed", saved.layer),
        }
        match (&saved.bf16_xlen_v, &buffer.bf16_xlen_v) {
            (None, None) => {}
            (Some(expected_bytes), Some(current)) => assert_eq!(
                slot_region(current, slot, n_seqs, "canary xlen V check"),
                expected_bytes,
                "{stage}: L{} unselected xlen V changed",
                saved.layer
            ),
            _ => panic!("{stage}: L{} xlen V layout changed", saved.layer),
        }
    }
}

fn live_buffer_image(
    buffer: &mlx_native::MlxBuffer,
    slot: SlotId,
    n_seqs: usize,
    cursor: usize,
    capacity: usize,
    context: &str,
) -> BufferImage {
    let shape = buffer.shape().to_vec();
    assert!(
        shape.len() >= 3,
        "{context}: expected slot/head/position axes"
    );
    assert_eq!(shape[0], n_seqs, "{context}: slot axis");
    assert_eq!(shape[2], capacity, "{context}: capacity axis");
    assert!(cursor <= capacity, "{context}: cursor exceeds capacity");
    let element_bytes = buffer.dtype().size_of();
    let inner: usize = shape[3..].iter().product();
    let heads = shape[1];
    let head_stride = capacity * inner * element_bytes;
    let slot_stride = heads * head_stride;
    let live_head_bytes = cursor * inner * element_bytes;
    let raw = buffer
        .as_slice::<u8>()
        .unwrap_or_else(|error| panic!("{context}: raw view failed: {error}"));
    assert_eq!(raw.len(), n_seqs * slot_stride, "{context}: byte extent");
    let slot_start = slot.0 as usize * slot_stride;
    let mut bytes = Vec::with_capacity(heads * live_head_bytes);
    for head in 0..heads {
        let start = slot_start + head * head_stride;
        bytes.extend_from_slice(&raw[start..start + live_head_bytes]);
    }
    BufferImage {
        dtype: format!("{:?}", buffer.dtype()),
        shape,
        sha256: sha256(&bytes),
        bytes,
    }
}

fn full_buffer_image(buffer: &mlx_native::MlxBuffer) -> BufferImage {
    let bytes = buffer
        .as_slice::<u8>()
        .expect("dummy cache buffer raw view")
        .to_vec();
    BufferImage {
        dtype: format!("{:?}", buffer.dtype()),
        shape: buffer.shape().to_vec(),
        sha256: sha256(&bytes),
        bytes,
    }
}

fn snapshot_selected(
    scaffold: &[MultiSeqHybridKvBuffers],
    slots: &[SlotId],
    expected_cursor: usize,
) -> Vec<SlotImage> {
    slots
        .iter()
        .copied()
        .map(|slot| {
            let layers = scaffold
                .iter()
                .enumerate()
                .map(|(layer, buffer)| {
                    let cursor = buffer.seq_lens[slot.0 as usize];
                    assert_eq!(cursor as usize, expected_cursor, "L{layer} slot cursor");
                    let n_seqs = buffer.n_seqs as usize;
                    let live = expected_cursor.min(buffer.capacity);
                    LayerImage {
                        layer,
                        cursor,
                        capacity: buffer.capacity,
                        is_sliding: buffer.is_sliding,
                        norms_per_pos: buffer.norms_per_pos,
                        k: live_buffer_image(
                            &buffer.k,
                            slot,
                            n_seqs,
                            live,
                            buffer.capacity,
                            "selected K",
                        ),
                        v_packed: live_buffer_image(
                            &buffer.v_packed,
                            slot,
                            n_seqs,
                            live,
                            buffer.capacity,
                            "selected V",
                        ),
                        v_norms: if buffer.v_norms.data_byte_len() == 4 {
                            full_buffer_image(&buffer.v_norms)
                        } else {
                            live_buffer_image(
                                &buffer.v_norms,
                                slot,
                                n_seqs,
                                live,
                                buffer.capacity,
                                "selected V norms",
                            )
                        },
                        bf16_xlen_k: buffer.bf16_xlen_k.as_ref().map(|xlen| {
                            live_buffer_image(
                                xlen,
                                slot,
                                n_seqs,
                                live,
                                buffer.capacity,
                                "selected xlen K",
                            )
                        }),
                        bf16_xlen_v: buffer.bf16_xlen_v.as_ref().map(|xlen| {
                            live_buffer_image(
                                xlen,
                                slot,
                                n_seqs,
                                live,
                                buffer.capacity,
                                "selected xlen V",
                            )
                        }),
                    }
                })
                .collect();
            SlotImage { slot, layers }
        })
        .collect()
}

fn install_cursor(
    hb: &mut [MultiSeqHbKvBuffers],
    hybrid: &mut [MultiSeqHybridKvBuffers],
    slots: &[SlotId],
    expected: usize,
    committed: usize,
) {
    let expected = expected as u32;
    let committed = committed as u32;
    for slot in slots {
        for (layer, buffer) in hb.iter_mut().enumerate() {
            assert_eq!(
                buffer.seq_lens[slot.0 as usize], expected,
                "HB L{layer} cursor"
            );
            buffer.seq_lens[slot.0 as usize] = committed;
        }
        for (layer, buffer) in hybrid.iter_mut().enumerate() {
            assert_eq!(
                buffer.seq_lens[slot.0 as usize], expected,
                "hybrid L{layer} cursor"
            );
            buffer.seq_lens[slot.0 as usize] = committed;
        }
    }
}

fn assert_buffer_exact(expected: &BufferImage, actual: &BufferImage, context: &str) {
    assert_eq!(expected.dtype, actual.dtype, "{context}: dtype");
    assert_eq!(expected.shape, actual.shape, "{context}: shape");
    assert_eq!(
        expected.bytes.len(),
        actual.bytes.len(),
        "{context}: extent"
    );
    if expected.bytes != actual.bytes {
        let first = expected
            .bytes
            .iter()
            .zip(&actual.bytes)
            .position(|(left, right)| left != right)
            .expect("unequal same-length buffers have a differing byte");
        panic!(
            "{context}: byte {first} differs: scalar=0x{:02x} rectangular=0x{:02x}; scalar_sha={} rectangular_sha={}",
            expected.bytes[first],
            actual.bytes[first],
            hex::encode(expected.sha256),
            hex::encode(actual.sha256),
        );
    }
    assert_eq!(expected.sha256, actual.sha256, "{context}: SHA receipt");
}

fn assert_state_exact(expected: &[SlotImage], actual: &[SlotImage], stage: &str) {
    assert_eq!(expected.len(), actual.len(), "{stage}: slot count");
    for (left_slot, right_slot) in expected.iter().zip(actual) {
        assert_eq!(left_slot.slot, right_slot.slot, "{stage}: slot order");
        assert_eq!(
            left_slot.layers.len(),
            right_slot.layers.len(),
            "{stage}: layer count"
        );
        for (left, right) in left_slot.layers.iter().zip(&right_slot.layers) {
            let prefix = format!("{stage} slot {} L{}", left_slot.slot.0, left.layer);
            assert_eq!(left.layer, right.layer, "{prefix}: layer");
            assert_eq!(left.cursor, right.cursor, "{prefix}: cursor");
            assert_eq!(left.capacity, right.capacity, "{prefix}: capacity");
            assert_eq!(left.is_sliding, right.is_sliding, "{prefix}: sliding");
            assert_eq!(
                left.norms_per_pos, right.norms_per_pos,
                "{prefix}: norms_per_pos"
            );
            assert_buffer_exact(&left.k, &right.k, &format!("{prefix} K"));
            assert_buffer_exact(&left.v_packed, &right.v_packed, &format!("{prefix} V"));
            assert_buffer_exact(&left.v_norms, &right.v_norms, &format!("{prefix} V norms"));
            match (&left.bf16_xlen_k, &right.bf16_xlen_k) {
                (None, None) => {}
                (Some(left), Some(right)) => {
                    assert_buffer_exact(left, right, &format!("{prefix} xlen K"))
                }
                _ => panic!("{prefix}: xlen K presence differs"),
            }
            match (&left.bf16_xlen_v, &right.bf16_xlen_v) {
                (None, None) => {}
                (Some(left), Some(right)) => {
                    assert_buffer_exact(left, right, &format!("{prefix} xlen V"))
                }
                _ => panic!("{prefix}: xlen V presence differs"),
            }
        }
    }
}

fn assert_output_exact(
    expected: &MultiSeqPrefillOutput,
    actual: &MultiSeqPrefillOutput,
    stage: &str,
) {
    assert_eq!(
        expected.first_tokens, actual.first_tokens,
        "{stage}: tokens"
    );
    assert_eq!(expected.logits.len(), actual.logits.len(), "{stage}: rows");
    for (lane, (left, right)) in expected.logits.iter().zip(&actual.logits).enumerate() {
        assert_eq!(left.len(), right.len(), "{stage} lane {lane}: vocab");
        assert!(
            left.iter().chain(right).all(|value| value.is_finite()),
            "{stage} lane {lane}: non-finite logit"
        );
        for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "{stage} lane {lane} logit {index}"
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WholeRouteCase {
    width: usize,
    initial_cursor: usize,
    rows_per_lane: usize,
    crosses_sliding_wrap: bool,
}

impl WholeRouteCase {
    fn boundary(self) -> usize {
        self.initial_cursor + self.rows_per_lane
    }

    fn prompt_len(self) -> usize {
        self.boundary() + 7
    }

    fn label(self) -> String {
        format!(
            "B{} start={} rows={} wrap={}",
            self.width, self.initial_cursor, self.rows_per_lane, self.crosses_sliding_wrap
        )
    }
}

struct WholeRouteReceipt {
    cue: MultiSeqPrefillOutput,
    cue_state: Vec<SlotImage>,
    restored_boundary_state: Vec<SlotImage>,
    continuation: MultiSeqPrefillOutput,
    continuation_state: Vec<SlotImage>,
    rectangular_slices: usize,
}

fn whole_route_slots(width: usize) -> (Vec<SlotId>, SlotId) {
    match width {
        2 => (vec![SlotId(3), SlotId(1)], SlotId(2)),
        4 => (vec![SlotId(0), SlotId(2), SlotId(3), SlotId(4)], SlotId(1)),
        _ => panic!("whole-route authority only covers B2/B4"),
    }
}

fn whole_route_prompts(case: WholeRouteCase) -> Vec<Vec<u32>> {
    (0..case.width)
        .map(|lane| {
            (0..case.prompt_len())
                .map(|position| 101 + ((lane as u32 * 2_003 + position as u32 * 29) % 24_000))
                .collect()
        })
        .collect()
}

fn reset_whole_route_scaffolds(
    hb: &mut [MultiSeqHbKvBuffers],
    hybrid: &mut [MultiSeqHybridKvBuffers],
    n_seqs: u32,
) {
    for raw_slot in 0..n_seqs {
        let slot = SlotId(raw_slot);
        for (layer, buffer) in hb.iter_mut().enumerate() {
            buffer
                .reset_for_slot(slot)
                .unwrap_or_else(|error| panic!("reset HB slot {raw_slot} L{layer}: {error}"));
        }
        for (layer, buffer) in hybrid.iter_mut().enumerate() {
            buffer
                .reset_for_slot(slot)
                .unwrap_or_else(|error| panic!("reset hybrid slot {raw_slot} L{layer}: {error}"));
        }
    }
}

fn push_current_gemma_output(
    g: &GemmaLoadedModel,
    token: u32,
    first_tokens: &mut Vec<u32>,
    logits: &mut Vec<Vec<f32>>,
    stage: &str,
) {
    first_tokens.push(token);
    logits.push(
        g.weights
            .logits_view()
            .unwrap_or_else(|error| panic!("{stage}: read logits: {error:#}"))
            .to_vec(),
    );
}

fn run_whole_route_arm(
    g: &mut GemmaLoadedModel,
    hb: &mut Vec<MultiSeqHbKvBuffers>,
    hybrid: &mut Vec<MultiSeqHybridKvBuffers>,
    case: WholeRouteCase,
    rectangular: bool,
) -> WholeRouteReceipt {
    let label = case.label();
    eprintln!(
        "HF2Q_GEMMA_STABLE_WHOLE_ROUTE_ARM label={label:?} route={}",
        if rectangular { "rectangular" } else { "scalar" }
    );
    let (selected, unselected) = whole_route_slots(case.width);
    let n_seqs = hybrid.first().expect("whole-route hybrid scaffold").n_seqs;
    reset_whole_route_scaffolds(hb, hybrid, n_seqs);
    clear_gemma4_self_mounts(g);
    let canary = seed_canary(hybrid, unselected);
    let prompts = whole_route_prompts(case);
    let boundary = case.boundary();
    let prompt_len = case.prompt_len();

    for (lane, slot) in selected.iter().copied().enumerate() {
        clear_gemma4_self_mounts(g);
        g.weights
            .forward_prefill_with_soft_tokens_slot_aware(
                &prompts[lane][..case.initial_cursor],
                &[],
                8,
                &mut g.ctx,
                slot,
                hb,
                Some(&mut *hybrid),
                None,
                None,
            )
            .unwrap_or_else(|error| panic!("{label}: prefix lane {lane}: {error:#}"));
    }
    clear_gemma4_self_mounts(g);
    install_cursor(hb, hybrid, &selected, 0, case.initial_cursor);
    assert_canary_unchanged(hybrid, unselected, &canary, &format!("{label} prefix"));

    let rows = boundary - case.initial_cursor;
    let mut resume_cursor = case.initial_cursor;
    let mut rectangular_slices = 0usize;
    if rectangular {
        let seqs: Vec<_> = selected
            .iter()
            .copied()
            .zip(prompts.iter())
            .map(|(slot, prompt)| {
                (
                    prompt[resume_cursor..boundary].to_vec(),
                    slot,
                    resume_cursor,
                )
            })
            .collect();
        let _ = g
            .weights
            .forward_prefill_batched_multi_seq_live(&seqs, hybrid, 8, &mut g.ctx)
            .unwrap_or_else(|error| panic!("{label}: full-width rectangle M{rows}: {error:#}"));
        clear_gemma4_self_mounts(g);
        assert_canary_unchanged(
            hybrid,
            unselected,
            &canary,
            &format!("{label} full-width rectangle"),
        );
        resume_cursor = boundary;
        rectangular_slices = 1;
    }

    let supervisor = EngineSupervisor::new();
    let mut anchors: Vec<GemmaHybridSlotAnchor> = Vec::with_capacity(case.width);
    let mut cue_tokens = Vec::with_capacity(case.width);
    let mut cue_logits = Vec::with_capacity(case.width);
    let params = SamplingParams {
        max_tokens: 8,
        stable_prompt_prefix_tokens: Some(boundary),
        ..SamplingParams::default()
    };
    for (lane, slot) in selected.iter().copied().enumerate() {
        let lane_resume = if rectangular {
            resume_cursor
        } else {
            case.initial_cursor
        };
        let (state, anchor) = Gemma4DecodeState::prefill_seed(
            g,
            &prompts[lane],
            &[],
            &params,
            None,
            slot,
            hb,
            Some(&mut *hybrid),
            None,
            None,
            &supervisor,
            lane_resume,
            case.initial_cursor,
        )
        .unwrap_or_else(|error| panic!("{label}: stable seed lane {lane}: {error:#}"));
        clear_gemma4_self_mounts(g);
        push_current_gemma_output(
            g,
            state.next_token,
            &mut cue_tokens,
            &mut cue_logits,
            &format!("{label} cue lane {lane}"),
        );
        let (anchor, _) = anchor.unwrap_or_else(|| {
            panic!("{label}: stable seed lane {lane} did not capture an anchor")
        });
        assert_eq!(anchor.prompt_len(), boundary, "{label}: anchor boundary");
        anchors.push(anchor);
    }
    let cue = MultiSeqPrefillOutput {
        first_tokens: cue_tokens,
        logits: cue_logits,
    };
    install_cursor(hb, hybrid, &selected, case.initial_cursor, prompt_len);
    assert_canary_unchanged(hybrid, unselected, &canary, &format!("{label} cue"));
    let cue_state = snapshot_selected(hybrid, &selected, prompt_len);

    for (lane, slot) in selected.iter().copied().enumerate() {
        crate::inference::models::gemma4::kv_cache::preflight_gemma_hybrid_slot_anchor_restore(
            hybrid,
            slot,
            &anchors[lane],
            boundary,
        )
        .unwrap_or_else(|error| panic!("{label}: anchor preflight lane {lane}: {error:#}"));
    }
    for (lane, slot) in selected.iter().copied().enumerate() {
        crate::inference::models::gemma4::kv_cache::restore_gemma_hybrid_slot_anchor(
            hybrid,
            slot,
            &anchors[lane],
            boundary,
        )
        .unwrap_or_else(|error| panic!("{label}: anchor restore lane {lane}: {error:#}"));
    }
    // Hybrid anchor restore publishes its own boundary cursor. HB is the
    // sibling scaffold whose host cursor still reflects the post-cue commit.
    // Bring only that sibling back to the same logical boundary.
    for slot in &selected {
        for (layer, buffer) in hb.iter_mut().enumerate() {
            assert_eq!(
                buffer.seq_lens[slot.0 as usize], prompt_len as u32,
                "{label}: HB L{layer} post-cue cursor"
            );
            buffer.seq_lens[slot.0 as usize] = boundary as u32;
        }
        for (layer, buffer) in hybrid.iter().enumerate() {
            assert_eq!(
                buffer.seq_lens[slot.0 as usize], boundary as u32,
                "{label}: hybrid L{layer} restored boundary cursor"
            );
        }
    }
    assert_canary_unchanged(
        hybrid,
        unselected,
        &canary,
        &format!("{label} anchor restore"),
    );
    let restored_boundary_state = snapshot_selected(hybrid, &selected, boundary);

    let mut replay_tokens = Vec::with_capacity(case.width);
    let mut replay_logits = Vec::with_capacity(case.width);
    for (lane, slot) in selected.iter().copied().enumerate() {
        clear_gemma4_self_mounts(g);
        let token = g
            .weights
            .forward_prefill_with_soft_tokens_slot_aware_resume(
                &prompts[lane],
                &[],
                8,
                &mut g.ctx,
                slot,
                hb,
                Some(&mut *hybrid),
                None,
                None,
                boundary,
            )
            .unwrap_or_else(|error| panic!("{label}: anchor cue replay lane {lane}: {error:#}"));
        clear_gemma4_self_mounts(g);
        push_current_gemma_output(
            g,
            token,
            &mut replay_tokens,
            &mut replay_logits,
            &format!("{label} anchor cue replay lane {lane}"),
        );
    }
    let replay = MultiSeqPrefillOutput {
        first_tokens: replay_tokens,
        logits: replay_logits,
    };
    install_cursor(hb, hybrid, &selected, boundary, prompt_len);
    let replay_state = snapshot_selected(hybrid, &selected, prompt_len);
    assert_output_exact(&cue, &replay, &format!("{label} anchor cue replay"));
    assert_state_exact(
        &cue_state,
        &replay_state,
        &format!("{label} anchor cue replay"),
    );
    assert_canary_unchanged(
        hybrid,
        unselected,
        &canary,
        &format!("{label} anchor cue replay"),
    );

    let mut continuation_tokens = Vec::with_capacity(case.width);
    let mut continuation_logits = Vec::with_capacity(case.width);
    for (lane, slot) in selected.iter().copied().enumerate() {
        clear_gemma4_self_mounts(g);
        let mut profile = None;
        let token = g
            .weights
            .forward_decode_slot_aware(
                cue.first_tokens[lane],
                prompt_len,
                &mut g.ctx,
                &mut profile,
                slot,
                hb,
                Some(&mut *hybrid),
                None,
                None,
            )
            .unwrap_or_else(|error| panic!("{label}: continuation lane {lane}: {error:#}"));
        clear_gemma4_self_mounts(g);
        push_current_gemma_output(
            g,
            token,
            &mut continuation_tokens,
            &mut continuation_logits,
            &format!("{label} continuation lane {lane}"),
        );
    }
    let continuation = MultiSeqPrefillOutput {
        first_tokens: continuation_tokens,
        logits: continuation_logits,
    };
    install_cursor(hb, hybrid, &selected, prompt_len, prompt_len + 1);
    assert_canary_unchanged(
        hybrid,
        unselected,
        &canary,
        &format!("{label} continuation"),
    );
    let continuation_state = snapshot_selected(hybrid, &selected, prompt_len + 1);

    WholeRouteReceipt {
        cue,
        cue_state,
        restored_boundary_state,
        continuation,
        continuation_state,
        rectangular_slices,
    }
}

fn assert_whole_route_exact(
    scalar: &WholeRouteReceipt,
    rectangular: &WholeRouteReceipt,
    case: WholeRouteCase,
) {
    let label = case.label();
    assert_eq!(
        scalar.rectangular_slices, 0,
        "{label}: scalar route receipt"
    );
    assert_eq!(
        rectangular.rectangular_slices, 1,
        "{label}: rectangular route receipt"
    );
    assert_output_exact(&scalar.cue, &rectangular.cue, &format!("{label} cue"));
    assert_state_exact(
        &scalar.cue_state,
        &rectangular.cue_state,
        &format!("{label} final cue state"),
    );
    assert_state_exact(
        &scalar.restored_boundary_state,
        &rectangular.restored_boundary_state,
        &format!("{label} restored boundary"),
    );
    assert_output_exact(
        &scalar.continuation,
        &rectangular.continuation,
        &format!("{label} continuation"),
    );
    assert_state_exact(
        &scalar.continuation_state,
        &rectangular.continuation_state,
        &format!("{label} continuation state"),
    );
}

/// Exact product-route authority for the stable-resume composition used by
/// admission: one B2/B4 rectangle retaining the complete scalar boundary
/// width, exact boundary checkpoint, native seven-token cue, and canonical
/// decode resume.
#[test]
#[ignore = "requires a real Gemma GGUF and uncontended Apple Silicon"]
fn gemma_stable_whole_route_rectangles_tail_cue_anchor_and_continuation_are_exact() {
    let Some(model_path) = gated_artifact() else {
        return;
    };
    let mut loaded =
        LoadedModel::load(&load_options(&model_path)).expect("load Gemma whole-route authority");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("expected Gemma GGUF")
    };
    g.provision_multi_seq_kv_for_slot_aware(5)
        .expect("provision whole-route authority slots");
    let mut hb = g.multi_seq_kv.take().expect("HB scaffold");
    let mut hybrid = g
        .multi_seq_kv_hybrid
        .take()
        .expect("hybrid scaffold required");
    assert!(g.multi_seq_kv_dense.is_none(), "dense KV must be disabled");
    assert!(g.multi_seq_kv_mlx.is_none(), "full TQ KV must be disabled");

    let mut cases = Vec::new();
    for width in [2usize, 4] {
        for rows_per_lane in [32usize, 33, 57, 63, 64, 65, 95, 127, 128, 129, 255, 256] {
            cases.push(WholeRouteCase {
                width,
                initial_cursor: 64,
                rows_per_lane,
                crosses_sliding_wrap: false,
            });
        }
    }
    for width in [2usize, 4] {
        for initial_cursor in [500usize, 1_000] {
            cases.push(WholeRouteCase {
                width,
                initial_cursor,
                rows_per_lane: 64,
                crosses_sliding_wrap: false,
            });
        }
    }

    let sliding_capacity = hybrid
        .iter()
        .find(|buffer| buffer.is_sliding)
        .expect("Gemma whole-route authority requires a sliding layer")
        .capacity;
    assert!(
        (80..=4_096).contains(&sliding_capacity),
        "sliding-wrap authority requires an achievable 80..=4096 capacity, got {sliding_capacity}"
    );
    cases.push(WholeRouteCase {
        width: 2,
        initial_cursor: sliding_capacity - 16,
        rows_per_lane: 64,
        crosses_sliding_wrap: true,
    });

    let mut executed = 0usize;
    for case in cases {
        executed += 1;
        let scalar = run_whole_route_arm(g, &mut hb, &mut hybrid, case, false);
        let rectangular = run_whole_route_arm(g, &mut hb, &mut hybrid, case, true);
        assert_whole_route_exact(&scalar, &rectangular, case);
        eprintln!(
            "HF2Q_GEMMA_STABLE_WHOLE_ROUTE_AUTHORITY {{\"width\":{},\"initial_cursor\":{},\"rows_per_lane\":{},\"cue_tokens\":7,\"rectangular_transactions\":1,\"crosses_sliding_wrap\":{},\"cue_logits\":\"byte_exact\",\"boundary_anchor_restore\":\"byte_exact\",\"final_cache\":\"byte_exact\",\"continuation\":\"byte_exact\",\"unselected_full_slot\":\"unchanged\"}}",
            case.width,
            case.initial_cursor,
            case.boundary() - case.initial_cursor,
            case.crosses_sliding_wrap,
        );
    }
    assert_eq!(executed, 29, "whole-route authority must execute every cell");
}
