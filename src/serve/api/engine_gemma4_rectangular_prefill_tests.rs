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

use crate::inference::models::gemma4::kv_cache::{MultiSeqHbKvBuffers, MultiSeqHybridKvBuffers};
use crate::inference::models::gemma4::model::MultiSeqPrefillOutput;
use sha2::{Digest, Sha256};

const TEST_NAME: &str = "gemma_live_rectangular_b2_b4_state_and_continuation_are_exact";

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

struct ArmReceipt {
    boundary: MultiSeqPrefillOutput,
    boundary_state: Vec<SlotImage>,
    continuation: Vec<MultiSeqPrefillOutput>,
    continuation_state: Vec<SlotImage>,
    boundary_micros: u128,
    continuation_micros: u128,
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

fn prompts(width: usize) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
    let prefixes = (0..width)
        .map(|lane| {
            (0..64u32)
                .map(|position| 101 + lane as u32 * 997 + position * 17)
                .collect()
        })
        .collect();
    let suffixes = (0..width)
        .map(|lane| {
            (0..32u32)
                .map(|position| 8_003 + lane as u32 * 991 + position * 19)
                .collect()
        })
        .collect();
    (prefixes, suffixes)
}

fn run_arm(model_path: &Path, width: usize, rectangular: bool) -> ArmReceipt {
    let selected: Vec<SlotId> = match width {
        2 => vec![SlotId(0), SlotId(2)],
        4 => vec![SlotId(0), SlotId(2), SlotId(3), SlotId(4)],
        _ => panic!("authority only covers B2/B4"),
    };
    let unselected = SlotId(1);
    let n_seqs = selected.iter().map(|slot| slot.0).max().unwrap() + 1;
    let (prefixes, suffixes) = prompts(width);
    let continuation_tokens: Vec<u32> = (0..width).map(|lane| 20_003 + lane as u32 * 101).collect();

    let mut loaded =
        LoadedModel::load(&load_options(model_path)).expect("load Gemma authority arm");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("expected Gemma GGUF")
    };
    g.provision_multi_seq_kv_for_slot_aware(n_seqs)
        .expect("provision authority slots");
    let mut hb = g.multi_seq_kv.take().expect("HB scaffold");
    let mut hybrid = g
        .multi_seq_kv_hybrid
        .take()
        .expect("hybrid scaffold required");
    assert!(g.multi_seq_kv_dense.is_none(), "dense KV must be disabled");
    assert!(g.multi_seq_kv_mlx.is_none(), "full TQ KV must be disabled");
    let canary = seed_canary(&mut hybrid, unselected);

    for (lane, slot) in selected.iter().copied().enumerate() {
        clear_gemma4_self_mounts(g);
        g.weights
            .forward_prefill_with_soft_tokens_slot_aware(
                &prefixes[lane],
                &[],
                8,
                &mut g.ctx,
                slot,
                &mut hb,
                Some(&mut hybrid),
                None,
                None,
            )
            .unwrap_or_else(|error| panic!("prefix lane {lane}: {error:#}"));
    }
    clear_gemma4_self_mounts(g);
    install_cursor(&mut hb, &mut hybrid, &selected, 0, 64);

    let boundary_started = Instant::now();
    let boundary = if rectangular {
        let seqs: Vec<_> = selected
            .iter()
            .copied()
            .zip(suffixes.iter().cloned())
            .map(|(slot, suffix)| (suffix, slot, 64usize))
            .collect();
        g.weights
            .forward_prefill_batched_multi_seq_live(&seqs, &hybrid, 8, &mut g.ctx)
            .expect("rectangular live suffix")
    } else {
        let mut first_tokens = Vec::with_capacity(width);
        let mut logits = Vec::with_capacity(width);
        for (lane, slot) in selected.iter().copied().enumerate() {
            let output = g
                .weights
                .forward_prefill_batched_multi_seq_live(
                    &[(suffixes[lane].clone(), slot, 64)],
                    &hybrid,
                    8,
                    &mut g.ctx,
                )
                .unwrap_or_else(|error| panic!("scalar suffix lane {lane}: {error:#}"));
            first_tokens.extend(output.first_tokens);
            logits.extend(output.logits);
        }
        MultiSeqPrefillOutput {
            first_tokens,
            logits,
        }
    };
    let boundary_micros = boundary_started.elapsed().as_micros();
    clear_gemma4_self_mounts(g);
    install_cursor(&mut hb, &mut hybrid, &selected, 64, 96);
    assert_canary_unchanged(&hybrid, unselected, &canary, "post-suffix");
    let boundary_state = snapshot_selected(&hybrid, &selected, 96);

    let continuation_started = Instant::now();
    let mut continuation = Vec::with_capacity(width);
    for (lane, slot) in selected.iter().copied().enumerate() {
        let output = g
            .weights
            .forward_prefill_batched_multi_seq_live(
                &[(vec![continuation_tokens[lane]], slot, 96)],
                &hybrid,
                8,
                &mut g.ctx,
            )
            .unwrap_or_else(|error| panic!("continuation lane {lane}: {error:#}"));
        continuation.push(output);
    }
    let continuation_micros = continuation_started.elapsed().as_micros();
    clear_gemma4_self_mounts(g);
    install_cursor(&mut hb, &mut hybrid, &selected, 96, 97);
    assert_canary_unchanged(&hybrid, unselected, &canary, "post-continuation");
    let continuation_state = snapshot_selected(&hybrid, &selected, 97);

    ArmReceipt {
        boundary,
        boundary_state,
        continuation,
        continuation_state,
        boundary_micros,
        continuation_micros,
    }
}

/// Real-model authority for both admitted rectangular widths at M=32.
///
/// This is ignored because it loads the configured GGUF and executes Metal.
/// The gate deliberately compares fresh scalar and rectangular arms and then
/// resumes every selected slot by one identical token, preventing a logit-only
/// pass from hiding corrupted persistent state.
#[test]
#[ignore = "requires a real Gemma GGUF and uncontended Apple Silicon"]
fn gemma_live_rectangular_b2_b4_state_and_continuation_are_exact() {
    let Some(model_path) = gated_artifact() else {
        return;
    };
    for width in [2usize, 4] {
        let scalar = run_arm(&model_path, width, false);
        let rectangular = run_arm(&model_path, width, true);
        assert_output_exact(&scalar.boundary, &rectangular.boundary, "post-suffix");
        assert_state_exact(
            &scalar.boundary_state,
            &rectangular.boundary_state,
            "post-suffix",
        );
        assert_eq!(
            scalar.continuation.len(),
            rectangular.continuation.len(),
            "continuation lane count"
        );
        for lane in 0..width {
            assert_output_exact(
                &scalar.continuation[lane],
                &rectangular.continuation[lane],
                &format!("continuation lane {lane}"),
            );
        }
        assert_state_exact(
            &scalar.continuation_state,
            &rectangular.continuation_state,
            "post-continuation",
        );
        eprintln!(
            "HF2Q_GEMMA_RECTANGULAR_AUTHORITY {{\"width\":{width},\"suffix_tokens_per_lane\":32,\"scalar_suffix_us\":{},\"rectangular_suffix_us\":{},\"scalar_continuation_us\":{},\"rectangular_continuation_us\":{},\"selected_state\":\"byte_exact\",\"unselected_full_slot\":\"unchanged\"}}",
            scalar.boundary_micros,
            rectangular.boundary_micros,
            scalar.continuation_micros,
            rectangular.continuation_micros,
        );
    }
}
