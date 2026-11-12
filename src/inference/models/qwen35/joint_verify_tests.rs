//! Model-free gates for joint MTP verify/commit (H1/H2) and the DFlash2
//! fixed-width target verifier, plus the harness reused by the Qwen3.8
//! artifact gate.
//!
//! A synthetic Qwen3.5-family model (8 layers: 6 DeltaNet, 2 gated full
//! attention with head_dim 256 on a TQ cache; Q4_K/Q6_K projections, dense
//! FFN and output head, like Qwen3.8) runs the spec-off joint decode step
//! sequentially with the true tokens.
//!
//! The MTP cases exercise cap-8 layouts (4×2, 3+3+2, 2×4, 1×4) in several
//! packing orders, force a first wrong draft at a chosen depth, and commit
//! full, partial, pending-only and empty prefixes. They require bitwise row
//! and cache identity against spec-off.
//!
//! The DFlash2 case exercises a full four-lane × eight-row tile, a lane alone
//! in the same fixed width-32 tile, and a lane permutation. It applies the
//! frozen Q1 Class B logit envelope and argmax gate against spec-off, records
//! exact row/cache/target-hidden cohort invariance.
//!
//!
//! Both routes check retained-prefix commit, attention cursors, TQ codes and
//! norms, recurrent state, untouched slots, and continuation after commit.
//! All tests need Apple Metal (`hf2q_gpu_test_lock`) and skip without a device:
//!
//! ```text
//! cargo test --release --bin hf2q joint_verify_tests
//! ```

use std::hash::{DefaultHasher, Hash, Hasher};

use mlx_native::ops::quantized_matmul_ggml::GgmlType;
use mlx_native::{DType, MlxBuffer, MlxDevice};

use super::forward_gpu::Qwen35JointVerifyLane;
use super::gpu_delta_net::DeltaNetWeightsGpu;
use super::gpu_ffn::DenseFfnWeightsGpuQ;
use super::gpu_full_attn::{upload_f32, FullAttnQGateWeightsGpu, FullAttnWeightsGpu};
use super::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use super::joint_decode_test_support::{next, random_f32};
use super::joint_projection::JointProjectionKernel;
use super::joint_step_options::JointStepOptions;
use super::kv_cache::HybridKvCache;
use super::model::{Qwen35FfnWeights, Qwen35LayerWeights, Qwen35Model};
use super::mtp::{MtpFfnWeightsGpu, MtpFullAttnWeightsGpu, MtpQGateWeightsGpu, MtpWeights};
use super::weight_loader::DenseFfnWeightsQ;
use super::{
    default_layer_types, Qwen35Config, Qwen35JointOutput, Qwen35JointRow, Qwen35LayerKind,
    Qwen35Variant,
};
use crate::inference::spec_decode::dflash2::adapters::DFlash2TargetCapture;
use crate::serve::forward_mlx_shared::MlxQWeight;
use crate::serve::gpu::QuantWeightInfo;
use crate::serve::multi_seq_kv::SlotId;

pub(super) const SLOTS: u32 = 4;
/// Spec-off steps run after the committed prefix to check the state.
const CONTINUE: usize = 2;

/// One spec-off arm: the reference joint step and the verify step share it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Arm {
    pub mode: JointRouteMode,
    pub dense_ffn: JointDenseFfnRoute,
    pub options: JointStepOptions,
}

/// One lane of a case: its slot, its draft count and the accepted rows the
/// commit keeps (`0..=drafts + 1`). The first wrong draft sits at depth
/// `accepted` when `1 <= accepted <= drafts`.
#[derive(Clone, Copy, Debug)]
pub(super) struct LaneCase {
    pub slot: u32,
    pub drafts: usize,
    pub accepted: usize,
}

pub(super) const fn lane(slot: u32, drafts: usize, accepted: usize) -> LaneCase {
    LaneCase {
        slot,
        drafts,
        accepted,
    }
}

/// A model with every slot at a fixed committed cursor and a true token
/// stream per slot.
pub(super) struct Fixture<'a> {
    pub model: &'a Qwen35Model,
    pub arm: Arm,
    /// A cache whose slots sit at `start`; identical bytes on every call.
    pub new_cache: &'a dyn Fn() -> HybridKvCache,
    pub start: [i32; SLOTS as usize],
    /// The true token of `slot` at `position`.
    pub token: &'a dyn Fn(u32, i32) -> u32,
}

/// Bits of one row: logits, then post-norm hidden.
pub(super) type RowBits = (Vec<u32>, Vec<u32>);

fn row_bits(model: &Qwen35Model, output: &super::Qwen35JointRowOutput) -> RowBits {
    let hidden = model.cfg.hidden_size as usize;
    let logits = output.logits.as_ref().expect("row logits");
    let hidden_row = output.nextn_hidden.as_ref().expect("row hidden");
    (
        logits.iter().map(|value| value.to_bits()).collect(),
        hidden_row.as_slice::<f32>().expect("read hidden")[..hidden]
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    )
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// A CPU-visible cache buffer's bytes, without copying.
fn bytes_of(buffer: &MlxBuffer) -> &[u8] {
    match buffer.dtype() {
        DType::U8 => buffer.as_slice::<u8>().expect("read U8 cache"),
        DType::F32 => bytemuck::cast_slice(buffer.as_slice::<f32>().expect("read F32 cache")),
        other => panic!("unexpected cache dtype {other:?}"),
    }
}

/// Bytes of `slot`'s region of a buffer whose outermost axis is the slot.
fn slot_region(bytes: &[u8], slot: u32) -> &[u8] {
    let per_slot = bytes.len() / SLOTS as usize;
    &bytes[slot as usize * per_slot..(slot as usize + 1) * per_slot]
}

/// Bytes of positions `0..cursor` of each KV head in one slot region laid
/// out `[n_kv_heads, max_seq_len, per_position]`.
fn kv_prefix(region: &[u8], n_kv_heads: usize, max_seq_len: usize, cursor: usize) -> Vec<u8> {
    let per_head = region.len() / n_kv_heads;
    let per_position = per_head / max_seq_len;
    (0..n_kv_heads)
        .flat_map(|head| {
            let start = head * per_head;
            region[start..start + cursor * per_position].iter().copied()
        })
        .collect()
}

/// One slot's observable state as the next step sees it: attention cursors
/// and the K/V (TQ codes and norms, or F32) below them, and the current
/// DeltaNet conv and recurrent state (whichever parity holds it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct LaneDigest {
    cursors: Vec<u32>,
    kv: Vec<Vec<u64>>,
    conv: Vec<u64>,
    recurrent: Vec<u64>,
}

pub(super) fn lane_digest(cache: &HybridKvCache, cfg: &Qwen35Config, slot: u32) -> LaneDigest {
    let index = slot as usize;
    let n_kv_heads = cfg.num_key_value_heads as usize;
    let max_seq_len = cache.max_seq_len as usize;
    let mut cursors = Vec::new();
    let mut kv = Vec::new();
    for layer in &cache.full_attn {
        let cursor = layer.current_len[index];
        cursors.push(cursor);
        let buffers: Vec<&MlxBuffer> = match &layer.tq {
            Some(tq) => vec![&tq.k_packed, &tq.k_norms, &tq.v_packed, &tq.v_norms],
            None => [&layer.k, &layer.v].into_iter().flatten().collect(),
        };
        kv.push(
            buffers
                .into_iter()
                .map(|buffer| {
                    hash_bytes(&kv_prefix(
                        slot_region(bytes_of(buffer), slot),
                        n_kv_heads,
                        max_seq_len,
                        cursor as usize,
                    ))
                })
                .collect(),
        );
    }
    let sid = SlotId(slot);
    let conv = cache
        .linear_attn
        .iter()
        .map(|layer| hash_bytes(slot_region(bytes_of(layer.conv_bufs_for_slot(sid).0), slot)))
        .collect();
    let recurrent = cache
        .linear_attn
        .iter()
        .map(|layer| {
            hash_bytes(slot_region(
                bytes_of(layer.recurrent_bufs_for_slot(sid).0),
                slot,
            ))
        })
        .collect();
    LaneDigest {
        cursors,
        kv,
        conv,
        recurrent,
    }
}

fn digests(fixture: &Fixture<'_>, cache: &HybridKvCache) -> Vec<LaneDigest> {
    (0..SLOTS)
        .map(|slot| lane_digest(cache, &fixture.model.cfg, slot))
        .collect()
}

/// One spec-off joint step: `slots` advance by one true token at
/// `positions`; returns their rows.
fn spec_off_step(
    fixture: &Fixture<'_>,
    cache: &mut HybridKvCache,
    slots: &[u32],
    positions: &[i32],
) -> Vec<RowBits> {
    let rows = slots
        .iter()
        .zip(positions)
        .map(|(&slot, &position)| Qwen35JointRow {
            slot_id: SlotId(slot),
            token: (fixture.token)(slot, position),
            position,
            output: Qwen35JointOutput::LogitsAndHidden,
        })
        .collect::<Vec<_>>();
    let arm = fixture.arm;
    fixture
        .model
        .forward_gpu_decode_joint_with_options(&rows, cache, arm.mode, arm.dense_ffn, arm.options)
        .unwrap_or_else(|error| panic!("spec-off step {positions:?}: {error:#}"))
        .iter()
        .map(|output| row_bits(fixture.model, output))
        .collect()
}

/// Per lane: the verify rows' bits (all rows) and the committed digest.
pub(super) struct CaseTrace {
    pub rows: Vec<Vec<RowBits>>,
    pub committed: Vec<LaneDigest>,
}

/// Run one verify case against the spec-off reference; panics on any
/// difference. Returns what the invariance gate compares.
pub(super) fn run_case(fixture: &Fixture<'_>, lanes: &[LaneCase], label: &str) -> CaseTrace {
    let model = fixture.model;
    let vocab = model.cfg.vocab_size;
    let start = |slot: u32| fixture.start[slot as usize];
    let mut slots = lanes.iter().map(|lane| lane.slot).collect::<Vec<_>>();
    slots.sort_unstable();
    let max_rows = lanes
        .iter()
        .map(|lane| lane.drafts + 1)
        .max()
        .expect("a lane");

    // Spec-off reference: the case's slots advance together with the true
    // tokens; rows per position and digests after every step.
    let mut reference = (fixture.new_cache)();
    let initial = digests(fixture, &reference);
    let mut ref_rows = vec![Vec::new(); SLOTS as usize];
    let mut ref_digests = vec![Vec::new(); SLOTS as usize];
    for &slot in &slots {
        ref_digests[slot as usize].push(lane_digest(&reference, &model.cfg, slot));
    }
    for step in 0..max_rows + CONTINUE {
        let positions = slots
            .iter()
            .map(|&slot| start(slot) + step as i32)
            .collect::<Vec<_>>();
        let rows = spec_off_step(fixture, &mut reference, &slots, &positions);
        for (&slot, row) in slots.iter().zip(rows) {
            ref_rows[slot as usize].push(row);
            ref_digests[slot as usize].push(lane_digest(&reference, &model.cfg, slot));
        }
    }

    // Verify: true tokens up to the forced first wrong draft.
    let mut cache = (fixture.new_cache)();
    let tokens = lanes
        .iter()
        .map(|lane| {
            (0..=lane.drafts)
                .map(|row| {
                    let truth = (fixture.token)(lane.slot, start(lane.slot) + row as i32);
                    if row >= 1 && row == lane.accepted {
                        (truth + 1) % vocab
                    } else {
                        truth
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let verify_lanes = lanes
        .iter()
        .zip(&tokens)
        .map(|(lane, tokens)| Qwen35JointVerifyLane {
            slot_id: SlotId(lane.slot),
            position: start(lane.slot),
            tokens,
            output: Qwen35JointOutput::LogitsAndHidden,
        })
        .collect::<Vec<_>>();
    let arm = fixture.arm;
    let verify = model
        .forward_gpu_joint_verify(
            &verify_lanes,
            &mut cache,
            arm.mode,
            arm.dense_ffn,
            arm.options,
        )
        .unwrap_or_else(|error| panic!("{label}: verify: {error:#}"));
    let mut traced = Vec::with_capacity(lanes.len());
    for (lane, rows) in lanes.iter().zip(&verify.rows) {
        assert_eq!(
            rows.len(),
            lane.drafts + 1,
            "{label}: lane {} rows",
            lane.slot
        );
        let mut bits = Vec::with_capacity(rows.len());
        for (row, output) in rows.iter().enumerate() {
            let observed = row_bits(model, output);
            assert!(
                output
                    .logits
                    .as_ref()
                    .expect("logits")
                    .iter()
                    .all(|value| value.is_finite()),
                "{label}: lane {} row {row} logits are not finite",
                lane.slot
            );
            // Rows after the first wrong draft have other inputs.
            if lane.accepted == 0 || row < lane.accepted {
                let expected = &ref_rows[lane.slot as usize][row];
                assert!(
                    observed.0 == expected.0,
                    "{label}: lane {} row {row} logits differ from spec-off",
                    lane.slot
                );
                assert!(
                    observed.1 == expected.1,
                    "{label}: lane {} row {row} hidden differs from spec-off",
                    lane.slot
                );
            }
            bits.push(observed);
        }
        traced.push(bits);
    }

    let accepted = lanes.iter().map(|lane| lane.accepted).collect::<Vec<_>>();
    model
        .commit_joint_verify(&mut cache, verify.pending, &accepted)
        .unwrap_or_else(|error| panic!("{label}: commit: {error:#}"));
    let mut committed = Vec::with_capacity(lanes.len());
    for lane in lanes {
        let digest = lane_digest(&cache, &model.cfg, lane.slot);
        assert!(
            digest == ref_digests[lane.slot as usize][lane.accepted],
            "{label}: lane {} state after accepting {} rows differs from spec-off",
            lane.slot,
            lane.accepted
        );
        committed.push(digest);
    }
    for slot in (0..SLOTS).filter(|slot| !slots.contains(slot)) {
        assert!(
            lane_digest(&cache, &model.cfg, slot) == initial[slot as usize],
            "{label}: slot {slot} outside the step changed"
        );
    }

    // Spec-off continues from the committed state.
    for step in 0..CONTINUE {
        let positions = slots
            .iter()
            .map(|&slot| {
                let accepted = lanes
                    .iter()
                    .find(|lane| lane.slot == slot)
                    .expect("lane")
                    .accepted;
                start(slot) + (accepted + step) as i32
            })
            .collect::<Vec<_>>();
        let rows = spec_off_step(fixture, &mut cache, &slots, &positions);
        for ((&slot, &position), row) in slots.iter().zip(&positions).zip(rows) {
            let offset = (position - start(slot)) as usize;
            assert!(
                row == ref_rows[slot as usize][offset],
                "{label}: slot {slot} spec-off step at {position} after the commit differs"
            );
        }
    }
    for lane in lanes {
        let advanced = lane.accepted + CONTINUE;
        assert!(
            lane_digest(&cache, &model.cfg, lane.slot) == ref_digests[lane.slot as usize][advanced],
            "{label}: lane {} state after continuing differs from spec-off",
            lane.slot
        );
    }
    CaseTrace {
        rows: traced,
        committed,
    }
}

/// The cap8 layouts in several packing orders, with full, partial,
/// pending-only and discarded windows.
pub(super) const M1_CASES: &[(&str, &[LaneCase])] = &[
    (
        "4x2 full/pending/full/discard",
        &[lane(0, 1, 2), lane(1, 1, 1), lane(2, 1, 2), lane(3, 1, 0)],
    ),
    (
        "4x2 permuted",
        &[lane(3, 1, 1), lane(1, 1, 2), lane(0, 1, 1), lane(2, 1, 2)],
    ),
    (
        "4x2 all pending-only",
        &[lane(0, 1, 1), lane(1, 1, 1), lane(2, 1, 1), lane(3, 1, 1)],
    ),
    (
        "3 lanes 2/2/1 full/partial/pending",
        &[lane(0, 2, 3), lane(2, 2, 2), lane(3, 1, 1)],
    ),
    (
        "3 lanes 1/2/2 permuted",
        &[lane(3, 1, 2), lane(0, 2, 1), lane(2, 2, 3)],
    ),
    (
        "3 lanes 2/2/1 all full",
        &[lane(1, 2, 3), lane(0, 2, 3), lane(2, 1, 2)],
    ),
    ("2x4 full/partial", &[lane(1, 3, 4), lane(2, 3, 2)]),
    ("2x4 permuted", &[lane(2, 3, 1), lane(1, 3, 3)]),
    ("1x4 full", &[lane(0, 3, 4)]),
    ("1x4 partial", &[lane(3, 3, 2)]),
    ("1x4 pending-only", &[lane(1, 3, 1)]),
    ("1x4 discard", &[lane(2, 3, 0)]),
];

/// Lane 2's window `drafts = 3, accepted = 2` alone and beside other
/// windows, first or last.
pub(super) const INVARIANCE_CASES: &[&[LaneCase]] = &[
    &[lane(2, 3, 2)],
    &[lane(0, 1, 2), lane(2, 3, 2)],
    &[lane(2, 3, 2), lane(0, 1, 1), lane(1, 1, 0)],
    &[lane(3, 3, 4), lane(2, 3, 2)],
    &[lane(1, 0, 1), lane(2, 3, 2), lane(3, 2, 3)],
];

pub(super) fn check_invariance(fixture: &Fixture<'_>, label: &str) {
    let target = |case: &[LaneCase], trace: &CaseTrace| {
        let index = case
            .iter()
            .position(|lane| lane.slot == 2)
            .expect("target lane");
        (trace.rows[index].clone(), trace.committed[index].clone())
    };
    let alone = run_case(fixture, INVARIANCE_CASES[0], &format!("{label} alone"));
    let (rows, committed) = target(INVARIANCE_CASES[0], &alone);
    for (index, case) in INVARIANCE_CASES.iter().enumerate().skip(1) {
        let trace = run_case(fixture, case, &format!("{label} cohort {index}"));
        let (other_rows, other_committed) = target(case, &trace);
        assert!(
            other_rows == rows,
            "{label}: lane 2 rows differ in cohort {case:?} vs alone"
        );
        assert!(
            other_committed == committed,
            "{label}: lane 2 committed state differs in cohort {case:?} vs alone"
        );
    }
}

// ---------------------------------------------------------------------
// Synthetic Qwen3.5-family model
// ---------------------------------------------------------------------

/// f16 ≈ 0.00195: keeps synthetic K-quant weights near unit scale.
const F16_SMALL: [u8; 2] = 0x1800_u16.to_le_bytes();

/// Valid Q4_K / Q6_K blocks with small scales and random quants.
fn quant_blocks(
    device: &MlxDevice,
    ggml_type: GgmlType,
    rows: usize,
    cols: usize,
    seed: &mut u32,
) -> MlxBuffer {
    let block_bytes = ggml_type.block_bytes() as usize;
    let blocks = rows * cols / ggml_type.block_values() as usize;
    let mut bytes = (0..blocks * block_bytes)
        .map(|_| (next(seed) >> 24) as u8)
        .collect::<Vec<_>>();
    for block in bytes.chunks_exact_mut(block_bytes) {
        match ggml_type {
            GgmlType::Q4_K => {
                block[0..2].copy_from_slice(&F16_SMALL);
                block[2..4].copy_from_slice(&F16_SMALL);
            }
            GgmlType::Q6_K => {
                for scale in &mut block[192..208] {
                    *scale = (next(seed) >> 24) as u8 % 8 + 1;
                }
                block[208..210].copy_from_slice(&F16_SMALL);
            }
            other => panic!("no synthetic layout for {other:?}"),
        }
    }
    let mut buffer = device
        .alloc_buffer(bytes.len(), DType::U8, vec![bytes.len()])
        .expect("allocate synthetic blocks");
    buffer
        .as_mut_slice::<u8>()
        .expect("map synthetic blocks")
        .copy_from_slice(&bytes);
    buffer
}

fn synthetic_cfg() -> Qwen35Config {
    let layer_types = default_layer_types(8, 4);
    assert_eq!(layer_types[3], Qwen35LayerKind::FullAttention);
    Qwen35Config {
        variant: Qwen35Variant::Dense,
        hidden_size: 256,
        num_hidden_layers: 8,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 256,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        layer_types,
        partial_rotary_factor: 0.25,
        rope_theta: 10_000.0,
        rotary_dim: 64,
        mrope_section: [11, 11, 10, 0],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 128,
        vocab_size: 512,
        attn_output_gate: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size: Some(512),
        moe: None,
    }
}

/// The synthetic model under `policy`, its GPU cache primed. With `mtp` it
/// also carries a synthetic MTP block (gated attention, dense Q4_K/Q6_K
/// FFN, Q6_K shared head) and its config declares one MTP layer, so caches
/// allocate the MTP arena.
pub(crate) fn synthetic_model(
    device: &MlxDevice,
    policy: mlx_native::GgmlRoutingPolicy,
    mtp: bool,
) -> Qwen35Model {
    synthetic_model_with_cfg(device, policy, mtp, synthetic_cfg())
}

/// [`synthetic_model`] for another attention geometry of the synthetic
/// config.
fn synthetic_model_with_cfg(
    device: &MlxDevice,
    policy: mlx_native::GgmlRoutingPolicy,
    mtp: bool,
    mut cfg: Qwen35Config,
) -> Qwen35Model {
    cfg.mtp_num_hidden_layers = u32::from(mtp);
    let mut model = Qwen35Model::empty_from_cfg(cfg.clone());
    let mut seed = 0x0A11_7E57_u32;
    let h = cfg.hidden_size as usize;
    let vocab = cfg.vocab_size as usize;
    let inter = cfg.intermediate_size.expect("dense FFN") as usize;
    let (nk, nv) = (
        cfg.linear_num_key_heads as usize,
        cfg.linear_num_value_heads as usize,
    );
    let (dk, dv) = (
        cfg.linear_key_head_dim as usize,
        cfg.linear_value_head_dim as usize,
    );
    let k_width = cfg.linear_conv_kernel_dim as usize;
    let qkv = 2 * nk * dk + nv * dv;
    let z = nv * dv;
    let (nh, nkv, hd) = (
        cfg.num_attention_heads as usize,
        cfg.num_key_value_heads as usize,
        cfg.head_dim as usize,
    );
    model.token_embd = random_f32(&mut seed, vocab * h);
    model.output_weight_native = Some(MlxQWeight {
        buffer: quant_blocks(device, GgmlType::Q6_K, vocab, h, &mut seed),
        info: QuantWeightInfo {
            ggml_dtype: GgmlType::Q6_K,
            rows: vocab,
            cols: h,
        },
        affine: None,
        f16_shadow: None,
        decode_record_q6k_m1: std::sync::OnceLock::new(),
    });
    let layers = std::mem::take(&mut model.layers);
    for (index, layer) in layers.into_iter().enumerate() {
        let ffn = Qwen35FfnWeights::DenseQ(DenseFfnWeightsQ {
            gate_q: quant_blocks(device, GgmlType::Q4_K, inter, h, &mut seed),
            up_q: quant_blocks(device, GgmlType::Q4_K, inter, h, &mut seed),
            down_q: quant_blocks(device, GgmlType::Q6_K, h, inter, &mut seed),
            ggml_type_gate_up: GgmlType::Q4_K,
            ggml_type_down: GgmlType::Q6_K,
            intermediate_size: inter as u32,
            hidden_size: h as u32,
        });
        let native = match layer {
            Qwen35LayerWeights::LinearAttn { mut attn, .. } => {
                attn.attn_norm = vec![1.0; h];
                attn.post_attn_norm = vec![1.0; h];
                attn.ssm_conv1d = random_f32(&mut seed, k_width * qkv);
                attn.ssm_dt_bias = random_f32(&mut seed, nv);
                attn.ssm_a = random_f32(&mut seed, nv)
                    .into_iter()
                    .map(|value| -value.abs() - 0.5)
                    .collect();
                attn.ssm_norm = vec![1.0; dv];
                let mut gpu = DeltaNetWeightsGpu::from_cpu(&attn, device, k_width, qkv)
                    .expect("upload DeltaNet weights");
                let qkv_type = if index % 2 == 0 {
                    GgmlType::Q4_K
                } else {
                    GgmlType::Q6_K
                };
                gpu.attn_qkv = quant_blocks(device, qkv_type, qkv, h, &mut seed);
                gpu.attn_qkv_ggml_type = qkv_type;
                gpu.attn_gate = quant_blocks(device, GgmlType::Q4_K, z, h, &mut seed);
                gpu.attn_gate_ggml_type = GgmlType::Q4_K;
                gpu.ssm_alpha = quant_blocks(device, GgmlType::Q4_K, nv, h, &mut seed);
                gpu.ssm_alpha_ggml_type = GgmlType::Q4_K;
                gpu.ssm_beta = quant_blocks(device, GgmlType::Q4_K, nv, h, &mut seed);
                gpu.ssm_beta_ggml_type = GgmlType::Q4_K;
                gpu.ssm_out = quant_blocks(device, GgmlType::Q4_K, h, z, &mut seed);
                gpu.ssm_out_ggml_type = GgmlType::Q4_K;
                Qwen35LayerWeights::NativeLinearAttn { attn: gpu, ffn }
            }
            Qwen35LayerWeights::FullAttn { mut attn, .. } => {
                attn.attn_norm = vec![1.0; h];
                attn.post_attn_norm = vec![1.0; h];
                attn.attn_q_norm = vec![1.0; hd];
                attn.attn_k_norm = vec![1.0; hd];
                let mut gpu =
                    FullAttnWeightsGpu::from_cpu(&attn, device).expect("upload attention weights");
                gpu.q_gate = FullAttnQGateWeightsGpu::Fused {
                    weight: quant_blocks(device, GgmlType::Q4_K, 2 * nh * hd, h, &mut seed),
                    ggml_type: GgmlType::Q4_K,
                };
                gpu.wk = quant_blocks(device, GgmlType::Q4_K, nkv * hd, h, &mut seed);
                gpu.wk_ggml_type = GgmlType::Q4_K;
                gpu.wv = quant_blocks(device, GgmlType::Q6_K, nkv * hd, h, &mut seed);
                gpu.wv_ggml_type = GgmlType::Q6_K;
                gpu.wo = quant_blocks(device, GgmlType::Q4_K, h, nh * hd, &mut seed);
                gpu.wo_ggml_type = GgmlType::Q4_K;
                Qwen35LayerWeights::NativeFullAttn { attn: gpu, ffn }
            }
            _ => panic!("empty_from_cfg yields float layers"),
        };
        model.layers.push(native);
    }
    if mtp {
        model.mtp = Some(synthetic_mtp(device, &cfg, &mut seed));
    }
    model.bind_ggml_routing_policy(policy);
    model.ensure_gpu_cache_primed().expect("prime GPU cache");
    model
}

fn ones(device: &MlxDevice, n: usize) -> MlxBuffer {
    upload_f32(&vec![1.0; n], device).expect("upload norm weight")
}

/// A synthetic MTP block for [`synthetic_model`].
fn synthetic_mtp(device: &MlxDevice, cfg: &Qwen35Config, seed: &mut u32) -> MtpWeights {
    let h = cfg.hidden_size as usize;
    let vocab = cfg.vocab_size as usize;
    let inter = cfg.intermediate_size.expect("dense FFN") as usize;
    let (nh, nkv, hd) = (
        cfg.num_attention_heads as usize,
        cfg.num_key_value_heads as usize,
        cfg.head_dim as usize,
    );
    MtpWeights {
        layer_index: cfg.num_hidden_layers,
        hidden_size: cfg.hidden_size,
        vocab_size: cfg.vocab_size,
        intermediate_size: inter as u32,
        loaded_tensor_names: Vec::new(),
        enorm: ones(device, h),
        hnorm: ones(device, h),
        eh_proj: quant_blocks(device, GgmlType::Q4_K, h, 2 * h, seed),
        eh_proj_ggml_type: GgmlType::Q4_K,
        embed_tokens: None,
        shared_head_norm: ones(device, h),
        shared_head_head: quant_blocks(device, GgmlType::Q6_K, vocab, h, seed),
        shared_head_head_ggml_type: GgmlType::Q6_K,
        attn: MtpFullAttnWeightsGpu {
            attn_norm: ones(device, h),
            post_attn_norm: ones(device, h),
            q_gate: MtpQGateWeightsGpu::Gated(FullAttnQGateWeightsGpu::Fused {
                weight: quant_blocks(device, GgmlType::Q4_K, 2 * nh * hd, h, seed),
                ggml_type: GgmlType::Q4_K,
            }),
            wk: quant_blocks(device, GgmlType::Q4_K, nkv * hd, h, seed),
            wk_ggml_type: GgmlType::Q4_K,
            wv: quant_blocks(device, GgmlType::Q6_K, nkv * hd, h, seed),
            wv_ggml_type: GgmlType::Q6_K,
            attn_q_norm: ones(device, hd),
            attn_k_norm: ones(device, hd),
            wo: quant_blocks(device, GgmlType::Q4_K, h, nh * hd, seed),
            wo_ggml_type: GgmlType::Q4_K,
        },
        ffn: MtpFfnWeightsGpu::DenseQ {
            weights: DenseFfnWeightsGpuQ {
                gate_q: quant_blocks(device, GgmlType::Q4_K, inter, h, seed),
                up_q: quant_blocks(device, GgmlType::Q4_K, inter, h, seed),
                down_q: quant_blocks(device, GgmlType::Q6_K, h, inter, seed),
                ggml_type_gate_up: GgmlType::Q4_K,
                ggml_type_down: GgmlType::Q6_K,
                intermediate_size: inter as u32,
                hidden_size: cfg.hidden_size,
                repacked_m32: Default::default(),
            },
        },
    }
}

/// Qwen3.8's serve policy: mvN off, `mul_mv_ext` on (the padded mode).
pub(crate) fn padded_policy() -> mlx_native::GgmlRoutingPolicy {
    let mut policy = mlx_native::ggml_routing_policy_from_environment();
    policy.dense_decode_mvn = false;
    policy.dense_decode_mv_ext = true;
    policy
}

/// The operator policy with `mul_mv_ext` off (the exact mode).
pub(crate) fn exact_policy() -> mlx_native::GgmlRoutingPolicy {
    let mut policy = mlx_native::ggml_routing_policy_from_environment();
    policy.dense_decode_mv_ext = false;
    policy
}

/// Prompt lengths per slot, so lanes sit at different positions.
const PREFIX: [i32; SLOTS as usize] = [3, 6, 2, 5];
const CAPACITY: u32 = 64;

fn synthetic_token(slot: u32, position: i32) -> u32 {
    (slot * 131 + position as u32 * 37 + 11) % 512
}

/// A TQ cache with every slot advanced through its prefix by spec-off
/// joint steps of the arm.
fn synthetic_cache(model: &Qwen35Model, arm: Arm) -> HybridKvCache {
    let mut cache = model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&model.cfg, device, CAPACITY, SLOTS, true)
        })
        .expect("allocate four-slot TQ cache");
    let longest = *PREFIX.iter().max().expect("slots");
    for position in 0..longest {
        let rows = (0..SLOTS)
            .filter(|&slot| PREFIX[slot as usize] > position)
            .map(|slot| Qwen35JointRow {
                slot_id: SlotId(slot),
                token: synthetic_token(slot, position),
                position,
                output: Qwen35JointOutput::GreedyToken,
            })
            .collect::<Vec<_>>();
        model
            .forward_gpu_decode_joint_with_options(
                &rows,
                &mut cache,
                arm.mode,
                arm.dense_ffn,
                arm.options,
            )
            .expect("prefix step");
    }
    cache
}

pub(super) fn with_synthetic_fixture(
    policy: mlx_native::GgmlRoutingPolicy,
    arm: Arm,
    run: impl FnOnce(&Fixture<'_>),
) {
    with_synthetic_fixture_cfg(policy, arm, synthetic_cfg(), run);
}

fn with_synthetic_fixture_cfg(
    policy: mlx_native::GgmlRoutingPolicy,
    arm: Arm,
    cfg: Qwen35Config,
    run: impl FnOnce(&Fixture<'_>),
) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let Ok(device) = MlxDevice::new() else {
        return;
    };
    let model = synthetic_model_with_cfg(&device, policy, false, cfg);
    assert_eq!(
        model.joint_decode_mode(),
        Ok(arm.mode),
        "joint mode admitted"
    );
    let new_cache = || synthetic_cache(&model, arm);
    let token = |slot, position| synthetic_token(slot, position);
    run(&Fixture {
        model: &model,
        arm,
        new_cache: &new_cache,
        start: PREFIX,
        token: &token,
    });
}

pub(super) fn dflash_arm() -> Arm {
    Arm {
        mode: JointRouteMode::PaddedMvExt,
        dense_ffn: JointDenseFfnRoute::SharedWeights,
        options: JointStepOptions {
            projection: JointProjectionKernel::Canonical,
            pooled_outputs: true,
            verify_attention_tile: false,
        },
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct DFlashCaseTrace {
    rows: Vec<Vec<RowBits>>,
    committed: Vec<LaneDigest>,
    captured: Vec<Vec<u32>>,
}

pub(super) fn run_dflash_case(
    fixture: &Fixture<'_>,
    lanes: &[LaneCase],
    label: &str,
) -> DFlashCaseTrace {
    let mut cache = (fixture.new_cache)();
    let token_rows = lanes
        .iter()
        .map(|lane| {
            (0..=lane.drafts)
                .map(|row| {
                    (fixture.token)(lane.slot, fixture.start[lane.slot as usize] + row as i32)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let verify_lanes = lanes
        .iter()
        .zip(&token_rows)
        .map(|(lane, tokens)| Qwen35JointVerifyLane {
            slot_id: SlotId(lane.slot),
            position: fixture.start[lane.slot as usize],
            tokens,
            output: Qwen35JointOutput::LogitsAndHidden,
        })
        .collect::<Vec<_>>();
    let capture_layers = [0, fixture.model.cfg.num_hidden_layers as usize - 1];
    let feature_width = fixture.model.cfg.hidden_size as usize * capture_layers.len();
    let (captured, verify_rows) = fixture
        .model
        .with_gpu_cache_mut(|device, _registry| {
            Ok((
                device.alloc_buffer(
                    SLOTS as usize * 8 * feature_width * 4,
                    DType::F32,
                    vec![SLOTS as usize * 8, feature_width],
                )?,
                device.alloc_buffer(SLOTS as usize * 4, DType::U32, vec![SLOTS as usize])?,
            ))
        })
        .unwrap_or_else(|error| panic!("{label}: allocate DFlash2 capture: {error:#}"));
    let capture_lanes = lanes
        .iter()
        .map(|lane| lane.slot as usize)
        .collect::<Vec<_>>();
    let capture = DFlash2TargetCapture {
        layer_ids: &capture_layers,
        output: &captured,
        lane_indices: &capture_lanes,
        verify_rows: &verify_rows,
        lanes: SLOTS as usize,
        rows_per_lane: 8,
        feature_width,
    };
    let arm = fixture.arm;
    let verify = fixture
        .model
        .forward_gpu_dflash2_verify(
            &verify_lanes,
            &mut cache,
            arm.mode,
            arm.dense_ffn,
            arm.options,
            Some(capture),
        )
        .unwrap_or_else(|error| panic!("{label}: DFlash2 verify: {error:#}"));
    let rows = verify
        .rows
        .iter()
        .map(|rows| {
            rows.iter()
                .map(|row| row_bits(fixture.model, row))
                .collect()
        })
        .collect();
    let captured_values = captured
        .as_slice::<f32>()
        .unwrap_or_else(|error| panic!("{label}: read DFlash2 capture: {error}"));
    let captured = lanes
        .iter()
        .map(|lane| {
            let start = lane.slot as usize * 8 * feature_width;
            captured_values[start..start + (lane.drafts + 1) * feature_width]
                .iter()
                .map(|value| value.to_bits())
                .collect()
        })
        .collect();
    fixture
        .model
        .commit_joint_verify(
            &mut cache,
            verify.pending,
            &lanes.iter().map(|lane| lane.accepted).collect::<Vec<_>>(),
        )
        .unwrap_or_else(|error| panic!("{label}: DFlash2 commit: {error:#}"));
    DFlashCaseTrace {
        rows,
        committed: lanes
            .iter()
            .map(|lane| lane_digest(&cache, &fixture.model.cfg, lane.slot))
            .collect(),
        captured,
    }
}

fn argmax_bits(logits: &[u32]) -> usize {
    let mut selected = 0;
    for index in 1..logits.len() {
        if f32::from_bits(logits[index]) > f32::from_bits(logits[selected]) {
            selected = index;
        }
    }
    selected
}

#[test]
fn dflash2_width32_meets_q1_and_is_cohort_invariant() {
    with_synthetic_fixture(padded_policy(), dflash_arm(), |fixture| {
        let full = [lane(0, 7, 0), lane(1, 7, 2), lane(2, 7, 5), lane(3, 7, 8)];

        let mut reference_cache = (fixture.new_cache)();
        let mut reference: [Vec<RowBits>; SLOTS as usize] = std::array::from_fn(|_| Vec::new());
        for step in 0..8 {
            let slots = [0, 1, 2, 3];
            let positions = slots.map(|slot| fixture.start[slot as usize] + step);
            for (slot, row) in slots.into_iter().zip(spec_off_step(
                fixture,
                &mut reference_cache,
                &slots,
                &positions,
            )) {
                reference[slot as usize].push(row);
            }
        }

        let observed = run_dflash_case(fixture, &full, "width32 full");
        for (lane_index, lane) in full.iter().enumerate() {
            for row in 0..=lane.drafts {
                let actual = &observed.rows[lane_index][row];
                let expected = &reference[lane.slot as usize][row];
                assert_eq!(
                    argmax_bits(&actual.0),
                    argmax_bits(&expected.0),
                    "lane {} row {row}: width32 argmax differs from spec-off",
                    lane.slot
                );
                let max_logit_delta = actual
                    .0
                    .iter()
                    .zip(&expected.0)
                    .map(|(&left, &right)| (f32::from_bits(left) - f32::from_bits(right)).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    max_logit_delta <= 0.25,
                    "lane {} row {row}: width32 max logit delta {max_logit_delta} exceeds Q1",
                    lane.slot
                );

                let hidden_metrics = actual.1.iter().zip(&expected.1).fold(
                    (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32),
                    |metrics, (&left, &right)| {
                        let left = f32::from_bits(left);
                        let right = f32::from_bits(right);
                        (
                            metrics.0.max((left - right).abs()),
                            metrics.1 + (left - right).abs(),
                            metrics.2 + left * left,
                            metrics.3 + right * right,
                        )
                    },
                );
                assert!(
                    hidden_metrics.0.is_finite()
                        && hidden_metrics.1.is_finite()
                        && hidden_metrics.2.is_finite()
                        && hidden_metrics.3.is_finite(),
                    "lane {} row {row}: hidden metrics are non-finite",
                    lane.slot
                );
            }
        }

        let alone_case = [lane(2, 7, 5)];
        let alone = run_dflash_case(fixture, &alone_case, "width32 alone");
        assert_eq!(
            alone.rows[0], observed.rows[2],
            "lane 2 rows changed by cohort"
        );
        assert_eq!(
            alone.committed[0], observed.committed[2],
            "lane 2 retained cache changed by cohort"
        );
        assert_eq!(
            alone.captured[0], observed.captured[2],
            "lane 2 target captures changed by cohort"
        );

        let permuted_case = [lane(3, 7, 8), lane(2, 7, 5), lane(1, 7, 2), lane(0, 7, 0)];
        let permuted = run_dflash_case(fixture, &permuted_case, "width32 permuted");
        assert_eq!(
            permuted.rows[1], observed.rows[2],
            "lane 2 rows changed by packing order"
        );
        assert_eq!(
            permuted.committed[1], observed.committed[2],
            "lane 2 retained cache changed by packing order"
        );
        assert_eq!(
            permuted.captured[1], observed.captured[2],
            "lane 2 target captures changed by packing order"
        );
    });
}

/// The DFlash2 arm with the tiled TQ verify attention requested.
fn dflash_tile_arm() -> Arm {
    let mut arm = dflash_arm();
    arm.options.verify_attention_tile = true;
    arm
}

/// The synthetic config with eight query heads on its KV head: a geometry
/// the tiled verify attention compiles (the default two do not).
fn tile_cfg() -> Qwen35Config {
    Qwen35Config {
        num_attention_heads: 8,
        ..synthetic_cfg()
    }
}

/// The tiled verify attention inside the whole DFlash2 verify step, against
/// the stock DFlash2 step of the same model and caches: the step takes the
/// route (its two full-attention layers issue six dispatches each instead
/// of the per-row chain's 225), retains the stock route's cache bytes and
/// cursors through the commit, keeps every row's argmax inside the Q1 logit
/// envelope, and a lane's rows, retained cache and captures do not depend on
/// its cohort or packing order. The synthetic model's residual stream
/// swamps its attention output, so the route's own numerics are gated at
/// the layer (`verify_attn_tile_tests`) and kernel (mlx-native
/// `test_tq_hb_verify_tile`) levels.
#[test]
fn dflash2_tiled_attention_meets_q1_and_is_cohort_invariant() {
    let full = [lane(0, 7, 0), lane(1, 7, 2), lane(2, 7, 5), lane(3, 7, 8)];
    with_synthetic_fixture_cfg(padded_policy(), dflash_tile_arm(), tile_cfg(), |fixture| {
        // The same model and caches under the stock arm (a decode step, and
        // so the cache prefix, never takes the tiled route).
        let before = mlx_native::dispatch_count();
        let stock = run_dflash_case(
            &Fixture {
                model: fixture.model,
                arm: dflash_arm(),
                new_cache: fixture.new_cache,
                start: fixture.start,
                token: fixture.token,
            },
            &full,
            "stock width32",
        );
        let stock_dispatches = mlx_native::dispatch_count() - before;
        let cache = (fixture.new_cache)();
        let available = fixture
            .model
            .with_gpu_cache_mut(|device, registry| {
                super::verify_attn_tile::tq_verify_tile_route(
                    registry,
                    device,
                    &cache.full_attn[0],
                    &[super::joint_decode_core::JointWindow {
                        slot: SlotId(0),
                        start: 0,
                        rows: 8,
                    }],
                    32,
                    fixture.model.cfg.num_attention_heads,
                    fixture.model.cfg.num_key_value_heads,
                    fixture.model.cfg.head_dim,
                    CAPACITY,
                )
            })
            .expect("probe tiled verify attention");
        if !available {
            eprintln!("RECORDED SKIP: tiled verify attention unavailable (no Metal tensor API)");
            return;
        }
        let before = mlx_native::dispatch_count();
        let tiled = run_dflash_case(fixture, &full, "tiled width32");
        let tiled_dispatches = mlx_native::dispatch_count() - before;
        // Per full-attention layer the per-row chain issues 225 dispatches
        // for these 32 rows (six per row, 32 row copies, the gate) and the
        // tiled route six; the model has two such layers.
        assert!(
            stock_dispatches >= tiled_dispatches + 400,
            "the DFlash2 verify step did not take the tiled attention route: \
             {stock_dispatches} stock and {tiled_dispatches} tiled dispatches"
        );
        assert_eq!(
            tiled.committed, stock.committed,
            "the tiled route retained other cache bytes than the stock route"
        );
        let mut worst = 0.0_f32;
        for (lane_index, lane) in full.iter().enumerate() {
            for row in 0..=lane.drafts {
                let actual = &tiled.rows[lane_index][row];
                let expected = &stock.rows[lane_index][row];
                assert_eq!(
                    argmax_bits(&actual.0),
                    argmax_bits(&expected.0),
                    "lane {} row {row}: tiled argmax differs from the stock route",
                    lane.slot
                );
                let delta = actual
                    .0
                    .iter()
                    .zip(&expected.0)
                    .map(|(&left, &right)| (f32::from_bits(left) - f32::from_bits(right)).abs())
                    .fold(0.0_f32, f32::max);
                assert!(
                    delta <= 0.25,
                    "lane {} row {row}: tiled max logit delta {delta} exceeds Q1",
                    lane.slot
                );
                worst = worst.max(delta);
            }
        }
        eprintln!("tiled vs stock DFlash2 verify: max logit delta {worst:.3e}");

        let alone = run_dflash_case(fixture, &[lane(2, 7, 5)], "tiled alone");
        assert_eq!(
            alone.rows[0], tiled.rows[2],
            "lane 2 rows changed by cohort"
        );
        assert_eq!(
            alone.committed[0], tiled.committed[2],
            "lane 2 retained cache changed by cohort"
        );
        assert_eq!(
            alone.captured[0], tiled.captured[2],
            "lane 2 target captures changed by cohort"
        );
        let permuted_case = [lane(3, 7, 8), lane(2, 7, 5), lane(1, 7, 2), lane(0, 7, 0)];
        let permuted = run_dflash_case(fixture, &permuted_case, "tiled permuted");
        assert_eq!(
            permuted.rows[1], tiled.rows[2],
            "lane 2 rows changed by packing order"
        );
        assert_eq!(
            permuted.committed[1], tiled.committed[2],
            "lane 2 retained cache changed by packing order"
        );
        assert_eq!(
            permuted.captured[1], tiled.captured[2],
            "lane 2 target captures changed by packing order"
        );

        // MTP verify rows are the joint decode step's: it refuses the route.
        let mut cache = (fixture.new_cache)();
        let tokens = [(fixture.token)(0, fixture.start[0])];
        let refused = fixture.model.forward_gpu_joint_verify(
            &[Qwen35JointVerifyLane {
                slot_id: SlotId(0),
                position: fixture.start[0],
                tokens: &tokens,
                output: Qwen35JointOutput::LogitsAndHidden,
            }],
            &mut cache,
            fixture.arm.mode,
            fixture.arm.dense_ffn,
            fixture.arm.options,
        );
        assert!(
            refused.is_err(),
            "MTP verify accepted the tiled verify attention"
        );
    });
}

/// Arm M: padded mode, shared-weights FFN, column-invariant MMA products.
pub(super) fn mma_arm() -> Arm {
    Arm {
        mode: JointRouteMode::PaddedMvExt,
        dense_ffn: JointDenseFfnRoute::SharedWeights,
        options: JointStepOptions {
            projection: JointProjectionKernel::ColumnInvariantMma,
            pooled_outputs: true,
            verify_attention_tile: false,
        },
    }
}

#[test]
fn verify_rows_and_commits_match_spec_off_mma_bitwise() {
    with_synthetic_fixture(padded_policy(), mma_arm(), |fixture| {
        for (label, case) in M1_CASES {
            run_case(fixture, case, &format!("mma {label}"));
        }
    });
}

#[test]
fn verify_lane_is_invariant_to_its_cohort_mma_bitwise() {
    with_synthetic_fixture(padded_policy(), mma_arm(), |fixture| {
        check_invariance(fixture, "mma");
    });
}

/// The padded `mul_mv_ext` products (no MMA): verify widths 4 and 8 keep
/// the spec-off width-4 tile.
#[test]
fn verify_rows_and_commits_match_spec_off_mv_ext_bitwise() {
    let arm = Arm {
        mode: JointRouteMode::PaddedMvExt,
        dense_ffn: JointDenseFfnRoute::SharedWeights,
        options: JointStepOptions::default(),
    };
    with_synthetic_fixture(padded_policy(), arm, |fixture| {
        for (label, case) in M1_CASES {
            run_case(fixture, case, &format!("mv_ext {label}"));
        }
        check_invariance(fixture, "mv_ext");
    });
}

/// Exact mode (the Qwen3.6 family route): packed rows are one-row
/// bitwise, at any verify width.
#[test]
fn verify_rows_and_commits_match_spec_off_exact_bitwise() {
    let arm = Arm {
        mode: JointRouteMode::Exact,
        dense_ffn: JointDenseFfnRoute::SerialBuilder,
        options: JointStepOptions::default(),
    };
    with_synthetic_fixture(exact_policy(), arm, |fixture| {
        for (label, case) in M1_CASES {
            run_case(fixture, case, &format!("exact {label}"));
        }
        check_invariance(fixture, "exact");
    });
}

/// Refusals happen before any cache write.
#[test]
fn verify_refuses_bad_windows_before_writing() {
    with_synthetic_fixture(padded_policy(), mma_arm(), |fixture| {
        let model = fixture.model;
        let arm = fixture.arm;
        let mut cache = (fixture.new_cache)();
        let before = digests(fixture, &cache);
        let nine = [1_u32; 9];
        let four = [1_u32; 4];
        let bad: [&[Qwen35JointVerifyLane<'_>]; 5] = [
            &[],
            &[Qwen35JointVerifyLane {
                slot_id: SlotId(0),
                position: PREFIX[0],
                tokens: &nine,
                output: Qwen35JointOutput::Logits,
            }],
            &[Qwen35JointVerifyLane {
                slot_id: SlotId(0),
                position: PREFIX[0] + 1,
                tokens: &four,
                output: Qwen35JointOutput::Logits,
            }],
            &[
                Qwen35JointVerifyLane {
                    slot_id: SlotId(1),
                    position: PREFIX[1],
                    tokens: &four,
                    output: Qwen35JointOutput::Logits,
                },
                Qwen35JointVerifyLane {
                    slot_id: SlotId(1),
                    position: PREFIX[1],
                    tokens: &four,
                    output: Qwen35JointOutput::Logits,
                },
            ],
            &[Qwen35JointVerifyLane {
                slot_id: SlotId(SLOTS),
                position: 0,
                tokens: &four,
                output: Qwen35JointOutput::Logits,
            }],
        ];
        for lanes in bad {
            assert!(model
                .forward_gpu_joint_verify(lanes, &mut cache, arm.mode, arm.dense_ffn, arm.options)
                .is_err());
        }
        assert!(
            digests(fixture, &cache) == before,
            "refusal wrote the cache"
        );
        let verify = model
            .forward_gpu_joint_verify(
                &[Qwen35JointVerifyLane {
                    slot_id: SlotId(2),
                    position: PREFIX[2],
                    tokens: &four,
                    output: Qwen35JointOutput::GreedyToken,
                }],
                &mut cache,
                arm.mode,
                arm.dense_ffn,
                arm.options,
            )
            .expect("valid window");
        assert!(verify.rows[0].iter().all(|row| row.greedy_token.is_some()));
        assert_eq!(verify.pending.lanes().collect::<Vec<_>>(), [(SlotId(2), 4)]);
        // A commit needs one accept count per lane, each within the window.
        assert!(model
            .commit_joint_verify(&mut (fixture.new_cache)(), verify.pending, &[5])
            .is_err());
    });
}
