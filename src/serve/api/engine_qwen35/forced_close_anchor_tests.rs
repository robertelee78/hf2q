//! ADR-063 Q4-QWEN36 forced-close anchor (design option B), decode side.
//!
//! - G1: `Qwen35ThinkingBudgetState::force_imminent`, the boundary planner,
//!   the per-route lookahead and the default-off switch.
//! - G4: a weightless decode state drives ordinary ticks (serial and joint
//!   preparation) against a real `HybridKvCache`. The forward is modelled by
//!   writing the fed token's K row and a cursor-derived DeltaNet state, so the
//!   anchor's cursor, DeltaNet bytes and the untouched rows below it are
//!   checked after more rows are appended and the anchor is restored. With
//!   the switch off no anchor is taken and decode is unchanged.

use std::sync::Arc;
use std::time::Duration;

use mlx_native::MlxDevice;

use super::joint_decode::Qwen35JointTarget;
use super::speculation_qualification_tests::loaded_model;
use super::{
    parse_qwen_forced_close_anchor, qwen35_forced_close_anchor_boundary,
    qwen35_forced_close_lookahead, Qwen35DecodeState, Qwen35LoadedModel, Qwen35ThinkingBudgetState,
    Qwen35TickKind, K3_VERIFY_ROWS, QWEN35_FORCED_CLOSE_LOOKAHEAD_ORDINARY,
};
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::{default_layer_types, Qwen35Config, Qwen35Variant};
use crate::serve::api::engine::SamplingParams;
use crate::serve::api::qwen35_speculation::QwenSpeculationPolicy;
use crate::serve::multi_seq_kv::{MultiSeqKvCache, SlotId};

const PROMPT: [u32; 6] = [3, 9, 27, 17, 5, 41];
const LIMIT: usize = 8;
const FORCED: [u32; 3] = [50, 51, 52];
const CLOSE: u32 = 52;
const MODEL_TOKEN: u32 = 7;
const SLOT: SlotId = SlotId(1);
const PEER: SlotId = SlotId(0);
const MAX_SEQ: u32 = 64;

fn budget_params(limit: usize, max_tokens: usize) -> SamplingParams {
    SamplingParams {
        max_tokens,
        reasoning_forced_open: true,
        thinking_token_budget: Some(limit),
        reasoning_end_tokens: Some(Arc::new(FORCED.to_vec())),
        reasoning_close_tokens: Some(Arc::new(vec![CLOSE])),
        ..SamplingParams::default()
    }
}

fn budget(limit: usize) -> Qwen35ThinkingBudgetState {
    Qwen35ThinkingBudgetState::from_params(&budget_params(limit, 64)).expect("budget state")
}

#[test]
fn g1_force_imminent_tracks_the_budget_window() {
    let mut state = budget(LIMIT);
    let mut generated = Vec::new();
    for _ in 0..LIMIT - 2 {
        generated.push(MODEL_TOKEN);
        state.observe_generated(&generated, false);
    }
    // reasoning = limit - 2: imminent for an ordinary step's lookahead (2),
    // not for a zero lookahead.
    assert!(state.force_imminent(QWEN35_FORCED_CLOSE_LOOKAHEAD_ORDINARY));
    assert!(!state.force_imminent(1));
    assert!(!state.force_imminent(0));
    generated.push(MODEL_TOKEN);
    state.observe_generated(&generated, false);
    generated.push(MODEL_TOKEN);
    state.observe_generated(&generated, false);
    assert!(
        state.force_imminent(0),
        "at the limit the force is imminent"
    );
    // Once forcing has begun the window is closed.
    assert_eq!(state.next_forced_token(), Some((FORCED[0], true)));
    assert!(!state.force_imminent(usize::MAX));
}

#[test]
fn g1_force_is_not_imminent_after_a_natural_close_or_far_from_the_limit() {
    let mut state = budget(LIMIT);
    assert!(!state.force_imminent(QWEN35_FORCED_CLOSE_LOOKAHEAD_ORDINARY));
    assert!(
        state.force_imminent(LIMIT),
        "a lookahead covering the limit"
    );
    state.observe_generated(&[CLOSE], false);
    assert!(state.closed);
    assert!(!state.force_imminent(usize::MAX), "already closed");
    assert!(!state.was_forced_closed());

    let mut tool = budget(LIMIT);
    tool.observe_generated(&[MODEL_TOKEN], true);
    assert!(
        !tool.force_imminent(usize::MAX),
        "a tool call closes reasoning"
    );
}

#[test]
fn g1_boundary_excludes_the_newest_token_and_requires_the_exact_cursor() {
    // prompt 6, generated 5: the newest token is the step's input.
    assert_eq!(
        qwen35_forced_close_anchor_boundary(true, true, 6, 5, 0, 10),
        Some(10)
    );
    // The graft shifts the physical cursor, not the token boundary.
    assert_eq!(
        qwen35_forced_close_anchor_boundary(true, true, 6, 5, 4, 14),
        Some(10)
    );
    // A cache ahead of (speculative rows) or behind the ledger takes nothing.
    assert_eq!(
        qwen35_forced_close_anchor_boundary(true, true, 6, 5, 0, 11),
        None
    );
    assert_eq!(
        qwen35_forced_close_anchor_boundary(true, true, 6, 5, 0, 9),
        None
    );
    assert_eq!(
        qwen35_forced_close_anchor_boundary(true, true, 6, 0, 0, 5),
        None
    );
    // Switch off or no imminent force: never.
    assert_eq!(
        qwen35_forced_close_anchor_boundary(false, true, 6, 5, 0, 10),
        None
    );
    assert_eq!(
        qwen35_forced_close_anchor_boundary(true, false, 6, 5, 0, 10),
        None
    );
}

#[test]
fn g1_lookahead_covers_each_route_and_drains_never_snapshot() {
    assert_eq!(qwen35_forced_close_lookahead(&Qwen35TickKind::Commit), None);
    assert_eq!(
        qwen35_forced_close_lookahead(&Qwen35TickKind::Ordinary),
        Some(2)
    );
    assert_eq!(
        qwen35_forced_close_lookahead(&Qwen35TickKind::K3),
        Some(K3_VERIFY_ROWS + 1)
    );
    assert_eq!(
        qwen35_forced_close_lookahead(&Qwen35TickKind::History(vec![1, 2, 3])),
        Some(5)
    );
}

#[test]
fn g1_switch_defaults_off() {
    assert!(!parse_qwen_forced_close_anchor(None));
    assert!(!parse_qwen_forced_close_anchor(Some("")));
    assert!(!parse_qwen_forced_close_anchor(Some("0")));
    assert!(!parse_qwen_forced_close_anchor(Some("off")));
    assert!(
        !parse_qwen_forced_close_anchor(Some("maybe")),
        "invalid keeps off"
    );
    assert!(parse_qwen_forced_close_anchor(Some("1")));
    assert!(parse_qwen_forced_close_anchor(Some("on")));
    assert!(parse_qwen_forced_close_anchor(Some("TRUE")));
}

fn tiny_cfg() -> Qwen35Config {
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
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size: Some(128),
        moe: None,
    }
}

/// F32 K row `(slot, head, pos)` of every full-attention layer.
fn k_row(cache: &mut HybridKvCache, slot: SlotId, pos: usize) -> Vec<Vec<f32>> {
    let (n_seqs, max_seq) = (cache.n_seqs as usize, cache.max_seq_len as usize);
    cache
        .full_attn
        .iter_mut()
        .map(|full| {
            let k = full.k.as_mut().expect("F32 K");
            let per_seq = k.element_count() / n_seqs;
            let head_dim = per_seq / max_seq / 2;
            let floats = k.as_slice::<f32>().expect("K floats");
            let mut row = Vec::new();
            for head in 0..2 {
                let start = slot.0 as usize * per_seq + (head * max_seq + pos) * head_dim;
                row.extend_from_slice(&floats[start..start + head_dim]);
            }
            row
        })
        .collect()
}

/// The modelled forward of one input row: append it, write its K row, and
/// set the slot's DeltaNet state to a value derived from the new cursor.
fn forward_row(cache: &mut HybridKvCache, slot: SlotId) {
    cache.append_for_seq(slot, 1).expect("append row");
    let cursor = cache.seq_len(slot).expect("cursor") as usize;
    write_rows(cache, slot, cursor - 1, cursor);
    write_delta_net(cache, slot, cursor);
}

fn write_rows(cache: &mut HybridKvCache, slot: SlotId, from: usize, to: usize) {
    let (n_seqs, max_seq) = (cache.n_seqs as usize, cache.max_seq_len as usize);
    for full in &mut cache.full_attn {
        let k = full.k.as_mut().expect("F32 K");
        let per_seq = k.element_count() / n_seqs;
        let head_dim = per_seq / max_seq / 2;
        let floats = k.as_mut_slice::<f32>().expect("K floats");
        for pos in from..to {
            for head in 0..2 {
                let start = slot.0 as usize * per_seq + (head * max_seq + pos) * head_dim;
                floats[start..start + head_dim].fill((slot.0 as usize * 1000 + pos + 1) as f32);
            }
        }
    }
}

fn delta_net_byte(cursor: usize) -> u8 {
    (cursor % 251) as u8
}

fn write_delta_net(cache: &mut HybridKvCache, slot: SlotId, cursor: usize) {
    let n_seqs = cache.n_seqs as usize;
    for linear in &mut cache.linear_attn {
        linear.pp_flipped[slot.0 as usize] = false;
        for buffer in [&mut linear.conv_state, &mut linear.recurrent] {
            let region = buffer.byte_len() / n_seqs;
            let start = slot.0 as usize * region;
            buffer.as_mut_slice::<u8>().expect("state bytes")[start..start + region]
                .fill(delta_net_byte(cursor));
        }
    }
}

fn delta_net_bytes_equal(cache: &HybridKvCache, slot: SlotId, byte: u8) -> bool {
    let n_seqs = cache.n_seqs as usize;
    cache.linear_attn.iter().all(|linear| {
        !linear.pp_flipped[slot.0 as usize]
            && [&linear.conv_state, &linear.recurrent]
                .iter()
                .all(|buffer| {
                    let region = buffer.byte_len() / n_seqs;
                    let start = slot.0 as usize * region;
                    buffer.as_slice::<u8>().expect("state bytes")[start..start + region]
                        .iter()
                        .all(|&value| value == byte)
                })
    })
}

/// Prefill the prompt on `SLOT` (and an unrelated peer) and seed the state.
fn admit(
    qwen: &Qwen35LoadedModel,
    cache: &mut HybridKvCache,
    params: &SamplingParams,
    enabled: bool,
) -> Qwen35DecodeState {
    cache.append_for_seq(PEER, 11).expect("peer rows");
    write_rows(cache, PEER, 0, 11);
    write_delta_net(cache, PEER, 11);
    cache
        .append_for_seq(SLOT, PROMPT.len() as u32)
        .expect("prompt rows");
    write_rows(cache, SLOT, 0, PROMPT.len());
    write_delta_net(cache, SLOT, PROMPT.len());
    let mut state = Qwen35DecodeState::from_prefill_logits(
        qwen,
        PROMPT.to_vec(),
        params.clone(),
        None,
        SLOT,
        0,
        &logits_for(&[]),
        Duration::ZERO,
        PROMPT.len(),
        None,
    )
    .expect("seed decode state");
    state.forced_close_anchor_enabled = enabled;
    state
}

/// The toy model always picks `MODEL_TOKEN`, except `CLOSE` at the
/// generated index `natural_close_at`.
fn logits_for(generated: &[u32]) -> Vec<f32> {
    logits_with_close(generated, None)
}

fn logits_with_close(generated: &[u32], natural_close_at: Option<usize>) -> Vec<f32> {
    let mut logits = vec![0.0f32; 64];
    let token = if natural_close_at == Some(generated.len()) {
        CLOSE
    } else {
        MODEL_TOKEN
    };
    logits[token as usize] = 10.0;
    logits
}

#[derive(Clone, Copy)]
enum Route {
    Serial,
    Joint,
}

/// One ordinary decode tick: the start-of-step capture (serial
/// `decode_tick` or joint `prepare_joint_tick`), the modelled forward, and
/// production selection and commit. Returns whether the request finished.
fn tick(
    qwen: &Qwen35LoadedModel,
    cache: &mut HybridKvCache,
    state: &mut Qwen35DecodeState,
    route: Route,
    natural_close_at: Option<usize>,
) -> bool {
    let forced = match route {
        Route::Serial => {
            state.capture_forced_close_anchor(cache, QWEN35_FORCED_CLOSE_LOOKAHEAD_ORDINARY);
            state.next_ordinary_forced_token()
        }
        Route::Joint => match state.prepare_joint_tick(cache).expect("joint lane").target {
            Qwen35JointTarget::Ordinary { forced_token, .. } => forced_token,
            Qwen35JointTarget::Coherent(_) => panic!("no MTP state in this fixture"),
        },
    };
    forward_row(cache, SLOT);
    let logits = logits_with_close(&state.generated_tokens, natural_close_at);
    let token = state
        .select_ordinary_logits_token(qwen, logits, forced)
        .expect("ordinary decision");
    state
        .commit_ordinary_target_token(qwen, token, Duration::ZERO)
        .expect("commit")
        .finished
}

struct Run {
    generated: Vec<u32>,
    cursor: usize,
    anchor_tokens: Option<Vec<u32>>,
}

fn run(
    route: Route,
    enabled: bool,
    natural_close_at: Option<usize>,
) -> (Run, HybridKvCache, Qwen35DecodeState) {
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Off);
    let mut cache = HybridKvCache::new(&tiny_cfg(), &device, MAX_SEQ, 2).expect("cache");
    let params = budget_params(LIMIT, 20);
    let mut state = admit(&qwen, &mut cache, &params, enabled);
    while !tick(&qwen, &mut cache, &mut state, route, natural_close_at) {}
    let run = Run {
        generated: state.generated_tokens.clone(),
        cursor: cache.seq_len(SLOT).expect("cursor") as usize,
        anchor_tokens: state
            .forced_close_anchor
            .as_ref()
            .map(|anchor| anchor.prompt_tokens.clone()),
    };
    (run, cache, state)
}

#[test]
fn g4_anchor_at_mid_decode_cursor_restores_rows_cursor_and_delta_net_bytes() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let (run, mut cache, mut state) = run(Route::Serial, true, None);

    // The seed and the first LIMIT - 1 ticks are model tokens; the force
    // emits FORCED; decode then continues to max_tokens.
    let first_forced = run
        .generated
        .iter()
        .position(|&token| token == FORCED[0])
        .expect("forced close happened");
    assert_eq!(first_forced, LIMIT);
    assert_eq!(&run.generated[LIMIT..LIMIT + FORCED.len()], &FORCED);
    assert_eq!(run.generated.len(), 20);

    // Captured at the first tick with reasoning + 2 >= LIMIT: generated =
    // LIMIT - 2, cursor = prompt + LIMIT - 3. It stops before the model's
    // last token (generated index LIMIT - 1) and every forced token.
    let anchor_tokens = run.anchor_tokens.expect("anchor captured");
    let boundary = PROMPT.len() + LIMIT - 3;
    assert_eq!(anchor_tokens.len(), boundary);
    assert!(anchor_tokens.len() < PROMPT.len() + LIMIT - 1);
    assert_eq!(&anchor_tokens[..PROMPT.len()], &PROMPT);
    assert_eq!(&anchor_tokens[PROMPT.len()..], &run.generated[..LIMIT - 3]);

    let anchor = state
        .take_forced_close_anchor()
        .expect("force-closed request publishes it");
    assert_eq!(anchor.kv.prompt_len(), boundary);
    assert!(anchor.kv.total_bytes() > 0);
    assert!(
        run.cursor > boundary,
        "decode appended rows past the anchor"
    );
    assert!(cache.slot_anchor_cursors_cover(SLOT, &anchor.kv));
    let rows_below: Vec<_> = (0..boundary)
        .map(|pos| k_row(&mut cache, SLOT, pos))
        .collect();
    let peer_rows: Vec<_> = (0..11).map(|pos| k_row(&mut cache, PEER, pos)).collect();

    cache
        .restore_slot_anchor(SLOT, &anchor.kv)
        .expect("restore");
    assert_eq!(cache.seq_len(SLOT).unwrap() as usize, boundary);
    assert!(
        delta_net_bytes_equal(&cache, SLOT, delta_net_byte(boundary)),
        "DeltaNet state must be the bytes at the anchor cursor"
    );
    for (pos, row) in rows_below.iter().enumerate() {
        assert_eq!(
            &k_row(&mut cache, SLOT, pos),
            row,
            "row {pos} below the anchor changed"
        );
        assert!(row
            .iter()
            .flatten()
            .all(|&v| v == (SLOT.0 as usize * 1000 + pos + 1) as f32));
    }
    assert_eq!(cache.seq_len(PEER).unwrap(), 11, "peer cursor");
    assert!(
        delta_net_bytes_equal(&cache, PEER, delta_net_byte(11)),
        "peer state"
    );
    for (pos, row) in peer_rows.iter().enumerate() {
        assert_eq!(&k_row(&mut cache, PEER, pos), row, "peer row {pos}");
    }
}

#[test]
fn g4_joint_preparation_captures_the_same_anchor_as_the_serial_tick() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let (serial, serial_cache, mut serial_state) = run(Route::Serial, true, None);
    let (joint, joint_cache, mut joint_state) = run(Route::Joint, true, None);
    assert_eq!(serial.generated, joint.generated);
    assert_eq!(serial.cursor, joint.cursor);
    assert_eq!(serial.anchor_tokens, joint.anchor_tokens);
    let (serial_anchor, joint_anchor) = (
        serial_state
            .take_forced_close_anchor()
            .expect("serial anchor"),
        joint_state
            .take_forced_close_anchor()
            .expect("joint anchor"),
    );
    assert_eq!(serial_anchor.kv.prompt_len(), joint_anchor.kv.prompt_len());
    assert_eq!(
        serial_anchor.kv.total_bytes(),
        joint_anchor.kv.total_bytes()
    );
    drop((serial_cache, joint_cache));
}

#[test]
fn g4_switch_off_takes_no_anchor_and_decodes_identically() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    for route in [Route::Serial, Route::Joint] {
        let (on, _, _) = run(route, true, None);
        let (off, _, mut off_state) = run(route, false, None);
        assert!(off.anchor_tokens.is_none(), "switch off took an anchor");
        assert!(off_state.take_forced_close_anchor().is_none());
        assert_eq!(on.generated, off.generated, "switch changed decode");
        assert_eq!(on.cursor, off.cursor, "switch changed the cache cursor");
    }
}

#[test]
fn g4_natural_close_inside_the_margin_discards_the_anchor() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    // The model closes on its own at generated index LIMIT - 1, after the
    // capture at LIMIT - 2 and before the force.
    let (run, _, mut state) = run(Route::Serial, true, Some(LIMIT - 1));
    assert_eq!(run.generated[LIMIT - 1], CLOSE);
    assert!(!run.generated.contains(&FORCED[0]), "no forced close");
    assert!(run.anchor_tokens.is_some(), "captured inside the margin");
    assert!(
        state.take_forced_close_anchor().is_none(),
        "a naturally closed request must not publish the anchor"
    );
}
