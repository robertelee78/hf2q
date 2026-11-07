//! ADR-063 Q4-QWEN36 forced-close anchor, real-model gates G5 and G6
//! (ignored; they load the artifact and need a GPU grant).
//!
//! - G5, same-route twin: twin A decodes the forced-close request only up to
//!   the anchor boundary P (`max_tokens`) and serves the follow-up R2 by live
//!   reuse of its slot; twin B decodes the whole forced-close request with
//!   `HF2Q_QWEN_FORCED_CLOSE_ANCHOR` on, restores the anchor and serves R2
//!   from it. R2's first-chunk logits, every decode tick, the final slot state
//!   and the probe logits must be bitwise equal.
//! - G6: twin B twice is bitwise equal (determinism); the forced-close request
//!   decoded through joint steps beside a peer captures the same anchor
//!   boundary and, under the exact joint route, the same bytes and follow-up
//!   as the per-slot route; the joint run is itself repeatable.
//!
//! Speculation is off (the Qwen3.6 SlotAware serving default with joint
//! decode and four slots), so every tick is an ordinary target decision.
//!
//! ```text
//! HF2Q_QWEN36_GGUF=/abs/qwen36.gguf \
//!   cargo test --release --bin hf2q qwen36_forced_close_anchor_ \
//!   -- --ignored --nocapture --test-threads=1
//! ```

use std::sync::Arc;

use super::joint_decode::Qwen35JointTarget;
use super::round_slot_independence_tests::{
    assert_same, bits, cursors, load_gate_artifact, new_cache, slot_state, tick_target,
    TargetTrace, Tick, PREFILL_CHUNK,
};
use super::{
    Qwen35DecodeState, Qwen35ForcedCloseAnchor, Qwen35LoadedModel, Qwen35PrefillAdvance,
    Qwen35PrefillState,
};
use crate::inference::models::qwen35::joint_decode_routes::{
    select_dense_ffn_route, JointRouteMode,
};
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::serve::api::engine::{LoadedModel, SamplingParams};
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::qwen35_speculation::{QwenSpeculationController, QwenSpeculationPolicy};
use crate::serve::multi_seq_kv::SlotId;

const ENV: &str = "HF2Q_QWEN36_GGUF";
const SLOT: SlotId = SlotId(2);
const PEER: SlotId = SlotId(0);
/// Small enough that the model is still reasoning when it is reached.
const BUDGET: usize = 64;
const FIRST_MAX_TOKENS: usize = 160;
const FOLLOW_UP_MAX_TOKENS: usize = 48;
const MAX_TICKS: usize = 256;
const TASK: &str = "Write a Rust function `parse_port(s: &str) -> Result<u16, String>` that \
    rejects zero, values above 65535, negative numbers and non-numeric input. Think through \
    every edge case carefully before you answer.";
const PEER_TASK: &str = "List three prime numbers and explain why each is prime.";
/// The client's re-render after the anchor: a reasoning tail that differs
/// from the generated one, the close, the answer, a tool result and the
/// next assistant cue.
const RE_RENDER_TAIL: &str = "\n</think>\n\nHere is the function.<|im_end|>\n<|im_start|>user\n\
    <tool_response>\ntests passed: 4 of 5\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n";

fn encode(qwen: &Qwen35LoadedModel, text: &str) -> Vec<u32> {
    qwen.tokenizer
        .encode(text, false)
        .unwrap_or_else(|error| panic!("tokenize {text:?}: {error}"))
        .get_ids()
        .to_vec()
}

fn first_prompt(qwen: &Qwen35LoadedModel, task: &str) -> Vec<u32> {
    encode(
        qwen,
        &format!("<|im_start|>user\n{task}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    )
}

/// The request params the Qwen thinking policy resolves for a forced-open
/// turn with a `BUDGET` thinking budget.
fn budget_params(qwen: &Qwen35LoadedModel, max_tokens: usize) -> SamplingParams {
    SamplingParams {
        max_tokens,
        reasoning_forced_open: true,
        thinking_token_budget: Some(BUDGET),
        reasoning_end_tokens: Some(Arc::new(encode(qwen, "\nI need to answer now.</think>"))),
        reasoning_close_tokens: Some(Arc::new(encode(qwen, "</think>"))),
        ..SamplingParams::default()
    }
}

fn load() -> Box<LoadedModel> {
    let mut loaded = load_gate_artifact(ENV);
    let LoadedModel::Qwen35(qwen) = loaded.as_mut() else {
        unreachable!("load_gate_artifact checks the family")
    };
    qwen.speculation = QwenSpeculationController::new(QwenSpeculationPolicy::Off);
    loaded
}

/// Bounded server prefill from `cached` tokens already in `slot`.
fn prefill(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    slot: SlotId,
    prompt: &[u32],
    cached: usize,
    params: SamplingParams,
    supervisor: &EngineSupervisor,
) -> (Qwen35DecodeState, Vec<f32>) {
    let mut pending = Qwen35PrefillState::begin(
        prompt.to_vec(),
        params,
        None,
        kv,
        slot,
        cached,
        None,
        None,
        None,
        qwen.hidden_size,
        None,
    )
    .unwrap_or_else(|error| panic!("slot {} admission: {error:#}", slot.0));
    loop {
        match pending
            .advance(qwen, None, kv, PREFILL_CHUNK, supervisor)
            .unwrap_or_else(|error| panic!("slot {} prefill: {error:#}", slot.0))
        {
            Qwen35PrefillAdvance::Pending { state, .. } => pending = state,
            Qwen35PrefillAdvance::Ready {
                state,
                prefill_logits,
                ..
            } => {
                assert!(
                    !state.finished_at_seed(),
                    "slot {} finished at its seed",
                    slot.0
                );
                return (state, prefill_logits);
            }
        }
    }
}

fn decode_serial(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    state: &mut Qwen35DecodeState,
    supervisor: &EngineSupervisor,
) -> Vec<Tick> {
    let mut ticks: Vec<Tick> = Vec::new();
    while !ticks.last().is_some_and(|tick| tick.finished) {
        assert!(ticks.len() < MAX_TICKS, "request did not finish");
        ticks.push(tick_target(qwen, kv, state.slot_id, state, supervisor));
    }
    ticks
}

/// Ordinary decode of `target` and `peer` through joint steps (both rows
/// while both are live, then the survivor alone), exactly as the worker's
/// joint turn prepares, runs and commits rows.
fn decode_joint(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    target: &mut Qwen35DecodeState,
    peer: &mut Qwen35DecodeState,
    mode: JointRouteMode,
    supervisor: &EngineSupervisor,
) {
    let dense_ffn = select_dense_ffn_route(mode, true);
    let (mut target_live, mut peer_live) = (true, true);
    for _ in 0..MAX_TICKS {
        if !target_live {
            return;
        }
        let mut lanes = Vec::new();
        for (index, (state, live)) in [(&mut *target, target_live), (&mut *peer, peer_live)]
            .into_iter()
            .enumerate()
        {
            if live && state.plan_tick().expect("plan").is_some() {
                state.consume_planned_tick();
                lanes.push((
                    index,
                    state.prepare_joint_tick(kv).expect("prepare joint row"),
                ));
            }
        }
        let rows: Vec<_> = lanes.iter().map(|(_, lane)| lane.row).collect();
        let outputs = qwen
            .forward_joint_decode(
                &rows,
                kv,
                mode,
                dense_ffn,
                JointStepOptions::default(),
                supervisor,
            )
            .expect("joint forward");
        for ((index, lane), output) in lanes.into_iter().zip(outputs) {
            let state = if index == 0 { &mut *target } else { &mut *peer };
            assert!(matches!(lane.target, Qwen35JointTarget::Ordinary { .. }));
            let outcome = state
                .finish_joint_tick(qwen, kv, supervisor, lane.target, output)
                .expect("finish joint row");
            if state.complete_tick(outcome).finished {
                if index == 0 {
                    target_live = false;
                } else {
                    peer_live = false;
                }
            }
        }
    }
    panic!("joint target did not finish");
}

/// The anchor a force-closed request publishes, with the request's tokens.
struct FirstTurn {
    generated: Vec<u32>,
    anchor: Qwen35ForcedCloseAnchor,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FirstRoute {
    Serial,
    Joint(JointRouteMode),
}

/// Twin B's first request, decoded to completion with the switch on.
fn forced_close_turn(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    route: FirstRoute,
    supervisor: &EngineSupervisor,
) -> FirstTurn {
    let prompt = first_prompt(qwen, TASK);
    let params = budget_params(qwen, FIRST_MAX_TOKENS);
    let (mut state, _) = prefill(qwen, kv, SLOT, &prompt, 0, params, supervisor);
    state.forced_close_anchor_enabled = true;
    match route {
        FirstRoute::Serial => {
            decode_serial(qwen, kv, &mut state, supervisor);
        }
        FirstRoute::Joint(mode) => {
            let peer_prompt = first_prompt(qwen, PEER_TASK);
            let peer_params = SamplingParams {
                max_tokens: FIRST_MAX_TOKENS + 32,
                ..SamplingParams::default()
            };
            let (mut peer, _) = prefill(qwen, kv, PEER, &peer_prompt, 0, peer_params, supervisor);
            decode_joint(qwen, kv, &mut state, &mut peer, mode, supervisor);
        }
    }
    let close = encode(qwen, "\nI need to answer now.</think>");
    assert!(
        state
            .generated_tokens
            .windows(close.len())
            .any(|window| window == close.as_slice()),
        "the request must be force-closed at the {BUDGET}-token budget; raise the task's \
         reasoning demand if the model closed on its own"
    );
    let generated = state.generated_tokens.clone();
    let anchor = state
        .take_forced_close_anchor()
        .expect("a force-closed request publishes its anchor");
    assert_eq!(
        &anchor.prompt_tokens[..prompt.len()],
        prompt.as_slice(),
        "anchor starts with the prompt"
    );
    assert!(
        anchor.prompt_tokens.len() < prompt.len() + BUDGET,
        "anchor excludes the model's last reasoning token"
    );
    FirstTurn { generated, anchor }
}

/// R2: the anchor's tokens plus the client's re-render tail.
fn follow_up_prompt(qwen: &Qwen35LoadedModel, anchor: &Qwen35ForcedCloseAnchor) -> Vec<u32> {
    let mut prompt = anchor.prompt_tokens.clone();
    prompt.extend(encode(qwen, RE_RENDER_TAIL));
    prompt
}

fn serve_follow_up(
    qwen: &mut Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    prompt: &[u32],
    cached: usize,
    supervisor: &EngineSupervisor,
) -> TargetTrace {
    let params = SamplingParams {
        max_tokens: FOLLOW_UP_MAX_TOKENS,
        ..SamplingParams::default()
    };
    let (mut state, prefill_logits) = prefill(qwen, kv, SLOT, prompt, cached, params, supervisor);
    let ticks = decode_serial(qwen, kv, &mut state, supervisor);
    let next = *state.generated_tokens.last().expect("generated");
    let position = prompt.len() + state.generated_tokens.len() - 1;
    let probe = if cursors(kv, SLOT).0 as usize == position {
        qwen.model
            .forward_gpu_last_logits(&[next], &[position as i32; 4], kv, SLOT)
            .expect("probe forward")
    } else {
        Vec::new()
    };
    TargetTrace {
        prefill_logits: bits(&prefill_logits),
        ticks,
        state: slot_state(kv, SLOT),
        probe_logits: bits(&probe),
    }
}

/// Twin B: forced-close request, anchor restore, follow-up.
fn twin_b(qwen: &mut Qwen35LoadedModel, route: FirstRoute) -> (FirstTurn, TargetTrace) {
    let supervisor = EngineSupervisor::new();
    let mut kv = new_cache(qwen);
    let first = forced_close_turn(qwen, &mut kv, route, &supervisor);
    let p = first.anchor.prompt_tokens.len();
    assert!(kv.slot_anchor_cursors_cover(SLOT, &first.anchor.kv));
    kv.restore_slot_anchor(SLOT, &first.anchor.kv)
        .expect("forced-close anchor restore");
    assert_eq!(cursors(&kv, SLOT).0 as usize, p);
    let r2 = follow_up_prompt(qwen, &first.anchor);
    let trace = serve_follow_up(qwen, &mut kv, &r2, p, &supervisor);
    assert!(supervisor.is_healthy());
    (first, trace)
}

/// Twin A: the same request stopped at the anchor boundary, then the
/// follow-up by live reuse of the slot (no restore).
fn twin_a(qwen: &mut Qwen35LoadedModel, anchor: &Qwen35ForcedCloseAnchor) -> TargetTrace {
    let supervisor = EngineSupervisor::new();
    let mut kv = new_cache(qwen);
    let prompt = first_prompt(qwen, TASK);
    let p = anchor.prompt_tokens.len();
    // A request finishing at max_tokens leaves prompt + generated − 1 rows.
    let max_tokens = p - prompt.len() + 1;
    let params = budget_params(qwen, max_tokens);
    let (mut state, _) = prefill(qwen, &mut kv, SLOT, &prompt, 0, params, &supervisor);
    decode_serial(qwen, &mut kv, &mut state, &supervisor);
    let cursor = cursors(&kv, SLOT).0 as usize;
    assert_eq!(cursor, p, "twin A stops at the anchor boundary");
    assert_eq!(
        state.retained_prefix(cursor),
        anchor.prompt_tokens,
        "same tokens up to P"
    );
    let r2 = follow_up_prompt(qwen, anchor);
    let trace = serve_follow_up(qwen, &mut kv, &r2, p, &supervisor);
    assert!(supervisor.is_healthy());
    trace
}

#[test]
#[ignore = "real Qwen3.6 artifact + GPU grant: HF2Q_QWEN36_GGUF"]
fn qwen36_forced_close_anchor_g5_same_route_twin_is_bitwise() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut loaded = load();
    let LoadedModel::Qwen35(qwen) = loaded.as_mut() else {
        unreachable!()
    };
    let (first, anchored) = twin_b(qwen, FirstRoute::Serial);
    let live = twin_a(qwen, &first.anchor);
    eprintln!(
        "G5: prompt+generated={} anchor_tokens={} follow_up_ticks={}",
        first_prompt(qwen, TASK).len() + first.generated.len(),
        first.anchor.prompt_tokens.len(),
        anchored.ticks.len()
    );
    assert_same(&live, &anchored, "G5 live twin vs forced-close anchor twin");
}

#[test]
#[ignore = "real Qwen3.6 artifact + GPU grant: HF2Q_QWEN36_GGUF"]
fn qwen36_forced_close_anchor_g6_determinism_and_joint_route() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let mut loaded = load();
    let LoadedModel::Qwen35(qwen) = loaded.as_mut() else {
        unreachable!()
    };
    let (first_a, trace_a) = twin_b(qwen, FirstRoute::Serial);
    let (first_b, trace_b) = twin_b(qwen, FirstRoute::Serial);
    assert_eq!(
        first_a.generated, first_b.generated,
        "G6 determinism: first request"
    );
    assert_eq!(first_a.anchor.prompt_tokens, first_b.anchor.prompt_tokens);
    assert_same(&trace_a, &trace_b, "G6 determinism: follow-up");

    let mode = qwen
        .model
        .joint_decode_mode()
        .expect("joint decode available on this artifact");
    let (joint_1, joint_trace_1) = twin_b(qwen, FirstRoute::Joint(mode));
    let (joint_2, joint_trace_2) = twin_b(qwen, FirstRoute::Joint(mode));
    assert_eq!(joint_1.generated, joint_2.generated, "G6 joint determinism");
    assert_same(
        &joint_trace_1,
        &joint_trace_2,
        "G6 joint determinism: follow-up",
    );
    let generated_prefix = first_a.anchor.prompt_tokens.len() - first_prompt(qwen, TASK).len();
    eprintln!(
        "G6: joint mode={} serial anchor={} joint anchor={} first divergence={:?}",
        mode.label(),
        first_a.anchor.prompt_tokens.len(),
        joint_1.anchor.prompt_tokens.len(),
        first_a
            .generated
            .iter()
            .zip(&joint_1.generated)
            .position(|(serial, joint)| serial != joint)
    );
    if mode == JointRouteMode::Exact {
        assert_eq!(
            first_a.generated, joint_1.generated,
            "G6 joint ≡ per-slot (exact route)"
        );
        assert_eq!(first_a.anchor.prompt_tokens, joint_1.anchor.prompt_tokens);
        assert_same(
            &trace_a,
            &joint_trace_1,
            "G6 joint ≡ per-slot follow-up (exact route)",
        );
    } else {
        // The padded route is Class B against serial decode: the boundary is
        // decided by the budget alone, so it is the same position whenever
        // the reasoning tokens before it agree.
        if first_a.generated[..generated_prefix] == joint_1.generated[..generated_prefix] {
            assert_eq!(first_a.anchor.prompt_tokens, joint_1.anchor.prompt_tokens);
        }
    }
}
