//! Joint MTP route gates (H3/H4), model-free.
//!
//! Host-only: the cap8 depth table and the draft-free accept walk.
//!
//! Metal (synthetic Qwen3.5-family model with a synthetic MTP block, built
//! by `joint_verify_tests::synthetic_model`): a scheduler loop drives four
//! lanes through whole generations, once with spec-off joint decode steps
//! (`prepare_joint_tick` / `forward_joint_decode` / `finish_joint_tick`, the
//! serving turn without speculation) and once with joint MTP steps
//! (`run_joint_mtp_step` plus the in-turn drain). Lane 0 is greedy, lane 1
//! has no MTP state (an ordinary row inside the MTP step), lane 2 decodes
//! with a repetition penalty and a short length limit, and lane 3 joins at
//! turn 3. Every lane's committed tokens and finish reason must equal
//! spec-off's:
//!
//! - with forced draft outcomes (full, partial and zero accepts, cycled
//!   per call; drafts taken from the spec-off stream), in the padded MMA,
//!   padded `mul_mv_ext` and exact modes;
//! - with the real batched MTP drafter, twice, with identical drafts in
//!   both runs (determinism).
//!
//! ```text
//! cargo test --release --bin hf2q joint_mtp -- --test-threads=1
//! ```

use anyhow::Result;
use mlx_native::MlxDevice;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use super::super::speculation_qualification_tests::loaded_model;
use super::super::{Qwen35CanonicalDecision, Qwen35DecodeState, Qwen35LoadedModel};
use super::{
    cap8_draft_depths, diagnostic_draft_depths, plan_qwen35_window, run_joint_mtp_step,
    Qwen35JointMtpDraftRequest, Qwen35JointMtpForward, Qwen35JointMtpStageTiming,
};
use crate::inference::models::qwen35::joint_decode_routes::{JointDenseFfnRoute, JointRouteMode};
use crate::inference::models::qwen35::joint_projection::JointProjectionKernel;
use crate::inference::models::qwen35::joint_step_options::JointStepOptions;
use crate::inference::models::qwen35::joint_verify_tests::{
    exact_policy, padded_policy, synthetic_model,
};
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::spec_decode::{last_hidden_row, positions_for_range};
use crate::serve::api::engine::SamplingParams;
use crate::serve::api::engine_supervisor::EngineSupervisor;
use crate::serve::api::qwen35_speculation::{QwenSpeculationController, QwenSpeculationPolicy};
use crate::serve::multi_seq_kv::{MultiSeqKvCache, SlotId};

#[test]
fn cap8_depths_follow_the_serving_width_rule() {
    let full = |lanes: usize| cap8_draft_depths(&vec![3; lanes]);
    assert_eq!(full(4), [1, 1, 1, 1]);
    assert_eq!(full(3), [2, 2, 1]);
    assert_eq!(full(2), [3, 3]);
    assert_eq!(full(1), [3]);
    assert_eq!(cap8_draft_depths(&[0, 3, 3, 3]), [0, 2, 1, 1]);
    assert_eq!(cap8_draft_depths(&[1, 3, 3]), [1, 2, 2]);
    for caps in [[3, 3, 3, 3], [0, 1, 2, 3], [3, 0, 0, 1]] {
        let depths = cap8_draft_depths(&caps);
        assert!(depths.iter().sum::<usize>() + caps.len() <= 8);
        assert!(depths.iter().zip(&caps).all(|(depth, cap)| depth <= cap));
    }
}

#[test]
fn diagnostic_depths_cover_four_by_four() {
    assert_eq!(diagnostic_draft_depths(&[3, 3, 3, 3]), [3, 3, 3, 3]);
    assert_eq!(diagnostic_draft_depths(&[3, 3, 0, 0]), [3, 3, 0, 0]);
    assert_eq!(diagnostic_draft_depths(&[3, 0, 0, 0]), [3, 0, 0, 0]);
}

#[test]
fn draft_free_window_commits_its_bonus_row() {
    let decision =
        |token, terminal| move |_: usize| Ok(Qwen35CanonicalDecision { token, terminal });
    let plan = plan_qwen35_window(&[], decision(7, false)).expect("plan");
    assert_eq!(plan.output, [7]);
    assert!(!plan.terminal_after_pending);
    assert_eq!((plan.valid_input_tokens, plan.carry_hidden_row), (1, 0));
    let stop = plan_qwen35_window(&[], decision(7, true)).expect("plan");
    assert!(stop.output.is_empty() && stop.terminal_after_pending);
    assert_eq!(stop.valid_input_tokens, 1);
}

const SLOTS: u32 = 4;
const CAPACITY: u32 = 128;
const VOCAB: u32 = 512;
const MAX_TURNS: usize = 400;

#[derive(Clone)]
struct LanePlan {
    slot: u32,
    prompt: Vec<u32>,
    max_tokens: usize,
    penalty: bool,
    /// Whether the lane holds MTP state in the spec-on arm.
    mtp: bool,
    join_turn: usize,
}

fn synthetic_prompt(slot: u32, len: u32) -> Vec<u32> {
    (0..len)
        .map(|index| (slot * 97 + index * 31 + 7) % VOCAB)
        .collect()
}

/// Lane 0 greedy; lane 1 without MTP state; lane 2 with a repetition
/// penalty and a short limit (leaves early); lane 3 joins at turn 3.
fn synthetic_lanes() -> Vec<LanePlan> {
    let lane = |slot, len, max_tokens, penalty, mtp, join_turn| LanePlan {
        slot,
        prompt: synthetic_prompt(slot, len),
        max_tokens,
        penalty,
        mtp,
        join_turn,
    };
    vec![
        lane(0, 18, 28, false, true, 0),
        lane(1, 21, 20, false, false, 0),
        lane(2, 17, 9, true, true, 0),
        lane(3, 24, 22, false, true, 3),
    ]
}

fn params(plan: &LanePlan) -> SamplingParams {
    SamplingParams {
        max_tokens: plan.max_tokens,
        repetition_penalty: if plan.penalty { 1.05 } else { 1.0 },
        ..SamplingParams::default()
    }
}

/// One lane's committed generation.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Outcome {
    tokens: Vec<u32>,
    finish: &'static str,
}

type Drafter<'a> = dyn FnMut(
        &Qwen35LoadedModel,
        &mut HybridKvCache,
        &[Qwen35JointMtpDraftRequest<'_>],
    ) -> Result<Vec<Vec<u32>>>
    + 'a;

/// The synthetic model with MTP, wrapped as a loaded Qwen model whose
/// speculation controller selects the joint MTP route (`spec_on`) or not.
fn qwen(policy: mlx_native::GgmlRoutingPolicy, eos: &[u32], spec_on: bool) -> Qwen35LoadedModel {
    let device = MlxDevice::new().expect("Metal device");
    let mut qwen = loaded_model(QwenSpeculationPolicy::Off);
    qwen.model = synthetic_model(&device, policy, true);
    qwen.eos_token_ids = eos.to_vec();
    qwen.vocab_size = VOCAB as usize;
    qwen.hidden_size = qwen.model.cfg.hidden_size as usize;
    qwen.speculation = speculation(spec_on);
    qwen
}

/// The arm's speculation controller: the joint MTP route, or off.
fn speculation(spec_on: bool) -> QwenSpeculationController {
    if spec_on {
        let mut controller = QwenSpeculationController::new(QwenSpeculationPolicy::Auto);
        assert!(controller.select_joint_mtp(true));
        controller
    } else {
        QwenSpeculationController::new(QwenSpeculationPolicy::Off)
    }
}

/// Prefill one lane as the matching serving arm would: with MTP prompt
/// catch-up for an MTP lane, the ordinary prefill otherwise.
fn seed(
    qwen: &Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    plan: &LanePlan,
    with_mtp: bool,
) -> Qwen35DecodeState {
    let slot = SlotId(plan.slot);
    let tokens = plan.prompt.clone();
    let positions = positions_for_range(0, tokens.len());
    let (logits, hidden) = if with_mtp {
        let (logits, nextn) = qwen
            .model
            .forward_gpu_last_logits_with_hidden(&tokens, &positions, kv, slot)
            .expect("prefill with hidden");
        let embed = qwen
            .model
            .embed_tokens_gpu(&tokens)
            .expect("prompt embeddings");
        let mtp = qwen.model.mtp.as_ref().expect("MTP weights");
        qwen.model
            .with_gpu_cache_mut(|device, registry| {
                mtp.process_target_batch(
                    &tokens,
                    None,
                    &nextn,
                    &embed,
                    kv,
                    slot,
                    &positions,
                    device,
                    registry,
                    &qwen.model.cfg,
                )
            })
            .expect("MTP prompt catch-up");
        let hidden = last_hidden_row(&nextn, qwen.model.cfg.hidden_size).expect("carry row");
        (logits, Some(hidden))
    } else {
        let logits = qwen
            .model
            .forward_gpu_last_logits(&tokens, &positions, kv, slot)
            .expect("prefill");
        (logits, None)
    };
    Qwen35DecodeState::from_prefill_logits(
        qwen,
        tokens,
        params(plan),
        None,
        slot,
        0,
        &logits,
        Duration::ZERO,
        plan.prompt.len(),
        hidden,
    )
    .expect("seed decode state")
}

/// Drive every lane to completion; `drafter: None` runs spec-off joint
/// decode steps. Returns each slot's outcome and the number of turns.
fn drive(
    qwen: &Qwen35LoadedModel,
    lanes: &[LanePlan],
    capacity: u32,
    forward: Qwen35JointMtpForward,
    mut drafter: Option<&mut Drafter<'_>>,
) -> (HashMap<u32, Outcome>, usize) {
    let supervisor = EngineSupervisor::new();
    let spec_on = drafter.is_some();
    // The loaded model may have moved since it was primed (the GPU cache is
    // keyed by the model's address): prime it where it now lives before
    // allocating the cache on its device.
    qwen.model
        .ensure_gpu_cache_primed()
        .expect("prime GPU cache at the model's final address");
    let mut kv = qwen
        .model
        .with_gpu_cache_mut(|device, _registry| {
            HybridKvCache::new_with_options(&qwen.model.cfg, device, capacity, SLOTS, true)
        })
        .expect("allocate four-slot TQ cache");
    let mut states: Vec<Option<Qwen35DecodeState>> = (0..SLOTS).map(|_| None).collect();
    let mut outcomes = HashMap::new();
    let finish = |state: Qwen35DecodeState, outcomes: &mut HashMap<u32, Outcome>| {
        outcomes.insert(
            state.slot_id.0,
            Outcome {
                tokens: state.generated_tokens.clone(),
                finish: state.finish_reason,
            },
        );
    };
    for turn in 0..MAX_TURNS {
        for plan in lanes.iter().filter(|plan| plan.join_turn == turn) {
            let state = seed(qwen, &mut kv, plan, spec_on && plan.mtp);
            if state.finished_at_seed() {
                finish(state, &mut outcomes);
            } else {
                states[plan.slot as usize] = Some(state);
            }
        }
        if outcomes.len() == lanes.len() {
            return (outcomes, turn);
        }
        let mut active = states
            .iter_mut()
            .filter_map(Option::as_mut)
            .collect::<Vec<_>>();
        let mut finished = Vec::new();
        for group in active.chunks_mut(4) {
            let ticks = match drafter.as_deref_mut() {
                Some(drafter) => run_joint_mtp_step(
                    qwen,
                    &mut kv,
                    group,
                    forward,
                    &supervisor,
                    drafter,
                    &|_| false,
                    None,
                    None,
                )
                .expect("joint MTP step"),
                None => spec_off_step(qwen, &mut kv, group, forward, &supervisor),
            };
            for (state, tick) in group.iter_mut().zip(ticks) {
                let slot = state.slot_id.0;
                let mut tick = tick.unwrap_or_else(|error| panic!("slot {slot}: {error:#}"));
                // The in-turn drain of verified output.
                while !tick.finished && state.has_pending_speculation_output() {
                    tick = state
                        .joint_mtp_drain_tick(qwen)
                        .unwrap_or_else(|error| panic!("slot {slot} drain: {error:#}"));
                }
                assert!(
                    tick.finished || !state.has_pending_speculation_output(),
                    "slot {slot} left verified output for a later turn"
                );
                if tick.finished {
                    finished.push(slot);
                }
            }
        }
        for slot in finished {
            let state = states[slot as usize].take().expect("finished lane");
            kv.reset_for_slot(SlotId(slot))
                .expect("reset finished slot");
            finish(state, &mut outcomes);
        }
    }
    panic!("lanes did not finish within {MAX_TURNS} turns");
}

/// One spec-off joint decode step (the serving turn without speculation).
fn spec_off_step(
    qwen: &Qwen35LoadedModel,
    kv: &mut HybridKvCache,
    states: &mut [&mut Qwen35DecodeState],
    forward: Qwen35JointMtpForward,
    supervisor: &EngineSupervisor,
) -> Vec<Result<super::super::Qwen35TickOutcome>> {
    let lanes = states
        .iter_mut()
        .map(|state| state.prepare_joint_tick(kv).expect("prepare joint row"))
        .collect::<Vec<_>>();
    let rows = lanes.iter().map(|lane| lane.row).collect::<Vec<_>>();
    let outputs = qwen
        .forward_joint_decode(
            &rows,
            kv,
            forward.mode,
            forward.dense_ffn,
            forward.options,
            supervisor,
        )
        .expect("spec-off joint step");
    states
        .iter_mut()
        .zip(lanes)
        .zip(outputs)
        .map(|((state, lane), output)| {
            state.finish_joint_tick(qwen, kv, supervisor, lane.target, output)
        })
        .collect()
}

/// Drafts from the spec-off stream: per call, the first `c` drafts are the
/// true next tokens and the rest are wrong, `c` cycling through every
/// accept count.
fn forced_drafter(
    lanes: &[LanePlan],
    reference: HashMap<u32, Outcome>,
) -> impl FnMut(
    &Qwen35LoadedModel,
    &mut HybridKvCache,
    &[Qwen35JointMtpDraftRequest<'_>],
) -> Result<Vec<Vec<u32>>> {
    let bases = lanes
        .iter()
        .map(|plan| (plan.slot, plan.prompt.len() as i32))
        .collect::<HashMap<_, _>>();
    let mut calls = 0_usize;
    move |_qwen, _kv, requests| {
        calls += 1;
        Ok(requests
            .iter()
            .map(|request| {
                let outcome = &reference[&request.slot.0];
                // The pending token is generated token `g - 1`.
                let g = (request.position - bases[&request.slot.0] + 1) as usize;
                let truth = |index: usize| outcome.tokens.get(index).copied().unwrap_or_default();
                let correct = (calls + request.slot.0 as usize) % (request.depth + 1);
                (0..request.depth)
                    .map(|depth| {
                        let token = truth(g + depth);
                        if depth < correct {
                            token
                        } else {
                            (token + 1) % VOCAB
                        }
                    })
                    .collect()
            })
            .collect())
    }
}

/// Arm M of the n4 serving step: padded mode, shared-weights FFN,
/// column-invariant MMA products and pooled step outputs.
fn mma_forward() -> Qwen35JointMtpForward {
    Qwen35JointMtpForward {
        mode: JointRouteMode::PaddedMvExt,
        dense_ffn: JointDenseFfnRoute::SharedWeights,
        options: JointStepOptions {
            projection: JointProjectionKernel::ColumnInvariantMma,
            pooled_outputs: true,
            verify_attention_tile: false,
        },
    }
}

fn assert_streams_equal(
    lanes: &[LanePlan],
    observed: &HashMap<u32, Outcome>,
    reference: &HashMap<u32, Outcome>,
) {
    for plan in lanes {
        assert_eq!(
            observed[&plan.slot], reference[&plan.slot],
            "slot {} spec-on stream differs from spec-off",
            plan.slot
        );
    }
}

fn forced_spec_on_matches_spec_off(
    policy: mlx_native::GgmlRoutingPolicy,
    forward: Qwen35JointMtpForward,
) {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    if MlxDevice::new().is_err() {
        return;
    }
    let lanes = synthetic_lanes();
    let qwen = qwen(policy, &[], true);
    let (reference, off_turns) = drive(&qwen, &lanes, CAPACITY, forward, None);
    let mut drafter = forced_drafter(&lanes, reference.clone());
    let (observed, on_turns) = drive(
        &qwen,
        &lanes,
        CAPACITY,
        forward,
        Some(&mut drafter as &mut Drafter<'_>),
    );
    assert_streams_equal(&lanes, &observed, &reference);
    assert!(
        on_turns < off_turns,
        "accepted drafts must shorten the run ({on_turns} vs {off_turns} turns)"
    );
}

#[test]
fn joint_mtp_forced_drafts_match_spec_off_mma() {
    forced_spec_on_matches_spec_off(padded_policy(), mma_forward());
}

#[test]
fn joint_mtp_forced_drafts_match_spec_off_mv_ext() {
    forced_spec_on_matches_spec_off(
        padded_policy(),
        Qwen35JointMtpForward {
            mode: JointRouteMode::PaddedMvExt,
            dense_ffn: JointDenseFfnRoute::SharedWeights,
            options: JointStepOptions::default(),
        },
    );
}

#[test]
fn joint_mtp_forced_drafts_match_spec_off_exact() {
    forced_spec_on_matches_spec_off(
        exact_policy(),
        Qwen35JointMtpForward {
            mode: JointRouteMode::Exact,
            dense_ffn: JointDenseFfnRoute::SerialBuilder,
            options: JointStepOptions::default(),
        },
    );
}

/// Spec-on with the real batched drafter, once: the outcomes and every
/// drafter call's `(slot, position, drafts)`.
type DraftLog = Vec<Vec<((u32, i32), Vec<u32>)>>;

fn spec_on_with_batched_drafter(
    qwen: &Qwen35LoadedModel,
    lanes: &[LanePlan],
    capacity: u32,
    forward: Qwen35JointMtpForward,
) -> (HashMap<u32, Outcome>, usize, DraftLog) {
    let supervisor = EngineSupervisor::new();
    let log = RefCell::new(Vec::new());
    let mut drafter = |qwen: &Qwen35LoadedModel,
                       kv: &mut HybridKvCache,
                       requests: &[Qwen35JointMtpDraftRequest<'_>]| {
        let drafts = qwen.draft_joint_mtp(kv, requests, &supervisor)?;
        log.borrow_mut().push(
            requests
                .iter()
                .map(|request| (request.slot.0, request.position))
                .zip(drafts.clone())
                .collect::<Vec<_>>(),
        );
        Ok(drafts)
    };
    let (observed, turns) = drive(
        qwen,
        lanes,
        capacity,
        forward,
        Some(&mut drafter as &mut Drafter<'_>),
    );
    (observed, turns, log.into_inner())
}

/// The real batched drafter (H3): streams equal spec-off, and two runs
/// draft identically.
#[test]
fn joint_mtp_batched_drafts_are_deterministic_and_match_spec_off() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    if MlxDevice::new().is_err() {
        return;
    }
    let forward = mma_forward();
    let policy = padded_policy();
    let lanes = synthetic_lanes();
    let qwen = qwen(policy, &[], true);
    let (reference, _) = drive(&qwen, &lanes, CAPACITY, forward, None);
    let (first, _, first_drafts) = spec_on_with_batched_drafter(&qwen, &lanes, CAPACITY, forward);
    let (second, _, second_drafts) = spec_on_with_batched_drafter(&qwen, &lanes, CAPACITY, forward);
    assert_streams_equal(&lanes, &first, &reference);
    assert_eq!(first, second, "spec-on runs differ");
    assert!(!first_drafts.is_empty(), "the batched drafter never ran");
    assert_eq!(
        first_drafts, second_drafts,
        "batched drafts are not deterministic"
    );
}

/// Qwen3.8 artifact: four greedy lanes × 64 tokens (lane 3 joins at turn
/// 4), spec-on (joint MTP, real batched drafter) against spec-off joint
/// decode in arm M, identical tokens and finish reasons; spec-on must take
/// fewer turns. `HF2Q_QWEN38_GGUF` must name a Qwen3.8 path (the serve
/// routing policy binds from it):
///
/// ```text
/// HF2Q_QWEN38_GGUF=/path/qwen3.8.gguf cargo test --release --bin hf2q \
///   qwen38_joint_mtp_matches_spec_off_tokens -- --ignored --exact --nocapture
/// ```
#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (a Qwen3.8-named path with its MTP head) and Apple Metal"]
fn qwen38_joint_mtp_matches_spec_off_tokens() {
    use crate::serve::api::engine::{LoadOptions, LoadedModel};

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let path = std::path::PathBuf::from(
        std::env::var_os("HF2Q_QWEN38_GGUF").expect("HF2Q_QWEN38_GGUF must name the artifact"),
    );
    assert!(path.is_file(), "artifact is absent at {}", path.display());
    let options = LoadOptions {
        model_path: path,
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
        lcp_resume_budget: crate::serve::kv_persist::lcp_registry::LcpResumeBudget::DEFAULT,
    };
    let LoadedModel::Qwen35(mut qwen) = LoadedModel::load(&options).expect("load Qwen3.8") else {
        panic!("HF2Q_QWEN38_GGUF must name a Qwen3.5-family GGUF");
    };
    assert!(
        qwen.model.mtp.is_some(),
        "the gate needs the native MTP head"
    );
    let forward = mma_forward();
    assert_eq!(qwen.model.joint_decode_mode(), Ok(forward.mode));
    let encode = |text: &str| -> Vec<u32> {
        let turn = format!(
            "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        qwen.tokenizer
            .encode(turn, false)
            .expect("tokenize prompt")
            .get_ids()
            .to_vec()
    };
    let prompts = [
        "Write a Rust function `fib(n: u64) -> u64` that computes Fibonacci numbers iteratively.",
        "List the integers from 1 to 40, separated by commas, with no other text.",
        "Explain in two sentences what a hash map is.",
        "Write a Python class `Stack` with push, pop and peek methods.",
    ];
    let lanes = prompts
        .iter()
        .enumerate()
        .map(|(slot, text)| LanePlan {
            slot: slot as u32,
            prompt: encode(text),
            max_tokens: 64,
            penalty: false,
            mtp: true,
            join_turn: if slot == 3 { 4 } else { 0 },
        })
        .collect::<Vec<_>>();
    let capacity = lanes
        .iter()
        .map(|plan| plan.prompt.len() as u32 + 64 + 8)
        .max()
        .expect("lanes");

    qwen.speculation = speculation(false);
    let (reference, off_turns) = drive(&qwen, &lanes, capacity, forward, None);
    qwen.speculation = speculation(true);
    let (observed, on_turns, drafts) =
        spec_on_with_batched_drafter(&qwen, &lanes, capacity, forward);
    for plan in &lanes {
        eprintln!(
            "slot {}: {} tokens ({}), spec-off {:?}",
            plan.slot,
            reference[&plan.slot].tokens.len(),
            reference[&plan.slot].finish,
            qwen.tokenizer.decode(&reference[&plan.slot].tokens, false)
        );
    }
    eprintln!(
        "turns: spec-off {off_turns}, spec-on {on_turns}; drafter calls {}",
        drafts.len()
    );
    assert_streams_equal(&lanes, &observed, &reference);
    assert!(
        on_turns < off_turns,
        "joint MTP must accept drafts ({on_turns} vs {off_turns} turns)"
    );
}

#[derive(Serialize)]
struct DiagnosticSample {
    context_tokens: usize,
    shape: &'static str,
    rows: usize,
    warmup: bool,
    repetition: usize,
    committed_tokens: Vec<usize>,
    prepare_ns: u64,
    draft_ns: u64,
    verify_readback_ns: u64,
    selection_ns: u64,
    capture_recovery_ns: u64,
    settle_catch_up_ns: u64,
    total_ns: u64,
}

#[derive(Serialize)]
struct DiagnosticReport {
    schema: &'static str,
    source_commit: String,
    model_path: String,
    q5_case_path: String,
    q5_case_sha256: String,
    grid: String,
    warmups_per_configuration: usize,
    timed_runs_per_configuration: usize,
    samples: Vec<DiagnosticSample>,
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).expect("diagnostic duration fits u64")
}

/// ADR-065 E7/E2 diagnostic-only driver. It never enters the HTTP serving
/// path. `smoke` covers the Q5-admitted 4×1 and 4×4 rounds at 2K context;
/// `full` covers all nine shapes at 2K, 16K and 64K with one warm-up and
/// seven retained measurements per configuration.
#[test]
#[ignore = "requires Qwen3.8, Q5 case identity, retained-output path and Apple Metal"]
fn qwen38_joint_verify_diagnostic_grid() {
    use crate::serve::api::engine::{LoadOptions, LoadedModel};

    const CONTEXTS: &[usize] = &[2_048, 16_384, 65_536];
    const SHAPES: &[(&str, &[usize])] = &[
        ("4x1", &[0, 0, 0, 0]),
        ("4x2", &[1, 1, 1, 1]),
        ("4x3", &[2, 2, 2, 2]),
        ("4x4", &[3, 3, 3, 3]),
        ("4+4+1+1", &[3, 3, 0, 0]),
        ("4+1+1+1", &[3, 0, 0, 0]),
        ("1x4", &[3]),
        ("2x4", &[3, 3]),
        ("3x4", &[3, 3, 3]),
    ];

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let model_path = PathBuf::from(
        std::env::var_os("HF2Q_QWEN38_GGUF").expect("HF2Q_QWEN38_GGUF must name the artifact"),
    );
    assert!(
        model_path.is_file(),
        "artifact is absent at {}",
        model_path.display()
    );
    let q5_case_path = PathBuf::from(
        std::env::var_os("HF2Q_ADR065_E_Q5_CASE")
            .expect("HF2Q_ADR065_E_Q5_CASE must name the frozen Q5 case"),
    );
    let q5_case = fs::read(&q5_case_path)
        .unwrap_or_else(|error| panic!("read Q5 case {}: {error}", q5_case_path.display()));
    let output_path = PathBuf::from(
        std::env::var_os("HF2Q_ADR065_E_OUTPUT")
            .expect("HF2Q_ADR065_E_OUTPUT must name the retained JSON report"),
    );
    let grid = std::env::var("HF2Q_ADR065_E_GRID").unwrap_or_else(|_| "smoke".into());
    assert!(
        matches!(grid.as_str(), "smoke" | "full"),
        "grid must be smoke or full"
    );
    let source_commit =
        std::env::var("HF2Q_ADR065_E_SOURCE_COMMIT").expect("source commit identity is required");

    let options = LoadOptions {
        model_path: model_path.clone(),
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
        lcp_resume_budget: crate::serve::kv_persist::lcp_registry::LcpResumeBudget::DEFAULT,
    };
    let LoadedModel::Qwen35(mut qwen) = LoadedModel::load(&options).expect("load Qwen3.8") else {
        panic!("HF2Q_QWEN38_GGUF must name a Qwen3.5-family GGUF");
    };
    assert!(
        qwen.model.mtp.is_some(),
        "the diagnostic needs the native MTP head"
    );
    qwen.speculation = speculation(true);
    let mut forward = mma_forward();
    // Width sixteen is a diagnostic mul_mv_ext pack; the few-row MMA route
    // remains bounded to its published serving envelope.
    forward.options.projection = JointProjectionKernel::Canonical;
    assert_eq!(qwen.model.joint_decode_mode(), Ok(forward.mode));
    qwen.model
        .ensure_gpu_cache_primed()
        .expect("prime GPU cache at the model's final address");
    let prompt_token = *qwen
        .tokenizer
        .encode("x", false)
        .expect("tokenize diagnostic seed")
        .get_ids()
        .first()
        .expect("diagnostic seed token");
    let supervisor = EngineSupervisor::new();
    let timed_runs = if grid == "full" { 7 } else { 1 };
    let contexts = if grid == "full" {
        CONTEXTS
    } else {
        &CONTEXTS[..1]
    };
    let mut samples = Vec::new();

    for &context in contexts {
        let mut kv = qwen
            .model
            .with_gpu_cache_mut(|device, _registry| {
                HybridKvCache::new_with_options(
                    &qwen.model.cfg,
                    device,
                    u32::try_from(context + 32).expect("context fits u32"),
                    SLOTS,
                    true,
                )
            })
            .expect("allocate four-slot diagnostic cache");
        for &(shape, caps) in SHAPES
            .iter()
            .filter(|(shape, _)| grid == "full" || matches!(*shape, "4x1" | "4x4"))
        {
            for repetition in 0..=timed_runs {
                let mut states = caps
                    .iter()
                    .enumerate()
                    .map(|(slot, _)| {
                        let slot = u32::try_from(slot).expect("slot fits u32");
                        kv.reset_for_slot(SlotId(slot))
                            .expect("reset diagnostic slot");
                        let plan = LanePlan {
                            slot,
                            prompt: vec![prompt_token],
                            max_tokens: 1_024,
                            penalty: false,
                            mtp: true,
                            join_turn: 0,
                        };
                        let mut state = seed(&qwen, &mut kv, &plan, true);
                        kv.append_for_seq(
                            SlotId(slot),
                            u32::try_from(context - 1).expect("context fits u32"),
                        )
                        .expect("advance synthetic context cursor");
                        state.prompt_len = context;
                        state.decode_position_base = context;
                        state.cached_tokens = context;
                        state
                    })
                    .collect::<Vec<_>>();
                let before = states
                    .iter()
                    .map(|state| state.generated_tokens.len())
                    .collect::<Vec<_>>();
                let mut state_refs = states.iter_mut().collect::<Vec<_>>();
                let mut timing = Qwen35JointMtpStageTiming::default();
                let complete_round_started = Instant::now();
                let ticks = run_joint_mtp_step(
                    &qwen,
                    &mut kv,
                    &mut state_refs,
                    forward,
                    &supervisor,
                    &mut |qwen, kv, requests| qwen.draft_joint_mtp(kv, requests, &supervisor),
                    &|_| false,
                    Some(caps),
                    Some(&mut timing),
                )
                .unwrap_or_else(|error| panic!("{shape} at {context}: {error:#}"));
                drop(state_refs);
                let drain_started = Instant::now();
                for (state, tick) in states.iter_mut().zip(ticks) {
                    let mut tick = tick.unwrap_or_else(|error| {
                        panic!("{shape} slot {}: {error:#}", state.slot_id.0)
                    });
                    while !tick.finished && state.has_pending_speculation_output() {
                        tick = state
                            .joint_mtp_drain_tick(&qwen)
                            .unwrap_or_else(|error| panic!("{shape} drain: {error:#}"));
                    }
                }
                timing.settle_catch_up += drain_started.elapsed();
                timing.total = complete_round_started.elapsed();
                let committed_tokens = states
                    .iter()
                    .zip(before)
                    .map(|(state, prior)| state.generated_tokens.len().saturating_sub(prior))
                    .collect::<Vec<_>>();
                samples.push(DiagnosticSample {
                    context_tokens: context,
                    shape,
                    rows: caps.iter().map(|cap| cap + 1).sum(),
                    warmup: repetition == 0,
                    repetition,
                    committed_tokens,
                    prepare_ns: duration_ns(timing.prepare),
                    draft_ns: duration_ns(timing.draft),
                    verify_readback_ns: duration_ns(timing.verify_readback),
                    selection_ns: duration_ns(timing.selection),
                    capture_recovery_ns: duration_ns(timing.capture_recovery),
                    settle_catch_up_ns: duration_ns(timing.settle_catch_up),
                    total_ns: duration_ns(timing.total),
                });
            }
        }
    }

    let report = DiagnosticReport {
        schema: "hf2q.adr065.joint-verify-timing.v1",
        source_commit,
        model_path: model_path.display().to_string(),
        q5_case_path: q5_case_path.display().to_string(),
        q5_case_sha256: hex::encode(Sha256::digest(&q5_case)),
        grid,
        warmups_per_configuration: 1,
        timed_runs_per_configuration: timed_runs,
        samples,
    };
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("create {}: {error}", parent.display()));
    }
    fs::write(
        &output_path,
        serde_json::to_vec_pretty(&report).expect("serialize diagnostic report"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", output_path.display()));
    eprintln!("wrote {}", output_path.display());
}
