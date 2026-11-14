//! ADR-063 Q4-GEMMA-COLD: real-artifact bitwise gates (design §6.2).
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF. Run
//! each test alone with `--exact --ignored --nocapture --test-threads=1`.
//!
//! `HF2Q_GEMMA_COLD_TURN0_PROMPTS` names a JSON array of the four actual
//! four-agent turn-0 prompts, each `{"prompt_tokens": [...],
//! "stable_prompt_prefix_tokens": S}` (S 458, P 465 on the #660 cases). When
//! it is unset the gates use four synthetic turn-0-shaped prompts that share
//! everything but one "agent id" token, print
//! `GEMMA_JOINT_COLD_TURN0_SOURCE=synthetic`, and the receipt is not the
//! official turn-0 record.
//! `gemma_joint_cold_capture_turn0_prompts` (run first in the gate session)
//! writes that file from a recorded four-agent run.
//!
//! * `gemma_joint_cold_prefill_matches_per_slot_bitwise`: transaction level,
//!   per-slot arm (`prefill_seed`) then cohort arm (Phase B, anchors, Phase
//!   C, commit) on reset slots: first token, logits bits, cursors, every
//!   layer's KV bytes through P, anchor bytes, and 16 greedy decode rows.
//! * `gemma_joint_cold_prefill_deterministic`: 20 cohort repeats of the
//!   turn-0 set, byte-identical to the first.
//! * `gemma_cold_multi_seq_matches_per_slot_bitwise`: the D2 cutover — the
//!   iter-G(a) cold batch equals per-slot cold prefill bitwise, and a 32-row
//!   lane takes the per-slot route; installed long prompts (#725), driven
//!   through every chunk by `advance_gemma4_prefill_quantum` (two 6000-token
//!   prompts, a lane with exactly 32 remaining rows, a joint cursor-0 chunk
//!   at a stable boundary), equal each prompt prefilled alone.
//! * `gemma_joint_cold_engine_cohort_matches_serial_turn0`: engine level,
//!   switch on against off, then turn 1 (joint warm) from both.
//! * `gemma_joint_cold_forward_fault_retries_each_lane_per_slot`,
//!   `gemma_joint_cold_cue_fault_restores_from_boundary_anchors`,
//!   `gemma_joint_cold_cancel_during_transaction`: failure semantics.
//! * `gemma_stable_warm_group_is_independent_of_the_admission_window` (#728):
//!   a cached stable-boundary group is admitted per slot, bitwise equal at
//!   coalesce windows 0 (serial) and 100 ms (concurrent), anchors included.
use super::gemma_cold_batch::{self, CueRoute, StableContext, StableLane};
use super::gemma_cold_stable_batch::{self as cohort, Lane};
use super::gemma_warm_batch_tests::{
    assert_same, lane_image, load_gate_model, load_options, runtime, synthetic_prompt, LaneImage,
};
use super::*;
use std::sync::atomic::Ordering;

const MAX_DECODE: usize = 16;
const DECODE_ROWS: usize = 16;

/// Render a recorded chat-completions request as `chat_completions` does
/// for a text Gemma 4 request (tools in scope, template-default thinking, no
/// kwargs): the prompt with the generation cue, and its stable boundary,
/// the cue-less render's length when that is a strict token prefix.
fn render_recorded_turn(model: &GemmaLoadedModel, request: &std::path::Path) -> (Vec<u32>, usize) {
    let wire = std::fs::read(request).unwrap_or_else(|e| panic!("read {request:?}: {e}"));
    let req: crate::serve::api::schema::ChatCompletionRequest =
        serde_json::from_slice(&wire).unwrap_or_else(|e| panic!("parse {request:?}: {e}"));
    assert!(
        req.chat_template_kwargs.is_none(),
        "{request:?} carries chat_template_kwargs"
    );
    let template = model.chat_template.as_str();
    let thinking = crate::serve::template_supports_enable_thinking(template);
    let tokens = |add_generation_prompt| {
        let rendered = render_chat_prompt_with_tools_generation_prompt(
            template,
            &req.messages,
            req.tools.as_deref(),
            thinking,
            None,
            add_generation_prompt,
        )
        .unwrap_or_else(|e| panic!("render {request:?}: {e:#}"));
        model
            .tokenizer
            .encode(rendered, false)
            .unwrap_or_else(|e| panic!("tokenize {request:?}: {e}"))
            .get_ids()
            .to_vec()
    };
    let prompt = tokens(true);
    let stable = tokens(false);
    assert!(
        !stable.is_empty() && stable.len() < prompt.len() && prompt.starts_with(&stable),
        "{request:?}: the cue-less render is not a strict prefix (no stable boundary)"
    );
    (prompt, stable.len())
}

#[test]
#[ignore = "official Gemma 4 GGUF: write HF2Q_GEMMA_COLD_TURN0_PROMPTS from a recorded four-agent run"]
fn gemma_joint_cold_capture_turn0_prompts() {
    let run = PathBuf::from(
        std::env::var("HF2Q_GEMMA_COLD_TURN0_RUN")
            .expect("HF2Q_GEMMA_COLD_TURN0_RUN names a recorded run directory (…/<run>-hf2q-…)"),
    );
    let out = PathBuf::from(
        std::env::var("HF2Q_GEMMA_COLD_TURN0_PROMPTS")
            .expect("HF2Q_GEMMA_COLD_TURN0_PROMPTS names the JSON file to write"),
    );
    let loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
    let LoadedModel::Gemma(model) = &loaded else {
        panic!("HF2Q_GEMMA4_GGUF is not a Gemma 4 artifact")
    };
    let mut lanes = Vec::new();
    for agent in 0..4 {
        let turn = run.join(format!("client/agent-{agent}/turn-0"));
        let request = turn.join("request.json");
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(turn.join("receipt.json")).expect("read turn-0 receipt.json"),
        )
        .expect("parse turn-0 receipt.json");
        let recorded = receipt["usage"]["prompt_tokens"]
            .as_u64()
            .expect("receipt usage.prompt_tokens") as usize;
        let (prompt, boundary) = render_recorded_turn(model, &request);
        assert_eq!(
            prompt.len(),
            recorded,
            "agent {agent}: rendered {} tokens, recorded prompt_tokens {recorded}",
            prompt.len()
        );
        let request_sha256 = {
            use sha2::{Digest, Sha256};
            hex::encode(Sha256::digest(std::fs::read(&request).unwrap()))
        };
        println!(
            "GEMMA_JOINT_COLD_TURN0_LANE agent={agent} prompt_tokens={} boundary={boundary} request_sha256={request_sha256}",
            prompt.len()
        );
        lanes.push(serde_json::json!({
            "agent": agent,
            "request": request,
            "request_sha256": request_sha256,
            "prompt_tokens": prompt,
            "stable_prompt_prefix_tokens": boundary,
        }));
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&lanes).unwrap())
        .unwrap_or_else(|e| panic!("write {out:?}: {e}"));
    println!(
        "GEMMA_JOINT_COLD_TURN0_CAPTURED path={} lanes=4",
        out.display()
    );
}

/// One gate lane: `(prompt, S)`.
type Shape = (Vec<u32>, usize);

fn params(boundary: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens: MAX_DECODE,
        stable_prompt_prefix_tokens: Some(boundary),
        ..SamplingParams::default()
    }
}

/// The four turn-0 prompts and their source label.
fn turn0(vocab: usize) -> (Vec<Shape>, &'static str) {
    if let Ok(path) = std::env::var("HF2Q_GEMMA_COLD_TURN0_PROMPTS") {
        let text = std::fs::read_to_string(&path).expect("read HF2Q_GEMMA_COLD_TURN0_PROMPTS");
        let value: serde_json::Value = serde_json::from_str(&text).expect("turn-0 prompts JSON");
        let lanes: Vec<Shape> = value
            .as_array()
            .expect("turn-0 prompts: array")
            .iter()
            .map(|lane| {
                let tokens = lane["prompt_tokens"]
                    .as_array()
                    .expect("prompt_tokens")
                    .iter()
                    .map(|token| token.as_u64().expect("token id") as u32)
                    .collect();
                let boundary = lane["stable_prompt_prefix_tokens"]
                    .as_u64()
                    .expect("stable_prompt_prefix_tokens") as usize;
                (tokens, boundary)
            })
            .collect();
        assert_eq!(lanes.len(), 4, "four turn-0 prompts");
        return (lanes, "case");
    }
    let base = synthetic_prompt(0, 465, vocab);
    let lanes = (0..4)
        .map(|agent| {
            let mut prompt = base.clone();
            // The agent id: one token inside the shared system prompt.
            prompt[40] = (2_000 + agent) as u32;
            (prompt, 458)
        })
        .collect();
    (lanes, "synthetic")
}

/// Synthetic lanes `(S, S + cue)`, distinct content per lane.
fn synthetic(shapes: &[(usize, usize)], vocab: usize) -> Vec<Shape> {
    shapes
        .iter()
        .enumerate()
        .map(|(lane, &(boundary, cue))| {
            (synthetic_prompt(lane + 1, boundary + cue, vocab), boundary)
        })
        .collect()
}

/// The production planner's routes for these lanes (all slots idle).
fn planned(lanes: &[Shape]) -> Vec<StableLane> {
    let params: Vec<SamplingParams> = lanes.iter().map(|(_, s)| params(*s)).collect();
    gemma_cold_batch::stable_plan(
        &StableContext {
            worker_active: true,
            all_slots_idle: true,
            graft_len: 0,
            max_seq_len: usize::MAX / 2,
        },
        lanes
            .iter()
            .map(|(prompt, _)| prompt.as_slice())
            .zip(&params),
        |_| None,
        |_| true,
    )
    .expect("gate shape is a cold stable cohort")
    .lanes
}

thread_local! {
    /// Each slot's sliding-ring bytes as loaded, captured at its first reset
    /// (before any gate writes it).
    static PRISTINE: std::cell::RefCell<
        Vec<Option<crate::inference::models::gemma4::kv_cache::GemmaHybridSlotAnchor>>,
    > = const { std::cell::RefCell::new(Vec::new()) };
}

/// Reset `slot` and return its sliding rings to their as-loaded bytes, so
/// both arms start from identical slot memory. The anchor gate compares the
/// whole saved ring (rows past the boundary included); without this, the
/// rows past S would carry whichever arm last wrote the slot.
fn reset_slot(guard: &mut Gemma4KvGuard<'_>, slot: usize) {
    use crate::inference::models::gemma4::kv_cache::{
        restore_gemma_hybrid_slot_anchor, snapshot_gemma_hybrid_slot_anchor,
    };
    let slot_id = SlotId(slot as u32);
    PRISTINE.with(|pristine| {
        let mut pristine = pristine.borrow_mut();
        if pristine.len() <= slot {
            pristine.resize_with(slot + 1, || None);
        }
        let hybrid = guard.hybrid.as_mut().unwrap();
        match &pristine[slot] {
            None => {
                pristine[slot] =
                    Some(snapshot_gemma_hybrid_slot_anchor(hybrid, slot_id, 1).unwrap())
            }
            Some(anchor) => restore_gemma_hybrid_slot_anchor(hybrid, slot_id, anchor, 0).unwrap(),
        }
    });
    for buf in guard.kv.iter_mut() {
        buf.reset_for_slot(slot_id).unwrap();
    }
    for buf in guard.hybrid.as_mut().unwrap().iter_mut() {
        buf.reset_for_slot(slot_id).unwrap();
    }
    install_gemma4_slot_cursor(guard, slot_id, 0).unwrap();
}

#[derive(Debug, PartialEq, Eq)]
struct ColdImage {
    lane: LaneImage,
    anchor: Option<[u8; 32]>,
}

fn assert_cold_equal(label: &str, joint: &ColdImage, reference: &ColdImage) {
    super::gemma_warm_batch_tests::assert_lane_equal(label, &joint.lane, &reference.lane);
    assert!(
        reference.anchor.is_some(),
        "{label}: reference anchor captured"
    );
    assert_eq!(joint.anchor, reference.anchor, "{label}: anchor bytes");
}

/// 16 greedy steps of every lane together through the batched decode body
/// (the decode that follows admission); logits bits per step and lane.
fn decode_rows(guard: &mut Gemma4KvGuard<'_>, lanes: &[(usize, usize, u32)]) -> Vec<Vec<Vec<u32>>> {
    let vocab = guard.model.weights.vocab_size;
    let slot_ids: Vec<SlotId> = lanes
        .iter()
        .map(|&(slot, _, _)| SlotId(slot as u32))
        .collect();
    let mut tokens: Vec<u32> = lanes.iter().map(|&(_, _, token)| token).collect();
    let mut rows = vec![Vec::new(); lanes.len()];
    for step in 0..DECODE_ROWS {
        let positions: Vec<usize> = lanes.iter().map(|&(_, p, _)| p + step).collect();
        let mut regime = crate::inference::models::gemma4::batched_body::BatchedKvRegime::Hybrid(
            guard.hybrid.as_mut().unwrap().as_mut_slice(),
        );
        let mut head = None;
        let lm = &mut *guard.model;
        let hidden = lm
            .weights
            .forward_decode_body_batched(
                &tokens,
                &slot_ids,
                &positions,
                &mut regime,
                &mut head,
                &mut lm.ctx,
            )
            .unwrap();
        let head = match head {
            Some(head) => head,
            None => lm
                .weights
                .lm_head_batched(&hidden, lanes.len(), &mut lm.ctx)
                .unwrap(),
        };
        for (lane, row) in head.logits.chunks(vocab).enumerate() {
            rows[lane].push(row.iter().map(|value| value.to_bits()).collect());
            tokens[lane] = argmax_f32_first_max(row).0;
        }
    }
    rows
}

/// Per-slot arm: `prefill_seed` of each lane alone on a reset slot (the
/// `admit_gemma4_slot` reference), then the following decode.
fn per_slot_arm(
    guard: &mut Gemma4KvGuard<'_>,
    supervisor: &EngineSupervisor,
    lanes: &[Shape],
) -> (Vec<ColdImage>, Vec<Vec<Vec<u32>>>) {
    let mut images = Vec::new();
    let mut seeds = Vec::new();
    for (slot, (prompt, boundary)) in lanes.iter().enumerate() {
        reset_slot(guard, slot);
        clear_gemma4_self_mounts(guard.model);
        let (state, anchor) = Gemma4DecodeState::prefill_seed(
            guard.model,
            prompt,
            &[],
            &params(*boundary),
            None,
            SlotId(slot as u32),
            &mut guard.kv,
            guard.hybrid.as_mut(),
            guard.dense.as_mut(),
            guard.mlx.as_mut(),
            supervisor,
            0,
        )
        .unwrap();
        clear_gemma4_self_mounts(guard.model);
        let logits = guard.model.weights.logits_view().unwrap().to_vec();
        commit_gemma4_slot_cursor(guard, SlotId(slot as u32), 0, prompt.len()).unwrap();
        images.push(ColdImage {
            lane: lane_image(guard, slot, prompt.len(), state.next_token, &logits),
            anchor: anchor.as_ref().map(|anchor| anchor.digest()),
        });
        seeds.push((slot, prompt.len(), state.next_token));
    }
    let decode = decode_rows(guard, &seeds);
    (images, decode)
}

/// Gate lanes with test replies and handles (no scheduler is involved).
fn gate_lanes(lanes: &[Shape]) -> Vec<Lane> {
    lanes
        .iter()
        .zip(planned(lanes))
        .enumerate()
        .map(|(slot, ((prompt, boundary), shape))| {
            let (reply, _) = oneshot::channel();
            Lane {
                handle: SlotHandle {
                    slot_id: SlotId(slot as u32),
                    generation: 0,
                },
                prompt: prompt.clone(),
                params: params(*boundary),
                reply: SlotReply::Unary(reply),
                shape,
                anchor: None,
                seed: None,
            }
        })
        .collect()
}

/// Cohort arm: reset, Phase B, anchors (FIFO), Phase C, commit (0, P), as
/// `gemma_cold_stable_batch::admit_cohort`, then the following decode.
fn cohort_arm(
    guard: &mut Gemma4KvGuard<'_>,
    supervisor: &EngineSupervisor,
    lanes: &[Shape],
) -> (Vec<ColdImage>, Vec<Vec<Vec<u32>>>) {
    let mut lanes = gate_lanes(lanes);
    for slot in 0..lanes.len() {
        reset_slot(guard, slot);
    }
    clear_gemma4_self_mounts(guard.model);
    cohort::boundary_forward(guard, supervisor, &lanes).unwrap();
    for lane in &mut lanes {
        lane.anchor = cohort::capture_boundary_anchor(guard, lane).unwrap();
    }
    cohort::encode_cues(guard, supervisor, &mut lanes).unwrap();
    clear_gemma4_self_mounts(guard.model);
    let commits: Vec<_> = lanes
        .iter()
        .map(|lane| (lane.handle.slot_id, 0, lane.shape.prompt_len))
        .collect();
    commit_gemma4_slot_cursors(guard, &commits).unwrap();
    let mut images = Vec::new();
    let mut seeds = Vec::new();
    for (slot, lane) in lanes.iter().enumerate() {
        let (token, logits) = lane.seed.as_ref().expect("cue seeded the lane");
        images.push(ColdImage {
            lane: lane_image(guard, slot, lane.shape.prompt_len, *token, logits),
            anchor: lane.anchor.as_ref().map(|anchor| anchor.kv.digest()),
        });
        seeds.push((slot, lane.shape.prompt_len, *token));
    }
    let decode = decode_rows(guard, &seeds);
    (images, decode)
}

fn gate_guard(loaded: &mut LoadedModel) -> Gemma4KvGuard<'_> {
    let LoadedModel::Gemma(g) = loaded else {
        unreachable!("load_gate_model returns Gemma")
    };
    let guard = Gemma4KvGuard::take(g).unwrap();
    assert_eq!(guard.model.weights.kv_graft_len(), 0, "no graft bound");
    guard
}

#[test]
#[ignore = "official Gemma 4 GGUF: cold stable cohort lanes equal per-slot admission bit for bit"]
fn gemma_joint_cold_prefill_matches_per_slot_bitwise() {
    let mut loaded = load_gate_model();
    let mut guard = gate_guard(&mut loaded);
    let supervisor = EngineSupervisor::new();
    let vocab = guard.model.weights.vocab_size;
    let (turn0, source) = turn0(vocab);
    println!("GEMMA_JOINT_COLD_TURN0_SOURCE={source}");
    let sets: Vec<(&str, Vec<Shape>, Vec<CueRoute>)> = vec![
        ("turn0", turn0, vec![CueRoute::Replay; 4]),
        (
            "lanes{33,64,65,458}+7",
            synthetic(&[(33, 7), (64, 7), (65, 7), (458, 7)], vocab),
            vec![CueRoute::Replay; 4],
        ),
        (
            "lanes{1100,900}+1",
            synthetic(&[(1100, 1), (900, 1)], vocab),
            vec![CueRoute::Replay; 2],
        ),
        (
            "lanes{200,300}+31",
            synthetic(&[(200, 31), (300, 31)], vocab),
            vec![CueRoute::Replay; 2],
        ),
        (
            "lanes{500,600}+40",
            synthetic(&[(500, 40), (600, 40)], vocab),
            vec![CueRoute::JointLive; 2],
        ),
        (
            "lanes{300,400}+32",
            synthetic(&[(300, 32), (400, 32)], vocab),
            vec![CueRoute::PerSlot; 2],
        ),
        (
            "mixed{400+7,500+40}",
            synthetic(&[(400, 7), (500, 40)], vocab),
            vec![CueRoute::Replay, CueRoute::PerSlot],
        ),
    ];
    for (label, lanes, routes) in &sets {
        assert_eq!(
            planned(lanes)
                .iter()
                .map(|lane| lane.cue)
                .collect::<Vec<_>>(),
            *routes,
            "{label}: planner routes"
        );
        let (reference, reference_decode) = per_slot_arm(&mut guard, &supervisor, lanes);
        let (joint, joint_decode) = cohort_arm(&mut guard, &supervisor, lanes);
        for (lane, (joint, reference)) in joint.iter().zip(&reference).enumerate() {
            assert_cold_equal(
                &format!(
                    "{label} lane {lane} S={} P={}",
                    lanes[lane].1,
                    lanes[lane].0.len()
                ),
                joint,
                reference,
            );
        }
        for (lane, (joint, reference)) in joint_decode.iter().zip(&reference_decode).enumerate() {
            let step = joint.iter().zip(reference).position(|(a, b)| a != b);
            assert!(
                step.is_none(),
                "{label} lane {lane}: decode logits differ at step {step:?}"
            );
        }
        println!(
            "GEMMA_JOINT_COLD_SET_EQUAL set={label} lanes={}",
            lanes.len()
        );
    }
    println!(
        "GEMMA_JOINT_COLD_BITWISE_COMPLETE turn0_source={source} sets={} logits_bits=true kv_bytes=true anchors=true cursors=true decode_rows={DECODE_ROWS}",
        sets.len()
    );
}

#[test]
#[ignore = "official Gemma 4 GGUF: 20 cold stable cohort repeats are byte-identical"]
fn gemma_joint_cold_prefill_deterministic() {
    let mut loaded = load_gate_model();
    let mut guard = gate_guard(&mut loaded);
    let supervisor = EngineSupervisor::new();
    let (lanes, source) = turn0(guard.model.weights.vocab_size);
    let first = cohort_arm(&mut guard, &supervisor, &lanes);
    for repeat in 1..20 {
        let again = cohort_arm(&mut guard, &supervisor, &lanes);
        assert!(again == first, "repeat {repeat} differs from repeat 0");
    }
    println!(
        "GEMMA_JOINT_COLD_DETERMINISM_COMPLETE turn0_source={source} repeats=20 identical=true"
    );
}

/// G-S1 per-slot arm: the four serial admissions' GPU work (reset, the
/// `prefill_seed` boundary forward, anchor and cue, cursor commit).
fn time_per_slot(
    guard: &mut Gemma4KvGuard<'_>,
    supervisor: &EngineSupervisor,
    lanes: &[Shape],
) -> f64 {
    let started = Instant::now();
    for (slot, (prompt, boundary)) in lanes.iter().enumerate() {
        reset_slot(guard, slot);
        clear_gemma4_self_mounts(guard.model);
        let seed = Gemma4DecodeState::prefill_seed(
            guard.model,
            prompt,
            &[],
            &params(*boundary),
            None,
            SlotId(slot as u32),
            &mut guard.kv,
            guard.hybrid.as_mut(),
            guard.dense.as_mut(),
            guard.mlx.as_mut(),
            supervisor,
            0,
        )
        .unwrap();
        clear_gemma4_self_mounts(guard.model);
        commit_gemma4_slot_cursor(guard, SlotId(slot as u32), 0, prompt.len()).unwrap();
        drop(seed);
    }
    started.elapsed().as_secs_f64() * 1000.0
}

/// G-S1 cohort arm: reset, Phase B, anchors, Phase C, commit. Returns
/// `(total, boundary, anchors, cue)` milliseconds.
fn time_cohort(
    guard: &mut Gemma4KvGuard<'_>,
    supervisor: &EngineSupervisor,
    lanes: &[Shape],
) -> (f64, f64, f64, f64) {
    let mut lanes = gate_lanes(lanes);
    let started = Instant::now();
    for slot in 0..lanes.len() {
        reset_slot(guard, slot);
    }
    clear_gemma4_self_mounts(guard.model);
    cohort::boundary_forward(guard, supervisor, &lanes).unwrap();
    let boundary = started.elapsed();
    for lane in &mut lanes {
        lane.anchor = cohort::capture_boundary_anchor(guard, lane).unwrap();
    }
    let anchors = started.elapsed();
    cohort::encode_cues(guard, supervisor, &mut lanes).unwrap();
    clear_gemma4_self_mounts(guard.model);
    let commits: Vec<_> = lanes
        .iter()
        .map(|lane| (lane.handle.slot_id, 0, lane.shape.prompt_len))
        .collect();
    commit_gemma4_slot_cursors(guard, &commits).unwrap();
    let total = started.elapsed();
    let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
    (
        ms(total),
        ms(boundary),
        ms(anchors - boundary),
        ms(total - anchors),
    )
}

/// G-S1 (design §7.1): turn-0 transaction, cohort against the per-slot sum,
/// 10 + 10 repeats in ABBA order after one untimed warm-up of each arm.
/// Diagnostic gate: cohort median ≤ 0.80 × per-slot median.
#[test]
#[ignore = "official Gemma 4 GGUF: G-S1 cold cohort transaction timing (GPU lane, quiet host)"]
fn gemma_joint_cold_gs1_transaction_timing() {
    let mut loaded = load_gate_model();
    let mut guard = gate_guard(&mut loaded);
    let supervisor = EngineSupervisor::new();
    let (lanes, source) = turn0(guard.model.weights.vocab_size);
    time_per_slot(&mut guard, &supervisor, &lanes);
    time_cohort(&mut guard, &supervisor, &lanes);
    let (mut serial, mut joint) = (Vec::new(), Vec::new());
    for block in 0..5 {
        for arm in [0, 1, 1, 0] {
            if arm == 0 {
                let ms = time_per_slot(&mut guard, &supervisor, &lanes);
                println!("GEMMA_COLD_GS1_RECORD {{\"block\":{block},\"arm\":\"per_slot\",\"ms\":{ms:.4}}}");
                serial.push(ms);
            } else {
                let (ms, b, a, c) = time_cohort(&mut guard, &supervisor, &lanes);
                println!(
                    "GEMMA_COLD_GS1_RECORD {{\"block\":{block},\"arm\":\"cohort\",\"ms\":{ms:.4},\"boundary_ms\":{b:.4},\"anchor_ms\":{a:.4},\"cue_ms\":{c:.4}}}"
                );
                joint.push(ms);
            }
        }
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    };
    let (serial_ms, joint_ms) = (median(&mut serial), median(&mut joint));
    let ratio = joint_ms / serial_ms;
    println!(
        "GEMMA_COLD_GS1_SUMMARY turn0_source={source} per_slot_median_ms={serial_ms:.2} cohort_median_ms={joint_ms:.2} ratio={ratio:.4} bound=0.80 verdict={}",
        if ratio <= 0.80 { "pass" } else { "fail" }
    );
}

/// Per-slot cold prefill of whole prompts (no boundary): the
/// `forward_prefill_with_soft_tokens_slot_aware` reference.
fn cold_per_slot(guard: &mut Gemma4KvGuard<'_>, prompts: &[Vec<u32>]) -> Vec<LaneImage> {
    prompts
        .iter()
        .enumerate()
        .map(|(slot, prompt)| {
            reset_slot(guard, slot);
            clear_gemma4_self_mounts(guard.model);
            let first = guard
                .model
                .weights
                .forward_prefill_with_soft_tokens_slot_aware(
                    prompt,
                    &[],
                    MAX_DECODE,
                    &mut guard.model.ctx,
                    SlotId(slot as u32),
                    &mut guard.kv,
                    guard.hybrid.as_mut(),
                    guard.dense.as_mut(),
                    guard.mlx.as_mut(),
                )
                .unwrap();
            clear_gemma4_self_mounts(guard.model);
            commit_gemma4_slot_cursor(guard, SlotId(slot as u32), 0, prompt.len()).unwrap();
            let logits = guard.model.weights.logits_view().unwrap().to_vec();
            lane_image(guard, slot, prompt.len(), first, &logits)
        })
        .collect()
}

fn cold_multi_seq(guard: &mut Gemma4KvGuard<'_>, prompts: &[Vec<u32>]) -> Vec<LaneImage> {
    for slot in 0..prompts.len() {
        reset_slot(guard, slot);
    }
    let seqs: Vec<(Vec<u32>, SlotId)> = prompts
        .iter()
        .enumerate()
        .map(|(slot, prompt)| (prompt.clone(), SlotId(slot as u32)))
        .collect();
    clear_gemma4_self_mounts(guard.model);
    let output = guard
        .model
        .weights
        .forward_prefill_batched_multi_seq(
            &seqs,
            guard.hybrid.as_deref().unwrap(),
            MAX_DECODE,
            &mut guard.model.ctx,
        )
        .unwrap();
    clear_gemma4_self_mounts(guard.model);
    let commits: Vec<_> = prompts
        .iter()
        .enumerate()
        .map(|(slot, prompt)| (SlotId(slot as u32), 0, prompt.len()))
        .collect();
    commit_gemma4_slot_cursors(guard, &commits).unwrap();
    prompts
        .iter()
        .enumerate()
        .map(|(slot, prompt)| {
            lane_image(
                guard,
                slot,
                prompt.len(),
                output.first_tokens[slot],
                &output.logits[slot],
            )
        })
        .collect()
}

#[test]
#[ignore = "official Gemma 4 GGUF: the per-sequence cold multi-sequence prefill equals per-slot cold prefill bit for bit"]
fn gemma_cold_multi_seq_matches_per_slot_bitwise() {
    let mut loaded = load_gate_model();
    let mut guard = gate_guard(&mut loaded);
    let vocab = guard.model.weights.vocab_size;
    // iter-G(a) cold batch lanes (no boundary).
    let prompts: Vec<Vec<u32>> = [33usize, 64, 458, 900]
        .iter()
        .enumerate()
        .map(|(lane, &len)| synthetic_prompt(lane + 5, len, vocab))
        .collect();
    assert!(
        !gemma_cold_batch::plan(prompts.iter().map(|p| (p.len(), 0)), 0).requires_slot_admission()
    );
    let reference = cold_per_slot(&mut guard, &prompts);
    let joint = cold_multi_seq(&mut guard, &prompts);
    for (lane, (joint, reference)) in joint.iter().zip(&reference).enumerate() {
        super::gemma_warm_batch_tests::assert_lane_equal(
            &format!("iter-G(a) lane {lane} rows {}", prompts[lane].len()),
            joint,
            reference,
        );
    }
    // The D2 floor: a group with a 32-row lane is admitted per slot.
    assert!(gemma_cold_batch::plan([(32, 0), (458, 0)], 0).has_tiny_prefill);

    // Issue #725: installed long prompts, driven through every chunk by
    // `advance_gemma4_prefill_quantum`, equal each prompt prefilled alone.
    let supervisor = EngineSupervisor::new();
    let long = |seed: usize, len: usize| synthetic_prompt(seed, len, vocab);
    // `(label, lanes (prompt, stable boundary), expected chunks per lane as
    // (start, end, joint))`.
    type Chunks = Vec<(usize, usize, bool)>;
    let cases: Vec<(&str, Vec<(Vec<u32>, Option<usize>)>, Vec<Chunks>)> = vec![
        (
            "two 6000-token prompts",
            vec![(long(9, 6_000), None), (long(10, 6_000), None)],
            vec![vec![(0, 4_096, false), (4_096, 6_000, true)]; 2],
        ),
        (
            "a lane with exactly 32 remaining rows",
            vec![
                (long(9, 6_000), None),
                (long(11, 4_128), None),
                (long(10, 6_000), None),
            ],
            vec![
                vec![(0, 4_096, false), (4_096, 6_000, true)],
                vec![(0, 4_096, false), (4_096, 4_128, false)],
                vec![(0, 4_096, false), (4_096, 6_000, true)],
            ],
        ),
        (
            "stable boundary 1000 (joint cursor-0 chunk)",
            vec![
                (long(9, 6_000), Some(1_000)),
                (long(10, 6_000), Some(1_000)),
            ],
            vec![vec![(0, 1_000, true), (1_000, 5_096, false), (5_096, 6_000, true)]; 2],
        ),
    ];
    for (label, lanes, expected) in &cases {
        let all: Vec<usize> = (0..lanes.len()).collect();
        let joint = installed_prefill(&mut guard, &supervisor, lanes, &all);
        for (slot, (image, chunks)) in joint.iter().enumerate() {
            assert_eq!(chunks, &expected[slot], "{label}: lane {slot} chunks");
            let alone = installed_prefill(&mut guard, &supervisor, lanes, &[slot]);
            let (reference, alone_chunks) = &alone[0];
            let alone_expected: Chunks = expected[slot]
                .iter()
                .map(|&(start, end, _)| (start, end, false))
                .collect();
            assert_eq!(
                alone_chunks, &alone_expected,
                "{label}: lane {slot} alone chunks"
            );
            super::gemma_warm_batch_tests::assert_lane_equal(
                &format!("{label}: installed lane {slot}"),
                image,
                reference,
            );
        }
    }
    println!(
        "GEMMA_COLD_MULTI_SEQ_BITWISE_COMPLETE iter_g_a_lanes=[33,64,458,900] installed_cases={} floor_33=true",
        cases.len()
    );
}

/// ADR-063 Q4-GEMMA host P1 on the cold routes: per-slot cold prefills and
/// the per-sequence cold multi-sequence prefill give the same bytes when
/// every `Init::Overwritten` activation is handed out `0xFF`-filled (NaN in
/// every float dtype) as when it is zero-filled (the pre-arena state).
#[test]
#[ignore = "official Gemma 4 GGUF: 0xFF-filled pooled cold prefill activations give the zero-filled bytes"]
fn gemma_cold_prefill_pool_hostile_fill_matches_zero_fill_bitwise() {
    use crate::serve::forward_prefill_batched::gemma_prefill_pool::set_overwritten_fill_for_test;
    let mut loaded = load_gate_model();
    let mut guard = gate_guard(&mut loaded);
    let vocab = guard.model.weights.vocab_size;
    // FA lanes (33, 64 rows) and tensor-mm-global lanes (458, 900 rows).
    let prompts: Vec<Vec<u32>> = [33usize, 64, 458, 900]
        .iter()
        .enumerate()
        .map(|(lane, &len)| synthetic_prompt(lane + 5, len, vocab))
        .collect();
    let mut arms = Vec::new();
    for fill in [0x00, 0xFF] {
        set_overwritten_fill_for_test(Some(fill));
        let per_slot = cold_per_slot(&mut guard, &prompts);
        let joint = cold_multi_seq(&mut guard, &prompts);
        arms.push((per_slot, joint));
    }
    set_overwritten_fill_for_test(None);
    let (zero, hostile) = (&arms[0], &arms[1]);
    for (lane, prompt) in prompts.iter().enumerate() {
        let rows = prompt.len();
        super::gemma_warm_batch_tests::assert_lane_equal(
            &format!("per-slot cold lane {lane} rows {rows}"),
            &hostile.0[lane],
            &zero.0[lane],
        );
        super::gemma_warm_batch_tests::assert_lane_equal(
            &format!("multi-seq cold lane {lane} rows {rows}"),
            &hostile.1[lane],
            &zero.1[lane],
        );
    }
}

/// Issue #725 driver: install `lanes` (prompt, stable boundary) as bounded
/// prefills on slots `0..`, keep only the `live` ones, and advance every
/// live lane in slot order through `advance_gemma4_prefill_quantum` (cross-
/// slot admission on) until all have seeded. Returns, per live lane, its
/// image and its committed chunks `(start, end, joint)`.
fn installed_prefill(
    guard: &mut Gemma4KvGuard<'_>,
    supervisor: &EngineSupervisor,
    lanes: &[(Vec<u32>, Option<usize>)],
    live: &[usize],
) -> Vec<(LaneImage, Vec<(usize, usize, bool)>)> {
    let n_slots = guard.hybrid.as_ref().unwrap()[0].n_seqs as usize;
    let mut scheduler = InflightBatchedScheduler::new_with_kv_budget(8, n_slots as u32, 0);
    scheduler.set_prefill_chunk_tokens(GEMMA4_SLOT_PREFILL_CHUNK_TOKENS);
    // A fresh scheduler mints slot ids in order.
    let handles: Vec<SlotHandle> = lanes
        .iter()
        .enumerate()
        .map(|(slot, (prompt, _))| {
            let handle = scheduler
                .admit(AdmitRequest {
                    prompt_tokens: prompt.len() as u32,
                    max_tokens: MAX_DECODE as u32,
                    kv_bytes_needed: 0,
                    prompt_kv_bytes: 0,
                    preferred_slot: None,
                })
                .unwrap()
                .handle
                .unwrap();
            assert_eq!(handle.slot_id, SlotId(slot as u32));
            handle
        })
        .collect();
    let mut slots: Vec<Option<Gemma4Slot>> = (0..n_slots).map(|_| None).collect();
    let mut receivers = Vec::new();
    for (slot, ((prompt, boundary), handle)) in lanes.iter().zip(&handles).enumerate() {
        if !live.contains(&slot) {
            scheduler.release(*handle);
            continue;
        }
        reset_slot(guard, slot);
        let params = SamplingParams {
            temperature: 0.0,
            max_tokens: MAX_DECODE,
            stable_prompt_prefix_tokens: *boundary,
            ..SamplingParams::default()
        };
        let state = Gemma4PrefillState::new(
            handle.slot_id,
            prompt.clone(),
            Vec::new(),
            params,
            0,
            None,
            *boundary,
        )
        .unwrap();
        let (send, receive) = oneshot::channel();
        receivers.push(receive);
        slots[slot] = Some((
            Gemma4SlotWork::Prefill(state),
            SlotReply::Unary(send),
            *handle,
        ));
    }
    let mut retained = vec![Vec::new(); n_slots];
    let mut anchors: Vec<Option<Gemma4PromptAnchor>> = (0..n_slots).map(|_| None).collect();
    let stats = Arc::new(Mutex::new(scheduler.stats()));
    GEMMA4_PREFILL_RECORD.with(|record| {
        *record.borrow_mut() = Some(Gemma4PrefillRecord::default());
    });
    loop {
        let mut advanced = false;
        for slot in 0..n_slots {
            let Some((Gemma4SlotWork::Prefill(state), _, handle)) = slots[slot].as_ref() else {
                continue;
            };
            let handle = *handle;
            let remaining = state.prompt_tokens.len() - state.progress.cursor();
            let requested = (remaining as u32).min(GEMMA4_SLOT_PREFILL_CHUNK_TOKENS);
            if let Some(fatal) = advance_gemma4_prefill_quantum(
                guard,
                &mut scheduler,
                &mut slots,
                &mut retained,
                &mut anchors,
                None,
                &stats,
                handle,
                requested,
                0,
                supervisor,
                true,
            ) {
                panic!("installed prefill slot {slot}: {:#}", fatal.error);
            }
            advanced = true;
        }
        if !advanced {
            break;
        }
    }
    let record = GEMMA4_PREFILL_RECORD
        .with(|record| record.borrow_mut().take())
        .unwrap();
    live.iter()
        .map(|&slot| {
            assert!(
                matches!(slots[slot], Some((Gemma4SlotWork::Decode(_), _, _))),
                "slot {slot} did not seed"
            );
            let (_, token, logits) = record
                .seeds
                .iter()
                .find(|(seeded, _, _)| *seeded == slot as u32)
                .unwrap_or_else(|| panic!("slot {slot} recorded no seed"));
            let chunks = record
                .chunks
                .iter()
                .filter(|(chunked, ..)| *chunked == slot as u32)
                .map(|&(_, start, end, joint)| (start, end, joint))
                .collect();
            (
                lane_image(guard, slot, lanes[slot].0.len(), *token, logits),
                chunks,
            )
        })
        .collect()
}

/// An engine whose worker resolves `HF2Q_GEMMA_JOINT_COLD_PREFILL=switch`.
/// The variable stays set until the caller has served and shut the engine
/// down (`shutdown`), so the worker can never start without it.
fn spawn(switch: &str) -> Engine {
    std::env::set_var(gemma_cold_batch::JOINT_COLD_PREFILL_ENV, switch);
    let loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
    Engine::spawn_with_mode(loaded, 8, None, EngineMode::SlotAware { max_slots: 4 }, &mut None)
        .expect("spawn SlotAware Gemma engine")
}

/// Shut down an engine from `spawn`, then clear the switch.
fn shutdown(rt: &tokio::runtime::Runtime, engine: Engine) {
    rt.block_on(engine.shutdown()).unwrap();
    std::env::remove_var(gemma_cold_batch::JOINT_COLD_PREFILL_ENV);
}

fn engine_turn0(engine: &Engine) -> Vec<Shape> {
    let vocab = engine.tokenizer().get_vocab_size(true);
    turn0(vocab).0
}

fn greedy(max_tokens: usize, boundary: Option<usize>) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        logprobs: true,
        top_logprobs: 4,
        stable_prompt_prefix_tokens: boundary,
        ..SamplingParams::default()
    }
}

fn concurrent(
    rt: &tokio::runtime::Runtime,
    engine: &Engine,
    lanes: &[(Vec<u32>, SamplingParams)],
) -> Vec<Result<GenerationResult>> {
    rt.block_on(async {
        let futures = lanes
            .iter()
            .map(|(prompt, params)| engine.generate(prompt.clone(), params.clone()));
        futures::future::join_all(futures).await
    })
}

/// Turn 1 of each agent: the boundary rows plus a 64-row tool result, so the
/// joint warm cohort restores every turn-0 anchor.
fn turn1(lanes: &[Shape]) -> Vec<(Vec<u32>, SamplingParams)> {
    lanes
        .iter()
        .enumerate()
        .map(|(agent, (prompt, boundary))| {
            let mut next = prompt[..*boundary].to_vec();
            next.extend((0..64).map(|i| 3_000 + ((agent * 64 + i) % 500) as u32));
            (next, greedy(16, None))
        })
        .collect()
}

fn turn0_requests(lanes: &[Shape]) -> Vec<(Vec<u32>, SamplingParams)> {
    lanes
        .iter()
        .map(|(prompt, boundary)| (prompt.clone(), greedy(18, Some(*boundary))))
        .collect()
}

fn cohorts() -> (usize, usize) {
    (
        cohort::COLD_STABLE_COHORTS.load(Ordering::Relaxed),
        cohort::COLD_STABLE_LANES.load(Ordering::Relaxed),
    )
}

/// Switch-off reference: turn 0 then turn 1, concurrently, per slot.
fn reference_turns(
    rt: &tokio::runtime::Runtime,
) -> (Vec<Shape>, Vec<GenerationResult>, Vec<GenerationResult>) {
    let engine = spawn("0");
    let lanes = engine_turn0(&engine);
    let before = cohorts();
    let t0: Vec<_> = concurrent(rt, &engine, &turn0_requests(&lanes))
        .into_iter()
        .map(Result::unwrap)
        .collect();
    let t1: Vec<_> = concurrent(rt, &engine, &turn1(&lanes))
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(cohorts(), before, "switch off never forms a cold cohort");
    shutdown(rt, engine);
    (lanes, t0, t1)
}

fn assert_turns(label: &str, actual: &[GenerationResult], expected: &[GenerationResult]) {
    assert_eq!(actual.len(), expected.len(), "{label}: lanes");
    for (agent, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_same(&format!("{label} agent {agent}"), actual, expected);
    }
}

#[test]
#[ignore = "official Gemma 4 GGUF: engine-level cold stable cohort equals serial turn 0 and the following joint warm turn"]
fn gemma_joint_cold_engine_cohort_matches_serial_turn0() {
    let rt = runtime();
    let (lanes, expected0, expected1) = reference_turns(&rt);
    for (agent, result) in expected0.iter().enumerate() {
        assert_eq!(
            result.cached_tokens, 0,
            "reference turn 0 agent {agent} cold"
        );
    }
    for (agent, result) in expected1.iter().enumerate() {
        assert_eq!(
            result.cached_tokens, lanes[agent].1,
            "reference turn 1 agent {agent} resumed S"
        );
    }
    let engine = spawn("1");
    let before = cohorts();
    let t0: Vec<_> = concurrent(&rt, &engine, &turn0_requests(&lanes))
        .into_iter()
        .map(Result::unwrap)
        .collect();
    let after = cohorts();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, 4),
        "turn 0: one cold stable cohort over 4 lanes"
    );
    assert_turns("turn 0", &t0, &expected0);
    // Turn 1 resumes every published boundary anchor (x-hf2q-cached-tokens).
    let t1: Vec<_> = concurrent(&rt, &engine, &turn1(&lanes))
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_turns("turn 1", &t1, &expected1);
    shutdown(&rt, engine);
    println!(
        "GEMMA_JOINT_COLD_ENGINE_COMPLETE cohort_lanes=4 turn0_bitwise=true turn1_bitwise=true"
    );
}

/// Turn-0 requests that carry a grammar, as tool-calling chat requests do
/// (the handler compiles the tool grammar into `params.grammar`). The
/// grammar admits every non-NUL byte, so it constrains nothing the model
/// emits here, but it moves seeding onto the sampler path and exercises the
/// cohort's per-lane grammar check.
fn grammar_turn0_requests(engine: &Engine, lanes: &[Shape]) -> Vec<(Vec<u32>, SamplingParams)> {
    let grammar = crate::serve::api::grammar::parser::parse("root ::= [^\\x00]*\n")
        .expect("permissive grammar");
    let token_bytes = engine.token_bytes_table();
    turn0_requests(lanes)
        .into_iter()
        .map(|(prompt, mut params)| {
            params.grammar = Some(grammar.clone());
            params.token_bytes = Some(token_bytes.clone());
            (prompt, params)
        })
        .collect()
}

#[test]
#[ignore = "official Gemma 4 GGUF: grammar-carrying (tool-calling) turn-0 requests form the cold cohort and equal serial"]
fn gemma_joint_cold_engine_cohort_with_grammar_matches_serial_turn0() {
    let rt = runtime();
    let off = spawn("0");
    let lanes = engine_turn0(&off);
    let requests = grammar_turn0_requests(&off, &lanes);
    assert!(requests.iter().all(|(_, params)| params.grammar.is_some()));
    let before = cohorts();
    let expected: Vec<_> = concurrent(&rt, &off, &requests)
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(cohorts(), before, "switch off never forms a cold cohort");
    shutdown(&rt, off);

    let on = spawn("1");
    let requests = grammar_turn0_requests(&on, &lanes);
    let before = cohorts();
    let actual: Vec<_> = concurrent(&rt, &on, &requests)
        .into_iter()
        .map(Result::unwrap)
        .collect();
    let after = cohorts();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, 4),
        "grammar turn 0: one cold stable cohort over 4 lanes"
    );
    assert_turns("grammar turn 0", &actual, &expected);
    shutdown(&rt, on);
    println!("GEMMA_JOINT_COLD_GRAMMAR_ENGINE_COMPLETE cohort_lanes=4 turn0_bitwise=true");
}

#[test]
#[ignore = "official Gemma 4 GGUF: a faulted cold cohort boundary forward retries each lane per slot in FIFO order"]
fn gemma_joint_cold_forward_fault_retries_each_lane_per_slot() {
    let rt = runtime();
    let (lanes, expected0, expected1) = reference_turns(&rt);
    let engine = spawn("1");
    cohort::RETRIED_LANES.lock().unwrap().clear();
    cohort::INJECT_COLD_FORWARD_FAULT.store(true, Ordering::SeqCst);
    let before = cohorts();
    let t0: Vec<_> = concurrent(&rt, &engine, &turn0_requests(&lanes))
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert!(
        !cohort::INJECT_COLD_FORWARD_FAULT.load(Ordering::SeqCst),
        "fault consumed"
    );
    assert_eq!(cohorts(), before, "the faulted cohort committed nothing");
    let retried = cohort::RETRIED_LANES.lock().unwrap().clone();
    assert_eq!(
        retried,
        *cohort::LAST_COHORT_FIFO.lock().unwrap(),
        "every lane retried per slot in FIFO order"
    );
    assert_turns("faulted turn 0", &t0, &expected0);
    let t1: Vec<_> = concurrent(&rt, &engine, &turn1(&lanes))
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_turns("turn 1 after fault", &t1, &expected1);
    shutdown(&rt, engine);
    println!("GEMMA_JOINT_COLD_FORWARD_FAULT_COMPLETE retried={retried:?} outputs_bitwise=true");
}

#[test]
#[ignore = "official Gemma 4 GGUF: a faulted cold cohort cue restores every lane from its boundary anchor"]
fn gemma_joint_cold_cue_fault_restores_from_boundary_anchors() {
    let rt = runtime();
    let (lanes, expected0, expected1) = reference_turns(&rt);
    let engine = spawn("1");
    cohort::CUE_RESTORED_LANES.lock().unwrap().clear();
    cohort::INJECT_COLD_CUE_FAULT.store(true, Ordering::SeqCst);
    let before = cohorts();
    let t0: Vec<_> = concurrent(&rt, &engine, &turn0_requests(&lanes))
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert!(
        !cohort::INJECT_COLD_CUE_FAULT.load(Ordering::SeqCst),
        "fault consumed"
    );
    let after = cohorts();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, 4),
        "restored lanes commit jointly"
    );
    let restored = cohort::CUE_RESTORED_LANES.lock().unwrap().clone();
    assert_eq!(
        restored,
        *cohort::LAST_COHORT_FIFO.lock().unwrap(),
        "every lane restored from its anchor in FIFO order"
    );
    assert_turns("cue-faulted turn 0", &t0, &expected0);
    let t1: Vec<_> = concurrent(&rt, &engine, &turn1(&lanes))
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_turns("turn 1 after cue fault", &t1, &expected1);
    shutdown(&rt, engine);
    println!("GEMMA_JOINT_COLD_CUE_FAULT_COMPLETE restored={restored:?} outputs_bitwise=true");
}

static CANCELLED_EVENTS: std::sync::Mutex<
    Option<mpsc::Receiver<crate::serve::api::sse::GenerationEvent>>,
> = std::sync::Mutex::new(None);

#[test]
#[ignore = "official Gemma 4 GGUF: a client closed during the cold cohort is recovered and released while the others decode"]
fn gemma_joint_cold_cancel_during_transaction() {
    let rt = runtime();
    let (lanes, expected0, _) = reference_turns(&rt);
    let engine = spawn("1");
    let (events_tx, events_rx) = mpsc::channel(64);
    *CANCELLED_EVENTS.lock().unwrap() = Some(events_rx);
    // Closes lane 3's stream after its reservation, before Phase B.
    *cohort::BEFORE_PHASE_B.lock().unwrap() = Some(Box::new(|| {
        drop(CANCELLED_EVENTS.lock().unwrap().take());
    }));
    let before = cohorts();
    let requests = turn0_requests(&lanes);
    let (unary, streamed) = rt.block_on(async {
        let unary = futures::future::join_all(
            requests[..3]
                .iter()
                .map(|(prompt, params)| engine.generate(prompt.clone(), params.clone())),
        );
        let streamed = engine.generate_stream(
            requests[3].0.clone(),
            requests[3].1.clone(),
            events_tx,
            None,
            Vec::new(),
        );
        tokio::join!(unary, streamed)
    });
    let _ = streamed;
    let after = cohorts();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, 4),
        "one cohort over 4 lanes"
    );
    for (agent, result) in unary.into_iter().enumerate() {
        assert_same(
            &format!("agent {agent}"),
            &result.unwrap(),
            &expected0[agent],
        );
    }
    // The cancelled lane's slot was recovered from its published boundary
    // anchor and released: the same turn-0 request now resumes warm from
    // the S-row anchor and answers like the reference.
    let again = rt
        .block_on(engine.generate(requests[3].0.clone(), requests[3].1.clone()))
        .unwrap();
    assert_eq!(
        again.cached_tokens, lanes[3].1,
        "agent 3 resumed its S-row anchor"
    );
    let expected = &expected0[3];
    assert_eq!(
        again.text, expected.text,
        "agent 3 after cancellation: text"
    );
    assert_eq!(
        again.logprobs, expected.logprobs,
        "agent 3 after cancellation: logprob bits"
    );
    assert_eq!(
        again.finish_reason, expected.finish_reason,
        "agent 3 after cancellation: finish"
    );
    assert_eq!(
        again.completion_tokens, expected.completion_tokens,
        "agent 3 after cancellation: completion tokens"
    );
    shutdown(&rt, engine);
    println!("GEMMA_JOINT_COLD_CANCEL_COMPLETE others_bitwise=true slot_recovered=true");
}

#[test]
fn gemma_cold_permissive_turn0_grammar_parses_and_runs() {
    let grammar = crate::serve::api::grammar::parser::parse("root ::= [^\\x00]*\n")
        .expect("permissive grammar");
    let root = grammar.rule_id("root").expect("root rule");
    assert!(crate::serve::api::grammar::GrammarRuntime::new(grammar, root).is_some());
}

/// #728: a group of cached stable-boundary requests whose boundary and cue
/// phases both reach 32 rows (the shape that used to form the tiled stable
/// batch) is admitted per slot, so the admission coalesce window cannot
/// change a request's numerics. Reference: window 0 and every request alone.
/// Arm: the 100 ms window with the requests concurrent. Turn 2 resumes each
/// turn-1 boundary anchor, so it also compares the anchors' KV.
#[test]
#[ignore = "official Gemma 4 GGUF: a cached stable-boundary group is bitwise independent of the admission window"]
fn gemma_stable_warm_group_is_independent_of_the_admission_window() {
    const COALESCE_ENV: &str = "HF2Q_ADMIT_COALESCE_US";
    // Turn 1 boundary phase and cue phase, both above the 32-row floor.
    const HISTORY: usize = 48;
    const CUE: usize = 40;
    let rt = runtime();
    let serve = |engine: &Engine, requests: &[(Vec<u32>, SamplingParams)], together: bool| {
        if together {
            concurrent(&rt, engine, requests)
                .into_iter()
                .map(Result::unwrap)
                .collect::<Vec<_>>()
        } else {
            requests
                .iter()
                .map(|(prompt, params)| {
                    rt.block_on(engine.generate(prompt.clone(), params.clone()))
                        .unwrap()
                })
                .collect()
        }
    };
    let run = |coalesce_us: &str, together: bool| {
        std::env::set_var(COALESCE_ENV, coalesce_us);
        let engine = spawn("1");
        let lanes = engine_turn0(&engine);
        // Turn 0 alone per lane: every slot then holds its boundary anchor.
        let turn0: Vec<_> = lanes
            .iter()
            .map(|(prompt, boundary)| (prompt.clone(), greedy(4, Some(*boundary))))
            .collect();
        let t0 = serve(&engine, &turn0, false);
        let grouped_before = GEMMA4_STABLE_GROUP_LANES.load(Ordering::Relaxed);
        let turn1: Vec<_> = lanes
            .iter()
            .enumerate()
            .map(|(agent, (prompt, boundary))| {
                let mut next = prompt[..*boundary].to_vec();
                next.extend((0..HISTORY + CUE).map(|i| 3_000 + ((agent * 97 + i) % 500) as u32));
                (next, greedy(16, Some(boundary + HISTORY)))
            })
            .collect();
        let t1 = serve(&engine, &turn1, together);
        let grouped = GEMMA4_STABLE_GROUP_LANES.load(Ordering::Relaxed) - grouped_before;
        let turn2: Vec<_> = turn1
            .iter()
            .enumerate()
            .map(|(agent, (prompt, params))| {
                let mut next = prompt[..params.stable_prompt_prefix_tokens.unwrap()].to_vec();
                next.extend((0..64).map(|i| 3_600 + ((agent * 61 + i) % 300) as u32));
                (next, greedy(16, None))
            })
            .collect();
        let t2 = serve(&engine, &turn2, together);
        shutdown(&rt, engine);
        std::env::remove_var(COALESCE_ENV);
        (lanes, grouped, [t0, t1, t2])
    };
    let (lanes, serial_grouped, expected) = run("0", false);
    assert_eq!(
        serial_grouped, 0,
        "serial turn-1 requests never share a window"
    );
    for (agent, (_, boundary)) in lanes.iter().enumerate() {
        assert_eq!(
            expected[1][agent].cached_tokens, *boundary,
            "reference turn 1 agent {agent} resumed its turn-0 boundary anchor"
        );
        assert_eq!(
            expected[2][agent].cached_tokens,
            boundary + HISTORY,
            "reference turn 2 agent {agent} resumed its turn-1 boundary anchor"
        );
    }
    let (_, grouped, actual) = run("100000", true);
    assert!(
        grouped >= 2,
        "the 100 ms window must group turn-1 lanes into one stable-boundary group (got {grouped})"
    );
    for (turn, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert_turns(&format!("window 100 ms turn {turn}"), actual, expected);
    }
    println!(
        "GEMMA_STABLE_WARM_WINDOW_INDEPENDENT_COMPLETE lanes={} grouped_lanes={grouped} boundary_rows={HISTORY} cue_rows={CUE} outputs_bitwise=true anchors_bitwise=true",
        lanes.len()
    );
}
