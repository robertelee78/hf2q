//! ADR-063 joint cue replay: real-artifact bitwise gates.
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF. Run
//! each test alone with `--exact --ignored --nocapture --test-threads=1`.
//! * `gemma_joint_cue_replay_matches_per_slot_bitwise`: the production joint
//!   replay (`replay_cues_jointly`: batched decode body per cue position,
//!   batched head on each lane's last token) equals each slot's inline
//!   per-slot cue replay (`forward_decode_slot_aware` per token): first
//!   token, final logits bits, every layer's K / TQ-HB V codes / V scales
//!   bytes over the valid rows, and cursors. Cases: four 7-token cues (three
//!   repeats), mixed cue lengths 7/3/12/31 (rows leave the batch at
//!   different steps), and 2- and 3-lane cohorts that must leave the other
//!   slots' bytes untouched. Boundaries 458/300/700/1100 include a ring that
//!   has already wrapped.
//! * `gemma_joint_cue_engine_turn_zero_matches_serial`: four concurrent cold
//!   stable-split requests (turn-0 shape: 458-token boundary + 7-token cue)
//!   through the real admission path form one joint cue cohort and answer
//!   byte-identically to the same requests sent one at a time. A second turn
//!   resumes every slot from its published stable-boundary checkpoint
//!   (warm lanes, joint again) and must also match, which proves the
//!   checkpoints are identical.
use super::gemma_warm_batch_tests::{
    assert_lane_equal, assert_same, encode_len, lane_image, load_options, runtime,
    slot_isolation_image, synthetic_prompt, LaneImage,
};
use super::*;
use crate::inference::models::gemma4::kv_cache::{
    restore_gemma_hybrid_slot_anchor, snapshot_gemma_hybrid_slot_anchor,
};
use std::sync::atomic::Ordering;

const BOUNDARIES: [usize; 4] = [458, 300, 700, 1100];
const MAX_CUE: usize = 31;

#[test]
#[ignore = "official Gemma 4 GGUF: joint cue replay equals per-slot cue replay bit for bit"]
fn gemma_joint_cue_replay_matches_per_slot_bitwise() {
    crate::debug::investigation_env::gemma_full_input::set_worker_active(true);
    assert!(
        gemma_cue_batch::routes_row_identical(&mlx_native::ggml_routing_policy_from_environment()),
        "run under the default GGML routing policy"
    );
    let mut loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("HF2Q_GEMMA4_GGUF is not a Gemma 4 artifact")
    };
    g.provision_multi_seq_kv_for_slot_aware(4).unwrap();
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    assert_eq!(guard.model.weights.kv_graft_len(), 0, "no graft bound");
    let supervisor = EngineSupervisor::new();
    let vocab = guard.model.weights.vocab_size;
    let max_decode = 24;
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|slot| synthetic_prompt(slot, BOUNDARIES[slot] + MAX_CUE, vocab))
        .collect();

    // Cold boundary prefill per slot (the unchanged per-slot route), then a
    // CPU copy of the stable-boundary checkpoint so every arm starts from
    // the bytes the inline path would hold right before its cue.
    let mut anchors = Vec::new();
    for slot in 0..4 {
        let boundary = BOUNDARIES[slot];
        guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware(
                &prompts[slot][..boundary],
                &[],
                max_decode,
                &mut guard.model.ctx,
                SlotId(slot as u32),
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
            )
            .unwrap();
        clear_gemma4_self_mounts(guard.model);
        anchors.push(
            snapshot_gemma_hybrid_slot_anchor(
                guard.hybrid.as_ref().unwrap(),
                SlotId(slot as u32),
                boundary,
            )
            .unwrap(),
        );
    }
    // Cold admission state during the cue: rings at the boundary, ledger 0.
    let restore = |guard: &mut Gemma4KvGuard<'_>, slot: usize| {
        restore_gemma_hybrid_slot_anchor(
            guard.hybrid.as_mut().unwrap(),
            SlotId(slot as u32),
            &anchors[slot],
            BOUNDARIES[slot],
        )
        .unwrap();
        install_gemma4_slot_cursor(guard, SlotId(slot as u32), 0).unwrap();
    };
    let prompt = |slot: usize, cue: usize| &prompts[slot][..BOUNDARIES[slot] + cue];

    // Per-slot reference: the inline stable cue call.
    let per_slot = |guard: &mut Gemma4KvGuard<'_>, slot: usize, cue: usize| {
        restore(guard, slot);
        clear_gemma4_self_mounts(guard.model);
        let first = gemma_cue_batch::replay_cue_per_slot(
            guard,
            SlotId(slot as u32),
            prompt(slot, cue),
            BOUNDARIES[slot],
            max_decode,
            &supervisor,
        )
        .unwrap();
        clear_gemma4_self_mounts(guard.model);
        commit_gemma4_slot_cursor(guard, SlotId(slot as u32), 0, BOUNDARIES[slot] + cue).unwrap();
        let logits = guard.model.weights.logits_view().unwrap().to_vec();
        assert!(logits.iter().all(|value| value.is_finite()));
        lane_image(guard, slot, BOUNDARIES[slot] + cue, first, &logits)
    };
    let joint = |guard: &mut Gemma4KvGuard<'_>, lanes: &[(usize, usize)]| -> Vec<LaneImage> {
        for &(slot, _) in lanes {
            restore(guard, slot);
        }
        let rows: Vec<gemma_cue_batch::CueRows<'_>> = lanes
            .iter()
            .map(|&(slot, cue)| gemma_cue_batch::CueRows {
                slot: SlotId(slot as u32),
                start: BOUNDARIES[slot],
                cue: &prompt(slot, cue)[BOUNDARIES[slot]..],
            })
            .collect();
        let outputs = gemma_cue_batch::replay_cues_jointly(guard, &rows, &supervisor).unwrap();
        let commits: Vec<_> = lanes
            .iter()
            .map(|&(slot, cue)| (SlotId(slot as u32), 0, BOUNDARIES[slot] + cue))
            .collect();
        commit_gemma4_slot_cursors(guard, &commits).unwrap();
        let guard: &Gemma4KvGuard<'_> = guard;
        lanes
            .iter()
            .zip(outputs)
            .map(|(&(slot, cue), output)| {
                lane_image(
                    guard,
                    slot,
                    BOUNDARIES[slot] + cue,
                    output.first_token,
                    &output.logits,
                )
            })
            .collect()
    };

    let cases: [&[(usize, usize)]; 6] = [
        &[(0, 7), (1, 7), (2, 7), (3, 7)],
        &[(0, 7), (1, 7), (2, 7), (3, 7)],
        &[(0, 7), (1, 7), (2, 7), (3, 7)],
        &[(0, 7), (1, 3), (2, 12), (3, 31)],
        &[(0, 7), (3, 7)],
        &[(1, 7), (2, 7), (3, 7)],
    ];
    let mut references: std::collections::BTreeMap<(usize, usize), LaneImage> =
        std::collections::BTreeMap::new();
    for lanes in cases.iter() {
        for &(slot, cue) in lanes.iter() {
            assert!(gemma_cue_batch::cue_replays_per_token(
                BOUNDARIES[slot] + cue,
                BOUNDARIES[slot]
            ));
            if !references.contains_key(&(slot, cue)) {
                let image = per_slot(&mut guard, slot, cue);
                references.insert((slot, cue), image);
            }
        }
    }
    let longest = BOUNDARIES.iter().max().unwrap() + MAX_CUE;
    for (case, lanes) in cases.iter().enumerate() {
        let outside: Vec<usize> = (0..4)
            .filter(|slot| lanes.iter().all(|&(lane, _)| lane != *slot))
            .collect();
        let before: Vec<_> = outside
            .iter()
            .map(|&slot| slot_isolation_image(&guard, slot, longest))
            .collect();
        for (image, &(slot, cue)) in joint(&mut guard, lanes).iter().zip(lanes.iter()) {
            assert_lane_equal(
                &format!(
                    "case {case} lanes={lanes:?} slot {slot} boundary {} cue {cue}",
                    BOUNDARIES[slot]
                ),
                image,
                &references[&(slot, cue)],
            );
        }
        for (&slot, before) in outside.iter().zip(&before) {
            assert!(
                slot_isolation_image(&guard, slot, longest) == *before,
                "case {case} lanes={lanes:?} touched non-cohort slot {slot}"
            );
        }
    }
    println!("GEMMA_JOINT_CUE_BITWISE_COMPLETE cases=4x3,mixed,2,3 logits_bits=true kv_bytes=true cursors=true isolation=true boundaries={BOUNDARIES:?}");
}

/// Greedy with top-4 logprobs; untagged so the stable split stays in force.
fn stable_turn(max_tokens: usize, boundary: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        logprobs: true,
        top_logprobs: 4,
        stable_prompt_prefix_tokens: Some(boundary),
        ..SamplingParams::default()
    }
}

#[test]
#[ignore = "official Gemma 4 GGUF: engine-level joint cue cohort equals serial stable turns"]
fn gemma_joint_cue_engine_turn_zero_matches_serial() {
    assert!(
        !matches!(
            std::env::var(gemma_cue_batch::JOINT_CUE_ENV).as_deref(),
            Ok("0") | Ok("off") | Ok("false")
        ),
        "run without HF2Q_GEMMA_JOINT_CUE=0"
    );
    let rt = runtime();
    let lanes = || gemma_cue_batch::JOINT_CUE_LANES.load(Ordering::Relaxed);
    let cohorts = || gemma_cue_batch::JOINT_CUE_COHORTS.load(Ordering::Relaxed);
    const TEXTS: [&str; 4] = [
        "You review a Rust parser that reads a u64 and returns a u16. ",
        "Explain how a sliding window attention cache overwrites its oldest rows. ",
        "Write a shell script that rotates log files older than seven days. ",
        "Summarize why the borrow checker rejects two mutable borrows. ",
    ];
    const BOUNDARY0: usize = 458;
    const TOOL_ROWS: usize = 64;
    const CUE: usize = 7;
    // Turn 0: boundary + cue (cold). Turn 1: turn 0's boundary + a tool
    // result, with the stable boundary after it, + cue (warm from the
    // turn-0 stable-boundary checkpoint).
    let conversation = |engine: &Engine| {
        (0..4)
            .map(|agent| {
                let stable = encode_len(engine, TEXTS[agent], BOUNDARY0);
                let cue = encode_len(engine, " Assistant, answer briefly.", CUE);
                let mut turn0 = stable.clone();
                turn0.extend_from_slice(&cue);
                let mut turn1 = stable;
                turn1.extend(encode_len(
                    engine,
                    " Tool result: cargo test passed, 4 tests, 0 failed.",
                    TOOL_ROWS,
                ));
                turn1.extend_from_slice(&cue);
                [turn0, turn1]
            })
            .collect::<Vec<_>>()
    };
    let boundaries = [BOUNDARY0, BOUNDARY0 + TOOL_ROWS];
    let spawn = || {
        let loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
        Engine::spawn_with_mode(loaded, 8, None, EngineMode::SlotAware { max_slots: 4 })
            .expect("spawn SlotAware Gemma engine")
    };

    // Reference: every request alone, so every cue replays per slot.
    let reference = spawn();
    let turns = conversation(&reference);
    let before = lanes();
    let mut expected = Vec::new();
    for turn in 0..2 {
        let mut results = Vec::new();
        for agent in 0..4 {
            let prompt = turns[agent][turn].clone();
            let result = rt
                .block_on(reference.generate(prompt, stable_turn(16, boundaries[turn])))
                .unwrap();
            let cached = if turn == 0 { 0 } else { BOUNDARY0 };
            assert_eq!(
                result.cached_tokens, cached,
                "reference agent {agent} turn {turn}"
            );
            results.push(result);
        }
        expected.push(results);
    }
    rt.block_on(reference.shutdown()).unwrap();
    assert_eq!(lanes(), before, "serial requests never form a cue cohort");

    // Joint: each turn's four requests concurrently.
    let joint = spawn();
    let turns = conversation(&joint);
    for turn in 0..2 {
        let (lanes_before, cohorts_before) = (lanes(), cohorts());
        let [a, b, c, d]: [Vec<u32>; 4] = std::array::from_fn(|agent| turns[agent][turn].clone());
        let params = || stable_turn(16, boundaries[turn]);
        let (a, b, c, d) = rt.block_on(async {
            tokio::join!(
                joint.generate(a, params()),
                joint.generate(b, params()),
                joint.generate(c, params()),
                joint.generate(d, params()),
            )
        });
        let results = [a.unwrap(), b.unwrap(), c.unwrap(), d.unwrap()];
        assert_eq!(
            (lanes() - lanes_before, cohorts() - cohorts_before),
            (4, 1),
            "turn {turn}: one joint cue cohort over 4 lanes"
        );
        for (agent, result) in results.iter().enumerate() {
            assert_same(
                &format!("turn {turn} agent {agent}"),
                result,
                &expected[turn][agent],
            );
        }
    }
    rt.block_on(joint.shutdown()).unwrap();
    println!("GEMMA_JOINT_CUE_ENGINE_COMPLETE turn0_lanes=4 turn1_lanes=4 checkpoints_equal=true outputs_bitwise=true");
}
