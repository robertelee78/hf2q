//! ADR-063 S3 G-C3: fault injection for the joint warm prefill per-slot retry.
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF. Run
//! alone with `--exact --ignored --nocapture --test-threads=1`.
//!
//! `gemma_joint_warm_forward_fault_retries_each_lane_per_slot`: four
//! concurrent tagged tool turns form one joint cohort. The test-only hook
//! (`gemma_warm_batch::INJECT_FORWARD_FAULT`) lets the real joint forward
//! write every lane's KV rows, then reports a non-fatal error. Settlement must
//! then:
//! * restore every lane from its own request-start rollback anchor; the joint
//!   forward has overwritten sliding-ring rows that the retry's first suffix
//!   rows attend to, so a missed or partial restore changes the outputs;
//! * re-prefill each lane per slot in FIFO admission order, which this test
//!   makes differ from slot order;
//! * answer every request exactly as a fault-free serial run does: text,
//!   reasoning, tokens, finish reason, logprob bits and cached tokens.
//!
//! A later fault-free tool turn then forms a joint cohort again and still
//! matches the serial reference.
use super::gemma_warm_batch_tests::{
    assert_same, encode_len, load_options, runtime, tool_turn, HISTORIES, SUFFIXES,
};
use super::*;
use std::sync::atomic::Ordering;

/// Admission order of the faulted turn: neither slot order nor its reverse.
const FAULT_ORDER: [usize; 4] = [2, 0, 3, 1];
/// Every turn-2 suffix exceeds the 32-row joint threshold, so turn 2 forms a
/// four-lane cohort after the retried turn.
const TURN2_SUFFIXES: [usize; 4] = [64, 128, 96, 48];

#[test]
#[ignore = "official Gemma 4 GGUF: a faulted joint warm forward restores every lane and retries it per slot in FIFO order"]
fn gemma_joint_warm_forward_fault_retries_each_lane_per_slot() {
    assert!(
        gemma_full_input::enabled_for_engine(true, EngineMode::SlotAware { max_slots: 4 }),
        "run without HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=0"
    );
    assert!(
        !matches!(
            std::env::var(gemma_warm_batch::JOINT_WARM_PREFILL_ENV).as_deref(),
            Ok("0") | Ok("off") | Ok("false")
        ),
        "run without HF2Q_GEMMA_JOINT_WARM_PREFILL=0"
    );
    let rt = runtime();
    let transactions = || gemma_warm_batch::JOINT_WARM_TRANSACTIONS.load(Ordering::Relaxed);
    let lanes = || gemma_warm_batch::JOINT_WARM_LANES.load(Ordering::Relaxed);
    const TEXTS: [&str; 4] = [
        "You review a Rust parser that reads a u64 and returns a u16. ",
        "Explain how a sliding window attention cache overwrites its oldest rows. ",
        "Write a shell script that rotates log files older than seven days. ",
        "Summarize why the borrow checker rejects two mutable borrows. ",
    ];
    let conversation = |engine: &Engine| {
        (0..4)
            .map(|agent| {
                let turn0 = encode_len(engine, TEXTS[agent], HISTORIES[agent]);
                let mut turn1 = turn0.clone();
                turn1.extend(encode_len(
                    engine,
                    " Tool result: cargo test passed, 4 tests, 0 failed.",
                    SUFFIXES[agent],
                ));
                let mut turn2 = turn1.clone();
                turn2.extend(encode_len(
                    engine,
                    " Tool result: the file was written successfully.",
                    TURN2_SUFFIXES[agent],
                ));
                [turn0, turn1, turn2]
            })
            .collect::<Vec<_>>()
    };
    let spawn = || {
        let loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
        Engine::spawn_with_mode(loaded, 8, None, EngineMode::SlotAware { max_slots: 4 })
            .expect("spawn SlotAware Gemma engine")
    };
    // Sends the four prompts in `order`; results come back in agent order.
    let concurrent = |engine: &Engine, turns: &[[Vec<u32>; 3]], turn: usize, order: [usize; 4]| {
        let prompt = |position: usize| turns[order[position]][turn].clone();
        let [a, b, c, d]: [Vec<u32>; 4] = std::array::from_fn(prompt);
        let sent = rt.block_on(async {
            tokio::join!(
                engine.generate(a.clone(), tool_turn(16, &a)),
                engine.generate(b.clone(), tool_turn(16, &b)),
                engine.generate(c.clone(), tool_turn(16, &c)),
                engine.generate(d.clone(), tool_turn(16, &d)),
            )
        });
        let mut by_agent: [Option<GenerationResult>; 4] = Default::default();
        for (position, result) in [sent.0, sent.1, sent.2, sent.3].into_iter().enumerate() {
            by_agent[order[position]] = Some(result.expect("request succeeds"));
        }
        by_agent.map(Option::unwrap)
    };

    // Reference: every request alone, so every warm turn prefills per slot
    // and no fault can be injected.
    let reference = spawn();
    let turns = conversation(&reference);
    let mut expected = Vec::new();
    for turn in 0..3 {
        let results = (0..4)
            .map(|agent| {
                let prompt = turns[agent][turn].clone();
                let max_tokens = if turn == 0 { 1 } else { 16 };
                rt.block_on(reference.generate(prompt.clone(), tool_turn(max_tokens, &prompt)))
                    .unwrap()
            })
            .collect::<Vec<_>>();
        expected.push(results);
    }
    rt.block_on(reference.shutdown()).unwrap();

    let joint = spawn();
    let turns = conversation(&joint);
    for agent in 0..4 {
        let prompt = turns[agent][0].clone();
        let result = rt
            .block_on(joint.generate(prompt.clone(), tool_turn(1, &prompt)))
            .unwrap();
        assert_same(
            &format!("turn 0 agent {agent}"),
            &result,
            &expected[0][agent],
        );
    }

    // Turn 1: the joint forward runs, then reports the injected fault.
    gemma_warm_batch::FAULTED_LANES.lock().unwrap().clear();
    gemma_warm_batch::RETRIED_LANES.lock().unwrap().clear();
    gemma_warm_batch::INJECT_FORWARD_FAULT.store(true, Ordering::SeqCst);
    let (transactions_before, lanes_before) = (transactions(), lanes());
    let results = concurrent(&joint, &turns[..], 1, FAULT_ORDER);
    assert!(
        !gemma_warm_batch::INJECT_FORWARD_FAULT.load(Ordering::SeqCst),
        "the four turns formed a joint transaction and consumed the fault"
    );
    let faulted = gemma_warm_batch::FAULTED_LANES.lock().unwrap().clone();
    let retried = gemma_warm_batch::RETRIED_LANES.lock().unwrap().clone();
    let fifo_lengths = FAULT_ORDER.map(|agent| turns[agent][1].len());
    assert_eq!(
        faulted.iter().map(|&(_, len)| len).collect::<Vec<_>>(),
        fifo_lengths,
        "the faulted cohort held all four lanes in FIFO admission order"
    );
    let fifo_slots = faulted.iter().map(|&(slot, _)| slot).collect::<Vec<_>>();
    let mut sorted_slots = fifo_slots.clone();
    sorted_slots.sort_unstable();
    assert_ne!(
        fifo_slots, sorted_slots,
        "FIFO order differs from slot order"
    );
    sorted_slots.dedup();
    assert_eq!(sorted_slots.len(), 4, "four distinct slots");
    assert_eq!(
        retried, faulted,
        "every lane was restored and retried per slot, in the faulted FIFO order"
    );
    assert_eq!(
        (transactions(), lanes()),
        (transactions_before, lanes_before),
        "the retries ran per slot; no joint transaction succeeded"
    );
    for (agent, result) in results.iter().enumerate() {
        assert_eq!(
            result.cached_tokens,
            turns[agent][0].len(),
            "turn 1 agent {agent} retried from its restored checkpoint, not cold"
        );
        assert_same(
            &format!("faulted turn 1 agent {agent}"),
            result,
            &expected[1][agent],
        );
    }

    // Turn 2, fault-free: the retried lanes published correct checkpoints and
    // form a joint cohort again.
    let (transactions_before, lanes_before) = (transactions(), lanes());
    let results = concurrent(&joint, &turns[..], 2, [0, 1, 2, 3]);
    assert_eq!(
        (transactions() - transactions_before, lanes() - lanes_before),
        (1, 4),
        "turn 2: one joint transaction over four lanes"
    );
    for (agent, result) in results.iter().enumerate() {
        assert_eq!(
            result.cached_tokens,
            turns[agent][1].len(),
            "turn 2 agent {agent}"
        );
        assert_same(
            &format!("turn 2 agent {agent}"),
            result,
            &expected[2][agent],
        );
    }
    rt.block_on(joint.shutdown()).unwrap();
    println!(
        "GEMMA_JOINT_WARM_FAULT_RETRY_COMPLETE fifo={FAULT_ORDER:?} faulted_lanes=4 restored=4 retried_per_slot_fifo=true outputs_bitwise=true rejoined_lanes=4"
    );
}
