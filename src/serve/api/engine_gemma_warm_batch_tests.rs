//! ADR-063 Stage 1 joint warm prefill: real-artifact bitwise gates.
//!
//! Ignored; `HF2Q_GEMMA4_GGUF` names the official Gemma 4 26B-A4B GGUF. Run
//! each test alone with `--exact --ignored --nocapture --test-threads=1`.
//! * `gemma_joint_warm_prefill_matches_per_slot_bitwise`: one joint
//!   `forward_prefill_batched_multi_seq_live` (`PerSlotExact`) over 4, 2 and
//!   3 restored slots equals each slot's own per-slot warm prefill: first
//!   token, final logits bits, every layer's K / TQ-HB V codes / V scales
//!   bytes over the valid (ring) rows, and the slot cursors.
//! * `gemma_joint_warm_engine_cohort_matches_serial_tool_turns`: four
//!   concurrent tagged tool turns through the real admission path form one
//!   cohort and answer byte-identically to the same turns sent one at a time;
//!   a lane whose suffix is 32 rows falls back to per-slot prefill.
use super::*;
use crate::inference::models::gemma4::kv_cache::{
    restore_gemma_hybrid_slot_anchor, snapshot_gemma_hybrid_slot_anchor,
};
use std::sync::atomic::Ordering;

const HISTORIES: [usize; 4] = [465, 714, 943, 1254];
const SUFFIXES: [usize; 4] = [56, 200, 311, 385];

fn official_gguf() -> PathBuf {
    let path = PathBuf::from(
        std::env::var("HF2Q_GEMMA4_GGUF")
            .expect("HF2Q_GEMMA4_GGUF must name the official Gemma 4 26B-A4B GGUF"),
    );
    assert!(path.exists(), "GGUF missing: {}", path.display());
    path
}

pub(super) fn load_options() -> LoadOptions {
    LoadOptions {
        model_path: official_gguf(),
        tokenizer_path: None,
        config_path: None,
        dwq_overlay_path: None,
        glp_path: None,
        glp_alpha: None,
        kv_graft_path: None,
        kv_persist_dir: None,
        kv_persist_budget_bytes: 0,
    }
}

/// Synthetic ids above the special/control range, distinct per slot.
pub(super) fn synthetic_prompt(slot: usize, len: usize, vocab: usize) -> Vec<u32> {
    let span = vocab - 2_048;
    (0..len)
        .map(|i| (1_024 + (i * 7_919 + slot * 104_729 + 17) % span) as u32)
        .collect()
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct LaneImage {
    first_token: u32,
    logits: Vec<u32>,
    /// One SHA-256 per (layer, K | V codes | V scales) over the slot's valid
    /// rows: `[0, len)` for global layers, the `min(len, cap)` live ring rows
    /// for sliding layers.
    kv: Vec<[u8; 32]>,
    cursors: Vec<u32>,
}

/// One SHA-256 per (layer, K | V codes | V scales) of `slot`: rows
/// `[0, global_rows)` of global layers and, for sliding layers, either the
/// live ring rows of a `valid_tokens`-long sequence or (`whole_ring`) every
/// ring row.
pub(super) fn slot_kv_hashes(
    guard: &Gemma4KvGuard<'_>,
    slot: usize,
    global_rows: usize,
    valid_tokens: usize,
    whole_ring: bool,
) -> Vec<[u8; 32]> {
    use sha2::{Digest, Sha256};
    let hybrid = guard.hybrid.as_ref().expect("hybrid scaffold");
    let mut kv = Vec::new();
    for layer in hybrid.iter() {
        let cap = layer.capacity;
        let rows: Vec<usize> = if layer.is_sliding && whole_ring {
            (0..cap).collect()
        } else if layer.is_sliding {
            (valid_tokens.saturating_sub(cap)..valid_tokens)
                .map(|position| position % cap)
                .collect()
        } else {
            assert!(global_rows <= cap);
            (0..global_rows).collect()
        };
        for buf in [&layer.k, &layer.v_packed, &layer.v_norms] {
            if buf.byte_len() == 4 {
                // Shared dummy norms (F16 V): no per-slot payload.
                continue;
            }
            let bytes = buf.as_slice::<u8>().unwrap();
            let per_slot = bytes.len() / layer.n_seqs as usize;
            let shape = buf.shape();
            assert!(shape.len() >= 3, "slot-major KV shape {shape:?}");
            let heads = shape[1];
            let row_bytes = per_slot / (heads * cap);
            let mut hash = Sha256::new();
            for head in 0..heads {
                let base = slot * per_slot + head * cap * row_bytes;
                for &row in &rows {
                    hash.update(&bytes[base + row * row_bytes..base + (row + 1) * row_bytes]);
                }
            }
            kv.push(<[u8; 32]>::from(hash.finalize()));
        }
    }
    kv
}

pub(super) fn slot_cursors(guard: &Gemma4KvGuard<'_>, slot: usize) -> Vec<u32> {
    guard
        .kv
        .iter()
        .map(|buffer| buffer.seq_lens[slot])
        .chain(
            guard
                .hybrid
                .as_ref()
                .expect("hybrid scaffold")
                .iter()
                .map(|buffer| buffer.seq_lens[slot]),
        )
        .collect()
}

pub(super) fn lane_image(
    guard: &Gemma4KvGuard<'_>,
    slot: usize,
    valid_tokens: usize,
    first_token: u32,
    logits: &[f32],
) -> LaneImage {
    LaneImage {
        first_token,
        logits: logits.iter().map(|value| value.to_bits()).collect(),
        kv: slot_kv_hashes(guard, slot, valid_tokens, valid_tokens, false),
        cursors: slot_cursors(guard, slot),
    }
}

/// Everything a misaddressed cohort write could touch in a non-cohort slot:
/// global rows up to the longest prompt, every ring row, and the cursors.
fn slot_isolation_image(
    guard: &Gemma4KvGuard<'_>,
    slot: usize,
    rows: usize,
) -> (Vec<[u8; 32]>, Vec<u32>) {
    (
        slot_kv_hashes(guard, slot, rows, rows, true),
        slot_cursors(guard, slot),
    )
}

pub(super) fn assert_lane_equal(label: &str, joint: &LaneImage, reference: &LaneImage) {
    assert_eq!(
        joint.first_token, reference.first_token,
        "{label}: first token"
    );
    assert_eq!(joint.logits.len(), reference.logits.len(), "{label}: vocab");
    let logit = joint
        .logits
        .iter()
        .zip(&reference.logits)
        .position(|(a, b)| a != b);
    assert!(
        logit.is_none(),
        "{label}: first logit bit mismatch at {logit:?}"
    );
    let kv = joint.kv.iter().zip(&reference.kv).position(|(a, b)| a != b);
    assert_eq!(joint.kv.len(), reference.kv.len(), "{label}: KV buffers");
    assert!(
        kv.is_none(),
        "{label}: KV bytes differ at (layer, buffer) index {kv:?} (3 buffers per TQ-HB layer)"
    );
    assert_eq!(joint.cursors, reference.cursors, "{label}: cursors");
}

#[test]
#[ignore = "official Gemma 4 GGUF: joint warm prefill lanes equal per-slot warm prefill bit for bit"]
fn gemma_joint_warm_prefill_matches_per_slot_bitwise() {
    // This test thread plays the SlotAware worker that owns the scaffold and
    // the CPU checkpoint quota.
    crate::debug::investigation_env::gemma_full_input::set_worker_active(true);
    let mut loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("HF2Q_GEMMA4_GGUF is not a Gemma 4 artifact")
    };
    g.provision_multi_seq_kv_for_slot_aware(4).unwrap();
    let mut guard = Gemma4KvGuard::take(g).unwrap();
    assert_eq!(guard.model.weights.kv_graft_len(), 0, "no graft bound");
    let vocab = guard.model.weights.vocab_size;
    let max_decode = 16;
    let prompts: Vec<Vec<u32>> = (0..4)
        .map(|slot| synthetic_prompt(slot, HISTORIES[slot] + SUFFIXES[slot], vocab))
        .collect();

    // Shared histories: one cold prefill per slot, checkpointed on the CPU so
    // the reference and every joint arm resume from identical bytes.
    let mut anchors = Vec::new();
    for slot in 0..4 {
        let history = HISTORIES[slot];
        guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware(
                &prompts[slot][..history],
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
        install_gemma4_slot_cursor(&mut guard, SlotId(slot as u32), history).unwrap();
        anchors.push(
            snapshot_gemma_hybrid_slot_anchor(
                guard.hybrid.as_ref().unwrap(),
                SlotId(slot as u32),
                history,
            )
            .unwrap(),
        );
    }
    let restore = |guard: &mut Gemma4KvGuard<'_>, slot: usize, history: usize| {
        restore_gemma_hybrid_slot_anchor(
            guard.hybrid.as_mut().unwrap(),
            SlotId(slot as u32),
            &anchors[slot],
            history,
        )
        .unwrap();
        install_gemma4_slot_cursor(guard, SlotId(slot as u32), history).unwrap();
    };

    // Per-slot reference: today's warm prefill route for each slot alone.
    let per_slot = |guard: &mut Gemma4KvGuard<'_>, slot: usize, prompt: &[u32], history: usize| {
        restore(guard, slot, history);
        let first = guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware_resume(
                prompt,
                &[],
                max_decode,
                &mut guard.model.ctx,
                SlotId(slot as u32),
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
                history,
            )
            .unwrap();
        clear_gemma4_self_mounts(guard.model);
        commit_gemma4_slot_cursor(guard, SlotId(slot as u32), history, prompt.len()).unwrap();
        let logits = guard.model.weights.logits_view().unwrap().to_vec();
        assert!(logits.iter().all(|value| value.is_finite()));
        lane_image(guard, slot, prompt.len(), first, &logits)
    };
    let reference: Vec<LaneImage> = (0..4)
        .map(|slot| per_slot(&mut guard, slot, &prompts[slot], HISTORIES[slot]))
        .collect();

    // Joint arm: restore the same histories, one shared forward. `lanes` are
    // `(slot, prompt, history, max_tokens)`; the decode reserve is the
    // production per-lane rule.
    let joint = |guard: &mut Gemma4KvGuard<'_>,
                 lanes: &[(usize, &[u32], usize, usize)],
                 reserve: usize|
     -> Result<Vec<LaneImage>> {
        for &(slot, _, history, _) in lanes {
            restore(guard, slot, history);
        }
        let seqs: Vec<(Vec<u32>, SlotId, usize)> = lanes
            .iter()
            .map(|&(slot, prompt, history, _)| {
                (prompt[history..].to_vec(), SlotId(slot as u32), history)
            })
            .collect();
        clear_gemma4_self_mounts(guard.model);
        let output = guard.model.weights.forward_prefill_batched_multi_seq_live(
            &seqs,
            guard.hybrid.as_deref().unwrap(),
            reserve,
            GemmaLiveAttention::PerSlotExact,
            &mut guard.model.ctx,
        );
        clear_gemma4_self_mounts(guard.model);
        let output = output?;
        let commits: Vec<_> = lanes
            .iter()
            .map(|&(slot, prompt, history, _)| (SlotId(slot as u32), history, prompt.len()))
            .collect();
        commit_gemma4_slot_cursors(guard, &commits)?;
        let guard: &Gemma4KvGuard<'_> = guard;
        let mut images = Vec::with_capacity(lanes.len());
        for ((&(slot, prompt, _, _), first), logits) in
            lanes.iter().zip(output.first_tokens).zip(output.logits)
        {
            images.push(lane_image(guard, slot, prompt.len(), first, &logits));
        }
        Ok(images)
    };
    let reserve_for = |lanes: &[(usize, &[u32], usize, usize)]| {
        gemma4_batch_decode_reserve(
            lanes
                .iter()
                .map(|&(_, prompt, _, max_tokens)| (prompt.len(), max_tokens)),
        )
    };
    let lane = |slot: usize| (slot, prompts[slot].as_slice(), HISTORIES[slot], max_decode);
    // Isolation: every row a misaddressed write could reach in a slot outside
    // the cohort (global rows up to the longest prompt, every ring row).
    let longest = prompts.iter().map(Vec::len).max().unwrap();
    // Five repeats of the 4-lane cohort (determinism), then subsets that
    // move each lane's rows within the shared stream.
    let cases: [&[usize]; 7] = [
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        &[0, 1, 2, 3],
        &[0, 3],
        &[1, 2, 3],
    ];
    for (case, slots) in cases.iter().enumerate() {
        let lanes: Vec<_> = slots.iter().map(|&slot| lane(slot)).collect();
        let rows: usize = slots.iter().map(|&slot| SUFFIXES[slot]).sum();
        assert!(rows <= gemma_warm_batch::ROW_CAP);
        assert!(slots
            .iter()
            .all(|&slot| gemma_warm_batch::lane_eligible(prompts[slot].len(), HISTORIES[slot])));
        let outside: Vec<usize> = (0..4).filter(|slot| !slots.contains(slot)).collect();
        let before: Vec<_> = outside
            .iter()
            .map(|&slot| slot_isolation_image(&guard, slot, longest))
            .collect();
        let images = joint(&mut guard, &lanes, reserve_for(&lanes)).unwrap();
        for (image, &slot) in images.iter().zip(slots.iter()) {
            assert_lane_equal(
                &format!(
                    "case {case} lanes={slots:?} slot {slot} history {} suffix {}",
                    HISTORIES[slot], SUFFIXES[slot]
                ),
                image,
                &reference[slot],
            );
        }
        for (&slot, before) in outside.iter().zip(&before) {
            assert!(
                slot_isolation_image(&guard, slot, longest) == *before,
                "case {case} lanes={slots:?} touched non-cohort slot {slot}"
            );
        }
    }

    // Mixed max_tokens near the slot capacity: lane 0 may decode up to the
    // last full tile of its own slot; the others decode 16 tokens. Every lane
    // fits alone. The old reserve (largest max_tokens with the longest lane)
    // is refused before submission; the production per-lane reserve is
    // accepted and stays bitwise equal.
    let capacity = guard
        .hybrid
        .as_ref()
        .unwrap()
        .iter()
        .find(|layer| !layer.is_sliding)
        .unwrap()
        .capacity;
    let near_limit = (capacity / 32) * 32 - prompts[0].len();
    let mut mixed: Vec<_> = (0..4).map(lane).collect();
    mixed[0].3 = near_limit;
    let per_lane_need = mixed
        .iter()
        .map(|&(_, prompt, _, max_tokens)| {
            crate::serve::forward_prefill::gemma_flash_attn_vec_capacity(prompt.len() + max_tokens)
        })
        .max()
        .unwrap();
    assert!(per_lane_need <= capacity, "every lane fits alone");
    let naive = mixed.iter().map(|lane| lane.3).max().unwrap();
    let refused = joint(&mut guard, &mixed, naive)
        .err()
        .expect("largest max_tokens with the longest lane exceeds the slot");
    assert!(
        format!("{refused:#}").contains("capacity"),
        "refused by the capacity check: {refused:#}"
    );
    let reserve = reserve_for(&mixed);
    assert_eq!(longest + reserve, prompts[0].len() + near_limit);
    // The decode reserve only sizes capacity checks: a per-slot prefill with
    // lane 0's own max_tokens equals the 16-token reference.
    let per_slot_near_limit = |guard: &mut Gemma4KvGuard<'_>| {
        restore(guard, 0, HISTORIES[0]);
        let first = guard
            .model
            .weights
            .forward_prefill_with_soft_tokens_slot_aware_resume(
                &prompts[0],
                &[],
                near_limit,
                &mut guard.model.ctx,
                SlotId(0),
                &mut guard.kv,
                guard.hybrid.as_mut(),
                guard.dense.as_mut(),
                guard.mlx.as_mut(),
                HISTORIES[0],
            )
            .unwrap();
        clear_gemma4_self_mounts(guard.model);
        commit_gemma4_slot_cursor(guard, SlotId(0), HISTORIES[0], prompts[0].len()).unwrap();
        let logits = guard.model.weights.logits_view().unwrap().to_vec();
        lane_image(guard, 0, prompts[0].len(), first, &logits)
    };
    assert_lane_equal(
        "per-slot lane 0 at its own near-limit max_tokens",
        &per_slot_near_limit(&mut guard),
        &reference[0],
    );
    for (image, slot) in joint(&mut guard, &mixed, reserve).unwrap().iter().zip(0..4) {
        assert_lane_equal(
            &format!("mixed max_tokens slot {slot}"),
            image,
            &reference[slot],
        );
    }

    // Below the threshold: the planner refuses a 32-row lane, so admission
    // prefills it per slot. Running it jointly anyway is a diagnostic of why
    // (MoE mv_id per slot vs mm_id jointly), never a gate.
    let short: Vec<u32> = prompts[0][..HISTORIES[0] + 32].to_vec();
    assert!(!gemma_warm_batch::lane_eligible(short.len(), HISTORIES[0]));
    let short_reference = per_slot(&mut guard, 0, &short, HISTORIES[0]);
    let mut lanes = vec![(0, short.as_slice(), HISTORIES[0], max_decode)];
    lanes.extend([1, 2, 3].map(lane));
    let images = joint(&mut guard, &lanes, reserve_for(&lanes)).unwrap();
    for (image, slot) in images.iter().skip(1).zip([1, 2, 3]) {
        assert_lane_equal(
            &format!("slot {slot} beside a 32-row lane"),
            image,
            &reference[slot],
        );
    }
    println!(
        "GEMMA_JOINT_WARM_THRESHOLD_DIAGNOSTIC rows=32 accepted_as_gate=false joint_equals_per_slot={}",
        images[0] == short_reference
    );
    println!("GEMMA_JOINT_WARM_BITWISE_COMPLETE lanes=4x5,2,3 mixed_max_tokens=true isolation=true logits_bits=true kv_bytes=true cursors=true histories={HISTORIES:?} suffixes={SUFFIXES:?}");
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
}

fn encode_len(engine: &Engine, text: &str, len: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    while ids.len() < len {
        ids.extend_from_slice(engine.tokenizer().encode(text, false).unwrap().get_ids());
    }
    ids.truncate(len);
    ids
}

/// Tool-turn shape: the handler tags the full input; greedy with top-4
/// logprobs so every emitted token and its alternatives compare bit for bit.
fn tool_turn(max_tokens: usize, input: &[u32]) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        logprobs: true,
        top_logprobs: 4,
        gemma_full_input_checkpoint: gemma_full_input::request_tag(true, true, input, input),
        ..SamplingParams::default()
    }
}

#[test]
#[ignore = "official Gemma 4 GGUF: engine-level joint warm cohort equals serial tool turns"]
fn gemma_joint_warm_engine_cohort_matches_serial_tool_turns() {
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
    let lanes = || gemma_warm_batch::JOINT_WARM_LANES.load(Ordering::Relaxed);
    let transactions = || gemma_warm_batch::JOINT_WARM_TRANSACTIONS.load(Ordering::Relaxed);
    const TEXTS: [&str; 4] = [
        "You review a Rust parser that reads a u64 and returns a u16. ",
        "Explain how a sliding window attention cache overwrites its oldest rows. ",
        "Write a shell script that rotates log files older than seven days. ",
        "Summarize why the borrow checker rejects two mutable borrows. ",
    ];
    // Turn 2: the last lane's 32-row suffix is below the joint threshold.
    const TURN2_SUFFIXES: [usize; 4] = [64, 128, 96, 32];
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
    let concurrent = |engine: &Engine, prompts: [Vec<u32>; 4]| {
        let [a, b, c, d] = prompts;
        let (a, b, c, d) = rt.block_on(async {
            tokio::join!(
                engine.generate(a.clone(), tool_turn(16, &a)),
                engine.generate(b.clone(), tool_turn(16, &b)),
                engine.generate(c.clone(), tool_turn(16, &c)),
                engine.generate(d.clone(), tool_turn(16, &d)),
            )
        });
        [a.unwrap(), b.unwrap(), c.unwrap(), d.unwrap()]
    };

    // Reference: every request alone, so every warm turn prefills per slot.
    let reference = spawn();
    let turns = conversation(&reference);
    let before = lanes();
    let mut expected = Vec::new();
    for turn in 0..3 {
        let mut results = Vec::new();
        for agent in 0..4 {
            let prompt = turns[agent][turn].clone();
            let max_tokens = if turn == 0 { 1 } else { 16 };
            let result = rt
                .block_on(reference.generate(prompt.clone(), tool_turn(max_tokens, &prompt)))
                .unwrap();
            let cached = if turn == 0 {
                0
            } else {
                turns[agent][turn - 1].len()
            };
            assert_eq!(
                result.cached_tokens, cached,
                "reference agent {agent} turn {turn}"
            );
            results.push(result);
        }
        expected.push(results);
    }
    rt.block_on(reference.shutdown()).unwrap();
    assert_eq!(lanes(), before, "serial requests never form a cohort");

    // Joint: turn 0 serially (cold), then each tool turn concurrently.
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
    for (turn, joint_lanes) in [(1usize, 4usize), (2, 3)] {
        let (lanes_before, transactions_before) = (lanes(), transactions());
        let results = concurrent(
            &joint,
            std::array::from_fn(|agent| turns[agent][turn].clone()),
        );
        assert_eq!(
            (lanes() - lanes_before, transactions() - transactions_before),
            (joint_lanes, 1),
            "turn {turn}: one joint transaction over {joint_lanes} lanes"
        );
        for (agent, result) in results.iter().enumerate() {
            assert_eq!(
                result.cached_tokens,
                turns[agent][turn - 1].len(),
                "turn {turn} agent {agent} resumed its checkpoint"
            );
            assert_same(
                &format!("turn {turn} agent {agent}"),
                result,
                &expected[turn][agent],
            );
        }
    }
    rt.block_on(joint.shutdown()).unwrap();
    println!("GEMMA_JOINT_WARM_ENGINE_COMPLETE turn1_lanes=4 turn2_lanes=3 below_threshold_fallback=true outputs_bitwise=true");
}

fn assert_same(label: &str, actual: &GenerationResult, expected: &GenerationResult) {
    assert_eq!(actual.text, expected.text, "{label}: text");
    assert_eq!(
        actual.reasoning_text, expected.reasoning_text,
        "{label}: reasoning"
    );
    assert_eq!(
        actual.completion_tokens, expected.completion_tokens,
        "{label}: completion tokens"
    );
    assert_eq!(
        actual.finish_reason, expected.finish_reason,
        "{label}: finish"
    );
    assert_eq!(actual.logprobs, expected.logprobs, "{label}: logprob bits");
    assert_eq!(
        actual.cached_tokens, expected.cached_tokens,
        "{label}: cached tokens"
    );
}
