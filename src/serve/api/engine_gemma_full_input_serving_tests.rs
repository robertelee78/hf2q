//! ADR-063 Gemma full-input checkpoint default-promotion controls.
//!
//! Real artifact (ignored; `HF2Q_GEMMA4_GGUF` = official Gemma 4 26B-A4B GGUF,
//! run each with `--exact --ignored --test-threads=1`):
//! * resume from a restored checkpoint is bitwise equal to the same-plan
//!   resume from a never-overwritten live image, through the real scheduler,
//!   admission, capture, publication, cancellation and eviction paths;
//! * the provisioned four-slot geometry holds two anchor generations per slot
//!   inside the CPU quota, and the quota refuses (not overbooks) beyond it;
//! * the SlotAware prefill-route warmup is invisible to request output.
//!
//! A monolithic cold prefill of the whole prompt is a different Metal graph
//! (different row tiling and the tiny-suffix decode replay), so it is printed
//! as a diagnostic, never used as the oracle — the same contract as
//! `gemma_fresh_and_reused_4096_8193_bounded_outputs_match`.
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

fn official_gguf() -> PathBuf {
    let path = PathBuf::from(
        std::env::var("HF2Q_GEMMA4_GGUF")
            .expect("HF2Q_GEMMA4_GGUF must name the official Gemma 4 26B-A4B GGUF"),
    );
    assert!(path.exists(), "GGUF missing: {}", path.display());
    path
}

fn load_options() -> LoadOptions {
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

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
}

fn spawn_engine(slots: u32) -> (Engine, usize) {
    let loaded = LoadedModel::load(&load_options()).expect("load official Gemma GGUF");
    let LoadedModel::Gemma(g) = &loaded else {
        panic!("HF2Q_GEMMA4_GGUF is not a Gemma 4 artifact")
    };
    let window = g.config.sliding_window;
    assert!(window > 0 && window <= 2048, "sliding window {window}");
    let engine =
        Engine::spawn_with_mode(loaded, 8, None, EngineMode::SlotAware { max_slots: slots })
            .expect("spawn SlotAware Gemma engine");
    (engine, window)
}

fn encode_len(engine: &Engine, text: &str, len: usize) -> Vec<u32> {
    let mut ids = Vec::new();
    while ids.len() < len {
        ids.extend_from_slice(engine.tokenizer().encode(text, false).unwrap().get_ids());
    }
    ids.truncate(len);
    ids
}

/// Tool-turn shape: the handler tags the full input; greedy with logprobs so
/// every emitted token and its top alternatives are compared bit-for-bit.
fn params(max_tokens: usize, tagged_input: &[u32]) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        max_tokens,
        logprobs: true,
        top_logprobs: 4,
        gemma_full_input_checkpoint: gemma_full_input::request_tag(
            true,
            true,
            tagged_input,
            tagged_input,
        ),
        ..SamplingParams::default()
    }
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
}

struct Prompts {
    /// Conversation A input: exceeds the sliding window so decode wraps the
    /// ring over rows the next suffix still attends to.
    a1: Vec<u32>,
    /// A1 plus a 64-token tool result (batched resume route).
    a2: Vec<u32>,
    /// Conversation B: differs from A at token 0.
    b1: Vec<u32>,
    /// B1 plus 5 tokens (tiny-suffix decode-replay resume route).
    b2: Vec<u32>,
}

fn prompts(engine: &Engine, window: usize) -> Prompts {
    let a1 = encode_len(
        engine,
        "You review a Rust parser that reads a u64, checks the range and returns a u16. ",
        window + 4,
    );
    let mut a2 = a1.clone();
    a2.extend(encode_len(
        engine,
        " Tool result: cargo test passed, 4 tests, 0 failed.",
        64,
    ));
    let b1 = encode_len(
        engine,
        "Explain how a sliding window attention cache overwrites its oldest rows. ",
        window + 4,
    );
    assert_ne!(a1[0], b1[0], "conversations must diverge at token 0");
    let mut b2 = b1.clone();
    b2.extend(encode_len(engine, " Continue.", 5));
    Prompts { a1, a2, b1, b2 }
}

#[test]
#[ignore = "official Gemma 4 GGUF: engine-level checkpoint resume/cancel/evict bitwise control"]
fn gemma_full_input_engine_resume_cancel_evict_match_uncheckpointed_reference() {
    assert!(
        gemma_full_input::enabled_for_engine(true, EngineMode::SlotAware { max_slots: 1 }),
        "run without HF2Q_GEMMA_FULL_INPUT_CHECKPOINT=0"
    );
    let rt = runtime();
    let restores = || gemma_full_input::ADMISSION_ANCHOR_RESTORES.load(Ordering::Relaxed);
    let cancel_restores = || gemma_full_input::CANCELLATION_ANCHOR_RESTORES.load(Ordering::Relaxed);

    // Reference: turn one stops at its prefill-selected token, so the slot's
    // live image is never overwritten and turn two resumes WITHOUT a restore.
    let (reference, window) = spawn_engine(1);
    let p = prompts(&reference, window);
    let r_a1 = rt
        .block_on(reference.generate(p.a1.clone(), params(1, &p.a1)))
        .unwrap();
    let before = restores();
    let r_a2 = rt
        .block_on(reference.generate(p.a2.clone(), params(16, &p.a2)))
        .unwrap();
    let r_b1 = rt
        .block_on(reference.generate(p.b1.clone(), params(1, &p.b1)))
        .unwrap();
    let r_b2 = rt
        .block_on(reference.generate(p.b2.clone(), params(16, &p.b2)))
        .unwrap();
    assert_eq!(
        restores(),
        before,
        "reference must resume from the live image"
    );
    let r_a2_cold = rt
        .block_on(reference.generate(p.a2.clone(), params(16, &p.a2)))
        .unwrap();
    rt.block_on(reference.shutdown()).unwrap();
    assert_eq!((r_a1.cached_tokens, r_b1.cached_tokens), (0, 0));
    assert_eq!(r_a2.cached_tokens, p.a1.len(), "reference live resume");
    assert_eq!(r_b2.cached_tokens, p.b1.len(), "reference tiny live resume");
    assert_eq!(r_a2_cold.cached_tokens, 0, "diverged slot must reset");

    // Checkpoint: turn one decodes past the window, overwriting ring rows;
    // turn two must restore the full-input checkpoint and match bit-for-bit.
    let (checkpointed, _) = spawn_engine(1);
    let t_a1 = rt
        .block_on(checkpointed.generate(p.a1.clone(), params(48, &p.a1)))
        .unwrap();
    assert!(
        t_a1.completion_tokens >= 3,
        "turn one must decode past the window to overwrite needed ring rows"
    );
    let before = restores();
    let t_a2 = rt
        .block_on(checkpointed.generate(p.a2.clone(), params(16, &p.a2)))
        .unwrap();
    assert_eq!(
        restores(),
        before + 1,
        "turn two must restore the checkpoint"
    );
    assert_eq!(
        t_a2.cached_tokens,
        p.a1.len(),
        "full-input checkpoint reused"
    );
    assert_same("restored checkpoint, batched suffix", &t_a2, &r_a2);
    // Conversation B evicts A's checkpoint (one slot) and never inherits it.
    let t_b1 = rt
        .block_on(checkpointed.generate(p.b1.clone(), params(48, &p.b1)))
        .unwrap();
    assert_eq!(t_b1.cached_tokens, 0, "foreign conversation got no credit");
    let before = restores();
    let t_b2 = rt
        .block_on(checkpointed.generate(p.b2.clone(), params(16, &p.b2)))
        .unwrap();
    assert_eq!(
        restores(),
        before + 1,
        "tiny suffix must restore the checkpoint"
    );
    assert_eq!(t_b2.cached_tokens, p.b1.len());
    assert_same("restored checkpoint, tiny suffix", &t_b2, &r_b2);
    // A returns after eviction: honest cold prefill, identical to reference cold.
    let t_a2_cold = rt
        .block_on(checkpointed.generate(p.a2.clone(), params(16, &p.a2)))
        .unwrap();
    assert_eq!(
        t_a2_cold.cached_tokens, 0,
        "evicted checkpoint must not be reused"
    );
    assert_same("post-eviction cold", &t_a2_cold, &r_a2_cold);
    rt.block_on(checkpointed.shutdown()).unwrap();

    // Cancellation mid-decode: recovery restores the checkpoint, so the next
    // turn is the same resume as the reference.
    let (cancelled, _) = spawn_engine(1);
    let counter = Arc::new(AtomicU64::new(0));
    let (events_tx, mut events_rx) = mpsc::channel(1);
    rt.block_on(cancelled.generate_stream(
        p.a1.clone(),
        params(64, &p.a1),
        events_tx,
        Some(Arc::clone(&counter)),
        Vec::new(),
    ))
    .unwrap();
    // Three token events guarantee decode wrote past the window before the
    // drop, so recovery has real ring damage to undo.
    for event in 0..3 {
        assert!(rt.block_on(events_rx.recv()).is_some(), "event {event}");
    }
    let cancel_before = cancel_restores();
    drop(events_rx);
    let x_a2 = rt
        .block_on(cancelled.generate(p.a2.clone(), params(16, &p.a2)))
        .unwrap();
    assert!(
        counter.load(Ordering::Relaxed) >= 1,
        "client cancellation recorded"
    );
    assert_eq!(
        cancel_restores(),
        cancel_before + 1,
        "cancellation restored the published checkpoint"
    );
    assert_eq!(
        x_a2.cached_tokens,
        p.a1.len(),
        "checkpoint survived cancellation"
    );
    assert_same("after cancellation", &x_a2, &r_a2);
    rt.block_on(cancelled.shutdown()).unwrap();

    println!(
        "GEMMA_FULL_INPUT_MONOLITHIC_DIAGNOSTIC accepted_as_oracle=false text_equal={} logprobs_equal={}",
        r_a2_cold.text == t_a2.text,
        r_a2_cold.logprobs == t_a2.logprobs
    );
    println!("GEMMA_FULL_INPUT_ENGINE_COMPLETE restore_bitwise=true tiny_bitwise=true cancel_bitwise=true evict_cold=true isolation=true reference_live=true");
}

#[test]
#[ignore = "official Gemma 4 GGUF: four-slot CPU anchor quota bound on the provisioned geometry"]
fn gemma_full_input_four_slot_geometry_holds_two_generations_within_quota() {
    use crate::inference::models::gemma4::kv_cache::{
        anchor_budget, snapshot_gemma_hybrid_slot_anchor,
    };
    let mut loaded = LoadedModel::load(&load_options()).unwrap();
    let LoadedModel::Gemma(g) = &mut loaded else {
        panic!("Gemma required")
    };
    g.provision_multi_seq_kv_for_slot_aware(4).unwrap();
    let context = g
        .context_length
        .unwrap_or(g.config.max_position_embeddings as usize);
    let (active, plan) =
        gemma_full_input::decide_for_scaffold(4, g.multi_seq_kv_hybrid.as_deref(), context)
            .unwrap();
    let plan = plan.expect("Hybrid scaffold provisioned");
    assert!(
        active && plan.fits(),
        "official geometry must activate: {plan:?}"
    );
    let guard = Gemma4KvGuard::take(g).unwrap();
    crate::debug::investigation_env::gemma_full_input::set_worker_active(true);
    assert_eq!(anchor_budget::used_bytes(), 0, "fresh isolated process");
    let hybrid = guard.hybrid.as_ref().unwrap();
    let mut held = Vec::new();
    for slot in 0..4 {
        for _ in 0..anchor_budget::GENERATIONS_PER_SLOT {
            held.push(snapshot_gemma_hybrid_slot_anchor(hybrid, SlotId(slot), context).unwrap());
        }
    }
    let total = plan.total_bytes.unwrap();
    assert_eq!(
        anchor_budget::used_bytes(),
        total,
        "reservation equals the plan"
    );
    // Beyond the in-domain bound the quota refuses instead of overbooking.
    let spare = (anchor_budget::LIMIT - total) / plan.per_slot_bytes;
    for _ in 0..spare {
        held.push(snapshot_gemma_hybrid_slot_anchor(hybrid, SlotId(0), context).unwrap());
    }
    let used = anchor_budget::used_bytes();
    let refused = snapshot_gemma_hybrid_slot_anchor(hybrid, SlotId(0), context)
        .err()
        .expect("quota must refuse past its limit");
    assert!(gemma_full_input::is_quota_refusal(&refused));
    assert_eq!(
        anchor_budget::used_bytes(),
        used,
        "refusal reserved nothing"
    );
    drop(held);
    assert_eq!(anchor_budget::used_bytes(), 0);
    println!(
        "GEMMA_FULL_INPUT_QUOTA_COMPLETE slots=4 per_slot_requested_bytes={} worst_case_bytes={total} limit={} spare_generations={spare}",
        plan.per_slot_bytes,
        anchor_budget::LIMIT
    );
}

#[test]
#[ignore = "official Gemma 4 GGUF: SlotAware prefill-route warmup is output-invisible"]
fn gemma_slot_route_warmup_does_not_change_outputs_or_retained_slots() {
    let rt = runtime();
    let greedy = |max_tokens| SamplingParams {
        temperature: 0.0,
        max_tokens,
        logprobs: true,
        top_logprobs: 4,
        ..SamplingParams::default()
    };
    let run = |warm: bool| {
        let (engine, _) = spawn_engine(4);
        let first_prompt = encode_len(
            &engine,
            "Write a Rust function that parses a u16 port. ",
            458,
        );
        let retained_prompt = encode_len(
            &engine,
            "Summarize why the borrow checker rejects two mutable borrows. ",
            458,
        );
        if warm {
            rt.block_on(engine.warmup()).unwrap();
        }
        let first = rt
            .block_on(engine.generate(first_prompt, greedy(24)))
            .unwrap();
        // Stops at its prefill token: the slot retains exactly the prompt.
        rt.block_on(engine.generate(retained_prompt.clone(), greedy(1)))
            .unwrap();
        // A warmup while slots retain prefixes must pick a clean slot.
        if warm {
            rt.block_on(engine.warmup()).unwrap();
        }
        let mut next = retained_prompt.clone();
        next.extend(encode_len(&engine, " Add a unit test for 65536.", 40));
        let second = rt.block_on(engine.generate(next, greedy(24))).unwrap();
        rt.block_on(engine.shutdown()).unwrap();
        assert_eq!(second.cached_tokens, retained_prompt.len(), "warm={warm}");
        (first, second)
    };
    let (cold_first, cold_second) = run(false);
    let (warm_first, warm_second) = run(true);
    assert_same("first request", &warm_first, &cold_first);
    assert_same(
        "continuation of a retained slot",
        &warm_second,
        &cold_second,
    );
    // Timing is not reported: the cold engine runs first in this process and
    // warms pipelines/pages for the warm engine. Measure benefit by A/B TTFT.
    println!("GEMMA_SLOT_WARMUP_COMPLETE outputs_equal=true retained_slot_untouched=true");
}
