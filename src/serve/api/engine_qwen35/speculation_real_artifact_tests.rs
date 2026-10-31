//! ADR-063 D1 real-artifact gate: SlotAware Qwen speculation `auto` against
//! `off` on the official Qwen3.8 GGUF, with its native MTP head, through the
//! production engine at four slots.
//!
//! Each arm loads the artifact, pins the speculation policy, spawns a
//! `SlotAware { max_slots: 4 }` engine and runs three phases:
//!
//! 1. **Solo.** Four requests one at a time. No joint step can form, so every
//!    eligible decode turn runs serial history lookup and K3 MTP rounds.
//! 2. **Cohort.** Four concurrent warm follow-ups that append to each solo
//!    turn, so admission reuses retained prefixes and their speculative
//!    boundaries. Joint rows (which never speculate) mix with serial K3 turns.
//! 3. **Cancellation.** A streaming follow-up is dropped after twelve deltas,
//!    while verified K3 output is usually still queued, and the same request
//!    is then served unary. Deterministic mid-drain coverage lives in
//!    `speculation_qualification_tests`.
//!
//! Both arms send identical prompts (follow-ups are built from the `off`
//! answers). The `off` arm also requests top-2 logprobs, which never changes
//! its greedy decisions. K3 verifies on the four-row target route, a declared
//! Q1 Class B alternate of the one-row route (largest observed benign
//! decision-boundary shift 0.093 nat on Qwen3.8), and `auto` chooses rounds
//! from wall-clock cost and cohort timing. So an `auto` response must equal
//! `off`, or diverge first at a decision whose `off` top-1/top-2 margin is
//! within the Q1 near-tie threshold τ = 0.25 nat. A divergence at a wider
//! margin is `RouteSensitive` and fails. The `auto` arm must also run MTP
//! rounds with both accepted and rejected drafts, and at least one request
//! must stop on EOS, so the comparison is not vacuous.
//!
//! ```text
//! HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 \
//! HF2Q_QWEN38_GGUF=/abs/Qwen3.8-27B-vanilla.gguf \
//!   cargo test --release --lib qwen38_slot_aware_speculation_auto_matches_off \
//!   -- --ignored --exact --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::serve::api::engine::{
    Engine, EngineMode, GenerationResult, LoadOptions, LoadedModel, SamplingParams,
};
use crate::serve::api::qwen35_speculation::{
    QwenSpeculationController, QwenSpeculationPolicy, TELEMETRY,
};
use crate::serve::api::sse::GenerationEvent;

const MODEL_ENV: &str = "HF2Q_QWEN38_GGUF";
const MAX_SLOTS: u32 = 4;
const MAX_TOKENS: usize = 160;
const CANCEL_AFTER_DELTAS: usize = 12;
/// Q1 §3.2 decision-envelope threshold τ (nats).
const NEAR_TIE_NAT: f32 = 0.25;

/// Coding-agent-like turns: repetitive code and lists keep history lookup and
/// MTP acceptance high; the one-word reply stops on EOS.
const PROMPTS: [&str; 4] = [
    "Write a Rust function `fib(n: u64) -> u64` that computes Fibonacci numbers iteratively, \
     then write the same function again for `u128`.",
    "List the integers from 1 to 60, separated by commas, with no other text.",
    "Reply with exactly the single word OK.",
    "Write a Python class `Stack` with push, pop, peek and is_empty methods, each with a \
     one-line docstring.",
];
const FOLLOW_UP: &str = "Now repeat your previous answer exactly, without any changes.";

#[derive(Debug)]
struct Observed {
    text: String,
    reasoning: Option<String>,
    completion_tokens: usize,
    finish_reason: &'static str,
    /// `off` arm only: committed token ids, and the top-1 minus top-2
    /// logprob margin of every target decision (a final EOS included).
    tokens: Vec<u32>,
    margins: Vec<f32>,
}

impl Observed {
    fn new(result: GenerationResult) -> Self {
        let records = result.logprobs.unwrap_or_default();
        Self {
            tokens: records
                .iter()
                .take(result.completion_tokens)
                .map(|record| record.token_id)
                .collect(),
            margins: records
                .iter()
                .map(|record| match record.top.as_slice() {
                    [first, second, ..] => first.1 - second.1,
                    _ => f32::INFINITY,
                })
                .collect(),
            text: result.text,
            reasoning: result.reasoning_text,
            completion_tokens: result.completion_tokens,
            finish_reason: result.finish_reason,
        }
    }

    fn same_response(&self, other: &Self) -> bool {
        self.text == other.text
            && self.reasoning == other.reasoning
            && self.completion_tokens == other.completion_tokens
            && self.finish_reason == other.finish_reason
    }
}

struct Arm {
    solo: Vec<Observed>,
    cohort: Vec<Observed>,
    after_cancel: Observed,
    /// MTP proposals, accepted and rejected MTP drafts, history proposals.
    speculation: [u64; 4],
    tokenizer: tokenizers::Tokenizer,
}

fn speculation_counters() -> [u64; 4] {
    [
        TELEMETRY.mtp_proposals.load(Ordering::Relaxed),
        TELEMETRY.mtp_accepted_tokens.load(Ordering::Relaxed),
        TELEMETRY.mtp_rejected_tokens.load(Ordering::Relaxed),
        TELEMETRY.history_lookup_proposals.load(Ordering::Relaxed),
    ]
}

fn user_turn(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

fn first_turns() -> Vec<String> {
    PROMPTS.iter().map(|text| user_turn(text)).collect()
}

fn follow_up(turn: &str, answer: &str) -> String {
    format!("{turn}{answer}<|im_end|>\n{}", user_turn(FOLLOW_UP))
}

/// Run one arm. `reference` is the `off` arm whose answers build the
/// follow-up prompts, so both arms see identical inputs.
fn run_arm(policy: QwenSpeculationPolicy, reference: Option<&Arm>) -> Arm {
    let path = PathBuf::from(
        std::env::var_os(MODEL_ENV)
            .unwrap_or_else(|| panic!("the D1 real-artifact gate requires {MODEL_ENV}")),
    );
    assert!(
        path.is_file(),
        "{MODEL_ENV} is not a file: {}",
        path.display()
    );
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
    };
    let mut loaded = LoadedModel::load(&options).expect("load Qwen3.8");
    let tokenizer = {
        let LoadedModel::Qwen35(qwen) = &mut loaded else {
            panic!("{MODEL_ENV} must name a Qwen3.5-family GGUF")
        };
        assert!(
            qwen.model.mtp.is_some(),
            "the D1 gate needs the native MTP head"
        );
        assert!(qwen.model.kv_graft.is_none(), "no graft may be bound");
        qwen.speculation = QwenSpeculationController::new(policy);
        qwen.tokenizer.clone()
    };
    let encode = |text: &str| -> Vec<u32> {
        tokenizer
            .encode(text, false)
            .expect("tokenize D1 prompt")
            .get_ids()
            .to_vec()
    };
    let with_logprobs = policy == QwenSpeculationPolicy::Off;
    let params = SamplingParams {
        max_tokens: MAX_TOKENS,
        logprobs: with_logprobs,
        top_logprobs: if with_logprobs { 2 } else { 0 },
        ..SamplingParams::default()
    };

    let before = speculation_counters();
    let engine = Engine::spawn_with_mode(
        loaded,
        8,
        None,
        EngineMode::SlotAware {
            max_slots: MAX_SLOTS,
        },
    )
    .expect("spawn Qwen3.8 SlotAware engine");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(MAX_SLOTS as usize)
        .enable_all()
        .build()
        .expect("tokio runtime");

    let first_turns = first_turns();
    let solo: Vec<Observed> = first_turns
        .iter()
        .map(|turn| {
            Observed::new(
                runtime
                    .block_on(engine.generate(encode(turn.as_str()), params.clone()))
                    .expect("solo generate"),
            )
        })
        .collect();

    let answers = reference.map_or(&solo, |arm| &arm.solo);
    let follow_ups: Vec<String> = first_turns
        .iter()
        .zip(answers)
        .map(|(turn, answer)| follow_up(turn, &answer.text))
        .collect();
    let cohort: Vec<Observed> = runtime.block_on(async {
        let handles: Vec<_> = follow_ups
            .iter()
            .map(|turn| {
                let engine = engine.clone();
                let prompt = encode(turn.as_str());
                let params = params.clone();
                tokio::spawn(async move { engine.generate(prompt, params).await })
            })
            .collect();
        let mut results: Vec<Observed> = Vec::with_capacity(handles.len());
        for handle in handles {
            results.push(Observed::new(
                handle.await.expect("join").expect("cohort generate"),
            ));
        }
        results
    });

    let cancelled_answer = reference.map_or(&cohort[0].text, |arm| &arm.cohort[0].text);
    let cancelled_prompt = encode(follow_up(&follow_ups[0], cancelled_answer).as_str());
    let after_cancel: Observed = runtime.block_on(async {
        let (events, mut receiver) = tokio::sync::mpsc::channel(4);
        engine
            .generate_stream(
                cancelled_prompt.clone(),
                params.clone(),
                events,
                None,
                Vec::new(),
            )
            .await
            .expect("stream admission");
        let mut deltas = 0;
        while deltas < CANCEL_AFTER_DELTAS {
            let event = receiver
                .recv()
                .await
                .expect("stream closed before cancellation");
            assert!(
                !matches!(
                    event,
                    GenerationEvent::Done { .. } | GenerationEvent::Error(_)
                ),
                "stream finished before cancellation: {event:?}"
            );
            if matches!(event, GenerationEvent::Delta { .. }) {
                deltas += 1;
            }
        }
        drop(receiver);
        Observed::new(
            engine
                .generate(cancelled_prompt, params.clone())
                .await
                .expect("generate after cancellation"),
        )
    });

    runtime
        .block_on(engine.shutdown())
        .expect("shutdown Qwen3.8 engine");
    let after = speculation_counters();
    Arm {
        solo,
        cohort,
        after_cancel,
        speculation: std::array::from_fn(|index| after[index] - before[index]),
        tokenizer,
    }
}

/// Byte offset in `left` of the first character that differs from `right`.
fn first_difference(left: &str, right: &str) -> usize {
    left.char_indices()
        .zip(right.chars())
        .find(|((_, left), right)| left != right)
        .map_or_else(|| left.len().min(right.len()), |((offset, _), _)| offset)
}

/// `None` when `auto` equals `off`; otherwise the near-tie note for the first
/// divergent `off` decision. Panics on a divergence outside the Q1 envelope.
fn classify_divergence(
    label: &str,
    auto: &Observed,
    off: &Observed,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<String> {
    if auto.same_response(off) {
        return None;
    }
    // The engine streams one decoded fragment per committed token, and the
    // response text is that stream after the reasoning split.
    let fragments: Vec<String> = off
        .tokens
        .iter()
        .map(|&token| tokenizer.decode(&[token], false).unwrap_or_default())
        .collect();
    let raw: String = fragments.concat();
    let lead = raw.find(off.text.as_str()).unwrap_or_else(|| {
        panic!(
            "{label}: cannot localize the off text in its token stream; classify with Q1 envelope"
        )
    });
    let offset = lead + first_difference(&off.text, &auto.text);
    let mut end = 0;
    let divergence = fragments
        .iter()
        .position(|fragment| {
            end += fragment.len();
            end > offset
        })
        .unwrap_or(fragments.len());
    let margin = off
        .margins
        .get(divergence)
        .copied()
        .unwrap_or(f32::INFINITY);
    assert!(
        margin <= NEAR_TIE_NAT,
        "{label}: auto diverges from off at decision {divergence}, where the off top-1/top-2 \
         margin is {margin} nat > {NEAR_TIE_NAT}: RouteSensitive, not a near-tie\n\
         off:  {:?}\nauto: {:?}",
        off.text,
        auto.text
    );
    Some(format!(
        "{label}: near-tie divergence at decision {divergence} (off margin {margin:.3} nat)"
    ))
}

#[test]
#[ignore = "requires HF2Q_QWEN38_GGUF (official Qwen3.8 artifact with its MTP head) and Apple Metal"]
fn qwen38_slot_aware_speculation_auto_matches_off() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let off = run_arm(QwenSpeculationPolicy::Off, None);
    let auto = run_arm(QwenSpeculationPolicy::Auto, Some(&off));
    eprintln!(
        "[D1] auto MTP proposals={} accepted={} rejected={} history proposals={}",
        auto.speculation[0], auto.speculation[1], auto.speculation[2], auto.speculation[3]
    );

    assert_eq!(off.speculation, [0; 4], "the off arm speculated");
    assert!(
        auto.speculation[0] > 0,
        "the auto arm never ran an MTP round"
    );
    assert!(
        auto.speculation[1] > 0 && auto.speculation[2] > 0,
        "the auto arm needs both accepted and rejected MTP drafts"
    );
    assert!(
        off.solo
            .iter()
            .chain(&off.cohort)
            .any(|result| result.finish_reason == "stop"),
        "no request stopped on EOS"
    );
    assert_eq!(auto.solo.len(), PROMPTS.len());
    assert_eq!(auto.cohort.len(), PROMPTS.len());

    let pairs = auto
        .solo
        .iter()
        .zip(&off.solo)
        .enumerate()
        .map(|(index, pair)| (format!("solo request {index}"), pair))
        .chain(
            auto.cohort
                .iter()
                .zip(&off.cohort)
                .enumerate()
                .map(|(index, pair)| (format!("cohort follow-up {index}"), pair)),
        )
        .chain(std::iter::once((
            "request after a cancelled stream".to_string(),
            (&auto.after_cancel, &off.after_cancel),
        )));
    let mut exact = 0;
    for (label, (speculated, ordinary)) in pairs {
        match classify_divergence(&label, speculated, ordinary, &off.tokenizer) {
            None => exact += 1,
            Some(note) => eprintln!("[D1] {note}"),
        }
    }
    eprintln!(
        "[D1] {exact}/{} responses identical to off",
        2 * PROMPTS.len() + 1
    );
}
