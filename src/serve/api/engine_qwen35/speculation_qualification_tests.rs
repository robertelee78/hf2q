//! ADR-063 D1: qualification gates for exact Qwen warm-turn speculation.
//!
//! A deterministic toy target stands in for the GPU forward. Everything else
//! is production code: `from_prefill_logits` seeds the request,
//! `Qwen35SpecSemanticState::select_and_observe` makes each verifier row's
//! canonical decision with the request's sampling and repetition state,
//! `plan_qwen35_verified_block` picks the committed prefix, and
//! `commit_speculation_output` / `commit_ordinary_target_token` stream the
//! ledger. The harness models only the physical cache (the token ids whose
//! target K/V the slot holds), truncated to `valid_input_tokens` exactly as the
//! K3 and history commits truncate the target and MTP cursors.
//!
//! The toy decision hashes every cached token, so a commit that kept a
//! rejected row, dropped an accepted one, or let a verified token skip the
//! repetition window changes later decisions and fails the comparison with
//! ordinary greedy decode.

use std::time::Duration;

use mlx_native::MlxDevice;

use super::{
    equivalent_target_decisions, is_qwen_server_speculation_exact_eligible,
    is_serial_mtp_exact_eligible, plan_qwen35_verified_block, take_pending_speculation_output,
    HybridPromptCache, Qwen35DecodeState, Qwen35LoadedModel, Qwen35SlotMtpState,
    Qwen35SpecSemanticState,
};
use crate::inference::models::qwen35::kv_cache::HybridKvCache;
use crate::inference::models::qwen35::model::Qwen35Model;
use crate::inference::models::qwen35::{
    default_layer_types, Qwen35Config, Qwen35MoeConfig, Qwen35Variant,
};
use crate::serve::api::engine::SamplingParams;
use crate::serve::api::qwen35_speculation::{
    classify_slot_request, prefill_mtp_catchup_allowed, QwenSpeculationController,
    QwenSpeculationDecision, QwenSpeculationPolicy,
};
use crate::serve::multi_seq_kv::{MultiSeqKvCache, SlotId};

const VOCAB: usize = 64;
const EOS: u32 = 63;
/// Never selected by the toy target, so a draft of it is always rejected.
const NEVER: u32 = 62;
const PROMPT: [u32; 6] = [3, 9, 27, 17, 5, 41];
const ORDINARY_AFTER_ROUND: usize = 8;

fn toy_cfg() -> Qwen35Config {
    Qwen35Config {
        variant: Qwen35Variant::Moe,
        hidden_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        linear_num_key_heads: 4,
        linear_num_value_heads: 8,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        full_attention_interval: 4,
        layer_types: default_layer_types(4, 4),
        partial_rotary_factor: 0.25,
        rope_theta: 1e7,
        rotary_dim: 4,
        mrope_section: [1, 1, 0, 0],
        mrope_interleaved: true,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 1024,
        vocab_size: 64,
        attn_output_gate: true,
        mtp_num_hidden_layers: 0,
        mtp_use_dedicated_embeddings: true,
        intermediate_size: None,
        moe: Some(Qwen35MoeConfig {
            moe_intermediate_size: 16,
            num_experts: 4,
            num_experts_per_tok: 2,
            shared_expert_intermediate_size: 16,
        }),
    }
}

/// Weightless loaded model: the decode-state paths under test read only the
/// vocabulary size, EOS ids, tokenizer fragments and speculation policy.
fn loaded_model(policy: QwenSpeculationPolicy) -> Qwen35LoadedModel {
    Qwen35LoadedModel {
        model: Qwen35Model::empty_from_cfg(toy_cfg()),
        tokenizer: tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
        chat_template: String::new(),
        model_id: "adr063-d1-toy".to_string(),
        model_path: std::path::PathBuf::from("/tmp/adr063-d1-toy.gguf"),
        eos_token_ids: vec![EOS],
        hidden_size: 64,
        vocab_size: VOCAB,
        context_length: Some(1024),
        quant_type: None,
        load_duration: Duration::ZERO,
        provenance: crate::core::provenance::Provenance::External,
        expected_projector_sha256: None,
        vision_projector_profile: None,
        vision_deepstack_output_count: None,
        vision_special_tokens_present: false,
        prompt_cache: HybridPromptCache::new(),
        lcp_registry: crate::serve::kv_persist::lcp_registry::LcpRegistry::new(1),
        kv_metrics_sink: None,
        disk_persistor: None,
        lcp_hydrated_for_cfg: std::collections::HashSet::new(),
        tq_kv_active: false,
        persistent_kv_cache: None,
        speculation: QwenSpeculationController::new(policy),
    }
}

fn greedy(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        max_tokens,
        ..SamplingParams::default()
    }
}

/// The Qwen launchers' default 1.05 penalty: decisions depend on the
/// 64-token prompt+generation window, which verified tokens must update.
fn penalized(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        max_tokens,
        repetition_penalty: 1.05,
        ..SamplingParams::default()
    }
}

#[derive(Clone, Copy, Default)]
struct ToyTarget {
    /// Generated-token index whose target decision is EOS.
    eos_at: Option<usize>,
}

impl ToyTarget {
    /// Logits for the token that follows `context`.
    fn logits(self, context: &[u32]) -> Vec<f32> {
        let mut logits = vec![0.0f32; VOCAB];
        if self
            .eos_at
            .is_some_and(|index| context.len() == PROMPT.len() + index)
        {
            logits[EOS as usize] = 10.0;
            return logits;
        }
        let hash = context
            .iter()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &token| {
                (hash ^ u64::from(token)).wrapping_mul(0x0000_0100_0000_01b3)
            });
        let top = 1 + (hash % 61) as usize;
        let mut runner_up = 1 + ((hash >> 32) % 61) as usize;
        if runner_up == top {
            runner_up = top % 61 + 1;
        }
        // A 1.05 penalty demotes a recently used winner below the runner-up.
        logits[top] = 10.0;
        logits[runner_up] = 9.8;
        logits
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Round {
    matched: usize,
    rejected: usize,
    equivalent_decisions: usize,
    terminal: bool,
}

struct Lane {
    state: Qwen35DecodeState,
    /// Token ids whose target and MTP K/V the slot holds (the cursor).
    cache: Vec<u32>,
    finished: bool,
}

impl Lane {
    fn start(qwen: &Qwen35LoadedModel, target: ToyTarget, params: &SamplingParams) -> Self {
        let prompt = PROMPT.to_vec();
        let prefill_logits = target.logits(&prompt);
        let state = Qwen35DecodeState::from_prefill_logits(
            qwen,
            prompt.clone(),
            params.clone(),
            None,
            SlotId(0),
            0,
            &prefill_logits,
            Duration::ZERO,
            prompt.len(),
            None,
        )
        .expect("seed Qwen decode state");
        let finished = state.finished_at_seed();
        Self {
            state,
            cache: prompt,
            finished,
        }
    }

    fn generated(&self) -> &[u32] {
        &self.state.generated_tokens
    }

    fn newest(&self) -> u32 {
        *self
            .state
            .generated_tokens
            .last()
            .expect("seeded generation")
    }

    /// `decode_tick`'s ordinary branch: feed the newest token, select with
    /// request semantics, commit.
    fn ordinary(&mut self, qwen: &Qwen35LoadedModel, target: ToyTarget) {
        assert!(
            !self.finished,
            "ordinary decision after the generation finished"
        );
        assert!(
            self.state.pending_speculation_output.is_empty(),
            "verified output must drain before any target forward"
        );
        let newest = self.newest();
        self.cache.push(newest);
        let logits = target.logits(&self.cache);
        let forced = self.state.next_ordinary_forced_token();
        let token = self
            .state
            .select_ordinary_logits_token(qwen, logits, forced)
            .expect("ordinary target decision");
        let outcome = self
            .state
            .commit_ordinary_target_token(qwen, token, Duration::from_micros(1))
            .expect("ordinary commit");
        self.finished = outcome.finished;
        self.assert_cache_matches_ledger();
    }

    /// One verified block as `decode_tick_mtp_k3` and
    /// `decode_tick_history_lookup` commit it, through the first token
    /// streamed in the same tick. Later tokens stay queued.
    fn verify(&mut self, qwen: &Qwen35LoadedModel, target: ToyTarget, drafts: &[u32]) -> Round {
        assert!(!self.finished && self.state.pending_speculation_output.is_empty());
        let prior = self.cache.len();
        let mut rows = Vec::with_capacity(drafts.len() + 1);
        rows.push(self.newest());
        rows.extend_from_slice(drafts);
        let mut verify_logits: Vec<Vec<f32>> = (1..=rows.len())
            .map(|fed_rows| {
                let mut context = self.cache.clone();
                context.extend_from_slice(&rows[..fed_rows]);
                target.logits(&context)
            })
            .collect();
        self.cache.extend_from_slice(&rows);

        let params = self.state.params.clone();
        let mut semantic = Qwen35SpecSemanticState::from_decode(&self.state);
        let plan = plan_qwen35_verified_block(drafts, |row| {
            semantic.select_and_observe(qwen, &params, &mut verify_logits[row])
        })
        .expect("canonical accept walk");
        assert_eq!(
            plan.carry_hidden_row + 1,
            plan.valid_input_tokens,
            "the carried verifier row must be the last committed row"
        );
        self.cache.truncate(prior + plan.valid_input_tokens);
        let round = Round {
            matched: plan.matched_drafts,
            rejected: plan.rejected_drafts,
            equivalent_decisions: equivalent_target_decisions(
                &plan.output,
                plan.terminal_after_pending,
            ),
            terminal: plan.terminal_after_pending,
        };
        self.state.terminal_after_pending = plan.terminal_after_pending;
        self.state.pending_speculation_output = plan.output;
        match take_pending_speculation_output(&mut self.state.pending_speculation_output) {
            Some(token) => self.commit_verified(qwen, token),
            None => {
                self.state.finish_reason = "stop";
                self.finished = true;
                self.assert_cache_matches_ledger();
            }
        }
        round
    }

    /// `decode_tick`'s first branch: queued verified output streams one token
    /// per tick without a forward.
    fn drain(&mut self, qwen: &Qwen35LoadedModel) {
        while !self.finished {
            let Some(token) =
                take_pending_speculation_output(&mut self.state.pending_speculation_output)
            else {
                break;
            };
            self.commit_verified(qwen, token);
        }
        assert!(
            self.state.pending_speculation_output.is_empty(),
            "a finished generation left verified output queued"
        );
    }

    fn speculate(&mut self, qwen: &Qwen35LoadedModel, target: ToyTarget, drafts: &[u32]) -> Round {
        let round = self.verify(qwen, target, drafts);
        self.drain(qwen);
        round
    }

    fn commit_verified(&mut self, qwen: &Qwen35LoadedModel, token: u32) {
        let outcome = self
            .state
            .commit_speculation_output(qwen, token)
            .expect("verified commit");
        self.finished = outcome.finished;
        self.assert_cache_matches_ledger();
    }

    /// The cursor holds the prompt, the streamed ledger and any queued
    /// verified output, except the newest token, which the next forward
    /// feeds. A stop was predicted from the newest committed (or queued)
    /// token, so after a stop decision that token is cached too and nothing
    /// after it is.
    fn assert_cache_matches_ledger(&self) {
        let mut expected = self.state.prompt_tokens.clone();
        expected.extend_from_slice(&self.state.generated_tokens);
        expected.extend(self.state.pending_speculation_output.iter().copied());
        let stop_decided = self.state.terminal_after_pending
            || (self.finished && self.state.finish_reason == "stop");
        if !stop_decided {
            expected.pop();
        }
        assert_eq!(
            self.cache, expected,
            "physical cache diverged from the committed ledger"
        );
    }
}

fn ordinary_tokens(
    qwen: &Qwen35LoadedModel,
    target: ToyTarget,
    params: &SamplingParams,
    len: usize,
) -> Vec<u32> {
    let mut lane = Lane::start(qwen, target, params);
    while !lane.finished && lane.generated().len() < len {
        lane.ordinary(qwen, target);
    }
    lane.state.generated_tokens
}

/// Compare every observable piece of a speculative lane with ordinary
/// decode advanced through the same number of target decisions.
fn assert_matches_ordinary(
    qwen: &Qwen35LoadedModel,
    target: ToyTarget,
    params: &SamplingParams,
    lane: &Lane,
    label: &str,
) {
    let mut reference = Lane::start(qwen, target, params);
    while !reference.finished && reference.generated().len() < lane.generated().len() {
        reference.ordinary(qwen, target);
    }
    if lane.finished && !reference.finished {
        // A speculative stop must be ordinary decode's very next decision.
        reference.ordinary(qwen, target);
    }
    assert_eq!(
        lane.generated(),
        reference.generated(),
        "{label}: streamed tokens"
    );
    assert!(
        !lane.generated().contains(&EOS),
        "{label}: EOS reached the stream"
    );
    assert_eq!(lane.finished, reference.finished, "{label}: finished");
    assert_eq!(
        lane.state.finish_reason, reference.state.finish_reason,
        "{label}: finish_reason"
    );
    assert_eq!(lane.cache, reference.cache, "{label}: physical cache");
    assert_eq!(
        lane.state.sampling_history, reference.state.sampling_history,
        "{label}: repetition window"
    );
    assert_eq!(lane.state.step, reference.state.step, "{label}: step");
    assert_eq!(
        lane.state.decoded_text, reference.state.decoded_text,
        "{label}: decoded text"
    );
}

#[derive(Clone, Copy, Debug)]
enum Acceptance {
    Full,
    Partial,
    Zero,
}

impl Acceptance {
    /// Three K3-sized drafts against the ordinary continuation `next`.
    fn drafts(self, next: &[u32]) -> [u32; 3] {
        match self {
            Self::Full => [next[0], next[1], next[2]],
            Self::Partial => [next[0], NEVER, NEVER],
            Self::Zero => [NEVER, NEVER, NEVER],
        }
    }

    fn expected_round(self) -> Round {
        let (matched, rejected, equivalent_decisions) = match self {
            Self::Full => (3, 0, 4),
            Self::Partial => (1, 1, 2),
            Self::Zero => (0, 1, 1),
        };
        Round {
            matched,
            rejected,
            equivalent_decisions,
            terminal: false,
        }
    }
}

#[test]
fn full_partial_and_zero_acceptance_then_eight_ordinary_decisions_match_greedy() {
    let qwen = loaded_model(QwenSpeculationPolicy::Off);
    let target = ToyTarget::default();
    let (plain, with_penalty) = (greedy(64), penalized(64));
    assert_ne!(
        ordinary_tokens(&qwen, target, &plain, 16),
        ordinary_tokens(&qwen, target, &with_penalty, 16),
        "vacuous: the repetition window never changed a toy decision"
    );
    for params in [plain, with_penalty] {
        let reference = ordinary_tokens(&qwen, target, &params, 32);
        for acceptance in [Acceptance::Full, Acceptance::Partial, Acceptance::Zero] {
            // Production admits a proposer only after an ordinary baseline.
            for warmup in [1, 3] {
                let label = format!(
                    "{acceptance:?} after {warmup} ordinary (penalty {})",
                    params.repetition_penalty
                );
                let mut lane = Lane::start(&qwen, target, &params);
                for _ in 0..warmup {
                    lane.ordinary(&qwen, target);
                }
                let committed = lane.generated().len();
                let drafts = acceptance.drafts(&reference[committed..]);
                let round = lane.speculate(&qwen, target, &drafts);
                assert_eq!(round, acceptance.expected_round(), "{label}");
                assert_eq!(
                    lane.generated().len(),
                    committed + round.equivalent_decisions,
                    "{label}: one streamed token per target decision"
                );
                for _ in 0..ORDINARY_AFTER_ROUND {
                    lane.ordinary(&qwen, target);
                }
                assert_matches_ordinary(&qwen, target, &params, &lane, &label);
            }
        }
    }
}

#[test]
fn chained_rounds_of_mixed_acceptance_match_greedy() {
    let qwen = loaded_model(QwenSpeculationPolicy::Off);
    let target = ToyTarget::default();
    for params in [greedy(64), penalized(64)] {
        let reference = ordinary_tokens(&qwen, target, &params, 40);
        let mut lane = Lane::start(&qwen, target, &params);
        lane.ordinary(&qwen, target);
        for acceptance in [
            Acceptance::Full,
            Acceptance::Zero,
            Acceptance::Partial,
            Acceptance::Full,
            Acceptance::Partial,
        ] {
            let committed = lane.generated().len();
            let round = lane.speculate(&qwen, target, &acceptance.drafts(&reference[committed..]));
            assert_eq!(round, acceptance.expected_round(), "{acceptance:?}");
        }
        for _ in 0..ORDINARY_AFTER_ROUND {
            lane.ordinary(&qwen, target);
        }
        assert_matches_ordinary(&qwen, target, &params, &lane, "chained rounds");
    }
}

#[test]
fn eos_inside_or_after_a_verified_block_stops_exactly_like_greedy() {
    let qwen = loaded_model(QwenSpeculationPolicy::Off);
    // After one ordinary decision the verifier rows decide generated tokens
    // 2, 3, 4 and the bonus 5 (index 1 is the seed's successor row 0).
    struct Case {
        label: &'static str,
        eos_at: usize,
        drafts: fn(&[u32]) -> [u32; 3],
        round: Round,
    }
    let cases = [
        Case {
            label: "accepted EOS draft in the middle of the run",
            eos_at: 3,
            drafts: |next| [next[0], EOS, NEVER],
            round: Round {
                matched: 2,
                rejected: 0,
                equivalent_decisions: 2,
                terminal: true,
            },
        },
        Case {
            label: "accepted EOS as the first draft",
            eos_at: 2,
            drafts: |_| [EOS, NEVER, NEVER],
            round: Round {
                matched: 1,
                rejected: 0,
                equivalent_decisions: 1,
                terminal: true,
            },
        },
        Case {
            label: "EOS bonus after full acceptance",
            eos_at: 5,
            drafts: |next| [next[0], next[1], next[2]],
            round: Round {
                matched: 3,
                rejected: 0,
                equivalent_decisions: 4,
                terminal: true,
            },
        },
        Case {
            label: "target EOS rejects a non-EOS draft",
            eos_at: 3,
            drafts: |next| [next[0], NEVER, NEVER],
            round: Round {
                matched: 1,
                rejected: 1,
                equivalent_decisions: 2,
                terminal: true,
            },
        },
        Case {
            label: "target EOS on the first verifier row",
            eos_at: 2,
            drafts: |_| [NEVER, NEVER, NEVER],
            round: Round {
                matched: 0,
                rejected: 1,
                equivalent_decisions: 1,
                terminal: true,
            },
        },
    ];
    for params in [greedy(64), penalized(64)] {
        for case in &cases {
            let target = ToyTarget {
                eos_at: Some(case.eos_at),
            };
            let reference = ordinary_tokens(&qwen, target, &params, 64);
            assert_eq!(
                reference.len(),
                case.eos_at,
                "{}: reference stop",
                case.label
            );
            let mut lane = Lane::start(&qwen, target, &params);
            lane.ordinary(&qwen, target);
            let committed = lane.generated().len();
            let drafts = (case.drafts)(&reference[committed..]);
            let round = lane.speculate(&qwen, target, &drafts);
            assert_eq!(round, case.round, "{}", case.label);
            assert!(lane.finished, "{}: generation must stop", case.label);
            assert_eq!(lane.state.finish_reason, "stop", "{}", case.label);
            assert_matches_ordinary(&qwen, target, &params, &lane, case.label);
        }

        // A proposed EOS the target rejects never terminates the request.
        let target = ToyTarget::default();
        let reference = ordinary_tokens(&qwen, target, &params, 16);
        let mut lane = Lane::start(&qwen, target, &params);
        lane.ordinary(&qwen, target);
        let committed = lane.generated().len();
        let round = lane.speculate(&qwen, target, &[reference[committed], EOS, NEVER]);
        assert_eq!(
            round,
            Round {
                matched: 1,
                rejected: 1,
                equivalent_decisions: 2,
                terminal: false,
            }
        );
        assert!(!lane.finished, "a rejected EOS draft stopped the request");
        for _ in 0..ORDINARY_AFTER_ROUND {
            lane.ordinary(&qwen, target);
        }
        assert_matches_ordinary(&qwen, target, &params, &lane, "rejected EOS draft");
    }
}

#[test]
fn verified_block_at_the_length_boundary_finishes_like_greedy() {
    let qwen = loaded_model(QwenSpeculationPolicy::Off);
    let target = ToyTarget::default();
    // K3 runs only while at least four output tokens remain: after two
    // committed tokens, max_tokens = 6 leaves exactly the four verifier rows.
    for params in [greedy(6), penalized(6)] {
        let reference = ordinary_tokens(&qwen, target, &params, 64);
        assert_eq!(reference.len(), 6);
        for acceptance in [Acceptance::Full, Acceptance::Partial, Acceptance::Zero] {
            let mut lane = Lane::start(&qwen, target, &params);
            lane.ordinary(&qwen, target);
            let committed = lane.generated().len();
            let round = lane.speculate(&qwen, target, &acceptance.drafts(&reference[committed..]));
            assert_eq!(round, acceptance.expected_round(), "{acceptance:?}");
            while !lane.finished {
                lane.ordinary(&qwen, target);
            }
            assert_eq!(lane.state.finish_reason, "length", "{acceptance:?}");
            assert_matches_ordinary(
                &qwen,
                target,
                &params,
                &lane,
                &format!("{acceptance:?} at the length boundary"),
            );
        }
    }
}

/// A client that disconnects while verified output is still queued leaves
/// the slot's GPU state ahead of its streamed ledger. The decode state must
/// refuse to publish a speculative boundary there, and cancellation recovery
/// (`restore_slot_anchor`) must rewind both the target and the MTP cursor to
/// the verified prompt boundary without touching a peer slot.
#[test]
fn cancellation_mid_verified_drain_publishes_nothing_and_anchor_rewinds_target_and_mtp() {
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let device = MlxDevice::new().expect("Metal device");
    let qwen = loaded_model(QwenSpeculationPolicy::Off);
    let target = ToyTarget::default();
    let params = greedy(64);
    let reference = ordinary_tokens(&qwen, target, &params, 16);

    let mut lane = Lane::start(&qwen, target, &params);
    lane.ordinary(&qwen, target);
    let hidden = 64;
    let verifier_hidden = device
        .alloc_buffer(
            hidden * std::mem::size_of::<f32>(),
            mlx_native::DType::F32,
            vec![1, hidden],
        )
        .expect("verifier hidden row");
    lane.state.mtp = Some(Qwen35SlotMtpState { verifier_hidden });
    let committed = lane.generated().len();
    let round = lane.verify(
        &qwen,
        target,
        &Acceptance::Full.drafts(&reference[committed..]),
    );
    assert_eq!(round, Acceptance::Full.expected_round());
    assert_eq!(lane.state.pending_speculation_output.len(), 3);
    let cursor = lane.cache.len();
    let ledger = lane.state.prompt_tokens.len() + lane.generated().len();
    assert!(
        cursor > ledger,
        "the verified block must run ahead of the stream"
    );
    for valid in [ledger, cursor] {
        assert!(
            lane.state.spec_prefix_candidate(valid).is_none(),
            "queued verified output published a speculative boundary at {valid}"
        );
    }
    assert!(
        lane.state.retained_prefix(cursor).len() < cursor,
        "the streamed ledger cannot describe the advanced cursor"
    );

    let mut cfg = toy_cfg();
    cfg.mtp_num_hidden_layers = 1;
    let mut kv = HybridKvCache::new(&cfg, &device, 64, 2).expect("two-slot cache with MTP");
    let (slot, peer) = (SlotId(1), SlotId(0));
    kv.append_for_seq(slot, PROMPT.len() as u32)
        .expect("prompt cursor");
    kv.append_for_seq(peer, 11).expect("peer cursor");
    kv.validate_speculative_cursors_for_slot(slot, PROMPT.len())
        .expect("prompt boundary cursors");
    let anchor = kv
        .snapshot_slot_anchor(slot, PROMPT.len())
        .expect("verified prompt anchor");
    kv.append_for_seq(slot, (cursor - PROMPT.len()) as u32)
        .expect("decode and verifier rows");
    kv.validate_speculative_cursors_for_slot(slot, cursor)
        .expect("mid-drain cursors");

    kv.restore_slot_anchor(slot, &anchor)
        .expect("cancellation anchor restore");
    kv.validate_speculative_cursors_for_slot(slot, PROMPT.len())
        .expect("target and MTP cursors must return to the prompt boundary");
    kv.validate_speculative_cursors_for_slot(peer, 11)
        .expect("a peer slot's speculative cursors must be untouched");

    // Draining instead of cancelling publishes the ordinary boundary.
    lane.drain(&qwen);
    let boundary = lane
        .state
        .spec_prefix_candidate(lane.cache.len())
        .expect("drained MTP state publishes its boundary");
    assert_eq!(boundary.token_count, lane.cache.len());
}

/// Server speculation compares each draft with the target's canonical
/// selection and has no rejection-sampling acceptance, so it is exact only
/// for deterministic selection. Every stochastic or response-shaping knob
/// must keep a request on ordinary target decode: no admission, no MTP
/// prompt catch-up and no history proposer.
#[test]
fn sampled_and_response_shaping_requests_never_speculate() {
    let qwen = loaded_model(QwenSpeculationPolicy::Auto);
    let target = ToyTarget::default();
    let eligible = |params: &SamplingParams| {
        let lane = Lane::start(&qwen, target, params);
        let exact = is_qwen_server_speculation_exact_eligible(params);
        let admitted = classify_slot_request(QwenSpeculationPolicy::Auto, exact, false, false)
            == QwenSpeculationDecision::Eligible;
        let captures_mtp =
            prefill_mtp_catchup_allowed(QwenSpeculationPolicy::Auto, exact, true, true, false);
        let history = lane.state.history_lookup.is_some();
        assert_eq!(admitted, captures_mtp);
        assert_eq!(admitted, history);
        admitted
    };
    assert!(eligible(&greedy(32)));
    assert!(eligible(&penalized(32)));

    let mutations: [(&str, fn(&mut SamplingParams)); 12] = [
        ("temperature", |p| p.temperature = 0.7),
        ("top_k", |p| p.top_k = 40),
        ("top_p", |p| p.top_p = 0.9),
        ("min_p", |p| p.min_p = 0.05),
        ("seed", |p| p.seed = Some(7)),
        ("logprobs", |p| p.logprobs = true),
        ("top_logprobs", |p| p.top_logprobs = 3),
        ("logit_bias", |p| {
            p.logit_bias.insert(7, 2.0);
        }),
        ("frequency_penalty", |p| p.frequency_penalty = 0.5),
        ("presence_penalty", |p| p.presence_penalty = 0.5),
        ("stop", |p| p.stop_strings.push("END".to_string())),
        ("parallel_tool_calls", |p| p.parallel_tool_calls = true),
    ];
    for (knob, mutate) in mutations {
        let mut params = greedy(32);
        mutate(&mut params);
        assert!(!eligible(&params), "{knob} admitted speculation");
        assert!(
            !is_serial_mtp_exact_eligible(&params),
            "{knob} admitted serial MTP"
        );
    }
}
