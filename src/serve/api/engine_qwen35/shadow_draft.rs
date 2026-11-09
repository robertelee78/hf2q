//! ADR-065 E3 counterfactual proposer diagnostic.
//!
//! `HF2Q_ADR065_SHADOW_DRAFT=1` runs MTP K3 and request-history proposals only
//! at joint-ready states. Proposals never enter target verification or commit.
//! Their prefixes are compared with later committed target decisions, so the
//! served transcript remains owned by ordinary decode.

use std::collections::VecDeque;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use serde_json::json;

use crate::inference::spec_decode::ngram_proposer::{HistoryLookupConfig, HistoryLookupIndex};
use crate::serve::api::qwen35_speculation::QwenSpeculationPolicy;
use crate::serve::multi_seq_kv::SlotId;

use super::k3_transaction::K3_VERIFY_ROWS;
use super::{
    effective_repetition_penalty, HybridKvCache, Qwen35DecodeState, Qwen35LoadedModel,
    ToolCallPolicy,
};
use crate::serve::api::engine_supervisor::EngineSupervisor;

pub(super) const ENV: &str = "HF2Q_ADR065_SHADOW_DRAFT";
pub(super) const SCHEMA: &str = "hf2q.adr065.shadow-draft.v1";

static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var(ENV).as_deref() == Ok("1"));

pub(super) fn enabled() -> bool {
    *ENABLED
}

pub(super) fn prefill_mtp_allowed(text_only: bool, mtp_runtime: bool, graft_bound: bool) -> bool {
    enabled() && text_only && mtp_runtime && !graft_bound
}

#[derive(Clone, Copy)]
struct TransformFlags {
    no_grammar: bool,
    no_repetition_penalty: bool,
    no_forced_thinking: bool,
    no_tool_trigger: bool,
}

impl TransformFlags {
    fn from_request(state: &Qwen35DecodeState) -> Self {
        Self {
            no_grammar: state.grammar_runtime.is_none(),
            no_repetition_penalty: effective_repetition_penalty(&state.params) == 1.0,
            no_forced_thinking: !state.params.reasoning_forced_open
                && state.params.thinking_token_budget.is_none(),
            no_tool_trigger: state.params.tool_call_policy != ToolCallPolicy::AutoLazyGrammar,
        }
    }

    fn all_clear(self) -> bool {
        self.no_grammar
            && self.no_repetition_penalty
            && self.no_forced_thinking
            && self.no_tool_trigger
    }
}

#[derive(Clone, Copy)]
enum Proposer {
    History,
    MtpK3,
}

impl Proposer {
    const fn label(self) -> &'static str {
        match self {
            Self::History => "history",
            Self::MtpK3 => "mtp_k3",
        }
    }
}

struct ActiveProposal {
    proposer: Proposer,
    drafts: Vec<u32>,
    matched: usize,
    start_position: usize,
    context_bucket: &'static str,
    proposal_cost: Duration,
    transforms: TransformFlags,
}

pub(super) struct ShadowDraftState {
    history: HistoryLookupIndex,
    active: VecDeque<ActiveProposal>,
    turn_type: &'static str,
}

impl ShadowDraftState {
    pub(super) fn new(
        policy: QwenSpeculationPolicy,
        prompt_tokens: &[u32],
        generated_tokens: &[u32],
        cached_tokens: usize,
    ) -> Result<Option<Self>> {
        if !enabled() {
            return Ok(None);
        }
        ensure!(
            policy == QwenSpeculationPolicy::Off,
            "{ENV}=1 requires HF2Q_QWEN_SPECULATION=off so proposals remain shadow-only"
        );
        let mut history = HistoryLookupIndex::new(HistoryLookupConfig {
            min_match: 6,
            max_match: 12,
            max_draft_tokens: 3,
            max_model_len: usize::MAX,
        });
        history.reset(prompt_tokens);
        history.extend_verified(generated_tokens);
        Ok(Some(Self {
            history,
            active: VecDeque::new(),
            turn_type: if cached_tokens == 0 { "cold" } else { "warm" },
        }))
    }

    fn history_proposal(
        &mut self,
        prompt_tokens: &[u32],
        generated_tokens: &[u32],
        remaining: usize,
    ) -> Option<(Vec<u32>, Duration)> {
        let expected = prompt_tokens.len() + generated_tokens.len();
        if self.history.verified_len() < expected {
            let generated_cursor = self
                .history
                .verified_len()
                .checked_sub(prompt_tokens.len())?;
            self.history
                .extend_verified(&generated_tokens[generated_cursor..]);
        }
        if self.history.verified_len() != expected {
            return None;
        }
        let started = Instant::now();
        let mut drafts = self.history.propose();
        let elapsed = started.elapsed();
        drafts.truncate(remaining.min(3));
        (!drafts.is_empty()).then_some((drafts, elapsed))
    }

    fn push(
        &mut self,
        proposer: Proposer,
        drafts: Vec<u32>,
        start_position: usize,
        proposal_cost: Duration,
        transforms: TransformFlags,
    ) {
        self.active.push_back(ActiveProposal {
            proposer,
            drafts,
            matched: 0,
            start_position,
            context_bucket: context_bucket(start_position),
            proposal_cost,
            transforms,
        });
    }

    fn observe(&mut self, slot_id: SlotId, committed: u32) {
        let mut pending = VecDeque::with_capacity(self.active.len());
        while let Some(mut proposal) = self.active.pop_front() {
            let equal = proposal.drafts.get(proposal.matched) == Some(&committed);
            if equal {
                proposal.matched += 1;
            }
            if !equal || proposal.matched == proposal.drafts.len() {
                emit(slot_id, self.turn_type, &proposal, false);
            } else {
                pending.push_back(proposal);
            }
        }
        self.active = pending;
    }

    fn flush_censored(&mut self, slot_id: SlotId) {
        while let Some(proposal) = self.active.pop_front() {
            emit(slot_id, self.turn_type, &proposal, true);
        }
    }
}

fn context_bucket(position: usize) -> &'static str {
    if position < 8_192 {
        "near_2k"
    } else if position < 32_768 {
        "near_16k"
    } else {
        "near_64k"
    }
}

fn emit(slot_id: SlotId, turn_type: &str, proposal: &ActiveProposal, censored: bool) {
    eprintln!(
        "[ADR065_SHADOW_DRAFT_JSON] {}",
        json!({
            "schema": SCHEMA,
            "slot": slot_id.0,
            "proposer": proposal.proposer.label(),
            "turn_type": turn_type,
            "context_bucket": proposal.context_bucket,
            "start_position": proposal.start_position,
            "draft_count": proposal.drafts.len(),
            "accepted_prefix": proposal.matched,
            "proposal_cost_ns": proposal.proposal_cost.as_nanos(),
            "censored": censored,
            "no_grammar": proposal.transforms.no_grammar,
            "no_repetition_penalty": proposal.transforms.no_repetition_penalty,
            "no_forced_thinking": proposal.transforms.no_forced_thinking,
            "no_tool_trigger": proposal.transforms.no_tool_trigger,
            "no_transforms": proposal.transforms.all_clear(),
        })
    );
}

impl Qwen35DecodeState {
    pub(in crate::serve::api) fn shadow_draft_before_joint(
        &mut self,
        qwen: &Qwen35LoadedModel,
        kv_cache: &mut HybridKvCache,
        supervisor: &EngineSupervisor,
    ) -> Result<()> {
        if self.shadow_draft.is_none() {
            return Ok(());
        }
        let remaining = self.max_tokens.saturating_sub(self.generated_tokens.len());
        let start_position = kv_cache.sequence_len_for_slot(self.slot_id)? as usize;
        let transforms = TransformFlags::from_request(self);

        let history = self.shadow_draft.as_mut().and_then(|shadow| {
            shadow.history_proposal(&self.prompt_tokens, &self.generated_tokens, remaining)
        });
        if let Some((drafts, elapsed)) = history {
            self.shadow_draft
                .as_mut()
                .expect("shadow checked above")
                .push(
                    Proposer::History,
                    drafts,
                    start_position,
                    elapsed,
                    transforms,
                );
        }

        if self.mtp.is_some() && remaining >= K3_VERIFY_ROWS {
            let entry = self.k3_entry(kv_cache)?;
            let started = Instant::now();
            let draft = self.k3_draft(qwen, kv_cache, supervisor, entry)?;
            let elapsed = started.elapsed();
            kv_cache.clear_la_capture();
            self.shadow_draft
                .as_mut()
                .expect("shadow checked above")
                .push(
                    Proposer::MtpK3,
                    draft.drafts.to_vec(),
                    start_position,
                    elapsed,
                    transforms,
                );
        }
        Ok(())
    }

    pub(super) fn shadow_observe_committed(&mut self, token: u32) {
        if let Some(shadow) = self.shadow_draft.as_mut() {
            shadow.observe(self.slot_id, token);
        }
    }

    pub(super) fn shadow_flush_censored(&mut self) {
        if let Some(shadow) = self.shadow_draft.as_mut() {
            shadow.flush_censored(self.slot_id);
        }
    }
}

pub(super) fn state_for_request(
    policy: QwenSpeculationPolicy,
    prompt_tokens: &[u32],
    generated_tokens: &[u32],
    cached_tokens: usize,
) -> Result<Option<ShadowDraftState>> {
    ShadowDraftState::new(policy, prompt_tokens, generated_tokens, cached_tokens)
        .context("initialize ADR-065 shadow drafter")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_buckets_have_stable_boundaries() {
        assert_eq!(context_bucket(8_191), "near_2k");
        assert_eq!(context_bucket(8_192), "near_16k");
        assert_eq!(context_bucket(32_767), "near_16k");
        assert_eq!(context_bucket(32_768), "near_64k");
    }

    #[test]
    fn active_proposals_track_overlapping_prefixes() {
        let flags = TransformFlags {
            no_grammar: true,
            no_repetition_penalty: true,
            no_forced_thinking: true,
            no_tool_trigger: true,
        };
        let mut state = ShadowDraftState {
            history: HistoryLookupIndex::new(HistoryLookupConfig {
                min_match: 6,
                max_match: 12,
                max_draft_tokens: 3,
                max_model_len: 100,
            }),
            active: VecDeque::new(),
            turn_type: "warm",
        };
        state.push(Proposer::MtpK3, vec![7, 8, 9], 16, Duration::ZERO, flags);
        state.observe(SlotId(2), 7);
        assert_eq!(state.active.len(), 1);
        assert_eq!(state.active[0].matched, 1);
        state.push(Proposer::History, vec![8, 10], 17, Duration::ZERO, flags);
        state.observe(SlotId(2), 8);
        assert_eq!(state.active.len(), 2);
        state.observe(SlotId(2), 11);
        assert!(state.active.is_empty());
    }
}
