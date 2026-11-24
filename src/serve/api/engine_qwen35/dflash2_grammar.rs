//! Bounded proposal-only grammar work for an in-flight DFlash2 target window.
//!
//! These are post-accept, pre-router-trigger snapshots, not token decisions.
//! The canonical sampler, decoder, tool router and thinking budget still run
//! normally. Only a matching canonical prefix can move a snapshot into place.

use super::{Qwen35DecodeState, Qwen35LoadedModel, ToolCallEvent};
use crate::serve::api::grammar::GrammarRuntime;
use anyhow::{Context, Result};

const MAX_PROPOSALS: usize = 7;

/// Final disposition is recorded at the actual hypothetical-planning consumer,
/// not inferred from callback invocation or the target's accepted-row count.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum DFlash2GrammarOutcome {
    Invalid,
    Prepared,
    PlanningConsumed,
    DivergenceDiscarded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(super) struct DFlash2GrammarAttempt {
    pub(super) awaiting_trigger: bool,
    pub(super) eos: bool,
    pub(super) outcome: DFlash2GrammarOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(super) struct DFlash2GrammarWork {
    pub(super) actual_proposals: usize,
    pub(super) awaiting_trigger_at_start: bool,
    pub(super) attempts: [Option<DFlash2GrammarAttempt>; MAX_PROPOSALS],
}

/// Close the exact attempted-proposal extent on invalid/EOS breaks and errors.
/// Created only for an enabled timeline; no clock reads on the ordinary path.
struct ProposalTiming;

impl ProposalTiming {
    fn begin() -> Self {
        mlx_native::command_timeline::mark("grammar.proposal_begin");
        Self
    }
}

impl Drop for ProposalTiming {
    fn drop(&mut self) {
        mlx_native::command_timeline::mark("grammar.proposal_end");
    }
}

#[derive(Clone)]
struct PreparedTransition {
    token: u32,
    runtime: GrammarRuntime,
}

/// One lane's hypothetical proposal path, excluding the already observed anchor.
/// Cloning is reserved for the explicit RouteCheck double-plan diagnostic.
#[derive(Clone)]
pub(in crate::serve::api) struct PreparedDFlash2Grammar {
    transitions: [Option<PreparedTransition>; MAX_PROPOSALS],
    next: usize,
    generated_len: usize,
    work: Option<DFlash2GrammarWork>,
}

impl PreparedDFlash2Grammar {
    pub(super) fn work(&self) -> Option<DFlash2GrammarWork> {
        self.work
    }

    pub(super) fn records_work(&self) -> bool {
        self.work.is_some()
    }

    pub(super) fn omit_work_recording(&mut self) {
        self.work = None;
    }

    pub(super) fn take_transition(
        &mut self,
        token: u32,
        generated_len: usize,
    ) -> Option<GrammarRuntime> {
        let matches = generated_len == self.generated_len
            && self.transitions.get(self.next).and_then(Option::as_ref)
                .is_some_and(|transition| transition.token == token);
        if !matches {
            // Never reuse a later coincidentally equal token after divergence.
            if let Some(work) = self.work.as_mut() {
                for attempt in work.attempts.iter_mut().flatten() {
                    if attempt.outcome == DFlash2GrammarOutcome::Prepared {
                        attempt.outcome = DFlash2GrammarOutcome::DivergenceDiscarded;
                    }
                }
            }
            self.transitions.fill_with(|| None);
            self.next = MAX_PROPOSALS;
            return None;
        }
        let transition = self.transitions[self.next].take()?;
        if let Some(attempt) = self.work.as_mut()
            .and_then(|work| work.attempts[self.next].as_mut())
        {
            attempt.outcome = DFlash2GrammarOutcome::PlanningConsumed;
        }
        self.next += 1;
        self.generated_len += 1;
        Some(transition.runtime)
    }
}

impl Qwen35DecodeState {
    /// Simulate at most seven actual proposals without sampling or touching the
    /// output ledger, repetition history, RNG, thinking budget or real router.
    /// The caller must establish producer completion before supplying drafts.
    pub(in crate::serve::api) fn prepare_dflash2_grammar(
        &self,
        qwen: &Qwen35LoadedModel,
        drafts: &[u32],
        collect_work: bool,
    ) -> Result<Option<PreparedDFlash2Grammar>> {
        let Some(runtime) = self.grammar_runtime.as_ref() else {
            // In particular, do not clone decoder/router state on this path.
            return Ok(None);
        };
        anyhow::ensure!(drafts.len() <= MAX_PROPOSALS,
            "DFlash2 grammar projection exceeds seven proposals");
        let table = self.params.token_bytes.as_deref()
            .context("DFlash2 grammar projection requires authoritative token bytes")?;
        anyhow::ensure!(table.len() == qwen.vocab_size,
            "DFlash2 grammar token-byte table length {} != vocabulary {}",
            table.len(), qwen.vocab_size);
        let mut prepared = PreparedDFlash2Grammar {
            transitions: std::array::from_fn(|_| None),
            next: 0,
            generated_len: self.generated_tokens.len(),
            work: collect_work.then(|| DFlash2GrammarWork {
                actual_proposals: drafts.len(),
                awaiting_trigger_at_start: runtime.is_awaiting_trigger(),
                attempts: [None; MAX_PROPOSALS],
            }),
        };
        if drafts.is_empty() {
            return Ok(Some(prepared));
        }
        let mut runtime = runtime.clone();
        let mut observer = None;
        for (index, &token) in drafts.iter().enumerate() {
            let _timing = collect_work.then(ProposalTiming::begin);
            anyhow::ensure!((token as usize) < table.len(),
                "DFlash2 grammar proposal {token} is outside the authoritative vocabulary");
            let eos = qwen.eos_token_ids.contains(&token);
            if let Some(work) = prepared.work.as_mut() {
                work.attempts[index] = Some(DFlash2GrammarAttempt {
                    awaiting_trigger: runtime.is_awaiting_trigger(),
                    eos,
                    outcome: DFlash2GrammarOutcome::Invalid,
                });
            }
            let accepted = if eos {
                runtime.accept_eog()
            } else {
                let bytes = &table[token as usize];
                !bytes.is_empty() && bytes.first() != Some(&0)
                    && runtime.accept_token(token, bytes)
            };
            if !accepted {
                // Grammar-invalid proposals are ordinary drafter misses. The
                // valid prefix remains useful; no rejected state is published.
                break;
            }
            if let Some(attempt) = prepared.work.as_mut()
                .and_then(|work| work.attempts[index].as_mut())
            {
                attempt.outcome = DFlash2GrammarOutcome::Prepared;
            }
            if eos || index + 1 == drafts.len() {
                prepared.transitions[index] = Some(PreparedTransition { token, runtime });
                break;
            }
            prepared.transitions[index] = Some(PreparedTransition {
                token,
                runtime: runtime.clone(),
            });
            // Match select_and_observe_from: accept first, then decode and
            // detect the tool marker. The saved state is deliberately BEFORE
            // this trigger, which real observation will apply exactly once.
            if index == 0 {
                observer = self.tool_splitter.as_ref()
                    .map(|splitter| (splitter.clone(), self.text_decoder.clone()));
            }
            if let Some((splitter, decoder)) = observer.as_mut() {
                let fragment = decoder.push(&qwen.tokenizer, token);
                if splitter.feed(&fragment).iter()
                    .any(|event| matches!(event, ToolCallEvent::ToolCallOpen)) {
                    runtime.trigger();
                }
            }
        }
        Ok(Some(prepared))
    }
}

/// A persistent ordinary control for one deterministic diagnostic window.
/// Its state is never refreshed from the projected path after the first row.
pub(super) struct OrdinaryDFlash2Check {
    semantic: super::Qwen35SpecSemanticState,
    checked_rows: usize,
}

impl OrdinaryDFlash2Check {
    pub(super) fn begin(
        prepared: Option<&PreparedDFlash2Grammar>,
        semantic: &super::Qwen35SpecSemanticState,
        params: &super::SamplingParams,
    ) -> Option<Self> {
        if !prepared.is_some_and(PreparedDFlash2Grammar::records_work)
            || !(f64::from(params.temperature) < crate::serve::sampler_pure::SAMPLING_EPS)
        {
            return None;
        }
        Some(Self {
            semantic: semantic.clone(),
            checked_rows: 0,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.checked_rows
    }
}

impl super::Qwen35SpecSemanticState {
    pub(super) fn select_and_observe_with_ordinary_check(
        &mut self,
        qwen: &Qwen35LoadedModel,
        params: &super::SamplingParams,
        source: super::Qwen35DecisionSource<'_>,
        prepared: Option<&mut PreparedDFlash2Grammar>,
        ordinary: Option<&mut OrdinaryDFlash2Check>,
    ) -> Result<super::Qwen35CanonicalDecision> {
        let Some(ordinary) = ordinary else {
            return self.select_and_observe_from(qwen, params, source, prepared);
        };
        // Both samplers need the same pristine row. No copy on the disabled path.
        let mut copied_logits;
        let ordinary_source = match &source {
            super::Qwen35DecisionSource::Argmax(token) =>
                super::Qwen35DecisionSource::Argmax(*token),
            super::Qwen35DecisionSource::Logits(logits) => {
                copied_logits = logits.to_vec();
                super::Qwen35DecisionSource::Logits(&mut copied_logits)
            }
        };
        let generated_len = self.generated_tokens.len();
        let projected = self.select_and_observe_from(qwen, params, source, prepared)?;
        let expected = ordinary.semantic.select_and_observe_from(
            qwen, params, ordinary_source, None,
        )?;
        anyhow::ensure!(
            projected == expected
                && self.generated_tokens == ordinary.semantic.generated_tokens
                && self.sampling_history == ordinary.semantic.sampling_history,
            "DFlash2 prepared grammar diverged from ordinary planning at {generated_len}: \
             projected {projected:?}, ordinary {expected:?}"
        );
        ordinary.checked_rows += 1;
        Ok(projected)
    }
}

#[cfg(test)]
#[path = "dflash2_grammar_tests.rs"]
mod tests;
