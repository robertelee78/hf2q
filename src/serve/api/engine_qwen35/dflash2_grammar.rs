//! Bounded proposal-only grammar work for an in-flight DFlash2 target window.
//!
//! These are post-accept, pre-router-trigger snapshots, not token decisions.
//! The canonical sampler, decoder, tool router and thinking budget still run
//! normally. Only a matching canonical prefix can move a snapshot into place.

use super::{Qwen35DecodeState, Qwen35LoadedModel, ToolCallEvent};
use crate::serve::api::grammar::GrammarRuntime;
use anyhow::{Context, Result};

const MAX_PROPOSALS: usize = 7;

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
}

impl PreparedDFlash2Grammar {
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
            self.transitions.fill_with(|| None);
            self.next = MAX_PROPOSALS;
            return None;
        }
        let transition = self.transitions[self.next].take()?;
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
        };
        if drafts.is_empty() {
            return Ok(Some(prepared));
        }
        let mut runtime = runtime.clone();
        let mut observer = None;
        for (index, &token) in drafts.iter().enumerate() {
            anyhow::ensure!((token as usize) < table.len(),
                "DFlash2 grammar proposal {token} is outside the authoritative vocabulary");
            let eos = qwen.eos_token_ids.contains(&token);
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

#[cfg(test)]
#[path = "dflash2_grammar_tests.rs"]
mod tests;
