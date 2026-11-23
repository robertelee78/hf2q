//! Input authority is frozen by never-restored execution before the candidate.
//! Descriptors constrain replay boundaries; they NEVER manufacture Q1 expected
//! extents or stand in for the raw numeric/cache stores used by the final arms.
use super::*;

#[derive(Serialize, PartialEq, Eq)]
struct Transaction {
    slot: u32,
    start: usize,
    input: Vec<u32>,
    route: String,
    logit_rows: usize,
    vocabulary: usize,
    hidden_rows: usize,
    hidden_width: usize,
}
#[derive(Serialize, PartialEq, Eq)]
pub(super) struct Signature {
    transactions: Vec<Transaction>,
    decisions: Vec<(usize, u32)>,
    completed_and_checked_routes: Vec<&'static str>,
}
impl Signature {
    pub(super) fn read(scope: &trace::Scope) -> Result<Self> {
        scope.inspect(|observed| Self {
            transactions: observed.forwards.iter().map(|forward| Transaction {
                slot: forward.slot, start: forward.start, input: forward.input.clone(), route: forward.route.clone(),
                logit_rows: forward.logits.len() / forward.vocab, vocabulary: forward.vocab,
                hidden_rows: if forward.hidden_width == 0 { 0 } else { forward.hidden.len() / forward.hidden_width },
                hidden_width: forward.hidden_width,
            }).collect(),
            decisions: observed.decisions.clone(),
            completed_and_checked_routes: observed.route_events.clone(),
        })
    }
}
#[derive(Serialize)]
pub(in super::super) struct Turn {
    pub(super) prompt: Vec<u32>,
    pub(super) prefill: Signature,
    pub(super) complete: Signature,
    pub(super) tokens: Vec<u32>,
    pub(super) retained: Vec<u32>,
}
impl Turn {
    pub(super) fn check_prefill(&self, scope: &trace::Scope) -> Result<()> {
        ensure!(self.prefill == Signature::read(scope)?,
            "DivergedExactInvariant: independent prefix replay changed prefill chunks/seed decisions");
        Ok(())
    }
    pub(super) fn check_complete(&self, scope: &trace::Scope, tokens: &[u32], retained: &[u32]) -> Result<()> {
        ensure!(self.complete == Signature::read(scope)? && self.tokens == tokens && self.retained == retained,
            "DivergedExactInvariant: prefix replay changed actual chunks/K3 steps/terminal tokens/retained ledger");
        Ok(())
    }
}
