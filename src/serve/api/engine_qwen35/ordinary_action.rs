//! Read-only ordinary-leaf preparation, not cohort/cache admission.
//!
//! The scalar producer remains authoritative. Installed history lookup may
//! temporarily recurse into ordinary MTP warmup. Unknown lookup decisions must
//! not be labelled ordinary merely from Option state.
use crate::serve::{multi_seq_kv::SlotId, scheduler::SlotHandle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dispatch { Finished, EmitVerified, History, Mtp, Ordinary }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrdinaryLeaf {
    NoTarget,
    FullLogits,
    GpuGreedy,
    /// Full logits AND actual post-output-RMSNorm hidden, followed by existing
    /// request-owned MTP catchup/carry/semantic completion.
    CoherentMtpFullLogits,
    /// Real lookup synchronization/proposal must decide this branch first.
    HistoryDecisionRequired,
    MtpProposal,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NextAction {
    pub dispatch: Dispatch,
    pub ordinary: OrdinaryLeaf,
}
#[derive(Clone, Copy)]
pub(super) struct Facts {
    pub step: usize,
    pub max_tokens: usize,
    pub generated_tokens: usize,
    pub pending_outputs: usize,
    pub history_present: bool,
    pub history_may_speculate: bool,
    pub mtp_present: bool,
    pub mtp_may_speculate: bool,
    pub greedy_without_logprobs: bool,
}
fn without_history(f: Facts) -> OrdinaryLeaf {
    if f.mtp_present {
        if f.max_tokens.saturating_sub(f.generated_tokens) < 4 || !f.mtp_may_speculate {
            OrdinaryLeaf::CoherentMtpFullLogits
        } else { OrdinaryLeaf::MtpProposal }
    } else if f.greedy_without_logprobs { OrdinaryLeaf::GpuGreedy }
    else { OrdinaryLeaf::FullLogits }
}
pub(super) fn classify(f: Facts) -> NextAction {
    if f.step >= f.max_tokens {
        return NextAction { dispatch: Dispatch::Finished, ordinary: OrdinaryLeaf::NoTarget };
    }
    if f.pending_outputs > 0 {
        return NextAction { dispatch: Dispatch::EmitVerified, ordinary: OrdinaryLeaf::NoTarget };
    }
    if f.history_present {
        return NextAction { dispatch: Dispatch::History,
            ordinary: if f.history_may_speculate { OrdinaryLeaf::HistoryDecisionRequired }
                else { without_history(f) } };
    }
    NextAction { dispatch: if f.mtp_present { Dispatch::Mtp } else { Dispatch::Ordinary },
        ordinary: without_history(f) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OwnerPreview {
    RefusedOwner,
    Cancelled,
    KnownLeaf(OrdinaryLeaf),
    NotKnownOrdinary,
}
impl OwnerPreview {
    pub(crate) fn known_leaf(self) -> Option<OrdinaryLeaf> {
        match self { Self::KnownLeaf(leaf) => Some(leaf), _ => None }
    }
}
/// KnownLeaf is NOT GPU admission: actual cursor/profile/position/geometry and
/// fresh cancellation still require owning-worker preflight before shared work.
pub(crate) fn preview(
    action: NextAction, state_slot: SlotId, scheduled: SlotHandle,
    installed: SlotHandle, client_closed: bool,
) -> OwnerPreview {
    if scheduled != installed || state_slot != scheduled.slot_id { return OwnerPreview::RefusedOwner; }
    if client_closed { return OwnerPreview::Cancelled; }
    match action.ordinary {
        leaf @ (OrdinaryLeaf::FullLogits | OrdinaryLeaf::GpuGreedy | OrdinaryLeaf::CoherentMtpFullLogits) => OwnerPreview::KnownLeaf(leaf),
        _ => OwnerPreview::NotKnownOrdinary,
    }
}

#[cfg(test)]
#[path = "ordinary_action_controls.rs"]
mod controls;
