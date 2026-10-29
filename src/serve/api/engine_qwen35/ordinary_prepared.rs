//! Owning host continuations, extracted from actual AUTO lookup decisions.
//! No re-proposal, forced policy, sampled-token replay or cloned state machine.
use super::*;
use super::ordinary_forward::{PreparedCoherentOrdinaryTarget,CoherentOrdinaryTargetResult};
use super::ordinary_plain::{PreparedPlainTarget,PlainTargetResult};

pub(in crate::serve::api) struct HistoryContinuation {
    lookup: HistoryLookupIndex,
    extend_after: Option<usize>,
}
impl HistoryContinuation {
    pub(in crate::serve::api) fn restore(mut self,state:&mut Qwen35DecodeState) {
        if let Some(before)=self.extend_after {
            if state.generated_tokens.len()>before {
                self.lookup.extend_verified(&state.generated_tokens[before..]);
            }
        }
        state.history_lookup=Some(self.lookup);
    }
}
pub(in crate::serve::api) enum HistoryDecision {
    Ordinary(HistoryContinuation),
    Proposals(Vec<u32>),
}
pub(in crate::serve::api) enum PreparedKind {
    Plain(PreparedPlainTarget), Coherent(PreparedCoherentOrdinaryTarget),
}
pub(in crate::serve::api) struct PreparedOrdinaryTick {
    kind: PreparedKind,
    history: Option<HistoryContinuation>,
    cost: super::ordinary_cost::OrdinaryCost,
    own_prepare: Duration,
}
impl PreparedOrdinaryTick {
    pub(in crate::serve::api) fn input(&self)->(u32,i32,SlotId) {
        match &self.kind { PreparedKind::Plain(p)=>p.input(),PreparedKind::Coherent(p)=>p.input() }
    }
    pub(in crate::serve::api) fn can_share(&self)->bool {
        match &self.kind { PreparedKind::Coherent(_)=>true,PreparedKind::Plain(p)=>!p.is_greedy() }
    }
    pub(in crate::serve::api) fn measured_target(&mut self,raw:Duration,width:usize)->Result<()> {
        self.cost=super::ordinary_cost::OrdinaryCost::measured(raw,width,self.own_prepare)?;
        Ok(())
    }
    /// Only used for a refused member BEFORE any shared mutation. Open peers
    /// retain their prepared action and use the prepared scalar producer.
    pub(in crate::serve::api) fn discard(self,state:&mut Qwen35DecodeState) {
        if let Some(h)=self.history { h.restore(state); }
    }
}
pub(in crate::serve::api) enum PreparedTargetResult {
    Plain(PlainTargetResult),Coherent(CoherentOrdinaryTargetResult),
}
pub(in crate::serve::api) enum Preparation {
    Ordinary(PreparedOrdinaryTick),HistoryProposals(Vec<u32>),Unsupported,
}

impl Qwen35DecodeState {
    pub(in crate::serve::api) fn prepare_history_decision(&mut self)->Result<HistoryDecision> {
        if !self.history_cost.may_speculate() {
            let lookup=self.history_lookup.take().expect("history lookup checked above");
            return Ok(HistoryDecision::Ordinary(HistoryContinuation {lookup,extend_after:None}));
        }
        let expected_len=self.prompt_tokens.len()+self.generated_tokens.len();
        let lookup=self.history_lookup.as_mut().expect("history lookup checked above");
        if lookup.verified_len()<expected_len {
            let cursor=lookup.verified_len().checked_sub(self.prompt_tokens.len())
                .context("Qwen history lookup precedes prompt boundary")?;
            lookup.extend_verified(&self.generated_tokens[cursor..]);
        }
        anyhow::ensure!(lookup.verified_len()==expected_len,"Qwen history lookup ledger cursor mismatch");
        let remaining=self.max_tokens.saturating_sub(self.generated_tokens.len());
        let mut drafts=lookup.propose();
        drafts.truncate(remaining.saturating_sub(1).min(3));
        if !drafts.is_empty() { return Ok(HistoryDecision::Proposals(drafts)); }
        super::super::qwen35_speculation::record_history_lookup_no_match();
        let lookup=self.history_lookup.take().expect("history lookup checked above");
        if self.mtp.is_some() && !may_route_history_miss_to_mtp(true,&self.mtp_cost) {
            self.mtp=None; // EXISTING cost-disabled proposer drop, not forced OFF.
        }
        Ok(HistoryDecision::Ordinary(HistoryContinuation {
            lookup,extend_after:Some(self.generated_tokens.len()),
        }))
    }
    pub(in crate::serve::api) fn shared_ordinary_semantics_supported(&self)->bool {
        self.params.vision_fingerprint.is_none()
    }
    pub(in crate::serve::api) fn prepare_shared_ordinary_tick(&mut self,kv:&HybridKvCache)->Result<Preparation> {
        let action=self.next_decode_action();
        if matches!(action.dispatch,ordinary_action::Dispatch::Finished|ordinary_action::Dispatch::EmitVerified) {
            return Ok(Preparation::Unsupported);
        }
        let history=if action.dispatch==ordinary_action::Dispatch::History {
            match self.prepare_history_decision()? {
                HistoryDecision::Proposals(d)=>return Ok(Preparation::HistoryProposals(d)),
                HistoryDecision::Ordinary(h)=>Some(h),
            }
        } else { None };
        // History is now temporarily owned by the continuation, exactly as in
        // the original recursive ordinary branch. No second lookup decision.
        let leaf=self.next_decode_action();
        let plain_started=Instant::now();
        let kind=match leaf.ordinary {
            ordinary_action::OrdinaryLeaf::CoherentMtpFullLogits=>
                self.prepare_coherent_ordinary_target(kv).map(PreparedKind::Coherent),
            ordinary_action::OrdinaryLeaf::FullLogits|ordinary_action::OrdinaryLeaf::GpuGreedy=>
                self.prepare_plain_target(kv,leaf.ordinary==ordinary_action::OrdinaryLeaf::GpuGreedy).map(PreparedKind::Plain),
            _=>{
                if let Some(h)=history {h.restore(self);}
                return Ok(Preparation::Unsupported);
            }
        };
        match kind {
            Ok(kind)=>{
                let (started,own_prepare)=match &kind {
                    PreparedKind::Coherent(_)=>(Instant::now(),Duration::ZERO),
                    PreparedKind::Plain(_)=>(plain_started,plain_started.elapsed()),
                };
                Ok(Preparation::Ordinary(PreparedOrdinaryTick {kind,history,
                    cost:super::ordinary_cost::OrdinaryCost::OriginalScalar(started),own_prepare}))
            },
            Err(e)=>{if let Some(h)=history {h.restore(self);}Err(e)}
        }
    }
    pub(in crate::serve::api) fn produce_prepared_scalar(&mut self,q:&mut Qwen35LoadedModel,
        kv:&mut HybridKvCache,s:&EngineSupervisor,p:&PreparedOrdinaryTick)->Result<PreparedTargetResult> {
        match &p.kind {
            PreparedKind::Plain(v)=>self.produce_scalar_plain_target(q,kv,s,v).map(PreparedTargetResult::Plain),
            PreparedKind::Coherent(v)=>self.produce_scalar_coherent_target(q,kv,s,v).map(PreparedTargetResult::Coherent),
        }
    }
    pub(in crate::serve::api) fn supplied_ordinary_result(p:&PreparedOrdinaryTick,logits:Vec<f32>,hidden:MlxBuffer)->PreparedTargetResult {
        match &p.kind {
            PreparedKind::Plain(_)=>PreparedTargetResult::Plain(PlainTargetResult::FullLogits(logits)),
            PreparedKind::Coherent(_)=>PreparedTargetResult::Coherent(CoherentOrdinaryTargetResult::from_actual_forward(logits,hidden)),
        }
    }
    pub(in crate::serve::api) fn validate_prepared_ordinary(&self,q:&Qwen35LoadedModel,kv:&HybridKvCache,
        p:&PreparedOrdinaryTick,r:&PreparedTargetResult)->Result<()> {
        match (&p.kind,r) {
            (PreparedKind::Plain(p),PreparedTargetResult::Plain(r))=>self.validate_plain_result(kv,p,r,q.vocab_size),
            (PreparedKind::Coherent(p),PreparedTargetResult::Coherent(r))=>self.validate_coherent_ordinary_result(q,kv,p,r),
            _=>anyhow::bail!("ordinary prepared/result kind mismatch"),
        }
    }
    pub(in crate::serve::api) fn finish_prepared_ordinary(&mut self,q:&mut Qwen35LoadedModel,
        kv:&mut HybridKvCache,s:&EngineSupervisor,p:PreparedOrdinaryTick,r:PreparedTargetResult)->Result<Qwen35TickOutcome> {
        let outcome=match (p.kind,r) {
            (PreparedKind::Plain(pv),PreparedTargetResult::Plain(rv))=>self.finish_plain_with_cost(q,kv,pv,rv,p.cost),
            (PreparedKind::Coherent(pv),PreparedTargetResult::Coherent(rv))=>self.finish_coherent_ordinary_with_cost(q,kv,s,pv,rv,p.cost),
            _=>Err(anyhow::anyhow!("ordinary prepared/result kind mismatch")),
        };
        if let Some(h)=p.history {h.restore(self);}
        outcome
    }
}

#[cfg(test)]
#[path="ordinary_prepared_controls.rs"]
pub(in crate::serve::api) mod controls;
