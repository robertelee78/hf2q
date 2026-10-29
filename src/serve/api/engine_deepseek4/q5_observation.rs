//! Snapshot only on the ACTUAL owning thread before mount/after swap-back.
//! Retained bootstrap and every actual session are observed independently.
use super::{Deepseek4LoadedModel, Deepseek4Session};
use crate::serve::{
    q5_deepseek::collector,
    scheduler::{InflightBatchedScheduler, SlotHandle},
};
use anyhow::{ensure, Result};

pub(in crate::serve::api) struct OwnerScope {
    handle: SlotHandle,
    request: Option<u64>,
}
impl OwnerScope {
    pub(in crate::serve::api) fn begin(
        scheduler: &InflightBatchedScheduler,
        handle: SlotHandle,
        model: &Deepseek4LoadedModel,
        sessions: &[Deepseek4Session],
    ) -> Result<Self> {
        if !collector::active() {
            return Ok(Self {
                handle,
                request: None,
            });
        }
        let request = scheduler.q5_request(handle)?;
        ensure!(
            sessions.len() == 4 && (handle.slot_id.0 as usize) < sessions.len(),
            "Q5 selected owner/session cardinality mismatch"
        );
        collector::owner_begin(request, handle.slot_id.0, handle.generation)?;
        snapshot(model, sessions)?;
        Ok(Self {
            handle,
            request: Some(request),
        })
    }
    pub(in crate::serve::api) fn finish(
        self,
        scheduler: &InflightBatchedScheduler,
        model: &Deepseek4LoadedModel,
        sessions: &[Deepseek4Session],
    ) -> Result<()> {
        if let Some(request) = self.request {
            ensure!(
                scheduler.q5_request(self.handle)? == request,
                "Q5 stale owning generation/request"
            );
            snapshot(model, sessions)?;
            collector::owner_finish(request, self.handle.slot_id.0, self.handle.generation)?;
        }
        Ok(())
    }
}
fn snapshot(model: &Deepseek4LoadedModel, sessions: &[Deepseek4Session]) -> Result<()> {
    collector::pools(crate::inference::models::deepseek4::q5_pool_snapshot()?)?;
    visit(
        "loaded",
        &model.cache,
        model.turn_anchor.as_ref(),
        model.pending_turn_anchor.as_ref(),
        model.live_logits.as_ref(),
    )?;
    for (index, session) in sessions.iter().enumerate() {
        visit(
            &format!("retained_slot/{index}"),
            &session.cache,
            session.turn_anchor.as_ref(),
            session.pending_turn_anchor.as_ref(),
            session.live_logits.as_ref(),
        )?;
    }
    Ok(())
}
fn visit(
    owner: &str,
    cache: &crate::inference::models::deepseek4::cache::Deepseek4Cache,
    committed: Option<&crate::inference::models::deepseek4::cache::Deepseek4CacheSnapshot>,
    pending: Option<&crate::inference::models::deepseek4::cache::Deepseek4CacheSnapshot>,
    logits: Option<&mlx_native::MlxBuffer>,
) -> Result<()> {
    cache.q5_visit(&mut |role, buffer| collector::field(&format!("{owner}/{role}"), buffer))?;
    for (kind, anchor) in [("committed", committed), ("pending", pending)] {
        if let Some(anchor) = anchor {
            anchor.q5_visit(&mut |role, buffer| {
                collector::field(&format!("{owner}/{kind}/{role}"), buffer)
            })?;
        } else {
            collector::field(&format!("{owner}/{kind}"), None)?;
        }
    }
    collector::field(&format!("{owner}/live_logits"), logits)
}
