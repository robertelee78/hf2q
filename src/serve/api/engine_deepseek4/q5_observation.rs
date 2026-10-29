//! Snapshot only on the ACTUAL owning thread before mount/after swap-back.
//! Retained bootstrap and every actual session are observed independently.
use super::{Deepseek4LoadedModel, Deepseek4Session};
use crate::serve::{
    q5_deepseek::collector,
    scheduler::{InflightBatchedScheduler, SlotHandle},
};
use anyhow::{ensure, Result};

pub(in crate::serve::api) struct OwnerScope {
    bound: Option<Vec<(SlotHandle, u64)>>,
    cleanup_only: bool,
}
impl OwnerScope {
    pub(in crate::serve::api) fn begin(
        scheduler: &InflightBatchedScheduler,
        handle: SlotHandle,
        model: &Deepseek4LoadedModel,
        sessions: &[Deepseek4Session],
    ) -> Result<Self> {
        Self::begin_many(scheduler, std::slice::from_ref(&handle), model, sessions)
    }
    pub(in crate::serve::api) fn begin_iter(
        scheduler: &InflightBatchedScheduler,
        handles: impl Iterator<Item = SlotHandle>,
        model: &Deepseek4LoadedModel,
        sessions: &[Deepseek4Session],
    ) -> Result<Self> {
        if !collector::active() {
            ensure!(
                model.q5_context.is_none(),
                "Q5 model context is not attached to actual worker"
            );
            return Ok(Self {
                bound: None,
                cleanup_only: false,
            });
        }
        let handles = handles.take(5).collect::<Vec<_>>();
        Self::begin_many(scheduler, &handles, model, sessions)
    }
    pub(in crate::serve::api) fn begin_many(
        scheduler: &InflightBatchedScheduler,
        handles: &[SlotHandle],
        model: &Deepseek4LoadedModel,
        sessions: &[Deepseek4Session],
    ) -> Result<Self> {
        if !collector::active() {
            ensure!(
                model.q5_context.is_none(),
                "Q5 model context is not attached to actual worker"
            );
            return Ok(Self {
                bound: None,
                cleanup_only: false,
            });
        }
        ensure!(
            model
                .q5_context
                .as_ref()
                .is_some_and(|context| context.is_current()),
            "Q5 collector does not belong to actual model"
        );
        ensure!(
            sessions.len() == 4 && !handles.is_empty() && handles.len() <= 4,
            "Q5 selected owner/session cardinality mismatch"
        );
        let bound = handles
            .iter()
            .map(|handle| Ok((*handle, scheduler.q5_request(*handle)?)))
            .collect::<Result<Vec<_>>>()?;
        let keys = bound
            .iter()
            .map(|(h, r)| (*r, h.slot_id.0, h.generation))
            .collect::<Vec<_>>();
        collector::owner_begin_many(&keys)?;
        snapshot(model, sessions)?;
        Ok(Self {
            bound: Some(bound),
            cleanup_only: false,
        })
    }
    pub(in crate::serve::api) fn begin_cleanup(
        scheduler: &InflightBatchedScheduler,
        handle: SlotHandle,
        model: &Deepseek4LoadedModel,
        sessions: &[Deepseek4Session],
    ) -> Result<Self> {
        if !collector::active() {
            ensure!(model.q5_context.is_none(), "Q5 cleanup context detached");
            return Ok(Self {
                bound: None,
                cleanup_only: true,
            });
        }
        ensure!(
            model
                .q5_context
                .as_ref()
                .is_some_and(|context| context.is_current())
                && sessions.len() == 4,
            "Q5 cleanup model/worker identity mismatch"
        );
        let request = cleanup_request(scheduler, handle)?;
        collector::owner_begin_many(&[(request, handle.slot_id.0, handle.generation)])?;
        snapshot(model, sessions)?;
        Ok(Self {
            bound: Some(vec![(handle, request)]),
            cleanup_only: true,
        })
    }
    pub(in crate::serve::api) fn finish(
        self,
        scheduler: &InflightBatchedScheduler,
        model: &Deepseek4LoadedModel,
        sessions: &[Deepseek4Session],
        healthy: bool,
    ) -> Result<()> {
        if let Some(bound) = self.bound {
            ensure!(collector::active(), "Q5 actual owning context disappeared");
            if !healthy {
                return collector::invalidate("actual native/host operation failed");
            }
            for (handle, request) in &bound {
                let current = if self.cleanup_only {
                    cleanup_request(scheduler, *handle)?
                } else {
                    scheduler.q5_request(*handle)?
                };
                ensure!(current == *request, "Q5 stale owning generation/request");
            }
            snapshot(model, sessions)?;
            let keys = bound
                .iter()
                .map(|(h, r)| (*r, h.slot_id.0, h.generation))
                .collect::<Vec<_>>();
            collector::owner_finish_many(&keys)?;
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

/// Explicit metadata-context transfer with the ACTUAL model. Never create a new
/// collector or fall back to a global worker when a context is missing.
pub(in crate::serve::api) fn attach_worker(
    model: &Deepseek4LoadedModel,
) -> Result<Option<collector::Scope>> {
    model
        .q5_context
        .as_ref()
        .map(|context| context.attach())
        .transpose()
}

fn cleanup_request(scheduler: &InflightBatchedScheduler, handle: SlotHandle) -> Result<u64> {
    if let Ok(request) = scheduler.q5_request(handle) {
        return Ok(request);
    }
    scheduler.q5_require_auto_released(handle)?;
    collector::retained_cleanup_request(handle.slot_id.0, handle.generation)
}
