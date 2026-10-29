//! Snapshot only on the ACTUAL owning thread before mount/after swap-back.
//! Retained bootstrap and every actual session are observed independently.
use super::{Deepseek4LoadedModel, Deepseek4Session};
use crate::serve::{
    q5_deepseek::collector,
    scheduler::{InflightBatchedScheduler, SlotHandle},
};
use anyhow::{ensure, Context, Result};

pub(in crate::serve::api) struct OwnerScope {
    bound: Option<Vec<(SlotHandle, u64)>>,
    cleanup_only: bool,
    origin: Option<collector::Handle>,
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
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
                origin: None,
                _thread: std::marker::PhantomData,
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
                origin: None,
                _thread: std::marker::PhantomData,
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
            origin: model.q5_context.clone(),
            _thread: std::marker::PhantomData,
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
                origin: None,
                _thread: std::marker::PhantomData,
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
            origin: model.q5_context.clone(),
            _thread: std::marker::PhantomData,
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
            self.origin
                .as_ref()
                .context("Q5 operation lost origin context")?
                .require_current_model(model.q5_context.as_ref())?;
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
    collector::source_unchanged()?;
    collector::snapshot_begin(model.model.ctx.device().registry_id())?;
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
    collector::snapshot_end()
}
fn visit(
    owner: &str,
    cache: &crate::inference::models::deepseek4::cache::Deepseek4Cache,
    committed: Option<&crate::inference::models::deepseek4::cache::Deepseek4CacheSnapshot>,
    pending: Option<&crate::inference::models::deepseek4::cache::Deepseek4CacheSnapshot>,
    logits: Option<&mlx_native::MlxBuffer>,
) -> Result<()> {
    collector::cache_state(owner, cache.q5_state()?)?;
    cache.q5_visit(&mut |role, buffer| collector::field(&format!("{owner}/{role}"), buffer))?;
    for (kind, anchor) in [("committed", committed), ("pending", pending)] {
        if let Some(anchor) = anchor {
            collector::cache_state(&format!("{owner}/{kind}"), anchor.q5_state()?)?;
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

/// Stage all real caches while attached to SAME model metadata context. No
/// native/physical allowance is deducted for bootstrap transfer or pool clearing.
pub(in crate::serve::api) fn provision(
    model: &mut Deepseek4LoadedModel,
    max_slots: u32,
) -> Result<()> {
    let Some(context) = model.q5_context.clone() else {
        return super::slot_provision::provision(model, max_slots);
    };
    ensure!(
        max_slots == 4,
        "Q5 selected provision requires four actual logical sessions"
    );
    let _attached = if context.is_current() {
        None
    } else {
        Some(context.attach()?)
    };
    collector::lifecycle_begin(collector::Lifecycle::Provision)?;
    snapshot(model, model.slot_sessions.as_deref().unwrap_or(&[]))?;
    let result = super::slot_provision::provision(model, max_slots);
    if result.is_ok() {
        snapshot(model, model.slot_sessions.as_deref().unwrap_or(&[]))?;
    }
    let observed = collector::lifecycle_finish(collector::Lifecycle::Provision, result.is_ok());
    combine_lifecycle(result, observed)
}

pub(in crate::serve::api) fn warmup(
    model: &mut Deepseek4LoadedModel,
    sessions: &[Deepseek4Session],
    quiescent: bool,
    supervisor: &crate::serve::api::engine_supervisor::EngineSupervisor,
) -> Result<()> {
    if model.q5_context.is_none() {
        return model.warmup(supervisor);
    }
    ensure!(
        quiescent
            && sessions.len() == 4
            && model
                .q5_context
                .as_ref()
                .is_some_and(|context| context.is_current()),
        "Q5 warmup not quiescent/model-bound"
    );
    ensure!(
        std::env::var("HF2Q_DEEPSEEK_SKIP_WARMUP").as_deref() != Ok("1"),
        "Q5 selected warmup cannot be skipped"
    );
    collector::lifecycle_begin(collector::Lifecycle::Warmup)?;
    snapshot(model, sessions)?;
    let result = model.warmup(supervisor);
    if result.is_ok() {
        snapshot(model, sessions)?;
    }
    let observed = collector::lifecycle_finish(collector::Lifecycle::Warmup, result.is_ok());
    combine_lifecycle(result, observed)
}

fn combine_lifecycle(actual: Result<()>, observed: Result<()>) -> Result<()> {
    match (actual, observed) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(actual), Err(observed)) => Err(actual).map_err(|error| {
            error.context(format!(
                "Q5 lifecycle observation also failed: {observed:#}"
            ))
        }),
    }
}

/// Preserve the real successful-promotion versus failed-delivery restore branch.
/// Retired lookup is cleanup-only and can never authorize another forward.
pub(in crate::serve::api) fn finalize_session(
    model: &Deepseek4LoadedModel,
    sessions: &mut [Deepseek4Session],
    scheduler: &InflightBatchedScheduler,
    handle: SlotHandle,
    delivered: bool,
) -> Result<()> {
    let observed = OwnerScope::begin_cleanup(scheduler, handle, model, sessions)?;
    let result = if delivered {
        sessions[handle.slot_id.0 as usize].commit_request_anchor();
        Ok(())
    } else {
        sessions[handle.slot_id.0 as usize].recover_after_cancellation()
    };
    let observation = observed.finish(scheduler, model, sessions, result.is_ok());
    combine_lifecycle(result, observation)
}

pub(in crate::serve::api) fn terminal(
    model: &Deepseek4LoadedModel,
    sessions: &[Deepseek4Session],
) -> Result<()> {
    let Some(context) = model.q5_context.as_ref() else {
        return Ok(());
    };
    context.require_current_model(model.q5_context.as_ref())?;
    // Failure may leave incomplete metadata. Emit it as incomplete; never force
    // a wait/reset/drop or turn it into a successful lifecycle.
    let _ = snapshot(model, sessions);
    context.emit_terminal()
}

/// AFTER actual swap; explicitly a partial pair, not all retained slots/rows.
pub(super) fn mounted_pair(session: &Deepseek4Session, model: &Deepseek4LoadedModel) {
    if model.q5_context.is_none() {
        return;
    }
    let observed = (|| -> Result<()> {
        model
            .q5_context
            .as_ref()
            .context("Q5 mounted model context absent")?
            .require_current_model(model.q5_context.as_ref())?;
        collector::source_unchanged()?;
        collector::mounted_begin(model.model.ctx.device().registry_id())?;
        visit(
            "swap/loaded",
            &model.cache,
            model.turn_anchor.as_ref(),
            model.pending_turn_anchor.as_ref(),
            model.live_logits.as_ref(),
        )?;
        visit(
            "swap/selected_session",
            &session.cache,
            session.turn_anchor.as_ref(),
            session.pending_turn_anchor.as_ref(),
            session.live_logits.as_ref(),
        )?;
        collector::mounted_end()
    })();
    if observed.is_err() {
        let _ = collector::invalidate("actual mounted pair observation failed");
    }
}
