//! Non-draining adapters at ACTUAL family terminal boundaries. The packet
//! selects explicit steps/layers/stages; unobserved intermediate coordinates
//! remain missing, never inferred from final hidden-state agreement.
use super::{
    capture,
    receipt::{CaptureTarget, Family, Stage, Trace},
    store::Store,
};
use mlx_native::MlxBuffer;
use std::{cell::RefCell, collections::BTreeMap};
struct Active {
    scope_id: u64,
    family: Family,
    store: Store,
    targets: BTreeMap<(u64, u32, Stage), CaptureTarget>,
    pending: Vec<(u64, u32, CaptureTarget, MlxBuffer)>,
    current: Option<(u64, u32)>,
}
thread_local! {static ACTIVE:RefCell<Option<Active>>=const{RefCell::new(None)};}
/// Same-thread lease: unwinding removes all capture buffers/TLS state.
static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
#[must_use]
pub struct Scope {
    id: u64,
    _same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.as_ref().is_some_and(|a| a.scope_id == self.id) {
                slot.take();
            }
        });
    }
}
pub fn install(
    family: Family,
    store: Store,
    targets: BTreeMap<(u64, u32, Stage), CaptureTarget>,
) -> Result<Scope, String> {
    if targets.is_empty() || targets.len() > 4096 {
        return Err("empty family diagnostic packet".into());
    }
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err("nested family diagnostic".into());
        }
        let id = NEXT_SCOPE
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |v| v.checked_add(1),
            )
            .map_err(|_| "capture scope identity overflow")?;
        *slot = Some(Active {
            scope_id: id,
            family,
            store,
            targets,
            pending: Vec::new(),
            current: None,
        });
        Ok(Scope {
            id,
            _same_thread: std::marker::PhantomData,
        })
    })
}
pub fn selected(family: Family, step: u64, layer: u32, stage: Stage) -> bool {
    ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|a| a.family == family && a.targets.contains_key(&(step, layer, stage)))
    })
}
/// Must be invoked by the concrete source path AFTER its own checked drain,
/// before any pool recycle/host cursor publication at that boundary.
pub fn drained(
    family: Family,
    step: u64,
    layer: u32,
    stage: Stage,
    buf: &MlxBuffer,
    drain: &str,
) -> Result<(), String> {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(active) = slot.as_mut() else {
            return Ok(());
        };
        if active.family != family {
            return Ok(());
        }
        let Some(target) = active.targets.get(&(step, layer, stage)).cloned() else {
            return Ok(());
        };
        let shape = buf.shape().to_vec();
        let result = if target.normalize_rows {
            capture::rows_after_checked_drain(&mut active.store, target.at, buf, None, drain)
        } else {
            capture::after_checked_drain(&mut active.store, target.at, buf, &shape, None, drain)
        };
        if let Err(e) = &result {
            active.store.fail(e.clone());
        }
        result
    })
}
pub fn finish() -> Result<Trace, String> {
    let mut active = ACTIVE
        .with(|slot| slot.borrow_mut().take())
        .ok_or("no family diagnostic")?;
    if !active.pending.is_empty() {
        active
            .store
            .fail("retained stages never reached their owning checked layer drain");
    }
    active.store.finish()
}

/// Retain an actual stage buffer until this layer's checked drain. Valid only
/// where the family source keeps issued pool roots unmodified until that drain.
/// No read, allocation of a Metal root, or submission occurs here.
pub fn retain_until_layer_drain(
    family: Family,
    step: u64,
    layer: u32,
    stage: Stage,
    buf: &MlxBuffer,
) -> Result<(), String> {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(a) = slot.as_mut() else {
            return Ok(());
        };
        if a.family != family {
            return Ok(());
        }
        let Some(target) = a.targets.get(&(step, layer, stage)).cloned() else {
            return Ok(());
        };
        if a.pending.len() >= 4096 || a.pending.iter().any(|(_, _, old, _)| old.at == target.at) {
            a.store.fail("duplicate/overquota pending actual stage");
            return Err("pending stage refusal".into());
        }
        a.pending.push((step, layer, target, buf.clone()));
        Ok(())
    })
}
pub fn layer_drained(family: Family, step: u64, layer: u32, drain: &str) -> Result<(), String> {
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(a) = slot.as_mut() else {
            return Ok(());
        };
        if a.family != family {
            return Ok(());
        }
        let mut keep = Vec::new();
        let mut error = None;
        for (source_step, source_layer, target, buf) in std::mem::take(&mut a.pending) {
            if source_step == step && source_layer == layer {
                let result = if target.normalize_rows {
                    capture::rows_after_checked_drain(&mut a.store, target.at, &buf, None, drain)
                } else {
                    capture::after_checked_drain(
                        &mut a.store,
                        target.at,
                        &buf,
                        buf.shape(),
                        None,
                        drain,
                    )
                };
                if let Err(e) = result {
                    a.store.fail(e.clone());
                    error = Some(e);
                }
            } else {
                keep.push((source_step, source_layer, target, buf));
            }
        }
        a.pending = keep;
        match error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    })
}
pub fn enabled(family: Family) -> bool {
    ACTIVE.with(|slot| slot.borrow().as_ref().is_some_and(|a| a.family == family))
}

pub fn enter_layer(family: Family, step: u64, layer: u32) {
    ACTIVE.with(|slot| {
        if let Some(a) = slot.borrow_mut().as_mut() {
            if a.family == family {
                a.current = Some((step, layer));
            }
        }
    });
}
pub fn retain_current(family: Family, stage: Stage, buf: &MlxBuffer) -> Result<(), String> {
    let current = ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .filter(|a| a.family == family)
            .and_then(|a| a.current)
    });
    if let Some((step, layer)) = current {
        retain_until_layer_drain(family, step, layer, stage, buf)
    } else {
        Ok(())
    }
}
