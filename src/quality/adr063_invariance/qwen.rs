//! Strict adapter at the EXISTING Qwen diagnostic drain boundary. This selects
//! the diagnostic route, not a claim about overlap/performance of serving.
use super::{
    capture,
    receipt::{CaptureTarget, Trace},
    store::Store,
};
use mlx_native::MlxBuffer;
use std::{cell::RefCell, collections::BTreeMap};
struct Active {
    store: Store,
    targets: BTreeMap<(u64, Option<usize>, String), CaptureTarget>,
    pending: Vec<(u64, Option<usize>, CaptureTarget, MlxBuffer)>,
    failed: bool,
}
thread_local! {static ACTIVE:RefCell<Option<Active>>=const {RefCell::new(None)};}
/// Same-thread ownership only. Nested captures refuse before replacing state.
pub fn install(
    store: Store,
    targets: BTreeMap<(u64, Option<usize>, String), CaptureTarget>,
) -> Result<(), String> {
    if targets.is_empty() {
        return Err("empty Qwen capture selection".into());
    }
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err("nested Qwen diagnostic".into());
        }
        *slot = Some(Active {
            store,
            targets,
            pending: Vec::new(),
            failed: false,
        });
        Ok(())
    })
}
pub fn enabled() -> bool {
    ACTIVE.with(|slot| slot.borrow().is_some())
}
pub fn selected(step: u64, layer: Option<usize>, op: &str) -> bool {
    ACTIVE.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|a| a.targets.contains_key(&(step, layer, op.into())))
    })
}
/// Called ONLY after dump_bisect's actual session-aware checked drain succeeds.
pub fn drained(step: u64, layer: Option<usize>, op: &str, buf: &MlxBuffer, shape: &[usize]) {
    ACTIVE.with(|slot| {
        if let Some(a) = slot.borrow_mut().as_mut() {
            let mut keep = Vec::new();
            for (source_step, source_layer, target, buffer) in std::mem::take(&mut a.pending) {
                if source_step == step && source_layer == layer {
                    let result = if target.normalize_rows {
                        capture::rows_after_checked_drain(
                            &mut a.store,
                            target.at,
                            &buffer,
                            None,
                            "Qwen actual FFN diagnostic checked drain",
                        )
                    } else {
                        capture::after_checked_drain(
                            &mut a.store,
                            target.at,
                            &buffer,
                            buffer.shape(),
                            None,
                            "Qwen actual FFN diagnostic checked drain",
                        )
                    };
                    if let Err(e) = result {
                        a.failed = true;
                        a.store.fail(e);
                    }
                } else {
                    keep.push((source_step, source_layer, target, buffer));
                }
            }
            a.pending = keep;
            if let Some(target) = a.targets.get(&(step, layer, op.into())).cloned() {
                let result = if target.normalize_rows {
                    capture::rows_after_checked_drain(
                        &mut a.store,
                        target.at,
                        buf,
                        None,
                        "qwen dump_bisect::flush_gpu checked success",
                    )
                } else {
                    capture::after_checked_drain(
                        &mut a.store,
                        target.at,
                        buf,
                        shape,
                        None,
                        "qwen dump_bisect::flush_gpu checked success",
                    )
                };
                if let Err(e) = result {
                    a.failed = true;
                    a.store.fail(format!("Qwen actual capture {op}: {e}"));
                }
            }
        }
    })
}
pub fn drain_failed(error: String) {
    ACTIVE.with(|slot| {
        if let Some(a) = slot.borrow_mut().as_mut() {
            a.failed = true;
            a.store.fail(error);
        }
    })
}
/// Take the sink even after a backend error. Missing expected stages remain
/// incomplete, and a swallowed legacy dump failure cannot qualify this trace.
pub fn finish() -> Result<Trace, String> {
    let mut active = ACTIVE
        .with(|slot| slot.borrow_mut().take())
        .ok_or("no active Qwen diagnostic")?;
    if !active.pending.is_empty() {
        active
            .store
            .fail("Qwen retained stages did not reach checked diagnostic drain");
    }
    active.store.finish()
}

pub fn needs_drain(step: u64, layer: Option<usize>, op: &str) -> bool {
    selected(step, layer, op)
        || (matches!(op, "ffn_out" | "layer_out")
            && ACTIVE.with(|slot| {
                slot.borrow().as_ref().is_some_and(|a| {
                    a.pending
                        .iter()
                        .any(|(s, l, _, _)| *s == step && *l == layer)
                })
            }))
}
pub fn retain_current(op: &str, buffer: &MlxBuffer, shape: Vec<usize>) -> Result<(), String> {
    let step = crate::inference::models::qwen35::dump_bisect::current_step_idx();
    let layer = crate::inference::models::qwen35::dump_bisect::current_layer_idx();
    ACTIVE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(a) = slot.as_mut() else {
            return Ok(());
        };
        let Some(target) = a.targets.get(&(step, layer, op.into())).cloned() else {
            return Ok(());
        };
        if a.pending.len() >= 4096 || a.pending.iter().any(|(_, _, old, _)| old.at == target.at) {
            return Err("Qwen pending stage duplicate/quota".into());
        }
        let elements = shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or("Qwen pending shape overflow")?;
        let bytes = elements
            .checked_mul(buffer.dtype().size_of())
            .ok_or("Qwen pending byte overflow")?;
        if bytes > buffer.data_byte_len() {
            return Err("Qwen pending stage exceeds logical source".into());
        }
        let view = buffer
            .slice_view(0, elements)
            .with_shape(shape)
            .map_err(|e| e.to_string())?;
        a.pending.push((step, layer, target, view));
        Ok(())
    })
}
