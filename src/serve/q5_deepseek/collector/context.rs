//! Metadata context transfers with the actual model; attachment is exclusive.
use super::Collector;
use anyhow::{ensure, Context, Result};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

struct Shared {
    state: Mutex<Collector>,
    failed: AtomicBool,
}
#[derive(Clone)]
pub(crate) struct Handle(Arc<Shared>);
pub(crate) struct Scope {
    handle: Handle,
    // Handle may transfer; an attached scope must drop on its owning thread.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}
thread_local! { static ACTIVE: RefCell<Option<Handle>> = const { RefCell::new(None) }; }

pub(super) fn active() -> bool {
    ACTIVE.with(|slot| slot.try_borrow().map_or(true, |s| s.is_some()))
}
pub(super) fn with_current(f: impl FnOnce(&mut Collector) -> Result<()>) -> Result<()> {
    let handle = ACTIVE.with(|slot| {
        slot.try_borrow()
            .map(|s| s.clone())
            .context("Q5 TLS callback reentry")
    })?;
    let Some(handle) = handle else {
        return Ok(());
    };
    let result = (|| {
        let mut state = handle
            .0
            .state
            .try_lock()
            .map_err(|_| anyhow::anyhow!("Q5 collector overlap/poison"))?;
        ensure!(
            state.attached == Some(std::thread::current().id()),
            "Q5 callback on unbound worker"
        );
        if handle.0.failed.load(Ordering::Acquire) {
            state.poisoned = true;
        }
        let result = f(&mut state);
        if result.is_err() {
            state.poisoned = true;
        }
        result
    })();
    if result.is_err() {
        handle.0.failed.store(true, Ordering::Release);
    }
    result
}
impl Scope {
    pub(crate) fn enter() -> Result<Self> {
        let handle = Handle(Arc::new(Shared {
            state: Mutex::new(Collector {
                roots: BTreeMap::new(),
                weights: BTreeMap::new(),
                source: None,
                events: 0,
                completed_results: 0,
                transient_observations: [(0, 0); 4],
                pool_latest: [(0, 0); 3],
                pool_observed_highwater: [0; 3],
                upper: 16_384,
                poisoned: false,
                owner: None,
                completed_owners: [None; 4],
                lifecycle: None,
                attached: None,
                next_chain: 1,
                commands: BTreeMap::new(),
            }),
            failed: AtomicBool::new(false),
        }));
        handle.attach()
    }
    pub(crate) fn handle(&self) -> Handle {
        self.handle.clone()
    }
}
impl Handle {
    pub(crate) fn require_current_model(&self, model: Option<&Handle>) -> Result<()> {
        ensure!(
            model.is_some_and(|model| Arc::ptr_eq(&model.0, &self.0)) && self.is_current(),
            "Q5 operation/model/originating worker context mismatch"
        );
        Ok(())
    }
    pub(crate) fn is_current(&self) -> bool {
        ACTIVE.with(|slot| {
            slot.try_borrow().is_ok_and(|slot| {
                slot.as_ref()
                    .is_some_and(|handle| Arc::ptr_eq(&handle.0, &self.0))
            })
        })
    }
    pub(crate) fn attach(&self) -> Result<Scope> {
        ACTIVE.with(|slot| {
            let mut slot = slot.try_borrow_mut().context("Q5 attachment reentry")?;
            ensure!(slot.is_none(), "Q5 overlapping diagnostic owner");
            let mut state = self
                .0
                .state
                .try_lock()
                .map_err(|_| anyhow::anyhow!("Q5 attachment state busy"))?;
            ensure!(
                !self.0.failed.load(Ordering::Acquire)
                    && !state.poisoned
                    && state.attached.is_none()
                    && state.owner.is_none()
                    && state.lifecycle.is_none()
                    && state.commands.is_empty(),
                "Q5 unsafe context transfer"
            );
            state.attached = Some(std::thread::current().id());
            *slot = Some(self.clone());
            Ok(Scope {
                handle: self.clone(),
                _thread: std::marker::PhantomData,
            })
        })
    }
    /// Actual observed metadata counters; NOT native/live/physical capacities.
    pub(crate) fn summary(&self) -> Result<(u64, u64, usize, bool)> {
        let state = self
            .0
            .state
            .try_lock()
            .map_err(|_| anyhow::anyhow!("Q5 summary state busy"))?;
        ensure!(
            !self.0.failed.load(Ordering::Acquire) && !state.poisoned,
            "Q5 summary health"
        );
        Ok((
            state.events,
            state.completed_results,
            state.roots.len(),
            state.source.is_some(),
        ))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|slot| {
            let Ok(mut slot) = slot.try_borrow_mut() else {
                self.handle.0.failed.store(true, Ordering::Release);
                return;
            };
            if !slot
                .as_ref()
                .is_some_and(|h| Arc::ptr_eq(&h.0, &self.handle.0))
            {
                self.handle.0.failed.store(true, Ordering::Release);
                return;
            }
            *slot = None;
            match self.handle.0.state.try_lock() {
                Ok(mut state) => {
                    if state.owner.is_some()
                        || state.lifecycle.is_some()
                        || !state.commands.is_empty()
                    {
                        state.poisoned = true;
                        self.handle.0.failed.store(true, Ordering::Release);
                    }
                    state.attached = None;
                }
                Err(_) => self.handle.0.failed.store(true, Ordering::Release),
            }
        });
    }
}
