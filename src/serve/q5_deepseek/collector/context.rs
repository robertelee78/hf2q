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
    emitted: AtomicBool,
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
                native_domain: super::native_domain::NativeDomain::default(),
                selected_load_profile: None,
                page_observations: BTreeMap::new(),
                events: 0,
                completed_results: 0,
                transient_observations: [(0, 0); 4],
                pool_latest: [(0, 0); 3],
                pool_observations: 0,
                pool_observed_highwater: [0; 3],
                upper: 16_384,
                poisoned: false,
                owner: None,
                completed_owners: [None; 4],
                lifecycle: None,
                manifest_sequence: 0,
                manifest: None,
                mounted_pair: None,
                mounted: false,
                attached: None,
                next_chain: 1,
                commands: BTreeMap::new(),
            }),
            failed: AtomicBool::new(false),
            emitted: AtomicBool::new(false),
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

#[derive(serde::Serialize)]
struct TerminalReceipt<'a> {
    schema: &'static str,
    verdict: &'static str,
    phase: &'static str,
    runtime_qualified: bool,
    physical_credit: bool,
    pid: u32,
    compiled_source: Option<&'static str>,
    compiled_native_pin: &'static str,
    process_uuid_authority: &'static str,
    actual_load_profile: &'a Option<crate::serve::q5_deepseek::selected_profile::SelectedLoadProfile>,
    native_whole_live_pending: &'a super::native_domain::Snapshot,
    bounded_page_metadata_full_charge:
        &'a BTreeMap<usize, crate::serve::q5_deepseek::page_observer::RootObservation>,
    unqueried_roots_full_charge: usize,
    model_still_retained: bool,
    source_fd_unchanged: Option<bool>,
    observation_failed: bool,
    unjoined_tracked_commands: usize,
    open_owner: bool,
    open_lifecycle: bool,
    allocation_observation_events_not_creation: u64,
    checked_results: u64,
    transient_observations_not_unique_live_or_peak: &'a [(u64, u64); 4],
    current_borrowed_role_manifest:
        &'a Option<crate::serve::q5_deepseek::manifest::CurrentManifest>,
    last_mounted_pair_partial: &'a Option<crate::serve::q5_deepseek::manifest::CurrentManifest>,
    weight_source_coverage: super::coverage::Coverage,
    weight_parent_observations: &'a std::collections::BTreeMap<usize, super::Root>,
    pools_current_retained_if_observed: Option<&'a [(usize, usize); 3]>,
    pool_observed_highwater_not_lifetime: Option<&'a [usize; 3]>,
    required_domains: Vec<(&'static str, &'static str)>,
}
impl Handle {
    pub(crate) fn emit_terminal(&self) -> Result<()> {
        ensure!(
            !self.0.emitted.swap(true, Ordering::AcqRel),
            "Q5 duplicate terminal receipt"
        );
        let bytes = {
            let mut state = self
                .0
                .state
                .try_lock()
                .map_err(|_| anyhow::anyhow!("Q5 terminal state unavailable"))?;
            state.native_domain.capture();
            let source_unchanged = state.source.as_ref().map(|source| {
                source
                    .file
                    .metadata()
                    .ok()
                    .is_some_and(|m| source.expected.matches(&m).is_ok())
            });
            let failed = self.0.failed.load(Ordering::Acquire)
                || state.poisoned
                || source_unchanged != Some(true)
                || state.mounted
                || state
                    .mounted_pair
                    .as_ref()
                    .is_some_and(|m| m.end_ns.is_none())
                || state.owner.is_some()
                || state.lifecycle.is_some()
                || !state.commands.is_empty()
                || state.manifest.as_ref().is_none_or(|m| m.end_ns.is_none());
            let record = TerminalReceipt {
                schema: "hf2q.q5.deepseek.collector-terminal.v1",
                verdict: if failed {
                    "StoppedIncomplete"
                } else {
                    "CapturedPartialAwaitingWholePolicy"
                },
                phase: "worker_exit_before_model_drop",
                runtime_qualified: false,
                physical_credit: false,
                pid: std::process::id(),
                compiled_source: option_env!("HF2Q_BUILD_GIT_SHA"),
                compiled_native_pin: env!("HF2Q_MLX_NATIVE_VERSION"),
                process_uuid_authority: "ExternalExactJoinRequired",
                actual_load_profile: &state.selected_load_profile,
                native_whole_live_pending: &state.native_domain.latest,
                bounded_page_metadata_full_charge: &state.page_observations,
                unqueried_roots_full_charge: state
                    .weights
                    .len()
                    .saturating_sub(state.page_observations.len()),
                model_still_retained: true,
                source_fd_unchanged: source_unchanged,
                observation_failed: failed,
                unjoined_tracked_commands: state.commands.len(),
                open_owner: state.owner.is_some(),
                open_lifecycle: state.lifecycle.is_some(),
                allocation_observation_events_not_creation: state.events,
                checked_results: state.completed_results,
                transient_observations_not_unique_live_or_peak: &state.transient_observations,
                current_borrowed_role_manifest: &state.manifest,
                last_mounted_pair_partial: &state.mounted_pair,
                weight_source_coverage: super::coverage::actual(&state)?,
                weight_parent_observations: &state.weights,
                pools_current_retained_if_observed: (state.pool_observations > 0)
                    .then_some(&state.pool_latest),
                pool_observed_highwater_not_lifetime: (state.pool_observations > 0)
                    .then_some(&state.pool_observed_highwater),
                required_domains: crate::serve::q5_deepseek::requirements::describe(
                    crate::serve::q5_deepseek::packet::Phase::Shutdown,
                ),
            };
            crate::serve::q5_deepseek::receipt::encode(
                &record,
                crate::serve::q5_deepseek::manifest::MAX_JSON_BYTES,
            )?
        }; // IO happens OUTSIDE all metadata/native locks.
        use std::io::Write;
        let mut stderr = std::io::stderr().lock();
        stderr.write_all(b"Q5_DEEPSEEK_COLLECTOR_TERMINAL ")?;
        stderr.write_all(&bytes)?;
        stderr.write_all(b"\n")?;
        Ok(())
    }
}
