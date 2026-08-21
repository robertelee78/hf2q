//! ADR-047 diagnostic model-lifecycle coordination.
//!
//! Ordinary OpenAI requests take a shared admission guard while resolving an
//! engine and acquiring a generation-bound lease. Explicit diagnostic
//! switching takes the exclusive guard, marks exact victims draining, and
//! waits for those leases (including SSE response bodies) to disappear before
//! spilling or stopping a worker.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use tokio::sync::{Notify, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use crate::serve::multi_model::{
    AdmissionOutcome, HotSwapError, HotSwapManager, LoadedEngine, LoadedSummary, NonEvictingLoad,
    PreparedEvictionError,
};
use crate::serve::quant_select::QuantType;

const SERVING: u8 = 0;
const DRAINING: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelIdentity {
    pub pool_key: String,
    pub generation: u64,
}

impl ModelIdentity {
    pub fn from_summary(summary: &LoadedSummary) -> Self {
        Self {
            pool_key: summary.pool_key.clone(),
            generation: summary.generation,
        }
    }

    pub fn from_engine<E>(engine: &LoadedEngine<E>) -> Self {
        Self {
            pool_key: format!("{}@{}", engine.repo, engine.quant.as_str()),
            generation: engine.generation,
        }
    }
}

#[derive(Debug)]
struct ModelActivity {
    identity: ModelIdentity,
    phase: AtomicU8,
    active: AtomicUsize,
    idle: Notify,
}

#[derive(Debug)]
struct LifecycleInner {
    activities: Mutex<HashMap<ModelIdentity, Arc<ModelActivity>>>,
}

/// AppState-owned admission gate and generation-bound activity registry.
#[derive(Debug, Clone)]
pub struct ModelLifecycleCoordinator {
    gate: Arc<RwLock<()>>,
    inner: Arc<LifecycleInner>,
}

impl Default for ModelLifecycleCoordinator {
    fn default() -> Self {
        Self {
            gate: Arc::new(RwLock::new(())),
            inner: Arc::new(LifecycleInner {
                activities: Mutex::new(HashMap::new()),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleError {
    Draining(ModelIdentity),
    DrainTimeout {
        victims: Vec<ModelIdentity>,
        timeout: Duration,
    },
    RegistryPoisoned,
    PoolPoisoned,
    StalePlan {
        expected_revision: u64,
        actual_revision: u64,
    },
    VictimPlanChanged,
    PrepareFailed(PreparedEvictionError),
    ShutdownFailed(String),
    LoadFailed(String),
    PostCommitFailed(String),
}

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Draining(identity) => write!(
                f,
                "model {} generation {} is draining; restart the server if it does not recover",
                identity.pool_key, identity.generation
            ),
            Self::DrainTimeout { victims, timeout } => write!(
                f,
                "timed out after {timeout:?} draining model generations: {}",
                victims
                    .iter()
                    .map(|id| format!("{}#{}", id.pool_key, id.generation))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::RegistryPoisoned => write!(f, "model lifecycle registry mutex poisoned"),
            Self::PoolPoisoned => write!(f, "model pool rwlock poisoned"),
            Self::StalePlan {
                expected_revision,
                actual_revision,
            } => write!(
                f,
                "confirmed pool revision is stale: expected {expected_revision}, actual {actual_revision}"
            ),
            Self::VictimPlanChanged => write!(f, "confirmed eviction victim plan changed"),
            Self::PrepareFailed(error) => write!(f, "eviction preparation failed: {error}"),
            Self::ShutdownFailed(error) => write!(f, "victim worker shutdown failed: {error}"),
            Self::LoadFailed(error) => write!(f, "replacement model load failed: {error}"),
            Self::PostCommitFailed(error) => {
                write!(f, "model switch failed after victim removal: {error}")
            }
        }
    }
}

impl std::error::Error for LifecycleError {}

impl LifecycleError {
    /// Whether a failed explicit switch can have crossed the draining or
    /// eviction boundary and therefore requires process restart before the
    /// endpoint can be trusted for further diagnostic activation.
    pub fn requires_restart(&self) -> bool {
        matches!(
            self,
            Self::DrainTimeout { .. }
                | Self::PrepareFailed(_)
                | Self::ShutdownFailed(_)
                | Self::LoadFailed(_)
                | Self::PostCommitFailed(_)
        )
    }
}

impl ModelLifecycleCoordinator {
    pub async fn read_admission(&self) -> OwnedRwLockReadGuard<()> {
        Arc::clone(&self.gate).read_owned().await
    }

    pub async fn write_admission(&self) -> OwnedRwLockWriteGuard<()> {
        Arc::clone(&self.gate).write_owned().await
    }

    /// Acquire a lease for the exact engine generation. Callers hold a
    /// shared admission guard through engine resolution and this call.
    pub fn acquire<E>(&self, engine: &Arc<LoadedEngine<E>>) -> Result<ModelLease, LifecycleError> {
        self.acquire_identity(ModelIdentity::from_engine(engine))
    }

    fn acquire_identity(&self, identity: ModelIdentity) -> Result<ModelLease, LifecycleError> {
        let mut activities = self
            .inner
            .activities
            .lock()
            .map_err(|_| LifecycleError::RegistryPoisoned)?;
        let activity = Arc::clone(activities.entry(identity.clone()).or_insert_with(|| {
            Arc::new(ModelActivity {
                identity,
                phase: AtomicU8::new(SERVING),
                active: AtomicUsize::new(0),
                idle: Notify::new(),
            })
        }));
        if activity.phase.load(Ordering::Acquire) == DRAINING {
            return Err(LifecycleError::Draining(activity.identity.clone()));
        }
        activity.active.fetch_add(1, Ordering::AcqRel);
        Ok(ModelLease {
            activity,
            coordinator: Arc::downgrade(&self.inner),
        })
    }

    /// Mark exact generations draining before waiting. The registry lock
    /// covers creation plus the phase transition, closing the last-lease
    /// drop/removal race.
    pub fn begin_drain(
        &self,
        victims: &[LoadedSummary],
    ) -> Result<Vec<DrainHandle>, LifecycleError> {
        let mut activities = self
            .inner
            .activities
            .lock()
            .map_err(|_| LifecycleError::RegistryPoisoned)?;
        let mut handles = Vec::with_capacity(victims.len());
        for victim in victims {
            let identity = ModelIdentity::from_summary(victim);
            let activity = Arc::clone(activities.entry(identity.clone()).or_insert_with(|| {
                Arc::new(ModelActivity {
                    identity,
                    phase: AtomicU8::new(SERVING),
                    active: AtomicUsize::new(0),
                    idle: Notify::new(),
                })
            }));
            activity.phase.store(DRAINING, Ordering::Release);
            handles.push(DrainHandle { activity });
        }
        Ok(handles)
    }

    pub async fn wait_for_zero(
        &self,
        drains: &[DrainHandle],
        timeout: Duration,
    ) -> Result<(), LifecycleError> {
        let victims = drains
            .iter()
            .map(|handle| handle.activity.identity.clone())
            .collect::<Vec<_>>();
        let wait = async {
            for handle in drains {
                loop {
                    let notified = handle.activity.idle.notified();
                    if handle.activity.active.load(Ordering::Acquire) == 0 {
                        break;
                    }
                    notified.await;
                }
            }
        };
        tokio::time::timeout(timeout, wait)
            .await
            .map_err(|_| LifecycleError::DrainTimeout { victims, timeout })
    }

    /// Forget successfully removed generations. Failure paths deliberately
    /// skip this call, leaving the generation unavailable for new leases.
    pub fn finish_removal(&self, drains: &[DrainHandle]) -> Result<(), LifecycleError> {
        let mut activities = self
            .inner
            .activities
            .lock()
            .map_err(|_| LifecycleError::RegistryPoisoned)?;
        for drain in drains {
            if activities
                .get(&drain.activity.identity)
                .is_some_and(|current| Arc::ptr_eq(current, &drain.activity))
            {
                activities.remove(&drain.activity.identity);
            }
        }
        Ok(())
    }

    /// Execute ADR-047's explicit, revision-bound switch. The exclusive
    /// admission guard is held for the state transition, but request execution
    /// is represented only by leases and therefore drains without deadlock.
    pub async fn switch<E, Shutdown, ShutdownFuture, Load, LoadFuture>(
        &self,
        pool: Arc<std::sync::RwLock<HotSwapManager<E>>>,
        confirmation: SwitchConfirmation,
        drain_timeout: Duration,
        shutdown: Shutdown,
        load: Load,
    ) -> Result<NonEvictingLoad<E>, LifecycleError>
    where
        E: Send + Sync + 'static,
        Shutdown: Fn(Arc<LoadedEngine<E>>) -> ShutdownFuture,
        ShutdownFuture: std::future::Future<Output = anyhow::Result<()>>,
        Load: FnOnce(Arc<std::sync::RwLock<HotSwapManager<E>>>) -> LoadFuture,
        LoadFuture: std::future::Future<Output = Result<NonEvictingLoad<E>, HotSwapError>>,
    {
        let _admission_guard = self.write_admission().await;
        self.validate_switch_confirmation(&pool, &confirmation)?;

        let drains = self.begin_drain(&confirmation.victims)?;
        self.wait_for_zero(&drains, drain_timeout).await?;

        let prepared = {
            let manager = pool.read().map_err(|_| LifecycleError::PoolPoisoned)?;
            manager
                .prepare_evictions(confirmation.expected_revision, &confirmation.victims)
                .map_err(LifecycleError::PrepareFailed)?
        };

        for engine in prepared.engines().cloned().collect::<Vec<_>>() {
            shutdown(engine)
                .await
                .map_err(|error| LifecycleError::ShutdownFailed(format!("{error:#}")))?;
        }

        {
            let mut manager = pool.write().map_err(|_| LifecycleError::PoolPoisoned)?;
            manager
                .commit_prepared(prepared)
                .map_err(LifecycleError::PrepareFailed)?;
        }
        self.finish_removal(&drains).map_err(|error| {
            LifecycleError::PostCommitFailed(format!("activity cleanup failed: {error}"))
        })?;

        match load(Arc::clone(&pool))
            .await
            .map_err(|error| LifecycleError::LoadFailed(error.to_string()))?
        {
            NonEvictingLoad::Conflict(_) => Err(LifecycleError::PostCommitFailed(
                "replacement admission conflicted after the confirmed victims were removed"
                    .to_owned(),
            )),
            loaded => Ok(loaded),
        }
    }

    /// Pure preflight for an explicit switch receipt. Callers that need the
    /// result to remain stable while acting must hold the admission write
    /// guard; [`Self::switch`] always re-runs this check under its own guard.
    pub fn validate_switch_confirmation<E>(
        &self,
        pool: &Arc<std::sync::RwLock<HotSwapManager<E>>>,
        confirmation: &SwitchConfirmation,
    ) -> Result<(), LifecycleError> {
        let manager = pool.read().map_err(|_| LifecycleError::PoolPoisoned)?;
        let actual_revision = manager.pool_stats().revision;
        if actual_revision != confirmation.expected_revision {
            return Err(LifecycleError::StalePlan {
                expected_revision: confirmation.expected_revision,
                actual_revision,
            });
        }
        let plan = manager.admission_plan(
            &confirmation.candidate_repo,
            confirmation.candidate_quant,
            confirmation.candidate_bytes,
        );
        let planned_keys = match plan.outcome {
            AdmissionOutcome::WouldEvict { victims, .. } => victims
                .into_iter()
                .map(|victim| victim.repo_id)
                .collect::<Vec<_>>(),
            _ => return Err(LifecycleError::VictimPlanChanged),
        };
        let confirmed_keys = confirmation
            .victims
            .iter()
            .map(|victim| victim.pool_key.clone())
            .collect::<Vec<_>>();
        if planned_keys != confirmed_keys {
            return Err(LifecycleError::VictimPlanChanged);
        }
        let current = manager
            .iter_loaded()
            .filter(|entry| confirmed_keys.contains(&entry.pool_key))
            .collect::<Vec<_>>();
        if current != confirmation.victims {
            return Err(LifecycleError::VictimPlanChanged);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SwitchConfirmation {
    pub candidate_repo: String,
    pub candidate_quant: QuantType,
    pub candidate_bytes: u64,
    pub expected_revision: u64,
    pub victims: Vec<LoadedSummary>,
}

/// RAII request lifetime. This type is intentionally non-cloneable: one
/// successful acquire corresponds to one decrement on Drop.
#[derive(Debug)]
pub struct ModelLease {
    activity: Arc<ModelActivity>,
    coordinator: Weak<LifecycleInner>,
}

impl Drop for ModelLease {
    fn drop(&mut self) {
        let previous = self.activity.active.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "model lease active count underflow");
        if previous != 1 {
            return;
        }
        self.activity.idle.notify_waiters();
        if self.activity.phase.load(Ordering::Acquire) == DRAINING {
            return;
        }
        if let Some(inner) = self.coordinator.upgrade() {
            if let Ok(mut activities) = inner.activities.lock() {
                if activities
                    .get(&self.activity.identity)
                    .is_some_and(|current| Arc::ptr_eq(current, &self.activity))
                {
                    activities.remove(&self.activity.identity);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct DrainHandle {
    activity: Arc<ModelActivity>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::multi_model::{
        EngineConfig, KvSpiller, LoadedHandle, LoadedPool, ModelLoader, RestoreOutcome,
        SpillOutcome,
    };
    use std::path::Path;

    fn summary(key: &str, generation: u64) -> LoadedSummary {
        LoadedSummary {
            pool_key: key.to_string(),
            quant: "Q4_K_M".to_string(),
            bytes_resident: 1,
            generation,
        }
    }

    #[tokio::test]
    async fn drain_waits_for_last_raii_lease() {
        let coordinator = ModelLifecycleCoordinator::default();
        let identity = ModelIdentity {
            pool_key: "a/1@Q4_K_M".into(),
            generation: 7,
        };
        let lease = coordinator.acquire_identity(identity).unwrap();
        let drains = coordinator
            .begin_drain(&[summary("a/1@Q4_K_M", 7)])
            .unwrap();
        assert!(matches!(
            coordinator.acquire_identity(ModelIdentity {
                pool_key: "a/1@Q4_K_M".into(),
                generation: 7,
            }),
            Err(LifecycleError::Draining(_))
        ));

        let wait = coordinator.wait_for_zero(&drains, Duration::from_secs(1));
        tokio::pin!(wait);
        assert!(futures::poll!(&mut wait).is_pending());
        drop(lease);
        wait.await.unwrap();
    }

    #[tokio::test]
    async fn re_admitted_pool_key_uses_fresh_generation_activity() {
        let coordinator = ModelLifecycleCoordinator::default();
        let old = coordinator
            .acquire_identity(ModelIdentity {
                pool_key: "a/1@Q4_K_M".into(),
                generation: 1,
            })
            .unwrap();
        let drains = coordinator
            .begin_drain(&[summary("a/1@Q4_K_M", 1)])
            .unwrap();
        let new = coordinator
            .acquire_identity(ModelIdentity {
                pool_key: "a/1@Q4_K_M".into(),
                generation: 2,
            })
            .expect("new generation must not inherit old draining state");
        drop(new);
        drop(old);
        coordinator
            .wait_for_zero(&drains, Duration::from_secs(1))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn drain_timeout_leaves_generation_fail_closed() {
        let coordinator = ModelLifecycleCoordinator::default();
        let identity = ModelIdentity {
            pool_key: "a/1@Q4_K_M".into(),
            generation: 3,
        };
        let _lease = coordinator.acquire_identity(identity.clone()).unwrap();
        let drains = coordinator
            .begin_drain(&[summary("a/1@Q4_K_M", 3)])
            .unwrap();
        let err = coordinator
            .wait_for_zero(&drains, Duration::from_millis(1))
            .await
            .unwrap_err();
        assert!(matches!(err, LifecycleError::DrainTimeout { .. }));
        assert!(matches!(
            coordinator.acquire_identity(identity),
            Err(LifecycleError::Draining(_))
        ));
    }

    #[derive(Debug)]
    struct TestEngine;

    struct TestLoader {
        calls: AtomicUsize,
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl ModelLoader<TestEngine> for TestLoader {
        fn load(&self, _path: &Path, _config: &EngineConfig) -> anyhow::Result<TestEngine> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.events.lock().unwrap().push("load");
            Ok(TestEngine)
        }
    }

    struct TestSpiller {
        events: Arc<Mutex<Vec<&'static str>>>,
    }

    impl KvSpiller<TestEngine> for TestSpiller {
        fn pre_evict(
            &self,
            _handle: &LoadedHandle,
            _engine: &Arc<LoadedEngine<TestEngine>>,
        ) -> SpillOutcome {
            self.events.lock().unwrap().push("spill");
            SpillOutcome::Skipped
        }

        fn post_admit(
            &self,
            _repo: &str,
            _quant: QuantType,
            _engine: &Arc<LoadedEngine<TestEngine>>,
        ) -> RestoreOutcome {
            self.events.lock().unwrap().push("restore");
            RestoreOutcome::Skipped
        }

        fn drop_family(&self, _repo: &str, _quant: QuantType) {
            self.events.lock().unwrap().push("drop");
        }
    }

    fn fixture_file(bytes: usize) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&vec![0; bytes]).unwrap();
        file
    }

    fn switch_fixture() -> (
        Arc<std::sync::RwLock<HotSwapManager<TestEngine>>>,
        Arc<TestLoader>,
        Arc<Mutex<Vec<&'static str>>>,
        tempfile::NamedTempFile,
        Arc<LoadedEngine<TestEngine>>,
        SwitchConfirmation,
    ) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let loader = Arc::new(TestLoader {
            calls: AtomicUsize::new(0),
            events: Arc::clone(&events),
        });
        let spiller = Arc::new(TestSpiller {
            events: Arc::clone(&events),
        });
        let mut manager = HotSwapManager::new_with_spiller(
            LoadedPool::with_capacity_and_budget(1, 800),
            loader.clone(),
            spiller,
        );
        let current_file = fixture_file(400);
        let current = manager
            .load_or_get(
                "a/1",
                QuantType::Q4_K_M,
                current_file.path(),
                &EngineConfig::default(),
            )
            .unwrap();
        let victims = manager.iter_loaded().collect::<Vec<_>>();
        let confirmation = SwitchConfirmation {
            candidate_repo: "b/2".into(),
            candidate_quant: QuantType::Q4_K_M,
            candidate_bytes: 500,
            expected_revision: manager.pool_stats().revision,
            victims,
        };
        events.lock().unwrap().clear();
        (
            Arc::new(std::sync::RwLock::new(manager)),
            loader,
            events,
            current_file,
            current,
            confirmation,
        )
    }

    #[tokio::test]
    async fn switch_orders_drain_spill_shutdown_commit_then_load() {
        let coordinator = ModelLifecycleCoordinator::default();
        let (pool, _loader, events, _current_file, _current, confirmation) = switch_fixture();
        let target_file = fixture_file(500);
        let shutdown_events = Arc::clone(&events);
        let result = coordinator
            .switch(
                Arc::clone(&pool),
                confirmation,
                Duration::from_secs(1),
                move |_engine| {
                    let events = Arc::clone(&shutdown_events);
                    async move {
                        events.lock().unwrap().push("shutdown");
                        Ok(())
                    }
                },
                move |pool| async move {
                    pool.write()
                        .map_err(|error| {
                            HotSwapError::LoaderFailed(anyhow::anyhow!(
                                "pool rwlock poisoned: {error}"
                            ))
                        })?
                        .load_or_get_non_evicting(
                            "b/2",
                            QuantType::Q4_K_M,
                            target_file.path(),
                            &EngineConfig::default(),
                        )
                },
            )
            .await
            .unwrap();
        assert!(matches!(result, NonEvictingLoad::Loaded(_)));
        assert_eq!(
            *events.lock().unwrap(),
            vec!["spill", "shutdown", "drop", "load", "restore"]
        );
        let manager = pool.read().unwrap();
        assert!(manager.try_get("a/1", QuantType::Q4_K_M).is_none());
        assert!(manager.try_get("b/2", QuantType::Q4_K_M).is_some());
    }

    #[tokio::test]
    async fn shutdown_failure_keeps_victim_and_never_loads_replacement() {
        let coordinator = ModelLifecycleCoordinator::default();
        let (pool, loader, _events, _current_file, current, confirmation) = switch_fixture();
        let target_file = fixture_file(500);
        let err = coordinator
            .switch(
                Arc::clone(&pool),
                confirmation,
                Duration::from_secs(1),
                |_engine| async move { anyhow::bail!("synthetic shutdown failure") },
                move |pool| async move {
                    pool.write()
                        .map_err(|error| {
                            HotSwapError::LoaderFailed(anyhow::anyhow!(
                                "pool rwlock poisoned: {error}"
                            ))
                        })?
                        .load_or_get_non_evicting(
                            "b/2",
                            QuantType::Q4_K_M,
                            target_file.path(),
                            &EngineConfig::default(),
                        )
                },
            )
            .await
            .unwrap_err();
        assert!(matches!(err, LifecycleError::ShutdownFailed(_)));
        assert_eq!(loader.calls.load(Ordering::SeqCst), 1);
        let manager = pool.read().unwrap();
        assert!(manager.try_get("a/1", QuantType::Q4_K_M).is_some());
        assert!(manager.try_get("b/2", QuantType::Q4_K_M).is_none());
        drop(manager);
        assert!(matches!(
            coordinator.acquire(&current),
            Err(LifecycleError::Draining(_))
        ));
    }

    #[tokio::test]
    async fn post_commit_admission_conflict_is_restart_required() {
        let coordinator = ModelLifecycleCoordinator::default();
        let (pool, loader, _events, _current_file, _current, confirmation) = switch_fixture();
        // The confirmation was issued for 500 bytes, but the materialized
        // local artifact grew beyond the entire 800-byte pool budget before
        // publication. This conflict is after victim commit, not a stale
        // preflight plan, so the endpoint must require restart.
        let target_file = fixture_file(900);
        let error = coordinator
            .switch(
                Arc::clone(&pool),
                confirmation,
                Duration::from_secs(1),
                |_engine| async move { Ok(()) },
                move |pool| async move {
                    pool.write()
                        .map_err(|error| {
                            HotSwapError::LoaderFailed(anyhow::anyhow!(
                                "pool rwlock poisoned: {error}"
                            ))
                        })?
                        .load_or_get_non_evicting(
                            "b/2",
                            QuantType::Q4_K_M,
                            target_file.path(),
                            &EngineConfig::default(),
                        )
                },
            )
            .await
            .unwrap_err();

        assert!(matches!(error, LifecycleError::PostCommitFailed(_)));
        assert!(error.requires_restart());
        assert_eq!(loader.calls.load(Ordering::SeqCst), 1);
        let manager = pool.read().unwrap();
        assert!(manager.try_get("a/1", QuantType::Q4_K_M).is_none());
        assert!(manager.try_get("b/2", QuantType::Q4_K_M).is_none());
    }

    #[tokio::test]
    async fn switch_drain_timeout_keeps_victim_and_never_loads_replacement() {
        let coordinator = ModelLifecycleCoordinator::default();
        let (pool, loader, _events, _current_file, current, confirmation) = switch_fixture();
        let held_request = coordinator.acquire(&current).unwrap();
        let target_file = fixture_file(500);

        let err = coordinator
            .switch(
                Arc::clone(&pool),
                confirmation,
                Duration::from_millis(1),
                |_engine| async move { Ok(()) },
                move |pool| async move {
                    pool.write()
                        .map_err(|error| {
                            HotSwapError::LoaderFailed(anyhow::anyhow!(
                                "pool rwlock poisoned: {error}"
                            ))
                        })?
                        .load_or_get_non_evicting(
                            "b/2",
                            QuantType::Q4_K_M,
                            target_file.path(),
                            &EngineConfig::default(),
                        )
                },
            )
            .await
            .unwrap_err();

        assert!(matches!(err, LifecycleError::DrainTimeout { .. }));
        assert_eq!(loader.calls.load(Ordering::SeqCst), 1);
        let manager = pool.read().unwrap();
        assert!(manager.try_get("a/1", QuantType::Q4_K_M).is_some());
        assert!(manager.try_get("b/2", QuantType::Q4_K_M).is_none());
        drop(manager);
        assert!(matches!(
            coordinator.acquire(&current),
            Err(LifecycleError::Draining(_))
        ));
        drop(held_request);
    }

    #[tokio::test]
    async fn switch_waits_for_unary_embedding_queued_and_sse_leases() {
        let coordinator = ModelLifecycleCoordinator::default();
        let (pool, loader, _events, _current_file, current, confirmation) = switch_fixture();
        // Every production pool-routed request acquires the same
        // generation-bound lease before dispatch. Give the four relevant
        // lifetimes distinct names here so one switch proves it cannot pass
        // the drain until all of them are gone.
        let unary = coordinator.acquire(&current).unwrap();
        let embedding = coordinator.acquire(&current).unwrap();
        let queued = coordinator.acquire(&current).unwrap();
        let sse_body = coordinator.acquire(&current).unwrap();
        let target_file = fixture_file(500);
        let switch_coordinator = coordinator.clone();
        let switch_pool = Arc::clone(&pool);

        let switching = tokio::spawn(async move {
            switch_coordinator
                .switch(
                    switch_pool,
                    confirmation,
                    Duration::from_secs(1),
                    |_engine| async move { Ok(()) },
                    move |pool| async move {
                        pool.write()
                            .map_err(|error| {
                                HotSwapError::LoaderFailed(anyhow::anyhow!(
                                    "pool rwlock poisoned: {error}"
                                ))
                            })?
                            .load_or_get_non_evicting(
                                "b/2",
                                QuantType::Q4_K_M,
                                target_file.path(),
                                &EngineConfig::default(),
                            )
                    },
                )
                .await
        });

        tokio::task::yield_now().await;
        for lease in [unary, embedding, queued] {
            drop(lease);
            tokio::task::yield_now().await;
            assert!(!switching.is_finished());
            assert_eq!(loader.calls.load(Ordering::SeqCst), 1);
        }
        drop(sse_body);

        assert!(matches!(
            switching.await.unwrap().unwrap(),
            NonEvictingLoad::Loaded(_)
        ));
        assert_eq!(loader.calls.load(Ordering::SeqCst), 2);
    }
}
