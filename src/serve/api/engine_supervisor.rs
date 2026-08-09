//! Mode-neutral fail-closed health lease for the one model worker owned by an
//! [`Engine`](super::engine::Engine). A Metal call may never return, so
//! receiver closure is not a sufficient readiness signal: the worker still
//! owns the receiver while blocked inside Objective-C. The supervisor lives
//! outside that worker and makes transaction expiry one-way and observable.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{mpsc, watch};

use super::sse::GenerationEvent;

const HEALTHY: u8 = 0;
const POISONED: u8 = 1;
const MONITOR_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Clone)]
pub(super) struct EngineSupervisor {
    inner: Arc<EngineSupervisorInner>,
}

struct EngineSupervisorInner {
    health: AtomicU8,
    next_epoch: AtomicU64,
    armed: Mutex<Option<ArmedTransaction>>,
    reason: Mutex<Option<String>>,
    health_tx: watch::Sender<bool>,
}

#[derive(Clone, Debug)]
struct ArmedTransaction {
    epoch: u64,
    kind: &'static str,
    deadline: Instant,
}

pub(super) struct WorkerTransactionLease {
    supervisor: EngineSupervisor,
    epoch: u64,
    finished: bool,
}

impl EngineSupervisor {
    pub(super) fn new() -> Self {
        Self::new_with_monitor_start(Self::spawn_monitor)
    }

    fn new_with_monitor_start(
        start: impl FnOnce(Weak<EngineSupervisorInner>) -> std::io::Result<()>,
    ) -> Self {
        let (health_tx, _health_rx) = watch::channel(true);
        let inner = Arc::new(EngineSupervisorInner {
            health: AtomicU8::new(HEALTHY),
            next_epoch: AtomicU64::new(1),
            armed: Mutex::new(None),
            reason: Mutex::new(None),
            health_tx,
        });
        if let Err(error) = start(Arc::downgrade(&inner)) {
            tracing::error!(error = %error, "failed to start engine supervisor monitor");
            inner.poison_transaction(ArmedTransaction {
                epoch: 0,
                kind: "supervisor-monitor-spawn",
                deadline: Instant::now(),
            });
        }
        Self { inner }
    }

    fn spawn_monitor(inner: Weak<EngineSupervisorInner>) -> std::io::Result<()> {
        std::thread::Builder::new()
            .name("hf2q-engine-supervisor".into())
            .spawn(move || loop {
                let Some(inner) = inner.upgrade() else {
                    return;
                };
                inner.expire_if_due(Instant::now());
                drop(inner);
                std::thread::sleep(MONITOR_INTERVAL);
            })
            .map(|_| ())
    }

    pub(super) fn is_healthy(&self) -> bool {
        self.inner.expire_if_due(Instant::now());
        self.inner.health.load(Ordering::Acquire) == HEALTHY
    }

    pub(super) fn unhealthy_message(&self) -> String {
        self.inner
            .reason
            .lock()
            .ok()
            .and_then(|reason| reason.clone())
            .unwrap_or_else(|| super::engine::ENGINE_UNHEALTHY_MESSAGE.to_string())
    }

    pub(super) async fn wait_unhealthy(&self) {
        if !self.is_healthy() {
            return;
        }
        let mut health = self.inner.health_tx.subscribe();
        while *health.borrow_and_update() {
            if health.changed().await.is_err() {
                return;
            }
        }
    }

    /// Forward one worker event stream while the engine stays healthy. A
    /// poisoned transaction wins a simultaneous race, emits exactly one
    /// terminal Error event, then drops the worker receiver so late Metal
    /// returns cannot emit a second terminal or keep a cancelled slot alive.
    pub(super) fn guard_generation_events(
        &self,
        mut source: mpsc::Receiver<GenerationEvent>,
    ) -> mpsc::Receiver<GenerationEvent> {
        let supervisor = self.clone();
        let (guarded_tx, guarded_rx) = mpsc::channel(64);
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    _ = supervisor.wait_unhealthy() => {
                        let _ = guarded_tx
                            .send(GenerationEvent::Error(supervisor.unhealthy_message()))
                            .await;
                        return;
                    }
                    _ = guarded_tx.closed() => {
                        // Propagate the HTTP/SSE consumer's disconnect to the
                        // raw worker channel.  Keeping `source` alive here
                        // would hide cancellation from `events.is_closed()`
                        // and let a bounded prefill/decode continue after the
                        // client had already gone away.
                        return;
                    }
                    event = source.recv() => {
                        let Some(event) = event else {
                            return;
                        };
                        let terminal = matches!(
                            event,
                            GenerationEvent::Done { .. } | GenerationEvent::Error(_)
                        );
                        match guarded_tx.try_send(event) {
                            Ok(()) if terminal => return,
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Closed(_)) => return,
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                // The HTTP consumer is not keeping up. Never
                                // let its bounded response queue backpressure
                                // the sole model worker: dropping `source`
                                // wakes any raw blocking send with an error,
                                // which the worker treats as request-local
                                // cancellation while other slots continue.
                                return;
                            }
                        }
                    }
                }
            }
        });
        guarded_rx
    }

    pub(super) fn arm(
        &self,
        kind: &'static str,
        timeout: Duration,
    ) -> Result<WorkerTransactionLease> {
        anyhow::ensure!(self.is_healthy(), "{}", self.unhealthy_message());
        anyhow::ensure!(!timeout.is_zero(), "worker transaction timeout is zero");
        let epoch = self.inner.next_epoch.fetch_add(1, Ordering::Relaxed);
        let mut armed = self
            .inner
            .armed
            .lock()
            .map_err(|_| anyhow::anyhow!("engine supervisor transaction lock poisoned"))?;
        anyhow::ensure!(
            self.inner.health.load(Ordering::Acquire) == HEALTHY,
            "{}",
            self.unhealthy_message()
        );
        anyhow::ensure!(
            armed.is_none(),
            "engine supervisor already has an armed worker transaction"
        );
        *armed = Some(ArmedTransaction {
            epoch,
            kind,
            deadline: Instant::now() + timeout,
        });
        Ok(WorkerTransactionLease {
            supervisor: self.clone(),
            epoch,
            finished: false,
        })
    }

    pub(super) fn run<T>(
        &self,
        kind: &'static str,
        timeout: Duration,
        call: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let lease = self.arm(kind, timeout)?;
        let result = call();
        lease.finish()?;
        result
    }

    #[cfg(test)]
    pub(super) fn force_expire_for_test(&self) {
        let mut armed = self.inner.armed.lock().expect("supervisor lock");
        if let Some(transaction) = armed.as_mut() {
            transaction.deadline = Instant::now();
        }
        drop(armed);
        self.inner.expire_if_due(Instant::now());
    }

    #[cfg(test)]
    pub(super) fn backdate_without_monitor_for_test(&self) {
        let mut armed = self.inner.armed.lock().expect("supervisor lock");
        if let Some(transaction) = armed.as_mut() {
            transaction.deadline = Instant::now();
        }
    }

    pub(super) fn poison_now(&self, kind: &'static str) {
        let now = Instant::now();
        self.inner.poison_transaction(ArmedTransaction {
            epoch: self.inner.next_epoch.fetch_add(1, Ordering::Relaxed),
            kind,
            deadline: now,
        });
    }
}

impl WorkerTransactionLease {
    pub(super) fn finish(mut self) -> Result<()> {
        self.complete();
        anyhow::ensure!(
            self.supervisor.is_healthy(),
            "{}",
            self.supervisor.unhealthy_message()
        );
        Ok(())
    }

    fn complete(&mut self) {
        if self.finished {
            return;
        }
        self.supervisor
            .inner
            .complete_epoch(self.epoch, Instant::now());
        self.finished = true;
    }
}

impl Drop for WorkerTransactionLease {
    fn drop(&mut self) {
        self.complete();
    }
}

impl EngineSupervisorInner {
    fn complete_epoch(&self, epoch: u64, now: Instant) {
        let Ok(mut armed) = self.armed.lock() else {
            self.poison_transaction(ArmedTransaction {
                epoch,
                kind: "supervisor-lock",
                deadline: now,
            });
            return;
        };
        let Some(transaction) = armed
            .as_ref()
            .filter(|transaction| transaction.epoch == epoch)
            .cloned()
        else {
            return;
        };
        if now >= transaction.deadline {
            // Linearize readiness failure before clearing the lease. Otherwise
            // a concurrent readiness check could observe HEALTHY + no armed
            // transaction in the gap between disarm and poison.
            self.poison_transaction(transaction);
        }
        *armed = None;
    }

    fn expire_if_due(&self, now: Instant) {
        if self.health.load(Ordering::Acquire) != HEALTHY {
            return;
        }
        let expired = self
            .armed
            .lock()
            .ok()
            .and_then(|armed| armed.clone())
            .filter(|transaction| now >= transaction.deadline);
        let Some(expired) = expired else {
            return;
        };
        self.poison_transaction(expired);
    }

    fn poison_transaction(&self, expired: ArmedTransaction) {
        if self
            .health
            .compare_exchange(HEALTHY, POISONED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let message = format!(
                "{}: worker transaction {} (epoch {}) exceeded its continuously-awake deadline; process restart required",
                super::engine::ENGINE_UNHEALTHY_SENTINEL,
                expired.kind,
                expired.epoch
            );
            if let Ok(mut reason) = self.reason.lock() {
                *reason = Some(message.clone());
            }
            tracing::error!(
                transaction = expired.kind,
                epoch = expired.epoch,
                "engine worker transaction deadline expired"
            );
            self.health_tx.send_replace(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn expired_transaction_poison_is_one_way_and_not_receiver_closure() {
        let supervisor = EngineSupervisor::new();
        let lease = supervisor
            .arm("synthetic-gpu", Duration::from_secs(60))
            .unwrap();
        supervisor.force_expire_for_test();
        supervisor.wait_unhealthy().await;
        assert!(!supervisor.is_healthy());
        assert!(supervisor.unhealthy_message().contains("synthetic-gpu"));
        assert!(lease.finish().is_err(), "late success must not heal poison");
        assert!(supervisor
            .arm("later-gpu", Duration::from_secs(60))
            .is_err());
    }

    #[test]
    fn timely_completion_disarms_without_poisoning() {
        let supervisor = EngineSupervisor::new();
        supervisor
            .arm("synthetic-gpu", Duration::from_secs(60))
            .unwrap()
            .finish()
            .unwrap();
        assert!(supervisor.is_healthy());
        supervisor
            .arm("second-gpu", Duration::from_secs(60))
            .unwrap()
            .finish()
            .unwrap();
    }

    #[test]
    fn finish_observes_its_own_deadline_before_monitor_poll() {
        let supervisor = EngineSupervisor::new();
        let lease = supervisor
            .arm("deadline-race", Duration::from_secs(60))
            .unwrap();
        supervisor.backdate_without_monitor_for_test();
        assert!(lease.finish().is_err());
        assert!(!supervisor.is_healthy());
        assert!(supervisor.unhealthy_message().contains("deadline-race"));
    }

    #[tokio::test]
    async fn poisoned_stream_emits_one_error_and_ignores_late_done() {
        let supervisor = EngineSupervisor::new();
        let lease = supervisor
            .arm("synthetic-stream-gpu", Duration::from_secs(60))
            .unwrap();
        let (source_tx, source_rx) = mpsc::channel(4);
        let mut guarded = supervisor.guard_generation_events(source_rx);
        source_tx
            .send(GenerationEvent::Delta {
                kind: super::super::sse::DeltaKind::Content,
                text: "partial".into(),
            })
            .await
            .unwrap();
        assert!(matches!(
            guarded.recv().await,
            Some(GenerationEvent::Delta { .. })
        ));

        supervisor.force_expire_for_test();
        assert!(matches!(
            guarded.recv().await,
            Some(GenerationEvent::Error(message)) if message.contains("synthetic-stream-gpu")
        ));
        assert!(guarded.recv().await.is_none());
        assert!(lease.finish().is_err());
        assert!(source_tx
            .send(GenerationEvent::Done {
                finish_reason: "stop".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                stats: Default::default(),
            })
            .await
            .is_err());
    }

    #[tokio::test]
    async fn poison_wins_over_an_already_buffered_done_event() {
        let supervisor = EngineSupervisor::new();
        let lease = supervisor
            .arm("buffered-done-race", Duration::from_secs(60))
            .unwrap();
        let (source_tx, source_rx) = mpsc::channel(1);
        source_tx
            .send(GenerationEvent::Done {
                finish_reason: "stop".into(),
                prompt_tokens: 1,
                completion_tokens: 1,
                stats: Default::default(),
            })
            .await
            .unwrap();
        supervisor.backdate_without_monitor_for_test();
        assert!(lease.finish().is_err());

        let mut guarded = supervisor.guard_generation_events(source_rx);
        assert!(matches!(
            guarded.recv().await,
            Some(GenerationEvent::Error(message)) if message.contains("buffered-done-race")
        ));
        assert!(guarded.recv().await.is_none());
    }

    #[tokio::test]
    async fn guarded_receiver_drop_closes_the_worker_source() {
        let supervisor = EngineSupervisor::new();
        let (source_tx, source_rx) = mpsc::channel(1);
        let guarded = supervisor.guard_generation_events(source_rx);

        drop(guarded);
        tokio::time::timeout(Duration::from_secs(1), source_tx.closed())
            .await
            .expect("guard task must drop the raw worker receiver promptly");
        assert!(source_tx
            .send(GenerationEvent::Delta {
                kind: super::super::sse::DeltaKind::Content,
                text: "late".into(),
            })
            .await
            .is_err());
    }

    #[tokio::test]
    async fn guarded_backpressure_closes_the_worker_source() {
        let supervisor = EngineSupervisor::new();
        let (source_tx, source_rx) = mpsc::channel(128);
        let _guarded = supervisor.guard_generation_events(source_rx);

        for index in 0..65 {
            source_tx
                .send(GenerationEvent::Delta {
                    kind: super::super::sse::DeltaKind::Content,
                    text: index.to_string(),
                })
                .await
                .expect("raw bridge accepts events until guarded capacity is exceeded");
        }
        tokio::time::timeout(Duration::from_secs(1), source_tx.closed())
            .await
            .expect("guard task must drop the raw receiver on slow-client backpressure");
        assert!(supervisor.is_healthy(), "slow clients are request-local");
    }

    #[tokio::test]
    async fn monitor_spawn_failure_poison_is_immediate_and_observable() {
        let supervisor = EngineSupervisor::new_with_monitor_start(|_| {
            Err(std::io::Error::other("injected monitor spawn failure"))
        });
        assert!(!supervisor.is_healthy());
        supervisor.wait_unhealthy().await;
        assert!(supervisor
            .unhealthy_message()
            .contains("supervisor-monitor-spawn"));
        assert!(supervisor
            .arm("unobservable-gpu", Duration::from_secs(60))
            .is_err());
    }
}
