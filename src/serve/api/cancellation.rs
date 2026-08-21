//! Explicit cancellation and supervision for request-time preparation work.
//!
//! Generation has its own engine/channel cancellation path. This module owns
//! the earlier phase where a diagnostic request may be resolving metadata or
//! transferring one exact hosted artifact in a child `hf2q` process.

use std::panic::AssertUnwindSafe;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use futures::FutureExt;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::{oneshot, Notify, OwnedSemaphorePermit};

const CHILD_REAP_TIMEOUT: Duration = Duration::from_secs(10);
const PIPE_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug)]
pub struct PreparationLimits {
    pub stdout_bytes: usize,
    pub stderr_bytes: usize,
    pub deadline: Option<Duration>,
}

impl PreparationLimits {
    pub const fn catalog() -> Self {
        Self {
            stdout_bytes: 256 * 1024,
            stderr_bytes: 64 * 1024,
            deadline: Some(Duration::from_secs(30)),
        }
    }

    pub const fn transfer_receipt() -> Self {
        Self {
            stdout_bytes: 4 * 1024,
            stderr_bytes: 64 * 1024,
            deadline: None,
        }
    }
}

#[derive(Debug)]
struct CancellationInner {
    flag: Arc<AtomicBool>,
    notify: Notify,
}

/// Race-free cancellation signal shared by request, server-shutdown, and
/// supervised preparation tasks.
#[derive(Clone, Debug)]
pub struct CancellationSignal(Arc<CancellationInner>);

impl Default for CancellationSignal {
    fn default() -> Self {
        Self(Arc::new(CancellationInner {
            flag: Arc::new(AtomicBool::new(false)),
            notify: Notify::new(),
        }))
    }
}

impl CancellationSignal {
    pub fn cancel(&self) {
        if !self.0.flag.swap(true, Ordering::AcqRel) {
            self.0.notify.notify_waiters();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.flag.load(Ordering::Acquire)
    }

    /// Compatibility bridge for the existing pre-generation vision path.
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0.flag)
    }

    pub async fn cancelled(&self) {
        loop {
            let notified = self.0.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Debug)]
struct SupervisorInner {
    shutdown: CancellationSignal,
    active: AtomicUsize,
    idle: Notify,
    cleanup_failures: Mutex<Vec<String>>,
}

/// Server-wide owner for preparation children. The shutdown path cancels the
/// root before asking Axum to drain handlers, and can then wait for exact
/// child reaping.
#[derive(Clone, Debug)]
pub struct PreparationSupervisor(Arc<SupervisorInner>);

impl Default for PreparationSupervisor {
    fn default() -> Self {
        Self(Arc::new(SupervisorInner {
            shutdown: CancellationSignal::default(),
            active: AtomicUsize::new(0),
            idle: Notify::new(),
            cleanup_failures: Mutex::new(Vec::new()),
        }))
    }
}

impl PreparationSupervisor {
    pub fn cancel_all(&self) {
        self.0.shutdown.cancel();
    }

    pub fn shutdown_signal(&self) -> CancellationSignal {
        self.0.shutdown.clone()
    }

    fn begin(&self) -> Result<PreparationGuard, PreparationError> {
        if self.0.shutdown.is_cancelled() {
            return Err(PreparationError::ShuttingDown);
        }
        self.0.active.fetch_add(1, Ordering::AcqRel);
        if self.0.shutdown.is_cancelled() {
            self.finish_one();
            return Err(PreparationError::ShuttingDown);
        }
        Ok(PreparationGuard(self.clone()))
    }

    pub async fn wait_idle(&self, timeout: Duration) -> Result<(), String> {
        tokio::time::timeout(timeout, async {
            loop {
                let notified = self.0.idle.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.0.active.load(Ordering::Acquire) == 0 {
                    return;
                }
                notified.await;
            }
        })
        .await
        .map_err(|_| format!("preparation helpers did not exit within {timeout:?}"))?;
        let failures = self
            .0
            .cleanup_failures
            .lock()
            .map_err(|_| "preparation cleanup failure state is poisoned".to_owned())?;
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }

    fn finish_one(&self) {
        if self.0.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.idle.notify_waiters();
        }
    }

    fn record_cleanup_failure(&self, error: &PreparationError) {
        if let PreparationError::Cleanup(_)
        | PreparationError::ReapTimeout
        | PreparationError::TransactionFailed(_)
        | PreparationError::TransactionPanicked = error
        {
            if let Ok(mut failures) = self.0.cleanup_failures.lock() {
                failures.push(error.to_string());
            }
        }
    }
}

struct PreparationGuard(PreparationSupervisor);

impl Drop for PreparationGuard {
    fn drop(&mut self) {
        self.0.finish_one();
    }
}

#[derive(Debug)]
pub struct PreparationOutput {
    pub status: std::process::ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum PreparationError {
    #[error("cannot start preparation helper: {0}")]
    Spawn(std::io::Error),
    #[error("preparation helper I/O failed: {0}")]
    Io(std::io::Error),
    #[error("preparation helper was cancelled: {0}")]
    Cancelled(&'static str),
    #[error("preparation helper exceeded its {0:?} deadline")]
    TimedOut(Duration),
    #[error("preparation helper did not exit after cancellation")]
    ReapTimeout,
    #[error("preparation supervisor stopped before returning a receipt")]
    SupervisorStopped,
    #[error("server is shutting down; preparation was not started")]
    ShuttingDown,
    #[error("preparation helper cleanup failed: {0}")]
    Cleanup(String),
    #[error("preparation helper {stream} exceeded its {limit}-byte output limit")]
    OutputLimitExceeded { stream: &'static str, limit: usize },
    #[error("model lifecycle transaction requires restart: {0}")]
    TransactionFailed(String),
    #[error("model lifecycle transaction panicked; process restart required")]
    TransactionPanicked,
}

/// Spawn one helper in a detached supervisor task. The task, not the request
/// handler, owns the child until `wait()` has reaped it. Dropping the HTTP
/// handler therefore cancels through `request`, while this task remains alive
/// long enough to perform bounded cleanup. Hidden hf2q helpers must remain
/// direct-child-only: inheriting the server's verified process group lets an
/// owned chat's parent lifeline kill server and helper atomically, while an
/// external server can cancel and reap the exact child without signaling
/// unrelated processes.
pub async fn run_preparation_command(
    mut command: Command,
    request: CancellationSignal,
    supervisor: PreparationSupervisor,
    limits: PreparationLimits,
    permit: Option<OwnedSemaphorePermit>,
) -> Result<PreparationOutput, PreparationError> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let (receipt_tx, receipt_rx) = oneshot::channel();
    let shutdown = supervisor.shutdown_signal();
    let guard = supervisor.begin()?;
    let failure_sink = supervisor.clone();
    tokio::spawn(async move {
        let _guard = guard;
        let _permit = permit;
        let result = run_child(&mut command, &request, &shutdown, limits).await;
        if let Err(error) = &result {
            failure_sink.record_cleanup_failure(error);
        }
        let _ = receipt_tx.send(result);
    });
    receipt_rx
        .await
        .map_err(|_| PreparationError::SupervisorStopped)?
}

/// Keep an irreversible pool/load transaction alive to a consistent terminal
/// state even if the HTTP request that initiated it disconnects. New tasks are
/// rejected after server shutdown begins; already-registered tasks are part
/// of the supervisor's bounded shutdown wait.
pub async fn run_consistent_result_task<T, E, F, RestartRequired>(
    supervisor: PreparationSupervisor,
    future: F,
    failure_requires_restart: RestartRequired,
) -> Result<Result<T, E>, PreparationError>
where
    T: Send + 'static,
    E: std::fmt::Display + Send + 'static,
    F: std::future::Future<Output = Result<T, E>> + Send + 'static,
    RestartRequired: Fn(&E) -> bool + Send + 'static,
{
    let guard = supervisor.begin()?;
    let (receipt_tx, receipt_rx) = oneshot::channel();
    let failure_sink = supervisor.clone();
    tokio::spawn(async move {
        let _guard = guard;
        let result = match AssertUnwindSafe(future).catch_unwind().await {
            Ok(result) => {
                if let Err(error) = &result {
                    if failure_requires_restart(error) {
                        failure_sink.record_cleanup_failure(&PreparationError::TransactionFailed(
                            error.to_string(),
                        ));
                    }
                }
                Ok(result)
            }
            Err(_) => {
                let error = PreparationError::TransactionPanicked;
                tracing::error!(error = %error, "supervised model lifecycle transaction panicked");
                failure_sink.record_cleanup_failure(&error);
                Err(error)
            }
        };
        if let Err(undelivered) = receipt_tx.send(result) {
            match undelivered {
                Ok(Err(error)) => {
                    tracing::error!(error = %error, "detached model lifecycle transaction failed");
                }
                Err(error) => failure_sink.record_cleanup_failure(&error),
                Ok(Ok(_)) => {}
            }
        }
    });
    receipt_rx
        .await
        .map_err(|_| PreparationError::SupervisorStopped)?
}

async fn run_child(
    command: &mut Command,
    request: &CancellationSignal,
    shutdown: &CancellationSignal,
    limits: PreparationLimits,
) -> Result<PreparationOutput, PreparationError> {
    let mut child = command.spawn().map_err(PreparationError::Spawn)?;
    let stdout = child
        .stdout
        .take()
        .expect("piped stdout must exist for preparation helper");
    let stderr = child
        .stderr
        .take()
        .expect("piped stderr must exist for preparation helper");
    let stdout_task = tokio::spawn(read_bounded(stdout, limits.stdout_bytes, false));
    let stderr_task = tokio::spawn(read_bounded(stderr, limits.stderr_bytes, true));

    enum Finish {
        Exited(Result<std::process::ExitStatus, std::io::Error>),
        Cancelled(&'static str),
        TimedOut(Duration),
    }
    let deadline_wait = async {
        match limits.deadline {
            Some(duration) => tokio::time::sleep(duration).await,
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(deadline_wait);
    let finish = tokio::select! {
        biased;
        _ = request.cancelled() => Finish::Cancelled("request disconnected"),
        _ = shutdown.cancelled() => Finish::Cancelled("server is shutting down"),
        _ = &mut deadline_wait => Finish::TimedOut(limits.deadline.expect("deadline future completed")),
        status = child.wait() => Finish::Exited(status),
    };
    let status = match finish {
        Finish::Exited(Ok(status)) => status,
        Finish::Exited(Err(wait_error)) => {
            let termination = terminate_and_reap(&mut child).await;
            let drainage = drain_after_termination(stdout_task, stderr_task).await;
            let mut detail = format!("initial child wait failed: {wait_error}");
            if let Err(error) = termination {
                detail.push_str(&format!("; termination/reap failed: {error}"));
            }
            if let Err(error) = drainage {
                detail.push_str(&format!("; pipe cleanup failed: {error}"));
            }
            return Err(PreparationError::Cleanup(detail));
        }
        Finish::Cancelled(reason) => {
            terminate_and_reap(&mut child).await?;
            drain_after_termination(stdout_task, stderr_task).await?;
            return Err(PreparationError::Cancelled(reason));
        }
        Finish::TimedOut(duration) => {
            terminate_and_reap(&mut child).await?;
            drain_after_termination(stdout_task, stderr_task).await?;
            return Err(PreparationError::TimedOut(duration));
        }
    };
    let (stdout, stderr) = collect_output(stdout_task, stderr_task).await?;
    if stdout.truncated {
        return Err(PreparationError::OutputLimitExceeded {
            stream: "stdout",
            limit: limits.stdout_bytes,
        });
    }
    Ok(PreparationOutput {
        status,
        stdout: stdout.bytes,
        stderr: stderr.bytes,
    })
}

async fn terminate_and_reap(child: &mut tokio::process::Child) -> Result<(), PreparationError> {
    let kill_error = match child.start_kill() {
        Ok(()) => None,
        Err(error) if error.kind() == std::io::ErrorKind::InvalidInput => None,
        Err(error) => Some(error),
    };
    let wait_result = tokio::time::timeout(CHILD_REAP_TIMEOUT, child.wait())
        .await
        .map_err(|_| PreparationError::ReapTimeout)?;
    match (kill_error, wait_result) {
        (None, Ok(_)) => Ok(()),
        (Some(kill), Ok(_)) => Err(PreparationError::Cleanup(format!(
            "signal failed ({kill}) although the child was reaped"
        ))),
        (None, Err(wait)) => Err(PreparationError::Cleanup(format!("wait failed: {wait}"))),
        (Some(kill), Err(wait)) => Err(PreparationError::Cleanup(format!(
            "signal failed ({kill}) and wait failed ({wait})"
        ))),
    }
}

#[derive(Debug)]
struct BoundedRead {
    bytes: Vec<u8>,
    truncated: bool,
}

async fn read_bounded(
    mut reader: impl AsyncRead + Unpin,
    limit: usize,
    retain_tail: bool,
) -> Result<BoundedRead, PreparationError> {
    let mut retained = Vec::new();
    let mut truncated = false;
    let mut chunk = [0_u8; 8192];
    loop {
        let read = reader
            .read(&mut chunk)
            .await
            .map_err(PreparationError::Io)?;
        if read == 0 {
            return Ok(BoundedRead {
                bytes: retained,
                truncated,
            });
        }
        if !retain_tail && retained.len() < limit {
            let keep = read.min(limit - retained.len());
            retained.extend_from_slice(&chunk[..keep]);
            truncated |= keep < read;
        } else if retain_tail {
            retained.extend_from_slice(&chunk[..read]);
            if retained.len() > limit {
                let excess = retained.len() - limit;
                retained.drain(..excess);
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }
}

async fn drain_after_termination(
    stdout: tokio::task::JoinHandle<Result<BoundedRead, PreparationError>>,
    stderr: tokio::task::JoinHandle<Result<BoundedRead, PreparationError>>,
) -> Result<(), PreparationError> {
    let _ = collect_output(stdout, stderr).await?;
    Ok(())
}

async fn collect_output(
    mut stdout: tokio::task::JoinHandle<Result<BoundedRead, PreparationError>>,
    mut stderr: tokio::task::JoinHandle<Result<BoundedRead, PreparationError>>,
) -> Result<(BoundedRead, BoundedRead), PreparationError> {
    match tokio::time::timeout(PIPE_DRAIN_TIMEOUT, async {
        let stdout = (&mut stdout)
            .await
            .map_err(|_| PreparationError::SupervisorStopped)??;
        let stderr = (&mut stderr)
            .await
            .map_err(|_| PreparationError::SupervisorStopped)??;
        Ok::<_, PreparationError>((stdout, stderr))
    })
    .await
    {
        Ok(result) => result,
        Err(_) => {
            stdout.abort();
            stderr.abort();
            Err(PreparationError::Cleanup(
                "helper descendants retained stdout/stderr after helper exit".to_owned(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelled_helper_is_killed_reaped_and_supervisor_becomes_idle() {
        let supervisor = PreparationSupervisor::default();
        let request = CancellationSignal::default();
        let mut command = Command::new("sh");
        command.args(["-c", "exec sleep 30"]);
        let task = tokio::spawn(run_preparation_command(
            command,
            request.clone(),
            supervisor.clone(),
            PreparationLimits::transfer_receipt(),
            None,
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        request.cancel();
        let error = task.await.unwrap().unwrap_err();
        assert!(matches!(error, PreparationError::Cancelled(_)));
        supervisor.wait_idle(Duration::from_secs(1)).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn server_shutdown_cancels_helper_before_http_drain_budget() {
        let supervisor = PreparationSupervisor::default();
        let mut command = Command::new("sh");
        command.args(["-c", "exec sleep 30"]);
        let task = tokio::spawn(run_preparation_command(
            command,
            CancellationSignal::default(),
            supervisor.clone(),
            PreparationLimits::transfer_receipt(),
            None,
        ));
        tokio::time::sleep(Duration::from_millis(50)).await;
        supervisor.cancel_all();
        let error = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("shutdown cancellation must not wait for HTTP drain")
            .unwrap()
            .unwrap_err();
        assert!(matches!(error, PreparationError::Cancelled(_)));
        supervisor.wait_idle(Duration::from_secs(1)).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancelled_root_refuses_to_spawn_a_new_helper() {
        let temp = tempfile::tempdir().unwrap();
        let marker = temp.path().join("spawned");
        let supervisor = PreparationSupervisor::default();
        supervisor.cancel_all();
        let mut command = Command::new("sh");
        command.args(["-c", &format!("touch {}", marker.display())]);
        let error = run_preparation_command(
            command,
            CancellationSignal::default(),
            supervisor,
            PreparationLimits::transfer_receipt(),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, PreparationError::ShuttingDown));
        assert!(!marker.exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn stdout_overflow_is_rejected_instead_of_truncated_into_a_receipt() {
        let mut command = Command::new("sh");
        command.args(["-c", "yes x | head -c 4096"]);
        let error = run_preparation_command(
            command,
            CancellationSignal::default(),
            PreparationSupervisor::default(),
            PreparationLimits {
                stdout_bytes: 128,
                stderr_bytes: 128,
                deadline: Some(Duration::from_secs(2)),
            },
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(
            error,
            PreparationError::OutputLimitExceeded {
                stream: "stdout",
                limit: 128
            }
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn descendant_retaining_pipes_is_recorded_as_cleanup_failure() {
        let temp = tempfile::tempdir().unwrap();
        let pid_file = temp.path().join("descendant.pid");
        let supervisor = PreparationSupervisor::default();
        let mut command = Command::new("sh");
        command
            .args([
                "-c",
                "sleep 30 & echo $! > \"$HF2Q_TEST_DESCENDANT_PID_FILE\"; exit 0",
            ])
            .env("HF2Q_TEST_DESCENDANT_PID_FILE", &pid_file);
        let error = run_preparation_command(
            command,
            CancellationSignal::default(),
            supervisor.clone(),
            PreparationLimits::transfer_receipt(),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, PreparationError::Cleanup(_)));
        assert!(error.to_string().contains("retained stdout/stderr"));

        let descendant: libc::pid_t = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        unsafe {
            libc::kill(descendant, libc::SIGKILL);
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            let result = unsafe { libc::kill(descendant, 0) };
            if result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "fixture descendant was not reaped after explicit test cleanup"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let shutdown_error = supervisor
            .wait_idle(Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(shutdown_error.contains("retained stdout/stderr"));
    }

    #[tokio::test]
    async fn disconnected_waiter_does_not_cancel_a_consistency_transaction() {
        let supervisor = PreparationSupervisor::default();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let completed = Arc::new(AtomicBool::new(false));
        let future_started = Arc::clone(&started);
        let future_release = Arc::clone(&release);
        let future_completed = Arc::clone(&completed);
        let task = tokio::spawn(run_consistent_result_task(
            supervisor.clone(),
            async move {
                future_started.notify_one();
                future_release.notified().await;
                future_completed.store(true, Ordering::Release);
                Ok::<_, &'static str>(())
            },
            |_| true,
        ));

        started.notified().await;
        task.abort();
        let _ = task.await;
        release.notify_one();

        supervisor.wait_idle(Duration::from_secs(1)).await.unwrap();
        assert!(completed.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn detached_restart_required_transaction_failure_is_recorded() {
        let supervisor = PreparationSupervisor::default();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let future_started = Arc::clone(&started);
        let future_release = Arc::clone(&release);
        let task = tokio::spawn(run_consistent_result_task(
            supervisor.clone(),
            async move {
                future_started.notify_one();
                future_release.notified().await;
                Err::<(), _>("restart required")
            },
            |_| true,
        ));

        started.notified().await;
        task.abort();
        let _ = task.await;
        release.notify_one();

        let error = supervisor
            .wait_idle(Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(error.contains("model lifecycle transaction requires restart"));
        assert!(error.contains("restart required"));
    }

    #[tokio::test]
    async fn delivered_restart_required_transaction_failure_is_still_recorded() {
        let supervisor = PreparationSupervisor::default();
        let result = run_consistent_result_task(
            supervisor.clone(),
            async { Err::<(), _>("restart required") },
            |_| true,
        )
        .await
        .unwrap();
        assert_eq!(result.unwrap_err(), "restart required");
        let error = supervisor
            .wait_idle(Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(error.contains("model lifecycle transaction requires restart"));
        assert!(error.contains("restart required"));
    }

    #[tokio::test]
    async fn consistency_transaction_panic_is_fail_closed() {
        let supervisor = PreparationSupervisor::default();
        let error = run_consistent_result_task(
            supervisor.clone(),
            async move {
                panic!("transaction fixture panic");
                #[allow(unreachable_code)]
                Ok::<(), &'static str>(())
            },
            |_| true,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, PreparationError::TransactionPanicked));
        let shutdown_error = supervisor
            .wait_idle(Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(shutdown_error.contains("transaction panicked"));
    }

    #[tokio::test]
    async fn shutdown_wait_refuses_to_report_idle_while_transaction_is_active() {
        let supervisor = PreparationSupervisor::default();
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let future_started = Arc::clone(&started);
        let future_release = Arc::clone(&release);
        let task = tokio::spawn(run_consistent_result_task(
            supervisor.clone(),
            async move {
                future_started.notify_one();
                future_release.notified().await;
                Ok::<_, &'static str>(())
            },
            |_| false,
        ));

        started.notified().await;
        supervisor.cancel_all();
        let error = supervisor
            .wait_idle(Duration::from_millis(10))
            .await
            .unwrap_err();
        assert!(error.contains("did not exit"));
        release.notify_one();
        assert!(task.await.unwrap().unwrap().is_ok());
        supervisor.wait_idle(Duration::from_secs(1)).await.unwrap();
    }
}
