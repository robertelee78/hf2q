use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Child;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};

use crate::cli::ChatArgs;

// hf2q may spend up to 30 seconds draining KV work before worker teardown.
// Leave bounded headroom so a healthy graceful exit is not reported as a
// failure by the diagnostic client.
const CHILD_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(45);
const CHILD_SIGNAL_TIMEOUT: Duration = Duration::from_secs(5);
const SHUTDOWN_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// A normalized HTTP endpoint. Automatic discovery must construct this from
/// a verified loopback port, never from DNS-SD TXT host/URL/PID metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Endpoint {
    base_url: String,
}

impl Endpoint {
    pub(crate) fn explicit(value: &str) -> Result<Self> {
        let parsed = reqwest::Url::parse(value).context("parse --url")?;
        if !matches!(parsed.scheme(), "http" | "https") {
            bail!("--url must use http or https");
        }
        if parsed.host_str().is_none() {
            bail!("--url must include a host");
        }
        if !parsed.username().is_empty() || parsed.password().is_some() {
            bail!("--url must not contain embedded credentials");
        }
        if parsed.query().is_some() || parsed.fragment().is_some() {
            bail!("--url must not include a query string or fragment");
        }
        let base_url = parsed.as_str().trim_end_matches('/').to_owned();
        Ok(Self { base_url })
    }

    /// Security boundary for a locally discovered service. The caller may
    /// supply only the resolved port; the host is forced to IPv4 loopback.
    pub(crate) fn discovered_loopback(port: u16) -> Self {
        Self {
            base_url: format!("http://127.0.0.1:{port}"),
        }
    }

    pub(crate) fn base_url(&self) -> &str {
        &self.base_url
    }

    pub(crate) fn route(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }
}

#[derive(Debug)]
enum Ownership {
    External,
    Owned {
        process: OwnedServerProcess,
        detached: bool,
    },
}

const GROUP_EXTINCTION_TIMEOUT: Duration = Duration::from_secs(5);

#[cfg(unix)]
pub(crate) type ParentLifeline = std::os::unix::net::UnixStream;
#[cfg(not(unix))]
pub(crate) type ParentLifeline = std::process::ChildStdin;

struct ServerLog {
    temporary: Option<tempfile::NamedTempFile>,
    retained: Option<PathBuf>,
}

impl ServerLog {
    fn new(temporary: tempfile::NamedTempFile) -> Self {
        Self {
            temporary: Some(temporary),
            retained: None,
        }
    }

    fn path(&self) -> &Path {
        self.retained.as_deref().unwrap_or_else(|| {
            self.temporary
                .as_ref()
                .expect("owned server log must have a path")
                .path()
        })
    }

    fn retain(&mut self) -> Result<PathBuf> {
        if let Some(path) = &self.retained {
            return Ok(path.clone());
        }
        let temporary = self
            .temporary
            .take()
            .context("chat-owned server log was unavailable")?;
        let (_file, path) = match temporary.keep() {
            Ok(retained) => retained,
            Err(error) => {
                self.temporary = Some(error.file);
                return Err(error.error).context("retain hf2q server diagnostic log");
            }
        };
        self.retained = Some(path.clone());
        Ok(path)
    }
}

#[derive(Debug)]
pub(crate) struct OwnedFailureDiagnostics {
    pub(crate) path: PathBuf,
}

/// Concrete process authority created only from the child spawned by chat.
/// The process group, parent-lifetime writer, and private server log remain
/// inseparable so cleanup cannot accidentally degrade to PID-only signaling.
pub(crate) struct OwnedServerProcess {
    child: Child,
    lifeline: Option<ParentLifeline>,
    #[cfg(unix)]
    readiness_frame: Vec<u8>,
    #[cfg(unix)]
    process_group: libc::pid_t,
    #[cfg(unix)]
    listener_guard: Option<std::net::TcpListener>,
    #[cfg(unix)]
    startup_progress: Option<std::os::unix::net::UnixDatagram>,
    server_log: ServerLog,
}

impl std::fmt::Debug for OwnedServerProcess {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("OwnedServerProcess");
        debug.field("pid", &self.child.id());
        #[cfg(unix)]
        debug.field("process_group", &self.process_group);
        debug
            .field("server_log", &self.server_log.path())
            .finish_non_exhaustive()
    }
}

impl OwnedServerProcess {
    pub(crate) fn from_spawned(
        mut child: Child,
        lifeline: ParentLifeline,
        #[cfg(unix)] startup_progress: Option<std::os::unix::net::UnixDatagram>,
        server_log: tempfile::NamedTempFile,
        #[cfg(unix)] listener_guard: Option<std::net::TcpListener>,
    ) -> Result<Self> {
        #[cfg(unix)]
        let process_group = child.id() as libc::pid_t;
        #[cfg(unix)]
        {
            let actual = unsafe { libc::getpgid(process_group) };
            if actual != process_group {
                let _ = child.kill();
                let _ = child.wait();
                bail!(
                    "chat-owned server did not enter its isolated process group (pid={process_group}, pgrp={actual})"
                );
            }
            lifeline
                .set_nonblocking(true)
                .context("make chat-owned readiness channel nonblocking")?;
            if let Some(progress) = startup_progress.as_ref() {
                progress
                    .set_nonblocking(true)
                    .context("make chat-owned startup-progress channel nonblocking")?;
            }
        }
        Ok(Self {
            child,
            lifeline: Some(lifeline),
            #[cfg(unix)]
            readiness_frame: Vec::new(),
            #[cfg(unix)]
            process_group,
            #[cfg(unix)]
            listener_guard,
            #[cfg(unix)]
            startup_progress,
            server_log: ServerLog::new(server_log),
        })
    }

    pub(crate) fn child_mut(&mut self) -> &mut Child {
        &mut self.child
    }

    pub(crate) fn force_stop(&mut self) -> Result<()> {
        force_stop_owned_process(self)
    }

    pub(crate) fn leader_exited_unreaped(&mut self) -> Result<bool> {
        leader_exited_unreaped(&mut self.child)
    }

    /// Read the bound loopback port from the private inherited Unix socket.
    /// DNS-SD PID/TXT hints are deliberately not accepted as child identity.
    #[cfg(unix)]
    pub(crate) fn poll_ready_port(&mut self) -> Result<Option<u16>> {
        use std::io::Read;

        const MAX_FRAME_BYTES: usize = 64;
        let Some(lifeline) = self.lifeline.as_mut() else {
            bail!("chat-owned readiness channel is no longer available");
        };
        let mut chunk = [0_u8; MAX_FRAME_BYTES];
        match lifeline.read(&mut chunk) {
            Ok(0) => bail!("chat-owned server closed its readiness channel before READY"),
            Ok(read) => self.readiness_frame.extend_from_slice(&chunk[..read]),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(error) => return Err(error).context("read chat-owned READY frame"),
        }
        if self.readiness_frame.len() > MAX_FRAME_BYTES {
            bail!("chat-owned READY frame exceeds the bounded protocol size");
        }
        let Some(newline) = self.readiness_frame.iter().position(|byte| *byte == b'\n') else {
            return Ok(None);
        };
        if newline + 1 != self.readiness_frame.len() {
            bail!("chat-owned readiness channel sent trailing data");
        }
        let frame = std::str::from_utf8(&self.readiness_frame[..newline])
            .context("chat-owned READY frame is not UTF-8")?;
        let port = frame
            .strip_prefix(crate::serve::CHAT_LIFELINE_READY_PREFIX)
            .context("chat-owned readiness channel sent an invalid frame")?
            .parse::<u16>()
            .context("chat-owned READY frame has an invalid port")?;
        if port == 0 {
            bail!("chat-owned READY frame reported port 0");
        }
        if let Some(listener) = self.listener_guard.as_ref() {
            let expected = listener
                .local_addr()
                .context("read retained chat-owned listener address")?
                .port();
            if port != expected {
                bail!(
                    "chat-owned READY port {port} does not match retained listener port {expected}"
                );
            }
        }
        self.readiness_frame.clear();
        Ok(Some(port))
    }

    /// Receive one best-effort progress datagram. Progress is informational:
    /// it never carries endpoint authority and malformed frames are ignored.
    #[cfg(unix)]
    pub(crate) fn poll_startup_event(
        &self,
    ) -> Option<crate::serve::startup_progress::StartupEvent> {
        let socket = self.startup_progress.as_ref()?;
        let mut bytes = [0_u8; crate::serve::CHAT_STARTUP_PROGRESS_MAX_BYTES + 1];
        let read = match socket.recv(&mut bytes) {
            Ok(read) => read,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => return None,
            Err(error) => {
                tracing::debug!(%error, "ignored chat-owned startup-progress receive failure");
                return None;
            }
        };
        if read == 0 || read > crate::serve::CHAT_STARTUP_PROGRESS_MAX_BYTES {
            return None;
        }
        let payload = bytes[..read].strip_prefix(crate::serve::CHAT_STARTUP_PROGRESS_PREFIX)?;
        serde_json::from_slice::<crate::serve::startup_progress::StartupEvent>(payload)
            .ok()
            .filter(crate::serve::startup_progress::StartupEvent::wire_valid)
    }

    #[cfg(not(unix))]
    pub(crate) fn poll_startup_event(
        &self,
    ) -> Option<crate::serve::startup_progress::StartupEvent> {
        None
    }

    fn detach_lifeline(&mut self) -> Result<PathBuf> {
        // Persist the log while lifecycle ownership is still intact. Once the
        // detach frame is accepted the server cannot safely be reclaimed on a
        // later filesystem error.
        let log_path = self.server_log.retain()?;
        let Some(lifeline) = self.lifeline.as_mut() else {
            return Ok(log_path);
        };
        lifeline
            .write_all(crate::serve::CHAT_LIFELINE_DETACH_FRAME)
            .context("detach chat-owned server parent lifetime")?;
        lifeline.flush().context("flush detach lifetime message")?;
        self.lifeline.take();
        Ok(log_path)
    }

    pub(crate) fn retain_log(&mut self) -> Result<PathBuf> {
        self.server_log.retain()
    }

    fn disarm_after_terminal_cleanup(&mut self) {
        self.lifeline.take();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EndpointKind {
    Explicit,
    DiscoveredHf2q,
}

/// Endpoint plus the only process authority chat is allowed to exercise.
/// PID metadata is deliberately absent; shutdown authority requires the
/// concrete Child handle created by this chat invocation.
#[derive(Debug)]
pub(crate) struct EndpointSession {
    endpoint: Endpoint,
    kind: EndpointKind,
    ownership: Ownership,
}

impl EndpointSession {
    pub(crate) fn external(endpoint: Endpoint) -> Self {
        Self {
            endpoint,
            kind: EndpointKind::Explicit,
            ownership: Ownership::External,
        }
    }

    pub(crate) fn discovered_hf2q(endpoint: Endpoint) -> Self {
        Self {
            endpoint,
            kind: EndpointKind::DiscoveredHf2q,
            ownership: Ownership::External,
        }
    }

    /// Integration seam for the automatic discovery/launcher lane.
    #[allow(dead_code)]
    pub(crate) fn spawned_loopback(port: u16, process: OwnedServerProcess) -> Self {
        Self {
            endpoint: Endpoint::discovered_loopback(port),
            kind: EndpointKind::DiscoveredHf2q,
            ownership: Ownership::Owned {
                process,
                detached: false,
            },
        }
    }

    pub(crate) fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    pub(crate) fn is_owned(&self) -> bool {
        matches!(self.ownership, Ownership::Owned { .. })
    }

    pub(crate) fn expects_hf2q_control(&self) -> bool {
        self.kind == EndpointKind::DiscoveredHf2q
    }

    pub(crate) fn is_detached(&self) -> bool {
        matches!(self.ownership, Ownership::Owned { detached: true, .. })
    }

    pub(crate) fn detach(&mut self) -> Result<Option<PathBuf>> {
        match &mut self.ownership {
            Ownership::External => Ok(None),
            Ownership::Owned { process, detached } => {
                let log_path = process.detach_lifeline()?;
                *detached = true;
                Ok(Some(log_path))
            }
        }
    }

    /// Persist the private server log for an operation that is already
    /// failing. Log contents are deliberately not returned: arbitrary loader
    /// context may contain local paths or credentials and must not be copied
    /// into terminal errors automatically. This changes no process authority.
    pub(crate) fn retain_failure_diagnostics(&mut self) -> Result<Option<OwnedFailureDiagnostics>> {
        let Ownership::Owned { process, .. } = &mut self.ownership else {
            return Ok(None);
        };
        let path = process.server_log.retain()?;
        Ok(Some(OwnedFailureDiagnostics { path }))
    }

    pub(crate) async fn shutdown_if_owned(
        &mut self,
        http: &reqwest::Client,
        auth_token: Option<&str>,
    ) -> Result<()> {
        self.shutdown_if_owned_with_timeouts(
            http,
            auth_token,
            CHILD_SHUTDOWN_TIMEOUT,
            CHILD_SIGNAL_TIMEOUT,
        )
        .await
    }

    async fn shutdown_if_owned_with_timeouts(
        &mut self,
        http: &reqwest::Client,
        auth_token: Option<&str>,
        graceful_timeout: Duration,
        signal_timeout: Duration,
    ) -> Result<()> {
        let Ownership::Owned { process, detached } = &mut self.ownership else {
            return Ok(());
        };
        if *detached || process.leader_exited_unreaped()? {
            if !*detached {
                stop_remaining_owned_descendants(process)?;
                process
                    .child_mut()
                    .wait()
                    .context("reap exited chat-owned server leader")?;
                process.disarm_after_terminal_cleanup();
                #[cfg(unix)]
                wait_for_group_extinction(process.process_group)?;
            }
            return Ok(());
        }

        let mut request = http.post(self.endpoint.route("/shutdown"));
        if let Some(token) = auth_token {
            request = request.bearer_auth(token);
        }
        let response = tokio::time::timeout(SHUTDOWN_REQUEST_TIMEOUT, request.send()).await;
        let graceful_requested = match response {
            Ok(Ok(response)) if response.status().is_success() => true,
            Ok(Ok(response)) => {
                tracing::warn!(
                    status = %response.status(),
                    "chat-owned server rejected /shutdown; signaling the owned child"
                );
                false
            }
            Ok(Err(error)) => {
                tracing::warn!(
                    error = %error,
                    "chat-owned server /shutdown failed; signaling the owned child"
                );
                false
            }
            Err(_) => {
                tracing::warn!(
                    timeout = ?SHUTDOWN_REQUEST_TIMEOUT,
                    "chat-owned server /shutdown timed out; signaling the owned child"
                );
                false
            }
        };
        if !graceful_requested {
            signal_owned_process(process, termination_signal())?;
            if wait_for_owned_process(process, graceful_timeout).await? {
                return Ok(());
            }
            tracing::warn!("chat-owned server ignored SIGTERM; force-stopping the owned child");
            return force_stop_owned_process(process);
        }

        if wait_for_owned_process(process, graceful_timeout).await? {
            return Ok(());
        }
        tracing::warn!(
            "chat-owned server exceeded graceful drain deadline; signaling the owned child"
        );
        signal_owned_process(process, termination_signal())?;
        if wait_for_owned_process(process, signal_timeout).await? {
            return Ok(());
        }
        tracing::warn!(
            "chat-owned server ignored graceful shutdown and SIGTERM; force-stopping the owned child"
        );
        force_stop_owned_process(process)
    }
}

#[cfg(unix)]
fn leader_exited_unreaped(child: &mut Child) -> Result<bool> {
    use rustix::process::{waitid, Pid, WaitId, WaitIdOptions};

    let pid = Pid::from_child(child);
    let status = waitid(
        WaitId::Pid(pid),
        WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
    )
    .context("peek chat-owned server leader without reaping")?;
    Ok(status.is_some())
}

#[cfg(not(unix))]
fn leader_exited_unreaped(child: &mut Child) -> Result<bool> {
    Ok(child
        .try_wait()
        .context("check chat-owned server leader")?
        .is_some())
}

async fn wait_for_owned_process(
    process: &mut OwnedServerProcess,
    timeout: Duration,
) -> Result<bool> {
    let deadline = Instant::now() + timeout;
    loop {
        if process.leader_exited_unreaped()? {
            stop_remaining_owned_descendants(process)?;
            process
                .child_mut()
                .wait()
                .context("reap chat-owned server leader")?;
            process.disarm_after_terminal_cleanup();
            #[cfg(unix)]
            wait_for_group_extinction(process.process_group)?;
            return Ok(true);
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(Duration::from_millis(50).min(deadline - now)).await;
    }
}

#[cfg(unix)]
fn signal_owned_group(process_group: libc::pid_t, signal: libc::c_int) -> Result<bool> {
    // The process group comes only from the concrete child chat spawned with
    // setpgid(0,0). Discovery metadata is never accepted as authority.
    let result = unsafe { libc::kill(-process_group, signal) };
    if result == 0 {
        Ok(true)
    } else {
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH)
            // Darwin may report EPERM when a verified owned group contains
            // only its unreaped zombie leader. There is no signalable member;
            // the caller reaps that leader before proving group extinction.
            || (signal == libc::SIGKILL && error.raw_os_error() == Some(libc::EPERM))
        {
            Ok(false)
        } else {
            Err(error).context("signal chat-owned server process group")
        }
    }
}

#[cfg(unix)]
fn termination_signal() -> libc::c_int {
    libc::SIGTERM
}

#[cfg(not(unix))]
fn termination_signal() -> i32 {
    0
}

#[cfg(unix)]
fn signal_owned_process(process: &mut OwnedServerProcess, signal: libc::c_int) -> Result<()> {
    signal_owned_group(process.process_group, signal).map(|_| ())
}

#[cfg(not(unix))]
fn signal_owned_process(process: &mut OwnedServerProcess, _signal: i32) -> Result<()> {
    process.child.kill().context("stop chat-owned server")
}

#[cfg(unix)]
fn owned_group_exists(process_group: libc::pid_t) -> Result<bool> {
    let result = unsafe { libc::kill(-process_group, 0) };
    if result == 0 {
        return Ok(true);
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        // POSIX defines EPERM from signal 0 as proof that at least one
        // process exists but is not signalable. Darwin can transiently
        // report this while a killed orphan is awaiting its new reaper.
        // Keep polling for ESRCH; never mistake EPERM for extinction.
        Some(libc::EPERM) => Ok(true),
        _ => Err(error).context("probe chat-owned server process group"),
    }
}

#[cfg(unix)]
fn wait_for_group_extinction(process_group: libc::pid_t) -> Result<()> {
    let deadline = Instant::now() + GROUP_EXTINCTION_TIMEOUT;
    while owned_group_exists(process_group)? {
        if Instant::now() >= deadline {
            bail!("chat-owned process group {process_group} did not terminate");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}

#[cfg(unix)]
fn stop_remaining_owned_descendants(process: &mut OwnedServerProcess) -> Result<()> {
    signal_owned_group(process.process_group, libc::SIGKILL)?;
    Ok(())
}

#[cfg(not(unix))]
fn stop_remaining_owned_descendants(_process: &mut OwnedServerProcess) -> Result<()> {
    Ok(())
}

fn force_stop_owned_process(process: &mut OwnedServerProcess) -> Result<()> {
    let log_path = process.server_log.path().display().to_string();
    force_stop_owned_process_inner(process)
        .with_context(|| format!("force-stop chat-owned server; private log={log_path}"))
}

fn force_stop_owned_process_inner(process: &mut OwnedServerProcess) -> Result<()> {
    #[cfg(unix)]
    signal_owned_group(process.process_group, libc::SIGKILL)?;
    #[cfg(not(unix))]
    process.child.kill().context("kill chat-owned server")?;
    process
        .child
        .wait()
        .context("reap force-stopped chat-owned server")?;
    process.disarm_after_terminal_cleanup();
    #[cfg(unix)]
    wait_for_group_extinction(process.process_group)?;
    Ok(())
}

impl Drop for OwnedServerProcess {
    fn drop(&mut self) {
        if self.lifeline.is_none() {
            return;
        }
        #[cfg(unix)]
        let _ = signal_owned_group(self.process_group, libc::SIGKILL);
        #[cfg(not(unix))]
        let _ = self.child.kill();
        let _ = self.child.wait();
        #[cfg(unix)]
        let _ = wait_for_group_extinction(self.process_group);
    }
}

pub(crate) trait EndpointResolver {
    fn resolve(&mut self, args: &ChatArgs) -> Result<EndpointSession>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn owned_sleep_process(command: &str) -> OwnedServerProcess {
        owned_sleep_process_with_log(command, tempfile::NamedTempFile::new().unwrap())
    }

    #[cfg(unix)]
    fn owned_sleep_process_with_log(
        command: &str,
        server_log: tempfile::NamedTempFile,
    ) -> OwnedServerProcess {
        use std::io::Read;
        use std::os::unix::process::CommandExt;
        let mut command_builder = std::process::Command::new("sh");
        command_builder
            .args(["-c", command])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .process_group(0);
        let (lifeline, mut peer) = std::os::unix::net::UnixStream::pair().unwrap();
        std::thread::spawn(move || {
            let mut frame = vec![0u8; crate::serve::CHAT_LIFELINE_DETACH_FRAME.len()];
            let _ = peer.read_exact(&mut frame);
        });
        OwnedServerProcess::from_spawned(
            command_builder.spawn().unwrap(),
            lifeline,
            None,
            server_log,
            None,
        )
        .unwrap()
    }

    #[cfg(unix)]
    fn owned_sleep_process_with_progress() -> (OwnedServerProcess, std::os::unix::net::UnixDatagram)
    {
        use std::io::Read;
        use std::os::unix::process::CommandExt;

        let mut command = std::process::Command::new("sh");
        command
            .args(["-c", "exec sleep 30"])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .process_group(0);
        let (lifeline, mut lifeline_peer) = std::os::unix::net::UnixStream::pair().unwrap();
        std::thread::spawn(move || {
            let mut frame = vec![0u8; crate::serve::CHAT_LIFELINE_DETACH_FRAME.len()];
            let _ = lifeline_peer.read_exact(&mut frame);
        });
        let (progress, progress_peer) = std::os::unix::net::UnixDatagram::pair().unwrap();
        let process = OwnedServerProcess::from_spawned(
            command.spawn().unwrap(),
            lifeline,
            Some(progress),
            tempfile::NamedTempFile::new().unwrap(),
            None,
        )
        .unwrap();
        (process, progress_peer)
    }

    #[test]
    fn discovered_endpoints_are_forced_to_loopback_and_have_no_pid_authority() {
        let endpoint = Endpoint::discovered_loopback(9123);
        assert_eq!(endpoint.base_url(), "http://127.0.0.1:9123");
        assert_eq!(
            endpoint.route("/v1/models"),
            "http://127.0.0.1:9123/v1/models"
        );
    }

    #[cfg(unix)]
    #[test]
    fn retained_listener_rejects_ready_mismatch_and_prevents_exit_rebind() {
        use std::io::Write;
        use std::os::unix::process::CommandExt;

        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let (lifeline, mut peer) = std::os::unix::net::UnixStream::pair().unwrap();
        let child = std::process::Command::new("sh")
            .args(["-c", "exec sleep 30"])
            .process_group(0)
            .spawn()
            .unwrap();
        let mut process = OwnedServerProcess::from_spawned(
            child,
            lifeline,
            None,
            tempfile::NamedTempFile::new().unwrap(),
            Some(listener),
        )
        .unwrap();
        writeln!(
            peer,
            "{}{}",
            crate::serve::CHAT_LIFELINE_READY_PREFIX,
            address.port().saturating_add(1)
        )
        .unwrap();
        peer.flush().unwrap();
        let error = process.poll_ready_port().unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match retained listener"));

        process.force_stop().unwrap();
        let rebind = std::net::TcpListener::bind(address).unwrap_err();
        assert_eq!(rebind.kind(), std::io::ErrorKind::AddrInUse);
        drop(process);
        std::net::TcpListener::bind(address)
            .expect("port is released only with endpoint authority");
    }

    #[cfg(unix)]
    #[test]
    fn startup_progress_accepts_only_bounded_valid_non_authoritative_events() {
        let (mut process, peer) = owned_sleep_process_with_progress();

        peer.send(b"not-an-hf2q-frame").unwrap();
        assert!(process.poll_startup_event().is_none());

        let oversized = vec![b'x'; crate::serve::CHAT_STARTUP_PROGRESS_MAX_BYTES + 1];
        peer.send(&oversized).unwrap();
        assert!(process.poll_startup_event().is_none());

        peer.send(b"HF2Q-P1:{\"phase\":\"ready\",\"port\":9123}")
            .unwrap();
        assert!(process.poll_startup_event().is_none());

        peer.send(
            b"HF2Q-P1:{\"phase\":\"local_search\",\"repository\":\"owner\\n/repo\",\"requested_quant\":null}",
        )
        .unwrap();
        assert!(process.poll_startup_event().is_none());

        let expected = crate::serve::startup_progress::StartupEvent::LocalSearch {
            repository: "owner/repo".into(),
            requested_quant: Some("Q6_K".into()),
        };
        let mut frame = crate::serve::CHAT_STARTUP_PROGRESS_PREFIX.to_vec();
        frame.extend(serde_json::to_vec(&expected).unwrap());
        peer.send(&frame).unwrap();
        assert_eq!(process.poll_startup_event(), Some(expected));

        process.force_stop().unwrap();
    }

    #[test]
    fn explicit_generic_endpoint_is_normalized() {
        let endpoint = Endpoint::explicit("https://example.test:9443/").unwrap();
        assert_eq!(endpoint.base_url(), "https://example.test:9443");
        assert!(Endpoint::explicit("file:///tmp/server").is_err());
        assert!(Endpoint::explicit("https://example.test/?token=secret").is_err());
        assert!(Endpoint::explicit("https://user:secret@example.test").is_err());
    }

    #[tokio::test]
    async fn external_endpoint_has_no_shutdown_authority() {
        let mut session = EndpointSession::external(Endpoint::discovered_loopback(9));
        tokio::time::timeout(
            Duration::from_millis(100),
            session.shutdown_if_owned(&reqwest::Client::new(), None),
        )
        .await
        .expect("external cleanup must not attempt a network request")
        .unwrap();
        assert!(session.retain_failure_diagnostics().unwrap().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn failure_diagnostics_retain_private_log_without_exposing_contents() {
        use std::io::Write;

        let directory = tempfile::tempdir().unwrap();
        let mut log = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        log.write_all(b"private-path=/private/operator/model.gguf token=hf_secret\n")
            .unwrap();
        log.flush().unwrap();

        let process = owned_sleep_process_with_log("sleep 30", log);
        let mut session = EndpointSession::spawned_loopback(9, process);
        let Ownership::Owned { process, .. } = &mut session.ownership else {
            panic!("expected owned child");
        };
        process.force_stop().unwrap();

        let diagnostics = session
            .retain_failure_diagnostics()
            .unwrap()
            .expect("owned failure must retain diagnostics");
        assert!(diagnostics.path.exists());
        let retained_contents = std::fs::read_to_string(&diagnostics.path).unwrap();
        assert!(retained_contents.contains("hf_secret"));
        let retained_path = diagnostics.path.clone();
        drop(session);
        assert!(
            retained_path.exists(),
            "retained failure log must survive session drop"
        );
    }

    #[cfg(unix)]
    #[test]
    fn clean_owned_session_deletes_its_temporary_log() {
        let process = owned_sleep_process("sleep 30");
        let temporary_path = process.server_log.path().to_owned();
        let mut session = EndpointSession::spawned_loopback(9, process);
        let Ownership::Owned { process, .. } = &mut session.ownership else {
            panic!("expected owned child");
        };
        process.force_stop().unwrap();
        drop(session);
        assert!(
            !temporary_path.exists(),
            "clean non-detached session must remove its temporary log"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn detach_relinquishes_an_owned_child_without_stopping_it() {
        let process = owned_sleep_process("sleep 30");
        let mut session = EndpointSession::spawned_loopback(9, process);
        assert!(session.detach().unwrap().is_some());
        session
            .shutdown_if_owned(&reqwest::Client::new(), None)
            .await
            .unwrap();
        let Ownership::Owned { process, .. } = &mut session.ownership else {
            panic!("expected owned child");
        };
        let child = process.child_mut();
        assert!(child.try_wait().unwrap().is_none());
        child.kill().unwrap();
        child.wait().unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn graceful_timeout_signals_only_the_owned_child_then_waits() {
        use axum::http::StatusCode;
        use axum::routing::post;
        use axum::Router;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route("/shutdown", post(|| async { StatusCode::ACCEPTED })),
            )
            .await
            .unwrap();
        });
        let process = owned_sleep_process("sleep 30");
        let mut session = EndpointSession::spawned_loopback(port, process);
        session
            .shutdown_if_owned_with_timeouts(
                &reqwest::Client::new(),
                None,
                Duration::from_millis(10),
                Duration::from_secs(2),
            )
            .await
            .unwrap();
        let Ownership::Owned { process, .. } = &mut session.ownership else {
            panic!("expected owned child");
        };
        let child = process.child_mut();
        assert!(child.try_wait().unwrap().is_some());
        server.abort();
    }

    #[cfg(unix)]
    #[test]
    fn force_stop_targets_the_owned_process_group_and_reaps_descendants() {
        let mut process = owned_sleep_process("trap '' TERM; sleep 30 & wait");
        let process_group = process.process_group;
        process.force_stop().unwrap();
        let result = unsafe { libc::kill(-process_group, 0) };
        assert_eq!(result, -1, "owned process group must no longer exist");
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }

    #[cfg(unix)]
    #[test]
    fn force_stop_cleans_descendants_after_the_group_leader_exits() {
        let mut process = owned_sleep_process("sleep 0.1; trap '' TERM; sleep 30 & exit 0");
        let process_group = process.process_group;
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if process.leader_exited_unreaped().unwrap() {
                break;
            }
            assert!(Instant::now() < deadline, "group leader did not exit");
            std::thread::sleep(Duration::from_millis(10));
        }
        process.force_stop().unwrap();
        let result = unsafe { libc::kill(-process_group, 0) };
        assert_eq!(result, -1, "orphaned descendants must be stopped");
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}
