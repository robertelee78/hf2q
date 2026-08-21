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
    Owned { child: Child, detached: bool },
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
    pub(crate) fn spawned_loopback(port: u16, child: Child) -> Self {
        Self {
            endpoint: Endpoint::discovered_loopback(port),
            kind: EndpointKind::DiscoveredHf2q,
            ownership: Ownership::Owned {
                child,
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

    pub(crate) fn detach(&mut self) -> bool {
        match &mut self.ownership {
            Ownership::External => false,
            Ownership::Owned { detached, .. } => {
                *detached = true;
                true
            }
        }
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
        let Ownership::Owned { child, detached } = &mut self.ownership else {
            return Ok(());
        };
        if *detached
            || child
                .try_wait()
                .context("check owned server child")?
                .is_some()
        {
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
            signal_owned_child(child)?;
            if wait_for_child(child, graceful_timeout).await? {
                return Ok(());
            }
            tracing::warn!("chat-owned server ignored SIGTERM; force-stopping the owned child");
            return force_stop_owned_child(child);
        }

        if wait_for_child(child, graceful_timeout).await? {
            return Ok(());
        }
        tracing::warn!(
            "chat-owned server exceeded graceful drain deadline; signaling the owned child"
        );
        signal_owned_child(child)?;
        if wait_for_child(child, signal_timeout).await? {
            return Ok(());
        }
        tracing::warn!(
            "chat-owned server ignored graceful shutdown and SIGTERM; force-stopping the owned child"
        );
        force_stop_owned_child(child)
    }
}

async fn wait_for_child(child: &mut Child, timeout: Duration) -> Result<bool> {
    let deadline = Instant::now() + timeout;
    loop {
        if child
            .try_wait()
            .context("wait for chat-owned server")?
            .is_some()
        {
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
fn signal_owned_child(child: &mut Child) -> Result<()> {
    // The PID comes only from the live Child handle created by this chat
    // process. Discovery metadata is never accepted as process authority.
    let result = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).context("signal chat-owned server child")
    }
}

#[cfg(not(unix))]
fn signal_owned_child(child: &mut Child) -> Result<()> {
    child.kill().context("stop chat-owned server child")
}

fn force_stop_owned_child(child: &mut Child) -> Result<()> {
    if child
        .try_wait()
        .context("check owned server child")?
        .is_some()
    {
        return Ok(());
    }
    child.kill().context("force-stop chat-owned server child")?;
    child
        .wait()
        .context("reap force-stopped chat-owned server")?;
    Ok(())
}

pub(crate) trait EndpointResolver {
    fn resolve(&mut self, args: &ChatArgs) -> Result<EndpointSession>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovered_endpoints_are_forced_to_loopback_and_have_no_pid_authority() {
        let endpoint = Endpoint::discovered_loopback(9123);
        assert_eq!(endpoint.base_url(), "http://127.0.0.1:9123");
        assert_eq!(
            endpoint.route("/v1/models"),
            "http://127.0.0.1:9123/v1/models"
        );
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
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn detach_relinquishes_an_owned_child_without_stopping_it() {
        let child = std::process::Command::new("sh")
            .args(["-c", "sleep 30"])
            .spawn()
            .unwrap();
        let mut session = EndpointSession::spawned_loopback(9, child);
        assert!(session.detach());
        session
            .shutdown_if_owned(&reqwest::Client::new(), None)
            .await
            .unwrap();
        let Ownership::Owned { child, .. } = &mut session.ownership else {
            panic!("expected owned child");
        };
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
        let child = std::process::Command::new("sh")
            .args(["-c", "sleep 30"])
            .spawn()
            .unwrap();
        let mut session = EndpointSession::spawned_loopback(port, child);
        session
            .shutdown_if_owned_with_timeouts(
                &reqwest::Client::new(),
                None,
                Duration::from_millis(10),
                Duration::from_secs(2),
            )
            .await
            .unwrap();
        let Ownership::Owned { child, .. } = &mut session.ownership else {
            panic!("expected owned child");
        };
        assert!(child.try_wait().unwrap().is_some());
        server.abort();
    }
}
