//! Automatic machine-local endpoint selection for diagnostic chat.
//!
//! DNS-SD provides only untrusted candidates. Every candidate is forced to
//! loopback by `serve::discovery`, then verified over HTTP before selection.
//! Process lifecycle authority is created only for the concrete child this
//! module spawns; discovery PID hints are used solely to correlate its advert.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{bail, Context, Result};

use crate::cli::ChatArgs;
use crate::serve::discovery::{
    self, DiscoveryIdentity, LocalDiscoveryBrowser, LocalDiscoveryEvent,
    UntrustedDiscoveryCandidate,
};

use super::client::fetch_models;
use super::endpoint::{Endpoint, EndpointResolver, EndpointSession, OwnedServerProcess};
use super::wire::Model;

const EXISTING_DISCOVERY_WINDOW: Duration = Duration::from_secs(2);
const STARTUP_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(15);
const MODEL_STARTUP_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(6 * 60 * 60);
const HTTP_PROBE_TIMEOUT: Duration = Duration::from_secs(3);
const MODEL_STARTUP_HEARTBEAT: Duration = Duration::from_secs(30);

#[derive(Debug)]
struct VerifiedServer {
    identity: DiscoveryIdentity,
    endpoint: Endpoint,
    models: Vec<Model>,
}

pub(crate) struct AutomaticEndpointResolver;

impl EndpointResolver for AutomaticEndpointResolver {
    fn resolve(&mut self, args: &ChatArgs) -> Result<EndpointSession> {
        if let Some(url) = args.url.as_deref() {
            return Endpoint::explicit(url).map(EndpointSession::external);
        }
        let auth_token = std::env::var("HF2Q_AUTH_TOKEN")
            .ok()
            .filter(|token| !token.is_empty());
        if args.target.is_none() {
            require_credentialless_automatic_discovery(auth_token.as_deref())?;
        }
        if !discovery::is_supported() {
            bail!("automatic local hf2q discovery is unavailable on this platform; use --url");
        }

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build local discovery runtime")?;
        let stdin = std::io::stdin();
        let stdout = std::io::stdout();
        let mut input = stdin.lock();
        let mut output = stdout.lock();
        runtime.block_on(resolve_local(
            args,
            &mut input,
            &mut output,
            auth_token.as_deref(),
        ))
    }
}

async fn resolve_local(
    args: &ChatArgs,
    input: &mut impl BufRead,
    output: &mut impl Write,
    owned_auth_token: Option<&str>,
) -> Result<EndpointSession> {
    let http = reqwest::Client::builder()
        .connect_timeout(HTTP_PROBE_TIMEOUT)
        .timeout(HTTP_PROBE_TIMEOUT)
        .build()
        .context("build local server probe client")?;

    if args.target.is_none() {
        let mut browser = LocalDiscoveryBrowser::start().context("start local hf2q discovery")?;
        let existing = collect_verified(&mut browser, &http, EXISTING_DISCOVERY_WINDOW).await?;
        if !existing.is_empty() {
            let selected = select_server(&existing, args.model.as_deref(), input, output)?;
            return Ok(EndpointSession::discovered_hf2q(selected.endpoint.clone()));
        }
    }

    if args.target.is_some() {
        writeln!(
            output,
            "starting an owned hf2q server for the requested model"
        )?;
        writeln!(
            output,
            "preparing the requested model; local verification, download, or native conversion may take time on first use"
        )?;
    } else {
        writeln!(output, "no local hf2q server found; starting one")?;
    }
    output.flush()?;
    let mut startup_browser =
        LocalDiscoveryBrowser::start().context("start owned-server discovery")?;
    let mut child = spawn_server(args.target.as_deref()).context("start hf2q serve")?;
    let child_pid = child.child_mut().id().to_string();
    let startup = wait_for_spawned_server(
        &mut startup_browser,
        &http,
        &mut child,
        &child_pid,
        if args.target.is_some() {
            MODEL_STARTUP_DISCOVERY_TIMEOUT
        } else {
            STARTUP_DISCOVERY_TIMEOUT
        },
        owned_auth_token,
        output,
        args.target.is_some(),
    )
    .await;
    match startup {
        Ok(server) => Ok(EndpointSession::spawned_loopback(
            endpoint_port(&server.endpoint)?,
            child,
        )),
        Err(error) => Err(finalize_failed_startup(error, &mut child)),
    }
}

fn finalize_failed_startup(error: anyhow::Error, child: &mut OwnedServerProcess) -> anyhow::Error {
    let cleanup = stop_failed_child(child);
    let retained_log = child.retain_log();
    match (cleanup, retained_log) {
        (Ok(()), Ok(path)) => anyhow::anyhow!(
            "{error:#}; private chat-owned server log retained at {}",
            path.display()
        ),
        (Err(stop_error), Ok(path)) => anyhow::anyhow!(
            "{error:#}; cleanup failed: {stop_error:#}; private chat-owned server log retained at {}",
            path.display()
        ),
        (Ok(()), Err(log_error)) => anyhow::anyhow!(
            "{error:#}; additionally failed to retain the private chat-owned server log: {log_error:#}"
        ),
        (Err(stop_error), Err(log_error)) => anyhow::anyhow!(
            "{error:#}; cleanup failed: {stop_error:#}; additionally failed to retain the private chat-owned server log: {log_error:#}"
        ),
    }
}

fn require_credentialless_automatic_discovery(auth_token: Option<&str>) -> Result<()> {
    if auth_token.is_some() {
        bail!(
            "automatic discovery is disabled while HF2Q_AUTH_TOKEN is set because DNS-SD candidates are untrusted; use --url with the intended local endpoint"
        );
    }
    Ok(())
}

#[cfg(unix)]
fn spawn_server(target: Option<&str>) -> Result<OwnedServerProcess> {
    use std::os::fd::AsRawFd;
    use std::os::unix::process::CommandExt;

    let executable = std::env::current_exe().context("locate current hf2q executable")?;
    let (parent_lifeline, child_lifeline) = std::os::unix::net::UnixStream::pair()
        .context("create private chat parent-lifetime channel")?;
    let child_fd = child_lifeline.as_raw_fd();
    let server_log = tempfile::Builder::new()
        .prefix("hf2q-chat-server-")
        .suffix(".log")
        .tempfile()
        .context("create chat-owned server log")?;
    let stderr = server_log
        .reopen()
        .context("open chat-owned server log writer")?;
    let mut command = Command::new(executable);
    command.arg("serve");
    if let Some(target) = target {
        command.arg(target);
    }
    command
        .args(["--host", "127.0.0.1", "--port", "0", "--quiet"])
        .args(["--operator-ui", "plain", "--chat-parent-lifeline-fd"])
        .arg(child_fd.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .process_group(0);
    // The socket pair is CLOEXEC by default. Clear it only in the child just
    // before exec; the server immediately restores CLOEXEC before spawning
    // any model-transfer or conversion descendants.
    unsafe {
        command.pre_exec(move || {
            let flags = libc::fcntl(child_fd, libc::F_GETFD);
            if flags < 0 || libc::fcntl(child_fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let child = command
        .spawn()
        .context("spawn current executable as hf2q serve")?;
    drop(child_lifeline);
    OwnedServerProcess::from_spawned(child, parent_lifeline, server_log)
}

#[cfg(not(unix))]
fn spawn_server(_target: Option<&str>) -> Result<OwnedServerProcess> {
    bail!("automatic chat-owned server lifecycle is unavailable on this platform; use --url")
}

async fn collect_verified(
    browser: &mut LocalDiscoveryBrowser,
    http: &reqwest::Client,
    timeout: Duration,
) -> Result<Vec<VerifiedServer>> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut servers = BTreeMap::new();
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match browser.next_event(remaining).await? {
            None => break,
            Some(LocalDiscoveryEvent::Added(candidate)) => {
                if let Some(server) = verify_candidate(http, candidate, None).await {
                    servers.insert(server.identity.clone(), server);
                }
            }
            Some(LocalDiscoveryEvent::Removed(identity)) => {
                servers.remove(&identity);
            }
            Some(LocalDiscoveryEvent::Rejected { identity, reason }) => {
                tracing::debug!(service = %identity.service_name, ?reason, "ignored unresolved hf2q discovery candidate");
            }
        }
    }
    Ok(servers.into_values().collect())
}

async fn wait_for_spawned_server(
    browser: &mut LocalDiscoveryBrowser,
    http: &reqwest::Client,
    process: &mut OwnedServerProcess,
    child_pid: &str,
    timeout: Duration,
    auth_token: Option<&str>,
    output: &mut impl Write,
    show_preparation_progress: bool,
) -> Result<VerifiedServer> {
    let started = tokio::time::Instant::now();
    let deadline = tokio::time::Instant::now() + timeout;
    let mut next_heartbeat = started + MODEL_STARTUP_HEARTBEAT;
    loop {
        if process.leader_exited_unreaped()? {
            bail!("chat-started hf2q serve exited before discovery");
        }
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            bail!(
                "chat-started hf2q serve was not discovered and HTTP-verified within {:?}",
                timeout
            );
        }
        if show_preparation_progress && tokio::time::Instant::now() >= next_heartbeat {
            let elapsed = tokio::time::Instant::now().duration_since(started);
            write_model_startup_heartbeat(output, elapsed)?;
            next_heartbeat += MODEL_STARTUP_HEARTBEAT;
        }
        let event = browser
            .next_event(remaining.min(Duration::from_millis(500)))
            .await?;
        let Some(LocalDiscoveryEvent::Added(candidate)) = event else {
            continue;
        };
        if candidate.hints.pid_hint.as_deref() != Some(child_pid) {
            continue;
        }
        // Credentials are sent only after the discovery PID has been matched
        // to the exact child process this chat invocation spawned.
        if let Some(server) = verify_candidate(http, candidate, auth_token).await {
            return Ok(server);
        }
    }
}

fn write_model_startup_heartbeat(output: &mut impl Write, elapsed: Duration) -> Result<()> {
    writeln!(
        output,
        "still preparing the requested model ({}s elapsed); first use may be downloading or converting",
        elapsed.as_secs()
    )?;
    output.flush()?;
    Ok(())
}

async fn verify_candidate(
    http: &reqwest::Client,
    candidate: UntrustedDiscoveryCandidate,
    auth_token: Option<&str>,
) -> Option<VerifiedServer> {
    let endpoint = Endpoint::discovered_loopback(candidate.endpoint.port());
    let mut request = http.get(endpoint.route("/health"));
    if let Some(token) = auth_token {
        request = request.bearer_auth(token);
    }
    let response = match request.send().await {
        Ok(response) if response.status().is_success() => response,
        Ok(response) => {
            if matches!(
                response.status(),
                reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
            ) {
                if auth_token.is_some() {
                    tracing::warn!(
                        url = %endpoint.base_url(),
                        status = %response.status(),
                        "chat-owned hf2q server rejected HF2Q_AUTH_TOKEN"
                    );
                } else {
                    tracing::warn!(
                        url = %endpoint.base_url(),
                        status = %response.status(),
                        "authenticated local hf2q servers require an explicit --url so credentials are never sent to an untrusted discovery candidate"
                    );
                }
            } else {
                tracing::debug!(url = %endpoint.base_url(), status = %response.status(), "ignored unhealthy local discovery candidate");
            }
            return None;
        }
        Err(error) => {
            tracing::debug!(url = %endpoint.base_url(), %error, "ignored unreachable local discovery candidate");
            return None;
        }
    };
    drop(response);
    let models = match fetch_models(http, &endpoint, auth_token).await {
        Ok(models) => models,
        Err(error) => {
            tracing::debug!(url = %endpoint.base_url(), %error, "ignored local discovery candidate without a usable model API");
            return None;
        }
    };
    Some(VerifiedServer {
        identity: candidate.identity,
        endpoint,
        models,
    })
}

fn select_server<'a>(
    servers: &'a [VerifiedServer],
    requested_model: Option<&str>,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<&'a VerifiedServer> {
    if servers.len() == 1 {
        return Ok(&servers[0]);
    }
    writeln!(output, "local hf2q servers:")?;
    for (index, server) in servers.iter().enumerate() {
        let resident = server
            .models
            .iter()
            .filter(|model| model.loaded.unwrap_or(false))
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>();
        let requested = requested_model
            .filter(|requested| server.models.iter().any(|model| model.id == *requested))
            .map(|_| " [requested model advertised]")
            .unwrap_or_default();
        writeln!(
            output,
            "  {}. {} resident={}{}",
            index + 1,
            server.endpoint.base_url(),
            if resident.is_empty() {
                "none".to_owned()
            } else {
                resident.join(",")
            },
            requested
        )?;
    }
    write!(output, "server> ")?;
    output.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        bail!("input ended before a server was selected");
    }
    let selection: usize = line
        .trim()
        .parse()
        .context("server selection must be a number")?;
    servers
        .get(
            selection
                .checked_sub(1)
                .context("server selection starts at 1")?,
        )
        .context("server selection is out of range")
}

fn endpoint_port(endpoint: &Endpoint) -> Result<u16> {
    reqwest::Url::parse(endpoint.base_url())?
        .port_or_known_default()
        .context("verified local endpoint had no port")
}

fn stop_failed_child(process: &mut OwnedServerProcess) -> Result<()> {
    process
        .force_stop()
        .context("stop unverified chat-started server process group")
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use std::io::Write;
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;
    use std::sync::{Arc, Mutex};

    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::get;
    use axum::{Json, Router};
    use tokio::sync::oneshot;

    use super::*;

    fn server(port: u16, models: &[(&str, bool)]) -> VerifiedServer {
        VerifiedServer {
            identity: DiscoveryIdentity {
                service_name: format!("server-{port}"),
                service_type: discovery::SERVICE_TYPE.to_owned(),
                domain: "local.".to_owned(),
            },
            endpoint: Endpoint::discovered_loopback(port),
            models: models
                .iter()
                .map(|(id, loaded)| Model {
                    id: (*id).to_owned(),
                    loaded: Some(*loaded),
                })
                .collect(),
        }
    }

    fn candidate(port: u16) -> UntrustedDiscoveryCandidate {
        UntrustedDiscoveryCandidate {
            identity: DiscoveryIdentity {
                service_name: format!("candidate-{port}"),
                service_type: discovery::SERVICE_TYPE.to_owned(),
                domain: "local.".to_owned(),
            },
            endpoint: format!("127.0.0.1:{port}").parse().unwrap(),
            hints: Default::default(),
        }
    }

    async fn serve(router: Router) -> (u16, oneshot::Sender<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (stop_tx, stop_rx) = oneshot::channel();
        tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stop_rx.await;
                })
                .await
                .unwrap();
        });
        (port, stop_tx)
    }

    #[test]
    fn one_server_is_automatic_and_multiple_servers_use_numbered_picker() {
        let one = vec![server(9001, &[("model-a", true)])];
        let mut unused_input = std::io::Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();
        assert_eq!(
            select_server(&one, None, &mut unused_input, &mut output)
                .unwrap()
                .endpoint
                .base_url(),
            "http://127.0.0.1:9001"
        );
        assert!(output.is_empty());

        let multiple = vec![
            server(9001, &[("resident", true)]),
            server(9002, &[("candidate", false)]),
        ];
        let mut input = std::io::Cursor::new(b"2\n");
        assert_eq!(
            select_server(&multiple, Some("candidate"), &mut input, &mut output)
                .unwrap()
                .endpoint
                .base_url(),
            "http://127.0.0.1:9002"
        );
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("resident=resident"));
        assert!(output.contains("requested model advertised"));
    }

    #[test]
    fn authenticated_automatic_discovery_requires_an_explicit_url() {
        assert!(require_credentialless_automatic_discovery(None).is_ok());
        let error = require_credentialless_automatic_discovery(Some("secret")).unwrap_err();
        assert!(error
            .to_string()
            .contains("DNS-SD candidates are untrusted"));
        assert!(error.to_string().contains("--url"));
    }

    #[cfg(unix)]
    #[test]
    fn failed_startup_stops_child_and_retains_path_without_echoing_private_log() {
        let directory = tempfile::tempdir().unwrap();
        let mut log = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        log.write_all(b"path=/private/operator/model.gguf token=hf_secret\n")
            .unwrap();
        log.flush().unwrap();
        let log_path = log.path().to_owned();
        let (parent_lifeline, _child_lifeline) = std::os::unix::net::UnixStream::pair().unwrap();
        let child = std::process::Command::new("sh")
            .arg("-c")
            .arg("exec sleep 30")
            .process_group(0)
            .spawn()
            .unwrap();
        let child_pid = child.id() as libc::pid_t;
        let mut process = OwnedServerProcess::from_spawned(child, parent_lifeline, log).unwrap();

        let error = finalize_failed_startup(anyhow::anyhow!("startup probe failed"), &mut process);
        let rendered = format!("{error:#}");
        assert!(rendered.contains("startup probe failed"));
        assert!(rendered.contains(&log_path.display().to_string()));
        assert!(!rendered.contains("/private/operator/model.gguf"));
        assert!(!rendered.contains("hf_secret"));
        assert!(log_path.exists());
        assert_eq!(unsafe { libc::kill(child_pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }

    #[tokio::test]
    async fn candidate_verification_uses_real_http_without_authorization() {
        #[derive(Clone, Default)]
        struct Recorded(Arc<Mutex<Vec<Option<String>>>>);

        async fn record(
            State(recorded): State<Recorded>,
            headers: HeaderMap,
        ) -> Json<serde_json::Value> {
            recorded.0.lock().unwrap().push(
                headers
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned),
            );
            Json(serde_json::json!({"status":"ok","data":[]}))
        }

        let recorded = Recorded::default();
        let router = Router::new()
            .route("/health", get(record))
            .route("/v1/models", get(record))
            .with_state(recorded.clone());
        let (port, stop) = serve(router).await;
        let verified = verify_candidate(&reqwest::Client::new(), candidate(port), None)
            .await
            .expect("healthy candidate must verify");
        assert_eq!(
            verified.endpoint.base_url(),
            format!("http://127.0.0.1:{port}")
        );
        assert_eq!(*recorded.0.lock().unwrap(), vec![None, None]);
        let _ = stop.send(());
    }

    #[tokio::test]
    async fn owned_candidate_verification_uses_authorization_after_pid_match() {
        #[derive(Clone, Default)]
        struct Recorded(Arc<Mutex<Vec<Option<String>>>>);

        async fn record(
            State(recorded): State<Recorded>,
            headers: HeaderMap,
        ) -> Json<serde_json::Value> {
            recorded.0.lock().unwrap().push(
                headers
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned),
            );
            Json(serde_json::json!({"status":"ok","data":[]}))
        }

        let recorded = Recorded::default();
        let router = Router::new()
            .route("/health", get(record))
            .route("/v1/models", get(record))
            .with_state(recorded.clone());
        let (port, stop) = serve(router).await;
        assert!(
            verify_candidate(&reqwest::Client::new(), candidate(port), Some("secret"))
                .await
                .is_some()
        );
        assert_eq!(
            *recorded.0.lock().unwrap(),
            vec![Some("Bearer secret".into()), Some("Bearer secret".into())]
        );
        let _ = stop.send(());
    }

    #[test]
    fn model_startup_heartbeat_is_operator_visible_without_private_log_content() {
        let mut output = Vec::new();
        write_model_startup_heartbeat(&mut output, Duration::from_secs(61)).unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("61s elapsed"));
        assert!(output.contains("downloading or converting"));
    }

    #[tokio::test]
    async fn unhealthy_http_candidate_is_rejected() {
        let router = Router::new().route(
            "/health",
            get(|| async { (StatusCode::SERVICE_UNAVAILABLE, "not ready") }),
        );
        let (port, stop) = serve(router).await;
        assert!(
            verify_candidate(&reqwest::Client::new(), candidate(port), None)
                .await
                .is_none()
        );
        let _ = stop.send(());
    }
}
