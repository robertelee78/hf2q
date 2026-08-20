//! Machine-local discovery for live hf2q HTTP servers (ADR-047).
//!
//! DNS-SD is only a source of untrusted candidates. The browser accepts the
//! resolved port, constructs a loopback endpoint itself, and deliberately
//! ignores the resolved host plus any TXT-provided host or URL. PID and start
//! values are display/correlation hints; neither conveys process ownership.

use std::net::SocketAddr;
#[cfg(target_os = "macos")]
use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use thiserror::Error;

pub(crate) const SERVICE_TYPE: &str = "_hf2q._tcp";
pub(crate) const SCHEMA_VERSION: &str = "1";
const REGISTRATION_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_EVENT_WAIT: Duration = Duration::from_secs(10);

#[derive(Debug, Error)]
pub(crate) enum DiscoveryError {
    #[cfg(not(target_os = "macos"))]
    #[error("automatic local discovery is available only on macOS; use an explicit URL")]
    UnsupportedPlatform,
    #[error("cannot read the bound HTTP listener address: {0}")]
    ListenerAddress(#[source] std::io::Error),
    #[error("the bound HTTP listener reported invalid port 0")]
    InvalidBoundPort,
    #[error("the system clock is before the Unix epoch")]
    ClockBeforeUnixEpoch,
    #[error("DNS-SD registration setup failed: {0}")]
    RegistrationSetup(#[source] std::io::Error),
    #[error("DNS-SD registration timed out after {0:?}")]
    RegistrationTimedOut(Duration),
    #[error("DNS-SD registration failed: {0}")]
    Registration(#[source] std::io::Error),
    #[error("DNS-SD browse failed: {0}")]
    Browse(#[source] std::io::Error),
}

impl DiscoveryError {
    #[cfg(target_os = "macos")]
    fn is_name_conflict(&self) -> bool {
        fn error_is_name_conflict(error: &std::io::Error) -> bool {
            error.to_string() == "name conflict"
                || std::error::Error::source(error)
                    .is_some_and(|source| source.to_string() == "name conflict")
        }

        match self {
            Self::RegistrationSetup(error) | Self::Registration(error) => {
                error_is_name_conflict(error)
            }
            _ => false,
        }
    }
}

/// Stable DNS-SD identity. It identifies a browse record, not a process.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct DiscoveryIdentity {
    pub(crate) service_name: String,
    pub(crate) service_type: String,
    pub(crate) domain: String,
}

/// Untrusted TXT values. These are display/correlation hints only.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct UntrustedDiscoveryHints {
    pub(crate) schema_hint: Option<String>,
    /// Never use this value to signal, stop, or otherwise control a process.
    pub(crate) pid_hint: Option<String>,
    pub(crate) start_hint: Option<String>,
}

/// A resolved but not yet HTTP-verified local service candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct UntrustedDiscoveryCandidate {
    pub(crate) identity: DiscoveryIdentity,
    /// Constructed by hf2q from the resolved port; always loopback.
    pub(crate) endpoint: SocketAddr,
    pub(crate) hints: UntrustedDiscoveryHints,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum DiscoveryRejection {
    ResolveTimedOut,
    ResolveStreamEnded,
    ResolveFailed(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum LocalDiscoveryEvent {
    Added(UntrustedDiscoveryCandidate),
    Removed(DiscoveryIdentity),
    Rejected {
        identity: DiscoveryIdentity,
        reason: DiscoveryRejection,
    },
}

/// Keeps the native DNS-SD registration alive for the HTTP listener lifetime.
pub(crate) struct LocalServiceRegistration {
    service_name: String,
    port: u16,
    #[cfg(target_os = "macos")]
    _native: async_dnssd::Registration,
}

impl std::fmt::Debug for LocalServiceRegistration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalServiceRegistration")
            .field("service_name", &self.service_name)
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

impl LocalServiceRegistration {
    pub(crate) fn service_name(&self) -> &str {
        &self.service_name
    }

    pub(crate) fn port(&self) -> u16 {
        self.port
    }
}

pub(crate) struct LocalDiscoveryBrowser {
    #[cfg(target_os = "macos")]
    native: async_dnssd::Browse,
}

pub(crate) const fn is_supported() -> bool {
    cfg!(target_os = "macos")
}

/// Register the listener's actual bound port, including when the requested
/// CLI port was zero. Registration failure never owns the listener lifecycle.
pub(crate) async fn register_bound_listener(
    listener: &tokio::net::TcpListener,
) -> Result<LocalServiceRegistration, DiscoveryError> {
    let address = listener
        .local_addr()
        .map_err(DiscoveryError::ListenerAddress)?;
    if address.port() == 0 {
        return Err(DiscoveryError::InvalidBoundPort);
    }

    #[cfg(target_os = "macos")]
    {
        register_macos(address.port()).await
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = address;
        Err(DiscoveryError::UnsupportedPlatform)
    }
}

impl LocalDiscoveryBrowser {
    pub(crate) fn start() -> Result<Self, DiscoveryError> {
        #[cfg(target_os = "macos")]
        {
            use async_dnssd::{BrowseData, Interface};

            let native = async_dnssd::browse_extended(
                SERVICE_TYPE,
                BrowseData {
                    interface: Interface::LocalOnly,
                    ..Default::default()
                },
            );
            Ok(Self { native })
        }

        #[cfg(not(target_os = "macos"))]
        {
            Err(DiscoveryError::UnsupportedPlatform)
        }
    }

    /// Wait for one browse event and, for additions, resolve it within the
    /// same total deadline. `Ok(None)` means the bounded wait elapsed.
    pub(crate) async fn next_event(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<LocalDiscoveryEvent>, DiscoveryError> {
        #[cfg(target_os = "macos")]
        {
            self.next_event_macos(timeout.min(MAX_EVENT_WAIT)).await
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = timeout;
            Err(DiscoveryError::UnsupportedPlatform)
        }
    }
}

#[cfg(target_os = "macos")]
async fn register_macos(port: u16) -> Result<LocalServiceRegistration, DiscoveryError> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let pid = std::process::id();
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| DiscoveryError::ClockBeforeUnixEpoch)?
        .as_millis();
    let start_hint = started.min(u64::MAX as u128).to_string();
    let pid_hint = pid.to_string();
    let service_name = format!("hf2q-{pid}");
    let txt = encode_txt(&[
        ("schema", SCHEMA_VERSION),
        ("pid", &pid_hint),
        ("start", &start_hint),
    ]);
    match register_macos_named(port, &service_name, &txt).await {
        Ok(registration) => Ok(registration),
        Err(error) if error.is_name_conflict() => {
            // LocalOnly registrations can report a synchronous name conflict
            // instead of applying DNS-SD's ordinary auto-rename behavior.
            // Retry once with a bounded unique suffix; never loop forever.
            let suffix = uuid::Uuid::new_v4().simple().to_string();
            let retry_name = format!("{service_name}-{}", &suffix[..8]);
            register_macos_named(port, &retry_name, &txt).await
        }
        Err(error) => Err(error),
    }
}

#[cfg(target_os = "macos")]
async fn register_macos_named(
    port: u16,
    service_name: &str,
    txt: &[u8],
) -> Result<LocalServiceRegistration, DiscoveryError> {
    use async_dnssd::{Interface, RegisterData};

    let registration = async_dnssd::register_extended(
        SERVICE_TYPE,
        port,
        RegisterData {
            interface: Interface::LocalOnly,
            name: Some(service_name),
            txt,
            ..Default::default()
        },
    )
    .map_err(DiscoveryError::RegistrationSetup)?;
    let (_native, result) = tokio::time::timeout(REGISTRATION_TIMEOUT, registration)
        .await
        .map_err(|_| DiscoveryError::RegistrationTimedOut(REGISTRATION_TIMEOUT))?
        .map_err(DiscoveryError::Registration)?;

    Ok(LocalServiceRegistration {
        service_name: result.name,
        port,
        _native,
    })
}

#[cfg(target_os = "macos")]
impl LocalDiscoveryBrowser {
    async fn next_event_macos(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<LocalDiscoveryEvent>, DiscoveryError> {
        use async_dnssd::BrowsedFlags;
        use futures::StreamExt;
        use tokio::time::Instant;

        let deadline = Instant::now() + timeout;
        let browse = match tokio::time::timeout(timeout, self.native.next()).await {
            Err(_) => return Ok(None),
            Ok(None) => return Ok(None),
            Ok(Some(Err(error))) => return Err(DiscoveryError::Browse(error)),
            Ok(Some(Ok(event))) => event,
        };
        let identity = DiscoveryIdentity {
            service_name: browse.service_name.clone(),
            service_type: browse.reg_type.clone(),
            domain: browse.domain.clone(),
        };
        if !browse.flags.contains(BrowsedFlags::ADD) {
            return Ok(Some(LocalDiscoveryEvent::Removed(identity)));
        }

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(Some(LocalDiscoveryEvent::Rejected {
                identity,
                reason: DiscoveryRejection::ResolveTimedOut,
            }));
        }
        let mut resolve = browse.resolve();
        let resolved = match tokio::time::timeout(remaining, resolve.next()).await {
            Err(_) => {
                return Ok(Some(LocalDiscoveryEvent::Rejected {
                    identity,
                    reason: DiscoveryRejection::ResolveTimedOut,
                }))
            }
            Ok(None) => {
                return Ok(Some(LocalDiscoveryEvent::Rejected {
                    identity,
                    reason: DiscoveryRejection::ResolveStreamEnded,
                }))
            }
            Ok(Some(Err(error))) => {
                return Ok(Some(LocalDiscoveryEvent::Rejected {
                    identity,
                    reason: DiscoveryRejection::ResolveFailed(error.to_string()),
                }))
            }
            Ok(Some(Ok(resolved))) => resolved,
        };

        // Security boundary: DNS-SD supplies the port only. Never follow its
        // host target, and never interpret TXT as a host or URL.
        Ok(Some(LocalDiscoveryEvent::Added(untrusted_candidate(
            identity,
            resolved.port,
            &resolved.txt,
        ))))
    }
}

#[cfg(target_os = "macos")]
fn encode_txt(entries: &[(&str, &str)]) -> Vec<u8> {
    let mut encoded = Vec::new();
    for (key, value) in entries {
        let field = format!("{key}={value}");
        debug_assert!(field.len() <= u8::MAX as usize);
        encoded.push(field.len() as u8);
        encoded.extend_from_slice(field.as_bytes());
    }
    encoded
}

#[cfg(target_os = "macos")]
fn parse_untrusted_hints(txt: &[u8]) -> UntrustedDiscoveryHints {
    let Some(record) = async_dnssd::TxtRecord::parse(txt) else {
        return UntrustedDiscoveryHints::default();
    };
    let mut hints = UntrustedDiscoveryHints::default();
    for (key, value) in record.iter() {
        let Some(value) = value else {
            continue;
        };
        let value = String::from_utf8_lossy(value).into_owned();
        match key {
            b"schema" => hints.schema_hint = Some(value),
            b"pid" => hints.pid_hint = Some(value),
            b"start" => hints.start_hint = Some(value),
            // Unknown fields, including host/url, are never surfaced or used.
            _ => {}
        }
    }
    hints
}

#[cfg(target_os = "macos")]
fn untrusted_candidate(
    identity: DiscoveryIdentity,
    resolved_port: u16,
    txt: &[u8],
) -> UntrustedDiscoveryCandidate {
    UntrustedDiscoveryCandidate {
        identity,
        endpoint: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), resolved_port),
        hints: parse_untrusted_hints(txt),
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[tokio::test(flavor = "current_thread")]
    async fn localonly_actual_port_simultaneous_collision_and_removal() {
        let mut browser = LocalDiscoveryBrowser::start().expect("start LocalOnly browser");
        let listener_a = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind listener A");
        let listener_b = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .expect("bind listener B");
        let port_a = listener_a.local_addr().unwrap().port();
        let port_b = listener_b.local_addr().unwrap().port();
        assert_ne!(port_a, 0);
        assert_ne!(port_b, 0);
        assert_ne!(port_a, port_b);

        let registration_a = register_bound_listener(&listener_a)
            .await
            .expect("register listener A");
        let registration_b = register_bound_listener(&listener_b)
            .await
            .expect("register listener B");
        assert_eq!(registration_a.port(), port_a);
        assert_eq!(registration_b.port(), port_b);
        // Both registrations first request `hf2q-<same test PID>`; LocalOnly
        // reports a conflict, and hf2q must retry with a unique suffix.
        assert_ne!(registration_a.service_name(), registration_b.service_name());

        let expected_ports = [port_a, port_b];
        let found = tokio::time::timeout(Duration::from_secs(10), async {
            let mut found = BTreeMap::new();
            while found.len() < 2 {
                match browser
                    .next_event(Duration::from_secs(5))
                    .await
                    .expect("browse event")
                    .expect("browse deadline")
                {
                    LocalDiscoveryEvent::Added(candidate)
                        if expected_ports.contains(&candidate.endpoint.port()) =>
                    {
                        assert!(candidate.endpoint.ip().is_loopback());
                        assert_eq!(candidate.endpoint.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
                        assert_eq!(candidate.hints.schema_hint.as_deref(), Some(SCHEMA_VERSION));
                        let expected_pid = std::process::id().to_string();
                        assert_eq!(
                            candidate.hints.pid_hint.as_deref(),
                            Some(expected_pid.as_str())
                        );
                        assert!(candidate
                            .hints
                            .start_hint
                            .as_deref()
                            .and_then(|value| value.parse::<u64>().ok())
                            .is_some_and(|value| value > 0));
                        found.insert(candidate.endpoint.port(), candidate);
                    }
                    LocalDiscoveryEvent::Added(_)
                    | LocalDiscoveryEvent::Removed(_)
                    | LocalDiscoveryEvent::Rejected { .. } => {}
                }
            }
            found
        })
        .await
        .expect("discover both registrations within the total deadline");
        assert_eq!(found.keys().copied().collect::<Vec<_>>(), {
            let mut ports = vec![port_a, port_b];
            ports.sort_unstable();
            ports
        });

        let removed_name = registration_a.service_name().to_owned();
        drop(registration_a);
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                match browser
                    .next_event(Duration::from_secs(5))
                    .await
                    .expect("removal event")
                    .expect("removal deadline")
                {
                    LocalDiscoveryEvent::Removed(identity)
                        if identity.service_name == removed_name =>
                    {
                        break;
                    }
                    _ => {}
                }
            }
        })
        .await
        .expect("observe removal within the total deadline");
        drop((registration_b, listener_a, listener_b));
    }

    #[test]
    fn txt_host_and_url_are_ignored_and_pid_stays_a_hint() {
        let txt = encode_txt(&[
            ("schema", "1"),
            ("pid", "42"),
            ("start", "43"),
            ("host", "attacker.example"),
            ("url", "https://attacker.example/steal"),
        ]);
        let candidate = untrusted_candidate(
            DiscoveryIdentity {
                service_name: "malicious".to_owned(),
                service_type: SERVICE_TYPE.to_owned(),
                domain: "local.".to_owned(),
            },
            4242,
            &txt,
        );
        assert_eq!(candidate.hints.schema_hint.as_deref(), Some("1"));
        assert_eq!(candidate.hints.pid_hint.as_deref(), Some("42"));
        assert_eq!(candidate.hints.start_hint.as_deref(), Some("43"));
        assert_eq!(
            candidate.endpoint,
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4242)
        );
    }
}

#[cfg(all(test, not(target_os = "macos")))]
mod non_macos_tests {
    use super::*;

    #[test]
    fn automatic_discovery_reports_explicit_url_fallback() {
        assert!(!is_supported());
        assert!(matches!(
            LocalDiscoveryBrowser::start(),
            Err(DiscoveryError::UnsupportedPlatform)
        ));
    }
}
