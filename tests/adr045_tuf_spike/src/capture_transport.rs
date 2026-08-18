//! Exact-byte recording transport used only by the spike.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use tough::{Transport, TransportError, TransportErrorKind, TransportStream};
use url::Url;

use crate::model::{RoleKind, SpikeError};
use crate::strict_json;

#[derive(Clone, Debug)]
pub(crate) enum ScriptedResponse {
    Bytes(Vec<u8>),
    NotFound,
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FetchOutcome {
    Complete,
    NotFound,
    Other,
    Rejected,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FetchRecord {
    pub(crate) request_name: String,
    pub(crate) role: Option<RoleKind>,
    pub(crate) bytes: Vec<u8>,
    pub(crate) outcome: FetchOutcome,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Roots,
    Timestamp,
    Snapshot,
    Targets,
    Done,
    Failed,
}

#[derive(Debug)]
struct State {
    phase: Phase,
    next_root: u64,
    records: Vec<FetchRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct CapturingTransport {
    metadata_base: Url,
    responses: Arc<HashMap<String, ScriptedResponse>>,
    state: Arc<Mutex<State>>,
}

impl CapturingTransport {
    pub(crate) fn new(
        metadata_base: Url,
        trusted_root_version: u64,
        responses: HashMap<String, ScriptedResponse>,
    ) -> Self {
        Self {
            metadata_base,
            responses: Arc::new(responses),
            state: Arc::new(Mutex::new(State {
                phase: Phase::Roots,
                next_root: trusted_root_version
                    .checked_add(1)
                    .expect("fixture root version must not overflow"),
                records: Vec::new(),
            })),
        }
    }

    pub(crate) fn records(&self) -> Vec<FetchRecord> {
        self.state
            .lock()
            .expect("transport state poisoned")
            .records
            .clone()
    }

    pub(crate) fn metadata_base(&self) -> &Url {
        &self.metadata_base
    }

    pub(crate) fn validate_complete(&self) -> Result<(), SpikeError> {
        let state = self.state.lock().expect("transport state poisoned");
        if state.phase != Phase::Done {
            return Err(SpikeError::TransportPolicy);
        }
        let saw_real_root_terminator = state.records.iter().any(|record| {
            record.role == Some(RoleKind::Root) && record.outcome == FetchOutcome::NotFound
        });
        if !saw_real_root_terminator {
            return Err(SpikeError::TransportPolicy);
        }
        Ok(())
    }

    fn request_name(&self, url: &Url) -> Result<String, ()> {
        if url.scheme() != self.metadata_base.scheme()
            || url.host_str() != self.metadata_base.host_str()
            || url.port_or_known_default() != self.metadata_base.port_or_known_default()
            || url.query().is_some()
            || url.fragment().is_some()
            || !url.path().starts_with(self.metadata_base.path())
        {
            return Err(());
        }
        let name = url
            .path()
            .strip_prefix(self.metadata_base.path())
            .ok_or(())?
            .trim_start_matches('/');
        if name.is_empty() || name.contains('/') {
            return Err(());
        }
        Ok(name.to_string())
    }

    fn classify(state: &mut State, name: &str) -> Result<RoleKind, ()> {
        match state.phase {
            Phase::Roots => {
                let expected = format!("{}.root.json", state.next_root);
                if name == expected {
                    Ok(RoleKind::Root)
                } else {
                    Err(())
                }
            }
            Phase::Timestamp if name == "timestamp.json" => Ok(RoleKind::Timestamp),
            Phase::Snapshot if metadata_name_matches(name, "snapshot.json") => {
                Ok(RoleKind::Snapshot)
            }
            Phase::Targets if metadata_name_matches(name, "targets.json") => Ok(RoleKind::Targets),
            _ => Err(()),
        }
    }
}

fn metadata_name_matches(name: &str, suffix: &str) -> bool {
    name == suffix
        || name
            .strip_suffix(suffix)
            .and_then(|prefix| prefix.strip_suffix('.'))
            .is_some_and(|version| {
                !version.is_empty() && version.bytes().all(|b| b.is_ascii_digit())
            })
}

#[async_trait]
impl Transport for CapturingTransport {
    async fn fetch(&self, url: Url) -> Result<TransportStream, TransportError> {
        let name = match self.request_name(&url) {
            Ok(name) => name,
            Err(()) => {
                self.state.lock().expect("transport state poisoned").phase = Phase::Failed;
                return Err(TransportError::new(TransportErrorKind::Other, url.as_str()));
            }
        };
        let mut state = self.state.lock().expect("transport state poisoned");
        let role = match Self::classify(&mut state, &name) {
            Ok(role) => role,
            Err(()) => {
                state.records.push(FetchRecord {
                    request_name: name,
                    role: None,
                    bytes: Vec::new(),
                    outcome: FetchOutcome::Rejected,
                });
                state.phase = Phase::Failed;
                return Err(TransportError::new(TransportErrorKind::Other, url.as_str()));
            }
        };
        let response = self
            .responses
            .get(&name)
            .cloned()
            .unwrap_or(ScriptedResponse::NotFound);
        match response {
            ScriptedResponse::Bytes(bytes) => {
                if strict_json::validate(&bytes, role.max_bytes()).is_err() {
                    state.records.push(FetchRecord {
                        request_name: name,
                        role: Some(role),
                        bytes: Vec::new(),
                        outcome: FetchOutcome::Rejected,
                    });
                    state.phase = Phase::Failed;
                    return Err(TransportError::new(TransportErrorKind::Other, url.as_str()));
                }
                match role {
                    RoleKind::Root => {
                        state.next_root = state.next_root.checked_add(1).ok_or_else(|| {
                            TransportError::new(TransportErrorKind::Other, url.as_str())
                        })?
                    }
                    RoleKind::Timestamp => state.phase = Phase::Snapshot,
                    RoleKind::Snapshot => state.phase = Phase::Targets,
                    RoleKind::Targets => state.phase = Phase::Done,
                }
                state.records.push(FetchRecord {
                    request_name: name,
                    role: Some(role),
                    bytes: bytes.clone(),
                    outcome: FetchOutcome::Complete,
                });
                Ok(Box::pin(futures::stream::iter(vec![Ok(Bytes::from(
                    bytes,
                ))])))
            }
            ScriptedResponse::NotFound => {
                state.records.push(FetchRecord {
                    request_name: name,
                    role: Some(role),
                    bytes: Vec::new(),
                    outcome: FetchOutcome::NotFound,
                });
                if role == RoleKind::Root {
                    state.phase = Phase::Timestamp;
                } else {
                    state.phase = Phase::Failed;
                }
                Err(TransportError::new(
                    TransportErrorKind::FileNotFound,
                    url.as_str(),
                ))
            }
            ScriptedResponse::Other => {
                state.records.push(FetchRecord {
                    request_name: name,
                    role: Some(role),
                    bytes: Vec::new(),
                    outcome: FetchOutcome::Other,
                });
                state.phase = Phase::Failed;
                Err(TransportError::new(TransportErrorKind::Other, url.as_str()))
            }
        }
    }
}
