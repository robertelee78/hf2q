//! Bounded server-authoritative artifact selections for diagnostic chat.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::input::hf_download::{HubGgufArtifact, HubGgufCatalog};
use crate::serve::api::local_artifacts::{LocalArtifactCatalog, LocalGgufArtifact};

pub const ARTIFACT_CATALOG_SCHEMA: &str = "hf2q.artifact-resolution.v2";
pub const LOCAL_ARTIFACT_CATALOG_SCHEMA: &str = "hf2q.local-artifact-resolution.v1";
const CANDIDATE_TTL: Duration = Duration::from_secs(10 * 60);
const MAX_CANDIDATES: usize = 512;
const HUB_CONCURRENCY: usize = 2;
const HUB_TRANSFER_CONCURRENCY: usize = 2;
const LOCAL_VERIFY_CONCURRENCY: usize = 1;

#[derive(Clone, Debug)]
pub enum StoredArtifact {
    Hosted(HubGgufArtifact),
    Local(LocalGgufArtifact),
}

#[derive(Clone, Debug)]
struct StoredCandidate {
    artifact: StoredArtifact,
    issued_at: Instant,
}

#[derive(Debug, Default)]
struct CatalogState {
    candidates: HashMap<String, StoredCandidate>,
}

/// The server retains exact revision/size/SHA descriptors and gives clients
/// only opaque candidate ids. A client therefore cannot downgrade immutable
/// catalog facts or make activation re-resolve a mutable branch.
#[derive(Clone, Debug)]
pub struct ArtifactCatalogCoordinator {
    state: Arc<Mutex<CatalogState>>,
    hub_slots: Arc<Semaphore>,
    transfer_slots: Arc<Semaphore>,
    local_verify_slots: Arc<Semaphore>,
}

impl Default for ArtifactCatalogCoordinator {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(CatalogState::default())),
            hub_slots: Arc::new(Semaphore::new(HUB_CONCURRENCY)),
            transfer_slots: Arc::new(Semaphore::new(HUB_TRANSFER_CONCURRENCY)),
            local_verify_slots: Arc::new(Semaphore::new(LOCAL_VERIFY_CONCURRENCY)),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CatalogError {
    #[error("artifact catalog is busy; retry shortly")]
    Busy,
    #[error("hosted artifact transfers are busy; retry shortly")]
    TransferBusy,
    #[error("local artifact verification is busy; retry shortly")]
    LocalVerifyBusy,
    #[error("artifact selection expired or is unknown; refresh the catalog")]
    Gone,
    #[error("artifact catalog state is unavailable")]
    State,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ArtifactCatalogView {
    pub schema_version: &'static str,
    pub repository: String,
    pub revision: String,
    pub candidates: Vec<ArtifactCandidateView>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct ArtifactCandidateView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
    pub filename: String,
    pub bytes: u64,
    pub quant_hint: Option<String>,
    pub role: String,
    pub selectable: bool,
    pub unavailable_reason: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct LocalArtifactCatalogView {
    pub schema_version: &'static str,
    pub candidates: Vec<LocalArtifactCandidateView>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct LocalArtifactCandidateView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
    pub repository: String,
    pub revision: String,
    pub filename: String,
    pub bytes: u64,
    pub quant_hint: String,
    pub origin: String,
    pub role: String,
    pub selectable: bool,
    pub unavailable_reason: Option<String>,
}

impl ArtifactCatalogCoordinator {
    pub fn try_hub_slot(&self) -> Result<OwnedSemaphorePermit, CatalogError> {
        Arc::clone(&self.hub_slots)
            .try_acquire_owned()
            .map_err(|_| CatalogError::Busy)
    }

    pub fn try_transfer_slot(&self) -> Result<OwnedSemaphorePermit, CatalogError> {
        Arc::clone(&self.transfer_slots)
            .try_acquire_owned()
            .map_err(|_| CatalogError::TransferBusy)
    }

    pub fn try_local_verify_slot(&self) -> Result<OwnedSemaphorePermit, CatalogError> {
        Arc::clone(&self.local_verify_slots)
            .try_acquire_owned()
            .map_err(|_| CatalogError::LocalVerifyBusy)
    }

    pub fn register_hosted(
        &self,
        catalog: HubGgufCatalog,
    ) -> Result<ArtifactCatalogView, CatalogError> {
        let now = Instant::now();
        let mut state = self.state.lock().map_err(|_| CatalogError::State)?;
        state
            .candidates
            .retain(|_, candidate| now.duration_since(candidate.issued_at) < CANDIDATE_TTL);
        let issued = catalog
            .artifacts
            .iter()
            .filter(|artifact| artifact.selectable)
            .count();
        if issued > MAX_CANDIDATES {
            return Err(CatalogError::State);
        }
        while state.candidates.len() + issued > MAX_CANDIDATES {
            remove_oldest(&mut state.candidates);
        }
        let mut candidates = Vec::with_capacity(catalog.artifacts.len());
        for artifact in catalog.artifacts {
            let candidate_id = if artifact.selectable {
                let id = uuid::Uuid::new_v4().to_string();
                state.candidates.insert(
                    id.clone(),
                    StoredCandidate {
                        artifact: StoredArtifact::Hosted(artifact.clone()),
                        issued_at: now,
                    },
                );
                Some(id)
            } else {
                None
            };
            candidates.push(ArtifactCandidateView {
                candidate_id,
                filename: artifact.filename,
                bytes: artifact.bytes,
                quant_hint: artifact.quant_hint,
                role: artifact.role,
                selectable: artifact.selectable,
                unavailable_reason: artifact.unavailable_reason,
            });
        }
        Ok(ArtifactCatalogView {
            schema_version: ARTIFACT_CATALOG_SCHEMA,
            repository: catalog.repository,
            revision: catalog.revision,
            candidates,
        })
    }

    pub fn register_local(
        &self,
        catalog: LocalArtifactCatalog,
    ) -> Result<LocalArtifactCatalogView, CatalogError> {
        let now = Instant::now();
        let mut state = self.state.lock().map_err(|_| CatalogError::State)?;
        prune_expired(&mut state, now);
        let issued = catalog
            .artifacts
            .iter()
            .filter(|artifact| artifact.selectable)
            .count();
        if issued > MAX_CANDIDATES {
            return Err(CatalogError::State);
        }
        while state.candidates.len() + issued > MAX_CANDIDATES {
            remove_oldest(&mut state.candidates);
        }
        let mut candidates = Vec::with_capacity(catalog.artifacts.len());
        for artifact in catalog.artifacts {
            let candidate_id = if artifact.selectable {
                let id = uuid::Uuid::new_v4().to_string();
                state.candidates.insert(
                    id.clone(),
                    StoredCandidate {
                        artifact: StoredArtifact::Local(artifact.clone()),
                        issued_at: now,
                    },
                );
                Some(id)
            } else {
                None
            };
            candidates.push(LocalArtifactCandidateView {
                candidate_id,
                repository: artifact.repository,
                revision: artifact.revision,
                filename: artifact.filename,
                bytes: artifact.bytes,
                quant_hint: artifact.quant_hint,
                origin: artifact.provenance.as_str().to_owned(),
                role: "text_model".into(),
                selectable: artifact.selectable,
                unavailable_reason: artifact.unavailable_reason,
            });
        }
        Ok(LocalArtifactCatalogView {
            schema_version: LOCAL_ARTIFACT_CATALOG_SCHEMA,
            candidates,
            warnings: catalog.warnings,
        })
    }

    pub fn resolve(&self, candidate_id: &str) -> Result<StoredArtifact, CatalogError> {
        let now = Instant::now();
        let mut state = self.state.lock().map_err(|_| CatalogError::State)?;
        let candidate = state
            .candidates
            .get(candidate_id)
            .filter(|candidate| now.duration_since(candidate.issued_at) < CANDIDATE_TTL)
            .cloned();
        if candidate.is_none() {
            state.candidates.remove(candidate_id);
        }
        candidate
            .map(|candidate| candidate.artifact)
            .ok_or(CatalogError::Gone)
    }
}

fn prune_expired(state: &mut CatalogState, now: Instant) {
    state
        .candidates
        .retain(|_, candidate| now.duration_since(candidate.issued_at) < CANDIDATE_TTL);
}

fn remove_oldest(candidates: &mut HashMap<String, StoredCandidate>) {
    let oldest = candidates
        .iter()
        .map(|(id, candidate)| (candidate.issued_at, id.clone()))
        .min_by_key(|(issued_at, _)| *issued_at);
    if let Some((_, id)) = oldest {
        candidates.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact(selectable: bool) -> HubGgufArtifact {
        HubGgufArtifact {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            filename: if selectable {
                "model-q5_k_m.gguf".into()
            } else {
                "model-bf16.gguf".into()
            },
            bytes: 42,
            sha256: "b".repeat(64),
            quant_hint: Some(if selectable { "Q5_K_M" } else { "BF16" }.into()),
            role: "text_model".into(),
            selectable,
            unavailable_reason: (!selectable).then(|| "unsupported".into()),
        }
    }

    #[test]
    fn only_server_selectable_candidates_receive_opaque_ids() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let view = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts: vec![artifact(true), artifact(false)],
            })
            .unwrap();
        let selected = view.candidates[0].candidate_id.as_deref().unwrap();
        let StoredArtifact::Hosted(selected) = coordinator.resolve(selected).unwrap() else {
            panic!("expected hosted authority");
        };
        assert_eq!(selected.sha256, "b".repeat(64));
        assert!(view.candidates[1].candidate_id.is_none());
    }

    #[test]
    fn earlier_candidate_remains_bound_to_its_immutable_catalog_revision() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let first = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts: vec![artifact(true)],
            })
            .unwrap();
        let first_id = first.candidates[0].candidate_id.as_deref().unwrap();

        let mut replacement = artifact(true);
        replacement.revision = "c".repeat(40);
        replacement.sha256 = "d".repeat(64);
        coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "c".repeat(40),
                artifacts: vec![replacement],
            })
            .unwrap();

        let StoredArtifact::Hosted(selected) = coordinator.resolve(first_id).unwrap() else {
            panic!("expected hosted authority");
        };
        assert_eq!(selected.revision, "a".repeat(40));
        assert_eq!(selected.sha256, "b".repeat(64));
    }

    #[test]
    fn local_catalog_exposes_only_opaque_authority_and_safe_metadata() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let private_root = std::path::PathBuf::from("/private/operator/models");
        let private_path = private_root.join("secret/model.gguf");
        let view = coordinator
            .register_local(LocalArtifactCatalog {
                artifacts: vec![LocalGgufArtifact {
                    repository: "owner/model".into(),
                    revision: "a".repeat(40),
                    filename: "model.gguf".into(),
                    root: private_root.clone(),
                    path: private_path.clone(),
                    bytes: 42,
                    sha256: "b".repeat(64),
                    quant_hint: "Q4_K_M".into(),
                    quant: Some(crate::serve::quant_select::QuantType::Q4_K_M),
                    selectable: true,
                    unavailable_reason: None,
                    provenance: crate::serve::api::local_artifacts::LocalArtifactProvenance::ConversionReceipt,
                }],
                warnings: Vec::new(),
            })
            .unwrap();
        let encoded = serde_json::to_string(&view).unwrap();
        assert!(!encoded.contains(private_root.to_str().unwrap()));
        assert!(!encoded.contains(private_path.to_str().unwrap()));
        assert!(!encoded.contains(&"b".repeat(64)));
        let id = view.candidates[0].candidate_id.as_deref().unwrap();
        let StoredArtifact::Local(selected) = coordinator.resolve(id).unwrap() else {
            panic!("expected local authority");
        };
        assert_eq!(selected.path, private_path);
    }

    #[test]
    fn every_id_returned_by_a_successful_catalog_resolves_immediately() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let mut artifacts = Vec::new();
        for index in 0..MAX_CANDIDATES {
            let mut candidate = artifact(true);
            candidate.filename = format!("model-{index:03}-q6_k.gguf");
            candidate.sha256 = format!("{index:064x}");
            artifacts.push(candidate);
        }
        let view = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts,
            })
            .unwrap();
        assert_eq!(view.candidates.len(), MAX_CANDIDATES);
        for candidate in &view.candidates {
            coordinator
                .resolve(candidate.candidate_id.as_deref().unwrap())
                .unwrap();
        }
    }

    #[test]
    fn oversized_single_catalog_is_rejected_without_issuing_stale_ids() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let artifacts = (0..=MAX_CANDIDATES).map(|_| artifact(true)).collect();
        let error = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts,
            })
            .unwrap_err();
        assert!(matches!(error, CatalogError::State));
    }

    #[test]
    fn hosted_transfer_children_are_bounded_independently_from_catalog_helpers() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let first = coordinator.try_transfer_slot().unwrap();
        let second = coordinator.try_transfer_slot().unwrap();
        assert!(matches!(
            coordinator.try_transfer_slot().unwrap_err(),
            CatalogError::TransferBusy
        ));
        drop(first);
        let replacement = coordinator.try_transfer_slot().unwrap();
        drop((second, replacement));
    }

    #[test]
    fn catalog_helper_slots_are_bounded_and_released() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let first = coordinator.try_hub_slot().unwrap();
        let second = coordinator.try_hub_slot().unwrap();
        assert!(matches!(
            coordinator.try_hub_slot().unwrap_err(),
            CatalogError::Busy
        ));
        drop(first);
        let replacement = coordinator.try_hub_slot().unwrap();
        drop((second, replacement));
    }

    #[test]
    fn expired_candidate_fails_closed_as_gone() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let view = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts: vec![artifact(true)],
            })
            .unwrap();
        let candidate_id = view.candidates[0].candidate_id.as_deref().unwrap();
        coordinator
            .state
            .lock()
            .unwrap()
            .candidates
            .get_mut(candidate_id)
            .unwrap()
            .issued_at = Instant::now() - CANDIDATE_TTL - Duration::from_secs(1);
        assert!(matches!(
            coordinator.resolve(candidate_id).unwrap_err(),
            CatalogError::Gone
        ));
    }
}
