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
    Hosted(HostedArtifactPair),
    Local(LocalGgufArtifact),
}

/// Server-private atomic activation authority. The client selects only the
/// text candidate. After the immutable text bytes are fetched, its exact
/// projector digest selects at most one companion from this same-revision
/// inventory; clients can never construct an invalid pair.
#[derive(Clone, Debug)]
pub struct HostedArtifactPair {
    pub text: HubGgufArtifact,
    pub companions: Vec<HubGgufArtifact>,
    /// Path-free private result of inspecting this immutable text artifact and
    /// its digest-bound projector, if any. Hub inventory metadata does not
    /// expose GGUF header keys or tensor layout, so this begins unresolved and
    /// is recorded after the first authenticated materialization.
    pub resolution: Option<HostedPairResolution>,
}

/// Immutable, path-free admission authority discovered from a hosted pair.
/// This stays server-private; public artifact schema v2 continues to expose
/// one opaque text candidate and the text object's transfer size only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostedPairResolution {
    companion_sha256: Option<String>,
    projected_weight_bytes: u64,
    cache_budget_bytes: u64,
    activation_bytes: u64,
}

impl HostedPairResolution {
    pub fn text_only(text_bytes: u64) -> Self {
        Self {
            companion_sha256: None,
            projected_weight_bytes: 0,
            cache_budget_bytes: 0,
            activation_bytes: text_bytes,
        }
    }

    pub fn projector(
        text_bytes: u64,
        companion_sha256: String,
        projected_weight_bytes: u64,
        cache_budget_bytes: u64,
    ) -> Result<Self, CatalogError> {
        if companion_sha256.len() != 64
            || !companion_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(CatalogError::AuthorityMismatch);
        }
        let activation_bytes = text_bytes
            .checked_add(projected_weight_bytes)
            .and_then(|bytes| bytes.checked_add(cache_budget_bytes))
            .ok_or(CatalogError::AuthorityMismatch)?;
        Ok(Self {
            companion_sha256: Some(companion_sha256.to_ascii_lowercase()),
            projected_weight_bytes,
            cache_budget_bytes,
            activation_bytes,
        })
    }

    pub fn activation_bytes(&self) -> u64 {
        self.activation_bytes
    }

    pub fn companion_sha256(&self) -> Option<&str> {
        self.companion_sha256.as_deref()
    }

    pub fn projected_weight_bytes(&self) -> u64 {
        self.projected_weight_bytes
    }

    pub fn cache_budget_bytes(&self) -> u64 {
        self.cache_budget_bytes
    }
}

#[derive(Clone, Debug)]
struct StoredCandidate {
    artifact: StoredArtifact,
    issued_at: Instant,
    active_leases: usize,
}

/// Keeps one already-resolved opaque candidate alive across authenticated
/// verification or transfer. The token is server-private and non-cloneable.
#[derive(Debug)]
pub struct ArtifactCandidateLease {
    candidate_id: String,
    state: Arc<Mutex<CatalogState>>,
}

impl Drop for ArtifactCandidateLease {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(candidate) = state.candidates.get_mut(&self.candidate_id) {
                candidate.active_leases = candidate.active_leases.saturating_sub(1);
            }
        }
    }
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
    #[error("hosted model/projector authority does not match its immutable candidate")]
    AuthorityMismatch,
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
        prune_expired(&mut state, now);
        let issued = catalog
            .artifacts
            .iter()
            .filter(|artifact| artifact.selectable)
            .count();
        if issued > MAX_CANDIDATES {
            return Err(CatalogError::State);
        }
        if !can_make_room(&state.candidates, issued) {
            return Err(CatalogError::State);
        }
        while state.candidates.len() + issued > MAX_CANDIDATES {
            if !remove_oldest(&mut state.candidates) {
                return Err(CatalogError::State);
            }
        }
        let mut candidates = Vec::with_capacity(catalog.artifacts.len());
        for artifact in catalog.artifacts.iter().cloned() {
            let candidate_id = if artifact.selectable {
                let companions = catalog
                    .artifacts
                    .iter()
                    .filter(|companion| {
                        companion.role == "companion"
                            && companion.repository == artifact.repository
                            && companion.revision == artifact.revision
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                let id = uuid::Uuid::new_v4().to_string();
                state.candidates.insert(
                    id.clone(),
                    StoredCandidate {
                        artifact: StoredArtifact::Hosted(HostedArtifactPair {
                            text: artifact.clone(),
                            companions,
                            resolution: None,
                        }),
                        issued_at: now,
                        active_leases: 0,
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
        if !can_make_room(&state.candidates, issued) {
            return Err(CatalogError::State);
        }
        while state.candidates.len() + issued > MAX_CANDIDATES {
            if !remove_oldest(&mut state.candidates) {
                return Err(CatalogError::State);
            }
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
                        active_leases: 0,
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
            .get_mut(candidate_id)
            .filter(|candidate| now.duration_since(candidate.issued_at) < CANDIDATE_TTL)
            .map(|candidate| {
                // A successful non-transactional access refreshes the opaque
                // candidate. Activation uses `resolve_pinned` below.
                candidate.issued_at = now;
                candidate.clone()
            });
        if candidate.is_none() {
            state.candidates.remove(candidate_id);
        }
        candidate
            .map(|candidate| candidate.artifact)
            .ok_or(CatalogError::Gone)
    }

    /// Resolve and pin a candidate for one in-flight activation transaction.
    /// Catalog TTL pruning and bounded-capacity replacement skip pinned
    /// candidates until the returned token is dropped.
    pub fn resolve_pinned(
        &self,
        candidate_id: &str,
    ) -> Result<(StoredArtifact, ArtifactCandidateLease), CatalogError> {
        let now = Instant::now();
        let artifact = {
            let mut state = self.state.lock().map_err(|_| CatalogError::State)?;
            let candidate = state
                .candidates
                .get_mut(candidate_id)
                .filter(|candidate| now.duration_since(candidate.issued_at) < CANDIDATE_TTL);
            let Some(candidate) = candidate else {
                state.candidates.remove(candidate_id);
                return Err(CatalogError::Gone);
            };
            candidate.issued_at = now;
            candidate.active_leases = candidate
                .active_leases
                .checked_add(1)
                .ok_or(CatalogError::State)?;
            candidate.artifact.clone()
        };
        Ok((
            artifact,
            ArtifactCandidateLease {
                candidate_id: candidate_id.to_owned(),
                state: Arc::clone(&self.state),
            },
        ))
    }

    /// Bind one authenticated, header-derived resolution to its opaque
    /// candidate. Re-observing the identical resolution is idempotent; any
    /// different text identity, companion, or byte contract fails closed.
    pub fn record_hosted_resolution(
        &self,
        candidate_id: &str,
        immutable_authority: &HostedArtifactPair,
        resolution: HostedPairResolution,
    ) -> Result<(), CatalogError> {
        let now = Instant::now();
        let mut state = self.state.lock().map_err(|_| CatalogError::State)?;
        let candidate = state
            .candidates
            .get_mut(candidate_id)
            .ok_or(CatalogError::Gone)?;
        let StoredArtifact::Hosted(pair) = &mut candidate.artifact else {
            return Err(CatalogError::AuthorityMismatch);
        };
        if pair.text != immutable_authority.text
            || pair.companions != immutable_authority.companions
        {
            return Err(CatalogError::AuthorityMismatch);
        }
        match resolution.companion_sha256() {
            Some(digest)
                if !pair
                    .companions
                    .iter()
                    .any(|companion| companion.sha256.eq_ignore_ascii_case(digest)) =>
            {
                return Err(CatalogError::AuthorityMismatch);
            }
            None if resolution.projected_weight_bytes() != 0
                || resolution.cache_budget_bytes() != 0 =>
            {
                return Err(CatalogError::AuthorityMismatch);
            }
            _ => {}
        }
        if pair
            .resolution
            .as_ref()
            .is_some_and(|current| current != &resolution)
        {
            return Err(CatalogError::AuthorityMismatch);
        }
        pair.resolution = Some(resolution);
        candidate.issued_at = now;
        Ok(())
    }
}

fn prune_expired(state: &mut CatalogState, now: Instant) {
    state.candidates.retain(|_, candidate| {
        candidate.active_leases > 0 || now.duration_since(candidate.issued_at) < CANDIDATE_TTL
    });
}

fn remove_oldest(candidates: &mut HashMap<String, StoredCandidate>) -> bool {
    let oldest = candidates
        .iter()
        .filter(|(_, candidate)| candidate.active_leases == 0)
        .map(|(id, candidate)| (candidate.issued_at, id.clone()))
        .min_by_key(|(issued_at, _)| *issued_at);
    if let Some((_, id)) = oldest {
        candidates.remove(&id);
        true
    } else {
        false
    }
}

fn can_make_room(candidates: &HashMap<String, StoredCandidate>, additional: usize) -> bool {
    let required_evictions = candidates
        .len()
        .saturating_add(additional)
        .saturating_sub(MAX_CANDIDATES);
    candidates
        .values()
        .filter(|candidate| candidate.active_leases == 0)
        .count()
        >= required_evictions
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
                "mmproj-model-f16.gguf".into()
            },
            bytes: 42,
            sha256: "b".repeat(64),
            quant_hint: selectable.then(|| "Q5_K_M".into()),
            role: if selectable {
                "text_model"
            } else {
                "companion"
            }
            .into(),
            selectable,
            unavailable_reason: (!selectable).then(|| "unsupported".into()),
        }
    }

    fn companion() -> HubGgufArtifact {
        HubGgufArtifact {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            filename: "model-q5_k_m-mmproj.gguf".into(),
            bytes: 24,
            sha256: "c".repeat(64),
            quant_hint: None,
            role: "companion".into(),
            selectable: false,
            unavailable_reason: Some("vision projector companion; not a text model".into()),
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
        assert_eq!(selected.text.sha256, "b".repeat(64));
        assert_eq!(selected.companions.len(), 1);
        assert_eq!(selected.companions[0].role, "companion");
        assert!(view.candidates[1].candidate_id.is_none());
    }

    #[test]
    fn one_opaque_text_candidate_retains_server_private_companion_authority() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let mut cross_revision = companion();
        cross_revision.revision = "d".repeat(40);
        let view = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts: vec![artifact(true), companion(), cross_revision],
            })
            .unwrap();
        let text_id = view.candidates[0].candidate_id.as_deref().unwrap();
        assert!(view.candidates[1].candidate_id.is_none());
        assert!(view.candidates[2].candidate_id.is_none());
        let StoredArtifact::Hosted(pair) = coordinator.resolve(text_id).unwrap() else {
            panic!("expected hosted pair authority");
        };
        assert_eq!(pair.text.role, "text_model");
        assert_eq!(pair.companions.len(), 1);
        assert_eq!(pair.companions[0].sha256, "c".repeat(64));
        assert_eq!(pair.companions[0].revision, pair.text.revision);
        assert!(pair.resolution.is_none());
    }

    #[test]
    fn hosted_pair_discovery_is_path_free_immutable_and_refreshes_exact_admission() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let view = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts: vec![artifact(true), companion()],
            })
            .unwrap();
        let id = view.candidates[0].candidate_id.as_deref().unwrap();
        let StoredArtifact::Hosted(authority) = coordinator.resolve(id).unwrap() else {
            panic!("expected hosted authority");
        };
        let resolution = HostedPairResolution::projector(42, "c".repeat(64), 100, 512).unwrap();
        coordinator
            .record_hosted_resolution(id, &authority, resolution.clone())
            .unwrap();

        let StoredArtifact::Hosted(pair) = coordinator.resolve(id).unwrap() else {
            panic!("expected hosted pair authority");
        };
        assert_eq!(pair.resolution, Some(resolution));
        assert_eq!(pair.resolution.unwrap().activation_bytes(), 654);
        // The public v2 candidate remains the opaque text activation unit.
        let encoded = serde_json::to_string(&view).unwrap();
        assert!(!encoded.contains(&"c".repeat(64)));
        assert!(!encoded.contains("654"));
    }

    #[test]
    fn hosted_pair_discovery_rejects_wrong_text_or_rebinding() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let view = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts: vec![artifact(true), companion()],
            })
            .unwrap();
        let id = view.candidates[0].candidate_id.as_deref().unwrap();
        let StoredArtifact::Hosted(authority) = coordinator.resolve(id).unwrap() else {
            panic!("expected hosted authority");
        };
        let resolution = HostedPairResolution::projector(42, "c".repeat(64), 100, 512).unwrap();
        let mut wrong_text = authority.clone();
        wrong_text.text.sha256 = "d".repeat(64);
        assert!(matches!(
            coordinator.record_hosted_resolution(id, &wrong_text, resolution.clone()),
            Err(CatalogError::AuthorityMismatch)
        ));
        coordinator
            .record_hosted_resolution(id, &authority, resolution)
            .unwrap();
        assert!(matches!(
            coordinator.record_hosted_resolution(
                id,
                &authority,
                HostedPairResolution::text_only(42)
            ),
            Err(CatalogError::AuthorityMismatch)
        ));
    }

    #[test]
    fn activation_lease_prevents_ttl_prune_until_materialization_finishes() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let view = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts: vec![artifact(true), companion()],
            })
            .unwrap();
        let id = view.candidates[0].candidate_id.as_deref().unwrap();
        let (_artifact, lease) = coordinator.resolve_pinned(id).unwrap();
        {
            let mut state = coordinator.state.lock().unwrap();
            state.candidates.get_mut(id).unwrap().issued_at =
                Instant::now() - CANDIDATE_TTL - Duration::from_secs(1);
            prune_expired(&mut state, Instant::now());
            assert_eq!(state.candidates.get(id).unwrap().active_leases, 1);
        }

        drop(lease);
        let mut state = coordinator.state.lock().unwrap();
        prune_expired(&mut state, Instant::now());
        assert!(!state.candidates.contains_key(id));
    }

    #[test]
    fn catalog_pressure_cannot_evict_pinned_activations_or_partially_mutate() {
        let coordinator = ArtifactCatalogCoordinator::default();
        let view = coordinator
            .register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                artifacts: vec![artifact(true), companion()],
            })
            .unwrap();
        let id = view.candidates[0].candidate_id.as_deref().unwrap();
        let (_artifact, _lease) = coordinator.resolve_pinned(id).unwrap();
        let before = {
            let mut state = coordinator.state.lock().unwrap();
            let template = state.candidates.get(id).unwrap().clone();
            for index in 1..MAX_CANDIDATES {
                state
                    .candidates
                    .insert(format!("pinned-{index:03}"), template.clone());
            }
            let mut keys = state.candidates.keys().cloned().collect::<Vec<_>>();
            keys.sort();
            keys
        };

        assert!(matches!(
            coordinator.register_hosted(HubGgufCatalog {
                schema_version: "hf2q.hub-gguf-catalog.v2".into(),
                repository: "owner/other".into(),
                revision: "d".repeat(40),
                artifacts: vec![artifact(true)],
            }),
            Err(CatalogError::State)
        ));
        let mut after = coordinator
            .state
            .lock()
            .unwrap()
            .candidates
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        after.sort();
        assert_eq!(after, before);
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
        assert_eq!(selected.text.revision, "a".repeat(40));
        assert_eq!(selected.text.sha256, "b".repeat(64));
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
