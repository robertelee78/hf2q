//! ADR-047 authenticated, versioned diagnostic lifecycle control plane.

use std::sync::Arc;

use axum::extract::{Extension, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::artifact_catalog::{
    ArtifactCatalogCoordinator, CatalogError, StoredArtifact, ARTIFACT_CATALOG_SCHEMA,
    LOCAL_ARTIFACT_CATALOG_SCHEMA,
};
use super::cancellation::{
    run_consistent_result_task, run_preparation_command, PreparationError, PreparationLimits,
};
use super::handlers::map_hotswap_error_to_response;
use super::lifecycle::{LifecycleError, SwitchConfirmation};
use super::middleware::RequestCancellation;
use super::schema::ApiError;
use super::state::AppState;
use super::{DIAGNOSTIC_NO_EVICT_HEADER, DIAGNOSTIC_NO_EVICT_VALUE};
use crate::serve::auto_pipeline;
use crate::serve::multi_model::{
    AdmissionOutcome, EngineConfig, HotSwapError, LoadedSummary, NonEvictingLoad,
};
use crate::serve::quant_select::QuantType;

const HF2Q_RUNTIME_SCHEMA: &str = "hf2q.runtime.v1";
const HF2Q_ACTIVATION_SCHEMA: &str = "hf2q.model-activation.v2";

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelActivationRequest {
    pub model: String,
    /// Opaque server-issued id for an exact revision/size/SHA artifact.
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub action: ModelActivationAction,
    pub expected_revision: Option<u64>,
    #[serde(default)]
    pub victims: Vec<ModelActivationVictim>,
}

#[derive(Debug, serde::Deserialize)]
pub struct HubGgufCatalogQuery {
    pub model: String,
}

#[derive(Debug, serde::Deserialize)]
pub struct LocalGgufCatalogQuery {
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelActivationAction {
    #[default]
    Load,
    Switch,
    Probe,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelActivationVictim {
    pub pool_key: String,
    pub quant: String,
    pub bytes_resident: u64,
    pub generation: u64,
}

fn normalize_hub_repository(model: &str) -> std::result::Result<Option<String>, Response> {
    let model = model.trim();
    if !model.contains("://") && !auto_pipeline::looks_like_hf_repo_id(model) {
        return Ok(None);
    }
    let reference =
        crate::input::hf_reference::HfModelReference::parse(model, None).map_err(|error| {
            ApiError::invalid_request(
                format!("invalid Hugging Face model reference: {error}"),
                Some("model".into()),
            )
            .into_response()
        })?;
    if reference.requested_revision().is_some() || reference.filename().is_some() {
        return Err(ApiError::invalid_request(
            "diagnostic hosted selection requires a repository id or base model URL; choose the exact artifact in the hf2q picker",
            Some("model".into()),
        )
        .into_response());
    }
    Ok(Some(reference.repo_id().to_owned()))
}

impl From<ModelActivationVictim> for LoadedSummary {
    fn from(value: ModelActivationVictim) -> Self {
        Self {
            pool_key: value.pool_key,
            quant: value.quant,
            bytes_resident: value.bytes_resident,
            generation: value.generation,
        }
    }
}

fn loaded_summary_json(summary: &LoadedSummary) -> serde_json::Value {
    serde_json::json!({
        "pool_key": summary.pool_key,
        "quant": summary.quant,
        "bytes_resident": summary.bytes_resident,
        "generation": summary.generation,
    })
}

fn engine_config_identity_json(
    identity: &crate::serve::multi_model::EngineConfigIdentity,
) -> serde_json::Value {
    let scheduler = match identity.engine_mode {
        crate::serve::api::engine::EngineMode::SerialFifo => {
            serde_json::json!({"mode": "serial_fifo"})
        }
        crate::serve::api::engine::EngineMode::SlotAware { max_slots } => {
            serde_json::json!({"mode": "slot_aware", "max_slots": max_slots})
        }
    };
    let projector = identity.projector.as_ref().map(|projector| {
        serde_json::json!({
            "artifact_sha256": projector.artifact_sha256,
            "source_sha256": projector.source_sha256,
            "pair_generation": projector.pair_generation,
            "profile": projector.profile,
            "weight_bytes": projector.weight_bytes,
            "cache_budget_bytes": projector.cache_budget_bytes,
        })
    });
    serde_json::json!({
        "queue_capacity": identity.queue_capacity,
        "warmup_synchronously": identity.warmup_synchronously,
        "kv_metrics_sink": identity.kv_metrics_sink,
        "scheduler": scheduler,
        "kv_cache_budget_bytes": identity.kv_cache_budget_bytes,
        "explicit_tokenizer": identity.explicit_tokenizer,
        "explicit_config": identity.explicit_config,
        "dwq_overlay": identity.dwq_overlay,
        "projector": projector,
    })
}

fn activation_candidate_json(
    model: &str,
    repo: &str,
    quant: QuantType,
    bytes_resident: u64,
) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "pool_key": format!("{repo}@{}", quant.as_str()),
        "repo": repo,
        "quant": quant.as_str(),
        "bytes_resident": bytes_resident,
    })
}

/// Versioned hf2q capability/runtime view. Existing global Bearer middleware
/// protects this route whenever serve auth is configured; no credential is
/// ever reflected in the response.
pub async fn hf2q_runtime(State(state): State<AppState>) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let manager = match state.pool.read() {
        Ok(manager) => manager,
        Err(_) => return ApiError::internal_error().into_response(),
    };
    let stats = manager.pool_stats();
    let resident = manager
        .snapshot_engines()
        .into_iter()
        .map(|engine| {
            let projector_cache_resident_bytes = engine
                .projector
                .as_ref()
                .map(|projector| projector.vision_cache.resident_bytes() as u64)
                .unwrap_or(0);
            let projector_load_ms = engine
                .projector
                .as_ref()
                .map(|projector| projector.load_duration_ms);
            serde_json::json!({
                "pool_key": format!("{}@{}", engine.repo, engine.quant.as_str()),
                "quant": engine.quant.as_str(),
                "bytes_resident": engine.bytes_resident,
                "resident_components": {
                    "text_gguf_bytes": engine.resident_bytes.text_gguf_bytes,
                    "projector_weight_bytes": engine.resident_bytes.projector_weight_bytes,
                    "projector_cache_budget_bytes": engine.resident_bytes.projector_cache_budget_bytes,
                    "projector_cache_resident_bytes": projector_cache_resident_bytes,
                },
                "projector_load_ms": projector_load_ms,
                "generation": engine.generation,
                "engine_load_ms": engine.engine.info().load_wall_clock.as_millis() as u64,
                "engine_config": engine_config_identity_json(&engine.config_identity),
            })
        })
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "schema_version": HF2Q_RUNTIME_SCHEMA,
            "backend": "mlx-native",
            "capabilities": {
                "model_activation": HF2Q_ACTIVATION_SCHEMA,
                "artifact_resolution": ARTIFACT_CATALOG_SCHEMA,
                "local_artifact_resolution": LOCAL_ARTIFACT_CATALOG_SCHEMA,
                "non_evicting_load": true,
                "diagnostic_no_evict_header": {
                    "name": DIAGNOSTIC_NO_EVICT_HEADER,
                    "value": DIAGNOSTIC_NO_EVICT_VALUE,
                },
                "explicit_revision_bound_switch": true,
                "request_generation_leases": true,
            },
            "pool": {
                "revision": stats.revision,
                "loaded_count": stats.loaded_count,
                "capacity_models": stats.capacity_models,
                "total_resident_bytes": stats.total_resident_bytes,
                "memory_budget_bytes": stats.memory_budget_bytes,
                "resident": resident,
            }
        })),
    )
        .into_response()
}

/// Bounded server-local hf2q artifact inventory. The optional model filter is
/// a bare Hub repository id; omitting it supports the initial diagnostic
/// picker on an empty chat-owned server.
pub async fn local_gguf_catalog(
    State(state): State<AppState>,
    Query(query): Query<LocalGgufCatalogQuery>,
) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let repository = match query.model.as_deref() {
        Some(model) if model.trim().is_empty() => {
            return ApiError::invalid_request("model must not be empty", Some("model".into()))
                .into_response()
        }
        Some(model) => match normalize_hub_repository(model) {
            Ok(Some(repository)) => Some(repository),
            Ok(None) => {
                return ApiError::invalid_request(
                    "local artifact filtering requires an owner/repository model id",
                    Some("model".into()),
                )
                .into_response()
            }
            Err(response) => return response,
        },
        None => None,
    };
    let (cache_root, manifest) = match state.cache.lock() {
        Ok(cache) => {
            let root = cache.root().to_path_buf();
            match cache.manifest_snapshot() {
                Ok(manifest) => (root, manifest),
                Err(error) => {
                    tracing::warn!(%error, "cannot refresh local artifact cache manifest");
                    (root, cache.manifest().clone())
                }
            }
        }
        Err(_) => return ApiError::internal_error().into_response(),
    };
    let inventory = state.local_artifacts.clone();
    let catalog = match tokio::task::spawn_blocking(move || {
        inventory.discover(repository.as_deref(), Some((&cache_root, &manifest)))
    })
    .await
    {
        Ok(catalog) => catalog,
        Err(_) => return ApiError::internal_error().into_response(),
    };
    match state.artifact_catalog.register_local(catalog) {
        Ok(view) => (StatusCode::OK, Json(view)).into_response(),
        Err(_) => ApiError::internal_error().into_response(),
    }
}

/// Metadata-only hosted GGUF inventory. This endpoint never downloads model
/// payload and is protected by the same bearer middleware as activation.
pub async fn hub_gguf_catalog(
    State(state): State<AppState>,
    Extension(cancellation): Extension<RequestCancellation>,
    Query(query): Query<HubGgufCatalogQuery>,
) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let requested_model = query.model.trim();
    if requested_model.is_empty() {
        return ApiError::invalid_request("model must not be empty", Some("model".into()))
            .into_response();
    }
    let model = match normalize_hub_repository(requested_model) {
        Ok(Some(repository)) => repository,
        Ok(None) => {
            return ApiError::invalid_request(
                "hosted artifact catalog requires an owner/repository model id",
                Some("model".into()),
            )
            .into_response()
        }
        Err(response) => return response,
    };
    let slot = match state.artifact_catalog.try_hub_slot() {
        Ok(slot) => slot,
        Err(CatalogError::Busy) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [(header::RETRY_AFTER, "1")],
                Json(serde_json::json!({"error":{"message":"artifact catalog is busy; retry shortly","type":"server_busy","code":"artifact_catalog_busy"}})),
            )
                .into_response()
        }
        Err(_) => return ApiError::internal_error().into_response(),
    };
    let executable = match std::env::current_exe() {
        Ok(executable) => executable,
        Err(_) => return ApiError::internal_error().into_response(),
    };
    let mut command = tokio::process::Command::new(executable);
    command.args(["__catalog-hub-gguf", "--repository", &model]);
    let output = match run_preparation_command(
        command,
        cancellation.0,
        state.preparations.clone(),
        PreparationLimits::catalog(),
        Some(slot),
    )
    .await
    {
        Ok(output) => output,
        Err(PreparationError::Cancelled(_)) => return StatusCode::REQUEST_TIMEOUT.into_response(),
        Err(error) => {
            return ApiError::generation_error(format!(
                "hosted artifact catalog helper failed: {error}"
            ))
            .into_response()
        }
    };
    if !output.status.success() {
        tracing::warn!(
            status = %output.status,
            stderr_bytes = output.stderr.len(),
            "hosted artifact catalog helper failed"
        );
        return ApiError::invalid_request(
            format!(
                "cannot catalog hosted GGUFs (helper exited {}); verify the repository, network access, and HF_TOKEN",
                output.status
            ),
            Some("model".into()),
        )
        .into_response();
    }
    let catalog: crate::input::hf_download::HubGgufCatalog =
        match serde_json::from_slice(&output.stdout) {
            Ok(catalog) => catalog,
            Err(_) => return ApiError::internal_error().into_response(),
        };
    if catalog.schema_version != "hf2q.hub-gguf-catalog.v2" {
        return ApiError::internal_error().into_response();
    }
    match state.artifact_catalog.register_hosted(catalog) {
        Ok(view) => (StatusCode::OK, Json(view)).into_response(),
        Err(_) => ApiError::internal_error().into_response(),
    }
}

fn activation_engine_config(
    state: &AppState,
    path: &std::path::Path,
) -> std::result::Result<EngineConfig, Response> {
    state.engine_config_for_path(path).map_err(|error| {
        tracing::error!(%error, "cannot resolve activation engine load policy");
        ApiError::internal_error().into_response()
    })
}

enum ActivationPayload {
    ExplicitLocal(std::path::PathBuf),
    VerifiedLocal(super::local_artifacts::LocalGgufArtifact),
    Hosted(crate::input::hf_download::HubGgufArtifact),
}

const fn diagnostic_gguf_file_type(quant: QuantType) -> u32 {
    quant.gguf_file_type()
}

fn diagnostic_quant_from_file_type(file_type: u32) -> Option<QuantType> {
    QuantType::from_gguf_file_type(file_type)
}

struct ActivationTarget {
    repo: String,
    quant: QuantType,
    bytes: u64,
    request_model: String,
    projector: Option<crate::serve::multi_model::ProjectorLoadSpec>,
    payload: ActivationPayload,
}

impl ActivationTarget {
    fn verified_local_path(&self) -> Option<&std::path::Path> {
        match &self.payload {
            ActivationPayload::VerifiedLocal(artifact) => Some(&artifact.path),
            _ => None,
        }
    }
}

impl ActivationTarget {
    async fn materialize(
        self,
        cancellation: super::cancellation::CancellationSignal,
        supervisor: super::cancellation::PreparationSupervisor,
        catalog: &ArtifactCatalogCoordinator,
    ) -> std::result::Result<
        (
            String,
            QuantType,
            std::path::PathBuf,
            u64,
            String,
            Option<crate::serve::multi_model::ProjectorLoadSpec>,
        ),
        Response,
    > {
        let mut projector = self.projector;
        let (path, validate_catalog_type) = match self.payload {
            ActivationPayload::ExplicitLocal(path) => (path, false),
            ActivationPayload::VerifiedLocal(artifact) => {
                let permit = catalog.try_local_verify_slot().map_err(|error| match error {
                    CatalogError::LocalVerifyBusy => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        [(header::RETRY_AFTER, "1")],
                        Json(serde_json::json!({"error":{"message":error.to_string(),"type":"server_busy","code":"local_artifact_verification_busy"}})),
                    )
                        .into_response(),
                    _ => ApiError::internal_error().into_response(),
                })?;
                (
                    verify_local_gguf_cancellable(&artifact, cancellation, supervisor, permit)
                        .await?,
                    false,
                )
            }
            ActivationPayload::Hosted(artifact) => {
                let permit = catalog.try_transfer_slot().map_err(|error| match error {
                    CatalogError::TransferBusy => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        [(header::RETRY_AFTER, "1")],
                        Json(serde_json::json!({"error":{"message":error.to_string(),"type":"server_busy","code":"artifact_transfer_busy"}})),
                    )
                        .into_response(),
                    _ => ApiError::internal_error().into_response(),
                })?;
                (
                    fetch_hub_gguf_cancellable(&artifact, cancellation, supervisor, permit).await?,
                    true,
                )
            }
        };
        if !validate_catalog_type {
            return Ok((
                self.repo,
                self.quant,
                path,
                self.bytes,
                self.request_model,
                projector,
            ));
        }
        let actual = mlx_native::gguf::GgufFile::open(&path).map_err(|error| {
            ApiError::invalid_request(
                format!("selected artifact is not a loadable GGUF: {error}"),
                Some("artifact".into()),
            )
            .into_response()
        })?;
        let actual_ftype = actual.metadata_u32("general.file_type");
        let expected_ftype = diagnostic_gguf_file_type(self.quant);
        if actual_ftype != Some(expected_ftype) {
            return Err(ApiError::invalid_request(
                format!(
                    "selected GGUF file type {:?} does not match catalog quant {}",
                    actual_ftype, self.quant
                ),
                Some("artifact".into()),
            )
            .into_response());
        }
        if projector.is_none() {
            let (resolved, _) = activation_projector_spec(&path, self.bytes)?;
            projector = resolved;
        }
        let bytes = activation_pair_bytes(self.bytes, projector.as_ref())?;
        Ok((
            self.repo,
            self.quant,
            path,
            bytes,
            self.request_model,
            projector,
        ))
    }
}

fn activation_projector_spec(
    text_path: &std::path::Path,
    text_bytes: u64,
) -> std::result::Result<(Option<crate::serve::multi_model::ProjectorLoadSpec>, u64), Response> {
    let projector_path = crate::serve::auto_pipeline::resolve_projector_companion(text_path, None)
        .map_err(|error| {
            ApiError::invalid_request(
                format!("cannot resolve exact text/projector pair: {error}"),
                Some("model".into()),
            )
            .into_response()
        })?;
    let projector = match projector_path {
        Some(path) => {
            let text = mlx_native::gguf::GgufFile::open(text_path).map_err(|error| {
                ApiError::invalid_request(
                    format!("cannot open text artifact for pair identity: {error}"),
                    Some("model".into()),
                )
                .into_response()
            })?;
            let expected = text
                .metadata_string(crate::core::provenance::KEY_MMPROJ_SHA256)
                .map(str::to_owned);
            Some(
                crate::serve::multi_model::ProjectorLoadSpec::preflight(
                    path,
                    expected,
                    crate::inference::vision::pipeline::DEFAULT_VISION_EMBEDDING_CACHE_BYTES as u64,
                )
                .map_err(|error| {
                    ApiError::invalid_request(
                        format!("projector preflight failed: {error}"),
                        Some("model".into()),
                    )
                    .into_response()
                })?,
            )
        }
        None => None,
    };
    let bytes = activation_pair_bytes(text_bytes, projector.as_ref())?;
    Ok((projector, bytes))
}

fn activation_pair_bytes(
    text_bytes: u64,
    projector: Option<&crate::serve::multi_model::ProjectorLoadSpec>,
) -> std::result::Result<u64, Response> {
    let components = match projector {
        Some(projector) => crate::serve::multi_model::ResidentByteComponents::new(
            text_bytes,
            projector.projected_weight_bytes,
            projector.cache_budget_bytes,
        ),
        None => Ok(crate::serve::multi_model::ResidentByteComponents::text_only(text_bytes)),
    }
    .map_err(|error| {
        ApiError::invalid_request(
            format!("model/projector resident-byte accounting failed: {error}"),
            Some("model".into()),
        )
        .into_response()
    })?;
    Ok(components.total_bytes())
}

async fn verify_local_gguf_cancellable(
    artifact: &super::local_artifacts::LocalGgufArtifact,
    cancellation: super::cancellation::CancellationSignal,
    supervisor: super::cancellation::PreparationSupervisor,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> std::result::Result<std::path::PathBuf, Response> {
    let executable =
        std::env::current_exe().map_err(|_| ApiError::internal_error().into_response())?;
    let quant = artifact.quant.ok_or_else(|| {
        ApiError::invalid_request(
            "selected local artifact has no supported quant",
            Some("artifact".into()),
        )
        .into_response()
    })?;
    let mut command = tokio::process::Command::new(executable);
    command
        .arg("__verify-local-gguf")
        .arg("--root")
        .arg(&artifact.root)
        .arg("--artifact")
        .arg(&artifact.path)
        .arg("--bytes")
        .arg(artifact.bytes.to_string())
        .arg("--sha256")
        .arg(&artifact.sha256)
        .arg("--quant")
        .arg(quant.as_str());
    let output = run_preparation_command(
        command,
        cancellation,
        supervisor,
        PreparationLimits::transfer_receipt(),
        Some(permit),
    )
    .await
    .map_err(|error| {
        ApiError::generation_error(format!("local GGUF verification failed: {error}"))
            .into_response()
    })?;
    if !output.status.success() {
        tracing::warn!(
            status = %output.status,
            stderr_bytes = output.stderr.len(),
            "local GGUF verification helper rejected selected artifact"
        );
        return Err(ApiError::invalid_request(
            "selected local GGUF no longer matches its hf2q receipt or cache authority",
            Some("artifact".into()),
        )
        .into_response());
    }
    let receipt: super::local_artifacts::LocalVerificationReceipt =
        serde_json::from_slice(&output.stdout).map_err(|_| {
            ApiError::generation_error("local GGUF verifier returned an invalid receipt")
                .into_response()
        })?;
    if receipt.path != artifact.path {
        return Err(ApiError::generation_error(
            "local GGUF verifier returned a different artifact",
        )
        .into_response());
    }
    Ok(receipt.path)
}

async fn fetch_hub_gguf_cancellable(
    artifact: &crate::input::hf_download::HubGgufArtifact,
    cancellation: super::cancellation::CancellationSignal,
    supervisor: super::cancellation::PreparationSupervisor,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> std::result::Result<std::path::PathBuf, Response> {
    let executable =
        std::env::current_exe().map_err(|_| ApiError::internal_error().into_response())?;
    let quant = artifact.quant_hint.as_deref().ok_or_else(|| {
        ApiError::invalid_request("selected artifact has no quant", Some("artifact".into()))
            .into_response()
    })?;
    let mut command = tokio::process::Command::new(executable);
    command.args([
        "__fetch-hub-gguf",
        "--repository",
        &artifact.repository,
        "--revision",
        &artifact.revision,
        "--artifact",
        &artifact.filename,
        "--bytes",
        &artifact.bytes.to_string(),
        "--sha256",
        &artifact.sha256,
        "--quant",
        quant,
    ]);
    let output = run_preparation_command(
        command,
        cancellation,
        supervisor,
        PreparationLimits::transfer_receipt(),
        Some(permit),
    )
    .await
    .map_err(|error| {
        ApiError::generation_error(format!("hosted GGUF transfer failed: {error}")).into_response()
    })?;
    if !output.status.success() {
        tracing::warn!(
            status = %output.status,
            stderr_bytes = output.stderr.len(),
            "hosted GGUF transfer helper failed"
        );
        return Err(ApiError::invalid_request(
            format!(
                "selected GGUF transfer failed (helper exited {}); verify disk space, network access, repository access, and HF_TOKEN",
                output.status
            ),
            Some("artifact".into()),
        )
        .into_response());
    }
    let path = String::from_utf8(output.stdout).map_err(|_| {
        ApiError::generation_error("hosted GGUF helper returned non-UTF-8 output").into_response()
    })?;
    let path = path.trim();
    if path.is_empty() || path.lines().count() != 1 {
        return Err(ApiError::generation_error(
            "hosted GGUF helper returned an invalid path receipt",
        )
        .into_response());
    }
    Ok(std::path::PathBuf::from(path))
}

async fn resolve_activation_target(
    state: &AppState,
    requested_model: &str,
    candidate_id: Option<&str>,
) -> std::result::Result<ActivationTarget, Response> {
    let model = requested_model.trim();
    if model.is_empty() {
        return Err(
            ApiError::invalid_request("model must not be empty", Some("model".into()))
                .into_response(),
        );
    }
    let hub_repository = normalize_hub_repository(model)?;
    if let Some(candidate_id) = candidate_id {
        let artifact = state.artifact_catalog.resolve(candidate_id).map_err(|error| {
            let status = match error {
                CatalogError::Gone => StatusCode::GONE,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            (
                status,
                Json(serde_json::json!({"error":{"message":error.to_string(),"type":"invalid_request_error","code":"artifact_selection_gone"}})),
            )
                .into_response()
        })?;
        return match artifact {
            StoredArtifact::Hosted(artifact) => {
                if hub_repository.as_deref() != Some(artifact.repository.as_str()) {
                    return Err(ApiError::invalid_request(
                        "artifact selection does not belong to requested model",
                        Some("candidate_id".into()),
                    )
                    .into_response());
                }
                let quant = QuantType::from_canonical_str(
                    artifact.quant_hint.as_deref().unwrap_or_default(),
                )
                .map_err(|error| {
                    ApiError::invalid_request(error, Some("candidate_id".into())).into_response()
                })?;
                Ok(ActivationTarget {
                    repo: artifact.request_model(),
                    quant,
                    bytes: artifact.bytes,
                    request_model: artifact.request_model(),
                    projector: None,
                    payload: ActivationPayload::Hosted(artifact),
                })
            }
            StoredArtifact::Local(artifact) => {
                if hub_repository.as_deref() != Some(artifact.repository.as_str()) {
                    return Err(ApiError::invalid_request(
                        "local artifact selection does not belong to requested model",
                        Some("candidate_id".into()),
                    )
                    .into_response());
                }
                let quant = artifact.quant.ok_or_else(|| {
                    ApiError::invalid_request(
                        "selected local artifact is unavailable",
                        Some("candidate_id".into()),
                    )
                    .into_response()
                })?;
                let request_model = format!(
                    "local://{}@{}/{}",
                    artifact.repository, artifact.revision, candidate_id
                );
                let (projector, bytes) = activation_projector_spec(&artifact.path, artifact.bytes)?;
                Ok(ActivationTarget {
                    repo: request_model.clone(),
                    quant,
                    bytes,
                    request_model,
                    projector,
                    payload: ActivationPayload::VerifiedLocal(artifact),
                })
            }
        };
    }

    // Diagnostic activation is intentionally conversion-free. A bare Hub id
    // must be resolved through the exact artifact catalog first.
    if hub_repository.is_some() {
        return Err(ApiError::invalid_request(
            "bare Hub model activation is disabled in diagnostic chat; select an exact hosted GGUF",
            Some("model".into()),
        )
        .into_response());
    }
    let gguf_path = std::path::PathBuf::from(model);
    if !gguf_path.is_file()
        || !gguf_path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        return Err(ApiError::invalid_request(
            "diagnostic activation requires an existing local GGUF file or a hosted artifact selection",
            Some("model".into()),
        )
        .into_response());
    }
    let header = mlx_native::gguf::GgufFile::open(&gguf_path).map_err(|error| {
        ApiError::invalid_request(
            format!("cannot open local GGUF header: {error}"),
            Some("model".into()),
        )
        .into_response()
    })?;
    let actual_file_type = header.metadata_u32("general.file_type");
    let quant = match actual_file_type.and_then(diagnostic_quant_from_file_type) {
        Some(quant) => quant,
        None => {
            return Err(ApiError::invalid_request(
                format!(
                    "local GGUF file type {actual_file_type:?} is not supported by diagnostic activation"
                ),
                Some("model".into()),
            )
            .into_response())
        }
    };
    let bytes = std::fs::metadata(&gguf_path)
        .map_err(|error| {
            tracing::error!(
                path = %gguf_path.display(),
                error = %error,
                "cannot inspect local diagnostic GGUF size"
            );
            ApiError::generation_error(
                "cannot inspect local GGUF file metadata; inspect server diagnostics",
            )
            .into_response()
        })?
        .len();
    let (projector, bytes) = activation_projector_spec(&gguf_path, bytes)?;
    let repo = crate::serve::pool_key_for_path(&gguf_path);
    Ok(ActivationTarget {
        request_model: repo.clone(),
        repo,
        quant,
        bytes,
        projector,
        payload: ActivationPayload::ExplicitLocal(gguf_path),
    })
}

fn activation_conflict_response(
    requested_model: &str,
    repo: &str,
    quant: QuantType,
    bytes: u64,
    revision: u64,
    victims: &[LoadedSummary],
    projected_bytes: u64,
) -> Response {
    (
        StatusCode::CONFLICT,
        Json(serde_json::json!({
            "schema_version": HF2Q_ACTIVATION_SCHEMA,
            "status": "conflict",
            "candidate": activation_candidate_json(requested_model, repo, quant, bytes),
            "pool_revision": revision,
            "victims": victims.iter().map(loaded_summary_json).collect::<Vec<_>>(),
            "projected_bytes": projected_bytes,
            "requires_explicit_switch": true,
        })),
    )
        .into_response()
}

fn lifecycle_error_response(error: LifecycleError) -> Response {
    let (status, code, restart_required) = match error {
        LifecycleError::StalePlan { .. } | LifecycleError::VictimPlanChanged => {
            (StatusCode::CONFLICT, "stale_activation_plan", false)
        }
        LifecycleError::DrainTimeout { .. }
        | LifecycleError::PrepareFailed(_)
        | LifecycleError::ShutdownFailed(_)
        | LifecycleError::LoadFailed(_)
        | LifecycleError::PostCommitFailed(_) => {
            (StatusCode::SERVICE_UNAVAILABLE, "restart_required", true)
        }
        _ => (StatusCode::SERVICE_UNAVAILABLE, "activation_failed", false),
    };
    (
        status,
        Json(serde_json::json!({
            "schema_version": HF2Q_ACTIVATION_SCHEMA,
            "status": "error",
            "code": code,
            "message": error.to_string(),
            "restart_required": restart_required,
        })),
    )
        .into_response()
}

/// Non-evicting diagnostic activation, plus an explicit revision-bound switch
/// action. Ordinary OpenAI requests retain ADR-005 auto-swap semantics.
pub async fn activate_model(
    State(state): State<AppState>,
    Extension(cancellation): Extension<RequestCancellation>,
    Json(request): Json<ModelActivationRequest>,
) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let normalized_hub = match normalize_hub_repository(&request.model) {
        Ok(repository) => repository,
        Err(response) => return response,
    };
    let resident_model = normalized_hub.as_deref().unwrap_or(request.model.trim());

    // A live model id, repo, or pool key is resident without touching disk.
    if request.candidate_id.is_none() {
        if let Ok(manager) = state.pool.read() {
            let hosted_prefix = format!("hf://{}@", resident_model);
            let local_prefix = format!("local://{}@", resident_model);
            let matches = manager
                .snapshot_engines()
                .into_iter()
                .filter(|engine| {
                    resident_model == engine.engine.model_id()
                        || resident_model == engine.repo
                        || resident_model == format!("{}@{}", engine.repo, engine.quant.as_str())
                        || engine.repo.starts_with(&hosted_prefix)
                        || engine.repo.starts_with(&local_prefix)
                })
                .collect::<Vec<_>>();
            if matches.len() > 1 {
                if request.action == ModelActivationAction::Probe {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(serde_json::json!({
                            "schema_version": HF2Q_ACTIVATION_SCHEMA,
                            "status": "ambiguous_resident",
                            "message": "multiple resident artifacts match; select an exact hosted artifact",
                        })),
                    )
                        .into_response();
                }
                return ApiError::invalid_request(
                    "multiple resident artifacts match this repository; use --quant or --artifact",
                    Some("model".into()),
                )
                .into_response();
            }
            if let Some(engine) = matches.into_iter().next() {
                let stats = manager.pool_stats();
                return (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "schema_version": HF2Q_ACTIVATION_SCHEMA,
                        "status": "resident",
                        "pool_revision": stats.revision,
                        "request_model": engine.repo,
                        "candidate": activation_candidate_json(
                            &request.model,
                            &engine.repo,
                            engine.quant,
                            engine.bytes_resident,
                        ),
                    })),
                )
                    .into_response();
            }
        }
    }

    if request.action == ModelActivationAction::Probe {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "schema_version": HF2Q_ACTIVATION_SCHEMA,
                "status": "not_resident",
            })),
        )
            .into_response();
    }

    let target =
        match resolve_activation_target(&state, &request.model, request.candidate_id.as_deref())
            .await
        {
            Ok(target) => target,
            Err(response) => return response,
        };
    let repo = target.repo.clone();
    let quant = target.quant;
    let bytes = target.bytes;
    if let Some(local_path) = target.verified_local_path() {
        if let Ok(manager) = state.pool.read() {
            if let Some(engine) = manager
                .snapshot_engines()
                .into_iter()
                .find(|engine| engine.quant == quant && engine.gguf_path == local_path)
            {
                let revision = manager.pool_stats().revision;
                return (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "schema_version": HF2Q_ACTIVATION_SCHEMA,
                        "status": "resident",
                        "pool_revision": revision,
                        "request_model": engine.repo,
                    })),
                )
                    .into_response();
            }
        }
    }

    if let Ok(manager) = state.pool.read() {
        if let Some(engine) = manager
            .snapshot_engines()
            .into_iter()
            .find(|engine| engine.repo == repo && engine.quant == quant)
        {
            let revision = manager.pool_stats().revision;
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "schema_version": HF2Q_ACTIVATION_SCHEMA,
                    "status": "resident",
                    "pool_revision": revision,
                    "request_model": engine.repo,
                    "candidate": activation_candidate_json(
                        &request.model, &repo, quant, engine.bytes_resident,
                    ),
                })),
            )
                .into_response();
        }
    }

    match request.action {
        ModelActivationAction::Load => {
            // Reject known conflict/impossible cases before transferring any
            // bytes, then release the lifecycle gate so resident generation
            // can continue while the exact artifact is prepared.
            {
                let _admission_guard = state.model_lifecycle.write_admission().await;
                let manager = match state.pool.read() {
                    Ok(manager) => manager,
                    Err(_) => return ApiError::internal_error().into_response(),
                };
                let plan = manager.admission_plan(&repo, quant, bytes);
                match &plan.outcome {
                    AdmissionOutcome::WouldEvict {
                        victims,
                        projected_bytes,
                    } => {
                        let victim_keys = victims
                            .iter()
                            .map(|victim| victim.repo_id.as_str())
                            .collect::<Vec<_>>();
                        let summaries = manager
                            .iter_loaded()
                            .filter(|entry| victim_keys.contains(&entry.pool_key.as_str()))
                            .collect::<Vec<_>>();
                        return activation_conflict_response(
                            &request.model,
                            &repo,
                            quant,
                            bytes,
                            plan.revision,
                            &summaries,
                            *projected_bytes,
                        );
                    }
                    AdmissionOutcome::Impossible { reason } => {
                        return (
                            StatusCode::UNPROCESSABLE_ENTITY,
                            Json(serde_json::json!({
                                "schema_version": HF2Q_ACTIVATION_SCHEMA,
                                "status": "impossible",
                                "candidate": activation_candidate_json(
                                    &request.model, &repo, quant, bytes
                                ),
                                "pool_revision": plan.revision,
                                "message": reason.to_string(),
                            })),
                        )
                            .into_response();
                    }
                    _ => plan,
                }
            };
            let (repo, quant, gguf_path, bytes, request_model, projector) = match target
                .materialize(
                    cancellation.0.clone(),
                    state.preparations.clone(),
                    &state.artifact_catalog,
                )
                .await
            {
                Ok(materialized) => materialized,
                Err(response) => return response,
            };
            let mut engine_config = match activation_engine_config(&state, &gguf_path) {
                Ok(config) => config,
                Err(response) => return response,
            };
            engine_config.projector = projector;
            // The pool may have changed during preparation. Re-plan under the
            // exclusive gate and publish only if admission is still
            // non-evicting.
            let admission_guard = state.model_lifecycle.write_admission().await;
            let plan = {
                let manager = match state.pool.read() {
                    Ok(manager) => manager,
                    Err(_) => return ApiError::internal_error().into_response(),
                };
                let plan = manager.admission_plan(&repo, quant, bytes);
                match &plan.outcome {
                    AdmissionOutcome::WouldEvict {
                        victims,
                        projected_bytes,
                    } => {
                        let victim_keys = victims
                            .iter()
                            .map(|victim| victim.repo_id.as_str())
                            .collect::<Vec<_>>();
                        let summaries = manager
                            .iter_loaded()
                            .filter(|entry| victim_keys.contains(&entry.pool_key.as_str()))
                            .collect::<Vec<_>>();
                        return activation_conflict_response(
                            &request.model,
                            &repo,
                            quant,
                            bytes,
                            plan.revision,
                            &summaries,
                            *projected_bytes,
                        );
                    }
                    AdmissionOutcome::Impossible { reason } => {
                        return (
                            StatusCode::UNPROCESSABLE_ENTITY,
                            Json(serde_json::json!({
                                "schema_version": HF2Q_ACTIVATION_SCHEMA,
                                "status": "impossible",
                                "candidate": activation_candidate_json(
                                    &request.model, &repo, quant, bytes
                                ),
                                "pool_revision": plan.revision,
                                "message": reason.to_string(),
                            })),
                        )
                            .into_response();
                    }
                    _ => plan,
                }
            };
            let pool = Arc::clone(&state.pool);
            let repo_for_load = repo.clone();
            let load = run_consistent_result_task(
                state.preparations.clone(),
                async move {
                    let _admission_guard = admission_guard;
                    tokio::task::spawn_blocking(move || {
                        let mut manager = pool.write().map_err(|error| {
                            HotSwapError::LoaderFailed(anyhow::anyhow!(
                                "pool rwlock poisoned: {error}"
                            ))
                        })?;
                        manager.load_or_get_non_evicting(
                            &repo_for_load,
                            quant,
                            &gguf_path,
                            &engine_config,
                        )
                    })
                    .await
                    .map_err(|error| {
                        HotSwapError::LoaderFailed(anyhow::anyhow!(
                            "model load task failed: {error}"
                        ))
                    })?
                },
                |_| false,
            )
            .await;
            match load {
                Ok(Ok(NonEvictingLoad::Resident(_))) => (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "schema_version": HF2Q_ACTIVATION_SCHEMA,
                        "status": "resident",
                        "pool_revision": plan.revision,
                        "request_model": request_model,
                    })),
                )
                    .into_response(),
                Ok(Ok(NonEvictingLoad::Loaded(_))) => {
                    let revision = state
                        .pool
                        .read()
                        .map(|manager| manager.pool_stats().revision)
                        .unwrap_or(plan.revision);
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "schema_version": HF2Q_ACTIVATION_SCHEMA,
                            "status": "loaded",
                            "pool_revision": revision,
                            "request_model": request_model,
                            "candidate": activation_candidate_json(
                                &request.model, &repo, quant, bytes
                            ),
                        })),
                    )
                        .into_response()
                }
                Ok(Ok(NonEvictingLoad::Conflict(_))) => {
                    lifecycle_error_response(LifecycleError::VictimPlanChanged)
                }
                Ok(Err(error)) => map_hotswap_error_to_response(error),
                Err(error) => ApiError::generation_error(format!(
                    "model activation transaction failed: {error}"
                ))
                .into_response(),
            }
        }
        ModelActivationAction::Switch => {
            let Some(expected_revision) = request.expected_revision else {
                return ApiError::invalid_request(
                    "expected_revision is required for action=switch",
                    Some("expected_revision".into()),
                )
                .into_response();
            };
            if request.victims.is_empty() {
                return ApiError::invalid_request(
                    "victims is required for action=switch",
                    Some("victims".into()),
                )
                .into_response();
            }
            let confirmation = SwitchConfirmation {
                candidate_repo: repo.clone(),
                candidate_quant: quant,
                candidate_bytes: bytes,
                expected_revision,
                victims: request.victims.into_iter().map(Into::into).collect(),
            };
            // Validate the receipt before a potentially large transfer. The
            // switch revalidates after transfer before draining anything.
            {
                let _admission_guard = state.model_lifecycle.write_admission().await;
                if let Err(error) = state
                    .model_lifecycle
                    .validate_switch_confirmation(&state.pool, &confirmation)
                {
                    return lifecycle_error_response(error);
                }
            }
            let (repo, quant, gguf_path, bytes, request_model, projector) = match target
                .materialize(
                    cancellation.0,
                    state.preparations.clone(),
                    &state.artifact_catalog,
                )
                .await
            {
                Ok(materialized) => materialized,
                Err(response) => return response,
            };
            let mut engine_config = match activation_engine_config(&state, &gguf_path) {
                Ok(config) => config,
                Err(response) => return response,
            };
            engine_config.projector = projector;
            let confirmation = SwitchConfirmation {
                candidate_repo: repo.clone(),
                candidate_quant: quant,
                candidate_bytes: bytes,
                expected_revision,
                victims: confirmation.victims,
            };
            let target_repo = repo.clone();
            let lifecycle = Arc::clone(&state.model_lifecycle);
            let switch_pool = Arc::clone(&state.pool);
            let result = run_consistent_result_task(
                state.preparations.clone(),
                async move {
                    lifecycle
                        .switch(
                            switch_pool,
                            confirmation,
                            std::time::Duration::from_secs(60),
                            |loaded| async move { loaded.engine.shutdown().await },
                            move |pool| async move {
                                tokio::task::spawn_blocking(move || {
                                    let mut manager = pool.write().map_err(|error| {
                                        HotSwapError::LoaderFailed(anyhow::anyhow!(
                                            "pool rwlock poisoned: {error}"
                                        ))
                                    })?;
                                    manager.load_or_get_non_evicting(
                                        &target_repo,
                                        quant,
                                        &gguf_path,
                                        &engine_config,
                                    )
                                })
                                .await
                                .map_err(|error| {
                                    HotSwapError::LoaderFailed(anyhow::anyhow!(
                                        "replacement model load task failed: {error}"
                                    ))
                                })?
                            },
                        )
                        .await
                },
                LifecycleError::requires_restart,
            )
            .await;
            match result {
                Ok(Ok(NonEvictingLoad::Loaded(_))) | Ok(Ok(NonEvictingLoad::Resident(_))) => {
                    let revision = state
                        .pool
                        .read()
                        .map(|manager| manager.pool_stats().revision)
                        .unwrap_or(expected_revision);
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "schema_version": HF2Q_ACTIVATION_SCHEMA,
                            "status": "switched",
                            "pool_revision": revision,
                            "request_model": request_model,
                            "candidate": activation_candidate_json(
                                &request.model, &repo, quant, bytes
                            ),
                        })),
                    )
                        .into_response()
                }
                Ok(Ok(NonEvictingLoad::Conflict(_))) => {
                    lifecycle_error_response(LifecycleError::VictimPlanChanged)
                }
                Ok(Err(error)) => lifecycle_error_response(error),
                Err(error) => {
                    ApiError::generation_error(format!("model switch transaction failed: {error}"))
                        .into_response()
                }
            }
        }
        ModelActivationAction::Probe => unreachable!("probe returns before target resolution"),
    }
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use super::*;

    #[test]
    fn canonical_hugging_face_base_url_normalizes_to_repository_id() {
        assert_eq!(
            normalize_hub_repository("https://huggingface.co/owner/model").unwrap(),
            Some("owner/model".to_owned())
        );
        assert!(normalize_hub_repository("https://huggingface.co/owner/model/tree/main").is_err());
    }

    #[test]
    fn diagnostic_hosted_quant_to_header_mapping_is_exact() {
        for (quant, file_type) in [
            (QuantType::Q3_K_M, 12),
            (QuantType::Q4_K_M, 15),
            (QuantType::Q5_K_M, 17),
            (QuantType::Q6_K, 18),
            (QuantType::Q8_0, 7),
        ] {
            assert_eq!(diagnostic_gguf_file_type(quant), file_type);
            assert_eq!(diagnostic_quant_from_file_type(file_type), Some(quant));
        }
        assert_eq!(diagnostic_quant_from_file_type(0), None);
    }

    #[test]
    fn projector_config_receipt_is_exact_and_path_free() {
        let identity = crate::serve::multi_model::EngineConfigIdentity {
            projector: Some(crate::serve::multi_model::ProjectorConfigIdentity {
                artifact_sha256: "a".repeat(64),
                source_sha256: Some("b".repeat(64)),
                pair_generation: Some("pair-generation-7".into()),
                profile: "qwen3vl_siglip".into(),
                weight_bytes: 123,
                cache_budget_bytes: 456,
            }),
            ..crate::serve::multi_model::EngineConfigIdentity::default()
        };
        let receipt = engine_config_identity_json(&identity);
        assert_eq!(receipt["projector"]["artifact_sha256"], "a".repeat(64));
        assert_eq!(receipt["projector"]["source_sha256"], "b".repeat(64));
        assert_eq!(receipt["projector"]["pair_generation"], "pair-generation-7");
        assert_eq!(receipt["projector"]["weight_bytes"], 123);
        assert_eq!(receipt["projector"]["cache_budget_bytes"], 456);
        let encoded = serde_json::to_string(&receipt).unwrap();
        assert!(!encoded.contains("/opt/"));
        assert!(!encoded.contains(".gguf"));
    }

    #[tokio::test]
    async fn post_commit_load_failure_requires_restart() {
        let response = lifecycle_error_response(LifecycleError::LoadFailed(
            crate::serve::load_diagnostic::PublicLoadDiagnostic::LoaderRejected,
        ));
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "restart_required");
        assert_eq!(json["restart_required"], true);
    }

    #[tokio::test]
    async fn local_candidate_is_repository_bound_and_keeps_path_server_private() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("model.gguf");
        let mut gguf = Vec::new();
        gguf.extend_from_slice(b"GGUF");
        gguf.extend_from_slice(&3u32.to_le_bytes());
        gguf.extend_from_slice(&0u64.to_le_bytes());
        gguf.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.file_type";
        gguf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        gguf.extend_from_slice(key);
        gguf.extend_from_slice(&4u32.to_le_bytes());
        gguf.extend_from_slice(&QuantType::Q4_K_M.gguf_file_type().to_le_bytes());
        std::fs::write(&path, &gguf).unwrap();
        let state = AppState::new(super::super::state::ServerConfig::default());
        let view = state
            .artifact_catalog
            .register_local(super::super::local_artifacts::LocalArtifactCatalog {
                artifacts: vec![super::super::local_artifacts::LocalGgufArtifact {
                    repository: "owner/model".into(),
                    revision: "a".repeat(40),
                    filename: "model.gguf".into(),
                    root: root.path().to_path_buf(),
                    path: path.clone(),
                    bytes: gguf.len() as u64,
                    sha256: "b".repeat(64),
                    quant_hint: "Q4_K_M".into(),
                    quant: Some(QuantType::Q4_K_M),
                    selectable: true,
                    unavailable_reason: None,
                    provenance:
                        super::super::local_artifacts::LocalArtifactProvenance::ConversionReceipt,
                }],
                warnings: Vec::new(),
            })
            .unwrap();
        let candidate_id = view.candidates[0].candidate_id.as_deref().unwrap();
        let target = resolve_activation_target(&state, "owner/model", Some(candidate_id))
            .await
            .unwrap();
        assert_eq!(target.verified_local_path(), Some(path.as_path()));
        assert!(!target.request_model.contains(&"b".repeat(64)));
        assert!(target.request_model.contains(candidate_id));

        let response =
            match resolve_activation_target(&state, "other/model", Some(candidate_id)).await {
                Ok(_) => panic!("candidate must be bound to its repository"),
                Err(response) => response,
            };
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
