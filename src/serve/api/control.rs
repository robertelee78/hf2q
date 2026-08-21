//! ADR-047 authenticated, versioned diagnostic lifecycle control plane.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::handlers::map_hotswap_error_to_response;
use super::lifecycle::{LifecycleError, SwitchConfirmation};
use super::schema::ApiError;
use super::state::AppState;
use super::{DIAGNOSTIC_NO_EVICT_HEADER, DIAGNOSTIC_NO_EVICT_VALUE};
use crate::serve::auto_pipeline;
use crate::serve::multi_model::{
    AdmissionOutcome, EngineConfig, HotSwapError, LoadedSummary, NonEvictingLoad,
};
use crate::serve::quant_select::QuantType;

const HF2Q_RUNTIME_SCHEMA: &str = "hf2q.runtime.v1";
const HF2Q_ACTIVATION_SCHEMA: &str = "hf2q.model-activation.v1";

#[derive(Debug, serde::Deserialize)]
pub struct ModelActivationRequest {
    pub model: String,
    #[serde(default)]
    pub action: ModelActivationAction,
    pub expected_revision: Option<u64>,
    #[serde(default)]
    pub victims: Vec<ModelActivationVictim>,
}

#[derive(Debug, Default, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelActivationAction {
    #[default]
    Load,
    Switch,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelActivationVictim {
    pub pool_key: String,
    pub quant: String,
    pub bytes_resident: u64,
    pub generation: u64,
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
        .iter_loaded()
        .map(|entry| loaded_summary_json(&entry))
        .collect::<Vec<_>>();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "schema_version": HF2Q_RUNTIME_SCHEMA,
            "backend": "mlx-native",
            "capabilities": {
                "model_activation": HF2Q_ACTIVATION_SCHEMA,
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

fn activation_engine_config(state: &AppState) -> EngineConfig {
    EngineConfig {
        tokenizer_path: None,
        config_path: None,
        queue_capacity: state.engine_queue_capacity,
        warmup_synchronously: true,
        kv_metrics_sink: Some(Arc::clone(&state.kv_spill_counters)
            as Arc<dyn crate::serve::kv_persist::metrics::KvCacheMetricsSink>),
        dwq_overlay_path: None,
        engine_mode: crate::serve::api::engine::EngineMode::SerialFifo,
        kv_cache_budget_bytes: None,
    }
}

async fn resolve_activation_target(
    state: &AppState,
    requested_model: &str,
) -> std::result::Result<(String, QuantType, std::path::PathBuf, u64), Response> {
    let model = requested_model.trim();
    if model.is_empty() {
        return Err(
            ApiError::invalid_request("model must not be empty", Some("model".into()))
                .into_response(),
        );
    }
    let cache = Arc::clone(&state.cache);
    let hardware = Arc::clone(&state.hardware);
    let model_arg = model.to_string();
    let no_integrity = state.no_integrity;
    let resolved = tokio::task::spawn_blocking(move || {
        let mut cache = cache
            .lock()
            .map_err(|error| anyhow::anyhow!("cache mutex poisoned: {error}"))?;
        auto_pipeline::resolve_or_prepare_model(
            &model_arg,
            &mut cache,
            hardware.as_ref(),
            no_integrity,
        )
    })
    .await
    .map_err(|_| ApiError::internal_error().into_response())?
    .map_err(|error| {
        ApiError::invalid_request(
            format!("cannot resolve activation model: {error:#}"),
            Some("model".into()),
        )
        .into_response()
    })?;
    let repo = resolved
        .repo_id
        .unwrap_or_else(|| crate::serve::pool_key_for_path(&resolved.gguf_path));
    let quant = resolved.quant.unwrap_or(QuantType::Q4_K_M);
    let bytes = std::fs::metadata(&resolved.gguf_path)
        .map_err(|error| {
            ApiError::generation_error(format!(
                "cannot read GGUF size at {}: {error}",
                resolved.gguf_path.display()
            ))
            .into_response()
        })?
        .len();
    Ok((repo, quant, resolved.gguf_path, bytes))
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
        | LifecycleError::LoadFailed(_) => {
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
    Json(request): Json<ModelActivationRequest>,
) -> Response {
    state
        .metrics
        .requests_total
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // A live model id, repo, or pool key is resident without touching disk.
    if let Ok(manager) = state.pool.read() {
        if let Some(engine) = manager.snapshot_engines().into_iter().find(|engine| {
            request.model == engine.engine.model_id()
                || request.model == engine.repo
                || request.model == format!("{}@{}", engine.repo, engine.quant.as_str())
        }) {
            let stats = manager.pool_stats();
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "schema_version": HF2Q_ACTIVATION_SCHEMA,
                    "status": "resident",
                    "pool_revision": stats.revision,
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

    let (repo, quant, gguf_path, bytes) =
        match resolve_activation_target(&state, &request.model).await {
            Ok(target) => target,
            Err(response) => return response,
        };
    let engine_config = activation_engine_config(&state);

    match request.action {
        ModelActivationAction::Load => {
            let _admission_guard = state.model_lifecycle.write_admission().await;
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
            let load = tokio::task::spawn_blocking(move || {
                let mut manager = pool.write().map_err(|error| {
                    HotSwapError::LoaderFailed(anyhow::anyhow!("pool rwlock poisoned: {error}"))
                })?;
                manager.load_or_get_non_evicting(&repo_for_load, quant, &gguf_path, &engine_config)
            })
            .await;
            match load {
                Ok(Ok(NonEvictingLoad::Resident(_))) => (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "schema_version": HF2Q_ACTIVATION_SCHEMA,
                        "status": "resident",
                        "pool_revision": plan.revision,
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
                Err(_) => ApiError::internal_error().into_response(),
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
            let target_repo = repo.clone();
            let result = state
                .model_lifecycle
                .switch(
                    Arc::clone(&state.pool),
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
                .await;
            match result {
                Ok(NonEvictingLoad::Loaded(_)) | Ok(NonEvictingLoad::Resident(_)) => {
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
                            "candidate": activation_candidate_json(
                                &request.model, &repo, quant, bytes
                            ),
                        })),
                    )
                        .into_response()
                }
                Ok(NonEvictingLoad::Conflict(_)) => {
                    lifecycle_error_response(LifecycleError::VictimPlanChanged)
                }
                Err(error) => lifecycle_error_response(error),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use super::*;

    #[tokio::test]
    async fn post_commit_load_failure_requires_restart() {
        let response = lifecycle_error_response(LifecycleError::LoadFailed("boom".to_string()));
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["code"], "restart_required");
        assert_eq!(json["restart_required"], true);
    }
}
