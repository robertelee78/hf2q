//! Explicit, generation-bound activation for the dedicated embedding slot.

mod artifact;

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::cancellation::run_consistent_result_task;
use super::control::{ModelActivationAction, ModelActivationRequest, HF2Q_ACTIVATION_SCHEMA};
use super::middleware::RequestCancellation;
use super::schema::ApiError;
use super::state::{AppState, EmbeddingModelSnapshot, EmbeddingSwapError};

fn candidate_json(
    model: &str,
    repo: &str,
    quant_hint: &str,
    bytes_resident: u64,
    exact_selection: &str,
) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "artifact_key": format!("{repo}@{quant_hint}"),
        "repo": repo,
        "quant": quant_hint,
        "bytes": bytes_resident,
        "exact_selection": exact_selection,
    })
}

pub(super) fn snapshot_json(snapshot: &EmbeddingModelSnapshot) -> serde_json::Value {
    serde_json::json!({
        "generation": snapshot.generation,
        "configured": snapshot.configured,
        "loading": snapshot.loading,
        "model_id": snapshot.model_id,
        "arch": snapshot.arch,
        "resident_bytes": snapshot.resident_bytes,
        "last_load_failed": snapshot.last_load_error.is_some(),
    })
}

fn same_startup_artifact(left: &std::path::Path, right: &std::path::Path) -> bool {
    match (std::fs::canonicalize(left), std::fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedDecision {
    Resident,
    Conflict,
    Busy,
    Stale { expected: u64, current: u64 },
    ReplaceAt(u64),
}

fn decide_after_preparation(
    current: &EmbeddingModelSnapshot,
    action: ModelActivationAction,
    requested_switch_generation: Option<u64>,
    request_model: &str,
) -> PreparedDecision {
    if current.loading {
        return PreparedDecision::Busy;
    }
    if current.model_id.as_deref() == Some(request_model) {
        return PreparedDecision::Resident;
    }
    match action {
        ModelActivationAction::Load if current.model_id.is_some() => PreparedDecision::Conflict,
        ModelActivationAction::Load => PreparedDecision::ReplaceAt(current.generation.unwrap_or(0)),
        ModelActivationAction::Switch => {
            let expected = requested_switch_generation
                .expect("switch generation was validated before preparation");
            let actual = current.generation.unwrap_or(0);
            if actual == expected {
                PreparedDecision::ReplaceAt(expected)
            } else {
                PreparedDecision::Stale {
                    expected,
                    current: actual,
                }
            }
        }
        ModelActivationAction::Probe => unreachable!("probe returned before preparation"),
    }
}

fn conflict_response(
    request: &ModelActivationRequest,
    repo: &str,
    quant_hint: &str,
    bytes: u64,
    exact_selection: &str,
    snapshot: &EmbeddingModelSnapshot,
) -> Response {
    (
        StatusCode::CONFLICT,
        Json(serde_json::json!({
            "schema_version": HF2Q_ACTIVATION_SCHEMA,
            "kind": "embedding",
            "status": "conflict",
            "code": "embedding_model_resident",
            "candidate": candidate_json(
                &request.model,
                repo,
                quant_hint,
                bytes,
                exact_selection,
            ),
            "embedding": snapshot_json(snapshot),
            "requires_explicit_switch": true,
        })),
    )
        .into_response()
}

fn swap_error_response(state: &AppState, error: EmbeddingSwapError) -> Response {
    let snapshot = state.embedding_model_snapshot();
    match error {
        EmbeddingSwapError::StaleGeneration { .. } => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "schema_version": HF2Q_ACTIVATION_SCHEMA,
                "kind": "embedding",
                "status": "error",
                "code": "stale_embedding_generation",
                "message": error.to_string(),
                "embedding": snapshot_json(&snapshot),
            })),
        )
            .into_response(),
        EmbeddingSwapError::ActiveLeases { .. } => (
            StatusCode::CONFLICT,
            [(header::RETRY_AFTER, "1")],
            Json(serde_json::json!({
                "schema_version": HF2Q_ACTIVATION_SCHEMA,
                "kind": "embedding",
                "status": "error",
                "code": "embedding_requests_active",
                "message": error.to_string(),
                "embedding": snapshot_json(&snapshot),
            })),
        )
            .into_response(),
        EmbeddingSwapError::Load(error) => {
            tracing::error!(%error, "embedding activation failed");
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "schema_version": HF2Q_ACTIVATION_SCHEMA,
                    "kind": "embedding",
                    "status": "error",
                    "code": "embedding_activation_failed",
                    "message": "embedding replacement failed; inspect server diagnostics",
                    "restart_required": false,
                    "embedding": snapshot_json(&snapshot),
                })),
            )
                .into_response()
        }
    }
}

pub(super) async fn activate(
    state: AppState,
    cancellation: RequestCancellation,
    request: ModelActivationRequest,
) -> Response {
    if !request.victims.is_empty() {
        return ApiError::invalid_request(
            "victims applies only to kind=generation; embedding replacement is confirmed by its exact generation",
            Some("victims".into()),
        )
        .into_response();
    }
    if request.action != ModelActivationAction::Probe && request.candidate_id.is_none() {
        return ApiError::invalid_request(
            "kind=embedding load/switch requires an exact catalog candidate_id; bare local paths are startup-only",
            Some("candidate_id".into()),
        )
        .into_response();
    }

    let before = state.embedding_model_snapshot();
    if request.action == ModelActivationAction::Probe {
        let requested = request.model.trim();
        let target = if let Some(candidate_id) = request.candidate_id.as_deref() {
            let target = match artifact::resolve(&state, &request.model, candidate_id).await {
                Ok(target) => target,
                Err(response) => return response,
            };
            Some(target)
        } else {
            None
        };
        // Resolution can await catalog work.  Observe the slot afterward so
        // probe never reports an evicted generation from its entry snapshot.
        let current = state.embedding_model_snapshot();
        let resident = if let Some(target) = target {
            current.model_id.as_deref() == Some(target.request_model.as_str())
        } else {
            current
                .gguf_path
                .as_deref()
                .is_some_and(|path| same_startup_artifact(path, std::path::Path::new(requested)))
        };
        return (
            if resident {
                StatusCode::OK
            } else {
                StatusCode::NOT_FOUND
            },
            Json(serde_json::json!({
                "schema_version": HF2Q_ACTIVATION_SCHEMA,
                "kind": "embedding",
                "status": if resident { "resident" } else { "not_resident" },
                "embedding": snapshot_json(&current),
            })),
        )
            .into_response();
    }
    if before.loading {
        return (
            StatusCode::CONFLICT,
            [(header::RETRY_AFTER, "1")],
            Json(serde_json::json!({
                "schema_version": HF2Q_ACTIVATION_SCHEMA,
                "kind": "embedding",
                "status": "error",
                "code": "embedding_activation_in_progress",
                "embedding": snapshot_json(&before),
            })),
        )
            .into_response();
    }

    let target = match artifact::resolve(
        &state,
        &request.model,
        request
            .candidate_id
            .as_deref()
            .expect("embedding mutation required candidate_id above"),
    )
    .await
    {
        Ok(target) => target,
        Err(response) => return response,
    };

    let known_same_embedding = before
        .model_id
        .as_deref()
        .is_some_and(|active| active == target.request_model.as_str());

    let requested_switch_generation = match request.action {
        ModelActivationAction::Load => {
            if before.model_id.is_some() && !known_same_embedding {
                return conflict_response(
                    &request,
                    &target.repo,
                    &target.quant_hint,
                    target.bytes,
                    &target.request_model,
                    &before,
                );
            }
            None
        }
        ModelActivationAction::Switch => {
            let Some(expected) = request.expected_revision else {
                return ApiError::invalid_request(
                    "expected_revision is required for kind=embedding action=switch",
                    Some("expected_revision".into()),
                )
                .into_response();
            };
            if before.generation != Some(expected) {
                return swap_error_response(
                    &state,
                    EmbeddingSwapError::StaleGeneration {
                        expected,
                        current: before.generation.unwrap_or(0),
                    },
                );
            }
            Some(expected)
        }
        ModelActivationAction::Probe => unreachable!("probe returned before resolution"),
    };

    let (repo, quant_hint, gguf_path, bytes, request_model) = match target
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

    // Artifact preparation is intentionally outside the write transaction.
    // Serialize all public activation decisions after preparation, then
    // re-read the generation under that admission.  This prevents a stale
    // pre-preparation snapshot from becoming a false resident response or
    // overwriting a newly published encoder.
    let admission_guard = state.model_lifecycle.write_admission().await;
    let current = state.embedding_model_snapshot();
    let expected_generation = match decide_after_preparation(
        &current,
        request.action,
        requested_switch_generation,
        &request_model,
    ) {
        PreparedDecision::Busy => {
            return (
                StatusCode::CONFLICT,
                [(header::RETRY_AFTER, "1")],
                Json(serde_json::json!({
                    "schema_version": HF2Q_ACTIVATION_SCHEMA,
                    "kind": "embedding",
                    "status": "error",
                    "code": "embedding_activation_in_progress",
                    "embedding": snapshot_json(&current),
                })),
            )
                .into_response()
        }
        PreparedDecision::Resident => {
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "schema_version": HF2Q_ACTIVATION_SCHEMA,
                    "kind": "embedding",
                    "status": "resident",
                    "candidate": candidate_json(
                        &request.model,
                        &repo,
                        &quant_hint,
                        bytes,
                        &request_model,
                    ),
                    "request_model": &request_model,
                    "embedding": snapshot_json(&current),
                })),
            )
                .into_response()
        }
        PreparedDecision::Conflict => {
            return conflict_response(
                &request,
                &repo,
                &quant_hint,
                bytes,
                &request_model,
                &current,
            )
        }
        PreparedDecision::Stale { expected, current } => {
            return swap_error_response(
                &state,
                EmbeddingSwapError::StaleGeneration { expected, current },
            )
        }
        PreparedDecision::ReplaceAt(generation) => generation,
    };

    let action = request.action;
    let state_for_load = state.clone();
    let identity_for_load = request_model.clone();
    let result = run_consistent_result_task(
        state.preparations.clone(),
        async move {
            let _admission_guard = admission_guard;
            tokio::task::spawn_blocking(move || {
                state_for_load.try_swap_embedding_model_at_generation(expected_generation, || {
                    let mut model = crate::serve::load_embedding_model_from_path(&gguf_path)?;
                    model.model_id = identity_for_load;
                    Ok(model)
                })
            })
            .await
            .map_err(|error| {
                EmbeddingSwapError::Load(anyhow::anyhow!(
                    "embedding activation task failed: {error}"
                ))
            })?
        },
        |_| false,
    )
    .await;

    let receipt = match result {
        Ok(Ok(receipt)) => receipt,
        Ok(Err(error)) => return swap_error_response(&state, error),
        Err(error) => {
            return ApiError::generation_error(format!(
                "embedding activation transaction failed: {error}"
            ))
            .into_response()
        }
    };
    // Use the transaction-owned receipt rather than re-reading the mutable
    // slot: a later activation may publish after this one commits but before
    // its HTTP body is serialized.
    let load_timing = receipt.load_timing;
    let published_embedding = serde_json::json!({
        "generation": receipt.generation,
        "configured": true,
        "loading": false,
        "model_id": &receipt.new_model_id,
        "arch": receipt.new_arch,
        "resident_bytes": receipt.resident_bytes,
        "last_load_failed": false,
    });
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "schema_version": HF2Q_ACTIVATION_SCHEMA,
            "kind": "embedding",
            "status": if action == ModelActivationAction::Switch { "switched" } else { "loaded" },
            "request_model": &request_model,
            "candidate": candidate_json(
                &request.model,
                &repo,
                &quant_hint,
                bytes,
                &request_model,
            ),
            "embedding": published_embedding,
            "reclaimed_bytes": receipt.reclaimed_bytes,
            "resident_bytes": receipt.resident_bytes,
            "timing_us": {
                "unload": receipt.unload_elapsed.as_micros(),
                "load": receipt.load_elapsed.as_micros(),
                // Milestones measured from the start of the replacement
                // loader: native storage/config/tokenizer ready, then the
                // same generation ready after its registry warm forward.
                "load_ready": load_timing.weight_load_elapsed.as_micros(),
                "post_warm": load_timing.total_elapsed.as_micros(),
                "weight_load": load_timing.weight_load_elapsed.as_micros(),
                "registry_warm": load_timing.registry_warm_elapsed.as_micros(),
                "total_load": load_timing.total_elapsed.as_micros(),
            },
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(generation: u64, model_id: Option<&str>) -> EmbeddingModelSnapshot {
        EmbeddingModelSnapshot {
            generation: Some(generation),
            configured: model_id.is_some(),
            loading: false,
            model_id: model_id.map(str::to_owned),
            gguf_path: None,
            arch: model_id.map(|_| "bert"),
            resident_bytes: if model_id.is_some() { 1 } else { 0 },
            last_load_error: None,
        }
    }

    #[test]
    fn post_preparation_decision_never_uses_a_stale_resident_snapshot() {
        let concurrent_b = snapshot(2, Some("exact-b"));
        assert_eq!(
            decide_after_preparation(&concurrent_b, ModelActivationAction::Load, None, "exact-a",),
            PreparedDecision::Conflict
        );
        assert_eq!(
            decide_after_preparation(
                &concurrent_b,
                ModelActivationAction::Switch,
                Some(1),
                "exact-c",
            ),
            PreparedDecision::Stale {
                expected: 1,
                current: 2,
            }
        );
        assert_eq!(
            decide_after_preparation(&concurrent_b, ModelActivationAction::Load, None, "exact-b",),
            PreparedDecision::Resident
        );

        let failed_empty = snapshot(2, None);
        assert_eq!(
            decide_after_preparation(&failed_empty, ModelActivationAction::Load, None, "exact-a",),
            PreparedDecision::ReplaceAt(2)
        );
    }

    #[test]
    fn same_basename_catalog_candidates_do_not_alias_by_path() {
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("first").join("encoder.gguf");
        let second = root.path().join("second").join("encoder.gguf");
        std::fs::create_dir_all(first.parent().unwrap()).unwrap();
        std::fs::create_dir_all(second.parent().unwrap()).unwrap();
        std::fs::write(&first, b"first").unwrap();
        std::fs::write(&second, b"second").unwrap();

        let mut current = snapshot(7, Some("local://owner/a@rev/candidate-a"));
        current.gguf_path = Some(first);
        assert_eq!(
            decide_after_preparation(
                &current,
                ModelActivationAction::Load,
                None,
                "local://owner/b@rev/candidate-b",
            ),
            PreparedDecision::Conflict,
            "catalog identity, not a shared basename or filesystem path, decides residency"
        );
        assert!(!same_startup_artifact(
            current.gguf_path.as_deref().unwrap(),
            &second
        ));
    }
}
