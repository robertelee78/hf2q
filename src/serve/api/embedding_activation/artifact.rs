//! Exact artifact authority for the dedicated embedding activation domain.

use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::super::artifact_catalog::{ArtifactCatalogCoordinator, CatalogError, StoredArtifact};
use super::super::cancellation::{CancellationSignal, PreparationSupervisor};
use super::super::control::{
    fetch_hub_gguf_cancellable, normalize_hub_repository, verify_local_gguf_cancellable,
};
use super::super::schema::ApiError;
use super::super::state::AppState;

enum Payload {
    VerifiedLocal(super::super::local_artifacts::LocalGgufArtifact),
    Hosted(crate::input::hf_download::HubGgufArtifact),
}

pub(super) struct Target {
    pub(super) repo: String,
    pub(super) quant_hint: String,
    pub(super) file_type: u32,
    pub(super) bytes: u64,
    pub(super) request_model: String,
    payload: Payload,
}

impl Target {
    pub(super) async fn materialize(
        self,
        cancellation: CancellationSignal,
        supervisor: PreparationSupervisor,
        catalog: &ArtifactCatalogCoordinator,
    ) -> std::result::Result<(String, String, std::path::PathBuf, u64, String), Response> {
        let path = match &self.payload {
            Payload::VerifiedLocal(artifact) => {
                let permit = catalog.try_local_verify_slot().map_err(|error| match error {
                    CatalogError::LocalVerifyBusy => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        [(header::RETRY_AFTER, "1")],
                        Json(serde_json::json!({"error":{"message":error.to_string(),"type":"server_busy","code":"local_artifact_verification_busy"}})),
                    )
                        .into_response(),
                    _ => ApiError::internal_error().into_response(),
                })?;
                verify_local_gguf_cancellable(artifact, cancellation, supervisor, permit).await?
            }
            Payload::Hosted(artifact) => {
                let permit = catalog.try_transfer_slot().map_err(|error| match error {
                    CatalogError::TransferBusy => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        [(header::RETRY_AFTER, "1")],
                        Json(serde_json::json!({"error":{"message":error.to_string(),"type":"server_busy","code":"artifact_transfer_busy"}})),
                    )
                        .into_response(),
                    _ => ApiError::internal_error().into_response(),
                })?;
                fetch_hub_gguf_cancellable(artifact, cancellation, supervisor, permit).await?
            }
        };
        let header = mlx_native::gguf::GgufFile::open(&path).map_err(|error| {
            tracing::warn!(%error, "selected embedding artifact failed GGUF header validation");
            ApiError::invalid_request(
                "selected embedding artifact is not a loadable GGUF",
                Some("artifact".into()),
            )
            .into_response()
        })?;
        let (file_type, _) = header_identity(&header)?;
        if file_type != self.file_type {
            return Err(ApiError::invalid_request(
                "selected embedding GGUF file type changed after catalog resolution",
                Some("artifact".into()),
            )
            .into_response());
        }
        Ok((
            self.repo,
            self.quant_hint,
            path,
            self.bytes,
            self.request_model,
        ))
    }
}

fn header_identity(
    header: &mlx_native::gguf::GgufFile,
) -> std::result::Result<(u32, String), Response> {
    let arch = header.metadata_string("general.architecture").unwrap_or("");
    if !matches!(arch, "bert" | "nomic-bert") {
        return Err(ApiError::invalid_request(
            format!(
                "kind=embedding requires general.architecture bert or nomic-bert, found {arch:?}"
            ),
            Some("model".into()),
        )
        .into_response());
    }
    let file_type = header.metadata_u32("general.file_type").ok_or_else(|| {
        ApiError::invalid_request(
            "embedding GGUF has no general.file_type",
            Some("model".into()),
        )
        .into_response()
    })?;
    let file_type_name = crate::quantize::ggml_quants::GgufFtype::try_from(file_type)
        .map_err(|_| {
            ApiError::invalid_request(
                format!("embedding GGUF file type {file_type} is not recognized"),
                Some("model".into()),
            )
            .into_response()
        })?
        .name()
        .to_ascii_uppercase();
    Ok((file_type, file_type_name))
}

pub(super) async fn resolve(
    state: &AppState,
    requested_model: &str,
    candidate_id: &str,
) -> std::result::Result<Target, Response> {
    let model = requested_model.trim();
    if model.is_empty() {
        return Err(
            ApiError::invalid_request("model must not be empty", Some("model".into()))
                .into_response(),
        );
    }
    let hub_repository = normalize_hub_repository(model)?;
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
    match artifact {
        StoredArtifact::Hosted(artifact) => {
            if artifact.role != "embedding_model"
                || hub_repository.as_deref() != Some(artifact.repository.as_str())
            {
                return Err(ApiError::invalid_request(
                    "artifact selection is not an embedding candidate for the requested model",
                    Some("candidate_id".into()),
                )
                .into_response());
            }
            let quant_hint = artifact.quant_hint.clone().ok_or_else(|| {
                ApiError::invalid_request(
                    "selected embedding artifact has no quant identity",
                    Some("candidate_id".into()),
                )
                .into_response()
            })?;
            let file_type = crate::quantize::ggml_quants::GgufFtype::from_name(
                &quant_hint.to_ascii_lowercase(),
            )
            .map(u32::from)
            .ok_or_else(|| {
                ApiError::invalid_request(
                    "selected embedding artifact has an unsupported file type",
                    Some("candidate_id".into()),
                )
                .into_response()
            })?;
            let request_model = artifact.request_model();
            Ok(Target {
                repo: artifact.repository.clone(),
                quant_hint,
                file_type,
                bytes: artifact.bytes,
                request_model,
                payload: Payload::Hosted(artifact),
            })
        }
        StoredArtifact::Local(artifact) => {
            if artifact.role != "embedding_model"
                || hub_repository.as_deref() != Some(artifact.repository.as_str())
            {
                return Err(ApiError::invalid_request(
                    "local artifact selection is not an embedding candidate for the requested model",
                    Some("candidate_id".into()),
                )
                .into_response());
            }
            let request_model = format!(
                "local://{}@{}/{}",
                artifact.repository, artifact.revision, candidate_id
            );
            Ok(Target {
                repo: artifact.repository.clone(),
                quant_hint: artifact.quant_hint.clone(),
                file_type: artifact.file_type,
                bytes: artifact.bytes,
                request_model,
                payload: Payload::VerifiedLocal(artifact),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::api::local_artifacts::{
        LocalArtifactCatalog, LocalArtifactProvenance, LocalGgufArtifact,
    };
    use crate::serve::api::state::ServerConfig;

    fn local_candidate(path: std::path::PathBuf, sha256: &str) -> LocalGgufArtifact {
        LocalGgufArtifact {
            repository: "owner/encoder".into(),
            revision: "a".repeat(40),
            filename: "encoder.gguf".into(),
            root: path.parent().unwrap().to_path_buf(),
            path,
            bytes: 4,
            sha256: sha256.into(),
            quant_hint: "Q4_0".into(),
            file_type: 2,
            quant: None,
            role: "embedding_model".into(),
            selectable: true,
            unavailable_reason: None,
            provenance: LocalArtifactProvenance::ConversionReceipt,
        }
    }

    #[tokio::test]
    async fn exact_catalog_ids_separate_same_basename_local_artifacts() {
        let state = AppState::new(ServerConfig::default());
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("first").join("encoder.gguf");
        let second = root.path().join("second").join("encoder.gguf");
        std::fs::create_dir_all(first.parent().unwrap()).unwrap();
        std::fs::create_dir_all(second.parent().unwrap()).unwrap();
        std::fs::write(&first, b"GGUF").unwrap();
        std::fs::write(&second, b"GGUF").unwrap();
        let catalog = state
            .artifact_catalog
            .register_local(LocalArtifactCatalog {
                artifacts: vec![
                    local_candidate(first, &"1".repeat(64)),
                    local_candidate(second, &"2".repeat(64)),
                ],
                warnings: Vec::new(),
            })
            .unwrap();
        let first_id = catalog.candidates[0].candidate_id.as_deref().unwrap();
        let second_id = catalog.candidates[1].candidate_id.as_deref().unwrap();

        let first_target = resolve(&state, "owner/encoder", first_id).await.unwrap();
        let second_target = resolve(&state, "owner/encoder", second_id).await.unwrap();
        assert_ne!(first_target.request_model, second_target.request_model);
        assert!(first_target.request_model.ends_with(first_id));
        assert!(second_target.request_model.ends_with(second_id));

        let wrong_repo = match resolve(&state, "owner/other", first_id).await {
            Ok(_) => panic!("candidate authority must be repository-bound"),
            Err(response) => response,
        };
        assert_eq!(wrong_repo.status(), StatusCode::BAD_REQUEST);
    }
}
