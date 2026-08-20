use std::fmt;
use std::path::{Path, PathBuf};

use hf_hub::api::sync::ApiRepo;
use hf_hub::{Repo, RepoType};
use thiserror::Error;

use super::{AuthorizedModelPreparationTransfer, ResolvedModelPreparationPlan};
use crate::core::integrity::{verify_shard, IntegrityError, ShardIntegrity};
use crate::input::hf_download::{
    bind_snapshot_parent, build_hub_api, download_file, DownloadError,
};
use crate::input::model_recipe::{canonical_future_directory, ModelPreparationError};
use crate::progress::ProgressReporter;

mod source;

pub use source::{
    authenticate_transferred_model_preparation, convert_authenticated_model_preparation,
    publish_converted_model_preparation_keep, AuthenticatedModelPreparationSource,
    ConvertedModelPreparation, ModelPreparationConversionError, ModelPreparationPublicationError,
    ModelPreparationSourceAuthenticationError, RegisteredModelPreparation,
};

/// Recipe-owned payload bytes fetched and individually authenticated at the
/// exact accepted Hub commit.
///
/// This non-cloneable token is deliberately inert. It grants no conversion,
/// source deletion, artifact publication, registration, calibration, or
/// serving authority. A later boundary must re-open and re-authenticate every
/// retained file before conversion.
pub struct TransferredModelPreparationPayload {
    resolved: ResolvedModelPreparationPlan,
    records: Vec<ShardIntegrity>,
    _snapshot_dir: PathBuf,
}

impl fmt::Debug for TransferredModelPreparationPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TransferredModelPreparationPayload")
            .field("recipe_id", &self.resolved.recipe_id())
            .field(
                "repository_id",
                &self.resolved.resolved_reference().repo_id(),
            )
            .field("revision", &self.resolved.resolved_reference().revision())
            .field("transferred_file_count", &self.records.len())
            .field("paths", &"[redacted]")
            .finish()
    }
}

impl TransferredModelPreparationPayload {
    pub fn recipe_id(&self) -> &str {
        self.resolved.recipe_id()
    }

    pub fn transferred_file_count(&self) -> usize {
        self.records.len()
    }

    pub fn model_root(&self) -> &Path {
        self.resolved.model_root()
    }
}

#[derive(Debug, Error)]
pub enum ModelPreparationPayloadError {
    #[error(transparent)]
    Download(#[from] DownloadError),
    #[error(transparent)]
    Integrity(#[from] IntegrityError),
    #[error(transparent)]
    Preparation(#[from] ModelPreparationError),
    #[error("model preparation payload filesystem: {0}")]
    Io(#[from] std::io::Error),
}

trait PayloadExecutor {
    fn transfer_one(
        &mut self,
        expected: &ShardIntegrity,
    ) -> Result<PathBuf, ModelPreparationPayloadError>;
}

struct HubPayloadExecutor {
    repo: ApiRepo,
    repository_id: String,
    revision: String,
}

impl PayloadExecutor for HubPayloadExecutor {
    fn transfer_one(
        &mut self,
        expected: &ShardIntegrity,
    ) -> Result<PathBuf, ModelPreparationPayloadError> {
        let local = download_file(&self.repo, &self.repository_id, &expected.filename)?;
        verify_shard(&self.repository_id, &self.revision, &local, expected)?;
        Ok(local)
    }
}

/// Consume exact metadata authorization and transfer only its closed 29-file
/// recipe inventory through the pinned in-process Hub client.
pub fn transfer_authorized_model_preparation(
    authorized: AuthorizedModelPreparationTransfer,
    progress: &ProgressReporter,
) -> Result<TransferredModelPreparationPayload, ModelPreparationPayloadError> {
    authorized
        .resolved
        .plan
        .revalidate_source_root_before_mutation()?;
    let source_root = authorized.resolved.source_root().to_path_buf();
    std::fs::create_dir_all(&source_root)?;
    require_exact_directory(&source_root)?;

    let reference = authorized.resolved.resolved_reference();
    let repo_spec = Repo::with_revision(
        reference.repo_id().to_owned(),
        RepoType::Model,
        reference.revision().to_owned(),
    );
    let repository_root = source_root.join(repo_spec.folder_name());
    let blobs_root = repository_root.join("blobs");
    let refs_root = repository_root.join("refs");
    let snapshots_root = repository_root.join("snapshots");
    let expected_snapshot = snapshots_root.join(reference.revision());
    for directory in [
        &repository_root,
        &blobs_root,
        &refs_root,
        &snapshots_root,
        &expected_snapshot,
    ] {
        require_unchanged_future_directory(directory)?;
    }

    let api = build_hub_api(&source_root, true)?;
    let repo = api.repo(repo_spec);
    let mut executor = HubPayloadExecutor {
        repo,
        repository_id: reference.repo_id().to_owned(),
        revision: reference.revision().to_owned(),
    };
    let bar = progress.bar(
        authorized.records.len() as u64,
        "Transferring authenticated model source",
    );
    let result = transfer_with_executor(authorized, &expected_snapshot, &mut executor, || {
        bar.inc(1);
    })
    .and_then(|payload| {
        for directory in [&repository_root, &blobs_root, &refs_root, &snapshots_root] {
            require_exact_directory(directory)?;
        }
        Ok(payload)
    });
    match &result {
        Ok(payload) => bar.finish_with_message(format!(
            "Transferred {} authenticated model files",
            payload.transferred_file_count()
        )),
        Err(_) => bar.abandon_with_message("Model source transfer stopped"),
    }
    result
}

fn transfer_with_executor(
    authorized: AuthorizedModelPreparationTransfer,
    expected_snapshot: &Path,
    executor: &mut impl PayloadExecutor,
    mut completed: impl FnMut(),
) -> Result<TransferredModelPreparationPayload, ModelPreparationPayloadError> {
    let AuthorizedModelPreparationTransfer { resolved, records } = authorized;
    let revision = resolved.resolved_reference().revision();
    let mut selected_snapshot = None;
    for expected in &records {
        let local = executor.transfer_one(expected)?;
        bind_snapshot_parent(&mut selected_snapshot, &local, &expected.filename, revision)?;
        completed();
    }
    let snapshot_dir =
        selected_snapshot.ok_or_else(|| DownloadError::InvalidRepositoryInventory {
            reason: "authorized recipe payload is empty".to_owned(),
        })?;
    if snapshot_dir != expected_snapshot {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: "downloaded recipe payload escaped its planned source cache".to_owned(),
        }
        .into());
    }
    require_exact_directory(&snapshot_dir)?;
    for directory in [
        resolved.source_root(),
        snapshot_dir
            .parent()
            .expect("snapshot has a snapshots parent"),
        snapshot_dir
            .parent()
            .and_then(Path::parent)
            .expect("snapshot has a repository parent"),
    ] {
        require_exact_directory(directory)?;
    }
    Ok(TransferredModelPreparationPayload {
        resolved,
        records,
        _snapshot_dir: snapshot_dir,
    })
}

fn require_unchanged_future_directory(path: &Path) -> Result<(), ModelPreparationPayloadError> {
    if canonical_future_directory(path)? != path {
        return Err(preparation_error(
            "model source cache namespace changed before payload transfer",
        )
        .into());
    }
    Ok(())
}

fn require_exact_directory(path: &Path) -> Result<(), ModelPreparationPayloadError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_dir() || path.canonicalize()? != path {
        return Err(preparation_error(
            "model source cache directory is not the sealed canonical directory",
        )
        .into());
    }
    Ok(())
}

fn preparation_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::PlanInvalid {
        reason: reason.into(),
    }
}

#[cfg(test)]
pub(in crate::input) fn transferred_payload_for_test(
    authorized: AuthorizedModelPreparationTransfer,
    snapshot_dir: PathBuf,
) -> TransferredModelPreparationPayload {
    let AuthorizedModelPreparationTransfer { resolved, records } = authorized;
    TransferredModelPreparationPayload {
        resolved,
        records,
        _snapshot_dir: snapshot_dir,
    }
}

#[cfg(test)]
mod tests;
