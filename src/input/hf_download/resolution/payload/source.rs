use std::fmt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use thiserror::Error;

use super::TransferredModelPreparationPayload;
use crate::core::integrity::{IntegrityError, ShardIntegrity};
use crate::input::integrity::verify_conversion_manifest;
use crate::input::model_recipe::{ModelPreparationError, VerifiedRecipeSource};

mod conversion;

pub use conversion::{
    convert_authenticated_model_preparation, ConvertedModelPreparation,
    ModelPreparationConversionError,
};

/// Exact recipe source bytes re-opened from the sealed payload cache and
/// authenticated against both Hub identity and hf2q's checked-in recipe.
///
/// This non-cloneable value remains inert until it is consumed by the one
/// recipe-owned conversion coordinator. It grants no source deletion,
/// registration, calibration, or serving authority, and its source proof and
/// paths remain private.
pub struct AuthenticatedModelPreparationSource {
    resolved: super::super::ResolvedModelPreparationPlan,
    source: VerifiedRecipeSource,
}

impl fmt::Debug for AuthenticatedModelPreparationSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthenticatedModelPreparationSource")
            .field("recipe_id", &self.resolved.recipe_id())
            .field(
                "repository_id",
                &self.resolved.resolved_reference().repo_id(),
            )
            .field("revision", &self.resolved.resolved_reference().revision())
            .field(
                "authenticated_file_count",
                &self.source.source_manifest().records().len(),
            )
            .field("paths", &"[redacted]")
            .finish()
    }
}

impl AuthenticatedModelPreparationSource {
    pub fn recipe_id(&self) -> &str {
        self.resolved.recipe_id()
    }

    pub fn authenticated_file_count(&self) -> usize {
        self.source.source_manifest().records().len()
    }

    pub fn model_root(&self) -> &Path {
        self.resolved.model_root()
    }
}

#[derive(Debug, Error)]
pub enum ModelPreparationSourceAuthenticationError {
    #[error(transparent)]
    Integrity(#[from] IntegrityError),
    #[error(transparent)]
    Preparation(#[from] ModelPreparationError),
    #[error("model preparation source filesystem: {0}")]
    Io(#[from] std::io::Error),
}

trait SourceAuthenticator {
    fn authenticate(
        &mut self,
        resolved: &super::super::ResolvedModelPreparationPlan,
        snapshot_dir: &Path,
        records: Vec<ShardIntegrity>,
    ) -> Result<VerifiedRecipeSource, ModelPreparationSourceAuthenticationError>;
}

struct OfflineSourceAuthenticator;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
}

impl SourceAuthenticator for OfflineSourceAuthenticator {
    fn authenticate(
        &mut self,
        resolved: &super::super::ResolvedModelPreparationPlan,
        snapshot_dir: &Path,
        records: Vec<ShardIntegrity>,
    ) -> Result<VerifiedRecipeSource, ModelPreparationSourceAuthenticationError> {
        let reference = resolved.resolved_reference();
        let verified = verify_conversion_manifest(
            reference.repo_id(),
            reference.revision(),
            snapshot_dir,
            records,
        )?;
        Ok(resolved.plan.authenticate_source(snapshot_dir, verified)?)
    }
}

/// Consume transferred payload evidence and re-open every exact recipe file
/// before creating a recipe-authenticated, still-inert source capability.
pub fn authenticate_transferred_model_preparation(
    transferred: TransferredModelPreparationPayload,
) -> Result<AuthenticatedModelPreparationSource, ModelPreparationSourceAuthenticationError> {
    authenticate_with(transferred, &mut OfflineSourceAuthenticator)
}

fn authenticate_with(
    transferred: TransferredModelPreparationPayload,
    authenticator: &mut impl SourceAuthenticator,
) -> Result<AuthenticatedModelPreparationSource, ModelPreparationSourceAuthenticationError> {
    let TransferredModelPreparationPayload {
        resolved,
        records,
        _snapshot_dir: snapshot_dir,
    } = transferred;
    resolved.plan.revalidate_source_root_before_mutation()?;
    let repository_dir = snapshot_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| source_error("snapshot has no repository parent"))?;
    let snapshots_dir = snapshot_dir
        .parent()
        .ok_or_else(|| source_error("snapshot has no snapshots parent"))?;
    let directories = [
        resolved.source_root(),
        repository_dir,
        snapshots_dir,
        &snapshot_dir,
    ];
    let identities = directories
        .iter()
        .map(|directory| require_exact_source_directory(directory))
        .collect::<Result<Vec<_>, _>>()?;
    let expected_snapshot = expected_snapshot_dir(&resolved);
    if snapshot_dir != expected_snapshot {
        return Err(source_error(
            "transferred payload snapshot does not match the sealed preparation plan",
        )
        .into());
    }

    let expected_count = records.len();
    let source = authenticator.authenticate(&resolved, &snapshot_dir, records)?;
    resolved.plan.revalidate_source_root_before_mutation()?;
    for (directory, expected_identity) in directories.into_iter().zip(identities) {
        if require_exact_source_directory(directory)? != expected_identity {
            return Err(source_error(
                "model source cache namespace changed during source authentication",
            )
            .into());
        }
    }
    let reference = resolved.resolved_reference();
    if source.recipe_id() != resolved.recipe_id()
        || source.local_dir() != snapshot_dir
        || source.source_manifest().records().len() != expected_count
        || source.source_manifest().repo() != reference.repo_id()
        || source.source_manifest().revision() != reference.revision()
    {
        return Err(source_error(
            "re-opened source proof does not match the transferred preparation",
        )
        .into());
    }
    Ok(AuthenticatedModelPreparationSource { resolved, source })
}

fn expected_snapshot_dir(resolved: &super::super::ResolvedModelPreparationPlan) -> PathBuf {
    let reference = resolved.resolved_reference();
    resolved
        .source_root()
        .join(
            hf_hub::Repo::with_revision(
                reference.repo_id().to_owned(),
                hf_hub::RepoType::Model,
                reference.revision().to_owned(),
            )
            .folder_name(),
        )
        .join("snapshots")
        .join(reference.revision())
}

fn source_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::PlanInvalid {
        reason: reason.into(),
    }
}

fn require_exact_source_directory(
    path: &Path,
) -> Result<DirectoryIdentity, ModelPreparationSourceAuthenticationError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_dir() || path.canonicalize()? != path {
        return Err(source_error(
            "model source cache directory is not the sealed canonical directory",
        )
        .into());
    }
    Ok(DirectoryIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
    })
}

#[cfg(test)]
mod tests;
