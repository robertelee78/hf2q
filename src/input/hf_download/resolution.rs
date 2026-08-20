use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

use hf_hub::{Repo, RepoType};
use thiserror::Error;

use super::{
    build_hub_api, fetch_expected_file_metadata, resolve_hf_cache_dir, validate_repo_inventory,
    DownloadError,
};
use crate::core::integrity::ShardIntegrity;
use crate::input::hf_reference::{HfModelReference, ResolvedHfModelReference};
use crate::input::model_recipe::{
    ModelPreparationError, ModelPreparationPlan, RecipeArtifactRole, RecipeSourceFile,
};

mod payload;

pub use payload::{
    authenticate_transferred_model_preparation, convert_authenticated_model_preparation,
    publish_converted_model_preparation_keep, transfer_authorized_model_preparation,
    AuthenticatedModelPreparationSource, ConvertedModelPreparation,
    ModelPreparationConversionError, ModelPreparationPayloadError,
    ModelPreparationPublicationError, ModelPreparationSourceAuthenticationError,
    RegisteredModelPreparation, TransferredModelPreparationPayload,
};

/// One bounded repository inventory resolved by the pinned in-process Hub
/// client before any model payload is transferred.
///
/// This value is structural resolution evidence only. It grants no download,
/// conversion, filesystem, or serving authority and is deliberately
/// non-Clone. Its inventory remains private so unrelated repository entries
/// cannot become caller-selected inputs.
pub struct ResolvedModelRepository {
    reference: ResolvedHfModelReference,
    inventory: BTreeSet<String>,
}

impl fmt::Debug for ResolvedModelRepository {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedModelRepository")
            .field("repository_id", &self.reference.repo_id())
            .field("revision", &self.reference.revision())
            .field("inventory_len", &self.inventory.len())
            .finish()
    }
}

impl ResolvedModelRepository {
    pub fn reference(&self) -> &ResolvedHfModelReference {
        &self.reference
    }

    pub fn inventory_len(&self) -> usize {
        self.inventory.len()
    }

    pub(in crate::input) fn contains(&self, filename: &str) -> bool {
        self.inventory.contains(filename)
    }

    fn new(reference: ResolvedHfModelReference, inventory: BTreeSet<String>) -> Self {
        Self {
            reference,
            inventory,
        }
    }

    pub(super) fn into_download_parts(self) -> (ResolvedHfModelReference, BTreeSet<String>) {
        (self.reference, self.inventory)
    }

    #[cfg(test)]
    pub(in crate::input) fn for_test(
        reference: ResolvedHfModelReference,
        inventory: impl IntoIterator<Item = String>,
    ) -> Self {
        Self::new(reference, inventory.into_iter().collect())
    }
}

/// The host-checked no-options plan after its exact original reference was
/// resolved to the accepted commit and complete recipe-owned name inventory.
///
/// This value remains inert and non-Clone. Keeping both inputs private in the
/// download module prevents later code from separating or recombining the
/// sealed resolution and plan before the payload-transfer transition lands.
pub struct ResolvedModelPreparationPlan {
    plan: ModelPreparationPlan,
    resolution: ResolvedModelRepository,
}

impl fmt::Debug for ResolvedModelPreparationPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResolvedModelPreparationPlan")
            .field("recipe_id", &self.plan.recipe_id())
            .field("repository_id", &self.resolution.reference().repo_id())
            .field("revision", &self.resolution.reference().revision())
            .field("inventory_len", &self.resolution.inventory_len())
            .field("paths", &"[redacted]")
            .finish()
    }
}

impl ResolvedModelPreparationPlan {
    pub fn recipe_id(&self) -> &str {
        self.plan.recipe_id()
    }

    pub fn resolved_reference(&self) -> &ResolvedHfModelReference {
        self.resolution.reference()
    }

    pub fn repository_inventory_len(&self) -> usize {
        self.resolution.inventory_len()
    }

    pub fn model_root(&self) -> &Path {
        self.plan.model_root()
    }

    pub fn source_root(&self) -> &Path {
        self.plan.source_root()
    }

    pub fn artifact_path(&self, role: RecipeArtifactRole) -> &Path {
        self.plan.artifact_path(role)
    }

    pub fn conversion_receipt_path(&self, role: RecipeArtifactRole) -> &Path {
        self.plan.conversion_receipt_path(role)
    }

    pub fn preparation_receipt_path(&self) -> &Path {
        self.plan.preparation_receipt_path()
    }

    pub fn profile_path(&self) -> &Path {
        self.plan.profile_path()
    }

    pub(super) fn expected_source_files(&self) -> &[RecipeSourceFile] {
        self.plan.expected_source_files()
    }
}

/// Exact recipe-owned Hub metadata authorized before any model payload is
/// transferred.
///
/// This token consumes the resolved preparation plan, retains only the 29
/// canonical recipe records, and is deliberately non-Clone. It grants no
/// payload-transfer, conversion, filesystem-mutation, deletion, registration,
/// calibration, or serving authority.
pub struct AuthorizedModelPreparationTransfer {
    resolved: ResolvedModelPreparationPlan,
    records: Vec<ShardIntegrity>,
}

impl fmt::Debug for AuthorizedModelPreparationTransfer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AuthorizedModelPreparationTransfer")
            .field("recipe_id", &self.resolved.recipe_id())
            .field(
                "repository_id",
                &self.resolved.resolved_reference().repo_id(),
            )
            .field("revision", &self.resolved.resolved_reference().revision())
            .field("authorized_file_count", &self.records.len())
            .field("records", &"[redacted]")
            .finish()
    }
}

impl AuthorizedModelPreparationTransfer {
    pub fn recipe_id(&self) -> &str {
        self.resolved.recipe_id()
    }

    pub fn resolved_reference(&self) -> &ResolvedHfModelReference {
        self.resolved.resolved_reference()
    }

    pub fn authorized_file_count(&self) -> usize {
        self.records.len()
    }

    pub fn model_root(&self) -> &Path {
        self.resolved.model_root()
    }

    pub fn source_root(&self) -> &Path {
        self.resolved.source_root()
    }
}

pub(super) fn bind_transfer_authorization(
    resolved: ResolvedModelPreparationPlan,
    records: Vec<ShardIntegrity>,
) -> Result<AuthorizedModelPreparationTransfer, ModelPreparationError> {
    let expected = resolved.expected_source_files();
    if records.len() != expected.len() {
        return Err(preparation_error(format!(
            "Hub metadata authorized {} files; recipe requires {}",
            records.len(),
            expected.len()
        )));
    }

    let mut canonical = Vec::with_capacity(expected.len());
    for (actual, expected) in records.iter().zip(expected) {
        let expected_lfs = expected.hf_lfs_sha256();
        let same = actual.filename == expected.path()
            && actual.bytes == expected.size()
            && actual.hf_etag.eq_ignore_ascii_case(expected.hub_etag())
            && actual.is_lfs == expected_lfs.is_some()
            && match (actual.sha256.as_deref(), expected_lfs) {
                (Some(actual), Some(expected)) => actual.eq_ignore_ascii_case(expected),
                (None, None) => true,
                _ => false,
            };
        if !same {
            return Err(preparation_error(format!(
                "Hub metadata does not match recipe source `{}`",
                expected.path()
            )));
        }
        canonical.push(ShardIntegrity {
            filename: expected.path().to_owned(),
            bytes: expected.size(),
            sha256: expected_lfs.map(str::to_owned),
            hf_etag: expected.hub_etag().to_owned(),
            is_lfs: expected_lfs.is_some(),
        });
    }

    Ok(AuthorizedModelPreparationTransfer {
        resolved,
        records: canonical,
    })
}

/// Authenticate the exact recipe-owned remote metadata before any model
/// payload is transferred.
///
/// This transition intentionally lives beside the sealed resolution and
/// authorization types. CI forbids payload-transfer and filesystem-mutation
/// authority in this module.
pub fn authorize_model_preparation_transfer(
    resolved: ResolvedModelPreparationPlan,
) -> Result<AuthorizedModelPreparationTransfer, ModelPreparationResolutionError> {
    let cache_dir = resolve_hf_cache_dir();
    let api = build_hub_api(&cache_dir, false)?;
    let reference = resolved.resolved_reference();
    let repo = api.repo(Repo::with_revision(
        reference.repo_id().to_owned(),
        RepoType::Model,
        reference.revision().to_owned(),
    ));
    let mut records = Vec::with_capacity(resolved.expected_source_files().len());
    for expected in resolved.expected_source_files() {
        records.push(fetch_expected_file_metadata(
            &api,
            &repo,
            reference,
            expected.path(),
        )?);
    }
    Ok(bind_transfer_authorization(resolved, records)?)
}

#[cfg(test)]
pub(in crate::input) fn bind_transfer_authorization_for_test(
    resolved: ResolvedModelPreparationPlan,
    records: Vec<ShardIntegrity>,
) -> Result<AuthorizedModelPreparationTransfer, ModelPreparationError> {
    bind_transfer_authorization(resolved, records)
}

fn preparation_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::PlanInvalid {
        reason: reason.into(),
    }
}

#[derive(Debug, Error)]
pub enum ModelPreparationResolutionError {
    #[error("model preparation Hub resolution: {0}")]
    Download(#[from] DownloadError),
    #[error(transparent)]
    Preparation(#[from] ModelPreparationError),
}

pub(super) fn bind_model_preparation_resolution(
    plan: ModelPreparationPlan,
    resolution: ResolvedModelRepository,
) -> Result<ResolvedModelPreparationPlan, ModelPreparationError> {
    plan.validate_resolution(resolution.reference(), |filename| {
        resolution.contains(filename)
    })?;
    Ok(ResolvedModelPreparationPlan { plan, resolution })
}

#[cfg(test)]
pub(in crate::input) fn bind_model_preparation_resolution_for_test(
    plan: ModelPreparationPlan,
    resolution: ResolvedModelRepository,
) -> Result<ResolvedModelPreparationPlan, ModelPreparationError> {
    bind_model_preparation_resolution(plan, resolution)
}

pub(super) fn resolve_repository_info(
    reference: HfModelReference,
    requested_revision: &str,
    info: &hf_hub::api::RepoInfo,
) -> Result<ResolvedModelRepository, DownloadError> {
    let requested_exact = requested_revision.len() == 40
        && requested_revision
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit());
    if requested_exact && !requested_revision.eq_ignore_ascii_case(&info.sha) {
        return Err(DownloadError::InvalidRepositoryInventory {
            reason: format!(
                "repository lookup returned commit `{}` instead of explicitly requested `{requested_revision}`",
                info.sha
            ),
        });
    }
    let resolved = reference.resolve(&info.sha)?;
    let inventory = validate_repo_inventory(
        info.siblings
            .iter()
            .map(|sibling| sibling.rfilename.as_str()),
    )?;
    Ok(ResolvedModelRepository::new(resolved, inventory))
}

#[cfg(test)]
pub(in crate::input) fn resolve_repository_info_for_test(
    reference: HfModelReference,
    requested_revision: &str,
    info: &hf_hub::api::RepoInfo,
) -> Result<ResolvedModelRepository, DownloadError> {
    resolve_repository_info(reference, requested_revision, info)
}
