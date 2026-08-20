use std::collections::BTreeSet;
use std::fmt;
use std::path::Path;

use thiserror::Error;

use super::{validate_repo_inventory, DownloadError};
use crate::input::hf_reference::{HfModelReference, ResolvedHfModelReference};
use crate::input::model_recipe::{ModelPreparationError, ModelPreparationPlan, RecipeArtifactRole};

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
