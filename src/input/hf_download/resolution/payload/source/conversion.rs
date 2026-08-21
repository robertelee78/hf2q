use std::fmt;
use std::path::{Path, PathBuf};

use thiserror::Error;

use super::AuthenticatedModelPreparationSource;
use crate::convert::cli_driver::{
    run_convert_with_recipe_producer_version, ConvertArgs, ConvertError,
};
use crate::convert::receipt::{ReceiptError, RemoteConversionSource};
use crate::convert::QuantSelector;
use crate::input::integrity::verify_conversion_manifest;
use crate::input::model_recipe::{
    ModelPreparationError, ModelRecipeError, RecipeArtifactRole, VerifiedModelPreparation,
    VerifiedRecipeConversion, VerifiedRecipeSource,
};
use crate::quantize::ggml_quants::GgufFtype;

mod publication;

pub use publication::{
    publish_converted_model_preparation_keep, ModelPreparationPublicationError,
    RegisteredModelPreparation,
};

/// Recipe-authenticated text/projector output pair produced or adopted at the
/// plan-owned paths.
///
/// This non-cloneable value is still inert. It grants no source deletion,
/// preparation-receipt publication, registry/profile mutation, calibration,
/// serving, installation, or activation authority.
pub struct ConvertedModelPreparation {
    prepared: VerifiedModelPreparation,
    model_root: PathBuf,
}

impl fmt::Debug for ConvertedModelPreparation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ConvertedModelPreparation")
            .field("recipe_id", &self.prepared.receipt().recipe_id())
            .field("state", &"awaiting_runtime_calibration")
            .field("paths", &"[redacted]")
            .finish()
    }
}

impl ConvertedModelPreparation {
    pub fn recipe_id(&self) -> &str {
        self.prepared.receipt().recipe_id()
    }

    pub fn model_root(&self) -> &Path {
        &self.model_root
    }

    pub fn preparation_receipt_bytes(&self) -> &[u8] {
        self.prepared.receipt_bytes()
    }
}

#[derive(Debug, Error)]
pub enum ModelPreparationConversionError {
    #[error(transparent)]
    Conversion(#[from] ConvertError),
    #[error(transparent)]
    Integrity(#[from] crate::core::integrity::IntegrityError),
    #[error(transparent)]
    Preparation(#[from] ModelPreparationError),
    #[error(transparent)]
    Recipe(#[from] ModelRecipeError),
    #[error(transparent)]
    Receipt(#[from] ReceiptError),
    #[error(transparent)]
    SourceAuthentication(#[from] super::ModelPreparationSourceAuthenticationError),
    #[error("model preparation conversion filesystem: {0}")]
    Io(#[from] std::io::Error),
}

trait ConversionBackend {
    fn reauthenticate_source(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
        previous: &VerifiedRecipeSource,
    ) -> Result<VerifiedRecipeSource, ModelPreparationConversionError>;

    fn ensure_layout(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
    ) -> Result<(), ModelPreparationConversionError>;

    fn ensure_role(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
        source: &VerifiedRecipeSource,
        role: RecipeArtifactRole,
    ) -> Result<VerifiedRecipeConversion, ModelPreparationConversionError>;

    fn reauthenticate_role(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
        role: RecipeArtifactRole,
    ) -> Result<VerifiedRecipeConversion, ModelPreparationConversionError>;
}

struct Hf2qConversionBackend;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoleRestartDisposition {
    Convert,
    ReconvertVerifiedArtifact,
    Adopt,
}

/// Consume the sealed offline source proof, repeat source authentication at
/// the conversion mutation boundary, and produce or exact-adopt the recipe's
/// canonical text/projector pair through hf2q's Rust converter.
pub fn convert_authenticated_model_preparation(
    authenticated: AuthenticatedModelPreparationSource,
) -> Result<ConvertedModelPreparation, ModelPreparationConversionError> {
    convert_with(authenticated, &mut Hf2qConversionBackend)
}

fn convert_with(
    authenticated: AuthenticatedModelPreparationSource,
    backend: &mut impl ConversionBackend,
) -> Result<ConvertedModelPreparation, ModelPreparationConversionError> {
    let AuthenticatedModelPreparationSource {
        resolved,
        source: initial_source,
    } = authenticated;

    let mut source = backend.reauthenticate_source(&resolved, &initial_source)?;
    backend.ensure_layout(&resolved)?;
    let _text = backend.ensure_role(&resolved, &source, RecipeArtifactRole::Text)?;
    source = backend.reauthenticate_source(&resolved, &source)?;
    let _projector =
        backend.ensure_role(&resolved, &source, RecipeArtifactRole::VisionProjector)?;
    source = backend.reauthenticate_source(&resolved, &source)?;
    let text = backend.reauthenticate_role(&resolved, RecipeArtifactRole::Text)?;
    let projector = backend.reauthenticate_role(&resolved, RecipeArtifactRole::VisionProjector)?;

    let model_root = resolved.model_root().to_path_buf();
    let prepared = resolved.plan.bind_prepared_pair(source, text, projector)?;
    Ok(ConvertedModelPreparation {
        prepared,
        model_root,
    })
}

impl ConversionBackend for Hf2qConversionBackend {
    fn reauthenticate_source(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
        previous: &VerifiedRecipeSource,
    ) -> Result<VerifiedRecipeSource, ModelPreparationConversionError> {
        resolved.plan.revalidate_source_root_before_mutation()?;
        let snapshot = previous.local_dir();
        if snapshot != super::expected_snapshot_dir(resolved) {
            return Err(conversion_plan_error(
                "authenticated source snapshot differs from the sealed plan",
            )
            .into());
        }
        let repository = snapshot
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| conversion_plan_error("source snapshot has no repository parent"))?;
        let snapshots = snapshot
            .parent()
            .ok_or_else(|| conversion_plan_error("source snapshot has no snapshots parent"))?;
        let directories = [resolved.source_root(), repository, snapshots, snapshot];
        let identities = directories
            .iter()
            .map(|path| super::require_exact_source_directory(path))
            .collect::<Result<Vec<_>, _>>()?;

        let reference = resolved.resolved_reference();
        let verified = verify_conversion_manifest(
            reference.repo_id(),
            reference.revision(),
            snapshot,
            previous.source_manifest().records().to_vec(),
        )?;
        let source = resolved.plan.authenticate_source(snapshot, verified)?;

        resolved.plan.revalidate_source_root_before_mutation()?;
        for (directory, expected) in directories.into_iter().zip(identities) {
            if super::require_exact_source_directory(directory)? != expected {
                return Err(conversion_plan_error(
                    "source namespace changed during conversion-bound authentication",
                )
                .into());
            }
        }
        Ok(source)
    }

    fn ensure_layout(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
    ) -> Result<(), ModelPreparationConversionError> {
        super::require_exact_source_directory(resolved.model_root())?;
        for directory in [
            resolved.plan.artifacts_root(),
            resolved.plan.receipts_root(),
        ] {
            match std::fs::create_dir(directory) {
                Ok(()) => sync_directory(resolved.model_root())?,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
            super::require_exact_source_directory(directory)?;
        }
        Ok(())
    }

    fn ensure_role(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
        source: &VerifiedRecipeSource,
        role: RecipeArtifactRole,
    ) -> Result<VerifiedRecipeConversion, ModelPreparationConversionError> {
        let artifact = resolved.artifact_path(role);
        let receipt = resolved.conversion_receipt_path(role);
        let artifact_exists = path_entry_exists(artifact)?;
        let receipt_exists = path_entry_exists(receipt)?;
        match role_restart_disposition(artifact_exists, receipt_exists)? {
            RoleRestartDisposition::Adopt => {
                return Ok(resolved.plan.verify_completed_conversion(role)?)
            }
            RoleRestartDisposition::ReconvertVerifiedArtifact => {
                resolved.plan.verify_completed_artifact(role)?;
            }
            RoleRestartDisposition::Convert => {}
        }

        let remote = RemoteConversionSource::from_verified(
            resolved.resolved_reference().clone(),
            source.local_dir(),
            source.source_manifest(),
        )?;
        run_convert_with_recipe_producer_version(
            ConvertArgs {
                hf_dir: source.local_dir().to_path_buf(),
                selector: QuantSelector::Standard(GgufFtype::MostlyQ4_K_M),
                output: artifact.to_path_buf(),
                dry_run: false,
                imatrix: None,
                imatrix_corpus: None,
                imatrix_out: None,
                imatrix_n_ctx: None,
                mmproj: role == RecipeArtifactRole::VisionProjector,
                remote_source: Some(remote),
            },
            receipt,
            resolved.plan.producer_version(),
        )?;
        sync_directory(resolved.plan.artifacts_root())?;
        sync_directory(resolved.plan.receipts_root())?;
        Ok(resolved.plan.verify_completed_conversion(role)?)
    }

    fn reauthenticate_role(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
        role: RecipeArtifactRole,
    ) -> Result<VerifiedRecipeConversion, ModelPreparationConversionError> {
        Ok(resolved.plan.verify_completed_conversion(role)?)
    }
}

fn path_entry_exists(path: &Path) -> Result<bool, std::io::Error> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn role_restart_disposition(
    artifact_exists: bool,
    receipt_exists: bool,
) -> Result<RoleRestartDisposition, ModelPreparationConversionError> {
    match (artifact_exists, receipt_exists) {
        (false, false) => Ok(RoleRestartDisposition::Convert),
        (true, false) => Ok(RoleRestartDisposition::ReconvertVerifiedArtifact),
        (true, true) => Ok(RoleRestartDisposition::Adopt),
        (false, true) => Err(conversion_plan_error(
            "conversion receipt exists without its canonical artifact",
        )
        .into()),
    }
}

fn sync_directory(path: &Path) -> Result<(), std::io::Error> {
    std::fs::File::open(path)?.sync_all()
}

fn conversion_plan_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::PlanInvalid {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
