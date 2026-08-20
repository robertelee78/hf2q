use std::fmt;
use std::path::{Path, PathBuf};

use thiserror::Error;

use super::ConvertedModelPreparation;
use crate::input::model_recipe::{
    ModelPreparationError, ModelRecipeError, PreparedModelProfileV1, SourceRetentionChoice,
    MAX_MODEL_PREPARATION_RECEIPT_BYTES, MAX_PREPARED_MODEL_PROFILE_BYTES,
};

mod authentication;
mod file;

const PREPARATION_RECEIPT_NAME: &str = "model-preparation.json";
const PREPARATION_RECEIPT_PARTIAL: &str = ".model-preparation.json.partial";
const PROFILE_NAME: &str = "profile.json";
const PROFILE_PARTIAL: &str = ".profile.json.partial";

/// Durable, exact prepared-model registry entry still awaiting calibration.
///
/// This non-cloneable value does not grant source deletion, model loading,
/// serving, preference, calibration, installation, or activation authority.
pub struct RegisteredModelPreparation {
    profile: PreparedModelProfileV1,
    model_root: PathBuf,
}

impl fmt::Debug for RegisteredModelPreparation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RegisteredModelPreparation")
            .field("recipe_id", &self.profile.recipe_id())
            .field("state", &"awaiting_runtime_calibration")
            .field("source_retention", &SourceRetentionChoice::Keep)
            .field("paths", &"[redacted]")
            .finish()
    }
}

impl RegisteredModelPreparation {
    pub fn recipe_id(&self) -> &str {
        self.profile.recipe_id()
    }

    pub fn model_root(&self) -> &Path {
        &self.model_root
    }

    pub fn profile(&self) -> &PreparedModelProfileV1 {
        &self.profile
    }
}

#[derive(Debug, Error)]
pub enum ModelPreparationPublicationError {
    #[error(transparent)]
    Integrity(#[from] crate::core::integrity::IntegrityError),
    #[error(transparent)]
    Preparation(#[from] ModelPreparationError),
    #[error(transparent)]
    Recipe(#[from] ModelRecipeError),
    #[error(transparent)]
    SourceAuthentication(#[from] super::super::ModelPreparationSourceAuthenticationError),
    #[error("model preparation publication filesystem: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, PartialEq, Eq)]
struct PublicationSnapshot {
    preparation_receipt: Vec<u8>,
    profile: Vec<u8>,
    model_root_identity: file::Identity,
    receipts_identity: file::Identity,
}

trait PublicationBackend {
    fn reauthenticate(
        &mut self,
        converted: &ConvertedModelPreparation,
    ) -> Result<PublicationSnapshot, ModelPreparationPublicationError>;

    fn publish_preparation_receipt(
        &mut self,
        converted: &ConvertedModelPreparation,
        expected: &PublicationSnapshot,
    ) -> Result<(), ModelPreparationPublicationError>;

    fn publish_profile(
        &mut self,
        converted: &ConvertedModelPreparation,
        expected: &PublicationSnapshot,
    ) -> Result<(), ModelPreparationPublicationError>;

    fn finish(
        &mut self,
        converted: &ConvertedModelPreparation,
        expected: &PublicationSnapshot,
    ) -> Result<PreparedModelProfileV1, ModelPreparationPublicationError>;
}

struct FilesystemPublicationBackend;

/// Consume the exact converted pair and publish its pair receipt followed by
/// the per-model profile commit. V1 deliberately records only `keep`; safe
/// recipe-owned source deletion requires a separate durable journal.
pub fn publish_converted_model_preparation_keep(
    converted: ConvertedModelPreparation,
) -> Result<RegisteredModelPreparation, ModelPreparationPublicationError> {
    publish_with(converted, &mut FilesystemPublicationBackend)
}

fn publish_with(
    converted: ConvertedModelPreparation,
    backend: &mut impl PublicationBackend,
) -> Result<RegisteredModelPreparation, ModelPreparationPublicationError> {
    let expected = backend.reauthenticate(&converted)?;
    backend.publish_preparation_receipt(&converted, &expected)?;
    require_snapshot_eq(&expected, backend.reauthenticate(&converted)?)?;
    backend.publish_profile(&converted, &expected)?;
    require_snapshot_eq(&expected, backend.reauthenticate(&converted)?)?;
    let profile = backend.finish(&converted, &expected)?;
    Ok(RegisteredModelPreparation {
        profile,
        model_root: converted.model_root,
    })
}

impl PublicationBackend for FilesystemPublicationBackend {
    fn reauthenticate(
        &mut self,
        converted: &ConvertedModelPreparation,
    ) -> Result<PublicationSnapshot, ModelPreparationPublicationError> {
        authentication::reauthenticate_pair(converted)
    }

    fn publish_preparation_receipt(
        &mut self,
        converted: &ConvertedModelPreparation,
        expected: &PublicationSnapshot,
    ) -> Result<(), ModelPreparationPublicationError> {
        let receipts = converted.model_root.join("receipts");
        file::publish_exact_private_file(
            &receipts,
            expected.receipts_identity,
            PREPARATION_RECEIPT_NAME,
            PREPARATION_RECEIPT_PARTIAL,
            &expected.preparation_receipt,
            MAX_MODEL_PREPARATION_RECEIPT_BYTES,
        )?;
        Ok(())
    }

    fn publish_profile(
        &mut self,
        converted: &ConvertedModelPreparation,
        expected: &PublicationSnapshot,
    ) -> Result<(), ModelPreparationPublicationError> {
        file::publish_exact_private_file(
            &converted.model_root,
            expected.model_root_identity,
            PROFILE_NAME,
            PROFILE_PARTIAL,
            &expected.profile,
            MAX_PREPARED_MODEL_PROFILE_BYTES,
        )?;
        Ok(())
    }

    fn finish(
        &mut self,
        converted: &ConvertedModelPreparation,
        expected: &PublicationSnapshot,
    ) -> Result<PreparedModelProfileV1, ModelPreparationPublicationError> {
        authentication::require_final_inventory(converted)?;
        let receipt_bytes = file::read_exact_private_file(
            &converted.model_root.join("receipts"),
            expected.receipts_identity,
            PREPARATION_RECEIPT_NAME,
            &expected.preparation_receipt,
            MAX_MODEL_PREPARATION_RECEIPT_BYTES,
        )?;
        let profile_bytes = file::read_exact_private_file(
            &converted.model_root,
            expected.model_root_identity,
            PROFILE_NAME,
            &expected.profile,
            MAX_PREPARED_MODEL_PROFILE_BYTES,
        )?;
        require_snapshot_eq(
            expected,
            PublicationSnapshot {
                preparation_receipt: receipt_bytes.clone(),
                profile: profile_bytes.clone(),
                model_root_identity: expected.model_root_identity,
                receipts_identity: expected.receipts_identity,
            },
        )?;
        let profile = PreparedModelProfileV1::parse(&profile_bytes)?;
        profile.verify_preparation_receipt(&receipt_bytes)?;
        Ok(profile)
    }
}

fn require_restart_order(
    receipt_final: bool,
    receipt_partial: bool,
    profile_final: bool,
    profile_partial: bool,
) -> Result<(), ModelPreparationPublicationError> {
    publication_require(
        !(profile_final || profile_partial) || receipt_final,
        "prepared-model profile state exists before the pair receipt commit",
    )?;
    publication_require(
        !receipt_partial || !(profile_final || profile_partial),
        "pair-receipt residue exists after the profile commit",
    )
}

fn require_snapshot_eq(
    expected: &PublicationSnapshot,
    actual: PublicationSnapshot,
) -> Result<(), ModelPreparationPublicationError> {
    publication_require(
        expected == &actual,
        "prepared pair changed during durable registry publication",
    )
}

fn publication_require(
    condition: bool,
    reason: impl Into<String>,
) -> Result<(), ModelPreparationPublicationError> {
    if condition {
        Ok(())
    } else {
        Err(publication_error(reason).into())
    }
}

fn publication_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::PlanInvalid {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests;
