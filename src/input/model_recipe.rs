//! Closed, checked-in preparation recipes for accepted Hugging Face models.
//!
//! A recipe is policy, not discovery. Production loads only bytes embedded in
//! the hf2q binary; a repository, URL, cache entry, or caller cannot supply a
//! replacement recipe. The first v1 recipe binds the exact ADR-044 Qwen3.8
//! source, paired text/projector artifacts, proven host profile, and peak disk
//! floor used by ADR-045's future no-options preparation coordinator. The
//! sibling preparation boundary additionally seals accepted host/disk facts
//! and both canonical conversion receipts into one inert text/projector pair.
//! The plan boundary derives the canonical no-options layout while retaining
//! the host proof, then accepts only a sealed exact Hub resolution from the
//! download module. The download module separately authenticates the complete
//! recipe-owned remote metadata set before transfer. None of these transitions
//! can transfer payloads or mutate that layout.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::input::hf_reference::HfModelReference;

mod plan;
mod preparation;
mod validation;
mod verification;

pub(in crate::input) use plan::canonical_future_directory;
#[cfg(test)]
pub(in crate::input) use plan::require_exact_regular_file_for_test;
pub use plan::{
    plan_current_model_preparation, ModelPreparationPlan, MAX_MODEL_PREPARATION_PATH_BYTES,
};
pub use preparation::{
    ModelPreparationError, ModelPreparationReceiptV1, VerifiedModelPreparation,
    VerifiedRecipeConversion, VerifiedRecipeHost, MAX_MODEL_PREPARATION_RECEIPT_BYTES,
    MODEL_PREPARATION_RECEIPT_SCHEMA_VERSION,
};
pub use verification::{VerifiedRecipeArtifact, VerifiedRecipeSource};

pub const MODEL_RECIPE_SCHEMA_VERSION: u32 = 1;
pub const MAX_MODEL_RECIPE_BYTES: usize = 64 * 1024;
pub const QWEN38_RECIPE_ID: &str = "qwen38-27b-official-v1";
pub const QWEN38_REPOSITORY_ID: &str = "Qwen/Qwen3.8-27B";
pub const QWEN38_ACCEPTED_REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";

const RECIPE_KIND: &str = "hf2q.model-preparation-recipe";
const QWEN38_RECIPE_BYTES: &[u8] =
    include_bytes!("../../data/model-recipes/qwen38-27b-official-v1.json");

#[derive(Debug, Error)]
pub enum ModelRecipeError {
    #[error("model recipe is {actual} bytes; limit is {limit}")]
    TooLarge { actual: usize, limit: usize },
    #[error("model recipe JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("model recipe is not in hf2q's canonical wire encoding")]
    NonCanonical,
    #[error("invalid model recipe: {reason}")]
    Invalid { reason: String },
    #[error(
        "{repo} revision `{requested}` is not accepted by recipe `{recipe}`; expected `{accepted}`"
    )]
    RevisionNotAccepted {
        repo: String,
        requested: String,
        recipe: String,
        accepted: String,
    },
    #[error(
        "no accepted preparation profile for target `{target}`, chip `{chip_model}`, and {total_memory_bytes} bytes unified memory"
    )]
    UnsupportedHardware {
        target: String,
        chip_model: String,
        total_memory_bytes: u64,
    },
    #[error(
        "recipe `{recipe}` requires at least {required_bytes} free bytes before preparation; found {available_bytes}"
    )]
    InsufficientDisk {
        recipe: String,
        required_bytes: u64,
        available_bytes: u64,
    },
    #[error("recipe source does not match: {reason}")]
    SourceMismatch { reason: String },
    #[error("recipe artifact does not match: {reason}")]
    ArtifactMismatch { reason: String },
    #[error("recipe verification I/O: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelRecipe {
    kind: String,
    schema_version: u32,
    recipe_id: String,
    status: RecipeStatus,
    conversion: RecipeConversion,
    acceptance: RecipeAcceptance,
    source: RecipeSource,
    artifacts: Vec<RecipeArtifact>,
    hardware_profiles: Vec<RecipeHardwareProfile>,
    disk: RecipeDisk,
    source_retention: RecipeSourceRetention,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RecipeConversion {
    producer_version: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecipeStatus {
    Accepted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RecipeAcceptance {
    decision: String,
    accepted_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecipeSource {
    repository_id: String,
    repository_type: String,
    canonical_url: String,
    revision: String,
    bundle_sha256: String,
    files: Vec<RecipeSourceFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecipeSourceFile {
    path: String,
    size: u64,
    sha256: String,
    hub_etag: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hf_lfs_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum RecipeArtifactRole {
    Text,
    VisionProjector,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecipeQuantization {
    #[serde(rename = "q4_k_m")]
    Q4KM,
    #[serde(rename = "f16-mmproj")]
    F16Mmproj,
}

impl RecipeQuantization {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q4KM => "q4_k_m",
            Self::F16Mmproj => "f16-mmproj",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecipeArtifact {
    role: RecipeArtifactRole,
    quantization: RecipeQuantization,
    filename: String,
    size: u64,
    sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecipeHardwareProfile {
    profile_id: String,
    target: String,
    chip_model: String,
    minimum_unified_memory_bytes: u64,
    text_quantization: RecipeQuantization,
    runtime_calibration_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RecipeDisk {
    source_bytes: u64,
    artifact_bytes: u64,
    safety_reserve_bytes: u64,
    minimum_free_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceRetentionChoice {
    Keep,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct RecipeSourceRetention {
    interactive_default: SourceRetentionChoice,
    non_interactive_requires_explicit: bool,
    deletion_scope: String,
}

impl ModelRecipe {
    pub fn parse(bytes: &[u8]) -> Result<Self, ModelRecipeError> {
        if bytes.len() > MAX_MODEL_RECIPE_BYTES {
            return Err(ModelRecipeError::TooLarge {
                actual: bytes.len(),
                limit: MAX_MODEL_RECIPE_BYTES,
            });
        }
        let recipe: Self = serde_json::from_slice(bytes)?;
        recipe.validate()?;
        let mut canonical = serde_json::to_vec(&recipe)?;
        canonical.push(b'\n');
        if canonical != bytes {
            return Err(ModelRecipeError::NonCanonical);
        }
        Ok(recipe)
    }

    pub fn recipe_id(&self) -> &str {
        &self.recipe_id
    }

    pub fn source(&self) -> &RecipeSource {
        &self.source
    }

    pub(in crate::input) fn producer_version(&self) -> &str {
        &self.conversion.producer_version
    }

    pub fn artifacts(&self) -> &[RecipeArtifact] {
        &self.artifacts
    }

    pub fn hardware_profiles(&self) -> &[RecipeHardwareProfile] {
        &self.hardware_profiles
    }

    pub fn minimum_free_bytes(&self) -> u64 {
        self.disk.minimum_free_bytes
    }

    pub fn interactive_retention_default(&self) -> SourceRetentionChoice {
        self.source_retention.interactive_default
    }

    pub fn non_interactive_retention_requires_explicit(&self) -> bool {
        self.source_retention.non_interactive_requires_explicit
    }

    pub fn recipe_sha256(&self) -> Result<String, ModelRecipeError> {
        let mut bytes = serde_json::to_vec(self)?;
        bytes.push(b'\n');
        Ok(hex::encode(Sha256::digest(bytes)))
    }

    pub fn artifact(&self, role: RecipeArtifactRole) -> Option<&RecipeArtifact> {
        self.artifacts.iter().find(|artifact| artifact.role == role)
    }

    pub fn select_hardware_profile(
        &self,
        target: &str,
        chip_model: &str,
        total_memory_bytes: u64,
    ) -> Result<&RecipeHardwareProfile, ModelRecipeError> {
        self.hardware_profiles
            .iter()
            .find(|profile| {
                profile.target == target
                    && profile.chip_model == chip_model
                    && total_memory_bytes >= profile.minimum_unified_memory_bytes
            })
            .ok_or_else(|| ModelRecipeError::UnsupportedHardware {
                target: target.to_owned(),
                chip_model: chip_model.to_owned(),
                total_memory_bytes,
            })
    }

    pub fn require_free_space(&self, available_bytes: u64) -> Result<(), ModelRecipeError> {
        if available_bytes < self.disk.minimum_free_bytes {
            return Err(ModelRecipeError::InsufficientDisk {
                recipe: self.recipe_id.clone(),
                required_bytes: self.disk.minimum_free_bytes,
                available_bytes,
            });
        }
        Ok(())
    }
}

impl RecipeSource {
    pub fn repository_id(&self) -> &str {
        &self.repository_id
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    pub fn bundle_sha256(&self) -> &str {
        &self.bundle_sha256
    }

    pub fn files(&self) -> &[RecipeSourceFile] {
        &self.files
    }
}

impl RecipeSourceFile {
    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn hub_etag(&self) -> &str {
        &self.hub_etag
    }

    pub fn hf_lfs_sha256(&self) -> Option<&str> {
        self.hf_lfs_sha256.as_deref()
    }
}

impl RecipeArtifact {
    pub fn role(&self) -> RecipeArtifactRole {
        self.role
    }

    pub fn quantization(&self) -> RecipeQuantization {
        self.quantization
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

impl RecipeHardwareProfile {
    pub fn profile_id(&self) -> &str {
        &self.profile_id
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn chip_model(&self) -> &str {
        &self.chip_model
    }

    pub fn minimum_unified_memory_bytes(&self) -> u64 {
        self.minimum_unified_memory_bytes
    }

    pub fn text_quantization(&self) -> RecipeQuantization {
        self.text_quantization
    }

    pub fn runtime_calibration_required(&self) -> bool {
        self.runtime_calibration_required
    }
}

pub fn embedded_qwen38_recipe() -> Result<ModelRecipe, ModelRecipeError> {
    ModelRecipe::parse(QWEN38_RECIPE_BYTES)
}

pub fn recipe_for_reference(
    reference: &HfModelReference,
) -> Result<Option<ModelRecipe>, ModelRecipeError> {
    if reference.repo_id() != QWEN38_REPOSITORY_ID || reference.filename().is_some() {
        return Ok(None);
    }
    if let Some(requested) = reference.requested_revision() {
        if requested != QWEN38_ACCEPTED_REVISION {
            return Err(ModelRecipeError::RevisionNotAccepted {
                repo: reference.repo_id().to_owned(),
                requested: requested.to_owned(),
                recipe: QWEN38_RECIPE_ID.to_owned(),
                accepted: QWEN38_ACCEPTED_REVISION.to_owned(),
            });
        }
    }
    embedded_qwen38_recipe().map(Some)
}
