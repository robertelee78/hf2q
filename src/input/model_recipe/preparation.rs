use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::convert::receipt::{ConversionReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION};

use super::{
    embedded_qwen38_recipe, ModelRecipe, ModelRecipeError, RecipeArtifactRole, RecipeQuantization,
    VerifiedRecipeArtifact, VerifiedRecipeSource,
};

mod host;
mod validation;

const PREPARATION_KIND: &str = "hf2q.model-preparation-receipt";
const PREPARATION_PACKAGE: &str = "hf2q";
pub const MODEL_PREPARATION_RECEIPT_SCHEMA_VERSION: u32 = 2;
pub const MAX_MODEL_PREPARATION_RECEIPT_BYTES: usize = 64 * 1024;
pub(in crate::input) const MAX_CONVERSION_RECEIPT_BYTES: usize = 64 * 1024;

#[derive(Debug, Error)]
pub enum ModelPreparationError {
    #[error("model preparation receipt is {actual} bytes; limit is {limit}")]
    TooLarge { actual: usize, limit: usize },
    #[error("model preparation JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("model preparation receipt is not in hf2q's canonical wire encoding")]
    NonCanonical,
    #[error("conversion receipt does not match the accepted recipe: {reason}")]
    ConversionMismatch { reason: String },
    #[error("model preparation pair does not match: {reason}")]
    PairMismatch { reason: String },
    #[error("model preparation host preflight failed: {reason}")]
    HostProbe { reason: String },
    #[error("model preparation plan is invalid: {reason}")]
    PlanInvalid { reason: String },
    #[error("model recipe: {0}")]
    Recipe(#[from] ModelRecipeError),
}

/// One recipe-bound conversion and its exact canonical conversion receipt.
/// It remains inert and grants no serving, registration, or deletion authority.
#[derive(Debug)]
pub struct VerifiedRecipeConversion {
    role: RecipeArtifactRole,
    recipe_id: String,
    recipe_sha256: String,
    artifact: VerifiedRecipeArtifact,
    receipt: ConversionReceipt,
    receipt_sha256: String,
}

/// Exact recipe policy selected from in-process OS host and free-space reads.
/// It is preparation policy evidence, not a serving or filesystem capability.
/// Production construction accepts no caller-provided hardware facts.
/// The live available-space observation is deliberately ephemeral: receipt v2
/// records only the stable recipe-owned required floor after this proof passes.
#[derive(Debug)]
pub struct VerifiedRecipeHost {
    recipe_id: String,
    recipe_sha256: String,
    profile_id: String,
    target: String,
    chip_model: String,
    minimum_unified_memory_bytes: u64,
    observed_unified_memory_bytes: u64,
    preflight_available_bytes: u64,
}

impl VerifiedRecipeConversion {
    pub fn role(&self) -> RecipeArtifactRole {
        self.role
    }

    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub fn artifact(&self) -> &VerifiedRecipeArtifact {
        &self.artifact
    }
}

/// Canonical, durable description of one complete hf2q-produced recipe pair.
/// Parsing this record is structural only and never creates a capability.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelPreparationReceiptV2 {
    kind: String,
    schema_version: u32,
    package: String,
    recipe: PreparationRecipe,
    source: PreparationSource,
    hardware_profile: PreparationHardwareProfile,
    converter: PreparationConverter,
    state: PreparationState,
    artifacts: Vec<PreparationArtifact>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PreparationRecipe {
    id: String,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PreparationSource {
    repository_id: String,
    revision: String,
    bundle_sha256: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PreparationHardwareProfile {
    id: String,
    target: String,
    chip_model: String,
    minimum_unified_memory_bytes: u64,
    observed_unified_memory_bytes: u64,
    preflight_required_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PreparationConverter {
    package: String,
    version: String,
    git_commit: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PreparationState {
    AwaitingRuntimeCalibration,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PreparationArtifact {
    role: RecipeArtifactRole,
    quantization: RecipeQuantization,
    filename: String,
    size: u64,
    sha256: String,
    conversion_receipt_sha256: String,
}

impl ModelPreparationReceiptV2 {
    pub fn parse(bytes: &[u8]) -> Result<Self, ModelPreparationError> {
        require_bound(bytes, MAX_MODEL_PREPARATION_RECEIPT_BYTES)?;
        let receipt: Self = serde_json::from_slice(bytes)?;
        receipt.validate()?;
        if receipt.to_deterministic_json()? != bytes {
            return Err(ModelPreparationError::NonCanonical);
        }
        Ok(receipt)
    }

    pub fn to_deterministic_json(&self) -> Result<Vec<u8>, ModelPreparationError> {
        let mut bytes = serde_json::to_vec(self)?;
        bytes.push(b'\n');
        require_bound(&bytes, MAX_MODEL_PREPARATION_RECEIPT_BYTES)?;
        Ok(bytes)
    }

    pub fn recipe_id(&self) -> &str {
        &self.recipe.id
    }

    pub fn recipe_sha256(&self) -> &str {
        &self.recipe.sha256
    }

    pub fn repository_id(&self) -> &str {
        &self.source.repository_id
    }

    pub fn revision(&self) -> &str {
        &self.source.revision
    }

    pub fn hardware_profile_id(&self) -> &str {
        &self.hardware_profile.id
    }

    pub fn artifact_receipt_sha256(&self, role: RecipeArtifactRole) -> Option<&str> {
        self.artifacts
            .iter()
            .find(|artifact| artifact.role == role)
            .map(|artifact| artifact.conversion_receipt_sha256.as_str())
    }

    fn validate(&self) -> Result<(), ModelPreparationError> {
        let recipe = embedded_qwen38_recipe()?;
        let recipe_sha256 = recipe.recipe_sha256()?;
        pair_require(self.kind == PREPARATION_KIND, "wrong kind")?;
        pair_require(
            self.schema_version == MODEL_PREPARATION_RECEIPT_SCHEMA_VERSION,
            "unsupported schema version",
        )?;
        pair_require(self.package == PREPARATION_PACKAGE, "wrong package")?;
        pair_require(
            self.recipe.id == recipe.recipe_id() && self.recipe.sha256 == recipe_sha256,
            "wrong recipe identity",
        )?;
        pair_require(
            self.source.repository_id == recipe.source().repository_id()
                && self.source.revision == recipe.source().revision()
                && self.source.bundle_sha256 == recipe.source().bundle_sha256(),
            "wrong source identity",
        )?;
        let profile = recipe
            .hardware_profiles()
            .first()
            .ok_or_else(|| pair_error("recipe has no hardware profile"))?;
        pair_require(
            self.hardware_profile.id == profile.profile_id()
                && self.hardware_profile.target == profile.target()
                && self.hardware_profile.chip_model == profile.chip_model()
                && self.hardware_profile.minimum_unified_memory_bytes
                    == profile.minimum_unified_memory_bytes()
                && self.hardware_profile.observed_unified_memory_bytes
                    >= profile.minimum_unified_memory_bytes()
                && self.hardware_profile.preflight_required_bytes == recipe.minimum_free_bytes(),
            "wrong hardware profile",
        )?;
        pair_require(
            self.converter.package == PREPARATION_PACKAGE
                && canonical_semver(&self.converter.version)
                && valid_lower_hex(&self.converter.git_commit, 40),
            "invalid converter identity",
        )?;
        pair_require(
            self.state == PreparationState::AwaitingRuntimeCalibration,
            "invalid preparation state",
        )?;
        pair_require(self.artifacts.len() == 2, "pair must contain two artifacts")?;
        for expected in recipe.artifacts() {
            let actual = self
                .artifacts
                .iter()
                .find(|artifact| artifact.role == expected.role())
                .ok_or_else(|| pair_error("pair is missing an artifact role"))?;
            pair_require(
                actual.quantization == expected.quantization()
                    && actual.filename == expected.filename()
                    && actual.size == expected.size()
                    && actual.sha256 == expected.sha256()
                    && valid_lower_hex(&actual.conversion_receipt_sha256, 64),
                "artifact identity mismatch",
            )?;
        }
        pair_require(
            self.artifacts[0].role == RecipeArtifactRole::Text
                && self.artifacts[1].role == RecipeArtifactRole::VisionProjector,
            "artifacts are not in canonical role order",
        )
    }
}

/// Sealed inert pair. A later calibration/registration coordinator must consume
/// this exact value; receipt parsing alone cannot recreate it.
#[derive(Debug)]
pub struct VerifiedModelPreparation {
    receipt: ModelPreparationReceiptV2,
    receipt_bytes: Vec<u8>,
    source: VerifiedRecipeSource,
    text: VerifiedRecipeConversion,
    projector: VerifiedRecipeConversion,
}

impl VerifiedModelPreparation {
    pub fn receipt(&self) -> &ModelPreparationReceiptV2 {
        &self.receipt
    }

    pub fn receipt_bytes(&self) -> &[u8] {
        &self.receipt_bytes
    }

    pub fn text_artifact(&self) -> &VerifiedRecipeArtifact {
        self.text.artifact()
    }

    pub fn projector_artifact(&self) -> &VerifiedRecipeArtifact {
        self.projector.artifact()
    }

    pub fn source(&self) -> &VerifiedRecipeSource {
        &self.source
    }
}

impl ModelRecipe {
    fn verify_host_and_disk_facts(
        &self,
        target: &str,
        chip_model: &str,
        total_unified_memory_bytes: u64,
        available_bytes: u64,
    ) -> Result<VerifiedRecipeHost, ModelPreparationError> {
        let profile =
            self.select_hardware_profile(target, chip_model, total_unified_memory_bytes)?;
        self.require_free_space(available_bytes)?;
        Ok(VerifiedRecipeHost {
            recipe_id: self.recipe_id.clone(),
            recipe_sha256: self.recipe_sha256()?,
            profile_id: profile.profile_id().to_owned(),
            target: profile.target().to_owned(),
            chip_model: profile.chip_model().to_owned(),
            minimum_unified_memory_bytes: profile.minimum_unified_memory_bytes(),
            observed_unified_memory_bytes: total_unified_memory_bytes,
            preflight_available_bytes: available_bytes,
        })
    }

    pub fn verify_conversion_receipt(
        &self,
        role: RecipeArtifactRole,
        artifact: VerifiedRecipeArtifact,
        receipt_bytes: &[u8],
    ) -> Result<VerifiedRecipeConversion, ModelPreparationError> {
        if receipt_bytes.len() > MAX_CONVERSION_RECEIPT_BYTES {
            return Err(ModelPreparationError::TooLarge {
                actual: receipt_bytes.len(),
                limit: MAX_CONVERSION_RECEIPT_BYTES,
            });
        }
        let receipt: ConversionReceipt = serde_json::from_slice(receipt_bytes)?;
        let mut canonical = serde_json::to_vec_pretty(&receipt)?;
        canonical.push(b'\n');
        conversion_require(canonical == receipt_bytes, "noncanonical receipt bytes")?;
        conversion_require(
            artifact.role() == role
                && artifact.recipe_id() == self.recipe_id()
                && artifact.recipe_sha256() == self.recipe_sha256()?,
            "artifact proof belongs to another recipe or role",
        )?;
        self.validate_conversion_receipt(role, &artifact, &receipt)?;
        Ok(VerifiedRecipeConversion {
            role,
            recipe_id: self.recipe_id.clone(),
            recipe_sha256: self.recipe_sha256()?,
            artifact,
            receipt,
            receipt_sha256: hex::encode(Sha256::digest(receipt_bytes)),
        })
    }

    #[cfg(test)]
    pub(in crate::input) fn verified_conversion_at_for_test(
        &self,
        role: RecipeArtifactRole,
        artifact_path: &std::path::Path,
        converter_git_commit: &str,
    ) -> VerifiedRecipeConversion {
        use crate::convert::receipt::{
            ConverterReceipt, ExcludedDsparkReceipt, OutputReceipt, SourceFileReceipt,
            SourceReceipt,
        };

        let expected = self.artifact(role).expect("recipe artifact role");
        let (strategy, scope) = match role {
            RecipeArtifactRole::Text => ("row_aligned_tensor_chunks", "all_streamed_tensors"),
            RecipeArtifactRole::VisionProjector => (
                "lazy_source_index_projector_only",
                "multimodal_projector_tensors",
            ),
        };
        let receipt = ConversionReceipt {
            schema_version: CONVERSION_RECEIPT_SCHEMA_VERSION,
            source: SourceReceipt {
                original_reference: self.source().repository_id().to_owned(),
                repository_id: self.source().repository_id().to_owned(),
                repository_type: "model".to_owned(),
                canonical_url: format!("https://huggingface.co/{}", self.source().repository_id()),
                revision: self.source().revision().to_owned(),
                filename: None,
                bundle_sha256: self.source().bundle_sha256().to_owned(),
                files: self
                    .source()
                    .files()
                    .iter()
                    .map(|file| SourceFileReceipt {
                        path: file.path().to_owned(),
                        size: file.size(),
                        sha256: file.sha256().to_owned(),
                        hf_lfs_sha256: file.hf_lfs_sha256().map(str::to_owned),
                    })
                    .collect(),
            },
            converter: ConverterReceipt {
                package: "hf2q".to_owned(),
                version: env!("CARGO_PKG_VERSION").to_owned(),
                git_commit: converter_git_commit.to_owned(),
            },
            quant_selector: expected.quantization().as_str().to_owned(),
            output: OutputReceipt {
                path: artifact_path.display().to_string(),
                size: expected.size(),
                sha256: expected.sha256().to_owned(),
            },
            excluded_dspark: ExcludedDsparkReceipt {
                tensor_count: 0,
                status: "none_detected".to_owned(),
            },
            peak_chunk_bound: crate::convert::receipt::PeakChunkBoundReceipt {
                strategy: strategy.to_owned(),
                scope: scope.to_owned(),
                ..Default::default()
            },
        };
        let mut receipt_bytes = serde_json::to_vec_pretty(&receipt).expect("test receipt JSON");
        receipt_bytes.push(b'\n');
        let artifact = self.verified_artifact_for_test(role, artifact_path.to_path_buf());
        self.verify_conversion_receipt(role, artifact, &receipt_bytes)
            .expect("test conversion proof")
    }

    pub fn bind_prepared_pair(
        &self,
        source: VerifiedRecipeSource,
        host: VerifiedRecipeHost,
        text: VerifiedRecipeConversion,
        projector: VerifiedRecipeConversion,
    ) -> Result<VerifiedModelPreparation, ModelPreparationError> {
        let recipe_sha256 = self.recipe_sha256()?;
        pair_require(
            source.recipe_id() == self.recipe_id() && source.recipe_sha256() == recipe_sha256,
            "source proof belongs to another recipe",
        )?;
        for conversion in [&text, &projector] {
            pair_require(
                conversion.recipe_id == self.recipe_id()
                    && conversion.recipe_sha256 == recipe_sha256,
                "conversion proof belongs to another recipe",
            )?;
        }
        pair_require(
            text.role == RecipeArtifactRole::Text
                && projector.role == RecipeArtifactRole::VisionProjector,
            "conversion roles do not form a text/projector pair",
        )?;
        pair_require(
            text.receipt.converter == projector.receipt.converter,
            "text and projector converter identities differ",
        )?;
        pair_require(
            host.recipe_id == self.recipe_id()
                && host.recipe_sha256 == recipe_sha256
                && host.observed_unified_memory_bytes >= host.minimum_unified_memory_bytes
                && host.preflight_available_bytes >= self.minimum_free_bytes(),
            "host proof belongs to another recipe or is below policy floors",
        )?;

        let receipt = ModelPreparationReceiptV2 {
            kind: PREPARATION_KIND.to_owned(),
            schema_version: MODEL_PREPARATION_RECEIPT_SCHEMA_VERSION,
            package: PREPARATION_PACKAGE.to_owned(),
            recipe: PreparationRecipe {
                id: self.recipe_id.clone(),
                sha256: recipe_sha256,
            },
            source: PreparationSource {
                repository_id: self.source.repository_id.clone(),
                revision: self.source.revision.clone(),
                bundle_sha256: self.source.bundle_sha256.clone(),
            },
            hardware_profile: PreparationHardwareProfile {
                id: host.profile_id,
                target: host.target,
                chip_model: host.chip_model,
                minimum_unified_memory_bytes: host.minimum_unified_memory_bytes,
                observed_unified_memory_bytes: host.observed_unified_memory_bytes,
                preflight_required_bytes: self.minimum_free_bytes(),
            },
            converter: PreparationConverter {
                package: text.receipt.converter.package.clone(),
                version: text.receipt.converter.version.clone(),
                git_commit: text.receipt.converter.git_commit.clone(),
            },
            state: PreparationState::AwaitingRuntimeCalibration,
            artifacts: vec![
                preparation_artifact(self, &text)?,
                preparation_artifact(self, &projector)?,
            ],
        };
        receipt.validate()?;
        let receipt_bytes = receipt.to_deterministic_json()?;
        Ok(VerifiedModelPreparation {
            receipt,
            receipt_bytes,
            source,
            text,
            projector,
        })
    }
}

fn preparation_artifact(
    recipe: &ModelRecipe,
    conversion: &VerifiedRecipeConversion,
) -> Result<PreparationArtifact, ModelPreparationError> {
    let artifact = recipe
        .artifact(conversion.role)
        .ok_or_else(|| pair_error("recipe artifact role is absent"))?;
    Ok(PreparationArtifact {
        role: artifact.role(),
        quantization: artifact.quantization(),
        filename: artifact.filename().to_owned(),
        size: artifact.size(),
        sha256: artifact.sha256().to_owned(),
        conversion_receipt_sha256: conversion.receipt_sha256.clone(),
    })
}

fn valid_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn canonical_semver(value: &str) -> bool {
    semver::Version::parse(value)
        .map(|parsed| parsed.to_string() == value)
        .unwrap_or(false)
}

fn require_bound(bytes: &[u8], limit: usize) -> Result<(), ModelPreparationError> {
    if bytes.len() > limit {
        Err(ModelPreparationError::TooLarge {
            actual: bytes.len(),
            limit,
        })
    } else {
        Ok(())
    }
}

fn conversion_require(
    condition: bool,
    reason: impl Into<String>,
) -> Result<(), ModelPreparationError> {
    if condition {
        Ok(())
    } else {
        Err(conversion_error(reason))
    }
}

fn conversion_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::ConversionMismatch {
        reason: reason.into(),
    }
}

fn pair_require(condition: bool, reason: impl Into<String>) -> Result<(), ModelPreparationError> {
    if condition {
        Ok(())
    } else {
        Err(pair_error(reason))
    }
}

fn pair_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::PairMismatch {
        reason: reason.into(),
    }
}
