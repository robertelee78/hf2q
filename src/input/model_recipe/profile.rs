use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    embedded_qwen38_recipe, ModelPreparationError, ModelPreparationReceiptV2, RecipeArtifactRole,
    RecipeQuantization, SourceRetentionChoice,
};

const PROFILE_KIND: &str = "hf2q.prepared-model-profile";
const PROFILE_PACKAGE: &str = "hf2q";
const PREPARATION_RECEIPT_PATH: &str = "receipts/model-preparation.json";
const MAX_PROFILE_JSON_DEPTH: usize = 64;
pub const PREPARED_MODEL_PROFILE_SCHEMA_VERSION: u32 = 1;
pub const MAX_PREPARED_MODEL_PROFILE_BYTES: usize = 64 * 1024;

/// Canonical registry entry for one exact hf2q-produced model pair.
///
/// Parsing this record is structural only. In particular, it grants no source
/// deletion, calibration, model loading, serving, or preference authority.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreparedModelProfileV1 {
    kind: String,
    schema_version: u32,
    package: String,
    repository: ProfileRepository,
    recipe: ProfileRecipe,
    preparation_receipt: ProfilePreparationReceipt,
    source_retention: SourceRetentionChoice,
    state: PreparedModelState,
    artifacts: Vec<ProfileArtifact>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProfileRepository {
    id: String,
    revision: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProfileRecipe {
    id: String,
    sha256: String,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProfilePreparationReceipt {
    path: String,
    schema_version: u32,
    sha256: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PreparedModelState {
    AwaitingRuntimeCalibration,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProfileArtifact {
    role: RecipeArtifactRole,
    quantization: RecipeQuantization,
    path: String,
    size: u64,
    sha256: String,
    conversion_receipt_path: String,
    conversion_receipt_sha256: String,
}

impl PreparedModelProfileV1 {
    pub fn parse(bytes: &[u8]) -> Result<Self, ModelPreparationError> {
        require_bound(bytes)?;
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        profile_require(
            json_depth(&value) <= MAX_PROFILE_JSON_DEPTH,
            "prepared-model profile exceeds its JSON depth cap",
        )?;
        let profile: Self = serde_json::from_value(value)?;
        profile.validate()?;
        if profile.to_deterministic_json()? != bytes {
            return Err(ModelPreparationError::NonCanonical);
        }
        Ok(profile)
    }

    pub fn to_deterministic_json(&self) -> Result<Vec<u8>, ModelPreparationError> {
        let mut bytes = serde_json::to_vec(self)?;
        bytes.push(b'\n');
        require_bound(&bytes)?;
        Ok(bytes)
    }

    pub fn repository_id(&self) -> &str {
        &self.repository.id
    }

    pub fn revision(&self) -> &str {
        &self.repository.revision
    }

    pub fn recipe_id(&self) -> &str {
        &self.recipe.id
    }

    pub fn preparation_receipt_sha256(&self) -> &str {
        &self.preparation_receipt.sha256
    }

    pub fn source_retention(&self) -> SourceRetentionChoice {
        self.source_retention
    }

    pub(in crate::input) fn build_keep(
        receipt: &ModelPreparationReceiptV2,
        receipt_bytes: &[u8],
    ) -> Result<Self, ModelPreparationError> {
        let reparsed = ModelPreparationReceiptV2::parse(receipt_bytes)?;
        profile_require(
            &reparsed == receipt,
            "preparation receipt bytes differ from proof",
        )?;
        let recipe = embedded_qwen38_recipe()?;
        let artifacts = recipe
            .artifacts()
            .iter()
            .map(|artifact| {
                let filename = artifact.filename();
                let receipt_sha256 = receipt
                    .artifact_receipt_sha256(artifact.role())
                    .ok_or_else(|| profile_error("preparation receipt is missing an artifact"))?;
                Ok(ProfileArtifact {
                    role: artifact.role(),
                    quantization: artifact.quantization(),
                    path: format!("artifacts/{filename}"),
                    size: artifact.size(),
                    sha256: artifact.sha256().to_owned(),
                    conversion_receipt_path: format!("receipts/{filename}.receipt.json"),
                    conversion_receipt_sha256: receipt_sha256.to_owned(),
                })
            })
            .collect::<Result<Vec<_>, ModelPreparationError>>()?;
        let profile = Self {
            kind: PROFILE_KIND.to_owned(),
            schema_version: PREPARED_MODEL_PROFILE_SCHEMA_VERSION,
            package: PROFILE_PACKAGE.to_owned(),
            repository: ProfileRepository {
                id: receipt.repository_id().to_owned(),
                revision: receipt.revision().to_owned(),
            },
            recipe: ProfileRecipe {
                id: receipt.recipe_id().to_owned(),
                sha256: receipt.recipe_sha256().to_owned(),
            },
            preparation_receipt: ProfilePreparationReceipt {
                path: PREPARATION_RECEIPT_PATH.to_owned(),
                schema_version: super::MODEL_PREPARATION_RECEIPT_SCHEMA_VERSION,
                sha256: hex::encode(Sha256::digest(receipt_bytes)),
            },
            source_retention: SourceRetentionChoice::Keep,
            state: PreparedModelState::AwaitingRuntimeCalibration,
            artifacts,
        };
        profile.validate()?;
        profile.verify_preparation_receipt(receipt_bytes)?;
        Ok(profile)
    }

    pub(in crate::input) fn verify_preparation_receipt(
        &self,
        receipt_bytes: &[u8],
    ) -> Result<ModelPreparationReceiptV2, ModelPreparationError> {
        let receipt = ModelPreparationReceiptV2::parse(receipt_bytes)?;
        profile_require(
            hex::encode(Sha256::digest(receipt_bytes)) == self.preparation_receipt.sha256,
            "preparation receipt digest mismatch",
        )?;
        profile_require(
            receipt.repository_id() == self.repository.id
                && receipt.revision() == self.repository.revision
                && receipt.recipe_id() == self.recipe.id
                && receipt.recipe_sha256() == self.recipe.sha256,
            "preparation receipt identity mismatch",
        )?;
        for artifact in &self.artifacts {
            profile_require(
                receipt.artifact_receipt_sha256(artifact.role)
                    == Some(artifact.conversion_receipt_sha256.as_str()),
                "conversion receipt digest mismatch",
            )?;
        }
        Ok(receipt)
    }

    fn validate(&self) -> Result<(), ModelPreparationError> {
        let recipe = embedded_qwen38_recipe()?;
        let recipe_sha256 = recipe.recipe_sha256()?;
        profile_require(self.kind == PROFILE_KIND, "wrong kind")?;
        profile_require(
            self.schema_version == PREPARED_MODEL_PROFILE_SCHEMA_VERSION,
            "unsupported schema version",
        )?;
        profile_require(self.package == PROFILE_PACKAGE, "wrong package")?;
        profile_require(
            self.repository.id == recipe.source().repository_id()
                && self.repository.revision == recipe.source().revision(),
            "wrong repository identity",
        )?;
        profile_require(
            self.recipe.id == recipe.recipe_id() && self.recipe.sha256 == recipe_sha256,
            "wrong recipe identity",
        )?;
        profile_require(
            self.preparation_receipt.path == PREPARATION_RECEIPT_PATH
                && self.preparation_receipt.schema_version
                    == super::MODEL_PREPARATION_RECEIPT_SCHEMA_VERSION
                && valid_lower_hex(&self.preparation_receipt.sha256, 64),
            "invalid preparation receipt descriptor",
        )?;
        profile_require(
            self.source_retention == SourceRetentionChoice::Keep,
            "profile v1 supports only retained recipe-owned source",
        )?;
        profile_require(
            self.state == PreparedModelState::AwaitingRuntimeCalibration,
            "invalid prepared-model state",
        )?;
        profile_require(
            self.artifacts.len() == recipe.artifacts().len(),
            "wrong artifact count",
        )?;
        for (actual, expected) in self.artifacts.iter().zip(recipe.artifacts()) {
            let filename = expected.filename();
            profile_require(
                actual.role == expected.role()
                    && actual.quantization == expected.quantization()
                    && actual.path == format!("artifacts/{filename}")
                    && actual.size == expected.size()
                    && actual.sha256 == expected.sha256()
                    && actual.conversion_receipt_path
                        == format!("receipts/{filename}.receipt.json")
                    && valid_lower_hex(&actual.conversion_receipt_sha256, 64),
                "artifact descriptor mismatch",
            )?;
        }
        Ok(())
    }
}

fn json_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(values) => values
            .iter()
            .map(json_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        serde_json::Value::Object(values) => values
            .values()
            .map(json_depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
        _ => 1,
    }
}

fn require_bound(bytes: &[u8]) -> Result<(), ModelPreparationError> {
    if bytes.len() > MAX_PREPARED_MODEL_PROFILE_BYTES {
        Err(ModelPreparationError::TooLarge {
            actual: bytes.len(),
            limit: MAX_PREPARED_MODEL_PROFILE_BYTES,
        })
    } else {
        Ok(())
    }
}

fn valid_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn profile_require(
    condition: bool,
    reason: impl Into<String>,
) -> Result<(), ModelPreparationError> {
    if condition {
        Ok(())
    } else {
        Err(profile_error(reason))
    }
}

fn profile_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::PlanInvalid {
        reason: reason.into(),
    }
}
