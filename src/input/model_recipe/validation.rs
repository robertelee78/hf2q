use std::path::{Component, Path};

use crate::core::provenance::source_shard::{compute_source_bundle_sha256, SourceShard};

use super::{
    ModelRecipe, ModelRecipeError, RecipeArtifact, RecipeArtifactRole, RecipeHardwareProfile,
    RecipeQuantization, RecipeSource, RecipeSourceFile, RecipeStatus, SourceRetentionChoice,
    MODEL_RECIPE_SCHEMA_VERSION, QWEN38_ACCEPTED_REVISION, QWEN38_RECIPE_ID, QWEN38_REPOSITORY_ID,
    RECIPE_KIND,
};

impl ModelRecipe {
    pub(super) fn validate(&self) -> Result<(), ModelRecipeError> {
        require(self.kind == RECIPE_KIND, "wrong kind")?;
        require(
            self.schema_version == MODEL_RECIPE_SCHEMA_VERSION,
            "unsupported schema version",
        )?;
        require(self.recipe_id == QWEN38_RECIPE_ID, "unknown recipe id")?;
        require(
            self.status == RecipeStatus::Accepted,
            "recipe is not accepted",
        )?;
        require(
            self.conversion.producer_version == "hf2q 0.1.6",
            "invalid accepted artifact producer version",
        )?;
        require(
            self.acceptance.decision == "docs/adr/ADR-044-qwen38-native.md"
                && self.acceptance.accepted_at == "2026-08-17",
            "invalid acceptance evidence",
        )?;
        self.source.validate()?;
        validate_artifacts(&self.artifacts)?;
        validate_hardware_profiles(&self.hardware_profiles)?;
        self.validate_disk()?;
        require(
            self.source_retention.interactive_default == SourceRetentionChoice::Keep
                && self.source_retention.non_interactive_requires_explicit
                && self.source_retention.deletion_scope == "recipe_owned_source_only",
            "invalid source-retention policy",
        )
    }

    fn validate_disk(&self) -> Result<(), ModelRecipeError> {
        let source_bytes = checked_sum(self.source.files.iter().map(|file| file.size))?;
        let artifact_bytes = checked_sum(self.artifacts.iter().map(|artifact| artifact.size))?;
        let minimum = source_bytes
            .checked_add(artifact_bytes)
            .and_then(|value| value.checked_add(self.disk.safety_reserve_bytes))
            .ok_or_else(|| invalid("disk byte total overflow"))?;
        require(
            self.disk.source_bytes == source_bytes,
            "source byte total mismatch",
        )?;
        require(
            self.disk.artifact_bytes == artifact_bytes,
            "artifact byte total mismatch",
        )?;
        require(
            self.disk.safety_reserve_bytes == 8 * 1024 * 1024 * 1024,
            "unexpected safety reserve",
        )?;
        require(
            self.disk.minimum_free_bytes == minimum,
            "minimum free byte total mismatch",
        )
    }
}

impl RecipeSource {
    fn validate(&self) -> Result<(), ModelRecipeError> {
        require(
            self.repository_id == QWEN38_REPOSITORY_ID,
            "wrong repository id",
        )?;
        require(self.repository_type == "model", "wrong repository type")?;
        require(
            self.canonical_url == "https://huggingface.co/Qwen/Qwen3.8-27B",
            "wrong canonical URL",
        )?;
        require(
            self.revision == QWEN38_ACCEPTED_REVISION,
            "wrong source revision",
        )?;
        validate_sha256(&self.bundle_sha256, "source bundle sha256")?;
        require(
            self.files.len() == 29,
            "Qwen3.8 recipe must contain 29 files",
        )?;

        let mut prior = None;
        for file in &self.files {
            file.validate()?;
            if let Some(previous) = prior {
                require(
                    previous < file.path.as_str(),
                    "source files are not unique/sorted",
                )?;
            }
            prior = Some(file.path.as_str());
        }

        let source_shards = self
            .files
            .iter()
            .map(|file| SourceShard {
                filename: file.path.clone(),
                bytes: file.size,
                sha256: file.hf_lfs_sha256.clone(),
                hf_etag: file.hub_etag.clone(),
                is_lfs: file.hf_lfs_sha256.is_some(),
                verified_at_secs: 0,
            })
            .collect::<Vec<_>>();
        require(
            compute_source_bundle_sha256(&source_shards).as_deref()
                == Some(self.bundle_sha256.as_str()),
            "source bundle sha256 does not match LFS entries",
        )
    }
}

impl RecipeSourceFile {
    fn validate(&self) -> Result<(), ModelRecipeError> {
        let path = Path::new(&self.path);
        require(
            !self.path.is_empty()
                && self.path.len() <= crate::input::hf_reference::MAX_HF_FILENAME_BYTES
                && self.path.is_ascii()
                && !self.path.contains('\\')
                && path.components().count() == 1
                && path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "unsafe recipe source path",
        )?;
        require(self.size > 0, "recipe source file is empty")?;
        validate_sha256(&self.sha256, "recipe source sha256")?;
        if let Some(lfs) = &self.hf_lfs_sha256 {
            validate_sha256(lfs, "recipe LFS sha256")?;
            require(
                lfs == &self.sha256 && lfs == &self.hub_etag,
                "LFS identities differ",
            )?;
        } else {
            validate_lower_hex(&self.hub_etag, 40, "recipe Git blob sha1")?;
        }
        require(
            !self.path.ends_with(".safetensors") || self.hf_lfs_sha256.is_some(),
            "safetensors source lacks LFS sha256",
        )
    }
}

fn validate_artifacts(artifacts: &[RecipeArtifact]) -> Result<(), ModelRecipeError> {
    require(
        artifacts.len() == 2,
        "recipe must contain exactly two artifacts",
    )?;
    let expected = [
        (
            RecipeArtifactRole::Text,
            RecipeQuantization::Q4KM,
            "Qwen3.8-27B-Q4_K_M.gguf",
        ),
        (
            RecipeArtifactRole::VisionProjector,
            RecipeQuantization::F16Mmproj,
            "Qwen3.8-27B-mmproj-F16.gguf",
        ),
    ];
    for (artifact, (role, quantization, filename)) in artifacts.iter().zip(expected) {
        require(
            artifact.role == role
                && artifact.quantization == quantization
                && artifact.filename == filename,
            "invalid artifact role/quantization/name",
        )?;
        require(artifact.size > 0, "empty accepted artifact")?;
        validate_sha256(&artifact.sha256, "artifact sha256")?;
    }
    Ok(())
}

fn validate_hardware_profiles(profiles: &[RecipeHardwareProfile]) -> Result<(), ModelRecipeError> {
    require(
        profiles.len() == 1,
        "recipe must contain one proven hardware profile",
    )?;
    let profile = &profiles[0];
    require(
        profile.profile_id == "qwen38-m5-max-128g-q4-k-m-v1"
            && profile.target == "aarch64-apple-darwin"
            && profile.chip_model == "Apple M5 Max"
            && profile.minimum_unified_memory_bytes == 128 * 1024 * 1024 * 1024
            && profile.text_quantization == RecipeQuantization::Q4KM
            && profile.runtime_calibration_required,
        "invalid Qwen3.8 hardware profile",
    )
}

fn validate_sha256(value: &str, field: &str) -> Result<(), ModelRecipeError> {
    validate_lower_hex(value, 64, field)
}

fn validate_lower_hex(value: &str, length: usize, field: &str) -> Result<(), ModelRecipeError> {
    require(
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
        field,
    )
}

fn checked_sum(values: impl IntoIterator<Item = u64>) -> Result<u64, ModelRecipeError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| invalid("byte total overflow"))
    })
}

fn require(condition: bool, reason: &str) -> Result<(), ModelRecipeError> {
    if condition {
        Ok(())
    } else {
        Err(invalid(reason))
    }
}

fn invalid(reason: &str) -> ModelRecipeError {
    ModelRecipeError::Invalid {
        reason: reason.to_owned(),
    }
}
