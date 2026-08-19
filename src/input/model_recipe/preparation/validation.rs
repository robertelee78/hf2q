use std::path::Path;

use crate::convert::receipt::{ConversionReceipt, SourceFileReceipt};
use crate::input::hf_reference::HfModelReference;

use super::*;
use crate::input::model_recipe::recipe_for_reference;

impl ModelRecipe {
    pub(super) fn validate_conversion_receipt(
        &self,
        role: RecipeArtifactRole,
        artifact: &VerifiedRecipeArtifact,
        receipt: &ConversionReceipt,
    ) -> Result<(), ModelPreparationError> {
        conversion_require(
            receipt.schema_version == CONVERSION_RECEIPT_SCHEMA_VERSION,
            "unsupported conversion receipt schema",
        )?;
        let original = HfModelReference::parse(&receipt.source.original_reference, None)
            .map_err(|error| conversion_error(format!("invalid original reference: {error}")))?;
        let original_recipe = recipe_for_reference(&original)?
            .ok_or_else(|| conversion_error("original reference is not accepted by the recipe"))?;
        conversion_require(
            original_recipe.recipe_id() == self.recipe_id()
                && original_recipe.recipe_sha256()? == self.recipe_sha256()?,
            "original reference selects another recipe",
        )?;
        conversion_require(
            receipt.source.repository_id == self.source.repository_id()
                && receipt.source.repository_type == "model"
                && receipt.source.canonical_url
                    == format!("https://huggingface.co/{}", self.source.repository_id())
                && receipt.source.revision == self.source.revision()
                && receipt.source.filename.is_none()
                && receipt.source.bundle_sha256 == self.source.bundle_sha256(),
            "source identity differs from recipe",
        )?;
        conversion_require(
            receipt.source.files.len() == self.source.files.len(),
            "source file count differs from recipe",
        )?;
        for (actual, expected) in receipt.source.files.iter().zip(&self.source.files) {
            validate_source_file(actual, expected)?;
        }
        let expected = self
            .artifact(role)
            .ok_or_else(|| conversion_error("recipe artifact role is absent"))?;
        let artifact_path = artifact
            .path()
            .to_str()
            .ok_or_else(|| conversion_error("artifact path is not valid UTF-8"))?;
        conversion_require(
            receipt.output.path == artifact_path
                && Path::new(&receipt.output.path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    == Some(expected.filename())
                && receipt.output.size == expected.size()
                && receipt.output.sha256 == expected.sha256()
                && artifact.sha256() == expected.sha256(),
            "output identity differs from artifact proof or recipe",
        )?;
        conversion_require(
            receipt.quant_selector == expected.quantization().as_str(),
            "quantization differs from recipe",
        )?;
        conversion_require(
            receipt.converter.package == PREPARATION_PACKAGE
                && canonical_semver(&receipt.converter.version)
                && valid_lower_hex(&receipt.converter.git_commit, 40),
            "invalid converter identity",
        )?;
        conversion_require(
            receipt.excluded_dspark.tensor_count == 0
                && receipt.excluded_dspark.status == "none_detected",
            "Qwen preparation unexpectedly excluded tensors",
        )?;
        let (strategy, scope) = match role {
            RecipeArtifactRole::Text => ("row_aligned_tensor_chunks", "all_streamed_tensors"),
            RecipeArtifactRole::VisionProjector => (
                "lazy_source_index_projector_only",
                "multimodal_projector_tensors",
            ),
        };
        conversion_require(
            receipt.peak_chunk_bound.strategy == strategy
                && receipt.peak_chunk_bound.scope == scope,
            "conversion strategy differs from recipe role",
        )
    }
}

fn validate_source_file(
    actual: &SourceFileReceipt,
    expected: &crate::input::model_recipe::RecipeSourceFile,
) -> Result<(), ModelPreparationError> {
    conversion_require(
        actual.path == expected.path()
            && actual.size == expected.size()
            && actual.sha256 == expected.sha256()
            && actual.hf_lfs_sha256.as_deref() == expected.hf_lfs_sha256(),
        "source file identity differs from recipe",
    )
}
