use std::ffi::OsString;
use std::fmt;
use std::path::{Component, Path, PathBuf};

use super::{
    recipe_for_reference, ModelPreparationError, ModelRecipe, RecipeArtifactRole, RecipeSourceFile,
    SourceRetentionChoice, VerifiedRecipeHost,
};
use crate::input::hf_reference::{HfModelReference, ResolvedHfModelReference};

pub const MAX_MODEL_PREPARATION_PATH_BYTES: usize = 4096;
const MAX_MODEL_PREPARATION_PATH_COMPONENTS: usize = 64;
const MAX_MODEL_PREPARATION_COMPONENT_BYTES: usize = 255;

/// One closed, host-checked layout for the no-options official-source path.
///
/// This value is inert: it owns the measured host proof and canonical paths,
/// but grants no download, conversion, deletion, registration, calibration,
/// serving, or filesystem-mutation authority. It is deliberately non-Clone.
pub struct ModelPreparationPlan {
    recipe: ModelRecipe,
    reference: HfModelReference,
    _host: VerifiedRecipeHost,
    accepted_revision: String,
    models_root: PathBuf,
    model_root: PathBuf,
    source_root: PathBuf,
    artifacts_root: PathBuf,
    receipts_root: PathBuf,
    text_artifact: PathBuf,
    projector_artifact: PathBuf,
    text_receipt: PathBuf,
    projector_receipt: PathBuf,
    preparation_receipt: PathBuf,
    profile: PathBuf,
}

impl fmt::Debug for ModelPreparationPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelPreparationPlan")
            .field("recipe_id", &self.recipe.recipe_id())
            .field("repository_id", &self.reference.repo_id())
            .field("accepted_revision", &self.accepted_revision)
            .field("paths", &"[redacted]")
            .finish()
    }
}

/// Select the one checked-in recipe, bind it to the current host and selected
/// filesystem, and derive the canonical future layout without creating it.
pub fn plan_current_model_preparation(
    reference: HfModelReference,
    models_root: &Path,
) -> Result<ModelPreparationPlan, ModelPreparationError> {
    let recipe = recipe_for_reference(&reference)?
        .ok_or_else(|| plan_error("reference has no accepted preparation recipe"))?;
    let layout = PreparationLayout::derive(&recipe, models_root)?;
    let host = recipe.verify_current_host_and_disk(&layout.model_root)?;
    Ok(ModelPreparationPlan::new(recipe, reference, host, layout))
}

impl ModelPreparationPlan {
    fn new(
        recipe: ModelRecipe,
        reference: HfModelReference,
        host: VerifiedRecipeHost,
        layout: PreparationLayout,
    ) -> Self {
        Self {
            accepted_revision: recipe.source().revision().to_owned(),
            recipe,
            reference,
            _host: host,
            models_root: layout.models_root,
            model_root: layout.model_root,
            source_root: layout.source_root,
            artifacts_root: layout.artifacts_root,
            receipts_root: layout.receipts_root,
            text_artifact: layout.text_artifact,
            projector_artifact: layout.projector_artifact,
            text_receipt: layout.text_receipt,
            projector_receipt: layout.projector_receipt,
            preparation_receipt: layout.preparation_receipt,
            profile: layout.profile,
        }
    }

    #[cfg(test)]
    pub(in crate::input) fn for_test(
        reference: HfModelReference,
        models_root: &Path,
        target: &str,
        chip_model: &str,
        total_unified_memory_bytes: u64,
        available_bytes: u64,
    ) -> Result<Self, ModelPreparationError> {
        let recipe = recipe_for_reference(&reference)?
            .ok_or_else(|| plan_error("reference has no accepted preparation recipe"))?;
        let layout = PreparationLayout::derive(&recipe, models_root)?;
        let host = recipe.verify_host_and_disk(
            target,
            chip_model,
            total_unified_memory_bytes,
            available_bytes,
        )?;
        Ok(Self::new(recipe, reference, host, layout))
    }

    pub fn recipe_id(&self) -> &str {
        self.recipe.recipe_id()
    }

    pub fn reference(&self) -> &HfModelReference {
        &self.reference
    }

    pub fn accepted_revision(&self) -> &str {
        &self.accepted_revision
    }

    pub fn models_root(&self) -> &Path {
        &self.models_root
    }

    pub fn model_root(&self) -> &Path {
        &self.model_root
    }

    pub fn source_root(&self) -> &Path {
        &self.source_root
    }

    pub fn artifacts_root(&self) -> &Path {
        &self.artifacts_root
    }

    pub fn receipts_root(&self) -> &Path {
        &self.receipts_root
    }

    pub fn artifact_path(&self, role: RecipeArtifactRole) -> &Path {
        match role {
            RecipeArtifactRole::Text => &self.text_artifact,
            RecipeArtifactRole::VisionProjector => &self.projector_artifact,
        }
    }

    pub fn conversion_receipt_path(&self, role: RecipeArtifactRole) -> &Path {
        match role {
            RecipeArtifactRole::Text => &self.text_receipt,
            RecipeArtifactRole::VisionProjector => &self.projector_receipt,
        }
    }

    pub fn preparation_receipt_path(&self) -> &Path {
        &self.preparation_receipt
    }

    pub fn profile_path(&self) -> &Path {
        &self.profile
    }

    pub fn source_retention_default(&self) -> SourceRetentionChoice {
        self.recipe.interactive_retention_default()
    }

    pub fn minimum_free_bytes(&self) -> u64 {
        self.recipe.minimum_free_bytes()
    }

    pub(in crate::input) fn validate_resolution<F>(
        &self,
        resolved: &ResolvedHfModelReference,
        contains: F,
    ) -> Result<(), ModelPreparationError>
    where
        F: Fn(&str) -> bool,
    {
        let same_identity = resolved.original() == self.reference.original()
            && resolved.repo_id() == self.reference.repo_id()
            && resolved.canonical_url() == self.reference.canonical_url()
            && resolved.filename().is_none()
            && resolved.revision() == self.accepted_revision;
        if !same_identity {
            return Err(plan_error(
                "Hub resolution does not match the planned reference and accepted revision",
            ));
        }
        if let Some(missing) = self
            .recipe
            .source()
            .files()
            .iter()
            .find(|file| !contains(file.path()))
        {
            return Err(plan_error(format!(
                "resolved repository is missing recipe source `{}`",
                missing.path()
            )));
        }
        Ok(())
    }

    pub(in crate::input) fn expected_source_files(&self) -> &[RecipeSourceFile] {
        self.recipe.source().files()
    }

    #[cfg(test)]
    pub(in crate::input) fn host_for_test(&self) -> &VerifiedRecipeHost {
        &self._host
    }
}

struct PreparationLayout {
    models_root: PathBuf,
    model_root: PathBuf,
    source_root: PathBuf,
    artifacts_root: PathBuf,
    receipts_root: PathBuf,
    text_artifact: PathBuf,
    projector_artifact: PathBuf,
    text_receipt: PathBuf,
    projector_receipt: PathBuf,
    preparation_receipt: PathBuf,
    profile: PathBuf,
}

impl PreparationLayout {
    fn derive(recipe: &ModelRecipe, models_root: &Path) -> Result<Self, ModelPreparationError> {
        let models_root = canonical_future_directory(models_root)?;
        let mut repository = recipe.source().repository_id().split('/');
        let owner = repository
            .next()
            .ok_or_else(|| plan_error("repository owner is absent"))?;
        let model = repository
            .next()
            .ok_or_else(|| plan_error("repository model is absent"))?;
        if repository.next().is_some() {
            return Err(plan_error("repository identity is not two components"));
        }

        let model_root = models_root
            .join("huggingface")
            .join(owner)
            .join(model)
            .join(recipe.source().revision());
        validate_path(&model_root)?;
        let source_root = model_root.join("source");
        let artifacts_root = model_root.join("artifacts");
        let receipts_root = model_root.join("receipts");
        let text = recipe
            .artifact(RecipeArtifactRole::Text)
            .ok_or_else(|| plan_error("text artifact is absent"))?;
        let projector = recipe
            .artifact(RecipeArtifactRole::VisionProjector)
            .ok_or_else(|| plan_error("projector artifact is absent"))?;
        let text_artifact = artifacts_root.join(text.filename());
        let projector_artifact = artifacts_root.join(projector.filename());
        let text_receipt = receipts_root.join(format!("{}.receipt.json", text.filename()));
        let projector_receipt =
            receipts_root.join(format!("{}.receipt.json", projector.filename()));
        let preparation_receipt = receipts_root.join("model-preparation.json");
        let profile = model_root.join("profile.json");
        for path in [
            &source_root,
            &artifacts_root,
            &receipts_root,
            &text_artifact,
            &projector_artifact,
            &text_receipt,
            &projector_receipt,
            &preparation_receipt,
            &profile,
        ] {
            validate_path(path)?;
        }
        Ok(Self {
            models_root,
            model_root,
            source_root,
            artifacts_root,
            receipts_root,
            text_artifact,
            projector_artifact,
            text_receipt,
            projector_receipt,
            preparation_receipt,
            profile,
        })
    }
}

fn canonical_future_directory(path: &Path) -> Result<PathBuf, ModelPreparationError> {
    validate_path(path)?;
    if !path.is_absolute() {
        return Err(plan_error("models root is not absolute"));
    }
    let mut candidate = path.to_path_buf();
    let mut suffix = Vec::<OsString>::new();
    loop {
        match candidate.metadata() {
            Ok(metadata) if metadata.is_dir() => {
                let mut canonical = candidate.canonicalize().map_err(|error| {
                    plan_error(format!("cannot canonicalize models root: {error}"))
                })?;
                for component in suffix.iter().rev() {
                    canonical.push(component);
                }
                validate_path(&canonical)?;
                return Ok(canonical);
            }
            Ok(_) => return Err(plan_error("models-root ancestor is not a directory")),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let name = candidate
                    .file_name()
                    .ok_or_else(|| plan_error("models root has no existing ancestor"))?;
                suffix.push(name.to_os_string());
                if !candidate.pop() {
                    return Err(plan_error("models root has no existing ancestor"));
                }
            }
            Err(error) => {
                return Err(plan_error(format!(
                    "cannot inspect models-root ancestor: {error}"
                )))
            }
        }
    }
}

fn validate_path(path: &Path) -> Result<(), ModelPreparationError> {
    let text = path
        .to_str()
        .ok_or_else(|| plan_error("preparation path is not valid UTF-8"))?;
    if text.is_empty() || text.len() > MAX_MODEL_PREPARATION_PATH_BYTES {
        return Err(plan_error("preparation path exceeds its byte cap"));
    }
    let mut components = 0usize;
    for component in path.components() {
        match component {
            Component::RootDir | Component::Prefix(_) => {}
            Component::Normal(value) => {
                components = components
                    .checked_add(1)
                    .ok_or_else(|| plan_error("path component count overflow"))?;
                let value = value
                    .to_str()
                    .ok_or_else(|| plan_error("path component is not valid UTF-8"))?;
                if value.is_empty() || value.len() > MAX_MODEL_PREPARATION_COMPONENT_BYTES {
                    return Err(plan_error("path component exceeds its byte cap"));
                }
            }
            Component::CurDir | Component::ParentDir => {
                return Err(plan_error("preparation path is not canonical"));
            }
        }
    }
    if components == 0 || components > MAX_MODEL_PREPARATION_PATH_COMPONENTS {
        return Err(plan_error("preparation path component count is invalid"));
    }
    Ok(())
}

fn plan_error(reason: impl Into<String>) -> ModelPreparationError {
    ModelPreparationError::PlanInvalid {
        reason: reason.into(),
    }
}
