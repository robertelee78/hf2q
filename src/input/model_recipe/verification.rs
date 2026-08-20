use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::core::integrity::ShardIntegrity;
use crate::core::sha256::compute_file_sha256;
use crate::input::integrity::VerifiedSourceManifest;

use super::{ModelRecipe, ModelRecipeError, RecipeArtifactRole, RecipeSourceFile};

/// Source bytes that matched both the authenticated Hub manifest and hf2q's
/// independently checked-in recipe. Construction is intentionally private.
#[derive(Debug)]
pub struct VerifiedRecipeSource {
    recipe_id: String,
    recipe_sha256: String,
    local_dir: PathBuf,
    verified: VerifiedSourceManifest,
}

impl VerifiedRecipeSource {
    pub fn recipe_id(&self) -> &str {
        &self.recipe_id
    }

    pub fn recipe_sha256(&self) -> &str {
        &self.recipe_sha256
    }

    pub fn local_dir(&self) -> &Path {
        &self.local_dir
    }

    pub fn source_manifest(&self) -> &VerifiedSourceManifest {
        &self.verified
    }
}

/// Exact accepted artifact bytes. This is still preparation evidence, not
/// serving, installation, or activation authority.
#[derive(Debug)]
pub struct VerifiedRecipeArtifact {
    recipe_id: String,
    recipe_sha256: String,
    role: RecipeArtifactRole,
    path: PathBuf,
    sha256: String,
}

impl VerifiedRecipeArtifact {
    pub fn recipe_id(&self) -> &str {
        &self.recipe_id
    }

    pub fn recipe_sha256(&self) -> &str {
        &self.recipe_sha256
    }

    pub fn role(&self) -> RecipeArtifactRole {
        self.role
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }
}

impl ModelRecipe {
    pub fn verify_source(
        &self,
        local_dir: &Path,
        verified: VerifiedSourceManifest,
    ) -> Result<VerifiedRecipeSource, ModelRecipeError> {
        let records = verified.records();
        if records.len() != self.source.files.len() {
            return Err(ModelRecipeError::SourceMismatch {
                reason: format!(
                    "expected {} files, authenticated {}",
                    self.source.files.len(),
                    records.len()
                ),
            });
        }
        let by_name = records
            .iter()
            .map(|record| (record.filename.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        for expected in &self.source.files {
            let actual = by_name.get(expected.path.as_str()).ok_or_else(|| {
                ModelRecipeError::SourceMismatch {
                    reason: format!("missing `{}`", expected.path),
                }
            })?;
            verify_source_record(expected, actual, local_dir)?;
        }
        Ok(VerifiedRecipeSource {
            recipe_id: self.recipe_id.clone(),
            recipe_sha256: self.recipe_sha256()?,
            local_dir: local_dir.to_path_buf(),
            verified,
        })
    }

    pub fn verify_artifact_path(
        &self,
        role: RecipeArtifactRole,
        path: &Path,
    ) -> Result<VerifiedRecipeArtifact, ModelRecipeError> {
        let size = std::fs::metadata(path)?.len();
        let sha256 = compute_file_sha256(path)?;
        self.verify_artifact_facts(role, size, &sha256)?;
        Ok(VerifiedRecipeArtifact {
            recipe_id: self.recipe_id.clone(),
            recipe_sha256: self.recipe_sha256()?,
            role,
            path: path.to_path_buf(),
            sha256,
        })
    }

    fn verify_artifact_facts(
        &self,
        role: RecipeArtifactRole,
        size: u64,
        sha256: &str,
    ) -> Result<(), ModelRecipeError> {
        let expected = self
            .artifact(role)
            .ok_or_else(|| ModelRecipeError::ArtifactMismatch {
                reason: format!("recipe has no {role:?} artifact"),
            })?;
        if size != expected.size || sha256 != expected.sha256 {
            return Err(ModelRecipeError::ArtifactMismatch {
                reason: format!(
                    "{role:?} expected {} bytes / {}, found {size} / {sha256}",
                    expected.size, expected.sha256
                ),
            });
        }
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::input) fn verify_artifact_facts_for_test(
        &self,
        role: RecipeArtifactRole,
        size: u64,
        sha256: &str,
    ) -> Result<(), ModelRecipeError> {
        self.verify_artifact_facts(role, size, sha256)
    }

    #[cfg(test)]
    pub(in crate::input) fn verified_artifact_for_test(
        &self,
        role: RecipeArtifactRole,
        path: PathBuf,
    ) -> VerifiedRecipeArtifact {
        let artifact = self.artifact(role).expect("recipe artifact");
        VerifiedRecipeArtifact {
            recipe_id: self.recipe_id.clone(),
            recipe_sha256: self.recipe_sha256().expect("recipe sha256"),
            role,
            path,
            sha256: artifact.sha256.clone(),
        }
    }

    #[cfg(test)]
    pub(in crate::input) fn verified_source_for_test(&self) -> VerifiedRecipeSource {
        self.verified_source_at_for_test(Path::new("/verified-recipe-source"), Vec::new())
    }

    #[cfg(test)]
    pub(in crate::input) fn verified_source_at_for_test(
        &self,
        local_dir: &Path,
        records: Vec<ShardIntegrity>,
    ) -> VerifiedRecipeSource {
        VerifiedRecipeSource {
            recipe_id: self.recipe_id.clone(),
            recipe_sha256: self.recipe_sha256().expect("recipe sha256"),
            local_dir: local_dir.to_path_buf(),
            verified: VerifiedSourceManifest::for_test_bound(
                &self.source.repository_id,
                &self.source.revision,
                records,
            ),
        }
    }
}

fn verify_source_record(
    expected: &RecipeSourceFile,
    actual: &ShardIntegrity,
    local_dir: &Path,
) -> Result<(), ModelRecipeError> {
    if actual.bytes != expected.size
        || !actual.hf_etag.eq_ignore_ascii_case(&expected.hub_etag)
        || actual.is_lfs != expected.hf_lfs_sha256.is_some()
        || actual.sha256.as_deref() != expected.hf_lfs_sha256.as_deref()
    {
        return Err(ModelRecipeError::SourceMismatch {
            reason: format!(
                "`{}` expected {} bytes / etag {}, found {} / {}",
                expected.path, expected.size, expected.hub_etag, actual.bytes, actual.hf_etag
            ),
        });
    }
    if expected.hf_lfs_sha256.is_none() {
        let digest = compute_file_sha256(&local_dir.join(&expected.path))?;
        if digest != expected.sha256 {
            return Err(ModelRecipeError::SourceMismatch {
                reason: format!(
                    "`{}` expected local SHA-256 {}, found {}",
                    expected.path, expected.sha256, digest
                ),
            });
        }
    }
    Ok(())
}
