use std::collections::BTreeSet;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use super::{
    file, publication_error, publication_require, ConvertedModelPreparation,
    ModelPreparationPublicationError, PublicationSnapshot, PREPARATION_RECEIPT_NAME,
    PREPARATION_RECEIPT_PARTIAL, PROFILE_NAME, PROFILE_PARTIAL,
};
use crate::input::integrity::verify_conversion_manifest;
use crate::input::model_recipe::{
    embedded_qwen38_recipe, require_exact_regular_file, ModelPreparationReceiptV2,
    PreparedModelProfileV1, RecipeArtifactRole, MAX_CONVERSION_RECEIPT_BYTES,
    MAX_MODEL_PREPARATION_RECEIPT_BYTES, MAX_PREPARED_MODEL_PROFILE_BYTES,
};

pub(super) fn reauthenticate_pair(
    converted: &ConvertedModelPreparation,
) -> Result<PublicationSnapshot, ModelPreparationPublicationError> {
    require_publication_inventory(converted)?;
    let receipts = converted.model_root.join("receipts");
    let model_root_identity = file::directory_identity(&converted.model_root)?;
    let receipts_identity = file::directory_identity(&receipts)?;
    super::require_restart_order(
        file::entry_exists(&receipts.join(PREPARATION_RECEIPT_NAME))?,
        file::entry_exists(&receipts.join(PREPARATION_RECEIPT_PARTIAL))?,
        file::entry_exists(&converted.model_root.join(PROFILE_NAME))?,
        file::entry_exists(&converted.model_root.join(PROFILE_PARTIAL))?,
    )?;
    let pair = ModelPreparationReceiptV2::parse(converted.prepared.receipt_bytes())?;
    publication_require(
        &pair == converted.prepared.receipt(),
        "sealed preparation receipt differs from its exact bytes",
    )?;
    let recipe = embedded_qwen38_recipe()?;
    let model_device = model_root_identity.device();
    let source = converted.prepared.source();
    publication_require(
        source
            .local_dir()
            .starts_with(converted.model_root.join("source"))
            && source.recipe_id() == pair.recipe_id()
            && source.recipe_sha256() == pair.recipe_sha256(),
        "sealed source does not belong to the model root and pair",
    )?;
    let source_directories = source_directories(source.local_dir(), &converted.model_root)?;
    let source_identities = source_directories
        .iter()
        .map(|path| -> Result<_, ModelPreparationPublicationError> {
            require_owned_device(path, model_device, true)?;
            Ok(super::super::super::require_exact_source_directory(path)?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let verified_manifest = verify_conversion_manifest(
        pair.repository_id(),
        pair.revision(),
        source.local_dir(),
        source.source_manifest().records().to_vec(),
    )?;
    let reopened_source = recipe.verify_source(source.local_dir(), verified_manifest)?;
    publication_require(
        reopened_source.recipe_id() == source.recipe_id()
            && reopened_source.recipe_sha256() == source.recipe_sha256()
            && reopened_source.source_manifest().records().len()
                == source.source_manifest().records().len(),
        "reopened source differs from the sealed source",
    )?;

    for (role, sealed_artifact) in [
        (RecipeArtifactRole::Text, converted.prepared.text_artifact()),
        (
            RecipeArtifactRole::VisionProjector,
            converted.prepared.projector_artifact(),
        ),
    ] {
        let expected = recipe
            .artifact(role)
            .ok_or_else(|| publication_error("recipe artifact is absent"))?;
        let artifact_path = converted
            .model_root
            .join("artifacts")
            .join(expected.filename());
        publication_require(
            sealed_artifact.path() == artifact_path,
            "sealed artifact path differs from the canonical model layout",
        )?;
        require_exact_regular_file(&artifact_path)?;
        require_owned_device(&artifact_path, model_device, false)?;
        let reopened_artifact = recipe.verify_artifact_path(role, &artifact_path)?;
        publication_require(
            reopened_artifact.sha256() == sealed_artifact.sha256(),
            "reopened artifact differs from the sealed artifact",
        )?;
        let receipt_name = format!("{}.receipt.json", expected.filename());
        let receipt_path = receipts.join(&receipt_name);
        require_exact_regular_file(&receipt_path)?;
        require_owned_device(&receipt_path, model_device, false)?;
        let receipt_bytes = file::read_bounded_owned_file(
            &receipts,
            receipts_identity,
            &receipt_name,
            MAX_CONVERSION_RECEIPT_BYTES,
        )?;
        let conversion =
            recipe.verify_conversion_receipt(role, reopened_artifact, &receipt_bytes)?;
        publication_require(
            pair.artifact_receipt_sha256(role) == Some(conversion.receipt_sha256()),
            "conversion receipt differs from the sealed pair",
        )?;
    }

    require_publication_inventory(converted)?;
    for (path, expected) in source_directories.into_iter().zip(source_identities) {
        publication_require(
            super::super::super::require_exact_source_directory(&path)? == expected,
            "source namespace changed during registry authentication",
        )?;
    }
    let profile = PreparedModelProfileV1::build_keep(&pair, converted.prepared.receipt_bytes())?;
    let snapshot = PublicationSnapshot {
        preparation_receipt: converted.prepared.receipt_bytes().to_vec(),
        profile: profile.to_deterministic_json()?,
        model_root_identity,
        receipts_identity,
    };
    if file::entry_exists(&receipts.join(PREPARATION_RECEIPT_NAME))? {
        file::read_exact_private_file(
            &receipts,
            receipts_identity,
            PREPARATION_RECEIPT_NAME,
            &snapshot.preparation_receipt,
            MAX_MODEL_PREPARATION_RECEIPT_BYTES,
        )?;
    }
    if file::entry_exists(&converted.model_root.join(PROFILE_NAME))? {
        file::read_exact_private_file(
            &converted.model_root,
            model_root_identity,
            PROFILE_NAME,
            &snapshot.profile,
            MAX_PREPARED_MODEL_PROFILE_BYTES,
        )?;
    }
    Ok(snapshot)
}

pub(super) fn require_final_inventory(
    converted: &ConvertedModelPreparation,
) -> Result<(), ModelPreparationPublicationError> {
    require_publication_inventory(converted)?;
    let receipts = converted.model_root.join("receipts");
    publication_require(
        file::entry_exists(&converted.model_root.join(PROFILE_NAME))?
            && !file::entry_exists(&converted.model_root.join(PROFILE_PARTIAL))?
            && file::entry_exists(&receipts.join(PREPARATION_RECEIPT_NAME))?
            && !file::entry_exists(&receipts.join(PREPARATION_RECEIPT_PARTIAL))?,
        "durable prepared-model publication is incomplete",
    )
}

fn source_directories(
    snapshot: &Path,
    model_root: &Path,
) -> Result<[PathBuf; 4], ModelPreparationPublicationError> {
    let snapshots = snapshot
        .parent()
        .ok_or_else(|| publication_error("source snapshot has no snapshots parent"))?;
    let repository = snapshots
        .parent()
        .ok_or_else(|| publication_error("source snapshot has no repository parent"))?;
    let source = repository
        .parent()
        .ok_or_else(|| publication_error("source repository has no source parent"))?;
    publication_require(
        source == model_root.join("source"),
        "source tree is outside the canonical model root",
    )?;
    Ok([
        source.to_path_buf(),
        repository.to_path_buf(),
        snapshots.to_path_buf(),
        snapshot.to_path_buf(),
    ])
}

fn require_publication_inventory(
    converted: &ConvertedModelPreparation,
) -> Result<(), ModelPreparationPublicationError> {
    require_names(
        &converted.model_root,
        &[
            "source",
            "artifacts",
            "receipts",
            PROFILE_NAME,
            PROFILE_PARTIAL,
        ],
        &["source", "artifacts", "receipts"],
    )?;
    let recipe = embedded_qwen38_recipe()?;
    let artifact_names = recipe
        .artifacts()
        .iter()
        .map(|artifact| artifact.filename().to_owned())
        .collect::<Vec<_>>();
    require_names_owned(
        &converted.model_root.join("artifacts"),
        &artifact_names,
        &artifact_names,
    )?;
    let mut receipt_names = recipe
        .artifacts()
        .iter()
        .map(|artifact| format!("{}.receipt.json", artifact.filename()))
        .collect::<Vec<_>>();
    let required_receipts = receipt_names.clone();
    receipt_names.extend([
        PREPARATION_RECEIPT_NAME.to_owned(),
        PREPARATION_RECEIPT_PARTIAL.to_owned(),
    ]);
    require_names_owned(
        &converted.model_root.join("receipts"),
        &receipt_names,
        &required_receipts,
    )
}

fn require_names(
    directory: &Path,
    allowed: &[&str],
    required: &[&str],
) -> Result<(), ModelPreparationPublicationError> {
    require_names_owned(
        directory,
        &allowed
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>(),
        &required
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>(),
    )
}

fn require_names_owned(
    directory: &Path,
    allowed: &[String],
    required: &[String],
) -> Result<(), ModelPreparationPublicationError> {
    super::super::super::require_exact_source_directory(directory)?;
    let metadata = std::fs::symlink_metadata(directory)?;
    publication_require(
        metadata.uid() == rustix::process::geteuid().as_raw(),
        "model publication directory is not owned by the current user",
    )?;
    let mut actual = BTreeSet::new();
    for entry in std::fs::read_dir(directory)?.take(allowed.len() + 1) {
        let name = entry?
            .file_name()
            .into_string()
            .map_err(|_| publication_error("model publication inventory is not UTF-8"))?;
        publication_require(
            allowed.iter().any(|allowed| allowed == &name),
            "model publication inventory contains an unrelated entry",
        )?;
        publication_require(actual.insert(name), "duplicate model publication entry")?;
    }
    publication_require(
        actual.len() <= allowed.len() && required.iter().all(|required| actual.contains(required)),
        "model publication inventory is missing a required entry",
    )
}

fn require_owned_device(
    path: &Path,
    expected_device: u64,
    directory: bool,
) -> Result<(), ModelPreparationPublicationError> {
    let metadata = std::fs::symlink_metadata(path)?;
    publication_require(
        metadata.uid() == rustix::process::geteuid().as_raw()
            && metadata.dev() == expected_device
            && if directory {
                metadata.file_type().is_dir()
            } else {
                metadata.file_type().is_file() && metadata.nlink() == 1
            },
        "model publication entry ownership, device, type, or links are invalid",
    )
}
