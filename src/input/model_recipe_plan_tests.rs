use std::path::{Path, PathBuf};

use crate::core::integrity::ShardIntegrity;

use super::hf_download::{
    authorize_model_preparation_transfer, bind_model_preparation_resolution_for_test,
    bind_transfer_authorization_for_test, resolve_model_preparation_plan,
    ResolvedModelPreparationPlan, ResolvedModelRepository,
};
use super::hf_reference::HfModelReference;
use super::model_recipe::{
    embedded_qwen38_recipe, plan_current_model_preparation, ModelPreparationError,
    ModelPreparationPlan, RecipeArtifactRole, SourceRetentionChoice, QWEN38_ACCEPTED_REVISION,
};

fn plan(reference: HfModelReference, root: &Path) -> ModelPreparationPlan {
    ModelPreparationPlan::for_test(
        reference,
        root,
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        100 * 1024 * 1024 * 1024,
    )
    .unwrap()
}

fn resolution(
    reference: HfModelReference,
    revision: &str,
    omitted_file: Option<&str>,
) -> ResolvedModelRepository {
    let recipe = embedded_qwen38_recipe().unwrap();
    let inventory = recipe
        .source()
        .files()
        .iter()
        .map(|file| file.path().to_owned())
        .filter(|file| Some(file.as_str()) != omitted_file)
        .chain(["unrelated-repository-entry.txt".to_owned()]);
    ResolvedModelRepository::for_test(reference.resolve(revision).unwrap(), inventory)
}

fn resolved_plan(root: &Path) -> ResolvedModelPreparationPlan {
    let reference = HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap();
    bind_model_preparation_resolution_for_test(
        plan(reference.clone(), root),
        resolution(reference, QWEN38_ACCEPTED_REVISION, None),
    )
    .unwrap()
}

fn exact_transfer_records() -> Vec<ShardIntegrity> {
    embedded_qwen38_recipe()
        .unwrap()
        .source()
        .files()
        .iter()
        .map(|file| ShardIntegrity {
            filename: file.path().to_owned(),
            bytes: file.size(),
            sha256: file.hf_lfs_sha256().map(str::to_owned),
            hf_etag: file.hub_etag().to_owned(),
            is_lfs: file.hf_lfs_sha256().is_some(),
        })
        .collect()
}

#[test]
fn canonical_no_options_layout_is_exact_and_inert() {
    let temp = tempfile::tempdir().unwrap();
    let models = temp.path().join("models");
    let canonical_models = temp.path().canonicalize().unwrap().join("models");
    let planned = plan(
        HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap(),
        &models,
    );
    let root = canonical_models
        .join("huggingface/Qwen/Qwen3.8-27B")
        .join(QWEN38_ACCEPTED_REVISION);
    assert_eq!(planned.models_root(), canonical_models);
    assert_eq!(planned.model_root(), root);
    assert_eq!(planned.source_root(), root.join("source"));
    assert_eq!(planned.artifacts_root(), root.join("artifacts"));
    assert_eq!(planned.receipts_root(), root.join("receipts"));
    assert_eq!(
        planned.artifact_path(RecipeArtifactRole::Text),
        root.join("artifacts/Qwen3.8-27B-Q4_K_M.gguf")
    );
    assert_eq!(
        planned.artifact_path(RecipeArtifactRole::VisionProjector),
        root.join("artifacts/Qwen3.8-27B-mmproj-F16.gguf")
    );
    assert_eq!(
        planned.conversion_receipt_path(RecipeArtifactRole::Text),
        root.join("receipts/Qwen3.8-27B-Q4_K_M.gguf.receipt.json")
    );
    assert_eq!(
        planned.conversion_receipt_path(RecipeArtifactRole::VisionProjector),
        root.join("receipts/Qwen3.8-27B-mmproj-F16.gguf.receipt.json")
    );
    assert_eq!(
        planned.preparation_receipt_path(),
        root.join("receipts/model-preparation.json")
    );
    assert_eq!(planned.profile_path(), root.join("profile.json"));
    assert_eq!(planned.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(planned.accepted_revision(), QWEN38_ACCEPTED_REVISION);
    assert_eq!(
        planned.source_retention_default(),
        SourceRetentionChoice::Keep
    );
    assert!(planned.minimum_free_bytes() > 0);
    assert!(!models.exists(), "planning must not create the layout");
    assert!(format!("{planned:?}").contains("paths: \"[redacted]\""));
    let _ = planned.host_for_test();
}

#[test]
fn equivalent_reference_spellings_select_one_plan_identity() {
    let temp = tempfile::tempdir().unwrap();
    let bare = plan(
        HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap(),
        temp.path(),
    );
    let url = plan(
        HfModelReference::parse(
            &format!("https://huggingface.co/Qwen/Qwen3.8-27B/tree/{QWEN38_ACCEPTED_REVISION}"),
            None,
        )
        .unwrap(),
        temp.path(),
    );
    assert_eq!(bare.model_root(), url.model_root());
    assert_eq!(bare.accepted_revision(), url.accepted_revision());
    assert_eq!(bare.reference().repo_id(), url.reference().repo_id());
}

#[test]
fn exact_hub_resolution_consumes_the_plan_and_preserves_its_layout() {
    let temp = tempfile::tempdir().unwrap();
    let reference = HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap();
    let planned = plan(reference.clone(), temp.path());
    let expected_root = planned.model_root().to_path_buf();
    let resolved = bind_model_preparation_resolution_for_test(
        planned,
        resolution(reference, QWEN38_ACCEPTED_REVISION, None),
    )
    .unwrap();
    assert_eq!(resolved.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(
        resolved.resolved_reference().revision(),
        QWEN38_ACCEPTED_REVISION
    );
    assert_eq!(resolved.repository_inventory_len(), 30);
    assert_eq!(resolved.model_root(), expected_root);
    assert_eq!(resolved.source_root(), expected_root.join("source"));
    assert_eq!(
        resolved.artifact_path(RecipeArtifactRole::Text),
        expected_root.join("artifacts/Qwen3.8-27B-Q4_K_M.gguf")
    );
    assert!(format!("{resolved:?}").contains("paths: \"[redacted]\""));
}

#[test]
fn exact_recipe_metadata_mints_only_an_inert_transfer_authorization() {
    let temp = tempfile::tempdir().unwrap();
    let authorized =
        bind_transfer_authorization_for_test(resolved_plan(temp.path()), exact_transfer_records())
            .unwrap();
    assert_eq!(authorized.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(
        authorized.resolved_reference().revision(),
        QWEN38_ACCEPTED_REVISION
    );
    assert_eq!(authorized.authorized_file_count(), 29);
    assert!(authorized.source_root().ends_with("source"));
    assert!(!authorized.model_root().exists());
    let debug = format!("{authorized:?}");
    assert!(debug.contains("authorized_file_count: 29"));
    assert!(debug.contains("records: \"[redacted]\""));
    assert!(!debug.contains("model-00001-of-00018.safetensors"));
}

#[test]
fn transfer_authorization_rejects_every_metadata_cross_binding() {
    let temp = tempfile::tempdir().unwrap();
    let rejects = |records| {
        bind_transfer_authorization_for_test(resolved_plan(temp.path()), records).is_err()
    };

    let mut missing = exact_transfer_records();
    missing.pop();
    assert!(rejects(missing));

    let mut extra = exact_transfer_records();
    extra.push(extra[0].clone());
    assert!(rejects(extra));

    let mut reordered = exact_transfer_records();
    reordered.swap(0, 1);
    assert!(rejects(reordered));

    let mut wrong_name = exact_transfer_records();
    wrong_name[0].filename.push_str(".other");
    assert!(rejects(wrong_name));

    let mut wrong_size = exact_transfer_records();
    wrong_size[0].bytes += 1;
    assert!(rejects(wrong_size));

    let mut wrong_etag = exact_transfer_records();
    wrong_etag[0].hf_etag = "0".repeat(40);
    assert!(rejects(wrong_etag));

    let mut wrong_kind = exact_transfer_records();
    wrong_kind[0].is_lfs = true;
    wrong_kind[0].sha256 = Some("0".repeat(64));
    assert!(rejects(wrong_kind));

    let lfs = exact_transfer_records()
        .iter()
        .position(|record| record.is_lfs)
        .unwrap();
    let mut wrong_sha = exact_transfer_records();
    wrong_sha[lfs].sha256 = Some("0".repeat(64));
    assert!(rejects(wrong_sha));
}

#[test]
fn resolution_must_match_the_original_plan_revision_and_complete_recipe_inventory() {
    let temp = tempfile::tempdir().unwrap();
    let bare = || HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap();

    let missing = plan(bare(), temp.path());
    assert!(bind_model_preparation_resolution_for_test(
        missing,
        resolution(bare(), QWEN38_ACCEPTED_REVISION, Some("config.json"),),
    )
    .is_err());

    let wrong_revision = plan(bare(), temp.path());
    assert!(bind_model_preparation_resolution_for_test(
        wrong_revision,
        resolution(bare(), &"b".repeat(40), None),
    )
    .is_err());

    let different_original = plan(bare(), temp.path());
    let exact_url = HfModelReference::parse(
        &format!("https://huggingface.co/Qwen/Qwen3.8-27B/tree/{QWEN38_ACCEPTED_REVISION}"),
        None,
    )
    .unwrap();
    assert!(bind_model_preparation_resolution_for_test(
        different_original,
        resolution(exact_url, QWEN38_ACCEPTED_REVISION, None,),
    )
    .is_err());
}

#[test]
fn current_no_options_plan_passes_on_the_explicit_proof_machine() {
    if std::env::var_os("HF2Q_TEST_QWEN38_HOST_PREFLIGHT").is_none() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let planned = plan_current_model_preparation(
        HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap(),
        &temp.path().join("models"),
    )
    .unwrap();
    assert_eq!(planned.accepted_revision(), QWEN38_ACCEPTED_REVISION);
    assert!(!planned.model_root().exists());
}

#[test]
fn current_plan_resolves_through_the_exact_production_hub_boundary() {
    if std::env::var_os("HF2Q_TEST_QWEN38_RESOLVED_PLAN").is_none() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let planned = plan_current_model_preparation(
        HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap(),
        &temp.path().join("models"),
    )
    .unwrap();
    let resolved = resolve_model_preparation_plan(planned).unwrap();
    assert_eq!(
        resolved.resolved_reference().revision(),
        QWEN38_ACCEPTED_REVISION
    );
    assert!(resolved.repository_inventory_len() >= 29);
    assert!(!resolved.model_root().exists());
}

#[test]
fn current_plan_authorizes_all_recipe_metadata_before_payload_transfer() {
    if std::env::var_os("HF2Q_TEST_QWEN38_TRANSFER_AUTH").is_none() {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let planned = plan_current_model_preparation(
        HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap(),
        &temp.path().join("models"),
    )
    .unwrap();
    let resolved = resolve_model_preparation_plan(planned).unwrap();
    let authorized = authorize_model_preparation_transfer(resolved).unwrap();
    assert_eq!(authorized.authorized_file_count(), 29);
    assert_eq!(
        authorized.resolved_reference().revision(),
        QWEN38_ACCEPTED_REVISION
    );
    assert!(!authorized.model_root().exists());
}

#[test]
fn unaccepted_or_file_specific_references_cannot_mint_a_plan() {
    let temp = tempfile::tempdir().unwrap();
    let other = HfModelReference::parse("other/model", None).unwrap();
    assert!(matches!(
        ModelPreparationPlan::for_test(
            other,
            temp.path(),
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
            u64::MAX,
        ),
        Err(ModelPreparationError::PlanInvalid { .. })
    ));

    let file = HfModelReference::parse(
        &format!(
            "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/{QWEN38_ACCEPTED_REVISION}/config.json"
        ),
        None,
    )
    .unwrap();
    assert!(ModelPreparationPlan::for_test(
        file,
        temp.path(),
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .is_err());

    let wrong_revision =
        HfModelReference::parse("Qwen/Qwen3.8-27B", Some(&"b".repeat(40))).unwrap();
    assert!(ModelPreparationPlan::for_test(
        wrong_revision,
        temp.path(),
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .is_err());
}

#[test]
fn preparation_root_is_absolute_canonical_bounded_and_directory_backed() {
    let reference = || HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap();
    for invalid in [
        PathBuf::from("relative"),
        PathBuf::from("/tmp/../tmp/models"),
    ] {
        assert!(ModelPreparationPlan::for_test(
            reference(),
            &invalid,
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
            u64::MAX,
        )
        .is_err());
    }

    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("file");
    std::fs::write(&file, b"occupied").unwrap();
    assert!(ModelPreparationPlan::for_test(
        reference(),
        &file.join("models"),
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .is_err());

    let oversized = temp.path().join("a".repeat(256));
    assert!(ModelPreparationPlan::for_test(
        reference(),
        &oversized,
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .is_err());

    let over_total_cap = temp.path().join("a".repeat(240)).join("b".repeat(240));
    let over_total_cap = (0..16).fold(over_total_cap, |path, index| {
        path.join(format!("{index:02}{}", "c".repeat(238)))
    });
    assert!(ModelPreparationPlan::for_test(
        reference(),
        &over_total_cap,
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .is_err());

    let over_component_cap = (0..65).fold(temp.path().to_path_buf(), |path, index| {
        path.join(format!("c{index}"))
    });
    assert!(ModelPreparationPlan::for_test(
        reference(),
        &over_component_cap,
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .is_err());
}

#[cfg(unix)]
#[test]
fn preparation_root_rejects_non_utf8_components() {
    use std::os::unix::ffi::OsStringExt;

    let temp = tempfile::tempdir().unwrap();
    let invalid = temp
        .path()
        .join(std::ffi::OsString::from_vec(vec![b'm', 0xff]));
    assert!(ModelPreparationPlan::for_test(
        HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap(),
        &invalid,
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .is_err());
}

#[cfg(unix)]
#[test]
fn preparation_root_symlink_is_bound_to_its_canonical_destination() {
    let temp = tempfile::tempdir().unwrap();
    let destination = temp.path().join("destination");
    std::fs::create_dir(&destination).unwrap();
    let link = temp.path().join("models-link");
    std::os::unix::fs::symlink(&destination, &link).unwrap();
    let planned = plan(
        HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap(),
        &link,
    );
    let canonical_destination = destination.canonicalize().unwrap();
    assert_eq!(planned.models_root(), canonical_destination);
    assert!(planned.model_root().starts_with(&canonical_destination));
}
