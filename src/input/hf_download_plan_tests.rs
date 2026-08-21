use std::collections::BTreeSet;

use super::hf_download::{
    bind_snapshot_parent, complete_download_files, default_revision_for, initial_download_files,
    metadata_size_cap, resolve_model_reference, resolve_repository_info_for_test,
    validate_file_metadata, validate_repo_inventory, DownloadError, MAX_HF_REPO_FILES,
    MAX_HF_SMALL_METADATA_BYTES, MAX_HF_TOKENIZER_BYTES,
};
use super::hf_reference::HfModelReference;

#[test]
fn repository_info_seals_a_mutable_request_to_the_returned_exact_commit() {
    let reference = HfModelReference::parse("org/model", Some("main")).unwrap();
    let info = hf_hub::api::RepoInfo {
        sha: "ABCDEF0123456789ABCDEF0123456789ABCDEF01".to_owned(),
        siblings: vec![
            hf_hub::api::Siblings {
                rfilename: "config.json".to_owned(),
            },
            hf_hub::api::Siblings {
                rfilename: "model.safetensors".to_owned(),
            },
        ],
    };
    let resolved = resolve_repository_info_for_test(reference, "main", &info).unwrap();
    assert_eq!(
        resolved.reference().revision(),
        "abcdef0123456789abcdef0123456789abcdef01"
    );
    assert_eq!(resolved.inventory_len(), 2);
    let debug = format!("{resolved:?}");
    assert!(!debug.contains("model.safetensors"));
    assert!(debug.contains("inventory_len: 2"));

    let invalid = hf_hub::api::RepoInfo {
        sha: "main".to_owned(),
        siblings: info.siblings.clone(),
    };
    assert!(resolve_repository_info_for_test(
        HfModelReference::parse("org/model", None).unwrap(),
        "main",
        &invalid,
    )
    .is_err());

    let wrong_exact = HfModelReference::parse(
        "org/model",
        Some("1111111111111111111111111111111111111111"),
    )
    .unwrap();
    assert!(resolve_repository_info_for_test(
        wrong_exact,
        "1111111111111111111111111111111111111111",
        &info
    )
    .is_err());
}

#[test]
fn file_specific_resolution_fails_before_any_hub_lookup() {
    let reference = HfModelReference::parse(
        "https://huggingface.co/org/model/resolve/main/config.json",
        None,
    )
    .unwrap();
    assert!(matches!(
        resolve_model_reference(reference),
        Err(DownloadError::FileReferenceUnsupported)
    ));
}

#[test]
fn downloaded_files_must_share_the_exact_resolved_snapshot() {
    let revision = "a".repeat(40);
    let root = std::path::PathBuf::from("/cache/snapshots").join(&revision);
    let mut selected = None;
    bind_snapshot_parent(
        &mut selected,
        &root.join("config.json"),
        "config.json",
        &revision,
    )
    .unwrap();
    bind_snapshot_parent(
        &mut selected,
        &root.join("sub/model.safetensors"),
        "sub/model.safetensors",
        &revision,
    )
    .unwrap();
    assert_eq!(selected.as_deref(), Some(root.as_path()));

    assert!(bind_snapshot_parent(
        &mut selected,
        &std::path::PathBuf::from("/cache/snapshots")
            .join("b".repeat(40))
            .join("tokenizer.json"),
        "tokenizer.json",
        &revision,
    )
    .is_err());
}

#[test]
fn qwen38_uses_the_adr044_accepted_revision_before_lookup() {
    assert_eq!(
        default_revision_for("Qwen/Qwen3.8-27B"),
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    );
    assert_eq!(default_revision_for("owner/other"), "main");
}

#[test]
fn indexed_plan_is_strict_deterministic_and_downloads_only_required_shards() {
    let inventory = validate_repo_inventory(
        [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "unused.safetensors",
            "pytorch_model.bin",
        ]
        .into_iter(),
    )
    .unwrap();
    let initial = initial_download_files(&inventory).unwrap();
    assert_eq!(
        initial,
        [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json"
        ]
    );
    let plan = complete_download_files(
        &inventory,
        &initial,
        &[
            "model-00002-of-00002.safetensors".to_owned(),
            "model-00001-of-00002.safetensors".to_owned(),
        ],
    )
    .unwrap();
    assert_eq!(
        plan,
        [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json"
        ]
    );
    assert!(!plan.iter().any(|name| name == "unused.safetensors"));
    assert!(!plan.iter().any(|name| name.ends_with(".bin")));
}

#[test]
fn model_card_is_selected_when_present_for_source_and_license_provenance() {
    let inventory =
        validate_repo_inventory(["README.md", "config.json", "model.safetensors"].into_iter())
            .unwrap();
    assert_eq!(
        initial_download_files(&inventory).unwrap(),
        ["config.json", "README.md"]
    );
}

#[test]
fn malformed_or_missing_index_shards_never_fall_back_to_all_weights() {
    let inventory = validate_repo_inventory(
        [
            "config.json",
            "model.safetensors.index.json",
            "present.safetensors",
            "unrelated.safetensors",
        ]
        .into_iter(),
    )
    .unwrap();
    let initial = initial_download_files(&inventory).unwrap();
    assert!(
        complete_download_files(&inventory, &initial, &["missing.safetensors".to_owned()]).is_err()
    );
    assert!(complete_download_files(&inventory, &initial, &[]).is_err());
}

#[test]
fn monolithic_model_plan_requires_the_exact_source_weight() {
    let inventory =
        validate_repo_inventory(["config.json", "model.safetensors", "tokenizer.json"].into_iter())
            .unwrap();
    let initial = initial_download_files(&inventory).unwrap();
    let plan =
        complete_download_files(&inventory, &initial, &["model.safetensors".to_owned()]).unwrap();
    assert!(plan.iter().any(|name| name == "model.safetensors"));

    let metadata_only = BTreeSet::from(["config.json".to_owned()]);
    assert!(initial_download_files(&metadata_only).is_err());
}

#[test]
fn repository_inventory_is_bounded_duplicate_free_and_path_safe() {
    for bad in ["", "/absolute", "../escape", "a/../b", "a\\b", "💥"] {
        assert!(validate_repo_inventory([bad].into_iter()).is_err(), "{bad}");
    }
    assert!(validate_repo_inventory(["config.json", "config.json"].into_iter()).is_err());

    let at_cap = (0..MAX_HF_REPO_FILES)
        .map(|index| format!("f{index}.json"))
        .collect::<Vec<_>>();
    assert_eq!(
        validate_repo_inventory(at_cap.iter().map(String::as_str))
            .unwrap()
            .len(),
        MAX_HF_REPO_FILES
    );
    let over_cap = (0..=MAX_HF_REPO_FILES)
        .map(|index| format!("f{index}.json"))
        .collect::<Vec<_>>();
    assert!(validate_repo_inventory(over_cap.iter().map(String::as_str)).is_err());
}

#[test]
fn immutable_file_metadata_is_checked_before_transfer() {
    let revision = "a".repeat(40);
    let git = "b".repeat(40);
    let lfs = "c".repeat(64);
    let config = validate_file_metadata("config.json", &revision, &revision, &git, 17).unwrap();
    assert!(!config.is_lfs);
    let shard = validate_file_metadata(
        "model-00001-of-00002.safetensors",
        &revision,
        &revision.to_ascii_uppercase(),
        &lfs,
        4096,
    )
    .unwrap();
    assert!(shard.is_lfs);

    assert!(validate_file_metadata("config.json", &revision, &"d".repeat(40), &git, 17).is_err());
    assert!(validate_file_metadata("config.json", &revision, &revision, "opaque", 17).is_err());
    assert!(validate_file_metadata("model.safetensors", &revision, &revision, &git, 4096).is_err());
    assert!(validate_file_metadata("config.json", &revision, &revision, &git, 0).is_err());
}

#[test]
fn metadata_transfer_caps_are_explicit_and_exact() {
    let revision = "a".repeat(40);
    let git = "b".repeat(40);
    assert_eq!(
        metadata_size_cap("model.safetensors.index.json"),
        Some(MAX_HF_SMALL_METADATA_BYTES)
    );
    assert_eq!(
        metadata_size_cap("tokenizer.json"),
        Some(MAX_HF_TOKENIZER_BYTES)
    );
    assert_eq!(metadata_size_cap("model.safetensors"), None);

    for (filename, cap) in [
        ("config.json", MAX_HF_SMALL_METADATA_BYTES),
        ("model.safetensors.index.json", MAX_HF_SMALL_METADATA_BYTES),
        ("tokenizer.json", MAX_HF_TOKENIZER_BYTES),
    ] {
        assert!(validate_file_metadata(filename, &revision, &revision, &git, cap).is_ok());
        assert!(validate_file_metadata(filename, &revision, &revision, &git, cap + 1).is_err());
    }
}
