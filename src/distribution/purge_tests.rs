use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::Path;

use super::purge::{execute_cache_purge, prepare_cache_purge_at};

fn private_dir(path: &Path) {
    fs::create_dir_all(path).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}

fn manifest(root: &Path, bytes: &[u8]) {
    fs::write(root.join("manifest.json"), bytes).unwrap();
    fs::set_permissions(
        root.join("manifest.json"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
}

#[test]
fn missing_cache_preview_and_execution_are_non_mutating_no_ops() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap().join("missing-cache");
    let plan = prepare_cache_purge_at(&root).unwrap();
    assert!(!plan.contains_data());
    assert!(!root.exists());
    assert_eq!(execute_cache_purge(&plan).unwrap(), 0);
    assert!(!root.exists());
}

#[test]
fn cache_purge_removes_only_models_and_resets_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap().join("cache");
    private_dir(&root);
    private_dir(&root.join("models/repo/quantized/Q4_K_M"));
    private_dir(&root.join("locks"));
    fs::write(
        root.join("models/repo/quantized/Q4_K_M/model.gguf"),
        b"model bytes",
    )
    .unwrap();
    fs::write(root.join("locks/preserve.lock"), b"lock").unwrap();
    fs::write(root.join("operator-note"), b"preserve").unwrap();
    manifest(&root, b"{\"schema_version\":2,\"models\":{}}\n");

    let plan = prepare_cache_purge_at(&root).unwrap();
    assert!(plan.contains_data());
    assert!(root.join("models/repo").exists(), "preview is read-only");
    assert_eq!(execute_cache_purge(&plan).unwrap(), 11);
    assert!(root.join("models").is_dir());
    assert!(fs::read_dir(root.join("models")).unwrap().next().is_none());
    assert_eq!(fs::read(root.join("operator-note")).unwrap(), b"preserve");
    assert_eq!(fs::read(root.join("locks/preserve.lock")).unwrap(), b"lock");
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["models"], serde_json::json!({}));
}

#[test]
fn cache_purge_rejects_unmanifested_data_corrupt_manifest_and_symlinked_models() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap().join("unmanifested");
    private_dir(&root.join("models"));
    fs::write(root.join("models/unknown"), b"keep").unwrap();
    assert!(prepare_cache_purge_at(&root).is_err());
    assert_eq!(fs::read(root.join("models/unknown")).unwrap(), b"keep");

    manifest(&root, b"not json");
    assert!(prepare_cache_purge_at(&root).is_err());
    assert_eq!(fs::read(root.join("models/unknown")).unwrap(), b"keep");

    let linked_root = temp.path().canonicalize().unwrap().join("linked");
    private_dir(&linked_root);
    let outside = temp.path().canonicalize().unwrap().join("outside-models");
    private_dir(&outside);
    fs::write(outside.join("keep"), b"keep").unwrap();
    symlink(&outside, linked_root.join("models")).unwrap();
    manifest(&linked_root, b"{\"schema_version\":2,\"models\":{}}\n");
    assert!(prepare_cache_purge_at(&linked_root).is_err());
    assert_eq!(fs::read(outside.join("keep")).unwrap(), b"keep");
}

#[test]
fn cache_purge_rejects_root_and_relative_targets() {
    assert!(prepare_cache_purge_at(Path::new("/")).is_err());
    assert!(prepare_cache_purge_at(Path::new("relative/cache")).is_err());
}
