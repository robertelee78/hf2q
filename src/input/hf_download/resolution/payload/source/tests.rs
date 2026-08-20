use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use super::*;
use crate::input::hf_download::{
    bind_model_preparation_resolution_for_test, bind_transfer_authorization_for_test,
    ResolvedModelRepository,
};
use crate::input::hf_reference::HfModelReference;
use crate::input::model_recipe::{
    embedded_qwen38_recipe, ModelPreparationPlan, QWEN38_ACCEPTED_REVISION,
};

struct RecordingAuthenticator {
    requested: Vec<String>,
    fail: bool,
    wrong_directory: bool,
    swap_snapshot: bool,
}

impl SourceAuthenticator for RecordingAuthenticator {
    fn authenticate(
        &mut self,
        resolved: &super::super::super::ResolvedModelPreparationPlan,
        snapshot_dir: &Path,
        records: Vec<ShardIntegrity>,
    ) -> Result<VerifiedRecipeSource, ModelPreparationSourceAuthenticationError> {
        self.requested = records
            .iter()
            .map(|record| record.filename.clone())
            .collect();
        if self.fail {
            return Err(IntegrityError::InvalidSourceManifest {
                reason: "injected source authentication failure".to_owned(),
            }
            .into());
        }
        if self.swap_snapshot {
            let detached = snapshot_dir.with_extension("detached");
            std::fs::rename(snapshot_dir, detached).unwrap();
            std::fs::create_dir(snapshot_dir).unwrap();
        }
        let local_dir = if self.wrong_directory {
            snapshot_dir.parent().unwrap()
        } else {
            snapshot_dir
        };
        Ok(resolved
            .plan
            .verified_source_at_for_test(local_dir, records))
    }
}

fn authorized(root: &Path) -> super::super::super::AuthorizedModelPreparationTransfer {
    let reference = HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap();
    let recipe = embedded_qwen38_recipe().unwrap();
    let plan = ModelPreparationPlan::for_test(
        reference.clone(),
        root,
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        recipe.minimum_free_bytes(),
    )
    .unwrap();
    let inventory = recipe
        .source()
        .files()
        .iter()
        .map(|file| file.path().to_owned())
        .collect::<BTreeSet<_>>();
    let resolution = ResolvedModelRepository::for_test(
        reference.resolve(QWEN38_ACCEPTED_REVISION).unwrap(),
        inventory,
    );
    let resolved = bind_model_preparation_resolution_for_test(plan, resolution).unwrap();
    let records = recipe
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
        .collect();
    bind_transfer_authorization_for_test(resolved, records).unwrap()
}

fn transferred(root: &Path) -> TransferredModelPreparationPayload {
    let authorized = authorized(root);
    let snapshot = expected_snapshot(&authorized);
    std::fs::create_dir_all(&snapshot).unwrap();
    super::super::transferred_payload_for_test(authorized, snapshot)
}

fn expected_snapshot(
    authorized: &super::super::super::AuthorizedModelPreparationTransfer,
) -> PathBuf {
    authorized
        .source_root()
        .join("models--Qwen--Qwen3.8-27B")
        .join("snapshots")
        .join(QWEN38_ACCEPTED_REVISION)
}

#[test]
fn reopened_source_consumes_the_exact_recipe_inventory_and_stays_inert() {
    let temp = tempfile::tempdir().unwrap();
    let transferred = transferred(temp.path());
    let mut authenticator = RecordingAuthenticator {
        requested: Vec::new(),
        fail: false,
        wrong_directory: false,
        swap_snapshot: false,
    };
    let authenticated = authenticate_with(transferred, &mut authenticator).unwrap();
    let expected = embedded_qwen38_recipe()
        .unwrap()
        .source()
        .files()
        .iter()
        .map(|file| file.path().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(authenticator.requested, expected);
    assert_eq!(authenticated.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(authenticated.authenticated_file_count(), 29);
    assert_eq!(
        authenticated.model_root(),
        temp.path()
            .canonicalize()
            .unwrap()
            .join("huggingface/Qwen/Qwen3.8-27B/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    );
    assert!(!format!("{authenticated:?}").contains("snapshots"));
}

#[test]
fn failed_or_cross_bound_reauthentication_mints_no_source_capability() {
    let temp = tempfile::tempdir().unwrap();
    let mut failed = RecordingAuthenticator {
        requested: Vec::new(),
        fail: true,
        wrong_directory: false,
        swap_snapshot: false,
    };
    assert!(authenticate_with(transferred(temp.path()), &mut failed).is_err());
    assert_eq!(failed.requested.len(), 29);

    let mut cross_bound = RecordingAuthenticator {
        requested: Vec::new(),
        fail: false,
        wrong_directory: true,
        swap_snapshot: false,
    };
    assert!(authenticate_with(transferred(temp.path()), &mut cross_bound).is_err());
}

#[test]
fn snapshot_replacement_during_authentication_mints_no_source_capability() {
    let temp = tempfile::tempdir().unwrap();
    let mut authenticator = RecordingAuthenticator {
        requested: Vec::new(),
        fail: false,
        wrong_directory: false,
        swap_snapshot: true,
    };
    assert!(authenticate_with(transferred(temp.path()), &mut authenticator).is_err());
    assert_eq!(authenticator.requested.len(), 29);
}

#[test]
fn changed_snapshot_namespace_fails_before_any_source_authentication() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let transferred = transferred(temp.path());
    let snapshot = expected_snapshot_dir(&transferred.resolved);
    std::fs::remove_dir(&snapshot).unwrap();
    let outside = tempfile::tempdir().unwrap();
    symlink(outside.path(), &snapshot).unwrap();
    let mut authenticator = RecordingAuthenticator {
        requested: Vec::new(),
        fail: false,
        wrong_directory: false,
        swap_snapshot: false,
    };
    assert!(authenticate_with(transferred, &mut authenticator).is_err());
    assert!(authenticator.requested.is_empty());
}

#[test]
fn current_recipe_cache_reauthenticates_when_explicitly_requested() {
    if std::env::var_os("HF2Q_TEST_QWEN38_SOURCE_AUTH").is_none() {
        return;
    }
    let models_root = PathBuf::from(
        std::env::var_os("HF2Q_TEST_QWEN38_MODELS_ROOT")
            .expect("HF2Q_TEST_QWEN38_MODELS_ROOT must name a persistent absolute directory"),
    );
    let reference = HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap();
    let planned =
        crate::input::model_recipe::plan_current_model_preparation(reference, &models_root)
            .unwrap();
    let resolved = crate::input::hf_download::resolve_model_preparation_plan(planned).unwrap();
    let authorized =
        crate::input::hf_download::authorize_model_preparation_transfer(resolved).unwrap();
    let snapshot = expected_snapshot(&authorized);
    let transferred = super::super::transferred_payload_for_test(authorized, snapshot);
    let authenticated = authenticate_transferred_model_preparation(transferred).unwrap();
    assert_eq!(authenticated.authenticated_file_count(), 29);
}
