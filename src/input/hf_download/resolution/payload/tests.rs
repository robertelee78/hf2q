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

struct RecordingExecutor {
    snapshot: PathBuf,
    requested: Vec<String>,
    fail_at: Option<usize>,
}

impl PayloadExecutor for RecordingExecutor {
    fn transfer_one(
        &mut self,
        expected: &ShardIntegrity,
    ) -> Result<PathBuf, ModelPreparationPayloadError> {
        if self.fail_at == Some(self.requested.len()) {
            return Err(DownloadError::DownloadFailed {
                reason: "injected transfer failure".to_owned(),
            }
            .into());
        }
        self.requested.push(expected.filename.clone());
        Ok(self.snapshot.join(&expected.filename))
    }
}

fn authorized(root: &Path) -> AuthorizedModelPreparationTransfer {
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

#[test]
fn payload_transfer_consumes_the_exact_recipe_order_and_stays_inert() {
    let temp = tempfile::tempdir().unwrap();
    let authorized = authorized(temp.path());
    let repo = Repo::with_revision(
        "Qwen/Qwen3.8-27B".to_owned(),
        RepoType::Model,
        QWEN38_ACCEPTED_REVISION.to_owned(),
    );
    let snapshot = authorized
        .source_root()
        .join(repo.folder_name())
        .join("snapshots")
        .join(QWEN38_ACCEPTED_REVISION);
    std::fs::create_dir_all(&snapshot).unwrap();
    let mut executor = RecordingExecutor {
        snapshot: snapshot.clone(),
        requested: Vec::new(),
        fail_at: None,
    };
    let mut completed = 0usize;
    let transferred =
        transfer_with_executor(authorized, &snapshot, &mut executor, || completed += 1).unwrap();
    let expected = embedded_qwen38_recipe()
        .unwrap()
        .source()
        .files()
        .iter()
        .map(|file| file.path().to_owned())
        .collect::<Vec<_>>();
    assert_eq!(executor.requested, expected);
    assert_eq!(completed, 29);
    assert_eq!(transferred.transferred_file_count(), 29);
    assert_eq!(transferred.recipe_id(), "qwen38-27b-official-v1");
    assert!(!format!("{transferred:?}").contains("model-00001"));
}

#[test]
fn failed_or_cross_cache_transfer_mints_no_payload() {
    let temp = tempfile::tempdir().unwrap();
    let initial = authorized(temp.path());
    let source = initial.source_root().to_path_buf();
    let snapshot = source
        .join("models--Qwen--Qwen3.8-27B")
        .join("snapshots")
        .join(QWEN38_ACCEPTED_REVISION);
    std::fs::create_dir_all(&snapshot).unwrap();
    let mut failed = RecordingExecutor {
        snapshot: snapshot.clone(),
        requested: Vec::new(),
        fail_at: Some(7),
    };
    assert!(transfer_with_executor(initial, &snapshot, &mut failed, || {}).is_err());
    assert_eq!(failed.requested.len(), 7);

    let authorized = authorized(temp.path());
    let escaped = temp
        .path()
        .join("elsewhere")
        .join("snapshots")
        .join(QWEN38_ACCEPTED_REVISION);
    std::fs::create_dir_all(&escaped).unwrap();
    let mut executor = RecordingExecutor {
        snapshot: escaped,
        requested: Vec::new(),
        fail_at: None,
    };
    assert!(transfer_with_executor(authorized, &snapshot, &mut executor, || {}).is_err());
}

#[test]
fn namespace_change_is_rejected_before_hub_or_payload_authority() {
    use std::os::unix::fs::symlink;

    let temp = tempfile::tempdir().unwrap();
    let authorized = authorized(temp.path());
    let outside = tempfile::tempdir().unwrap();
    let namespace = temp.path().join("huggingface");
    symlink(outside.path(), &namespace).unwrap();
    let error =
        transfer_authorized_model_preparation(authorized, &ProgressReporter::new()).unwrap_err();
    assert!(error
        .to_string()
        .contains("source root changed after the plan was sealed"));
    assert!(std::fs::read_dir(outside.path()).unwrap().next().is_none());
}

#[test]
fn current_recipe_payload_transfers_and_verifies_when_explicitly_requested() {
    if std::env::var_os("HF2Q_TEST_QWEN38_PAYLOAD_TRANSFER").is_none() {
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
    let transferred = transfer_authorized_model_preparation(
        authorized,
        &crate::progress::ProgressReporter::new(),
    )
    .unwrap();
    assert_eq!(transferred.transferred_file_count(), 29);
    assert!(transferred.model_root().join("source").is_dir());
}
