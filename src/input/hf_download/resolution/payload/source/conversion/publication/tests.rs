use super::*;
use crate::input::hf_reference::HfModelReference;
use crate::input::model_recipe::{ModelPreparationPlan, RecipeArtifactRole, QWEN38_REPOSITORY_ID};

struct RecordingBackend {
    reauthentications: usize,
    receipt_publications: usize,
    profile_publications: usize,
    finishes: usize,
    drift_at: Option<usize>,
}

impl RecordingBackend {
    fn success() -> Self {
        Self {
            reauthentications: 0,
            receipt_publications: 0,
            profile_publications: 0,
            finishes: 0,
            drift_at: None,
        }
    }
}

impl PublicationBackend for RecordingBackend {
    fn reauthenticate(
        &mut self,
        converted: &ConvertedModelPreparation,
    ) -> Result<PublicationSnapshot, ModelPreparationPublicationError> {
        self.reauthentications += 1;
        let receipt = converted.prepared.receipt();
        let profile =
            PreparedModelProfileV1::build_keep(receipt, converted.prepared.receipt_bytes())?;
        let mut profile_bytes = profile.to_deterministic_json()?;
        if self.drift_at == Some(self.reauthentications) {
            profile_bytes.push(b' ');
        }
        Ok(PublicationSnapshot {
            preparation_receipt: converted.prepared.receipt_bytes().to_vec(),
            profile: profile_bytes,
            model_root_identity: file::Identity::for_test(1),
            receipts_identity: file::Identity::for_test(2),
        })
    }

    fn publish_preparation_receipt(
        &mut self,
        _converted: &ConvertedModelPreparation,
        _expected: &PublicationSnapshot,
    ) -> Result<(), ModelPreparationPublicationError> {
        self.receipt_publications += 1;
        Ok(())
    }

    fn publish_profile(
        &mut self,
        _converted: &ConvertedModelPreparation,
        _expected: &PublicationSnapshot,
    ) -> Result<(), ModelPreparationPublicationError> {
        self.profile_publications += 1;
        Ok(())
    }

    fn finish(
        &mut self,
        _converted: &ConvertedModelPreparation,
        expected: &PublicationSnapshot,
    ) -> Result<PreparedModelProfileV1, ModelPreparationPublicationError> {
        self.finishes += 1;
        Ok(PreparedModelProfileV1::parse(&expected.profile)?)
    }
}

fn converted(root: &Path) -> ConvertedModelPreparation {
    let reference = HfModelReference::parse(QWEN38_REPOSITORY_ID, None).unwrap();
    let plan = ModelPreparationPlan::for_test(
        reference,
        root,
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .unwrap();
    let source = plan.verified_source_at_for_test(Path::new("/sealed/source"), Vec::new());
    let text = plan.verified_conversion_for_test(RecipeArtifactRole::Text, &"a".repeat(40));
    let projector =
        plan.verified_conversion_for_test(RecipeArtifactRole::VisionProjector, &"a".repeat(40));
    let model_root = plan.model_root().to_path_buf();
    let prepared = plan.bind_prepared_pair(source, text, projector).unwrap();
    ConvertedModelPreparation {
        prepared,
        model_root,
    }
}

#[test]
fn publication_reauthenticates_before_and_after_each_durable_record() {
    let root = tempfile::tempdir().unwrap();
    let mut backend = RecordingBackend::success();
    let registered = publish_with(converted(root.path()), &mut backend).unwrap();
    assert_eq!(backend.reauthentications, 3);
    assert_eq!(backend.receipt_publications, 1);
    assert_eq!(backend.profile_publications, 1);
    assert_eq!(backend.finishes, 1);
    assert_eq!(registered.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(
        registered.profile().source_retention(),
        SourceRetentionChoice::Keep
    );
    let debug = format!("{registered:?}");
    assert!(debug.contains("[redacted]"));
    assert!(!debug.contains(root.path().to_str().unwrap()));
}

#[test]
fn drift_after_receipt_prevents_profile_commit_and_drift_after_profile_prevents_authority() {
    for drift_at in [2, 3] {
        let root = tempfile::tempdir().unwrap();
        let mut backend = RecordingBackend {
            drift_at: Some(drift_at),
            ..RecordingBackend::success()
        };
        assert!(publish_with(converted(root.path()), &mut backend).is_err());
        assert_eq!(backend.receipt_publications, 1);
        assert_eq!(backend.profile_publications, usize::from(drift_at == 3),);
        assert_eq!(backend.finishes, 0);
    }
}

#[test]
fn restart_order_accepts_only_write_order_prefixes_and_exact_hardlink_residue() {
    for accepted in [
        (false, false, false, false),
        (false, true, false, false),
        (true, false, false, false),
        (true, true, false, false),
        (true, false, false, true),
        (true, false, true, false),
        (true, false, true, true),
    ] {
        assert!(require_restart_order(accepted.0, accepted.1, accepted.2, accepted.3).is_ok());
    }
    for rejected in [
        (false, false, false, true),
        (false, false, true, false),
        (false, true, true, false),
        (false, true, true, true),
        (true, true, true, false),
        (true, true, true, true),
    ] {
        assert!(require_restart_order(rejected.0, rejected.1, rejected.2, rejected.3).is_err());
    }
}
