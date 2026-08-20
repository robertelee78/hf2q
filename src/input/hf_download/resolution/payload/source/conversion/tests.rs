use super::*;
use crate::input::hf_download::resolution::{
    ResolvedModelPreparationPlan, ResolvedModelRepository,
};
use crate::input::hf_reference::HfModelReference;
use crate::input::model_recipe::{
    ModelPreparationPlan, QWEN38_ACCEPTED_REVISION, QWEN38_REPOSITORY_ID,
};

struct RecordingBackend {
    reauthentication_count: usize,
    layout_count: usize,
    roles: Vec<RecipeArtifactRole>,
    final_roles: Vec<RecipeArtifactRole>,
    fail_reauthentication_at: Option<usize>,
    fail_role: Option<RecipeArtifactRole>,
    fail_final_role: Option<RecipeArtifactRole>,
}

impl RecordingBackend {
    fn success() -> Self {
        Self {
            reauthentication_count: 0,
            layout_count: 0,
            roles: Vec::new(),
            final_roles: Vec::new(),
            fail_reauthentication_at: None,
            fail_role: None,
            fail_final_role: None,
        }
    }
}

impl ConversionBackend for RecordingBackend {
    fn reauthenticate_source(
        &mut self,
        resolved: &ResolvedModelPreparationPlan,
        previous: &VerifiedRecipeSource,
    ) -> Result<VerifiedRecipeSource, ModelPreparationConversionError> {
        self.reauthentication_count += 1;
        if self.fail_reauthentication_at == Some(self.reauthentication_count) {
            return Err(conversion_plan_error("scripted source drift").into());
        }
        Ok(resolved.plan.verified_source_at_for_test(
            previous.local_dir(),
            previous.source_manifest().records().to_vec(),
        ))
    }

    fn ensure_layout(
        &mut self,
        _resolved: &ResolvedModelPreparationPlan,
    ) -> Result<(), ModelPreparationConversionError> {
        self.layout_count += 1;
        Ok(())
    }

    fn ensure_role(
        &mut self,
        resolved: &ResolvedModelPreparationPlan,
        _source: &VerifiedRecipeSource,
        role: RecipeArtifactRole,
    ) -> Result<VerifiedRecipeConversion, ModelPreparationConversionError> {
        self.roles.push(role);
        if self.fail_role == Some(role) {
            return Err(conversion_plan_error("scripted conversion failure").into());
        }
        Ok(resolved
            .plan
            .verified_conversion_for_test(role, &"a".repeat(40)))
    }

    fn reauthenticate_role(
        &mut self,
        resolved: &ResolvedModelPreparationPlan,
        role: RecipeArtifactRole,
    ) -> Result<VerifiedRecipeConversion, ModelPreparationConversionError> {
        self.final_roles.push(role);
        if self.fail_final_role == Some(role) {
            return Err(conversion_plan_error("scripted final artifact drift").into());
        }
        Ok(resolved
            .plan
            .verified_conversion_for_test(role, &"a".repeat(40)))
    }
}

fn authenticated(root: &Path) -> AuthenticatedModelPreparationSource {
    let reference = HfModelReference::parse(QWEN38_REPOSITORY_ID, None).unwrap();
    let plan = ModelPreparationPlan::for_test(
        reference.clone(),
        root,
        "aarch64-apple-darwin",
        "Apple M5 Max",
        128 * 1024 * 1024 * 1024,
        u64::MAX,
    )
    .unwrap();
    let source = plan.verified_source_at_for_test(Path::new("/sealed/source"), Vec::new());
    let resolved_reference = reference.resolve(QWEN38_ACCEPTED_REVISION).unwrap();
    let resolution = ResolvedModelRepository::for_test(resolved_reference, Vec::new());
    AuthenticatedModelPreparationSource {
        resolved: ResolvedModelPreparationPlan { plan, resolution },
        source,
    }
}

#[test]
fn coordinator_reauthenticates_around_both_roles_and_returns_only_inert_pair() {
    let root = tempfile::tempdir().unwrap();
    let mut backend = RecordingBackend::success();
    let converted = convert_with(authenticated(root.path()), &mut backend).unwrap();

    assert_eq!(backend.reauthentication_count, 3);
    assert_eq!(backend.layout_count, 1);
    assert_eq!(
        backend.roles,
        [
            RecipeArtifactRole::Text,
            RecipeArtifactRole::VisionProjector
        ]
    );
    assert_eq!(
        backend.final_roles,
        [
            RecipeArtifactRole::Text,
            RecipeArtifactRole::VisionProjector
        ]
    );
    assert_eq!(converted.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(
        crate::input::model_recipe::ModelPreparationReceiptV2::parse(
            converted.preparation_receipt_bytes()
        )
        .unwrap()
        .recipe_id(),
        converted.recipe_id()
    );
    let debug = format!("{converted:?}");
    assert!(debug.contains("[redacted]"));
    assert!(!debug.contains(root.path().to_str().unwrap()));
}

#[test]
fn source_drift_after_text_conversion_prevents_projector_and_pair_authority() {
    let root = tempfile::tempdir().unwrap();
    let mut backend = RecordingBackend {
        fail_reauthentication_at: Some(2),
        ..RecordingBackend::success()
    };
    assert!(convert_with(authenticated(root.path()), &mut backend).is_err());
    assert_eq!(backend.reauthentication_count, 2);
    assert_eq!(backend.roles, [RecipeArtifactRole::Text]);
    assert!(backend.final_roles.is_empty());
}

#[test]
fn projector_failure_prevents_final_reauthentication_and_pair_authority() {
    let root = tempfile::tempdir().unwrap();
    let mut backend = RecordingBackend {
        fail_role: Some(RecipeArtifactRole::VisionProjector),
        ..RecordingBackend::success()
    };
    assert!(convert_with(authenticated(root.path()), &mut backend).is_err());
    assert_eq!(backend.reauthentication_count, 2);
    assert_eq!(
        backend.roles,
        [
            RecipeArtifactRole::Text,
            RecipeArtifactRole::VisionProjector
        ]
    );
    assert!(backend.final_roles.is_empty());
}

#[test]
fn final_artifact_drift_prevents_pair_authority_after_both_conversions() {
    let root = tempfile::tempdir().unwrap();
    let mut backend = RecordingBackend {
        fail_final_role: Some(RecipeArtifactRole::Text),
        ..RecordingBackend::success()
    };
    assert!(convert_with(authenticated(root.path()), &mut backend).is_err());
    assert_eq!(backend.reauthentication_count, 3);
    assert_eq!(
        backend.roles,
        [
            RecipeArtifactRole::Text,
            RecipeArtifactRole::VisionProjector
        ]
    );
    assert_eq!(backend.final_roles, [RecipeArtifactRole::Text]);
}

#[test]
fn restart_shape_adopts_only_complete_pairs_and_rejects_orphan_receipts() {
    assert_eq!(
        role_restart_disposition(false, false).unwrap(),
        RoleRestartDisposition::Convert
    );
    assert_eq!(
        role_restart_disposition(true, false).unwrap(),
        RoleRestartDisposition::ReconvertVerifiedArtifact
    );
    assert_eq!(
        role_restart_disposition(true, true).unwrap(),
        RoleRestartDisposition::Adopt
    );
    assert!(role_restart_disposition(false, true).is_err());
}

#[test]
fn dangling_symlink_is_retained_as_hostile_evidence_instead_of_absence() {
    use std::os::unix::fs::symlink;

    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("artifact.gguf");
    symlink(root.path().join("missing-target"), &artifact).unwrap();
    assert!(path_entry_exists(&artifact).unwrap());
    assert_eq!(
        role_restart_disposition(path_entry_exists(&artifact).unwrap(), false).unwrap(),
        RoleRestartDisposition::ReconvertVerifiedArtifact
    );
    assert!(std::fs::symlink_metadata(&artifact)
        .unwrap()
        .file_type()
        .is_symlink());
}

#[test]
fn current_recipe_converts_and_reopens_the_exact_pair_when_explicitly_requested() {
    if std::env::var_os("HF2Q_TEST_QWEN38_CONVERSION").is_none() {
        return;
    }
    let models_root = PathBuf::from(
        std::env::var_os("HF2Q_TEST_QWEN38_MODELS_ROOT")
            .expect("HF2Q_TEST_QWEN38_MODELS_ROOT must name a persistent absolute directory"),
    );
    let reference = HfModelReference::parse(QWEN38_REPOSITORY_ID, None).unwrap();
    let planned =
        crate::input::model_recipe::plan_current_model_preparation(reference, &models_root)
            .unwrap();
    let resolved = crate::input::hf_download::resolve_model_preparation_plan(planned).unwrap();
    let authorized =
        crate::input::hf_download::authorize_model_preparation_transfer(resolved).unwrap();
    let snapshot = authorized
        .source_root()
        .join("models--Qwen--Qwen3.8-27B")
        .join("snapshots")
        .join(QWEN38_ACCEPTED_REVISION);
    let transferred = super::super::super::transferred_payload_for_test(authorized, snapshot);
    let authenticated = super::super::authenticate_transferred_model_preparation(transferred)
        .expect("cached source authentication");
    let converted = convert_authenticated_model_preparation(authenticated)
        .expect("exact recipe-owned conversion");
    assert_eq!(converted.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(
        crate::input::model_recipe::ModelPreparationReceiptV2::parse(
            converted.preparation_receipt_bytes()
        )
        .unwrap()
        .recipe_id(),
        converted.recipe_id()
    );
    let registered = publish_converted_model_preparation_keep(converted)
        .expect("durable exact preparation receipt and profile publication");
    assert_eq!(registered.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(
        registered.profile().source_retention(),
        crate::input::model_recipe::SourceRetentionChoice::Keep
    );
    assert_eq!(
        crate::input::model_recipe::PreparedModelProfileV1::parse(
            &std::fs::read(registered.model_root().join("profile.json")).unwrap()
        )
        .unwrap(),
        *registered.profile()
    );
}
