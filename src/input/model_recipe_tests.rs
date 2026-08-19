use std::path::Path;

use serde_json::Value;

use crate::core::integrity::ShardIntegrity;

use super::hf_reference::HfModelReference;
use super::integrity::VerifiedSourceManifest;
use super::model_recipe::{
    embedded_qwen38_recipe, recipe_for_reference, ModelRecipe, ModelRecipeError,
    RecipeArtifactRole, RecipeQuantization, SourceRetentionChoice, MAX_MODEL_RECIPE_BYTES,
    QWEN38_ACCEPTED_REVISION, QWEN38_RECIPE_ID,
};

const RECIPE_BYTES: &[u8] = include_bytes!("../../data/model-recipes/qwen38-27b-official-v1.json");
const RECIPE_SHA256: &str = "47a4cec7eb3b19ad68727f557ff47e83f1ef88c791734a76b5bd052d921c9d9d";

fn mutated(mutator: impl FnOnce(&mut Value)) -> Vec<u8> {
    let mut value: Value = serde_json::from_slice(RECIPE_BYTES).unwrap();
    mutator(&mut value);
    let mut bytes = serde_json::to_vec(&value).unwrap();
    bytes.push(b'\n');
    bytes
}

#[test]
fn embedded_recipe_is_exact_canonical_and_binds_accepted_outputs() {
    let recipe = embedded_qwen38_recipe().expect("embedded recipe");
    assert_eq!(recipe.recipe_id(), QWEN38_RECIPE_ID);
    assert_eq!(recipe.recipe_sha256().unwrap(), RECIPE_SHA256);
    assert_eq!(recipe.source().repository_id(), "Qwen/Qwen3.8-27B");
    assert_eq!(recipe.source().revision(), QWEN38_ACCEPTED_REVISION);
    assert_eq!(
        recipe.source().bundle_sha256(),
        "73ded708c49c2d0a47c790ce1d6181e848ac7591dab741de83dbb57218cc6873"
    );
    assert_eq!(recipe.source().files().len(), 29);

    let text = recipe.artifact(RecipeArtifactRole::Text).unwrap();
    assert_eq!(text.quantization(), RecipeQuantization::Q4KM);
    assert_eq!(text.filename(), "Qwen3.8-27B-Q4_K_M.gguf");
    assert_eq!(text.size(), 16_810_714_752);
    assert_eq!(
        text.sha256(),
        "0fa8acc661d0edc60276c43705619fd848682dbf768ced9fe46cd8a572b8043d"
    );
    let projector = recipe
        .artifact(RecipeArtifactRole::VisionProjector)
        .unwrap();
    assert_eq!(projector.quantization(), RecipeQuantization::F16Mmproj);
    assert_eq!(projector.filename(), "Qwen3.8-27B-mmproj-F16.gguf");
    assert_eq!(projector.size(), 927_606_848);
    assert_eq!(
        projector.sha256(),
        "6fa039b75244c0a28a013da30b92b1d221c61029acc19f9efa882b75a495b0d0"
    );

    assert_eq!(
        recipe.interactive_retention_default(),
        SourceRetentionChoice::Keep
    );
    assert!(recipe.non_interactive_retention_requires_explicit());
}

#[test]
fn source_inventory_and_disk_floor_are_derived_exactly() {
    let recipe = embedded_qwen38_recipe().unwrap();
    assert_eq!(
        recipe
            .source()
            .files()
            .iter()
            .map(|file| file.size())
            .sum::<u64>(),
        55_586_101_511
    );
    assert_eq!(
        recipe
            .artifacts()
            .iter()
            .map(|artifact| artifact.size())
            .sum::<u64>(),
        17_738_321_600
    );
    assert_eq!(recipe.minimum_free_bytes(), 81_914_357_703);
    recipe.require_free_space(81_914_357_703).unwrap();
    assert!(matches!(
        recipe.require_free_space(81_914_357_702),
        Err(ModelRecipeError::InsufficientDisk { .. })
    ));

    let names = recipe
        .source()
        .files()
        .iter()
        .map(|file| file.path())
        .collect::<Vec<_>>();
    assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(
        names
            .iter()
            .filter(|name| name.ends_with(".safetensors"))
            .count(),
        18
    );
    assert!(names.contains(&"README.md"));
    assert!(names.contains(&"chat_template.jinja"));
    assert!(names.contains(&"preprocessor_config.json"));
    assert!(names.contains(&"video_preprocessor_config.json"));
}

#[test]
fn only_the_independently_proven_host_profile_is_selected() {
    let recipe = embedded_qwen38_recipe().unwrap();
    let exact = recipe
        .select_hardware_profile(
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
        )
        .unwrap();
    assert_eq!(exact.profile_id(), "qwen38-m5-max-128g-q4-k-m-v1");
    assert_eq!(exact.target(), "aarch64-apple-darwin");
    assert_eq!(exact.chip_model(), "Apple M5 Max");
    assert_eq!(
        exact.minimum_unified_memory_bytes(),
        128 * 1024 * 1024 * 1024
    );
    assert_eq!(exact.text_quantization(), RecipeQuantization::Q4KM);
    assert!(exact.runtime_calibration_required());

    for (target, chip, bytes) in [
        (
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024 - 1,
        ),
        (
            "aarch64-apple-darwin",
            "Apple M4 Max",
            128 * 1024 * 1024 * 1024,
        ),
        (
            "x86_64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
        ),
    ] {
        assert!(matches!(
            recipe.select_hardware_profile(target, chip, bytes),
            Err(ModelRecipeError::UnsupportedHardware { .. })
        ));
    }
}

#[test]
fn recipe_lookup_accepts_bare_or_exact_source_and_rejects_ambiguity() {
    let bare = HfModelReference::parse("Qwen/Qwen3.8-27B", None).unwrap();
    assert_eq!(
        recipe_for_reference(&bare).unwrap().unwrap().recipe_id(),
        QWEN38_RECIPE_ID
    );
    let exact =
        HfModelReference::parse("Qwen/Qwen3.8-27B", Some(QWEN38_ACCEPTED_REVISION)).unwrap();
    assert!(recipe_for_reference(&exact).unwrap().is_some());

    let mutable = HfModelReference::parse("Qwen/Qwen3.8-27B", Some("main")).unwrap();
    assert!(matches!(
        recipe_for_reference(&mutable),
        Err(ModelRecipeError::RevisionNotAccepted { .. })
    ));
    let wrong = HfModelReference::parse("Qwen/Qwen3.8-27B", Some(&"a".repeat(40))).unwrap();
    assert!(matches!(
        recipe_for_reference(&wrong),
        Err(ModelRecipeError::RevisionNotAccepted { .. })
    ));
    let file = HfModelReference::parse(
        &format!(
            "https://huggingface.co/Qwen/Qwen3.8-27B/blob/{QWEN38_ACCEPTED_REVISION}/config.json"
        ),
        None,
    )
    .unwrap();
    assert!(recipe_for_reference(&file).unwrap().is_none());
    let other = HfModelReference::parse("owner/model", None).unwrap();
    assert!(recipe_for_reference(&other).unwrap().is_none());
}

#[test]
fn parser_rejects_size_duplicate_unknown_trailing_depth_and_noncanonical_bytes() {
    let oversized = vec![b' '; MAX_MODEL_RECIPE_BYTES + 1];
    assert!(matches!(
        ModelRecipe::parse(&oversized),
        Err(ModelRecipeError::TooLarge { .. })
    ));

    let text = std::str::from_utf8(RECIPE_BYTES).unwrap();
    let duplicate = text.replacen(
        "\"schema_version\":1,",
        "\"schema_version\":1,\"schema_version\":1,",
        1,
    );
    assert!(ModelRecipe::parse(duplicate.as_bytes()).is_err());

    let unknown = mutated(|value| {
        value
            .as_object_mut()
            .unwrap()
            .insert("extra".into(), Value::Bool(true));
    });
    assert!(ModelRecipe::parse(&unknown).is_err());

    let mut trailing = RECIPE_BYTES.to_vec();
    trailing.extend_from_slice(b"x");
    assert!(ModelRecipe::parse(&trailing).is_err());

    let mut deep = serde_json::from_slice::<Value>(RECIPE_BYTES).unwrap();
    let mut nested = Value::Null;
    for _ in 0..150 {
        nested = Value::Array(vec![nested]);
    }
    deep.as_object_mut().unwrap().insert("extra".into(), nested);
    assert!(ModelRecipe::parse(&serde_json::to_vec(&deep).unwrap()).is_err());

    let pretty =
        serde_json::to_vec_pretty(&serde_json::from_slice::<Value>(RECIPE_BYTES).unwrap()).unwrap();
    assert!(matches!(
        ModelRecipe::parse(&pretty),
        Err(ModelRecipeError::NonCanonical)
    ));
}

#[test]
fn semantic_cross_bindings_fail_closed() {
    let cases: Vec<Box<dyn FnOnce(&mut Value)>> = vec![
        Box::new(|v| v["kind"] = Value::String("other".into())),
        Box::new(|v| v["schema_version"] = Value::from(2)),
        Box::new(|v| v["recipe_id"] = Value::String("other".into())),
        Box::new(|v| v["acceptance"]["accepted_at"] = Value::String("2026-08-18".into())),
        Box::new(|v| v["source"]["repository_id"] = Value::String("attacker/model".into())),
        Box::new(|v| v["source"]["revision"] = Value::String("a".repeat(40))),
        Box::new(|v| v["source"]["bundle_sha256"] = Value::String("a".repeat(64))),
        Box::new(|v| v["source"]["files"][0]["path"] = Value::String("../README.md".into())),
        Box::new(|v| v["source"]["files"][0]["sha256"] = Value::String("A".repeat(64))),
        Box::new(|v| v["source"]["files"][5]["hf_lfs_sha256"] = Value::String("b".repeat(64))),
        Box::new(|v| v["artifacts"][0]["quantization"] = Value::String("q5_k_m".into())),
        Box::new(|v| v["artifacts"][0]["size"] = Value::from(1)),
        Box::new(|v| {
            v["hardware_profiles"][0]["chip_model"] = Value::String("Apple M4 Max".into())
        }),
        Box::new(|v| v["disk"]["minimum_free_bytes"] = Value::from(1)),
        Box::new(|v| v["source_retention"]["deletion_scope"] = Value::String("any_cache".into())),
    ];
    for mutate in cases {
        let bytes = mutated(mutate);
        assert!(ModelRecipe::parse(&bytes).is_err());
    }
}

#[test]
fn artifact_and_source_evidence_cannot_be_cross_bound() {
    let recipe = embedded_qwen38_recipe().unwrap();
    let text = recipe.artifact(RecipeArtifactRole::Text).unwrap();
    recipe
        .verify_artifact_facts_for_test(RecipeArtifactRole::Text, text.size(), text.sha256())
        .unwrap();
    assert!(recipe
        .verify_artifact_facts_for_test(RecipeArtifactRole::Text, text.size() - 1, text.sha256(),)
        .is_err());
    assert!(recipe
        .verify_artifact_facts_for_test(RecipeArtifactRole::Text, text.size(), &"a".repeat(64),)
        .is_err());
    assert!(recipe
        .verify_artifact_facts_for_test(
            RecipeArtifactRole::VisionProjector,
            text.size(),
            text.sha256(),
        )
        .is_err());

    let empty = VerifiedSourceManifest::for_test(Vec::new());
    assert!(matches!(
        recipe.verify_source(Path::new("/unused"), empty),
        Err(ModelRecipeError::SourceMismatch { .. })
    ));
}

#[test]
fn live_recipe_metadata_matches_the_official_immutable_revision() {
    if std::env::var("HF2Q_NETWORK_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping network test (set HF2Q_NETWORK_TESTS=1 to run)");
        return;
    }
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let recipe = embedded_qwen38_recipe().unwrap();
    let api = ApiBuilder::new()
        .with_endpoint("https://huggingface.co".to_owned())
        .with_progress(false)
        .build()
        .expect("build exact-origin Hub client");
    let repo = api.repo(Repo::with_revision(
        recipe.source().repository_id().to_owned(),
        RepoType::Model,
        recipe.source().revision().to_owned(),
    ));
    let local = tempfile::tempdir().unwrap();
    let mut records = Vec::with_capacity(recipe.source().files().len());
    for expected in recipe.source().files() {
        let metadata = api
            .metadata(&repo.url(expected.path()))
            .unwrap_or_else(|error| panic!("metadata {}: {error}", expected.path()));
        assert_eq!(metadata.commit_hash(), recipe.source().revision());
        assert_eq!(metadata.size() as u64, expected.size());
        assert_eq!(metadata.etag().to_ascii_lowercase(), expected.hub_etag());
        let record =
            ShardIntegrity::from_metadata(expected.path(), metadata.etag(), metadata.size() as u64);
        if !record.is_lfs {
            let cached = repo
                .get(expected.path())
                .unwrap_or_else(|error| panic!("download support {}: {error}", expected.path()));
            std::fs::copy(cached, local.path().join(expected.path())).unwrap();
        }
        records.push(record);
    }
    let verified = recipe
        .verify_source(
            local.path(),
            VerifiedSourceManifest::for_test(records.clone()),
        )
        .expect("exact official source matches checked-in recipe");
    assert_eq!(verified.recipe_id(), QWEN38_RECIPE_ID);
    assert_eq!(verified.recipe_sha256(), RECIPE_SHA256);
    assert_eq!(verified.local_dir(), local.path());
    assert_eq!(verified.source_manifest().records(), records);

    std::fs::write(local.path().join("config.json"), b"tampered").unwrap();
    assert!(matches!(
        recipe.verify_source(local.path(), VerifiedSourceManifest::for_test(records)),
        Err(ModelRecipeError::SourceMismatch { .. })
    ));
}

#[test]
fn local_adr044_artifacts_match_when_explicitly_requested() {
    let Some(root) = std::env::var_os("HF2Q_QWEN38_ACCEPTED_ARTIFACT_ROOT") else {
        eprintln!("skipping artifact test (set HF2Q_QWEN38_ACCEPTED_ARTIFACT_ROOT to run)");
        return;
    };
    let root = Path::new(&root);
    let recipe = embedded_qwen38_recipe().unwrap();
    for role in [
        RecipeArtifactRole::Text,
        RecipeArtifactRole::VisionProjector,
    ] {
        let artifact = recipe.artifact(role).unwrap();
        let verified = recipe
            .verify_artifact_path(role, &root.join(artifact.filename()))
            .unwrap();
        assert_eq!(verified.role(), role);
        assert_eq!(verified.sha256(), artifact.sha256());
    }
}
