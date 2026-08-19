use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::convert::receipt::{
    ConversionReceipt, ConverterReceipt, ExcludedDsparkReceipt, OutputReceipt,
    PeakChunkBoundReceipt, SourceFileReceipt, SourceReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
};

use super::model_recipe::{
    embedded_qwen38_recipe, ModelPreparationError, ModelPreparationReceiptV1, RecipeArtifactRole,
    MAX_MODEL_PREPARATION_RECEIPT_BYTES,
};

const PREPARATION_GOLDEN: &[u8] =
    include_bytes!("../../data/model-recipes/qwen38-27b-preparation-receipt-v1.json");

fn conversion_receipt(role: RecipeArtifactRole, artifact_path: &Path, git_commit: &str) -> Vec<u8> {
    let recipe = embedded_qwen38_recipe().unwrap();
    let expected = recipe.artifact(role).unwrap();
    let (strategy, scope) = match role {
        RecipeArtifactRole::Text => ("row_aligned_tensor_chunks", "all_streamed_tensors"),
        RecipeArtifactRole::VisionProjector => (
            "lazy_source_index_projector_only",
            "multimodal_projector_tensors",
        ),
    };
    let receipt = ConversionReceipt {
        schema_version: CONVERSION_RECEIPT_SCHEMA_VERSION,
        source: SourceReceipt {
            original_reference: "Qwen/Qwen3.8-27B".into(),
            repository_id: recipe.source().repository_id().into(),
            repository_type: "model".into(),
            canonical_url: "https://huggingface.co/Qwen/Qwen3.8-27B".into(),
            revision: recipe.source().revision().into(),
            filename: None,
            bundle_sha256: recipe.source().bundle_sha256().into(),
            files: recipe
                .source()
                .files()
                .iter()
                .map(|file| SourceFileReceipt {
                    path: file.path().into(),
                    size: file.size(),
                    sha256: file.sha256().into(),
                    hf_lfs_sha256: file.hf_lfs_sha256().map(str::to_owned),
                })
                .collect(),
        },
        converter: ConverterReceipt {
            package: "hf2q".into(),
            version: "0.1.7".into(),
            git_commit: git_commit.into(),
        },
        quant_selector: expected.quantization().as_str().into(),
        output: OutputReceipt {
            path: artifact_path.display().to_string(),
            size: expected.size(),
            sha256: expected.sha256().into(),
        },
        excluded_dspark: ExcludedDsparkReceipt {
            tensor_count: 0,
            status: "none_detected".into(),
        },
        peak_chunk_bound: PeakChunkBoundReceipt {
            strategy: strategy.into(),
            scope: scope.into(),
            max_chunk_elements: if role == RecipeArtifactRole::Text {
                1024
            } else {
                0
            },
            max_input_f32_bytes: if role == RecipeArtifactRole::Text {
                4096
            } else {
                0
            },
            max_f16_roundtrip_f32_bytes: 0,
            max_quantized_payload_bytes: 0,
            max_working_vec_bytes: 0,
        },
    };
    let mut bytes = serde_json::to_vec_pretty(&receipt).unwrap();
    bytes.push(b'\n');
    bytes
}

fn verified_conversion(
    role: RecipeArtifactRole,
    git_commit: &str,
) -> super::model_recipe::VerifiedRecipeConversion {
    let recipe = embedded_qwen38_recipe().unwrap();
    let path = PathBuf::from("/models").join(recipe.artifact(role).unwrap().filename());
    let artifact = recipe.verified_artifact_for_test(role, path.clone());
    recipe
        .verify_conversion_receipt(role, artifact, &conversion_receipt(role, &path, git_commit))
        .unwrap()
}

fn prepared_pair() -> super::model_recipe::VerifiedModelPreparation {
    let recipe = embedded_qwen38_recipe().unwrap();
    let host = recipe
        .verify_host_and_disk(
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
            recipe.minimum_free_bytes(),
        )
        .unwrap();
    recipe
        .bind_prepared_pair(
            recipe.verified_source_for_test(),
            host,
            verified_conversion(RecipeArtifactRole::Text, &"a".repeat(40)),
            verified_conversion(RecipeArtifactRole::VisionProjector, &"a".repeat(40)),
        )
        .unwrap()
}

#[test]
fn exact_pair_receipt_is_canonical_and_structurally_reparseable() {
    let prepared = prepared_pair();
    assert_eq!(prepared.receipt_bytes(), PREPARATION_GOLDEN);
    let receipt = prepared.receipt();
    assert_eq!(receipt.recipe_id(), "qwen38-27b-official-v1");
    assert_eq!(receipt.repository_id(), "Qwen/Qwen3.8-27B");
    assert_eq!(
        receipt.revision(),
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    );
    assert_eq!(
        receipt.hardware_profile_id(),
        "qwen38-m5-max-128g-q4-k-m-v1"
    );
    assert_eq!(
        ModelPreparationReceiptV1::parse(prepared.receipt_bytes()).unwrap(),
        *receipt
    );
    assert_eq!(
        prepared.text_artifact().path(),
        Path::new("/models/Qwen3.8-27B-Q4_K_M.gguf")
    );
    assert_eq!(
        prepared.projector_artifact().path(),
        Path::new("/models/Qwen3.8-27B-mmproj-F16.gguf")
    );
}

#[test]
fn host_and_disk_policy_is_sealed_before_pair_construction() {
    let recipe = embedded_qwen38_recipe().unwrap();
    recipe
        .verify_host_and_disk(
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
            recipe.minimum_free_bytes(),
        )
        .unwrap();
    for (target, chip, memory, disk) in [
        (
            "x86_64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
            recipe.minimum_free_bytes(),
        ),
        (
            "aarch64-apple-darwin",
            "Apple M4 Max",
            128 * 1024 * 1024 * 1024,
            recipe.minimum_free_bytes(),
        ),
        (
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024 - 1,
            recipe.minimum_free_bytes(),
        ),
        (
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
            recipe.minimum_free_bytes() - 1,
        ),
    ] {
        assert!(recipe
            .verify_host_and_disk(target, chip, memory, disk)
            .is_err());
    }
}

#[test]
fn conversion_receipt_is_bounded_canonical_and_role_exact() {
    let recipe = embedded_qwen38_recipe().unwrap();
    let role = RecipeArtifactRole::Text;
    let path = PathBuf::from("/models").join(recipe.artifact(role).unwrap().filename());

    let artifact = recipe.verified_artifact_for_test(role, path.clone());
    let mut pretty = conversion_receipt(role, &path, &"a".repeat(40));
    pretty.insert(0, b' ');
    assert!(matches!(
        recipe.verify_conversion_receipt(role, artifact, &pretty),
        Err(ModelPreparationError::ConversionMismatch { .. })
    ));

    let artifact = recipe.verified_artifact_for_test(role, path.clone());
    assert!(matches!(
        recipe.verify_conversion_receipt(
            role,
            artifact,
            &vec![b' '; MAX_MODEL_PREPARATION_RECEIPT_BYTES + 1],
        ),
        Err(ModelPreparationError::TooLarge { .. })
    ));

    let artifact = recipe.verified_artifact_for_test(role, path.clone());
    let projector = conversion_receipt(RecipeArtifactRole::VisionProjector, &path, &"a".repeat(40));
    assert!(matches!(
        recipe.verify_conversion_receipt(role, artifact, &projector),
        Err(ModelPreparationError::ConversionMismatch { .. })
    ));

    let canonical = conversion_receipt(role, &path, &"a".repeat(40));
    let text = std::str::from_utf8(&canonical).unwrap();
    let duplicate = text.replacen(
        "\"schema_version\": 3,",
        "\"schema_version\": 3,\"schema_version\": 3,",
        1,
    );
    assert_ne!(duplicate.as_bytes(), canonical);
    assert!(recipe
        .verify_conversion_receipt(
            role,
            recipe.verified_artifact_for_test(role, path.clone()),
            duplicate.as_bytes(),
        )
        .is_err());

    let mut unknown: Value = serde_json::from_slice(&canonical).unwrap();
    unknown
        .as_object_mut()
        .unwrap()
        .insert("extra".into(), Value::Bool(true));
    let mut unknown_bytes = serde_json::to_vec_pretty(&unknown).unwrap();
    unknown_bytes.push(b'\n');
    assert!(recipe
        .verify_conversion_receipt(
            role,
            recipe.verified_artifact_for_test(role, path.clone()),
            &unknown_bytes,
        )
        .is_err());

    let mut trailing = canonical;
    trailing.extend_from_slice(b"x");
    assert!(recipe
        .verify_conversion_receipt(
            role,
            recipe.verified_artifact_for_test(role, path),
            &trailing,
        )
        .is_err());

    #[cfg(unix)]
    {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let mut bytes = b"/models/".to_vec();
        bytes.push(0xff);
        bytes.extend_from_slice(b"/Qwen3.8-27B-Q4_K_M.gguf");
        let non_utf8 = PathBuf::from(OsString::from_vec(bytes));
        let receipt = conversion_receipt(role, &non_utf8, &"a".repeat(40));
        assert!(recipe
            .verify_conversion_receipt(
                role,
                recipe.verified_artifact_for_test(role, non_utf8),
                &receipt,
            )
            .is_err());
    }
}

#[test]
fn every_conversion_cross_binding_fails_closed() {
    let cases: Vec<Box<dyn FnOnce(&mut Value)>> = vec![
        Box::new(|value| value["schema_version"] = Value::from(2)),
        Box::new(|value| {
            value["source"]["original_reference"] = Value::String("attacker/model".into())
        }),
        Box::new(|value| value["source"]["repository_id"] = Value::String("attacker/model".into())),
        Box::new(|value| value["source"]["revision"] = Value::String("b".repeat(40))),
        Box::new(|value| value["source"]["bundle_sha256"] = Value::String("b".repeat(64))),
        Box::new(|value| value["source"]["files"][0]["size"] = Value::from(1)),
        Box::new(|value| value["source"]["files"][0]["sha256"] = Value::String("b".repeat(64))),
        Box::new(|value| value["converter"]["package"] = Value::String("other".into())),
        Box::new(|value| value["converter"]["version"] = Value::String("v0.1.7".into())),
        Box::new(|value| value["converter"]["git_commit"] = Value::String("A".repeat(40))),
        Box::new(|value| value["quant_selector"] = Value::String("q5_k_m".into())),
        Box::new(|value| value["output"]["path"] = Value::String("/other/model.gguf".into())),
        Box::new(|value| value["output"]["size"] = Value::from(1)),
        Box::new(|value| value["output"]["sha256"] = Value::String("b".repeat(64))),
        Box::new(|value| value["excluded_dspark"]["tensor_count"] = Value::from(1)),
        Box::new(|value| value["peak_chunk_bound"]["strategy"] = Value::String("other".into())),
    ];
    for mutate in cases {
        let recipe = embedded_qwen38_recipe().unwrap();
        let role = RecipeArtifactRole::Text;
        let path = PathBuf::from("/models").join(recipe.artifact(role).unwrap().filename());
        let mut value: Value =
            serde_json::from_slice(&conversion_receipt(role, &path, &"a".repeat(40))).unwrap();
        mutate(&mut value);
        let mut bytes = serde_json::to_vec_pretty(&value).unwrap();
        bytes.push(b'\n');
        let artifact = recipe.verified_artifact_for_test(role, path);
        assert!(recipe
            .verify_conversion_receipt(role, artifact, &bytes)
            .is_err());
    }
}

#[test]
fn pair_rejects_converter_or_role_mix_and_consumes_sealed_proofs() {
    let recipe = embedded_qwen38_recipe().unwrap();
    let host = recipe
        .verify_host_and_disk(
            "aarch64-apple-darwin",
            "Apple M5 Max",
            128 * 1024 * 1024 * 1024,
            recipe.minimum_free_bytes(),
        )
        .unwrap();
    let mismatch = recipe.bind_prepared_pair(
        recipe.verified_source_for_test(),
        host,
        verified_conversion(RecipeArtifactRole::Text, &"a".repeat(40)),
        verified_conversion(RecipeArtifactRole::VisionProjector, &"b".repeat(40)),
    );
    assert!(matches!(
        mismatch,
        Err(ModelPreparationError::PairMismatch { .. })
    ));
}

#[test]
fn structural_pair_parser_rejects_unknown_duplicate_trailing_and_semantic_mutation() {
    let bytes = prepared_pair().receipt_bytes().to_vec();
    assert!(matches!(
        ModelPreparationReceiptV1::parse(&vec![b' '; MAX_MODEL_PREPARATION_RECEIPT_BYTES + 1]),
        Err(ModelPreparationError::TooLarge { .. })
    ));
    let text = std::str::from_utf8(&bytes).unwrap();
    let duplicate = text.replacen(
        "\"schema_version\":1,",
        "\"schema_version\":1,\"schema_version\":1,",
        1,
    );
    assert!(ModelPreparationReceiptV1::parse(duplicate.as_bytes()).is_err());

    let mut value: Value = serde_json::from_slice(&bytes).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("extra".into(), Value::Bool(true));
    assert!(ModelPreparationReceiptV1::parse(&serde_json::to_vec(&value).unwrap()).is_err());

    let mut trailing = bytes.clone();
    trailing.extend_from_slice(b"x");
    assert!(ModelPreparationReceiptV1::parse(&trailing).is_err());

    let mut value: Value = serde_json::from_slice(&bytes).unwrap();
    value["artifacts"][0]["conversion_receipt_sha256"] = Value::String("A".repeat(64));
    let mut mutated = serde_json::to_vec(&value).unwrap();
    mutated.push(b'\n');
    assert!(matches!(
        ModelPreparationReceiptV1::parse(&mutated),
        Err(ModelPreparationError::PairMismatch { .. })
    ));

    let cases: Vec<Box<dyn FnOnce(&mut Value)>> = vec![
        Box::new(|value| value["kind"] = Value::String("other".into())),
        Box::new(|value| value["schema_version"] = Value::from(2)),
        Box::new(|value| value["package"] = Value::String("other".into())),
        Box::new(|value| value["recipe"]["id"] = Value::String("other".into())),
        Box::new(|value| value["source"]["revision"] = Value::String("b".repeat(40))),
        Box::new(|value| value["hardware_profile"]["target"] = Value::String("other".into())),
        Box::new(|value| {
            value["hardware_profile"]["observed_unified_memory_bytes"] =
                Value::from(128_u64 * 1024 * 1024 * 1024 - 1)
        }),
        Box::new(|value| {
            value["hardware_profile"]["preflight_available_bytes"] = Value::from(81_914_357_702_u64)
        }),
        Box::new(|value| value["converter"]["version"] = Value::String("v0.1.7".into())),
        Box::new(|value| value["state"] = Value::String("ready".into())),
        Box::new(|value| value["artifacts"].as_array_mut().unwrap().swap(0, 1)),
        Box::new(|value| value["artifacts"][0]["role"] = Value::String("vision_projector".into())),
        Box::new(|value| value["artifacts"][0]["filename"] = Value::String("other.gguf".into())),
        Box::new(|value| value["artifacts"][0]["size"] = Value::from(1)),
        Box::new(|value| value["artifacts"][0]["sha256"] = Value::String("b".repeat(64))),
    ];
    for mutate in cases {
        let mut value: Value = serde_json::from_slice(&bytes).unwrap();
        mutate(&mut value);
        let mut hostile = serde_json::to_vec(&value).unwrap();
        hostile.push(b'\n');
        assert!(ModelPreparationReceiptV1::parse(&hostile).is_err());
    }
}
