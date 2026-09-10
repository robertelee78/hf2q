//! Offline export → reader → checkpoint binder contract, without derivation
//! or device allocation. An ancestor differs deliberately from the source.

use super::*;
use crate::backends::gguf::{types::MetaValue, writer::GgufWriter};
use crate::convert::receipt::{
    receipt_path, ConversionReceipt, ConverterReceipt, ExcludedDsparkReceipt, OutputReceipt,
    PeakChunkBoundReceipt, SourceReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
};
use crate::inference::glp::{validate_glp_for_model, Compatibility};
use mlx_native::gguf::GgufFile;

fn model_fixture(path: &PathBuf, with_receipt: bool) {
    let mut writer = GgufWriter::new(std::fs::File::create(path).unwrap());
    let metadata = [
        ("general.architecture", "deepseek4"),
        ("general.name", "Example Finetune"),
        ("general.organization", "Example"),
        ("general.base_model.0.name", "Other Ancestor"),
        (
            "general.base_model.0.repo_url",
            "https://huggingface.co/other/Other-Ancestor",
        ),
        (
            "general.base_model.0.version",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ),
    ];
    writer.write_header(0, metadata.len() as u64).unwrap();
    for (key, value) in metadata {
        writer
            .write_metadata_kv(key, &MetaValue::String(value.into()))
            .unwrap();
    }
    writer.pad_to_alignment().unwrap();
    drop(writer);
    if !with_receipt {
        return;
    }
    let receipt = ConversionReceipt {
        schema_version: CONVERSION_RECEIPT_SCHEMA_VERSION,
        source: SourceReceipt {
            original_reference: "example/Example-Finetune".into(),
            repository_id: "example/Example-Finetune".into(),
            repository_type: "model".into(),
            canonical_url: "https://huggingface.co/example/Example-Finetune".into(),
            revision: "a".repeat(40),
            filename: None,
            bundle_sha256: "d".repeat(64),
            files: vec![],
        },
        converter: ConverterReceipt {
            package: "hf2q".into(),
            version: "test".into(),
            git_commit: "e".repeat(40),
        },
        quant_selector: "q4_k_m".into(),
        output: OutputReceipt {
            path: path.display().to_string(),
            size: std::fs::metadata(path).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(path).unwrap(),
        },
        excluded_dspark: ExcludedDsparkReceipt {
            tensor_count: 0,
            status: "none_detected".into(),
        },
        peak_chunk_bound: PeakChunkBoundReceipt::default(),
    };
    std::fs::write(receipt_path(path), serde_json::to_vec(&receipt).unwrap()).unwrap();
}

fn export_and_check(with_receipt: bool) {
    use sha2::Digest;
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("model.gguf");
    model_fixture(&model_path, with_receipt);
    let model = GgufFile::open(&model_path).unwrap();
    let checkpoint = CheckpointIdentity::for_model_path(&model_path, &model).unwrap();
    let direction = [0.0f32, 0.6, -0.8];
    let raw: Vec<u8> = direction
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let digest = hex::encode(sha2::Sha256::digest(&raw));
    let output = dir.path().join("candidate.gguf");
    write_glp_gguf(&output, 3, &direction, 1.25, &digest, &checkpoint).unwrap();

    let status = validate_glp_for_model(&output, &model_path, &model).unwrap();
    assert_eq!(
        status,
        if with_receipt {
            Compatibility::Checkpoint
        } else {
            Compatibility::Named
        }
    );
    let exported = GgufFile::open(&output).unwrap();
    let identity = CheckpointIdentity::from_gguf(&exported).unwrap();
    assert_eq!(identity, checkpoint);
    assert_ne!(identity.name.as_deref(), Some("Other Ancestor"));
    assert_eq!(identity.revision, with_receipt.then(|| "a".repeat(40)));

    let vector = GlpVector::load(&output).unwrap();
    assert_eq!(vector.hook_point, GlpHookPoint::FfnOutPreResidual);
    assert_eq!(
        vector.derived_at.as_deref(),
        Some("residual_stream_post_layer")
    );
    assert_eq!(vector.mode, crate::inference::glp::GlpMode::Project);
    assert_eq!(vector.alpha_default, 1.25);
    assert_eq!(vector.content_sha256.as_deref(), Some(digest.as_str()));
    assert_eq!(vector.layers.len(), 1);
    let exported_raw: Vec<u8> = vector.layers[&3]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(exported_raw, raw);
}

#[test]
fn converted_finetune_export_binds_actual_source_not_ancestor() {
    export_and_check(true);
}

#[test]
fn unpinned_finetune_export_binds_own_name_without_invented_revision() {
    export_and_check(false);
}
