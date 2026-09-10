use super::*;
use crate::backends::gguf::types::MetaValue;
use crate::backends::gguf::writer::GgufWriter;
use crate::convert::receipt::{
    ConverterReceipt, ExcludedDsparkReceipt, OutputReceipt, PeakChunkBoundReceipt, SourceReceipt,
};

fn converted_fixture(path: &Path) {
    let mut writer = GgufWriter::new(std::fs::File::create(path).unwrap());
    let metadata = [
        ("general.architecture", "qwen35"),
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
        (
            "hf2q.source_sha256",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
        ),
    ];
    writer.write_header(0, metadata.len() as u64).unwrap();
    for (key, value) in metadata {
        writer
            .write_metadata_kv(key, &MetaValue::String(value.into()))
            .unwrap();
    }
    writer.pad_to_alignment().unwrap();
}

fn receipt(path: &Path) -> ConversionReceipt {
    ConversionReceipt {
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
            // Never follow this path; receipts move alongside their output.
            path: "/must-not-be-read/model.gguf".into(),
            size: std::fs::metadata(path).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(path).unwrap(),
        },
        excluded_dspark: ExcludedDsparkReceipt {
            tensor_count: 0,
            status: "none_detected".into(),
        },
        peak_chunk_bound: PeakChunkBoundReceipt::default(),
    }
}

fn save_receipt(path: &Path, receipt: &ConversionReceipt) {
    std::fs::write(receipt_path(path), serde_json::to_vec(receipt).unwrap()).unwrap();
}

#[test]
fn converted_finetune_uses_verified_source_and_rejects_ancestor_vector() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    converted_fixture(&path);
    save_receipt(&path, &receipt(&path));
    let model = GgufFile::open(&path).unwrap();
    let actual = CheckpointIdentity::for_model_path(&path, &model).unwrap();
    let vector = CheckpointIdentity {
        name: Some("Example-Finetune".into()),
        organization: None,
        repository: Some("example/Example-Finetune".into()),
        revision: Some("a".repeat(40)),
    };
    assert_eq!(
        vector.validate_for(&actual).unwrap(),
        super::super::compatibility::Compatibility::Checkpoint
    );
    let ancestor = CheckpointIdentity::from_gguf(&model).unwrap();
    assert!(ancestor.validate_for(&actual).is_err());
}

#[test]
fn missing_receipt_leaves_actual_name_known_and_checkpoint_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    converted_fixture(&path);
    let model = GgufFile::open(&path).unwrap();
    let actual = CheckpointIdentity::for_model_path(&path, &model).unwrap();
    assert_eq!(actual.name.as_deref(), Some("Example Finetune"));
    assert!(actual.repository.is_none());
    assert!(actual.revision.is_none());
    let mut vector = CheckpointIdentity {
        name: Some("Example-Finetune".into()),
        ..Default::default()
    };
    assert_eq!(
        vector.validate_for(&actual).unwrap(),
        super::super::compatibility::Compatibility::Named
    );
    vector.revision = Some("a".repeat(40));
    assert!(vector.validate_for(&actual).is_err());
    assert!(CheckpointIdentity::from_gguf(&model)
        .unwrap()
        .validate_for(&actual)
        .is_err());
}

#[test]
fn corrupt_output_receipt_or_source_bundle_cannot_supply_checkpoint_identity() {
    for kind in [
        "output_sha",
        "output_size",
        "bundle",
        "repository",
        "schema",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        converted_fixture(&path);
        let mut record = receipt(&path);
        match kind {
            "output_sha" => record.output.sha256 = "0".repeat(64),
            "output_size" => record.output.size += 1,
            "bundle" => record.source.bundle_sha256 = "f".repeat(64),
            "repository" => record.source.repository_id = "different/Example-Finetune".into(),
            "schema" => record.schema_version += 1,
            _ => unreachable!(),
        }
        save_receipt(&path, &record);
        let model = GgufFile::open(&path).unwrap();
        assert!(
            CheckpointIdentity::for_model_path(&path, &model).is_err(),
            "{kind}"
        );
    }
}

#[test]
fn changed_same_length_model_invalidates_cached_output_verification() {
    use std::io::{Seek, Write};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    converted_fixture(&path);
    save_receipt(&path, &receipt(&path));
    let model = GgufFile::open(&path).unwrap();
    CheckpointIdentity::for_model_path(&path, &model).unwrap();
    let mut file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(std::io::SeekFrom::End(-1)).unwrap();
    file.write_all(&[255]).unwrap();
    file.sync_all().unwrap();
    assert!(CheckpointIdentity::for_model_path(&path, &model).is_err());
}

#[test]
fn a_malformed_adjacent_receipt_is_not_silently_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.gguf");
    converted_fixture(&path);
    std::fs::write(receipt_path(&path), b"{bad-json").unwrap();
    let model = GgufFile::open(&path).unwrap();
    assert!(CheckpointIdentity::for_model_path(&path, &model).is_err());
}
