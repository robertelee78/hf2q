use super::*;

use crate::convert::receipt::{
    ConversionReceipt, ConverterReceipt, ExcludedDsparkReceipt, OutputReceipt,
    PeakChunkBoundReceipt, SourceReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
};
use crate::core::sha256::compute_file_sha256;
use crate::serve::cache::{CacheManifest, ModelEntry, QuantEntry};
use std::collections::BTreeMap;
use std::io::Write;

fn write_quant_gguf(path: &Path, file_type: u32) {
    let key = b"general.file_type";
    let mut gguf = Vec::new();
    gguf.extend_from_slice(b"GGUF");
    gguf.extend_from_slice(&3_u32.to_le_bytes());
    gguf.extend_from_slice(&0_u64.to_le_bytes());
    gguf.extend_from_slice(&1_u64.to_le_bytes());
    gguf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    gguf.extend_from_slice(key);
    gguf.extend_from_slice(&4_u32.to_le_bytes());
    gguf.extend_from_slice(&file_type.to_le_bytes());
    gguf.resize(256, 0);
    fs::write(path, gguf).unwrap();
}

fn write_q4_k_m_gguf(path: &Path) {
    write_quant_gguf(path, 15);
}

fn receipt_for(artifact: &Path, repository: &str) -> ConversionReceipt {
    ConversionReceipt {
        schema_version: CONVERSION_RECEIPT_SCHEMA_VERSION,
        source: SourceReceipt {
            original_reference: repository.into(),
            repository_id: repository.into(),
            repository_type: "model".into(),
            canonical_url: format!("https://huggingface.co/{repository}"),
            revision: "a".repeat(40),
            filename: None,
            bundle_sha256: "b".repeat(64),
            files: Vec::new(),
        },
        converter: ConverterReceipt {
            package: "hf2q".into(),
            version: "0.1.7".into(),
            git_commit: "c".repeat(40),
        },
        quant_selector: "q4_k_m".into(),
        output: OutputReceipt {
            path: artifact.display().to_string(),
            size: fs::metadata(artifact).unwrap().len(),
            sha256: compute_file_sha256(artifact).unwrap(),
        },
        excluded_dspark: ExcludedDsparkReceipt {
            tensor_count: 0,
            status: "none_detected".into(),
        },
        peak_chunk_bound: PeakChunkBoundReceipt::default(),
    }
}

fn write_receipt(artifact: &Path, receipt: &ConversionReceipt) {
    fs::write(
        crate::convert::receipt::receipt_path(artifact),
        serde_json::to_vec(receipt).unwrap(),
    )
    .unwrap();
}

#[test]
fn schema_v3_receipt_discovers_nested_local_gguf_for_exact_repository() {
    let root = tempfile::tempdir().unwrap();
    let nested = root.path().join("qwen3.8");
    fs::create_dir(&nested).unwrap();
    let artifact = nested.join("model-q4_k_m.gguf");
    write_q4_k_m_gguf(&artifact);
    write_receipt(&artifact, &receipt_for(&artifact, "owner/model"));

    let catalog = LocalArtifactInventory::for_test(vec![root.path().to_path_buf()])
        .discover(Some("owner/model"), None);

    assert_eq!(catalog.artifacts.len(), 1);
    let found = &catalog.artifacts[0];
    assert_eq!(found.repository, "owner/model");
    assert_eq!(found.path, artifact.canonicalize().unwrap());
    assert_eq!(found.filename, "model-q4_k_m.gguf");
    assert_eq!(found.quant, Some(QuantType::Q4_K_M));
    assert!(found.selectable);
    assert_eq!(found.provenance, LocalArtifactProvenance::ConversionReceipt);
}

#[test]
fn q5_k_m_receipt_is_selectable_with_exact_header_identity() {
    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("model-q5_k_m.gguf");
    write_quant_gguf(&artifact, 17);
    let mut receipt = receipt_for(&artifact, "owner/model");
    receipt.quant_selector = "q5_k_m".into();
    write_receipt(&artifact, &receipt);

    let catalog = LocalArtifactInventory::for_test(vec![root.path().to_path_buf()])
        .discover(Some("owner/model"), None);
    assert_eq!(catalog.artifacts.len(), 1);
    assert_eq!(catalog.artifacts[0].quant, Some(QuantType::Q5_K_M));
    assert!(catalog.artifacts[0].selectable);
}

#[test]
fn stale_size_wrong_repo_and_symlinks_never_become_candidates() {
    let root = tempfile::tempdir().unwrap();
    let wrong_repo = root.path().join("wrong.gguf");
    write_q4_k_m_gguf(&wrong_repo);
    write_receipt(&wrong_repo, &receipt_for(&wrong_repo, "other/model"));
    let stale = root.path().join("stale.gguf");
    write_q4_k_m_gguf(&stale);
    write_receipt(&stale, &receipt_for(&stale, "owner/model"));
    fs::OpenOptions::new()
        .append(true)
        .open(&stale)
        .unwrap()
        .write_all(b"changed")
        .unwrap();
    #[cfg(unix)]
    {
        let linked = root.path().join("linked.gguf");
        std::os::unix::fs::symlink(&stale, &linked).unwrap();
        write_receipt(&linked, &receipt_for(&stale, "owner/model"));
    }

    let catalog = LocalArtifactInventory::for_test(vec![root.path().to_path_buf()])
        .discover(Some("owner/model"), None);

    assert!(catalog.artifacts.is_empty());
}

#[test]
fn recorded_output_path_is_never_dereferenced() {
    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("model.gguf");
    write_q4_k_m_gguf(&artifact);
    let mut receipt = receipt_for(&artifact, "owner/model");
    receipt.output.path = "/private/old/location/../../secret.gguf".into();
    write_receipt(&artifact, &receipt);

    let catalog = LocalArtifactInventory::for_test(vec![root.path().to_path_buf()])
        .discover(Some("owner/model"), None);
    assert_eq!(catalog.artifacts.len(), 1);
    assert_eq!(catalog.artifacts[0].path, artifact.canonicalize().unwrap());
}

#[test]
fn unsupported_receipt_quant_is_visible_but_not_selectable() {
    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("model-bf16.gguf");
    write_q4_k_m_gguf(&artifact);
    let mut receipt = receipt_for(&artifact, "owner/model");
    receipt.quant_selector = "bf16".into();
    write_receipt(&artifact, &receipt);

    let catalog = LocalArtifactInventory::for_test(vec![root.path().to_path_buf()])
        .discover(Some("owner/model"), None);
    assert_eq!(catalog.artifacts.len(), 1);
    assert!(!catalog.artifacts[0].selectable);
    assert_eq!(catalog.artifacts[0].quant_hint, "BF16");
    assert!(catalog.artifacts[0]
        .unavailable_reason
        .as_deref()
        .unwrap()
        .contains("mlx-native"));
}

#[test]
fn managed_cache_path_must_equal_canonical_layout() {
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::NamedTempFile::new().unwrap();
    write_q4_k_m_gguf(outside.path());
    let entry = ModelEntry {
        repo_id: "owner/model".into(),
        revision: "a".repeat(40),
        source: None,
        quantizations: BTreeMap::from([(
            "Q4_K_M".into(),
            QuantEntry {
                quant_type: "Q4_K_M".into(),
                gguf_path: outside.path().to_path_buf(),
                mmproj_path: None,
                bytes: fs::metadata(outside.path()).unwrap().len(),
                sha256: compute_file_sha256(outside.path()).unwrap(),
                quantized_at_secs: 1,
                quantized_by_version: "0.1.7".into(),
            },
        )]),
        last_accessed_secs: 1,
        source_shards: Vec::new(),
    };
    let manifest = CacheManifest {
        schema_version: crate::serve::cache::MANIFEST_SCHEMA_VERSION,
        models: BTreeMap::from([("owner/model".into(), entry)]),
    };

    let catalog = LocalArtifactInventory::default()
        .discover(Some("owner/model"), Some((root.path(), &manifest)));
    assert!(catalog.artifacts.is_empty());
    assert!(catalog
        .warnings
        .iter()
        .any(|warning| warning.contains("canonical layout")));
}

#[test]
fn verifier_rejects_post_catalog_digest_and_quant_changes() {
    let root = tempfile::tempdir().unwrap();
    let artifact = root.path().join("model.gguf");
    write_q4_k_m_gguf(&artifact);
    let sha = compute_file_sha256(&artifact).unwrap();
    let bytes = fs::metadata(&artifact).unwrap().len();
    verify_local_artifact(LocalVerificationRequest {
        root: root.path(),
        artifact: &artifact,
        bytes,
        sha256: &sha,
        quant: QuantType::Q4_K_M,
    })
    .unwrap();

    fs::OpenOptions::new()
        .write(true)
        .open(&artifact)
        .unwrap()
        .write_all(b"NOPE")
        .unwrap();
    let error = verify_local_artifact(LocalVerificationRequest {
        root: root.path(),
        artifact: &artifact,
        bytes,
        sha256: &sha,
        quant: QuantType::Q4_K_M,
    })
    .unwrap_err();
    assert!(error.to_string().contains("SHA-256"));
}

#[test]
fn inventory_observes_a_conventional_root_created_after_server_start() {
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("models");
    let inventory = LocalArtifactInventory::for_test(vec![root.clone()]);
    fs::create_dir(&root).unwrap();
    let artifact = root.join("late.gguf");
    write_q4_k_m_gguf(&artifact);
    write_receipt(&artifact, &receipt_for(&artifact, "owner/model"));

    assert_eq!(
        inventory
            .discover(Some("owner/model"), None)
            .artifacts
            .len(),
        1
    );
}

#[test]
fn inventory_bounds_reject_oversized_receipts_deep_paths_and_excess_roots() {
    let root = tempfile::tempdir().unwrap();

    let oversized = root.path().join("oversized.gguf");
    write_q4_k_m_gguf(&oversized);
    fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(crate::convert::receipt::receipt_path(&oversized))
        .unwrap()
        .set_len(MAX_RECEIPT_BYTES + 1)
        .unwrap();

    let mut too_deep = root.path().to_path_buf();
    for depth in 0..=MAX_SCAN_DEPTH {
        too_deep.push(format!("depth-{depth}"));
        fs::create_dir(&too_deep).unwrap();
    }
    let deep_artifact = too_deep.join("deep.gguf");
    write_q4_k_m_gguf(&deep_artifact);
    write_receipt(&deep_artifact, &receipt_for(&deep_artifact, "owner/model"));

    let catalog = LocalArtifactInventory::for_test(vec![root.path().to_path_buf()])
        .discover(Some("owner/model"), None);
    assert!(catalog.artifacts.is_empty());
    assert!(catalog
        .warnings
        .iter()
        .any(|warning| warning.contains("oversized")));

    let excess = vec![PathBuf::from("unused"); MAX_ROOTS];
    assert!(LocalArtifactInventory::for_serve(&excess).is_err());
}

#[test]
fn malformed_receipt_authorities_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let cases = ["schema", "converter", "revision", "digest", "quant"];
    for case in cases {
        let artifact = root.path().join(format!("{case}.gguf"));
        write_q4_k_m_gguf(&artifact);
        let mut receipt = receipt_for(&artifact, "owner/model");
        match case {
            "schema" => receipt.schema_version += 1,
            "converter" => receipt.converter.package = "not-hf2q".into(),
            "revision" => receipt.source.revision = "mutable-main".into(),
            "digest" => receipt.output.sha256 = "not-a-digest".into(),
            "quant" => receipt.quant_selector = "../../Q4_K_M\u{1b}".into(),
            _ => unreachable!(),
        }
        write_receipt(&artifact, &receipt);
    }

    let catalog = LocalArtifactInventory::for_test(vec![root.path().to_path_buf()])
        .discover(Some("owner/model"), None);
    assert!(catalog.artifacts.is_empty());
}

#[test]
fn canonical_managed_cache_artifact_is_selectable() {
    let root = tempfile::tempdir().unwrap();
    let artifact = cache_model_path(root.path(), "owner/model", QuantType::Q4_K_M).unwrap();
    fs::create_dir_all(artifact.parent().unwrap()).unwrap();
    write_q4_k_m_gguf(&artifact);
    let entry = ModelEntry {
        repo_id: "owner/model".into(),
        revision: "a".repeat(40),
        source: None,
        quantizations: BTreeMap::from([(
            "Q4_K_M".into(),
            QuantEntry {
                quant_type: "Q4_K_M".into(),
                gguf_path: artifact.clone(),
                mmproj_path: None,
                bytes: fs::metadata(&artifact).unwrap().len(),
                sha256: compute_file_sha256(&artifact).unwrap(),
                quantized_at_secs: 1,
                quantized_by_version: "0.1.7".into(),
            },
        )]),
        last_accessed_secs: 1,
        source_shards: Vec::new(),
    };
    let manifest = CacheManifest {
        schema_version: crate::serve::cache::MANIFEST_SCHEMA_VERSION,
        models: BTreeMap::from([("owner/model".into(), entry)]),
    };

    let catalog = LocalArtifactInventory::default()
        .discover(Some("owner/model"), Some((root.path(), &manifest)));
    assert_eq!(catalog.artifacts.len(), 1);
    assert_eq!(
        catalog.artifacts[0].provenance,
        LocalArtifactProvenance::ManagedCache
    );
    assert!(catalog.artifacts[0].selectable);
}
