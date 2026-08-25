use super::*;

fn hosted(quant: QuantType, filename: &str) -> HubGgufArtifact {
    HubGgufArtifact {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        filename: filename.into(),
        bytes: 1024,
        sha256: "b".repeat(64),
        quant_hint: Some(quant.as_str().into()),
        role: "text_model".into(),
        selectable: true,
        unavailable_reason: None,
    }
}

fn write_quant_gguf(path: &Path, file_type: u32) {
    write_quant_gguf_with_metadata(path, file_type, &[]);
}

fn write_quant_gguf_with_metadata(path: &Path, file_type: u32, strings: &[(&str, &str)]) {
    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    let mut gguf = Vec::new();
    gguf.extend_from_slice(b"GGUF");
    gguf.extend_from_slice(&3_u32.to_le_bytes());
    gguf.extend_from_slice(&0_u64.to_le_bytes());
    gguf.extend_from_slice(&(1 + strings.len() as u64).to_le_bytes());
    write_string(&mut gguf, "general.file_type");
    gguf.extend_from_slice(&4_u32.to_le_bytes());
    gguf.extend_from_slice(&file_type.to_le_bytes());
    for (key, value) in strings {
        write_string(&mut gguf, key);
        gguf.extend_from_slice(&8_u32.to_le_bytes());
        write_string(&mut gguf, value);
    }
    let padded = ((gguf.len().max(256) + 31) / 32) * 32;
    gguf.resize(padded, 0);
    fs::write(path, gguf).unwrap();
}

fn write_structurally_valid_mmproj(path: &Path) {
    use crate::ir::DType;
    use crate::models::vit::convert::VitTensor;
    use crate::models::vit::VisionConfig;
    use std::collections::HashMap;

    let config = VisionConfig {
        hidden_size: 4,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        patch_size: 2,
        image_size: 4,
        intermediate_size: 8,
        layer_norm_eps: 1e-5,
        projector_type: "mlp".into(),
        projection_dim: Some(4),
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
        image_min_pixels: None,
        image_max_pixels: None,
        spatial_merge_size: None,
        deepstack_visual_indexes: None,
        temporal_patch_size: None,
    };
    let mut tensors = HashMap::new();
    for name in [
        "v.patch_embd.weight",
        "v.position_embd.weight",
        "v.blk.0.attn_q.weight",
        "v.blk.0.attn_k.weight",
        "v.blk.0.attn_v.weight",
        "v.blk.0.attn_out.weight",
        "v.blk.0.attn_norm.weight",
        "mm.0.weight",
    ] {
        tensors.insert(
            name.to_owned(),
            VitTensor {
                gguf_name: name.to_owned(),
                shape: vec![1],
                dtype: DType::F16,
                data: vec![0_u8; 2],
            },
        );
    }
    crate::models::vit::gguf_emit::write_mmproj_gguf(path, &config, &tensors).unwrap();
}

fn conversion_receipt(
    artifact: &Path,
    repository: &str,
    revision: &str,
    quant: &str,
) -> crate::convert::receipt::ConversionReceipt {
    use crate::convert::receipt::{
        ConverterReceipt, ExcludedDsparkReceipt, OutputReceipt, PeakChunkBoundReceipt,
        SourceReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
    };

    crate::convert::receipt::ConversionReceipt {
        schema_version: CONVERSION_RECEIPT_SCHEMA_VERSION,
        source: SourceReceipt {
            original_reference: repository.into(),
            repository_id: repository.into(),
            repository_type: "model".into(),
            canonical_url: format!("https://huggingface.co/{repository}"),
            revision: revision.into(),
            filename: None,
            bundle_sha256: "d".repeat(64),
            files: Vec::new(),
        },
        converter: ConverterReceipt {
            package: "hf2q".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            git_commit: "e".repeat(40),
        },
        quant_selector: quant.into(),
        output: OutputReceipt {
            path: artifact.display().to_string(),
            size: fs::metadata(artifact).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(artifact).unwrap(),
        },
        excluded_dspark: ExcludedDsparkReceipt {
            tensor_count: 0,
            status: "none_detected".into(),
        },
        peak_chunk_bound: PeakChunkBoundReceipt::default(),
    }
}

fn write_conversion_receipt(artifact: &Path, receipt: &crate::convert::receipt::ConversionReceipt) {
    fs::write(
        crate::convert::receipt::receipt_path(artifact),
        serde_json::to_vec(receipt).unwrap(),
    )
    .unwrap();
}

#[test]
fn exact_hosted_quant_wins_and_missing_exact_falls_back_to_native_conversion() {
    let choices = vec![
        hosted(QuantType::Q4_K_M, "model-q4_k_m.gguf"),
        hosted(QuantType::Q8_0, "model-q8_0.gguf"),
    ];
    assert_eq!(
        select_hosted(&choices, Some(QuantType::Q8_0), QuantType::Q4_K_M)
            .unwrap()
            .unwrap()
            .quant_hint
            .as_deref(),
        Some("Q8_0")
    );
    assert!(
        select_hosted(&choices, Some(QuantType::Q6_K), QuantType::Q4_K_M)
            .unwrap()
            .is_none()
    );

    let ambiguous = vec![
        hosted(QuantType::Q8_0, "model-a-q8_0.gguf"),
        hosted(QuantType::Q8_0, "model-b-q8_0.gguf"),
    ];
    let error = select_hosted(&ambiguous, Some(QuantType::Q8_0), QuantType::Q8_0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("model-a-q8_0.gguf"));
    assert!(error.contains("model-b-q8_0.gguf"));
}

#[test]
fn bare_repository_uses_setup_quant_while_an_exact_suffix_remains_authoritative() {
    let hardware = HardwareProfile {
        chip_model: "test".into(),
        total_memory_bytes: 16 << 30,
        available_memory_bytes: 14 << 30,
        performance_cores: 4,
        efficiency_cores: 4,
        total_cores: 8,
        memory_bandwidth_gbs: 100.0,
    };
    assert_eq!(
        repository_recommended_quant(None, Some("q6_k"), &hardware).unwrap(),
        QuantType::Q6_K
    );
    assert_eq!(
        repository_recommended_quant(Some(QuantType::Q4_K_M), Some("q6_k"), &hardware,).unwrap(),
        QuantType::Q4_K_M
    );
}

#[test]
fn non_runtime_setup_quant_fails_only_when_remote_fallback_is_needed() {
    let hardware = HardwareProfile {
        chip_model: "test".into(),
        total_memory_bytes: 16 << 30,
        available_memory_bytes: 14 << 30,
        performance_cores: 4,
        efficiency_cores: 4,
        total_cores: 8,
        memory_bandwidth_gbs: 100.0,
    };
    let error = repository_recommended_quant(None, Some("q5_0"), &hardware).unwrap_err();
    let message = error.to_string();
    assert!(message.contains("cannot drive automatic repository serving"));
    assert!(message.contains("repository:QUANT"));
}

#[test]
fn exact_incompatible_hosted_quant_falls_through_but_transport_failure_does_not() {
    let choices = vec![hosted(QuantType::Q8_0, "model-q8_0.gguf")];
    let mut warnings = Vec::new();
    let selected = select_compatible_hosted(
        &choices,
        Some(QuantType::Q8_0),
        QuantType::Q8_0,
        |_| {
            Err(DownloadError::IncompatibleHostedGguf {
                reason: "authenticated header has the wrong architecture".into(),
            })
        },
        &mut warnings,
    )
    .unwrap();
    assert!(
        selected.is_none(),
        "native exact-quant fallback must remain reachable"
    );
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("wrong architecture")));

    let error = select_compatible_hosted(
        &choices,
        Some(QuantType::Q8_0),
        QuantType::Q8_0,
        |_| {
            Err(DownloadError::DownloadFailed {
                reason: "transport interrupted".into(),
            })
        },
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("transport interrupted"));
}

#[test]
fn unqualified_hosted_selection_uses_recommendation_then_nearest_lower() {
    let choices = vec![
        hosted(QuantType::Q3_K_M, "model-q3_k_m.gguf"),
        hosted(QuantType::Q5_K_M, "model-q5_k_m.gguf"),
        hosted(QuantType::Q8_0, "model-q8_0.gguf"),
    ];
    assert_eq!(
        select_hosted(&choices, None, QuantType::Q6_K)
            .unwrap()
            .unwrap()
            .quant_hint
            .as_deref(),
        Some("Q5_K_M")
    );

    let q2_only = vec![hosted(QuantType::Q2_K, "model-q2_k.gguf")];
    assert_eq!(
        select_hosted(&q2_only, None, QuantType::Q4_K_M)
            .unwrap()
            .unwrap()
            .quant_hint
            .as_deref(),
        Some("Q2_K")
    );

    let q8_only = vec![hosted(QuantType::Q8_0, "model-q8_0.gguf")];
    assert!(select_hosted(&q8_only, None, QuantType::Q4_K_M)
        .unwrap()
        .is_none());
}

#[test]
fn automatic_admission_uses_exact_artifact_bytes_and_runtime_headroom() {
    let gib = 1024_u64 * 1024 * 1024;
    assert!(automatic_artifact_admissible(16 * gib, 32 * gib, 25 * gib));
    assert!(!automatic_artifact_admissible(30 * gib, 32 * gib, 31 * gib));
    assert!(!automatic_artifact_admissible(53 * gib, 64 * gib, 51 * gib));
}

#[test]
fn native_fallback_uses_a_fail_closed_output_bound_for_automatic_admission() {
    let gib = 1024_u64 * 1024 * 1024;
    assert_eq!(
        select_native_fallback_quant(None, QuantType::Q4_K_M, Some(11 * gib), 16 * gib, 14 * gib,)
            .unwrap(),
        QuantType::Q4_K_M
    );
    assert!(select_native_fallback_quant(
        None,
        QuantType::Q4_K_M,
        Some(15 * gib),
        16 * gib,
        14 * gib,
    )
    .is_err());
    assert_eq!(
        select_native_fallback_quant(
            Some(QuantType::Q4_K_M),
            QuantType::Q4_K_M,
            None,
            16 * gib,
            14 * gib,
        )
        .unwrap(),
        QuantType::Q4_K_M
    );
}

#[test]
fn automatic_native_selection_steps_down_using_exact_product_sizes() {
    let gib = 1024_u64.pow(3);
    let mut visited = Vec::new();
    let mut warnings = Vec::new();
    let (quant, bytes) = select_native_quant_from_exact_plans(
        None,
        QuantType::Q8_0,
        24 * gib,
        20 * gib,
        |quant| {
            visited.push(quant);
            Ok(match quant {
                QuantType::Q8_0 => 22 * gib,
                QuantType::Q6_K => 18 * gib,
                _ => 12 * gib,
            })
        },
        &mut warnings,
    )
    .unwrap();
    assert_eq!((quant, bytes), (QuantType::Q6_K, 18 * gib));
    assert_eq!(visited, vec![QuantType::Q8_0, QuantType::Q6_K]);
    assert_eq!(warnings.len(), 1);

    let (exact, _) = select_native_quant_from_exact_plans(
        Some(QuantType::Q8_0),
        QuantType::Q4_K_M,
        1,
        1,
        |_| Ok(22 * gib),
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(exact, QuantType::Q8_0, "an explicit quant is never changed");
}

#[test]
fn source_only_native_fallback_refuses_an_upper_bound_that_does_not_fit() {
    let gib = 1024_u64 * 1024 * 1024;
    assert!(select_native_fallback_quant(
        None,
        QuantType::Q4_K_M,
        Some(102 * gib),
        16 * gib,
        13 * gib,
    )
    .is_err());
    assert_eq!(
        select_native_fallback_quant(
            Some(QuantType::Q4_K_M),
            QuantType::Q4_K_M,
            Some(102 * gib),
            16 * gib,
            13 * gib,
        )
        .unwrap(),
        QuantType::Q4_K_M
    );
}

#[test]
fn source_only_native_fallback_refuses_without_bounded_size_evidence() {
    let error = select_native_fallback_quant(None, QuantType::Q4_K_M, None, u64::MAX, u64::MAX)
        .unwrap_err()
        .to_string();
    assert!(error.contains("bounded native output plan"), "{error}");
}

#[test]
fn native_multimodal_disk_plan_includes_the_f16_projector_before_transfer() {
    let gib = 1024_u64 * 1024 * 1024;
    let source = 50 * gib;
    let text = 18 * gib;

    assert_eq!(
        planned_native_product_bytes(source, text, true, false, false),
        68 * gib
    );
    assert_eq!(
        planned_native_product_bytes(source, text, true, true, false),
        text
    );
    assert_eq!(
        planned_native_product_bytes(source, text, true, false, true),
        source
    );
    assert_eq!(
        planned_native_product_bytes(source, text, false, false, false),
        text
    );
}

#[test]
fn local_qwen_tensor_layout_admission_matches_runtime_role_support() {
    use mlx_native::ops::quantized_matmul_ggml::GgmlType;

    assert!(
        local_runtime_tensor_incompatibility("qwen35", "token_embd.weight", GgmlType::Q3_K,)
            .is_some()
    );
    assert!(
        local_runtime_tensor_incompatibility("qwen35", "token_embd.weight", GgmlType::Q6_K,)
            .is_none()
    );
    assert!(local_runtime_tensor_incompatibility(
        "qwen35",
        "blk.0.ffn_gate.weight",
        GgmlType::Q5_1,
    )
    .is_some());
    assert!(local_runtime_tensor_incompatibility(
        "qwen35moe",
        "blk.0.ffn_gate_exps.weight",
        GgmlType::Q3_K,
    )
    .is_none());
    assert!(
        local_runtime_tensor_incompatibility("qwen35", "blk.0.attn_q.weight", GgmlType::F16,)
            .is_some()
    );
    assert!(local_runtime_tensor_incompatibility(
        "qwen3_vl",
        "blk.0.attn_q.weight",
        GgmlType::Q3_K,
    )
    .is_none());
}

#[test]
fn production_manual_structural_admission_never_hashes_the_text_payload() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let model = tempfile::tempdir().unwrap();
    let text = model.path().join("APEX-Q4_K_M.gguf");
    let bytes = crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text);
    symlink(model.path(), library.path().join("qwen3.6")).unwrap();

    // A deliberately incorrect catalog digest proves that the interactive
    // structural-admission path neither computes nor compares full SHA-256.
    // Publication through explicit --output retains the exact-digest gate.
    let mut artifact = hosted(QuantType::Q4_K_M, "gguf/APEX-Q4_K_M.gguf");
    artifact.bytes = bytes;
    artifact.sha256 = "0".repeat(64);
    let mut warnings = Vec::new();
    let mut events = Vec::new();
    let selected = find_best_matching_loose_with_progress(
        &[artifact],
        Some(QuantType::Q4_K_M),
        &[library.path().to_path_buf()],
        &[],
        &mut warnings,
        &mut |event| events.push(event),
    )
    .unwrap()
    .expect("compatible GGUF under a direct model-directory link must be reused");

    assert_eq!(selected.path, text.canonicalize().unwrap());
    assert_eq!(selected.artifact.sha256, "0".repeat(64));
    assert!(warnings.is_empty(), "{warnings:?}");
    assert!(events.iter().any(|event| matches!(
        event,
        StartupEvent::LocalCandidate {
            origin: StartupOrigin::ManualStructuralMatch,
            ..
        }
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        StartupEvent::VerifyStart { .. } | StartupEvent::VerifyProgress { .. }
    )));
}

#[test]
fn production_structural_admission_rejects_same_quant_size_ambiguity_even_on_filename_match() {
    let directory = tempfile::tempdir().unwrap();
    let local = directory.path().join("preferred-q4_k_m.gguf");
    let bytes = crate::input::hf_download::tests::write_complete_qwen_test_gguf(&local);
    let mut preferred = hosted(QuantType::Q4_K_M, "preferred-q4_k_m.gguf");
    preferred.bytes = bytes;
    let mut other = hosted(QuantType::Q4_K_M, "other-q4_k_m.gguf");
    other.bytes = bytes;
    other.sha256 = "c".repeat(64);

    let mut warnings = Vec::new();
    let selected = find_best_matching_loose_with_progress(
        &[preferred, other],
        Some(QuantType::Q4_K_M),
        &[directory.path().to_path_buf()],
        &[],
        &mut warnings,
        &mut |_| {},
    )
    .unwrap();

    assert!(selected.is_none());
    assert!(warnings.iter().any(|warning| {
        warning.contains("structurally ambiguous")
            && warning.contains("filenames are hints, not identity authority")
    }));
}

#[test]
fn malformed_projector_binding_is_a_warning_and_text_only_fallback() {
    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("model-q4_k_m.gguf");
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", "not-a-sha256"),
        ],
    );
    let candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: "0".repeat(64),
        path: text.clone(),
        quant: QuantType::Q4_K_M,
        origin: "manual_structural".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let mut candidate = candidate;
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![],
    };
    let mut warnings = Vec::new();
    let resolved = best_effort_projector_with_catalog(
        &mut candidate,
        &[directory.path().to_path_buf()],
        &catalog,
        true,
        &mut warnings,
    );

    assert!(resolved.is_none());
    assert!(warnings.iter().any(|warning| {
        warning.contains("automatic mmproj preparation failed")
            && warning.contains("hf2q.mmproj_sha256 must be exactly 64 hexadecimal characters")
    }));
}

#[test]
fn in_place_markerless_structural_match_prepares_exact_catalog_projector() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let model = tempfile::tempdir().unwrap();
    let companions = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let text = model.path().join("APEX-Q4_K_M.gguf");
    let text_bytes = crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text);
    symlink(model.path(), library.path().join("qwen3.6")).unwrap();

    let projector_source = companions.path().join("operator-mmproj.gguf");
    fs::write(&projector_source, b"exact projector fixture").unwrap();
    let projector_bytes = fs::metadata(&projector_source).unwrap().len();
    let projector_sha = crate::core::sha256::compute_file_sha256(&projector_source).unwrap();
    let revision = "a".repeat(40);
    let text_artifact = HubGgufArtifact {
        repository: "owner/model".into(),
        revision: revision.clone(),
        filename: "gguf/APEX-Q4_K_M.gguf".into(),
        bytes: text_bytes,
        // Text serving is structural, so this intentionally need not match.
        sha256: "0".repeat(64),
        quant_hint: Some("Q4_K_M".into()),
        role: "text_model".into(),
        selectable: true,
        unavailable_reason: None,
    };
    let projector_artifact = HubGgufArtifact {
        repository: "owner/model".into(),
        revision: revision.clone(),
        filename: "gguf/mmproj-qwen36-F16.gguf".into(),
        bytes: projector_bytes,
        sha256: projector_sha,
        quant_hint: None,
        role: "companion".into(),
        selectable: false,
        unavailable_reason: Some("vision projector companion; not a text model".into()),
    };
    let expected_projector_sha = projector_artifact.sha256.clone();
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision,
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![text_artifact, projector_artifact],
    };
    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let mut events = Vec::new();
    let mut catalog = Some(catalog);
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: "owner/model".into(),
            quant: Some(QuantType::Q4_K_M),
        },
        None,
        &[
            library.path().to_path_buf(),
            companions.path().to_path_buf(),
        ],
        &mut cache,
        &hardware,
        true,
        None,
        &mut |event| events.push(event),
        |_| Ok(catalog.take().expect("catalog resolver is called once")),
    )
    .unwrap();

    assert_eq!(resolved.gguf_path, text.canonicalize().unwrap());
    let prepared_projector = model.path().join("mmproj-qwen36-F16.gguf");
    assert_eq!(
        resolved.mmproj_path.as_deref(),
        Some(prepared_projector.canonicalize().unwrap().as_path())
    );
    assert_eq!(
        fs::read(prepared_projector).unwrap(),
        b"exact projector fixture"
    );
    assert!(!resolved.track_success_history);
    assert!(resolved.activation_authority.is_some());
    assert_eq!(
        resolved.mmproj_sha256.as_deref(),
        Some(expected_projector_sha.as_str())
    );
    assert!(resolved.mmproj_activation_authority.is_some());
    assert!(events.iter().any(|event| matches!(
        event,
        StartupEvent::ProjectorPrepare { filename, bytes }
            if filename == "mmproj-qwen36-F16.gguf" && *bytes == projector_bytes
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        StartupEvent::VerifyStart { artifact, .. } if artifact == "text GGUF"
    )));
}

#[test]
fn fresh_hosted_projector_snapshot_is_retained_from_its_exact_blob() {
    use sha2::Digest;
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("model-q4_k_m.gguf");
    write_quant_gguf_with_metadata(&text, 15, &[("general.architecture", "qwen3vl")]);
    let revision = "a".repeat(40);
    let projector_bytes = b"fresh hosted projector";
    let projector_sha = hex::encode(sha2::Sha256::digest(projector_bytes));
    let repository_cache = directory.path().join("models--owner--model");
    let blob = repository_cache.join("blobs").join(&projector_sha);
    let snapshot = repository_cache
        .join("snapshots")
        .join(&revision)
        .join("gguf")
        .join("mmproj-model-f16.gguf");
    fs::create_dir_all(blob.parent().unwrap()).unwrap();
    fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
    fs::write(&blob, projector_bytes).unwrap();
    symlink(Path::new("../../../blobs").join(&projector_sha), &snapshot).unwrap();

    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: revision.clone(),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: "0".repeat(64),
        path: text.clone(),
        quant: QuantType::Q4_K_M,
        origin: "hf_hub_cache_structural".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let text_authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![HubGgufArtifact {
            repository: "owner/model".into(),
            revision,
            filename: "gguf/mmproj-model-f16.gguf".into(),
            bytes: projector_bytes.len() as u64,
            sha256: projector_sha.clone(),
            quant_hint: None,
            role: "companion".into(),
            selectable: false,
            unavailable_reason: Some("vision projector companion".into()),
        }],
    };
    let mut events = Vec::new();
    let prepared = prepare_cached_projector_in_place_with_sources(
        &mut candidate,
        &text_authority,
        &catalog,
        &mut |event| events.push(event),
        |_| None,
        |_| Ok(snapshot.clone()),
    )
    .unwrap()
    .expect("fresh snapshot must authenticate into its retained blob");

    assert_eq!(prepared, blob.canonicalize().unwrap());
    assert!(!fs::symlink_metadata(&prepared)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(
        candidate
            .projector
            .as_ref()
            .map(|binding| binding.2.as_str()),
        Some(projector_sha.as_str())
    );
    assert!(crate::core::bounded_file::StableRegularFile::open_exact(
        &prepared,
        projector_bytes.len() as u64,
    )
    .unwrap()
    .is_some());
}

#[test]
fn in_place_manual_structural_match_reuses_present_sibling_mmproj() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let model = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let text = model.path().join("Qwen3.8-Q4_K_M.gguf");
    let text_bytes = crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text);
    let projector = model.path().join("mmproj-Qwen3.8-F16.gguf");
    write_structurally_valid_mmproj(&projector);
    let projector_sha = crate::core::sha256::compute_file_sha256(&projector).unwrap();
    symlink(model.path(), library.path().join("qwen3.8")).unwrap();
    let revision = "a".repeat(40);
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![HubGgufArtifact {
            repository: "owner/model".into(),
            revision,
            filename: "Qwen3.8-Q4_K_M.gguf".into(),
            bytes: text_bytes,
            sha256: "0".repeat(64),
            quant_hint: Some("Q4_K_M".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        }],
    };
    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let mut catalog = Some(catalog);
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: "owner/model".into(),
            quant: Some(QuantType::Q4_K_M),
        },
        None,
        &[library.path().to_path_buf()],
        &mut cache,
        &hardware,
        true,
        None,
        &mut |_| {},
        |_| Ok(catalog.take().unwrap()),
    )
    .unwrap();

    assert_eq!(
        resolved.mmproj_path.as_deref(),
        Some(projector.canonicalize().unwrap().as_path())
    );
    assert_eq!(
        resolved.mmproj_sha256.as_deref(),
        Some(projector_sha.as_str())
    );
    assert!(resolved.mmproj_activation_authority.is_some());
    assert!(resolved.warnings.is_empty());
}

#[test]
fn direct_file_symlink_pair_is_discovered_and_retained_in_place() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let text_target = payloads.path().join("Qwen3.8-Q4_K_M.gguf");
    let text_bytes = crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text_target);
    let projector_target = payloads.path().join("mmproj-Qwen3.8-F16.gguf");
    write_structurally_valid_mmproj(&projector_target);
    let text = library.path().join("Qwen3.8-Q4_K_M.gguf");
    let projector = library.path().join("mmproj-Qwen3.8-F16.gguf");
    symlink(&text_target, &text).unwrap();
    symlink(&projector_target, &projector).unwrap();
    let projector_sha = crate::core::sha256::compute_file_sha256(&projector_target).unwrap();
    let revision = "a".repeat(40);
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![HubGgufArtifact {
            repository: "owner/model".into(),
            revision,
            filename: "Qwen3.8-Q4_K_M.gguf".into(),
            bytes: text_bytes,
            sha256: "0".repeat(64),
            quant_hint: Some("Q4_K_M".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        }],
    };
    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let mut catalog = Some(catalog);
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: "owner/model".into(),
            quant: Some(QuantType::Q4_K_M),
        },
        None,
        &[library.path().to_path_buf()],
        &mut cache,
        &hardware,
        true,
        None,
        &mut |_| {},
        |_| Ok(catalog.take().unwrap()),
    )
    .unwrap();

    assert_eq!(resolved.gguf_path, text);
    assert_eq!(resolved.mmproj_path.as_deref(), Some(projector.as_path()));
    assert_eq!(
        resolved.mmproj_sha256.as_deref(),
        Some(projector_sha.as_str())
    );
    assert!(resolved.activation_authority.is_some());
    assert!(resolved.mmproj_activation_authority.is_some());
}

#[test]
fn symlinked_conversion_receipt_wins_without_hosted_gguf_artifacts() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let revision = "a".repeat(40);
    let text_target = payloads.path().join("converted-q4_k_m.gguf");
    crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text_target);
    let projector_target = payloads.path().join("converted-q4_k_m-mmproj.gguf");
    write_structurally_valid_mmproj(&projector_target);
    write_conversion_receipt(
        &text_target,
        &conversion_receipt(&text_target, repository, &revision, "q4_k_m"),
    );
    write_conversion_receipt(
        &projector_target,
        &conversion_receipt(&projector_target, repository, &revision, "f16-mmproj"),
    );
    let text = library.path().join("linked-q4_k_m.gguf");
    let projector = library.path().join("linked-q4_k_m-mmproj.gguf");
    symlink(&text_target, &text).unwrap();
    symlink(&projector_target, &projector).unwrap();
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let text_authority = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    mark_successful_use(
        repository,
        &revision,
        QuantType::Q4_K_M,
        &text,
        &text_authority,
        &mut cache,
    )
    .unwrap();
    let usage_sidecar = sidecar_path(&text);
    assert!(usage_sidecar.is_file());
    let bound = scan_bindings(&[library.path().to_path_buf()], Some(repository)).unwrap();
    let [bound] = bound.as_slice() else {
        panic!("one receipt-bound logical model expected: {bound:?}");
    };
    assert_eq!(bound.path, text);
    assert!(bound.last_used_at_secs > 0);
    assert!(bound.sidecar.is_none());

    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut events = Vec::new();
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: repository.into(),
            quant: Some(QuantType::Q4_K_M),
        },
        None,
        &[library.path().to_path_buf()],
        &mut cache,
        &hardware,
        true,
        None,
        &mut |event| events.push(event),
        |_| panic!("successfully used receipt-bound symlink must not query Hub metadata"),
    )
    .unwrap();

    assert_eq!(resolved.gguf_path, text);
    assert_eq!(resolved.mmproj_path.as_deref(), Some(projector.as_path()));
    assert_eq!(resolved.origin, "local_receipt");
    assert!(resolved.activation_authority.is_some());
    assert!(resolved.mmproj_activation_authority.is_some());
    assert!(!events.iter().any(|event| matches!(
        event,
        StartupEvent::HubMetadata { .. }
            | StartupEvent::HostedDownload { .. }
            | StartupEvent::NativeConversion { .. }
    )));
}

#[test]
fn receipt_history_sidecar_hub_filename_cannot_select_projector() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let revision = "a".repeat(40);
    let target = payloads.path().join("converted-q4_k_m.gguf");
    write_quant_gguf(&target, 15);
    write_conversion_receipt(
        &target,
        &conversion_receipt(&target, repository, &revision, "q4_k_m"),
    );
    let logical = models.path().join("linked-q4_k_m.gguf");
    symlink(&target, &logical).unwrap();
    let target_sha = crate::core::sha256::compute_file_sha256(&target).unwrap();
    write_binding(
        &sidecar_path(&logical),
        &ManagedBinding {
            schema_version: SCHEMA_VERSION,
            repository: repository.into(),
            revision: revision.clone(),
            quant: "Q4_K_M".into(),
            origin: "local_receipt".into(),
            materialized_at_secs: 1,
            last_used_at_secs: 42,
            artifact: ArtifactBinding {
                local_filename: "linked-q4_k_m.gguf".into(),
                hub_filename: "forged-text.gguf".into(),
                bytes: fs::metadata(&target).unwrap().len(),
                sha256: target_sha,
            },
            projector: None,
        },
    )
    .unwrap();

    let candidates = scan_bindings(&[models.path().to_path_buf()], Some(repository)).unwrap();
    let [candidate] = candidates.as_slice() else {
        panic!("one receipt-bound candidate expected: {candidates:?}");
    };
    assert_eq!(candidate.last_used_at_secs, 42);
    assert!(candidate.sidecar.is_none());

    let companion = |filename: &str| HubGgufArtifact {
        repository: repository.into(),
        revision: revision.clone(),
        filename: filename.into(),
        bytes: 1,
        sha256: "f".repeat(64),
        quant_hint: None,
        role: "companion".into(),
        selectable: false,
        unavailable_reason: Some("vision projector companion; not a text model".into()),
    };
    let forged_match = companion("forged-text-mmproj.gguf");
    let other = companion("another-text-mmproj.gguf");
    assert!(
        select_projector_companion(candidate, vec![&forged_match, &other], None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn symlinked_conversion_receipt_explicit_output_clears_source_identity() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let revision = "a".repeat(40);
    let text_target = payloads.path().join("converted-q4_k_m.gguf");
    crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text_target);
    write_conversion_receipt(
        &text_target,
        &conversion_receipt(&text_target, repository, &revision, "q4_k_m"),
    );
    let text = library.path().join("linked-q4_k_m.gguf");
    let output = library.path().join("published-q4_k_m.gguf");
    symlink(&text_target, &text).unwrap();
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: repository.into(),
        revision: revision.clone(),
        requires_projector: false,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: Vec::new(),
    };
    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let mut events = Vec::new();
    let mut catalog = Some(catalog);
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: repository.into(),
            quant: Some(QuantType::Q4_K_M),
        },
        Some(&output),
        &[library.path().to_path_buf()],
        &mut cache,
        &hardware,
        false,
        None,
        &mut |event| events.push(event),
        |_| Ok(catalog.take().unwrap()),
    )
    .unwrap();

    assert_eq!(resolved.gguf_path, output);
    assert_eq!(resolved.origin, "local_adoption");
    assert!(resolved.activation_authority.is_some());
    assert!(resolved.mmproj_activation_authority.is_none());
    assert!(!events.iter().any(|event| matches!(
        event,
        StartupEvent::HostedDownload { .. } | StartupEvent::NativeConversion { .. }
    )));
}

#[test]
fn symlinked_managed_sidecar_is_not_local_authority() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let text = models.path().join("model-q4_k_m.gguf");
    write_quant_gguf(&text, 15);
    let binding = ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        quant: "Q4_K_M".into(),
        origin: "forged-link".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 2,
        artifact: ArtifactBinding {
            local_filename: "model-q4_k_m.gguf".into(),
            hub_filename: "model-q4_k_m.gguf".into(),
            bytes: fs::metadata(&text).unwrap().len(),
            sha256: "0".repeat(64),
        },
        projector: None,
    };
    let outside_sidecar = outside.path().join("binding.json");
    write_binding(&outside_sidecar, &binding).unwrap();
    symlink(&outside_sidecar, sidecar_path(&text)).unwrap();

    assert!(
        scan_bindings(&[models.path().to_path_buf()], Some("owner/model"))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn symlinked_conversion_receipt_for_another_repository_cannot_win() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let target = payloads.path().join("other-model-q4_k_m.gguf");
    write_quant_gguf(&target, 15);
    write_conversion_receipt(
        &target,
        &conversion_receipt(&target, "owner/other-model", &"b".repeat(40), "q4_k_m"),
    );
    symlink(&target, models.path().join("linked-q4_k_m.gguf")).unwrap();

    assert!(scan_bindings(
        &[models.path().to_path_buf()],
        Some("owner/requested-model"),
    )
    .unwrap()
    .is_empty());
}

#[test]
fn symlinked_receipt_target_retarget_is_rejected_before_activation() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let first = payloads.path().join("first-q4_k_m.gguf");
    let second = payloads.path().join("second-q4_k_m.gguf");
    write_quant_gguf(&first, 15);
    write_quant_gguf(&second, 15);
    write_conversion_receipt(
        &first,
        &conversion_receipt(&first, "owner/model", &"a".repeat(40), "q4_k_m"),
    );
    let logical = models.path().join("linked-q4_k_m.gguf");
    symlink(&first, &logical).unwrap();
    let mut candidates =
        scan_bindings(&[models.path().to_path_buf()], Some("owner/model")).unwrap();
    let candidate = candidates.pop().expect("receipt-bound candidate");
    fs::remove_file(&logical).unwrap();
    symlink(&second, &logical).unwrap();

    let error = match verify_candidate(&candidate) {
        Ok(_) => panic!("retargeted receipt-bound candidate must not verify"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("linked conversion target changed after receipt authentication"));
}

#[test]
fn malformed_adjacent_conversion_receipt_does_not_abort_other_candidates() {
    use crate::convert::receipt::receipt_path;

    let models = tempfile::tempdir().unwrap();
    let malformed = models.path().join("malformed-q4_k_m.gguf");
    let valid = models.path().join("valid-q4_k_m.gguf");
    write_quant_gguf(&malformed, 15);
    write_quant_gguf(&valid, 15);
    fs::write(receipt_path(&malformed), b"{not-json").unwrap();
    write_conversion_receipt(
        &valid,
        &conversion_receipt(&valid, "owner/model", &"a".repeat(40), "q4_k_m"),
    );

    let candidates = scan_bindings(&[models.path().to_path_buf()], Some("owner/model"))
        .expect("one malformed receipt must remain candidate-local");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].path, valid);
}

#[test]
fn receipt_bridge_rejects_non_model_and_malformed_schema_v3_identities() {
    use std::os::unix::fs::symlink;

    for (index, mutate) in [
        0_u8, // non-model repository type
        1_u8, // malformed source bundle digest
        2_u8, // malformed converter commit
    ]
    .into_iter()
    .enumerate()
    {
        let models = tempfile::tempdir().unwrap();
        let payloads = tempfile::tempdir().unwrap();
        let target = payloads
            .path()
            .join(format!("converted-{index}-q4_k_m.gguf"));
        write_quant_gguf(&target, 15);
        let mut receipt = conversion_receipt(&target, "owner/model", &"a".repeat(40), "q4_k_m");
        match mutate {
            0 => receipt.source.repository_type = "dataset".into(),
            1 => receipt.source.bundle_sha256 = "not-a-digest".into(),
            2 => receipt.converter.git_commit = "not-a-commit".into(),
            _ => unreachable!(),
        }
        write_conversion_receipt(&target, &receipt);
        symlink(
            &target,
            models.path().join(format!("linked-{index}-q4_k_m.gguf")),
        )
        .unwrap();

        assert!(
            scan_bindings(&[models.path().to_path_buf()], Some("owner/model"))
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn mismatched_logical_sidecars_do_not_import_receipt_use_history() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let target = payloads.path().join("converted-q4_k_m.gguf");
    write_quant_gguf(&target, 15);
    let revision = "a".repeat(40);
    write_conversion_receipt(
        &target,
        &conversion_receipt(&target, "owner/model", &revision, "q4_k_m"),
    );
    let bytes = fs::metadata(&target).unwrap().len();
    let sha256 = crate::core::sha256::compute_file_sha256(&target).unwrap();
    for index in 0..4 {
        let filename = format!("linked-{index}-q4_k_m.gguf");
        let logical = models.path().join(&filename);
        symlink(&target, &logical).unwrap();
        let mut binding = ManagedBinding {
            schema_version: SCHEMA_VERSION,
            repository: "owner/model".into(),
            revision: revision.clone(),
            quant: "Q4_K_M".into(),
            origin: "local_receipt".into(),
            materialized_at_secs: 1,
            last_used_at_secs: 99,
            artifact: ArtifactBinding {
                local_filename: filename,
                hub_filename: "converted-q4_k_m.gguf".into(),
                bytes,
                sha256: sha256.clone(),
            },
            projector: None,
        };
        match index {
            0 => binding.repository = "owner/other".into(),
            1 => binding.revision = "b".repeat(40),
            2 => binding.quant = "Q8_0".into(),
            3 => binding.artifact.sha256 = "f".repeat(64),
            _ => unreachable!(),
        }
        write_binding(&sidecar_path(&logical), &binding).unwrap();
    }

    let candidates = scan_bindings(&[models.path().to_path_buf()], Some("owner/model")).unwrap();
    assert_eq!(candidates.len(), 4);
    assert!(candidates
        .iter()
        .all(|candidate| candidate.last_used_at_secs == 0 && candidate.sidecar.is_none()));
}

#[test]
fn receipt_success_history_is_created_for_links_and_regular_outputs_and_selects_last_used() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let old_revision = "a".repeat(40);
    let new_revision = "b".repeat(40);
    let old_target = payloads.path().join("old-q4_k_m.gguf");
    let new_target = payloads.path().join("new-q4_k_m.gguf");
    write_quant_gguf(&old_target, 15);
    write_quant_gguf(&new_target, 15);
    write_conversion_receipt(
        &old_target,
        &conversion_receipt(&old_target, repository, &old_revision, "q4_k_m"),
    );
    write_conversion_receipt(
        &new_target,
        &conversion_receipt(&new_target, repository, &new_revision, "q4_k_m"),
    );
    let old_link = models.path().join("old-link-q4_k_m.gguf");
    let new_link = models.path().join("new-link-q4_k_m.gguf");
    symlink(&old_target, &old_link).unwrap();
    symlink(&new_target, &new_link).unwrap();
    let old_authority = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        &old_link,
        fs::metadata(&old_link).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    mark_successful_use(
        repository,
        &old_revision,
        QuantType::Q4_K_M,
        &old_link,
        &old_authority,
        &mut cache,
    )
    .unwrap();

    let spec = RepositoryModelSpec {
        repository: repository.into(),
        quant: None,
    };
    let mut warnings = Vec::new();
    let (selected, _, _) = select_local(
        &spec,
        &[models.path().to_path_buf()],
        &cache,
        None,
        16 << 30,
        16 << 30,
        &mut warnings,
    )
    .unwrap()
    .expect("receipt-backed local candidate");
    assert_eq!(selected.path, old_link);
    assert!(selected.last_used_at_secs > 0);

    let regular = models.path().join("regular-q4_k_m.gguf");
    write_quant_gguf(&regular, 15);
    let regular_revision = "c".repeat(40);
    write_conversion_receipt(
        &regular,
        &conversion_receipt(&regular, repository, &regular_revision, "q4_k_m"),
    );
    let regular_authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &regular,
        fs::metadata(&regular).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    mark_successful_use(
        repository,
        &regular_revision,
        QuantType::Q4_K_M,
        &regular,
        &regular_authority,
        &mut cache,
    )
    .unwrap();
    let regular_candidate = scan_bindings(&[models.path().to_path_buf()], Some(repository))
        .unwrap()
        .into_iter()
        .find(|candidate| candidate.path == regular)
        .expect("regular receipt-backed candidate");
    assert!(regular_candidate.last_used_at_secs > 0);
}

#[test]
fn receipt_success_history_never_touches_a_different_manifest_artifact() {
    use crate::serve::cache::{QuantEntry, SourcePointer};

    let models = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let loaded_revision = "a".repeat(40);

    let loaded = models.path().join("loaded-q4_k_m.gguf");
    write_quant_gguf(&loaded, 15);
    write_conversion_receipt(
        &loaded,
        &conversion_receipt(&loaded, repository, &loaded_revision, "q4_k_m"),
    );
    let loaded_authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &loaded,
        fs::metadata(&loaded).unwrap().len(),
    )
    .unwrap()
    .unwrap();

    let manifest_artifact =
        crate::serve::cache::cache_model_path(cache_root.path(), repository, QuantType::Q4_K_M)
            .unwrap();
    fs::create_dir_all(manifest_artifact.parent().unwrap()).unwrap();
    write_quant_gguf(&manifest_artifact, 15);
    let manifest_bytes = fs::metadata(&manifest_artifact).unwrap().len();
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    cache
        .record_source(
            repository,
            &loaded_revision,
            SourcePointer::Local {
                path: cache_root.path().join("source"),
                sha256: "0".repeat(64),
            },
        )
        .unwrap();
    cache
        .record_quantized(
            repository,
            QuantEntry {
                quant_type: QuantType::Q4_K_M.as_str().into(),
                gguf_path: manifest_artifact,
                mmproj_path: None,
                bytes: manifest_bytes,
                sha256: "f".repeat(64),
                quantized_at_secs: 1,
                last_used_at_secs: 0,
                quantized_by_version: env!("CARGO_PKG_VERSION").into(),
            },
        )
        .unwrap();

    mark_successful_use(
        repository,
        &loaded_revision,
        QuantType::Q4_K_M,
        &loaded,
        &loaded_authority,
        &mut cache,
    )
    .unwrap();

    assert!(
        read_binding(&sidecar_path(&loaded))
            .unwrap()
            .unwrap()
            .last_used_at_secs
            > 0
    );
    let current = ModelCache::open_at(cache_root.path()).unwrap();
    assert_eq!(
        current
            .lookup(repository, QuantType::Q4_K_M)
            .unwrap()
            .last_used_at_secs,
        0,
        "a different manifest artifact must not inherit successful-use recency"
    );
}

#[test]
fn receipt_success_history_rejects_retarget_to_another_valid_revision() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let loaded_revision = "a".repeat(40);
    let replacement_revision = "b".repeat(40);
    let loaded = payloads.path().join("loaded-q4_k_m.gguf");
    let replacement = payloads.path().join("replacement-q4_k_m.gguf");
    write_quant_gguf(&loaded, 15);
    write_quant_gguf(&replacement, 15);
    write_conversion_receipt(
        &loaded,
        &conversion_receipt(&loaded, repository, &loaded_revision, "q4_k_m"),
    );
    write_conversion_receipt(
        &replacement,
        &conversion_receipt(&replacement, repository, &replacement_revision, "q4_k_m"),
    );
    let logical = models.path().join("model-q4_k_m.gguf");
    symlink(&loaded, &logical).unwrap();
    let loaded_authority = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        &logical,
        fs::metadata(&logical).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    fs::remove_file(&logical).unwrap();
    symlink(&replacement, &logical).unwrap();
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let error = mark_successful_use(
        repository,
        &loaded_revision,
        QuantType::Q4_K_M,
        &logical,
        &loaded_authority,
        &mut cache,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("changed before successful-use publication"));
    assert!(!sidecar_path(&logical).exists());
}

#[test]
fn regular_receipt_output_never_promotes_a_stale_sidecar_or_forged_projector() {
    let models = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let revision = "a".repeat(40);
    let artifact = models.path().join("regular-q4_k_m.gguf");
    write_quant_gguf(&artifact, 15);
    write_conversion_receipt(
        &artifact,
        &conversion_receipt(&artifact, repository, &revision, "q4_k_m"),
    );
    let stale = ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: repository.into(),
        revision: revision.clone(),
        quant: "Q4_K_M".into(),
        origin: "stale-managed-sidecar".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        artifact: ArtifactBinding {
            local_filename: "regular-q4_k_m.gguf".into(),
            hub_filename: "regular-q4_k_m.gguf".into(),
            bytes: fs::metadata(&artifact).unwrap().len(),
            sha256: "f".repeat(64),
        },
        projector: Some(ArtifactBinding {
            local_filename: "forged-mmproj.gguf".into(),
            hub_filename: "forged-mmproj.gguf".into(),
            bytes: 1,
            sha256: "e".repeat(64),
        }),
    };
    write_binding(&sidecar_path(&artifact), &stale).unwrap();
    let authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &artifact,
        fs::metadata(&artifact).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let error = mark_successful_use(
        repository,
        &revision,
        QuantType::Q4_K_M,
        &artifact,
        &authority,
        &mut cache,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("artifact changed"));

    let candidates = scan_bindings(&[models.path().to_path_buf()], Some(repository)).unwrap();
    let [candidate] = candidates.as_slice() else {
        panic!("one receipt-authoritative candidate expected: {candidates:?}");
    };
    assert_eq!(candidate.origin, "local_receipt");
    assert_eq!(candidate.last_used_at_secs, 0);
    assert!(candidate.projector.is_none());
    assert!(candidate.sidecar.is_none());
}

#[test]
fn exact_receipt_history_clears_forged_sidecar_projector_and_prefers_receipt_pair() {
    let models = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let revision = "a".repeat(40);
    let text = models.path().join("model-q4_k_m.gguf");
    write_quant_gguf_with_metadata(&text, 15, &[("general.architecture", "qwen3vl")]);
    let receipt_projector = models.path().join("model-q4_k_m-mmproj.gguf");
    let forged_projector = models.path().join("forged-mmproj.gguf");
    write_structurally_valid_mmproj(&receipt_projector);
    write_structurally_valid_mmproj(&forged_projector);
    write_conversion_receipt(
        &text,
        &conversion_receipt(&text, repository, &revision, "q4_k_m"),
    );
    write_conversion_receipt(
        &receipt_projector,
        &conversion_receipt(&receipt_projector, repository, &revision, "f16-mmproj"),
    );
    let text_sha = crate::core::sha256::compute_file_sha256(&text).unwrap();
    let forged_sha = crate::core::sha256::compute_file_sha256(&forged_projector).unwrap();
    write_binding(
        &sidecar_path(&text),
        &ManagedBinding {
            schema_version: SCHEMA_VERSION,
            repository: repository.into(),
            revision: revision.clone(),
            quant: "Q4_K_M".into(),
            origin: "forged-managed-sidecar".into(),
            materialized_at_secs: 1,
            last_used_at_secs: 0,
            artifact: ArtifactBinding {
                local_filename: "model-q4_k_m.gguf".into(),
                hub_filename: "model-q4_k_m.gguf".into(),
                bytes: fs::metadata(&text).unwrap().len(),
                sha256: text_sha,
            },
            projector: Some(ArtifactBinding {
                local_filename: "forged-mmproj.gguf".into(),
                hub_filename: "forged-mmproj.gguf".into(),
                bytes: fs::metadata(&forged_projector).unwrap().len(),
                sha256: forged_sha,
            }),
        },
    )
    .unwrap();
    let authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    mark_successful_use(
        repository,
        &revision,
        QuantType::Q4_K_M,
        &text,
        &authority,
        &mut cache,
    )
    .unwrap();
    assert!(read_binding(&sidecar_path(&text))
        .unwrap()
        .unwrap()
        .projector
        .is_none());
    assert_eq!(
        resolve_local_path_projector_required(&text)
            .unwrap()
            .as_deref(),
        Some(receipt_projector.as_path())
    );
}

#[test]
fn logical_projector_alias_satisfies_generation_marked_pair_guard() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let generation = "123e4567-e89b-12d3-a456-426614174000";
    let projector_target = payloads.path().join("converted-q4_k_m-mmproj.gguf");
    write_quant_gguf_with_metadata(
        &projector_target,
        1,
        &[
            ("hf2q.pair_generation", generation),
            ("hf2q.pair_schema_version", "1"),
        ],
    );
    let projector_sha = crate::core::sha256::compute_file_sha256(&projector_target).unwrap();
    let text_target = payloads.path().join("converted-q4_k_m.gguf");
    write_quant_gguf_with_metadata(
        &text_target,
        15,
        &[
            ("hf2q.pair_generation", generation),
            ("hf2q.pair_schema_version", "1"),
            ("hf2q.mmproj_sha256", &projector_sha),
        ],
    );
    let revision = "a".repeat(40);
    write_conversion_receipt(
        &text_target,
        &conversion_receipt(&text_target, "owner/model", &revision, "q4_k_m"),
    );
    write_conversion_receipt(
        &projector_target,
        &conversion_receipt(&projector_target, "owner/model", &revision, "f16-mmproj"),
    );
    let text = models.path().join("linked-q4_k_m.gguf");
    let projector = models.path().join("linked-q4_k_m-mmproj.gguf");
    symlink(&text_target, &text).unwrap();
    symlink(&projector_target, &projector).unwrap();

    let candidates = scan_bindings(&[models.path().to_path_buf()], Some("owner/model")).unwrap();
    let candidate = candidates
        .iter()
        .find(|candidate| candidate.path == text)
        .expect("receipt-bound text candidate");
    assert_eq!(
        candidate
            .projector
            .as_ref()
            .map(|binding| binding.0.as_path()),
        Some(projector.as_path())
    );
    let guard = crate::core::paired_artifact::PairReadGuard::acquire(&text, &projector).unwrap();
    let text_gguf = mlx_native::gguf::GgufFile::open(&text).unwrap();
    let projector_gguf = mlx_native::gguf::GgufFile::open(&projector).unwrap();
    guard
        .validate_static(&text_gguf, &projector_gguf, Some(&projector_sha))
        .unwrap();
}

#[test]
fn retained_target_pair_guard_uses_adjacent_receipt_projector_without_a_logical_alias() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let generation = "123e4567-e89b-12d3-a456-426614174000";
    let projector_target = payloads.path().join("converted-q4_k_m-mmproj.gguf");
    write_quant_gguf_with_metadata(
        &projector_target,
        1,
        &[
            ("hf2q.pair_generation", generation),
            ("hf2q.pair_schema_version", "1"),
        ],
    );
    let projector_sha = crate::core::sha256::compute_file_sha256(&projector_target).unwrap();
    let text_target = payloads.path().join("converted-q4_k_m.gguf");
    write_quant_gguf_with_metadata(
        &text_target,
        15,
        &[
            ("hf2q.pair_generation", generation),
            ("hf2q.pair_schema_version", "1"),
            ("hf2q.mmproj_sha256", &projector_sha),
        ],
    );
    let revision = "a".repeat(40);
    write_conversion_receipt(
        &text_target,
        &conversion_receipt(&text_target, "owner/model", &revision, "q4_k_m"),
    );
    write_conversion_receipt(
        &projector_target,
        &conversion_receipt(&projector_target, "owner/model", &revision, "f16-mmproj"),
    );
    let text = models.path().join("linked-q4_k_m.gguf");
    symlink(&text_target, &text).unwrap();

    let candidate = scan_bindings(&[models.path().to_path_buf()], Some("owner/model"))
        .unwrap()
        .into_iter()
        .find(|candidate| candidate.path == text)
        .expect("receipt-bound text candidate");
    assert_eq!(
        candidate
            .projector
            .as_ref()
            .map(|(path, _, _)| path.as_path()),
        Some(projector_target.canonicalize().unwrap().as_path())
    );
    let text_authority = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let projector_authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &projector_target,
        fs::metadata(&projector_target).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let (text_guard_path, projector_guard_path) =
        crate::serve::automatic_pair_guard_authority_paths(&text_authority, &projector_authority)
            .unwrap()
            .expect("target-adjacent receipt pair must use its conversion namespace");
    let guard = crate::core::paired_artifact::PairReadGuard::acquire(
        &text_guard_path,
        &projector_guard_path,
    )
    .unwrap();
    let text_gguf =
        mlx_native::gguf::GgufFile::from_file(text_authority.try_clone().unwrap()).unwrap();
    let projector_gguf =
        mlx_native::gguf::GgufFile::from_file(projector_authority.try_clone().unwrap()).unwrap();
    guard
        .validate_static(&text_gguf, &projector_gguf, Some(&projector_sha))
        .unwrap();

    let replacement = payloads.path().join("replacement.gguf");
    fs::write(&replacement, fs::read(&text_target).unwrap()).unwrap();
    fs::remove_file(&text).unwrap();
    symlink(&replacement, &text).unwrap();
    assert!(text_authority
        .canonical_path_for_identity()
        .unwrap()
        .is_none());
}

#[test]
fn hosted_projector_beside_logical_text_keeps_the_logical_pair_namespace() {
    use std::os::unix::fs::symlink;

    let models = tempfile::tempdir().unwrap();
    let payloads = tempfile::tempdir().unwrap();
    let generation = "123e4567-e89b-12d3-a456-426614174000";
    let projector = models.path().join("downloaded-mmproj.gguf");
    write_quant_gguf_with_metadata(
        &projector,
        1,
        &[
            ("hf2q.pair_generation", generation),
            ("hf2q.pair_schema_version", "1"),
        ],
    );
    let projector_sha = crate::core::sha256::compute_file_sha256(&projector).unwrap();
    let text_target = payloads.path().join("converted-q4_k_m.gguf");
    write_quant_gguf_with_metadata(
        &text_target,
        15,
        &[
            ("hf2q.pair_generation", generation),
            ("hf2q.pair_schema_version", "1"),
            ("hf2q.mmproj_sha256", &projector_sha),
        ],
    );
    let text = models.path().join("linked-q4_k_m.gguf");
    symlink(&text_target, &text).unwrap();
    let text_authority = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let projector_authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &projector,
        fs::metadata(&projector).unwrap().len(),
    )
    .unwrap()
    .unwrap();

    assert!(crate::serve::automatic_pair_guard_authority_paths(
        &text_authority,
        &projector_authority,
    )
    .unwrap()
    .is_none());
    let guard = crate::core::paired_artifact::PairReadGuard::acquire(&text, &projector).unwrap();
    let text_gguf =
        mlx_native::gguf::GgufFile::from_file(text_authority.try_clone().unwrap()).unwrap();
    let projector_gguf =
        mlx_native::gguf::GgufFile::from_file(projector_authority.try_clone().unwrap()).unwrap();
    guard
        .validate_static(&text_gguf, &projector_gguf, Some(&projector_sha))
        .unwrap();
}

#[test]
fn structurally_valid_wrong_local_sibling_falls_through_to_exact_hosted_projector() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let model = tempfile::tempdir().unwrap();
    let companions = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let text = model.path().join("Qwen3.8-Q4_K_M.gguf");
    let text_bytes = crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text);
    let wrong_local = model.path().join("mmproj-wrong.gguf");
    write_structurally_valid_mmproj(&wrong_local);
    let mut wrong_bytes = fs::read(&wrong_local).unwrap();
    *wrong_bytes.last_mut().unwrap() ^= 1;
    fs::write(&wrong_local, wrong_bytes).unwrap();
    let hosted_projector = companions.path().join("exact-hosted-mmproj.gguf");
    write_structurally_valid_mmproj(&hosted_projector);
    let projector_bytes = fs::metadata(&hosted_projector).unwrap().len();
    let projector_sha = crate::core::sha256::compute_file_sha256(&hosted_projector).unwrap();
    symlink(model.path(), library.path().join("qwen3.8")).unwrap();
    let revision = "a".repeat(40);
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![
            HubGgufArtifact {
                repository: "owner/model".into(),
                revision: revision.clone(),
                filename: "Qwen3.8-Q4_K_M.gguf".into(),
                bytes: text_bytes,
                sha256: "0".repeat(64),
                quant_hint: Some("Q4_K_M".into()),
                role: "text_model".into(),
                selectable: true,
                unavailable_reason: None,
            },
            HubGgufArtifact {
                repository: "owner/model".into(),
                revision,
                filename: "exact-hosted-mmproj.gguf".into(),
                bytes: projector_bytes,
                sha256: projector_sha.clone(),
                quant_hint: None,
                role: "companion".into(),
                selectable: false,
                unavailable_reason: Some("vision projector companion".into()),
            },
        ],
    };
    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let mut catalog = Some(catalog);
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: "owner/model".into(),
            quant: Some(QuantType::Q4_K_M),
        },
        None,
        &[
            library.path().to_path_buf(),
            companions.path().to_path_buf(),
        ],
        &mut cache,
        &hardware,
        true,
        None,
        &mut |_| {},
        |_| Ok(catalog.take().unwrap()),
    )
    .unwrap();

    let prepared = model.path().join("exact-hosted-mmproj.gguf");
    assert_eq!(
        resolved.mmproj_path.as_deref(),
        Some(prepared.canonicalize().unwrap().as_path())
    );
    assert_eq!(
        resolved.mmproj_sha256.as_deref(),
        Some(projector_sha.as_str())
    );
    assert!(resolved.mmproj_activation_authority.is_some());
    assert!(resolved.warnings.iter().any(|warning| {
        warning.contains("ignored structurally compatible local sibling mmproj")
            && warning.contains("does not match exact hosted companion")
    }));
}

#[test]
fn malformed_local_projector_is_rejected_before_binding() {
    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("model-q4_k_m.gguf");
    let projector = directory.path().join("mmproj-corrupt.gguf");
    write_quant_gguf(&text, 15);
    fs::write(&projector, b"not a GGUF").unwrap();
    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        root: directory.path().to_path_buf(),
        path: text,
        bytes: 256,
        sha256: "0".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "manual_structural".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };

    assert!(bind_existing_local_projector(&mut candidate, projector)
        .unwrap_err()
        .to_string()
        .contains("not a readable GGUF"));
    assert!(candidate.projector.is_none());
}

#[test]
fn disappearing_automatic_sibling_mmproj_degrades_to_text_only() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let model = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let text = model.path().join("Qwen3.8-Q4_K_M.gguf");
    let text_bytes = crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text);
    let projector = model.path().join("mmproj-Qwen3.8-F16.gguf");
    write_structurally_valid_mmproj(&projector);
    symlink(model.path(), library.path().join("qwen3.8")).unwrap();
    let revision = "a".repeat(40);
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![HubGgufArtifact {
            repository: "owner/model".into(),
            revision,
            filename: "Qwen3.8-Q4_K_M.gguf".into(),
            bytes: text_bytes,
            sha256: "0".repeat(64),
            quant_hint: Some("Q4_K_M".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        }],
    };
    let remove = projector.canonicalize().unwrap();
    let _hook = set_after_automatic_projector_prepared(move |path| {
        assert_eq!(path, remove);
        fs::remove_file(path).unwrap();
    });
    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let mut catalog = Some(catalog);
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: "owner/model".into(),
            quant: Some(QuantType::Q4_K_M),
        },
        None,
        &[library.path().to_path_buf()],
        &mut cache,
        &hardware,
        true,
        None,
        &mut |_| {},
        |_| Ok(catalog.take().unwrap()),
    )
    .unwrap();

    assert!(resolved.mmproj_path.is_none());
    assert!(resolved.mmproj_sha256.is_none());
    assert!(resolved.mmproj_activation_authority.is_none());
    assert!(resolved.warnings.iter().any(|warning| {
        warning.contains("changed or no longer matches") && warning.contains("serving text-only")
    }));
}

#[test]
fn same_size_projector_replacement_after_hash_admission_degrades_to_text_only() {
    use std::os::unix::fs::symlink;

    let library = tempfile::tempdir().unwrap();
    let model = tempfile::tempdir().unwrap();
    let cache_root = tempfile::tempdir().unwrap();
    let text = model.path().join("Qwen3.8-Q4_K_M.gguf");
    let text_bytes = crate::input::hf_download::tests::write_complete_qwen_test_gguf(&text);
    let projector = model.path().join("mmproj-Qwen3.8-F16.gguf");
    let replacement = model.path().join("replacement-mmproj.bin");
    let parked = model.path().join("parked-mmproj.gguf");
    write_structurally_valid_mmproj(&projector);
    let mut replacement_bytes = fs::read(&projector).unwrap();
    *replacement_bytes.last_mut().unwrap() ^= 1;
    fs::write(&replacement, replacement_bytes).unwrap();
    assert_eq!(
        fs::metadata(&projector).unwrap().len(),
        fs::metadata(&replacement).unwrap().len()
    );
    symlink(model.path(), library.path().join("qwen3.8")).unwrap();
    let revision = "a".repeat(40);
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![HubGgufArtifact {
            repository: "owner/model".into(),
            revision,
            filename: "Qwen3.8-Q4_K_M.gguf".into(),
            bytes: text_bytes,
            sha256: "0".repeat(64),
            quant_hint: Some("Q4_K_M".into()),
            role: "text_model".into(),
            selectable: true,
            unavailable_reason: None,
        }],
    };
    let swap_projector = projector.canonicalize().unwrap();
    let _hook = set_after_automatic_projector_prepared(move |path| {
        assert_eq!(path, swap_projector);
        fs::rename(path, &parked).unwrap();
        fs::rename(&replacement, path).unwrap();
    });
    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let mut catalog = Some(catalog);
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: "owner/model".into(),
            quant: Some(QuantType::Q4_K_M),
        },
        None,
        &[library.path().to_path_buf()],
        &mut cache,
        &hardware,
        true,
        None,
        &mut |_| {},
        |_| Ok(catalog.take().unwrap()),
    )
    .unwrap();

    assert!(resolved.mmproj_path.is_none());
    assert!(resolved.mmproj_sha256.is_none());
    assert!(resolved.mmproj_activation_authority.is_none());
    assert!(
        resolved.warnings.iter().any(|warning| {
            warning.contains("serving text-only")
                && (warning.contains("changed") || warning.contains("no longer matches"))
        }),
        "warnings: {:?}",
        resolved.warnings
    );
}

#[test]
fn exact_loose_digest_adopts_misnamed_bytes_and_hashes_once() {
    use std::cell::Cell;

    let directory = tempfile::tempdir().unwrap();
    let local = directory.path().join("operator-says-q8_0-but-is-q4.gguf");
    write_quant_gguf(&local, 15);
    let bytes = fs::metadata(&local).unwrap().len();
    let digest = crate::core::sha256::compute_file_sha256(&local).unwrap();
    let mut wrong = hosted(QuantType::Q4_K_M, "first-q4_k_m.gguf");
    wrong.bytes = bytes;
    wrong.sha256 = "0".repeat(64);
    let mut exact = hosted(QuantType::Q4_K_M, "second-q4_k_m.gguf");
    exact.bytes = bytes;
    exact.sha256 = digest;

    let mut warnings = Vec::new();
    let hash_count = Cell::new(0_u8);
    let (artifact, found) = find_best_matching_loose_with_hash(
        &[wrong, exact],
        Some(QuantType::Q4_K_M),
        &[directory.path().to_path_buf()],
        &mut warnings,
        |_, _| Ok(()),
        |path, _| {
            hash_count.set(hash_count.get() + 1);
            Ok(crate::core::sha256::compute_file_sha256(path)?)
        },
    )
    .unwrap()
    .expect("owned bytes must disambiguate the hosted filenames");
    assert_eq!(artifact.filename, "second-q4_k_m.gguf");
    assert_eq!(found, local);
    assert_eq!(
        hash_count.get(),
        1,
        "one local path matching multiple catalog rows must be hashed once"
    );
}

#[test]
fn incompatible_newest_loose_digest_continues_to_older_compatible_local_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let older = directory.path().join("older-q4_k_m.gguf");
    let newer = directory.path().join("newer-q8_0.gguf");
    write_quant_gguf(&older, 15);
    write_quant_gguf(&newer, 7);
    for (path, modified) in [(&older, 10), (&newer, 20)] {
        fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(
                fs::FileTimes::new()
                    .set_modified(UNIX_EPOCH + std::time::Duration::from_secs(modified)),
            )
            .unwrap();
    }
    let mut q4 = hosted(QuantType::Q4_K_M, "older-q4_k_m.gguf");
    q4.bytes = fs::metadata(&older).unwrap().len();
    q4.sha256 = crate::core::sha256::compute_file_sha256(&older).unwrap();
    let mut q8 = hosted(QuantType::Q8_0, "newer-q8_0.gguf");
    q8.bytes = fs::metadata(&newer).unwrap().len();
    q8.sha256 = crate::core::sha256::compute_file_sha256(&newer).unwrap();
    let mut warnings = Vec::new();
    let selected = find_best_matching_loose_with(
        &[q4, q8],
        None,
        &[directory.path().to_path_buf()],
        &mut warnings,
        |_, artifact| {
            (artifact.quant_hint.as_deref() != Some("Q8_0"))
                .then_some(())
                .ok_or_else(|| "newest exact digest has no executable storage route".into())
        },
    )
    .unwrap()
    .expect("older compatible exact bytes must remain eligible");
    assert_eq!(selected.0.quant_hint.as_deref(), Some("Q4_K_M"));
    assert_eq!(selected.1, older);
    assert_eq!(warnings.len(), 1);
}

#[test]
fn cached_hub_q8_is_selected_locally_before_a_q4_fallback_tier() {
    let directory = tempfile::tempdir().unwrap();
    let q4_path = directory.path().join("model-q4_k_m.gguf");
    let q8_path = directory.path().join("model-q8_0.gguf");
    fs::write(&q4_path, b"older cached q4").unwrap();
    fs::write(&q8_path, b"newer cached q8").unwrap();
    let mut q4 = hosted(QuantType::Q4_K_M, "model-q4_k_m.gguf");
    q4.bytes = fs::metadata(&q4_path).unwrap().len();
    q4.sha256 = crate::core::sha256::compute_file_sha256(&q4_path).unwrap();
    let mut q8 = hosted(QuantType::Q8_0, "model-q8_0.gguf");
    q8.bytes = fs::metadata(&q8_path).unwrap().len();
    q8.sha256 = crate::core::sha256::compute_file_sha256(&q8_path).unwrap();
    let mut warnings = Vec::new();
    let mut admitted = Vec::new();
    let selected = find_best_matching_cached_hub_with(
        &[q4, q8],
        None,
        &mut warnings,
        |artifact| {
            let (path, age) = if artifact.quant_hint.as_deref() == Some("Q8_0") {
                (q8_path.clone(), 2)
            } else {
                (q4_path.clone(), 1)
            };
            Some((path, UNIX_EPOCH + std::time::Duration::from_secs(age)))
        },
        |_, artifact| {
            admitted.push(artifact.quant_hint.clone().unwrap());
            Ok(())
        },
    )
    .unwrap()
    .expect("newest compatible cached Hub quant must win before recommendation");
    assert_eq!(selected.0.quant_hint.as_deref(), Some("Q8_0"));
    assert_eq!(selected.1, q8_path);
    assert_eq!(admitted, ["Q8_0"]);
    assert!(warnings.is_empty());
}

#[test]
fn candidate_recency_prefers_use_history_then_materialization() {
    let candidate = |used, materialized| Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: PathBuf::from("/tmp/model.gguf"),
        root: PathBuf::from("/tmp"),
        bytes: 1,
        sha256: "b".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: materialized,
        last_used_at_secs: used,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    assert!(candidate_recency(&candidate(5, 1)) > candidate_recency(&candidate(0, 100)));
    assert!(candidate_recency(&candidate(0, 10)) > candidate_recency(&candidate(0, 9)));
    assert!(!bound_candidate_is_at_least_as_recent(&candidate(0, 1), 2));
    assert!(
        bound_candidate_is_at_least_as_recent(&candidate(5, 1), 100),
        "successful-use history outranks a merely newer Hub-cache download"
    );
}

#[test]
fn newer_same_quant_hub_cache_revision_is_not_shadowed_by_post_lock_old_binding() {
    let old_bound = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: PathBuf::from("/managed/old-q4_k_m.gguf"),
        root: PathBuf::from("/managed"),
        bytes: 1,
        sha256: "b".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "managed_binding".into(),
        materialized_at_secs: 10,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    assert!(
        !post_lock_local_candidate_wins(&old_bound, Some(20)),
        "the post-lock recheck must preserve the already selected newer Hub-cache revision"
    );
    let mut used = old_bound;
    used.last_used_at_secs = 1;
    assert!(
        post_lock_local_candidate_wins(&used, Some(20)),
        "a concurrent successful-use publication remains stronger than materialization time"
    );
}

#[test]
fn admissible_recent_local_quant_is_not_capped_by_the_hardware_recommendation() {
    let gib = 1024_u64 * 1024 * 1024;
    let candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: PathBuf::from("/tmp/model-q8_0.gguf"),
        root: PathBuf::from("/tmp"),
        bytes: 28 * gib,
        sha256: "b".repeat(64),
        quant: QuantType::Q8_0,
        origin: "test".into(),
        materialized_at_secs: 10,
        last_used_at_secs: 20,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let spec = RepositoryModelSpec {
        repository: "owner/model".into(),
        quant: None,
    };

    assert!(local_candidate_eligible(
        &spec,
        &candidate,
        None,
        48 * gib,
        40 * gib,
    ));
}

#[test]
fn projector_pairing_prefers_exact_text_stem_then_one_generic_companion() {
    let candidate = |filename: &str| Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: PathBuf::from("/tmp").join(filename),
        root: PathBuf::from("/tmp"),
        bytes: 1,
        sha256: "b".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let companion = |filename: &str| HubGgufArtifact {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        filename: filename.into(),
        bytes: 1,
        sha256: "c".repeat(64),
        quant_hint: None,
        role: "companion".into(),
        selectable: false,
        unavailable_reason: Some("vision projector companion; not a text model".into()),
    };
    let generic = companion("gguf/mmproj-model-f16.gguf");
    let q4 = companion("gguf/model-q4_k_m-mmproj.gguf");
    let companions = vec![&generic, &q4];

    assert_eq!(
        select_projector_companion(&candidate("model-q4_k_m.gguf"), companions.clone(), None)
            .unwrap()
            .unwrap()
            .filename,
        q4.filename
    );
    assert_eq!(
        select_projector_companion(&candidate("model-q5_k_m.gguf"), companions, None)
            .unwrap()
            .unwrap()
            .filename,
        generic.filename
    );

    let generic_a = companion("gguf/mmproj-a-f16.gguf");
    let generic_b = companion("gguf/mmproj-b-f16.gguf");
    assert!(select_projector_companion(
        &candidate("model-q6_k.gguf"),
        vec![&generic_a, &generic_b],
        None,
    )
    .unwrap()
    .is_none());
}

#[test]
fn older_local_multimodal_candidate_requests_its_exact_revision_not_head_projector() {
    let revision = "a".repeat(40);
    let candidate = Candidate {
        repository: "owner/model".into(),
        revision: revision.clone(),
        path: PathBuf::from("/tmp/model-q4_k_m.gguf"),
        root: PathBuf::from("/tmp"),
        bytes: 1,
        sha256: "b".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 2,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let head_catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: "c".repeat(40),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: Vec::new(),
    };

    let reference = exact_local_projector_catalog_reference(&candidate, &head_catalog)
        .unwrap()
        .expect("a stale local revision must replace the HEAD catalog");
    assert_eq!(reference.repo_id(), candidate.repository);
    assert_eq!(reference.requested_revision(), Some(revision.as_str()));

    let exact_catalog = HubGgufCatalog {
        revision,
        ..head_catalog
    };
    assert!(
        exact_local_projector_catalog_reference(&candidate, &exact_catalog)
            .unwrap()
            .is_none()
    );
}

#[test]
fn markerless_local_multimodal_uses_candidate_revision_config_and_companion() {
    use std::cell::Cell;

    let text_root = tempfile::tempdir().unwrap();
    let loose_root = tempfile::tempdir().unwrap();
    let text = text_root.path().join("model-q4_k_m.gguf");
    let loose_projector = loose_root.path().join("operator-projector.gguf");
    write_quant_gguf(&text, 15);
    fs::write(&loose_projector, b"exact revision projector").unwrap();
    let revision = "a".repeat(40);
    let candidate = Candidate {
        repository: "owner/model".into(),
        revision: revision.clone(),
        path: text.clone(),
        root: text_root.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        quant: QuantType::Q4_K_M,
        origin: "manual".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 2,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let text_authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    assert!(!text_requires_projector(&candidate.path).unwrap());
    let head_catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: "c".repeat(40),
        requires_projector: false,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: Vec::new(),
    };
    let exact_projector = HubGgufArtifact {
        repository: "owner/model".into(),
        revision: revision.clone(),
        filename: "mmproj-exact-f16.gguf".into(),
        bytes: fs::metadata(&loose_projector).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&loose_projector).unwrap(),
        quant_hint: None,
        role: "companion".into(),
        selectable: false,
        unavailable_reason: Some("vision projector companion; not a text model".into()),
    };
    let mut exact_catalog = Some(HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![exact_projector],
    });
    let resolver_calls = Cell::new(0_u8);
    let mut warnings = Vec::new();

    let (prepared, projector) = prepare_local_candidate_with_catalog_resolver(
        candidate,
        &text_authority,
        Some(text_root.path()),
        &[loose_root.path().to_path_buf()],
        &head_catalog,
        true,
        &mut warnings,
        &mut |_| {},
        |reference| {
            resolver_calls.set(resolver_calls.get() + 1);
            assert_eq!(reference.repo_id(), "owner/model");
            assert_eq!(reference.requested_revision(), Some(revision.as_str()));
            Ok(exact_catalog.take().unwrap())
        },
    )
    .unwrap();

    let projector = projector.expect("exact-revision config requires its exact companion");
    assert_eq!(resolver_calls.get(), 1);
    assert_eq!(prepared.path, text);
    assert_eq!(projector, text_root.path().join("mmproj-exact-f16.gguf"));
    assert_eq!(fs::read(projector).unwrap(), b"exact revision projector");
    assert!(warnings.is_empty());
}

#[test]
fn hosted_projector_is_materialized_and_ambiguity_falls_back_text_only() {
    let text_root = tempfile::tempdir().unwrap();
    let loose_root = tempfile::tempdir().unwrap();
    let text = text_root.path().join("model-q4_k_m.gguf");
    write_quant_gguf_with_metadata(&text, 15, &[("general.architecture", "qwen3vl")]);
    let companion = loose_root.path().join("operator-mmproj.gguf");
    fs::write(&companion, b"verified-projector").unwrap();
    let companion_sha = crate::core::sha256::compute_file_sha256(&companion).unwrap();
    let revision = "a".repeat(40);
    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: revision.clone(),
        path: text.clone(),
        root: text_root.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let companion_artifact = HubGgufArtifact {
        repository: "owner/model".into(),
        revision: revision.clone(),
        filename: "mmproj-model-f16.gguf".into(),
        bytes: fs::metadata(&companion).unwrap().len(),
        sha256: companion_sha,
        quant_hint: None,
        role: "companion".into(),
        selectable: false,
        unavailable_reason: Some("vision projector companion; not a text model".into()),
    };
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![companion_artifact.clone()],
    };
    let mut warnings = Vec::new();
    let resolved = resolve_projector_with_catalog(
        &mut candidate,
        &[loose_root.path().to_path_buf()],
        &catalog,
        &mut warnings,
    )
    .unwrap()
    .expect("one exact loose companion should be materialized");
    assert_eq!(resolved, text_root.path().join("mmproj-model-f16.gguf"));
    assert!(warnings.is_empty());

    let mut ambiguous_candidate = candidate.clone();
    ambiguous_candidate.projector = None;
    ambiguous_candidate.sidecar = None;
    let mut second = companion_artifact;
    second.filename = "mmproj-other-f16.gguf".into();
    let ambiguous = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision,
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![
            HubGgufArtifact {
                filename: "mmproj-first-f16.gguf".into(),
                ..second.clone()
            },
            second,
        ],
    };
    let mut fallback_warnings = Vec::new();
    assert!(resolve_projector_with_catalog(
        &mut ambiguous_candidate,
        &[],
        &ambiguous,
        &mut fallback_warnings,
    )
    .unwrap()
    .is_none());
    assert!(fallback_warnings
        .iter()
        .any(|warning| warning.contains("text-only")));
}

#[test]
fn prepared_existing_projector_rejects_same_inode_mutation_before_activation() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("mmproj-model-f16.gguf");
    let alias = directory.path().join("operator-alias.gguf");
    fs::write(&destination, b"projector-one").unwrap();
    fs::hard_link(&destination, &alias).unwrap();
    let artifact = HubGgufArtifact {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        filename: "mmproj-model-f16.gguf".into(),
        bytes: fs::metadata(&destination).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&destination).unwrap(),
        quant_hint: None,
        role: "companion".into(),
        selectable: false,
        unavailable_reason: Some("vision projector companion; not a text model".into()),
    };
    let plan = prepare_projector_action(artifact, destination, &[]).unwrap();
    fs::write(alias, b"projector-two").unwrap();
    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: directory.path().join("text.gguf"),
        root: directory.path().to_path_buf(),
        bytes: 0,
        sha256: String::new(),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 0,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let error = materialize_prepared_projector(plan, &mut candidate, &mut Vec::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("changed before activation"));
    assert!(candidate.projector.is_none());
}

#[test]
fn cached_projector_snapshot_symlink_is_retained_from_its_repo_blob() {
    use std::os::unix::fs::symlink;

    let cache = tempfile::tempdir().unwrap();
    let revision = "a".repeat(40);
    let repository = cache.path().join("models--owner--model");
    let blobs = repository.join("blobs");
    let nested = repository.join("snapshots").join(&revision).join("gguf");
    fs::create_dir_all(&blobs).unwrap();
    fs::create_dir_all(&nested).unwrap();
    let blob = blobs.join("projector-digest");
    fs::write(&blob, b"cached-projector").unwrap();
    let snapshot = nested.join("mmproj-model-f16.gguf");
    symlink("../../../blobs/projector-digest", &snapshot).unwrap();
    let artifact = HubGgufArtifact {
        repository: "owner/model".into(),
        revision,
        filename: "gguf/mmproj-model-f16.gguf".into(),
        bytes: fs::metadata(&blob).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&blob).unwrap(),
        quant_hint: None,
        role: "companion".into(),
        selectable: false,
        unavailable_reason: Some("vision projector companion; not a text model".into()),
    };
    let mut retained = retain_cached_projector_at(&artifact, &snapshot).unwrap();
    assert_eq!(retained.path, blob.canonicalize().unwrap());
    assert_eq!(
        retained.retained.sha256().unwrap().unwrap(),
        artifact.sha256
    );
}

#[test]
fn retained_adoption_has_an_independent_inode_and_source_alias_mutation_isolated() {
    use std::os::unix::fs::MetadataExt;

    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("operator.gguf");
    let alias = directory.path().join("operator-alias.gguf");
    let destination = directory.path().join("managed.gguf");
    fs::write(&source, b"immutable-model-bytes").unwrap();
    fs::hard_link(&source, &alias).unwrap();
    let bytes = fs::metadata(&source).unwrap().len();
    let mut retained = crate::core::bounded_file::StableRegularFile::open_exact(&source, bytes)
        .unwrap()
        .unwrap();
    let digest = retained.sha256().unwrap().unwrap();
    materialize_retained_exact(retained, &destination, "owner/model", bytes, &digest).unwrap();
    assert_ne!(
        fs::metadata(&source).unwrap().ino(),
        fs::metadata(&destination).unwrap().ino()
    );
    fs::write(alias, b"different-model-bytes").unwrap();
    assert_eq!(fs::read(destination).unwrap(), b"immutable-model-bytes");
}

#[test]
fn verified_projector_load_survives_sidecar_history_persistence_failure() {
    let text_root = tempfile::tempdir().unwrap();
    let loose_root = tempfile::tempdir().unwrap();
    let text = text_root.path().join("model-q4_k_m.gguf");
    write_quant_gguf_with_metadata(&text, 15, &[("general.architecture", "qwen3vl")]);
    let companion = loose_root.path().join("operator-mmproj.gguf");
    fs::write(&companion, b"verified-projector").unwrap();
    let revision = "a".repeat(40);
    let invalid_sidecar = text_root.path().join("binding-is-a-directory");
    fs::create_dir(&invalid_sidecar).unwrap();
    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: revision.clone(),
        path: text.clone(),
        root: text_root.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: Some(invalid_sidecar),
        receipt_target_identity: None,
    };
    let artifact = HubGgufArtifact {
        repository: "owner/model".into(),
        revision: revision.clone(),
        filename: "mmproj-model-f16.gguf".into(),
        bytes: fs::metadata(&companion).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&companion).unwrap(),
        quant_hint: None,
        role: "companion".into(),
        selectable: false,
        unavailable_reason: Some("vision projector companion; not a text model".into()),
    };
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision,
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![artifact],
    };
    let mut warnings = Vec::new();
    assert!(resolve_projector_with_catalog(
        &mut candidate,
        &[loose_root.path().to_path_buf()],
        &catalog,
        &mut warnings,
    )
    .unwrap()
    .is_some());
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("will be loaded")));
}

#[test]
fn text_only_qwen_does_not_load_a_digest_valid_stale_projector() {
    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("text-q4_k_m.gguf");
    write_quant_gguf_with_metadata(&text, 15, &[("general.architecture", "qwen35")]);
    let projector = directory.path().join("mmproj-stale.gguf");
    fs::write(&projector, b"digest-valid-but-not-requested").unwrap();
    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: text.clone(),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: Some((
            projector.clone(),
            fs::metadata(&projector).unwrap().len(),
            crate::core::sha256::compute_file_sha256(&projector).unwrap(),
        )),
        sidecar: None,
        receipt_target_identity: None,
    };
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        requires_projector: false,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: Vec::new(),
    };
    assert!(resolve_projector_with_catalog(
        &mut candidate,
        &[directory.path().to_path_buf()],
        &catalog,
        &mut Vec::new(),
    )
    .unwrap()
    .is_none());
}

#[test]
fn text_only_local_projector_resolution_stops_before_any_hub_lookup() {
    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("text-q4_k_m.gguf");
    write_quant_gguf_with_metadata(&text, 15, &[("general.architecture", "qwen35")]);
    let mut candidate = Candidate {
        repository: "not a hub repository".into(),
        revision: "not-a-revision".into(),
        path: text.clone(),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let mut warnings = Vec::new();
    assert!(resolve_projector(&mut candidate, &[], &mut warnings)
        .unwrap()
        .is_none());
    assert!(warnings.is_empty());
}

#[test]
fn qwen_vision_profile_is_gguf_authority_for_projector_requirement() {
    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("vision-q4_k_m.gguf");
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[
            ("general.architecture", "qwen35"),
            ("hf2q.vision.projector_profile", "qwen3vl_siglip"),
        ],
    );
    assert!(text_requires_projector(&text).unwrap());
}

#[test]
fn authenticated_gguf_vision_marker_overrides_absent_repository_config_marker() {
    assert!(hosted_pair_requires_projector(false, true));
    assert!(hosted_pair_requires_projector(true, false));
    assert!(!hosted_pair_requires_projector(false, false));
}

#[test]
fn automatic_projector_preflight_failure_keeps_text_plan_and_suppresses_retry() {
    use std::cell::Cell;

    let pair_calls = Cell::new(0);
    let text_calls = Cell::new(0);
    let mut warnings = Vec::new();
    let suppressed = admit_automatic_projector_preflight(
        Err(anyhow!("conflicting automatic projector destination")),
        |_| {
            pair_calls.set(pair_calls.get() + 1);
            Ok(())
        },
        || {
            text_calls.set(text_calls.get() + 1);
            Ok(())
        },
        &mut warnings,
    )
    .unwrap();
    assert!(suppressed);
    assert_eq!(pair_calls.get(), 0);
    assert_eq!(text_calls.get(), 1);
    assert!(warnings.iter().any(|warning| warning.contains("text-only")));

    let projector = hosted(QuantType::Q4_K_M, "mmproj-qwen-f16.gguf");
    let suppressed = admit_automatic_projector_preflight(
        Ok(Some((projector, PathBuf::from("/managed/mmproj.gguf")))),
        |_| {
            pair_calls.set(pair_calls.get() + 1);
            Err(anyhow!("insufficient projector extent"))
        },
        || {
            text_calls.set(text_calls.get() + 1);
            Ok(())
        },
        &mut warnings,
    )
    .unwrap();
    assert!(suppressed);
    assert_eq!(pair_calls.get(), 1);
    assert_eq!(text_calls.get(), 2);
}

#[test]
fn managed_binding_rejects_traversal_and_malformed_identity() {
    let binding = ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        quant: "Q4_K_M".into(),
        origin: "hosted_download".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        artifact: ArtifactBinding {
            local_filename: "../model.gguf".into(),
            hub_filename: "model-q4_k_m.gguf".into(),
            bytes: 1,
            sha256: "b".repeat(64),
        },
        projector: None,
    };
    assert!(validate_binding(&binding).is_err());
}

#[test]
fn retained_materialization_creates_independent_bytes_and_refuses_conflicts() {
    use std::os::unix::fs::MetadataExt;

    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.gguf");
    let destination = directory.path().join("model.gguf");
    std::fs::write(&source, b"exact hosted bytes").unwrap();
    let sha256 = crate::core::sha256::compute_file_sha256(&source).unwrap();
    materialize_preverified_exact(
        &source,
        &destination,
        "owner/model",
        std::fs::metadata(&source).unwrap().len(),
        &sha256,
    )
    .unwrap();
    assert_ne!(
        std::fs::metadata(&source).unwrap().ino(),
        std::fs::metadata(&destination).unwrap().ino()
    );
    materialize_preverified_exact(
        &source,
        &destination,
        "owner/model",
        std::fs::metadata(&source).unwrap().len(),
        &sha256,
    )
    .unwrap();

    let conflict = directory.path().join("conflict.gguf");
    std::fs::write(&conflict, b"other").unwrap();
    assert!(materialize_preverified_exact(
        &source,
        &conflict,
        "owner/model",
        std::fs::metadata(&source).unwrap().len(),
        &sha256,
    )
    .is_err());
    assert_eq!(std::fs::read(conflict).unwrap(), b"other");

    let digest_mismatch = directory.path().join("digest-mismatch.gguf");
    assert!(materialize_preverified_exact(
        &source,
        &digest_mismatch,
        "owner/model",
        std::fs::metadata(&source).unwrap().len(),
        &"0".repeat(64),
    )
    .is_err());
    assert!(!digest_mismatch.exists());
}

#[test]
fn destination_preflight_reuses_exact_bytes_and_rejects_conflict_before_transfer() {
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("outside-scan-roots.gguf");
    assert!(!verify_or_refuse_existing_destination(&destination, 5, &"0".repeat(64)).unwrap());

    fs::write(&destination, b"exact").unwrap();
    let digest = crate::core::sha256::compute_file_sha256(&destination).unwrap();
    assert!(verify_or_refuse_existing_destination(&destination, 5, &digest).unwrap());
    assert!(verify_or_refuse_existing_destination(&destination, 5, &"0".repeat(64)).is_err());
    assert_eq!(fs::read(destination).unwrap(), b"exact");
}

#[test]
fn clone_errors_select_only_supported_copy_fallbacks() {
    for code in [libc::EXDEV, libc::ENOTSUP, libc::EOPNOTSUPP] {
        assert!(clone_requires_copy(&std::io::Error::from_raw_os_error(
            code
        )));
    }
    assert!(!clone_requires_copy(&std::io::Error::from_raw_os_error(
        libc::EIO
    )));
}

#[test]
fn concurrent_exact_destination_winner_is_rechecked_and_reused() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.gguf");
    let destination = directory.path().join("managed.gguf");
    fs::write(&source, b"exact bytes").unwrap();
    let digest = crate::core::sha256::compute_file_sha256(&source).unwrap();
    let workers = (0..2)
        .map(|_| {
            let source = source.clone();
            let destination = destination.clone();
            let digest = digest.clone();
            std::thread::spawn(move || {
                materialize_preverified_exact(&source, &destination, "owner/model", 11, &digest)
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap().unwrap();
    }
    assert_eq!(fs::read(destination).unwrap(), b"exact bytes");
}

#[test]
fn retained_exact_destination_replacement_after_preflight_refuses_activation() {
    use std::os::unix::fs::MetadataExt;

    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.gguf");
    let destination = directory.path().join("destination.gguf");
    let prior_destination = directory.path().join("prior-destination.gguf");
    let bytes = b"same exact artifact bytes";
    fs::write(&source, bytes).unwrap();
    fs::write(&destination, bytes).unwrap();
    let digest = crate::core::sha256::compute_file_sha256(&source).unwrap();
    let mut retained =
        crate::core::bounded_file::StableRegularFile::open_exact(&source, bytes.len() as u64)
            .unwrap()
            .unwrap();
    assert_eq!(retained.sha256().unwrap().as_deref(), Some(digest.as_str()));
    let plan = PreparedLocalArtifact::prepare_retained(
        retained,
        &destination,
        bytes.len() as u64,
        &digest,
    )
    .unwrap();
    assert!(!plan.needs_copy());
    let original_inode = fs::metadata(&destination).unwrap().ino();

    fs::rename(&destination, &prior_destination).unwrap();
    fs::write(&destination, bytes).unwrap();
    assert_ne!(fs::metadata(&destination).unwrap().ino(), original_inode);
    assert!(!plan.is_current().unwrap());
    let error = plan
        .materialize("owner/model", bytes.len() as u64, &digest)
        .unwrap_err()
        .to_string();
    assert!(error.contains("changed before activation"));
    assert!(!sidecar_path(&destination).exists());
}

#[test]
fn hub_cache_symlink_source_materializes_as_a_regular_file() {
    use std::os::unix::fs::symlink;

    let directory = tempfile::tempdir().unwrap();
    let blob = directory.path().join("blob");
    let pointer = directory.path().join("snapshot.gguf");
    let destination = directory.path().join("managed.gguf");
    std::fs::write(&blob, b"hub blob bytes").unwrap();
    symlink(&blob, &pointer).unwrap();
    let sha256 = crate::core::sha256::compute_file_sha256(&pointer).unwrap();
    materialize_preverified_exact(
        &pointer,
        &destination,
        "owner/model",
        std::fs::metadata(&pointer).unwrap().len(),
        &sha256,
    )
    .unwrap();
    assert!(!std::fs::symlink_metadata(&destination)
        .unwrap()
        .file_type()
        .is_symlink());
    assert_eq!(std::fs::read(destination).unwrap(), b"hub blob bytes");
}

#[test]
fn explicit_output_is_honored_for_an_existing_verified_local_candidate() {
    let source_directory = tempfile::tempdir().unwrap();
    let output_directory = tempfile::tempdir().unwrap();
    let source = source_directory.path().join("model-q4_k_m.gguf");
    std::fs::write(&source, b"verified local conversion").unwrap();
    let mut warnings = Vec::new();
    let prepared = prepare_selected_local(
        Candidate {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            path: source.clone(),
            root: source_directory.path().to_path_buf(),
            bytes: std::fs::metadata(&source).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(&source).unwrap(),
            quant: QuantType::Q4_K_M,
            origin: "cache_manifest".to_owned(),
            materialized_at_secs: 1,
            last_used_at_secs: 0,
            projector: None,
            sidecar: None,
            receipt_target_identity: None,
        },
        Some(output_directory.path()),
        &mut warnings,
    )
    .unwrap();
    assert!(warnings.is_empty());
    assert_eq!(
        prepared.path,
        output_directory.path().join("model-q4_k_m.gguf")
    );
    assert!(sidecar_path(&prepared.path).is_file());
    assert_eq!(
        std::fs::read(prepared.path).unwrap(),
        b"verified local conversion"
    );
}

#[test]
fn explicit_local_output_projector_conflict_warns_and_materializes_text_only() {
    let source_directory = tempfile::tempdir().unwrap();
    let output_directory = tempfile::tempdir().unwrap();
    let source = source_directory.path().join("model-q4_k_m.gguf");
    let projector = source_directory.path().join("model-mmproj.gguf");
    write_quant_gguf(&source, 15);
    write_structurally_valid_mmproj(&projector);
    let projector_destination = output_directory.path().join("model-mmproj.gguf");
    fs::write(&projector_destination, b"conflicting bytes").unwrap();
    let mut warnings = Vec::new();
    let prepared = prepare_selected_local(
        Candidate {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            path: source.clone(),
            root: source_directory.path().to_path_buf(),
            bytes: fs::metadata(&source).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(&source).unwrap(),
            quant: QuantType::Q4_K_M,
            origin: "cache_manifest".into(),
            materialized_at_secs: 1,
            last_used_at_secs: 0,
            projector: Some((
                projector.clone(),
                fs::metadata(&projector).unwrap().len(),
                crate::core::sha256::compute_file_sha256(&projector).unwrap(),
            )),
            sidecar: None,
            receipt_target_identity: None,
        },
        Some(output_directory.path()),
        &mut warnings,
    )
    .unwrap();
    assert_eq!(
        prepared.path,
        output_directory.path().join("model-q4_k_m.gguf")
    );
    assert_eq!(fs::read(&prepared.path).unwrap(), fs::read(source).unwrap());
    assert!(prepared.projector.is_none());
    assert_eq!(
        fs::read(projector_destination).unwrap(),
        b"conflicting bytes"
    );
    assert!(warnings.iter().any(|warning| warning.contains("text-only")));
    let binding = read_binding(prepared.sidecar.as_ref().unwrap())
        .unwrap()
        .unwrap();
    assert!(binding.projector.is_none());
}

#[test]
fn local_pair_extent_refusal_suppresses_projector_retry_after_text_materialization() {
    use std::cell::Cell;

    let source_directory = tempfile::tempdir().unwrap();
    let output_directory = tempfile::tempdir().unwrap();
    let source = source_directory.path().join("model-q4_k_m.gguf");
    let projector = source_directory.path().join("model-mmproj.gguf");
    write_quant_gguf(&source, 15);
    write_structurally_valid_mmproj(&projector);
    let pair_preflights = Cell::new(0_u8);
    let mut warnings = Vec::new();
    let (prepared, suppress_projector) = prepare_selected_local_decision_with_preflight(
        Candidate {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            path: source.clone(),
            root: source_directory.path().to_path_buf(),
            bytes: fs::metadata(&source).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(&source).unwrap(),
            quant: QuantType::Q4_K_M,
            origin: "manual".into(),
            materialized_at_secs: 1,
            last_used_at_secs: 0,
            projector: Some((
                projector.clone(),
                fs::metadata(&projector).unwrap().len(),
                crate::core::sha256::compute_file_sha256(&projector).unwrap(),
            )),
            sidecar: None,
            receipt_target_identity: None,
        },
        Some(output_directory.path()),
        &mut warnings,
        |_, _, _, _, _, projector| {
            pair_preflights.set(pair_preflights.get() + 1);
            if projector.is_some() {
                Err(
                    crate::input::hf_download::DownloadError::InvalidRepositoryInventory {
                        reason: "forced aggregate extent refusal".into(),
                    },
                )
            } else {
                Ok(())
            }
        },
    )
    .unwrap();
    assert!(suppress_projector);
    assert_eq!(pair_preflights.get(), 2, "pair then text-only preflight");
    assert!(prepared.path.is_file());
    assert!(prepared.projector.is_none());

    let projector_retries = Cell::new(0_u8);
    let resolved_projector = (!suppress_projector)
        .then(|| {
            projector_retries.set(projector_retries.get() + 1);
            Some(PathBuf::from("unexpected-mmproj.gguf"))
        })
        .flatten();
    assert!(resolved_projector.is_none());
    assert_eq!(projector_retries.get(), 0);
    assert!(warnings.iter().any(|warning| warning.contains("text-only")));
}

#[test]
fn local_pair_source_and_destination_parent_replacement_refuses_before_first_write() {
    use std::os::unix::fs::symlink;

    let source_directory = tempfile::tempdir().unwrap();
    let output_root = tempfile::tempdir().unwrap();
    let output_directory = output_root.path().join("output");
    let parked_output = output_root.path().join("parked");
    let outside_output = output_root.path().join("outside");
    fs::create_dir(&output_directory).unwrap();
    fs::create_dir(&outside_output).unwrap();
    let source = source_directory.path().join("model-q4_k_m.gguf");
    let parked_source = source_directory.path().join("parked-model.gguf");
    let projector = source_directory.path().join("model-mmproj.gguf");
    write_quant_gguf(&source, 15);
    write_structurally_valid_mmproj(&projector);
    let original_source = fs::read(&source).unwrap();
    let mut warnings = Vec::new();

    let error = prepare_selected_local_decision_with_preflight(
        Candidate {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            path: source.clone(),
            root: source_directory.path().to_path_buf(),
            bytes: fs::metadata(&source).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(&source).unwrap(),
            quant: QuantType::Q4_K_M,
            origin: "manual".into(),
            materialized_at_secs: 1,
            last_used_at_secs: 0,
            projector: Some((
                projector.clone(),
                fs::metadata(&projector).unwrap().len(),
                crate::core::sha256::compute_file_sha256(&projector).unwrap(),
            )),
            sidecar: None,
            receipt_target_identity: None,
        },
        Some(&output_directory),
        &mut warnings,
        |_, _, _, _, _, projector| {
            if projector.is_some() {
                fs::rename(&source, &parked_source).unwrap();
                fs::write(&source, &original_source).unwrap();
                fs::rename(&output_directory, &parked_output).unwrap();
                symlink(&outside_output, &output_directory).unwrap();
            }
            Ok(())
        },
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("authority changed after disk preflight"));
    assert!(!parked_output.join("model-q4_k_m.gguf").exists());
    assert!(!outside_output.join("model-q4_k_m.gguf").exists());
    assert!(!parked_output.join("model-mmproj.gguf").exists());
    assert!(!outside_output.join("model-mmproj.gguf").exists());
}

#[test]
fn managed_binding_round_trips_atomically_and_touches_only_after_success() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("model.gguf");
    fs::write(&artifact, b"x").unwrap();
    let sidecar = sidecar_path(&artifact);
    let binding = ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        quant: "Q4_K_M".into(),
        origin: "hosted_download".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        artifact: ArtifactBinding {
            local_filename: "model.gguf".into(),
            hub_filename: "gguf/model-q4_k_m.gguf".into(),
            bytes: 1,
            sha256: "b".repeat(64),
        },
        projector: None,
    };
    write_binding(&sidecar, &binding).unwrap();
    let loaded = read_binding(&sidecar).unwrap().unwrap();
    assert_eq!(loaded.last_used_at_secs, 0);
    assert_eq!(loaded.artifact.hub_filename, "gguf/model-q4_k_m.gguf");

    let cache_root = tempfile::tempdir().unwrap();
    let mut cache = ModelCache::open_at(cache_root.path()).unwrap();
    let authority = crate::core::bounded_file::StableRegularFile::open_exact(&artifact, 1)
        .unwrap()
        .unwrap();
    mark_successful_use(
        "owner/model",
        &"a".repeat(40),
        QuantType::Q4_K_M,
        &artifact,
        &authority,
        &mut cache,
    )
    .unwrap();
    assert!(read_binding(&sidecar).unwrap().unwrap().last_used_at_secs > 0);
}

#[test]
fn local_adoption_preserves_a_conflicting_operator_sidecar() {
    let source_directory = tempfile::tempdir().unwrap();
    let output_directory = tempfile::tempdir().unwrap();
    let source = source_directory.path().join("model-q4_k_m.gguf");
    write_quant_gguf(&source, 15);
    let destination = output_directory.path().join("model-q4_k_m.gguf");
    let sidecar = sidecar_path(&destination);
    fs::write(&sidecar, b"operator-owned conflicting sidecar").unwrap();
    let mut warnings = Vec::new();

    let prepared = prepare_selected_local(
        Candidate {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            path: source.clone(),
            root: source_directory.path().to_path_buf(),
            bytes: fs::metadata(&source).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(&source).unwrap(),
            quant: QuantType::Q4_K_M,
            origin: "manual".into(),
            materialized_at_secs: 1,
            last_used_at_secs: 0,
            projector: None,
            sidecar: None,
            receipt_target_identity: None,
        },
        Some(output_directory.path()),
        &mut warnings,
    )
    .unwrap();

    assert_eq!(prepared.path, destination);
    assert!(prepared.sidecar.is_none());
    assert_eq!(
        fs::read(sidecar).unwrap(),
        b"operator-owned conflicting sidecar"
    );
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("could not persist")));
}

#[test]
fn stale_managed_binding_does_not_create_an_available_candidate() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("model.gguf");
    let sidecar = sidecar_path(&artifact);
    let binding = ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        quant: "Q4_K_M".into(),
        origin: "hosted_download".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        artifact: ArtifactBinding {
            local_filename: "model.gguf".into(),
            hub_filename: "model-q4_k_m.gguf".into(),
            bytes: 5,
            sha256: "b".repeat(64),
        },
        projector: Some(ArtifactBinding {
            local_filename: "missing-mmproj.gguf".into(),
            hub_filename: "mmproj-model-f16.gguf".into(),
            bytes: 10,
            sha256: "c".repeat(64),
        }),
    };
    write_binding(&sidecar, &binding).unwrap();
    assert!(candidate_from_binding(binding.clone(), artifact.clone(), sidecar.clone()).is_err());

    fs::write(&artifact, b"exact").unwrap();
    let candidate = candidate_from_binding(binding, artifact, sidecar).unwrap();
    assert!(candidate.projector.is_none());
}

#[test]
fn verified_conversion_pair_is_reused_and_published_as_managed_authority() {
    let directory = tempfile::tempdir().unwrap();
    let revision = "a".repeat(40);
    let text = directory.path().join("model-q4_k_m.gguf");
    let projector = directory.path().join("model-q4_k_m-mmproj.gguf");
    write_structurally_valid_mmproj(&projector);
    let projector_digest = crate::core::sha256::compute_file_sha256(&projector).unwrap();
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &projector_digest),
        ],
    );
    write_conversion_receipt(
        &text,
        &conversion_receipt(&text, "owner/model", &revision, "q4_k_m"),
    );
    write_conversion_receipt(
        &projector,
        &conversion_receipt(&projector, "owner/model", &revision, "f16-mmproj"),
    );

    let candidate = conversion_authority(&text)
        .unwrap()
        .expect("text receipt must be authoritative");
    assert_eq!(
        verify_candidate_projector(&candidate).unwrap(),
        Some(projector.clone())
    );
    assert_eq!(
        resolve_local_path_projector(&text).unwrap(),
        Some(projector.clone())
    );

    let mut warnings = Vec::new();
    let prepared = prepare_selected_local(candidate, None, &mut warnings).unwrap();
    assert!(warnings.is_empty());
    let binding = read_binding(prepared.sidecar.as_ref().unwrap())
        .unwrap()
        .unwrap();
    assert!(binding.projector.is_some());
}

#[test]
fn native_fallback_reuses_exact_conversion_and_refuses_conflicting_destination() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("native-q4_k_m.gguf");
    write_quant_gguf(&output, 15);
    let revision = "a".repeat(40);
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: revision.clone(),
        requires_projector: false,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: Vec::new(),
    };
    write_conversion_receipt(
        &output,
        &conversion_receipt(&output, "owner/model", &revision, "q4_k_m"),
    );

    let reused = native_convert(&catalog, QuantType::Q4_K_M, Some(output.as_path()), None)
        .expect("valid exact native receipt should be reused without a subprocess");
    assert_eq!(reused.path, output);

    write_conversion_receipt(
        &output,
        &conversion_receipt(&output, "other/model", &revision, "q4_k_m"),
    );
    let error = native_convert(&catalog, QuantType::Q4_K_M, Some(output.as_path()), None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("conflicts"));
}

#[test]
fn projector_receipt_and_text_digest_mismatches_fail_closed() {
    let directory = tempfile::tempdir().unwrap();
    let revision = "a".repeat(40);
    let text = directory.path().join("model-q4_k_m.gguf");
    let projector = directory.path().join("model-q4_k_m-mmproj.gguf");
    fs::write(&projector, b"wrong projector").unwrap();
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &"f".repeat(64)),
        ],
    );
    let mut receipt = conversion_receipt(&projector, "other/model", &revision, "f16-mmproj");
    write_conversion_receipt(&projector, &receipt);
    assert!(
        projector_authority_from_receipt(&projector, "owner/model", &revision)
            .unwrap()
            .is_none()
    );

    receipt.source.repository_id = "owner/model".into();
    write_conversion_receipt(&projector, &receipt);
    let candidate = Candidate {
        repository: "owner/model".into(),
        revision,
        path: text.clone(),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: Some((
            projector.clone(),
            fs::metadata(&projector).unwrap().len(),
            crate::core::sha256::compute_file_sha256(&projector).unwrap(),
        )),
        sidecar: None,
        receipt_target_identity: None,
    };
    assert!(verify_candidate_projector(&candidate).unwrap().is_none());
    assert!(resolve_local_path_projector(&text).is_err());
}

#[test]
fn bound_projector_digest_is_authenticated_before_it_can_suppress_hosted_repair() {
    let directory = tempfile::tempdir().unwrap();
    let revision = "a".repeat(40);
    let text = directory.path().join("model-q4_k_m.gguf");
    let projector = directory.path().join("model-q4_k_m-mmproj.gguf");
    write_quant_gguf(&text, 15);
    write_structurally_valid_mmproj(&projector);
    let projector_bytes = fs::metadata(&projector).unwrap().len();
    let projector_sha = crate::core::sha256::compute_file_sha256(&projector).unwrap();

    let managed = Candidate {
        repository: "owner/model".into(),
        revision: revision.clone(),
        path: text.clone(),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        quant: QuantType::Q4_K_M,
        origin: "managed".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 1,
        projector: Some((projector.clone(), projector_bytes, "f".repeat(64))),
        sidecar: None,
        receipt_target_identity: None,
    };
    assert_ne!(projector_sha, "f".repeat(64));
    assert!(verify_candidate_projector(&managed).unwrap().is_none());

    write_conversion_receipt(
        &text,
        &conversion_receipt(&text, "owner/model", &revision, "q4_k_m"),
    );
    write_conversion_receipt(
        &projector,
        &conversion_receipt(&projector, "owner/model", &revision, "f16-mmproj"),
    );
    let mut mutated = fs::read(&projector).unwrap();
    *mutated.last_mut().unwrap() ^= 1;
    fs::write(&projector, mutated).unwrap();
    let receipt_bound = conversion_authority(&text)
        .unwrap()
        .expect("text receipt authority");
    assert!(receipt_bound.projector.is_some());
    assert!(verify_candidate_projector(&receipt_bound)
        .unwrap()
        .is_none());
}

#[test]
fn retained_text_descriptor_drives_projector_metadata_through_public_swap_and_restore() {
    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("model-q4_k_m.gguf");
    let parked = directory.path().join("model-original.gguf");
    let alternate = directory.path().join("model-alternate.gguf");
    let displaced = directory.path().join("model-displaced.gguf");
    let projector_sha = "d".repeat(64);
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &projector_sha),
        ],
    );
    write_quant_gguf_with_metadata(&alternate, 15, &[("general.architecture", "llama")]);
    let authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    fs::rename(&text, &parked).unwrap();
    fs::rename(&alternate, &text).unwrap();
    let hook_text = text.clone();
    let hook_parked = parked.clone();
    let hook_displaced = displaced.clone();
    let _hook = set_after_retained_text_projector_metadata(move || {
        fs::rename(&hook_text, &hook_displaced).unwrap();
        fs::rename(&hook_parked, &hook_text).unwrap();
    });

    let error = retained_text_projector_contract(&authority)
        .unwrap_err()
        .to_string();
    assert!(error.contains("changed during retained projector planning"));
}

#[test]
fn hub_cache_loose_projector_selection_uses_retained_text_through_swap_restore() {
    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("model-q4_k_m.gguf");
    let parked = directory.path().join("model-original.gguf");
    let alternate = directory.path().join("model-alternate.gguf");
    let displaced = directory.path().join("model-displaced.gguf");
    let expected_sha = "d".repeat(64);
    let alternate_sha = "e".repeat(64);
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &expected_sha),
        ],
    );
    write_quant_gguf_with_metadata(
        &alternate,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &alternate_sha),
        ],
    );
    let authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: text.clone(),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: "f".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "hf_hub_cache_structural".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![
            HubGgufArtifact {
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                filename: "mmproj-expected.gguf".into(),
                bytes: 10,
                sha256: expected_sha,
                quant_hint: None,
                role: "companion".into(),
                selectable: false,
                unavailable_reason: Some("companion".into()),
            },
            HubGgufArtifact {
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                filename: "mmproj-alternate.gguf".into(),
                bytes: 11,
                sha256: alternate_sha,
                quant_hint: None,
                role: "companion".into(),
                selectable: false,
                unavailable_reason: Some("companion".into()),
            },
        ],
    };
    fs::rename(&text, &parked).unwrap();
    fs::rename(&alternate, &text).unwrap();
    let hook_text = text.clone();
    let hook_parked = parked.clone();
    let hook_displaced = displaced.clone();
    let _hook = set_after_retained_text_projector_metadata(move || {
        fs::rename(&hook_text, &hook_displaced).unwrap();
        fs::rename(&hook_parked, &hook_text).unwrap();
    });
    let mut selected = None;
    let error = prepare_cached_projector_in_place_with_sources(
        &mut candidate,
        &authority,
        &catalog,
        &mut |_| {},
        |_| None,
        |artifact| {
            selected = Some(artifact.filename.clone());
            bail!("stop after retained selection")
        },
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("changed during retained projector planning"));
    assert!(
        selected.is_none(),
        "no projector transfer may start after a text authority swap"
    );
}

#[test]
fn manual_loose_sibling_selection_uses_retained_text_through_swap_restore() {
    let directory = tempfile::tempdir().unwrap();
    let text = directory.path().join("model-q4_k_m.gguf");
    let parked = directory.path().join("model-original.gguf");
    let alternate = directory.path().join("model-alternate.gguf");
    let displaced = directory.path().join("model-displaced.gguf");
    let exact_projector = directory.path().join("mmproj-exact.gguf");
    let alternate_projector = directory.path().join("mmproj-alternate.gguf");
    write_structurally_valid_mmproj(&exact_projector);
    write_structurally_valid_mmproj(&alternate_projector);
    let mut alternate_bytes = fs::read(&alternate_projector).unwrap();
    *alternate_bytes.last_mut().unwrap() ^= 1;
    fs::write(&alternate_projector, alternate_bytes).unwrap();
    let exact_sha = crate::core::sha256::compute_file_sha256(&exact_projector).unwrap();
    let alternate_sha = crate::core::sha256::compute_file_sha256(&alternate_projector).unwrap();
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &exact_sha),
        ],
    );
    write_quant_gguf_with_metadata(
        &alternate,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &alternate_sha),
        ],
    );
    let authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: text.clone(),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: "f".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "manual_structural".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![HubGgufArtifact {
            repository: "owner/model".into(),
            revision: "a".repeat(40),
            filename: "mmproj-exact.gguf".into(),
            bytes: fs::metadata(&exact_projector).unwrap().len(),
            sha256: exact_sha,
            quant_hint: None,
            role: "companion".into(),
            selectable: false,
            unavailable_reason: Some("companion".into()),
        }],
    };
    fs::rename(&text, &parked).unwrap();
    fs::rename(&alternate, &text).unwrap();
    let hook_text = text.clone();
    let hook_parked = parked.clone();
    let hook_displaced = displaced.clone();
    let _hook = set_after_retained_text_projector_metadata(move || {
        fs::rename(&hook_text, &hook_displaced).unwrap();
        fs::rename(&hook_parked, &hook_text).unwrap();
    });
    let mut warnings = Vec::new();
    let selected = best_effort_manual_projector_with_catalog(
        &mut candidate,
        &authority,
        &[directory.path().to_path_buf()],
        &catalog,
        &mut warnings,
        &mut |_| {},
    );
    assert!(selected.is_none());
    assert!(warnings.iter().any(|warning| {
        warning.contains("text authority changed") && warning.contains("serving text-only")
    }));
}

#[test]
fn manual_loose_hosted_fallback_keeps_retained_digest_after_public_swap() {
    let text_root = tempfile::tempdir().unwrap();
    let sources = tempfile::tempdir().unwrap();
    let text = text_root.path().join("model-q4_k_m.gguf");
    let parked = text_root.path().join("model-original.gguf");
    let alternate = text_root.path().join("model-alternate.gguf");
    let displaced = text_root.path().join("model-displaced.gguf");
    let exact_source = sources.path().join("mmproj-exact.gguf");
    let alternate_source = sources.path().join("mmproj-alternate.gguf");
    fs::write(&exact_source, b"exact-projector").unwrap();
    fs::write(&alternate_source, b"other-projector").unwrap();
    let exact_sha = crate::core::sha256::compute_file_sha256(&exact_source).unwrap();
    let alternate_sha = crate::core::sha256::compute_file_sha256(&alternate_source).unwrap();
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &exact_sha),
        ],
    );
    write_quant_gguf_with_metadata(
        &alternate,
        15,
        &[
            ("general.architecture", "qwen3vl"),
            ("hf2q.mmproj_sha256", &alternate_sha),
        ],
    );
    let authority = crate::core::bounded_file::StableRegularFile::open_exact(
        &text,
        fs::metadata(&text).unwrap().len(),
    )
    .unwrap()
    .unwrap();
    let mut candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        path: text.clone(),
        root: text_root.path().to_path_buf(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: "f".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "manual_structural".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: vec![
            HubGgufArtifact {
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                filename: "mmproj-exact.gguf".into(),
                bytes: fs::metadata(&exact_source).unwrap().len(),
                sha256: exact_sha,
                quant_hint: None,
                role: "companion".into(),
                selectable: false,
                unavailable_reason: Some("companion".into()),
            },
            HubGgufArtifact {
                repository: "owner/model".into(),
                revision: "a".repeat(40),
                filename: "mmproj-alternate.gguf".into(),
                bytes: fs::metadata(&alternate_source).unwrap().len(),
                sha256: alternate_sha,
                quant_hint: None,
                role: "companion".into(),
                selectable: false,
                unavailable_reason: Some("companion".into()),
            },
        ],
    };
    let hook_text = text.clone();
    let hook_parked = parked.clone();
    let hook_alternate = alternate.clone();
    let _hook = set_before_manual_hosted_projector_fallback(move || {
        fs::rename(&hook_text, &hook_parked).unwrap();
        fs::rename(&hook_alternate, &hook_text).unwrap();
    });
    let mut warnings = Vec::new();
    let selected = best_effort_manual_projector_with_catalog(
        &mut candidate,
        &authority,
        &[sources.path().to_path_buf()],
        &catalog,
        &mut warnings,
        &mut |_| {},
    );
    assert_eq!(
        selected.as_deref(),
        Some(text_root.path().join("mmproj-exact.gguf").as_path())
    );
    fs::rename(&text, &displaced).unwrap();
    fs::rename(&parked, &text).unwrap();
}

#[test]
fn actual_local_selection_prefers_successful_use_and_exact_quant_ignores_low_memory() {
    let models = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let revision = "a".repeat(40);
    let mut paths = Vec::new();
    for (name, used) in [("older.gguf", 10), ("winner.gguf", 20)] {
        let path = models.path().join(name);
        write_quant_gguf(&path, 15);
        let binding = ManagedBinding {
            schema_version: SCHEMA_VERSION,
            repository: repository.into(),
            revision: revision.clone(),
            quant: "Q4_K_M".into(),
            origin: "test".into(),
            materialized_at_secs: 1,
            last_used_at_secs: used,
            artifact: ArtifactBinding {
                local_filename: name.into(),
                hub_filename: name.into(),
                bytes: fs::metadata(&path).unwrap().len(),
                sha256: crate::core::sha256::compute_file_sha256(&path).unwrap(),
            },
            projector: None,
        };
        write_binding(&sidecar_path(&path), &binding).unwrap();
        paths.push(path);
    }
    let mut cache = ModelCache::open_at(cache_dir.path()).unwrap();
    let spec = RepositoryModelSpec {
        repository: repository.into(),
        quant: Some(QuantType::Q4_K_M),
    };
    let hardware = HardwareProfile {
        chip_model: "busy-test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 0,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    reset_verify_candidate_calls();
    let resolved = resolve_repository(
        &spec,
        None,
        &[models.path().to_path_buf()],
        &mut cache,
        &hardware,
        false,
        None,
    )
    .unwrap();
    assert_eq!(resolved.gguf_path, paths[1]);
    assert_eq!(
        verify_candidate_calls(),
        1,
        "a successfully-used bound quant must receive one bounded GGUF metadata admission"
    );

    let bare_spec = RepositoryModelSpec {
        repository: repository.into(),
        quant: None,
    };
    let locally_admissible = HardwareProfile {
        available_memory_bytes: 16 << 30,
        ..hardware
    };
    reset_verify_candidate_calls();
    let resolved = resolve_repository(
        &bare_spec,
        None,
        &[models.path().to_path_buf()],
        &mut cache,
        &locally_admissible,
        false,
        Some("q5_0"),
    )
    .expect("verified admissible local bytes must win before setup fallback parsing");
    assert_eq!(resolved.gguf_path, paths[1]);
    assert_eq!(verify_candidate_calls(), 1);
}

#[test]
fn successfully_used_managed_pair_returns_before_hub_download_or_native_conversion() {
    use std::cell::Cell;

    let models = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let revision = "a".repeat(40);
    let text = models.path().join("model-q4_k_m.gguf");
    let projector = models.path().join("mmproj-model-f16.gguf");
    write_quant_gguf_with_metadata(
        &text,
        15,
        &[("hf2q.vision.projector_profile", "test-vision")],
    );
    write_structurally_valid_mmproj(&projector);
    let binding = ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: repository.into(),
        revision: revision.clone(),
        quant: "Q4_K_M".into(),
        origin: "manual_adoption".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 2,
        artifact: ArtifactBinding {
            local_filename: "model-q4_k_m.gguf".into(),
            hub_filename: "gguf/model-q4_k_m.gguf".into(),
            bytes: fs::metadata(&text).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        },
        projector: Some(ArtifactBinding {
            local_filename: "mmproj-model-f16.gguf".into(),
            hub_filename: "gguf/mmproj-model-f16.gguf".into(),
            bytes: fs::metadata(&projector).unwrap().len(),
            sha256: crate::core::sha256::compute_file_sha256(&projector).unwrap(),
        }),
    };
    write_binding(&sidecar_path(&text), &binding).unwrap();
    let mut cache = ModelCache::open_at(cache_dir.path()).unwrap();
    let hardware = HardwareProfile {
        chip_model: "test-host".into(),
        total_memory_bytes: 128 << 30,
        available_memory_bytes: 64 << 30,
        performance_cores: 1,
        efficiency_cores: 1,
        total_cores: 2,
        memory_bandwidth_gbs: 1.0,
    };
    let mut events = Vec::new();
    let catalog_calls = Cell::new(0_u8);
    reset_verify_candidate_calls();
    reset_verify_projector_calls();
    let resolved = resolve_repository_with_progress_and_catalog(
        &RepositoryModelSpec {
            repository: repository.into(),
            quant: None,
        },
        None,
        &[models.path().to_path_buf()],
        &mut cache,
        &hardware,
        true,
        None,
        &mut |event| events.push(event),
        |_| {
            catalog_calls.set(catalog_calls.get() + 1);
            panic!("a successfully-used verified pair must not query Hub metadata")
        },
    )
    .unwrap();

    assert_eq!(resolved.gguf_path, text);
    assert_eq!(resolved.mmproj_path.as_deref(), Some(projector.as_path()));
    assert_eq!(resolved.repository, repository);
    assert_eq!(resolved.revision, revision);
    assert_eq!(resolved.quant, QuantType::Q4_K_M);
    assert_eq!(resolved.origin, "manual_adoption");
    assert!(resolved.warnings.is_empty());
    assert!(resolved.activation_authority.is_some());
    assert!(resolved.mmproj_activation_authority.is_some());
    assert_eq!(catalog_calls.get(), 0);
    assert_eq!(verify_candidate_calls(), 1);
    assert_eq!(verify_projector_calls(), 1);
    assert!(matches!(
        events.first(),
        Some(StartupEvent::LocalSearch { .. })
    ));
    assert!(matches!(
        events.last(),
        Some(StartupEvent::LocalReady { .. })
    ));
    assert!(!events.iter().any(|event| matches!(
        event,
        StartupEvent::VerifyStart { .. } | StartupEvent::VerifyProgress { .. }
    )));
    assert!(!events.iter().any(|event| matches!(
        event,
        StartupEvent::HubMetadata { .. }
            | StartupEvent::HostedDownload { .. }
            | StartupEvent::NativeConversion { .. }
    )));
}

#[test]
fn managed_candidate_same_quant_swap_after_verification_is_rejected() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model-q4_k_m.gguf");
    let replacement = directory.path().join("replacement-q4_k_m.gguf");
    let prior = directory.path().join("prior-q4_k_m.gguf");
    write_quant_gguf(&path, 15);
    write_quant_gguf(&replacement, 15);
    let bytes = fs::metadata(&path).unwrap().len();
    let candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        root: directory.path().to_path_buf(),
        path: path.clone(),
        bytes,
        sha256: "0".repeat(64),
        quant: QuantType::Q4_K_M,
        origin: "managed".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 2,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let authority = verify_candidate(&candidate).unwrap();
    fs::rename(&path, &prior).unwrap();
    fs::rename(&replacement, &path).unwrap();

    let error = match candidate.into_resolved(None, Vec::new(), Some(authority)) {
        Ok(_) => panic!("same-quant replacement must not replace the verified activation inode"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("changed after bounded verification"));
}

#[test]
fn first_use_bound_candidate_hashes_once_across_catalog_and_loose_scan() {
    let models = tempfile::tempdir().unwrap();
    let aliases = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();
    let path = models.path().join("first-use.gguf");
    write_quant_gguf(&path, 15);
    let alias = aliases.path().join("same-inode-manual.gguf");
    fs::hard_link(&path, &alias).unwrap();
    let bytes = fs::metadata(&path).unwrap().len();
    let sha256 = crate::core::sha256::compute_file_sha256(&path).unwrap();
    let binding = ManagedBinding {
        schema_version: SCHEMA_VERSION,
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        quant: "Q4_K_M".into(),
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        artifact: ArtifactBinding {
            local_filename: "first-use.gguf".into(),
            hub_filename: "model-q4_k_m.gguf".into(),
            bytes,
            sha256: sha256.clone(),
        },
        projector: None,
    };
    write_binding(&sidecar_path(&path), &binding).unwrap();
    let cache = ModelCache::open_at(cache_dir.path()).unwrap();
    let mut warnings = Vec::new();
    reset_verify_candidate_calls();
    let (candidate, authority, lock) = select_local(
        &RepositoryModelSpec {
            repository: "owner/model".into(),
            quant: Some(QuantType::Q4_K_M),
        },
        &[models.path().to_path_buf(), aliases.path().to_path_buf()],
        &cache,
        None,
        64 << 30,
        64 << 30,
        &mut warnings,
    )
    .unwrap()
    .expect("bound first-use candidate");
    drop(lock);
    let identity = authority.identity();
    assert_eq!(verify_candidate_calls(), 1);

    let mut hosted_artifact = hosted(QuantType::Q4_K_M, "model-q4_k_m.gguf");
    hosted_artifact.bytes = bytes;
    hosted_artifact.sha256 = sha256;
    assert!(find_best_matching_loose(
        &[hosted_artifact],
        Some(QuantType::Q4_K_M),
        &[models.path().to_path_buf(), aliases.path().to_path_buf()],
        &[identity],
        &mut warnings,
    )
    .unwrap()
    .is_none());
    assert!(reverify_candidate_after_catalog(
        &candidate,
        identity,
        &mut warnings,
    ));
    assert_eq!(
        verify_candidate_calls(),
        1,
        "catalog lookup, the bound path, and a hard-link alias must not repeat GGUF admission"
    );
}

#[test]
fn local_candidate_mutated_during_catalog_latency_is_discarded_for_fallback() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("changed-q4_k_m.gguf");
    write_quant_gguf(&path, 15);
    let candidate = Candidate {
        repository: "owner/model".into(),
        revision: "a".repeat(40),
        root: directory.path().to_path_buf(),
        bytes: fs::metadata(&path).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&path).unwrap(),
        path: path.clone(),
        quant: QuantType::Q4_K_M,
        origin: "test".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 0,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let identity =
        crate::core::bounded_file::StableRegularFile::open_exact(&candidate.path, candidate.bytes)
            .unwrap()
            .unwrap()
            .identity();
    fs::remove_file(path).unwrap();
    let mut warnings = Vec::new();
    assert!(!reverify_candidate_after_catalog(
        &candidate,
        identity,
        &mut warnings
    ));
    assert!(warnings
        .iter()
        .any(|warning| warning.contains("continuing with a hosted/native fallback")));
}

#[test]
fn cache_selection_tracks_the_exact_quant_used_not_the_repository_lru() {
    use crate::serve::cache::{QuantEntry, SourcePointer};

    let cache_dir = tempfile::tempdir().unwrap();
    let repository = "owner/model";
    let revision = "a".repeat(40);
    let mut cache = ModelCache::open_at(cache_dir.path()).unwrap();
    cache
        .record_source(
            repository,
            &revision,
            SourcePointer::Local {
                path: cache_dir.path().join("source"),
                sha256: "0".repeat(64),
            },
        )
        .unwrap();

    let q4 = crate::serve::cache::cache_model_path(cache_dir.path(), repository, QuantType::Q4_K_M)
        .unwrap();
    fs::create_dir_all(q4.parent().unwrap()).unwrap();
    write_quant_gguf(&q4, 15);
    let q8 = crate::serve::cache::cache_model_path(cache_dir.path(), repository, QuantType::Q8_0)
        .unwrap();
    fs::create_dir_all(q8.parent().unwrap()).unwrap();
    write_quant_gguf(&q8, 7);

    for (quant, path) in [(QuantType::Q4_K_M, &q4), (QuantType::Q8_0, &q8)] {
        cache
            .record_quantized(
                repository,
                QuantEntry {
                    quant_type: quant.as_str().into(),
                    gguf_path: path.clone(),
                    mmproj_path: None,
                    bytes: fs::metadata(path).unwrap().len(),
                    sha256: crate::core::sha256::compute_file_sha256(path).unwrap(),
                    quantized_at_secs: if quant == QuantType::Q8_0 { 2 } else { 1 },
                    last_used_at_secs: 0,
                    quantized_by_version: env!("CARGO_PKG_VERSION").into(),
                },
            )
            .unwrap();
    }
    cache.touch_quant(repository, QuantType::Q4_K_M).unwrap();
    assert!(
        cache
            .lookup(repository, QuantType::Q4_K_M)
            .unwrap()
            .last_used_at_secs
            > 0
    );
    assert_eq!(
        cache
            .lookup(repository, QuantType::Q8_0)
            .unwrap()
            .last_used_at_secs,
        0,
        "a repository-level touch must not make an unused quant recent"
    );

    let mut warnings = Vec::new();
    let selected = select_local(
        &RepositoryModelSpec {
            repository: repository.into(),
            quant: None,
        },
        &[cache_dir.path().to_path_buf()],
        &cache,
        None,
        16 << 30,
        16 << 30,
        &mut warnings,
    )
    .unwrap()
    .unwrap_or_else(|| panic!("one compatible local quant must be selected: {warnings:?}"))
    .0;
    assert_eq!(selected.quant, QuantType::Q4_K_M);
    assert_eq!(
        selected.path.canonicalize().unwrap(),
        q4.canonicalize().unwrap()
    );
}
