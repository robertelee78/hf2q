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
    gguf.resize(256, 0);
    fs::write(path, gguf).unwrap();
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
    };
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
        Some(text_root.path()),
        &[loose_root.path().to_path_buf()],
        &head_catalog,
        true,
        &mut warnings,
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
    fs::write(&projector, b"verified projector").unwrap();
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
    fs::write(&projector, b"verified projector").unwrap();
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
    fs::write(&projector, b"verified projector").unwrap();
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
    mark_successful_use("owner/model", QuantType::Q4_K_M, &artifact, &mut cache).unwrap();
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
    fs::write(&projector, b"projector bytes").unwrap();
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
    };
    assert!(verify_candidate_projector(&candidate).unwrap().is_none());
    assert!(resolve_local_path_projector(&text).is_err());
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
        "a successfully-used bound quant must be hashed exactly once on repeat serve"
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
    let (candidate, identity, lock) = select_local(
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
        "catalog lookup, the bound path, and a hard-link alias must not trigger another full hash"
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
    .unwrap_or_else(|| panic!("one verified local quant must be selected: {warnings:?}"))
    .0;
    assert_eq!(selected.quant, QuantType::Q4_K_M);
    assert_eq!(
        selected.path.canonicalize().unwrap(),
        q4.canonicalize().unwrap()
    );
}
