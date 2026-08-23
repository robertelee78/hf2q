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
fn exact_hosted_quant_wins_and_missing_exact_fails_closed() {
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
    assert!(select_hosted(&choices, Some(QuantType::Q6_K), QuantType::Q4_K_M).is_err());

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
}

#[test]
fn exact_loose_digest_disambiguates_same_quant_hosted_filenames() {
    let directory = tempfile::tempdir().unwrap();
    let local = directory.path().join("operator-q4_k_m.gguf");
    write_quant_gguf(&local, 15);
    let bytes = fs::metadata(&local).unwrap().len();
    let digest = crate::core::sha256::compute_file_sha256(&local).unwrap();
    let mut wrong = hosted(QuantType::Q4_K_M, "first-q4_k_m.gguf");
    wrong.bytes = bytes;
    wrong.sha256 = "0".repeat(64);
    let mut exact = hosted(QuantType::Q4_K_M, "second-q4_k_m.gguf");
    exact.bytes = bytes;
    exact.sha256 = digest;

    let (artifact, found) = find_best_matching_loose(
        &[wrong, exact],
        Some(QuantType::Q4_K_M),
        QuantType::Q4_K_M,
        &[directory.path().to_path_buf()],
    )
    .unwrap()
    .expect("owned bytes must disambiguate the hosted filenames");
    assert_eq!(artifact.filename, "second-q4_k_m.gguf");
    assert_eq!(found, local);
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
fn materialization_hard_links_exact_bytes_and_refuses_conflicts() {
    use std::os::unix::fs::MetadataExt;

    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.gguf");
    let destination = directory.path().join("model.gguf");
    std::fs::write(&source, b"exact hosted bytes").unwrap();
    let sha256 = crate::core::sha256::compute_file_sha256(&source).unwrap();
    materialize_exact(
        &source,
        &destination,
        "owner/model",
        std::fs::metadata(&source).unwrap().len(),
        &sha256,
    )
    .unwrap();
    assert_eq!(
        std::fs::metadata(&source).unwrap().ino(),
        std::fs::metadata(&destination).unwrap().ino()
    );

    let conflict = directory.path().join("conflict.gguf");
    std::fs::write(&conflict, b"other").unwrap();
    assert!(materialize_exact(
        &source,
        &conflict,
        "owner/model",
        std::fs::metadata(&source).unwrap().len(),
        &sha256,
    )
    .is_err());
    assert_eq!(std::fs::read(conflict).unwrap(), b"other");

    let digest_mismatch = directory.path().join("digest-mismatch.gguf");
    assert!(materialize_exact(
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
fn cross_filesystem_link_failure_uses_verified_atomic_copy() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.gguf");
    let destination = directory.path().join("copied.gguf");
    std::fs::write(&source, b"copy fallback bytes").unwrap();
    let sha256 = crate::core::sha256::compute_file_sha256(&source).unwrap();
    materialize_exact_with_link(
        &source,
        &destination,
        "owner/model",
        std::fs::metadata(&source).unwrap().len(),
        &sha256,
        |_, _| Err(std::io::Error::from_raw_os_error(libc::EXDEV)),
    )
    .unwrap();
    assert_eq!(std::fs::read(destination).unwrap(), b"copy fallback bytes");
}

#[test]
fn hard_link_publication_detects_a_same_size_post_verification_mutation() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.gguf");
    let destination = directory.path().join("managed.gguf");
    fs::write(&source, b"original").unwrap();
    let digest = crate::core::sha256::compute_file_sha256(&source).unwrap();
    let error = materialize_exact_with_link(
        &source,
        &destination,
        "owner/model",
        8,
        &digest,
        |source, destination| {
            fs::hard_link(source, destination)?;
            fs::write(source, b"mutated!")?;
            Ok(())
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("final SHA-256"));
    assert!(!destination.exists());
}

#[test]
fn concurrent_exact_destination_winner_is_rechecked_and_reused() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.gguf");
    let destination = directory.path().join("managed.gguf");
    fs::write(&source, b"exact bytes").unwrap();
    let digest = crate::core::sha256::compute_file_sha256(&source).unwrap();
    materialize_exact_with_link(
        &source,
        &destination,
        "owner/model",
        11,
        &digest,
        |source, destination| {
            fs::copy(source, destination)?;
            Err(std::io::Error::from(std::io::ErrorKind::AlreadyExists))
        },
    )
    .unwrap();
    assert_eq!(fs::read(destination).unwrap(), b"exact bytes");
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
    materialize_exact(
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
    let resolved = resolve_repository(
        &spec,
        None,
        &[models.path().to_path_buf()],
        &mut cache,
        &hardware,
        false,
    )
    .unwrap();
    assert_eq!(resolved.gguf_path, paths[1]);
}
