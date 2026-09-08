use super::*;

fn fixture(directory: &Path, linked: bool) -> (Candidate, HubGgufCatalog) {
    let blob = directory.join("blob");
    write_quant_gguf(&blob, 15);
    let text = directory.join("model-UD-Q4_K_M.gguf");
    if linked {
        std::os::unix::fs::symlink(&blob, &text).unwrap();
    } else {
        fs::rename(&blob, &text).unwrap();
    }
    let candidate = Candidate {
        hub_filename: Some("model-UD-Q4_K_M.gguf".into()),
        repository: "activation-regression/model".into(),
        revision: "a".repeat(40),
        root: directory.to_path_buf(),
        path: text.clone(),
        bytes: fs::metadata(&text).unwrap().len(),
        sha256: crate::core::sha256::compute_file_sha256(&text).unwrap(),
        quant: QuantType::Q4_K_M,
        origin: "hosted_download".into(),
        materialized_at_secs: 1,
        last_used_at_secs: 2,
        projector: None,
        sidecar: None,
        receipt_target_identity: None,
    };
    let catalog = HubGgufCatalog {
        schema_version: "hf2q.hub-gguf-catalog.v2".into(),
        repository: candidate.repository.clone(),
        revision: candidate.revision.clone(),
        requires_projector: true,
        source_weight_bytes: None,
        source_uncached_weight_bytes: None,
        artifacts: Vec::new(),
    };
    (candidate, catalog)
}

#[test]
fn automatic_projector_planning_preserves_selected_text_authority() {
    for linked in [false, true] {
        for companion_count in [0, 1, 2] {
            let directory = tempfile::tempdir().unwrap();
            let sources = tempfile::tempdir().unwrap();
            let (candidate, mut catalog) = fixture(directory.path(), linked);
            let original = candidate.path.clone();
            let authority = crate::core::bounded_file::StableRegularFile::open_operator_path_exact(
                &original,
                candidate.bytes,
            )
            .unwrap()
            .unwrap();
            for index in 0..companion_count {
                let path = sources.path().join(format!("mmproj-{index}.gguf"));
                write_structurally_valid_mmproj(&path);
                catalog.artifacts.push(HubGgufArtifact {
                    repository: candidate.repository.clone(),
                    revision: candidate.revision.clone(),
                    filename: path.file_name().unwrap().to_str().unwrap().into(),
                    bytes: fs::metadata(&path).unwrap().len(),
                    sha256: crate::core::sha256::compute_file_sha256(&path).unwrap(),
                    quant_hint: None,
                    role: "companion".into(),
                    selectable: false,
                    unavailable_reason: Some("companion".into()),
                });
            }
            let mut warnings = Vec::new();
            let (prepared, projector) = prepare_local_candidate_with_catalog_resolver(
                candidate,
                &authority,
                None,
                &[sources.path().to_path_buf()],
                &catalog,
                true,
                &mut warnings,
                &mut |_| {},
                |_| panic!("exact revision catalog is already available"),
            )
            .unwrap();
            assert_eq!(
                prepared.path, original,
                "automatic projector work must serve text in place"
            );
            assert_eq!(projector.is_some(), companion_count == 1);
            if let Some(path) = projector.as_ref() {
                assert_eq!(path.parent(), original.parent());
            } else {
                assert!(warnings.iter().any(|warning| warning.contains("text-only")));
            }
            let resolved = prepared
                .into_resolved(projector, warnings, Some(authority))
                .unwrap();
            let authority = resolved.activation_authority.unwrap();
            assert!(authority.is_stable().unwrap());
            assert!(
                crate::core::bounded_file::regular_path_matches_identity(
                    &resolved.gguf_path,
                    authority.identity(),
                )
                .unwrap(),
                "resolver result must pass the runtime activation check"
            );
            assert_eq!(
                fs::read(authority.activation_path().unwrap()).unwrap(),
                fs::read(original).unwrap()
            );
        }
    }
}

#[test]
fn resolver_rejects_different_path_with_identical_bytes_and_stable_original() {
    let directory = tempfile::tempdir().unwrap();
    let (mut candidate, _) = fixture(directory.path(), false);
    let authority =
        crate::core::bounded_file::StableRegularFile::open_exact(&candidate.path, candidate.bytes)
            .unwrap()
            .unwrap();
    let other = directory.path().join("copy.gguf");
    fs::copy(&candidate.path, &other).unwrap();
    candidate.path = other;
    assert!(authority.is_stable().unwrap());
    assert!(
        candidate
            .into_resolved(None, Vec::new(), Some(authority))
            .is_err(),
        "a stable original descriptor cannot authorize a different destination inode"
    );
}
