//! Offline export → reader → checkpoint binder contract, without derivation
//! or device-heavy work. An ancestor differs deliberately from the source.
//!
//! #300 W9: the round-trip is parameterized over every `{writer, residual}
//! × {writer, residual}` site pair and both family bind contracts
//! (DeepSeek mHC residual site; Qwen dense), plus the architecture
//! dispatch matrix — the reader/bind must accept exactly what calibrate
//! writes for each family, and unsupported architectures (incl. gemma4)
//! are refused before any config parse.

use super::*;
use crate::backends::gguf::{types::MetaValue, writer::GgufWriter};
use crate::convert::receipt::{
    receipt_path, ConversionReceipt, ConverterReceipt, ExcludedDsparkReceipt, OutputReceipt,
    PeakChunkBoundReceipt, SourceReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
};
use crate::inference::glp::{validate_glp_for_model, Compatibility, GlpResidualSite};
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
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
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
    write_glp_gguf(
        &output,
        3,
        &direction,
        1.25,
        &digest,
        &checkpoint,
        GlpHookPoint::FfnOutPreResidual,
        "residual_stream_post_layer",
    )
    .unwrap();

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

/// #300 W9 — every `{writer, residual} × {writer, residual}` site pair
/// round-trips through the reader and binds on BOTH family contracts:
/// the DeepSeek mHC residual site (hook set both spec sites, widths
/// `hidden`/`hc*hidden`, project-only at the residual) and the Qwen
/// dense contract. A cross-site pair carries its honest `derived_at`;
/// a same-site pair declares matching strings.
#[test]
fn site_pair_round_trips_on_every_family_hook_set() {
    use crate::inference::glp::BoundGlp;
    use sha2::Digest;
    let device = match mlx_native::MlxDevice::new() {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping site-pair bind test (no Metal device): {error}");
            return;
        }
    };
    let dir = tempfile::tempdir().unwrap();
    let direction = [0.0f32, 0.6, -0.8];
    let raw: Vec<u8> = direction
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let digest = hex::encode(sha2::Sha256::digest(&raw));
    let checkpoint = CheckpointIdentity {
        name: Some("test-model".into()),
        organization: Some("test-org".into()),
        ..Default::default()
    };
    let hooks = [
        (GlpHookPoint::FfnOutPreResidual, "ffn_out_pre_residual"),
        (
            GlpHookPoint::ResidualStreamPostLayer,
            "residual_stream_post_layer",
        ),
    ];
    let derived_sites = ["ffn_out_pre_residual", "residual_stream_post_layer"];
    for (hook, hook_str) in hooks {
        for derived_str in derived_sites {
            let path = dir
                .path()
                .join(format!("pair-{hook_str}-{derived_str}.glp.gguf"));
            write_glp_gguf(
                &path,
                3,
                &direction,
                1.0,
                &digest,
                &checkpoint,
                hook,
                derived_str,
            )
            .unwrap();
            let vector = GlpVector::load(&path).unwrap();
            assert_eq!(vector.hook_point, hook, "hook round-trip {hook_str}");
            assert_eq!(
                vector.derived_at.as_deref(),
                Some(derived_str),
                "derived_at round-trip {derived_str}"
            );
            // The reader must accept what calibrate writes for each
            // family: bind the SAME vector on both family contracts.
            BoundGlp::bind_for_family(
                vector.clone(),
                None,
                &device,
                &[
                    GlpHookPoint::ResidualStreamPostLayer,
                    GlpHookPoint::FfnOutPreResidual,
                ],
                43,
                3,
                GlpResidualSite::MhcStreams { hc: 4 },
            )
            .unwrap_or_else(|error| panic!("deepseek bind {hook_str}/{derived_str}: {error}"));
            BoundGlp::bind(
                vector,
                None,
                &device,
                &[
                    GlpHookPoint::ResidualStreamPostLayer,
                    GlpHookPoint::FfnOutPreResidual,
                ],
                4,
                3,
            )
            .unwrap_or_else(|error| panic!("qwen bind {hook_str}/{derived_str}: {error}"));
        }
    }
}

/// #300 W9 — the dispatch matrix: qwen35/qwen35moe/deepseek4 dispatch;
/// gemma4 and unknown architectures are refused with the family set
/// named (the serve-side `--glp` refusal set — no drift between the two
/// surfaces).
#[test]
fn calibrate_dispatch_matrix_matches_the_serve_side_glp_family_set() {
    use super::family::{family_kind_for_architecture, FamilyKind};

    assert_eq!(
        family_kind_for_architecture("deepseek4").unwrap(),
        FamilyKind::Deepseek4
    );
    for architecture in ["qwen35", "qwen35moe"] {
        assert_eq!(
            family_kind_for_architecture(architecture).unwrap(),
            FamilyKind::Qwen35,
            "{architecture} dispatches to the shared Qwen adapter"
        );
    }
    for refused in ["gemma4", "llama3", "qwen3vl", ""] {
        let error = family_kind_for_architecture(refused)
            .expect_err(&format!("{refused:?} must be refused"));
        let message = format!("{error:#}");
        assert!(
            message.contains("qwen35, qwen35moe, deepseek4"),
            "refusal names the supported family set: {message}"
        );
        assert!(
            message.contains("gemma4"),
            "refusal names the gemma4 exclusion: {message}"
        );
    }
}

/// #300 W9 — `load_family` refuses an unsupported architecture BEFORE any
/// config parse or weight load (a metadata-only fixture cannot load a
/// model, so reaching the adapter would fail differently — the refusal
/// must fire first). A missing `general.architecture` is its own named
/// refusal.
#[test]
fn load_family_refuses_unsupported_architectures_before_loading() {
    let dir = tempfile::tempdir().unwrap();
    for (name, architecture) in [
        ("gemma4", Some("gemma4")),
        ("unknown", Some("coolnewarch")),
        ("missing", None),
    ] {
        let path = dir.path().join(format!("{name}.gguf"));
        let mut writer = GgufWriter::new(std::fs::File::create(&path).unwrap());
        let count = usize::from(architecture.is_some()) + 1;
        writer.write_header(0, count as u64).unwrap();
        if let Some(architecture) = architecture {
            writer
                .write_metadata_kv(
                    "general.architecture",
                    &MetaValue::String(architecture.into()),
                )
                .unwrap();
        }
        writer
            .write_metadata_kv("general.name", &MetaValue::String(name.into()))
            .unwrap();
        writer.pad_to_alignment().unwrap();
        drop(writer);

        let gguf = GgufFile::open(&path).unwrap();
        let error = match super::family::load_family(&gguf) {
            Ok(_) => panic!("{name} must be refused by load_family"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        if architecture.is_some() {
            assert!(
                message.contains("qwen35, qwen35moe, deepseek4"),
                "{name} refusal names the supported family set: {message}"
            );
        } else {
            assert!(
                message.contains("general.architecture"),
                "missing-architecture refusal names the missing key: {message}"
            );
        }
    }
}
