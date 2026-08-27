//! Automatic multimodal text + projector conversion (ADR-004 §4).
//!
//! The projector is completed first and its digest is embedded in the text
//! GGUF as `hf2q.mmproj_sha256`. The text GGUF is promoted after the
//! projector and every optional receipt, so it remains the sole commit
//! marker. A durable journal and same-filesystem backups keep the previous
//! complete pair recoverable until that final text rename is durable.

use std::fs;
use std::path::{Path, PathBuf};

use super::paired_transaction::PairWorkspace;
use super::{
    detect_arch, run_convert_internal, ConvertArgs, ConvertError, ConvertMode, PairBinding,
};
use crate::convert::quant_selector::QuantSelector;
use crate::convert::receipt::{receipt_path, ConversionReceipt, RemoteConversionSource};
use crate::convert::tensor_lineage::tensor_conversion_receipt_path;
use crate::convert::HfModelSource;
use crate::core::paired_artifact::{
    canonical_parent, file_name, PairMemberRole, KEY_PAIR_GENERATION,
};
use crate::core::provenance::KEY_MMPROJ_SHA256;
use crate::core::sha256::compute_file_sha256;
use crate::models::vit::VisionConfig;
use crate::quantize::ggml_quants::{is_vision_tensor_pattern, ArchName, GgufFtype};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectorEmitter {
    NativeVit,
    GemmaMapper,
}

pub(super) fn run(mut args: ConvertArgs) -> Result<(), ConvertError> {
    let requested_projector_output = match &args.mode {
        ConvertMode::TextOnly | ConvertMode::ProjectorOnly => {
            return run_convert_internal(args, None, PairBinding::default(), true).map(|_| ());
        }
        ConvertMode::Paired { projector_output } => projector_output.clone(),
    };

    // This metadata-only source index proves that vision_config is backed by
    // actual vision tensors before either output is touched.
    let source = HfModelSource::open(&args.hf_dir)?;
    let has_vision_config = source.config.get("vision_config").is_some();
    let has_vision_tensors = source
        .tensor_metas()
        .any(|tensor| is_vision_tensor_pattern(&tensor.name));
    if !has_vision_config {
        if requested_projector_output.is_some() {
            return Err(pair_error(
                "--mmproj-output requires a source with vision_config",
            ));
        }
        if has_vision_tensors {
            return Err(pair_error(
                "source contains vision tensors but has no vision_config; refusing a silent text-only conversion (pass --text-only only when that omission is intentional)",
            ));
        }
        args.mode = ConvertMode::TextOnly;
        return run_convert_internal(args, None, PairBinding::default(), true).map(|_| ());
    }

    if !has_vision_tensors {
        return Err(pair_error(
            "source has vision_config but no vision tensors; refusing a silent text-only conversion (pass --text-only only when that omission is intentional)",
        ));
    }
    let arch = detect_arch(&source.config)?;
    let emitter = projector_emitter(arch)?;
    let requires_processor_config = emitter == ProjectorEmitter::NativeVit
        && VisionConfig::from_hf_config(&source.config)
            .map_err(crate::models::vit::VitConvertError::from)?
            .is_qwen_vision();
    if requires_processor_config && !args.hf_dir.join("preprocessor_config.json").is_file() {
        return Err(pair_error(format!(
            "Qwen multimodal conversion requires {} before either pair member is written",
            args.hf_dir.join("preprocessor_config.json").display()
        )));
    }

    let projector_output = requested_projector_output
        .map(Ok)
        .unwrap_or_else(|| default_projector_output(&args.output))?;
    validate_pair_paths(&args.output, &projector_output)?;

    if args.dry_run {
        tracing::info!(
            target: "convert",
            text_output = %args.output.display(),
            projector_output = %projector_output.display(),
            projector_quant = "f16-mmproj",
            emitter = ?emitter,
            "multimodal dry run plans a text + projector pair; no output is written"
        );
        args.mode = ConvertMode::TextOnly;
        return run_convert_internal(args, None, PairBinding::default(), true).map(|_| ());
    }

    drop(source);
    run_pair(args, projector_output)
}

fn projector_emitter(arch: ArchName) -> Result<ProjectorEmitter, ConvertError> {
    match arch {
        ArchName::Qwen35 | ArchName::Qwen35MoeFull => {
            Ok(ProjectorEmitter::NativeVit)
        }
        ArchName::Gemma4 => Ok(ProjectorEmitter::GemmaMapper),
        other => Err(ConvertError::UnsupportedArch {
            arch_name: format!(
                "{other:?} has vision_config but no automatic projector converter; pass --text-only only when vision is intentionally excluded"
            ),
        }),
    }
}

fn default_projector_output(text_output: &Path) -> Result<PathBuf, ConvertError> {
    let file_name = text_output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| pair_error("text output must have a UTF-8 filename"))?;
    let stem = file_name
        .strip_suffix(".gguf")
        .or_else(|| file_name.strip_suffix(".GGUF"))
        .ok_or_else(|| pair_error("automatic paired output requires a .gguf text output"))?;
    if stem.ends_with("-mmproj") {
        return Err(pair_error(
            "text output cannot already end in -mmproj.gguf in automatic pair mode",
        ));
    }
    Ok(text_output.with_file_name(format!("{stem}-mmproj.gguf")))
}

fn validate_pair_paths(text: &Path, projector: &Path) -> Result<(), ConvertError> {
    if text == projector {
        return Err(pair_error(
            "text and projector outputs must be different files",
        ));
    }
    if normalized_parent(text) != normalized_parent(projector) {
        return Err(pair_error(
            "paired text and projector outputs must share one directory",
        ));
    }
    let destinations = [
        text.to_path_buf(),
        receipt_path(text),
        tensor_conversion_receipt_path(text),
        projector.to_path_buf(),
        receipt_path(projector),
        tensor_conversion_receipt_path(projector),
    ];
    for (index, path) in destinations.iter().enumerate() {
        if destinations[index + 1..].contains(path) {
            return Err(pair_error(format!(
                "paired conversion destination collision at {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn normalized_parent(path: &Path) -> PathBuf {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn run_pair(args: ConvertArgs, projector_output: PathBuf) -> Result<(), ConvertError> {
    let text_output = args.output.clone();
    let parent = normalized_parent(&text_output);
    fs::create_dir_all(&parent)?;
    let canonical_output_parent =
        canonical_parent(&text_output).map_err(|error| pair_error(error.to_string()))?;
    if canonical_parent(&projector_output).map_err(|error| pair_error(error.to_string()))?
        != canonical_output_parent
    {
        return Err(pair_error(
            "paired text and projector outputs must resolve to the same directory",
        ));
    }
    let canonical_text_output = canonical_output_parent
        .join(file_name(&text_output).map_err(|error| pair_error(error.to_string()))?);
    let canonical_projector_output = canonical_output_parent
        .join(file_name(&projector_output).map_err(|error| pair_error(error.to_string()))?);
    validate_pair_paths(&canonical_text_output, &canonical_projector_output)?;
    let workspace = PairWorkspace::create(&canonical_text_output)
        .map_err(|error| pair_error(error.to_string()))?;
    let staged_projector = workspace.staged_path(PairMemberRole::Projector);
    let staged_text = workspace.staged_path(PairMemberRole::Text);

    preflight_pair_plan(
        &args,
        workspace.transaction_id(),
        &staged_text,
        &canonical_text_output,
    )?;

    let result = run_pair_in_workspace(
        &args,
        &workspace,
        &staged_projector,
        &staged_text,
        &text_output,
        &projector_output,
        &canonical_text_output,
        &canonical_projector_output,
    );
    if result.is_err() {
        workspace.discard_unpublished();
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn run_pair_in_workspace(
    args: &ConvertArgs,
    workspace: &PairWorkspace,
    staged_projector: &Path,
    staged_text: &Path,
    text_output: &Path,
    projector_output: &Path,
    canonical_text_output: &Path,
    canonical_projector_output: &Path,
) -> Result<(), ConvertError> {
    // Projector first: its exact digest is part of the text artifact's
    // immutable consumer contract.
    let mut projector_args = args.clone();
    projector_args.selector = QuantSelector::Standard(GgufFtype::MostlyF16);
    projector_args.output = staged_projector.to_path_buf();
    projector_args.mode = ConvertMode::ProjectorOnly;
    projector_args.imatrix = None;
    projector_args.imatrix_corpus = None;
    projector_args.imatrix_out = None;
    projector_args.imatrix_n_ctx = None;
    run_convert_internal(
        projector_args,
        None,
        PairBinding {
            projector_sha256: None,
            generation: Some(workspace.transaction_id()),
        },
        true,
    )?;
    let staged_projector_receipt = retarget_receipt(staged_projector, projector_output)?;
    let projector_sha256 = fresh_artifact_sha256(
        staged_projector,
        staged_projector_receipt.as_deref(),
        projector_output,
        "f16-mmproj",
    )?;

    let mut text_args = args.clone();
    text_args.output = staged_text.to_path_buf();
    text_args.mode = ConvertMode::TextOnly;
    run_convert_internal(
        text_args,
        None,
        PairBinding {
            projector_sha256: Some(&projector_sha256),
            generation: Some(workspace.transaction_id()),
        },
        true,
    )?;
    let staged_text_receipt = retarget_receipt(staged_text, text_output)?;
    validate_bound_text(staged_text, &projector_sha256, workspace.transaction_id())?;
    validate_bound_projector(staged_projector, workspace.transaction_id())?;
    validate_receipt_pair(
        args,
        staged_text_receipt.as_deref(),
        staged_projector_receipt.as_deref(),
    )?;

    let destinations = vec![
        (
            PairMemberRole::Projector,
            canonical_projector_output.to_path_buf(),
        ),
        (
            PairMemberRole::ProjectorReceipt,
            receipt_path(canonical_projector_output),
        ),
        (
            PairMemberRole::ProjectorTensorReceipt,
            tensor_conversion_receipt_path(canonical_projector_output),
        ),
        (
            PairMemberRole::TextReceipt,
            receipt_path(canonical_text_output),
        ),
        (
            PairMemberRole::TextTensorReceipt,
            tensor_conversion_receipt_path(canonical_text_output),
        ),
        (PairMemberRole::Text, canonical_text_output.to_path_buf()),
    ];
    if args.no_clobber {
        workspace
            .publish_no_clobber(&destinations)
            .map_err(|error| pair_error(error.to_string()))?;
    } else {
        workspace
            .publish(&destinations)
            .map_err(|error| pair_error(error.to_string()))?;
    }

    tracing::info!(
        target: "convert",
        text_output = %text_output.display(),
        projector_output = %projector_output.display(),
        projector_sha256 = %projector_sha256,
        "published source-bound multimodal text + projector pair"
    );
    Ok(())
}

fn preflight_pair_plan(
    args: &ConvertArgs,
    generation: &str,
    staged_text: &Path,
    destination: &Path,
) -> Result<(), ConvertError> {
    let planned_projector = crate::models::vit::planned_vision_tower_output_bytes(
        &args.hf_dir,
        args.remote_source
            .as_ref()
            .map(RemoteConversionSource::source_sha256),
        Some(generation),
    )?;
    let placeholder_projector_sha = "0".repeat(64);
    let mut text_plan_args = args.clone();
    text_plan_args.output = staged_text.to_path_buf();
    text_plan_args.mode = ConvertMode::TextOnly;
    text_plan_args.dry_run = true;
    text_plan_args.imatrix_out = None;
    let planned_text = run_convert_internal(
        text_plan_args,
        None,
        PairBinding {
            projector_sha256: Some(&placeholder_projector_sha),
            generation: Some(generation),
        },
        false,
    )?
    .planned_output_bytes;
    let total = planned_text
        .checked_add(planned_projector)
        .ok_or_else(|| pair_error("paired conversion output plan exceeds u64"))?;
    let label = args
        .remote_source
        .as_ref()
        .map(|source| source.reference().repo_id())
        .unwrap_or("local multimodal model");
    crate::input::hf_download::check_conversion_output_preflight(label, destination, total)
        .map_err(ConvertError::HfDownload)
}

fn retarget_receipt(
    staged_output: &Path,
    final_output: &Path,
) -> Result<Option<PathBuf>, ConvertError> {
    let path = receipt_path(staged_output);
    if !path.exists() {
        return Ok(None);
    }
    let mut receipt: ConversionReceipt = serde_json::from_slice(&fs::read(&path)?)
        .map_err(|error| pair_error(format!("parse staged receipt: {error}")))?;
    receipt.output.path = final_output.display().to_string();
    write_receipt_synced(&path, &receipt)?;
    Ok(Some(path))
}

fn write_receipt_synced(path: &Path, receipt: &ConversionReceipt) -> Result<(), ConvertError> {
    use std::io::Write;

    let parent = normalized_parent(path);
    let mut temporary = tempfile::NamedTempFile::new_in(&parent)?;
    serde_json::to_writer_pretty(&mut temporary, receipt)
        .map_err(|error| pair_error(format!("serialize staged receipt: {error}")))?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .map_err(|error| ConvertError::Io(error.error))?;
    Ok(())
}

fn fresh_artifact_sha256(
    staged_artifact: &Path,
    staged_receipt: Option<&Path>,
    final_artifact: &Path,
    expected_quant: &str,
) -> Result<String, ConvertError> {
    let metadata = fs::metadata(staged_artifact)?;
    let Some(path) = staged_receipt else {
        return Ok(compute_file_sha256(staged_artifact)?);
    };
    let receipt: ConversionReceipt = serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| pair_error(format!("parse staged receipt: {error}")))?;
    if receipt.output.path != final_artifact.display().to_string()
        || receipt.output.size != metadata.len()
        || receipt.quant_selector != expected_quant
    {
        return Err(pair_error(
            "fresh projector receipt does not describe the staged projector contract",
        ));
    }
    Ok(receipt.output.sha256)
}

fn validate_bound_text(
    text: &Path,
    projector_sha256: &str,
    generation: &str,
) -> Result<(), ConvertError> {
    let gguf = mlx_native::gguf::GgufFile::open(text)
        .map_err(|error| pair_error(format!("reopen staged text GGUF: {error}")))?;
    if gguf.metadata_string(KEY_MMPROJ_SHA256) != Some(projector_sha256) {
        return Err(pair_error(
            "staged text GGUF does not contain the exact projector digest binding",
        ));
    }
    if gguf.metadata_string(KEY_PAIR_GENERATION) != Some(generation) {
        return Err(pair_error(
            "staged text GGUF does not contain the pair generation",
        ));
    }
    Ok(())
}

fn validate_bound_projector(projector: &Path, generation: &str) -> Result<(), ConvertError> {
    let gguf = mlx_native::gguf::GgufFile::open(projector)
        .map_err(|error| pair_error(format!("reopen staged projector GGUF: {error}")))?;
    if gguf.metadata_string(KEY_PAIR_GENERATION) != Some(generation) {
        return Err(pair_error(
            "staged projector GGUF does not contain the pair generation",
        ));
    }
    Ok(())
}

fn validate_receipt_pair(
    args: &ConvertArgs,
    text_path: Option<&Path>,
    projector_path: Option<&Path>,
) -> Result<(), ConvertError> {
    let Some(remote) = args.remote_source.as_ref() else {
        return Ok(());
    };
    let text_path = text_path.ok_or_else(|| pair_error("remote text artifact has no receipt"))?;
    let projector_path =
        projector_path.ok_or_else(|| pair_error("remote projector artifact has no receipt"))?;
    let text: ConversionReceipt = serde_json::from_slice(&fs::read(text_path)?)
        .map_err(|error| pair_error(format!("parse staged text receipt: {error}")))?;
    let projector: ConversionReceipt = serde_json::from_slice(&fs::read(projector_path)?)
        .map_err(|error| pair_error(format!("parse staged projector receipt: {error}")))?;
    if text.source != projector.source
        || text.converter != projector.converter
        || text.source.repository_id != remote.reference().repo_id()
        || text.source.revision != remote.reference().revision()
        || text.source.bundle_sha256 != remote.source_sha256()
        || projector.quant_selector != "f16-mmproj"
    {
        return Err(pair_error(
            "text and projector receipts do not share the exact verified source, converter, and projector quant contract",
        ));
    }
    Ok(())
}

fn pair_error(detail: impl Into<String>) -> ConvertError {
    ConvertError::Pair {
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::receipt::{
        ConverterReceipt, ExcludedDsparkReceipt, OutputReceipt, PeakChunkBoundReceipt,
        RemoteConversionSource, SourceReceipt, CONVERSION_RECEIPT_SCHEMA_VERSION,
    };

    fn remote_test_args(output: PathBuf) -> ConvertArgs {
        let reference = crate::input::hf_reference::HfModelReference::parse(
            "org/source-multimodal",
            Some("main"),
        )
        .unwrap()
        .resolve(&"a".repeat(40))
        .unwrap();
        ConvertArgs {
            hf_dir: PathBuf::from("/cache/exact-source"),
            selector: QuantSelector::Standard(GgufFtype::MostlyQ8_0),
            output,
            no_clobber: false,
            dry_run: false,
            imatrix: None,
            imatrix_corpus: None,
            imatrix_out: None,
            imatrix_n_ctx: None,
            mode: ConvertMode::Paired {
                projector_output: None,
            },
            remote_source: Some(RemoteConversionSource::for_test(reference, "b".repeat(64))),
        }
    }

    fn remote_test_receipt(args: &ConvertArgs, quant_selector: &str) -> ConversionReceipt {
        let remote = args.remote_source.as_ref().unwrap();
        ConversionReceipt {
            schema_version: CONVERSION_RECEIPT_SCHEMA_VERSION,
            source: SourceReceipt {
                original_reference: remote.reference().original().into(),
                repository_id: remote.reference().repo_id().into(),
                repository_type: remote.reference().repository_type().as_str().into(),
                canonical_url: remote.reference().canonical_url().into(),
                revision: remote.reference().revision().into(),
                filename: remote.reference().filename().map(str::to_owned),
                bundle_sha256: remote.source_sha256().into(),
                files: Vec::new(),
            },
            converter: ConverterReceipt {
                package: "hf2q".into(),
                version: "0.1.7".into(),
                git_commit: "c".repeat(40),
            },
            quant_selector: quant_selector.into(),
            output: OutputReceipt {
                path: args.output.display().to_string(),
                size: 1,
                sha256: "d".repeat(64),
            },
            excluded_dspark: ExcludedDsparkReceipt {
                tensor_count: 0,
                status: "none_detected".into(),
            },
            peak_chunk_bound: PeakChunkBoundReceipt::default(),
        }
    }

    fn write_receipt(path: &Path, receipt: &ConversionReceipt) {
        fs::write(path, serde_json::to_vec(receipt).unwrap()).unwrap();
    }

    #[test]
    fn default_projector_path_is_a_deterministic_sibling() {
        assert_eq!(
            default_projector_output(Path::new("/models/model-Q4_K_M.gguf")).unwrap(),
            PathBuf::from("/models/model-Q4_K_M-mmproj.gguf")
        );
        assert!(default_projector_output(Path::new("/models/model.bin")).is_err());
        assert!(default_projector_output(Path::new("/models/model-mmproj.gguf")).is_err());
    }

    #[test]
    fn automatic_projector_emitters_are_closed_and_explicit() {
        assert_eq!(
            projector_emitter(ArchName::Qwen35).unwrap(),
            ProjectorEmitter::NativeVit
        );
        assert_eq!(
            projector_emitter(ArchName::Gemma4).unwrap(),
            ProjectorEmitter::GemmaMapper
        );
        assert!(projector_emitter(ArchName::Llama3).is_err());
    }

    #[test]
    fn remote_receipts_require_one_exact_source_and_projector_contract() {
        let dir = tempfile::tempdir().unwrap();
        let args = remote_test_args(dir.path().join("model.gguf"));
        let text_receipt_path = dir.path().join("staged-text.receipt.json");
        let projector_receipt_path = dir.path().join("staged-projector.receipt.json");
        write_receipt(&text_receipt_path, &remote_test_receipt(&args, "q8_0"));
        write_receipt(
            &projector_receipt_path,
            &remote_test_receipt(&args, "f16-mmproj"),
        );

        validate_receipt_pair(
            &args,
            Some(&text_receipt_path),
            Some(&projector_receipt_path),
        )
        .unwrap();

        let mut mismatched = remote_test_receipt(&args, "f16-mmproj");
        mismatched.source.bundle_sha256 = "e".repeat(64);
        write_receipt(&projector_receipt_path, &mismatched);
        assert!(validate_receipt_pair(
            &args,
            Some(&text_receipt_path),
            Some(&projector_receipt_path),
        )
        .is_err());

        write_receipt(&projector_receipt_path, &remote_test_receipt(&args, "q8_0"));
        assert!(validate_receipt_pair(
            &args,
            Some(&text_receipt_path),
            Some(&projector_receipt_path),
        )
        .is_err());
    }
}
