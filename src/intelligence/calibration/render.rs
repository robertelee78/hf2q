use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::manifest::{ordered_evidence_json_sha256, sha256_bytes};
use super::types::*;

fn checked_u32(value: usize, what: &str) -> Result<u32, CalibrationInputError> {
    u32::try_from(value).map_err(|_| {
        CalibrationInputError::InvalidDataset(format!("{what} does not fit the evidence format"))
    })
}

fn checked_u64(value: usize, what: &str) -> Result<u64, CalibrationInputError> {
    u64::try_from(value).map_err(|_| {
        CalibrationInputError::InvalidDataset(format!("{what} does not fit the evidence format"))
    })
}

fn frame_bytes(
    hasher: &mut Sha256,
    tag: &[u8],
    id: &str,
    payload: &[u8],
) -> Result<(), CalibrationInputError> {
    hasher.update(checked_u32(tag.len(), "frame tag length")?.to_le_bytes());
    hasher.update(tag);
    hasher.update(checked_u32(id.len(), "stable id length")?.to_le_bytes());
    hasher.update(id.as_bytes());
    hasher.update(checked_u64(payload.len(), "frame payload length")?.to_le_bytes());
    hasher.update(payload);
    Ok(())
}

fn framed_token_bytes(id: &str, ids: &[u32]) -> Result<Vec<u8>, CalibrationInputError> {
    let token_bytes = ids.len().checked_mul(4).ok_or_else(|| {
        CalibrationInputError::InvalidDataset("token evidence size overflow".into())
    })?;
    let capacity = 24usize
        .checked_add(id.len())
        .and_then(|value| value.checked_add(token_bytes))
        .ok_or_else(|| {
            CalibrationInputError::InvalidDataset("token evidence size overflow".into())
        })?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(b"hf2q-token-ids-v1");
    out.extend_from_slice(&checked_u32(id.len(), "stable id length")?.to_le_bytes());
    out.extend_from_slice(id.as_bytes());
    out.extend_from_slice(&checked_u64(ids.len(), "token count")?.to_le_bytes());
    for token in ids {
        out.extend_from_slice(&token.to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
pub(super) fn framed_token_bytes_for_test(id: &str, ids: &[u32]) -> Vec<u8> {
    framed_token_bytes(id, ids).expect("small test token sequence is representable")
}

fn token_window_hashes(
    stable_id: &str,
    ids: &[u32],
    width: usize,
) -> Result<Vec<String>, CalibrationInputError> {
    if ids.len() < width {
        return Err(CalibrationInputError::TokenWindowTooShort {
            stable_id: stable_id.to_owned(),
            tokens: ids.len(),
            width,
        });
    }
    let encoded_width = checked_u64(width, "token-window width")?;
    ids.windows(width)
        .map(|window| {
            let token_bytes = window.len().checked_mul(4).ok_or_else(|| {
                CalibrationInputError::InvalidDataset("token-window size overflow".into())
            })?;
            let capacity = 24usize.checked_add(token_bytes).ok_or_else(|| {
                CalibrationInputError::InvalidDataset("token-window size overflow".into())
            })?;
            let mut bytes = Vec::with_capacity(capacity);
            bytes.extend_from_slice(b"hf2q-token-window-v1");
            bytes.extend_from_slice(&encoded_width.to_le_bytes());
            for token in window {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
            Ok(sha256_bytes(&bytes))
        })
        .collect()
}

#[cfg(test)]
pub(super) fn token_window_hashes_for_test(
    stable_id: &str,
    ids: &[u32],
    width: usize,
) -> Result<Vec<String>, CalibrationInputError> {
    token_window_hashes(stable_id, ids, width)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn source_valid(
    source: &crate::intelligence::measured_auto_quant::SourceIdentity,
) -> bool {
    !source.model_id.is_empty()
        && !source.revision.is_empty()
        && is_sha256(&source.config_sha256)
        && is_sha256(&source.tensor_bundle_sha256)
        && is_sha256(&source.tokenizer_bundle_sha256)
        && is_sha256(&source.chat_template_sha256)
}

fn kwargs_map(
    kwargs: &BTreeMap<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    kwargs
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn encode(
    tokenizer: &tokenizers::Tokenizer,
    stable_id: &str,
    rendered: &str,
) -> Result<Vec<u32>, CalibrationInputError> {
    tokenizer
        .encode(rendered, false)
        .map(|encoding| encoding.get_ids().to_vec())
        .map_err(|error| CalibrationInputError::Tokenize {
            stable_id: stable_id.to_owned(),
            detail: error.to_string(),
        })
}

#[derive(Serialize)]
struct RenderedManifestHashView<'a> {
    schema_version: u32,
    split: DatasetSplit,
    source: &'a crate::intelligence::measured_auto_quant::SourceIdentity,
    verified_source_manifest_sha256: &'a str,
    structured_dataset_sha256: &'a str,
    chat_template_source: crate::core::chat_template_resolver::ChatTemplateSource,
    chat_template_sha256: &'a str,
    tokenizer_json_sha256: &'a str,
    renderer_revision: &'a str,
    max_tokens_per_example: usize,
    token_window_size: usize,
    examples: &'a [RenderedExampleReceipt],
    rendered_text_stream_sha256: &'a str,
    token_id_stream_sha256: &'a str,
}

fn rendered_manifest_sha256(
    manifest: &RenderedDatasetManifest,
) -> Result<String, CalibrationInputError> {
    ordered_evidence_json_sha256(&RenderedManifestHashView {
        schema_version: manifest.schema_version,
        split: manifest.split,
        source: &manifest.source,
        verified_source_manifest_sha256: &manifest.verified_source_manifest_sha256,
        structured_dataset_sha256: &manifest.structured_dataset_sha256,
        chat_template_source: manifest.chat_template_source,
        chat_template_sha256: &manifest.chat_template_sha256,
        tokenizer_json_sha256: &manifest.tokenizer_json_sha256,
        renderer_revision: &manifest.renderer_revision,
        max_tokens_per_example: manifest.max_tokens_per_example,
        token_window_size: manifest.token_window_size,
        examples: &manifest.examples,
        rendered_text_stream_sha256: &manifest.rendered_text_stream_sha256,
        token_id_stream_sha256: &manifest.token_id_stream_sha256,
    })
}

/// Recompute every public receipt field from the retained structured,
/// rendered, and token material. No caller-provided digest is trusted.
pub(super) fn validate_rendered_dataset(
    dataset: &RenderedDataset,
) -> Result<(), CalibrationInputError> {
    super::manifest::validate_structured_dataset_manifest(&dataset.structured)?;
    let manifest = &dataset.manifest;
    if manifest.schema_version != CALIBRATION_INPUT_SCHEMA_VERSION
        || manifest.split != dataset.structured.split
        || manifest.structured_dataset_sha256 != dataset.structured.manifest_sha256
        || manifest.examples.is_empty()
        || manifest.max_tokens_per_example == 0
        || manifest.token_window_size == 0
        || manifest.renderer_revision.is_empty()
        || !source_valid(&manifest.source)
        || dataset.token_ids.len() != manifest.examples.len()
        || dataset.rendered_utf8.len() != manifest.examples.len()
        || !is_sha256(&manifest.chat_template_sha256)
        || !is_sha256(&manifest.verified_source_manifest_sha256)
        || !is_sha256(&manifest.tokenizer_json_sha256)
        || !is_sha256(&manifest.rendered_text_stream_sha256)
        || !is_sha256(&manifest.token_id_stream_sha256)
    {
        return Err(CalibrationInputError::InvalidDataset(
            "rendered dataset identity or dimensions are invalid".into(),
        ));
    }

    let mut rendered_stream = Sha256::new();
    let mut token_stream = Sha256::new();
    let mut stable_ids = BTreeSet::new();
    for ((receipt, expected_id), structured) in manifest
        .examples
        .iter()
        .zip(&dataset.structured.example_order)
        .zip(&dataset.structured.examples)
    {
        let Some(ids) = dataset.token_ids.get(&receipt.stable_id) else {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "missing token ids for {}",
                receipt.stable_id
            )));
        };
        let Some(rendered) = dataset.rendered_utf8.get(&receipt.stable_id) else {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "missing rendered bytes for {}",
                receipt.stable_id
            )));
        };
        if receipt.stable_id != *expected_id
            || receipt.stable_id != structured.stable_id
            || receipt.stable_id.is_empty()
            || !stable_ids.insert(receipt.stable_id.clone())
            || receipt.source_record_sha256
                != dataset.structured.source_record_sha256[&receipt.stable_id]
            || receipt.raw_example_sha256
                != dataset.structured.raw_example_sha256[&receipt.stable_id]
            || receipt.rendered_utf8_sha256 != sha256_bytes(rendered.as_bytes())
            || receipt.truncated
            || receipt.token_count == 0
            || receipt.token_count != ids.len()
            || receipt.token_count > manifest.max_tokens_per_example
            || receipt.add_generation_prompt
                != (structured.render_mode == RenderMode::GenerationPrompt)
            || receipt.requested_enable_thinking != structured.enable_thinking
            || !is_sha256(&receipt.source_record_sha256)
            || !is_sha256(&receipt.raw_example_sha256)
            || receipt
                .token_window_sha256
                .iter()
                .any(|hash| !is_sha256(hash))
            || receipt.scoring_ranges.iter().any(|range| {
                range.start >= range.end
                    || range.end > receipt.token_count
                    || range.end > manifest.max_tokens_per_example
            })
            || receipt
                .scoring_ranges
                .windows(2)
                .any(|ranges| ranges[0].end > ranges[1].start)
        {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "invalid rendered receipt for {}",
                receipt.stable_id
            )));
        }
        let framed = framed_token_bytes(&receipt.stable_id, ids)?;
        if receipt.token_ids_sha256 != sha256_bytes(&framed)
            || receipt.token_window_sha256
                != token_window_hashes(&receipt.stable_id, ids, manifest.token_window_size)?
        {
            return Err(CalibrationInputError::InvalidDataset(format!(
                "token evidence mismatch for {}",
                receipt.stable_id
            )));
        }
        frame_bytes(
            &mut rendered_stream,
            b"rendered-example-v1",
            &receipt.stable_id,
            rendered.as_bytes(),
        )?;
        frame_bytes(
            &mut token_stream,
            b"token-example-v1",
            &receipt.stable_id,
            &framed,
        )?;
    }
    if hex::encode(rendered_stream.finalize()) != manifest.rendered_text_stream_sha256
        || hex::encode(token_stream.finalize()) != manifest.token_id_stream_sha256
        || rendered_manifest_sha256(manifest)? != manifest.manifest_sha256
    {
        return Err(CalibrationInputError::InvalidDataset(
            "rendered stream or manifest hash mismatch".into(),
        ));
    }
    Ok(())
}

fn verified_source_manifest_sha256(
    manifest: &crate::input::integrity::VerifiedSourceManifest,
) -> Result<String, CalibrationInputError> {
    ordered_evidence_json_sha256(manifest)
}

fn verify_source_file_bytes(
    request: &RenderDatasetRequest,
    filename: &str,
    bytes: &[u8],
) -> Result<(), CalibrationInputError> {
    let record = request
        .verified_source
        .records()
        .iter()
        .find(|record| record.filename == filename)
        .ok_or_else(|| {
            CalibrationInputError::InvalidDataset(format!(
                "verified source manifest is missing render input {filename}"
            ))
        })?;
    let actual_len = u64::try_from(bytes.len()).map_err(|_| {
        CalibrationInputError::InvalidDataset(format!(
            "render input {filename} length is not representable"
        ))
    })?;
    let identity_matches = if let Some(expected) = &record.sha256 {
        sha256_bytes(bytes).eq_ignore_ascii_case(expected)
    } else {
        let expected = record.hf_etag.trim().trim_matches('"');
        let mut git_blob = sha1::Sha1::new();
        git_blob.update(format!("blob {}\0", bytes.len()).as_bytes());
        git_blob.update(bytes);
        expected.len() == 40
            && expected.bytes().all(|byte| byte.is_ascii_hexdigit())
            && hex::encode(git_blob.finalize()).eq_ignore_ascii_case(expected)
    };
    if actual_len != record.bytes || !identity_matches {
        return Err(CalibrationInputError::InvalidDataset(format!(
            "render input {filename} does not match its verified source record"
        )));
    }
    Ok(())
}

fn render_unverified(
    dataset: &StructuredDatasetManifest,
    request: &RenderDatasetRequest,
    global_limits: Option<TeacherPredictionPlanLimits>,
) -> Result<RenderedDataset, CalibrationInputError> {
    super::manifest::validate_structured_dataset_manifest(dataset)?;
    if request.verified_source.repo() != request.source.model_id
        || request.verified_source.revision() != request.source.revision
    {
        return Err(CalibrationInputError::InvalidDataset(
            "verified render source is bound to a different repository or revision".into(),
        ));
    }
    let source_shards = request
        .verified_source
        .records()
        .iter()
        .map(crate::core::provenance::SourceShard::from_integrity)
        .collect::<Vec<_>>();
    if crate::core::provenance::compute_source_bundle_sha256(&source_shards).as_deref()
        != Some(request.source.tensor_bundle_sha256.as_str())
    {
        return Err(CalibrationInputError::InvalidDataset(
            "verified render source has a different tensor bundle".into(),
        ));
    }
    if request.max_tokens_per_example == 0 || request.token_window_size == 0 {
        return Err(CalibrationInputError::InvalidDataset(
            "token and overlap-window bounds must be positive".into(),
        ));
    }
    if let Some(limits) = global_limits {
        super::prediction_plan::validate_prediction_plan_limits(limits)?;
        if dataset.examples.len() > limits.max_examples {
            return Err(CalibrationInputError::InvalidDataset(
                "rendered Calibration split exceeds its global preflight bounds".into(),
            ));
        }
    }

    let tokenizer_path = request.model_dir.join("tokenizer.json");
    let tokenizer_bytes =
        std::fs::read(&tokenizer_path).map_err(|source| CalibrationInputError::Read {
            path: tokenizer_path.clone(),
            source,
        })?;
    let inputs = crate::core::chat_template_resolver::resolve_chat_render_inputs(
        &request.model_dir,
        &tokenizer_bytes,
        &request.arch,
    )?;
    let config_path = request.model_dir.join("config.json");
    let config_bytes =
        std::fs::read(&config_path).map_err(|source| CalibrationInputError::Read {
            path: config_path.clone(),
            source,
        })?;
    verify_source_file_bytes(request, "config.json", &config_bytes)?;
    if sha256_bytes(&config_bytes) != request.source.config_sha256 {
        return Err(CalibrationInputError::InvalidDataset(
            "config.json does not match the source identity".into(),
        ));
    }
    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).map_err(|error| CalibrationInputError::Parse {
            path: config_path,
            detail: error.to_string(),
        })?;
    let resolved_arch = crate::core::model_arch::detect_model_arch(&config)
        .map_err(|error| CalibrationInputError::InvalidDataset(error.observed))?;
    if resolved_arch.name() != request.arch {
        return Err(CalibrationInputError::InvalidDataset(format!(
            "requested calibration architecture {} disagrees with config architecture {}",
            request.arch,
            resolved_arch.name()
        )));
    }
    let expects_tokenizer_config = request
        .verified_source
        .records()
        .iter()
        .any(|record| record.filename == "tokenizer_config.json");
    let expects_sidecar = request
        .verified_source
        .records()
        .iter()
        .any(|record| record.filename == "chat_template.jinja");
    if inputs.tokenizer_config_bytes.is_some() != expects_tokenizer_config
        || inputs.chat_template_sidecar_bytes.is_some() != expects_sidecar
    {
        return Err(CalibrationInputError::InvalidDataset(
            "render-input presence does not match the verified source manifest".into(),
        ));
    }
    verify_source_file_bytes(request, "tokenizer.json", &tokenizer_bytes)?;
    if let Some(bytes) = &inputs.tokenizer_config_bytes {
        verify_source_file_bytes(request, "tokenizer_config.json", bytes)?;
    }
    if let Some(bytes) = &inputs.chat_template_sidecar_bytes {
        verify_source_file_bytes(request, "chat_template.jinja", bytes)?;
    }
    let template = inputs
        .template
        .ok_or_else(|| CalibrationInputError::MissingTemplate(request.arch.clone()))?;
    if template.sha256 != request.source.chat_template_sha256 {
        return Err(CalibrationInputError::SourceTemplateMismatch);
    }
    if inputs.tokenizer_bundle_sha256 != request.source.tokenizer_bundle_sha256 {
        return Err(CalibrationInputError::SourceTokenizerBundleMismatch);
    }
    let tokenizer = tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(|error| {
        CalibrationInputError::Parse {
            path: tokenizer_path,
            detail: error.to_string(),
        }
    })?;

    let mut rendered_stream = Sha256::new();
    let mut token_stream = Sha256::new();
    let mut receipts = Vec::with_capacity(dataset.examples.len());
    let mut rendered_utf8 = BTreeMap::new();
    let mut token_ids = BTreeMap::new();
    let mut total_token_count = 0usize;
    let mut total_rendered_utf8_bytes = 0u64;
    for example in &dataset.examples {
        if example.messages.iter().any(|message| {
            message
                .content
                .as_ref()
                .is_some_and(|content| content.has_images())
        }) {
            return Err(CalibrationInputError::MediaUnsupported(
                example.stable_id.clone(),
            ));
        }
        let tools = (!example.tools.is_empty()).then_some(example.tools.as_slice());
        let kwargs = kwargs_map(&example.chat_template_kwargs);
        let add_generation_prompt = example.render_mode == RenderMode::GenerationPrompt;
        let rendered = crate::serve::api::engine::render_chat_prompt_with_tools_generation_prompt(
            &template.template,
            &example.messages,
            tools,
            example.enable_thinking,
            Some(&kwargs),
            add_generation_prompt,
        )
        .map_err(|error| CalibrationInputError::Render {
            stable_id: example.stable_id.clone(),
            detail: error.to_string(),
        })?;
        let ids = encode(&tokenizer, &example.stable_id, &rendered)?;
        if ids.len() > request.max_tokens_per_example {
            return Err(CalibrationInputError::TokenLimit {
                stable_id: example.stable_id.clone(),
                tokens: ids.len(),
                maximum: request.max_tokens_per_example,
            });
        }
        if let Some(limits) = global_limits {
            total_token_count = total_token_count.checked_add(ids.len()).ok_or_else(|| {
                CalibrationInputError::InvalidDataset(
                    "rendered Calibration token count overflow".into(),
                )
            })?;
            total_rendered_utf8_bytes = total_rendered_utf8_bytes
                .checked_add(u64::try_from(rendered.len()).map_err(|_| {
                    CalibrationInputError::InvalidDataset(
                        "rendered Calibration byte count is not representable".into(),
                    )
                })?)
                .ok_or_else(|| {
                    CalibrationInputError::InvalidDataset(
                        "rendered Calibration byte count overflow".into(),
                    )
                })?;
            if total_token_count > limits.max_total_tokens
                || total_rendered_utf8_bytes > limits.max_rendered_utf8_bytes
            {
                return Err(CalibrationInputError::InvalidDataset(
                    "rendered Calibration split exceeds its global token or byte bound".into(),
                ));
            }
        }

        let scoring_ranges = if example.render_mode == RenderMode::CompletedAssistantTranscript {
            if example.messages.last().map(|message| message.role.as_str()) != Some("assistant") {
                return Err(CalibrationInputError::MissingAssistantTarget(
                    example.stable_id.clone(),
                ));
            }
            let prefix =
                crate::serve::api::engine::render_chat_prompt_with_tools_generation_prompt(
                    &template.template,
                    &example.messages[..example.messages.len() - 1],
                    tools,
                    example.enable_thinking,
                    Some(&kwargs),
                    true,
                )
                .map_err(|error| CalibrationInputError::Render {
                    stable_id: example.stable_id.clone(),
                    detail: error.to_string(),
                })?;
            let prefix_ids = encode(&tokenizer, &example.stable_id, &prefix)?;
            if !ids.starts_with(&prefix_ids) || prefix_ids.len() >= ids.len() {
                return Err(CalibrationInputError::NonPrefixAssistantTarget(
                    example.stable_id.clone(),
                ));
            }
            vec![TokenRange {
                start: prefix_ids.len(),
                end: ids.len(),
            }]
        } else {
            Vec::new()
        };

        let token_bytes = framed_token_bytes(&example.stable_id, &ids)?;
        frame_bytes(
            &mut rendered_stream,
            b"rendered-example-v1",
            &example.stable_id,
            rendered.as_bytes(),
        )?;
        frame_bytes(
            &mut token_stream,
            b"token-example-v1",
            &example.stable_id,
            &token_bytes,
        )?;
        receipts.push(RenderedExampleReceipt {
            stable_id: example.stable_id.clone(),
            source_record_sha256: dataset.source_record_sha256[&example.stable_id].clone(),
            raw_example_sha256: dataset.raw_example_sha256[&example.stable_id].clone(),
            rendered_utf8_sha256: sha256_bytes(rendered.as_bytes()),
            token_ids_sha256: sha256_bytes(&token_bytes),
            token_count: ids.len(),
            scoring_ranges,
            token_window_sha256: token_window_hashes(
                &example.stable_id,
                &ids,
                request.token_window_size,
            )?,
            add_generation_prompt,
            requested_enable_thinking: example.enable_thinking,
            truncated: false,
        });
        rendered_utf8.insert(example.stable_id.clone(), rendered);
        token_ids.insert(example.stable_id.clone(), ids);
    }

    let mut manifest = RenderedDatasetManifest {
        schema_version: CALIBRATION_INPUT_SCHEMA_VERSION,
        split: dataset.split,
        source: request.source.clone(),
        verified_source_manifest_sha256: verified_source_manifest_sha256(&request.verified_source)?,
        structured_dataset_sha256: dataset.manifest_sha256.clone(),
        chat_template_source: template.source,
        chat_template_sha256: template.sha256,
        tokenizer_json_sha256: inputs.tokenizer_json_sha256,
        renderer_revision: request.renderer_revision.clone(),
        max_tokens_per_example: request.max_tokens_per_example,
        token_window_size: request.token_window_size,
        examples: receipts,
        rendered_text_stream_sha256: hex::encode(rendered_stream.finalize()),
        token_id_stream_sha256: hex::encode(token_stream.finalize()),
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = rendered_manifest_sha256(&manifest)?;
    Ok(RenderedDataset {
        structured: dataset.clone(),
        manifest,
        rendered_utf8,
        token_ids,
    })
}

/// Render and tokenize one immutable split through the production chat path.
pub fn render_and_tokenize_split(
    dataset: &StructuredDatasetManifest,
    request: &RenderDatasetRequest,
) -> Result<RenderedDataset, CalibrationInputError> {
    let rendered = render_unverified(dataset, request, None)?;
    validate_rendered_dataset(&rendered)?;
    Ok(rendered)
}

/// Render an owned, hash-authenticated structured corpus through the exact
/// source tokenizer/template path. D3 target production uses this entrypoint;
/// the manifest-only form remains for compatibility and non-authority tools.
pub(crate) fn render_and_tokenize_verified_split(
    dataset: &VerifiedCalibrationCorpus,
    request: &RenderDatasetRequest,
    limits: TeacherPredictionPlanLimits,
) -> Result<RenderedDataset, CalibrationInputError> {
    let rendered = render_unverified(dataset.manifest(), request, Some(limits))?;
    validate_rendered_dataset(&rendered)?;
    Ok(rendered)
}

/// Independently rerender a claimed split from its structured inputs and the
/// exact source tokenizer/template snapshot, then compare every retained byte.
pub fn verify_rendered_dataset_from_source(
    claimed: &RenderedDataset,
    request: &RenderDatasetRequest,
) -> Result<(), CalibrationInputError> {
    validate_rendered_dataset(claimed)?;
    let rebuilt = render_unverified(&claimed.structured, request, None)?;
    validate_rendered_dataset(&rebuilt)?;
    let claimed_manifest = serde_json::to_vec(&claimed.manifest)
        .map_err(|error| CalibrationInputError::Serialization(error.to_string()))?;
    let rebuilt_manifest = serde_json::to_vec(&rebuilt.manifest)
        .map_err(|error| CalibrationInputError::Serialization(error.to_string()))?;
    if claimed_manifest != rebuilt_manifest
        || claimed.rendered_utf8 != rebuilt.rendered_utf8
        || claimed.token_ids != rebuilt.token_ids
    {
        return Err(CalibrationInputError::InvalidDataset(
            "rendered dataset does not reproduce from its exact source inputs".into(),
        ));
    }
    Ok(())
}
