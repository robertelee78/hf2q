//! Bounded, owned structured-corpus artifact verification.

use std::io::Read;

use sha2::{Digest, Sha256};

use super::types::*;

const MAX_CORPUS_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CORPUS_EXAMPLES: usize = 16_384;
const MAX_CORPUS_MESSAGES: usize = 131_072;
const MAX_CORPUS_TOOLS: usize = 65_536;

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_limits(limits: CalibrationCorpusArtifactLimits) -> Result<(), CalibrationInputError> {
    if limits.max_artifact_bytes == 0
        || limits.max_examples == 0
        || limits.max_messages == 0
        || limits.max_tools == 0
        || limits.max_artifact_bytes > MAX_CORPUS_ARTIFACT_BYTES
        || limits.max_examples > MAX_CORPUS_EXAMPLES
        || limits.max_messages > MAX_CORPUS_MESSAGES
        || limits.max_tools > MAX_CORPUS_TOOLS
    {
        return Err(CalibrationInputError::InvalidDataset(
            "calibration corpus artifact bounds must be positive".into(),
        ));
    }
    Ok(())
}

pub(crate) fn verify_calibration_corpus_artifact(
    request: &VerifyCalibrationCorpusRequest,
) -> Result<VerifiedCalibrationCorpus, CalibrationInputError> {
    validate_limits(request.limits)?;
    if !is_lower_sha256(&request.expected_sha256)
        || request.expected_dataset_id.is_empty()
        || request.expected_revision.is_empty()
        || request.expected_declared_license.is_empty()
    {
        return Err(CalibrationInputError::InvalidDataset(
            "calibration corpus expected identity is invalid".into(),
        ));
    }
    let file =
        std::fs::File::open(&request.path).map_err(|source| CalibrationInputError::Read {
            path: request.path.clone(),
            source,
        })?;
    let observed_len = file
        .metadata()
        .map_err(|source| CalibrationInputError::Read {
            path: request.path.clone(),
            source,
        })?
        .len();
    if observed_len == 0 || observed_len > request.limits.max_artifact_bytes {
        return Err(CalibrationInputError::InvalidDataset(
            "calibration corpus artifact is empty or exceeds its byte bound".into(),
        ));
    }
    let capacity = usize::try_from(observed_len).map_err(|_| {
        CalibrationInputError::InvalidDataset(
            "calibration corpus artifact length is not addressable".into(),
        )
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(request.limits.max_artifact_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| CalibrationInputError::Read {
            path: request.path.clone(),
            source,
        })?;
    if bytes.is_empty()
        || u64::try_from(bytes.len()).unwrap_or(u64::MAX) > request.limits.max_artifact_bytes
    {
        return Err(CalibrationInputError::InvalidDataset(
            "calibration corpus artifact changed size or exceeded its byte bound".into(),
        ));
    }
    let sha256 = hex::encode(Sha256::digest(&bytes));
    if sha256 != request.expected_sha256 {
        return Err(CalibrationInputError::InvalidDataset(
            "calibration corpus artifact SHA-256 mismatch".into(),
        ));
    }
    let manifest: StructuredDatasetManifest =
        serde_json::from_slice(&bytes).map_err(|error| CalibrationInputError::Parse {
            path: request.path.clone(),
            detail: error.to_string(),
        })?;
    super::manifest::validate_structured_dataset_manifest(&manifest)?;
    let message_count = manifest
        .examples
        .iter()
        .try_fold(0usize, |total, example| {
            total.checked_add(example.messages.len()).ok_or_else(|| {
                CalibrationInputError::InvalidDataset("calibration message count overflow".into())
            })
        })?;
    let tool_count = manifest
        .examples
        .iter()
        .try_fold(0usize, |total, example| {
            total.checked_add(example.tools.len()).ok_or_else(|| {
                CalibrationInputError::InvalidDataset("calibration tool count overflow".into())
            })
        })?;
    if manifest.dataset_id != request.expected_dataset_id
        || manifest.revision != request.expected_revision
        || manifest.license != request.expected_declared_license
        || manifest.split != request.expected_split
        || manifest.examples.len() > request.limits.max_examples
        || message_count > request.limits.max_messages
        || tool_count > request.limits.max_tools
    {
        return Err(CalibrationInputError::InvalidDataset(
            "calibration corpus identity or collection bounds differ from the request".into(),
        ));
    }
    Ok(VerifiedCalibrationCorpus {
        artifact: crate::core::provenance::tensor_execution::ArtifactEvidence {
            artifact_id: format!("{}@{}", manifest.dataset_id, manifest.revision),
            role: "owned_structured_calibration_corpus_json_v1".into(),
            byte_len: u64::try_from(bytes.len()).unwrap(),
            sha256,
        },
        manifest,
    })
}
