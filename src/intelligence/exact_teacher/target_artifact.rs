use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::core::provenance::tensor_execution::ArtifactEvidence;
#[cfg(test)]
use crate::intelligence::calibration::VerifiedCalibrationPredictionPlan;

use super::types::*;

mod publication;
mod reference;
mod reservation;
mod stream;
mod verify;

pub(crate) use reference::{
    open_receipted_structural_teacher_target, reconstruct_structural_teacher_reference_target,
};

pub(crate) use reservation::UnpublishedStructuralTeacherTargetReservation;
#[cfg(test)]
pub(crate) use stream::StructuralTeacherTargetPreflight;
pub(crate) use stream::{
    preflight_structural_teacher_target, StructuralTeacherTargetStream,
    UnpublishedStructuralTeacherTargetArtifact,
};

#[cfg(test)]
pub(super) use verify::verify_for_test;

const TARGET_MAGIC: &[u8] = b"hf2q-exact-teacher-targets-v1\0";
const ROW_MAGIC: &[u8; 4] = b"ROW1";
const ROW_FRAME_BYTES: u64 = 4 + 8 + 8 + 32 + 8;
const TARGET_SEMANTICS: &str = "structural_full_vocab_f32_le_rows_no_execution_authority_v1";
const MAX_TARGET_VOCABULARY_SIZE: usize = 1_000_000;
const MAX_TARGET_PREDICTION_ROWS: usize = 16_384;
const MAX_TARGET_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const MAX_TARGET_TOP_K: usize = 4_096;
const MAX_TARGET_SUMMARY_ENTRIES: usize = 1_000_000;

#[derive(Serialize)]
struct ReceiptHashView<'a> {
    schema_version: u32,
    semantics: &'a str,
    prediction_plan_sha256: &'a str,
    limits: TeacherTargetArtifactLimits,
    vocabulary_size: usize,
    prediction_point_count: usize,
    generation_prompt_count: usize,
    target_artifact: &'a ArtifactEvidence,
    rows: &'a [TeacherTargetRowReceipt],
    greedy_trajectories: &'a [TeacherGreedyTrajectoryReceipt],
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn receipt_sha256(receipt: &ExactTeacherTargetReceipt) -> Result<String, ExactTeacherTargetError> {
    let bytes = serde_json::to_vec(&ReceiptHashView {
        schema_version: receipt.schema_version,
        semantics: &receipt.semantics,
        prediction_plan_sha256: &receipt.prediction_plan_sha256,
        limits: receipt.limits,
        vocabulary_size: receipt.vocabulary_size,
        prediction_point_count: receipt.prediction_point_count,
        generation_prompt_count: receipt.generation_prompt_count,
        target_artifact: &receipt.target_artifact,
        rows: &receipt.rows,
        greedy_trajectories: &receipt.greedy_trajectories,
    })
    .map_err(|error| ExactTeacherTargetError::Serialization(error.to_string()))?;
    Ok(hash_bytes(&bytes))
}

fn checked_row_bytes(vocabulary_size: usize) -> Result<u64, ExactTeacherTargetError> {
    u64::try_from(vocabulary_size)
        .ok()
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| ExactTeacherTargetError::Invalid("logit row byte length overflow".into()))
}

fn checked_target_bytes(
    rows: usize,
    vocabulary_size: usize,
) -> Result<u64, ExactTeacherTargetError> {
    let row_bytes = checked_row_bytes(vocabulary_size)?;
    let framed_row = ROW_FRAME_BYTES
        .checked_add(row_bytes)
        .ok_or_else(|| ExactTeacherTargetError::Invalid("framed row size overflow".into()))?;
    u64::try_from(TARGET_MAGIC.len())
        .ok()
        .and_then(|header| {
            u64::try_from(rows)
                .ok()
                .and_then(|rows| rows.checked_mul(framed_row))
                .and_then(|payload| header.checked_add(payload))
        })
        .ok_or_else(|| ExactTeacherTargetError::Invalid("target artifact size overflow".into()))
}

fn row_bytes(logits: &[f32]) -> Result<Vec<u8>, ExactTeacherTargetError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher logits must be non-empty and finite".into(),
        ));
    }
    let capacity = logits
        .len()
        .checked_mul(4)
        .ok_or_else(|| ExactTeacherTargetError::Invalid("logit encoding overflow".into()))?;
    let mut bytes = Vec::with_capacity(capacity);
    for value in logits {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(bytes)
}

fn row_summary(
    logits: &[f32],
    top_k: usize,
) -> Result<(u32, Vec<TeacherTopKLogit>, u64), ExactTeacherTargetError> {
    if top_k == 0 || top_k > logits.len() || logits.len() > u32::MAX as usize {
        return Err(ExactTeacherTargetError::Invalid(
            "top-k or vocabulary size is outside the target schema".into(),
        ));
    }
    let mut ordered = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ordered.sort_by(|(left_id, left), (right_id, right)| {
        right
            .partial_cmp(left)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left_id.cmp(right_id))
    });
    let argmax_token_id = u32::try_from(ordered[0].0)
        .map_err(|_| ExactTeacherTargetError::Invalid("argmax token id overflow".into()))?;
    let top = ordered
        .iter()
        .take(top_k)
        .map(|(token_id, value)| {
            Ok(TeacherTopKLogit {
                token_id: u32::try_from(*token_id).map_err(|_| {
                    ExactTeacherTargetError::Invalid("top-k token id overflow".into())
                })?,
                logit_f32_bits: value.to_bits(),
            })
        })
        .collect::<Result<Vec<_>, ExactTeacherTargetError>>()?;
    let maximum = f64::from(ordered[0].1);
    let sum = logits
        .iter()
        .map(|value| (f64::from(*value) - maximum).exp())
        .sum::<f64>();
    let logsumexp = maximum + sum.ln();
    if !logsumexp.is_finite() {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher log-sum-exp is non-finite".into(),
        ));
    }
    Ok((argmax_token_id, top, logsumexp.to_bits()))
}

/// Canonical finite-logit argmax shared by structural row summaries and the
/// family-owned greedy trajectory. Equal numeric values, including signed
/// zero, resolve to the lowest token id.
pub(crate) fn canonical_teacher_argmax(logits: &[f32]) -> Result<u32, ExactTeacherTargetError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher logits must be non-empty and finite".into(),
        ));
    }
    let mut best_id = 0usize;
    let mut best = logits[0];
    for (token_id, value) in logits.iter().copied().enumerate().skip(1) {
        if value > best {
            best = value;
            best_id = token_id;
        }
    }
    u32::try_from(best_id)
        .map_err(|_| ExactTeacherTargetError::Invalid("argmax token id overflow".into()))
}

fn trajectory_sha256(tokens: &[u32]) -> Result<String, ExactTeacherTargetError> {
    let count = u64::try_from(tokens.len())
        .map_err(|_| ExactTeacherTargetError::Invalid("trajectory length overflow".into()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"hf2q-exact-teacher-greedy-tokens-v1");
    hasher.update(count.to_le_bytes());
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    Ok(hex::encode(hasher.finalize()))
}

fn hash_open_file_bounded(
    file: &mut File,
    expected_len: u64,
) -> Result<String, ExactTeacherTargetError> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?;
    let mut hasher = Sha256::new();
    let mut remaining = expected_len;
    let mut buffer = [0u8; 64 * 1024];
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        let read = file
            .read(&mut buffer[..wanted])
            .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?;
        if read == 0 {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target file ended before its bounded length".into(),
            ));
        }
        remaining -= u64::try_from(read).unwrap();
        hasher.update(&buffer[..read]);
    }
    let mut trailing = [0u8; 1];
    if file
        .read(&mut trailing)
        .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?
        != 0
    {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher target file exceeds its bounded length".into(),
        ));
    }
    Ok(hex::encode(hasher.finalize()))
}

fn digest_to_array(value: &str) -> Result<[u8; 32], ExactTeacherTargetError> {
    let bytes = hex::decode(value)
        .map_err(|_| ExactTeacherTargetError::Invalid("invalid SHA-256 hex".into()))?;
    bytes
        .try_into()
        .map_err(|_| ExactTeacherTargetError::Invalid("invalid SHA-256 length".into()))
}

/// Write and independently reread a bounded target artifact supplied by an
/// arbitrary logit source. The returned type is structural only and therefore
/// cannot authorize sensitivity or allocation.
#[cfg(test)]
pub(crate) fn write_structural_teacher_target_artifact<LF, GF>(
    plan: &VerifiedCalibrationPredictionPlan,
    output: &Path,
    vocabulary_size: usize,
    limits: TeacherTargetArtifactLimits,
    mut logits_for: LF,
    mut greedy_for: GF,
) -> Result<StructurallyVerifiedTeacherTargetArtifact, ExactTeacherTargetError>
where
    LF: FnMut(TeacherTargetLogitRequest<'_>) -> Result<Vec<f32>, ExactTeacherTargetError>,
    GF: FnMut(TeacherGreedyRequest<'_>) -> Result<Vec<u32>, ExactTeacherTargetError>,
{
    let preflight: StructuralTeacherTargetPreflight<'_> =
        preflight_structural_teacher_target(plan, vocabulary_size, limits)?;
    let mut stream: StructuralTeacherTargetStream<'_> = preflight.begin(output)?;
    plan.visit_prediction_points(|point, prefix_token_ids| {
        let logits = logits_for(TeacherTargetLogitRequest {
            point,
            prefix_token_ids,
        })?;
        stream.write_row(point, &logits).map(|_| ())
    })?;
    plan.visit_greedy_prompts(|prompt, prompt_token_ids| {
        let token_ids = greedy_for(TeacherGreedyRequest {
            prompt,
            prompt_token_ids,
        })?;
        stream.write_trajectory(prompt, &token_ids)
    })?;
    stream.finish()
}
