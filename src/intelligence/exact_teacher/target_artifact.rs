use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::core::provenance::tensor_execution::ArtifactEvidence;
use crate::intelligence::calibration::VerifiedCalibrationPredictionPlan;

use super::types::*;

mod verify;

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
    if vocabulary_size == 0
        || vocabulary_size > limits.max_vocabulary_size
        || plan.prediction_point_count() == 0
        || plan.prediction_point_count() > limits.max_prediction_rows
        || limits.top_k == 0
        || limits.top_k > vocabulary_size
        || limits.max_vocabulary_size > MAX_TARGET_VOCABULARY_SIZE
        || limits.max_prediction_rows > MAX_TARGET_PREDICTION_ROWS
        || limits.max_target_bytes > MAX_TARGET_ARTIFACT_BYTES
        || limits.top_k > MAX_TARGET_TOP_K
        || plan
            .prediction_point_count()
            .checked_mul(limits.top_k)
            .is_none_or(|entries| entries > MAX_TARGET_SUMMARY_ENTRIES)
    {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher target dimensions exceed their declared bounds".into(),
        ));
    }
    let preflight_bytes = checked_target_bytes(plan.prediction_point_count(), vocabulary_size)?;
    if preflight_bytes > limits.max_target_bytes {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher target artifact exceeds its preflight byte bound".into(),
        ));
    }
    plan.visit_prediction_points(|point, prefix_token_ids| {
        if prefix_token_ids
            .iter()
            .any(|token_id| usize::try_from(*token_id).unwrap_or(usize::MAX) >= vocabulary_size)
            || matches!(
                point.kind,
                crate::intelligence::calibration::TeacherPredictionPointKind::TeacherForced {
                    target_token_id,
                    ..
                } if usize::try_from(target_token_id).unwrap_or(usize::MAX) >= vocabulary_size
            )
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher prediction point contains a token outside the declared vocabulary".into(),
            ));
        }
        Ok(())
    })?;
    plan.visit_greedy_prompts(|_prompt, prompt_token_ids| {
        if prompt_token_ids
            .iter()
            .any(|token_id| usize::try_from(*token_id).unwrap_or(usize::MAX) >= vocabulary_size)
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher greedy prompt contains a token outside the declared vocabulary".into(),
            ));
        }
        Ok(())
    })?;
    let parent = output.parent().ok_or_else(|| {
        ExactTeacherTargetError::Invalid("teacher target path has no parent".into())
    })?;
    std::fs::create_dir_all(parent).map_err(|error| ExactTeacherTargetError::io(parent, error))?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| ExactTeacherTargetError::io(parent, error))?;
    let mut writer = BufWriter::new(temporary.as_file());
    writer
        .write_all(TARGET_MAGIC)
        .map_err(|error| ExactTeacherTargetError::io(output, error))?;
    let mut offset = u64::try_from(TARGET_MAGIC.len()).unwrap();
    let mut rows = Vec::with_capacity(plan.prediction_point_count());
    plan.visit_prediction_points(|point, prefix_token_ids| {
        if matches!(
            point.kind,
            crate::intelligence::calibration::TeacherPredictionPointKind::TeacherForced {
                target_token_id,
                ..
            } if usize::try_from(target_token_id).unwrap_or(usize::MAX) >= vocabulary_size
        ) {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher-forced target token is outside the declared vocabulary".into(),
            ));
        }
        let logits = logits_for(TeacherTargetLogitRequest {
            point,
            prefix_token_ids,
        })?;
        if logits.len() != vocabulary_size {
            return Err(ExactTeacherTargetError::Invalid(format!(
                "teacher row {} has vocabulary {}, expected {}",
                point.point_ordinal,
                logits.len(),
                vocabulary_size
            )));
        }
        let payload = row_bytes(&logits)?;
        let (argmax_token_id, top_k, logsumexp_f64_bits) = row_summary(&logits, limits.top_k)?;
        let payload_offset = offset
            .checked_add(ROW_FRAME_BYTES)
            .ok_or_else(|| ExactTeacherTargetError::Invalid("row offset overflow".into()))?;
        let prefix_digest = digest_to_array(&point.prefix_token_ids_sha256)?;
        let point_ordinal = u64::try_from(point.point_ordinal).map_err(|_| {
            ExactTeacherTargetError::Invalid("prediction point ordinal overflow".into())
        })?;
        let vocabulary_size_u64 = u64::try_from(vocabulary_size).map_err(|_| {
            ExactTeacherTargetError::Invalid("target vocabulary size overflow".into())
        })?;
        let payload_size = u64::try_from(payload.len()).map_err(|_| {
            ExactTeacherTargetError::Invalid("target row byte length overflow".into())
        })?;
        writer
            .write_all(ROW_MAGIC)
            .and_then(|_| writer.write_all(&point_ordinal.to_le_bytes()))
            .and_then(|_| writer.write_all(&vocabulary_size_u64.to_le_bytes()))
            .and_then(|_| writer.write_all(&prefix_digest))
            .and_then(|_| writer.write_all(&payload_size.to_le_bytes()))
            .and_then(|_| writer.write_all(&payload))
            .map_err(|error| ExactTeacherTargetError::io(output, error))?;
        let payload_bytes = payload_size;
        offset = payload_offset
            .checked_add(payload_bytes)
            .ok_or_else(|| ExactTeacherTargetError::Invalid("row end overflow".into()))?;
        rows.push(TeacherTargetRowReceipt {
            point_ordinal: point.point_ordinal,
            stable_id: point.stable_id.clone(),
            point_kind: point.kind,
            prefix_token_count: point.prefix_token_count,
            prefix_token_ids_sha256: point.prefix_token_ids_sha256.clone(),
            vocabulary_size,
            payload_offset,
            payload_bytes,
            payload_sha256: hash_bytes(&payload),
            argmax_token_id,
            top_k,
            logsumexp_f64_bits,
        });
        Ok(())
    })?;
    writer
        .flush()
        .map_err(|error| ExactTeacherTargetError::io(output, error))?;
    drop(writer);
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| ExactTeacherTargetError::io(output, error))?;
    if offset != preflight_bytes {
        return Err(ExactTeacherTargetError::Invalid(
            "written teacher target size differs from preflight".into(),
        ));
    }

    let mut trajectories = Vec::with_capacity(plan.manifest().greedy_prompts.len());
    plan.visit_greedy_prompts(|prompt, prompt_token_ids| {
        let token_ids = greedy_for(TeacherGreedyRequest {
            prompt,
            prompt_token_ids,
        })?;
        if token_ids.len() != EXACT_TEACHER_GREEDY_TOKEN_COUNT {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher greedy trajectory must contain exactly 32 tokens".into(),
            ));
        }
        if token_ids
            .iter()
            .any(|token_id| usize::try_from(*token_id).unwrap_or(usize::MAX) >= vocabulary_size)
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher greedy trajectory contains a token outside the declared vocabulary".into(),
            ));
        }
        let token_ids_sha256 = trajectory_sha256(&token_ids)?;
        trajectories.push(TeacherGreedyTrajectoryReceipt {
            stable_id: prompt.stable_id.clone(),
            prompt_token_ids_sha256: prompt.prefix_token_ids_sha256.clone(),
            token_ids,
            token_ids_sha256,
        });
        Ok(())
    })?;

    let artifact_sha256 = hash_open_file_bounded(temporary.as_file_mut(), preflight_bytes)?;
    let artifact_bytes = preflight_bytes;
    let mut receipt = ExactTeacherTargetReceipt {
        schema_version: EXACT_TEACHER_TARGET_SCHEMA_VERSION,
        semantics: TARGET_SEMANTICS.into(),
        prediction_plan_sha256: plan.manifest().manifest_sha256.clone(),
        limits,
        vocabulary_size,
        prediction_point_count: plan.prediction_point_count(),
        generation_prompt_count: plan.manifest().greedy_prompts.len(),
        target_artifact: ArtifactEvidence {
            artifact_id: "exact_teacher_logits".into(),
            role: "structural_full_vocabulary_f32_target_rows".into(),
            byte_len: artifact_bytes,
            sha256: artifact_sha256,
        },
        rows,
        greedy_trajectories: trajectories,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_sha256(&receipt)?;
    verify::verify_structural_teacher_target_artifact(temporary.as_file_mut(), &receipt)?;
    let file = temporary
        .persist_noclobber(output)
        .map_err(|error| ExactTeacherTargetError::io(output, error.error))?;
    Ok(StructurallyVerifiedTeacherTargetArtifact {
        receipt,
        file,
        path: output.to_owned(),
    })
}
