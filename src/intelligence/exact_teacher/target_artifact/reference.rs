//! Cross-process structural reopening for matched-reference validation.
//!
//! These transitions reconstruct framing and plan closure only. They cannot
//! recreate the family-owned source-teacher completion authority that existed
//! in the producing process.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use rustix::fs::{self, FileType, Mode, OFlags};

use crate::intelligence::calibration::VerifiedTeacherPredictionPlan;

use super::*;

const RETAINED_OPEN_FLAGS: OFlags = OFlags::RDONLY
    .union(OFlags::NOFOLLOW)
    .union(OFlags::NONBLOCK)
    .union(OFlags::CLOEXEC);

pub(crate) fn open_receipted_structural_teacher_target(
    path: &Path,
    plan: &VerifiedTeacherPredictionPlan,
    receipt: ExactTeacherTargetReceipt,
) -> Result<StructurallyVerifiedTeacherTargetArtifact, ExactTeacherTargetError> {
    validate_receipt_against_plan(plan, &receipt)?;
    let mut file = open_regular_target(path, receipt.target_artifact.byte_len)?;
    verify::verify_structural_teacher_target_artifact(&mut file, &receipt)?;
    Ok(StructurallyVerifiedTeacherTargetArtifact {
        receipt,
        _file: file,
        path: path.to_owned(),
    })
}

pub(crate) fn reconstruct_structural_teacher_reference_target(
    path: &Path,
    plan: &VerifiedTeacherPredictionPlan,
    vocabulary_size: usize,
    limits: TeacherTargetArtifactLimits,
    trajectories: Vec<TeacherGreedyTrajectoryReceipt>,
) -> Result<StructurallyVerifiedTeacherTargetArtifact, ExactTeacherTargetError> {
    preflight_structural_teacher_target(plan, vocabulary_size, limits)?;
    let expected_len = checked_target_bytes(plan.prediction_point_count(), vocabulary_size)?;
    let mut file = open_regular_target(path, expected_len)?;
    let artifact_sha256 = hash_open_file_bounded(&mut file, expected_len)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| ExactTeacherTargetError::io(path, error))?;
    let mut reader = BufReader::new(&mut file);
    let mut magic = vec![0_u8; TARGET_MAGIC.len()];
    reader
        .read_exact(&mut magic)
        .map_err(|error| ExactTeacherTargetError::io(path, error))?;
    if magic != TARGET_MAGIC {
        return Err(ExactTeacherTargetError::Invalid(
            "reference target magic is invalid".into(),
        ));
    }

    let payload_bytes = checked_row_bytes(vocabulary_size)?;
    let payload_len = usize::try_from(payload_bytes)
        .map_err(|_| ExactTeacherTargetError::Invalid("reference row is too large".into()))?;
    let mut rows = Vec::with_capacity(plan.prediction_point_count());
    let mut offset = u64::try_from(TARGET_MAGIC.len()).unwrap();
    for point in &plan.manifest().prediction_points {
        let mut row_magic = [0_u8; 4];
        reader
            .read_exact(&mut row_magic)
            .map_err(|error| ExactTeacherTargetError::io(path, error))?;
        let ordinal = read_u64(&mut reader, path)?;
        let observed_vocabulary = read_u64(&mut reader, path)?;
        let mut prefix_sha256 = [0_u8; 32];
        reader
            .read_exact(&mut prefix_sha256)
            .map_err(|error| ExactTeacherTargetError::io(path, error))?;
        let observed_payload_bytes = read_u64(&mut reader, path)?;
        offset = offset.checked_add(ROW_FRAME_BYTES).ok_or_else(|| {
            ExactTeacherTargetError::Invalid("reference row offset overflow".into())
        })?;
        if row_magic != *ROW_MAGIC
            || ordinal != u64::try_from(point.point_ordinal).unwrap_or(u64::MAX)
            || observed_vocabulary != u64::try_from(vocabulary_size).unwrap_or(u64::MAX)
            || prefix_sha256 != digest_to_array(&point.prefix_token_ids_sha256)?
            || observed_payload_bytes != payload_bytes
        {
            return Err(ExactTeacherTargetError::Invalid(
                "reference target row framing differs from the prediction plan".into(),
            ));
        }
        let mut payload = vec![0_u8; payload_len];
        reader
            .read_exact(&mut payload)
            .map_err(|error| ExactTeacherTargetError::io(path, error))?;
        let logits = payload
            .chunks_exact(4)
            .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap())))
            .collect::<Vec<_>>();
        let (argmax_token_id, top_k, logsumexp_f64_bits) = row_summary(&logits, limits.top_k)?;
        rows.push(TeacherTargetRowReceipt {
            point_ordinal: point.point_ordinal,
            stable_id: point.stable_id.clone(),
            point_kind: point.kind,
            prefix_token_count: point.prefix_token_count,
            prefix_token_ids_sha256: point.prefix_token_ids_sha256.clone(),
            vocabulary_size,
            payload_offset: offset,
            payload_bytes,
            payload_sha256: hash_bytes(&payload),
            argmax_token_id,
            top_k,
            logsumexp_f64_bits,
        });
        offset = offset
            .checked_add(payload_bytes)
            .ok_or_else(|| ExactTeacherTargetError::Invalid("reference row end overflow".into()))?;
    }
    let mut trailing = [0_u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| ExactTeacherTargetError::io(path, error))?
        != 0
        || offset != expected_len
    {
        return Err(ExactTeacherTargetError::Invalid(
            "reference target has trailing bytes or an invalid extent".into(),
        ));
    }
    drop(reader);

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
            byte_len: expected_len,
            sha256: artifact_sha256,
        },
        rows,
        greedy_trajectories: trajectories,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = receipt_sha256(&receipt)?;
    validate_receipt_against_plan(plan, &receipt)?;
    verify::verify_structural_teacher_target_artifact(&mut file, &receipt)?;
    Ok(StructurallyVerifiedTeacherTargetArtifact {
        receipt,
        _file: file,
        path: path.to_owned(),
    })
}

fn validate_receipt_against_plan(
    plan: &VerifiedTeacherPredictionPlan,
    receipt: &ExactTeacherTargetReceipt,
) -> Result<(), ExactTeacherTargetError> {
    if receipt.prediction_plan_sha256 != plan.manifest().manifest_sha256
        || receipt.prediction_point_count != plan.prediction_point_count()
        || receipt.generation_prompt_count != plan.manifest().greedy_prompts.len()
        || receipt.rows.len() != plan.prediction_point_count()
        || receipt
            .rows
            .iter()
            .zip(&plan.manifest().prediction_points)
            .any(|(row, point)| {
                row.point_ordinal != point.point_ordinal
                    || row.stable_id != point.stable_id
                    || row.point_kind != point.kind
                    || row.prefix_token_count != point.prefix_token_count
                    || row.prefix_token_ids_sha256 != point.prefix_token_ids_sha256
            })
        || receipt
            .greedy_trajectories
            .iter()
            .zip(&plan.manifest().greedy_prompts)
            .any(|(trajectory, prompt)| {
                trajectory.stable_id != prompt.stable_id
                    || trajectory.prompt_token_ids_sha256 != prompt.prefix_token_ids_sha256
            })
    {
        return Err(ExactTeacherTargetError::Invalid(
            "structural target receipt differs from the exact prediction plan".into(),
        ));
    }
    Ok(())
}

fn open_regular_target(path: &Path, expected_len: u64) -> Result<File, ExactTeacherTargetError> {
    let descriptor = fs::open(path, RETAINED_OPEN_FLAGS, Mode::empty())
        .map_err(std::io::Error::from)
        .map_err(|error| ExactTeacherTargetError::io(path, error))?;
    let file = File::from(descriptor);
    let stat = fs::fstat(&file)
        .map_err(std::io::Error::from)
        .map_err(|error| ExactTeacherTargetError::io(path, error))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile
        || stat.st_size < 0
        || u64::try_from(stat.st_size).ok() != Some(expected_len)
        || stat.st_nlink == 0
    {
        return Err(ExactTeacherTargetError::Invalid(
            "reference target is not the expected retained regular file".into(),
        ));
    }
    Ok(file)
}

fn read_u64(reader: &mut impl Read, path: &Path) -> Result<u64, ExactTeacherTargetError> {
    let mut bytes = [0_u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| ExactTeacherTargetError::io(path, error))?;
    Ok(u64::from_le_bytes(bytes))
}
