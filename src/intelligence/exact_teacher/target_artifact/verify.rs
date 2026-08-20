//! Independent retained-file verification for structural teacher targets.

use std::io::BufReader;

use super::*;

fn read_u64(reader: &mut impl Read) -> Result<u64, ExactTeacherTargetError> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?;
    Ok(u64::from_le_bytes(bytes))
}

pub(super) fn verify_structural_teacher_target_artifact(
    file: &mut File,
    receipt: &ExactTeacherTargetReceipt,
) -> Result<(), ExactTeacherTargetError> {
    if receipt.schema_version != EXACT_TEACHER_TARGET_SCHEMA_VERSION
        || receipt.semantics != TARGET_SEMANTICS
        || !is_lower_sha256(&receipt.prediction_plan_sha256)
        || !is_lower_sha256(&receipt.target_artifact.sha256)
        || !is_lower_sha256(&receipt.receipt_sha256)
        || receipt.rows.is_empty()
        || receipt.rows.len() != receipt.prediction_point_count
        || receipt.rows.len() > receipt.limits.max_prediction_rows
        || receipt.greedy_trajectories.len() != receipt.generation_prompt_count
        || receipt.vocabulary_size == 0
        || receipt.vocabulary_size > receipt.limits.max_vocabulary_size
        || receipt.limits.max_vocabulary_size > MAX_TARGET_VOCABULARY_SIZE
        || receipt.limits.max_prediction_rows > MAX_TARGET_PREDICTION_ROWS
        || receipt.limits.max_target_bytes > MAX_TARGET_ARTIFACT_BYTES
        || receipt.limits.top_k == 0
        || receipt.limits.top_k > receipt.vocabulary_size
        || receipt.limits.top_k > MAX_TARGET_TOP_K
        || receipt
            .rows
            .len()
            .checked_mul(receipt.limits.top_k)
            .is_none_or(|entries| entries > MAX_TARGET_SUMMARY_ENTRIES)
        || receipt.target_artifact.artifact_id != "exact_teacher_logits"
        || receipt.target_artifact.role != "structural_full_vocabulary_f32_target_rows"
        || receipt_sha256(receipt)? != receipt.receipt_sha256
    {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher target receipt identity is invalid".into(),
        ));
    }
    let canonical_bytes = checked_target_bytes(receipt.rows.len(), receipt.vocabulary_size)?;
    let metadata_bytes = file
        .metadata()
        .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?
        .len();
    if metadata_bytes != canonical_bytes
        || metadata_bytes != receipt.target_artifact.byte_len
        || metadata_bytes > receipt.limits.max_target_bytes
    {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher target artifact length is invalid".into(),
        ));
    }
    let actual_sha256 = hash_open_file_bounded(file, canonical_bytes)?;
    if actual_sha256 != receipt.target_artifact.sha256 {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher target artifact identity is invalid".into(),
        ));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?;
    let mut reader = BufReader::new(file);
    let mut magic = vec![0u8; TARGET_MAGIC.len()];
    reader
        .read_exact(&mut magic)
        .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?;
    if magic != TARGET_MAGIC {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher target magic is invalid".into(),
        ));
    }
    let mut expected_offset = u64::try_from(TARGET_MAGIC.len()).unwrap();
    for (ordinal, expected) in receipt.rows.iter().enumerate() {
        let mut row_magic = [0u8; 4];
        reader
            .read_exact(&mut row_magic)
            .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?;
        let observed_ordinal = read_u64(&mut reader)?;
        let observed_vocab = read_u64(&mut reader)?;
        let mut observed_prefix_hash = [0u8; 32];
        reader
            .read_exact(&mut observed_prefix_hash)
            .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?;
        let payload_bytes = read_u64(&mut reader)?;
        expected_offset = expected_offset
            .checked_add(ROW_FRAME_BYTES)
            .ok_or_else(|| ExactTeacherTargetError::Invalid("row offset overflow".into()))?;
        if row_magic != *ROW_MAGIC
            || observed_ordinal
                != u64::try_from(ordinal).map_err(|_| {
                    ExactTeacherTargetError::Invalid("target row ordinal overflow".into())
                })?
            || expected.point_ordinal != ordinal
            || observed_vocab != receipt.vocabulary_size as u64
            || observed_prefix_hash != digest_to_array(&expected.prefix_token_ids_sha256)?
            || payload_bytes != expected.payload_bytes
            || expected.payload_offset != expected_offset
            || expected.vocabulary_size != receipt.vocabulary_size
            || expected.top_k.len() != receipt.limits.top_k
            || matches!(
                expected.point_kind,
                crate::intelligence::calibration::TeacherPredictionPointKind::TeacherForced {
                    target_token_id,
                    ..
                } if usize::try_from(target_token_id).unwrap_or(usize::MAX)
                    >= receipt.vocabulary_size
            )
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target row framing or order is invalid".into(),
            ));
        }
        let canonical_row_bytes = checked_row_bytes(receipt.vocabulary_size)?;
        let row_end = expected_offset
            .checked_add(payload_bytes)
            .ok_or_else(|| ExactTeacherTargetError::Invalid("row end overflow".into()))?;
        if payload_bytes != canonical_row_bytes || row_end > metadata_bytes {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target row byte extent is invalid".into(),
            ));
        }
        let payload_len = usize::try_from(payload_bytes)
            .map_err(|_| ExactTeacherTargetError::Invalid("row payload is too large".into()))?;
        let mut payload = vec![0u8; payload_len];
        reader
            .read_exact(&mut payload)
            .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?;
        if hash_bytes(&payload) != expected.payload_sha256
            || payload.len() != receipt.vocabulary_size.checked_mul(4).unwrap_or(usize::MAX)
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target row payload is invalid".into(),
            ));
        }
        let logits = payload
            .chunks_exact(4)
            .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap())))
            .collect::<Vec<_>>();
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target row contains non-finite logits".into(),
            ));
        }
        let (argmax, top_k, logsumexp) = row_summary(&logits, receipt.limits.top_k)?;
        if argmax != expected.argmax_token_id
            || top_k != expected.top_k
            || logsumexp != expected.logsumexp_f64_bits
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target row summary does not reproduce".into(),
            ));
        }
        expected_offset = row_end;
    }
    let mut trailing = [0u8; 1];
    if reader
        .read(&mut trailing)
        .map_err(|error| ExactTeacherTargetError::io(Path::new("<retained-target>"), error))?
        != 0
        || expected_offset != receipt.target_artifact.byte_len
        || receipt.greedy_trajectories.is_empty()
        || receipt.greedy_trajectories.iter().any(|trajectory| {
            trajectory.stable_id.is_empty()
                || !is_lower_sha256(&trajectory.prompt_token_ids_sha256)
                || trajectory.token_ids.len() != EXACT_TEACHER_GREEDY_TOKEN_COUNT
                || trajectory.token_ids.iter().any(|token_id| {
                    usize::try_from(*token_id).unwrap_or(usize::MAX) >= receipt.vocabulary_size
                })
                || trajectory_sha256(&trajectory.token_ids)
                    .map(|hash| hash != trajectory.token_ids_sha256)
                    .unwrap_or(true)
        })
    {
        return Err(ExactTeacherTargetError::Invalid(
            "teacher target trailing bytes or greedy trajectory is invalid".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(in crate::intelligence::exact_teacher) fn verify_for_test(
    artifact: &mut StructurallyVerifiedTeacherTargetArtifact,
) -> Result<(), ExactTeacherTargetError> {
    verify_structural_teacher_target_artifact(&mut artifact.file, &artifact.receipt)
}
