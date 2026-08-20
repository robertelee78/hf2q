use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::core::provenance::tensor_execution::ArtifactEvidence;
use crate::intelligence::calibration::{
    TeacherGreedyPromptReceipt, TeacherPredictionPointReceipt, VerifiedCalibrationPredictionPlan,
};

use super::publication::RetainedTargetTemp;
use super::reservation::{
    reservation_contract_sha256, StructuralTeacherTargetReservationReceiptV1,
    UnpublishedStructuralTeacherTargetReservation,
};
use super::*;

impl UnpublishedStructuralTeacherTargetReservation {
    pub(crate) fn validate_private(&self) -> Result<(), ExactTeacherTargetError> {
        if reservation_contract_sha256(&self.receipt)? != self.receipt.reservation_contract_sha256 {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target reservation receipt does not reproduce".into(),
            ));
        }
        self.temporary
            .verify_private_and_absent(u64::try_from(TARGET_MAGIC.len()).unwrap())?;
        let mut magic = vec![0_u8; TARGET_MAGIC.len()];
        self.temporary
            .as_file()
            .read_exact_at(&mut magic, 0)
            .map_err(|error| ExactTeacherTargetError::io(self.temporary.output(), error))?;
        if magic != TARGET_MAGIC {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target reservation magic differs".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn begin<'a>(
        self,
        plan: &'a VerifiedCalibrationPredictionPlan,
    ) -> Result<StructuralTeacherTargetStream<'a>, ExactTeacherTargetError> {
        self.validate_private()?;
        let preflight = preflight_structural_teacher_target(
            plan,
            self.receipt.vocabulary_size,
            self.receipt.limits,
        )?;
        if plan.manifest().manifest_sha256 != self.receipt.prediction_plan_sha256
            || plan.prediction_point_count() != self.receipt.prediction_point_count
            || plan.manifest().greedy_prompts.len() != self.receipt.generation_prompt_count
            || preflight.preflight_bytes != self.receipt.final_artifact_bytes
            || reservation_contract_sha256(&self.receipt)?
                != self.receipt.reservation_contract_sha256
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target reservation differs from its opaque prediction plan".into(),
            ));
        }
        Ok(StructuralTeacherTargetStream {
            plan,
            temporary: self.temporary,
            vocabulary_size: self.receipt.vocabulary_size,
            limits: self.receipt.limits,
            preflight_bytes: self.receipt.final_artifact_bytes,
            offset: u64::try_from(TARGET_MAGIC.len()).unwrap(),
            rows: Vec::with_capacity(self.receipt.prediction_point_count),
            trajectories: Vec::with_capacity(self.receipt.generation_prompt_count),
        })
    }
}

/// Checked target dimensions and token vocabulary closure. This remains a
/// structural capability, but it is intentionally produced before any family
/// runner allocates model weights or Metal buffers.
pub(crate) struct StructuralTeacherTargetPreflight<'a> {
    plan: &'a VerifiedCalibrationPredictionPlan,
    vocabulary_size: usize,
    limits: TeacherTargetArtifactLimits,
    preflight_bytes: u64,
}

pub(crate) fn preflight_structural_teacher_target(
    plan: &VerifiedCalibrationPredictionPlan,
    vocabulary_size: usize,
    limits: TeacherTargetArtifactLimits,
) -> Result<StructuralTeacherTargetPreflight<'_>, ExactTeacherTargetError> {
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
    plan.visit_examples(|_receipt, token_ids, _points, _greedy| {
        if token_ids
            .iter()
            .any(|token_id| usize::try_from(*token_id).unwrap_or(usize::MAX) >= vocabulary_size)
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher prediction example contains a token outside the declared vocabulary"
                    .into(),
            ));
        }
        Ok(())
    })?;
    Ok(StructuralTeacherTargetPreflight {
        plan,
        vocabulary_size,
        limits,
        preflight_bytes,
    })
}

/// Canonical row-at-a-time target writer used by the future family-owned
/// source teacher. It proves framing and plan closure only; callers cannot
/// promote its result into execution or allocator authority.
pub(crate) struct StructuralTeacherTargetStream<'a> {
    plan: &'a VerifiedCalibrationPredictionPlan,
    temporary: RetainedTargetTemp,
    vocabulary_size: usize,
    limits: TeacherTargetArtifactLimits,
    preflight_bytes: u64,
    offset: u64,
    rows: Vec<TeacherTargetRowReceipt>,
    trajectories: Vec<TeacherGreedyTrajectoryReceipt>,
}

impl<'a> StructuralTeacherTargetPreflight<'a> {
    pub(crate) fn preflight_bytes(&self) -> u64 {
        self.preflight_bytes
    }

    pub(crate) fn begin(
        self,
        output: &Path,
    ) -> Result<StructuralTeacherTargetStream<'a>, ExactTeacherTargetError> {
        let plan = self.plan;
        self.reserve(output)?.begin(plan)
    }

    pub(crate) fn reserve(
        self,
        output: &Path,
    ) -> Result<UnpublishedStructuralTeacherTargetReservation, ExactTeacherTargetError> {
        let Self {
            plan,
            vocabulary_size,
            limits,
            preflight_bytes,
        } = self;

        let mut temporary = RetainedTargetTemp::create(output)?;
        temporary
            .as_file_mut()
            .write_all(TARGET_MAGIC)
            .map_err(|error| ExactTeacherTargetError::io(output, error))?;

        let receipt = StructuralTeacherTargetReservationReceiptV1::new(
            plan.manifest().manifest_sha256.clone(),
            limits,
            vocabulary_size,
            plan.prediction_point_count(),
            plan.manifest().greedy_prompts.len(),
            preflight_bytes,
        )?;
        Ok(UnpublishedStructuralTeacherTargetReservation { receipt, temporary })
    }
}

impl StructuralTeacherTargetStream<'_> {
    pub(crate) fn write_row(
        &mut self,
        point: &TeacherPredictionPointReceipt,
        logits: &[f32],
    ) -> Result<(), ExactTeacherTargetError> {
        let expected = self
            .plan
            .manifest()
            .prediction_points
            .get(self.rows.len())
            .ok_or_else(|| {
                ExactTeacherTargetError::Invalid(
                    "teacher target contains an extra prediction row".into(),
                )
            })?;
        if point != expected {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target row differs from canonical prediction-plan order".into(),
            ));
        }
        if logits.len() != self.vocabulary_size {
            return Err(ExactTeacherTargetError::Invalid(format!(
                "teacher row {} has vocabulary {}, expected {}",
                point.point_ordinal,
                logits.len(),
                self.vocabulary_size
            )));
        }
        let payload = row_bytes(logits)?;
        let (argmax_token_id, top_k, logsumexp_f64_bits) = row_summary(logits, self.limits.top_k)?;
        let payload_offset = self
            .offset
            .checked_add(ROW_FRAME_BYTES)
            .ok_or_else(|| ExactTeacherTargetError::Invalid("row offset overflow".into()))?;
        let prefix_digest = digest_to_array(&point.prefix_token_ids_sha256)?;
        let point_ordinal = u64::try_from(point.point_ordinal).map_err(|_| {
            ExactTeacherTargetError::Invalid("prediction point ordinal overflow".into())
        })?;
        let vocabulary_size = u64::try_from(self.vocabulary_size).map_err(|_| {
            ExactTeacherTargetError::Invalid("target vocabulary size overflow".into())
        })?;
        let payload_bytes = u64::try_from(payload.len()).map_err(|_| {
            ExactTeacherTargetError::Invalid("target row byte length overflow".into())
        })?;
        self.temporary
            .as_file_mut()
            .write_all(ROW_MAGIC)
            .and_then(|_| {
                self.temporary
                    .as_file_mut()
                    .write_all(&point_ordinal.to_le_bytes())
            })
            .and_then(|_| {
                self.temporary
                    .as_file_mut()
                    .write_all(&vocabulary_size.to_le_bytes())
            })
            .and_then(|_| self.temporary.as_file_mut().write_all(&prefix_digest))
            .and_then(|_| {
                self.temporary
                    .as_file_mut()
                    .write_all(&payload_bytes.to_le_bytes())
            })
            .and_then(|_| self.temporary.as_file_mut().write_all(&payload))
            .map_err(|error| ExactTeacherTargetError::io(self.temporary.output(), error))?;
        self.offset = payload_offset
            .checked_add(payload_bytes)
            .ok_or_else(|| ExactTeacherTargetError::Invalid("row end overflow".into()))?;
        self.rows.push(TeacherTargetRowReceipt {
            point_ordinal: point.point_ordinal,
            stable_id: point.stable_id.clone(),
            point_kind: point.kind,
            prefix_token_count: point.prefix_token_count,
            prefix_token_ids_sha256: point.prefix_token_ids_sha256.clone(),
            vocabulary_size: self.vocabulary_size,
            payload_offset,
            payload_bytes,
            payload_sha256: hash_bytes(&payload),
            argmax_token_id,
            top_k,
            logsumexp_f64_bits,
        });
        Ok(())
    }

    pub(crate) fn write_trajectory(
        &mut self,
        prompt: &TeacherGreedyPromptReceipt,
        token_ids: &[u32],
    ) -> Result<(), ExactTeacherTargetError> {
        let expected = self
            .plan
            .manifest()
            .greedy_prompts
            .get(self.trajectories.len())
            .ok_or_else(|| {
                ExactTeacherTargetError::Invalid(
                    "teacher target contains an extra greedy trajectory".into(),
                )
            })?;
        if prompt != expected {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher greedy trajectory differs from canonical prediction-plan order".into(),
            ));
        }
        if token_ids.len() != EXACT_TEACHER_GREEDY_TOKEN_COUNT {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher greedy trajectory must contain exactly 32 tokens".into(),
            ));
        }
        if token_ids.iter().any(|token_id| {
            usize::try_from(*token_id).unwrap_or(usize::MAX) >= self.vocabulary_size
        }) {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher greedy trajectory contains a token outside the declared vocabulary".into(),
            ));
        }
        self.trajectories.push(TeacherGreedyTrajectoryReceipt {
            stable_id: prompt.stable_id.clone(),
            prompt_token_ids_sha256: prompt.prefix_token_ids_sha256.clone(),
            token_ids: token_ids.to_vec(),
            token_ids_sha256: trajectory_sha256(token_ids)?,
        });
        Ok(())
    }

    /// Seal and independently verify the exact temporary inode without
    /// publishing it at the requested destination.
    pub(crate) fn finish_unpublished(
        mut self,
    ) -> Result<UnpublishedStructuralTeacherTargetArtifact, ExactTeacherTargetError> {
        if self.rows.len() != self.plan.prediction_point_count()
            || self.trajectories.len() != self.plan.manifest().greedy_prompts.len()
        {
            return Err(ExactTeacherTargetError::Invalid(
                "teacher target is missing prediction rows or greedy trajectories".into(),
            ));
        }
        if self.offset != self.preflight_bytes {
            return Err(ExactTeacherTargetError::Invalid(
                "written teacher target size differs from preflight".into(),
            ));
        }
        self.temporary
            .as_file_mut()
            .flush()
            .and_then(|_| self.temporary.as_file().sync_all())
            .map_err(|error| ExactTeacherTargetError::io(self.temporary.output(), error))?;
        let artifact_sha256 =
            hash_open_file_bounded(self.temporary.as_file_mut(), self.preflight_bytes)?;
        let mut receipt = ExactTeacherTargetReceipt {
            schema_version: EXACT_TEACHER_TARGET_SCHEMA_VERSION,
            semantics: TARGET_SEMANTICS.into(),
            prediction_plan_sha256: self.plan.manifest().manifest_sha256.clone(),
            limits: self.limits,
            vocabulary_size: self.vocabulary_size,
            prediction_point_count: self.plan.prediction_point_count(),
            generation_prompt_count: self.plan.manifest().greedy_prompts.len(),
            target_artifact: ArtifactEvidence {
                artifact_id: "exact_teacher_logits".into(),
                role: "structural_full_vocabulary_f32_target_rows".into(),
                byte_len: self.preflight_bytes,
                sha256: artifact_sha256,
            },
            rows: self.rows,
            greedy_trajectories: self.trajectories,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = receipt_sha256(&receipt)?;
        verify::verify_structural_teacher_target_artifact(self.temporary.as_file_mut(), &receipt)?;
        Ok(UnpublishedStructuralTeacherTargetArtifact {
            receipt,
            temporary: self.temporary,
        })
    }

    /// Compatibility transition for structural-only callers. Family-owned
    /// execution builds its completion receipt between `finish_unpublished`
    /// and `publish_noclobber` instead.
    pub(crate) fn finish(
        self,
    ) -> Result<StructurallyVerifiedTeacherTargetArtifact, ExactTeacherTargetError> {
        self.finish_unpublished()?.publish_noclobber()
    }
}

impl UnpublishedStructuralTeacherTargetArtifact {
    /// Reverify the retained temporary inode and publish it without replacing
    /// an existing destination. This is the final fallible transition: after
    /// successful publication no additional receipt construction is needed.
    pub(crate) fn publish_noclobber(
        self,
    ) -> Result<StructurallyVerifiedTeacherTargetArtifact, ExactTeacherTargetError> {
        let receipt = self.receipt;
        let output = self.temporary.output().to_owned();
        let expected_len = receipt.target_artifact.byte_len;
        let file = self.temporary.publish_noclobber(expected_len, |file| {
            verify::verify_structural_teacher_target_artifact(file, &receipt)
        })?;
        Ok(StructurallyVerifiedTeacherTargetArtifact {
            receipt,
            file,
            path: output,
        })
    }
}

/// A byte-verified target retained under a private temporary name until a
/// family-owned completion receipt is ready.
pub(crate) struct UnpublishedStructuralTeacherTargetArtifact {
    receipt: ExactTeacherTargetReceipt,
    temporary: RetainedTargetTemp,
}

impl UnpublishedStructuralTeacherTargetArtifact {
    pub(crate) fn receipt(&self) -> &ExactTeacherTargetReceipt {
        &self.receipt
    }

    #[cfg(test)]
    pub(in crate::intelligence::exact_teacher) fn retained_file_for_test(&mut self) -> &mut File {
        self.temporary.as_file_mut()
    }
}
