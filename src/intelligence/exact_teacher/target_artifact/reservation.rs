//! Empty retained target inode bound to an exact structural contract.

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::publication::RetainedTargetTemp;
use super::{ExactTeacherTargetError, TeacherTargetArtifactLimits};

const TARGET_RESERVATION_SCHEMA_VERSION: u32 = 1;
const TARGET_RESERVATION_PROFILE: &str = "structural_teacher_target_private_reservation_v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct StructuralTeacherTargetReservationReceiptV1 {
    pub(super) schema_version: u32,
    pub(super) profile: &'static str,
    pub(super) prediction_plan_sha256: String,
    pub(super) limits: TeacherTargetArtifactLimits,
    pub(super) vocabulary_size: usize,
    pub(super) prediction_point_count: usize,
    pub(super) generation_prompt_count: usize,
    pub(super) final_artifact_bytes: u64,
    pub(super) reservation_contract_sha256: String,
}

#[derive(Serialize)]
struct ReservationHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    prediction_plan_sha256: &'a str,
    limits: TeacherTargetArtifactLimits,
    vocabulary_size: usize,
    prediction_point_count: usize,
    generation_prompt_count: usize,
    final_artifact_bytes: u64,
}

impl StructuralTeacherTargetReservationReceiptV1 {
    pub(super) fn new(
        prediction_plan_sha256: String,
        limits: TeacherTargetArtifactLimits,
        vocabulary_size: usize,
        prediction_point_count: usize,
        generation_prompt_count: usize,
        final_artifact_bytes: u64,
    ) -> Result<Self, ExactTeacherTargetError> {
        let mut receipt = Self {
            schema_version: TARGET_RESERVATION_SCHEMA_VERSION,
            profile: TARGET_RESERVATION_PROFILE,
            prediction_plan_sha256,
            limits,
            vocabulary_size,
            prediction_point_count,
            generation_prompt_count,
            final_artifact_bytes,
            reservation_contract_sha256: String::new(),
        };
        receipt.reservation_contract_sha256 = reservation_contract_sha256(&receipt)?;
        Ok(receipt)
    }

    pub(crate) fn contract_sha256(&self) -> &str {
        &self.reservation_contract_sha256
    }

    #[cfg(test)]
    pub(crate) fn prediction_plan_sha256(&self) -> &str {
        &self.prediction_plan_sha256
    }

    #[cfg(test)]
    pub(crate) fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    pub(crate) fn prediction_point_count(&self) -> usize {
        self.prediction_point_count
    }

    pub(crate) fn generation_prompt_count(&self) -> usize {
        self.generation_prompt_count
    }

    pub(crate) fn final_artifact_bytes(&self) -> u64 {
        self.final_artifact_bytes
    }
}

/// An empty private target inode bound to an exact structural contract.
///
/// This type owns no logits and cannot finish or publish a target. A later
/// consuming runner must rebind it to the same opaque prediction plan before
/// it can obtain the row-at-a-time stream.
pub(crate) struct UnpublishedStructuralTeacherTargetReservation {
    pub(super) receipt: StructuralTeacherTargetReservationReceiptV1,
    pub(super) temporary: RetainedTargetTemp,
}

impl UnpublishedStructuralTeacherTargetReservation {
    pub(crate) fn receipt(&self) -> &StructuralTeacherTargetReservationReceiptV1 {
        &self.receipt
    }
}

pub(super) fn reservation_contract_sha256(
    receipt: &StructuralTeacherTargetReservationReceiptV1,
) -> Result<String, ExactTeacherTargetError> {
    let bytes = serde_json::to_vec(&ReservationHashView {
        schema_version: receipt.schema_version,
        profile: receipt.profile,
        prediction_plan_sha256: &receipt.prediction_plan_sha256,
        limits: receipt.limits,
        vocabulary_size: receipt.vocabulary_size,
        prediction_point_count: receipt.prediction_point_count,
        generation_prompt_count: receipt.generation_prompt_count,
        final_artifact_bytes: receipt.final_artifact_bytes,
    })
    .map_err(|error| ExactTeacherTargetError::Serialization(error.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}
