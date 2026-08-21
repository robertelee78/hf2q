//! Final opaque-owner revalidation before worker placement.

use anyhow::{ensure, Result};

use crate::inference::models::qwen35::kv_cache::PreparedQwen35BaseTextCacheV1;
use crate::inference::models::qwen35::source_precision::teacher_execution_plan::Qwen35SourceTeacherExpectedWorkV1;
use crate::intelligence::calibration::VerifiedCalibrationPredictionPlan;
use crate::intelligence::exact_teacher::UnpublishedStructuralTeacherTargetReservation;

use super::super::super::PreparedQwen35SourceTeacherV1;
use super::super::{
    catalog_sha256, receipt_sha256, PreparedQwen35SourceTeacherRunInputsReceiptV1,
    PreparedQwen35SourceTeacherRunInputsV1,
};

pub(super) struct SourceTeacherWorkerPartsV1 {
    pub(super) teacher: PreparedQwen35SourceTeacherV1,
    pub(super) cache: PreparedQwen35BaseTextCacheV1,
    pub(super) prediction_plan: VerifiedCalibrationPredictionPlan,
    pub(super) target_reservation: UnpublishedStructuralTeacherTargetReservation,
    pub(super) expected_work: Qwen35SourceTeacherExpectedWorkV1,
    pub(super) receipt: PreparedQwen35SourceTeacherRunInputsReceiptV1,
}

impl PreparedQwen35SourceTeacherRunInputsV1 {
    pub(super) fn into_worker_parts(self) -> Result<SourceTeacherWorkerPartsV1> {
        ensure!(
            catalog_sha256(&self.receipt)? == self.receipt.run_inputs_catalog_sha256
                && receipt_sha256(&self.receipt)? == self.receipt.run_inputs_receipt_sha256
                && self._prediction_plan.manifest().manifest_sha256
                    == self.receipt.prediction_plan_sha256
                && self._expected_work == self.receipt.expected_work
                && self._teacher.receipt.topology_sha256 == self.receipt.topology_sha256
                && self._teacher.receipt.graph_catalog_sha256
                    == self.receipt.prepared_graph_catalog_sha256
                && self._teacher.receipt.preparation_receipt_sha256
                    == self.receipt.preparation_receipt_sha256
                && self._teacher.device.name() == self.receipt.device_name
                && self._teacher.device.registry_id() == self.receipt.device_registry_id
                && self._cache.receipt().plan().layout_sha256() == self.receipt.cache_layout_sha256
                && self._cache.receipt().receipt_sha256() == self.receipt.cache_receipt_sha256
                && self._target_reservation.receipt().contract_sha256()
                    == self.receipt.target_reservation_contract_sha256
                && self
                    ._teacher
                    .receipt
                    .runtime
                    .unmeasured_runtime_reserve_bytes
                    >= self._teacher.receipt.runtime.one_logit_row_bytes
                && !self.receipt.graph_encoded
                && !self.receipt.submitted
                && !self.receipt.completed
                && !self.receipt.target_finished
                && !self.receipt.target_published
                && !self.receipt.q4_repack
                && !self.receipt.dwq
                && !self.receipt.tq
                && !self.receipt.mtp_executed,
            "source-teacher run inputs do not reproduce at worker consumption"
        );
        self._target_reservation.validate_private()?;
        self._teacher.snapshot.rehash_retained_files()?;
        Ok(SourceTeacherWorkerPartsV1 {
            teacher: self._teacher,
            cache: self._cache,
            prediction_plan: self._prediction_plan,
            target_reservation: self._target_reservation,
            expected_work: self._expected_work,
            receipt: self.receipt,
        })
    }
}
