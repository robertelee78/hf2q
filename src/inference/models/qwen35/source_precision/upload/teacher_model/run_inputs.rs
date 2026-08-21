//! Ordered ownership boundary for a completed source-teacher transaction.
//!
//! This module consumes the source-bound work capability, reserves the target
//! destination before weight allocation, prepares the exact source weights,
//! then attaches a fresh base-text-only cache from the prepared teacher's own
//! device/config. The result is intentionally inert and exposes no graph,
//! cache, buffer, target writer, or execution method; the family worker can
//! only consume the whole owner once.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use mlx_native::MlxDevice;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::inference::models::qwen35::kv_cache::{
    plan_qwen35_base_text_cache, prepare_qwen35_base_text_cache, PreparedQwen35BaseTextCacheV1,
};
use crate::inference::models::qwen35::source_precision::teacher_execution_plan::{
    Qwen35SourceTeacherExpectedWorkV1, StructurallyBoundQwen35SourceTeacherWorkV1,
};
use crate::inference::models::qwen35::source_precision::topology::VerifiedQwen35Bf16TopologyV1;
use crate::inference::models::qwen35::source_precision::upload_plan::{
    QwenSourceMetalCapacityV1, QwenSourceMetalUploadLimits,
};
use crate::intelligence::calibration::VerifiedCalibrationPredictionPlan;
use crate::intelligence::exact_teacher::{
    preflight_structural_teacher_target, UnpublishedStructuralTeacherTargetReservation,
};

use super::{
    combined_capacity_preflight, observe_capacity, prepare_qwen35_source_teacher, runtime_envelope,
    validate_incremental_capacity, PreparedQwen35SourceTeacherV1,
    Qwen35SourceTeacherCapacityPreflightV1, Qwen35SourceTeacherLimitsV1,
};

#[cfg(test)]
mod tests;
mod worker;

pub(crate) use worker::run_qwen35_source_teacher;

const RUN_INPUTS_SCHEMA_VERSION: u32 = 1;
const RUN_INPUTS_PROFILE: &str = "dense_qwen35_source_teacher_ordered_run_inputs_v1";
const CAPACITY_RECHECK_PROFILE: &str = "dense_qwen35_post_weight_capacity_recheck_v1";

/// Caller-selected allowances that do not change semantic work dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct Qwen35SourceTeacherPreparationPolicyV1 {
    pub max_cpu_control_mirror_bytes: u64,
    pub unmeasured_runtime_reserve_bytes: u64,
}

#[derive(Serialize)]
struct RunInputsCatalogHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    work_plan_sha256: &'a str,
    topology_sha256: &'a str,
    prediction_plan_sha256: &'a str,
    projected_execution_config_sha256: &'a str,
    prepared_graph_catalog_sha256: &'a str,
    cache_layout_sha256: &'a str,
    target_reservation_contract_sha256: &'a str,
    expected_work: Qwen35SourceTeacherExpectedWorkV1,
    weight_precision: &'static str,
    cache_precision: &'static str,
    q4_repack: bool,
    dwq: bool,
    tq: bool,
    mtp_executed: bool,
    graph_encoded: bool,
    submitted: bool,
    completed: bool,
    target_finished: bool,
    target_published: bool,
}

#[derive(Serialize)]
struct RunInputsReceiptHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    run_inputs_catalog_sha256: &'a str,
    preparation_receipt_sha256: &'a str,
    cache_receipt_sha256: &'a str,
    runtime_capacity_recheck: &'a RuntimeCapacityRecheckV1,
    device_name: &'a str,
    device_registry_id: u64,
}

#[derive(Serialize)]
struct RuntimeCapacityRecheckHashView {
    profile: &'static str,
    capacity: QwenSourceMetalCapacityV1,
    accounted_runtime_payload_bytes: u64,
    unmeasured_runtime_reserve_bytes: u64,
    host_reserve_bytes: u64,
    metal_reserve_bytes: u64,
    host_required_bytes: u64,
    metal_required_bytes: u64,
    metal_available_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct RuntimeCapacityRecheckV1 {
    profile: &'static str,
    capacity: QwenSourceMetalCapacityV1,
    accounted_runtime_payload_bytes: u64,
    unmeasured_runtime_reserve_bytes: u64,
    host_reserve_bytes: u64,
    metal_reserve_bytes: u64,
    host_required_bytes: u64,
    metal_required_bytes: u64,
    metal_available_bytes: u64,
    capacity_recheck_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PreparedQwen35SourceTeacherRunInputsReceiptV1 {
    schema_version: u32,
    profile: &'static str,
    work_plan_sha256: String,
    topology_sha256: String,
    prediction_plan_sha256: String,
    projected_execution_config_sha256: String,
    prepared_graph_catalog_sha256: String,
    preparation_receipt_sha256: String,
    cache_layout_sha256: String,
    cache_receipt_sha256: String,
    runtime_capacity_recheck: RuntimeCapacityRecheckV1,
    target_reservation_contract_sha256: String,
    device_name: String,
    device_registry_id: u64,
    expected_work: Qwen35SourceTeacherExpectedWorkV1,
    weight_precision: &'static str,
    cache_precision: &'static str,
    q4_repack: bool,
    dwq: bool,
    tq: bool,
    mtp_executed: bool,
    graph_encoded: bool,
    submitted: bool,
    completed: bool,
    target_finished: bool,
    target_published: bool,
    run_inputs_catalog_sha256: String,
    run_inputs_receipt_sha256: String,
}

/// Opaque ownership of every input required by the subsequent runner.
///
/// The prediction plan, private target reservation, prepared weights, and
/// cache remain live and inseparable. This type cannot encode or submit work.
pub(crate) struct PreparedQwen35SourceTeacherRunInputsV1 {
    _teacher: PreparedQwen35SourceTeacherV1,
    _cache: PreparedQwen35BaseTextCacheV1,
    _prediction_plan: VerifiedCalibrationPredictionPlan,
    _target_reservation: UnpublishedStructuralTeacherTargetReservation,
    _expected_work: Qwen35SourceTeacherExpectedWorkV1,
    receipt: PreparedQwen35SourceTeacherRunInputsReceiptV1,
}

impl PreparedQwen35SourceTeacherRunInputsV1 {
    #[cfg(test)]
    pub(crate) fn catalog_sha256(&self) -> &str {
        &self.receipt.run_inputs_catalog_sha256
    }

    #[cfg(test)]
    pub(crate) fn receipt_sha256(&self) -> &str {
        &self.receipt.run_inputs_receipt_sha256
    }

    #[cfg(test)]
    fn receipt_for_test(&self) -> &PreparedQwen35SourceTeacherRunInputsReceiptV1 {
        &self.receipt
    }
}

pub(crate) fn prepare_qwen35_source_teacher_run_inputs(
    work: StructurallyBoundQwen35SourceTeacherWorkV1,
    output: &Path,
    device: &MlxDevice,
    upload_limits: QwenSourceMetalUploadLimits,
    preparation_policy: Qwen35SourceTeacherPreparationPolicyV1,
) -> Result<PreparedQwen35SourceTeacherRunInputsV1> {
    prepare_run_inputs_with(
        work,
        output,
        preparation_policy,
        |topology, teacher_limits| {
            prepare_qwen35_source_teacher(topology, device, upload_limits, teacher_limits)
        },
        prepare_cache_from_teacher,
    )
}

/// Observe the exact combined preparation requirement without allocating
/// source weights or cache state. The production transition independently
/// re-observes and revalidates immediately before allocation.
pub(crate) fn preflight_qwen35_source_teacher_run_inputs_capacity(
    work: &StructurallyBoundQwen35SourceTeacherWorkV1,
    device: &MlxDevice,
    upload_limits: QwenSourceMetalUploadLimits,
    preparation_policy: Qwen35SourceTeacherPreparationPolicyV1,
) -> Result<Qwen35SourceTeacherCapacityPreflightV1> {
    let (topology, expected_work) = work.preparation_parts();
    let config = topology.projected_config_for_teacher()?;
    let limits = Qwen35SourceTeacherLimitsV1 {
        max_sequence_tokens: u32::try_from(expected_work.max_cache_tokens)
            .context("source teacher cache work is not representable")?,
        max_target_rows: u32::try_from(expected_work.prediction_row_count)
            .context("source teacher target rows are not representable")?,
        max_cpu_control_mirror_bytes: preparation_policy.max_cpu_control_mirror_bytes,
        unmeasured_runtime_reserve_bytes: preparation_policy.unmeasured_runtime_reserve_bytes,
    };
    let runtime = runtime_envelope(&config, limits)?;
    combined_capacity_preflight(
        topology.planned_output_bytes()?,
        &runtime,
        upload_limits,
        observe_capacity(device),
    )
}

fn prepare_cache_from_teacher(
    teacher: &PreparedQwen35SourceTeacherV1,
    max_sequence_tokens: u32,
) -> Result<(PreparedQwen35BaseTextCacheV1, RuntimeCapacityRecheckV1)> {
    let capacity = observe_capacity(&teacher.device);
    let recheck = runtime_capacity_recheck(teacher, capacity)?;
    let cache =
        prepare_qwen35_base_text_cache(&teacher.config, &teacher.device, max_sequence_tokens)?;
    Ok((cache, recheck))
}

fn prepare_run_inputs_with<PrepareTeacher, PrepareCache>(
    work: StructurallyBoundQwen35SourceTeacherWorkV1,
    output: &Path,
    preparation_policy: Qwen35SourceTeacherPreparationPolicyV1,
    prepare_teacher: PrepareTeacher,
    prepare_cache: PrepareCache,
) -> Result<PreparedQwen35SourceTeacherRunInputsV1>
where
    PrepareTeacher: FnOnce(
        VerifiedQwen35Bf16TopologyV1,
        Qwen35SourceTeacherLimitsV1,
    ) -> Result<PreparedQwen35SourceTeacherV1>,
    PrepareCache: FnOnce(
        &PreparedQwen35SourceTeacherV1,
        u32,
    ) -> Result<(PreparedQwen35BaseTextCacheV1, RuntimeCapacityRecheckV1)>,
{
    let (topology, prediction_plan, target_limits, _run_limits, expected_work, work_plan_sha256) =
        work.into_parts();
    let projected = topology.projected_config_for_teacher()?;
    let work_topology_sha256 = topology.topology_sha256().to_owned();
    let max_sequence_tokens = u32::try_from(expected_work.max_cache_tokens)
        .context("source teacher cache work is not representable")?;
    let max_target_rows = u32::try_from(expected_work.prediction_row_count)
        .context("source teacher target rows are not representable")?;
    let teacher_limits = Qwen35SourceTeacherLimitsV1 {
        max_sequence_tokens,
        max_target_rows,
        max_cpu_control_mirror_bytes: preparation_policy.max_cpu_control_mirror_bytes,
        unmeasured_runtime_reserve_bytes: preparation_policy.unmeasured_runtime_reserve_bytes,
    };
    let vocabulary_size = usize::try_from(projected.vocab_size)
        .context("source teacher vocabulary is not representable")?;
    let projected_execution_config_sha256 = super::config_hash(&projected)?;
    let expected_cache_plan = plan_qwen35_base_text_cache(&projected, max_sequence_tokens)?;
    let target_reservation =
        preflight_structural_teacher_target(&prediction_plan, vocabulary_size, target_limits)?
            .reserve(output)?;
    ensure!(
        target_reservation.receipt().final_artifact_bytes() == expected_work.target_artifact_bytes
            && target_reservation.receipt().prediction_point_count()
                == expected_work.prediction_row_count
            && target_reservation.receipt().generation_prompt_count()
                == expected_work.generation_prompt_count,
        "source teacher target reservation differs from expected work"
    );

    let teacher = prepare_teacher(topology, teacher_limits)?;
    ensure!(
        teacher.receipt.topology_sha256 == work_topology_sha256
            && teacher.receipt.projected_execution_config_sha256
                == projected_execution_config_sha256
            && super::config_hash(&teacher.config)? == projected_execution_config_sha256
            && teacher.device.name() == teacher.receipt.device_name
            && teacher.device.registry_id() == teacher.receipt.device_registry_id
            && teacher.receipt.runtime.max_sequence_tokens == max_sequence_tokens
            && teacher.receipt.runtime.max_target_rows == max_target_rows
            && !teacher.receipt.graph_executed
            && !teacher.receipt.q4_repack
            && !teacher.receipt.dwq
            && !teacher.receipt.tq
            && !teacher.receipt.mtp_executed,
        "source teacher preparation differs from the ordered work contract"
    );
    target_reservation.validate_private()?;
    let (cache, capacity_recheck) = prepare_cache(&teacher, max_sequence_tokens)?;
    let cache_receipt = cache.receipt();
    ensure!(
        runtime_capacity_recheck(&teacher, capacity_recheck.capacity)? == capacity_recheck,
        "source teacher post-weight capacity recheck does not reproduce"
    );
    ensure!(
        cache_receipt.device_registry_id() == teacher.receipt.device_registry_id
            && cache_receipt.device_name() == teacher.receipt.device_name
            && cache_receipt.plan() == &expected_cache_plan
            && expected_cache_plan.base_full_attention_cache_bytes()
                == teacher.receipt.runtime.base_full_attention_cache_bytes
            && expected_cache_plan.base_linear_attention_state_bytes()
                == teacher.receipt.runtime.base_linear_attention_state_bytes,
        "source teacher cache differs from the prepared teacher"
    );
    target_reservation.validate_private()?;

    let mut receipt = PreparedQwen35SourceTeacherRunInputsReceiptV1 {
        schema_version: RUN_INPUTS_SCHEMA_VERSION,
        profile: RUN_INPUTS_PROFILE,
        work_plan_sha256,
        topology_sha256: work_topology_sha256,
        prediction_plan_sha256: prediction_plan.manifest().manifest_sha256.clone(),
        projected_execution_config_sha256,
        prepared_graph_catalog_sha256: teacher.receipt.graph_catalog_sha256.clone(),
        preparation_receipt_sha256: teacher.receipt.preparation_receipt_sha256.clone(),
        cache_layout_sha256: cache_receipt.plan().layout_sha256().into(),
        cache_receipt_sha256: cache_receipt.receipt_sha256().into(),
        runtime_capacity_recheck: capacity_recheck,
        target_reservation_contract_sha256: target_reservation.receipt().contract_sha256().into(),
        device_name: teacher.receipt.device_name.clone(),
        device_registry_id: teacher.receipt.device_registry_id,
        expected_work,
        weight_precision: "source_bf16_controls_f32",
        cache_precision: "base_text_f32_one_sequence",
        q4_repack: false,
        dwq: false,
        tq: false,
        mtp_executed: false,
        graph_encoded: false,
        submitted: false,
        completed: false,
        target_finished: false,
        target_published: false,
        run_inputs_catalog_sha256: String::new(),
        run_inputs_receipt_sha256: String::new(),
    };
    receipt.run_inputs_catalog_sha256 = catalog_sha256(&receipt)?;
    receipt.run_inputs_receipt_sha256 = receipt_sha256(&receipt)?;
    Ok(PreparedQwen35SourceTeacherRunInputsV1 {
        _teacher: teacher,
        _cache: cache,
        _prediction_plan: prediction_plan,
        _target_reservation: target_reservation,
        _expected_work: expected_work,
        receipt,
    })
}

fn catalog_sha256(receipt: &PreparedQwen35SourceTeacherRunInputsReceiptV1) -> Result<String> {
    let bytes = serde_json::to_vec(&RunInputsCatalogHashView {
        schema_version: receipt.schema_version,
        profile: receipt.profile,
        work_plan_sha256: &receipt.work_plan_sha256,
        topology_sha256: &receipt.topology_sha256,
        prediction_plan_sha256: &receipt.prediction_plan_sha256,
        projected_execution_config_sha256: &receipt.projected_execution_config_sha256,
        prepared_graph_catalog_sha256: &receipt.prepared_graph_catalog_sha256,
        cache_layout_sha256: &receipt.cache_layout_sha256,
        target_reservation_contract_sha256: &receipt.target_reservation_contract_sha256,
        expected_work: receipt.expected_work,
        weight_precision: receipt.weight_precision,
        cache_precision: receipt.cache_precision,
        q4_repack: receipt.q4_repack,
        dwq: receipt.dwq,
        tq: receipt.tq,
        mtp_executed: receipt.mtp_executed,
        graph_encoded: receipt.graph_encoded,
        submitted: receipt.submitted,
        completed: receipt.completed,
        target_finished: receipt.target_finished,
        target_published: receipt.target_published,
    })?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn receipt_sha256(receipt: &PreparedQwen35SourceTeacherRunInputsReceiptV1) -> Result<String> {
    let bytes = serde_json::to_vec(&RunInputsReceiptHashView {
        schema_version: receipt.schema_version,
        profile: receipt.profile,
        run_inputs_catalog_sha256: &receipt.run_inputs_catalog_sha256,
        preparation_receipt_sha256: &receipt.preparation_receipt_sha256,
        cache_receipt_sha256: &receipt.cache_receipt_sha256,
        runtime_capacity_recheck: &receipt.runtime_capacity_recheck,
        device_name: &receipt.device_name,
        device_registry_id: receipt.device_registry_id,
    })?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn runtime_capacity_recheck(
    teacher: &PreparedQwen35SourceTeacherV1,
    capacity: QwenSourceMetalCapacityV1,
) -> Result<RuntimeCapacityRecheckV1> {
    validate_incremental_capacity(&teacher.receipt.runtime, teacher.upload_limits, capacity)?;
    let accounted_runtime_payload_bytes = teacher.receipt.runtime.accounted_runtime_payload_bytes;
    let unmeasured_runtime_reserve_bytes = teacher.receipt.runtime.unmeasured_runtime_reserve_bytes;
    let accounted = accounted_runtime_payload_bytes
        .checked_add(unmeasured_runtime_reserve_bytes)
        .context("source teacher runtime capacity accounting overflow")?;
    let host_required_bytes = accounted
        .checked_add(teacher.upload_limits.host_reserve_bytes)
        .context("source teacher runtime host requirement overflow")?;
    let metal_required_bytes = accounted
        .checked_add(teacher.upload_limits.metal_reserve_bytes)
        .context("source teacher runtime Metal requirement overflow")?;
    let metal_available_bytes = capacity
        .metal_recommended_working_set_bytes
        .checked_sub(capacity.metal_current_allocated_bytes)
        .context("source teacher Metal working set is already exhausted")?;
    let mut recheck = RuntimeCapacityRecheckV1 {
        profile: CAPACITY_RECHECK_PROFILE,
        capacity,
        accounted_runtime_payload_bytes,
        unmeasured_runtime_reserve_bytes,
        host_reserve_bytes: teacher.upload_limits.host_reserve_bytes,
        metal_reserve_bytes: teacher.upload_limits.metal_reserve_bytes,
        host_required_bytes,
        metal_required_bytes,
        metal_available_bytes,
        capacity_recheck_sha256: String::new(),
    };
    recheck.capacity_recheck_sha256 = capacity_recheck_sha256(&recheck)?;
    Ok(recheck)
}

fn capacity_recheck_sha256(recheck: &RuntimeCapacityRecheckV1) -> Result<String> {
    let bytes = serde_json::to_vec(&RuntimeCapacityRecheckHashView {
        profile: recheck.profile,
        capacity: recheck.capacity,
        accounted_runtime_payload_bytes: recheck.accounted_runtime_payload_bytes,
        unmeasured_runtime_reserve_bytes: recheck.unmeasured_runtime_reserve_bytes,
        host_reserve_bytes: recheck.host_reserve_bytes,
        metal_reserve_bytes: recheck.metal_reserve_bytes,
        host_required_bytes: recheck.host_required_bytes,
        metal_required_bytes: recheck.metal_required_bytes,
        metal_available_bytes: recheck.metal_available_bytes,
    })?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

#[cfg(test)]
fn prepare_run_inputs_with_for_test<PrepareTeacher, PrepareCache>(
    work: StructurallyBoundQwen35SourceTeacherWorkV1,
    output: &Path,
    preparation_policy: Qwen35SourceTeacherPreparationPolicyV1,
    prepare_teacher: PrepareTeacher,
    prepare_cache: PrepareCache,
) -> Result<PreparedQwen35SourceTeacherRunInputsV1>
where
    PrepareTeacher: FnOnce(
        VerifiedQwen35Bf16TopologyV1,
        Qwen35SourceTeacherLimitsV1,
    ) -> Result<PreparedQwen35SourceTeacherV1>,
    PrepareCache: FnOnce(
        &PreparedQwen35SourceTeacherV1,
        u32,
    ) -> Result<(PreparedQwen35BaseTextCacheV1, RuntimeCapacityRecheckV1)>,
{
    prepare_run_inputs_with(
        work,
        output,
        preparation_policy,
        prepare_teacher,
        prepare_cache,
    )
}
