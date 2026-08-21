use anyhow::{ensure, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::core::provenance::tensor_execution::ArtifactEvidence;
use crate::inference::models::qwen35::source_precision::teacher_execution_plan::Qwen35SourceTeacherExpectedWorkV1;
use crate::intelligence::exact_teacher::ExactTeacherTargetReceipt;

use super::super::PreparedQwen35SourceTeacherRunInputsReceiptV1;

const COMPLETION_SCHEMA_VERSION: u32 = 1;
const COMPLETION_PROFILE: &str = "dense_qwen35_source_bf16_completed_teacher_v1";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub(super) struct Qwen35SourceTeacherObservedWorkV1 {
    pub(super) example_count: usize,
    pub(super) completed_transcript_count: usize,
    pub(super) generation_prompt_count: usize,
    pub(super) prediction_row_count: usize,
    pub(super) trajectory_count: usize,
    pub(super) cache_reset_count: usize,
    pub(super) forward_call_count: u64,
    pub(super) input_tokens_processed: u64,
    pub(super) output_head_evaluation_count: u64,
    pub(super) terminal_call_completion_count: u64,
    pub(super) max_cache_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(super) struct Qwen35SourceTeacherCompletionReceiptV1 {
    schema_version: u32,
    profile: &'static str,
    run_inputs_catalog_sha256: String,
    run_inputs_receipt_sha256: String,
    work_plan_sha256: String,
    topology_sha256: String,
    prediction_plan_sha256: String,
    projected_execution_config_sha256: String,
    prepared_graph_catalog_sha256: String,
    preparation_receipt_sha256: String,
    cache_layout_sha256: String,
    cache_receipt_sha256: String,
    target_reservation_contract_sha256: String,
    source_snapshot_catalog_sha256: String,
    source_graph_policy_sha256: String,
    completed_call_schedule_sha256: String,
    device_name: String,
    device_registry_id: u64,
    expected_work: Qwen35SourceTeacherExpectedWorkV1,
    observed_work: Qwen35SourceTeacherObservedWorkV1,
    structural_target_receipt_sha256: String,
    target_artifact: ArtifactEvidence,
    weight_precision: &'static str,
    cache_precision: &'static str,
    base_text_only: bool,
    checked_synchronous_source_commits: bool,
    source_inodes_reverified_after_execution: bool,
    q4_repack: bool,
    ggml: bool,
    dwq: bool,
    tq: bool,
    mtp_executed: bool,
    vision_executed: bool,
    target_published: bool,
    complete_native_route_bound: bool,
    runtime_peak_proven: bool,
    performance_authority: bool,
    sensitivity_authority: bool,
    allocator_authority: bool,
    selector_authority: bool,
    autoquant_authority: bool,
    completion_receipt_sha256: String,
}

#[derive(Serialize)]
struct CompletionHashView<'a> {
    schema_version: u32,
    profile: &'static str,
    run_inputs_catalog_sha256: &'a str,
    run_inputs_receipt_sha256: &'a str,
    work_plan_sha256: &'a str,
    topology_sha256: &'a str,
    prediction_plan_sha256: &'a str,
    projected_execution_config_sha256: &'a str,
    prepared_graph_catalog_sha256: &'a str,
    preparation_receipt_sha256: &'a str,
    cache_layout_sha256: &'a str,
    cache_receipt_sha256: &'a str,
    target_reservation_contract_sha256: &'a str,
    source_snapshot_catalog_sha256: &'a str,
    source_graph_policy_sha256: &'a str,
    completed_call_schedule_sha256: &'a str,
    device_name: &'a str,
    device_registry_id: u64,
    expected_work: Qwen35SourceTeacherExpectedWorkV1,
    observed_work: Qwen35SourceTeacherObservedWorkV1,
    structural_target_receipt_sha256: &'a str,
    target_artifact: &'a ArtifactEvidence,
    weight_precision: &'static str,
    cache_precision: &'static str,
    base_text_only: bool,
    checked_synchronous_source_commits: bool,
    source_inodes_reverified_after_execution: bool,
    q4_repack: bool,
    ggml: bool,
    dwq: bool,
    tq: bool,
    mtp_executed: bool,
    vision_executed: bool,
    target_published: bool,
    complete_native_route_bound: bool,
    runtime_peak_proven: bool,
    performance_authority: bool,
    sensitivity_authority: bool,
    allocator_authority: bool,
    selector_authority: bool,
    autoquant_authority: bool,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_completion_receipt(
    predecessor: &PreparedQwen35SourceTeacherRunInputsReceiptV1,
    source_snapshot_catalog_sha256: String,
    source_graph_policy_sha256: String,
    completed_call_schedule_sha256: String,
    observed_work: Qwen35SourceTeacherObservedWorkV1,
    target: &ExactTeacherTargetReceipt,
) -> Result<Qwen35SourceTeacherCompletionReceiptV1> {
    validate_observed(predecessor.expected_work, observed_work)?;
    ensure!(
        target.prediction_plan_sha256 == predecessor.prediction_plan_sha256
            && target.prediction_point_count == observed_work.prediction_row_count
            && target.generation_prompt_count == observed_work.trajectory_count
            && target.target_artifact.byte_len == predecessor.expected_work.target_artifact_bytes,
        "completed structural target differs from source-teacher work"
    );
    let mut receipt = Qwen35SourceTeacherCompletionReceiptV1 {
        schema_version: COMPLETION_SCHEMA_VERSION,
        profile: COMPLETION_PROFILE,
        run_inputs_catalog_sha256: predecessor.run_inputs_catalog_sha256.clone(),
        run_inputs_receipt_sha256: predecessor.run_inputs_receipt_sha256.clone(),
        work_plan_sha256: predecessor.work_plan_sha256.clone(),
        topology_sha256: predecessor.topology_sha256.clone(),
        prediction_plan_sha256: predecessor.prediction_plan_sha256.clone(),
        projected_execution_config_sha256: predecessor.projected_execution_config_sha256.clone(),
        prepared_graph_catalog_sha256: predecessor.prepared_graph_catalog_sha256.clone(),
        preparation_receipt_sha256: predecessor.preparation_receipt_sha256.clone(),
        cache_layout_sha256: predecessor.cache_layout_sha256.clone(),
        cache_receipt_sha256: predecessor.cache_receipt_sha256.clone(),
        target_reservation_contract_sha256: predecessor.target_reservation_contract_sha256.clone(),
        source_snapshot_catalog_sha256,
        source_graph_policy_sha256,
        completed_call_schedule_sha256,
        device_name: predecessor.device_name.clone(),
        device_registry_id: predecessor.device_registry_id,
        expected_work: predecessor.expected_work,
        observed_work,
        structural_target_receipt_sha256: target.receipt_sha256.clone(),
        target_artifact: target.target_artifact.clone(),
        weight_precision: "source_bf16_controls_f32",
        cache_precision: "base_text_f32_one_sequence",
        base_text_only: true,
        checked_synchronous_source_commits: true,
        source_inodes_reverified_after_execution: true,
        q4_repack: false,
        ggml: false,
        dwq: false,
        tq: false,
        mtp_executed: false,
        vision_executed: false,
        target_published: true,
        complete_native_route_bound: false,
        runtime_peak_proven: false,
        performance_authority: false,
        sensitivity_authority: false,
        allocator_authority: false,
        selector_authority: false,
        autoquant_authority: false,
        completion_receipt_sha256: String::new(),
    };
    receipt.completion_receipt_sha256 = completion_receipt_sha256(&receipt)?;
    validate_completion_receipt(&receipt)?;
    Ok(receipt)
}

pub(super) fn validate_completion_receipt(
    receipt: &Qwen35SourceTeacherCompletionReceiptV1,
) -> Result<()> {
    validate_observed(receipt.expected_work, receipt.observed_work)?;
    ensure!(
        receipt.schema_version == COMPLETION_SCHEMA_VERSION
            && receipt.profile == COMPLETION_PROFILE
            && receipt.weight_precision == "source_bf16_controls_f32"
            && receipt.cache_precision == "base_text_f32_one_sequence"
            && receipt.base_text_only
            && receipt.checked_synchronous_source_commits
            && receipt.source_inodes_reverified_after_execution
            && receipt.target_published
            && !receipt.q4_repack
            && !receipt.ggml
            && !receipt.dwq
            && !receipt.tq
            && !receipt.mtp_executed
            && !receipt.vision_executed
            && !receipt.complete_native_route_bound
            && !receipt.runtime_peak_proven
            && !receipt.performance_authority
            && !receipt.sensitivity_authority
            && !receipt.allocator_authority
            && !receipt.selector_authority
            && !receipt.autoquant_authority
            && receipt.completion_receipt_sha256 == completion_receipt_sha256(receipt)?,
        "source-teacher completion receipt does not reproduce"
    );
    Ok(())
}

fn validate_observed(
    expected: Qwen35SourceTeacherExpectedWorkV1,
    observed: Qwen35SourceTeacherObservedWorkV1,
) -> Result<()> {
    ensure!(
        observed.example_count == expected.example_count
            && observed.completed_transcript_count == expected.completed_transcript_count
            && observed.generation_prompt_count == expected.generation_prompt_count
            && observed.prediction_row_count == expected.prediction_row_count
            && observed.trajectory_count == expected.generation_prompt_count
            && observed.cache_reset_count == expected.example_count
            && observed.forward_call_count == expected.forward_call_count
            && observed.input_tokens_processed == expected.input_tokens_processed
            && observed.output_head_evaluation_count == expected.output_head_evaluation_count
            && observed.terminal_call_completion_count == expected.forward_call_count
            && observed.max_cache_tokens == expected.max_cache_tokens
            && expected.greedy_token_count_per_prompt == 32,
        "observed source-teacher work differs from its preflight"
    );
    Ok(())
}

fn completion_receipt_sha256(receipt: &Qwen35SourceTeacherCompletionReceiptV1) -> Result<String> {
    let view = CompletionHashView {
        schema_version: receipt.schema_version,
        profile: receipt.profile,
        run_inputs_catalog_sha256: &receipt.run_inputs_catalog_sha256,
        run_inputs_receipt_sha256: &receipt.run_inputs_receipt_sha256,
        work_plan_sha256: &receipt.work_plan_sha256,
        topology_sha256: &receipt.topology_sha256,
        prediction_plan_sha256: &receipt.prediction_plan_sha256,
        projected_execution_config_sha256: &receipt.projected_execution_config_sha256,
        prepared_graph_catalog_sha256: &receipt.prepared_graph_catalog_sha256,
        preparation_receipt_sha256: &receipt.preparation_receipt_sha256,
        cache_layout_sha256: &receipt.cache_layout_sha256,
        cache_receipt_sha256: &receipt.cache_receipt_sha256,
        target_reservation_contract_sha256: &receipt.target_reservation_contract_sha256,
        source_snapshot_catalog_sha256: &receipt.source_snapshot_catalog_sha256,
        source_graph_policy_sha256: &receipt.source_graph_policy_sha256,
        completed_call_schedule_sha256: &receipt.completed_call_schedule_sha256,
        device_name: &receipt.device_name,
        device_registry_id: receipt.device_registry_id,
        expected_work: receipt.expected_work,
        observed_work: receipt.observed_work,
        structural_target_receipt_sha256: &receipt.structural_target_receipt_sha256,
        target_artifact: &receipt.target_artifact,
        weight_precision: receipt.weight_precision,
        cache_precision: receipt.cache_precision,
        base_text_only: receipt.base_text_only,
        checked_synchronous_source_commits: receipt.checked_synchronous_source_commits,
        source_inodes_reverified_after_execution: receipt.source_inodes_reverified_after_execution,
        q4_repack: receipt.q4_repack,
        ggml: receipt.ggml,
        dwq: receipt.dwq,
        tq: receipt.tq,
        mtp_executed: receipt.mtp_executed,
        vision_executed: receipt.vision_executed,
        target_published: receipt.target_published,
        complete_native_route_bound: receipt.complete_native_route_bound,
        runtime_peak_proven: receipt.runtime_peak_proven,
        performance_authority: receipt.performance_authority,
        sensitivity_authority: receipt.sensitivity_authority,
        allocator_authority: receipt.allocator_authority,
        selector_authority: receipt.selector_authority,
        autoquant_authority: receipt.autoquant_authority,
    };
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&view)?)))
}

impl Qwen35SourceTeacherCompletionReceiptV1 {
    pub(super) fn receipt_sha256(&self) -> &str {
        &self.completion_receipt_sha256
    }

    #[cfg(test)]
    pub(super) fn work_for_test(
        &self,
    ) -> (
        Qwen35SourceTeacherExpectedWorkV1,
        Qwen35SourceTeacherObservedWorkV1,
    ) {
        (self.expected_work, self.observed_work)
    }

    #[cfg(test)]
    pub(super) fn corrupt_graph_policy_for_test(&mut self) {
        let replacement = if self.source_graph_policy_sha256.starts_with('0') {
            "1"
        } else {
            "0"
        };
        self.source_graph_policy_sha256
            .replace_range(..1, replacement);
    }

    #[cfg(test)]
    pub(super) fn decrement_completion_for_test(&mut self) {
        self.observed_work.terminal_call_completion_count -= 1;
    }
}
