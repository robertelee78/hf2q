use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::super::measured_auto_quant::{InferenceRegime, SourceIdentity};
use super::types::*;
use crate::core::provenance::tensor_execution::{
    canonicalized_tensor_execution_manifest, runtime_capability_binding_bundle_sha256,
    runtime_regime_binding_bundle_sha256, tensor_lineage_slice, verify_tensor_execution_manifest,
    RuntimeOperationBinding, TensorExecutionManifest, TensorExecutionRegime, TensorExecutionScope,
    TensorStateNode, TensorStateStage, ValidatedTensorExecutionManifest,
};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DynamicAllocationError {
    #[error("invalid allocation problem: {0}")]
    InvalidProblem(String),
    #[error("minimum executable policy needs {required} bytes, exceeding budget {budget}")]
    BudgetTooSmall { required: u64, budget: u64 },
    #[error(
        "exact Pareto frontier has {states} live states after {after_units} units and {after_options} options, limit {max_states} (generated {states_generated})"
    )]
    FrontierLimitExceeded {
        after_units: usize,
        after_options: usize,
        states: usize,
        max_states: usize,
        states_generated: u64,
    },
    #[error("integer overflow while totaling allocation evidence")]
    Overflow,
    #[error("failed to serialize canonical allocation evidence: {0}")]
    Serialize(String),
    #[error("policy frontier does not match a fresh exact proposal")]
    FrontierMismatch,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn source_valid(source: &SourceIdentity) -> bool {
    !source.model_id.is_empty()
        && !source.revision.is_empty()
        && is_sha256(&source.config_sha256)
        && is_sha256(&source.tensor_bundle_sha256)
        && is_sha256(&source.tokenizer_bundle_sha256)
        && is_sha256(&source.chat_template_sha256)
}

fn hash_serialized<T: Serialize>(value: &T) -> Result<String, DynamicAllocationError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| DynamicAllocationError::Serialize(error.to_string()))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn normalize_option(option: &mut TensorOption) {
    option
        .execution_plans
        .sort_by(|left, right| left.source_tensor_name.cmp(&right.source_tensor_name));
    option
        .operations
        .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    for operation in &mut option.operations {
        operation.source_tensor_names.sort();
        operation.executed_tensor_node_ids.sort();
        for cost in operation.regime_costs.values_mut() {
            cost.runtime_binding_ids.sort();
        }
    }
}

fn normalize_execution_scope(scope: &TensorExecutionScope) -> TensorExecutionScope {
    let mut normalized = scope.clone();
    normalized.included_paths.sort();
    normalized
}

fn normalize_problem(problem: &DynamicAllocationProblem) -> DynamicAllocationProblem {
    let mut normalized = problem.clone();
    normalized.required_regimes.sort();
    normalized.execution_scope = normalize_execution_scope(&normalized.execution_scope);
    for manifest in &mut normalized.execution_manifest_catalog {
        *manifest = canonicalized_tensor_execution_manifest(manifest);
    }
    normalized
        .execution_manifest_catalog
        .sort_by(|left, right| left.manifest_sha256.cmp(&right.manifest_sha256));
    normalized
        .units
        .sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    for unit in &mut normalized.units {
        unit.members
            .sort_by(|left, right| left.name.cmp(&right.name));
        unit.expected_expert_ids.sort();
        unit.operations
            .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
        for operation in &mut unit.operations {
            operation.tensor_names.sort();
        }
        unit.options
            .sort_by(|left, right| left.option_id.cmp(&right.option_id));
        for option in &mut unit.options {
            normalize_option(option);
        }
    }
    normalized
}

pub fn tensor_catalog_sha256(
    units: &[TensorAllocationUnit],
) -> Result<String, DynamicAllocationError> {
    let mut members: Vec<_> = units
        .iter()
        .flat_map(|unit| unit.members.iter().cloned())
        .collect();
    members.sort_by(|left, right| left.name.cmp(&right.name));
    hash_serialized(&members)
}

pub fn execution_manifest_catalog_sha256(
    manifests: &[TensorExecutionManifest],
) -> Result<String, DynamicAllocationError> {
    let mut digests = Vec::with_capacity(manifests.len());
    let mut unique = BTreeSet::new();
    for manifest in manifests {
        let validated = verify_tensor_execution_manifest(manifest).map_err(|error| {
            DynamicAllocationError::InvalidProblem(format!(
                "execution manifest is not structurally valid: {error}"
            ))
        })?;
        let digest = validated.manifest().manifest_sha256.clone();
        if !unique.insert(digest.clone()) {
            return Err(DynamicAllocationError::InvalidProblem(
                "execution manifest catalog contains duplicate manifests".into(),
            ));
        }
        digests.push(digest);
    }
    digests.sort();
    hash_serialized(&digests)
}

fn validated_manifest_catalog(
    problem: &DynamicAllocationProblem,
) -> Result<BTreeMap<String, ValidatedTensorExecutionManifest>, DynamicAllocationError> {
    let mut catalog = BTreeMap::new();
    for manifest in &problem.execution_manifest_catalog {
        let validated = verify_tensor_execution_manifest(manifest).map_err(|error| {
            DynamicAllocationError::InvalidProblem(format!(
                "execution manifest is not structurally valid: {error}"
            ))
        })?;
        let digest = validated.manifest().manifest_sha256.clone();
        if catalog.insert(digest, validated).is_some() {
            return Err(DynamicAllocationError::InvalidProblem(
                "execution manifest catalog contains duplicate manifests".into(),
            ));
        }
    }
    Ok(catalog)
}

struct ValidationContext {
    catalog: BTreeMap<String, ValidatedTensorExecutionManifest>,
}

impl ValidationContext {
    fn new(problem: &DynamicAllocationProblem) -> Result<Self, DynamicAllocationError> {
        Ok(Self {
            catalog: validated_manifest_catalog(problem)?,
        })
    }

    fn catalog_sha256(&self) -> Result<String, DynamicAllocationError> {
        hash_serialized(&self.catalog.keys().collect::<Vec<_>>())
    }
}

#[derive(Debug)]
struct DerivedOptionEvidence<'a> {
    manifest: &'a ValidatedTensorExecutionManifest,
    stored_nodes: BTreeMap<String, TensorStateNode>,
    executed_nodes: BTreeMap<String, TensorStateNode>,
    operations: BTreeMap<String, RuntimeOperationBinding>,
}

fn derive_option_evidence<'a>(
    context: &'a ValidationContext,
    option: &TensorOption,
) -> Result<DerivedOptionEvidence<'a>, DynamicAllocationError> {
    let manifests: BTreeSet<_> = option
        .execution_plans
        .iter()
        .map(|plan| plan.execution_manifest_sha256.as_str())
        .collect();
    if manifests.len() != 1 {
        return Err(DynamicAllocationError::InvalidProblem(format!(
            "option `{}` must bind exactly one physical execution manifest",
            option.option_id
        )));
    }
    let manifest_sha256 = *manifests
        .iter()
        .next()
        .ok_or_else(|| DynamicAllocationError::InvalidProblem("option has no plans".into()))?;
    let manifest = context.catalog.get(manifest_sha256).ok_or_else(|| {
        DynamicAllocationError::InvalidProblem(format!(
            "option `{}` references an absent execution manifest",
            option.option_id
        ))
    })?;
    let mut stored_nodes = BTreeMap::new();
    let mut executed_nodes = BTreeMap::new();
    let mut operations = BTreeMap::new();
    for plan in &option.execution_plans {
        if plan.source_tensor_name.is_empty()
            || !is_sha256(&plan.execution_manifest_sha256)
            || !is_sha256(&plan.lineage_slice_sha256)
        {
            return Err(DynamicAllocationError::InvalidProblem(format!(
                "option `{}` has an incomplete execution plan",
                option.option_id
            )));
        }
        let slice = tensor_lineage_slice(&manifest, &plan.source_tensor_name).map_err(|error| {
            DynamicAllocationError::InvalidProblem(format!(
                "option `{}` has an invalid lineage slice: {error}",
                option.option_id
            ))
        })?;
        if slice.slice_sha256 != plan.lineage_slice_sha256
            || slice.execution_manifest_sha256 != plan.execution_manifest_sha256
        {
            return Err(DynamicAllocationError::InvalidProblem(format!(
                "option `{}` has a stale lineage slice",
                option.option_id
            )));
        }
        for node in &slice.nodes {
            let target = match node.stage {
                TensorStateStage::Stored => Some(&mut stored_nodes),
                TensorStateStage::Executed => Some(&mut executed_nodes),
                _ => None,
            };
            if let Some(target) = target {
                if target
                    .insert(node.node_id.clone(), node.clone())
                    .is_some_and(|existing| existing != *node)
                {
                    return Err(DynamicAllocationError::InvalidProblem(format!(
                        "option `{}` has conflicting physical node aliases",
                        option.option_id
                    )));
                }
            }
        }
        for binding in &slice.operations {
            if operations
                .insert(binding.binding_id.clone(), binding.clone())
                .is_some_and(|existing| existing != *binding)
            {
                return Err(DynamicAllocationError::InvalidProblem(format!(
                    "option `{}` has conflicting runtime binding aliases",
                    option.option_id
                )));
            }
        }
    }
    if stored_nodes.is_empty() || executed_nodes.is_empty() || operations.is_empty() {
        return Err(DynamicAllocationError::InvalidProblem(format!(
            "option `{}` has no physical stored, executed, or operation evidence",
            option.option_id
        )));
    }
    Ok(DerivedOptionEvidence {
        manifest,
        stored_nodes,
        executed_nodes,
        operations,
    })
}

pub fn stored_payload_bytes(
    problem: &DynamicAllocationProblem,
    option: &TensorOption,
) -> Result<u64, DynamicAllocationError> {
    let context = ValidationContext::new(problem)?;
    derive_option_evidence(&context, option)?
        .stored_nodes
        .values()
        .try_fold(0u64, |total, node| {
            total
                .checked_add(node.byte_len)
                .ok_or(DynamicAllocationError::Overflow)
        })
}

pub fn final_executed_tensor_bundle_sha256(
    problem: &DynamicAllocationProblem,
    option: &TensorOption,
) -> Result<String, DynamicAllocationError> {
    let context = ValidationContext::new(problem)?;
    hash_serialized(&derive_option_evidence(&context, option)?.executed_nodes)
}

fn operation_costs(
    option: &TensorOption,
    required: &BTreeSet<InferenceRegime>,
) -> Result<BTreeMap<InferenceRegime, u64>, DynamicAllocationError> {
    let mut totals = BTreeMap::new();
    for operation in &option.operations {
        for regime in required {
            let cost = &operation.regime_costs[regime];
            let total = totals.entry(*regime).or_insert(0u64);
            *total = total
                .checked_add(cost.median_nanoseconds)
                .ok_or(DynamicAllocationError::Overflow)?;
        }
    }
    Ok(totals)
}

fn physical_regime(regime: InferenceRegime) -> TensorExecutionRegime {
    match regime {
        InferenceRegime::TextPrefill => TensorExecutionRegime::TextPrefill,
        InferenceRegime::TextDecodeM1 => TensorExecutionRegime::TextDecodeM1,
        InferenceRegime::TextDecodeWidthN => TensorExecutionRegime::TextDecodeWidthN,
        InferenceRegime::LongContextDecode => TensorExecutionRegime::LongContextDecode,
        InferenceRegime::MultimodalPrefill => TensorExecutionRegime::MultimodalPrefill,
    }
}

fn validate_option(
    problem: &DynamicAllocationProblem,
    context: &ValidationContext,
    unit: &TensorAllocationUnit,
    option: &TensorOption,
    required: &BTreeSet<InferenceRegime>,
) -> Result<(), DynamicAllocationError> {
    let invalid = |message: String| DynamicAllocationError::InvalidProblem(message);
    if option.option_id.is_empty()
        || option.payload_bytes == 0
        || option.capability_profile_sha256 != problem.capability_profile_sha256
        || option.sensitivity.calibration_manifest_sha256 != problem.calibration_manifest_sha256
        || !is_sha256(&option.sensitivity.sensitivity_receipt_sha256)
        || !is_sha256(&option.storage_manifest_receipt_sha256)
        || option.sensitivity.activation_rows == 0
    {
        return Err(invalid(format!(
            "unit `{}` has malformed option `{}`",
            unit.unit_id, option.option_id
        )));
    }
    let members: BTreeSet<_> = unit
        .members
        .iter()
        .map(|member| member.name.as_str())
        .collect();
    let plans: BTreeSet<_> = option
        .execution_plans
        .iter()
        .map(|plan| plan.source_tensor_name.as_str())
        .collect();
    if plans != members || option.execution_plans.len() != unit.members.len() {
        return Err(invalid(format!(
            "option `{}` does not plan every member exactly once",
            option.option_id
        )));
    }
    let derived = derive_option_evidence(context, option)?;
    let manifest_sources: BTreeSet<_> = derived
        .manifest
        .manifest()
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Source)
        .map(|node| node.semantic_name.as_str())
        .collect();
    if manifest_sources != members {
        return Err(invalid(format!(
            "option `{}` execution manifest does not exactly match its atomic unit sources",
            option.option_id
        )));
    }
    for member in &unit.members {
        let source_node = derived
            .manifest
            .manifest()
            .nodes
            .iter()
            .find(|node| {
                node.stage == TensorStateStage::Source && node.semantic_name == member.name
            })
            .expect("exact source-name equality was checked above");
        if source_node.shape
            != member
                .shape
                .iter()
                .map(|dimension| *dimension as u64)
                .collect::<Vec<_>>()
            || source_node.byte_sha256 != member.source_tensor_sha256
        {
            return Err(invalid(format!(
                "option `{}` source node does not match the tensor catalog",
                option.option_id
            )));
        }
    }
    let all_stored: BTreeSet<_> = derived
        .manifest
        .manifest()
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Stored)
        .map(|node| node.node_id.as_str())
        .collect();
    let all_executed: BTreeSet<_> = derived
        .manifest
        .manifest()
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Executed)
        .map(|node| node.node_id.as_str())
        .collect();
    if all_stored
        != derived
            .stored_nodes
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
        || all_executed
            != derived
                .executed_nodes
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
    {
        return Err(invalid(format!(
            "option `{}` does not cover every physical stored and executed node",
            option.option_id
        )));
    }
    if option.storage_manifest_receipt_sha256
        != derived.manifest.manifest().conversion_receipt_sha256
    {
        return Err(invalid(format!(
            "option `{}` does not bind the selected manifest's storage receipt",
            option.option_id
        )));
    }
    let planned_payload = derived
        .stored_nodes
        .values()
        .try_fold(0u64, |total, node| {
            total
                .checked_add(node.byte_len)
                .ok_or(DynamicAllocationError::Overflow)
        })?;
    if planned_payload != option.payload_bytes {
        return Err(invalid(format!(
            "option `{}` payload breakdown does not match its total",
            option.option_id
        )));
    }
    if hash_serialized(&derived.executed_nodes)?
        != option.sensitivity.final_executed_tensor_bundle_sha256
    {
        return Err(invalid(format!(
            "option `{}` final tensor bundle hash is stale",
            option.option_id
        )));
    }

    let mut derived_operations: BTreeMap<&str, Vec<&RuntimeOperationBinding>> = BTreeMap::new();
    for binding in derived.operations.values() {
        derived_operations
            .entry(binding.operation_id.as_str())
            .or_default()
            .push(binding);
    }
    let mut operation_ids = BTreeSet::new();
    for operation in &option.operations {
        if operation.operation_id.is_empty()
            || operation.graph_path.is_empty()
            || !operation_ids.insert(operation.operation_id.as_str())
            || !is_sha256(&operation.capability_binding_bundle_sha256)
        {
            return Err(invalid(format!(
                "option `{}` has malformed operation evidence",
                option.option_id
            )));
        }
        let bindings = derived_operations
            .get(operation.operation_id.as_str())
            .ok_or_else(|| {
                invalid(format!(
                    "operation `{}` is absent from the selected physical manifest",
                    operation.operation_id
                ))
            })?;
        let physical_regimes: BTreeSet<_> = bindings
            .iter()
            .map(|binding| match binding.workload_regime {
                crate::core::provenance::tensor_execution::TensorExecutionRegime::TextPrefill => {
                    InferenceRegime::TextPrefill
                }
                crate::core::provenance::tensor_execution::TensorExecutionRegime::TextDecodeM1 => {
                    InferenceRegime::TextDecodeM1
                }
                crate::core::provenance::tensor_execution::TensorExecutionRegime::TextDecodeWidthN => {
                    InferenceRegime::TextDecodeWidthN
                }
                crate::core::provenance::tensor_execution::TensorExecutionRegime::LongContextDecode => {
                    InferenceRegime::LongContextDecode
                }
                crate::core::provenance::tensor_execution::TensorExecutionRegime::MultimodalPrefill => {
                    InferenceRegime::MultimodalPrefill
                }
            })
            .collect();
        let expected_sources: BTreeSet<_> = bindings
            .iter()
            .flat_map(|binding| binding.source_tensor_names.iter().map(String::as_str))
            .collect();
        let actual_sources: BTreeSet<_> = operation
            .source_tensor_names
            .iter()
            .map(String::as_str)
            .collect();
        let expected_executed: BTreeSet<_> = bindings
            .iter()
            .flat_map(|binding| binding.inputs.iter().map(|input| input.node_id.as_str()))
            .collect();
        let actual_executed: BTreeSet<_> = operation
            .executed_tensor_node_ids
            .iter()
            .map(String::as_str)
            .collect();
        if expected_sources.is_empty()
            || expected_sources != actual_sources
            || actual_sources.len() != operation.source_tensor_names.len()
            || expected_executed != actual_executed
            || actual_executed.len() != operation.executed_tensor_node_ids.len()
            || bindings
                .iter()
                .any(|binding| binding.graph_path != operation.graph_path)
            || physical_regimes != *required
            || runtime_capability_binding_bundle_sha256(&derived.manifest, &operation.operation_id)
                .map_err(|error| invalid(error.to_string()))?
                != operation.capability_binding_bundle_sha256
        {
            return Err(invalid(format!(
                "operation `{}` does not exactly cover its source and executed tensors",
                operation.operation_id
            )));
        }
        let regimes: BTreeSet<_> = operation.regime_costs.keys().copied().collect();
        if regimes != *required {
            return Err(invalid(format!(
                "operation `{}` has missing or unexpected regimes",
                operation.operation_id
            )));
        }
        for (regime, cost) in &operation.regime_costs {
            let expected_binding_ids: BTreeSet<_> = bindings
                .iter()
                .filter(|binding| binding.workload_regime == physical_regime(*regime))
                .map(|binding| binding.binding_id.as_str())
                .collect();
            let actual_binding_ids: BTreeSet<_> = cost
                .runtime_binding_ids
                .iter()
                .map(String::as_str)
                .collect();
            if cost.regime != *regime
                || !cost.executable
                || expected_binding_ids.is_empty()
                || expected_binding_ids != actual_binding_ids
                || actual_binding_ids.len() != cost.runtime_binding_ids.len()
                || runtime_regime_binding_bundle_sha256(
                    &derived.manifest,
                    &operation.operation_id,
                    physical_regime(*regime),
                )
                .map_err(|error| invalid(error.to_string()))?
                    != cost.runtime_binding_bundle_sha256
                || cost.median_nanoseconds == 0
                || cost.p95_nanoseconds < cost.median_nanoseconds
                || cost.warmup_runs == 0
                || cost.measured_runs < 3
                || !is_sha256(&cost.runtime_binding_bundle_sha256)
                || !is_sha256(&cost.measurement_receipt_sha256)
            {
                return Err(invalid(format!(
                    "operation `{}` has ineligible {:?} evidence",
                    operation.operation_id, regime
                )));
            }
        }
    }
    if option.operations.len() != operation_ids.len()
        || derived_operations
            .keys()
            .any(|operation_id| !operation_ids.contains(operation_id))
    {
        return Err(invalid(format!(
            "option `{}` has an unbound operation",
            option.option_id
        )));
    }

    let expected_experts: BTreeSet<_> = unit.expected_expert_ids.iter().copied().collect();
    let observed_experts: BTreeSet<_> = option
        .sensitivity
        .expert_activation_rows
        .keys()
        .copied()
        .collect();
    if expected_experts != observed_experts
        || observed_experts.iter().any(|expert| {
            option.sensitivity.expert_activation_rows[expert]
                < problem.minimum_expert_activation_rows
        })
    {
        return Err(invalid(format!(
            "option `{}` has insufficient per-expert coverage",
            option.option_id
        )));
    }
    operation_costs(option, required)?;
    Ok(())
}

fn validate_problem(problem: &DynamicAllocationProblem) -> Result<(), DynamicAllocationError> {
    let invalid = |message: &str| DynamicAllocationError::InvalidProblem(message.into());
    if problem.schema_version != DYNAMIC_ALLOCATION_SCHEMA_VERSION {
        return Err(invalid("allocation schema version mismatch"));
    }
    if !source_valid(&problem.source)
        || problem.execution.hf2q_revision.is_empty()
        || problem.execution.mlx_native_version.is_empty()
        || problem.execution.hardware_id.is_empty()
        || problem.execution.os_build.is_empty()
        || problem.tensor_runtime.hf2q_revision != problem.execution.hf2q_revision
        || problem.tensor_runtime.mlx_native_version != problem.execution.mlx_native_version
        || problem.tensor_runtime.capability_profile_sha256 != problem.capability_profile_sha256
        || problem.tensor_runtime.mlx_native_capability_schema_version != 1
        || problem.tensor_runtime.dwq_overlay_sha256.is_some()
        || problem.execution_scope.model_family.is_empty()
        || problem.execution_scope.profile.is_empty()
        || problem.execution_scope.included_paths.is_empty()
        || !is_sha256(&problem.tensor_catalog_sha256)
        || !is_sha256(&problem.dataset_partition_manifest_sha256)
        || !is_sha256(&problem.tensor_partition_manifest_sha256)
        || !is_sha256(&problem.execution_manifest_catalog_sha256)
        || !is_sha256(&problem.calibration_manifest_sha256)
        || !is_sha256(&problem.capability_profile_sha256)
        || !is_sha256(&problem.proposal_workload_profile_sha256)
        || problem.sensitivity_model.method.is_empty()
        || problem.sensitivity_model.version.is_empty()
        || problem.sensitivity_model.fixed_point_scale == 0
        || !is_sha256(&problem.sensitivity_model.component_weights_sha256)
        || !is_sha256(&problem.sensitivity_model.coverage_contract_sha256)
        || !is_sha256(&problem.sensitivity_model.coverage_receipt_sha256)
    {
        return Err(invalid(
            "source, calibration, runtime, or workload identity is incomplete",
        ));
    }
    let context = ValidationContext::new(problem)?;
    let execution_manifests = &context.catalog;
    if execution_manifests.is_empty()
        || context.catalog_sha256()? != problem.execution_manifest_catalog_sha256
    {
        return Err(invalid(
            "execution manifest catalog is empty, duplicated, or stale",
        ));
    }
    if execution_manifests.values().any(|validated| {
        let manifest = validated.manifest();
        manifest.tensor_partition_manifest_sha256 != problem.tensor_partition_manifest_sha256
            || manifest.runtime != problem.tensor_runtime
            || normalize_execution_scope(&manifest.scope)
                != normalize_execution_scope(&problem.execution_scope)
    }) {
        return Err(invalid(
            "execution manifests do not match the allocation partition or runtime identity",
        ));
    }
    let SearchContract::ExactPareto { max_states } = problem.search;
    if max_states == 0 || problem.variable_payload_budget_bytes == 0 || problem.units.is_empty() {
        return Err(invalid("search bound, budget, and units must be non-zero"));
    }
    let required: BTreeSet<_> = problem.required_regimes.iter().copied().collect();
    if required.is_empty() || required.len() != problem.required_regimes.len() {
        return Err(invalid("required regimes must be non-empty and unique"));
    }

    let mut unit_ids = BTreeSet::new();
    let mut tensor_names = BTreeSet::new();
    let mut global_operation_ids = BTreeSet::new();
    let mut referenced_execution_manifests = BTreeMap::new();
    let mut tensor_count = 0usize;
    for unit in &problem.units {
        if unit.unit_id.is_empty()
            || !unit_ids.insert(unit.unit_id.as_str())
            || unit.members.is_empty()
            || unit.options.is_empty()
        {
            return Err(invalid(
                "allocation units require unique ids, members, and options",
            ));
        }
        let expected_experts: BTreeSet<_> = unit.expected_expert_ids.iter().copied().collect();
        if expected_experts.len() != unit.expected_expert_ids.len()
            || (!expected_experts.is_empty() && problem.minimum_expert_activation_rows == 0)
            || unit.members.iter().any(|member| {
                member
                    .expert_index
                    .is_some_and(|expert| !expected_experts.contains(&expert))
            })
        {
            return Err(invalid(
                "expert ids must be unique, explicitly declared, and have positive coverage",
            ));
        }
        let mut member_names = BTreeSet::new();
        for member in &unit.members {
            if member.name.is_empty()
                || member.role.is_empty()
                || member.shape.is_empty()
                || member.shape.contains(&0)
                || !is_sha256(&member.source_tensor_sha256)
                || !member_names.insert(member.name.as_str())
                || !tensor_names.insert(member.name.as_str())
            {
                return Err(invalid("tensor catalog has an invalid or duplicate member"));
            }
            tensor_count = tensor_count
                .checked_add(1)
                .ok_or(DynamicAllocationError::Overflow)?;
        }
        let mut operation_covered = BTreeSet::new();
        let mut canonical_topology = BTreeMap::new();
        for operation in &unit.operations {
            let operation_tensors: BTreeSet<_> =
                operation.tensor_names.iter().map(String::as_str).collect();
            if operation.operation_id.is_empty()
                || operation.graph_path.is_empty()
                || !global_operation_ids.insert(operation.operation_id.as_str())
                || operation_tensors.is_empty()
                || operation_tensors.len() != operation.tensor_names.len()
                || operation_tensors
                    .iter()
                    .any(|name| !member_names.contains(name))
            {
                return Err(invalid(
                    "logical operations must be unique, non-empty, and cover only unit members",
                ));
            }
            operation_covered.extend(operation_tensors.iter().copied());
            canonical_topology.insert(
                operation.operation_id.as_str(),
                (operation.graph_path.as_str(), operation_tensors),
            );
        }
        if operation_covered != member_names {
            return Err(invalid(
                "logical operations must collectively cover every unit member",
            ));
        }
        let mut option_ids = BTreeSet::new();
        for option in &unit.options {
            if !option_ids.insert(option.option_id.as_str()) {
                return Err(invalid("allocation unit has duplicate option ids"));
            }
            validate_option(problem, &context, unit, option, &required)?;
            let manifest_sha256 = option.execution_plans[0].execution_manifest_sha256.as_str();
            *referenced_execution_manifests
                .entry(manifest_sha256)
                .or_insert(0usize) += 1;
            let topology: BTreeMap<_, _> = option
                .operations
                .iter()
                .map(|operation| {
                    (
                        operation.operation_id.as_str(),
                        (
                            operation.graph_path.as_str(),
                            operation
                                .source_tensor_names
                                .iter()
                                .map(String::as_str)
                                .collect(),
                        ),
                    )
                })
                .collect();
            if topology != canonical_topology {
                return Err(invalid(
                    "every option must match its unit's canonical logical operation topology",
                ));
            }
        }
    }
    if tensor_count != problem.expected_tensor_count
        || tensor_catalog_sha256(&problem.units)? != problem.tensor_catalog_sha256
    {
        return Err(invalid("tensor catalog count or canonical hash mismatch"));
    }
    if referenced_execution_manifests
        .keys()
        .copied()
        .collect::<BTreeSet<_>>()
        != execution_manifests
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
        || referenced_execution_manifests
            .values()
            .any(|count| *count != 1)
    {
        return Err(invalid(
            "every execution manifest must be referenced and no option may reference an extra manifest",
        ));
    }
    Ok(())
}

pub fn allocation_problem_sha256(
    problem: &DynamicAllocationProblem,
) -> Result<String, DynamicAllocationError> {
    validate_problem(problem)?;
    hash_serialized(&normalize_problem(problem))
}

#[derive(Clone, Debug)]
struct State {
    bytes: u64,
    loss: u64,
    costs: Vec<u64>,
    choices: Vec<usize>,
}

fn metrics_equal(left: &State, right: &State) -> bool {
    left.bytes == right.bytes && left.loss == right.loss && left.costs == right.costs
}

fn dominates(left: &State, right: &State) -> bool {
    let all_no_worse = left.bytes <= right.bytes
        && left.loss <= right.loss
        && left.costs.iter().zip(&right.costs).all(|(a, b)| a <= b);
    let any_better = left.bytes < right.bytes
        || left.loss < right.loss
        || left.costs.iter().zip(&right.costs).any(|(a, b)| a < b);
    all_no_worse && any_better
}

fn insert_nondominated(
    frontier: &mut Vec<State>,
    candidate: State,
    dominated: &mut u64,
    equivalent: &mut u64,
) {
    for existing in frontier.iter_mut() {
        if dominates(existing, &candidate) {
            *dominated += 1;
            return;
        }
        if metrics_equal(existing, &candidate) {
            *equivalent += 1;
            if candidate.choices < existing.choices {
                *existing = candidate;
            }
            return;
        }
    }
    frontier.retain(|existing| {
        if dominates(&candidate, existing) {
            *dominated += 1;
            false
        } else {
            true
        }
    });
    frontier.push(candidate);
}

/// Produce one deterministic representative for each nondominated proxy-metric
/// vector under the payload budget. Equal vectors with different tensor
/// assignments are counted in the receipt and collapsed; full-model repair is
/// responsible for exploring interaction effects. The live-state bound is
/// fail-closed and never an implicit heuristic.
pub(super) fn allocate_dynamic_frontier(
    problem: &DynamicAllocationProblem,
) -> Result<PolicyFrontier, DynamicAllocationError> {
    validate_problem(problem)?;
    let normalized = normalize_problem(problem);
    let problem_sha256 = hash_serialized(&normalized)?;
    let regimes = normalized.required_regimes.clone();
    let required: BTreeSet<_> = regimes.iter().copied().collect();
    let SearchContract::ExactPareto { max_states } = normalized.search;

    let minimum = normalized.units.iter().try_fold(0u64, |total, unit| {
        let bytes = unit
            .options
            .iter()
            .map(|option| option.payload_bytes)
            .min()
            .unwrap();
        total
            .checked_add(bytes)
            .ok_or(DynamicAllocationError::Overflow)
    })?;
    if minimum > normalized.variable_payload_budget_bytes {
        return Err(DynamicAllocationError::BudgetTooSmall {
            required: minimum,
            budget: normalized.variable_payload_budget_bytes,
        });
    }

    let mut frontier = vec![State {
        bytes: 0,
        loss: 0,
        costs: vec![0; regimes.len()],
        choices: Vec::new(),
    }];
    let mut generated = 0u64;
    let mut dominated = 0u64;
    let mut equivalent = 0u64;
    let mut peak = frontier.len();
    for (unit_index, unit) in normalized.units.iter().enumerate() {
        let mut next = Vec::new();
        for state in &frontier {
            for (option_index, option) in unit.options.iter().enumerate() {
                generated = generated
                    .checked_add(1)
                    .ok_or(DynamicAllocationError::Overflow)?;
                let bytes = state
                    .bytes
                    .checked_add(option.payload_bytes)
                    .ok_or(DynamicAllocationError::Overflow)?;
                if bytes > normalized.variable_payload_budget_bytes {
                    continue;
                }
                let option_costs = operation_costs(option, &required)?;
                let mut costs = Vec::with_capacity(regimes.len());
                for (index, regime) in regimes.iter().enumerate() {
                    costs.push(
                        state.costs[index]
                            .checked_add(option_costs[regime])
                            .ok_or(DynamicAllocationError::Overflow)?,
                    );
                }
                let mut choices = state.choices.clone();
                choices.push(option_index);
                insert_nondominated(
                    &mut next,
                    State {
                        bytes,
                        loss: state
                            .loss
                            .checked_add(option.sensitivity.loss_units)
                            .ok_or(DynamicAllocationError::Overflow)?,
                        costs,
                        choices,
                    },
                    &mut dominated,
                    &mut equivalent,
                );
                peak = peak.max(next.len());
                if next.len() > max_states {
                    return Err(DynamicAllocationError::FrontierLimitExceeded {
                        after_units: unit_index + 1,
                        after_options: option_index + 1,
                        states: next.len(),
                        max_states,
                        states_generated: generated,
                    });
                }
            }
        }
        next.sort_by(|left, right| {
            left.bytes
                .cmp(&right.bytes)
                .then_with(|| left.loss.cmp(&right.loss))
                .then_with(|| left.costs.cmp(&right.costs))
                .then_with(|| left.choices.cmp(&right.choices))
        });
        frontier = next;
    }

    let mut policies = Vec::with_capacity(frontier.len());
    for state in frontier {
        let mut decisions = Vec::with_capacity(normalized.units.len());
        for (unit_index, unit) in normalized.units.iter().enumerate() {
            let option = unit.options[state.choices[unit_index]].clone();
            decisions.push(UnitDecision {
                unit_id: unit.unit_id.clone(),
                regime_cost_nanoseconds: operation_costs(&option, &required)?,
                selected_option: option,
            });
        }
        policies.push(PrecisionPolicyManifest {
            schema_version: DYNAMIC_ALLOCATION_SCHEMA_VERSION,
            allocation_problem_sha256: problem_sha256.clone(),
            source: normalized.source.clone(),
            execution: normalized.execution.clone(),
            tensor_runtime: normalized.tensor_runtime.clone(),
            execution_scope: normalized.execution_scope.clone(),
            tensor_catalog_sha256: normalized.tensor_catalog_sha256.clone(),
            dataset_partition_manifest_sha256: normalized.dataset_partition_manifest_sha256.clone(),
            tensor_partition_manifest_sha256: normalized.tensor_partition_manifest_sha256.clone(),
            execution_manifest_catalog_sha256: normalized.execution_manifest_catalog_sha256.clone(),
            calibration_manifest_sha256: normalized.calibration_manifest_sha256.clone(),
            capability_profile_sha256: normalized.capability_profile_sha256.clone(),
            proposal_workload_profile_sha256: normalized.proposal_workload_profile_sha256.clone(),
            variable_payload_budget_bytes: normalized.variable_payload_budget_bytes,
            total_variable_payload_bytes: state.bytes,
            total_loss_units: state.loss,
            total_regime_cost_nanoseconds: regimes
                .iter()
                .copied()
                .zip(state.costs.into_iter())
                .collect(),
            decisions,
        });
    }
    policies.sort_by(|left, right| {
        left.total_variable_payload_bytes
            .cmp(&right.total_variable_payload_bytes)
            .then_with(|| left.total_loss_units.cmp(&right.total_loss_units))
            .then_with(|| {
                left.total_regime_cost_nanoseconds
                    .cmp(&right.total_regime_cost_nanoseconds)
            })
            .then_with(|| {
                canonical_policy_bytes(left)
                    .unwrap()
                    .cmp(&canonical_policy_bytes(right).unwrap())
            })
    });
    Ok(PolicyFrontier {
        schema_version: DYNAMIC_ALLOCATION_SCHEMA_VERSION,
        allocation_problem_sha256: problem_sha256,
        search_receipt: SearchReceipt {
            algorithm: "exact-multi-choice-pareto-dp-v1".into(),
            exhaustive_within_proxy_metrics: true,
            state_limit: max_states,
            states_generated: generated,
            states_pruned_dominated: dominated,
            equivalent_states_collapsed: equivalent,
            peak_frontier_states: peak,
            frontier_size: policies.len(),
        },
        policies,
    })
}

pub fn canonical_policy_bytes(
    policy: &PrecisionPolicyManifest,
) -> Result<Vec<u8>, DynamicAllocationError> {
    let mut normalized = policy.clone();
    normalized.execution_scope = normalize_execution_scope(&normalized.execution_scope);
    normalized
        .decisions
        .sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    for decision in &mut normalized.decisions {
        normalize_option(&mut decision.selected_option);
    }
    serde_json::to_vec(&normalized)
        .map_err(|error| DynamicAllocationError::Serialize(error.to_string()))
}

pub fn precision_policy_sha256(
    policy: &PrecisionPolicyManifest,
) -> Result<String, DynamicAllocationError> {
    Ok(hex::encode(Sha256::digest(canonical_policy_bytes(policy)?)))
}

pub fn canonical_frontier_bytes(
    frontier: &PolicyFrontier,
) -> Result<Vec<u8>, DynamicAllocationError> {
    let mut normalized = frontier.clone();
    for policy in &mut normalized.policies {
        policy.execution_scope = normalize_execution_scope(&policy.execution_scope);
        policy
            .decisions
            .sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
        for decision in &mut policy.decisions {
            normalize_option(&mut decision.selected_option);
        }
    }
    normalized.policies.sort_by(|left, right| {
        left.total_variable_payload_bytes
            .cmp(&right.total_variable_payload_bytes)
            .then_with(|| left.total_loss_units.cmp(&right.total_loss_units))
            .then_with(|| {
                left.total_regime_cost_nanoseconds
                    .cmp(&right.total_regime_cost_nanoseconds)
            })
            .then_with(|| {
                let left_ids: Vec<_> = left
                    .decisions
                    .iter()
                    .map(|decision| (&decision.unit_id, &decision.selected_option.option_id))
                    .collect();
                let right_ids: Vec<_> = right
                    .decisions
                    .iter()
                    .map(|decision| (&decision.unit_id, &decision.selected_option.option_id))
                    .collect();
                left_ids.cmp(&right_ids)
            })
    });
    serde_json::to_vec(&normalized)
        .map_err(|error| DynamicAllocationError::Serialize(error.to_string()))
}

/// Independently reproduce the exact frontier and reject any mutated or
/// stale policy/search receipt.
#[cfg(test)]
pub(super) fn validate_policy_frontier(
    problem: &DynamicAllocationProblem,
    frontier: &PolicyFrontier,
) -> Result<(), DynamicAllocationError> {
    let expected = allocate_dynamic_frontier(problem)?;
    if canonical_frontier_bytes(&expected)? == canonical_frontier_bytes(frontier)? {
        Ok(())
    } else {
        Err(DynamicAllocationError::FrontierMismatch)
    }
}
