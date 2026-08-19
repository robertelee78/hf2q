use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::super::measured_auto_quant::{InferenceRegime, SourceIdentity};
use super::types::*;

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

fn codec_valid(codec: &TensorCodec) -> bool {
    match codec {
        TensorCodec::Gguf { .. } | TensorCodec::Dense { .. } => true,
        TensorCodec::MlxAffine {
            bits,
            group_size,
            layout_abi,
            ..
        } => matches!(bits, 4 | 6 | 8) && matches!(group_size, 32 | 64) && !layout_abi.is_empty(),
    }
}

fn lossless_repack_compatible(from: &TensorCodec, to: &TensorCodec) -> bool {
    match (from, to) {
        (TensorCodec::Gguf { codec: left }, TensorCodec::Gguf { codec: right }) => left == right,
        (
            TensorCodec::MlxAffine {
                bits: left_bits,
                group_size: left_group,
                scale_dtype: left_scale,
                bias_dtype: left_bias,
                ..
            },
            TensorCodec::MlxAffine {
                bits: right_bits,
                group_size: right_group,
                scale_dtype: right_scale,
                bias_dtype: right_bias,
                ..
            },
        ) => {
            left_bits == right_bits
                && left_group == right_group
                && left_scale == right_scale
                && left_bias == right_bias
        }
        (TensorCodec::Dense { dtype: left }, TensorCodec::Dense { dtype: right }) => left == right,
        _ => false,
    }
}

fn normalize_option(option: &mut TensorOption) {
    option
        .execution_plans
        .sort_by(|left, right| left.tensor_name.cmp(&right.tensor_name));
    option
        .operations
        .sort_by(|left, right| left.operation_id.cmp(&right.operation_id));
    for operation in &mut option.operations {
        operation.tensor_names.sort();
    }
}

fn normalize_problem(problem: &DynamicAllocationProblem) -> DynamicAllocationProblem {
    let mut normalized = problem.clone();
    normalized.required_regimes.sort();
    normalized
        .units
        .sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
    for unit in &mut normalized.units {
        unit.members
            .sort_by(|left, right| left.name.cmp(&right.name));
        unit.expected_expert_ids.sort();
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

pub fn final_executed_tensor_bundle_sha256(
    plans: &[TensorExecutionPlan],
) -> Result<String, DynamicAllocationError> {
    let mut entries: Vec<_> = plans
        .iter()
        .map(|plan| {
            (
                plan.tensor_name.clone(),
                plan.executed_codec.clone(),
                plan.final_executed_tensor_sha256.clone(),
            )
        })
        .collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    hash_serialized(&entries)
}

fn validate_transform(plan: &TensorExecutionPlan) -> Result<(), String> {
    if plan.tensor_name.is_empty()
        || plan.operation_id.is_empty()
        || !codec_valid(&plan.stored_codec)
        || !is_sha256(&plan.stored_tensor_sha256)
        || plan.stored_payload_bytes == 0
        || !codec_valid(&plan.executed_codec)
        || !is_sha256(&plan.final_executed_tensor_sha256)
        || plan.transformations.is_empty()
    {
        return Err(format!(
            "tensor `{}` has an incomplete execution plan",
            plan.tensor_name
        ));
    }
    let mut expected_codec = &plan.stored_codec;
    let mut expected_tensor_hash = plan.stored_tensor_sha256.as_str();
    let mut expected_logical_hash: Option<&str> = None;
    for step in &plan.transformations {
        if &step.from != expected_codec
            || !codec_valid(&step.from)
            || !codec_valid(&step.to)
            || step.input_tensor_sha256 != expected_tensor_hash
            || !is_sha256(&step.input_tensor_sha256)
            || !is_sha256(&step.transform_receipt_sha256)
            || !is_sha256(&step.output_tensor_sha256)
            || !is_sha256(&step.input_logical_tensor_sha256)
            || !is_sha256(&step.output_logical_tensor_sha256)
            || expected_logical_hash
                .is_some_and(|expected| step.input_logical_tensor_sha256 != expected)
        {
            return Err(format!(
                "tensor `{}` has a broken transform chain",
                plan.tensor_name
            ));
        }
        match step.kind {
            ExecutionTransformKind::Identity
                if step.from != step.to
                    || step.input_tensor_sha256 != step.output_tensor_sha256
                    || step.input_logical_tensor_sha256 != step.output_logical_tensor_sha256 =>
            {
                return Err(format!(
                    "tensor `{}` has a non-identity identity step",
                    plan.tensor_name
                ));
            }
            ExecutionTransformKind::Identity => {}
            ExecutionTransformKind::LosslessRepack
                if !lossless_repack_compatible(&step.from, &step.to)
                    || step.input_logical_tensor_sha256 != step.output_logical_tensor_sha256 =>
            {
                return Err(format!(
                    "tensor `{}` has an unproved lossless repack",
                    plan.tensor_name
                ));
            }
            ExecutionTransformKind::LosslessRepack => {}
            ExecutionTransformKind::LossyRequantize => {}
            ExecutionTransformKind::DequantizeExpand => {
                if !matches!(step.to, TensorCodec::Dense { .. })
                    || step.input_logical_tensor_sha256 != step.output_logical_tensor_sha256
                {
                    return Err(format!(
                        "tensor `{}` expands into a non-dense codec",
                        plan.tensor_name
                    ));
                }
            }
        }
        expected_codec = &step.to;
        expected_tensor_hash = &step.output_tensor_sha256;
        expected_logical_hash = Some(&step.output_logical_tensor_sha256);
    }
    let last = plan.transformations.last().expect("checked non-empty");
    if expected_codec != &plan.executed_codec
        || last.output_tensor_sha256 != plan.final_executed_tensor_sha256
    {
        return Err(format!(
            "tensor `{}` does not bind its final executed bytes",
            plan.tensor_name
        ));
    }
    Ok(())
}

fn operation_costs(
    option: &TensorOption,
    required: &BTreeSet<InferenceRegime>,
) -> Result<BTreeMap<InferenceRegime, u64>, DynamicAllocationError> {
    let mut totals = BTreeMap::new();
    for operation in &option.operations {
        for regime in required {
            let cost = &operation.regime_costs[regime];
            let weighted = cost
                .median_nanoseconds
                .checked_mul(cost.invocation_count)
                .ok_or(DynamicAllocationError::Overflow)?;
            let total = totals.entry(*regime).or_insert(0u64);
            *total = total
                .checked_add(weighted)
                .ok_or(DynamicAllocationError::Overflow)?;
        }
    }
    Ok(totals)
}

fn validate_option(
    problem: &DynamicAllocationProblem,
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
        .map(|plan| plan.tensor_name.as_str())
        .collect();
    if plans != members || option.execution_plans.len() != unit.members.len() {
        return Err(invalid(format!(
            "option `{}` does not plan every member exactly once",
            option.option_id
        )));
    }
    for plan in &option.execution_plans {
        validate_transform(plan).map_err(invalid)?;
    }
    let planned_payload =
        option
            .execution_plans
            .iter()
            .try_fold(option.shared_metadata_bytes, |total, plan| {
                total
                    .checked_add(plan.stored_payload_bytes)
                    .ok_or(DynamicAllocationError::Overflow)
            })?;
    if planned_payload != option.payload_bytes {
        return Err(invalid(format!(
            "option `{}` payload breakdown does not match its total",
            option.option_id
        )));
    }
    if final_executed_tensor_bundle_sha256(&option.execution_plans)?
        != option.sensitivity.final_executed_tensor_bundle_sha256
    {
        return Err(invalid(format!(
            "option `{}` final tensor bundle hash is stale",
            option.option_id
        )));
    }

    let mut operation_ids = BTreeSet::new();
    for operation in &option.operations {
        if operation.operation_id.is_empty()
            || !operation_ids.insert(operation.operation_id.as_str())
            || !is_sha256(&operation.capability_decision_sha256)
        {
            return Err(invalid(format!(
                "option `{}` has malformed operation evidence",
                option.option_id
            )));
        }
        let expected: BTreeSet<_> = option
            .execution_plans
            .iter()
            .filter(|plan| plan.operation_id == operation.operation_id)
            .map(|plan| plan.tensor_name.as_str())
            .collect();
        let actual: BTreeSet<_> = operation.tensor_names.iter().map(String::as_str).collect();
        if expected.is_empty() || expected != actual || actual.len() != operation.tensor_names.len()
        {
            return Err(invalid(format!(
                "operation `{}` does not exactly cover its tensor plans",
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
            if cost.regime != *regime
                || !cost.executable
                || cost.route.is_empty()
                || cost.invocation_count == 0
                || cost.median_nanoseconds == 0
                || cost.p95_nanoseconds < cost.median_nanoseconds
                || cost.warmup_runs == 0
                || cost.measured_runs < 3
                || !is_sha256(&cost.workload_shape_sha256)
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
        || option
            .execution_plans
            .iter()
            .any(|plan| !operation_ids.contains(plan.operation_id.as_str()))
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
        || !is_sha256(&problem.tensor_catalog_sha256)
        || !is_sha256(&problem.calibration_manifest_sha256)
        || !is_sha256(&problem.capability_profile_sha256)
        || !is_sha256(&problem.proposal_workload_profile_sha256)
        || problem.sensitivity_model.method.is_empty()
        || problem.sensitivity_model.version.is_empty()
        || problem.sensitivity_model.fixed_point_scale == 0
        || !is_sha256(&problem.sensitivity_model.component_weights_sha256)
        || !is_sha256(&problem.sensitivity_model.coverage_contract_sha256)
    {
        return Err(invalid(
            "source, calibration, runtime, or workload identity is incomplete",
        ));
    }
    let SearchContract::ExactPareto { max_states } = problem.search;
    if max_states == 0 || problem.payload_budget_bytes == 0 || problem.units.is_empty() {
        return Err(invalid("search bound, budget, and units must be non-zero"));
    }
    let required: BTreeSet<_> = problem.required_regimes.iter().copied().collect();
    if required.is_empty() || required.len() != problem.required_regimes.len() {
        return Err(invalid("required regimes must be non-empty and unique"));
    }

    let mut unit_ids = BTreeSet::new();
    let mut tensor_names = BTreeSet::new();
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
        let mut option_ids = BTreeSet::new();
        let mut operation_topology: Option<BTreeMap<&str, BTreeSet<&str>>> = None;
        for option in &unit.options {
            if !option_ids.insert(option.option_id.as_str()) {
                return Err(invalid("allocation unit has duplicate option ids"));
            }
            validate_option(problem, unit, option, &required)?;
            let topology: BTreeMap<_, _> = option
                .operations
                .iter()
                .map(|operation| {
                    (
                        operation.operation_id.as_str(),
                        operation.tensor_names.iter().map(String::as_str).collect(),
                    )
                })
                .collect();
            if operation_topology
                .as_ref()
                .is_some_and(|expected| expected != &topology)
            {
                return Err(invalid(
                    "options in one allocation unit must share a stable operation topology",
                ));
            }
            operation_topology = Some(topology);
        }
    }
    if tensor_count != problem.expected_tensor_count
        || tensor_catalog_sha256(&problem.units)? != problem.tensor_catalog_sha256
    {
        return Err(invalid("tensor catalog count or canonical hash mismatch"));
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
pub fn allocate_dynamic_frontier(
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
    if minimum > normalized.payload_budget_bytes {
        return Err(DynamicAllocationError::BudgetTooSmall {
            required: minimum,
            budget: normalized.payload_budget_bytes,
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
                if bytes > normalized.payload_budget_bytes {
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
            tensor_catalog_sha256: normalized.tensor_catalog_sha256.clone(),
            calibration_manifest_sha256: normalized.calibration_manifest_sha256.clone(),
            capability_profile_sha256: normalized.capability_profile_sha256.clone(),
            proposal_workload_profile_sha256: normalized.proposal_workload_profile_sha256.clone(),
            payload_budget_bytes: normalized.payload_budget_bytes,
            total_payload_bytes: state.bytes,
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
        left.total_payload_bytes
            .cmp(&right.total_payload_bytes)
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
        policy
            .decisions
            .sort_by(|left, right| left.unit_id.cmp(&right.unit_id));
        for decision in &mut policy.decisions {
            normalize_option(&mut decision.selected_option);
        }
    }
    normalized.policies.sort_by(|left, right| {
        left.total_payload_bytes
            .cmp(&right.total_payload_bytes)
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
pub fn validate_policy_frontier(
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
