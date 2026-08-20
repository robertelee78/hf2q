use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::intelligence::measured_auto_quant::{
    ExecutionIdentity, InferenceRegime, SourceIdentity,
};

fn digest(label: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(label.as_bytes()))
}

fn source() -> SourceIdentity {
    SourceIdentity {
        model_id: "Qwen/Qwen3.8-27B".into(),
        revision: "source-revision".into(),
        config_sha256: digest("config"),
        tensor_bundle_sha256: digest("tensors"),
        tokenizer_bundle_sha256: digest("tokenizer"),
        chat_template_sha256: digest("template"),
    }
}

fn codec(id: &str) -> TensorCodec {
    TensorCodec::Gguf {
        codec: match id {
            "q4" => GgufCodec::Q4K,
            "q6" => GgufCodec::Q6K,
            "q8" => GgufCodec::Q8_0,
            _ => GgufCodec::Q5K,
        },
    }
}

fn member(unit: &str) -> TensorMember {
    TensorMember {
        name: format!("blk.{unit}.ffn_down.weight"),
        shape: vec![64, 128],
        role: "ffn_down".into(),
        source_dtype: ScalarDType::Bf16,
        source_tensor_sha256: digest(&format!("source-{unit}")),
        layer_index: None,
        expert_index: None,
    }
}

fn regime_cost(id: &str, regime: InferenceRegime, nanoseconds: u64) -> RegimeCost {
    RegimeCost {
        regime,
        workload_shape_sha256: digest(&format!("shape-{regime:?}")),
        executable: true,
        specialized_for_regime: true,
        route: format!("route-{id}-{regime:?}"),
        invocation_count: 1,
        median_nanoseconds: nanoseconds,
        p95_nanoseconds: nanoseconds + 5,
        warmup_runs: 2,
        measured_runs: 5,
        measurement_receipt_sha256: digest(&format!("measure-{id}-{regime:?}")),
    }
}

fn option(
    id: &str,
    member: &TensorMember,
    bytes: u64,
    loss: u64,
    decode_ns: u64,
    prefill_ns: u64,
) -> TensorOption {
    let stored = codec(id);
    let stored_hash = digest(&format!("stored-{}-{id}", member.name));
    let logical_hash = digest(&format!("logical-{}-{id}", member.name));
    let plan = TensorExecutionPlan {
        tensor_name: member.name.clone(),
        stored_codec: stored.clone(),
        stored_tensor_sha256: stored_hash.clone(),
        stored_payload_bytes: bytes,
        executed_codec: stored.clone(),
        transformations: vec![ExecutionTransformStep {
            kind: ExecutionTransformKind::Identity,
            from: stored.clone(),
            to: stored,
            input_tensor_sha256: stored_hash.clone(),
            transform_receipt_sha256: digest(&format!("transform-{}-{id}", member.name)),
            output_tensor_sha256: stored_hash.clone(),
            input_logical_tensor_sha256: logical_hash.clone(),
            output_logical_tensor_sha256: logical_hash,
        }],
        final_executed_tensor_sha256: stored_hash,
        operation_id: format!("op-{}", member.name),
    };
    let operation = OperationExecutionEvidence {
        operation_id: plan.operation_id.clone(),
        graph_path: format!("qwen35.{unit}", unit = member.name),
        tensor_names: vec![member.name.clone()],
        capability_decision_sha256: digest(&format!("cap-decision-{}-{id}", member.name)),
        regime_costs: BTreeMap::from([
            (
                InferenceRegime::TextDecodeM1,
                regime_cost(id, InferenceRegime::TextDecodeM1, decode_ns),
            ),
            (
                InferenceRegime::TextPrefill,
                regime_cost(id, InferenceRegime::TextPrefill, prefill_ns),
            ),
        ]),
    };
    let plans = vec![plan];
    TensorOption {
        option_id: id.into(),
        payload_bytes: bytes,
        sensitivity: SensitivityEvidence {
            calibration_manifest_sha256: digest("calibration"),
            sensitivity_receipt_sha256: digest(&format!("sensitivity-{}-{id}", member.name)),
            loss_units: loss,
            imatrix_weighted_error_units: loss,
            teacher_kl_alignment_units: 0,
            block_output_error_units: loss,
            uncertainty_units: 1,
            activation_rows: 4_096,
            expert_activation_rows: BTreeMap::new(),
            final_executed_tensor_bundle_sha256: final_executed_tensor_bundle_sha256(&plans)
                .unwrap(),
        },
        execution_plans: plans,
        operations: vec![operation],
        shared_metadata_bytes: 0,
        storage_manifest_receipt_sha256: digest(&format!("storage-{}-{id}", member.name)),
        capability_profile_sha256: digest("capability-profile"),
    }
}

fn unit(id: &str, specs: &[(&str, u64, u64, u64, u64)]) -> TensorAllocationUnit {
    let member = member(id);
    let operation_id = format!("op-{}", member.name);
    let graph_path = format!("qwen35.{}", member.name);
    TensorAllocationUnit {
        unit_id: id.into(),
        expected_expert_ids: Vec::new(),
        operations: vec![TensorOperation {
            operation_id,
            graph_path,
            tensor_names: vec![member.name.clone()],
        }],
        options: specs
            .iter()
            .map(|(name, bytes, loss, decode, prefill)| {
                option(name, &member, *bytes, *loss, *decode, *prefill)
            })
            .collect(),
        members: vec![member],
    }
}

fn make_problem(
    units: Vec<TensorAllocationUnit>,
    budget: u64,
    max_states: usize,
) -> DynamicAllocationProblem {
    let count = units.iter().map(|unit| unit.members.len()).sum();
    let catalog = tensor_catalog_sha256(&units).unwrap();
    DynamicAllocationProblem {
        schema_version: DYNAMIC_ALLOCATION_SCHEMA_VERSION,
        source: source(),
        execution: ExecutionIdentity {
            hf2q_revision: "deadbeef".into(),
            mlx_native_version: "0.10.14".into(),
            hardware_id: "apple-m5-max-128gb".into(),
            os_build: "25A123".into(),
        },
        tensor_catalog_sha256: catalog,
        expected_tensor_count: count,
        dataset_partition_manifest_sha256: digest("dataset-partition"),
        tensor_partition_manifest_sha256: digest("tensor-partition"),
        calibration_manifest_sha256: digest("calibration"),
        sensitivity_model: SensitivityModelIdentity {
            method: "dynamic-kl-gradient-plus-imatrix".into(),
            version: "1".into(),
            fixed_point_scale: 1_000_000,
            component_weights_sha256: digest("component-weights"),
            coverage_contract_sha256: digest("coverage"),
            coverage_receipt_sha256: digest("coverage-receipt"),
        },
        capability_profile_sha256: digest("capability-profile"),
        proposal_workload_profile_sha256: digest("workload-profile"),
        required_regimes: vec![InferenceRegime::TextDecodeM1, InferenceRegime::TextPrefill],
        payload_budget_bytes: budget,
        minimum_expert_activation_rows: 64,
        search: SearchContract::ExactPareto { max_states },
        units,
    }
}

#[test]
fn exact_frontier_beats_greedy_quality_per_byte_counterexample() {
    let problem = make_problem(
        vec![
            unit("a", &[("q4", 10, 100, 10, 10), ("q6", 16, 88, 10, 10)]),
            unit("b", &[("q4", 10, 100, 10, 10), ("q6", 15, 91, 10, 10)]),
            unit("c", &[("q4", 10, 100, 10, 10), ("q6", 15, 91, 10, 10)]),
        ],
        40,
        64,
    );
    let frontier = allocate_dynamic_frontier(&problem).unwrap();
    let best = frontier
        .policies
        .iter()
        .min_by_key(|policy| policy.total_loss_units)
        .unwrap();
    assert_eq!((best.total_payload_bytes, best.total_loss_units), (40, 282));
    let selected: BTreeMap<_, _> = best
        .decisions
        .iter()
        .map(|decision| {
            (
                decision.unit_id.as_str(),
                decision.selected_option.option_id.as_str(),
            )
        })
        .collect();
    assert_eq!(
        selected,
        BTreeMap::from([("a", "q4"), ("b", "q6"), ("c", "q6")])
    );
}

fn metric(policy: &PrecisionPolicyManifest) -> (u64, u64, Vec<u64>) {
    (
        policy.total_payload_bytes,
        policy.total_loss_units,
        policy
            .total_regime_cost_nanoseconds
            .values()
            .copied()
            .collect(),
    )
}

fn dominates_metric(left: &(u64, u64, Vec<u64>), right: &(u64, u64, Vec<u64>)) -> bool {
    let no_worse =
        left.0 <= right.0 && left.1 <= right.1 && left.2.iter().zip(&right.2).all(|(a, b)| a <= b);
    let better =
        left.0 < right.0 || left.1 < right.1 || left.2.iter().zip(&right.2).any(|(a, b)| a < b);
    no_worse && better
}

#[test]
fn exact_frontier_matches_brute_force_metrics() {
    let problem = make_problem(
        vec![
            unit("a", &[("q4", 10, 12, 9, 5), ("q6", 15, 6, 12, 4)]),
            unit("b", &[("q4", 8, 10, 6, 8), ("q8", 17, 1, 13, 3)]),
            unit("c", &[("q4", 7, 11, 4, 9), ("q6", 12, 4, 9, 4)]),
        ],
        40,
        128,
    );
    let actual: BTreeSet<_> = allocate_dynamic_frontier(&problem)
        .unwrap()
        .policies
        .iter()
        .map(metric)
        .collect();
    let mut all = Vec::new();
    for a in &problem.units[0].options {
        for b in &problem.units[1].options {
            for c in &problem.units[2].options {
                let selected = [a, b, c];
                let bytes = selected.iter().map(|option| option.payload_bytes).sum();
                if bytes > problem.payload_budget_bytes {
                    continue;
                }
                let loss = selected
                    .iter()
                    .map(|option| option.sensitivity.loss_units)
                    .sum();
                let mut regimes = problem.required_regimes.clone();
                regimes.sort();
                let costs = regimes
                    .iter()
                    .map(|regime| {
                        selected
                            .iter()
                            .map(|option| {
                                option.operations[0].regime_costs[regime].median_nanoseconds
                            })
                            .sum()
                    })
                    .collect();
                all.push((bytes, loss, costs));
            }
        }
    }
    let expected: BTreeSet<_> = all
        .iter()
        .filter(|candidate| !all.iter().any(|other| dominates_metric(other, candidate)))
        .cloned()
        .collect();
    assert_eq!(actual, expected);
}

#[test]
fn preserves_quality_size_and_each_regime_tradeoff() {
    let frontier = allocate_dynamic_frontier(&make_problem(
        vec![unit(
            "a",
            &[
                ("q4", 10, 20, 5, 20),
                ("q6", 15, 10, 10, 10),
                ("q8", 20, 5, 20, 5),
            ],
        )],
        20,
        16,
    ))
    .unwrap();
    assert_eq!(frontier.policies.len(), 3);
}

#[test]
fn exact_state_limit_fails_closed() {
    let error = allocate_dynamic_frontier(&make_problem(
        vec![unit(
            "a",
            &[
                ("q4", 10, 20, 5, 20),
                ("q6", 15, 10, 10, 10),
                ("q8", 20, 5, 20, 5),
            ],
        )],
        20,
        2,
    ))
    .unwrap_err();
    assert_eq!(
        error,
        DynamicAllocationError::FrontierLimitExceeded {
            after_units: 1,
            after_options: 3,
            states: 3,
            max_states: 2,
            states_generated: 3,
        }
    );
}

#[test]
fn live_state_limit_rejects_before_a_late_option_could_dominate() {
    let problem = make_problem(
        vec![unit(
            "a",
            &[("a", 10, 20, 5, 20), ("b", 20, 5, 20, 5), ("z", 9, 4, 4, 4)],
        )],
        20,
        1,
    );
    assert_eq!(
        allocate_dynamic_frontier(&problem).unwrap_err(),
        DynamicAllocationError::FrontierLimitExceeded {
            after_units: 1,
            after_options: 2,
            states: 2,
            max_states: 1,
            states_generated: 2,
        }
    );
}

#[test]
fn unit_option_and_regime_order_do_not_change_bytes() {
    let mut problem = make_problem(
        vec![
            unit("a", &[("q4", 10, 12, 9, 5), ("q6", 15, 6, 12, 4)]),
            unit("b", &[("q4", 8, 10, 6, 8), ("q8", 17, 1, 13, 3)]),
        ],
        32,
        64,
    );
    let expected = allocate_dynamic_frontier(&problem).unwrap();
    problem.units.reverse();
    for unit in &mut problem.units {
        unit.options.reverse();
    }
    problem.required_regimes.reverse();
    assert_eq!(expected, allocate_dynamic_frontier(&problem).unwrap());
    validate_policy_frontier(&problem, &expected).unwrap();
}

#[test]
fn rejects_incomplete_or_stale_tensor_catalog() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 10, 8);
    problem.expected_tensor_count += 1;
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
    problem.expected_tensor_count -= 1;
    problem.tensor_catalog_sha256 = digest("stale");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn rejects_precision_change_without_lossy_transform_receipt() {
    let mut problem = make_problem(vec![unit("a", &[("q6", 10, 10, 5, 5)])], 10, 8);
    let option = &mut problem.units[0].options[0];
    let plan = &mut option.execution_plans[0];
    plan.executed_codec = codec("q4");
    plan.transformations[0].kind = ExecutionTransformKind::LosslessRepack;
    plan.transformations[0].to = codec("q4");
    option.sensitivity.final_executed_tensor_bundle_sha256 =
        final_executed_tensor_bundle_sha256(&option.execution_plans).unwrap();
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn rejects_broken_identity_hash_chain_and_unproved_dense_quant_repack() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 10, 8);
    problem.units[0].options[0].execution_plans[0].transformations[0].output_tensor_sha256 =
        digest("different-output");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 10, 8);
    let option = &mut problem.units[0].options[0];
    let plan = &mut option.execution_plans[0];
    plan.stored_codec = TensorCodec::Dense {
        dtype: ScalarDType::Bf16,
    };
    let step = &mut plan.transformations[0];
    step.kind = ExecutionTransformKind::LosslessRepack;
    step.from = plan.stored_codec.clone();
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn rejects_payload_total_without_storage_breakdown_provenance() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 11, 8);
    problem.units[0].options[0].payload_bytes = 11;
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn packed_expert_unit_requires_explicit_per_expert_coverage() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 10, 8);
    problem.units[0].expected_expert_ids = vec![0, 1];
    problem.units[0].options[0]
        .sensitivity
        .expert_activation_rows = BTreeMap::from([(0, 64), (1, 63)]);
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
    problem.units[0].options[0]
        .sensitivity
        .expert_activation_rows
        .insert(1, 64);
    assert!(allocate_dynamic_frontier(&problem).is_ok());
}

#[test]
fn rejects_mismatched_common_evidence_and_extra_regime() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 10, 8);
    problem.units[0].options[0].capability_profile_sha256 = digest("other-capability");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 10, 8);
    let operation = &mut problem.units[0].options[0].operations[0];
    operation.regime_costs.insert(
        InferenceRegime::LongContextDecode,
        regime_cost("q4", InferenceRegime::LongContextDecode, 10),
    );
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn canonical_frontier_round_trip_and_mutation_validation() {
    let problem = make_problem(
        vec![
            unit("a", &[("q4", 10, 10, 5, 8), ("q6", 15, 5, 9, 4)]),
            unit("b", &[("q4", 10, 10, 8, 5), ("q6", 15, 5, 4, 9)]),
        ],
        30,
        32,
    );
    let frontier = allocate_dynamic_frontier(&problem).unwrap();
    let bytes = canonical_frontier_bytes(&frontier).unwrap();
    let decoded: PolicyFrontier = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded, frontier);
    validate_policy_frontier(&problem, &decoded).unwrap();
    let mut reordered = decoded.clone();
    reordered.policies.reverse();
    for policy in &mut reordered.policies {
        policy.decisions.reverse();
    }
    assert_eq!(canonical_frontier_bytes(&reordered).unwrap(), bytes);
    validate_policy_frontier(&problem, &reordered).unwrap();
    let mut reordered_policy = decoded.policies[0].clone();
    reordered_policy.decisions.reverse();
    assert_eq!(
        precision_policy_sha256(&reordered_policy).unwrap(),
        precision_policy_sha256(&decoded.policies[0]).unwrap()
    );
    let mut tampered = decoded;
    tampered.policies[0].total_loss_units += 1;
    assert_eq!(
        validate_policy_frontier(&problem, &tampered),
        Err(DynamicAllocationError::FrontierMismatch)
    );
}

#[test]
fn qwen_scale_catalog_stays_bounded_when_local_options_are_dominated() {
    let units = (0..866)
        .map(|index| {
            unit(
                &format!("u{index:04}"),
                &[
                    ("q4", 10, 1, 1, 1),
                    ("q6", 12, 2, 2, 2),
                    ("q8", 14, 3, 3, 3),
                ],
            )
        })
        .collect();
    let frontier = allocate_dynamic_frontier(&make_problem(units, 8_660, 8)).unwrap();
    assert_eq!(frontier.policies.len(), 1);
    assert_eq!(frontier.search_receipt.peak_frontier_states, 1);
    assert_eq!(frontier.search_receipt.states_generated, 2_598);
}

#[test]
fn fused_operation_cost_is_counted_once_for_two_members() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 7, 11)])], 20, 8);
    let second = TensorMember {
        name: "blk.a.ffn_up.weight".into(),
        source_tensor_sha256: digest("source-up"),
        ..problem.units[0].members[0].clone()
    };
    problem.units[0].members.push(second.clone());
    problem.units[0].operations[0]
        .tensor_names
        .push(second.name.clone());
    let option = &mut problem.units[0].options[0];
    let mut plan = option.execution_plans[0].clone();
    plan.tensor_name = second.name.clone();
    plan.stored_tensor_sha256 = digest("stored-up");
    plan.final_executed_tensor_sha256 = plan.stored_tensor_sha256.clone();
    plan.transformations[0].input_tensor_sha256 = plan.stored_tensor_sha256.clone();
    plan.transformations[0].output_tensor_sha256 = plan.stored_tensor_sha256.clone();
    plan.transformations[0].input_logical_tensor_sha256 = digest("logical-up");
    plan.transformations[0].output_logical_tensor_sha256 = digest("logical-up");
    option.execution_plans.push(plan);
    option.operations[0].tensor_names.push(second.name.clone());
    option.payload_bytes = 20;
    option.sensitivity.final_executed_tensor_bundle_sha256 =
        final_executed_tensor_bundle_sha256(&option.execution_plans).unwrap();
    problem.expected_tensor_count = 2;
    problem.tensor_catalog_sha256 = tensor_catalog_sha256(&problem.units).unwrap();
    let policy = &allocate_dynamic_frontier(&problem).unwrap().policies[0];
    assert_eq!(
        policy.total_regime_cost_nanoseconds[&InferenceRegime::TextDecodeM1],
        7
    );
}
