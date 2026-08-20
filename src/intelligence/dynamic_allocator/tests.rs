use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::core::provenance::tensor_execution::*;
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

fn member_with_rows(unit: &str, rows: usize) -> TensorMember {
    TensorMember {
        name: format!("blk.{unit}.ffn_down.weight"),
        shape: vec![rows, 256],
        role: "ffn_down".into(),
        source_dtype: ScalarDType::Bf16,
        source_tensor_sha256: digest(&format!("source-{unit}")),
        layer_index: None,
        expert_index: None,
    }
}

fn regime_cost(
    id: &str,
    regime: InferenceRegime,
    nanoseconds: u64,
    validated: &ValidatedTensorExecutionManifest,
    operation_id: &str,
) -> RegimeCost {
    let physical_regime = match regime {
        InferenceRegime::TextPrefill => TensorExecutionRegime::TextPrefill,
        InferenceRegime::TextDecodeM1 => TensorExecutionRegime::TextDecodeM1,
        InferenceRegime::TextDecodeWidthN => TensorExecutionRegime::TextDecodeWidthN,
        InferenceRegime::LongContextDecode => TensorExecutionRegime::LongContextDecode,
        InferenceRegime::MultimodalPrefill => TensorExecutionRegime::MultimodalPrefill,
    };
    let runtime_binding_ids = validated
        .manifest()
        .operations
        .iter()
        .filter(|binding| {
            binding.operation_id == operation_id && binding.workload_regime == physical_regime
        })
        .map(|binding| binding.binding_id.clone())
        .collect();
    RegimeCost {
        regime,
        runtime_binding_ids,
        runtime_binding_bundle_sha256: runtime_regime_binding_bundle_sha256(
            validated,
            operation_id,
            physical_regime,
        )
        .unwrap(),
        executable: true,
        specialized_for_regime: true,
        median_nanoseconds: nanoseconds,
        p95_nanoseconds: nanoseconds + 5,
        warmup_runs: 2,
        measured_runs: 5,
        measurement_receipt_sha256: digest(&format!("measure-{id}-{regime:?}")),
    }
}

fn physical_manifest(member: &TensorMember, id: &str) -> TensorExecutionManifest {
    let row_width = *member.shape.last().unwrap() as u64;
    let outer_rows = member.shape[..member.shape.len() - 1]
        .iter()
        .map(|dimension| *dimension as u64)
        .product::<u64>();
    let (wire_type_id, type_name, block_values, block_bytes) = match id {
        "q6" | "b" => (14, "Q6_K", 256, 210),
        "q8" | "z" => (8, "Q8_0", 32, 34),
        "q5" => (13, "Q5_K", 256, 176),
        _ => (12, "Q4_K", 256, 144),
    };
    let stored_bytes = outer_rows * (row_width / block_values) * block_bytes;
    let source_bytes = outer_rows * row_width * 2;
    let source_node = TensorStateNode {
        node_id: format!("source:{}:{id}", member.name),
        stage: TensorStateStage::Source,
        semantic_name: member.name.clone(),
        shape: member
            .shape
            .iter()
            .map(|dimension| *dimension as u64)
            .collect(),
        layout: "row-major-outermost-first-v1".into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::Bf16,
        },
        byte_len: source_bytes,
        byte_sha256: member.source_tensor_sha256.clone(),
        logical_f32_sha256: digest(&format!("source-logical:{}:{id}", member.name)),
        artifact_region: Some(ArtifactRegion {
            artifact_id: format!("source-artifact:{}:{id}", member.name),
            byte_offset: 0,
            byte_len: source_bytes,
        }),
    };
    let stored_hash = digest(&format!("stored:{}:{id}", member.name));
    let logical_hash = digest(&format!("stored-logical:{}:{id}", member.name));
    let stored_node = TensorStateNode {
        node_id: format!("stored:{}:{id}", member.name),
        stage: TensorStateStage::Stored,
        semantic_name: member.name.clone(),
        shape: member
            .shape
            .iter()
            .map(|dimension| *dimension as u64)
            .collect(),
        layout: "row-major-outermost-first-v1".into(),
        codec: PhysicalTensorCodec::Ggml {
            wire_type_id,
            type_name: type_name.into(),
        },
        byte_len: stored_bytes,
        byte_sha256: stored_hash.clone(),
        logical_f32_sha256: logical_hash.clone(),
        artifact_region: Some(ArtifactRegion {
            artifact_id: format!("gguf-artifact:{}:{id}", member.name),
            byte_offset: 0,
            byte_len: stored_bytes,
        }),
    };
    let converted_node = TensorStateNode {
        node_id: format!("converted:{}:{id}:f32", member.name),
        stage: TensorStateStage::Converted,
        semantic_name: format!("{}.converted", member.name),
        shape: member
            .shape
            .iter()
            .map(|dimension| *dimension as u64)
            .collect(),
        layout: "row-major-outermost-first-v1".into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::F32,
        },
        byte_len: outer_rows * row_width * 4,
        byte_sha256: digest(&format!("converted:{}:{id}", member.name)),
        logical_f32_sha256: source_node.logical_f32_sha256.clone(),
        artifact_region: None,
    };
    let roundtripped_node = TensorStateNode {
        node_id: format!("converted:{}:{id}:f16-roundtrip", member.name),
        semantic_name: format!("{}.roundtripped", member.name),
        byte_sha256: digest(&format!("roundtripped:{}:{id}", member.name)),
        logical_f32_sha256: logical_hash.clone(),
        ..converted_node.clone()
    };
    let loaded_node = TensorStateNode {
        node_id: format!("loaded:{}:{id}", member.name),
        stage: TensorStateStage::Loaded,
        artifact_region: None,
        ..stored_node.clone()
    };
    let executed_node = TensorStateNode {
        node_id: format!("executed:{}:{id}", member.name),
        stage: TensorStateStage::Executed,
        artifact_region: None,
        ..stored_node.clone()
    };
    let operation_id = format!("op-{}", member.name);
    let json = "{}".to_string();
    let json_evidence = CanonicalJsonEvidence {
        sha256: digest(&json),
        canonical_json: json,
    };
    let revision = &digest("hf2q-revision")[..40];
    let mut manifest = TensorExecutionManifest {
        schema_version: TENSOR_EXECUTION_MANIFEST_SCHEMA_VERSION,
        source_manifest_sha256: digest("verified-source"),
        source_tensor_inventory_sha256: digest("inventory"),
        tensor_partition_manifest_sha256: digest("tensor-partition"),
        conversion_receipt_sha256: digest(&format!("conversion:{}:{id}", member.name)),
        logical_hash_encoding: "hf2q-framed-f32-le-v1".into(),
        runtime: TensorRuntimeBinding {
            hf2q_revision: revision.into(),
            mlx_native_version: "0.10.14".into(),
            mlx_native_capability_schema_version: 1,
            routing_policy_sha256: digest("routing"),
            graph_configuration_sha256: digest("graph"),
            capability_profile_sha256: digest("capability-profile"),
            hardware_profile_sha256: digest("hardware"),
            dwq_overlay_sha256: None,
        },
        scope: TensorExecutionScope {
            model_family: "qwen35_dense".into(),
            profile: "autoregressive-text".into(),
            included_paths: vec!["qwen35.autoregressive_text".into()],
            excluded_paths: BTreeMap::from([("dwq".into(), "out-of-scope".into())]),
        },
        artifacts: vec![
            ArtifactEvidence {
                artifact_id: format!("source-artifact:{}:{id}", member.name),
                role: "source".into(),
                byte_len: source_bytes,
                sha256: member.source_tensor_sha256.clone(),
            },
            ArtifactEvidence {
                artifact_id: format!("gguf-artifact:{}:{id}", member.name),
                role: "converted".into(),
                byte_len: stored_bytes,
                sha256: digest(&format!("gguf:{}:{id}", member.name)),
            },
        ],
        nodes: vec![
            source_node.clone(),
            converted_node.clone(),
            roundtripped_node.clone(),
            stored_node.clone(),
            loaded_node.clone(),
            executed_node.clone(),
        ],
        transforms: vec![
            TensorTransformEdge {
                edge_id: format!("decode:{}:{id}", member.name),
                inputs: vec![TransformPort {
                    role: "source".into(),
                    node_id: source_node.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: converted_node.node_id.clone(),
                }],
                operation: TensorTransformOperation::SourceDecode,
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("decode-receipt:{}:{id}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("roundtrip:{}:{id}", member.name),
                inputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: converted_node.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "converted_roundtripped".into(),
                    node_id: roundtripped_node.node_id.clone(),
                }],
                operation: TensorTransformOperation::F16Roundtrip,
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("roundtrip-receipt:{}:{id}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("quantize:{}:{id}", member.name),
                inputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: roundtripped_node.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "stored".into(),
                    node_id: stored_node.node_id.clone(),
                }],
                operation: TensorTransformOperation::GgufQuantize {
                    implementation_id: "test-q4k".into(),
                    calibration_receipt_sha256: None,
                },
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("quantize-receipt:{}:{id}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("load:{}:{id}", member.name),
                inputs: vec![TransformPort {
                    role: "stored".into(),
                    node_id: stored_node.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "loaded".into(),
                    node_id: loaded_node.node_id.clone(),
                }],
                operation: TensorTransformOperation::DirectBlockLoad,
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("load-receipt:{}:{id}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("bind:{}:{id}", member.name),
                inputs: vec![TransformPort {
                    role: "loaded".into(),
                    node_id: loaded_node.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "executed".into(),
                    node_id: executed_node.node_id.clone(),
                }],
                operation: TensorTransformOperation::RuntimeBind,
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("bind-receipt:{}:{id}", member.name)),
            },
        ],
        operations: [
            ("decode", TensorExecutionRegime::TextDecodeM1),
            ("prefill", TensorExecutionRegime::TextPrefill),
        ]
        .into_iter()
        .map(|(label, workload_regime)| RuntimeOperationBinding {
            binding_id: format!("binding:{}:{id}:{label}", member.name),
            operation_id: operation_id.clone(),
            graph_path: format!("qwen35.{}", member.name),
            entrypoint: "quantized_matmul_ggml_with_policy".into(),
            workload_regime,
            invocation_count: 1,
            source_tensor_names: vec![member.name.clone()],
            inputs: vec![TransformPort {
                role: "weight".into(),
                node_id: executed_node.node_id.clone(),
            }],
            capability: RuntimeCapabilityEvidence::Ggml {
                request: json_evidence.clone(),
                decision: json_evidence.clone(),
                requires_device_probe: false,
                resolved_runtime_trace: None,
            },
        })
        .collect(),
        dispositions: vec![SourceTensorDisposition {
            source_node_id: source_node.node_id,
            disposition: SourceTensorDispositionKind::Variable,
            reason: "dynamic candidate".into(),
            terminal_node_ids: vec![executed_node.node_id],
        }],
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = tensor_execution_manifest_sha256(&manifest).unwrap();
    verify_tensor_execution_manifest(&manifest).unwrap();
    manifest
}

fn option(
    id: &str,
    member: &TensorMember,
    _nominal_bytes: u64,
    loss: u64,
    decode_ns: u64,
    prefill_ns: u64,
) -> (TensorOption, TensorExecutionManifest) {
    let manifest = physical_manifest(member, id);
    let validated = verify_tensor_execution_manifest(&manifest).unwrap();
    let slice = tensor_lineage_slice(&validated, &member.name).unwrap();
    let operation_id = format!("op-{}", member.name);
    let executed_tensor_node_ids: Vec<_> = slice
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Executed)
        .map(|node| node.node_id.clone())
        .collect();
    let final_bundle: BTreeMap<_, _> = slice
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Executed)
        .map(|node| (node.node_id.clone(), node.clone()))
        .collect();
    let final_bundle_sha256 = {
        let bytes = serde_json::to_vec(&final_bundle).unwrap();
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(bytes))
    };
    let candidate = TensorOption {
        option_id: id.into(),
        execution_plans: vec![TensorExecutionPlan {
            source_tensor_name: member.name.clone(),
            execution_manifest_sha256: manifest.manifest_sha256.clone(),
            lineage_slice_sha256: slice.slice_sha256,
        }],
        operations: vec![OperationExecutionEvidence {
            operation_id: operation_id.clone(),
            graph_path: format!("qwen35.{}", member.name),
            source_tensor_names: vec![member.name.clone()],
            executed_tensor_node_ids,
            capability_binding_bundle_sha256: runtime_capability_binding_bundle_sha256(
                &validated,
                &operation_id,
            )
            .unwrap(),
            regime_costs: BTreeMap::from([
                (
                    InferenceRegime::TextDecodeM1,
                    regime_cost(
                        id,
                        InferenceRegime::TextDecodeM1,
                        decode_ns,
                        &validated,
                        &operation_id,
                    ),
                ),
                (
                    InferenceRegime::TextPrefill,
                    regime_cost(
                        id,
                        InferenceRegime::TextPrefill,
                        prefill_ns,
                        &validated,
                        &operation_id,
                    ),
                ),
            ]),
        }],
        payload_bytes: slice
            .nodes
            .iter()
            .filter(|node| node.stage == TensorStateStage::Stored)
            .map(|node| node.byte_len)
            .sum(),
        storage_manifest_receipt_sha256: manifest.conversion_receipt_sha256.clone(),
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
            final_executed_tensor_bundle_sha256: final_bundle_sha256,
        },
        capability_profile_sha256: digest("capability-profile"),
    };
    (candidate, manifest)
}

struct UnitFixture {
    unit: TensorAllocationUnit,
    manifests: Vec<TensorExecutionManifest>,
}

fn unit(id: &str, specs: &[(&str, u64, u64, u64, u64)]) -> UnitFixture {
    unit_with_rows(id, 1, specs)
}

fn unit_with_rows(id: &str, rows: usize, specs: &[(&str, u64, u64, u64, u64)]) -> UnitFixture {
    let member = member_with_rows(id, rows);
    let mut options = Vec::new();
    let mut manifests = Vec::new();
    for (name, bytes, loss, decode, prefill) in specs {
        let (candidate, manifest) = option(name, &member, *bytes, *loss, *decode, *prefill);
        options.push(candidate);
        manifests.push(manifest);
    }
    UnitFixture {
        unit: TensorAllocationUnit {
            unit_id: id.into(),
            members: vec![member.clone()],
            expected_expert_ids: Vec::new(),
            operations: vec![TensorOperation {
                operation_id: format!("op-{}", member.name),
                graph_path: format!("qwen35.{}", member.name),
                tensor_names: vec![member.name],
            }],
            options,
        },
        manifests,
    }
}

fn make_problem(
    fixtures: Vec<UnitFixture>,
    budget: u64,
    max_states: usize,
) -> DynamicAllocationProblem {
    let mut units = Vec::new();
    let mut manifests = Vec::new();
    for fixture in fixtures {
        units.push(fixture.unit);
        manifests.extend(fixture.manifests);
    }
    let tensor_runtime = manifests[0].runtime.clone();
    let execution_scope = manifests[0].scope.clone();
    DynamicAllocationProblem {
        schema_version: DYNAMIC_ALLOCATION_SCHEMA_VERSION,
        source: source(),
        execution: ExecutionIdentity {
            hf2q_revision: digest("hf2q-revision")[..40].into(),
            mlx_native_version: "0.10.14".into(),
            hardware_id: "apple-m5-max-128gb".into(),
            os_build: "25A123".into(),
        },
        tensor_runtime,
        execution_scope,
        tensor_catalog_sha256: tensor_catalog_sha256(&units).unwrap(),
        expected_tensor_count: units.iter().map(|unit| unit.members.len()).sum(),
        dataset_partition_manifest_sha256: digest("dataset-partition"),
        tensor_partition_manifest_sha256: digest("tensor-partition"),
        execution_manifest_catalog_sha256: execution_manifest_catalog_sha256(&manifests).unwrap(),
        execution_manifest_catalog: manifests,
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
        variable_payload_budget_bytes: budget,
        minimum_expert_activation_rows: 64,
        search: SearchContract::ExactPareto { max_states },
        units,
    }
}

fn metric(policy: &PrecisionPolicyManifest) -> (u64, u64, Vec<u64>) {
    (
        policy.total_variable_payload_bytes,
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
fn exact_frontier_selects_best_quality_under_real_payload_budget() {
    let problem = make_problem(
        vec![
            unit("a", &[("q4", 10, 100, 10, 10), ("q6", 16, 88, 10, 10)]),
            unit("b", &[("q4", 10, 100, 10, 10), ("q6", 15, 91, 10, 10)]),
            unit("c", &[("q4", 10, 100, 10, 10), ("q6", 15, 91, 10, 10)]),
        ],
        564,
        64,
    );
    let frontier = allocate_dynamic_frontier(&problem).unwrap();
    let best = frontier
        .policies
        .iter()
        .min_by_key(|policy| policy.total_loss_units)
        .unwrap();
    assert_eq!(
        (best.total_variable_payload_bytes, best.total_loss_units),
        (564, 279)
    );
}

#[test]
fn exact_frontier_beats_greedy_quality_per_byte_counterexample() {
    let problem = make_problem(
        vec![
            unit_with_rows("a", 5, &[("q4", 0, 100, 10, 10), ("q6", 0, 83, 10, 10)]),
            unit_with_rows("b", 3, &[("q4", 0, 100, 10, 10), ("q6", 0, 90, 10, 10)]),
            unit_with_rows("c", 3, &[("q4", 0, 100, 10, 10), ("q6", 0, 90, 10, 10)]),
        ],
        1_980,
        64,
    );
    let best = allocate_dynamic_frontier(&problem)
        .unwrap()
        .policies
        .into_iter()
        .min_by_key(|policy| policy.total_loss_units)
        .unwrap();
    assert_eq!(
        (best.total_variable_payload_bytes, best.total_loss_units),
        (1_980, 280)
    );
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

#[test]
fn exact_frontier_matches_brute_force_metrics() {
    let problem = make_problem(
        vec![
            unit("a", &[("q4", 10, 12, 9, 5), ("q6", 15, 6, 12, 4)]),
            unit("b", &[("q4", 8, 10, 6, 8), ("q8", 17, 1, 13, 3)]),
            unit("c", &[("q4", 7, 11, 4, 9), ("q6", 12, 4, 9, 4)]),
        ],
        472,
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
                if bytes > problem.variable_payload_budget_bytes {
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
        272,
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
        272,
        2,
    ))
    .unwrap_err();
    assert!(matches!(
        error,
        DynamicAllocationError::FrontierLimitExceeded {
            states: 3,
            max_states: 2,
            ..
        }
    ));
}

#[test]
fn unit_option_regime_and_manifest_order_do_not_change_bytes() {
    let mut problem = make_problem(
        vec![
            unit("a", &[("q4", 10, 12, 9, 5), ("q6", 15, 6, 12, 4)]),
            unit("b", &[("q4", 8, 10, 6, 8), ("q8", 17, 1, 13, 3)]),
        ],
        320,
        64,
    );
    let expected = allocate_dynamic_frontier(&problem).unwrap();
    problem.units.reverse();
    problem.execution_manifest_catalog.reverse();
    for unit in &mut problem.units {
        unit.options.reverse();
    }
    problem.required_regimes.reverse();
    assert_eq!(expected, allocate_dynamic_frontier(&problem).unwrap());
}

#[test]
fn rejects_incomplete_catalog_and_stale_lineage_projection() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.expected_tensor_count += 1;
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.units[0].options[0].execution_plans[0].lineage_slice_sha256 = digest("stale");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn rejects_manifest_payload_capability_and_storage_substitution() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 155, 8);
    problem.units[0].options[0].payload_bytes += 1;
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.units[0].options[0].operations[0].capability_binding_bundle_sha256 = digest("fake");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.units[0].options[0].storage_manifest_receipt_sha256 = digest("fake");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.units[0].options[0].operations[0]
        .regime_costs
        .get_mut(&InferenceRegime::TextDecodeM1)
        .unwrap()
        .runtime_binding_ids = vec!["fabricated-binding".into()];
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.units[0].options[0].operations[0]
        .regime_costs
        .get_mut(&InferenceRegime::TextDecodeM1)
        .unwrap()
        .runtime_binding_bundle_sha256 = digest("fabricated-regime-bundle");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn every_option_shares_the_exact_runtime_and_execution_scope() {
    let mut problem = make_problem(
        vec![unit("a", &[("q4", 10, 10, 5, 5), ("q6", 8, 8, 6, 6)])],
        210,
        8,
    );
    problem.tensor_runtime.hardware_profile_sha256 = digest("different-hardware");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.execution_scope.profile = "different-graph-scope".into();
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(
        vec![unit("a", &[("q4", 10, 10, 5, 5), ("q6", 8, 8, 6, 6)])],
        210,
        8,
    );
    let manifest = &mut problem.execution_manifest_catalog[1];
    manifest.runtime.routing_policy_sha256 = digest("different-routing-policy");
    manifest.manifest_sha256 = tensor_execution_manifest_sha256(manifest).unwrap();
    problem.execution_manifest_catalog_sha256 =
        execution_manifest_catalog_sha256(&problem.execution_manifest_catalog).unwrap();
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn packed_expert_unit_requires_explicit_per_expert_coverage() {
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
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
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.units[0].options[0].capability_profile_sha256 = digest("other-capability");
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    let mut extra_cost = problem.units[0].options[0].operations[0].regime_costs
        [&InferenceRegime::TextDecodeM1]
        .clone();
    extra_cost.regime = InferenceRegime::LongContextDecode;
    problem.units[0].options[0].operations[0]
        .regime_costs
        .insert(InferenceRegime::LongContextDecode, extra_cost);
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    problem.units[0].options[0].operations[0]
        .regime_costs
        .remove(&InferenceRegime::TextPrefill);
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));

    let mut problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    let manifest = &mut problem.execution_manifest_catalog[0];
    manifest.runtime.capability_profile_sha256 = digest("other-capability");
    manifest.manifest_sha256 = tensor_execution_manifest_sha256(manifest).unwrap();
    let validated = verify_tensor_execution_manifest(manifest).unwrap();
    let slice = tensor_lineage_slice(&validated, &problem.units[0].members[0].name).unwrap();
    problem.units[0].options[0].execution_plans[0].execution_manifest_sha256 =
        manifest.manifest_sha256.clone();
    problem.units[0].options[0].execution_plans[0].lineage_slice_sha256 = slice.slice_sha256;
    problem.execution_manifest_catalog_sha256 =
        execution_manifest_catalog_sha256(&problem.execution_manifest_catalog).unwrap();
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn one_physical_manifest_cannot_back_two_candidate_options() {
    let mut problem = make_problem(
        vec![unit("a", &[("q4", 10, 10, 5, 5), ("q6", 10, 8, 6, 6)])],
        210,
        8,
    );
    let q4 = problem.units[0].options[0].clone();
    problem.units[0].options[1] = TensorOption {
        option_id: "q6-alias".into(),
        ..q4
    };
    assert!(matches!(
        allocate_dynamic_frontier(&problem),
        Err(DynamicAllocationError::InvalidProblem(_))
    ));
}

#[test]
fn two_source_atomic_concatenation_counts_one_stored_payload() {
    let left = member_with_rows("gate", 1);
    let right = member_with_rows("up", 1);
    let (_, mut manifest) = option("q4", &left, 0, 10, 5, 5);
    let left_source = manifest
        .nodes
        .iter()
        .find(|node| node.stage == TensorStateStage::Source)
        .unwrap()
        .clone();
    let mut right_source = left_source.clone();
    right_source.node_id = format!("source:{}:q4", right.name);
    right_source.semantic_name = right.name.clone();
    right_source.byte_sha256 = right.source_tensor_sha256.clone();
    right_source.logical_f32_sha256 = digest("right-source-logical");
    right_source.artifact_region = Some(ArtifactRegion {
        artifact_id: "source-artifact:fused-right".into(),
        byte_offset: 0,
        byte_len: right_source.byte_len,
    });
    manifest.artifacts.push(ArtifactEvidence {
        artifact_id: "source-artifact:fused-right".into(),
        role: "source".into(),
        byte_len: right_source.byte_len,
        sha256: right.source_tensor_sha256.clone(),
    });
    let left_converted = manifest
        .nodes
        .iter()
        .find(|node| node.stage == TensorStateStage::Converted && node.node_id.ends_with(":q4:f32"))
        .unwrap()
        .clone();
    let mut right_converted = left_converted.clone();
    right_converted.node_id = format!("converted:{}:q4:f32", right.name);
    right_converted.semantic_name = format!("{}.converted", right.name);
    right_converted.byte_sha256 = digest("right-converted");
    right_converted.logical_f32_sha256 = right_source.logical_f32_sha256.clone();
    let concatenated = TensorStateNode {
        node_id: "converted:fused-gate-up".into(),
        stage: TensorStateStage::Converted,
        semantic_name: "fused-gate-up".into(),
        shape: vec![2, 256],
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::F32,
        },
        byte_len: 2_048,
        byte_sha256: digest("fused-converted"),
        logical_f32_sha256: digest("fused-converted-logical"),
        artifact_region: None,
        ..left_source.clone()
    };
    manifest.nodes.extend([
        right_source.clone(),
        right_converted.clone(),
        concatenated.clone(),
    ]);
    let roundtripped = manifest
        .nodes
        .iter_mut()
        .find(|node| node.node_id.ends_with(":q4:f16-roundtrip"))
        .unwrap();
    roundtripped.shape = vec![2, 256];
    roundtripped.byte_len = 2_048;
    roundtripped.logical_f32_sha256 = digest("fused-roundtripped-logical");
    for node in manifest.nodes.iter_mut().filter(|node| {
        matches!(
            node.stage,
            TensorStateStage::Stored | TensorStateStage::Loaded | TensorStateStage::Executed
        )
    }) {
        node.shape = vec![2, 256];
        node.byte_len = 288;
        if let Some(region) = &mut node.artifact_region {
            region.byte_len = 288;
        }
    }
    manifest
        .artifacts
        .iter_mut()
        .find(|artifact| artifact.role == "converted")
        .unwrap()
        .byte_len = 288;
    let roundtrip = manifest
        .transforms
        .iter_mut()
        .find(|edge| matches!(edge.operation, TensorTransformOperation::F16Roundtrip))
        .unwrap();
    roundtrip.inputs = vec![TransformPort {
        role: "converted".into(),
        node_id: concatenated.node_id.clone(),
    }];
    manifest.transforms.extend([
        TensorTransformEdge {
            edge_id: "decode:fused-up".into(),
            inputs: vec![TransformPort {
                role: "source".into(),
                node_id: right_source.node_id.clone(),
            }],
            outputs: vec![TransformPort {
                role: "converted".into(),
                node_id: right_converted.node_id.clone(),
            }],
            operation: TensorTransformOperation::SourceDecode,
            implementation_revision: digest("hf2q-revision")[..40].into(),
            receipt_sha256: digest("decode-up-receipt"),
        },
        TensorTransformEdge {
            edge_id: "concatenate:fused-gate-up".into(),
            inputs: vec![
                TransformPort {
                    role: "gate".into(),
                    node_id: left_converted.node_id.clone(),
                },
                TransformPort {
                    role: "up".into(),
                    node_id: right_converted.node_id.clone(),
                },
            ],
            outputs: vec![TransformPort {
                role: "fused".into(),
                node_id: concatenated.node_id.clone(),
            }],
            operation: TensorTransformOperation::Concatenate { axis: 0 },
            implementation_revision: digest("hf2q-revision")[..40].into(),
            receipt_sha256: digest("fuse-gate-up-receipt"),
        },
    ]);
    let operation_id = format!("op-{}", left.name);
    for binding in &mut manifest.operations {
        binding.source_tensor_names = vec![left.name.clone(), right.name.clone()];
    }
    let terminal_node_id = manifest.dispositions[0].terminal_node_ids[0].clone();
    manifest.dispositions.push(SourceTensorDisposition {
        source_node_id: right_source.node_id,
        disposition: SourceTensorDispositionKind::Variable,
        reason: "atomic fused candidate".into(),
        terminal_node_ids: vec![terminal_node_id],
    });
    manifest.manifest_sha256 = tensor_execution_manifest_sha256(&manifest).unwrap();
    let validated = verify_tensor_execution_manifest(&manifest).unwrap();
    let left_slice = tensor_lineage_slice(&validated, &left.name).unwrap();
    let right_slice = tensor_lineage_slice(&validated, &right.name).unwrap();
    let executed_nodes: BTreeMap<_, _> = manifest
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Executed)
        .map(|node| (node.node_id.clone(), node.clone()))
        .collect();
    let executed_node_ids = executed_nodes.keys().cloned().collect();
    let final_bundle_sha256 = {
        use sha2::{Digest, Sha256};
        hex::encode(Sha256::digest(serde_json::to_vec(&executed_nodes).unwrap()))
    };
    let stored_bytes: u64 = manifest
        .nodes
        .iter()
        .filter(|node| node.stage == TensorStateStage::Stored)
        .map(|node| node.byte_len)
        .sum();
    assert_eq!(stored_bytes, 288);
    let candidate = TensorOption {
        option_id: "fused-q4".into(),
        execution_plans: vec![
            TensorExecutionPlan {
                source_tensor_name: left.name.clone(),
                execution_manifest_sha256: manifest.manifest_sha256.clone(),
                lineage_slice_sha256: left_slice.slice_sha256,
            },
            TensorExecutionPlan {
                source_tensor_name: right.name.clone(),
                execution_manifest_sha256: manifest.manifest_sha256.clone(),
                lineage_slice_sha256: right_slice.slice_sha256,
            },
        ],
        operations: vec![OperationExecutionEvidence {
            operation_id: operation_id.clone(),
            graph_path: format!("qwen35.{}", left.name),
            source_tensor_names: vec![left.name.clone(), right.name.clone()],
            executed_tensor_node_ids: executed_node_ids,
            capability_binding_bundle_sha256: runtime_capability_binding_bundle_sha256(
                &validated,
                &operation_id,
            )
            .unwrap(),
            regime_costs: BTreeMap::from([
                (
                    InferenceRegime::TextDecodeM1,
                    regime_cost(
                        "fused",
                        InferenceRegime::TextDecodeM1,
                        5,
                        &validated,
                        &operation_id,
                    ),
                ),
                (
                    InferenceRegime::TextPrefill,
                    regime_cost(
                        "fused",
                        InferenceRegime::TextPrefill,
                        5,
                        &validated,
                        &operation_id,
                    ),
                ),
            ]),
        }],
        payload_bytes: stored_bytes,
        storage_manifest_receipt_sha256: manifest.conversion_receipt_sha256.clone(),
        sensitivity: SensitivityEvidence {
            calibration_manifest_sha256: digest("calibration"),
            sensitivity_receipt_sha256: digest("fused-sensitivity"),
            loss_units: 10,
            imatrix_weighted_error_units: 10,
            teacher_kl_alignment_units: 0,
            block_output_error_units: 10,
            uncertainty_units: 1,
            activation_rows: 4_096,
            expert_activation_rows: BTreeMap::new(),
            final_executed_tensor_bundle_sha256: final_bundle_sha256,
        },
        capability_profile_sha256: digest("capability-profile"),
    };
    let fixture = UnitFixture {
        unit: TensorAllocationUnit {
            unit_id: "fused-gate-up".into(),
            members: vec![left.clone(), right.clone()],
            expected_expert_ids: Vec::new(),
            operations: vec![TensorOperation {
                operation_id,
                graph_path: format!("qwen35.{}", left.name),
                tensor_names: vec![left.name, right.name],
            }],
            options: vec![candidate],
        },
        manifests: vec![manifest],
    };
    let problem = make_problem(vec![fixture], stored_bytes, 8);
    let frontier = allocate_dynamic_frontier(&problem).unwrap();
    assert_eq!(
        frontier.policies[0].total_variable_payload_bytes,
        stored_bytes
    );
    assert_eq!(
        stored_payload_bytes(&problem, &problem.units[0].options[0]).unwrap(),
        stored_bytes
    );
}

#[test]
fn canonical_frontier_round_trip_and_mutation_validation() {
    let problem = make_problem(
        vec![
            unit("a", &[("q4", 10, 10, 5, 8), ("q6", 15, 5, 9, 4)]),
            unit("b", &[("q4", 10, 10, 8, 5), ("q6", 15, 5, 4, 9)]),
        ],
        318,
        32,
    );
    let frontier = allocate_dynamic_frontier(&problem).unwrap();
    let bytes = canonical_frontier_bytes(&frontier).unwrap();
    let decoded: PolicyFrontier = serde_json::from_slice(&bytes).unwrap();
    validate_policy_frontier(&problem, &decoded).unwrap();
    let mut reordered = decoded.clone();
    reordered.policies.reverse();
    for policy in &mut reordered.policies {
        policy.decisions.reverse();
    }
    assert_eq!(canonical_frontier_bytes(&reordered).unwrap(), bytes);
    let mut tampered = decoded;
    tampered.policies[0].total_loss_units += 1;
    assert_eq!(
        validate_policy_frontier(&problem, &tampered),
        Err(DynamicAllocationError::FrontierMismatch)
    );
}

#[test]
fn catalog_manifest_body_reordering_is_canonical() {
    let problem = make_problem(vec![unit("a", &[("q4", 10, 10, 5, 5)])], 154, 8);
    let expected = allocation_problem_sha256(&problem).unwrap();
    let mut reordered = problem.clone();
    reordered.execution_manifest_catalog[0].nodes.reverse();
    reordered.execution_manifest_catalog[0].transforms.reverse();
    assert_eq!(allocation_problem_sha256(&reordered).unwrap(), expected);
}

#[test]
fn qwen_scale_catalog_stays_bounded_when_local_options_are_dominated() {
    let fixtures = (0..128)
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
    let frontier = allocate_dynamic_frontier(&make_problem(fixtures, 128 * 154, 8)).unwrap();
    assert_eq!(frontier.policies.len(), 1);
    assert_eq!(frontier.search_receipt.peak_frontier_states, 1);
    assert_eq!(frontier.search_receipt.states_generated, 384);
}
