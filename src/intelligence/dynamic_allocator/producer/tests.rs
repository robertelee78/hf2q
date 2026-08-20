use std::collections::BTreeMap;

use safetensors::tensor::TensorView;
use safetensors::Dtype;
use sha1::Sha1;
use sha2::{Digest, Sha256};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

use super::*;
use crate::core::integrity::ShardIntegrity;
use crate::core::provenance::{compute_source_bundle_sha256, SourceShard};
use crate::input::integrity::VerifiedSourceManifest;
use crate::intelligence::calibration::{
    build_structured_dataset_manifest, render_and_tokenize_split, verify_dataset_partition,
    DatasetSplit, ExampleProvenance, RenderDatasetRequest, RenderMode, StructuredExample,
};
use crate::intelligence::dynamic_allocator::*;
use crate::intelligence::measured_auto_quant::{
    ExecutionIdentity, InferenceRegime, SourceIdentity,
};
use crate::serve::api::schema::{ChatMessage, MessageContent};

fn digest(label: &str) -> String {
    super::partition::producer_hash(&label).unwrap()
}

struct SourceFixture {
    _temp: tempfile::TempDir,
    source: SourceIdentity,
    verified_files: VerifiedSourceManifest,
    inventory: VerifiedSourceTensorInventory,
}

fn source_fixture() -> SourceFixture {
    let temp = tempfile::tempdir().unwrap();
    let config = br#"{"model_type":"qwen3_5"}"#;
    let template = "{% for message in messages %} {{ message.role }} {{ message.content }}{% endfor %}{% if add_generation_prompt %} assistant{% endif %}";
    std::fs::write(temp.path().join("config.json"), config).unwrap();
    std::fs::write(temp.path().join("chat_template.jinja"), template).unwrap();
    let words = ["<unk>", "user", "assistant", "cal", "val", "hold"];
    let vocab: serde_json::Map<String, serde_json::Value> = words
        .iter()
        .enumerate()
        .map(|(index, word)| ((*word).into(), serde_json::json!(index)))
        .collect();
    let vocab_path = temp.path().join("vocab.json");
    std::fs::write(&vocab_path, serde_json::to_vec(&vocab).unwrap()).unwrap();
    let model = WordLevel::from_file(vocab_path.to_str().unwrap(), "<unk>".into()).unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
        .save(temp.path().join("tokenizer.json"), false)
        .unwrap();
    std::fs::remove_file(vocab_path).unwrap();

    let git_record = |filename: &str, bytes: &[u8]| {
        let mut git = Sha1::new();
        git.update(format!("blob {}\0", bytes.len()).as_bytes());
        git.update(bytes);
        ShardIntegrity {
            filename: filename.into(),
            bytes: bytes.len() as u64,
            sha256: None,
            hf_etag: hex::encode(git.finalize()),
            is_lfs: false,
        }
    };
    let tokenizer_bytes = std::fs::read(temp.path().join("tokenizer.json")).unwrap();
    let config_record = git_record("config.json", config);
    let tokenizer_record = git_record("tokenizer.json", &tokenizer_bytes);
    let template_record = git_record("chat_template.jinja", template.as_bytes());
    let embed = vec![0u8; 8 * 16 * 2];
    let output = vec![1u8; 8 * 16 * 2];
    let experts = vec![2u8; 2 * 8 * 8 * 2];
    let embed_view = TensorView::new(Dtype::BF16, vec![8, 16], &embed).unwrap();
    let output_view = TensorView::new(Dtype::BF16, vec![8, 16], &output).unwrap();
    let expert_view = TensorView::new(Dtype::BF16, vec![2, 8, 8], &experts).unwrap();
    let bytes = safetensors::tensor::serialize(
        vec![
            ("model.embed_tokens.weight".to_owned(), &embed_view),
            ("lm_head.weight".to_owned(), &output_view),
            ("model.layers.0.experts.weight".to_owned(), &expert_view),
        ],
        None,
    )
    .unwrap();
    std::fs::write(temp.path().join("model.safetensors"), &bytes).unwrap();
    let weight_sha = hex::encode(Sha256::digest(&bytes));
    let weight_record = ShardIntegrity {
        filename: "model.safetensors".into(),
        bytes: bytes.len() as u64,
        sha256: Some(weight_sha.clone()),
        hf_etag: weight_sha,
        is_lfs: true,
    };
    let verified_files = crate::input::integrity::verify_conversion_manifest(
        "Qwen/Qwen3.8-27B",
        "revision",
        temp.path(),
        vec![
            config_record,
            tokenizer_record,
            template_record,
            weight_record.clone(),
        ],
    )
    .unwrap();
    let tensor_bundle_sha256 =
        compute_source_bundle_sha256(&[SourceShard::from_integrity(&weight_record)]).unwrap();
    let render_inputs = crate::core::chat_template_resolver::resolve_chat_render_inputs(
        temp.path(),
        &tokenizer_bytes,
        "qwen35",
    )
    .unwrap();
    let source = SourceIdentity {
        model_id: "Qwen/Qwen3.8-27B".into(),
        revision: "revision".into(),
        config_sha256: hex::encode(Sha256::digest(config)),
        tensor_bundle_sha256,
        tokenizer_bundle_sha256: render_inputs.tokenizer_bundle_sha256,
        chat_template_sha256: hex::encode(Sha256::digest(template.as_bytes())),
    };
    let inventory =
        derive_source_tensor_inventory(temp.path(), source.clone(), &verified_files).unwrap();
    SourceFixture {
        _temp: temp,
        source,
        verified_files,
        inventory,
    }
}

fn member_from_record(
    inventory: &VerifiedSourceTensorInventory,
    name: &str,
    role: &str,
) -> TensorMember {
    let record = inventory
        .manifest()
        .tensors
        .iter()
        .find(|record| record.name == name)
        .unwrap();
    TensorMember {
        name: record.name.clone(),
        shape: record.source_shape.clone(),
        role: role.into(),
        source_dtype: ScalarDType::Bf16,
        source_tensor_sha256: record.source_tensor_sha256.clone(),
        layer_index: None,
        expert_index: None,
    }
}

fn units(inventory: &VerifiedSourceTensorInventory) -> Vec<TensorAllocationUnit> {
    vec![
        TensorAllocationUnit {
            unit_id: "experts".into(),
            members: vec![member_from_record(
                inventory,
                "model.layers.0.experts.weight",
                "expert",
            )],
            expected_expert_ids: vec![0, 1],
            operations: vec![TensorOperation {
                operation_id: "layer0-expert-input".into(),
                graph_path: "qwen35.layer0.ffn.experts".into(),
                tensor_names: vec!["model.layers.0.experts.weight".into()],
            }],
            options: Vec::new(),
        },
        TensorAllocationUnit {
            unit_id: "output".into(),
            members: vec![member_from_record(inventory, "lm_head.weight", "lm_head")],
            expected_expert_ids: Vec::new(),
            operations: vec![TensorOperation {
                operation_id: "lm-head-input".into(),
                graph_path: "qwen35.output_head".into(),
                tensor_names: vec!["lm_head.weight".into()],
            }],
            options: Vec::new(),
        },
    ]
}

fn fixed_embedding(inventory: &VerifiedSourceTensorInventory) -> NonVariableTensor {
    NonVariableTensor {
        source: inventory
            .manifest()
            .tensors
            .iter()
            .find(|record| record.name == "model.embed_tokens.weight")
            .unwrap()
            .clone(),
        disposition: NonVariableDisposition::Protected,
        reason: "embedding is protected until the execution-truth manifest proves its route".into(),
    }
}

fn allocatable_units(
    inventory: &VerifiedSourceTensorInventory,
    calibration_manifest_sha256: &str,
) -> Vec<TensorAllocationUnit> {
    let mut units = units(inventory);
    for unit in &mut units {
        let codec = TensorCodec::Gguf {
            codec: GgufCodec::Q8_0,
        };
        let mut plans = Vec::new();
        for member in &unit.members {
            let stored_hash = digest(&format!("stored-{}", member.name));
            let logical_hash = digest(&format!("logical-{}", member.name));
            let operation_id = unit
                .operations
                .iter()
                .find(|operation| operation.tensor_names.contains(&member.name))
                .unwrap()
                .operation_id
                .clone();
            plans.push(TensorExecutionPlan {
                tensor_name: member.name.clone(),
                stored_codec: codec.clone(),
                stored_tensor_sha256: stored_hash.clone(),
                stored_payload_bytes: 256,
                executed_codec: codec.clone(),
                transformations: vec![ExecutionTransformStep {
                    kind: ExecutionTransformKind::Identity,
                    from: codec.clone(),
                    to: codec.clone(),
                    input_tensor_sha256: stored_hash.clone(),
                    transform_receipt_sha256: digest(&format!("transform-{}", member.name)),
                    output_tensor_sha256: stored_hash.clone(),
                    input_logical_tensor_sha256: logical_hash.clone(),
                    output_logical_tensor_sha256: logical_hash,
                }],
                final_executed_tensor_sha256: stored_hash,
                operation_id,
            });
        }
        let operations = unit
            .operations
            .iter()
            .map(|operation| OperationExecutionEvidence {
                operation_id: operation.operation_id.clone(),
                graph_path: operation.graph_path.clone(),
                tensor_names: operation.tensor_names.clone(),
                capability_decision_sha256: digest(&format!(
                    "capability-{}",
                    operation.operation_id
                )),
                regime_costs: BTreeMap::from([(
                    InferenceRegime::TextDecodeM1,
                    RegimeCost {
                        regime: InferenceRegime::TextDecodeM1,
                        workload_shape_sha256: digest("decode-shape"),
                        executable: true,
                        specialized_for_regime: true,
                        route: "q8-decode".into(),
                        invocation_count: 1,
                        median_nanoseconds: 10,
                        p95_nanoseconds: 12,
                        warmup_runs: 2,
                        measured_runs: 5,
                        measurement_receipt_sha256: digest(&format!(
                            "measurement-{}",
                            operation.operation_id
                        )),
                    },
                )]),
            })
            .collect();
        let payload_bytes = plans.iter().map(|plan| plan.stored_payload_bytes).sum();
        unit.options = vec![TensorOption {
            option_id: "q8".into(),
            execution_plans: plans.clone(),
            operations,
            payload_bytes,
            shared_metadata_bytes: 0,
            storage_manifest_receipt_sha256: digest(&format!("storage-{}", unit.unit_id)),
            sensitivity: SensitivityEvidence {
                calibration_manifest_sha256: calibration_manifest_sha256.into(),
                sensitivity_receipt_sha256: digest(&format!("sensitivity-{}", unit.unit_id)),
                loss_units: 1,
                imatrix_weighted_error_units: 1,
                teacher_kl_alignment_units: 0,
                block_output_error_units: 1,
                uncertainty_units: 1,
                activation_rows: 128,
                expert_activation_rows: unit
                    .expected_expert_ids
                    .iter()
                    .copied()
                    .map(|expert| (expert, 128))
                    .collect(),
                final_executed_tensor_bundle_sha256: final_executed_tensor_bundle_sha256(&plans)
                    .unwrap(),
            },
            capability_profile_sha256: digest("capability-profile"),
        }];
    }
    units
}

fn rendered_split(
    fixture: &SourceFixture,
    split: DatasetSplit,
    id: &str,
    content: &str,
) -> crate::intelligence::calibration::RenderedDataset {
    let structured = build_structured_dataset_manifest(
        "dynamic-test".into(),
        "v1".into(),
        "apache-2.0".into(),
        split,
        7,
        vec![StructuredExample {
            stable_id: id.into(),
            provenance: ExampleProvenance {
                dataset_id: "dynamic-test".into(),
                revision: "v1".into(),
                record_id: format!("record-{id}"),
                license: "apache-2.0".into(),
            },
            domains: vec!["agentic-coding".into()],
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Text(content.into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: Vec::new(),
            render_mode: RenderMode::GenerationPrompt,
            enable_thinking: false,
            chat_template_kwargs: BTreeMap::new(),
        }],
    )
    .unwrap();
    render_and_tokenize_split(
        &structured,
        &RenderDatasetRequest {
            model_dir: fixture._temp.path().into(),
            arch: "qwen35".into(),
            source: fixture.source.clone(),
            verified_source: fixture.verified_files.clone(),
            renderer_revision: "production-renderer-v1".into(),
            max_tokens_per_example: 16,
            token_window_size: 3,
        },
    )
    .unwrap()
}

#[test]
fn source_inventory_is_derived_and_independently_reproduced() {
    let fixture = source_fixture();
    validate_source_tensor_inventory(fixture.inventory.manifest()).unwrap();
    let reproduced = verify_source_tensor_inventory_from_source(
        fixture.inventory.manifest(),
        fixture._temp.path(),
        fixture.source.clone(),
        &fixture.verified_files,
    )
    .unwrap();
    assert_eq!(reproduced.manifest(), fixture.inventory.manifest());

    let mut omitted = fixture.inventory.manifest().clone();
    omitted.tensors.pop();
    omitted.manifest_sha256 = super::partition::inventory_hash(&omitted).unwrap();
    validate_source_tensor_inventory(&omitted).unwrap();
    assert!(matches!(
        verify_source_tensor_inventory_from_source(
            &omitted,
            fixture._temp.path(),
            fixture.source.clone(),
            &fixture.verified_files,
        ),
        Err(DynamicProducerError::InvalidInventory(_))
    ));

    let shard_path = fixture._temp.path().join("model.safetensors");
    let mut changed_bytes = std::fs::read(&shard_path).unwrap();
    let last = changed_bytes.last_mut().unwrap();
    *last ^= 1;
    std::fs::write(&shard_path, changed_bytes).unwrap();
    assert!(matches!(
        derive_source_tensor_inventory(
            fixture._temp.path(),
            fixture.source,
            &fixture.verified_files,
        ),
        Err(DynamicProducerError::InvalidInventory(_))
    ));
}

#[test]
fn source_inventory_rejects_duplicate_tensors_across_authenticated_shards() {
    let temp = tempfile::tempdir().unwrap();
    let config = br#"{"model_type":"qwen3_5"}"#;
    std::fs::write(temp.path().join("config.json"), config).unwrap();
    let data = vec![0u8; 16];
    let tensor = TensorView::new(Dtype::BF16, vec![2, 4], &data).unwrap();
    let first =
        safetensors::tensor::serialize(vec![("duplicate.weight".to_owned(), &tensor)], None)
            .unwrap();
    let second = first.clone();
    std::fs::write(temp.path().join("model-00001-of-00002.safetensors"), &first).unwrap();
    std::fs::write(
        temp.path().join("model-00002-of-00002.safetensors"),
        &second,
    )
    .unwrap();
    let index = br#"{"weight_map":{"logical.a":"model-00001-of-00002.safetensors","logical.b":"model-00002-of-00002.safetensors"}}"#;
    std::fs::write(temp.path().join("model.safetensors.index.json"), index).unwrap();

    let record = |filename: &str, bytes: &[u8], lfs: bool| {
        let sha256 = hex::encode(Sha256::digest(bytes));
        let mut git = Sha1::new();
        git.update(format!("blob {}\0", bytes.len()).as_bytes());
        git.update(bytes);
        ShardIntegrity {
            filename: filename.into(),
            bytes: bytes.len() as u64,
            sha256: lfs.then_some(sha256.clone()),
            hf_etag: if lfs {
                sha256
            } else {
                hex::encode(git.finalize())
            },
            is_lfs: lfs,
        }
    };
    let first_record = record("model-00001-of-00002.safetensors", &first, true);
    let second_record = record("model-00002-of-00002.safetensors", &second, true);
    let records = vec![
        record("config.json", config, false),
        record("model.safetensors.index.json", index, false),
        first_record.clone(),
        second_record.clone(),
    ];
    let verified = crate::input::integrity::verify_conversion_manifest(
        "Qwen/Qwen3.8-27B",
        "revision",
        temp.path(),
        records,
    )
    .unwrap();
    let tensor_bundle_sha256 = compute_source_bundle_sha256(&[
        SourceShard::from_integrity(&first_record),
        SourceShard::from_integrity(&second_record),
    ])
    .unwrap();
    let source = SourceIdentity {
        model_id: "Qwen/Qwen3.8-27B".into(),
        revision: "revision".into(),
        config_sha256: hex::encode(Sha256::digest(config)),
        tensor_bundle_sha256,
        tokenizer_bundle_sha256: digest("tokenizer"),
        chat_template_sha256: digest("template"),
    };
    assert!(matches!(
        derive_source_tensor_inventory(temp.path(), source, &verified),
        Err(DynamicProducerError::InvalidInventory(message))
            if message.contains("duplicated across source shards")
    ));
}

#[test]
fn source_partition_binds_atomic_grouping_and_is_complete() {
    let fixture = source_fixture();
    let allocation_units = units(&fixture.inventory);
    let partition = build_tensor_partition(
        &fixture.inventory,
        &allocation_units,
        vec![fixed_embedding(&fixture.inventory)],
    )
    .unwrap();
    validate_tensor_partition(&partition, &fixture.inventory, &allocation_units).unwrap();
    assert_eq!(partition.source_tensor_count, 3);
    assert_eq!(partition.variable_units.len(), 2);
    assert_eq!(partition.non_variable_tensors.len(), 1);

    let mut regrouped = allocation_units.clone();
    let output = regrouped[1].members.pop().unwrap();
    regrouped[0].members.push(output);
    assert!(matches!(
        validate_tensor_partition(&partition, &fixture.inventory, &regrouped),
        Err(DynamicProducerError::InvalidPartition(_))
    ));
    assert!(matches!(
        build_tensor_partition(&fixture.inventory, &allocation_units, Vec::new()),
        Err(DynamicProducerError::InvalidPartition(_))
    ));
}

#[test]
fn partition_rejects_drift_and_duplicate_expert_topology() {
    let fixture = source_fixture();
    let allocation_units = units(&fixture.inventory);
    let mut drifted = fixed_embedding(&fixture.inventory);
    drifted.source.source_shape[0] += 1;
    assert!(matches!(
        build_tensor_partition(&fixture.inventory, &allocation_units, vec![drifted]),
        Err(DynamicProducerError::InvalidPartition(_))
    ));

    let mut duplicate_expert = allocation_units;
    duplicate_expert[0].expected_expert_ids.push(1);
    assert!(matches!(
        build_tensor_partition(
            &fixture.inventory,
            &duplicate_expert,
            vec![fixed_embedding(&fixture.inventory)]
        ),
        Err(DynamicProducerError::InvalidPartition(_))
    ));
}

#[test]
fn coverage_binds_exact_taps_and_packed_experts() {
    let fixture = source_fixture();
    let allocation_units = units(&fixture.inventory);
    let partition = build_tensor_partition(
        &fixture.inventory,
        &allocation_units,
        vec![fixed_embedding(&fixture.inventory)],
    )
    .unwrap();
    let verified_topology = verify_collector_topology(&allocation_units).unwrap();
    let contract = build_coverage_contract(
        &partition,
        &fixture.inventory,
        digest("calibration-manifest"),
        "qwen35-collector-v1".into(),
        digest("collector-execution"),
        64,
        &allocation_units,
        &verified_topology,
    )
    .unwrap();

    let observations = vec![
        UnitCoverageObservation {
            unit_id: "output".into(),
            tensor_names: vec!["lm_head.weight".into()],
            collector_taps: vec![CollectorTapObservation {
                operation_id: "lm-head-input".into(),
                graph_path: "qwen35.output_head".into(),
                tensor_names: vec!["lm_head.weight".into()],
                activation_rows: 128,
                expert_activation_rows: BTreeMap::new(),
                activation_materialization_sha256: digest("output-activations"),
            }],
        },
        UnitCoverageObservation {
            unit_id: "experts".into(),
            tensor_names: vec!["model.layers.0.experts.weight".into()],
            collector_taps: vec![CollectorTapObservation {
                operation_id: "layer0-expert-input".into(),
                graph_path: "qwen35.layer0.ffn.experts".into(),
                tensor_names: vec!["model.layers.0.experts.weight".into()],
                activation_rows: 256,
                expert_activation_rows: BTreeMap::from([(0, 96), (1, 80)]),
                activation_materialization_sha256: digest("expert-activations"),
            }],
        },
    ];
    let receipt = verify_coverage_receipt(&contract, observations.clone()).unwrap();
    assert_eq!(receipt.observed_unit_count, 2);
    assert_eq!(receipt.observed_tensor_count, 2);

    let mut wrong_operation = observations.clone();
    wrong_operation[0].collector_taps[0].operation_id = "made-up".into();
    assert!(matches!(
        verify_coverage_receipt(&contract, wrong_operation),
        Err(DynamicProducerError::InvalidCoverage(_))
    ));
    let mut incomplete = observations;
    incomplete[1].collector_taps[0]
        .expert_activation_rows
        .remove(&1);
    assert!(matches!(
        verify_coverage_receipt(&contract, incomplete),
        Err(DynamicProducerError::InvalidCoverage(_))
    ));
}

#[test]
fn verified_dynamic_binding_chain_accepts_exact_evidence_and_rejects_substitution() {
    let fixture = source_fixture();
    let calibration = rendered_split(&fixture, DatasetSplit::Calibration, "cal", "cal");
    let validation = rendered_split(&fixture, DatasetSplit::PolicyValidation, "val", "val");
    let holdout = rendered_split(&fixture, DatasetSplit::AcceptanceHoldout, "hold", "hold");
    let dataset_partition = verify_dataset_partition(&calibration, &validation, &holdout).unwrap();
    let allocation_units = allocatable_units(
        &fixture.inventory,
        &dataset_partition.calibration_manifest_sha256,
    );
    let tensor_partition = build_tensor_partition(
        &fixture.inventory,
        &allocation_units,
        vec![fixed_embedding(&fixture.inventory)],
    )
    .unwrap();
    let topology = verify_collector_topology(&allocation_units).unwrap();
    let contract = build_coverage_contract(
        &tensor_partition,
        &fixture.inventory,
        dataset_partition.calibration_manifest_sha256.clone(),
        "qwen35-structural-collector-v1".into(),
        digest("collector-execution"),
        64,
        &allocation_units,
        &topology,
    )
    .unwrap();
    let observations = contract
        .units
        .iter()
        .map(|unit| UnitCoverageObservation {
            unit_id: unit.unit_id.clone(),
            tensor_names: unit.tensor_names.clone(),
            collector_taps: unit
                .collector_operations
                .iter()
                .map(|operation| CollectorTapObservation {
                    operation_id: operation.operation_id.clone(),
                    graph_path: operation.graph_path.clone(),
                    tensor_names: operation.tensor_names.clone(),
                    activation_rows: 128,
                    expert_activation_rows: unit
                        .expected_expert_ids
                        .iter()
                        .copied()
                        .map(|expert| (expert, 128))
                        .collect(),
                    activation_materialization_sha256: digest(&format!(
                        "materialization-{}",
                        operation.operation_id
                    )),
                })
                .collect(),
        })
        .collect();
    let receipt = verify_coverage_receipt(&contract, observations).unwrap();
    let problem = DynamicAllocationProblem {
        schema_version: DYNAMIC_ALLOCATION_SCHEMA_VERSION,
        source: fixture.source.clone(),
        execution: ExecutionIdentity {
            hf2q_revision: "test-revision".into(),
            mlx_native_version: "0.10.14".into(),
            hardware_id: "apple-test".into(),
            os_build: "test-os".into(),
        },
        tensor_catalog_sha256: tensor_partition.tensor_catalog_sha256.clone(),
        expected_tensor_count: allocation_units.iter().map(|unit| unit.members.len()).sum(),
        dataset_partition_manifest_sha256: dataset_partition.manifest_sha256.clone(),
        tensor_partition_manifest_sha256: tensor_partition.manifest_sha256.clone(),
        calibration_manifest_sha256: dataset_partition.calibration_manifest_sha256.clone(),
        sensitivity_model: SensitivityModelIdentity {
            method: "model-free-test".into(),
            version: "1".into(),
            fixed_point_scale: 1,
            component_weights_sha256: digest("component-weights"),
            coverage_contract_sha256: contract.contract_sha256.clone(),
            coverage_receipt_sha256: receipt.receipt_sha256.clone(),
        },
        capability_profile_sha256: digest("capability-profile"),
        proposal_workload_profile_sha256: digest("workload-profile"),
        required_regimes: vec![InferenceRegime::TextDecodeM1],
        payload_budget_bytes: 1024,
        minimum_expert_activation_rows: 64,
        search: SearchContract::ExactPareto { max_states: 16 },
        units: allocation_units,
    };

    let admit = |problem: &DynamicAllocationProblem,
                 contract: &CoverageContract,
                 receipt: &CoverageReceipt| {
        validate_dynamic_allocation_bindings(
            problem,
            &dataset_partition,
            &calibration,
            &validation,
            &holdout,
            &tensor_partition,
            &fixture.inventory,
            &topology,
            contract,
            receipt,
        )
    };
    let verified = admit(&problem, &contract, &receipt).unwrap();
    assert_eq!(
        allocate_verified_dynamic_frontier(&verified)
            .unwrap()
            .policies
            .len(),
        1
    );

    let mut wrong_source = problem.clone();
    wrong_source.source.model_id = "other/model".into();
    assert!(admit(&wrong_source, &contract, &receipt).is_err());

    let mut regrouped = problem.clone();
    let moved = regrouped.units[1].members.pop().unwrap();
    regrouped.units[0].members.push(moved);
    assert!(admit(&regrouped, &contract, &receipt).is_err());

    let mut wrong_operation = problem.clone();
    wrong_operation.units[0].options[0].operations[0].operation_id = "substituted-op".into();
    assert!(admit(&wrong_operation, &contract, &receipt).is_err());

    let mut wrong_contract = contract.clone();
    wrong_contract.collector_revision = "substituted-collector".into();
    assert!(admit(&problem, &wrong_contract, &receipt).is_err());

    let mut wrong_receipt = receipt.clone();
    wrong_receipt.observations[0].collector_taps[0].activation_materialization_sha256 =
        digest("substituted-materialization");
    assert!(admit(&problem, &contract, &wrong_receipt).is_err());
}
