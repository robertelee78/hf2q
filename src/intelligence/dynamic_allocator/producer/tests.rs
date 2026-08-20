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
use crate::core::provenance::{compute_source_bundle_sha256, tensor_execution::*, SourceShard};
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

fn execution_manifest(
    unit: &TensorAllocationUnit,
    member: &TensorMember,
    tensor_partition_manifest_sha256: &str,
    verified_source_manifest_sha256: &str,
    source_tensor_inventory_sha256: &str,
) -> TensorExecutionManifest {
    let source_elements = member.shape.iter().product::<usize>() as u64;
    let row_width = *member.shape.last().unwrap() as u64;
    let outer = source_elements / row_width;
    let padded_row_width = row_width.div_ceil(32) * 32;
    let mut padded_shape: Vec<u64> = member.shape.iter().map(|dim| *dim as u64).collect();
    *padded_shape.last_mut().unwrap() = padded_row_width;
    let source_node = TensorStateNode {
        node_id: format!("source:{}", member.name),
        stage: TensorStateStage::Source,
        semantic_name: member.name.clone(),
        shape: member.shape.iter().map(|dim| *dim as u64).collect(),
        layout: "row-major-outermost-first-v1".into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::Bf16,
        },
        byte_len: source_elements * 2,
        byte_sha256: member.source_tensor_sha256.clone(),
        logical_f32_sha256: digest(&format!("source-logical-{}", member.name)),
        artifact_region: Some(ArtifactRegion {
            artifact_id: format!("source-artifact-{}", member.name),
            byte_offset: 0,
            byte_len: source_elements * 2,
        }),
    };
    let decoded = TensorStateNode {
        node_id: format!("decoded:{}", member.name),
        stage: TensorStateStage::Converted,
        semantic_name: member.name.clone(),
        shape: member.shape.iter().map(|dim| *dim as u64).collect(),
        layout: "row-major-outermost-first-v1".into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::F32,
        },
        byte_len: source_elements * 4,
        byte_sha256: digest(&format!("decoded-{}", member.name)),
        logical_f32_sha256: source_node.logical_f32_sha256.clone(),
        artifact_region: None,
    };
    let converted = TensorStateNode {
        node_id: format!("converted:{}", member.name),
        stage: TensorStateStage::Converted,
        semantic_name: member.name.clone(),
        shape: padded_shape.clone(),
        layout: "row-major-outermost-first-v1".into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::F32,
        },
        byte_len: outer * padded_row_width * 4,
        byte_sha256: digest(&format!("converted-{}", member.name)),
        logical_f32_sha256: digest(&format!("converted-logical-{}", member.name)),
        artifact_region: None,
    };
    let roundtripped = TensorStateNode {
        node_id: format!("roundtripped:{}", member.name),
        semantic_name: member.name.clone(),
        byte_sha256: digest(&format!("roundtripped-{}", member.name)),
        logical_f32_sha256: digest(&format!("roundtripped-logical-{}", member.name)),
        ..converted.clone()
    };
    let stored_bytes = outer * (padded_row_width / 32) * 34;
    let stored_hash = digest(&format!("stored-{}", member.name));
    let stored = TensorStateNode {
        node_id: format!("stored:{}", member.name),
        stage: TensorStateStage::Stored,
        semantic_name: member.name.clone(),
        shape: padded_shape,
        layout: "row-major-outermost-first-v1".into(),
        codec: PhysicalTensorCodec::Ggml {
            wire_type_id: 8,
            type_name: "Q8_0".into(),
        },
        byte_len: stored_bytes,
        byte_sha256: stored_hash.clone(),
        logical_f32_sha256: digest(&format!("stored-logical-{}", member.name)),
        artifact_region: Some(ArtifactRegion {
            artifact_id: format!("gguf-artifact-{}", member.name),
            byte_offset: 0,
            byte_len: stored_bytes,
        }),
    };
    let loaded = TensorStateNode {
        node_id: format!("loaded:{}", member.name),
        stage: TensorStateStage::Loaded,
        artifact_region: None,
        ..stored.clone()
    };
    let executed = TensorStateNode {
        node_id: format!("executed:{}", member.name),
        stage: TensorStateStage::Executed,
        artifact_region: None,
        ..stored.clone()
    };
    let operation = unit
        .operations
        .iter()
        .find(|operation| operation.tensor_names.contains(&member.name))
        .unwrap();
    let revision = &digest("producer-revision")[..40];
    let json = CanonicalJsonEvidence {
        canonical_json: "{}".into(),
        sha256: hex::encode(Sha256::digest(b"{}")),
    };
    let mut manifest = TensorExecutionManifest {
        schema_version: TENSOR_EXECUTION_MANIFEST_SCHEMA_VERSION,
        source_manifest_sha256: verified_source_manifest_sha256.into(),
        source_tensor_inventory_sha256: source_tensor_inventory_sha256.into(),
        tensor_partition_manifest_sha256: tensor_partition_manifest_sha256.into(),
        conversion_receipt_sha256: digest(&format!("storage-{}", unit.unit_id)),
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
            profile: "test".into(),
            included_paths: vec!["qwen35.autoregressive_text".into()],
            excluded_paths: BTreeMap::new(),
        },
        artifacts: vec![
            ArtifactEvidence {
                artifact_id: format!("source-artifact-{}", member.name),
                role: "source".into(),
                byte_len: source_elements * 2,
                sha256: member.source_tensor_sha256.clone(),
            },
            ArtifactEvidence {
                artifact_id: format!("gguf-artifact-{}", member.name),
                role: "converted".into(),
                byte_len: stored_bytes,
                sha256: digest(&format!("gguf-{}", member.name)),
            },
        ],
        nodes: vec![
            source_node.clone(),
            decoded.clone(),
            converted.clone(),
            roundtripped.clone(),
            stored.clone(),
            loaded.clone(),
            executed.clone(),
        ],
        transforms: vec![
            TensorTransformEdge {
                edge_id: format!("decode:{}", member.name),
                inputs: vec![TransformPort {
                    role: "source".into(),
                    node_id: source_node.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: decoded.node_id.clone(),
                }],
                operation: TensorTransformOperation::SourceDecode,
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("decode-{}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("pad:{}", member.name),
                inputs: vec![TransformPort {
                    role: "input".into(),
                    node_id: decoded.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "output".into(),
                    node_id: converted.node_id.clone(),
                }],
                operation: TensorTransformOperation::ZeroPad {
                    axis: u32::try_from(member.shape.len() - 1).unwrap(),
                    before: 0,
                    after: padded_row_width - row_width,
                },
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("pad-{}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("roundtrip:{}", member.name),
                inputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: converted.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "converted_roundtripped".into(),
                    node_id: roundtripped.node_id.clone(),
                }],
                operation: TensorTransformOperation::F16Roundtrip,
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("roundtrip-{}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("quantize:{}", member.name),
                inputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: roundtripped.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "stored".into(),
                    node_id: stored.node_id.clone(),
                }],
                operation: TensorTransformOperation::GgufQuantize {
                    implementation_id: "test-q8".into(),
                    calibration_receipt_sha256: None,
                },
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("quantize-{}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("load:{}", member.name),
                inputs: vec![TransformPort {
                    role: "stored".into(),
                    node_id: stored.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "loaded".into(),
                    node_id: loaded.node_id.clone(),
                }],
                operation: TensorTransformOperation::DirectBlockLoad,
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("load-{}", member.name)),
            },
            TensorTransformEdge {
                edge_id: format!("bind:{}", member.name),
                inputs: vec![TransformPort {
                    role: "loaded".into(),
                    node_id: loaded.node_id.clone(),
                }],
                outputs: vec![TransformPort {
                    role: "executed".into(),
                    node_id: executed.node_id.clone(),
                }],
                operation: TensorTransformOperation::RuntimeBind,
                implementation_revision: revision.into(),
                receipt_sha256: digest(&format!("bind-{}", member.name)),
            },
        ],
        operations: vec![RuntimeOperationBinding {
            binding_id: format!("binding:{}", operation.operation_id),
            operation_id: operation.operation_id.clone(),
            graph_path: operation.graph_path.clone(),
            entrypoint: "quantized_matmul_ggml_with_policy".into(),
            workload_regime: TensorExecutionRegime::TextDecodeM1,
            invocation_count: 1,
            source_tensor_names: vec![member.name.clone()],
            inputs: vec![TransformPort {
                role: "weight".into(),
                node_id: executed.node_id.clone(),
            }],
            capability: RuntimeCapabilityEvidence::Ggml {
                request: json.clone(),
                decision: json,
                requires_device_probe: false,
                resolved_runtime_trace: None,
            },
        }],
        dispositions: vec![SourceTensorDisposition {
            source_node_id: source_node.node_id,
            disposition: SourceTensorDispositionKind::Variable,
            reason: "test candidate".into(),
            terminal_node_ids: vec![executed.node_id],
        }],
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = tensor_execution_manifest_sha256(&manifest).unwrap();
    manifest
}

fn allocatable_units(
    mut units: Vec<TensorAllocationUnit>,
    inventory: &VerifiedSourceTensorInventory,
    tensor_partition_manifest_sha256: &str,
    calibration_manifest_sha256: &str,
) -> (Vec<TensorAllocationUnit>, Vec<TensorExecutionManifest>) {
    let mut manifests = Vec::new();
    for unit in &mut units {
        let mut plans = Vec::new();
        for member in &unit.members {
            let manifest = execution_manifest(
                unit,
                member,
                tensor_partition_manifest_sha256,
                &inventory.manifest().verified_source_manifest_sha256,
                &inventory.manifest().manifest_sha256,
            );
            let validated = verify_tensor_execution_manifest(&manifest).unwrap();
            let slice = tensor_lineage_slice(&validated, &member.name).unwrap();
            plans.push(TensorExecutionPlan {
                source_tensor_name: member.name.clone(),
                execution_manifest_sha256: manifest.manifest_sha256.clone(),
                lineage_slice_sha256: slice.slice_sha256,
            });
            manifests.push(manifest);
        }
        let operations = unit
            .operations
            .iter()
            .map(|operation| {
                let manifest = manifests
                    .iter()
                    .find(|manifest| {
                        manifest
                            .operations
                            .iter()
                            .any(|binding| binding.operation_id == operation.operation_id)
                    })
                    .unwrap();
                let validated = verify_tensor_execution_manifest(manifest).unwrap();
                let binding = manifest
                    .operations
                    .iter()
                    .find(|binding| binding.operation_id == operation.operation_id)
                    .unwrap();
                OperationExecutionEvidence {
                    operation_id: operation.operation_id.clone(),
                    graph_path: operation.graph_path.clone(),
                    source_tensor_names: operation.tensor_names.clone(),
                    executed_tensor_node_ids: binding
                        .inputs
                        .iter()
                        .map(|input| input.node_id.clone())
                        .collect(),
                    capability_binding_bundle_sha256: runtime_capability_binding_bundle_sha256(
                        &validated,
                        &operation.operation_id,
                    )
                    .unwrap(),
                    regime_costs: BTreeMap::from([(
                        InferenceRegime::TextDecodeM1,
                        RegimeCost {
                            regime: InferenceRegime::TextDecodeM1,
                            runtime_binding_ids: vec![binding.binding_id.clone()],
                            runtime_binding_bundle_sha256: runtime_regime_binding_bundle_sha256(
                                &validated,
                                &operation.operation_id,
                                TensorExecutionRegime::TextDecodeM1,
                            )
                            .unwrap(),
                            executable: true,
                            specialized_for_regime: true,
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
                }
            })
            .collect();
        let selected_manifest = manifests
            .iter()
            .find(|manifest| manifest.manifest_sha256 == plans[0].execution_manifest_sha256)
            .unwrap();
        let validated = verify_tensor_execution_manifest(selected_manifest).unwrap();
        let slice = tensor_lineage_slice(&validated, &plans[0].source_tensor_name).unwrap();
        let stored_bytes = slice
            .nodes
            .iter()
            .filter(|node| node.stage == TensorStateStage::Stored)
            .map(|node| node.byte_len)
            .sum();
        let final_nodes: BTreeMap<_, _> = slice
            .nodes
            .iter()
            .filter(|node| node.stage == TensorStateStage::Executed)
            .map(|node| (node.node_id.clone(), node.clone()))
            .collect();
        let final_hash = hex::encode(Sha256::digest(serde_json::to_vec(&final_nodes).unwrap()));
        unit.options = vec![TensorOption {
            option_id: "q8".into(),
            execution_plans: plans.clone(),
            operations,
            payload_bytes: stored_bytes,
            storage_manifest_receipt_sha256: selected_manifest.conversion_receipt_sha256.clone(),
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
                final_executed_tensor_bundle_sha256: final_hash,
            },
            capability_profile_sha256: digest("capability-profile"),
        }];
    }
    (units, manifests)
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
    let structural_units = units(&fixture.inventory);
    let tensor_partition = build_tensor_partition(
        &fixture.inventory,
        &structural_units,
        vec![fixed_embedding(&fixture.inventory)],
    )
    .unwrap();
    let (allocation_units, execution_manifests) = allocatable_units(
        structural_units,
        &fixture.inventory,
        &tensor_partition.manifest_sha256,
        &dataset_partition.calibration_manifest_sha256,
    );
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
    let tensor_runtime = execution_manifests[0].runtime.clone();
    let execution_scope = execution_manifests[0].scope.clone();
    let problem = DynamicAllocationProblem {
        schema_version: DYNAMIC_ALLOCATION_SCHEMA_VERSION,
        source: fixture.source.clone(),
        execution: ExecutionIdentity {
            hf2q_revision: digest("producer-revision")[..40].into(),
            mlx_native_version: "0.10.14".into(),
            hardware_id: "apple-test".into(),
            os_build: "test-os".into(),
        },
        tensor_runtime,
        execution_scope,
        tensor_catalog_sha256: tensor_partition.tensor_catalog_sha256.clone(),
        expected_tensor_count: allocation_units.iter().map(|unit| unit.members.len()).sum(),
        dataset_partition_manifest_sha256: dataset_partition.manifest_sha256.clone(),
        tensor_partition_manifest_sha256: tensor_partition.manifest_sha256.clone(),
        execution_manifest_catalog_sha256: execution_manifest_catalog_sha256(&execution_manifests)
            .unwrap(),
        execution_manifest_catalog: execution_manifests,
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
        variable_payload_budget_bytes: 1024,
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

    let mut wrong_disposition = problem.clone();
    let manifest = &mut wrong_disposition.execution_manifest_catalog[0];
    let previous_manifest_sha256 = manifest.manifest_sha256.clone();
    manifest.dispositions[0].disposition = SourceTensorDispositionKind::Fixed;
    manifest.manifest_sha256 = tensor_execution_manifest_sha256(manifest).unwrap();
    let validated = verify_tensor_execution_manifest(manifest).unwrap();
    for unit in &mut wrong_disposition.units {
        for option in &mut unit.options {
            for plan in &mut option.execution_plans {
                if plan.execution_manifest_sha256 == previous_manifest_sha256 {
                    plan.execution_manifest_sha256 = manifest.manifest_sha256.clone();
                    plan.lineage_slice_sha256 =
                        tensor_lineage_slice(&validated, &plan.source_tensor_name)
                            .unwrap()
                            .slice_sha256;
                }
            }
        }
    }
    wrong_disposition.execution_manifest_catalog_sha256 =
        execution_manifest_catalog_sha256(&wrong_disposition.execution_manifest_catalog).unwrap();
    assert!(admit(&wrong_disposition, &contract, &receipt).is_err());
}
