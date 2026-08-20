use sha2::{Digest, Sha256};

use super::*;

pub(super) fn digest(label: &str) -> String {
    hex::encode(Sha256::digest(label.as_bytes()))
}

fn json_evidence(json: &str) -> CanonicalJsonEvidence {
    CanonicalJsonEvidence {
        canonical_json: json.into(),
        sha256: digest(json),
    }
}

pub(super) fn manifest() -> TensorExecutionManifest {
    let source = TensorStateNode {
        node_id: "source:q_proj".into(),
        stage: TensorStateStage::Source,
        semantic_name: "model.layers.0.self_attn.q_proj.weight".into(),
        shape: vec![16, 256],
        layout: TENSOR_LAYOUT_ROW_MAJOR_OUTERMOST_FIRST_V1.into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::Bf16,
        },
        byte_len: 8192,
        byte_sha256: digest("source-bytes"),
        logical_f32_sha256: digest("source-logical"),
        artifact_region: Some(ArtifactRegion {
            artifact_id: "source-shard".into(),
            byte_offset: 0,
            byte_len: 8192,
        }),
    };
    let stored = TensorStateNode {
        node_id: "stored:attn_q".into(),
        stage: TensorStateStage::Stored,
        semantic_name: "blk.0.attn_q.weight".into(),
        shape: vec![16, 256],
        layout: TENSOR_LAYOUT_ROW_MAJOR_OUTERMOST_FIRST_V1.into(),
        codec: PhysicalTensorCodec::Ggml {
            wire_type_id: 12,
            type_name: "Q4_K".into(),
        },
        byte_len: 2304,
        byte_sha256: digest("stored-bytes"),
        logical_f32_sha256: digest("stored-logical"),
        artifact_region: Some(ArtifactRegion {
            artifact_id: "model-gguf".into(),
            byte_offset: 4096,
            byte_len: 2304,
        }),
    };
    let decoded = TensorStateNode {
        node_id: "converted:attn_q:f32".into(),
        stage: TensorStateStage::Converted,
        semantic_name: "blk.0.attn_q.weight.decoded".into(),
        shape: vec![16, 256],
        layout: TENSOR_LAYOUT_ROW_MAJOR_OUTERMOST_FIRST_V1.into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::F32,
        },
        byte_len: 16384,
        byte_sha256: digest("decoded-bytes"),
        logical_f32_sha256: source.logical_f32_sha256.clone(),
        artifact_region: None,
    };
    let roundtripped = TensorStateNode {
        node_id: "converted:attn_q:f16-roundtrip".into(),
        semantic_name: "blk.0.attn_q.weight.roundtripped".into(),
        byte_sha256: digest("roundtripped-bytes"),
        logical_f32_sha256: stored.logical_f32_sha256.clone(),
        ..decoded.clone()
    };
    let loaded = TensorStateNode {
        node_id: "loaded:attn_q:f32".into(),
        stage: TensorStateStage::Loaded,
        semantic_name: "blk.0.attn_q.weight".into(),
        shape: vec![16, 256],
        layout: TENSOR_LAYOUT_ROW_MAJOR_OUTERMOST_FIRST_V1.into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::F32,
        },
        byte_len: 16384,
        byte_sha256: digest("loaded-bytes"),
        logical_f32_sha256: digest("stored-logical"),
        artifact_region: None,
    };
    let q = TensorStateNode {
        node_id: "executed:q".into(),
        stage: TensorStateStage::Executed,
        semantic_name: "layer0.wq".into(),
        shape: vec![8, 256],
        layout: TENSOR_LAYOUT_ROW_MAJOR_OUTERMOST_FIRST_V1.into(),
        codec: PhysicalTensorCodec::Ggml {
            wire_type_id: 2,
            type_name: "Q4_0".into(),
        },
        byte_len: 1152,
        byte_sha256: digest("executed-q"),
        logical_f32_sha256: digest("executed-q-logical"),
        artifact_region: None,
    };
    let gate = TensorStateNode {
        node_id: "executed:q_gate".into(),
        semantic_name: "layer0.w_gate".into(),
        byte_sha256: digest("executed-gate"),
        logical_f32_sha256: digest("executed-gate-logical"),
        ..q.clone()
    };
    let loaded_q = TensorStateNode {
        node_id: "loaded:q".into(),
        stage: TensorStateStage::Loaded,
        semantic_name: "layer0.wq.f32".into(),
        shape: vec![8, 256],
        layout: TENSOR_LAYOUT_ROW_MAJOR_OUTERMOST_FIRST_V1.into(),
        codec: PhysicalTensorCodec::Dense {
            dtype: TensorScalarDType::F32,
        },
        byte_len: 8192,
        byte_sha256: digest("loaded-q"),
        logical_f32_sha256: digest("loaded-q-logical"),
        artifact_region: None,
    };
    let loaded_gate = TensorStateNode {
        node_id: "loaded:q_gate".into(),
        semantic_name: "layer0.w_gate.f32".into(),
        byte_sha256: digest("loaded-gate"),
        logical_f32_sha256: digest("loaded-gate-logical"),
        ..loaded_q.clone()
    };
    let mut result = TensorExecutionManifest {
        schema_version: TENSOR_EXECUTION_MANIFEST_SCHEMA_VERSION,
        source_manifest_sha256: digest("source-manifest"),
        source_tensor_inventory_sha256: digest("source-inventory"),
        tensor_partition_manifest_sha256: digest("tensor-partition"),
        conversion_receipt_sha256: digest("conversion-receipt"),
        logical_hash_encoding: "hf2q-framed-f32-le-v1".into(),
        runtime: TensorRuntimeBinding {
            hf2q_revision: "1".repeat(40),
            mlx_native_version: "0.10.14".into(),
            mlx_native_capability_schema_version: 1,
            routing_policy_sha256: digest("routing"),
            graph_configuration_sha256: digest("graph"),
            capability_profile_sha256: digest("capability-profile"),
            hardware_profile_sha256: digest("hardware"),
            dwq_overlay_sha256: None,
        },
        scope: TensorExecutionScope {
            model_family: "qwen3_5_dense".into(),
            profile: "autoregressive_text_v1".into(),
            included_paths: vec!["text.decode".into(), "text.prefill".into()],
            excluded_paths: [("mtp".into(), "not accepted for serving".into())]
                .into_iter()
                .collect(),
        },
        artifacts: vec![
            ArtifactEvidence {
                artifact_id: "model-gguf".into(),
                role: "converted-model".into(),
                byte_len: 16384,
                sha256: digest("gguf"),
            },
            ArtifactEvidence {
                artifact_id: "source-shard".into(),
                role: "source-weights".into(),
                byte_len: 8192,
                sha256: digest("source-shard"),
            },
        ],
        nodes: vec![
            source,
            decoded,
            roundtripped,
            stored,
            loaded,
            loaded_q,
            loaded_gate,
            q,
            gate,
        ],
        transforms: vec![
            TensorTransformEdge {
                edge_id: "decode:q".into(),
                inputs: vec![TransformPort {
                    role: "source".into(),
                    node_id: "source:q_proj".into(),
                }],
                outputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: "converted:attn_q:f32".into(),
                }],
                operation: TensorTransformOperation::SourceDecode,
                implementation_revision: "1".repeat(40),
                receipt_sha256: digest("decode-edge"),
            },
            TensorTransformEdge {
                edge_id: "roundtrip:q".into(),
                inputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: "converted:attn_q:f32".into(),
                }],
                outputs: vec![TransformPort {
                    role: "converted_roundtripped".into(),
                    node_id: "converted:attn_q:f16-roundtrip".into(),
                }],
                operation: TensorTransformOperation::F16Roundtrip,
                implementation_revision: "1".repeat(40),
                receipt_sha256: digest("roundtrip-edge"),
            },
            TensorTransformEdge {
                edge_id: "convert:q".into(),
                inputs: vec![TransformPort {
                    role: "converted".into(),
                    node_id: "converted:attn_q:f16-roundtrip".into(),
                }],
                outputs: vec![TransformPort {
                    role: "stored".into(),
                    node_id: "stored:attn_q".into(),
                }],
                operation: TensorTransformOperation::GgufQuantize {
                    implementation_id: "hf2q-q4_k-v1".into(),
                    calibration_receipt_sha256: None,
                },
                implementation_revision: "1".repeat(40),
                receipt_sha256: digest("convert-edge"),
            },
            TensorTransformEdge {
                edge_id: "load:q".into(),
                inputs: vec![TransformPort {
                    role: "stored".into(),
                    node_id: "stored:attn_q".into(),
                }],
                outputs: vec![TransformPort {
                    role: "loaded".into(),
                    node_id: "loaded:attn_q:f32".into(),
                }],
                operation: TensorTransformOperation::GgufDequantize {
                    implementation_id: "hf2q-gguf-dequant-v1".into(),
                },
                implementation_revision: "1".repeat(40),
                receipt_sha256: digest("load-edge"),
            },
            TensorTransformEdge {
                edge_id: "split:q_gate".into(),
                inputs: vec![TransformPort {
                    role: "loaded".into(),
                    node_id: "loaded:attn_q:f32".into(),
                }],
                outputs: vec![
                    TransformPort {
                        role: "q".into(),
                        node_id: "loaded:q".into(),
                    },
                    TransformPort {
                        role: "gate".into(),
                        node_id: "loaded:q_gate".into(),
                    },
                ],
                operation: TensorTransformOperation::SplitInterleavedQGate {
                    implementation_id: "qwen35-split-q-gate-v1".into(),
                    heads: 4,
                    head_dim: 2,
                    hidden_size: 256,
                },
                implementation_revision: "1".repeat(40),
                receipt_sha256: digest("split-edge"),
            },
            TensorTransformEdge {
                edge_id: "execute:q".into(),
                inputs: vec![TransformPort {
                    role: "loaded".into(),
                    node_id: "loaded:q".into(),
                }],
                outputs: vec![TransformPort {
                    role: "executed".into(),
                    node_id: "executed:q".into(),
                }],
                operation: TensorTransformOperation::Qwen35LoadQ4Amax7V1,
                implementation_revision: "1".repeat(40),
                receipt_sha256: digest("execute-edge"),
            },
            TensorTransformEdge {
                edge_id: "execute:q_gate".into(),
                inputs: vec![TransformPort {
                    role: "loaded".into(),
                    node_id: "loaded:q_gate".into(),
                }],
                outputs: vec![TransformPort {
                    role: "executed".into(),
                    node_id: "executed:q_gate".into(),
                }],
                operation: TensorTransformOperation::Qwen35LoadQ4Amax7V1,
                implementation_revision: "1".repeat(40),
                receipt_sha256: digest("execute-gate-edge"),
            },
        ],
        operations: vec![
            RuntimeOperationBinding {
                binding_id: "layer0.q_proj.decode".into(),
                operation_id: "layer0.q_proj".into(),
                graph_path: "qwen35.full_attention.q".into(),
                entrypoint: "quantized_matmul_ggml_with_policy".into(),
                workload_regime: TensorExecutionRegime::TextDecodeM1,
                invocation_count: 1,
                source_tensor_names: vec!["model.layers.0.self_attn.q_proj.weight".into()],
                inputs: vec![TransformPort {
                    role: "weight".into(),
                    node_id: "executed:q".into(),
                }],
                capability: RuntimeCapabilityEvidence::Ggml {
                    request: json_evidence("{\"m\":1,\"type\":\"q4_0\"}"),
                    decision: json_evidence("{\"executable\":true,\"route\":\"dense_mv\"}"),
                    requires_device_probe: false,
                    resolved_runtime_trace: None,
                },
            },
            RuntimeOperationBinding {
                binding_id: "layer0.q_gate.decode".into(),
                operation_id: "layer0.q_gate".into(),
                graph_path: "qwen35.full_attention.q_gate".into(),
                entrypoint: "quantized_matmul_ggml_with_policy".into(),
                workload_regime: TensorExecutionRegime::TextDecodeM1,
                invocation_count: 1,
                source_tensor_names: vec!["model.layers.0.self_attn.q_proj.weight".into()],
                inputs: vec![TransformPort {
                    role: "weight".into(),
                    node_id: "executed:q_gate".into(),
                }],
                capability: RuntimeCapabilityEvidence::Ggml {
                    request: json_evidence("{\"m\":1,\"type\":\"q4_0\"}"),
                    decision: json_evidence("{\"executable\":true,\"route\":\"dense_mv\"}"),
                    requires_device_probe: false,
                    resolved_runtime_trace: None,
                },
            },
        ],
        dispositions: vec![SourceTensorDisposition {
            source_node_id: "source:q_proj".into(),
            disposition: SourceTensorDispositionKind::Variable,
            reason: "Dynamic allocation unit".into(),
            terminal_node_ids: vec!["executed:q".into(), "executed:q_gate".into()],
        }],
        manifest_sha256: String::new(),
    };
    result.manifest_sha256 = tensor_execution_manifest_sha256(&result).unwrap();
    result
}

#[test]
fn fused_source_fanout_manifest_verifies_and_is_order_invariant() {
    let expected = manifest();
    verify_tensor_execution_manifest(&expected).unwrap();

    let mut reordered = expected.clone();
    reordered.nodes.reverse();
    reordered.artifacts.reverse();
    reordered.transforms.reverse();
    reordered.scope.included_paths.reverse();
    assert_eq!(
        canonical_tensor_execution_manifest_bytes(&expected).unwrap(),
        canonical_tensor_execution_manifest_bytes(&reordered).unwrap()
    );
    verify_tensor_execution_manifest(&reordered).unwrap();
}

#[test]
fn mutation_and_missing_source_disposition_fail_closed() {
    let mut wrong_hash = manifest();
    wrong_hash.nodes[0].byte_sha256 = digest("tampered");
    assert_eq!(
        verify_tensor_execution_manifest(&wrong_hash).unwrap_err(),
        TensorExecutionManifestError::DigestMismatch
    );

    let mut omitted = manifest();
    omitted.dispositions.clear();
    omitted.manifest_sha256 = tensor_execution_manifest_sha256(&omitted).unwrap();
    assert!(matches!(
        verify_tensor_execution_manifest(&omitted),
        Err(TensorExecutionManifestError::Invalid(message))
            if message.contains("every source tensor")
    ));
}

#[test]
fn device_selected_capability_requires_runtime_trace() {
    let mut candidate = manifest();
    let RuntimeCapabilityEvidence::Ggml {
        requires_device_probe,
        ..
    } = &mut candidate.operations[0].capability
    else {
        panic!("fixture must use GGML evidence")
    };
    *requires_device_probe = true;
    candidate.manifest_sha256 = tensor_execution_manifest_sha256(&candidate).unwrap();
    assert!(matches!(
        verify_tensor_execution_manifest(&candidate),
        Err(TensorExecutionManifestError::Invalid(message))
            if message.contains("device-probe")
    ));
}

#[test]
fn canonical_json_evidence_rejects_noncanonical_object_order() {
    let mut candidate = manifest();
    let noncanonical = "{\"type\":\"q4_0\",\"m\":1}";
    let RuntimeCapabilityEvidence::Ggml { request, .. } = &mut candidate.operations[0].capability
    else {
        panic!("fixture must use GGML evidence")
    };
    *request = json_evidence(noncanonical);
    candidate.manifest_sha256 = tensor_execution_manifest_sha256(&candidate).unwrap();
    assert!(matches!(
        verify_tensor_execution_manifest(&candidate),
        Err(TensorExecutionManifestError::Invalid(message))
            if message.contains("canonical JSON")
    ));
}
