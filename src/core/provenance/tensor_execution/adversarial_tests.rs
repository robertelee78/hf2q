use super::tests::{digest, manifest};
use super::*;

fn resign(manifest: &mut TensorExecutionManifest) {
    manifest.manifest_sha256 = tensor_execution_manifest_sha256(manifest).unwrap();
}

fn invalid_message(manifest: &TensorExecutionManifest) -> String {
    match verify_tensor_execution_manifest(manifest) {
        Err(TensorExecutionManifestError::Invalid(message)) => message,
        other => panic!("expected invalid manifest, got {other:?}"),
    }
}

#[test]
fn unique_stored_payload_is_deduplicated_and_stage_checked() {
    let validated = verify_tensor_execution_manifest(&manifest()).unwrap();
    assert_eq!(
        unique_stored_payload_bytes(
            &validated,
            &["stored:attn_q".into(), "stored:attn_q".into()]
        )
        .unwrap(),
        2304
    );
    assert!(matches!(
        unique_stored_payload_bytes(&validated, &["executed:q".into()]),
        Err(TensorExecutionManifestError::Invalid(message)) if message.contains("not stored")
    ));
}

#[test]
fn operation_and_disposition_closures_reject_substitution_or_omission() {
    let mut wrong_source = manifest();
    wrong_source.operations[0].source_tensor_names = vec!["other.weight".into()];
    resign(&mut wrong_source);
    assert!(invalid_message(&wrong_source).contains("source-tensor closure"));

    let mut unconsumed = manifest();
    unconsumed
        .operations
        .retain(|operation| operation.operation_id != "layer0.q_gate");
    resign(&mut unconsumed);
    assert!(invalid_message(&unconsumed).contains("every executed tensor"));

    let mut omitted_terminal = manifest();
    omitted_terminal.dispositions[0]
        .terminal_node_ids
        .retain(|node_id| node_id != "executed:q_gate");
    resign(&mut omitted_terminal);
    assert!(invalid_message(&omitted_terminal).contains("exact executed descendants"));
}

#[test]
fn artifact_overlap_and_row_invalid_gguf_are_rejected() {
    let mut overlap = manifest();
    let source = overlap
        .nodes
        .iter_mut()
        .find(|node| node.node_id == "source:q_proj")
        .unwrap();
    source.artifact_region = Some(ArtifactRegion {
        artifact_id: "model-gguf".into(),
        byte_offset: 0,
        byte_len: source.byte_len,
    });
    resign(&mut overlap);
    assert!(invalid_message(&overlap).contains("overlap"));

    let mut bad_row = manifest();
    let stored = bad_row
        .nodes
        .iter_mut()
        .find(|node| node.node_id == "stored:attn_q")
        .unwrap();
    stored.shape = vec![64, 64];
    resign(&mut bad_row);
    assert!(invalid_message(&bad_row).contains("tensor node"));
}

#[test]
fn duplicate_ports_and_illegal_stage_shortcuts_are_rejected() {
    let mut duplicate_role = manifest();
    duplicate_role.operations[0].inputs.push(TransformPort {
        role: "weight".into(),
        node_id: "executed:q_gate".into(),
    });
    duplicate_role.operations[0].source_tensor_names.clear();
    duplicate_role.operations[0]
        .source_tensor_names
        .push("model.layers.0.self_attn.q_proj.weight".into());
    resign(&mut duplicate_role);
    assert!(invalid_message(&duplicate_role).contains("unique executed nodes"));

    let mut shortcut = manifest();
    shortcut.transforms[1].operation = TensorTransformOperation::Qwen35LoadQ4Amax7V1;
    resign(&mut shortcut);
    assert!(invalid_message(&shortcut).contains("illegal stage signature"));
}

#[test]
fn same_stage_cycle_is_rejected() {
    let mut cyclic = manifest();
    let template = cyclic
        .nodes
        .iter()
        .find(|node| node.node_id == "loaded:q")
        .unwrap()
        .clone();
    cyclic.nodes.push(TensorStateNode {
        node_id: "loaded:cycle-a".into(),
        semantic_name: "cycle-a".into(),
        ..template.clone()
    });
    cyclic.nodes.push(TensorStateNode {
        node_id: "loaded:cycle-b".into(),
        semantic_name: "cycle-b".into(),
        ..template
    });
    cyclic.transforms.extend([
        TensorTransformEdge {
            edge_id: "cycle:a".into(),
            inputs: vec![TransformPort {
                role: "input".into(),
                node_id: "loaded:cycle-b".into(),
            }],
            outputs: vec![TransformPort {
                role: "output".into(),
                node_id: "loaded:cycle-a".into(),
            }],
            operation: TensorTransformOperation::Transpose {
                permutation: vec![0, 1],
            },
            implementation_revision: "2".repeat(40),
            receipt_sha256: digest("cycle-a"),
        },
        TensorTransformEdge {
            edge_id: "cycle:b".into(),
            inputs: vec![TransformPort {
                role: "input".into(),
                node_id: "loaded:cycle-a".into(),
            }],
            outputs: vec![TransformPort {
                role: "output".into(),
                node_id: "loaded:cycle-b".into(),
            }],
            operation: TensorTransformOperation::Transpose {
                permutation: vec![0, 1],
            },
            implementation_revision: "2".repeat(40),
            receipt_sha256: digest("cycle-b"),
        },
    ]);
    resign(&mut cyclic);
    assert!(invalid_message(&cyclic).contains("cycle"));
}

#[test]
fn regime_bindings_and_named_ports_are_canonical_sets() {
    let mut expected = manifest();
    let mut prefill = expected.operations[0].clone();
    prefill.binding_id = "layer0.q_proj.prefill".into();
    prefill.workload_regime = TensorExecutionRegime::TextPrefill;
    expected.operations.push(prefill);
    resign(&mut expected);
    verify_tensor_execution_manifest(&expected).unwrap();

    let mut reordered = expected.clone();
    reordered.operations.reverse();
    for edge in &mut reordered.transforms {
        edge.inputs.reverse();
        edge.outputs.reverse();
    }
    for operation in &mut reordered.operations {
        operation.inputs.reverse();
        operation.source_tensor_names.reverse();
    }
    assert_eq!(
        canonical_tensor_execution_manifest_bytes(&expected).unwrap(),
        canonical_tensor_execution_manifest_bytes(&reordered).unwrap()
    );
    verify_tensor_execution_manifest(&reordered).unwrap();
}

#[test]
fn stale_digest_and_noncanonical_implementation_revision_fail_closed() {
    let mut stale = manifest();
    stale.scope.profile.push_str("-changed");
    assert_eq!(
        verify_tensor_execution_manifest(&stale).unwrap_err(),
        TensorExecutionManifestError::DigestMismatch
    );

    let mut uppercase = manifest();
    uppercase.transforms[0].implementation_revision = "A".repeat(40);
    resign(&mut uppercase);
    assert!(invalid_message(&uppercase).contains("transform edge"));
}

#[test]
fn transform_roles_layout_and_quantize_arity_fail_closed() {
    let mut wrong_split_role = manifest();
    let split = wrong_split_role
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "split:q_gate")
        .unwrap();
    split.outputs[0].role = "query".into();
    resign(&mut wrong_split_role);
    assert!(invalid_message(&wrong_split_role).contains("codec or geometry semantics"));

    let mut many_output_quantize = manifest();
    let mut extra = many_output_quantize
        .nodes
        .iter()
        .find(|node| node.node_id == "stored:attn_q")
        .unwrap()
        .clone();
    extra.node_id = "stored:attn_q:extra".into();
    extra.artifact_region = Some(ArtifactRegion {
        artifact_id: "model-gguf".into(),
        byte_offset: 6_500,
        byte_len: extra.byte_len,
    });
    many_output_quantize.nodes.push(extra);
    many_output_quantize
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "convert:q")
        .unwrap()
        .outputs
        .push(TransformPort {
            role: "stored_extra".into(),
            node_id: "stored:attn_q:extra".into(),
        });
    resign(&mut many_output_quantize);
    assert!(invalid_message(&many_output_quantize).contains("codec or geometry semantics"));

    let mut wrong_layout = manifest();
    wrong_layout.nodes[0].layout = "backend-native-order".into();
    resign(&mut wrong_layout);
    assert!(invalid_message(&wrong_layout).contains("tensor node"));
}

#[test]
fn float_ggml_wire_types_are_physically_representable() {
    let mut float_stored = manifest();
    let store = float_stored
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "convert:q")
        .unwrap();
    store.inputs[0].node_id = "converted:attn_q:f32".into();
    store.operation = TensorTransformOperation::GgufDenseStore {
        implementation_id: "hf2q-bf16-store-v1".into(),
    };
    float_stored
        .transforms
        .retain(|edge| edge.edge_id != "roundtrip:q");
    float_stored
        .nodes
        .retain(|node| node.node_id != "converted:attn_q:f16-roundtrip");
    float_stored
        .artifacts
        .iter_mut()
        .find(|artifact| artifact.artifact_id == "model-gguf")
        .unwrap()
        .byte_len = 32_768;

    for (wire_type_id, type_name, byte_len) in
        [(0, "F32", 16_384), (1, "F16", 8_192), (30, "BF16", 8_192)]
    {
        let mut candidate = float_stored.clone();
        let stored = candidate
            .nodes
            .iter_mut()
            .find(|node| node.node_id == "stored:attn_q")
            .unwrap();
        stored.codec = PhysicalTensorCodec::Ggml {
            wire_type_id,
            type_name: type_name.into(),
        };
        stored.byte_len = byte_len;
        stored.artifact_region.as_mut().unwrap().byte_len = byte_len;
        resign(&mut candidate);
        verify_tensor_execution_manifest(&candidate).unwrap();
    }
}

#[test]
fn quantized_store_cannot_omit_the_converter_f16_roundtrip() {
    let mut omitted = manifest();
    let quantize = omitted
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "convert:q")
        .unwrap();
    quantize.inputs[0].node_id = "converted:attn_q:f32".into();
    resign(&mut omitted);
    assert!(invalid_message(&omitted).contains("canonical F16 roundtrip"));
}

#[test]
fn architecture_bake_and_concatenate_geometry_fail_closed() {
    let mut bad_bake = manifest();
    let decoded = bad_bake
        .nodes
        .iter()
        .find(|node| node.node_id == "converted:attn_q:f32")
        .unwrap()
        .clone();
    let roundtripped = bad_bake
        .nodes
        .iter_mut()
        .find(|node| node.node_id == "converted:attn_q:f16-roundtrip")
        .unwrap();
    roundtripped.shape = vec![8, 256];
    roundtripped.byte_len = 8_192;
    let roundtrip = bad_bake
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "roundtrip:q")
        .unwrap();
    roundtrip.operation = TensorTransformOperation::ArchitectureBake {
        operation: "invalid-half-size".into(),
        parameters_sha256: digest("invalid-half-size"),
    };
    assert_eq!(decoded.shape, vec![16, 256]);
    resign(&mut bad_bake);
    assert!(invalid_message(&bad_bake).contains("codec or geometry semantics"));

    let mut bad_concat = manifest();
    let concat = bad_concat
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "roundtrip:q")
        .unwrap();
    concat.operation = TensorTransformOperation::Concatenate { axis: 9 };
    resign(&mut bad_concat);
    assert!(invalid_message(&bad_concat).contains("codec or geometry semantics"));
}

#[test]
fn typed_squeeze_represents_rank_changing_qwen_bake() {
    let mut squeezed = manifest();
    for node_id in ["source:q_proj", "converted:attn_q:f32"] {
        squeezed
            .nodes
            .iter_mut()
            .find(|node| node.node_id == node_id)
            .unwrap()
            .shape = vec![16, 1, 256];
    }
    let decoded = squeezed
        .nodes
        .iter()
        .find(|node| node.node_id == "converted:attn_q:f32")
        .unwrap()
        .clone();
    let squeezed_node = TensorStateNode {
        node_id: "converted:attn_q:squeezed".into(),
        semantic_name: "blk.0.attn_q.weight.squeezed".into(),
        shape: vec![16, 256],
        byte_sha256: digest("squeezed-bytes"),
        ..decoded
    };
    squeezed.nodes.push(squeezed_node.clone());
    squeezed.transforms.push(TensorTransformEdge {
        edge_id: "squeeze:q".into(),
        inputs: vec![TransformPort {
            role: "input".into(),
            node_id: "converted:attn_q:f32".into(),
        }],
        outputs: vec![TransformPort {
            role: "output".into(),
            node_id: squeezed_node.node_id.clone(),
        }],
        operation: TensorTransformOperation::Squeeze { axis: 1 },
        implementation_revision: "4".repeat(40),
        receipt_sha256: digest("squeeze-edge"),
    });
    squeezed
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "roundtrip:q")
        .unwrap()
        .inputs[0]
        .node_id = squeezed_node.node_id;
    resign(&mut squeezed);
    verify_tensor_execution_manifest(&squeezed).unwrap();

    let mut wrong_axis = squeezed;
    let edge = wrong_axis
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "squeeze:q")
        .unwrap();
    edge.operation = TensorTransformOperation::Squeeze { axis: 0 };
    resign(&mut wrong_axis);
    assert!(invalid_message(&wrong_axis).contains("codec or geometry semantics"));
}

#[test]
fn ggml_capability_rejects_mixed_physical_weight_inputs() {
    let mut mixed = manifest();
    let loaded = mixed
        .nodes
        .iter()
        .find(|node| node.node_id == "loaded:q")
        .unwrap()
        .clone();
    let dense_executed = TensorStateNode {
        node_id: "executed:q:dense-aux".into(),
        stage: TensorStateStage::Executed,
        semantic_name: "layer0.wq.dense_aux".into(),
        artifact_region: None,
        ..loaded
    };
    mixed.nodes.push(dense_executed.clone());
    mixed.transforms.push(TensorTransformEdge {
        edge_id: "bind:q:dense-aux".into(),
        inputs: vec![TransformPort {
            role: "loaded".into(),
            node_id: "loaded:q".into(),
        }],
        outputs: vec![TransformPort {
            role: "executed".into(),
            node_id: dense_executed.node_id.clone(),
        }],
        operation: TensorTransformOperation::RuntimeBind,
        implementation_revision: "3".repeat(40),
        receipt_sha256: digest("bind-dense-aux"),
    });
    mixed.operations[0].inputs.push(TransformPort {
        role: "dense_aux".into(),
        node_id: dense_executed.node_id.clone(),
    });
    mixed.dispositions[0]
        .terminal_node_ids
        .push(dense_executed.node_id);
    resign(&mut mixed);
    assert!(invalid_message(&mixed).contains("all-GGML inputs"));
}

#[test]
fn split_geometry_overflow_rejects_without_panicking() {
    let mut overflow = manifest();
    let split = overflow
        .transforms
        .iter_mut()
        .find(|edge| edge.edge_id == "split:q_gate")
        .unwrap();
    split.operation = TensorTransformOperation::SplitInterleavedQGate {
        implementation_id: "qwen35-split-q-gate-v1".into(),
        heads: u32::MAX,
        head_dim: u32::MAX,
        hidden_size: 256,
    };
    resign(&mut overflow);
    let result = std::panic::catch_unwind(|| verify_tensor_execution_manifest(&overflow));
    assert!(result.is_ok());
    assert!(matches!(
        result.unwrap(),
        Err(TensorExecutionManifestError::Invalid(_))
    ));
}
