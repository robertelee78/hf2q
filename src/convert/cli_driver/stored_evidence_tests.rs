use super::*;
use crate::convert::tensor_lineage::MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES;
use crate::core::integrity::{compute_git_blob_sha1, ShardIntegrity};
use crate::intelligence::dynamic_allocator::producer::SourceTensorRecord;
use crate::intelligence::measured_auto_quant::SourceIdentity;
use half::f16;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::json;
use sha2::{Digest, Sha256};

fn f16_tensor_bytes(shape: &[usize], seed: usize) -> Vec<u8> {
    let elements = shape.iter().product::<usize>();
    (0..elements)
        .flat_map(|index| {
            f16::from_f32(((index + seed) as f32 % 131.0 - 65.0) / 48.0)
                .to_bits()
                .to_le_bytes()
        })
        .collect()
}

#[test]
fn dense_qwen_stored_evidence_runs_the_authoritative_conversion_loop() {
    let dir = tempfile::tempdir().unwrap();
    let tensor_specs: Vec<(String, Vec<usize>, Vec<u8>)> = vec![
        (
            "model.embed_tokens.weight".into(),
            vec![32, 256],
            f16_tensor_bytes(&[32, 256], 1),
        ),
        (
            "model.norm.weight".into(),
            vec![256],
            f16_tensor_bytes(&[256], 2),
        ),
        (
            "lm_head.weight".into(),
            vec![32, 256],
            f16_tensor_bytes(&[32, 256], 3),
        ),
        (
            "model.layers.0.input_layernorm.weight".into(),
            vec![256],
            f16_tensor_bytes(&[256], 4),
        ),
        (
            "model.layers.0.post_attention_layernorm.weight".into(),
            vec![256],
            f16_tensor_bytes(&[256], 5),
        ),
        (
            "model.layers.0.linear_attn.in_proj_qkv.weight".into(),
            vec![512, 256],
            f16_tensor_bytes(&[512, 256], 6),
        ),
        (
            "model.layers.0.linear_attn.in_proj_z.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 7),
        ),
        (
            "model.layers.0.linear_attn.conv1d.weight".into(),
            vec![512, 1, 4],
            f16_tensor_bytes(&[512, 1, 4], 8),
        ),
        (
            "model.layers.0.linear_attn.in_proj_a.weight".into(),
            vec![2, 256],
            f16_tensor_bytes(&[2, 256], 9),
        ),
        (
            "model.layers.0.linear_attn.dt_bias".into(),
            vec![2],
            f16_tensor_bytes(&[2], 10),
        ),
        (
            "model.layers.0.linear_attn.in_proj_b.weight".into(),
            vec![2, 256],
            f16_tensor_bytes(&[2, 256], 11),
        ),
        (
            "model.layers.0.linear_attn.A_log".into(),
            vec![2],
            f16_tensor_bytes(&[2], 12),
        ),
        (
            "model.layers.0.linear_attn.norm.weight".into(),
            vec![128],
            f16_tensor_bytes(&[128], 13),
        ),
        (
            "model.layers.0.linear_attn.out_proj.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 14),
        ),
        (
            "model.layers.0.mlp.gate_proj.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 15),
        ),
        (
            "model.layers.0.mlp.up_proj.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 16),
        ),
        (
            "model.layers.0.mlp.down_proj.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 17),
        ),
        (
            "model.layers.1.input_layernorm.weight".into(),
            vec![256],
            f16_tensor_bytes(&[256], 18),
        ),
        (
            "model.layers.1.post_attention_layernorm.weight".into(),
            vec![256],
            f16_tensor_bytes(&[256], 19),
        ),
        (
            "model.layers.1.self_attn.q_proj.weight".into(),
            vec![512, 256],
            f16_tensor_bytes(&[512, 256], 20),
        ),
        (
            "model.layers.1.self_attn.k_proj.weight".into(),
            vec![64, 256],
            f16_tensor_bytes(&[64, 256], 21),
        ),
        (
            "model.layers.1.self_attn.v_proj.weight".into(),
            vec![64, 256],
            f16_tensor_bytes(&[64, 256], 22),
        ),
        (
            "model.layers.1.self_attn.o_proj.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 23),
        ),
        (
            "model.layers.1.self_attn.q_norm.weight".into(),
            vec![64],
            f16_tensor_bytes(&[64], 24),
        ),
        (
            "model.layers.1.self_attn.k_norm.weight".into(),
            vec![64],
            f16_tensor_bytes(&[64], 25),
        ),
        (
            "model.layers.1.mlp.gate_proj.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 26),
        ),
        (
            "model.layers.1.mlp.up_proj.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 27),
        ),
        (
            "model.layers.1.mlp.down_proj.weight".into(),
            vec![256, 256],
            f16_tensor_bytes(&[256, 256], 28),
        ),
    ];
    let tensor_views: Vec<(String, TensorView<'_>)> = tensor_specs
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(Dtype::F16, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    let model_bytes = safetensors::tensor::serialize(
        tensor_views.iter().map(|(name, view)| (name.clone(), view)),
        None,
    )
    .unwrap();
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, &model_bytes).unwrap();

    let config = json!({
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "full_attention_interval": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 0.000001,
        "vocab_size": 32,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128
    });
    let config_bytes = serde_json::to_vec(&config).unwrap();
    let config_path = dir.path().join("config.json");
    std::fs::write(&config_path, &config_bytes).unwrap();

    let mut vocab = serde_json::Map::new();
    for index in 0..28 {
        vocab.insert(format!("tok{index}"), json!(index));
    }
    std::fs::write(
        dir.path().join("tokenizer.json"),
        serde_json::to_vec(&json!({
            "model": {"type": "BPE", "byte_fallback": true, "vocab": vocab, "merges": []},
            "added_tokens": [
                {"id": 28, "content": "<bos>", "special": true},
                {"id": 29, "content": "<eos>", "special": true},
                {"id": 30, "content": "<pad>", "special": true},
                {"id": 31, "content": "<unk>", "special": true}
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::to_vec(&json!({
            "bos_token": "<bos>", "eos_token": "<eos>",
            "pad_token": "<pad>", "unk_token": "<unk>",
            "add_bos_token": true, "add_eos_token": false
        }))
        .unwrap(),
    )
    .unwrap();

    let revision = "a".repeat(40);
    let model_sha = hex::encode(Sha256::digest(&model_bytes));
    let verified_source = VerifiedSourceManifest::for_test_bound(
        "org/tiny-qwen",
        revision.clone(),
        vec![
            ShardIntegrity {
                filename: "config.json".into(),
                bytes: config_bytes.len() as u64,
                sha256: None,
                hf_etag: compute_git_blob_sha1(&config_path, config_bytes.len() as u64).unwrap(),
                is_lfs: false,
            },
            ShardIntegrity {
                filename: "model.safetensors".into(),
                bytes: model_bytes.len() as u64,
                sha256: Some(model_sha.clone()),
                hf_etag: model_sha,
                is_lfs: true,
            },
        ],
    );
    let verified_manifest_sha = hex::encode(Sha256::digest(
        serde_json::to_vec(&verified_source).unwrap(),
    ));
    let source = SourceIdentity {
        model_id: "org/tiny-qwen".into(),
        revision: revision.clone(),
        config_sha256: hex::encode(Sha256::digest(&config_bytes)),
        tensor_bundle_sha256: "b".repeat(64),
        tokenizer_bundle_sha256: "c".repeat(64),
        chat_template_sha256: "d".repeat(64),
    };
    let source_records = tensor_specs
        .iter()
        .map(|(name, shape, bytes)| SourceTensorRecord {
            name: name.clone(),
            source_shape: shape.clone(),
            source_dtype: "F16".into(),
            source_byte_len: bytes.len() as u64,
            source_tensor_sha256: hex::encode(Sha256::digest(bytes)),
        })
        .collect();
    let dispositions = tensor_specs
        .iter()
        .map(|(name, _, _)| {
            (
                name.clone(),
                crate::convert::tensor_lineage::ConversionSourceDisposition::Variable,
            )
        })
        .collect();
    let tensor_count = tensor_specs.len();
    let context = VerifiedConversionEvidenceContext::for_test(
        source,
        verified_manifest_sha,
        source_records,
        dispositions,
    );
    for (selector, output_name, expected_embedding_type, expected_output_type, expected_ffn_type) in [
        (
            QuantSelector::Standard(GgufFtype::MostlyQ8_0),
            "tiny-qwen-q8.gguf",
            "q8_0",
            "q8_0",
            "q8_0",
        ),
        (
            QuantSelector::Standard(GgufFtype::MostlyQ4_K_M),
            "tiny-qwen-q4-k-m.gguf",
            "q4_k",
            "q6_k",
            "q4_k",
        ),
    ] {
        let remote = RemoteConversionSource::for_test(
            crate::input::hf_reference::HfModelReference::parse("org/tiny-qwen", None)
                .unwrap()
                .resolve(&revision)
                .unwrap(),
            "b".repeat(64),
        );
        let output = dir.path().join(output_name);
        let verified = run_convert_with_stored_evidence(
            ConvertArgs {
                hf_dir: dir.path().into(),
                selector,
                output: output.clone(),
                dry_run: false,
                imatrix: None,
                imatrix_corpus: None,
                imatrix_out: None,
                imatrix_n_ctx: None,
                mode: ConvertMode::TextOnly,
                remote_source: Some(remote),
            },
            verified_source.clone(),
            context.clone(),
        )
        .unwrap();
        assert!(output.is_file());
        assert_eq!(verified.receipt().tensor_lineages.len(), tensor_count);
        let embedding = verified
            .receipt()
            .tensor_lineages
            .iter()
            .find(|lineage| lineage.gguf_tensor_name == "token_embd.weight")
            .unwrap();
        let ffn_up = verified
            .receipt()
            .tensor_lineages
            .iter()
            .find(|lineage| lineage.gguf_tensor_name == "blk.0.ffn_up.weight")
            .unwrap();
        let ffn_down = verified
            .receipt()
            .tensor_lineages
            .iter()
            .find(|lineage| lineage.gguf_tensor_name == "blk.0.ffn_down.weight")
            .unwrap();
        let output_projection = verified
            .receipt()
            .tensor_lineages
            .iter()
            .find(|lineage| lineage.gguf_tensor_name == "output.weight")
            .unwrap();
        assert_eq!(embedding.stored.ggml_type_name, expected_embedding_type);
        assert_eq!(
            output_projection.stored.ggml_type_name,
            expected_output_type
        );
        assert_eq!(ffn_up.stored.ggml_type_name, expected_ffn_type);
        assert_eq!(ffn_down.stored.ggml_type_name, expected_ffn_type);
        assert!(embedding.stored.f16_roundtrip_f32_bytes_sha256.is_some());
        assert!(ffn_up.stored.f16_roundtrip_f32_bytes_sha256.is_some());
        assert_ne!(
            ffn_up.stored.converted_logical_f32_sha256, ffn_up.stored.stored_logical_f32_sha256,
            "the receipt must hash decoded stored values, not copy the pre-quant hash"
        );
        let tensor_receipt_path =
            crate::convert::tensor_lineage::tensor_conversion_receipt_path(&output);
        assert!(tensor_receipt_path.is_file());
        let persisted: crate::convert::tensor_lineage::StoredTensorConversionReceipt =
            serde_json::from_slice(&std::fs::read(&tensor_receipt_path).unwrap()).unwrap();
        crate::convert::tensor_lineage::validate_stored_tensor_conversion_receipt(&persisted)
            .unwrap();
        assert_eq!(&persisted, verified.receipt());
        let replayed =
            verify_persisted_stored_evidence(dir.path(), &output, &verified_source, &context)
                .unwrap();
        assert_eq!(replayed.receipt(), &persisted);

        {
            let retained =
                verify_persisted_stored_artifact(dir.path(), &output, &verified_source, &context)
                    .unwrap();
            let displaced = dir.path().join(format!("{output_name}.displaced"));
            std::fs::rename(&output, &displaced).unwrap();
            std::fs::write(&output, b"pathname replacement is not the admitted GGUF").unwrap();
            let expected_payload_sha = persisted
                .tensor_lineages
                .iter()
                .find(|lineage| lineage.gguf_tensor_name == "blk.0.ffn_up.weight")
                .unwrap()
                .stored
                .payload_sha256
                .clone();
            let loaded = retained
                .load_and_reconcile(|gguf, source_config, _conversion| {
                    assert_eq!(source_config["hidden_size"], 256);
                    let payload = gguf.read_tensor_bytes_host("blk.0.ffn_up.weight")?;
                    Ok(hex::encode(Sha256::digest(payload)))
                })
                .unwrap();
            assert_eq!(loaded.conversion().receipt(), &persisted);
            assert_eq!(loaded.value(), &expected_payload_sha);
            drop(loaded);
            std::fs::remove_file(&output).unwrap();
            std::fs::rename(&displaced, &output).unwrap();

            if mlx_native::MlxDevice::new().is_ok() {
                let rejected = verify_persisted_stored_artifact(
                    dir.path(),
                    &output,
                    &verified_source,
                    &context,
                )
                .unwrap();
                let mut rejected_progress = crate::serve::header::LoadProgress::new(false, 1, 1);
                assert!(
                    crate::inference::models::qwen35::execution_evidence::load_no_dwq_qwen35_candidate(
                        rejected,
                        Some(std::path::Path::new("forbidden.dwq.safetensors")),
                        &mut rejected_progress,
                    )
                    .is_err(),
                    "the loaded-candidate join must reject a DWQ overlay before loading"
                );

                let retained = verify_persisted_stored_artifact(
                    dir.path(),
                    &output,
                    &verified_source,
                    &context,
                )
                .unwrap();
                let mut progress = crate::serve::header::LoadProgress::new(false, 1, 1);
                let candidate = crate::inference::models::qwen35::execution_evidence::load_no_dwq_qwen35_candidate(
                    retained,
                    None,
                    &mut progress,
                )
                .unwrap();
                assert_eq!(candidate.config().hidden_size, 256);
                assert_eq!(candidate.config().num_hidden_layers, 2);
                assert_eq!(
                    candidate.conversion_receipt_sha256(),
                    persisted.receipt_sha256
                );
                assert_eq!(candidate.loaded_tensor_count(), tensor_count);
                assert_eq!(candidate.loaded_catalog_sha256().len(), 64);
                assert_eq!(
                    candidate.loaded_catalog_conversion_receipt_sha256(),
                    persisted.receipt_sha256
                );
                candidate.execution().validate().unwrap();
                let mut session = candidate.start_text_session(16).unwrap();
                assert_eq!(session.executed_catalog_sha256().len(), 64);
                let executed_catalog_sha256 = session.executed_catalog_sha256().to_owned();
                assert_eq!(session.executed_tensor_count(), tensor_count);
                assert_eq!(
                    session.executed_catalog_loaded_parent_sha256(),
                    candidate.loaded_catalog_sha256()
                );
                let mut mislabeled = candidate.start_text_session(16).unwrap();
                assert!(
                    mislabeled.forward(&[0], &[0, 0, 0, 0]).is_err(),
                    "a one-token call cannot satisfy the prompt regime"
                );
                assert!(
                    mislabeled.forward_greedy(1, [0, 0, 0, 0]).is_err(),
                    "decode cannot precede the admitted prompt"
                );
                let prompt_tokens = [0_u32; 9];
                let prompt_positions = [0_i32; 36];
                let logits = session.forward(&prompt_tokens, &prompt_positions).unwrap();
                assert_eq!(logits.len(), prompt_tokens.len() * 32);
                let next = session.forward_greedy(1, [9, 9, 9, 9]).unwrap();
                assert!(next < 32);
                assert_eq!(session.encoded_dispatches().len(), 30);
                let operation_ids = session
                    .encoded_dispatches()
                    .iter()
                    .map(|observation| observation.operation_id.as_str())
                    .collect::<std::collections::BTreeSet<_>>();
                assert!(operation_ids.contains("blk.0.ffn_gate_up_silu"));
                assert!(operation_ids.contains("blk.0.ffn_down.weight"));
                assert!(operation_ids.contains("blk.0.attn_qkv.weight"));
                assert!(operation_ids.contains("blk.0.ssm_out.weight"));
                assert!(operation_ids.contains("blk.1.attn_q.weight"));
                assert!(operation_ids.contains("output.weight"));
                let observed_bindings = session
                    .encoded_dispatches()
                    .iter()
                    .map(|observation| {
                        (
                            match observation.trace.request.workload {
                                mlx_native::GgmlWorkloadClass::Prompt => "prompt",
                                mlx_native::GgmlWorkloadClass::DecodeSingle => "decode",
                                _ => "unexpected",
                            },
                            observation.operation_id.clone(),
                            observation.executed_tensor_node_ids.clone(),
                        )
                    })
                    .collect::<std::collections::BTreeSet<_>>();
                let node = |semantic: &str| vec![format!("executed:{semantic}")];
                let mut expected_bindings = vec![
                    (
                        "prompt",
                        "blk.0.attn_qkv.weight",
                        node("blk.0.attn_qkv.weight"),
                    ),
                    (
                        "prompt",
                        "blk.0.attn_gate.weight",
                        node("blk.0.attn_gate.weight"),
                    ),
                    (
                        "prompt",
                        "blk.0.ssm_alpha.weight",
                        node("blk.0.ssm_alpha.weight"),
                    ),
                    (
                        "prompt",
                        "blk.0.ssm_beta.weight",
                        node("blk.0.ssm_beta.weight"),
                    ),
                    (
                        "prompt",
                        "blk.0.ssm_out.weight",
                        node("blk.0.ssm_out.weight"),
                    ),
                    (
                        "prompt",
                        "blk.0.ffn_gate.weight",
                        node("blk.0.ffn_gate.weight"),
                    ),
                    ("prompt", "blk.0.ffn_up.weight", node("blk.0.ffn_up.weight")),
                    (
                        "prompt",
                        "blk.0.ffn_down.weight",
                        node("blk.0.ffn_down.weight"),
                    ),
                    ("prompt", "blk.1.attn_q.weight", node("blk.1.attn_q.weight")),
                    ("prompt", "blk.1.attn_k.weight", node("blk.1.attn_k.weight")),
                    ("prompt", "blk.1.attn_v.weight", node("blk.1.attn_v.weight")),
                    (
                        "prompt",
                        "blk.1.attn_output.weight",
                        node("blk.1.attn_output.weight"),
                    ),
                    (
                        "prompt",
                        "blk.1.ffn_gate.weight",
                        node("blk.1.ffn_gate.weight"),
                    ),
                    ("prompt", "blk.1.ffn_up.weight", node("blk.1.ffn_up.weight")),
                    (
                        "prompt",
                        "blk.1.ffn_down.weight",
                        node("blk.1.ffn_down.weight"),
                    ),
                    ("prompt", "output.weight", node("output.weight")),
                ];
                for operation in [
                    "blk.0.attn_qkv.weight",
                    "blk.0.attn_gate.weight",
                    "blk.0.ssm_alpha.weight",
                    "blk.0.ssm_beta.weight",
                    "blk.0.ssm_out.weight",
                    "blk.1.attn_q.weight",
                    "blk.1.attn_k.weight",
                    "blk.1.attn_v.weight",
                    "blk.1.attn_output.weight",
                    "blk.0.ffn_down.weight",
                    "blk.1.ffn_down.weight",
                    "output.weight",
                ] {
                    expected_bindings.push(("decode", operation, node(operation)));
                }
                expected_bindings.push((
                    "decode",
                    "blk.0.ffn_gate_up_silu",
                    vec![
                        "executed:blk.0.ffn_gate.weight".into(),
                        "executed:blk.0.ffn_up.weight".into(),
                    ],
                ));
                expected_bindings.push((
                    "decode",
                    "blk.1.ffn_gate_up_silu",
                    vec![
                        "executed:blk.1.ffn_gate.weight".into(),
                        "executed:blk.1.ffn_up.weight".into(),
                    ],
                ));
                let expected_bindings = expected_bindings
                    .into_iter()
                    .map(|(workload, operation, nodes)| (workload, operation.to_owned(), nodes))
                    .collect::<std::collections::BTreeSet<_>>();
                assert_eq!(observed_bindings, expected_bindings);
                for (index, observation) in session.encoded_dispatches().iter().enumerate() {
                    assert_eq!(observation.trace.mlx_native_version, "0.12.1");
                    assert_eq!(
                        observation.trace.request.workload,
                        if index < 16 {
                            mlx_native::GgmlWorkloadClass::Prompt
                        } else {
                            mlx_native::GgmlWorkloadClass::DecodeSingle
                        }
                    );
                    assert_eq!(
                        observation.trace.capability,
                        mlx_native::ggml_capability(observation.trace.request)
                    );
                    assert_eq!(
                        &observation.trace.request.routing,
                        candidate.execution().ggml_routing_policy()
                    );
                    assert!(!observation.trace.dispatches.is_empty());
                    assert!(!observation.trace.device.name.is_empty());
                    if observation.operation_id == "blk.0.ffn_down.weight" {
                        assert_eq!(
                            observation.trace.request.ggml_type,
                            if expected_ffn_type == "q4_k" {
                                mlx_native::ops::quantized_matmul_ggml::GgmlType::Q4_K
                            } else {
                                mlx_native::ops::quantized_matmul_ggml::GgmlType::Q8_0
                            }
                        );
                    }
                }
                assert!(
                    session.duplicate_observation_fails_sealing(),
                    "duplicate operation/workload evidence must fail closed"
                );
                let encoded_catalog = session.seal_encoded_dispatches().unwrap();
                assert_eq!(encoded_catalog.observations().len(), 30);
                assert_eq!(encoded_catalog.catalog_sha256().len(), 64);
                assert_eq!(
                    encoded_catalog.executed_catalog_sha256(),
                    executed_catalog_sha256
                );
                assert_eq!(
                    encoded_catalog.graph_configuration_sha256(),
                    candidate.execution().graph_configuration_sha256()
                );
                assert_eq!(
                    encoded_catalog.routing_policy_sha256(),
                    candidate.execution().routing_policy_sha256()
                );

                let mut incomplete = candidate.start_text_session(16).unwrap();
                incomplete
                    .forward(&prompt_tokens, &prompt_positions)
                    .unwrap();
                assert!(
                    incomplete.seal_encoded_dispatches().is_err(),
                    "an encoded catalog missing decode coverage must fail closed"
                );
            }
        }

        if expected_ffn_type == "q8_0" {
            let oversized = std::fs::File::create(&tensor_receipt_path).unwrap();
            oversized
                .set_len(MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES + 1)
                .unwrap();
            drop(oversized);
            assert!(
                verify_persisted_stored_evidence(dir.path(), &output, &verified_source, &context,)
                    .is_err(),
                "persisted replay must bound bytes read from one opened sidecar identity"
            );
            std::fs::write(
                &tensor_receipt_path,
                serde_json::to_vec(&persisted).unwrap(),
            )
            .unwrap();
        }

        let mut tampered = persisted.clone();
        tampered.tensor_lineages[0].stored.payload_sha256 = "0".repeat(64);
        assert!(
            crate::convert::tensor_lineage::validate_stored_tensor_conversion_receipt(&tampered)
                .is_err(),
            "a mutated tensor receipt must fail its canonical self-hash"
        );
        std::fs::write(&tensor_receipt_path, serde_json::to_vec(&tampered).unwrap()).unwrap();
        assert!(
            verify_persisted_stored_evidence(dir.path(), &output, &verified_source, &context,)
                .is_err(),
            "independent replay must reject a mutated sidecar"
        );
        std::fs::write(
            &tensor_receipt_path,
            serde_json::to_vec(&persisted).unwrap(),
        )
        .unwrap();

        let mut relabeled = persisted.clone();
        relabeled.policy.selector = if relabeled.policy.selector == "q8_0" {
            "q4_k_m".into()
        } else {
            "q8_0".into()
        };
        relabeled.receipt_sha256 =
            crate::convert::tensor_lineage::stored_tensor_conversion_receipt_sha256(&relabeled)
                .unwrap();
        crate::convert::tensor_lineage::validate_stored_tensor_conversion_receipt(&relabeled)
            .unwrap();
        std::fs::write(
            &tensor_receipt_path,
            serde_json::to_vec(&relabeled).unwrap(),
        )
        .unwrap();
        assert!(
            verify_persisted_stored_evidence(dir.path(), &output, &verified_source, &context,)
                .is_err(),
            "replay must rerun the declared policy and reject selector relabeling"
        );

        let mut wrong_commit = persisted.clone();
        wrong_commit.producer.git_commit = "1".repeat(40);
        wrong_commit.receipt_sha256 =
            crate::convert::tensor_lineage::stored_tensor_conversion_receipt_sha256(&wrong_commit)
                .unwrap();
        std::fs::write(
            &tensor_receipt_path,
            serde_json::to_vec(&wrong_commit).unwrap(),
        )
        .unwrap();
        assert!(
            verify_persisted_stored_evidence(dir.path(), &output, &verified_source, &context,)
                .is_err(),
            "replay must reject a different claimed converter revision"
        );
        std::fs::write(
            &tensor_receipt_path,
            serde_json::to_vec(&persisted).unwrap(),
        )
        .unwrap();
        assert_eq!(
            std::fs::metadata(&output).unwrap().len(),
            verified.artifact_bytes()
        );
        assert_eq!(
            crate::core::sha256::compute_file_sha256(&output).unwrap(),
            verified.artifact_sha256()
        );
        if expected_ffn_type == "q4_k" {
            let original_artifact = std::fs::read(&output).unwrap();
            let mut mutated_artifact = original_artifact.clone();
            let last = mutated_artifact.last_mut().unwrap();
            *last ^= 0x01;
            std::fs::write(&output, &mutated_artifact).unwrap();
            assert!(
                verify_persisted_stored_evidence(dir.path(), &output, &verified_source, &context,)
                    .is_err(),
                "independent replay must reject changed GGUF bytes"
            );
            std::fs::write(&output, original_artifact).unwrap();
        }
    }
}
