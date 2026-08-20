use super::*;
use crate::convert::tensor_lineage::MAX_STORED_TENSOR_CONVERSION_RECEIPT_BYTES;
use crate::core::integrity::{ShardIntegrity, compute_git_blob_sha1};
use crate::intelligence::dynamic_allocator::producer::SourceTensorRecord;
use crate::intelligence::measured_auto_quant::SourceIdentity;
use half::f16;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::json;
use sha2::{Digest, Sha256};

#[test]
fn dense_qwen_stored_evidence_runs_the_authoritative_conversion_loop() {
    let dir = tempfile::tempdir().unwrap();
    let weight_values: Vec<f16> = (0..32 * 256)
        .map(|index| f16::from_f32((index as f32 % 97.0 - 48.0) / 32.0))
        .collect();
    let weight_bytes: Vec<u8> = weight_values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    let weight_view = TensorView::new(Dtype::F16, vec![32, 256], &weight_bytes).unwrap();
    let ffn_values: Vec<f16> = (0..256 * 256)
        .map(|index| f16::from_f32((index as f32 % 131.0 - 65.0) / 48.0))
        .collect();
    let ffn_bytes: Vec<u8> = ffn_values
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect();
    let ffn_view = TensorView::new(Dtype::F16, vec![256, 256], &ffn_bytes).unwrap();
    let model_bytes = safetensors::tensor::serialize(
        vec![
            ("model.embed_tokens.weight".to_string(), &weight_view),
            ("model.layers.0.mlp.up_proj.weight".to_string(), &ffn_view),
        ],
        None,
    )
    .unwrap();
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, &model_bytes).unwrap();

    let config = json!({
        "model_type": "qwen3_5_text",
        "architectures": ["Qwen3_5ForCausalLM"],
        "hidden_size": 256,
        "num_hidden_layers": 1,
        "full_attention_interval": 1,
        "layer_types": ["full_attention"],
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 0.000001,
        "vocab_size": 32,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
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
    let context = VerifiedConversionEvidenceContext::for_test(
        source,
        verified_manifest_sha,
        vec![
            SourceTensorRecord {
                name: "model.embed_tokens.weight".into(),
                source_shape: vec![32, 256],
                source_dtype: "F16".into(),
                source_byte_len: weight_bytes.len() as u64,
                source_tensor_sha256: hex::encode(Sha256::digest(&weight_bytes)),
            },
            SourceTensorRecord {
                name: "model.layers.0.mlp.up_proj.weight".into(),
                source_shape: vec![256, 256],
                source_dtype: "F16".into(),
                source_byte_len: ffn_bytes.len() as u64,
                source_tensor_sha256: hex::encode(Sha256::digest(&ffn_bytes)),
            },
        ],
        vec![
            (
                "model.embed_tokens.weight".into(),
                crate::convert::tensor_lineage::ConversionSourceDisposition::Variable,
            ),
            (
                "model.layers.0.mlp.up_proj.weight".into(),
                crate::convert::tensor_lineage::ConversionSourceDisposition::Variable,
            ),
        ],
    );
    for (selector, output_name, expected_embedding_type, expected_ffn_type) in [
        (
            QuantSelector::Standard(LlamaFtype::MostlyQ8_0),
            "tiny-qwen-q8.gguf",
            "q8_0",
            "q8_0",
        ),
        (
            QuantSelector::Standard(LlamaFtype::MostlyQ4_K_M),
            "tiny-qwen-q4-k-m.gguf",
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
                mmproj: false,
                remote_source: Some(remote),
            },
            verified_source.clone(),
            context.clone(),
        )
        .unwrap();
        assert!(output.is_file());
        assert_eq!(verified.receipt().tensor_lineages.len(), 2);
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
        assert_eq!(embedding.stored.ggml_type_name, expected_embedding_type);
        assert_eq!(ffn_up.stored.ggml_type_name, expected_ffn_type);
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
