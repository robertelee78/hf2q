use half::bf16;
use safetensors::tensor::TensorView;
use safetensors::Dtype;
use sha1::Sha1;
use sha2::{Digest, Sha256};

use super::topology::{
    admit_qwen35_bf16_topology, expected_profile_for_config_for_test, Qwen35FutureDType,
    Qwen35QGateBranch, Qwen35SourceTransformV1,
};
use super::*;
use crate::core::integrity::ShardIntegrity;
use crate::core::provenance::{compute_source_bundle_sha256, SourceShard};
use crate::input::integrity::verify_conversion_manifest;
use crate::intelligence::dynamic_allocator::producer::{
    build_tensor_partition, derive_source_tensor_inventory, NonVariableDisposition,
    NonVariableTensor, VerifiedSourceTensorInventory,
};
use crate::intelligence::dynamic_allocator::{
    ScalarDType, TensorAllocationUnit, TensorMember, TensorOperation,
};
use crate::intelligence::measured_auto_quant::SourceIdentity;

const MODEL_ID: &str = "Qwen/Qwen3.8-topology-test";
const REVISION: &str = "1123456789abcdef0123456789abcdef01234567";

#[derive(Clone)]
pub(super) struct TensorSpec {
    pub(super) name: String,
    pub(super) shape: Vec<usize>,
}

pub(super) struct TopologyFixture {
    pub(super) temp: tempfile::TempDir,
    verified: crate::input::integrity::VerifiedSourceManifest,
    inventory: VerifiedSourceTensorInventory,
    partition: crate::intelligence::dynamic_allocator::producer::TensorPartitionManifest,
    units: Vec<TensorAllocationUnit>,
}

pub(super) fn config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 4,
            "intermediate_size": 8,
            "vocab_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 2,
            "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "max_position_embeddings": 128,
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "attn_output_gate": true
        }
    })
}

fn specs() -> Vec<TensorSpec> {
    let mut specs = vec![
        spec("model.language_model.embed_tokens.weight", &[8, 4]),
        spec("model.language_model.norm.weight", &[4]),
        spec("lm_head.weight", &[8, 4]),
    ];
    for layer in 0..2 {
        let p = format!("model.language_model.layers.{layer}");
        specs.extend([
            spec(&format!("{p}.input_layernorm.weight"), &[4]),
            spec(&format!("{p}.post_attention_layernorm.weight"), &[4]),
            spec(&format!("{p}.mlp.gate_proj.weight"), &[8, 4]),
            spec(&format!("{p}.mlp.up_proj.weight"), &[8, 4]),
            spec(&format!("{p}.mlp.down_proj.weight"), &[4, 8]),
        ]);
        if layer == 0 {
            specs.extend([
                spec(&format!("{p}.linear_attn.A_log"), &[2]),
                spec(&format!("{p}.linear_attn.conv1d.weight"), &[8, 1, 2]),
                spec(&format!("{p}.linear_attn.dt_bias"), &[2]),
                spec(&format!("{p}.linear_attn.in_proj_a.weight"), &[2, 4]),
                spec(&format!("{p}.linear_attn.in_proj_b.weight"), &[2, 4]),
                spec(&format!("{p}.linear_attn.in_proj_qkv.weight"), &[8, 4]),
                spec(&format!("{p}.linear_attn.in_proj_z.weight"), &[4, 4]),
                spec(&format!("{p}.linear_attn.norm.weight"), &[2]),
                spec(&format!("{p}.linear_attn.out_proj.weight"), &[4, 4]),
            ]);
        } else {
            specs.extend([
                spec(&format!("{p}.self_attn.q_proj.weight"), &[8, 4]),
                spec(&format!("{p}.self_attn.k_proj.weight"), &[2, 4]),
                spec(&format!("{p}.self_attn.v_proj.weight"), &[2, 4]),
                spec(&format!("{p}.self_attn.o_proj.weight"), &[4, 4]),
                spec(&format!("{p}.self_attn.q_norm.weight"), &[2]),
                spec(&format!("{p}.self_attn.k_norm.weight"), &[2]),
            ]);
        }
    }
    specs.extend([
        spec("mtp.fc.weight", &[4, 8]),
        spec("mtp.layers.0.input_layernorm.weight", &[4]),
        spec("mtp.layers.0.post_attention_layernorm.weight", &[4]),
        spec("mtp.layers.0.mlp.gate_proj.weight", &[8, 4]),
        spec("mtp.layers.0.mlp.up_proj.weight", &[8, 4]),
        spec("mtp.layers.0.mlp.down_proj.weight", &[4, 8]),
        spec("mtp.layers.0.self_attn.q_proj.weight", &[8, 4]),
        spec("mtp.layers.0.self_attn.k_proj.weight", &[2, 4]),
        spec("mtp.layers.0.self_attn.v_proj.weight", &[2, 4]),
        spec("mtp.layers.0.self_attn.o_proj.weight", &[4, 4]),
        spec("mtp.layers.0.self_attn.q_norm.weight", &[2]),
        spec("mtp.layers.0.self_attn.k_norm.weight", &[2]),
        spec("mtp.norm.weight", &[4]),
        spec("mtp.pre_fc_norm_embedding.weight", &[4]),
        spec("mtp.pre_fc_norm_hidden.weight", &[4]),
        spec("model.visual.patch.weight", &[2, 2]),
    ]);
    specs
}

fn spec(name: &str, shape: &[usize]) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        shape: shape.to_vec(),
    }
}

fn git_record(filename: &str, bytes: &[u8]) -> ShardIntegrity {
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
}

fn lfs_record(filename: &str, bytes: &[u8]) -> ShardIntegrity {
    let sha256 = hex::encode(Sha256::digest(bytes));
    ShardIntegrity {
        filename: filename.into(),
        bytes: bytes.len() as u64,
        sha256: Some(sha256.clone()),
        hf_etag: sha256,
        is_lfs: true,
    }
}

pub(super) fn fixture(
    dtype: Dtype,
    mutate: impl FnOnce(&mut serde_json::Value, &mut Vec<TensorSpec>),
) -> TopologyFixture {
    fixture_with_payload(dtype, mutate, |tensor_index, element_index| {
        (tensor_index + element_index) as u16
    })
}

pub(super) fn finite_bf16_fixture(
    mutate: impl FnOnce(&mut serde_json::Value, &mut Vec<TensorSpec>),
) -> TopologyFixture {
    fixture_with_payload(Dtype::BF16, mutate, |tensor_index, element_index| {
        let integer = ((tensor_index * 31 + element_index * 17) % 31) as i32 - 15;
        bf16::from_f32(integer as f32 / 1024.0).to_bits()
    })
}

fn fixture_with_payload(
    dtype: Dtype,
    mutate: impl FnOnce(&mut serde_json::Value, &mut Vec<TensorSpec>),
    payload_word: impl Fn(usize, usize) -> u16,
) -> TopologyFixture {
    let temp = tempfile::tempdir().unwrap();
    let mut config_value = config();
    let mut tensor_specs = specs();
    mutate(&mut config_value, &mut tensor_specs);
    let config_bytes = serde_json::to_vec(&config_value).unwrap();
    std::fs::write(temp.path().join("config.json"), &config_bytes).unwrap();

    let payloads: Vec<Vec<u8>> = tensor_specs
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            let numel = tensor.shape.iter().product::<usize>();
            (0..numel)
                .flat_map(|offset| payload_word(index, offset).to_le_bytes())
                .collect()
        })
        .collect();
    let views: Vec<TensorView<'_>> = tensor_specs
        .iter()
        .zip(&payloads)
        .map(|(tensor, payload)| TensorView::new(dtype, tensor.shape.clone(), payload).unwrap())
        .collect();
    let shard = safetensors::tensor::serialize(
        tensor_specs
            .iter()
            .zip(&views)
            .map(|(tensor, view)| (tensor.name.clone(), view))
            .collect::<Vec<_>>(),
        None,
    )
    .unwrap();
    std::fs::write(temp.path().join("model.safetensors"), &shard).unwrap();

    let config_record = git_record("config.json", &config_bytes);
    let shard_record = lfs_record("model.safetensors", &shard);
    let verified = verify_conversion_manifest(
        MODEL_ID,
        REVISION,
        temp.path(),
        vec![config_record, shard_record.clone()],
    )
    .unwrap();
    let source = SourceIdentity {
        model_id: MODEL_ID.into(),
        revision: REVISION.into(),
        config_sha256: hex::encode(Sha256::digest(&config_bytes)),
        tensor_bundle_sha256: compute_source_bundle_sha256(&[SourceShard::from_integrity(
            &shard_record,
        )])
        .unwrap(),
        tokenizer_bundle_sha256: "3".repeat(64),
        chat_template_sha256: "4".repeat(64),
    };
    let inventory = derive_source_tensor_inventory(temp.path(), source, &verified).unwrap();
    let source_dtype = if dtype == Dtype::BF16 {
        ScalarDType::Bf16
    } else {
        ScalarDType::F16
    };
    let variable_names: Vec<_> = inventory
        .manifest()
        .tensors
        .iter()
        .filter(|tensor| {
            !tensor.name.starts_with("mtp.") && !tensor.name.starts_with("model.visual.")
        })
        .map(|tensor| tensor.name.clone())
        .collect();
    let members = variable_names
        .iter()
        .map(|name| {
            let source = inventory
                .manifest()
                .tensors
                .iter()
                .find(|tensor| tensor.name == *name)
                .unwrap();
            TensorMember {
                name: name.clone(),
                shape: source.source_shape.clone(),
                role: "source_teacher".into(),
                source_dtype,
                source_tensor_sha256: source.source_tensor_sha256.clone(),
                layer_index: None,
                expert_index: None,
            }
        })
        .collect();
    let units = vec![TensorAllocationUnit {
        unit_id: "dense-qwen-base".into(),
        members,
        expected_expert_ids: Vec::new(),
        operations: vec![TensorOperation {
            operation_id: "source-teacher".into(),
            graph_path: "qwen35.source_teacher".into(),
            tensor_names: variable_names,
        }],
        options: Vec::new(),
    }];
    let non_variable = inventory
        .manifest()
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.name.starts_with("mtp.") || tensor.name.starts_with("model.visual.")
        })
        .map(|tensor| NonVariableTensor {
            source: tensor.clone(),
            disposition: if tensor.name.starts_with("model.visual.") {
                NonVariableDisposition::Excluded
            } else {
                NonVariableDisposition::Protected
            },
            reason: "source-teacher topology fixture".into(),
        })
        .collect();
    let partition = build_tensor_partition(&inventory, &units, non_variable).unwrap();
    TopologyFixture {
        temp,
        verified,
        inventory,
        partition,
        units,
    }
}

pub(super) fn open(fixture: &TopologyFixture) -> anyhow::Result<VerifiedQwenSourceSnapshot> {
    open_verified_qwen_source_snapshot(
        fixture.temp.path(),
        &fixture.verified,
        &fixture.inventory,
        &fixture.partition,
        &fixture.units,
        QwenSourceSnapshotLimits {
            max_shards: 1,
            max_tensors: 64,
            max_header_bytes_per_shard: 64 * 1024,
            max_total_header_bytes: 64 * 1024,
            max_config_bytes: 64 * 1024,
            max_total_source_bytes: fixture
                .verified
                .records()
                .iter()
                .map(|record| record.bytes)
                .sum(),
        },
    )
}

#[test]
fn official_config_has_exact_closed_topology_counts() {
    let config: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../tests/fixtures/qwen38/config.json"
    ))
    .unwrap();
    let (sources, bf16, f32, bf16_bytes, f32_bytes, max_bytes, transforms) =
        expected_profile_for_config_for_test(&config).unwrap();
    assert_eq!(sources, 866);
    assert_eq!(bf16, 514);
    assert_eq!(f32, 353);
    assert_eq!(bf16_bytes, 53_786_705_920);
    assert_eq!(f32_bytes, 10_582_016);
    assert_eq!(bf16_bytes + f32_bytes, 53_797_287_936);
    assert_eq!(max_bytes, 2_542_796_800);
    assert_eq!(
        transforms,
        [290, 161, 240, 48, 48, 48, 32],
        "identity, add-one, reorder, reorder+neg-exp, squeeze+reorder, per-row reorder, split"
    );
}

#[test]
fn two_layer_snapshot_admits_exact_delta_full_and_q_gate_fanout() {
    let fixture = fixture(Dtype::BF16, |_, _| {});
    let topology = admit_qwen35_bf16_topology(open(&fixture).unwrap()).unwrap();
    assert_eq!(topology.source_count(), 44);
    assert_eq!(topology.future_tensor_count(), 29);
    assert_eq!(topology.future_bf16_tensor_count(), 18);
    assert_eq!(topology.future_f32_tensor_count(), 11);
    assert_eq!(topology.topology_sha256().len(), 64);

    let q = topology
        .records_for_test()
        .iter()
        .find(|record| {
            record
                .source_name
                .ends_with("layers.1.self_attn.q_proj.weight")
        })
        .unwrap();
    assert_eq!(q.outputs.len(), 2);
    assert_eq!(q.outputs[0].shape, vec![4, 4]);
    assert_eq!(q.outputs[0].dtype, Qwen35FutureDType::Bf16);
    assert!(matches!(
        q.outputs[0].transform,
        Qwen35SourceTransformV1::SplitInterleavedQGate {
            branch: Qwen35QGateBranch::Query,
            ..
        }
    ));
    assert!(matches!(
        q.outputs[1].transform,
        Qwen35SourceTransformV1::SplitInterleavedQGate {
            branch: Qwen35QGateBranch::Gate,
            ..
        }
    ));
    let conv = topology
        .records_for_test()
        .iter()
        .find(|record| record.source_name.ends_with("linear_attn.conv1d.weight"))
        .unwrap();
    assert_eq!(conv.outputs[0].shape, vec![8, 2]);
    assert_eq!(conv.outputs[0].dtype, Qwen35FutureDType::F32);
}

#[test]
fn topology_rejects_f16_shape_drift_and_irregular_schedule() {
    let f16 = fixture(Dtype::F16, |_, _| {});
    let error = admit_qwen35_bf16_topology(open(&f16).unwrap())
        .err()
        .expect("F16 topology must reject");
    assert!(error.to_string().contains("rejects non-BF16"));

    let wrong_shape = fixture(Dtype::BF16, |_, specs| {
        specs
            .iter_mut()
            .find(|tensor| tensor.name == "lm_head.weight")
            .unwrap()
            .shape = vec![7, 4];
    });
    let error = admit_qwen35_bf16_topology(open(&wrong_shape).unwrap())
        .err()
        .expect("shape drift must reject");
    assert!(error.to_string().contains("lm_head.weight has shape"));

    let irregular = fixture(Dtype::BF16, |config, _| {
        config["text_config"]["layer_types"] =
            serde_json::json!(["full_attention", "full_attention"]);
    });
    let error = admit_qwen35_bf16_topology(open(&irregular).unwrap())
        .err()
        .expect("irregular schedule must reject");
    assert!(error.to_string().contains("production mapper"));
}

#[test]
fn topology_rejects_zero_heads_and_huge_layer_count_without_panicking_or_allocating() {
    let zero_heads = fixture(Dtype::BF16, |config, _| {
        config["text_config"]["linear_num_key_heads"] = serde_json::json!(0);
    });
    let error = admit_qwen35_bf16_topology(open(&zero_heads).unwrap())
        .err()
        .expect("zero key heads must reject");
    assert!(error.to_string().contains("production mapper"));

    let huge_layers = fixture(Dtype::BF16, |config, _| {
        config["text_config"]["num_hidden_layers"] = serde_json::json!(1_000_000_000_u64);
        config["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("layer_types");
    });
    let error = admit_qwen35_bf16_topology(open(&huge_layers).unwrap())
        .err()
        .expect("huge declared topology must reject before schedule allocation");
    assert!(error.to_string().contains("requires"));
}
