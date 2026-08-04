use std::fs;

use mlx_native::gguf::GgufFile;
use mlx_native::{DType, MlxDevice};

use crate::backends::gguf::writer::GgufWriter;
use crate::quantize::ggml_quants::GgmlType;

use super::residency::{Deepseek4Weights, WeightLookupError, WeightResidencyError};
use super::weights::{required_tensor_specs, TensorRole, WeightCatalogError};
use super::Deepseek4Config;

fn tiny_config() -> Deepseek4Config {
    Deepseek4Config {
        num_hidden_layers: 2,
        hidden_size: 32,
        hidden_size_out: 128,
        max_position_embeddings: 128,
        vocab_size: 32,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 32,
        rope_head_dim: 8,
        rope_theta: 10000.0,
        rope_factor: 1.0,
        original_context_length: 128,
        yarn_beta_fast: 32.0,
        yarn_beta_slow: 1.0,
        q_lora_rank: 32,
        o_lora_rank: 32,
        output_groups: 2,
        sliding_window: 128,
        compress_ratios: vec![4, 128],
        compress_rope_theta: 160000.0,
        index_num_heads: 2,
        index_head_dim: 16,
        index_top_k: 4,
        rms_norm_eps: 1e-6,
        num_experts: 4,
        num_experts_per_tok: 2,
        num_shared_experts: 1,
        expert_intermediate_size: 32,
        route_scale: 1.5,
        normalize_topk: true,
        swiglu_clamp_experts: vec![10.0; 2],
        swiglu_clamp_shared: vec![10.0; 2],
        hyper_connection_count: 4,
        hyper_connection_sinkhorn_iterations: 20,
        hyper_connection_epsilon: 1e-6,
        hash_layer_count: 1,
    }
}

fn tensor_type(role: TensorRole, name: &str, bad_lookup: bool) -> GgmlType {
    match role {
        TensorRole::RawMatrix => GgmlType::Q4_0,
        TensorRole::ElementwiseF32 if name.contains("_ape.weight") => GgmlType::Q4_0,
        TensorRole::ElementwiseF32 => GgmlType::F32,
        TensorRole::IntegerLookupI32 if bad_lookup => GgmlType::F32,
        TensorRole::IntegerLookupI32 => GgmlType::I32,
    }
}

fn payload(spec_shape: &[usize], ggml_type: GgmlType) -> Vec<u8> {
    let rows = if spec_shape.len() <= 1 {
        1
    } else {
        spec_shape[..spec_shape.len() - 1].iter().product()
    };
    let columns = *spec_shape.last().unwrap();
    let bytes = rows * ggml_type.row_size(columns);
    match ggml_type {
        GgmlType::F32 => (0..spec_shape.iter().product::<usize>())
            .flat_map(|_| 1.0_f32.to_le_bytes())
            .collect(),
        GgmlType::I32 => (0..spec_shape.iter().product::<usize>())
            .flat_map(|value| (value as i32).to_le_bytes())
            .collect(),
        _ => vec![0; bytes],
    }
}

fn open_fixture(
    cfg: &Deepseek4Config,
    omit_last: bool,
    bad_lookup: bool,
) -> (tempfile::TempDir, GgufFile) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("weights.gguf");
    let mut specs = required_tensor_specs(cfg);
    if omit_last {
        specs.pop();
    }
    let mut writer = GgufWriter::new(fs::File::create(&path).unwrap());
    writer.write_header(specs.len() as u64, 0).unwrap();
    let mut types = Vec::with_capacity(specs.len());
    for spec in &specs {
        let ggml_type = tensor_type(spec.role, &spec.name, bad_lookup);
        let dims: Vec<u64> = spec.shape.iter().rev().map(|&dim| dim as u64).collect();
        writer
            .reserve_tensor_info(&spec.name, &dims, ggml_type)
            .unwrap();
        types.push(ggml_type);
    }
    writer.pad_to_alignment().unwrap();
    for (index, (spec, ggml_type)) in specs.iter().zip(types).enumerate() {
        writer
            .stream_tensor_payload(index, &payload(&spec.shape, ggml_type))
            .unwrap();
    }
    writer.finalize().unwrap();
    (directory, GgufFile::open(&path).unwrap())
}

#[test]
fn loader_preserves_raw_blocks_and_expands_only_elementwise_state() {
    let cfg = tiny_config();
    let (_directory, gguf) = open_fixture(&cfg, false, false);
    let specs = required_tensor_specs(&cfg);
    let expected_bytes = specs
        .iter()
        .map(|spec| match spec.role {
            TensorRole::ElementwiseF32 => spec.shape.iter().product::<usize>() * 4,
            _ => gguf.tensor_info(&spec.name).unwrap().byte_len,
        })
        .sum::<usize>() as u64;

    let _gpu = crate::inference::hf2q_gpu_test_lock();
    let weights = Deepseek4Weights::load_from_gguf(&gguf, &cfg, MlxDevice::new().unwrap()).unwrap();
    assert_eq!(weights.len(), specs.len());
    assert!(!weights.is_empty());
    assert_eq!(weights.resident_bytes(), expected_bytes);

    let embedding = weights.raw_matrix("token_embd.weight").unwrap();
    assert_eq!(embedding.dtype(), DType::U8);
    assert_eq!(embedding.byte_len(), 32 * 18);

    let norm = weights.f32_state("output_norm.weight").unwrap();
    assert_eq!(norm.dtype(), DType::F32);
    assert_eq!(norm.byte_len(), 32 * 4);
    let ape = weights
        .f32_state("blk.0.attn_compressor_ape.weight")
        .unwrap();
    assert_eq!(ape.dtype(), DType::F32);
    assert_eq!(ape.byte_len(), 4 * 64 * 4);

    let lookup = weights.i32_lookup("blk.0.ffn_gate_tid2eid.weight").unwrap();
    assert_eq!(lookup.dtype(), DType::I32);
    assert_eq!(lookup.as_slice::<i32>().unwrap()[1], 1);

    assert!(matches!(
        weights.f32_state("token_embd.weight"),
        Err(WeightLookupError::RoleMismatch {
            expected: TensorRole::ElementwiseF32,
            actual: TensorRole::RawMatrix,
            ..
        })
    ));
    assert!(matches!(
        weights.raw_matrix("missing.weight"),
        Err(WeightLookupError::Missing { .. })
    ));
}

#[test]
fn loader_rejects_catalog_and_i32_storage_before_residency() {
    let cfg = tiny_config();
    let (_directory, missing) = open_fixture(&cfg, true, false);
    let _gpu = crate::inference::hf2q_gpu_test_lock();
    assert!(matches!(
        Deepseek4Weights::load_from_gguf(&missing, &cfg, MlxDevice::new().unwrap()),
        Err(WeightResidencyError::Catalog(
            WeightCatalogError::Missing { .. }
        ))
    ));

    let (_directory, wrong_type) = open_fixture(&cfg, false, true);
    assert!(matches!(
        Deepseek4Weights::load_from_gguf(&wrong_type, &cfg, MlxDevice::new().unwrap()),
        Err(WeightResidencyError::StorageType {
            role: TensorRole::IntegerLookupI32,
            ..
        })
    ));
}
