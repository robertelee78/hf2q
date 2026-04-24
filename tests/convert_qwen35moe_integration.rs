//! Integration tests for the qwen35moe (MoE) convert pipeline.
//!
//! Uses a synthetic tiny qwen35moe model with deterministic seed weights.
//! The model is deliberately small (4 layers, hidden=64, 4 experts) to
//! keep test runtime and memory usage negligible.
//!
//! Assertions follow the ADR-012 P7 spec:
//!   - `.gguf` structurally valid (magic, version ≥ 3, tensor_count > 0, kv_count > 0)
//!   - Sidecar files present byte-identical in output dir
//!   - Missing sidecars silently skipped
//!
//! Hparams for the synthetic MoE model:
//!   hidden_size: 64
//!   num_hidden_layers: 4
//!   num_attention_heads: 4
//!   num_key_value_heads: 1
//!   head_dim: 16
//!   linear_num_value_heads: 8
//!   num_experts: 4
//!   num_experts_per_tok: 2
//!   moe_intermediate_size: 16
//!   shared_expert_intermediate_size: 16
//!   vocab_size: 128
//!   full_attention_interval: 4  (layer 3 is full_attention; 0,1,2 are linear)

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use assert_cmd::Command;

// ─── GGUF binary constants ───────────────────────────────────────────────────

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"
const GGUF_VERSION: u32 = 3;

// ─── Synthetic qwen35moe model config ────────────────────────────────────────

/// Note: 4 experts is the smallest count that exercises expert-merge logic
/// without requiring real MoE scale. shared_expert_intermediate_size > 0
/// ensures the shared-expert singleton path is tested.
const QWEN35MOE_CONFIG: &str = r#"{
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "model_type": "qwen3_5_moe",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "vocab_size": 128,
    "head_dim": 16,
    "linear_num_value_heads": 8,
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000.0,
    "rope_scaling": {
        "mrope_section": [3, 3, 2],
        "type": "mrope"
    },
    "mtp_num_hidden_layers": 0,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16"
}"#;

// ─── Safetensors builder ─────────────────────────────────────────────────────

/// Build safetensors bytes for a 4-layer, 4-expert qwen35moe model.
///
/// Tensor naming follows the HF safetensors conventions that the convert
/// pipeline's tensor-name mapper (ADR-012 P4 Decision 8) transforms.
///
/// Expert naming: `model.layers.L.mlp.experts.E.{gate,up,down}_proj.weight`
/// Shared expert: `model.layers.L.mlp.shared_expert.{gate,up,down}_proj.weight`
/// Router:        `model.layers.L.mlp.gate.weight`
fn build_qwen35moe_safetensors() -> Vec<u8> {
    let hidden: usize = 64;
    let vocab: usize = 128;
    let moe_inter: usize = 16;
    let shared_inter: usize = 16;
    let num_experts: usize = 4;
    let num_heads: usize = 4;
    let kv_heads: usize = 1;
    let head_dim: usize = 16;
    let lin_v_heads: usize = 8;
    let num_layers: usize = 4;
    let f16 = 2usize;

    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();

    // Global tensors
    tensors.push((
        "model.embed_tokens.weight".into(),
        vec![vocab, hidden],
        "F16",
        vec![0u8; vocab * hidden * f16],
    ));
    tensors.push((
        "lm_head.weight".into(),
        vec![vocab, hidden],
        "F16",
        vec![1u8; vocab * hidden * f16],
    ));
    tensors.push((
        "model.norm.weight".into(),
        vec![hidden],
        "F16",
        vec![0u8; hidden * f16],
    ));

    for layer in 0..num_layers {
        let is_full = layer == 3;
        let prefix = format!("model.layers.{layer}");

        // Shared norms
        tensors.push((
            format!("{prefix}.input_layernorm.weight"),
            vec![hidden],
            "F16",
            vec![0u8; hidden * f16],
        ));
        tensors.push((
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![hidden],
            "F16",
            vec![0u8; hidden * f16],
        ));

        if is_full {
            // Standard full-attention block
            let q_size = num_heads * head_dim;
            let kv_size = kv_heads * head_dim;
            tensors.push((
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![q_size, hidden],
                "F16",
                vec![2u8; q_size * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![kv_size, hidden],
                "F16",
                vec![3u8; kv_size * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![kv_size, hidden],
                "F16",
                vec![4u8; kv_size * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![hidden, q_size],
                "F16",
                vec![5u8; hidden * q_size * f16],
            ));
            tensors.push((
                format!("{prefix}.self_attn.gate.weight"),
                vec![1, hidden],
                "F16",
                vec![0u8; hidden * f16],
            ));
        } else {
            // Linear attention block — shapes match the spec per
            // convert_hf_to_gguf.py Qwen3Next/Qwen3_5 + llama-model.cpp.
            // Pre-2026-04-24 fixtures used [num_heads, ...] instead of
            // [linear_num_value_heads, ...]; silently worked only because
            // the V-head reorder transform was never invoked.
            let nk = 4usize;
            let nv = 8usize;
            let head_k_dim = 16usize;
            let head_v_dim = 16usize;
            let qkv_rows = nk * head_k_dim + nk * head_k_dim + nv * head_v_dim;
            tensors.push((
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                vec![qkv_rows, hidden],
                "F16",
                vec![6u8; qkv_rows * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.out_proj.weight"),
                vec![hidden, nv * head_v_dim],
                "F16",
                vec![7u8; hidden * nv * head_v_dim * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.in_proj_a.weight"),
                vec![nv, hidden],
                "F16",
                vec![0u8; nv * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                vec![nv, hidden],
                "F16",
                vec![0u8; nv * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                vec![nv * head_v_dim, hidden],
                "F16",
                vec![0u8; nv * head_v_dim * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.A_log"),
                vec![nv],
                "F32",
                vec![0u8; nv * 4],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.dt_bias"),
                vec![nv],
                "F32",
                vec![0u8; nv * 4],
            ));
            let conv_channels = 2 * nk * head_k_dim + nv * head_v_dim;
            let conv_kernel_dim = 4usize;
            tensors.push((
                format!("{prefix}.linear_attn.conv1d.weight"),
                vec![conv_channels, 1, conv_kernel_dim],
                "F16",
                vec![0u8; conv_channels * conv_kernel_dim * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.norm.weight"),
                vec![hidden],
                "F16",
                vec![0u8; hidden * f16],
            ));
        }

        // MoE FFN: router + per-expert + shared expert
        // Router gate
        tensors.push((
            format!("{prefix}.mlp.gate.weight"),
            vec![num_experts, hidden],
            "F16",
            vec![0u8; num_experts * hidden * f16],
        ));

        // Per-expert weights (gate_proj, up_proj, down_proj)
        for expert in 0..num_experts {
            tensors.push((
                format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                vec![moe_inter, hidden],
                "F16",
                vec![0u8; moe_inter * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                vec![moe_inter, hidden],
                "F16",
                vec![0u8; moe_inter * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                vec![hidden, moe_inter],
                "F16",
                vec![0u8; hidden * moe_inter * f16],
            ));
        }

        // Shared expert (emitted as singletons, not merged)
        tensors.push((
            format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            vec![shared_inter, hidden],
            "F16",
            vec![8u8; shared_inter * hidden * f16],
        ));
        tensors.push((
            format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            vec![shared_inter, hidden],
            "F16",
            vec![9u8; shared_inter * hidden * f16],
        ));
        tensors.push((
            format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            vec![hidden, shared_inter],
            "F16",
            vec![10u8; hidden * shared_inter * f16],
        ));
    }

    build_safetensors_bytes(tensors)
}

fn build_safetensors_bytes(tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)>) -> Vec<u8> {
    let mut header_map = BTreeMap::new();
    let mut offset = 0usize;
    let mut payload = Vec::new();

    for (name, shape, dtype, data) in &tensors {
        let end = offset + data.len();
        header_map.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        payload.extend_from_slice(data);
        offset = end;
    }

    let header_json = serde_json::to_string(&header_map).unwrap();
    let hbytes = header_json.as_bytes();
    let hsize = hbytes.len() as u64;

    let mut out = Vec::new();
    out.extend_from_slice(&hsize.to_le_bytes());
    out.extend_from_slice(hbytes);
    out.extend_from_slice(&payload);
    out
}

/// Write a synthetic qwen35moe HF model directory.
fn setup_qwen35moe(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), QWEN35MOE_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    fs::write(dir.join("generation_config.json"), r#"{"do_sample":false}"#).unwrap();
    fs::write(dir.join("special_tokens_map.json"), r#"{"bos_token":"<|im_start|>"}"#).unwrap();
    fs::write(dir.join("chat_template.jinja"), "{% for msg in messages %}{{ msg.content }}{% endfor %}").unwrap();
    let st = build_qwen35moe_safetensors();
    fs::write(dir.join("model.safetensors"), st).unwrap();
}

// ─── GGUF header reader ──────────────────────────────────────────────────────

struct GgufHeader {
    version: u32,
    tensor_count: u64,
    kv_count: u64,
}

impl GgufHeader {
    fn read(data: &[u8]) -> Self {
        assert!(data.len() >= 24, "GGUF file too small: {} bytes", data.len());
        assert_eq!(&data[0..4], &GGUF_MAGIC, "Bad GGUF magic");
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
        GgufHeader { version, tensor_count, kv_count }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_convert_qwen35moe_q4_produces_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35moe-q4-input");
    let output = tmp.path().join("qwen35moe-q4-output");
    setup_qwen35moe(&input);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4",
            "--output", output.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let gguf_files: Vec<_> = fs::read_dir(&output)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .collect();
    assert!(!gguf_files.is_empty(), "No GGUF produced for qwen35moe");
}

#[test]
fn test_convert_qwen35moe_gguf_has_valid_magic_and_version() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35moe-hdr-input");
    let output = tmp.path().join("qwen35moe-hdr-output");
    setup_qwen35moe(&input);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4",
            "--output", output.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let gguf_path = fs::read_dir(&output)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .expect("No GGUF file")
        .path();

    let data = fs::read(&gguf_path).unwrap();
    let hdr = GgufHeader::read(&data);

    assert_eq!(hdr.version, GGUF_VERSION, "GGUF version must be 3");
    assert!(hdr.tensor_count > 0, "Tensor count must be > 0");
    assert!(hdr.kv_count > 0, "KV metadata count must be > 0");
}

#[test]
fn test_convert_qwen35moe_sidecars_copied_byte_identical() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35moe-sc-input");
    let output = tmp.path().join("qwen35moe-sc-output");
    setup_qwen35moe(&input);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4",
            "--output", output.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    // All sidecar files are present in setup_qwen35moe — all should be copied.
    let sidecars = [
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "config.json",
    ];

    for sidecar in &sidecars {
        let src = input.join(sidecar);
        let dst = output.join(sidecar);
        assert!(
            dst.exists(),
            "Sidecar '{sidecar}' must be present in qwen35moe output"
        );
        assert_eq!(
            fs::read(&src).unwrap(),
            fs::read(&dst).unwrap(),
            "Sidecar '{sidecar}' must be byte-identical"
        );
    }
}

#[test]
fn test_convert_qwen35moe_missing_sidecar_skipped_silently() {
    // Omit all sidecars; verify the convert does not fail.
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35moe-nosc-input");
    let output = tmp.path().join("qwen35moe-nosc-output");

    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("config.json"), QWEN35MOE_CONFIG).unwrap();
    let st = build_qwen35moe_safetensors();
    fs::write(input.join("model.safetensors"), st).unwrap();

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4",
            "--output", output.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();
}

#[test]
fn test_convert_qwen35moe_f16_produces_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35moe-f16-input");
    let output = tmp.path().join("qwen35moe-f16-output");
    setup_qwen35moe(&input);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "f16",
            "--output", output.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let gguf_files: Vec<_> = fs::read_dir(&output)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .collect();
    assert!(!gguf_files.is_empty(), "No GGUF produced for f16 quant on qwen35moe");
}

/// Verify the qwen35moe GGUF contains more tensors than the dense variant
/// would (due to per-expert weights being merged into stacked tensors).
///
/// This test does not make dense/MoE count claims — it asserts only that
/// the count is greater than zero and that the file is structurally valid.
#[test]
fn test_convert_qwen35moe_tensor_count_nonzero() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35moe-tc-input");
    let output = tmp.path().join("qwen35moe-tc-output");
    setup_qwen35moe(&input);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4",
            "--output", output.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let gguf_path = fs::read_dir(&output)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .expect("No GGUF file")
        .path();

    let data = fs::read(&gguf_path).unwrap();
    let hdr = GgufHeader::read(&data);
    assert!(
        hdr.tensor_count > 0,
        "qwen35moe GGUF must contain at least 1 tensor, got {}",
        hdr.tensor_count
    );
    // Also assert at least some metadata keys (arch, etc.)
    assert!(
        hdr.kv_count >= 4,
        "qwen35moe GGUF must have ≥4 metadata keys (arch, name, block_count, file_type); got {}",
        hdr.kv_count
    );
}
