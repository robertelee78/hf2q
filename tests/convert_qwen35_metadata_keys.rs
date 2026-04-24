//! ADR-012 P4 Decision 7 — GGUF metadata key emission gate.
//!
//! Converts a synthetic qwen35 (dense) and qwen35moe model, opens the
//! output GGUF via mlx-native's reader, and asserts every Decision 7-
//! mandated `{arch}.*` key is present. llama.cpp's loader accepts or
//! rejects the file based on these keys — if any is missing, inference
//! fails silently (loader-rejected-file is the silent-failure mode
//! Decision 16 / P8 was designed to catch, but we shouldn't need a
//! 100 GB real-model download to prove this key by key).
//!
//! This file proves key presence on a 2-layer synthetic model where
//! the convert is cheap and the failure mode is obvious.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use mlx_native::gguf::GgufFile;

// Synthetic qwen35 configs — note the use of nested `rope_parameters`
// object (not `rope_scaling`); the hf2q config parser keys off
// `rope_parameters` per Qwen3.6 HF config convention
// (src/input/config_parser.rs:328).
const QWEN35_DENSE_CONFIG: &str = r#"{
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "intermediate_size": 128,
    "vocab_size": 128,
    "head_dim": 16,
    "linear_num_value_heads": 8,
    "full_attention_interval": 2,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000.0,
    "rope_parameters": {
        "mrope_section": [3, 3, 2],
        "rope_theta": 10000000.0,
        "rope_type": "mrope",
        "mrope_interleaved": true,
        "partial_rotary_factor": 0.25
    },
    "mtp_num_hidden_layers": 0,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16",
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_value_head_dim": 128,
    "linear_num_value_heads": 8
}"#;

const QWEN35_MOE_CONFIG: &str = r#"{
    "architectures": ["Qwen3_5MoeForCausalLM"],
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "intermediate_size": 128,
    "vocab_size": 128,
    "head_dim": 16,
    "linear_num_value_heads": 8,
    "full_attention_interval": 4,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000.0,
    "rope_parameters": {
        "mrope_section": [3, 3, 2],
        "rope_theta": 10000000.0,
        "rope_type": "mrope",
        "mrope_interleaved": true,
        "partial_rotary_factor": 0.25
    },
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "mtp_num_hidden_layers": 0,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16",
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_value_head_dim": 128,
    "linear_num_value_heads": 8
}"#;

fn push_f16(
    tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>,
    name: &str,
    shape: Vec<usize>,
) {
    let n = shape.iter().product::<usize>() * 2;
    tensors.push((name.to_string(), shape, "F16", vec![0u8; n]));
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
    let hdr = serde_json::to_string(&header_map).unwrap();
    let hbytes = hdr.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    out.extend_from_slice(hbytes);
    out.extend_from_slice(&payload);
    out
}

fn setup_dense(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), QWEN35_DENSE_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    let h = 64usize;
    let inter = 128usize;
    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    push_f16(&mut tensors, "model.embed_tokens.weight", vec![128, h]);
    push_f16(&mut tensors, "lm_head.weight", vec![128, h]);
    push_f16(&mut tensors, "model.norm.weight", vec![h]);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        push_f16(&mut tensors, &format!("{p}.input_layernorm.weight"), vec![h]);
        push_f16(&mut tensors, &format!("{p}.post_attention_layernorm.weight"), vec![h]);
        if layer == 1 {
            push_f16(&mut tensors, &format!("{p}.self_attn.q_proj.weight"), vec![h, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.k_proj.weight"), vec![16, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.v_proj.weight"), vec![16, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.o_proj.weight"), vec![h, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.gate.weight"), vec![1, h]);
        } else {
            let qkv = (4 + 1 + 8) * 16;
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"), vec![qkv, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.out_proj.weight"), vec![h, 128]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), vec![4, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), vec![4, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"), vec![128, h]);
        }
        push_f16(&mut tensors, &format!("{p}.mlp.gate_proj.weight"), vec![inter, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.up_proj.weight"), vec![inter, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.down_proj.weight"), vec![h, inter]);
    }
    fs::write(
        dir.join("model.safetensors"),
        build_safetensors_bytes(tensors),
    )
    .unwrap();
}

fn setup_moe(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), QWEN35_MOE_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    let h = 64usize;
    let moe_inter = 64usize;
    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    push_f16(&mut tensors, "model.embed_tokens.weight", vec![128, h]);
    push_f16(&mut tensors, "lm_head.weight", vec![128, h]);
    push_f16(&mut tensors, "model.norm.weight", vec![h]);
    for layer in 0..4 {
        let p = format!("model.layers.{layer}");
        push_f16(&mut tensors, &format!("{p}.input_layernorm.weight"), vec![h]);
        push_f16(&mut tensors, &format!("{p}.post_attention_layernorm.weight"), vec![h]);
        if layer == 3 {
            push_f16(&mut tensors, &format!("{p}.self_attn.q_proj.weight"), vec![h, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.k_proj.weight"), vec![16, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.v_proj.weight"), vec![16, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.o_proj.weight"), vec![h, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.gate.weight"), vec![1, h]);
        } else {
            let qkv = (4 + 1 + 8) * 16;
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"), vec![qkv, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.out_proj.weight"), vec![h, 128]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), vec![4, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), vec![4, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"), vec![128, h]);
        }
        push_f16(&mut tensors, &format!("{p}.mlp.gate.weight"), vec![4, h]);
        for e in 0..4 {
            push_f16(&mut tensors, &format!("{p}.mlp.experts.{e}.gate_proj.weight"), vec![moe_inter, h]);
            push_f16(&mut tensors, &format!("{p}.mlp.experts.{e}.up_proj.weight"), vec![moe_inter, h]);
            push_f16(&mut tensors, &format!("{p}.mlp.experts.{e}.down_proj.weight"), vec![h, moe_inter]);
        }
        push_f16(&mut tensors, &format!("{p}.mlp.shared_expert_gate.weight"), vec![1, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.shared_expert.gate_proj.weight"), vec![moe_inter, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.shared_expert.up_proj.weight"), vec![moe_inter, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.shared_expert.down_proj.weight"), vec![h, moe_inter]);
    }
    fs::write(
        dir.join("model.safetensors"),
        build_safetensors_bytes(tensors),
    )
    .unwrap();
}

fn convert(input: &Path, output: &Path) {
    let assert = assert_cmd::Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "f16",
            "--output", output.to_str().unwrap(),
            "--yes",
            "--skip-quality",
        ])
        .assert();
    let out = assert.get_output().clone();
    if !out.status.success() {
        panic!(
            "hf2q convert failed: stdout={} stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// Decision 7 required keys for both qwen35 dense and qwen35moe.
/// Names hand-transcribed from src/models/qwen35/{dense,moe}.rs
/// doc comments with llama-arch.cpp line citations.
const DECISION_7_SHARED_KEYS: &[&str] = &[
    "{arch}.block_count",
    "{arch}.context_length",
    "{arch}.embedding_length",
    "{arch}.feed_forward_length",
    "{arch}.attention.head_count",
    "{arch}.attention.head_count_kv",
    "{arch}.attention.key_length",
    "{arch}.attention.value_length",
    "{arch}.attention.layer_norm_rms_epsilon",
    "{arch}.rope.freq_base",
    "{arch}.rope.dimension_count",
    "{arch}.rope.dimension_sections",
    "{arch}.full_attention_interval",
    "{arch}.ssm.conv_kernel",
    "{arch}.ssm.inner_size",
    "{arch}.ssm.state_size",
    "{arch}.ssm.time_step_rank",
    "{arch}.ssm.group_count",
];

fn assert_required_keys_present(
    gguf: &GgufFile,
    arch: &str,
    keys: &[&str],
) {
    for key_template in keys {
        let key = key_template.replace("{arch}", arch);
        assert!(
            gguf.metadata(&key).is_some(),
            "Decision 7 required key {:?} missing from converted {} GGUF",
            key,
            arch
        );
    }
}

#[test]
fn qwen35_dense_emits_all_decision_7_required_keys() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("dense-in");
    let output = tmp.path().join("dense.gguf");
    setup_dense(&input);
    convert(&input, &output);

    let gguf = GgufFile::open(&output).expect("open dense GGUF");
    assert_required_keys_present(&gguf, "qwen35", DECISION_7_SHARED_KEYS);

    // arch string is qwen35 (not qwen3_5_moe_text etc.)
    let arch = gguf
        .metadata_string("general.architecture")
        .expect("general.architecture");
    assert_eq!(arch, "qwen35", "Decision 1 arch string");
}

#[test]
fn qwen35moe_emits_all_decision_7_shared_keys_and_moe_specific() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("moe-in");
    let output = tmp.path().join("moe.gguf");
    setup_moe(&input);
    convert(&input, &output);

    let gguf = GgufFile::open(&output).expect("open MoE GGUF");
    assert_required_keys_present(&gguf, "qwen35moe", DECISION_7_SHARED_KEYS);

    // MoE-specific keys. Hand-transcribed from src/models/qwen35/moe.rs:549+
    // and llama-arch.cpp for the qwen35moe arch.
    let moe_keys = &[
        "qwen35moe.expert_count",
        "qwen35moe.expert_used_count",
    ];
    for k in moe_keys {
        assert!(
            gguf.metadata(k).is_some(),
            "qwen35moe MUST emit {:?} (llama.cpp loader rejects without it)",
            k
        );
    }

    let arch = gguf.metadata_string("general.architecture").expect("arch");
    assert_eq!(arch, "qwen35moe");
}

/// Legacy `rope_scaling` form proof. Some HF configs ship rope as
/// `rope_scaling: {mrope_section, type}` with `rope_theta` on the
/// parent config (older Qwen / Llama convention), not `rope_parameters`.
/// The parser's 2026-04-24 schema-flexibility fix accepts both forms;
/// this test asserts the legacy form still yields all Decision 7
/// rope keys in the emitted GGUF — proving the fix at the end-to-end
/// convert layer, not just in unit tests.
#[test]
fn qwen35_dense_with_legacy_rope_scaling_still_emits_all_rope_keys() {
    const LEGACY_CONFIG: &str = r#"{
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "intermediate_size": 128,
        "vocab_size": 128,
        "head_dim": 16,
        "linear_num_value_heads": 8,
        "full_attention_interval": 2,
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
        "dtype": "float16",
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_value_head_dim": 128,
        "linear_num_value_heads": 8
    }"#;

    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("legacy-in");
    let output = tmp.path().join("legacy.gguf");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("config.json"), LEGACY_CONFIG).unwrap();
    fs::write(input.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(
        input.join("tokenizer_config.json"),
        r#"{"model_max_length":131072}"#,
    )
    .unwrap();

    // Reuse the dense-model safetensors builder by setting up a dense
    // dir next to it.
    let h = 64usize;
    let inter = 128usize;
    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    push_f16(&mut tensors, "model.embed_tokens.weight", vec![128, h]);
    push_f16(&mut tensors, "lm_head.weight", vec![128, h]);
    push_f16(&mut tensors, "model.norm.weight", vec![h]);
    for layer in 0..2 {
        let p = format!("model.layers.{layer}");
        push_f16(&mut tensors, &format!("{p}.input_layernorm.weight"), vec![h]);
        push_f16(
            &mut tensors,
            &format!("{p}.post_attention_layernorm.weight"),
            vec![h],
        );
        if layer == 1 {
            push_f16(&mut tensors, &format!("{p}.self_attn.q_proj.weight"), vec![h, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.k_proj.weight"), vec![16, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.v_proj.weight"), vec![16, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.o_proj.weight"), vec![h, h]);
            push_f16(&mut tensors, &format!("{p}.self_attn.gate.weight"), vec![1, h]);
        } else {
            let qkv = (4 + 1 + 8) * 16;
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"), vec![qkv, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.out_proj.weight"), vec![h, 128]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), vec![4, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), vec![4, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"), vec![128, h]);
        }
        push_f16(&mut tensors, &format!("{p}.mlp.gate_proj.weight"), vec![inter, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.up_proj.weight"), vec![inter, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.down_proj.weight"), vec![h, inter]);
    }
    fs::write(
        input.join("model.safetensors"),
        build_safetensors_bytes(tensors),
    )
    .unwrap();

    convert(&input, &output);
    let gguf = GgufFile::open(&output).expect("open legacy-form GGUF");
    // These are the keys that would have been missing before the
    // parser fix landed — proves the fix at the GGUF layer.
    for key in &[
        "qwen35.rope.freq_base",
        "qwen35.rope.dimension_count",
        "qwen35.rope.dimension_sections",
    ] {
        assert!(
            gguf.metadata(key).is_some(),
            "legacy rope_scaling config MUST still emit {:?} per parser schema fix",
            key
        );
    }
    // Positive value check — rope_theta from parent config should flow
    // through, not default to zero.
    use mlx_native::gguf::MetadataValue;
    let theta = gguf.metadata("qwen35.rope.freq_base").expect("present");
    match theta {
        MetadataValue::Float32(v) => {
            assert!(
                (*v - 10_000_000.0).abs() < 1.0,
                "rope_theta from parent config must flow through; got {}",
                v
            );
        }
        other => panic!("expected Float32, got {:?}", other),
    }
}

/// Decision 7 explicit call-out: `{arch}.rope.dimension_sections` is
/// MANDATORY — llama-model.cpp:2808 reads it and rejects the file
/// without it. Having a separate named test anchors this so a future
/// refactor that accidentally drops the emission (or conditionalizes
/// on an hparam being present) fires a clearly-named regression.
#[test]
fn qwen35_rope_dimension_sections_is_mandatory_and_has_4_entries() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("dense-rope-in");
    let output = tmp.path().join("dense-rope.gguf");
    setup_dense(&input);
    convert(&input, &output);

    let gguf = GgufFile::open(&output).expect("open");
    let v = gguf
        .metadata("qwen35.rope.dimension_sections")
        .expect("qwen35.rope.dimension_sections MUST be emitted (Decision 7 mandatory)");
    // convert_hf_to_gguf.py:1149-1154 emits mrope_section padded to 4.
    // The synthetic config has mrope_section=[3,3,2] — emitter pads to
    // [3,3,2,0].
    use mlx_native::gguf::MetadataValue;
    match v {
        MetadataValue::Array(arr) => {
            assert_eq!(
                arr.len(),
                4,
                "rope.dimension_sections must be padded to exactly 4 entries per \
                 convert_hf_to_gguf.py:1149-1154; got {} entries",
                arr.len()
            );
        }
        other => panic!("expected Array for rope.dimension_sections, got {:?}", other),
    }
}
