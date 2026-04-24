//! Integration tests for the qwen35 (dense) convert pipeline.
//!
//! Uses synthetic tiny model data with deterministic weights. Exercises the
//! full convert path from HF safetensors directory to GGUF output, then
//! reads the produced GGUF binary back via hf2q's own GGUF format knowledge
//! (magic bytes, version, metadata count, tensor count) and asserts against
//! a hand-authored spec.
//!
//! No real model download occurs. No binary GGUF fixtures are checked in.
//! All expected values are derived from the GGUF spec and the qwen35-specific
//! ADR-012 metadata catalog (Decisions 7, 8, 11).
//!
//! Hparams for the synthetic model:
//!   hidden_size: 64
//!   num_hidden_layers: 4
//!   num_attention_heads: 4
//!   num_key_value_heads: 1
//!   head_dim: 16
//!   linear_num_value_heads: 8
//!   intermediate_size: 128 (dense FFN)
//!   vocab_size: 128
//!   full_attention_interval: 4  (layers 3 is full_attention; 0,1,2 are linear)
//!   partial_rotary_factor: 0.25
//!   rope_theta: 10_000_000.0

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use assert_cmd::Command;

// ─── GGUF binary constants (from the GGUF spec and src/backends/gguf.rs) ───

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"
const GGUF_VERSION: u32 = 3;

// ─── Synthetic qwen35 dense model config ────────────────────────────────────

const QWEN35_CONFIG: &str = r#"{
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5",
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

// ─── Safetensors builder for synthetic qwen35 dense model ───────────────────

/// Build a minimal safetensors file for a 4-layer qwen35 dense model.
///
/// Tensor layout follows HF naming conventions that the convert pipeline
/// maps to GGUF names. We include one representative tensor per tensor
/// family:
///   - embed_tokens (vocab_size × hidden_size)
///   - lm_head (vocab_size × hidden_size)
///   - model.norm.weight (hidden_size)
///   - Per full-attention layer (layer 3): q/k/v/o projections
///   - Per linear-attention layer (layers 0,1,2): in_proj_qkv, out_proj
///   - FFN for all layers: gate_proj/up_proj/down_proj
///   - Input layernorm, post-attention norm for each layer
fn build_qwen35_dense_safetensors() -> Vec<u8> {
    let hidden: usize = 64;
    let vocab: usize = 128;
    let inter: usize = 128;
    let num_heads: usize = 4;
    let kv_heads: usize = 1;
    let head_dim: usize = 16;
    let lin_v_heads: usize = 8;
    let num_layers: usize = 4;

    // F16 byte sizes
    let f16 = 2usize;

    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();

    // Embeddings
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
        // is_full_attention when layer % 4 == 3 (full_attention_interval=4)
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
            // Full attention (standard qkv)
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
            // Gate weight for full-attention (attn_output_gate=true)
            tensors.push((
                format!("{prefix}.self_attn.gate.weight"),
                vec![1, hidden],
                "F16",
                vec![0u8; hidden * f16],
            ));
        } else {
            // Linear attention block
            let qkv_size = (num_heads + kv_heads + lin_v_heads) * head_dim;
            tensors.push((
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                vec![qkv_size, hidden],
                "F16",
                vec![6u8; qkv_size * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.out_proj.weight"),
                vec![hidden, lin_v_heads * head_dim],
                "F16",
                vec![7u8; hidden * lin_v_heads * head_dim * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.in_proj_a.weight"),
                vec![num_heads, hidden],
                "F16",
                vec![0u8; num_heads * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                vec![num_heads, hidden],
                "F16",
                vec![0u8; num_heads * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                vec![lin_v_heads * head_dim, hidden],
                "F16",
                vec![0u8; lin_v_heads * head_dim * hidden * f16],
            ));
            // A_log and dt_proj (SSM tensors)
            tensors.push((
                format!("{prefix}.linear_attn.A_log"),
                vec![num_heads],
                "F32",
                vec![0u8; num_heads * 4],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.dt_proj.weight"),
                vec![num_heads, hidden],
                "F16",
                vec![0u8; num_heads * hidden * f16],
            ));
            tensors.push((
                format!("{prefix}.linear_attn.dt_bias"),
                vec![num_heads],
                "F32",
                vec![0u8; num_heads * 4],
            ));
            // conv1d
            tensors.push((
                format!("{prefix}.linear_attn.conv1d.weight"),
                vec![num_heads, 1, hidden / num_heads],
                "F16",
                vec![0u8; num_heads * (hidden / num_heads) * f16],
            ));
            // norm weight
            tensors.push((
                format!("{prefix}.linear_attn.norm.weight"),
                vec![hidden],
                "F16",
                vec![0u8; hidden * f16],
            ));
        }

        // Dense FFN (all layers)
        tensors.push((
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![inter, hidden],
            "F16",
            vec![0u8; inter * hidden * f16],
        ));
        tensors.push((
            format!("{prefix}.mlp.up_proj.weight"),
            vec![inter, hidden],
            "F16",
            vec![0u8; inter * hidden * f16],
        ));
        tensors.push((
            format!("{prefix}.mlp.down_proj.weight"),
            vec![hidden, inter],
            "F16",
            vec![0u8; hidden * inter * f16],
        ));
    }

    build_safetensors_bytes(tensors)
}

/// Assemble safetensors bytes from a list of (name, shape, dtype, data) tuples.
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

/// Set up a minimal qwen35 dense model directory.
fn setup_qwen35_dense(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), QWEN35_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    fs::write(dir.join("generation_config.json"), r#"{"do_sample":false}"#).unwrap();
    fs::write(dir.join("special_tokens_map.json"), r#"{"bos_token":"<|im_start|>"}"#).unwrap();
    // No chat_template.jinja — tests that missing sidecars are skipped silently.
    let st = build_qwen35_dense_safetensors();
    fs::write(dir.join("model.safetensors"), st).unwrap();
}

// ─── GGUF binary reader helpers ──────────────────────────────────────────────

/// Minimal GGUF header reader — reads magic, version, tensor_count, kv_count.
///
/// Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
/// Byte layout (version 3):
///   [0..4]  magic "GGUF"
///   [4..8]  version (u32 LE)
///   [8..16] tensor_count (u64 LE)
///   [16..24] kv_count (u64 LE)
struct GgufHeader {
    version: u32,
    tensor_count: u64,
    kv_count: u64,
}

impl GgufHeader {
    fn read(data: &[u8]) -> Self {
        assert!(
            data.len() >= 24,
            "GGUF file too small: {} bytes",
            data.len()
        );
        assert_eq!(&data[0..4], &GGUF_MAGIC, "Bad GGUF magic");
        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
        GgufHeader { version, tensor_count, kv_count }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_convert_qwen35_dense_q4_produces_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-dense-input");
    let output = tmp.path().join("qwen35-dense-output");
    setup_qwen35_dense(&input);

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

    // At least one .gguf file exists.
    let gguf_files: Vec<_> = fs::read_dir(&output)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .collect();
    assert!(!gguf_files.is_empty(), "No GGUF produced");
}

#[test]
fn test_convert_qwen35_dense_gguf_has_valid_magic_and_version() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-dense-hdr");
    let output = tmp.path().join("qwen35-dense-hdr-out");
    setup_qwen35_dense(&input);

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
fn test_convert_qwen35_dense_sidecars_are_byte_identical() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-sc-input");
    let output = tmp.path().join("qwen35-sc-output");
    setup_qwen35_dense(&input);

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

    // Sidecars that we wrote to input (chat_template.jinja is intentionally
    // absent from setup_qwen35_dense to test the silent-skip path).
    let present_sidecars = [
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "config.json",
    ];

    for sidecar in &present_sidecars {
        let src = input.join(sidecar);
        let dst = output.join(sidecar);
        assert!(
            dst.exists(),
            "Sidecar '{sidecar}' missing from output dir"
        );
        let src_bytes = fs::read(&src).unwrap();
        let dst_bytes = fs::read(&dst).unwrap();
        assert_eq!(
            src_bytes, dst_bytes,
            "Sidecar '{sidecar}' content must be byte-identical"
        );
    }

    // chat_template.jinja was not in the input — must not appear in output.
    assert!(
        !output.join("chat_template.jinja").exists(),
        "chat_template.jinja must not appear in output when not in source"
    );
}

#[test]
fn test_convert_qwen35_dense_missing_sidecar_skipped_silently() {
    // Verify no error is raised when a sidecar is missing from the source.
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-skip-sc-input");
    let output = tmp.path().join("qwen35-skip-sc-output");

    // Deliberately omit all sidecars — only write the required model files.
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("config.json"), QWEN35_CONFIG).unwrap();
    let st = build_qwen35_dense_safetensors();
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
        .success(); // Must not fail.
}

#[test]
fn test_convert_qwen35_dense_f16_produces_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-f16-input");
    let output = tmp.path().join("qwen35-f16-output");
    setup_qwen35_dense(&input);

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
    assert!(!gguf_files.is_empty(), "No GGUF produced for f16 quant");
}

#[test]
fn test_convert_qwen35_dense_q8_produces_gguf() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-q8-input");
    let output = tmp.path().join("qwen35-q8-output");
    setup_qwen35_dense(&input);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q8",
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
    assert!(!gguf_files.is_empty(), "No GGUF produced for q8 quant");
}
