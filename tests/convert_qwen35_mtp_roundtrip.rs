//! ADR-012 P11 — MTP tensor round-trip integrity gate (Decision 19).
//!
//! Both Qwen3.5 dense and Qwen3.5-MoE may carry a single MTP block
//! (`mtp_num_hidden_layers: 1` for the apex models).  Decision 11 (shipped
//! P4) emits the tensors at `blk.{num_hidden_layers + idx}.nextn.*`; Decision
//! 19 (this file) **proves** they are loadable by ADR-013's weight loader at
//! `src/inference/models/qwen35/mtp.rs::load_mtp_weights_if_present`.
//!
//! # What the test does
//!
//! 1. Build a synthetic tiny qwen35/qwen35moe HF model directory with
//!    `mtp_num_hidden_layers: 1` and full HF MTP tensors at
//!    `mtp.layers.0.{enorm,hnorm,embed_tokens,eh_proj}.weight`.
//! 2. Drive `hf2q convert --quant q4` end-to-end.
//! 3. Open the produced GGUF and assert every expected tensor in
//!    `EXPECTED_MTP_TENSORS` is present at the resolved block index
//!    `blk.{num_hidden_layers}.nextn.*`.
//! 4. The same scan logic the inference loader uses
//!    (`prefix = "blk.{num_hidden_layers}.nextn."`) is applied here — if the
//!    test passes the loader will populate.
//!
//! # Why this exists
//!
//! Without this gate, a one-letter typo in the MTP rename (e.g.
//! `nextn.eh_proj` → `nextn.eh_pjroj`) would silently strip MTP from every
//! Qwen3.5 GGUF we ship.  No unit-level test catches that — only the
//! end-to-end convert+inspect roundtrip does.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use assert_cmd::Command;
use mlx_native::gguf::GgufFile;

// ─── Expected MTP tensors (hand-transcribed) ─────────────────────────────────
//
// Source: /opt/llama.cpp/src/llama-arch.cpp:447-450, LLM_TENSOR_NEXTN_*.
//
//   LLM_TENSOR_NEXTN_EH_PROJ       → "blk.%d.nextn.eh_proj"
//   LLM_TENSOR_NEXTN_EMBED_TOKENS  → "blk.%d.nextn.embed_tokens"
//   LLM_TENSOR_NEXTN_ENORM         → "blk.%d.nextn.enorm"
//   LLM_TENSOR_NEXTN_HNORM         → "blk.%d.nextn.hnorm"
//
// The four tensors above correspond 1:1 with the HF source tensors
// `mtp.layers.0.{eh_proj,embed_tokens,enorm,hnorm}.weight`.  llama-arch.cpp
// has two additional NEXTN tensors (`shared_head_head`, `shared_head_norm`)
// but Qwen3.5 does not emit those at the HF source level — they're
// DeepSeek-V3-only fields whose Qwen3.5 analog is the main `output.weight`
// + `output_norm.weight`.
// Each entry is the suffix after the `blk.{num_hidden_layers}.nextn.` prefix —
// matches the loader's scan key in
// `src/inference/models/qwen35/mtp.rs::MtpWeights::has_tensor_suffix`.
const EXPECTED_MTP_SUFFIXES: &[&str] = &[
    "enorm.weight",
    "hnorm.weight",
    "embed_tokens.weight",
    "eh_proj.weight",
];

// ─── Synthetic model configs ─────────────────────────────────────────────────

const QWEN35_DENSE_CONFIG_MTP: &str = r#"{
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "intermediate_size": 64,
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
    "mtp_num_hidden_layers": 1,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16"
}"#;

const QWEN35_MOE_CONFIG_MTP: &str = r#"{
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
    "full_attention_interval": 4,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000.0,
    "rope_scaling": {
        "mrope_section": [3, 3, 2],
        "type": "mrope"
    },
    "mtp_num_hidden_layers": 1,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16"
}"#;

// ─── Synthetic model builders ────────────────────────────────────────────────

const NUM_LAYERS: usize = 4;
const HIDDEN: usize = 64;
const VOCAB: usize = 128;
const NUM_HEADS: usize = 4;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 16;
const LIN_V_HEADS: usize = 8;
const F16: usize = 2;

fn add_global_tensors(tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>) {
    tensors.push((
        "model.embed_tokens.weight".into(),
        vec![VOCAB, HIDDEN],
        "F16",
        vec![0u8; VOCAB * HIDDEN * F16],
    ));
    tensors.push((
        "lm_head.weight".into(),
        vec![VOCAB, HIDDEN],
        "F16",
        vec![1u8; VOCAB * HIDDEN * F16],
    ));
    tensors.push((
        "model.norm.weight".into(),
        vec![HIDDEN],
        "F16",
        vec![0u8; HIDDEN * F16],
    ));
}

fn add_full_attn_block(
    tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>,
    layer: usize,
) {
    let prefix = format!("model.layers.{layer}");
    let q_size = NUM_HEADS * HEAD_DIM;
    let kv_size = KV_HEADS * HEAD_DIM;
    tensors.push((format!("{prefix}.input_layernorm.weight"), vec![HIDDEN], "F16", vec![0u8; HIDDEN * F16]));
    tensors.push((format!("{prefix}.post_attention_layernorm.weight"), vec![HIDDEN], "F16", vec![0u8; HIDDEN * F16]));
    tensors.push((format!("{prefix}.self_attn.q_proj.weight"), vec![q_size, HIDDEN], "F16", vec![2u8; q_size * HIDDEN * F16]));
    tensors.push((format!("{prefix}.self_attn.k_proj.weight"), vec![kv_size, HIDDEN], "F16", vec![3u8; kv_size * HIDDEN * F16]));
    tensors.push((format!("{prefix}.self_attn.v_proj.weight"), vec![kv_size, HIDDEN], "F16", vec![4u8; kv_size * HIDDEN * F16]));
    tensors.push((format!("{prefix}.self_attn.o_proj.weight"), vec![HIDDEN, q_size], "F16", vec![5u8; HIDDEN * q_size * F16]));
    tensors.push((format!("{prefix}.self_attn.gate.weight"), vec![1, HIDDEN], "F16", vec![0u8; HIDDEN * F16]));
}

fn add_linear_attn_block(
    tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>,
    layer: usize,
) {
    let prefix = format!("model.layers.{layer}");
    let qkv_size = (NUM_HEADS + KV_HEADS + LIN_V_HEADS) * HEAD_DIM;
    tensors.push((format!("{prefix}.input_layernorm.weight"), vec![HIDDEN], "F16", vec![0u8; HIDDEN * F16]));
    tensors.push((format!("{prefix}.post_attention_layernorm.weight"), vec![HIDDEN], "F16", vec![0u8; HIDDEN * F16]));
    tensors.push((format!("{prefix}.linear_attn.in_proj_qkv.weight"), vec![qkv_size, HIDDEN], "F16", vec![6u8; qkv_size * HIDDEN * F16]));
    tensors.push((format!("{prefix}.linear_attn.out_proj.weight"), vec![HIDDEN, LIN_V_HEADS * HEAD_DIM], "F16", vec![7u8; HIDDEN * LIN_V_HEADS * HEAD_DIM * F16]));
    tensors.push((format!("{prefix}.linear_attn.in_proj_a.weight"), vec![NUM_HEADS, HIDDEN], "F16", vec![0u8; NUM_HEADS * HIDDEN * F16]));
    tensors.push((format!("{prefix}.linear_attn.in_proj_b.weight"), vec![NUM_HEADS, HIDDEN], "F16", vec![0u8; NUM_HEADS * HIDDEN * F16]));
    tensors.push((format!("{prefix}.linear_attn.in_proj_z.weight"), vec![LIN_V_HEADS * HEAD_DIM, HIDDEN], "F16", vec![0u8; LIN_V_HEADS * HEAD_DIM * HIDDEN * F16]));
    tensors.push((format!("{prefix}.linear_attn.A_log"), vec![NUM_HEADS], "F32", vec![0u8; NUM_HEADS * 4]));
    tensors.push((format!("{prefix}.linear_attn.dt_proj.weight"), vec![NUM_HEADS, HIDDEN], "F16", vec![0u8; NUM_HEADS * HIDDEN * F16]));
    tensors.push((format!("{prefix}.linear_attn.dt_bias"), vec![NUM_HEADS], "F32", vec![0u8; NUM_HEADS * 4]));
    tensors.push((format!("{prefix}.linear_attn.conv1d.weight"), vec![NUM_HEADS, 1, HIDDEN / NUM_HEADS], "F16", vec![0u8; NUM_HEADS * (HIDDEN / NUM_HEADS) * F16]));
    tensors.push((format!("{prefix}.linear_attn.norm.weight"), vec![HIDDEN], "F16", vec![0u8; HIDDEN * F16]));
}

fn add_dense_ffn(tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>, layer: usize) {
    let prefix = format!("model.layers.{layer}.mlp");
    let inter = 64usize;
    tensors.push((format!("{prefix}.gate_proj.weight"), vec![inter, HIDDEN], "F16", vec![1u8; inter * HIDDEN * F16]));
    tensors.push((format!("{prefix}.up_proj.weight"), vec![inter, HIDDEN], "F16", vec![1u8; inter * HIDDEN * F16]));
    tensors.push((format!("{prefix}.down_proj.weight"), vec![HIDDEN, inter], "F16", vec![1u8; HIDDEN * inter * F16]));
}

fn add_moe_ffn(tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>, layer: usize) {
    let prefix = format!("model.layers.{layer}.mlp");
    let moe_inter = 16usize;
    let shared_inter = 16usize;
    let num_experts = 4usize;

    tensors.push((format!("{prefix}.gate.weight"), vec![num_experts, HIDDEN], "F16", vec![0u8; num_experts * HIDDEN * F16]));
    for e in 0..num_experts {
        tensors.push((format!("{prefix}.experts.{e}.gate_proj.weight"), vec![moe_inter, HIDDEN], "F16", vec![0u8; moe_inter * HIDDEN * F16]));
        tensors.push((format!("{prefix}.experts.{e}.up_proj.weight"), vec![moe_inter, HIDDEN], "F16", vec![0u8; moe_inter * HIDDEN * F16]));
        tensors.push((format!("{prefix}.experts.{e}.down_proj.weight"), vec![HIDDEN, moe_inter], "F16", vec![0u8; HIDDEN * moe_inter * F16]));
    }
    tensors.push((format!("{prefix}.shared_expert.gate_proj.weight"), vec![shared_inter, HIDDEN], "F16", vec![8u8; shared_inter * HIDDEN * F16]));
    tensors.push((format!("{prefix}.shared_expert.up_proj.weight"), vec![shared_inter, HIDDEN], "F16", vec![9u8; shared_inter * HIDDEN * F16]));
    tensors.push((format!("{prefix}.shared_expert.down_proj.weight"), vec![HIDDEN, shared_inter], "F16", vec![10u8; HIDDEN * shared_inter * F16]));
}

fn add_mtp_block(tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>) {
    // MTP idx 0 → resolves to blk.{num_hidden_layers}.nextn.* in GGUF.
    // The four tensors are the Qwen3.5-side analog of llama-arch.cpp's
    // LLM_TENSOR_NEXTN_{ENORM,HNORM,EMBED_TOKENS,EH_PROJ}.
    let prefix = "mtp.layers.0";
    tensors.push((format!("{prefix}.enorm.weight"), vec![HIDDEN], "F16", vec![20u8; HIDDEN * F16]));
    tensors.push((format!("{prefix}.hnorm.weight"), vec![HIDDEN], "F16", vec![21u8; HIDDEN * F16]));
    tensors.push((format!("{prefix}.embed_tokens.weight"), vec![VOCAB, HIDDEN], "F16", vec![22u8; VOCAB * HIDDEN * F16]));
    tensors.push((format!("{prefix}.eh_proj.weight"), vec![HIDDEN, 2 * HIDDEN], "F16", vec![23u8; HIDDEN * 2 * HIDDEN * F16]));
}

fn build_dense_safetensors() -> Vec<u8> {
    let mut tensors: Vec<(String, Vec<usize>, &'static str, Vec<u8>)> = Vec::new();
    add_global_tensors(&mut tensors);
    for layer in 0..NUM_LAYERS {
        let is_full = layer == NUM_LAYERS - 1; // last layer is full-attn (interval=4)
        if is_full {
            add_full_attn_block(&mut tensors, layer);
        } else {
            add_linear_attn_block(&mut tensors, layer);
        }
        add_dense_ffn(&mut tensors, layer);
    }
    add_mtp_block(&mut tensors);
    build_safetensors_bytes(tensors)
}

fn build_moe_safetensors() -> Vec<u8> {
    let mut tensors: Vec<(String, Vec<usize>, &'static str, Vec<u8>)> = Vec::new();
    add_global_tensors(&mut tensors);
    for layer in 0..NUM_LAYERS {
        let is_full = layer == NUM_LAYERS - 1;
        if is_full {
            add_full_attn_block(&mut tensors, layer);
        } else {
            add_linear_attn_block(&mut tensors, layer);
        }
        add_moe_ffn(&mut tensors, layer);
    }
    add_mtp_block(&mut tensors);
    build_safetensors_bytes(tensors)
}

fn build_safetensors_bytes(tensors: Vec<(String, Vec<usize>, &'static str, Vec<u8>)>) -> Vec<u8> {
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

fn write_sidecars(dir: &Path) {
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    fs::write(dir.join("generation_config.json"), r#"{"do_sample":false}"#).unwrap();
    fs::write(dir.join("special_tokens_map.json"), r#"{"bos_token":"<|im_start|>"}"#).unwrap();
    fs::write(
        dir.join("chat_template.jinja"),
        "{% for msg in messages %}{{ msg.content }}{% endfor %}",
    )
    .unwrap();
}

fn convert_to_gguf(input: &Path, output: &Path) {
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

fn locate_gguf(output_dir: &Path) -> std::path::PathBuf {
    fs::read_dir(output_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .expect("no GGUF emitted")
        .path()
}

/// Mirror of `src/inference/models/qwen35/mtp.rs::load_mtp_weights_if_present`'s
/// scan logic — if every expected MTP suffix appears under the
/// `blk.{num_hidden_layers}.nextn.` prefix, the loader will populate.
fn assert_mtp_present(gguf_path: &Path, num_hidden_layers: u32) {
    let gguf = GgufFile::open(gguf_path).expect("open gguf");
    let prefix = format!("blk.{}.nextn.", num_hidden_layers);
    let names: Vec<String> = gguf
        .tensor_names()
        .into_iter()
        .map(|s: &str| s.to_string())
        .collect();

    let found: Vec<&str> = names.iter().map(|s| s.as_str()).filter(|n| n.starts_with(&prefix)).collect();
    assert!(
        !found.is_empty(),
        "no MTP tensors found at prefix '{prefix}'; \
         GGUF carried {} tensors, none under that prefix — \
         this is the silent-stub failure mode the gate exists to catch",
        names.len()
    );

    for expected_suffix in EXPECTED_MTP_SUFFIXES {
        let expected_name = format!("{prefix}{expected_suffix}");
        assert!(
            names.iter().any(|n| n == &expected_name),
            "missing MTP tensor '{expected_name}' \
             (Decision 11 + 19; llama-arch.cpp:447-450 LLM_TENSOR_NEXTN_*); \
             tensors at prefix were: {found:?}",
        );
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn qwen35_dense_mtp_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-mtp-dense-input");
    let output = tmp.path().join("qwen35-mtp-dense-output");
    fs::create_dir_all(&input).unwrap();

    fs::write(input.join("config.json"), QWEN35_DENSE_CONFIG_MTP).unwrap();
    write_sidecars(&input);
    fs::write(input.join("model.safetensors"), build_dense_safetensors()).unwrap();

    convert_to_gguf(&input, &output);

    // Decision 11: MTP block lands at blk.{num_hidden_layers} = blk.4 for the
    // synthetic 4-layer model.  Decision 19's resolver makes this real.
    assert_mtp_present(&locate_gguf(&output), NUM_LAYERS as u32);
}

#[test]
fn qwen35moe_mtp_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-mtp-moe-input");
    let output = tmp.path().join("qwen35-mtp-moe-output");
    fs::create_dir_all(&input).unwrap();

    fs::write(input.join("config.json"), QWEN35_MOE_CONFIG_MTP).unwrap();
    write_sidecars(&input);
    fs::write(input.join("model.safetensors"), build_moe_safetensors()).unwrap();

    convert_to_gguf(&input, &output);

    assert_mtp_present(&locate_gguf(&output), NUM_LAYERS as u32);
}

#[test]
fn qwen35_dense_mtp_emitted_at_resolved_block_not_placeholder() {
    // The placeholder `blk.mtp0.nextn.*` MUST NOT appear in the output —
    // that would mean the resolver in src/backends/gguf.rs:resolve_mtp_block_index
    // failed to fire, and the inference loader would silently skip the tensors.
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-mtp-noplace-input");
    let output = tmp.path().join("qwen35-mtp-noplace-output");
    fs::create_dir_all(&input).unwrap();
    fs::write(input.join("config.json"), QWEN35_DENSE_CONFIG_MTP).unwrap();
    write_sidecars(&input);
    fs::write(input.join("model.safetensors"), build_dense_safetensors()).unwrap();

    convert_to_gguf(&input, &output);

    let gguf = GgufFile::open(&locate_gguf(&output)).expect("open gguf");
    let placeholders: Vec<String> = gguf
        .tensor_names()
        .into_iter()
        .filter(|n: &&str| n.starts_with("blk.mtp"))
        .map(|n: &str| n.to_string())
        .collect();
    assert!(
        placeholders.is_empty(),
        "found unresolved MTP placeholders {placeholders:?} — \
         resolve_mtp_block_index never fired",
    );
}
