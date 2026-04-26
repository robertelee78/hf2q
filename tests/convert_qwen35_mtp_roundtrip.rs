//! ADR-012 P11 (Decision 19) — MTP tensor round-trip integrity gate.
//!
//! Convert a synthetic qwen35 / qwen35moe model with `mtp_num_hidden_layers: 1`,
//! then assert the emitted GGUF carries the 4 MTP tensors at the exact names
//! ADR-013's weight loader (`src/inference/models/qwen35/mtp.rs::load_mtp_weights_if_present`)
//! and llama.cpp's loader (`llama-arch.cpp:447-450`) expect:
//!
//!   `blk.{num_hidden_layers}.nextn.{enorm,hnorm,embed_tokens,eh_proj}.weight`
//!
//! This test catches the P4 stub bug that emitted `blk.mtp0.nextn.*` with a
//! literal "mtp" block label instead of the resolved block index — invisible
//! to unit tests on the mapper alone, but fatal on the load side.
//!
//! A companion negative test asserts the STUB name form (`blk.mtp0.nextn.*`)
//! is NOT present — a one-letter regression in the mapper would trip it.
//!
//! Runs under 30 s on a laptop; no disk download. CI-green by default.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use assert_cmd::Command;
use mlx_native::gguf::GgufFile;

// ---------------------------------------------------------------------------
// EXPECTED_MTP_TENSORS — hand-authored, derived from llama-arch.cpp:447-450.
//
// The ADR says "shapes + names derived from /opt/llama.cpp/src/llama-arch.cpp:447-450".
// We assert NAME presence only; shape is per-model (hand-transcribed separately
// inside each test case from the synthetic safetensors config).
// ---------------------------------------------------------------------------

/// Expected MTP suffixes at `blk.{N}.nextn.{suffix}` where N = num_hidden_layers.
const EXPECTED_MTP_SUFFIXES: &[&str] = &[
    // llama-arch.cpp:449 LLM_TENSOR_NEXTN_ENORM → "blk.%d.nextn.enorm"
    "nextn.enorm.weight",
    // llama-arch.cpp:450 LLM_TENSOR_NEXTN_HNORM → "blk.%d.nextn.hnorm"
    "nextn.hnorm.weight",
    // llama-arch.cpp:448 LLM_TENSOR_NEXTN_EMBED_TOKENS → "blk.%d.nextn.embed_tokens"
    "nextn.embed_tokens.weight",
    // llama-arch.cpp:447 LLM_TENSOR_NEXTN_EH_PROJ → "blk.%d.nextn.eh_proj"
    "nextn.eh_proj.weight",
];

// ---------------------------------------------------------------------------
// Synthetic qwen35 dense model with MTP — 4 layers, hidden=64, MTP=1.
// ---------------------------------------------------------------------------

const QWEN35_DENSE_MTP_CONFIG: &str = r#"{
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
    "mtp_num_hidden_layers": 1,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16"
}"#;

const QWEN35MOE_MTP_CONFIG: &str = r#"{
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
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "mtp_num_hidden_layers": 1,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16"
}"#;

// ---------------------------------------------------------------------------
// Safetensors builder — appends 4 MTP tensors to a base model layout.
// ---------------------------------------------------------------------------

fn push_f16_zeros(
    tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>,
    name: &str,
    shape: Vec<usize>,
) {
    let size: usize = shape.iter().product::<usize>() * 2; // F16 = 2 bytes
    tensors.push((name.to_string(), shape, "F16", vec![0u8; size]));
}

fn push_f32_zeros(
    tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>,
    name: &str,
    shape: Vec<usize>,
) {
    let size: usize = shape.iter().product::<usize>() * 4;
    tensors.push((name.to_string(), shape, "F32", vec![0u8; size]));
}

/// Build a dense qwen35 safetensors with 4 layers + 1 MTP block.
/// Layer 3 is full-attention (full_attention_interval=4); layers 0,1,2 are
/// linear-attention. MTP adds 4 tensors at `mtp.layers.0.*`.
fn build_qwen35_dense_mtp_safetensors() -> Vec<u8> {
    let hidden: usize = 64;
    let vocab: usize = 128;
    let inter: usize = 128;
    let num_heads: usize = 4;
    let kv_heads: usize = 1;
    let head_dim: usize = 16;
    let lin_v_heads: usize = 8;
    let num_layers: usize = 4;

    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    push_f16_zeros(&mut tensors, "model.embed_tokens.weight", vec![vocab, hidden]);
    push_f16_zeros(&mut tensors, "lm_head.weight", vec![vocab, hidden]);
    push_f16_zeros(&mut tensors, "model.norm.weight", vec![hidden]);

    for layer in 0..num_layers {
        let is_full = layer == 3;
        let prefix = format!("model.layers.{layer}");
        push_f16_zeros(&mut tensors, &format!("{prefix}.input_layernorm.weight"), vec![hidden]);
        push_f16_zeros(&mut tensors, &format!("{prefix}.post_attention_layernorm.weight"), vec![hidden]);
        if is_full {
            let q_size = num_heads * head_dim;
            let kv_size = kv_heads * head_dim;
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.q_proj.weight"), vec![q_size, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.k_proj.weight"), vec![kv_size, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.v_proj.weight"), vec![kv_size, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.o_proj.weight"), vec![hidden, q_size]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.gate.weight"), vec![1, hidden]);
        } else {
            // Spec-correct shapes (nk=4, nv=lin_v_heads=8, head_k/v_dim=head_dim).
            let nk = 4usize;
            let hk = head_dim;
            let hv = head_dim;
            let qkv_rows = nk * hk * 2 + lin_v_heads * hv;
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.in_proj_qkv.weight"), vec![qkv_rows, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.out_proj.weight"), vec![hidden, lin_v_heads * hv]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.in_proj_a.weight"), vec![lin_v_heads, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.in_proj_b.weight"), vec![lin_v_heads, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.in_proj_z.weight"), vec![lin_v_heads * hv, hidden]);
            push_f32_zeros(&mut tensors, &format!("{prefix}.linear_attn.A_log"), vec![lin_v_heads]);
            push_f32_zeros(&mut tensors, &format!("{prefix}.linear_attn.dt_bias"), vec![lin_v_heads]);
            let conv_channels = 2 * nk * hk + lin_v_heads * hv;
            push_f32_zeros(&mut tensors, &format!("{prefix}.linear_attn.conv1d.weight"), vec![conv_channels, 1, 4]);
            push_f32_zeros(&mut tensors, &format!("{prefix}.linear_attn.norm.weight"), vec![lin_v_heads * hv]);
        }
        // Dense FFN
        push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.gate_proj.weight"), vec![inter, hidden]);
        push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.up_proj.weight"), vec![inter, hidden]);
        push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.down_proj.weight"), vec![hidden, inter]);
    }

    // MTP tensors (mtp_num_hidden_layers=1).
    push_f32_zeros(&mut tensors, "mtp.layers.0.enorm.weight", vec![hidden]);
    push_f32_zeros(&mut tensors, "mtp.layers.0.hnorm.weight", vec![hidden]);
    push_f16_zeros(&mut tensors, "mtp.layers.0.embed_tokens.weight", vec![vocab, hidden]);
    push_f16_zeros(&mut tensors, "mtp.layers.0.eh_proj.weight", vec![hidden, hidden * 2]);

    build_safetensors_bytes(tensors)
}

/// Build an MoE qwen35 safetensors with 4 layers + 4 experts + 1 MTP block.
/// Layer 3 is full-attention, layers 0/1/2 are linear-attention (shared spine
/// with dense); FFN is 4-expert + shared-expert instead of dense.
fn build_qwen35moe_mtp_safetensors() -> Vec<u8> {
    let hidden: usize = 64;
    let vocab: usize = 128;
    let moe_inter: usize = 64;
    let shared_inter: usize = 64;
    let num_heads: usize = 4;
    let kv_heads: usize = 1;
    let head_dim: usize = 16;
    let lin_v_heads: usize = 8;
    let num_layers: usize = 4;
    let num_experts: usize = 4;

    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    push_f16_zeros(&mut tensors, "model.embed_tokens.weight", vec![vocab, hidden]);
    push_f16_zeros(&mut tensors, "lm_head.weight", vec![vocab, hidden]);
    push_f16_zeros(&mut tensors, "model.norm.weight", vec![hidden]);

    for layer in 0..num_layers {
        let is_full = layer == 3;
        let prefix = format!("model.layers.{layer}");
        push_f16_zeros(&mut tensors, &format!("{prefix}.input_layernorm.weight"), vec![hidden]);
        push_f16_zeros(&mut tensors, &format!("{prefix}.post_attention_layernorm.weight"), vec![hidden]);
        if is_full {
            let q_size = num_heads * head_dim;
            let kv_size = kv_heads * head_dim;
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.q_proj.weight"), vec![q_size, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.k_proj.weight"), vec![kv_size, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.v_proj.weight"), vec![kv_size, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.o_proj.weight"), vec![hidden, q_size]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.self_attn.gate.weight"), vec![1, hidden]);
        } else {
            // Spec-correct shapes (nk=4, nv=lin_v_heads=8, head_k/v_dim=head_dim).
            let nk = 4usize;
            let hk = head_dim;
            let hv = head_dim;
            let qkv_rows = nk * hk * 2 + lin_v_heads * hv;
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.in_proj_qkv.weight"), vec![qkv_rows, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.out_proj.weight"), vec![hidden, lin_v_heads * hv]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.in_proj_a.weight"), vec![lin_v_heads, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.in_proj_b.weight"), vec![lin_v_heads, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.linear_attn.in_proj_z.weight"), vec![lin_v_heads * hv, hidden]);
            push_f32_zeros(&mut tensors, &format!("{prefix}.linear_attn.A_log"), vec![lin_v_heads]);
            push_f32_zeros(&mut tensors, &format!("{prefix}.linear_attn.dt_bias"), vec![lin_v_heads]);
            let conv_channels = 2 * nk * hk + lin_v_heads * hv;
            push_f32_zeros(&mut tensors, &format!("{prefix}.linear_attn.conv1d.weight"), vec![conv_channels, 1, 4]);
            push_f32_zeros(&mut tensors, &format!("{prefix}.linear_attn.norm.weight"), vec![lin_v_heads * hv]);
        }
        // MoE router.
        push_f32_zeros(&mut tensors, &format!("{prefix}.mlp.gate.weight"), vec![num_experts, hidden]);
        // Per-expert weights.
        for e in 0..num_experts {
            push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.experts.{e}.gate_proj.weight"), vec![moe_inter, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.experts.{e}.up_proj.weight"), vec![moe_inter, hidden]);
            push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.experts.{e}.down_proj.weight"), vec![hidden, moe_inter]);
        }
        // Shared expert + its gate.
        push_f32_zeros(&mut tensors, &format!("{prefix}.mlp.shared_expert_gate.weight"), vec![1, hidden]);
        push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.shared_expert.gate_proj.weight"), vec![shared_inter, hidden]);
        push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.shared_expert.up_proj.weight"), vec![shared_inter, hidden]);
        push_f16_zeros(&mut tensors, &format!("{prefix}.mlp.shared_expert.down_proj.weight"), vec![hidden, shared_inter]);
    }

    // MTP tensors.
    push_f32_zeros(&mut tensors, "mtp.layers.0.enorm.weight", vec![hidden]);
    push_f32_zeros(&mut tensors, "mtp.layers.0.hnorm.weight", vec![hidden]);
    push_f16_zeros(&mut tensors, "mtp.layers.0.embed_tokens.weight", vec![vocab, hidden]);
    push_f16_zeros(&mut tensors, "mtp.layers.0.eh_proj.weight", vec![hidden, hidden * 2]);

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

fn setup_dense_mtp(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), QWEN35_DENSE_MTP_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    fs::write(dir.join("generation_config.json"), r#"{"do_sample":false}"#).unwrap();
    fs::write(dir.join("special_tokens_map.json"), r#"{"bos_token":"<|im_start|>"}"#).unwrap();
    fs::write(dir.join("model.safetensors"), build_qwen35_dense_mtp_safetensors()).unwrap();
}

fn setup_moe_mtp(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), QWEN35MOE_MTP_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(dir.join("tokenizer_config.json"), r#"{"model_max_length":131072}"#).unwrap();
    fs::write(dir.join("generation_config.json"), r#"{"do_sample":false}"#).unwrap();
    fs::write(dir.join("special_tokens_map.json"), r#"{"bos_token":"<|im_start|>"}"#).unwrap();
    fs::write(dir.join("model.safetensors"), build_qwen35moe_mtp_safetensors()).unwrap();
}

/// Convert a synthetic model dir → GGUF and return the output path.
fn convert_to_gguf(input: &Path, output: &Path) {
    let assert = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "f16",
            "--output", output.to_str().unwrap(),
            "--yes",
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn qwen35_mtp_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("dense-mtp-in");
    let output = tmp.path().join("dense-mtp-out.gguf");
    setup_dense_mtp(&input);
    convert_to_gguf(&input, &output);

    let gguf = GgufFile::open(&output).expect("open converted GGUF");
    let names: Vec<&str> = gguf.tensor_names();

    // Positive: every EXPECTED_MTP_SUFFIX is present at blk.4.* (num_hidden_layers=4).
    let num_hidden_layers = 4u32;
    for suffix in EXPECTED_MTP_SUFFIXES {
        let expected = format!("blk.{}.{}", num_hidden_layers, suffix);
        let found = names.iter().any(|n| *n == expected);
        assert!(
            found,
            "missing MTP tensor {:?} in converted GGUF. Found names: {:?}",
            expected,
            names.iter().filter(|n| n.contains("nextn") || n.contains("mtp")).collect::<Vec<_>>()
        );
    }

    // Negative: no tensor carries the P4 stub form `blk.mtp0.nextn.*`.
    // If this assertion fires, the P4 stub has been re-introduced — read
    // docs/ADR-012-qwen35moe-conversion.md P11 before changing src/backends/gguf.rs.
    for name in &names {
        assert!(
            !name.contains("blk.mtp"),
            "P4 stub MTP placeholder {:?} should never reach GGUF — see ADR-012 P11",
            name
        );
    }

    // Shape sanity on the embed_tokens MTP tensor (spot check).
    // mlx-native's TensorInfo.shape preserves the GGUF wire order (innermost-first
    // is how ggml writes ne), so HF PyTorch shape [vocab=128, hidden=64] travels
    // through convert_hf_to_gguf transpose conventions and materializes as a
    // two-element shape containing both dims; assert just the dim set.
    let embed_name = format!("blk.{}.nextn.embed_tokens.weight", num_hidden_layers);
    let info = gguf
        .tensor_info(&embed_name)
        .unwrap_or_else(|| panic!("info for {} present", embed_name));
    assert_eq!(info.shape.len(), 2, "embed_tokens is 2-D");
    let mut dims = info.shape.clone();
    dims.sort();
    assert_eq!(dims, vec![64, 128], "MTP embed_tokens dim set = {{64, 128}}");
}

#[test]
fn qwen35moe_mtp_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("moe-mtp-in");
    let output = tmp.path().join("moe-mtp-out.gguf");
    setup_moe_mtp(&input);
    convert_to_gguf(&input, &output);

    let gguf = GgufFile::open(&output).expect("open converted GGUF");
    let names: Vec<&str> = gguf.tensor_names();

    let num_hidden_layers = 4u32;
    for suffix in EXPECTED_MTP_SUFFIXES {
        let expected = format!("blk.{}.{}", num_hidden_layers, suffix);
        assert!(
            names.iter().any(|n| *n == expected),
            "missing MTP tensor {:?} in qwen35moe GGUF. nextn-like names: {:?}",
            expected,
            names.iter().filter(|n| n.contains("nextn")).collect::<Vec<_>>()
        );
    }

    // Negative: no placeholder.
    for name in &names {
        assert!(
            !name.contains("blk.mtp"),
            "qwen35moe: P4 stub {:?} must not reach GGUF — see ADR-012 P11",
            name
        );
    }

    // Strict MoE-path assertion (upgraded from evidence-only 2026-04-24):
    // the P4/P5 merge + metadata pipeline MUST produce these canonical
    // GGUF names per llama-arch.cpp:393-394 and src/models/qwen35/moe.rs.
    // If they're missing, the expert-merge was never invoked OR the
    // hf_name_to_gguf mapper isn't recognizing the merged names —
    // both are silent P4/P5 regressions.
    let router = names
        .iter()
        .find(|n| n.contains("ffn_gate_inp") && !n.contains("shexp"));
    assert!(
        router.is_some(),
        "qwen35moe GGUF must include blk.{{L}}.ffn_gate_inp.weight (MoE router). \
         Full names: {:?}",
        names
    );

    let merged_gate_exps = names.iter().find(|n| n.contains("ffn_gate_exps.weight"));
    assert!(
        merged_gate_exps.is_some(),
        "qwen35moe GGUF must include merged blk.{{L}}.ffn_gate_exps.weight \
         — if this is missing, P5's merge_moe_experts_in_place was never invoked. \
         Raw per-expert tensors visible: {:?}",
        names.iter().filter(|n| n.contains("experts")).collect::<Vec<_>>()
    );
}

/// Bisection proof: the 4 MTP suffixes are each distinct in the GGUF.
/// If a mapper regression renamed `embed_tokens.weight` to `emb_tokens.weight`
/// (one-letter typo), this assertion fires by name — the test explicitly
/// documents that form per ADR-012 P11 Decision 19 §2.
#[test]
fn qwen35_mtp_suffix_set_is_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("dense-mtp-exact-in");
    let output = tmp.path().join("dense-mtp-exact-out.gguf");
    setup_dense_mtp(&input);
    convert_to_gguf(&input, &output);

    let gguf = GgufFile::open(&output).expect("open");
    let mtp_names: Vec<String> = gguf
        .tensor_names()
        .iter()
        .filter(|n| n.contains(".nextn."))
        .map(|s| s.to_string())
        .collect();

    assert_eq!(
        mtp_names.len(),
        EXPECTED_MTP_SUFFIXES.len(),
        "expected exactly {} MTP tensors, got {}: {:?}",
        EXPECTED_MTP_SUFFIXES.len(),
        mtp_names.len(),
        mtp_names
    );

    // Name audit — exact suffix match, no typos.
    for suffix in EXPECTED_MTP_SUFFIXES {
        let hits: Vec<&String> = mtp_names.iter().filter(|n| n.ends_with(suffix)).collect();
        assert_eq!(
            hits.len(),
            1,
            "suffix {:?} must appear exactly once; got {:?} all_mtp={:?}",
            suffix,
            hits,
            mtp_names
        );
    }
}
