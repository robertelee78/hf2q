//! ADR-014 P11-prereq Iter A — codec-direct fast-path regression tests.
//!
//! Locks the consumer-side fix in `src/backends/gguf.rs`:
//! `quant_info_to_ggml_type` AND `repack_to_ggml_blocks` now recognise
//! `method == METHOD_K_QUANT_CODEC_DIRECT` (set by both
//! `KQuantCodecQuantizer` and `VariantKQuantizer`) and route via
//! `info.ggml_type` BEFORE the bits-fallback fires.
//!
//! Pre-Iter A the path was:
//!   `bits = 0` (sentinel) + `ggml_type = Some("Q4_K")` + Q4_K bytes
//!     → `quant_info_to_ggml_type` matched the K-quant fall-through arm
//!       at `gguf.rs:1645-1649` (no return), then fell into the `_ =>`
//!       arm at `gguf.rs:1672-1677` and returned `GGML_TYPE_F16` (1).
//!     → `repack_to_ggml_blocks` hit the warn-and-return-raw branch at
//!       `gguf.rs:1064-1075` because `info.scales` is `None`.
//!   Result: GGUF header reported **F16** (type 1) while on-disk bytes
//!   were **Q4_K** — MALFORMED for llama.cpp.
//!
//! Post-Iter A: header type code matches on-disk bytes for every
//! `--quant q4_k_m`, `--quant q5_k_m`, `--quant q6_k` invocation that
//! routes through `KQuantCodecQuantizer` or `VariantKQuantizer`.
//!
//! These tests drive the full binary end-to-end and read back the GGUF
//! header via the `mlx_native::gguf::GgufFile` reader (already a
//! sibling-crate dep — `Cargo.toml:50`).
//!
//! Sentinel string under test: `"k_quant_codec_direct"` —
//! `pub const METHOD_K_QUANT_CODEC_DIRECT` at
//! `src/quantize/k_quant_codec_quantizer.rs:114`. Used by:
//!   * `KQuantCodecQuantizer::quantize_tensor` at line 222.
//!   * `VariantKQuantizer::quantize_tensor` at line 225.
//! Both quantizers produce identical sentinel strings, so a single
//! consumer-side recogniser covers both.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use mlx_native::gguf::GgufFile;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;

const QK_K: usize = 256;

/// Build a minimum-viable HF input directory whose tensors are exactly
/// one K-quant super-block per row (`hidden_size = QK_K = 256`). This
/// is the same shape used by `tests/cmd_convert_dispatch.rs`'s
/// `setup_p2_iter2_fixture` (precedent).
fn setup_block_aligned_fixture(dir: &Path) {
    fs::create_dir_all(dir).expect("create input dir");
    fs::write(
        dir.join("config.json"),
        r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "vocab_size": 256,
            "intermediate_size": 256,
            "dtype": "float16"
        }"#,
    )
    .expect("write config.json");
    fs::write(dir.join("tokenizer.json"), "{}").expect("write tokenizer.json");
    fs::write(dir.join("tokenizer_config.json"), "{}").expect("write tokenizer_config.json");
    fs::write(
        dir.join("model.safetensors"),
        build_block_aligned_safetensors(),
    )
    .expect("write model.safetensors");
}

fn build_block_aligned_safetensors() -> Vec<u8> {
    let tensors: Vec<(&str, Vec<usize>)> = vec![
        ("model.layers.0.self_attn.q_proj.weight", vec![QK_K, QK_K]),
        ("model.layers.0.input_layernorm.weight", vec![QK_K]),
        ("model.layers.1.self_attn.q_proj.weight", vec![QK_K, QK_K]),
        ("model.layers.1.input_layernorm.weight", vec![QK_K]),
        ("model.embed_tokens.weight", vec![32, QK_K]),
    ];

    let mut header_map = BTreeMap::new();
    let mut current_offset = 0usize;
    let mut all_data = Vec::new();
    for (name, shape) in &tensors {
        let numel: usize = shape.iter().product();
        let mut bytes = Vec::with_capacity(numel * 2);
        for i in 0..numel {
            // Deterministic ramp so the test is byte-stable across runs;
            // the value range matters less than the codec exercising
            // a real per-block min/max scale search.
            let v = (i as f32 - (numel as f32 / 2.0)) / (numel as f32);
            bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        let nbytes = bytes.len();
        let entry = serde_json::json!({
            "dtype": "F16",
            "shape": shape,
            "data_offsets": [current_offset, current_offset + nbytes],
        });
        header_map.insert(name.to_string(), entry);
        current_offset += nbytes;
        all_data.extend_from_slice(&bytes);
    }
    let header_json = serde_json::to_vec(&header_map).expect("safetensors header json");
    let mut out = Vec::with_capacity(8 + header_json.len() + all_data.len());
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    out.extend_from_slice(&all_data);
    out
}

fn locate_gguf(dir: &Path) -> PathBuf {
    fs::read_dir(dir)
        .expect("read output dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|e| e == "gguf"))
        .expect("output dir must contain a .gguf file")
}

/// Convert with the given `--quant` and return the GGML type code
/// observed for the canonical Q-projection tensor of layer 0.
fn convert_and_read_q_proj_type(quant: &str, out_subdir: &str) -> GgmlType {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join(out_subdir);
    setup_block_aligned_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            quant,
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
            "-vv",
        ])
        .assert()
        .success();

    let gguf = locate_gguf(&output_dir);
    let parsed = GgufFile::open(&gguf).expect("output GGUF must parse");
    let q_name = "blk.0.attn_q.weight";
    let info = parsed
        .tensor_info(q_name)
        .unwrap_or_else(|| panic!("missing {q_name} in {gguf:?}"));
    info.ggml_type
}

// ---------------------------------------------------------------------
// Codec-direct sentinel: header type code matches on-disk K-quant bytes
// ---------------------------------------------------------------------

/// Pre-Iter A this test would observe `GgmlType::F16`. Post-Iter A the
/// codec-direct fast-path in `quant_info_to_ggml_type` routes via
/// `info.ggml_type = Some("Q4_K")` and the header type code matches the
/// on-disk Q4_K block bytes produced by `KQuantCodecQuantizer`.
#[test]
fn kquant_codec_direct_q4_k_writes_q4_k_header() {
    let observed = convert_and_read_q_proj_type("q4_k_m", "out_q4_k_m");
    assert_eq!(
        observed,
        GgmlType::Q4_K,
        "q4_k_m must write a Q4_K-typed GGUF header (codec-direct fast-path); \
         pre-Iter A this returned F16 atop Q4_K bytes — MALFORMED for llama.cpp"
    );
}

/// Same regression for Q5_K — the second of the three KQuantTarget
/// formats KQuantCodecQuantizer emits via VariantKQuantizer.
#[test]
fn kquant_codec_direct_q5_k_writes_q5_k_header() {
    let observed = convert_and_read_q_proj_type("q5_k_m", "out_q5_k_m");
    assert_eq!(
        observed,
        GgmlType::Q5_K,
        "q5_k_m must write a Q5_K-typed GGUF header"
    );
}

/// Q6_K is in the existing K-quant code path that already returned
/// `GGML_TYPE_Q6_K` correctly via the `upper.starts_with("Q6_K")` guard
/// at `gguf.rs:1640-1642`. This test is the third leg of the
/// {Q4_K, Q5_K, Q6_K} regression triangle and locks the codec-direct
/// path's correctness for Q6_K too — protecting against a regression
/// where the new fast-path's `ggml_type_from_name` lookup breaks the
/// previously-correct Q6_K case.
#[test]
fn kquant_codec_direct_q6_k_writes_q6_k_header() {
    let observed = convert_and_read_q_proj_type("q6_k", "out_q6_k");
    assert_eq!(
        observed,
        GgmlType::Q6_K,
        "q6_k must write a Q6_K-typed GGUF header"
    );
}

/// `--quant q4_k_m` routes through `VariantKQuantizer`, which emits
/// `method == METHOD_K_QUANT_CODEC_DIRECT` for every quantizable
/// tensor. The codec-direct fast-path must therefore route correctly
/// across multiple layers and multiple tensor names — locking against
/// a regression where the sentinel match accidentally only fires for
/// `blk.0`. Both `blk.0.attn_q.weight` and `blk.1.attn_q.weight` must
/// land Q4_K-typed in the header; the input `input_layernorm.weight`
/// tensors are preserved at F16 by design (per `TensorRef::is_weight`
/// at `src/ir/mod.rs:155-190`) so the codec-direct path is NOT exercised
/// for those — they exit through the `info.preserved` arm at
/// `quant_info_to_ggml_type:1631`.
#[test]
fn kquant_codec_direct_variant_q4_k_m_routes_across_layers() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_q4_k_m_variant");
    setup_block_aligned_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "q4_k_m",
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
            "-vv",
        ])
        .assert()
        .success();

    let parsed = GgufFile::open(&locate_gguf(&output_dir)).expect("GGUF parse");

    for tname in ["blk.0.attn_q.weight", "blk.1.attn_q.weight"] {
        let info = parsed
            .tensor_info(tname)
            .unwrap_or_else(|| panic!("missing {tname}"));
        assert_eq!(
            info.ggml_type,
            GgmlType::Q4_K,
            "{tname}: Q4_K_M variant must produce Q4_K-typed header for every \
             quantizable layer (codec-direct fast-path)"
        );
    }
}
