//! ADR-033 P4 — integration test for `hf2q convert-v2`.
//!
//! Synthesizes a tiny Llama-3-shaped HuggingFace directory in a
//! tempdir, runs the `hf2q convert-v2` CLI against it at `q8_0`, and
//! re-parses the output GGUF via `mlx_native::gguf::GgufFile` to
//! assert tensor count + metadata count + per-tensor name / ggml_type.
//!
//! Mirrors the synthetic fixture used by the in-crate
//! `convert::tests::llama3_end_to_end_tiny_safetensors_round_trip` test
//! (`src/convert/mod.rs`) so the two end-to-end suites stay in lockstep:
//! that test calls `HfModelSource::load` + `llama3::map_tensor_name` +
//! `ConvertOrchestrator` directly; THIS one drives the CLI subcommand
//! that wires the same three pieces but adds the arch-detection +
//! arch-mapper-dispatch layer.
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: we also verify the
//! `UnsupportedArch` typed error fires on an unknown `model_type` —
//! the CLI surfaces it as a non-zero exit (input-error code 3) and a
//! diagnostic on stderr.

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use safetensors::tensor::{Dtype, TensorView};

/// Synthesize a tiny Llama-3-shaped safetensors directory WITHOUT 1-D
/// RMSNorm tensors. Rationale: the orchestrator's StandardPolicy
/// routes 1-D norm tensors to F32, and `quantizer_for(F32)` returns
/// `Err(NoQuantizerForType)` by design (see the same caveat in
/// `src/convert/mod.rs::tests::synthesize_tiny_llama3`). The driver
/// has no per-tensor skip hook; the cleanest path is to omit norms
/// from the input safetensors so the convert-v2 pipeline only sees
/// quantizable tensors. Shapes are 32-aligned so Q8_0 lands without
/// any shape fallback.
fn synthesize_tiny_llama3_no_norms(dir: &Path) {
    const HIDDEN: usize = 32;
    const FFN: usize = 64;
    const VOCAB: usize = 64;
    const LAYERS: usize = 2;

    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

    let mk_f32_bytes = |numel: usize, seed: u32| -> Vec<u8> {
        (0..numel)
            .flat_map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as i32;
                let f = (x as f32) / (i32::MAX as f32);
                f.to_le_bytes()
            })
            .collect()
    };

    // Globals (2 quantizable; norm omitted).
    tensors.push((
        "model.embed_tokens.weight".into(),
        vec![VOCAB, HIDDEN],
        mk_f32_bytes(VOCAB * HIDDEN, 1),
    ));
    tensors.push((
        "lm_head.weight".into(),
        vec![VOCAB, HIDDEN],
        mk_f32_bytes(VOCAB * HIDDEN, 3),
    ));

    // Per-block (7 × 2 = 14 quantizable; norms omitted).
    for li in 0..LAYERS {
        let s = (li as u32) * 100;
        tensors.push((
            format!("model.layers.{li}.self_attn.q_proj.weight"),
            vec![HIDDEN, HIDDEN],
            mk_f32_bytes(HIDDEN * HIDDEN, s + 12),
        ));
        tensors.push((
            format!("model.layers.{li}.self_attn.k_proj.weight"),
            vec![HIDDEN, HIDDEN],
            mk_f32_bytes(HIDDEN * HIDDEN, s + 13),
        ));
        tensors.push((
            format!("model.layers.{li}.self_attn.v_proj.weight"),
            vec![HIDDEN, HIDDEN],
            mk_f32_bytes(HIDDEN * HIDDEN, s + 14),
        ));
        tensors.push((
            format!("model.layers.{li}.self_attn.o_proj.weight"),
            vec![HIDDEN, HIDDEN],
            mk_f32_bytes(HIDDEN * HIDDEN, s + 15),
        ));
        tensors.push((
            format!("model.layers.{li}.mlp.gate_proj.weight"),
            vec![FFN, HIDDEN],
            mk_f32_bytes(FFN * HIDDEN, s + 16),
        ));
        tensors.push((
            format!("model.layers.{li}.mlp.up_proj.weight"),
            vec![FFN, HIDDEN],
            mk_f32_bytes(FFN * HIDDEN, s + 17),
        ));
        tensors.push((
            format!("model.layers.{li}.mlp.down_proj.weight"),
            vec![HIDDEN, FFN],
            mk_f32_bytes(HIDDEN * FFN, s + 18),
        ));
    }

    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(n, sh, b)| {
            let v = TensorView::new(Dtype::F32, sh.clone(), b).expect("TensorView");
            (n.clone(), v)
        })
        .collect();
    let view_refs: Vec<(String, &TensorView<'_>)> =
        views.iter().map(|(n, v)| (n.clone(), v)).collect();
    let st_bytes =
        safetensors::tensor::serialize(view_refs, None).expect("serialize safetensors");
    fs::write(dir.join("model.safetensors"), st_bytes).expect("write safetensors");

    let cfg = serde_json::json!({
        "_name_or_path": "synthetic/Llama-3-Tiny-Test",
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "intermediate_size": FFN,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1.0e-5,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
    });
    fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .expect("write config.json");
}

/// Drive `hf2q convert-v2` end-to-end over a tiny Llama-3 fixture and
/// assert the output GGUF round-trips. Targets `q8_0` so every 2-D
/// weight lands on Q8_0 (positional code 3 in mlx_native's enum).
///
/// Acceptance:
/// - exit code 0
/// - 16 tensors written (2 globals + 7 × 2-layer per-block)
/// - 11 metadata KV pairs (Llama3 build_metadata output)
/// - every expected tensor name present with ggml_type Q8_0
#[test]
fn convert_v2_llama3_tiny_round_trip() {
    let model_dir = tempfile::tempdir().unwrap();
    synthesize_tiny_llama3_no_norms(model_dir.path());

    let out = tempfile::NamedTempFile::new().unwrap();
    Command::cargo_bin("hf2q")
        .unwrap()
        .arg("convert-v2")
        .arg(model_dir.path())
        .arg("--quant")
        .arg("q8_0")
        .arg("-o")
        .arg(out.path())
        .assert()
        .success();

    let gguf = mlx_native::gguf::GgufFile::open(out.path()).expect("parse output GGUF");

    // Tensor count: 16 quantizable 2-D tensors
    //   (2 globals: token_embd + output;
    //    7 per-block × 2 layers: q, k, v, o, ffn_gate, ffn_up, ffn_down).
    assert_eq!(gguf.tensor_count(), 16);

    // Metadata count: Llama3's build_metadata emits 11 KV pairs.
    assert_eq!(gguf.metadata_count(), 11);
    assert_eq!(gguf.metadata_string("general.architecture"), Some("llama"));
    assert_eq!(gguf.metadata_u32("llama.embedding_length"), Some(32));
    assert_eq!(gguf.metadata_u32("llama.block_count"), Some(2));
    assert_eq!(gguf.metadata_u32("llama.attention.head_count"), Some(2));
    // file_type round-trips as the LlamaFtype u32 (MostlyQ8_0 = 7).
    assert_eq!(gguf.metadata_u32("general.file_type"), Some(7));

    // Per-tensor name + ggml_type assertions. mlx_native's GgmlType
    // enum is positional: F32=0, F16=1, Q4_0=2, Q8_0=3, ... For Q8_0
    // every quantizable tensor lands on Q8_0 (positional 3).
    let expected_names: &[&str] = &[
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_up.weight",
        "blk.1.ffn_down.weight",
    ];
    for name in expected_names {
        let info = gguf
            .tensor_info(name)
            .unwrap_or_else(|| panic!("missing GGUF tensor `{name}`"));
        assert_eq!(
            info.ggml_type as u32, 3,
            "tensor `{name}` expected positional Q8_0 (3), got {}",
            info.ggml_type as u32
        );
        // ALIGNMENT round-trip — every offset must be 32-aligned per spec.
        assert_eq!(info.offset % 32, 0, "tensor `{name}` offset not aligned");
    }
}

/// Sibling test — feeding an unsupported `model_type` surfaces typed
/// `ConvertV2Error::UnsupportedArch`, which the CLI dispatcher maps to
/// `AppError::Input` → exit code 3 + diagnostic mentioning the
/// offending arch. Per [[feedback-no-loop-suppression-2026-05-17]]:
/// never silent fallback.
#[test]
fn convert_v2_unsupported_arch_errors_typed() {
    let dir = tempfile::tempdir().unwrap();

    // Write a minimal safetensors + config with an unsupported model_type.
    let f32_bytes: Vec<u8> = (0..4).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let view = TensorView::new(Dtype::F32, vec![4], &f32_bytes).unwrap();
    let bytes =
        safetensors::tensor::serialize(vec![("a.weight".to_string(), &view)], None).unwrap();
    fs::write(dir.path().join("model.safetensors"), bytes).unwrap();
    let cfg = serde_json::json!({ "model_type": "mamba" });
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .unwrap();

    let out = tempfile::NamedTempFile::new().unwrap();
    let assert = Command::cargo_bin("hf2q")
        .unwrap()
        .arg("convert-v2")
        .arg(dir.path())
        .arg("--quant")
        .arg("q8_0")
        .arg("-o")
        .arg(out.path())
        .assert()
        .failure();
    // Exit code 3 = EXIT_INPUT_ERROR per src/main.rs (AppError::Input
    // dispatch in cmd_convert_v2 wrapper).
    assert
        .code(3)
        .stderr(predicates::str::contains("mamba"));
}
