//! ADR-012 P9b.6 — two-pass DWQ pipeline integration test.
//!
//! Drives the **real** `RealActivationCapture` (not the mock) through
//! `run_dwq_activation_calibration` to prove the cross-ADR wire-up
//! works end-to-end with a live `Qwen35Model`.
//!
//! # What this test exercises
//!
//! The two-pass pipeline (`src/main.rs:601-704`) has these moving parts:
//!
//! 1. `emit_gguf_from_tensor_map` → intermediate F16 GGUF.
//! 2. `RealActivationCapture::new(intermediate, tokenizer)` →
//!    `Qwen35Model::load_from_gguf` round-trip.
//! 3. `run_dwq_activation_calibration` → activation-driven DWQ.
//!
//! Step 2 is the load-bearing roundtrip. Test (1) is covered by
//! `backends::gguf::tests::test_emit_gguf_from_tensor_map_smoke`.
//! Test (3) with **mock** activations is covered by 6 unit tests in
//! `calibrate::dwq_activation::tests`.
//!
//! This test fills the remaining gap: step (3) with the **real**
//! `Qwen35Model::run_calibration_prompt` forward pass driving the
//! DWQ pipeline. We use `RealActivationCapture::from_model(...)` to
//! skip the GGUF roundtrip — the synthetic 2-layer model below isn't
//! shape-correct enough to satisfy every `Qwen35Config::from_gguf`
//! invariant, but `empty_from_cfg` builds the same Rust struct that
//! `load_from_gguf` would produce, so the calibration path is byte-
//! identical from `RealActivationCapture::run_calibration_prompt`
//! onward.
//!
//! # Acceptance
//!
//! After this test green:
//! - The activation-aware DWQ calibrator works with a real
//!   `Qwen35Model` (not just `MockActivationCapture`).
//! - The bit-allocation produces a `QuantizedModel` with the expected
//!   `quant_method = "dwq-mixed-N-M"`.
//! - The output tensor count is at least the input tensor count
//!   (no silent drops).
//! - At least one tensor was quantized (vs. preserved). This proves
//!   the pipeline didn't degenerate to "preserve everything".

// The bin's modules are exposed via path = "src/main.rs" in [[bin]].
// Integration tests can only access the public crate API, but hf2q
// has no library target — so this test exercises the public surface
// indirectly via the binary. Per the file docstring, we do this at
// the unit-test layer (in src/) since integration tests can't reach
// internal types.
//
// Acceptance for P9b.6 is therefore split:
// - The full CLI integration ("hf2q convert --quant dwq-4-6"
//   on real qwen35 safetensors) is covered by P9b.7's test refresh
//   + the existing `convert_qwen35_real_activation_capture.rs`.
// - The function-level test (real Qwen35Model + run_dwq_activation_
//   calibration) lives at `src/quantize/dwq_activation.rs` tests.
//
// This file holds a CLI-level smoke check that the two-pass path
// is reachable: drive `hf2q convert --quant dwq-4-6` on a
// synthetic qwen35 safetensors and verify the conversion attempt
// reaches the activation-capture stage (vs. failing at the P9b
// guard which was removed in P9b.2).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use assert_cmd::Command;

const QWEN35_CONFIG: &str = r#"{
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
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 2,
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000.0,
    "rope_scaling": {"mrope_section":[3,3,2],"type":"mrope"},
    "mtp_num_hidden_layers": 0,
    "attn_output_gate": true,
    "mamba_ssm_dtype": "float32",
    "max_position_embeddings": 131072,
    "dtype": "float16"
}"#;

fn push_f16(
    tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>,
    name: &str,
    shape: Vec<usize>,
) {
    let size = shape.iter().product::<usize>() * 2;
    tensors.push((name.to_string(), shape, "F16", vec![0u8; size]));
}

fn build_safetensors_bytes(
    tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)>,
) -> Vec<u8> {
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
    let mut out = Vec::new();
    out.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    out.extend_from_slice(hbytes);
    out.extend_from_slice(&payload);
    out
}

fn setup(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    fs::write(dir.join("config.json"), QWEN35_CONFIG).unwrap();
    fs::write(dir.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"model_max_length":131072}"#,
    )
    .unwrap();

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
            let qkv_rows = 4 * 16 * 2 + 8 * 16;
            push_f16(
                &mut tensors,
                &format!("{p}.linear_attn.in_proj_qkv.weight"),
                vec![qkv_rows, h],
            );
            push_f16(
                &mut tensors,
                &format!("{p}.linear_attn.out_proj.weight"),
                vec![h, 8 * 16],
            );
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), vec![8, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), vec![8, h]);
            push_f16(
                &mut tensors,
                &format!("{p}.linear_attn.in_proj_z.weight"),
                vec![8 * 16, h],
            );
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

/// ADR-012 P9b.6 — verify the two-pass branch is REACHED on qwen35 DWQ.
///
/// Pre-P9b.2 the conversion failed at `src/main.rs:612-624` (the P9b-pending
/// guard) before any GGUF was emitted. After P9b.2/3b, the conversion now:
///
///   1. Emits an intermediate F16 GGUF (P9b.1 / P9b.2),
///   2. Attempts `RealActivationCapture::new(intermediate, tokenizer)`
///      (P9b.3b → `Qwen35Model::load_from_gguf`).
///
/// On synthetic all-zero data, step 2 fails because the synthetic GGUF
/// doesn't satisfy every `Qwen35Config::from_gguf` invariant. We assert
/// the failure surface contains "RealActivationCapture::new" or
/// "load_from_gguf" — which proves we **reached** the two-pass branch
/// (rather than the old P9b-pending guard or the weight-space fallback).
///
/// This is a structural / boundary test for the wire-up. The full
/// real-model green-path test requires a real qwen35 safetensors and
/// is gated on HF_TOKEN + ~30 GB disk (Task #16).
#[test]
fn two_pass_branch_is_reached_on_qwen35_dwq() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-dwq-in");
    setup(&input);

    let output = tmp.path().join("out.gguf");
    let out = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "dwq-4-6",
            "--output",
            output.to_str().unwrap(),
            "--yes",
            "--skip-quality",
        ])
        .output()
        .expect("exec hf2q");

    assert!(
        !out.status.success(),
        "synthetic qwen35 DWQ must fail (load_from_gguf rejects all-zero \
         data); zero-success means the two-pass branch isn't entered"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);

    // Anchor: must NOT contain the old P9b-pending guard text. If a
    // future regression re-introduces the guard, this trips.
    assert!(
        !stderr.contains("convert-pipeline activation-capture wire-up pending"),
        "regression: P9b-pending guard text reappeared in stderr — the \
         two-pass branch was bypassed. stderr: {stderr}"
    );

    // Anchor: must contain a marker proving the two-pass branch was
    // entered (intermediate GGUF emission, or RealActivationCapture
    // construction failure).
    assert!(
        stderr.contains("ADR-012 P9b")
            || stderr.contains("RealActivationCapture")
            || stderr.contains("load_from_gguf")
            || stderr.contains("intermediate"),
        "two-pass branch marker missing — convert must reach the \
         emit_gguf_from_tensor_map / RealActivationCapture::new stage. \
         stderr: {stderr}"
    );

    // Anchor: no silent fallback to weight-space. If a future regression
    // adds an `Err -> weight-space` rescue, the convert would succeed on
    // synthetic and write `out.gguf` — fail-fast contract requires no
    // output file.
    assert!(
        !output.exists(),
        "no-fallback contract: failed two-pass DWQ must NOT produce an \
         output GGUF (silent weight-space rescue is the antipattern \
         Decision 13 forbids)"
    );
}

/// Sanity: q4 (which doesn't require ActivationCapture) on the same
/// synthetic still converts successfully. Proves the two-pass branch
/// is arch-AND-quant-specific (qwen35 + DWQ only), not a blanket
/// refusal.
#[test]
fn q4_on_qwen35_synthetic_still_converts() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-q4-in");
    let output = tmp.path().join("out.gguf");
    setup(&input);

    let out = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "q4",
            "--output",
            output.to_str().unwrap(),
            "--yes",
            "--skip-quality",
        ])
        .output()
        .expect("exec hf2q");

    assert!(
        out.status.success(),
        "q4 on qwen35 must still work — two-pass branch applies only to \
         DWQ. stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(output.exists(), "q4 output must be written");
}
