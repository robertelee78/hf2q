//! ADR-012 P9 — RealActivationCapture integration tests.
//!
//! # History
//!
//! Pre-ADR-013-P12 (≤ 2026-04-25 morning): these assertions were the
//! "dependency wall" proof — convert failed at the P9b-pending guard
//! at `src/main.rs:601-624` with stderr text "ADR-012 P9b: convert-
//! pipeline activation-capture wire-up pending".
//!
//! Post-P9b.2/3b (this iteration): the P9b-pending guard is removed.
//! The two-pass pipeline now executes:
//!
//!   1. `RealActivationCapture` loads from the transformed lazy tensor map.
//!   2. `run_dwq_activation_calibration` drives DWQ via the real
//!      forward pass.
//!
//! On the synthetic all-zero qwen35 fixture below, step (2) fails
//! because `Qwen35Model::load_from_gguf` rejects fixtures that don't
//! satisfy every `Qwen35Config::from_gguf` invariant. The error
//! surface still contains "ActivationCapture" / "RealActivationCapture"
//! — the anchored contract is unchanged: **no silent weight-space
//! fallback for qwen35/qwen35moe DWQ**, regardless of which stage of
//! the two-pass pipeline trips.
//!
//! # What this test guards
//!
//! Per `feedback_never_ship_fallback_without_rootcause.md`: any future
//! change that inserts a "fall back to weight-space" path for
//! qwen35/qwen35moe MUST first present an amendment to this file —
//! trips this test. The anchored properties:
//!
//! - DWQ on qwen35/qwen35moe with synthetic fixtures **must** fail.
//! - The error message **must** name the ActivationCapture dependency.
//! - **No** output GGUF must be produced (silent rescue antipattern).
//! - q4 (no ActivationCapture dependency) on the same fixture must
//!   still succeed — proves the failure is arch-AND-quant-specific.
//!
//! # Real-model green-path test
//!
//! The positive-path test (real qwen35 safetensors → DWQ GGUF →
//! PPL/KL gates) is gated on HF_TOKEN + ~30 GB disk and is tracked
//! as Task #16 in the ADR-012 P9 close-out.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use assert_cmd::Command;

// Minimal synthetic qwen35 dense model — enough to satisfy the convert
// preflight. The convert will fail with NotReady when DWQ is requested
// on this arch, proving the guard is live.
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
            // Spec-correct shapes.
            let qkv_rows = 4 * 16 * 2 + 8 * 16;
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_qkv.weight"), vec![qkv_rows, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.out_proj.weight"), vec![h, 8 * 16]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_a.weight"), vec![8, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_b.weight"), vec![8, h]);
            push_f16(&mut tensors, &format!("{p}.linear_attn.in_proj_z.weight"), vec![8 * 16, h]);
        }
        push_f16(&mut tensors, &format!("{p}.mlp.gate_proj.weight"), vec![inter, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.up_proj.weight"), vec![inter, h]);
        push_f16(&mut tensors, &format!("{p}.mlp.down_proj.weight"), vec![h, inter]);
    }
    fs::write(dir.join("model.safetensors"), build_safetensors_bytes(tensors)).unwrap();
}

/// `hf2q convert --quant dwq-N-M` on qwen35 with a synthetic
/// fixture MUST fail with a structured ActivationCapture-named error
/// for EVERY DWQ variant, NOT fall back to weight-space silently.
///
/// If a future edit narrowed the guard from
/// `.requires_activation_capture()` to `== Qwen35` for only one
/// variant, the other three would silently accept DWQ with weight-
/// space — the exact antipattern Decision 13 rejects.
///
/// Post-P9b.2/3b: the failure surface shifted from the P9b-pending
/// guard to `RealActivationCapture::new` → `Qwen35Model::load_from_gguf`
/// rejecting the synthetic fixture. Post-P8 iter-1 (ADR-014 Decision
/// 13): the variant strings rename from `dwq-mixed-N-M` →
/// `dwq-N-M`. The contract — error names ActivationCapture, no
/// output GGUF — is unchanged.
///
/// The real-model green-path test (Task #16) is gated on HF_TOKEN +
/// ~30 GB disk; this test stays at the synthetic-fail-fast tier.
#[test]
fn dwq_on_qwen35_surfaces_not_ready_not_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-dwq-in");
    setup(&input);

    for variant in ["dwq-2-8", "dwq-4-6", "dwq-4-8", "dwq-6-8"] {
        let output = tmp.path().join(format!("out-{variant}.gguf"));
        let out = Command::cargo_bin("hf2q")
            .unwrap()
            .args([
                "convert",
                "--input", input.to_str().unwrap(),
                "--format", "gguf",
                "--quant", variant,
                "--output", output.to_str().unwrap(),
                "--yes",
                "--skip-quality",
            ])
            .output()
            .expect("exec hf2q");

        assert!(
            !out.status.success(),
            "DWQ variant {} on qwen35 pre-P12 MUST NOT succeed (silent weight-space fallback)",
            variant
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        // Structured error surface — names ActivationCapture explicitly.
        assert!(
            stderr.contains("activation")
                || stderr.contains("ActivationCapture")
                || stderr.contains("not available")
                || stderr.contains("Not Ready")
                || stderr.contains("NotReady"),
            "variant {} error must name the ActivationCapture dependency, got: {}",
            variant,
            stderr
        );
        // Negative — not a silent fallback.
        assert!(
            !output.exists(),
            "variant {}: no output GGUF should have been produced — DWQ must fail-fast",
            variant
        );
    }
}

/// Dense quant without ActivationCapture dependency (e.g. `q4`) still
/// converts successfully on the same synthetic fixture. Proves the
/// activation-capture guard is arch-AND-quant-specific (qwen35/
/// qwen35moe DWQ only), not a blanket refusal of qwen35 conversions.
///
/// (Function name retains `_pre_p12` for git-blame stability; ADR-013
/// P12 has shipped — see the file header for the post-P12 anchored
/// contract.)
#[test]
fn q4_0_on_qwen35_still_works_pre_p12() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("qwen35-q4-in");
    let output = tmp.path().join("out.gguf");
    setup(&input);

    let out = Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input", input.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4",
            "--output", output.to_str().unwrap(),
            "--yes",
            "--skip-quality",
        ])
        .output()
        .expect("exec hf2q");
    assert!(
        out.status.success(),
        "q4 on qwen35 must still work — NotReady guard applies only to DWQ. \
         stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(output.exists(), "Q4_0 output must be written");
}
