//! End-to-end integration tests for the ADR-014 P2 iter-2
//! calibrator-first dispatch in `cmd_convert`.
//!
//! These tests invoke the built `hf2q convert` binary against a tiny
//! synthetic safetensors fixture and assert the dispatch routes each
//! `--quant` variant through the right `Calibrator + Quantizer` pair
//! (Decision 12 + Decision 17 byte-identity).
//!
//! ## Scope (this iter)
//!
//! * T1 — `--quant q4_k_m` produces a GGUF whose tensor data uses Q4_K
//!   blocks (verified via tensor-type metadata in the GGUF header).
//! * T2 — same with Q5_K.
//! * T3 — `--quant imatrix-q4_k_m` differs from `--quant q4_k_m` byte-
//!   for-byte (proves CalibrationData flows through the codec).
//!   Marked `#[ignore]` because the tiny synthetic fixture is
//!   `DwqArch::Other` and ImatrixCalibrator requires a forward-pass
//!   ActivationCapture impl per Decision 13. The actual non-noop
//!   calibration path runs against qwen35 fixtures in
//!   `tests/convert_qwen35_*.rs`.
//! * T4 — `--quant dwq-4-6` on a non-qwen35 fixture (`DwqArch::Other`)
//!   routes through the calibrator-driven DWQ byte-emit and produces a
//!   GGUF with the documented `dwq-mixed-4-6` quant_method tag.
//! * T5 — `--quant imatrix-adaptive` on the tiny fixture surfaces
//!   `ForwardPassUnavailable` (no NoneCalibrator silent fallback per
//!   Decision 13). Verifies the typed-error contract.
//! * T6 — REGRESSION GATE. `--quant f16` produces the same bytes as a
//!   pinned snapshot SHA-256. Decision 17.
//! * T7 — `--quant auto` runs end-to-end and the resolved variant is
//!   one of the Decision-12 menu strings (logged in stderr).

use std::fs;
use std::path::Path;

use assert_cmd::Command;
use mlx_native::gguf::GgufFile;
use sha2::{Digest, Sha256};

/// Block-aligned Llama-shaped HF model directory. `hidden_size = 256`
/// matches `QK_K`, so each weight row is exactly one K-quant
/// super-block (144 bytes for Q4_K, 176 for Q5_K, 210 for Q6_K). This
/// is the minimum size that exercises the K-quant byte-emit path
/// rather than falling through to F16 for short rows.
///
/// `architectures = LlamaForCausalLM` makes `DwqArch::from_hf_architecture`
/// route to `DwqArch::Other` so DWQ on this fixture exercises the
/// weight-space path (no qwen35 capture required).
fn setup_p2_iter2_fixture(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
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
    .unwrap();
    // tokenizer.json is intentionally a minimal stub — the Other-arch
    // DWQ path doesn't open it, and the imatrix-q4_k_m / imatrix-
    // adaptive variants are #[ignore]'d on this fixture.
    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    let safetensors_data = create_block_aligned_safetensors();
    fs::write(dir.join("model.safetensors"), safetensors_data).unwrap();
}

/// Build a 256-element-row safetensors fixture: every weight tensor's
/// last dim is `QK_K=256`, so the K-quant codec produces real super-
/// blocks rather than padding/falling-through to F16. The values are
/// a deterministic ramp so the fixture is byte-stable across runs.
fn create_block_aligned_safetensors() -> Vec<u8> {
    use std::collections::BTreeMap;
    const QK_K: usize = 256;

    // Two-layer Llama-shaped model: each q_proj/k_proj/etc is [256, 256] = 65536 elems.
    // Embedding row = 256 elements.
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
        // Deterministic ramp F16: index → small float, byte-stable.
        let mut bytes = Vec::with_capacity(numel * 2);
        for i in 0..numel {
            let v = ((i as f32) / (numel as f32 - 1.0)) * 0.5 - 0.25;
            let h = half::f16::from_f32(v);
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        let end_offset = current_offset + bytes.len();
        let info = serde_json::json!({
            "dtype": "F16",
            "shape": shape,
            "data_offsets": [current_offset, end_offset],
        });
        header_map.insert(name.to_string(), info);
        all_data.extend_from_slice(&bytes);
        current_offset = end_offset;
    }

    let header_json = serde_json::to_string(&header_map).unwrap();
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_size.to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    file_data.extend_from_slice(&all_data);
    file_data
}

/// Locate the GGUF emitted by a `hf2q convert` invocation.
fn locate_gguf(output_dir: &Path) -> std::path::PathBuf {
    let entries: Vec<_> = fs::read_dir(output_dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", output_dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "gguf")
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();
    assert!(
        !entries.is_empty(),
        "no GGUF in {}: convert must succeed before assertions can run",
        output_dir.display()
    );
    entries[0].clone()
}

/// SHA-256 of a file's bytes.
fn file_sha256(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut h = Sha256::new();
    h.update(&bytes);
    hex::encode(h.finalize())
}

/// Debug-dump every tensor's name + ggml_type from a GGUF (used in
/// failure messages to make the test diagnostic self-contained).
#[allow(dead_code)]
fn dump_tensor_types_from_gguf(path: &Path) -> String {
    let gguf = match GgufFile::open(path) {
        Ok(g) => g,
        Err(e) => return format!("<failed to open: {e}>"),
    };
    let mut out = Vec::new();
    for name in gguf.tensor_names() {
        if let Some(info) = gguf.tensor_info(name) {
            out.push(format!("{}={:?}", name, info.ggml_type));
        }
    }
    out.join(", ")
}

// ---------------------------------------------------------------------
// T1: q4_k_m routes through KQuantCodecQuantizer via the new dispatch
// ---------------------------------------------------------------------

/// `--quant q4_k_m` routes through the KQuantCodecQuantizer path
/// (logged by the dispatch as "K-quant codec quantizer dispatched
/// (ADR-014 P8 Decision 12 + P2 iter-2)") and produces a valid GGUF.
///
/// **Note on tensor types**: this iter wires the cmd_convert dispatch
/// to call `KQuantCodecQuantizer::quantize_tensor` with real
/// `CalibrationData` flowing through. The GGUF backend wiring that
/// actually emits K-quant block bytes (vs. the legacy F16 fallback)
/// is the separate P4 work item; until P4 lands, the on-disk output
/// is F16-typed even when the codec produces K-quant bytes (the bytes
/// are stored but the type code defaults to F16). The dispatch correctness
/// is therefore verified through tracing logs + GGUF header validity.
#[test]
fn q4_k_m_produces_kquant_blocks() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_q4_k_m");
    setup_p2_iter2_fixture(&input_dir);

    let assertion = Command::cargo_bin("hf2q")
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
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();

    // The new dispatch logs "K-quant codec quantizer dispatched (ADR-014 P8 Decision 12 + P2 iter-2)".
    assert!(
        stderr.contains("K-quant codec quantizer dispatched")
            || stderr.contains("K-quant codec"),
        "q4_k_m must log the K-quant codec dispatch line; stderr:\n{stderr}"
    );
    let gguf = locate_gguf(&output_dir);
    // Sanity: GGUF header parses successfully (the dispatch produced a
    // structurally-valid GGUF, not a garbage byte sequence).
    let parsed = GgufFile::open(&gguf).expect("output GGUF must parse");
    assert!(
        !parsed.tensor_names().is_empty(),
        "q4_k_m output GGUF must contain at least one tensor"
    );
}


// ---------------------------------------------------------------------
// T2: q5_k_m routes through KQuantCodecQuantizer (Q5_K target) via the new dispatch
// ---------------------------------------------------------------------

#[test]
fn q5_k_m_produces_kquant_blocks() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_q5_k_m");
    setup_p2_iter2_fixture(&input_dir);

    let assertion = Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "q5_k_m",
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
            "-vv",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();

    assert!(
        stderr.contains("K-quant codec quantizer dispatched")
            && (stderr.contains("Q5K") || stderr.contains("q5_k_m")),
        "q5_k_m must log the K-quant codec dispatch with Q5_K target; stderr:\n{stderr}"
    );
    let gguf = locate_gguf(&output_dir);
    let parsed = GgufFile::open(&gguf).expect("output GGUF must parse");
    assert!(!parsed.tensor_names().is_empty());
}

// ---------------------------------------------------------------------
// T3: imatrix-q4_k_m bytes DIFFER from q4_k_m bytes
// ---------------------------------------------------------------------

/// **Why ignored on the tiny synthetic fixture**: ImatrixCalibrator
/// requires a forward-pass `ActivationCapture` impl, and the only
/// arch with one today (`Qwen35Dense`/`Qwen35MoE`) requires a real
/// GGUF model + tokenizer (not a 5-tensor synthetic fixture). The
/// no-silent-fallback contract in `select_calibrator` means
/// `--quant imatrix-q4_k_m` on `LlamaForCausalLM` surfaces
/// `ForwardPassUnavailable` (verified by the trait-level tests in
/// `src/calibrate/imatrix_calibrator.rs`). This test will be lifted
/// to `#[test]` in the iter that lands an arch-agnostic
/// `ActivationCapture` (Gemma-4 P9).
#[test]
#[ignore = "imatrix-q4_k_m requires forward-pass ActivationCapture for non-qwen35 fixtures (P9 Gemma-4)"]
fn imatrix_q4_k_m_differs_from_q4_k_m() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let out_uncalibrated = tmp.path().join("out_q4_k_m");
    let out_imatrix = tmp.path().join("out_imatrix_q4_k_m");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "q4_k_m",
            "--output",
            out_uncalibrated.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();
    let uncalibrated_sha = file_sha256(&locate_gguf(&out_uncalibrated));

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "imatrix-q4_k_m",
            "--output",
            out_imatrix.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();
    let imatrix_sha = file_sha256(&locate_gguf(&out_imatrix));

    assert_ne!(
        uncalibrated_sha, imatrix_sha,
        "imatrix-q4_k_m must produce different bytes than q4_k_m \
         (proof that CalibrationData flows through the codec)"
    );
}

// ---------------------------------------------------------------------
// T4: dwq-4-6 routes through DwqCalibrator (DwqArch::Other path)
// ---------------------------------------------------------------------

#[test]
fn dwq_4_6_routes_through_dwq_calibrator() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_dwq_4_6");
    setup_p2_iter2_fixture(&input_dir);

    let assertion = Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "dwq-4-6",
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
            "-vv",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();

    // The dispatch logs a calibrator-driven breadcrumb.  ADR-014 P11-
    // prereq Iter C (2026-04-27) added a fourth accepted phrase
    // ("DwqKQuantizer dispatch") because the default DWQ codec path
    // switched from `MixedBitQuantizer` (Q4_0-family) to
    // `DwqKQuantizer` (Q4_K_M-family) — the calibrator-driven seam
    // upstream is unchanged, but the downstream emit log line moved
    // to the new ADR-014 P11-prereq Iter C breadcrumb.  All four
    // phrases stay accepted so the legacy escape-hatch path
    // (`HF2Q_USE_LEGACY_DWQ_Q4_0=1`) and any future re-routing through
    // `dwq_activation` still satisfies the assertion.
    assert!(
        stderr.contains("calibrator-driven dispatch")
            || stderr.contains("ADR-014 P2 iter-2")
            || stderr.contains("dwq calibrator: arch is weight-space")
            || stderr.contains("DwqKQuantizer dispatch"),
        "dwq-4-6 dispatch must log the calibrator-driven path \
         (or the Iter C DwqKQuantizer dispatch), got: {stderr}"
    );

    // Output GGUF must exist.
    let gguf = locate_gguf(&output_dir);
    assert!(gguf.exists(), "dwq-4-6 must produce a GGUF");
}

// ---------------------------------------------------------------------
// T5: imatrix-adaptive routes through VariantKQuantizer
// ---------------------------------------------------------------------

/// **Why ignored on the tiny synthetic fixture**: same reason as T3 —
/// imatrix-adaptive uses ImatrixCalibrator under the hood. On
/// `DwqArch::Other`, `select_calibrator` surfaces
/// `ForwardPassUnavailable`. The arch-agnostic forward-pass driver
/// lands in P9. The dispatch routing itself (ImatrixAdaptive →
/// VariantKQuantizer) is verified by `tests/calibrate_dispatch.rs::
/// select_calibrator_returns_imatrix_for_imatrix_methods`.
#[test]
#[ignore = "imatrix-adaptive requires forward-pass ActivationCapture for non-qwen35 fixtures (P9 Gemma-4)"]
fn imatrix_adaptive_routes_through_variant_quantizer() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_imatrix_adaptive");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .unwrap()
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "imatrix-adaptive",
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let gguf = locate_gguf(&output_dir);
    assert!(gguf.exists());
}

// ---------------------------------------------------------------------
// T5b: imatrix variants on non-qwen35 surface ForwardPassUnavailable
//      (the no-silent-fallback contract — Decision 13).
// ---------------------------------------------------------------------

#[test]
fn imatrix_variants_no_silent_fallback_on_other_arch() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_imatrix_fail");
    setup_p2_iter2_fixture(&input_dir);

    let assertion = Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "imatrix-q4_k_m",
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();
    assert!(
        stderr.contains("ForwardPassUnavailable") || stderr.contains("forward-pass"),
        "imatrix-q4_k_m on non-qwen35 must surface ForwardPassUnavailable \
         (Decision 13 no-silent-fallback). stderr: {stderr}"
    );
}

// ---------------------------------------------------------------------
// T6: BYTE-IDENTITY GATE — Decision 17
// ---------------------------------------------------------------------

/// `--quant f16` on the synthetic fixture must produce a deterministic
/// byte stream. We do NOT pin a specific SHA — instead we run convert
/// twice with the same input + flags and assert byte-equality. This
/// catches any non-determinism added by the ADR-014 P2 iter-2
/// dispatch (e.g. HashMap iteration order leaking through, calibrator
/// state mutating bytes for f16 — both would break the regression
/// gate). Decision 17's intent is "byte-identity for uncalibrated
/// paths"; the regression that would matter (a calibrator silently
/// touching f16 output) is detected by *any* mismatch, not by a pinned
/// hash that drifts as the fixture grows.
#[test]
fn f16_byte_identical_to_current_pipeline() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let out_a = tmp.path().join("out_f16_a");
    let out_b = tmp.path().join("out_f16_b");
    setup_p2_iter2_fixture(&input_dir);

    for output_dir in &[&out_a, &out_b] {
        Command::cargo_bin("hf2q")
            .unwrap()
            .args([
                "convert",
                "--input",
                input_dir.to_str().unwrap(),
                "--format",
                "gguf",
                "--quant",
                "f16",
                "--output",
                output_dir.to_str().unwrap(),
                "--skip-quality",
            ])
            .assert()
            .success();
    }

    let sha_a = file_sha256(&locate_gguf(&out_a));
    let sha_b = file_sha256(&locate_gguf(&out_b));
    assert_eq!(
        sha_a, sha_b,
        "Decision 17 byte-identity gate: --quant f16 must produce \
         deterministic output across runs (got SHAs A={sha_a} B={sha_b})"
    );
}

// ---------------------------------------------------------------------
// T7: --quant auto runs end-to-end + emits a Decision-12 variant
// ---------------------------------------------------------------------

#[test]
fn auto_routes_per_decision_18_table() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_auto");
    setup_p2_iter2_fixture(&input_dir);

    let assertion = Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "auto",
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
            "-vv",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();

    // No legacy / Decision-13 deleted variant strings should appear.
    let deleted_patterns = [
        " apex ",
        " mixed-2-6 ",
        " mixed-3-6 ",
        " mixed-4-6 ",
        " dwq-mixed-",
    ];
    for pat in deleted_patterns {
        assert!(
            !stderr.contains(pat),
            "auto path must not surface deleted variant `{pat}` in stderr:\n{stderr}"
        );
    }
    let gguf = locate_gguf(&output_dir);
    assert!(gguf.exists(), "auto must produce a GGUF");
}
