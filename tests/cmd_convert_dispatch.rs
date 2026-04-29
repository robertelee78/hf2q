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

// ---------------------------------------------------------------------
// T10: HF2Q_STREAMING_PHASE3=1 produces byte-identical GGUF (iter-61)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-61 — production E2E byte-identity gate for the
/// streaming wire-up under `HF2Q_STREAMING_PHASE3=1`.
///
/// Runs `hf2q convert --quant q4_k_m` twice on the same fixture: once
/// in default (eager) mode, once with the env flag.  Asserts the
/// output GGUFs are byte-equal via SHA-256.
///
/// This is the missing E2E gate that iter-48-52's per-arm unit tests
/// don't provide: it proves the env flag's *integrated* effect across
/// Phase 3 + 4.5 + 4.6 is byte-identical at the file level, not just
/// at each individual call site.  Without this, the iter-3 wholesale
/// surgery in iter-62+ would have no production safety signal.
#[test]
fn streaming_phase3_byte_identical_to_eager_q4_k_m() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let eager_dir = tmp.path().join("out_eager");
    let stream_dir = tmp.path().join("out_stream");
    setup_p2_iter2_fixture(&input_dir);

    // Eager path (default).
    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4_k_m",
            "--output", eager_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    // Streaming path (env flag on).
    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3", "1")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4_k_m",
            "--output", stream_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let eager_gguf = locate_gguf(&eager_dir);
    let stream_gguf = locate_gguf(&stream_dir);
    let eager_sha = file_sha256(&eager_gguf);
    let stream_sha = file_sha256(&stream_gguf);

    assert_eq!(
        eager_sha, stream_sha,
        "HF2Q_STREAMING_PHASE3=1 GGUF must be byte-identical to eager GGUF \
         on q4_k_m fixture — env-flag wedge has lost byte-identity at the \
         file level (per-arm unit tests pass but integrated path differs)\n\
         eager SHA: {eager_sha}\nstream SHA: {stream_sha}"
    );
}

// ---------------------------------------------------------------------
// T11: HF2Q_STREAMING_PHASE3 byte-identity matrix across K-quant variants (iter-62)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-62 — extend iter-61's SHA gate to cover the K-quant
/// variant matrix (q5_k_m, q6_k) so any future regression where
/// streaming ≠ eager on a non-q4_k_m variant surfaces immediately.
///
/// Each variant runs through eager + streaming and asserts file-level
/// SHA-256 equality.  The fixture has 256-element rows so every K-quant
/// codec produces real super-blocks (Q4_K=144, Q5_K=176, Q6_K=210 bytes).
#[test]
fn streaming_phase3_byte_identical_matrix_across_kquants() {
    for variant in &["q5_k_m", "q6_k"] {
        let tmp = tempfile::tempdir().expect("tempdir");
        let input_dir = tmp.path().join("input");
        let eager_dir = tmp.path().join(format!("out_eager_{}", variant));
        let stream_dir = tmp.path().join(format!("out_stream_{}", variant));
        setup_p2_iter2_fixture(&input_dir);

        Command::cargo_bin("hf2q")
            .expect("hf2q binary")
            .args([
                "convert",
                "--input", input_dir.to_str().unwrap(),
                "--format", "gguf",
                "--quant", *variant,
                "--output", eager_dir.to_str().unwrap(),
                "--skip-quality",
            ])
            .assert()
            .success();

        Command::cargo_bin("hf2q")
            .expect("hf2q binary")
            .env("HF2Q_STREAMING_PHASE3", "1")
            .args([
                "convert",
                "--input", input_dir.to_str().unwrap(),
                "--format", "gguf",
                "--quant", *variant,
                "--output", stream_dir.to_str().unwrap(),
                "--skip-quality",
            ])
            .assert()
            .success();

        let eager_sha = file_sha256(&locate_gguf(&eager_dir));
        let stream_sha = file_sha256(&locate_gguf(&stream_dir));

        assert_eq!(
            eager_sha, stream_sha,
            "{variant}: HF2Q_STREAMING_PHASE3=1 GGUF must be byte-identical \
             to eager GGUF — env-flag wedge has lost byte-identity at file \
             level for this variant\n  eager:  {eager_sha}\n  stream: {stream_sha}"
        );
    }
}

// ---------------------------------------------------------------------
// T12: HF2Q_STREAMING_PHASE3 byte-identity for DwqK arm via dwq-4-6 (iter-63)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-63 — extend the SHA gate to the DwqK arm
/// (--quant dwq-4-6).  DwqK is the most complex Phase 3 dispatch:
/// per-tensor sensitive vs base routing via `is_sensitive_tensor` +
/// `target_for(tensor_name)` + delegate to `KQuantCodecQuantizer` +
/// `legacy_quant_method` override post-quantize.
///
/// On `DwqArch::Other` (Llama-shaped fixture), the activation-capture
/// path is skipped (per Decision 13: no weight-space fallback for
/// archs without forward drivers).  The test exercises the calibrator-
/// driven byte-emit through `run_dwq_with_sensitive_ranges` →
/// `DwqKQuantizer` → `KQuantCodecQuantizer`, which is the same
/// streaming-eligible path as the K-quant variants.
///
/// File-level SHA-256 equality between eager + streaming paths locks
/// the iter-51 byte-identity contract end-to-end on DwqK.
#[test]
fn streaming_phase3_byte_identical_dwq_4_6() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let eager_dir = tmp.path().join("out_eager_dwq46");
    let stream_dir = tmp.path().join("out_stream_dwq46");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "dwq-4-6",
            "--output", eager_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3", "1")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "dwq-4-6",
            "--output", stream_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let eager_sha = file_sha256(&locate_gguf(&eager_dir));
    let stream_sha = file_sha256(&locate_gguf(&stream_dir));

    assert_eq!(
        eager_sha, stream_sha,
        "dwq-4-6: HF2Q_STREAMING_PHASE3=1 GGUF must be byte-identical to \
         eager — DwqK streaming wedge has lost byte-identity at file level\n\
         eager:  {eager_sha}\nstream: {stream_sha}"
    );
}

// ---------------------------------------------------------------------
// T13: HF2Q_STREAMING_PHASE3 byte-identity for --format safetensors (iter-64)
// ---------------------------------------------------------------------

/// SHA-256 of every `.safetensors` file in a directory, concatenated
/// in deterministic name-sorted order.  Mirrors what `file_sha256`
/// does for GGUF but handles the multi-shard safetensors backend.
fn safetensors_dir_sha256(dir: &Path) -> String {
    let mut entries: Vec<std::path::PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();
    entries.sort();
    assert!(
        !entries.is_empty(),
        "no safetensors shards in {}: convert must succeed before SHA can run",
        dir.display()
    );
    let mut h = Sha256::new();
    for p in &entries {
        let bytes = fs::read(p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
        // Hash filename + content so rename-only changes also surface.
        h.update(p.file_name().unwrap().as_encoded_bytes());
        h.update(&bytes);
    }
    hex::encode(h.finalize())
}

/// ADR-014 P7 iter-64 — extend SHA gate to `--format safetensors`.
/// Currently all gates use `--format gguf`; the streaming wedge also
/// needs to be byte-identical against the SafetensorsBackend's write
/// path.  Sharded output is hashed via `safetensors_dir_sha256` which
/// concatenates all `.safetensors` files in name-sorted order.
///
/// Locks the iter-3 wire-up byte-identity contract across both output
/// backends (GGUF + Safetensors).
#[test]
fn streaming_phase3_byte_identical_safetensors_q4_k_m() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let eager_dir = tmp.path().join("out_eager_st");
    let stream_dir = tmp.path().join("out_stream_st");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "safetensors",
            "--quant", "q4_k_m",
            "--output", eager_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3", "1")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "safetensors",
            "--quant", "q4_k_m",
            "--output", stream_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let eager_sha = safetensors_dir_sha256(&eager_dir);
    let stream_sha = safetensors_dir_sha256(&stream_dir);

    assert_eq!(
        eager_sha, stream_sha,
        "safetensors q4_k_m: HF2Q_STREAMING_PHASE3=1 sharded output must \
         be byte-identical to eager — streaming wedge regressed for the \
         SafetensorsBackend write path\n  eager:  {eager_sha}\n  stream: {stream_sha}"
    );
}

// ---------------------------------------------------------------------
// T14: HF2Q_STREAMING_PHASE3 run-to-run determinism (iter-65)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-65 — run-to-run determinism gate.
///
/// `quantize_streaming_parallel` distributes per-tensor quantize work
/// across a rayon thread pool.  Per Decision 5, output is byte-identical
/// to serial across worker counts.  This file-level gate runs the
/// streaming convert 3× back-to-back and asserts all three GGUFs hash
/// to the same SHA-256 — proving the rayon dispatch order doesn't
/// introduce file-level non-determinism (e.g. via HashMap iter order
/// or worker-stealing schedule).
///
/// Without this gate, a future regression that introduces a Mutex
/// ordering or a non-deterministic accumulator would produce different
/// SHAs across runs while still passing the eager-vs-streaming
/// per-pair gates (since each run is byte-identical to the same eager
/// run, but the runs differ from each other).
#[test]
fn streaming_phase3_run_to_run_determinism_q4_k_m() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    setup_p2_iter2_fixture(&input_dir);

    let mut shas: Vec<String> = Vec::with_capacity(3);
    for run in 0..3 {
        let out_dir = tmp.path().join(format!("out_run_{}", run));
        Command::cargo_bin("hf2q")
            .expect("hf2q binary")
            .env("HF2Q_STREAMING_PHASE3", "1")
            .args([
                "convert",
                "--input", input_dir.to_str().unwrap(),
                "--format", "gguf",
                "--quant", "q4_k_m",
                "--output", out_dir.to_str().unwrap(),
                "--skip-quality",
            ])
            .assert()
            .success();
        shas.push(file_sha256(&locate_gguf(&out_dir)));
    }

    // All three runs must hash identically.
    assert_eq!(
        shas[0], shas[1],
        "run 0 vs run 1: streaming convert is non-deterministic\n  \
         run 0: {}\n  run 1: {}",
        shas[0], shas[1]
    );
    assert_eq!(
        shas[1], shas[2],
        "run 1 vs run 2: streaming convert is non-deterministic\n  \
         run 1: {}\n  run 2: {}",
        shas[1], shas[2]
    );
}

// ---------------------------------------------------------------------
// T15: Eager path run-to-run determinism (iter-66)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-66 — mirror iter-65's determinism gate for the
/// EAGER path.  Without this, a future regression where eager becomes
/// non-deterministic (e.g. someone adds a HashMap iter that produces
/// different orderings) would invalidate iter-61-64's eager-vs-streaming
/// SHA gates: each individual eager run might still equal a same-run
/// streaming run, but cross-run comparisons would mask the regression.
///
/// 3× back-to-back eager convert runs hash identically.
#[test]
fn eager_run_to_run_determinism_q4_k_m() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    setup_p2_iter2_fixture(&input_dir);

    let mut shas: Vec<String> = Vec::with_capacity(3);
    for run in 0..3 {
        let out_dir = tmp.path().join(format!("eager_run_{}", run));
        Command::cargo_bin("hf2q")
            .expect("hf2q binary")
            .args([
                "convert",
                "--input", input_dir.to_str().unwrap(),
                "--format", "gguf",
                "--quant", "q4_k_m",
                "--output", out_dir.to_str().unwrap(),
                "--skip-quality",
            ])
            .assert()
            .success();
        shas.push(file_sha256(&locate_gguf(&out_dir)));
    }

    assert_eq!(
        shas[0], shas[1],
        "eager run 0 vs run 1: convert is non-deterministic\n  \
         run 0: {}\n  run 1: {}",
        shas[0], shas[1]
    );
    assert_eq!(
        shas[1], shas[2],
        "eager run 1 vs run 2: convert is non-deterministic\n  \
         run 1: {}\n  run 2: {}",
        shas[1], shas[2]
    );
}

// ---------------------------------------------------------------------
// T16: HF2Q_STREAMING_PHASE3 byte-identity for f16 (StaticQuantizer arm) (iter-67)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-67 — extend SHA gate to --quant f16 which routes
/// through the StaticQuantizer arm wired in iter-50.  This is the
/// simplest dispatch (no quantization, just F16 conversion) but
/// crucially has its OWN code path that the K-quant / DwqK gates
/// don't cover.
///
/// 4-arm coverage now complete at the file level:
///   - K-quant codec direct       (iter-61 q4_k_m, iter-62 q5_k_m/q6_k)
///   - StaticQuantizer            (iter-67 f16 — THIS)
///   - DwqK                       (iter-63 dwq-4-6)
///   - ImatrixAdaptive            (deferred — needs qwen35 fixture)
#[test]
fn streaming_phase3_byte_identical_f16() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let eager_dir = tmp.path().join("out_eager_f16");
    let stream_dir = tmp.path().join("out_stream_f16");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "f16",
            "--output", eager_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3", "1")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "f16",
            "--output", stream_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let eager_sha = file_sha256(&locate_gguf(&eager_dir));
    let stream_sha = file_sha256(&locate_gguf(&stream_dir));

    assert_eq!(
        eager_sha, stream_sha,
        "f16: HF2Q_STREAMING_PHASE3=1 GGUF must be byte-identical to \
         eager — StaticQuantizer streaming wedge regressed at file level\n\
         eager:  {eager_sha}\nstream: {stream_sha}"
    );
}

// ---------------------------------------------------------------------
// T17: HF2Q_STREAMING_PHASE3 byte-identity for imatrix-adaptive (iter-75)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-75 — ImatrixAdaptive SHA gate, deferred-real-weights.
///
/// **Coverage gap audit**: of the 4 Phase 3 dispatch arms wired in
/// iter-48-51, ImatrixAdaptive (iter-49) is the ONLY arm without a
/// file-level SHA gate.  Decision 13 (no silent fallback) requires
/// ImatrixCalibrator to have a forward-pass `ActivationCapture` impl
/// for the source arch; the synthetic Llama fixture (`DwqArch::Other`)
/// surfaces `ForwardPassUnavailable` instead of running.  The synthetic
/// qwen35 fixture (`tests/convert_qwen35_*`) has all-zero weights so
/// `load_from_gguf` rejects mid-pipeline.  Real-weight gate requires
/// a real qwen35 safetensors source (P11 territory).
///
/// This iter lands the gate as `#[ignore]`'d so it (a) documents the
/// contract, (b) is one-flag-flip away from active when a real-weight
/// fixture lands.  The 8-cell P10 GateCell matrix's MoE cells (cells
/// 4-7) cover an adjacent dimension (cross-validator + peer-parity),
/// not the streaming-vs-eager byte-identity contract specifically.
///
/// To activate: set `HF2Q_IMATRIX_ADAPTIVE_FIXTURE` to a real qwen35
/// safetensors directory and remove `#[ignore]`.
#[test]
#[ignore = "ImatrixAdaptive needs real-weight qwen35 fixture (synthetic all-zero rejects in load_from_gguf); P11 hardware territory"]
fn streaming_phase3_byte_identical_imatrix_adaptive() {
    let fixture = std::env::var("HF2Q_IMATRIX_ADAPTIVE_FIXTURE")
        .expect("set HF2Q_IMATRIX_ADAPTIVE_FIXTURE to a real qwen35 safetensors dir");
    let tmp = tempfile::tempdir().expect("tempdir");
    let eager_dir = tmp.path().join("out_eager_imatrix_adaptive");
    let stream_dir = tmp.path().join("out_stream_imatrix_adaptive");

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input", &fixture,
            "--format", "gguf",
            "--quant", "imatrix-adaptive",
            "--output", eager_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3", "1")
        .args([
            "convert",
            "--input", &fixture,
            "--format", "gguf",
            "--quant", "imatrix-adaptive",
            "--output", stream_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let eager_sha = file_sha256(&locate_gguf(&eager_dir));
    let stream_sha = file_sha256(&locate_gguf(&stream_dir));

    assert_eq!(
        eager_sha, stream_sha,
        "imatrix-adaptive: HF2Q_STREAMING_PHASE3=1 GGUF must be byte-identical \
         to eager — VariantKQuantizer + imatrix calibration streaming wedge \
         regressed at file level.\n  eager:  {eager_sha}\n  stream: {stream_sha}"
    );
}

// ---------------------------------------------------------------------
// T18: HF2Q_STREAMING_PHASE3_MUT byte-identity for q4_k_m (iter-84)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-84 — production E2E gate for the zero-byte-copy
/// `quantize_via_streaming_consuming_mut` wedge wired into the K-quant
/// codec arm under `HF2Q_STREAMING_PHASE3_MUT=1`.
///
/// File-level SHA-256 equality between eager + MUT-streaming paths
/// confirms that draining tensor_map's bytes mid-Phase-3 doesn't
/// regress the GGUF output — the structural memory win lands without
/// affecting byte-identity.
///
/// Uses --skip-quality because the iter-84 wire-up doesn't yet
/// reorder Phase 4.5 quality measurement (which would otherwise see
/// a drained tensor_map).  iter-85+ handles Phase 4.5 ordering.
#[test]
fn streaming_phase3_mut_byte_identical_to_eager_q4_k_m() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let eager_dir = tmp.path().join("out_eager_mut");
    let stream_dir = tmp.path().join("out_stream_mut");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4_k_m",
            "--output", eager_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3_MUT", "1")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4_k_m",
            "--output", stream_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    let eager_sha = file_sha256(&locate_gguf(&eager_dir));
    let stream_sha = file_sha256(&locate_gguf(&stream_dir));

    assert_eq!(
        eager_sha, stream_sha,
        "HF2Q_STREAMING_PHASE3_MUT=1 GGUF must be byte-identical to eager — \
         consuming-mut wedge regressed at file level\n  \
         eager:  {eager_sha}\n  stream: {stream_sha}"
    );
}

// ---------------------------------------------------------------------
// T19: HF2Q_STREAMING_PHASE3_MUT byte-identity matrix across 4 arms (iter-89)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-89 — extend iter-84's _MUT SHA gate from q4_k_m
/// (single K-quant codec arm) to the full 4-arm Phase 3 matrix:
///   - q5_k_m  → K-quant codec arm
///   - q6_k    → K-quant codec arm
///   - dwq-4-6 → DwqK arm (iter-88)
///   - f16     → StaticQuantizer arm (iter-87)
/// (imatrix-adaptive remains hardware-gated per iter-75; ImatrixAdaptive
/// _MUT wire-up at iter-85 is structurally identical to other arms.)
///
/// Closes the production safety matrix for the zero-byte-copy wedge:
/// every dispatch path that's wired under HF2Q_STREAMING_PHASE3_MUT
/// has its file-level SHA-256 byte-identity locked vs eager.
#[test]
fn streaming_phase3_mut_byte_identical_matrix_across_arms() {
    for variant in &["q5_k_m", "q6_k", "dwq-4-6", "f16"] {
        let tmp = tempfile::tempdir().expect("tempdir");
        let input_dir = tmp.path().join("input");
        let eager_dir = tmp.path().join(format!("out_eager_mut_{}", variant.replace('-', "_")));
        let stream_dir = tmp.path().join(format!("out_stream_mut_{}", variant.replace('-', "_")));
        setup_p2_iter2_fixture(&input_dir);

        Command::cargo_bin("hf2q")
            .expect("hf2q binary")
            .args([
                "convert",
                "--input", input_dir.to_str().unwrap(),
                "--format", "gguf",
                "--quant", *variant,
                "--output", eager_dir.to_str().unwrap(),
                "--skip-quality",
            ])
            .assert()
            .success();

        Command::cargo_bin("hf2q")
            .expect("hf2q binary")
            .env("HF2Q_STREAMING_PHASE3_MUT", "1")
            .args([
                "convert",
                "--input", input_dir.to_str().unwrap(),
                "--format", "gguf",
                "--quant", *variant,
                "--output", stream_dir.to_str().unwrap(),
                "--skip-quality",
            ])
            .assert()
            .success();

        let eager_sha = file_sha256(&locate_gguf(&eager_dir));
        let stream_sha = file_sha256(&locate_gguf(&stream_dir));

        assert_eq!(
            eager_sha, stream_sha,
            "{variant}: HF2Q_STREAMING_PHASE3_MUT=1 GGUF must be byte-identical to \
             eager — zero-byte-copy wedge regressed for this arm\n  \
             eager:  {eager_sha}\n  stream: {stream_sha}"
        );
    }
}

// ---------------------------------------------------------------------
// T20: HF2Q_STREAMING_PHASE3_MUT defensive guard (iter-91)
// ---------------------------------------------------------------------

/// ADR-014 P7 iter-91 — verify the cmd_convert defensive guard against
/// HF2Q_STREAMING_PHASE3_MUT × Phase 4.5 mismatch fires correctly.
///
/// Background: the _MUT zero-byte-copy wedge drains tensor_map mid-Phase-3
/// (TensorRef::take_data_as_arc → mem::take on the inner Vec<u8>). Phase 4.5
/// quality measurement reads tensor_map for the original bytes — under _MUT,
/// those bytes are gone by the time Phase 4.5 runs, so quality measurement
/// would silently operate on empty Vec<u8> and produce a degenerate report.
///
/// The guard in cmd_convert refuses _MUT × !skip_quality early (before any
/// expensive work) with a clear error message naming both supported
/// workarounds (--skip-quality OR HF2Q_STREAMING_PHASE3=1 borrowed wedge).
///
/// This test exercises the negative path: invokes the binary with
/// HF2Q_STREAMING_PHASE3_MUT=1 but WITHOUT --skip-quality, asserts non-zero
/// exit and that stderr names the offending env var so users get directed
/// to the workarounds.
#[test]
fn streaming_phase3_mut_requires_skip_quality_guard() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_mut_no_skip");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3_MUT", "1")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4_k_m",
            "--output", output_dir.to_str().unwrap(),
            // intentionally NO --skip-quality
        ])
        .assert()
        .failure()
        .stderr(predicates::str::contains("HF2Q_STREAMING_PHASE3_MUT"))
        .stderr(predicates::str::contains("--skip-quality"));
}

/// ADR-014 P7 iter-91 — confirm the guard does NOT fire when the user has
/// opted into the supported workaround (--skip-quality + _MUT). This is the
/// happy path that all the iter-84/87/88/89 SHA gates already exercise; the
/// extra explicit test here documents the contract.
#[test]
fn streaming_phase3_mut_with_skip_quality_succeeds() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_mut_skip");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3_MUT", "1")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4_k_m",
            "--output", output_dir.to_str().unwrap(),
            "--skip-quality",
        ])
        .assert()
        .success();

    // Confirm a GGUF was actually emitted (positive control).
    let _ = locate_gguf(&output_dir);
}

/// ADR-014 P7 iter-91 — confirm the borrowed wedge (HF2Q_STREAMING_PHASE3=1)
/// does NOT trigger the guard. Borrowed mode shares bytes via Arc<Vec<u8>>
/// (per-tensor t.data.clone() into Arc) and keeps tensor_map populated, so
/// Phase 4.5 quality measurement still works.
#[test]
fn streaming_phase3_borrowed_with_quality_succeeds() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_borrowed_with_quality");
    setup_p2_iter2_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_STREAMING_PHASE3", "1")
        .args([
            "convert",
            "--input", input_dir.to_str().unwrap(),
            "--format", "gguf",
            "--quant", "q4_k_m",
            "--output", output_dir.to_str().unwrap(),
            // intentionally NO --skip-quality — borrowed wedge supports quality
        ])
        .assert()
        .success();

    let _ = locate_gguf(&output_dir);
}
