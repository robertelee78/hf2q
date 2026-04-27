//! ADR-014 P11-prereq Iter C — end-to-end integration tests for
//! `--quant dwq46/dwq48/dwq68` driven through the actual `hf2q convert`
//! binary against a tiny synthetic safetensors fixture, then read back
//! via `mlx_native::gguf::GgufFile` to assert the on-disk header type
//! code is the canonical K-quant code (not the legacy Q4_0 family).
//!
//! ## Why this crate exists
//!
//! Iter C wires `DwqKQuantizer` (Iter B's deliverable) into
//! `cmd_convert`'s Dwq46/48/68/28 dispatch arm — production
//! `--quant dwq46/48/68/28` now emits Q4_K_M-family GGUFs by default.
//! These tests close the loop by driving the CLI end-to-end (matching
//! the Iter A precedent at `tests/codec_direct_type_code.rs`), so a
//! future regression that re-routes the Dwq dispatch back through the
//! legacy `MixedBitQuantizer` (or any other Q4_0-family path) is caught
//! at the byte-on-disk surface, not just at the in-source unit-test
//! level.
//!
//! ## Coverage
//!
//! * `dwq46_emits_q4_k_m_gguf_via_cli` — `--quant dwq-4-6` against a
//!   block-aligned `LlamaForCausalLM` fixture; assert at least one
//!   tensor reports `GGML_TYPE_Q4_K` (12), NOT `GGML_TYPE_Q4_0` (2).
//! * `dwq48_emits_q4_k_base_with_q8_0_sensitive` — `--quant dwq-4-8`
//!   with a sensitive layer marked via `--sensitive-layers`; assert the
//!   base tensor is Q4_K AND the sensitive tensor is Q8_0.  Both type
//!   codes must coexist in the same GGUF — locks the per-tensor
//!   dispatch policy at the byte-on-disk layer.
//! * `dwq68_emits_q6_k_base_via_cli` — `--quant dwq-6-8`; assert at
//!   least one tensor is `GGML_TYPE_Q6_K` (14).  Locks the Q6_K base
//!   leg of `DwqKVariant::P68` end-to-end.
//! * `legacy_q4_0_dwq_path_still_available_via_env_var` —
//!   `HF2Q_USE_LEGACY_DWQ_Q4_0=1 hf2q convert --quant dwq-4-6`; assert
//!   tensors are Q4_0 (the OLD format) — proves the escape hatch works
//!   and the legacy path is genuinely byte-distinct from the new one.
//!
//! ## Fixture
//!
//! Block-aligned 256-element-row Llama-shaped synthetic safetensors —
//! mirrors `tests/cmd_convert_dispatch.rs::setup_p2_iter2_fixture` so
//! the Iter C tests share the established CFA test-fixture precedent.
//! `architectures = LlamaForCausalLM` makes
//! `DwqArch::from_hf_architecture` route to `DwqArch::Other`, which
//! exercises the weight-space DWQ path (no qwen35 capture required).
//! `hidden_size = 256 == QK_K` so each weight row is exactly one
//! K-quant super-block (144 bytes for Q4_K, 210 for Q6_K, 33 bytes/32
//! for Q8_0); short rows would fall through to F16 in the codec, which
//! would mask the type-code regression we're locking here.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use assert_cmd::Command;
use mlx_native::gguf::GgufFile;
use mlx_native::ops::quantized_matmul_ggml::GgmlType;

const QK_K: usize = 256;

/// Build a 256-element-row synthetic safetensors fixture.  Mirrors
/// `tests/cmd_convert_dispatch.rs::setup_p2_iter2_fixture` (bit-for-bit
/// identical schema + shapes); kept inline here so this test crate is
/// self-contained and the per-fixture rebuild stays cheap.
fn setup_dwq_fixture(dir: &Path) {
    fs::create_dir_all(dir).unwrap_or_else(|e| {
        panic!("create_dir_all {}: {e}", dir.display());
    });
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
    // Minimal tokenizer.json — DwqArch::Other path doesn't open it for
    // dwq-4-6/4-8/6-8 (qwen35 is the only arch that requires the
    // tokenizer for capture-spec construction).
    fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    fs::write(dir.join("tokenizer_config.json"), "{}").unwrap();
    fs::write(dir.join("model.safetensors"), build_block_aligned_safetensors()).unwrap();
}

/// Build the safetensors blob.  Five tensors: two layers' q_proj +
/// input_layernorm + embeddings.  Each weight tensor's last dim is
/// `QK_K = 256` so the K-quant codec produces real super-blocks.
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
            // Deterministic ramp F16, byte-stable across runs.
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

/// Locate the GGUF emitted by an `hf2q convert` invocation.
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

/// Collect every tensor's `(name, ggml_type)` from a GGUF.  Used to
/// build the assertion-failure diagnostics so the failing test prints
/// exactly which tensors landed which type code.
fn collect_tensor_types(gguf_path: &Path) -> Vec<(String, GgmlType)> {
    let gguf = GgufFile::open(gguf_path)
        .unwrap_or_else(|e| panic!("open {}: {e}", gguf_path.display()));
    let mut out: Vec<(String, GgmlType)> = Vec::new();
    for name in gguf.tensor_names() {
        if let Some(info) = gguf.tensor_info(name) {
            out.push((name.to_string(), info.ggml_type));
        }
    }
    out
}

/// Pretty-print the type table for failure diagnostics.
fn dump_types(table: &[(String, GgmlType)]) -> String {
    let mut sorted = table.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    sorted
        .iter()
        .map(|(n, t)| format!("  {n} → {t:?}"))
        .collect::<Vec<_>>()
        .join("\n")
}

// ────────────────────────────────────────────────────────────────────
// T1 — `--quant dwq-4-6` emits Q4_K_M-family GGUFs (NOT Q4_0)
// ────────────────────────────────────────────────────────────────────

/// Iter C lock: `--quant dwq-4-6` must produce a GGUF whose at-least-
/// one weight tensor reports `GGML_TYPE_Q4_K = 12`.  Pre-Iter C, the
/// same command emitted `GGML_TYPE_Q4_0 = 2` for every weight (the
/// legacy `MixedBitQuantizer` + `StaticQuantizer("q4")` path).  This
/// is the headline behaviour switch the iter ships.
#[test]
fn dwq46_emits_q4_k_m_gguf_via_cli() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_dwq46_q4_k_m");
    setup_dwq_fixture(&input_dir);

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

    assert!(
        stderr.contains("DwqKQuantizer dispatch")
            || stderr.contains("Q4_K_M-family default"),
        "dwq-4-6 must log the Iter C DwqKQuantizer dispatch line; stderr:\n{stderr}"
    );

    let gguf_path = locate_gguf(&output_dir);
    let table = collect_tensor_types(&gguf_path);
    assert!(
        !table.is_empty(),
        "GGUF must contain at least one tensor; got empty table from {}",
        gguf_path.display()
    );

    // At least one tensor must be Q4_K (the headline lock).
    let q4_k_count = table.iter().filter(|(_, t)| *t == GgmlType::Q4_K).count();
    let q4_0_count = table.iter().filter(|(_, t)| *t == GgmlType::Q4_0).count();
    assert!(
        q4_k_count > 0,
        "Iter C lock: --quant dwq-4-6 must emit at least one Q4_K (12) tensor \
         (NOT Q4_0 = 2); table:\n{}",
        dump_types(&table)
    );
    assert_eq!(
        q4_0_count, 0,
        "Iter C lock: --quant dwq-4-6 must NOT emit Q4_0 (2) tensors after the \
         DwqKQuantizer wiring; legacy Q4_0 emit is gated behind \
         HF2Q_USE_LEGACY_DWQ_Q4_0=1.  Table:\n{}",
        dump_types(&table)
    );
}

// ────────────────────────────────────────────────────────────────────
// T2 — `--quant dwq-4-8` emits Q4_K base + Q8_0 sensitive
// ────────────────────────────────────────────────────────────────────

/// Iter C lock: `--quant dwq-4-8` with `--sensitive-layers 1` puts
/// layer 0 in the base bucket (→ Q4_K) AND layer 1 in the sensitive
/// bucket (→ Q8_0).  Both type codes must coexist in the same GGUF —
/// proves the per-tensor `target_for(tensor_name)` dispatch survives
/// the CLI → backend → on-disk round-trip.
#[test]
fn dwq48_emits_q4_k_base_with_q8_0_sensitive() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_dwq48_mixed");
    setup_dwq_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "dwq-4-8",
            // Layer 1 sensitive (singleton); layer 0 stays base.  The
            // synthetic fixture is `DwqArch::Other` so the calibrator
            // returns no derived ranges → the CLI flag is honoured
            // bit-for-bit (final_sensitive_ranges = config.sensitive_layers).
            "--sensitive-layers",
            "1",
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
            "-vv",
        ])
        .assert()
        .success();

    let gguf_path = locate_gguf(&output_dir);
    let table = collect_tensor_types(&gguf_path);

    // Base bucket (layer 0 weight) must be Q4_K; sensitive bucket
    // (layer 1 weight) must be Q8_0.  Lookup by gguf-renamed tensor
    // name pattern (`blk.<N>.attn_q.weight`).
    let base_layer0 = table
        .iter()
        .find(|(n, _)| n.starts_with("blk.0.attn_q"))
        .unwrap_or_else(|| {
            panic!(
                "expected blk.0.attn_q.weight tensor in GGUF; table:\n{}",
                dump_types(&table)
            )
        });
    let sensitive_layer1 = table
        .iter()
        .find(|(n, _)| n.starts_with("blk.1.attn_q"))
        .unwrap_or_else(|| {
            panic!(
                "expected blk.1.attn_q.weight tensor in GGUF; table:\n{}",
                dump_types(&table)
            )
        });

    assert_eq!(
        base_layer0.1,
        GgmlType::Q4_K,
        "Iter C: dwq-4-8 base-bucket (layer 0, NOT in sensitive set) must be \
         Q4_K; got {:?} for {}.  Full table:\n{}",
        base_layer0.1,
        base_layer0.0,
        dump_types(&table)
    );
    assert_eq!(
        sensitive_layer1.1,
        GgmlType::Q8_0,
        "Iter C: dwq-4-8 sensitive-bucket (layer 1, in --sensitive-layers 1) \
         must be Q8_0; got {:?} for {}.  Full table:\n{}",
        sensitive_layer1.1,
        sensitive_layer1.0,
        dump_types(&table)
    );
    // No tensor should be Q4_0 — that would mean the legacy path leaked.
    let q4_0_count = table.iter().filter(|(_, t)| *t == GgmlType::Q4_0).count();
    assert_eq!(
        q4_0_count, 0,
        "Iter C: dwq-4-8 must NOT emit Q4_0 tensors; table:\n{}",
        dump_types(&table)
    );
}

// ────────────────────────────────────────────────────────────────────
// T3 — `--quant dwq-6-8` emits Q6_K base
// ────────────────────────────────────────────────────────────────────

/// Iter C lock: `--quant dwq-6-8` produces at least one
/// `GGML_TYPE_Q6_K = 14` tensor (the base bucket for `DwqKVariant::P68`).
/// Locks the Q6_K base leg end-to-end so a regression that drops
/// `KQuantTarget::Q6K` from the variant table is caught at the
/// byte-on-disk surface.
#[test]
fn dwq68_emits_q6_k_base_via_cli() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_dwq68_q6_k");
    setup_dwq_fixture(&input_dir);

    Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .args([
            "convert",
            "--input",
            input_dir.to_str().unwrap(),
            "--format",
            "gguf",
            "--quant",
            "dwq-6-8",
            "--output",
            output_dir.to_str().unwrap(),
            "--skip-quality",
            "-vv",
        ])
        .assert()
        .success();

    let gguf_path = locate_gguf(&output_dir);
    let table = collect_tensor_types(&gguf_path);
    let q6_k_count = table.iter().filter(|(_, t)| *t == GgmlType::Q6_K).count();
    let q4_0_count = table.iter().filter(|(_, t)| *t == GgmlType::Q4_0).count();
    assert!(
        q6_k_count > 0,
        "Iter C lock: --quant dwq-6-8 must emit at least one Q6_K (14) tensor; \
         table:\n{}",
        dump_types(&table)
    );
    assert_eq!(
        q4_0_count, 0,
        "Iter C lock: --quant dwq-6-8 must NOT emit Q4_0 (2) tensors after the \
         DwqKQuantizer wiring.  Table:\n{}",
        dump_types(&table)
    );
}

// ────────────────────────────────────────────────────────────────────
// T4 — `HF2Q_USE_LEGACY_DWQ_Q4_0=1` re-activates the legacy path
// ────────────────────────────────────────────────────────────────────

/// Iter C escape-hatch lock: `HF2Q_USE_LEGACY_DWQ_Q4_0=1` MUST keep
/// the pre-Iter-C `MixedBitQuantizer` + `StaticQuantizer("q4")` path
/// reachable for benchmarking comparison.  Asserting that the env-
/// gated invocation produces Q4_0 (NOT Q4_K) proves three things at
/// once:
///   1. the env-var switch is wired into the dispatch arm and
///      effective at runtime,
///   2. the legacy code path was preserved in-place (no silent drift
///      via the iter's diff),
///   3. the new default path is genuinely byte-distinct from the
///      legacy path (otherwise this test would pass even if the new
///      path silently produced Q4_0 too — guarding against a wiring
///      regression in T1).
#[test]
fn legacy_q4_0_dwq_path_still_available_via_env_var() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let input_dir = tmp.path().join("input");
    let output_dir = tmp.path().join("out_dwq46_legacy");
    setup_dwq_fixture(&input_dir);

    let assertion = Command::cargo_bin("hf2q")
        .expect("hf2q binary")
        .env("HF2Q_USE_LEGACY_DWQ_Q4_0", "1")
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

    // Legacy path emits a distinct INFO breadcrumb.
    assert!(
        stderr.contains("LEGACY MixedBitQuantizer")
            || stderr.contains("HF2Q_USE_LEGACY_DWQ_Q4_0"),
        "legacy path must log the LEGACY breadcrumb when HF2Q_USE_LEGACY_DWQ_Q4_0=1; \
         stderr:\n{stderr}"
    );

    let gguf_path = locate_gguf(&output_dir);
    let table = collect_tensor_types(&gguf_path);
    let q4_0_count = table.iter().filter(|(_, t)| *t == GgmlType::Q4_0).count();
    let q4_k_count = table.iter().filter(|(_, t)| *t == GgmlType::Q4_K).count();
    let q6_k_count = table.iter().filter(|(_, t)| *t == GgmlType::Q6_K).count();
    assert!(
        q4_0_count > 0,
        "Iter C escape hatch: HF2Q_USE_LEGACY_DWQ_Q4_0=1 + --quant dwq-4-6 must \
         emit at least one Q4_0 (2) tensor; table:\n{}",
        dump_types(&table)
    );
    assert_eq!(
        q4_k_count, 0,
        "Iter C escape hatch: legacy path must NOT emit Q4_K (12) tensors \
         (would mean the env switch was ignored); table:\n{}",
        dump_types(&table)
    );
    assert_eq!(
        q6_k_count, 0,
        "Iter C escape hatch: legacy path must NOT emit Q6_K (14) tensors; \
         table:\n{}",
        dump_types(&table)
    );
}

// ────────────────────────────────────────────────────────────────────
// Defensive: helper sanity tests for the in-test fixture + GGUF
// reader.  These run instantly + with no external deps so a CI
// regression in `mlx_native::gguf` or our fixture builder surfaces
// directly without needing the slow `--release` build.
// ────────────────────────────────────────────────────────────────────

#[test]
fn fixture_safetensors_blob_is_byte_stable() {
    // Two builds back-to-back must be byte-identical (deterministic
    // ramp).  Catches a regression where the fixture builder grows
    // hidden non-determinism (e.g. HashMap iteration in header_map
    // — currently BTreeMap, this test locks that contract).
    let a = build_block_aligned_safetensors();
    let b = build_block_aligned_safetensors();
    assert_eq!(
        a, b,
        "fixture safetensors blob must be byte-identical across rebuilds"
    );
    assert!(
        a.len() > 8,
        "safetensors blob too short ({} bytes); fixture builder bug",
        a.len()
    );
}

#[test]
fn ggml_type_enum_variants_referenced_by_iter_c_exist() {
    // Spot-check that every `GgmlType` variant referenced by the Iter C
    // assertions is constructible by name.  Coupling against the
    // mlx_native enum (rather than raw numeric codes) means the assertion
    // surface stays valid even if the on-disk codec table renumbers
    // (mlx_native owns the mapping).  Mirrors the lock pattern in
    // `tests/codec_direct_type_code.rs` (Iter A precedent).
    let _ = GgmlType::Q4_0;
    let _ = GgmlType::Q8_0;
    let _ = GgmlType::Q4_K;
    let _ = GgmlType::Q6_K;
}
