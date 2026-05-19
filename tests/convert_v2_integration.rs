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
//! that test calls `HfModelSource::open` + `llama3::map_tensor_name` +
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

/// Drop a minimal `tokenizer.json` + `tokenizer_config.json` alongside
/// every fixture so the convert-v2 driver's tokenizer-metadata
/// emitter has something to read. The driver hard-fails when
/// tokenizer.json is missing per ADR-033's no-silent-skip contract
/// (see `convert::tokenizer::TokenizerError::TokenizerJsonMissing`);
/// the test fixture must therefore mirror what a real HF repo ships.
///
/// `vocab_size` MUST match `config.json::vocab_size` /
/// `text_config::vocab_size` — the emitter cross-checks both before
/// writing the GGUF KV section.
///
/// Returns `(bos_id, eos_id)` so test assertions can pin them
/// against the recovered metadata.
fn write_minimal_tokenizer_fixture(dir: &Path, vocab_size: usize) -> (u32, u32) {
    assert!(
        vocab_size >= 16,
        "minimal tokenizer needs at least 16 ids for the 4 special tokens"
    );
    // Base BPE: ids 0..(vocab_size-4). Use a deterministic synthetic
    // pattern so the emitter sees a well-formed vocab map.
    let base_count = vocab_size - 4;
    let mut vocab = serde_json::Map::with_capacity(base_count);
    for i in 0..base_count {
        vocab.insert(format!("tok{i}"), serde_json::json!(i as u64));
    }
    let bos_id = (vocab_size - 4) as u64;
    let eos_id = (vocab_size - 3) as u64;
    let pad_id = (vocab_size - 2) as u64;
    let unk_id = (vocab_size - 1) as u64;
    let tokenizer_json = serde_json::json!({
        "model": {
            "type": "BPE",
            "byte_fallback": true,
            "vocab": vocab,
            "merges": []
        },
        "added_tokens": [
            {"id": bos_id, "content": "<bos>", "special": true},
            {"id": eos_id, "content": "<eos>", "special": true},
            {"id": pad_id, "content": "<pad>", "special": true},
            {"id": unk_id, "content": "<unk>", "special": true},
        ]
    });
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string_pretty(&tokenizer_json).unwrap(),
    )
    .unwrap();

    let tokenizer_config = serde_json::json!({
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "add_bos_token": true,
        "add_eos_token": false,
    });
    fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string_pretty(&tokenizer_config).unwrap(),
    )
    .unwrap();

    (bos_id as u32, eos_id as u32)
}

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

    // ADR-033 tokenizer wiring: every fixture must drop a tokenizer.json
    // so the convert-v2 driver's tokenizer-metadata emitter can produce
    // the GGUF KV pairs llama.cpp's vocab loader requires.
    write_minimal_tokenizer_fixture(dir, VOCAB);
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

    // Metadata count: Llama3's build_metadata emits 11 KV pairs + the
    // ADR-033 tokenizer-metadata emitter adds 11 (model + tokens +
    // scores + token_type + bos/eos/unk/pad + add_bos_token +
    // add_space_prefix + pre; merges array is empty in this fixture so
    // the KV is skipped; no chat_template for llama).
    assert_eq!(gguf.metadata_count(), 11 + 11);
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

// ----------------------------------------------------------------------------
// Apex tier end-to-end (P4a — ADR-033 §"Plan / Pa")
// ----------------------------------------------------------------------------

/// Synthesize a tiny Qwen3MoE-shaped HF safetensors directory for Apex
/// tier testing.
///
/// Shape choices:
/// - `hidden_size`           = 256  (256-aligned innermost for K-quants)
/// - `moe_intermediate_size` = 256
/// - `vocab_size`            = 256
/// - `num_experts`           = 4
/// - `num_experts_per_tok`   = 2
/// - `num_hidden_layers`     = 2  (every layer lands in EDGE region for
///                                 EXP/SHARED/ATTN partitions, which
///                                 makes the expected ggml_type
///                                 assertions arch-independent of the
///                                 NEAR/MID branches)
///
/// Tensors emitted (norms + router-gate omitted on purpose — see below):
/// - `model.embed_tokens.weight`                  `[256, 256]`  → token_embd → Q6_K
/// - `lm_head.weight`                             `[256, 256]`  → output → Q6_K
/// - per-layer × 2:
///   - `self_attn.{q,k,v,o}_proj.weight`          `[256, 256]`  → attn_* (EDGE) → Q6_K
///   - `mlp.experts.<E>.{gate,up,down}_proj`      `[256, 256]`  → fused exps (EDGE) → Q6_K
///
/// Total: 2 globals + 4 attn + 3 expert-groups per layer × 2 layers
/// = 2 + (4 + 3) × 2 = 16 GGUF tensors.
///
/// **Norms omitted**: ApexPolicy routes Norm tensors to F32, and
/// `quantizer_for(F32)` returns `NoQuantizerForType` (the same caveat
/// noted in `synthesize_tiny_llama3_no_norms` above).
///
/// **Router gate omitted**: `mlp.gate.weight` maps to
/// `blk.<i>.ffn_gate_inp.weight` which ApexPolicy routes to Q5_0.
/// mlx-native's GgufFile reader does NOT recognize the Q5_0 wire code
/// (only Q4_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, F32, F16, I16, IQ4_NL).
/// Including the router gate would cause the reader to error at parse
/// time — orthogonal to what this test asserts (ApexPolicy correctly
/// dispatches Q6_K for the EDGE-region tensors). Coverage for the
/// router gate's Q5_0 pick lives in the ApexPolicy unit tests at
/// `src/quantize/ggml_quants/apex/policy.rs::apex_policy_router_gate_q5_0`.
fn synthesize_tiny_qwen35moe_for_apex(dir: &Path) {
    const HIDDEN: usize = 256;
    const MOE_FFN: usize = 256;
    const VOCAB: usize = 256;
    const LAYERS: usize = 2;
    const N_EXPERTS: usize = 4;

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
        mk_f32_bytes(VOCAB * HIDDEN, 2),
    ));

    // Per-layer attention + per-expert FFN slices.
    for li in 0..LAYERS {
        let s = (li as u32) * 1000;

        // Attention projections (4 tensors per layer).
        for (idx, suffix) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
            tensors.push((
                format!("model.layers.{li}.self_attn.{suffix}.weight"),
                vec![HIDDEN, HIDDEN],
                mk_f32_bytes(HIDDEN * HIDDEN, s + 10 + idx as u32),
            ));
        }

        // Per-expert FFN tensors (each expert is `[ffn, hidden]` for
        // gate/up and `[hidden, ffn]` for down). The convert driver
        // fuses the per-expert slices into a single 3-D GGUF tensor
        // `[innermost, outermost, n_experts]` — we feed the raw slices
        // here and let the driver do the fusion.
        for expert in 0..N_EXPERTS {
            // gate_proj: [ffn, hidden] (HF order)
            tensors.push((
                format!("model.layers.{li}.mlp.experts.{expert}.gate_proj.weight"),
                vec![MOE_FFN, HIDDEN],
                mk_f32_bytes(MOE_FFN * HIDDEN, s + 100 + expert as u32),
            ));
            // up_proj: [ffn, hidden]
            tensors.push((
                format!("model.layers.{li}.mlp.experts.{expert}.up_proj.weight"),
                vec![MOE_FFN, HIDDEN],
                mk_f32_bytes(MOE_FFN * HIDDEN, s + 200 + expert as u32),
            ));
            // down_proj: [hidden, ffn]
            tensors.push((
                format!("model.layers.{li}.mlp.experts.{expert}.down_proj.weight"),
                vec![HIDDEN, MOE_FFN],
                mk_f32_bytes(HIDDEN * MOE_FFN, s + 300 + expert as u32),
            ));
        }
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

    // Minimal Qwen3MoE config.json. All required keys per
    // `qwen35moe::build_metadata` are present.
    let cfg = serde_json::json!({
        "_name_or_path": "synthetic/Qwen3-MoE-Tiny-Apex-Test",
        "model_type": "qwen3_moe",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN, // unused for MoE per the mapper
        "moe_intermediate_size": MOE_FFN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "num_experts": N_EXPERTS,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1.0e-6,
        "rope_theta": 1000000.0,
        "vocab_size": VOCAB,
    });
    fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .expect("write config.json");

    // ADR-033 tokenizer wiring: drop a minimal tokenizer.json so the
    // convert-v2 driver's tokenizer-metadata emitter has input.
    write_minimal_tokenizer_fixture(dir, VOCAB);
}

/// Drive `hf2q convert-v2 --quant apex-balanced` end-to-end on a synthetic
/// tiny Qwen3MoE fixture and assert the output GGUF round-trips with the
/// correct per-tensor ggml_types per `tier_rules(Balanced)`.
///
/// At 2 layers, every layer lands in the EDGE region for EXP / SHARED /
/// ATTN (all three partitions' edge band covers `i ∈ [0, 1]`):
/// - token_embd                        → Q6_K (hard-coded)
/// - output                            → Q6_K (hard-coded)
/// - attn_{q,k,v,output} (EDGE)        → balanced.edge_attn = Q6_K
/// - ffn_{gate,up,down}_exps (EDGE)    → balanced.edge_exp  = Q6_K
///
/// All 16 expected tensors → Q6_K (mlx-native positional code 6).
///
/// Acceptance:
/// - exit code 0 (CLI parses `apex-balanced`, builds ApexPolicy,
///   orchestrator routes through it)
/// - 16 tensors emitted (2 globals + 4 attn × 2 layers + 3 expert-groups × 2 layers)
/// - `general.file_type` byte = approximate_for_apex(Balanced) =
///   MostlyQ5_K_M = 17 (cosmetic header pick; tensors themselves are Q6_K)
/// - every tensor's recorded ggml_type is Q6_K (positional code 6)
///
/// Per [[feedback-no-loop-suppression-2026-05-17]]: the apex path must
/// reject genuinely-unsupported configs with a typed error, never silent
/// F16 demotion. Verified separately by ApexPolicy's
/// `apex_policy_unsupported_arch_errors` and
/// `apex_policy_dense_model_errors` unit tests; this integration test
/// covers the happy-path wiring only.
#[test]
fn convert_v2_apex_balanced_tiny_qwen35moe_round_trip() {
    let model_dir = tempfile::tempdir().unwrap();
    synthesize_tiny_qwen35moe_for_apex(model_dir.path());

    let out = tempfile::NamedTempFile::new().unwrap();
    Command::cargo_bin("hf2q")
        .unwrap()
        .arg("convert-v2")
        .arg(model_dir.path())
        .arg("--quant")
        .arg("apex-balanced")
        .arg("-o")
        .arg(out.path())
        .assert()
        .success();

    let gguf = mlx_native::gguf::GgufFile::open(out.path()).expect("parse output GGUF");

    // Tensor count: 2 globals (token_embd, output) + per-layer
    //   (4 attn projections + 3 fused expert tensors) × 2 layers
    //   = 2 + 7 × 2 = 16.
    assert_eq!(
        gguf.tensor_count(),
        16,
        "expected 16 tensors (2 globals + 7 per-layer × 2 layers)"
    );

    // `general.architecture` = "qwen3moe" per qwen35moe::build_metadata.
    assert_eq!(gguf.metadata_string("general.architecture"), Some("qwen3moe"));

    // `general.file_type` = approximate_for_apex(Balanced) = MostlyQ5_K_M = 17.
    // Apex tiers map to the CLOSEST standard ftype for the header byte;
    // per-tensor ggml_types come from ApexPolicy::target_for and are
    // asserted independently below.
    assert_eq!(
        gguf.metadata_u32("general.file_type"),
        Some(17),
        "Balanced tier's approximate LlamaFtype is MostlyQ5_K_M = 17"
    );

    // Expected tensor names. ApexPolicy at 2 layers + Balanced tier
    // routes EVERY tensor below to Q6_K (EDGE region for all three
    // partitions; token_embd & output are hard-coded Q6_K regardless
    // of tier).
    let expected_names: &[&str] = &[
        "token_embd.weight",
        "output.weight",
        // layer 0
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        // layer 1
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
        "blk.1.ffn_gate_exps.weight",
        "blk.1.ffn_up_exps.weight",
        "blk.1.ffn_down_exps.weight",
    ];

    // mlx_native's GgmlType is positional: F32=0, F16=1, Q4_0=2, Q8_0=3,
    // Q4_K=4, Q5_K=5, Q6_K=6. We assert every tensor lands on Q6_K (6).
    for name in expected_names {
        let info = gguf
            .tensor_info(name)
            .unwrap_or_else(|| panic!("missing GGUF tensor `{name}`"));
        assert_eq!(
            info.ggml_type as u32, 6,
            "tensor `{name}`: expected Q6_K (positional 6) per Apex Balanced \
             EDGE-region rule, got {}",
            info.ggml_type as u32
        );
        // ALIGNMENT round-trip — every offset must be 32-aligned per spec.
        assert_eq!(info.offset % 32, 0, "tensor `{name}` offset not aligned");
    }
}

// ----------------------------------------------------------------------------
// Gemma 4 real-arch round-trip
// ----------------------------------------------------------------------------

/// Synthesize a tiny Gemma 4-shaped HF safetensors directory matching
/// the REAL google-gemma-4-26b-a4b-it tensor layout:
///   - multimodal wrapper (`model.language_model.*` prefix on every
///     text-decoder tensor)
///   - pre-fused MoE experts (`experts.gate_up_proj` 3-D + `experts.down_proj`
///     3-D)
///   - parallel dense FFN (`mlp.{gate,up,down}_proj`) alongside the
///     routed experts
///   - router projection (`router.proj.weight`)
///   - one off-path vision-tower tensor (verifies the
///     `MappedTensor::Drop` path silently discards rather than erroring)
///
/// Shape choices (all Q8_0 block-aligned at block=32):
///   - hidden_size = 32, moe_intermediate_size = 32, intermediate_size = 32
///   - num_experts = 4
///   - num_hidden_layers = 2
///   - vocab_size = 64
///
/// Norms / `layer_scalar` / `router.scale` / `router.per_expert_scale`
/// are intentionally omitted — they're 1-D / scalar tensors that the
/// orchestrator's StandardPolicy routes to F32, which has no quantizer
/// (the same constraint the Llama-3 and Qwen3MoE integration tests
/// document). The mapper's handling of those names is exercised by the
/// unit tests in `src/convert/arch/gemma4.rs`.
fn synthesize_tiny_gemma4_real_arch(dir: &Path) {
    const HIDDEN: usize = 32;
    const MOE_FFN: usize = 32;
    const DENSE_FFN: usize = 32;
    const VOCAB: usize = 64;
    const LAYERS: usize = 2;
    const N_EXPERTS: usize = 4;

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

    // ---- Globals (under language_model. wrapper) -------------------------
    tensors.push((
        "model.language_model.embed_tokens.weight".into(),
        vec![VOCAB, HIDDEN],
        mk_f32_bytes(VOCAB * HIDDEN, 1),
    ));

    // ---- One off-path vision-tower tensor: verifies Drop --------------
    // Per the gemma4 mapper, anything matching `model.vision_tower.` /
    // `model.embed_vision.` / etc. returns `MappedTensor::Drop` and the
    // driver silently discards it. If the driver instead routed it
    // through `UnmappedTensor`, the convert run would fail — that
    // failure would surface here as a non-zero exit code.
    tensors.push((
        "model.vision_tower.patch_embedder.input_proj.weight".into(),
        vec![HIDDEN, HIDDEN],
        mk_f32_bytes(HIDDEN * HIDDEN, 999),
    ));

    // ---- Per-block -------------------------------------------------------
    for li in 0..LAYERS {
        let s = (li as u32) * 1000;

        // Attention Q/K/V/O.
        for (idx, suffix) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
            tensors.push((
                format!("model.language_model.layers.{li}.self_attn.{suffix}.weight"),
                vec![HIDDEN, HIDDEN],
                mk_f32_bytes(HIDDEN * HIDDEN, s + 10 + idx as u32),
            ));
        }

        // Parallel dense FFN (Gemma 4 has BOTH this AND experts).
        for (idx, suffix) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            // gate/up: [ffn, hidden]; down: [hidden, ffn].
            let py_shape = match *suffix {
                "down_proj" => vec![HIDDEN, DENSE_FFN],
                _ => vec![DENSE_FFN, HIDDEN],
            };
            let numel: usize = py_shape.iter().product();
            tensors.push((
                format!("model.language_model.layers.{li}.mlp.{suffix}.weight"),
                py_shape,
                mk_f32_bytes(numel, s + 100 + idx as u32),
            ));
        }

        // Pre-fused MoE experts. `gate_up_proj` is gate+up concatenated
        // along axis 1 → HF shape `[n_experts, 2*moe_ffn, hidden]`.
        // `down_proj` is the standard per-expert down → HF shape
        // `[n_experts, hidden, moe_ffn]`.
        tensors.push((
            format!("model.language_model.layers.{li}.experts.gate_up_proj"),
            vec![N_EXPERTS, 2 * MOE_FFN, HIDDEN],
            mk_f32_bytes(N_EXPERTS * 2 * MOE_FFN * HIDDEN, s + 200),
        ));
        tensors.push((
            format!("model.language_model.layers.{li}.experts.down_proj"),
            vec![N_EXPERTS, HIDDEN, MOE_FFN],
            mk_f32_bytes(N_EXPERTS * HIDDEN * MOE_FFN, s + 201),
        ));

        // Router projection — `[n_experts, hidden]`.
        tensors.push((
            format!("model.language_model.layers.{li}.router.proj.weight"),
            vec![N_EXPERTS, HIDDEN],
            mk_f32_bytes(N_EXPERTS * HIDDEN, s + 300),
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

    // Minimal Gemma 4 config matching the real google-gemma-4-26b-a4b-it
    // shape (multimodal wrapper + nested text_config). The driver's
    // `effective_config` flattens `text_config` so the per-arch mapper
    // sees the inner text-decoder fields directly.
    let cfg = serde_json::json!({
        "_name_or_path": "synthetic/Gemma-4-Tiny-Real-Arch-Test",
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": HIDDEN,
            "intermediate_size": DENSE_FFN,
            "moe_intermediate_size": MOE_FFN,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1.0e-6,
            "sliding_window": 1024,
            "num_experts": N_EXPERTS,
            "top_k_experts": 2,
            "vocab_size": VOCAB,
            // ---- Gemma 4-specific required hparams (gemma.py:659-666) ----
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_double_wide_mlp": false,
            "rope_parameters": {
                "full_attention": {
                    "rope_theta": 1_000_000.0,
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                }
            }
        },
    });
    fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .expect("write config.json");

    // ADR-033 tokenizer wiring: drop a minimal tokenizer.json so the
    // convert-v2 driver's tokenizer-metadata emitter has input. Vocab
    // size MUST match text_config.vocab_size above (VOCAB).
    write_minimal_tokenizer_fixture(dir, VOCAB);
}

/// Drive `hf2q convert-v2 --quant q8_0` end-to-end on the synthetic
/// real-arch Gemma 4 fixture and assert the output GGUF round-trips
/// with every expected tensor name + ggml_type.
///
/// Acceptance:
///  - exit code 0 (the prior gemma3-shape mapper failed with
///    `UnmappedTensor` on `experts.gate_up_proj`,
///    `post_feedforward_layernorm_2`, etc.; this test locks in the
///    real-arch port)
///  - 21 tensors emitted (1 global + 10 per-layer × 2 layers; the off-path
///    vision-tower tensor is silently dropped via `MappedTensor::Drop`)
///  - `general.architecture = "gemma4"` (NOT `"gemma3"` — the prior
///    mapper got this wrong; locked in per `gemma4.rs` quirk #8)
///  - every expected GGUF tensor name present with Q8_0 (positional 3)
///  - the fused expert tensors land as 3-D with the right inner dim
#[test]
fn convert_v2_gemma4_real_arch_round_trip() {
    let model_dir = tempfile::tempdir().unwrap();
    synthesize_tiny_gemma4_real_arch(model_dir.path());

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

    // Tensor count: 1 global (embed) + per-layer (4 attn + 3 mlp + 2 experts
    // + 1 router) × 2 layers + 1 synthesized rope_freqs.weight = 22. The
    // vision-tower tensor is dropped silently by the gemma4 mapper.
    assert_eq!(
        gguf.tensor_count(),
        22,
        "expected 22 tensors (1 embed + 10 per-layer × 2 + 1 rope_freqs; vision dropped)"
    );

    // Architecture must be `gemma4`, NOT `gemma3` (the prior mapper got
    // this wrong; see gemma4.rs quirk #8).
    assert_eq!(
        gguf.metadata_string("general.architecture"),
        Some("gemma4"),
        "Gemma 4 emits general.architecture=gemma4 (LLM_ARCH_GEMMA4)"
    );

    // Required Gemma 4 KV pairs — all under the `gemma4.*` prefix.
    assert_eq!(gguf.metadata_u32("gemma4.embedding_length"), Some(32));
    assert_eq!(gguf.metadata_u32("gemma4.block_count"), Some(2));
    assert_eq!(gguf.metadata_u32("gemma4.expert_count"), Some(4));
    assert_eq!(gguf.metadata_u32("gemma4.expert_used_count"), Some(2));
    assert_eq!(
        gguf.metadata_u32("gemma4.expert_feed_forward_length"),
        Some(32)
    );

    // ---- New Gemma 4-specific KV pairs (gemma.py:659-700) ----
    assert_eq!(
        gguf.metadata_u32("gemma4.attention.shared_kv_layers"),
        Some(0),
        "shared_kv_layers from num_kv_shared_layers (gemma.py:660)"
    );
    assert_eq!(
        gguf.metadata_u32("gemma4.embedding_length_per_layer_input"),
        Some(0),
        "embedding_length_per_layer_input default 0 (gemma.py:663)"
    );
    // sliding_window_pattern is array-of-bool, len=block_count=2.
    use mlx_native::gguf::MetadataValue;
    let swa_meta = gguf
        .metadata("gemma4.attention.sliding_window_pattern")
        .expect("sliding_window_pattern present");
    let swa_arr = match swa_meta {
        MetadataValue::Array(v) => v,
        other => panic!("expected Array for sliding_window_pattern, got {other:?}"),
    };
    assert_eq!(swa_arr.len(), 2, "swa pattern array length = block_count");
    let bools: Vec<bool> = swa_arr
        .iter()
        .map(|v| match v {
            MetadataValue::Bool(b) => *b,
            other => panic!("swa pattern element not Bool: {other:?}"),
        })
        .collect();
    assert_eq!(
        bools,
        vec![true, false],
        "layer 0 sliding, layer 1 full per fixture layer_types"
    );
    // RoPE dims: global = global_head_dim = 8; swa = head_dim * 1.0 = 8.
    assert_eq!(gguf.metadata_u32("gemma4.rope.dimension_count"), Some(8));
    assert_eq!(gguf.metadata_u32("gemma4.rope.dimension_count_swa"), Some(8));
    // head_count_kv stays scalar (no num_global_key_value_heads in fixture).
    assert_eq!(gguf.metadata_u32("gemma4.attention.head_count_kv"), Some(4));
    // feed_forward_length stays scalar (use_double_wide_mlp=false).
    assert_eq!(gguf.metadata_u32("gemma4.feed_forward_length"), Some(32));

    // file_type round-trips as Q8_0 (= 7 per LlamaFtype::MostlyQ8_0).
    assert_eq!(gguf.metadata_u32("general.file_type"), Some(7));

    // ---- Synthesized rope_freqs.weight tensor (gemma.py:702-718) ----
    let rope_freqs = gguf
        .tensor_info("rope_freqs.weight")
        .expect("rope_freqs.weight present (synthesized by build_synthesized_tensors)");
    // F32 = positional 0 in mlx_native's GgmlType enum.
    assert_eq!(
        rope_freqs.ggml_type as u32, 0,
        "rope_freqs.weight must be F32 (positional 0), got {}",
        rope_freqs.ggml_type as u32
    );
    // Shape: 1-D of length global_head_dim/2 = 8/2 = 4.
    assert_eq!(
        rope_freqs.shape,
        vec![4],
        "rope_freqs.weight shape = [global_head_dim/2]"
    );
    // byte_len: 4 elements × 4 bytes/F32 = 16.
    assert_eq!(rope_freqs.byte_len, 16);

    // ---- On-disk PAYLOAD assertion (codex re-review 2026-05-18 §3) ----
    //
    // gemma.py:713-715 specifies the synthesized values exactly:
    //   n_rot_full = int(global_head_dim * partial_rotary_factor_full / 2)
    //              = int(8 * 0.25 / 2) = 1
    //   table_len  = global_head_dim / 2 = 4
    //   values     = [1.0]*n_rot_full + [1e30]*(table_len - n_rot_full)
    //              = [1.0, 1e30, 1e30, 1e30]
    //
    // We read the raw 16 bytes at `tensor_data_offset + info.offset` and
    // parse them as 4 little-endian f32s. The presence of 1e30 is the
    // load-bearing semantic — it tells the proportional-rope path to
    // collapse the unrotated dims via `freq * 1e30 ≈ +inf`. Any silent
    // truncation by the writer / dequantizer would show up here.
    use std::io::{Read, Seek, SeekFrom};
    let abs_offset = gguf.tensor_data_offset() + rope_freqs.offset;
    let mut f = std::fs::File::open(out.path()).expect("re-open output gguf");
    f.seek(SeekFrom::Start(abs_offset))
        .expect("seek to rope_freqs payload");
    let mut bytes = [0u8; 16];
    f.read_exact(&mut bytes)
        .expect("read 16 bytes of rope_freqs payload");
    let payload: [f32; 4] = [
        f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
        f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        f32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        f32::from_le_bytes(bytes[12..16].try_into().unwrap()),
    ];
    assert_eq!(
        payload,
        [1.0_f32, 1.0e30_f32, 1.0e30_f32, 1.0e30_f32],
        "rope_freqs.weight payload must be exactly [1.0, 1e30, 1e30, 1e30] \
         (gemma.py:713-715: n_rot_full=1, n_unrot_full=3 for global_head_dim=8, prf=0.25)"
    );

    // Per-tensor name + ggml_type. Q8_0 = positional 3.
    let expected_names: &[&str] = &[
        "token_embd.weight",
        // layer 0
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.0.ffn_gate_inp.weight",
        // layer 1
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_up.weight",
        "blk.1.ffn_down.weight",
        "blk.1.ffn_gate_up_exps.weight",
        "blk.1.ffn_down_exps.weight",
        "blk.1.ffn_gate_inp.weight",
    ];
    for name in expected_names {
        let info = gguf
            .tensor_info(name)
            .unwrap_or_else(|| panic!("missing GGUF tensor `{name}`"));
        // `ffn_gate_inp.weight` (router-gate projection) is F32-keep per
        // /opt/llama.cpp/src/llama-quant.cpp:307. Every other tensor here
        // is a regular 2-D weight that lands on Q8_0 (positional code 3).
        let expected_ggml_type: u32 = if name.contains("ffn_gate_inp.weight") {
            0 // F32
        } else {
            3 // Q8_0
        };
        assert_eq!(
            info.ggml_type as u32, expected_ggml_type,
            "tensor `{name}` expected ggml_type {}, got {}",
            expected_ggml_type, info.ggml_type as u32
        );
        assert_eq!(info.offset % 32, 0, "tensor `{name}` offset not aligned");
    }

    // The vision-tower tensor MUST be absent from the text-decoder GGUF
    // (silently dropped by the gemma4 mapper's `Drop` variant).
    assert!(
        gguf.tensor_info("v.patch_embedder.input_proj.weight")
            .is_none(),
        "vision-tower tensor must be absent from the text-decoder GGUF"
    );

    // Fused expert tensors land as 3-D with the right inner dim (hidden).
    // GGUF shape order is innermost-first, so `blk.0.ffn_gate_up_exps`
    // has shape `[hidden=32, 2*moe_ffn=64, n_experts=4]`.
    let exps = gguf
        .tensor_info("blk.0.ffn_gate_up_exps.weight")
        .expect("ffn_gate_up_exps present");
    assert_eq!(
        exps.shape.len(),
        3,
        "ffn_gate_up_exps must be 3-D (got {:?})",
        exps.shape
    );
    // The mlx-native reader reverses GGUF's innermost-first storage back
    // to outermost-first `[rows, cols]` for downstream Candle-style
    // loaders (see /opt/mlx-native/src/gguf/mod.rs:1005-1008). So the
    // user-facing shape lines up with the original HF `[n_experts,
    // 2*moe_ffn, hidden]` orientation, even though the on-disk GGUF
    // bytes encode `[hidden, 2*moe_ffn, n_experts]` innermost-first.
    assert_eq!(
        exps.shape, vec![4_usize, 64, 32],
        "ffn_gate_up_exps shape mismatch — expected reader-orientation \
         [n_experts=4, 2*moe_ffn=64, hidden=32]"
    );

    let down = gguf
        .tensor_info("blk.0.ffn_down_exps.weight")
        .expect("ffn_down_exps present");
    // HF `[n_experts=4, hidden=32, moe_ffn=32]` round-trips to the same
    // shape via the reader's reverse-on-load.
    assert_eq!(
        down.shape,
        vec![4_usize, 32, 32],
        "ffn_down_exps shape mismatch"
    );

    // ---- Tokenizer metadata (ADR-033 P3 tokenizer wiring) ----
    // The convert-v2 driver now emits the `tokenizer.*` KV block per
    // gemma.py:625-653 (`Gemma4Model.set_vocab`). Real-model load on
    // /opt/hf2q/models/google-gemma-4-26b-a4b-it failed pre-port with
    // llama.cpp's "error loading model vocabulary: key not found in
    // model: tokenizer.ggml.model"; pin every emitted key here.
    assert_eq!(
        gguf.metadata_string("tokenizer.ggml.model"),
        Some("gemma4"),
        "Gemma 4 tokenizer.ggml.model = `gemma4` per gemma.py:649"
    );
    let tokens_meta = gguf
        .metadata("tokenizer.ggml.tokens")
        .expect("tokens array must be present");
    let tokens_len = match tokens_meta {
        mlx_native::gguf::MetadataValue::Array(v) => v.len(),
        other => panic!("tokenizer.ggml.tokens not Array: {other:?}"),
    };
    assert_eq!(
        tokens_len, 64,
        "tokens array length must equal config.json::text_config::vocab_size = 64"
    );
    // BOS/EOS ids from the fixture's added_tokens (4 special tokens at
    // the tail: bos=vocab-4, eos=vocab-3 → 60, 61).
    assert_eq!(
        gguf.metadata_u32("tokenizer.ggml.bos_token_id"),
        Some(60),
        "bos_token_id = vocab_size - 4 per fixture layout"
    );
    assert_eq!(
        gguf.metadata_u32("tokenizer.ggml.eos_token_id"),
        Some(61),
        "eos_token_id = vocab_size - 3 per fixture layout"
    );
}

// ============================================================================
// Streaming memory bound — convert-v2 must NOT load the full model into RAM
// ============================================================================

/// **STREAMING MEMORY BOUND** — the convert-v2 driver must NOT buffer
/// every tensor's F32 dequant in RAM. Pre-fix this test would fail
/// catastrophically on a 26B-param model (~104 GB peak RSS on a 48 GB
/// source).
///
/// We synthesize an artificially-fat fixture: 8 tensors of `[1024, 256]`
/// BF16 → 4 MB each on disk, 8 MB each as F32 = 64 MB total source, 64
/// MB total F32 working set if buffered. With streaming, peak F32
/// working set should be ~1 single-tensor = 8 MB.
///
/// The assertion bound is intentionally loose
/// (`4 × largest_single_tensor_F32 + 512 MB`) to absorb test-harness
/// overhead (cargo loads ~150 MB of debug-info itself on first invoke).
/// This is the spec'd bound from ADR-033's "Open Issues" section.
#[test]
fn convert_v2_streaming_rss_under_bound_2026_05_18() {
    use safetensors::tensor::TensorView;
    let dir = tempfile::tempdir().unwrap();

    // Fixture: 8 BF16 tensors at [1024, 256] = 4 MB on-disk each.
    // n_per_row=256 → Q8_0-compatible (block_size=32 divides 256).
    const N_TENSORS: usize = 8;
    const ROWS: usize = 1024;
    const COLS: usize = 256;
    let bf16_bytes_per_tensor = ROWS * COLS * 2;
    let f32_bytes_per_tensor = ROWS * COLS * 4;
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    for i in 0..N_TENSORS {
        let mut buf: Vec<u8> = Vec::with_capacity(bf16_bytes_per_tensor);
        for j in 0..(ROWS * COLS) {
            let v = ((i * ROWS * COLS + j) as f32) * 1e-4;
            buf.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
        }
        blobs.push(buf);
    }
    // Map them to canonical Llama-3 names so the convert-v2 driver
    // recognizes them through the standard arch dispatcher.
    // We use 2 layers × 4 weights per layer = 8 tensors.
    let names: Vec<String> = (0..2)
        .flat_map(|li| {
            vec![
                format!("model.layers.{li}.self_attn.q_proj.weight"),
                format!("model.layers.{li}.self_attn.k_proj.weight"),
                format!("model.layers.{li}.self_attn.v_proj.weight"),
                format!("model.layers.{li}.self_attn.o_proj.weight"),
            ]
        })
        .collect();

    let views: Vec<(String, TensorView<'_>)> = names
        .iter()
        .zip(blobs.iter())
        .map(|(n, b)| {
            let v = TensorView::new(safetensors::Dtype::BF16, vec![ROWS, COLS], b).unwrap();
            (n.clone(), v)
        })
        .collect();
    let view_refs: Vec<(String, &TensorView<'_>)> =
        views.iter().map(|(n, v)| (n.clone(), v)).collect();
    let st_bytes = safetensors::tensor::serialize(view_refs, None).unwrap();
    fs::write(dir.path().join("model.safetensors"), st_bytes).unwrap();
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": COLS,
            "num_hidden_layers": 2,
            "intermediate_size": COLS,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1.0e-5,
            "rope_theta": 10000.0,
            "vocab_size": 64,
        }))
        .unwrap(),
    )
    .unwrap();

    // ADR-033 tokenizer wiring: drop a minimal tokenizer.json so the
    // convert-v2 driver succeeds end-to-end. Vocab size must match
    // config.json::vocab_size = 64 above.
    write_minimal_tokenizer_fixture(dir.path(), 64);

    // Drive convert-v2 as a subprocess and parse its peak RSS via
    // `/usr/bin/time -l` (macOS) / `time -v` (Linux). assert_cmd's
    // Command runs the hf2q binary directly; we shell out to the OS's
    // `time` to capture max-RSS as a structured field after the
    // subprocess exits.
    let out = tempfile::NamedTempFile::new().unwrap();

    let hf2q_bin = assert_cmd::cargo::cargo_bin("hf2q");
    let mut cmd = std::process::Command::new("/usr/bin/time");
    #[cfg(target_os = "macos")]
    cmd.arg("-l");
    #[cfg(target_os = "linux")]
    cmd.args(["-f", "%M"]);
    cmd.arg(&hf2q_bin)
        .arg("convert-v2")
        .arg(dir.path())
        .arg("--quant")
        .arg("q8_0")
        .arg("-o")
        .arg(out.path());
    let result = cmd.output().expect("spawn /usr/bin/time hf2q convert-v2");

    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success(),
        "convert-v2 subprocess failed (status {:?}): stderr={stderr}",
        result.status.code()
    );

    // Sanity: the output is parseable and has the right tensor count.
    let gguf = mlx_native::gguf::GgufFile::open(out.path()).expect("parse output");
    assert_eq!(gguf.tensor_count() as usize, N_TENSORS);

    // Parse the OS-reported peak RSS.
    let rss_bytes = parse_time_max_rss(&stderr);

    if let Some(peak) = rss_bytes {
        // Bound: 4 × largest single-tensor F32 + 512 MB.
        // Pre-fix would overshoot by ~64 MB even on this small fixture,
        // and by ~104 GB on a real 26B model. Spec'd in ADR-033 §Open
        // Issues / Real-Model Findings.
        let bound = (4 * f32_bytes_per_tensor as u64) + (512 * 1024 * 1024);
        assert!(
            peak < bound,
            "convert-v2 peak RSS {} bytes ({} MiB) exceeded streaming bound {} bytes ({} MiB) — \
             the pipeline is buffering tensors instead of streaming them. \
             Buffered worst case would be ~{} MiB. See ADR-033 §Open Issues / Real-Model Findings.",
            peak,
            peak / (1024 * 1024),
            bound,
            bound / (1024 * 1024),
            (N_TENSORS * f32_bytes_per_tensor) / (1024 * 1024)
        );
    }
    // No assertion when /usr/bin/time isn't available (test is a soft
    // probe in that environment).
}

/// Parse the peak resident-set-size from /usr/bin/time's stderr output.
/// macOS prints `<bytes>  maximum resident set size`; Linux's GNU
/// /usr/bin/time -f "%M" prints KiB. Returns `Some(bytes)` on either,
/// `None` if no recognizable marker is found.
fn parse_time_max_rss(stderr: &str) -> Option<u64> {
    // macOS BSD time -l: "    1234567  maximum resident set size"
    for line in stderr.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_suffix("maximum resident set size") {
            let n: u64 = rest.trim().parse().ok()?;
            return Some(n);
        }
    }
    // GNU time -f "%M": prints just an integer KiB on the last stderr line.
    if let Some(last) = stderr.lines().rev().find(|l| {
        !l.trim().is_empty() && l.trim().chars().all(|c| c.is_ascii_digit())
    }) {
        let kib: u64 = last.trim().parse().ok()?;
        return Some(kib * 1024);
    }
    None
}
