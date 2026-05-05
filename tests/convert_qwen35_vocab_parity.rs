//! Falsifier suite for the 2026-04-30 DWQ48/46 vocab-truncation regression
//! (CFA 2026-05-05 — see `project_qwen35_dwq_pre_505b5b8_broken_2026_05_05`).
//!
//! Bug recap: hf2q's GGUF converter pre-`505b5b8` (2026-05-02) built
//! `vocab_entries` from `tokenizer.json["model"]["vocab"]` only — the base
//! BPE table — and used `vocab_obj.len()` as the total vocab size. The
//! `added_tokens` region (Qwen's `<|im_end|>`, `<|endoftext|>`, vision/audio
//! special tokens) was silently dropped. The resulting GGUFs had vocab
//! 248044 instead of 248320, no `eos_token_id` metadata, and `output.weight`
//! tensor rows truncated to match — at runtime the model couldn't emit
//! `<|im_end|>` and rambled until `max_new_tokens`.
//!
//! These tests pin three invariants discovered during the post-mortem:
//!
//! 1. **Happy path** — a correctly-shaped HF dir with `vocab_size=144` +
//!    `added_tokens` for ids 128..143 + `eos_token = "<|im_end|>"` produces
//!    a GGUF with **exactly** 144 tokens, `eos_token_id = 130`, and the
//!    string at id 130 IS `<|im_end|>`.
//!
//! 2. **Bail when vocab_size is missing** — the silent-fallback to
//!    max-observed-id+1 (= base BPE size) is what produced the broken
//!    files. Hard-fail at convert time with `BackendError::ValidationFailed`.
//!
//! 3. **Bail when eos_token is unresolvable in the merged vocab** — the
//!    runtime symptom (no terminator) starts here. If `tokenizer_config.json`
//!    declares an `eos_token` but the converter can't map it to an id in
//!    the merged vocab, hard-fail.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use mlx_native::gguf::{GgufFile, MetadataValue};

// Synthetic Qwen-shaped dimensions sized to reproduce the bug class on
// a small footprint. Numbers chosen so:
//   * BASE_VOCAB (128) < TOTAL_VOCAB (144): added_tokens land ABOVE the
//     base BPE range — exactly the Qwen3.6 shape (248044 base + 276
//     added) at 1/2000th scale.
//   * ID_IM_END = 130: same offset within the added_tokens region as
//     real Qwen (`<|im_end|>` at base+2 because base+0 = `<|endoftext|>`,
//     base+1 = `<|im_start|>`).
//   * HIDDEN = 64: matches existing convert_qwen35_metadata_keys.rs
//     fixture so the safetensors helpers stay reusable.
const HIDDEN: usize = 64;
const BASE_VOCAB: usize = 128;
const TOTAL_VOCAB: usize = 144;
const ID_ENDOFTEXT: usize = 128;
const ID_IM_START: usize = 129;
const ID_IM_END: usize = 130;

/// Qwen-flavoured config.json. Keep in lockstep with the dense test
/// fixture in tests/convert_qwen35_metadata_keys.rs (same schema, larger
/// vocab_size). Populated below at runtime so tests can mutate single
/// fields (e.g. drop vocab_size) for the failure-path assertions.
fn dense_config_json(with_vocab_size: bool) -> serde_json::Value {
    let mut cfg = serde_json::json!({
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "intermediate_size": 128,
        "head_dim": 16,
        "linear_num_value_heads": 8,
        "full_attention_interval": 2,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000.0,
        "rope_parameters": {
            "mrope_section": [3, 3, 2],
            "rope_theta": 10000000.0,
            "rope_type": "mrope",
            "mrope_interleaved": true,
            "partial_rotary_factor": 0.25
        },
        "mtp_num_hidden_layers": 0,
        "attn_output_gate": true,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 131072,
        "dtype": "float16",
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_num_key_heads": 4,
        "linear_value_head_dim": 16,
        "linear_num_value_heads": 8,
    });
    if with_vocab_size {
        cfg["vocab_size"] = serde_json::json!(TOTAL_VOCAB);
    }
    cfg
}

fn push_f16(
    tensors: &mut Vec<(String, Vec<usize>, &'static str, Vec<u8>)>,
    name: &str,
    shape: Vec<usize>,
) {
    let n = shape.iter().product::<usize>() * 2;
    tensors.push((name.to_string(), shape, "F16", vec![0u8; n]));
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
    let hdr = serde_json::to_string(&header_map).unwrap();
    let hbytes = hdr.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(hbytes.len() as u64).to_le_bytes());
    out.extend_from_slice(hbytes);
    out.extend_from_slice(&payload);
    out
}

/// Build the dense-arch safetensors file. Embedding / lm_head rows are
/// `TOTAL_VOCAB` to match the declared `vocab_size`, exercising the merged-
/// vocab gap-fill path in `gguf::load_tokenizer_metadata`.
fn write_safetensors(dir: &Path) {
    let h = HIDDEN;
    let inter = 128usize;
    let mut tensors: Vec<(String, Vec<usize>, &str, Vec<u8>)> = Vec::new();
    push_f16(&mut tensors, "model.embed_tokens.weight", vec![TOTAL_VOCAB, h]);
    push_f16(&mut tensors, "lm_head.weight", vec![TOTAL_VOCAB, h]);
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
            let qkv_rows = 4 * 16 * 2 + 8 * 16; // 256 — matches metadata-keys fixture.
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
    fs::write(
        dir.join("model.safetensors"),
        build_safetensors_bytes(tensors),
    )
    .unwrap();
}

/// Realistic `tokenizer.json` with base BPE + Qwen-style added_tokens.
/// `with_added_tokens=false` lets the unresolvable-eos test reproduce the
/// "added_tokens parse silently dropped" failure mode.
fn tokenizer_json(with_added_tokens: bool) -> serde_json::Value {
    // Base BPE: 128 tokens, ids 0..127. Single-char strings keep the JSON
    // small and avoid collision with the `<|...|>` added-token forms.
    let mut vocab = serde_json::Map::new();
    for i in 0..BASE_VOCAB {
        vocab.insert(format!("base{i}"), serde_json::json!(i));
    }
    let mut added: Vec<serde_json::Value> = Vec::new();
    if with_added_tokens {
        // Mirror real Qwen3 tail: <|endoftext|>, <|im_start|>, <|im_end|>,
        // then 13 [PAD]-style filler ids so we hit TOTAL_VOCAB exactly.
        added.push(serde_json::json!({
            "id": ID_ENDOFTEXT, "content": "<|endoftext|>", "special": true,
        }));
        added.push(serde_json::json!({
            "id": ID_IM_START, "content": "<|im_start|>", "special": true,
        }));
        added.push(serde_json::json!({
            "id": ID_IM_END, "content": "<|im_end|>", "special": true,
        }));
        for id in (ID_IM_END + 1)..TOTAL_VOCAB {
            added.push(serde_json::json!({
                "id": id, "content": format!("<|extra_{id}|>"), "special": true,
            }));
        }
    }
    serde_json::json!({
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": [],
        },
        "added_tokens": added,
    })
}

fn tokenizer_config_json(eos_token: Option<&str>) -> serde_json::Value {
    let mut cfg = serde_json::json!({
        "model_max_length": 131072,
        "bos_token": "<|endoftext|>",
    });
    if let Some(eos) = eos_token {
        cfg["eos_token"] = serde_json::json!(eos);
    }
    cfg
}

struct FixtureSpec {
    with_vocab_size_in_config: bool,
    with_added_tokens: bool,
    eos_token: Option<&'static str>,
}

fn write_fixture(dir: &Path, spec: &FixtureSpec) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("config.json"),
        serde_json::to_string(&dense_config_json(spec.with_vocab_size_in_config)).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_string(&tokenizer_json(spec.with_added_tokens)).unwrap(),
    )
    .unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        serde_json::to_string(&tokenizer_config_json(spec.eos_token)).unwrap(),
    )
    .unwrap();
    write_safetensors(dir);
}

fn convert_cmd(input: &Path, output: &Path) -> assert_cmd::Command {
    let mut cmd = assert_cmd::Command::cargo_bin("hf2q").unwrap();
    cmd.args([
        "convert",
        "--input",
        input.to_str().unwrap(),
        "--format",
        "gguf",
        "--quant",
        "f16",
        "--output",
        output.to_str().unwrap(),
        "--yes",
        "--skip-quality",
    ]);
    cmd
}

fn metadata_array_string<'a>(
    gguf: &'a GgufFile,
    key: &str,
) -> Option<&'a [String]> {
    match gguf.metadata(key)? {
        MetadataValue::Array(a) => {
            // Array contains MetadataValue::String entries — we want the
            // strings themselves. The mlx-native API exposes them via
            // pattern match per-element; collect a borrowed slice if all
            // elements are strings (they should be for tokenizer.ggml.tokens).
            // Cheap sanity — first element type tells us the array type.
            if matches!(a.first(), Some(MetadataValue::String(_))) {
                // Safety/correctness: rather than allocating a Vec<&str>,
                // we use a sibling helper that walks the array. We can't
                // easily return a `&[String]` here without lifetime gymnastics,
                // so callers should use `metadata_array_string_owned` instead.
                // This helper exists for the type-shape sanity check only.
                Some(&[])
            } else {
                None
            }
        }
        _ => None,
    }
}

fn metadata_array_string_owned(gguf: &GgufFile, key: &str) -> Option<Vec<String>> {
    match gguf.metadata(key)? {
        MetadataValue::Array(a) => {
            let mut out = Vec::with_capacity(a.len());
            for v in a {
                match v {
                    MetadataValue::String(s) => out.push(s.clone()),
                    _ => return None,
                }
            }
            Some(out)
        }
        _ => None,
    }
}

fn metadata_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    match gguf.metadata(key)? {
        MetadataValue::Uint32(v) => Some(*v),
        _ => None,
    }
}

// ────────────────────────────────────────────────────────────────────
// Happy path — the canonical post-fix invariants.
// ────────────────────────────────────────────────────────────────────

#[test]
fn vocab_size_from_config_includes_added_tokens_and_eos_resolves() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("hf-in");
    let output = tmp.path().join("dense.gguf");
    write_fixture(
        &input,
        &FixtureSpec {
            with_vocab_size_in_config: true,
            with_added_tokens: true,
            eos_token: Some("<|im_end|>"),
        },
    );

    convert_cmd(&input, &output)
        .assert()
        .success();

    let gguf = GgufFile::open(&output).expect("open converted GGUF");

    // (a) Token count parity — the canonical invariant the bug violated.
    let tokens = metadata_array_string_owned(&gguf, "tokenizer.ggml.tokens")
        .expect("tokenizer.ggml.tokens must be ArrayString");
    assert_eq!(
        tokens.len(),
        TOTAL_VOCAB,
        "tokenizer.ggml.tokens length MUST match config.vocab_size; pre-fix \
         converter dropped added_tokens and emitted only the base BPE size"
    );

    // (b) Sentinel string at the canonical added-token position.
    assert_eq!(
        tokens[ID_IM_END], "<|im_end|>",
        "id {ID_IM_END} MUST resolve to the literal added_token string \
         '<|im_end|>'; the bug-class symptom was BPE-decoded shards like \
         '<|_end|>' appearing at runtime"
    );
    assert_eq!(tokens[ID_ENDOFTEXT], "<|endoftext|>");
    assert_eq!(tokens[ID_IM_START], "<|im_start|>");

    // (c) eos_token_id is present and points at the merged-vocab string.
    let eos_id = metadata_u32(&gguf, "tokenizer.ggml.eos_token_id")
        .expect("tokenizer.ggml.eos_token_id MUST be emitted when \
                 tokenizer_config.json declares eos_token");
    assert_eq!(
        eos_id as usize, ID_IM_END,
        "eos_token_id MUST resolve through the MERGED vocab — pre-fix \
         the resolver only saw base BPE so eos_token_id was never written"
    );

    // (d) bos_token_id parity (we declared bos_token = '<|endoftext|>').
    let bos_id = metadata_u32(&gguf, "tokenizer.ggml.bos_token_id")
        .expect("bos_token_id missing");
    assert_eq!(bos_id as usize, ID_ENDOFTEXT);
}

// ────────────────────────────────────────────────────────────────────
// Bail #1 — config.json missing vocab_size. Pre-fix this silently fell
// back to max-observed-id+1; post-fix it MUST hard-error.
// ────────────────────────────────────────────────────────────────────

#[test]
fn convert_bails_when_vocab_size_missing_from_config() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("hf-in");
    let output = tmp.path().join("nope.gguf");
    write_fixture(
        &input,
        &FixtureSpec {
            with_vocab_size_in_config: false, // ← THE bug-trigger
            with_added_tokens: true,
            eos_token: Some("<|im_end|>"),
        },
    );

    let assert = convert_cmd(&input, &output).assert().failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    assert!(
        stderr.contains("vocab_size") || stderr.contains("ValidationFailed"),
        "convert MUST fail with a vocab_size-related ValidationFailed when \
         config.json has no vocab_size. stderr was: {stderr}"
    );
    assert!(
        !output.exists(),
        "convert MUST NOT write a partial GGUF when validation fails; found \
         {} on disk",
        output.display()
    );
}

// ────────────────────────────────────────────────────────────────────
// Bail #2 — eos_token declared in tokenizer_config.json but unresolvable
// in the merged vocab (the runtime-symptom signature: GGUF emits no
// eos_token_id, model never terminates).
// ────────────────────────────────────────────────────────────────────

#[test]
fn convert_bails_when_eos_token_unresolvable_in_merged_vocab() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("hf-in");
    let output = tmp.path().join("nope2.gguf");
    write_fixture(
        &input,
        &FixtureSpec {
            with_vocab_size_in_config: true,
            with_added_tokens: false, // ← added_tokens silently dropped …
            eos_token: Some("<|im_end|>"), // … so this will not resolve
        },
    );

    let assert = convert_cmd(&input, &output).assert().failure();
    let stderr = String::from_utf8_lossy(&assert.get_output().stderr).into_owned();
    assert!(
        stderr.contains("eos_token") || stderr.contains("ValidationFailed"),
        "convert MUST fail with an eos_token-related ValidationFailed when \
         tokenizer_config.eos_token is unresolvable in the merged vocab. \
         stderr was: {stderr}"
    );
    assert!(
        !output.exists(),
        "convert MUST NOT write a partial GGUF when validation fails; found \
         {} on disk",
        output.display()
    );
}

// ────────────────────────────────────────────────────────────────────
// Suppressed lint: the `metadata_array_string` helper above is unused
// outside its own debug aid; keep it compiled for documentation only.
// ────────────────────────────────────────────────────────────────────
#[allow(dead_code)]
fn _silence_dead_code_warning() {
    let _ = metadata_array_string;
}
