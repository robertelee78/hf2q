//! ADR-033 P3 — convert-v2 tokenizer metadata builder.
//!
//! Ports the tokenizer-emission block previously buried inside
//! `src/backends/gguf.rs::load_tokenizer_metadata` (legacy `cmd_convert`
//! pipeline) into a focused, convert-v2-only module. The legacy
//! implementation stays load-bearing until P6 deletes it; this module
//! becomes the canonical convert-v2 path.
//!
//! Surface:
//!
//!  - [`build_tokenizer_metadata`] reads `tokenizer.json` +
//!    `tokenizer_config.json` from an HF model directory and returns the
//!    `(key, value)` pairs the orchestrator writes into the GGUF KV
//!    section. Keys emitted:
//!
//!    * `tokenizer.ggml.model`           (string)
//!    * `tokenizer.ggml.tokens`          (array of strings, vocab-size long)
//!    * `tokenizer.ggml.scores`          (array of f32)
//!    * `tokenizer.ggml.token_type`      (array of i32)
//!    * `tokenizer.ggml.merges`          (array of strings, BPE rules — when present)
//!    * `tokenizer.ggml.bos_token_id`    (u32, when resolvable)
//!    * `tokenizer.ggml.eos_token_id`    (u32, when resolvable)
//!    * `tokenizer.ggml.unknown_token_id`(u32, when resolvable)
//!    * `tokenizer.ggml.padding_token_id`(u32, when resolvable)
//!    * `tokenizer.ggml.add_bos_token`   (bool, true — gemma.py:653)
//!    * `tokenizer.ggml.add_space_prefix`(bool, false — gemma.py:652)
//!    * `tokenizer.ggml.pre`             (string, llama-vocab.cpp pre-tokenizer bucket)
//!    * `tokenizer.chat_template`        (string, when present)
//!
//! Per [[feedback-no-loop-suppression-2026-05-17]]: every parse /
//! resolve failure surfaces as a typed [`TokenizerError`]; we never
//! silently fall back to a partial / empty metadata block (that's the
//! exact silent-corruption signature the 2026-04-30 DWQ48/46 GGUFs
//! produced).
//!
//! Per [[feedback-no-backwards-compat-2026-05-18]]: this module does
//! NOT delegate to the legacy emitter — every byte is generated here.
//! When P6 retires `cmd_convert`, the legacy block at
//! `src/backends/gguf.rs:2742-3200` gets deleted, not aliased.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::backends::gguf::types::MetaValue;
use crate::quantize::ggml_quants::ArchName;

// ---------------------------------------------------------------------------
// Token-type constants (mirror llama.cpp's TokenType enum)
// ---------------------------------------------------------------------------

const TOKEN_TYPE_NORMAL: i32 = 1;
// 2 = UNKNOWN — unused; `<unk>` lives in added_tokens with special=true,
// so it's classified as CONTROL in line with LlamaHfVocab.get_token_type().
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
const TOKEN_TYPE_UNUSED: i32 = 5;
const TOKEN_TYPE_BYTE: i32 = 6;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Typed-error surface for [`build_tokenizer_metadata`]. Each variant
/// matches one of the "silent corruption" failure modes the legacy
/// emitter had to learn the hard way (2026-04-30 DWQ48/46 regression),
/// so the convert-v2 pipeline refuses to ship a GGUF whose vocab is
/// truncated, mis-classified, or whose declared EOS token is
/// unresolvable. Per [[feedback-no-loop-suppression-2026-05-17]], we
/// never wallpaper over these by emitting an empty / partial KV block.
#[derive(Debug)]
pub enum TokenizerError {
    /// `tokenizer.json` is missing from the HF model directory. Real
    /// production models always ship one; absence is a typed error
    /// rather than a silent skip (the pre-port behavior emitted an
    /// empty GGUF that llama.cpp rejects with `key not found in model:
    /// tokenizer.ggml.model`).
    TokenizerJsonMissing { dir: String },
    /// `tokenizer.json` is present but unreadable / malformed.
    TokenizerJsonMalformed { dir: String, source: String },
    /// `tokenizer.json` parsed but is missing the `model` section, or
    /// `model.vocab` is not a non-empty object. Real BPE / SentencePiece
    /// tokenizers always have this; absence is a corrupt fixture.
    TokenizerJsonMissingModel { dir: String },
    /// `config.json` has no `vocab_size` (and no nested
    /// `text_config.vocab_size`). The legacy emitter previously fell
    /// back to `max(observed id) + 1`, which silently produced the
    /// 2026-04-30 DWQ48/46 broken GGUFs whose `<|im_end|>` etc. were
    /// truncated. This typed error replaces that silent fallback.
    ConfigMissingVocabSize { dir: String },
    /// `tokenizer_config.json` declares an `eos_token` (or `bos_token`)
    /// string, but that string is not present in the merged
    /// base-BPE + added_tokens vocab. Same silent-corruption signature
    /// as the 2026-04-30 regression; surface here.
    SpecialTokenUnresolvable {
        which: &'static str,
        token: String,
        merged_vocab_size: usize,
    },
    /// `added_tokens` carries an id that is `>= vocab_size`. Emitting
    /// would gap-fill the id with `[PAD]` and silently drop the named
    /// special token. Surface here.
    AddedTokenIdOutOfRange { id: u32, vocab_size: usize },
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::TokenizerJsonMissing { dir } => write!(
                f,
                "tokenizer: no tokenizer.json in {dir} (convert-v2 requires one — \
                 producing a GGUF without tokenizer metadata yields llama.cpp's \
                 `key not found in model: tokenizer.ggml.model` rejection)"
            ),
            TokenizerError::TokenizerJsonMalformed { dir, source } => write!(
                f,
                "tokenizer: tokenizer.json in {dir} is unreadable / malformed: {source}"
            ),
            TokenizerError::TokenizerJsonMissingModel { dir } => write!(
                f,
                "tokenizer: tokenizer.json in {dir} is missing the `model` section \
                 (or `model.vocab` is empty / non-object)"
            ),
            TokenizerError::ConfigMissingVocabSize { dir } => write!(
                f,
                "tokenizer: config.json in {dir} has no `vocab_size` (and no \
                 `text_config.vocab_size`). Refusing to fall back to \
                 max-observed-id+1 — that is the silent-corruption path that \
                 produced the 2026-04-30 DWQ48/46 broken GGUFs."
            ),
            TokenizerError::SpecialTokenUnresolvable {
                which,
                token,
                merged_vocab_size,
            } => write!(
                f,
                "tokenizer: {which} = {token:?} declared in tokenizer_config.json \
                 but not present in the merged vocab of {merged_vocab_size} ids — \
                 same silent-corruption signature as the 2026-04-30 DWQ48/46 \
                 regression. Refusing to emit."
            ),
            TokenizerError::AddedTokenIdOutOfRange { id, vocab_size } => write!(
                f,
                "tokenizer: added_token id {id} is >= resolved vocab_size {vocab_size}; \
                 emitting would gap-fill with `[PAD]` and silently drop the special \
                 token. Either config.json[vocab_size] is too small or \
                 tokenizer.json[added_tokens] has an out-of-range entry."
            ),
        }
    }
}

impl std::error::Error for TokenizerError {}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the `tokenizer.*` GGUF metadata KV list from a HuggingFace
/// model directory.
///
/// `model_dir` must contain `tokenizer.json` and (typically)
/// `tokenizer_config.json`. `arch` drives the per-arch
/// `tokenizer.ggml.model` and `tokenizer.ggml.pre` dispatch (Gemma 4 →
/// `"gemma4"` per gemma.py:649; Qwen3.5/3.6 MoE → `"qwen35"` per
/// llama-vocab.cpp:2042; etc.).
///
/// Returns a vec of `(key, MetaValue)` ready to be appended to the
/// orchestrator's metadata.
pub fn build_tokenizer_metadata(
    model_dir: &Path,
    arch: ArchName,
) -> Result<Vec<(String, MetaValue)>, TokenizerError> {
    // ----- 1. Load tokenizer.json ---------------------------------------
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(TokenizerError::TokenizerJsonMissing {
            dir: model_dir.display().to_string(),
        });
    }
    let tokenizer_json: serde_json::Value = match std::fs::read_to_string(&tokenizer_path) {
        Ok(s) => match serde_json::from_str(&s) {
            Ok(v) => v,
            Err(e) => {
                return Err(TokenizerError::TokenizerJsonMalformed {
                    dir: model_dir.display().to_string(),
                    source: e.to_string(),
                });
            }
        },
        Err(e) => {
            return Err(TokenizerError::TokenizerJsonMalformed {
                dir: model_dir.display().to_string(),
                source: e.to_string(),
            });
        }
    };

    let model_section = tokenizer_json
        .get("model")
        .ok_or_else(|| TokenizerError::TokenizerJsonMissingModel {
            dir: model_dir.display().to_string(),
        })?;
    let vocab_obj = model_section
        .get("vocab")
        .and_then(|v| v.as_object())
        .ok_or_else(|| TokenizerError::TokenizerJsonMissingModel {
            dir: model_dir.display().to_string(),
        })?;

    // ----- 2. Optional tokenizer_config.json (special tokens + flags) ---
    // Missing or malformed is non-fatal — the BOS/EOS lookup just yields
    // None and we skip those KV entries.
    let tokenizer_config: Option<serde_json::Value> = std::fs::read_to_string(
        model_dir.join("tokenizer_config.json"),
    )
    .ok()
    .and_then(|s| serde_json::from_str(&s).ok());

    // ----- 3. Merge base BPE + added_tokens ---------------------------
    // Same logic as legacy load_tokenizer_metadata; ports the
    // 2026-05-02 fix for added_tokens that lie above the base BPE range
    // (Qwen3.6 ids 248044-248069 for `<|im_end|>` etc.).
    let base_vocab_size = vocab_obj.len();
    let mut id_to_token: HashMap<u32, String> =
        HashMap::with_capacity(base_vocab_size + 64);
    for (tok, id_val) in vocab_obj.iter() {
        if let Some(id) = id_val.as_u64() {
            id_to_token.insert(id as u32, tok.clone());
        }
    }

    let added_tokens_arr = tokenizer_json
        .get("added_tokens")
        .and_then(|v| v.as_array());
    let mut added_ids: HashSet<u32> = HashSet::new();
    let mut added_special_flag: HashSet<u32> = HashSet::new();
    if let Some(added) = added_tokens_arr {
        for entry in added {
            let id_opt = entry.get("id").and_then(|v| v.as_u64()).map(|x| x as u32);
            let content_opt = entry.get("content").and_then(|v| v.as_str());
            let is_special = entry
                .get("special")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if let (Some(id), Some(content)) = (id_opt, content_opt) {
                id_to_token.insert(id, content.to_string());
                added_ids.insert(id);
                if is_special {
                    added_special_flag.insert(id);
                }
            }
        }
    }

    // ----- 4. Resolve target vocab_size from config.json --------------
    let max_observed_id = id_to_token.keys().max().copied().unwrap_or(0) as usize;
    let target_vocab_size = match read_full_vocab_size_from_config(model_dir) {
        Some(v) => {
            let v = v as usize;
            // Defensive: config's value should always cover the observed
            // tokens. If it's smaller, we'd silently drop ids — surface
            // the same `ConfigMissingVocabSize` typed error rather than
            // emit a truncated GGUF.
            if v < max_observed_id + 1 {
                return Err(TokenizerError::ConfigMissingVocabSize {
                    dir: model_dir.display().to_string(),
                });
            }
            v
        }
        None => {
            return Err(TokenizerError::ConfigMissingVocabSize {
                dir: model_dir.display().to_string(),
            });
        }
    };

    // ----- 5. Build ordered tokens array, gap-filling with [PAD{i}]. ----
    let mut tokens: Vec<String> = (0..target_vocab_size)
        .map(|i| format!("[PAD{i}]"))
        .collect();
    let mut filled_ids: HashSet<u32> = HashSet::new();
    for (id, token) in &id_to_token {
        if (*id as usize) < target_vocab_size {
            tokens[*id as usize] = token.clone();
            filled_ids.insert(*id);
        }
    }
    if let Some(&out_of_range) = added_ids
        .iter()
        .find(|id| (**id as usize) >= target_vocab_size)
    {
        return Err(TokenizerError::AddedTokenIdOutOfRange {
            id: out_of_range,
            vocab_size: target_vocab_size,
        });
    }

    // ----- 6. Build merged vocab_entries for special-token lookup ------
    let mut vocab_entries: Vec<(String, u32)> = id_to_token
        .iter()
        .map(|(id, t)| (t.clone(), *id))
        .collect();
    vocab_entries.sort_by_key(|(_, id)| *id);

    // ----- 7. Token-type classification (mirrors legacy 2940-2986) ---
    let visible_tokens: HashSet<&str> = [
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|\"|>",
    ]
    .iter()
    .copied()
    .collect();

    let scores: Vec<f32> = vec![-1000.0; target_vocab_size];
    let token_types: Vec<i32> = tokens
        .iter()
        .enumerate()
        .map(|(id, token)| {
            let id_u32 = id as u32;
            if !filled_ids.contains(&id_u32) {
                return TOKEN_TYPE_UNUSED;
            }
            if arch == ArchName::Gemma4 && visible_tokens.contains(token.as_str()) {
                return TOKEN_TYPE_USER_DEFINED;
            }
            if is_byte_token(token) {
                return TOKEN_TYPE_BYTE;
            }
            if added_ids.contains(&id_u32) {
                if added_special_flag.contains(&id_u32) || does_token_look_special(token) {
                    return TOKEN_TYPE_CONTROL;
                }
                return TOKEN_TYPE_USER_DEFINED;
            }
            TOKEN_TYPE_NORMAL
        })
        .collect();

    // ----- 8. Resolve special token ids against merged vocab ----------
    let bos_id = resolve_special_token_id("bos_token", &tokenizer_config, &vocab_entries);
    let eos_id = resolve_special_token_id("eos_token", &tokenizer_config, &vocab_entries);
    let unk_id = resolve_special_token_id("unk_token", &tokenizer_config, &vocab_entries);
    let pad_id = resolve_special_token_id("pad_token", &tokenizer_config, &vocab_entries);

    // Hard-fail if tokenizer_config.json declared an eos_token but we
    // cannot map it to an id in the resolved vocab. That exact
    // silent-skip is what produced the 2026-04-30 DWQ48/46 regression.
    if let Some(cfg) = tokenizer_config.as_ref() {
        if let Some(eos_str) = extract_special_token_string(cfg, "eos_token") {
            if eos_id.is_none() {
                return Err(TokenizerError::SpecialTokenUnresolvable {
                    which: "eos_token",
                    token: eos_str,
                    merged_vocab_size: target_vocab_size,
                });
            }
        }
    }

    // ----- 9. Tokenizer model + pre-tokenizer per arch ----------------
    let tokenizer_model_name = determine_tokenizer_model_name(model_section, arch);
    let pre_tokenizer = determine_pre_tokenizer_type(arch);

    // ----- 10. Merges --------------------------------------------------
    let merges = extract_merges(model_section);

    // ----- 11. Assemble KV list ---------------------------------------
    let mut kv: Vec<(String, MetaValue)> = Vec::with_capacity(16);
    kv.push((
        "tokenizer.ggml.model".into(),
        MetaValue::String(tokenizer_model_name),
    ));
    kv.push((
        "tokenizer.ggml.tokens".into(),
        MetaValue::ArrayString(tokens),
    ));
    kv.push((
        "tokenizer.ggml.scores".into(),
        MetaValue::ArrayF32(scores),
    ));
    kv.push((
        "tokenizer.ggml.token_type".into(),
        MetaValue::ArrayI32(token_types),
    ));
    if !merges.is_empty() {
        kv.push((
            "tokenizer.ggml.merges".into(),
            MetaValue::ArrayString(merges),
        ));
    }
    if let Some(id) = bos_id {
        kv.push(("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(id)));
    }
    if let Some(id) = eos_id {
        kv.push(("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(id)));
    }
    if let Some(id) = unk_id {
        kv.push((
            "tokenizer.ggml.unknown_token_id".into(),
            MetaValue::U32(id),
        ));
    }
    if let Some(id) = pad_id {
        kv.push((
            "tokenizer.ggml.padding_token_id".into(),
            MetaValue::U32(id),
        ));
    }
    // gemma.py:652-653: add_bos_token=True + add_space_prefix=False.
    // The legacy emitter applied these unconditionally; we preserve
    // that surface — every arch convert-v2 supports today wants both.
    kv.push((
        "tokenizer.ggml.add_bos_token".into(),
        MetaValue::Bool(true),
    ));
    kv.push((
        "tokenizer.ggml.add_space_prefix".into(),
        MetaValue::Bool(false),
    ));
    kv.push((
        "tokenizer.ggml.pre".into(),
        MetaValue::String(pre_tokenizer),
    ));

    // ----- 12. Chat template (when available) -------------------------
    // Priority chain (ADR-012 chat-template-auto-inject 2026-04-30):
    //   1. chat_template.jinja sidecar file
    //   2. tokenizer_config.json[chat_template]
    //   3. arch-default from `chat_templates::arch_default_chat_template`
    //   4. graceful skip — the runtime falls back to its built-in
    //      template loader.
    let chat_template_path = model_dir.join("chat_template.jinja");
    let template: Option<String> = if chat_template_path.exists() {
        std::fs::read_to_string(&chat_template_path).ok()
    } else if let Some(s) = tokenizer_config
        .as_ref()
        .and_then(|c| c.get("chat_template"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
    {
        Some(s)
    } else {
        crate::core::chat_templates::arch_default_chat_template(arch.name())
            .map(|s| s.to_string())
    };
    if let Some(tmpl) = template {
        kv.push((
            "tokenizer.chat_template".into(),
            MetaValue::String(tmpl),
        ));
    }

    Ok(kv)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read the model's full `vocab_size` from `config.json`, preferring
/// `text_config.vocab_size` (Gemma 4 / Qwen3-VL multimodal-wrapper
/// style) before the top-level key. Same precedence as the legacy
/// reader at `gguf.rs:2717`.
fn read_full_vocab_size_from_config(dir: &Path) -> Option<u64> {
    let path = dir.join("config.json");
    let s = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
    v.get("text_config")
        .and_then(|tc| tc.get("vocab_size"))
        .and_then(|x| x.as_u64())
        .or_else(|| v.get("vocab_size").and_then(|x| x.as_u64()))
}

/// `<0xNN>` byte-fallback token detector (mirrors llama.cpp).
fn is_byte_token(token: &str) -> bool {
    let b = token.as_bytes();
    b.len() == 6
        && b[0] == b'<'
        && b[1] == b'0'
        && b[2] == b'x'
        && b[3].is_ascii_hexdigit()
        && b[4].is_ascii_hexdigit()
        && b[5] == b'>'
}

/// Mirror of llama.cpp's `Model.does_token_look_special` heuristic.
fn does_token_look_special(token: &str) -> bool {
    if token.starts_with("<|") && token.ends_with("|>") {
        return true;
    }
    if token.starts_with('<') && token.ends_with('>') && token.len() > 2 && token != "<unk>" {
        return true;
    }
    false
}

/// Resolve a special-token name (`bos_token` / `eos_token` /
/// `unk_token` / `pad_token`) from tokenizer_config.json to its
/// merged-vocab id. Accepts both `"<bos>"` bare-string and
/// `{"content": "<bos>", ...}` AddedToken object shapes per
/// HuggingFace convention.
fn resolve_special_token_id(
    key: &str,
    config: &Option<serde_json::Value>,
    vocab_entries: &[(String, u32)],
) -> Option<u32> {
    let cfg = config.as_ref()?;
    let v = cfg.get(key)?;
    let token_str = v
        .as_str()
        .map(|s| s.to_string())
        .or_else(|| v.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()))?;
    vocab_entries
        .iter()
        .find(|(t, _)| t == &token_str)
        .map(|(_, id)| *id)
}

/// Extract the raw `eos_token` / `bos_token` string from
/// tokenizer_config.json. Used for the post-resolve cross-check
/// that's load-bearing for the 2026-04-30 DWQ48/46 guardrail.
fn extract_special_token_string(config: &serde_json::Value, key: &str) -> Option<String> {
    let v = config.get(key)?;
    v.as_str()
        .map(|s| s.to_string())
        .or_else(|| v.get("content").and_then(|c| c.as_str()).map(|s| s.to_string()))
}

/// Per-arch `tokenizer.ggml.model` dispatch — mirrors gemma.py:649 +
/// the legacy `determine_tokenizer_model_name` at `gguf.rs:3246`.
fn determine_tokenizer_model_name(model_section: &serde_json::Value, arch: ArchName) -> String {
    // Gemma 4 is hard-wired: gemma.py:649 calls
    // `self.gguf_writer.add_tokenizer_model("gemma4")` unconditionally.
    if arch == ArchName::Gemma4 {
        return "gemma4".into();
    }
    let model_type = model_section
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let byte_fallback = model_section
        .get("byte_fallback")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if model_type == "BPE" {
        if byte_fallback {
            // SentencePiece-style → llama family per legacy mapping.
            "llama".into()
        } else {
            "gpt2".into()
        }
    } else {
        "llama".into()
    }
}

/// Per-arch `tokenizer.ggml.pre` dispatch — enumeration at
/// `/opt/llama.cpp/src/llama-vocab.cpp` around line 1948-2061. The
/// `pre` field selects the regex-bucket pre-tokenizer applied before
/// BPE; different model families use distinct rules.
fn determine_pre_tokenizer_type(arch: ArchName) -> String {
    match arch {
        // llama-vocab.cpp:2042 — Qwen3.5 / Qwen3.6 family.
        ArchName::Qwen35Moe => "qwen35".into(),
        // llama-vocab.cpp:2035 — Qwen2-family pre-tokenizer also used
        // by Qwen3-VL text-side decoders.
        ArchName::Qwen3VlText => "qwen2".into(),
        // llama-vocab.cpp:2005 — Gemma 4 (and Gemma 3 → same bucket).
        ArchName::Gemma4 | ArchName::Gemma4Mmproj => "gemma4".into(),
        // llama-vocab.cpp:1951-1959 — LLaMA 3 BPE family.
        ArchName::Llama3 | ArchName::MiniMaxM2 => "llama-bpe".into(),
        // Bert + Nomic-BERT use the default pre-tokenizer.
        ArchName::Bert | ArchName::NomicBert => "default".into(),
        // Falcon is a `target_for` placeholder, never reaches convert-v2.
        ArchName::Falcon => "default".into(),
    }
}

/// Extract BPE merges from `tokenizer.json[model][merges]`. Handles
/// both the old (`Vec<String>`: `["a b", ...]`) and the new
/// (`Vec<Vec<String>>`: `[["a","b"], ...]`) formats. In the new
/// format, spaces inside merge parts are encoded as chr(288) per
/// llama.cpp's SpecialVocab convention.
fn extract_merges(model_section: &serde_json::Value) -> Vec<String> {
    let merges_val = match model_section.get("merges") {
        Some(v) => v,
        None => return Vec::new(),
    };
    let merges_arr = match merges_val.as_array() {
        Some(a) if !a.is_empty() => a,
        _ => return Vec::new(),
    };
    if merges_arr[0].is_string() {
        merges_arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    } else if merges_arr[0].is_array() {
        let space_replacement = '\u{0120}'; // chr(ord(' ') + 256) = 'Ġ'
        merges_arr
            .iter()
            .filter_map(|pair| {
                let arr = pair.as_array()?;
                if arr.len() != 2 {
                    return None;
                }
                let left = arr[0].as_str()?;
                let right = arr[1].as_str()?;
                let left_e: String = left
                    .chars()
                    .map(|c| if c == ' ' { space_replacement } else { c })
                    .collect();
                let right_e: String = right
                    .chars()
                    .map(|c| if c == ' ' { space_replacement } else { c })
                    .collect();
                Some(format!("{left_e} {right_e}"))
            })
            .collect()
    } else {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Synthesize a tiny but realistic Gemma 4 tokenizer fixture into
    /// `dir`. Returns the (bos_id, eos_id, total_vocab_size) the
    /// caller can assert against.
    ///
    /// Vocab layout:
    ///   ids 0..12: base BPE tokens ("a".."l")
    ///   ids 12..16: added special tokens (BOS / EOS / PAD / UNK)
    ///   total vocab_size = 16
    fn write_tiny_gemma4_tokenizer(dir: &Path) -> (u32, u32, usize) {
        // tokenizer.json — BPE with byte_fallback + 12 base tokens +
        // 4 added special tokens.
        let mut vocab = serde_json::Map::new();
        for (i, ch) in ('a'..='l').enumerate() {
            vocab.insert(ch.to_string(), serde_json::json!(i as u64));
        }
        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "byte_fallback": true,
                "vocab": vocab,
                "merges": [
                    ["a", "b"],
                    ["c", "d"]
                ]
            },
            "added_tokens": [
                {"id": 12, "content": "<bos>", "special": true},
                {"id": 13, "content": "<eos>", "special": true},
                {"id": 14, "content": "<pad>", "special": true},
                {"id": 15, "content": "<unk>", "special": true},
            ]
        });
        fs::write(
            dir.join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();

        // tokenizer_config.json — declare BOS/EOS/PAD/UNK strings (bare
        // string form on bos/pad/unk, AddedToken-object form on eos to
        // exercise both branches).
        let tokenizer_config = serde_json::json!({
            "bos_token": "<bos>",
            "eos_token": {"content": "<eos>"},
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

        // config.json — vocab_size lives in text_config (Gemma 4
        // multimodal-wrapper convention).
        let config = serde_json::json!({
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "vocab_size": 16
            }
        });
        fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        (12, 13, 16)
    }

    #[test]
    fn gemma4_tokenizer_emits_canonical_kv() {
        let tmp = tempfile::tempdir().unwrap();
        let (bos_id, eos_id, vocab_size) = write_tiny_gemma4_tokenizer(tmp.path());

        let kv = build_tokenizer_metadata(tmp.path(), ArchName::Gemma4)
            .expect("must succeed for a well-formed Gemma 4 fixture");

        // The set of keys is fixed — quickly assert it without ordering.
        let keys: HashSet<&str> = kv.iter().map(|(k, _)| k.as_str()).collect();
        for required in [
            "tokenizer.ggml.model",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.scores",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.add_space_prefix",
            "tokenizer.ggml.pre",
        ] {
            assert!(keys.contains(required), "missing key {required}");
        }

        // Per-value sanity (gemma.py:649 contract).
        let model_name = lookup_str(&kv, "tokenizer.ggml.model");
        assert_eq!(model_name, "gemma4");

        let pre = lookup_str(&kv, "tokenizer.ggml.pre");
        assert_eq!(pre, "gemma4");

        let bos = lookup_u32(&kv, "tokenizer.ggml.bos_token_id");
        assert_eq!(bos, bos_id);
        let eos = lookup_u32(&kv, "tokenizer.ggml.eos_token_id");
        assert_eq!(eos, eos_id);

        // gemma.py:653: add_bos_token = True.
        let add_bos = lookup_bool(&kv, "tokenizer.ggml.add_bos_token");
        assert!(add_bos);
        // gemma.py:652: add_space_prefix = False.
        let add_space = lookup_bool(&kv, "tokenizer.ggml.add_space_prefix");
        assert!(!add_space);

        // Tokens array is exactly vocab_size long, ordered by id.
        let tokens = match kv.iter().find(|(k, _)| k == "tokenizer.ggml.tokens") {
            Some((_, MetaValue::ArrayString(v))) => v.clone(),
            other => panic!("tokens not ArrayString: {other:?}"),
        };
        assert_eq!(tokens.len(), vocab_size);
        // First 12 are base BPE 'a'..'l'.
        assert_eq!(tokens[0], "a");
        assert_eq!(tokens[11], "l");
        // Added tokens at the tail.
        assert_eq!(tokens[12], "<bos>");
        assert_eq!(tokens[13], "<eos>");

        // Scores: every entry is -1000.0 per LlamaHfVocab.
        let scores = match kv.iter().find(|(k, _)| k == "tokenizer.ggml.scores") {
            Some((_, MetaValue::ArrayF32(v))) => v.clone(),
            other => panic!("scores not ArrayF32: {other:?}"),
        };
        assert_eq!(scores.len(), vocab_size);
        for s in scores {
            assert_eq!(s, -1000.0);
        }

        // token_type: base BPE → NORMAL (1); <bos>, <eos>, <pad> →
        // CONTROL (3) per added_special_flag; <unk> → CONTROL (3) per
        // legacy classification (special=true).
        let toktypes = match kv.iter().find(|(k, _)| k == "tokenizer.ggml.token_type") {
            Some((_, MetaValue::ArrayI32(v))) => v.clone(),
            other => panic!("token_type not ArrayI32: {other:?}"),
        };
        assert_eq!(toktypes.len(), vocab_size);
        for v in &toktypes[..12] {
            assert_eq!(*v, TOKEN_TYPE_NORMAL);
        }
        for v in &toktypes[12..16] {
            assert_eq!(*v, TOKEN_TYPE_CONTROL);
        }

        // Merges round-trip (new format → space-encoded via 'Ġ').
        let merges = match kv.iter().find(|(k, _)| k == "tokenizer.ggml.merges") {
            Some((_, MetaValue::ArrayString(v))) => v.clone(),
            other => panic!("merges not ArrayString: {other:?}"),
        };
        assert_eq!(merges, vec!["a b".to_string(), "c d".to_string()]);
    }

    #[test]
    fn llama3_tokenizer_emits_llama_model_and_llama_bpe_pre() {
        let tmp = tempfile::tempdir().unwrap();
        // BPE + byte_fallback → "llama" model per
        // determine_tokenizer_model_name. Llama3 arch → "llama-bpe" pre.
        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "byte_fallback": true,
                "vocab": {"a": 0, "b": 1, "c": 2, "d": 3},
            },
            "added_tokens": [
                {"id": 4, "content": "<|begin_of_text|>", "special": true},
                {"id": 5, "content": "<|end_of_text|>", "special": true},
            ]
        });
        fs::write(
            tmp.path().join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
        let cfg = serde_json::json!({
            "vocab_size": 6,
            "bos_token": "<|begin_of_text|>",
            "eos_token": "<|end_of_text|>"
        });
        // tokenizer_config: use config.json for vocab_size, separate
        // file for special tokens.
        fs::write(
            tmp.path().join("tokenizer_config.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "bos_token": "<|begin_of_text|>",
                "eos_token": "<|end_of_text|>"
            }))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();

        let kv = build_tokenizer_metadata(tmp.path(), ArchName::Llama3).unwrap();
        assert_eq!(lookup_str(&kv, "tokenizer.ggml.model"), "llama");
        assert_eq!(lookup_str(&kv, "tokenizer.ggml.pre"), "llama-bpe");
        // BOS / EOS ids.
        assert_eq!(lookup_u32(&kv, "tokenizer.ggml.bos_token_id"), 4);
        assert_eq!(lookup_u32(&kv, "tokenizer.ggml.eos_token_id"), 5);
    }

    #[test]
    fn missing_tokenizer_json_errors() {
        let tmp = tempfile::tempdir().unwrap();
        // No tokenizer.json at all.
        let err = build_tokenizer_metadata(tmp.path(), ArchName::Gemma4)
            .expect_err("must error on missing tokenizer.json");
        assert!(matches!(err, TokenizerError::TokenizerJsonMissing { .. }));
    }

    #[test]
    fn missing_vocab_size_in_config_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "byte_fallback": true,
                "vocab": {"a": 0, "b": 1}
            }
        });
        fs::write(
            tmp.path().join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
        // config.json present but with no vocab_size and no
        // text_config.vocab_size.
        fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string_pretty(&serde_json::json!({"model_type": "llama"}))
                .unwrap(),
        )
        .unwrap();
        let err = build_tokenizer_metadata(tmp.path(), ArchName::Llama3)
            .expect_err("must error when vocab_size is missing");
        assert!(matches!(err, TokenizerError::ConfigMissingVocabSize { .. }));
    }

    #[test]
    fn eos_token_unresolvable_errors() {
        let tmp = tempfile::tempdir().unwrap();
        // tokenizer_config.json declares eos_token=<|eom|> but it is
        // NOT in the merged vocab. Same silent-corruption signature as
        // the 2026-04-30 DWQ48/46 regression — typed error.
        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "byte_fallback": true,
                "vocab": {"a": 0, "b": 1}
            }
        });
        fs::write(
            tmp.path().join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
        fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string_pretty(&serde_json::json!({"vocab_size": 2})).unwrap(),
        )
        .unwrap();
        fs::write(
            tmp.path().join("tokenizer_config.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "eos_token": "<|eom|>"
            }))
            .unwrap(),
        )
        .unwrap();
        let err = build_tokenizer_metadata(tmp.path(), ArchName::Llama3)
            .expect_err("must error when eos is unresolvable");
        assert!(matches!(err, TokenizerError::SpecialTokenUnresolvable { .. }));
    }

    #[test]
    fn qwen35moe_dispatch_emits_qwen35_pre() {
        let tmp = tempfile::tempdir().unwrap();
        // BPE without byte_fallback → "gpt2" model, but Qwen35Moe arch
        // → "qwen35" pre. This exercises the per-arch pre-tokenizer
        // dispatch separately from model-name.
        let tokenizer_json = serde_json::json!({
            "model": {
                "type": "BPE",
                "byte_fallback": false,
                "vocab": {"a": 0, "b": 1}
            }
        });
        fs::write(
            tmp.path().join("tokenizer.json"),
            serde_json::to_string_pretty(&tokenizer_json).unwrap(),
        )
        .unwrap();
        fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string_pretty(&serde_json::json!({"vocab_size": 2})).unwrap(),
        )
        .unwrap();
        let kv = build_tokenizer_metadata(tmp.path(), ArchName::Qwen35Moe).unwrap();
        assert_eq!(lookup_str(&kv, "tokenizer.ggml.model"), "gpt2");
        assert_eq!(lookup_str(&kv, "tokenizer.ggml.pre"), "qwen35");
    }

    #[test]
    fn does_token_look_special_pinned_pattern() {
        // Mirrors gguf.rs:6865-6892 — pin the heuristic so reviewers
        // see the same behavior across both emitters.
        for s in &["<|im_end|>", "<eos>", "<think>", "<|tool_call|>"] {
            assert!(does_token_look_special(s), "{s} should look special");
        }
        for s in &["abc", "<unk>", "<>", "<", ">"] {
            assert!(!does_token_look_special(s), "{s} should NOT look special");
        }
    }

    #[test]
    fn is_byte_token_pinned_pattern() {
        for s in &["<0x00>", "<0xff>", "<0xAB>", "<0x1f>"] {
            assert!(is_byte_token(s), "{s} should be a byte token");
        }
        for s in &["<0x0>", "<0xfff>", "<0xZZ>", "abc"] {
            assert!(!is_byte_token(s), "{s} should NOT be a byte token");
        }
    }

    // ---- Tiny KV lookup helpers (test-only) ----------------------------

    fn lookup_str(kv: &[(String, MetaValue)], key: &str) -> String {
        match kv.iter().find(|(k, _)| k == key) {
            Some((_, MetaValue::String(s))) => s.clone(),
            other => panic!("expected String at {key}, got {other:?}"),
        }
    }

    fn lookup_u32(kv: &[(String, MetaValue)], key: &str) -> u32 {
        match kv.iter().find(|(k, _)| k == key) {
            Some((_, MetaValue::U32(v))) => *v,
            other => panic!("expected U32 at {key}, got {other:?}"),
        }
    }

    fn lookup_bool(kv: &[(String, MetaValue)], key: &str) -> bool {
        match kv.iter().find(|(k, _)| k == key) {
            Some((_, MetaValue::Bool(v))) => *v,
            other => panic!("expected Bool at {key}, got {other:?}"),
        }
    }
}
