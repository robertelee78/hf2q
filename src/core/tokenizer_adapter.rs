//! Tokenizer adapter — bridges HF `tokenizers::Tokenizer` and GGUF
//! metadata.
//!
//! ## Why this module exists
//!
//! HuggingFace-format `tokenizer.json` files encode special-token
//! handling in a `post_processor` template. Many real-world Gemma /
//! Qwen variants ship `tokenizer.json` files whose `post_processor`
//! template does NOT include BOS, even when the corresponding GGUF
//! sets `tokenizer.ggml.add_bos_token = true`. The HF
//! `tokenizer.encode(text, add_special_tokens=true)` call is then a
//! no-op for BOS, producing token sequences that diverge from
//! `llama.cpp`'s `common_tokenize` output by exactly the leading
//! BOS token.
//!
//! For Gemma 4 31B Q4_K_M, that missing BOS caused saturated garbage
//! output (token 240017 `"額"` repeated) — see ADR-038 G4-CFA-5e
//! commit `b0423671`.
//!
//! ## Semantics — `llama.cpp` parity
//!
//! Mirrors `llama.cpp`'s `common_tokenize(text, add_special=true,
//! parse_special=true)`:
//!
//! 1. Run `tokenizer.encode(text, add_special_tokens=false)` — bypass
//!    the (often broken) post_processor template entirely.
//! 2. If GGUF `tokenizer.ggml.add_bos_token == true` AND the token
//!    stream doesn't already start with `bos_token_id`, prepend BOS.
//! 3. If GGUF `tokenizer.ggml.add_eos_token == true` AND the token
//!    stream doesn't already end with `eos_token_id`, append EOS.
//!
//! BOS / EOS IDs are resolved by:
//! - GGUF `tokenizer.ggml.bos_token_id` / `eos_token_id` (preferred,
//!   matches `llama.cpp`'s preferred path), then
//! - `llama.cpp`'s tokenizer-model defaults (e.g. `gpt2` defaults
//!   both BOS and EOS to id 11) per
//!   `/opt/llama.cpp/src/llama-vocab.cpp`.
//!
//! ## Why not fix the tokenizer.json file?
//!
//! `tokenizer.json` is a data file, not source. Users routinely
//! re-download GGUF model directories from third parties (bartowski,
//! unsloth, official Google) whose sibling `tokenizer.json` files
//! may or may not include BOS in post_processor — silent regression
//! risk. The GGUF-metadata-driven approach (matching `llama.cpp`) is
//! robust to that variance.
//!
//! For belt-and-suspenders, the [`fix_tokenizer_json_bos`] helper
//! can patch a `tokenizer.json` in place to add BOS to its
//! `post_processor.single` template — useful for environments where
//! the HF tokenizer is consumed by code outside hf2q's adapter
//! boundary.
//!
//! ## Callers
//!
//! All hf2q paths that take a user prompt string and produce token
//! IDs MUST route through [`tokenize_with_bos_eos_from_gguf`] (or a
//! sibling helper added to this module). Direct
//! `tokenizer.encode(text, true)` calls in production paths are a
//! bug class — covered by [`tests::raw_encode_misses_bos_when_post_processor_lacks_it`].
//!
//! Related: ADR-015 iter42 (the original CLI-side BOS fix at
//! `serve/mod.rs:1090`), ADR-038 G4-CFA-5e (the EAGLE-3-side BOS fix
//! at `inference/spec_decode/eagle3_orchestrator.rs`), this module
//! (the consolidation).

use anyhow::Result;

/// Returns true if the GGUF metadata key resolves to `Bool(true)`.
fn gguf_bool(gguf: &mlx_native::gguf::GgufFile, key: &str) -> bool {
    matches!(
        gguf.metadata(key),
        Some(mlx_native::gguf::MetadataValue::Bool(true))
    )
}

/// Resolve a special-token ID from GGUF metadata, validated against
/// the HF tokenizer's vocab. Returns `None` if absent or if the ID is
/// out of the tokenizer's vocabulary.
fn resolve_token_id(
    gguf: &mlx_native::gguf::GgufFile,
    tokenizer: &tokenizers::Tokenizer,
    metadata_key: &str,
) -> Option<u32> {
    llama_cpp_special_token_id(gguf, metadata_key)
        .and_then(|id| tokenizer.id_to_token(id).map(|_| id))
}

fn llama_cpp_special_token_id(
    gguf: &mlx_native::gguf::GgufFile,
    metadata_key: &str,
) -> Option<u32> {
    if let Some(id) = gguf.metadata_u32(metadata_key) {
        return Some(id);
    }
    let tokenizer_model = gguf.metadata_string("tokenizer.ggml.model")?;
    llama_cpp_special_token_id_for_model(tokenizer_model, metadata_key)
}

fn llama_cpp_special_token_id_for_model(tokenizer_model: &str, metadata_key: &str) -> Option<u32> {
    match (tokenizer_model, metadata_key) {
        // Mirrors `/opt/llama.cpp/src/llama-vocab.cpp`: for tokenizer
        // model `gpt2`, llama.cpp initializes both BOS and EOS to
        // token id 11 before applying GGUF metadata overrides.
        ("gpt2", "tokenizer.ggml.bos_token_id") | ("gpt2", "tokenizer.ggml.eos_token_id") => {
            Some(11)
        }
        _ => None,
    }
}

/// Tokenize a rendered prompt with `llama.cpp` `common_tokenize(...,
/// add_special=true, parse_special=true)` semantics.
///
/// See module docs for the contract.
pub fn tokenize_with_bos_eos_from_gguf(
    gguf: &mlx_native::gguf::GgufFile,
    tokenizer: &tokenizers::Tokenizer,
    prompt_text: &str,
) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(prompt_text, false)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
    let mut prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

    if gguf_bool(gguf, "tokenizer.ggml.add_bos_token") {
        if let Some(bos) = resolve_token_id(gguf, tokenizer, "tokenizer.ggml.bos_token_id") {
            if prompt_tokens.first() != Some(&bos) {
                prompt_tokens.insert(0, bos);
            }
        }
    }

    if gguf_bool(gguf, "tokenizer.ggml.add_eos_token") {
        if let Some(eos) = resolve_token_id(gguf, tokenizer, "tokenizer.ggml.eos_token_id") {
            if prompt_tokens.last() != Some(&eos) {
                prompt_tokens.push(eos);
            }
        }
    }

    Ok(prompt_tokens)
}

/// Resolve the BOS token ID for a GGUF + tokenizer pair, honoring the
/// `tokenizer.ggml.add_bos_token` flag.
///
/// Returns `Some(id)` when the GGUF declares `add_bos_token=true` AND
/// `bos_token_id` resolves to a valid tokenizer vocab entry; `None`
/// otherwise. Callers that hold pre-tokenized prompts (e.g. tests
/// passing literal token vectors) use this to decide whether to
/// prepend BOS themselves.
pub fn resolve_bos_token_id(
    gguf: &mlx_native::gguf::GgufFile,
    tokenizer: &tokenizers::Tokenizer,
) -> Option<u32> {
    if !gguf_bool(gguf, "tokenizer.ggml.add_bos_token") {
        return None;
    }
    resolve_token_id(gguf, tokenizer, "tokenizer.ggml.bos_token_id")
}

/// Patch a HuggingFace `tokenizer.json` in place to add a BOS
/// `SpecialToken` to its `post_processor.single` template (and the
/// `special_tokens` map). After patching, `tokenizer.encode(text,
/// add_special_tokens=true)` produces tokens with leading BOS,
/// matching what most modern Gemma / Qwen tokenizers ship by default.
///
/// This is a **belt-and-suspenders** complement to
/// [`tokenize_with_bos_eos_from_gguf`]. The runtime helper is the
/// load-bearing fix; this function is for environments where a
/// `tokenizer.json` is consumed by code outside hf2q's adapter
/// boundary (other tools, Python notebooks, etc.).
///
/// # Behavior
///
/// - If `post_processor` is already a `TemplateProcessing` whose
///   `single` template starts with a `SpecialToken` for BOS (id
///   matches `bos_token_text` arg), no-op (returns `Ok(false)`).
/// - Otherwise, prepends `SpecialToken { id: <bos_token_text>,
///   type_id: 0 }` to `single`, adds the entry to `special_tokens`,
///   writes the file back, returns `Ok(true)`.
///
/// # Errors
///
/// - File read / write / JSON parse failures.
/// - `post_processor` exists but is not `type: TemplateProcessing` (we
///   refuse to mutate other processor types — they were chosen
///   deliberately).
pub fn fix_tokenizer_json_bos(
    path: &std::path::Path,
    bos_token_text: &str,
    bos_token_id: u32,
) -> Result<bool> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("fix_tokenizer_json_bos: read {}: {e}", path.display()))?;
    let mut tk: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
        anyhow::anyhow!("fix_tokenizer_json_bos: parse JSON {}: {e}", path.display())
    })?;

    let pp = tk.get_mut("post_processor").ok_or_else(|| {
        anyhow::anyhow!(
            "fix_tokenizer_json_bos: tokenizer.json has no post_processor field at {}",
            path.display()
        )
    })?;

    let pp_type = pp.get("type").and_then(|v| v.as_str()).unwrap_or("");
    if pp_type != "TemplateProcessing" {
        anyhow::bail!(
            "fix_tokenizer_json_bos: refusing to mutate post_processor of type {:?} \
             (only TemplateProcessing is supported); file: {}",
            pp_type,
            path.display()
        );
    }

    let single = pp
        .get_mut("single")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "fix_tokenizer_json_bos: post_processor.single is not an array at {}",
                path.display()
            )
        })?;

    let already_starts_with_bos = single
        .first()
        .and_then(|v| v.get("SpecialToken"))
        .and_then(|st| st.get("id"))
        .and_then(|id| id.as_str())
        == Some(bos_token_text);

    if already_starts_with_bos {
        return Ok(false);
    }

    // Prepend BOS SpecialToken to single.
    let bos_entry = serde_json::json!({
        "SpecialToken": {
            "id": bos_token_text,
            "type_id": 0,
        }
    });
    single.insert(0, bos_entry);

    // Add BOS to special_tokens map.
    let special_tokens = pp
        .get_mut("special_tokens")
        .and_then(|v| v.as_object_mut())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "fix_tokenizer_json_bos: post_processor.special_tokens is not an object at {}",
                path.display()
            )
        })?;
    special_tokens.insert(
        bos_token_text.to_string(),
        serde_json::json!({
            "id": bos_token_text,
            "ids": [bos_token_id],
            "tokens": [bos_token_text],
        }),
    );

    let patched = serde_json::to_string_pretty(&tk)
        .map_err(|e| anyhow::anyhow!("fix_tokenizer_json_bos: serialize: {e}"))?;
    std::fs::write(path, patched)
        .map_err(|e| anyhow::anyhow!("fix_tokenizer_json_bos: write {}: {e}", path.display()))?;

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Synthesize a minimal HF tokenizer.json with vocab `<pad>=0`,
    /// `<bos>=2`, plus a handful of word tokens, and a
    /// `post_processor` whose `single` template does NOT prepend BOS.
    /// Returns (path, parsed tokenizer).
    fn synth_legacy_tokenizer_json() -> NamedTempFile {
        let body = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "pair": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}}
                ],
                "special_tokens": {}
            },
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<pad>": 0,
                    "<eos>": 1,
                    "<bos>": 2,
                    "hello": 10,
                    "world": 11
                },
                "unk_token": "<pad>"
            }
        });
        let mut tmp = NamedTempFile::new().expect("tempfile");
        tmp.write_all(serde_json::to_string(&body).unwrap().as_bytes())
            .expect("write");
        tmp
    }

    /// Sanity: a legacy tokenizer.json (no BOS in post_processor)
    /// drops BOS even with `add_special_tokens=true`. This is the bug
    /// class that motivated this module.
    #[test]
    fn raw_encode_misses_bos_when_post_processor_lacks_it_2026_05_23() {
        let tmp = synth_legacy_tokenizer_json();
        let tk = tokenizers::Tokenizer::from_file(tmp.path()).expect("load");
        let enc_true = tk.encode("hello world", true).expect("encode true");
        let enc_false = tk.encode("hello world", false).expect("encode false");
        // The "true" call should add specials, but with the broken
        // post_processor it's a no-op — identical to "false".
        assert_eq!(enc_true.get_ids(), enc_false.get_ids());
        assert_eq!(enc_true.get_ids(), &[10, 11]);
        // The buggy behavior: no leading BOS=2.
        assert_ne!(enc_true.get_ids().first(), Some(&2u32));
    }

    /// After `fix_tokenizer_json_bos` patches the file, `encode(text,
    /// true)` produces tokens with leading BOS.
    #[test]
    fn fix_tokenizer_json_bos_makes_encode_prepend_bos_2026_05_23() {
        let tmp = synth_legacy_tokenizer_json();
        let patched =
            fix_tokenizer_json_bos(tmp.path(), "<bos>", 2).expect("fix_tokenizer_json_bos");
        assert!(patched, "first patch should mutate the file");

        let tk = tokenizers::Tokenizer::from_file(tmp.path()).expect("load patched");
        let enc = tk.encode("hello world", true).expect("encode");
        assert_eq!(enc.get_ids(), &[2, 10, 11], "BOS=2 should now be prepended");

        // Idempotent: second call no-ops.
        let patched2 = fix_tokenizer_json_bos(tmp.path(), "<bos>", 2)
            .expect("fix_tokenizer_json_bos idempotent");
        assert!(!patched2, "second patch should be no-op (already fixed)");
    }
}
