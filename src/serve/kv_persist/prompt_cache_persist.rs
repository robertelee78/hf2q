//! ADR-017 Closure iter-5 (Phase E, 2026-05-04) — cross-process
//! `PromptCache` serialization layer.
//!
//! ## Why this module exists
//!
//! Phase D's spill→restore plumbing (iters 1-4) correctly persists
//! block-aligned K/V state to disk and restores it into `dense_kvs`
//! on cold-process resume — but `forward_prefill.rs:446` resets
//! `cache.write_pos = 0` at the start of every prefill, overwriting
//! the restored state. Per the doc-comment at `engine.rs:1442`, the
//! "LCP-based partial-prefill resume" feature that would actually
//! consume the restored KV state is iter-97+ scope, deferred.
//!
//! Iter-96's `PromptCache` (`engine.rs:1487`) is a RAM-only,
//! single-process, full-equality cache: it stores the prior
//! request's `(tokens, key, text, metadata)` and replays the prior
//! response when `new_prompt == cached_prompt`. The cache hit path
//! returns `prefill_duration: 0` and `decode_duration: 0` — the
//! model never runs. **Cross-process replay** is what iter-5 adds:
//! serialize `PromptCache` to a disk-backed envelope, restore it on
//! cold-process resume, and the existing iter-96 lookup path takes
//! over from there.
//!
//! ## Subset semantics
//!
//! `PromptCache` contains a `Grammar` runtime object that does not
//! implement `serde::Serialize`. For full-fidelity persistence we
//! would either (a) add serde derives to `Grammar` and every nested
//! type or (b) re-compile the grammar from a serializable
//! representation on restore. Both are outside iter-5 scope.
//!
//! Iter-5 ships a defensive subset:
//!   * Persist only when `cache.key.grammar.is_none()`. Grammar-
//!     constrained requests skip persistence and yield no cross-
//!     process replay until iter-6+ extends Grammar serde.
//!   * `cache.fragments` (per-emit SSE event sequence for fragment-
//!     replay) is also dropped on persist. Restore yields
//!     `fragments = None`, which iter-96's lookup handles via the
//!     splitter-rerun fallback path (Worker AA design §3b option (a)).
//!   * `tool_call_policy` and `grammar_kind` are re-defaulted on
//!     restore — the cache lookup is structural-equality, so a
//!     request with non-default values for these enums simply
//!     misses the restored entry. Correct behavior for the iter-5
//!     subset; iter-6+ adds explicit enum persistence if needed.
//!
//! Forward compatibility: the on-disk schema is a JSON envelope
//! versioned by `format_version`. Future iters can add fields with
//! `#[serde(default)]` attributes and bump the version.
//!
//! ## File layout
//!
//! Persistence rides on the existing `BlockPrefixCacheSpiller` /
//! `DiskBlockStore` infrastructure. The serialized payload is
//! enqueued as a `WriteJob` with `payload_kind = "prompt-cache"` and
//! stored under the per-model fingerprint subtree alongside the KV
//! blocks. On restore, the spiller's `post_admit` dispatches by
//! `payload_kind`: `kv-spiller-l<N>` routes to `restore_block`
//! (existing); `prompt-cache` routes to `restore_prompt_cache` (new).

use serde::{Deserialize, Serialize};

use crate::serve::api::engine::{
    GrammarKind, PromptCache, PromptCacheKey, ToolCallPolicy,
};

/// On-disk schema version for prompt-cache snapshots. Bump on any
/// breaking schema change.
pub const PROMPT_CACHE_FORMAT_VERSION: u32 = 1;

/// Payload-kind tag the spiller uses to route this envelope back to
/// `restore_prompt_cache` on `post_admit`. Distinct from the
/// per-layer `kv-spiller-l<N>` kind to keep the dispatch unambiguous.
pub const PROMPT_CACHE_PAYLOAD_KIND: &str = "prompt-cache";

/// Serializable subset of [`PromptCache`]. See module-level docs for
/// what's persisted vs dropped vs defaulted-on-restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCacheSnapshot {
    pub format_version: u32,
    pub tokens: Vec<u32>,
    pub key: PromptCacheKeyPersist,
    pub text: String,
    pub reasoning_text: Option<String>,
    pub completion_tokens: usize,
    pub reasoning_tokens: Option<usize>,
    /// One of `"stop"` / `"length"` / `"tool_calls"`. Stored as
    /// owned `String` because [`PromptCache::finish_reason`] is
    /// `&'static str` and serde does not serialize string slices.
    pub finish_reason: String,
}

/// Serializable subset of [`PromptCacheKey`]. Grammar / grammar_kind
/// / tool_call_policy are re-defaulted on restore (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCacheKeyPersist {
    pub max_tokens: usize,
    pub stop_strings: Vec<String>,
    pub logit_bias_sorted: Vec<(u32, u32)>,
    pub frequency_penalty_bits: u32,
    pub presence_penalty_bits: u32,
    pub min_p_bits: u32,
    pub logprobs: bool,
    pub top_logprobs: u32,
    pub parallel_tool_calls: bool,
    /// `true` iff `cache.key.grammar.is_none()`. Persistence is
    /// gated on this being `true` (see `try_serialize`), so the
    /// flag's only useful state is `true`. Stored explicitly anyway
    /// so a future iter can add a separate "grammar payload"
    /// alongside the prompt-cache envelope without breaking on
    /// pre-iter-6 snapshots.
    pub grammar_was_none: bool,
}

/// Serialize a [`PromptCache`] to JSON bytes for cross-process
/// persistence. Returns `None` when the cache is in a state we
/// cannot faithfully restore (empty cache; grammar-constrained
/// request). Callers that get `None` should skip the prompt-cache
/// envelope entirely — KV-block persistence is independent.
///
/// The returned byte format is JSON with a versioned envelope, so
/// future iters can extend without breaking pre-iter-5 snapshots.
pub fn try_serialize(cache: &PromptCache) -> Option<Vec<u8>> {
    if cache.tokens.is_empty() {
        // Empty cache (fresh worker) — nothing to persist.
        return None;
    }
    if cache.key.grammar.is_some() {
        // Grammar-constrained request — Grammar runtime doesn't have
        // serde derives. Skip persistence rather than persist a
        // grammar-less subset that would silently mis-replay on
        // restore.
        return None;
    }
    let snap = PromptCacheSnapshot {
        format_version: PROMPT_CACHE_FORMAT_VERSION,
        tokens: cache.tokens.clone(),
        key: PromptCacheKeyPersist {
            max_tokens: cache.key.max_tokens,
            stop_strings: cache.key.stop_strings.clone(),
            logit_bias_sorted: cache.key.logit_bias_sorted.clone(),
            frequency_penalty_bits: cache.key.frequency_penalty_bits,
            presence_penalty_bits: cache.key.presence_penalty_bits,
            min_p_bits: cache.key.min_p_bits,
            logprobs: cache.key.logprobs,
            top_logprobs: cache.key.top_logprobs,
            parallel_tool_calls: cache.key.parallel_tool_calls,
            grammar_was_none: true,
        },
        text: cache.text.clone(),
        reasoning_text: cache.reasoning_text.clone(),
        completion_tokens: cache.completion_tokens,
        reasoning_tokens: cache.reasoning_tokens,
        finish_reason: cache.finish_reason.to_string(),
    };
    serde_json::to_vec(&snap).ok()
}

/// Deserialize a JSON byte payload into a [`PromptCache`]. Returns
/// `None` on any parse failure or schema-version mismatch.
///
/// Restored fields:
///   * tokens, text, reasoning_text, completion_tokens,
///     reasoning_tokens, finish_reason — restored verbatim.
///   * `key.grammar` defaults to `None` (persistence gated on
///     `grammar_was_none == true`).
///   * `key.grammar_kind` defaults to `GrammarKind::default()`.
///   * `key.tool_call_policy` defaults to `ToolCallPolicy::default()`.
///   * `fragments` defaults to `None` — iter-96 lookup falls back
///     to the splitter-rerun replay path on `fragments=None`.
///
/// `finish_reason` is decoded from the persisted string back into
/// the `&'static str` literal contract. Unknown strings yield
/// `None` (defensive: rather than coerce to "length" and silently
/// mis-attribute the finish reason).
pub fn try_deserialize(bytes: &[u8]) -> Option<PromptCache> {
    let snap: PromptCacheSnapshot = serde_json::from_slice(bytes).ok()?;
    if snap.format_version != PROMPT_CACHE_FORMAT_VERSION {
        return None;
    }
    let finish_reason: &'static str = match snap.finish_reason.as_str() {
        "stop" => "stop",
        "length" => "length",
        "tool_calls" => "tool_calls",
        _ => return None,
    };
    Some(PromptCache {
        tokens: snap.tokens,
        key: PromptCacheKey {
            max_tokens: snap.key.max_tokens,
            stop_strings: snap.key.stop_strings,
            logit_bias_sorted: snap.key.logit_bias_sorted,
            grammar: None,
            grammar_kind: GrammarKind::default(),
            frequency_penalty_bits: snap.key.frequency_penalty_bits,
            presence_penalty_bits: snap.key.presence_penalty_bits,
            min_p_bits: snap.key.min_p_bits,
            tool_call_policy: ToolCallPolicy::default(),
            logprobs: snap.key.logprobs,
            top_logprobs: snap.key.top_logprobs,
            parallel_tool_calls: snap.key.parallel_tool_calls,
        },
        text: snap.text,
        reasoning_text: snap.reasoning_text,
        completion_tokens: snap.completion_tokens,
        reasoning_tokens: snap.reasoning_tokens,
        finish_reason,
        fragments: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_cache(tokens: Vec<u32>, text: &str) -> PromptCache {
        let mut c = PromptCache::new();
        c.tokens = tokens;
        c.text = text.to_string();
        c.completion_tokens = 4;
        c.finish_reason = "length";
        c
    }

    #[test]
    fn empty_cache_yields_none() {
        let c = PromptCache::new();
        assert!(try_serialize(&c).is_none());
    }

    #[test]
    fn round_trip_default_key_byte_recoverable() {
        let original = fresh_cache(vec![1, 2, 3, 4], "hello world");
        let bytes = try_serialize(&original).expect("serialize");
        let restored = try_deserialize(&bytes).expect("deserialize");

        assert_eq!(restored.tokens, original.tokens);
        assert_eq!(restored.text, original.text);
        assert_eq!(restored.completion_tokens, original.completion_tokens);
        assert_eq!(restored.finish_reason, original.finish_reason);
        assert_eq!(restored.key.max_tokens, original.key.max_tokens);
        assert_eq!(
            restored.key.parallel_tool_calls,
            original.key.parallel_tool_calls
        );
        assert!(restored.key.grammar.is_none());
        assert!(restored.fragments.is_none());
    }

    #[test]
    fn unknown_finish_reason_yields_none_on_deserialize() {
        let mut original = fresh_cache(vec![1], "x");
        // SAFETY: we're constructing an in-memory test fixture with a
        // bogus finish_reason to drive the defensive deserialize path.
        // PromptCache.finish_reason is &'static str; we leak a String to
        // produce a 'static reference — leaked memory is acceptable in
        // a #[cfg(test)] unit test.
        let leaked: &'static str = Box::leak(Box::new("bogus".to_string()));
        original.finish_reason = leaked;
        let bytes = try_serialize(&original).expect("serialize");
        let restored = try_deserialize(&bytes);
        assert!(restored.is_none(), "unknown finish_reason should yield None");
    }

    #[test]
    fn version_mismatch_yields_none() {
        let original = fresh_cache(vec![1], "x");
        let bytes = try_serialize(&original).expect("serialize");
        // Hand-edit the JSON to bump the format_version to an unknown value.
        let mut s = String::from_utf8(bytes).expect("utf8");
        s = s.replace(
            &format!("\"format_version\":{}", PROMPT_CACHE_FORMAT_VERSION),
            "\"format_version\":9999",
        );
        let restored = try_deserialize(s.as_bytes());
        assert!(
            restored.is_none(),
            "future-version snapshot should yield None on this hf2q"
        );
    }
}
