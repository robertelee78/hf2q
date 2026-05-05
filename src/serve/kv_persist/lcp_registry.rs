//! ADR-017 Phase E option (a) iter-1 — `LcpRegistry` standalone module.
//!
//! Per the research dossier at
//! `docs/research/adr017-phase-e-option-a-2026-05-05.md` §9 + §10, this
//! module ships the per-`(model_fingerprint, tenant, params)` registry
//! that maps cached prompts to per-layer KV-state Arc handles. It is
//! the load-bearing substrate for iter-3's high-risk
//! `forward_prefill.rs` LCP partial-prefill resume modification.
//!
//! ## What this module does (iter-1 scope)
//!
//! - Stores `(LcpKey, prompt_token_ids, payload, sliding_window,
//!   linear_capacity)` tuples.
//! - On `lookup(key, new_prompt_token_ids)`:
//!   - returns `None` on key mismatch (different fingerprint, tenant, params)
//!   - returns `None` on full equality (`K == new_tokens.len()`) so
//!     Phase E option (b) PromptCache full-equality replay isn't masked
//!   - returns `None` on zero overlap (`K == 0`) so the caller's
//!     "no LCP" path detection is unambiguous
//!   - returns `Some(LcpPrefix { k, dense_kvs, ... })` on partial prefix
//!     match, where `dense_kvs` is a `Vec<Arc<T>>` clone (Arc-pinned
//!     handle that survives concurrent registry eviction — §10.3
//!     Strategy A; §6 R3).
//! - LRU eviction on capacity overflow; `lookup` promotes to MRU.
//!
//! ## What this module does NOT do (deferred to later iters)
//!
//! - **iter-2:** wire the registry into `engine.rs` request flow at the
//!   site identified in dossier §2.1 (after PromptCache full-equality
//!   miss; before fresh-prefill). v2 includes the `soft_tokens.is_empty()`
//!   text-only scope gate from §10.5.
//! - **iter-2.5:** refactor `MlxModelWeights::dense_kvs` from
//!   `Option<Vec<DenseKvBuffers>>` to `Option<Vec<Arc<DenseKvBuffers>>>`
//!   per §10.3 Strategy A.
//! - **iter-3:** modify `forward_prefill.rs:446` wholesale-reset to
//!   conditionally accept `restored_lcp: Option<usize>` and skip the
//!   `[0..K)` token range. Highest-risk iter; requires Codex Phase-2b
//!   audit per project memory `feedback_codex_review_catches_unified_memory_races`.
//!
//! ## Generic over the payload type `T`
//!
//! Production wires `T = DenseKvBuffers` (per dossier §10.3 Strategy A,
//! the registry stores `Vec<Arc<DenseKvBuffers>>` per entry — one
//! `Arc<DenseKvBuffers>` per layer). Unit tests at
//! `tests/lcp_registry_unit.rs` use `T = Vec<u8>` as a marker payload
//! since `MlxBuffer` requires a Metal device that unit tests don't have.
//! The registry is generic over `T: Send + Sync + 'static` to support
//! both regimes.
//!
//! ## LRU implementation
//!
//! Mirrors the `LoadedPool` LRU pattern at `multi_model.rs:225-510`:
//! `Vec<LcpKey>` for `lru_order` + `HashMap<LcpKey, LcpEntry<T>>` for
//! `entries`. LRU = front, MRU = back. Per the project's "no new deps"
//! standing rule (`multi_model.rs:61`), we don't pull in the `lru`
//! crate.
//!
//! ## Concurrency model
//!
//! `LcpRegistry<T>` is **NOT** `Sync`-by-default — `&mut self` on
//! `store` and `lookup` (lookup needs &mut for LRU promotion). The
//! intended concurrency model is: hold a `Mutex<LcpRegistry<T>>` at
//! the engine-level call site (iter-2 wiring); take the lock briefly
//! to do the lookup, clone the Arc handles out, drop the lock, then
//! the in-flight prefill reads the Arc handles without further lock
//! contention. The Arc-pinning invariant (§6 R3) means a concurrent
//! registry eviction during prefill is safe — the in-flight Arc clones
//! keep the payload alive.

use std::collections::HashMap;
use std::sync::Arc;

use crate::serve::kv_persist::format::ModelFingerprint;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Composite key for `LcpRegistry` lookups. Three-axis isolation:
///
/// - `model_fingerprint`: KV state is bound to model weights + RoPE
///   config + dtype (per dossier §3 + §6 R5). Different model →
///   incompatible KV.
/// - `tenant_id`: multi-tenant isolation (§6 R7). Two requests with
///   identical token IDs from different tenants must not cross-hit.
///   Single-tenant deployments use a constant string (e.g. `"default"`).
/// - `params_hash`: opaque hash over sampling params (temperature,
///   top_p, top_k, seed, repetition_penalty, etc.). Different sampling
///   doesn't actually invalidate KV reuse for the prefill phase, BUT
///   sets that affect tokenization (e.g. chat-template variants) DO.
///   Conservative default: include all sampling params.
///
/// `Clone` so callers can build the key once + reuse for `store` then
/// `lookup`. `Hash + Eq` for the underlying `HashMap`.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct LcpKey {
    pub model_fingerprint: ModelFingerprint,
    pub tenant_id: String,
    pub params_hash: u64,
}

/// Result of a successful `LcpRegistry::lookup`.
///
/// `dense_kvs` is the per-layer Arc-pinned payload clone; the caller
/// (iter-3 forward_prefill) reads from this directly and the Arc keeps
/// the payload alive through any concurrent registry eviction (§6 R3).
#[derive(Clone, Debug)]
pub struct LcpPrefix<T> {
    /// Length of the longest common prefix (in tokens). Guaranteed:
    /// `0 < k < new_tokens.len()` and `k <= cached_prompt_len`.
    pub k: usize,
    /// Per-layer payload Arc clones. Length matches the cached
    /// payload's per-layer length (production: `num_layers`).
    pub dense_kvs: Vec<Arc<T>>,
    /// Cached entry's sliding-window size. Used by iter-3 to compute
    /// `min(k, sliding_window)` for the sliding-layer `seq_len`.
    pub sliding_window: usize,
    /// Cached entry's linear-layer capacity. Used by iter-3's three-
    /// case capacity protocol (§10.2): zero-copy if `cached_capacity
    /// >= new_seq_len + max_decode_tokens`; resize-and-copy otherwise.
    pub linear_capacity: usize,
    /// Cached entry's full prompt length (= `cached_prompt.len()`).
    /// Caller can assert `k <= cached_prompt_len` as a defensive
    /// invariant.
    pub cached_prompt_len: usize,
}

/// Errors from `LcpRegistry::store`. `lookup` returns `Option`, not
/// `Result`, because cache misses are normal (not errors).
#[derive(Debug, PartialEq, Eq)]
pub enum LcpStoreError {
    /// Empty prompt (`prompt_tokens.is_empty()`) has no LCP semantics.
    EmptyPrompt,
    /// Empty payload (`dense_kvs.is_empty()`) — every model has ≥1
    /// layer; an empty payload is a misconfiguration.
    EmptyPayload,
}

// ─────────────────────────────────────────────────────────────────────────────
// LcpRegistry<T>
// ─────────────────────────────────────────────────────────────────────────────

/// One entry per `(model_fingerprint, tenant, params)` triple. The
/// caller's prompt is cached in full so subsequent lookups can compute
/// the LCP exactly.
struct LcpEntry<T> {
    /// The full prompt (token IDs) that produced this cached payload.
    /// Lookup compares the new prompt's prefix against this byte-for-
    /// byte (well, token-ID-for-token-ID — see dossier §2.2 for why
    /// token-ID granularity is mandatory).
    prompt: Vec<u32>,
    /// Per-layer Arc-wrapped payload. Length is the model's `num_layers`.
    dense_kvs: Vec<Arc<T>>,
    sliding_window: usize,
    linear_capacity: usize,
}

/// Generic over the cached payload type `T: Send + Sync + 'static`.
/// Production wires `T = DenseKvBuffers`; tests use `T = Vec<u8>`.
pub struct LcpRegistry<T>
where
    T: Send + Sync + 'static,
{
    /// Maximum number of entries. Inserts beyond this evict the LRU
    /// entry first.
    capacity: usize,
    /// Insertion-order vector tracking LRU → MRU. `lru_order[0]` is
    /// the LRU candidate; `lru_order.last()` is the MRU. `lookup` hits
    /// promote the entry to MRU.
    lru_order: Vec<LcpKey>,
    /// Keyed by `LcpKey`; storage for the actual cached state.
    entries: HashMap<LcpKey, LcpEntry<T>>,
}

impl<T> LcpRegistry<T>
where
    T: Send + Sync + 'static,
{
    /// Construct a new registry with the supplied LRU capacity (in
    /// entries, NOT in bytes).
    ///
    /// **Panics** when `capacity == 0` — a zero-capacity registry
    /// would immediately evict every store, which is always a
    /// misconfiguration. Mirrors `LoadedPool` at `multi_model.rs:208-211`.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "LcpRegistry::new(0) is a misconfiguration — every store would \
             immediately evict itself; refusing to construct"
        );
        Self {
            capacity,
            lru_order: Vec::with_capacity(capacity),
            entries: HashMap::with_capacity(capacity),
        }
    }

    /// Number of currently-cached entries (≤ `capacity`).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when the registry has zero entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Store a prompt + payload tuple under `key`. Evicts the LRU
    /// entry if `len() >= capacity` (re-inserting under an existing
    /// key does NOT count toward capacity — it's an overwrite).
    ///
    /// `prompt_tokens` is the full prompt as token-IDs (NOT the
    /// prompt + decoded-output concatenation; see dossier §10.6 R12).
    /// `dense_kvs` is the per-layer payload (one `Arc<T>` per layer,
    /// non-empty).
    ///
    /// Same-key store overwrites the prior payload (matches `PromptCache`
    /// + `BlockPrefixCacheSpiller::register_family` semantics — the
    /// freshest registration wins).
    ///
    /// # Errors
    ///
    /// - [`LcpStoreError::EmptyPrompt`] when `prompt_tokens.is_empty()`.
    /// - [`LcpStoreError::EmptyPayload`] when `dense_kvs.is_empty()`.
    pub fn store(
        &mut self,
        key: LcpKey,
        prompt_tokens: Vec<u32>,
        dense_kvs: Vec<Arc<T>>,
        sliding_window: usize,
        linear_capacity: usize,
    ) -> Result<(), LcpStoreError> {
        if prompt_tokens.is_empty() {
            return Err(LcpStoreError::EmptyPrompt);
        }
        if dense_kvs.is_empty() {
            return Err(LcpStoreError::EmptyPayload);
        }

        let entry = LcpEntry {
            prompt: prompt_tokens,
            dense_kvs,
            sliding_window,
            linear_capacity,
        };

        // Re-insert path: overwrite + promote to MRU. Doesn't grow
        // size; doesn't trigger eviction of OTHER entries because the
        // total entry count is unchanged.
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), entry);
            // Promote to MRU.
            if let Some(pos) = self.lru_order.iter().position(|k| k == &key) {
                let k = self.lru_order.remove(pos);
                self.lru_order.push(k);
            }
            return Ok(());
        }

        // Capacity pass: evict LRU if at capacity.
        while self.entries.len() >= self.capacity {
            if let Some(victim) = self.lru_order.first().cloned() {
                self.entries.remove(&victim);
                self.lru_order.remove(0);
            } else {
                // Defensive: lru_order out-of-sync with entries.
                break;
            }
        }

        self.entries.insert(key.clone(), entry);
        self.lru_order.push(key);
        Ok(())
    }

    /// Look up the longest common prefix between `new_tokens` and the
    /// cached prompt under `key`.
    ///
    /// Returns `Some(LcpPrefix)` iff:
    /// - An entry exists for `key`.
    /// - The token-ID prefix length `k` satisfies `0 < k < new_tokens.len()`
    ///   AND `k <= cached.prompt.len()`. (Strictly partial prefix; full
    ///   equality returns None so Phase E option (b) PromptCache replay
    ///   isn't masked. Zero overlap returns None so the caller's "no
    ///   LCP" path detection is unambiguous.)
    ///
    /// On a hit, the entry is promoted to MRU.
    ///
    /// **The returned `LcpPrefix.dense_kvs` is a `Vec<Arc<T>>` clone
    /// — `Arc::clone` per layer.** This means a concurrent
    /// `clear()` / `store()` that evicts this entry from the registry
    /// will NOT free the underlying payload until the caller drops
    /// the returned `Arc<T>` clones. This is the load-bearing safety
    /// guarantee for iter-3's in-flight prefill (§6 R3).
    pub fn lookup(&mut self, key: &LcpKey, new_tokens: &[u32]) -> Option<LcpPrefix<T>> {
        let entry = self.entries.get(key)?;

        // Compute the longest common token-ID prefix.
        let cached = &entry.prompt[..];
        let max_compare = cached.len().min(new_tokens.len());
        let mut k = 0usize;
        while k < max_compare && cached[k] == new_tokens[k] {
            k += 1;
        }

        // §2.3 line 172 — gate out 0-overlap and full-equality.
        if k == 0 || k == new_tokens.len() {
            return None;
        }

        // Build the result (Arc-clones per layer — the load-bearing
        // pinning invariant).
        let result = LcpPrefix {
            k,
            dense_kvs: entry.dense_kvs.iter().map(Arc::clone).collect(),
            sliding_window: entry.sliding_window,
            linear_capacity: entry.linear_capacity,
            cached_prompt_len: entry.prompt.len(),
        };

        // Promote to MRU.
        if let Some(pos) = self.lru_order.iter().position(|kk| kk == key) {
            let touched = self.lru_order.remove(pos);
            self.lru_order.push(touched);
        }

        Some(result)
    }

    /// Drop all entries. Used by tests and by future iter-2 wiring on
    /// model reload (when the registry's cached payloads become
    /// invalid — e.g. weights file changed). The underlying payloads
    /// remain alive as long as in-flight callers hold their `Arc<T>`
    /// clones — see §6 R3.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.lru_order.clear();
    }

    /// Configured LRU capacity. Diagnostic accessor; the registry
    /// enforces the bound automatically on `store`.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> std::fmt::Debug for LcpRegistry<T>
where
    T: Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LcpRegistry")
            .field("capacity", &self.capacity)
            .field("len", &self.entries.len())
            .field("lru_order_len", &self.lru_order.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// iter-2 — engine-side detection helper.
// ---------------------------------------------------------------------------

/// ADR-017 Phase E.a iter-2 — engine-side LCP detection probe.
///
/// Given a registry, key, new prompt tokens, and a multimodal flag,
/// returns `Some(K)` where `0 < K < new_tokens.len()` if a partial-
/// prefix resume opportunity exists, else `None`. Used by
/// `engine.rs::generate_once_with_soft_tokens` immediately AFTER the
/// `PromptCache` full-equality miss (so E.b's win path is preserved)
/// and BEFORE `forward_prefill_with_soft_tokens` so the iter-3
/// partial-prefill modification has a single, named decision point.
///
/// ## Multimodal gate (dossier §10.5)
///
/// `has_soft_tokens == true` ⇒ always `None`. The cached KV state was
/// generated under text-only embedding-lookup; replaying its prefix
/// into a request whose first-N tokens have per-position soft-token
/// overrides (vision / audio / deepstack) would silently corrupt
/// outputs because the cached `[0..K)` KV state would not reflect the
/// new request's per-position soft-token deltas. The bail is a hard
/// precondition, not a hint.
///
/// ## Why a separate function (vs a method on `LcpRegistry`)
///
/// 1. **Engine call site has the multimodal flag, registry doesn't.**
///    `LcpRegistry` is payload-agnostic; threading `has_soft_tokens`
///    into `lookup` would couple the registry to the engine's
///    multimodal scope. Keeping the gate in a free function lets the
///    registry stay generic.
/// 2. **Iter-2 ships an observability layer.** The probe returns
///    `Option<usize>` (just K) — it never hands out the full
///    `LcpPrefix<T>` payload, because iter-2 doesn't yet pre-warm
///    `dense_kvs`. Iter-3 will swap callers to `LcpRegistry::lookup`
///    directly to get the Arc-cloned payload.
///
/// Returns the LCP length `K` (1-indexed token count) on hit, `None`
/// on any of: zero overlap (`K == 0`), full equality (`K == N`),
/// fingerprint/tenant/params mismatch, or `has_soft_tokens == true`.
///
/// **Why `&mut`** — `LcpRegistry::lookup` is `&mut self` (it promotes
/// the hit entry to MRU on each access). The probe inherits that.
/// Engine sites already hold `loaded.lcp_registry` mutably (the worker
/// thread is the sole owner per the `LoadedModel` ownership model
/// mirrored from `prompt_cache`).
pub fn probe_lcp_opportunity<T>(
    registry: &mut LcpRegistry<T>,
    key: &LcpKey,
    new_tokens: &[u32],
    has_soft_tokens: bool,
) -> Option<usize>
where
    T: Send + Sync + 'static,
{
    if has_soft_tokens {
        // Multimodal request — never probe LCP. The cached KV state
        // was text-only; per-position soft-token overrides would
        // diverge under partial-prefill resume.
        return None;
    }
    registry.lookup(key, new_tokens).map(|prefix| prefix.k)
}
