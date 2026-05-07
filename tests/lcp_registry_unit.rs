//! ADR-017 Phase E option (a) iter-1 — `LcpRegistry` unit tests.
//!
//! Per the research dossier at
//! `docs/research/adr017-phase-e-option-a-2026-05-05.md` §9 ("If green-lit,
//! here's iter-1's plan"), iter-1 ships the registry as a STANDALONE
//! struct with zero changes to `forward_prefill.rs` or `engine.rs`. These
//! 8 unit tests pin the correctness invariants identified in §4.1 + §6
//! R3 + §10.6 (R10/R11/R12) so that iter-3's high-risk
//! `forward_prefill.rs` modification has a verified foundation.
//!
//! Walk-discipline marker: tests are written BEFORE the registry impl
//! per the "test BEFORE code" rule (Phase A0 falsification gate
//! discipline applied at the iter-1 unit-test level).
//!
//! The registry is generic over the cached payload type `T: Send + Sync +
//! 'static + ByteSized`. Production wires `T = DenseKvBuffers` (per dossier
//! §10.3 Strategy A); these tests use `T = Marker` (a newtype over `Vec<u8>`)
//! as a marker payload since `MlxBuffer` requires a Metal device. The orphan
//! rule prohibits `impl ByteSized for Vec<u8>` (Vec is foreign), so the
//! newtype is necessary. The Arc-pin invariant (§6 R3) is testable without
//! Metal because `Arc::strong_count` is payload-agnostic.

use hf2q::serve::kv_persist::lcp_registry::{
    ByteSized, LcpKey, LcpRegistry, LcpStoreError, default_lcp_byte_budget,
};
use hf2q::serve::kv_persist::format::ModelFingerprint;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Marker payload for unit tests.
//
// Production types use MlxBuffer::byte_len / HybridKvCacheSnapshot::total_bytes.
// Unit tests cannot instantiate MlxBuffer (requires a Metal device), so we use
// a newtype wrapper `Marker(Vec<u8>)` whose ByteSized impl reports the inner
// Vec's length. The orphan rule prohibits `impl ByteSized for Vec<u8>` (Vec is
// foreign), so the newtype is the correct local-type solution.
// ─────────────────────────────────────────────────────────────────────────────

/// Newtype marker payload for unit tests. Byte size == inner Vec length.
#[derive(Clone, Debug)]
pub struct Marker(pub Vec<u8>);

impl ByteSized for Marker {
    fn byte_len(&self) -> u64 {
        self.0.len() as u64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test fixtures
// ─────────────────────────────────────────────────────────────────────────────

fn fp(b: u8) -> ModelFingerprint {
    ModelFingerprint([b; 32])
}

fn key(fp_byte: u8, tenant: &str, params: u64) -> LcpKey {
    LcpKey {
        model_fingerprint: fp(fp_byte),
        tenant_id: tenant.to_string(),
        params_hash: params,
    }
}

/// Synthetic per-layer payload — replaces `Vec<DenseKvBuffers>` for
/// unit tests. Length = num_layers (mirrors production semantics).
/// Each `Marker` wraps a 2-byte Vec, so `byte_len() == 2` per layer.
fn payload(num_layers: usize, marker: u8) -> Vec<Arc<Marker>> {
    (0..num_layers)
        .map(|li| Arc::new(Marker(vec![marker, li as u8])))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — 8 from dossier §9 (with #1 reinterpreted per §10.4 — "full
// equality returns None" is the spec, so test #1 verifies that
// behavior, eliminating the §9 contradiction).
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn lcp_registry_returns_partial_match() {
    // Dossier §9 test #2. The load-bearing user case: turn N's prompt
    // shares a prefix with the cached prompt; LCP returns Some with k =
    // length of common prefix.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let k = key(0xAA, "tenant-a", 42);
    let cached_prompt: Vec<u32> = (1..=20).collect();
    let pl = payload(4, 0x11);

    reg.store(k.clone(), cached_prompt.clone(), pl.clone(), 4096, 4096)
        .expect("store");

    // New prompt: shares first 12 tokens, diverges at position 12.
    let new_prompt: Vec<u32> = (1..=12).chain(900..=905).collect();
    let hit = reg
        .lookup(&k, &new_prompt)
        .expect("partial match must hit");

    assert_eq!(hit.k, 12, "LCP=12 expected (1..=12 shared)");
    assert_eq!(hit.dense_kvs.len(), 4, "per-layer payload preserved");
    // Arc identity: handed-out Arc points to same allocation as cached.
    assert!(
        Arc::ptr_eq(&hit.dense_kvs[0], &pl[0]),
        "lookup must return Arc-clone of cached payload (zero-copy hand-off)"
    );
}

#[test]
fn lcp_registry_returns_none_on_full_equality() {
    // Dossier §9 test #8 (replaces the contradictory #1). Full equality
    // must NEVER mask Phase E option (b) PromptCache replay; LCP path
    // returns None when K == new_tokens.len() so the upstream caller
    // proceeds to PromptCache full-equality lookup OR fresh prefill.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let k = key(0xAA, "tenant-a", 42);
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    reg.store(k.clone(), prompt.clone(), payload(2, 0xCC), 4096, 4096)
        .expect("store");

    let hit = reg.lookup(&k, &prompt);
    assert!(
        hit.is_none(),
        "K == new_tokens.len() must return None — never mask option (b)"
    );
}

#[test]
fn lcp_registry_returns_none_on_zero_overlap() {
    // Defensive: prompts with no shared prefix must not produce a 0-byte
    // "match" (which would race with the upstream caller's "no LCP" path
    // detection). §2.3 line 172 — `if k == 0 ... { return None; }`.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let k = key(0xAA, "tenant-a", 42);
    reg.store(k.clone(), vec![1, 2, 3], payload(2, 0xCC), 4096, 4096)
        .expect("store");

    // Different first token = zero overlap.
    let hit = reg.lookup(&k, &[999, 998, 997]);
    assert!(hit.is_none(), "zero-overlap must return None");
}

#[test]
fn lcp_registry_returns_none_on_fingerprint_mismatch() {
    // Dossier §6 R5 + §10.6 R12 — model fingerprint MUST match for KV
    // reuse to be sound. Same prompt + same tenant + same params, but
    // different model = miss.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let store_key = key(0xAA, "tenant-a", 42);
    let lookup_key = key(0xBB, "tenant-a", 42); // different fingerprint

    reg.store(
        store_key,
        vec![1, 2, 3, 4, 5],
        payload(2, 0xCC),
        4096,
        4096,
    )
    .expect("store");

    let hit = reg.lookup(&lookup_key, &[1, 2, 3]);
    assert!(
        hit.is_none(),
        "fingerprint mismatch must return None (KV state is model-bound)"
    );
}

#[test]
fn lcp_registry_returns_none_on_tenant_mismatch() {
    // Dossier §6 R7 — multi-tenant isolation. Two requests with
    // identical token IDs but different tenant_ids must NOT cross-hit.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let store_key = key(0xAA, "tenant-a", 42);
    let lookup_key = key(0xAA, "tenant-b", 42); // different tenant

    reg.store(
        store_key,
        vec![1, 2, 3, 4, 5],
        payload(2, 0xCC),
        4096,
        4096,
    )
    .expect("store");

    let hit = reg.lookup(&lookup_key, &[1, 2, 3]);
    assert!(
        hit.is_none(),
        "tenant mismatch must return None (multi-tenant isolation)"
    );
}

#[test]
fn lcp_registry_returns_none_on_params_mismatch() {
    // Dossier §3.3 — sampling params are part of the byte-identity
    // precondition. Different temperature / top_p / seed / etc. → miss
    // even though prefix tokens match.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let store_key = key(0xAA, "tenant-a", 42);
    let lookup_key = key(0xAA, "tenant-a", 99); // different params_hash

    reg.store(
        store_key,
        vec![1, 2, 3, 4, 5],
        payload(2, 0xCC),
        4096,
        4096,
    )
    .expect("store");

    let hit = reg.lookup(&lookup_key, &[1, 2, 3]);
    assert!(
        hit.is_none(),
        "params_hash mismatch must return None"
    );
}

#[test]
fn lcp_registry_pins_dense_kvs_arc_across_eviction() {
    // Dossier §6 R3 + §10.3 — concurrent eviction race. The handed-out
    // Arc<DenseKvBuffers> from `lookup` must keep the underlying
    // payload alive even AFTER the registry entry is evicted. This is
    // the load-bearing safety guarantee for iter-3's
    // forward_prefill.rs LCP path: between `lookup` and the prefill's
    // last KV read, the registry could be cleared by a concurrent
    // model-eviction trigger, but the in-flight prefill's Arc clone
    // must stay valid until prefill completes.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(2); // small capacity → easy eviction
    let k = key(0xAA, "tenant-a", 42);
    let pl = payload(2, 0x55);

    reg.store(k.clone(), vec![1, 2, 3, 4, 5], pl.clone(), 4096, 4096)
        .expect("store");

    // strong_count = 2 (caller's `pl[0]` + registry's clone of pl[0]).
    assert_eq!(Arc::strong_count(&pl[0]), 2);

    // Lookup with prefix [1,2,3] PLUS a divergent suffix token so we
    // get a partial match (k=3, not full equality). Per registry
    // contract, k == new_tokens.len() returns None.
    let hit = reg
        .lookup(&k, &[1, 2, 3, 999])
        .expect("must hit (partial match k=3, divergent at 4)");
    let pinned: Arc<Marker> = Arc::clone(&hit.dense_kvs[0]);
    // Now strong_count = 4 (caller `pl[0]` + registry + `hit.dense_kvs[0]` + `pinned`).
    assert_eq!(Arc::strong_count(&pinned), 4, "pl + reg + hit + pinned");

    // Drop the lookup result (the in-flight prefill would do this once
    // it's done reading); only `pinned` remains as the "in-flight"
    // proxy. strong_count = 3 (caller + registry + pinned).
    drop(hit);
    assert_eq!(Arc::strong_count(&pinned), 3, "after dropping hit: pl + reg + pinned");

    // Now clear the registry — simulating concurrent eviction during
    // an in-flight prefill.
    reg.clear();
    // Registry's clone dropped; caller's `pl[0]` + `pinned` remain.
    assert_eq!(
        Arc::strong_count(&pinned),
        2,
        "Arc handed-out by lookup() must survive registry.clear()"
    );

    // Drop caller's local `pl` too.
    drop(pl);
    assert_eq!(
        Arc::strong_count(&pinned),
        1,
        "pinned is the last live Arc — payload still valid for in-flight read"
    );
}

#[test]
fn lcp_registry_evicts_lru_on_capacity_pressure() {
    // Dossier §6 R10 + §10.7 — LRU eviction enforces a bounded
    // memory footprint. With capacity = 2, inserting a 3rd entry
    // evicts the LRU one.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(2);
    let k1 = key(0x11, "t", 1);
    let k2 = key(0x22, "t", 2);
    let k3 = key(0x33, "t", 3);

    reg.store(k1.clone(), vec![1, 2], payload(1, 0xA1), 4096, 4096)
        .expect("store k1");
    reg.store(k2.clone(), vec![3, 4], payload(1, 0xA2), 4096, 4096)
        .expect("store k2");
    assert_eq!(reg.len(), 2);

    // Insert k3 → k1 (LRU) evicted.
    reg.store(k3.clone(), vec![5, 6], payload(1, 0xA3), 4096, 4096)
        .expect("store k3");
    assert_eq!(reg.len(), 2, "capacity bounded at 2");

    // k1 is gone; k2 + k3 remain. Use partial-prefix lookups (cached
    // prompt + divergent suffix) so we don't trip full-equality gate.
    assert!(reg.lookup(&k1, &[1, 999]).is_none(), "k1 LRU-evicted");
    assert!(reg.lookup(&k2, &[3, 999]).is_some(), "k2 still cached");
    assert!(reg.lookup(&k3, &[5, 999]).is_some(), "k3 just inserted");
}

#[test]
fn lcp_registry_lookup_promotes_to_mru() {
    // Implementation detail of LRU: lookup hits must promote the entry
    // to MRU so they don't get evicted while in use. Mirrors the
    // LoadedPool::touch semantic at multi_model.rs:415-419.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(2);
    let k1 = key(0x11, "t", 1);
    let k2 = key(0x22, "t", 2);
    let k3 = key(0x33, "t", 3);

    reg.store(k1.clone(), vec![1, 2], payload(1, 0xA1), 4096, 4096)
        .expect("store k1");
    reg.store(k2.clone(), vec![3, 4], payload(1, 0xA2), 4096, 4096)
        .expect("store k2");

    // Touch k1 to promote it to MRU.
    let _ = reg.lookup(&k1, &[1, 999]);

    // Insert k3 → LRU is now k2 (not k1, because lookup promoted k1).
    reg.store(k3.clone(), vec![5, 6], payload(1, 0xA3), 4096, 4096)
        .expect("store k3");

    assert!(reg.lookup(&k1, &[1, 999]).is_some(), "k1 survived (was MRU)");
    assert!(reg.lookup(&k2, &[3, 999]).is_none(), "k2 LRU-evicted");
    assert!(reg.lookup(&k3, &[5, 999]).is_some(), "k3 just inserted");
}

#[test]
fn lcp_registry_store_rejects_zero_capacity_construction() {
    // Defensive: capacity=0 is a misconfiguration and would mean every
    // store immediately evicts itself. Reject at construction time.
    let result = std::panic::catch_unwind(|| LcpRegistry::<Marker>::new(0));
    assert!(
        result.is_err(),
        "LcpRegistry::new(0) must panic — capacity=0 is a misconfiguration"
    );
}

#[test]
fn lcp_registry_store_overwrites_same_key() {
    // Storing under an existing key replaces the prior payload (matches
    // PromptCache + spiller register_family overwrite semantics: the
    // freshest store wins).
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(4);
    let k = key(0xAA, "t", 1);

    let pl_v1 = payload(1, 0xA1);
    let pl_v2 = payload(1, 0xA2);

    reg.store(k.clone(), vec![1, 2, 3], pl_v1.clone(), 4096, 4096)
        .expect("store v1");
    reg.store(k.clone(), vec![1, 2, 3, 4, 5], pl_v2.clone(), 4096, 4096)
        .expect("store v2 (same key, overwrite)");

    assert_eq!(reg.len(), 1, "same-key overwrite must not grow size");

    // Use [1,2,3,4,999] so k=4 partial-prefix (not k=4 full-equality).
    let hit = reg.lookup(&k, &[1, 2, 3, 4, 999]).expect("must hit v2");
    assert_eq!(hit.k, 4, "v2's prompt[1,2,3,4,5] vs lookup [1,2,3,4,999] → LCP=4");
    assert!(
        Arc::ptr_eq(&hit.dense_kvs[0], &pl_v2[0]),
        "lookup returns v2's payload (overwrite-wins)"
    );
}

#[test]
fn lcp_registry_store_validates_layer_count_invariant() {
    // The cached payload's per-layer Vec length is later read by
    // forward_prefill (one Arc per layer). Storing a payload with a
    // different per-layer length than the model's num_layers would
    // panic at iter-3 prefill time. Detect at store() time with an
    // explicit error.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(4);
    let k = key(0xAA, "t", 1);

    // Empty payload (0 layers) is invalid — every Gemma 4 model has
    // ≥1 layer. The registry doesn't know the model's num_layers, but
    // it MUST reject empty payload as a sanity floor.
    let err = reg
        .store(k.clone(), vec![1, 2, 3], vec![], 4096, 4096)
        .expect_err("empty payload must reject");
    assert!(matches!(err, LcpStoreError::EmptyPayload));
}

#[test]
fn lcp_registry_store_rejects_empty_prompt() {
    // A 0-token prompt has no LCP semantics — reject at store time.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(4);
    let k = key(0xAA, "t", 1);

    let err = reg
        .store(k, vec![], payload(1, 0xCC), 4096, 4096)
        .expect_err("empty prompt must reject");
    assert!(matches!(err, LcpStoreError::EmptyPrompt));
}

// ─────────────────────────────────────────────────────────────────────────────
// Iter-2 — `probe_lcp_opportunity` engine-side detection helper.
//
// These three tests pin the iter-2 gating contract: the engine.rs probe
// returns `None` for any non-text-only request (multimodal soft_tokens
// present) so the iter-3 partial-prefill path NEVER fires for vision
// inputs, even when the cached LCP exists. Per dossier §10.5 — the
// multimodal bail is a hard precondition, not a hint.
//
// Walk-discipline marker: tests are written BEFORE the helper impl.
// Each test pins one independent invariant.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn probe_lcp_opportunity_gates_off_multimodal() {
    // ADR-017 Phase E.a §10.5: requests with non-empty `soft_tokens`
    // (vision / audio / per-position embedding overrides) MUST short-
    // circuit to None even when a partial-prefix would otherwise hit.
    // Reason: the cached KV state was generated under text-only
    // embedding-lookup; replaying its prefix into a request whose
    // first-N tokens have per-position overrides would silently
    // corrupt outputs.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(4);
    let k = key(0xAA, "t", 0);
    let cached_prompt: Vec<u32> = (1..=10).collect();
    let pl = payload(2, 0x55);
    reg.store(k.clone(), cached_prompt.clone(), pl, 4096, 4096)
        .expect("store");

    // Same prompt, same key: a text-only probe would return Some(K) — but the
    // multimodal flag must override.
    let new_prompt: Vec<u32> = (1..=8).chain([900u32].iter().copied()).collect();
    use hf2q::serve::kv_persist::lcp_registry::probe_lcp_opportunity;

    let multimodal = probe_lcp_opportunity(&mut reg, &k, &new_prompt, /*has_soft_tokens=*/ true);
    assert_eq!(
        multimodal, None,
        "multimodal request must NEVER probe LCP, even on cache-prone prompt"
    );

    // Sanity: same call with multimodal=false DOES hit, proving the gate
    // (not absence of cache) caused the None above.
    let text_only = probe_lcp_opportunity(&mut reg, &k, &new_prompt, /*has_soft_tokens=*/ false);
    assert_eq!(
        text_only,
        Some(8),
        "text-only probe on same prompt MUST hit (proves multimodal gate caused first None)"
    );
}

#[test]
fn probe_lcp_opportunity_returns_none_on_full_equality() {
    // Iter-2 contract: full equality is the existing `PromptCache`
    // (Phase E.b) win — `probe_lcp_opportunity` MUST NOT mask that path.
    // The registry's `lookup` already enforces this (returns None when
    // K == new_tokens.len()); the helper just preserves that semantic.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(4);
    let k = key(0xAA, "t", 0);
    let cached: Vec<u32> = (1..=10).collect();
    reg.store(k.clone(), cached.clone(), payload(2, 0x66), 4096, 4096)
        .expect("store");

    use hf2q::serve::kv_persist::lcp_registry::probe_lcp_opportunity;

    // Identical prompt — full equality.
    let probe_full =
        probe_lcp_opportunity(&mut reg, &k, &cached, /*has_soft_tokens=*/ false);
    assert_eq!(
        probe_full, None,
        "full-equality lookup must yield None — PromptCache (E.b) handles that case"
    );
}

#[test]
fn probe_lcp_opportunity_returns_lcp_k_on_partial_match() {
    // Iter-2 happy path: partial overlap returns `Some(K)` so the
    // engine can bump `kv_lcp_detected_total` for observability.
    // K equals the length of the longest common token-id prefix.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(4);
    let k = key(0xAA, "t", 0);
    let cached: Vec<u32> = (10..=29).collect(); // 20 tokens
    reg.store(k.clone(), cached.clone(), payload(2, 0x77), 4096, 4096)
        .expect("store");

    use hf2q::serve::kv_persist::lcp_registry::probe_lcp_opportunity;

    // Diverge at position 7 (token 17 → 999).
    let new_prompt: Vec<u32> = (10..=16).chain([999u32].iter().copied()).collect();
    let detected =
        probe_lcp_opportunity(&mut reg, &k, &new_prompt, /*has_soft_tokens=*/ false);
    assert_eq!(
        detected,
        Some(7),
        "partial match must return Some(K=7) — 7 shared prefix tokens"
    );

    // Sanity: zero-overlap returns None (still respects K==0 ⇒ no opportunity).
    let no_overlap: Vec<u32> = vec![1000, 1001, 1002];
    let nope = probe_lcp_opportunity(&mut reg, &k, &no_overlap, /*has_soft_tokens=*/ false);
    assert_eq!(
        nope, None,
        "zero-overlap prompt must yield None — no resume opportunity"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Iter-3 — `take_prefix` consuming-lookup tests.
//
// These pin the contract that take_prefix removes the entry on hit
// (so caller becomes sole Arc holder, ready for in-place mutation
// during partial-prefill resume) and is a no-op on miss.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn take_prefix_returns_same_k_as_lookup_on_hit() {
    // Iter-3 contract: take_prefix is a consuming variant; on hit it
    // returns the SAME LcpPrefix shape as lookup.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let k = key(0xCC, "tenant", 7);
    let cached: Vec<u32> = (1..=20).collect();
    let pl = payload(2, 0xAB);
    reg.store(k.clone(), cached.clone(), pl, 4096, 4096)
        .expect("store");

    // Diverging suffix at position 12.
    let new_prompt: Vec<u32> = (1..=12).chain([900u32, 901].iter().copied()).collect();
    let prefix = reg
        .take_prefix(&k, &new_prompt)
        .expect("take_prefix must hit on partial overlap");
    assert_eq!(prefix.k, 12, "K must equal lookup's K");
    assert_eq!(prefix.dense_kvs.len(), 2, "per-layer payload preserved");
}

#[test]
fn take_prefix_removes_entry_so_subsequent_take_misses() {
    // Iter-3 contract: after take_prefix succeeds, the entry is gone
    // from the registry. A second take with the same prompt must miss.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let k = key(0xDD, "tenant", 7);
    let cached: Vec<u32> = (1..=10).collect();
    let pl = payload(2, 0xCD);
    reg.store(k.clone(), cached.clone(), pl, 4096, 4096)
        .expect("store");

    let new_prompt: Vec<u32> = (1..=8).chain([999u32].iter().copied()).collect();

    // First take hits.
    assert!(
        reg.take_prefix(&k, &new_prompt).is_some(),
        "first take_prefix must hit"
    );

    // Second take on same key + same prompt now misses (entry consumed).
    assert!(
        reg.take_prefix(&k, &new_prompt).is_none(),
        "second take_prefix on same prompt must miss after consume"
    );

    // Registry size confirms removal.
    assert_eq!(reg.len(), 0, "registry must be empty after take");
}

#[test]
fn take_prefix_no_op_on_miss() {
    // Iter-3 safety: take_prefix on miss must NOT touch registry state.
    // Otherwise a probe-and-take sequence on a key that doesn't fit
    // could spuriously evict an unrelated entry.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(8);
    let k_a = key(0xAA, "tenant", 7);
    let k_b = key(0xBB, "tenant", 7); // different fingerprint
    reg.store(k_a.clone(), vec![1, 2, 3], payload(2, 0xCD), 4096, 4096)
        .expect("store a");

    // Try take_prefix with key_b (no entry exists for k_b).
    let miss = reg.take_prefix(&k_b, &[1, 2, 3]);
    assert!(miss.is_none(), "take on missing key must be None");
    assert_eq!(reg.len(), 1, "registry must keep the unrelated entry intact");

    // Sanity: lookup against k_a still hits (entry preserved).
    let hit = reg.lookup(&k_a, &[1, 2, 999]);
    assert!(
        hit.is_some(),
        "unrelated entry must remain after miss-take on different key"
    );
}

#[test]
fn take_prefix_drops_caller_arc_to_strong_count_one() {
    // Iter-3 load-bearing invariant: after take_prefix, the per-layer
    // Arc<T> clones in the returned LcpPrefix have strong_count == 1
    // (caller is sole holder). This is what enables `Arc::get_mut`
    // / move-into-weights / in-place mutation during partial-prefill
    // resume.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::new(4);
    let k = key(0xEE, "tenant", 7);
    let cached: Vec<u32> = (1..=10).collect();

    // Build payload with our own clone we drop before strong-count check.
    let pl_orig: Vec<Arc<Marker>> = (0..3)
        .map(|i| Arc::new(Marker(vec![0x77, i as u8])))
        .collect();
    // Clone to keep originals' counts as registry-side reference only.
    let pl_for_store: Vec<Arc<Marker>> = pl_orig.iter().map(Arc::clone).collect();
    reg.store(k.clone(), cached.clone(), pl_for_store, 4096, 4096)
        .expect("store");

    // Drop caller's originals so registry holds the only Arc per layer.
    drop(pl_orig);

    let new_prompt: Vec<u32> = (1..=8).chain([42u32].iter().copied()).collect();
    let prefix = reg
        .take_prefix(&k, &new_prompt)
        .expect("take_prefix must hit");

    // Each per-layer Arc should now have strong_count == 1 (only the
    // returned prefix holds it; registry dropped its set during take).
    for (layer_idx, arc) in prefix.dense_kvs.iter().enumerate() {
        let count = Arc::strong_count(arc);
        assert_eq!(
            count, 1,
            "per-layer Arc after take_prefix must have strong_count=1 (layer {} got {})",
            layer_idx, count
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte-budget tests — ADR-017 Phase E.a default-on
//
// These 5 tests cover the byte-budget LRU model introduced by the
// `with_byte_budget` constructor. The marker payload is `Vec<u8>`; byte_len
// is `Vec::len`, so we control exact byte counts by controlling Vec length.
// ─────────────────────────────────────────────────────────────────────────────

/// Build a single-layer payload whose total byte count equals `bytes`.
/// The single Arc<Marker> wraps a Marker(Vec<u8>) of `bytes` zeros.
fn payload_bytes(bytes: usize) -> Vec<Arc<Marker>> {
    vec![Arc::new(Marker(vec![0u8; bytes]))]
}

#[test]
fn byte_budget_evicts_lru_when_second_entry_exceeds_budget() {
    // (a) Budget = 1024 bytes. Insert two entries of 600 bytes each.
    // The second insert must evict the first (600 + 600 > 1024).
    // After insert: current_bytes == 600, only the second entry remains.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::with_byte_budget(1024);
    let k1 = key(0x11, "t", 1);
    let k2 = key(0x22, "t", 2);

    reg.store(k1.clone(), vec![1, 2], payload_bytes(600), 4096, 4096)
        .expect("store k1");
    assert_eq!(reg.current_bytes(), 600, "first insert: current_bytes == 600");
    assert_eq!(reg.len(), 1);

    reg.store(k2.clone(), vec![3, 4], payload_bytes(600), 4096, 4096)
        .expect("store k2 — must evict k1");
    assert_eq!(
        reg.current_bytes(),
        600,
        "after k2 insert: k1 evicted → current_bytes == 600"
    );
    assert_eq!(reg.len(), 1, "only k2 remains");

    // k1 evicted — use partial-prefix lookup (prompt [3, 999] vs cached [3, 4]).
    assert!(
        reg.lookup(&k1, &[1, 999]).is_none(),
        "k1 must have been evicted"
    );
    assert!(
        reg.lookup(&k2, &[3, 999]).is_some(),
        "k2 must still be present"
    );
}

#[test]
fn byte_budget_evicts_lru_under_varying_sizes() {
    // (b) Budget = 500 bytes. Insert entries of sizes [100, 200, 50, 800].
    //
    // Step 1: insert 100 → current_bytes = 100, fits.
    // Step 2: insert 200 → current_bytes = 300, fits (100+200=300 ≤ 500).
    // Step 3: insert 50  → current_bytes = 350, fits (300+50=350 ≤ 500).
    // Step 4: insert 800 → 800 > budget(500) → EntryExceedsBudget error.
    //
    // After all steps: 3 entries remain, k4 rejected; current_bytes == 350.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::with_byte_budget(500);
    let k1 = key(0x11, "t", 1);
    let k2 = key(0x22, "t", 2);
    let k3 = key(0x33, "t", 3);
    let k4 = key(0x44, "t", 4);

    reg.store(k1.clone(), vec![1, 2], payload_bytes(100), 4096, 4096)
        .expect("store 100 bytes");
    reg.store(k2.clone(), vec![3, 4], payload_bytes(200), 4096, 4096)
        .expect("store 200 bytes");
    reg.store(k3.clone(), vec![5, 6], payload_bytes(50), 4096, 4096)
        .expect("store 50 bytes");

    assert_eq!(reg.current_bytes(), 350);
    assert_eq!(reg.len(), 3);

    // k4 with 800 bytes > budget(500) — must return EntryExceedsBudget.
    let err = reg
        .store(k4.clone(), vec![7, 8], payload_bytes(800), 4096, 4096)
        .expect_err("entry larger than budget must error");
    assert!(
        matches!(
            err,
            LcpStoreError::EntryExceedsBudget {
                entry_bytes: 800,
                budget_bytes: 500
            }
        ),
        "unexpected error variant: {:?}",
        err
    );

    // Registry unchanged after the rejected insert.
    assert_eq!(reg.current_bytes(), 350, "current_bytes unchanged after rejection");
    assert_eq!(reg.len(), 3, "entry count unchanged after rejection");
}

#[test]
fn byte_budget_returns_err_when_entry_exceeds_entire_budget() {
    // (c) Entry larger than the entire budget → Err(EntryExceedsBudget).
    // The registry must NOT silently no-op. Even an empty registry returns
    // this error — the entry can never fit.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::with_byte_budget(100);
    let k = key(0xAA, "t", 1);

    let err = reg
        .store(k, vec![1, 2], payload_bytes(200), 4096, 4096)
        .expect_err("entry (200 bytes) > budget (100 bytes) must error");
    assert!(
        matches!(
            err,
            LcpStoreError::EntryExceedsBudget {
                entry_bytes: 200,
                budget_bytes: 100
            }
        ),
        "unexpected error variant: {:?}",
        err
    );
    assert_eq!(reg.current_bytes(), 0, "registry must be empty after failed insert");
    assert_eq!(reg.len(), 0);
}

#[test]
fn byte_budget_same_key_reinsert_no_double_count() {
    // (d) Same-key reinsert with different byte sizes must not double-count.
    // After overwrite the current_bytes must equal only the NEW entry's size.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::with_byte_budget(1024);
    let k = key(0xBB, "t", 1);

    // First insert: 300 bytes.
    reg.store(k.clone(), vec![1, 2, 3], payload_bytes(300), 4096, 4096)
        .expect("store v1");
    assert_eq!(reg.current_bytes(), 300, "after v1 insert: 300 bytes");

    // Overwrite with 150 bytes (different-sized payload on same key).
    reg.store(k.clone(), vec![1, 2, 3, 4, 5], payload_bytes(150), 4096, 4096)
        .expect("store v2 (overwrite)");
    assert_eq!(
        reg.current_bytes(),
        150,
        "after v2 overwrite: must be 150 (not 450 double-count)"
    );
    assert_eq!(reg.len(), 1, "still one entry");

    // Overwrite again with a larger payload (400 bytes).
    reg.store(k.clone(), vec![1, 2, 3, 4, 5, 6, 7], payload_bytes(400), 4096, 4096)
        .expect("store v3 (overwrite)");
    assert_eq!(
        reg.current_bytes(),
        400,
        "after v3 overwrite: must be 400 (not 550 accumulated)"
    );
    assert_eq!(reg.len(), 1);
}

#[test]
fn byte_budget_same_key_reinsert_exceeds_budget_returns_err() {
    // Codex Phase-2b 2026-05-06 (HIGH): same-key reinsert with new_bytes
    // > byte_budget MUST return Err(EntryExceedsBudget) and leave state
    // untouched. The pre-fix path silently accepted the oversized payload
    // and let current_bytes climb above byte_budget.
    let mut reg: LcpRegistry<Marker> = LcpRegistry::with_byte_budget(1024);
    let k = key(0xCC, "t", 1);

    // First insert: 300 bytes — fits. Use a 3-token prompt so we can probe
    // with a strict prefix later (lookup gates on `0 < K < new_tokens.len()`,
    // returning None on full equality).
    reg.store(k.clone(), vec![1, 2, 3], payload_bytes(300), 4096, 4096)
        .expect("v1 store under-budget");
    let bytes_before = reg.current_bytes();
    assert_eq!(bytes_before, 300);

    // Overwrite same key with 2048 bytes — single entry > 1024 byte budget.
    let result = reg.store(
        k.clone(),
        vec![1, 2, 3, 4],
        payload_bytes(2048),
        4096,
        4096,
    );
    match result {
        Err(LcpStoreError::EntryExceedsBudget {
            entry_bytes,
            budget_bytes,
        }) => {
            assert_eq!(entry_bytes, 2048);
            assert_eq!(budget_bytes, 1024);
        }
        other => panic!(
            "expected EntryExceedsBudget on same-key oversize reinsert, got {:?}",
            other
        ),
    }

    // Registry state unchanged after the rejected store.
    assert_eq!(reg.len(), 1, "registry length must remain 1 after rejected reinsert");
    assert_eq!(
        reg.current_bytes(),
        bytes_before,
        "current_bytes must remain 300 after rejected reinsert (not climb to 2048)"
    );
    // Original v1 payload is still indexed (lookup with a strict-prefix
    // probe; full-equality lookup returns None by design).
    let prefix = reg
        .lookup(&k, &[1, 2, 3, 99])
        .expect("v1 entry still indexed under k after rejected oversize reinsert");
    assert_eq!(prefix.k, 3, "v1 prompt [1,2,3] still matches as prefix of [1,2,3,99]");
}

#[test]
fn byte_budget_zero_panics() {
    // (e) byte_budget == 0 panics at construction (mirrors capacity==0 panic).
    let result = std::panic::catch_unwind(|| LcpRegistry::<Marker>::with_byte_budget(0));
    assert!(
        result.is_err(),
        "LcpRegistry::with_byte_budget(0) must panic — zero budget refuses all stores"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Memory-probe smoke tests — ADR-017 Phase E.a default-on (subtask D)
//
// These three tests pin the contract of `default_lcp_byte_budget()`:
//   (i)   The probe result falls within the documented [1 GiB, 16 GiB] clamp
//          when no override env is set.
//   (ii)  A `2g` byte-suffix override is parsed correctly.
//   (iii) A legacy bare-integer entry-count (< 4096) is converted via the
//          300 MB heuristic.
//
// Env-isolation caveat: `default_lcp_byte_budget()` reads a process-global
// env var. Tests that mutate env must restore it before returning to avoid
// contaminating other tests running in the same process. There is no
// `serial_test` dependency in this project; instead each env-mutating test
// uses a guard struct that restores the original value via `Drop`.
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard that restores `HF2Q_KV_LCP_RESUME_CAPACITY` to its original
/// value (or removes it if it was absent) when dropped.
struct EnvGuard {
    key: &'static str,
    original: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let original = std::env::var(key).ok();
        // SAFETY: single-threaded test context; Rust test harness runs
        // each test in a separate thread but std::env::set_var is not
        // thread-safe in multi-threaded test contexts. Tests that mutate
        // env should be run with `RUST_TEST_THREADS=1` to avoid races.
        // This file documents that caveat in the test comment.
        unsafe { std::env::set_var(key, value) };
        Self { key, original }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(v) => unsafe { std::env::set_var(self.key, v) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

/// Smoke test: verify `default_lcp_byte_budget()` returns a value within
/// the documented [1 GiB, 16 GiB] clamp range.
///
/// This test does NOT set any env override, so it exercises the sysinfo
/// probe path. On any machine with nonzero available memory the result
/// must fall in [1 GiB, 16 GiB].
///
/// Env-isolation note: if `HF2Q_KV_LCP_RESUME_CAPACITY` is already set in
/// the test process env (e.g. by a parent test), the value tested here
/// reflects that override, not the sysinfo path. The clamp assertion still
/// holds for any conforming value in the documented range.
#[test]
fn default_lcp_byte_budget_within_clamp_range() {
    const GIB: u64 = 1024 * 1024 * 1024;
    let budget = default_lcp_byte_budget();
    assert!(
        budget >= GIB,
        "default_lcp_byte_budget() returned {budget} bytes — below 1 GiB floor. \
         Check the sysinfo probe path in lcp_registry.rs."
    );
    assert!(
        budget <= 16 * GIB,
        "default_lcp_byte_budget() returned {budget} bytes — above 16 GiB ceiling. \
         Check the clamp in lcp_registry.rs."
    );
}

/// Env override: `HF2Q_KV_LCP_RESUME_CAPACITY=2g` must parse to exactly
/// 2 × 1024³ bytes.
///
/// The `EnvGuard` restores the env after the test returns (or panics).
#[test]
fn default_lcp_byte_budget_env_override_2g() {
    const TWO_GIB: u64 = 2 * 1024 * 1024 * 1024;
    let _guard = EnvGuard::set("HF2Q_KV_LCP_RESUME_CAPACITY", "2g");
    let budget = default_lcp_byte_budget();
    assert_eq!(
        budget, TWO_GIB,
        "HF2Q_KV_LCP_RESUME_CAPACITY=2g must parse to {TWO_GIB} bytes, \
         got {budget}. Check the g-suffix multiplier in default_lcp_byte_budget()."
    );
}

/// Legacy entry-count path: bare integer < 4096 is converted via
/// `n × 300 MiB`. `HF2Q_KV_LCP_RESUME_CAPACITY=8` → `8 × 300 × 1024² bytes`.
///
/// This preserves backward-compat for operators who had `CAPACITY=8` set
/// (the original default from commit c04d5d2 — see chesterton_findings).
#[test]
fn default_lcp_byte_budget_legacy_entry_count_8() {
    const EXPECTED: u64 = 8 * 300 * 1024 * 1024;
    let _guard = EnvGuard::set("HF2Q_KV_LCP_RESUME_CAPACITY", "8");
    let budget = default_lcp_byte_budget();
    assert_eq!(
        budget, EXPECTED,
        "HF2Q_KV_LCP_RESUME_CAPACITY=8 (legacy entry-count) must give \
         8 × 300 MiB = {EXPECTED} bytes, got {budget}. \
         Check the bare-integer < 4096 path in default_lcp_byte_budget()."
    );
}
