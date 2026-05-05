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
//! The registry is generic over the cached payload type `T: Send + Sync
//! + 'static`. Production wires `T = Vec<DenseKvBuffers>` (per dossier
//! §10.3 Strategy A); these tests use `T = Vec<u8>` as a marker payload
//! since `MlxBuffer` requires a Metal device that unit tests don't
//! have. The Arc-pin invariant (§6 R3) is testable without touching
//! Metal because `Arc::strong_count` is payload-agnostic.

use hf2q::serve::kv_persist::lcp_registry::{LcpKey, LcpRegistry, LcpStoreError};
use hf2q::serve::kv_persist::format::ModelFingerprint;
use std::sync::Arc;

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
fn payload(num_layers: usize, marker: u8) -> Vec<Arc<Vec<u8>>> {
    (0..num_layers)
        .map(|li| Arc::new(vec![marker, li as u8]))
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(8);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(8);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(8);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(8);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(8);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(8);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(2); // small capacity → easy eviction
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
    let pinned: Arc<Vec<u8>> = Arc::clone(&hit.dense_kvs[0]);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(2);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(2);
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
    let result = std::panic::catch_unwind(|| LcpRegistry::<Vec<u8>>::new(0));
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(4);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(4);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(4);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(4);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(4);
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
    let mut reg: LcpRegistry<Vec<u8>> = LcpRegistry::new(4);
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
