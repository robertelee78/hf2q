//! ADR-017 Phase E.a default-on — byte-budget invariant stress falsifier.
//!
//! ## Purpose
//!
//! This file is the **load-bearing falsifier** for AC-2 and AC-5 (spec
//! `cfa-lcp-default-on-2026-05-06`). It proves that `LcpRegistry`'s
//! byte-budget enforcement is correct under 1 000 randomised insertions
//! with mixed evictions, lookups, and consuming `take_prefix` calls.
//!
//! The falsifier is designed to catch two distinct bug classes:
//!
//! 1. **Store-path accounting bugs** — any `store()` implementation that
//!    fails to subtract evicted entries' bytes, double-counts same-key
//!    reinserts, or mis-computes new-entry bytes would violate
//!    `current_bytes <= byte_budget` after a store.
//!
//! 2. **take_prefix subtraction bugs** — Subtask B's bug catch: the
//!    original `take_prefix()` was NOT subtracting `entry.bytes` from
//!    `current_bytes` on removal. This test catches that class of bug
//!    via the `pre - post == entry_bytes` assertion on every
//!    `take_prefix` call.
//!
//! ## Payload type
//!
//! Uses `T = Marker` (the same newtype as `tests/lcp_registry_unit.rs`).
//! The orphan rule prohibits `impl ByteSized for Vec<u8>` (Vec is
//! foreign), so `Marker(Vec<u8>)` is the correct local-type solution.
//!
//! ## Determinism
//!
//! The RNG is a simple linear-congruential generator seeded from the
//! constant `42`. No external `rand` crate is required — the project
//! does not declare `rand` as a dependency (neither main nor dev). The
//! LCG is sufficient for mixing key selection and payload sizes across
//! 1 000 iterations.
//!
//! ## Run
//!
//! ```bash
//! cargo test --release lcp_registry_byte_budget_falsifier
//! ```
//!
//! Expected: 1/0/0 (one test, zero failures, zero ignored). Typical
//! run time: well under 1 second on release build.

use hf2q::serve::kv_persist::lcp_registry::{ByteSized, LcpKey, LcpRegistry, LcpStoreError};
use hf2q::serve::kv_persist::format::ModelFingerprint;
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// Marker payload — mirrors tests/lcp_registry_unit.rs pattern exactly.
// ─────────────────────────────────────────────────────────────────────────────

/// Newtype marker payload. Byte size == inner `Vec` length.
/// Necessary because the orphan rule prohibits `impl ByteSized for Vec<u8>`.
#[derive(Clone, Debug)]
struct Marker(Vec<u8>);

impl ByteSized for Marker {
    fn byte_len(&self) -> u64 {
        self.0.len() as u64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal deterministic LCG — no external crate needed.
// ─────────────────────────────────────────────────────────────────────────────

/// Linear-congruential generator seeded deterministically.
///
/// Parameters from Knuth Vol. 2, §3.3.4 table 1 (the classic C `rand`
/// values, widely reproducible): m=2^32, a=1664525, c=1013904223.
/// Produces a full-period sequence over u32.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn seed_from_u64(seed: u64) -> Self {
        // Mix the seed into the lower 32 bits.
        Self { state: seed & 0xFFFF_FFFF }
    }

    /// Next pseudorandom u32.
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) & 0xFFFF_FFFF;
        self.state as u32
    }

    /// Uniform in `[lo, hi)`.
    fn next_in_range(&mut self, lo: u64, hi: u64) -> u64 {
        let range = hi - lo;
        lo + (self.next_u32() as u64 % range)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test fixtures
// ─────────────────────────────────────────────────────────────────────────────

const KEY_POOL_SIZE: usize = 32;

/// Build a key from a pool index. Same fingerprint byte and tenant for all;
/// `params_hash` distinguishes keys (mirrors how iter-3 uses params_hash for
/// per-request isolation).
fn pool_key(idx: usize) -> LcpKey {
    LcpKey {
        model_fingerprint: ModelFingerprint([0xABu8; 32]),
        tenant_id: "stress-tenant".to_string(),
        params_hash: idx as u64,
    }
}

/// Build a single-layer payload whose total byte count equals `bytes`.
fn marker_payload(bytes: usize) -> Vec<Arc<Marker>> {
    vec![Arc::new(Marker(vec![0u8; bytes]))]
}

/// A short synthetic prompt for a given key index (2 tokens — enough for
/// LcpRegistry::store to accept). Distinct per key so fingerprint + params_hash
/// remain the primary isolation axis (the prompt is only used for LCP lookups).
fn key_prompt(idx: usize) -> Vec<u32> {
    vec![idx as u32 + 1, idx as u32 + 1000]
}

// ─────────────────────────────────────────────────────────────────────────────
// Stress test
// ─────────────────────────────────────────────────────────────────────────────

/// Randomised 1 000-iteration stress falsifier for the byte-budget invariant.
///
/// **Invariant under test**: `registry.current_bytes() <= byte_budget` after
/// every `store()` call.
///
/// **take_prefix subtraction assertion** (B's bug-catch pattern): for every
/// `take_prefix` hit, `pre_bytes - post_bytes == entry_byte_len` — where
/// `entry_byte_len` is computed from the returned payload's single-layer
/// `byte_len()`. This assertion is the precise check that would have caught
/// the original bug in which `take_prefix()` omitted the `current_bytes -=
/// removed.bytes` step.
///
/// **Loop structure**: 1 000 store iterations with a mix-in of lookup and
/// take_prefix calls every 10 iterations.
#[test]
fn lcp_registry_byte_budget_stress_1000_iter() {
    const BYTE_BUDGET: u64 = 100_000;
    const ITERATIONS: usize = 1_000;
    // Payload sizes: [1 024, 50 000) bytes.
    const PAYLOAD_LO: u64 = 1_024;
    const PAYLOAD_HI: u64 = 50_001; // exclusive upper bound

    let mut rng = Lcg::seed_from_u64(42);
    let mut reg: LcpRegistry<Marker> = LcpRegistry::with_byte_budget(BYTE_BUDGET);

    for i in 0..ITERATIONS {
        // Pick a random key from the pool.
        let key_idx = rng.next_in_range(0, KEY_POOL_SIZE as u64) as usize;
        let key = pool_key(key_idx);
        let prompt = key_prompt(key_idx);

        // Generate a random payload size in [1 024, 50 000].
        let size = rng.next_in_range(PAYLOAD_LO, PAYLOAD_HI) as usize;
        let payload = marker_payload(size);

        // Attempt the store. EntryExceedsBudget is never expected here
        // (50 000 < 100 000), but we handle it defensively — the invariant
        // must hold in either branch.
        match reg.store(key, prompt, payload, 4096, 4096) {
            Ok(()) => {}
            Err(LcpStoreError::EntryExceedsBudget { entry_bytes, budget_bytes }) => {
                // This path is only reachable if the payload exceeds the entire
                // budget. With PAYLOAD_HI=50_000 and BYTE_BUDGET=100_000 this
                // should never fire; if it does the invariant still holds (registry
                // is unchanged after EntryExceedsBudget).
                panic!(
                    "iter {i}: unexpected EntryExceedsBudget \
                     (entry_bytes={entry_bytes}, budget={budget_bytes}); \
                     max payload {PAYLOAD_HI} should be well under budget {BYTE_BUDGET}"
                );
            }
            Err(other) => {
                panic!("iter {i}: unexpected store error: {other:?}");
            }
        }

        // INVARIANT: current_bytes must never exceed the budget.
        let cur = reg.current_bytes();
        assert!(
            cur <= BYTE_BUDGET,
            "INVARIANT VIOLATED at iter {i}: \
             current_bytes={cur} > byte_budget={BYTE_BUDGET}. \
             This indicates a byte-accounting bug in store() or eviction.",
        );

        // Every 10 iterations: mix in a lookup or take_prefix call on a
        // randomly chosen key to exercise the accounting on the read/remove
        // paths.
        if i % 10 == 9 {
            let probe_idx = rng.next_in_range(0, KEY_POOL_SIZE as u64) as usize;
            let probe_key = pool_key(probe_idx);
            // Build a probe prompt that shares the stored prefix then diverges,
            // so we get a partial-match (k > 0) rather than full-equality-None.
            let mut probe_prompt = key_prompt(probe_idx);
            probe_prompt.push(99_999); // diverging suffix token

            // Alternate between take_prefix (even half-iterations) and lookup
            // (odd half-iterations).
            if (i / 10) % 2 == 0 {
                // take_prefix path: the KEY assertion for catching B's bug.
                //
                // Pre-condition: record current_bytes BEFORE the take.
                let pre_bytes = reg.current_bytes();

                if let Some(prefix) = reg.take_prefix(&probe_key, &probe_prompt) {
                    // The payload has exactly one layer (we always build
                    // marker_payload with a single Vec<Arc<Marker>>).
                    let entry_byte_len = prefix.dense_kvs[0].byte_len();
                    let post_bytes = reg.current_bytes();

                    // THE LOAD-BEARING ASSERTION:
                    // pre_bytes - post_bytes must equal the removed entry's bytes.
                    // If take_prefix did NOT subtract entry.bytes from current_bytes,
                    // this fires immediately — exactly the class of bug that B caught.
                    assert_eq!(
                        pre_bytes.saturating_sub(post_bytes),
                        entry_byte_len,
                        "take_prefix byte accounting BROKEN at iter {i}: \
                         pre_bytes={pre_bytes}, post_bytes={post_bytes}, \
                         entry_byte_len={entry_byte_len}. \
                         Difference should equal entry size but got {}.",
                        pre_bytes.saturating_sub(post_bytes),
                    );

                    // After removal the invariant must still hold.
                    assert!(
                        post_bytes <= BYTE_BUDGET,
                        "INVARIANT VIOLATED after take_prefix at iter {i}: \
                         current_bytes={post_bytes} > byte_budget={BYTE_BUDGET}",
                    );
                }
                // On miss: current_bytes must be unchanged.
                else {
                    let post_bytes = reg.current_bytes();
                    assert_eq!(
                        pre_bytes, post_bytes,
                        "take_prefix miss must NOT change current_bytes (iter {i}): \
                         pre={pre_bytes}, post={post_bytes}",
                    );
                }
            } else {
                // lookup path: does not mutate registry; invariant still holds.
                let _ = reg.lookup(&probe_key, &probe_prompt);
                let cur_after = reg.current_bytes();
                assert!(
                    cur_after <= BYTE_BUDGET,
                    "INVARIANT VIOLATED after lookup at iter {i}: \
                     current_bytes={cur_after} > byte_budget={BYTE_BUDGET}",
                );
            }
        }
    }

    // Final state: walk all remaining entries by iterating over all pool keys,
    // summing the byte size implied by each entry still present, and comparing
    // to current_bytes(). We use lookup (non-consuming) for this — an entry is
    // present if lookup returns Some on a probe prompt with a diverging suffix.
    //
    // Note: we cannot directly iterate the internal map without a test-only
    // accessor. Instead we assert the global invariant: current_bytes() <=
    // byte_budget (already checked after every store above), and additionally
    // verify that the registry never reports negative bytes (saturating_sub
    // floor) via the unsigned type invariant.
    let final_bytes = reg.current_bytes();
    assert!(
        final_bytes <= BYTE_BUDGET,
        "FINAL INVARIANT VIOLATED: current_bytes={final_bytes} > byte_budget={BYTE_BUDGET}",
    );

    // Verify that current_bytes() reports a non-zero value if any entry
    // was stored and not yet taken (heuristic sanity: after 1 000 stores
    // into a 100 KB budget the registry should hold at least one entry,
    // so current_bytes > 0 unless every entry was take_prefix'd away).
    // We only assert the lower bound when len() > 0.
    if reg.len() > 0 {
        assert!(
            final_bytes > 0,
            "registry.len()={} but current_bytes==0 — byte accounting drifted",
            reg.len(),
        );
    }
}
