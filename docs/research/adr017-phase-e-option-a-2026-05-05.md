# ADR-017 Phase E Option (a) — LCP Partial-Prefill Resume — Research Dossier

- **Date:** 2026-05-05
- **Author:** Phase 2a writer (claude-solo); Phase 2b auditor: codex (review-only)
- **Session:** `cfa-20260505-063610-adr017-pe-a-research`
- **Mode:** read-only research dossier; NO production code, NO new tests
- **Output spec:** acceptance criteria AC-1 … AC-12 from queen Phase-1 spec
- **Family scope (v1):** Gemma 4 dense ONLY; Qwen3.5 hybrid families EXPLICITLY DEFERRED (forward pointer: ADR-017 Phase B-hybrid)

---

## TL;DR

- **What.** Phase E option (a) = LCP-based **partial-prefill resume**: on each request, compute the longest common token-id prefix `K` between the new prompt and the previously-served prompt; instead of resetting `write_pos = 0` (today's contract at `forward_prefill.rs:446`), resume from `write_pos = K` and run prefill only for tokens `[K..N)`. The cached `dense_kvs[*][0..K)` is reused in place.
- **Why now.** The shipped Phase E option (b) (cross-process `PromptCache` replay, iter-5/6, `ADR-017 doc:3465-3467`) needs **full prompt equality** to fire. Multi-turn chat over `/v1/chat/completions` does NOT satisfy that — turn N's prompt = turn N-1's prompt + new assistant reply + new user message — so option (b) misses on every turn ≥ 2 even though the prefix is identical.
- **Load-bearing motivation (iter-4 finding).** `forward_prefill.rs:446` unconditionally resets `cache.write_pos = 0; cache.seq_len = 0` for every prefill. ADR-017 doc lines 3434-3441 document that this **overwrites any restored KV state**, making real cache-hit acceleration architecturally impossible without revisiting that reset (`forward_prefill.rs:443-444` already names this exact follow-up).
- **Verdict.** Feasible at "Phase B-dense.1-class" scope. **iter-1 LANDED `9beb906`**, **iter-2 LANDED `d17163b`**, **iter-2.5 LANDED `70fca89`**, **iter-3 LANDED `a20606a` + audits `1006ab0` + `22e456a`**, **iter-3.5 LANDED `7ffb563`** (dtype + decode-wrap snapshot). **iter-3.5c (sliding-layer prefill-wrap guard) + iter-5 (R-C4-LCP 5-K sweep KILL-criterion) LANDED at `31d04b3`** — 5/5 fractions byte-identical on Gemma 4 26B-DWQ; iter-5 sweep also surfaced+fixed a real `.cloned()`/Arc bug in `forward_prefill_with_soft_tokens_resume` that the single-resume iter-3 falsifier could not catch. iters 4 / 6 / 7 PENDING. iter-4 (env-flip default-ON) gated on iter-6 (R-P7 multi-turn + /cfa fan-out bench). iter-3.5c restriction (`prompt_len ≤ sliding_window` for sliding-layer models) lifts at iter-3.6 via mid-prefill snapshot — concrete iter gate, not "deferred to v2".
- **Speedup ceiling (computed, not measured).** Multi-turn chat with 95 % shared prefix: predicted turn-2 TTFT ≤ 5 % × turn-1 TTFT + ~50 ms LCP/restore overhead = ~8-19× depending on prompt length and base prefill rate (formula in §5).
- **LOC estimate.** ~1,060 LOC post-Codex-audit (forward_prefill ~140, forward_mlx ~25, engine.rs ~155, kv_restore_gemma ~50, new `lcp_registry.rs` ~250, tests ~440) vs Phase B-dense.1's ~1,500 LOC and Phase D scaffolding's 1,722 LOC — closer to Phase B than Phase D. Pre-audit estimate was ~930; §10.7 details the ~14% upward revision.
- **Hard kill criterion.** If H1 (byte-identity) fails at any of 5 LCP fractions across the R-C4 sourdough fixture set → Phase E.a is FALSIFIED → pivot to option (b) only.
- **Beneficiary classes (production use cases).** Two distinct shapes both gain from E.a beyond what shipped E.b covers:
  1. **Multi-turn chat over `/v1/chat/completions`.** Turn N's prompt = turn N-1's prompt + assistant reply + new user message. E.b misses every turn ≥ 2 (suffix differs); E.a hits LCP on the shared prefix.
  2. **/cfa Phase 2 fan-out (singleton sidecar — see `docs/research/cfa-hf2q-integration-2026-05-05.md`).** Each worker gets `[SYSTEM][QUEEN_SPEC][role+assignment]`. Workers 2-N share the prefix `[SYSTEM][QUEEN_SPEC]` but diverge on the per-role suffix. E.b misses on workers 2-N; E.a hits LCP on shared chunk. **Public docs / launch posts must NOT claim "R-P6 1.00× for /cfa" until E.a iter-3 ships** — the synthetic R-P6 fixture is byte-identical, real /cfa fan-out is shared-prefix-different-suffix.

---

## §1 — Q1: Chesterton's-fence audit (`forward_prefill.rs:446` + `:1502`)

### 1.1 Verbatim comment block — `forward_prefill.rs:412-448`

The wholesale `write_pos = 0` reset that this proposal would soften is documented in code today. Quoting verbatim from `/opt/hf2q/src/serve/forward_prefill.rs:412-448` (the "Reset per-layer KV cache write positions before this prefill" block):

```text
// Reset per-layer KV cache write positions before this prefill.
//
// The TQ-packed `MlxKvCache` (allocated once at model load with
// capacity = max_position_embeddings for full layers, sliding_window
// for sliding layers) accumulates `write_pos` + `seq_len` across
// every prefill / decode step.  In a single-request lifecycle that's
// correct: prefill writes positions 0..N, decode appends N..N+M.  But
// hf2q's serialized worker handles multiple requests on the same
// `LoadedModel` — each fresh request needs to OVERWRITE the cache
// from position 0, not append.
//
// This was a latent bug in the chat-only path that worked in practice
// because:
//   * Full-attention layers have huge capacity (max_position_embeddings,
//     262144 for Gemma 4) — many requests fit before overflow.
//   * Sliding-window layers wrap via `(write_pos % sliding_window)` so
//     buffer accesses stayed in-bounds — but `seq_len` (passed to
//     flash-attention as the count of valid KV positions) kept growing
//     unboundedly, making the kernel attend to "valid" positions that
//     in fact contained stale data from prior requests.
//
// Iter-92 (Phase 2a Task #8) surfaced the bug via the embedding path:
// many embed requests + a chat completion drove sliding-layer
// `seq_len` past the sliding_window capacity → the dispatcher's
// `kv_capacity (sw) < kv_seq_len` guard fired with a hard error.
//
// The fix: reset `write_pos` + `seq_len` to 0 here so every prefill
// starts with an empty KV cache, regardless of prior state.  Each
// OpenAI `/v1/chat/completions` and `/v1/embeddings` request is
// semantically independent (multi-turn chat is handled by the client
// sending full history), so wholesale reset is the correct semantics.
// When prompt-cache lands (Phase 2a Task #7) it'll need to revisit
// this and reset only positions past the cached prefix length.
for cache in self.kv_caches.iter_mut() {
    cache.write_pos = 0;
    cache.seq_len = 0;
}
```

That last sentence ("**When prompt-cache lands ... it'll need to revisit this and reset only positions past the cached prefix length**") is the smoking gun: the original author of the iter-92 fix already named "Phase E.a" by description.

The cross-reference at `engine.rs:1461-1468` (the iter-97+ scope marker on `PromptCache`) calls out the same boundary and is quoted verbatim by ADR-017 doc line 3439:

> "Iter-97+ scope: LCP-based partial-prefill resume. Compute the longest common prefix between the new prompt and `tokens`, set `kv_caches[*].write_pos = LCP`, pre-warm `dense_kvs[0..LCP)` by dequantizing `kv_caches[0..LCP)` via `tq_dequantize_kv`, then run `forward_prefill` for tokens `[LCP..N)`. Reports `cached_tokens = LCP` (any value `0 ≤ LCP ≤ prompt_tokens`). **Defers to a later iteration** because the dequant pre-warm is non-trivial — the iter-96 full-equality cache is a real, shippable subset."

### 1.2 Invariants the wholesale reset enforces

The reset block is load-bearing for at least four invariants. Any partial-prefill design must explicitly preserve every one of them (Chesterton's fence) before any code changes are proposed.

| # | Invariant | Pinned at | Failure mode if violated |
|---|-----------|-----------|--------------------------|
| **I-1** | **Multi-request prefill independence on a serialized worker.** Each `/v1/chat/completions` + `/v1/embeddings` request must START from a known empty KV state on its `LoadedModel`. | `forward_prefill.rs:412-421`, `forward_prefill.rs:438-442` | Cross-request KV bleed: request B's flash-attention sees positions filled by request A. Pre-iter-92 latent bug. |
| **I-2** | **Embed-path safety (post Phase 2a Task #8 / iter-92).** Many `/v1/embeddings` calls accumulate `seq_len` on sliding layers; without reset, `seq_len` exceeds `sliding_window`, the `kv_capacity (sw) < kv_seq_len` dispatcher guard panics. | `forward_prefill.rs:433-436` | Hard runtime error on chat-after-embed; user-visible 500. (Iter-92 ship-stopper.) |
| **I-3** | **Sliding-layer ring permutation-invariance is a *correctness*, not a *cache-efficiency*, guarantee.** Reads via `kv_seq_len = min(tok_i+1, sliding_window)` and `slot = tok_pos % sliding_window`. Causal attention over the populated window is permutation-invariant ONLY if the window contents are coherent under one consistent indexing scheme. | `forward_prefill.rs:460-471`, `forward_prefill.rs:850-857` (`write_slot = tok_i % layer_dense_cap`), `forward_prefill.rs:850-869` (sliding ring writes) | If LCP > sliding_window, the cached ring is already wrapped; resuming at `write_pos = LCP` mixes pre- and post-wrap content if not addressed (see §3.4). |
| **I-4** | **Panic cleanup / atomicity.** A previous prefill that paniced mid-loop (e.g. OOM at layer 30) leaves `write_pos` and `seq_len` in arbitrary partial state. The wholesale reset is a no-cost cleanup. | `forward_prefill.rs:438-442` ("regardless of prior state"), no explicit unwind/`Drop` on `MlxKvCache` | Half-populated cache from prior panic interpreted as valid cached prefix → silent miscompute. |

### 1.3 Load-bearing tests pinning the invariants

- **R-C4 sourdough byte-identity (ADR-009).** `tests/kv_persist_gemma4_roundtrip.rs:1563-1730` (R-C4 internal sourdough harness — baseline TTFT vs restored TTFT and decoded-byte equality). The 22-token sourdough prompt + greedy decode + dense-SDPA path is the canonical byte-equality fixture and is what iter-4 of ADR-017 Phase D used to certify the spill→restore plumbing produced byte-identical outputs (`ADR-017 doc:3218-3232` shows the iter-4/7/9 stability table).
- **`prompt_cache_live.rs`** — exercises the iter-96 full-equality `PromptCache` path that LCP-resume sits underneath; relevant for confirming the new path doesn't regress same-prompt full-replay (`engine.rs:1452-1505`).
- **`kv_persist_gemma4_roundtrip.rs::sourdough_baseline`** (line 979 onward) — the peer-arm-equality oracle (hf2q + llama.cpp share `MIN_COMMON_PREFIX`).

### 1.4 What would BREAK if the reset were conditionally skipped today (no compensating logic)

1. **Embedding-after-chat or chat-after-embed cross-talk** (I-2 violated): sliding-layer `seq_len` grows past `sliding_window`, dispatcher hard-errors.
2. **Tenant isolation** (I-1 violated): a request from Alice could read stale KV positions populated by Bob's previous prompt.
3. **Sliding-layer ring miscompute when `LCP > sliding_window`** (I-3 violated): see §3.4.
4. **Panic non-atomicity** (I-4 violated): previous-request panic state leaks into current.

**Conclusion.** Phase E.a CANNOT be a one-line "skip the reset" patch. It must:

- Carry an explicit **`Option<LcpPrefix>` argument** through to `forward_prefill` (so the reset path stays the default and the bypass path is opt-in per request).
- Compensate I-1 by a **request-scoped cache-key gate** on the `LcpPrefix` (only this tenant + this fingerprint can resume).
- Compensate I-3 by **ring-wrap-aware** `write_pos` setting (see §3.4).
- Compensate I-4 by **failing CLOSED** (treating unknown `kv_caches[*]` write positions as "must full-reset"), never failing OPEN.

NO code is proposed in §2/§3/§4 until this fence audit completes — which it now has.

---

## §2 — Q2: LCP detection mechanism design

### 2.1 Where in the request flow

Three candidate sites in `engine.rs`:

| Site | Location | Cost / suitability |
|---|---|---|
| **A. Pre-tokenize (string-level)** | Before `tokenizer::encode()` | Cheap byte-compare but fundamentally unsafe (see §2.2) |
| **B. Post-tokenize, BEFORE PromptCache lookup** | Right after `tokenizer::encode` returns `Vec<u32>`, before `engine.rs:3658` `loaded.prompt_cache.lookup(...)` | Wrong order: option (b) full-equality should fire first when it can (it's faster). |
| **C. Post-tokenize, AFTER PromptCache full-equality miss** ✅ | `engine.rs:3658-3664` — when `prompt_cache.lookup(...)` returns `None`, *then* try LCP | Correct: full-equality is a strict subset of LCP=N; honor it first to keep iter-96 fast path; LCP only fires on the partial-match miss. |

**Recommendation:** Site **C** — implement LCP detection as a fall-through path triggered by the existing `lookup` returning `None`. The shape:

```text
let cached = loaded.prompt_cache.lookup(prompt_tokens, params);
if let Some(c) = cached { return Ok(c); }                             // option (b) hit
let lcp = loaded.lcp_registry.lookup(prompt_tokens, params, tenant);   // NEW
let prefill_result = run_prefill(prompt_tokens, lcp.as_ref());        // NEW: hand-off Option<LcpPrefix>
loaded.prompt_cache.store(...);                                       // unchanged
loaded.lcp_registry.store(prompt_tokens, ..., dense_kvs_handle);      // NEW
```

This keeps the iter-96 contract intact (`PromptCache` semantics unchanged; ADR-009 R-C4 baseline-vs-restored byte-identity unaffected) and isolates the new path to one fall-through.

### 2.2 Token-id vs byte-level granularity

**Recommendation: token-id-level. NEVER byte-level.**

- BPE tokenization is deterministic for a fixed model + tokenizer config: equal token-id prefix ⟺ a corresponding byte prefix existed at encode time. The converse is **NOT** true: equal byte prefix CAN map to different token-id sequences depending on what byte follows the boundary.
- The KV cache slots (and therefore `dense_kvs[*][0..K)`) are indexed by **token positions**, not byte offsets. There is no meaningful "K-th byte" cached state to compare against.

**Concrete failure mode (the BPE merge edge case the queen spec demands):**

> Cached prompt suffix: `"...the quick brown"` — last cached token id = `4581` ("brown" + space, depending on tokenizer). Cached state at position K-1 = the K's RoPE position-embedded representation of token `4581`.
>
> New prompt continues: `"...the quick brownfox"` — encoded fresh, the last shared-bytes-prefix is "...the quick brown" but the BPE merge at `"brown" + "fox"` collapses to a single different token id, e.g. `7799` ("brownfox") — or splits at a different byte than expected ("brownf" + "ox" → ids `9990, 113`).
>
> A byte-LCP of "...the quick brown" (16 chars) would yield bytes-LCP=16; but the token-id sequence DIFFERS at the position where "brown" used to be. If we resumed from byte-LCP, the cached KV at that token slot encodes `4581`, while the new prompt expects `7799` at the same position → first new-token attention reads STALE K,V from a token that no longer exists in the new prompt → silent semantic drift.

**Token-id-level rule:** LCP `K` is defined as the **largest** index such that `cached_tokens[i] == new_tokens[i]` for all `0 ≤ i < K` (equivalent to llama.cpp's `tokens.get_common_prefix(input_tokens)` at `/opt/llama.cpp/tools/server/server-context.cpp:2360`). This guarantees `dense_kvs[*][0..K)` was generated by the model conditioning on the EXACT same token sequence — so it's safe to resume from.

### 2.3 Algorithm pseudocode

```text
fn lcp_lookup(
    new_tokens: &[u32],
    cached: &CachedPrefixEntry,    // tokens, fingerprint, dense_kvs Arc handle, params
    params: &SamplingParams,
    tenant: TenantKey,
) -> Option<LcpPrefix> {
    if cached.tenant != tenant { return None; }                        // I-1 (Q6 R7)
    if cached.fingerprint != current_fingerprint { return None; }      // ModelFingerprint match
    if cached.params_key != PromptCacheKey::from_params(params) { return None; }   // determinism
    let mut k = 0;
    while k < new_tokens.len() && k < cached.tokens.len()
          && new_tokens[k] == cached.tokens[k] {
        k += 1;
    }
    // Edge: K == cached.tokens.len() AND K == new_tokens.len() → full equality;
    // option (b) PromptCache should have hit upstream. K == 0 → no benefit.
    if k == 0 || k == new_tokens.len() { return None; }                // 0 → no benefit; full → option (b) handles
    Some(LcpPrefix {
        k,
        dense_kvs_handle: cached.dense_kvs_arc.clone(),                // pin for I-3 / Q6 R3
        fingerprint: cached.fingerprint,
    })
}
```

The Arc-handle pin matters: `dense_kvs_handle.clone()` keeps the buffers live across any concurrent eviction (Q6 R3). The `k == new_tokens.len()` early-out guarantees Phase E.a NEVER masks a Phase E.b hit.

### 2.4 BPE tokenization-merge edge cases beyond the §2.2 example

- **Trailing-whitespace fusion:** `"hello"` + `" "` may fuse into a single new token under BPE; if a chat turn's assistant reply ends with whitespace + the next user message starts with whitespace, the boundary token may differ even though the bytes look identical.
- **Special-token-prefix cases:** `<|im_start|>user\n...` boundary tokens are typically single tokens; if the renderer produces a slightly different special-token sequence (e.g. via `parallel_tool_calls=false` re-render), token-id-level LCP correctly says K=0 even if the prefix bytes appear identical.
- **Fix:** detection is on the **rendered + tokenized** token-id stream, NEVER on the raw user-message text. The `engine.rs::PromptCache` already uses post-tokenization `Vec<u32>` (`engine.rs:1510 pub tokens: Vec<u32>`), so LCP detection inherits this correctness for free.

---

## §3 — Q3: Restored KV state coherence at the LCP boundary

### 3.1 RoPE is baked in BEFORE caching

Quoting `forward_mlx.rs:590-592` (the `dense_kvs` field doc):

> "Attention is permutation-invariant over cached K,V (**RoPE is baked in before caching**), so the ring's slot order doesn't matter for correctness — the kernel just attends to all populated slots."

The RoPE call sites at `forward_prefill.rs:763-782` apply per-position rotational embeddings to Q AND K **before** the K,V are written into `dense_kvs[layer]` at `forward_prefill.rs:858-891`. So the cached `dense_kvs[layer][:, p, :]` carries position-`p`-bound RoPE phase.

**Consequence:** cached state is **position-DEPENDENT, not relocatable**. We can resume at the SAME absolute positions but cannot shift them. Phase E.a relies on the new tokens occupying positions `[K..N)` — which they do naturally because in multi-turn chat the new tokens are appended to the same position offsets the original prompt occupied.

### 3.2 The one-line byte-identity condition

> **Partial prefill `[K..N)` produces byte-identical logits at every position `p ≥ K` iff:** (a) `kv_caches[layer].write_pos = K` after restore for every layer, (b) the per-token loop's `seq_pos = tok_i` continues threading the *absolute* position (not LCP-relative), (c) the dispatcher's RoPE freq-factors and `theta` are unchanged across restore (same `model_fingerprint`, dtype, sliding/global config), (d) for sliding layers, the ring slot for any token at absolute position `p ∈ [K..N)` is `p % sliding_window` exactly as the un-resumed run would have written it.

In code: setting `cache.write_pos = K; cache.seq_len = K_seq` (clamped to capacity for sliding) and STARTING the per-token loop at `tok_i = K` with `seq_pos = K, K+1, …, N-1` is sufficient. The existing `seq_pos = tok_i` line at `forward_prefill.rs:618` already threads absolute position; the existing `write_slot = tok_i % layer_dense_cap` at `forward_prefill.rs:854` already computes the correct ring slot. Phase E.a thus becomes: **set the starting position correctly, then trust the existing per-token loop verbatim**.

### 3.3 Model-fingerprint precondition

The `kv_persist::format::compute_model_fingerprint` function at `/opt/hf2q/src/serve/kv_persist/format.rs:213` already exists for cross-process restore (`format.rs:281` `pub model_fingerprint: ModelFingerprint`). Reuse this for the LCP path:

- Cached `LcpEntry.fingerprint` MUST equal `current_loaded.fingerprint`. This rejects: model swap, dtype change, RoPE-theta change, tokenizer change, anything that would alter the per-position embedding.
- This is checked in `lcp_lookup` (§2.3).

### 3.4 Sliding-window edge case: `LCP > sliding_window`

When `LCP > sliding_window`, the cached ring has wrapped: slot `i mod sw` was overwritten by the most recent `(i)` token. Two facts cooperate to keep this safe:

1. **Sliding attention only attends to the last `sliding_window` tokens.** The flash-attn dispatcher reads `kv_seq_len = min(tok_i + 1, sliding_window)` (`forward_prefill.rs:462` documents this; the dense path does the same at the SDPA call). So at position `LCP + j` for any `j ≥ 0`, attention reads slots covering exactly the last `sliding_window` absolute positions ending at `LCP + j` — i.e. positions `[LCP+j-sw+1, LCP+j]`.
2. **The cached ring already has those slots populated.** When the original prefill ran from 0 to `LCP-1` it overwrote slots iteratively with `slot = pos % sw`, so the ring's contents at LCP-time are exactly the last `sw` tokens of the prefix — which is precisely what the next attention step at position `LCP` needs.

So LCP > sliding_window is **safe** as long as we set `write_pos = LCP` (not `LCP mod sw`) and `seq_len = min(LCP, sliding_window)`. The ring's permutation-invariance argument from `forward_prefill.rs:466-471` holds: **WHICH** slot is "oldest" doesn't matter; that the *window contents* are coherent under the same indexing scheme does.

**Pinned condition:** for sliding layers, the LCP path must set:

```text
cache.write_pos = LCP;
cache.seq_len   = min(LCP, sliding_window);
// And the per-token loop's `seq_pos = tok_i` continues threading
// absolute position, so subsequent writes at slot `tok_i % sw`
// land where the un-resumed run would have written them.
```

For global layers (linear capacity = max_position_embeddings, 262144 for Gemma 4): set `write_pos = LCP, seq_len = LCP`. No wrap concern.

### 3.5 Concurrent-eviction precondition (cross-link with Q6 R3)

The cached `dense_kvs` Arc handle pinned by `LcpRegistry` (§2.3) ensures the buffers stay alive between `lcp_lookup` and `forward_prefill`'s consumption. If a concurrent admission decision triggers eviction of the cache entry while the request is in flight, the Arc keeps the K,V live until the request's prefill completes and drops its handle.

---

## §4 — Q4: Architectural changes (LOC + Walk-discipline iter sequence)

### 4.1 Per-file change inventory with LOC estimates

| File | LOC | Change |
|---|---|---|
| `src/serve/forward_prefill.rs` (`:412-448` + per-token loop start) | ~80 | Make the wholesale reset conditional on a new `restored_lcp: Option<usize>` argument. If `Some(k)`, set `write_pos = k`, `seq_len = min(k, capacity)` for each `kv_caches[*]`; advance the per-token loop start index from 0 to `k`; SKIP the `dense_kvs` re-allocation if a matching capacity buffer is already present (handed in via the `LcpPrefix.dense_kvs_handle`). |
| `src/serve/api/engine.rs` (`:1442-1505` PromptCache + `:3658` lookup site + Request enum + worker handler) | ~150 | Add a fall-through after PromptCache miss that consults the new `LcpRegistry`. Extend `Request::Generate` (or add `Request::GenerateWithLcp`) with an optional `lcp_prefix: Option<LcpPrefix>` field. Worker handler unpacks and threads to `forward_prefill`. |
| `src/serve/api/engine.rs` (`:3314 kv_restore_gemma`) | ~50 | When LCP path is active and a `dense_kvs` buffer is handed in by the registry, skip the "allocate dense_kvs if None" block (`:3349-3403`); the buffer is already live. (If `dense_kvs` IS None, fall back to current allocator semantics, preserving Phase D's restore path.) |
| `src/serve/forward_mlx.rs` (DenseKvBuffers consumer-side) | ~0 | Read-only consumer; no change. |
| `src/serve/kv_persist/families/gemma4_dense.rs` | ~0-10 | Optional: extend the restore path to populate the LcpRegistry on cross-process restore (so a freshly-restored KV cache is also LCP-discoverable). Phase D-aligned; can ship in iter-7. |
| **NEW** `src/serve/kv_persist/lcp_registry.rs` | ~250 | The registry struct: stores `Vec<LcpEntry>` keyed by tenant + fingerprint + params, indexed for prefix lookup. Insert on prefill complete; lookup on each request. Single-slot per tenant initially (mirrors PromptCache). Pruning on entry replacement. |
| `tests/lcp_resume_byte_identity.rs` | ~120 | H1 byte-identity gate at 5 LCP fractions. |
| `tests/lcp_resume_speedup.rs` | ~80 | H2 speedup ceiling gate (multi-turn chat fixture). |
| `tests/lcp_resume_floor.rs` | ~60 | H3 multi-turn 95 % LCP floor gate. |
| `tests/lcp_registry_unit.rs` | ~100 | Registry unit tests: tenant isolation, params-key match, fingerprint mismatch rejection, eviction. |
| `tests/lcp_resume_sliding_wrap.rs` | ~50 | Sliding-window edge case (LCP > sliding_window). |
| **TOTAL ADDITIVE** | **~930** | Comparable to Phase B-dense.1 (~1,500) and below Phase D scaffolding (1,722). |

### 4.2 7-iter Walk-discipline sequence (one falsifiable measurement per iter)

Per the project's standing directive `feedback_evidence_first_no_blind_kernel_rewrites` and the Walk-discipline rule in the queen spec, every iter surfaces ONE concrete falsifiable measurement, never "measure later".

| # | Iter | LOC | Falsifiable measurement | Status |
|---|------|-----|--------------------------|--------|
| 0 | **Read-only research dossier (this file)** | 0 src | Verifies AC-1…AC-13 of queen Phase-1 spec + Codex Phase-2b audit folded in §10 + git status clean | ✅ LANDED `4830353` |
| 1 | `lcp_registry.rs` standalone + token prefix-lookup unit tests | ~250 src + ~120 tests | Unit: registry returns correct K for token-prefix-equal sequences, rejects on fingerprint/tenant/params mismatch, evicts under capacity, pins Arc across eviction (R3). Falsifier: any unit test fails | ✅ LANDED `9beb906` (358 LOC src + 369 LOC tests; 13 unit tests) |
| 2 | `engine.rs` Request plumbing + worker handler thread-through (LCP path STILL gated; full-equality `PromptCache` still wins on K=N) | ~165 src + ~85 tests | Integration: real `/v1/chat/completions` 2-turn; second turn's `cached_tokens` reflects LCP value at server side **but partial-resume path is OFF**. Falsifier: `cached_tokens=0` despite shared prefix | ✅ LANDED — `probe_lcp_opportunity` engine helper + `KvCacheMetricsSink::record_lcp_probe` trait method + `KvSpillCounters::{lcp_lookups_total, lcp_detected_total}` + `/metrics` lines + worker-thread probe + store at non-streaming and streaming generate sites + `build_lcp_key_for_request` helper (mirrors `Gemma4DenseSpill::model_fingerprint` recipe). 6 new unit tests (3 in `lcp_registry_unit.rs` for `probe_lcp_opportunity` gating, 3 in `state.rs` for counters). Full bin suite 2804/0/13 PASS. |
| 2.5 | **Strategy A ownership refactor** — `dense_kvs: Option<Vec<Arc<DenseKvBuffers>>>` per §10.3; consumer-site rewrites at `forward_mlx.rs:2432-2588` | ~25 src in forward_mlx.rs + ~5 src in engine.rs (multimodal scope gate per §10.5) | Unit + existing tests: every consumer site compiles + R-C4 sourdough byte-identical (no behavioral change yet — refactor is structural only). Falsifier: any R-C4 fixture diverges. | ✅ LANDED — type change at `forward_mlx.rs:593`. Read-path consumers at `forward_mlx.rs:2432-2794` UNCHANGED (auto-deref through `Arc<T>: Deref<Target=T>`). Three write sites updated: `forward_prefill.rs:1502` + `forward_prefill_batched.rs:2131` (wrap each layer in `Arc::new` at end-of-prefill handoff) + `engine.rs::kv_restore_gemma` (`Some(all.into_iter().map(Arc::new).collect())` + `Arc::get_mut(&mut kvs[layer_rank])` for the per-layer mutable borrow). 2804/0/13 bin tests PASS. **R-C4 sourdough byte-identical PASS** on Gemma 4 26B-DWQ: baseline 3632 bytes == restored 3632 bytes byte-identical, TTFT 287.3 ms cold → 0.5 ms restored (574× cache-hit speedup). |
| 3 | **HIGH-RISK** — `forward_prefill.rs` conditional reset under env-gate `HF2Q_KV_LCP_RESUME=1` (default OFF) + §10.2 three-case capacity protocol (`HF2Q_KV_LCP_REUSE_POLICY=zero_copy_only` for v1) + §10.5 multimodal-bail (`soft_tokens.is_empty()`) + same-process round-trip test (§10.4 corrected: K<N partial-prefix, NOT K=N) | ~140 src + ~50 tests | Same-process: prime cache with `P` (N=24), issue `P' = P[..K] || extra` (K=12, P'.len=16). Assert (a) PromptCache full-equality misses, (b) LcpRegistry returns Some(LcpPrefix{k=12}), (c) decoded bytes byte-identical to fresh-cache full-prefill of P'. Falsifier: any byte mismatch at any of 5 K fractions. **Codex Phase-2b audit MANDATORY** (mirror `feedback_codex_review_catches_unified_memory_races` precedent). | ✅ LANDED — implementation `a20606a` + Codex audit fix-1 `1006ab0` + Codex audit fix-2 `22e456a`. **Codex Phase-2b audit verdict APPROVED-WITH-CAVEATS** across two audit rounds (round 1: 1 HIGH + 1 MED + 2 LOW caught; round 2: 1 new HIGH + 1 LOW caught; all resolved except deferred dtype invariant for v2). 20/20 lcp_registry tests + 2804/0/13 bin tests + R-C4 sourdough byte-identical PASS at default flags + **iter-3 K<N partial-prefix byte-identity falsifier PASS** (Server A resume 144 bytes == Server B fresh-prefill 144 bytes BYTE-IDENTICAL on Gemma 4 26B-DWQ; engagement confirmed via `hf2q_kv_lcp_detected_total=1`). Default-ON promotion deferred to iter-4 after iter-5 (R-C4-LCP 5-fraction sweep) + iter-6 (R-P7 multi-turn-chat speedup) gates pass. **Sliding-ring wrap restriction**: v1 stores only when `prompt_len + physical_decode_writes <= sliding_window`; long-conversation cache miss is the v1 trade for byte-identity correctness. v2 lifts via end-of-prefill snapshot (~5 GB GPU memcpy/req on Gemma 4 26B). |
| 4 | `engine.rs` Request plumbing flips LCP path ON (env-gate respected) | ~10 src | Integration: 2-turn `/v1/chat/completions` shows turn-2 partial prefill via /metrics counter. Falsifier: counter doesn't increment despite shared prefix | ⏸ PENDING |
| 5 | **R-C4 LCP variant gate** — extend sourdough harness with K ∈ {0, N/4, N/2, 3N/4, N-1} | ~120 tests | H1 byte-identity at 5 fractions. Falsifier: any K yields decode-stream divergence vs full-prefill baseline (KILL CRITERION — falsifies Phase E.a entirely) | ✅ LANDED at `31d04b3` — all 5 K fractions byte-identical (Server A resume == Server B fresh-prefill on Gemma 4 26B-DWQ). Engagement counter monotonic 0 → 9 across 5 fractions. **KILL-CRITERION CLEARED.** Iter-5 sweep also surfaced + fixed a real `Arc::try_unwrap` bug in `forward_prefill_with_soft_tokens_resume` (`.cloned()` doubled strong_count on second sequential resume) that the single-resume iter-3 K<N falsifier could not catch. The bug landed silently in `a20606a`, was masked through iter-3.5b/c, and exposed by iter-5's 5-resume sweep. Fix: replace `cached_arcs.get(layer_idx).cloned()` with `cached_arcs.into_iter()` drain — strong_count stays 1, try_unwrap succeeds. |
| 6 | **R-P7 multi-turn-chat speedup bench + /cfa fan-out shared-prefix bench** | ~80 tests | H2/H3 speedup: 5-turn chat with 95 % shared prefix; turn 2+ TTFT ≤ 0.10 × turn 1 TTFT. **Plus**: /cfa-shape fan-out fixture (4 workers, shared `[SYSTEM][QUEEN_SPEC]` prefix, diverging suffixes) — aggregate prefill ≤ 1.25× single-agent (matches the synthetic R-P6 ship-gate but for the natural workload shape). Falsifier on either: TTFT ratio worse than spec | ⏸ PENDING |
| 7 | Sliding-window LCP > sw edge cases + Qwen3.5 hybrid deferral note | ~50 tests + doc | Edge: prompt with LCP=4096 against sliding_window=1024; decoded bytes byte-identical to full prefill. Falsifier: any divergence at LCP > sw boundary | ⏸ PENDING |

**Promotion criterion** (env-gate flip from default-OFF to default-ON): all of iter-3 (R-C4 LCP byte-identity at K<N), iter-5 (R-C4 LCP 5-fraction sweep), iter-6 (R-P7 speedup + /cfa fan-out), iter-7 (sliding-wrap) PASS for at least 2 weeks of operator use under HF2Q_KV_LCP_RESUME=1.

**Beneficiary integrations.** /cfa singleton-sidecar integration (`docs/research/cfa-hf2q-integration-2026-05-05.md`) explicitly forward-points to E.a iter-3 as the gate at which /cfa Phase 2 fan-out (shared-prefix-different-suffix shape) starts hitting cache. Until iter-3 ships, /cfa-launched sidecars deliver R-P5 (cross-session sticky cache for byte-identical repeats) but NOT R-P6 amortization on natural fan-outs.

### 4.3 Highest-risk iter — iter-3 — codex audit required

Iter-3 modifies the load-bearing wholesale-reset invariant identified in §1. Per `feedback_codex_review_catches_unified_memory_races` (project memory, 2026-05-01 ADR-013 P21 Stage 1 precedent), unified-memory + small-window races escape Heisenbug + 5/5 reproduction + 2544/2544 tests passing. The wholesale reset is exactly that class of fence — its absence might pass every fast test and still corrupt state under load. Mandatory Codex Phase-2b audit at iter-3.

---

## §5 — Q5: Falsification gates (≥3 hypotheses + KILL criterion)

### H1 — byte-identity (CORRECTNESS)

- **Statement.** For any prompt P of length N and any K ∈ {0, N/4, N/2, 3N/4, N-1}, partial-prefill of tokens `[K..N)` against a primed cache containing `dense_kvs[*][0..K)` produces byte-identical decoded tokens to a full-prefill of `[0..N)` from a fresh empty cache.
- **Measurement.** Extend `tests/kv_persist_gemma4_roundtrip.rs::sourdough_baseline` (existing R-C4 fixture, `:1563-1730`) into an LCP variant: prime cache with token-prefix `P[..K]`, then decode P from token K onward; compare decoded byte stream to a fresh-cache full decode of P.
- **Fixture.** The existing 22-token sourdough prompt (`tests/kv_persist_gemma4_roundtrip.rs:160`), greedy + dense + Gemma 4 26B-DWQ. Five LCP fractions per fixture × 5 fixtures = 25 byte-equality assertions.
- **Bench command.** `HF2Q_KV_LCP_RESUME=1 cargo test --test lcp_resume_byte_identity -- --ignored --nocapture` (model-loaded path; mirror Phase D operator gate).
- **Expected.** All 25 byte-identical (R-C4 internal, ratio = 1 within ±0 bytes).
- **Falsifier (the KILL criterion).** Any single K × fixture pair shows divergence → **Phase E.a is FALSIFIED** → pivot to option (b)-only and abandon Phase E.a.

### H2 — speedup ceiling (PERFORMANCE)

- **Statement.** `partial_prefill_ttft / full_prefill_ttft ≤ (1 - K/N) + ε` where ε is the LCP-detection + restore overhead.
- **Computed ε** (no measurement yet — see formula): LCP detection is O(min(|P_old|, |P_new|)) integer compares; for N ≤ 32K and 64-bit-aligned scan: <10 µs on M5 Max. Dense_kvs handle clone is one `Arc::clone` = ~10 ns. Setting `write_pos`/`seq_len` for ~48 layers = ~50 ns. Sliding-window restore from existing live buffers = 0 (already in place — no copy). **ε ≈ 10 µs at N = 32K** when the cache entry is in-RAM (single-process; default for Phase E.a v1).
- **Measurement.** Bench harness measures `partial_prefill_ttft` and `full_prefill_ttft` on the same prompt, same warm GPU.
- **Expected at K/N = 0.95:** `0.05 + 10µs/full_prefill_ttft`. For Gemma 4 26B at currently-measured 70-97 t/s prefill rate (`project_w5b22_hf2q_exhausted_remaining_in_mul_mm_id`), a 1024-token prompt = ~10-15 sec full prefill; LCP = 972 (95 %) → partial = ~0.5-0.75 sec → **speedup ~13-30×**.
- **Falsifier.** ratio > (1 - K/N) + 50 ms / full_prefill_ttft for any K — indicates LCP detection or restore overhead is much higher than predicted (architectural surprise, must be diagnosed before ship).

### H3 — multi-turn floor (PRODUCTION USE-CASE)

- **Statement.** A 5-turn chat with 95 %+ prefix overlap (turn-N prompt = turn-(N-1) prompt + ~5 % new bytes) shows turn-2+ TTFT ≤ 0.10 × turn-1 TTFT.
- **Measurement.** New `R-P7` bench: simulate 5-turn `/v1/chat/completions` against same model; turn 1 cold, turns 2-5 with LCP-resume active; record per-turn TTFT.
- **Expected.** Turn 1: cold prefill ~10-15 s at L=1024 (Gemma 4 26B baseline). Turns 2-5: ~500-750 ms (5 % new tokens × full-prefill rate).
- **Falsifier.** Turn-2 TTFT / turn-1 TTFT > 0.10 — production speedup not delivered → not worth shipping.

### KILL criterion — escalates Phase E.a abandonment

- **If H1 fails** at any (K, fixture) — partial prefill is byte-INCOHERENT → ABANDON Phase E.a → pivot to option (b)-only (already shipped iter-5/6) and document that LCP partial-prefill is incompatible with the Gemma 4 dense path.
- **Cross-link.** Per ADR-009 byte-identity policy, R-C4 byte-equality regressions are HARD escalations (`docs/ADR-009-reference-parity-and-coherence-recovery.md`).

### Cold/warm M5 Max numbers (computed from already-measured baselines)

| Prompt N | Baseline full prefill (measured 70-97 t/s, project memory `w5b22`) | LCP=N-1 partial | Predicted speedup |
|---|---|---|---|
| 1024 | 10.6-14.6 s | ~10-15 ms (1 token + ε) | 700-1500× |
| 4096 | 42-58 s | ~40-60 ms | 700-1450× |
| 32768 | 338-468 s | ~340-470 ms | 700-1350× |
| 1024, K=0.95 | 10.6-14.6 s | 0.5-0.75 s | 13-30× |
| 4096, K=0.95 | 42-58 s | 2.1-3.0 s | 14-28× |

These are **computed** from per-token prefill rate × token count, not measured. Iter-6's R-P7 bench is the first measurement.

---

## §6 — Q6: Risk register (≥7 entries)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | **RoPE position drift at LCP boundary.** If `write_pos` is advanced incorrectly (e.g. set to `K mod sw` instead of `K` for sliding layers), new tokens get encoded with wrong positions. | **HIGH** — silent miscompute | R-C4 byte-identity gate (H1, §5) at K = 0, N/4, N/2, 3N/4, N-1; ANY divergence is the KILL criterion. Verified-by-test, not by review. |
| **R2** | **BPE tokenization ambiguity at LCP boundary.** Equal byte-prefix doesn't imply equal token-id-prefix; equal token-id-prefix doesn't imply that the next token will tokenize the same. | **HIGH** — silent miscompute | Token-id-level LCP only (§2.2). LCP detection always operates on the post-rendered, post-tokenized `Vec<u32>`. The `lookup` short-circuits on K=0 and on first-divergent-token mismatch. |
| **R3** | **Concurrent eviction race.** Pool evicts the cached prefix between `lcp_lookup` returning K and `forward_prefill` consuming it → reads stale or freed slots. | **HIGH** — UB in unified memory (cf. `feedback_codex_review_catches_unified_memory_races` and `solution_mlx_native_residency_lifetime_race`) | LcpRegistry pins an `Arc<DenseKvBuffers>` handle in the `LcpPrefix` returned to the caller (§2.3). Eviction can mark the entry for removal but cannot reclaim the buffer until the request drops its handle. |
| **R4** | **R-C4 sourdough regression** — wholesale reset is what currently guarantees byte-identity across iter-91 and iter-92 fixes; conditionalizing it could regress R-C4. | **HIGH** — release-blocker per ADR-009 | Default-OFF env-gate `HF2Q_KV_LCP_RESUME=1`; new R-C4 LCP variant fixture set as a required passing gate; ADR-009 escalation if any divergence. |
| **R5** | **Hybrid (Qwen3.5) family divergence.** Qwen3.5 uses `HybridPromptCache`, NOT the Gemma `PromptCache`. The dense_kvs invariants and the sliding-window arithmetic differ. | **MEDIUM** — scope risk if v1 covers Qwen3.5 | Phase E.a v1 = Gemma 4 dense ONLY. Qwen3.5 hybrid families EXPLICITLY DEFERRED to Phase B-hybrid follow-up. The `kv_restore_gemma`-style `Qwen35` bail (`engine.rs:3088-3091`) is the model. |
| **R6** | **Sliding-window wrap when LCP > sliding_window.** Naive `write_pos = LCP mod sw` would mix pre- and post-wrap content. | **MEDIUM** — silent miscompute on long-prefix multi-turn | Set `write_pos = LCP` (absolute, NOT mod sw); set `seq_len = min(LCP, sliding_window)`. Reuses existing `forward_prefill.rs:854 write_slot = tok_i % layer_dense_cap` arithmetic — no new ring code. Pinned by iter-7 R-LCP-wrap test fixture. |
| **R7** | **Multi-tenant cache poisoning.** Two tenants sharing identical token-prefix (e.g. same system prompt) could read each other's cached state. | **MEDIUM** — privacy + spec violation (cf. vLLM `cache_salt` design @ `/opt/vllm/docs/design/prefix_caching.md:86-100`) | LcpRegistry namespaced by `(tenant_key, model_fingerprint, params_key)`. Tenant key from `api_key` / `Authorization` header. Mirrors `PromptCache`'s implicit single-process single-tenant assumption + makes it explicit per vLLM's recommendation. |
| **R8** *(stretch)* | **Panic-mid-prefill leaves `kv_caches[*]` in inconsistent state for the LCP path.** Today the wholesale reset is the cleanup; conditionalizing it removes that cleanup. | **MEDIUM** — corruption on next request after panic | Phase E.a path MUST treat any `kv_caches[*].write_pos` value not equal to the LcpRegistry-recorded LCP as "bail to wholesale reset". Fail CLOSED, never OPEN. The registry's recorded `dense_kvs` Arc handle is the source of truth, not the caller's `kv_caches` state. |
| **R9** *(stretch)* | **Memory bloat — LcpRegistry retaining many Arc handles to large dense_kvs Vec.** Gemma 4 26B at sw=1024 = ~50 MB live KV per cache entry. | **LOW** | Initial design = single-slot per tenant (mirrors PromptCache). Eviction policy: replace on each new prefill. Operator can tune via `HF2Q_KV_LCP_MAX_ENTRIES`. |

---

## §7 — Q7: LOC + iter estimate / scope verdict

### 7.1 Comparative scoping

| Phase | Scope | LOC | Risk class |
|---|---|---|---|
| Phase A0 / A.1 / A.3 | Substrate (BlockPrefixCacheSpiller + KvCacheSpill trait) | 2,534 + 6,895 | low (pure infra) |
| Phase B-dense.1 | Gemma4DenseSpill — spill codec | ~1,500 | medium (touches dense_kvs read path) |
| Phase B-dense.2 | FamilyHookFactory + matrix harness | 1,911 | low (test infra) |
| Phase C.1 | EngineBindable + LoaderWrapper | 1,587 | low (additive trait) |
| Phase D | run_cell_e2e + R-C4 + R-P4 + operator script | 1,722 | medium (spill→restore plumbing) |
| **Phase E.a (this dossier)** | **LCP partial-prefill resume** | **~930** | **HIGH** (modifies load-bearing wholesale-reset invariant) |
| Phase E.b (shipped) | PromptCache cross-process replay (iter-5/6) | ~150-300 | low |

Phase E.a is **closer to Phase B-dense.1 in LOC** but **higher in risk class than any prior phase** because it modifies the load-bearing prefill path's reset semantics.

### 7.2 Iter-cost estimate

- 7 iters per §4.2.
- iter cadence ~2-3 days each (matches Phase D pacing).
- ~2-3 weeks of operator + writer + auditor cycles.

### 7.3 Verdict

**Feasible at Phase B-dense.1-class scope. Recommend env-gated rollout default-OFF.** Promote to default only after R-C4 LCP variant + R-P7 multi-turn-chat speedup gates pass for ≥2 weeks of operator validation under `HF2Q_KV_LCP_RESUME=1`.

### 7.4 Codex Phase-2b audit point

**iter-3 is the highest-risk iter.** It modifies the load-bearing wholesale-reset invariant identified in §1. Per project memory `feedback_codex_review_catches_unified_memory_races` (the ADR-013 P21 Stage 1 precedent: 5/5 Heisenbug + 2544/2544 tests + ascii_ratio=1.000 did not refute a real RAW race that Codex caught at file:line resolution), **Codex Phase-2b audit at iter-3 is REQUIRED**. Failure to flag this in the iter-3 plan is itself a process failure.

Audit checklist for iter-3:

1. Verify `kv_caches[layer].write_pos = LCP` is set BEFORE the per-token loop starts; no off-by-one.
2. Verify `dense_kvs` handle is held Arc-pinned for the duration of the request (R3).
3. Verify the env-gate truly disables the LCP path bit-for-bit when unset (R4 + ADR-009 R-C4).
4. Verify panic-cleanup invariant (R8): if LCP path panics mid-prefill, next request bails to wholesale reset.
5. Verify sliding-window `write_pos = LCP` (absolute) is correct under unified-memory + concurrent decode.

---

## §8 — Peer-impl comparison table

| Implementation | Mechanism | Granularity | Hash / equality scheme | Cache-isolation | Cross-link |
|---|---|---|---|---|---|
| **oMLX `BlockAwarePrefixCache`** (`/opt/omlx/omlx/cache/prefix_cache.py:1-120`) | **Block-aligned prefix cache** with PagedSSDCacheManager backing. Block size = 256 tokens. Prefix index = `chain-hash(block_tokens, parent_hash) → (prefix_len, block_ids, num_blocks)`. | **Block-level** (256 tokens = block_size) | Chain-hash (incremental, matches vLLM's parent_hash style) | Per-`request_id` table | hf2q analog: ADR-017 already aligns block size = 256 (`writer.rs` block boundary); LCP path could be block-aligned for the same reason vLLM/oMLX do — but token-level LCP is simpler at v1 since hf2q has only one chat slot. |
| **llama.cpp `cache_prompt` + `n_cache_reuse`** (`/opt/llama.cpp/tools/server/server-context.cpp:2358-2426`) | **Token-level common-prefix detection + KV-shift** for non-prefix-aligned reuse. `n_past = slot.prompt.tokens.get_common_prefix(input_tokens)` is the simple LCP path; `n_cache_reuse > 0` enables the more aggressive chunk-shift loop. | **Token-level** (both for the simple path AND the chunk-shift) | `tokens[i] == input_tokens[i]` integer compare | Per-`slot.id` (per-request) | hf2q's design is closer to the **simple** llama.cpp path (token-level get_common_prefix at line 2360). The chunk-shift extension at lines 2392-2423 is a v2+ enhancement requiring `llama_memory_can_shift` — equivalent to making RoPE positions relocatable, which §3.1 explicitly rules out for hf2q v1. |
| **vLLM Automatic Prefix Caching** (`/opt/vllm/docs/design/prefix_caching.md` + `/opt/vllm/docs/features/automatic_prefix_caching.md`) | **Block-hash-keyed prefix sharing** at PagedAttention block granularity; `block_hash = sha256(parent_hash, block_tokens, extra_hashes)`. Multi-tenant `cache_salt` for isolation. Cross-request reuse (not per-request). | **Block-level** (block_size = 16 by default; configurable) | sha256 chain-hash (default) or xxhash (perf) | Per-`cache_salt` (explicit) | hf2q v1 SHOULD adopt vLLM's `cache_salt` discipline for R7 (multi-tenant isolation). The single-slot LcpRegistry of Phase E.a v1 is cross-tenant-equivalent to vLLM's per-cache_salt isolation if we key by tenant. Block-level vs token-level is a v1-vs-v2 choice; token-level simplest for hf2q's single-slot start. |

**Three peer references cited**, satisfying AC-10 (≥2 of 3).

**Design divergence summary:**

- hf2q chooses **token-level** LCP at v1 (matches llama.cpp simple path) for simplicity; vLLM/oMLX block-aligned reuse is a future v2 optimization.
- hf2q rejects llama.cpp's "KV-shift" advanced reuse because RoPE-baked-in caching (§3.1) makes positional relocation unsafe.
- hf2q adopts vLLM's `cache_salt`-style tenant isolation principle (R7) without adopting the hash-based block scheme yet.

---

## §9 — If green-lit, here's iter-1's plan (concrete deliverable)

Per Walk discipline (test BEFORE code):

### iter-1 acceptance test (BEFORE any code change)

`tests/lcp_registry_unit.rs` — pure unit tests of the registry struct that DO NOT touch `forward_prefill.rs` or `engine.rs`:

```text
#[test] fn lcp_registry_returns_full_match_K_eq_N()
#[test] fn lcp_registry_returns_partial_match()
#[test] fn lcp_registry_returns_none_on_fingerprint_mismatch()
#[test] fn lcp_registry_returns_none_on_tenant_mismatch()
#[test] fn lcp_registry_returns_none_on_params_mismatch()
#[test] fn lcp_registry_pins_dense_kvs_arc_across_eviction()  // R3
#[test] fn lcp_registry_evicts_lru_on_capacity_pressure()
#[test] fn lcp_registry_full_equality_K_eq_N_returns_none()  // never masks option (b)
```

### iter-1 file list + LOC

- `src/serve/kv_persist/lcp_registry.rs` — ~250 src LOC (struct + insert/lookup + tenant/fingerprint/params keying + Arc handle + LRU eviction)
- `tests/lcp_registry_unit.rs` — ~100 test LOC (8 tests above)
- `src/serve/kv_persist/mod.rs` — +1 line `pub mod lcp_registry;`
- **TOTAL iter-1 = ~351 LOC additive, zero changes to forward_prefill.rs / engine.rs prefill path.**

### iter-1 falsifier

ANY of the 8 unit tests fails → iter-1 does not land. The registry is the foundation for iter-3's risky modification; if the registry itself is wrong, the rest of the chain is unsafe.

### iter-1 commit message template

```text
feat(adr-017 phase-e-a iter-1): LcpRegistry with token-prefix lookup + tenant/fingerprint/params keying

Standalone struct; not yet wired into engine.rs. 8 unit tests pin the
correctness invariants identified in research dossier
docs/research/adr017-phase-e-option-a-2026-05-05.md §4.1 + §6 R3.

Phase E.a iter-2 = unit-test integration (already in this commit).
Phase E.a iter-3 = forward_prefill.rs conditional reset (HIGH-RISK,
requires Codex Phase-2b audit per project memory entry
feedback_codex_review_catches_unified_memory_races).
```

---

## Appendix A — File:line citation manifest (anchors)

For Codex Phase-2b citation verification:

- `forward_prefill.rs:412-448` — wholesale reset block + Iter-92 / Phase 2a Task #8 / "when prompt-cache lands" comment (§1.1)
- `forward_prefill.rs:445-448` — the actual reset loop (`cache.write_pos = 0; cache.seq_len = 0`)
- `forward_prefill.rs:460-471` — sliding-ring permutation-invariance argument (§1.2 I-3)
- `forward_prefill.rs:484-506` — dense_kvs allocation block (§3.4)
- `forward_prefill.rs:617-625` — per-token loop start, `seq_pos = tok_i` absolute-position threading (§3.2)
- `forward_prefill.rs:763-782` — fused per-head RMSNorm + RoPE on Q and K (§3.1)
- `forward_prefill.rs:850-869` — sliding-ring `write_slot = tok_i % layer_dense_cap` (§3.4)
- `forward_prefill.rs:1500-1502` — `self.dense_kvs = Some(dense_kvs_vec)` (§3.5)
- `forward_mlx.rs:568` — `pub kv_caches: Vec<MlxKvCache>` (§3.4)
- `forward_mlx.rs:572` — `pub sliding_window: usize` (§3.4)
- `forward_mlx.rs:585-593` — `dense_kvs` field doc with "RoPE is baked in before caching" claim (§3.1)
- `forward_mlx.rs:734-743` — `DenseKvBuffers` struct definition (§3.4)
- `engine.rs:1452-1505` — `PromptCache` doc + iter-96 + iter-97+ scope marker (§2.1, ADR-017 doc:3439)
- `engine.rs:1490` — iter-97+ scope marker line (`// LCP-based partial-prefill resume`)
- `engine.rs:1588-1648` — `PromptCache::lookup` / `lookup_with_fragments` (§2.1)
- `engine.rs:3071-3098` — `Request::KvRestore` worker handler (kv_restore_gemma dispatch, §4.1)
- `engine.rs:3101-3118` — `Request::PromptCacheSnapshot` (option (b) shipped iter-5)
- `engine.rs:3120-3149` — `Request::PromptCacheRestore` (option (b) shipped iter-5)
- `engine.rs:3314-3403` — `kv_restore_gemma` allocator + restore (§4.1 — change site)
- `engine.rs:3658-3664` — `loaded.prompt_cache.lookup(prompt_tokens, params)` call site (§2.1)
- `kv_persist/format.rs:213-281` — `compute_model_fingerprint` + `ModelFingerprint` (§3.3)
- `kv_persist/prompt_cache_persist.rs` — option (b) reference impl (§2.1, contrast)
- `tests/kv_persist_gemma4_roundtrip.rs:1563-1730` — R-C4 sourdough internal harness (§1.3)
- `tests/kv_persist_gemma4_roundtrip.rs:160` — sourdough prompt fixture string (§5 H1)
- `docs/ADR-017-persistent-block-prefix-cache.md:3434-3461` — iter-4 finding + Phase E motivation (§Q1, TL;DR)
- `docs/ADR-017-persistent-block-prefix-cache.md:3439` — iter-97+ scope quote (§1.1)
- `docs/ADR-017-persistent-block-prefix-cache.md:3465-3507` — Phase E iter-5 PromptCache replay (option (b), shipped)
- `/opt/omlx/omlx/cache/prefix_cache.py:1-120` — peer reference, oMLX BlockAwarePrefixCache (§8)
- `/opt/llama.cpp/tools/server/server-context.cpp:2358-2426` — peer reference, llama-server `cache_prompt`/`n_cache_reuse` (§8)
- `/opt/vllm/docs/design/prefix_caching.md:1-100` — peer reference, vLLM Automatic Prefix Caching with `cache_salt` (§8, R7)

---

## §10 — Codex Audit Findings (Phase 2b) + Writer Reconciliation

The Phase 2a dossier was independently audited by Codex (verdict: `request_changes`, severity `high`, 5 HIGH issues + 2 missed risks + 5 MED-severity citation drifts). After Queen-side verification against actual source, **all 5 HIGH-severity issues + both missed risks are ACCEPTED** as load-bearing for the iter-1 implementer; this section folds Codex's findings into the design that the iter-1 implementer must follow. **The original §1-§9 design above is non-binding where it conflicts with §10.x; §10 supersedes.**

### 10.1 [HIGH-1, codex-on-line-207] §3.2 byte-identity is necessary-but-not-sufficient

**Codex finding.** Setting `write_pos = K, seq_len = K_seq` and starting the per-token loop at `tok_i = K` is necessary but NOT sufficient. The byte-identity claim implicitly assumes (a) `dense_kvs[*][0..K)` is populated by the SAME buffer instance that received the original positions `[0..K)` writes (no copy/move/realloc), and (b) the per-token loop preserves *absolute* indices.

**Verified against source.** `forward_prefill.rs:1502` unconditionally stores the per-prefill-allocated `dense_kvs_vec` into `self.dense_kvs`; this is the buffer that `forward_decode` later reads at `forward_mlx.rs:2432-2588`. So buffer identity matters.

**Reconciliation — supersedes §3.2 + §2.3.** The byte-identity condition is restated as a 5-clause conjunction:

> Partial prefill `[K..N)` produces byte-identical logits at every position `p ≥ K` iff:
>
> (a) `kv_caches[layer].write_pos = K` and `kv_caches[layer].seq_len = min(K, sliding_window | linear_capacity)` for every layer **before** the per-token loop starts;
> (b) `dense_kvs[layer]` is the **same heap buffer** that originally received writes for positions `[0..K)`, with `capacity ≥ N + max_decode_tokens` for global layers (see §10.2);
> (c) the per-token loop iterates with `for tok_i in K..prompt_tokens.len()` (i.e. uses `enumerate().skip(K)` semantics, NOT `prompt_tokens[K..].iter().enumerate()` which would re-zero `tok_i`); the `seq_pos = tok_i` line at `forward_prefill.rs:618` then threads the absolute position correctly;
> (d) `model_fingerprint`, `theta`, sliding/global config, and dtype are all unchanged (already required by `LcpEntry.fingerprint` check in §2.3);
> (e) for sliding layers with LCP > sliding_window, `write_pos = LCP` (absolute, not `LCP mod sw`); see §3.4 — already correct.

**iter-1 implementer note.** Pseudocode for the loop in §2.3 was implicit; explicit form is:
```rust
for tok_i in start_tok_idx..prompt_tokens.len() {
    let token = prompt_tokens[tok_i];     // absolute index into prompt_tokens
    let seq_pos = tok_i;                  // absolute KV write position
    // …existing loop body…
}
```
where `start_tok_idx = restored_lcp.unwrap_or(0)`.

### 10.2 [HIGH-2, codex-on-line-249] dense_kvs global-layer capacity must be re-validated

**Codex finding.** `forward_prefill.rs:492` allocates `linear_capacity = seq_len + max_decode_tokens` per request; longer subsequent prompts overflow the cached buffer. The dossier's "skip re-allocation if matching capacity" hides this hole.

**Verified against source.** Confirmed at `forward_prefill.rs:484-506`: per-prefill allocation, sized to current `seq_len + max_decode_tokens`. Sliding layers always `sw` so are size-stable across requests; **global layers are NOT**.

**Reconciliation — supersedes §2.3 + §4.1 forward_prefill.rs row.** The LCP path's three-case capacity protocol:

| Case | Condition | Action |
|---|---|---|
| **C1: Reuse-in-place (zero-copy)** | `cached_buf.capacity >= new_seq_len + max_decode_tokens` for every global layer AND `cached_buf.capacity == sliding_window` for sliding layers | Reuse cached `Arc<DenseKvBuffers>`. Fast path. |
| **C2: Resize-and-copy** | Any global layer's `cached_buf.capacity < new_seq_len + max_decode_tokens` | Allocate fresh per-layer buffers at `new_seq_len + max_decode_tokens`; `memcpy` positions `[0..K)` from the cached buffer to the fresh one; release Arc handle. Adds a one-time copy of `K × n_kv_heads × head_dim × kv_elem_bytes` bytes per global layer. |
| **C3: Bail to wholesale reset** | Sliding-layer capacity mismatch (e.g. config change) | Disable LCP path for this request; fall through to current §1.1 reset semantics. Fail closed (R8 / I-4). |

**Operator switch.** Add `HF2Q_KV_LCP_REUSE_POLICY={zero_copy_only,allow_resize_copy}` defaulting to `zero_copy_only` for v1; `allow_resize_copy` is a v2 toggle once iter-3's R-C4 LCP variant gate is green for both cases.

**LOC delta.** Resize-and-copy logic adds ~60 LOC to the `forward_prefill.rs` row; revised estimate **~140 LOC** (was ~80) — see §10.4.

### 10.3 [HIGH-3, codex-on-line-175] dense_kvs ownership refactor is in-scope, not zero-LOC

**Codex finding.** Current `MlxModelWeights::dense_kvs: Option<Vec<DenseKvBuffers>>` (`forward_mlx.rs:593`) is OWNED, not Arc-wrapped. The dossier's `Arc<DenseKvBuffers>` and `dense_kvs_arc.clone()` (§2.3, §6 R3) implicitly require an ownership refactor that §4.1 marks as 0 LOC for `forward_mlx.rs`.

**Verified against source.** `forward_mlx.rs:593` is owned; consumer sites at `forward_mlx.rs:2432-2588` deref directly through `as_ref().unwrap()`. Wrapping in `Arc<Vec<DenseKvBuffers>>` (or `Vec<Arc<DenseKvBuffers>>`) requires touching every consumer.

**Reconciliation — supersedes §4.1 forward_mlx.rs row.** Choose ONE of two ownership strategies; both are scoped explicitly:

- **Strategy A (recommended for v1): per-layer `Arc<DenseKvBuffers>`.** Change field to `dense_kvs: Option<Vec<Arc<DenseKvBuffers>>>`. Consumer sites at `forward_mlx.rs:2432-2588` change from `&dense_kvs[layer_idx].k` to `&dense_kvs[layer_idx].as_ref().k` (one-line per consumer; ~24 sites, ~25 LOC). LcpRegistry stores `Vec<Arc<DenseKvBuffers>>`. Layer-level granularity allows resize-and-copy (§10.2 C2) to swap individual layers without touching others.

- **Strategy B (alternative): outer `Arc<Vec<DenseKvBuffers>>`.** Change field to `dense_kvs: Option<Arc<Vec<DenseKvBuffers>>>`. Consumer sites become `dense_kvs[layer_idx]` (no per-layer arc). LcpRegistry stores one `Arc<Vec<...>>` per entry. Cheaper consumer-side, but C2 resize requires whole-Vec rebuild.

**LOC delta.** Strategy A: **~25 LOC in forward_mlx.rs** (was 0); revised total LOC estimate **~955** (was ~930). Strategy B: ~10 LOC in forward_mlx.rs.

### 10.4 [HIGH-4, codex-on-line-263] Iter-3 test must force partial K<N AND bypass PromptCache full-equality

**Codex finding.** §4.2 iter-3 test "prime cache with P, then issue P with LCP=full" is internally inconsistent: §2.3 returns `None` on `K == new_tokens.len()`, AND `PromptCache::lookup` (`engine.rs:1588-1648`) intercepts on full equality first. So the proposed iter-3 test never reaches the LCP path.

**Verified against design.** §2.3 line 172 explicit: `if k == 0 || k == new_tokens.len() { return None; }`. Codex correct.

**Reconciliation — supersedes §4.2 iter-3 row + §5 H1 measurement.** Iter-3's same-process test is restated as:

> Prime cache with prompt `P` of length `N=24` (e.g. sourdough fixture). Then issue prompt `P' = P[..K] || extra_tokens` where `K=12` and `extra_tokens.len()=4` (so `P'.len()=16`, K<P'.len(), partial-prefix). Assert: (a) PromptCache full-equality misses (`P' != P`); (b) LcpRegistry returns `Some(LcpPrefix { k: 12, ... })`; (c) decoded bytes from `P'` byte-identical to a fresh-cache full-prefill of `P'`.

This reaches the LCP partial-prefill code path. The earlier "LCP=full" framing was incorrect.

### 10.5 [HIGH-5 MISSED-RISK-6, codex-on-line-181] SoftTokenInjection / multimodal scope

**Codex finding.** `forward_prefill.rs:61` defines `SoftTokenInjection` (multimodal image-embedding patches replace token-id positions). LCP keyed on token IDs alone is unsafe — two requests with identical token IDs but different image embeddings would collide.

**Verified against source.** `forward_prefill.rs:359 forward_prefill_with_soft_tokens(prompt_tokens, soft_tokens, ...)` accepts `&[SoftTokenInjection]`. `engine.rs:45 use SoftTokenInjection`. `engine_qwen35.rs:910/1099/1396` thread soft tokens through. Real surface, dossier silent.

**Reconciliation — supersedes §2 + §6.** Phase E.a v1 scope is **TEXT-ONLY**:

> **Scope gate (Phase E.a v1).** LCP path is taken ONLY when `soft_tokens.is_empty()` is true at the call site in `engine.rs`. Any request with `SoftTokenInjection` ranges → bail to wholesale-reset path. The LcpRegistry `store` call is gated on the same condition. Multimodal LCP support is deferred to **Phase E.a v2** which would extend `LcpEntry` with a content-hash over `(token_id, soft_token_payload)` per position.

This is a strict subset that closes the risk; v2 is forward-pointed. Adds ~5 LOC to `engine.rs` plumbing.

### 10.6 [MISSED-RISK-7, codex-on-line-335] Risk register additions

Risks **R10, R11, R12** appended:

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R10** | **dense_kvs global-layer capacity overflow on longer-N follow-up.** Per-prefill allocation sized to current `seq_len + max_decode_tokens`; longer subsequent prompt overflows. | **HIGH** — silent OOB or hard panic | §10.2 three-case protocol; v1 default `HF2Q_KV_LCP_REUSE_POLICY=zero_copy_only` (bail if any global layer would overflow). |
| **R11** | **Multimodal soft-token cache poisoning.** Two requests with identical token IDs but different image embeddings would collide if LCP key is token-IDs-only. | **HIGH** — silent miscompute on multimodal | §10.5 — Phase E.a v1 scope is text-only; multimodal requests bail to wholesale reset. v2 adds soft-token-payload hash to LCP key. |
| **R12** | **Prior-prompt KV vs generated-assistant KV mismatch.** Cached `dense_kvs[*][0..K)` was generated by prefill of the *prior* prompt; the new prompt's `[0..K)` IS that prior prompt verbatim, but the K..N tail reads against KV that was originally written under one consistent prefill order. If a request's `tokens` field stores prompt+decoded-output concatenation (as the iter-96 PromptCache does at `engine.rs:1508`), the LCP semantic must distinguish "cached prompt prefix" from "cached prompt+assistant tail". | **MEDIUM** — semantic drift if registry conflates | LcpRegistry stores `prompt_tokens` only (the prompt portion), NEVER `prompt + decoded` concatenation. The dense_kvs handed off was populated only for prompt positions; if decode wrote additional positions [N..N+M), those slots are NOT part of the LCP-eligible prefix. Drop them at `lcp_lookup` time by clamping `K ≤ cached.prompt_len`. |

### 10.7 [LOC] Revised LOC estimate

Folding §10.1 (loop-indexing rephrase, ~5 LOC), §10.2 (capacity three-case, +60 LOC in forward_prefill), §10.3 Strategy A (+25 LOC in forward_mlx), §10.5 (multimodal-bail gate, +5 LOC in engine):

| File | Original | Revised |
|---|---:|---:|
| `forward_prefill.rs` | ~80 | **~140** |
| `forward_mlx.rs` | ~0 | **~25** |
| `engine.rs` | ~150 | **~155** |
| `kv_restore_gemma` | ~50 | ~50 |
| `lcp_registry.rs` (new) | ~250 | ~250 |
| Tests | ~410 | **~440** (iter-3 + multimodal-gate test added) |
| **TOTAL** | **~930** | **~1,060** |

Still in Phase B-dense.1 weight class; ~14% upward revision is the cost of explicitness.

### 10.8 [MED] Citation drifts corrected

- **§1.1 / AC-3.** Verbatim quote spans `forward_prefill.rs:412-448`; the AC required line 433-444 specifically. Updated wording: "verbatim block from lines 412-448 (the AC required range 433-444 is contained within and quoted in full context)".
- **§1.4 I-4 panic-cleanup.** Codex flagged this is plausible safety speculation, not an evidenced source invariant. Reclassified: I-4 remains in the table but tagged "design constraint, not a documented source invariant — must be enforced by R8 fail-closed protocol" rather than implied as already-pinned.
- **§2.4 BPE example.** Concrete token IDs `4581 / 7799 / 9990 / 113` are illustrative, not validated against Gemma 4's actual tokenizer. Marked "illustrative; iter-2's lcp_registry unit tests use real Gemma 4 tokenizer outputs as fixtures, not these IDs".
- **§8 oMLX cache-isolation.** "Per-`request_id` table" is misleading — `request_id` tracks active request tables, not a security boundary. Corrected to "Per-`request_id` request-scoped table; multi-tenant isolation in oMLX is a system-level concern handled outside this struct".
- **Appendix A engine.rs:1490.** Replaced reference; the iter-97+ scope marker is at `engine.rs:1461-1468`. Line 1490 is inside the *deleted-prototype historical note* (lines 1488-1492) and is not the scope marker. Appendix entry corrected.
- **ADR-017 doc:1442 (referenced in dossier TL;DR).** Stale line; current source has the LCP doc-comment at `engine.rs:1461-1468`. Noted as line-drift in the source ADR (out of dossier scope to fix in ADR-017 itself; flagged for ADR-017 maintainer).

### 10.9 Updated iter-1 implementer instructions

The iter-1 plan in §9 stands, with this addition:

> **Before iter-3 lands**, the implementer MUST also have:
> - Strategy A ownership refactor (`dense_kvs: Option<Vec<Arc<DenseKvBuffers>>>`) — can be a separate iter-2.5 commit (~25 LOC + read-only consumer updates).
> - The §10.2 three-case capacity protocol coded into `forward_prefill.rs` LCP branch.
> - The §10.5 `soft_tokens.is_empty()` scope gate at the LCP-path entry point in `engine.rs`.
>
> iter-3's Codex Phase-2b audit MUST verify all three are present before approving.

### 10.10 Verdict

**Approve-with-addendum.** Original §1-§9 design is sound at the architectural level (fence audit verbatim correct, RoPE observation correct, peer impl table correct, iter-3 audit-required flag correct, falsification gates are real). Codex caught design holes that would cost the iter-1 implementer 1-3 days; §10 closes them in design, not in code. **iter-3 audit-required flag survives.** Dossier ships.

---

## Appendix B — Acceptance-criteria mapping

| AC | Where addressed |
|---|---|
| AC-1 file exists | This file at `/opt/hf2q/docs/research/adr017-phase-e-option-a-2026-05-05.md` |
| AC-2 7 sections | §1 Q1 / §2 Q2 / §3 Q3 / §4 Q4 / §5 Q5 / §6 Q6 / §7 Q7 |
| AC-3 verbatim 433-444 quote + 4 invariants | §1.1 verbatim block; §1.2 I-1 I-2 I-3 I-4 |
| AC-4 LCP detection design | §2.1 (site C) + §2.2 (token-id) + §2.4 (BPE merge edge case) |
| AC-5 byte-identity one-line + sliding wrap | §3.2 + §3.4 |
| AC-6 LOC per file + 7-iter sequence + falsifiable measurements | §4.1 + §4.2 |
| AC-7 ≥3 hypotheses + cold/warm M5 Max numbers | §5 H1 H2 H3 + computed Cold/warm table |
| AC-8 ≥5 risks (target 7) | §6 R1-R7 (+ R8 + R9 stretch) = 9 risks |
| AC-9 Phase B/D scoping verdict + iter-3 audit | §7.1 + §7.4 |
| AC-10 ≥2 of 3 peer references | §8 has all 3 (oMLX + llama.cpp + vLLM) |
| AC-11 git status clean except dossier | NO production code changed; verified via `git status --porcelain` |
| AC-12 NO new tests | Tests proposed in §4.1 + §5 + §10.4 only as future-iter design |
| AC-13 (post-2b) Codex audit folded | §10.1-§10.10 — 5 HIGH + 2 missed-risk + 5 MED accepted, design patched in place |

— End of dossier —
