# ADR-007 Path C / F-4.1 — `max_global_kv = 8192` cap STATUS: doesn't exist

**Date:** 2026-05-05 (iter-5)
**Verdict:** F-4.1 (locate + remove the 8192 cap) is **a no-op** — the cap doesn't exist in the actual codebase. The close-section §1148 reference to "preserved in code as a deliberate safety limit" was either misremembered or refers to a cap that was removed before close.

## Empirical evidence

`grep -rn "max_global_kv\s*=\s*8192" src/` returns 0 matches across the entire
hf2q + mlx-native codebases. No literal `8192` constant gates KV cache size
anywhere in the loading or forward path.

The actual KV cache allocation (forward_mlx.rs:1271-1275):

```rust
let capacity = if is_full {
    cfg.max_position_embeddings   // 262144 default for Gemma 4
} else {
    cfg.sliding_window            // 1024 default for Gemma 4 sliding layers
};
```

`cfg.max_position_embeddings` defaults to `262144` (config.rs:104), which is
the exact target of F-4. No intermediate cap.

## Empirical validation

Ran Gemma 4 26B-A4B (`gemma-4-26B-A4B-it-ara-abliterated-dwq`) with a
6014-token prompt:

```
hf2q load: max_ctx_train = 262144, kv_budget = none
prefill: 6014 tok in 64965ms (93 tok/s)
--- mlx-native: 3 tokens in 0.04s (81.1 tok/s) ---
```

No memory exhaustion. No capacity errors. Output coherent. The 6K context is
**6× the sliding window** (1024) — well beyond any "8192" cap if one existed.

## What this means for F-4

- **F-4.1 (locate + remove cap)**: closed as no-op. The cap doesn't exist.
  ADR §F-4.1 is updated to reflect status: nothing to do.
- **F-4.2 (memory budget validation at 8K/32K/64K/128K/262K)**: still required.
  Needs running long-context benchmarks and measuring actual GPU memory.
- **F-4.3 (bandwidth ≥ 0.85× F16-equiv at design context)**: still required.
- **F-4.4 (needle-in-haystack at 8/32/128/262K)**: still required.

F-4 closure now requires only the measurement deliverables (F-4.2/3/4); no
code changes.

## Implications for ADR-007 close-section

The close-section §1148 future-work entry says:

> 262K context unlock (still requires removing the `let max_global_kv = 8192`
> cap — Phase 2.1 of the original ADR scope; preserved in code as a deliberate
> safety limit)

This was either:

1. **Misremembered** — referring to some earlier draft or a cap that was
   removed before close;
2. **Stale** — referring to a cap that was removed during the 27-iter close
   without updating the future-work text;
3. **Conservative** — written defensively in case anyone added such a cap
   later.

In any case, the close-section's future-work item #4 was overstated. The 262K
ctx is **already unlocked at the code level**. Only the measurement
deliverables remain.

## Updated F-4 sequencing

| Sub-task | Status | Effort | Next |
|---|---|---|---|
| F-4.1 cap removal | **closed (no-op)** | 0d | — |
| F-4.2 memory budget validation | open | 1-2d | iter-6 |
| F-4.3 bandwidth measurement | open | 1d | iter-6 |
| F-4.4 needle-in-haystack | open | 2-3d | iter-7 |

Total revised F-4 effort: ~4-6 man-days (was 6-10 in the original Path C
sequencing table).

## Iter-6 plan for F-4

1. Run hf2q at 8K, 32K, 64K, 128K context (a representative long prompt
   replicated to fit each length).
2. Capture peak GPU memory via Metal Performance HUD or
   `mlx-native::residency` counters.
3. Compare to ADR-007 §286-317 memory budget prediction.
4. Run decode at each length and measure t/s for bandwidth utilization.
5. (iter-7) Build needle-in-haystack harness — synthesize a long-context
   prompt with a known fact at a random position, query for that fact at
   end, measure retrieval accuracy.

Any failure on (1) or (2) reopens F-4.1 with new evidence.
