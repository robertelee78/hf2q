# ADR-017 Phase A0 — falsification-harness results (Phase A0.2b run)

**Status:** Phase A0.2b matrix executed against the patched substrate (3 substrate defects + 1 chat-template fallback fix). Per-cell numbers below come from a real `hf2q serve` subprocess driven by `tests/kv_persist_harness.rs::kv_persist_matrix_e2e` against the production Gemma 4 26B Q4_0 GGUF (`/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/`). Matrix wall: **2,659.30 s** (~44 min) for 12 ship-gate-relevant cells.

**Verdict (per `feedback_substrate_must_not_synthesize_ship_gates` — these are the honest evaluations against the measured fields, not massaged numbers):**

| Gate | Status | Cell | Ratio / wall | Bar |
|---|---|---|---|---|
| **R-P4** | **PASS** | L32K SwapBackInSameCtx | ratio = 0.009 (cache_hit=4,973ms / no_cache=575,502ms) | ≤ 0.20 |
| **R-P5** | **PASS** | L32K SsdColdPostRestart | ratio = 0.009 (cache_hit=5,398ms / no_cache=630,212ms) | ≤ 0.15 |
| R-P6 | N/A | (no SharedPrefix4Agents L4K-equiv cell with measured `shared_prefix_ratio`) | — | ≤ 1.25 |
| R-C1 | **PASS** | 12 dense cells | byte-exact `kv_sha256_pre == kv_sha256_post` | match |
| R-P1 | **PASS** | 12 cells | decode regression ≤ 1 % | ≤ 1 % |
| **R-P3** | **FAIL** | L32K (4 of 12 cells with prefix=L32K) | pre_evict ≈ **517 ms** | ≤ 200 ms |

**Verdict line:** `PARTIAL` — R-P4 / R-P5 PASS; R-P3 (eviction overhead at L32K) **FAIL**.

**Caveat on R-P4 / R-P5 attestation (load-bearing for the Phase A.1 next step):** the `cache_hit_ttft_ms` column is the output of `subprocess_driver::synthesize_cache_hit_prediction` (`tests/kv_persist_harness.rs:~1200-1280`), which is a CLOSED-FORM PROXY that combines real disk I/O timing (via `BlockStore::time_round_trip`) with a `no_cache_ttft / n_blocks` final-block engine cost. It is NOT the cache-hit TTFT a production KV-spill / restore path would emit — that cost lives in Phase A.1's per-family `KvCacheSpill` impl. The substrate's R-P4 / R-P5 verdicts therefore say "**disk-I/O round-trip + per-block engine proxy stays well below 20 % of no-cache prefill at 32K on M5 Max APFS**", not "production cache-hit TTFT meets the ship-gate." That's the falsification-instrument intent (per ADR-017 §A0); production attestation is the Phase A.1 deliverable. R-C1 is the only A0 gate that asserts on actual K/V byte-trip semantics (and PASSES on every cell).

**R-P3 honest reading (the FAIL):** `pre_evict_ms` ≈ **517 ms** at L32K is the synthetic spiller fixture's `BlockStore::pre_evict_blocks` real-disk-I/O wall for the L32K block count (~128 blocks @ 1 MiB each). The 200 ms ceiling in §Performance R-P3 was specced before A0 measured M5 Max APFS write+fsync throughput at 128-block batch; the floor is **APFS write throughput**, not a hf2q-side bug. Phase A.1 must either (a) raise the R-P3 ceiling to ~600 ms after re-spec on measured floor (consistent with the "measure before optimize" mantra), or (b) introduce a `flush_buffer` write-aggregation pattern (write multiple blocks to a single tempfile + atomic-rename — already flagged as a B-dense reserve in `docs/ADR-017-persistent-block-prefix-cache.md:312`). The data is the input to that decision.

## Reproducer

```bash
# 1. Pre-bench process audit (fail if mcp-brain-server / llama-server / ollama running)
ps -Ao comm,pid,%cpu | grep -E 'mcp-brain-server|llama-server|llama-cli|ollama' || echo OK

# 2. STOP-pause mcp-brain-server (per feedback_bench_process_audit)
kill -STOP $(pgrep mcp-brain-server) 2>/dev/null || true

# 3. Run matrix on M5 Max (clean SoC required)
HF2Q_KV_PERSIST_E2E=1 \
HF2Q_KV_PERSIST_E2E_SHIP_GATE_ONLY=1 \
HF2Q_KV_PERSIST_E2E_MODEL_PATH=/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
RUST_LOG=hf2q=warn \
  cargo test --release --test kv_persist_harness \
  -- --test-threads=1 --nocapture kv_persist_matrix_e2e

# 4. Resume mcp-brain-server
kill -CONT $(pgrep mcp-brain-server) 2>/dev/null || true
```

## Thermal-state log

Run executed 2026-04-30 13:51 → 14:36 PDT on M5 Max 128 GiB. Pre-bench `pre_bench_process_audit_or_panic` PASSED — no `mcp-brain-server` / `llama-server` / `llama-cli` / `ollama` detected. SoC was warm (mid-day run after a 50-minute build cycle); per `feedback_perf_gate_thermal_methodology`, perf-grade re-runs should sequence cold-SoC first.

## mcp-brain-server STOP/RESUME log

`pgrep mcp-brain-server` returned no PIDs at run start (clean SoC); explicit STOP/CONT pairs not required.

## Per-cell results

| Cell label | Ran | no_cache TTFT (ms) | cache_hit TTFT (ms) | pre_evict (ms) | insert (ms) | load (ms) | kv_sha256 pre/post match | actual_prompt_tokens | retry_count | note |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemma4_26b/Q4_0/Dense/L0/Miss/ColdResume | yes | 213.303 | 213.335 | 0.000 | NaN | NaN | match | 11 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(11), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L0/Miss/SharedPrefix4Agents | yes | 220.596 | 220.615 | 0.000 | NaN | NaN | match | 11 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(11), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L512/Miss/ColdResume | yes | 4213.684 | 2113.843 | 9.658 | NaN | NaN | match | 409 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(409), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L512/Miss/SharedPrefix4Agents | yes | 4205.284 | 2109.825 | 8.002 | NaN | NaN | match | 409 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(409), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L2048/Miss/ColdResume | yes | 20358.648 | 2574.292 | 31.208 | NaN | NaN | match | 1945 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(1945), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L2048/Miss/SharedPrefix4Agents | yes | 21555.775 | 2738.632 | 32.265 | NaN | NaN | match | 1945 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(1945), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L8192/Miss/ColdResume | yes | 103738.701 | 3366.127 | 126.221 | NaN | NaN | match | 9137 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(9137), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L8192/Miss/SharedPrefix4Agents | yes | 108181.531 | 3503.532 | 124.197 | NaN | NaN | match | 9137 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(9137), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L32768/Miss/ColdResume | yes | 568638.986 | 4914.284 | 517.100 | NaN | NaN | match | 39857 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(39857), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L32768/Miss/SharedPrefix4Agents | yes | 567354.079 | 4903.272 | 518.175 | NaN | NaN | match | 39857 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(39857), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L32768/Miss/SwapBackInSameCtx | yes | 575502.293 | 4973.321 | 519.153 | 2855.513 | 2855.496 | match | 39857 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(39857), retry_count=0 |
| Gemma4_26b/Q4_0/Dense/L32768/SsdColdPostRestart/ColdResume | yes | 630212.171 | 5397.802 | 512.285 | NaN | NaN | match | 39857 | 0 | subprocess driven; tokens_total=16, log_lines=29, actual_prompt_tokens=Some(39857), retry_count=0 |

## TTFT scaling — direct evidence the substrate fixes landed

Pre-fix (iter `b74284c`, log captured in this file's prior revision): `no_cache_ttft` was **flat at ~234 ms** across every prefix length — the BPE-collapse defect drove the user prompt to <50 actual tokens, and ship-gate ratios were denominator-broken. Post-fix:

| `actual_prompt_tokens` | `no_cache_ttft` (ms) | ms / token (rough) |
|---:|---:|---:|
| 11 | 213 | — (boilerplate floor) |
| 409 | 4,214 | ~10.3 |
| 1,945 | 20,359 | ~10.5 |
| 9,137 | 103,739 | ~11.4 |
| 39,857 | 568,639 | ~14.3 |

The TTFT now scales linearly-with-quadratic with actual prompt token count, which is the expected `O(n)` prefill + attention `O(n²)` shape on a Gemma 4 26B Q4_0 dense kernel. **`actual_prompt_tokens` IS the load-bearing signal** — the harness's nominal `cell.prefix_len` was a target; the SSE-`usage`-block parse landed in defect-1 fix gives ground truth. All 12 cells reported a finite `actual_prompt_tokens` value.

## Substrate-defect closure

| # | Defect | Status | Evidence |
|---|---|---|---|
| 1 | TTFT does not scale with prefix length | **FIXED** | `actual_prompt_tokens` ranges 11 → 39,857 across the 12 cells; `no_cache_ttft` ranges 213 ms → 575,502 ms |
| 2 | SwapBackInSameCtx returns HTTP 500 "Failed to parse config.json" | **FIXED** | `SwapBackInSameCtx` cell at L32K finished with finite `insert_ms=2855.5` and `load_ms=2855.5`; no HTTP 500 |
| 3 | SSE transport errors on short prompts | **FIXED** | every cell reports `retry_count=0` (no transient failures observed in this run; the retry harness is in place for any future flake) |
| 4 | API-path Gemma4 chat-template fallback dropped user content | **FIXED** | 200-word probe pre-fix: 14 prompt_tokens; post-fix: 697 prompt_tokens; matrix shows full token-stream-sized prefills |

## Summary

- **yes:** 12 cell(s)
- **Total cells:** 12
- **KV cache format version:** 1
- **Block size:** 256 tokens
- **Matrix wall:** 2,659.30 s (~44 min) on M5 Max 128 GiB Q4_0 dense Gemma 4 26B
- **Verdict:** `PARTIAL` — R-P4 / R-P5 PASS (synthetic predictor); R-P3 FAIL (eviction overhead at L32K, 517 ms vs 200 ms ceiling)
- **Discipline:** zero shipped `// TODO` markers; ratios reported honestly (substrate must not synthesize ship gates per `feedback_substrate_must_not_synthesize_ship_gates`)
