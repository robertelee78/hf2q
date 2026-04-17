# Spike: Gate A (prefill parity) — concrete measurement + architectural finding

**Date:** 2026-04-16
**Author:** party-mode session (Task #5 of ADR-005 Phase 1b closeout amendment)
**Status:** Complete — raises an architectural question back to the amendment

## Goal

Produce the concrete number for Phase 1b Gate A (prefill tok/s parity vs llama.cpp on a ≥2048-token prompt), per the 2026-04-16 Closeout Amendment in ADR-005.

## Method

**Prompt fixture** (committed this session): `tests/evals/prompts/prefill_2048.txt`. Constructed as `adversarial_1000.txt × 3` (repeat, no separators); 11,592 bytes → **2455 tokens** per hf2q's tokenizer, **2443 tokens** per llama.cpp's tokenizer (12-token delta is normal for the Unicode/BOS path difference).

**Hardware / model**: M5 Max, Gemma 4 26B MoE DWQ GGUF at `/opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf`. HEAD `ee322b6`.

**hf2q invocation:**
```
target/release/hf2q generate \
  --model <gguf> --prompt-file tests/evals/prompts/prefill_2048.txt \
  --max-tokens 1 --temperature 0
```
`max-tokens=1` means the run times prefill + one decode step; prefill dominates.

**llama.cpp invocation:**
```
llama-completion --model <gguf> --file tests/evals/prompts/prefill_2048.txt \
  --predict 1 --temp 0 --seed 42 --no-display-prompt -no-cnv -ngl 999 -st </dev/null
```
`-no-cnv` avoids the conversation-mode auto-re-template crash documented in ADR-005 `:1158`.

## Measured results

Two runs each, no warm-up filter (model load fresh each run).

| Tool | Path | Run 1 | Run 2 | Mean |
|---|---|---|---|---|
| hf2q | **per-token prefill (default)** | 94.68 tok/s (2455 tok / 25928 ms) | 94.31 tok/s (2455 tok / 26032 ms) | **94.50 tok/s** |
| hf2q | **batched prefill (`HF2Q_BATCHED_PREFILL=1`)** | **ERROR** | **ERROR** | — |
| llama.cpp | batched prompt eval (default) | 3216.80 tok/s (2443 tok / 759.45 ms) | 3244.83 tok/s (2443 tok / 752.89 ms) | **3230.82 tok/s** |

**Observed gap: ~34×** (llama.cpp prefill / hf2q per-token prefill).

## The batched-prefill error

`HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1` at 2455 tokens:

```
Batched prefill: KV=F32, seq_len=2455
ERROR hf2q: batched prefill L0: seq_len=2455 exceeds dense cap=1024 (sliding layer).
  Ring-wrap in the seq kernel is a follow-up;
  use HF2Q_BATCHED_PREFILL=0 (default) for now.
```

Root cause: hf2q's batched prefill path (ADR-009 Phase 3A) does not handle `seq_len > sliding_window=1024`. The ring-wrap equivalent of the mlx-native ring-buffer KV cache (that makes decode work at arbitrary context lengths on HEAD `388ad3d`) is not yet implemented in the prefill kernel.

So on prompts ≥ 1025 tokens, the per-token prefill path is the **only functioning prefill in hf2q today**. That's what the 94.50 tok/s number reflects.

## Why per-token prefill is ~decode-speed

Per-token prefill dispatches each of 2455 prompt tokens through the full forward pass sequentially, just like decode. At ~10.6 ms/token that matches hf2q's current decode envelope (101.7 tok/s median → 9.8 ms/token) almost exactly — prefill is serialised where llama.cpp's is parallel.

llama.cpp's 3230 tok/s prefill is a highly parallel matmul over all 2443 tokens in a single graph execution: batched Q/K/V projection, batched RoPE, batched attention (with causal masking via flash-attn), batched MLP, batched MoE dispatch. hf2q's batched prefill path (when it works, at ≤1024 tokens) is architected the same way — the gap is the missing sliding-window ring-wrap in the prefill seq kernel.

## Implication for Phase 1b Gate A

The Closeout Amendment defined Gate A as "prefill tok/s parity vs llama.cpp on a ≥2048-token prompt." Today that gate is:

- **Unreachable as stated.** At 2455 tokens, hf2q cannot use batched prefill; per-token is 34× slower than llama.cpp and there's no tuning path that closes a 34× gap.
- **Gated on a prerequisite.** Ring-wrap in the batched prefill seq kernel must land before Gate A can be measured honestly.

Three reconciliation options back to the amendment:

**A1. Keep Gate A as stated; add ring-wrap as an explicit Phase 1b prerequisite item.** Phase 1b closure now requires the ring-wrap fix before release-check.sh can evaluate Gate A. This aligns with "no shortcuts, no stubs" mantra — we don't declare Phase 1b closed while hf2q's prefill is 34× short of peer by architecture.

**A2. Narrow Gate A to ≤1024-token prefill (batched path).** Measure prefill parity where the batched path works; treat prompts >1024 as Run-scope until ring-wrap lands. This lets release-check.sh enforce something meaningful today but concedes that long-prompt prefill is Run, not Walk.

**A3. Two-tier Gate A.** Gate A.1 = batched prefill parity at ≤1024 tokens (closable today); Gate A.2 = ≥2048-token prefill parity (blocked on ring-wrap). Phase 1b closes when A.1 passes; A.2 tracked as a follow-up with a named owner. This formalises the split without deferring the long-prompt work indefinitely.

My recommendation is **A1**: the Closeout Amendment's intent was "no tracked-downstream-not-blocking" evasions. If ring-wrap is the blocker for real Gate A parity, ring-wrap is a Phase 1b item — same way the sourdough nondeterminism was reclassified from Run to Phase 1b blocker for gate F.

## Secondary finding: prompt-token count drift between tools

hf2q tokenises `prefill_2048.txt` to 2455 tokens; llama.cpp tokenises the same bytes to 2443 tokens. Delta 12 tokens. Consistent with the ADR-005 `:1158` lesson on BOS/special-token handling and non-identical Unicode paths. Not a defect — a measurement note for future Gate A harness work: if Gate A asserts "within variance of peer," the variance floor must account for the N ≠ M prompt-token-count mismatch (per-token ms/token is the denominator-stable metric, not total tok/s).

## Provisional floor, pending amendment decision

If the amendment is updated to option A2 (≤1024-token prefill parity), the floor is measurable today and release-check.sh can enforce it.

If A1 or A3, Gate A stays `[ ]` until ring-wrap lands.

## Artifacts

- `tests/evals/prompts/prefill_2048.txt` — 2455-token / 11,592-byte fixture, committed this session.
- This doc.

## Next actions (pending user decision)

1. Pick A1 / A2 / A3 above.
2. If A1 or A3: open a task for ring-wrap in the batched prefill seq kernel. Expected to slot into the existing mlx-native ring-buffer KV cache pattern (`src/serve/forward_mlx.rs`).
3. Update ADR-005 Closeout Amendment Gate A row to reflect the picked option.

## Addendum — 2026-04-16 PM (post-A1 lock, Task #7 execution)

User picked A1: ring-wrap lands as an explicit Phase 1b prerequisite. Implementation started. Two findings:

### Finding 1 — Ring-wrap KV writes landed (correct)

`mlx-native` commit `3b09de7` (pushed to github origin main, NOT yet published to crates.io) adds modular-slot writes to `kv_cache_copy_seq_f32` / `kv_cache_copy_seq_f32_to_f16` plus a new `src_tok_offset` parameter so the host can deduplicate the overwritten prefix. Two new unit tests pass at ε=1e-4:

- `test_kv_cache_copy_seq_f32_sliding_ring_wrap`: 8 tokens, capacity=4 → surviving tokens 4/5/6/7 land at slots 0/1/2/3 in modular order.
- `test_kv_cache_copy_seq_f32_no_wrap`: 3 tokens, capacity=8 → linear writes, untouched slots retain sentinel.

Initial naive implementation (mod without `src_tok_offset`) was **incorrect** — when n_tokens > capacity, multiple threads wrote to the same slot and Metal did not serialise them, producing nondeterministic survivors. The `src_tok_offset` param fixes this by dispatching exactly one write per surviving slot.

### Finding 2 — Blocker: `sdpa_sliding` broken at prefill shapes

The kv-write side is ready. The attention side is not: `mlx-native`'s `sdpa_sliding` kernel — the correct attention path for sliding layers at `seq_len > sliding_window` — emits `pad` (token 0) at prefill shapes.

Reproducer (hf2q, HEAD this session, `.cargo/config.toml` patching mlx-native to the local 0.2.0+`3b09de7`):

| Path | seq_len | First decode token | Verdict |
|---|---|---|---|
| Per-token prefill (baseline) | 576 | `2021` ("To") | reference |
| Batched prefill, plain `sdpa` all layers | 576 | `2021` ("To") | matches baseline |
| Batched prefill, `sdpa_sliding` on sliding layers | 576 | `0` ("pad") | **broken** |

At seq_len=576 < sliding_window=1024, `sdpa_sliding` with `window_size=1024` should behave identically to plain `sdpa` (every position is within-window). It doesn't. Plain sdpa emits the correct token; sdpa_sliding emits pad.

The kernel is orphaned in mlx-native — nothing else in hf2q uses it, so this bug has never been exercised. It's a Gate A hard blocker: without a working sliding-attention kernel, batched prefill at `seq_len > sliding_window` cannot produce correct output.

### Durable state after this session

- **mlx-native `3b09de7`** (pushed, not published): ring-wrap KV writes + two passing unit tests. Durable infra. When `sdpa_sliding` is fixed the ring-wrap side is already in place.
- **hf2q**: no commits. Local working tree has `forward_prefill_batched.rs` + `Cargo.toml` changes that pass the new 11-arg signature and narrow the guard to sliding-only (global layers can still exceed allocation safely). These don't commit cleanly until mlx-native publishes 0.2.1+ including the ring-wrap kernel.
- **New finding**: at `seq_len ≤ sliding_window` (≤1024), batched prefill with plain sdpa runs **2.85× faster** than per-token (283 vs 99 tok/s at 576 tokens) and is **byte-identical** in output. Useful for medium-prompt workloads independent of Gate A closure.

### Updated Gate A dependency chain

Gate A closure requires, in order:

1. **Task #8 (new)** — debug + fix `sdpa_sliding` in mlx-native at prefill shapes. Read kernel, compare math vs mlx-lm/llama.cpp references, write unit tests at prefill shapes, fix, publish.
2. **Task #7 (in-progress, blocked by #8)** — once sdpa_sliding is fixed, re-enable sliding-layer routing in `forward_prefill_batched.rs`, drop the sliding-layer bail, rebench.
3. Post-ring-wrap prefill tok/s measurement (this spike doc gets a Finding 3).
4. Gate A concrete floor set in `scripts/release-check.sh`.

### Open question for the user

Two paths forward:

- **Path X — Publish `mlx-native` 0.2.1 now** (with the ring-wrap + src_tok_offset kernel change). That lets hf2q commit its source change + Cargo.toml bump immediately and lands the batched-prefill 2.85× win for ≤1024-token prompts as a shipping feature, decoupled from the Gate A closure schedule. Requires `cargo publish` to crates.io.
- **Path Y — Hold the publish, debug sdpa_sliding first**. Everything sits in working-tree / github-only until Gate A can close in one landing. Conservative — no half-landed feature — but the 2.85× ≤1024-token batched-prefill win stays unshipped in the meantime.
