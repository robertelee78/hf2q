# ADR-030 iter-73 — Perf trajectory (iter-68 → iter-72) + Option A scoping

**Date**: 2026-05-14
**Status**: planning artifact; consolidates the perf optimization arc from
initial measurement (iter-68) to the current architecture frontier
(iter-72), then scopes the next big lever (Option A — cross-length
SDPA in batched prefill).

---

## 1. Perf trajectory iter-68 → iter-72

All numbers from `scripts/adr030/bench_spec_decode.sh` on
gemma-4-26b-a4b-it-ara-abliterated Q5_K_M, 24-token chat-templated
prompt, --temperature 0 --ignore-eos, M5 Max, 20s cool-downs, median
of 2 trials.  σ within each (arm, N) pair: <1%.

### Spec-decode throughput

| Iter | Description | N=8 t/s | N=16 t/s | N=32 t/s | Mission gap |
|---|---|---|---|---|---|
| iter-68 | Initial measurement | 4.65 | 4.40 | 4.05 | 22.2× slower |
| iter-69 | Eliminate redundant prior_ctx prefill | 7.0 | 6.9 | 6.0 | 14.9× |
| iter-70 | Profile + slice argmax to verify-window | 9.00 | 8.75 | 8.50 | 11.4× |
| iter-71 | Incremental drafter cache + batched argmax | 9.15 | 9.05 | 8.80 | 11.1× |
| iter-72 | Truly batched per_position_argmax stages | **9.55** | **9.45** | **9.20** | **10.5×** |
| baseline | hf2q forward_decode (per-token, no spec) | 103.8 | 99.5 | 97.0 | 1× |

**Cumulative**: at N=16, **4.40 → 9.45 t/s = +115% (2.15× spec-decode improvement)**.

### Per-stage profile at N=16 (iter-72 state)

```
embed=0.03         extract=0.10        drafter_fwd=12.91
drafter_argmax=6.22  verify_prefill=79.07  target_argmax=7.10  trim=0.10
TOTAL=105.52 ms/round
```

verify_prefill (75%) is the bottleneck.  Other stages have been
optimized to near-architectural limits given the Option C re-prefill
design.

### Coherence

- Untemplated 12-token prompt at N=16: **GREEN byte-identity** vs
  forward_decode baseline (iter-65, re-verified every iter through 72).
- Chat-templated 24-token prompt: axis-3 orchestrator self-consistency
  GREEN; axis-1 full byte-identity RED due to inherited
  `forward_prefill_batched` vs `forward_decode` bug (iter-67 finding;
  out of ADR-030 scope per task #12).

---

## 2. What's left vs mission gate

**Mission gate** (per ADR-030 §1.5): ≥1.07× hf2q baseline = ~107 t/s
on this fixture.  Current: 9.45 t/s = **10.5× slower**.

Required gain: ~11× speedup over iter-72.  Cannot be achieved via the
current Option C re-prefill architecture — each round pays
`O(prompt_len + accepted_so_far + K) × per-token-prefill` for the
verify forward.  At N=16 mid-generation: ~30 tokens × ~2.6 ms/tok =
~78 ms/round = exactly the verify_prefill time we measure.

**Mathematical lower bound for the mission gate**:

Per-round cost ≤ baseline_per_tok × n_committed
  = 10.1 ms × n_committed
Throughput = n_committed / round_cost

For 1.07× baseline = 107 t/s:
- With 100% accept (n_committed = 8): round_cost ≤ 75 ms → already close
- With  50% accept (n_committed = 4): round_cost ≤ 37 ms
- With   0% accept (n_committed = 1): round_cost ≤  9 ms → impossible

The mission gate requires BOTH:
1. **Lower round_cost** via Option A (cross-length SDPA) — projected ~30-50 ms/round
2. **Higher acceptance rate** — requires fixing the iter-67 axis-2
   bug (out of ADR-030 scope; tracked as task #12 for hf2q-internal
   batched_prefill coherence)

---

## 3. Option A — Cross-length SDPA in batched prefill

### 3.1 The gap

The current verify_prefill calls `forward_prefill_batched(verify_prefix,
…, 0, gpu)` where `verify_prefix = [output + drafts]`.  This is a
FULL TARGET FORWARD over ~32 tokens — re-projects, re-attends, re-norms
ALL the prior committed tokens every round just to update the K+1
verify positions.

Peer dflash (MLX) achieves O(K+1) per-round by feeding ONLY the K+1
new tokens with cross-length attention against the persistent
target_cache (`/opt/dflash/dflash/model_mlx.py:513`):

```python
logits = model(verify_input, target_cache)  # K+1 tokens, cache holds prior
hidden = mx.concatenate(model._hidden_states, axis=-1)
```

The MLX framework's attention layer handles cross-length attention
natively — Q has K+1 positions, K/V span (prior_cache + new K+1).

### 3.2 hf2q's existing building blocks

**Kernel surface** — already present in mlx-native:
- `flash_attn_prefill_bf16_d256_with_blk` (sliding layers, head_dim=256)
- `flash_attn_prefill_bf16_d512_with_blk` (full-attn layers, head_dim=512)
- BOTH support `qL ≠ kL` natively via the `qL_off` and `seq_len_k` params
  (kernel source: `/opt/mlx-native/src/shaders/flash_attn_prefill.metal`
  comments at lines 1330-1346 — the kernel handles the
  "resume scenarios where the chunk Q lands far into a populated slot"
  case explicitly).

**Cross-length params already in `AttnParamsGpu`** (line 328):
```rust
pub seq_len_q: u32,  // qL
pub seq_len_k: u32,  // kL
// + qL_off (offset in query sequence start)
```

**K/V layout** — `hybrid_kv.k` is shape `[H_kv, capacity, head_dim]`
F16; with `B=1` implicit, this matches the kernel's expected `[B, H_kv,
kL, D]` layout directly (kernel reads only kL positions).

**Existing usage** — the DFlash drafter ITSELF uses cross-length SDPA
via `dispatch_dflash_sdpa_cross_length`
(`src/inference/spec_decode/dflash/forward.rs`), which calls
`mlx_native::ops::sdpa::sdpa` with qL ≠ kL.  Proof that the building
block works.

### 3.3 The gap

Current `forward_prefill_batched` SDPA dispatch
(`forward_prefill_batched.rs:1310-1327` for sliding /
:1329+ for full):

```rust
mlx_native::ops::flash_attn_prefill::dispatch_flash_attn_prefill_bf16_d256_with_blk(
    s.encoder_mut(), dev, reg,
    &pf_q_perm, &pf_k_perm, &pf_v_perm,  // ← all from CURRENT chunk
    Some(&sliding_mask),
    Some(&blk_sliding),
    &mut pf_sdpa_out_perm,
    &FlashAttnPrefillParams {
        seq_len_q: seq_len as u32,
        seq_len_k: seq_len as u32,  // ← FORCED to seq_len
        ...
    },
)
```

The kernel could accept `seq_len_k = start_pos + seq_len` and read K/V
from a buffer holding the full kL positions.  Currently we always pass
`seq_len`-sized K/V from the current chunk → kernel attends only over
the chunk.

### 3.4 What needs to change (Option A implementation outline)

For each layer in the verify path:
1. Detect "spec-decode verify mode": `start_pos > 0` AND
   `self.dflash_capture.is_some()`.
2. K, V buffers come from `self.hybrid_kv[layer_idx]` (already populated
   correctly at positions `[start_pos..start_pos+seq_len)` thanks to
   iter-64's `dst_seq_pos_start = start_pos + src_tok_offset` fix).
3. Build a cross-length mask `[seq_len, start_pos+seq_len]` BF16 with
   causal semantics + sliding-window for sliding layers.  Set
   `qL_off = start_pos` so the causal pivot is correctly positioned.
4. Call the same kernel with `seq_len_k = start_pos + seq_len`,
   `qL = seq_len`, `qL_off = start_pos`.
5. Q (`pf_q_perm`) is BF16; K/V from hybrid_kv are F16.  Either:
   - Cast Q to F16 once per layer (small cost), use
     `flash_attn_prefill_f16_d{256,512}`, OR
   - Require `HF2Q_FULL_F16_KV=1` to also have K F16 (V already F16 in
     that mode); then cast Q to F16.

### 3.5 Risks

- **V dequant**: by default `hybrid_kv.v_packed` is U8 quantized
  TQ-HB.  The flash_attn_prefill kernels expect dense bf16/f16 V.
  REQUIRES `HF2Q_FULL_F16_KV=1` for the simple path, OR a V dequant
  dispatch.  Recommend the former for the first cut (well-documented
  trade-off: doubles V memory).
- **Mask construction**: the current mask is `[seq_len, seq_len]`; need
  to build a `[seq_len, start_pos+seq_len]` mask per round.  Small
  GPU dispatch; well-understood pattern.
- **Capture hook timing**: iter-65's `s.finish()` sync on
  `dflash_capture.is_some()` already handles capture-timing
  correctness.  Should carry over to Option A unchanged.
- **Coherence regression risk**: substantial.  The e2e gate must run
  after every layer plumbing change.

### 3.6 Estimated impact

Per-layer SDPA cost scales linearly with K (= kL × qL within tile).
At qL = K+1 = 8 (vs current qL = ~32) the per-layer SDPA work drops
~4×.  Plus the projections (Q/K/V/O) and MLP scale with qL = 8 vs 32
linearly.

Optimistic projection: verify_prefill 79 ms → 25-35 ms (~50-65%
reduction).  At N=16 round breakdown:
- verify_prefill: 79 → 30 ms (−49)
- All other stages unchanged: 26 ms
- TOTAL: 105 → 56 ms/round
- Throughput: 9.5 → 17.8 t/s (1.87×) at 0% accept
- At 50% accept: ~4 committed tokens / 56ms = 71 t/s = 0.71× baseline

Mission gate (≥1.07× baseline = 107 t/s) still requires ≥75% acceptance
on top of Option A.  Both axes needed.

### 3.7 Implementation scope estimate

- ~500-800 LOC for the verify-mode SDPA branch + mask builder + cast Q dispatch
- ~200 LOC for the new e2e test verifying parity with existing path
- ~3-5 days engineering wall-clock (single dev)
- Multi-iter scope in /loop terms; iter-74+ work

---

## 4. Recommended iter-74+ sequence

1. **iter-74**: implement cross-length mask builder.  Unit-test
   shape/values against a CPU reference for both causal-only and
   sliding-window cases.  Small contained scope.
2. **iter-75**: implement BF16 → F16 Q cast dispatch.  Wire it into a
   new "verify SDPA branch" in forward_prefill_batched that only fires
   when `start_pos > 0` AND `dflash_capture.is_some()`.  Use
   `flash_attn_prefill_f16_d{256,512}` with the cross-length params.
   Require `HF2Q_FULL_F16_KV=1`.
3. **iter-76**: e2e GPU test that runs spec-decode with the new SDPA
   path.  Verify untemplated coherence GREEN; measure perf.
4. **iter-77+**: if perf gate met, integrate to production wire-up.
   If not, profile + sub-iter optimize.

Throughout: keep the existing Option C re-prefill path as the FALLBACK
when `HF2Q_FULL_F16_KV != 1` or `HF2Q_DFLASH_XLEN_SDPA != 1` (the new
env flag gating the cross-length SDPA path).

---

## 5. Mission state at iter-73

| Metric | Value |
|---|---|
| Coherence (untemplated N=16) | ✅ byte-identical |
| Production wire-up | ✅ `HF2Q_SPEC_DFLASH=1` |
| Perf at N=16 | 9.45 t/s = 0.096× baseline = **10.5× slower** |
| Mission gate | ≥1.07× baseline |
| Remaining | Option A + axis-2 acceptance fix |

5 perf iters (iter-68 → 72) achieved **2.15× speedup**.  Architectural
ceiling reached under Option C; iter-74+ Option A work is required for
further perf progress toward the mission gate.

---

## 6. Artifacts

- `docs/research/adr030_iter68_bench/results.tsv` — iter-68 baseline
- `docs/research/adr030_iter68_bench/results_iter{69,70,71,72}.tsv` — perf progression
- `scripts/adr030/bench_spec_decode.sh` — reproducible bench harness
- ADR-030 status log §iter-{68..72} — per-iter detailed entries
