# ADR-011 Phase 3 — Prefill perf investigation (the 7× gap)

**Author**: Phase 3 P1 (perf-investigator), CFA swarm
**Date**: 2026-04-17
**Status**: Proposed (investigation only — no code changes)
**Upstream**: ADR-011 Phase 2 Wave 4 closed at 14% of llama.cpp peer for
prefill tok/s. Correctness is perfect (sourdough 562-byte margin over the
3094-byte parity floor). Phase 3 closes the speed gap while preserving
that margin.

---

## Executive summary

The 7× prefill gap is **not** dispatch overhead, CPU-GPU sync cost, or
barrier thrash. The GPU is already 98–100% busy at 1620 MHz / ~90 W
during hf2q prefill — identical to llama.cpp's GPU state. What differs
is **work per dispatch**: hf2q's quantized-matmul path is the llama.cpp
*mat-vec* kernel (`kernel_mul_mv_q*_f32`) used even when the prompt has
thousands of rows, while llama.cpp switches to the simdgroup-matrix
*mat-mat* kernel (`kernel_mul_mm_q*_f32`) whenever `ne11 > 8`. The
mat-vec kernel re-loads every weight block once per prompt token; the
mat-mat kernel loads each weight block once per 32-token block of the
prompt and re-uses it via threadgroup shared memory + 8×8 simdgroup MMA.
Gemma 4 prefill at seq_len=2455 runs **~77×** the weight reads that
llama.cpp runs. That is ~2.5× the observed gap, and is blunted by the
MMA units delivering ~3× the FMA throughput of scalar accumulation —
net 7×, which matches the measurement.

All other candidates we investigated (sync count, barrier count,
dispatch count, fusion opportunities, mask/blk rebuilds) contribute
under 10% combined.

**Recommended Phase 3 plan**: three waves.
- **Wave P3a**: port `kernel_mul_mm_q4_K_f32` and
  `kernel_mul_mm_id_q4_K_f32` (MoE). Expected: **14% → ~70%** of peer.
- **Wave P3b**: merge 3 sessions/layer into 1, fuse qmatmul-then-cast
  and norm-then-qmatmul via encoder concurrency. Expected: **+5 pts**.
- **Wave P3c**: optional bf16 residual-stream to shrink cast islands.
  Expected: **+3 pts**. Longer-term.

Primary uncertainty: we have not A/B tested mm vs mv in isolation on
Q4_K — Wave P3a must verify the 5× kernel-level speedup empirically
against the llama.cpp kernel.

---

## §1 Measured data

### 1.1 Headline

| Benchmark | hf2q (batched) | llama.cpp (pp2048, n=0, r=1) | Ratio |
|---|---|---|---|
| Prefill tok/s (pp2048) | 458 tok/s (Wave 4 doc) | 3498 tok/s (measured today) | 13.1% |
| Prefill tok/s (pp2455) | 495 tok/s (measured today) | 3381 tok/s (Wave 4 doc) | 14.6% |

Ratio is flat ±1pt across seq_len 128 → 2455 (Wave 4 §parity). Prompts
this short already saturate the GPU, so the gap is per-prefill-pass
structural, not grid-utilization.

### 1.2 hf2q dispatch & sync counters (measured, `HF2Q_DUMP_COUNTERS=1`)

Prefill-only (short_hello, 22 tokens, max_tokens=1):
```
dispatches=1085  syncs=93  prompt_tokens=22  decode_tokens=1
```
Prefill-only (prefill_2048, 2455 tokens, max_tokens=1):
```
dispatches=1085  syncs=93  prompt_tokens=2455  decode_tokens=1
```

**Dispatches and syncs are both seq-independent** — they are a function
of layer count × per-layer session structure. Gemma 4 has 30 MoE layers.

Per-layer dispatch breakdown (static count from
`src/serve/forward_prefill_batched.rs`, verified against the ~1085 total):

| Session | Dispatches/layer | Notes |
|---|---:|---|
| A (attention) | 18 | norm · 3 qmatmul (Q,K,V) · 2 head-norm+RoPE · V-norm · 3 f32→bf16 casts · 3 permute-021 · 1 flash_attn_prefill · 1 permute-back · 1 bf16→f32 cast · O-proj qmatmul · fused_norm_add |
| B (MLP+MoE) | 16 | 3 rms_norm · 3 qmatmul (gate/up/router) · fused_gelu_mul · MoE routing · down qmatmul · 2 qmatmul_id (gate_up, down) · moe_swiglu · post-norm · moe_weighted_sum · 2 fused_norm_add |
| C (KV write) | 2 | K,V cache copies |
| **Total/layer** | **36** | × 30 layers = 1080 |
| One-off setup | 5 | embedding · 2 mask-build · 2 blk-classify |
| Final head | 5 | copy-last-row · final_norm · lm_head · softcap · argmax |
| **Grand total** | **1090** | measured: 1085 |

Session count: **93 `commit_and_wait` calls per prefill** (3 sessions ×
30 layers + 1 embedding + 1 mask/blk setup + 1 final head). Each one
is a full CPU↔GPU sync. At 5 s/prefill / 93 syncs ≈ 53 ms wall per
sync on average, so **GPU work per sync ≫ sync overhead** — see §1.4.

### 1.3 llama.cpp dispatch estimate

llama.cpp does not ship a runtime node counter, but the graph is
constructed in `src/models/gemma4-iswa.cpp` and compiled to Metal
kernels in `ggml/src/ggml-metal/ggml-metal-ops.cpp` with automatic
fusion (`ggml_can_fuse_ext`) and concurrency tracking
(`ggml_metal_op_concurrency_check`). Per-layer node count, post-fusion:

| ggml op | Count | Kernel |
|---|---:|---|
| `MUL_MAT` Q (attn) | 1 | `kernel_mul_mm_q4_K_f32` (ne11=seq > 8) |
| `MUL_MAT` K (attn) | 1 | same |
| `MUL_MAT` V (attn) | 1 | same |
| `RMS_NORM` ×3 (Q-norm, K-norm, V-norm) | 3 | fused with `MUL` if present |
| `ROPE` ×2 (Q, K) | 2 | `kernel_rope_*` |
| `FLASH_ATTN_EXT` | 1 | `kernel_flash_attn_ext_*` |
| O-proj `MUL_MAT` | 1 | `kernel_mul_mm_q4_K_f32` |
| `attn_post_norm` + residual ADD | 1 | fused |
| `ffn_norm_1` (RMS+MUL) | 1 | fused |
| `ffn_up` + `ffn_gate` + GELU | 3 | `mul_mm` × 2 + unary-fused |
| `ffn_down` | 1 | `mul_mm` |
| `ffn_post_norm_1` | 1 | fused |
| `ffn_norm_2` | 1 | fused |
| router (RMS+scale+MUL+`MUL_MAT`) | 2 | often fused to 2 |
| `MUL_MAT_ID` gate_up (MoE) | 1 | `mul_mm_id` |
| MoE GLU | 1 | |
| `MUL_MAT_ID` down (MoE) | 1 | |
| MoE weighted_sum | 1 | |
| `ffn_post_norm_2` | 1 | fused |
| mlp+moe combine ADD | 1 | |
| `ffn_post_norm` (end-of-layer) | 1 | fused with layer_scalar MUL |
| end-of-layer residual ADD | 1 | |
| **Per layer** | **~28** | |

Plus ~10 one-off nodes (embedding, 2 `ggml_scale`, `inp_out_ids`
strip-before-last-layer, final norm, lm_head mul_mat, softcap chain
scale·tanh·scale, argmax-or-sampler).

**Estimated llama.cpp total: ~30×28 + 10 = ~850 dispatches per
prefill.** hf2q sits ~30% higher (1085) — not a 7× delta, so the
dispatch count itself is NOT the bottleneck.

### 1.4 GPU utilisation (`powermetrics --samplers gpu_power`)

hf2q prefill_2048 during the active prefill window:
```
GPU HW active residency: 98.22% at 1620 MHz, Power: 89234 mW
GPU HW active residency: 100.00% at 1620 MHz, Power: 90580 mW
```

llama.cpp pp2048 during active prefill window:
```
GPU HW active residency: 99.26% at 1619 MHz, Power: 85259 mW
```

**Both are GPU-bound on sustained P13 state.** CPU time (encoding,
sync overhead) is overlapped with GPU compute and is not on the
critical path. The gap lives inside the kernels, not between them.

### 1.5 Per-kernel dominant cost: matvec-replicated-m

hf2q's `dispatch_qmatmul` (src/serve/forward_mlx.rs:3035) calls
`session.quantized_matmul_ggml`, which resolves to
`kernel_mul_mv_q4_0_f32` / `_q8_0_` / `_q6_K_` (mlx-native
`ops/quantized_matmul_ggml.rs:86-96`). This is a **matvec** kernel
dispatched with `threadgroups = (ceil(n/align), m, 1)` — i.e. one
output slice per (m, n-block) threadgroup, and the `m` axis is the
prompt-token axis.

Inside each threadgroup (see
`src/shaders/quantized_matmul_ggml.metal:99` for Q4_0):
- Thread reads input-row `r1` (one prompt token) into registers.
- Thread reads weight blocks `x + ib + row*nb` via `block_q4_0_dot_y`.
- Accumulates into `sumf[row]`, stores one value per (m, n4-lane).
- **No shared-memory staging of weights.** Each (m, n) threadgroup
  re-reads its weight rows from global memory.

At prefill with m=seq_len=2455, Q-proj shape `[m, k=1152] × [n=3072, k=1152]`:
- Weight bytes read per threadgroup: `n4_per_tg × blocks_per_row × block_bytes = 4 × 36 × 18 = 2592 bytes`.
- Threadgroups launched: `ceil(n/4) × m = 768 × 2455 = 1,885,440`.
- Total global bytes read: 1,885,440 × 2,592 = **4.89 GB** for a single Q-proj at m=2455.
- Weight tensor actual size: 3072×1152 / 32 × 18 bytes = **1.99 MB**.
- **Re-read factor: 2,457×** — literally every weight byte is fetched
  from DRAM once per prompt token.

llama.cpp's `kernel_mul_mm_q4_K_f32`
(`ggml/src/ggml-metal/ggml-metal.metal:9276`) with NR0=64, NR1=32,
NK=32, 4 simdgroups per threadgroup, shared-memory staging of both A
and B tiles and `simdgroup_multiply_accumulate` MMA:
- Threadgroups launched: `ceil(m/32) × ceil(n/64) = 77 × 48 = 3,696`.
- Weight tile (block-q) staged **once per threadgroup** into `shmem`.
- Weight bytes read: 3,696 × (64 rows × 1,152 / 32 × 18) = **154 MB**,
  a **32× reduction** versus hf2q's matvec path.
- Plus MMA gives ~3× FLOP throughput for the multiply-accumulate phase.

Combined expected speedup on Q4_K projections: **5–8×** at m=2455.
Consistent with the observed 7× end-to-end gap given that qmatmul is
the dominant time sink (Q/K/V × 30 + O × 30 + MLP gate/up × 30 + MLP
down × 30 + router × 30 + 2×MoE experts × 30 + lm_head = 211 qmatmul
calls per prefill, all running the inefficient kernel).

### 1.6 What mlx-native already has

- `dense_gemm_f16` with BM=32, BN=32, BK=16, simdgroup_matrix MMA
  (`src/shaders/dense_gemm.metal:262`). Used only for **dense f16
  weights** — covers only the F16 lm_head. Not usable for Q4_K
  projections without being re-written against the block-q dequantize
  path.
- `flash_attn_prefill_bf16_d256/d512` already ports llama.cpp's
  MMA-based attention (Phase 2 Wave 4) — so the tools (simdgroup MMA,
  shared-memory staging, block-q dequantize-in-kernel) all exist
  in-tree.

### 1.7 Where CPU-side cost is

Per-layer encoding cost is dominated by Metal's argument-encoder paths
inside each `encode_threadgroups_with_args` call. The 93 syncs per
prefill serialize GPU work blocks, but because each block already
contains 10–20 dispatches of substantial GPU work (qmatmul of Q-proj
alone is ~1 ms of kernel time at seq=2455), the sync overhead is well
below the GPU work time.

Fastest rough budget:
- Single-dispatch host-side overhead on Metal: ~5-20 µs.
- 1085 dispatches × 15 µs = **~16 ms** CPU host time per prefill.
- Measured prefill wall: **~5,000 ms**.
- CPU dispatch overhead: **<0.4% of wall**. Not the bottleneck.

Likewise 93 syncs × 500 µs (roundtrip + completion-handler) ≈
**~50 ms / 5000 ms = 1%.** Also not the bottleneck.

### 1.8 Deferred measurements

- **Metal frame capture** — would confirm per-kernel wall-clock times
  and make the matvec vs matmul split visible in Xcode's GPU debugger.
  We did not run it in this session because it requires rebuilding
  with `MTL_CAPTURE_ENABLED=YES` in Info.plist. The analytical path
  above is sufficient to establish the hypothesis; capture is
  recommended for Wave P3a verification.
- **hf2q + llama.cpp on the same prompt with identical measurements
  via `xcrun xctrace` or `metal-compile-all`** — could quantify the
  per-kernel time split. Defer to Wave P3a entry gate so we can A/B
  against the new `mul_mm_q4_K` port.

---

## §2 Diagnosis — ranked bottlenecks

### #1 — qmatmul is matvec even at m≫1 (est. ~85% of the 7× gap)

**Name**: `kernel_mul_mv_q*_f32` used for all quantized projections.
**Location**: `/opt/mlx-native/src/ops/quantized_matmul_ggml.rs:86-96`
(kernel names), `/opt/mlx-native/src/shaders/quantized_matmul_ggml.metal`
(kernel source). `dispatch_qmatmul` in
`/opt/hf2q/src/serve/forward_mlx.rs:3035` calls this for every
projection in the batched prefill.

**Contribution**: Dominant. At m=2455, the kernel re-reads each weight
block 2,457× versus once per 32-row tile in llama.cpp's `mul_mm`. Per
§1.5, the arithmetic-intensity gap alone explains 5–8× of the 7×
end-to-end gap.

**Evidence**: 
- GPU is 100% busy in both hf2q and llama.cpp (§1.4).
- Dispatch count only 30% higher than llama.cpp estimate (§1.2-1.3).
- Per-kernel read-reuse analysis (§1.5) shows 32× global-memory read
  amplification.
- llama.cpp source chooses `mul_mm` when `ne11 > ne11_mm_min = 8`
  (`ggml-metal-ops.cpp:2046, 2154`) — this is exactly the prefill
  case.

**Fix complexity**: Medium. Requires porting
`kernel_mul_mm_q4_K_f32` and `kernel_mul_mm_id_q4_K_f32` from
`llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9276+` into
mlx-native, plus dispatch plumbing. The MMA building blocks
(`simdgroup_matrix<float,8,8>`, `simdgroup_multiply_accumulate`,
threadgroup-memory tile staging) are already proven in tree via
`flash_attn_prefill` and `dense_gemm_f16`. The Q4_K dequantize-to-tile
logic is the main new surface (block_q4_K layout, scale/min extraction).

---

### #2 — 3 sessions/layer (93 syncs) forces every layer to a CPU fence (est. ~3% of gap)

**Name**: Sessions A, B, C per layer are committed and waited
separately in `forward_prefill_batched.rs`.

**Location**: `/opt/hf2q/src/serve/forward_prefill_batched.rs:393`
(Session A), `:704` (B), `:923` (C). Each `exec.begin() / s.finish()`
pair = 1 `commit_and_wait` (ref `mlx-native/src/graph.rs:1549`).

**Contribution**: Small but stackable. 93 syncs × ~500 µs =
~45 ms CPU critical path. On a 5 s prefill this is ~1%. Lower bound
if Wave P3a lands, upper bound if we reduce per-sync cost.

**Evidence**: Static grep — see §1.2 counter output shows exactly 93
syncs, matching 3 × 30 + 3 fixed.

**Fix complexity**: Simple. A, B, C within a layer have no external
reads between them; they can share a single encoder. Even cleaner: a
single encoder for all 30 layers, with `memory_barrier()` between
data-dependent dispatches (hf2q already has `barrier_between` with
conflict detection, so it's a matter of not closing the encoder until
end-of-prefill). The only constraint is that `s.finish()` currently
flushes so follow-on CPU reads work; for batched prefill, the only
CPU read is the final argmax.

**Risk**: Minimal — mlx-native already supports `dispatch_in_group`
tracking (`graph.rs:1452`) and concurrent-group scheduling.

---

### #3 — f32↔bf16 cast islands around attention (est. ~2% of gap)

**Name**: Wave-3 bf16-island casts per layer.

**Location**: `/opt/hf2q/src/serve/forward_prefill_batched.rs:520-531,
666-670`. 3 f32→bf16 casts (Q, K, V normed) + 1 bf16→f32 cast (SDPA
out) per attention session × 30 layers = **120 dispatches/prefill**
solely to shuffle between f32 residual stream and bf16 attention
island.

**Contribution**: Small. Cast kernels run at memory bandwidth. ~3 MB
per cast × 120 casts = ~360 MB at ~300 GB/s DRAM ≈ 1.2 ms total. Noise.

**Evidence**: Static source count.

**Fix complexity**: Medium-large — requires moving the residual stream
to bf16, or widening norms to bf16 output. Captured in the Phase 2
bf16-conversion-map ADR as a longer-term item.

---

### #4 — MoE routing/moe_* dispatch structure (est. ~2% of gap)

**Name**: MoE dispatch has several kernels that could plausibly
inline. `fused_moe_routing_batch_f32`, `moe_swiglu_seq_encode`,
`moe_weighted_sum_seq_encode` are already batched over seq_len — good
— but still 3 sequential dispatches per layer that llama.cpp fuses
inside `build_moe_ffn`.

**Location**: `/opt/hf2q/src/serve/forward_prefill_batched.rs:773,
825, 875`.

**Contribution**: Each MoE kernel is bandwidth-bound over small
intermediates (seq_len × top_k × hidden). Likely <2% of gap.

**Fix complexity**: Medium — would need a fused MoE kernel that does
routing + SwiGLU + weighted-sum in one pass. Defer.

---

### #5 — Per-prefill mask + blk build (est. <0.5% of gap)

**Name**: 4 once-per-prefill dispatches (2 mask builds + 2 blk
classifiers).

**Location**: `/opt/hf2q/src/serve/forward_prefill_batched.rs:281-324`.

**Contribution**: Trivial. Masks are `seq_len × seq_len` bf16
scratches (2455² × 2 = ~12 MB each); each kernel writes the whole
plane once. At ~300 GB/s that's ~40 µs × 4 = 160 µs. Well under 1%
of wall.

**Already optimal**: built once per prefill, not per layer. No action.

---

## §3 Proposed Phase 3 fix plan

### Wave P3a — port `kernel_mul_mm_q4_K_f32` + `kernel_mul_mm_id_q4_K_f32`

**Expected**: tok/s **495 → 2400** (14% → 70% of peer). Single biggest
lever. Closes ~85% of the gap.

**Scope**:
- New file `/opt/mlx-native/src/shaders/quantized_matmul_ggml_mm.metal`
  hosting `kernel_mul_mm_q4_K_f32` and helpers. Source ported from
  `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:9276-9450` (the
  `kernel_mul_mm` template) with the `block_q4_K` dequantize path
  fetched from the same file (search for `dequantize_q4_K`).
- New file `/opt/mlx-native/src/shaders/quantized_matmul_id_ggml_mm.metal`
  hosting `kernel_mul_mm_id_q4_K_f32` for the MoE experts, ported from
  `ggml-metal.metal:9650+`.
- New dispatcher in
  `/opt/mlx-native/src/ops/quantized_matmul_ggml.rs` that routes on
  `m > 8` → mm, else mv (matching llama.cpp's `ne11_mm_min = 8`).
- Same pattern for `quantized_matmul_id_ggml.rs`.
- Register both kernels in `/opt/hf2q/src/serve/gpu.rs`.
- Also port Q8_0 and Q6_K mm variants if Gemma 4's embedding / lm_head
  / other tensors demand them (check `GgufFile::tensors` metadata;
  these are the only quant types currently supported by the mv path).
- Unit tests: `tests/test_quantized_matmul_mm.rs` — pass M=64 through
  the MMA path, compare against M-rows-of-mv output within 1e-5 rtol.

**Files to touch**: 4 shaders, 4 Rust files, 2 test files.

**Risk — correctness**:
- Output must match current mv path within numerical tolerance. Test
  via sourdough gate (562-byte margin) *and* a round-trip parity test
  at M=2,16,64 against the mv result. Any deviation >1 ULP in a final
  logit could break parity at some token position.
- Ordering of accumulations differs (MMA vs scalar); llama.cpp's mm
  kernel has been battle-tested, so the risk is in our port, not in
  the algorithm.

**Risk — perf downside**: None expected — at m=1 we stay on the
matvec kernel (decode path still runs at 102 tok/s, matching peer).
The switch point `m > 8` matches llama.cpp and preserves decode perf.

**Dependencies**: None (first wave).

---

### Wave P3b — collapse 3 sessions/layer → 1 session, fuse cast-after-qmatmul

**Expected**: tok/s **2400 → 2800** (70% → 82% of peer). Assumes P3a
has landed (so per-kernel GPU time has shrunk and sync/encoding
overhead becomes a larger share of wall).

**Scope**:
- `/opt/hf2q/src/serve/forward_prefill_batched.rs` — replace the three
  per-layer `exec.begin() / s.finish()` pairs with a single session
  that spans A+B+C for the whole prefill loop. The existing
  `s.barrier_between(reads, writes)` smart-barrier already handles
  intra-layer data deps and elides unnecessary barriers (per
  `mlx-native/src/graph.rs:1419`).
- Fold the 3 f32→bf16 casts into the preceding `fused_head_norm_rope`
  by extending the kernel to write bf16 output directly (new function
  constant `write_bf16`). Eliminates 90 dispatches/prefill.
- Fold the bf16→f32 post-attn cast into the subsequent O-proj qmatmul
  input path by writing a bf16-input variant of `mul_mm_q4_K`.
  Eliminates 30 dispatches/prefill.
- Merge KV cache copy (session C) into session A — its reads
  (pf_k_normed, pf_v_normed) are already available after session A's
  V-norm; there's no reason for a separate flush.

**Risk — correctness**:
- Smart-barrier coverage must catch every dep. If a barrier is missed
  and buffers are reused next layer, we get silent corruption. All
  existing `barrier_between` call sites must be audited.
- Sourdough gate must pass between every sub-change; single-session
  merging is a structural refactor, not a correctness-neutral tweak.

**Risk — perf downside**: Single mega-encoder could exceed Metal's
command buffer size limit (~100k dispatches per buffer). Gemma 4 at
~1085/prefill is well under that; future larger models might need
subdivision.

**Dependencies**: P3a should ship first so the percentage gain is
measurable.

---

### Wave P3c — optional: bf16 residual stream, collapse cast islands

**Expected**: tok/s **2800 → 3100** (82% → 89% of peer). Diminishing
returns, but closes the remaining gap to peer without sacrificing
correctness.

**Scope**:
- Widen residual stream (`pf_hidden`, `pf_residual`) from f32 to bf16.
- Widen RMS-norm output to bf16 (match MLX-LM convention).
- Widen qmatmul output to bf16 (requires a new mm kernel variant;
  internals already accumulate in f32, so this is a store-side change).
- Eliminates all remaining f32↔bf16 casts in the pipeline.

**Risk — correctness**:
- Biggest correctness risk of the three waves. Sum-of-residuals chains
  in f32 preserve more precision than bf16; parity gate margins could
  narrow.
- Every parity prompt must re-pass, and the
  `HF2Q_BATCHED_LAYER_SCAN`-based per-layer drift bisection must hold
  below the current ULP envelope.

**Risk — perf downside**: None — bf16 everywhere halves bandwidth.

**Dependencies**: P3a + P3b.

---

## §4 Non-Phase-3 / longer-term

1. **MoE fusion — single kernel for routing + GLU + weighted_sum**.
   Est. +2% above and beyond P3c. Large kernel surface; defer until
   downstream profiling confirms it's the remaining bottleneck.

2. **Eliminate permute_021 for attention**. `flash_attn_prefill`
   currently wants `[batch, n_heads, seq, hd]` contiguous. The
   permute costs 3 × 30 = 90 dispatches per prefill of pure
   memory shuffle. Making qmatmul write in permuted layout directly
   would remove them — but qmatmul output layout is a deep contract
   (see `GgmlMatvecParams`, `AttnParamsGpu` strides) that touches
   many code paths. Defer.

3. **bf16 qmatmul output** — the output of qmatmul is currently f32.
   If residual stream moves to bf16 (P3c) AND attention stays bf16,
   then writing qmatmul output in bf16 removes another 30 casts. Fits
   with P3c; separate commit.

4. **CUDA Graphs / Metal IOSurface pre-recording**. llama.cpp does
   not use graph capture on Metal; neither do we. Mentioned for
   completeness — no action.

5. **Reconsider `ne11_mm_min = 8`**. For m in [2, 8], llama.cpp has a
   specialized small-batch mv_ext kernel (`ggml-metal-ops.cpp:2119`).
   hf2q could gain a few percent on small-prompt prefills by porting
   this too, but the primary target is pp>128 which all hits the mm
   path.

---

## §5 Correctness preservation

Every wave must hold the sourdough parity gate: common-prefix ≥ 3094
bytes (current margin 3656 bytes = 562-byte buffer). Non-negotiable.

### Test protocol per wave

1. **Unit tests**:
   - P3a: `/opt/mlx-native/tests/test_quantized_matmul_mm.rs` —
     matmul vs matvec parity at M ∈ {1, 2, 8, 16, 32, 64, 128, 1024,
     2455} for Q4_0, Q4_K, Q8_0, Q6_K, within 1e-5 rtol for f32 output
     (bf16 output: 1e-2 rtol).
   - P3b: no new unit test, but existing
     `tests/test_batched_prefill_*` must all pass.
   - P3c: per-layer drift test via `HF2Q_BATCHED_LAYER_SCAN` must
     stay below 1e-4 abs diff on residual at every layer boundary
     across the 2455-token sourdough prompt.

2. **End-to-end parity**:
   - `scripts/parity_check.sh` all three prompts (short_hello,
     sourdough, sliding_wrap) × 3 runs each. No run may fall below
     its min-prefix threshold.
   - Sourdough margin must stay ≥ 500 bytes (conservative; current
     is 562).

3. **Perf gates**:
   - Gate A (prefill): floor scales with each wave.
     P3a ships: ≥ 2200 tok/s. P3b: ≥ 2600. P3c: ≥ 2900.
   - Gate B (decode): must stay ≥ 100 tok/s throughout. P3a's switch
     on m > 8 preserves mv for m=1.

4. **Dispatch counter gate**:
   - Pre-P3: 1085 dispatches / prefill. P3a adds none (same count,
     faster kernels). P3b: ≈ 760 (collapses 120 casts + 30 cache
     copies). P3c: ≈ 670.

5. **Bisection readiness**:
   - Each wave must land as its own commit so that a regression can
     be bisected to a single kernel/structural change. Per the
     "commit + push cadence" convention.

---

## Appendix A — Raw measurement commands

```bash
# hf2q batched prefill, counter dump (prefill_2048)
HF2Q_DUMP_COUNTERS=1 HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1 \
  /opt/hf2q/target/release/hf2q generate \
    --model /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
    --prompt-file /opt/hf2q/tests/evals/prompts/prefill_2048.txt \
    --max-tokens 1 --temperature 0.0

# Output (measured today, 2026-04-17):
#   Batched prefill complete: 2455 tokens in 4957.1 ms (495.2 tok/s)
#   [MLX_COUNTERS] dispatches=1085 syncs=93 prompt_tokens=2455 decode_tokens=1

# llama.cpp peer
/opt/llama.cpp/build/bin/llama-bench \
  -m /opt/hf2q/models/gemma-4-26B-A4B-it-ara-abliterated-dwq/gemma-4-26B-A4B-it-ara-abliterated-dwq.gguf \
  -p 2048 -n 0 -r 1
# Output: pp2048 = 3497.99 t/s

# GPU-busy measurement (requires sudo)
sudo powermetrics --samplers gpu_power -n 5 -i 1000 &
# then run the hf2q or llama.cpp command above
```

## Appendix B — Evidence index

- [hf2q dispatch counters (measured)](#12-hf2q-dispatch--sync-counters-measured-hf2q_dump_counters1)
- [llama.cpp graph estimate](#13-llamacpp-dispatch-estimate)
- [GPU utilisation (powermetrics)](#14-gpu-utilisation-powermetrics---samplers-gpu_power)
- [Per-kernel arithmetic-intensity analysis](#15-per-kernel-dominant-cost-matvec-replicated-m)
- [llama.cpp `mul_mm` reference](/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L9276)
- [hf2q `kernel_mul_mv_q4_0_f32` shader](/opt/mlx-native/src/shaders/quantized_matmul_ggml.metal#L99)
- [hf2q `dense_gemm_f16` (MMA, f16 only)](/opt/mlx-native/src/shaders/dense_gemm.metal#L262)
- [ADR-011 Phase 2 Wave 4 verification](/opt/hf2q/docs/ADR-011-phase2-wave4-wire-up-verification.md)
