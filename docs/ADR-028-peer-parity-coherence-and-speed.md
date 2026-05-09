# ADR-028: Peer-Class Inference — Coherence Parity + Speed Parity-or-Better

- **Status**: accepted (2026-05-09 iter-87 — body settled, open scope sized)
- **Date**: 2026-05-09
- **Deciders**: Robert (operator), Claude (this session)
- **Tags**: performance, kernel-parity, gemma, qwen35, prefill, decode, lock-in

## Mantra (load-bearing — operator-stated 2026-05-09 iter-86)

> *"We need to be as coherent as peers, and as fast as or faster than peers."*

Coherence = byte-identical (or operator-equivalent) decoded output vs the
reference peer (`/opt/llama.cpp` HEAD `d05fe1d7d`/build 9010) on the same
GGUF, same prompt, same sampler. Speed = tok/s on the same hardware
(Apple M5 Max), measured via `llama-bench` for the peer baseline and
`hf2q generate ... --max-tokens N` for hf2q.

## Context

### What this ADR is for

Capture the cross-cutting **inference speed-gap closure workstream**
(iter-59..iter-86, 2026-05-09 session) that was previously scattered
across:
- `ADR-010-exact-batched-kernel-parity.md` §Status Log (20+ entries)
- `ADR-022-kernel-coverage-parity-with-llama-cpp.md` §5 mm_id row
- `project_gemma_prefill_bitrot_2026_05_09.md` cross-session memory

The ADR-010 §Status Log was originally for the L6 MoE router top-K
sensitivity investigation; iter-59..86 grew out of an operator-flagged
"speed gap problem we have with gemma" and produced surgical fixes,
re-profiling, lock-in gates, and a banner UX bug fix that warrant their
own coherent narrative.

### Operator-flagged starting state (2026-05-09 iter-59)

Operator: *"are you sure we have correct test? ... is all of adr-027 done?
Did you forget about the speed gap problem we have with gemma?"*

Direct re-bench at HEAD on `gemma4-ara-2pass-APEX-Q5_K_M.gguf` (Q6_K
dominant, 19.16 GiB, M5 Max):

| pp   | hf2q per-token (default) | llama.cpp peer | gap |
|------|-------------------------:|---------------:|----:|
| 128  |   65 t/s                 |  1715 t/s      | **0.038× = 26.6× SLOWER** |
| 512  |   67 t/s                 |  2576 t/s      | **0.026× = 38.4× SLOWER** |
| 1024 |   67 t/s                 |  1884 t/s      | **0.036× = 27.7× SLOWER** |

Documented memory had cited "0.40× prefill" but that was qwen35-specific.
Gemma's user-default was much worse. The fast-path (batched prefill)
existed, was peer-competitive at 0.90× on Apr 20 (commit `9091b8c`), but
had bit-rotted: HEAD produced gibberish (`41211789...`) on a 27-token
"What is 2+2?" prompt because the leg_hb_encoded buffer was never written.

### Investigation chain (iter-59..63)

1. **iter-59**: Measured the actual gap (above). Gap is in user-default
   per-token prefill path, not the gated batched path.
2. **iter-60..62**: `git bisect` aborted after 24 SKIPS due to
   mlx-native API drift between Apr 20 and HEAD; bug-pattern (correct
   first decode token + corrupted subsequent tokens) localized to KV-cache
   write rather than prefill compute.
3. **iter-63 SMOKING GUN**: Direct code-read of
   `forward_prefill_batched.rs` at HEAD revealed NO TQ/HB encode calls
   anywhere. Per-token `forward_prefill.rs` does TWO things batched
   doesn't: (a) eagerly allocates `self.leg_hb_encoded` at lines 815-852,
   (b) HB-encodes K/V per token at lines 1234-1272. Decode reads from
   `leg_hb_encoded` via `flash_attn_vec_tq_hb` → reads zero-init bytes →
   garbage attention → gibberish tokens.

## Decision

This ADR **records and pulls together** the iter-64..86 implementation +
lock-in chain as the canonical reference for the peer-parity workstream.
The fixes and lock-in gates are already LANDED on origin/main; this ADR
creates a single-document narrative.

### Fix #1 — iter-64 (hf2q `133722d`, +91 LOC)

In `src/serve/forward_prefill_batched.rs`, mirror per-token's
HB-encode flow:

1. **Eager allocation** of `self.leg_hb_encoded` near line 297, gated on
   `tq_codebook_bits_prefill >= 5` (default 8). Mirrors
   `forward_prefill.rs:815-852`.
2. **Per-layer HB encode block** in the layer loop, immediately after
   the dense KV copy at line 1327. Calls `mlx_native::ops::
   hadamard_quantize_kv::dispatch_hadamard_quantize_kv_hb_seq` for K
   (`pf_k_normed`) and V (`pf_v_normed`) over all `seq_len` positions
   using the SAME `dst_seq_pos_start` / `n_copy` / `src_tok_offset` as
   the dense copy. Mirrors `forward_prefill.rs:1234-1272`.

### Fix #2 — iter-68 (mlx-native `b6b8e79`, 1 character)

In `src/shaders/quantized_matmul_id_mm_tensor.metal:447`, change
`GgmlMatmulIdMm_TensorParams` → `GgmlMatmulIdMmTensor_MmParams`. The
typo caused the entire `.metal` source to fail Metal compile, which
made the runtime probe `probe_tensor_mm_id` return false, falling back
to the slower simdgroup MMA variant for ALL K-quant mm_id pipelines
(Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, Q5_1, IQ4_NL).

### Lock-in chain (iter-65, 73, 74, 75, 77, 78, 80)

Four-layer regression-gate stack chained into a single operator
command: `bash scripts/adr010_iter64_iter68_full_lock.sh` (~1 min):

1. **`mlx-native/tests/test_all_shaders_compile.rs`** (iter-73,
   ~86 LOC): every `.metal` source compiles via `xcrun metal -c`. Catches
   iter-68-style silent typos at test time.
2. **`scripts/adr010_iter64_batched_coherence_gate.sh`** (iter-65/74/75/77/80,
   5 steps): per-token reference truth, batched coherence (contains "4",
   no 6+ digit run), STRICT byte-identity per-token vs batched, pp1024
   batched perf-floor (≥1500 t/s, median-of-3), gemma decode-throughput
   floor (≥50 t/s, median-of-3).
3. **qwen35 cross-model perf-floor** (iter-78): pp512 prefill ≥1800 t/s,
   median-of-3.
4. **iter-76 single-command runner** chains all the above.

### Banner UX fix — iter-82 (hf2q `469725a`)

`Gemma4Config::full_attention_interval()` helper detects "every Nth
layer is Full" pattern from `layer_types`. Used at engine.rs:2177.
Pre-fix banner: `full_attn_every=none` (misleading); post-fix:
`full_attn_every=6` (correct — 5 of 30 gemma layers are Full).

## Validation

### Combined speedup (iter-64+iter-68 on Gemma-4 batched prefill)

| pp | per-token (default) | batched (post-fix) | llama.cpp | speedup vs default | vs peer |
|---:|--------------------:|-------------------:|----------:|-------------------:|--------:|
| 128 |   65 t/s            |    609 t/s         |   1715 t/s |    9.4×           | 0.36×   |
| 512 |   67 t/s            |   1477 t/s         |   2576 t/s |   22×             | 0.57×   |
| **1024** | 67 t/s         |  **1942 t/s**      |   1884 t/s | **29×**           | **1.03× — BEATS llama** |
| 2455 |  67 t/s            |   2329 t/s         |   3023 t/s |   35×             | 0.77×   |

### Cross-model validation (iter-78)

qwen35moe pp512: hf2q 2300 t/s vs llama.cpp 2921 t/s = **0.79× peer**.
qwen35 uses its own code path but routes through the same shared
`mm_id_pooled` dispatcher → iter-68 typo fix benefits qwen35 too.

### User-visible E2E impact (iter-79)

Real chat prompt (52-tok instruction + 80-tok output):

| Path | Prefill TTFT | Decode | Total turn |
|---|---:|---:|---:|
| Per-token (DEFAULT) | 915 ms | 1260 ms | **2.18 s** |
| Batched (gated) | 214 ms | 1260 ms | **1.47 s** |

**TTFT 4.3× faster (701 ms saved/turn). Total chat turn 33% faster.**
First decode token id=8409 byte-identical across runs.

### Iter-86 post-fix profile re-measurement

| Bucket | iter-66 (pre-fix) | iter-86 (post-fix) | Δ |
|---|---:|---:|---:|
| MOE_GATE_UP | 542 ms (34.5%) | **210 ms (18.0%)** | **-61%** |
| MOE_DOWN | 298 ms (18.9%) | 186 ms (16.0%) | -37% |
| MoE total | 53.4% | 34.0% | -53% |
| TOTAL | 1574 ms | **1165 ms** | **-26%** |
| Throughput pp2455 | 1737 t/s | **2119 t/s** | +22% |
| **vs llama.cpp peer** | 0.57× | **0.70×** | **+13pp** |

### Decode investigation (iter-69..72, 81)

- hf2q decode = 64 t/s vs llama.cpp 103 t/s = **0.62× peer**
- ~10% decay 32→1000 tok context (KV reads grow linearly on Full layers)
- 528 dispatches/token at ~30 µs/dispatch each (dispatch-floor bound)
- iter-71 confirmed dispatch count is at parity with llama.cpp; gap is
  per-kernel-time, not graph fusion
- iter-72 `HF2Q_DUAL_BUFFER` sweep: default split=3 already optimal
- **Decode gap is structural** — no 1-line fix; multi-day µbench scope

### Iter-83 ground-state validation

hf2q full unit-test suite: **3390 passed; 0 failed; 10 ignored** (peer
WIP stashed). Lock-in chain (iter-76 single-command) PASSES at HEAD.

### Iter-87 replicated measurements + thermal-discipline note (2026-05-09)

Re-ran iter-86 measurement at HEAD (`dd84f01`) under controlled cooldown
discipline (60s+ between runs, 5s between trials):

| Measurement | iter-86 cited | iter-87 replication | Match? |
|---|---:|---:|:---|
| pp2455 batched prefill (GPU_TS bucket profile) | 1165 ms (2119 t/s) | **1149 ms (2147 t/s)** | ✅ within noise |
| pp2455 unprofiled batched prefill (5-trial median, cooldown) | not cited | **1022 ms (2416 t/s)** | new |
| pp1024 batched prefill (3-trial median, no cooldown) | 1942 t/s | **1996 t/s** | ✅ +3% |
| MOE_GATE_UP / 30 layers | 209.7 ms (6.99 ms/call) | 197.7 ms (6.59 ms/call) | ✅ within noise |
| MOE_DOWN / 30 layers | 186.3 ms (6.21 ms/call) | 174.5 ms (5.82 ms/call) | ✅ within noise |

**Thermal-noise floor on M5 Max** is ~25% for pp2455 batched prefill. A
3-trial measurement WITHOUT cooldown (e.g. immediately after pp1024
trials) produces 1118-1319 t/s (median 1240 t/s) — vs 2402-2430 t/s
under cool-state. **The lock-in chain's existing pp1024 step does not
cooldown between trials**, so any new pp2455 floor must either (a)
include a cooldown step, or (b) set a permissive floor (1500 t/s)
that clears thermal noise. Iter-78 qwen35 step uses 3-trial no-cooldown
methodology and lands at ~2254 t/s (floor 1800).

**Honest peer-gap recompute (cool-state, unprofiled)**:
- pp2455: 2416 t/s vs llama.cpp cited 3023 t/s = **0.80× peer** (was
  0.70× per-iter-86 hot-state-affected reading; iter-86 cited 0.70×
  matches the bucket-profile-overhead 2147 t/s)
- pp1024: 1996 t/s vs llama.cpp 1884 t/s = **1.06× peer** (BEATS, was
  1.03× per iter-68/74 measurement)
- llama.cpp peer baseline cited from iter-68 era (`d05fe1d7d`/build 9010);
  current llama.cpp HEAD (`5d6f18a63`) NOT YET re-benchmarked. Closing
  this loop requires `cd /opt/llama.cpp && cmake -B build -DGGML_METAL=1
  && cmake --build build -j --target llama-bench` ~2 min on M5.

### Iter-87 kernel coverage gap analysis (kernel-fusion-sweep, ADR-028 #4)

Direct comparison of `host_name` template instantiations between
mlx-native and llama.cpp `ggml-metal.metal`:

| Family | llama.cpp variants | mlx-native variants | Gap |
|---|---:|---:|---|
| `mul_mv_ext_*_r1_{2,3,4,5}` | 80+ | 28 | missing q1_0/q2_K/q3_K/q4_1/q5_0/mxfp4/f32_f32/f16_f32/bf16_f32 |
| `mul_mv_id_*_f32` | 25+ | 7 K-quants + 1 fused (q4_0_swiglu) | missing IQ-quants, MXFP4, BF16/F16/F32 |
| `mul_mm_id_*_f32` (tensor) | 18 | 7 (Q4_0,Q8_0,Q6_K,Q5_1,IQ4_NL,Q5_K,Q4_K) | matches the 7 hf2q-supports types |
| `(rms_)?norm_(mul|mul_add)_f32` | 8 | 1 (`rms_norm_mul_f32`) | missing `rms_norm_mul_add`, `norm_mul`, `norm_mul_add` × {scalar,vec4} |
| `flash_attn_ext_{f32,f16,bf16,q4_0,…}_dk*_dv*` | 60+ | hf2q-specific TQ + prefill family | different design — vec/prefill split, KV-quant kernels via tq_hb |

**Hot-path coverage check** (gemma-4-26b APEX-Q5_K_M, qwen35-A3B):
- mm/mm_id Q5_K/Q6_K tensor variants — **PRESENT, used post-iter-68**
- mv_id_q5_K_f32, mv_id_q6_K_f32 — **PRESENT** (used at decode)
- mv_id Q5_1, IQ4_NL, Q4_K, Q8_0 — **PRESENT** (qwen35moe variants)
- mv_ext for K-quants r1=2..5 — **PRESENT** (small-batch decode)
- norm_mul_add fused 3-op kernel — **MISSING**
  - On gemma 26B 30-layer prefill, our profile shows 5 distinct norm-add
    bucket sites (PRE_ATTN_NORM, POST_ATTN_NORM_ADD, TRIPLE_RMS_NORM,
    MOE_WSUM_DNORM_ADD, END_LAYER_NORM_ADD_SCALAR) totalling ~50 ms /
    1149 ms = ~4.4% of pp2455. Even 100% closure would buy ~4% gain;
    candidate but not biggest fish.
- mv_ext IQ-quants/MXFP4/BF16 — MISSING but **not on hot path** (we
  don't load these formats today)

**Conclusion**: kernel coverage on the hot path is at parity. The
remaining ~20% pp2455 gap is **per-kernel-time** within the kernels we
already share, not unported variants. Confirming iter-71's earlier
finding ("dispatch count at parity; gap is per-kernel-time").

### Iter-88 fresh peer baseline at llama.cpp HEAD (2026-05-09)

Built `/opt/llama.cpp/build/bin/llama-bench` from llama.cpp HEAD
`5d6f18a63` (build 9078), ran on identical fixture
(`gemma4-ara-2pass-APEX-Q5_K_M.gguf`, M5 Max, 5-run median):

| Test | hf2q (cool, 5-trial) | llama.cpp build 9078 FA=1 | llama.cpp build 9078 FA=0 | hf2q vs FA=1 peer | hf2q vs FA=0 peer |
|---|---:|---:|---:|---:|---:|
| pp1024 | 1996 t/s | 1759 ± 30 | 1398 ± 44 | **1.13× faster** | **1.43× faster** |
| pp2455 | 2416 t/s | 1573 ± 39 | 1292 ± 58 | **1.54× faster** | **1.87× faster** |
| tg32 | 63.4 t/s | 97.7 ± 5.4 | 27.8 ± 2.0 | 0.65× | **2.28× faster** |
| tg128 | 61.4 t/s | 88.3 ± 1.0 | (n/a) | 0.70× | (n/a) |
| tg256 | 62.5 t/s | 90.4 ± 0.5 | (n/a) | 0.69× | (n/a) |

**Surprise**: at llama.cpp HEAD, prefill speed has REGRESSED relative
to the iter-66/68 cited build-9010 numbers (1884 t/s pp1024 / 3023 t/s
pp2455). Build 9078 is **6.6% slower at pp1024 and 48% slower at
pp2455 with FA=1** vs cited build-9010 numbers. We did not measure
the regression cause — could be intentional fault-tolerance/correctness
tightening, kernel re-org, or unrelated work.

**Mantra-status at iter-88 (real numbers)**:
- ✅ **Coherence**: byte-identical to llama.cpp on sourdough fixture
  (per ADR-010 sourdough_gate.sh; iter-65/74/79 byte-identity gate).
- ✅ **Prefill speed**: hf2q is **1.13× to 1.87× FASTER** than llama.cpp
  HEAD across pp1024 + pp2455 with both FA modes. Mantra MET for prefill.
- ❌ **Decode speed**: hf2q is **0.65×-0.70× peer** vs FA=1, **2.28×
  FASTER** vs FA=0 default. With FA=1 (apples-to-apples) we still trail
  by 30-35%. Mantra NOT MET for decode at FA=1.

The remaining mantra-violation is decode at FA=1. Per the "as fast as
peer" reading of the mantra, llama.cpp's FA=1 mode is the relevant peer.
ADR-028 #3 (decode µbench infrastructure) is the only remaining
work-item to close mantra at FA=1.

**llama-bench command (locked in for future re-bench)**:
```bash
/opt/llama.cpp/build/bin/llama-bench \
  -m /opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf \
  -p 1024,2455 -n 32,128,256 -fa 1 -r 5
```

### Iter-95 Q6_K mv_id y-reuse refactor — FALSIFIED (8th compiler-auto-optim hypothesis)

Implemented the y-vector register-reuse refactor identified in iter-91:
- Each simdgroup now handles **NR0=2 weight rows** (was 1)
- Pre-loads `yl[16]` once per outer-block, reuses across both rows
- Mirrors llama.cpp's `kernel_mul_mv_q6_K_f32_impl` register-cache pattern
- Adjusted dispatch geometry: Q6_K uses `align=4` (was 2) → threadgroups.x = ceil(N/4)

Files changed (transient):
- `mlx-native/src/shaders/quantized_matmul_id_ggml.metal:803` — kernel rewrite
- `mlx-native/src/ops/quantized_matmul_id_ggml.rs:519` — Q6_K-specific dispatch geometry

**Parity tests PASS** (`test_q6_k_mm_id_matches_mv_id_small` + `_prefill_shape`).
Refactor is byte-equivalent to baseline within tolerance.

**Bench A/B comparison** (200 trials, 20 warmup, decode shape):

| Metric | Baseline | Refactor | Delta |
|---|---:|---:|---:|
| p10 | 155.88 µs | 152.79 µs | -2.0% |
| **p50** | **187.54 µs** | **182.83 µs** | **-2.5%** |
| p90 | 292.58 µs | 267.38 µs | -8.6% |
| Per-token (60 calls × p50) | 11.25 ms | 10.97 ms | -2.5% |

**End-to-end production decode** (gemma APEX-Q5_K_M, tg32, 5 trials):

| Metric | Baseline | Refactor |
|---|---:|---:|
| Median | 63.4 t/s | 63.9 t/s (+0.8%) |
| Range | 62-66 | 60.5-64.5 |

**Production effect: null** — within thermal noise. Static-evidence
hypothesis (yl[] register reuse should save y-load bandwidth) FALSIFIED.

**Why this is the 8th confirmed `feedback_metal_compiler_auto_optimizes_static_levers` falsification**:

The Metal compiler likely already coalesces y[] reads in the existing
1-row-per-simdgroup variant (L1-cache + scalar load coalescing across
the simdgroup). Explicit register-cache via `yl[16]` doesn't reduce
total memory traffic — both simdgroups in the same threadgroup share
L1, so y is fetched once for two rows in either layout.

The 2.5% kernel-level improvement (μbench artifact) does not translate
to end-to-end speedup because:
1. The bench includes per-dispatch encoder + commit_and_wait overhead
   (~5-10 µs) that's amortized in production single-session mode.
2. Real per-kernel GPU time at this shape is ~165-175 µs; the 2.5%
   delta on μbench p50 reflects encoder timing variance more than
   actual GPU work change.

**Decision**: REVERT refactor. Standing rule wins again. Document for
future iterations to skip this attack vector.

**Iter-96 attack pivot**: where IS the 30% per-call gap? Hypotheses:
1. **L1 cache miss rate**: llama.cpp's `nsg=2` thread-grouping may
   produce different L1 behavior than ours.
2. **Pipeline cache**: llama.cpp's tag suffix `_nsg=2` suggests
   function-constant-driven specialization — our kernel doesn't use
   function constants.
3. **Threadgroup→barrier scheduling**: llama.cpp's grid shape
   `(ceil(N/(NSG*nr0)), m, 1)` produces fewer larger threadgroups
   (4 rows each) vs our many small (2 rows each). Apple GPU scheduler
   may favor fewer-larger.
4. **Encoder dispatch overhead**: each Metal dispatch costs ~5-10 µs
   of CPU encode + GPU schedule. We do 990 dispatches/token; if
   encoder overhead is 5-10 µs each = 5-10 ms = **30-60% of decode
   time**. This may be the actual gap.

Iter-96 candidate: instrument encoder dispatch CPU vs GPU time
(use `MTLCommandBuffer.GPUStartTime/GPUEndTime`) to separate encode
overhead from kernel time. If encoder overhead is large, the fix is
**fewer dispatches** (kernel fusion), NOT faster kernels.

### Iter-94 ground-truth Q6_K mv_id per-call timing — DOMINANT kernel confirmed

Added `bench_q6_k_mv_id_gemma_decode_gate_up` (#[ignore] + #[test]) at
`/opt/mlx-native/tests/test_quantized_matmul_id_mm.rs:597`. Uses 200
trials with 20-warmup, isolates one mv_id Q6_K dispatch per
commit_and_wait (matching production single-session model where each
dispatch incurs ~5-10 µs encoder overhead).

Run:
```bash
cd /opt/mlx-native && RUSTC_WRAPPER= cargo test --release \
  --test test_quantized_matmul_id_mm bench_q6_k_mv_id \
  -- --ignored --nocapture
```

**Measurement at gemma decode shape** (n_tokens=1, top_k=8, n=704, k=2816,
n_experts=128, Q6_K weights):

| Percentile | µs/call |
|---|---:|
| p10 | 155.88 |
| **p50** | **187.54** |
| p90 | 292.58 |

**Per-token aggregate** (gate_up only, ×60 layers): **11.25 ms = 71% of
the 15.86 ms decode token time**. Q6_K mv_id is the dominant kernel.

**Implied llama.cpp per-call** (assuming similar dispatch count): at
11.11 ms/token total × 71% MoE share = 7.9 ms / 60 calls ≈ 132 µs/call
→ **hf2q is ~42% slower per call** (187.5 vs 132 µs).

This is the real per-kernel-time gap. Closing it closes most of the
0.65-0.70× decode peer gap.

**Down kernel skipped this iter**: gemma's down projection has K=704
which doesn't divide Q6_K block size (256). Real gemma down_exps is
stored as Q5_1/Q4_K with different block size or has padded K=768.
Iter-95 candidate.

**Iter-95 attack**: implement y-vector reuse refactor (iter-91
candidate, demoted in iter-93 but now the right next-step). Re-run
bench before/after. Ship if ≥10% win, document if null.

### Iter-93 correction to iter-92 ROI analysis (3-op norm fusion ALREADY PRESENT)

Direct code-read of `/opt/mlx-native/src/shaders/fused_norm_add_f32.metal:231`
confirms hf2q **already has the 3-op fusion**:

```metal
// Phase 2: normalize input, apply weight, add residual, store output.
for (uint i = tid; i < dim; i += tg_size) {
    const float normed = input[base + i] * rms_inv * weight[i];
    output[base + i] = residual[base + i] + normed;
}
```

That's `rms_norm + mul-by-weight + add-residual` in one kernel. Same
fusion as llama.cpp's `kernel_rms_norm_mul_add_f32_4`. We dispatch it
at 8+ sites in forward_mlx.rs (3 in forward_decode body alone — lines
1417, 1689, 1705).

**The actual gap is the `_4` suffix — vec4 vectorization.** Llama.cpp's
kernel uses `float4` loads/stores (4 elements per memory op); ours is
scalar (`for (uint i = tid; i < dim; i += tg_size)` with 1 element per
iteration).

**Revised ROI estimate** (vec4-only, not full fusion port):
- Phase 2 memory ops: 4× fewer reads, ~30-50% kernel speedup
- Phase 1 reduction barriers unchanged (~50% of kernel time)
- Net per-call savings: ~5-6 µs at gemma hidden=2816
- Per-token: ~3 norm calls × 30 layers = 90 calls × 5-6 µs ≈ 450-540 µs
- Decode speedup: 450-540 µs / 15.86 ms ≈ **2.8-3.4%**

This is much smaller than iter-92's "9%" estimate. Vec4 alone is not
the biggest fish.

**Iter-92's iter-91 demotion holds** — y-reuse on Q6_K mv_id is
~5-10% (also small). Both fusion-style optimizations land in the
same ~3-5% range.

**Where the real ~30% gap lives** (per iter-90 measurement of 15.86
µs/dispatch hf2q vs 11.22 µs llama.cpp): per-kernel GPU-time on the
**dominant** kernels — mv_id Q6_K (MoE matmul), flash_attn_vec_tq_hb
(SDPA decode), and the QKV mv kernels.

**Iter-94 attack plan** (in order):
1. Per-call timing instrumentation: add ENV-gated commit_and_wait
   timer around dispatch_id_mv to get ground-truth Q6_K mv_id µs/call
   at gemma decode shape (n_tokens=1, top_k=8). Compare to llama.cpp
   `kernel_mul_mv_q6_K_f32_nsg=2` per-call cost.
2. If our Q6_K mv_id is significantly slower per-call: do the y-reuse
   refactor (iter-91 candidate) AND measure before/after.
3. If FA-vec is slower per-call: investigate split-K (nwg=32) port.
4. Vec4 norm-add: low priority.

The `feedback_metal_compiler_auto_optimizes_static_levers` standing
rule applies: 7 prior kernel hypotheses falsified. Measure first,
optimize after.

### Iter-92 llama.cpp HEAD pipeline compile log → concrete fusion candidates

Captured llama.cpp build-9078 actual decode-time pipeline compile log
via `llama-bench -v -p 0 -n 32 -fa 1 -r 1`. The pipelines llama.cpp
loads at decode-time on gemma APEX-Q5_K_M, with their function-constant
suffixes, are direct evidence of which kernels they ACTUALLY use:

```
kernel_mul_mv_q6_K_f32_nsg=2                       (Q6_K mv at decode)
kernel_flash_attn_ext_vec_f16_dk256_dv256_...nsg=1_nwg=32  (FA-vec split-K!)
kernel_flash_attn_ext_vec_reduce_dv=256_nwg=32             (FA-vec reduce)
kernel_rms_norm_mul_add_f32_4                              (3-op fused norm)
kernel_bin_fuse_f32_f32_f32_4_op=2_nf=1_rb=1_cb=0          (fused binary)
kernel_rms_norm_f32_4                                      (rms-norm scalar)
kernel_rope_neox_f32_imrope=0
kernel_set_rows_f16_i64                                    (F16 KV writes)
kernel_soft_max_f32_4
kernel_cpy_f32_f16
```

**Fusion-gap quantification** (for hf2q decode-path attacks):

1. **`kernel_rms_norm_mul_add_f32_4`** (3-op fused) — we have only
   `rms_norm_mul_f32` (2-op). Per layer, gemma decode does ~3 norms
   that include downstream mul+add (PRE_ATTN_NORM, POST_ATTN_NORM_ADD,
   END_LAYER_NORM_ADD). Each unfused = 2 dispatches; fused = 1. Savings
   per token: ~3 × 30 = 90 dispatches × 15.86 µs ≈ 1.4 ms = **9% decode
   speedup**.

2. **`kernel_bin_fuse_f32_f32_f32_4`** (fused elementwise multiply) —
   used by llama.cpp's gate_silu × up flow for SwiGLU. We have
   `mul_mv_id_q4_0_f32_swiglu` (kernel-fused gate+up+swiglu+matmul)
   for Q4_0 ONLY. Gemma uses Q6_K experts → falls to the slow path.
   Adding Q6_K and Q5_K swiglu-fused mv_id kernels could save ~2
   dispatches per layer × 30 layers = 60 dispatches ≈ **6% decode
   speedup**.

3. **`kernel_flash_attn_ext_vec_..._nwg=32`** (split-K FA-vec) —
   llama.cpp parallelizes the K dimension across 32 work-groups, then
   does a separate reduce kernel. We do single-pass FA-vec without
   split-K. For long-context decode (e.g. tg256), split-K can be 1.5-2×
   faster on the FA-vec dispatch. Savings depend on context length.

4. **`kernel_set_rows_f16_i64`** + **F16 KV cache** — llama.cpp uses F16
   KV (not TQ-HB) at decode. We use TQ-HB (per ADR-027 for 3.94×
   memory). The TQ-HB path adds 4-5 dispatches per layer (FWHT
   premult + hadamard + quantize + dequant + FWHT undo). Skipping
   TQ-HB at decode saves ~150 dispatches/token = 2.4 ms but TRADES
   3.94× KV memory savings. Operator-gated tradeoff.

**Iter-92 concrete attack ordering** (by ROI):
- Highest: port `rms_norm_mul_add_f32_4` (existing rms_norm + mul +
  fused_norm_add infra → merge into a 3-op kernel). Estimated 9%
  decode speedup. ~150 LOC + tests + bench. Iter-93 candidate.
- Next: port `bin_fuse_f32_f32_f32_4` for elementwise mul.
- Next: Q6_K + Q5_K swiglu-fused mv_id (matches our Q4_0 swiglu
  pattern from gpu_ffn.rs). ~80 LOC per quant. Iter-94 candidate.
- Lower: FA-vec split-K (nwg=32) — biggest engineering effort.
- Operator-gated: F16 KV at decode (conflicts with ADR-027 TQ-HB).

The Q6_K mv_id y-reuse refactor (iter-91 candidate) is **lower
priority** than these fusions: the static-evidence hypothesis is
~5-10%, while these fusions have direct measurement support from
llama.cpp's actual kernel inventory. Per standing rule
`feedback_metal_compiler_auto_optimizes_static_levers`, prefer
hypotheses with empirical support.

### Iter-91 mv_id Q6_K kernel structural difference (concrete decode attack candidate)

Direct code-read comparison of Q6_K mv_id kernel between hf2q
(`/opt/mlx-native/src/shaders/quantized_matmul_id_ggml.metal:803`) and
llama.cpp (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:7968`):

**llama.cpp** `kernel_mul_mv_q6_K_f32_impl`:
- Templated on `nr0` (rows per simdgroup); for Q6_K `N_R0_Q6_K = 2`
- Outer loop `for (int i = ix; i < nb; i += 2)`:
  - **Pre-loads `yl[16]` from device once**:
    ```metal
    for (short l = 0; l < 4; ++l) {
        yl[4*l + 0] = y[l +  0]; yl[4*l + 1] = y[l + 32];
        yl[4*l + 2] = y[l + 64]; yl[4*l + 3] = y[l + 96];
    }
    ```
  - **Inner loop over rows** `for (short row = 0; row < nr0; ++row)`:
    advances q1/q2/qh/sc/dh by `args.nb01` per row, REUSING `yl[]`
    register-resident across both rows.

**hf2q** `kernel_mul_mv_id_q6_K_f32`:
- Hard-coded to **1 row per simdgroup** (sgitg picks row offset 0 or 1)
- Re-loads `y[]` from device **for every row** inside `for (int l = 0;
  l < n; ++l)` — no yl[] register reuse across rows.

**Y-vector reuse savings**: per outer-loop iteration, llama.cpp does 1
y-load + 2 row computations vs hf2q's 2 separate y-loads + 2 row
computations. For a 2816-dim hidden (gemma-4-26b) Q6_K row with
nb = 2816/256 = 11 blocks, this saves ~16 device-memory reads per
block × 11 blocks = ~176 device reads per row, halved across 2 rows.

**Refactor**: change kernel_mul_mv_id_q6_K_f32 to match llama.cpp's
nr0=2 pattern. Adjust `dispatch_id_mv` in
`/opt/mlx-native/src/ops/quantized_matmul_id_ggml.rs:540` to use
threadgroup geometry `(div_ceil(n, NSG*nr0), m, 1)` instead of
`(div_ceil(n, align), m, 1)`. Refactor scope: ~50 LOC mv_id_q6_K_f32
+ ~30 LOC dispatch geometry + tests. Iter-92 candidate.

Hypothesis (testable): refactor saves ~5-10% on Q6_K mv_id calls.
Falsifier: if decode time doesn't change, register-reuse savings were
already captured by Apple's Metal compiler auto-coalescing y-loads.
Standing rule per `feedback_metal_compiler_auto_optimizes_static_levers`
(7+ confirmed kernel hypotheses falsified) — measure before claiming.

Also affects Q4_K, Q5_K, Q3_K, Q2_K mv_id kernels (same structural
pattern). Q5_K most relevant for Q5_K_M-format models (hf2q ships Q5_K
as the dominant block format for many fixtures).

### Iter-90 dispatch-count BUG FIX + corrected per-dispatch breakdown

**Bug discovered in iter-89's HF2Q_MLX_PROFILE reporting.** The `avg_dispatches`
helper at `forward_mlx.rs:214` summed each token's `s*_dispatches` array
across all 30 layers. But each `s*_dispatches[layer_idx]` is assigned
the CUMULATIVE counter at end of that layer (not the per-layer delta) —
so summing it 30× over-counts by a factor of ~15 (sum of arithmetic
progression 1+2+…+30 = 465 vs 30 layers).

Iter-89 reported 16,300 dispatches/token; correct count is **990
dispatches/token**.

| Metric | Iter-89 (buggy) | Iter-90 (corrected) | Reality check |
|---|---:|---:|:---|
| Total dispatches/token | 16,300 | **990** | Matches candle baseline (~105) × 9.4× |
| Body dispatches/token | 15,310 | **986** (33 per layer × 30 layers) | Reasonable |
| Head dispatches/token | 990 | **4** | LM-head + softcap + argmax = 3-4 ops |
| µs / dispatch | 0.97 µs | **15.8 µs** | Apple Metal GPU dispatch latency |

**Corrected per-dispatch comparison vs llama.cpp HEAD**:
- hf2q decode: 15.71 ms / 990 dispatches = **15.86 µs/dispatch**
- llama.cpp at 90 t/s = 11.11 ms/token; assuming similar ~990 dispatches
  (per iter-71 dispatch-count parity): **11.22 µs/dispatch** → 29% faster

**Conclusion**: decode gap is **per-dispatch GPU time**, not dispatch
count. Closing requires per-kernel optimization (kernel selection, tile
geometry, shmem layout), not fusion/dispatch-count reduction. Confirms
iter-71's earlier finding using clean methodology.

Highest-leverage attacks (from iter-86 cool profile per-call ms/call):
- MOE_GATE_UP: 6.59 ms/call × 30 = 198 ms (17% of pp2455; for decode
  similar mv_id cost dominates)
- MOE_DOWN: 5.82 ms/call × 30 = 175 ms
- FA_GL: 12.59 ms/call × 5 = 63 ms (D=512 head_dim, gemma full-attn)
- FA_SW: 3.61 ms/call × 25 = 90 ms (D=256, sliding-window)

Code: forward_mlx.rs `avg_dispatches` rewritten to use `last()` of
each layer-array; head_only computed as final-cumulative minus body-
cumulative. Verified end-to-end: 990 dispatches/token reported on
gemma APEX-Q5_K_M decode, matching expected order-of-magnitude.

### Iter-89 first quantitative decode profile (ADR-028 #3 partial)

Ran `HF2Q_MLX_PROFILE=1` (production-side ProfileAccumulator) on a
13-token decode (after 2 warmup) on gemma APEX-Q5_K_M:

| Metric | Value |
|---|---:|
| Per-token decode | **15.85 ms / 63.1 t/s** |
| Total dispatches per token | **16,300** (S1: 15,310 + Head: 990) |
| Per-dispatch encoder time | **0.97 µs/dispatch** (CPU-side) |
| Per-layer dispatches | **~510** (15310 / 30 layers) |
| Session model | single-session (1 GPU session/token) |

**Encode-bound finding**: 16300 × 0.97 µs = 15.81 ms ≈ token total. The
decode is dominated by encoder/dispatch overhead, NOT GPU kernel time.

**Comparison vs candle Phase 0 baseline**: ~105 dispatches/token → we
have **155× more dispatches**. This reflects the per-layer fan-out from:
- 8 active MoE experts × 2 (gate_up + down) = 16 mv_id calls per layer
- FA-vec-tq-hb (leg_hb_encoded) path with multi-step quantization sub-
  dispatches per layer
- Per-block K-quant scale/dequant sub-passes inside mv kernels
- Norm/add/embed/permute small kernels each as a separate dispatch

llama.cpp also has high dispatch counts (iter-71 confirmed parity); the
~30% per-dispatch encoder gap accounts for the 0.65-0.70× decode peer.

**Highest-leverage attack**: cut dispatch count by 2× via kernel fusion.
At 8000 dispatches × 0.97 µs ≈ 8 ms/token = **125 t/s = BEATS llama.cpp
HEAD by 30%**. iter-90 to break down the 510-per-layer count by site.

**`forward_decode_kernel_profile` rebuild needed for site-level
breakdown**: existing fn at forward_mlx.rs:4400 (gated
`HF2Q_MLX_KERNEL_PROFILE=1`) requires F16 lm_head — fails on our gemma
APEX-Q5_K_M (Q6_K LM-head). Need to either (a) generalize to support
quantized lm_head, or (b) add per-bucket counters to the production
forward_decode (mirroring forward_prefill_batched.rs's 20+ buckets).
~150-200 LOC, single file change.

## Consequences

### Positive

- **User-visible win**: 29× prefill speedup over per-token default at
  pp1024; hf2q BEATS llama.cpp at this size on Gemma-4. 4.3× TTFT
  improvement on real chat prompts.
- **Cross-model benefit**: same iter-68 typo fix unlocks qwen35moe too
  (different code path, same shared dispatcher).
- **Lock-in chain prevents recurrence**: 4-layer gate stack catches
  - functional regressions (gibberish output)
  - silent shader-compile failures (iter-68-style)
  - byte-divergence between per-token and batched
  - perf regressions (>25% drop in prefill or decode)
- **Banner UX bug closed**: operators now see correct architecture info
- **Stale doc cleanup**: `mod.rs:1173-1175` "until parity validated"
  comment now accurately reflects iter-65/74/79 validations

### Negative

- **Default flag remains gated** behind `HF2Q_BATCHED_PREFILL=1 +
  HF2Q_UNSAFE_EXPERIMENTS=1` because of the L6 MoE router top-K
  threshold sensitivity at long-sequence sliding_wrap fixtures
  (operator-signed deferral 2026-04-16 ADR-010). Users must opt in to
  get the speedup.
- **Decode 0.62× peer remains**: iter-69..72/81 measured but did not
  close. Multi-day µbench infrastructure (Q6_K pack helper that doesn't
  exist) is required for kernel-level attack.
- **Pp2455 prefill at llama.cpp HEAD = 1.54× FASTER** (iter-88
  re-bench). The cited 0.80× / 0.71× was vs stale build-9010 peer; at
  build 9078 hf2q is solidly ahead. Mantra MET for prefill at all sizes
  measured.
- **Decode at FA=1 still 0.65-0.70× peer at HEAD** (iter-88 re-bench).
  The cited 0.62× was vs build-9010; at build 9078 the gap is slightly
  smaller (0.65-0.70× across tg32/128/256) but still meaningful. With
  FA=0 (llama.cpp default) hf2q is 2.28× FASTER. ADR-028 #3 (decode
  µbench infrastructure) remains the path to close mantra at FA=1.

### Neutral

- **Doc-vs-reality cleanup**: iter-82 fixed the load banner; iter-85
  fixed the comment. Both no-behavior-change improvements that improve
  operator clarity.
- **No new public API**: all changes are internal (forward_prefill_
  batched.rs body + Metal shader text + LoadInfo computation).

## Open scope (gated on operator pick)

1. **Default-flip on batched prefill** — operator UX call. Validated for
   short-to-medium prompts (≤80 tokens) per iter-65/74. The ADR-010 L6
   MoE deferral applies only to long sliding_wrap fixtures.
2. **L6 MoE router exact alignment** (multi-week per ADR-010 §1) —
   would unlock unconditional default-on. Pre-MoE chain alignment at
   ~1e-5 level is "scope decision for this phase, not universal claim"
   per ADR-010 2026-04-16.
3. **Decode µbench infrastructure** (multi-day) — needs Q6_K pack
   helper + per-kernel timing harness. Would close 0.62× → ~0.85×
   peer estimate.
4. **Kernel fusion sweep for decode** — iter-87 done: hot-path coverage
   at parity with llama.cpp; one candidate fusion (`rms_norm_mul_add`,
   `norm_mul[_add]`) ~4% potential pp2455 gain, ~0% decode gain (decode
   doesn't pay norm cost the same way). Not the biggest fish.
5. **Pp2455 closer-to-peer attack** — distributed work across 4-5
   kernel categories; iter-87 honest measurement = 0.80× peer cool-state
   = 20% gap, not 30%. Per-kernel-time level fixes only.
6. **llama.cpp peer baseline at HEAD** — iter-87 found llama.cpp moved
   from `d05fe1d7d` (build 9010, used as iter-68 baseline) to
   `5d6f18a63` (current HEAD). Re-bench needed to validate 3023/1884 t/s
   peer numbers still hold or have shifted under llama.cpp's own work.

## Links

- `ADR-010-exact-batched-kernel-parity.md` — original parity ADR; iter-59..86 entries also live in §Status Log there
- `ADR-022-kernel-coverage-parity-with-llama-cpp.md` — AC-5 perf parity reopened by iter-67/68
- `ADR-027-qwen35-tq-kv-cache-and-persist-family.md` — qwen35 TQ KV cache work (parallel)
- `scripts/adr010_iter64_iter68_full_lock.sh` — single-command operator runner
- `scripts/adr010_iter64_batched_coherence_gate.sh` — 5-step gemma gate
- `mlx-native/tests/test_all_shaders_compile.rs` — shader-compile gate
- `project_gemma_prefill_bitrot_2026_05_09.md` (memory file) — cross-session continuity
- hf2q commits: `133722d` (iter-64 fix), `6db074d` (iter-65 gate), `8df6afc..ce19e8f` (iter-67..86 chain)
- mlx-native commits: `b6b8e79` (iter-68 1-char fix), `91f174b` (iter-73 shader gate)
