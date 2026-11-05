# ADR-063 Q4-GEMMA-KERNEL (#659) and Q4-GEMMA-ATTN (#668): precise tiled warm attention for Gemma 4

Status: falsified 2026-11-05 (see §12); historical design. It was approved
2026-11-03; revision 2 applied the ACCEPT-WITH-CONDITIONS review
(ReviewGemmaKernelDesign, conditions 1–9; answers in §11). Sections 1–11 are
kept as written; no code, build or hardware run backed them. Every number
below is quoted from a retained receipt, with its path, or marked
`[INFERENCE]`.

Stories:
- **Q4-GEMMA-KERNEL** (#659, mlx-native) provides the primitive.
- **Q4-GEMMA-ATTN** (#668, hf2q) integrates it and closes the Gemma 4 family
  verdict. Its predecessors are #659, Q4-GEMMA (#660, the profile) and Q5
  (#371).

Sources:
- ADR-063 sections Q4, Q4-GEMMA-KERNEL, Q4-GEMMA, Q4-GEMMA-ATTN and "Speed
  program, Q1 successors and release mapping".
- The Q1 numeric policy `hf2q.q1.numeric-policy.v1`
  (`docs/research/adr063-q1-numeric-policy.md`).
- Closed PR #552 on branch `exp/gemma-tiled-precise` (`1427dfb0`), with its
  `docs/research/adr063-gemma-tiled-precise.md` and
  `docs/research/adr063-gemma-tiled-precise-mlx-native.patch`.
- PR #544, which turned the default off.
- Stage 2 in `docs/research/adr063-gemma-four-agent-reuse.md`.
- The Q3 scoreboard (`docs/research/adr063-f2-scoreboard.md`).
- The mlx-native 0.15.16 crate source, the current hf2q pin.

`$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`.

## 1. Decision summary

1. **The primitive.** mlx-native ships tiled warm attention with F16 Q and F16
   K. The scores, softmax, P, V operand, O accumulator and stored O are all
   F32, and the scale is applied to the F32 scores (§4). It is one new
   dispatcher pair, d256 (sliding) and d512 (global causal resume). The
   existing F16/BF16 entry points stay bit-identical.
2. **The #552 patch is extended, not adopted as is.** #552's
   `TiledF16Precise` did not remove the Q mismatch. It swapped it for a
   different one (§2.3.1), so its hardware result cannot separate Q error
   from V/O storage error. The contract therefore removes **all three**
   mismatches:
   - Q is rounded once, unscaled, exactly as vec-hybrid rounds it;
   - the V operand is the exact F32 dequantized TQ-HB value;
   - O is stored as F32 (§3.3).

   Three model-free negative controls (§5.1, N1–N3) measure the cost of each
   mismatch individually. The patch's `OutT`/`ScaleScores` template work is
   the starting point.
3. **Numerical acceptance has three layers, frozen before any data.**
   - Primitive tolerance against an F64 reference on the kernel's exact
     inputs (§5.1).
   - hf2q Class A bitwise gates and a layer-0 localization bound (§8.2).
   - The Q1 decision envelope at the canonical divergence coordinate with the
     0.10-nat near-tie policy, plus the paired 24-instance quality campaign
     (§8.3, §9.1).

   No threshold is learned from candidate data (Q1 §3.7).
4. **hf2q integration.** The new `GemmaLiveKernel::TiledF32` is selected by
   `HF2Q_GEMMA_WARM_TILED_ATTENTION=precise`. The default stays `VecHybrid`
   until every gate in §8–§9 passes, and `0` remains the kill switch. The
   route fails closed at model load if its pipelines cannot be built. The
   selected kernel is published in `/hf2q/v1/runtime` active controls.
5. **The kernel alone cannot produce a Gemma Win.** hf2q's makespan is 1.155×
   the peer's (Q3 v9). Tiled warm attention recovers at most about 8–9%
   (§10.1), and v7 with tiled on was still a Loss (1.058). Q4-GEMMA-ATTN closes
   only through a combined candidate that also carries the changes the
   Q4-GEMMA profile names, each with its own design. This design owns the
   attention stage and the closing campaign shape.

## 2. Evidence: what failed and why

### 2.1 The two routes and where they round

Both routes attend the same F16 K bytes and the same TQ-HB V codes. The
rounding map (from #552, re-checked against 0.15.16):

| Quantity | `VecHybrid` (`flash_attn_vec_hybrid_batched`) | `TiledF16` d256 sliding | `TiledF16` d512 global |
|---|---|---|---|
| Q | F32 → `half4` once | F32 → F16, then **× `half(scale·log2e)` in F16** (factor 1.4423828, temperature 1.000216) | F32 → F16, unscaled; scale applied in F32 after Q·K |
| K | F16 cache | same bytes, compacted | same bytes, compacted |
| Scores, softmax, P | F32 | F32 (`fast::exp2`) | F32 |
| V operand | F32: `centroid · norm/√256` (d256) or `norm/sf` (d512), per use | **F16** (`tq_dequantize_hb_kv_seq_f16`) | **F16**, read as `simdgroup_matrix<half>` |
| O accumulator | F32 | F32 (`Otile`) | F32 (`so`) |
| O store | F32 | **F16**, then cast back to F32 | **F16**, then cast back to F32 |

Semantics (window, causal offset, graft offset, scale 1.0, no attention
softcap, GQA mapping, V codebooks) were checked equal by source reading in
#552. No semantic mismatch was found.

### 2.2 Hardware results of record

Canonical divergence coordinate: the token after `if port == 0` in the turn-2
`write_file` (`$E/gemma-divergence-v1/bisect-summary.json`, run
`n4-004-b01-experiment-a0`, binary main `f3e37690`):

| Arm | agent-3 top-2 (` {` / ` \|\|`) | margin | agent-0 top-2 | chosen (rep. penalty 1.05) |
|---|---|---|---|---|
| warm, vec-hybrid (`WARM_TILED_ATTENTION=0`) | −0.149 / −1.978 | 1.829 | −0.058 / −2.877 | ` {` |
| warm, `TiledF16`, four agents | −0.368 / −1.179 | 0.811 | −0.103 / −2.327 | agent-3 ` \|\|` |
| solo cold (reference context) | −0.208 / −1.672 | 1.464 | −0.043 / −3.157 | ` {` |

At agent-3, `TiledF16` moves the decision boundary by D = |1.829 − 0.811| =
1.018 nat (Q1 §3.2 statistic). Log-prob differences equal logit differences
because the normalizer cancels. With the 1.05 repetition penalty the chosen
token flips: tiled wrote a different (correct) patch in 21/21 v7 runs (PR
#544).

Route envelope, four lanes, last row, against vec-hybrid on identical
histories (`$E/gemma-tiled-precise-gates-1427dfb0/gemma_warm_attention_route_envelope.log`):

| Lane (history/suffix) | vec margin (nat) | `TiledF16`: top-1, D | `TiledF16Precise` (#552; Q fed as round(q·log₂e)): top-1, D |
|---|---|---|---|
| 0 (465/56) | 1.161 | agree, 0.099 | agree, **0.510** |
| 1 (714/200) | 0.167 | **flip**, D ≥ 0.168 | **flip**, D ≥ 0.626 |
| 2 (943/311) | 0.392 | **flip**, D ≥ 0.427 | agree, 0.047 |
| 3 (1254/385) | 3.156 | agree, 0.014 | agree, 0.101 |
| whole-vocabulary max \|Δlogprob\| | 0 | 3.93 | 3.75 |

D on flipped lanes is a lower bound: the gate log has no route top-1 log-prob
on the reference side. Under Q1 §3.2 both tiled variants are `RouteSensitive`
(D > 0.25), and both fail the 0.10-nat near-tie policy: a flip where the vec
margin is at least 0.10 nat.

Per-layer localization, history 714 / suffix 200
(`…/gemma_warm_tiled_precise_attention_layer_localization.log`). Layer 0 sees
identical inputs, so its `attn_rel` is the pure kernel difference. The probe's
`attn_rel` is max|Δ| / max|ref| (`diff_stats`), the **nmax** metric of §5.1,
not a relative RMS:

| Kernel | layer-0 `attn_rel` (nmax) | layer-0 `hidden_rel` | layer-1 `hidden_rel` | layer-7 `hidden_rel` |
|---|---|---|---|---|
| `TiledF16` (Stage 2 record) | 3.6e-4 | — | — (residual Δ ×40 at layer 1) | — |
| `TiledF16Precise` | 4.15e-4 | 1.59e-4 | 7.93e-3 | 5.98e-2 |

### 2.3 Diagnosis

1. **#552 changed the Q mismatch; it did not remove it.** `TiledF16Precise`
   ran `scalar_mul_f32(Q, log₂e)` and then the F32 → F16 permute, so the kernel
   saw round(q·log₂e), with an exact in-kernel factor of 1.0.
   - Vec-hybrid uses round(q).
   - `TiledF16` uses round(round(q)·1.4423828). Its first rounding matches
     vec-hybrid's; the second adds a biased temperature of 1.000216.
   - The #552 variant replaced that biased error with an unbiased per-element
     rounding mismatch of similar size. Lanes 0 and 3 show it: D went from
     0.099 to 0.510 and from 0.014 to 0.101.

   The experiment therefore does **not** show that the Q pre-scale was minor,
   nor that the residual error is the F16 V/O storage. The split between the
   Q, V and O contributions is unmeasured. §5.1 N1–N3 measure it model-free,
   which makes the cost of F32 V evidence-based and supports diagnosis if
   E1 or E2 fail. `[INFERENCE]` from the rounding map: each of the three
   mismatches (Q rounding, V at 2^-11 relative per element, O at 2^-11 before
   the O projection's own F16 staging) is individually of the same order as
   the layer-0 nmax.
2. **Gemma amplifies any perturbation through MoE routing.** In the
   greedy144 routing witness, the gap between the 8th and 9th expert
   probabilities has median 7.5e-4, p5 5.1e-5 and p1 1.0e-5. 171 of 17,280
   rows (1.0%) fall below 1e-5 and 15 below 1e-6 (Q1 policy §2.2).
   Hidden-state differences grow about 50× from layer 0 to layer 1 and
   saturate near 5–10% by layer 7. A non-bitwise route therefore cannot be
   shown neutral by element metrics. It can only shrink the perturbation
   until routing flips become rare.
3. **What a qualifying primitive must remove.** Every rounding that
   vec-hybrid does not also make:
   - F16 V operand: F32 in vec-hybrid.
   - F16 O store: F32 in vec-hybrid.
   - F16 Q pre-scale: no counterpart in vec-hybrid.

   What may remain is what vec-hybrid shares: the F16 K cache, Q rounded to
   F16 once (round(q), unscaled), F32 arithmetic, and the reduction order. The
   target is a layer-0 kernel difference from vec-hybrid at F32 rounding level:
   nmax ≤ 2e-5 (§8.2), 18× below `TiledF16`'s 3.6e-4 nmax, with the primitive
   bounds of §5.1.
   `[INFERENCE]` The O projection stages its F32 input as `half`
   (`quantized_matmul_mm_tensor.metal`, #552 citation). At that level most
   projection inputs would round to the same half on both routes, so the
   difference reaching the residual would be sparse 1-ulp flips instead of a
   dense 1e-4 field.
4. **This primitive may still fail.** Lane 1's vec margin is 0.167 nat. A
   perturbation smaller than `TiledF16`'s can still flip that lane through a
   thin routing gap. This design therefore treats a §8.3 failure as a
   recorded falsification (§8.5), not a reason to relax the policy.

## 3. Primitive contract (Q4-GEMMA-KERNEL; the text of the linked mlx-native issue)

The acceptance of #659 requires a linked mlx-native issue that states this
contract before implementation. §3–§5 are that text.

### 3.1 Entry points

Two new dispatchers in mlx-native. Existing dispatchers keep their
all-F16/BF16 validation and outputs unchanged.

| Dispatcher | Kernel entry point(s) | Use |
|---|---|---|
| `dispatch_flash_attn_prefill_precise_d256` | `flash_attn_prefill_precise_d256_v32`, `…_v16` (and `…_vcodes` if §3.7 option c is chosen) | sliding layers: external mask plus optional `blk`, or the in-kernel window (§3.4) |
| `dispatch_flash_attn_prefill_precise_d512_resume` | `flash_attn_prefill_precise_d512_vcodes`, `…_v16`, and `…_v32` (parity-only, §5.1 P9; hf2q never dispatches it, which an hf2q unit test asserts) | global layers: causal resume with `q_offset_in_k`. K and V are read **in place** from the slot's global cache with a `kv_capacity` head stride (§3.8). |

The names are proposals. The binding parts are:
- one dispatcher per head-dim family;
- the V input form selects the entry point;
- d512 never takes a dense dequantized V.

V input forms:
- `_v32`: dense F32 V.
- `_v16`: F16 V (exact stored values).
- `_vcodes`: the TQ-HB V codes (U8) and per-position F32 norms, dequantized
  inside the kernel to values **bitwise identical** to `tq_dequantize_hb_kv_seq`
  (`centroid · scale_norm`, with `scale_norm = norm·rsqrt(D)` for D = 256 or
  `norm / scale_factor_d512` for D = 512).

The kernels are template instantiations of the existing `attention<…>` (d256)
and `flash_attn_prefill_d512_impl<…>` (d512) templates. The new template
parameters are `OutT = float`, the V input form and score-domain scaling (d256),
as in the #552 patch. Defaults reproduce the existing instantiations exactly.

### 3.2 Inputs, layouts, shapes

| Operand | dtype | Layout | Notes |
|---|---|---|---|
| Q | F16 | `[B, H, qL, D]` contiguous | Unscaled. Produced by the caller's F32 → F16 permute, the same single rounding vec-hybrid applies. |
| K | F16 | d256: `[B, H_kv, kL, D]` (the compacted sliding stage); d512: the global K cache in place, `[H_kv, kv_capacity, D]` with keys `[0, kL)` linear | The F16 K cache bytes. Never re-rounded. |
| V | d256: F32 (`_v32`), F16 (`_v16`) or codes (`_vcodes`); d512: codes (`_vcodes`) or F16 (`_v16`), in place | as K | `_v32`: TQ-HB V dequantized by `tq_dequantize_hb_kv_seq` (F32 output). `_vcodes`: U8 codes `[H_kv, cap, D]` plus F32 norms `[H_kv, cap, D/256]`, codebook bits ∈ {5, 6, 8}, `scale_factor_d512`. `_v16`: an F16 V cache. |
| mask (d256, optional) | F16 | `[qL, kL]` rank-2 broadcast or `[B, H, qL, kL]` | Values exactly 0 or −∞ (§3.4). |
| blk (d256, optional) | U8 | `ceil(qL/BQ) × ceil(kL/BK)` of the **chosen** d256 tile (§3.7) | Requires the mask. Built from the fully written mask (§3.4). The classifier's `bq`/`bk` must equal the attention tile; a mismatch is `InvalidArgument`. |
| O | **F32** | `[B, H, qL, D]` contiguous | Every element `< qL` written; nothing beyond. |

Parameters:
- `n_heads`, `n_kv_heads` (`n_heads % n_kv_heads == 0`; GQA kv head =
  `h / (n_heads / n_kv_heads)`).
- `head_dim ∈ {256 (d256), 512 (d512)}`; anything else is
  `InvalidArgument`.
- `qL ≥ 1`, `kL ≥ 1`.
- d512: `q_offset_in_k + qL ≤ kL ≤ kv_capacity`.
- `batch ≥ 1` (hf2q uses 1).
- `scale`: finite and > 0.
- `do_causal`.
- No `softcap` parameter. Gemma 4 has no attention-logit soft cap, so the
  dispatcher has no cap input and the kernel applies none.

Gemma 4 26B-A4B served shapes (GGUF `bf65b7e1…`):
- 30 layers. The global layers are 5, 11, 17, 23 and 29; the other 25 are
  sliding.
- `n_heads` 16.
- Sliding layers: D = 256, `n_kv_heads` 8 (GQA 2), window 1024.
- Global layers: D = 512, `n_kv_heads` 2 (GQA 8).
- Warm suffix rows per sequence: 32–4096. Below 32, hf2q keeps vec-hybrid.
- Sliding `kL ≤ (cap − 1) + min(qL, 1024) ≤ 2047`; global `kL ≤ 262144`.

### 3.3 Arithmetic (normative)

1. S = Q·Kᵀ with F16 operands widened to F32 and accumulated in F32
   (simdgroup F32 MMA).
2. Scale and exponential, per kernel:
   - d256: the scale is applied to the F32 scores as S·(scale·log₂e), with the
     factor formed in F32, followed by `exp2`.
   - d512: the scale is applied to the F32 scores (S·scale) followed by natural
     `exp`, as the existing d512 kernel already does (§2.1).

   Both are allowed. In neither kernel is Q multiplied in F16.
3. Mask: excluded keys get −∞. Online softmax: running max, sum and P all in
   F32.
4. P stays F32 into P·V. The V operand is F32:
   - `_v32` reads the F32 value as stored;
   - `_vcodes` dequantizes to the identical F32 value;
   - `_v16` widens the stored F16 exactly.

   No intermediate V, P or O value is rounded below F32.
5. O is accumulated in F32, normalized by the F32 row sum, and stored as F32.
6. Math mode: the same default compile mode as every other attention kernel
   on both metallib paths (§3.6). `requires_precise_fp32_math` is not
   extended. The `exp2`/`exp` precision class is the one vec-hybrid's `exp`
   already uses.

### 3.4 Masks, windows, offsets

Visibility of key `k` to query row `i`, with `q_abs(i) = q_offset + i`:
- Sliding: `k ≤ q_abs(i)` and `q_abs(i) − k < W`. In hf2q `q_offset` is the
  staged history length `min(start, cap − 1)` and W is `sliding_window`.
- Global: `k ≤ q_abs(i)`, with `q_offset = q_offset_in_k = graft + start`.

Rules:
- The mask carries only 0 or −∞. No partial additive bias is supported, and
  the result must not depend on the mask's F16 rounding.
- The d256 entry may take the existing external mask plus `blk`. A function
  constant may instead enable an **in-kernel window predicate** (W,
  `q_offset`), which removes the mask build, the barrier and the `blk` pass.
  §6 decides between them. Both must give bitwise-identical O because the
  predicate is exact.
- `blk` lesson (Stage 2 race, fixed in hf2q `b0f96d86`): the mask
  builder's write is not declared to `GraphSession`. Callers must fence after
  the mask fill before `blk`. The parity tests build their reference from the
  predicate, not from the mask buffer, so a racing `blk` is caught.
- A row with no visible key is a caller error. Warm resumes always see the
  query's own key. The kernel writes 0 for such a row, and that is tested.

### 3.5 Buckets, specializations, determinism

- **No data-dependent buckets.** Vec-hybrid picks its split-K width by key
  length (`compute_nwg`: NWG 16 at `kL ≤ 512`, 32 above). hf2q therefore
  splits vec-hybrid query chunks at positions 512/1024
  (`vec_query_chunk_len`) so a dispatch shares one bucket. The precise kernel
  has fixed tile geometry:
  - d256: one of the §3.7 tiles (BQ 32 / BK 16 / WM 4; BQ 16 / BK 16 / WM 2;
    or BQ 32 / BK 8 / WM 4), fixed per build. `blk` follows the chosen tile.
  - d512: Q 8 × C 64, NSG fixed at 8 for resume.

  Only function-constant specializations vary: `align_Q` (200), `align_K`
  (201), `has_mask` (300), `do_causal` (301), `has_blk` (303), plus the NSG
  constant on d512 and the window predicate constant if adopted. Every
  specialization is a distinct pipeline. Both metallib paths must build
  each one (§3.6).
- **Determinism.** No atomics. Each threadgroup reduces its Q tile over K
  tiles in a fixed order. Identical inputs and shapes give identical bits
  across runs and processes on one metallib path.
- **Sequence independence.** A dispatch reads only its own sequence's
  buffers. hf2q's joint == per-slot bitwise property therefore follows from
  issuing identical per-sequence dispatches, as it does today.
- **Tail safety.** Partial Q/K/V tiles are loaded guarded. NaN or garbage in
  scratch beyond `kL` (K/V) or beyond `qL` (Q) must not reach O, because
  `NaN · 0` is NaN. The d512 V-tail defense is the precedent.
- **Not promised:** bitwise invariance of a row across different chunkings.
  The K-stage tile alignment depends on the staged history length.

### 3.6 Both metallib paths

mlx-native resolves a pipeline in one of two ways:
- **Precompiled** (default). `build.rs` compiles every
  `src/shaders/*.metal` with `xcrun metal -O3` into `default.metallib`.
  Function-constant variants come from the precompiled function when
  `MLX_PRECOMPILED_METALLIB_FCV` is on, which is the default.
- **Runtime source.** `new_library_with_source` with default options. This
  path is taken when `MLX_PRECOMPILED_METALLIB=0`, when a source override is
  registered, when the build used `MLX_NATIVE_SKIP_METALLIB`, or when that
  shader failed to precompile. The last case is silent apart from a cargo
  warning, because the build links the shaders that compiled.

Contract:
- The new entry points are registered in the kernel registry's source map
  under the shader file that defines them. They are compiled into
  `default.metallib` and resolvable on the runtime path.
- `KernelRegistry::pipeline_identity` reports `origin = PrecompiledMetallib`
  in a default build and `RuntimeSource` under `MLX_PRECOMPILED_METALLIB=0`.
  The parity tests assert the origin, so a precompile failure cannot pass
  unnoticed.
- Every test in §5 runs on both paths. Each path must pass the tolerance and
  determinism tests independently. Cross-path bitwise equality of O is
  recorded but not required: the two paths use different compilers, the
  build-time `xcrun` and the OS runtime compiler.

### 3.7 Threadgroup memory budget (d256, ≤ 32 KiB)

Today: `Q_smem` 32 × 264 × 2 B = 16,896 B, plus a shared K/V tile of
max(24 × 256, 16 × 264) halves = 12,288 B, for 29,184 B in total
(`padQ = padK = padV = 16/sizeof(T)`). An F32 V tile of 16 × 260 × 4 B =
16,640 B pushes the total to 33,536 B, which is over the limit. The
implementations that meet the contract:

| Option | How | Cost |
|---|---|---|
| a | Load F32 V fragments directly from device memory into `simdgroup_matrix<float>` | All 4 simdgroups load the same V fragments, so each threadgroup reads F32 V about 4×: roughly 8× the bytes of today's staged F16 V tile |
| b | BQ 16 for `_v32`, with **WM 2** (`static_assert(BQ ≥ kNWarps·8)`, TQ == 1) | `Q_smem` 8,448 B; twice as many threadgroups, half the simdgroups per threadgroup, less K reuse; `blk` grid becomes ceil(qL/16) × ceil(kL/16) |
| BK 8 | BK 8 with an F32 V tile | 16,896 + max(8,192, 8,320) = 25,216 B; twice as many K/V iterations; `blk` grid becomes ceil(qL/32) × ceil(kL/8) |
| c | Stage the TQ-HB V codes (16 × 256 B) and norms in threadgroup memory; dequantize to F32 at fragment load (`_vcodes`) | No dense V buffer; a third V input form |

The contract fixes values, not the implementation:
- options a, b and BK 8 consume the F32 dequant output;
- option c must produce bitwise-identical V values, proven by P9 against
  `tq_dequantize_hb_kv_seq`.

§6 picks the option, and the `blk` geometry follows it. d512 has no budget
issue, because it reads V from device memory.

### 3.8 Transient memory bound

A dense F32 global V costs 4 KiB per key row (2 KV heads × 512 × 4 B). Dense
global K and V at kL = 262,144 would be 1 GiB of V plus 0.5 GiB of K per
chunk. `[INFERENCE]` Scratch buffers live until their command buffer
completes, so 5 global layers × 4 joint sequences could reach tens of GiB.
`TiledF16` already carries the F16 half of this exposure.

The contract therefore bounds it:
- **d512 allocates no K or V scratch.** It reads the F16 K cache and the
  TQ-HB V codes and norms in place (`_vcodes`), or the F16 V cache in place
  (`_v16`), with `kv_capacity` as the head stride. For global layers hf2q's
  live path already treats rows `[0, kv_seq_len)` of the cache as linear;
  that is the compaction `TiledF16` copies today.
  - `[INFERENCE]` The in-place read is therefore value-identical to the
    compacted read.
  - mlx-native P10 (§5.1) proves the in-place stride handling.
  - hf2q A7 (§8.2) proves that the cache rows read in place are bitwise the
    rows `linearize_rows` compacts for `TiledF16` today.
- **d256 dense scratch is bounded by the window.** Sliding `kL ≤ 2,047`, so
  the K/V scratch per sliding chunk is at most 8.4 MB (F16 K) + 16.8 MB (F32
  V), independent of context length.
- **Q and O32 scratch scale with the chunk rows.** A global chunk at
  qL = 4096, D = 512 carries Q 67.1 MB + O32 134.2 MB ≈ 201 MB.

**Frozen TOTAL formula.** It is an upper bound on one warm forward's transient
scratch. `[INFERENCE]` It assumes every chunk's scratch stays alive until the
command buffer completes, because Metal retains buffers referenced by an
encoded command buffer.

Per sliding chunk with `L` query rows and `kL = min(start, cap − 1) + L`
keys:

    S(L, kL) = 4096·kL (K F16) + 8192·kL (V F32; 0 under `_vcodes`)
             + 8192·L (Q F16) + 16384·L (O32)
             + 2·L·kL (mask; 0 with the in-kernel window)
             + max(32, ⌈ceil(L/BQ)·ceil(kL/BK) / 32⌉·32)
               (blk: exactly `blk_buffer_byte_len`,
                `flash_attn_prefill_blk.rs` 0.15.16 lines 188–212;
                0 with the in-kernel window)

Per global chunk with `L` rows:

    G(L) = 16384·L (Q F16) + 32768·L (O32)

The formula then sums over sequences:

    seq   = 25 · Σ_sliding chunks S(L, kL) + 5 · Σ_global chunks G(L)
    TOTAL = Σ over the forward's sequences (joint lanes, or 1 for a
            single-slot resume) of seq

**Scope (frozen): the TOTAL and the logged observed peak cover exactly the
same buffers.** That is the `TiledF32` route's per-chunk scratch:
- sliding: K, V, Q, O32, mask, `blk`;
- global: Q, O32.

The logged "observed peak scratch" is the sum of `byte_len()` of exactly
those buffers as allocated in the warm forward. Both sides exclude the same
route-independent allocations, and code this design leaves unchanged makes
them:
- the sliding ring stage (`(cap − 1 + min(rows, 1024)) × (4096 + 2048 + 32)`
  B per sliding layer per sequence);
- the vec-hybrid slot ids, stage positions and split-K tmp, used for lanes
  under 32 rows;
- the layer's Q/K/V/O stream buffers.

A logged peak above TOTAL therefore means the route allocated a buffer the
formula does not know about, or allocated a larger one. Either blocks the
receipt.

Worked values:
- A full 1,024-row sliding chunk at kL = 2,047 gives S ≈ 54.5 MB.
- A 4,096-row sequence gives about 25 × 4 × 54.5 MB + 5 × 201 MB ≈ 6.5 GB
  with the external mask.

**Against `TiledF16`, per chunk** (same L and kL, same mask and `blk`).
`TiledF16` allocates:
- sliding: K F16 4096·kL, V F16 4096·kL, Q 8192·L, O16 8192·L, O32 16384·L;
- global (2 KV heads × 512): dense K F16 2048·kL, V F16 2048·kL, Q 16384·L,
  O16 16384·L, O32 32768·L.

Hence `TiledF32 − TiledF16`:

| Layer | Difference | When `TiledF32` is larger |
|---|---|---|
| Sliding (dense F32 V) | **+4096·kL − 8192·L** | when kL > 2L |
| Sliding (`_vcodes`) | −4096·kL − 8192·L | never |
| Global | **−(4096·kL + 16384·L)**: dense K/V and O16 are gone | never |

Sliding kL ≤ 2,047, so the sliding excess is at most 8.4 MB per chunk. The
global saving grows with context.

The review's closure note gave the global saving as 4096·kL + 8192·L. The
O16 term is 16 heads × 512 × 2 B = 16384·L per row, so the figure above
uses 16384·L.

Checks:
- The warm forward logs its observed peak, scoped as above, and the TOTAL
  computed from its own logged shapes (§8.1 observability).
- The long-context Q5 receipt (§9.2) passes only if every logged peak is
  ≤ its TOTAL.
- Memory admission itself stays Q5's job.

## 4. Starting point and branch hygiene

- Rebase the #552 patch (written against 0.15.15, uncompiled) onto mlx-native
  0.15.16, which added only routing and RMS-norm race fixes. Work in a new
  worktree under `/opt/mlx-native-worktrees/`.
- Keep from the patch: `OutT`, `ScaleScores` (scale·log₂e applied to F32
  scores, no Q pre-scale) and the d512 `OutT` store.
- Change against the patch:
  - add the V input forms of §3.1 (`_v32`/`_v16`/`_vcodes` on d256,
    `_vcodes`/`_v16` on d512) to both templates;
  - d512 reads K and V in place with the `kv_capacity` stride (§3.8);
  - add the separate precise dispatchers (§3.1) instead of switching
    existing dispatchers on `out.dtype()`;
  - implement the §3.7 option chosen in §6, with `blk` geometry following the
    tile;
  - add the optional in-kernel window predicate.
- Branch `exp/gemma-tiled-precise` is retained until the kernel lands or is
  falsified (ADR-063 Q4-GEMMA-KERNEL).

## 5. Primitive verification (mlx-native; model-free; GPU)

`tests/test_flash_attn_prefill_precise.rs`, run on both metallib paths (§3.6).
Inputs are seeded: Q and K are RMS-normalized like Gemma's, and V comes from
real TQ-HB encoding of random rows through
`dispatch_hadamard_quantize_kv_hb_seq` (the encoder hf2q's `write_kv_rows`
uses), so the codes and norms are realistic. Shapes are the §6 grid plus edge
cases:
- `qL ∈ {1, 31, 32, 33, 1024}`;
- `kL` not a multiple of 16 or 64;
- window boundaries falling mid-tile;
- fully masked tiles;
- `q_offset > 0`;
- GQA 2 and GQA 8;
- batch 2 once.

### 5.1 Tolerances (normative, frozen here)

The reference is CPU F64 attention over the kernel's exact inputs: the F16 Q
and K values, the V values as given, scale, and the visibility predicate.
Metrics per (shape, layer type) over all O elements: `rel_rms = rms(O −
ref)/rms(ref)` and `nmax = max|O − ref| / max|ref|`.

| ID | Check | Bound |
|---|---|---|
| P1 | precise `_v32`/`_vcodes` (d256) and `_vcodes` (d512) against reference | `rel_rms ≤ 1e-5` and `nmax ≤ 5e-5` |
| P1′ | precise `_v16` against reference (F16 V cache inputs) | same bounds |
| P2 | precise against vec-hybrid accuracy | `rel_rms(precise) ≤ max(2·rel_rms(vec), 2e-6)`, with vec measured against the same reference |
| P3 | direct route parity | `rel_rms(precise − vec) ≤ 2e-5` and `nmax(precise − vec) ≤ 2e-5` on the same TQ-HB codes and F16 K |
| C+ | positive control | vec-hybrid passes P1 |
| C− | negative control | the `TiledF16` chain (F16 Q pre-scale, F16 V, F16 O) **fails** P1 and P3 at served shapes, so the test discriminates. `[INFERENCE]` Expected nmax near 1e-4–4e-4, the layer-0 `attn_rel` range. |
| N1 | negative control, Q mismatch only | the precise kernel fed Q = round(q·log₂e) with scale = ln 2 (the #552 input) **fails** P3; its rel_rms and nmax against vec are recorded |
| N2 | negative control, V only | the precise kernel `_v16` fed `tq_dequantize_hb_kv_seq_f16` output in place of the F32 dequant **fails** P1 against the F32-V reference; recorded |
| N3 | negative control, O only | the precise `_v32`/`_vcodes` output rounded to F16 and widened back **fails** P1; recorded |
| P4 | determinism | 20 repeats bitwise equal, per specialization and per path |
| P5 | tail poison | NaN-filled K/V rows beyond `kL` and Q rows beyond `qL` give bitwise the clean-run O |
| P6 | masks | external mask plus `blk` and the in-kernel predicate (if adopted) are bitwise equal, and both match the predicate reference |
| P7 | no regression | every existing `flash_attn_prefill*` entry point gives bitwise its 0.15.16 output on fixed seeds (golden digests recorded before the change) |
| P8 | origin | `pipeline_identity(...).origin` asserted per path; cross-path O digest equality recorded |
| P9 | `_vcodes` values | method: the `_vcodes` O is bitwise equal to the `_v32` O of the same template (d256 `_v32`; d512 parity-only `_v32`), fed `tq_dequantize_hb_kv_seq` output for the same codes and norms, on both paths. Both entries share all arithmetic except the V load, so O equality proves the in-kernel V values equal the dequant output. |
| P10 | d512 in place | `_vcodes` with `kv_capacity > kL` gives bitwise the O of `_vcodes` on a compacted copy (`kv_capacity = kL`). Rows `[kL, kv_capacity)` are poisoned: K F16 = NaN, V norms = NaN, V codes = 0xFF. |

N1–N3 are diagnostic. Each must fail its bound, which proves that the
mismatch alone is detectable at served shapes. Their recorded magnitudes split
`TiledF16`'s error into Q, V and O parts, which #552 could not do (§2.3.1).
They also make the cost of requiring F32 V evidence-based. If N2 turns out to
be negligible beside N1 and N3, that is recorded, but the F32 V requirement
stands: vec-hybrid's V is F32.

**An unexpected pass of N1, N2 or N3 blocks §5** until it is explained in
the receipt. A pass means the test cannot detect that mismatch at that shape,
so the bounds or shapes are suspect. Bounds are not loosened to explain it
(Q1 §3.7).

Bound rationale `[INFERENCE]`: the F32 error of a 256- or 512-term dot
product plus a ≤ 2,048-key online softmax sits near √n · 2⁻²⁴ ≈ 3e-6
relative. The P1 nmax bound of 5e-5 is 7× below `TiledF16`'s measured layer-0
nmax (3.6e-4), and the P3/§8.2 nmax bound of 2e-5 is 18× below it. The two
metrics are not compared with each other: rel_rms bounds apply only to
rel_rms, and nmax bounds to nmax.

## 6. Microbenchmark plan at served shapes

Bench: `benches/bench_flash_attn_warm_precise.rs` (mlx-native, harness
false). It loads no model, but it needs the root's GPU lane and a quiet host
(README-agents).

**Shapes.** Single-sequence cells (batch 1, Gemma heads per §3.2) for each
(history H, suffix S):
- **Canonical run lanes**, from `$E/gemma-tiled-off-smoke-5d421a2c/server.stderr`
  (cached → prompt per turn): (458, 58), (516, 191), (707, 220), (927, 288),
  (1215, 150), (1379, 348).
- **Gate lanes:** (465, 56), (714, 200), (943, 311), (1254, 385).
- **Long-context stress** (the D2 long-context workload variant, PR #551):
  (4096, 1024), (8192, 4096), (32768, 4096).

**Four-lane back-to-back cells (the gating cells).** For each joint warm
transaction of the canonical run, the four lanes' (H, S) pairs are read from
the same `server.stderr`. That run has six joint transactions:
- the five joint ones: `seq_len` 232, 764, 880, 1152 and 528 rows over four
  lanes;
- the final turn's per-slot 348-row resumes, encoded as four consecutive
  single lanes.

The per-lane split comes from the `slot admission planned … cached_tokens`
and `bounded prefill transaction complete … advanced_tokens` lines; the
extraction script and its output are part of the receipt.

Each cell encodes the four lanes' per-layer chains **back to back in one
command buffer**, the way `encode_live_layer_attention` issues a joint cohort.
Small suffixes underfill tiled d256: 56 rows give 2 Q tiles × 16 heads = 32
threadgroups, while vec-hybrid launches one threadgroup per (query, head) ×
NWG. Only these cells capture that occupancy effect.

Sliding layers stage `min(H, 1023)` history rows plus the chunk. Global
layers attend H + S.

**Arms.** Each arm is timed as the full per-chunk chain hf2q encodes, and
separately as the attention kernel alone:

| Arm | Chain |
|---|---|
| `VecHybrid` | `flash_attn_vec_hybrid_batched` over F16 K and TQ-HB codes with its reduce, 256-query chunks split at 512/1024 |
| `TiledF16` | K compaction, `tq_dequantize_hb_kv_seq_f16`, Q permute, mask, fence, `blk`, `…_f16_d256_with_blk` / `…_f16_d512_resume`, O cast, O permute |
| `Precise` (per §3.7 option, ± in-kernel window) | d256: K compaction, F32 V dequant (none for `_vcodes`), Q permute, [mask, fence, `blk`], precise dispatch, O permute (no cast). d512: Q permute, precise in-place `_vcodes` dispatch (no compaction, no dequant), O permute. |

**Method.**
- Timing comes from GPU timestamps (the mlx-native measurement API, #74).
- Per cell: 10 warmup command buffers, then 50 measured. Report the median
  and p90.
- Arms are interleaved A B C C B A within each cell.
- Each cell is measured on both metallib paths. Precompiled is the
  production number.
- Every receipt records the device, the macOS build, the mlx-native commit
  and version, the metallib SHA-256, the pipeline origins and the bench
  binary SHA-256.
- Evidence goes to `~/.local/state/hf2q/v0130/q4-gemma-kernel/`.

**Model-level projection** `[INFERENCE, not a receipt]`. For each arm, sum
the four-lane cells over the six transactions: 25 × sliding chain + 5 ×
global chain per transaction. Compare with the Stage 2 fit, in which
vec-hybrid accounted for about 1.91 s of the 3.43 s of joint warm forward per
run.

**Go / no-go for hf2q integration** (frozen):
- M1: on the **four-lane back-to-back cells**, the precise total (all six
  transactions, 25 sliding + 5 global layers each) is ≤ 0.5 × the VecHybrid
  total. Otherwise the speed basis is falsified and recorded. The
  single-sequence cells are reported but do not gate.
- M2 (reported, not gated): the precise/`TiledF16` ratio on the same cells.
  Above 1.5 is flagged, because the projected saving shrinks.
- M3 (reported): long-context stress cells, with their peak transient
  allocation against the §3.8 TOTAL for the same shapes.
- The §3.7 option and the window-predicate choice: the fastest variant that
  passes §5.1, with ties broken by the simpler code.

## 7. Release (Q4-GEMMA-KERNEL delivery boundary)

mlx-native PR with the contract issue linked; the §5 receipts on both paths;
§6 receipts; independent review; merge; publish (the next 0.15.x);
`cargo info`/crates.io checksum verification; then hf2q pins
`mlx-native = "=<that version>"` (exact) in Q4-GEMMA-ATTN. No hf2q work
consumes a path dependency.

## 8. hf2q integration (Q4-GEMMA-ATTN)

### 8.1 Route selection, switch, kill switch

`GemmaLiveKernel` gains `TiledF32`:
- `src/inference/models/gemma4/model.rs`: the new variant; `is_tiled`
  includes it.
- `src/serve/gemma_live_attention.rs`:
  - `LiveRoute::seq_kernel`: `TiledF32` when `rows ≥ WARM_TILED_MIN_ROWS` (32),
    otherwise `VecHybrid`, as for `TiledF16`.
  - `chunk_len` and `stage_query_rows`: the tiled values (sliding 1,024 and
    global 4,096 query chunks; ring stage `cap − 1 + min(rows, 1024)`).
  - `label`: `tiled-f32 (vec-hybrid below 32 rows)`.
- Installed long-prompt lanes (`GemmaLiveAttention::Tiled`,
  `HF2Q_GEMMA_TILED_LIVE`) are unchanged. Moving them, or the cold prefill's
  d256 Q pre-scale (#552 secondary finding), is out of scope and needs its
  own Class B design.

`encode_tiled_chunk` takes the kernel and, for `TiledF32`:
- sliding (d256) layers: compacts K as today. V is dequantized by
  `dispatch_tq_dequantize_hb_kv_seq` into an F32 dense-V scratch (bounded by
  the window, §3.8), or linearized for an F16 V cache (`_v16`), or not
  dequantized at all under §3.7 option c (`_vcodes`);
- global (d512) layers: **no K compaction and no V scratch**. The dispatch
  reads the slot's global F16 K cache and TQ-HB V codes and norms (or F16 V
  cache) in place with `kv_capacity = src_capacity` (§3.8). Because nothing
  is compacted, the dispatch itself must declare its reads. Before the d512
  dispatch, `s.barrier_between(&[k_cache, v_codes, v_norms (or v_f16),
  &q_f16], &[&out_f32_head])` orders it after this layer's `write_kv_rows`
  suffix append on the concurrent encoder. `TiledF16`'s compaction got that
  ordering from its own declared reads. An hf2q unit test asserts the
  declaration, and A1/A2 cover it on hardware;
- keeps the Q F32 → F16 permute;
- for sliding layers, builds the mask, fences and builds `blk`, or uses the
  in-kernel window, per §6;
- writes the precise dispatch's F32 O straight into the head-major O32
  scratch, then the existing `permute_021_f32` writes it into the layer
  output. The O16 scratch and the F16 → F32 cast are removed on this route.

`HF2Q_GEMMA_WARM_TILED_ATTENTION`, resolved once per model load (unchanged
mechanism):

| Value | Before promotion | After promotion (§9.4) |
|---|---|---|
| unset | `VecHybrid` | `TiledF32` |
| `0`/`off`/`false` | `VecHybrid` (kill switch) | `VecHybrid` (kill switch) |
| `1`/`on`/`true` | `TiledF16` (measured baseline arm) | `TiledF32` ("tiled on" keeps its meaning; the warm `TiledF16` path is deleted, and `TiledF16` remains only for installed lanes) |
| `precise` | `TiledF32` | `TiledF32` |
| anything else | warning, then the default | warning, then the default |

**Fail closed.** When the resolved warm kernel is `TiledF32`, model load
prewarms every precise pipeline specialization the route can issue. Model
load then fails with the mlx-native error if any of them cannot be built. A
silent fall-back to another kernel is not allowed.

**Observability.**
- Startup log: `Gemma warm live attention kernel kernel=TiledF32`.
- Prefill banner: `live=tiled-f32 (vec-hybrid below 32 rows)`.
- New `active_controls.gemma_warm_attention_kernel` in `/hf2q/v1/runtime`
  (`src/serve/api/measurement.rs`), so Q5 receipts bind the arm.
- The Q5 `gemma4` case gains a readiness check of that field per arm.
- The warm forward logs its observed peak transient scratch bytes and the
  §3.8 TOTAL computed from its own shapes. The long-context Q5 receipt
  (§9.2) requires every logged peak ≤ its TOTAL.

Surfaces updated in the same change:
- `scripts/serve_gemma4_opencode.sh` accepts `0|1|precise`.
- `scripts/four_agent/q5_cases/gemma4.json` notes: the stale "default (on)"
  wording is corrected.
- `test_q5_calibration.py`.
- `docs/operator-env-vars.md`, `docs/shipping-contract.md`,
  `docs/env-var-inventory.md` and CHANGELOG.
- `src/quality/adr063_invariance/gemma_actual.rs`: `WarmLiveKernel` gains
  `TiledF32`; `require_warm_kernel` maps it. `gemma_controls` pins the
  default.

### 8.2 Class A gates (bitwise; any byte fails)

All of these are real-artifact, ignored tests. Run one process each, with
`--exact --ignored --test-threads=1`.

| Gate | Test (existing harness, new route arm) | Required |
|---|---|---|
| A1 joint == per-slot | `gemma_joint_warm_prefill_matches_per_slot_bitwise_tiled_f32`, with cohorts {0..3} ×5, {0,3}, {1,2,3} ×2, {3,2,1,0}, {2,3} | first token, logits bits, KV bytes, cursors equal |
| A2 repeat determinism | `gemma_warm_tiled_attention_is_deterministic` on `TiledF32`, run once in a default build and once under `MLX_PRECOMPILED_METALLIB=0` | bitwise within each path |
| A3 tiny suffix | suffixes < 32 rows on `TiledF32` equal vec-hybrid | bitwise |
| A4 vec control | the `VecHybrid` rerun inside the envelope gate equals the reference | bitwise (already asserted) |
| A5 unchanged routes | unit tests: installed lanes ignore the warm switch; single slot and joint resolve the same route for all kernels and rows | pass |
| A6 transcripts | Q5 A/B arms of the same kernel: `q1_decision_policy.py transcripts` across repeats | `ExactTranscripts` |
| A7 in-place global read | GPU fixture on hf2q's global cache layout (graft 0 and graft > 0): the rows `[0, kv_seq_len)` that `TiledF32` reads in place are bitwise equal to `linearize_rows`'s compacted K and V codes/norms | bitwise |

**Localization bound** (frozen), checked before the envelope gate:
`gemma_warm_tiled_attention_layer_localization` on `TiledF32` (lane 1,
history 714 / suffix 200) must show layer-0 `attn_rel` **nmax ≤ 2e-5**. The
probe's `attn_rel` is max|Δ| / max|ref| (`diff_stats`), so the bound is in the
same metric as `TiledF16`'s 3.6e-4. The probe additionally records layer-0
rel_rms, descriptive only. Layer 0 is the pure kernel difference, so this
shows the §5 precision survives integration. A miss blocks §8.3 and goes back
to §5.

### 8.3 Q1 decision envelope

Class B against `VecHybrid`, policy `hf2q.q1.numeric-policy.v1`, thresholds
unchanged.

**E1, route envelope gate** (`gemma_warm_attention_route_envelope`, 4 lanes,
arms `VecHybrid` control, `TiledF16` baseline, `TiledF32`).
- **Row coverage (frozen): every suffix row of every lane** — 56 + 200 +
  311 + 385 = 952 rows. Q1 §3.2: never a subset. The last-row-only check of
  the current gate is extended to all rows; the gate fails if the extended
  capture does not cover all 952 rows.
- The `TiledF32` arm must have no near-tie-policy failure on any of those
  rows. A failure is a top-1 change where the vec margin is ≥ 0.10 nat.
- The gate reports the Q1 D for every row. Recording it needs an extension of
  the gate: `L[c]`, `R[c]` and the route top-1 log-prob.
- Per lane it reports max D, top-1 agreement and the number of rows with
  D > 0.25 (`RouteSensitive` rows). These are recorded, not gated.

**E2, canonical divergence coordinate.** Serving replay of the
`gemma-divergence-v1` agent-0 and agent-3 turn-2 requests:
- the same text-model hard link and harness;
- solo, and the four-agent concurrent trial ×3;
- arms `0`, `1` and `precise`.

Rule (frozen):
- **Prefix condition.** Every generated token of the `precise` arm before
  the decision coordinate must equal the `0` arm's.
- **Earlier divergence.** If the `precise` arm diverges before the canonical
  coordinate, the replay stops at its first differing token. That row is a
  **defined non-pass outcome**, never a pass:
  - **Fail** if the `0` arm's margin at that row is ≥ 0.10 nat or D > 0.25.
    Disposal: the route is falsified (§8.5).
  - **Inconclusive** otherwise (a near-tie row, D ≤ 0.25). This goes to the
    operator as `needs_decision`. Disposal: as for an E2(c)-only miss
    (§8.5). The operator weighs D at that row, E1 and §9.1.
  - The canonical coordinate is reported as not reached in both cases.
- **At the decision row**, with D computed against the `0` arm, `precise`
  must have:
  - (a) the same raw top-1;
  - (b) the same chosen token after the 1.05 repetition penalty;
  - (c) D ≤ 0.25 nat (`WithinDecisionEnvelope`).

The vec arm's margin at the canonical coordinate is 1.829 nat, so that row is
not a near-tie. The logs must show `kernel=TiledF32` and the `live=tiled-f32`
banner.

**E3, greedy144 classification.** The warm kernel runs only on the 64-row
resumes of T1, T3 and T6. T0 is a cold 128-token prefill with nothing cached,
so a T0 comparison would be a vacuous `ExactEqual`. E3 therefore compares:
- the Q1 Gemma endpoint **T3** (cold 64 + warm 64) with
  `warm_live_kernel = TiledF32` against T3 on `VecHybrid`;
- **T1** (resume at 64) on both kernels.

Both use the accepted dataset `842768` and `q1_decision_policy.py envelope`.
Each result is classified and recorded. A `RouteSensitive` result must carry
the localizer's first divergence (`gemma_localizer`, `[64,64]` route).

**Disposition rule.**
- E1 and E2 gate.
- E3 is recorded. Under Q1 §3.2, `RouteSensitive` forbids any "numerically
  neutral" claim and makes the paired quality campaign mandatory. That
  campaign is required here in any case.
- Context only, never a gate: hf2q's own accepted routes span margins
  1.46–1.83 nat at this coordinate (cold 1.464, warm vec 1.829). Their
  cold-versus-warm D is 0.365 nat, which shows the coordinate is
  route-sensitive, not that τ is wrong. τ stays 0.25 (Q1 §3.7).
- **E2(c)-only miss.** If (a), (b), E1 and the prefix condition hold but
  (c) misses, the result goes to the operator as `needs_decision`; the
  threshold is not changed. The operator weighs, as declared now:
  1. (a) and (b) at every E2 replay;
  2. E1's full-row result;
  3. the §9.1 paired quality verdict. §9.1 runs in this case so that the
     decision has it.

### 8.4 Model-free checks

`cargo test --locked --bin hf2q gemma_live_attention` covers:
- the switch table in §8.1;
- route resolution for every kernel and row count;
- chunking;
- stage sizing.

It also runs the Q5 calibration unit tests, the env-surface audit
(`scripts/audit_hf2q_env_surface.sh`) and the Q1 oracle controls
(`adr063_invariance`).

### 8.5 Falsification outcomes

- **Fails E1, the localization bound, E2 (a)/(b), or an earlier-divergence
  Fail:** the route is falsified for default use. Record the receipts. The
  kernel may still ship in mlx-native. **hf2q removes the `precise` value and
  the `TiledF32` variant** in a clean cutover. The `1` arm (`TiledF16`)
  already provides the speed A/B surface. Gemma's Win then has to come
  entirely from the other profiled stages (§10).
- **E2(c)-only miss or earlier-divergence Inconclusive:** `precise` stays
  opt-in, default `VecHybrid`, until the operator rules on the
  `needs_decision` (§8.3). It is then promoted or removed per the ruling.
- **Passes E1/E2 but fails M1 or the §9.2 speed gate:** the speed basis is
  falsified. The default stays `VecHybrid`.

## 9. Campaigns (root, GPU lane, one model process at a time)

Every run records:
- the source commit;
- the binary SHA-256;
- the artifact digest (`bf65b7e1…`);
- the mlx-native version and metallib SHA-256;
- the case file and its SHA-256.

Every run passes Q5 `CalibrationPass` with the accepted negative-control
receipt. Evidence goes to `~/.local/state/hf2q/v0130/q4-gemma-attn/`.

### 9.1 Paired task-quality campaign (Q1 §3.5)

Same binary in both arms:
- subject: `--subject-var GEMMA_WARM_TILED_ATTENTION=precise`;
- reference: `--reference-kind stock-hf2q --reference-var GEMMA_WARM_TILED_ATTENTION=0`.

Settings: 24 instances (`fixtures/coding-instances.json`, 6 labels × 4
agents), ABBA order, no reruns.

```sh
python3 scripts/four_agent/quality_campaign.py \
  --subject-case scripts/four_agent/q5_cases/gemma4.json \
  --subject-var HF2Q_BIN=$BIN --subject-var MODEL=$GEMMA --subject-var GEMMA_WARM_TILED_ATTENTION=precise \
  --subject-var CLIENT_APPROVAL_TEMPLATE=$T/gemma4-client-approval.json \
  --reference-kind stock-hf2q --reference-case scripts/four_agent/q5_cases/gemma4.json \
  --reference-var HF2Q_BIN=$BIN --reference-var MODEL=$GEMMA --reference-var GEMMA_WARM_TILED_ATTENTION=0 \
  --reference-var CLIENT_APPROVAL_TEMPLATE=$T/gemma4-client-approval.json \
  --negative-control-receipt "$NC" --output ~/.local/state/hf2q/v0130/q4-gemma-attn/quality-precise-vs-vec
```

Required verdict: `Pass`. `Loss`, `BelowFloor` or `Inconclusive` blocks
promotion. Passing patches also need the independent patch review. The Q4
closing gates (§9.3) repeat the quality campaign on the combined candidate,
against v0.1.28 and against the peer.

### 9.2 Speed A/B of the route (Q5, same binary)

Three arms, `GEMMA_WARM_TILED_ATTENTION ∈ {0, 1, precise}`. ABC/CBA order
with a thermal gap, at least 3 runs per arm, the canonical four-agent coding
workload:

```sh
for ARM in 0 precise 1 1 precise 0; do
  HF2Q_Q5_GEMMA_WARM_TILED_ATTENTION=$ARM python3 scripts/four_agent/q5_calibration.py \
    scripts/four_agent/q5_cases/gemma4.json --negative-control-receipt "$NC" \
    --output ~/.local/state/hf2q/v0130/q4-gemma-attn/q5-ab-$ARM-$(date -u +%Y%m%dT%H%M%SZ)
done   # repeat the block until ≥ 3 per arm
```

Read from each run:
- client makespan;
- per-turn joint warm transaction `elapsed_ms`;
- the banner and runtime-control kernel;
- worst-slot TTFT p95 and SSE-gap max.

The arms generate different transcripts, so later turns' prompts differ.
The primary diagnostic is therefore joint warm `elapsed_ms` on **matched
transactions**: transactions whose per-lane (cached, advanced) token counts
are identical across arms, which always includes turn 1 and every turn
before the first transcript divergence. Unmatched transactions are reported
separately.

Frozen gate:
- The median `precise` makespan is ≤ 0.95 × the median of the `0` arm.
- It recovers at least 60% of the `1` arm's saving. For reference, Stage 2
  measured `TiledF16` at −7.8% against vec-hybrid.
- A6 holds within each arm.

**Long-context receipt.** One Q5 run on the `precise` arm with the D2
long-context workload variant (PR #551). It uses a new
`q5_cases/gemma4-long-context.json` that mirrors
`q5_cases/qwen38-long-context.json`. The run must:
- pass `CalibrationPass`;
- complete the task;
- log, for every warm forward, a peak transient scratch ≤ the §3.8 TOTAL
  computed from that forward's logged shapes.

A failure blocks promotion.

### 9.3 Four-agent hf2q-versus-peer campaign (Q4 closure; Q3 design)

Run only on the **combined candidate**: this route plus the further changes
the Q4-GEMMA profile requires, each landed under its own design. Run it
against the peer's best valid configuration from #660:

```sh
python3 scripts/four_agent/q3_campaign.py --family gemma4 \
  --stock-bin $V0128 --stock-commit $V0128_SHA \
  --experiment-bin $CAND --experiment-commit $CAND_SHA --experiment-label "Q4-GEMMA-ATTN candidate" \
  --peer-case $PEER_BEST_CASE \
  --quality-verdict $QUAL_V0128/manifest.json --quality-verdict $QUAL_PEER/manifest.json \
  --patch-review $PATCH_REVIEW \
  --negative-control-receipt "$NC" --qe-identity "$QE" --output $OUT
# PEER_BEST_CASE: q5_cases/peer-gemma4.json or the best-valid case #660 establishes.
# Before promotion the experiment arm also needs --experiment-var GEMMA_WARM_TILED_ATTENTION=precise.
```

Settings: n4 primary plus n1 control, Williams blocks (5–20).

Win criteria, all from ADR-063 Q4:
- `family_verdict == Win`: the hf2q/peer makespan ratio has a 95% CI upper
  bound ≤ 0.95.
- Worst-slot TTFT p95 and SSE-gap max against v0.1.28 stay within 1.025.
  The SSE gap failed at 1.047 in v9, on the first tool-call argument burst.
  That stage belongs to the profile, not to attention.
- Paired quality `Pass` against v0.1.28 and against the peer, on the exact
  candidate binary.
- A fresh Q5 on the release candidate.

Sequencing: §9.1 and §9.2 decide the route. §9.3 runs once per release
candidate, never per experiment.

### 9.4 Promotion

`TiledF32` becomes the default (§8.1 table) only when all of these hold:
- §5 passed and the version is published and pinned;
- A1–A7 and the localization bound passed;
- E1 and E2 passed (or an E2(c)-only or earlier-divergence Inconclusive
  `needs_decision` ruled in favour), and
  E3 is recorded;
- the long-context receipt (§9.2) passed;
- §9.1 is `Pass`;
- the §9.2 speed gate passed.

The promotion commit:
- flips the default;
- deletes the warm `TiledF16` arm;
- keeps `0` as the kill switch;
- updates every surface listed in §8.1;
- records the receipts in this document and in ADR-063 Q4-GEMMA-ATTN.

If any gate fails, the default stays `VecHybrid` (§8.5).

## 10. How this addresses the profiled stages

### 10.1 Size of the attention stage against the gap

Q3 v9 numbers:
- The peer (speculation off) finishes 13.4% sooner (CI 10.6–16.2%). hf2q's
  makespan is 1.155× the peer's.
- The peer's per-output-token rate is 18.3% higher.
- hf2q's worst-slot TTFT p95 is 1.31× the peer's.
- The stock SSE-gap gate fails at 1.047×.

Turning warm tiled attention off cost 7.8–9.2% (v9 review), and v7 with tiled
on was still a Loss (1.058).

`[INFERENCE]` Two projections:
- **Kernel alone.** Recovering about 8% would leave the ratio near
  1.155 × 0.92 ≈ 1.06, still a Loss. That is consistent with v7.
- **Required for a Win.** A CI upper bound ≤ 0.95 with a half-width of about
  3 points needs a point ratio near 0.92. That is about a 20% total makespan
  reduction against the v9 hf2q. The peer's best valid configuration (#660)
  may widen the gap.

The remaining reduction has to come from the stages #660 attributes: MoE and
decode rate, dispatch/encode, scheduling and SSE cadence.

**Reviewer note (recorded).** The kernel alone cannot win:
- About 8% is a fair upper bound. `TiledF16` saved 1.16 s of 14.34 s
  (makespan medians 13.22 s against 14.34 s, Stage 2).
- `[INFERENCE]` The extra F32 V traffic costs well under 1% of makespan.
  M1 at its 0.5× limit projects about 6.7%.
- v9's n1 control shows the same gap: the peer is 13.5% sooner with a single
  agent. Most of the gap is therefore single-stream decode, MoE and prefill
  throughput, which is #660's job.
- The stock SSE-gap failure (1.047×, first tool-call argument burst) is a
  separate blocker that attention does not touch.

**Alternative for #668: batched-query vec-hybrid spike.** A vec-hybrid kernel
that processes several queries per threadgroup and shares K/V staging across
them, while keeping each query's arithmetic and reduction order exactly as
today, could be **Class A** (bitwise against `VecHybrid`).
- If it is, it removes the envelope risk entirely, because E1–E3 and the
  §9.1 campaign reduce to `ExactTranscripts` (Q1 §3.6). The envelope risk is
  real: in the Q1 data, rows bitwise equal through layer 5 still reached
  0.46% relative RMS at layer 6.
- It is a spike, not a condition of this design. It needs its own design
  before implementation.
- Its deciding measurement is the §6 four-lane cells: speed against
  VecHybrid, plus a bitwise parity test against VecHybrid.
- If it reaches M1 bitwise, it supersedes `TiledF32` for #668. Otherwise it
  is recorded as falsified.

### 10.2 Stage ownership

| Profiled stage (#660 taxonomy) | Owner |
|---|---|
| Warm-prefill attention (joint warm and single-slot resumes, sliding and global layers) | this design: the `TiledF32` route |
| Cold turn-0 prefill attention (d256 Q pre-scale in F16/BF16) | out of scope; its own Class B design if #660 attributes time or numerics to it |
| Decode per-output-token rate, MoE, dispatch/encode | further Gemma changes, each with its own design (ADR-063 Q4-GEMMA-ATTN acceptance) |
| SSE cadence (stock SSE-gap max 1.047) | its own design. The §9.3 gate applies to the combined candidate. |

When #660 lands, this document gains an appendix. For each stage it gives
#660's attributed seconds, the share §9.2 recovers on that stage, and the
design that owns the rest. That appendix is a precondition for running §9.3.
It does not change the route design.

## 11. Recorded decisions (reviewer recommendations, ReviewGemmaKernelDesign)

1. **Falsified route:** if E1, the localization bound, E2 (a)/(b) or an
   earlier-divergence Fail falsify `TiledF32`, the `precise` value and the
   variant are removed (clean
   cutover). The `1` arm keeps the speed A/B surface. The exceptions are an
   E2(c)-only miss or an earlier-divergence Inconclusive (`needs_decision`):
   `precise` stays opt-in until the operator rules (§8.5).
2. **τ stays 0.25 nat.** Q1 §3.7 forbids changing tie thresholds or learning
   tolerances. The 0.365-nat cold-versus-warm gap shows the coordinate is
   route-sensitive, not that τ is wrong. What the operator weighs in an
   E2(c)-only case is predeclared in §8.3: (a), (b), E1 and the §9.1 verdict.
3. **Cold prefill stays out of scope.** Both routes resume from the same cold
   histories, and changing cold prefill perturbs every history, so it needs
   its own Class B design and quality campaign. Noted for #660: the BF16
   multi-sequence cold path has a larger temperature error (0.9982, from
   bf16(log₂e) = 1.4453125) than the F16 path (1.000216). The precise d256
   entry point would make that fix cheap.

## 12. Outcome (2026-11-05): `TiledF32` falsified

The route was implemented as designed on hf2q PR #698. The design's §8.1 text
needed one delta: under mlx-native #83 the sliding layers also read the ring
stage in place. It was gated on the M5 Max at head `bb2e50ea` with mlx-native
`=0.15.18` (crate sha256 `9d10f97e…c524bee`), test binary sha256
`822add9a…`, macOS `26B5112a`. Evidence:
`~/.local/state/hf2q/v0130/q4-gemma-attn/gates-bb2e50ea/`.

| Gate | Result |
|---|---|
| G1 Metal fixtures: declared in-place reads, mask fence, A7 (sliding stage `[8, 1023+min(rows,1024), 256]` and global cache, graft 0 and 37) | pass on both metallib paths (`GEMMA_TILED_F32_A7_COMPLETE … bitwise=true`) |
| A1 joint == per-slot (§8.2 cohorts) | bitwise pass; 18/18 §3.8 scratch lines `within_total=true` |
| A2 determinism | bitwise pass; precise pipeline origins `PrecompiledMetallib`×16 on the precompiled path, `RuntimeSource`×16 on the runtime path |
| Localization bound (lane 1) | layer-0 `attn_rel` nmax **4.95e-7** ≤ 2e-5 (rel_rms 3.35e-7); `TiledF16` 3.63e-4 in the same run |
| E1 (952 rows, `VecHybrid` reference) | **fail**: 72 near-tie-policy failures |
| A4 vec control | bitwise pass |
| A3 tiny suffix | not reached: a harness bug stopped the run, fixed in `d1cc5503` |

E1 detail (`E1-summary-from-rows.txt`, from the 952 rows per route logged
before the stop):

| Route | flips | near-tie failures (vec margin ≥ 0.10) | rows D > 0.25 | median D | max D |
|---|---|---|---|---|---|
| `TiledF32` | 121 | 72 | 352 | 0.168 | 4.42 |
| `TiledF16` | 135 | 86 | 383 | 0.188 | 4.06 |

The `TiledF16` last-row values reproduce §2.2: lane 0 D 0.099, lane 3 D 0.014,
lanes 1 and 2 flip. This cross-checks the extended harness. On `TiledF32` the
last row of lane 1 no longer flips (D 0.183), but lane 2 still flips
(D 0.770 at vec margin 0.392).

Growth through the model (lane 1, `TiledF32` against `VecHybrid`):
- attention `attn_rel`: 4.95e-7 at layer 0, 6.3e-4 at layer 1, 0.249 at layer 7;
- residual `hidden_rel` at layer 29: 0.106;
- final max |Δlogit|: 2.00 (`TiledF16` 4.07).

A 40× margin on the frozen kernel bound does not keep the route inside the
decision envelope. §2.3.2 and §2.3.4 anticipated this: Gemma's MoE routing
amplifies any non-bitwise perturbation until it saturates.

Disposition (§8.5, operator ruling 2026-11-05):
- the route is falsified for default use;
- hf2q removes `precise`, `TiledF32`, its prewarm, the scratch ledger and the
  Metal fixtures in a clean cutover on PR #698;
- the mlx-native precise kernels stay (0.15.18).

Kept from this work:
- the every-row route envelope with per-row Q1 D and the A3/A4 asserts;
- the per-row logits probe;
- the localization relative RMS;
- `active_controls.gemma_warm_attention_kernel` in `/hf2q/v1/runtime`;
- the Q5 `skip_if_empty` readiness check.

The warm-attention lever moves to the bitwise-exact successor story
(`docs/research/adr063-q4-gemma-attn-exact-design.md`, draft on PR #701).
