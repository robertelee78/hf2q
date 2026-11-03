# ADR-063 Q4-GEMMA-KERNEL (#659) and Q4-GEMMA-ATTN (#668): precise tiled warm attention for Gemma 4

Status: design for approval (2026-11-03). No code, build or hardware run backs
this document. Every number below is quoted from a retained receipt, with its
path, or marked `[INFERENCE]`.

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
2. **The #552 patch is extended, not adopted as is.** On hardware, removing
   only the F16 Q pre-scale did not lower the pure kernel difference at layer
   0 (`attn_rel` 4.15e-4 against 3.6e-4 for `TiledF16`). Removing the F16 O
   store alone leaves the F16 V operand. The contract therefore also requires
   the **V operand to be the exact F32 dequantized TQ-HB value** (§3.3). The
   patch's `OutT`/`ScaleScores` template work is the starting point.
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

| Lane (history/suffix) | vec margin (nat) | `TiledF16`: top-1, D | `TiledF16Precise` (Q pre-scale fix only): top-1, D |
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
identical inputs, so its `attn_rel` is the pure kernel difference:

| Kernel | layer-0 `attn_rel` | layer-0 `hidden_rel` | layer-1 `hidden_rel` | layer-7 `hidden_rel` |
|---|---|---|---|---|
| `TiledF16` (Stage 2 record) | 3.6e-4 | — | — (residual Δ ×40 at layer 1) | — |
| `TiledF16Precise` | 4.15e-4 | 1.59e-4 | 7.93e-3 | 5.98e-2 |

### 2.3 Diagnosis

1. **The Q pre-scale was not the dominant error.** Removing it left the
   layer-0 kernel difference unchanged (4.15e-4 against 3.6e-4) and moved
   which lanes flip. It did not reduce how many lanes flip. The remaining
   differences are the two F16 storage roundings: V (relative 2^-11 per
   element; it does not average out when attention concentrates on a few
   keys) and O (2^-11 per element, before the O projection's own F16
   staging). `[INFERENCE]` from the rounding map; the per-item split was never
   measured.
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
   F16 once, F32 arithmetic, and the reduction order. The target is a
   kernel-level difference from vec-hybrid at F32 rounding level (≤ 2e-5
   relative, §5.1). That is at least 18× below `TiledF16`'s 3.6e-4.
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
| `dispatch_flash_attn_prefill_precise_d256` | `flash_attn_prefill_precise_d256_v32`, `…_v16` | sliding layers: external mask plus optional `blk`, or the in-kernel window (§3.4) |
| `dispatch_flash_attn_prefill_precise_d512_resume` | `flash_attn_prefill_precise_d512_v32`, `…_v16` | global layers: causal resume with `q_offset_in_k` and a `kv_capacity` stride |

The names are proposals. The binding parts are one dispatcher per head-dim
family and the V dtype selecting the entry point.

`_v32` takes F32 V and `_v16` takes F16 V. The kernels are template
instantiations of the existing `attention<…>` (d256) and
`flash_attn_prefill_d512_impl<…>` (d512) templates. The new template
parameters are `OutT = float`, `VT ∈ {float, half}` and score-domain scaling,
as in the #552 patch. Defaults reproduce the existing instantiations exactly.

### 3.2 Inputs, layouts, shapes

| Operand | dtype | Layout | Notes |
|---|---|---|---|
| Q | F16 | `[B, H, qL, D]` contiguous | Unscaled. Produced by the caller's F32 → F16 permute, the same single rounding vec-hybrid applies. |
| K | F16 | `[B, H_kv, kL, D]`; d512: head stride `kv_capacity·D` | The F16 K cache bytes, compacted. Never re-rounded. |
| V | F32 (`_v32`) or F16 (`_v16`) | as K | `_v32`: TQ-HB V dequantized by `tq_dequantize_hb_kv_seq` (F32 output, `centroid · scale_norm`). `_v16`: an F16 V cache (exact stored values). |
| mask (d256, optional) | F16 | `[qL, kL]` rank-2 broadcast or `[B, H, qL, kL]` | Values exactly 0 or −∞ (§3.4). |
| blk (d256, optional) | U8 | `ceil(qL/32) × ceil(kL/16)` | Requires the mask. Built from the fully written mask (§3.4). |
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
2. The scale is applied to the F32 scores: S·(scale·log₂e), with the factor
   formed in F32. Q is never multiplied in F16.
3. Mask: excluded keys get −∞. Online softmax with `exp2`: running max, sum
   and P all in F32.
4. P stays F32 into P·V. The V operand is F32: `_v32` reads the F32 value as
   stored; `_v16` widens the stored F16 exactly. No intermediate V, P or O
   value is rounded below F32.
5. O is accumulated in F32, normalized by the F32 row sum, and stored as F32.
6. Math mode: the same default compile mode as every other attention kernel
   on both metallib paths (§3.6). `requires_precise_fp32_math` is not
   extended. The `exp2` precision class is the one vec-hybrid's `exp`
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
  - d256: BQ 32, BK 16 (or BK 8 if §3.7 requires it);
  - d512: Q 8 × C 64, NSG from the existing selector.

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
max(24 × 256, 16 × 264) halves = 12,288 B, for 29,184 B in total. An F32 V
tile of 16 × 260 × 4 B = 16,640 B pushes the total to 33,536 B, which is over
the limit. Three implementations meet the contract:

| Option | How | Cost |
|---|---|---|
| a | Load V fragments directly from device memory into `simdgroup_matrix<float>`, as the d512 kernel already does | 2× V read bandwidth against F16 V |
| b | BQ 16 for `_v32` | `Q_smem` 8,448 B; twice as many threadgroups and less K reuse |
| c | Stage the TQ-HB V codes (16 × 256 B) and norms in threadgroup memory, then dequantize to F32 at fragment load with the `tq_dequantize_hb_kv_seq` expression | No dense V buffer at all; requires a third V input form |

The contract fixes values, not the implementation. Options a and b consume
the F32 dequant output, and option c must produce bitwise-identical V values,
proven by a test against `tq_dequantize_hb_kv_seq`. §6 picks the option.
d512 has no budget issue, because it already reads V from device memory.

## 4. Starting point and branch hygiene

- Rebase the #552 patch (written against 0.15.15, uncompiled) onto mlx-native
  0.15.16, which added only routing and RMS-norm race fixes. Work in a new
  worktree under `/opt/mlx-native-worktrees/`.
- Keep from the patch: `OutT`, `ScaleScores` (scale·log₂e applied to F32
  scores, no Q pre-scale) and the d512 `OutT` store.
- Change against the patch:
  - add `VT` (F32 V) to both templates;
  - add the separate precise dispatchers (§3.1) instead of switching
    existing dispatchers on `out.dtype()`;
  - implement the §3.7 option chosen in §6;
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
| P1 | precise `_v32` against reference | `rel_rms ≤ 1e-5` and `nmax ≤ 5e-5` |
| P1′ | precise `_v16` against reference (F16 V cache inputs) | same bounds |
| P2 | precise against vec-hybrid accuracy | `rel_rms(precise) ≤ max(2·rel_rms(vec), 2e-6)`, with vec measured against the same reference |
| P3 | direct route parity | `rel_rms(precise − vec) ≤ 2e-5` on the same TQ-HB codes and F16 K |
| C+ | positive control | vec-hybrid passes P1 |
| C− | negative control | the `TiledF16` chain (F16 V, F16 O) **fails** P1 at served shapes, so the test discriminates. `[INFERENCE]` Expected near 1e-4–4e-4, the layer-0 range. |
| P4 | determinism | 20 repeats bitwise equal, per specialization and per path |
| P5 | tail poison | NaN-filled K/V rows beyond `kL` and Q rows beyond `qL` give bitwise the clean-run O |
| P6 | masks | external mask plus `blk` and the in-kernel predicate (if adopted) are bitwise equal, and both match the predicate reference |
| P7 | no regression | every existing `flash_attn_prefill*` entry point gives bitwise its 0.15.16 output on fixed seeds (golden digests recorded before the change) |
| P8 | origin | `pipeline_identity(...).origin` asserted per path; cross-path O digest equality recorded |
| P9 | option-c only | in-kernel V values are bitwise equal to the `tq_dequantize_hb_kv_seq` output |

Bound rationale `[INFERENCE]`: the F32 error of a 256- or 512-term dot
product plus a ≤ 2,048-key online softmax with `exp2` sits near
√n · 2⁻²⁴ ≈ 3e-6 relative. 1e-5 leaves headroom while staying 36× below
`TiledF16`'s measured layer-0 difference.

## 6. Microbenchmark plan at served shapes

Bench: `benches/bench_flash_attn_warm_precise.rs` (mlx-native, harness
false). It loads no model, but it needs the root's GPU lane and a quiet host
(README-agents).

**Shapes.** One sequence (batch 1), with Gemma heads (§3.2), for each
(history H, suffix S):
- **Canonical run lanes**, from `$E/gemma-tiled-off-smoke-5d421a2c/server.stderr`
  (cached → prompt per turn): (458, 58), (516, 191), (707, 220), (927, 288),
  (1215, 150), (1379, 348).
- **Gate lanes:** (465, 56), (714, 200), (943, 311), (1254, 385).
- **Long-context stress** (the D2 long-context workload variant, PR #551):
  (4096, 1024), (8192, 4096), (32768, 4096).

Sliding layers stage `min(H, 1023)` history rows plus the chunk. Global
layers attend H + S.

**Arms.** Each arm is timed as the full per-chunk chain hf2q encodes, and
separately as the attention kernel alone:

| Arm | Chain |
|---|---|
| `VecHybrid` | `flash_attn_vec_hybrid_batched` over F16 K and TQ-HB codes with its reduce, 256-query chunks split at 512/1024 |
| `TiledF16` | K compaction, `tq_dequantize_hb_kv_seq_f16`, Q permute, mask, fence, `blk`, `…_f16_d256_with_blk` / `…_f16_d512_resume`, O cast, O permute |
| `Precise` (per §3.7 option a/b/c, ± in-kernel window) | K compaction, F32 V dequant (none for c), Q permute, [mask, fence, `blk`], precise dispatch, O permute (no cast) |

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
over the canonical lanes: 25 × sliding chain + 5 × global chain per lane per
turn. Compare with the Stage 2 fit, in which vec-hybrid accounted for about
1.91 s of the 3.43 s of joint warm forward per run.

**Go / no-go for hf2q integration** (frozen):
- M1: precise chain total over the canonical lanes ≤ 0.5 × VecHybrid total.
  Otherwise the speed basis is falsified and recorded.
- M2 (reported, not gated): precise/`TiledF16` chain ratio. Above 1.5 is
  flagged, because the projected saving shrinks.
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
- compacts K as today;
- V: `dispatch_tq_dequantize_hb_kv_seq` into an F32 dense-V scratch, or
  linearizes an F16 V cache for `_v16`, or skips the dequant under §3.7
  option c;
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

Localization bound, checked before the envelope gate:
`gemma_warm_tiled_attention_layer_localization` on `TiledF32` (lane 1,
history 714 / suffix 200) must show layer-0 `attn_rel ≤ 2e-5`. Layer 0 is the
pure kernel difference, so this shows the §5 precision survives integration.
A miss blocks §8.3 and goes back to §5.

### 8.3 Q1 decision envelope

Class B against `VecHybrid`, policy `hf2q.q1.numeric-policy.v1`, thresholds
unchanged.

**E1, route envelope gate** (`gemma_warm_attention_route_envelope`, 4 lanes,
arms `VecHybrid` control, `TiledF16` baseline, `TiledF32`):
- The `TiledF32` arm must have no near-tie-policy failure. A failure is a
  top-1 change where the vec margin is ≥ 0.10 nat.
- The gate reports the Q1 D for every suffix row, not only the last.
  Recording it needs an extension of the gate: `L[c]`, `R[c]` and the route
  top-1 log-prob.
- Per lane it reports max D, top-1 agreement and the number of rows with
  D > 0.25 (`RouteSensitive` rows).

**E2, canonical divergence coordinate.** Serving replay of the
`gemma-divergence-v1` agent-0 and agent-3 turn-2 requests:
- the same text-model hard link and harness;
- solo, and the four-agent concurrent trial ×3;
- arms `0`, `1` and `precise`.

At the decision token, with D computed against the `0` arm, `precise` must
have:
- (a) the same raw top-1;
- (b) the same chosen token after the 1.05 repetition penalty;
- (c) D ≤ 0.25 nat (`WithinDecisionEnvelope`).

The vec arm's margin there is 1.829 nat, so this is not a near-tie row. The
logs must show `kernel=TiledF32` and the `live=tiled-f32` banner.

**E3, greedy144 classification.** Q1 Gemma endpoint T0 with
`warm_live_kernel = TiledF32` against the same endpoint on `VecHybrid`
(accepted dataset `842768`), using `q1_decision_policy.py envelope`. The
result is classified and recorded. A `RouteSensitive` result must carry the
localizer's first divergence (`gemma_localizer`).

**Disposition rule.**
- E1 and E2 gate.
- E3 is recorded. Under Q1 §3.2, `RouteSensitive` forbids any "numerically
  neutral" claim and makes the paired quality campaign mandatory. That
  campaign is required here in any case.
- Context only, never a gate: hf2q's own accepted routes span margins
  1.46–1.83 nat at this coordinate (cold 1.464, warm vec 1.829). Their
  cold-versus-warm D is 0.365 nat, which is already above τ = 0.25. E2 is
  therefore strict. If E2(c) misses while (a) and (b) hold, the result goes
  to the operator as `needs_decision`; the threshold is not changed.

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

- **Fails E1 or E2:** the route is falsified for default use. Record the
  receipts. The kernel may still ship in mlx-native. hf2q keeps
  `precise` as an opt-in only if the operator wants the A/B surface;
  otherwise the variant is removed, a clean cutover. Gemma's Win then has to
  come entirely from the other profiled stages.
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

Frozen gate: the median `precise` makespan is ≤ 0.95 × the median of the `0`
arm, and recovers at least 60% of the `1` arm's saving. For reference, Stage
2 measured `TiledF16` at −7.8% against vec-hybrid. A6 must hold within each
arm.

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
- A1–A6 and the localization bound passed;
- E1 and E2 passed, and E3 is recorded;
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

## 11. Open questions for the approver

1. Should `precise` stay as an opt-in if E1/E2 falsify the route (§8.5), or
   should the variant be removed?
2. Should E2(c) stay strict at 0.25 nat, given that hf2q's own cold and warm
   routes differ by 0.365 nat at the coordinate (§8.3)? This design keeps the
   frozen τ and routes a miss to `needs_decision`.
3. Should the cold prefill's d256 path also move to score-domain scaling? It
   is listed out of scope (§10.2).
