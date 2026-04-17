# ADR-011: Flash-Attention-Style Tiled Prefill Kernel (hf2q Gate A Speed-to-Peer)

**Status:** Proposed (2026-04-17). Supersedes the "Gate A speed-to-peer is Run-scope" disposition in ADR-005's Closeout Amendment.
**Decision Makers:** Robert, Claude
**Related ADRs:** ADR-005 (Inference Server, Gate A), ADR-006 (mlx-native destination), ADR-008 (candle divorce)

## Engineering Mantra (load-bearing — read before every session)

Source: `~/Documents/mantra.txt` (Robert, undated). Quoted verbatim.

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

**Operational reading for this ADR:**
- **Measure 3x, cut once** — prefill tok/s measurements must be cold-cache, median-of-N, on the canonical harness. Throughput claims without the numbers don't land.
- **Dive deep** — two reference implementations (llama.cpp `kernel_flash_attn_ext`, candle `steel_attention` aka MLX-origin) are fully readable on disk at `/opt/llama.cpp` and `/opt/candle`. Chesterton's fence applies to each.
- **No fallback** — do not ship the old one-thread-per-q_pos `sdpa.metal` as a fallback. Per-op cutover (ADR-005 line 21 discipline). Once the flash-attn kernel lands and is correctness-verified, the old path is deleted.
- **Pure excellence** — closing a 21× gap is not "add some MMAs and call it done." It's an architectural rewrite. The ADR scopes accordingly.

## Problem Statement

hf2q's current `mlx_native::ops::sdpa` kernel at `/opt/mlx-native/src/shaders/sdpa.metal:40-151` uses a **one-thread-per-Q-position** dispatch: each thread owns one query row, iterates sequentially over all K positions accumulating into a private `float acc[512]` stack array. For seq_len=1 (decode), this is essentially optimal — one q_pos, one thread, bandwidth-limited.

For seq_len=2455 (batched prefill), it is **21× slower than llama.cpp's `kernel_flash_attn_ext` family on identical hardware, identical weights, identical prompt**:

| Path | Prefill tok/s (M5 Max, 2455-tok prompt, Gemma 4 26B MoE DWQ) |
|---|---|
| hf2q batched prefill (this kernel) | **~152 tok/s** |
| llama.cpp `llama-completion` (flash_attn_ext vec) | **~3260 tok/s** |
| **Gap** | **~21×** |

For a real user, this is the difference between a 2048-token prompt rendering in ~15 seconds (hf2q today) versus under 1 second (llama.cpp today). It dominates time-to-first-token, and it is the single largest remaining item on the ADR-005 Closeout Amendment that the Closeout classified as "Run-scope" and deferred.

### Why the current kernel is slow

1. **No simdgroup matrix hardware.** Apple Silicon has dedicated 8×8×8 FP16 matrix multiply-accumulate hardware (simdgroup MMA: `simdgroup_multiply_accumulate`, `simdgroup_load`, `simdgroup_store`, `simdgroup_barrier`). Our kernel uses plain scalar `float` FMA in a sequential `for (k = 0; k < head_dim; k++)` loop. On matrix-shaped workloads, scalar FMA is ~10× slower per FLOP than MMA on M-series chips. **Reference:** philipturner/metal-flash-attention benchmark — 4400 GINSTs/s on M1 Max at 83% ALU utilization using MMA; forward pass at head_dim=256 reaches 86% of peak on M1 and 82% on M3.
2. **No K/V tile reuse across Q positions.** Every thread reads K and V rows independently. In a flash-attn tiled kernel, a K-tile is loaded into threadgroup shared memory once per K iteration and reused by all Q rows in the tile — a ~(number-of-Q-rows-per-tile)× reduction in K/V memory traffic.
3. **Per-thread register spill.** `float acc[512]` per thread × 32 threads × 16 heads × 77 tiles ≈ heavy spill pressure; the Metal compiler demotes some of this to stack/device memory, paying the bandwidth cost again. Flash-attn-ext mitigates by explicitly managing which matrices live where (threadgroup scratch vs. register tiles vs. device).
4. **Online softmax state in registers, per q_pos.** Rebuilt from scratch for each thread; flash-attn kernels maintain the running `max`, `sum`, and `output_accum` per Q-tile with simdgroup-level reductions across the tile.

These are all algorithmic rather than constant-factor — this is why memory_barrier insertions, pool tuning, and other recent Phase 1b levers could not shrink the gap from the `sdpa.metal`-side. The fix is a different kernel.

### What passes on the current implementation

- Decode: ~103 tok/s hf2q vs ~102 tok/s llama.cpp → at parity (ADR-005 Gate B met). Decode has one q_pos, where our kernel is near-optimal.
- Correctness: hf2q prefill output is byte-identical to per-token reference at all measured lengths (Gate A correctness met post-b31505d).
- Short-prompt prefill (seq_len ≤ 128): the 21× gap is less painful in absolute milliseconds; real user pain starts at seq_len > 512.

## Reference Implementations (exhaustive landscape)

Three reference paths exist. Two were deep-read on disk by dedicated research agents; the third (metal-flash-attention) was studied via upstream source and published benchmarks. Full per-agent reports at `/tmp/swarm-flashattn-llamacpp.md` and `/tmp/swarm-flashattn-candle-mlx.md`.

### A. llama.cpp `kernel_flash_attn_ext_impl` (non-vec MMA path)

**Key correction from initial skeleton**: for prefill, llama.cpp dispatches the **non-vec** `flash_attn_ext_impl` template, **not** the vec family. The vec path is gated on `ne01 < 20` (`/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp:2505`) — it is the DECODE path, not prefill.

- **Main kernel** — `kernel_flash_attn_ext_impl<...>` at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal:5767-6375`. 135 instantiations across `{f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0}` × 15 `(DK, DV)` pairs. For our Gemma 4 head_dim=256:
  - f32 variant at `ggml-metal.metal:6475`
  - f16 variant at `:6491`
  - bf16 variant at `:6508`
- **Dispatch for Gemma 4 2455-token prefill** (`ggml-metal-ops.cpp:2807, 2861`): grid `(307, 16, 1)` × threadgroup `(32, 4, 1)` = **4912 threadgroups × 128 threads (4 simdgroups) each**, 16 KiB shared memory per threadgroup, K swept in **39 tiles of C=64**.
- **Hot loop** (`ggml-metal.metal:6044-6058`): **32 `simdgroup_multiply_accumulate(8×8×8)` calls per Q·K^T tile**. `lo[4]` output 8×8 register accumulators held register-resident across the entire K-sweep (`:6186-6256`).
- **Online softmax** (`ggml-metal.metal:6131-6174`): `M` and `S` are per-thread register arrays, `simd_max` / `simd_sum` handle cross-lane reductions, `ms = exp(M_old - M_new)` rescales the threadgroup-resident O accumulator each K-tile.
- **Sliding-window handling — critical discovery, differs from my initial framing**:
  - **No dedicated kernel.** Sliding window is encoded in the F16 mask itself, built CPU-side at `/opt/llama.cpp/src/llama-graph.cpp:345-442`.
  - **`flash_attn_ext_blk` pre-pass** (`ggml-metal.metal:5666-5719`) classifies each 8×64 mask tile into `{skip, load, no-op}` categories.
  - Consumed at `ggml-metal.metal:5951-5981`.
  - **~59% of tiles are skipped entirely** for window=1024 at seq_len=2455 — this is a major throughput win specific to SWA layers that pure flash-attn doesn't get.
- **License**: MIT.

### B. Candle `steel_attention` — recommended port source

Candle's kernel is a **verbatim MLX-upstream snapshot**. File header at line 1 of `/opt/candle/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal`: `// Updated from MLX commit has f70764a`. Internal banner at `:1796` points to `mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h`. Web-fetch of MLX `main` confirmed identical kernel structure — same `tile_matmad`, same `MMAFrag_acc_t::mma`, same online-softmax loop.

- **Kernel body** — `attention<BQ, BK, BD, WM, WN, ...>` template at `/opt/candle/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal:1895-2295`.
- **Instantiation table** at `:2315-2337`. For Gemma 4 head_dim=256 bf16: `instantiate_attn(iname, itype, 32, 16, 256, 4, 1, ...)` → **`BQ=32, BK=16, BD=256, WM=4, WN=1`**.
- **Tile geometry**: 128 threads per threadgroup = 4 simdgroups × 32 lanes; Q tile 32×256 in `Qs`, K/V shared tile 16×256 in `KV_smem` (sequentially reused K then V); **~28.5 KB threadgroup memory**; derived fragment counts `TQ=1, TK=2, TD=32`.
- **Simdgroup MMA primitive** at `scaled_dot_product_attention.metal:1243`: single `simdgroup_multiply_accumulate(D, A, B, C)` on 8×8×8 fragments, invoked via the triple-nested `tile_matmad` helper at `:1482-1504`. **Count: 32 MMAs/simdgroup for Q·K^T + 64 for P·V per KV block.**
- **Online softmax** at `scaled_dot_product_attention.metal:2198-2239`: register-resident, uses `fast::exp2` after pre-scaling Q by `log2(e)` at `:2000` (saves the `log(2)` per exp).
- **Candle's extensions on top of MLX upstream** (both worth keeping in the port):
  - NaN guards via `ExpSubOp` at `:1878-1886`
  - `-inf` factor in the softmax rescale at `:2216-2220`
  - Added per candle commit `46928bce` (Fix sliding window full sdpa corner case, #3438).
- **Host-side dispatch** — `/opt/candle/candle-metal-kernels/src/kernels/sdpa.rs`:
  - `call_sdpa_full` at lines 22-247
  - Tile-selection logic at 86-98 (picks `(BQ, BK)` from head_dim)
  - Kernel-name formatter at 123-124
  - Grid geometry at 230-239
- **Sliding window**: via mask buffer passed as a kernel argument; no separate kernel. The candle-patched `-inf` factor handling ensures window-excluded positions contribute 0 after softmax.
- **License**: Candle is dual Apache-2.0 / MIT; MLX upstream is Apache-2.0. Port into mlx-native (MIT/Apache) is clean under either license.
- **Proven on our shapes**: hf2q pre-ADR-008 used exactly this kernel on Gemma 4 26B MoE head_dim=256, seq_len up to several thousand. No dk=256 unknowns.

### C. philipturner/metal-flash-attention + metal-benchmarks (upper-bound calibrator)

- **Repo**: https://github.com/philipturner/metal-flash-attention (MIT / permissive).
- **Algorithm**: **same** as MLX / candle — online softmax + MMA tiles; additional `simdgroup_async_copy` pattern for K/V load overlap.
- **Benchmarks**:
  - M1 Max: **4400 GINSTs/s, 83% ALU utilization**.
  - D=256 forward: 86% of peak on M1, **82% on M3**. No M5 Max numbers published.
- **Not directly portable** — research/benchmark codebase, but calibrates our upper bound. Apple Silicon MMA throughput at head_dim=256 tops out near 80% of peak; a well-engineered port on M5 Max should reach that regime.

### D. Our current `mlx_native::ops::sdpa` (to be replaced for prefill)

- **Source** — `/opt/mlx-native/src/shaders/sdpa.metal:40-151` (f32), `:154-239` (bf16). Scalar-FMA loop over K per thread.
- **Dispatch** — `(batch, n_heads, n_tiles) × (TILE_Q=32, 1, 1)` — 32 threads per threadgroup, each thread owns one q_pos.
- **Throughput** — ~152 tok/s on 2455-token prompt. Scalar-compute-bound; 21× below flash-attn peer.
- **Retention policy**: this kernel stays for the DECODE path (seq_len=1, where one thread per q_pos is optimal). It is DELETED from the prefill path (`seq_len > 1`) once ADR-011 Phase 3 lands.

## Decision

**Port candle's `steel_attention` kernel (MLX-origin) into mlx-native as the prefill attention kernel for `seq_len > 1`.** Replace `sdpa.metal`'s and `sdpa_sliding.metal`'s prefill use entirely. Retain the current scalar kernel only for the decode path (where it's near-optimal at seq_len=1).

Rationale for choosing candle/MLX over llama.cpp as the port source (both agents independently concluded the same):

1. **Proven at our shapes**: hf2q's pre-ADR-008 candle build ran exactly this kernel at Gemma 4 26B MoE head_dim=256, seq_len up to several thousand. Zero dk=256 unknowns.
2. **License fit**: candle dual Apache-2.0 / MIT, MLX upstream Apache-2.0. Direct port into mlx-native (Apache-2.0) is clean under either.
3. **Upstream lineage**: candle's file is a verbatim MLX snapshot (`commit f70764a` per the header comment). Porting it KEEPS mlx-native aligned with MLX upstream, and future MLX improvements can be pulled in by re-syncing. This matches the ADR-006 ownership thesis.
4. **Structural match**: MLX's kernel is **single-dispatch per op** (one kernel launch per attention call), matching mlx-native's session model. llama.cpp's prefill is also single-kernel (the `_impl`, not the vec two-kernel pattern), but the code structure is heavier (C-preprocessor macro expansion, 135 instantiations in one file, `flash_attn_ext_blk` pre-pass for SWA). MLX's `attention<BQ, BK, BD, WM, WN>` template is more readable.
5. **Correctness extensions**: candle's added NaN guards (`ExpSubOp` at `:1878-1886`, `-inf` factor at `:2216-2220`, commit `46928bce`) are worth inheriting — llama.cpp does not have these specific guards.
6. **Mask-based sliding window** (both refs agree): both flash-attn kernels route SWA through a mask buffer rather than a separate kernel, so porting once gives us both global and sliding layer coverage. Our current architecture has two kernels (`sdpa.metal` + `sdpa_sliding.metal`) — the port **simplifies** that to one.

**Rejected alternatives:**

- **llama.cpp `flash_attn_ext_impl` port** — viable and its MMA/softmax structure is identical to MLX's at the 8×8×8 MMA level. Rejected for: (a) MLX lineage preservation argument above, (b) the `flash_attn_ext_blk` 8×64 mask pre-pass is a separate kernel add, (c) the 135-instantiation macro soup is harder to cherry-pick from. That said, if the candle/MLX port hits performance trouble, the `flash_attn_ext_blk` **59%-tile-skip trick is a Phase 5 Run-scope optimization** worth cherry-picking as an ORTHOGONAL layer on top of the ported kernel.
- **Extend current `sdpa.metal`** — doesn't address scalar FMA root cause. Not a 21× fix.
- **Write from scratch using metal-flash-attention as reference** — reinvents a proven kernel. Not Walk-phase work.

## Implementation Phases

The port is scoped into discrete phases, each with a correctness gate. Each phase lands with file:line citations and bench numbers; no phase "in progress across commits."

### Phase 0 — Reference capture (this ADR)

Land the port decision, confirm the reference kernel is readable, and confirm the target shapes. Done by this ADR + the two background research reports.

**Entry criteria met.** Exit criteria:
- [x] ADR-011 landed
- [x] `/tmp/swarm-flashattn-llamacpp.md` produced (background research)
- [x] `/tmp/swarm-flashattn-candle-mlx.md` produced (background research)

### Phase 1 — Vendor the MLX kernel into mlx-native

Copy `scaled_dot_product_attention.metal`'s `attention<BQ=32, BK=16, BD=256, WM=4, WN=1>` instantiation into `/opt/mlx-native/src/shaders/flash_attn_prefill.metal` with the upstream copyright/license header preserved. Add a Rust dispatch function at `/opt/mlx-native/src/ops/flash_attn_prefill.rs` mirroring the candle host-side dispatch logic.

Include: Q tile layout, KV_smem reuse pattern, simdgroup matmul instructions, online softmax state, output store. Start with f32 Q/K/V/O for cheap correctness verification; add bf16 variant after f32 lands.

Instantiations needed for Gemma 4:
- `dk=256, dv=256` (sliding layers)
- `dk=512, dv=512` (global layers — needs separate template instantiation; verify candle has it or add)
- Both with optional window_size param for SWA support

**Exit criteria:**
- [ ] Kernel compiles, matches candle's simdgroup matmul count and tile geometry exactly (diff-check against candle source)
- [ ] Unit tests in `/opt/mlx-native/tests/test_flash_attn_prefill.rs` at fixed small shapes (seq_len=32, 128, 512) pass byte-identical against a scalar CPU reference at f32
- [ ] Benchmark: standalone kernel on 2048-token shape ≥ 2000 tok/s (sanity: 21× → 13× gap at minimum, before integration overhead)

### Phase 2 — Integrate into hf2q batched prefill (correctness only)

Route `forward_prefill_batched.rs:386-430` SDPA dispatch through the new kernel for both sliding and global attention layers. Gate behind `HF2Q_FLASH_ATTN_PREFILL=1` to allow A/B comparison; leave the current `s.sdpa` / `sdpa_sliding` paths as the default until Phase 3 correctness is proven.

**Exit criteria:**
- [ ] 576-token batched prefill: 10/10 deterministic, byte-identical to per-token reference
- [ ] 2455-token batched prefill: 10/10 deterministic, byte-identical output to the current (post-b31505d) hf2q baseline
- [ ] Sourdough parity gate at min-prefix 3094: PASS
- [ ] sliding_wrap parity at min-prefix 700: PASS
- [ ] All `tests/evals/reference/*_hf2q.txt` Gate D self-baselines match byte-for-byte

### Phase 3 — Benchmark, tune, flip default

Benchmark the new kernel at the canonical harness shapes. Target: **≥2500 tok/s on 2455-token prompt (77% of llama.cpp peer's 3260 tok/s)**; stretch: ≥3000 tok/s. Tune tile sizes if below target (BQ/BK sweep, simdgroup count sweep) using the candle-proven geometry as the starting point.

Once the target is hit, flip `HF2Q_FLASH_ATTN_PREFILL=1` to default on, delete the old per-thread sdpa prefill path (keep decode scalar kernel), update `scripts/release-check.sh` Gate 3 floor from the current 130 tok/s (thermal-accommodating thermals-after-gate-2) to the new steady-state.

**Exit criteria:**
- [ ] Gate A prefill floor raised to ≥2500 tok/s and `scripts/release-check.sh` updated
- [ ] `docs/spike-gate-a-prefill.md` addendum with post-port measurement table
- [ ] Old `sdpa.metal` prefill path deleted (cutover, not coexistence)
- [ ] Gate A speed-to-peer status in ADR-005 Closeout Amendment flipped to MET

### Phase 4 — bf16 variant + sliding SWA mask

Add the bf16 I/O variant of the kernel (Q/K/V/O all bf16) for the FP16 path that candle's SDPA used. Port the sliding-window mask handling so we can retire `sdpa_sliding.metal` and use one kernel for both global and sliding layers at prefill.

**Exit criteria:**
- [ ] Gemma 4 sliding-layer prefill runs through the new kernel with correct window_size mask
- [ ] Sourdough + sliding_wrap parity both PASS
- [ ] bf16 path landed and on by default where candle used bf16

## Success Criteria

The ADR is complete when all of the following are true on HEAD, measured via the canonical release-check harness on M5 Max with Gemma 4 26B MoE DWQ:

1. **Prefill tok/s on 2455-token prompt ≥ 2400.** Walk-phase floor per the llama.cpp researcher's perf model ("2400-3000 tok/s, closes ~15× of the 21× gap"). Stretch target ≥ 3000. True parity-to-peer (3260 tok/s) is **Phase 5 / Run-scope** and may require the `flash_attn_ext_blk` 59%-tile-skip optimization on sliding layers.
2. **Prefill output byte-identical** to the current (pre-port) per-token reference path at all 4 canonical prompts (short_hello, sourdough, sliding_wrap, prefill_2048). Gate D self-baselines continue to match.
3. **No decode regression.** Decode tok/s stays ≥ 100 (current median 103 → peer 102).
4. **No per-forward sync regression.** Gate G counter thresholds (dispatches/decode_tok ≤ 1300, total syncs ≤ 60) still pass.
5. **Old `sdpa.metal` + `sdpa_sliding.metal` prefill paths deleted.** Decode scalar kernel retained (it's bandwidth-optimal at seq_len=1). Per-op cutover, no coexistence. The new kernel covers both global and sliding via a mask argument — TWO files become ONE.
6. **Release-check Gate 3 floor raised** from 130 tok/s to the ≥2400 target (allow thermal headroom — 2200 is the practical floor).

## Risks and Mitigations

**R1: Large-head-dim register spill (D=256 doesn't fit in registers).**
Per the metal-flash-attention author's note: "At large head dimensions (e.g. 256), none of the matrix blocks can fit into registers. Not even the accumulator can. Therefore, intentional register spilling is done." Our target is D=256 and D=512 (global). Mitigation: start from the candle `BD=256, WM=4, WN=1` geometry — this is proven on exactly our shapes. If tuning shows D=512 needs a different geometry, add a separate instantiation.

**R2: Correctness regression — flash-attn numerics differ from the current scalar kernel at f32/bf16 round-off boundaries.**
The current kernel accumulates in f32 across the full K axis per thread. The flash-attn online softmax rescales the running accumulator on every max update; numerical error is different (theoretically tighter, empirically close). Mitigation: Phase 2 gates require byte-identical output; if the 3656-byte sourdough prefix drops even one byte, the port must bisect the algorithmic difference before landing.

**R3: Sliding-window mask divergence from `sdpa_sliding.metal`.**
The current `sdpa_sliding` kernel uses `window_start = max(0, abs_pos - window_size)` and loops K from there. Flash-attn mask typically adds `-inf` before softmax for out-of-window positions. Both are correct but produce different round-off. Mitigation: Phase 4 uses the existing sliding_wrap test fixture (min-prefix 700) as the sliding-mask correctness gate; the reference `*_hf2q.txt` baseline was captured with the `sdpa_sliding` kernel so byte-identical to that is the test.

**R4: Hf2q's batched prefill session-layer plumbing assumes a single SDPA dispatch.**
`forward_prefill_batched.rs:386-430` is structured around `s.sdpa(...)`. The new kernel is drop-in at the session level if its dispatch signature matches. Mitigation: add a parallel `s.flash_attn_prefill(...)` method on the session rather than overloading `s.sdpa`; route only for batched prefill, not decode.

**R5: Phase 3's "flip default + delete old path" risks a known-good → unknown cutover.**
Mitigation: Phase 2's `HF2Q_FLASH_ATTN_PREFILL=1` env-var gate lives through Phase 2 and Phase 3. Only flipped to default when every correctness gate passes. Per ADR-005 mantra discipline: no fallback, but no premature cutover either.

**R6: Estimated effort.**
Realistic: 2-5 engineering days for Phase 1+2 (vendor + integrate + correctness), 1-2 days for Phase 3 (benchmark + tune + flip). Phase 4 (bf16 + mask unification) another 1-2 days. Total 4-9 days of focused work. Not a session-scale patch; that's why it lived as Run-scope through Phase 1b closure.

## Reversibility

If the port lands at substantially below the 2500 tok/s target AND correctness gates fail AND tuning doesn't close either, the decision is reversed by:
1. Re-adding the `sdpa.metal` prefill dispatch path (git revert of Phase 3's deletion commit)
2. Restoring Gate 3 floor to 130 tok/s
3. Closing ADR-011 as "attempted, see rejection rationale"

No hf2q source code outside `forward_prefill_batched.rs`'s SDPA dispatch changes during Phase 1–3. Reversibility footprint is a git revert of 3–4 files.

## Cross-References

- **ADR-005** §Gate A Closeout Amendment: Gate A speed-to-peer was classified Run-scope because closing the 21× gap requires exactly this work. ADR-011 is that work, lifted out of "Run-scope deferral" into an explicit phased plan.
- **ADR-006** §mlx-native-owned backend: flash_attn_prefill.metal goes in /opt/mlx-native, matching ADR-006's ownership.
- **ADR-008** §candle divorce: we divorced the CANDLE FRAMEWORK, not its Metal kernels. Vendoring a single Metal shader from candle (MLX-origin) is permitted under ADR-008; we're not reinstating the candle dep.

## Resolved Questions (answers from the two research reports)

- [x] **Exact simdgroup matmul instruction sequence** — `simdgroup_multiply_accumulate(D, A, B, C)` on 8×8×8 fragments. Candle at `scaled_dot_product_attention.metal:1243`, invoked via `tile_matmad` at `:1482-1504`. llama.cpp at `ggml-metal.metal:6044-6058`. **Algorithm is identical across both references.** 32 MMAs per Q·K^T tile (both confirm).
- [x] **D=512 (global layer head_dim)** — candle's instantiation table at `:2315-2337` covers head_dim values via the `BD` template parameter. llama.cpp instantiates 15 `(DK, DV)` pairs. We need to verify the exact `(BQ, BK, BD, WM, WN)` constants for D=512; candle's table is the reference. If D=512 is not currently instantiated in candle, adding the instantiation is a template parameter change, not a kernel rewrite.
- [x] **MLX upstream vs candle's snapshot** — candle is `MLX commit f70764a` verbatim plus candle's own NaN-guard additions (commit `46928bce`). Pulling fresher MLX main is a Phase 5 optimization. Candle's snapshot is sufficient for Walk-phase.
- [x] **Threads per threadgroup** — confirmed 128 (4 simdgroups × 32 lanes) for dk=256. Fragment counts TQ=1, TK=2, TD=32. ~28.5 KB threadgroup memory.
- [x] **Throughput ceiling on M5 Max** — M1 Max: 4400 GINSTs/s, 83% ALU; D=256 forward on M3: 82% of peak. M5 Max has more ALUs and modestly faster memory; the realistic regime per both agents is **2500–3000 tok/s post-port**, conservative floor **1800 tok/s**. Closes 12–20× of the current 21× gap.

## Open Questions (for Phase 1 execution)

- [ ] Exact `(BQ, BK, BD, WM, WN)` for D=512 (global layers). Candle's instantiation table resolves this — to be confirmed at Phase 1 kick-off.
- [ ] Whether to port candle's NaN-guard extensions in Phase 1 (recommended) or Phase 4 (simpler Phase 1).
- [ ] Whether the `flash_attn_ext_blk` SWA tile-skip optimization (llama.cpp `ggml-metal.metal:5666-5719`, ~59% tile skip at window=1024 seq=2455) is worth porting as an orthogonal layer — if the Walk-phase port lands at ~2400 tok/s on sliding layers, this is the one known trick to push closer to peer.

## Research Artifacts

Full reports from the two background agents that drove this ADR:

- `/tmp/swarm-flashattn-llamacpp.md` — llama.cpp deep-dive; 7 sections covering kernel catalog, simdgroup MMA usage, tiling geometry, online softmax, vec vs impl split, SWA via mask+blk pre-pass, perf model.
- `/tmp/swarm-flashattn-candle-mlx.md` — candle / MLX / metal-flash-attention comparison; recommendation to port candle steel_attention verbatim.

Both converge on the same port source and the same perf expectation — strong convergent signal.

## Status and Next Action

- **ADR-011 status: Proposed → ready for Accepted review.** Phase 0 (reference capture) complete; both research reports delivered with file:line citations merged into this ADR.
- **Phase 1 can begin immediately.** Literal vendoring of candle's `scaled_dot_product_attention.metal:1895-2337` + `kernels/sdpa.rs:22-247` into `/opt/mlx-native/src/shaders/flash_attn_prefill.metal` + `/opt/mlx-native/src/ops/flash_attn_prefill.rs`.
- **Estimated elapsed**: 4-9 engineering days total across all 4 phases, per the initial risk analysis (R6).
- **Owner**: Robert + Claude, next focused session.
