# Peer SDPA vs hf2q TQ-HB Chain — Structural Comparison (iter-485, Worker A)

**Purpose:** Identify exactly what llama.cpp's `kernel_flash_attn_ext_impl` fuses
into a single GPU dispatch that hf2q currently splits across six separate kernels
in the TQ-HB decode chain. Inform fusion priorities for Phase 7d (preserve TQ-HB
3.94× memory savings; no Path E).

**Files audited (line-by-line):**

- `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` lines 5801–6464 (impl + dispatch wrapper)
- `/opt/mlx-native/src/shaders/flash_attn_vec_tq_hb.metal` (724 lines, complete)
- `/opt/mlx-native/src/ops/flash_attn_vec_tq_hb.rs` lines 1–365 (dispatcher)
- `/opt/mlx-native/src/shaders/hadamard_quantize_kv_fast.metal` lines 140–325 (K encoder), 860–1004 (HB dual K+V)
- `/opt/mlx-native/src/shaders/fwht_standalone.metal` (212 lines, complete) — premult + undo
- `/opt/mlx-native/src/ops/fwht_standalone.rs` lines 78–152 (premult+undo dispatchers)
- `/opt/mlx-native/src/ops/hadamard_quantize_kv.rs` lines 680–738 (HB dual dispatcher)
- `/opt/mlx-native/src/shaders/flash_attn_vec.metal` lines 340–408 (reduce kernel)

---

## 1. Peer (`kernel_flash_attn_ext_impl`) — Single-Dispatch Anatomy

llama.cpp performs the **entire** prefill-or-decode attention in **one** kernel
launch per (KV-head-tile × Q-tile) workgroup. No producer/consumer barriers
across separate dispatches.

### 1.1 CPU dispatch wrapper (`ggml-metal-ops.cpp`, kernel switch at line 6439)

- One `[encoder dispatchThreadgroups: ...]` call per FA op.
- Wrapper at 6439–6464 is a 2-case `switch (FC_flash_attn_ext_nsg)` selecting
  the NSG=4 or NSG=8 template instantiation; NSG=1/2 paths are commented out
  for library load-time savings. Template parameters bake DK, DV, dequant fn,
  data types — all dead-code-eliminated by the Metal compiler.
- One `[[host_name(...)]]` instantiation per (precision × DK × DV) combination
  (see ~60 templates at 6498–6543). Pipeline picked once per ggml-op; no
  per-token CPU branching.

### 1.2 Inside the single dispatch (5801–6409)

| Phase | Lines | Op |
|---|---|---|
| Shared-mem bind | 5850–5865 | `sq`, `so`, `ss`, `sk`, `sv`, `sm2` all aliased to one threadgroup half-heap; per-simdgroup banks via `sgitg*(4*16*KV)` offset. |
| Mask pointers | 5867–5873 | Per-query `pm2[NQ]` precomputed from device pointers. |
| Q load | 5893–5905 | `FOR_UNROLL jj`: each thread copies DK4 float4s into shared sq4. Out-of-bounds rows zeroed. |
| Zero accumulators | 5907–5920 | `so4` (output) and `ss` (scratch) zeroed; one `threadgroup_barrier`. |
| Online softmax state init | 5922–5925 | `S[NQ]=0`, `M[NQ]=-FLT_MAX/2` per-query, in registers. |
| ALiBi/slope | 5927–5937 | Only if `FC_flash_attn_ext_has_bias`; constant-folded by FC. |
| **KV outer loop** | 5941–6360 | `for (ic0 = 0; ic*C < ne11; ++ic0)`. Each iteration covers C=8/16/32 KV positions. |
| ↳ pad-buffer remap | 5948–5983 | Last partial chunk reads from `pad` buffer; remaps mask. FC-gated. |
| ↳ mask load | 5988–6038 | Read `blk[ic0]` (precomputed mask sparsity flag) — value 0 skips entire chunk, 2 skips mask add. Mask half2 → `sm2`. |
| ↳ **Q·Kᵀ (simdgroup matmul)** | 6041–6161 | Two branches: **f16/bf16 K (line 6043)**: `simdgroup_load` + `simdgroup_multiply_accumulate` directly from device memory. 8×8 simdgroup matrix; `DK8/2` unroll up to 4×NSG. **Quantized K (6100)**: per-lane block dequant → `sk4x4` shared → `simdgroup_load` from shared. Either way produces `mqk` matrix `simdgroup_store(mqk, ss + 8*cc, SH, 0, false)`. |
| ↳ barrier | 6163 | `threadgroup_barrier` (single, intra-kernel). |
| ↳ **Online softmax + scale rescale** | 6166–6208 | Per-query: `s2 = ss2[...]*scale`; softcap if FC; `+= slope*mask` if bias; `M[jj] = simd_max(...)`; `ms = exp(m-M)`, `vs = exp(s2-M)`; `S[jj] = S[jj]*ms + simd_sum(vs)`. **Immediately** rescales `so4[i] *= ms` in-place (no separate kernel). |
| ↳ barrier | 6210 | `threadgroup_barrier`. |
| ↳ **O += P·V (simdgroup matmul)** | 6213–6357 | Two branches: **f16/bf16 V (6215)**: `simdgroup_load` V directly + accumulate into per-thread register tile `lo[NO]`; final `simdgroup_store(lo[ii], sot, ...)` writes back to `so`. **Quantized V (6291)**: per-lane dequant → `sv4x4` shared → `simdgroup_load` + accumulate. |
| ↳ barrier | 6359 | `threadgroup_barrier`. |
| Sinks | 6362–6380 | `FC_flash_attn_ext_has_sinks` — apply sink token; rescale `so4 *= ms`. |
| **Output write + final softmax norm** | 6384–6405 | `dst4[i] = (float4)so4[j*PV4 + i] * (1/S[jj])`. **Final 1/S divide happens here**, fused with the global-memory store. No separate "reduce" or "normalize" kernel. |

**Key peer properties:**

1. **One dispatch, one barrier set.** Internal barriers are *intra-encoder
   threadgroup barriers* (≈100s of ns), not the inter-kernel
   `enc.memory_barrier()` we use (full RAW barrier, microseconds).
2. **simdgroup matrix ops** (`simdgroup_multiply_accumulate`) instead of
   per-thread float-dot — uses tensor-core-style 8×8 matmuls.
3. **Online softmax is in-register-only.** S, M live in `float[NQ]`
   thread-local registers across the entire KV loop; never spilled to global.
4. **Final 1/S divide is folded into the write-to-global**, line 6398. No
   separate reduce when running with NSG sharing one tile.
5. **K cache is not Hadamard-rotated.** Peer's K is f16 or scalar-quantized
   block (Q4_0/Q5_K) and dequantized on-the-fly inside the matmul loop.
6. **No FWHT.** Peer does not need pre-rotation because K is not TQ-quantized.
7. **NSG splitting** (8 or 4 simdgroups per WG) reduces K-iters per simdgroup;
   cross-SG reduce happens *inside the same kernel* via shared-memory passes,
   not via a follow-up dispatch.

### 1.3 What peer fuses that hf2q does NOT

- **Q load + matmul input prep** (peer reads Q directly into shared and uses
  `simdgroup_load`; hf2q reads Q after a separate fwht_sign_premult dispatch).
- **Softmax rescale of so** is inside the inner loop, in registers.
- **Final 1/S normalize** is fused into the device write of `dst4[i]`.
- **NSG cross-simdgroup reduce** (analog of our `flash_attn_vec_reduce` for
  the NWG>1 case) is folded into the same kernel via shared-memory passes
  (peer uses NSG inside one WG; hf2q uses NWG across multiple WGs which
  requires a follow-up reduce kernel).
- **Mask materialization** uses precomputed `blk[ic0]` sparsity flags allowing
  whole-chunk skips; hf2q rebuilds the mask per-chunk inside the kernel from
  ring-buffer arithmetic.

---

## 2. hf2q TQ-HB Decode Chain — Six-Kernel Breakdown

Order of dispatch per layer per token (decode), as wired in qwen/gemma decode:

| # | Kernel | Source | Purpose | Dispatch grid | Barrier after |
|---|---|---|---|---|---|
| 1 | `hadamard_quantize_kv_fast_d{256,512}` | `hadamard_quantize_kv_fast.metal:143` | Encode current K row → byte-packed indices + per-position norm(s). | `(num_kv_heads, 1, 1)` × `(32, 1, 1)` (1 simdgroup/head). | RAW barrier before SDPA reads K_packed. |
| 2 | `hadamard_quantize_kv_hb_dual_d{256,512}` | `hadamard_quantize_kv_fast.metal:864` | Encode V row only (the dual-stream form is invoked with K stream typically a no-op when called via the V-only path; full dual is used elsewhere). At decode, V is encoded via this kernel. | `(num_kv_heads, 1, 2)` × `(32, 1, 1)` — z=0,1 picks K|V. | RAW barrier before SDPA reads V_packed. |
| 3 | `fwht_sign_premult_f32_d{256,512}` | `fwht_standalone.metal:122` | Apply D1 sign + FWHT + 1/√d normalize to Q in place. | `(num_heads, 1, 1)` × `(32, 1, 1)`. | RAW barrier before SDPA reads Q. |
| 4 | `flash_attn_vec_tq_hb_dk{256,512}` | `flash_attn_vec_tq_hb.metal:330` | The actual TQ-HB SDPA — Q·Kᵀ + online softmax + P·V. Per-WG splits K across NWG×NSG simdgroups. | `(1, num_heads, NWG)` × `(32, NSG, 1)`. | RAW barrier before reduce (only when NWG>1). |
| 5 | `flash_attn_vec_reduce_dk{256,512}` | `flash_attn_vec.metal:351` | Cross-NWG reduce: read NWG partial (S, M, so) banks, compute global M, rescale ms, sum S, write final `so * (1/S_total)`. | `(num_heads, 1, 1)` × `(32 * NWG, 1, 1)`. Only fires when NWG > 1. | RAW barrier before fwht_sign_undo. |
| 6 | `fwht_sign_undo_f32_d{256,512}` | `fwht_standalone.metal:168` | Apply FWHT + 1/√d normalize + D1 sign undo to attention output in place. | `(num_heads, 1, 1)` × `(32, 1, 1)`. | RAW barrier before next layer's projection reads it. |

**Note** there is *already-landed* infrastructure (iter-106) to fuse kernel 3
into kernel 4 via `params.fuse_fwht_pre`. Per `ops/flash_attn_vec_tq_hb.rs`
line 47–53, this is **OFF by default** ("caller-rotated path — production
default, byte-identical"). H1 below is to *flip the default*, given that the
fused path is already implemented in the kernel.

### 2.1 Per-kernel GPU/CPU detail

#### Kernel 1 — `hadamard_quantize_kv_fast` (K encoder, 1175-line file, lines 143–323)
- **GPU work:** EPT=DK/32 elements/thread in registers. (i) Load → (ii) D1 sign
  pre-mult (TBQ_SIGNS_256/512 lookup) → (iii) `fwht_simd<EPT>` (zero
  threadgroup barriers; uses `simd_shuffle_xor`) → (iv) 1/√d normalize →
  (v) compute L2 (D=256) or 2× per-block RMS (D=512) via `simd_sum` →
  (vi) scale to N(0,1) → (vii) Lloyd-Max quantize via 4-stage binary search →
  (viii) pack nibbles → (ix) write `packed[head, pos, coord/2]` and `norms[head, pos]`.
- **Threadgroup mem:** none (all-register + simd shuffles).
- **simd ops:** `simd_shuffle_xor`, `simd_sum`.
- **CPU dispatch overhead:** ~one Rust encode call + pipeline-state-binding +
  buffer arg binding (7 args). Pipeline is cached after first build.
- **Producer-consumer barrier with #4:** SDPA reads `packed` + `norms`;
  requires `enc.memory_barrier()` between #1 and #4 (RAW).

#### Kernel 2 — `hadamard_quantize_kv_hb_dual` (lines 864–992)
- **GPU work:** Same algorithm as #1 but byte-packed (1B/elem) and two-stream
  via `tgpig.z ∈ {0,1}` selecting K vs V. At decode, used to encode V; K is
  encoded by #1 (4-bit) — *or*, when running HB-only, the K stream is the
  `hadamard_quantize_kv_hb` single-stream form (line 557). The dual form is
  one of two encode paths.
- **CPU overhead:** identical to #1 (1 Rust encode call). Critically, the dual
  form is the **already-fused K+V encoder** — if K is byte-packed too, one
  dispatch produces both.
- **Producer-consumer barrier with #4:** RAW barrier required.

#### Kernel 3 — `fwht_sign_premult` (lines 122–160)
- **GPU work:** EPT=DK/32. Load → D1 sign pre-mult → fwht_simd → 1/√d
  normalize → write back. All-register; zero threadgroup barriers internally.
- **Threadgroup mem:** none.
- **simd ops:** `simd_shuffle_xor`.
- **CPU overhead:** 1 dispatch, 2 buffer args. Trivially small *kernel* but
  the **CPU encode + RAW barrier** is the cost — measured in iter-104 as ~9% of
  decode (~1.44 ms/token).
- **Producer-consumer barrier with #4:** RAW (Q rewritten in place).

#### Kernel 4 — `flash_attn_vec_tq_hb` (724-line shader, lines 330–712)
- **GPU work:** Online-softmax FA-vec.
  - Q load from shared `sq4` (caller-rotated *unless* `fuse_fwht_pre=1`).
  - K-outer loop over `ic0` with C=32 KV positions/iter; NWG×NSG splitting.
  - Per-iter: build mask scalar → dequant K (per-thread `dequant_hb_float4`
    via 4-byte load + codebook lookup, with function-constant cbits) →
    per-thread float-dot accumulate → `simd_sum` → online softmax update of
    M,S in registers → dequant V → weighted accumulate into `lo[DV4/NL]`
    register tile → flush `lo` into `so4` shared bank.
  - Final cross-simdgroup reduce when NSG>1 (lines 654–692): pass1 finds
    `M_global`, pass2 sums `S_total`, pass3 accumulates `so` banks.
  - Final write (695–711): `dst4[rid * DV4 * NWG + NWG * i + iwg] = so4[i] * inv_S`.
    Crucially, **inv_S = 1/S only when NWG=1**, else inv_S=1 (deferred to reduce).
- **Threadgroup mem:** `PK + NSG*(SH + 2*PV)` halfs. At NSG=1, DK=DV=256:
  `512 + 1*(128 + 256) = 896` halfs = 1792 B. Modest.
- **simd ops:** `simd_max`, `simd_sum`, **NOT** simdgroup_matmul (because
  K and V are byte-packed quantized — no f16 matrix form available).
- **CPU overhead:** 1 encode call, 7 buffer args + 1 push-constant FC for cbits.
- **Producer-consumer barrier with #5/#6:** RAW when NWG>1 (between #4 and #5).

#### Kernel 5 — `flash_attn_vec_reduce` (lines 351–399)
- **GPU work:** Only fires when NWG > 1. Each thread = one workgroup's S,M.
  `M_global = simd_max(M_wg)`; `ms = exp(M_wg - M_global)`;
  `S_total = simd_sum(S_wg * ms)`; loop over DV4 chunks summing
  `htmp4[i*NWG+iwg] * ms` and writing `dst4[i] = reduced * (1/S_total)`.
- **Threadgroup mem:** none.
- **CPU overhead:** 1 encode call, 4 args; only when NWG>1.
- **Producer-consumer barrier with #6:** RAW.

#### Kernel 6 — `fwht_sign_undo` (lines 168–211)
- **GPU work:** EPT=DV/32. Load → fwht_simd → 1/√d normalize → D1 sign undo →
  write back. Mirror of #3.
- **CPU overhead:** 1 dispatch, 2 args. Same cost class as #3.
- **Producer-consumer barrier with downstream o_proj matmul:** RAW.

### 2.2 hf2q chain totals per decode token

- 6 dispatches (5 when NWG=1) per layer × (Gemma 26 / Qwen 36) layers.
- ≥5 inter-dispatch `enc.memory_barrier()` calls per layer.
- ~16% of decode dispatches per the prompt's measurement.
- Per-dispatch wall is 2.07× peer per the prompt; pure kernel+CPU exec is
  1.81× slower than peer despite 1.54× **fewer** dispatches.

---

## 3. Fusion Opportunities (ranked by structural feasibility)

### H1 — Fuse `fwht_sign_premult` into `flash_attn_vec_tq_hb` Q-load (Q-FWHT in-kernel)
- **Status:** **Already implemented**, off by default. Kernel code at
  flash_attn_vec_tq_hb.metal:386–423 runs when `params.fuse_fwht_pre != 0u`.
  Inlined `fwht_simd_fa<EPT>` byte-for-byte matches fwht_standalone's `fwht_simd`.
- **Feasibility: HIGH.** Code exists; just flip default. Requires a parity
  sweep across all Gemma+Qwen prefill+decode shapes (Q is rewritten in shared
  memory only, so device-side outputs should be byte-identical at the FA
  output — the per-kernel-call FWHT result lands in the same `sq4` shared
  layout the caller-rotated path produces).
- **Dispatches saved/token:** 1 dispatch per layer = 26 (Gemma) / 36 (Qwen).
- **Barriers saved/token:** 1 RAW barrier per layer (the one between #3 and #4).
- **Expected kernel-µs saved:** iter-104 attributed `~1.44 ms/token = ~9% of
  decode` to this dispatch. Lower bound estimate: 0.5–1.0 ms; upper bound 1.4 ms.
- **Risk:** LOW. The fused path's TBQ_SIGNS_256_FA tables (lines 267–282) and
  `fwht_simd_fa` (lines 292–311) are byte-for-byte copies of the standalone
  forms; the `inv_sqrt_d` factor and store-as-half4 match the caller-rotated
  path. Parity test exists structurally (kernel author asserts byte-identity
  in the comment at line 313).
- **Recommended action:** flip default + parity-gate.

### H2 — Inline `flash_attn_vec_reduce` (NWG>1 case) into FA kernel
- **Concept:** Today, NWG=32 at long context → kernel 5 fires. Restructure the
  SDPA kernel to do cross-WG reduce internally (impossible without
  cross-workgroup sync) OR force NSG-only splitting (NWG=1) at the kL
  ranges where reduce overhead exceeds NWG parallelism gain.
- **Feasibility: MEDIUM.** True intra-kernel cross-WG reduce is **not
  possible** on Metal (no global workgroup barrier). But: at gemma decode
  shapes (kL ≤ 1024) NWG≤16 yields **identical throughput** to NWG=32
  (per `compute_nwg` doc, iter-100). At those kL we already have NWG=16; the
  reduce kernel still fires (NWG=16 > 1).
  - Sub-option H2a: **reduce-only-when-needed** — gate reduce dispatch on
    `kv_seq_len > threshold` where NWG benefit > reduce overhead. If we can
    drop NWG to 1 at short kL (set `compute_nwg` to 1 for kL ≤ N), kernel #5
    skipped entirely.
  - Sub-option H2b: **rolling-write reduce** — write final result during the
    last WG's epilogue. Requires atomic-add across WGs for S,M,so → infeasible
    on Metal at f32 precision (no f32 atomic ops with proper online-softmax
    semantics).
- **Dispatches saved/token:** Up to 1 per layer when NWG drops to 1, i.e.
  26–36/token. Conditional on kL.
- **Expected kernel-µs saved:** Small — reduce kernel is ~one simdgroup of
  work over `num_heads` rows. CPU encode + RAW barrier is the dominant cost
  (microseconds, not tens). Maybe 0.1–0.3 ms/token total at short kL.
- **Risk:** MEDIUM. compute_nwg is empirically tuned; dropping NWG hurts
  long-context throughput. Conditional schedule needs careful kL threshold
  measurement (the existing iter-119 `kv_seq_len > 512` policy is a good
  starting point).
- **Recommended action:** measure NWG=1 throughput at kL ∈ {64, 128, 256, 512}
  and adopt as default below the breakeven threshold.

### H3 — Fuse `fwht_sign_undo` into FA-vec-reduce or FA-vec output write
- **Concept:** Output FWHT-undo is element-wise in DV, with no cross-head or
  cross-position interaction. Apply during the global-memory write that today
  goes `dst4[rid * DV4 * NWG + ...] = so4[i] * inv_S` (line 703) when NWG=1,
  or during reduce kernel's `dst4[i] = reduced * inv_S` (flash_attn_vec.metal:396)
  when NWG>1.
  - Sub-option H3a: when NWG=1, fuse undo into FA kernel's epilogue (lines
    694–711). EPT=DV/32 register reload + sign-undo + fwht_simd + 1/√d
    normalize — same compute as kernel #6, all per-thread + simd shuffles.
  - Sub-option H3b: when NWG>1, fuse undo into reduce kernel's per-chunk write
    (line 396). Same per-thread + simd cost; one fewer dispatch + barrier.
- **Feasibility: HIGH (NWG=1)**, **MEDIUM (NWG>1)**.
  The FA kernel epilogue (line 694) is `if (sgitg == 0)` and `for (i = tiisg;
  i < DV4; i += NW)` — that's a single simdgroup writing one DV-vector. Adding
  an FWHT pass before write requires materializing all DV elements in
  per-thread registers (currently distributed across 32 threads × DV4/NW
  iters). For DV=256, NW=32, DV4=64: each thread writes 2 float4s = 8 floats.
  EPT=DV/32=8 → exactly matches; fwht_simd<8> can run on the same per-thread
  register window.
- **Dispatches saved/token:** 1 per layer × 26–36 layers = 26–36/token.
- **Barriers saved/token:** 1 RAW barrier per layer.
- **Expected kernel-µs saved:** Similar magnitude to H1 (mirror kernel,
  mirror cost). 0.5–1.4 ms/token.
- **Risk:** MEDIUM. The fused undo runs *after* the optional cross-NSG reduce
  (line 654–692) AND the 1/S normalize. Need to verify:
  (a) `so4[i] * inv_S` correctly multiplies sign-undo input;
  (b) FWHT and sign-undo are well-defined element-wise after `* inv_S`
      (yes — undo = sign_j * FWHT(x); both linear, commute with scalar mult).
  (c) DV (output dim) FWHT tables match the DK-rotation tables in the
      common DK==DV case (Gemma both 256; Qwen both 256; OK).
  Parity sweep required: 1e-4 abs/rel tolerance + 1000-tok coherence gate.
- **Recommended action:** prototype H3a first (NWG=1, simpler epilogue);
  if successful, H3b for NWG>1 reduce path.

### H4 — `hadamard_quantize_kv_fast` (K) + `hadamard_quantize_kv_hb_dual` (V) → one dispatch
- **Status:** `hadamard_quantize_kv_hb_dual` (line 864) **already** does both
  K and V in one dispatch (z-axis = stream selector). The fast (4-bit) K + HB
  (high-bit) V two-encoder pattern is the legacy path; production today on
  gemma4 HB-cache uses HB encoding for both K and V (`hadamard_quantize_kv_hb`
  for K, `hadamard_quantize_kv_hb_dual` for V — *or* one `_hb_dual` call
  encoding both).
- **Feasibility: HIGH for HB-only paths.** Check `inference/forward_decode.rs`
  call sites to confirm decode is already on the dual path. If gemma4 decode
  is calling `_fast` (4-bit K) + `_hb_dual` (V), there is an asymmetry: K is
  4-bit, V is 5/6/8-bit. Cannot trivially fuse into one kernel because they
  pack differently (nibble vs byte). But if the production config is HB for
  both, both K and V can flow through one `_hb_dual` dispatch.
- **Dispatches saved/token:** 1 per layer when symmetric HB (26–36/token).
- **Barriers saved/token:** 1 RAW barrier per layer (between K-encode and V-encode).
- **Expected kernel-µs saved:** the encoder is one simdgroup/head; CPU+barrier
  dominate. ~0.2–0.5 ms/token.
- **Risk:** LOW for HB-symmetric. The dual kernel is byte-for-byte verified
  ("Result is byte-identical to two `hadamard_quantize_kv_hb` dispatches at
  identical params", line 861–862).
- **Recommended action:** audit production decode call site; if currently
  doing K+V as two separate `_hb` dispatches, switch to the existing dual.

### H5 (additional, novel) — `hadamard_quantize_kv_hb_dual` ⊕ `fwht_sign_premult` cofusion (Q-encode in same dispatch)
- **Concept:** The K/V encoder kernel does (a) load → (b) sign pre-mult →
  (c) fwht_simd → (d) 1/√d normalize → (e) quantize → (f) write. The Q rotation
  kernel does (a) load → (b) sign pre-mult → (c) fwht_simd → (d) 1/√d normalize
  → (f') write back F32. Stages a–d are identical algorithms. If Q has the
  same head_dim as K/V (Gemma 256/256; Qwen 256/256), we can dispatch one
  kernel that processes 3 streams (K, V, Q) on z-axis ∈ {0,1,2}.
- **Feasibility: MEDIUM.** Z-axis already used for K/V selector; extending
  to 3 streams trivial. Q is NOT quantized — branch at step (e): if
  `kv_sel == 2u` skip quantize, write F32 to Q out buffer. Two write paths
  in one kernel — Metal handles fine.
- **Dispatches saved/token:** 2 (one K encode + one fwht_premult) → 1 = save 2
  per layer = 52–72/token. With H4+H5 combined, K+V+Q all in one dispatch.
- **Barriers saved/token:** 2 per layer (K/V→Q and Q→SDPA collapsed).
- **Expected kernel-µs saved:** if K+V dual + Q-FWHT both at ~0.5–1.0 ms each →
  1.0–2.0 ms/token additional on top of H1+H4.
- **Risk:** MEDIUM-HIGH. New kernel introduces additional shader, new
  binding layout, new test fixtures. Coherence gate sensitive — need byte-
  identity vs current 3-dispatch sequence under tight tolerance.
- **Recommended action:** defer until H1+H3+H4 measured; revisit only if the
  remaining 1.81× CPU-exec gap demands more aggressive fusion.

### H6 (additional) — Use Metal `MTLComputeCommandEncoder` `useResources` + pipeline-state caching to reduce CPU dispatch overhead
- **Concept:** Per-dispatch CPU overhead (Rust encode → buffer-arg-binding →
  setComputePipelineState → dispatchThreadgroups) is a significant chunk of
  the 1.81× CPU-exec gap. Peer's `kernel_flash_attn_ext` switch wrapper has
  a similar overhead per dispatch but does it *once* per attention op vs hf2q's
  *six* times.
- **Feasibility: MEDIUM.** Examine `encoder.rs` for per-dispatch overhead;
  consider batching the 6 dispatches' arg binding into a single setBytes
  block where possible. Won't reduce the structural barrier count but may
  shave 10–30% of CPU exec time.
- **Recommended action:** Worker B / future research, not Phase 7d.

---

## 4. Summary Table — Fusion Priority Ranking

| Hyp | Description | Feasibility | Dispatches saved /layer | Barriers saved /layer | Expected µs saved /token | Risk to coherence |
|-----|-------------|-------------|-------------------------|------------------------|--------------------------|-------------------|
| H1  | Q-FWHT into FA (fuse_fwht_pre=1) | **HIGH** (code exists) | 1 | 1 | 500–1400 µs | LOW (byte-identical comment) |
| H4  | K+V encode → one `_hb_dual` dispatch | **HIGH** (kernel exists) | 1 (if asymmetric today) | 1 | 200–500 µs | LOW (verified byte-identical) |
| H3a | Output-FWHT-undo fused into FA epilogue (NWG=1) | HIGH | 1 | 1 | 500–1400 µs | MEDIUM (new code path) |
| H3b | Output-FWHT-undo fused into reduce (NWG>1) | MEDIUM | 1 (when NWG>1) | 1 | 500–1400 µs | MEDIUM |
| H2a | Drop NWG=1 at short kL → skip reduce | MEDIUM | 1 (when applicable) | 1 | 100–300 µs | LOW (no algorithm change) |
| H5  | K+V+Q in one z-axis 3-stream encoder | MEDIUM | 2 | 2 | 1000–2000 µs (on top of H1+H4) | MEDIUM-HIGH |
| H2b | True cross-WG reduce in-kernel | infeasible | — | — | — | — |
| H6  | CPU-side dispatch batching | MEDIUM (Worker B) | 0 | 0 | unknown | LOW |

**Phase 7d Recommended Order:**

1. **H1** (flip `fuse_fwht_pre` default to 1) — code exists, lowest risk,
   highest expected ROI. Parity-gate at 1e-4 + 1000-tok coherence.
2. **H4** (audit + flip to dual K+V encoder if asymmetric today) — code exists.
3. **H3a** (NWG=1 path — fuse FWHT-undo into FA epilogue) — new kernel work
   but well-bounded; the existing FA kernel already has the epilogue scaffold.
4. **H2a** (NWG schedule tuning) — measurement-driven; complements H3.

**Combined ceiling (H1+H4+H3+H2a):** Per-token decode could shed
2–4 dispatches and 3–4 RAW barriers per layer. At Gemma 26 layers ×
~0.5–1.5 ms each (encoder kernel µs + Rust encode overhead + barrier wait),
this is **0.4–1.6 ms/token** total — roughly translates to **+3% to +12%**
decode tokens/sec on a 67 t/s baseline.

**Phase 7d does NOT propose Path E.** TQ-HB 3.94× memory savings stay intact;
all fusions are within the TQ-HB kernel family (re-arranging dispatch
boundaries, not changing the cache representation).

---

## 5. Coherence + Parity Gates (mandatory for any change)

- 1e-4 abs/rel tolerance vs current HEAD output (per-layer dump at FA output
  + per-head L2 norm).
- 1000-token char-by-char generation match at temp=0 (gemma4 APEX + qwen3.6 APEX).
- TQ-HB regression-pin: per-slot KV bytes unchanged (3.94× savings preserved).
- Production-runner bench: gemma4 decode ≥ HEAD t/s (no regression below).

