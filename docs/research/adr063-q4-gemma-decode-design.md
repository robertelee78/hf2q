# ADR-063 Q4-GEMMA-DECODE: exact width-4 decode kernels for Gemma 4 (dense, routed MoE, attention)

Status: design for approval, revision 2 (2026-11-04). Revision 2 applies the
ReviewGemmaDecode ACCEPT-WITH-CONDITIONS findings 1–8 and the root's
decisions on them. The root added decode attention to this story's scope as
lever A. §14 records how each finding was answered. No code, build or hardware
run backs this document. Every number below is quoted from a retained receipt
(with its path), computed from shapes and block sizes, or marked
`[INFERENCE]`.

Story: **Q4-GEMMA-DECODE** (new, v0.1.30; operator decision 2026-11-04 to add
four Gemma stories under ADR-063 Q4). mlx-native provides the kernels; hf2q
integrates them into the Gemma batched decode body. §12 is the ADR-063 story
text, ready to extract.

Sources:
- **Q4-GEMMA profile** (#660, PR #687):
  `docs/research/adr063-q4-gemma-profile-result.md` on branch
  `feat/q4-gemma-profile` (record commits `87d5df5c`/`ddf781b6`). The profiled
  binary was built from `7f295b38`. Evidence is under `$G`.
- **ADR-063** sections Q4, Q4-GEMMA, Q4-GEMMA-ATTN, and the Q1 numeric policy
  `hf2q.q1.numeric-policy.v1` (`docs/research/adr063-q1-numeric-policy.md`).
- **ADR-040** §0.21b/§0.21c (the `mul_mv_ext` lever and the bit-exact Q6_K
  `mN` kernel), and the K1 record
  `docs/research/adr063-k1-q8-multirow-disposition.md`.
- **hf2q** `main` at `ff4af4bd`.
- **mlx-native** `=0.15.16`, crates.io source, the current pin. `/opt/mlx-native`
  main is 0.15.17, with the same anchors.
- **The peer** source at `e86f5fd1`, the build the profile measured
  (`b128-e86f5fd`).
- **Host**: Apple M5 Max, 40-core GPU, 614 GB/s peak memory bandwidth (from
  `receipt.json` and system_profiler, per ReviewGemmaDecode).

Abbreviations: `$G` = `/Users/robert/.local/state/hf2q/v0130/q4-gemma`;
`$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`;
`BB` = `src/inference/models/gemma4/batched_body.rs`; `M:` = the mlx-native
0.15.16 crate source; `P:` = the peer source at `e86f5fd1`.

## 1. Decision summary

1. **Every change is Q1 Class A against stock.** Each new kernel writes the
   same bits, for every output element, as the stock 0.15.16 kernel today's
   decode uses for that tensor (§3).
   - The Gemma batched body stays bitwise equal to serial width-1 decode
     (ADR-040 S2/S3).
   - The candidate stays bitwise equal to the kill-switch arm, so task quality
     is inherited through `ExactTranscripts` (Q1 §3.6).
   - The peer's few-row MMA kernels change the reduction order, so they are
     not adopted (§4.6).
2. **D1: row-tiled K-quant `mN` (Q4_K, Q6_K).** Each SIMD group computes R0
   weight rows for up to 4 columns. One activation load and one `sumy` then
   serve R0 rows. It applies to q/k/v/o, dense gate/up and, independently,
   the Q6_K lm head.
   - Per element-column the dense kernels already outrun the expert kernels:
     1.00–1.19 T against 0.83 T element-columns per second (§2.2). D1 removes
     the duplicated per-row loads and adds, not the FMAs, so the expected
     speed-up is **1.3–2×** (§2.4).
   - Every adoptable saving rides on D1. A Q4_K R0 = 2 parity spike (S0)
     therefore runs before M0.
3. **D2: Q5_0/Q8_0 `mN` for the dense `ffn_down`.** Today's plain mat-vec
   streams the weight once per row (`grid.y = m`). D2 replaces it with clones
   of the stock per-row kernels that read the weight once.
4. **E1: expert-grouped `mv_id`.** It covers gate_up (Q4_K) and down
   (Q5_0/Q8_0).
   - The first (token, expert) pair of each distinct expert computes every
     pair that chose that expert, so each expert's weights are read once per
     step.
   - It admits up to 16 tokens × top-8 for the companion speculation story's
     verify rows.
   - Its saving depends on the real expert overlap, which the V2 diagnostic
     measures **before** E1 can be adopted.
5. **A: decode attention at width 4** (root decision, 2026-11-04).
   - Today's vec-hybrid kernel runs one threadgroup per query head. Every K
     row and every packed V row is therefore loaded and decoded once per query
     head: 2× on sliding layers and 8× on global layers.
   - A1 groups the query heads that share a KV head, with per-head arithmetic
     unchanged.
   - A2 runs mixed-bucket slots in one batched dispatch with per-slot
     parameters, removing today's per-slot fallback.
6. **Fusion.** Two epilogues are optional and decided by M0: dense gate+up as
   one dual-weight launch, and the GELU·mul epilogues. Router `mN`, batched
   routing and other dispatch-count reductions are not proposed: ADR-040
   measured them throughput-neutral (§2.4).
7. **Gates, in order:**
   - primitive bitwise parity;
   - M0 microbenchmarks, with an adopt/drop threshold per lever;
   - M1 in situ: GPU per step, with body and lm head reported separately;
   - M2, the four-agent A/B against the kill switch. It requires a median
     makespan ratio ≤ 0.96 with CI upper < 1.00, `ExactTranscripts`, and
     fairness ≤ 1.025.

   Stated plainly (§5): **the lowest adoptable case fails M2.** That case is
   D1 alone at its adopt threshold, about 0.42–0.48 s and a ratio of about
   0.97. A 0.96 ratio closes the Gemma Win only if the other three stories
   reach their high estimates; otherwise this story needs about 0.93. If M2
   misses, the root's preference is to re-budget the Win across the other
   Gemma stories (§13).
8. **Switch: `HF2Q_GEMMA_DECODE_KERNELS`.**
   - `0` is the kill switch, which is today's code path.
   - `1` selects the adopted exact routes. It refuses startup unless the
     reference routing flags hold (§4.8).
   - The default flips to `1` only after every gate passes.
   - New mlx-native policy fields are per lever, default off and
     `#[serde(default)]`, so no other family changes.

## 2. Evidence

### 2.1 What the profile measured

From #660 (`adr063-q4-gemma-profile-result.md` §2–§4):
- **Decode stage.** hf2q takes 9.203 s per agent against the peer MT's
  7.958 s (Δ +1.170 s, CI +0.961 to +2.012).
  - Makespan is 14.867 s against 12.477 s; a Win needs ≤ 11.85 s.
  - MT runs speculation, at 0.79 accepted per drafted and 2.34 tokens per
    verify step. Part of the decode gap is the peer's fewer target steps, not
    kernel speed. MT versus P0 is −0.48 s of decode.
- **GPU (L2 trace, 3 blocks).** Decode-only GPU busy is 7.67–8.24 s for hf2q
  and 5.40–5.71 s for the peer, i.e. +2.0–2.5 s per run.
  - hf2q's in-flight GPU idle is about 0.5 s *lower* than the peer's, so the
    gap is GPU work.
  - The peer's figure includes drafting and its wider verify steps.
- **Decode split (L3, `HF2Q_DECODE_CATSPLIT`).**

  | Class | Share |
  |---|---:|
  | MoE | 40% (gate/up 21.8%, down 15.3%, other 3.1%) |
  | Dense | 37.5% |
  | Attention | 17.8% |
  | Norms and embedding | 4.5% |

  The lm head and sampling are excluded.

The per-step view below comes from the worker-exit 12-bucket dump of the three
catsplit runs (`$G/profile/l3/0{21,23,25}-b0*-hc-*/server.stderr`, `[CATSPLIT]`
lines):
- **Steps per run.** `dense_q_proj` has 13,800 / 15,300 / 13,350 dispatches,
  one per layer-step, which gives 460 / 510 / 445 decode steps.
- **Mean width: 3.64 rows.** `dense_ffn` issues `3 + W` dispatches per
  layer-step in 2 command buffers (BB:1913, BB:2021). That gives W = 3.617 /
  3.576 / 3.724, so most steps are 4 wide.
- **Fixed command-buffer cost: at most 6 µs.** Catsplit commits one command
  buffer per category boundary, about 225,000–258,000 per run (232,760 in
  run b01). The smallest categories bound the fixed cost: `attn_pre` takes
  89.8 ms over 13,800 buffers, or 6.5 µs per buffer for three tiny
  dispatches.
  - The "net" column below subtracts 6 µs per buffer. That is an upper bound
    on the correction, so the true value lies between gross and net.
- **Distortion.** Catsplit runs 4.2× longer (#660 §4), so its shares are a
  relative ranking. Every class below comes from the same runs.
  - The net body is 16.85 ms per step.
  - The decode-only GPU per step (16.2–17.5 ms) *includes* the 605.6 MB lm
    head. The factor between net catsplit and production is therefore
    0.78–0.90, depending on the lm head's unmeasured cost (§5.1).

### 2.2 Bytes, work and achieved rates per class (mean over the 3 runs)

Definitions:
- Weight bytes per step come from shapes and block sizes (§2.5), summed over
  30 layers at W = 3.64. "Logical" bytes count every weight read a kernel
  issues, including repeats.
- An **element-column** is one weight element applied to one input column.
  It is the unit of arithmetic work: dense `mN` does W element-columns per
  weight element, and `mv_id` does one per (pair, element).

| Class (stock kernel at width 4) | Gross ms/step | Net ms/step | Logical weight MB/step | Weight GB/s (net) | Element-columns/s (net) |
|---|---:|---:|---:|---:|---:|
| q_proj (Q4_K `mN`) | 1.535 | 1.355 | 227.1 | 168 | 1.08 T |
| k_proj (Q4_K `mN`) | 0.757 | 0.577 | 89.2 | 155 | 1.00 T |
| o_proj (Q4_K `mN`) | 1.412 | 1.232 | 227.1 | 184 | 1.19 T |
| v_proj (13 Q6_K + 12 Q4_K `mN`) | 0.717 | 0.567 | 100.4 | 177 | 0.93 T (the Q6_K layers alone: about 0.86 T, per review) |
| dense_ffn: gate+up (Q4_K `mN`), router (F32, 4 × m=1), down (Q5_0/Q8_0 `Mv`, grid.y = W) | 3.044 | 2.684 | 200.7 + 157.4 + 560.0 | — (split not measured) | — |
| expert gate_up (Q4_K `mv_id`, one pass per pair) | 4.334 | 4.154 | 1,947.9 | 469 | 0.83 T |
| expert down (Q5_0/Q8_0 `mv_id`) | 3.043 | 2.863 | 1,493.4 | 522 | 0.61 T |
| attention: KV encode + vec-hybrid flash + FWHT undo (lever A) | 3.337 | 3.157 | — | — | — |
| lm head (Q6_K `mN`), outside catsplit | not measured | — | 605.6 | — | — |

Reading the table:
- **The expert kernels are mostly weight-streaming limited.** They run at 469
  and 522 GB/s logical against the 614 GB/s peak, although the rows share
  experts (§2.4).
- **The dense K-quant projections are not.** They move weight bytes at only
  155–184 GB/s, yet per element-column they already run at or above the
  expert kernels' rate. Their cost is the per-column work on short rows, not
  memory bandwidth. For the K = 2816 Q4_K rows, that work is the per-lane
  activation loads, `sumy` adds, nibble converts and FMAs.
- **D1 cannot reach 470 GB/s.** At four columns that would need about 3.0 T
  element-columns per second, 3.6× the expert kernel's arithmetic rate. D1's
  high case is capped at 2× (§2.4).

### 2.3 Where hf2q and the peer differ at Gemma shapes

Both engines run the same GGUF (`bf65b7e1…`). Per layer at four rows:

| Op | hf2q (default route) | The peer at `ne11 = 4` | Difference |
|---|---|---|---|
| q/k/v/o, dense gate/up, lm head (Q4_K, Q6_K) | `kernel_mul_mv_{q4_K,q6_K}_f32_mN_r1_4` reads the weight once. Each column's activations and `sumy` are recomputed per weight row: Q4_K runs 1 row per SIMD group and reloads `yl` per column; Q6_K runs 2 rows and caches `yl_c[R1][16]` (`M:src/shaders/quantized_matmul_ggml.metal:1573-1682`, `1159-1287`; route `M:src/ggml_capability.rs:633-650`) | Few-row MMA `kernel_mul_mv_mma_*`: 8×8 simdgroup tiles with 8–32 SIMD groups splitting K, and each weight dequantized once per lane for all columns (`P:ggml/src/ggml-metal/ggml-metal-common.cpp:143-154`; `rows_min` Q4_K 3, Q6_K 2 at `:121-141`; tiling `P:…/ggml-metal-device.cpp:849-907`; kernel `P:…/kernels/mul_mv_mma.metal:504-606`) | The peer pays per-column work once per 8-row tile. Not exact: the K split reorders every row's sum |
| dense `ffn_down` (Q5_0 ×16, Q8_0 ×14; K = 2112) | `Mv` with `grid.y = m` (`kernel_mul_mv_q5_0_f32`, `kernel_mul_mv_q8_0_f32_nr2`): the weight is streamed once per row (`M:src/ops/quantized_matmul_ggml.rs:1680-1767`) | MMA (2112 % 64 = 0 for Q5_0; Q8_0 uses the 32-wide block kind) | Weight read 4× against 1× |
| router (F32 2816→128) | 4 × `dense_matvec_f32` at m=1 (BB:1896-1911, 466-487) | `mul_mv_ext`, r1ptg = 4, 1 dispatch (`P:…/ggml-metal-ops.cpp:2742-2848`) | 1.44 MB read 4×. ADR-040 M4 measured amortization neutral (BB:464-480) |
| routing (softmax, top-8, scale) | per slot, 4 × `fused_moe_routing_f32` (BB:1952-1982) | fused `TOPK_MOE`, 1 dispatch (`P:…/ggml-metal-fusion.cpp:357-502`) | Dispatch count only |
| expert gate_up (Q4_K, 2816→1408 × 128) | `kernel_mul_mv_id_q4_K_f32`: 1 row per SIMD group, grid (704, 4·8) (`M:src/shaders/quantized_matmul_id_ggml.metal:1582-1680`; BB:2062) | `mul_mv_id` with N_R0 = 2, N_SG = 2, grid z = 32 (`P:…/ggml-metal-ops.cpp:3080-3138`; `P:…/ggml-metal-impl.h:57-58`) | **Neither engine groups tokens that share an expert** |
| expert down (704→2816) | `mv_id` Q8_0 (`…id_ggml.metal:561-623`) and Q5_0 (`:1027-1078`), 4 rows per SIMD group. Q8_0 `nr2` is off (`M:src/ggml_capability.rs:180`) | `mul_mv_id`; Q8_0 N_R0 = 2 × N_SG = 4 (a K split) | The K split is not exact, so it is not adopted. No grouping in either engine |
| `mm_id` (expert-grouped tiles) | Only above 32 tokens; Gemma decode pins `mv_id` (BB:2053-2061) | Only at `ne21 ≥ 32` (`P:…/ggml-metal-common.cpp:44-50`) | None at decode width |
| decode attention | `flash_attn_vec_hybrid_batched_dk{256,512}`, F16 K and byte-packed TQ-HB V. Grid (n_q, 16 heads, nwg), one **query head** per threadgroup; `kv_head = head / heads_per_kv` (`M:src/shaders/flash_attn_vec_hybrid.metal:772-822`; `M:src/ops/flash_attn_vec_hybrid.rs:394-429`). Batched only if every slot shares the (nwg, nsg) bucket, else per-slot dispatches (BB:888-901). Plus 2 KV-encode dispatches and 1 FWHT undo | FA vec, F16 K and F16 V. Grid (⌈ne01/Q⌉, n_head = 16, ne03·32), one head per threadgroup. (Q, NE) chosen from an Apple10 tuning table; reduce kernel (`P:…/ggml-metal-ops.cpp:3186-3193`, `:4074-4192`; `P:…/ggml-metal-tuning.cpp:781-793`) | Both re-read KV per query head. The KV representations differ (a Class C difference), so the peer kernel is a cost reference only |
| expert GELU·mul, weighted top-8 sum | `moe_swiglu_batch` (GELU-tanh), `moe_weighted_sum_seq`: 1 dispatch each | `geglu` (no GLU fusion), `MOE_REDUCE` fusion | Equivalent |
| norms, residuals | fused norm+add; fused post-FF norm2 + end-of-layer (BB:1801, 2236) | `NORM_MUL(_ADD)` fusions | Equivalent |
| per step | about 1,050 dispatches; concurrent encoder with conflict barriers; 4 async command buffers, one wait (BB:2505-2590) | about 36 per layer [INFERENCE], so about 1,080; concurrent encoder; 2 command buffers (`P:…/ggml-metal-ops.cpp:159-246`, `P:…/ggml-metal-context.m:486-512`) | None material |

### 2.4 Diagnosis

**Dense K-quant `mN` is bound by per-column work, not by weight bandwidth.**
How the stock Q4_K row kernel works (`…ggml.metal:1452-1560`):
- Each lane, for each super-block it visits, loads 32 activation floats,
  forms 4 `sumy` sums and does 32 nibble-convert FMAs.
- `mN` repeats this per column for each weight row. Only the scale decode and
  the weight bytes are shared.

An exact kernel cannot change the lane partition, because that fixes every
row's summation order. What D1 can do is let one column's activation loads
and `sumy` serve R0 rows. It removes the duplicated loads and adds, but not
the per-(row, column) FMAs and converts. [INFERENCE] That bounds D1 to about
1.3–2×. The Q6_K `mN` already shares activations across its 2 rows and still
runs about 0.86 T element-columns per second. That fits a limit on dequantize
or issue rate more than on loads, which is why the high case is capped at 2×.

ADR-040 §0.21c pointed the same way for small-N projections: "occupancy-aware
rework (cache only the per-block weight dequant, stream Y per column;
re-prove bit-identity) is the obvious next lever"
(`docs/adr/ADR-040-continuous-batching-reopen.md:790`). M0 reports both
weight GB/s and element-columns per second for every cell.

**Dense `ffn_down` streams its weight 4×.** That is 560 MB per step logical
against 154 MB unique. [INFERENCE] Part of the re-read may be served from the
system cache, since one layer's `ffn_down` is only 4.1–6.3 MB. M0 measures how
much.

**The four rows share experts that both engines read separately.** The Q1
routing witness on the accepted greedy144 dataset
(`$E/q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl`; 4
lanes × 144 positions × 30 layers, top-8 of 128) gives the number of distinct
experts U among the 4 lanes' 32 selections at one (layer, position):

| Rows | Lanes identical to lane 0 | Mean U (of 32) | p10 / p50 / p90 | Duplicate fraction 1 − U/32 |
|---|---|---:|---|---:|
| 128–143 (greedy continuation, decode-like) | 0 / 1,440 | 20.9 | 16 / 21 / 26 | 0.35 |
| control: each lane at an independent random position | — | 18.7 | — | 0.41 |
| 29–127 | 1,622 / 8,910 | 10.5 | 8 / 9 / 16 | 0.67 (lanes partly identical) |

Expert use is skewed. In rows 128–143, the 16 most-used of the 128 experts
take 72% of selections (per layer, 0.49–0.93).

Whether removing duplicate pairs saves time depends on whether duplicates
already hit the system cache:
- The expert kernels stream at 469–522 GB/s logical, against a 614 GB/s
  peak, which suggests they are limited by weight streaming.
- If duplicates are re-streamed from DRAM, E1 saves in proportion to the
  duplicate fraction. If the system cache already serves them cheaply, E1
  saves little.

There is no derivation for an in-between share, so §5 brackets E1 at 0 and at
1 − U/32. [INFERENCE] These are synthetic lanes, not the four coding agents.
V2 (§7.4) measures U on the real workload, and E1's adoption uses that
distribution.

**Decode attention re-reads KV per query head.** In
`flash_attn_vec_hybrid_batched_impl`, each threadgroup owns one
(query, head, split) and loads its KV head's F16 K rows and packed V rows
plus norms, then decodes V (`…flash_attn_vec_hybrid.metal:953-1091`).
`heads_per_kv` is 16/8 = 2 on the 25 sliding layers and 16/2 = 8 on the 5
global layers. Every K/V byte is therefore loaded and V-decoded 2× or 8× per
step. [INFERENCE] How much of the 3.16 ms per step this costs depends on how
the time splits between KV traffic and per-head arithmetic. M0 measures it at
the workload's context lengths (§7.2, §7.4).

**Dispatch count is not a cause.** hf2q issues about 1,050 dispatches per
step and the peer about 1,080 [INFERENCE]. ADR-040 §24 iter-J measured a
−209 dispatch/step reduction "THROUGHPUT-NEUTRAL (measured ~noise)"
(BB:2151-2164), and amortizing the F16/F32 weight reads was also neutral
(BB:477-480).

### 2.5 Shapes and types (q4_k_m artifact `bf65b7e1…`)

Architecture:
- 30 layers: 25 sliding (8 KV heads, head dim 256, window 1024) and 5 global
  (2 KV heads, head dim 512, V = K).
- Hidden 2816; 16 query heads; dense FFN 2112.
- 128 experts, top-8, expert FFN 704.
- Vocabulary 262,144, tied embeddings.

Inventory: Q4_K 192, Q5_0 32, Q8_0 28, Q6_K 14, F32 392
(`docs/research/adr063-f1-parity-validation-2026-10-23.md:46`). The per-role
types below are derived from `src/quantize/ggml_quants/standard_policy.rs`
(`:125-170`, `:330`, `:572`, `:653`), and they match the inventory counts
exactly:

| Role | K → N | Type | Bytes per layer |
|---|---|---|---:|
| attn_q | 2816 → 4096 (global 8192) | Q4_K | 6.49 MB (12.98) |
| attn_k | 2816 → 2048 (global 1024) | Q4_K | 3.24 MB (1.62) |
| attn_v (sliding only) | 2816 → 2048 | Q6_K ×13, Q4_K ×12 | 4.73 / 3.24 MB |
| attn_output | 4096 (8192) → 2816 | Q4_K | 6.49 MB (12.98) |
| ffn_gate, ffn_up | 2816 → 2112 | Q4_K | 3.35 MB each |
| ffn_down | 2112 → 2816 | Q8_0 ×14, Q5_0 ×16 | 6.32 / 4.09 MB |
| ffn_gate_inp (router) | 2816 → 128 | F32 | 1.44 MB |
| ffn_gate_up_exps (per expert) | 2816 → 1408 | Q4_K | 2.23 MB |
| ffn_down_exps (per expert) | 704 → 2816 | Q8_0 ×14, Q5_0 ×16 | 2.11 / 1.36 MB |
| token_embd = lm head | 2816 → 262,144 | Q6_K | 605.6 MB |

The Q8_0 layers are 0–2, 5, 8, 11, 14, 17, 20, 23 and 26–29.

KV per position and slot:
- Sliding: 8 heads × 256 × (2 B for F16 K + 1 B for packed V) plus norms,
  about 6 KiB.
- Global: 2 heads × 512 × 3 B plus norms, about 3 KiB.

## 3. Exactness contract (normative; the text of the linked mlx-native issue)

### 3.1 Reference kernels

The reference for each class is the stock 0.15.16 kernel used today when the
default routing policy holds (`M:src/ggml_capability.rs:168-184`). §4.8
refuses the new routes whenever the process policy differs on any flag listed
below.

| Class | Reference | Required reference flag | New entry point |
|---|---|---|---|
| Q4_K dense | `kernel_mul_mv_q4_K_f32` (m = 1) | — | D1 `kernel_mul_mv_q4_K_f32_rt{R0}_r1_{R1}` |
| Q6_K dense, including the lm head | `kernel_mul_mv_q6_K_f32_nr2` (m = 1; the stock `mN` clones it, ADR-040 §0.21c) | `dense_q6k_mv_nr2 = true` | D1 `kernel_mul_mv_q6_K_f32_rt{R0}_r1_{R1}` |
| Q5_0 dense | `kernel_mul_mv_q5_0_f32` (m = 1) | — | D2 `kernel_mul_mv_q5_0_f32_mN_r1_{R1}` |
| Q8_0 dense | `kernel_mul_mv_q8_0_f32_nr2` (m = 1) | `dense_q8_0_mv_nr2 = true` | D2 `kernel_mul_mv_q8_0_f32_nr2_mN_r1_{R1}` |
| Q4_K expert | `kernel_mul_mv_id_q4_K_f32` (per pair) | — | E1 `kernel_mul_mv_id_q4_K_f32_grouped_rt{R0}` |
| Q5_0 / Q8_0 expert | `kernel_mul_mv_id_q5_0_f32` / `kernel_mul_mv_id_q8_0_f32` (per pair) | `expert_q8_0_mv_nr2 = false` | E1 `kernel_mul_mv_id_{q5_0,q8_0}_f32_grouped` |
| decode attention, d256 and d512 | `flash_attn_vec_hybrid_batched_impl` and its reduce, run per slot at that slot's (nwg, nsg) | `HF2Q_BATCHED_FLASH` not `0`; the `HF2Q_HYBRID_NWG` and `HF2Q_TQ_NSG` overrides unset | A1 `flash_attn_vec_hybrid_batched_gqa{G}_dk{256,512}`; A2 per-slot-bucket batched variant |

All dense and expert classes also require `dense_decode_mv_ext = false`. The
kernel names are proposals; mlx-native may rename them.

### 3.2 Invariants (bitwise on the raw u32 of every F32 output)

1. **Per-element identity.** For every output (column c, row r), the bits
   equal those of one reference dispatch on column c alone. For attention,
   the reference is one slot's dispatch at that slot's own (nwg, nsg); the
   invariant covers the output O and every persistent buffer the step writes.
2. **Width invariance.**
   - Dense: holds for m ∈ 1..8. D1 and D2 tile columns in chunks of at most
     4. Stock `mN` tiles m = 5 as [5], 6 as [3, 3], 7 as [4, 3] and 8 as
     [4, 4] (`M:src/ops/quantized_matmul_ggml.rs:2135-2148`); D1 may tile
     5 as [3, 2].
   - Grouped expert kernels: hold for n_tokens ∈ 1..16 at top-8 (gate_up)
     and n_tokens · top_k ≤ 128 at top-1 (down).
   - Attention: holds for n_q ∈ 1..16.
   - No tiling, threadgroup shape, R0, NSG or G value the dispatcher picks
     may change any output's bits.
3. **Column independence and permutation.**
   - Column c's output depends only on column c's input and the weights,
     never on other columns' count, order or values. A neighbouring column
     holding NaN or ±∞ must not change it.
   - Permuting input columns permutes the output columns.
   - For attention, the same holds per slot and per query head: grouping
     heads never couples their values.
4. **Grouping invariance (E1).** The bits written for pair p depend only on
   the weights of `ids[p]` and p's input row. They do not depend on which or
   how many other pairs chose that expert, or on pair order. This must hold
   for:
   - all experts distinct;
   - all pairs sharing one expert;
   - mixed patterns;
   - out-of-range ids, which produce exactly the stock
     `poison_invalid_expert_id` output for their pair
     (`…id_ggml.metal:586-587`, `1047-1048`, `1606-1607`). An invalid id
     never joins a group.
5. **Tails and views.** The Gemma N values 704, 1024, 1408, 2048, 2112, 2816,
   4096, 8192 and 262,144 divide evenly into every proposed tile, so they
   exercise no row tails. The tests therefore also cover:
   - synthetic N with remainders against every tile size (for example 7, 1027,
     2050, 2817);
   - K tails: Q5_0 at K = 2112 is 66 blocks, so lanes run 4 or 5 iterations;
     expert K = 704 is 22 blocks;
   - attention KV lengths at every nwg/nsg bucket edge and ring wrap;
   - `slice_view` and `BufferWithOffset` views at non-zero offsets;
   - no writes outside the output views.
6. **Compile flags.** Each new kernel is compiled with **exactly the same
   flags as its reference kernel**, on both metallib paths: precompiled
   (`MLX_PRECOMPILED_METALLIB=1`, the default) and runtime source compile
   (`=0`).
   - In 0.15.16 the quantized matmul, `mv_id`, GELU and flash-attention
     kernels all build with Metal's default math mode. Fast math is disabled
     only for `deepseek_tail_rope*` (`M:build.rs:189-195`,
     `M:src/kernel_registry.rs:116-118, 2001-2003`).
   - New kernels must **not** add `-fno-fast-math`, a safe math mode or
     precise functions. Doing so would change contraction and break parity.
   - On each path, the new kernel equals the reference on that same path.
     The comparison between the two paths is recorded.
7. **Operand-form rule (implementation guidance; the tests decide).**
   - Each per-(row, column) or per-(head, key) expression must be a literal
     clone of the reference: the same cached `yl`/`yh` operand form, `sumy`
     accumulation, `acc` lanes, scale decode, `sumf` expression, softmax
     update, `simd_sum` and threadgroup-reduction order.
   - Interleaving rows, columns or heads is allowed. Reassociating any sum is
     not.
   - Precedent: caching activations in another register form flipped Apple's
     FMA-contraction decision and caused a 1-ULP drift (ADR-040 §0.21c, Q6_K
     `nr2` against plain `mv`).

### 3.3 Compiler dependence

Bitwise equality between two different kernels holds for the compiler that
built them, not by construction across compilers. A macOS or Metal toolchain
update can change contraction in one kernel but not the other. Every
receipt therefore records:
- the macOS build;
- the Metal compiler version (`xcrun metal --version` for the precompiled
  metallib, and the OS for the runtime path);
- the metallib SHA-256.

Parity tests T-N1 to T-N8 rerun on every pin bump and every host OS update
before a Gemma timed run.

### 3.4 Q1 classes

- **Against stock (`HF2Q_GEMMA_DECODE_KERNELS=0`): Class A.** Every captured
  byte must be equal: logits, token ids, KV, expert ids and their order, and
  cursors (Q1 §3.1 item 4). One differing byte is `DivergedExactInvariant`,
  which blocks the route. No tolerance applies.
- **Batched against serial (ADR-040 S2/S3 byte identity): Class A**,
  unchanged.
- **Quality inheritance.** A Class A route inherits task quality (Q1 §3.6).
  This story's A/B therefore needs `ExactTranscripts`, not a paired quality
  campaign. The Q4 closing gates still run on the combined candidate (§8.4).
- **Not allowed here.** Each of the following is Class B against stock:
  - the peer's MMA few-row kernels;
  - `mul_mv_ext` for K-quants;
  - the Q8_0 `nr2` K split for experts;
  - `mm_id` at decode widths;
  - any change to the KV representation or to the NWG/NSG bucket heuristics;
  - any kernel whose per-row order differs from §3.1.

  Gemma's routing gaps make Class B routes route-sensitive (Q1 §2.2:
  8th-versus-9th softmax gap median 7.5e-4; `RouteSensitive` on every measured
  Gemma alternate route).

## 4. Proposed changes (LLD)

### 4.1 S0: pre-M0 parity spike (D1, Q4_K, R0 = 2)

Every adoptable saving rides on D1 (§5). Before the full M0 sweep, mlx-native
builds only the Q4_K R0 = 2 kernel and runs T-N1's Q4_K cells on both
metallib paths:
- every Gemma Q4_K shape;
- m = 1..8;
- the synthetic tails.

It also records one M0 cell (q_proj, m = 4). If parity fails and an
operand-form fix does not clear it, D1 is redesigned before any other lever is
built. A D1 that cannot be exact leaves too little adoptable saving for M2
(§5.1).

### 4.2 D1: row-tiled K-quant `mN` (Q4_K, Q6_K)

Structure (Q4_K; the Q6_K variant wraps the `nr2` per-row body the same way):

```
threadgroup = (NSG, 32, 1); SIMD group s owns rows r0 + s·R0 … + R0 − 1
for block i = ix; i < nb; i += 4:              // stock lane partition, unchanged
    for each own row j < R0: load q1/q2 bytes, scales → sc8[j], dall[j], dmin[j]
    for column c < R1:
        load yl[16], yh[16], sumy[4] for column c   // stock expression, once per (i, c)
        for row j < R0:
            acc1 = stock acc1 loop over (yl, yh, q1[j], q2[j])
            sumf[j][c] += stock sumf expression (dall[j], dmin[j], sc8[j], acc1, sumy)
simd_sum(sumf[j][c]) for each (j, c); lane 0 stores rows < ne01, columns < ne1
```

- **What is shared.** The activation loads and `sumy` for a (block, column)
  serve R0 rows. The stock kernel recomputes them per row; the recomputation
  is deterministic, so the values are equal. The per-(row, column) FMAs and
  nibble converts are unchanged, which is why the gain is capped (§2.4).
- **Register budget** [INFERENCE]. At (R0, R1) = (4, 4), one lane holds
  `yl`/`yh` (32), `sumy` (4), `sumf` (R0·R1 = 16), and per-row quant bytes and
  scales (about 4·R0). That is about 70–90 registers.
  - Only one column's activations are live at a time. The stock Q4_K `mN`
    already reloads `yl` per column.
  - The Q6_K stock `mN` keeps `yl_c[R1][16]`. D1's Q6_K variant streams one
    column at a time instead.
  - R1 stays at most 4 per tile, because of ADR-040's spill cliff at R1 ≥ 6.
- **Geometry search (M0 decides; geometry never changes bits).** R0 ∈
  {1, 2, 4}, NSG ∈ {2, 4, 8}, R1 ∈ {1, …, 4}, with optional single-column
  activation staging in threadgroup memory (2816 floats = 11 KiB). The
  chosen geometry per (type, N, K, column tile) is frozen into a static table
  in mlx-native code.
- **Where it applies.** q, k, v, o, and dense gate and up at width ≥ 2, and
  optionally at m = 1 (M0 decides; stock m = 1 stays the reference). The lm
  head (Q6_K, 262,144 rows, `batched_head.rs:244-322`) is adopted or
  dropped separately (§4.8).
- **F1: dual-weight gate+up (optional).** Gate row r and up row r become the
  R0 rows of one SIMD group, read from two tensors. That saves the shared
  loads and one dispatch. It is adopted only if it beats two D1 launches in
  M0.

### 4.3 D2: Q5_0 and Q8_0 `mN` (dense `ffn_down`)

- **Q5_0.** Clone `kernel_mul_mv_q5_0_f32` (`…ggml.metal:407-459`: 4 rows per
  SIMD group, 2 SIMD groups). Wrap the `sumy[2]`/`yl[16]` packing and the
  per-row `block_q5_0_dot_y` in a column loop of at most 4.
- **Q8_0.** Clone `kernel_mul_mv_q8_0_f32_nr2` (`:837-934`: 2 rows, with 4
  SIMD groups splitting K).
  - The K split is the reference's own order, so it stays exact.
  - Each column gets its own `shmem_rows` slice, 256 B per column.
  - The two barriers and the final `simd_sum` are unchanged.
- **Grid.** `(⌈N/rows_per_tg⌉, ⌈m/R1⌉, 1)`, so the weight is read once per
  column tile.

### 4.4 E1: expert-grouped `mv_id` (leader pairs)

**Leader rule.** At pair index p (flat `token·top_k + slot`), let
`e = ids[p]`. Pair p is a *leader* if `e` is valid and no earlier pair
`q < p` has `ids[q] == e`.
- Top-k ids are distinct within a token, so a group has at most n_tokens
  members.
- An invalid id is always its own leader and runs the stock poison path.

**Kernel.**
- The grid is the stock one, `(row tiles, n_tokens·top_k)` threadgroups.
- A non-leader threadgroup reads `ids[0..p]`, at most 512 B and uniform
  across the threadgroup, then returns.
- A leader collects its members: p itself and every `q > p` with
  `ids[q] == e`. It loads the weight rows once and, for each member, runs the
  reference per-pair arithmetic:
  - the input is that member's token row (gate_up) or pair row (down);
  - the output goes to exactly the row the stock kernel writes.
- Members are processed in column chunks of at most 4.
- **Variants.** For Q4_K gate_up, R0 ∈ {1, 2, 4} rows per SIMD group (the
  peer uses 2). The Q5_0 and Q8_0 down kernels keep their stock 4 rows per
  SIMD group.

**Alternatives and edges.**
- **No map pass by default.** The leader scan adds no dispatch, buffer or
  barrier. M0 also times a compact-grid variant with one map dispatch per
  layer, shared by gate_up and down. Both variants are exact.
- **Load balance.** A group with c members costs c column passes. M0
  includes a skewed pattern for this.
- **Width.** Up to 16 tokens × top-8 (gate_up) and 128 pairs (down). That
  covers the speculation story's verify rows (4 slots × up to 4 rows), as its
  design requested (DesignGemmaSpec, 2026-11-04). Above 32 tokens, `mm_id`
  and its threshold are untouched. The prefill MoE design does not change
  `mv_id` (DesignGemmaPrefillMoe, 2026-11-04).
- **F2: optional GELU epilogue on gate_up.** Gate row r (column r) and up
  row r + 704 (column r + 704 of `moe_swiglu_batch`, which places gate at
  0–703 and up at 704–1407) become one SIMD group's two rows. The epilogue
  applies the `moe_swiglu_batch` expression, GELU-tanh(g)·u
  (`M:src/shaders/moe_dispatch.metal:145-163`). It is adopted only if it is
  bitwise equal (T-N6) and M0 shows a gain.

### 4.5 A: decode attention at width 4 (Class A against today's vec-hybrid route)

- **A1: GQA-grouped vec-hybrid.** One threadgroup per (query, KV head,
  split), computing G query heads that share that KV head, with
  G | `heads_per_kv` (sliding: 2; global: 2, 4 or 8).
  - Shared across the G heads: each K row (F16) and each packed V row plus
    norms is loaded once, and V is decoded once. The decode is a
    deterministic function of the bytes, so every head sees the same values.
  - Per head, unchanged: each head's QK dot, softcap and scale, online-softmax
    update, V accumulation and cross-SG/nwg partial layout are a literal
    clone of today's per-head code. They write the same partial buffers, so
    the reduce kernel is unchanged.
  - Threadgroup memory grows with G. Today it is
    `pk + nsg·(sh + 2·pv)` halves (`M:src/ops/flash_attn_vec_hybrid.rs:387-388`),
    and it must stay within 32 KiB. [INFERENCE] G = 8 at d512 may not fit,
    in which case M0 picks G ∈ {2, 4} for global layers.
- **A2: mixed-bucket batched flash.** Today any step whose slots straddle an
  (nwg, nsg) bucket falls back to per-slot dispatches (BB:888-901,
  1118-1345).
  - A2 passes each slot's own (nwg, nsg) in a per-slot parameter array, and
    each slot's threadgroups run its solo split-K and SIMD-group count. It is
    exact by construction, and the bucket heuristics are unchanged.
  - It is adopted only if V2 shows mixed-bucket layer-steps at 10% or more
    and M0 shows a gain.
- **Out of scope:** the KV representation, the NWG/NSG heuristics, KV-encode
  fusion, and warm/prefill attention (#668).

### 4.6 Not proposed, and Class B alternatives

| Candidate | Status |
|---|---|
| Router F32 `mN` (1 dispatch instead of 4) | Not proposed. ADR-040 M4 measured F16/F32 amortization neutral (BB:464-480) |
| Batched per-token-identical routing | Not proposed. Dispatch count only, and ADR-040 §24 measured −209 dispatches/step neutral (BB:2151-2164) |
| RMSNorm + matvec fusion; residual add in the O-projection epilogue | Not proposed. Neither engine benefits; hf2q already fuses norm + add |
| Per-step `BatchedDecodeBuffers` allocation (BB:2390-2455) | Not proposed. Host cost; host idle is not the gap |
| The peer's few-row MMA (dense), `mul_mv_ext` for K-quants (ADR-040 §0.21b: +24% at N = 8, not exact), Q8_0 expert `nr2` (K split), the peer's FA vec | Class B against stock, not built. §13 decision 2 records when an MMA spike may be opened |

### 4.7 mlx-native API (additive; existing entry points and defaults bit-identical)

- **`GgmlRoutingPolicy`** (`M:src/ggml_capability.rs:145-159`; `Copy`,
  `deny_unknown_fields`) gains one `bool` per lever, each
  `#[serde(default)]` and default `false`:
  - `dense_row_tiled_q4k`, `dense_row_tiled_q6k`, `dense_row_tiled_m1`
    (D1);
  - `dense_mn_q5_0`, `dense_mn_q8_0` (D2);
  - `expert_grouped_q4k`, `expert_grouped_q5_0`, `expert_grouped_q8_0`
    (E1).
- The fields stay `Copy`. Geometry lives in a static per-shape table in code,
  never in the policy, and the dispatch trace records the geometry used.
- `ggml_routing_policy_from_environment()` (`M:src/ggml_routing_policy.rs:20`)
  leaves the new fields `false`; they are never resolved from the
  environment. `combine_routing_policies` (`:36`) carries each new field.
- `plan_dense_auto_route` and the expert planner honour a field only when it
  is set, the reference flags of §3.1 hold, and the shape is in the frozen
  table. `MM_ROUTING_THRESHOLD = 8` and `MM_ID_ROUTING_THRESHOLD = 32` are
  unchanged.
- Callers pass the policy through the existing
  `quantized_matmul_ggml_with_policy`
  (`M:src/ops/quantized_matmul_ggml.rs:627`) and
  `quantized_matmul_id_ggml_mv_with_policy`
  (`M:src/ops/quantized_matmul_id_ggml.rs:360`).
- Attention is a new explicit entry point next to
  `flash_attn_vec_hybrid_batched`, with G and per-slot buckets as explicit
  parameters. It is not a routing-policy field.
- Every new kernel is registered on both metallib paths, including
  runtime-source registration; P4 §2.3 records a prior gap there.

### 4.8 hf2q integration

- **Switch: `HF2Q_GEMMA_DECODE_KERNELS`.** It is read once at model load.
  - `0`, the default until promotion, is today's code path, unchanged: the
    kill switch.
  - `1` selects the adopted routes.
  - Any other value is a startup error.
- **Base policy.** With `1`, Gemma captures
  `base = mlx_native::ggml_routing_policy_from_environment()` once at load.
- **Refusal.** Startup fails, naming the flag, unless `base` holds the §3.1
  reference flags:
  - `dense_q6k_mv_nr2 = true`;
  - `dense_q8_0_mv_nr2 = true`;
  - `expert_q8_0_mv_nr2 = false`;
  - `dense_decode_mv_ext = false`;
  - the attention overrides unset.

  Without this check, for example, `HF2Q_Q8_0_MV_NR2=0` would silently change
  the reference.
- **Three derived policies,** each `base` plus the adopted flags:
  - `body_policy`: D1 and D2 for q/k/v/o, gate/up and down;
  - `lm_head_policy`: `dense_row_tiled_q6k` only if the lm head was adopted;
  - `expert_policy`: E1.

  They are threaded through:
  - a new `dispatch_qmatmul_with_policy` (`src/serve/forward_mlx_shared.rs`;
    `dispatch_qmatmul` takes no policy today), used by
    `dispatch_dense_rowident` (BB:451) and `encode_lm_head_into`
    (`batched_head.rs:244`);
  - `dispatch_expert_matmul_id_mv` (`src/inference/expert_dispatch.rs:90-109`).
- **Attention.** The A1/A2 calls replace the batched and fallback flash
  calls in BB:922-1345 when adopted.
- **Scope.** Other families and the serial path (`HF2Q_BATCHED_BODY=0`) keep
  the default policy. Weight types outside §3.1 keep their stock route, the
  `joint_decode_routes.rs` pattern (Q1 §3.1 item 4).
- **Diagnostics: `HF2Q_GEMMA_ROUTING_OVERLAP=1`** (§7.4). This is hf2q-only,
  lands before the native release, and is off by default with no cost when
  off.
- **Q5 case surfaces.** `scripts/four_agent/q5_cases/gemma4.json` declares
  the variables `GEMMA_DECODE_KERNELS` and `GEMMA_ROUTING_OVERLAP` (empty
  defaults) and the serve env entries `DECODE_KERNELS` and
  `ROUTING_OVERLAP`. The launcher forwards them as
  `${DECODE_KERNELS_ENV:+HF2Q_GEMMA_DECODE_KERNELS=…}` and
  `${ROUTING_OVERLAP_ENV:+HF2Q_GEMMA_ROUTING_OVERLAP=…}`, the same pattern
  as `WARM_TILED_ATTENTION`.
  - `q5_case.py:152-154` rejects any `HF2Q_Q5_*` override that names an
    undeclared variable, so both must be declared.
  - `scripts/four_agent/test_q5_calibration.py` gains the matching
    stock-absent and forwarding assertions (pattern at `:495-501`), plus
    case-file notes.
- **Other surfaces.** The banner and runtime-control report print the
  switch, the adopted lever set and the base reference flags. Catsplit
  buckets are unchanged, so profiles stay comparable.
- **Comment hygiene.** The BB:1995-1997 comment says `down_proj` is F16 and
  runs per row. On this artifact it is Q5_0/Q8_0 and takes the batched `Mv`
  route; the integration commit corrects the comment.

## 5. Expected savings (estimates, not measurements)

### 5.1 Per lever

Scale:
- 472 decode steps per run (mean of 445–510).
- Production class time is net catsplit time × f, with f between 0.78 and
  0.90. f is 0.78 if the lm head costs about 3.1 ms per step (Q6_K `mN` at
  about 195 GB/s), and 0.90 if it costs about 1.3 ms. The low column below
  uses 0.78; the high column uses 0.90.
- Dense gate+up are estimated at the pooled q/k/o rate, 1.16 ms per step.
- Down is estimated at 1.19 ms per step, from the rest of `dense_ffn` split
  by logical bytes.

"At threshold" is the smallest saving that still passes that lever's §7.5
adopt threshold. Below it, the lever is dropped and saves 0.

| Lever (net ms/step now) | Below threshold | At threshold | High | Basis |
|---|---:|---:|---:|---|
| D1 q/k/v/o + gate/up (4.89) | 0 | 1.25× → 0.42 s/run | **2× cap** → 1.04 s/run | §2.4: removes shared loads and `sumy`, not FMAs |
| D1 lm head (unmeasured: 1.3–3.1 ms/step) | 0 | 1.25× → 0.12–0.29 s/run | 2× → 0.30–0.73 s/run | Measured in M0 and M1 before §5 is relied on |
| D2 dense down (≈ 1.19) | 0 (the 25% case misses ≤ 0.67) | 1.5× → 0.15 s/run | unique bytes at 470 GB/s → 0.37 s/run | System-cache share of re-reads unknown |
| E1 experts (7.02) | 0 (duplicates already cache-served) | 0.85× → 0.39 s/run | time ∝ distinct pairs, d = 0.35 → 1.04 s/run | No derivation for an in-between share; V2 replaces d |
| A attention (3.16) | 0 (per-head arithmetic bound) | 0.85× → 0.17 s/run | [INFERENCE] 40% → 0.54 s/run | Sliding 2× and global 8× KV re-read (§2.4) |

Totals of decode GPU time per run:

| Case | Saving | Ratio | M2 (≤ 0.96) |
|---|---:|---:|---|
| Lowest adoptable (D1 body alone, at threshold) | 0.42–0.48 s | ≈ 0.97 | **Fails** |
| All levers at threshold | ≈ 1.25–1.6 s | ≈ 0.89–0.92 | Passes |
| All levers high | ≈ 3.7 s | — | Passes; exceeds the measured 2.0–2.5 s gap to the peer |

The high case exceeds the gap because the peer's MT figure includes its
speculation, and E1 and A remove work the peer also does.

[INFERENCE] **Makespan.** Joint decode steps are GPU-bound: in-flight idle is
small and the host sample loop is 7% of `decode_batch` (#660 §4). A
GPU-second saved in a 4-wide step therefore saves about one wall second for
every active agent at once. Makespan savings are taken as roughly equal to
decode GPU savings, less the width-1 tail steps.

### 5.2 Against the Win budget

The Win needs 14.87 s → ≤ 11.85 s, about 3.0 s. The #660 §7 estimates for the
other levers:

| Lever | Estimate |
|---|---:|
| #668 precise warm attention | 1.2–1.4 s |
| Cold-turn admission | 0.4–0.5 s |
| Serving speculation | 0.4–0.5 s |
| **Together** | **2.0–2.4 s** |

- **This story's share.** It needs ≥ 0.6 s if those three reach their high
  estimates, and about 1.0 s (ratio ≈ 0.93) if they reach their low ones.
- **#660's own estimate** sized decode kernels at 0.3–0.7 s, and decode
  attention was unowned then.
- **M2's ≤ 0.96 is a promotion gate, not the full budget.** If M2 lands
  between 0.93 and 0.96, the remaining Win budget is re-budgeted across the
  other Gemma stories (§13 decision 2).
- **Interactions.** Speculation removes target steps, which shrinks D1, D2
  and A in proportion. Wider verify rows raise E1's share. Only the combined
  candidate gives a valid total (§8.4).

## 6. Tests

### 6.1 mlx-native primitive parity (model-free, GPU; both metallib paths)

These follow the style of `tests/q4k_mv_mn_byte_parity.rs` and
`tests/adr_040_q6k_mv_mN_byte_parity.rs`.

Inputs are seeded:
- weights are valid blocks, including zero and subnormal half scales;
- activations include ±0, subnormals, large magnitudes, and one NaN/∞ column
  (for §3.2 item 3);
- KV includes ring-wrapped positions.

Comparisons are u32 equality against the §3.1 reference on the same path:

| Test | Covers |
|---|---|
| T-N1 `gemma_dense_row_tiled_q4k_q6k_bitwise` | Every §2.5 Q4_K/Q6_K shape, including the lm head at N = 262,144; synthetic tail N (§3.2 item 5); m = 1..8; every (R0, NSG, staging) cell of M0. S0 is its Q4_K R0 = 2 subset |
| T-N2 `gemma_dense_q5_0_q8_0_mn_bitwise` | K = 2112 → N = 2816, K = 704 → N = 2816, tail N; m = 1..8 |
| T-N3 `gemma_expert_grouped_mv_id_bitwise` | gate_up Q4_K (n = 1408, k = 2816) and down Q5_0/Q8_0 (n = 2816, k = 704); n_tokens 1..16; patterns: all distinct, all one expert, the §2.4 witness rows replayed, the V2 patterns, random, and invalid ids at first/middle/last position; leader-scan and map variants |
| T-N4 `gemma_decode_kernels_permutation_and_isolation` | Column, slot and head permutation; a NaN/∞ column does not perturb its neighbours; views at non-zero offsets; bytes outside the views unchanged |
| T-N5 `gemma_decode_kernels_metallib_paths` | Precompiled and runtime outputs recorded for every T-N1, T-N2, T-N3 and T-N8 cell, with toolchain identity (§3.3) |
| T-N6 `gemma_gelu_epilogue_bitwise` (only if F1/F2 are adopted) | Epilogue equals `fused_gelu_mul` / `moe_swiglu_batch` applied after the stock products |
| T-N7 `routing_policy_default_unchanged` | The default policy and `ggml_routing_policy_from_environment()` plan exactly the stock routes for every type × m; serde round-trip of old receipts (missing fields default to `false`); `combine_routing_policies` covers every field |
| T-N8 `gemma_decode_attn_gqa_and_mixed_bucket_bitwise` | d256 sliding and d512 global; KV lengths 1, 31, 32, 511, 512, 513, 1024 (ring wrap), 2048, 4096 and every `compute_nsg` edge; n_q 1..16; G ∈ {1, 2, 4, 8} where it fits; A2 with mixed buckets, compared against per-slot solo dispatches |

### 6.2 hf2q model-free

- Switch parsing (`0`, `1`, any other value rejected) and the banner text.
- Refusal when any §3.1 reference flag differs.
- Derivation of `body_policy`, `lm_head_policy` and `expert_policy` from a
  captured base.
- Refusal for weight types outside §3.1.
- Q5 case-file declaration and forwarding tests (§4.8).

### 6.3 hf2q real-artifact Class A gates

These are ignored tests, one process each, run with
`--exact --ignored --test-threads=1`:

| Gate | What it proves |
|---|---|
| A1, new: `gemma_decode_kernels_match_stock_bitwise` | One process decodes the same four ragged lanes for ≥ 256 steps under the stock policy and under the candidate policy. Each step must match bitwise: logits, sampled token, F16 K and TQ-HB V cache bytes, and per-layer router ids and weights (stage capture at BB:1986-1993). Covers sliding and global layers, widths 4 → 3 → 1 as lanes finish, one mid-run admission, and mixed-bucket steps |
| A2, with `HF2Q_GEMMA_DECODE_KERNELS=1`: `slot_aware_n1_byte_equivalent_to_serial_slot_aware`, `slot_aware_n4_per_slot_parity_vs_serial`, `slot_aware_n4_batched_body_same_prompt_parity_vs_serial`, `slot_aware_n8_per_slot_parity_vs_serial`, `slot_aware_staggered_eviction_no_peer_perturbation` (`src/serve/api/engine.rs:36956, 37831, 38061, 38191, 39642`) | Batched still equals serial, in a release build |
| A3: `gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant` (`engine_gemma4_tiny_prefill_tests.rs:16`) | Width-8 decode followed by a tiny cold prefill is repeat-invariant |
| A4: `q1_gemma_actual_cohort_endpoint_trace` (`src/quality/adr063_invariance/gemma_actual.rs:183`) under `0` and `1`, two processes, then `q1_decision_policy.py exact` | `ExactEqual` between the arms over the T2 cohort store |

## 7. Microbenchmark plan (M0; model-free; root GPU lane; quiet host)

### 7.1 Harness

The bench is `benches/bench_gemma_decode_w4.rs` in mlx-native (`harness = false`).

Timing:
- GPU time comes from command-buffer `GPUStartTime`/`GPUEndTime`, with 30
  dispatches per command buffer (one per "layer") so the fixed
  command-buffer cost does not inflate it as it inflates catsplit.
- Each cell runs 10 warm-up and at least 50 timed iterations and reports the
  median and p10/p90.

Setup:
- **Cold weights.** 30 distinct weight buffers (and KV caches) per shape are
  cycled, so a cell's working set exceeds the system cache. Activations are
  fresh each iteration.
- **Roofline.** A streaming read over ≥ 2 GiB gives achievable bandwidth
  against the 614 GB/s peak.
- **Metallib.** Timings use the precompiled metallib, the production default.

Each cell writes a JSON receipt with:
- shape, type, m / pattern / KV length, kernel and geometry;
- median ns;
- logical and unique bytes, **both** weight GB/s and element-columns per
  second (§2.2);
- the mlx-native commit, metallib SHA-256, host, macOS build, Metal compiler
  version, and the HOST-LOCK holder.

### 7.2 Cells

| Family | Shapes | m, patterns or KV lengths | Arms |
|---|---|---|---|
| D1 | q 2816→4096 and →8192; k →2048 and →1024; v Q4_K and Q6_K 2816→2048; o 4096 and 8192→2816; gate/up 2816→2112 | m = 1..4, plus 5..8 as non-regression | stock `mN` (stock m = 1 at m = 1); D1 over (R0, NSG, staging); F1 |
| lm head | Q6_K 2816→262,144 | m = 1..4, plus 8 | stock `mN`; D1 |
| D2 | 2112→2816, Q5_0 and Q8_0 | m = 1..4, plus 5..8 | stock `Mv` with grid.y = m; D2 |
| E1 | gate_up Q4_K 2816→1408 × 128; down Q5_0/Q8_0 704→2816 × 128 | n_tokens = 4 with four patterns: U = 32; the V2 empirical replay; U = 8 (each row picks the same 8 experts, every expert shared by all 4 rows); skewed (one expert in all 4 rows, the rest distinct). Also n_tokens = 8 and 16 (verify widths) | stock `mv_id`; E1 leader-scan and map variants, R0 ∈ {1, 2, 4}; F2 |
| A | d256 sliding (window 1024) and d512 global | n_q = 4 (and 1, 8, 16); KV lengths from the V2 histogram (and 512, 1024, 2048, 4096); uniform and mixed buckets | stock batched / per-slot flash; A1 by G; A2 |

### 7.3 Peer reference (M0-P; approved by the root, reference only, not a gate)

M0-P runs the peer's backend-op performance mode (`test-backend-ops perf`)
for `MUL_MAT`, `MUL_MAT_ID` and `FLASH_ATTN_EXT` at the §7.2 shapes and
decode widths, on the same host, before adoption is frozen.
- **Build.** From the reference checkout
  `/opt/hf2q-worktrees/adr063-peer-reference-e86f5fd1`, into a *separate*
  build directory. A test-only case-list patch is kept with its diff. The
  serving build and its receipt stay untouched.
- **KV setting.** The flash-attention cells use the peer's F16 KV, its
  serving setting; the representation difference is recorded.
- **Purpose.** M0-P gives the peer's per-kernel cost at Gemma shapes. It is
  the evidence for or against a later Class B MMA spike (§13 decision 2).

### 7.4 Workload distributions (V2; diagnostic; one supervised run, untimed)

`HF2Q_GEMMA_ROUTING_OVERLAP=1` (hf2q only; declared in the Q5 case, §4.8)
works as follows:
- **Ids.** After each layer's routing it blit-copies `moe_expert_ids` into a
  per-step `[30, N, 8]` u32 buffer (3.8 KiB per step). The buffer is read
  after the step's existing final wait, so no synchronization is added.
- **Attention.** Per step it records each slot's KV length per layer type
  and whether the step was mixed-bucket.
- **Output.** At worker exit it prints `[ROUTING_OVERLAP]` histograms: U per
  (layer, width), KV length, and the mixed-bucket share.

One four-agent run with Q5 `CalibrationPass` produces:
- the E1 replay patterns;
- the A KV-length cells and the A2 trigger;
- the inputs for recomputing §5.

**V2 must finish before E1, A1 or A2 can be adopted.**

### 7.5 M0 adoption thresholds (frozen here)

| Lever | Adopt if (precompiled, median) | Never |
|---|---|---|
| D1 body | At m = 4, over the §2.5 shape mix weighted by per-step count: ≤ 0.80 × stock `mN` (≥ 1.25×) | Any cell at m ∈ 1..8 more than 2% slower than its stock route |
| D1 lm head | ≤ 0.80 × stock at m = 4 | More than 2% slower at any m |
| D2 | Q5_0 and Q8_0 each, at m = 4: ≤ 0.67 × stock (≥ 1.5×) | More than 2% slower at any m |
| E1 | gate_up + down at the V2 empirical pattern: ≤ 0.85 × stock | More than 2% slower at U = 32 |
| A1 | Weighted over the V2 KV-length histogram at n_q = 4: ≤ 0.85 × stock | More than 2% slower in any cell |
| A2 | V2 mixed-bucket share ≥ 10% of layer-steps, and ≤ 0.90 × stock on mixed-bucket steps | More than 2% slower on uniform steps |
| F1, F2 | ≥ 3% better than the adopted non-fused route | Any T-N6 mismatch |

The best geometry per shape is frozen into the static table, so run-time data
or timing never selects geometry (§3.2 item 2). A lever below its threshold
is dropped and recorded, not integrated.

## 8. Acceptance gates

### 8.1 Exactness (all must pass; any byte fails)

- S0, then T-N1 to T-N5, T-N7 and T-N8, plus T-N6 if a fused epilogue is
  adopted. All run on both metallib paths, with the toolchain recorded
  (§3.3).
- A1 to A4 on the release-candidate build of the integration branch.
- `ExactTranscripts` (`q1_decision_policy.py transcripts`) over every M2 run,
  candidate against kill switch: 0 same-request/different-output groups.

### 8.2 Speed

| Gate | Measurement | Threshold |
|---|---|---|
| M0 | §7.5 | Per lever |
| M1 in situ | `slot_aware_n4_batched_body_throughput_probe` (`engine.rs:39291`) with `HF2Q_GPU_BUSY=1`, `HF2Q_BENCH_N=4` and `=1`, release build, the real artifact. At least 512 steps per round, 3 rounds per arm alternating `0`/`1`, with the warm-up round discarded. The metric is GPU busy per step, not wall tok/s (ADR-040 §0.21c: wall varies ±14% on this harness). One extra round per arm runs with `HF2Q_FUSE_LMHEAD=0`, so the lm head is its own command buffer, and reports **body and lm-head GPU per step separately**. That fusion does not change bits | N = 4: candidate ≤ 0.90 × kill per-step GPU. N = 1: ≤ 1.01 × kill. Body/head split recorded; §5 is recomputed from it before M2 |
| M2 four-agent A/B | Q5 supervisor, same binary, `GEMMA_DECODE_KERNELS ∈ {0, 1}`, ABBA blocks with a thermal gap, at least 4 runs per arm (extended to 8 if the CI half-width exceeds 2.5 points; no other extension) | Median makespan ratio `1`/`0` ≤ 0.96, with the block-paired 95% CI upper < 1.00. Decode stage reported per turn |
| M2 diagnostics (recorded, not gated) | `gemma_stage_profile.py` per-stage medians; one catsplit run per arm | Decode-category shares before and after |

### 8.3 Fairness and memory

- M2, candidate against kill switch: worst-slot TTFT p95 and SSE-gap max
  ratios ≤ 1.025.
- Every run passes Q5 `CalibrationPass`. There are no new persistent buffers:
  the leader scan needs none, the optional map is ≤ 1 KiB per layer, and A1
  uses threadgroup memory only.
- A fresh Q5 on the release candidate is still required, because the source
  changes.

### 8.4 Quality and the Q4 verdict (combined candidate; owned by the Q4 closing campaign)

This story adds no Class B route, so its own quality obligation is
`ExactTranscripts` (§8.1). The Gemma verdict is decided once per release
candidate on the **combined** candidate: this story's routes, #668 and the
companion Gemma stories. It runs through the `q3_campaign.py` closing
campaign (`adr063-gemma-tiled-attn-design.md` §9.3), which requires:
- `family_verdict == Win` against the peer's MT configuration (CI upper
  ≤ 0.95);
- a paired quality `Pass` against v0.1.28 and against the peer (24 instances);
- worst-slot TTFT p95 and SSE-gap max against v0.1.28 ≤ 1.025;
- a fresh Q5.

## 9. GPU validation plan (root-run; GPU lane and HOST-LOCK; one process at a time)

Every run records:
- source commit, binary SHA-256 and artifact digest `bf65b7e1…`;
- mlx-native version, metallib SHA-256, macOS build and Metal compiler
  version;
- case file and its SHA-256.

Evidence goes under `~/.local/state/hf2q/v0130/q4-gemma-decode/`.

| Step | Where | Model? | Command (shape) | Output |
|---|---|---|---|---|
| S0 | mlx-native branch | no | `MLX_PRECOMPILED_METALLIB=1 cargo test --release --test gemma_decode_w4_exact q4k_rt2 -- --test-threads=1`, then `=0`; one q_proj m = 4 bench cell | D1 feasibility; halt or redesign on failure |
| V0 | mlx-native branch | no | the full `gemma_decode_w4_exact` suite on both paths | T-N1 to T-N8 receipts |
| V2 | hf2q diagnostic commit (lands first) | yes | `HF2Q_Q5_GEMMA_ROUTING_OVERLAP=1 python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json --negative-control-receipt "$NC" --output …/v2-overlap-<ts>` | U, KV-length and mixed-bucket histograms |
| V1 | mlx-native branch | no | `cargo bench --bench bench_gemma_decode_w4` (quiet host; E1 and A cells use V2's distributions) | M0 cells and roofline; geometry table frozen; adoption per §7.5 |
| V1-P | peer reference build directory | no | `test-backend-ops perf -o MUL_MAT`, `-o MUL_MAT_ID`, `-o FLASH_ATTN_EXT` with the §7.2 case list | Peer per-kernel times (reference) |
| V3 | after the native release and exact pin | yes | A1, A3 and A2 as `HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF=$GEMMA HF2Q_GEMMA_DECODE_KERNELS=1 cargo test --release --bin hf2q <name> -- --exact --ignored --test-threads=1`; A4 under both values, then `q1_decision_policy.py exact` | Class A receipts |
| V4 | same binary | yes | M1 probe: `HF2Q_GPU_BUSY=1 HF2Q_BATCHED_BENCH=1 HF2Q_BENCH_N={4,1}`, alternating `HF2Q_GEMMA_DECODE_KERNELS`, plus the `HF2Q_FUSE_LMHEAD=0` split round | M1 receipt; body/lm-head split; recomputed §5 |
| V5 | same binary | yes | `for ARM in 0 1 1 0 …; do HF2Q_Q5_GEMMA_DECODE_KERNELS=$ARM python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json --negative-control-receipt "$NC" --output …/q5-ab-$ARM-<ts>; done`, then `q1_decision_policy.py transcripts` over all runs and `gemma_stage_profile.py` | M2, `ExactTranscripts`, fairness |
| V6 | combined release candidate | yes | `q3_campaign.py --family gemma4 …` (§8.4) | Q4 verdict (root, once per candidate) |

Order:
1. S0, then V0.
2. V2 can run as soon as the hf2q diagnostic lands, in parallel with V0.
3. V1 runs after both S0 and V2, then V1-P, then adoption is frozen.
4. Native merge, release, crates.io check and exact pin.
5. V3, V4, V5, promotion, then V6.

Halt at the first failure and preserve its receipts; never retry until a
pass.

## 10. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The lowest adoptable case fails M2 (§5.1); kernel wins do not reach makespan (K1: exact Q8_0 multi-row +1.2%, n.s.; Qwen projected 2.2×, got 4.9%) | Story gate missed | Stated up front. S0 checks D1 feasibility first. M1 gates before M2. If M2 misses, the operator re-budgets (§13) |
| The D1 gain is capped by per-column arithmetic (§2.4) | D1 below 1.25× | M0 reports element-columns per second; staging and NSG are searched; if D1 is dropped, recorded honestly |
| Catsplit-derived rates mislead (serialized buffers, 4.2× run length); the lm-head cost is unknown (f 0.78–0.90) | Wrong priorities or estimates | M0 uses cold weights and a roofline; M1 measures in situ with the body/head split; §5 is recomputed before M2 |
| Compiler contraction differences break bits, or a toolchain update breaks parity later | Route blocked, or silently non-exact after an OS update | Same flags as the reference (§3.2 item 6); operand-form rule; parity at every shape on both paths; reruns on every pin bump and OS update (§3.3); A1 at model level |
| Register pressure or occupancy at (R0, R1) = (4, 4), or threadgroup memory for A1 at G = 8 | Slower at small N, or A1 G limited | Per-shape geometry search (the table may pick R0 = 1 or G = 1, which is stock); no-regression rule |
| Real-workload expert overlap below the proxy, or duplicates already cache-served | E1 saves little | V2 first; E1 threshold on the empirical pattern; no-regression at U = 32 |
| Leader-scan imbalance | E1 tail latency | Skewed-pattern cells; map variant |
| A process-policy env flag changes the reference silently | Non-exact against stock | §4.8 refusal; T-N7 |
| Interaction with speculation and #668 | Savings not additive | The combined candidate is the only Q4 total; E1 and attention widths cover verify rows |
| Other families pick up the new routes | Unreviewed numerics elsewhere | Per-lever fields default off, not env-resolved; T-N7 |
| The peer reference build drifts from the measured peer | Misleading per-kernel comparison | Separate build directory at `e86f5fd1`; reference only |

## 11. Release and promotion

1. **mlx-native.**
   - A contract issue carries the §3 text.
   - The PR carries the S0 and T-N1 to T-N8 receipts on both paths, the V1
     receipts and the frozen geometry table.
   - After independent review: merge, publish the next 0.15.x, and verify it
     on crates.io.
   - hf2q pins it exactly (`=0.15.x`) in its own commit, from a clean
     registry build.
2. **hf2q.**
   - The V2 diagnostic commit lands first, with no native dependency.
   - Then the integration commit, with the switch defaulting to `0`, followed
     by V3 to V5.
3. **The promotion commit** happens only when every §8.1–§8.3 gate passes.
   It:
   - flips the default to `1`, keeping `0` as the kill switch;
   - deletes the code of any lever that was not adopted (clean cutover);
   - updates the banner, the case-file notes, `scripts/four_agent/README.md`
     and the ADR-063 Q4-GEMMA-DECODE receipts.
4. **If M2 fails,** the default stays `0`, the receipts are recorded, and §13
   decision 2 applies.

## 12. ADR-063 story section (extraction-ready)

### Q4-GEMMA-DECODE. Make Gemma 4 four-slot decode kernels exact and faster

Outcome: Gemma 4 26B-A4B joint decode at four slots spends less GPU time per step in its dense projections, routed experts and attention. Every output stays bitwise equal to today's decode.

Requirement: mlx-native MUST provide exact row-tiled dense, expert-grouped and GQA-grouped attention decode kernels under a stated contract (Q1 Class A against the stock kernels). The program MUST integrate them into the Gemma batched decode body behind a kill switch, and MUST measure the per-step GPU and makespan saving against that switch.

Acceptance:

- Starting point (Q4-GEMMA #660, binary `7f295b38`, mlx-native 0.15.16):
  - The decode body splits as follows: routed MoE 40% (gate/up 21.8%, down 15.3%), dense projections and FFN 37.5%, attention 17.8%, norms 4.5%.
  - In decode-only windows, hf2q is GPU-busy 2.0–2.5 s per run longer than the peer's MT configuration. Part of that gap is MT's speculation (2.34 tokens per verify step, i.e. fewer target steps), not kernel speed; #660 sized decode kernels at 0.3–0.7 s.
  - At width 4, the Q4_K/Q6_K dense projections move their weights at 155–184 GB/s. Per element-column they already match or exceed the expert kernels (1.0–1.19 T against 0.83 T per second), so the exact lever is capped near 2×.
  - Dense `ffn_down` (Q5_0/Q8_0) streams its weight once per row.
  - Both engines stream each (token, expert) pair separately, although the four rows share about 35% of their expert selections (Q1 witness proxy).
  - Decode attention loads and decodes each KV row once per query head: 2× on sliding layers, 8× on global layers.

- Design approved before implementation: `docs/research/adr063-q4-gemma-decode-design.md`, revision 2: §3 (exactness contract), §4 (S0 spike; levers D1, D2, E1, A1/A2; optional F1/F2; policy and refusal rules), §7 (M0 thresholds), §8 (gates).

- A linked mlx-native issue states the §3 contract before implementation. The kernels are:
  - row-tiled Q4_K/Q6_K `mN`, with the lm head decided independently;
  - Q5_0/Q8_0 `mN`;
  - expert-grouped `mv_id` for Q4_K gate_up and Q5_0/Q8_0 down, admitting up to 16 tokens × top-8;
  - GQA-grouped and mixed-bucket vec-hybrid decode attention.

  New per-lever routing-policy fields default off and are never resolved from the environment. Existing entry points stay bit-identical. Each new kernel is compiled with its reference kernel's flags.

- Exactness, verified in order:
  - A Q4_K R0 = 2 parity spike runs before the microbenchmark.
  - Primitive parity is u32-bitwise against the stock kernels at every Gemma shape, synthetic row tails, m = 1..8, all expert patterns (all-distinct, all-shared, empirical, invalid id), attention KV lengths across every bucket edge, and both metallib paths, with the toolchain recorded.
  - M0 uses cold-weight rotation and a host roofline. It reports weight GB/s and element-columns per second, uses the real-workload routing and KV-length distributions, and includes a peer per-kernel reference. Each lever is adopted only at its frozen threshold.

- hf2q switch and Class A gates:
  - `HF2Q_GEMMA_DECODE_KERNELS` (`0` is the kill switch) captures the environment routing policy at load. It refuses unless the reference flags hold.
  - A stock-equivalence test compares logits, tokens, KV, and expert ids and weights bitwise against `0`.
  - The slot-aware n1/n4/n8 and eviction parity tests pass.
  - The two arms are `ExactEqual` on the Q1 T2 cohort trace.

- Speed:
  - In situ, decode-step GPU at width 4 is ≤ 0.90 × the kill switch, with no width-1 regression and the lm head reported separately.
  - Four-agent Q5 A/B, same binary (ABBA, at least 4 runs per arm): median makespan ratio ≤ 0.96, block-paired 95% CI upper below 1.00.
  - `ExactTranscripts` holds across the A/B.
  - Worst-slot TTFT p95 and SSE-gap max ratios are ≤ 1.025.
  - The 0.96 closes this story's share of the Gemma Win only if the other Gemma stories reach their high estimates; otherwise about 0.93 is needed. Any shortfall is re-budgeted across the Gemma stories (operator preference, 2026-11-04).

- The default flips only after every gate passes. The Gemma verdict is then decided on the combined candidate in the Q4 closing campaign: Win against the peer's MT configuration, paired quality Pass against v0.1.28 and against the peer, stock fairness 1.025, and a fresh Q5.

Repository ownership: mlx-native owns the kernels, dispatchers, policy fields, contract, parity tests, microbenchmarks and release. hf2q owns route selection, the switch and refusal rules, the overlap diagnostic, integration, the Class A gates and the campaigns, consuming an exact published mlx-native pin.

Validation: Independent review of the contract, of the spike, parity and microbenchmark receipts (including the peer reference), of the Class A receipts, and of the A/B from raw receipts.

Predecessors: Q4-GEMMA (#660, profiling result), Q5 (#371). It is independent of Q4-GEMMA-KERNEL and Q4-GEMMA-ATTN. Its grouped expert and attention kernels serve the companion Gemma speculation story's verify rows.

Priority: High. Core scope (second-tier family).

Delivery boundary: A published mlx-native release pinned exactly by hf2q, plus the integration merged for v0.1.30 (or the first published hf2q release containing the accepted Gemma 4 configuration).

## 13. Decisions (recorded 2026-11-04, root on review)

1. **Class A-only scope: approved.** The peer's MMA, `mul_mv_ext` and K-split
   routes are excluded, and the S0 spike (D1 Q4_K R0 = 2 parity, before M0)
   is added.
2. **If M2 misses 0.96: prefer (a).** The measured saving is accepted and the
   Win is re-budgeted across the other Gemma stories.
   - (b), a Class B few-row MMA spike under its own design, is considered
     only if M0-P shows a per-kernel peer advantage large enough to pay for a
     Q1 envelope plus a paired quality campaign.
   - (c), a re-scope of the Gemma Win, stays the last resort.
3. **M0-P peer reference build: approved.** It runs before adoption is
   frozen.
4. **Decode attention: added to this story (root decision)** as lever A. It
   is Class A, bitwise against today's vec-hybrid route, the peer's decode
   attention is included in M0-P, and A1/A2 have adopt/drop thresholds.
5. **Switch name and promotion rule: approved,** with the §4.8 rules: base
   policy captured at load, refusal unless the reference flags hold,
   per-lever fields with `serde(default)`, default `0` until §8.1–§8.3 pass,
   then `1` with `0` kept as the kill switch.

## 14. Review record (ReviewGemmaDecode, ACCEPT-WITH-CONDITIONS)

| Finding | Resolution |
|---|---|
| 1. The low savings case fails the design's own gates; E1's 30% share had no derivation | §5.1 now has below-threshold, at-threshold and high columns, and states that the lowest adoptable case (≈ 0.42–0.48 s, ratio ≈ 0.97) fails M2. E1 is bracketed at 0 and at 1 − U/32, and the 30% share is dropped |
| 2. The diagnosis compared unlike units | §2.2 adds element-columns per second; §2.4 restates the diagnosis as per-column work; the D1 high case is capped at 2×; M0 reports both metrics |
| 3. The fast-math claim was false | §3.2 item 6 now requires the same compile flags as the reference (Metal defaults; fast math is disabled only for `deepseek_tail_rope*`). §3.3 and §10 add the compiler-dependence risk |
| 4. The policy could not express what M0 adopts, and had no base | §4.7: one `bool` per lever with `serde(default)`, `Copy`-safe, geometry in a static table, `combine_routing_policies` updated. §4.8: base captured at load from `ggml_routing_policy_from_environment()`, refusal on reference-flag mismatch, separate body, lm-head and expert policies |
| 5. V2 failed as written | §4.8 declares `GEMMA_ROUTING_OVERLAP` and `GEMMA_DECODE_KERNELS` in `gemma4.json`, with launcher forwarding and tests. §9 orders V2 before V1's adoption |
| 6. The 0.9 factor and the lm-head lever are coupled | §2.1 and §5.1 use f = 0.78–0.90 with the lm head at 1.3–3.1 ms/step; M0 and M1 (§8.2) measure the lm head per step, and §5 is recomputed before M2 |
| 7. Text corrections | Command-buffer count (≈ 225,000–258,000 per run); stock `mN` tiling (5 → [5], 6 → [3, 3], 7 → [4, 3], 8 → [4, 4]); synthetic row tails; `yl_c` exists only in Q6_K; Q8_0/Q5_0 `mv_id` line ranges; BB:2151-2164 cited for the neutral dispatch reduction; record commits `87d5df5c`/`ddf781b6`; top-16 share 72% (0.49–0.93) for rows 128–143; rows 29–127 mean U 10.5 (p10/p50/p90 8/9/16); 13.1 (not 12.9) ms/step of lever classes in revision 1 |
| 8. ADR text | §12 states the MT speculation share and the budget condition (0.96 vs ≈ 0.93) |

## Source identities

Line numbers help reading only.

| Source | SHA-256 |
|---|---|
| `M:src/shaders/quantized_matmul_ggml.metal` | `5aae347d4e3bba7f25ae91a10900edbbc430052862bf8ebf63c367839a01acaa` |
| `M:src/shaders/quantized_matmul_id_ggml.metal` | `bb4094c4d4e91428fc962c26dcdefdbba9dfea31247079651b1b0b6ec409d46a` |
| `M:src/ggml_capability.rs` | `1d79fdee9b0731d588aa5c8d814b19d7035438dde43198066f86ab8959070947` |
| `M:src/ops/quantized_matmul_ggml.rs` | `b51ce1b48c9a3b4f664bee0a2d0532a408212350048278d8d306ab9beb12efce` |
| `M:src/ops/quantized_matmul_id_ggml.rs` | `9d92db4880e85b2dcd59551a309445a6e00045e86174a054740f2101a605dd12` |
| hf2q `ff4af4bd` `src/inference/models/gemma4/batched_body.rs` | `f7e8af55ebb3cfb747a3592a25016d7608d35f5c518f09747bfa44184e4655da` |
| hf2q `ff4af4bd` `src/inference/expert_dispatch.rs` | `cd54a84485ec4ccc38d070b6f598d484187b9e739cf8d458f7792f2a9040e227` |
| `P:ggml/src/ggml-metal/ggml-metal-common.cpp` | `ade4f77559e687901f84c17cf4411a35fb1bc0500818dd27cab74905fecf5978` |
| `P:ggml/src/ggml-metal/ggml-metal-ops.cpp` | `c996f88af4c4e5f7dc354271c387a62161febec37457194b8ff25500ec945f22` |
| `P:ggml/src/ggml-metal/kernels/mul_mv_mma.metal` | `115e843e70a6b5f8b6d4dc691a14706fadf6e9929b7f661989b5d7ae3d7716f9` |
| `P:ggml/src/ggml-metal/kernels/mul_mv.metal` | `3b6250aa1ec74091bbadedbae35355c2d6ea5d566c2033f91d9aecd896c69059` |
| `P:src/models/gemma4.cpp` | `247d458f93ddace3091141d7d7e4da0c0b589bc843e5737d3b39f804eb889339` |
