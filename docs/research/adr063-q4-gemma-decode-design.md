# ADR-063 Q4-GEMMA-DECODE: exact width-4 decode kernels for Gemma 4 (dense and routed MoE)

Status: design for approval (2026-11-04). No code, build or hardware run backs
this document. Every number below is quoted from a retained receipt (with its
path), computed from shapes and block sizes, or marked `[INFERENCE]`.

Story: **Q4-GEMMA-DECODE** (new, v0.1.30; operator decision 2026-11-04 to add
four Gemma stories under ADR-063 Q4). mlx-native provides the kernels; hf2q
integrates them into the Gemma batched decode body. §12 is the ADR-063 story
text, ready to extract.

Sources:
- The Q4-GEMMA profile (#660, PR #687): `docs/research/adr063-q4-gemma-profile-result.md`
  on branch `feat/q4-gemma-profile` (`7f295b38`), with its evidence under `$G`.
- ADR-063 sections Q4, Q4-GEMMA, Q4-GEMMA-ATTN; the Q1 numeric policy
  `hf2q.q1.numeric-policy.v1` (`docs/research/adr063-q1-numeric-policy.md`).
- ADR-040 §0.21b/§0.21c (the `mul_mv_ext` lever and the bit-exact Q6_K `mN`
  kernel) and the K1 record `docs/research/adr063-k1-q8-multirow-disposition.md`.
- hf2q `main` `ff4af4bd`; mlx-native `=0.15.16` (crates.io source, the current
  pin; `/opt/mlx-native` main is 0.15.17); the peer source at `e86f5fd1`, the
  build the profile measured (`b128-e86f5fd`).

Abbreviations: `$G` = `/Users/robert/.local/state/hf2q/v0130/q4-gemma`;
`$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`;
`BB` = `src/inference/models/gemma4/batched_body.rs`; `M:` = the mlx-native
0.15.16 crate source; `P:` = the peer source at `e86f5fd1`.

## 1. Decision summary

1. **Every change is Q1 Class A against stock.** Each new kernel writes, for
   every output element, the same bits as the stock 0.15.16 kernel that
   today's decode uses for that tensor (§3). The Gemma batched body stays
   bitwise equal to serial width-1 decode (ADR-040 S2/S3), and the candidate
   stays bitwise equal to the kill-switch arm. Task quality is inherited
   through `ExactTranscripts` (Q1 §3.6). No decision-envelope campaign and no
   near-tie risk are added to the Gemma Win. The peer's few-row MMA kernels
   change the reduction order, so they are not adopted (§4.5).
2. **D1, row-tiled K-quant `mN` (Q4_K, Q6_K).** Each SIMD group computes R0
   weight rows (2 or 4) for all R1 ≤ 4 columns. Each activation load then
   serves R0 rows instead of one. This targets q/k/v/o, dense gate/up and the
   Q6_K lm head. Today these run at 155–184 GB/s of weight bytes at width 4,
   against 469–522 GB/s for the expert products in the same run (§2.2).
3. **D2, Q5_0/Q8_0 `mN` (dense `ffn_down`).** It replaces the plain mat-vec,
   which streams the weight once per row (`grid.y = 4`). Clones of the stock
   per-row kernels read the weight once for all columns.
4. **E1, expert-grouped `mv_id` (gate_up Q4_K; down Q5_0/Q8_0).** The first
   (token, expert) pair of each distinct expert computes every pair that
   chose that expert, so each expert's weights are read once per step. The
   four rows share about 35% of their expert selections (§2.4). E1 combines
   with D1's R0 for Q4_K, and it admits up to 16 tokens × top-8 for the
   companion speculation story's verify rows.
5. **Fusion.** Two epilogues are optional and decided by M0: dense gate+up as
   one dual-weight launch, and the GELU·mul epilogues. Router `mN`, batched
   routing and other dispatch-count reductions are not proposed. ADR-040
   measured them neutral, and hf2q's dispatch count already matches the
   peer's (§2.3).
6. **Gates.** Primitive bitwise parity; then M0 microbenchmarks with an
   adopt/drop threshold per lever; then the in-situ GPU-per-step probe M1;
   then the four-agent A/B M2 against the kill switch. M2 requires a median
   makespan ratio ≤ 0.96 with CI upper < 1.00, plus `ExactTranscripts` and
   fairness ≤ 1.025. The 0.96 is the decode-kernel share of the Gemma Win
   budget (§5.2).
7. **Switch.** `HF2Q_GEMMA_DECODE_KERNELS`: `0` is the kill switch (stock
   routes) and `1` selects the adopted exact routes. The default flips to `1`
   only after every gate passes. New mlx-native policy fields default off, so
   no other family changes.

## 2. Evidence

### 2.1 What the profile measured

From #660 (`adr063-q4-gemma-profile-result.md` §2–§4):
- **Decode stage.** hf2q 9.203 s against the peer MT's 7.958 s per agent
  (Δ +1.170 s, CI +0.961 to +2.012). Makespan is 14.867 against 12.477 s; a
  Win needs ≤ 11.85 s.
- **GPU, L2 trace, 3 blocks.** Decode-only GPU busy is 7.67–8.24 s for hf2q
  and 5.40–5.71 s for the peer, so +2.0–2.5 s of decode GPU per run. hf2q's
  in-flight GPU idle is about 0.5 s *lower* than the peer's, so the gap is
  GPU work.
- **Decode split, L3 `HF2Q_DECODE_CATSPLIT`.** MoE 40% (gate/up 21.8%, down
  15.3%, other 3.1%), dense 37.5%, attention 17.8%, norms and embedding 4.5%.
  The lm head and sampling are excluded.

The worker-exit 12-bucket dump of the three catsplit runs
(`$G/profile/l3/0{21,23,25}-b0*-hc-*/server.stderr`, `[CATSPLIT]` lines) gives
the per-step view used below:
- **Steps per run.** `dense_q_proj` has 13,800 / 15,300 / 13,350 dispatches,
  one per layer-step, so 460 / 510 / 445 decode steps.
- **Mean width 3.64 rows.** `dense_ffn` dispatches per layer-step are
  `3 + W`: gate, up and down are one dispatch each, and the router is one per
  row (BB:1897). That gives W = 3.617 / 3.576 / 3.724. Most steps are 4
  wide.
- **Fixed command-buffer cost ≤ 6 µs.** Catsplit commits one command buffer
  per category boundary. Tiny categories bound the fixed cost of each: for
  example `attn_pre` is 89.8 ms over 13,800 buffers, 6.5 µs per buffer for
  three small dispatches. The "net" column subtracts 6 µs per buffer. It is
  an upper bound on the correction, so the true per-class time lies between
  the gross and net values.
- **Distortion.** Catsplit serializes about 14,000 command buffers per run
  and makes the run 4.2× longer (#660 §4). Its shares are a relative ranking.
  The per-class rates below use the same run for every class, so the
  comparisons between classes hold. [INFERENCE] The net body of 16.9 ms per
  step is within about 10% of the unprofiled decode GPU per step. That
  estimate takes the decode-only GPU busy of 7.67–8.24 s over about 470
  steps, less the lm head.

### 2.2 Bytes and achieved rates per class (mean over the 3 runs)

Weight bytes per step are computed from shapes and block sizes (§2.5) and
summed over the 30 layers at W = 3.64. Logical bytes count every weight read a
kernel issues, including repeats.

| Class (stock kernel at width 4) | Gross ms/step | Net ms/step | Logical weight MB/step | Weight GB/s (net) |
|---|---:|---:|---:|---:|
| q_proj (Q4_K `mN`) | 1.535 | 1.355 | 227.1 | 168 |
| k_proj (Q4_K `mN`) | 0.757 | 0.577 | 89.2 | 155 |
| v_proj (13 Q6_K + 12 Q4_K `mN`) | 0.717 | 0.567 | 100.4 | 177 |
| o_proj (Q4_K `mN`) | 1.412 | 1.232 | 227.1 | 184 |
| dense_ffn: gate+up (Q4_K `mN`), router (F32, 4 × m=1), down (Q5_0/Q8_0 `Mv`, grid.y = W) | 3.044 | 2.684 | 200.7 + 157.4 + 560.0 | — (split unmeasured) |
| expert gate_up (Q4_K `mv_id`, one pass per pair) | 4.334 | 4.154 | 1,947.9 | 469 |
| expert down (Q5_0/Q8_0 `mv_id`) | 3.043 | 2.863 | 1,493.4 | 522 |
| attention (KV encode + flash), out of scope | 3.337 | 3.157 | — | — |
| lm head (Q6_K `mN`), outside catsplit | not measured | — | 605.6 per step | — |

The finding: **at width 4 the dense K-quant projections move their weight
about 2.7× slower per byte than the expert kernels in the same run.** Both
families use the same Q4_K lane partition, the same 64-thread threadgroups and
the same short K (2816 = 11 super-blocks).

### 2.3 Where hf2q and the peer differ at Gemma shapes

Both engines run the same GGUF (`bf65b7e1…`). Per layer at four rows:

| Op | hf2q (default route) | The peer at `ne11 = 4` | Difference |
|---|---|---|---|
| q/k/v/o, dense gate/up, lm head (Q4_K, Q6_K) | `kernel_mul_mv_{q4_K,q6_K}_f32_mN_r1_4`: one weight pass, but each column's activations are re-loaded per weight row (Q4_K: 1 row per SIMD group; Q6_K: 2) (`M:src/shaders/quantized_matmul_ggml.metal:1573-1682`, `1159-1287`; route `M:src/ggml_capability.rs:633-650`) | Few-row MMA `kernel_mul_mv_mma_*`. 8×8 simdgroup tiles, 8–32 SIMD groups splitting K; each weight is dequantized once per lane for all columns (`P:ggml/src/ggml-metal/ggml-metal-common.cpp:143-154`, `rows_min` Q4_K 3, Q6_K 2 at `:121-141`; tiling `P:…/ggml-metal-device.cpp:849-907`; kernel `P:…/kernels/mul_mv_mma.metal:504-606`) | Activation traffic per weight byte. Not exact: the K split reorders each row's sum |
| dense `ffn_down` (Q5_0 ×16, Q8_0 ×14; K = 2112) | `Mv` with `grid.y = 4` (`kernel_mul_mv_q5_0_f32`, `kernel_mul_mv_q8_0_f32_nr2`): the weight is streamed once per row (`M:src/ops/quantized_matmul_ggml.rs:1680-1767`) | MMA (2112 % 64 = 0 for Q5_0; Q8_0 uses the 32-wide block kind) | Weight read 4× vs 1× |
| router (F32 2816→128) | 4 × `dense_matvec_f32` m=1 (BB:1896-1911, 466-487) | `mul_mv_ext`, r1ptg = 4, 1 dispatch (`P:…/ggml-metal-ops.cpp:2742-2848`) | 1.44 MB re-read 4×; ADR-040 M4 measured amortizing F32 neutral (BB:464-480) |
| routing (softmax, top-8, scale) | per slot, 4 × `fused_moe_routing_f32` (BB:1952-1982) | fused `TOPK_MOE`, 1 dispatch (`P:…/ggml-metal-fusion.cpp:357-502`) | dispatch count only |
| expert gate_up (Q4_K, 2816→1408 × 128) | `kernel_mul_mv_id_q4_K_f32`: 1 row per SIMD group, 2 per threadgroup, grid (704, 4·8) (`M:src/shaders/quantized_matmul_id_ggml.metal:1582-1680`; BB:2062) | `mul_mv_id`: N_R0 = 2, N_SG = 2, 4 rows per threadgroup, grid z = 32 (`P:…/ggml-metal-ops.cpp:3080-3138`; `P:…/ggml-metal-impl.h:57-58`) | Peer shares activation loads over 2 rows. **Neither groups tokens that share an expert** |
| expert down (Q5_0/Q8_0, 704→2816) | `mv_id` Q5_0 / Q8_0, 4 rows per SIMD group, 2 SIMD groups (`…id_ggml.metal:561-623`, `1027-1078`); Q8_0 `nr2` is off (`M:src/ggml_capability.rs:180`) | `mul_mv_id`; Q8_0 N_R0 = 2 × N_SG = 4 (K split) | Q8_0 K split is not exact, so it is not adopted; no grouping in either engine |
| `mm_id` (expert-grouped tiles) | only above 32 tokens; Gemma decode pins `mv_id` (BB:2053-2061) | only at `ne21 ≥ 32` (`P:…/ggml-metal-common.cpp:44-50`) | none at decode width |
| expert GELU·mul, weighted top-8 sum | `moe_swiglu_batch` (GELU-tanh), `moe_weighted_sum_seq`: one dispatch each | `geglu` (no GLU fusion), `MOE_REDUCE` fusion | equivalent |
| norms, residuals | fused norm+add, fused post-FF norm2 + end-of-layer (BB:1801, 2236) | `NORM_MUL(_ADD)` fusions | equivalent |
| dispatches and command buffers per step | about 1,050 dispatches; concurrent encoder with conflict barriers; 4 async command buffers, one wait (BB:2505-2590) | about 36 per layer [INFERENCE] ≈ 1,080; concurrent encoder; 2 command buffers (`P:…/ggml-metal-ops.cpp:159-246`, `P:…/ggml-metal-context.m:486-512`) | none material |

Decode attention (17.8%) uses different kernels in the two engines. It is not
in this story's scope (§13, decision 4).

### 2.4 Diagnosis

**Dense K-quant `mN` is limited by activation re-loads, not weight bandwidth.**
The stock Q4_K row kernel (`…ggml.metal:1452-1560`) is laid out as follows:
- Each lane loads 32 activation floats (128 B) per super-block it visits,
  against 144 B of weight for the whole super-block, which 8 lanes share.
- Per weight row, one SIMD group therefore loads 11 KiB of activations for
  1.58 KiB of weight, a ratio of about 7:1.
- `mN` keeps that per-row structure and loops over R1 = 4 columns, so the
  ratio becomes 28:1.

The `mv_id` kernels run one column per pair at about 7:1 and reach 469 GB/s.
`mN` at 28:1 reaches 155–184 GB/s. An exact kernel cannot change the lane
partition, because that partition fixes each row's summation order. It can,
however, let one activation load serve R0 rows: R0 = 4 restores the 7:1 ratio
at four columns. ADR-040 §0.21c recorded the same hypothesis for small-N
projections: "occupancy-aware rework (cache only the per-block weight dequant,
stream Y per column; re-prove bit-identity) is the obvious next lever"
(`docs/adr/ADR-040-continuous-batching-reopen.md:790`).

**Dense `ffn_down` streams 4× its weight.** This is 560 MB per step logical
against 154 MB unique. [INFERENCE] Part of the re-read may come from the
system cache, since one layer's `ffn_down` is only 4.1–6.3 MB; M0 measures
how much.

**The four rows share expert weights that both engines read separately.** The
Q1 routing witness on the accepted greedy144 dataset
(`$E/q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl`; 4
lanes × 144 positions × 30 layers, top-8 of 128) gives the number of distinct
experts U among the 4 lanes' 32 selections at one (layer, position):

| Rows | Lanes' selections identical to lane 0 | Mean U (of 32) | p10 / p50 / p90 | Duplicate fraction 1 − U/32 |
|---|---|---:|---|---:|
| 128–143 (greedy continuation, decode-like) | 0 / 1,440 | 20.9 | 16 / 21 / 26 | 0.35 |
| control: each lane at an independent random position | — | 18.7 | — | 0.41 |
| 29–127 | 1,622 / 8,910 | 11.9 | 8 / 10 / 20 | 0.63 (lanes partly identical) |

Expert use is skewed: the 16 most-used of 128 experts take 84.5% of all
selections (layer range 0.70–0.99). So even unrelated tokens overlap heavily.
[INFERENCE] These are synthetic lanes, not the four coding agents. M0-R (§7.4)
measures U on the real workload before any MoE saving is claimed.

**Dispatch count is not a cause.** hf2q issues about 1,050 dispatches per step
and the peer about 1,080 [INFERENCE]. Host in-flight idle is lower on hf2q.
ADR-040 measured the batched weighted sum "byte-identical but
throughput-NEUTRAL" (BB:1946-1949), and F16/F32 weight amortization neutral
(BB:477-480).

### 2.5 Shapes and types (q4_k_m artifact `bf65b7e1…`)

30 layers: 25 sliding (8 KV heads, head dim 256) and 5 global (2 KV heads,
head dim 512, V = K). Hidden 2816, 16 query heads, dense FFN 2112, 128 experts
top-8, expert FFN 704, vocab 262,144 (tied). Inventory: Q4_K 192, Q5_0 32,
Q8_0 28, Q6_K 14, F32 392 (`docs/research/adr063-f1-parity-validation-2026-10-23.md:46`).
The per-role mapping below is derived from `src/quantize/ggml_quants/standard_policy.rs`
(`:125-170`, `:330`, `:572`, `:653`); it matches the counts exactly:

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

## 3. Exactness contract (normative; the text of the linked mlx-native issue)

### 3.1 Reference kernels

For each tensor class, the reference is the stock 0.15.16 kernel that the
m = 1 decode uses today under the default routing policy (`M:src/ggml_capability.rs:168-184`):

| Class | Reference kernel (m = 1) | New entry point |
|---|---|---|
| Q4_K dense | `kernel_mul_mv_q4_K_f32` | D1 `kernel_mul_mv_q4_K_f32_rt{R0}_r1_{R1}` |
| Q6_K dense (incl. lm head) | `kernel_mul_mv_q6_K_f32_nr2` (the stock `mN` already clones it, ADR-040 §0.21c) | D1 `kernel_mul_mv_q6_K_f32_rt{R0}_r1_{R1}` |
| Q5_0 dense | `kernel_mul_mv_q5_0_f32` | D2 `kernel_mul_mv_q5_0_f32_mN_r1_{R1}` |
| Q8_0 dense | `kernel_mul_mv_q8_0_f32_nr2` (`dense_q8_0_mv_nr2 = true`) | D2 `kernel_mul_mv_q8_0_f32_nr2_mN_r1_{R1}` |
| Q4_K expert | `kernel_mul_mv_id_q4_K_f32` | E1 `kernel_mul_mv_id_q4_K_f32_grouped_rt{R0}` |
| Q5_0 / Q8_0 expert | `kernel_mul_mv_id_q5_0_f32` / `kernel_mul_mv_id_q8_0_f32` (`expert_q8_0_mv_nr2 = false`) | E1 `kernel_mul_mv_id_{q5_0,q8_0}_f32_grouped` |

Kernel names are proposals; mlx-native may rename them.

### 3.2 Invariants (bitwise, raw u32 of every F32 output)

1. **Per-element identity.** For each output (column c, row r):
   `bits(new[c][r]) == bits(reference(weight, input column c)[r])`, where the
   reference is one m = 1 dispatch on column c alone.
2. **Width invariance.**
   - Dense: holds for m ∈ 1..8. For m > 4, columns are tiled `4 + (m − 4)`,
     as the stock adaptive `mN` tiles them (`M:src/ops/quantized_matmul_ggml.rs:2135-2148`).
   - Grouped: holds for n_tokens ∈ 1..16 at top-8 (gate_up) and
     n_tokens · top_k ≤ 128 at top-1 (down).
   - Any column count, tiling, threadgroup shape or R0 the dispatcher picks
     leaves every output's bits unchanged.
3. **Column independence and permutation.** Column c's output depends only on
   column c's input and the weights. It does not depend on the other columns,
   their order, their count or their values, including NaN and ±∞ in a
   neighbouring column. Permuting input columns permutes output columns.
4. **Grouping invariance (E1).** The bits written for pair p depend only on
   `(weights[ids[p]], input row of p)`. They do not depend on how many or
   which other pairs chose the same expert, or on the pair order. Cases that
   must hold:
   - all 32 experts distinct;
   - all pairs sharing one expert;
   - every mixed pattern;
   - out-of-range ids. These produce exactly the stock
     `poison_invalid_expert_id` output for that pair (`…id_ggml.metal:586-587`,
     `1047-1048`, `1606-1607`), and an invalid id never joins a group.
5. **Tails and views.**
   - N not a multiple of rows per threadgroup: k_proj global N = 1024,
     N = 2112 and N = 2816 with 4- and 8-row tiles.
   - K tails: Q5_0 K = 2112 is 66 blocks, so lanes do 4 or 5 iterations;
     expert K = 704 is 22 blocks.
   - `slice_view` and `BufferWithOffset` inputs and outputs at non-zero
     offsets.
   - Bytes outside the output views are unchanged.
6. **Both metallib paths.**
   - Precompiled (`MLX_PRECOMPILED_METALLIB=1`, the default) and runtime
     source compile (`=0`).
   - Both compile with fast math disabled (`M:build.rs:191`,
     `M:src/kernel_registry.rs:2002`).
   - On each path, the new kernel equals the reference kernel *on that
     path*. The two paths' outputs are also compared with each other and the
     result recorded.
7. **Operand-form rule (implementation guidance; the tests above decide).**
   - Each per-(row, column) expression is a literal clone of the reference
     kernel's: the same cached `yl`/`yh` operand form, the same `sumy`
     accumulation, the same `acc` lanes, scale decode, `sumf` expression,
     `simd_sum` and threadgroup reduction order.
   - Caching activations in a different register form flipped Apple's
     FMA-contraction decision once and caused a 1-ULP drift (ADR-040 §0.21c,
     Q6_K `nr2` vs plain `mv`).
   - Interleaving R0 rows or R1 columns is allowed. Reassociating any sum is
     not.

### 3.3 Q1 classes

- **Against stock (`HF2Q_GEMMA_DECODE_KERNELS=0`): Class A.** Every captured
  byte is equal: logits, token ids, KV, expert ids and order, cursors (Q1
  §3.1 item 4). One differing byte is `DivergedExactInvariant`, a failure
  that blocks the route. No tolerance applies.
- **Batched vs serial (ADR-040 S2/S3 byte identity): Class A**, unchanged.
- **Quality inheritance.** A Class A route inherits task quality (Q1 §3.6).
  The story's A/B needs `ExactTranscripts`, not a paired quality campaign.
  The Q4 closing gates still run on the combined candidate (§8.4).
- **Not allowed in this story:** the peer's MMA few-row kernels, `mul_mv_ext`
  for K-quants, the Q8_0 `nr2` K split for experts, `mm_id` at decode widths,
  and any kernel whose per-row order differs from §3.1. Each is Class B
  against stock, and Gemma's routing gaps make Class B routes route-sensitive
  (Q1 §2.2: 8th/9th softmax gap median 7.5e-4; `RouteSensitive` on every
  measured Gemma alternate route). §4.5 records them as falsified-by-policy
  alternatives; §13 decision 2 asks whether any may be opened.

## 4. Proposed changes (LLD)

### 4.1 D1: row-tiled K-quant `mN` (Q4_K, Q6_K)

Structure, Q4_K (the Q6_K variant is identical in shape around the `nr2`
per-row body):

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

- **What is shared.** The activation loads and `sumy` for one (block,
  column) serve R0 rows. In the stock kernel each row recomputes them; the
  recomputation is deterministic, so it yields the same value.
- **Register budget** [INFERENCE]. Per lane: `yl/yh` 32 floats, `sumy` 4,
  `sumf` R0·R1 = 16 at (4, 4), and per-row quant bytes and scales of about
  4·R0 registers. That is about 70–90 registers at (R0, R1) = (4, 4). Only
  one column's activations are live at a time, unlike the stock `mN`, which
  keeps `yl_c[R1][16]`. ADR-040 hit a spill cliff at R1 ≥ 6 with the stock
  layout, so R1 stays ≤ 4 per tile.
- **Geometry search space (M0 decides; values do not affect bits).**
  - R0 ∈ {1, 2, 4}, NSG ∈ {2, 4, 8}, R1 ∈ {1, …, 4};
  - optional activation staging of one column at a time in threadgroup
    memory (2816 floats = 11 KiB). Staging moves loads, not arithmetic.
- **Where it applies.**
  - Width ≥ 2: q, k, v, o, dense gate and up, and the lm head (Q6_K
    262,144 rows; `batched_head.rs:244-322`).
  - m = 1: optional. Width-1 steps are a minority (mean width 3.64). M0
    decides, and stock m = 1 stays the reference.
- **F1, dual-weight gate+up (optional).** Gate row r and up row r are
  treated as the R0 rows of one SIMD group, from two weight tensors. This
  saves the activation loads and one dispatch. Adopted only if M0 shows it
  beats two D1 launches.

### 4.2 D2: Q5_0 and Q8_0 `mN` (dense `ffn_down`)

- **Q5_0.** Clone `kernel_mul_mv_q5_0_f32` (`…ggml.metal:407-459`): 4 rows per
  SIMD group, 2 SIMD groups, 8 rows per threadgroup. Add an R1 ≤ 4 column
  loop around the `sumy[2]` / `yl[16]` activation packing and the per-row
  `block_q5_0_dot_y`.
- **Q8_0.** Clone `kernel_mul_mv_q8_0_f32_nr2` (`:837-934`): 2 rows and 4
  K-split SIMD groups per threadgroup, with the reduction through threadgroup
  memory in the stock order.
  - The K split is the reference's own order, so cloning it is exact.
  - Each column gets its own `shmem_rows` slice: 256 B per column, 1 KiB at
    R1 = 4.
  - The two barriers and the final `simd_sum` are unchanged.
- Grid `(⌈N/rows_per_tg⌉, ⌈m/R1⌉, 1)`. The weight is read once per tile.

### 4.3 E1: expert-grouped `mv_id` (leader pairs)

**Leader rule.** At pair index p (flat `token·top_k + slot`), let `e = ids[p]`.
Pair p is a *leader* iff `e` is valid and no `q < p` has `ids[q] == e`.
- One token never selects an expert twice (top-k ids are distinct), so each
  group has at most n_tokens members.
- An invalid id is always its own leader and runs the stock poison path.

**Kernel.**
- The grid is unchanged: `(row tiles, n_tokens·top_k)` threadgroups, as
  stock.
- A non-leader threadgroup reads `ids[0..p]` (at most 512 B, uniform across
  the threadgroup) and returns.
- A leader threadgroup collects its members `{p} ∪ {q > p : ids[q] == e}`.
  It then runs the reference per-pair arithmetic for each member:
  - each member's activation is its own token row (gate_up) or its own pair
    row (down);
  - each member's output row is written exactly where the stock kernel
    writes it;
  - the weight rows are loaded once for all members.
- Members are processed in chunks of ≤ 4 columns (register budget as in §4.1).

**Variants.**
- Q4_K gate_up: R0 ∈ {1, 2, 4} rows per SIMD group (the peer uses 2).
- Q5_0 and Q8_0 down: the stock kernels already hold 4 rows per SIMD group.

**Why no map pass by default.** The leader scan needs no extra dispatch, no
buffers and no barrier. It keeps E1 a drop-in for
`quantized_matmul_id_ggml_mv_with_policy`. M0 also times a compact-grid
variant, with one map dispatch per layer shared by gate_up and down, and
picks the faster. Both are exact.

**Load balance.** A group with c members costs c column passes. With
mean U ≈ 20 and c ≤ 4, the step's critical path is bounded by the most
popular expert. M0's U-pattern cells include a skewed pattern for this
(§7.2).

**Width.** n_tokens ≤ 16 at top-8 for gate_up and ≤ 128 pairs for down. This
covers the companion speculation story's verify rows (4 slots × up to 4 rows),
on request from its design (DesignGemmaSpec, 2026-11-04). Above 32 tokens the
existing `mm_id` route and its threshold are untouched. The prefill MoE
design does not change `mv_id` (DesignGemmaPrefillMoe, 2026-11-04).

**F2, optional GELU epilogue on gate_up.** Gate row r and up row r + 704
become one SIMD group's two rows. The epilogue applies the
`moe_swiglu_batch` expression (GELU-tanh(g)·u,
`M:src/shaders/moe_dispatch.metal:145-163`) and writes the activation buffer
directly. That removes one dispatch, one barrier and a 180 KiB round trip per
layer. It is adopted only if the expression is bitwise equal (T-N6) and M0
shows a gain. The same rule applies to a dense GELU·mul epilogue on F1.

### 4.4 Not proposed (with reasons)

| Candidate | Why not |
|---|---|
| Router F32 `mN` (1 dispatch instead of 4) | ADR-040 M4 measured amortizing F16/F32 weight reads throughput-neutral (BB:464-480) |
| Batched per-token-identical routing | Dispatch count only; ADR-040 found the per-slot MoE loops "never a cost" (BB:1946-1949) |
| RMSNorm + matvec fusion | Neither engine does it; one 11 KiB row per norm |
| Residual add in the O-proj epilogue (the peer's `MUL_MAT_ADD`) | Never fires for Gemma on the peer; hf2q already fuses norm + add |
| Per-step `BatchedDecodeBuffers` allocation (BB:2390-2455) | Host cost; the profile shows host in-flight idle is not the gap |

### 4.5 Alternatives recorded (Class B against stock)

- **The peer's few-row MMA (dense).** It is likely the fastest dense option
  at 3–16 rows: weights are dequantized once, activation tiles are shared
  over 8-row tiles, and K is split across 8–32 SIMD groups. Its 8×8 MMA
  accumulation and K-split reduction change every row's bits, so it would be
  a Class B route needing the full Q1 envelope at the canonical coordinate
  plus the paired quality campaign.
- **`mul_mv_ext` for K-quants** (ADR-040 §0.21b: +24% at N = 8, not
  bit-exact) and **Q8_0 expert `nr2`** (K split).

None is built in this story. If M2 misses its gate with the Class A levers,
§13 decision 2 is where the operator may open a Class B spike under its own
design.

### 4.6 mlx-native API

Additive only. Existing entry points, kernels and the default policy are
bit-identical, so other families are unaffected.
- **`GgmlRoutingPolicy` gains three fields, all defaulting to the current
  behaviour:**
  - `dense_decode_row_tiled: Option<RowTile>`: `None` is stock; `Some` holds
    (R0, NSG) and covers D1 and D2.
  - `dense_decode_m1_row_tiled: bool` (D1 at m = 1).
  - `expert_mv_id_grouped: Option<GroupedMvId>`: R0, and leader-scan or
    map.
  - They are **not** env-resolved inside mlx-native. Callers set them
    explicitly through the existing `quantized_matmul_ggml_with_policy`
    (`M:src/ops/quantized_matmul_ggml.rs:627`) and
    `quantized_matmul_id_ggml_mv_with_policy` (`M:src/ops/quantized_matmul_id_ggml.rs:360`).
- `plan_dense_auto_route` gains the D1/D2 plans only when the field is set.
  `MM_ROUTING_THRESHOLD = 8` and `MM_ID_ROUTING_THRESHOLD = 32` are
  unchanged.
- Kernels are registered on both metallib paths. The runtime-source
  registration must cover every new kernel; P4 §2.3 records a prior gap.
- Any hf2q struct-literal construction of `GgmlRoutingPolicy` adds
  `..Default::default()` in the pin-bump commit.

### 4.7 hf2q integration

- **Switch.** `HF2Q_GEMMA_DECODE_KERNELS`, read once at model load:
  - `0` is stock: the kill switch, and the default until promotion;
  - `1` selects the routes M0 adopted;
  - any other value fails startup with an error.
  - Gemma resolves one `GgmlRoutingPolicy` from it at load and threads it
    through `dispatch_dense_rowident` (BB:451),
    `dispatch_expert_matmul_id_mv` (`src/inference/expert_dispatch.rs:90-109`)
    and `encode_lm_head_into` (`batched_head.rs:244`).
  - Other families and the serial path (`HF2Q_BATCHED_BODY=0`) keep the
    default policy.
- **Refusal.** The routes apply only to the types in §3.1. Any other weight
  type keeps its stock route, following the `joint_decode_routes.rs` pattern
  (Q1 §3.1 item 4).
- **Surfaces.** The server banner and runtime-control report print the
  resolved decode-kernel policy. The Q5 case file
  `scripts/four_agent/q5_cases/gemma4.json` gains variable
  `GEMMA_DECODE_KERNELS`; the launcher forwards `HF2Q_Q5_GEMMA_DECODE_KERNELS`
  as `HF2Q_GEMMA_DECODE_KERNELS`, in the same pattern as
  `GEMMA_WARM_TILED_ATTENTION`. Catsplit buckets stay the same, so profiles
  are comparable.
- **Comment hygiene.** The BB:1995-1997 comment says `down_proj` is F16 and
  runs per row. On this artifact it is Q5_0/Q8_0 and takes the batched `Mv`
  route. The integration commit corrects the comment.

## 5. Expected savings (estimates, not measurements)

### 5.1 Per lever

Scale: 472 decode steps per run (mean of 445–510) × 0.9 (net catsplit to
production, §2.1) ≈ 425 step-equivalents. The dense gate+up time is not split
out by catsplit. It is estimated at the pooled q/k/v/o rate (172.5 GB/s →
1.16 ms per step). The rest of `dense_ffn` (1.52 ms) is split between router
and down by logical bytes.

| Lever | Class now (ms/step, net) | Low | High | Basis |
|---|---:|---:|---:|---|
| D1 dense K-quant (q/k/v/o + gate/up; 844.5 MB/step) | 4.89 | 1.3× → 0.48 s/run | to the expert-kernel rate 470 GB/s → 1.32 s/run (2× → 1.04 s) | §2.4: R0 = 4 restores the 7:1 ratio the expert kernels run at |
| D1 lm head (Q6_K, 605.6 MB/step) | not measured | 0 | [INFERENCE] ≥ 0.2 s/run | measured in M0; not counted in the totals |
| D2 dense down (560 → 154 MB/step) | ≈ 1.19 | 25% → 0.13 s/run | unique bytes at 470 GB/s → 0.37 s/run | low case if the system cache already serves the re-reads |
| E1 (+R0) experts (3,441 MB/step logical) | 7.02 | only the weight-stream share (30%) × d = 0.35 → 0.30 s/run | time ∝ pairs × (1 − d) → 1.04 s/run | d from §2.4; M0-R replaces it |
| **Total decode GPU per run** | 12.9 of 16.9 | **≈ 0.9 s** | **≈ 2.7 s** | the measured decode-only gap to the peer is 2.0–2.5 s |

[INFERENCE] **Makespan.** Joint decode steps are GPU-bound: in-flight idle is
small, and the host sample loop is 7% of `decode_batch` (#660 §4). A
GPU-second saved in a 4-wide step therefore saves about a wall second for
every active agent at once. Makespan savings are taken as roughly equal to
decode GPU savings, minus the width-1 tail steps.

### 5.2 Against the Win budget

Win needs 14.87 → ≤ 11.85 s, about 3.0 s. The other levers, from the #660 §7
estimates:
- #668 precise warm attention: 1.2–1.4 s;
- cold-turn admission: 0.4–0.5 s;
- serving speculation: 0.4–0.5 s.

Together that is 2.0–2.4 s, which leaves **≥ 0.6–1.0 s for decode kernels**.
#660 §7 put this lever at 0.3–0.7 s from the decode delta alone. The kernel
analysis above supports more.

The story's M2 gate is the low end of the budget: a median ratio of ≤ 0.96
against the kill switch, about 0.6 s at 14.87 s.

**Overlaps.**
- Speculation removes target decode steps, which shrinks D1 and D2 in
  proportion.
- Wider verify rows raise E1's share, because more rows share each expert.
- The combined candidate is the only valid total (§8.4).

## 6. Tests

### 6.1 mlx-native primitive parity (model-free, GPU; both metallib paths)

New test files follow the style of `tests/q4k_mv_mn_byte_parity.rs` and
`tests/adr_040_q6k_mv_mN_byte_parity.rs`. Inputs are seeded random weights
(valid blocks; scales including 0 and subnormal halves) and activations
(including ±0, subnormals, large magnitudes, and one NaN/∞ column for
invariant 3). All comparisons are u32 equality against the §3.1 reference on
the same path:

| Test | Covers |
|---|---|
| T-N1 `gemma_dense_row_tiled_q4k_q6k_bitwise` | every §2.5 Q4_K/Q6_K shape incl. lm head N = 262,144; m = 1..8; every (R0, NSG) in the M0 space; tiled m > 4 |
| T-N2 `gemma_dense_q5_0_q8_0_mn_bitwise` | K = 2112 → N = 2816 (and K = 704, N = 2816); m = 1..8 |
| T-N3 `gemma_expert_grouped_mv_id_bitwise` | gate_up Q4_K (n = 1408, k = 2816) and down Q5_0/Q8_0 (n = 2816, k = 704); n_tokens 1..16; patterns: all distinct, all one expert, the §2.4 empirical patterns (rows 128–143 of the witness, replayed), random, invalid ids at first/middle/last position; leader-scan and map variants |
| T-N4 `gemma_decode_kernels_permutation_and_isolation` | column permutation; NaN/∞ column does not perturb neighbours; output views at non-zero offsets; bytes outside views unchanged |
| T-N5 `gemma_decode_kernels_metallib_paths` | precompiled vs runtime outputs recorded for every T-N1..T-N3 cell (each path already equals its own reference) |
| T-N6 (only if F1/F2 adopted) `gemma_gelu_epilogue_bitwise` | epilogue output equals `fused_gelu_mul` / `moe_swiglu_batch` after the stock products |
| T-N7 `routing_policy_default_unchanged` | default policy plans exactly the stock routes for every type × m (a table test, model-free, no GPU) |

### 6.2 hf2q model-free

- Switch parsing (`0`, `1`, rejects others) and banner text.
- Policy threading: with `0`, every call site resolves the default policy
  (asserted through the existing route trace `HF2Q_DECODE_MVN_TRACE`-style
  hooks or a pure plan function).
- Refusal for types outside §3.1.

### 6.3 hf2q real-artifact Class A gates (ignored; one process each; `--exact --ignored --test-threads=1`)

| Gate | What it proves |
|---|---|
| A1 new `gemma_decode_kernels_match_stock_bitwise` | One process, the same four ragged lanes decoded ≥ 256 steps under the stock policy and under the candidate policy. Bitwise per step: logits, sampled token, F16 K and TQ-HB V cache bytes, per-layer router ids and weights (stage capture at BB:1986-1993). It exercises sliding *and* global layers, widths 4 → 3 → 1 as lanes finish, and one mid-run admission |
| A2 `slot_aware_n1_byte_equivalent_to_serial_slot_aware`, `slot_aware_n4_per_slot_parity_vs_serial`, `slot_aware_n4_batched_body_same_prompt_parity_vs_serial`, `slot_aware_n8_per_slot_parity_vs_serial`, `slot_aware_staggered_eviction_no_peer_perturbation` (`src/serve/api/engine.rs:36956, 37831, 38061, 38191, 39642`), with `HF2Q_GEMMA_DECODE_KERNELS=1` | Batched = serial still holds; release build |
| A3 `gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant` (`engine_gemma4_tiny_prefill_tests.rs:16`) | Width-8 decode then tiny cold prefill, repeat-invariant |
| A4 `q1_gemma_actual_cohort_endpoint_trace` (`src/quality/adr063_invariance/gemma_actual.rs:182`) under `0` and `1`, two processes, then `q1_decision_policy.py exact` | `ExactEqual` between arms over the T2 cohort store (Q1 tooling) |

## 7. Microbenchmark plan (M0; model-free; root GPU lane, quiet host)

### 7.1 Harness

`benches/bench_gemma_decode_w4.rs` (mlx-native, `harness = false`):
- **Timing.**
  - GPU time comes from command-buffer `GPUStartTime`/`GPUEndTime`, with 30
    dispatches per command buffer (one per "layer") to amortize the fixed
    command-buffer cost that inflates catsplit.
  - 10 warm-up and ≥ 50 timed iterations per cell; report the median and
    p10/p90.
- **Cold weights.** There are 30 distinct weight buffers per shape, one per
  layer, cycled so that a cell's weights exceed the system cache, as in
  production. Reusing one weight buffer would measure cache-warm reads.
  Activations are fresh per iteration.
- **Roofline.** A streaming read kernel over ≥ 2 GiB gives the host's
  achievable read bandwidth. Every cell reports GB/s against it.
- **Paths.** Timings run on the precompiled metallib, the production default.
- **Receipt.** JSON per cell: shape, type, m, pattern, kernel and geometry,
  median ns, logical and unique bytes, GB/s. It also records the mlx-native
  commit, metallib SHA-256, host, macOS build and HOST-LOCK holder.

### 7.2 Cells

| Family | Shapes | m / patterns | Arms |
|---|---|---|---|
| D1 | q 2816→4096 and →8192; k →2048 and →1024; v Q4_K and Q6_K 2816→2048; o 4096→2816 and 8192→2816; gate/up 2816→2112; lm head 2816→262,144 | m = 1, 2, 3, 4 (+ 5..8 non-regression) | stock `mN` (and stock m = 1 for m = 1); D1 over (R0, NSG, staging); F1 dual gate+up |
| D2 | 2112→2816 Q5_0 and Q8_0 | m = 1..4 (+ 5..8) | stock `Mv` grid.y = m; D2 |
| E1 | gate_up Q4_K 2816→1408 × 128; down Q5_0 and Q8_0 704→2816 × 128 | n_tokens = 4: U = 32 (distinct), U from M0-R (empirical replay), U = 8 (one expert per slot shared by all rows), skewed (one expert in all 4 rows + distinct rest); n_tokens = 8 and 16 (verify widths) | stock `mv_id`; E1 leader-scan and map, R0 ∈ {1, 2, 4}; F2 epilogue |

### 7.3 Peer reference (M0-P; recommended, not a gate)

This runs the peer's backend-op performance mode (`test-backend-ops perf`,
`MUL_MAT` and `MUL_MAT_ID`) at the §7.2 shapes on the same host.
- It is built from the reference checkout
  `/opt/hf2q-worktrees/adr063-peer-reference-e86f5fd1` into a *separate*
  build directory. A test-only case-list patch is kept with its diff; the
  serving build and its receipt stay untouched.
- It gives the peer's per-kernel time at Gemma's shapes, which is the direct
  answer to "what does the peer pay per kernel".
- It is not a gate, because the peer kernels are not exact and other host
  state differs.

### 7.4 Routing overlap on the real workload (M0-R; diagnostic, one run, untimed)

A diagnostic switch `HF2Q_GEMMA_ROUTING_OVERLAP=1` works as follows:
- After each layer's routing, it blit-copies `moe_expert_ids` into a per-step
  `[30, N, 8]` u32 buffer: 3.8 KiB per step.
- It reads the buffer after the step's existing final wait, so it adds no
  extra synchronization.
- At worker exit it prints a histogram of U per (layer, width), in the style
  of `[CATSPLIT]`. It is off by default and costs nothing when off.

One supervised four-agent run with Q5 `CalibrationPass` gives the U
distribution that replaces §2.4's proxy. The histogram feeds the E1 replay
cells and the §5 recomputation.

### 7.5 M0 adoption thresholds (frozen here)

| Lever | Adopt if (precompiled, median) | And never |
|---|---|---|
| D1 | Over the §2.5 shape mix weighted by per-step count, at m = 4: ≤ 0.80 × stock `mN` time (≥ 1.25×). The lm head is decided separately at the same threshold | any cell at m ∈ 1..8 slower than its stock route by > 2% |
| D2 | Each of Q5_0 and Q8_0 at m = 4: ≤ 0.67 × stock (≥ 1.5×) | > 2% slower at any m |
| E1 | At the M0-R empirical pattern, gate_up + down: ≤ 0.85 × stock | > 2% slower at U = 32 (all distinct) |
| F1, F2 | ≥ 3% better than the adopted non-fused route | any T-N6 mismatch |

The best geometry per shape is frozen into the dispatcher as a static table.
Geometry never depends on data or timing at run time, so outputs stay fixed
(§3.2 item 2). A lever that misses its threshold is dropped and recorded, not
integrated.

## 8. Acceptance gates

### 8.1 Exactness (all must pass; any byte fails)

- T-N1..T-N5 (and T-N6 if fused) on both metallib paths; T-N7.
- A1–A4 on the release candidate build of the integration branch.
- `ExactTranscripts` (`q1_decision_policy.py transcripts`) over every M2 run,
  candidate vs kill switch: 0 same-request/different-output groups.

### 8.2 Speed

| Gate | Measurement | Threshold |
|---|---|---|
| M0 | §7.5 | per lever |
| M1 in-situ | `slot_aware_n4_batched_body_throughput_probe` (`engine.rs:39291`) with `HF2Q_GPU_BUSY=1`, `HF2Q_BENCH_N=4` and `=1`, release, the real artifact; ≥ 512 steps per round, 3 rounds per arm alternating `0`/`1`, warm-up round discarded; per-step GPU busy, not wall tok/s (ADR-040 §0.21c: wall ±14% on this harness) | N = 4: candidate ≤ 0.90 × kill per-step GPU (≥ 10%). N = 1: ≤ 1.01 × kill |
| M2 four-agent A/B | Q5 supervisor, same binary, `GEMMA_DECODE_KERNELS ∈ {0, 1}`, ABBA blocks with a thermal gap, ≥ 4 runs per arm (extend to 8 if the CI half-width exceeds 2.5 points; no other extension) | median makespan ratio `1`/`0` ≤ 0.96 and block-paired 95% CI upper < 1.00; decode stage (L1 extraction) reported per turn |
| M2 diagnostics (recorded, not gated) | `gemma_stage_profile.py` per-stage medians; one catsplit run per arm | decode-category shares before/after |

### 8.3 Fairness and memory

- In M2, the candidate vs the kill switch: worst-slot TTFT p95 and SSE-gap
  max ratios ≤ 1.025. Faster steps should only shorten gaps.
- Every run passes Q5 `CalibrationPass`. No new persistent buffers exist:
  the leader scan needs none, and the optional map is ≤ 1 KiB per layer. The
  fresh Q5 on the release candidate is still required, because the source
  changes (Q5 recalibration rule).

### 8.4 Quality and the Q4 verdict (combined candidate; owned by the Q4 closing campaign)

This story adds no Class B route, so its own quality obligation is
`ExactTranscripts` (§8.1). The Gemma family verdict is decided once per
release candidate on the **combined** candidate: this route + #668 + the
companion Gemma stories. The `q3_campaign.py` closing campaign
(`adr063-gemma-tiled-attn-design.md` §9.3) requires:
- `family_verdict == Win` against the peer's MT configuration (CI upper ≤
  0.95);
- paired quality `Pass` against v0.1.28 and against the peer (24 instances);
- worst-slot TTFT p95 and SSE-gap max against v0.1.28 ≤ 1.025;
- a fresh Q5.

## 9. GPU validation plan (root-run; GPU lane and HOST-LOCK; one process at a time)

Every run records the source commit, binary SHA-256, artifact digest
`bf65b7e1…`, mlx-native version and metallib SHA-256, and case file and
SHA-256. Evidence goes under `~/.local/state/hf2q/v0130/q4-gemma-decode/`.

| Step | Where | Model? | Command (shape) | Output |
|---|---|---|---|---|
| V0 | mlx-native branch | no | `MLX_PRECOMPILED_METALLIB=1 cargo test --release --test gemma_decode_w4_exact -- --test-threads=1`, then `=0` | T-N1..T-N7 receipts, both paths |
| V1 | mlx-native branch | no | `cargo bench --bench bench_gemma_decode_w4` (quiet host) | M0 cells, roofline; geometry table frozen |
| V1-P | peer reference build dir | no | `test-backend-ops perf -o MUL_MAT` / `-o MUL_MAT_ID` with the §7.2 case list | peer per-kernel times (reference) |
| V2 | hf2q diagnostic build | yes | `HF2Q_Q5_GEMMA_ROUTING_OVERLAP=1 python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json --negative-control-receipt "$NC" --output …/m0r-overlap-<ts>` | U histogram; replay patterns into V1's E1 cells; recompute §5 |
| V3 | after native release + exact pin | yes | A1, A3 and A2 with `HF2Q_BYTE_EQUIV_E2E=1 HF2Q_BYTE_EQUIV_E2E_GGUF=$GEMMA HF2Q_GEMMA_DECODE_KERNELS=1 cargo test --release --bin hf2q <name> -- --exact --ignored --test-threads=1`; A4 under both values, then `q1_decision_policy.py exact` | Class A receipts |
| V4 | same binary | yes | M1 probe, `HF2Q_GPU_BUSY=1 HF2Q_BATCHED_BENCH=1 HF2Q_BENCH_N={4,1}`, alternating `HF2Q_GEMMA_DECODE_KERNELS` | M1 receipt |
| V5 | same binary | yes | `for ARM in 0 1 1 0 …; do HF2Q_Q5_GEMMA_DECODE_KERNELS=$ARM python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json --negative-control-receipt "$NC" --output …/q5-ab-$ARM-<ts>; done`; then `q1_decision_policy.py transcripts` over all runs and `gemma_stage_profile.py` | M2, `ExactTranscripts`, fairness |
| V6 | combined release candidate | yes | `q3_campaign.py --family gemma4 …` (§8.4) | Q4 verdict (root, once per candidate) |

Order: V0 → V1 (→ V1-P) → V2 can run in parallel with the native review →
native merge, release, crates.io check, exact pin → V3 → V4 → V5 → promotion
→ V6. Halt at the first failure. Preserve the failed receipts; no retry until
pass.

## 10. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Kernel wins do not reach makespan (K1: exact Q8_0 multi-row +1.2%, n.s.; Qwen projected 2.2×, got 4.9%) | Story gate missed | M1 in-situ GPU gate before the A/B. §5 states low and high cases. M2 is the only promotion basis |
| The catsplit-derived rates mislead (serialized command buffers, 4.2× run length) | Wrong lever priority | M0 measures each kernel in isolation with cold weights and a roofline. M1 measures in situ. §2.2 is used for ranking only |
| FMA contraction or other compiler differences break bits | Route blocked | §3.2 item 7 operand-form rule; T-N1..T-N5 at every shape on both paths; A1 at model level |
| Register pressure or occupancy at (R0, R1) = (4, 4) (ADR-040 spill cliff) | D1 slower at small N | Geometry search per shape, R1 ≤ 4 per tile, the frozen table may pick R0 = 1 (= stock) per shape; the no-regression rule |
| Real-workload expert overlap below the proxy | E1 saves less | M0-R measures it first. The E1 adopt threshold uses the empirical pattern. U = 32 no-regression |
| Leader-scan imbalance (popular experts serialize) | E1 tail | Skewed-pattern cells. The map variant with compact grid is the alternative |
| Interaction with speculation and #668 | Savings not additive | The combined candidate is the only Q4 total. E1 widths cover verify rows |
| Other families pick up the new routes unintentionally | Unreviewed numerics elsewhere | Policy fields default off, not env-resolved in mlx-native; T-N7 |
| Peer reference build drifts from the measured peer | Misleading per-kernel comparison | Separate build dir at `e86f5fd1`; reference only, never a gate |

## 11. Release and promotion

1. **mlx-native.**
   - A contract issue carries the §3 text.
   - The PR carries T-N1..T-N7 receipts on both paths, the V1 receipts and
     the frozen geometry table.
   - Independent review, merge, then publish the next 0.15.x and verify it on
     crates.io.
   - hf2q pins it exactly (`=0.15.x`) in its own commit, from a clean
     registry build.
2. **hf2q integration** with the switch defaulting to `0`; V3–V5.
3. **The promotion commit** runs only when all §8.1–§8.3 gates pass. It:
   - flips the default to `1`, keeping `0` as the kill switch;
   - deletes any non-adopted lever's code (clean cutover);
   - updates the banner, case-file notes, `scripts/four_agent/README.md` and
     the ADR-063 Q4-GEMMA-DECODE receipts.
4. **If M2 fails,** the default stays `0`. The receipts are recorded, and §13
   decision 2 applies.

## 12. ADR-063 story section (extraction-ready)

### Q4-GEMMA-DECODE. Make Gemma 4 four-slot decode kernels exact and weight-bound

Outcome: Gemma 4 26B-A4B joint decode at four slots spends less GPU time per step in its dense projections and routed experts, and every output stays bitwise equal to today's decode.

Requirement: mlx-native MUST provide row-tiled dense and expert-grouped decode kernels under a stated exactness contract (Q1 Class A against the stock per-row kernels). The program MUST integrate them into the Gemma batched decode body behind a kill switch. It MUST deliver the per-step GPU and makespan saving that the Gemma Win budget assigns to decode kernels.

Acceptance:

- Starting point (Q4-GEMMA #660, hf2q `7f295b38`, mlx-native 0.15.16): the decode body splits into routed MoE 40% (gate/up 21.8%, down 15.3%), dense projections and FFN 37.5%, attention 17.8% and norms 4.5%. In decode-only windows hf2q is GPU-busy 2.0–2.5 s per run longer than the peer's MT configuration. At width 4 the Q4_K/Q6_K dense projections move their weights at 155–184 GB/s, against 469–522 GB/s for the expert products in the same run. Dense `ffn_down` (Q5_0/Q8_0) streams its weight once per row. Both engines stream each (token, expert) pair separately, although the four rows share about 35% of their expert selections (Q1 witness proxy).

- Design approved before implementation: `docs/research/adr063-q4-gemma-decode-design.md` §3 (exactness contract), §4 (kernels D1, D2, E1 and optional F1/F2), §7 (M0 thresholds), §8 (gates).

- A linked mlx-native issue states the §3 contract before implementation. The kernels are:
  - row-tiled Q4_K/Q6_K `mN`, including the lm head;
  - Q5_0/Q8_0 `mN`;
  - expert-grouped `mv_id` for Q4_K gate_up and Q5_0/Q8_0 down, admitting up to 16 tokens × top-8.

  New routing-policy fields default off, and every existing entry point stays bit-identical.

- Primitive parity: u32-bitwise against the stock m=1 kernels at every Gemma shape, for m = 1..8, all-distinct, all-shared, empirical and invalid-id expert patterns, and both metallib paths. The M0 microbenchmark uses cold-weight rotation and a host roofline; each lever is adopted only at its frozen §7.5 threshold.

- hf2q: `HF2Q_GEMMA_DECODE_KERNELS` (`0` is the kill switch). Class A gates:
  - a stock-equivalence test that compares logits, tokens, KV, and expert ids and weights bitwise against `0` across sliding and global layers;
  - the slot-aware n1/n4/n8 and eviction parity tests;
  - `ExactEqual` between arms on the Q1 T2 cohort trace.

- Speed:
  - in-situ decode-step GPU at width 4 ≤ 0.90 × the kill switch, with no width-1 regression;
  - four-agent Q5 A/B on the same binary (ABBA, at least 4 runs per arm): median makespan ratio ≤ 0.96 with a block-paired 95% CI upper bound below 1.00;
  - `ExactTranscripts`;
  - worst-slot TTFT p95 and SSE-gap max ratios ≤ 1.025.

- The default flips only after every gate passes. The Gemma verdict is then decided on the combined candidate in the Q4 closing campaign: Win against the peer's MT configuration, paired quality Pass against v0.1.28 and against the peer, stock fairness 1.025, and a fresh Q5.

Repository ownership: mlx-native owns the kernels, dispatchers, policy fields, contract, parity tests, microbenchmarks and release. hf2q owns route selection, the switch, integration, the Class A gates and the campaigns, consuming an exact published mlx-native pin.

Validation: Independent review of the contract, the parity and microbenchmark receipts, the Class A receipts and the A/B from raw receipts.

Predecessors: Q4-GEMMA (#660, profiling result), Q5 (#371). Independent of Q4-GEMMA-KERNEL and Q4-GEMMA-ATTN. Its grouped expert kernel serves the companion Gemma speculation story's verify rows.

Priority: High. Core scope (second-tier family).

Delivery boundary: A published mlx-native release pinned exactly by hf2q, and the integration merged for v0.1.30 (or the first published hf2q release containing the accepted Gemma 4 configuration).

## 13. Decisions for the operator

1. **Approve the Class A-only scope.** Every lever must be bitwise equal to
   stock, and the peer's MMA, `mul_mv_ext` and K-split routes are excluded.
   The alternative is to admit a Class B dense route now, at the cost of a
   full Q1 envelope and a paired quality campaign on a route-sensitive
   family.
2. **If M2 misses ≤ 0.96:** do one of the following:
   - (a) accept the measured saving and re-budget the Win across the other
     Gemma stories;
   - (b) open a Class B few-row MMA spike under its own design;
   - (c) re-scope the Gemma Win (ADR-063 Q4).
3. **Approve the M0-P peer reference build:** a separate build directory
   with a test-only case patch, not a gate.
4. **Decode attention (17.8% of the decode body) is unowned.** No v0.1.30
   story covers it. Decide whether to leave it, or to add it to this story
   or a later one.
5. **Approve the switch name and the promotion rule:** default `0` until
   §8.1–§8.3 pass, then `1`, with `0` kept as the kill switch.

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
