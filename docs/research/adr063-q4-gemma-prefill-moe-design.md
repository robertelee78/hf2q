# ADR-063 Q4-GEMMA-PREFILL-MOE: Gemma 4 warm-prefill MoE and dense cost per row — HLD/LLD

Status: design for approval (2026-11-04). Revision 2 applies the
ACCEPT-WITH-CONDITIONS review (ReviewGemmaPrefillMoe, findings 1–14). No code,
build or hardware run backs this document. Every number is quoted from a
retained receipt (path given), computed from source or file metadata (method
given), or marked `[INFERENCE]`.

Story: **Q4-GEMMA-PREFILL-MOE** (new, v0.1.30, operator decision 2026-11-04).
It is one of four stories that close the Gemma 4 26B-A4B gap beside
Q4-GEMMA-ATTN (#668). It feeds the combined candidate that Q4-GEMMA-ATTN's
closing campaign measures.

Sources:
- ADR-063 sections Q4, Q4-GEMMA, Q4-GEMMA-ATTN, Q5.
- The Q4-GEMMA profile record (#660, PR #687, branch `feat/q4-gemma-profile`):
  `docs/research/adr063-q4-gemma-profile-result.md`.
- The Q1 numeric policy `hf2q.q1.numeric-policy.v1`
  (`docs/research/adr063-q1-numeric-policy.md`).
- The style and campaign conventions of `adr063-gemma-tiled-attn-design.md`.
- hf2q main `ff4af4bd`; mlx-native 0.15.16, the current pin (the routed-matmul
  files are byte-identical in 0.15.17, §13).
- The peer's source at its build commit `e86f5fd` (build `b128-e86f5fd`, the
  binary the profile measured): the expert-routed matrix kernels and their host
  dispatch.
- Tensor metadata of the target GGUF (`bf65b7e1…`), read from the file header
  only. No model was loaded.

`$G` = `/Users/robert/.local/state/hf2q/v0130/q4-gemma` (profile evidence).
`$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`.

## 1. Decision summary

1. **The MoE stage is the lever. Dense and pre-attention are not.** Per FLOP,
   warm-prefill MoE runs at about half the rate of the dense and pre-attention
   projections on the same tensor-API kernels (§2.4): 13.0 vs 27.1 and 29.4
   TFLOP/s, all categories inclusive. The dense GEMM tiles are already 93% full
   at the served row counts. This story changes MoE only. Dense and
   pre-attention get attribution, not kernel changes (§2.6).
2. **Hypothesis: much of the MoE excess is launch geometry.** hf2q's `mm_id`
   launches a uniform grid of 128 experts × ⌈rows/32⌉ tiles × N tiles. The
   Gemma MoE down call also flattens its rows to `8·n` tokens, which
   multiplies its grid by 8. Over one four-agent run that is about **225 M
   threadgroups launched, of which an estimated 3.1 M have work** (routing
   proxy, §2.4). The peer build's dispatch launches only a compact list of
   tiles that have work, and it skips the empty half of half-filled tiles
   (§2.5). This stays a hypothesis until M0b measures it. Part of the category
   is expert-weight streaming (about 0.15–0.30 s per run `[INFERENCE]`, §2.4),
   and no change here reduces that part.
3. **Four changes, all Class A (bitwise against today's route):**
   - **P1 (hf2q only, on the 0.15.16 pin).** The Gemma prefill MoE down call
     moves to the existing slotted `mm_id` entry point. This removes 184.5 M of
     the 224.8 M launched threadgroups per run with the same rows, addresses and
     tiles.
   - **P2 (mlx-native).** `map0` emits a compact per-expert tile list, and the
     `mm_id` grid is bounded by it (the peer build's scheme), with the same tile
     membership and order. This brings the run to about 4.0 M launched
     threadgroups.
   - **P2b (mlx-native).** The down call reuses the gate/up routing schedule,
     saving one `map0` and one barrier per layer.
   - **P2c (mlx-native, conditional).** Half-tile skip. It ships **only** if it
     is bitwise against today's kernel; otherwise it is recorded as falsified
     and dropped.
4. **Exactness.** Every change preserves the row-independence rule that joint
   warm (and the planned joint cold) prefill relies on: a row's output depends
   only on its input, its experts and the weights (§3.1). The `mm_id`/`mv_id`
   threshold (32), the dense `mm`/`mv` threshold (8), the tensor-API selection
   and the expert storage types are unchanged. Quality is inherited from stock
   through `ExactTranscripts` (Q1 §3.6).
5. **Expected saving: 0.08–0.28 s per run** (estimate, §9.1), within the
   profile's "≤ 0.3 s after #668" bound. The range holds only if an empty
   threadgroup costs c_e ≥ 0.25 ns of GPU time, and c_e is unmeasured. The
   first GPU step (M0, §8) measures it model-free, on today's kernels, before
   any native implementation. §6 freezes the stop rule:
   - if the predicted saving of P1 + P2 + P2b is below 0.05 s per run, P2,
     P2b and P2c are not built;
   - P1 still merges as a bitwise cleanup, with no speed claim;
   - its switch is removed in the same PR (§7.4).
6. **This story alone cannot make Gemma Win.** Its gates are a category
   saving with no makespan regression. The family Win is measured on the
   combined candidate in Q4-GEMMA-ATTN's closing campaign.

## 2. Evidence

### 2.1 Where warm-prefill time goes

Profile split (`HF2Q_GEMMA_PREFILL_CATSPLIT=1`, one command buffer per
category per layer, serialized; `$G/profile/l3/0{21,23,25}-*/server.stderr`,
`[PREFILL_CATSPLIT_JSON]`). Each run has 4,988 warm rows over 24 lanes:

| Run | Forwards | pre-attn ms | attention ms | dense ms | MoE + layer tail ms | total ms |
|---|---:|---:|---:|---:|---:|---:|
| 021-b01-hc | 6 | 238.4 | 1530.5 | 348.3 | 1094.0 | 3211.2 |
| 023-b02-hc | 7 | 242.9 | 1542.2 | 351.9 | 1112.3 | 3249.4 |
| 025-b03-hc | 6 | 238.7 | 1528.5 | 348.7 | 1090.7 | 3206.6 |
| share (mean) | | 7.5% | 47.6% | 10.8% | 34.1% | |

The split perturbs warm prefill by +0.47 s (+13%, profile §4, HC vs H). The
gates in §7 therefore use the **unprofiled** per-forward wall that every run
logs (`Batched prefill complete: N tokens in T ms`).

Unprofiled warm forwards. Each line is one `forward_prefill_batched_multi_seq_live`
transaction (`transaction="joint warm" requests=4`), from `server.stderr`:

| Turn | Rows (4 lanes) | L3 `h` runs 022–030, ms | L1 hf2q runs 001–013, ms |
|---|---:|---|---|
| 1 | 232 | 137.4–138.2 | 138.5–150.2 |
| 2 | 764 | 405.0–406.3 | 428.8–447.2 |
| 3 | 880 | 492.4–494.0 | 492.4–573.3 |
| 4 | 1172 / 1180 | 767.1–780.7 | 781.6–833.9 |
| 5 | 528 | 389.1–390.4 | 390.1–430.5 |
| 6 | 1412 / 1420 | 999.8–1001.9 | 1001.1–1050.3 |
| sum per run | 4988 / 5004 | 3194.8–3243.1 | 3255.0–3529.7 |

Two of 12 runs split one cohort: L1 `004-b02` ran turn 6 as 4 × 355 (1,145 ms
against 1,006–1,050 ms joint), and L3 `030-b05` ran turn 4 as 879 + 293. **Every
other warm turn is a four-lane joint cohort of 232–1,420 rows.** All four agents
wait on that one forward, so a saving in its GPU time reaches each agent's span
and the makespan almost one for one `[INFERENCE]`. The warm-prefill fit (2.13
vs 1.18 s per 1k new rows) charges the whole joint forward to each lane and
includes attention per row. It sizes the gap but does not apportion it; this
design uses the measured split.

### 2.2 Served shapes (target GGUF header, `bf65b7e1…`)

| Tensor | Type | Shape (K → N) | Layers |
|---|---|---|---|
| `ffn_gate_up_exps` | Q4_K | 2816 → 1408 (gate 704 + up 704), 128 experts | 30 |
| `ffn_down_exps` | Q5_0 | 704 → 2816, 128 experts | 16 (3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25) |
| `ffn_down_exps` | Q8_0 | same | 14 |
| `ffn_gate`, `ffn_up` | Q4_K | 2816 → 2112 | 30 |
| `ffn_down` | Q5_0 / Q8_0 | 2112 → 2816 | same split as the expert down |
| `ffn_gate_inp` (router) | F32 | 2816 → 128 | 30 |
| `attn_q` / `attn_k` / `attn_output` | Q4_K | sliding 2816→4096 / 2048 / 4096→2816; global 2816→8192 / 1024 / 8192→2816 | 25 sliding, 5 global |
| `attn_v` | Q4_K (12) / Q6_K (13) | 2816 → 2048 | 25 sliding (global layers share K/V) |

The model has `expert_used_count` 8 and `expert_count` 128.

### 2.3 Route map today (warm forward of n rows, n ≥ 33)

From `forward_prefill_batched.rs:4804-4823` (gate/up), `:4959-4978` (down),
`expert_dispatch.rs:111-132`, and mlx-native `quantized_matmul_id_ggml.rs` /
`ggml_capability.rs`:

| Op | hf2q call | mlx-native route | B operand | Grid (threadgroups) |
|---|---|---|---|---|
| MoE gate/up | pooled, `n_tokens = n`, `top_k = 8` | `map0_ne20_8` + `kernel_mul_mm_id_q4_K_tensor_f32` | staged as half | 128 × ⌈n/32⌉ × 22 |
| GELU·up | `moe_swiglu_seq` (F32, `precise::tanh`) | elementwise | F32 | — |
| MoE down | pooled, **`n_tokens = 8n`, `top_k = 1`** (flattened) | `map0_ne20_1` + `…q8_0_tensor…` (14 layers) or `…q5_0_f32` simdgroup (16 layers; Q5_0 is excluded from the tensor path, `ggml_capability.rs:572-583`) | half (tensor) / F32 (simdgroup) | 128 × ⌈8n/32⌉ × 44 |
| weighted sum + norm + add | `fused_moe_wsum_dnorm_add_f32` | elementwise | F32 | — |
| dense gate, up, down, O-proj, Q/K/V | `dispatch_qmatmul`, m = n | V2 large-tile tensor mm (64 × 128) for Q4_K/Q6_K/Q8_0; simdgroup V1 for Q5_0 | F32 from device (V2) | ⌈n/128⌉ × ⌈N/64⌉ |
| router | `dispatch_qmatmul` (F32 weight) | `dense_matmul_f32_f32_tensor` | F32 | — |

Thresholds: expert `mm_id` when `n_tokens > 32` and `top_k ∈ {1, 6, 8}`
(`MM_ID_ROUTING_THRESHOLD`, `plan_expert_auto_route`); dense `mm` when `m > 8`.
At the served row counts (232–1,420), every op takes the matrix-matrix route.

### 2.4 Diagnosis

**Efficiency by category** (FLOPs computed from §2.2 shapes; the times are the
catsplit means, which include each category's non-matmul ops):

| Category | GFLOP per row (30 layers) | TFLOP per run | GPU s per run | TFLOP/s |
|---|---:|---:|---:|---:|
| MoE + tail | 2.855 | 14.24 | 1.099 | 13.0 |
| dense | 1.900 | 9.48 | 0.350 | 27.1 |
| pre-attention | 1.413 | 7.05 | 0.240 | 29.4 |

At the dense rate MoE would take about 0.53 s, about 0.57 s less. That is a
ceiling for discussion, not a target.

**Launched vs useful threadgroups.** Per layer and forward:
- `mm_id` launches 128 experts × ⌈rows/32⌉ × ⌈N/64⌉ threadgroups. Every
  threadgroup reads its expert's count and exits if its tile is past the end
  (`quantized_matmul_id_mm.metal:660-664`; same in the tensor kernel).
- Busy tiles are Σₑ⌈cₑ/32⌉, where cₑ is expert e's routed rows.

| Rows n | gate/up launched | down launched (flattened) | down launched (slotted) | busy tiles × N tiles (gate/up, down) | compact grid (gate/up, down) |
|---:|---:|---:|---:|---:|---:|
| 232 | 22,528 | 326,656 | 45,056 | ≈ 2,506 / 5,012 | 4,070 / 8,140 |
| 1,420 | 126,720 | 1,999,360 | 253,440 | ≈ 8,906 / 17,811 | 10,604 / 21,208 |
| **per run** (6 forwards × 30 layers) | 13.43 M | **211.37 M** | 26.86 M | ≈ 1.04 M / 2.07 M | 1.33 M / 2.66 M |

The per-run row uses the forward sizes 232, 764, 880, 1,180, 528 and 1,420. Busy
tiles use a routing proxy:
- 20 bootstrap draws per layer of n rows from the greedy144 routing witness
  (`$E/q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl`,
  17,280 rows, top-8 of 128);
- averaged over layers, a layer's hottest expert receives 79% of the drawn
  rows, so the proxy is far more skewed than uniform routing;
- uniform routing gives a busy-tile count within 12% of the proxy's (1–12%
  across the six sizes).

The proxy is `[INFERENCE]` until M0b records real agent routing (§8).

**Hypothesis: today's run launches 224.8 M threadgroups, of which an estimated
3.1 M do work** (proxy). The empty ones only read one word and exit, but each
still costs launch and scheduling time. If an empty threadgroup costs c_e of
GPU time (amortized across cores), the launch waste per run is:
- 0.056 s at c_e = 0.25 ns;
- 0.11 s at 0.5 ns;
- 0.22 s at 1 ns;
- 0.45 s at 2 ns.

c_e is unmeasured. M0a measures it, and M0b confirms or refutes the
hypothesis on served shapes.

**Tile fill.** With 32-row tiles, routed rows fill 51% of the computed tile
rows at n = 232 and 88% at n = 1,420. Weighted by the workload's rows, the fill
is 80%. If tiles with ≤ 16 rows compute only their lower half (P2c), the
weighted fill rises to 90%, so the matrix work falls by 11% (proxy).

**`map0`** runs as one threadgroup of 128 threads, so a single core works
while the rest of the GPU waits at the barrier:
- each thread scans every token and repeats the O(top_k²) distinctness check
  that every other thread also does (`quantized_matmul_id_mm.metal:515-555`);
- the flattened down call scans 8n "tokens";
- `[INFERENCE]` cost: tens of µs per call, 360 calls per run, so a few ms to
  about 10 ms per run.

**Not causes at these shapes:**
- the `mm_id`/`mv_id` threshold: no warm forward has ≤ 32 rows;
- the dense route: V2 tiles are 82–99.5% full, 93% weighted;
- cohort width: already four lanes, except in the 2 of 12 runs above.

**A real cost this story does not reduce: expert-weight streaming.**
`[INFERENCE]` from §2.2 sizes:
- The gate/up stack is 285 MB per layer. The down stack is 174 MB (Q5_0) or
  269 MB (Q8_0).
- The proxy gives 2.7–3.2 busy tiles per expert at n = 1,180–1,420, not ≤ 2.
- Per run, the expert slabs stream about 91 GB if each expert is read once per
  forward, and about 186 GB if read once per busy tile.
- At this host's bandwidth that is about 0.15–0.30 s of the 1.1 s category.

P1, P2 and P2c do not change it. Only fewer row tiles per expert would, that
is, larger or fuller tiles, which §2.6 rejects. So at most about 0.8–0.95 s of
the category is launch, `map0`, staging, matrix and elementwise work that this
design can address.

### 2.5 What the peer build does differently

Peer source at `e86f5fd`, `kernels/mul_mm.metal` and `ggml-metal-ops.cpp`. hf2q's
`mm_id` is a port of an older form of the same kernel:

| Aspect | hf2q (mlx-native 0.15.16) | peer build `e86f5fd` |
|---|---|---|
| Routing table for down | flattened `[8n]` ids, `top_k = 1` | ids stay `[n, top_k]`; the input is `[K, top_k, n]` (slotted) |
| `mm_id` grid | 128 × ⌈n_tokens/32⌉ × N/64 | `max_tiles` × N/64, where `max_tiles = min(rows, ⌈rows/32⌉ + min(n_expert, rows) − 1)` |
| `map0` output | per-expert count plus `[n_experts, n_tokens]` lists | pass 1 counts; thread 0 prefix-sums and emits one (expert, first entry, valid rows) descriptor per 32-row tile; pass 2 writes a compact list `[rows + n_expert]` |
| Partial tiles | always 32 rows of MMA | tensor path: two 16-row `matmul2d` halves, the upper skipped when `nr1 ≤ 16`; simdgroup path: simdgroups 2–3 idle |
| Invalid ids | bounds and distinctness checked; NaN poisons the whole output | `-1` marks a skipped slot, and duplicates are allowed |

hf2q keeps its validation contract (§4.2). The schedule and the half-tile skip
port directly.

### 2.6 Levers considered and their dispositions

| Lever | Disposition | Reason |
|---|---|---|
| P1 slotted down | **adopt** | bitwise; existing API (DeepSeek-V4 uses it); −184.5 M threadgroups per run |
| P2 compact tile schedule | **adopt** | bitwise; −36.3 M more; the peer build's scheme |
| P2b schedule reuse for down | **adopt** | bitwise; −1 `map0` + barrier per layer |
| P2c half-tile skip | **adopt only if bitwise** (§5.1 N5) | −11% matrix rows (proxy); tensor `matmul2d` at M = 16 might reduce in a different order than at M = 32 |
| Change `MM_ID_ROUTING_THRESHOLD` | **reject** | no served forward is in the 5–32 row band; `mv_id` (F32) and `mm_id` (half B) differ numerically, so a change would be Class B and would move the joint-warm lane rule (`engine_gemma_warm_batch.rs:57-65, 135-141`) |
| Tensor path for Q5_0 down (16 layers) | **reject here** | mlx-native excluded it for exceeding the F32 parity bound (`ggml_capability.rs:572-577`); this is a numerics change that needs its own Class B design |
| Large (64 × 128) `mm_id` tile | **reject** | per-expert rows average 14–89 at the served sizes, so larger M tiles fill worse |
| Fuse GELU·up into the gate/up epilogue | **defer** | `[INFERENCE]` it would save about 10 GB of F32 traffic per run, about 0.025 s, at the cost of a new paired-N-tile kernel per type. Reopen only if M1 shows the activation pass above 5% of the MoE category |
| Fuse the weighted sum into down | **reject** | tiles group rows by expert, not by token, so the 8-way per-token sum would need atomics; not bitwise |
| Dense: pair gate/up, retile | **attribution only** | already 27 TFLOP/s and 93% full; M1 records sub-buckets. A dense op below half the V2 rate gets a follow-up note, not a change here |
| Larger or more frequent joint cohorts | **out of scope** | four lanes per forward already; splits cost about 0.1 s in 2 of 12 runs; cohort formation is scheduling (needs every slot idle) |

## 3. Exactness contract (Q1 classes, ADR-063 numeric policy)

### 3.1 Row-independence invariant (normative)

Joint warm prefill is declared exact against per-slot prefill (Q1 §3.1 item 4;
`engine_gemma_warm_batch.rs:20-22`). The joint cold design relies on the same
property. Its basis is that, on the matrix-matrix routes, one row's output does
not depend on the other rows in the batch. Every change here MUST keep:

- **R1.** For `mm_id`, each output element is computed by the same per-tile
  instruction sequence as today: same expert slab, same N tile, same K loop,
  same MMA shape and order, same B staging precision. Which tile holds a row,
  which rows share the tile, the tile count and the launch order may change
  only in ways that today's kernel already shows are irrelevant per element.
- **R2.** Each expert's routed rows are listed in ascending token order and cut
  into 32-row tiles from the expert's first entry, exactly as today. This makes
  tile membership identical, so R1 does not even need the weaker "row position
  is irrelevant" property.
- **R3.** Routing between `mv_id` and `mm_id` depends only on `(n_tokens,
  top_k, K)` exactly as today. Joint and per-slot lanes ≥ 33 rows both take
  `mm_id`, and the lane rule `MIN_SUFFIX_TOKENS = 33` stands.
- **R4.** Integer decisions (expert IDs, their order, routing weights) and
  every non-MoE kernel are untouched.

### 3.2 Class per change

| Change | Class against today's route | Proof obligation |
|---|---|---|
| P1 slotted down | **A** | Address identity: slotted `i11 = s`, `i12 = t` reads `(8t+s)·K·4`, the same as flattened `i12 = 8t+s`; output offset `r0 + s·N + 8t·N` = `r0 + (8t+s)·N`; `map0_ne20_8` lists `8t+s` in token order, the same values and order as `map0_ne20_1` over the flattened ids. Gated bitwise (§5.1 N1, §5.3) |
| P2 compact schedule | **A** | R1 + R2 by construction; gated bitwise on every `mm_id` type, both paths, every entry point (§5.1 N2) |
| P2b schedule reuse | **A** | the reused schedule equals a fresh one (same ids, `n_tokens`, `top_k`, `n_experts`); misuse fails closed (§5.1 N4) |
| P2c half-tile skip | **A or not shipped** | simdgroup path: skipping simdgroups 2–3's MMA when `nr1 ≤ 16` is bitwise by construction, because they own rows 16–31. Tensor path: full tiles (`nr1 > 16`) keep today's 32-row descriptor, so they are bitwise by construction; only partial tiles use a 16-row descriptor, which §5.1 N5 must show bitwise against the 32-row body per type. If the tensor path is not bitwise, P2c ships for the simdgroup path only. No Class B route enters production through this story |

All changes are Class A, so Q1 §3.6 applies. The task-quality obligation
against stock is met by `q1_decision_policy.py transcripts` returning
`ExactTranscripts` over the A/B runs (§7.2), and any same-request,
different-output pair voids the claim. Gemma amplifies any perturbation through
routing (median top-8/9 gap 7.5e-4; Q1 §2.2), so a non-bitwise variant could not
be called neutral. That is why §3.2 refuses Class B here.

### 3.3 Never allowed

Changing either routing threshold, the tensor-API predicate or the storage
types; compensating with any tolerance; smoothing routing; comparing a subset of
an extent; retry-until-pass (Q1 §3.7).

## 4. Design (LLD)

### 4.1 P1 — hf2q: Gemma prefill MoE down on the slotted entry point

`forward_prefill_batched.rs` (the batched and live-resume prefill body). The
down call today (`:4959-4978`) passes `n_tokens = seq_len·top_k`, `top_k = 1`.

The new rule copies DeepSeek-V4's paired-prefill rule (`deepseek4/ffn_forward.rs:40-51`,
`use_paired_expert_prefill`), because `plan_expert_auto_route` is `pub(crate)`
in mlx-native. The down call takes the slotted route when all of these hold:
- `seq_len > mlx_native::MM_ID_ROUTING_THRESHOLD`;
- no `HF2Q_MM_ID_ROUTING_THRESHOLD` override is set (the DeepSeek check,
  `ffn_forward.rs:423`). An override can send gate/up to `mv_id`, where the
  slotted contract refuses, so with an override present the flattened call
  stays;
- the expert stack is not affine (below).

The call has these parameters:
- `n_tokens = seq_len`, `top_k = top_k`;
- the same `pf_moe_swiglu` input, `[seq_len, top_k, moe_int]` F32;
- the same `pf_expert_ids`, output `pf_moe_down` and scratch;
- `n = hs`, `k = moe_int`, `expert_stride = down_expert_stride`,
  `ggml_type = down_ggml_dtype`.

Otherwise the existing flattened call stays. At 5–32 rows the flattened down
call is `mm_id` while the slotted contract refuses below the threshold, so
today's routing is kept in every case, as R3 requires.

**Entry point.** The flattened call goes through
`expert_dispatch::dispatch_expert_matmul_id_pooled`, which also serves scalar
(F32/F16/BF16) expert stacks via `dense_matmul_id`. P1 therefore adds
`expert_dispatch::dispatch_expert_matmul_id_pooled_slotted` with the same
storage split:
- **scalar stacks:** `dense_matmul_id` with `DenseMatmulIdInputLayout::Slotted`.
  The input is `[n_tokens, top_k, K]`, the output and IDs are unchanged, and
  `MayRepeat` / `Direct` are as today. The model-free test must show it is
  bitwise equal to today's flattened scalar call, and the type ships on the
  slotted route only if that test passes; otherwise scalar stacks keep the
  flattened call;
- **block-quantized stacks:** `GraphSession::quantized_matmul_id_ggml_pooled_slotted`.

Further rules:
- **Affine stacks.** The `gemma_moe_use_affine` branch (`quantized_matmul_id_into`)
  and the imatrix intercept keep their flattened form. They are not on the
  served GGUF route.
- **Scratch.** `IdMmScratch::alloc(dev, num_experts, seq_len·top_k_max)`
  (`:1519-1521`) already covers `n_experts × seq_len`.
- **Switch.** `HF2Q_GEMMA_PREFILL_MOE_DOWN = flat | slotted`, resolved once at
  model load in a small module beside `gemma_live_attention.rs`:
  - default `slotted` after §7's Class A gates pass; `flat` is the A/B arm;
  - an invalid value warns and keeps the default;
  - published in `/hf2q/v1/runtime` active controls;
  - it exists only for the same-binary A/B. It is deleted either by P3's
    promotion commit (§4.5) or, on the stop path (§7.4), by the PR that
    records the M3 result. Either way, slotted becomes the only route for
    the cases above.
- **Joint warm.** `engine_gemma_warm_batch::worker_enabled` does **not** refuse
  either value, because both are Class A. The model-free tests assert this.
- **Q5 case.** `q5_cases/gemma4.json` and `scripts/serve_gemma4_opencode.sh`
  gain the pass-through `HF2Q_Q5_GEMMA_PREFILL_MOE_DOWN` → `HF2Q_GEMMA_PREFILL_MOE_DOWN`
  (the `JOINT_WARM_PREFILL` pattern).

P1 needs no mlx-native release. It can land and be measured on the 0.15.16 pin
while P2 is in review.

### 4.2 P2 — mlx-native: compact `mm_id` tile schedule (contract text for the linked mlx-native issue)

Scope: the `quantized_matmul_id_ggml` family (auto, pooled, pooled pair, pooled
slotted, forced prepared-schedule test helpers) for every type with an `mm_id`
kernel (Q4_0, Q5_0, Q8_0, Q2_K–Q6_K, Q5_1, IQ4_NL, IQ4_XS) on both the tensor
and simdgroup paths. Out of scope: `fused_gate_up_silu_mm_id_q6_K` and the
`moe_mm_id_q4_0` family. They keep their grid schedule and their own `map0`.

**Schedule selection.** `GgmlRoutingPolicy` gains
`expert_mm_schedule: ExpertMmSchedule { UniformGrid, CompactTiles }`:
- default `CompactTiles` once §5.1 passes on both metallib paths;
- `HF2Q_MM_ID_SCHEDULE=grid` selects `UniformGrid`, a diagnostic and A/B arm
  parsed in `expert_routing_policy_from_environment` like
  `HF2Q_DISABLE_TENSOR_MM_ID`;
- adding the field changes the serialized routing policy, so
  `GgmlCapabilityRequest::schema_version` increments and the capability
  receipts and registry-adoption record are regenerated (as each prior pin bump
  did);
- the capability's `GgmlScratchRequirement::ExpertMm` reports the bytes of the
  selected schedule.

**`map0` v2** (`kernel_mul_mm_id_map0c_ne20_{1,6,8}`). One threadgroup of
`n_experts` threads, as today:
1. Stage `ntg` tokens' ids. The **staging thread** checks its own token: every
   id `< n_experts` and the `top_k` ids pairwise distinct. It ORs a
   threadgroup-shared `invalid` flag. This is the same predicate as today,
   evaluated once per token instead of once per token per thread.
2. Pass 1: each thread counts its expert's entries.
3. Thread 0 takes an exclusive prefix sum over the counts (each expert's slice
   padded by one entry). For each expert it emits one descriptor
   `(expert, first_entry, valid_rows ≤ 32)` per 32-row tile of that expert's
   entries in order, and writes `tile_count` to word 0.
4. Pass 2: each thread writes its expert's entries `t·top_k + slot` in
   ascending `t` into its slice. That is R2's order.
5. If `invalid`, word 0 = `POISON (0x80000000)`.

**`mm_id` v2** (same kernel names plus a `_c` suffix, one per type and path):
- Grid: `max_tiles` tile slots × `⌈N/64⌉` N tiles, with
  `max_tiles = min(rows, ⌈rows/32⌉ + min(n_experts, rows) − 1)` and
  `rows = n_tokens·top_k`. Word 0 holds `tile_count`.
- **Grid order.** Two orders are bitwise-neutral, because each output element
  has one writer:
  - (a) tile fastest, `(max_tiles, ⌈N/64⌉, 1)`, which is the peer's order;
  - (b) N tile fastest, `(⌈N/64⌉, max_tiles, 1)`.

  (a) sweeps every expert's tiles for one N block, so the activation re-read
  distance is the whole input (about 16 MB for gate/up and 32 MB for down at
  n = 1,420). (b) keeps one expert's rows resident across its N blocks, as
  today's grid does. M0c measures both, and the faster one at the served
  shapes becomes the only order shipped. The descriptor format does not
  depend on the order.
- Word 0 has POISON: threadgroup (0, 0) writes NaN over the whole logical output
  and every threadgroup returns before forming a weight or activation address.
  This is today's contract (`quantized_matmul_id_ggml.rs:289-296`).
- `tile ≥ tile_count`: return.
- Otherwise load `(im, idb, nr1)` and run **today's tile body unchanged**, with
  `ids_i32[im·ne21 + r1 + j]` replaced by `ids_i32[idb + j]`.

**Scratch** (compatible with every existing caller):
- **Constructor.** `IdMmScratch::alloc(device, n_experts, max_n_tokens)` keeps
  its signature and meaning: the largest `n_tokens` per call. A second
  constructor, `IdMmScratch::alloc_with_top_k(device, n_experts, max_n_tokens,
  max_top_k)`, sizes the compact areas for `rows = max_n_tokens·max_top_k`.
  `alloc` stays safe without a top-k bound: it assumes `max_top_k = 8`, the
  largest `map0` instantiation, and may over-allocate.
- **Words.** `htpe` holds `max(n_experts, 1 + 3·max_tiles(rows))` u32. The
  legacy-`map0` consumers (Qwen's fused gate/up-SiLU path, `gpu_ffn.rs:2896`)
  still find at least `n_experts` words.
- **List.** `hids` holds `max(n_experts × max_n_tokens, rows + n_experts)` i32
  while `UniformGrid` exists.
- **DeepSeek-V4** allocates with token rows and calls with `top_k = 6`
  (`verifier_forward.rs:646, 919`). Under `alloc` it is covered by the
  `max_top_k = 8` assumption; it may move to `alloc_with_top_k(…, 6)`.
- **Encapsulation.** The `htpe`/`hids` fields become private. Read access is
  through `capacity_n_experts()`, `capacity_n_tokens()`, `capacity_rows()`
  and buffer accessors, so callers stop deriving capacity from element counts.
  Qwen's pool check (`qwen35/decode_pool.rs:239-247`), which divides
  `hids.element_count()` by `htpe.element_count()` and would otherwise
  reallocate on every call, moves to these accessors.
- **Key invalidation.** A buffer access through the mutable accessors
  invalidates the P2b schedule key. **Every kernel that writes the scratch
  must take it through the invalidating mutable accessor**, including GPU
  writers that only borrow the buffers today. Examples are the legacy `map0`
  inside Qwen's fused gate/up-SiLU call (`gpu_ffn.rs:2896`) and any test
  helper. A read accessor never hands out a buffer that a kernel writes.
- **Capacity check.** `check_capacity(n_experts, n_tokens, top_k)` checks the
  experts, the tokens and `n_tokens·top_k ≤ capacity_rows`.

**Digests and receipts that change** (to be coordinated with any other native
release in the same window, decision D7):
- `GgmlCapabilityRequest::schema_version` and the capability receipts;
- the serialized `GgmlRoutingPolicy`, which feeds Qwen's
  `routing_policy_sha256` (`qwen35/execution_config.rs:105`) and the
  tensor-execution provenance manifests (`core/provenance/tensor_execution`);
- the native registry-adoption record and the metallib SHA-256.

Regenerate them once per pin.

**Capture mode.** `set_pending_buffer_ranges` declares `ids` as read and the
scratch as written for `map0`. For `mm_id` it declares `weight, input, scratch`
as read and `output` as written, as today.

### 4.3 P2b — schedule reuse for the down call

A new entry point, `quantized_matmul_id_ggml_pooled_slotted_reusing_schedule`
(`GraphSession` wrapper of the same name), encodes `mm_id` against the schedule
already in `scratch`. `IdMmScratch` records a schedule key at every `map0`
encode. The key is:
- the ids buffer's base address, offset and byte extent;
- `n_tokens`, `top_k`, `n_experts`, the schedule kind;
- the encoder identity, meaning the `CommandEncoder` **object**, not its
  current command buffer. A `commit_wait_and_rotate` keeps the encoder and the
  schedule bytes, so it does not invalidate the key. A new session or encoder
  does invalidate it;
- a per-scratch encode serial.

The new call validates that key against its own `(ids, n_tokens, top_k,
n_experts)`, the current encoder and "no `map0` since". On any mismatch it
returns `InvalidArgument` before encoding.

**hf2q integration.** hf2q does not let a valid mismatch become an error. The
Gemma body tracks whether it rotated its session between gate/up and down.
Rotation happens under `profile_buckets_on` or when an imatrix collector is
installed. The imatrix intercept runs `commit_wait_and_rotate` between gate/up
and down. That is safe under the encoder-object key, but the fallback covers it
anyway so collection can never hit a propagated `InvalidArgument`:
- an installed `ImatrixCollector` (`intercept_qmatmul_id_with_hint` active);
- `HF2Q_PROFILE_MOE`;
- `HF2Q_PROFILE_BUCKETS`;
- `HF2Q_PROFILE_GPU_TS`, via `bucket_finish!` and the `moe_dn` pre-finish
  (`forward_prefill_batched.rs:941, 4889`).

When the session was rotated, hf2q calls the plain slotted entry point, which
builds its own schedule and is bitwise equal. Only when the session was not
rotated does it call the reusing entry point, and an error there is a bug that
propagates. A test (§5.2) covers both branches.

The existing `map0` barrier orders schedule construction. The graph's
`barrier_between` before down already covers `pf_moe_swiglu` and `pf_expert_ids`
and is unchanged. This generalizes what `pooled_pair` does internally.

### 4.4 P2c — half-tile skip (conditional)

The `mm_id` v2 body skips work only for partial tiles with `nr1 ≤ 16`:
- **Simdgroup path.** Simdgroups 2–3 own rows 16–31 (`lsmb = sb + 2·64·(sgitg/2)`).
  When `nr1 ≤ 16` they skip their MMA and their `simdgroup_store` into shared
  memory. **The global copy loop (`j = sgitg; j < nr1; j += 4`) keeps all four
  simdgroups**; otherwise rows j ≡ 2, 3 (mod 4) would never be written. This is
  bitwise by construction.
- **Tensor path.** Full tiles (`nr1 > 16`) keep today's
  `matmul2d_descriptor(32, 64, 32)`, so they are unchanged. Partial tiles use a
  16-row descriptor over the lower half of the B tile. The peer instead runs two
  16-row products on every tile; that is not adopted, because it would change
  full tiles too. N5 must show the 16-row product bitwise against the 32-row
  body on partial tiles, per type.

P2c ships under `CompactTiles` only for the paths where N5 passes. A failed
path keeps the 32-row body. The receipt records which paths ship.

### 4.5 P3 — hf2q: pin, integrate, promote

1. Pin the mlx-native release that contains P2/P2b(/P2c), with the
   registry-adoption record, the Cargo.lock checksum and the regenerated
   digests listed in §4.2.
2. Gemma down (P1's slotted branch) calls the reusing entry point when the
   session was not rotated since gate/up (§4.3).
3. The runtime-controls surface reports `prefill_moe_down` and
   `mm_id_schedule`.
4. The Q5 case gains a second pass-through, `HF2Q_Q5_MM_ID_SCHEDULE` →
   `HF2Q_MM_ID_SCHEDULE` (case variable `MM_ID_SCHEDULE`, forwarded by
   `serve_gemma4_opencode.sh`). The empty default leaves the native default.
5. After §7 passes, the promotion commit:
   - deletes the `flat` arm and `HF2Q_GEMMA_PREFILL_MOE_DOWN`;
   - keeps `HF2Q_MM_ID_SCHEDULE=grid` and its pass-through as the native
     diagnostic until the next native release removes `UniformGrid`
     (decision D3);
   - records the receipts in this document and in the ADR story.

## 5. Tests

### 5.1 mlx-native, model-free (GPU; both `MLX_PRECOMPILED_METALLIB=1` and `=0`)

New `tests/test_mm_id_compact_schedule.rs`, extending `test_quantized_matmul_id_mm.rs`:

| ID | Check | Matrix |
|---|---|---|
| N1 | slotted == flattened, bitwise (output bytes) | every `mm_id` type × {tensor, simdgroup} × `n_tokens` ∈ {33, 40, 64, 232, 355, 528, 1420, 4101} × routing {uniform, witness-shaped hot expert, all rows to one expert, 40 experts empty} × Gemma down shape (704→2816, E128, k8) and a Qwen-like shape (512→2048, E256, k8) |
| N2 | `CompactTiles` == `UniformGrid`, bitwise | N1's matrix × entry points {auto, pooled shared, pooled pair, pooled slotted} × `top_k` ∈ {1, 6, 8} × Gemma gate/up (2816→1408), down, DeepSeek-V4 shape (k6) |
| N3 | invalid metadata: id ≥ n_experts, a duplicate within a token, both | whole output NaN on both schedules; no weight read (a guarded weight buffer one byte short of a valid id's slab is never touched); `htpe`/list writes stay inside capacity |
| N4 | reuse key | a reuse after a different ids buffer, a different `n_tokens`/`top_k`, another `map0`, or another encoder returns `Err` before encoding; a valid reuse equals a fresh schedule bitwise |
| N5 | half-tile skip vs the 32-row body, bitwise | N2's matrix restricted to tiles with `nr1` ∈ {1, 15, 16, 17, 31, 32} by construction; per type and path, pass/fail recorded. Full tiles (`nr1` ≥ 17) must be bitwise on both paths by construction, and any difference there is a defect |
| N6 | capacity | `max_tiles` bound attained (one row per expert plus full tiles); scratch too small → `Err`; `UniformGrid`-sized scratch fits `CompactTiles` |
| N7 | row independence | a row's output is bitwise equal whether it is computed alone (n = 33 padding rows with fixed ids) or within any of N1's batches. This is the primitive form of joint == per-slot |
| N8 | existing suites unchanged | `adr_022_phase2_q5_k_mm_id_parity`, `adr_033_pi_iq4_xs_mm_id_parity`, `moe_pooled_pair_contract`, `test_quantized_matmul_id_mm`, `deepseek_q2_k::q2_k_slotted_down_mm_is_byte_identical_to_flattened_rows`, the fused gate/up-SiLU Q6_K parity (legacy `map0` on the new scratch), `ggml_capability` receipts regenerated with the new schema version |
| N9 | scratch compatibility | `alloc(n_experts, max_n_tokens)` and `alloc_with_top_k` serve every entry point at `top_k` ∈ {1, 6, 8}; `htpe` ≥ `n_experts` words; the capacity accessors report the constructor's bounds; a mutable buffer access invalidates the P2b key |

### 5.2 hf2q, model-free

- `gemma_prefill_moe_down_switch`: parse table (`flat`/`slotted`/invalid →
  default); runtime-controls publication; `worker_enabled` stays active under
  both values.
- Route selection, with a recorded-dispatch fake that captures the params:
  - `seq_len ∈ {5, 32}` keeps the flattened call;
  - `seq_len ∈ {33, 1420}` selects slotted;
  - with `HF2Q_MM_ID_ROUTING_THRESHOLD` set, every `seq_len` keeps the
    flattened call;
  - affine stacks keep the flattened call.
- Scalar expert stacks (F32, F16, BF16): `dispatch_expert_matmul_id_pooled_slotted`
  is bitwise equal to the flattened scalar call at `seq_len` ∈ {33, 232}. The
  type is routed slotted only if this passes.
- P2b fallback: with `profile_buckets_on` forced (session rotated between
  gate/up and down), down takes the self-scheduling slotted call, and without
  rotation it takes the reusing call. A forced key mismatch without rotation
  propagates the error.

### 5.3 hf2q, real artifact (Class A; one process each)

Invocation rules (frozen):
- Every test runs alone with `--exact --nocapture --test-threads=1`. The
  `#[ignore]` tests add `--ignored`; the env-gated `#[test]`s (items 3, 4 and
  §5.4's Qwen test) must **not** get `--ignored`, which would filter them out.
- Env-gated tests need `HF2Q_BYTE_EQUIV_E2E=1` and `HF2Q_BYTE_EQUIV_E2E_GGUF`
  (plus `HF2Q_QWEN35_E2E_GGUF` for Qwen). Without them they print a skip line
  and pass, so that counts as not run.
- The `#[ignore]` Gemma tests need `HF2Q_GEMMA4_GGUF`.
- A receipt counts only if libtest prints `1 passed` with 0 filtered-out
  ignored for the named test, and the output has no skip line.

Run each test with the switch at its default and again under the A arm
(`HF2Q_GEMMA_PREFILL_MOE_DOWN=flat`, plus `HF2Q_MM_ID_SCHEDULE=grid` after P2):
1. `gemma_joint_warm_prefill_matches_per_slot_bitwise` and `…_vec_hybrid`
   (`engine_gemma_warm_batch_tests.rs`, `#[ignore]`). The precise `TiledF32`
   route was falsified and removed (#668, 2026-11-05), so no third arm is
   gated.
2. `gemma_joint_warm_engine_cohort_matches_serial_tool_turns`,
   `gemma_joint_warm_forward_fault_retries_each_lane_per_slot` (`#[ignore]`).
3. `slot_aware_n4_per_slot_parity_vs_serial`,
   `slot_aware_n4_batched_body_same_prompt_parity_vs_serial`,
   `iter_g_a_multi_seq_prefill_first_token_isolation` (`engine.rs`,
   env-gated).
4. `gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant`
   (`engine_gemma4_tiny_prefill_tests.rs`, env-gated).
5. **New: `gemma_prefill_moe_route_matches_previous_route_bitwise`**
   (`#[ignore]`). On the official GGUF, with one process and both arms in
   sequence, it runs a cold prefill (458 rows) and a four-lane joint warm
   forward (the turn-1 shapes, 4 × 58 rows over 458-row histories) and
   compares, A against B:
   - every layer's `pf_moe_down` and `pf_moe_accum` bytes;
   - the logits rows, KV (K bytes, V codes and scales over valid rows) and the
     next decode token.

   Any byte fails.
6. **Q1 endpoint Class A.** `HF2Q_Q1_GEMMA_ENDPOINT_PACKET` with greedy144
   (`842768`), `q1_gemma_actual_ordinary_endpoint_trace` T0 under arm A and arm
   B (two processes), then `q1_decision_policy.py exact`: `ExactEqual` required.

### 5.4 Shared-kernel regression gates (run at the P2 pin bump)

The compact schedule changes every hf2q `mm_id` consumer. Because it is Class
A, these gates must pass unchanged:
- `official_artifact_cooperative_warm_prefill_is_exact_and_faster` and
  `official_artifact_b4_decode_body_is_exact_and_measured` (DeepSeek-V4);
- `slot_aware_qwen35_n8_per_slot_parity_vs_serial` (Qwen MoE; env-gated, run
  under §5.3's invocation rules with `HF2Q_QWEN35_E2E_GGUF`). It also
  exercises Qwen's decode-pool scratch check (`decode_pool.rs:239-247`)
  against the new accessors;
- an A-vs-B `ExactTranscripts` smoke on Qwen3.6 and DeepSeek-V4 (one run per
  arm each) when the root's schedule allows. Otherwise, decision D6.

## 6. Microbenchmark plan (mlx-native, model-free; needs the GPU lane and a quiet host)

`benches/bench_mm_id_schedule.rs` (harness false), modeled on
`bench_mul_mm_id_qwen36_ffn.rs`.
- **Clock.** GPU time per dispatch from `MTLCommandBuffer` `GPUStartTime`/`GPUEndTime`.
  One command buffer per measured op, 20 warm-up iterations and 200 measured,
  median and p90.
- **Receipt fields.** mlx-native commit, metallib SHA-256, device name, thermal
  state, macOS build.
- **M0a, empty-threadgroup cost** (today's kernels, before P2 is written).
  `dispatch_id_mm_with_prepared_schedule_for_test` encodes `mm_id` against a
  **synthetic schedule** written into the scratch from the host:
  - `htpe[0] = 32`, all other experts 0, and `hids[0..32]` = rows 0..31;
  - Q8_0 down shape (704→2816, E128);
  - `n_tokens` swept over {33, 256, 1024, 4096, 11360} with `top_k = 1`.

  Busy work stays exactly 1 tile × 44 N tiles, and only the empty grid grows
  (128 × ⌈n/32⌉ × 44). The slope of GPU time against launched threadgroups is
  c_e. A second sweep puts the busy expert at 127 to check that position does
  not matter. (Real routing cannot do this: with every id on expert 0, busy
  work grows with the grid.)
- **M0b, served shapes, today's kernels.** For n ∈ {232, 528, 764, 880, 1172,
  1412, 1420} it measures:
  - gate/up (Q4_K, 2816→1408, k8), on its single shared-input route;
  - down (Q8_0 and Q5_0, 704→2816), flattened and slotted. Both forms have
    identical busy tiles, so Δt / ΔTG gives a second, served-shape estimate of
    c_e;
  - `map0_ne20_8` vs `map0_ne20_1` alone;
  - the activation pass and the fused weighted-sum pass at the same n.

  Routing tables come from:
  - uniform random;
  - the witness bootstrap of §2.4;
  - a **recorded agent table**: the expert-id buffers of one real four-agent
    run's warm forwards. M1 captures them read-only with the existing Q1
    routing-witness hook on the canonical prompts. If that hook cannot serve the
    serving path, the witness bootstrap stands in and this is recorded.
- **M0c, after P2.** The same shapes and tables for `UniformGrid` against
  `CompactTiles`, for both grid orders of §4.2, ± P2b reuse and ± P2c.
- **Output.** Per shape: the launched and busy threadgroups, ms and Δ ms.

**Prediction formulas (frozen).** The sum Σ runs over the six workload forward
sizes n ∈ {232, 764, 880, 1180, 528, 1420} with 30 layers per forward. The down
layers are weighted 16 : 14 for Q5_0 : Q8_0, and the recorded agent table is
used if present (otherwise the witness bootstrap).
- `S_pred(P1) = Σ [t_down,flat(n) − t_down,slotted(n)]`, measured in M0b.
- `S_pred(P2) = c_e · ΔTG_P2 + Δmap0`:
  - `ΔTG_P2` is the launched-threadgroup reduction from slotted/grid to
    compact, 36.3 M per run (§2.4), recomputed from the recorded table;
  - c_e is M0a's slope, cross-checked against M0b's; if they disagree by more
    than 2×, the larger is used for the stop rule and both are recorded;
  - `Δmap0` is M0b's per-call `map0_ne20_8` time × 180 calls per run (the
    `map0` removed by P2b).
- **P2c is excluded from the stop rule.** M0 cannot predict matrix savings.
  Its bound, for disclosure only, is ≤ 0.05 s (the §9.1 upper estimate).
- After P2 exists, M0c's measured per-shape deltas replace both predictions:
  `S_meas = Σ [t_grid(n) − t_compact(n)]`, summed over gate/up and down plus
  the `map0` delta.

**Decision rule (frozen here):**
- If `S_pred(P1) < 0.03 s`, P1 is still implemented, because it is cheap and
  bitwise, but it gets no speed claim.
- If `S_pred(P1) + S_pred(P2) < 0.05 s`, the story takes the **stop path**
  (§7.4): P2, P2b and P2c are not built; P1 merges; the switch
  `HF2Q_GEMMA_PREFILL_MOE_DOWN` is deleted in the PR that records the M3
  result; the falsified lever is recorded.
- At no measured shape may `CompactTiles` be slower than `UniformGrid` by more
  than max(2%, 5 µs).

## 7. Acceptance gates (frozen before any data)

### 7.1 Speed, against the measured shares

Baselines are from `$G` at hf2q `7f295b38` / native 0.15.16 (§2.1). Each gate is
re-baselined on the A arm of the same binary at the candidate commit.

| Gate | Metric | Pass |
|---|---|---|
| S1 (primitive) | M0c per-shape GPU time | no regression beyond max(2%, 5 µs); `S_meas ≥ 0.05 s` per run |
| S2 (warm prefill, primary) | per-run sum of unprofiled joint-warm forward walls (`Batched prefill complete` lines), over **structure-matched** runs (rule below) | paired-block median Δ (B − A) ≤ −0.05 s, 95% bootstrap CI upper bound < 0, over ≥ 6 structure-matched ABBA pairs. Baseline A (unsplit runs) 3.19–3.21 s |
| S3 (category) | catsplit MoE + layer tail GPU s per run (`HF2Q_GEMMA_PREFILL_CATSPLIT=1`, 3 runs per arm). Prerequisite met: PR #687 is merged on main (`2dfe71ba`), so the split is available to every candidate built from main | median reduction ≥ 50% of `S_meas` (or of `S_pred(P1)` for M3); pre-attention, attention and dense change by less than ±3%. Baseline 1.09–1.11 s |
| S4 (makespan) | Q5 four-agent client makespan, same runs as S2 | no regression: median Δ ≤ 0 and 95% CI upper bound ≤ +0.05 s; Δ reported |
| S5 (n1) | one-agent control, 2 runs per arm | disclosed, not gated (ADR-063 Q4) |

**S2 transaction-structure rule (frozen).**
- A run's *structure* is the ordered list of its warm transactions, each with
  its lane count and per-lane advanced rows.
- The arms are bitwise, so their prompts are identical. Cohort splits, however,
  depend on request arrival timing, not on outputs. In 2 of 12 profile runs one
  cohort split (L1 `004-b02`, L3 `030-b05`), and one split adds about 40 ms.
- S2 pairs an A run with a B run only if their structures are equal.
- Runs whose structure differs from the modal structure of the campaign are
  excluded from S2, reported with their transaction lists, and replaced by
  further ABBA pairs until 6 structure-matched pairs exist. Replacement is
  capped at 6 extra pairs.
- If 6 pairs cannot be reached, S2 is instead computed **per transaction**:
  each modal-structure transaction is paired by turn and row count across arms,
  the per-run sum is formed over the matched transactions only, and the
  substitution is recorded.
- S4 (makespan) uses every run, split or not.

The story's "≤ 0.3 s" is the profile's bound on this whole lever, not a gate.
A result between 0.05 s and S_meas passes. A result below 0.05 s is §7.4.

### 7.2 Quality

- Class A proofs: §5.3 items 1–6 and §5.1 N1, N2, N4, N7 (and N5 for shipped
  P2c paths).
- `q1_decision_policy.py transcripts` over every S2/S4 run pair:
  `ExactTranscripts`, 0 same-request/different-output groups. That meets the
  quality obligation against stock (Q1 §3.6). This story runs no paired
  quality campaign.
- **Against v0.1.28 and against the peer:** inherited by the combined
  candidate in Q4-GEMMA-ATTN's closing campaign (tiled-attn design §9.1 and
  §9.3). This story adds no separate quality gate beyond its Class A identity.

### 7.3 Fairness and memory

- Fairness, same binary B against A: worst-slot TTFT p95 and SSE-gap max
  ratios ≤ 1.025 (`NEUTRAL_RATIO`). Against v0.1.28 stock: measured on the
  combined candidate (ADR-063 Q4).
- Memory: every run passes Q5 `CalibrationPass` with the accepted
  negative-control receipt. While `UniformGrid` exists, the `hids` list is
  unchanged. `htpe` grows from `n_experts` to `max(n_experts, 1 + 3·max_tiles)`
  words, about 5.8 KB at 11,360 rows (arithmetic). Removing `UniformGrid`
  later reduces the `mm_id` list for a 4,096-row forward from 16.8 MB to about
  131 KB. The closing candidate gets a fresh Q5 calibration in any case.

### 7.4 Falsification outcomes

- **Any Class A gate fails** (one differing byte): the change is a defect, not
  a near-tie. Localize it with the §5.1 matrix. The change does not ship.
- **S1 fails on a shape:** P2 does not become the default for that type; that
  type keeps `UniformGrid`, and this is recorded per type.
- **Stop path: `S_pred(P1) + S_pred(P2) < 0.05 s` (§6), or S2 fails.**
  - Record "launch geometry is not a material Gemma cost on this host", with
    the M0 receipts.
  - P1 still merges, because it is bitwise and costs nothing. The PR that
    records the M3 result deletes `HF2Q_GEMMA_PREFILL_MOE_DOWN` and its Q5
    pass-through, which makes slotted the only route (§4.1).
  - P2/P2b/P2c are not built, or are reverted before release; no native
    release or pin is needed.
  - The root updates the Gemma gap plan in the profile §7.
  - The story closes through this recorded falsification (§11 delivery
    boundary).

## 8. GPU validation plan (root only; GPU lane; one model process at a time)

Every run records the source commit, binary SHA-256, artifact digest
(`bf65b7e1…`), native version and metallib SHA-256, and the case file and its
SHA-256. It passes Q5 `CalibrationPass`. Evidence goes under
`/Users/robert/.local/state/hf2q/v0130/q4-gemma-prefill-moe/`. Check `df -h /`
(≥ 20 GB) and free memory first. Hold HOST-LOCK for each timed sequence.

| Step | What | Model | Est. time |
|---|---|---|---|
| M0a/M0b | §6 on an mlx-native worktree at 0.15.16 (today's kernels); M0a uses the synthetic schedule | none | ~15 min |
| M1 | one four-agent run at hf2q main (≥ `2dfe71ba`) with `HF2Q_PROFILE_GPU_TS=1` for the per-op split (`moe_gu`, `moe_swiglu`, `moe_dn`, `moe_wsum_add`, `gur`, dense down, O-proj), with the distortion against an unprofiled run stated; plus the read-only capture of the warm forwards' expert-id tables for M0b | Gemma | ~10 min |
| M2 | P1 Class A gates §5.3 (1–6), under §5.3's invocation rules | Gemma, 8–10 processes | ~60 min |
| M3 | P1 A/B: `HF2Q_Q5_GEMMA_PREFILL_MOE_DOWN ∈ {flat, slotted}`, ABBA ≥ 6 structure-matched pairs (§7.1 rule), then catsplit 3 + 3 | Gemma | ~45 min |
| M4 | P2 native qualification: §5.1 on both metallib paths; M0c (both grid orders) | none | ~30 min |
| M5 | P3 pin: §5.3 again (arm A = flat + grid) and §5.4; A/B (A = `HF2Q_Q5_GEMMA_PREFILL_MOE_DOWN=flat` + `HF2Q_Q5_MM_ID_SCHEDULE=grid`, B = defaults), ABBA ≥ 6 structure-matched pairs; catsplit 3 + 3; n1 2 + 2 | Gemma, DeepSeek-V4, Qwen | ~2.5 h |
| M6 | combined candidate (Q4-GEMMA-ATTN §9.3); not this story's gate | — | — |

A/B loop (M3, M5), following the tiled design §9.2:

```sh
for ARM in A B B A A B B A A B B A; do
  set_arm_env "$ARM"   # M3: HF2Q_Q5_GEMMA_PREFILL_MOE_DOWN=flat|slotted
                       # M5: adds HF2Q_Q5_MM_ID_SCHEDULE=grid on A
  python3 scripts/four_agent/q5_calibration.py scripts/four_agent/q5_cases/gemma4.json \
    --negative-control-receipt "$NC" \
    --output ~/.local/state/hf2q/v0130/q4-gemma-prefill-moe/ab-$ARM-$(date -u +%Y%m%dT%H%M%SZ)
done
python3 scripts/four_agent/q1_decision_policy.py transcripts <A runs> <B runs>   # ExactTranscripts
```

From each run, read:
- the per-forward `Batched prefill complete` lines and the transaction
  structure (S2);
- the `joint warm` transaction `elapsed_ms`;
- the client makespan (S4);
- TTFT p95 and SSE-gap max (§7.3);
- the runtime-controls record, which confirms the arm.

## 9. Expected savings and relation to the profile

### 9.1 Estimate (makespan s per run, `[INFERENCE]`)

| Change | Mechanism | Estimate |
|---|---|---:|
| P1 slotted down | −184.5 M launched threadgroups | 0.05–0.18 (c_e 0.25–1 ns) |
| P2 compact schedule | −36.3 M more (40.3 M → 4.0 M) | 0.01–0.04 |
| P2b reuse | −180 `map0` + barrier per run | 0.002–0.01 |
| P2c half-tile skip | −11% matrix rows (proxy), if MMA + B staging is 25–50% of `mm_id` time; not part of the stop rule | 0.02–0.05 |
| **Total** | holds only if c_e ≥ 0.25 ns; expert-weight streaming (about 0.15–0.30 s, §2.4) is untouched | **0.08–0.28** |

### 9.2 Stage ownership and cross-story coupling

| Profiled stage (#660) | Owner |
|---|---|
| Warm-prefill attention, 48% | #668: TiledF32 falsified 2026-11-05; the bit-exact successor story (Q4-GEMMA-ATTN-EXACT) |
| Warm-prefill MoE + tail, 34% | **this story** |
| Warm-prefill dense 11%, pre-attention 7.5% | this story's attribution (M1); no change |
| Cold-turn admission | Q4-GEMMA cold design. Its joint cold forward (about 4 × 465 rows) takes the same `mm_id` route and gains from P1/P2 by the same mechanism |
| Decode MoE/dense (`mv_id`, width 4) | Q4-GEMMA decode design; unchanged here |
| Speculation verify rows | Q4-GEMMA spec design. Verify forwards of ≤ 32 rows keep `mv_id` for gate/up; the flattened down call at 5–32 rows is untouched by P1, and P2 changes only its launch grid, bitwise |

After #668 removes about 1.2–1.4 s of attention, MoE becomes the largest
warm-prefill class (about 1.1 of about 1.8 s GPU per run `[INFERENCE]`). The two
savings act on disjoint kernels and add.

## 10. Risks

| Risk | Mitigation |
|---|---|
| c_e is tiny, so the lever is small | M0a measures it first with a synthetic schedule, on today's kernels, before P2 work, and M0b cross-checks it; the §6 decision rule takes the stop path |
| The routing proxy misstates real tile fill | M0b uses recorded agent routing tables (M1); S3/S2 measure the real effect |
| `matmul2d` at M = 16 reduces in a different order from M = 32 | full tiles keep the 32-row descriptor; partial tiles are bitwise-gated per type (N5); otherwise the tensor path drops P2c |
| Compact `map0` bug: out-of-bounds list write, wrong descriptor at the tile boundary, poison regression | N3, N6, N7; guard-sized buffers; the descriptor bound is asserted against `max_tiles` |
| Shared kernel regresses another family | §5.4 gates at the pin bump; the schedule is per policy, so `grid` stays available until the gates pass |
| Scratch layout breaks other callers (Qwen decode-pool capacity check, Qwen fused legacy `map0`, DeepSeek-V4 `top_k = 6` sizing, direct field writes) | `alloc` signature kept with a `top_k ≤ 8` assumption; `htpe ≥ n_experts` words; capacity accessors and private fields; N8/N9 and §5.4 (§4.2) |
| Compact grid order loses activation locality | both bitwise-neutral orders measured in M0c; the faster ships (§4.2) |
| Schedule reuse reads a stale schedule, or a valid profile run aborts | keyed reuse fails closed (N4); hf2q uses the self-scheduling slotted call whenever its session was rotated (§4.3, §5.2) |
| Thread 0's prefix sum serializes for `n_experts` = 256 | O(n_experts + tiles) per call; included in M0c timings |
| Capability/receipt schema and routing-policy digests change | regenerate the digests listed in §4.2 once per pin; coordinate with any concurrent native release (D7) |
| Measurement distortion | primary gate unprofiled (S2); catsplit only for the category share (+13% known distortion) |
| Cohort splits (2 of 12 runs; about 40 ms each) add noise to S2 | splits depend on timing, not outputs; the frozen structure rule (§7.1) pairs only structure-matched runs, or falls back to per-transaction pairing; splits are reported |
| Gates that silently skip | §5.3 invocation rules: env gates set, `--ignored` only for `#[ignore]` tests, receipt must show `1 passed` with no skip line |

## 11. ADR-063 story section (extraction-ready)

### Q4-GEMMA-PREFILL-MOE. Reduce the Gemma 4 warm-prefill MoE cost per row

Outcome: Gemma 4 26B-A4B warm prefill spends measurably less GPU time per row in the routed experts, bitwise identical to the previous route, and the saving enters the combined Gemma candidate.

Requirement: The program MUST reduce the GPU time per row of the warm-prefill routed-expert stage (34% of warm-prefill GPU time in the Q4-GEMMA profile) without changing any output byte. It MUST first test, with a model-free microbenchmark, the hypothesis that expert matrix-multiply launch geometry is a material part of that stage. It MUST record the measured saving, or the falsification, against the profiled MoE share.

Acceptance:

- Starting point (Q4-GEMMA #660, hf2q `7f295b38`, native 0.15.16): warm-prefill MoE and layer tail are 34% of warm-prefill GPU time (1.09–1.11 s per run serialized; attention 48%, dense 11%, pre-attention 7.5%). About 225 M expert-matmul threadgroups are launched per four-agent run, of which an estimated 3.1 M have work (routing proxy; design §2.4).

- A written design, approved before implementation: `docs/research/adr063-q4-gemma-prefill-moe-design.md`. It covers the Gemma prefill MoE down call on the slotted expert entry point (P1, hf2q); a compact per-expert tile schedule with schedule reuse (P2, P2b, mlx-native); and a half-tile skip only where it is bitwise (P2c). The `mm_id`/`mv_id` threshold (32), dense thresholds, tensor-API selection and storage types are unchanged.

- Every shipped change is Q1 Class A against the previous route. The mlx-native parity matrix (design §5.1) is bitwise on both metallib paths, and so are the hf2q real-artifact gates (joint warm == per-slot on every warm attention route, engine cohort == serial tool turns, per-slot parity, the route A/B layer-byte test, and the greedy144 Q1 endpoint `ExactEqual`). `ExactTranscripts` holds over every A/B run pair. A non-bitwise variant does not ship under this story.

- Speed, with one binary and the previous route as the A arm: at least 6 ABBA pairs matched by warm-transaction structure (design §7.1). The per-run sum of unprofiled joint-warm forward walls improves by at least 0.05 s (95% CI upper bound < 0). The profiled MoE category falls by at least half the measured microbenchmark saving, with the other categories within ±3%. The four-agent makespan does not regress (CI upper bound ≤ +0.05 s). The worst-slot TTFT p95 and SSE-gap max ratios stay within 1.025.

- Stop path (design §6, §7.4): if the frozen microbenchmark prediction for P1 + P2 + P2b is below 0.05 s per run, or the warm-prefill gate fails, P1 still merges as a bitwise change with its switch removed, P2/P2b/P2c are not built, and the story closes with the recorded falsification.

- The real-artifact gates run under the design's invocation rules (`--exact`, env gates set, `1 passed` with no skip line). The shared-kernel regression gates for DeepSeek-V4 and Qwen MoE pass at the native pin. Every run passes Q5 `CalibrationPass`.

Repository ownership: mlx-native owns the compact schedule, the schedule-reuse entry point, the half-tile skip, the parity tests, the microbenchmark and the release. hf2q owns the slotted down call, route switches, integration, the real-artifact gates and the A/B campaign, consuming mlx-native at an exact published pin.

Validation: Independent review of the design, the parity and microbenchmark receipts, the Class A real-artifact receipts and the A/B campaign from raw receipts.

Predecessors: Q4-GEMMA (#660, profile result PR #687, merged on main as `2dfe71ba`), Q5 (#371). Consumed by Q4-GEMMA-ATTN's combined candidate, which adds this story to its predecessors.

Priority: High. Core scope (second-tier family).

Delivery boundary: P1 merged to main; and either the mlx-native release containing P2 pinned and integrated by hf2q (P3), or, on the stop path, the recorded falsification. For v0.1.30, or the first published hf2q release containing the accepted Gemma 4 configuration.

## 12. Decisions for the operator

- **D1. Exactness scope.** Only Class A changes ship under this story, and a
  non-bitwise half-tile skip, threshold change or Q5_0 tensor path is out of
  scope. *Recommended: yes.* A Class B MoE route would need its own envelope
  and quality campaign, and Gemma's routing amplifies perturbations.
- **D2. One story or two.** One story delivered in three PRs: P1 (hf2q, on the
  current pin), P2 (mlx-native, with a linked mlx-native issue carrying the
  §4.2–§4.4 contract text), and P3 (hf2q pin and integration). This follows
  the Q4-QWEN38 pattern. The alternative is to split into a native KERNEL story
  and an hf2q story, as #659/#668 did. *Recommended: one story.* The native
  part is bitwise, and its contract is the issue text.
- **D3. Kill switches.**
  - `HF2Q_GEMMA_PREFILL_MOE_DOWN` is deleted at promotion (P3) or, on the stop
    path, in the PR that records M3.
  - `HF2Q_MM_ID_SCHEDULE=grid` stays as a native diagnostic for one release;
    the release after Q4 closes removes `UniformGrid`.
  - *Recommended as written.* The alternative is to remove `UniformGrid` in
    the same release and run the A/B across two pins, which is weaker evidence.
- **D4. Falsification threshold.** Take the stop path if `S_pred(P1) +
  S_pred(P2) < 0.05 s` per run (§6 formulas; P2c excluded). *Recommended:
  0.05 s,* about a sixth of the lever's profile bound. Without splits, the S2
  noise is about 5 ms; the main noise source, cohort splits, is handled by the
  structure rule.
- **D5. What the story's own acceptance requires.** A category saving with no
  makespan regression; the Win is measured only on the combined candidate.
  *Recommended: yes.* A 0.1–0.3 s saving cannot be resolved as a makespan Win on
  its own.
- **D6. Cross-family gates at the pin.** DeepSeek-V4 and Qwen MoE bitwise gates
  (with §5.3's invocation rules and §4.2's scratch obligations) plus one
  `ExactTranscripts` smoke each, against Gemma-only gates. *Recommended: run
  them,* since every `mm_id` consumer changes.
- **D7. Release coupling.** Ship P2 in the same mlx-native release as
  Q4-GEMMA-KERNEL (#659) or on its own. *Recommended: whichever is ready first,
  coordinated.* If both releases bump `GGML_CAPABILITY_SCHEMA_VERSION` or the
  routing-policy digest, sequence them so the receipts and digests of §4.2 are
  regenerated once per pin.

## 13. Source identities read

| Path | Identity |
|---|---|
| hf2q `src/serve/forward_prefill_batched.rs` @ `ff4af4bd` | sha256 `3f6541a4…e08e090`. Line numbers in this document refer to `ff4af4bd`. On main `2dfe71ba` (`f2072ab3…fe76`, +39 catsplit lines) they shift by +32 around the MoE body: gate/up 4836–4855, down 4991–5010, `moe_dn` pre-finish 4921, scratch 1525–1527; `profile_buckets_on` stays at 941 |
| hf2q `src/serve/api/engine_gemma_warm_batch.rs` @ `ff4af4bd` | `6ae9d390…b4fa53` |
| hf2q `src/inference/expert_dispatch.rs` @ `ff4af4bd` | `cd54a844…f9e040e227` |
| hf2q `src/serve/forward_mlx_shared.rs` @ `ff4af4bd` | `56ffe77c…aa59` |
| hf2q `docs/research/adr063-q1-numeric-policy.md` @ `ff4af4bd` | `16df680e…fe35` |
| hf2q `src/inference/models/deepseek4/ffn_forward.rs` @ `2dfe71ba` | `150f70c8…6989` (route rule, §4.1) |
| hf2q `src/inference/models/qwen35/decode_pool.rs` @ `2dfe71ba` | `7ed8a9be…e6f3` (scratch capacity check, §4.2) |
| hf2q `src/inference/models/qwen35/gpu_ffn.rs` @ `2dfe71ba` | `f27745a6…80dc` (fused legacy `map0` on the scratch) |
| hf2q `src/inference/models/qwen35/execution_config.rs` @ `2dfe71ba` | `e776ac97…79a8` (`routing_policy_sha256`) |
| hf2q `src/serve/api/engine_gemma4_tiny_prefill_tests.rs` @ `2dfe71ba` | `37133bf7…2715` (env-gated test, §5.3) |
| mlx-native 0.15.16 `src/ops/dense_matmul_id.rs` | `d7950752…ada3` (`DenseMatmulIdInputLayout::Slotted`) |
| mlx-native 0.15.16 `tests/deepseek_q2_k.rs` | `508f079e…e375` (slotted == flattened precedent) |
| mlx-native 0.15.16 `src/ops/quantized_matmul_id_ggml.rs` (= 0.15.17) | `9d92db48…dd12` |
| mlx-native 0.15.16 `src/ggml_capability.rs` | `1d79fdee…0947` |
| mlx-native 0.15.16 `src/shaders/quantized_matmul_id_mm.metal` | `0a17b978…3890` |
| mlx-native 0.15.16 `src/shaders/quantized_matmul_id_mm_tensor.metal` | `fd87b189…32ed` |
| mlx-native 0.15.16 `src/ops/quantized_matmul_ggml.rs` | `b51ce1b4…efce` |
| mlx-native 0.15.16 `src/shaders/quantized_matmul_mm_tensor.metal` | `47ef9594…6eacb` |
| mlx-native 0.15.16 `src/shaders/moe_dispatch.metal` | `deeec52e…94ce` |
| peer `e86f5fd` `kernels/mul_mm.metal` | `c3653dbd…307f` |
| peer `e86f5fd` `ggml-metal-ops.cpp` | `c996f88a…5f22` |
| Target GGUF header | `bf65b7e1…8ef53` (tensor types and shapes, §2.2) |
| Profile evidence | `$G/profile/l1/*/server.stderr`, `$G/profile/l3/*/server.stderr`, `$G/profile/l3/*.catsplit.json` |
| Routing proxy | `$E/q1-gemma-routing1511-t0-on-actual-v1/output/routing-witness.jsonl` |
