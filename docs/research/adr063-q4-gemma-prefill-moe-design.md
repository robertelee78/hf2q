# ADR-063 Q4-GEMMA-PREFILL-MOE: Gemma 4 warm-prefill MoE and dense cost per row — HLD/LLD

Status: design for approval (2026-11-04). No code, build or hardware run backs
this document. Every number is quoted from a retained receipt (path given),
computed from source or file metadata (method given), or marked `[INFERENCE]`.

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
2. **The cause is launch geometry, not arithmetic.** hf2q's `mm_id` launches a
   uniform grid of 128 experts × ⌈rows/32⌉ tiles × N tiles. The Gemma MoE down
   call also flattens its rows to `8·n` tokens, which multiplies its grid by 8.
   Over one four-agent run that is about **225 M threadgroups launched, of which
   about 3.1 M have work** (§2.4). The peer build's dispatch launches only a
   compact list of tiles that have work, and it skips the empty half of
   half-filled tiles (§2.5).
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
   profile's "≤ 0.3 s after #668" bound. The range depends mostly on one
   unmeasured constant: the GPU cost of an empty threadgroup. The first GPU step
   (M0, §8) measures it model-free, on today's kernels, before any
   implementation. If M0 predicts less than 0.05 s per run, the story stops
   after recording the result (§7.4).
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

**Today's run launches 224.8 M threadgroups, and about 3.1 M of them do
work.** The empty ones only read one word and exit, but each still costs launch
and scheduling time. If an empty threadgroup costs c_e of GPU time (amortized
across cores), the launch waste per run is 0.056 s at c_e = 0.25 ns, 0.11 s at
0.5 ns, 0.22 s at 1 ns and 0.45 s at 2 ns. c_e is unmeasured. M0a measures it.

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
- cohort width: already four lanes, except in the 2 of 12 runs above;
- expert weight bandwidth: `[INFERENCE]` the gate/up and down expert stacks are
  285 MB and 174–269 MB per layer, read about once per row tile. At ≤ 2 tiles
  per expert that stays well under the time measured.

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
| P2c half-tile skip | **A or not shipped** | bitwise against today's kernel per type and path (§5.1 N5). If the tensor path is not bitwise, P2c ships for the simdgroup path only if that is bitwise, and otherwise not at all. No Class B route enters production through this story |

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

The new rule follows gate/up's own routing. When gate/up ran `mm_id`
(`plan_expert_auto_route(seq_len, top_k, hs, false, policy) == Mm`, i.e.
`seq_len > 32`), the down call uses
`GraphSession::quantized_matmul_id_ggml_pooled_slotted` with:
- `n_tokens = seq_len`, `top_k = top_k`;
- the same `pf_moe_swiglu` input, `[seq_len, top_k, moe_int]` F32;
- the same `pf_expert_ids`, output `pf_moe_down` and scratch;
- `n = hs`, `k = moe_int`, `expert_stride = down_expert_stride`,
  `ggml_type = down_ggml_dtype`.

Otherwise (`seq_len ≤ 32`) the existing flattened call stays, because at 5–32
rows the flattened down call is `mm_id` while the slotted contract refuses below
the threshold. Today's routing is kept in both cases, as R3 requires.

Further rules:
- **Affine stacks.** The `gemma_moe_use_affine` branch (`quantized_matmul_id_into`)
  and the imatrix intercept keep their flattened form. They are not on the
  served GGUF route.
- **Scratch.** `IdMmScratch::alloc(dev, num_experts, seq_len·top_k_max)`
  (`:1519-1521`) already covers `n_experts × seq_len`.
- **Switch.** `HF2Q_GEMMA_PREFILL_MOE_DOWN = flat | slotted`, resolved once at
  model load in a small module beside `gemma_live_attention.rs`:
  - default `slotted` after §7 passes; `flat` is the A/B arm;
  - an invalid value warns and keeps the default;
  - published in `/hf2q/v1/runtime` active controls;
  - P2's promotion commit deletes the `flat` arm (§4.5).
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
- Grid: `(max_tiles, ⌈N/64⌉, 1)`, with
  `max_tiles = min(rows, ⌈rows/32⌉ + min(n_experts, rows) − 1)` and
  `rows = n_tokens·top_k`.
- Word 0 has POISON: threadgroup (0, 0) writes NaN over the whole logical output
  and every threadgroup returns before forming a weight or activation address.
  This is today's contract (`quantized_matmul_id_ggml.rs:289-296`).
- `tile ≥ tile_count`: return.
- Otherwise load `(im, idb, nr1)` and run **today's tile body unchanged**, with
  `ids_i32[im·ne21 + r1 + j]` replaced by `ids_i32[idb + j]`.

**Scratch.** `IdMmScratch::alloc(device, n_experts, max_rows)`, where
`max_rows` bounds `n_tokens·top_k` over the session. The scratch holds:
- word 0 plus 3·`max_tiles` descriptor words;
- `max_rows + n_experts` list entries.

While `UniformGrid` exists, the list is sized `max(n_experts × max_rows,
max_rows + n_experts)` so either schedule fits. Every hf2q caller already passes
`seq_len·top_k` (Gemma) or its row bound (DeepSeek-V4 `verifier_forward.rs:646, 919`,
`ffn_forward.rs`), and each is re-checked at the pin bump. `check_capacity`
checks `(n_experts, rows)`.

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
- the encoder identity and a per-scratch encode serial.

The new call validates that key against its own `(ids, n_tokens, top_k,
n_experts)`, the current encoder and "no `map0` since". On any mismatch it
returns `InvalidArgument` before encoding. hf2q treats that as a bug and
propagates it.

The existing `map0` barrier orders schedule construction. The graph's
`barrier_between` before down already covers `pf_moe_swiglu` and `pf_expert_ids`
and is unchanged. This generalizes what `pooled_pair` does internally.

### 4.4 P2c — half-tile skip (conditional)

The `mm_id` v2 body splits its 32-row B tile into two 16-row halves, as the peer
build does:
- **Tensor path:** two `matmul2d_descriptor(16, 64, 32)` products, the upper one
  skipped when `nr1 ≤ 16`.
- **Simdgroup path:** simdgroups 2–3 skip their MMA and write-back when
  `nr1 ≤ 16`.

It ships under the same `CompactTiles` schedule only if §5.1 N5 is bitwise
against the 32-row body for that type and path. A failed path keeps the 32-row
body. The receipt records which paths ship.

### 4.5 P3 — hf2q: pin, integrate, promote

1. Pin the mlx-native release that contains P2/P2b(/P2c), with the
   registry-adoption record and Cargo.lock checksum.
2. Gemma down (P1's slotted branch) calls the reusing entry point, because
   gate/up built the schedule in the same session.
3. The runtime-controls surface reports `prefill_moe_down` and
   `mm_id_schedule`.
4. After §7 passes, the promotion commit:
   - deletes the `flat` arm and `HF2Q_GEMMA_PREFILL_MOE_DOWN`, since slotted
     is then the only route for `seq_len > 32`;
   - keeps `HF2Q_MM_ID_SCHEDULE=grid` as the native diagnostic until the next
     native release removes `UniformGrid` (decision D3);
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
| N5 | half-tile skip vs the 32-row body, bitwise | N2's matrix restricted to tiles with `nr1` ∈ {1, 15, 16, 17, 31, 32} by construction; per type and path, pass/fail recorded |
| N6 | capacity | `max_tiles` bound attained (one row per expert plus full tiles); scratch too small → `Err`; `UniformGrid`-sized scratch fits `CompactTiles` |
| N7 | row independence | a row's output is bitwise equal whether it is computed alone (n = 33 padding rows with fixed ids) or within any of N1's batches. This is the primitive form of joint == per-slot |
| N8 | existing suites unchanged | `adr_022_phase2_q5_k_mm_id_parity`, `adr_033_pi_iq4_xs_mm_id_parity`, `moe_pooled_pair_contract`, `test_quantized_matmul_id_mm`, `ggml_capability` receipts regenerated with the new schema version |

### 5.2 hf2q, model-free

- `gemma_prefill_moe_down_switch`: parse table (`flat`/`slotted`/invalid →
  default); runtime-controls publication; `worker_enabled` stays active under
  both values.
- Route selection: `seq_len ∈ {5, 32}` keeps the flattened call, and
  `seq_len ∈ {33, 1420}` selects slotted, with a recorded-dispatch fake that
  captures the params.

### 5.3 hf2q, real artifact (Class A; ignored tests; `--exact --ignored --test-threads=1`, one process each)

Run each with the switch at its default and again under the A arm
(`HF2Q_GEMMA_PREFILL_MOE_DOWN=flat`, plus `HF2Q_MM_ID_SCHEDULE=grid` after P2):
1. `gemma_joint_warm_prefill_matches_per_slot_bitwise` and `…_vec_hybrid`
   (`engine_gemma_warm_batch_tests.rs`); also on `TiledF32` once #668 lands.
2. `gemma_joint_warm_engine_cohort_matches_serial_tool_turns`,
   `gemma_joint_warm_forward_fault_retries_each_lane_per_slot`.
3. `slot_aware_n4_per_slot_parity_vs_serial`,
   `slot_aware_n4_batched_body_same_prompt_parity_vs_serial`,
   `iter_g_a_multi_seq_prefill_first_token_isolation`
   (`HF2Q_BYTE_EQUIV_E2E=1`).
4. `gemma_n8_decode_then_tiny_cold_prefill_is_repeat_invariant`.
5. **New: `gemma_prefill_moe_route_matches_previous_route_bitwise`.** On the
   official GGUF, with one process and both arms in sequence, a cold prefill
   (458 rows) and a four-lane joint warm forward (the turn-1 shapes, 4 × 58 rows
   over 458-row histories) compare, A against B:
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
- `slot_aware_qwen35_n8_per_slot_parity_vs_serial` (Qwen MoE);
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
- **M0a, empty-threadgroup cost** (today's kernels, before P2 is written):
  - `mm_id` (Q8_0 down shape) with every id = expert 0 and `n_tokens` swept
    over {33, 256, 1024, 4096, 11360} with `top_k = 1`. Busy work is constant
    and only the empty grid grows;
  - the slope of GPU time against launched empty threadgroups is c_e;
  - a second sweep with all ids on expert 127 checks that empty-threadgroup
    position does not matter.
- **M0b, served shapes, today's kernels.** For n ∈ {232, 528, 764, 880, 1172,
  1412, 1420} it measures:
  - gate/up (Q4_K, 2816→1408, k8) and down (Q8_0 and Q5_0, 704→2816), each
    flattened and slotted;
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
  `CompactTiles` (± P2b reuse, ± P2c).
- **Output.** Per shape, the launched and busy threadgroups, ms, and Δ ms. Then
  `S_pred`: the per-run saving predicted by weighting the shapes with the
  workload's forward sizes, 30 layers, and the Q5_0/Q8_0 layer split.

**Decision rule (frozen here):**
- If M0a/M0b predict `S_pred(P1) < 0.03 s`, P1 is still implemented, because it
  is cheap and bitwise, but it gets no speed claim.
- If `S_pred(P1 + P2 + P2b + P2c) < 0.05 s`, the story stops after P1 and its
  disposition is recorded: falsified lever, not implemented (§7.4).
- At no measured shape may `CompactTiles` be slower than `UniformGrid` by more
  than max(2%, 5 µs).

## 7. Acceptance gates (frozen before any data)

### 7.1 Speed, against the measured shares

Baselines are from `$G` at hf2q `7f295b38` / native 0.15.16 (§2.1). Each gate is
re-baselined on the A arm of the same binary at the candidate commit.

| Gate | Metric | Pass |
|---|---|---|
| S1 (primitive) | M0c per-shape GPU time | no regression beyond max(2%, 5 µs); `S_pred ≥ 0.05 s` per run |
| S2 (warm prefill, primary) | per-run sum of unprofiled joint-warm forward walls; all forwards are matched across arms because the arms are bitwise (verified, §7.2) | paired-block median Δ (B − A) ≤ −0.05 s, 95% bootstrap CI upper bound < 0, over ≥ 6 ABBA pairs. Baseline A ≈ 3.19–3.24 s |
| S3 (category) | catsplit MoE + layer tail GPU s per run (`HF2Q_GEMMA_PREFILL_CATSPLIT=1`, 3 runs per arm) | median reduction ≥ 50% of `S_pred`; pre-attention, attention and dense change by less than ±3%. Baseline 1.09–1.11 s |
| S4 (makespan) | Q5 four-agent client makespan, same runs as S2 | no regression: median Δ ≤ 0 and 95% CI upper bound ≤ +0.05 s; Δ reported |
| S5 (n1) | one-agent control, 2 runs per arm | disclosed, not gated (ADR-063 Q4) |

The story's "≤ 0.3 s" is the profile's bound on this whole lever, not a gate.
A result between 0.05 s and S_pred passes. A result below 0.05 s is §7.4.

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
  negative-control receipt. While `UniformGrid` exists, scratch bytes are
  unchanged or lower. Removing `UniformGrid` later reduces the `mm_id` list for
  a 4,096-row forward from 16.8 MB to about 131 KB (arithmetic). The closing
  candidate gets a fresh Q5 calibration in any case.

### 7.4 Falsification outcomes

- **Any Class A gate fails** (one differing byte): the change is a defect, not
  a near-tie. Localize it with the §5.1 matrix. The change does not ship.
- **S1 fails on a shape:** P2 does not become the default for that type; that
  type keeps `UniformGrid`, and this is recorded per type.
- **S_pred < 0.05 s or S2 fails:** record "launch geometry is not a material
  Gemma cost on this host", with the M0 receipts. P1 still merges, because it
  is bitwise and costs nothing. P2/P2b/P2c are not built, or are reverted
  before release. The Gemma gap plan in the profile §7 is updated by the root.

## 8. GPU validation plan (root only; GPU lane; one model process at a time)

Every run records the source commit, binary SHA-256, artifact digest
(`bf65b7e1…`), native version and metallib SHA-256, and the case file and its
SHA-256. It passes Q5 `CalibrationPass`. Evidence goes under
`/Users/robert/.local/state/hf2q/v0130/q4-gemma-prefill-moe/`. Check `df -h /`
(≥ 20 GB) and free memory first. Hold HOST-LOCK for each timed sequence.

| Step | What | Model | Est. time |
|---|---|---|---|
| M0a/M0b | §6 on an mlx-native worktree at 0.15.16 (today's kernels) | none | ~15 min |
| M1 | one four-agent run at hf2q main with `HF2Q_PROFILE_GPU_TS=1` for the per-op split (`moe_gu`, `moe_swiglu`, `moe_dn`, `moe_wsum_add`, `gur`, dense down, O-proj), with the distortion against an unprofiled run stated; plus the read-only capture of the warm forwards' expert-id tables for M0b | Gemma | ~10 min |
| M2 | P1 Class A gates §5.3 (1–6) | Gemma, 8–10 processes | ~60 min |
| M3 | P1 A/B: `HF2Q_Q5_GEMMA_PREFILL_MOE_DOWN ∈ {flat, slotted}`, ABBA ≥ 6 pairs, then catsplit 3 + 3 | Gemma | ~45 min |
| M4 | P2 native qualification: §5.1 on both metallib paths; M0c | none | ~30 min |
| M5 | P3 pin: §5.3 again (arm A = flat + grid) and §5.4; A/B (A = flat + `HF2Q_MM_ID_SCHEDULE=grid`, B = default), ABBA ≥ 6 pairs; catsplit 3 + 3; n1 2 + 2 | Gemma, DeepSeek-V4, Qwen | ~2.5 h |
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
- the per-forward `Batched prefill complete` lines (S2);
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
| P2c half-tile skip | −11% matrix rows (proxy), if MMA + B staging is 25–50% of `mm_id` time | 0.02–0.05 |
| **Total** | | **0.08–0.28** |

### 9.2 Stage ownership and cross-story coupling

| Profiled stage (#660) | Owner |
|---|---|
| Warm-prefill attention, 48% | #659 → #668 (TiledF32) |
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
| c_e is tiny, so the lever is small | M0a measures it first, on today's kernels, before P2 work; the §6 decision rule stops the story |
| The routing proxy misstates real tile fill | M0b uses recorded agent routing tables (M1); S3/S2 measure the real effect |
| `matmul2d` at M = 16 reduces in a different order from M = 32 | P2c is bitwise-gated per path (N5); otherwise dropped |
| Compact `map0` bug: out-of-bounds list write, wrong descriptor at the tile boundary, poison regression | N3, N6, N7; guard-sized buffers; the descriptor bound is asserted against `max_tiles` |
| Shared kernel regresses another family | §5.4 gates at the pin bump; the schedule is per policy, so `grid` stays available until the gates pass |
| Schedule reuse reads a stale schedule | keyed reuse fails closed (N4); only same-encoder, same-ids reuse |
| Thread 0's prefix sum serializes for `n_experts` = 256 | O(n_experts + tiles) per call; included in M0c timings |
| Capability/receipt schema change breaks consumers | schema version bump and a registry-adoption record at the pin |
| Measurement distortion | primary gate unprofiled (S2); catsplit only for the category share (+13% known distortion) |
| Cohort splits (2 of 12 runs) add noise to S2 | Class A arms give identical transactions, so pairing is per transaction; split runs are reported, not dropped |

## 11. ADR-063 story section (extraction-ready)

### Q4-GEMMA-PREFILL-MOE. Reduce the Gemma 4 warm-prefill MoE cost per row

Outcome: Gemma 4 26B-A4B warm prefill spends measurably less GPU time per row in the routed experts, bitwise identical to the previous route, and the saving enters the combined Gemma candidate.

Requirement: The program MUST remove the expert matrix-multiply launch overhead that the Q4-GEMMA profile attributes to warm-prefill MoE, without changing any output byte, and MUST record the measured saving against the profiled MoE share.

Acceptance:

- Starting point (Q4-GEMMA #660, hf2q `7f295b38`, native 0.15.16): warm-prefill MoE and layer tail are 34% of warm-prefill GPU time (1.09–1.11 s per run serialized; attention 48%, dense 11%, pre-attention 7.5%). About 225 M expert-matmul threadgroups are launched per four-agent run, of which about 3.1 M have work.

- A written design, approved before implementation: `docs/research/adr063-q4-gemma-prefill-moe-design.md`. It covers the Gemma prefill MoE down call on the slotted expert entry point (P1, hf2q); a compact per-expert tile schedule with schedule reuse (P2, P2b, mlx-native); and a half-tile skip only where it is bitwise (P2c). The `mm_id`/`mv_id` threshold (32), dense thresholds, tensor-API selection and storage types are unchanged.

- Every shipped change is Q1 Class A against the previous route. The mlx-native parity matrix (design §5.1) is bitwise on both metallib paths, and so are the hf2q real-artifact gates (joint warm == per-slot on every warm attention route, engine cohort == serial tool turns, per-slot parity, the route A/B layer-byte test, and the greedy144 Q1 endpoint `ExactEqual`). `ExactTranscripts` holds over every A/B run pair. A non-bitwise variant does not ship under this story.

- Speed, with one binary and the previous route as the A arm: at least 6 ABBA pairs. The per-run sum of unprofiled joint-warm forward walls improves by at least 0.05 s (95% CI upper bound < 0). The profiled MoE category falls by at least half the microbenchmark prediction, with the other categories within ±3%. The four-agent makespan does not regress (CI upper bound ≤ +0.05 s). The worst-slot TTFT p95 and SSE-gap max ratios stay within 1.025. If the microbenchmark predicts less than 0.05 s per run, the story closes with the recorded falsification and P1 only.

- The shared-kernel regression gates for DeepSeek-V4 and Qwen MoE pass at the native pin. Every run passes Q5 `CalibrationPass`.

Repository ownership: mlx-native owns the compact schedule, the schedule-reuse entry point, the half-tile skip, the parity tests, the microbenchmark and the release. hf2q owns the slotted down call, route switches, integration, the real-artifact gates and the A/B campaign, consuming mlx-native at an exact published pin.

Validation: Independent review of the design, the parity and microbenchmark receipts, the Class A real-artifact receipts and the A/B campaign from raw receipts.

Predecessors: Q4-GEMMA (#660, profile result PR #687), Q5 (#371). Consumed by Q4-GEMMA-ATTN's combined candidate, which adds this story to its predecessors.

Priority: High. Core scope (second-tier family).

Delivery boundary: P1 merged to main, and the mlx-native release containing P2 pinned by hf2q, for v0.1.30 (or the first published hf2q release containing the accepted Gemma 4 configuration).

## 12. Decisions for the operator

- **D1. Exactness scope.** Only Class A changes ship under this story, and a
  non-bitwise half-tile skip, threshold change or Q5_0 tensor path is out of
  scope. *Recommended: yes.* A Class B MoE route would need its own envelope
  and quality campaign, and Gemma's routing amplifies perturbations.
- **D2. One story or two.** One story with two PRs (mlx-native, then the hf2q
  pin), in the Q4-QWEN38 pattern, or split into a native KERNEL story and an
  hf2q story as #659/#668 did. *Recommended: one story.* The native part is
  bitwise and has no separate contract issue beyond §4.2.
- **D3. Kill switches.**
  - `HF2Q_GEMMA_PREFILL_MOE_DOWN` is deleted at promotion.
  - `HF2Q_MM_ID_SCHEDULE=grid` stays as a native diagnostic for one release;
    the release after Q4 closes removes `UniformGrid`.
  - *Recommended as written.* The alternative is to remove `UniformGrid` in
    the same release and run the A/B across two pins, which is weaker evidence.
- **D4. Falsification threshold.** Stop after P1 if the microbenchmark
  predicts < 0.05 s per run. *Recommended: 0.05 s,* about a sixth of the lever's
  profile bound and about twice the S2 noise floor `[INFERENCE]`.
- **D5. What the story's own acceptance requires.** A category saving with no
  makespan regression; the Win is measured only on the combined candidate.
  *Recommended: yes.* A 0.1–0.3 s saving cannot be resolved as a makespan Win on
  its own.
- **D6. Cross-family gates at the pin.** DeepSeek-V4 and Qwen MoE bitwise gates
  plus one `ExactTranscripts` smoke each, against Gemma-only gates.
  *Recommended: run them,* since every `mm_id` consumer changes.
- **D7. Release coupling.** Ship P2 in the same mlx-native release as
  Q4-GEMMA-KERNEL (#659) or on its own. *Recommended: whichever is ready first.*
  The changes touch disjoint kernels and registry entries.

## 13. Source identities read

| Path | Identity |
|---|---|
| hf2q `src/serve/forward_prefill_batched.rs` @ `ff4af4bd` | sha256 `3f6541a4…e08e090` |
| hf2q `src/serve/api/engine_gemma_warm_batch.rs` @ `ff4af4bd` | `6ae9d390…b4fa53` |
| hf2q `src/inference/expert_dispatch.rs` @ `ff4af4bd` | `cd54a844…f9e040e227` |
| hf2q `src/serve/forward_mlx_shared.rs` @ `ff4af4bd` | `56ffe77c…aa59` |
| hf2q `docs/research/adr063-q1-numeric-policy.md` @ `ff4af4bd` | `16df680e…fe35` |
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
