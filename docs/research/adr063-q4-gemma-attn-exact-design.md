# ADR-063 Q4-GEMMA-ATTN-EXACT: bitwise-exact batched warm-prefill attention for Gemma 4

Status: **draft revision 3, needs design review** (2026-11-05).
- Revision 1 was reviewed by ReviewGemmaExact (REVISE, 8 findings) and
  revision 2 by the same reviewer (REVISE narrowly, R1–R5). §11 records how
  each finding was answered.
- No code, build or hardware run backs this document. Numbers are quoted from
  the receipts named, or marked `[INFERENCE]`.

Story: the operator opened it on 2026-11-05 after `TiledF32` was falsified
(`docs/research/adr063-gemma-tiled-attn-design.md` §12). The root assigns the
issue number.

`$E` = `/Users/robert/.local/state/hf2q/campaigns/adr063-preflight-20261023`.
`K:` = the four-lane bench receipt
`~/.local/state/hf2q/v0130/q4-gemma-kernel/bench-943fc0e-precompiled-fourlane-rerun.*`.

## 1. Decision summary

1. **Goal.** Make Gemma 4 warm-prefill attention faster without changing any
   output bit. The new mlx-native kernel `flash_attn_vec_hybrid_batched_rows`,
   followed by the unchanged NWG reduce, returns per (query row, head) a
   u32-identical F32 output to today's warm route
   (`flash_attn_vec_hybrid_batched` + reduce) over the same inputs.
   - That makes it **Class A** (Q1 §3.1): no decision envelope, near-tie
     policy or paired quality campaign (Q1 §3.6).
2. **How.** Every arithmetic expression and its order per (row, head) stays
   (§3). Only loads and scheduling change:
   - one threadgroup serves P (row, head) pairs that attend one KV head of one
     sequence (one slot) over the same key range;
   - the K rows, TQ-HB V code bytes and V norms of a key sub-tile are staged
     in threadgroup memory once per slice and read by all P pairs.
   - The centroid lookup and the `centroid · (norm · … · w)` product stay per
     consumer: the reference folds the softmax weight into the V scale
     (finding 1).
3. **R = 1 first.** The first measured configuration groups only the
   G = n_heads / n_kv_heads query heads of one row (P = G; G = 2 sliding,
   8 global). That is decode lever A1's shape. Multi-row blocks (R > 1) are
   measured only after R = 1 has passed parity (§5).
4. **Shared native contract.** One mlx-native contract issue and PR serve
   this story and Q4-GEMMA-DECODE lever A1; their R = 1 case is the same
   kernel on the same reference. The hf2q stories, adoption thresholds and
   decode's A2 (mixed buckets) stay separate.
5. **Spike first, kill criteria frozen.** S0 (§5) must show u32 parity on
   the full matrix and a speed gain before any hf2q work. §9 freezes the stop
   rules.

## 2. Evidence

- **Size of the lever.** The share of record for warm-prefill attention is
  the v9 tiled-off cost, 7.8–9.2% of Gemma makespan, i.e. 1.2–1.4 s of
  14.87 s (`docs/research/adr063-q4-gemma-profile-result.md` §5, §7).
  - That is `TiledF16`'s route saving, not a ceiling for this kernel.
  - The profile's 48% attention segment (KV writes, staging, FWHT undo) is an
    upper bound and is not used for sizing.
- **Chain time, `K:`.** Four-lane cells over the six transactions of the
  canonical run, 25 sliding + 5 global layers per transaction:
  - vec-hybrid **chain** (kernel + NWG reduce): 1,430 ms projected per run;
  - vec-hybrid kernel only: 1,393.7 ms;
  - `TiledF16` chain: 441 ms.
  - Sliding layers are 71.5% of the vec chain `[INFERENCE from the cells]`.
  - About 65% of the vec chain runs at NSG = 4 (kL > 1024, second t4 chunk,
    t5, t6) `[INFERENCE from the cells' (H, S) and the 512/1024 split]`.
- **Why vec-hybrid is slow at prefill.** Each (query, head) threadgroup
  reloads K and the V codes for every visible key, as in decode. A suffix of L
  rows re-reads each key about L × G times. The cells imply about
  1.4–1.7 TB/s of cache-served K/V requests, which points to a load-bound
  kernel `[INFERENCE]`.
- **Why tiled routes do not qualify.** They change the reduction tree. At a
  layer-0 difference of 4.95e-7, MoE routing still amplified `TiledF32` to 72
  near-tie failures over 952 rows (tiled-attn design §12). Only a bitwise
  route avoids the envelope.

## 3. Exactness contract (normative; text of the shared native contract issue)

Reference: `flash_attn_vec_hybrid_batched_impl<DK, DV>`
(`mlx-native/src/shaders/flash_attn_vec_hybrid.metal`, at the hf2q pin)
dispatched by `flash_attn_vec_hybrid_batched`, followed by its NWG reduce.

hf2q calls it from `encode_live_seq` with these settings:
- 256-query chunks split at 512/1024 (`vec_query_chunk_len`);
- NWG = `compute_nwg(kv_seq_len)`: 16 at kL ≤ 512, 32 above;
- NSG = `compute_nsg(kv_seq_len)`: 1 at kL ≤ 1024, 4 above;
- `HF2Q_HYBRID_NWG` and `HF2Q_TQ_NSG` unset;
- scale 1.0;
- sliding layers: `mask_type` 3, window 1024, over the ring stage
  (`kv_capacity` = stage capacity);
- global layers: `mask_type` 1, over the cache.

Invariants (bitwise on the raw u32 of every F32 output and partial):

1. **Per-pair identity.** For every (row, head) pair, O and the NWG partials
   `(S, M)` equal those of the reference dispatch on that row alone.
   - This holds for any block composition: which, and how many, other pairs
     share the threadgroup never changes a pair's bits.
   - A neighbouring row holding NaN or ±∞ must not change it.
2. **Q staging.** Q is staged as `half4`, as today.
3. **Chunk ownership.** Slice (iwg, sgitg) owns chunks
   `ic0 = iwg·NSG + sgitg + k·NWG·NSG` of C = 32 keys. This is the
   reference's sequence, with the reference's NWG and NSG for the dispatch.
4. **Q·K.** Per key, the partial products use the same lane-to-term mapping
   (`tx*4 + ii*NL*4`, `NL` lanes per key) and the same `simd_sum`.
5. **Mask and window.** Evaluated per (row, key) as today, including the
   per-row chunk skip `simd_max(ss) <= -65504.0f` (`continue`). Pairs in one
   block may skip different chunks. A pair that skips still takes part in the
   block's barriers and changes no state.
6. **Online softmax.** Per (pair, slice):
   - `M = simd_max(max(M, s_new))`;
   - `ms = exp(M_old − M)`, `vs = exp(s − M)`;
   - `S = S·ms + simd_sum(vs)`;
   - `lo = lo·ms + …` in the same `cc` order.
7. **V operand.** It is formed per consumer exactly as the reference does:
   - d256: `v_sw = v_norm_val * inv_sqrt_dv * ss[cc]`, then
     `lo[ii] += dequant_hb_float4(v_base, …, v_sw, cbits)`;
   - d512: `sn0 = vnorm[0] / sf_d512 * w`, then
     `lo[ii] += dequant_hb_float4(v_base, coord, sn0, cbits)`;
   - the F16-V form (function constant 51) is the reference's own expression.
   - Allowed to share: the code bytes, the norms, and (optionally) a
     threadgroup copy of the codebook's exact floats. A weight-free
     "dequantized V" is never staged.
8. **Merge and reduce.**
   - The cross-simdgroup merge (`j = 0..NSG`) runs in the reference's order
     with the reference's arithmetic.
   - The NWG partials go to the same addresses.
   - The NWG reduce kernel is the stock one, unchanged.
9. **One slot per block.** All pairs of a threadgroup belong to one slot:
   one `k_off`/`v_off` base, one `kv_capacity`. A dispatch whose rows can
   carry different slot ids (decode) must use P = G, i.e. one row per block.
10. **Compile flags and operand form.** Decode design §3.2 items 6–7 are
    imported verbatim:
    - the same compile mode as the reference on both metallib paths, with no
      `-fno-fast-math`, safe math or precise functions;
    - every per-(pair, key) expression is a literal clone of the reference;
    - interleaving pairs is allowed; reassociating any sum is not.
    - Reading K or codes from threadgroup memory instead of device memory is
      exactly the kind of change that flipped Apple's FMA contraction before
      (ADR-040 §0.21c). Only S0-a's tests decide.
    - **Running output in registers is an operand-form change.** In the
      reference the running output `so4` lives in threadgroup memory: in the
      softmax block `so4[ii*NL] *= ms`, then after the V loop
      `so4[ii*NL] += lo[ii]`, separated by stores and barriers (`lo` is a
      per-chunk local). Holding `so` in registers lets the compiler fuse
      `so·ms + lo` into one `fma`, which is one rounding instead of two.
      S0-a must prove register `so` bitwise. If it does not hold, the layout
      keeps `so` in threadgroup memory (L-A-tg, where it fits) or falls back
      to L-B (§4.2).
11. **Compiler dependence.** Decode design §3.3 applies:
    - parity is per metallib path and per compiler;
    - every receipt records the macOS build, `xcrun metal --version` and the
      metallib SHA-256;
    - the parity tests rerun on every pin bump and OS update.

## 4. Proposed kernel and layout

`flash_attn_vec_hybrid_batched_rows<DK, DV>` with function-constant P, R and
KS.

### 4.1 Layout L-A (primary): pair-major simdgroups, two-phase per-chunk staging

- **Grid:** (row block, KV head, iwg). A block holds R consecutive query rows
  of one sequence chunk × the G query heads sharing the KV head, so P = R·G
  pairs.
- **Simdgroups:** P·NSG simdgroups, with P·NSG ≤ 32 (1,024 threads).
  - Simdgroup (p, s) runs pair p on slice s: the reference's
    (iwg, sgitg = s) chunk sequence.
  - Allowed P: NSG = 1 → P ≤ 32; NSG = 4 → P ≤ 8. Global at NSG = 4 and
    G = 8 is therefore R = 1, P = 8.
- **Two-phase lockstep per chunk.** All simdgroups advance the chunk index k
  together. A pair whose row skips the chunk (§3 item 5), or has no chunk at
  k, still takes part in every barrier and changes no state. For chunk
  `ic0 = iwg·NSG + s + k·NWG·NSG` of each slice s:
  1. **Scores.** For each K sub-tile of KS keys (KS ∈ {2, 4, 8, 16, 32}):
     barrier; load the KS F16 K rows of every slice into bank s; barrier;
     each simdgroup (p, s) runs the reference Q·K code for those keys of its
     pair, writing its scores into its own `ss`.
  2. **Softmax.** After all 32 scores of the chunk, each simdgroup runs the
     reference per-chunk softmax on its `ss`:
     - the mask/skip;
     - `M = simd_max(max(M, s_new))`;
     - `ms`, the weights `ss[cc]`;
     - `S = S·ms + simd_sum(vs)`;
     - `so *= ms`.
  3. **V.** For each V sub-tile of KS keys: barrier; load the KS rows of V
     code bytes and V norms of every slice into bank s; barrier; each
     simdgroup accumulates `lo += dequant_hb_float4(…, v_sw or sn0, …)` in
     `cc` order (§3 item 7). Then `so += lo` as in the reference.
- **Storage.**
  - `ss` is per simdgroup in threadgroup memory, as in the reference
    (SH = 128 halves = 256 B).
  - `lo` is a per-chunk local, as in the reference.
  - The running `so` stays in registers, an operand-form change S0-a must
    prove (§3 item 10). L-A-tg keeps `so` in threadgroup memory where the
    budget allows: rows 1, 2 and 5 of the table below.
- **Merge.** After the loop, pairs merge one at a time in the reference's
  order. Each pair writes its slices' `so` into a merge area of NSG·2·PV
  halves, then runs the reference `j = 0..NSG` arithmetic with its `ss`
  values (already resident). The merge area **reuses the staging banks**,
  which are idle after the loop.
- **Threadgroup memory**, ≤ 32,768 B. PK = PV = DK halves and SH = 128 halves
  (`flash_attn_vec_hybrid.rs`: `pad2(head_dim, 128)`, `sh = 4·32`). Columns:
  - Q: P·DK·2 B;
  - `ss`: P·NSG·256 B;
  - bank: NSG·max(K tile KS·DK·2, V tile KS·(DK + 4·norms_per_pos)), and the
    K tile always dominates;
  - merge area: NSG·DK·4 B, which must fit inside the bank.

| Shape | Q | `ss` | bank | merge (in bank) | total |
|---|---|---|---|---|---|
| 1. d256, NSG 1, P 2 (R 1), KS 32 | 1 KiB | 0.5 KiB | 16 KiB | 1 KiB | 17.5 KiB |
| 2. d256, NSG 1, P 16 (R 8), KS 16 | 8 KiB | 4 KiB | 8 KiB | 1 KiB | 20 KiB |
| 3. d256, NSG 4, P 2 (R 1), KS 8 | 1 KiB | 2 KiB | 16 KiB | 4 KiB | 19 KiB |
| 4. d256, NSG 4, P 8 (R 4), KS 4 | 4 KiB | 8 KiB | 8 KiB | 4 KiB | 20 KiB |
| 5. d512, NSG 1, P 8 (R 1), KS 16 | 8 KiB | 2 KiB | 16 KiB | 2 KiB | 26 KiB |
| 6. d512, NSG 4, P 8 (R 1), KS 2 | 8 KiB | 8 KiB | 8 KiB | 8 KiB | 24 KiB |

  - L-A-tg adds P·NSG·2·PV·2 B for `so`: +2 KiB for row 1, +16 KiB for row 2
    (36 KiB, so not at R = 8, KS = 16; KS 8 gives 32 KiB), +4 KiB for row 5.
  - At d512, NSG 4, P 8, `so` alone is 64 KiB, so L-A-tg is impossible there;
    if register `so` fails, that shape uses L-B or the stock fallback.
  - S0 replaces every figure with the measured `threadgroupMemoryLength`.
- **Reuse at NSG = 4.** Each slice owns at most one of the ≤ 54 chunks, so
  there is no chunk loop to amortise. All reuse is across the P pairs inside
  one sub-tile.

### 4.2 Layout L-B (fallback only if L-A fails parity or budget)

sgitg becomes a grid axis, with `so` and `ss` in threadgroup memory per
(pair, slice) exactly as in the reference. Each slice writes its (M, S, so)
partials to device memory. A separate merge pass then runs the reference's
`j = 0..NSG` merge arithmetic in the reference order, before the NWG partial
write.
- This changes how §3 item 8 is executed: a separate pass instead of
  threadgroup banks. The arithmetic stays the same.
- It needs its own S0-a pass, and the doc is amended before any use.

### 4.3 Out of scope

- **Larger chunks.** Query chunking stays exactly hf2q's 256-row chunks
  split at 512/1024.
  - On sliding layers merging chunks would shift every staged key's position
    once the history is full, and move keys to other (iwg, sgitg) owners.
  - Global merging is not proposed either.
- **Mixed slots.** The grid never spans slots (§3 item 9).
- **Decode A2** (mixed buckets) stays decode-only.

### 4.4 Native API (additive)

`dispatch_flash_attn_vec_hybrid_batched_rows(…)`:
- same buffers and params as `flash_attn_vec_hybrid_batched`, plus
  `rows_share_slot: bool` (§3 item 9; false forces P = G);
- (R, KS) come from a frozen table keyed by (DK, G, NWG, NSG, V form,
  codebook bits), set by S0-b (§5);
- shapes outside the table fall back to the stock dispatch, which is
  bitwise by definition;
- existing entry points stay bit-identical.

## 5. S0 feasibility spike (mlx-native worktree, GPU lane, model-free, time-boxed)

**S0-a parity.** Every u32 of O and of the NWG partials, then of the reduced
output, must equal the reference. Both metallib paths; pipeline origin
asserted. Inputs: real TQ-HB-encoded V (`dispatch_hadamard_quantize_kv_hb_seq`)
and RMS-normed Q/K. Matrix:
- **Buckets:** kL at 1, 31, 32, 33, 511, 512, 513, 1023, 1024, 1025, 2047,
  4096. This reaches the three (NWG, NSG) pairs the policy can produce:
  - (16, 1) at kL ≤ 512;
  - (32, 1) at 513–1024;
  - (32, 4) above 1024.

  (16, 4) needs the `HF2Q_HYBRID_NWG`/`HF2Q_TQ_NSG` overrides, which hf2q
  does not set; it is out of scope.
- **Storage form:** register `so` (L-A), and L-A-tg and L-B where they are
  candidates, each checked on the full matrix.
- **Sliding d256 (G = 2):**
  - ring stage with history 0, 1, 465, 714, 943, 1022, 1023;
  - `q_offset > 0`;
  - window boundaries mid-chunk;
  - fully skipped chunks;
  - tail chunks (kL mod 32 ≠ 0).
- **Global d512 (G = 8):** the same kL edges; `q_offset_in_k` with graft 0
  and 37.
- **Row counts:** 1, 3, 7, 9, 31, 32, 54, 56, 92, 97, 132, 191, 200, 220,
  256. These include the canonical chunk lengths and non-multiples of every R.
- **Block composition:**
  - each configuration is checked at R = 1 and at every candidate R > 1;
  - the chunk start is shifted by 1 and by R − 1 rows;
  - NaN neighbour rows are injected;
  - every row must be bitwise the same in all variants.
- **V forms:**
  - TQ-HB codes at codebook bits 8 (Gemma default), 6 and 5;
  - F16 V (function constant 51). If a form fails or is not implemented, the
    dispatcher must route it to the stock kernel, and S0-a checks that
    fallback.
- **Slots:** `rows_share_slot = false` forces P = G (checked with differing
  slot ids).
- **Decode cells** (shared contract): Q4-GEMMA-DECODE T-N8's R = 1 matrix,
  run in the same suite.

**S0-b speed.** GPU timestamps; 10 warmup + 50 measured per cell; medians;
precompiled path for the decision, runtime path recorded.
- **Arms:** the new kernel + the stock reduce, against the vec chain
  (kernel + reduce), on `K:`'s four-lane cells (`warm-lanes.json`).
- **Weighting:** 25 sliding + 5 global layers per transaction, summed over the
  six transactions.
- **Report:** r per layer type, per (NWG, NSG), and per R. R = 1 is measured
  first; R > 1 only after R = 1 passes S0-a.
- **R table rule:** for each (DK, G, NWG, NSG) the fastest (R, KS) that
  passes S0-a. Ties within 1% go to the smaller R. The table is frozen in the
  native release receipt and approved by the root.

**S0 output.** A receipt with:
- the parity matrix on both paths;
- the chosen layout (L-A, or L-B with the amendment);
- the R/KS table;
- measured register and threadgroup-memory use;
- r per cell and projected chain total;
- compiler and OS identities.

## 6. hf2q integration (after the native release and an exact pin)

- **Variant and selector:** `GemmaLiveKernel::VecHybridBatched`, a bitwise
  sibling of `VecHybrid`.
  - `HF2Q_GEMMA_WARM_ATTENTION_EXACT` (read once per model load; `1` by
    default after promotion, `0` the kill switch) maps a resolved `VecHybrid`
    warm kernel to `VecHybridBatched`.
  - `TiledF16` (the `1` arm of `HF2Q_GEMMA_WARM_TILED_ATTENTION`) is
    unaffected.
  - The real-artifact tests select the variant in-process through
    `weights.warm_live_kernel`, as they do today.
- **Route:** `LiveRoute` treats `VecHybridBatched` exactly like `VecHybrid`:
  - 256-query chunks with the 512/1024 split;
  - stage rows `cap − 1 + min(rows, 256)`;
  - the same `LiveVecScratch`;
  - no 32-row threshold, so all rows take it.
  - `encode_live_seq` calls the new dispatcher with `rows_share_slot = true`
    (one sequence per dispatch).
- **Dispatch points that must name the new variant.** Each of these, in
  `src/serve/gemma_live_attention.rs` today, otherwise silently routes
  `VecHybridBatched` to the stock kernel:
  - `LiveRoute::select`: `PerSlotExact`/single-slot returns the warm kernel
    as resolved;
  - `LiveRoute::seq_kernel`: its catch-all `_ => VecHybrid` must return
    `VecHybridBatched` for that route;
  - `LiveRoute::uses_vec` (scratch allocation): true for both vec variants;
  - `chunk_len` and `stage_query_rows`: the vec arms;
  - `fill_vec_stage_positions`: the vec chunking, shared by both variants;
  - the `encode_live_seq` kernel match.

  Every `match` on `GemmaLiveKernel` is made exhaustive without a wildcard,
  so a new variant fails to compile until each point decides.
- **`Tiled` descriptor.** Installed long-prompt lanes
  (`GemmaLiveAttention::Tiled`) with `HF2Q_GEMMA_TILED_LIVE=0` resolve to
  `VecHybrid`, and they **stay on the stock kernel** in this story. Those
  lanes are not bitwise-coupled to per-slot prefill and are outside the warm
  route; moving them is a separate change. A unit test pins this.
- **Out-of-table shapes.** The native dispatcher's stock fallback (§4.4) is
  allowed; X2–X4 measure how often it runs and require it not to take every
  dispatch.
- **Runtime:** `wire_name()` = `"VecHybridBatched"`, published as
  `active_controls.gemma_warm_attention_kernel`; Q5 binds it with
  `HF2Q_Q5_GEMMA_WARM_ATTENTION_KERNEL` (`skip_if_empty`).
  - The banner reads `live=vec-hybrid-batched`.
  - The launcher gains `WARM_ATTENTION_EXACT=0|1`.
- **Decode:** untouched in this story. Lever A1 of Q4-GEMMA-DECODE consumes
  the same native kernel at P = G.

## 7. Gates (all Class A; any byte fails)

| Gate | Test | Required |
|---|---|---|
| X1 primitive parity | the S0-a matrix as mlx-native ignored tests, both metallib paths, origin asserted; rerun at every pin bump and OS update | u32-equal |
| X2 model route | `gemma_warm_attention_exact_matches_stock_bitwise`: one model load; the four gate lanes per slot with `warm_live_kernel` = `VecHybrid` then `VecHybridBatched`; plus a 31-row and a 1-row suffix; run on both metallib paths | first token, logits bits, KV bytes, cursors; **and the new kernel ran**: with `MLX_DISP_BUCKET=1`, `pipeline_dispatch_buckets()` reset per arm shows `flash_attn_vec_hybrid_batched_rows_*` count > 0 in the batched arm and = 0 in the stock arm |
| X3 joint == per-slot | `gemma_joint_warm_prefill_matches_per_slot_bitwise_vec_hybrid_batched` (Stage 1 cohorts) | bitwise; the same dispatch-count assertion per case |
| X4 envelope harness | `gemma_warm_attention_route_envelope` with `VecHybridBatched` in `ENVELOPE_ROUTES` | every row `d_nat = 0`, `q1_class_b = ExactEqualDecision`, 0 flips; the same dispatch-count assertion |
| X5 Q1 | T1/T3 greedy144 endpoints, packet `warm_live_kernel: VecHybridBatched` against `VecHybrid` | `q1_decision_policy.py exact` → `ExactEqual` |
| X6 Q5 | A/B of `HF2Q_GEMMA_WARM_ATTENTION_EXACT` 0/1 on `q5_cases/gemma4.json`, ABBA, ≥ 5 runs per arm, readiness bound to the kernel | `CalibrationPass`; `transcripts` → `ExactTranscripts` across arms; the receipt records the share of warm vec dispatches that took the new kernel (`MLX_DISP_BUCKET=1` dump at shutdown) against the stock fallback; the §9 speed rules |

## 8. Expected saving `[INFERENCE]`

- **r is unmeasured;** S0-b measures it.
  - At the S0 go threshold (chain ratio 0.70), the chain saving is
    (1 − 0.70) × 1,430 ms ≈ **0.43 s** per run.
  - `TiledF16`'s chain saved 0.99 s and its route A/B saved 1.16–1.4 s of
    makespan, so 0.43 s of chain is about **0.50–0.60 s** of makespan.
- **Feasibility of 0.70 under finding 1.** Codebook lookups stay per
  consumer. Sliding layers (71.5%) reuse only G = 2 at R = 1, so K and code
  loads at most halve.
  - If sliding reaches r ≈ 0.75, global must reach ≈ 0.57 for 0.70 overall.
  - R > 1 on sliding is likely needed.
- **Budget effect.** It would lift the revised Gemma budget (ADR-063
  2026-11-05: 0.64–2.15 s) by about 0.5–0.6 s at the threshold, and more if
  r is lower. That is still short of 3.0 s at the low end.

## 9. Kill criteria (frozen)

- **S0-a, parity.**
  - Fixes are limited to operand-form changes (§3 item 10: literal clones,
    load placement, register form). Changing any §3 arithmetic item is not a
    fix.
  - Time box: 2 spike iterations or 5 working days, whichever ends first.
  - Any remaining non-equal u32 in any cell, on either path, stops the story.
    It is recorded as falsified; no Class B fallback is designed here.
  - Shapes routed to the stock fallback are not failures; they only reduce
    the saving.
- **S0-b, speed.**
  - Weighted chain ratio > 0.70 against the vec chain on the precompiled path
    stops the story.
  - Any cell more than 2% slower than the vec chain blocks that cell's
    (DK, G, NWG, NSG) table entry; it falls back to stock.
- **M0, after the hf2q build (X6 runs).** Promotion is stopped if either:
  - the median over matched transactions (identical per-lane cached/advanced
    counts across arms) of per-run joint-warm `elapsed_ms` saving is below
    0.25 s;
  - the makespan regresses: ≥ 5 runs per arm in ABBA pairs; promotion stops
    if the 95% t-interval of the paired difference (arm 1 − arm 0, per ABBA
    pair) lies entirely above 0. A 1-run-per-arm noise band (about ±4% in
    #660's MT vs P0) makes a fixed-percentage median rule unreliable at these
    counts.
- **X1–X6.** Any non-bitwise result blocks promotion. The route stays off
  pending localization and is never relaxed to an envelope.

## 10. Release and sequencing

1. One mlx-native contract issue, shared with Q4-GEMMA-DECODE lever A1. It
   carries:
   - §3 of this document;
   - the A1 correction to finding 1: V is decoded per consumer, never
     "decoded once";
   - the one-slot-per-block rule;
   - one combined parity matrix (decode T-N8 + S0-a).
2. S0 → native PR → independent review → release → crates.io checksum → hf2q
   exact pin.
3. hf2q integration PR (§6) → X2–X5 on the GPU lane → X6 → promotion
   commit.

## 11. Review record

### 11.1 ReviewGemmaExact on revision 1: REVISE

| # | Finding | Answer in revision 2 |
|---|---|---|
| 1 (P1) | The reference folds the softmax weight into the V scale; a weight-free dequantized V cannot be staged bit-equal | §3 item 7: stage codes, norms and optionally codebook floats; lookup and multiply per consumer. The shared contract corrects decode A1 (§10) |
| 2 (P1) | NSG = 4 (≈ 65% of the vec chain) has no workable layout; memory and thread limits | §4.1 L-A: P·NSG ≤ 32 simdgroups, per-slice staging banks, register accumulators, one reused merge bank, per-shape budget table; §4.2 L-B fallback; reuse at NSG 4 is across pairs only |
| 3 (P2) | Larger chunks change sliding key ownership | §4.3: chunking unchanged for both layer types |
| 4 (P2) | S0-a/X1 matrix gaps | §5: all NWG/NSG buckets (incl. 512/513), sliding histories below 1023, non-multiple-of-8 and canonical row counts, block-composition check, F16 V and codebook 5/6/8 or stock fallback, slot rule |
| 5 (P2) | One slot per row block | §3 item 9 and `rows_share_slot` (§4.4) |
| 6 (P2) | X2–X4 not runnable without a reachable selector | §6: `VecHybridBatched` variant, in-process selection, runtime publication; X2–X4 rewritten (§7) |
| 7 (P3) | Saving numbers inconsistent | §2/§8: 1,430 ms is the chain (kernel only 1,393.7 ms); 0.43 s chain at 0.70 ≈ 0.50–0.60 s makespan; r unmeasured; S0-b spec |
| 8 (P3) | Kill criteria not fully frozen | §9: operand-form-only fixes and a time box; matched-transaction statistic, run count, 1% makespan rule; per-cell 2% rule; R-table rule (§5) |

Open questions from revision 1, as answered by the reviewer and adopted:
1. Measure R = 1 first; R > 1 follows only after R = 1 parity (§1 item 3,
   §5).
2. Dequantized V cannot be staged bit-equal (finding 1, §3 item 7).
3. Shared native contract issue and PR with decode lever A1; separate hf2q
   stories and thresholds (§1 item 4, §10).

### 11.2 ReviewGemmaExact on revision 2: REVISE narrowly

| # | Finding | Answer in revision 3 |
|---|---|---|
| R1 (P2) | Running output in registers is not the reference's storage; `so·ms + lo` may contract | §3 item 10 lists register `so` as an operand-form change S0-a must prove; L-A-tg keeps `so` in threadgroup memory where it fits; otherwise L-B (§4.2) or the stock fallback |
| R2 (P2) | Softmax runs per chunk; sub-tile V cannot be consumed before it | §4.1 is now two-phase per chunk: K sub-tiles → Q·K into `ss` → per-chunk softmax and `so *= ms` → V code/norm sub-tiles in `cc` order → `so += lo`. Bank = max(K tile, V tile); table recomputed |
| R3 (P2) | Per-simdgroup `ss` scratch missing; merge figures wrong | §4.1: `ss` column (P·NSG·256 B); merge area NSG·DK·4 B reuses the staging banks after the loop; all six rows ≤ 26 KiB in L-A, with L-A-tg figures |
| R4 (P2) | Silent fallbacks can pass X2–X4 vacuously | §6 names `select`, `seq_kernel`, `uses_vec`, `chunk_len`/`stage_query_rows`, `fill_vec_stage_positions` and the `encode_live_seq` match, all exhaustive; the `Tiled` descriptor stays on stock; X2–X4 assert the new pipeline's dispatch count > 0 / = 0; X6 records the share |
| R5 (P3) | M0 at 3 runs is weak; (16, 4) unreachable | §9: ≥ 5 runs per arm, paired 95% t-interval rule; §5 lists the three reachable (NWG, NSG) pairs |

