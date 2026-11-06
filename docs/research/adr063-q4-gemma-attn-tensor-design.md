# ADR-063 Gemma 4 gap program, F1: tensor flash attention for warm and cold prefill

Status: **approved by root (2026-11-06, gap program decision)**. Owner: GemmaAttnImpl.

No hardware run backs the claims in this document unless it names a receipt. Estimates are marked `[INFERENCE]`.

## 1. Problem (measured)

The per-op deficit table is `~/.local/state/hf2q/v0130/gemma-attn-gap/REPORT.md`, with evidence in `run-20261106T172501Z/`. It is model-free and compares hf2q against the peer's `FLASH_ATTN_EXT` on identical shapes, M5 Max.

| phase | deficit per four-agent run | hf2q vs peer |
|---|---|---|
| warm prefill attention | +1.256 s (sliding +0.89, global +0.37) | 3.1–10.0× slower per lane |
| decode attention, sliding | +0.26 s | 1.4–1.5× (owned by Q4-GEMMA-DECODE lever A) |
| cold prefill attention | +0.024 s | 1.7–2.05× |

**Why warm prefill is slow.** The warm route runs `flash_attn_vec_hybrid_batched`, the decode kernel, once per query row. Every (query, head) threadgroup streams the whole K/V history by itself.

**What the peer does instead.** For every warm and cold lane here the peer runs `kernel_flash_attn_ext_tensor` (`ggml-metal/kernels/fa_tensor.metal`).
- Routing: it takes that kernel whenever the batch is ≥ 20, `ceil(nb/Q)·heads·dk ≥ 8192` and `kv % 64 == 0`.
- Shape: 32 queries per threadgroup (16 at d512), 64 keys per step, 8 simdgroups.
- QKᵀ and PV run as Metal 4 `mpp::tensor_ops::matmul2d` (matrix units).
- Online softmax with lazy rescaling: O is rescaled only when a row's running max grows by more than 8. P is stored in half precision.
- A `blk` pre-pass marks fully masked 64-key tiles, which the kernel skips.

hf2q's own tiled F16 kernels spend at most ~95 µs per lane on preparation. Their kernel time alone is still 1.7× the peer's (sliding) and 2.9× (global).

## 2. Change

### mlx-native

New `flash_attn_prefill_tensor` (shader + dispatcher), a port of `kernel_flash_attn_ext_tensor` for `T = half`:
- **Instantiations:** DK = DV = 256 with Q = 32, and DK = DV = 512 with Q = 16. C = 64 and NSG = 8 for both.
- **Inputs:**
  - Q: F32 `[rows, n_heads, hd]`, read with row and head strides, exactly hf2q's layer Q layout. Q is converted to half in threadgroup memory with the scale folded in, as in the peer kernel.
  - K and V: F16 `[n_kv_heads, kv_pad, hd]` with row and head strides, GQA by `head / (n_heads / n_kv_heads)`.
  - mask: F16 `[rows, kv_pad]` additive, shared by all heads.
  - blk: `[ceil(rows/Q), kv_pad/64]` from mlx-native's existing `flash_attn_prefill_blk_f16` with `(bq, bk) = (Q, 64)`. Encoding: 0 skip, 1 mixed, 2 all-attended.
- **Output:** F32 `[rows, n_heads, hd]`, hf2q's layer output layout. No Q/O permute or cast is needed.
- **Host contract:** `kv_pad % 64 == 0`. Every K/V row below `kv_pad` must be finite; pad rows are zero and masked `-inf`. The dispatcher refuses a call it cannot satisfy.
- **Availability:** the kernel exists only where the device has the tensor API (`MlxDevice::has_tensor`), the same gate the tensor matmuls use. Elsewhere the dispatcher returns an error, and hf2q does not offer the route.

### hf2q

New `GemmaLiveKernel::TensorF16`, selected by `HF2Q_GEMMA_WARM_TILED_ATTENTION=tensor`:
- **Warm chain per sequence and chunk:**
  - K compaction and V dequantize into F16 scratch of `kv_pad` rows. These kernels exist from the `TiledF16` chain. The tail is zero-filled.
  - The serving mask (causal, sliding window, padding `-inf`) and blk.
  - Tensor attention writing the layer output rows in place.
  - The FWHT sign undo stays layer-level, as today.
- **Chunks:** sliding 1,024 queries, global 4,096, the `TiledF16` values. Suffixes under 32 rows keep vec-hybrid, as for `TiledF16`.
- **Cold:** the sliding F16 FA and the global tensor-mm routes are replaced by the same kernel on the per-lane head-major operands, behind the same selector. This is step 2, after warm.
- **Observability:** `wire_name` `TensorF16` in `active_controls.gemma_warm_attention_kernel`, plus the banner, launcher and Q5 bindings. Defaults stay off until the gates in §4 pass.

## 3. Consistency (operator requirement: no timing- or width-dependent numerics)

The result for a row must be a function of that row's sequence only. It must not depend on cohort composition, the number of query rows in the dispatch, the row's position in a query tile, or how the suffix is chunked.
- **Per-sequence dispatch.** The joint cohort still issues one dispatch per sequence and chunk, as both existing routes do. That makes joint ≡ per-slot hold by construction.
- **Row independence inside the kernel.**
  - Each row's max, sum and rescale are row-local.
  - A row that does not move its max is multiplied by exactly 1.0.
  - Masked cells contribute `exp(-inf) = 0` times a finite V, i.e. +0.0.
  - A tile skipped by blk is fully masked for every row of the query tile, so skipping it equals adding zeros.
  - Assumption (it is a test): matmul2d computes each output element in an order that does not depend on the other rows of its tile.
- **Key tiles aligned to absolute positions.**
  - Global keys start at absolute position 0 (graft rows first), so they are aligned by construction.
  - The sliding stage starts at an absolute position that is a multiple of 64, at or below the first visible key. Rows before the window are written as zeros and masked.
  - A row's 64-key tile boundaries therefore do not depend on its chunk start or the stage history length.
- **Class A tests in mlx-native**, bitwise on both metallib paths:
  - **width invariance:** a row's output is identical when computed in dispatches of 1, 7, 32, 58 and 311 rows at every tile offset;
  - **chunking invariance:** a suffix split into 1, 2 or 3 chunks gives identical rows;
  - **determinism:** repeated runs are identical.

## 4. Correctness gates (root decision 2026-11-06)

**Class B**, per the gap program's numerics rule:
1. **F64 reference.** A CPU F64 attention over the same F16-rounded Q/K/V. The error bound scales with key count and the half-precision P: max |Δ| ≤ c · 2⁻¹¹ · max|V| per output element, with c fixed in the test and the same for every shape. Shapes: Gemma warm lanes and cold rows (256, 458, 1,024), sliding and global.
2. **Determinism and joint ≡ per-slot (Class A, must hold).** The §3 kernel tests, plus the hf2q real-artifact gates:
   - `gemma_warm_tiled_attention_is_deterministic` on `TensorF16`;
   - `gemma_joint_warm_prefill_matches_per_slot_bitwise` on `TensorF16`.
3. **Q1 policy envelope on T1/T3 greedy144** (`q1_decision_policy.py`, τ = 0.25 nat), against the shipped route.

**Not a gate: the every-row E1 envelope against `VecHybrid`.** Any route that is not bitwise equal to `VecHybrid` fails it by construction. `TiledF32` failed it with a layer-0 attention error of 4.95e-7 (ADR-063 Q4-GEMMA-ATTN outcome). E1 stays a recorded diagnostic.

**Speed evidence (gap program):** the microbench at the warm and cold shapes must show hf2q ≤ peer, or the remaining gap explained. That comes before any timed four-agent run, and root decides when a speed test is justified.

## 5. Expected effect `[INFERENCE]`

Warm attention would fall from ~1.43 s to about the peer's ~0.21 s plus the chain preparation. That is roughly **1.0–1.2 s of GPU time per run**. The cold share is small (≤ 0.02 s).

## 6. Superseded work

Q4-GEMMA-ATTN-EXACT-G2 (#706) is parked (root, 2026-11-06), because F1 replaces the warm route it accelerated. Its sliding-only gain was 0.26 s of chain per run.
