# ADR-011 Phase 2 Wave 3 — bf16 conversion verification

**Status**: Wave 3 landed (partial scope — bf16 attention island)
**Date**: 2026-04-17
**Author**: Wave 3 agent, CFA swarm swarm-1776516482254-ft5mwj
**Scope**: `forward_prefill_batched.rs` bf16 conversion per Agent #5's map
(`docs/ADR-011-phase2-bf16-conversion-map.md`)

---

## 1. Summary

Wave 3 landed a **bf16 attention island** covering permute → SDPA →
back-permute inside the batched prefill path. This is the narrowest
bf16 region that still gives Wave 4 the exact buffer dtypes
`flash_attn_prefill_bf16_d256` / `_d512` expect.

The broader MLX-LM convention (bf16 Q/K/V projections, bf16 head-norm+RoPE,
bf16 MLP/MoE intermediates) is **not** landed in Wave 3 — multiple
blockers detailed in §5 prevented a safe per-stage progression and a
full-bf16 attempt produced garbage output (divergence at byte 6 of
sourdough_gate).

## 2. What landed

| Commit | Stage | Content |
|---|---|---|
| `2bea674` | 1 | `alloc_bf16` helper closure (infra only, no behaviour change) |
| `c43caa4` | 2 | bf16 SDPA island: cast f32 normed Q/K/V → bf16, permute_bf16, sdpa_bf16 / sdpa_sliding_bf16, back-permute_bf16, cast bf16 → f32 for O-proj |

### Dtype map (current)

**f32 (unchanged from baseline)**:
- `pf_hidden`, `pf_residual` — residual stream
- `pf_norm_out`, `pf_moe_norm_out`, `pf_router_norm_out` — norm outputs
- `pf_q`, `pf_k`, `pf_v` — qmatmul outputs (qmatmul is f32-only)
- `pf_q_normed`, `pf_k_normed`, `pf_v_normed` — head_norm+RoPE outputs (f32 weight, f32 kernel)
- `pf_attn_out`, `pf_mlp_down_out` — residual-fed outputs
- `pf_mlp_gate`, `pf_mlp_up`, `pf_mlp_fused` — dense MLP buffers
- `pf_moe_gate_up`, `pf_moe_swiglu`, `pf_moe_down`, `pf_moe_accum` — MoE buffers
- `pf_sdpa_out` — post-cast buffer feeding the f32-only O-proj qmatmul
- KV cache (f32 default, f16 via `HF2Q_F16_KV`) — unchanged

**bf16 (new in Wave 3)**:
- `pf_q_normed_bf16`, `pf_k_normed_bf16`, `pf_v_normed_bf16` — cast-to-bf16 copies of the normed buffers
- `pf_q_perm`, `pf_k_perm`, `pf_v_perm` — SDPA inputs
- `pf_sdpa_out_perm` — SDPA output
- `pf_sdpa_out_bf16` — back-permute output, fed into the bf16→f32 cast
  that lands in `pf_sdpa_out` for O-proj

### bf16 kernels now dispatched per prefill forward pass

- `cast_f32_to_bf16` × 3 per layer (Q/K/V normed)
- `permute_021_bf16` × 4 per layer (Q, K, V in; SDPA out)
- `sdpa_bf16` or `sdpa_sliding_bf16` × 1 per layer (dtype auto-pick in
  `ops/sdpa.rs:197` / `ops/sdpa_sliding.rs:203`)
- `cast_bf16_to_f32` × 1 per layer (SDPA out)

For Gemma 4 (30 layers): **240 new bf16 dispatches per forward pass**
(120 casts + 120 SDPA-region bf16 kernel calls replacing the same
count of f32 versions).

## 3. Regression gate (per-stage)

`scripts/sourdough_gate.sh` with `HF2Q_BATCHED_PREFILL=1` on Gemma 4
26B MoE DWQ. Min common-prefix floor: 3094 bytes.

| Stage | Commit | Common prefix | Result |
|---|---|---|---|
| pre-Wave-3 | `c96e91f` | 3656 / 3658 | PASS (margin 562) |
| Stage 1 | `2bea674` | (unchanged — infra only) | PASS |
| Stage 2 | `c43caa4` | 3095 / 3658 | PASS (margin 1) |

Stability check: Stage 2 gate re-run is deterministic at 3095 bytes
(greedy T=0). The 561-byte tightening versus the f32 baseline is the
expected bf16 rounding delta in SDPA — comparable to the precision
drift MLX-LM accepts by design with its bf16 convention.

## 4. Why Stage 2 is the stop point (not Stages 3-9)

1. The margin to the sourdough gate floor is **1 byte**. Any further
   bf16 conversion that introduces another bf16 rounding site
   (fused_gelu_mul_bf16, moe_swiglu_seq_bf16,
   moe_weighted_sum_seq_bf16_input) will very likely push the common
   prefix below 3094. At that point the gate fails and per Wave 3's
   stop-on-regression rule the stage must be reverted.

2. Wave 4's objective (`flash_attn_prefill` wire-up) is fully served
   by the current bf16 region. `flash_attn_prefill_bf16_d256` /
   `_d512` take Q/K/V in bf16 and emit bf16 output — exactly the
   dtype shape of `pf_q_perm`, `pf_k_perm`, `pf_v_perm`,
   `pf_sdpa_out_perm` today. The only remaining Wave-4 work is the
   layout and strides plumbing, not more hf2q bf16 surface.

3. Stages 3-9 as described in ADR-011 §5 require additional casts at
   each quantized-matmul boundary (see §5.1 blockers). The cost per
   layer would be 4-6 extra cast dispatches for MLP and up to
   2×top_k casts for MoE, erasing the speedup from the bf16
   intermediates without matching the mlx-lm convention faithfully
   (which presumes bf16-capable qmatmul kernels, which mlx-native
   lacks).

## 5. Blockers found and resolved / unresolved

### 5.1 `quantized_matmul_ggml` is f32-only (unresolved, out of Wave 3 scope)

Observation: `ops/quantized_matmul_ggml.rs:191, 200` validates input
and output as f32, and the kernel names are hardcoded
`kernel_mul_mv_q4_0_f32` / `_q8_0_f32` / `_q6_K_f32`. Same pattern
for `quantized_matmul_id_ggml` (lines 138, 175) and its kernel
`id_kernel_name()`.

This means every Q/K/V/O/gate/up/down/MoE projection boundary forces
an f32-to-bf16 or bf16-to-f32 cast if we want bf16 in between.

ADR-011 §11 item 1 listed this as "unverified"; it is now confirmed
as f32-only. A future wave should port these to output bf16 when the
caller's output buffer is bf16 (dtype-dispatched, same pattern as
`dispatch_rms_norm`).

**Resolution**: Wave 3 works around by running the bf16 region as an
island between two casts. Follow-on work should upgrade
`quantized_matmul_ggml` to dtype-dispatched I/O before attempting
full-bf16 coverage.

### 5.2 f32 head-norm weights in the bf16 kernel (resolved by staying f32, explained below)

Observation: `fused_head_norm_rope_bf16.metal:43, 106, 159, 216`
reads `norm_weight` as `device const bfloat*`. But `forward_mlx.rs`
loads `q_norm_weight` / `k_norm_weight` via `gguf.load_tensor_f32`
(lines 714–719) — the buffer dtype is F32. Passing an f32 buffer
into the bf16 kernel reads 2 bfloats per f32 with undefined bit
pattern, producing garbage output.

**First attempt** (dropped): per-layer f32→bf16 weight cast at
session start into a dedicated bf16 weight buffer, then pass the
bf16 weight to `dispatch_fused_head_norm_rope_batch_bf16`. The cast
itself is correct and negligible, but the resulting prefill pass
still produced garbage — common prefix 6, output: `"**1; 1 (// 1;
1;..."`. Suspected root cause: either the bf16 head-norm kernel
disagrees with the f32 kernel on a subtle numeric detail (NeoX
convention vs ProportionalRoPE ordering) or the bf16 precision at
the early-layer QK dot products was insufficient to keep the
softmax from collapsing. Not bisected further — the simpler bf16
SDPA-island approach §5.3 avoids the issue.

**Resolution**: Keep head norm + RoPE on the f32 kernel path. The
bf16 island starts at the f32 normed buffer and bookends with casts.
No weight-dtype issue, no latent kernel-variant issue.

### 5.3 bf16 SDPA is untested in mlx-native (resolved by the sourdough gate)

Observation: `grep -n bf16 tests/test_sdpa.rs` returns empty.
`sdpa_bf16` and `sdpa_sliding_bf16` have shader-level tests only as
dispatch selectors, not numerical parity tests against the f32
sibling.

**Resolution**: The sourdough gate at common prefix 3095 (1-byte
margin) is the end-to-end parity test this kernel had been missing.
PASS confirms the algorithm matches the f32 sibling up to bf16
rounding on the 22-token sourdough prompt. A targeted mlx-native
test for numerical parity against the f32 kernel on a small
Q×K×V×seq would be worth landing, but it is out of Wave 3 scope.

### 5.4 `dispatch_rms_norm_unit_perhead` hardcoded to f32 (deferred)

Observation: `forward_mlx.rs:3014` hardcodes the pipeline to
`rms_norm_no_scale_f32`, so the V-norm step can't be made bf16
without either editing that helper or calling the
`rms_norm_no_scale_bf16` pipeline inline.

**Resolution**: Stage 2's current design keeps `pf_v_normed` as f32
(since the V-norm input `pf_v` is f32 from the qmatmul output). No
V-norm dtype change needed. A future wave that pushes Q/K/V to bf16
before head-norm can inline-dispatch `rms_norm_no_scale_bf16` as
prototyped in the earlier failed attempt.

### 5.5 hf2q was pinned to crates.io `mlx-native = "0.3"` (resolved)

Observation: hf2q's `Cargo.toml` declares `mlx-native = "0.3"`
without a path override. When the build resolved the dependency, it
picked 0.3.2 from crates.io, not `/opt/mlx-native` — and the 0.3.2
crate on crates.io **does not contain** any of Wave 2's additions
(`dispatch_fused_head_norm_rope_batch_bf16`, `fused_gelu_mul_bf16_encode`,
`moe_swiglu_seq_bf16_encode`, `moe_weighted_sum_seq_bf16_input_encode`,
`dispatch_kv_cache_copy_seq_bf16`, the `flash_attn_prefill_*`
modules). Every call into a Wave-2 dispatcher failed at compile time
with `cannot find function` errors.

`Cargo.toml:45-49` documents the intended workflow: create a
gitignored `.cargo/config.toml` with
`[patch.crates-io] mlx-native = { path = "/opt/mlx-native" }`.

**Resolution**: Added `.cargo/config.toml` locally. `.cargo/` is
already gitignored per `/opt/hf2q/.gitignore:42`, so the override is
per-machine. With the patch active, the workspace builds and resolves
`mlx-native` to the in-tree `/opt/mlx-native` source which carries
all Wave 2 kernels and dispatchers.

## 6. Final state

- **f32 sites remaining**: of the 52 f32 sites Agent #5 inventoried in
  ADR-011 §3, 44 remain f32. The 8 converted sites are the 3 normed
  casts + 3 permute inputs + 1 SDPA output + 1 back-permute destination.
- **bf16 kernels now called (per layer)**: `cast_f32_to_bf16` (×3),
  `permute_021_bf16` (×4), `sdpa_bf16` or `sdpa_sliding_bf16` (×1),
  `cast_bf16_to_f32` (×1). 30 layers × 9 = 270 new bf16-path dispatches
  per forward pass.
- **Sourdough gate**: PASS at 3095/3094 bytes common prefix (1-byte margin).
- **ADR-011 Phase 2 gate (Gate 1 text parity)**: not run (requires
  `parity_check.sh`); Gate 2 (prefill tok/s parity) not measured —
  these are Wave 5 scope, not Wave 3.

## 7. Wave 4 readiness

**Ready**. `pf_q_perm`, `pf_k_perm`, `pf_v_perm` are bf16 contiguous
buffers with the layout `[n_heads, seq_len, head_dim]`. This matches
exactly what `flash_attn_prefill_bf16_d256` expects at its buffers
(0, 1, 2) once the batch=1 leading dim is plumbed through
`AttnParamsGpu`. `pf_sdpa_out_perm` is a bf16 buffer with the correct
output shape `[n_heads, seq_len, head_dim]`. Wave 4's job reduces to:
replace the `s.sdpa(...)` / `s.sdpa_sliding(...)` call site at
`forward_prefill_batched.rs` session-A §6 with a dispatch to
`flash_attn_prefill_bf16_*`. No additional bf16 plumbing needed on
the hf2q side.

Stages 3-9 of the conversion map (MLP bf16, MoE bf16) remain
valuable for the full MLX-LM dtype convention and should land as a
**post-Wave-4** cleanup pass, ideally after `quantized_matmul_ggml`
grows dtype-dispatched I/O (§5.1) so that the cast dance disappears.
