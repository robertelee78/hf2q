# Batched-prefill race — root-cause identified

**Date:** 2026-04-16
**Investigation:** 2-agent swarm (llama.cpp researcher + candle researcher), independent convergence
**Report inputs:** `/tmp/swarm-report-llamacpp.md`, `/tmp/swarm-report-candle.md`
**Related:** ADR-005 Gate A (Task #7 / Task #9), `docs/spike-gate-a-prefill.md`, user memory `project_inference_bugs_session2.md`

## Headline

**The race is a write-after-read hazard on threadgroup memory in mlx-native's `fused_head_norm_rope_f32` kernel** at `/opt/mlx-native/src/shaders/fused_head_norm_rope_f32.metal:95↔105`. The bf16 sibling `fused_head_norm_rope_bf16.metal` has the identical hazard at its lines 78↔101.

No barrier separates Phase-1's broadcast-read of `shared[0]` (for `rms_inv`) from Phase-2's first write to `shared[0]` (tid=0, i=0). On Apple Silicon, simdgroups within one threadgroup are not in lockstep, so simdgroups 1–7 can read `shared[0]` **after** simdgroup 0 has overwritten it with a normalized value, producing a garbage `rms_inv` for those threads and thus corrupt Q/K projections.

Both reference implementations (llama.cpp and candle) are structurally immune — neither reuses the reduction scratch for phase-2 output, and neither fuses norm into rope. Candle uses a dedicated `threadgroup float &total` scalar for the broadcast; llama.cpp writes phase-2 results directly to device memory.

## The exact hazard

Source: `/opt/mlx-native/src/shaders/fused_head_norm_rope_f32.metal`

```metal
88:  for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
89:      if (tid < stride) { shared[tid] += shared[tid + stride]; }
92:      threadgroup_barrier(mem_flags::mem_threadgroup);
93:  }
94:
95:  const float rms_inv = rsqrt(shared[0] / float(head_dim) + eps);   // ALL threads read shared[0]
96:
97:  // ... no barrier here ...
100: for (uint i = tid; i < head_dim; i += tg_size) {
101:     float val = input[base + i] * rms_inv;
102:     if (has_weight) { val *= norm_weight[i]; }
105:     shared[i] = val;                                               // tid=0 writes shared[0]
106: }
107: threadgroup_barrier(mem_flags::mem_threadgroup);
```

After the stride=1 barrier at `:92`, every thread's view of `shared[0]` is correct (the reduced sum-of-squares). Each thread then reads `shared[0]` at `:95` to compute its own register-resident `rms_inv`. There is **no barrier** between `:95` (load) and `:105` (store). Thread `tid=0` proceeds to Phase-2 and overwrites `shared[0] = val` in the first loop iteration. Threads in simdgroups that have not yet reached `:95` then read the post-Phase-2 value and compute a corrupted `rms_inv`.

## Why every observed symptom maps to this hazard

| Symptom | Explanation |
|---|---|
| ~80% broken at seq_len=576, head_dim=256 | tg_size = `min(256, head_dim.next_power_of_two())` = 256 → 8 simdgroups per TG → maximum skew surface. Race outcome depends on runtime scheduling. |
| Decode (seq_len=1) always deterministic | 4–8 threadgroups total → negligible GPU scheduler pressure → simdgroups effectively lockstep → race window never opens. |
| Prefill at seq_len=576 triggers heavy race | 2304 TGs (K, nkv=4) / 4608 TGs (Q, nh=8) → saturated scheduler → wide inter-simdgroup skew → race fires reliably. |
| Metal API validation shifts rate but doesn't fix | Validation injects timing perturbations that change simdgroup scheduling; they don't serialize simdgroups within a TG. |
| Forcing `memory_barrier()` on every `barrier_between` doesn't fix | The race is INSIDE one kernel launch, between simdgroups in one TG. Inter-dispatch fences cannot order intra-TG execution. |
| Layer 8 is first divergent | Cumulative Q/K corruption across layers; threshold is model-data-dependent. The race fires at every layer; layer 8 is where accumulated drift exceeds the argmax tolerance on this prompt. |
| Sliding layers specifically | Sliding layers have head_dim=256 → tg_size=256 → 8 simdgroups. Global layers have head_dim=512 → same tg_size=256 → same race surface; but the race fires at every layer so this observation is just about when cumulative drift crosses the threshold. |

## Side-by-side: ours vs the references

| Aspect | mlx-native (racy) | llama.cpp (immune) | candle (immune) |
|---|---|---|---|
| Norm + RoPE | Fused into one kernel | Two separate dispatches | Two separate dispatches |
| RMS-norm Phase-2 output destination | Shared memory (`shared[i] = val`) — reused across phases | Device memory (`y[i00]`) — reference: `ggml-metal.metal:3038–3048` | Device memory (`dst[i]`) — reference: `reduce.metal:1103–1113` |
| Broadcast of reduced scalar | Re-read from `shared[0]` by every thread | `simd_sum` across simdgroups, register-only — reference: `ggml-metal.metal:3021,3032` | Dedicated `threadgroup float &total` written by tid==0 under a full barrier — reference: `reduce.metal:1066, 1094–1099` |
| Barrier between Phase-1 read and Phase-2 write | **MISSING** | N/A (Phase-2 doesn't touch shared) | Present (`reduce.metal:1099`) |
| Shared-memory reuse for phase-3 RoPE reads | Yes (the motivation for fusion) | N/A (RoPE is a separate kernel, reads device memory) — reference: `ggml-metal.metal:4385–4435` (zero shared mem) | N/A (RoPE is a separate kernel) — reference: `reduce.metal:1390–1420` |

**Reference commits in candle worth citing as precedent:**
- `db3d5d98` ([Metal] improve normalization, #3283) — introduced the dedicated `threadgroup float &total` scalar for broadcast. Explicit historical evidence that candle *replaced* a pattern similar to mlx-native's with the safer one.
- `46928bce` (Fix sliding window full sdpa corner case, #3438) — shows candle actively hardening SDPA for Gemma-4-style ISWA at shapes equivalent to ours.

## SDPA and permute_021 cleared

Both researchers ruled out tiled SDPA and the permute kernel independently:

- **`sdpa.metal` / `sdpa_sliding.metal`** — zero threadgroup memory, zero barriers, zero cross-thread shared state. Each thread owns a private `acc[512]` stack array and writes its own `O[q_pos]` slot. No internal race possible; if SDPA output is corrupted, the corruption came from corrupted Q/K inputs upstream (which is exactly what the fused-kernel race produces).
- **`permute_021_f32`** — single thread per element, simple copy (`elementwise.metal:413`). No reduction, no scratch, no synchronization requirement.

## Proposed fix

### Minimal patch (single barrier)

Insert one `threadgroup_barrier(mem_flags::mem_threadgroup)` between the `rms_inv` broadcast-read and the Phase-2 loop, in both f32 and bf16 variants.

```diff
--- a/src/shaders/fused_head_norm_rope_f32.metal
+++ b/src/shaders/fused_head_norm_rope_f32.metal
@@ -92,6 +92,13 @@
         threadgroup_barrier(mem_flags::mem_threadgroup);
     }

     const float rms_inv = rsqrt(shared[0] / float(head_dim) + eps);

+    // All threads must finish reading shared[0] (broadcast of the reduced
+    // sum-of-squares) BEFORE any thread (notably tid==0) can overwrite
+    // shared[0] in Phase 2. Without this, simdgroups that race ahead of
+    // simdgroup 0 read a clobbered shared[0] and compute a corrupt rms_inv.
+    // Matches candle's pattern at reduce.metal:1099 and llama.cpp's
+    // architectural choice of writing Phase-2 to device memory.
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+
     // -----------------------------------------------------------------
     // Phase 2: normalize (optionally scale with weight), store in shared
     // -----------------------------------------------------------------
     for (uint i = tid; i < head_dim; i += tg_size) {
```

Apply the identical insertion in `/opt/mlx-native/src/shaders/fused_head_norm_rope_bf16.metal` between its line 78 (read of `shared[0]`) and line 101 (write to `shared[i]`).

**Semantic impact:** zero — the kernel's arithmetic is unchanged.
**Perf impact:** one extra threadgroup barrier per head per token — ~9000 extra barriers per 576-token prefill, each a few ns on Apple Silicon. Negligible.

### Stronger refactor (recommended follow-up)

Promote `rms_inv` to a dedicated threadgroup scalar, matching candle's `total` pattern one-for-one. This makes the race impossible by construction even if a future edit changes `tg_size` relative to `head_dim`.

```metal
// In the threadgroup memory declarations (same scope as `shared`):
threadgroup float rms_inv_shared;

// Replacing current line 95:
if (tid == 0) {
    rms_inv_shared = rsqrt(shared[0] / float(head_dim) + eps);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float rms_inv = rms_inv_shared;
```

Reference: `/opt/candle/candle-metal-kernels/src/metal_src/reduce.metal:1094–1099`.

Keep the minimal patch for the immediate hotfix; land the refactor as a follow-up to match the reference pattern exactly.

## Verification plan

1. Apply the minimal patch to both f32 and bf16 shaders in `/opt/mlx-native`.
2. Rebuild hf2q. Run `/opt/hf2q/target/release/hf2q generate` with `HF2Q_BATCHED_PREFILL=1 HF2Q_UNSAFE_EXPERIMENTS=1` on the 576-token fixture at T=0 `max-tokens=1` across ≥20 runs.
3. Assert every run produces token `2021` (deterministic).
4. Re-run at seq_len=14 and seq_len=2455 (will bail on seq_len=2455 per Gate A guard until sdpa_sliding is addressed separately — Task #8).
5. Sourdough gate: `scripts/release-check.sh` at HEAD with batched prefill on, confirm stable passes.

## Convergence note

Both swarm agents (one studying `/opt/llama.cpp`, one studying `/opt/candle`) independently identified the **exact same lines** (`fused_head_norm_rope_f32.metal:95↔105` and the matching bf16 lines) as the hazard, with the **same proposed fix**. That's two independent reference-implementation comparisons converging on the same micro-location. High confidence.

## Credit and references

- Full llama.cpp comparison: `/tmp/swarm-report-llamacpp.md`
- Full candle comparison: `/tmp/swarm-report-candle.md`
- Suspect kernel: `/opt/mlx-native/src/shaders/fused_head_norm_rope_f32.metal:95–107` (f32) and `/opt/mlx-native/src/shaders/fused_head_norm_rope_bf16.metal:78–101` (bf16)
- Dispatch: `/opt/mlx-native/src/ops/fused_head_norm_rope.rs:261` (shared_slots sizing — past threadgroup-scratch fix; the one-liner fix here is the natural next sibling to that earlier fix)

## Status

- **Root cause: identified, convergent evidence from two independent reference comparisons.**
- **Proposed fix: ready to apply (one-line barrier in each of two shaders).**
- **No code changes landed yet.** Waiting for your review before patching.
