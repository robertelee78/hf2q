# ADR-013 P21 — Comprehensive Arena Refactor (formal /cfa spec)

**Status:** SPEC for /cfa dual-mode (Claude+Codex parallel implementations, queen-judged)
**Date:** 2026-05-01
**Owners:** Robert (gate) · Claude+Codex (impl) · queen-coordinator (judge)
**Predecessor:** P20 K-batched FFN terminal (commit `bc93697`) — landed but wall-impact null
**Cross-ref:** ADR-013 §17 (match-or-beat ship gate), P19 H9 measurement (commit `270eaae`), P20 honest null-result (commit `bc93697`)

---

## Why this spec exists

P19 H9 measured that hf2q does **161 commit_and_wait per Qwen3.6 35B-A3B prefill** at a fixed cost of ~1.32 ms/commit average → **213 ms of pure CB-sync overhead**. P20 confirmed structurally that K-batching the per-layer FFN terminal saves 30 commits but produces **no measurable wall improvement** because those FFN-terminal commits are the cheap ones (GPU is already idle when they fire).

The expensive 130+ commits live INSIDE the kernel-wrapper functions:

- `apply_gated_delta_net_chunk` (`/opt/hf2q/src/inference/models/qwen35/gpu_delta_net.rs:1214`) — DANGEROUS class per iter58b regression doc-comment
- `apply_flash_attn_prefill_seq_major` (`gpu_full_attn.rs:1158`)
- DN ops5-9 terminal (`gpu_delta_net.rs:1808` / `:1862`)
- DN ops1-3 terminal (`gpu_delta_net.rs:1574`)
- FA ops1-4 terminal (`gpu_full_attn.rs:1550`)
- FA ops6-7 terminal (`gpu_full_attn.rs:1710`)

Each of these wrappers allocates internal scratch (`q_expanded`, `k_expanded`, BF16 staging, etc.) inside the function body. The terminal `commit_and_wait` is required because the scratches drop at function return; without the wait, dropped scratches stage `removeAllocation:` on the residency set, which flushes at the NEXT encoder's `commit*` boundary, demoting pages mid-flight on still-running CBs and producing garbage tensor values (the iter58b mechanism, exhaustively documented in `solution_mlx_native_residency_lifetime_race.md`).

To remove these waits safely, the wrapper-internal scratches must be lifted to **caller-owned arenas** whose lifetimes span the entire prefill. Once scratch outlives the CB, the terminal `commit_and_wait` becomes optional — replace with `commit()` and let the next encoder's submission carry GPU-side ordering on the Metal serial queue.

**Hardware/model bound:** the K=40 single-super-CB target is INFEASIBLE per W-5b.14 — dense-Q FFN scratches accumulate ~1 GB/layer × 33 dense layers ≈ 33 GB, overrunning the M5 Max 115 GB recommendedMaxWorkingSetSize. The realistic target on Qwen3.6 35B-A3B (MoE, 40 layers, smaller per-expert scratch) is K=4 to K=8 windows; on Qwen3.6 27B (dense) the bound is tighter and may require K=2 or K=4.

---

## Goal

Reduce `mlx_native::sync_count()` per Qwen3.6 35B-A3B Q4_K dwq48 prefill from **161 to ≤ 25** without coherence regression, while keeping decode tok/s ≥ current and total memory ≤ 115 GB working set.

| Phase | sync_count target | wall savings target |
|---|---|---|
| Current (post-P20, K=1 default) | 161 | baseline |
| P21 stage 1 (FA wrapper arenas) | ≤ 130 | ≥ 30 ms at pp101 |
| P21 stage 2 (DN wrapper arenas) | ≤ 70 | ≥ 60 ms at pp101 |
| P21 stage 3 (chunk_gdn caller-owned scratch) | ≤ 30 | ≥ 100 ms at pp101 |
| P21 stage 4 (final K=4 window with deferred FFN reset) | ≤ 25 | ≥ 130 ms at pp101 |

Stretch (no longer a wall savings target — already structurally near ceiling): `sync_count` ≤ 10 requires either (a) lifting the FA `flash_attn_prefill` bridge's terminal commit (one of the highest-cost ones) which means lifting its internal `out_bf16_hm` staging buffer, or (b) accepting that some kernel-internal commits are unavoidable on Apple Silicon.

---

## Non-goals

- **Not** a kernel rewrite. The Q4_K kernel, the flash_attn_prefill kernel, and the gated_delta_net kernel are byte-equivalent to llama.cpp's per P18 audit. Touch zero `.metal` files; touch zero kernel template instantiations.
- **Not** a graph-builder rewrite. hf2q's `forward_gpu_impl` continues as an imperative per-layer loop. We are NOT building a ggml-style compute DAG.
- **Not** a decode-path refactor. Decode (`seq_len == 1`) is already at parity (1.05× faster than llama.cpp). Don't touch `forward_gpu_greedy` beyond what's needed to keep the partial-chain mechanism (`HF2Q_PARTIAL_CHAIN_N`) compatible with the new arena.

---

## Acceptance criteria (queen-judged)

All must pass on `/opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf` on the M5 Max:

1. **`sync_count` reduction.** With `HF2Q_PROFILE_SYNC=1` on a pp101 prefill, `sync_count ≤ 25` (stretch ≤ 10).
2. **Wall improvement.** 3-rep cold P17b harness (`scripts/p17b_q4k_bench.sh`) median pp101 prefill_tok_s ≥ 350 t/s (vs baseline 247.50). Stretch ≥ 500 t/s.
3. **Coherence.** End-of-bench coherence prompt produces `ascii_ratio ≥ 0.85`, `<think>` reasoning blocks land correctly, no degenerate-loop output.
4. **Sourdough byte-prefix gate.** `scripts/sourdough_qwen35.sh` PASSES with common prefix ≥ 160 bytes (the P13.1 floor).
5. **Unit-test invariants.** `cargo test --release --bin hf2q inference::models::qwen35` 0 failed (currently 172 passed).
6. **Decode parity.** 3-rep cold tg64 median ≥ 120 t/s (vs current ~124 t/s, allow 3% drift budget).
7. **Memory budget.** Peak Metal heap usage at pp4096 ≤ 30 GB (current ~21 GB load + scratch). Verify via `vm_stat` delta or `/proc/<pid>/io` if available; conservatively run `host_vm_info` snapshot pre/post.
8. **No iter58b regression flicker.** `cargo test --release qwen35::gpu_delta_net::tests::chunk_path_first_token_matches_autoregressive_at_seq128` PASSES on 5 consecutive cold runs (Heisenbug guard).
9. **Build cleanliness.** `cargo build --release --bin hf2q` 0 warnings introduced. `cargo clippy --release` doesn't add new warnings under the existing allow-list.

Queen scoring: each criterion 0–10. Hard fail if any < 6. Promote winner: max total. Tie → reviewer (Codex if Claude won, Claude if Codex won) cross-reviews and queen reconciles.

---

## Method (4 stages, each with its own queen gate)

### Stage 1 — FA wrapper arenas (target: 161 → ≤ 130)

**Scope:** `gpu_full_attn.rs::build_gated_attn_layer` + `apply_flash_attn_prefill_seq_major`.

**Refactor.** Both wrappers currently allocate scratch internally (`q_bf16`, `k_bf16`, `v_bf16`, `out_bf16_hm`, `q_normed`, `k_normed`, `q_rope`, `k_rope`, etc.). Lift these to a new `FaPrefillArena` struct owned by the caller (`forward_gpu_impl`), allocated ONCE at prefill start and passed through. The wrapper functions take `&mut FaPrefillArena` and reuse its buffers across all 10 FA layers.

After lifting, the three FA terminal `commit_and_wait` calls (`:1158`, `:1550`, `:1710`) can become `commit()` because the scratches no longer drop at function return — they live inside the caller-owned arena until end of prefill.

**sync_count delta:** −30 (10 FA layers × 3 commits each).

**Risk class:** MEDIUM. The FA path has fewer branches than DN; the legacy SDPA path (`:1381`) is already gated off in production (head_dim=256 + cur_len=0 → flash_attn route). Arena needs to be sized for max-prefill-tokens; size = `seq_len * n_heads * head_dim * sizeof(BF16) * 6 buffers ≈ 50 MB at pp4096`.

### Stage 2 — DN wrapper arenas (target: ≤ 70)

**Scope:** `gpu_delta_net.rs::build_delta_net_layer` (the 30 DN-layer body) + `apply_ssm_conv` + the ops5-9 dispatch.

**Refactor.** Same pattern as Stage 1: introduce `DnPrefillArena` for the per-layer scratches (`x_norm`, `qkv_split`, `q_normed`, `k_normed`, conv state ping-pong, etc.). The two DN terminal commits (`:1574`, `:1808`/`:1862`) become `commit()`.

The 30 `apply_gated_delta_net_chunk` commits at `:1214` remain UNCHANGED in this stage. They are the iter58b-DANGEROUS class — `chunk_gated_delta_rule.rs:399-543` allocates ~16 internal scratches that cannot be lifted without changing the kernel-side allocation contract.

**sync_count delta:** −60 (30 DN layers × 2 commits each).

**Risk class:** MEDIUM-HIGH. DN has heterogeneous code paths (chunked vs autoregressive vs decode). Arena must be sized for max-prefill-tokens AND outlive the chunk dispatch. Memory: per-layer ~10 MB at pp4096 → 300 MB sustained for 30 DN layers' worth of scratch in arena.

### Stage 3 — chunk_gdn caller-owned scratch (target: ≤ 30)

**Scope:** `mlx_native::ops::chunk_gated_delta_rule.rs:399-543` and its 16 internal scratches.

**Refactor.** This is the iter58b mine. The chunk kernel allocates via `MlxBuffer::new` inside the function and the buffers drop at return. iter58b's restored `commit_and_wait` is the load-bearing safety belt.

To safely remove the wait, refactor the chunk function signature: `apply_gated_delta_net_chunk_with_arena(arena: &mut ChunkArena, ...)` where `ChunkArena` holds the 16 scratches sized for the largest in-flight chunk. The caller (`forward_gpu_impl`) owns the arena; lifetime = entire prefill.

After refactor, the `:1214` commit_and_wait becomes `commit()`.

**sync_count delta:** −30 (30 DN layers × 1 commit each).

**Risk class:** HIGH. This is exactly the surface iter58b regressed on. Required mitigations:
- Run `chunk_path_first_token_matches_autoregressive_at_seq128` 5× cold to verify the Heisenbug doesn't return
- Run sourdough on apex Q4_0 MoE GGUF to verify the chunk-prefill regression doesn't re-emerge
- Add a unit test that drops the arena BEFORE the next encoder commits and verifies no removeAllocation: staging fires (mock the Metal residency set if possible)

### Stage 4 — K=4 window with deferred FFN reset (target: ≤ 25)

**Scope:** `forward_gpu.rs::forward_gpu_impl` layer loop (the K-batch infrastructure already added by P20 commit `bc93697`).

**Refactor.** Promote `HF2Q_FFN_TERMINAL_K_BATCH=4` from opt-in to default, gated only on the residency-quota fitting at pp4096. With Stages 1-3 complete, the per-layer FFN terminal becomes the primary remaining bottleneck. K=4 saves 30 of 40 FFN terminal commits.

**sync_count delta:** −30 (post-Stage-3 baseline 70 → 40, then K=4 saves another 30 → 10. Plus the unavoidable output-head terminal = 11. Round to ≤ 25 stretch.)

**Risk class:** LOW. Infrastructure already shipped in P20. Just flip the default after verifying memory + coherence at K=4 with the new Stage 1-3 arenas in place.

---

## Operator runbook (post-merge, before flipping defaults)

```bash
# 1. Verify mechanical reduction
HF2Q_PROFILE_SYNC=1 ./target/release/hf2q generate \
  --model /opt/hf2q/models/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48/qwen3.6-35b-a3b-abliterix-ega-abliterated-dwq48.gguf \
  --prompt 'Test' --max-tokens 8 --temperature 0
# Expect: [P19 H9] ... sync_count <= 25

# 2. Coherence + sourdough
./scripts/sourdough_qwen35.sh /opt/hf2q/models/.../apex.gguf
# Expect: PASS, common prefix >= 160 bytes

# 3. 3-rep cold bench
./scripts/p17b_q4k_bench.sh \
  /opt/hf2q/models/.../dwq48.gguf --reps 3 --pp 31,101 --tg 64
# Expect: pp101 prefill_tok_s median >= 350

# 4. iter58b Heisenbug guard
for i in 1 2 3 4 5; do
  cargo test --release --bin hf2q -- \
    inference::models::qwen35::gpu_delta_net::tests::chunk_path_first_token_matches_autoregressive_at_seq128 \
    --test-threads=1 --nocapture
done
# Expect: 5/5 PASS, no flicker

# 5. Memory peak guard
/usr/bin/time -l ./target/release/hf2q generate ... 2>&1 | grep "maximum resident set size"
# Expect: <= 30 GB
```

If ANY check fails → revert. If ALL pass → promote default `HF2Q_FFN_TERMINAL_K_BATCH=4`, close P21.

---

## Risks and mitigations

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| iter58b regression returns | medium | critical | Stage 3 has explicit Heisenbug guard (5× cold); Stage 1-2 don't touch chunk_gdn |
| Memory budget exceeded at pp4096 | low-medium | high | arena sizing computed up-front; fail-fast on first too-small arena |
| Coherence ascii_ratio < 0.85 | low | critical | sourdough gate runs as part of acceptance; immediate revert |
| Decode regression | low | high | decode path untouched; existing partial-chain (`HF2Q_PARTIAL_CHAIN_N`) keeps working |
| Build introduces warnings | low | low | `cargo clippy --release` in acceptance |

---

## Why dual-mode (Claude + Codex)

Stage 3 (chunk_gdn caller-owned scratch) has multiple valid implementation approaches:

- **Approach A:** `ChunkArena` is a struct of MlxBuffers, sized for max chunk; threaded through every `chunk_*` callsite.
- **Approach B:** Use `mlx_native`'s existing buffer pool extended to a per-prefill arena; less invasive but harder to size correctly.
- **Approach C:** Lift only the LARGEST 4 scratches (the ones that actually trigger residency demotion); leave the small ones inline.

Each approach has different code-shape tradeoffs. Dual-mode lets Claude and Codex implement different approaches in parallel git worktrees, queen judges based on the acceptance criteria above, winner gets merged. This is exactly the kind of high-risk-multiple-valid-paths refactor that benefits from competitive implementation.

Stages 1-2 (FA/DN wrapper arenas) are mechanically straightforward and could be solo-implemented; running them in dual-mode anyway gives free cross-review and might surface a more elegant API.

---

## How to invoke

```
/cfa Refactor Qwen3.5/Qwen3.6 inference forward path to lift wrapper-internal
scratch into caller-owned per-prefill arenas, reducing
mlx_native::sync_count() per pp101 dwq48 prefill from 161 to <=25 without
coherence or decode regression. See docs/research/adr-013-p21-arena-refactor-cfa-spec.md
for full spec, stages, acceptance criteria, and operator runbook.
```

The /cfa skill will read this spec, compute its plan, present the queen+worker roster, and ask for approval before launching workers.

---

## Cost estimate

- LoC: ~800-1500 lines across `forward_gpu.rs` + `gpu_full_attn.rs` + `gpu_delta_net.rs` + new arena modules
- Time: 4-12 hours of focused work × 2 (dual-mode) + queen judge time
- Token cost: ~$50-150 across both teams (queen=opus, workers=sonnet)
- Hardware: must run on M5 Max with the dwq48 GGUF on disk (already staged)

---

## Closing the loop

If P21 lands cleanly, ADR-013 §17 ship-gate (match-or-beat llama.cpp on prefill) is met for the first time post-implementation. Decode parity is already met (1.05× faster than llama.cpp). The structural floor is no longer the per-layer commit pattern but kernel-internal compute, which is already at-or-better-than llama.cpp's per-token rate (P19 H9 measured 0.90 ms/token vs llama-completion's 1.18 ms/token).

If P21 stalls at Stage 3 (the iter58b mine) — the conservative outcome is to ship Stages 1-2 alone, which delivers `sync_count ≤ 70` (-91 commits, ~120 ms wall savings stretch) without touching the chunk kernel. That's still a 50% reduction in per-prefill sync overhead and would put pp101 at ~157 t/s ≈ 0.18× of llama-completion's 850 t/s — closing the gap by 4×.

Either outcome is a real, measurable improvement on the structural floor identified by P19 H9.
