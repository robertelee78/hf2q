# ADR-029: gemma4-APEX-Q5_K_M Decode Gap is in the MoE Pipeline, Not TQ

- **Status**: accepted (2026-05-11)
- **Date**: 2026-05-11
- **Supersedes**: nothing structurally; corrects ADR-028 §iter-486 + §iter-487 closure verdict and the iter-308 "Q5_K nr0=2" smoking-gun claim
- **Deciders**: Robert (operator), Claude (cfa-adr028-gap-20260511 swarm: queen-phase1 / claude-impl perf-engineer / codex-impl structural-reviewer)
- **Tags**: performance, root-cause, gemma4, moe, measurement-discipline, baseline-correction

## Decision (one sentence)

The 27% decode gap between hf2q and llama.cpp on `gemma4-ara-2pass-APEX-Q5_K_M.gguf` (Apple M5 Max) is **localized to hf2q's MoE pipeline** (`router_proj` + softmax/top-k + routed `mul_mv_id` experts), which accounts for ~49% of hf2q's per-token wall-clock; **TQ work is at most 7% of the gap and is no longer a viable optimization target for this model**.

## Context

ADR-028 ran 141 /loop iterations between 2026-04 and 2026-05 chasing the decode gap on gemma4-APEX-Q5_K_M. Two recent closures (iter-486 reframe → "0.96–1.04× tied with peer"; iter-487 → "mission CLOSED at +4.5%") were claimed but were not reproducible. Operator re-measured on 2026-05-11 and got 73.5 vs 97.4 t/s (single-shot). Re-running the apples-to-apples bench in a thermally-stable session yielded the empirical result this ADR is built on.

The investigation was scoped via `/cfa` in **review-only** mode (Codex's `--full-auto` sandbox blocks Metal device access, so Codex cannot bench — Claude measures, Codex audits peer source). Session: `cfa-adr028-gap-20260511`.

## Empirical baseline (this session's fresh measurements)

**Hardware**: Apple M5 Max, 128 GB · **Thermal**: no warnings, single session · **HEAD**: hf2q `cfbdc469` · **Peer build**: `d05fe1d7d (9010)` (homebrew llama.cpp) · **Model**: `/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf` (19 GB, mtime May 6 2026)

| Source | Tool | n | Result | σ-pct |
|---|---|---:|---:|---:|
| peer cold | `llama-bench -p 0 -n 200 -r 10` | 10 | 103.72 ± 0.31 t/s | 0.30% |
| peer post-hf2q | same | 10 | 100.09 ± 0.78 t/s | 0.78% |
| hf2q HEAD `--benchmark` | `hf2q generate ... --max-tokens 200 --benchmark` | 5 | 75.10 t/s median (75.14 mean, σ 0.167) | 0.22% |

**Ratio**: 75.10 / 101.9 ≈ **0.7245× peer** (27.55% slower). Coherence ✓ (deterministic, matches peer output).

All σ-pct values are well below the 5% thermal-stability gate; this is a clean measurement.

## TQ attribution (S2 — operator's primary question)

> **Q (operator)**: "Is hf2q's TQ work (Hadamard pre-processing + HB-shadow-cache encode + `flash_attn_vec_tq_hb`) the dominant driver of the 27% decode gap?"
>
> **A**: **NO. Conclusively.**

| probe | env | median t/s | coherent | gap recovered |
|---|---|---:|:---:|---:|
| baseline (full TQ chain) | default | **75.10** | ✓ | 0% |
| **2a `HF2Q_USE_DENSE=1`** (TQ surgically removed) | env | **77.10** | **✓** | **+7.00%** |
| 2b skip TQ encode only | `UNSAFE+SKIP_TQ_ENCODE` | 77.90 | ✗ | +9.80% |
| 2c skip TQ-HB SDPA only | `UNSAFE+SKIP_TQ_SDPA` | 87.80 | ✗ | +44.47% |
| 2d skip both | `UNSAFE+SKIP_TQ_ENCODE+SKIP_TQ_SDPA` | 88.30 | ✗ | +46.22% |

The only coherent comparison is probe 2a: substituting an equivalent dense F32 K/V + `flash_attn_vec` SDPA recovers **+7.0% of the gap** (net TQ-chain cost vs dense: 0.345 ms/tok = 9.4% of the 3.67 ms/tok gap).

The 2c/2d probes recover up to +46% but break coherence at token 33 (the SDPA dispatch is skipped entirely). They establish that the **incoherent timing budget** of the TQ chain is ~2.0 ms/tok, but most of that budget is unavoidable attention work that any coherent kernel would still pay.

## MoE attribution (S6 — the smoking gun)

| probe | env | median t/s | coherent | gap recovered |
|---|---|---:|:---:|---:|
| baseline | default | 75.10 | ✓ | 0% |
| **H5B-all** (skip router+experts+swiglu+weighted_sum) | `UNSAFE+SKIP_ROUTING=1 SKIP_MOE_EXPERTS=1 SKIP_MOE_SWIGLU=1 SKIP_WEIGHTED_SUM=1` | **148.40** | ✗ | **+232%** (1.43× peer) |
| H5B-routing only | `UNSAFE+SKIP_ROUTING=1` | 120.50 | ✗ | +159% |
| H5B-experts only | `UNSAFE+SKIP_MOE_EXPERTS=1` | 92.60 | ✗ | +61% |

**MoE pipeline = 6.58 ms/tok = 49% of hf2q's 13.5 ms wall-clock**:
- **Routing** (`router_proj` qmatmul + softmax/top-k): **~5.02 ms/tok = 38% of wall-clock**
- **Experts** (`mul_mv_id` × {gate, up, down} × top-8): **~2.52 ms/tok = 19% of wall-clock**
- (Pipeline overlap: split sums exceed total because skipping experts still runs routing.)

These probes break coherence at token 33; ms/tok savings are **upper bounds**. A real MoE optimization will save less than 6.58 ms/tok because peer also pays MoE work. But:
- hf2q MoE cost: ~6.58 ms/tok
- Peer total decode cost: 9.65 ms/tok (ENTIRE per-token budget)
- → If peer's MoE cost is even modestly under 6.58 ms/tok (likely, given peer's NR2-fused `mul_mv_id` Q6_K path), the gap is fully explained by MoE-side per-dispatch inefficiency.

## Dispatch arithmetic (closes the budget)

| | dispatches/tok | µs/dispatch | ms/tok | matches measured |
|---|---:|---:|---:|:---:|
| hf2q | 925 | 14.5 | 13.4 | ✓ (13.54) |
| peer | 105 (candle Phase 0 ref) | 92 (derived) | 9.65 | ✓ (9.65) |
| **gap** | **+820** | **−77.5** | **+3.67** | ✓ |

hf2q's per-dispatch wall-clock is small (14.5 µs); the gap is from **issuing 8.9× more dispatches per token**, not from any single dispatch being slow. The ~820 excess dispatches per token are predominantly in the MoE path (top-8 experts × 3 mat-vecs × 30 layers = 720 `mul_mv_id` dispatches minimum, plus per-layer router proj + softmax + top-k).

## Gap-scaling regime (S4 — corroborates dispatch-floor signature)

| shape | hf2q t/s | peer t/s | ratio | gap ms/tok |
|---|---:|---:|---:|---:|
| n=200 | 75.10 | 103.66 | 0.7245× | 3.669 |
| n=500 | 74.60 | 102.32 | 0.7291× | 3.632 |
| n=1000 | 73.90 | 91.30 | 0.8094× | 2.579 |

Gap is **constant in absolute ms/tok at n=200/500**, then **shrinks at n=1000** because peer slows -11.9% with KV growth while hf2q only slows -1.6%. This is the **classic fixed-per-token-overhead signature** — dispatch + barrier floor, not KV-bandwidth-bound. The n=1000 narrowing is peer paying a tax hf2q's compressed KV avoids, not hf2q improving.

## Corrections to prior claims

### Correction 1 — iter-486 peer baseline (77 ± 15 t/s) was wrong

iter-486 declared the gap "tied at HEAD" with peer at 77.18 ± 15.36 t/s. **σ = 15.36 (20% of mean) was the thermal-throttle smoking gun**, not flagged at the time. Fresh re-measurement on the same machine, same GGUF, same llama.cpp build gives peer = 103.66 ± 0.30 t/s (σ-pct 0.29%). The "tied" verdict was an artifact.

### Correction 2 — iter-487 "mission CLOSED" verdict invalidated for gemma4

Built on iter-486's bad baseline. Specifically: iter-487 cited "operator's own llama-cli: 70.1 tok/s peer, hf2q 73.3 = +4.5% hf2q faster." The 70.1 peer number is consistent with thermal throttle; it does not survive a clean session bench. **Mission re-opened iter-488; this ADR records the actual root cause.**

### Correction 3 — iter-308 "peer Q5_K is N_R0=2; hf2q is N_R0=1" is WRONG

Codex S5 axis 4 source-read at `/opt/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`: peer Q5_K is `N_R0 = 1`. Peer Q6_K is `N_R0 = 2`, and hf2q already has the matching Q6_K NR2 port. The iter-308 "smoking gun" claim was either misremembered or stale from a different commit. The actual structural diff at qmatmul kernels is much smaller than claimed.

### Correction 4 — "gemma4 is dense FFN" assumption was wrong

The CFA spec assumed gemma4-26B-A4B is dense FFN. Both peer (`/opt/llama.cpp/src/models/gemma4.cpp`) and hf2q (`/opt/hf2q/src/serve/forward_mlx.rs`) load and execute MoE for this architecture (`expert_count = 128`, `expert_used_count = 8`). Future investigations must NOT skip the MoE path under a "dense FFN" assumption.

### Correction 5 — gemma4 shared-KV reuse is DORMANT on this APEX GGUF

Codex S5 axis 1 found that peer marks upper gemma4 layers as `has_kv == false` and reuses earlier KV caches (`/opt/llama.cpp/src/models/gemma4.cpp:240-245` + `llama-model.cpp:2004-2011`). BUT this APEX GGUF declares `attention.shared_kv_layers = 0` — the reuse branch is inactive in BOTH engines for this file. H5A: falsified by metadata for this model.

## Hypothesis verdicts

| hypothesis | predicted | measured | verdict |
|---|---|---|---|
| H1 `fuse_fwht_pre` | (iter-485 falsified 3×) | not re-tested | **DO NOT RE-TRY** |
| H2 cross-WG in-kernel reduce | (Apple Metal infeasible at NWG=16/32) | structural | **INFEASIBLE** |
| H3 FWHT-undo into reduce | (iter-485 Worker C falsified, bit-exact parity) | not re-tested | **DO NOT RE-TRY** |
| H4 dual 4-bit K+V encode | (iter-485 Worker D falsified, -3.8% to -18.5%) | not re-tested | **DO NOT RE-TRY** |
| **H5A** missing gemma4 shared-KV reuse | 5-15% if `shared_kv_layers > 0` | GGUF metadata = 0 → inactive in both engines | **FALSIFIED (source-only)** |
| **H5B** MoE under-attributed | 5-20% | +97.6% (148.4 t/s); routing alone +60.5% | **STANDING — DOMINANT** |
| **H5C** TQ-HB compute/control-bound | 5-25% via coherent dense | +7.0% (≪ 30% gate) | **FALSIFIED for dominant-driver status** |

## Decision

1. **Halt** all TQ / FWHT / SDPA / qmatmul-fusion optimization work on gemma4-APEX-Q5_K_M. Ceiling for any of these levers is +7-9% of the gap. They are off the critical path.
2. **Pivot** next-iter work to a focused MoE-pipeline investigation. Three concrete sub-targets, ranked by likely ROI:
   - **MoE-1**: Count exact `router_proj` + `mul_mv_id` dispatch issuance per gemma4 layer per decoded token; compare against llama.cpp's MoE graph layout. Identify which kernels in the routing pipeline (`router_proj` qmatmul, softmax, top-8 selection, expert mat-vec) are being issued as separate dispatches that could be fused.
   - **MoE-2**: Verify whether `quantized_matmul_id_ggml` (the `mul_mv_id` dispatcher) shares the Q6_K NR2 path or is still on a slower NR1 dispatcher. If NR1, port the NR2 optimization.
   - **MoE-3**: Audit barrier cadence around router-prep and expert dispatch (currently 510 barriers/tok). Many MoE barriers may be redundant if the `mul_mv_id` kernels are reads-only on the routing output.
3. **Fix** `HF2Q_MLX_KERNEL_PROFILE=1` gating at `src/serve/forward_mlx.rs:5871` so it doesn't hard-fail on Q6_K lm_head — gate the LM-head-specific path separately so per-kernel-type buckets unlock for any GGUF using Q6_K head. Without this, every future investigation hits the same blind spot.
4. **Re-test** qwen3.6 mantra (iter-487 claimed 1.27× peer): given iter-486 was wrong about gemma4, the qwen3.6 baseline may also be a thermal artifact. Re-measure in a thermally-stable session with σ-pct check before citing.
5. **Adopt** as standing practice that every "X× peer" claim in memories, ADRs, and commit messages must include σ-as-pct-of-mean. Refuse to consume claims with σ > 5% without re-measuring. (Standing rule already saved as `feedback_do_not_trust_file_claims_re_measure_2026_05_11.md`.)

## Consequences

Positive:
- 4-day investigation thrash resolved with a clean empirical answer.
- The 7+ falsified hypotheses (H1 × 3, H2, H3, H4, H5A, H5C) are now closed with explicit evidence; future iters won't re-try them.
- The next-iter direction is concrete (MoE-1/-2/-3) and falsifiable (counter measurements + bench probes already shown viable).
- Standing measurement discipline (σ-pct check, re-measure-don't-cite) reduces risk of recurrence.

Negative / risks:
- MoE optimization is a multi-week structural project (kernel fusion + dispatch graph rewrite), not a 1-line flag flip. Operator should expect longer iteration cycles than the TQ flag-tweak loop.
- We did NOT directly measure peer's MoE budget (no equivalent skip flag in llama.cpp). H5B remains "standing" not "proven" until a like-for-like comparison is set up.
- `HF2Q_SKIP_*_MOE` probes break coherence at token 33, so they cannot validate any partial MoE optimization in product mode — coherent partial wins must be verified with end-to-end output comparison instead.

## Validation (this ADR's own gates)

To reproduce the core finding (any future iter must show these numbers within σ-pct < 5%):

```bash
MODEL=/opt/hf2q/models/gemma-4-26b-a4b-it-ara-abliterated/gemma4-ara-2pass-APEX-Q5_K_M.gguf
# Peer baseline (repeat twice, discard first; check σ-pct < 5%):
/opt/homebrew/bin/llama-bench -m "$MODEL" -p 0 -n 200 -r 10
# Expected: ~100-104 t/s, σ < 1 t/s

# hf2q baseline:
/opt/hf2q/target/release/hf2q generate --model "$MODEL" \
  --prompt "Write a long story about a sentient telescope" \
  --max-tokens 200 --benchmark
# Expected: ~75 t/s, σ < 0.5 t/s, coherent output, ratio ~0.72× peer

# TQ-attribution probe (coherent):
HF2Q_USE_DENSE=1 /opt/hf2q/target/release/hf2q generate --model "$MODEL" \
  --prompt "Write a long story about a sentient telescope" \
  --max-tokens 200 --benchmark
# Expected: ~77 t/s, coherent, +7% gap recovered

# MoE-attribution probe (incoherent timing only):
HF2Q_UNSAFE_EXPERIMENTS=1 HF2Q_SKIP_ROUTING=1 HF2Q_SKIP_MOE_EXPERTS=1 \
  HF2Q_SKIP_MOE_SWIGLU=1 HF2Q_SKIP_WEIGHTED_SUM=1 \
  /opt/hf2q/target/release/hf2q generate --model "$MODEL" \
  --prompt "Write a long story about a sentient telescope" \
  --max-tokens 200 --benchmark
# Expected: ~148 t/s (1.43× peer!), incoherent output, +232% gap recovered
```

If any of these fail to reproduce within σ-pct < 5%, suspect (a) thermal throttle (let chip cool, retry), (b) HEAD divergence from `cfbdc469`, or (c) GGUF substitution (the APEX file mtime should be May 6 2026).

## Links

- `~/.claude/projects/-opt-hf2q/memory/project_adr028_synthesis_moe_pipeline_2026_05_11.md` (this finding, memory-form)
- `~/.claude/projects/-opt-hf2q/memory/project_adr028_iter488_mission_REOPENED_2026_05_11.md` (the re-open trigger)
- `~/.claude/projects/-opt-hf2q/memory/feedback_do_not_trust_file_claims_re_measure_2026_05_11.md` (standing measurement-discipline rule)
- CFA session artifacts:
  - `~/.claude/teams/cfa-adr028-gap-20260511/shared/findings.md` (Claude S7 synthesis)
  - `~/.claude/teams/cfa-adr028-gap-20260511/shared/reviews/codex-structural.md` (Codex S5 review, 300 lines, 7 axes + 3 hypotheses)
  - `~/.claude/teams/cfa-adr028-gap-20260511/shared/agents/claude-impl/{S1,S2,S3,S4,S6}.md` (per-subtask evidence)
  - `~/.claude/teams/cfa-adr028-gap-20260511/shared/agents/claude-impl/raw/*` (raw stderr from every probe)
- `docs/ADR-028-peer-parity-coherence-and-speed.md` — the prior mission ADR; this ADR-029 corrects its iter-486/487 closure verdicts but does not formally supersede the body
- `docs/ADR-027-qwen35-tq-kv-cache-and-persist-family.md` — TQ-KV mantra source (gemma4 does NOT use TQ-KV by default; `tq_kv = inactive` at load banner)
