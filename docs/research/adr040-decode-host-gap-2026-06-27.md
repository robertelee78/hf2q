# ADR-040 §23 — N=8 decode host-gap LOCALIZED (2026-06-27)

Follow-on to the TQ-fold refutation: the N=8 decode gap to llama.cpp is NOT kernel/TQ work; it is
the WALL-vs-GPU-busy delta (host work serialized while the GPU is idle). This localizes WHERE.

## Method

Added gated host-phase wall-clock timers (`HF2Q_HOST_PHASES=1`, zero-cost off) — a `host_phases`
module mirroring `catsplit` (batched_body.rs), wrapping the per-step host phases inside
`decode_batch_gemma4` + the worker loop: the two `commit_and_wait` syncs, the two GPU→host
readbacks, the Pass-2 sample loop, the pre-forward gather, `scheduler.step()`, `publish()`.
Body/lm_head ENCODE time derived by subtraction (`DECODE_BODY_GPU_NS − body_wait − readback`).

## Measured (N=8, gemma4 Ara Q5_K_M, M5 Max, 128 steps, 229 t/s)

per step: wall 34.93ms, GPU-busy 27.65ms → gap 7.28ms (21%).

| phase | ms/step | GPU idle | note |
|---|---|---|---|
| body GPU exec (inside body_wait 25.43) | ~25.4 | no | IS the work; ≈ llama's whole step |
| lm_head GPU exec (inside lmhead_wait 3.51) | ~3.5 | no | irreducible |
| body+lmhead ENCODE (CPU records ~1731 dispatches) | **2.44** | **yes** | single session commits only at finish → GPU idle during encode |
| sample_loop (8× full-vocab argmax + finalize + detok + sched-advance) | **1.55** | **yes** | vocab≈256K → 2M compares/step |
| worker-loop misc (tokio / between-iteration / handles Vec) | **~1.85** | **yes** | unaccounted remainder; scheduler.step()+publish() measured ≈0 |
| sync/commit overhead (2 commit_and_wait beyond GPU exec) | **1.29** | **yes** | body_wait+lmhead_wait 28.94 − GPU-busy 27.65 |
| readbacks (hidden 0.005 + logits 0.15) | 0.15 | yes | unified memory → ~free (NOT a round-trip cost) |
| scheduler.step(), publish(), gather, mount-clear | ~0.00 | yes | negligible |

**Sum of GPU-idle host work ≈ 7.3ms ≈ the gap.** Fully accounted.

## Key findings

1. **The "hidden GPU→host→GPU round-trip" is a NON-issue (0.005ms).** Unified memory makes
   `as_slice`/`to_vec` near-free. Fusing body+lm_head saves only sync *granularity* (~0.6ms),
   not data movement. (Refutes the pre-instrumentation suspicion.)
2. **The gap is FRAGMENTED host work, all serialized with the GPU idle** — no single dominant
   cost. Biggest recoverable chunks: encode 2.44ms, worker-misc 1.85ms, sample 1.55ms, sync 1.29ms.
3. **GPU-busy (27.65ms/step) already ≈ llama's entire step (27.5ms).** To match llama we must get
   wall down to ~GPU-busy, i.e. overlap/eliminate ~7ms of host work → near-perfect CPU/GPU overlap.
4. The decode is a fully-serial CPU↔GPU ping-pong: encode(GPU idle) → commit+wait(GPU runs) →
   sample(GPU idle) → bookkeeping(GPU idle) → next. The autoregressive dependency is ONLY the
   argmax; detok/scheduler/publish/emit and the next-step encode-structure do not block the next
   token and could overlap GPU exec.

## Ranked fix hypotheses (to codex-check, then spike top one)

- **H1 — Pipeline post-sample bookkeeping off the GPU critical path.** After argmax, IMMEDIATELY
  submit the next step's body; do detok/scheduler-advance/publish/emit/stop-checks for the prior
  step AFTER submit (overlapped with GPU). Recovers ~sample(1.55) + worker-misc(1.85) ≈ up to 3.4ms.
  No kernel changes; touches the worker loop ordering. Risk: byte-identity of slot output, EOS/stop
  timing. Medium.
- **H2 — Overlap next-step body ENCODE with current GPU exec** (inter-step pipelining). Recovers
  ~2.44ms. Blocked-ish by token dependency (encode needs the sampled token in the embed-gather),
  but most of the 1731-dispatch recording is static structure. Intra-step CB-chunking already
  tried → +1.5-2.7% only. Higher risk/complexity.
- **H3 — Fuse body + lm_head into ONE session/sync.** Recovers ~0.6ms sync overhead. Low risk,
  low payoff. Hidden stays on GPU (already ~free), one commit_and_wait instead of two.
- **H4 — Faster argmax** (SIMD the 8× full-vocab scan, or reuse the GPU-argmax already computed
  bit-exactly). Recovers part of 1.55ms. Low risk.

H1 is the highest payoff-to-risk: it attacks the largest GPU-idle host chunk (bookkeeping+sample,
~3.4ms) with no kernel/quant changes, just reordering the worker loop to launch-then-bookkeep.
The realistic ceiling of H1+H3+H4 is ~wall 30-31ms → ~260-270 t/s (vs llama 291). H2 (encode
overlap) is needed to fully close it but is the riskiest.
