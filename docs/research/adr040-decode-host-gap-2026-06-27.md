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

## UPDATE — H1 codex-REJECTED + finer localization (2026-06-27)

**codex REJECTed H1 as a worker-loop reorder.** Blockers: (Q5) deferring
`scheduler.advance_after_decode`/release past the next `scheduler.step()` violates the
scheduler's state contract (scheduler.rs:355-361) — stale `in_flight` → delayed auto-release,
delayed promotion, stale handles in the next batch. (Q4) detok/splitters are NOT purely
observational — the tool-call splitter triggers grammar runtime (engine.rs:5648-5656) that masks
the NEXT logits (5599-5600); EOS returns BEFORE pushing the token (5629-5634). (Q7) body/lm_head
are `commit_and_wait` (synchronous) — H1 is not a reorder, it needs a real async submit/wait split
(the risky refactor). The SAFE deferrable subset (text assembly + publish only, keeping
advance/release/EOS inline before step()) recovers only ~0.5ms.

**Finer localization (decode_batch_TOTAL + worker_iter_TOTAL timers):** per step wall 35.97ms,
decode_batch 33.52ms, worker_iter 35.97ms → **2.45ms/step is OUTSIDE decode_batch** in the worker
loop, and it is NOT scheduler.step/publish/admit (all ≈0). This is **async-runtime / thread-
scheduling contention** — the dedicated model-worker thread competing with the probe's 4-thread
tokio runtime + 8 concurrent generate() tasks for cores. llama.cpp has NO async runtime (one tight
C++ loop) → it does not pay this. May be partly benchmark-specific (the probe's runtime config).

**Final gap decomposition (per step, GPU-idle host work ≈ 7-8ms):**
- worker-outside-decode 2.45ms — runtime/thread contention (hard; environmental; llama avoids by design)
- body+lmhead ENCODE ~1.9ms — GPU-idle CPU dispatch recording; needs async submit/wait split (risky)
- sample_loop ~1.5-2.5ms — 8× full-vocab(256K) argmax + detok; argmax is on the critical path (SIMD/GPU = H4, low risk)
- sync overhead ~1.3ms — 2 commit_and_wait beyond GPU exec; fuse body+lmhead to 1 sync (H3, ~0.6ms, low risk)
- readbacks ~0.15ms — unified memory, ~free

**Revised conclusion:** no single big lever. The LOW-RISK bundle is **H3 (fuse body+lm_head to one
sync, ~0.6ms) + H4 (SIMD/GPU argmax, ~0.8ms)** ≈ 1.4ms → ~235 t/s, byte-identity-safe. The bigger
chunks (encode overlap, runtime contention) need the risky async-pipeline refactor + runtime tuning
for ~250-260 t/s. Coherence parity (the harder goal) is already met. DECISION: low-risk H3+H4
bundle vs risky async-pipeline vs accept-current. The instrumentation (HF2Q_HOST_PHASES) is the
durable artifact for whichever path.

## CORRECTION (2026-06-27) — the 2.45ms was a benchmark artifact, not a per-step cost

Two things resolved after building the finer timers + iter-I:

1. **H4 (argmax) DONE — iter-I, +1.1%.** The scalar first-max loop (`v > bv`, loop-carried dep)
   over the 8×256K vocab was 1.5ms/step. Replaced with `argmax_f32_first_max` (max-reduction +
   first-equal scan, both auto-vectorize). BYTE-IDENTICAL (unit test + real-model
   `slot_aware_n8_per_slot_parity_vs_serial` green). argmax+finalize 1.5→~1.0ms; 229.5→232 t/s.
   (Candidate-scan vectorization gave 0 gain → reverted.)

2. **The "2.45ms/step worker-loop misc" is NOT runtime contention and NOT a per-step cost.**
   - tokio-thread A/B (`HF2Q_BENCH_TOKIO_THREADS` 1/2/4/8): throughput FLAT 231-233 t/s,
     worker_iter FLAT ~34.4ms. → async-runtime contention REFUTED.
   - `worker_iter − decode_batch ≈ 2.4ms` is constant. It is the benchmark's **8 prefills
     (admit, eager) amortized over 128 decode tokens** (8 × ~30ms / 128 ≈ 1.9ms/step) — a STARTUP
     artifact that → 0 in long generation. NOT a per-step decode cost.

**Corrected per-step budget — `decode_batch_TOTAL` = 31.9ms is the TRUE per-step decode:**
GPU-busy 27.6 + encode 1.9 + sync 1.3 + argmax ~1.0 + readback 0.15 = 32.0 ✓ (fully accounted).
So the genuine gap to llama (27.5ms/step) is **~4.4ms/step of real host overhead**, NOT 7ms:
- **encode ~1.9ms** — CPU recording ~1731 dispatches/step (GPU idle). Lever: DISPATCH FUSION
  (fewer/bigger kernels, the KVENC pattern extended). This is the most llama-like lever — llama
  issues far fewer dispatches/token. Also reduces per-dispatch GPU launch overhead.
- **sync ~1.3ms** — two `commit_and_wait` (body, lm_head). Lever: fuse body+lm_head into ONE
  session/sync (H3, ~0.6ms, byte-identity-preserving).
- **argmax ~1.0ms** — residual after iter-I (the finalize rerank candidate dots). Lever: trim
  the rerank work.
- readback 0.15ms — unified memory, ~free.

**Reframed target:** drive `decode_batch_TOTAL` 31.9 → ~27.6ms (= GPU-busy) → ~llama. The real
levers are dispatch fusion (encode + GPU launch overhead) and sync fusion — both byte-identity-
targetable, NO async-pipeline refactor needed, NO runtime/language excuse. Same silicon, fewer
dispatches + fewer syncs = llama parity.
