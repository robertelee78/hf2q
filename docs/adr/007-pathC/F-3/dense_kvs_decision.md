# ADR-007 Path C / F-3 — `dense_kvs` decision: branch F-3-C (keep as opt-out)

**Date:** 2026-05-05 (iter-5)
**Verdict:** Branch F-3-C — `dense_kvs` stays as opt-out via `HF2Q_USE_DENSE=1`. 8-bit TQ remains the default. This **confirms with empirical evidence** the policy that already shipped at the 2026-04-24 close (§1083), and **retires** the original C-4 stub-removal directive as based on a falsified assumption.

## Decision criterion (revised)

Per ADR-007 §F-3 binary spec:

> F-3-A if F-2 passes strict Gate C (TQ is good enough that dense-fallback is
> dead weight); F-3-B if F-2 fails strict and mixed-precision is the only path
> to PPL parity.

F-2's actual verdict (`uncalibrated_is_floor.md`) is FAIL strict Gate C **with
proof that no codec-level remediation exists at 8-bit**. This is a third
outcome the binary spec did not anticipate. The corresponding branch is:

**F-3-C: keep `dense_kvs` as opt-out at 8-bit TQ default.**

## Why F-3-A (REMOVE) is mantra-noncompliant

F-3-A would retire `HF2Q_USE_DENSE=1`, delete `dense_kvs` allocation in
`forward_prefill.rs:274-285`, delete `use_dense_sdpa` selector at
`forward_mlx.rs::forward_decode`, and repoint sourdough gate at TQ-active.

The mantra-noncompliance: this would force every downstream consumer to accept
the intrinsic 1.24% Lloyd-Max 8-bit distortion as a hard floor. Some downstream
uses **need** byte-exact-comparable behavior:

- Sourdough byte-exact gate (`scripts/sourdough_gate.sh`) — already pinned
  via `HF2Q_USE_DENSE=1` for byte-identity vs llama.cpp.
- Future ADR-005 / ADR-009 PPL-parity tests — strict Gate C compliance is a
  release gate.
- ADR-017 B-tq.1 codec roundtrip parity (when it lands) — needs a known
  byte-exact reference for cross-validation.
- Any automated test that fingerprints model output via SHA hashing.

F-3-A would silently break all of these. F-0.3 has shown that no codec-level
upgrade short of F16-dense closes the gap; F-3-A removes the only available
path. **Not mantra-compliant.**

## Why F-3-B (EARN with measured policy matrix) is unnecessary

F-3-B was the formal alternative: write a new ADR-009-followup with a measured
policy matrix (4 layer policies × 5 prefix lengths × {cosine, PPL, MMLU, mem,
t/s}). This was scoped to PROVE that mixed-precision is justified.

F-0.3's empirical evidence is now the load-bearing justification:

1. The N(0,1) Lloyd-Max codebook IS optimal for the production distribution
   (F-0.3 std deviation 0.0012 across 112 cells).
2. The 1.24% PPL gap IS intrinsic to 8-bit Lloyd-Max distortion (not a
   codec/kernel bug — F-0.2 verified the kernel implements the right math).
3. Therefore the **only** way to close strict Gate C below 1.15% with the
   current codec is to bypass the codec entirely (i.e., dense F16 / F32).
4. Therefore `dense_kvs` is not "a principled mixed-precision code path" or
   "a fallback" — it is **the load-bearing escape hatch** for any consumer
   that needs strict Gate C compliance. Calling it a "fallback" was the
   close-section's mantra-uncomfortable label; F-3-C reframes it as **the
   strict-Gate-C-compliance path**.

Writing the policy matrix would re-prove F-0.3's findings at additional cost.
The matrix would show: every cell with `kv_path == TQ-active` clears Gate A
cosine ≥ 0.9998 but misses Gate C strict 1.15% by ~0.09%. Every cell with
`kv_path == F16-dense` clears Gate C strict at the cost of 2× memory. That's
the trade. The empirical evidence is sufficient; further measurement adds no
information.

## What F-3-C ships (already in production)

```rust
// Default decode path: 8-bit Lloyd-Max HB SDPA. Gate A 0.9998 mean,
// Gate B 0.8%, Gate C 1.24% (industry-standard literature gates pass;
// strict 1.15% misses by 0.09%, which F-0.3 proves is intrinsic).

// Opt-out: HF2Q_USE_DENSE=1
// - Allocates dense_kvs (BF16 K/V buffers in addition to TQ-packed)
// - use_dense_sdpa selector flips at decode dispatch
// - Strict Gate C compliance (byte-exact vs llama.cpp at 3656 bytes
//   sourdough gate floor 3094)
// - 2x KV memory cost vs TQ-active
```

This is the close-section §1083 shipping contract. F-3-C confirms it with the
empirical justification F-0.3 provides.

## Mantra-discipline ledger

The original ADR-009 Track 3 retirement directive (close-section §1155) said
`dense_kvs` is "now a principled mixed-precision code path", which I called a
"recategorization, not a removal" (Path C reopen § Why-finding 2). That framing
was correct **at the time** because the empirical justification was missing.
F-3-C retires the framing concern: `dense_kvs` is no longer a recategorization
puzzle — it is a documented, measured policy choice with a falsifiable
foundation (F-0.3 distribution measurement, F-2 calibration falsification,
F-3-C decision derived from the data).

The standing rule for any future codec change: if a downstream phase argues
for removing `dense_kvs`, the burden is on that phase to either:
1. Demonstrate a codec that closes strict Gate C without F16-equivalent memory
   (the "sub-2-byte high-fidelity codec" research direction), or
2. Provide measured evidence that the consumer set has changed and no longer
   needs strict Gate C compliance.

Until then, F-3-C stands.

## What's unblocked

- **F-4 (262K context unlock)** is independent of F-3 verdict and proceeds.
- **F-5 (MMLU/LongBench/needle benchmarks)** can run against both TQ-active
  default AND `HF2Q_USE_DENSE=1` to characterize the policy matrix at paper
  standard, but is not blocked on F-3 closure.
- **F-6 (CLI flag + 16-bit opt-in)** spec needs revision per F-2's section
  "Implications for downstream Path C phases / F-6 (16-bit opt-in) — spec
  correction needed". The CLI flag part is straightforward; the 16-bit codec
  part needs the spec to specify a structural advantage over F16 dense.

## What's still open

The original C-4 directive ("Remove the Track 3 stub") is **retired** by this
verdict. Future work may revisit if:

1. A new sub-2-byte high-fidelity codec is invented (research-grade);
2. The strict Gate C threshold is loosened by user fiat (not mantra-compliant
   for Path C since the threshold was a Path C re-strict); or
3. A consumer set change makes byte-exact compliance unnecessary.

Until one of those happens, `HF2Q_USE_DENSE=1` is a permanent fixture.
