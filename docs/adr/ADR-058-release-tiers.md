# ADR-058: Release tiers and the full-gate machine contract

- **Status:** Accepted — implemented (v0.1.22): host preflight in the
  qualification workflow, fast-path tier guard in publish, exact-SHA CI
  check relaxed to accept `workflow_dispatch` runs.
- **Date:** 2026-09-10
- **Related:** ADR-056 (conformance battery), the `release.yml` and
  `cache-lifecycle.yml` workflows, `scripts/release_gate_preflight.sh`

## Context

Shipping v0.1.21 measured the cost of a single-tier release process that
always runs the full cross-family qualification gate:

1. **Environmental failures surface mid-run.** The gate asserts a quiet
   host (zero swap movement while the DeepSeek decode-cohort benchmark
   holds ~115 GiB resident; no existing model runtimes). Two of three
   failed release attempts died on machine state — stray servers at
   minute 1, swapout growth during model load at minute 20 — each costing
   a partial run before the condition was visible.
2. **The publish gate's exact-SHA CI check assumed unbatched pushes.**
   Only the head commit of a pushed batch receives a `push`-event CI run;
   a release SHA pushed together with later commits had no matching run,
   and publish failed on a fully-green candidate.
3. **The fast path had no guard.** `skip_qualification` could publish
   inference-path changes without the qualification gate — the flag was
   usable to dodge the gate rather than to skip what the gate does not
   cover.

## Decision

Two release tiers, explicitly chosen, with the machine contract made
mechanical:

- **Full tier** (default): the cross-family qualification gate, run on a
  freshly rebooted, otherwise-idle machine. `scripts/release_gate_preflight.sh`
  checks the gate's own conditions up front — no model runtimes, zero swap,
  ≥90% system-wide free memory, nominal thermal via the sourced thermal
  guard — so a dirty host fails in seconds with an actionable message
  instead of 20–40 minutes in. No gate assertion is weakened; the preflight
  only moves the same conditions earlier.
- **Fast tier** (`skip_qualification`): build, sign, notarize, exact-SHA CI,
  publish — for releases that do not touch the model path. A tier guard in
  publish refuses the fast path when `src/inference/**` or
  `src/serve/api/grammar/**` changed since the previous tag: those paths are
  what the qualification gate exists to test.
- The exact-SHA CI requirement accepts a green run with `push` **or**
  `workflow_dispatch` event on the release SHA, so a cancelled-as-superseded
  run can be rerun into validity instead of requiring a re-push.

## Consequences

- Releases are no longer hostage to machine state discovered mid-run: the
  preflight turns the reboot contract into a fast, legible failure.
- The fast path cannot silently dodge qualification for model-path work;
  tier choice is enforced, not assumed.
- The full gate keeps every assertion (exact-artifact binding, model
  verification receipts, memory/thermal/contention guards, cross-family
  waves) — streamlining changed where conditions are checked and which
  releases need them, not what is checked.
- Operational rule for releasers: full-tier releases start from a reboot;
  the preflight will say so if you forget.
