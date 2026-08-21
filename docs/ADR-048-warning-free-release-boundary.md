# ADR-048: Warning-free release boundary

- **Status:** Accepted and implemented
- **Date:** 2026-08-20
- **Related:** ADR-045, ADR-046, ADR-047

## Context

At merge `23dfb7c0359b`, `cargo build --release --locked` succeeded but emitted
351 compiler warnings across roughly 2,600 lines. That made a normal operator
build look broken and made new diagnostics difficult to distinguish from
known noise. The diagnostic-chat merge was initially suspected because it was
the latest visible change, but compiler spans and first-parent history proved
that chat and local discovery contributed zero warnings.

The exact warning inventory was 345 `dead_code` and 6 `unused_imports`:

| Source family | Warnings | Root cause |
|---|---:|---|
| Qwen source-precision and copied-execution evidence | 269 | ADR-046 validation substrate compiled into release without an operator entrypoint |
| Exact-teacher target substrate | 58 | Structural/test consumer only |
| Calibration substrate | 18 | Structural/test consumer only |
| Dynamic allocator exact frontier | 5 | Test-only algorithm left outside its `cfg(test)` boundary |
| Scheduler compatibility wrapper | 1 | Production moved to the generalized parser; only unit tests retained the wrapper |

The same-source, first-parent release-check progression was:

```text
0 → 44 → 91 → 96 → 137 → 169 → 204 → 237 → 248 → 262 → 264 → 265 → 294 → 301 → 351
```

The first warning-bearing boundary was PR #82. Subsequent staged ADR-046
slices through PR #105 accumulated the initial research-path warnings. PR #101
introduced the scheduler wrapper warning. PR #106 and diagnostic-chat PR #107
each changed the warning set by zero. PR #108 completed the bounded B3b
source-teacher transaction but still supplied no CLI/API root, adding 50 more
release-unreachable diagnostics.

## Root-cause analysis

### Technical cause

ADR-046 deliberately stops short of an operator-reachable source-teacher
command, sensitivity generation, candidate materialization, and `--quant auto`
activation. PR #108 completed an internal, bounded one-shot source-teacher
transaction, but no command or API can construct and invoke it. Its private
validation modules nevertheless crossed the default binary's compile boundary.
Inline tests provided their only reachable roots, which explains why a test
build was nearly clean while the release binary reported the code as dead.

Ordinary Qwen dispatch was also sharing a thread-local evidence wrapper. Even
when no evidence scope was active, the release path carried dormant policy and
trace machinery. Warning suppression would have hidden that architectural
mistake rather than removed it.

### Process cause

Rust warnings were non-fatal. CI ran both `cargo check` and the release build,
but treated a zero exit status as success and did not enforce a warning-free
crate. Reviews repeatedly described the output as pre-existing warning debt,
so each staged slice could add more noise without failing its landing gate.

### Why existing checks missed it

- Unit tests supplied reachability for the staged modules and therefore did
  not reproduce the release warning profile.
- `cargo build --release --locked` returned success despite the 301
  diagnostics.
- The crate had no lint policy making warnings blocking.
- Some future-slice re-exports carried local `allow` attributes, reinforcing
  the mistaken assumption that compile-now/use-later was acceptable.

### Non-causes

- `src/chat/**` had no warning span and has a real `Command::Chat` root.
- DNS-SD discovery and model lifecycle control had no warning span.
- No dependency warning caused this incident; all 301 diagnostics belonged to
  the hf2q binary.

## Decision

The default binary compiles operator-reachable product behavior. Research or
validation code without an accepted production command/API root compiles with
the tests that prove it, not merely because a later phase may use it.

Specifically:

1. Calibration, exact-teacher, Qwen source-precision, copied-execution
   evidence, and source-teacher cache modules are one `cfg(test)` validation
   island until ADR-046 has an accepted product entrypoint. The completed B3b
   bounded runner remains part of that fully tested island.
2. Reusable Dynamic schemas, hashes, partitioning, and conversion lineage stay
   in production. Its exact Pareto search and calibration-bound adapter remain
   test-only while they have no production selector root.
3. Ordinary Qwen release dispatch calls the legacy/native MLX operations
   directly and preserves the existing environment-backed behavior. Evidence
   TLS, policy routing, source-teacher rejection, and trace capture compile
   only for tests.
4. Evidence identity/cache observation fields and load hooks compile only with
   that island. Release cache identity is the established model pointer.
5. The obsolete scheduler convenience wrapper is test-only; production keeps
   the generalized parser.
6. `[lints.rust] warnings = "deny"` is crate policy. A new hf2q warning is a
   build failure in developer, CI, test, and release profiles.

### ADR-046 source-teacher reconciliation

The original boundary above was correct at `c9443b15`: the completed internal
transaction had no command or API root. ADR-046's subsequent hidden,
fixed-profile `source-teacher` operator changes that reachability fact without
changing this ADR's governing rule. The release binary now compiles the exact
authenticated calibration, exact-target, Qwen source-precision,
base-text-cache, and fixed source-graph-scope chain consumed by that command.
The production source scope freezes its graph switches and rejects GGML or
fused-quantized dispatch; it does not restore copied-GGML policy or trace
machinery.

Everything not reachable from that operator stays on the original boundary:
copied-execution evidence and trace capture, Dynamic frontier search and its
calibration adapter, selector/autoquant activation, compatibility writers,
and replay remain test-only or unavailable. No Cargo feature or warning
allowance is used. This is the smallest promotion consistent with both the
new product root and warning-free release builds.

Three integration harnesses path-include deliberately wider source/stub modules
because the required internals are not library exports. Their test-module
boundaries use narrow `expect(dead_code)` declarations with reasons. Unlike an
`allow`, each expectation becomes a diagnostic if the expected partial-use
condition disappears.

No Cargo feature is introduced for the evidence island. An all-features build
would otherwise compile code that still has no usable product root, while the
feature name would falsely advertise an unsupported capability. A future
feature becomes appropriate only with the accepted bounded runner, explicit
operator surface, and shipping gates.

## Consequences

- A normal release build is quiet except for Cargo's compile/finish progress.
- Supported Qwen, serve, conversion, distribution, and chat surfaces remain
  compiled normally.
- ADR-046 model-free validation remains fully exercised by unit tests.
- Future staged work must either supply a reachable accepted surface or stay
  behind a coherent test/experimental boundary; warning allowances are not an
  admission mechanism.
- Promoting the evidence island requires removing the boundary as part of the
  same change that adds its real product root and proof.

## Validation contract

The landing commit must pass, from a clean source tree:

```sh
cargo build --release --locked
cargo test --locked --bin hf2q --no-run
cargo check --locked --all-targets --all-features
cargo test --locked --bin hf2q
```

The first two commands must emit no compiler warnings. Focused Qwen execution,
source-precision, calibration, exact-teacher, Dynamic allocator, scheduler,
distribution, and diagnostic-chat suites must remain green. A full real-model
quality/performance gate is not required because the change removes dormant
release instrumentation while preserving the established ordinary dispatch
calls; a bounded release-binary generation smoke proves that path executes but
makes no model-quality or performance claim.

## Landing evidence

Fail-first `cargo rustc --release --locked --bin hf2q -- -D warnings` failed on
the original pre-PR-#108 tree with all 301 diagnostics. Revalidation on current
main measured 351 diagnostics, including the 50 added by the still-unrooted B3b
runner. After the boundary correction:

- `cargo build --release --locked` completed in 32.05 seconds with zero
  compiler warnings;
- `cargo test --locked --bin hf2q --no-run` and
  `cargo check --locked --all-targets --all-features` completed warning-free;
- 132 focused affected tests passed, including all 41 source-precision/B3b
  worker tests;
- the serial binary suite passed 4,811 tests with zero failures and 53 ignored;
- the four changed integration harnesses passed 122 tests with zero failures
  and 10 explicitly ignored hardware/real-model gates; and
- release-binary `hf2q generate` loaded the exact local artifact
  `Qwen3.8-27B-Q4_K_M.gguf` (SHA-256
  `0fa8acc661d0edc60276c43705619fd848682dbf768ced9fe46cd8a572b8043d`),
  prefetched 16 tokens through the ordinary Qwen path, generated `OK`, exited
  successfully, and left no inference listener or model process resident.

The one-token smoke is execution proof, not a throughput or quality claim.
