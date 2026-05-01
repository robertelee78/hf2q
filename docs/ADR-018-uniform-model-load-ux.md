# ADR-018: Uniform Model-Load UX Across Families

- **Status:** **Accepted** (C1-C2 implemented 2026-05-01).
- **Supersedes:** none.
- **Related:**
  - **ADR-005** (Phase 1b: SERVE-mode load + iter-215 `LoadedModel` family-split that surfaced the divergence this ADR closes).
  - **ADR-008** (mlx-native is the sole inference backend — `LoadInfo::backend` is `&'static str` because the only legal value is compile-time fixed under ADR-008).
  - **ADR-013** (Qwen3.5/3.6 hybrid serve-side load — the family whose load path diverges from Gemma's today; this ADR makes both families speak the same surface).
  - **ADR-017** (KV-spill descriptor; `LoadInfo::kv_spill_active` mirrors `Engine::kv_spill_active()` once C2/C4 wire it).
- **Authoritative spec:** [`docs/research/model_load_ux_uniformity_2026-05-01.md`](research/model_load_ux_uniformity_2026-05-01.md). The migration plan in §5.7 of that document is the source of truth for commit boundaries; this ADR records the architectural decision to adopt it.

## Engineering Mantra (load-bearing)

> **DO NOT BE LAZY. We have plenty of time to do it right. No short cuts. Never make assumptions. Always dive deep and ensure you know the problem you're solving. Make use of search as needed. Measure 3x, cut once. No fallback. No stub (todo later) code. Just pure excellence, done the right way the entire time. Also recall Chesterton's fence; always understand current fully before changing it.**

C1 is a relocation, not an invention. The mantra-relevant constraints below are concrete:

- **No stub code.** `infer_quant_label` and `compute_bpw` are real bodies, exercised by ≥6 tests in the C1 commit. There are no `todo!()` markers; the design doc's `print_banner` / `emit_tracing` placeholders that say "filled in by commit C2" are deferred to C2 and are explicitly NOT in C1's scope.
- **Chesterton's fence.** Two byte-identical 27-LOC `infer_quant_type_from_gguf` bodies shipped at `engine.rs:3148-3174` and `engine_qwen35.rs:246-272` prior to C1. The author of the second copy explicitly noted at `engine_qwen35.rs:240-245` that "a refactor would touch a load-bearing file beyond iter-215's scope." That fence is now removed by ADR — iter-215's scope was the family split itself, and the ledger entry says explicitly this is the right time for the relocation.
- **Plenty of time.** C1 ships only the type-and-helper foundation. C2/C3/C4/C5 land in subsequent commits; each is independently revertable; each is compilable.

## Context

`hf2q`'s SERVE-mode and CLI-generate paths construct two different `LoadedModel` variants today (`GemmaLoadedModel` and `Qwen35LoadedModel`, post-iter-215 split). Each variant emits its own subset of load facts at its own time, in its own format:

| Surface | Gemma | Qwen35 |
| --- | --- | --- |
| CLI banner (3-line `print_header_top` + `print_header_prefill`) | yes | yes (slightly different shape) |
| SERVE-mode banner | none (silent — operator sees only `tracing` output) | none |
| `LoadProgress::on_layer` `\r loading i/n` | yes | no (Qwen35 just runs silently for the load duration) |
| `LoadedModel::provenance()` | populated | always returns `External` |
| `infer_quant_type_from_gguf` | private fn at engine.rs:3148-3174 | byte-identical private fn at engine_qwen35.rs:246-272 |
| `/v1/models` `quantization` field | reads via the engine.rs copy | reads via the engine.rs copy (already shared by handler) |
| Vision projector pairing | `VisionPairing` struct | not supported (501) |

The divergence shows up as:

1. Operators get different load output for different models, and have to remember which arch surfaces what.
2. The `infer_quant_type_from_gguf` body has been duplicated; any future change (e.g. adding BPW to the histogram step) has to be edited in two places under penalty of silent drift.
3. Structured tracing emits different field names per arch, so `journalctl -u hf2q | jq` cross-arch filtering doesn't compose.
4. `/v1/models` synthesizes its `quantization` field from a hand-written subset of facts rather than reading a shared snapshot.

## Decision

Adopt the strangler-fig migration laid out in design doc §5.7. The migration introduces a single `src/serve/load_info` module that owns:

- A `LoadInfo` snapshot struct (29 fields per design doc §5.1) that captures every fact the unified banner, structured tracing, and `/v1/models` need.
- Per-arch `LoadInfoBuilder` impls (added in C2) that produce a `LoadInfo` from a successful load.
- `print_banner` + `emit_tracing` (added in C2) that render the snapshot uniformly.
- The two GGUF derivation helpers `infer_quant_label` and `compute_bpw` that the existing private `infer_quant_type_from_gguf` body becomes part of, plus a new `compute_bpw` helper for the unified BPW headline number.

The migration is split into 5 commits, **each independently compilable and independently revertable**:

- **C1** (this commit) — Introduce the module + types + 2 helpers + relocate the duplicated `infer_quant_type_from_gguf` body. Visible behaviour delta: zero.
- **C2** — Add `LoadInfoBuilder` trait + per-arch impls + `print_banner` + `emit_tracing`. Visible behaviour delta: zero (library code, not yet invoked).
- **C3** — Switch CLI generate paths to `print_banner`. Visible behaviour delta: every CLI generate emits the new banner.
- **C4** — Add SERVE-mode banner + `Engine::info()` + `LoadInfo` reachable from request handlers. Visible behaviour delta: SERVE on TTY prints the same banner CLI does.
- **C5** (optional) — `/v1/models` reads `LoadInfo` directly; retire legacy 2-line header.

The trait choice (one new trait, narrowly scoped) is per design doc §5.2: per-variant impls read variant-specific fields (`g.weights`, `q.model.cfg.moe`, etc.), so a free function would need to take the enum and the enum-arm match would itself be the duplication we're trying to remove.

## Migration Ledger

| Commit | Status | Hash | Notes |
| --- | --- | --- | --- |
| **C1** — module + helpers + ADR | **Implemented 2026-05-01** | `<commit-sha-pending>` | Zero behaviour delta. Helpers tested via 6 synthetic-GGUF tests. Both legacy `infer_quant_type_from_gguf` bodies deleted. |
| C2 — `LoadInfoBuilder` + `print_banner` + `emit_tracing` | **Implemented 2026-05-01** | `<commit-sha-pending>` | Library only, not invoked. Qwen35 provenance field is populated and `LoadedModel::provenance()` reads it. |
| C3 — CLI generate switches to `print_banner` | not started | — | First user-visible delta. |
| C4 — SERVE-mode banner + `Engine::info()` | not started | — | TTY-gated; redirected stdout sees only tracing + bind line. |
| C5 (optional) — `/v1/models` consumes `LoadInfo` | not started | — | Retires legacy `print_header_top`. |

## Consequences

**Positive:**
- Single home for the GGUF-derivation logic. Future BPW improvements, quant-label refinements, and provenance fields live in one place.
- The `LoadInfo` snapshot is tracing-friendly (field names are stable; struct derives `Debug + Clone`) and lifecycle-friendly (no live model state owned).
- Removes a Chesterton fence that had been explicitly punted by iter-215.

**Negative:**
- One new file (~250 LOC at C2 completion; ~470 LOC at C1 today including tests).
- One new module-level dependency (`serve::load_info` references `serve::provenance::Provenance`); cycle-free.

**Neutral:**
- C1 itself adds ≥6 tests and removes ~54 LOC (two 27-LOC duplicate bodies); net code change at C1 boundary is +~470 LOC of which ~330 is tests + module docs.

## Verification

C1 acceptance:

```bash
# Workspace build (single bin crate, no workspace).
cargo build --release

# Test baseline (2425+ tests at HEAD).
cargo test --release

# Specifically the new tests.
cargo test --release load_info::tests

# Lint (ADR-018 should not introduce new clippy warnings).
cargo clippy --release -- -D warnings

# Verify the relocated helper agrees with the legacy body byte-for-byte
# (the `infer_quant_label_matches_legacy_body` test pins this).
cargo test --release infer_quant_label_matches_legacy_body
```

C2 / C3 / C4 / C5 verification will be added under the same heading in subsequent commits.

## References

- Design doc: [`docs/research/model_load_ux_uniformity_2026-05-01.md`](research/model_load_ux_uniformity_2026-05-01.md) (specifically §5.1 type spec, §5.2 trait rationale, §5.3 wiring, §5.6 test plan, §5.7 migration plan).
- Pre-C1 duplicated bodies (deleted in C1): `src/serve/api/engine.rs:3148-3174` and `src/serve/api/engine_qwen35.rs:246-272` at parent commit.
- New module home: `src/serve/load_info.rs`.
