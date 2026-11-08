<!-- Archived model response. Only machine-specific checkout paths were normalized. Raw response SHA256: 3d436dc7588cea7c4e984d2af4ded1ac8bca71532abe90c4fe689872b930ee8f -->

# Final bounded re-review — revised MLX-affine/DWQ retirement plan
Baseline unchanged: worktree @ `cca6425860c1fb770c6223c2bbdfa5d2d769dc70`, read-only. I spot-checked only the newly added/changed claims (not the broad audit): `tests/quality_thresholds.rs` existence, `cmd_smoke` behavior, the arch closure targets, `tests/common/mod.rs` exports, and `max_median_kl`/`median_kl` consumers.

## Verdict: READY WITH NONBLOCKING NOTES (for later execution; plan-level only — no execution or publication approved)

## B1–B4 disposition

**B1 — CLOSED.** §4 "Four legacy workflow scripts" now explicitly deletes `scripts/peer_parity_run.sh` with the shipping-contract links, and characterizes it accurately as a standalone external-peer P10 timing wrapper, not a caller of the Rust test binary — consistent with source (it invokes peer binaries directly). No contradiction with the preserved matched-reference gate tooling or corpus rows.

**B2 — CLOSED.** §4 diagnostic row and §5 now name **both** wire fields (`engine_config.dwq_overlay` from `control.rs::engine_config_identity_json`, emitted in the control response and measurement snapshot; `active_controls.dwq_overlay` from `measurement.rs`), require both to stay literal-false v1 tombstones, name the zero-dose identity test, loaded snapshots, and grammar-probe validators, and state "deleting either while leaving schema v1 unchanged is a regression." This matches the verified source exactly (control.rs:139/184, measurement.rs:104/120/160).

**B3 — CLOSED.** The new arch row retires the `dwq*` Skipped route, `QualityReport`, `QualityThresholds`, `ArchEntry.quality_thresholds`, entry initializers, printing/re-exports, constant-pinning tests, and `tests/quality_thresholds.rs`, while fencing the same-named `src/quality` types. I verified: all closure targets exist (`registry.rs:23-35,72,29`; `mod.rs:29` re-export; `entries/qwen35.rs:215`, `entries/qwen35moe.rs:239`; `smoke.rs:799-810` printing; `tests/quality_thresholds.rs` is indeed a source-string drift test). The new RCA claim is accurate: `cmd_smoke` lowercases a free-form quant via `normalize_quant_label` without `QuantSelector` (`main.rs:1616`) and treats `Skipped` as success (`main.rs:1631-1636`). The replacement contract (selector-based rejection before dry-run/preflight/external work, nonzero rejection tests, DWQ-named GGUF operands still valid, completion negative assertion kept as "unsupported") is coherent with the `quant_selector.rs` row and §7's new Smoke/completion row. R5's forbidden-symbol list now includes the qualified arch DWQ gate symbols.

**B4 — CLOSED.** Explicit decision: delete **both** peer wrappers plus the exclusive `metrics`/`env_lock` closure and exports; preserve `child_guard`, `serve_driver`, the independent `lcp_registry_unit` lock, PPL/corpus consumers; recheck on execution main with a named-consumer escape hatch. Verified against `tests/common/mod.rs:43-48` (exports are exactly `child_guard, env_lock, llama_cpp_runner, metrics, mlx_lm_runner, serve_driver` — the delete/preserve split matches). R3 is consistent: no relocation of wrapper tests merely to preserve dead wrappers.

## Contradiction check
None found. The new §3 rows (971-line wrapper/helper closure + threshold test; 65-line arch cohort) are labeled whole-file context/separate cohorts, do not overlap the 2,890 selected spans, and are not presented as deletion or performance estimates. §2/§3/§4/§5/R3/R5/§7 are internally consistent, and the preserved sets (ppl_driver comments-only sweep, corpus fixtures, receipts, `measured_auto_quant` vocabulary, DWQ-named GGUF fixtures, cache proof-or-epoch as a blocking condition) agree across sections.

## Nonblocking notes
1. `QualityThresholds` also carries `max_median_kl` (0.02 nats, ADR-012) alongside the two DWQ ratio fields. I verified its only machine consumers are inside the retired cohort (`conformance.rs` `QualityReport`, `smoke.rs` printing, registry validation, the drift test), so deletion is safe — but R4's doc sweep should also catch any current docs presenting `max_median_kl`/`ppl_ratio_dwq*` as live quality gates, so no active doc references a deleted threshold.
2. The row's framing "unimplemented DWQ gate closure" describes a mixed struct; harmless given note 1's sweep, but the executor should not be surprised that the KL bound retires with it.
3. The plan's own mandated execution-time rechecks remain the real gate and are correctly retained: R0 rebase/caller refresh, §5 cache proof-or-epoch (blocking), R3 smoke-dispatch trace, tests/common and corpus-fetcher closure recheck on execution main, R5 guard self-test, R6 hardware acceptance.

**Limitations:** read/grep only; no builds, tests, Metal runs, or hash recomputation; spot-checks limited to the revised claims above per your instruction. This verdict approves the plan for later execution readiness only — it authorizes neither implementation nor publication.
