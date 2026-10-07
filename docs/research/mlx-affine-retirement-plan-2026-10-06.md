# MLX-affine/DWQ retirement: RCA and execution plan

**Status:** proposed cleanup; implementation NOT started. Requested 2026-10-06.
This is a plan for later execution, not a report that the capability was removed.
**Source baseline:** hf2q `cca6425860c1fb770c6223c2bbdfa5d2d769dc70`
(main after PRs #197 and #198); published `mlx-native =0.15.1`.
The user requested a complete cleanup plan followed by GLM-5.3 review.
No performance improvement, release tag, deadline, or feature implementation is
promised. The decision proposed here retires the legacy overlay and shelves
ADR-046 Phase C. It does not retire MLX inference or all of ADR-046.

## 1. Outcome and scope

hf2q continues to own Rust HF-source-to-GGUF conversion and inference through
mlx-native. Remove the incomplete MLX-affine/DWQ product surface, its isolated
implementation, unsupported future-feature promises, and obsolete validation
scaffolding. Keep current conversion, model support, source-teacher evidence,
GLP calibration, imatrix, serving, quality measurement, and cache correctness.

The reason is maintainability and an honest product contract. Research has not
established an incremental full-model hf2q speed benefit from affine. That is
not evidence that affine can never help. Reopening it requires a named workload,
a measured limitation in current hf2q, and a bounded experiment against the same
runtime's GGUF path. Do not retain unused runtime code in anticipation of that.

The executable finish line is:

1. No CLI, engine option, loader, weight field, or dispatch branch can activate
   an affine overlay. No hidden flag, Cargo feature, inert setter, or no-op
   compatibility implementation replaces it.
2. Ordinary GGUF weight bytes, declared codecs, projection routes, supported
   families, and agentic serving behavior remain correct.
3. No current user guide or shipping gate promises a DWQ trainer, MLX affine
   output, or a retired conversion command. Historical records remain labeled.
4. Evidence schemas still reject unsupported encodings; diagnostic consumers
   and old receipts have an explicit compatibility policy.
5. The removal is measured, independently reviewed, tested on the affected
   hardware paths, and bound to the actual landing commit before being called done.

## 2. RCA: what happened and why it survived

### Timeline supported by Git and current source

| Event | Evidence | Consequence |
|---|---|---|
| May 7–8: affine loader/writer and DWQ overlay were introduced | `c41e3d69`, `9aa20761`, public serve root `4541a1e6` | This began as a concrete experimental trainer/consumer chain, not an empty new stub from the September research. |
| May 16: loader moved from calibration into core | `1396f769` | Generic float decoding/shard discovery and affine parsing were co-located; current conversion still needs the former. |
| May 19: legacy producer and trainer deleted | `ecea510cccfec51f06652a3a58d4f9f98b994036`; [P6 deletion audit](../adr/033-real-model-findings/2026-05-19-p6-deletion-plan.md), §§1, 4, 5.3 | The commit removed 96,494 physical lines across the old stack, including the writer, trainer and round-trip integration test. It explicitly retained the loader and migrated its drift helper to a surviving codec. Runtime/core were protected by the deletion boundary. Producer retirement did not retire the consumer. |
| Subsequent serving/model changes carried overlay options and fields forward | Current `EngineConfig`, `LoadOptions`, Gemma/Qwen loaders and dispatch | A public root kept the old subsystem reachable. Native Qwen admission now rejects several of its legacy mutation routes, leaving partial and divergent support. |
| August: warning-free compile boundary established | [ADR-048](../adr/ADR-048-warning-free-release-boundary.md) | This correctly handles release-unreachable research. It cannot identify an unwanted feature with a still-reachable CLI root. Its later source-teacher promotion is useful, separate work. |
| September 23: new affine plan documented | `69f2ea6a` / PR #197: README +2/-2, ADR +334/-21, research +490 | Docs only: +826/-23, net +803 lines. No Rust, shader, test, dependency, or CI change in that PR. It made Phase C a substantial future plan before its incremental product benefit was established. |
| October 6: benefit challenged; retirement requested | This plan and [benefit research](mlx-affine-benefit-evidence-2026-10-06.md) | No matched full-model hf2q affine win was established. Cleanup is justified by unsupported surface/maintenance debt, not an invented performance result. |

### Causal chain

The retirement unit was a set of producer modules, rather than the complete
operator capability. The consumer had a CLI root and shared a module with
generic source readers, so reachability-based deletion preserved it. Later
constructor changes, test fixtures, diagnostics, and dispatch code inherited
its optional fields. Documentation and old gate scaffolds were not reconciled
to producer removal. A later source audit then treated the surviving pieces as
a starting point for a new feature plan before establishing why that feature
was worth building.

These are source/history conclusions. There is no evidence here that the
original implementer deliberately intended misleading documentation. The
planning error in the recent discussion was advancing benefit and acceptance
expectations before a matched measurement; that must not be turned into a
claim that removing these lines will speed inference up.

### Why the checks missed it

- A warning-free build proves compiler reachability, not accepted product need.
- Parser/round-trip/kernel tests prove narrow components, not complete artifact
  production, consistent prefill/decode, or multi-turn serving.
- `tests/peer_parity_gates.rs` has **21** `#[test]` attributes, of which **8**
  are ignored and assert `Verdict::NotMeasured`. The old header's 31-test count
  is stale. `run_cell` always supplies missing hf2q timing metrics and its
  path-included Qwen loader is a stub. These are not eight passing model gates.
- A typed `DwqReserved` error and its tests preserve a future promise without
  supplying an implementation.
- A separate smoke path recognizes `dwq*`, returns `Skipped` with a future-P9
  promise, and `cmd_smoke` treats Skipped as success (`src/arch/smoke.rs:380`,
  `src/main.rs:1633`). Threshold constants and source-string tests are not a
  working quality gate. Independent review found this additional root.
- Current shipping docs still describe deleted producers and commands. No
  retirement check bound those claims to current entrypoints and real gates.

### Observed risks versus unmeasured effects

The legacy Gemma overlay stores separate affine expert stacks. Batched prefill
can use them while ordinary decode/other paths retain GGML stacks. Engine
identity records overlay presence, and the audited persistent fingerprints do
not bind overlay contents. Those are source-level consistency/identity gaps,
documented in the [September audit](mlx-affine-support-2026-09-23.md), §4.
They were not reproduced as a GPU failure or cache leak in this investigation.
Removal should close the routes, not spend this project repairing affine.

Normal GGUF loads initialize affine fields to `None`. No slowdown, resident
memory increase, compiled-size cost, or build-time cost has been measured here.
Source counts cannot answer those questions. Affine math and useful mlx-native
kernels are not inherently stubs; the unwanted hf2q integration is the target.

## 3. Measured inventory and limitations

[inventory.json](affine-retirement-2026-10-06/inventory.json) records exact
commit, file hashes, named source spans, per-file totals, and discovery lists.
[measure.py](affine-retirement-2026-10-06/measure.py) reads committed Git bytes,
not a dirty checkout; it performs no build or model execution.

```sh
python3 docs/research/affine-retirement-2026-10-06/measure.py --repo . > /tmp/affine-inventory.json
cmp /tmp/affine-inventory.json docs/research/affine-retirement-2026-10-06/inventory.json
```

| Measurement | Count | Meaning |
|---|---:|---|
| Selected affine-only definitions/tests | **2,890 physical lines** | Non-overlapping, named spans in five Rust files. A conservative selected footprint, not the whole removal diff. |
| Loader's selected retirement spans | 1,189 | Affine types, packing, serialization, drift and their tests. |
| Loader's selected generic spans to keep | **151** | Two functions and three tests; imports/comments/module framing are additional. |
| Gemma overlay method | 326 | Excludes its fields, initialization, dispatch branches, imports and surrounding comments. |
| Qwen overlay method and private role/conversion helpers | 587 | Excludes its overlay tests, GPU dispatch and option plumbing. |
| Shared affine definitions/parser/test module | 778 | Excludes the mixed dispatcher branch, field edits and comments outside items. |
| Qwen affine stack attachment method | 10 | A small additional selected definition. |
| Old parity harness / MLX-LM helper | 1,710 / 353 | Whole-file context; all 8 ignored model cells are `NotMeasured` scaffolding. These retire with their exclusive helper closure below. |
| Original four script inventory | 951 total | 294 AC7 + 185 AC8 + 294 corpus fetcher + 178 P11 re-emit. The caller audit resolves the corpus fetcher to retirement. |
| Review-added wrapper/helper closure and threshold test | 971 total | 180-line `peer_parity_run.sh`, 513-line peer runner, 100-line metrics, 96-line env lock, 82-line threshold test. Whole-file context, no overlap with the rows above. |
| Review-added architecture definitions | 65 physical lines | Separate selected cohort: arch QualityReport + impl and QualityThresholds + impl; excludes smoke branches, registry field/entry edits and tests. |
| Rust files containing `affine` | 35 | Discovery count, not 35 disposable modules. |
| Source files containing `dwq[_-]overlay` | 28 | Includes receipts and diagnostic compatibility, not only execution. |
| Recent landed research/planning addition | 803 net lines | PR #197, three docs files only. |

Do not add overlapping context counts to selected spans and call that exact
deletable SLOC. The selected spans omit cross-cutting edits and some tests;
whole-file counts include comments and blank lines. The inventory's formatted
item delimiter scan is a reproducible physical-span counter, not a Rust parser
or proof of call-graph closure. At execution, re-run caller searches and compile
checks on the actual new base. Removed spans will intentionally stop matching;
retain the pinned baseline and produce a separate after-state receipt.

No single aggregate “pollution percentage” is useful: it would conflate
executable experiments, useful utilities, schemas, tests and historical docs.
The acceptance measurement is zero activation routes and zero false current
claims, plus reviewed diff/test counts and preserved product behavior.

## 4. Complete disposition map

Paths below are relative to repository root. Source spans are in the inventory.
New callers on a later main must be classified by behavior before removal.

| Surface | Action and boundary |
|---|---|
| `src/cli.rs` | Delete `ServeArgs::dwq_overlay` and its help. Remove the future DWQ promise in conversion help. Clap should reject `--dwq-overlay` before model/network/device work. |
| `src/convert/quant_selector.rs` | Remove `DwqReserved`, its special parser arm, and future-pipeline tests. `dwq` should use the existing unsupported-quant error and supported-value list. Keep a real CLI rejection test; no dormant trainer reservation. Leave unrelated unsupported-name policies alone. |
| `src/arch/{smoke.rs,conformance.rs,registry.rs,mod.rs}`, `src/arch/entries/{qwen35.rs,qwen35moe.rs}`, `tests/quality_thresholds.rs` | Retire the `dwq*` Skipped/future-P9 route, `arch::conformance::QualityReport` and its tests, `arch::registry::QualityThresholds`, `ArchEntry.quality_thresholds`, entry initializers, threshold printing/re-exports and constant-pinning tests. This is the unimplemented DWQ gate closure. Keep tensor catalogs, generic conformance/metadata checks, EvalCorpus and working smoke behavior. Similarly named `src/quality` report/regression types are a different module and are not deletion targets. |
| `src/main.rs::cmd_smoke`, `src/cli.rs::SmokeArgs`, `src/cli/complete.rs` | `cmd_smoke` currently lowercases a free-form quant string without calling `QuantSelector`; reject unsupported DWQ quant selections before dry-run, download, conversion or peer execution, using the supported selector contract. Replace the smoke-Skipped test with nonzero rejection tests (normal and dry-run); valid existing GGUF filenames containing DWQ remain allowed. Keep the negative completion assertion that `dwq` is never advertised, but describe it as unsupported rather than reserved. Preserve unrelated smoke skip policies. |
| `src/serve/{mod.rs,multi_model.rs}`, `src/serve/api/{engine.rs,engine_qwen35.rs,state.rs}` | Remove path options, constructor plumbing, debug output, identity boolean, overlay load calls and overlay-specific activation inheritance tests. Preserve identity and inheritance checks for real controls such as GLP/grammar/vision. |
| `src/serve/forward_mlx_shared.rs` | Remove `MlxAffineExtra`, `MlxAffineMoeStack`, `MlxQWeight.affine`, its clone/constructor logic, fake-F32-sentinel route, overlay role parsers and affine test module. Preserve `MlxQWeight`, native GGML/F32/BF16 dispatch, head-major routing, norms, cosine tests and ordinary storage behavior. |
| `src/inference/models/gemma4/model.rs`, `src/serve/forward_prefill_batched.rs` | Remove `apply_dwq_overlay`, affine expert fields and initialization, and both affine MoE prefill branches. Keep the former no-overlay GGML branch byte-for-byte where possible, including batching/arena/evidence behavior. Do not rewrite forward math. |
| `src/inference/models/qwen35/{model.rs,gpu_ffn.rs}` | Remove overlay loader, `AttnRole`/`DenseFfnRole`, DWQ transpose/requantization/overwrite helpers and tests, affine expert fields/setter and branch parameters/call arguments. Preserve synthetic F32 and ordinary GGUF paths that have independent tests/callers. |
| `src/core/mlx_safetensors_loader.rs`, `src/core/mod.rs` | First extract `read_floats_to_f32` and `discover_shards` with their three tests into a neutrally named focused core module (proposed `safetensors_source.rs`). Then remove the affine module. Preserve exact dtype, ordering, error, sharded-index and single-file sentinel semantics. Do not merge the distinct `src/input/safetensors.rs` discovery API in this cleanup. |
| `src/convert/{source_reader.rs,evidence_source.rs,source_dtype/mod.rs}` | Point generic imports/docs to the extracted module. This also protects FP8 scale decoding and source-teacher evidence input. No algorithm or numeric change. |
| Other `MlxQWeight` constructors/guards | Remove now-impossible fields/checks in Gemma/Qwen native storage, forward/attention/profile, MTP, BERT, vision, EAGLE3 and native projection files listed by the inventory. Preserve all codec/storage/shape guards unrelated to affine. |
| Load-option constructors and evidence wrappers | Update `src/quantize/imatrix/forward.rs`, `src/serve/kv_persist/loader_wrapper.rs`, engine tests, Qwen weight loader/execution config/execution evidence and source-teacher tests. Eliminate overlay-path parameters without removing source-precision/copy-evidence isolation. |
| Diagnostic JSON: `src/serve/api/{control.rs,measurement.rs}` | Remove executable identity state but preserve **both** `engine_config.dwq_overlay: false` and `active_controls.dwq_overlay: false` as v1 wire tombstones. `control.rs::engine_config_identity_json` supplies the first field to runtime control output and the measurement snapshot; `measurement.rs` emits the second. Test both endpoint representations and both fields, including the existing zero-dose identity test, loaded snapshots and grammar-probe validators. Do not silently omit either field. |
| Serialized recipes / tensor lineage / execution manifests | Preserve historical schema vocabulary and explicit non-admission (`WeightEncoding::MlxAffine`, `dwq_overlay_sha256`, offline algorithm identifiers) where existing receipt hashes/decoders use it. Explicitly include `src/intelligence/measured_auto_quant{.rs,/types.rs,/tests.rs}`, tensor-execution provenance, tensor-lineage receipts, exact-teacher comparison and Dynamic producer/solver tests. Mark it unsupported/retired, not forthcoming; remove only construction scaffolding exclusive to the retired runtime. |
| `tests/peer_parity_gates.rs` | Retire the obsolete P10 experiment harness, duplicated Qwen/progress type stubs, frozen 8-cell product grid and `NotMeasured` pseudo-gates. This includes six non-affine cells from the same nonfunctional harness; it does not delete working GGUF parity gates. Internal table/ratio tests describe a retired contract, not current product coverage. Do not create a replacement fake model adapter. |
| `tests/common/{mlx_lm_runner.rs,llama_cpp_runner.rs,metrics.rs,env_lock.rs,mod.rs}` | Delete **both** obsolete peer wrappers, their exclusive RunMetrics/env-lock helpers and exports. At the pinned base, remaining apparent consumers outside these files/P10 are comments or unrelated same-named local functions. Do not keep caller-less utilities merely to relocate their own tests. Preserve `child_guard`, `serve_driver`, the separate lock in `tests/lcp_registry_unit.rs`, current PPL/quality code and corpus fixtures. Recheck this closure on execution main; if a real new consumer exists, preserve its minimal generic helper and name that consumer in the receipt. |
| Four legacy workflow scripts | Delete `scripts/adr020_ac7_boundary_validate.sh`, `scripts/adr020_ac8_validate.sh`, `scripts/p11_re_emit_dwq.sh`, **`scripts/peer_parity_run.sh`** and their current shipping-contract links. The first three require removed interfaces. The fourth is a standalone external-peer P10 timing wrapper, not a caller of the Rust test binary; retire it with the obsolete experiment rather than retain an unowned generic benchmark. Keep current matched-reference gate tooling. |
| ADR020 corpus fetcher | **Delete** `scripts/adr020_fetch_wikitext_calibration_jsonl.sh`: the pinned whole-tree caller search found no references outside the script itself, and its advertised consumer is the retired DWQ trainer. Recheck on execution main. Preserve the independently used `scripts/fetch_wikitext2.sh`, corpora, caches and model files. |
| `README.md`, `docs/ARCHITECTURE.md` | Correct output claims, diagrams and module tables. Remove Phase C as an active product promise. Describe current `src/calibrate/` as GLP direction derivation, not DWQ training/autograd. |
| `docs/shipping-contract.md` | Remove/replace the stale activation-DWQ subsection, P10/P11 8-cell appendix, deleted calibrator cross-validation contract and dead test references. Preserve current conversion/serving/model/release gates at the beginning of the file. Separate historical measured results from current requirements. |
| `docs/converting-qwen35.md` | Rewrite current command examples from the actual CLI (including dense/MoE, MTP and sidecar constraints), remove dead DWQ flags/env switches/safetensors twins/P11 instructions. Keep useful current conversion/metadata advice only after source verification. |
| `docs/calibrator-onboarding.md` | Retire this guide to a deleted trait. Keep a short notice at the old path pointing to current imatrix/source-teacher/GLP guidance as appropriate; preserve the former body in Git or a clearly historical archive only if needed for ADR links. Do not recreate the removed Calibrator abstraction. |
| ADR-046 / ADR indexes / research | Amend Phase C to “deferred; legacy overlay retired by <landing>” only when removal lands. Remove active C dependency arrows/promises for the shelved initiative; retain the useful GGUF/evidence decisions and acceptance receipts. Mark September implementation-handoff research and the October benefit draft as historical, not execution instructions. Keep original observations and source identities. |
| ADR-020 / ADR-033 / ADR-038 / ADR-048 | Preserve chronology and superseded status. Add a brief dated retirement cross-reference where an otherwise-current entry suggests overlay support. Correct architectural maps; do not rewrite past validation as if it never happened. Reindex memory/ADR graph only after canonical Git docs change. |
| `mlx-native` and `Cargo.*` | Keep the published dependency and its useful backend/kernels. No native-repo deletion or downgrade in this plan. Adjust obsolete Cargo comments only if they imply hf2q uses affine. Other consumers and independent crate APIs require a separate retirement analysis. Preserve the rustls fix. |
| Local drafts, worktrees, model files, pinned release fixtures | Preserve. They are not repository cleanup targets. In particular, do not rewrite `scripts/fixtures/deepseek4-agentic-repo-context.txt`: it is pinned acceptance input even if its historic text mentions retired code. Update only live navigation around historical fixtures. |

The stale-reference sweep also includes `src/quality/ppl_driver.rs`,
`tests/{p0_apex_moe_lazy_iter.rs,qwen3vl_streaming_e2e.rs}`, the corpus fixture
README, `src/intelligence/heuristics.rs`, `src/serve/sampler_pure.rs` and
comments in `scripts/reconvert-broken-2026-05-05.sh`. The latter references
P11 in comments; it does not invoke the retiring script. Rewrite those comments
without deleting its separate workflow. Retire helper comments together with
the helper files. Update the template assertion in `src/serve/api/state.rs`
when removing the overlay field, preserving its remaining control checks.

Examples of explicitly retained DWQ-named GGUF fixtures include tokenizer
references in `src/core/chat_templates.rs` and `src/serve/api/registry.rs`,
ordinary model examples in `src/serve/api/handlers.rs`, Qwen forward tests,
and `.github/workflows/release-check.yml`. Their names are not quantizer
support promises. Preserve fixture bytes and applicable hashes.

## 5. Compatibility and cache decisions

**CLI:** intentional removal, not indefinite deprecation. Old overlay commands
fail at parse time with a nonzero exit and ordinary unknown-option diagnostic.
`convert --quant dwq` fails through normal unsupported-quant validation, before
tensor conversion. Release notes say there is no supported full-model affine
replacement. Never translate to another quantizer or silently ignore the option.

**Diagnostic JSON:** keep v1 constant-false tombstones, with exact-path tests and
comments that they do not represent configurable capability. Current grammar
probes reject missing or true overlay flags; preserve that fail-closed behavior,
including against older servers that actually have an active overlay. Removing
these wire fields belongs to a separately versioned API migration, not this
retirement. This explicitly allowed residue is bounded and has a real consumer.
Both `engine_config.dwq_overlay` (control response and measurement snapshot)
and `active_controls.dwq_overlay` (measurement snapshot) remain present and
false; deleting either while leaving schema v1 unchanged is a regression.

**Receipts:** preserve serialized field names, canonical hashes and typed
rejection of old affine/overlay receipts. Do not silently accept affine as GGUF,
recalculate stored evidence, change historical artifact hashes, or weaken the
verifier to make deleted tests pass. Unused constructors are removable; format
vocabulary with negative-admission consumers is not a runnable stub.

**Persistent cache:** removal of a flag alone does not prove pre-retirement
cache contents are safe. Trace both Qwen and Gemma fingerprint construction,
block store and prompt/LCP restoration paths. The old audit found no overlay
content identity. Require either proof that caches generated with overlays were
never persisted/admitted, or an intentional retirement epoch in affected family
cache identity so pre-retirement artifacts cannot be restored. Cover both the
block-store and prompt/LCP namespaces and their fallback restoration paths. Prefer a bounded
family identity epoch over unrelated global cache-format churn. If the old
identity cannot distinguish overlay/plain artifacts, all old caches for those
families must miss once. Record that operational cost, preserve files on disk,
and verify fresh cache reuse after restart. Do not claim a cache leak occurred.
This proof or invalidation is a blocking delivery condition, not optional polish.

**Rollback:** a code revert must revert the complete atomic runtime slice;
never restore only a flag/loader. New and old cache namespaces must not mix.
The first response to a regression is to hold publication or repair the ordinary
GGUF path; do not reopen an unqualified overlay as a workaround. No data deletion
or irreversible receipt migration is needed.

## 6. Ordered work packages for later execution

These are reviewable execution boundaries, not created GitHub tasks. No package
has started. An executing agent must first bind these to approved AWA work keys.

### R0 — Freeze the retirement contract and refresh evidence

Rebase the inventory onto the actual current main in an isolated clean worktree.
Read AGENTS, `ak status`, applicable ADRs and live release coordination before
build/model work. Compare this plan's exact source commit and caller inventory
to that main; record changed/new callers and existing operator scripts. Check
GitHub with `robertelee78` for any new affine issues/project/release assignment.

Acceptance: every live root and retained exception in §4 has a disposition;
source/model/native identities for later parity are recorded; no unrelated dirty
checkout changes are included. Confirm cache retirement policy in §5 against
current source before runtime edits. No speedup target is invented.

### R1 — Extract generic safetensors input utilities

Depends on R0. Move the two helpers, imports and three tests without altering
behavior; update production conversion/evidence callers. Add only missing
meaningful malformed-dtype/index tests needed to preserve the boundary.
Run focused source-reader/evidence-source tests and real existing fixture
conversion tests. Compare fixed fixture output tensors/metadata to the baseline;
compare complete bytes where the format is deterministic.

Acceptance: generic conversion no longer imports the affine module, existing
source semantics and performance-oriented Rayon implementation remain intact,
and a locked check passes. This can be a standalone reviewable commit; it is
preparation, not completion of retirement.

### R2 — Remove the complete runtime capability atomically

Depends on R1. Delete CLI/config roots, model overlays, shared affine types and
dispatch branches, and then the affine loader. Update all constructors and
source/evidence option wrappers in §4 in the same buildable slice. Remove dead
tests only when they exercise removed behavior; preserve or retarget tests
protecting ordinary GGUF storage, tied heads, routing and activation inheritance.
Keep the former ordinary branches unchanged and do not mix in optimizations.

Implement v1 JSON tombstones and cache proof/epoch from §5 together with their
tests. Retain rejection-only serialized vocabulary explicitly. Review changes
to Gemma/Qwen pooled/nonpooled MoE routes, serial/batched prefill and width-N;
do not assume deleting a field mechanically proves equivalent dispatch.

Acceptance: negative CLI tests prove rejection before load; no executable
activation route remains; locked checks and model-free regression pass; cache
admission tests cannot restore unqualified pre-retirement state; diagnostic
consumers remain compatible. Hardware acceptance remains required at R6.

### R3 — Retire producer reservations and obsolete gate scaffolding

Depends on R2 for the final integration (can be prepared after R0). Remove
`DwqReserved` and the future promise; update CLI diagnostics tests. Retire the
separate architecture DWQ smoke/quality scaffold and the P10 experiment,
**both** exclusive peer wrappers, metrics/env-lock closure, five workflow scripts
and associated exports. The decision is deletion of the caller-less P10 utility
closure, not moving its tests to preserve it as future infrastructure. Current
corpus/PPL consumers keep their independent checks; if a new execution-base
caller changes ownership, name it and preserve only its necessary helper and
meaningful tests. Log which tests moved, were redundant, or were retired.

Trace smoke dispatch through `cmd_smoke`: unsupported DWQ selections must
fail before dry-run or preflight can report success/skip. Retire the architecture
threshold source-string tests without changing independent real quality gates.
Keep completion's unsupported-DWQ negative assertion. Do not retain matrix/table
tests for a retired product grid solely to preserve a test count.

Audit Cargo explicit test entries, `.github`, script callers, source includes
and docs. Preserve working GGUF parity/conversion gates and shared corpus tools.
The current caller audit resolves the ADR020 corpus fetcher to deletion;
reconfirm that fact on execution main.

Acceptance: no runnable script calls removed commands; no “hardware gate”
passes by asserting `NotMeasured`; genuine current model gates remain fail-closed
when missing inputs. The reduced test count is explained, not called lost
product coverage or hidden by reclassifying a stub as a benchmark.

### R4 — Reconcile active docs and ADR lifecycle

Depends on R2/R3 for factual completion language. Apply all doc dispositions
in §4, correct README paragraph/table/diagram/module map together, and remove
dead links from active guidance. Preserve historical records and fixture bytes.
Use one concise retirement note linking this RCA, the landed implementation,
the before/after inventory and acceptance proof; avoid a new speculative roadmap.

Amend ADR-046 only in the affine scope. Keep the accepted source-teacher and
GGUF evidence contracts. Do not mark all ADR-046 rejected or complete. An index
or memory record must agree with the canonical Git text, not overwrite it.

Acceptance: a reader can determine supported commands without interpreting
superseded diaries; active docs have no trainer/output/gate claims contradicted
by source; current examples parse and use supported flags; historical sections
have clear status. Check relative links and the existing peer-reference ratchet
without lowering its baseline to evade new failures.

### R5 — Add a narrow retirement regression guard

Depends on R2–R4. Use a small explicit inventory of forbidden production
symbols/options (overlay loader/constructor/stack types, path options, affine
dispatch call, old loader module and qualified architecture DWQ gate symbols),
with exact-path exceptions for serialized
receipt rejection, v1 wire tombstones, negative CLI tests and historical docs.
Check active script invocations and doc capability examples as well. Prefer
one focused CI check over pervasive mock tests or a new framework.

Do not ban the word `affine`, `DWQ`, safetensors, MLX, or old model filenames
across the whole repo. The guard is a backstop to review/compilation; comments
alone are not a source call graph. Prove the guard fails when a forbidden root
or obsolete active-doc claim is inserted in a temporary fixture/copy, then
passes on the candidate. Preserve diagnostic/receipt exception tests.

Acceptance: no feature can return accidentally via a constructor/default or a
copy-pasted operator guide; the guard does not reject frozen release fixtures,
valid GGUF fixtures named DWQ, legitimate calibration, or archived research.

### R6 — Prove preservation and measure the actual removal

Depends on R1–R5. Complete §7 on the exact candidate. Report `git diff --numstat`,
the removed runtime roots/fields/branches and test disposition ledger. Record
remaining allowed occurrences with paths and reasons. Separately record whether
binary-size/build-time/RSS/performance comparisons were actually run.

Acceptance: supported inference/conversion and real agentic behavior pass;
old caches are safe by proof or forced miss; no guessed performance result;
no publishable claim based on tests that skipped Metal. Independent review has
no unresolved blocker and the source/receipts all describe the same candidate.

### R7 — Deliver and reconcile handoff

Depends on R6 and authorization to execute/publish. Use focused conventional
commits/PRs with AWA work identity and reviewed scope. Recheck the final merged
candidate/source-bound proofs according to repo policy. Record the actual
landing SHA, native pin, CI, hardware receipt, known one-time cache misses and
intentional CLI removal in release notes. The release milestone remains
`next` until the operator chooses a tag; no fabricated due date.

If prior affine implementation work items exist by then, record the user's
decision and mark only those exact items Won't do; do not alter unrelated
ADR-059/060 work. ADR cleanup delivered by merge and runtime cleanup delivered
by release must have their actual distinct delivery boundaries. Reconcile
AWA/ADR index/memory after landing. A local green branch is not publication.

## 7. Validation matrix

The following are requirements for future implementation, **not tests run by
this planning pass**. Resolve test names against the execution base and ensure
each command executes a nonzero number of the intended tests.

| Area | Required proof |
|---|---|
| Source utility extraction | F32/F16/BF16 parity, invalid byte-length/type behavior, indexed/single-file discovery, conversion/source-evidence fixture parity; FP8 scale decoding still reaches the same helper. |
| CLI | `serve --help` omits overlay; old flag fails before model/device/network work; `convert --quant dwq` fails unsupported without promising future delivery or falling back. Supported conversion/help examples still work. |
| Smoke/completion | Unsupported `dwq`/`dwq-*` smoke selectors fail nonzero in normal and dry-run paths before external work; working supported selectors and GGUF operands (including old DWQ-named files) remain valid; completion never advertises unsupported DWQ. No threshold-printing or Skipped-as-success DWQ gate survives. |
| Model storage and compute | Existing Gemma/Qwen native storage and tied-head tests; representative dense and MoE matmul/gather paths, pooled/unpooled and ordinary BF16/F32/GGML routes. Constructors for BERT/vision/MTP/EAGLE3 still compile and retain their independent tests. |
| Diagnostic and evidence compatibility | Snapshot/control JSON preserves v1 literal false; Python grammar-probe unit suite passes and rejects old active-overlay servers; no receipt hash/schema drift; affine/non-null-overlay manifests still fail admission. |
| Cache | Prove non-admission or epoch separation for prior potentially overlaid Qwen/Gemma artifacts; plain current-model fresh/reused and restart reuse pass; mismatched identity rejected; no cross-conversation state leak. |
| Serving | Real Gemma and Qwen dense/MoE artifact coverage for affected branches; cold prefill, single-token decode, multi-token decode, batched prefill and width-N where supported; realistic multi-turn coding with native tool calls, tool-result continuation, unary/SSE, retained-prefix counts and first semantic token. Include Gemma MoE because that is the divergent legacy branch. |
| Quality/regression | Source-bound pre-cleanup no-overlay baseline and candidate use identical GGUFs/prompts/settings/hardware. Require established exact/quality tolerances from the governing family gates; do not invent new affine thresholds. |
| Current release gates | Locked check/build/tests and required CI activation matrix; relevant existing cache/model/structured-output gates. Keep unrelated release gate infrastructure/receipts intact. |

Baseline Rust commands:

```sh
cargo check --locked --all-targets --all-features
cargo build --release --locked
cargo test --locked
```

Use focused module/tests first (`convert::`, Gemma/Qwen native tests,
source-reader/evidence, provenance, cache and serving). Do not cite the deleted
`tests/convert_integration.rs` as a valid target. Run Python tests through the
existing grammar-probe test entrypoints. Inspect skip/ignore output: a successful
process with zero Metal cases is not hardware validation.

Performance is a **no-regression observation**, not a promised optimization.
During already-required matched serving gates, record cached tokens, prefill
and decode rates, first semantic token, quality and hardware/process conditions.
Use repeated matched medians and existing family bounds. If within measurement
noise, say so. Optional maintenance-cost evidence may compare release file bytes,
clean build wall time and idle/loaded memory under controlled conditions; do not
delay retirement to optimize those numbers or infer them from LOC.

Before any build/model run, consult release coordination. Run one large runtime
at a time on the 128 GiB host, and stop test servers. No model/corpus downloads,
native dependency publication, or model-file cleanup is needed to write this plan.

## 8. Risks, stop conditions, and prevention

- **Hidden consumers:** new main may add uses. Refresh the union of the two
  discovery lists plus symbol searches; stop a deletion that would remove a
  supported consumer, extract its generic dependency, and update the plan.
- **Semantic drift in dispatch:** deleting an `if` can change scratch allocation,
  range tracking or batching. Preserve the old no-overlay branch and prove it
  with the affected model gates; do not accept compile-only evidence.
- **Receipt/diagnostic drift:** preserve the explicit compatibility exceptions
  in §5. A constant wire declaration of absence is not a hidden runtime mode.
- **Cache ambiguity:** retirement is incomplete until old state cannot be
  mistaken for the current model. A bounded cold-cache migration is acceptable;
  unexplained reuse is not.
- **Scope expansion:** no generic “remove all stubs” campaign, no deletion of
  live GLP/calibration/source-teacher/Dynamic/GGUF work, no whole mlx-native
  cleanup. Independent debt may be recorded separately with its own evidence.
- **Historical claims resurfacing:** active docs link to a capability contract,
  historical docs carry status, and each future feature needs an accepted root,
  end-to-end proof and a measured reason before staged code becomes product.

The retirement review must check the whole capability: producer, consumer,
CLI/API/serialized fields, cache identity, tests, scripts, docs, ADR status and
tracker. Compiler warnings and keyword totals remain useful tools; neither is
the retirement contract.

## 9. Planning handoff and review

This packet includes the pinned measurement script/JSON and benefit research;
it makes no product-code edits. Earlier local benefit/ADR drafts are unpublished
and must not be mistaken for this retirement decision or blindly merged.
The shared working checkout has unrelated changes; preserve them. No worktrees,
models, receipts, databases or caches were deleted during this investigation.

AWA discovery found an opted-in repository with unrelated managed work. No
unambiguous affine work key, affine Project, or release milestone was present
in the read-only check. Leave existing phases unchanged; do not invent a cleanup
attempt, completed story or release assignment. Task creation remains after
readiness and the user's scope decision.

GLM-5.3 must review this exact plan plus the source-bound inventory. Ask it to
identify unsafe deletions, missing consumers, schema/cache hazards, fake gates,
scope creep, weak acceptance and anything still blocking execution. Preserve
its model/session identity, plan hash, findings and disposition, then revise
the plan and obtain a final verdict on remaining blockers. Model agreement is
a planning review, not implementation or hardware acceptance.
