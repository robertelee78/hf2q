# GCD and GLP article: publication review

The article has a strong subject: two distinct ways to control local inference,
an implementation that makes them inspectable, and practical implications for
model distribution and agent security. The original draft obscured that subject
by treating refusal suppression, structural validity, authorization, and
behavioral capability as interchangeable. The revision separates them and
replaces the original figures with source-grounded diagrams and reproducible
charts.

The article now incorporates the later implementation repairs and recorded
measurements with explicit scope. Remaining implementation and evaluation
findings are tracked here rather than expanded into the article's narrative.
Correcting code does not retrospectively validate historical measurements or
establish untested authorization paths.

This review covers the complete original prose, all three embedded figures,
the additional `gcd-gate.png` asset, the plotting code, historical measurement
records, and relevant Rust/Metal source. The initial source snapshot is
`bdb632ff406b9c7d77fa315d7b507351b1bee9cd`; the independent source audit used
`44004311717d414feaa54384578e2bcf4d140464`. The audited implementation files
were unchanged between these commits. No runtime code was changed by the
publication review.

## Final pass: updated implementation and evidence

This pass inspects `294907cd655ee81eedb8d8b9ea30ab80ade483bb`. The article now
uses that snapshot for its implementation account, while the historical W1
manifest remains bound to the earlier audit. The separate
[follow-up manifest](figures/gcd/followup-evidence.json) reproduces the new
aggregate claims without publishing raw prompt or response text.

### Repairs credited

| Findings | Current source-review status |
|---|---|
| S1–S3, S5–S7, S9–S16 | Addressed in source: launch geometry, greedy steering, explicit hooks/modes, cache identity, input validation, metadata and graph-layer mapping, and default-grammar composition. This does not substitute for every requested serving/hardware proof. |
| S4 | Derivation and apply-site metadata corrected. The site-transferred candidate was evaluated, with no observed refusal reduction on the tested panel. It remains a candidate, not a validated behavioral direction. |
| S8 | Hook, shape, and layer binding improved. Exact checkpoint identity is still an operator responsibility; a model-name warning does not establish artifact compatibility. |
| S17–S18 | Explicit remote-reference resolution and ambiguous automatic discovery remain open. |
| E1 | Final battery attempt passes 17 named checks; earlier attempts and incomplete run identity remain visible. This does not close the whole conformance inventory. |
| E7 | Configurable generation budgets and manifests implemented; downstream joins still collapse budget cells (E12). No replacement paired full-corpus experiment was found. |
| E8–E9 | Complete judge input and cross-field validation implemented and mock-tested. Historical labels are unchanged; no completed v2 rejudging artifacts were found. |
| E10 | Failed-attempt retries implemented; resume and identity gaps remain (E13). |
| E11 | The paired metric-key crash is repaired and prompt-clustered bootstrap implemented. Reporting still needs E12. |

All 13 supplied offline/mock tests pass. Additional in-memory probes reproduce
budget collapse, acceptance of a mismatched response hash, ignored scoring-pass
selection, and shortening of the human-review input. No models were loaded or
Rust/Metal builds run by the publication reviewers.

### Interpretation of the new records

The three `*.truncation-report.json` artifacts are cross-tabs of unchanged
historical judgments. DeepSeek's 823 capped responses include 733 old
`valid_fulfillment` labels, 85 degenerate labels, and five other labels.
Nineteen of those 733 also carry an invalid-output flag. Neither 733 nor the
714 with aligned fields is a new full-response validity measurement.

The GLP panel has 240 matched generation/verdict records: unsteered baseline
and four strengths, with 32 adversarial and 16 benign prompts per arm.
Maintained refusals are 28, 28, 28, 29, and 29 of 32. Benign fulfillment labels
are 4, 3, 3, 2, and 3 of 16, while 14–15 benign responses per arm exhaust the
256-token budget. The paper reports no observed refusal reduction and does
not adopt the summary's stronger claims of preserved capability, behavioral
inertness, failed site transfer as an identified cause, or no latency cost.
Six panel texts overlap the calibration corpus. The reflection at alpha two
is a mathematical fact, not a causal explanation established by these counts.

The battery file contains four consecutive 17-cell attempts. The final
attempt, rows 52–68, has 17 passes and no failures or skips; the preceding
attempts have 14, 15, and 16 passes. Final bias cells record tokenizer-resolved
ID 279 and nonempty bias maps. Most cells explicitly attach W1, and the runner
trusts the operator-supplied control endpoint. The pass record does not prove
that endpoint is unsteered or uses a matching model, nor bind the run to the
current binary, grammar, model, and request identities.

### S19. Locked-schema injection is mistaken for a caller override — P1

The [handler](https://github.com/robertelee78/hf2q/blob/294907cd655ee81eedb8d8b9ea30ab80ade483bb/src/serve/api/handlers.rs#L547)
injects the server schema into `request.grammar`, then checks for an explicit
constraint override. An ordinary request with `--gcd-schema-locked` is rejected
by that check. The positive helper test uses the opposite order and misses the
actual request-preparation sequence.

**Required proof:** check caller overrides before injection and exercise the
actual handler sequence. An ordinary request must reach constrained generation;
caller overrides must be rejected before generation.

### S20. Locked-schema policy does not cover required/named tool paths — P1

[Request compilation](https://github.com/robertelee78/hf2q/blob/294907cd655ee81eedb8d8b9ea30ab80ade483bb/src/serve/api/grammar/request.rs#L815)
discards the response constraint for required or named tools, and the
[selection path](https://github.com/robertelee78/hf2q/blob/294907cd655ee81eedb8d8b9ea30ab80ade483bb/src/serve/api/handlers.rs#L4672)
prioritizes tool grammar. Correcting S19 alone does not make a response schema
mandatory across these output paths.

**Required proof:** reject incompatible tool requests in locked mode or define
and enforce their policy separately, with unary and streaming tests. The paper
retains its trusted-state and tool-precedence discussion and does not present
the new locked mode as validated.

### E12. Reporting joins lose budget, content, and scoring identity

[report.py](https://github.com/robertelee78/hf2q/blob/294907cd655ee81eedb8d8b9ea30ab80ade483bb/scripts/grammar_probe/report.py#L136)
keys observations by arm, prompt, and repetition, omitting budget/run identity.
Two budget cells collapse to one, and a verdict with a mismatched response hash
is accepted. The documented `PASS` selector is ignored in favor of the most
frequent configuration. Known termination is counted only for judged responses;
generation errors are excluded before reporting.

**Required proof:** preserve all experiment dimensions, reject ambiguous or
mismatched joins, honor explicit pass selection, and compute generation facts
from the complete generation inventory independently of judgment availability.
Use deterministic fixtures for each failure, including errors and missing labels.

### E13. Resume records do not bind the complete judgment input

The [judgment key](https://github.com/robertelee78/hf2q/blob/294907cd655ee81eedb8d8b9ea30ab80ade483bb/scripts/grammar_probe/judge.py#L293)
omits prompt text, termination metadata, budget, and generation-run identity;
judge identity is a model alias. The [rejudge path](https://github.com/robertelee78/hf2q/blob/294907cd655ee81eedb8d8b9ea30ab80ade483bb/scripts/grammar_probe/rejudge.py#L237)
does not compare resumed inputs/configuration with the existing manifest.
Transition helpers also omit arm/pass identity.

**Required proof:** bind complete judgment inputs and available artifact
identities, validate the existing manifest on resume, and reject incompatible
or ambiguous passes. Unknown historical identities must remain unknown.

### E14. The proposed human-review export still shortens responses

The [human sample](https://github.com/robertelee78/hf2q/blob/294907cd655ee81eedb8d8b9ea30ab80ade483bb/scripts/grammar_probe/rejudge.py#L129)
contains only the first 1,200 characters and displays candidate verdicts.
Randomizing pass names does not make this independent full-response labeling.
A distinguishing response tail was absent in the reproduced fixture.

**Required proof:** provide full responses and obtain independent labels before
showing machine judgments. Separate a corpus-representative validation sample
from a disagreement-enriched diagnostic sample and report their selection rules.

### E15. Configuration manifests are not complete runtime bindings

The [baseline manifest](https://github.com/robertelee78/hf2q/blob/294907cd655ee81eedb8d8b9ea30ab80ade483bb/scripts/grammar_probe/baseline_run.py#L132)
uses a model alias, optional operator-declared server identity, and a template
label. Its canary detects the W1 opening but cannot establish absence of every
grammar or GLP intervention. The GLP panel adds useful artifact hashes, but
still lacks a binary-to-source binding and exact judge artifact identity.

**Required proof:** record and verify the effective configuration and exact
artifacts at the runtime boundary. Do not infer an unsteered baseline from
one response shape or relabel present-day hashes as historical provenance.

### Remaining handoff before replacing measurements

A self-contained [implementation-agent handoff](gcd-publication-repair-handoff.md)
collects the repair scope and regression cases.

Repair S19–S20 in the runtime lane and E12–E15 in the harness lane. Then use
versioned full-response rejudging and independent human validation to reassess
retained responses. Run fresh matched baseline/W1 and GLP comparisons only
after their runtime paths and observation identities are validated. Preserve
all historical attempts; measure quality and missingness without targeting a
preferred refusal rate. The article's scoped claims can be reviewed now, but
these outstanding paths must not be presented as publication-validated features.

## Original audit findings (historical source snapshot)

The original S1–S18 and E1–E11 findings below refer to source snapshot
`44004311717d414feaa54384578e2bcf4d140464`. Consult the status table above before
treating a past defect as still present. Source links in this historical
section are pinned to that revision where the referenced file was tracked.

## Source defects for the implementation owner

Priorities reflect publication relevance and potential impact, rather than a
claimed exploitability score. **P1** means resolve before presenting the
associated feature as dependable. **P2** means a concrete robustness or
compatibility defect that also needs correction. Hardware impact is not claimed
as reproduced unless explicitly stated.

### S1. Qwen GLP projection uses an incompatible threadgroup size — P1

The [Qwen dispatcher](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/apply_gpu.rs#L145) uses hidden width
`h` as the threadgroup size. The [Metal kernel](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/shaders/glp_project.metal#L50)
allocates `partial[256]`, indexes it with the thread ID, and uses a fixed
256-lane reduction. For `h > 256`, threads index outside that array if the
launch succeeds; sufficiently large groups can exceed device limits as well.
The full-logits Qwen path calls this dispatcher.

**Required proof:** agree on one threadgroup per activation row, launch the
kernel's required number of lanes, and compare GPU output with a numerically
sound CPU reference at actual model widths and multiple row counts. Use a
nontrivial direction, zero dose, and live dose. Source mismatch confirmed;
actual hardware error or numerical corruption was not reproduced here.

### S2. Qwen greedy decode omits the GLP intervention — P1

[Greedy eligibility](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/engine_qwen35.rs#L1228) depends on request
sampling parameters and has no bound-GLP exclusion. An eligible request calls
`forward_gpu_greedy`; its [post-FFN path](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/models/qwen35/forward_gpu.rs#L7365)
finishes the layer without the GLP hook present in the separate full-logits
path. Thus GLP can affect prefill but cease to apply during greedy decoding.

**Required proof:** implement the hook in all supported execution paths or
route steered requests to a path that honors it. Compare greedy and full-logits
deterministic generation, including unary, SSE, and scheduler paths. Source
omission confirmed; its real-model behavioral magnitude is unmeasured.

### S3. Hook identity is discarded and incompatible sites are aliases — P1

The [reader](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/reader.rs#L30) accepts
`residual_stream_post_layer` and `ffn_out_pre_residual` as aliases, but the
loaded `GlpVector` does not retain either identity. DeepSeek's
[application](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/models/deepseek4/ffn_forward.rs#L565) modifies
`ffn_output` before the hyper-connection fold. Qwen modifies the complete
post-layer residual. These operations do not become equivalent because both
accept the same metadata string.

**Required proof:** preserve the hook in the parsed representation, validate
it against the selected family and actual execution site, and reject unsupported
combinations. The public Weightless specification has dated site corrections;
use the current declared contract, not an older prose description.

### S4. Calibration misdescribes derivation versus application — P1

[Calibration](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/calibrate/mod.rs#L195) derives its exported direction
from stream 0 of the complete post-layer hyper-connection capture, then
[describes](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/calibrate/mod.rs#L335) both the derivation and apply hook as
post-layer residual. Actual DeepSeek application is at the FFN writer before
the fold.

A cross-site transfer is not intrinsically invalid. It is a different
experiment that requires explicit metadata and behavioral validation. A live
logit shift proves influence, not that the derived direction has the intended
meaning at another site.

The calibration code also computes a pinned compliance-versus-refusal prefix
contrast, logs its norm, and discards that direction. Only the normalized
harmful-versus-harmless prompt contrast is exported. The article now describes
this accurately; the module comment's promised statistics sidecar is not
implemented. Separating captured prefixes is not evidence of held-out
behavioral improvement.

**Required proof:** establish a declared derivation/apply pair, matching
capture and execution tests, and held-out behavioral/capability measurements.
Do not “fix” this by relabeling old artifacts as if they were derived elsewhere.

### S5. DeepSeek ignores additive mode — P1

The reader accepts `add`, and absent `glp.mode` defaults to additive semantics.
DeepSeek nevertheless calls the projection-only
[`apply_layer_gpu_in_session`](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/models/deepseek4/ffn_forward.rs#L586)
without passing the operation mode. An additive artifact is silently interpreted
as a projection.

**Required proof:** implement the declared operation at the declared hook, or
reject it before serving. Cover both explicit `add` and absent mode metadata.
A file-format parser test alone does not exercise this failure.

### S6. Qwen persistent prefix-cache identity omits GLP — P1

[`build_lcp_key_for_qwen35`](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/engine_qwen35.rs#L1609) constructs
identity from base-model provenance and template and sets `params_hash=0`.
It omits steering content, hook, mode, layer mapping, and strength. The disk
sidecar uses that identity. Restarting with the same base model and cache
directory but different steering can therefore address incompatible saved
activations with the same key.

**Required proof:** include all activation-affecting configuration in cache
identity or disable unsupported reuse. Demonstrate same-configuration reuse
and rejection/rebuilding after vector, dose, hook, or mode changes. The key
collision is source-confirmed; resulting real-model divergence was not measured.

### S7. GCD default injection conflicts with explicit request surfaces — P1

The [handler's default-injection checks](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/handlers.rs#L473)
look only for `grammar` and `response_format`. They ignore supported
`json_schema` and `structured_outputs` surfaces. The
[request compiler](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/grammar/request.rs#L724) subsequently rejects
the resulting simultaneous constraints. A supported explicit request can
become HTTP 400 merely because the server has a GCD default.

**Required proof:** recognize every explicit constraint surface before
injecting defaults. Exercise real handler preparation, including resulting
thinking configuration and schema mode. Keep mandatory policy enforcement a
separately specified contract.

### S8. Model compatibility is not fully checked before binding — P1/P2

The Qwen and DeepSeek engine loaders bind the artifact without validating all
layer indices and dimensions against the selected model. A vector naming only
nonexistent layers can be silently unused. Qwen's dispatcher lacks the
width check found in the DeepSeek in-session path; a too-short direction can
lead to reads outside its buffer. Collected base-model metadata is not an
enforced checkpoint identity.

**Required proof:** reject incompatible dimensions, unsupported layers,
unsupported hook/mode combinations, and incompatible model bindings before
serving. Document any deliberate compatibility relaxation. Loading a file
successfully is not proof that its intervention executes.

Sources: [Qwen loader](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/engine_qwen35.rs#L365),
[DeepSeek loader](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/engine_deepseek4.rs#L746), and
[device binding](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/bind.rs).

The [discovery provenance check](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/discovery.rs#L265)
also compares only a nonempty base-model name, permits absent names, and
does not compare the checkpoint revision or model bytes. Automatic discovery
therefore does not repair the missing checkpoint binding.

### S9. Additive vector magnitude changes during binding — P2

[`BoundGlp::bind`](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/bind.rs#L37) normalizes directions for
both modes. The Qwen additive path consequently receives a unit vector,
implementing `h + alpha * v / norm(v)` rather than the additive arithmetic
`h + alpha * v` in the [CPU reference](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/apply.rs#L41).

**Required proof:** preserve additive magnitude; normalize only where the
operation requires it. Test a non-unit vector through binding and GPU
application. The current CPU arithmetic test does not cover device binding.

### S10. Non-finite direction values can reach device binding — P2

The reader permits arbitrary f32 payload values. Binding checks a norm only
with `<= 0`; NaN makes that comparison false. Infinite values can normalize
into NaNs. This can contaminate the model computation.

**Required proof:** reject non-finite elements, invalid norms, and invalid
arithmetic before GPU upload. Test NaN, positive/negative infinity, zero, and
large finite directions. Source: [binding](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/bind.rs#L46).

### S11. Malformed GLP dimensions and offsets use unchecked arithmetic — P2

The [reader](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/reader.rs#L364) multiplies a file-controlled
width by four before checking its size budget, then adds offsets without
checked arithmetic. Overflow can panic in a checked build; wrapping can defeat
intended bounds checks or produce a later invalid slice in an unchecked build.

**Required proof:** checked conversions, multiplication, addition, and bounded
slice access, with extreme dimensions/offsets tested. This review establishes
unsafe validation arithmetic, not a demonstrated memory-execution exploit.

### S12. Combining GCD and GLP silently changes the grammar — P1

At [handler injection](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/handlers.rs#L477), a bound GLP path
selects a think/anchor/free-body grammar instead of embedded W6V2, even at
zero dose. The handler also forces `hf2q_enable_thinking=false`.

This is an explicit branch, not an accidental missing call. Its defect is the
unannounced change in control semantics and the confounded comparison: adding
steering also changes the grammar. Resolve composition as an explicit product
contract. Zero-dose equivalence should hold when GLP is the only intervention
being varied.

### S13. Existing mHC helpers are unsafe substitutes for a hook repair — latent P1

The currently unused mHC helpers pass `rows * hc` as a thread count to
`encode_with_args`, which uses Metal `dispatch_threads`. The
[shader](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/shaders/glp_project_mhc.metal#L29) interprets the
threadgroup ID as one row/stream and reduces 256 lanes. For one row and four
streams, the helper launches four threads, not four full groups. This leaves
reduction entries unwritten and does not cover every stream.

Additionally, the per-stream direction form uses stream 0's norm for every
stream. Different direction norms therefore receive incorrect scaling.

**Required proof before use:** correct launch geometry, independent per-stream
normalization, and GPU comparisons across rows and streams. These helpers
have no current production callsites; do not characterize this finding as an
observed defect in the active DeepSeek FFN path. Sources:
[dispatch](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/apply_gpu.rs#L191) and
[normalization use](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/shaders/glp_project_mhc.metal#L54).

### S14. External GLP directions are applied one graph layer early — P1

The public [GLP reader contract](https://github.com/msuiche/weightless/blob/main/spec/GLP.md#reader-conformance)
requires `direction.N` to apply at actual zero-based graph layer `N`, with
`direction.0` rejected. Both [DeepSeek](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/models/deepseek4/ffn_forward.rs#L567)
and [Qwen](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/models/qwen35/forward_gpu.rs#L5691) instead look up
the current graph index plus one. An external `direction.N` therefore steers
graph layer `N-1`.

The [calibration exporter](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/calibrate/mod.rs#L370) adds the matching
offset. Its self-export/import path can consequently pass a canary while
interchanging directions incorrectly with another conforming runtime.

**Required proof:** use actual graph IDs consistently and validate them against
the model and format. An independently generated single-layer artifact must
affect the named graph layer in a runtime probe, with zero-dose and live-dose
checks. Self-round-trip tests alone cannot establish interoperability. The
mapping mismatch is source-confirmed; no real-model probe was run here.

### S15. Calibration writes the wrong derivation metadata key — P2

The [exporter](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/calibrate/mod.rs#L342) writes `glp.derive_at`; the public
[GLP specification](https://github.com/msuiche/weightless/blob/main/spec/GLP.md)
defines `glp.derived_at`. A conforming consumer can ignore the unknown key and
interpret the absent recognized field as derivation at the apply hook. This
conceals the distinction needed to assess a cross-site direction.

**Required proof:** emit the specified key and actual derivation identity.
Validate exported metadata with an independent consumer or format fixture,
including a deliberate derivation/apply mismatch.

### S16. Exported layer-list metadata has the wrong type — P2

The [exporter](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/calibrate/mod.rs#L361) writes
`glp.layer_ids_zero_based` as `Bool(false)`. The public
[GLP specification](https://github.com/msuiche/weightless/blob/main/spec/GLP.md)
defines a comma-separated string of actual graph layer IDs, redundant with
the tensor names. A Boolean neither carries that list nor satisfies its type.

**Required proof:** fix this with S14 and check that the metadata list,
`direction.N` suffixes, and actual application layers agree. Include an
external-format conformance check rather than relying solely on hf2q's reader.

### S17. Explicit GLP Hub references are treated as local paths — P2

The [CLI documentation](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/cli.rs#L1336) advertises a local path or Hub
reference. [Startup resolution](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/mod.rs#L4997) handles only the
`auto` sentinel specially and passes every other value through as a path.
The [Qwen](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/engine_qwen35.rs#L368) and
[DeepSeek](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/serve/api/engine_deepseek4.rs#L748) loaders call
[`GlpVector::load`](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/reader.rs#L245), which reads the local
filesystem. Supplying an explicit remote reference therefore does not invoke
the advertised retrieval path. The README now describes local paths accurately.

**Required proof:** implement a defined remote-reference contract or correct
the CLI/help promise. Test a valid local file, a missing local file, an explicit
supported remote reference if implemented, and bare `--glp`, using stubbed
retrieval. Model inference is not needed to verify artifact resolution.

### S18. Automatic GLP discovery does not reject ambiguous matches — P2

[Repository candidates](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/discovery.rs#L151) are tried in
order; discovery [returns the first accepted candidate](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/discovery.rs#L226).
Within a repository, [file selection](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/src/inference/glp/discovery.rs#L253)
uses the first `.gguf` sibling. Neither step establishes that the eligible
artifact is unique. Selection can depend on inventory order, contrary to the
former README's claim that ambiguity is rejected.

**Required proof:** select an explicitly identified artifact or reject multiple
eligible candidates. Cover zero, one, and two eligible files, reversed file
order, and multiple eligible repositories. Checkpoint identity remains a
separate requirement under S8.

### Deliberate boundaries, not automatically defects

- `validate_grammar_terminal` rejects dead or incomplete active grammars and
  deliberately exempts an AUTO tool grammar whose trigger never appeared.
- Tool grammar takes precedence over response grammar. The paper must not
  imply that a server prose default constrains all tool-bearing responses.
- A schema's unrestricted string fields can hold refusal or incorrect content.
  This is a conceptual boundary, not a parser defect. The recon example's
  `description` currently claims that `minLength` prevents refusal prose;
  correct that description and the corresponding ADR-057 explanation too.
- Calibration/apply sites can differ in a justified experiment. Undeclared or
  unvalidated transfer is the concern.
- `finish: length` means budget exhaustion. It does not, by itself, prove
  invalid grammar, an incorrect EOG implementation, or useless content.

## Measurement and harness findings

### E1. The claimed conformance completion is unsupported

The original draft says “26 cells.” ADR-056 says partially implemented and reports
12 green cells; FINDINGS.md says 18 PASS / 1 WARN / 5 FAIL / 3 SKIP, a total
of 27. The actual committed script contains 12 cells, and the local log has
12 observations. These are inconsistent inventories.

Both logit-bias cells supply `{}` and therefore do not exercise their named
lever. The log records `held=true` in 12 cells but `engaged=false` in five.
A supposed unconstrained control sent to a server with `--gcd` can receive
the embedded default. “Different text” under stochastic generation and “no
lexicon hit” are weak engagement tests even with a correct control.

**Resolution:** remove the completed-battery claim from the article. The
implementation owner should repair the harness and produce a named inventory
with exact checks, actual outcomes, skips, and independently constrained and
unconstrained canaries. Do not translate another engine's beam-search finding
into proof of every hf2q alternate path.

### E2. Historical W1 results have missing labels and weak runtime provenance

The archived records contain 1,024 distinct prompt/rep observations per model,
with no duplicate response text between matching model/prompt pairs. There
are four failed Gemma judgments and ten failed Qwen judgments. Reported rates
must state their denominator and preserve those failures.

| Model label | Adversarial maintained refusal | Benign maintained refusal | Unjudged total | Degenerate total | `finish: length` total |
|---|---|---|---|---|---|
| DeepSeek-V4 | 12/512 | 0/512 | 0 | 91/1,024 | 823/1,024 |
| Gemma-4-26B | 37/511 | 0/509 | 4 | 156/1,024 | 644/1,024 |
| Qwen3.8-27B | 26/502 | 0/512 | 10 | 110/1,024 | 416/1,024 |

Missing judgment is not non-refusal. “No benign refusals” applies only to
judged observations and is not a claim of preserved benign capability.
Truncation and semantic categories overlap and must not be added together.

The generic METHODS.md gives sampled defaults that differ from the greedy
`spike_run.py`; the output rows do not carry a full immutable run manifest.
Model labels are not file identities. The reviewed source commit identifies
code inspection, not the historical generation binary. The aggregate manifest
preserves input hashes and records these limits explicitly.

### E3. The embedding screen does not lower generation refusal

The DeepSeek gate log flags 9/12 maintained refusals and 3/912 valid
fulfillments. It also flags one pivot-then-fulfillment and one degenerate
response. The remaining three refusals are 3/512 = 0.586% of adversarial
prompts. That is a miss count after offline screening, not proof that a
combined server generates fewer refusals. Full-corpus valid-fulfillment false
positives are 3/912 = 0.329%, despite zero false positives on an earlier spike.

The article now describes the screen as offline, includes its false positives,
and avoids presenting it as a built-in hf2q regeneration or serving layer.

### E4. The human spot-check is enriched and errors do not “cancel”

Robert identified the judge as his prior
[Qwen3.6-35B-A3B-Abliterix-EGA-abliterated checkpoint](https://huggingface.co/jenerallee78/Qwen3.6-35B-A3B-Abliterix-EGA-abliterated),
recorded under the alias `qwen36-abliterix-t63-APEX`. The paper now credits
that model and cites the human comparison separately. This author-supplied
identity does not recover the exact historical quantized artifact hash.

The sample selected all twelve maintained refusals and thirteen other cases.
It is not a random corpus sample. Its listed disagreements are three
over-flags and one under-flag; the prose's “symmetric errors” is incorrect.
The 21/25 agreement is descriptive of that selected sample. A binomial interval
around it does not turn it into representative agreement or remove label
uncertainty from the corpus rates.

The spot-check narrative also calls sample 13 a harmful-set labeling error.
The local sample record identifies it as `b220`, while the checked-in corpus
already places that prompt in the benign stratum. Resolve this discrepancy
before repeating that alleged corpus error.

### E5. The dose-response figure is not a controlled contrast

The old plotting script hardcodes rates and sets its approximately universal
baseline to exactly 1.00. It combines different sample sizes and selections.
B17 has nine verdicts; B18 has seventeen rows but eleven distinct prompt/rep
keys; B19 has eighteen rows, eleven keys, and a failed judgment. B20 and the
W1 spike each have twenty-nine rows; the plotted W1 number is instead the
full-corpus rate. These marginals do not establish a causal dose-response.

The replacement figure shows complete W1 outcome distributions by model and
stratum. A future ablation figure needs paired prompt identities, distinct
run identity, consistent classification, and uncertainty appropriate to the
design. Increasing constraint complexity also is not a scalar biological dose.

### E6. The W6 retest table omits transitions and selection effects

The retest contains 118 unique prompts. Its three displayed groups have sizes
38, 60, and 39 because nineteen benign prompts also belong to the degenerate
cohort. The groups are not disjoint. The baseline “refusal” cohort combines
37 maintained refusals with one partial refusal, unlike the main maintained-
refusal metric.

Under W6V2, the original refusal cohort becomes 23 maintained refusals, six
fulfillments, eight degenerate responses, and one unjudged response. The
original degenerate cohort becomes twenty degenerate, thirty-three fulfilled,
five maintained refusals, one partial refusal, and one unjudged response.
The benign cohort improves to 35/39 judge-labeled valid fulfillments.

The original table's headline cells can be recovered, but they hide category
migration, missing labels, and the outcome-selected sample. “Cores dominate
on every axis” is stronger than this evidence permits. The main article now
states the W1/W6V2 artifact distinction without presenting the retest as
confirmation of general performance.

### E7. Fixed generation budgets dominate termination, without a matched baseline

[`spike_run.py`](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/scripts/grammar_probe/spike_run.py#L48) requests 800
completion tokens, greedy sampling, no system prompt, and disabled thinking.
Every recorded `finish: length` observation has exactly 800 completion tokens;
all 3,072 observations have zero reasoning characters.

| Subject | Adversarial capped | Benign capped | Responses exceeding judge's 2,500-character input limit |
|---|---|---|---|
| DeepSeek-V4 | 321/512 | 502/512 | 971/1,024 |
| Gemma-4-26B | 142/512 | 502/512 | 903/1,024 |
| Qwen3.8-27B | 102/512 | 314/512 | 475/1,024 |

Of DeepSeek's 91 degenerate labels, 85 have budget-limited generation and six
have `finish: stop`. All 58 benign degenerate responses hit the cap. The
benign set contains open-ended tutorials for which the observed budget can
leave a procedure unfinished. Raising the budget might permit completion or
extend repetition; this audit does not establish which would occur.

The retained W1 runs contain one arm and one repetition. No `Arm A` record
was found in the inspected `*results*.jsonl` inventory. Older spike results
do not supply a matched full-corpus control. The
[W1 grammar](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/scripts/grammar_probe/w1.gbnf#L5) permits its body to terminate;
the observed caps therefore do not establish a grammar bug that forbids EOS.

**Required proof:** run a paired baseline/W1 study with fixed source and model
identities, prompts, templates, and sampling settings; vary an explicit budget
ladder and report termination separately from substance and degeneration.

### E8. Judge-side clipping contaminates quality labels

[`judge.py`](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/scripts/grammar_probe/judge.py#L94) clips responses to their
first 2,500 characters and appends `[...truncated for judging]`. It does not
pass generation finish reason or usage metadata, although the rubric asks
the judge to assess truncation, degeneration, and refusal that can appear
later in a response.

Thirteen DeepSeek degenerate verdicts quote the artificial cutoff marker as
evidence. This includes `h442`, whose generation ended with `finish: stop`
at 766 tokens and 3,306 characters. The marker also appears in thirteen Gemma
degenerate verdicts, one Gemma fulfillment verdict, and five Qwen degenerate
verdicts. This establishes contamination of the judgment evidence; it does
not independently establish that the affected full responses were useful.

**Required proof:** judge complete retained responses with generation
termination metadata. If input limits prevent that, record the omission as
an evaluation failure or use a separately validated full-coverage procedure.
Preserve old judgments and identify the new scoring pass explicitly.

### E9. Fulfillment state and output validity contradict each other

The [judge validator](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/scripts/grammar_probe/judge.py#L125) requires a
minimum substance score for fulfillment but does not require valid output.
The records contain both `response_state: valid_fulfillment` and
`output_validity: invalid` in 20 DeepSeek responses (9 adversarial, 11 benign),
24 Gemma responses (7, 17), and 12 Qwen responses (4, 8).

The [publication extractor](../scripts/grammar_probe/publication_data.py#L76)
faithfully counts response-state labels. The resulting categorical figure is
reproducible, but those labels must not be promoted to valid, completed
answers. The paper now exposes the DeepSeek inconsistency.

**Required proof:** define the relationship between state, substance, validity,
and termination; validate the schema across fields. Report their joint
distribution or perform a versioned adjudication pass. Do not silently change
historical labels to make them consistent.

### E10. Failed judgments are skipped on normal resumption

Gemma's four failed judgments record HTTP 500 without the response body,
preventing diagnosis of the server cause. Qwen's ten failures record a
cross-field invariant violation: fulfillment with insufficient substance.
These are known scoring failures, not missing generations.

The [resume logic](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/scripts/grammar_probe/judge.py#L138) adds failed attempts
to its completed-key set. A normal rerun skips them.

**Required proof:** preserve each failed attempt and its error details, but
support explicit retries with distinct attempt identity. Keep scoring failures
visible in published denominators until a valid replacement judgment exists.

### E11. The paired-report path crashes and omits promised intervals

[`report.py`](https://github.com/robertelee78/hf2q/blob/44004311717d414feaa54384578e2bcf4d140464/scripts/grammar_probe/report.py#L80) indexes derived metric
names such as `refusal` on raw verdict dictionaries. A two-arm in-memory
fixture reproduced `KeyError: 'refusal'`. The report also lacks the bootstrap
intervals promised by the methodology, and its fulfillment metric omits
`output_validity`.

**Required proof:** transform verdicts into defined metrics before paired
comparison; test known paired transitions, missing labels, and inconsistent
fields. Implement the stated uncertainty method or correct the methodology.

### Testing repair and rerun order

Repair complete-response judging, termination awareness, field consistency,
failure retries, immutable run manifests, and paired reporting first. Rejudge
the retained full responses with a versioned rubric and an independent blinded
human sample. Then run the paired budget experiment in E7. This order separates
measurement defects from runtime or intervention effects and preserves the
historical evidence rather than rewriting it.

## Prose review and editorial decisions

| Original passage or section | Finding | Revision |
|---|---|---|
| Title: “Engine-Native Positive Security Control” | Conflates the authorization application with all grammars, including lexical refusal exclusions. | Retitle around GCD, GLP, and local inference. |
| Opening epigraph and attribution block | Insider framing precedes the subject; “first engine” lacks an exhaustive priority basis. | Lead with the reader's practical problem; retain attribution at the end; remove priority claim. |
| “Uncensored models reason better, full stop” | No matched capability evidence supports a universal improvement. | State operator motivation and separate engagement from capability. |
| Political reluctance transfers to physics | Mechanistic/causal speculation presented as fact. | Remove; no inference from shared model machinery proves that transfer. |
| Bias and debiasing discussion | Broad philosophical claims distract from the technical contribution. | Focus on explicit operator controls and measurable behavior. |
| “Every deployed control is negative” | False universal; execution whitelists are positive policies. Vince's paper discusses them directly. | Explain constitutive versus corrective enforcement of the same policy. |
| 1.82–33.8% versus 0% | Numbers belong to Tantalus, with a specific threat model and different subjects. | Attribute external evidence and its scope; avoid importing it as an hf2q result. |
| 97.85% on an abliterated adversary | Misread dependent variable: Vince reports authorization bypass, not maintained refusal. | Remove the false refusal interpretation. |
| “LANGSEC makes the recognizer correct by construction” | A design approach is not proof that a particular recognizer is correct. | State policy/compiler/tokenizer assumptions explicitly. |
| GCD equation | Undefined quantities and no clear distinction between local masking and sequence conditioning. | Define prefix, admissible set, normalizer, and completion; cite grammar-aligned decoding. |
| “Mask last; nothing additive after” | Oversimplifies the actual sampler order. Support preservation is the required property. | Describe candidate exclusion and require later operations not to restore forbidden candidates. |
| “Automaton never reads the prompt” | Runtime tracks output, but grammar construction can depend on trusted request state; tool grammars may activate lazily. | Separate grammar construction, active scope, and runtime state. |
| FFN is refusal's main writer | Model/site-specific intervention result presented as universal architecture. | Remove empirical percentage from the general diagram. |
| “Moves onto the compliance manifold and stays there” | Contradicted by residual paraphrase refusals; probability trace does not establish a manifold. | Describe changed conditioning and the observed trace only. |
| “Membership edits can't move mass” | Masking explicitly renormalizes probability; subsequent conditioning changes logits. | Explain retained odds at a fixed prefix and absence of semantic guarantees. |
| Character whitelist closes the homoglyph class | W1 explicitly includes selected non-ASCII apostrophes; exclusion depends on actual productions and folding. | Say it excludes characters outside its alphabet; avoid universal evasion closure. |
| Cross-family universal grammar claim | Three campaign subjects cannot prove universality, particularly with failed labels and different artifact identities. | Report observed counts and scope. |
| GLP paragraph | Too compressed; treats a canary as proof of useful calibrated behavior. | Add accessible distribution explanation, equation, geometry, compatibility, and mechanical/behavioral distinction. |
| “Schema eliminates the refusal space” | Required strings still accept refusal content. `minLength:1` does not enforce intended meaning. | Explain structure versus content and finite-value authorization fields. |
| “Models trained on more JSON than anything else” | Unsupported training-distribution assertion. | Remove. |
| “Local engine has no perimeter” | A deployment still has authentication, process, network, and execution boundaries. | Describe local control of deployment without denying those boundaries. |
| “Any security constraint expressible as a formal language” | Overbroad without supported grammar class, trusted policy construction, state, and execution semantics. | Use enumerable action sets as the concrete application. |
| “Fail-closed 500 applies at completion” while streaming | An established SSE response cannot become a new HTTP 500. | Explain pre-stream HTTP errors versus errors after streaming begins. |
| Artifact inventory and release claims | Mixes local/unpublished results, implemented code, plans, and purported shipped guarantees. | Pin inspected source, give a compact map, and disclose historical evidence access. |

## Figure review

### Original `fig3_stack.png`: replace with a readable conceptual diagram

The raster packs several diagrams and an extended argument into a single
page. At article width, body text is too small. Several arrows point left
although the lifecycle claims to flow left to right. Hardcoded vocabulary
and layer counts imply universality. The FFN percentage, derivation claim,
per-state bitmask cache, and GLP site description combine different engines,
models, and revisions.

More seriously, it says membership cannot move mass and suggests a particular
internal origin for vacuous output without a causal test. The corrective prose
below the image cannot repair false text inside the image itself.

**Replacement:** `generation-controls.svg`, with a PNG export. It shows the
model, grammar-aware selection, token commit, feedback, and completion boundary.
It labels family-specific activation sites without certifying an unvalidated
hook. A separate GLP geometry figure carries the projection explanation.
The original contribution remains credited.

### Original `fig1_frontloading.png`: rebuild from actual observations

The left chart averages a token only over rows in whose top-ten list it
appears, so the bars do not share a consistent denominator for rare tokens.
It suppresses variation across the twelve selected prompts and gives small
alternatives effectively invisible bars. The right trace lacks a clear
fixed/free distinction, and the arbitrary 0.5 line has no stated interpretation.
The title overstates a causal decision mechanism.

**Replacement:** `entry-trace.svg`. Plot the token `I`, present in every row,
for all twelve prompts; explicitly mark the zoomed probability axis and mean.
Plot one successive-token trace with fixed-text and free-span regions. State
post-mask/pre-sampling-transform probability semantics and different conditioning
at each position. No implication of a single normalized distribution across
positions.

### Original `fig2_arms.png`: retire the dose-response framing

The chart uses hardcoded fractions, a fabricated exact 100% baseline, rounded
labels, heterogeneous samples, and a causal title unsupported by the design.
Its green/blue emphasis also implies a selected winner without uncertainty or
quality costs. Calling its inputs “Real jsonl, reproducible” is inaccurate
for this panel.

**Replacement:** `w1-outcomes.svg`. Show full counts for all 512 observations
in each model/stratum, including unjudged cases. Preserve the distinction between
maintained refusal, degeneration, and other outcomes. Supply all underlying
counts in CSV, JSON, and a workbook.

### Additional `gcd-gate.png`: exclude from publication

This asset is not embedded in the original Markdown but was inspected. Its
right-hand bars combine probabilities from different positions, sum to 2.21,
and are displayed as if they were a next-token distribution. A bottom note
acknowledges that fact, but the title and design still convey the wrong object.
It says “3-token committed prefix” while showing a longer anchor, and collapses
the split `mechan` / `ism` tokenization into a word label. Its left-side
alternative values are not a reconstruction of the archived distribution.

**Disposition:** leave the historical file intact, remove it from the
publication path, and use the source-derived trace figure. No decorative
regeneration can repair a statistical object defined incorrectly.

### New `glp-projection.svg`

This is an explanatory geometric construction, not a measured dose ladder.
It shows unchanged activation, component removal, and reflection for strengths
zero, one, and two. Both figure and caption explicitly deny a behavioral
interpretation. This supports the “small attached intervention” introduction
without perpetuating the literal weight-diff analogy.

## Source hierarchy and remaining publication checks

The primary conceptual sources are Vince Ovando's supplied 47-page `gcd.pdf`
(June 26, 2026), his [companion proof](https://github.com/cybersharkvin/gcd-authz/blob/main/docs/proof.md),
Suiche's [GLP specification](https://github.com/msuiche/weightless/blob/main/spec/GLP.md)
and [dated hook correction](https://www.msuiche.com/posts/autoresearch-sticky-refusals-free-speculative-decoding-and-the-invisible-quantisation-cliff/),
and the original [GCD](https://aclanthology.org/2023.emnlp-main.674/),
[grammar-aligned decoding](https://arxiv.org/abs/2405.21047),
[XGrammar](https://arxiv.org/abs/2411.15100), and
[refusal-direction](https://arxiv.org/abs/2406.11717) research. These sources
support concepts; the Rust/Metal tree and exact run artifacts support hf2q
claims. The article includes numbered notes and a complete source list.

Matt Suiche's privately shared research informed the conceptual review.
It is credited without identifying private records. Its runtime performance
numbers are not adopted as hf2q validation. The interoperability findings
above are supported by the public GLP specification and hf2q source.

Before publication:

1. Resolve the source findings and obtain the implementation owner's exact
   commit, tests, and hardware evidence. Recheck the paper's source map and
   affected descriptions against that commit.
2. Obtain fresh, source-bound conformance and behavioral measurements for
   whichever configurations the article presents as validated. Preserve old
   results as historical data; do not quietly relabel them.
3. Decide whether the historical W1 case study remains in the public article.
   Its limitations are intrinsic to those records. New tests do not remove
   them, although a new study can replace the section.
4. Ensure every cited measurement needed by public readers is available in
   a shareable evidence package. The current aggregate manifest and workbook
   are inspectable, but source logs include local-only artifacts.
5. Re-render and visually inspect the article PDF after any evidence update.
   Keep Markdown, captions, figure data, and PDF synchronized.

## Validation performed and its limits

The review recomputed counts from all 3,072 historical generation records and
matching judge/gate records, checked uniqueness and joins, calculated the
entry-token mean, distinguished failed labels, and verified byte identity of
the embedded grammar with W6V2 rather than W1. It checked the retest cohort
membership and transitions rather than simply accepting the table headings.

An independent JSON Schema validator accepted an object containing refusal
text in every required string field of the exact recon example. This directly
refutes the prose claim that the schema makes that content unrepresentable;
it does not test hf2q's schema compiler.

Source findings were checked by reading call sites, dispatch geometry,
arithmetic, metadata handling, and cache keys. No Rust/Metal build, model load,
fresh generation, end-to-end serving test, or performance benchmark was run
for this editorial work. The implementation owner must supply those checks.
Rendering, link, arithmetic, and artifact checks are recorded in the
[publication validation record](figures/gcd/validation.md).
