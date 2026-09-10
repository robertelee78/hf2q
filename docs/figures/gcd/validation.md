# Publication artifact validation

Final editorial pass: September 10, 2026. This record covers the article,
figures, source review, archived measurements, and rendering. It does not
certify unresolved runtime or measurement paths as validated.

## Source and evidence

- The current implementation account is bound to
  `294907cd655ee81eedb8d8b9ea30ab80ade483bb`. Current source references use that
  revision; historical judge/probe references retain their original snapshot.
- `evidence.json` remains byte-identical to the prior historical audit. Its
  source snapshot is `44004311717d414feaa54384578e2bcf4d140464` and its 30 input
  hashes and 3,072 observations are unchanged.
- Fresh extraction reproduces that historical JSON exactly after reading
  source-only inputs from the pinned Git revision. Local tracked observation
  files must match that revision; later source repairs are not mislabeled as
  historical code.
- `followup-evidence.json` separately hashes 59 inputs and recomputes all
  three historical termination/state/validity cross-tabs, 240 later GLP
  generation/verdict pairs, calibration-text overlap, and battery attempts.
  A second extraction reproduces this JSON exactly.
- DeepSeek's 823 historical capped responses comprise 733 fulfillment labels,
  85 degenerate labels, and five other labels; 19 of those 733 fulfillment
  labels also carry an invalid-output flag. The paper does not present these
  cross-tabs as a new judging pass.
- The five GLP arms reproduce adversarial maintained-refusal counts of
  28, 28, 28, 29, and 29 of 32. Benign fulfillment labels are 4, 3, 3, 2, and
  3 of 16; benign token-limit terminations are 14, 15, 14, 15, and 15.
  Six panel texts overlap the calibration corpus. All 240 responses fit
  below that run's 2,500-character judge-input limit.
- The battery contains 68 rows in four 17-cell attempts, with 14, 15, 16,
  and 17 passes respectively. The final attempt has no failures or skips.
  Its incomplete run identity and narrow assertions remain explicit.
- All 29 historical CSV outcome rows remain unchanged. The workbook has
  12 sheets, including separate GLP counts/transitions, termination cross-tabs,
  battery attempts, and follow-up source hashes. GLP count and termination
  cells were checked against the JSON, independently of prose tables.

## Implementation and measurement review

Independent source, concept, and harness reviews were incorporated. The
source review credits the landed hook/mode, graph-layer, dispatch, greedy-path,
cache-identity, and grammar-composition repairs. It identifies the remaining
checkpoint/discovery limitations and two new locked-schema path defects.

All 13 supplied offline/mock harness tests passed; historical artifacts were
unchanged by those tests. Additional in-memory probes reproduced reporting
budget collapse, acceptance of a wrong response hash, ignored pass selection,
and shortened human-review input. The companion review records those findings
without changing the runtime, harness, or original measurements. The separate
repair handoff includes an executed offline reproducer for budget collapse,
ignored pass selection, and shortened human-review input.

The paper distinguishes repaired full-response dispatch and cross-field
validation from a completed replacement scoring pass. It does not claim
preserved benign capability, a causal reflection effect, exhaustive battery
conformance, or a latency improvement from the new observations.

## Prose, figures, and PDF

- Seven A4 pages, with a full-width title and abstract, seven main sections,
  three numbered equations, five captioned tables, four captioned figures,
  and 17 references. All 31 numeric citation links target the references on
  the final page.
- Every page received independent visual review. The final opening,
  implementation, GLP-result, and reference pages were also inspected after
  the last wording changes. No material clipping, gutter collision, or
  unreadable table/figure text was found.
- All four SVGs parse and all PNGs decode. Figure 1 now calls an accepting
  state structural completion; Figure 4 identifies historical judgment
  labels. Geometry and entry-probe figures retain their prior meaning.
- The twelve-prompt mean remains 99.97% in prose and Figure 3; exact raw
  probabilities remain in the historical snapshot and workbook.
- Extracted PDF text has no replacement glyphs or words outside page bounds.
- Twenty-four local file links in the article, companion review, repair handoff,
  figure guide, and getting-started guide resolve. Historical source links in the companion
  review were pinned instead of pointing old line numbers into repaired code.
- Private research is credited only broadly. Publication text, figure sources,
  JSON, workbook cells, and PDF text/links were checked for private repository
  identifiers and record filenames, and for the author's terminology preference.
- README and getting-started guidance now describe the corrected default
  grammar composition. Source-build and exact-checkpoint caveats remain.
- Tracked and staged diffs pass Git whitespace checks. The PDF remains scoped
  as binary in the documentation attributes file.

## Tools and reproduction

Python 3.13.12; Matplotlib 3.11.1; openpyxl 3.1.5; PyMuPDF 1.28.2;
Pandoc 3.11; Typst 0.15.1.

```bash
python3 scripts/grammar_probe/publication_data.py \
  --source-root /path/to/research-checkout \
  --review-commit 44004311717d414feaa54384578e2bcf4d140464 \
  --out /tmp/gcd-historical-recomputed.json
python3 scripts/grammar_probe/publication_followup.py \
  --source-root /path/to/research-checkout \
  --out /tmp/gcd-followup-recomputed.json
python3 scripts/grammar_probe/publication_figures.py
python3 scripts/grammar_probe/publication_pdf.py
git diff --check
```

The extractors require the exact raw historical records. Figure and PDF
rendering use the included snapshots and require no model or judge service.
PDF creation timestamps can change the output hash on a later render.

## Artifact identity at validation

- `docs/gcd-in-hf2q.md`
  SHA-256: `a5caef5464b05e2563c4cca1e2dfa509d20b66234be7ee18591e0cba5b3bfc3b`
- `docs/gcd-in-hf2q.pdf`
  SHA-256: `4082a0d318876c81770cb55891cebba7a53c256dab2794407739140d6384e70b`
- `docs/figures/gcd/evidence.json`
  SHA-256: `3eab958f32452dfb6960f9491baf0b923a7a7865377648d3f952d35e39362650`
- `docs/figures/gcd/followup-evidence.json`
  SHA-256: `99a0951f84e47a68ad5a142b98203264f9f86ca585d206736c7175beae1e4448`

## Remaining implementation work

No Rust/Metal compilation, model load, fresh generation, live rejudging, or
performance benchmark was run by the publication reviewers. The
[companion review](../../gcd-in-hf2q-review.md) distinguishes source-addressed
findings from remaining S19–S20 and E12–E15, partial checkpoint binding, and
open discovery behavior. A completed paper revision is not proof that those
paths are validated. The retained historical observations remain unchanged.
