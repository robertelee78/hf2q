# Publication artifact validation

Final editorial pass: September 10, 2026. This record covers the article,
figures, source inspection, archived measurements, and PDF rendering.

## Source and evidence

The implementation account is bound to `92afc4a317c90ea527033228e0492f392778e412`. This source identity
is separate from the binary identities in experimental records.
The historical W1 snapshot remains
`44004311717d414feaa54384578e2bcf4d140464`; the follow-up extractor retains
`294907cd655ee81eedb8d8b9ea30ab80ade483bb` for the later archived inputs.

- `evidence.json` is unchanged from the historical audit: 30 input hashes
  and 3,072 observations. Its 29 outcome CSV rows are unchanged.
- `followup-evidence.json` hashes 59 inputs and recomputes historical
  termination/validity cross-tabs, 240 GLP generation/verdict pairs,
  calibration-text overlap, and the operational battery. Both evidence
  snapshots were independently reproduced from their pinned source inputs.
- DeepSeek's 823 capped responses comprise 733 historical fulfillment labels,
  85 degenerate labels, and five other labels. Nineteen of the 733 fulfillment
  labels also carry an invalid-output flag. This cross-tab is not new judging.
- The five GLP arms contain 28, 28, 28, 29, and 29 adversarial refusals of 32.
  Benign fulfillment labels are 4, 3, 3, 2, and 3 of 16; token-limit endings
  are 14, 15, 14, 15, and 15. Six panel texts overlap calibration. None of
  these 240 responses reaches that judge driver's 2,500-character input limit.
- The battery contains four 17-cell attempts, with 14, 15, 16, and 17 passes.
  The final attempt has no failures or skips; its scope and incomplete
  runtime identity remain explicit in the article.
- The workbook contains 12 sheets. GLP and termination cells were checked
  against JSON independently of the prose tables.

## Implementation and measurement validation

Source review verified the operation/hook, graph-layer, greedy-path,
cache-identity, and default-grammar corrections. The final pass also repairs
locked-schema policy composition, GLP artifact selection and checkpoint
binding, calibration export provenance, and measurement record integrity.

The focused Python suite exercises complete-response judging, versioned
passes, hash-checked joins, separate run/budget conditions, independent human
review exports, owned runtime identity, configuration drift, and campaign
process cleanup. The CI workflow runs these tests alongside focused Rust
contracts for policy enforcement, runtime snapshots, GLP reading/discovery,
checkpoint binding, and calibration exports. Exact-head CI results are
recorded on the associated pull request; this document does not substitute
for those results.

Offline and mocked checks do not constitute real-model validation of the new
paths. The existing DeepSeek campaign retains its original generation harness
and configuration; subsequent managed judging produces a separate scoring
pass. No completed replacement pass is included in this article. Historical
observations and unknown historical identities have not been rewritten.

## Prose, figures, and PDF

- Seven A4 pages with two columns, a full-width title and abstract, seven
  main sections, three numbered equations, five tables, four figures, and
  17 references. All 34 numeric citation links resolve to the bibliography.
- Every page was visually reviewed. No material clipping, gutter collision,
  or unreadable figure/table text was found. Bibliography spacing keeps the
  final reference on page seven without reducing the reference font size.
- Four SVGs parse and all accompanying PNGs decode. The control diagram
  describes structural acceptance; the outcome figure identifies historical
  judge labels. The geometry figure is explicitly illustrative.
- The twelve-prompt mean is 99.97% in prose and Figure 3. The exact raw
  observations remain in the evidence snapshot and workbook.
- Extracted PDF text contains no replacement glyphs or words outside page
  bounds. All 18 checked local links in the article, figure guide, and
  getting-started guide resolve.
- Private research is credited broadly. Article text, figures, JSON, workbook
  cells, and PDF text/links were checked for private repository identifiers
  and filenames, and for the author's terminology preference.
- README and getting-started examples describe the same grammar composition,
  locked-schema behavior, and GLP compatibility rules as the reviewed source.
- Git whitespace checks pass. The PDF is marked binary in documentation
  attributes.

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
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s scripts/grammar_probe -p 'test*.py'
git diff --check
```

The extractors require the exact raw historical records. Figure and PDF
rendering use the included snapshots and require no model or judge service.
PDF creation timestamps can change the output hash on a later render.

## Artifact identity at validation

- `docs/gcd-in-hf2q.md`
  SHA-256: `5fbbf29149e17b39f70b688da725204784c6299d3e94ac2a359612c343df4980`
- `docs/gcd-in-hf2q.pdf`
  SHA-256: `1d69547e57d1b0c8612a6a7ca1ba86e0a90d4dbb66b960bf341bd80b47689eed`
- `docs/figures/gcd/evidence.json`
  SHA-256: `3eab958f32452dfb6960f9491baf0b923a7a7865377648d3f952d35e39362650`
- `docs/figures/gcd/followup-evidence.json`
  SHA-256: `99a0951f84e47a68ad5a142b98203264f9f86ca585d206736c7175beae1e4448`
