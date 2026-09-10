# Publication artifact validation

Validated September 10, 2026. These checks cover the article, source review,
historical-data arithmetic, and rendering. They do not certify the runtime
findings as fixed or establish publication readiness.

## Source and evidence

- The article and aggregate snapshot identify hf2q source commit
  `44004311717d414feaa54384578e2bcf4d140464`.
- An earlier prose/source inspection used
  `bdb632ff406b9c7d77fa315d7b507351b1bee9cd`. The audited Rust/Metal files,
  schemas, and grammar inputs are unchanged between these commits.
- All 28 input-file hashes in `evidence.json` matched the inspected files.
- A fresh offline extraction to a separate temporary JSON reproduced the
  snapshot exactly, including counts, source hashes, and provenance fields.
- The extractor joined 3,072 generation records to corresponding judge and
  embedding-screen records, rejecting duplicate or mismatched observation
  keys. All six model/stratum panels contain 512 observations.
- All 29 outcome rows in the CSV match the JSON and workbook. Every panel's
  outcome counts and termination counts separately sum to 512. Workbook
  fractions use the recorded numerator and denominator; unjudged records
  remain visible. Termination and judgment categories are not added together.
- The twelve entry probabilities reproduce the reported 99.9744% mean. The
  sixteen-point trace retains exact selected-token spellings and probabilities.
- The embedded grammar is byte-identical to W6V2 and differs from historical W1.

## Prose and rendering

- Both independent concept and source reviews were incorporated. The source
  reviewer confirmed S1–S18 and their distinctions between active defects,
  latent helpers, intended boundaries, and unmeasured hardware consequences.
- A Draft 7 JSON Schema validator accepted a recon object with refusal text in
  every required string field of the exact checked-in example. This tests the
  prose claim about the schema language, not hf2q's compiler implementation.
- All four SVG files parse and all four PNG files decode. The figures were
  visually inspected individually and at their final PDF scale.
- SVG trailing whitespace and CSV line endings were normalized for Git.
  Parsed SVG geometry/text and CSV values were unchanged. The updated exporter
  reproduced all four SVGs, all four PNGs, and the CSV byte-for-byte in a
  separate temporary directory.
- The two-column paper edition has six A4 pages, a full-width abstract, seven numbered
  main sections, three numbered equations, four table captions, and four
  figure captions. The expected reported rates and title/author metadata
  remain present. All 24 citation links resolve to the references in the final
  page's right column. All six pages and detailed views of the opening page,
  authorization equation, and references received independent visual review
  for clipping, gutter collisions, caption placement, and figure readability.
- Private research is credited only in general terms. Publication text,
  supporting artifacts, and PDF text and links were checked for private
  repository identifiers, record filenames, and commit IDs. The judge is
  identified as the author's prior Qwen3.6 checkpoint, with its model-card
  link separate from the human spot-check evidence.
- Extracted PDF text has no replacement glyphs or words outside page bounds.
  Figure labels use vector paths and are covered by visual inspection.
- Local Markdown file targets resolve, including the source-line references
  in the review. This does not claim that all local evidence is public, or
  that every external URL will remain available.
- The README's GCD/GLP guidance was reconciled with the paper and source,
  including artifact selection, calibration, schemas, and current composition
  behavior. Its revised section's local links and CLI flag declarations were
  checked without loading a model.
- `git diff --check` and `git diff --cached --check` pass. The article PDF is marked binary in the scoped
  documentation attributes file so Git does not mistake its long textual
  preamble and compressed payload for a source-code diff.

## Tools and reproduction

Python 3.13.12; Matplotlib 3.11.1; openpyxl 3.1.5; jsonschema 4.26.0;
PyMuPDF 1.28.2; Pandoc 3.11; Typst 0.15.1.

From the repository root:

```bash
python3 scripts/grammar_probe/publication_data.py --out /tmp/gcd-recomputed.json
python3 scripts/grammar_probe/publication_figures.py
python3 scripts/grammar_probe/publication_pdf.py
git diff --check
```

The aggregation command requires the exact raw local records; figure and PDF
rendering use the included aggregate snapshot. See [README.md](README.md) for
file roles and dependency requirements. A new PDF render can differ in file
hash because its creation timestamp changes.

## Artifact identity at validation

- `docs/gcd-in-hf2q.md`
  SHA-256: `89e26f4ff43da0af114b09853ae380a17545d69dc01c3345de2b8ed6fa327e04`
- `docs/gcd-in-hf2q.pdf`
  SHA-256: `c0d4c30a47cab6778272719845ec4b1d3d876e47a13124ec7f5052d832f460c6`
- `docs/figures/gcd/evidence.json`
  SHA-256: `090efc7c3deecd0366b806c49db8580462dfc38ed6ff0cb9169f8bcd89583430`

## Remaining implementation work

No runtime source was changed by this review. No Rust/Metal compilation,
model loading, fresh generation, serving conformance, real-model cache test,
or performance measurement was run. Those remain with the implementation
owner, together with corrected harnesses and source-bound replacement results.
The [publication review](../../gcd-in-hf2q-review.md) gives the exact findings
and required evidence. Code fixes cannot remove limitations intrinsic to
historical records.
