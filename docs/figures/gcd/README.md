# GCD/GLP article figures and evidence

These assets accompany [the article](../../gcd-in-hf2q.md). This package contains
historical W1 aggregates, later recorded GLP and battery observations, and
explanatory figures. Publication tools do not run inference.

| Article figure | Asset stem | Basis |
|---|---|---|
| 1 | `generation-controls` | Conceptual generation loop and intervention sites |
| 2 | `glp-projection` | Geometric illustration of projective GLP |
| 3 | `entry-trace` | Twelve archived entry-token observations and one W1 trace |
| 4 | `w1-outcomes` | All 3,072 historical W1 observations, split by model and stratum |

Every figure has SVG and PNG exports. SVG text is converted to vector paths
so print rendering does not depend on locally installed plot fonts. The
Markdown supplies descriptive alternative text and separate captions. The
plotting script is the editable source. The older figure files remain intact
outside this directory and are no longer embedded in the article.

`evidence.json` preserves the historical plotted observations, outcome and
termination counts, source hashes, and provenance limits. The separate
`followup-evidence.json` records the later source snapshot, all five GLP panel
arms, historical termination/state/validity cross-tabs, calibration overlap,
and all four battery attempts. It retains earlier failures and skips rather
than silently selecting only the final passing attempt. `outcomes.csv` exposes every outcome
category. `figure-data.xlsx` includes outcome formulas, entry probabilities,
the token trace, termination counts, judging audit, and source hashes. Additional
sheets contain GLP counts and paired label transitions, battery attempts, and
separate follow-up source identities. Small chart segments remain in the bars
even when there is insufficient room for an interior label.

The JSON, CSV, and workbook permit readers to inspect the plotted values.
Independently validating the classifications requires the underlying campaign
records. Some of those records are local-only; their access status is recorded
in the manifest. Model labels and the source-review commit are not historical
runtime or model-artifact identities.

## Reproduce the figures and PDF

From the repository root, use Python with Matplotlib and openpyxl installed:

```bash
python3 scripts/grammar_probe/publication_figures.py
python3 scripts/grammar_probe/publication_pdf.py
```

The first command regenerates the four figures, CSV, and workbook from the
existing aggregate snapshots without loading raw logs. The second needs Pandoc
and Typst on `PATH` and renders `docs/gcd-in-hf2q.pdf` from the Markdown.

The PDF uses a compact two-column ML-paper layout, with a full-width title,
author/affiliation block, and abstract. It retains numbered sections and
equations, captioned tables, floating figures, and bracketed references.
Times New Roman supplies the body, STIX Two Math supplies mathematics, and
DejaVu Sans Mono supplies code. The body is 10 pt, with a 6 mm column gutter,
18 mm horizontal margins, and 20 mm vertical margins on A4. The four figures
and Tables 3–5 span both columns; Tables 1–2 fit within one column. Long
equations are aligned over multiple lines without changing their meaning.
The bibliography follows the back matter. The presentation rules live in
[`publication_style.typ`](../../../scripts/grammar_probe/publication_style.typ).
The renderer preserves each image/caption pair and links numeric citations
to their bibliography entries. An optional output argument permits rendering
a preview elsewhere.

If the exact raw campaign files are available, recompute aggregates into a
separate file before reviewing any change to the published snapshot:

```bash
python3 scripts/grammar_probe/publication_data.py \
  --review-commit 44004311717d414feaa54384578e2bcf4d140464 \
  --out /tmp/gcd-recomputed.json
```

Use `--source-root /path/to/research-checkout` when the raw records live in
another checkout. Source-only inputs are read from the requested Git revision;
tracked observation files must match that revision. This permits reproduction
after later source repairs without assigning new code the historical identity.
Omitting `--review-commit` records that checkout's current HEAD.
The extractor rejects duplicate observation keys and mismatched generation,
judge, and screen joins. It retains failed judgments, contradictory validity
fields, and evidence of judging-side clipping. Compare both statistics and input hashes; do not replace
the historical snapshot merely because a newer runtime has been fixed.

Recompute the separate follow-up manifest with:

```bash
python3 scripts/grammar_probe/publication_followup.py \
  --source-root /path/to/research-checkout \
  --out /tmp/gcd-followup-recomputed.json
```

The follow-up extractor reads tracked artifacts at its explicit reviewed
commit and verifies local historical records against `evidence.json`. It
exports counts and hashes, without absolute host paths, prompts, or completions.

The validated tool versions and checks are in [validation.md](validation.md).
