# GCD/GLP article figures and evidence

These assets accompany [the article](../../gcd-in-hf2q.md) and its
[publication review](../../gcd-in-hf2q-review.md). The review records source
defects and the evidence needed before publication. This package contains
historical aggregates and explanatory figures; it does not report a new
inference experiment.

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

`evidence.json` holds all plotted observations, outcome and termination counts,
input file hashes, and provenance limits. `outcomes.csv` exposes every outcome
category. `figure-data.xlsx` includes outcome formulas, entry probabilities,
the token trace, termination counts, and source hashes. Small chart segments
remain in the bars even when there is insufficient room for an interior label.

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
existing aggregate snapshot without loading raw logs. The second needs Pandoc
and Typst on `PATH` and renders `docs/gcd-in-hf2q.pdf` from the Markdown.

The PDF uses a compact two-column ML-paper layout, with a full-width title,
author/affiliation block, and abstract. It retains numbered sections and
equations, captioned tables, floating figures, and bracketed references.
Times New Roman supplies the body, STIX Two Math supplies mathematics, and
DejaVu Sans Mono supplies code. The body is 10 pt, with a 6 mm column gutter,
18 mm horizontal margins, and 20 mm vertical margins on A4. The four figures
and Tables 3–4 span both columns; Tables 1–2 fit within one column. Long
equations are aligned over multiple lines without changing their meaning.
The bibliography begins in a fresh column. The presentation rules live in
[`publication_style.typ`](../../../scripts/grammar_probe/publication_style.typ).
The renderer preserves each image/caption pair and links numeric citations
to their bibliography entries. An optional output argument permits rendering
a preview elsewhere.

If the exact raw campaign files are available, recompute aggregates into a
separate file before reviewing any change to the published snapshot:

```bash
python3 scripts/grammar_probe/publication_data.py --out /tmp/gcd-recomputed.json
```

The extractor rejects duplicate observation keys and mismatched generation,
judge, and screen joins. It retains failed judgments and records the current
source-review commit. Compare both statistics and input hashes; do not replace
the historical snapshot merely because a newer runtime has been fixed.

The validated tool versions and checks are in [validation.md](validation.md).
