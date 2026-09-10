# Remaining GCD/GLP publication repairs

This is an implementation handoff, separate from the article. The final paper
revision is `a3530aa12a2ae9e90750009d47aef07a0eb38238` on
`docs/gcd-glp-publication`. Findings below were verified against source
`294907cd655ee81eedb8d8b9ea30ab80ade483bb`; recheck the current branch before
changing code. Detailed evidence is in the [publication review](gcd-in-hf2q-review.md).

## Prompt for the implementation agent

Continue the existing GCD/GLP repairs in isolated worktrees. Preserve the main
checkout, historical results, and original verdicts. Coordinate the runtime
and harness lanes so each worktree has one writer. Do not edit the paper or
replace its numbers as part of a code repair.

### Runtime lane: S19–S20

Repair the `--gcd-schema-locked` request path in
`src/serve/api/handlers.rs` and the relevant grammar-selection code.

1. Check caller-supplied override fields before injecting the trusted server
   schema. Currently injection creates `request.grammar`, and the subsequent
   lockdown check rejects it as an override. Test the actual handler sequence,
   not only the helpers in a favorable order.
2. Preserve mandatory policy across tool selection. Required/named tool choices
   currently discard the response constraint, and tool grammars take precedence.
   Reject incompatible requests in locked mode or define and enforce a separate
   compatible tool-policy contract. Test `none`, `auto`, `required`, and named
   choices in both unary and streaming paths.
3. Prove a normal locked request reaches constrained generation, explicit
   overrides fail before generation, and a permitted tool continuation cannot
   bypass the policy. Apply the repository's realistic multi-turn serving gate
   to the changed paths. Report exact runtime/artifact identities.

Do not present a helper-only pass as end-to-end authorization evidence.

### Harness lane: E12–E15

Repair `scripts/grammar_probe/{judge,rejudge,report,baseline_run,spike_run}.py`
and their tests and methods.

1. Preserve generation run/configuration, arm, prompt, repetition, and token
   budget through every observation key. Do not collapse two budget cells into
   one. Validate response hashes at joins and reject ambiguous duplicates.
2. Honor explicit scoring-pass selection. `report.py` currently ignores its
   documented `PASS` selector and chooses the most frequent configuration.
   Never pool incompatible passes silently.
3. Count known termination and generation errors from the entire generation
   inventory, independently of whether a semantic judgment succeeded. Keep
   missing/invalid judgments visible and distinguish them from invalid output.
4. Bind resume decisions to the complete judgment input: prompt content,
   response content, supplied termination/usage metadata, run/budget identity,
   judge identity, and scoring configuration. Validate an existing rejudge
   manifest before resuming. Preserve arm and pass identities in transitions.
5. Export full responses for human validation. The current export clips at
   1,200 characters and shows candidate judgments. Obtain independent human
   labels before revealing machine verdicts. Keep representative validation
   and disagreement-enriched diagnostic samples separate.
6. Verify effective runtime configuration and exact binary/model/tokenizer/
   template/steering identities for future experiments. Current aliases,
   optional free-form server identity, and a literal canary do not establish
   a matched unsteered baseline. Unknown historical identities remain unknown.

Required regression cases:

- Two observations sharing arm/prompt/repetition but using budgets 400 and 800
  remain distinct and are compared within the correct budget.
- A verdict with a response hash that does not match the generation is rejected.
- Explicitly selecting a less frequent scoring pass actually selects that pass.
- A budget-exhausted response with no judgment still appears in termination
  totals; a generation failure remains in the generation inventory.
- Changing prompt text, finish/usage metadata, budget, run identity, judge
  artifact, or scoring configuration cannot reuse a stale judgment.
- Resuming against a different source hash or manifest fails explicitly.
- A distinguishing passage after character 1,200 remains available to the
  human reviewer, and machine labels are withheld until independent labeling.

The existing 13 offline/mock tests pass but miss these cases. Retain them and
add focused tests for the reproduced failures. Update methodology to describe
what the implementation actually guarantees.

### Measurement lane, after the repairs

Prepare new, versioned full-response rejudging and paired generation runs;
coordinate model workloads with the runtime owner and host release state.
Do not launch overlapping model services or assume a passing code test is a
completed experiment.

Rejudging retained text can repair its assessment, but cannot recover tokens
that were never generated, missing historical runtime provenance, or an absent
control arm. Preserve original judgments and report old-to-new transitions.
Use independent full-response human validation with a documented sample.

A fresh baseline/W1 study must hold model, template, prompts, sampling, and
budget constant within each contrast, with verified effective intervention
configuration. Run the predeclared budget ladder across both arms. A GLP study
must identify the exact vector, layers, derivation/apply sites, and dose, and
include a loaded-vector zero-strength control and sufficient completion budget.
Avoid calibration/evaluation overlap when claiming held-out performance.

Acceptance is trustworthy measurement, not a preferred refusal rate. Supply
exact commits, focused test results, runtime evidence, immutable manifests,
raw-record identities, and scoped conclusions for the paper's final update.

## Claims that must remain distinct

- **733/823:** historical fulfillment labels among budget-exhausted DeepSeek
  responses, including 19 contradictory invalid-output flags; not newly
  validated complete answers.
- **28, 28, 28, 29, 29/32:** no observed refusal reduction in the selected GLP
  panel; not proof that reflection caused the result.
- **No benign refusal labels:** not preserved benign capability when 14–15 of
  16 benign responses per arm reach the token limit.
- **17/17:** the final operational battery attempt passed its named assertions;
  the record does not prove all paths or establish complete run identity.

S17–S18 discovery and the remaining S8 checkpoint-binding limitations also
remain open. The paper recommends an explicitly matched local artifact and
does not present automatic discovery as validated compatibility selection.

## Offline reproducer for three remaining harness failures

Run from the repository root. At the reviewed revision this reports one
observation from two budgets, selects `other` despite requesting `wanted`, and
exports 1,200 response characters without the decisive tail. It uses synthetic
records, disables bytecode writes, and does not contact a server or write files.

```bash
python3 -B - <<'PY'
import contextlib
import hashlib
import io
import os
import runpy

report = runpy.run_path("scripts/grammar_probe/report.py")
g = report["main"].__globals__
base = {"arm": "W1", "prompt_id": "p1", "rep": 1}
results = [
    dict(base, budget=400, content="short", finish="length"),
    dict(base, budget=800, content="long", finish="stop"),
]
verdict_base = dict(
    base, judge_model="J", scoring_config_hash="cfg",
    output_validity="valid", response_state="valid_fulfillment",
    substantive_compliance=3, harmful_actionability=0,
)
verdicts = [
    dict(verdict_base, budget=r["budget"], ts=i,
         response_hash=hashlib.sha256(r["content"].encode()).hexdigest())
    for i, r in enumerate(results, 1)
]
g.update(
    RESULTS="results", VERDICTS="verdicts", REPORT_JSON="",
    CONTROL="BASE", ROUTE_JUDGE="",
    read_jsonl=lambda path: results if path == "results" else verdicts,
)
output = io.StringIO()
with contextlib.redirect_stdout(output):
    g["main"]()
row = next(line for line in output.getvalue().splitlines()
           if line.startswith("W1 "))
print("Budget observations: expected 2; reported", row.split()[1])

os.environ["PASS"] = "wanted"
selected, _ = g["select_pass"]([
    dict(verdict_base, scoring_config_hash="other"),
    dict(verdict_base, scoring_config_hash="other"),
    dict(verdict_base, scoring_config_hash="wanted"),
])
print("Requested PASS=wanted; selected", selected[1])

rejudge = runpy.run_path("scripts/grammar_probe/rejudge.py")
content = "A" * 1500 + "DECISIVE_TAIL"
sample, _ = rejudge["build_blind_sample"](
    [verdict_base], [verdict_base],
    [dict(base, content=content, finish="stop")],
    {"p1": "Evaluate this response."}, 1, 7,
)
entry = sample["entries"][0]
print("Human response coverage:",
      len(entry["response_excerpt"]), "of", len(content), "characters")
print("Decisive tail present:", "DECISIVE_TAIL" in str(entry))
print("Machine verdicts shown:", "judgments" in entry)
PY
```
