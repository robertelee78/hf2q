# Grammar-control measurement methods

The measured unit is a generated response identified by run, configuration,
arm, prompt, repetition and budget. Grammar comparisons use identical model
artifacts, templates, prompts, sampling settings and token budgets within each
pair. Refusal, substance, validity and termination are distinct outcomes.
`finish:length` records budget exhaustion; `finish:stop` does not establish
semantic completeness or correctness.

## Runtime and generation binding

New generation and scoring require a managed runtime attestation:
`RUNTIME_MANIFEST` for generation and `JUDGE_RUNTIME_MANIFEST` for scoring.
The harness checks the running process and exact live `/hf2q/v1/runtime`
measurement snapshot against the launcher manifest before and after requests.
The manifest binds binary/model artifacts, tokenizer, effective template,
sampling defaults and active controls. The requested model must equal the
attested resident model. Source commit may be explicitly unknown; the harness
never manufactures a binary-to-source relationship or historical identity.

`baseline_run.py` generates paired BASE and W1 responses. BASE requires
verified inactive default grammar, GLP, DWQ overlays and vision projector. A literal reply cannot
prove these conditions; the runtime attestation supplies that evidence.
Separate canaries check the grammar path and historical W1 opening. The W1
file must match its historical SHA256, not the newer embedded default.

`BUDGET_LADDER` declares every budget before the run. Defaults are 800 tokens,
greedy temperature, no system prompt, disabled thinking and low reasoning
effort. These reproduce the historical request settings, not its unrecorded
runtime identity. Arm order alternates across prompt/repetition/budget cells.
Request overrides and the complete runtime identity enter the configuration
hash. Rows preserve budget, prompt/content hashes and run/configuration IDs.

Existing generation files require their matching manifest. Changed corpus,
settings, grammar, budget ladder or runtime binding requires a new output.
`DRY_RUN=1` plans generation without contacting or loading a model.

## Full-response scoring and resumption

`judge.py` writes schema-v3 records. The full response and complete prompt are
sent to the judge with a separate generation metadata block. The harness never
clips or summarizes them. A declared input limit or API failure produces an
explicit failed evaluation. A judgment requiring `valid_fulfillment` must have
valid output and substance >=2; degeneration/nonresponse requires invalid
output. Contradictory judgments remain failures, not outcome labels.

The judgment-input digest includes prompt/response hashes, full observation
identity, termination and token usage. Scoring configuration binds the current
judge runtime, rubric/schema hashes and sampling settings. Successful matching
attempts resume without duplication; failed attempts retain bounded diagnostics
and can retry up to `JUDGE_MAX_ATTEMPTS`. `JUDGE_FORCE_RETRY=1` permits further
failed-attempt retries. Source and scoring manifests are immutable: changed
inputs or configuration require a new output/pass.

`rejudge.py` reads retained generations and writes a new `rejudge_v3_<label>`
directory containing `verdicts_v3.jsonl`, a source/scorer manifest and optional
old/new transitions. `--resume` validates the existing manifest. Hashes measured
now identify retained files, not the historical generation binary. Original
v1/v2 judgments remain unchanged and readable.

## Reporting and uncertainty

`report.py` preserves run/configuration/budget cells. Supplied response hashes
must match; ambiguous joins fail. Legacy rows missing identities can join only
when unambiguous and remain explicitly marked legacy. They are not promoted
to schema v3. Orphan verdicts and inconsistent historical fields are visible.

Select `JUDGE` and `PASS` explicitly when several scoring configurations are
present. An absent requested pass or ambiguous default fails; no alternative
pass is silently substituted. Latest successful attempts are selected within
the same complete observation and judgment input.

Reports include all generation observations, successful responses, generation
errors, judge failures and unjudged responses. Termination counts and rates use
all generated responses, including those without judgments. Semantic rates
show both labeled events per response and events per valid judgment; the
former are observed fractions, not imputed labels for missing judgments.

Paired semantic differences use shared judged responses within the same run,
configuration and budget. Termination differences use all shared generated
responses. Missing generations/judgments are counted separately. Bootstrap
95% intervals cluster by prompt so repeated responses travel together; seed
and resample count are reported. Selection and judge uncertainty are additional
limitations that bootstrap intervals do not remove.

## Independent human review

`rejudge.py --sample N` draws uniformly without replacement from all retained
responses, independent of machine agreement or judgment availability. Cases
contain complete prompt/response text, hashes and termination metadata, with
machine judgments and run/arm identities withheld in a separate key file.
Reviewers record independent labels before consulting that key.

`--diagnostic-sample N` produces a separate uniformly sampled set of old/new
label disagreements, with its own population and key. It is useful for error
diagnosis; it does not estimate corpus-wide agreement. Both sample types record
the population, selection procedure and seed. Full responses, not excerpts,
are used for both.

## Offline validation

Run `PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/grammar_probe/test_harness_repair.py`.
The suite uses synthetic data, temporary outputs and mock HTTP servers. It
covers multi-budget/run joins, hashes, pass selection, missingness, immutable
resumption, full human cases and runtime controls. It loads no model. Offline
correctness does not certify a judge's semantic accuracy or validate the
historical behavioral conclusions; those require a new scoring pass and
independent human labels.
