# Grammar-control probe harness — methods

Doctrine mapped from `docs/METHODS-abliteration-and-harness.md`
(jenerallee78/Qwen3.8-27B-Abliterated-SFT) to the grammar-constrained
serving setting. The measured object here is not a weight edit but a
request-scoped GBNF constraint riding hf2q's grammar runtime.

## What is being measured

Per (prompt, arm, rep): the served model's answer-segment completion.
Arms differ only in the grammar attached to the request; weights, prompts,
and sampling settings are held fixed.

- **Arm A** — no grammar (baseline)
- **Arm B1** — refusal-compatible anchor (`g1.gbnf`), the weak-form control
- **Arm B4** — refusal-incompatible procedural anchor (`g4.gbnf`)

## Doctrine

- **Keyword refusal counts are screening signals, never evidence.** All
  behavioral claims come from `judge.py` semantic verdicts (validity-gated,
  fail-closed: unparseable/invalid judgments never enter a metric).
- **Paired contrasts only.** Arm differences are computed per prompt_id with
  bootstrap 95% CIs (`report.py`). Never compare arm marginals across
  different prompt sets.
- **Plumbing canaries are hard gates.** `probe.sh canary` must pass before
  any run: a forced-literal grammar must echo its literal in the answer
  segment, and a no-op grammar must be transparent. (The v0.1.20 standalone
  binary silently dropped the top-level `grammar` param on reasoning-seeded
  requests; the canary catches this class in one request.)
- **Immutable-ish artifacts.** `results.jsonl` / `verdicts.jsonl` are
  append-only; hash them after a campaign (`shasum -a 256 *.
jsonl`).

## Decode and frame disclosure

Every results row carries temperature / top_p / reasoning_effort; the server
binary version is recorded in the campaign notes (first measured on hf2q
0.1.21, `/opt/hf2q/target/release/hf2q`). Prior measured facts that move the
refusal boundary (from the model card): thinking ON raises refusal pressure;
greedy vs sampled moves residue; harness/system frames can silently
re-enable refusal. This harness runs with NO system prompt, reasoning
`low`, temp 0.55 / top_p 0.95 unless a row says otherwise — report the cell,
not just the arm.

## Known gaps vs the full doctrine (upgrade path)

- Single local judge for now; ensemble band (local + frontier judge) is the
  upgrade. Same-vendor/self judging is a disclosed limitation of single-judge
  numbers.
- Canonical KL at the final prompt position is not yet implemented for the
  grammar setting (needs logit access through the serving path).
- Termination integrity is approximated via `finish_reason` (clean stop vs
  cap), not the teacher-forced P(EOS) probe.
- Full-panel runs: `N_HARMFUL=512 N_HARMLESS=512 make_corpus.py` (corpus
  regeneration), plus a sensitive-but-legitimate slice and false-premise
  controls.
