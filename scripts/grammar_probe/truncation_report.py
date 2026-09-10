#!/usr/bin/env python3
"""truncation_report.py — honest finish-reason × verdict classification.

Section 3 of the GLP/GCD implementation review: `finish: length` means
budget exhaustion. It does not by itself prove grammar violation, engine
fault, or useless content — and it must not be silently conflated with
either refusal suppression or degeneration. This script cross-tabulates
finish reasons against judge verdicts for a (results, verdicts) pair and
writes the table to a NEW artifact (original logs are never modified).

Classification taxonomy (per the review):
  - budget_exhausted_judged_valid  — hit the token budget AND the judge
    scored the content a valid fulfillment (substantive content, cut off)
  - budget_exhausted_degenerate     — hit the budget AND judged degenerate
    (repetition class)
  - budget_exhausted_other          — hit the budget with any other judge
    state (refusal, pivot, nonresponsive, unjudged)
  - natural_stop_*                  — the same split for finish: stop

Engine behavior note (verified in source): an open-body grammar (W1/W6V2
prose) admits termination at any body position, so a budget-truncated body
is a legal member of the language and passes terminal validation —
`finish: length` with content is the honest result. A closed schema
grammar fails loudly at truncation instead. Neither is a defect; the
defect this script repairs is reporting rates without this context.

Usage:
  python3 truncation_report.py RESULTS_JSONL VERDICTS_JSONL [OUT_JSON]
"""
import json
import sys
from collections import Counter
from pathlib import Path

BUDGET_NOTE = (
    "budget: 800 completion tokens (spike_run.py); a budget comparison at "
    "2048 is a separate, documented experiment — this report does not "
    "raise or reinterpret the historical budget"
)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    results_path, verdicts_path = Path(sys.argv[1]), Path(sys.argv[2])
    out_path = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else results_path.with_name(results_path.stem + ".truncation-report.json")
    )

    rows = [json.loads(line) for line in results_path.open()]
    verdicts = {}
    for line in verdicts_path.open():
        v = json.loads(line)
        verdicts[(v["prompt_id"], v["rep"])] = v

    cross = Counter()
    unjudged = 0
    for r in rows:
        v = verdicts.get((r["prompt_id"], r["rep"]))
        state = (v or {}).get("response_state")
        if state is None:
            unjudged += 1
            state = "unjudged"
        finish = r.get("finish") or r.get("finish_reason") or "unknown"
        cross[(finish, state)] += 1

    def bucket(finish: str) -> str:
        return "budget_exhausted" if finish == "length" else "natural_stop"

    classified = Counter()
    for (finish, state), n in cross.items():
        if state == "valid_fulfillment":
            key = f"{bucket(finish)}_judged_valid"
        elif state == "degenerate":
            key = f"{bucket(finish)}_degenerate"
        else:
            key = f"{bucket(finish)}_other"
        classified[key] += n
        classified[f"{bucket(finish)}_total"] += n

    report = {
        "results_file": str(results_path),
        "verdicts_file": str(verdicts_path),
        "rows": len(rows),
        "unjudged": unjudged,
        "finish_x_state": {
            f"{finish}|{state}": n for (finish, state), n in sorted(cross.items())
        },
        "classified": dict(classified),
        "note": BUDGET_NOTE,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["classified"], indent=2))
    print(f"unjudged: {unjudged} / {len(rows)}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
