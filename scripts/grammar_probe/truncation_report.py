#!/usr/bin/env python3
"""Cross-tab existing judgments and termination with checked observation joins.

This performs no new judging. Fulfillment labels and output-validity flags
remain distinct; termination does not establish usefulness or degeneration.
Run/config/budget cells remain separate. Historical reports are never replaced.
Usage: python3 -B truncation_report.py RESULTS VERDICTS [NEW_OUTPUT]
"""
import json
from pathlib import Path
import sys

from measurement import artifact, read_jsonl
from report import build_report


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    results, verdicts = map(Path, sys.argv[1:3])
    output = (Path(sys.argv[3]) if len(sys.argv) == 4 else
              results.with_name(results.stem + ".truncation-report.v3.json"))
    if output.exists():
        raise ValueError("report output already exists; choose a new artifact path")
    report = build_report(read_jsonl(results), read_jsonl(verdicts))
    report["sources"] = {"results": artifact(results), "verdicts": artifact(verdicts)}
    report["note"] = ("Cross-tab of the selected existing scoring pass; no new judgments. "
                      "Response-state labels do not override contradictory validity flags.")
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
