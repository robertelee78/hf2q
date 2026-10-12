#!/usr/bin/env python3
"""#310 requalification capability spot-check scorer (GSM8K-class).

Scores a results file produced by requal_generate.py (suite
probes_capability) with the panel.sh extraction contract: the answer is
the final number in the content (plain arms; thinking is off per request
so there is no reasoning to contaminate extraction). Expected answers
come from probes_capability.jsonl by prompt id.

Usage: python3 requal_capability.py <results-capability.jsonl>
Prints {"accuracy": k/n, "details": [...]} JSON.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(x):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def main() -> int:
    path = sys.argv[1]
    answers = {}
    for line in open(os.path.join(HERE, "probes_capability.jsonl")):
        if line.strip():
            probe = json.loads(line)
            answers[probe["id"]] = probe.get("answer")
    correct = 0
    details = []
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        expected = answers.get(r.get("prompt_id"))
        content = r.get("content") or ""
        got = None
        m = re.search(r'"answer"\s*:\s*"([^"]*)"', content)
        if m:
            got = m.group(1).strip()
        if got is None:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", content)
            got = numbers[-1] if numbers else None
        ok = norm(got) is not None and norm(got) == norm(expected)
        correct += int(ok)
        details.append({"prompt_id": r.get("prompt_id"), "expected": expected,
                        "got": got, "ok": ok,
                        "finish_reason": r.get("finish_reason")})
    print(json.dumps({"accuracy": correct, "n": len(details),
                      "details": details}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
