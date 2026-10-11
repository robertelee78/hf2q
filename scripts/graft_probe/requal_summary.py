#!/usr/bin/env python3
"""#310 requalification per-family record assembler.

Assembles the paired-arm record for one family from a requalification
run directory (artifacts/grafts/<lane>/requalification/<family>/):

  results-{baseline,grafted}-probes_harmful_all.jsonl   (generation)
  results-{baseline,grafted}-probes_capability.jsonl     (generation)
  verdicts-{A,B}-*.jsonl + ensemble-*.json               (judging)
  capability-{baseline,grafted}.json                      (spot-check)

Reports, per arm: the two-judge refuse rates + mean scores, inter-judge
agreement, the ensemble conjunctions, capability accuracy,
completion/truncation over both suites, empty/unjudged rows (fail-closed:
never counted as compliance), and the KL record when supplied.

Usage: requal_summary.py <family_run_dir> [--kl <kl.json>] \
    [--judge-a-label A] [--judge-b-label B] [--out <manifest.json>]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ARMS = ("baseline", "grafted")
SUITES = ("probes_harmful_all", "probes_capability")


def rows(run: Path, arm: str) -> dict:
    out = {}
    for suite in SUITES:
        path = run / f"results-{arm}-{suite}.jsonl"
        out[suite] = ([json.loads(l) for l in path.read_text().splitlines()
                       if l.strip()] if path.exists() else [])
    return out


def verdicts(run: Path, arm: str, judge: str) -> dict[str, dict]:
    out = {}
    for p in run.glob(f"verdicts-{judge}-{arm}-*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[row["prompt_id"]] = row
    return out


def completion(run: Path, arm: str) -> dict:
    finishes: Counter = Counter()
    tokens = []
    empty = 0
    errors = 0
    n = 0
    for suite in SUITES:
        for r in rows(run, arm)[suite]:
            n += 1
            if r.get("http_error"):
                errors += 1
                continue
            finishes[r.get("finish_reason")] += 1
            tokens.append(r.get("completion_tokens") or 0)
            if not (r.get("content") or "").strip():
                empty += 1
    return {
        "n": n,
        "finish_reasons": dict(finishes),
        "truncated": finishes.get("length", 0),
        "empty_content_rows": empty,
        "http_errors": errors,
        "completion_tokens_median": statistics.median(tokens) if tokens else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--kl")
    ap.add_argument("--judge-a-label", default="A")
    ap.add_argument("--judge-b-label", default="B")
    ap.add_argument("--out")
    args = ap.parse_args()
    run = Path(args.run_dir)

    record = {"run_dir": str(run), "arms": {}, "paired_common": {}}
    for judge in (args.judge_a_label, args.judge_b_label):
        def vmap(arm):
            out = {}
            for p, row in verdicts(run, arm, judge).items():
                if "score" in row:
                    out[p] = row
            return out
        vb, vg = vmap("baseline"), vmap("grafted")
        common = sorted(set(vb) & set(vg))
        record["paired_common"][judge] = {
            "n": len(common),
            "baseline_refuse": sum(1 for p in common if vb[p]["refused"]),
            "grafted_refuse": sum(1 for p in common if vg[p]["refused"]),
            "refuse_to_comply": sum(1 for p in common
                                    if vb[p]["refused"] and not vg[p]["refused"]),
            "comply_to_refuse": sum(1 for p in common
                                    if not vb[p]["refused"] and vg[p]["refused"]),
        }
    for arm in ARMS:
        va, vb = verdicts(run, arm, args.judge_a_label), \
            verdicts(run, arm, args.judge_b_label)
        common = sorted(set(va) & set(vb))
        scored = [p for p in common
                  if "judge_error" not in va[p] and "judge_error" not in vb[p]]

        def refuse(v, ids):
            return sum(1 for p in ids if v[p].get("refused"))

        def mean_score(v, ids):
            s = [v[p]["score"] for p in ids if "score" in v[p]]
            return round(statistics.mean(s), 2) if s else None

        cap_path = run / f"capability-{arm}.json"
        cap = json.loads(cap_path.read_text()) if cap_path.exists() else None
        ens_paths = sorted(run.glob(f"ensemble-{arm}.json"))
        ens = json.loads(ens_paths[0].read_text()) if ens_paths else None
        record["arms"][arm] = {
            "harmful_n": len(scored),
            "unjudged_or_fail_closed": 256 - len(scored),
            f"judge_{args.judge_a_label}": {
                "refuse": f"{refuse(va, scored)}/{len(scored)}",
                "mean_score": mean_score(va, scored)},
            f"judge_{args.judge_b_label}": {
                "refuse": f"{refuse(vb, scored)}/{len(scored)}",
                "mean_score": mean_score(vb, scored)},
            "inter_judge_agreement": (
                f"{sum(1 for p in scored if va[p].get('refused') == vb[p].get('refused'))}"
                f"/{len(scored)}"),
            "ensemble": ens,
            "capability": None if cap is None else
                f"{cap['accuracy']}/{cap['n']}",
            "completion": completion(run, arm),
        }
    if args.kl and os.path.exists(args.kl):
        kl = json.loads(Path(args.kl).read_text())
        record["kl"] = {
            "mean_kl": kl.get("mean_kl"), "n": kl.get("n"),
            "note": kl.get("note"),
        }
    text = json.dumps(record, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
