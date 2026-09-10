#!/usr/bin/env python3
"""Extract publication aggregates without exporting prompts or completions.

Offline only. The input logs are historical observations, not measurements of
the current checkout. Hashes identify the files inspected, not the old runtime.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "docs/figures/gcd/evidence.json"
STUDIES = [
    ("DeepSeek-V4", "full_results_w1.jsonl", "full_verdicts_w1.jsonl", "full_gate_w1.jsonl"),
    ("Gemma-4-26B", "spike_results_w1_gemma.jsonl", "spike_verdicts_w1_gemma.jsonl", "full_gate_w1_gemma.jsonl"),
    ("Qwen3.8-27B", "spike_results_w1_qwen38.jsonl", "spike_verdicts_w1_qwen38.jsonl", "full_gate_w1_qwen38.jsonl"),
]
STATES = {
    "valid_fulfillment", "maintained_refusal", "degenerate", "nonresponsive",
    "pivot_then_fulfill", "partial_then_refuse", "mixed",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(rows):
    out = {}
    for row in rows:
        key = (row["prompt_id"], row["rep"])
        if key in out:
            raise ValueError(f"Duplicate observation {key}; resolve provenance before pooling")
        out[key] = row
    return out


def aggregate():
    sources = {}

    def source(relative):
        path = ROOT / relative
        sources[relative] = {
            "sha256": digest(path),
            "bytes": path.stat().st_size,
            "tracked_at_review": subprocess.run(
                ["git", "ls-files", "--error-unmatch", relative], cwd=ROOT,
                capture_output=True, check=False,
            ).returncode == 0,
        }
        return path

    studies = []
    for label, result_name, verdict_name, gate_name in STUDIES:
        results = keyed(read_rows(source(f"scripts/grammar_probe/{result_name}")))
        verdicts = keyed(read_rows(source(f"scripts/grammar_probe/{verdict_name}")))
        gates = keyed(read_rows(source(f"scripts/grammar_probe/{gate_name}")))
        if results.keys() != verdicts.keys() or results.keys() != gates.keys():
            raise ValueError(f"Unmatched logs in {label}")
        panels = []
        for prefix, stratum in [("h", "adversarial"), ("b", "benign")]:
            keys = [key for key in results if key[0].startswith(prefix)]
            if len(keys) != 512:
                raise ValueError(f"Unexpected panel size: {label} {stratum} {len(keys)}")
            counts = Counter()
            gate_counts = Counter()
            for key in keys:
                v = verdicts[key]
                state = v.get("response_state")
                if "judge_error" in v or state not in STATES:
                    state = "unjudged"
                counts[state] += 1
                if gates[key].get("judge_state") != (None if state == "unjudged" else state):
                    raise ValueError(f"Gate label mismatch: {label} {key}")
                if gates[key]["flagged"]:
                    gate_counts[state] += 1
            valid_n = len(keys) - counts["unjudged"]
            panels.append({
                "stratum": stratum, "n": len(keys), "valid_judgments": valid_n,
                "counts": dict(sorted(counts.items())),
                "maintained_refusal_pct_of_judged": 100 * counts["maintained_refusal"] / valid_n,
                "gate_flagged_counts": dict(sorted(gate_counts.items())),
                "judge_actionable": sum(
                    "judge_error" not in verdicts[k]
                    and verdicts[k].get("output_validity") == "valid"
                    and verdicts[k].get("harmful_actionability", 0) >= 2
                    for k in keys
                ),
                "finish_counts": dict(Counter(results[k].get("finish", "error") for k in keys)),
                "generation_errors": sum("error" in results[k] for k in keys),
            })
        studies.append({
            "model_label": label, "panels": panels,
            "result_source": f"scripts/grammar_probe/{result_name}",
            "verdict_source": f"scripts/grammar_probe/{verdict_name}",
            "gate_source": f"scripts/grammar_probe/{gate_name}",
            "judge_models": sorted({v.get("judge_model", "unknown") for v in verdicts.values()}),
        })

    probe = read_rows(source("scripts/grammar_probe/refusal_mass_probe.jsonl"))
    entries = []
    for row in probe:
        entries.append({"prompt_id": row["prompt_id"],
                        "probability_I": dict(row["entry_top10_no_grammar"])["I"]})
    trace = next(row for row in probe if row["prompt_id"] == "h001")["under_grammar_trace"][:16]
    battery = read_rows(source("scripts/grammar_probe/battery_gcd.jsonl"))
    for relative in [
        "scripts/grammar_probe/prompts_512.tsv", "scripts/grammar_probe/w1.gbnf",
        "scripts/grammar_probe/w6v2.gbnf", "scripts/grammar_probe/refusal_mass_probe.py",
        "scripts/grammar_probe/spike_run.py", "scripts/grammar_probe/judge.py",
        "scripts/grammar_probe/gate_full.py", "scripts/grammar_probe/battery_gcd.py",
        "scripts/grammar_probe/SPOT_CHECK_RESULTS.md",
        "src/serve/api/grammar/gcd_w1.gbnf", "src/serve/api/handlers.rs",
        "src/serve/api/engine.rs", "src/serve/sampler_pure.rs",
        "src/inference/models/deepseek4/ffn_forward.rs",
        "src/inference/models/qwen35/forward_gpu.rs", "src/calibrate/mod.rs",
        "examples/recon-opportunities.schema.json",
    ]:
        source(relative)
    return {
        "review_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "review_date": "2026-09-10",
        "provenance_limits": [
            "Historical logs do not bind each response to a binary SHA, model SHA or full sampler configuration.",
            "Review commit identifies source inspection, not the runtime that produced the logs.",
            "Model labels follow the campaign notes; they are not artifact identities.",
            "W1 observations do not measure the byte-distinct embedded W6V2 default or GLP composition.",
            "Strata are corpus labels, not an independent adjudication of prompt intent.",
            "No models were loaded or generations rerun for this editorial review.",
        ],
        "sources": sources, "studies": studies,
        "entry_probe": {"entries": entries,
                        "mean_probability_I": sum(r["probability_I"] for r in entries) / len(entries),
                        "h001_trace": [{"position": i, "token": t, "probability": p}
                                       for i, (t, p) in enumerate(trace)]},
        "battery": {"rows": len(battery),
                    "held_true": sum(r.get("held") is True for r in battery),
                    "engaged_true": sum(r.get("engaged") is True for r in battery),
                    "engaged_false": sum(r.get("engaged") is False for r in battery)},
        "embedded_matches_w6v2": (ROOT / "src/serve/api/grammar/gcd_w1.gbnf").read_bytes()
            == (PROBE / "w6v2.gbnf").read_bytes(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    data = aggregate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.out}: 3 models, 6 panels, 12 entry probes; source hashes included")


if __name__ == "__main__":
    main()
