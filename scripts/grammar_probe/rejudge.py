#!/usr/bin/env python3
"""Versioned full-response rejudging; immutable inputs and independent human cases.

Historical files are read-only. --sample selects a uniform corpus sample;
--diagnostic-sample separately selects old/new disagreements. Both exports
contain complete prompts/responses without machine verdicts. Their machine
labels and observation identities live in separate key files.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
import random

from measurement import (artifact, checked_join, digest, ensure_manifest,
                         generation_inventory, identity, read_jsonl,
                         select_scoring_pass, text_hash)
from runtime_identity import attested_model, fetch_runtime_identity, verify_unchanged

SCRIPT_DIR = Path(__file__).resolve().parent


def selected_labels(rows, results, selected_pass=None, judge=None):
    wanted, others = select_scoring_pass(rows, judge, selected_pass)
    selected = [r for r in rows if (r.get("judge_model", "unknown"),
                r.get("scoring_config_hash", "legacy-v1")) == wanted]
    labels, orphan = checked_join(results, selected)
    return labels, {"pass": wanted, "other_passes": others,
                    "orphan_observations": [identity(r) for r in orphan]}


def label(row):
    if row is None:
        return "unjudged"
    return "judge_error" if "judge_error" in row else row["response_state"]


def compute_transitions(old_rows, new_rows, result_rows, old_pass=None, new_pass=None):
    inventory = generation_inventory(result_rows)
    old, old_info = selected_labels(old_rows, inventory, old_pass)
    new, new_info = selected_labels(new_rows, inventory, new_pass)
    transitions = collections.Counter((label(old.get(k)), label(new.get(k))) for k in inventory)
    return {"n_results": len(inventory), "n_old_verdicts": len(old),
            "n_new_verdicts": len(new), "old_scoring": old_info, "new_scoring": new_info,
            "transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
            "relabels": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items()) if a != b}}


def build_blind_sample(old_rows, new_rows, results, prompts, n, seed,
                       diagnostic=False, old_pass=None, new_pass=None):
    inventory = generation_inventory(results)
    old, _ = selected_labels(old_rows, inventory, old_pass)
    new, _ = selected_labels(new_rows, inventory, new_pass)
    eligible = [k for k, r in inventory.items() if "error" not in r and "content" in r]
    if diagnostic:
        eligible = [k for k in eligible if k in old and k in new
                    and "judge_error" not in old[k] and "judge_error" not in new[k]
                    and label(old[k]) != label(new[k])]
    eligible.sort(key=str)
    picked = random.Random(seed).sample(eligible, min(n, len(eligible)))
    entries, keys = [], {}
    for number, k in enumerate(picked, 1):
        row = inventory[k]
        if row["prompt_id"] not in prompts:
            raise ValueError(f"missing full prompt for human case {row['prompt_id']}")
        prompt = prompts[row["prompt_id"]]
        entries.append({"sample_id": number, "prompt": prompt,
                        "response": row["content"], "prompt_sha256": text_hash(prompt),
                        "response_sha256": text_hash(row["content"]),
                        "finish_reason": row.get("finish"),
                        "completion_tokens": row.get("completion_tokens"),
                        "human_judgment": None})
        keys[number] = {"observation": identity(row), "historical": old.get(k),
                        "new": new.get(k)}
    return {"schema_version": 3, "sample_kind": "disagreement_diagnostic" if diagnostic else "corpus_random",
            "selection": "uniform without replacement from the stated population",
            "population_size": len(eligible), "n_sampled": len(entries), "seed": seed,
            "blinding": "machine judgments and run/arm identities are withheld in a separate key file",
            "entries": entries}, keys


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--old-verdicts", default="")
    parser.add_argument("--old-pass", default=None)
    parser.add_argument("--prompts", default=str(SCRIPT_DIR / "prompts_512.tsv"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8081"))
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""))
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--diagnostic-sample", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.limit, args.sample, args.diagnostic_sample) < 0:
        raise ValueError("limits and sample sizes must be non-negative")
    results = read_jsonl(args.results)
    inventory = generation_inventory(results)
    old_rows = read_jsonl(args.old_verdicts) if args.old_verdicts else []
    out_dir = Path(os.environ.get("REJUDGE_DIR", str(SCRIPT_DIR / f"rejudge_v3_{args.label}")))
    sources = {"results": artifact(args.results), "prompts": artifact(args.prompts)}
    if args.old_verdicts:
        sources["old_verdicts"] = artifact(args.old_verdicts)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "sources": sources, "output_dir": str(out_dir),
                          "n_result_rows": len(inventory),
                          "n_with_content": sum("content" in r and "error" not in r for r in results),
                          "sample": args.sample, "diagnostic_sample": args.diagnostic_sample}, indent=2))
        return
    os.environ["BASE_URL"] = args.base_url
    import judge
    prompts = judge.load_prompts(args.prompts)
    runtime_path = os.environ.get("JUDGE_RUNTIME_MANIFEST")
    runtime = fetch_runtime_identity(args.base_url, runtime_path)
    model = attested_model(runtime, args.judge_model)
    cfg = judge.scoring_config(model, runtime)
    cfg_hash = judge.config_hash(cfg)
    manifest = {"schema_version": 3, "label": args.label, "judge": cfg,
                "scoring_config_hash": cfg_hash, "sources": sources,
                "old_pass": args.old_pass,
                "provenance_note": "Source hashes measured for this scoring pass; unknown historical runtime identities remain unknown."}
    manifest_path = out_dir / "rejudge_manifest.json"
    if out_dir.exists():
        if not args.resume:
            raise ValueError("output directory exists; use --resume or a new label")
        if not manifest_path.exists():
            raise ValueError("existing output lacks its immutable manifest")
    else:
        out_dir.mkdir(parents=True)
    ensure_manifest(manifest_path, manifest)
    new_rows = judge.run_pass(results, prompts, model, cfg,
                              str(out_dir / "verdicts_v3.jsonl"), args.limit,
                              lambda: verify_unchanged(args.base_url, runtime_path, runtime))
    if old_rows:
        write_json(out_dir / "transitions.json", compute_transitions(
            old_rows, new_rows, results, args.old_pass, cfg_hash))
    for diagnostic, size, prefix in ((False, args.sample, "human_review"),
                                      (True, args.diagnostic_sample, "diagnostic_review")):
        if size:
            sample, key = build_blind_sample(old_rows, new_rows, results, prompts, size,
                                             args.sample_seed, diagnostic, args.old_pass, cfg_hash)
            write_json(out_dir / f"{prefix}_sample.json", sample)
            write_json(out_dir / f"{prefix}_key.json", key)


if __name__ == "__main__":
    main()
