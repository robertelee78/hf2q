#!/usr/bin/env python3
"""Versioned rejudging pass over retained full responses (historical results).

Repairs E8/E10 for EXISTING data without touching a single historical byte:
- Reads a historical results JSONL (full responses ARE retained there; the
  old rows already carry `finish` and `completion_tokens`, which are passed
  to the judge as generation metadata).
- Judges the COMPLETE response with the v2 judge (judge.py): no 2,500-char
  clipping, no synthetic truncation marker, finish reason + token usage
  supplied, cross-field-validated verdicts, preserved failed attempts with
  bounded diagnostics, explicit attempt identity, response + scoring-config
  hashes in every key (no stale reuse).
- Writes ONLY to a NEW versioned output directory
  (rejudge_v2_<label>/ by default): verdicts_v2.jsonl, a provenance manifest
  (hashes measured at rejudge time — today's measurement of the artifact,
  never an invented historical binding), an old-to-new label transition
  table, and (optionally) a blinded human-review sample with a separate key
  file. Historical results/verdict files are opened read-only.
- Old verdicts are never rewritten: the historical labels stay in place and
  the transition table makes every old->new relabeling explicit.
- --dry-run plans the pass (counts, joins, sample candidates) with NO
  network calls and writes NOTHING.

Usage:
  python3 rejudge.py --results full_results_w1.jsonl \
      --old-verdicts full_verdicts_w1.jsonl --label w1_deepseek \
      [--base-url http://127.0.0.1:8081] [--judge-model M] [--limit N] \
      [--resume] [--sample 40] [--sample-seed 7] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def old_label(row):
    if row is None:
        return "unjudged"
    if "judge_error" in row or row.get("response_state") is None:
        return "judge_error"
    return row["response_state"]


def new_label(row):
    return old_label(row)


def compute_transitions(old_rows, new_rows, results_keys):
    """old->new label transition counts over the results keys, plus explicit
    counts for keys missing on either side."""
    old_by = {(r["prompt_id"], r["rep"]): r for r in old_rows}
    new_by = {(r.get("prompt_id"), r.get("rep")): r for r in new_rows}
    transitions = collections.Counter()
    for key in sorted(results_keys):
        old = old_label(old_by.get(key))
        new = new_label(new_by.get(key))
        transitions[(old, new)] += 1
    return {
        "n_results": len(results_keys),
        "n_old_verdicts": len(old_by),
        "n_new_verdicts": sum(1 for k in results_keys if k in new_by),
        "transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
        "relabels": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())
                     if a != b},
    }


def build_blind_sample(old_rows, new_rows, results, prompts, n, seed):
    """Blinded human-review sample over jointly judged keys.

    Presentation is blinded: each entry shows two categorical judgments under
    randomly assigned pass labels (alpha/beta); which pass is which lives
    ONLY in the separate key file. Evidence quotes are excluded (the v1
    evidence can quote the artificial truncation marker and unblind the
    reviewer). Disagreements (label transitions) are sampled first.
    """
    old_by = {(r["prompt_id"], r["rep"]): r for r in old_rows}
    new_by = {(r.get("prompt_id"), r.get("rep")): r for r in new_rows
              if "judge_error" not in r and r.get("response_state")}
    res_by = {(r["prompt_id"], r["rep"]): r for r in results}
    both = [k for k in sorted(new_by) if k in old_by and k in res_by
            and "judge_error" not in old_by[k] and old_by[k].get("response_state")]
    disagreements = [k for k in both
                     if old_by[k]["response_state"] != new_by[k]["response_state"]]
    agreements = [k for k in both if k not in set(disagreements)]
    rng = random.Random(seed)
    rng.shuffle(disagreements)
    rng.shuffle(agreements)
    picked = (disagreements + agreements)[:n]
    entries, key = [], {}
    for i, k in enumerate(picked):
        old_v, new_v, res = old_by[k], new_by[k], res_by[k]
        flip = rng.random() < 0.5
        first, second = (old_v, new_v) if not flip else (new_v, old_v)
        names = ("historical_v1", "rejudge_v2")
        alpha_pass, beta_pass = names if not flip else names[::-1]
        cat = lambda v: {"response_state": v["response_state"],
                         "output_validity": v.get("output_validity"),
                         "substantive_compliance": v.get("substantive_compliance"),
                         "harmful_actionability": v.get("harmful_actionability")}
        prompt_text = prompts.get(k[0], "")
        entries.append({
            "sample_id": i + 1, "prompt_id": k[0], "rep": k[1],
            "prompt_excerpt": prompt_text[:300],
            "response_excerpt": (res.get("content") or "")[:1200],
            "finish_reason": res.get("finish"),
            "completion_tokens": res.get("completion_tokens"),
            "judgments": {"alpha": cat(first), "beta": cat(second)},
        })
        key[i + 1] = {"alpha": alpha_pass, "beta": beta_pass}
    return {"n_sampled": len(entries),
            "n_disagreements_available": len(disagreements),
            "n_agreements_available": len(agreements),
            "blinding": "per-entry random alpha/beta assignment; mapping only "
                        "in human_review_key.json; evidence quotes excluded",
            "entries": entries}, key


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help="historical results JSONL (full responses; read-only)")
    ap.add_argument("--old-verdicts", default="",
                    help="historical verdict JSONL for the transition table "
                         "(read-only; optional)")
    ap.add_argument("--prompts", default="",
                    help="prompt corpus TSV for the judged prompts (default: "
                         "prompts_512.tsv, then prompts.tsv, next to this script)")
    ap.add_argument("--label", required=True, help="versioned output label")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL",
                    "http://127.0.0.1:8081"))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""))
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--resume", action="store_true",
                    help="allow an existing output directory (failed attempts "
                         "are retried; successes are never duplicated)")
    ap.add_argument("--sample", type=int, default=0,
                    help="size of the blinded human-review sample")
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: no network, writes nothing")
    args = ap.parse_args()

    results_path = os.path.abspath(args.results)
    if not os.path.exists(results_path):
        sys.exit(f"results file not found: {results_path}")
    results = read_jsonl(results_path)
    rows = [r for r in results if "error" not in r and "content" in r]
    prompts = {}
    if args.prompts:
        prompts_path = os.path.abspath(args.prompts)
    else:
        prompts_path = os.path.join(SCRIPT_DIR, "prompts_512.tsv")
        if not os.path.exists(prompts_path):
            prompts_path = os.path.join(SCRIPT_DIR, "prompts.tsv")
    if os.path.exists(prompts_path):
        with open(prompts_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"):
                    continue
                pid, _, text = line.partition("\t")
                prompts[pid] = text

    out_dir = os.environ.get(
        "REJUDGE_DIR", os.path.join(SCRIPT_DIR, f"rejudge_v2_{args.label}"))
    verdicts_path = os.path.join(out_dir, "verdicts_v2.jsonl")
    old_rows = read_jsonl(os.path.abspath(args.old_verdicts)) if args.old_verdicts else []

    if args.dry_run:
        gen_errors = len(results) - len(rows)
        plan = {
            "dry_run": True, "label": args.label,
            "results_file": results_path,
            "results_sha256_measured_now": sha256_file(results_path),
            "n_result_rows": len(results), "n_with_content": len(rows),
            "n_generation_errors": gen_errors,
            "old_verdicts": (os.path.abspath(args.old_verdicts) if args.old_verdicts else None),
            "n_old_verdict_rows": len(old_rows),
            "output_dir_would_be": out_dir,
            "n_would_judge": len(rows),
            "sample": args.sample, "sample_seed": args.sample_seed,
            "note": "no network calls, nothing written; historical files "
                    "are read-only in every mode",
        }
        if old_rows:
            plan["transition_preview"] = compute_transitions(
                old_rows, [], [(r["prompt_id"], r["rep"]) for r in rows])["transitions"]
        print(json.dumps(plan, indent=2))
        return

    if os.path.exists(out_dir) and not args.resume:
        sys.exit(f"output directory exists: {out_dir} (use --resume to continue "
                 "or a new --label)")
    os.makedirs(out_dir, exist_ok=True)

    # configure + load the v2 judge (env must be set before import: judge.py
    # reads its configuration from the environment at import time)
    os.environ["BASE_URL"] = args.base_url
    if args.judge_model:
        os.environ["JUDGE_MODEL"] = args.judge_model
    sys.path.insert(0, SCRIPT_DIR)
    import judge as judge_mod  # noqa: E402  (import after env configuration)

    model = args.judge_model or judge_mod.resolve_model()
    cfg = judge_mod.scoring_config(model)
    cfg_hash = judge_mod.config_hash(cfg)

    manifest_path = os.path.join(out_dir, "rejudge_manifest.json")
    if not os.path.exists(manifest_path):
        manifest = {
            "label": args.label, "created": time.time(),
            "judge": cfg, "scoring_config_hash": cfg_hash,
            "sources": {
                "results": {"path": results_path,
                            "sha256_measured_at_rejudge_time": sha256_file(results_path),
                            "bytes": os.path.getsize(results_path)},
            },
            "provenance_note": "Hashes are of the artifact files as measured at "
                               "rejudge time. Historical logs do not bind each "
                               "response to a binary SHA or full sampler config; "
                               "this manifest invents no historical provenance.",
        }
        if args.old_verdicts:
            old_path = os.path.abspath(args.old_verdicts)
            manifest["sources"]["old_verdicts"] = {
                "path": old_path,
                "sha256_measured_at_rejudge_time": sha256_file(old_path),
                "bytes": os.path.getsize(old_path)}
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    existing = judge_mod.read_rows(verdicts_path) if os.path.exists(verdicts_path) else []
    attempts: dict[tuple, int] = {}
    succeeded: set[tuple] = set()
    for r in existing:
        key, attempt_no, ok = judge_mod.row_attempt_state(r)
        if key is None:
            continue
        attempts[key] = max(attempts.get(key, 0), attempt_no)
        if ok:
            succeeded.add(key)

    attempted = 0
    with open(verdicts_path, "a", encoding="utf-8") as out_fh:
        for row in rows:
            content = row["content"]
            response_hash = judge_mod.sha256_text(content)
            key = judge_mod.attempt_key(row.get("arm", ""), row["prompt_id"],
                                        row["rep"], model, cfg_hash, response_hash)
            if key in succeeded:
                continue
            if (not judge_mod.JUDGE_FORCE_RETRY
                    and attempts.get(key, 0) >= judge_mod.JUDGE_MAX_ATTEMPTS):
                print(f"  [{row['prompt_id']}#{row['rep']}] attempts exhausted "
                      f"({attempts[key]}); failure stays visible", file=sys.stderr)
                continue
            if attempted >= args.limit:
                break
            attempt_no = attempts.get(key, 0) + 1
            attempted += 1
            prompt = prompts.get(row["prompt_id"], "")
            base = {
                "ts": time.time(), "arm": row.get("arm", ""), 
                "prompt_id": row["prompt_id"], "rep": row["rep"],
                "judge_model": model,
                "judge_temperature": judge_mod.JUDGE_TEMPERATURE,
                "schema_version": judge_mod.SCHEMA_VERSION,
                "rubric_version": judge_mod.RUBRIC_VERSION,
                "attempt": attempt_no, "scoring_config_hash": cfg_hash,
                "response_hash": response_hash, "input_chars": len(content),
                "finish_reason": row.get("finish"),
                "completion_tokens": row.get("completion_tokens"),
            }
            try:
                verdict = judge_mod.judge_one(
                    model, prompt, content, row.get("finish"),
                    {"prompt_tokens": row.get("prompt_tokens"),
                     "completion_tokens": row.get("completion_tokens")})
                base["input_coverage"] = 1.0
                base.update(verdict)
            except judge_mod.JudgeError as exc:
                base["judge_error"] = str(exc)
                base["judge_error_type"] = exc.error_type
                base["judge_error_detail"] = exc.detail
            except Exception as exc:  # unexpected — still fail-closed
                base["judge_error"] = f"unexpected: {exc}"[:400]
                base["judge_error_type"] = "unexpected"
                base["judge_error_detail"] = repr(exc)[:400]
            out_fh.write(json.dumps(base) + "\n")
            out_fh.flush()
            state = base.get("response_state",
                             f"JUDGE_ERROR({base.get('judge_error_type', '?')})")
            print(f"  [{row['prompt_id']}#{row['rep']}] attempt {attempt_no}: "
                  f"{state}", file=sys.stderr)
    print(f"attempted {attempted} rejudgments -> {verdicts_path} "
          f"(scoring config {cfg_hash[:12]})", file=sys.stderr)

    # old -> new transition table + blinded review sample
    new_rows = judge_mod.read_rows(verdicts_path)
    if old_rows:
        trans = compute_transitions(old_rows, new_rows,
                                    [(r["prompt_id"], r["rep"]) for r in rows])
        with open(os.path.join(out_dir, "transitions.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(trans, fh, indent=2)
        print(f"\nold -> new label transitions ({trans['n_results']} results rows):")
        for t, n in sorted(trans["transitions"].items()):
            mark = "" if t.split("->")[0] == t.split("->")[1] else "  <- relabel"
            print(f"  {t:<48}{n:>6}{mark}")
    if args.sample > 0 and old_rows:
        sample, key = build_blind_sample(old_rows, new_rows, rows, prompts,
                                         args.sample, args.sample_seed)
        with open(os.path.join(out_dir, "human_review_sample.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(sample, fh, indent=2)
        with open(os.path.join(out_dir, "human_review_key.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(key, fh, indent=2)
        print(f"\nblinded review sample: {sample['n_sampled']} entries "
              f"({sample['n_disagreements_available']} disagreements available) "
              "-> human_review_sample.json (key in human_review_key.json)")


if __name__ == "__main__":
    main()
