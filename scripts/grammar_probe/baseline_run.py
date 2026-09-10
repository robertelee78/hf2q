#!/usr/bin/env python3
"""Matched BASE/W1 generation with verified runtime identity.

Requires RUNTIME_MANIFEST from the managed launcher and a live matching
/hf2q/v1/runtime snapshot. BASE requires inactive GLP, default grammar and
DWQ overlays. Literal/grammar canaries are additional plumbing checks; their
response shape alone does not establish runtime configuration.

Both arms use the recorded runtime, prompt corpus, sampling settings and
budget ladder. W1 is the exact historical artifact, not the embedded default.
Results bind (run, config, arm, prompt, rep, budget) and prompt/content hashes.
A changed attestation or configuration prevents resumption. No model process
is started or stopped by this script. DRY_RUN=1 performs no network requests.

Env: BASE_URL, MODEL (optional; must equal attested resident model),
RUNTIME_MANIFEST, PROMPTS, OUT, BUDGET_LADDER (800), TEMPERATURE (0), TOP_P,
REASONING_EFFORT (low), ENABLE_THINKING (0), REPEATS (1), W1_GRAMMAR, LIMIT,
DRY_RUN. SERVER_IDENTITY is retained as an operator note, never verification.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
import uuid

from measurement import text_hash
from runtime_identity import attested_model, fetch_runtime_identity, require_unconstrained, verify_unchanged

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8081")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
W1_GRAMMAR = os.environ.get("W1_GRAMMAR", os.path.join(SCRIPT_DIR, "w1.gbnf"))
PROMPTS = os.environ.get("PROMPTS", os.path.join(SCRIPT_DIR, "prompts_512.tsv"))
OUT = os.environ.get("OUT", os.path.join(SCRIPT_DIR, "paired_results.jsonl"))
DRY_RUN = os.environ.get("DRY_RUN", "0") not in ("0", "", "false")
LIMIT = int(os.environ.get("LIMIT", "100000"))
REPEATS = int(os.environ.get("REPEATS", "1"))
ARMS = ["BASE", "W1"]
W1_ANCHOR = "Here is the technical breakdown."
TEMPLATE = "user-only-no-system"

UNCONSTRAINED_CANARY_PROMPT = ("Reply with exactly this word and nothing else: "
                               "BANANA")
CANARY_ECHO_GRAMMAR = os.path.join(SCRIPT_DIR, "canary_echo.gbnf")
CANARY_ECHO_LITERAL = "CANARY-7f3d9."


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def parse_budgets():
    ladder = os.environ.get("BUDGET_LADDER", "800").strip()
    budgets = [int(b) for b in ladder.split(",") if b.strip()]
    if not budgets or any(b <= 0 for b in budgets) or len(set(budgets)) != len(budgets):
        sys.exit("BUDGET_LADDER must be a comma list of positive integers")
    return budgets


def check_loopback():
    import re
    if not re.match(r"^http://(127\.0\.0\.1|localhost):[0-9]+$", BASE_URL):
        sys.exit(f"BASE_URL must be a loopback endpoint without /v1 (got {BASE_URL})")


def resolve_model():
    with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as resp:
        return json.load(resp)["data"][0]["id"]


def load_prompts():
    rows = []
    with open(PROMPTS, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            pid, _, text = line.partition("\t")
            rows.append((pid, text))
    if len({pid for pid, _ in rows}) != len(rows):
        raise ValueError("duplicate prompt IDs in corpus")
    return rows


def build_config(model, budgets, w1_text, runtime=None):
    sampling = {
        "temperature": float(os.environ.get("TEMPERATURE", "0")),
        "reasoning_effort": os.environ.get("REASONING_EFFORT", "low"),
        "hf2q_enable_thinking": os.environ.get("ENABLE_THINKING", "0")
        not in ("0", "", "false"),
    }
    top_p = os.environ.get("TOP_P", "")
    if top_p != "":
        sampling["top_p"] = float(top_p)
    return {
        "runner": "baseline_run.py", "runner_version": 2,
        "runtime_identity": runtime,
        "arms": {
            "BASE": {"grammar": None,
                     "constraint_mode": "no_gcd_server",
                     "constraint_rationale": (
                         "server started without --gcd/--gcd-schema/--glp; "
                         "no request grammar or response_format; injection "
                         "impossible per src/serve/api/handlers.rs ADR-053/057; "
                         "verified from the managed runtime attestation; canaries check plumbing")},
            "W1": {"grammar": os.path.abspath(W1_GRAMMAR),
                   "grammar_sha256": hashlib.sha256(w1_text.encode()).hexdigest(),
                   "artifact": "historical W1 artifact (scripts/grammar_probe/w1.gbnf)"},
        },
        "model": model, "server_identity": os.environ.get("SERVER_IDENTITY", ""),
        "prompt_corpus": os.path.abspath(PROMPTS),
        "prompt_corpus_sha256": sha256_file(PROMPTS),
        "template": TEMPLATE, "budgets": budgets, "repeats": REPEATS,
        "sampling": sampling,
        "arm_order": "balanced_alternation_by_prompt_rep_budget",
        "canaries": ["unconstrained_literal", "grammar_path_echo", "w1_anchor"],
    }


def post_completion(model, messages, max_tokens, sampling, grammar_text=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": sampling["temperature"],
            "reasoning_effort": sampling["reasoning_effort"],
            "hf2q_enable_thinking": sampling["hf2q_enable_thinking"]}
    if "top_p" in sampling:
        body["top_p"] = sampling["top_p"]
    if grammar_text is not None:
        body["grammar"] = grammar_text
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.load(resp)


def run_canaries(model, sampling):
    """Three runtime proofs; any failure aborts BEFORE measurement."""
    results = {}
    # 1. unconstrained: literal must appear; the W1 anchor must not
    r = post_completion(model, [{"role": "user",
                                "content": UNCONSTRAINED_CANARY_PROMPT}],
                        64, sampling)
    content = (r["choices"][0]["message"].get("content") or "").strip()
    ok = ("banana" in content.lower()
          and not content.startswith(W1_ANCHOR)
          and "technical breakdown" not in content[:80].lower())
    results["unconstrained_literal"] = {
        "pass": bool(ok), "observed_prefix": content[:80],
        "expected": "the literal BANANA, without the W1 anchor"}
    if not ok:
        sys.exit("CANARY FAIL (unconstrained): the server appears to inject a "
                 "default grammar into no-grammar requests. Restart it WITHOUT "
                 "--gcd/--gcd-schema/--glp; observed: " + content[:120])
    # 2. grammar path live (mirror of probe.sh canary)
    gtext = open(CANARY_ECHO_GRAMMAR).read()
    r = post_completion(model, [{"role": "user", "content": "Describe a tree."}],
                        900, sampling, grammar_text=gtext)
    content = (r["choices"][0]["message"].get("content") or "")
    ok = content.startswith(CANARY_ECHO_LITERAL)
    results["grammar_path_echo"] = {"pass": bool(ok),
                                    "observed_prefix": content[:20]}
    if not ok:
        sys.exit("CANARY FAIL (grammar path): forced literal missing — the "
                 "grammar path is not live; do not measure.")
    # 3. the historical W1 artifact itself engages
    w1_text = open(W1_GRAMMAR).read()
    r = post_completion(model, [{"role": "user", "content": "Describe a tree."}],
                        200, sampling, grammar_text=w1_text)
    content = (r["choices"][0]["message"].get("content") or "")
    ok = content.startswith(W1_ANCHOR)
    results["w1_anchor"] = {"pass": bool(ok), "observed_prefix": content[:48]}
    if not ok:
        sys.exit("CANARY FAIL (W1 artifact): the historical w1.gbnf did not "
                 "force its anchor; do not measure.")
    return results


def check_resume_compat(cfg_hash):
    if not os.path.exists(OUT):
        return set()
    done, foreign, mismatched = set(), 0, 0
    with open(OUT, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_hash = r.get("config_sha256")
            if not row_hash:
                foreign += 1
            elif row_hash != cfg_hash:
                mismatched += 1
            else:
                done.add((r["arm"], r["prompt_id"], r["rep"], r.get("budget")))
    if foreign:
        sys.exit(f"refusing to append to {OUT}: {foreign} unbound rows "
                 "(historical/unbound file is preserved; use a new OUT)")
    if mismatched:
        sys.exit(f"refusing to append to {OUT}: {mismatched} rows bound to a "
                 "different run config; use a new OUT for the new config")
    return done


def main():
    check_loopback()
    budgets = parse_budgets()
    if REPEATS <= 0:
        raise ValueError("REPEATS must be positive")
    for path in (W1_GRAMMAR, CANARY_ECHO_GRAMMAR, PROMPTS):
        if not os.path.exists(path):
            sys.exit(f"required file not found: {path}")
    w1_text = open(W1_GRAMMAR).read()
    if hashlib.sha256(w1_text.encode()).hexdigest() != "db29d79fc43f6184c2b37c2e6c91cda609dfb4bd49667922f06c48095e3bee3e":
        raise ValueError("W1_GRAMMAR must match the historical W1 artifact")
    if DRY_RUN:
        model = os.environ.get("MODEL") or "(unresolved: dry-run)"
    else:
        model = None
    prompts = load_prompts()
    runtime_path = os.environ.get("RUNTIME_MANIFEST")
    runtime = None if DRY_RUN else fetch_runtime_identity(BASE_URL, runtime_path)
    if runtime is not None:
        model = attested_model(runtime, os.environ.get("MODEL"))
    if runtime is not None:
        require_unconstrained(runtime)
    cfg = build_config(model, budgets, w1_text, runtime)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    done = check_resume_compat(cfg_hash)
    manifest_path = OUT + ".manifest.json"
    if os.path.exists(OUT) and not os.path.exists(manifest_path):
        raise ValueError("existing generation output lacks its manifest")

    plan = {
        "dry_run": True, "out": OUT, "manifest": manifest_path,
        "config": cfg, "config_sha256": cfg_hash,
        "n_prompts": len(prompts), "repeats": REPEATS, "budgets": budgets,
        "n_pairs_per_arm": len(prompts) * REPEATS * len(budgets),
        "would_request": 2 * len(prompts) * REPEATS * len(budgets) - len(done),
        "canaries_planned": cfg["canaries"],
        "baseline_unconstrained_by": cfg["arms"]["BASE"]["constraint_rationale"],
    }
    if DRY_RUN:
        print(json.dumps(plan, indent=2))
        return

    verify_unchanged(BASE_URL, runtime_path, runtime)
    canary_results = run_canaries(model, cfg["sampling"])
    verify_unchanged(BASE_URL, runtime_path, runtime)
    if os.path.exists(manifest_path):
        prior_manifest = json.load(open(manifest_path, encoding="utf-8"))
        if prior_manifest.get("config") != cfg or prior_manifest.get("config_sha256") != cfg_hash:
            raise ValueError("generation manifest differs from requested configuration")
    run_id = prior_manifest.get("generation_run_id") if os.path.exists(manifest_path) else uuid.uuid4().hex
    if not run_id:
        raise ValueError("existing generation manifest lacks its run identity; use a new output")
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump({"config": cfg, "config_sha256": cfg_hash, "generation_run_id": run_id,
                       "created": time.time(), "out": OUT,
                       "canary_results": canary_results}, fh, indent=2)

    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as stream:
            if any(json.loads(line).get("generation_run_id") != run_id for line in stream if line.strip()):
                raise ValueError("generation rows disagree with manifest run identity")
    out = open(OUT, "a", encoding="utf-8")
    requested = 0
    for prompt_index, (pid, prompt) in enumerate(prompts):
        for rep in range(1, REPEATS + 1):
            for budget_index, budget in enumerate(budgets):
                order = ARMS if (prompt_index + rep - 1 + budget_index) % 2 == 0 else ARMS[::-1]
                for arm in order:
                    if (arm, pid, rep, budget) in done or requested >= LIMIT:
                        continue
                    verify_unchanged(BASE_URL, runtime_path, runtime)
                    requested += 1
                    grammar_text = w1_text if arm == "W1" else None
                    start = time.time()
                    try:
                        r = post_completion(
                            model, [{"role": "user", "content": prompt}],
                            budget, cfg["sampling"], grammar_text=grammar_text)
                    except Exception as exc:
                        out.write(json.dumps({
                            "ts": time.time(), "arm": arm, "prompt_id": pid,
                            "rep": rep, "budget": budget,
                            "config_sha256": cfg_hash, "generation_run_id": run_id,
                            "prompt_sha256": text_hash(prompt),
                            "error": str(exc)[:300],
                            "latency_s": int(time.time() - start)}) + "\n")
                        out.flush()
                        print(f"  [{arm}/{pid}#{rep}@{budget}] HTTP ERROR "
                              "(recorded)", file=sys.stderr)
                        continue
                    verify_unchanged(BASE_URL, runtime_path, runtime)
                    msg = r["choices"][0]["message"]
                    content = msg.get("content") or ""
                    out.write(json.dumps({
                        "ts": time.time(), "arm": arm, "prompt_id": pid,
                        "rep": rep, "budget": budget,
                        "config_sha256": cfg_hash, "generation_run_id": run_id,
                        "prompt_sha256": text_hash(prompt),
                        "latency_s": int(time.time() - start),
                        "finish": r["choices"][0].get("finish_reason"),
                        "content": content,
                        "response_sha256": text_hash(content),
                        "reasoning_chars": len(msg.get("reasoning_content") or ""),
                        "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                        "completion_tokens": r.get("usage", {}).get(
                            "completion_tokens")}) + "\n")
                    out.flush()
                    print(f"  [{arm}/{pid}#{rep}@{budget}] "
                          f"finish={r['choices'][0].get('finish_reason')}",
                          file=sys.stderr)
    out.close()
    print(f"requested {requested} generations under config {cfg_hash[:12]} "
          f"-> {OUT} (report with: CONTROL=BASE python3 report.py "
          f"VERDICTS=... RESULTS={OUT})", file=sys.stderr)


if __name__ == "__main__":
    main()
