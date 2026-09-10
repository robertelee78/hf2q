#!/usr/bin/env python3
"""Matched unconstrained-baseline vs W1 paired experiment runner (E7 repair).

Design (fresh paired generation; run ONLY after coordination with the
runtime owner — this file implements and dry-runs the design, it does not
launch workloads by itself):

PAIRED ARMS, FIXED IDENTITIES
  Arm "BASE" — unconstrained. Arm "W1" — the EXACT historical artifact
  scripts/grammar_probe/w1.gbnf (byte-identical file; sha256 recorded and
  checked into the manifest). Both arms share: model identity, server
  identity, prompt corpus (same TSV), template (user-only, no system
  prompt), sampling (greedy by default: temperature 0, no top_p unless
  given, reasoning_effort low, thinking disabled — the historical W1 cell),
  the same budget ladder, and the same reps. Pairing key:
  (prompt_id, rep, budget).

WHY "NO GRAMMAR IN THE REQUEST" IS NOT ENOUGH FOR THE BASELINE
  A server started with --gcd injects its embedded grammar into requests
  that carry no grammar and no response_format (src/serve/api/handlers.rs,
  ADR-053: `state.config.gcd && request.grammar.is_none() &&
  request.response_format.is_none()`; ADR-057 --gcd-schema does the same for
  its startup schema). Omitting the request grammar on such a server yields
  a CONSTRAINED baseline (the embedded default), not an unconstrained one.
  There is no request-level "empty grammar" opt-out: any explicit constraint
  surface (grammar or response_format) both (a) suppresses the default
  injection and (b) makes the arm itself constrained — a third condition,
  not an unconstrained baseline. Therefore:

  THE RUNNER'S APPROACH: target a server started WITHOUT --gcd and WITHOUT
  --gcd-schema (and without --glp), and prove unconstrained-ness at runtime
  with canaries before any measurement:
    1. unconstrained canary — a no-grammar request for an exact literal
       ("BANANA") must be produced verbatim and must NOT begin with the W1
       anchor ("Here is the technical breakdown."), which any injected
       anchor-forcing grammar would force. Failure aborts the run.
    2. grammar-path canary — canary_echo.gbnf (forced literal
       "CANARY-7f3d9.") proves the grammar path is live (mirrors probe.sh).
    3. W1-artifact canary — a trivial prompt under w1.gbnf must begin with
       the W1 anchor, proving the historical artifact itself engages.
  Canary results are recorded in the manifest; a failed canary aborts before
  any measurement row is written.

BUDGET LADDER (E7): BUDGET_LADDER (default "800", the historical cell) is
  predeclared in the manifest; every (prompt, rep) is generated under every
  budget in BOTH arms; termination (finish=length) is recorded per row and
  reported separately from substance and degeneration (report.py).

IDENTITY BINDING / RESUME: the full config (model, server identity, corpus
  + sha, template, grammar artifact + sha for W1 / explicit null for BASE,
  sampling, budgets, reps, canary protocol) is hashed into config_sha256,
  recorded on every row, and written to <OUT>.manifest.json. Resume is
  rejected unless existing rows carry the identical config_sha256;
  unbound/historical files are refused outright.

Env: BASE_URL (loopback only), MODEL, SERVER_IDENTITY, PROMPTS (default
  prompts_512.tsv), OUT, BUDGET_LADDER, TEMPERATURE (0), TOP_P (unset =
  not sent), REASONING_EFFORT (low), ENABLE_THINKING (0), REPEATS (1),
  W1_GRAMMAR (default w1.gbnf next to this script), LIMIT, DRY_RUN=1.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

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
    if not budgets or any(b <= 0 for b in budgets):
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
    return rows


def build_config(model, budgets, w1_text):
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
        "runner": "baseline_run.py", "runner_version": 1,
        "arms": {
            "BASE": {"grammar": None,
                     "constraint_mode": "no_gcd_server",
                     "constraint_rationale": (
                         "server started without --gcd/--gcd-schema/--glp; "
                         "no request grammar or response_format; injection "
                         "impossible per src/serve/api/handlers.rs ADR-053/057; "
                         "verified by the unconstrained canary")},
            "W1": {"grammar": os.path.abspath(W1_GRAMMAR),
                   "grammar_sha256": hashlib.sha256(w1_text.encode()).hexdigest(),
                   "artifact": "historical W1 artifact (scripts/grammar_probe/w1.gbnf)"},
        },
        "model": model, "server_identity": os.environ.get("SERVER_IDENTITY", ""),
        "prompt_corpus": os.path.abspath(PROMPTS),
        "prompt_corpus_sha256": sha256_file(PROMPTS),
        "template": TEMPLATE, "budgets": budgets, "repeats": REPEATS,
        "sampling": sampling,
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
    for path in (W1_GRAMMAR, CANARY_ECHO_GRAMMAR, PROMPTS):
        if not os.path.exists(path):
            sys.exit(f"required file not found: {path}")
    w1_text = open(W1_GRAMMAR).read()
    if DRY_RUN:
        model = os.environ.get("MODEL") or "(unresolved: dry-run)"
    else:
        model = os.environ.get("MODEL") or resolve_model()
    prompts = load_prompts()
    cfg = build_config(model, budgets, w1_text)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    done = check_resume_compat(cfg_hash)
    manifest_path = OUT + ".manifest.json"

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

    canary_results = run_canaries(model, cfg["sampling"])
    if not os.path.exists(manifest_path):
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump({"config": cfg, "config_sha256": cfg_hash,
                       "created": time.time(), "out": OUT,
                       "canary_results": canary_results}, fh, indent=2)

    out = open(OUT, "a", encoding="utf-8")
    requested = 0
    for pid, prompt in prompts:
        for rep in range(1, REPEATS + 1):
            for budget in budgets:
                for arm in ARMS:
                    if (arm, pid, rep, budget) in done or requested >= LIMIT:
                        continue
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
                            "config_sha256": cfg_hash,
                            "error": str(exc)[:300],
                            "latency_s": int(time.time() - start)}) + "\n")
                        out.flush()
                        print(f"  [{arm}/{pid}#{rep}@{budget}] HTTP ERROR "
                              "(recorded)", file=sys.stderr)
                        continue
                    msg = r["choices"][0]["message"]
                    content = msg.get("content") or ""
                    out.write(json.dumps({
                        "ts": time.time(), "arm": arm, "prompt_id": pid,
                        "rep": rep, "budget": budget,
                        "config_sha256": cfg_hash,
                        "latency_s": int(time.time() - start),
                        "finish": r["choices"][0].get("finish_reason"),
                        "content": content,
                        "response_sha256": hashlib.sha256(
                            content.encode()).hexdigest(),
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
