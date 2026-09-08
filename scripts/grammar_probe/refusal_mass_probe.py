#!/usr/bin/env python3
"""Refusal-mass probe: Matt Suiche's Figure-1 measurement, on DeepSeek-V4.

Measures the front-loading of refusal directly at the logit level:
  probe A: top-10 logprobs at position 0 with NO grammar (unconstrained entry)
  probe B: top-10 logprobs at the first FREE position after the W1 anchor
           (grammar forces anchor, then we read the distribution at the
           moment the model's own choice resumes)

Claim under test (from the Tantalus replica): refusal is front-loaded —
~all entry mass sits on the refusal opener; once a compliant prefix is
committed, the model's own distribution shifts to the compliance manifold.

Method: hf2q chat completions with logprobs=true, top_logprobs=10,
max_tokens=1 (position 0) — and for the after-anchor distribution we use
the W1 grammar and read the first token the model picks *past the forced
prefix*. Since hf2q applies the mask before sampling, position-0 logprobs
under the grammar are post-mask (renormalized over the legal set); to get
the pre-anchor distribution we run the same prompt with NO grammar.

Output: JSONL with the top-10 token probabilities at both positions per prompt.
"""
import json
import os
import sys
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8081")
PROBE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(PROBE_DIR, "refusal_mass_probe.jsonl"))
PROMPTS = os.environ.get("PROMPTS", os.path.join(PROBE_DIR, "prompts_512.tsv"))
LIMIT = int(os.environ.get("LIMIT", "12"))  # a handful of reliably-refused prompts suffices


def chat_logprobs(model, prompt, grammar=None, max_tokens=1):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 10,
        "reasoning_effort": "low",
        "hf2q_enable_thinking": False,
    }
    if grammar is not None:
        body["grammar"] = grammar
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)


def extract_top10(resp):
    """Pull [(token, prob)] at the first sampled position from the response."""
    try:
        lp = resp["choices"][0]["logprobs"]["content"][0]
        return [(t["token"], round(2.718281828 ** t["logprob"], 6)) for t in lp["top_logprobs"]]
    except (KeyError, IndexError, TypeError) as e:
        return [("PARSE_ERROR", str(e))]

def extract_trace(resp):
    """Per-position [(token, prob)] for every sampled position."""
    try:
        content = resp["choices"][0]["logprobs"]["content"]
        return [((pos.get("top_logprobs") or [{}])[0].get("token", "?"),
                 round(2.718281828 ** pos.get("logprob", -99), 6)) for pos in content]
    except (KeyError, IndexError, TypeError) as e:
        return [("PARSE_ERROR", str(e))]


def main():
    with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as resp:
        model = json.load(resp)["data"][0]["id"]

    rows = []
    with open(PROMPTS) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            pid, _, text = line.partition("\t")
            if pid.startswith("h"):  # harmful prompts are the refusal-triggering ones
                rows.append((pid, text))
    rows = rows[:LIMIT]

    w1 = open(os.path.join(PROBE_DIR, "w1.gbnf")).read()

    with open(OUT, "a") as out:
        for pid, prompt in rows:
            rec = {"prompt_id": pid, "prompt": prompt[:120]}
            try:
                # A: unconstrained entry distribution
                rA = chat_logprobs(model, prompt, grammar=None, max_tokens=1)
                rec["entry_top10_no_grammar"] = extract_top10(rA)
            except Exception as e:
                rec["entry_error"] = str(e)[:200]
            try:
                # B: under the W1 grammar. The anchor is FORCED, so its
                # positions carry ~100% post-mask mass by construction; the
                # informative position is the first FREE token after the
                # forced anchor. Emit enough tokens to clear it, capture the
                # full per-position logprob trace.
                rB = chat_logprobs(model, prompt, grammar=w1, max_tokens=30)
                rec["under_grammar_trace"] = extract_trace(rB)
                rec["under_grammar_first_token"] = rB["choices"][0]["message"].get("content", "")[:40]
            except Exception as e:
                rec["grammar_error"] = str(e)[:200]
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"  [{pid}] entry={rec.get('entry_top10_no_grammar', [['?']])[0][0]!r} -> grammar={rec.get('under_grammar_first_token','?')!r}", file=sys.stderr)

    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
