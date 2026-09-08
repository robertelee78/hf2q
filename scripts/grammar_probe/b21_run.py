#!/usr/bin/env python3
"""B21 spike: per-prompt derived grammars (gcd answer-key learning).

Unlike spike_run.py (one grammar for all prompts), B21 derives each prompt's
grammar from the prompt itself: topic_term ::= <salient prompt terms>, so the
first sentence must lexically engage the actual task.
"""
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b21_gen import build_grammar

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8081")
ARM = os.environ.get("ARM", "B21_SPIKE")
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "spike_results_b21.jsonl"))
PROMPTS = os.environ.get("PROMPTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.tsv"))
LIMIT = int(os.environ.get("LIMIT", "100000"))


def main() -> None:
    with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as resp:
        model = json.load(resp)["data"][0]["id"]

    rows = []
    with open(PROMPTS) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            pid, _, text = line.partition("\t")
            rows.append((pid, text))
    rows = rows[:LIMIT]

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r["arm"], r["prompt_id"], r["rep"]))
                except (json.JSONDecodeError, KeyError):
                    continue

    print(f"arm={ARM} model={model} prompts={len(rows)} (per-prompt grammars)", file=sys.stderr)
    with open(OUT, "a") as out:
        for pid, prompt in rows:
            if (ARM, pid, 1) in done:
                continue
            grammar = build_grammar(prompt)
            body = json.dumps({
                "model": model, "messages": [{"role": "user", "content": prompt}],
                "grammar": grammar, "max_tokens": 800, "temperature": 0,
                "reasoning_effort": "low", "hf2q_enable_thinking": False,
            }).encode()
            start = time.time()
            req = urllib.request.Request(
                f"{BASE_URL}/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    r = json.load(resp)
                choice = r["choices"][0]
                row = {"ts": time.time(), "arm": ARM, "prompt_id": pid, "rep": 1,
                       "content": choice["message"].get("content") or "",
                       "finish": choice.get("finish_reason"),
                       "latency_s": int(time.time() - start)}
            except Exception as exc:
                row = {"ts": time.time(), "arm": ARM, "prompt_id": pid, "rep": 1,
                       "error": str(exc)[:300], "latency_s": int(time.time() - start)}
            out.write(json.dumps(row) + "\n"); out.flush()
            state = row.get("finish", "ERROR")
            print(f"  [{ARM}/{pid}] {state}", file=sys.stderr)


if __name__ == "__main__":
    main()
