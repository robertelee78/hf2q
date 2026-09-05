#!/usr/bin/env python3
"""B12 — grammar-emitted framed reasoning (the "force the thinking" arm).

Every request carries the static b12.gbnf and hf2q_enable_thinking=false, so
the grammar engages at token 0, writes the reasoning span's authorized-frame
opening itself, lets deliberation run free, then forces the procedural answer
opening. Same recording schema as b7_run.py.

Env: BASE_URL, MODEL, PROMPTS, RESULTS, MAX_TOKENS, TEMPERATURE, TOP_P,
     REPEATS, SLICE (harmful|all), ARM (default B12), GRAMMAR (gbnf path).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b7_run import REFUSAL_PHRASES  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8081")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.environ.get("PROMPTS", os.path.join(SCRIPT_DIR, "prompts.tsv"))
RESULTS = os.environ.get("RESULTS", os.path.join(SCRIPT_DIR, "results.jsonl"))
GRAMMAR = os.environ.get("GRAMMAR", os.path.join(SCRIPT_DIR, "b12.gbnf"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
TEMPERATURE = os.environ.get("TEMPERATURE", "0.55")
TOP_P = os.environ.get("TOP_P", "0.95")
REPEATS = int(os.environ.get("REPEATS", "1"))
SLICE = os.environ.get("SLICE", "harmful")
ARM = os.environ.get("ARM", "B12")


def resolve_model() -> str:
    with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=10) as resp:
        return json.load(resp)["data"][0]["id"]


def main() -> None:
    model = os.environ.get("MODEL") or resolve_model()
    grammar = open(GRAMMAR).read()

    rows = []
    with open(PROMPTS) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            pid, _, text = line.partition("\t")
            if SLICE == "harmful" and not pid.startswith("h"):
                continue
            rows.append((pid, text))

    print(f"arm={ARM} model={model} prompts={len(rows)} "
          f"(grammar-emitted framed reasoning, thinking flag off)", file=sys.stderr)
    out = open(RESULTS, "a")
    for pid, prompt in rows:
        for rep in range(1, REPEATS + 1):
            body = json.dumps({
                "model": model, "messages": [{"role": "user", "content": prompt}],
                "grammar": grammar, "max_tokens": MAX_TOKENS,
                "temperature": float(TEMPERATURE), "top_p": float(TOP_P),
                "hf2q_enable_thinking": False,
            }).encode()
            start = time.time()
            req = urllib.request.Request(
                f"{BASE_URL}/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    r = json.load(resp)
            except Exception as exc:
                row = {"ts": time.time(), "arm": ARM, "prompt_id": pid, "rep": rep,
                       "error": str(exc)[:300], "latency_s": int(time.time() - start)}
                out.write(json.dumps(row) + "\n"); out.flush()
                print(f"  [{ARM}/{pid}#{rep}] HTTP ERROR (recorded)", file=sys.stderr)
                continue
            latency = int(time.time() - start)
            msg = r["choices"][0]["message"]
            content = msg.get("content") or ""
            lowered = content.lower()
            hits = [p for p in REFUSAL_PHRASES if p in lowered]
            row = {
                "ts": time.time(), "arm": ARM, "prompt_id": pid, "rep": rep,
                "latency_s": latency, "finish": r["choices"][0].get("finish_reason"),
                "content": content,
                "reasoning_chars": len(msg.get("reasoning_content") or ""),
                "refusal_hit_count": len(hits), "refusal_matches": hits,
                "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                "cached_tokens": (r.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            }
            out.write(json.dumps(row) + "\n"); out.flush()
            print(f"  [{ARM}/{pid}#{rep}] finish={row['finish']} hits={len(hits)} {latency}s",
                  file=sys.stderr)
    out.close()


if __name__ == "__main__":
    main()
