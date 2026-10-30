#!/usr/bin/env python3
"""Summarize ADR-063 C5 `frontend timing` lines from hf2q server logs.

Reads one or more `server.stderr` files written with `HF2Q_FRONTEND_TIMING=1`
and decides the C5 gate: frontend work (render, tokenize, stable-prefix
render/tokenize/match, probe render, budget markers, constraint compile and
any overflow rewrite) against warm semantic TTFT (request receipt to the
first semantic stream event). Admission (pool lease), vision and worker
enqueue are reported but are not frontend work. Warm = streamed request with
more than `--cold-messages` messages (the four-agent client opens every
agent with system + user). No launch, no retries, no model access.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

SCHEMA = "hf2q.c5.frontend-summary.v1"
GATE_SHARE = 0.10
FRONTEND_SPANS = ("constraint_us", "render_us", "tokenize_us", "overflow_rewrite_us",
                  "stable_render_us", "stable_tokenize_us", "prefix_match_us",
                  "probe_render_us", "budget_us")
REPORTED = FRONTEND_SPANS + ("admission_us", "vision_us", "enqueue_us", "prepare_us")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")
MARKER = "frontend timing"


def parse_line(line):
    """Fields of one `frontend timing` line, or None for any other line."""
    text = ANSI.sub("", line)
    if MARKER not in text:
        return None
    if text.lstrip().startswith("{"):
        record = json.loads(text)
        fields = dict(record.get("fields", {}))
        if fields.pop("message", None) != MARKER:
            return None
        return fields
    fields = {}
    for key, raw in FIELD.findall(text.split(MARKER, 1)[1]):
        value = raw.strip('"')
        if value in ("true", "false"):
            fields[key] = value == "true"
        elif re.fullmatch(r"-?\d+", value):
            fields[key] = int(value)
        else:
            fields[key] = value
    return fields


def frontend_us(record):
    return sum(int(record.get(name, 0)) for name in FRONTEND_SPANS)


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return {"n": len(ordered), "min": ordered[0], "p50": statistics.median(ordered),
            "p90": p90, "max": ordered[-1]}


def summarize(records, cold_messages=2):
    families = {}
    for record in records:
        families.setdefault(record.get("family", "unknown"), []).append(record)
    out = {"schema": SCHEMA, "gate_share": GATE_SHARE, "families": {}}
    for family, rows in sorted(families.items()):
        warm = [r for r in rows if r.get("stream") and "first_output_us" in r
                and int(r.get("messages", 0)) > cold_messages]
        shares = [frontend_us(r) / r["first_output_us"] for r in warm if r["first_output_us"] > 0]
        share = quantiles(shares)
        out["families"][family] = {
            "requests": len(rows),
            "warm_streamed": len(warm),
            "frontend_us_all": quantiles([frontend_us(r) for r in rows]),
            "frontend_us_warm": quantiles([frontend_us(r) for r in warm]),
            "first_output_us_warm": quantiles([r["first_output_us"] for r in warm]),
            "frontend_share_of_warm_ttft": share,
            "prompt_tokens_warm": quantiles([int(r.get("prompt_tokens", 0)) for r in warm]),
            "spans_us_warm": {name: quantiles([int(r.get(name, 0)) for r in warm]) for name in REPORTED},
            "verdict": ("unmeasured" if share is None
                        else "material" if share["p50"] > GATE_SHARE else "not_material"),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path, help="server.stderr files")
    parser.add_argument("--cold-messages", type=int, default=2)
    args = parser.parse_args(argv)
    records = []
    for path in args.logs:
        with path.open(encoding="utf-8", errors="replace") as handle:
            records.extend(r for r in map(parse_line, handle) if r is not None)
    if not records:
        print(f"no '{MARKER}' lines found; was HF2Q_FRONTEND_TIMING=1 set on the server?",
              file=sys.stderr)
        return 2
    json.dump(summarize(records, args.cold_messages), sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
