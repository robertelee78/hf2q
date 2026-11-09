#!/usr/bin/env python3
"""Reduce strict ADR-065 E5 DeepSeek joint-step expert-union records."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PREFIX = "[ADR065_EXPERT_UNION_JSON] "
SCHEMA = "hf2q.adr065.deepseek-expert-union.v1"
RECORD_KEYS = {
    "schema", "width", "top_k", "layers", "routed_bytes_naive",
    "routed_bytes_unique", "reduction_share",
}
LAYER_KEYS = {
    "layer", "routing", "unique_experts", "routed_expert_rows", "expert_bytes",
    "routed_bytes_naive", "routed_bytes_unique", "reduction_share",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reduction(naive, unique):
    return 0.0 if naive == 0 else (naive - unique) / naive


def parse_record(value, source, line_number):
    where = f"{source}:{line_number}"
    require(isinstance(value, dict), f"{where}: record must be an object")
    require(set(value) == RECORD_KEYS, f"{where}: record keys differ: {sorted(set(value) ^ RECORD_KEYS)}")
    require(value["schema"] == SCHEMA, f"{where}: unexpected schema")
    width, top_k = value["width"], value["top_k"]
    require(type(width) is int and 2 <= width <= 4, f"{where}: width must be 2..4")
    require(type(top_k) is int and top_k > 0, f"{where}: top_k must be positive")
    require(isinstance(value["layers"], list) and value["layers"], f"{where}: layers must be non-empty")
    naive_total = unique_total = 0
    seen = set()
    for layer in value["layers"]:
        require(isinstance(layer, dict) and set(layer) == LAYER_KEYS, f"{where}: layer keys differ")
        index = layer["layer"]
        require(type(index) is int and index >= 0 and index not in seen, f"{where}: invalid/duplicate layer")
        seen.add(index)
        require(layer["routing"] in {"hash", "score"}, f"{where}: invalid routing class")
        unique = layer["unique_experts"]
        routed_rows = layer["routed_expert_rows"]
        expert_bytes = layer["expert_bytes"]
        require(type(unique) is int and 1 <= unique <= width * top_k, f"{where}: invalid unique count")
        require(routed_rows == width * top_k, f"{where}: routed row count mismatch")
        require(type(expert_bytes) is int and expert_bytes > 0, f"{where}: expert bytes must be positive")
        naive = expert_bytes * routed_rows
        unique_bytes = expert_bytes * unique
        require(layer["routed_bytes_naive"] == naive, f"{where}: layer naive bytes mismatch")
        require(layer["routed_bytes_unique"] == unique_bytes, f"{where}: layer unique bytes mismatch")
        require(abs(layer["reduction_share"] - reduction(naive, unique_bytes)) < 1e-12,
                f"{where}: layer reduction mismatch")
        naive_total += naive
        unique_total += unique_bytes
    require(seen == set(range(len(seen))), f"{where}: layers must be contiguous from zero")
    require(value["routed_bytes_naive"] == naive_total, f"{where}: total naive bytes mismatch")
    require(value["routed_bytes_unique"] == unique_total, f"{where}: total unique bytes mismatch")
    require(abs(value["reduction_share"] - reduction(naive_total, unique_total)) < 1e-12,
            f"{where}: total reduction mismatch")
    return value


def load(paths):
    records = []
    for path in paths:
        text = Path(path).read_text(errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            if PREFIX not in line:
                continue
            raw = line.split(PREFIX, 1)[1]
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            records.append(parse_record(value, path, line_number))
    require(records, "no ADR065 expert-union records")
    return records


def summarize(records, required_widths=()):
    widths = Counter(record["width"] for record in records)
    missing = sorted(set(required_widths) - set(widths))
    require(not missing, f"missing required widths: {missing}")
    route_bytes = defaultdict(lambda: [0, 0])
    layer_hist = defaultdict(Counter)
    for record in records:
        width = record["width"]
        for layer in record["layers"]:
            route = layer["routing"]
            route_bytes[route][0] += layer["routed_bytes_naive"]
            route_bytes[route][1] += layer["routed_bytes_unique"]
            layer_hist[(width, layer["layer"], route)][layer["unique_experts"]] += 1
    routes = {
        route: {
            "routed_bytes_naive": values[0],
            "routed_bytes_unique": values[1],
            "reduction_share": reduction(*values),
        }
        for route, values in sorted(route_bytes.items())
    }
    histograms = [
        {
            "width": width,
            "layer": layer,
            "routing": route,
            "unique_experts_histogram": {str(k): v for k, v in sorted(hist.items())},
        }
        for (width, layer, route), hist in sorted(layer_hist.items())
    ]
    naive = sum(values[0] for values in route_bytes.values())
    unique = sum(values[1] for values in route_bytes.values())
    return {
        "schema": "hf2q.adr065.deepseek-expert-union-report.v1",
        "records": len(records),
        "steps_by_width": {str(k): v for k, v in sorted(widths.items())},
        "routes": routes,
        "routed_bytes_naive": naive,
        "routed_bytes_unique": unique,
        "reduction_share": reduction(naive, unique),
        "layer_histograms": histograms,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--require-width", type=int, action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        require(all(2 <= width <= 4 for width in args.require_width), "required widths must be 2..4")
        report = summarize(load(args.logs), args.require_width)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
