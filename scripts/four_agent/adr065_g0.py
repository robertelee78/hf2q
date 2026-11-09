#!/usr/bin/env python3
"""Compute ADR-065 E8's pre-registered G0 projection from retained receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 650_511
REQUIRED_CONTEXTS = (2_048, 16_384, 65_536)
REQUIRED_SHAPES = (
    "4x1",
    "4x2",
    "4x3",
    "4x4",
    "4+4+1+1",
    "4+1+1+1",
    "1x4",
    "2x4",
    "3x4",
)
ACTIVE_WIDTH_SHAPE = {1: "1x4", 2: "2x4", 3: "3x4", 4: "4x4"}
STAGES = (
    "prepare_ns",
    "draft_ns",
    "verify_readback_ns",
    "selection_ns",
    "capture_recovery_ns",
    "settle_catch_up_ns",
    "total_ns",
)
THRESHOLDS = {"s": 1.15, "S": 1.05, "S_ci95_low": 1.0}


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def interval(values: list[float]) -> list[float]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def median_ns(samples: list[dict[str, Any]], field: str) -> float:
    return statistics.median(float(sample[field]) for sample in samples)


def validate_e2(receipt: dict[str, Any]) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    if receipt.get("schema") != "hf2q.adr065.joint-verify-timing.v1":
        raise ValueError("E2 receipt schema mismatch")
    if receipt.get("grid") != "full":
        raise ValueError("E2 receipt is not the full grid")
    if receipt.get("warmups_per_configuration") != 1 or receipt.get("timed_runs_per_configuration") != 7:
        raise ValueError("E2 receipt must contain one warm-up and seven timed runs")
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    warmups: list[dict[str, Any]] = []
    for sample in receipt.get("samples", []):
        context = int(sample["context_tokens"])
        shape = str(sample["shape"])
        if context not in REQUIRED_CONTEXTS or shape not in REQUIRED_SHAPES:
            raise ValueError(f"unexpected E2 configuration: {context}/{shape}")
        if sample["warmup"]:
            warmups.append(sample)
            continue
        grouped.setdefault((context, shape), []).append(sample)
    expected = {(context, shape) for context in REQUIRED_CONTEXTS for shape in REQUIRED_SHAPES}
    if set(grouped) != expected:
        raise ValueError("E2 full-grid configurations are incomplete")
    for key, samples in grouped.items():
        repetitions = sorted(int(sample["repetition"]) for sample in samples)
        if repetitions != list(range(1, 8)):
            raise ValueError(f"E2 repetitions are incomplete for {key}: {repetitions}")
        if any(int(sample[field]) < 0 for sample in samples for field in STAGES):
            raise ValueError(f"E2 timing is negative for {key}")
    if len(warmups) != len(expected):
        raise ValueError("E2 must retain exactly one warm-up per configuration")
    return grouped, warmups


def b1_rates(campaign: dict[str, Any]) -> list[float]:
    stages = [stage for stage in campaign.get("stages", []) if stage.get("workload") == "n4"]
    if len(stages) != 1 or stages[0].get("status") != "Complete":
        raise ValueError("B1 must contain one complete n4 stage")
    rates = []
    for block in stages[0].get("blocks", []):
        entries = block.get("entries", [])
        if len(entries) != 1 or entries[0].get("outcome") != "Pass":
            raise ValueError("B1 n4 block is not a single passing baseline run")
        metrics = entries[0]["metrics"]
        completion = float(metrics["tokens"]["completion"])
        makespan_ms = float(metrics["makespan_s"]) * 1_000.0
        rates.append(completion / makespan_ms)
    if len(rates) != 20:
        raise ValueError(f"B1 must supply 20 n4 runs, got {len(rates)}")
    return rates


def expected_committed_tokens(e3: dict[str, Any]) -> float:
    histogram = e3["proposers"]["mtp_k3"]["accepted_prefix_histogram"]
    counts = {int(prefix): int(count) for prefix, count in histogram.items()}
    if set(counts) != {0, 1, 2, 3} or sum(counts.values()) <= 0:
        raise ValueError("E3 MTP K3 histogram is incomplete")
    return sum((prefix + 1) * count for prefix, count in counts.items()) / sum(counts.values())


def e4_clusters(e4: dict[str, Any]) -> list[dict[str, Any]]:
    runs = e4.get("runs", [])
    if len(runs) < 2:
        raise ValueError("E4 needs at least two independent run clusters")
    clusters = []
    for run in runs:
        widths = {int(width): int(row["duration_ns"]) for width, row in run["joint_width_distribution"].items()}
        if set(widths) != {1, 2, 3, 4}:
            raise ValueError("E4 run lacks a joint-width duration")
        clusters.append({"f": float(run["verify_eligibility"]["f"]), "width_ns": widths})
    return clusters


def configuration_weights(clusters: list[dict[str, Any]]) -> dict[int, float]:
    totals = {width: sum(cluster["width_ns"][width] for cluster in clusters) for width in ACTIVE_WIDTH_SHAPE}
    denominator = sum(totals.values())
    if denominator <= 0:
        raise ValueError("E4 joint-width duration is zero")
    return {width: duration / denominator for width, duration in totals.items()}


def speed_factor(
    grouped: dict[tuple[int, str], list[dict[str, Any]]],
    committed_per_lane: float,
    weights: dict[int, float],
    baseline_rate: float,
) -> tuple[float, dict[int, float]]:
    candidate_rates = {}
    for width, shape in ACTIVE_WIDTH_SHAPE.items():
        round_ms = median_ns(grouped[(2_048, shape)], "total_ns") / 1_000_000.0
        candidate_rates[width] = width * committed_per_lane / round_ms
    weighted_rate = sum(weights[width] * candidate_rates[width] for width in weights)
    return weighted_rate / baseline_rate, candidate_rates


def bootstrap(
    grouped: dict[tuple[int, str], list[dict[str, Any]]],
    committed_per_lane: float,
    baseline: list[float],
    clusters: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    rng = random.Random(BOOTSTRAP_SEED)
    s_values: list[float] = []
    f_values: list[float] = []
    S_values: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_clusters = [rng.choice(clusters) for _ in clusters]
        weights = configuration_weights(sampled_clusters)
        sampled_grouped = dict(grouped)
        for width, shape in ACTIVE_WIDTH_SHAPE.items():
            source = grouped[(2_048, shape)]
            sampled_grouped[(2_048, shape)] = [rng.choice(source) for _ in source]
        sampled_baseline = [rng.choice(baseline) for _ in baseline]
        s, _ = speed_factor(
            sampled_grouped,
            committed_per_lane,
            weights,
            statistics.mean(sampled_baseline),
        )
        f = statistics.mean(cluster["f"] for cluster in sampled_clusters)
        S = 1.0 / ((1.0 - f) + f / s)
        s_values.append(s)
        f_values.append(f)
        S_values.append(S)
    return s_values, f_values, S_values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--e3", type=Path, required=True)
    parser.add_argument("--e4", type=Path, required=True)
    parser.add_argument("--b1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {"e2": args.e2, "e3": args.e3, "e4": args.e4, "b1": args.b1}
    e2 = load(args.e2)
    e3 = load(args.e3)
    e4 = load(args.e4)
    b1 = load(args.b1)
    grouped, warmups = validate_e2(e2)
    baseline = b1_rates(b1)
    committed = expected_committed_tokens(e3)
    clusters = e4_clusters(e4)
    weights = configuration_weights(clusters)
    s, candidate_rates = speed_factor(grouped, committed, weights, statistics.mean(baseline))
    f = statistics.mean(cluster["f"] for cluster in clusters)
    S = 1.0 / ((1.0 - f) + f / s)
    s_boot, f_boot, S_boot = bootstrap(grouped, committed, baseline, clusters)
    S_ci = interval(S_boot)
    go = s >= THRESHOLDS["s"] and S >= THRESHOLDS["S"] and S_ci[0] > THRESHOLDS["S_ci95_low"]

    configuration_summary = []
    for context in REQUIRED_CONTEXTS:
        for shape in REQUIRED_SHAPES:
            samples = grouped[(context, shape)]
            configuration_summary.append(
                {
                    "context_tokens": context,
                    "shape": shape,
                    "timed_runs": len(samples),
                    "stage_median_ns": {field: median_ns(samples, field) for field in STAGES},
                }
            )

    report = {
        "schema": "hf2q.adr065.g0.v1",
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "estimator": {
            "bootstrap": {
                "method": "run-clustered percentile bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "clusters": {"e2_per_configuration": 7, "e3": 1, "e4": len(clusters), "b1": len(baseline)},
                "e3_single_cluster_rule": "fixed aggregate; one retained run cannot estimate between-run variation",
            },
            "baseline_rate": "mean B1 n4 completion tokens / makespan ms",
            "candidate_rate": "E4-duration-weighted active-width rate using E2 near-2k median complete-round time",
            "committed_tokens": "E3 MTP accepted prefix + one canonical bonus token",
        },
        "e2": {"warmups": len(warmups), "configurations": configuration_summary},
        "point": {
            "expected_committed_tokens_per_lane": committed,
            "baseline_tokens_per_ms": statistics.mean(baseline),
            "configuration_weights": {str(width): weight for width, weight in weights.items()},
            "candidate_tokens_per_ms": {str(width): rate for width, rate in candidate_rates.items()},
            "s": s,
            "f": f,
            "S": S,
        },
        "ci95": {"s": interval(s_boot), "f": interval(f_boot), "S": S_ci},
        "thresholds": THRESHOLDS,
        "verdict": "Go" if go else "NoGo",
        "threshold_checks": {
            "s_at_least_1_15": s >= THRESHOLDS["s"],
            "S_at_least_1_05": S >= THRESHOLDS["S"],
            "S_ci95_low_above_1": S_ci[0] > THRESHOLDS["S_ci95_low"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
