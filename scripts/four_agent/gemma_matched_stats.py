"""Frozen pure plan/metric/comparability math. No evidence or host access on import."""
import math
from itertools import chain
import random
import statistics

from qwen38_n1_verdict import require

PLAN = tuple({"index": block * 4 + position, "block": block, "position": position, "arm": arm}
             for block in range(5) for position, arm in enumerate(("OFF", "ON", "ON", "OFF")))
BOOTSTRAP_SEED = 6306
BOOTSTRAP_DRAWS = 10000
AGENTS = tuple(f"agent-{i}" for i in range(4))


def positive(value):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            "positive finite measured value required")
    return value


def cohort_score(agents):
    require(set(agents) == set(AGENTS), "four complete independent conversations")
    times = []
    for agent in AGENTS:
        result = agents[agent]
        require(result.get("benchmark_profile") == "gemma-matched-v1" and
                result.get("status") == "ProtocolCompleteAwaitingIndependentPatchReview" and
                result.get("cleanup") is True, "actual benchmark conversation/cleanup required")
        turns = result.get("turns", [])
        require(len(turns) == 7 and [t["turn"] for t in turns] == list(range(7)) and
                [t["stream"] for t in turns] == [False, True, False, True, False, True, False],
                "fixed seven-turn transport matrix")
        require(len(result.get("tools", [])) == 6, "all six real tools")
        for index in (3, 5):
            tool = result["tools"][index]
            require(tool.get("benchmark_profile") == "gemma-matched-v1" and
                    tool.get("passed") is True and tool.get("status") == "Complete",
                    "both actual semantic test tools required")
        times.extend(positive(turns[t]["semantic_ttft_seconds"]) for t in (1, 3, 5))
    return {"schema": "hf2q.gemma-matched-cohort-score.v1",
            "primary_sse_geomean_seconds": math.exp(sum(map(math.log, times)) / 12),
            "sse_observations": 12, "unary_semantic_ttft_claimed": False,
            "input_comparability": "PendingProductionTokenVectors", "performance_qualified": False}


def exact_vectors(left, right):
    """No hashes-only or canonicalized-transcript surrogate for exact token equality."""
    require(set(left) == set(right) == set(AGENTS), "all four token-vector conversations")
    for agent in AGENTS:
        require(len(left[agent]) == len(right[agent]) == 7, "all28 rendered inputs")
        for a, b in zip(left[agent], right[agent]):
            require(isinstance(a, list) and isinstance(b, list) and 0 < len(a) <= 262144 - 8192 and 0 < len(b) <= 262144 - 8192 and
                    all(type(t) is int and 0 <= t <= 0xffffffff for t in chain(a, b)),
                    "actual production u32 input token vectors required")
            require(a == b, "incomparable rendered prompt tokens; no favorable subset")


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lo = int(position)
    return sorted_values[lo] + (sorted_values[min(lo + 1, len(sorted_values) - 1)] - sorted_values[lo]) * (position - lo)


def analyze(runs, vectors):
    """Caller must bind actual quality/rendered-vector evidence before this pure math.

    This function deliberately cannot mark evidence/performance qualified. The
    later reviewed retained-evidence adapter must join source and raw proof.
    """
    require(len(runs) == len(vectors) == 20, "fixed20 cohorts; no missing/extra/retry replacement")
    scores, walls = [], []
    for plan, run in zip(PLAN, runs):
        require(all(type(run.get(k)) is type(v) and run.get(k) == v for k, v in plan.items()) and
                run.get("quality_verified") is True and run.get("cleanup_verified") is True,
                "fixed ABBA plan/all quality and cleanup required")
        scores.append(positive(run["primary_sse_geomean_seconds"]))
        walls.append(positive(run["coordinator_wall_seconds"]))
    for block in range(5):
        start = 4 * block
        exact_vectors(vectors[start], vectors[start + 1])
        exact_vectors(vectors[start + 3], vectors[start + 2])
    off = [i for i, p in enumerate(PLAN) if p["arm"] == "OFF"]
    on = [i for i, p in enumerate(PLAN) if p["arm"] == "ON"]
    ratio = statistics.median(scores[i] for i in on) / statistics.median(scores[i] for i in off)
    rng, ratios = random.Random(BOOTSTRAP_SEED), []
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = [rng.randrange(5) for _ in range(5)]
        a = [scores[b * 4 + p] for b in chosen for p in (0, 3)]
        b = [scores[k * 4 + p] for k in chosen for p in (1, 2)]
        ratios.append(statistics.median(b) / statistics.median(a))
    ratios.sort()
    lo, hi = percentile(ratios, .025), percentile(ratios, .975)
    improvement = 100 * (1 - ratio)
    interval = [100 * (1 - hi), 100 * (1 - lo)]
    halfwidth = (interval[1] - interval[0]) / 2
    precision = halfwidth <= 2.5
    return {"schema": "hf2q.gemma-matched-study-calculation.v2", "cohorts": 20,
            "primary_ratio_on_over_off": ratio, "paired_block_ci95": [lo, hi],
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
            "component_metric": "SSE1_3_5_semanticTTFT_geomean_not_N4_makespan_goodput",
            "point_improvement_percent": improvement,
            "ci95_improvement_percent": interval,
            "ci95_halfwidth_percentage_points": halfwidth,
            "precision_target_halfwidth_pp": 2.5, "precision_target_met": precision,
            "stock_point_at_least_5percent_met": improvement >= 5,
            "component_latency_improvement_observed": ratio < 1 and hi < 1 and precision,
            "component_study_status": "PreciseComponentLatencyEstimate" if precision else "InconclusiveFixedPlanPrecisionMiss",
            "peer_comparison_missing": True, "campaign_primary_acceptance": False,
            "secondary_wall_ratio_on_over_off": statistics.median(walls[i] for i in on) / statistics.median(walls[i] for i in off),
            "status": "TimingCalculatedAwaitingIndependentEvidenceReview",
            "performance_qualified": False, "default_promotion": False}
