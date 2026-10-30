"""ADR-063 Q3 crossover design, paired effect estimator, stopping rule and family verdict. Pure.

Everything here is predeclared in docs/research/adr063-q3-campaign.md ("Design",
"Estimator", "Verdict"); change the doc and this module together, before data.
"""
import math
import statistics

MIN_BLOCKS = 5
MAX_BLOCKS = 20
HALF_WIDTH_PP = 2.5    # target 95% half-width, percentage points of the arm ratio
PEER_MARGIN = 0.95     # Q4: at least 5% lower four-agent makespan than the peer
NEUTRAL_RATIO = 1.025  # fairness regression band (the F2 2.5-point neutral band)
# Two-sided 95% Student t quantiles, df = complete blocks - 1 (blocks <= 20).
T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
        9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
        16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930}


def williams(t):
    """Williams sequences over treatment indices 0..t-1: each treatment once per position and each
    ordered pair adjacent equally often. t rows for even t; t rows plus their reversals for odd t."""
    if t < 1:
        raise ValueError("at least one arm")
    first = [0] + [(j + 1) // 2 if j % 2 else t - j // 2 for j in range(1, t)]
    rows = [[(x + i) % t for x in first] for i in range(t)]
    if t % 2:
        rows += [row[::-1] for row in rows]
    return rows


def block_order(arms, block):
    """Arm order of 1-based `block`: the Williams sequences in turn."""
    rows = williams(len(arms))
    return [arms[i] for i in rows[(block - 1) % len(rows)]]


def paired_effect(pairs, higher_is_better=False):
    """Paired log-ratio effect of subject vs reference over complete blocks.

    `pairs` holds one (subject, reference) positive value per block, so each block (an independent
    pair of runs, each a whole four-agent cohort) is the unit of clustering. d_b = ln(subject_b) -
    ln(reference_b); the estimate is mean(d), its 95% interval mean(d) +- t(0.975, n-1) * sd(d)/sqrt(n).
    The ratio is exp(mean(d)); the half-width in percentage points is 100 * (exp(h) - 1).
    """
    for subject, reference in pairs:
        if not (subject > 0 and reference > 0 and math.isfinite(subject) and math.isfinite(reference)):
            raise ValueError("effect values must be positive and finite")
    d = [math.log(s) - math.log(r) for s, r in pairs]
    n = len(d)
    out = {"blocks": n, "log_ratios": d, "mean_log_ratio": None, "sd_log_ratio": None,
           "half_width_log": None, "ratio": None, "ratio_ci95": None, "half_width_pp": None,
           "improvement_percent": None, "improvement_ci95": None,
           "direction": "higher_is_better" if higher_is_better else "lower_is_better"}
    if n < 2:
        return out
    if n - 1 not in T975:
        raise ValueError("more blocks than the predeclared maximum")
    mean, sd = statistics.fmean(d), statistics.stdev(d)
    h = T975[n - 1] * sd / math.sqrt(n)
    low, high = math.exp(mean - h), math.exp(mean + h)
    sign = 1 if higher_is_better else -1
    improvement = sorted(100 * sign * (x - 1) for x in (low, high))
    out.update(mean_log_ratio=mean, sd_log_ratio=sd, half_width_log=h, ratio=math.exp(mean),
               ratio_ci95=[low, high], half_width_pp=100 * (math.exp(h) - 1),
               improvement_percent=100 * sign * (math.exp(mean) - 1), improvement_ci95=improvement)
    return out


def precise(effect):
    return (effect is not None and effect["half_width_pp"] is not None and effect["blocks"] >= MIN_BLOCKS
            and effect["half_width_pp"] <= HALF_WIDTH_PP)


def stop_decision(completed_blocks, required):
    """None to continue; else the stop reason. `required` maps contrast name -> effect."""
    if completed_blocks >= MIN_BLOCKS and required and all(precise(e) for e in required.values()):
        return "precision"
    if completed_blocks >= MAX_BLOCKS:
        return "max_blocks"
    return None


def regression(effect):
    """Lower-is-better metric significantly worse than the reference beyond the neutral band."""
    return effect is not None and effect["ratio_ci95"] is not None and effect["ratio_ci95"][0] > NEUTRAL_RATIO


def family_verdict(state):
    """Per-family Q3/Q4 verdict from the primary (pure-phase four-agent) workload.

    `state` keys: experiment_failures, peer_unavailable (dict or None), peer_failures, quality
    ({"status": "pass"|"loss"|"unresolved"|"none", ...}), complete (bool), vs_stock, vs_peer,
    fairness ({name: effect} of lower-is-better experiment-vs-stock effects), thermal_valid
    (False only when a serious/critical state was observed), memory_valid (bool).
    """
    if state["experiment_failures"]:
        return "Failed", ["experiment arm failed its task: " + "; ".join(state["experiment_failures"])]
    if state["quality"]["status"] == "loss":
        return "Failed", ["experiment task quality loss: " + state["quality"]["reason"]]
    if state["peer_unavailable"] is not None:
        return "Unavailable", ["no runnable peer arm: " + state["peer_unavailable"]["reason"]]
    if state["peer_failures"]:
        return "Failed", ["peer arm failed its task (a failed peer cannot yield a win): "
                          + "; ".join(state["peer_failures"])]
    if not state["complete"]:
        return "Pending", ["primary workload has not reached its stopping rule"]
    vs_stock, vs_peer = state["vs_stock"], state["vs_peer"]
    if vs_peer["ratio_ci95"] is not None and vs_peer["ratio_ci95"][0] > 1:
        return "Loss", ["experiment makespan is longer than the peer's (95% CI of the ratio above 1)"]
    reasons = []
    if not (precise(vs_stock) and precise(vs_peer)):
        reasons.append(f"precision not reached: 95% half-width must be <= {HALF_WIDTH_PP} points for both contrasts")
    if vs_stock["ratio_ci95"] is None or vs_stock["ratio_ci95"][1] >= 1:
        reasons.append("no demonstrated makespan gain over stock (95% CI of the ratio reaches 1)")
    if vs_peer["ratio_ci95"] is None or vs_peer["ratio_ci95"][1] > PEER_MARGIN:
        reasons.append("peer margin not demonstrated (95% CI upper bound of the ratio above 0.95)")
    for name, effect in sorted(state["fairness"].items()):
        if regression(effect):
            reasons.append(f"fairness regression vs stock: {name}")
    if state["quality"]["status"] == "unresolved":
        reasons.append("task quality unresolved: " + state["quality"]["reason"])
    if state["thermal_valid"] is False:
        reasons.append("a run observed serious or critical thermal state")
    if not state["memory_valid"]:
        reasons.append("a measured run lacks a passing Q5 memory verdict")
    return ("Win", []) if not reasons else ("Inconclusive", reasons)
