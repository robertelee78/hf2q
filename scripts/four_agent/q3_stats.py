"""ADR-063 Q3 crossover design, paired effect estimator, stopping rule and family verdict. Pure.

Everything here is predeclared in docs/research/adr063-q3-campaign.md ("Design",
"Estimator", "Comparison-arm failure policy", "Verdict"); change the doc and this module together, before data.
"""
import math
import statistics

MIN_BLOCKS = 5
MAX_BLOCKS = 20
MAX_SUPPORTED_BLOCKS = 40
HALF_WIDTH_PP = 2.5          # target 95% half-width, percentage points of the arm ratio
LEGACY_STOCK_MARGIN = 1.0    # Q3: any demonstrated gain over v0.1.28 stock
LEGACY_PEER_MARGIN = 0.95    # original Q4: at least 5% lower makespan than the peer
STOCK_MARGIN = 0.95          # v0.1.30: at least 5% lower makespan than v0.1.29
PEER_PARITY = 1.0            # v0.1.30: statistically no slower than every valid peer
NEUTRAL_RATIO = 1.025        # fairness regression band (the F2 2.5-point neutral band)
# Stop reason of a no-stock stage (--no-stock-arm) whose only contrast (the peer) became unavailable or failed.
NO_CONTRAST = "no_contrast"
# Comparison-arm failure policy (ADR-063 Q4, operator decision 2026-11-12, "strict on us, fair on them"). A plan that
# carries `comparison_failure_policy` reruns a comparison arm (stock, the peer, Splash; never the experiment) whose run
# failed its own task, in the same block position, at most `reruns_per_position` times. A plan without the field keeps
# the rule it was planned under (LEGACY_FAILURE_RULE): no reruns.
FAILURE_POLICY_VERSION = 1
RERUNS_PER_POSITION = 2
RERUN_FAILURE_KINDS = ("task",)   # q3_metrics.classify kind of a client that ran to its end and did not complete the task
COMPARISON_ARMS = ("stock", "peer", "splash")
LEGACY_FAILURE_RULE = "legacy: no reruns; a peer or required-Splash task failure is Failed"
# Two-sided 95% Student t quantiles, df = complete blocks - 1.
T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
        9: 2.2622, 10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314,
        16: 2.1199, 17: 2.1098, 18: 2.1009, 19: 2.0930, 20: 2.0860, 21: 2.0796, 22: 2.0739,
        23: 2.0687, 24: 2.0639, 25: 2.0595, 26: 2.0555, 27: 2.0518, 28: 2.0484, 29: 2.0452,
        30: 2.0423, 31: 2.0395, 32: 2.0369, 33: 2.0345, 34: 2.0322, 35: 2.0301, 36: 2.0281,
        37: 2.0262, 38: 2.0244, 39: 2.0227}


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


def stop_decision(completed_blocks, required, max_blocks=MAX_BLOCKS):
    """None to continue; else the stop reason. `required` maps contrast name -> effect."""
    if max_blocks not in (MAX_BLOCKS, MAX_SUPPORTED_BLOCKS):
        raise ValueError("max_blocks must be the predeclared 20 or 40 block cap")
    if completed_blocks >= MIN_BLOCKS and required and all(precise(e) for e in required.values()):
        return "precision"
    if completed_blocks >= max_blocks:
        return "max_blocks"
    return None


def failure_policy(plan):
    """The plan's frozen comparison-arm failure policy, validated; None for a plan recorded before the policy (or
    planned with zero reruns), which keeps the legacy rule."""
    policy = plan.get("comparison_failure_policy")
    if policy is None:
        return None
    if (not isinstance(policy, dict) or policy.get("version") != FAILURE_POLICY_VERSION
            or policy.get("reruns_per_position") != RERUNS_PER_POSITION
            or policy.get("rerun_failure_kinds") != list(RERUN_FAILURE_KINDS)
            or not isinstance(policy.get("arms"), list) or not set(policy["arms"]) <= set(COMPARISON_ARMS)):
        raise ValueError(f"unsupported comparison_failure_policy {policy!r}")
    return policy


def failure_rule(plan):
    """One line naming the rule a verdict was computed under."""
    policy = failure_policy(plan)
    if policy is None:
        return LEGACY_FAILURE_RULE
    return (f"comparison_failure_policy v{policy['version']}: a task failure of {', '.join(policy['arms']) or 'no arm'} "
            f"is rerun in its block position, at most {policy['reruns_per_position']} reruns; an experiment failure "
            "is Failed")


def rerun_decision(policy, arm, outcome, failure_kind, prior_failures):
    """What a settled attempt means under the policy: None (the legacy handling applies), "rerun" (discard the failed
    attempt and run the same arm again in the same position) or "exhausted" (every allowed attempt of the position
    failed). The inputs are the arm, the attempt's outcome and failure kind and the number of failed attempts already
    discarded in this position; no timing, metric or effect can reach the decision, so a slow passing run is never
    rerun."""
    if policy is None or arm not in policy["arms"] or outcome != "Fail" or failure_kind not in policy["rerun_failure_kinds"]:
        return None
    return "rerun" if prior_failures < policy["reruns_per_position"] else "exhausted"


def _binomial_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_pearson(failures, attempts, confidence=0.95):
    """Exact two-sided binomial interval [low, high] for a failure rate; None without attempts."""
    if not (isinstance(failures, int) and isinstance(attempts, int) and 0 <= failures <= attempts):
        raise ValueError("failures must be an integer in 0..attempts")
    if attempts == 0:
        return None
    alpha = (1 - confidence) / 2

    def solve(tail, target):  # tail is increasing in p
        low, high = 0.0, 1.0
        for _ in range(200):
            mid = (low + high) / 2
            low, high = (mid, high) if tail(mid) < target else (low, mid)
        return (low + high) / 2

    lower = 0.0 if failures == 0 else solve(lambda p: _binomial_tail(failures, attempts, p), alpha)
    upper = 1.0 if failures == attempts else solve(lambda p: _binomial_tail(failures + 1, attempts, p), 1 - alpha)
    return [lower, upper]


def failure_rate(failures, attempts):
    """{attempts, failures, failure_rate, failure_rate_ci95} of task-scored attempts (passes plus task failures)."""
    return {"attempts": attempts, "failures": failures, "failure_rate": failures / attempts if attempts else None,
            "failure_rate_ci95": clopper_pearson(failures, attempts)}


def regression(effect):
    """Lower-is-better metric significantly worse than the reference beyond the neutral band."""
    return effect is not None and effect["ratio_ci95"] is not None and effect["ratio_ci95"][0] > NEUTRAL_RATIO


def family_verdict(state):
    """Per-family Q3/Q4 verdict from the primary (pure-phase four-agent) workload.

    `state` keys: experiment_failures, peer_unavailable (dict or None), peer_failures, quality
    ({"status": "pass"|"loss"|"unresolved"|"none", ...}), complete (bool), vs_stock, vs_peer,
    fairness ({name: effect} of lower-is-better experiment-vs-stock effects), thermal_valid
    (False only when a serious/critical state was observed), memory_valid (bool). Optional:
    stock_margin_ratio and peer_margin_ratio select the source-bound release contract; new v0.1.30
    campaigns use STOCK_MARGIN and PEER_PARITY. ADR-063 "Splash peer of record for the Qwen families"
    adds splash_required (bool), splash_unavailable (dict or None), splash_failures (list), and
    vs_splash (effect). Where Splash is required, a missing, unavailable or failed Splash arm never
    yields a Win and the vs-Splash ratio must meet the same peer margin. ADR-063 Q4-DS4-RACE decision
    D1 optionally adds stock_source ({"record", "sha256", "status", "reason"}), naming the bound
    same-artifact record that supplies vs_stock and fairness; every vs-stock reason names it, and a
    missing or non-Complete record yields Pending with its reason. The comparison-arm failure policy (operator
    decision 2026-11-12) optionally adds stock_unavailable (dict or None): a stock arm that failed its task in every
    allowed attempt of one block position. Under that policy the driver records the same exhaustion of the peer or
    Splash as peer_unavailable or splash_unavailable, so an exhausted comparison arm is Unavailable, never Failed and
    never a Win; a comparison failure that was rerun to a pass is not in peer_failures or splash_failures.
    """
    splash = bool(state.get("splash_required"))
    source = state.get("stock_source")
    named = "" if source is None else f" (from the bound stock record {source['record']}, sha256 {source['sha256']})"
    if state["experiment_failures"]:
        return "Failed", ["experiment arm failed its task: " + "; ".join(state["experiment_failures"])]
    if state["quality"]["status"] == "loss":
        return "Failed", ["experiment task quality loss: " + state["quality"]["reason"]]
    if state["peer_unavailable"] is not None:
        return "Unavailable", ["no runnable peer arm: " + state["peer_unavailable"]["reason"]]
    if splash and state.get("splash_unavailable") is not None:
        return "Unavailable", ["no runnable Splash arm: " + state["splash_unavailable"]["reason"]]
    if state.get("stock_unavailable") is not None:
        return "Unavailable", ["no usable stock arm: " + state["stock_unavailable"]["reason"] + named]
    if state["peer_failures"]:
        return "Failed", ["peer arm failed its task (a failed peer cannot yield a win): "
                          + "; ".join(state["peer_failures"])]
    if splash and state.get("splash_failures"):
        return "Failed", ["Splash arm failed its task (a failed Splash arm cannot yield a win): "
                          + "; ".join(state["splash_failures"])]
    if source is not None and source["status"] != "Complete":
        return "Pending", [f"the bound stock record {source['record']} is {source['status']}: {source['reason']}; "
                           "no vs-stock effect"]
    if not state["complete"]:
        return "Pending", ["primary workload has not reached its stopping rule"]
    vs_stock, vs_peer = state["vs_stock"], state["vs_peer"]
    vs_splash = state["vs_splash"] if splash else None
    stock_margin = state.get("stock_margin_ratio", STOCK_MARGIN)
    peer_margin = state.get("peer_margin_ratio", PEER_PARITY)
    if vs_peer["ratio_ci95"] is not None and vs_peer["ratio_ci95"][0] > 1:
        return "Loss", ["experiment makespan is longer than the peer's (95% CI of the ratio above 1)"]
    if splash and vs_splash["ratio_ci95"] is not None and vs_splash["ratio_ci95"][0] > 1:
        return "Loss", ["experiment makespan is longer than Splash's (95% CI of the ratio above 1)"]
    reasons = []
    if not (precise(vs_stock) and precise(vs_peer) and (not splash or precise(vs_splash))):
        reasons.append(f"precision not reached: 95% half-width must be <= {HALF_WIDTH_PP} points for "
                       + ("every contrast (stock, peer, Splash)" if splash else "both contrasts")
                       + ("" if precise(vs_stock) else named))
    if vs_stock["ratio_ci95"] is None or vs_stock["ratio_ci95"][1] > stock_margin:
        reasons.append(f"stock margin not demonstrated (95% CI upper bound of the ratio above "
                       f"{stock_margin:.2f})" + named)
    if vs_peer["ratio_ci95"] is None or vs_peer["ratio_ci95"][1] > peer_margin:
        reasons.append(f"peer parity not demonstrated (95% CI upper bound of the ratio above "
                       f"{peer_margin:.2f})")
    if splash and (vs_splash["ratio_ci95"] is None or vs_splash["ratio_ci95"][1] > peer_margin):
        reasons.append(f"Splash parity not demonstrated (95% CI upper bound of the ratio above "
                       f"{peer_margin:.2f})")
    for name, effect in sorted(state["fairness"].items()):
        if regression(effect):
            reasons.append(f"fairness regression vs stock: {name}" + named)
    if state["quality"]["status"] == "unresolved":
        reasons.append("task quality unresolved: " + state["quality"]["reason"])
    if state["thermal_valid"] is False:
        reasons.append("a run observed serious or critical thermal state")
    if not state["memory_valid"]:
        reasons.append("a measured run lacks a passing Q5 memory verdict")
    return ("Win", []) if not reasons else ("Inconclusive", reasons)
