"""Pure ADR-063 C2 §8.3–8.5 paired acceptance, not a full C2 qualification.

One ordered row per complete Williams block, never one row per slot or turn.
See README's C2 API contract. Source/calibration identity is the consumer's duty.
"""
import math

import q3_stats
from c2_metrics import APPEND, DIVERGENT, positive, require

ELIGIBILITY_SCOPE = "decode_ready_on_turns_with_at_least_two_decode_lanes"
REQUIRED = {"divergent": (DIVERGENT,), "append": APPEND}


def eligibility(record):
    """Validate supplied scoped lane-step counts/rate and explicit NotEligible reasons."""
    require(isinstance(record, dict) and record.get("scope") == ELIGIBILITY_SCOPE,
            "scoped decode-ready eligibility evidence required")
    counts = ("joint_lane_steps", "serial_lane_steps")
    have_counts = any(key in record for key in counts)
    rate = record.get("rate")
    if have_counts:
        require(all(type(record.get(key)) is int and record[key] >= 0 for key in counts),
                "both nonnegative integer lane-step counts required")
        joint, serial = (record[key] for key in counts)
        require(joint + serial > 0, "eligibility denominator must be positive")
        measured = joint / (joint + serial)
        if "rate" in record:
            require(type(rate) in (int, float) and math.isfinite(rate) and
                    math.isclose(rate, measured, rel_tol=0, abs_tol=1e-12),
                    "eligibility rate disagrees with counters")
        rate = measured
    require(type(rate) in (int, float) and math.isfinite(rate) and 0 <= rate <= 1,
            "eligibility rate must be finite in [0,1]")
    reasons = record.get("not_eligible")
    require(isinstance(reasons, dict) and {"anchor_capture", "anchor_restore"} <= set(reasons),
            "explicit anchor_capture and anchor_restore counts required")
    require(all(isinstance(key, str) and key and type(value) is int and value >= 0
                for key, value in reasons.items()), "nonnegative integer NotEligible counts required")
    return {**record, "rate": rate,
            "anchor_reasons_zero": all(value == 0 for key, value in reasons.items() if key.startswith("anchor_"))}


def _effect(pairs, higher=False):
    try:
        return q3_stats.paired_effect(pairs, higher_is_better=higher)
    except (OverflowError, ValueError) as exc:
        raise ValueError("C2 paired effect is not finite/representable") from exc


def campaign(kind, blocks):
    """Return an immediate hard Fail, otherwise decide at the first C2 statistical stop.

    Invalid/incomplete data raises ValueError. Zero eligibility, nonzero anchor
    reasons and footprint violations fail immediately, even before five blocks.
    stop_reason names only the statistical stop and can be None on a hard Fail.
    Pass covers these numerical checks only.
    Every run supplies its block_id, outcome='Pass', metrics and eligibility;
    each subject also supplies its peak and matching fresh calibration peak.
    """
    require(kind in REQUIRED, "campaign kind must be divergent or append")
    require(isinstance(blocks, list) and len(blocks) <= 20, "C2 requires at most 20 ordered blocks")
    names = REQUIRED[kind]
    pairs = {name: [] for name in names}
    rate_pairs, audit, seen = [], [], set()
    zero_blocks, anchor_failures, footprint_failures = [], [], []
    first_stop = None
    effects = {name: _effect([]) for name in names}
    for position, block in enumerate(blocks, 1):
        require(isinstance(block, dict), "paired block required")
        identity = block.get("block_id")
        require(isinstance(identity, str) and bool(identity) and identity not in seen,
                "unique nonempty block_id required")
        seen.add(identity)
        require(first_stop is None, f"blocks extend beyond preregistered stop at {first_stop}")
        rates = []
        row = {"block_id": identity}
        for arm in ("subject", "reference"):
            run = block.get(arm)
            require(isinstance(run, dict) and run.get("block_id") == identity,
                    f"{identity}: mismatched or incomplete {arm} block")
            require(run.get("outcome") == "Pass", f"{identity}: {arm} run did not pass")
            metrics = run.get("metrics")
            require(isinstance(metrics, dict), f"{identity}: {arm} metrics required")
            for name in names:
                positive(metrics.get(name), f"{identity}: {arm} {name}")
            evidence = eligibility(run.get("eligibility"))
            rates.append(evidence["rate"])
            row[arm] = {"metrics": metrics, "eligibility": evidence}
            if not evidence["anchor_reasons_zero"]:
                anchor_failures.append({"block_id": identity, "arm": arm})
        for name in names:
            pairs[name].append((block["subject"]["metrics"][name], block["reference"]["metrics"][name]))
        rate_pairs.append(tuple(rates))
        if 0 in rates:
            zero_blocks.append(identity)
        subject = block["subject"]
        peak = positive(subject.get("peak_footprint_bytes"), f"{identity}: subject peak footprint")
        calibration = positive(subject.get("calibration_peak_footprint_bytes"), f"{identity}: fresh calibration peak")
        ratio = peak / calibration
        require(math.isfinite(ratio), "footprint ratio must be finite")
        row["footprint_ratio"] = ratio
        if ratio > 1.25:
            footprint_failures.append(identity)
        audit.append(row)
        effects = {name: _effect(values) for name, values in pairs.items()}
        # Eligibility is acceptance, deliberately NOT an additional stopping contrast.
        stop = q3_stats.stop_decision(position, effects, max_blocks=20)
        if stop is not None:
            first_stop = identity
    n = len(blocks)
    stop = q3_stats.stop_decision(n, effects, max_blocks=20)
    eligible_effect = None if zero_blocks else _effect(rate_pairs, higher=True)
    checks = {name: None for name in names}
    checks.update(eligibility=False if zero_blocks else None, anchor_reasons_zero=not anchor_failures,
                  footprint=not footprint_failures)
    reasons = []
    if zero_blocks:
        reasons.append("zero eligibility: fail closed pending owner review")
    if anchor_failures:
        reasons.append("nonzero anchor_* NotEligible count")
    if footprint_failures:
        reasons.append("subject footprint exceeds 1.25 times supplied fresh calibration peak")
    if stop is not None:
        for name, effect in effects.items():
            interval = effect["ratio_ci95"]
            checks[name] = interval is not None and (interval[1] < 1 if kind == "divergent" else interval[1] <= 1.025)
            if not checks[name]:
                reasons.append(f"{name}: required upper 95% CI bound not met")
        interval = None if eligible_effect is None else eligible_effect["ratio_ci95"]
        checks["eligibility"] = interval is not None and interval[1] >= 1
        if not checks["eligibility"] and not zero_blocks:
            reasons.append("eligibility CI lies below 1")
    hard_failure = bool(zero_blocks or anchor_failures or footprint_failures)
    status = "Fail" if hard_failure else ("Continue" if stop is None else ("Pass" if all(checks.values()) else "Fail"))
    return {"scope": "C2 statistical component only", "campaign": kind, "blocks": n,
            "block_ids": [row["block_id"] for row in audit], "status": status, "stop_reason": stop,
            "effects": effects, "eligibility_effect": eligible_effect, "checks": checks,
            "zero_eligibility_blocks": zero_blocks, "anchor_failures": anchor_failures,
            "footprint_failures": footprint_failures, "reasons": reasons, "runs": audit}
