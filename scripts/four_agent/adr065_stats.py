"""ADR-065 two-arm fixed-count A/B analysis and the B1 fairness ceilings. Pure.

Predeclared in docs/research/adr065-b1-campaign.md and scripts/four_agent/adr065/campaign-b1.json
(ADR-065 "Measurement and acceptance rules"); change them together, before data.

Inputs are campaign.json manifests written by adr065_campaign.py (the q3_campaign.Driver layout: stages,
warmups, blocks, entries with outcome and q3_metrics.run_metrics). Arm A is the previous default, arm B the
candidate; the effect is the reduction of B's makespan against A's.
"""
import q3_report
import q3_stats

DEFAULT_BLOCKS = 6
ALLOWED_BLOCKS = (6, 8, 12, 16, 20)
MARGIN_PP = 2.5
VERDICTS = ("Loss", "Win", "Equivalent", "Inconclusive")
PRIMARY = "n4"
MIXED = ("3+1", "2+2", "1+3")
CEILING_SCHEMA = "hf2q.adr065.fairness-ceilings.v1"
ANALYSIS_SCHEMA = "hf2q.adr065.ab-analysis.v1"
TTFT, SSE = "ttft_p95_s", "sse_gap_max_s"  # q3_metrics.slot_metrics keys (ADR-063 F2 estimands)
CAP_FACTOR = 1.5  # absolute safety cap: frozen baseline ceiling + 50%
CEILING_Q5 = "CalibrationPass"


def effect(stage):
    """Paired log-makespan effect of B against A over complete blocks whose two runs both passed
    (q3_report.effect: block-paired ln(B/A), Student-t 95% interval; improvement = reduction in points)."""
    return q3_report.effect(stage, "B", "A", q3_report.PRIMARY_METRIC)


def verdict(effect, fixed_blocks):
    """(verdict, reasons) in the ADR-065 order: Loss, Win, Equivalent, Inconclusive.

    The interval is effect["improvement_ci95"] (percentage-point reduction, positive = B faster). The Win
    half-width is q3_stats' 100 * (exp(h) - 1), the ADR-063 precision definition; it bounds the half-width
    of the reduction interval from above whenever B is faster. Fewer quality-eligible blocks than the fixed
    count keep a Loss but never yield Win or Equivalent."""
    ci = effect["improvement_ci95"]
    if ci is None:
        return "Inconclusive", [f"no interval: {effect['blocks']} quality-eligible blocks of {fixed_blocks}"]
    low, high = ci
    if high < 0:
        return "Loss", [f"B is slower: reduction interval [{low:.2f}, {high:.2f}] points excludes zero"]
    if effect["blocks"] < fixed_blocks:
        return "Inconclusive", [f"only {effect['blocks']} of {fixed_blocks} blocks are quality-eligible"]
    if low > 0 and effect["half_width_pp"] <= MARGIN_PP:
        return "Win", []
    if low <= 0 <= high and -MARGIN_PP <= low and high <= MARGIN_PP:
        return "Equivalent", []
    if low > 0:
        return "Inconclusive", [f"faster but imprecise: half-width {effect['half_width_pp']:.2f} > {MARGIN_PP} points"]
    return "Inconclusive", [f"interval [{low:.2f}, {high:.2f}] points contains zero and leaves +-{MARGIN_PP}"]


def within_margin(effect):
    """N=1 control rule: the whole interval lies within +-2.5 points."""
    ci = effect["improvement_ci95"]
    return ci is not None and -MARGIN_PP <= ci[0] and ci[1] <= MARGIN_PP


# --- runs ---------------------------------------------------------------------------------------


def runs(stage, kinds=("measured",)):
    for entry in stage["warmups"] + [e for block in stage["blocks"] for e in block["entries"]]:
        if entry["kind"] in kinds:
            yield entry


def retained(manifest):
    """Every run of the campaign, failed and discarded ones included, as one row each."""
    rows = []
    for stage in manifest["stages"]:
        for entry in runs(stage, ("warmup", "measured")):
            metrics = entry.get("metrics") or {}
            rows.append({"name": entry["name"], "workload": stage["workload"], "arm": entry["arm"], "kind": entry["kind"],
                         "block": entry["block"], "outcome": entry["outcome"], "failure_kind": entry.get("failure_kind"),
                         "reason": entry.get("reason"), "attempts": [a["name"] for a in entry["attempts"]],
                         "quality_eligible": entry["outcome"] == "Pass",
                         "makespan_s": metrics.get("makespan_s"),
                         "q5_verdict": (metrics.get("memory") or {}).get("q5_verdict"),
                         "thermal_valid": (metrics.get("thermal") or {}).get("valid"),
                         "slots": fairness_values(metrics) if metrics else None})
    return rows


def fairness_values(metrics):
    """Per-slot p95 semantic TTFT and maximum SSE gap, and the run's maximum SSE gap."""
    slots = {name: {TTFT: s.get(TTFT), SSE: s.get(SSE)} for name, s in sorted((metrics.get("slots") or {}).items())}
    gaps = [s[SSE] for s in slots.values() if s[SSE] is not None]
    return {"slots": slots, "run_sse_gap_max_s": max(gaps, default=None)}


# --- ceilings -----------------------------------------------------------------------------------


def ceiling_exclusion(entry):
    """Why a measured baseline run may not set a ceiling; None when it may (a Pass with a CalibrationPass
    Q5 verdict and no serious or critical thermal state)."""
    if entry["outcome"] != "Pass":
        return f"outcome {entry['outcome']}"
    metrics = entry["metrics"]
    if metrics["memory"]["q5_verdict"] != CEILING_Q5:
        return f"Q5 verdict {metrics['memory']['q5_verdict']}"
    if metrics["thermal"]["valid"] is False:
        return "serious or critical thermal state"
    return None


def stage_ceilings(stage):
    """Largest per-slot p95 TTFT and largest per-run maximum SSE gap over the stage's eligible measured runs
    (ceiling_exclusion). A metric no eligible run measured (one semantic event per streamed turn) is None."""
    best, excluded, eligible = {TTFT: None, SSE: None}, [], 0
    for entry in runs(stage):
        why = ceiling_exclusion(entry)
        if why is not None:
            excluded.append({"run": entry["name"], "reason": why})
            continue
        eligible += 1
        values = fairness_values(entry["metrics"])
        for slot, slot_values in values["slots"].items():
            for key in (TTFT, SSE):
                value = slot_values[key]
                if value is not None and (best[key] is None or value > best[key]["value"]):
                    best[key] = {"value": value, "run": entry["name"], "slot": slot}
    return {"eligible_runs": eligible, "measured_runs": sum(1 for _ in runs(stage)), "excluded_runs": excluded,
            TTFT: best[TTFT] or {"value": None, "run": None, "slot": None},
            SSE: best[SSE] or {"value": None, "run": None, "slot": None}}


def ceilings(manifest, sources):
    """The ceilings document of a complete baseline campaign; `sources` binds the campaign files by sha256."""
    plan = manifest["plan"]
    if plan.get("mode") != "baseline":
        raise ValueError(f"ceilings come only from a baseline campaign (this one is {plan.get('mode')})")
    if manifest["status"] != "Complete" or any(s["status"] != "Complete" for s in manifest["stages"]):
        raise ValueError(f"the baseline campaign is {manifest['status']}, stages "
                         + ", ".join(f"{s['workload']}={s['status']}" for s in manifest["stages"]))
    stages = {stage["workload"]: stage_ceilings(stage) for stage in manifest["stages"]}
    empty = [key for key, value in stages.items() if value[TTFT]["value"] is None]
    if empty:
        raise ValueError("no eligible measured run (Pass, CalibrationPass, thermal valid) defines a TTFT ceiling for "
                         + ", ".join(empty))
    rows = retained(manifest)
    arm = manifest["identities"]["arms"]["A"]
    return {"schema": CEILING_SCHEMA, "family": plan["family"], "sources": sources,
            "baseline": {**plan["baseline"], "binary_sha256": arm["binary"]["sha256"],
                         "artifact": {k: (arm["artifact"] or {}).get(k) for k in ("path", "bytes", "sha256")},
                         "case": plan["arms"]["A"]["case"], "case_sha256": plan["arms"]["A"]["case_sha256"],
                         "harness_commit": manifest["identities"]["harness"]["git_commit"]},
            "rule": "each ceiling is the largest baseline value over eligible measured runs (Pass, Q5 CalibrationPass, "
                    "no serious or critical thermal state) and their slots of the workload; candidate runs and slots "
                    f"must stay at or below {CAP_FACTOR} x the ceiling (absolute safety cap)",
            "cap_factor": CAP_FACTOR,
            "ceilings": stages,
            "retained_failures": [r for r in rows if r["kind"] == "measured" and r["outcome"] != "Pass"],
            "runs": rows, "independent_review_required": True}


def fairness_check(entry, ceiling):
    """Safety-cap problems of one candidate run: every slot's p95 TTFT and the run's maximum SSE gap at or
    below CAP_FACTOR x the frozen ceiling; [] when it complies. 'Unverifiable' when a needed value is missing."""
    if not entry.get("metrics"):
        return [f"{entry['name']}: Unverifiable: no metrics"]
    values, problems = fairness_values(entry["metrics"]), []
    if not values["slots"]:
        return [f"{entry['name']}: Unverifiable: no slots"]
    ttft = CAP_FACTOR * ceiling[TTFT]["value"]
    for slot, slot_values in values["slots"].items():
        if slot_values[TTFT] is None:
            problems.append(f"{entry['name']} {slot}: Unverifiable: no streamed TTFT")
        elif slot_values[TTFT] > ttft:
            problems.append(f"{entry['name']} {slot}: p95 semantic TTFT {slot_values[TTFT]:.4g} s > cap {ttft:.4g} s")
    sse, gap = ceiling[SSE]["value"], values["run_sse_gap_max_s"]
    if sse is None and gap is not None:
        problems.append(f"{entry['name']}: Unverifiable: SSE gap {gap:.4g} s measured but the baseline has no SSE ceiling")
    elif sse is not None and gap is None:
        problems.append(f"{entry['name']}: Unverifiable: no SSE gap measured against ceiling {sse:.4g} s")
    elif sse is not None and gap > CAP_FACTOR * sse:
        problems.append(f"{entry['name']}: max semantic SSE gap {gap:.4g} s > cap {CAP_FACTOR * sse:.4g} s")
    return problems


def paired_fairness(stage):
    """Q3's paired fairness test (q3_report.FAIRNESS, q3_stats.regression): per metric, the block-paired B/A
    effect and whether B is reliably worse beyond the neutral band. ({metric: record}, problems)."""
    measured = list(runs(stage))
    records, problems = {}, []
    for metric in q3_report.FAIRNESS:
        result = q3_report.effect(stage, "B", "A", metric)
        measurable = any((e.get("metrics") or {}).get(metric) is not None for e in measured if e["outcome"] == "Pass")
        regressed = q3_stats.regression(result)
        records[metric] = {"effect": result, "measurable": measurable, "regression": regressed,
                           "neutral_ratio": q3_stats.NEUTRAL_RATIO}
        if regressed:
            problems.append(f"{stage['workload']}: {metric} paired regression: B/A ratio 95% CI "
                            f"[{result['ratio_ci95'][0]:.4f}, {result['ratio_ci95'][1]:.4f}] above {q3_stats.NEUTRAL_RATIO}")
        elif measurable and result["ratio_ci95"] is None:
            problems.append(f"{stage['workload']}: {metric} Unverifiable: {result['blocks']} paired blocks, no interval")
    return records, problems


# --- A/B analysis -------------------------------------------------------------------------------


def analyze(manifest, ceilings_doc, sources):
    """The analysis document of a two-arm A/B (or a labelled pilot, which never has a verdict)."""
    plan = manifest["plan"]
    if plan.get("mode") not in ("ab", "pilot"):
        raise ValueError(f"analysis needs an A/B or pilot campaign (this one is {plan.get('mode')})")
    if ceilings_doc.get("schema") != CEILING_SCHEMA or ceilings_doc["family"] != plan["family"]:
        raise ValueError(f"ceilings must be a {CEILING_SCHEMA} document for family {plan['family']}")
    fixed, gates, stages = plan["design"]["blocks"], [], {}
    for stage in manifest["stages"]:
        key, measured = stage["workload"], list(runs(stage))
        result = effect(stage)
        stage_verdict, reasons = verdict(result, fixed)
        if stage["status"] != "Complete":
            stage_verdict, reasons = "Inconclusive", [f"stage {stage['status']}: {len(stage['blocks'])} of {fixed} blocks"]
        paired, paired_problems = paired_fairness(stage)
        ceiling = ceilings_doc["ceilings"].get(key)
        cap = ([f"{key}: no frozen ceiling for this workload"] if ceiling is None else
               [p for e in measured if e["arm"] == "B" for p in fairness_check(e, ceiling)])
        ineligible = [e["name"] for e in measured if e["outcome"] != "Pass"]
        stages[key] = {"status": stage["status"], "blocks": len(stage["blocks"]), "fixed_blocks": fixed,
                       "effect": result, "verdict": stage_verdict, "reasons": reasons,
                       "paired_fairness": paired, "fairness_problems": paired_problems + cap,
                       "quality_ineligible_runs": ineligible,
                       "q5_non_pass_runs": [e["name"] for e in measured
                                            if e.get("metrics") and not e["metrics"]["memory"]["q5_pass"]],
                       "thermal_invalid_runs": [e["name"] for e in measured
                                                if e.get("metrics") and e["metrics"]["thermal"]["valid"] is False]}
        gates += [f"fairness: {p}" for p in stages[key]["fairness_problems"]]
        gates += [f"{key}: quality-ineligible candidate run {e['name']} ({e['failure_kind']}: {e['reason']})"
                  for e in measured if e["arm"] == "B" and e["outcome"] != "Pass"]
        gates += [f"{key}: run {name} lacks a passing Q5 verdict" for name in stages[key]["q5_non_pass_runs"]]
        gates += [f"{key}: run {name} observed serious or critical thermal state" for name in stages[key]["thermal_invalid_runs"]]
    status = manifest["status"]
    if status != "Complete":
        gates.insert(0, f"campaign {status}, not Complete: {(manifest.get('halted') or {}).get('reason')}")
    controls = {}
    required = plan["n1_control"] == "required"
    if "n1" in stages:
        n1 = stages["n1"]
        ok = n1["status"] == "Complete" and n1["effect"]["blocks"] == fixed and within_margin(n1["effect"])
        controls["n1"] = {"rule": "interval within +-2.5 points over the fixed block count" if required
                          else "reported (the change engages one agent)", "within_margin": ok}
        if required and n1["status"] != "Complete":
            gates.append(f"n1 control stage {n1['status']}, not Complete")
        elif required and n1["effect"]["blocks"] != fixed:
            gates.append(f"n1 control: {n1['effect']['blocks']} of {fixed} blocks quality-eligible")
        elif required and not ok:
            gates.append("n1 control: interval does not lie within +-2.5 points")
    for key in MIXED:
        if key in stages:
            controls[key] = {"rule": "reported", "status": stages[key]["status"],
                             "improvement_ci95": stages[key]["effect"]["improvement_ci95"]}
    if required and "n1" not in stages:
        gates.append("n1 control required but not run")
    primary = stages.get(PRIMARY)
    if plan["mode"] == "pilot":
        result, reasons = "Pilot", ["a pilot is never a verdict and its runs are never pooled"]
    elif status in ("Invalid", "IdentityDrift") or primary is None:
        result, reasons = None, [f"campaign {status}: {(manifest.get('halted') or {}).get('reason')}"]
    else:
        result, reasons = primary["verdict"], primary["reasons"]
    return {"schema": ANALYSIS_SCHEMA, "family": plan["family"], "mode": plan["mode"], "campaign_status": status,
            "sources": sources, "fixed_blocks": fixed, "primary_workload": PRIMARY, "verdict": result,
            "reasons": reasons, "gate_failures": gates, "gates_pass": not gates, "controls": controls,
            "stages": stages, "ceilings_family": ceilings_doc["family"], "runs": retained(manifest),
            "independent_review_required": True}
