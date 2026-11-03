"""ADR-065 two-arm fixed-count A/B analysis and the B1 fairness ceilings. Pure.

Predeclared in docs/research/adr065-b1-campaign.md and scripts/four_agent/adr065/campaign-b1.json
(ADR-065 "Measurement and acceptance rules"); change them together, before data.

Inputs are campaign.json manifests written by adr065_campaign.py (the q3_campaign.Driver layout: stages,
warmups, blocks, entries with outcome and q3_metrics.run_metrics). Arm A is the previous default, arm B the
candidate; the effect is the reduction of B's makespan against A's.
"""
import q3_report

DEFAULT_BLOCKS = 6
ALLOWED_BLOCKS = (6, 8, 12, 16, 20)
MARGIN_PP = 2.5
VERDICTS = ("Loss", "Win", "Equivalent", "Inconclusive")
PRIMARY = "n4"
MIXED = ("3+1", "2+2", "1+3")
CEILING_SCHEMA = "hf2q.adr065.fairness-ceilings.v1"
ANALYSIS_SCHEMA = "hf2q.adr065.ab-analysis.v1"
TTFT, SSE = "ttft_p95_s", "sse_gap_max_s"  # q3_metrics.slot_metrics keys (ADR-063 F2 estimands)


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


def stage_ceilings(stage):
    """Largest per-slot p95 TTFT and largest per-run maximum SSE gap over the stage's passing measured runs.
    A metric no passing run measured (one semantic event per streamed turn) has value None."""
    best = {TTFT: None, SSE: None}
    for entry in runs(stage):
        if entry["outcome"] != "Pass":
            continue
        values = fairness_values(entry["metrics"])
        for slot, slot_values in values["slots"].items():
            for key in (TTFT, SSE):
                value = slot_values[key]
                if value is not None and (best[key] is None or value > best[key]["value"]):
                    best[key] = {"value": value, "run": entry["name"], "slot": slot}
    passed = sum(entry["outcome"] == "Pass" for entry in runs(stage))
    return {"passing_runs": passed, "measured_runs": sum(1 for _ in runs(stage)),
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
        raise ValueError("no passing measured run defines a TTFT ceiling for " + ", ".join(empty))
    rows = retained(manifest)
    arm = manifest["identities"]["arms"]["A"]
    return {"schema": CEILING_SCHEMA, "family": plan["family"], "sources": sources,
            "baseline": {**plan["baseline"], "binary_sha256": arm["binary"]["sha256"],
                         "artifact": {k: (arm["artifact"] or {}).get(k) for k in ("path", "bytes", "sha256")},
                         "case": plan["arms"]["A"]["case"], "case_sha256": plan["arms"]["A"]["case_sha256"],
                         "harness_commit": manifest["identities"]["harness"]["git_commit"]},
            "rule": "each ceiling is the largest baseline value over passing measured runs (and their slots) of the "
                    "workload; every measured candidate run and slot must stay at or below both",
            "ceilings": stages,
            "retained_failures": [r for r in rows if r["kind"] == "measured" and r["outcome"] != "Pass"],
            "runs": rows, "independent_review_required": True}


def fairness_check(entry, ceiling):
    """Problems of one candidate run against its workload's ceilings; [] when it complies.
    'Unverifiable' when a value or ceiling needed for the comparison is missing."""
    if not entry.get("metrics"):
        return [f"{entry['name']}: Unverifiable: no metrics"]
    values, problems = fairness_values(entry["metrics"]), []
    if not values["slots"]:
        return [f"{entry['name']}: Unverifiable: no slots"]
    ttft = ceiling[TTFT]["value"]
    for slot, slot_values in values["slots"].items():
        if slot_values[TTFT] is None:
            problems.append(f"{entry['name']} {slot}: Unverifiable: no streamed TTFT")
        elif slot_values[TTFT] > ttft:
            problems.append(f"{entry['name']} {slot}: p95 semantic TTFT {slot_values[TTFT]:.4g} s > ceiling {ttft:.4g} s")
    sse, gap = ceiling[SSE]["value"], values["run_sse_gap_max_s"]
    if sse is None and gap is not None:
        problems.append(f"{entry['name']}: Unverifiable: SSE gap {gap:.4g} s measured but the baseline has no SSE ceiling")
    elif sse is not None and gap is None:
        problems.append(f"{entry['name']}: Unverifiable: no SSE gap measured against ceiling {sse:.4g} s")
    elif sse is not None and gap > sse:
        problems.append(f"{entry['name']}: max semantic SSE gap {gap:.4g} s > ceiling {sse:.4g} s")
    return problems


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
        ceiling = ceilings_doc["ceilings"].get(key)
        fairness = ([f"{key}: no frozen ceiling for this workload"] if ceiling is None else
                    [p for e in measured if e["arm"] == "B" for p in fairness_check(e, ceiling)])
        ineligible = [e["name"] for e in measured if e["outcome"] != "Pass"]
        stages[key] = {"status": stage["status"], "blocks": len(stage["blocks"]), "fixed_blocks": fixed,
                       "effect": result, "verdict": stage_verdict, "reasons": reasons,
                       "fairness_problems": fairness, "quality_ineligible_runs": ineligible,
                       "q5_non_pass_runs": [e["name"] for e in measured
                                            if e.get("metrics") and not e["metrics"]["memory"]["q5_pass"]],
                       "thermal_invalid_runs": [e["name"] for e in measured
                                                if e.get("metrics") and e["metrics"]["thermal"]["valid"] is False]}
        gates += [f"fairness: {p}" for p in fairness]
        gates += [f"{key}: quality-ineligible candidate run {e['name']} ({e['failure_kind']}: {e['reason']})"
                  for e in measured if e["arm"] == "B" and e["outcome"] != "Pass"]
        gates += [f"{key}: run {name} lacks a passing Q5 verdict" for name in stages[key]["q5_non_pass_runs"]]
        gates += [f"{key}: run {name} observed serious or critical thermal state" for name in stages[key]["thermal_invalid_runs"]]
    controls = {}
    if "n1" in stages:
        required = plan["n1_control"] == "required"
        ok = within_margin(stages["n1"]["effect"])
        controls["n1"] = {"rule": "interval within +-2.5 points" if required else "reported (the change engages one agent)",
                          "within_margin": ok}
        if required and not ok:
            gates.append("n1 control: interval does not lie within +-2.5 points")
    for key in MIXED:
        if key in stages:
            controls[key] = {"rule": "reported", "improvement_ci95": stages[key]["effect"]["improvement_ci95"]}
    if plan["n1_control"] == "required" and "n1" not in stages:
        gates.append("n1 control required but not run")
    primary = stages.get(PRIMARY)
    status = manifest["status"]
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
