"""ADR-063 Q3 effects, family verdict.json and F2 scoreboard rows from a campaign.json. No launches."""
import hashlib
import json
import statistics
from pathlib import Path

import q3_metrics
import q3_stats

VERDICT_SCHEMA = "hf2q.adr063.q3-verdict.v1"
PATCH_REVIEW_SCHEMA = "hf2q.adr063.q3-patch-review.v1"
PRIMARY_METRIC = "makespan_s"
# Secondary paired effects (metric -> higher is better). Fairness metrics may not regress vs stock.
SECONDARY = {"goodput_tasks_per_hour": True, "client_elapsed_s": False,
             "worst_slot_ttft_p95_s": False, "worst_slot_ttft_max_s": False,
             "worst_slot_sse_gap_p95_s": False, "worst_slot_sse_gap_max_s": False}
FAIRNESS = ("worst_slot_ttft_p95_s", "worst_slot_sse_gap_max_s")
CONTRASTS = (("vs_stock", "stock"), ("vs_peer", "peer"), ("diagnostic_vs_peer_matched", "peer_matched"))
SETTLED = ("Pass", "Fail", "Unavailable", "Skipped")
F2_STATUS = {"Win": "Descriptive", "Inconclusive": "Descriptive", "Unavailable": "Descriptive",
             "Loss": "NegativeResult", "Failed": "Failure"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entries(manifest, arm=None, kind=None, stages=None):
    for stage in manifest["stages"]:
        if stages is not None and stage["workload"] not in stages:
            continue
        for entry in stage["warmups"] + [e for block in stage["blocks"] for e in block["entries"]]:
            if (arm is None or entry["arm"] == arm) and (kind is None or entry["kind"] == kind):
                yield entry


def block_complete(block):
    return all(entry["outcome"] in SETTLED for entry in block["entries"])


def passing_value(block, arm, key):
    for entry in block["entries"]:
        if entry["arm"] == arm and entry["outcome"] == "Pass":
            value = (entry.get("metrics") or {}).get(key)
            return value if isinstance(value, (int, float)) and value > 0 else None
    return None


def effect(stage, subject, reference, key, higher_is_better=False):
    pairs = []
    for block in stage["blocks"]:
        if block_complete(block):
            a, b = passing_value(block, subject, key), passing_value(block, reference, key)
            if a is not None and b is not None:
                pairs.append((a, b))
    return q3_stats.paired_effect(pairs, higher_is_better)


def stage_effects(stage, arms):
    """Experiment-vs-arm effects on makespan (primary) and the secondary metrics."""
    present = [(name, arm) for name, arm in CONTRASTS if arm in arms]
    return {"primary_metric": PRIMARY_METRIC,
            "effects": {name: effect(stage, "experiment", arm, PRIMARY_METRIC) for name, arm in present},
            "secondary": {metric: {name: effect(stage, "experiment", arm, metric, higher)
                                   for name, arm in present} for metric, higher in SECONDARY.items()}}


def required_effects(stage, manifest):
    """Contrasts the stopping rule waits for: stock always, the peer while it is available and has not failed."""
    effects = stage_effects(stage, manifest["plan"]["measured_arms"])["effects"]
    required = {"vs_stock": effects["vs_stock"]}
    if "vs_peer" in effects and "peer" not in manifest["unavailable"] and not failures(manifest, "peer"):
        required["vs_peer"] = effects["vs_peer"]
    return required


def failures(manifest, arm):
    return [f"{e['name']}: {e['failure_kind']}: {e['reason']}" for e in entries(manifest, arm) if e["outcome"] == "Fail"]


def arm_summary(stage, arm):
    runs = [e for block in stage["blocks"] for e in block["entries"] if e["arm"] == arm]
    passed = [e["metrics"] for e in runs if e["outcome"] == "Pass"]
    observed = [e["metrics"] for e in runs if e.get("metrics")]
    def median(key):
        values = [m[key] for m in passed if m.get(key) is not None]
        return statistics.median(values) if values else None

    pressure = {}
    for m in observed:
        for level, count in (m["memory"]["pressure_counts"] or {}).items():
            pressure[level] = pressure.get(level, 0) + count
    floor = [m["memory"]["min_free_plus_cached_bytes"] for m in observed if m["memory"]["min_free_plus_cached_bytes"] is not None]
    swap = [m["memory"]["swap_growth_bytes"] for m in observed if m["memory"]["swap_growth_bytes"] is not None]
    states = [s for m in observed for s in (m["thermal"]["preflight_state"], m["thermal"]["post_run_state"]) if s is not None]
    return {"runs": len(runs), "outcomes": {o: sum(e["outcome"] == o for e in runs) for o in sorted({e["outcome"] for e in runs if e["outcome"]})},
            "makespan_median_s": median("makespan_s"), "goodput_median_tasks_per_hour": median("goodput_tasks_per_hour"),
            "worst_slot_ttft_p95_median_s": median("worst_slot_ttft_p95_s"),
            "worst_slot_sse_gap_max_median_s": median("worst_slot_sse_gap_max_s"),
            "memory": {"min_free_plus_cached_bytes": min(floor, default=None), "max_swap_growth_bytes": max(swap, default=None),
                       "pressure_sample_counts": pressure,
                       "q5_non_pass_runs": [e["name"] for e in runs if e.get("metrics") and not e["metrics"]["memory"]["q5_pass"]]},
            "thermal": {"max_observed_state": max(states, default=None), "in_run": "unrecorded"}}


def quality_state(manifest, quality_paths, review_path):
    """Q1 paired verdicts (quality_campaign.py manifests) bound to the experiment binary, and an optional
    independent patch review; without them, task-completion parity (every experiment run completed)."""
    experiment_sha = manifest["identities"]["arms"]["experiment"]["binary"]["sha256"]
    inputs, status, reasons = [], "parity", []
    for path in quality_paths:
        document = q3_metrics.load_json(path)
        if document is None:
            raise ValueError(f"unreadable quality verdict {path}")
        subject = {(q3_metrics.load_json(Path(a["output"]) / "receipt.json") or {}).get("binary", {}).get("sha256")
                   for run in document.get("runs", []) if run.get("arm") == "subject" for a in run.get("attempts", [])}
        family = document.get("plan", {}).get("arms", {}).get("subject", {}).get("family")
        verdict = (document.get("verdict") or {}).get("verdict")
        record = {"path": str(path), "sha256": sha256(path), "family": family, "verdict": verdict,
                  "reference_kind": document.get("plan", {}).get("reference_kind"),
                  "binds_experiment_binary": subject == {experiment_sha}}
        inputs.append(record)
        if family != manifest["plan"]["family"]:
            raise ValueError(f"quality verdict {path} is for family {family}")
        if not record["binds_experiment_binary"]:
            status, reasons = ("unresolved" if status != "loss" else status), reasons + [f"{path} was not run on the experiment binary"]
        elif verdict in ("Loss", "BelowFloor"):
            status, reasons = "loss", reasons + [f"Q1 paired verdict {verdict} ({path})"]
        elif verdict != "Pass":
            status, reasons = ("unresolved" if status != "loss" else status), reasons + [f"Q1 paired verdict {verdict} ({path})"]
        elif status == "parity":
            status = "pass"
    hashes = sorted({h for e in entries(manifest, "experiment") if e["outcome"] == "Pass"
                     for h in (e["metrics"]["final_source_sha256"] or {}).values() if h})
    review = {"file": None, "reviewed": [], "pending": hashes}
    if review_path is not None:
        document = q3_metrics.load_json(review_path)
        if not isinstance(document, dict) or document.get("schema") != PATCH_REVIEW_SCHEMA or not isinstance(document.get("verdicts"), dict):
            raise ValueError(f"patch review must be {PATCH_REVIEW_SCHEMA} with a verdicts object")
        verdicts = document["verdicts"]
        rejected = [h for h in hashes if verdicts.get(h) == "Fail"]
        review = {"file": str(review_path), "sha256": sha256(review_path),
                  "reviewed": [h for h in hashes if verdicts.get(h) in ("Pass", "Fail")],
                  "pending": [h for h in hashes if verdicts.get(h) not in ("Pass", "Fail")], "rejected": rejected}
        if rejected:
            status, reasons = "loss", reasons + ["independent patch review rejected final source " + ", ".join(rejected)]
    return {"status": status, "reason": "; ".join(reasons) or None, "q1_paired": inputs, "patch_review": review}


def compute_verdict(manifest, quality_paths=(), review_path=None, campaign_sha256=None):
    plan = manifest["plan"]
    stages = {stage["workload"]: stage for stage in manifest["stages"]}
    primary = stages[plan["primary"]]
    report = {name: {"phase_plan": stage["phase_plan"], "status": stage["status"], "stop": stage["stop"],
                     "blocks": len(stage["blocks"]), **stage_effects(stage, plan["measured_arms"]),
                     "arms": {arm: arm_summary(stage, arm) for arm in plan["measured_arms"]}}
              for name, stage in stages.items()}
    effects = report[plan["primary"]]["effects"]
    empty = q3_stats.paired_effect([])
    quality = quality_state(manifest, quality_paths, review_path)
    measured = [e for e in entries(manifest, kind="measured") if e.get("metrics")]
    thermal = [e["metrics"]["thermal"]["valid"] for e in measured]
    state = {"experiment_failures": failures(manifest, "experiment"),
             "peer_unavailable": manifest["unavailable"].get("peer"),
             "peer_failures": failures(manifest, "peer"), "quality": quality,
             "complete": primary["stop"] is not None,
             "vs_stock": effects["vs_stock"], "vs_peer": effects.get("vs_peer", empty),
             "fairness": {m: report[plan["primary"]]["secondary"][m]["vs_stock"] for m in FAIRNESS},
             "thermal_valid": False if False in thermal else (True if thermal and None not in thermal else None),
             "memory_valid": all(e["metrics"]["memory"]["q5_pass"] for e in measured
                                 if e["arm"] == "experiment" and e["outcome"] == "Pass")}
    verdict, reasons = q3_stats.family_verdict(state)
    return {"schema": VERDICT_SCHEMA, "family": plan["family"], "campaign_sha256": campaign_sha256,
            "identities_sha256": manifest["identities_sha256"], "campaign_status": manifest["status"],
            "verdict": verdict, "reasons": reasons, "primary_workload": plan["primary"],
            "primary": report[plan["primary"]],
            "controls": {k: v for k, v in report.items() if k != plan["primary"]},
            "peer": {"unavailable": manifest["unavailable"], "failures": state["peer_failures"]},
            "stock_failures": failures(manifest, "stock"), "experiment_failures": state["experiment_failures"],
            "quality": quality, "thermal_valid": state["thermal_valid"], "memory_valid": state["memory_valid"],
            "rejected_trials": [{"name": e["name"], "outcome": e["outcome"], "reason": e.get("reason")}
                                for e in entries(manifest) if e["outcome"] in ("Invalid", "Abandoned", "Skipped")
                                or any(a["outcome"] == "NotStarted" for a in e["attempts"])],
            "warmups_discarded": sum(1 for _ in entries(manifest, kind="warmup")),
            "design": plan["design"], "declared": plan["declared"],
            "independent_review_required": True}


def rows(campaign_dir, evidence_root):
    """F2 scoreboard rows (hf2q.f2.scoreboard-rows.v1 vocabulary) for the primary effects of one campaign."""
    campaign_dir, root = Path(campaign_dir).resolve(), Path(evidence_root).resolve()
    if not campaign_dir.is_relative_to(root):
        raise ValueError("campaign directory must be under --evidence-root to be cited as $E/...")
    manifest = json.loads((campaign_dir / "campaign.json").read_text())
    verdict = json.loads((campaign_dir / "verdict.json").read_text())
    family, plan = manifest["plan"]["family"], manifest["plan"]
    arms = manifest["identities"]["arms"]
    ref = "$E/" + str((campaign_dir / "verdict.json").relative_to(root))
    group = f"{family}-q3"
    out = {"groups": {group: {"family": family, "title": f"{family} Q3 three-arm campaign",
                              "workload": "Four-agent coding client, 262144 context, 4 slots; stock "
                                          f"{plan['declared']['stock']['label']} vs experiment "
                                          f"{plan['declared']['experiment']['label']} vs the peer, Williams blocks."}},
           "rows": []}
    status = F2_STATUS.get(verdict["verdict"])
    if status is None:
        return out
    if not arms["experiment"]["artifact"] or not arms["experiment"]["artifact"].get("sha256"):
        raise ValueError("F2 rows need the model sha256 (conversion receipt or HF2Q_Q5_MODEL_SHA256)")
    for name, baseline in (("vs_stock", "hf2q_current"), ("vs_peer", "peer")):
        e = verdict["primary"]["effects"].get(name)
        if e is None or e["improvement_percent"] is None or (name == "vs_peer" and verdict["verdict"] == "Unavailable"):
            continue
        path = f"primary.effects.{name}"
        out["rows"].append({
            "id": f"{group}-makespan-{name.replace('_', '-')}", "family": family, "group": group,
            "metric": "makespan_s", "kind": "effect", "arm": "candidate",
            "variant": f"Q3 experiment {plan['declared']['experiment']['label']}",
            "value": round(e["improvement_percent"], 3), "decimals": 3, "unit": "% improvement",
            "ci95": [round(x, 3) for x in e["improvement_ci95"]], "baseline_arm": baseline,
            "n_slots": 4, "context": "262144", "date": manifest["created_utc"][:10],
            "source": {"commit": plan["declared"]["experiment"]["commit"], "binary_sha256": arms["experiment"]["binary"]["sha256"]},
            "artifact_sha256": arms["experiment"]["artifact"]["sha256"], "receipt": ref, "status": status,
            "verification": {"state": "Verified", "key": f"verdict.json {path}.improvement_percent",
                             "extract": {"value": {"kind": "json", "file": ref, "path": f"{path}.improvement_percent", "reduce": "one"},
                                         "ci95": {"kind": "json", "file": ref, "path": f"{path}.improvement_ci95.*", "reduce": "list"}}},
            "note": f"Q3 verdict {verdict['verdict']}; {e['blocks']} paired blocks, half-width {e['half_width_pp']:.2f} points "
                    f"(baseline {'stock ' + plan['declared']['stock']['label'] if name == 'vs_stock' else 'peer best valid configuration'})."})
    return out
