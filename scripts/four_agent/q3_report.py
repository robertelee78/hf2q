"""ADR-063 Q3 effects, family verdict.json and F2 scoreboard rows from a campaign.json. No launches."""
import hashlib
import json
import re
import statistics
from pathlib import Path

import f2_q3_tier
import native_peers
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
# Controls-only campaign (#369 amendment): no primary verdict; the primary is of record in another campaign.
CONTROLS_ONLY = "ControlsOnly"
Q2_RUN_AUDIT = "scripts/four_agent/q2_run_audit.py"


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


def primary_of_record(manifest, record):
    """verdict.json `primary_of_record`: the bound primary verdict, its effects, and both harness source maps."""
    path = Path(record["verdict"]["path"])
    for bound in (record["verdict"], record["campaign"]):
        if sha256(bound["path"]) != bound["sha256"]:
            raise ValueError(f"primary of record {bound['path']} changed since the campaign bound it")
    then, now = record["harness"]["sources"], manifest["identities"]["harness"]["sources"]
    return {"path": str(path), "sha256": record["verdict"]["sha256"], "verdict": record["verdict_of_record"],
            "primary_workload": record["primary_workload"], "effects": record["effects"],
            "campaign": record["campaign"], "identities_sha256": record["identities_sha256"], "identity": record["identity"],
            "harness": {"of_record": record["harness"],
                        "this_campaign": {"git_commit": manifest["identities"]["harness"]["git_commit"], "sources": now},
                        "differs": sorted(k for k in set(then) | set(now) if then.get(k) != now.get(k))},
            "q2_run_audit_sha256": {"of_record": then.get(Q2_RUN_AUDIT), "this_campaign": now.get(Q2_RUN_AUDIT)}}


def controls_only_reasons(manifest, record, state):
    reasons = [f"controls only: the {record['primary_workload']} primary verdict is of record in "
               f"{record['verdict']['path']} ({record['verdict_of_record']}); no primary verdict is computed (#369 amendment)"]
    if manifest["status"] != "Complete":
        reasons.append(f"campaign {manifest['status']}: {(manifest.get('halted') or {}).get('reason')}")
    reasons += [f"experiment failed its task: {f}" for f in state["experiment_failures"]]
    reasons += [f"peer failed its task: {f}" for f in state["peer_failures"]]
    if state["peer_unavailable"]:
        reasons.append(f"peer unavailable: {state['peer_unavailable']['reason']}")
    if state["quality"]["status"] == "loss":
        reasons.append(f"quality loss: {state['quality']['reason']}")
    return reasons


def stopped_experiment_failures(manifest):
    """(workload, campaign.json path, entry) of each failed experiment run in a Stopped stage."""
    for i, stage in enumerate(manifest["stages"]):
        if stage["status"] != "Stopped":
            continue
        located = [(f"stages.{i}.warmups.{j}", e) for j, e in enumerate(stage["warmups"])]
        located += [(f"stages.{i}.blocks.{b}.entries.{j}", e) for b, block in enumerate(stage["blocks"])
                    for j, e in enumerate(block["entries"])]
        for path, entry in located:
            if entry["arm"] == "experiment" and entry["outcome"] == "Fail":
                yield stage["workload"], path, entry


def compute_verdict(manifest, quality_paths=(), review_path=None, campaign_sha256=None):
    plan = manifest["plan"]
    record = plan.get("controls_of")  # controls-only: the primary is of record in another campaign
    stages = {stage["workload"]: stage for stage in manifest["stages"]}
    primary = None if record else stages[plan["primary"]]
    report = {name: {"phase_plan": stage["phase_plan"], "status": stage["status"], "stop": stage["stop"],
                     "blocks": len(stage["blocks"]), **stage_effects(stage, plan["measured_arms"]),
                     "arms": {arm: arm_summary(stage, arm) for arm in plan["measured_arms"]}}
              for name, stage in stages.items()}
    quality = quality_state(manifest, quality_paths, review_path)
    measured = [e for e in entries(manifest, kind="measured") if e.get("metrics")]
    thermal = [e["metrics"]["thermal"]["valid"] for e in measured]
    state = {"experiment_failures": failures(manifest, "experiment"),
             "peer_unavailable": manifest["unavailable"].get("peer"),
             "peer_failures": failures(manifest, "peer"), "quality": quality,
             "thermal_valid": False if False in thermal else (True if thermal and None not in thermal else None),
             "memory_valid": all(e["metrics"]["memory"]["q5_pass"] for e in measured
                                 if e["arm"] == "experiment" and e["outcome"] == "Pass")}
    native = native_peers.peer_kind(plan)
    comparison = plan.get("native_comparison")
    if record:
        verdict, reasons = CONTROLS_ONLY, controls_only_reasons(manifest, record, state)
    else:
        effects = report[plan["primary"]]["effects"]
        state.update(complete=primary["stop"] is not None, vs_stock=effects["vs_stock"],
                     vs_peer=effects.get("vs_peer", q3_stats.paired_effect([])),
                     fairness={m: report[plan["primary"]]["secondary"][m]["vs_stock"] for m in FAIRNESS})
        if native:  # the native arm never yields the same-GGUF verdict; it is reported separately as native_peer
            state["peer_unavailable"] = {"reason": comparison or native_peers.adapter(native).SAME_GGUF_UNAVAILABLE,
                                         "evidence": plan["peer_unavailable_evidence"]}
        verdict, reasons = q3_stats.family_verdict(state)
        if comparison and verdict == "Unavailable":
            reasons = [comparison]
    # Default mode: a control stage that halts the campaign (Invalid, IdentityDrift) does not replace the
    # completed primary's verdict; campaign_status and the halt record travel with it into every row note.
    result = {"schema": VERDICT_SCHEMA, "family": plan["family"], "campaign_sha256": campaign_sha256,
            "identities_sha256": manifest["identities_sha256"], "campaign_status": manifest["status"],
            "halted": manifest.get("halted"),
            "verdict": verdict, "reasons": reasons, "primary_workload": plan["primary"],
            "primary": None if record else report[plan["primary"]],
            **({"primary_of_record": primary_of_record(manifest, record)} if record else {}),
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
    if native:
        audit = native_peers.audit(native)
        label = f2_q3_tier.experiment_name(plan["declared"]["experiment"])
        result["native_peer"] = ({key: audit.q3_section(stage, manifest, label) for key, stage in stages.items()}
                                 if record else audit.q3_section(primary, manifest, label))
    if comparison:
        result.update(campaign_label=f2_q3_tier.NATIVE_COMPARISON, verdict_of_record=False)
    return result


def cite(path, root, what):
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{what} must be under --evidence-root to be cited as $E/...")
    return "$E/" + str(path.relative_to(root))


def record_inputs(root, quality_gate, review):
    """(reference kind, `$E/` manifest ref, manifest or None while not written) per quality gate, and the review ref."""
    gates = []
    for reference, path in quality_gate:
        if reference not in f2_q3_tier.REFERENCES:
            raise ValueError(f"unknown quality gate reference {reference!r}; one of {sorted(f2_q3_tier.REFERENCES)}")
        gates.append((reference, cite(path, root, "a quality gate manifest"),
                      json.loads(Path(path).read_text()) if Path(path).is_file() else None))
    if review is not None and not Path(review).is_file():
        raise ValueError(f"review {review} does not exist")
    return gates, None if review is None else cite(review, root, "the independent review")


def _usage(ref, kind):
    return {"kind": "json", "file": ref, "path": f"agents.*.turns.*.usage.{kind}_tokens", "reduce": "sum", "skip_null": True}


def arm_runs(stage, arm, root):
    """(`$E/` summary ref, token volume, makespan, block index) of each passing measured run of one arm, each re-read
    from the run's client/summary.json; [] when any of them lacks usage."""
    runs = []
    for b, block in enumerate(stage["blocks"]):
        for entry in (e for e in block["entries"] if e["arm"] == arm and e["outcome"] == "Pass"):
            path = Path(entry["attempts"][-1]["output"]).resolve() / "client" / "summary.json"
            summary = json.loads(path.read_text())
            volume = q3_metrics.token_volume(summary.get("agents") or {})
            makespan = summary.get("own_output_four_task_makespan_seconds")
            if volume is None or not makespan:
                return []
            runs.append((cite(path, root, "a run summary"), volume, makespan, b))
    return runs


def token_rows(manifest, root, base, status, campaign_ref, experiment):
    """Per arm: median prompt and completion tokens and median completion tokens per makespan-second over the
    primary stage's passing runs, each re-read from the run's client/summary.json. An arm with a run that lacks
    usage gets no rows (the tier shows 'not recorded')."""
    plan, arms = manifest["plan"], manifest["identities"]["arms"]
    stage = next(s for s in manifest["stages"] if s["workload"] == plan["primary"])
    out = []
    for arm, f2arm in (("experiment", "candidate"), ("stock", "v0128_stock"), ("peer", "peer")):
        runs = [run[:3] for run in arm_runs(stage, arm, root)]
        if not runs:
            continue
        rate = [{"kind": "ratio", "numerator": _usage(r, "completion"), "denominator": {
            "kind": "json", "file": r, "path": "own_output_four_task_makespan_seconds", "reduce": "one"}} for r, _, _ in runs]
        specs = {"prompt_tokens": ([v["prompt"] for _, v, _ in runs], [_usage(r, "prompt") for r, _, _ in runs], 1, "tokens"),
                 "completion_tokens": ([v["completion"] for _, v, _ in runs], [_usage(r, "completion") for r, _, _ in runs], 1, "tokens"),
                 "output_tokens_per_s": ([v["completion"] / m for _, v, m in runs], rate, 3, "tok/s")}
        declared = plan["declared"].get(arm) or {}
        for metric, (values, items, decimals, unit) in specs.items():
            out.append({"id": f"{base['group']}-{metric.replace('_', '-')}-{f2arm.replace('_', '-')}", **base,
                        "metric": metric, "kind": "measurement", "arm": f2arm,
                        "variant": f"Q3 {arm}" + (f" {experiment}" if arm == "experiment" else f" {declared['label']}" if declared else ""),
                        "value": round(statistics.median(values), decimals), "decimals": decimals, "unit": unit,
                        "source": {"commit": declared.get("commit"), "binary_sha256": arms[arm]["binary"]["sha256"]},
                        "artifact_sha256": arms[arm]["artifact"]["sha256"], "receipt": campaign_ref, "status": status,
                        "verification": {"state": "Verified", "key": f"median over the {len(items)} passing {arm} runs "
                                         f"of {plan['primary']} client/summary.json usage",
                                         "extract": {"value": {"kind": "median_of", "items": items}}},
                        "note": "Descriptive companion of the Q3 verdict of record; never a verdict input."})
    return out


def _whole(x):
    """A token count; medians of an even number of runs end in .5 and round half up."""
    return f"{int(x + 0.5):,}"


def peer_sentence(stage, report, root, experiment):
    """The operator's peer wording for one stage's same-GGUF peer contrast: task makespan on the slower arm's base
    (X = 1 - faster/slower), then peer/hf2q completion and prompt token ratios and the per-output-token rate ratio
    (paired over blocks, with per-arm medians), and hf2q's worst-slot p95 TTFT relative to the peer's. Never a win."""
    effect, arms = report["effects"]["vs_peer"], report["arms"]
    r, (low, high) = effect["ratio"], effect["ratio_ci95"]
    mine, theirs = arms["experiment"]["makespan_median_s"], arms["peer"]["makespan_median_s"]
    if r <= 1:
        text = (f"{experiment} finishes the same coding task {100 * (1 - r):.1f}% sooner than the peer (95% CI "
                f"{100 * (1 - high):.1f}–{100 * (1 - low):.1f}%; median {mine:.1f} s vs {theirs:.1f} s).")
    else:
        text = (f"The peer finishes the same coding task {100 * (1 - 1 / r):.1f}% sooner than {experiment} (95% CI "
                f"{100 * (1 - 1 / low):.1f}–{100 * (1 - 1 / high):.1f}%; median {theirs:.1f} s vs {mine:.1f} s).")
    runs = {arm: {b: (v, m) for _, v, m, b in arm_runs(stage, arm, root)} for arm in ("experiment", "peer")}
    blocks = sorted(set(runs["experiment"]) & set(runs["peer"]))
    if len(blocks) >= 2:
        measures = {"completion": lambda v, m: v["completion"], "prompt": lambda v, m: v["prompt"],
                    "rate": lambda v, m: v["completion"] / m}
        paired = {k: q3_stats.paired_effect([(f(*runs["peer"][b]), f(*runs["experiment"][b])) for b in blocks], True)
                  for k, f in measures.items()}
        median = {(arm, k): statistics.median(f(*runs[arm][b]) for b in blocks)
                  for arm in ("experiment", "peer") for k, f in measures.items()}
        rate = paired["rate"]
        text += (f" The peer generated {paired['completion']['ratio']:.2f}× the completion tokens "
                 f"({_whole(median['peer', 'completion'])} vs {_whole(median['experiment', 'completion'])} per run) and "
                 f"{paired['prompt']['ratio']:.2f}× the prompt tokens ({_whole(median['peer', 'prompt'])} vs "
                 f"{_whole(median['experiment', 'prompt'])}). The peer's per-output-token rate was {rate['ratio']:.2f}× "
                 f"hf2q's ({median['peer', 'rate']:.1f} vs {median['experiment', 'rate']:.1f} completion tokens per "
                 f"makespan-second; CI {rate['ratio_ci95'][0]:.2f}–{rate['ratio_ci95'][1]:.2f}×).")
    ttft = (report["secondary"].get("worst_slot_ttft_p95_s") or {}).get("vs_peer") or {}
    if ttft.get("ratio") is not None:
        text += (f" hf2q's worst-slot p95 TTFT was {ttft['ratio']:.2f}× the peer's "
                 f"({arms['experiment']['worst_slot_ttft_p95_median_s']:.3g} s vs "
                 f"{arms['peer']['worst_slot_ttft_p95_median_s']:.3g} s; "
                 f"CI {ttft['ratio_ci95'][0]:.2f}–{ttft['ratio_ci95'][1]:.2f}×).")
    return text


def failure_detail(entry):
    """The agents that failed an experiment run on their own (not those the coordinator stopped afterwards), each
    with its error and the target and compiler error code of its last failed run_tests (client/summary.json)."""
    if not entry["attempts"]:
        return ""
    path = Path(entry["attempts"][-1]["output"]) / "client" / "summary.json"
    if not path.is_file():
        return ""
    parts = []
    for name, agent in sorted((json.loads(path.read_text()).get("agents") or {}).items()):
        error = agent.get("error") or ""
        if agent.get("status") != "Failed" or not error or error.startswith("KeyboardInterrupt"):
            continue
        failed = [json.loads(t["result"]) for t in agent.get("tools") or []
                  if t.get("name") == "run_tests" and isinstance(t.get("result"), str)
                  and json.loads(t["result"]).get("passed") is False]
        tests = ""
        if failed:
            code = re.search(r"error\[(E\d{4})\]", failed[-1].get("output") or "")
            tests = f" run_tests {failed[-1].get('target')} failed" + (f" with {code.group(1)}" if code else "") + ", then"
        parts.append(f"{name}:{tests} {error}")
    return "; ".join(parts)


def campaign_note(verdict, manifest, root):
    """'campaign <status>', with the halted stage and entry when a stage halted the campaign."""
    halt = f2_q3_tier.halt_record(manifest, root)
    return f"campaign {verdict['campaign_status']}" + (
        "" if halt is None else f": halted at {halt['stage'] or 'campaign'} entry {halt['entry']} ({halt['reason']})")


def rows(campaign_dir, evidence_root, of_record=False, quality_gate=(), review=None):
    """F2 scoreboard rows (hf2q.f2.scoreboard-rows.v1 vocabulary) for the primary effects of one campaign; for a
    controls-only campaign, the completed control stages' effects (status Descriptive) and each failed experiment
    run of a stopped control stage (task_pass, status Failure), all labeled `control_of` the primary of record:
    never a win or loss.

    of_record (a reviewed verdict.json of record, F2 option B): group `<family>-q3-record` carries the
    f2_q3_tier record (verdict, campaign status, halt and stages, quality gate per (reference, manifest) of
    quality_gate bound to the campaign's arm identities, pending patch reviews, the review). Its primary rows are
    flagged `q3_of_record`: makespan and worst-slot fairness effects vs stock and peer with their half-widths, plus
    per-arm token volume and output-rate rows. Its completed control stages (n1, mixed) are Descriptive rows with
    `control_of` this verdict, rendered in the Q3 controls section."""
    campaign_dir, root = Path(campaign_dir).resolve(), Path(evidence_root).resolve()
    if not campaign_dir.is_relative_to(root):
        raise ValueError("campaign directory must be under --evidence-root to be cited as $E/...")
    manifest = json.loads((campaign_dir / "campaign.json").read_text())
    verdict = json.loads((campaign_dir / "verdict.json").read_text())
    family, plan = manifest["plan"]["family"], manifest["plan"]
    arms = manifest["identities"]["arms"]
    peer_arm = plan["arms"].get("peer", {})
    native = native_peers.peer_kind(plan)
    native_label = peer_arm.get("label") if native else None
    comparison = f2_q3_tier.NATIVE_COMPARISON if plan.get("native_comparison") else None
    peer_note = (f"{native_label}; different artifact, task makespan only, not the same-GGUF peer verdict"
                 if native_label else "peer best valid configuration")
    if native_label:
        native_note = native_peers.audit(native).note
    ref = "$E/" + str((campaign_dir / "verdict.json").relative_to(root))
    campaign_ref = "$E/" + str((campaign_dir / "campaign.json").relative_to(root))
    record = verdict.get("primary_of_record")
    if of_record and (record is not None or comparison or native):
        raise ValueError("--of-record needs a default same-GGUF campaign: controls-only, native-comparison and "
                         "native-peer campaigns are never the Q3 verdict of record (their rows keep their own sections)")
    if record is not None:
        of_record_path = Path(record["path"]).resolve()
        if not of_record_path.is_relative_to(root):
            raise ValueError("the primary of record must be under --evidence-root to be cited as $E/...")
        control_of = "$E/" + str(of_record_path.relative_to(root))
    native_baseline = native_peers.baseline_of(native, native_label) if native_label else "peer"
    group = (f"{family}-q3-record" if of_record else (f"{family}-q3" if record is None else f"{family}-q3-controls")
             + ("" if comparison is None else f"-{native_baseline.replace('_', '-')}"))
    declared = plan["declared"]["experiment"]
    experiment = f2_q3_tier.experiment_name(declared)
    gone = f2_q3_tier.unavailable((verdict["peer"]["unavailable"] or {}).get("peer"), root)
    out = {"groups": {group: {"family": family,
                              "title": (f"{family} Q3 verdict of record" if of_record else f"{family} Q3 three-arm campaign"
                                        if record is None else f"{family} Q3 controls of the primary of record")
                                       + (f" ({comparison})" if comparison else ""),
                              "workload": "Four-agent coding client, 262144 context, 4 slots; stock "
                                          f"{plan['declared']['stock']['label']} vs experiment {experiment} vs "
                                          + (native_label or ("the peer" if gone is None else f"the peer (peer unavailable: {gone})"))
                                          + ", Williams blocks."
                                          + ("" if record is None else f" Controls (client --phase-plan) of the primary "
                                                                       f"of record {control_of}; never a win or loss.")}},
           "rows": []}
    if of_record:
        gates, review_ref = record_inputs(root, quality_gate, review)
        out["groups"][group]["q3_record"] = f2_q3_tier.build_record(verdict, manifest, root, (ref, campaign_ref), gates, review_ref)
        control_of = ref  # the record's own completed controls (n1, mixed) render as controls of this verdict
    if record is None:
        status = F2_STATUS.get(verdict["verdict"])
        sections = [] if status is None else [("primary", None, verdict["primary"], verdict.get("native_peer"))]
        if of_record:
            sections += [(f"controls.{key}", key, stage, None) for key, stage in verdict["controls"].items()
                         if stage["status"] == "Complete"]
    else:
        status = "Descriptive"  # a control of a primary of record never wins or loses (#369 amendment)
        sections = [(f"controls.{key}", key, stage, (verdict.get("native_peer") or {}).get(key))
                    for key, stage in verdict["controls"].items() if stage["status"] == "Complete"]
    # A failed experiment run (Q3 rule 1) is a Failure row: of a stopped control stage in a controls-only
    # campaign, or of the primary stage of a verdict of record (the family verdict is then Failed).
    failed = ([f for f in stopped_experiment_failures(manifest) if f[0] == plan["primary"]] if of_record
              else [] if record is None else list(stopped_experiment_failures(manifest)))
    if not sections and not failed:
        return out
    if not arms["experiment"]["artifact"] or not arms["experiment"]["artifact"].get("sha256"):
        raise ValueError("F2 rows need the model sha256 (conversion receipt or HF2Q_Q5_MODEL_SHA256)")
    base = {"family": family, "group": group, "n_slots": 4, "context": "262144", "date": manifest["created_utc"][:10]}
    for prefix, control, stage, section_native in sections:
        of_record_row = of_record and control is None
        peer_gone = verdict["verdict"] == "Unavailable" if control is None else bool(verdict["peer"]["unavailable"].get("peer"))
        # A campaign whose peer arm is a native engine (ds4, TensorFold) is never the verdict of record: it emits
        # only its native row(s), never experiment-vs-stock (that effect belongs to the same-GGUF campaign of record).
        contrasts = (() if native else (("vs_stock", "v0128_stock"),)) + (("vs_peer", native_baseline),)
        tables = [("makespan_s", "effects", stage["effects"])]
        tables += [(m, f"secondary.{m}", stage["secondary"][m]) for m in f2_q3_tier.FAIRNESS] if of_record_row else []
        for metric, where, table in tables:
            for name, baseline in contrasts:
                e = table.get(name)
                withheld = native_label and (section_native or {}).get("statement_withheld")
                if e is None or e["improvement_percent"] is None or (name == "vs_peer" and (
                        withheld or (peer_gone and not native_label))):
                    continue
                path = f"{prefix}.{where}.{name}"
                what = "makespan" if metric == "makespan_s" else metric.replace("_", "-")
                extract = {"value": {"kind": "json", "file": ref, "path": f"{path}.improvement_percent", "reduce": "one"},
                           "ci95": {"kind": "json", "file": ref, "path": f"{path}.improvement_ci95.*", "reduce": "list"}}
                blocks = (f"{e['blocks']} paired blocks, half-width {e['half_width_pp']:.2f} points "
                          f"(baseline {'stock ' + plan['declared']['stock']['label'] if name == 'vs_stock' else peer_note}).")
                if control is not None and name == "vs_peer" and not native_label:
                    blocks += " " + peer_sentence(next(s for s in manifest["stages"] if s["workload"] == control),
                                                  stage, root, experiment)
                if record is not None:
                    blocks += f" {campaign_note(verdict, manifest, root)}."
                row = {
                    "id": f"{group}-{'' if control is None else control.replace('+', 'p') + '-'}{what}-{name.replace('_', '-')}",
                    **base, **({"q3_of_record": True} if of_record_row else {}),
                    "metric": metric, "kind": "effect", "arm": "candidate",
                    "variant": f"Q3 experiment {experiment}"
                               + ("" if control is None else f", control {control}")
                               + (f" ({comparison})" if comparison else ""),
                    "value": round(e["improvement_percent"], 3), "decimals": 3, "unit": "% improvement",
                    "ci95": [round(x, 3) for x in e["improvement_ci95"]], "baseline_arm": baseline,
                    "source": {"commit": declared["commit"], "binary_sha256": arms["experiment"]["binary"]["sha256"]},
                    "artifact_sha256": arms["experiment"]["artifact"]["sha256"], "receipt": ref,
                    "status": status if control is None else "Descriptive",
                    "verification": {"state": "Verified", "key": f"verdict.json {path}.improvement_percent", "extract": extract},
                    "note": (f"Q3 verdict of record ({'task makespan only' if name == 'vs_peer' else 'see the tier'}); {blocks}"
                             if of_record_row else (f"{comparison}. " if comparison else "")
                             + (f"Q3 verdict {verdict['verdict']}, campaign {verdict['campaign_status']}; " if control is None else
                                f"Q3 control {control} (client --phase-plan {stage['phase_plan']}) of the primary of record "
                                f"{control_of}; never a win or loss; ")
                             + blocks + (f" {native_note(section_native)}." if name == "vs_peer" and native_label else ""))}
                if of_record_row:  # full precision: the tier renders it, --verify compares at the row's decimals
                    row["half_width_pp"] = e["half_width_pp"]
                    extract["half_width_pp"] = {"kind": "json", "file": ref, "path": f"{path}.half_width_pp", "reduce": "one"}
                if control is not None:
                    row["control_of"] = control_of
                out["rows"].append(row)
    if of_record and status is not None:
        out["rows"] += token_rows(manifest, root, {**base, "q3_of_record": True}, status, campaign_ref, experiment)
    for key, where, entry in failed:
        path = f"{where}.metrics.tasks_passed"
        detail = failure_detail(entry)
        row = {
            "id": f"{group}-{key.replace('+', 'p')}-task-pass-{entry['name']}", "family": family, "group": group,
            **({"q3_of_record": True} if of_record else {}),
            "metric": "task_pass", "kind": "measurement", "arm": "candidate",
            "variant": f"Q3 experiment {experiment}" + ("" if of_record else f", control {key}")
                       + (f" ({comparison})" if comparison else ""),
            "value": entry["metrics"]["tasks_passed"], "decimals": 0, "unit": "tasks",
            "n_slots": 4, "context": "262144", "date": manifest["created_utc"][:10],
            "source": {"commit": plan["declared"]["experiment"]["commit"], "binary_sha256": arms["experiment"]["binary"]["sha256"]},
            "artifact_sha256": arms["experiment"]["artifact"]["sha256"], "receipt": campaign_ref, "status": "Failure",
            "verification": {"state": "Verified", "key": f"campaign.json {path}",
                             "extract": {"value": {"kind": "json", "file": campaign_ref, "path": path, "reduce": "one"}}},
            "note": (f"{comparison}. " if comparison else "")
                    + (f"Q3 verdict of record {verdict['verdict']} (rule 1): the {key} {entry['kind']} experiment run "
                       if of_record else f"Q3 control {key} of the primary of record {control_of}: experiment run ")
                    + f"{entry['name']} failed its task ({entry['failure_kind']})"
                    + (f": {detail}" if detail else "")
                    + "; the stage stopped and later stages did not run."
                    + (" The count is the agents that finished before the coordinator stopped the run; it is not "
                       "a partial success." if of_record else "")}
        if not of_record:
            row["control_of"] = control_of
        out["rows"].append(row)
    return out
