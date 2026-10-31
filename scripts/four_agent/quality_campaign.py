#!/usr/bin/env python3
"""ADR-063 Q1 paired task-quality campaign over distinct four-agent coding instances.

Runs instances 0..N-1 of the four-agent coding client (qwen_n4_live_smoke.py
--instance, rows of fixtures/coding-instances.json) on a subject arm (hf2q) and,
unless floor-only, a reference arm (the peer, or stock hf2q when judging a
candidate). Arm order alternates in ABBA blocks: even instances run the subject
first, odd instances the reference first. Every run is one q5_calibration.py
supervised runtime, started only after the previous one has exited.

An instance passes on an arm when the four-agent client reports
FourAgentProtocolCompleteAwaitingIndependentReview. The manifest keeps every Q5
receipt and client summary. When all runs have a task outcome, the paired
outcomes go through the Q1 `paired` rule (hf2q.q1.numeric-policy.v1). Passing
patches still need independent review. No run is repeated once the task
started; a run that never reached the four-agent release can be resumed.
See README "Quality campaign".
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import coding_instances
import q2_run_audit
import q5_case

HERE = Path(__file__).resolve().parent
SCHEMA = "hf2q.adr063.quality-campaign.v1"
OUTCOMES_SCHEMA = "hf2q.q1.paired-task-outcomes.v1"
CLIENT = "{REPO}/scripts/four_agent/qwen_n4_live_smoke.py"
TABLE = "{REPO}/scripts/four_agent/fixtures/coding-instances.json"
SUCCESS = "FourAgentProtocolCompleteAwaitingIndependentReview"
# Supervisor stops that are the arm's own failure once the task has started.
ARM_FAILURE_STOPS = ("server_exited", "client_timeout")
# Reference kind -> the reference client's --engine-kind.
REFERENCE_KINDS = {"peer": "peer", "stock-hf2q": "hf2q"}
OVERRIDE = q5_case.OVERRIDE_PREFIX

# Port of q1_decision_policy.py `paired` (hf2q.q1.numeric-policy.v1,
# docs/research/adr063-q1-numeric-policy.md §3.5); thresholds are frozen there.
POLICY = {"schema": "hf2q.q1.numeric-policy.v1", "paired_min_instances": 24, "paired_alpha": 0.05,
          "paired_max_excess_failure_fraction": 0.125, "paired_min_pass_fraction": 0.75}


class CampaignError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise CampaignError(reason)


def one_sided_sign_p(b, c):
    n = b + c
    return math.fsum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0


def paired(document, policy=POLICY):
    require(document.get("schema") == OUTCOMES_SCHEMA, "unexpected outcomes schema")
    instances = document["instances"]
    ids = [i["id"] for i in instances]
    require(len(ids) == len(set(ids)), "duplicate instance id")
    require(all(type(i["subject"]) is bool and (i["reference"] is None or type(i["reference"]) is bool)
                for i in instances), "outcomes must be booleans (reference may be null)")
    with_reference = {i["reference"] is not None for i in instances}
    require(len(with_reference) <= 1, "mixed referenced and unreferenced instances")
    n = len(instances)
    subject = sum(i["subject"] for i in instances)
    out = {"schema": "hf2q.q1.paired-task-verdict.v1", "policy": policy["schema"],
           "reference_kind": document.get("reference_kind"), "instances": n, "subject_pass": subject}
    if n < policy["paired_min_instances"]:
        return {**out, "verdict": "Insufficient"}
    floor_ok = subject >= policy["paired_min_pass_fraction"] * n
    if with_reference == {False}:
        return {**out, "verdict": "Pass" if floor_ok else "BelowFloor"}
    b = sum(not i["subject"] and i["reference"] for i in instances)
    c = sum(i["subject"] and not i["reference"] for i in instances)
    p = one_sided_sign_p(b, c)
    out.update(reference_pass=sum(i["reference"] for i in instances),
               subject_only_fail=b, reference_only_fail=c, one_sided_p=p)
    if p < policy["paired_alpha"]:
        return {**out, "verdict": "Loss"}
    if not floor_ok:
        return {**out, "verdict": "BelowFloor"}
    if b - c > math.floor(policy["paired_max_excess_failure_fraction"] * n):
        return {**out, "verdict": "Inconclusive"}
    return {**out, "verdict": "Pass"}


def schedule(instances, arms):
    """ABBA: instance i runs `arms` in order when i is even, reversed when odd."""
    order = []
    for instance in range(instances):
        for arm in (arms if instance % 2 == 0 else arms[::-1]):
            order.append({"index": len(order), "instance": instance, "arm": arm})
    return order


def engine_kind(case):
    argv = case["client"]["argv"]
    return argv[argv.index("--engine-kind") + 1] if "--engine-kind" in argv else "hf2q"


def derive_case(case, instance):
    """The arm's Q5 case with the client pinned to one task instance and its sources frozen."""
    q5_case.validate(case)
    derived = json.loads(json.dumps(case))
    client = derived["client"]
    require(CLIENT in client["argv"] and "--instance" not in client["argv"],
            "case client must be the canonical four-agent coding client without --instance")
    require(set(q5_case.PLACEHOLDER.findall(client.get("summary", ""))) == {"OUT"},
            "client.summary must be a path under {OUT}")
    approval = client.get("approval")
    require(approval is not None and not {"instance", "instance_table_sha256"} & set(approval["set"]),
            "case approval must not already set a task instance")
    sources = approval["set"].get("sources")
    require(isinstance(sources, dict) and set(sources) == {"sha256_of"} and isinstance(sources["sha256_of"], list),
            "case approval must freeze client sources with sha256_of")
    client["argv"] = client["argv"] + ["--instance", str(instance)]
    approval["set"]["instance"] = instance
    approval["set"]["instance_table_sha256"] = {"sha256_of": TABLE}
    sources["sha256_of"] = sources["sha256_of"] + [f"{{REPO}}/scripts/four_agent/{name}"
                                                   for name in coding_instances.MODULES]
    derived["workload"]["id"] = f"{case['workload']['id']}-quality-i{instance:02d}"
    derived["notes"] = list(case.get("notes", [])) + [
        f"Derived by quality_campaign.py: task instance {instance} of fixtures/coding-instances.json."]
    return q5_case.validate(derived)


def overrides(case, explicit, environ):
    """Effective HF2Q_Q5_<NAME> overrides for one arm: ambient ones it declares, then explicit ones."""
    declared = case.get("variables", {})
    values = {k[len(OVERRIDE):]: v for k, v in environ.items()
              if k.startswith(OVERRIDE) and k[len(OVERRIDE):] in declared}
    for item in explicit:
        name, sep, value = item.partition("=")
        require(sep and name in declared, f"arm variable {name!r} is not declared by its case")
        values[name] = value
    return dict(sorted(values.items()))


def arm_environ(environ, values):
    env = {k: v for k, v in environ.items() if not k.startswith(OVERRIDE)}
    env.update({OVERRIDE + k: v for k, v in values.items()})
    return env


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_plan(args, environ):
    rows, table_sha = coding_instances.load()
    require(1 <= args.instances <= len(rows), f"--instances must be 1..{len(rows)} (rows in the instance table)")
    arms, cases = {}, {}
    specs = [("subject", args.subject_case, args.subject_var)]
    if args.reference_case is not None:
        specs.append(("reference", args.reference_case, args.reference_var))
    for arm, path, explicit in specs:
        path = Path(path).resolve()
        cases[arm] = q5_case.validate(q5_case.loads(path.read_bytes()))
        arms[arm] = {"case": str(path), "case_sha256": sha256(path), "family": cases[arm]["family"],
                     "engine_kind": engine_kind(cases[arm]),
                     "overrides": overrides(cases[arm], explicit or [], environ)}
    require(arms["subject"]["engine_kind"] == "hf2q", "the subject arm must be an hf2q case")
    if "reference" in arms:
        require(arms["reference"]["family"] == arms["subject"]["family"], "arms must be the same model family")
        require(arms["reference"]["engine_kind"] == REFERENCE_KINDS[args.reference_kind],
                f"--reference-kind {args.reference_kind} needs a {REFERENCE_KINDS[args.reference_kind]} client case")
    ambient = {k[len(OVERRIDE):] for k in environ if k.startswith(OVERRIDE)}
    unused = sorted(ambient - {n for c in cases.values() for n in c.get("variables", {})})
    require(not unused, "ambient overrides name no variable of either case: " + ",".join(unused))
    plan = {"schema": SCHEMA, "policy": POLICY["schema"], "instances": args.instances,
            "reference_kind": args.reference_kind if "reference" in arms else None,
            "instance_table": {"path": str(coding_instances.TABLE), "sha256": table_sha},
            "sources": {name: sha256(HERE / name) for name in
                        ("quality_campaign.py", *coding_instances.MODULES, "qwen_n4_live_smoke.py")},
            "arms": arms, "schedule": schedule(args.instances, tuple(arms))}
    if args.negative_control_receipt is not None:
        receipt = Path(args.negative_control_receipt).resolve()
        plan["negative_control_receipt"] = {"path": str(receipt), "sha256": sha256(receipt)}
    return plan, cases


def plan_sha256(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def run_name(entry, attempt):
    return f"{entry['index']:02d}-i{entry['instance']:02d}-{entry['arm']}-a{attempt}"


def q5_argv(case_path, output, negative_control):
    return [sys.executable, "-B", str(HERE / "q5_calibration.py"), str(case_path), "--output", str(output),
            "--negative-control-receipt", negative_control]


def classify(receipt, summary, instance, table_sha256):
    """(outcome, reason): Pass/Fail are task outcomes; NotStarted is resumable; Invalid is terminal."""
    if receipt is None:
        return "Invalid", "Q5 receipt missing"
    verdict = receipt.get("verdict", {})
    if receipt.get("client") is None:
        return "NotStarted", f"no client launched: {verdict.get('verdict')}: {verdict.get('reason')}"
    if summary is None:
        return "Invalid", "client launched but wrote no summary"
    if summary.get("barrier", {}).get("status") != "Released":
        return "NotStarted", "client ended before the four-agent release: " + str(summary.get("error"))
    if summary.get("instance") != {"id": instance, "table_sha256": table_sha256}:
        return "Invalid", "client summary task instance differs from the campaign's"
    if receipt.get("supervisor_error"):
        return "Invalid", "supervisor error after the task started: " + receipt["supervisor_error"]
    stop = receipt.get("stop")
    if stop is not None and stop.get("reason") not in ARM_FAILURE_STOPS:
        return "Invalid", f"supervisor stop {stop.get('reason')} after the task started"
    if stop is None and receipt["client"].get("exit_code") == 0 and summary.get("status") == SUCCESS:
        return "Pass", None
    return "Fail", (f"supervisor stop {stop['reason']}" if stop is not None else str(summary.get("error")))


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def save(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(pending, path)


def launch(entry, case, plan, out):
    """Write the derived case and append a Launched attempt (saved before the run starts)."""
    case_path = out / "cases" / f"i{entry['instance']:02d}-{entry['arm']}.json"
    text = json.dumps(derive_case(case, entry["instance"]), indent=2) + "\n"
    if case_path.exists():
        require(case_path.read_text() == text, f"derived case {case_path.name} changed")
    else:
        case_path.write_text(text)
    name = run_name(entry, len(entry["attempts"]))
    output = out / "runs" / name
    record = {"name": name, "output": str(output), "case": str(case_path), "case_sha256": sha256(case_path),
              "argv": q5_argv(case_path, output, plan["negative_control_receipt"]["path"]), "outcome": "Launched"}
    entry["attempts"].append(record)
    return record


def settle(entry, record, case, plan, code):
    """Score an attempt from its Q5 receipt and client summary on disk."""
    output = Path(record["output"])
    receipt = load_json(output / "receipt.json")
    summary_path = Path(q5_case.expand(case["client"]["summary"], {"OUT": str(output)}))
    summary = load_json(summary_path)
    report = None
    if output.exists():
        outcome, reason = classify(receipt, summary, entry["instance"], plan["instance_table"]["sha256"])
    else:
        outcome, reason = "NotStarted", "Q5 never created the run directory"
    if outcome == "Pass":
        # A claimed task success counts only after its raw client evidence re-derives it (ADR-063 Q2).
        report, failure = q2_run_audit.check_run(receipt, summary_path.parent, summary, None,
                                                 plan["arms"][entry["arm"]]["engine_kind"], require_width=False)
        if failure is not None:
            outcome, reason = "Invalid", failure
    record.update(q5_exit=code, outcome=outcome, reason=reason,
                  receipt_sha256=sha256(output / "receipt.json") if receipt is not None else None,
                  q5_verdict=None if receipt is None else receipt.get("verdict"),
                  stop=None if receipt is None else receipt.get("stop"), audit=report)
    if summary is not None:
        record.update(client_summary=str(summary_path), client_summary_sha256=sha256(summary_path),
                      client_status=summary.get("status"), client_error=summary.get("error"),
                      agents={a: {"status": v.get("status"), "error": v.get("error")}
                              for a, v in sorted(summary.get("agents", {}).items())})
    if outcome in ("Pass", "Fail"):
        entry["outcome"] = outcome


def subprocess_runner(argv, env, stdout, stderr):
    return subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, cwd=HERE).returncode


def outcomes(manifest):
    runs = {(r["instance"], r["arm"]): r["outcome"] == "Pass" for r in manifest["runs"]}
    floor_only = manifest["plan"]["reference_kind"] is None
    return {"schema": OUTCOMES_SCHEMA, "reference_kind": manifest["plan"]["reference_kind"],
            "instances": [{"id": f"i{i:02d}", "subject": runs[(i, "subject")],
                           "reference": None if floor_only else runs[(i, "reference")]}
                          for i in range(manifest["plan"]["instances"])]}


def run(plan, cases, out, environ, runner=subprocess_runner, resume=False):
    path = out / "manifest.json"
    digest = plan_sha256(plan)
    if resume:
        manifest = load_json(path)
        require(manifest is not None and manifest.get("plan_sha256") == digest,
                "resume needs the same cases, overrides, instance table, sources and negative control")
        require(manifest["status"] != "Complete", "campaign already complete")
    else:
        require(not out.exists(), "campaign output must be a new directory")
        require(not any((p / ".git").exists() for p in (out.parent.resolve(), *out.parent.resolve().parents)),
                "campaign output must be outside any Git tree")
        out.mkdir(mode=0o700)
        for name in ("cases", "runs", "logs"):
            (out / name).mkdir(mode=0o700)
        manifest = {"schema": SCHEMA, "plan": plan, "plan_sha256": digest,
                    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "independent_patch_review_required": True,
                    "runs": [{**entry, "attempts": [], "outcome": None} for entry in plan["schedule"]]}
    for entry in manifest["runs"]:
        case = cases[entry["arm"]]
        if entry["attempts"] and entry["attempts"][-1]["outcome"] == "Launched":
            settle(entry, entry["attempts"][-1], case, plan, None)  # the driver ended during this run
            save(path, manifest)
        if entry["outcome"] is not None:
            continue
        last = entry["attempts"][-1] if entry["attempts"] else {}
        require(last.get("outcome") != "Invalid", f"run {entry['index']} is Invalid ({last.get('reason')}); "
                "no reruns after the task started, so start a new campaign")
        record = launch(entry, case, plan, out)
        manifest["status"] = "Running"
        save(path, manifest)
        logs = [out / "logs" / f"{record['name']}.{stream}" for stream in ("stdout", "stderr")]
        with open(logs[0], "xb") as stdout, open(logs[1], "xb") as stderr:
            code = runner(record["argv"], arm_environ(environ, plan["arms"][entry["arm"]]["overrides"]),
                          stdout, stderr)
        settle(entry, record, case, plan, code)
        if entry["outcome"] is None:
            manifest.update(status="Halted", halted_at=entry["index"])
            save(path, manifest)
            return manifest
        save(path, manifest)
    document = outcomes(manifest)
    save(out / "outcomes.json", document)
    manifest.update(status="Complete", outcomes_sha256=sha256(out / "outcomes.json"), verdict=paired(document))
    manifest.pop("halted_at", None)
    save(path, manifest)
    return manifest


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-case", required=True, help="hf2q Q5 case (q5_cases/<family>.json)")
    parser.add_argument("--reference-case", help="peer or stock-hf2q Q5 case; omit for a floor-only verdict")
    parser.add_argument("--reference-kind", choices=sorted(REFERENCE_KINDS))
    parser.add_argument("--subject-var", action="append", metavar="NAME=VALUE",
                        help="HF2Q_Q5_<NAME> override for the subject arm only (repeatable)")
    parser.add_argument("--reference-var", action="append", metavar="NAME=VALUE",
                        help="HF2Q_Q5_<NAME> override for the reference arm only (repeatable)")
    parser.add_argument("--instances", type=int, default=POLICY["paired_min_instances"],
                        help="run instances 0..N-1 (default 24, the policy minimum)")
    parser.add_argument("--negative-control-receipt", help="accepted NegativeControlPass receipt, passed to every run")
    parser.add_argument("--output", type=Path, help="new campaign directory outside Git (existing with --resume)")
    parser.add_argument("--resume", action="store_true", help="continue a Halted campaign with an identical plan")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and schedule; launch nothing")
    args = parser.parse_args(argv)
    if (args.reference_case is None) != (args.reference_kind is None):
        parser.error("--reference-case and --reference-kind go together")
    if args.reference_var and args.reference_case is None:
        parser.error("--reference-var needs --reference-case")
    if not args.dry_run and (args.output is None or args.negative_control_receipt is None):
        parser.error("--output and --negative-control-receipt are required unless --dry-run")
    return args


def main(argv=None):
    args = parse_args(argv)
    environ = dict(os.environ)
    try:
        plan, cases = build_plan(args, environ)
        if args.dry_run:
            import q5_calibration
            arms = {}
            for arm, case in cases.items():
                derived = derive_case(case, 0)
                values = q5_case.build_plan(derived, q5_calibration.builtins(Path("/DRY-RUN") / arm),
                                            arm_environ(environ, plan["arms"][arm]["overrides"]))
                arms[arm] = {"missing_variables": values["missing_variables"],
                             "client_argv": values["client"]["argv"]}
            print(json.dumps({"valid": True, "dry_run": True, "plan_sha256": plan_sha256(plan), "plan": plan,
                              "instance_0": arms,
                              "q5_argv_template": q5_argv("<OUTPUT>/cases/iNN-<arm>.json",
                                                          "<OUTPUT>/runs/<index>-iNN-<arm>-a0",
                                                          plan.get("negative_control_receipt", {}).get("path", "<NC_RECEIPT>"))},
                             indent=2))
            return 0
        manifest = run(plan, cases, args.output.absolute(), environ, resume=args.resume)
    except (OSError, CampaignError, q5_case.CaseError, ValueError) as error:
        print(json.dumps({"valid": False, "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 2
    print(json.dumps({"manifest": str(args.output.absolute() / "manifest.json"), "status": manifest["status"],
                      "halted_at": manifest.get("halted_at"), "verdict": manifest.get("verdict")}, indent=2))
    return 0 if manifest["status"] == "Complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
