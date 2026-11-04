#!/usr/bin/env python3
"""ADR-065 fixed-count campaigns: the B1 phase-1 baseline, two-arm A/Bs and their analysis (#503).

Everything a run needs (binary identity, official artifact, Q5 case, approval template, sandbox proof,
negative control, block counts, cooldown, run budget) comes from the checked-in campaign file
scripts/four_agent/adr065/campaign-b1.json; docs/research/adr065-b1-campaign.md states the design.

  baseline  the predeclared phase-1 baseline: arm A only, the released default binary, a fixed number
            of measured runs per workload after one discarded warmup. Feeds `ceilings`.
  ab        a two-arm A/B on one binary: A the previous default (the family case), B a case that differs
            only in serve.env (the kill switch). Fixed block count (6, or 8/12/16/20 chosen from --pilot-of);
            odd blocks A then B, even blocks B then A (ABBA over each pair of blocks).
  pilot     a separately labelled A/B whose runs are never pooled and which never has a verdict.
  ceilings  BASELINE_DIR/ceilings.json: per-run per-slot p95 semantic TTFT and maximum semantic SSE gap
            of every baseline run, and the largest value per workload (the frozen fairness ceilings).
  analyze   AB_DIR/analysis.json: paired log-makespan interval, Loss/Win/Equivalent/Inconclusive, controls,
            per-run fairness checks against the ceilings and every retained (failed or discarded) run.

Runs reuse the ADR-063 Q3 machinery: q3_campaign.Driver (frozen identities, warmups, settle, Q2 run audit,
resume), q3_metrics and q5_calibration.py; adr065_stats.py holds the pure estimator and verdict rules.
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import adr065_stats
import q2_traced_run
import q3_campaign as q3
import q3_identity
import q3_metrics
import q3_report
import q3_stats
import q5_artifacts
import q5_case
import q5_host
import quality_campaign as qc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CAMPAIGN_FILE = HERE / "adr065" / "campaign-b1.json"
CAMPAIGN_SCHEMA = "hf2q.adr065.campaign.v1"
SCHEMA = "hf2q.adr065.campaign-run.v1"
BUILD_RECEIPT_SCHEMA = "hf2q.adr065.build-receipt.v1"
NATIVE = re.compile(rb"mlx-native-(\d+\.\d+\.\d+)")
ROLES = {"A": "previous default", "B": "candidate (one kill switch changed)"}
# Campaign-file baseline fields that stay null until the pinned main commit is built (a placeholder plan).
BASELINE_FIELDS = ("source_commit", "sha256", "bytes", "native_checksum", "build_receipt.sha256")


def load_campaign(path=CAMPAIGN_FILE):
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    qc.require(doc.get("schema") == CAMPAIGN_SCHEMA, f"{path} is not a {CAMPAIGN_SCHEMA} campaign file")
    return doc, {"path": str(path), "sha256": q5_artifacts.sha256(path)}


def expand(text, roots):
    for name, value in roots.items():
        text = text.replace("{" + name + "}", value)
    return text


def native_versions(path, chunk=1 << 22):
    """mlx-native crate versions named in a binary's embedded source paths (streamed scan)."""
    found, tail = set(), b""
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            window = tail + block
            found.update(m.decode() for m in NATIVE.findall(window))
            tail = window[-32:]
    return sorted(found)


def budget(stage_runs, arms, allowance):
    """Planned runs (one warmup per arm per stage, then the fixed runs) and the attempt cap."""
    planned = sum(len(arms) * (1 + blocks) for blocks in stage_runs.values())
    return {"planned_runs": planned, "max_attempts": planned + allowance}


def hours(runs, seconds, cooldown):
    return [round(runs * (s + cooldown) / 3600, 1) for s in seconds]


def kill_switch(a_case, b_case):
    """The B case may differ from the A case only in serve.env (and notes): the changed variables."""
    def rest(case):
        out = json.loads(json.dumps(case))
        out.pop("notes", None)
        out["serve"].pop("env", None)
        return out
    qc.require(rest(a_case) == rest(b_case), "the B case must equal the family case except serve.env (and notes)")
    env_a, env_b = a_case["serve"].get("env", {}), b_case["serve"].get("env", {})
    changed = {k: [env_a.get(k), env_b.get(k)] for k in sorted(set(env_a) | set(env_b)) if env_a.get(k) != env_b.get(k)}
    qc.require(changed, "the B case changes no serve.env variable (no kill switch)")
    return changed


def read_case(path, sha256=None):
    path = Path(path).resolve()
    if sha256 is not None:
        qc.require(q5_artifacts.sha256(path) == sha256, f"{path} sha256 differs from the campaign file")
    return path, q5_case.validate(q5_case.loads(path.read_bytes()))


def build_plan(args, campaign, source):
    """(plan, cases). The plan is everything a resume must reproduce exactly."""
    family = args.family
    qc.require(family in campaign["families"], f"family {family} is not in the campaign file")
    fam, roots, env = campaign["families"][family], campaign["roots"], campaign["environment"]
    execution, design = campaign["execution"], campaign["design"]
    case_path, case = read_case(REPO / fam["q5_case"]["path"], fam["q5_case"]["sha256"])
    qc.require(case["family"] == family and qc.engine_kind(case) == "hf2q" and q3.CLIENT in case["client"]["argv"],
               f"{case_path} must be the canonical four-agent hf2q case of {family}")
    if args.mode == "baseline":
        binary = args.baseline_bin or expand(campaign["baseline"]["binary"]["path"], roots)
    else:
        binary = args.binary
    qc.require(binary, "--binary is required")
    values = {"HF2Q_BIN": str(Path(binary).resolve()), "MODEL": expand(fam["artifact"]["path"], roots),
              "MODEL_SHA256": fam["artifact"]["sha256"],
              "CLIENT_APPROVAL_TEMPLATE": expand(fam["client_approval_template"]["path"], roots),
              "CARGO": env["CARGO"], "TOOLCHAIN": env["TOOLCHAIN"],
              "SANDBOX_PROOF": expand(env["sandbox_proof"]["path"], roots),
              "SANDBOX_PROOF_SHA256": env["sandbox_proof"]["sha256"]}
    overrides = qc.overrides(case, [f"{k}={v}" for k, v in values.items()], {})
    arms = {"A": {"case": str(case_path), "case_sha256": fam["q5_case"]["sha256"], "engine_kind": "hf2q",
                  "role": ROLES["A"], "overrides": overrides}}
    cases = {"A": case}
    workloads = q3.parse_workloads(args.workloads or ",".join(campaign["workloads"]["order"]))
    plan = {"schema": SCHEMA, "mode": args.mode, "family": family, "primary": adr065_stats.PRIMARY,
            "workloads": workloads, "campaign_file": source, "peer_unavailable_evidence": [],
            "negative_control_receipt": {"path": expand(env["negative_control_receipt"]["path"], roots),
                                         "sha256": env["negative_control_receipt"]["sha256"]},
            "artifact": fam["artifact"], "client_approval_template": {
                "path": values["CLIENT_APPROVAL_TEMPLATE"], "sha256": fam["client_approval_template"]["sha256"]},
            "sandbox_proof": {"path": values["SANDBOX_PROOF"], "sha256": env["sandbox_proof"]["sha256"]},
            "cooldown_seconds": execution["cooldown_seconds"],
            "declared": {"binary": values["HF2Q_BIN"], "qe_identity": args.qe_identity}}
    if args.mode == "baseline":
        qc.require(args.workloads is None, "the baseline campaign runs every predeclared workload")
        stage_runs = dict(campaign["baseline_campaign"]["runs_per_stage"])
        base = campaign["baseline"]
        receipt = base["binary"]["build_receipt"]
        plan["baseline"] = {"label": base["label"], "kind": base["kind"],
                            **{k: base["binary"][k] for k in ("source_commit", "sha256", "bytes")},
                            "build_receipt": {"path": expand(receipt["path"], roots), "sha256": receipt["sha256"]},
                            "native_version": base["native"]["version"], "native_checksum": base["native"]["checksum"]}
        plan["declared"]["commit"] = base["binary"]["source_commit"]
        plan["n1_control"] = None
        fixed = None
    else:
        b_path, b_case = read_case(args.b_case)
        switch = kill_switch(case, b_case)
        arms["B"] = {"case": str(b_path), "case_sha256": q5_artifacts.sha256(b_path), "engine_kind": "hf2q",
                     "role": ROLES["B"], "overrides": qc.overrides(b_case, [f"{k}={v}" for k, v in values.items()], {}),
                     "kill_switch": switch}
        cases["B"] = b_case
        fixed = args.blocks or fam["ab_blocks"]
        allowed = design["ab_block_count"]["allowed"]
        qc.require(fixed in allowed, f"--blocks must be one of {allowed}")
        if args.mode == "ab" and fixed != design["ab_block_count"]["default"]:
            qc.require(args.pilot_of is not None, f"{fixed} blocks needs --pilot-of (the separately labelled pilot)")
        if args.pilot_of is not None:
            qc.require(args.mode == "ab", "--pilot-of applies to an A/B")
            pilot = Path(args.pilot_of).resolve() / "campaign.json"
            record = q3_metrics.load_json(pilot)
            qc.require(isinstance(record, dict) and record.get("schema") == SCHEMA and record["plan"]["mode"] == "pilot"
                       and record["plan"]["family"] == family, f"{pilot} is not a {family} pilot campaign")
            plan["pilot_of"] = {"path": str(pilot), "sha256": q5_artifacts.sha256(pilot), "never_pooled": True}
        plan["declared"].update(commit=args.commit, label=args.label)
        plan["n1_control"] = "reported" if args.n1_engages else "required"
        stage_runs = {key: fixed for key in workloads}
    stage_runs = {key: stage_runs[key] for key in workloads}
    plan["measured_arms"] = list(arms)
    plan["arms"] = arms
    plan["design"] = {"metric": "client summary own_output_four_task_makespan_seconds", "blocks": fixed,
                      "stage_blocks": stage_runs,
                      "order": "q3_stats.block_order over the measured arms: odd blocks A,B; even blocks B,A",
                      "warmup": "one discarded run per measured arm per stage per driver session",
                      "estimator": design["effect"], "verdict_order": design["verdict_order"],
                      "margin_points": design["margin_points"]}
    plan["budget"] = budget(stage_runs, arms, execution["retry_allowance_attempts"])
    plan["budget"]["hours_estimate"] = hours(plan["budget"]["planned_runs"], fam["run_seconds_estimate"],
                                             execution["cooldown_seconds"])
    return plan, cases


def placeholders(base):
    """Campaign-file baseline identity fields still null (the pinned main commit is not built yet)."""
    values = {**base, "build_receipt.sha256": base["build_receipt"]["sha256"]}
    return [field for field in BASELINE_FIELDS if values[field] is None]


def baseline_problems(base, shas, sizes, versions, lock=q2_traced_run.git_lock):
    """The served binary must be the campaign file's main build: its sha256 and size, its build receipt (bound by
    sha256: source commit, binary sha256, native pin), the mlx-native pin in Cargo.lock at that commit, and the one
    mlx-native version embedded in the binary."""
    missing = placeholders(base)
    if missing:
        return [f"the campaign file's baseline identity is a placeholder ({', '.join(missing)} unset): build the "
                "pinned main commit, write its build receipt and finalize campaign-b1.json"]
    problems = []
    if shas != {base["sha256"]} or sizes != {base["bytes"]}:
        problems.append(f"the baseline binary must be the main build {base['sha256']} ({base['bytes']} bytes)")
    receipt = base["build_receipt"]
    document = q3_metrics.load_json(receipt["path"]) if os.path.isfile(receipt["path"]) else None
    if document is None or q5_artifacts.sha256(receipt["path"]) != receipt["sha256"]:
        problems.append(f"build receipt {receipt['path']} missing or not {receipt['sha256']}")
    else:
        expected = {"schema": BUILD_RECEIPT_SCHEMA, "source": base["source_commit"], "sha256": base["sha256"],
                    "bytes": base["bytes"], "native_version": base["native_version"],
                    "native_checksum": base["native_checksum"]}
        wrong = [k for k, v in expected.items() if document.get(k) != v]
        if wrong:
            problems.append(f"build receipt {receipt['path']} differs from the campaign file: {', '.join(wrong)}")
    try:
        pin = q2_traced_run.native_pin(lock(base["source_commit"]))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        problems.append(f"no mlx-native pin readable from Cargo.lock at {base['source_commit']}: {error}")
    else:
        if pin != (base["native_version"], base["native_checksum"]):
            problems.append(f"Cargo.lock at {base['source_commit']} pins mlx-native {pin[0]} ({pin[1]}), not "
                            f"{base['native_version']} ({base['native_checksum']})")
    if versions != [base["native_version"]]:
        problems.append(f"the binary must embed exactly one mlx-native version ({base['native_version']}); "
                        f"found {versions}")
    return problems


def build_receipt(binary, source_checkout, command, lock=q2_traced_run.git_lock):
    """The B1 build receipt of a binary built from a clean checkout of a main commit."""
    checkout = Path(source_checkout).resolve()
    git = lambda *argv: subprocess.run(["git", "-C", str(checkout), *argv], check=True, capture_output=True,
                                       text=True, timeout=30).stdout
    commit = git("rev-parse", "HEAD").strip()
    qc.require(git("status", "--porcelain", "--untracked-files=no") == "", f"{checkout} has uncommitted changes")
    version, checksum = q2_traced_run.native_pin(lock(commit))
    embedded = native_versions(binary)
    qc.require(embedded == [version], f"{binary} embeds mlx-native {embedded}, Cargo.lock pins {version}")
    stat = os.stat(binary)
    return {"schema": BUILD_RECEIPT_SCHEMA, "source": commit, "sha256": q5_artifacts.sha256(binary),
            "bytes": stat.st_size, "native_version": version, "native_checksum": checksum,
            "embedded_native_versions": embedded, "binary": str(Path(binary).resolve()),
            "source_checkout": str(checkout), "command": command, "binary_mtime_ns": stat.st_mtime_ns}


def launch_problems(plan, ids, versions, lock=q2_traced_run.git_lock):
    """Everything that keeps an ADR-065 campaign from starting; empty when launchable."""
    problems = q3_identity.arm_problems(ids)
    artifact = plan["artifact"]
    for name, identity in ids["arms"].items():
        found = identity["artifact"] or {}
        if (found.get("sha256"), found.get("bytes")) != (artifact["sha256"], artifact["bytes"]):
            problems.append(f"{name}: artifact {found.get('path')} is not the official {artifact['sha256']} "
                            f"({artifact['bytes']} bytes)")
        inputs = identity.get("inputs") or {}
        for key in ("client_approval_template", "sandbox_proof"):
            if inputs.get(plan[key]["path"]) not in (None, plan[key]["sha256"]):
                problems.append(f"{name}: {plan[key]['path']} sha256 differs from the campaign file")
    shas = {identity["binary"]["sha256"] for identity in ids["arms"].values()}
    if len(shas) != 1:
        problems.append("every arm must run the same binary (arms differ only in a kill switch)")
    if plan["mode"] == "baseline":
        bytes_ = {identity["binary"].get("bytes") for identity in ids["arms"].values()}
        problems += baseline_problems(plan["baseline"], shas, bytes_, versions, lock)
    elif len(versions) != 1:
        problems.append(f"the binary must embed exactly one mlx-native version; found {versions}")
    commit = plan["declared"].get("commit")
    if not (isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit)):
        problems.append("--commit must be the 40-hex source commit of the binary")
    nc = plan["negative_control_receipt"]
    if not os.path.isfile(nc["path"]) or q5_artifacts.sha256(nc["path"]) != nc["sha256"]:
        problems.append(f"negative control receipt {nc['path']} missing or not {nc['sha256']}")
    if not q3_identity.harness_clean(ids):
        problems.append("the harness checkout must be a clean Git commit (scripts/ has uncommitted changes)")
    if not plan["declared"]["qe_identity"]:
        problems.append("--qe-identity (the named session/host executing the campaign) is required")
    return problems


class Driver(q3.Driver):
    """q3_campaign.Driver with a fixed block count per stage, the campaign cooldown and the run budget."""
    schema = SCHEMA

    def __init__(self, *args, clock=time.monotonic, sleep=time.sleep, versions=native_versions,
                 lock=q2_traced_run.git_lock, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock, self.sleep, self.versions, self.lock, self.last_end = clock, sleep, versions, lock, None

    def launch_problems(self, frozen):
        binary = self.plan["declared"]["binary"]
        return launch_problems(self.plan, frozen, self.versions(binary) if os.path.isfile(binary) else [], self.lock)

    def finished(self, stage):
        fixed = self.plan["design"]["stage_blocks"][stage["workload"]]
        if len(stage["blocks"]) < fixed or not q3_report.block_complete(stage["blocks"][-1]):
            return False
        stage.update(status="Complete", stop={"reason": "fixed_count", "blocks": len(stage["blocks"])})
        self.save()
        return True

    def step(self, stage, entry):
        attempts = sum(len(e["attempts"]) for s in self.manifest["stages"]
                       for e in adr065_stats.runs(s, ("warmup", "measured")))
        if attempts >= self.plan["budget"]["max_attempts"]:
            self.stop("Halted", entry, f"run budget exhausted: {attempts} attempts "
                      f"(max {self.plan['budget']['max_attempts']})", False)
        cooldown = self.plan["cooldown_seconds"]
        wait = cooldown if self.last_end is None else max(0.0, cooldown - (self.clock() - self.last_end))
        self.manifest.setdefault("cooldowns", []).append({"before": entry["name"], "waited_s": round(wait, 3)})
        if wait > 0:
            self.sleep(wait)
        try:
            return super().step(stage, entry)
        finally:
            self.last_end = self.clock()


def schedule(plan):
    arms = plan["measured_arms"]
    return {key: {"phase_plan": q3.WORKLOADS[key], "warmup_order": q3_stats.block_order(arms, 1),
                  "blocks": [q3_stats.block_order(arms, b) for b in range(1, blocks + 1)]}
            for key, blocks in plan["design"]["stage_blocks"].items()}


def write_ceilings(directory, campaign_source):
    directory = Path(directory).resolve()
    manifest = q3_metrics.load_json(directory / "campaign.json")
    qc.require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA, f"{directory} has no ADR-065 campaign.json")
    qc.require(manifest["plan"]["campaign_file"]["sha256"] == campaign_source["sha256"],
               "the campaign file changed since the baseline ran")
    sources = {"campaign": {"path": str(directory / "campaign.json"), "sha256": q5_artifacts.sha256(directory / "campaign.json")},
               "campaign_file": campaign_source}
    document = adr065_stats.ceilings(manifest, sources)
    qc.save(directory / "ceilings.json", document)
    return document


def write_analysis(directory, ceilings_path, campaign_source):
    directory, ceilings_path = Path(directory).resolve(), Path(ceilings_path).resolve()
    manifest = q3_metrics.load_json(directory / "campaign.json")
    qc.require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA, f"{directory} has no ADR-065 campaign.json")
    ceilings = q3_metrics.load_json(ceilings_path)
    qc.require(isinstance(ceilings, dict), f"unreadable ceilings {ceilings_path}")
    qc.require(ceilings["sources"]["campaign_file"]["sha256"] == manifest["plan"]["campaign_file"]["sha256"]
               == campaign_source["sha256"], "the A/B, its ceilings and the campaign file must bind the same campaign file")
    sources = {"campaign": {"path": str(directory / "campaign.json"), "sha256": q5_artifacts.sha256(directory / "campaign.json")},
               "ceilings": {"path": str(ceilings_path), "sha256": q5_artifacts.sha256(ceilings_path)},
               "campaign_file": campaign_source}
    document = adr065_stats.analyze(manifest, ceilings, sources)
    qc.save(directory / "analysis.json", document)
    return document


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-file", type=Path, default=CAMPAIGN_FILE)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("baseline", "ab", "pilot"):
        p = sub.add_parser(mode)
        p.add_argument("--family", required=True, choices=q5_case.FAMILIES)
        p.add_argument("--qe-identity", help="named session/host executing the campaign")
        p.add_argument("--output", type=Path, help="new campaign directory outside Git (existing with --resume)")
        p.add_argument("--resume", action="store_true", help="continue a Halted campaign with an identical plan")
        p.add_argument("--dry-run", action="store_true", help="print plan, identities, problems and schedule only")
        if mode == "baseline":
            p.add_argument("--baseline-bin", help="the phase-1 default binary (default: the campaign file's "
                           "baseline.binary.path); its identity must equal the campaign file's")
            p.set_defaults(workloads=None)
        else:
            p.add_argument("--binary", required=True, help="the one binary both arms run")
            p.add_argument("--commit", required=True, help="40-hex source commit of --binary")
            p.add_argument("--label", default="candidate")
            p.add_argument("--b-case", required=True, help="B's Q5 case: the family case with the kill switch in serve.env")
            p.add_argument("--blocks", type=int, help="fixed block count (default the family's ab_blocks)")
            p.add_argument("--n1-engages", action="store_true",
                           help="the change engages with one agent: the n1 control is reported, not required")
            p.add_argument("--workloads", help="stages in order, n4 first (default every workload of the campaign file)")
            if mode == "ab":
                p.add_argument("--pilot-of", type=Path, help="pilot campaign directory that chose a block count other than 6")
            else:
                p.set_defaults(pilot_of=None)
    p = sub.add_parser("build-receipt", help="write the build receipt of a main-built baseline binary")
    p.add_argument("--binary", type=Path, required=True, help="the built hf2q binary")
    p.add_argument("--source-checkout", type=Path, required=True, help="clean checkout of the main commit it was built from")
    p.add_argument("--command", required=True, help="the exact build command (recorded verbatim)")
    p.add_argument("--output", type=Path, required=True, help="receipt path (new file)")
    p = sub.add_parser("ceilings")
    p.add_argument("baseline", type=Path, help="baseline campaign directory")
    p = sub.add_parser("analyze")
    p.add_argument("campaign", type=Path, help="A/B or pilot campaign directory")
    p.add_argument("--ceilings", type=Path, required=True, help="the frozen ceilings of the family")
    args = parser.parse_args(argv)
    if args.mode in ("baseline", "ab", "pilot") and not args.dry_run and args.output is None:
        parser.error("a campaign run needs --output")
    return args


def main(argv=None):
    args = parse_args(argv)
    environ = dict(os.environ)
    try:
        if args.mode == "build-receipt":
            qc.require(not args.output.exists(), f"{args.output} exists")
            document = build_receipt(args.binary, args.source_checkout, args.command)
            qc.save(args.output, document)
            print(json.dumps({"receipt": str(args.output.resolve()), "sha256": q5_artifacts.sha256(args.output),
                              **{k: document[k] for k in ("source", "sha256", "bytes", "native_version", "native_checksum")}},
                             indent=2))
            return 0
        campaign, source = load_campaign(args.campaign_file)
        if args.mode == "ceilings":
            document = write_ceilings(args.baseline, source)
            print(json.dumps({"ceilings": str(Path(args.baseline).resolve() / "ceilings.json"),
                              "values": {k: {m: v[m]["value"] for m in (adr065_stats.TTFT, adr065_stats.SSE)}
                                         for k, v in document["ceilings"].items()},
                              "retained_failures": [r["name"] for r in document["retained_failures"]]}, indent=2))
            return 0
        if args.mode == "analyze":
            document = write_analysis(args.campaign, args.ceilings, source)
            print(json.dumps({"analysis": str(Path(args.campaign).resolve() / "analysis.json"),
                              "verdict": document["verdict"], "reasons": document["reasons"],
                              "gate_failures": document["gate_failures"]}, indent=2))
            return 0
        plan, cases = build_plan(args, campaign, source)
        identify = lambda: q3_identity.identities(plan, cases, environ)
        if args.dry_run:
            ids = identify()
            binary = plan["declared"]["binary"]
            versions = native_versions(binary) if os.path.isfile(binary) else []
            print(json.dumps({"valid": True, "dry_run": True, "plan_sha256": qc.plan_sha256(plan), "plan": plan,
                              "identities": ids, "native_versions": versions,
                              "launch_problems": launch_problems(plan, ids, versions), "schedule": schedule(plan)}, indent=2))
            return 0
        out = args.output.absolute()
        driver = Driver(plan, cases, out, environ, qc.subprocess_runner, q5_host.thermal_state, identify)
        manifest = driver.run(resume=args.resume)
    except (OSError, qc.CampaignError, q5_case.CaseError, q5_artifacts.ArtifactError, ValueError, KeyError) as error:
        print(json.dumps({"valid": False, "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 2
    print(json.dumps({"campaign": str(out / "campaign.json"), "status": manifest["status"],
                      "halted": manifest.get("halted")}, indent=2))
    return 0 if manifest["status"] == "Complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
