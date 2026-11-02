#!/usr/bin/env python3
"""ADR-063 Q3 three-arm crossover campaign for one model family.

Arms: stock hf2q (the released binary), experiment hf2q (the candidate binary) and the
peer in its best valid configuration (q5_cases/peer-<family>.json), plus an optional
labeled matched-settings peer diagnostic. A native-engine --peer-case (native_peers.py: its own
artifact) runs in the peer slot either with --peer-unavailable-evidence for the same-GGUF peer
(ds4) or as a separately labeled --native-comparison campaign beside the verdict of record
(TensorFold); verdict.json native_peer reports it, task makespan only. Every run is one q5_calibration.py supervised
runtime, started only after the previous one exited. Identities (binaries, model, peer
build, harness sources and commit, Python, host) are frozen at campaign start and
re-checked before every run; any change halts the campaign for good.

Each workload stage (n4 primary pure-phase, then the n1 and 3+1/2+2/1+3 controls) runs
one discarded warmup per arm per driver session, then Williams-ordered blocks: at least 5,
at most 20, stopping once the 95% half-width of every required experiment contrast is
<= 2.5 points. Writes campaign.json and verdict.json. Design, estimator, verdict rules
and root commands: docs/research/adr063-q3-campaign.md.
--controls-of runs control stages only, as controls of a primary verdict of record (#369
amendment): its family and arm identities must match that campaign's frozen ones.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

import native_peers
import q2_run_audit
import q3_identity
import q3_metrics
import q3_report
import q3_stats
import q5_artifacts
import q5_case
import q5_host
import quality_campaign as qc

HERE = Path(__file__).resolve().parent
SCHEMA = "hf2q.adr063.q3-campaign.v1"
CLIENT = qc.CLIENT
# Workload key -> client --phase-plan (None: canonical, all four agents released together).
WORKLOADS = {"n4": None, "n1": "1", "3+1": "3+1", "2+2": "2+2", "1+3": "1+3"}
SLUGS = {"n4": "n4", "n1": "n1", "3+1": "p3p1", "2+2": "p2p2", "1+3": "p1p3"}
PRIMARY = "n4"
CONTROLS = tuple(key for key in WORKLOADS if key != PRIMARY)
ARMS = ("stock", "experiment", "peer", "peer_matched")
ROLES = {"stock": "stock hf2q: frozen released baseline", "experiment": "experimental hf2q on that baseline",
         "peer": "peer, best valid configuration (mandatory comparator)",
         "peer_matched": "peer, matched-settings diagnostic (never enters the verdict)"}
# The peer arm may instead be a native engine (its own artifact): labeled, reported separately.
ARM_KINDS = {"stock": ("hf2q",), "experiment": ("hf2q",), "peer": ("peer",) + native_peers.KINDS,
             "peer_matched": ("peer",)}
TERMINAL = ("Complete", "Invalid", "IdentityDrift")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    return q5_artifacts.sha256(path)


def save(path, value):
    qc.save(path, value)


def derive_case(case, workload):
    """The arm's Q5 case for one workload: unchanged for n4, plus --phase-plan for a control."""
    q5_case.validate(case)
    derived = json.loads(json.dumps(case))
    plan = WORKLOADS[workload]
    if plan is None:
        return derived
    client = derived["client"]
    qc.require(CLIENT in client["argv"] and not {"--phase-plan", "--instance"} & set(client["argv"]),
               "case client must be the canonical four-agent client without --phase-plan/--instance")
    client["argv"] = client["argv"] + ["--phase-plan", plan]
    derived["workload"]["id"] = f"{case['workload']['id']}-q3-{SLUGS[workload]}"
    derived["notes"] = list(case.get("notes", [])) + [
        f"Derived by q3_campaign.py: Q3 control {workload} (client --phase-plan {plan})."]
    return q5_case.validate(derived)


def parse_workloads(text, controls_only=False):
    keys = [k.strip() for k in text.split(",") if k.strip()]
    if controls_only:
        qc.require(keys and len(set(keys)) == len(keys) and set(keys) <= set(CONTROLS),
                   f"--workloads with --controls-of must name distinct control keys of {','.join(CONTROLS)} (never {PRIMARY})")
        return keys
    qc.require(keys and keys[0] == PRIMARY and len(set(keys)) == len(keys) and set(keys) <= set(WORKLOADS),
               f"--workloads must start with {PRIMARY} and name distinct keys of {','.join(WORKLOADS)}")
    return keys


def primary_of_record(path, family):
    """--controls-of: the campaign of record for the primary (#369 amendment), bound by path and sha256."""
    verdict_path = Path(path).resolve()
    campaign_path = verdict_path.parent / "campaign.json"
    verdict, manifest = q3_metrics.load_json(verdict_path), q3_metrics.load_json(campaign_path)
    qc.require(verdict_path.name == "verdict.json" and isinstance(verdict, dict)
               and verdict.get("schema") == q3_report.VERDICT_SCHEMA
               and isinstance(manifest, dict) and manifest.get("schema") == SCHEMA,
               f"--controls-of {verdict_path} must be a Q3 verdict.json beside its campaign.json")
    qc.require(verdict.get("primary") is not None and manifest["plan"].get("controls_of") is None,
               f"--controls-of {verdict_path} carries no primary verdict")
    qc.require(verdict["primary"]["status"] == "Complete",
               f"the primary of record {verdict_path} did not complete (stage {verdict['primary']['status']})")
    qc.require(verdict["campaign_sha256"] == sha256(campaign_path)
               and verdict["identities_sha256"] == manifest["identities_sha256"] == qc.plan_sha256(manifest["identities"]),
               f"{verdict_path} does not bind {campaign_path}")
    qc.require(verdict["family"] == manifest["plan"]["family"] == family,
               f"the primary of record {verdict_path} is family {verdict['family']}, not {family}")
    harness = manifest["identities"]["harness"]
    return {"verdict": {"path": str(verdict_path), "sha256": sha256(verdict_path)},
            "campaign": {"path": str(campaign_path), "sha256": verdict["campaign_sha256"]},
            "family": family, "verdict_of_record": verdict["verdict"], "campaign_status": verdict["campaign_status"],
            "stages": {stage["workload"]: stage["status"] for stage in manifest["stages"]},
            "primary_workload": verdict["primary_workload"], "effects": verdict["primary"]["effects"],
            "plan_sha256": manifest["plan_sha256"], "identities_sha256": manifest["identities_sha256"],
            "identity": q3_identity.controls_identity(manifest["plan"], manifest["identities"]),
            "harness": {"git_commit": harness["git_commit"], "sources": harness["sources"]}}


def arm_cases(args, family):
    paths = {"stock": args.hf2q_case or HERE / "q5_cases" / f"{family}.json",
             "experiment": args.hf2q_case or HERE / "q5_cases" / f"{family}.json"}
    if args.peer_case or not args.peer_unavailable_evidence:
        paths["peer"] = args.peer_case or HERE / "q5_cases" / f"peer-{family}.json"
        if args.peer_matched_case:
            paths["peer_matched"] = args.peer_matched_case
    return {arm: Path(path).resolve() for arm, path in paths.items()}


def build_plan(args, environ, family):
    """(plan, cases). The plan is everything a resume must reproduce exactly."""
    qc.require("HF2Q_Q5_HF2Q_BIN" not in environ, "set the hf2q binaries with --stock-bin/--experiment-bin, "
               "not HF2Q_Q5_HF2Q_BIN (it would apply to both hf2q arms)")
    record = None if args.controls_of is None else primary_of_record(args.controls_of, family)
    cases, arms = {}, {}
    explicit = {"stock": [f"HF2Q_BIN={args.stock_bin}"] if args.stock_bin else [],
                "experiment": [f"HF2Q_BIN={args.experiment_bin}"] if args.experiment_bin else []}
    extra = {"stock": args.stock_var, "experiment": args.experiment_var, "peer": args.peer_var,
             "peer_matched": args.peer_matched_var}
    qc.require(not any(v.startswith("HF2Q_BIN=") for arm in ("stock", "experiment") for v in extra[arm] or []),
               "set the hf2q binaries with --stock-bin/--experiment-bin")
    for arm, path in arm_cases(args, family).items():
        cases[arm] = q5_case.validate(q5_case.loads(path.read_bytes()))
        kind = qc.engine_kind(cases[arm])
        qc.require(cases[arm]["family"] == family, f"{arm} case {path} is not family {family}")
        qc.require(kind in ARM_KINDS[arm], f"{arm} case {path} has engine kind {kind}")
        qc.require(CLIENT in cases[arm]["client"]["argv"], f"{arm} case must run the canonical four-agent client")
        arms[arm] = {"case": str(path), "case_sha256": sha256(path), "engine_kind": kind, "role": ROLES[arm],
                     "overrides": qc.overrides(cases[arm], explicit.get(arm, []) + (extra[arm] or []), environ)}
        if native_peers.is_native(kind):
            module = native_peers.adapter(kind)
            arms[arm].update(role=module.q3_role(cases[arm]), label=module.label(cases[arm]))
    native = native_peers.peer_kind({"arms": arms})
    evidence, comparison = bool(args.peer_unavailable_evidence), bool(args.native_comparison)
    qc.require((native is None and not comparison and not (args.peer_case and evidence))
               or (native is not None and evidence != comparison),
               native_peers.adapter(native).PEER_CASE_REFUSAL if native is not None else
               "a native --peer-case needs exactly one of --peer-unavailable-evidence (no runnable same-GGUF peer, "
               "ADR-063 Q4) or --native-comparison (a separately labeled campaign beside the verdict of record), "
               "and neither admits a same-GGUF peer case")
    if native is not None:
        module = native_peers.adapter(native)
        qc.require((module.SAME_GGUF_UNAVAILABLE if evidence else module.NATIVE_COMPARISON) is not None,
                   f"the native {native} arm does not run with "
                   + ("--peer-unavailable-evidence" if evidence else "--native-comparison"))
    for arm in ("stock", "experiment"):  # without --<arm>-bin the case default would silently stand in
        arms[arm]["overrides"].setdefault("HF2Q_BIN", None)
    ambient = {k[len(qc.OVERRIDE):] for k in environ if k.startswith(qc.OVERRIDE)}
    unused = sorted(ambient - {n for case in cases.values() for n in case.get("variables", {})})
    qc.require(not unused, "ambient overrides name no variable of any arm case: " + ",".join(unused))
    evidence = [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in args.peer_unavailable_evidence or []]
    workloads = parse_workloads(args.workloads, record is not None)
    done = [key for key in workloads if record and record["stages"].get(key) == "Complete"]
    qc.require(not done, f"--workloads {','.join(done)}: Complete in the campaign of record "
               f"{record and record['campaign']['path']}; a completed stage is never re-run")
    plan = {"schema": SCHEMA, "family": family, "primary": None if record else PRIMARY, "workloads": workloads,
            "measured_arms": [arm for arm in ARMS if arm in arms], "arms": arms,
            "peer_unavailable_evidence": evidence,
            "design": {"metric": "client summary own_output_four_task_makespan_seconds",
                       "estimator": "mean of per-block ln(experiment/reference), 95% t interval over complete blocks",
                       "blocks": "Williams sequences over the measured arms", "min_blocks": q3_stats.MIN_BLOCKS,
                       "max_blocks": q3_stats.MAX_BLOCKS, "half_width_pp": q3_stats.HALF_WIDTH_PP,
                       "peer_margin_ratio": q3_stats.PEER_MARGIN, "fairness_neutral_ratio": q3_stats.NEUTRAL_RATIO,
                       "warmup": "one discarded run per measured arm per stage per driver session"},
            "declared": {"stock": {"label": args.stock_label, "commit": args.stock_commit, "binary": args.stock_bin},
                         "experiment": {"label": args.experiment_label, "commit": args.experiment_commit,
                                        "binary": args.experiment_bin},
                         "qe_identity": args.qe_identity}}
    if comparison:
        plan["native_comparison"] = native_peers.adapter(native).NATIVE_COMPARISON
    if args.negative_control_receipt is not None:
        receipt = Path(args.negative_control_receipt).resolve()
        plan["negative_control_receipt"] = {"path": str(receipt), "sha256": sha256(receipt)}
    if record is not None:
        plan["controls_of"] = record
    return plan, cases


def schedule_preview(plan):
    arms = plan["measured_arms"]
    return {key: {"phase_plan": WORKLOADS[key], "warmup_order": q3_stats.block_order(arms, 1),
                  "blocks": [{"block": b, "order": q3_stats.block_order(arms, b)} for b in range(1, q3_stats.MAX_BLOCKS + 1)],
                  "stop": f"after block >= {q3_stats.MIN_BLOCKS} when every required contrast has 95% half-width "
                          f"<= {q3_stats.HALF_WIDTH_PP} points; inconclusive at {q3_stats.MAX_BLOCKS}"}
            for key in plan["workloads"]}


# --- driver ------------------------------------------------------------------------------------


class Halt(Exception):
    pass


class Driver:
    def __init__(self, plan, cases, out, environ, runner, thermal, identify):
        self.plan, self.cases, self.out, self.environ = plan, cases, out, environ
        self.runner, self.thermal, self.identify = runner, thermal, identify
        self.manifest = None

    def save(self):
        save(self.out / "campaign.json", self.manifest)

    def stop(self, status, entry, reason, resumable):
        self.manifest.update(status=status, halted={"entry": entry and entry["name"], "reason": reason, "resumable": resumable})
        self.save()
        raise Halt(reason)

    def case_path(self, stage, arm):
        return self.out / "cases" / f"{SLUGS[stage['workload']]}-{arm}.json"

    def new_entry(self, stage, kind, block, arm):
        seq = self.manifest["next_seq"]
        self.manifest["next_seq"] += 1
        tag = f"w{self.session:02d}" if kind == "warmup" else f"b{block:02d}"
        return {"name": f"{SLUGS[stage['workload']]}-{seq:03d}-{tag}-{arm}", "arm": arm, "kind": kind, "block": block,
                "session": self.session, "attempts": [], "outcome": None, "failure_kind": None, "reason": None,
                "metrics": None}

    def active(self, arms):
        return [arm for arm in arms if arm not in self.manifest["unavailable"]]

    def start(self, resume):
        path = self.out / "campaign.json"
        frozen = self.identify()
        digest = qc.plan_sha256(self.plan)
        if resume:
            self.manifest = q3_metrics.load_json(path)
            qc.require(self.manifest is not None and self.manifest.get("plan_sha256") == digest,
                       "resume needs the identical plan (cases, overrides, workloads, declarations, evidence)")
            qc.require(self.manifest["status"] not in TERMINAL and (self.manifest.get("halted") or {}).get("resumable", True),
                       f"campaign is {self.manifest['status']}: {(self.manifest.get('halted') or {}).get('reason')}")
            drift = q3_identity.diff(self.manifest["identities"], frozen)
            if drift:
                self.manifest["identity_drift"] = drift
                self.stop("IdentityDrift", None, "identities changed since campaign start: " + ", ".join(drift[:8]), False)
        else:
            qc.require(not self.out.exists(), "campaign output must be a new directory")
            qc.require(not any((p / ".git").exists() for p in (self.out.parent.resolve(), *self.out.parent.resolve().parents)),
                       "campaign output must be outside any Git tree")
            problems = q3_identity.launch_problems(self.plan, frozen)
            qc.require(not problems, "not launchable: " + "; ".join(problems))
            self.out.mkdir(mode=0o700)
            for name in ("cases", "runs", "logs"):
                (self.out / name).mkdir(mode=0o700)
            unavailable = {}
            if self.plan["peer_unavailable_evidence"] and "peer" not in self.plan["measured_arms"]:
                unavailable["peer"] = {"reason": "declared from prior evidence", "evidence": self.plan["peer_unavailable_evidence"]}
            self.manifest = {"schema": SCHEMA, "plan": self.plan, "plan_sha256": digest, "identities": frozen,
                             "identities_sha256": qc.plan_sha256(frozen), "created_utc": now(), "status": "Running",
                             "halted": None, "sessions": [], "next_seq": 0, "unavailable": unavailable,
                             "independent_review_required": True,
                             "stages": [{"workload": key, "phase_plan": WORKLOADS[key], "agents": 1 if key == "n1" else 4,
                                         "status": "Pending", "stop": None, "warmups": [], "blocks": []}
                                        for key in self.plan["workloads"]]}
            for stage in self.manifest["stages"]:
                for arm in self.plan["measured_arms"]:
                    self.case_path(stage, arm).write_text(json.dumps(derive_case(self.cases[arm], stage["workload"]), indent=2) + "\n")
        self.frozen = self.manifest["identities"]
        self.session = len(self.manifest["sessions"])
        self.manifest["sessions"].append({"index": self.session, "started_utc": now()})
        self.manifest.update(status="Running", halted=None)
        for stage in self.manifest["stages"]:
            for entry in stage["warmups"] + [e for block in stage["blocks"] for e in block["entries"]]:
                if entry["attempts"] and entry["attempts"][-1]["outcome"] == "Launched":
                    record = entry["attempts"][-1]  # the driver ended during this run: score what is on disk
                    self.settle(stage, entry, record, None, None)
                    self.consequences(stage, entry, record, halt_unstarted=False)
                if entry["kind"] == "warmup" and entry["outcome"] is None:
                    entry["outcome"] = "Abandoned"  # a resumed session runs fresh warmups
        self.save()

    def run(self, resume=False):
        try:
            self.start(resume)
            for stage in self.manifest["stages"]:
                if stage["status"] in ("Complete", "Stopped", "NotRun"):
                    continue
                if self.manifest["status"] == "Complete":
                    stage["status"] = "NotRun"
                    continue
                stage["status"] = "Running"
                self.run_stage(stage)
            self.manifest["status"] = "Complete"
            self.save()
        except Halt:
            pass
        return self.manifest

    def finished(self, stage):
        """Apply the stopping rule once the last block is complete; True when the stage is done."""
        if not stage["blocks"] or not q3_report.block_complete(stage["blocks"][-1]):
            return False
        reason = q3_stats.stop_decision(len(stage["blocks"]), q3_report.required_effects(stage, self.manifest))
        if reason is not None:
            stage.update(status="Complete", stop={"reason": reason, "blocks": len(stage["blocks"])})
            self.save()
        return reason is not None

    def run_stage(self, stage):
        arms = self.plan["measured_arms"]
        if self.finished(stage):
            return
        if not any(w["session"] == self.session for w in stage["warmups"]):
            order = self.active(q3_stats.block_order(arms, len(stage["blocks"]) + 1))
            stage["warmups"] += [self.new_entry(stage, "warmup", None, arm) for arm in order]
            self.save()
        for entry in stage["warmups"]:
            if entry["session"] == self.session and entry["outcome"] is None:
                if entry["arm"] in self.manifest["unavailable"]:
                    entry.update(outcome="Skipped", reason=f"{entry['arm']} became unavailable")
                elif not self.step(stage, entry):
                    return
        while True:
            block = stage["blocks"][-1] if stage["blocks"] else None
            if block is None or q3_report.block_complete(block):
                if self.finished(stage):
                    return
                number = len(stage["blocks"]) + 1
                order = self.active(q3_stats.block_order(arms, number))
                block = {"block": number, "order": order, "entries": [self.new_entry(stage, "measured", number, a) for a in order]}
                stage["blocks"].append(block)
                self.save()
            for entry in block["entries"]:
                if entry["outcome"] is None and entry["arm"] in self.manifest["unavailable"]:
                    entry.update(outcome="Skipped", reason=f"{entry['arm']} became unavailable")
                elif entry["outcome"] is None and not self.step(stage, entry):
                    return

    def step(self, stage, entry):
        """Run one entry to a settled outcome; False when its stage stops (experiment failure)."""
        drift = q3_identity.diff(self.frozen, self.identify())
        if drift:
            self.manifest["identity_drift"] = drift
            self.stop("IdentityDrift", entry, "identities changed since campaign start: " + ", ".join(drift[:8]), False)
        case_path = self.case_path(stage, entry["arm"])
        name = f"{entry['name']}-a{len(entry['attempts'])}"
        output = self.out / "runs" / name
        record = {"name": name, "output": str(output), "case": str(case_path), "case_sha256": sha256(case_path),
                  "argv": qc.q5_argv(case_path, output, self.plan["negative_control_receipt"]["path"]),
                  "outcome": "Launched", "started_utc": now()}
        entry["attempts"].append(record)
        self.save()
        logs = [self.out / "logs" / f"{name}.{stream}" for stream in ("stdout", "stderr")]
        with open(logs[0], "xb") as stdout, open(logs[1], "xb") as stderr:
            code = self.runner(record["argv"], qc.arm_environ(self.environ, self.plan["arms"][entry["arm"]]["overrides"]),
                               stdout, stderr)
        self.settle(stage, entry, record, code, self.thermal())
        self.save()
        return self.consequences(stage, entry, record, halt_unstarted=True)

    def consequences(self, stage, entry, record, halt_unstarted):
        """Halt or record what a settled attempt implies; False when its stage stops (experiment failure)."""
        if record.get("identity_drift"):
            self.stop("IdentityDrift", entry, "run receipt contradicts the frozen identities: "
                      + ", ".join(record["identity_drift"]), False)
        if record["outcome"] == "Invalid":
            entry.update(outcome="Invalid", reason=record["reason"])
            self.stop("Invalid", entry, f"{entry['name']} is Invalid ({record['reason']}); no reruns after the task "
                      "started, so start a new campaign", False)
        if entry["outcome"] is None:
            if halt_unstarted:
                self.stop("Halted", entry, f"{entry['name']} did not start: {record['reason']}", True)
            return True  # a resumed driver retries it with a new attempt
        if entry["outcome"] == "Unavailable":
            self.manifest["unavailable"][entry["arm"]] = {"reason": f"{record['failure_kind']}: {record['reason']}",
                                                          "evidence": [record["output"]]}
        if entry["arm"] == "experiment" and entry["outcome"] == "Fail":
            stage.update(status="Stopped", stop={"reason": "experiment_failed", "blocks": len(stage["blocks"])})
            self.manifest["status"] = "Complete"  # Failed is terminal for the family; later stages do not run
            self.save()
            return False
        self.save()
        return True

    def settle(self, stage, entry, record, code, post_thermal):
        """Score an attempt from its Q5 receipt and client summary on disk."""
        output = Path(record["output"])
        case = json.loads(Path(record["case"]).read_text())
        receipt = q3_metrics.load_json(output / "receipt.json")
        summary_path = Path(q5_case.expand(case["client"]["summary"], {"OUT": str(output)}))
        summary = q3_metrics.load_json(summary_path)
        audit = None
        if not output.exists():
            outcome, kind, reason = "NotStarted", "other", "Q5 never created the run directory"
        else:
            outcome, kind, reason = q3_metrics.classify(receipt, summary, False)
            if outcome in ("Fail", "NotStarted"):
                outcome, kind, reason = q3_metrics.classify(receipt, summary, q3_metrics.has_oom_marker(output / "server.stderr"))
        if outcome == "Pass":
            # A claimed Pass counts only after its raw client evidence re-derives it (ADR-063 Q2).
            audit, failure = q2_run_audit.check_run(receipt, summary_path.parent, summary, stage["phase_plan"],
                                                   self.plan["arms"][entry["arm"]]["engine_kind"], require_width=True)
            if failure is not None:
                outcome, kind, reason = "Invalid", None, failure
        metrics = q3_metrics.run_metrics(output, receipt, summary, stage["agents"], post_thermal)
        metrics["audit"] = audit
        record.update(q5_exit=code, outcome=outcome, failure_kind=kind, reason=reason,
                      receipt_sha256=sha256(output / "receipt.json") if receipt is not None else None,
                      q5_verdict=None if receipt is None else receipt.get("verdict"),
                      stop=None if receipt is None else receipt.get("stop"),
                      client_status=None if summary is None else summary.get("status"),
                      settled_utc=now())
        frozen = self.frozen["arms"][entry["arm"]]
        drift = []
        if metrics["binary_sha256"] is not None and metrics["binary_sha256"] != frozen["binary"]["sha256"]:
            drift.append("binary.sha256")
        if receipt is not None and receipt.get("artifact") and frozen["artifact"] is not None:
            drift += [f"artifact.{k}" for k in ("bytes", "sha256") if receipt["artifact"].get(k) != frozen["artifact"].get(k)]
        if entry["arm"].startswith("peer") and metrics["peer_build_info"] not in (None, frozen["peer_build_info"]):
            drift.append("peer_build_info")
        if drift:
            record["identity_drift"] = drift
        peer = entry["arm"].startswith("peer")
        if outcome in ("Pass", "Fail"):
            unavailable = peer and kind in ("out_of_memory", "memory_stop")
            entry.update(outcome="Unavailable" if unavailable else outcome, failure_kind=kind, reason=reason, metrics=metrics)
        elif outcome == "NotStarted" and peer and kind == "load_failure":
            entry.update(outcome="Unavailable", failure_kind=kind, reason=reason, metrics=metrics)


def write_verdict(out, manifest, quality, review):
    path = out / "campaign.json"
    verdict = q3_report.compute_verdict(manifest, quality, review, sha256(path) if path.exists() else None)
    save(out / "verdict.json", verdict)
    return verdict


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", action="append", choices=q5_case.FAMILIES,
                        help="model family (repeatable with --dry-run; default every family in --dry-run)")
    parser.add_argument("--stock-bin", help="stock hf2q binary (the released v0.1.28 standalone binary)")
    parser.add_argument("--stock-commit", help="40-hex source commit of the stock binary")
    parser.add_argument("--stock-label", default="v0.1.28 release")
    parser.add_argument("--experiment-bin", help="experiment hf2q binary (root-built main HEAD)")
    parser.add_argument("--experiment-commit", help="40-hex source commit of the experiment binary")
    parser.add_argument("--experiment-label", default="main HEAD")
    parser.add_argument("--hf2q-case", help="hf2q Q5 case for both hf2q arms (default q5_cases/<family>.json)")
    parser.add_argument("--peer-case", help="peer best valid configuration (default q5_cases/peer-<family>.json); "
                        "or a native-engine case (native_peers.py) with --peer-unavailable-evidence or --native-comparison")
    parser.add_argument("--peer-matched-case", help="optional matched-settings peer diagnostic arm (labeled, not in the verdict)")
    parser.add_argument("--peer-unavailable-evidence", action="append", metavar="FILE",
                        help="receipt proving no runnable same-GGUF peer arm (repeatable); runs stock and "
                             "experiment only, or a native --peer-case whose arm accepts it (ds4)")
    parser.add_argument("--native-comparison", action="store_true",
                        help="the native --peer-case (TensorFold) runs beside a runnable same-GGUF peer: a separately "
                             "labeled campaign whose verdict and rows are never the verdict of record")
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}-var", action="append", metavar="NAME=VALUE",
                            help=f"HF2Q_Q5_<NAME> override for the {arm} arm only (repeatable)")
    parser.add_argument("--workloads", default=",".join(WORKLOADS),
                        help="stages in order; n4 (primary) first, or control keys only with --controls-of")
    parser.add_argument("--controls-of", type=Path, metavar="PRIMARY_VERDICT_JSON",
                        help="controls-only campaign (#369 amendment): run control stages only, as controls of this "
                             "Q3 verdict.json of record; family and arm identities must match its campaign")
    parser.add_argument("--negative-control-receipt", help="accepted NegativeControlPass receipt, passed to every run")
    parser.add_argument("--qe-identity", help="named QE session/host executing the campaign")
    parser.add_argument("--quality-verdict", action="append", default=[], metavar="MANIFEST",
                        help="quality_campaign.py manifest.json with a Q1 paired verdict on the experiment binary")
    parser.add_argument("--patch-review", help=f"{q3_report.PATCH_REVIEW_SCHEMA} file: final-source sha256 -> Pass/Fail")
    parser.add_argument("--output", type=Path, help="new campaign directory outside Git (existing with --resume)")
    parser.add_argument("--resume", action="store_true", help="continue a Halted campaign with an identical plan")
    parser.add_argument("--dry-run", action="store_true", help="print plans, identities and block schedules; launch nothing")
    parser.add_argument("--verdict", type=Path, metavar="CAMPAIGN", help="recompute CAMPAIGN/verdict.json only")
    parser.add_argument("--rows", type=Path, metavar="CAMPAIGN", help="print F2 scoreboard rows for CAMPAIGN")
    parser.add_argument("--evidence-root", type=Path, help="evidence root ($E) for --rows")
    parser.add_argument("--of-record", action="store_true",
                        help="--rows for a reviewed Q3 verdict.json of record (F2 'Q3 verdicts of record' tier)")
    parser.add_argument("--quality-gate", action="append", default=[], metavar="REFERENCE=MANIFEST",
                        help="with --of-record: quality_campaign.py manifest per reference kind (stock-hf2q, peer, "
                             "<kind>-native); a manifest not written yet is Pending (repeatable)")
    parser.add_argument("--review", type=Path, help="with --of-record: the independent review of the verdict")
    args = parser.parse_args(argv)
    if args.rows is not None and args.evidence_root is None:
        parser.error("--rows needs --evidence-root")
    if (args.of_record and args.rows is None) or ((args.quality_gate or args.review) and not args.of_record):
        parser.error("--of-record applies to --rows; --quality-gate and --review need --of-record")
    if any("=" not in item for item in args.quality_gate):
        parser.error("--quality-gate takes REFERENCE=MANIFEST")
    if args.rows is None and args.verdict is None:
        if not args.dry_run and (args.output is None or not args.family or len(args.family) != 1):
            parser.error("a campaign run needs --output and exactly one --family")
        if args.hf2q_case and args.family and len(args.family) > 1:
            parser.error("--hf2q-case applies to one family")
        if args.controls_of is not None and (not args.family or len(args.family) != 1):
            parser.error("--controls-of applies to exactly one --family")
        if (args.peer_unavailable_evidence or args.native_comparison) and args.peer_matched_case:
            parser.error("--peer-unavailable-evidence and --native-comparison run no same-GGUF peer arm")
    return args


def dry_run(args, environ):
    report = []
    for family in args.family or q5_case.FAMILIES:
        plan, cases = build_plan(args, environ, family)
        ids = q3_identity.identities(plan, cases, environ)
        report.append({"family": family, "launch_problems": q3_identity.launch_problems(plan, ids),
                       "plan_sha256": qc.plan_sha256(plan), "plan": plan, "identities": ids,
                       "same_model": q3_identity.same_model(ids, q3_identity.native_arms(plan)),
                       "schedule": schedule_preview(plan),
                       "q5_argv_template": qc.q5_argv("<OUTPUT>/cases/<workload>-<arm>.json", "<OUTPUT>/runs/<entry>-aK",
                                                      plan.get("negative_control_receipt", {}).get("path", "<NC_RECEIPT>"))})
    print(json.dumps({"valid": True, "dry_run": True, "families": report}, indent=2))
    return 0


def main(argv=None):
    args = parse_args(argv)
    environ = dict(os.environ)
    try:
        if args.rows is not None:
            gates = [tuple(item.split("=", 1)) for item in args.quality_gate]
            print(json.dumps(q3_report.rows(args.rows, args.evidence_root, args.of_record, gates, args.review), indent=2))
            return 0
        if args.verdict is not None:
            manifest = json.loads((args.verdict / "campaign.json").read_text())
            verdict = write_verdict(args.verdict, manifest, args.quality_verdict, args.patch_review)
            print(json.dumps({"verdict": verdict["verdict"], "reasons": verdict["reasons"]}, indent=2))
            return 0
        if args.dry_run:
            return dry_run(args, environ)
        plan, cases = build_plan(args, environ, args.family[0])
        out = args.output.absolute()
        driver = Driver(plan, cases, out, environ, qc.subprocess_runner, q5_host.thermal_state,
                        lambda: q3_identity.identities(plan, cases, environ))
        manifest = driver.run(resume=args.resume)
        verdict = write_verdict(out, manifest, args.quality_verdict, args.patch_review) if (out / "campaign.json").exists() else None
    except (OSError, qc.CampaignError, q5_case.CaseError, q5_artifacts.ArtifactError, ValueError, KeyError) as error:
        print(json.dumps({"valid": False, "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 2
    print(json.dumps({"campaign": str(out / "campaign.json"), "status": manifest["status"], "halted": manifest.get("halted"),
                      "verdict": None if verdict is None else verdict["verdict"],
                      "reasons": None if verdict is None else verdict["reasons"]}, indent=2))
    return 0 if manifest["status"] == "Complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
