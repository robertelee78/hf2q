#!/usr/bin/env python3
"""C2 case/plan preparation and raw statistical replay.

Driver reuses Q3 scheduling and the ADR-065 bounded attempt/cooldown mechanism.
The release controller supplies its source-bound owned runner; this CLI never
launches unmanaged models. A statistical Pass is not whole-C2 qualification.
"""
import argparse
import json
import os
from pathlib import Path
import re

import adr065_campaign
import c2_evidence
import c2_observation
import c2_stats
import coding_divergent_edit
import q3_campaign as q3
import q3_identity
import q3_mechanism
import q3_metrics
import q3_report
import q3_stats
import q5_admission
import q5_case
import q5_calibration
import quality_campaign as qc

HERE = Path(__file__).resolve().parent
SCHEMA = "hf2q.adr063.c2-campaign.v1"
ARMS = {"A": 1, "B": 4}
ROUTES = ("legacy-off", "legacy-mtp-auto", "current-default")
require = c2_observation.require
sha = c2_observation.sha


def read_case(path):
    path = Path(path).resolve()
    return path, q5_case.loads(path.read_bytes())


def derive_case(base, depth, kind, route, capture_bytes=None):
    q5_case.validate(base)
    require(base["family"] in ("qwen38", "qwen36") and depth in (1, 4) and
            kind in ("append", "divergent") and route in ROUTES, "invalid C2 case axes")
    case = json.loads(json.dumps(base))
    argv = case["client"]["argv"]
    require(q3.CLIENT in argv and "--phase-plan" not in argv and "--instance" not in argv and
            "--c2-server-log" not in argv and "--workload" not in argv, "unmodified canonical N4 base required")
    if case["family"] == "qwen36":
        require("--repair-turns" in argv and argv[argv.index("--repair-turns") + 1] == "1" and
                case["client"]["approval"]["set"].get("repair_turns") == 1,
                "Qwen3.6 C2 uses the declared repair-one workload")
    bound = q5_case.loads((HERE / "q5_cases" / f'{case["family"]}-c2.json').read_bytes())["workload"]
    quota, owners = bound["anchor_bytes_per_slot_max"], bound["anchors_per_slot"] + 1
    require(type(quota) is int and quota > 0 and quota % owners == 0, "exact source anchor bound required")
    reservation = (quota // owners) * (depth + 1)
    if route == "legacy-off":
        require(capture_bytes in (None, 0), "off control has no speculative capture declaration")
        capture_bytes = 0
    else:
        require(type(capture_bytes) is int and capture_bytes > 0,
                "actual source-derived speculative capture bytes must be supplied; no guessed 2.25GiB default")
    env = case["serve"]["env"]
    env["HF2Q_QWEN_ANCHOR_DEPTH"] = str(depth)
    env["HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX"] = str(reservation)
    if route != "current-default":
        env["HF2Q_DFLASH2"] = "0"
        env["HF2Q_QWEN_SPECULATION"] = "auto" if route == "legacy-mtp-auto" else "off"
        case["variables"]["DFLASH2_POLICY"] = "off"
        case["variables"]["DFLASH2_PROFILE"] = ""
        case["variables"]["DFLASH2_QUALIFICATION"] = ""
    else:
        require({"DFLASH2_POLICY", "DFLASH2_PROFILE", "DFLASH2_QUALIFICATION"} <= case["variables"].keys(),
                "current default needs the canonical resolved-route readiness contract")
        case["variables"].update(DFLASH2_POLICY="auto", DFLASH2_PROFILE="full", DFLASH2_QUALIFICATION="pinned_pair")
    case["variables"]["LOG_FILTER"] = "hf2q=info,mlx_native=info," + c2_observation.TARGET + "=debug"
    env["LOG_FILTER"] = "{LOG_FILTER}"
    argv += ["--c2-server-log", "{OUT}/server.stderr"]
    case["client"]["approval"]["set"].update(
        c2_server_log="{OUT}/server.stderr",
        c2_observation_sources={"sha256_of": ["{REPO}/scripts/four_agent/c2_observation.py"]})
    case["workload"].update(anchors_per_slot=depth, anchor_bytes_per_slot_max=reservation,
                            speculation_capture_bytes=capture_bytes)
    case["workload"]["id"] += f"-c2-k{depth}-{route}"
    case["notes"] = list(case.get("notes", [])) + [
        f"C2 {route}, K={depth}; committed+pending reservation {reservation}/slot; capture {capture_bytes} bytes. "
        "Fresh source-matched Q5 and actual resolved dispatch observations are required."]
    if kind == "divergent":
        case = coding_divergent_edit.derive_case(case)
    return q5_case.validate(case)


def comparable(cases):
    """Only K, its declared reservation/id and notes differ; no artifact/workload/route shopping."""
    normalized = []
    for arm, depth in ARMS.items():
        case = json.loads(json.dumps(cases[arm]))
        require(case["workload"].get("anchors_per_slot") == depth and
                case["serve"]["env"].get("HF2Q_QWEN_ANCHOR_DEPTH") == str(depth), "C2 arm depth/declaration mismatch")
        require(case["serve"]["env"].get("HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX") ==
                str(case["workload"]["anchor_bytes_per_slot_max"]), "C2 reservation differs from worker quota")
        case.pop("notes", None)
        for key in ("id", "anchors_per_slot", "anchor_bytes_per_slot_max"):
            case["workload"].pop(key)
        for key in ("HF2Q_QWEN_ANCHOR_DEPTH", "HF2Q_QWEN_ANCHOR_BYTES_PER_SLOT_MAX"):
            case["serve"]["env"].pop(key)
        normalized.append(case)
    require(normalized[0] == normalized[1], "C2 arms differ beyond the declared anchor kill switch/reservation")


def preflight_controls(case, overrides, route):
    """Reject unavailable telemetry or a different control route before any model load."""
    q3_mechanism.freeze_telemetry(case, overrides)
    builtins = q5_calibration.builtins(Path("/c2-preflight"))
    plan = q5_case.build_plan(case, builtins, qc.arm_environ({}, overrides))
    controls = q5_case.child_env({}, plan["serve"]["env"])
    require("HF2Q_ADR065_STAGE_PROFILE" not in controls and "MLX_PROFILE_CB" not in controls,
            "C2 timing requires stage and command-buffer profiling unset")
    if route == "current-default":
        require("HF2Q_DFLASH2" not in controls,
                "current-default must exercise the unset DFlash2 policy")
    else:
        require(controls.get("HF2Q_DFLASH2") == "0" and
                controls.get("HF2Q_QWEN_SPECULATION") == ("off" if route == "legacy-off" else "auto"),
                "C2 declared legacy route differs from resolved serving controls")


def build_plan(args, environ):
    require(re.fullmatch(r"[0-9a-f]{40}", args.commit) is not None and args.qe_identity,
            "exact binary source commit and named release driver required")
    cases, arms, calibrations = {}, {}, {}
    binary = str(Path(args.binary).resolve())
    for arm, depth in ARMS.items():
        path, case = read_case(args.k1_case if depth == 1 else args.k4_case)
        q5_case.validate(case)
        cases[arm] = case
        require(case["family"] == args.family and qc.engine_kind(case) == "hf2q", "same explicit Qwen family/engine required")
        require("--c2-server-log" in case["client"]["argv"], "case lacks real C2 collection")
        require(("--workload" in case["client"]["argv"]) == (args.kind == "divergent"), "C2 workload kind mismatch")
        if args.kind == "divergent":
            require(case["client"]["argv"][case["client"]["argv"].index("--workload") + 1] == coding_divergent_edit.ID,
                    "pinned original divergent-edit workload required")
        overrides = qc.overrides(case, ["HF2Q_BIN=" + binary, *args.set], environ)
        require(overrides.get("HF2Q_BIN") == binary, "C2 binary override contradicts declared binary")
        preflight_controls(case, overrides, args.route)
        arms[arm] = {"case": str(path), "case_sha256": sha(path), "engine_kind": "hf2q",
                     "role": f"K={depth}, same binary", "overrides": overrides}
        calibration = Path(args.k1_calibration if depth == 1 else args.k4_calibration).resolve()
        calibrations[arm] = {"path": str(calibration), "sha256": sha(calibration)}
    comparable(cases)
    negative = Path(args.negative_control_receipt).resolve()
    q5_admission.negative_control_latency(c2_evidence.load(negative))
    plan = {"schema": SCHEMA, "mode": "c2", "family": args.family,
            "harness_root": str(HERE.parent.parent),
            "workloads": ["n4"], "measured_arms": list(ARMS), "arms": arms,
            "peer_unavailable_evidence": [], "negative_control_receipt": {"path": str(negative), "sha256": sha(negative)},
            "declared": {"binary": binary, "commit": args.commit, "qe_identity": args.qe_identity},
            "c2": {"kind": args.kind, "route": args.route, "calibrations": calibrations,
                   "scope": "paired C2 statistical component; exactness/transcripts/quality/cancel/owned guards separate"},
            "cooldown_seconds": 60,
            "budget": {"max_attempts": 42, "retries": 0},
            "design": {"min_blocks": 5, "max_blocks": 20, "half_width_pp": 2.5,
                       "order": "q3_stats.block_order: alternating A,B then B,A",
                       "warmup": "one discarded full cohort per arm per session"}}
    return plan, cases


def identities(plan, cases, environ):
    value = q3_identity.identities(plan, cases, environ)
    value["c2_calibration_files"] = {arm: sha(reference["path"]) for arm, reference in plan["c2"]["calibrations"].items()}
    return value


def paired_blocks(stage):
    blocks = []
    for block in stage["blocks"]:
        if not q3_report.block_complete(block):
            continue
        require(block["order"] == q3_stats.block_order(list(ARMS), block["block"]), "C2 Williams order changed")
        entries = {entry["arm"]: entry for entry in block["entries"]}
        require(set(entries) == set(ARMS) and len(block["entries"]) == 2, "one complete cohort per arm/block required")
        identity = f'williams-{block["block"]}'
        row = {"block_id": identity}
        for arm, label in (("A", "reference"), ("B", "subject")):
            entry = entries[arm]
            require(entry["outcome"] == "Pass", "C2 cannot drop a failed or missing cohort")
            row[label] = {"block_id": identity, "outcome": "Pass", **entry["metrics"]["c2"]}
        blocks.append(row)
    return blocks


def validate_route(data, route):
    prefixes = {"legacy-off": ("ordinary:",), "legacy-mtp-auto": ("mtp:", "k3:"),
                "current-default": ("dflash2:",)}[route]
    require(any(key.startswith(prefixes) and value > 0 for key, value in data["routes"].items()),
            "required actual resolved dispatch route was not exercised")
    if route != "current-default":
        require(not any(key.startswith("dflash2:") for key in data["routes"]),
                "DFlash2 active in explicitly legacy control")


class Driver(adr065_campaign.Driver):
    """Owned-runner injection is mandatory, just as for the existing release controller."""
    schema = SCHEMA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibrations = {}

    def _attempt_argv(self, stage, entry, case_path, output):
        return [*super()._attempt_argv(stage, entry, case_path, output),
                "--admission-receipt", self.plan["c2"]["calibrations"][entry["arm"]]["path"]]

    def launch_problems(self, frozen):
        problems = q3_identity.arm_problems(frozen)
        if not q3_identity.same_model(frozen):
            problems.append("C2 must serve the same artifact")
        if any(not isinstance(item.get("artifact"), dict) or
               re.fullmatch(r"[0-9a-f]{64}", item["artifact"].get("sha256") or "") is None
               for item in frozen["arms"].values()):
            problems.append("C2 requires the exact artifact SHA256, not a path/size fallback")
        if len({item["binary"]["sha256"] for item in frozen["arms"].values()}) != 1:
            problems.append("C2 must run the same binary in both arms")
        if not q3_identity.harness_clean(frozen):
            problems.append("C2 harness must be a clean source commit")
        try:
            c2_evidence.bound_json(self.plan["negative_control_receipt"])
            for arm in ARMS:
                preflight_controls(self.cases[arm], self.plan["arms"][arm]["overrides"], self.plan["c2"]["route"])
            for arm, depth in ARMS.items():
                self.calibrations[arm] = c2_evidence.calibration(
                    self.plan["c2"]["calibrations"][arm], self.cases[arm], frozen["arms"][arm], frozen["host"],
                    depth, self.plan["c2"]["kind"], self.plan["arms"][arm]["overrides"], self.plan["harness_root"])
                validate_route(self.calibrations[arm], self.plan["c2"]["route"])
        except (OSError, ValueError, KeyError, q5_admission.AdmissionError) as error:
            problems.append("fresh C2 calibration binding: " + str(error))
        return problems

    def start(self, resume):
        if resume:
            require(not self.launch_problems(self.identify()), "resumed C2 calibration/source binding invalid")
        super().start(resume)

    def settle(self, stage, entry, record, code, post_thermal):
        super().settle(stage, entry, record, code, post_thermal)
        if record["outcome"] != "Pass":
            return
        if record.get("identity_drift"):
            return
        try:
            run = Path(record["output"])
            receipt, summary = c2_evidence.load(run / "receipt.json"), c2_evidence.load(run / "client/summary.json")
            arm = entry["arm"]
            c2_evidence.run_identity(
                run, receipt, self.cases[arm], self.frozen["arms"][arm], self.frozen["host"],
                self.plan["arms"][arm]["overrides"], self.plan["harness_root"],
                self.plan["c2"]["calibrations"][arm])
            data = c2_evidence.run_evidence(run, receipt, summary, ARMS[entry["arm"]], self.plan["c2"]["kind"])
            data["calibration_peak_footprint_bytes"] = self.calibrations[entry["arm"]]["peak_footprint_bytes"]
            validate_route(data, self.plan["c2"]["route"])
            entry["metrics"]["c2"] = data
            record["c2_observation_sha256"] = data["observation_sha256"]
        except (OSError, ValueError, KeyError) as error:
            record.update(outcome="Invalid", reason="C2 raw evidence: " + str(error))
            entry.update(outcome="Invalid", reason=record["reason"])

    def consequences(self, stage, entry, record, halt_unstarted):
        if record.get("identity_drift"):
            self.stop("IdentityDrift", entry, "run receipt contradicts frozen identities: " +
                      ", ".join(record["identity_drift"]), False)
        if record["outcome"] != "Pass":
            self.stop("Invalid", entry, record.get("reason") or "C2 cohort failed; no retry or dropped position", False)
        data = entry["metrics"]["c2"]
        if any(data["eligibility"]["not_eligible"].values()):
            self.stop("Invalid", entry, "anchor-caused joint refusal during C2 cohort", False)
        if entry["arm"] == "B" and data["peak_footprint_bytes"] > 1.25 * data["calibration_peak_footprint_bytes"]:
            self.stop("Invalid", entry, "K4 footprint exceeds its fresh calibration by25%", False)
        return True

    def finished(self, stage):
        result = c2_stats.campaign(self.plan["c2"]["kind"], paired_blocks(stage))
        self.manifest["c2_statistics"] = result
        if result["status"] == "Continue":
            return False
        stage.update(status="Complete", stop={"reason": result["stop_reason"] or "hard_failure", "blocks": result["blocks"]})
        self.save()
        return True


def evaluate(path):
    manifest = c2_evidence.load(Path(path) / "campaign.json")
    plan, frozen = manifest["plan"], manifest["identities"]
    require(manifest.get("schema") == SCHEMA and plan["schema"] == SCHEMA, "not a C2 campaign")
    require(manifest.get("plan_sha256") == qc.plan_sha256(plan) and
            manifest.get("identities_sha256") == qc.plan_sha256(frozen), "C2 frozen identities/plan changed")
    require(len(manifest["stages"]) == 1 and manifest["status"] in ("Running", "Complete"),
            "one non-invalid C2 workload required")
    require(set(plan["arms"]) == set(frozen["arms"]) == set(ARMS) and q3_identity.same_model(frozen) and
            len({item["binary"]["sha256"] for item in frozen["arms"].values()}) == 1,
            "C2 replay requires exactly two same-binary, same-artifact arms")
    cases, calibrations = {}, {}
    for arm, depth in ARMS.items():
        spec = plan["arms"][arm]
        case = c2_evidence.bound_json({"path": spec["case"], "sha256": spec["case_sha256"]})
        cases[arm] = case
        preflight_controls(case, spec["overrides"], plan["c2"]["route"])
        calibrations[arm] = c2_evidence.calibration(
            plan["c2"]["calibrations"][arm], case, frozen["arms"][arm], frozen["host"], depth, plan["c2"]["kind"],
            spec["overrides"], plan["harness_root"])
        validate_route(calibrations[arm], plan["c2"]["route"])
    comparable(cases)
    stage = manifest["stages"][0]
    for entry in stage["warmups"] + [row for block in stage["blocks"] for row in block["entries"]]:
        if entry["outcome"] == "Abandoned" and not entry["attempts"]:
            continue
        if entry["outcome"] is None:
            require(not entry["attempts"], "unsettled launched C2 cohort cannot be dropped")
            continue
        require(entry["outcome"] == "Pass" and len(entry["attempts"]) == 1, "failed/retried C2 cohort cannot be dropped")
        record, arm = entry["attempts"][0], entry["arm"]
        require(record["outcome"] == "Pass", "C2 attempt did not pass")
        run = Path(record["output"])
        receipt = c2_evidence.bound_json({"path": str(run / "receipt.json"), "sha256": record["receipt_sha256"]})
        summary = c2_evidence.load(run / "client/summary.json")
        require(q3_metrics.classify(receipt, summary, False)[0] == "Pass", "C2 raw Q5/client outcome contradicted")
        c2_evidence.run_identity(run, receipt, cases[arm], frozen["arms"][arm], frozen["host"],
                                 plan["arms"][arm]["overrides"], plan["harness_root"],
                                 plan["c2"]["calibrations"][arm])
        data = c2_evidence.run_evidence(run, receipt, summary, ARMS[arm], plan["c2"]["kind"])
        validate_route(data, plan["c2"]["route"])
        data["calibration_peak_footprint_bytes"] = calibrations[arm]["peak_footprint_bytes"]
        require(data == entry["metrics"]["c2"], "C2 cached statistics differ from raw evidence")
    result = c2_stats.campaign(plan["c2"]["kind"], paired_blocks(stage))
    if result["status"] != "Continue":
        require(all(q3_report.block_complete(block) for block in stage["blocks"]),
                "C2 campaign continued into an incomplete block after its first stop")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    case = commands.add_parser("case", help="derive an independently calibrated K1/K4 C2 case")
    case.add_argument("--base-case", type=Path, required=True)
    case.add_argument("--depth", type=int, choices=(1, 4), required=True)
    case.add_argument("--speculation-capture-bytes", type=int)
    case.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan", help="freeze inputs for the owned release-controller Driver")
    plan.add_argument("--family", choices=("qwen38", "qwen36"), required=True)
    for field in ("k1-case", "k4-case", "k1-calibration", "k4-calibration", "binary", "commit", "qe-identity", "negative-control-receipt"):
        plan.add_argument("--" + field, required=True)
    plan.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    plan.add_argument("--output", type=Path, required=True)
    for item in (case, plan):
        item.add_argument("--kind", choices=("append", "divergent"), required=True)
        item.add_argument("--route", choices=ROUTES, required=True)
    review = commands.add_parser("evaluate", help="recompute retained paired statistics; no model qualification")
    review.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "case":
            _, base = read_case(args.base_case)
            value = derive_case(base, args.depth, args.kind, args.route, args.speculation_capture_bytes)
        elif args.command == "plan":
            value, _ = build_plan(args, dict(os.environ))
        else:
            print(json.dumps(evaluate(args.directory), indent=2))
            return 0
        with args.output.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "sha256": sha(args.output), "models_started": 0}))
        return 0
    except (OSError, ValueError, KeyError, IndexError, qc.CampaignError, q5_case.CaseError) as error:
        print(json.dumps({"valid": False, "error": f"{type(error).__name__}: {error}"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
