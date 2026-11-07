#!/usr/bin/env python3
"""Freeze q2_certify inputs for one q2_traced_run.py cohort; qualify its population windows offline.

  plan     (n4 only) writes create-new into --out: initial-<run>/agent-<i>.json (canonical turn-0
           requests derived from the fixture beside this file), expected-plan-<run>.json
           (hf2q.q2.expected-plan.v1) and manifest-input-<run>.json (hf2q.q2.manifest-input.v1).
  windows  (every population) runs f2_trace.validate_trace once and f2_trace.qualify_window for
           each pre-registered window of the population; writes windows-<run>.json.

Windows are fixed before any trace body is read: the composition comes from the population,
the arrival from client CLOCK_MONOTONIC clocks minus the trace header origin (header line only):

  n4   (4,0) at barrier/release.json; (0,4) at the last agent's first turn-1 decoded delta
  n1   (1,0) at barrier/release.json; (0,1) at agent-0's first turn-1 decoded delta
  3+1  (1,3), 2+2 (2,2), 1+3 (3,1) at barrier/release-2.json (the second cohort's release)

Phase-plan and n1 runs are not q2_certify cohorts (it certifies the canonical N4 release or the
N1 smoke layout); they are qualified here, offline, for Q2 item 4.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import f2_trace
from q2_certify import POPULATION_POLICIES, BoundReader, TraceFile, digest
import q2_traced_run as traced
import qwen_n4_live_smoke as coordinator
from qwen38_n1_smoke import build_request, initial_messages
from qwen38_n1_verdict import loads, require

HERE = Path(__file__).resolve().parent
REQUIRED_SOURCES = ("q2_certify.py", "qwen38_n1_verdict.py", "qwen38_n1_smoke.py", "qwen38_n1_tools.py",
                    "qwen_n4_live_smoke.py", "coding_family.py", "f2_trace.py", "f2_semantic.py")
WINDOWS = {"n4": (((4, 0), "release"), ((0, 4), "decode")),
           "n1": (((1, 0), "release"), ((0, 1), "decode")),
           "3+1": (((1, 3), "release-2"),), "2+2": (((2, 2), "release-2"),), "1+3": (((3, 1), "release-2"),)}


def read(path):
    return loads(Path(path).read_bytes())


def write_new(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return {"path": str(Path(path).resolve()), "sha256": digest(path)}


def load_run(run):
    """The completed traced run, with its one run id checked across receipt, approval, config and trace."""
    require(run.is_absolute() and run.is_dir() and run == run.resolve(), "canonical run directory")
    receipt = read(run / "receipt.json")
    require(receipt.get("schema") == traced.RECEIPT_SCHEMA and receipt.get("status") == traced.COMPLETE
            and all(receipt.get(k) is True for k in ("protocol_complete", "final_verified", "cleanup_proved")),
            "run receipt is not a complete traced cohort")
    require(receipt.get("population") in WINDOWS, "unknown traced population")
    approval, config = read(run / "client-approval.json"), read(run / traced.CONFIG_NAME)
    with (run / traced.TRACE_NAME).open("rb") as stream:
        header = json.loads(stream.readline(f2_trace.EVENT_LIMIT + 1))["payload"]
    require(receipt["run_id"] == approval.get("run_id") == config.get("run_id") == header.get("run_id"),
            "receipt, client approval, trace config and trace header must share one run id")
    require(header.get("identity") == config.get("identity") == receipt.get("identity"),
            "trace header identity differs from the frozen identity")
    require(approval.get("engine_kind") == "hf2q" and approval.get("server_pid") == receipt.get("server_pid"),
            "hf2q client approval bound to the launched server")
    # Q2 windows and certification are Qwen3.8 only: scoring DeepSeek-V4/Gemma windows waits for the F1
    # multi-owner-epoch rule (ADR-063, Q2 v3). Other traced families are read with f2_trace/q4_trace_attribution.
    require(approval.get("family") == "qwen" and
            config.get("native_observer", {}).get("witness_manifest", {}).get("input_sources") ==
            traced.WITNESS_SOURCES["qwen38"], "Q2 freezing takes a Qwen3.8 traced run only")
    return {"run": run, "receipt": receipt, "approval": approval, "config": config, "header": header}


def arrival_ns(info, rule):
    """Client CLOCK_MONOTONIC instant of a window rule, relative to the trace origin."""
    client = info["run"] / "client"
    if rule in ("release", "release-2"):
        release = read(client / "barrier" / ("release.json" if rule == "release" else "release-2.json"))
        require(release.get("f2_clock_id") == "CLOCK_MONOTONIC", "release clock domain")
        at = release["release_monotonic_ns"]
    else:
        require(rule == "decode", "unknown window rule")
        firsts = []
        for agent in coordinator.active_agents(info["receipt"]["phase_plan"]):
            turn = read(client / agent / "summary.json")["turns"][1]
            require(turn.get("f2_clock_id") == "CLOCK_MONOTONIC" and turn.get("progress_event_monotonic_ns"),
                    f"{agent} turn 1 has no decoded-delta clock")
            firsts.append(turn["progress_event_monotonic_ns"][0])
        at = max(firsts)
    origin = info["header"]["origin_monotonic_ns"]
    require(type(at) is int and at >= origin, "window arrival precedes the trace origin")
    return at - origin


def scenario(info, policy=None, composition=None, arrival=None):
    config = info["config"]
    value = {"schema": "hf2q.f2.trace-expectation.v1", "run_id": info["receipt"]["run_id"],
             "config_sha256": digest(info["run"] / traced.CONFIG_NAME),
             "metadata_budget_profile": config["metadata_budget_profile"],
             "forward_witness_manifest": config["native_observer"]["witness_manifest"]}
    if policy is not None:
        require(policy in POPULATION_POLICIES, "population policy must be one of " + ", ".join(POPULATION_POLICIES))
        value.update(composition=list(composition), arrival_ns=arrival, population_policy=policy)
    return value


def initial_request(model, run_id, agent):
    """Same derivation as the client's turn 0, from the canonical fixture beside the auditor."""
    fixture = read(HERE / "fixtures" / "canonical-coding-requests.json")
    initial = fixture[0]["request"]
    messages = initial_messages(initial["messages"], coordinator.identity(agent, run_id, None), None)
    for tool in initial["tools"]:
        tool["function"]["strict"] = False
    return build_request(model, messages, initial["tools"], 0, "qwen", "hf2q")


def freeze_plan(info, out, policy):
    require(info["receipt"]["population"] == "n4",
            "q2_certify certifies the canonical N4 release only; use `windows` for n1 and phase-plan populations")
    run, approval = info["run"], info["approval"]
    composition, rule = WINDOWS["n4"][0]
    directory = out / f"initial-{run.name}"
    directory.mkdir()
    initial = {agent: write_new(directory / f"{agent}.json", initial_request(approval["model"], approval["run_id"], agent))
               for agent in coordinator.AGENTS}
    plan = {"schema": "hf2q.q2.expected-plan.v1", "cohort_id": f"adr063-q2-traced-{run.name}",
            "identity": info["config"]["identity"],
            "source_files": {name: digest(HERE / name) for name in REQUIRED_SOURCES},
            "family": approval["family"], "engine_kind": "hf2q", "model": approval["model"],
            "server_pid": approval["server_pid"], "run_id": approval["run_id"], "required_cardinality": 4,
            "initial_requests": initial, "barrier_token": approval["barrier_token"],
            "trace_scenario": scenario(info, policy, composition, arrival_ns(info, rule)), "qe_attestation": None}
    plan_ref = write_new(out / f"expected-plan-{run.name}.json", plan)
    bound = lambda p: {"path": str(p), "sha256": digest(p)}  # noqa: E731
    spec = {"schema": "hf2q.q2.manifest-input.v1", "expected_plan": plan_ref, "client_root": str(run / "client"),
            "server_receipt": bound(run / "receipt.json"), "readiness": bound(run / "readiness.json"),
            "trace": bound(run / traced.TRACE_NAME)}
    return {"expected_plan": plan_ref, "manifest_input": write_new(out / f"manifest-input-{run.name}.json", spec)}


def qualify(info, out, policy):
    """validate_trace once, then qualify_window per pre-registered window of the population."""
    windows = [(composition, rule, arrival_ns(info, rule)) for composition, rule in WINDOWS[info["receipt"]["population"]]]
    path = info["run"] / traced.TRACE_NAME
    trace = TraceFile(BoundReader(), {"path": str(path), "sha256": digest(path)})
    report = f2_trace.validate_trace(trace, info["config"]["identity"], scenario(info))
    require(report.get("qualified") is True, "trace did not validate")
    submissions = report.pop("submissions", None)
    report["native_witness_submissions"] = None if submissions is None else len(submissions)
    results = [{"composition": list(composition), "rule": rule, "arrival_ns": arrival, "population_policy": policy,
                **f2_trace.qualify_window(trace, composition, arrival, population_policy=policy)}
               for composition, rule, arrival in windows]
    value = {"schema": "hf2q.q2.traced-windows.v1", "run": str(info["run"]), "run_id": info["receipt"]["run_id"],
             "population": info["receipt"]["population"], "trace_sha256": trace.sha256,
             "validate_trace": report, "windows": results}
    return write_new(out / f"windows-{info['run'].name}.json", value), results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "windows"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--population-policy", required=True, choices=POPULATION_POLICIES)
    args = parser.parse_args(argv)
    require(args.out.is_absolute() and args.out.is_dir() and args.out == args.out.resolve(), "canonical --out directory")
    info = load_run(args.run)
    if args.command == "plan":
        print(json.dumps(freeze_plan(info, args.out, args.population_policy), indent=1))
        return 0
    ref, results = qualify(info, args.out, args.population_policy)
    print(json.dumps({"windows": ref, "status": [(w["composition"], w["status"]) for w in results]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
