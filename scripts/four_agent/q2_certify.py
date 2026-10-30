"""Independent raw-cohort audit; no inference, subprocess, tools or import actions.
A raw protocol audit awaits independent QE: only a separately hash-bound QE
attestation supplied by the externally frozen expected plan grants Qualified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re

from qwen38_n1_verdict import (SSE, Invalid, loads, qualify, require, test_harness,
                              unary, wire)
import qwen38_n1_tools as tool_source
import qwen_n4_live_smoke as coordinator

RAW_LIMIT = 32 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{64}\Z")
IDENTITY = {"model_sha256", "server_binary_sha256", "server_source", "native_version",
            "native_checksum", "tokenizer_sha256", "template_sha256", "profile_sha256"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


class BoundReader:
    """Verify each frozen raw reference, retaining references, not executable bytes."""
    def __init__(self):
        self.refs = []
        self.limits = {}

    def path(self, bound, maximum=RAW_LIMIT):
        require(isinstance(bound, dict) and set(bound) == {"path", "sha256"}, "closed BoundFile")
        path = Path(bound["path"])
        require(path.is_absolute() and not path.is_symlink() and path.is_file()
                and path == path.resolve(), "canonical regular evidence file")
        require(isinstance(bound["sha256"], str) and SHA.fullmatch(bound["sha256"]), "BoundFile SHA")
        require(path.stat().st_size <= maximum, "raw evidence size bound")
        require(digest(path) == bound["sha256"], "raw evidence hash changed")
        self.refs.append(dict(bound))
        self.limits[str(path)] = maximum
        return path

    def raw(self, bound, maximum=RAW_LIMIT):
        data = self.path(bound, maximum).read_bytes()
        require(len(data) <= maximum and hashlib.sha256(data).hexdigest() == bound["sha256"],
                "evidence mutated during read")
        return data

    def verify(self):
        for bound in self.refs:
            path = Path(bound["path"])
            require(path.is_file() and not path.is_symlink() and path == path.resolve()
                    and path.stat().st_size <= self.limits[str(path)]
                    and digest(path) == bound["sha256"], "final frozen evidence changed")

    def obj(self, bound, maximum=RAW_LIMIT):
        return loads(self.raw(bound, maximum))


def replay_response(raw, request, turn, seen, engine, semantic_frames=None, *, assistant_policy=None):
    """Use mature actual parser; framing is identical to exchange, not a new oracle."""
    policy = {} if assistant_policy is None else {"assistant_policy": assistant_policy}
    if not request["stream"]:
        return unary(raw, request["model"], turn, seen, engine, **policy)
    parser = SSE(request["model"], turn, seen, engine, **policy)
    frame = []
    require(raw.endswith(b"\n"), "incomplete SSE line")
    for raw_line in raw.split(b"\n")[:-1]:
        line = raw_line[:-1] if raw_line.endswith(b"\r") else raw_line
        if line.startswith(b":"):
            continue
        if line:
            require(line.startswith(b"data: "), "unexpected SSE field")
            frame.append(line[6:])
        else:
            require(len(frame) <= 1, "multiline SSE data")
            if frame:
                semantic = parser.feed(frame[0].decode("utf8", "strict"))
                if semantic and semantic_frames is not None:
                    semantic_frames.append(True)
                frame = []
    require(not frame, "incomplete SSE frame")
    return parser.result()


def require_clock(value):
    require(isinstance(value, dict) and value.get("f2_clock_id") == "CLOCK_MONOTONIC",
            "actual shared timestamp clock domain")


def checked_cleanup(value):
    require(isinstance(value, dict) and value.get("verified") is True
            and value.get("reaped") is True and not value.get("errors")
            and value.get("final_observation", {}).get("available") is True
            and value["final_observation"].get("live_owned_pids") == [],
            "actual owned cleanup unknown/live")


def process_result(reader, refs, expected_binary=None):
    require(set(refs) == {"receipt", "stdout", "stderr", "binary"}, "closed process artifact")
    r = reader.obj(refs["receipt"])
    out, err = reader.raw(refs["stdout"]), reader.raw(refs["stderr"])
    require(r.get("exit_code") == 0 and r.get("cleanup") is True and not r.get("error")
            and not r.get("cleanup_error"), "actual tool process failure")
    checked_cleanup(r.get("cleanup_receipt"))
    if expected_binary is not None:
        p = reader.path(refs["binary"], 128 * 1024 * 1024)
        require(refs["binary"]["sha256"] == expected_binary and digest(p) == expected_binary,
                "actual test binary identity")
        require(r.get("argv", [])[-1:] == ["--test-threads=1"], "fixed test argv")
        test_harness(out, 2)  # Not exit-zero or model/compiler self-attestation.
    else:
        require(refs["binary"] is None, "compile executable is toolchain provenance, not test binary")
    return r, out, err


def verify_tests(reader, group, tool, fixture_manifest):
    require(set(group) == {"compile", "executions"}, "closed compiler group")
    compile_receipt, stdout, _ = process_result(reader, group["compile"])
    argv = compile_receipt.get("argv", [])
    require("test" in argv and "--offline" in argv and "--locked" in argv
            and "--no-run" in argv and "--message-format=json" in argv,
            "actual locked offline compiler command")
    require(tool["target"] == tool["arguments"]["target"], "tool target/arguments join")
    require("--manifest-path" in argv and argv[argv.index("--manifest-path") + 1] == fixture_manifest,
            "compiler used this agent's actual immutable fixture")
    require((tool["target"] == "ports" and "--test" in argv and argv[argv.index("--test") + 1] == "ports")
            or (tool["target"] == "all" and "--tests" in argv), "actual requested test scope")
    artifacts, finished, names = [], [], []
    for line in stdout.splitlines():
        item = loads(line)
        if item.get("reason") == "build-finished":
            finished.append(item.get("success"))
        if item.get("reason") == "compiler-artifact" and item.get("executable"):
            require(item.get("profile", {}).get("test") is True
                    and item.get("target", {}).get("name") in {"ports", "qwen38_n1_fixture"},
                    "actual compiler test artifact")
            artifacts.append(Path(item["executable"]))
            names.append(item["target"]["name"])
    count = 1 if tool["target"] == "ports" else 2
    require(finished == [True] and len(artifacts) == len(set(artifacts)) == count
            and len(group["executions"]) == count and len(tool["artifact_sha256"]) == count,
            "positive compiler/artifact counts")
    require(set(names) == ({"ports"} if tool["target"] == "ports" else {"ports", "qwen38_n1_fixture"}),
            "all requested positive test targets")
    len_done = 0
    for refs, expected in zip(group["executions"], tool["artifact_sha256"]):
        run, _, _ = process_result(reader, refs, expected)
        require(len(run.get("argv", [])) >= 2 and Path(run["argv"][-2]) == artifacts[len_done],
                "executed compiler-emitted target")
        len_done += 1


def audit_agent(reader, agent, expected, name):
    require(set(agent) == {"summary", "ready", "turns", "tool_results", "test_groups", "fixture_files"},
            "closed agent raw manifest")
    summary = reader.obj(agent["summary"])
    require(len(agent["turns"]) == 7 and len(agent["tool_results"]) == 6, "whole actual conversation")
    require_clock(summary)
    require(summary.get("server_pid") == expected["server_pid"]
            and summary.get("family", "qwen") == expected["family"]
            and summary.get("engine_kind", "hf2q") == expected["engine_kind"], "agent identity")
    messages, tools, receipts, seen = None, [], [], set()
    for turn, refs in enumerate(agent["turns"]):
        require(set(refs) == {"request", "response", "receipt", "assembled"}, "closed raw turn")
        request_raw = reader.raw(refs["request"])
        request = loads(request_raw)
        require(request["model"] == expected["model"] and request.get("stream") is (turn % 2 == 1),
                "actual model/transport")
        # Actual builder enforces family + peer-only reasoning controls; no mirrored policy.
        from qwen38_n1_smoke import build_request
        require(request == build_request(expected["model"], request["messages"], request["tools"],
                    turn, expected["family"], expected["engine_kind"]), "actual request builder drift")
        if expected["engine_kind"] == "hf2q":
            require(wire(request, expected["family"]) == request_raw, "canonical wire/no silent truncation")
        else:
            # Same validation boundary as actual peer exchange, retaining native metadata.
            from family_peer import canonical
            wire(canonical(request, expected["family"]), expected["family"])
            require(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
                    == request_raw, "canonical native peer wire")
        if messages is None:
            require(request == reader.obj(expected["initial_requests"][name]),
                    "externally frozen canonical initial task/schema")
            messages = request["messages"]
        else:
            require(request["messages"] == messages, "assistant/reasoning/tool result history changed")
        semantic_frames = []
        response_raw = reader.raw(refs["response"])
        message, call, stats = replay_response(response_raw, request, turn, seen,
                                               expected["engine_kind"], semantic_frames)
        require(reader.obj(refs["assembled"]) == message, "assembled assistant differs from actual wire")
        receipt = reader.obj(refs["receipt"])
        require_clock(receipt)
        require(receipt.get("turn") == turn and receipt.get("stream") is request["stream"]
                and receipt.get("usage") == stats and receipt.get("request_sha256") == refs["request"]["sha256"]
                and receipt.get("response_bytes") == len(response_raw) and receipt.get("response_sha256") == refs["response"]["sha256"],
                "actual usage/turn/response join")
        from f2_semantic import semantic_timing_v3
        require(len(receipt["semantic_event_monotonic_ns"]) == len(semantic_frames),
                "semantic timestamps must match actual raw SSE predicate events")
        require(receipt["semantic_timing_v3"] == semantic_timing_v3(
                receipt["send_monotonic_ns"], receipt["semantic_event_monotonic_ns"],
                receipt["validated_terminal_monotonic_ns"], request["stream"], True),
                "actual v3 timing/direction/unit corruption")
        receipts.append(receipt)
        messages = messages + [message]
        if call is not None:
            cid, function, args = call
            seen.add(cid)
            tool = reader.obj(agent["tool_results"][turn])
            require(tool.get("name") == function and tool.get("arguments") == args
                    and tool.get("status") == "Complete" and tool.get("cleanup") is True,
                    "actual authorized tool/argument/cleanup join")
            result = tool.get("result")
            require(isinstance(result, str) and len(result.encode()) <= 65536, "whole real tool result")
            tools.append(tool)
            messages.append({"role": "tool", "tool_call_id": cid, "name": function, "content": result})
    groups = agent["test_groups"]
    require(set(groups) == {"3", "5"}, "both actual run_tests groups required")
    for index in (3, 5):
        require(tools[index]["name"] == "run_tests" and tools[index].get("passed") is True,
                "actual test tool result")
        verify_tests(reader, groups[str(index)], tools[index],
                     agent["fixture_files"]["Cargo.toml"]["path"])
    require(set(agent["fixture_files"]) == set(tool_source.IMMUTABLE) | {"src/ports.rs"}, "all fixture files")
    for path, text in tool_source.IMMUTABLE.items():
        require(reader.raw(agent["fixture_files"][path]) == text.encode(), "read-only fixture/test modified")
    source = reader.raw(agent["fixture_files"]["src/ports.rs"])
    source_hash = hashlib.sha256(source).hexdigest()
    require(source != tool_source.INITIAL.encode(), "unchanged broken patch")
    require(qualify(receipts, tools, summary.get("cleanup"), source_hash, source_hash)
            == summary.get("status"), "real protocol/source scorer")
    require(summary.get("final_source_sha256") == source_hash, "actual final patch identity")
    if expected["required_cardinality"] == 4:
        require(summary.get("agent_id") == name and summary.get("run_id") == expected["run_id"],
                "slot/conversation cross-agent identity")
    summary = dict(summary, turns=receipts, tools=tools, observed_final_source_sha256=source_hash)
    return summary


def certify_cohort(cohort_manifest_path, expected_plan):
    require(set(expected_plan) == {"schema", "cohort_id", "identity", "source_files", "family",
            "engine_kind", "model", "server_pid", "run_id", "required_cardinality", "initial_requests", "barrier_token", "trace_scenario", "qe_attestation"},
            "externally frozen expected-plan keys")
    require(expected_plan["schema"] == "hf2q.q2.expected-plan.v1"
            and expected_plan["required_cardinality"] in (1, 4), "N1/N4 separate plan")
    require(set(expected_plan["identity"]) == IDENTITY, "complete model/binary/native/template identity")
    required_sources = {"q2_certify.py", "qwen38_n1_verdict.py", "qwen38_n1_smoke.py",
                        "qwen38_n1_tools.py", "qwen_n4_live_smoke.py", "coding_family.py", "f2_trace.py", "f2_semantic.py"}
    if expected_plan["engine_kind"] == "peer":
        required_sources |= {"qwen_n4_peer.py", "family_peer.py"}
    require(required_sources <= set(expected_plan["source_files"]), "all actual audit/production sources bound")
    require(isinstance(expected_plan["trace_scenario"], dict) and expected_plan["trace_scenario"].get("run_id") == expected_plan["run_id"],
            "externally frozen actual trace population scenario")
    reader = BoundReader()
    for name, expected in expected_plan["source_files"].items():
        require(Path(name).name == name and SHA.fullmatch(expected), "source whitelist name/SHA")
        reader.path({"path": str(Path(__file__).resolve().with_name(name)), "sha256": expected})
    require(Path(cohort_manifest_path).stat().st_size <= 1048576, "bounded cohort manifest")
    manifest = loads(Path(cohort_manifest_path).read_bytes())
    require(set(manifest) == {"schema", "cohort_id", "identity", "family", "engine_kind", "model",
            "server_pid", "run_id", "logical_clients", "server_receipt", "readiness", "client_summary",
            "agents", "release", "trace"}, "closed raw cohort manifest")
    require(manifest["schema"] == "hf2q.q2.cohort.v1", "cohort schema")
    for key in ("cohort_id", "identity", "family", "engine_kind", "model", "server_pid", "run_id"):
        require(manifest[key] == expected_plan[key], "frozen cohort identity:" + key)
    count = expected_plan["required_cardinality"]
    require(manifest["logical_clients"] == count and set(manifest["agents"]) ==
            {f"agent-{i}" for i in range(count)}, "actual cardinality, not four HTTP labels")
    outer = reader.obj(manifest["server_receipt"])
    require(all(outer.get(k) is True for k in ("protocol_complete", "final_verified", "cleanup_proved"))
            and not outer.get("failure") and not outer.get("listener_cleanup_error"), "outer failed/held/cleanup")
    require(isinstance(outer.get("cleanup"), list) and outer["cleanup"]
            and all(c.get("reaped") is True and c.get("group_empty") is True
                    and not c.get("errors") for c in outer["cleanup"]), "raw server/client process cleanup")
    ready = reader.obj(manifest["readiness"])
    if expected_plan["engine_kind"] == "peer":
        from family_peer import validate_readiness
        require(count == 4 and set(ready) == {"health", "props", "slots"}, "raw peer readiness envelope")
        validate_readiness(ready["health"], ready["props"], ready["slots"], expected_plan["model"],
                           expected_plan["family"])
    else:
        require(expected_plan["engine_kind"] == "hf2q", "closed engine family")
        require(ready["health"]["model"] == expected_plan["model"]
                and ready["health"]["context_length"] == 262144, "actual model/full context readiness")
        require(ready["runtime"]["measurement_snapshot"]["process_pid"] == expected_plan["server_pid"]
                and ready["runtime"]["serve"]["max_slots"] == count, "actual worker/slot readiness")
    results = {name: audit_agent(reader, refs, expected_plan, name)
               for name, refs in manifest["agents"].items()}
    whole = reader.obj(manifest["client_summary"])
    require_clock(whole)
    if count == 4:
        require(set(whole["exits"]) == set(results), "actual coordinator exits")
        release = reader.obj(manifest["release"])
        require_clock(release)
        require(whole["barrier"]["release_sha256"] == manifest["release"]["sha256"],
                "actual release identity")
        for name in results:
            ready_ref = manifest["agents"][name]["ready"]
            require(reader.obj(ready_ref) == whole["barrier"]["ready"][name], "raw ready/aggregate join")
            coordinator.validate_release(release, name, expected_plan["run_id"],
                expected_plan["server_pid"], expected_plan["barrier_token"], ready_ref["sha256"])
            checked_cleanup(whole["client_cleanup"][name])
        require(coordinator.aggregate(results, whole["exits"],
                {name: whole["client_cleanup"][name]["verified"] for name in results},
                expected_plan["server_pid"], expected_plan["run_id"],
                expected_plan["engine_kind"], expected_plan["family"]) == whole["status"],
                "actual four-task aggregate")
        started_ns = release["release_monotonic_ns"]
        require(type(started_ns) is int and started_ns == whole["makespan_started_ns"]
                == whole["barrier"]["release_monotonic_ns"], "actual declared cohort release clock")
        for result in results.values():
            require_clock(result["barrier"])
            require(result["barrier"]["release_monotonic_ns"] == started_ns,
                    "actual agent observes same release")
        finished_ns = max(result["task_completed_monotonic_ns"] for result in results.values())
        require(whole["makespan_finished_ns"] == finished_ns, "actual four-task completion clock")
    else:
        require(manifest["release"] is None and manifest["agents"]["agent-0"]["ready"] is None,
                "N1 has no fabricated four-client barrier")
        require(whole == reader.obj(manifest["agents"]["agent-0"]["summary"]), "actual N1 summary")
        started_ns = results["agent-0"]["turns"][0]["send_monotonic_ns"]
        finished_ns = results["agent-0"]["task_completed_monotonic_ns"]
    for result in results.values():
        require(type(result["task_completed_monotonic_ns"]) is int
                and result["task_completed_monotonic_ns"] >= result["turns"][-1]["validated_terminal_monotonic_ns"]
                and result["turns"][0]["send_monotonic_ns"] >= started_ns,
                "task clocks bound actual validated requests")
    require(type(finished_ns) is int and 0 <= started_ns < finished_ns, "real task interval")
    # F1 owns actual production trace validation; absence is Pending, never a mock PASS.
    from f2_trace import validate_trace, qualify_window
    trace = reader.obj(manifest["trace"], 256 * 1024 * 1024)
    trace_report = validate_trace(trace, expected_plan["identity"], expected_plan["trace_scenario"])
    require(isinstance(trace_report, dict) and trace_report.get("qualified") is True,
            "actual trace validator must qualify frozen progress scenario")
    scenario = expected_plan["trace_scenario"]
    window = qualify_window(trace, tuple(scenario["composition"]), scenario["arrival_ns"])
    require(window.get("status") == "Qualified", "actual admitted-slot population/progress not qualified")
    from f2_semantic import slot_semantic_summary
    report = dict(schema="hf2q.q2.certification.v1", cohort_id=manifest["cohort_id"],
                  identity=manifest["identity"], required_cardinality=count,
                  manifest_sha256=digest(cohort_manifest_path), raw_protocol_verified=True,
                  makespan_started_ns=started_ns, makespan_finished_ns=finished_ns,
                  trace=trace_report, phase_window=window,
                  per_slot_metrics={name: slot_semantic_summary(result["turns"]) for name, result in results.items()}, verified_refs=reader.refs, status="PendingIndependentQE",
                  independently_qe_qualified=False, campaign_primary_eligible=False,
                  sampling_lane="canonical_internal", profile_sha256=manifest["identity"]["profile_sha256"])
    if expected_plan["qe_attestation"] is not None:
        qe = reader.obj(expected_plan["qe_attestation"])
        require(set(qe) == {"schema", "reviewer", "independent", "verdict", "manifest_sha256",
                "auditor_sha256", "cohort_id", "identity"} and qe["schema"] == "hf2q.q2.qe-verdict.v1"
                and qe["independent"] is True and qe["verdict"] == "PASS"
                and isinstance(qe["reviewer"], str) and bool(qe["reviewer"])
                and qe["manifest_sha256"] == report["manifest_sha256"]
                and qe["auditor_sha256"] == digest(__file__)
                and qe["cohort_id"] == report["cohort_id"] and qe["identity"] == report["identity"],
                "independent QE verdict must bind exact raw cohort/source")
        report.update(status="Qualified", independently_qe_qualified=True,
                      campaign_primary_eligible=False, qe=expected_plan["qe_attestation"])
    reader.verify()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-plan", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(digest(args.expected_plan) == args.expected_plan_sha256, "frozen plan identity")
    require(Path(args.expected_plan).stat().st_size <= 1048576, "bounded externally frozen plan")
    try:
        result = certify_cohort(args.manifest, loads(Path(args.expected_plan).read_bytes()))
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = dict(schema="hf2q.q2.certification.v1", status="Rejected", failure=str(error),
                      manifest_sha256=digest(args.manifest), independently_qe_qualified=False,
                      campaign_primary_eligible=False)
    destination = Path(args.output)
    require(destination.is_absolute() and destination.parent == destination.parent.resolve()
            and not any(os.path.lexists(p / ".git") for p in (destination.parent, *destination.parent.parents)),
            "canonical private certificate outside Git")
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    return 0 if result["status"] == "Qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
