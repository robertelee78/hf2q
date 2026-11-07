#!/usr/bin/env python3
"""ADR-063 Q2 items 4 and 6: one-attempt traced four-agent cohort (Qwen3.8 or DeepSeek-V4).

The family comes from the --case (`qwen38` or `deepseek4`) and selects the predeclared witness manifest:
the input sources that family's engine actually publishes. Runs ONE canonical four-agent coding cohort (`--population n4`) or one Q3 population
control (`n1`, `3+1`, `2+2`, `1+3`: the client's --phase-plan) on the given hf2q binary
with HF2Q_PHASE_TRACE_CONFIG set, under the Q5 supervision (q5_calibration preflight,
500 ms memory sampler, stop latch and owned process-group cleanup). Client and trace share
one run id. Writes, into a new directory outside Git, the retained raw files that
q2_traced_freeze.py, q2_manifest.py and q2_certify.py read:

  receipt.json              hf2q.q2.traced-run-receipt.v1: protocol_complete, final_verified,
                            cleanup_proved and one cleanup entry per owned process group
  readiness.json            {health, runtime, readyz}, fetched once /readyz reports ready
  native-config.json        the HF2Q_PHASE_TRACE_CONFIG file (its sha256 is config_sha256)
  native-phase-trace.jsonl  the server phase trace (native_command_capture_v3)
  client/                   coordinator output; every agent's test executables are kept

One attempt: no retry, no favorable sample. See README "Q2 traced cohort".
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import tomllib

import q2_manifest
from q2_certify import IDENTITY
import q3_campaign
import q5_admission
import q5_artifacts
from q5_artifacts import sha256
import q5_calibration as q5
import q5_case
import q5_host
import q5_policy
import q5_runtime
import qwen_n4_live_smoke as coordinator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RECEIPT_SCHEMA = "hf2q.q2.traced-run-receipt.v1"
COMPLETE = "TracedCohortCompleteAwaitingQ2Certification"
POPULATIONS = tuple(q3_campaign.WORKLOADS)  # n4, n1, 3+1, 2+2, 1+3
CLIENT_SUCCESS = "FourAgentProtocolCompleteAwaitingIndependentReview"
MIB, GIB = 1 << 20, 1 << 30
CONFIG_NAME, TRACE_NAME = "native-config.json", "native-phase-trace.jsonl"
SERVER_BUDGET, CLIENT_BUDGET = 6 * GIB, 128 * MIB
EVENT_LIMIT, METADATA_LIMIT = 8_000_000, 16 * MIB
PROFILE = "native_command_capture_v3"
# Per family: the actual-input witness sources its engine publishes (native_observation adapters).
WITNESS_SOURCES = {
    "qwen38": ["qwen35.greedy.actual_tokens", "qwen35.forward.actual_tokens",
               "qwen35.joint_decode.actual_slot_rows"],
    "deepseek4": ["deepseek4.sorted_cache_lane", "deepseek4.prefill.actual_cooperative_lane",
                  "deepseek4.prefill.actual_cooperative_pack", "deepseek4.prefill.actual_cache_span",
                  "deepseek4.scalar_verifier"],
}
# Families whose traced case serves a text-only locator: readiness must show no vision projector.
TEXT_ONLY = {"qwen38"}
WITNESS = {"schema": "hf2q.f2.forward-witness-policy.v1", "operator": "target_input_rms_norm",
           "coverage": "each_admitted_lane_in_first_fixed_window",
           "batching_scope": "operator_only_full_body_unknown"}
ARM_READBACK_LIMIT = 8 * MIB  # the pre-client trace holds only header, baseline and arm ACK
ARM_SECONDS = 30
RUN_ID = re.compile(r"[A-Za-z0-9_-]{16,64}\Z")
HEX64, HEX40 = re.compile(r"[0-9a-f]{64}\Z"), re.compile(r"[0-9a-f]{40}\Z")
INTERRUPTED = []


def require(ok, why):
    if not ok:
        raise ValueError(why)


def run_id_for(name):
    """The one id of client approval and trace config, derived from the output directory name."""
    value = "q2t-" + re.sub(r"[^A-Za-z0-9_-]", "-", name)
    require(RUN_ID.fullmatch(value) is not None,
            "output directory name must give a 16..64 character run id (q2t-<name>, [A-Za-z0-9_-])")
    return value


def phase_plan(population):
    require(population in POPULATIONS, "population must be one of " + ", ".join(POPULATIONS))
    return q3_campaign.WORKLOADS[population]


def witness(family):
    require(family in WITNESS_SOURCES, "traced families: " + ", ".join(sorted(WITNESS_SOURCES)))
    return {**json.loads(json.dumps(WITNESS)), "input_sources": list(WITNESS_SOURCES[family])}


def traced_case(case, population):
    """A traced family's four-slot case for one population, with the server phase trace config added."""
    require(case.get("family") in WITNESS_SOURCES and case.get("workload", {}).get("slots") == 4,
            "the traced cohort runs a four-slot case of a traced family (" + ", ".join(sorted(WITNESS_SOURCES)) + ")")
    derived = q3_campaign.derive_case(case, population)
    env = derived["serve"].setdefault("env", {})
    require("HF2Q_PHASE_TRACE_CONFIG" not in env, "case already sets HF2Q_PHASE_TRACE_CONFIG")
    env["HF2Q_PHASE_TRACE_CONFIG"] = "{OUT}/" + CONFIG_NAME
    derived["notes"] = list(derived.get("notes", [])) + [
        "Derived by q2_traced_run.py: server phase trace (native_command_capture_v3)."]
    return q5_case.validate(derived)


def native_pin(lock_text):
    """(version, checksum) of the one registry mlx-native package in a Cargo.lock."""
    found = [p for p in tomllib.loads(lock_text).get("package", []) if p.get("name") == "mlx-native"]
    require(len(found) == 1 and found[0].get("source") == "registry+https://github.com/rust-lang/crates.io-index"
            and isinstance(found[0].get("checksum"), str),
            "Cargo.lock must pin exactly one published mlx-native")
    return found[0]["version"], found[0]["checksum"]


def git_lock(commit):
    return subprocess.run(["git", "-C", str(REPO), "show", commit + ":Cargo.lock"], check=True,
                          capture_output=True, text=True, timeout=10).stdout


def check_identity(identity, *, binary_sha256, build, template_sha256, model_sha256, native):
    """The frozen trace/plan identity must be what this run actually serves."""
    require(isinstance(identity, dict) and set(identity) == IDENTITY,
            "identity must hold exactly the eight q2_certify identity keys")
    for key in sorted(IDENTITY - {"server_source", "native_version"}):
        require(isinstance(identity[key], str) and HEX64.fullmatch(identity[key]) is not None,
                f"identity {key} must be 64 lowercase hex")
    require(isinstance(identity["server_source"], str) and HEX40.fullmatch(identity["server_source"]) is not None,
            "identity server_source must be a 40-hex commit")
    require(identity["server_binary_sha256"] == binary_sha256, "server_binary_sha256 is not the sha256 of --binary")
    require(isinstance(build, dict) and build.get("sha256") == binary_sha256,
            "build receipt sha256 differs from --binary")
    require(build.get("source") == identity["server_source"], "server_source differs from the build receipt source")
    require(identity["profile_sha256"] == template_sha256,
            "profile_sha256 is not the sha256 of the client approval template")
    require(model_sha256 is not None and identity["model_sha256"] == model_sha256,
            "model_sha256 differs from the artifact's conversion-receipt sha256")
    require((identity["native_version"], identity["native_checksum"]) == tuple(native),
            "native_version/native_checksum differ from mlx-native in the server_source Cargo.lock")
    return identity


def trace_config(out, run_id, identity, nonce, family, gpu_timestamps=False):
    require(RUN_ID.fullmatch(run_id) is not None, "invalid run id")
    require(32 <= len(nonce) <= 128 and re.fullmatch(r"[A-Za-z0-9_-]+", nonce) is not None, "invalid control nonce")
    return {"schema": "hf2q.f2.phase-trace.v1", "output": str(out / TRACE_NAME), "run_id": run_id,
            "server_budget_bytes": SERVER_BUDGET, "client_budget_bytes": CLIENT_BUDGET,
            "metadata_budget_profile": PROFILE,
            "native_observer": {"event_limit": EVENT_LIMIT, "metadata_limit_bytes": METADATA_LIMIT,
                                "diagnostic_case": True, "control_nonce": nonce,
                                "witness_manifest": witness(family),
                                # ADR-063 Q4-TRACE-GPU-TIMING: present only when on, so default
                                # configs (and config_sha256) are unchanged.
                                **({"gpu_timestamps": True} if gpu_timestamps else {})},
            "identity": dict(identity)}


def readiness_envelope(health, runtime, readyz, *, model_id, server_pid, identity, family):
    """The single readiness file q2_certify reads, checked against the launched process."""
    require(isinstance(readyz, dict) and readyz.get("ready") is True, "readyz is not ready")
    require(isinstance(health, dict) and health.get("model") == model_id and health.get("context_length") == 262144,
            f"health must report model {model_id!r} at context_length 262144")
    serve = runtime.get("serve") or {} if isinstance(runtime, dict) else {}
    snapshot = runtime.get("measurement_snapshot") or {} if isinstance(runtime, dict) else {}
    require(serve.get("max_slots") == 4, "runtime serve.max_slots is not 4")
    require(snapshot.get("process_pid") == server_pid, "measurement_snapshot.process_pid is not the launched server pid")
    if family in TEXT_ONLY:
        require((snapshot.get("active_controls") or {}).get("vision_projector") is False,
                "vision projector loaded: the traced cohort is text-only")
    require(snapshot.get("tokenizer_sha256") == identity["tokenizer_sha256"]
            and snapshot.get("template_sha256") == identity["template_sha256"],
            "served tokenizer/template sha256 differ from the frozen identity")
    return {"health": health, "runtime": runtime, "readyz": readyz}


def complete_lines(path, limit):
    with open(path, "rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "pre-client trace larger than its readback bound")
    return [json.loads(line) for line in data[:data.rfind(b"\n") + 1].splitlines() if line]


def arm_readback(path, config):
    """Header and worker-checked arm ACK before any request; None until the ACK is written."""
    if not path.is_file():
        return None
    records = complete_lines(path, ARM_READBACK_LIMIT)
    if not records:
        return None
    header = records[0]
    payload = header.get("payload") or {}
    require(header.get("kind") == "header" and payload.get("run_id") == config["run_id"]
            and payload.get("identity") == config["identity"]
            and payload.get("metadata_budget_profile") == config["metadata_budget_profile"]
            and payload.get("native_observer") == config["native_observer"],
            "trace header differs from native-config.json")
    acks = [r for r in records if r.get("kind") == "native_arm_ack"]
    if not acks:
        return None
    ack = acks[0]["payload"]
    require(len(acks) == 1 and ack.get("nonce") == config["native_observer"]["control_nonce"]
            and ack.get("worker_queue_checked") is True and ack.get("quiescent") is True,
            "native arm ACK nonce/worker check failed")
    gpu = config["native_observer"].get("gpu_timestamps") is True
    require(("clock_pair" in ack) == gpu, "arm ACK clock pair must match the gpu_timestamps opt-in")
    baseline = ack.get("baseline") or {}
    require(baseline.get("pending_commands") == 0 and baseline.get("invalid") is False
            and baseline.get("dropped_events") == 0, "native arm baseline is not clean")
    require(not any(r.get("kind") == "slots" and r["payload"].get("slots") for r in records),
            "a slot was admitted before the client launched")
    return {"header": header, "ack": acks[0]}


def trace_final(path, config):
    """First-line binding and the final record; reads only the head line and the last 64 KiB."""
    with open(path, "rb") as stream:
        header = json.loads(stream.readline(64 * 1024))
        size = os.fstat(stream.fileno()).st_size
        stream.seek(max(0, size - 64 * 1024))
        tail = stream.read()
    require(header.get("kind") == "header" and header["payload"].get("run_id") == config["run_id"]
            and header["payload"].get("identity") == config["identity"],
            "trace header run id/identity differ from native-config.json")
    require(tail.endswith(b"\n"), "trace does not end with a complete record")
    last = json.loads(tail.rstrip(b"\n").rsplit(b"\n", 1)[-1])
    require(last.get("kind") == "end" and last.get("payload") == {"error": None, "dropped_events": 0},
            "trace has no clean end record (error null, dropped_events 0)")
    return {"bytes": size, "origin_monotonic_ns": header["payload"].get("origin_monotonic_ns"),
            "end": last}


def retained_tests(client_root, plan):
    """Every active agent's compile receipt and both retained test executables (q2_manifest's own reader)."""
    retained = {}
    for agent in coordinator.active_agents(plan):
        groups = [q2_manifest.test_group(client_root / agent, number) for number in (4, 6)]
        retained[agent] = [run["binary"] for group in groups for run in group["executions"]]
    return retained


def protocol_complete(client_root, plan, outcome):
    require(isinstance(outcome, dict) and outcome.get("exit_code") == 0 and outcome.get("status_ok") is True,
            "client exit/status not complete")
    summary = json.loads((client_root / "summary.json").read_text())
    require(summary.get("status") == CLIENT_SUCCESS and summary.get("cleanup") is True,
            "coordinator summary not complete")
    require(set(summary.get("agents", {})) == set(coordinator.active_agents(plan)),
            "coordinator agent set differs from the population")
    return retained_tests(client_root, plan)


def cleanup_entries(result, labelled):
    """One q2_certify cleanup entry per owned group, from q5_runtime.terminate's proof."""
    remaining = result.get("remaining")
    left = None if remaining is None else {row["pgid"] for row in remaining}
    return [{"label": label, "pid": proc.pid, "pgid": proc.pid, "exit_code": proc.returncode,
             "reaped": proc.returncode is not None,
             "group_empty": left is not None and proc.pid not in left and proc.pid not in result["refused_pgids"],
             "errors": list(result["errors"])} for label, proc in labelled if proc is not None]


def final_verified(out, config_sha256, binary, binary_sha256, template, template_sha256):
    config = json.loads((out / CONFIG_NAME).read_text())
    require(sha256(out / CONFIG_NAME) == config_sha256, "native-config.json changed during the run")
    require(sha256(binary) == binary_sha256, "server binary changed during the run")
    require(sha256(template) == template_sha256, "client approval template changed during the run")
    approval = json.loads((out / "client-approval.json").read_text())
    for name, digest in approval["sources"].items():
        require(sha256(HERE / name) == digest, f"client source {name} changed during the run")
    return trace_final(out / TRACE_NAME, config)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--population", required=True, choices=POPULATIONS)
    parser.add_argument("--binary", required=True, type=Path, help="frozen hf2q binary")
    parser.add_argument("--build-receipt", required=True, type=Path, help="its build receipt {source, sha256}")
    parser.add_argument("--identity", required=True, type=Path, help="the eight-key q2_certify identity JSON")
    parser.add_argument("--negative-control-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new run directory outside any Git tree")
    parser.add_argument("--case", type=Path, default=HERE / "q5_cases" / "qwen38.json")
    parser.add_argument("--dry-run", action="store_true", help="check identities and print the plan; launch nothing")
    parser.add_argument("--gpu-timestamps", action="store_true",
                        help="record per-command GPU timing (ADR-063 Q4-TRACE-GPU-TIMING); default off")
    args = parser.parse_args(argv)
    for name in ("binary", "build_receipt", "identity", "negative_control_receipt", "output", "case"):
        if not getattr(args, name).is_absolute():
            parser.error(f"--{name.replace('_', '-')} must be an absolute path")
    return args


def prepare(args, environ):
    """Everything checkable before launch; raises ValueError/OSError with the reason."""
    out = args.output
    require(out.parent == out.parent.resolve() and not os.path.lexists(out),
            "output must be a new directory under a canonical parent")
    require(not q5.inside_git(out.parent), "output must be outside any Git tree")
    require("HF2Q_Q5_HF2Q_BIN" not in environ, "set the binary with --binary, not HF2Q_Q5_HF2Q_BIN")
    case = traced_case(q5.load_json(args.case), args.population)
    base = q5.builtins(out)
    base["RUN_ID"] = run_id_for(out.name)
    plan = q5_case.build_plan(case, base, {**environ, "HF2Q_Q5_HF2Q_BIN": str(args.binary)})
    require(plan["complete"], "missing case variables: " + ", ".join(plan["missing_variables"]))
    checks = q5_artifacts.file_checks(plan)
    require(all(checks.values()), "missing files: " + ", ".join(k for k, ok in checks.items() if not ok))
    require(plan["client"]["approval"]["set"].get("run_id") == base["RUN_ID"],
            "the client approval run_id must be the shared {RUN_ID}")
    binary_sha = sha256(args.binary)
    template = plan["client"]["approval"]["template"]
    identity = q5.load_json(args.identity)
    artifact = q5_artifacts.artifact_identity(plan["artifact"]["path"], plan["artifact"]["sha256"])
    require(isinstance(identity, dict) and isinstance(identity.get("server_source"), str), "identity server_source")
    check_identity(identity, binary_sha256=binary_sha, build=q5.load_json(args.build_receipt),
                   template_sha256=sha256(template), model_sha256=artifact["sha256"],
                   native=native_pin(git_lock(identity["server_source"])))
    config = trace_config(out, base["RUN_ID"], identity, "q2t_" + secrets.token_hex(24), case["family"],
                          args.gpu_timestamps)
    latency = q5_admission.negative_control_latency(q5.load_json(args.negative_control_receipt))
    return {"plan": plan, "values": {**base, **plan["variables"]}, "config": config, "identity": identity,
            "artifact": artifact, "latency_ns": latency, "binary_sha256": binary_sha, "template": template}


def wait_ready(plan, server, latch, stop, identity, family):
    spec, start = plan["readiness"], time.monotonic()
    while not latch.event.is_set():
        if INTERRUPTED:
            return stop("interrupted", "operator signal before readiness")
        if server.poll() is not None:
            return stop("server_exited", f"server exit {server.returncode} before readiness")
        if time.monotonic() - start > spec["timeout_seconds"]:
            return stop("readiness_timeout", f"not ready within {spec['timeout_seconds']} s")
        try:
            if q5.same(q5.field(q5.get_json(spec["url"]), spec["field"]), spec["equals"]):
                break
        except (OSError, ValueError):
            pass
        latch.event.wait(0.5)
    if latch.event.is_set():
        return None
    base = f"http://127.0.0.1:{plan['port']}"
    try:
        return readiness_envelope(q5.get_json(base + "/health"), q5.get_json(base + "/hf2q/v1/runtime"),
                                  q5.get_json(base + "/readyz"), model_id=plan["variables"]["MODEL_ID"],
                                  server_pid=server.pid, identity=identity, family=family)
    except (OSError, ValueError) as error:
        return stop("readiness_mismatch", str(error))


def wait_arm(path, config, server, latch, stop):
    deadline = time.monotonic() + ARM_SECONDS
    while not latch.event.is_set():
        if INTERRUPTED or server.poll() is not None:
            return stop("interrupted" if INTERRUPTED else "server_exited", "before the native arm ACK")
        try:
            readback = arm_readback(path, config)
        except (OSError, ValueError) as error:
            return stop("native_arm_unverified", str(error))
        if readback is not None:
            return readback
        if time.monotonic() > deadline:
            return stop("native_arm_unverified", f"no native arm ACK within {ARM_SECONDS} s of readiness")
        latch.event.wait(0.2)
    return None


def supervise(prepared, out, receipt, state):
    plan, values, config = prepared["plan"], prepared["values"], prepared["config"]
    policy = plan["policy"]
    reason = q5.preflight(plan, policy, receipt)
    require(reason is None, f"preflight refused: {reason}")
    locator = q5_artifacts.create_locator(plan["artifact"]["path"], str(out))
    state["locator"] = receipt["text_only_locator"] = locator
    require(locator["served_path"] == plan["served_model"], "text-only locator path differs from SERVED_MODEL")
    fd = os.open(out / CONFIG_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(config, indent=2) + "\n")
    receipt["native_config"] = {"path": str(out / CONFIG_NAME), "sha256": sha256(out / CONFIG_NAME)}
    monitor = q5_policy.Monitor(receipt["baseline"], policy, prepared["latency_ns"], "negative-control receipt", 0)
    latch = q5_runtime.StopLatch()
    sampler = q5_runtime.Sampler(monitor, policy["interval_seconds"], out / "samples.jsonl", latch)
    state.update(monitor=monitor, latch=latch, sampler=sampler, popens=[], server=None, client=None)

    def stop(name, detail):
        if latch.trip({"reason": name, "detail": detail, "sample_index": None, "sample_start_ns": None,
                       "sample_end_ns": None}):
            monitor.stop = latch.stop
        return None

    sampler.start()
    try:
        server = state["server"] = q5.launch(plan["serve"]["argv"], plan["serve"]["env"], out, "server")
        state["popens"].append(server)
        latch.groups.append(server.pid)
        sampler.pgid, sampler.roots = server.pid, {server.pid}
        values["SERVER_PID"] = receipt["server_pid"] = server.pid
        envelope = wait_ready(plan, server, latch, stop, prepared["identity"], plan["family"])
        if envelope is None:
            return
        q5.save(out / "readiness.json", envelope)
        monitor.ready_ns = time.monotonic_ns()
        readback = wait_arm(out / TRACE_NAME, config, server, latch, stop)
        if readback is None:
            return
        q5.save(out / "native-arm-readback.json", readback)
        values.update(q5.render_approval(plan["client"]["approval"], values, out))
        approval = json.loads(Path(values["CLIENT_APPROVAL"]).read_text())
        require(approval.get("run_id") == config["run_id"] and approval.get("server_pid") == server.pid,
                "client approval must carry the shared run id and the launched server pid")
        argv = q5_case.validate_argv(q5_case.expand(plan["client"]["argv"], values), "client")
        receipt["client_argv"] = argv
        started = time.monotonic()
        client = state["client"] = q5.launch(argv, plan["client"]["env"], out, "client")
        state["popens"].append(client)
        latch.groups.append(client.pid)
        sampler.roots = {server.pid, client.pid}
        while client.poll() is None and not latch.event.is_set():
            if INTERRUPTED:
                stop("interrupted", "operator signal")
            elif server.poll() is not None:
                stop("server_exited", f"server exit {server.returncode} during client run")
            elif time.monotonic() - started > plan["client"]["timeout_seconds"]:
                stop("client_timeout", f"client exceeded {plan['client']['timeout_seconds']} s")
            latch.event.wait(0.25)
        if latch.stop is None:
            receipt["client"] = q5.client_outcome(plan, client, started)
    finally:
        latch.armed = False
        popens = state["popens"]
        result = q5_runtime.terminate([p.pid for p in popens], sampler.known, popens, policy["term_grace_seconds"],
                                      policy["kill_grace_seconds"], latch.signal_ns)
        sampler.halt.set()
        sampler.join(10)
        receipt["cleanup_detail"] = result
        receipt["cleanup"] = cleanup_entries(result, [("server", state["server"]), ("client", state["client"])])
        receipt["stop"] = latch.stop
        receipt["sampler_error"] = sampler.error
        receipt["memory"] = monitor.summary()
        if latch.stop is None and monitor.stop is not None:
            receipt["post_completion_violation"] = monitor.stop


def settle(prepared, out, receipt, state):
    """Cleanup proof, protocol completion and final re-verification, each with its own reason."""
    locator = state.get("locator")
    if locator is not None:
        receipt["text_only_locator"]["removal"] = q5_artifacts.remove_locator(locator)
    if state.get("server") is not None:
        try:
            pids = q5_host.listeners(prepared["plan"]["port"])
            receipt["listener_after_cleanup"] = {"pids": pids, "clear": not pids}
            if pids:
                receipt["listener_cleanup_error"] = f"port {prepared['plan']['port']} still has listeners {pids}"
        except q5_host.ObservationError as error:
            receipt["listener_cleanup_error"] = repr(error)
    entries = receipt.get("cleanup") or []
    receipt["cleanup_proved"] = bool(
        entries and receipt.get("cleanup_detail", {}).get("proven") is True and not receipt.get("listener_cleanup_error")
        and all(e["reaped"] and e["group_empty"] and not e["errors"] for e in entries)
        and (locator is None or receipt["text_only_locator"]["removal"].get("directory_removed") is True))
    if receipt.get("failure") is None and receipt.get("stop") is None and state.get("client") is not None:
        plan = phase_plan(receipt["population"])
        try:
            receipt["retained_test_executables"] = protocol_complete(out / "client", plan, receipt.get("client"))
            receipt["protocol_complete"] = True
        except (OSError, ValueError, KeyError) as error:
            receipt["failure"] = "protocol: " + str(error)
        try:
            receipt["trace_final"] = final_verified(out, receipt["native_config"]["sha256"], prepared["plan"]["binary"],
                                                    prepared["binary_sha256"], prepared["template"],
                                                    receipt["identity"]["profile_sha256"])
            receipt["final_verified"] = receipt["protocol_complete"] and receipt.get("post_completion_violation") is None
        except (OSError, ValueError, KeyError) as error:
            receipt["failure"] = receipt.get("failure") or "final verification: " + str(error)
    elif receipt.get("failure") is None and receipt.get("stop") is not None:
        receipt["failure"] = f"stopped: {receipt['stop'].get('reason')}: {receipt['stop'].get('detail')}"
    done = all(receipt[k] is True for k in ("protocol_complete", "final_verified", "cleanup_proved"))
    receipt["status"] = COMPLETE if done and not receipt.get("failure") else "FailedOrHeld"


def main(argv=None):
    args = parse_args(argv)
    try:
        prepared = prepare(args, os.environ)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"valid": False, "error": str(error)}, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps({"valid": True, "dry_run": True, "population": args.population,
                          "run_id": prepared["config"]["run_id"], "native_config": prepared["config"],
                          "plan": prepared["plan"]}, indent=2))
        return 0
    out = args.output
    os.umask(0o077)
    out.mkdir(mode=0o700)
    receipt = {"schema": RECEIPT_SCHEMA, "population": args.population, "phase_plan": phase_plan(args.population),
               "run_id": prepared["config"]["run_id"], "identity": prepared["identity"],
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "host": q5_host.host_identity(), "binary": {"path": str(args.binary), "sha256": prepared["binary_sha256"]},
               "build_receipt": {"path": str(args.build_receipt), "sha256": sha256(args.build_receipt)},
               "case": {"path": str(args.case), "sha256": sha256(args.case)}, "artifact": prepared["artifact"],
               "serve_argv": prepared["plan"]["serve"]["argv"], "serve_env": prepared["plan"]["serve"]["env"],
               "policy": prepared["plan"]["policy"], "server_pid": None,
               "supervisor_sources": {n: sha256(HERE / n) for n in ("q2_traced_run.py", *q5.SOURCES)},
               "protocol_complete": False, "final_verified": False, "cleanup_proved": False, "cleanup": [],
               "failure": None, "status": "Pending"}
    q5.save(out / "receipt.json", receipt)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, _frame: INTERRUPTED.append(signum))
    state = {}
    try:
        supervise(prepared, out, receipt, state)
    except Exception as error:
        receipt["failure"] = repr(error)
    finally:
        try:
            settle(prepared, out, receipt, state)
        except Exception as error:
            receipt["failure"], receipt["status"] = receipt.get("failure") or "settle: " + repr(error), "FailedOrHeld"
        receipt["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        q5.save(out / "receipt.json", receipt)
    print(json.dumps({"receipt": str(out / "receipt.json"), "status": receipt["status"], "run_id": receipt["run_id"],
                      "failure": receipt["failure"], "cleanup_proved": receipt["cleanup_proved"]}, indent=2))
    return 0 if receipt["status"] == COMPLETE else 2 if receipt["server_pid"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
