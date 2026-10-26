#!/usr/bin/env python3
"""Four independent coding clients; NEVER starts a server or model process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from qwen38_n1_verdict import Invalid, WORKFLOW, loads, require, qualify
import qwen38_n1_process as child
import coding_family as family_contract

AGENTS = tuple(f"agent-{i}" for i in range(4))
PROFILE = "adr063-qwen-n4-live-v1"
WORKLOAD_ID = "adr063-qwen-n4-parse-port-v1"
MODULES = ("qwen38_n1_smoke.py", "qwen38_n1_tools.py", "qwen38_n1_verdict.py",
           "qwen38_n1_process.py", "test_qwen38_n1_verdict.py",
           "qwen_n4_live_smoke.py", "test_qwen_n4_live_smoke.py",
           "coding_family.py", "test_coding_family.py", "qwen_n4_peer.py", "test_qwen_n4_peer.py")
INTERRUPTED = False


def identity(agent, run):
    require(agent in AGENTS and isinstance(run, str) and re.fullmatch(r"[a-zA-Z0-9_-]{16,64}", run),
            "invalid independent agent/run identity")
    return (f"Independent coding agent {agent}, workload {WORKLOAD_ID}. "
            "Use only your own isolated fixture and this conversation's tool results. "
            "Do not reuse another agent's transcript, files, or claimed test results.")


def preflight(base):
    return (set(base) - {"ctx262144_slots1_verified", "cpu_owned_io_3g_tool2g_client128m",
                         "profile24g_ack_verified"}) | {
        "ctx262144_slots4_verified", "dedicated_host_live_pressure_availablememory",
        "pool_profile_selection_verified"}


def validate_preflight(value, base, engine_kind="hf2q", family="qwen"):
    family_contract.validate_selection(family, engine_kind, 4)
    keys = family_contract.preflight(preflight(base), family)
    if engine_kind == "peer":
        from qwen_n4_peer import preflight as peer_preflight
        keys = peer_preflight(keys)
    require(isinstance(value, dict) and set(value) == keys and
            all(v is True for v in value.values()), "four-slot actual pool selection/preflight")


def validate_agent_binding(packet, agent, run, server_pid, barrier, token, output, family="qwen", engine_kind="hf2q"):
    family_contract.validate_selection(family, engine_kind, 4)
    require(packet.get("engine_kind", "hf2q") == engine_kind, "engine approval mismatch")
    family_contract.validate_binding(packet, family, 4)
    require(packet.get("schema") == family_contract.schema(family, "agent-approval", 4) and
            packet.get("profile") == family_contract.profile(family, 4) and packet.get("diagnostic_slots") == 4,
            "N4 approval cannot claim slots1")
    identity(agent, run)
    expected = {"agent_id": agent, "run_id": run, "server_pid": server_pid,
                "barrier_dir": str(Path(barrier).resolve()), "barrier_token": token,
                "agent_output": str(Path(output).resolve()),
                "initial_identity_sha256": hashlib.sha256(identity(agent, run).encode()).hexdigest()}
    require(all(packet.get(k) == v for k, v in expected.items()), "agent/server/barrier identity drift")
    require(isinstance(token, str) and re.fullmatch(r"[a-f0-9]{32}", token), "barrier nonce")


def save_new(path, value):
    path = Path(path)
    pending = path.with_name(path.name + f".pending-{os.getpid()}")
    with pending.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(pending, path)  # atomic publication, refuses an existing/stale record
    finally:
        pending.unlink()


def validate_ready(records, run, pid, token):
    require(set(records) == set(AGENTS), "all four independent agents required")
    outputs, hashes = set(), set()
    for agent in AGENTS:
        record = records[agent]
        require(record.get("schema") == "hf2q.n4-agent-ready.v1" and record.get("agent_id") == agent
                and record.get("run_id") == run and record.get("server_pid") == pid
                and record.get("barrier_token") == token, "ready binding")
        require(record.get("identity_sha256") == hashlib.sha256(identity(agent, run).encode()).hexdigest(),
                "ready initial identity")
        require(isinstance(record.get("initial_messages_sha256"), str) and
                re.fullmatch(r"[a-f0-9]{64}", record["initial_messages_sha256"]), "ready prompt fingerprint")
        outputs.add(record.get("output"))
        hashes.add(record["initial_messages_sha256"])
    require(len(outputs) == 4 and None not in outputs and len(hashes) == 4, "shared agent output/prompt")


def barrier_release(records, run, pid, token, hashes):
    validate_ready(records, run, pid, token)
    require(set(hashes) == set(AGENTS), "missing ready hashes")
    return {"schema": "hf2q.n4-barrier-release.v1", "run_id": run, "server_pid": pid,
            "barrier_token": token, "agents": list(AGENTS), "ready_sha256": hashes}


def validate_release(value, agent, run, pid, token, own_hash):
    require(value.get("schema") == "hf2q.n4-barrier-release.v1" and value.get("run_id") == run and
            value.get("server_pid") == pid and value.get("barrier_token") == token and
            value.get("agents") == list(AGENTS), "stale/mismatched barrier release")
    hashes = value.get("ready_sha256", {})
    require(set(hashes) == set(AGENTS) and all(isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)
                                               for v in hashes.values()), "incomplete release hashes")
    require(hashes[agent] == own_hash, "own ready changed")


def wait_barrier(directory, agent, run, pid, token, output, initial_hash, approval_hash):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "owned barrier directory")
    record = {"schema": "hf2q.n4-agent-ready.v1", "agent_id": agent, "run_id": run,
              "server_pid": pid, "barrier_token": token, "output": str(Path(output).resolve()),
              "identity_sha256": hashlib.sha256(identity(agent, run).encode()).hexdigest(),
              "initial_messages_sha256": initial_hash, "approval_sha256": approval_hash}
    ready_path = directory / f"{agent}.ready.json"
    save_new(ready_path, record)
    started = time.monotonic()
    release = directory / "release.json"
    while not release.exists():
        require(time.monotonic() - started < 120, "four-client barrier timeout")
        time.sleep(0.02)
    value = loads(release.read_bytes())
    validate_release(value, agent, run, pid, token, child.digest(ready_path))
    return {"status": "Released", "ready_sha256": child.digest(ready_path),
            "release_sha256": child.digest(release), "wait_seconds": time.monotonic() - started}


def aggregate(results, exits, cleanup, pid, run, engine_kind="hf2q", family="qwen"):
    family_contract.validate_selection(family, engine_kind, 4)
    require(set(results) == set(exits) == set(cleanup) == set(AGENTS), "missing agent result")
    initial = set()
    for agent in AGENTS:
        result = results[agent]
        require(result.get("engine_kind", "hf2q") == engine_kind, "mixed engine result")
        require(result.get("family", "qwen") == family, "mixed diagnostic family result")
        require(exits[agent] == 0 and cleanup[agent] is True and result.get("cleanup") is True,
                "client exit/cleanup unknown")
        require(result.get("schema") == family_contract.schema(family, "agent-result", 4) and result.get("profile") == family_contract.profile(family, 4)
                and result.get("agent_id") == agent and result.get("run_id") == run
                and result.get("server_pid") == pid and result.get("diagnostic_slots") == 4,
                "aggregate agent/server identity")
        require(result.get("status") == "ProtocolCompleteAwaitingIndependentPatchReview", "agent incomplete")
        require(result.get("barrier", {}).get("status") == "Released", "agent bypassed barrier")
        require(result.get("full_prefix_reuse") == "PendingPostOracle" and
                result.get("model_qualified") is False and result.get("runtime_qualified") is False,
                "qualification/prefix verdict scope")
        require(len(result.get("turns", [])) == 7 and len(result.get("tools", [])) == 6,
                "not all28requests/24tools")
        require(result.get("initial_identity_sha256") == hashlib.sha256(identity(agent, run).encode()).hexdigest(),
                "aggregate prompt identity")
        require(result.get("final_source_sha256") == result.get("observed_final_source_sha256")
                and isinstance(result.get("final_source_sha256"), str), "final fixture changed")
        require(qualify(result["turns"], result["tools"], result["cleanup"],
                        result["final_source_sha256"], result["observed_final_source_sha256"]) == result["status"],
                "child protocol receipt contradiction")
        initial.add(result.get("initial_messages_sha256"))
    require(len(initial) == 4 and None not in initial, "agent prompts not independent")
    return "FourAgentProtocolCompleteAwaitingIndependentReview"


def descendants(rows, leader, known):
    require(all(p not in rows or rows[p][3] == stamp for p, stamp in known.items()), "client PID reused")
    owned = {leader} | (set(rows) & set(known))
    while True:
        extra = {p for p, row in rows.items() if row[0] in owned or row[1] == leader}
        if extra <= owned:
            break
        owned |= extra
    known.update({p: rows[p][3] for p in owned if p in rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "model", "output", "cargo", "toolchain", "sandbox-exec", "sandbox-proof",
                 "sandbox-proof-sha256", "approval", "approval-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--family", choices=family_contract.FAMILIES, default="qwen")
    parser.add_argument("--engine-kind", choices=("hf2q", "peer"), default="hf2q")
    args = parser.parse_args()
    family_contract.validate_selection(args.family, args.engine_kind, 4)
    from qwen38_n1_smoke import new_output, PREFLIGHT
    os.umask(0o077)
    def stop(_signum, _frame):
        global INTERRUPTED
        INTERRUPTED = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    output = new_output(args.output)
    summary = {"schema": family_contract.schema(args.family, "coordinator-result", 4),
               "profile": family_contract.profile(args.family, 4), "family": args.family,
               "status": "Pending", "server_pid": args.server_pid, "diagnostic_slots": 4,
               "cleanup": False, "runtime_qualified": False, "model_qualified": False,
               "full_prefix_reuse": "PendingPostOracle", "server_owned_by_outer_supervisor": True}
    processes, known, streams, receipts = {}, {}, [], {}
    results, exits, cleanup = {}, {}, {}
    started = time.monotonic()
    try:
        require(child.digest(args.approval) == args.approval_sha256, "coordinator approval hash")
        packet = loads(Path(args.approval).read_bytes())
        require(packet.get("schema") == family_contract.schema(args.family, "coordinator-approval", 4) and
                packet.get("profile") == family_contract.profile(args.family, 4) and packet.get("approved") is True, "N4 approval")
        family_contract.validate_binding(packet, args.family, 4)
        require(packet.get("server_pid") == args.server_pid and packet.get("base_url") == args.base_url
                and packet.get("model") == args.model, "single server binding")
        module_dir = Path(__file__).resolve().parent
        names = MODULES
        sources = {name: child.digest(module_dir / name) for name in names}
        require(packet.get("engine_kind", "hf2q") == args.engine_kind, "engine approval mismatch")
        summary["engine_kind"] = args.engine_kind
        if args.engine_kind == "peer":
            from qwen_n4_peer import readiness
            summary["peer_readiness"] = readiness(args.base_url, args.model, output)
        require(packet.get("sources") == sources, "N4 source freeze drift")
        require(packet.get("python_sha256") == child.digest(Path(sys.executable).resolve()), "Python binary")
        run, token = packet.get("run_id"), packet.get("barrier_token")
        identity(AGENTS[0], run)
        require(isinstance(token, str) and re.fullmatch(r"[a-f0-9]{32}", token), "barrier nonce")
        template = packet.get("agent_template", {})
        validate_preflight(template.get("preflight"), PREFLIGHT, args.engine_kind, args.family)
        summary["run_id"] = run
        barrier = output / "barrier"
        barrier.mkdir(mode=0o700)
        for agent in AGENTS:
            require(not INTERRUPTED, "interrupted before any further client launch")
            path = output / agent
            binding = {**template, "schema": family_contract.schema(args.family, "agent-approval", 4),
                       "profile": family_contract.profile(args.family, 4), "family": args.family,
                       "engine_kind": args.engine_kind, "family_controls": packet.get("family_controls"),
                       "diagnostic_slots": 4, "agent_id": agent, "run_id": run,
                       "barrier_dir": str(barrier), "barrier_token": token, "agent_output": str(path),
                       "initial_identity_sha256": hashlib.sha256(identity(agent, run).encode()).hexdigest(),
                       "sources": sources, "server_pid": args.server_pid, "base_url": args.base_url,
                       "model": args.model, "approved": True}
            approval_path = output / f"{agent}.approval.json"
            save_new(approval_path, binding)
            command = [sys.executable, "-E", "-s", "-B", str(module_dir / "qwen38_n1_smoke.py")]
            for key in ("base_url", "model", "cargo", "toolchain", "sandbox_exec", "sandbox_proof",
                        "sandbox_proof_sha256", "server_pid"):
                command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
            command += ["--output", str(path), "--approval", str(approval_path), "--approval-sha256",
                        child.digest(approval_path), "--diagnostic-slots", "4", "--family", args.family, "--engine-kind", args.engine_kind, "--agent-id", agent,
                        "--run-id", run, "--barrier-dir", str(barrier), "--barrier-token", token]
            stdout, stderr = [(output / f"{agent}.{name}").open("xb") for name in ("stdout", "stderr")]
            streams += [stdout, stderr]
            require(not INTERRUPTED, "interrupted immediately before child launch")
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True, cwd=output)
            processes[agent], known[agent] = process, {}
            save_new(output / f"{agent}.command.json", {"argv": command, "pid": process.pid})
        records, hashes = {}, {}
        until = time.monotonic() + 120
        while len(records) != 4:
            require(not INTERRUPTED and time.monotonic() < until, "N4 barrier deadline/interruption")
            for agent, process in processes.items():
                require(process.poll() is None, "agent exited before shared release")
                ready = barrier / f"{agent}.ready.json"
                if ready.exists():
                    records[agent], hashes[agent] = loads(ready.read_bytes()), child.digest(ready)
                    require(records[agent].get("output") == str(output / agent) and
                            records[agent].get("approval_sha256") == child.digest(output / f"{agent}.approval.json"),
                            "ready output/approval mismatch")
            time.sleep(0.02)
        release = barrier_release(records, run, args.server_pid, token, hashes)
        require(not INTERRUPTED, "interrupted before shared release")
        save_new(barrier / "release.json", release)
        summary["barrier"] = {"status": "Released", "ready": records,
                              "release_sha256": child.digest(barrier / "release.json")}
        while any(process.poll() is None for process in processes.values()):
            require(not INTERRUPTED and time.monotonic() - started < 7320, "N4 deadline/interruption")
            os.kill(args.server_pid, 0)
            rows = child.process_snapshot()
            for agent, process in processes.items():
                descendants(rows, process.pid, known[agent])
                require(process.poll() in (None, 0), "one agent failed; no partial success")
            time.sleep(0.05)
    except BaseException as exc:
        summary["status"], summary["error"] = "Failed", f"{type(exc).__name__}: {exc}"
    finally:
        for agent, process in processes.items():
            receipt = child.stop_group(process, known[agent])
            receipts[agent], cleanup[agent] = receipt, receipt["verified"]
            exits[agent] = process.returncode
            result_path = output / agent / "summary.json"
            if result_path.exists():
                try:
                    results[agent] = loads(result_path.read_bytes())
                    source = output / agent / "tools" / "fixture" / "src" / "ports.rs"
                    results[agent]["observed_final_source_sha256"] = child.digest(source)
                except BaseException as exc:
                    summary["error"] = str(exc)
        for stream in streams:
            stream.close()
        if summary["status"] != "Failed":
            try:
                summary["status"] = aggregate(results, exits, cleanup, args.server_pid, run, args.engine_kind, args.family)
            except BaseException as exc:
                summary["status"], summary["error"] = "Failed", str(exc)
        summary.update(agents=results, exits=exits, client_cleanup=receipts,
                       cleanup=len(cleanup) == 4 and all(cleanup.values()),
                       elapsed_seconds=time.monotonic() - started)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["status"] == "FourAgentProtocolCompleteAwaitingIndependentReview" else 1


if __name__ == "__main__":
    raise SystemExit(main())
