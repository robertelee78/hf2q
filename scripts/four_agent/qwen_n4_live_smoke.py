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

from f2_semantic import clock_ns, CLOCK_ID

AGENTS = tuple(f"agent-{i}" for i in range(4))
PROFILE = "adr063-qwen-n4-live-v1"
WORKLOAD_ID = "adr063-qwen-n4-parse-port-v1"
MODULES = ("qwen38_n1_smoke.py", "qwen38_n1_tools.py", "qwen38_n1_verdict.py",
           "qwen38_n1_process.py", "test_qwen38_n1_verdict.py",
           "qwen_n4_live_smoke.py", "test_qwen_n4_live_smoke.py",
           "coding_family.py", "test_coding_family.py", "qwen_n4_peer.py", "test_qwen_n4_peer.py",
           "family_peer.py", "test_family_peer.py", "f2_semantic.py")
INTERRUPTED = False
# ADR-063 Q3 controls (--phase-plan): agents in release cohorts on the same four-slot server.
# "1" is the one-agent control; "K+M" releases K agents, then the remaining M once every one of
# the first K has received its first decoded token. Absent, all four start together (pure phase,
# the canonical workload), byte-identical to runs before phase plans existed.
PHASE_PLANS = {"1": (1,), "3+1": (3, 1), "2+2": (2, 2), "1+3": (1, 3)}
FIRST_DECODE_SCHEMA = "hf2q.n4-first-decode.v1"


def phase_cohorts(plan):
    """Agent ids per release cohort; None is the canonical single release of all four agents."""
    if plan is None:
        return (AGENTS,)
    require(plan in PHASE_PLANS, "unknown phase plan")
    cohorts, start = [], 0
    for size in PHASE_PLANS[plan]:
        cohorts.append(AGENTS[start:start + size])
        start += size
    return tuple(cohorts)


def active_agents(plan):
    return tuple(agent for cohort in phase_cohorts(plan) for agent in cohort)


def agent_phase(plan, agent):
    """The per-agent phase record ({"plan", "cohort"}), None for the canonical workload."""
    if plan is None:
        return None
    for index, cohort in enumerate(phase_cohorts(plan), 1):
        if agent in cohort:
            return {"plan": plan, "cohort": index}
    raise Invalid("agent outside the phase plan")


def release_name(cohort):
    return "release.json" if cohort == 1 else f"release-{cohort}.json"


def identity(agent, run, instance=None):
    require(agent in AGENTS and isinstance(run, str) and re.fullmatch(r"[a-zA-Z0-9_-]{16,64}", run),
            "invalid independent agent/run identity")
    require(instance is None or (type(instance) is int and instance >= 0), "invalid task instance")
    label = "" if instance is None else f", task instance {instance}"
    return (f"Independent coding agent {agent}, workload {WORKLOAD_ID}{label}. "
            "Use only your own isolated fixture and this conversation's tool results. "
            "Do not reuse another agent's transcript, files, or claimed test results.")


def task_instance(packet, instance):
    """Explicit quality-campaign instance bound by the approval, or None for the canonical task."""
    if instance is None:
        require("instance" not in packet and "instance_table_sha256" not in packet,
                "canonical workload approval carries a task instance")
        return None
    from coding_instances import select
    workload = select(instance)
    require(type(packet.get("instance")) is int and packet["instance"] == instance and
            packet.get("instance_table_sha256") == workload.table_sha256, "task instance approval binding")
    return workload


def preflight(base):
    return (set(base) - {"ctx262144_slots1_verified", "cpu_owned_io_3g_tool2g_client128m",
                         "profile24g_ack_verified"}) | {
        "ctx262144_slots4_verified", "dedicated_host_live_pressure_availablememory",
        "pool_profile_selection_verified"}


def validate_preflight(value, base, engine_kind="hf2q", family="qwen"):
    family_contract.validate_selection(family, engine_kind, 4)
    keys = family_contract.preflight(preflight(base), family)
    if engine_kind == "peer":
        from family_peer import preflight as peer_preflight
        keys = peer_preflight(keys, family)
    require(isinstance(value, dict) and set(value) == keys and
            all(v is True for v in value.values()), "four-slot actual pool selection/preflight")


def validate_agent_binding(packet, agent, run, server_pid, barrier, token, output, family="qwen", engine_kind="hf2q",
                           instance=None):
    family_contract.validate_selection(family, engine_kind, 4)
    require(packet.get("engine_kind", "hf2q") == engine_kind, "engine approval mismatch")
    family_contract.validate_binding(packet, family, 4)
    require(packet.get("schema") == family_contract.schema(family, "agent-approval", 4) and
            packet.get("profile") == family_contract.profile(family, 4) and packet.get("diagnostic_slots") == 4,
            "N4 approval cannot claim slots1")
    identity(agent, run, instance)
    expected = {"agent_id": agent, "run_id": run, "server_pid": server_pid,
                "barrier_dir": str(Path(barrier).resolve()), "barrier_token": token,
                "agent_output": str(Path(output).resolve()),
                "initial_identity_sha256": hashlib.sha256(identity(agent, run, instance).encode()).hexdigest()}
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


def validate_ready(records, run, pid, token, instance=None, agents=AGENTS):
    require(set(records) == set(agents), "all four independent agents required")
    outputs, hashes = set(), set()
    for agent in agents:
        record = records[agent]
        require(record.get("schema") == "hf2q.n4-agent-ready.v1" and record.get("agent_id") == agent
                and record.get("run_id") == run and record.get("server_pid") == pid
                and record.get("barrier_token") == token, "ready binding")
        require(record.get("identity_sha256") == hashlib.sha256(identity(agent, run, instance).encode()).hexdigest(),
                "ready initial identity")
        require(isinstance(record.get("initial_messages_sha256"), str) and
                re.fullmatch(r"[a-f0-9]{64}", record["initial_messages_sha256"]), "ready prompt fingerprint")
        outputs.add(record.get("output"))
        hashes.add(record["initial_messages_sha256"])
    require(len(outputs) == len(agents) and None not in outputs and len(hashes) == len(agents),
            "shared agent output/prompt")


def barrier_release(records, run, pid, token, hashes, instance=None, agents=AGENTS):
    validate_ready(records, run, pid, token, instance, agents)
    require(set(hashes) == set(agents), "missing ready hashes")
    return {"schema": "hf2q.n4-barrier-release.v1", "run_id": run, "server_pid": pid,
            "barrier_token": token, "agents": list(agents), "ready_sha256": hashes}


def validate_release(value, agent, run, pid, token, own_hash, agents=AGENTS):
    require(value.get("schema") == "hf2q.n4-barrier-release.v1" and value.get("run_id") == run and
            value.get("server_pid") == pid and value.get("barrier_token") == token and
            value.get("agents") == list(agents), "stale/mismatched barrier release")
    hashes = value.get("ready_sha256", {})
    require(set(hashes) == set(agents) and all(isinstance(v, str) and re.fullmatch(r"[a-f0-9]{64}", v)
                                               for v in hashes.values()), "incomplete release hashes")
    require(hashes[agent] == own_hash, "own ready changed")


def wait_barrier(directory, agent, run, pid, token, output, initial_hash, approval_hash, instance=None, phase=None):
    """`phase` is the agent's {"plan", "cohort"} under --phase-plan, None for the canonical release."""
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), "owned barrier directory")
    agents, name, limit = AGENTS, "release.json", 120
    if phase is not None:
        require(agent_phase(phase["plan"], agent) == phase, "agent outside its phase cohort")
        agents, name = active_agents(phase["plan"]), release_name(phase["cohort"])
        if phase["cohort"] > 1:
            limit = 7200  # a later cohort waits for the earlier cohort's first decoded tokens
    record = {"schema": "hf2q.n4-agent-ready.v1", "agent_id": agent, "run_id": run,
              "server_pid": pid, "barrier_token": token, "output": str(Path(output).resolve()),
              "identity_sha256": hashlib.sha256(identity(agent, run, instance).encode()).hexdigest(),
              "initial_messages_sha256": initial_hash, "approval_sha256": approval_hash}
    ready_path = directory / f"{agent}.ready.json"
    save_new(ready_path, record)
    started = time.monotonic()
    release = directory / name
    while not release.exists():
        require(time.monotonic() - started < limit, "four-client barrier timeout")
        time.sleep(0.02)
    value = loads(release.read_bytes())
    validate_release(value, agent, run, pid, token, child.digest(ready_path), agents)
    if phase is not None:
        require(value.get("phase_plan") == phase["plan"] and value.get("cohort") == phase["cohort"],
                "phase release mismatch")
    return {"status": "Released", "ready_sha256": child.digest(ready_path),
            "release_sha256": child.digest(release), "wait_seconds": time.monotonic() - started,
            "release_monotonic_ns": value.get("release_monotonic_ns"),
            "f2_clock_id": value.get("f2_clock_id")}


def first_decode_marker(directory, agent, run, token, monotonic_ns):
    """Published by an agent of a non-final cohort at its first decoded token (first streamed turn)."""
    save_new(Path(directory) / f"{agent}.first-decode.json",
             {"schema": FIRST_DECODE_SCHEMA, "agent_id": agent, "run_id": run, "barrier_token": token,
              "f2_clock_id": CLOCK_ID, "first_decode_monotonic_ns": monotonic_ns})


def phase_trigger(directory, agents, run, token):
    """{agent: first-decode clock} once every agent in `agents` has published its marker, else None."""
    found = {}
    for agent in agents:
        path = Path(directory) / f"{agent}.first-decode.json"
        if not path.exists():
            return None
        value = loads(path.read_bytes())
        require(value.get("schema") == FIRST_DECODE_SCHEMA and value.get("agent_id") == agent and
                value.get("run_id") == run and value.get("barrier_token") == token and
                value.get("f2_clock_id") == CLOCK_ID and type(value.get("first_decode_monotonic_ns")) is int,
                "first-decode marker binding")
        found[agent] = value["first_decode_monotonic_ns"]
    return found


def aggregate(results, exits, cleanup, pid, run, engine_kind="hf2q", family="qwen", instance_record=None,
              phase_plan=None):
    """`instance_record` is the coordinator's {"id", "table_sha256"}; None for the canonical task.
    `phase_plan` selects the active agents and each agent's cohort; None is all four together."""
    family_contract.validate_selection(family, engine_kind, 4)
    agents = active_agents(phase_plan)
    require(set(results) == set(exits) == set(cleanup) == set(agents), "missing agent result")
    initial = set()
    for agent in agents:
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
        instance = None if instance_record is None else instance_record["id"]
        require(result.get("initial_identity_sha256") == hashlib.sha256(identity(agent, run, instance).encode()).hexdigest(),
                "aggregate prompt identity")
        require(result.get("instance") == instance_record, "aggregate task instance")
        require(result.get("phase") == agent_phase(phase_plan, agent), "aggregate phase cohort")
        require(result.get("final_source_sha256") == result.get("observed_final_source_sha256")
                and isinstance(result.get("final_source_sha256"), str), "final fixture changed")
        require(qualify(result["turns"], result["tools"], result["cleanup"],
                        result["final_source_sha256"], result["observed_final_source_sha256"]) == result["status"],
                "child protocol receipt contradiction")
        initial.add(result.get("initial_messages_sha256"))
    require(len(initial) == len(agents) and None not in initial, "agent prompts not independent")
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


def agent_command(args, module_dir, agent, path, approval_path, barrier, run, token):
    """Exact child argv; phase options are appended only under --phase-plan."""
    command = [sys.executable, "-E", "-s", "-B", str(module_dir / "qwen38_n1_smoke.py")]
    for key in ("base_url", "model", "cargo", "toolchain", "sandbox_exec", "sandbox_proof",
                "sandbox_proof_sha256", "server_pid"):
        command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    command += ["--output", str(path), "--approval", str(approval_path), "--approval-sha256",
                child.digest(approval_path), "--diagnostic-slots", "4", "--family", args.family, "--engine-kind", args.engine_kind, "--agent-id", agent,
                "--run-id", run, "--barrier-dir", str(barrier), "--barrier-token", token]
    if args.benchmark_profile is not None:
        command += ["--benchmark-profile", args.benchmark_profile]
    if args.instance is not None:
        command += ["--instance", str(args.instance)]
    phase = agent_phase(args.phase_plan, agent)
    if phase is not None:
        command += ["--phase-plan", phase["plan"], "--phase-cohort", str(phase["cohort"])]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-url", "model", "output", "cargo", "toolchain", "sandbox-exec", "sandbox-proof",
                 "sandbox-proof-sha256", "approval", "approval-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--family", choices=family_contract.FAMILIES, default="qwen")
    parser.add_argument("--engine-kind", choices=("hf2q", "peer"), default="hf2q")
    parser.add_argument("--benchmark-profile", choices=("gemma-matched-v1",))
    parser.add_argument("--instance", type=int,
                        help="quality-campaign task instance (row of fixtures/coding-instances.json); "
                             "omit for the canonical parse_port workload")
    parser.add_argument("--phase-plan", choices=sorted(PHASE_PLANS),
                        help="ADR-063 Q3 control: '1' runs agent-0 alone; 'K+M' releases K agents, then the "
                             "other M once each of the K has its first decoded token; omit for all four at once")
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
    cohorts, agents = phase_cohorts(args.phase_plan), active_agents(args.phase_plan)
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
        if args.benchmark_profile is not None:
            from gemma_matched_profile import EXTRA_MODULES, validate_activation
            validate_activation(args.benchmark_profile, packet, args.family, args.engine_kind, 4)
            validate_activation(args.benchmark_profile, packet.get("agent_template", {}), args.family, args.engine_kind, 4)
            names += EXTRA_MODULES
            summary["benchmark_profile"] = args.benchmark_profile
        else:
            require(packet.get("benchmark_profile") is None and
                    packet.get("agent_template", {}).get("benchmark_profile") is None,
                    "benchmark approval requires explicit CLI opt-in")
        workload = task_instance(packet, args.instance)
        if workload is not None:
            require(args.benchmark_profile is None, "task instances use the canonical scorer, not a benchmark profile")
            from coding_instances import MODULES as INSTANCE_MODULES
            names += INSTANCE_MODULES
            summary["instance"] = workload.record()
        require(args.phase_plan is None or (workload is None and args.benchmark_profile is None),
                "phase plans are controls of the canonical workload")
        sources = {name: child.digest(module_dir / name) for name in names}
        require(packet.get("engine_kind", "hf2q") == args.engine_kind, "engine approval mismatch")
        summary["engine_kind"] = args.engine_kind
        if args.engine_kind == "peer":
            from family_peer import readiness
            summary["peer_readiness"] = readiness(args.base_url, args.model, output, args.family)
        require(packet.get("sources") == sources, "N4 source freeze drift")
        require(packet.get("python_sha256") == child.digest(Path(sys.executable).resolve()), "Python binary")
        run, token = packet.get("run_id"), packet.get("barrier_token")
        identity(AGENTS[0], run)
        require(isinstance(token, str) and re.fullmatch(r"[a-f0-9]{32}", token), "barrier nonce")
        template = packet.get("agent_template", {})
        validate_preflight(template.get("preflight"), PREFLIGHT, args.engine_kind, args.family)
        summary["run_id"] = run
        if args.phase_plan is not None:
            summary["phase"] = {"plan": args.phase_plan, "cohorts": [list(c) for c in cohorts], "releases": []}
        barrier = output / "barrier"
        barrier.mkdir(mode=0o700)
        for agent in agents:
            require(not INTERRUPTED, "interrupted before any further client launch")
            path = output / agent
            binding = {**template, "schema": family_contract.schema(args.family, "agent-approval", 4),
                       "profile": family_contract.profile(args.family, 4), "family": args.family,
                       "engine_kind": args.engine_kind, "family_controls": packet.get("family_controls"),
                       "diagnostic_slots": 4, "agent_id": agent, "run_id": run,
                       "barrier_dir": str(barrier), "barrier_token": token, "agent_output": str(path),
                       "initial_identity_sha256": hashlib.sha256(identity(agent, run, args.instance).encode()).hexdigest(),
                       "sources": sources, "server_pid": args.server_pid, "base_url": args.base_url,
                       "model": args.model, "approved": True, **({} if workload is None else workload.binding())}
            approval_path = output / f"{agent}.approval.json"
            save_new(approval_path, binding)
            command = agent_command(args, module_dir, agent, path, approval_path, barrier, run, token)
            stdout, stderr = [(output / f"{agent}.{name}").open("xb") for name in ("stdout", "stderr")]
            streams += [stdout, stderr]
            require(not INTERRUPTED, "interrupted immediately before child launch")
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                       start_new_session=True, cwd=output)
            processes[agent], known[agent] = process, {}
            save_new(output / f"{agent}.command.json", {"argv": command, "pid": process.pid})
        records, hashes = {}, {}
        until = time.monotonic() + 120
        while len(records) != len(agents):
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
        release = barrier_release(records, run, args.server_pid, token, hashes, args.instance, agents)
        require(not INTERRUPTED, "interrupted before shared release")
        if args.phase_plan is not None:
            release.update(phase_plan=args.phase_plan, cohort=1)
        release["f2_clock_id"] = CLOCK_ID
        release["release_monotonic_ns"] = clock_ns()
        summary["f2_clock_id"] = CLOCK_ID
        save_new(barrier / "release.json", release)
        summary["makespan_started_ns"] = release["release_monotonic_ns"]
        summary["barrier"] = {"status": "Released", "ready": records,
                              "release_sha256": child.digest(barrier / "release.json"),
                              "release_monotonic_ns": release["release_monotonic_ns"]}
        if args.phase_plan is not None:
            summary["phase"]["releases"].append({"cohort": 1, "release_monotonic_ns": release["release_monotonic_ns"],
                                                 "trigger_first_decode_monotonic_ns": None})
        next_cohort = 2
        while any(process.poll() is None for process in processes.values()):
            require(not INTERRUPTED and time.monotonic() - started < 7320, "N4 deadline/interruption")
            os.kill(args.server_pid, 0)
            rows = child.process_snapshot()
            for agent, process in processes.items():
                descendants(rows, process.pid, known[agent])
                require(process.poll() in (None, 0), "one agent failed; no partial success")
            if next_cohort <= len(cohorts):
                waiting_on = cohorts[next_cohort - 2]
                trigger = phase_trigger(barrier, waiting_on, run, token)
                if trigger is None:
                    require(all(processes[a].poll() is None for a in waiting_on),
                            "phase trigger agent exited before its first decoded token")
                else:
                    later = {**release, "cohort": next_cohort, "release_monotonic_ns": clock_ns()}
                    save_new(barrier / release_name(next_cohort), later)
                    summary["phase"]["releases"].append({"cohort": next_cohort,
                                                         "release_monotonic_ns": later["release_monotonic_ns"],
                                                         "trigger_first_decode_monotonic_ns": trigger})
                    next_cohort += 1
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
                summary["status"] = aggregate(results, exits, cleanup, args.server_pid, run, args.engine_kind, args.family,
                                              summary.get("instance"), args.phase_plan)
            except BaseException as exc:
                summary["status"], summary["error"] = "Failed", str(exc)
        if summary["status"] == "FourAgentProtocolCompleteAwaitingIndependentReview":
            completed = [results[agent].get("task_completed_monotonic_ns") for agent in agents]
            if all(type(t) is int and t >= summary["makespan_started_ns"] for t in completed):
                summary["makespan_finished_ns"] = max(completed)
                summary["own_output_four_task_makespan_seconds"] = (
                    summary["makespan_finished_ns"] - summary["makespan_started_ns"]) / 1e9
            else:
                summary["status"], summary["error"] = "Failed", "missing validated task completion clocks"
        summary.update(agents=results, exits=exits, client_cleanup=receipts,
                       cleanup=len(cleanup) == len(agents) and all(cleanup.values()),
                       elapsed_seconds=time.monotonic() - started)
        if args.benchmark_profile is not None and summary["status"] == "FourAgentProtocolCompleteAwaitingIndependentReview":
            try:
                from gemma_matched_stats import cohort_score
                summary["benchmark_metrics"] = cohort_score(results)
            except BaseException as exc:
                summary["status"], summary["error"] = "Failed", str(exc)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if summary["status"] == "FourAgentProtocolCompleteAwaitingIndependentReview" else 1


if __name__ == "__main__":
    raise SystemExit(main())
