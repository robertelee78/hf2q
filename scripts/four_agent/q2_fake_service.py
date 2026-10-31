#!/usr/bin/env python3
"""Model-free OpenAI-compatible fake service and honest four-agent runs for the ADR-063 Q2 controls.

Test support only: test_q2_adversarial_controls.py, test_q3_campaign.py and
test_quality_campaign.py import it. Nothing here is inference, model, GPU or hardware
evidence. The service answers the canonical seven-turn coding workflow over real loopback
HTTP. A fault table turns single responses adversarial. Clients are the actual
`qwen38_n1_smoke.exchange` and parser; coordinator verdicts come from the actual
`qwen_n4_live_smoke.aggregate`. Tools are stubs; their receipts carry no compiler evidence.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import atexit
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import coding_family as family_contract
import coding_instances
from f2_semantic import CLOCK_ID, clock_ns, semantic_timing_v3
import qwen38_n1_process as child
from qwen38_n1_smoke import build_request, exchange, initial_messages
from qwen38_n1_verdict import EXPECTED, loads, qualify
import qwen_n4_live_smoke as coordinator

HERE = Path(__file__).resolve().parent
MODEL = "fake-q2-model"
RUN = "q2-fake-run-0000000001"
TOKEN = "0123456789abcdef0123456789abcdef"
SERVER_PID = 4242
REASONING_FRAMES = 4
FIXED_SOURCE = ("pub fn parse_port(s: &str) -> Result<u16, String> {\n"
                "    match s.parse::<u16>() { Ok(0) | Err(_) => Err(format!(\"invalid port {s}\")), Ok(p) => Ok(p) }\n}\n")
FINAL_SOURCE_SHA256 = hashlib.sha256(FIXED_SOURCE.encode()).hexdigest()
CLEAN = {"actions": [], "errors": [], "warnings": [], "reaped": True, "verified": True,
         "final_observation": {"available": True, "live_owned_pids": []}}


class Timing:
    """Known server-side delays (seconds): before the first streamed delta, between frames, unary latency."""
    def __init__(self, first=0.0, gap=0.0, unary=0.0):
        self.first, self.gap, self.unary = first, gap, unary

    def scaled(self, factor):
        return Timing(self.first * factor, self.gap * factor, self.unary * factor)


def tool_arguments(turn):
    name, arguments = EXPECTED[turn]
    return name, ({"path": "src/ports.rs", "content": FIXED_SOURCE} if arguments is None else arguments)


class FakeService:
    """Loopback /v1/chat/completions; `serialized=True` serves one request at a time end to end."""

    def __init__(self, timing=None, serialized=False, faults=None, model=MODEL):
        self.timing, self.serialized, self.model = timing or Timing(), serialized, model
        self.faults = dict(faults or {})
        self.lock, self.state = threading.Lock(), threading.Lock()
        self.requests, self.call_ids, self.response_ids = [], {}, {}
        self.counter = 0
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                accepted = clock_ns()
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if service.serialized:
                    with service.lock:
                        service.respond(self, body, accepted)
                else:
                    service.respond(self, body, accepted)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def fault(self, agent, turn):
        return self.faults.get((agent, turn), self.faults.get((None, turn)))

    def respond(self, handler, body, accepted):
        started = clock_ns()
        request = json.loads(body)
        messages = request["messages"]
        agent = re.search(r"coding agent (agent-\d)", messages[0]["content"]).group(1)
        turn = sum(1 for m in messages if m["role"] == "assistant")
        fault = self.fault(agent, turn)
        with self.state:
            self.counter += 1
            rid = f"chatcmpl-{uuid.uuid4().hex}"
            if fault == "reused_response_id":
                rid = self.response_ids[(agent, 0)]
            self.response_ids.setdefault((agent, turn), rid)
            cid = f"call_{agent}_{turn}_{self.counter:06d}"
            if fault == "reused_call_id":
                cid = self.call_ids[(agent, 0)]
            self.call_ids.setdefault((agent, turn), cid)
        model = "swapped-model" if fault == "model_swap" else request["model"]
        prompt = max(256, len(body) // 4)
        usage = {"prompt_tokens": prompt, "completion_tokens": 40 + turn, "total_tokens": prompt + 40 + turn,
                 "prompt_tokens_details": {"cached_tokens": 0 if turn == 0 else prompt - 128}}
        if fault == "truncated_stop":
            usage.update(completion_tokens=8192, total_tokens=prompt + 8192)
        elif fault == "inflated_budget":
            usage.update(completion_tokens=9000, total_tokens=prompt + 9000)
        elif fault == "usage_total_lie":
            usage["total_tokens"] += 1
        elif fault == "cached_over_prompt":
            usage["prompt_tokens_details"]["cached_tokens"] = prompt + 1
        elif fault == "cold_cached":
            usage["prompt_tokens_details"]["cached_tokens"] = 64
        elif fault == "no_reuse":
            usage["prompt_tokens_details"]["cached_tokens"] = 0
        reasoning = f"Step {turn + 1}: following the six-step workflow."
        if fault == "foreign_identity":
            other = coordinator.AGENTS[(coordinator.AGENTS.index(agent) + 1) % len(coordinator.AGENTS)]
            reasoning += f" Independent coding agent {other}, workload {coordinator.WORKLOAD_ID}."
        if turn == 6:
            call, finish, content = None, "stop", "" if fault == "empty_final" else \
                "Fixed parse_port: it rejects zero and values above 65535; ports and all tests passed."
        else:
            name, arguments = tool_arguments(turn)
            if fault == "wrong_args":
                arguments = {"path": "tests/ports.rs"} if name == "read_file" else {"target": "everything"}
            text = json.dumps(arguments, separators=(",", ":"))
            if fault == "truncated_args":
                text = text[:len(text) // 2]
            call, finish, content = {"id": cid, "type": "function", "function": {"name": name, "arguments": text}}, \
                "tool_calls", None
        if fault == "length_finish":
            finish = "length"
        if request.get("stream"):
            frames = self.stream_frames(rid, model, reasoning, call, content, finish, usage, fault)
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Connection", "close")
            handler.end_headers()
            for index, (delay, frame) in enumerate(frames):
                if delay:
                    time.sleep(delay)
                handler.wfile.write(frame)
                handler.wfile.flush()
        else:
            time.sleep(self.timing.unary)
            message = {"role": "assistant", "content": content, "reasoning_content": reasoning}
            if call is not None:
                message["tool_calls"] = [call]
            payload = json.dumps({"id": rid, "object": "chat.completion", "model": model,
                                  "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                                  "usage": usage}).encode()
            handler.send_response(200)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(payload)))
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(payload)
        handler.close_connection = True
        with self.state:
            self.requests.append({"agent": agent, "turn": turn, "accepted_ns": accepted, "started_ns": started,
                                  "finished_ns": clock_ns(), "stream": bool(request.get("stream"))})

    def stream_frames(self, rid, model, reasoning, call, content, finish, usage, fault):
        def frame(choices, extra=None):
            obj = {"id": rid, "object": "chat.completion.chunk", "model": model, "choices": choices, **(extra or {})}
            return ("data: " + json.dumps(obj) + "\n\n").encode()

        def delta(value):
            return frame([{"index": 0, "delta": value}])

        gap = self.timing.gap
        frames = [(0, delta({"role": "assistant"}))]
        if fault != "role_only":
            size = -(-len(reasoning) // REASONING_FRAMES)
            for index in range(REASONING_FRAMES):
                piece = reasoning[index * size:(index + 1) * size]
                frames.append((self.timing.first if index == 0 else gap, delta({"reasoning_content": piece})))
            if fault != "reasoning_only":
                if call is not None:
                    frames.append((gap, delta({"tool_calls": [{"index": 0, "id": call["id"], "type": "function",
                                                               "function": {"name": call["function"]["name"]}}]})))
                    frames.append((gap, delta({"tool_calls": [{"index": 0, "function": {
                        "arguments": call["function"]["arguments"]}}]})))
                else:
                    frames.append((gap, delta({"content": content})))
        frames.append((0, frame([{"index": 0, "delta": {}, "finish_reason": finish}])))
        frames.append((0, frame([], {"usage": usage})))
        frames.append((0, b"data: [DONE]\n\n"))
        return frames


class StubTools:
    """Tool-receipt shape the coordinator scores; no file system sandbox or compiler is involved."""

    def __init__(self, root, workload=None):
        self.fixture = Path(root) / "tools" / "fixture"
        (self.fixture / "src").mkdir(parents=True)
        self.files = {"src/ports.rs": "pub fn parse_port(s: &str) -> u16 { s.parse().unwrap() }\n" if workload is None
                      else workload.initial,
                      "tests/ports.rs": "#[test] fn ports() {}\n"}
        self.receipts = []
        self.save()

    def save(self):
        (self.fixture / "src" / "ports.rs").write_text(self.files["src/ports.rs"])

    @property
    def source_hash(self):
        return hashlib.sha256(self.files["src/ports.rs"].encode()).hexdigest()

    def invoke(self, name, args):
        receipt = {"name": name, "arguments": args}
        if name == "read_file":
            result = self.files[args["path"]]
        elif name == "write_file":
            self.files[args["path"]] = args["content"]
            self.save()
            result = f"wrote {len(args['content'].encode())} bytes"
        else:
            receipt.update(target=args["target"], passed=True)
            result = json.dumps({"target": args["target"], "passed": True, "source_sha256": self.source_hash,
                                 "output": "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"})
        receipt.update(result=result, source_sha256=self.source_hash, cleanup=True, status="Complete")
        self.receipts.append(receipt)
        return result


def drive_agent(base, out, agent, run=RUN, model=MODEL, engine_kind="hf2q", family="qwen", instance=None,
                phase=None, release_ns=None, on_first_token=None, server_pid=SERVER_PID):
    """The agent child's turn loop with stub tools; raises the actual client rejection on a bad response."""
    out = Path(out)
    out.mkdir(parents=True)
    workload = None if instance is None else coding_instances.select(instance)
    fixture = loads((HERE / "fixtures" / "canonical-coding-requests.json").read_bytes())
    initial = fixture[0]["request"]
    prompt_identity = coordinator.identity(agent, run, instance)
    messages = initial_messages(initial["messages"], prompt_identity, workload)
    schemas = initial["tools"]
    for tool in schemas:
        tool["function"]["strict"] = False
    release_ns = clock_ns() if release_ns is None else release_ns
    summary = {"schema": family_contract.schema(family, "agent-result", 4), "profile": family_contract.profile(family, 4),
               "family": family, "engine_kind": engine_kind, "status": "Pending", "cleanup": False,
               "model_qualified": False, "runtime_qualified": False, "agent_id": agent, "run_id": run,
               "server_pid": server_pid, "diagnostic_slots": 4, "full_prefix_reuse": "PendingPostOracle",
               "barrier": {"status": "Released", "release_monotonic_ns": release_ns, "f2_clock_id": CLOCK_ID},
               "initial_identity_sha256": hashlib.sha256(prompt_identity.encode()).hexdigest(),
               "initial_messages_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
                                                         .encode()).hexdigest(),
               "turns": [], "tools": [], "f2_clock_id": CLOCK_ID}
    if workload is not None:
        summary["instance"] = workload.record()
    if phase is not None:
        summary["phase"] = phase
    tools, seen = StubTools(out, workload), set()
    for turn in range(7):
        request = build_request(model, messages, schemas, turn, family, engine_kind)
        turn_out = out / f"turn-{turn}"
        turn_out.mkdir()
        callback = on_first_token if turn == 1 else None
        message, call, receipt = exchange(base, request, turn_out, turn, seen, deadline=30, engine_kind=engine_kind,
                                          family=family, on_first_token=callback)
        summary["turns"].append(receipt)
        messages.append(message)
        if call is not None:
            cid, name, arguments = call
            seen.add(cid)
            messages.append({"role": "tool", "tool_call_id": cid, "name": name, "content": tools.invoke(name, arguments)})
    summary["tools"] = tools.receipts
    summary["status"] = qualify(summary["turns"], tools.receipts, True, tools.source_hash, tools.source_hash)
    summary.update(cleanup=True, final_source_sha256=tools.source_hash, task_completed_monotonic_ns=clock_ns(),
                   observed_final_source_sha256=child.digest(tools.fixture / "src" / "ports.rs"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def assemble(client_dir, results, releases, triggers, phase_plan, engine_kind="hf2q", family="qwen", instance=None,
             cleanup=None, run=RUN, server_pid=SERVER_PID):
    """Coordinator summary over `results`; its status is the actual coordinator aggregate."""
    agents = coordinator.active_agents(phase_plan)
    cleanup = cleanup or {agent: copy.deepcopy(CLEAN) for agent in agents}
    exits = {agent: 0 for agent in agents}
    record = None if instance is None else coding_instances.select(instance).record()
    summary = {"schema": family_contract.schema(family, "coordinator-result", 4), "profile": family_contract.profile(family, 4),
               "family": family, "status": "Pending", "server_pid": server_pid, "diagnostic_slots": 4, "cleanup": False,
               "runtime_qualified": False, "model_qualified": False, "full_prefix_reuse": "PendingPostOracle",
               "server_owned_by_outer_supervisor": True, "engine_kind": engine_kind, "run_id": run, "f2_clock_id": CLOCK_ID,
               "makespan_started_ns": releases[0],
               "barrier": {"status": "Released", "ready": {}, "release_sha256": "0" * 64,
                           "release_monotonic_ns": releases[0]}}
    if record is not None:
        summary["instance"] = record
    if phase_plan is not None:
        summary["phase"] = {"plan": phase_plan, "cohorts": [list(c) for c in coordinator.phase_cohorts(phase_plan)],
                            "releases": [{"cohort": i + 1, "release_monotonic_ns": at, "trigger_first_decode_monotonic_ns": t}
                                         for i, (at, t) in enumerate(zip(releases, triggers))]}
    summary["status"] = coordinator.aggregate(results, exits, {a: cleanup[a]["verified"] for a in agents}, server_pid, run,
                                              engine_kind, family, record, phase_plan)
    finished = max(results[a]["task_completed_monotonic_ns"] for a in agents)
    summary.update(makespan_finished_ns=finished, own_output_four_task_makespan_seconds=(finished - releases[0]) / 1e9,
                   agents=results, exits=exits, client_cleanup=cleanup, cleanup=True,
                   elapsed_seconds=(finished - releases[0]) / 1e9)
    Path(client_dir).mkdir(parents=True, exist_ok=True)
    (Path(client_dir) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def retime(results, phase_plan, makespan_s, serialized=False, origin=10 ** 12):
    """Deterministic lockstep (or one-at-a-time) clocks over real receipts; returns (releases, triggers).

    Event counts stay those of the raw frames. Semantic events are the last frames of each stream,
    as the fake service sends them, and every v3 timing record is recomputed with the actual function."""
    cohorts = coordinator.phase_cohorts(phase_plan)
    agents = coordinator.active_agents(phase_plan)
    total = round(makespan_s * 1e9)
    slots = 7 * len(agents) if serialized else 7 + (2 if len(cohorts) > 1 else 0)
    duration = total // (slots + 1)
    gap = duration // 10
    releases, triggers, start = [origin], [None], {}
    for index, cohort in enumerate(cohorts):
        if index:
            trigger = {a: start[a] + duration + agents.index(a) * (gap // 8) + gap for a in cohorts[index - 1]}
            releases.append(max(trigger.values()) + 1)
            triggers.append(trigger)
        for agent in cohort:
            start[agent] = releases[index] + 1 + (agents.index(agent) * 7 * duration if serialized else 0)
    for agent in agents:
        lane = 0 if serialized else agents.index(agent) * (gap // 8)
        result = results[agent]
        result["barrier"]["release_monotonic_ns"] = releases[[agent in c for c in cohorts].index(True)]
        for turn, receipt in enumerate(result["turns"]):
            send = start[agent] + turn * duration + lane
            count, semantic = len(receipt["progress_event_monotonic_ns"]), len(receipt["semantic_event_monotonic_ns"])
            progress = [send + (j + 1) * gap for j in range(count)]
            terminal = send + (count + 1) * gap if receipt["stream"] else send + duration // 2
            receipt.update(send_monotonic_ns=send, validated_terminal_monotonic_ns=terminal,
                           progress_event_monotonic_ns=progress,
                           semantic_event_monotonic_ns=progress[count - semantic:] if semantic else [],
                           elapsed_seconds=(terminal - send) / 1e9)
            receipt["semantic_timing_v3"] = semantic_timing_v3(send, receipt["semantic_event_monotonic_ns"], terminal,
                                                               receipt["stream"], True)
            receipt["semantic_ttft_seconds"] = receipt["semantic_timing_v3"]["semantic_ttft_seconds"]
        result["task_completed_monotonic_ns"] = result["turns"][-1]["validated_terminal_monotonic_ns"] + 1
    last = max(agents, key=lambda a: results[a]["task_completed_monotonic_ns"])
    assert results[last]["task_completed_monotonic_ns"] <= origin + total
    results[last]["task_completed_monotonic_ns"] = origin + total
    return releases, triggers


_TEMPLATES = {}
_TEMPLATE_ROOT = None


def template(engine_kind="hf2q", phase_plan=None, instance=None, family="qwen"):
    """Raw evidence of one honest run (cached per process): directory and per-agent results."""
    global _TEMPLATE_ROOT
    key = (engine_kind, phase_plan, instance, family)
    if key not in _TEMPLATES:
        if _TEMPLATE_ROOT is None:
            _TEMPLATE_ROOT = tempfile.TemporaryDirectory(prefix="q2-fake-templates-")
            atexit.register(_TEMPLATE_ROOT.cleanup)
        root = Path(_TEMPLATE_ROOT.name) / f"t{len(_TEMPLATES)}"
        results = {}
        with FakeService() as service:
            for agent in coordinator.active_agents(phase_plan):
                results[agent] = drive_agent(service.url, root / agent, agent, engine_kind=engine_kind, family=family,
                                             instance=instance, phase=coordinator.agent_phase(phase_plan, agent))
        _TEMPLATES[key] = (root, results)
    return _TEMPLATES[key]


def honest_run(client_dir, makespan_s=100.0, engine_kind="hf2q", phase_plan=None, instance=None, serialized=False,
               family="qwen"):
    """Write a complete client directory (raw turns + coordinator summary) whose evidence re-derives its Pass."""
    root, results = template(engine_kind, phase_plan, instance, family)
    client_dir = Path(client_dir)
    client_dir.mkdir(parents=True)
    for agent in results:
        shutil.copytree(root / agent, client_dir / agent)
    results = copy.deepcopy(results)
    releases, triggers = retime(results, phase_plan, makespan_s, serialized)
    return assemble(client_dir, results, releases, triggers, phase_plan, engine_kind, family, instance)


def write_json(path, value):
    pending = Path(str(path) + ".pending")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def run_live(service, client_dir, phase_plan=None, engine_kind="hf2q", family="qwen", deadline=60):
    """Four actual client processes against `service`, released together (or by --phase-plan cohorts)."""
    client_dir = Path(client_dir)
    barrier = client_dir / "barrier"
    barrier.mkdir(parents=True)
    cohorts = coordinator.phase_cohorts(phase_plan)
    processes = {}
    for agent in coordinator.active_agents(phase_plan):
        config = {"base": service.url, "out": str(client_dir / agent), "agent": agent, "engine_kind": engine_kind,
                  "family": family, "phase_plan": phase_plan, "barrier": str(barrier)}
        processes[agent] = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "agent",
                                             json.dumps(config)], cwd=HERE, start_new_session=True,
                                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    until = time.monotonic() + deadline
    try:
        while not all((barrier / f"{a}.ready").exists() for a in processes):
            assert time.monotonic() < until and all(p.poll() is None for p in processes.values()), "agents not ready"
            time.sleep(0.005)
        releases, triggers = [clock_ns()], [None]
        write_json(barrier / "release-1.json", {"release_monotonic_ns": releases[0]})
        while any(p.poll() is None for p in processes.values()):
            assert time.monotonic() < until, "live agents exceeded the deadline"
            if len(releases) < len(cohorts):
                trigger = coordinator.phase_trigger(barrier, cohorts[len(releases) - 1], RUN, TOKEN)
                if trigger is not None:
                    releases.append(clock_ns())
                    triggers.append(trigger)
                    write_json(barrier / f"release-{len(releases)}.json", {"release_monotonic_ns": releases[-1]})
            time.sleep(0.002)
    finally:
        cleanup = {agent: child.stop_group(process, {}) for agent, process in processes.items()}
    errors = {}
    for agent, process in processes.items():
        with process.stderr:
            if process.returncode != 0:
                errors[agent] = process.stderr.read().decode()
    assert not errors, errors
    results = {a: json.loads((client_dir / a / "summary.json").read_text()) for a in processes}
    return assemble(client_dir, results, releases, triggers, phase_plan, engine_kind, family, cleanup=cleanup)


def agent_main(config):
    """One live agent process: ready, wait for its cohort release, then the actual exchange loop."""
    barrier, agent, plan = Path(config["barrier"]), config["agent"], config["phase_plan"]
    phase = coordinator.agent_phase(plan, agent)
    cohort = 1 if phase is None else phase["cohort"]
    (barrier / f"{agent}.ready").write_text("ready\n")
    release = barrier / f"release-{cohort}.json"
    while not release.exists():
        time.sleep(0.001)
    time.sleep(0.001)
    release_ns = json.loads(release.read_text())["release_monotonic_ns"]
    marker = None
    if phase is not None and cohort < len(coordinator.phase_cohorts(plan)):
        def marker(ns):
            coordinator.first_decode_marker(barrier, agent, RUN, TOKEN, ns)
    drive_agent(config["base"], config["out"], agent, engine_kind=config["engine_kind"], family=config["family"],
                phase=phase, release_ns=release_ns, on_first_token=marker)
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["agent"]:
        raise SystemExit(agent_main(json.loads(sys.argv[2])))
    raise SystemExit("usage: q2_fake_service.py agent CONFIG_JSON (test support; launched by run_live)")
