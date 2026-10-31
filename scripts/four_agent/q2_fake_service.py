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
import socket
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


RENDEZVOUS_HOLD = 0.1
# Every construction wait (rendezvous, admission gate, client close) fails after this many seconds.
# It stays below drive_agent's 30 s request deadline, so a stall is reported as the named wait.
WAIT_TIMEOUT = 20.0


def rendezvous_groups(phase_plan):
    """Streamed (agent, turn) sets a genuinely concurrent service holds open together.

    n4: every agent's first streamed turn. K+M: the earlier cohort's second streamed turn (turn 3)
    with the later cohort's first (turn 1); the later cohort is released during the earlier
    cohort's turn 1, so both are reached without either waiting on the other. "1": none."""
    cohorts = coordinator.phase_cohorts(phase_plan)
    if len(cohorts) == 1:
        return [] if len(cohorts[0]) == 1 else [{(agent, 1) for agent in cohorts[0]}]
    return [{(agent, 3) for agent in cohorts[0]} | {(agent, 1) for agent in cohorts[1]}]


def first_progress_ack(directory, agent, turn):
    """Marker a live client writes once it has stamped its first progress frame of `turn`."""
    return Path(directory) / f"{agent}.turn-{turn}.first-progress"


def acknowledging(path, then=None):
    """`on_first_token` hook: write the rendezvous acknowledgement `path`, then run `then`.

    `exchange` calls the hook after it has appended the first progress timestamp."""
    def hook(monotonic_ns):
        path.touch(exist_ok=False)
        if then is not None:
            then(monotonic_ns)
    return hook


class Rendezvous:
    """One group of member streams, (agent, turn), that a concurrent service holds mid-stream together.

    A member stamps `sent` just before it writes its first progress frame. It then waits until every
    member has done so and every member's client has acknowledged stamping that frame
    (`first_progress_ack`). Observing all of that at clock t, it holds until
    t + max(RENDEZVOUS_HOLD, t - earliest sent) before it writes the rest of its stream.

    Why the audit's overlap fraction is then >= 0.5 whatever the scheduling: let F be the group's
    latest first-progress stamp, and j the member whose last progress stamp L is earliest. Each
    acknowledgement is written after its stamp, so F <= t_j. Member j's last frame is written after
    its hold, so L >= t_j + hold_j and the common window L - F >= hold_j. A client cannot stamp a
    frame before it is written, so f_j >= earliest sent and F - f_j <= t_j - earliest sent <= hold_j.
    Member j's active interval (L - F) + (F - f_j) is therefore at most twice the window, and so is
    the shortest member's. All stamps are CLOCK_MONOTONIC on the one host."""

    def __init__(self, members, acks):
        self.members, self.acks = sorted(members), acks
        self.sent, self.lock = {}, threading.Lock()

    def hold(self, member, sent_ns):
        """Called after `member` wrote its first progress frame; raises TimeoutError naming what is missing."""
        with self.lock:
            self.sent[member] = sent_ns
        limit = time.monotonic() + WAIT_TIMEOUT
        while True:
            with self.lock:
                written, earliest = len(self.sent), min(self.sent.values())
            missing = [m for m in self.members if not first_progress_ack(self.acks, *m).exists()]
            if written == len(self.members) and not missing:
                break
            if time.monotonic() > limit:
                raise TimeoutError(f"rendezvous of {member[0]} turn {member[1]}: first progress of "
                                   f"{', '.join(f'{a} turn {t}' for a, t in missing)} not acknowledged "
                                   f"after {WAIT_TIMEOUT}s")
            time.sleep(0.001)
        now = clock_ns()
        until = now + max(round(RENDEZVOUS_HOLD * 1e9), now - earliest)
        while (now := clock_ns()) < until:
            time.sleep((until - now) / 1e9)


class FakeService:
    """Loopback /v1/chat/completions.

    Concurrency, and its absence, is by construction, never by timing luck:
    - `rendezvous=[{(agent, turn), ...}]` (concurrent service only; see `Rendezvous`): the members
      of a group are mid-stream together with an overlap fraction of at least 0.5. Live clients
      acknowledge their first progress frame in `self.acks` (run_live passes it to drive_agent).
    - `slots=N` serves at most N requests at a time end to end (the rest wait with their HTTP
      requests open); `slots=1` is a serialized service. A slot is held until the client has closed
      its connection, i.e. after it stamped every frame, so more than N client-observed active
      intervals never share an instant, however loaded the host is.
    - `admit_together=N` holds the first N requests until all N are open (deterministic HTTP overlap).
    A rendezvous, admission or close wait that fails is recorded and fails `__exit__` loudly.
    `bursty=True` streams the low-frame shape of hf2q Gemma: one reasoning frame, a pause of
    `timing.gap`, then the tool name and two argument frames back to back (four progress frames)."""

    def __init__(self, timing=None, faults=None, model=MODEL, slots=None, bursty=False, rendezvous=(),
                 admit_together=None):
        self.timing, self.model, self.bursty = timing or Timing(), model, bursty
        assert not (slots and rendezvous), "a slot-capped service cannot hold a rendezvous group open"
        self.slots = None if slots is None else threading.BoundedSemaphore(slots)
        self.acks = Path(tempfile.mkdtemp(prefix="q2-rendezvous-")) if rendezvous else None
        self.rendezvous = {member: group for members in rendezvous
                           for group in [Rendezvous(members, self.acks)] for member in members}
        self.admission = None if admit_together is None else threading.Barrier(admit_together)
        self.faults = dict(faults or {})
        self.state = threading.Lock()
        self.requests, self.call_ids, self.response_ids, self.errors = [], {}, {}, []
        self.counter = self.admitted = 0
        service = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                accepted = clock_ns()
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if service.admission is not None:
                    service.admit()
                if service.slots is not None:
                    with service.slots:
                        service.respond(self, body, accepted)
                        service.await_client_close(self)
                else:
                    service.respond(self, body, accepted)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)

    def fail(self, reason):
        with self.state:
            self.errors.append(reason)

    def admit(self):
        """The first `admit_together` requests proceed only once all of them are open."""
        with self.state:
            self.admitted += 1
            gated = self.admitted <= self.admission.parties
        if gated:
            try:
                self.admission.wait(WAIT_TIMEOUT)
            except threading.BrokenBarrierError:
                self.fail(f"admission gate of {self.admission.parties} requests not filled after {WAIT_TIMEOUT}s")

    def await_client_close(self, handler):
        """Half-close (the client sees end of stream), then hold the slot until the client closes."""
        try:
            handler.wfile.flush()
            handler.connection.shutdown(socket.SHUT_WR)
            handler.connection.settimeout(WAIT_TIMEOUT)
            while handler.connection.recv(4096):
                pass
        except OSError as error:
            self.fail(f"client close not observed: {type(error).__name__}: {error}")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exception):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        if self.acks is not None:
            shutil.rmtree(self.acks, ignore_errors=True)
        if self.errors:  # raised over (and chained to) any exception already propagating
            raise AssertionError("fake service construction failed: " + "; ".join(self.errors))

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
            group = self.rendezvous.get((agent, turn))
            for index, (delay, frame) in enumerate(frames):
                if delay:
                    time.sleep(delay)
                sent = clock_ns()
                handler.wfile.write(frame)
                handler.wfile.flush()
                if index == 1 and group is not None:  # frames[1] is the first progress frame
                    try:
                        group.hold((agent, turn), sent)
                    except TimeoutError as error:
                        self.fail(str(error))
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
            pieces = 1 if self.bursty else REASONING_FRAMES
            size = -(-len(reasoning) // pieces)
            for index in range(pieces):
                piece = reasoning[index * size:(index + 1) * size]
                frames.append((self.timing.first if index == 0 else gap, delta({"reasoning_content": piece})))
            if fault != "reasoning_only":
                if call is not None:
                    frames.append((gap, delta({"tool_calls": [{"index": 0, "id": call["id"], "type": "function",
                                                               "function": {"name": call["function"]["name"]}}]})))
                    arguments = call["function"]["arguments"]
                    for part in (arguments[:1], arguments[1:]) if self.bursty else (arguments,):
                        frames.append((0 if self.bursty else gap,
                                       delta({"tool_calls": [{"index": 0, "function": {"arguments": part}}]})))
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
                phase=None, release_ns=None, on_first_token=None, server_pid=SERVER_PID, acks=None):
    """The agent child's turn loop with stub tools; raises the actual client rejection on a bad response.

    `acks` (a rendezvous service's `acks` directory): acknowledge the first progress frame of every
    streamed turn there, after the actual `exchange` stamped it."""
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
        if acks is not None:
            callback = acknowledging(first_progress_ack(acks, agent, turn), callback)
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


def retime(results, phase_plan, makespan_s, slots=None, origin=10 ** 12):
    """Deterministic lockstep clocks over real receipts; returns (releases, triggers).

    `slots=N` (canonical plan only) runs the agents N at a time, each group after the previous one
    finished, as a server capped at N slots would; `slots=1` is a serialized service. A later
    cohort's streamed turns line up with the earlier cohort's (its turn t with their turn t+2), so
    a genuine mixed population reaches all four co-active agents. Event counts stay those of the
    raw frames. Semantic events are the last frames of each stream, as the fake service sends them,
    and every v3 timing record is recomputed with the actual function."""
    cohorts = coordinator.phase_cohorts(phase_plan)
    agents = coordinator.active_agents(phase_plan)
    assert slots is None or len(cohorts) == 1, "slot-capped retiming is for single-release plans"
    groups = 1 if slots is None else -(-len(agents) // slots)
    total = round(makespan_s * 1e9)
    duration = total // (7 * groups + (2 if len(cohorts) > 1 else 0) + 1)
    gap = duration // 10
    releases, triggers, start = [origin], [None], {}
    for index, cohort in enumerate(cohorts):
        if index:
            trigger = {a: start[a] + duration + agents.index(a) * (gap // 8) + gap for a in cohorts[index - 1]}
            releases.append(max(trigger.values()) + 1)
            triggers.append(trigger)
        for agent in cohort:
            group = 0 if slots is None else agents.index(agent) // slots
            start[agent] = releases[index] + 1 + group * 7 * duration
    for agent in agents:
        lane = agents.index(agent) * (gap // 8)
        cohort = [agent in c for c in cohorts].index(True)
        result = results[agent]
        result["barrier"]["release_monotonic_ns"] = releases[cohort]
        for turn, receipt in enumerate(result["turns"]):
            if cohort and turn:
                send = start[cohorts[0][0]] + (turn + 2 * cohort) * duration + lane
            else:
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


def honest_run(client_dir, makespan_s=100.0, engine_kind="hf2q", phase_plan=None, instance=None, slots=None,
               family="qwen"):
    """Write a complete client directory (raw turns + coordinator summary) whose evidence re-derives its Pass."""
    root, results = template(engine_kind, phase_plan, instance, family)
    client_dir = Path(client_dir)
    client_dir.mkdir(parents=True)
    for agent in results:
        shutil.copytree(root / agent, client_dir / agent)
    results = copy.deepcopy(results)
    releases, triggers = retime(results, phase_plan, makespan_s, slots)
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
                  "family": family, "phase_plan": phase_plan, "barrier": str(barrier),
                  "acks": None if service.acks is None else str(service.acks)}
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
                phase=phase, release_ns=release_ns, on_first_token=marker, acks=config["acks"])
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["agent"]:
        raise SystemExit(agent_main(json.loads(sys.argv[2])))
    raise SystemExit("usage: q2_fake_service.py agent CONFIG_JSON (test support; launched by run_live)")
