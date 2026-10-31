"""ADR-063 Q2 settled-run audit: re-derive a four-agent client run from its raw evidence.

The Q3 speed campaign and the task-quality campaign call `check_run` before a Pass counts.
It does not trust the coordinator's status, makespan or timing fields. It re-reads every
archived request/response, replays it through the client's own parser, recomputes the
v3 timing and makespan arithmetic, checks the cohort/phase membership, and measures how
many agents made streamed progress at the same time.

Co-progress is measured from client progress events (`progress_event_monotonic_ns`: every
SSE frame that grew reasoning or semantic output). Agent B co-progresses with agent A when,
inside one of A's streamed requests, their frames alternate at least MIN_INTERLEAVE times.
Overlapping HTTP lifetimes alone count for nothing. This is a client-side lower bound on
concurrent slot progress. Physical batch membership (which owners share one forward)
needs the server phase trace (f2_trace.qualify_window).
"""
import bisect
import hashlib
import json
from pathlib import Path

from f2_semantic import semantic_timing_v3
from q2_certify import checked_cleanup, replay_response, request_wire, require_clock
from qwen38_n1_verdict import RAW_LIMIT, loads, require
import qwen_n4_live_smoke as coordinator

SUCCESS = "FourAgentProtocolCompleteAwaitingIndependentReview"
MIN_INTERLEAVE = 4
PASSING_Q5 = ("CalibrationPass", "AdmittedRunPass")


def expected_width(phase_plan):
    """Co-progress a genuine run of this plan must show: every agent of the largest cohort."""
    if phase_plan is None:
        return len(coordinator.AGENTS)
    return max(len(cohort) for cohort in coordinator.phase_cohorts(phase_plan))


def interleavings(own, other):
    """Agent switches in the merged frame order inside the span of `own` (one streamed request)."""
    low, high = own[0], own[-1]
    inside = other[bisect.bisect_right(other, low):bisect.bisect_left(other, high)]
    merged = sorted([(t, 0) for t in own] + [(t, 1) for t in inside])
    return sum(1 for a, b in zip(merged, merged[1:]) if a[1] != b[1])


def progress_width(agents):
    """{agent: turn receipts} -> observed co-progress width; 0 when no streamed request progressed twice."""
    events = {name: sorted(t for turn in turns for t in turn.get("progress_event_monotonic_ns") or [])
              for name, turns in agents.items()}
    width, witness = 0, None
    for name, turns in agents.items():
        for turn in turns:
            own = turn.get("progress_event_monotonic_ns") or []
            if len(own) < 2:
                continue
            peers = sorted(other for other in agents
                           if other != name and interleavings(own, events[other]) >= MIN_INTERLEAVE)
            if 1 + len(peers) > width:
                width, witness = 1 + len(peers), {"agent": name, "turn": turn.get("turn"), "co_progressing": peers}
    return {"width": width, "witness": witness, "min_interleave": MIN_INTERLEAVE,
            "basis": "client progress events (reasoning or semantic SSE frames)"}


def http_overlap(agents):
    """Maximum number of simultaneously open requests; never evidence of progress by itself."""
    edges = sorted(edge for turns in agents.values() for turn in turns
                   for edge in ((turn["send_monotonic_ns"], 1), (turn["validated_terminal_monotonic_ns"], -1)))
    open_now = peak = 0
    for _, step in edges:
        open_now += step
        peak = max(peak, open_now)
    return peak


def archived(path, sha256, size, what):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"archived {what} missing")
    require(type(size) is int and 0 <= size <= RAW_LIMIT and path.stat().st_size == size,
            f"archived {what} size differs from its receipt")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha256, f"archived {what} hash differs from its receipt")
    return data


def response_id(raw, stream):
    if not stream:
        return loads(raw)["id"]
    for line in raw.split(b"\n"):
        if line.startswith(b"data: {"):
            return loads(line[6:])["id"]
    raise ValueError("SSE response without an identified frame")


def ordered_within(stamps, low, high):
    return (type(stamps) is list and all(type(t) is int and low <= t <= high for t in stamps)
            and all(a <= b for a, b in zip(stamps, stamps[1:])))


def audit_agent(directory, name, result, family, engine, response_ids):
    """Replay one agent's seven archived exchanges; returns its turn receipts."""
    turns, tools = result["turns"], result["tools"]
    foreign = [f"coding agent {other}, workload {coordinator.WORKLOAD_ID}"
               for other in coordinator.AGENTS if other != name]
    seen, messages, model, previous_terminal = set(), None, None, None
    for turn, receipt in enumerate(turns):
        require_clock(receipt)
        base = Path(directory) / f"turn-{turn}"
        request_raw = archived(base / "request.json", receipt.get("request_sha256"), receipt.get("request_bytes"), "request")
        response_raw = archived(base / "response.bin", receipt.get("response_sha256"), receipt.get("response_bytes"),
                                "response")
        request = loads(request_raw)
        require(receipt.get("turn") == turn and receipt.get("stream") is request.get("stream") is (turn % 2 == 1),
                "turn/transport receipt")
        request_wire(request, request_raw, family, engine)
        model = request["model"] if model is None else model
        require(request["model"] == model, "model identity changed between turns")
        if turn == 0:
            initial = json.dumps(request["messages"], ensure_ascii=False, separators=(",", ":")).encode()
            require(hashlib.sha256(initial).hexdigest() == result.get("initial_messages_sha256"),
                    "stale lineage: first request is not this agent's initial transcript")
        else:
            require(request["messages"] == messages, "stale lineage: request history does not extend the validated transcript")
        semantic, progress = [], []
        message, call, stats = replay_response(response_raw, request, turn, seen, engine, semantic,
                                               progress_frames=progress)
        require(receipt.get("usage") == stats, "receipt usage differs from the raw response (usage/cache accounting)")
        rid = response_id(response_raw, request["stream"])
        require(rid not in response_ids, f"stale lineage: response id of {name} turn {turn} repeats {response_ids.get(rid)}")
        response_ids[rid] = f"{name} turn {turn}"
        text = "".join([message.get("content") or "", message.get("reasoning_content") or ""] +
                       [c["function"]["arguments"] for c in message.get("tool_calls") or []])
        require(not any(marker in text for marker in foreign), f"cross-slot leakage: {name} output carries another agent's identity")
        send, terminal = receipt.get("send_monotonic_ns"), receipt.get("validated_terminal_monotonic_ns")
        sem, prog = receipt.get("semantic_event_monotonic_ns"), receipt.get("progress_event_monotonic_ns")
        require(type(sem) is list and type(prog) is list and len(sem) == len(semantic) and len(prog) == len(progress),
                "receipt event counts differ from the raw SSE frames")
        require(receipt.get("semantic_timing_v3") == semantic_timing_v3(send, sem, terminal, request["stream"], True),
                "receipt timing differs from its clocks (timer direction/unit corruption)")
        require(ordered_within(prog, send, terminal) and set(sem) <= set(prog), "progress events outside their request")
        require(previous_terminal is None or send >= previous_terminal, "request sent before the previous response was validated")
        previous_terminal = terminal
        messages = request["messages"] + [message]
        if call is not None:
            cid, function, args = call
            seen.add(cid)
            tool = tools[turn]
            require(tool.get("name") == function and tool.get("arguments") == args and tool.get("status") == "Complete"
                    and tool.get("cleanup") is True and isinstance(tool.get("result"), str), "tool/argument join")
            messages.append({"role": "tool", "tool_call_id": cid, "name": function, "content": tool["result"]})
    return turns


def cohort_releases(summary, phase_plan, cohorts, results):
    """Release clock of each cohort, after checking the recorded phase plan and its triggers."""
    started = summary.get("makespan_started_ns")
    require(type(started) is int and started == summary["barrier"]["release_monotonic_ns"],
            "makespan start is not the recorded release")
    if phase_plan is None:
        require("phase" not in summary, "canonical run carries a phase record")
        return [started]
    phase = summary["phase"]
    require(phase["plan"] == phase_plan and phase["cohorts"] == [list(c) for c in cohorts], "recorded phase plan differs")
    releases = phase["releases"]
    require([r["cohort"] for r in releases] == list(range(1, len(cohorts) + 1)), "missing cohort release")
    require(releases[0]["release_monotonic_ns"] == started and releases[0]["trigger_first_decode_monotonic_ns"] is None,
            "first cohort release")
    clocks = [started]
    for index, release in enumerate(releases[1:], 1):
        earlier, trigger, at = cohorts[index - 1], release["trigger_first_decode_monotonic_ns"], release["release_monotonic_ns"]
        require(isinstance(trigger, dict) and set(trigger) == set(earlier) and type(at) is int, "cohort trigger membership")
        for agent in earlier:
            first_stream = results[agent]["turns"][1]
            require(type(trigger[agent]) is int and first_stream["send_monotonic_ns"] <= trigger[agent]
                    <= first_stream["validated_terminal_monotonic_ns"],
                    "first-decode trigger outside the agent's first streamed request")
            require(results[agent]["task_completed_monotonic_ns"] > at,
                    f"population never mixed: {agent} finished before cohort {index + 1} was released")
        require(at >= max(trigger.values()) and at >= clocks[-1], "cohort released before its trigger")
        clocks.append(at)
    return clocks


def audit_run(client_dir, summary, phase_plan, engine_kind, require_width):
    """Raises ValueError with the rejection reason; returns the observed progress report."""
    require(summary.get("status") == SUCCESS, "client did not claim four-agent success")
    require_clock(summary)
    require(summary.get("engine_kind") == engine_kind, "client summary engine differs from the arm")
    family, cohorts = summary.get("family"), coordinator.phase_cohorts(phase_plan)
    agents = coordinator.active_agents(phase_plan)
    results = summary.get("agents")
    require(isinstance(results, dict) and set(results) == set(agents), "agent set differs from the phase plan")
    cleanup = summary["client_cleanup"]
    for name in agents:
        checked_cleanup(cleanup[name])
    require(coordinator.aggregate(results, summary["exits"], {n: cleanup[n]["verified"] for n in agents},
                                  summary["server_pid"], summary["run_id"], engine_kind, family,
                                  summary.get("instance"), phase_plan) == summary["status"],
            "result scoring disagrees with the claimed status")
    releases = cohort_releases(summary, phase_plan, cohorts, results)
    response_ids, turns = {}, {}
    for index, cohort in enumerate(cohorts):
        for name in cohort:
            result = results[name]
            require_clock(result["barrier"])
            require(result["barrier"]["release_monotonic_ns"] == releases[index], "agent did not observe its cohort release")
            turns[name] = audit_agent(Path(client_dir) / name, name, result, family, engine_kind, response_ids)
            require(turns[name][0]["send_monotonic_ns"] >= releases[index], "agent request before its cohort release")
            require(type(result.get("task_completed_monotonic_ns")) is int and
                    result["task_completed_monotonic_ns"] >= turns[name][-1]["validated_terminal_monotonic_ns"],
                    "task completion before its last validated response")
    finished = max(results[name]["task_completed_monotonic_ns"] for name in agents)
    require(summary.get("makespan_finished_ns") == finished, "makespan end is not the last task completion")
    require(summary.get("own_output_four_task_makespan_seconds") == (finished - releases[0]) / 1e9,
            "makespan seconds disagree with its clocks (unit/arithmetic corruption)")
    progress = progress_width(turns)
    report = {"progress": progress, "expected_width": expected_width(phase_plan), "http_overlap_max": http_overlap(turns),
              "response_ids": len(response_ids), "cohort_release_ns": releases}
    require(not require_width or progress["width"] >= report["expected_width"],
            f"serialized service: observed co-progress width {progress['width']} < {report['expected_width']} "
            f"with up to {report['http_overlap_max']} overlapping HTTP requests")
    return report


def audit_receipt(receipt):
    """A passing Q5 verdict must agree with the receipt's own cleanup proof."""
    if (receipt.get("verdict") or {}).get("verdict") in PASSING_Q5:
        require(receipt.get("cleanup_proven") is True and (receipt.get("cleanup") or {}).get("proven") is True,
                "Q5 receipt contradiction: passing verdict without proven cleanup")


def check_run(receipt, client_dir, summary, phase_plan, engine_kind, require_width):
    """(report, None) for an audited run, (None, reason) for one whose evidence contradicts the claimed Pass."""
    try:
        audit_receipt(receipt)
        return audit_run(client_dir, summary, phase_plan, engine_kind, require_width), None
    except ValueError as error:
        return None, f"run audit: {error}"
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as error:
        return None, f"run audit: malformed evidence ({type(error).__name__}: {error})"
