"""ADR-063 Q3 per-run outcome and metrics from existing Q5/client receipts. Pure apart from file reads.

Inputs are what q5_calibration.py and qwen_n4_live_smoke.py already write: receipt.json,
server.stderr, readiness documents, client/summary.json with every agent's turn receipts.
Definitions are in docs/research/adr063-q3-campaign.md ("Per-run metrics").
"""
import json
from pathlib import Path

from q1_decision_policy import nearest_rank
from q5_policy import MEMORY_STOPS

SUCCESS = "FourAgentProtocolCompleteAwaitingIndependentReview"
AGENT_SUCCESS = "ProtocolCompleteAwaitingIndependentPatchReview"
PASSING_Q5 = ("CalibrationPass", "AdmittedRunPass")
ARM_FAILURE_STOPS = ("server_exited", "client_timeout")
LOAD_FAILURE_STOPS = ("server_exited", "readiness_timeout") + MEMORY_STOPS
# Metal allocation failures as the runtimes log them (the 2026-10-30 DeepSeek-V4 peer arm).
OOM_MARKERS = (b"kIOGPUCommandBufferCallbackErrorOutOfMemory", b"Insufficient Memory")
THERMAL_INVALID = 2  # NSProcessInfo.thermalState: 0 nominal, 1 fair, 2 serious, 3 critical


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def has_oom_marker(path, chunk=1 << 20):
    """Streamed search of a possibly large server log for a Metal out-of-memory marker."""
    overlap = max(len(m) for m in OOM_MARKERS)
    try:
        with open(path, "rb") as handle:
            tail = b""
            for block in iter(lambda: handle.read(chunk), b""):
                window = tail + block
                if any(m in window for m in OOM_MARKERS):
                    return True
                tail = window[-overlap:]
    except OSError:
        return False
    return False


def classify(receipt, summary, oom):
    """(outcome, kind, reason). Outcomes: Pass, Fail, NotStarted, Invalid.

    Kinds qualify failures: 'out_of_memory' (Metal OOM in server.stderr), 'memory_stop' (a Q5 live
    stop after the four-agent release), 'server_exited'/'client_timeout', 'task'; NotStarted is
    'preflight', 'load_failure' (server exit, readiness timeout or memory stop before the client) or
    'before_release' (client ended before the four-agent release).
    """
    if receipt is None:
        return "Invalid", None, "Q5 receipt missing"
    verdict, stop = receipt.get("verdict") or {}, receipt.get("stop")
    if receipt.get("client") is None:
        if verdict.get("verdict") == "PreflightRefused":
            return "NotStarted", "preflight", f"preflight refused: {verdict.get('reason')}"
        if stop is not None and stop.get("reason") in LOAD_FAILURE_STOPS:
            kind = "out_of_memory" if oom else "load_failure"
            return "NotStarted", "load_failure", f"{kind}: {stop['reason']}: {stop.get('detail')}"
        return "NotStarted", "other", f"no client launched: {verdict.get('verdict')}: {verdict.get('reason')}"
    if summary is None:
        return "Invalid", None, "client launched but wrote no summary"
    if summary.get("barrier", {}).get("status") != "Released":
        return "NotStarted", "before_release", "client ended before the four-agent release: " + str(summary.get("error"))
    if receipt.get("supervisor_error"):
        return "Invalid", None, "supervisor error after the task started: " + receipt["supervisor_error"]
    if stop is not None:
        reason = stop.get("reason")
        if reason in MEMORY_STOPS:
            return "Fail", "memory_stop", f"Q5 memory stop {reason}: {stop.get('detail')}"
        if reason in ARM_FAILURE_STOPS:
            return "Fail", "out_of_memory" if oom else reason, f"supervisor stop {reason}"
        return "Invalid", None, f"supervisor stop {reason} after the task started"
    if receipt["client"].get("exit_code") == 0 and summary.get("status") == SUCCESS:
        return "Pass", None, None
    return "Fail", "out_of_memory" if oom else "task", str(summary.get("error"))


def slot_metrics(agent):
    """Per-slot streamed semantic TTFT and adjacent semantic-event (SSE) gaps, nearest-rank p95."""
    streamed = [t for t in agent.get("turns", []) if (t.get("semantic_timing_v3") or {}).get("stream")]
    ttft = [t["semantic_timing_v3"]["semantic_ttft_seconds"] for t in streamed]
    gaps = []
    for turn in streamed:
        events = turn.get("semantic_event_monotonic_ns") or []
        gaps += [(b - a) / 1e9 for a, b in zip(events, events[1:])]
    return {"streamed_turns": len(ttft), "ttft_p95_s": nearest_rank(ttft, 0.95), "ttft_max_s": max(ttft, default=None),
            "sse_gaps": len(gaps), "sse_gap_p95_s": nearest_rank(gaps, 0.95), "sse_gap_max_s": max(gaps, default=None)}


def worst(slots, key):
    values = [s[key] for s in slots.values() if s[key] is not None]
    return max(values, default=None)


def peer_build_info(run_dir, receipt):
    for url, name in ((receipt.get("readiness") or {}).get("documents") or {}).items():
        if url.endswith("/props"):
            document = load_json(Path(run_dir) / name)
            return None if document is None else document.get("build_info")
    return None


def run_metrics(run_dir, receipt, summary, agents_expected, post_thermal):
    """Metrics of one settled run; values are None where the receipts do not define them."""
    agents = (summary or {}).get("agents") or {}
    slots = {name: slot_metrics(agent) for name, agent in sorted(agents.items())}
    passed = sum(1 for agent in agents.values() if agent.get("status") == AGENT_SUCCESS)
    makespan = (summary or {}).get("own_output_four_task_makespan_seconds")
    elapsed = ((receipt or {}).get("client") or {}).get("elapsed_seconds")
    samples = (receipt or {}).get("samples") or {}
    q5 = ((receipt or {}).get("verdict") or {}).get("verdict")
    preflight = ((receipt or {}).get("preflight") or {}).get("thermal_state")
    observed = [x for x in (preflight, post_thermal) if x is not None]
    return {
        "makespan_s": makespan, "client_elapsed_s": elapsed,
        "tasks_passed": passed, "tasks_expected": agents_expected,
        "goodput_tasks_per_hour": None if not makespan else passed * 3600 / makespan,
        "slots": slots,
        "worst_slot_ttft_p95_s": worst(slots, "ttft_p95_s"), "worst_slot_ttft_max_s": worst(slots, "ttft_max_s"),
        "worst_slot_sse_gap_p95_s": worst(slots, "sse_gap_p95_s"), "worst_slot_sse_gap_max_s": worst(slots, "sse_gap_max_s"),
        "final_source_sha256": {name: agent.get("final_source_sha256") for name, agent in sorted(agents.items())},
        "phase": (summary or {}).get("phase"),
        "memory": {"q5_verdict": q5, "q5_pass": q5 in PASSING_Q5,
                   "min_free_plus_cached_bytes": samples.get("min_free_plus_cached_bytes"),
                   "swap_growth_bytes": (samples.get("deltas") or {}).get("swap_used_bytes"),
                   "pressure_counts": samples.get("pressure_counts"),
                   "peak_footprint_bytes": samples.get("peak_footprint_bytes"),
                   "max_compressor_bytes": samples.get("max_compressor_bytes"),
                   "effective_floor_bytes": ((receipt or {}).get("effective_floor") or {}).get("effective_floor_bytes")},
        "thermal": {"preflight_state": preflight, "post_run_state": post_thermal, "in_run": "unrecorded",
                    "valid": None if not observed else all(x < THERMAL_INVALID for x in observed)},
        "binary_sha256": ((receipt or {}).get("binary") or {}).get("sha256"),
        "artifact": {k: ((receipt or {}).get("artifact") or {}).get(k) for k in ("path", "bytes", "sha256")},
        "peer_build_info": None if receipt is None else peer_build_info(run_dir, receipt),
    }
