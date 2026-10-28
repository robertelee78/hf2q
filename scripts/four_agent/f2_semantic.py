"""F2 v3 timing arithmetic; semantic classification belongs to SSEAccumulator.feed."""
import math
import time

CLOCK_ID = "CLOCK_MONOTONIC"

def clock_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

SCHEMA = "hf2q.f2.semantic-timing.v3"
MAX_SEMANTIC_EVENTS = 131072
# Four independent client processes reserve <=128MiB combined, including Python
# event objects, references and serialization headroom, across ALL seven turns.
CLIENT_TRACE_BUDGET = 32 * 1024 * 1024
_events_reserved = 0


def append_semantic_event(events, timestamp_ns):
    global _events_reserved
    require(len(events) < MAX_SEMANTIC_EVENTS and
            (_events_reserved + 1) * 128 <= CLIENT_TRACE_BUDGET,
            "client trace quota exceeded; no dropped event can qualify")
    require(type(timestamp_ns) is int, "monotonic event timestamp")
    events.append(timestamp_ns)
    _events_reserved += 1


def require(ok, why):
    if not ok:
        raise ValueError(why)


def semantic_timing_v3(send_ns, semantic_ns, terminal_ns, stream, validated):
    require(type(send_ns) is int and type(terminal_ns) is int and
            0 <= send_ns <= terminal_ns, "monotonic send/validated terminal required")
    require(type(stream) is bool and validated is True, "validated terminal required")
    require(type(semantic_ns) is list and len(semantic_ns) <= MAX_SEMANTIC_EVENTS,
            "bounded semantic event vector required")
    require(all(type(x) is int and send_ns <= x <= terminal_ns for x in semantic_ns)
            and all(a <= b for a, b in zip(semantic_ns, semantic_ns[1:])),
            "ordered in-request semantic timestamps required")
    if not stream:
        require(not semantic_ns, "unary has no fictional streamed token timestamp")
        return {"schema": SCHEMA, "clock_id": CLOCK_ID, "stream": False,
                "unary_completion_seconds": (terminal_ns - send_ns) / 1e9,
                "semantic_ttft_seconds": None, "semantic_gap_end_to_end_seconds": None,
                "semantic_gap_inter_events_seconds": None,
                "semantic_terminal_gap_seconds": None, "semantic_events": 0}
    require(bool(semantic_ns), "no semantic streamed progress")
    first = semantic_ns[0] - send_ns
    terminal = terminal_ns - semantic_ns[-1]
    adjacent = [b - a for a, b in zip(semantic_ns, semantic_ns[1:])]
    return {"schema": SCHEMA, "clock_id": CLOCK_ID, "stream": True, "unary_completion_seconds": None,
            "semantic_ttft_seconds": first / 1e9,
            "semantic_gap_end_to_end_seconds": max([first, terminal, *adjacent]) / 1e9,
            "semantic_gap_inter_events_seconds": max(adjacent) / 1e9 if adjacent else None,
            "semantic_terminal_gap_seconds": terminal / 1e9,
            "semantic_events": len(semantic_ns)}


def slot_semantic_summary(turns):
    """Nearest-rank per-slot summaries; never pool agents or unary latency as TTFT."""
    require(bool(turns), "turn receipts required")
    values = [r["semantic_timing_v3"]["semantic_ttft_seconds"] for r in turns
              if r["semantic_timing_v3"]["stream"]]
    require(bool(values) and all(type(x) in (float, int) and math.isfinite(x) and x >= 0
                                 for x in values), "eligible streamed TTFT values required")
    ordered = sorted(values)
    return {"schema": SCHEMA, "clock_id": CLOCK_ID, "streamed_requests": len(values), "raw_ttft_seconds": values,
            "ttft_p50_seconds": ordered[math.ceil(.5 * len(ordered)) - 1],
            "ttft_p95_seconds": ordered[math.ceil(.95 * len(ordered)) - 1],
            "ttft_max_seconds": ordered[-1],
            "semantic_gap_end_to_end_max_seconds": max(
                r["semantic_timing_v3"]["semantic_gap_end_to_end_seconds"]
                for r in turns if r["semantic_timing_v3"]["stream"])}
