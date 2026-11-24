"""Pure C2 cohort metrics from retained client summaries; no model qualification."""
import math

from coding_divergent_edit import TURNS, select
from f2_semantic import CLOCK_ID, SCHEMA
from q3_metrics import AGENT_SUCCESS, SUCCESS, slot_metrics, worst
from qwen38_n1_verdict import EXPECTED, max_tools

AGENTS = tuple(f"agent-{i}" for i in range(4))
DIVERGENT = "divergent_turn_worst_slot_ttft_s"
APPEND = ("makespan_s", "worst_slot_ttft_p95_s", "worst_slot_sse_gap_max_s")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def positive(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f"{name} must be positive and finite")
    return value


def _agents(summary):
    require(isinstance(summary, dict) and summary.get("status") == SUCCESS,
            "complete successful coordinator summary required")
    agents = summary.get("agents")
    require(isinstance(agents, dict) and set(agents) == set(AGENTS), "exactly four agents required")
    for name, agent in agents.items():
        require(isinstance(agent, dict) and agent.get("agent_id") == name and
                agent.get("status") == AGENT_SUCCESS, f"{name}: complete matching agent required")
    return agents


def _turns(agent):
    turns = agent.get("turns")
    require(isinstance(turns, list) and bool(turns), "retained turns required")
    by_id = {}
    for turn in turns:
        require(isinstance(turn, dict) and type(turn.get("turn")) is int and turn["turn"] >= 0,
                "actual nonnegative integer turn identity required")
        identity = turn["turn"]
        require(identity not in by_id, f"duplicate turn {identity}")
        by_id[identity] = turn
    return by_id


def _timing(turn):
    timing = turn.get("semantic_timing_v3")
    require(isinstance(timing, dict) and timing.get("schema") == SCHEMA and
            timing.get("clock_id") == CLOCK_ID and type(timing.get("stream")) is bool and
            type(turn.get("stream")) is bool and timing["stream"] == turn["stream"],
            "matching semantic_timing_v3 required")
    if timing["stream"]:
        require(type(timing.get("semantic_events")) is int and timing["semantic_events"] > 0,
                "semantic streamed progress required, not an empty role event")
        positive(timing.get("semantic_ttft_seconds"), "semantic TTFT")
    return timing


def divergent_metrics(summary):
    """Select actual turns 3/5, max per slot then max slots; retain every selected value."""
    require(summary.get("workload_variant") == select().record(),
            "pinned divergent workload required")
    slots = {}
    for name, agent in _agents(summary).items():
        turns = _turns(agent)
        selected = {}
        for identity in TURNS:
            require(identity in turns, f"{name}: missing divergent turn {identity}")
            timing = _timing(turns[identity])
            require(timing["stream"] is True, f"{name}: divergent turn {identity} must stream")
            selected[str(identity)] = timing["semantic_ttft_seconds"]
        slots[name] = {"turn_ttft_s": selected, "divergent_turns": list(TURNS),
                       "divergent_ttft_max_s": max(selected.values())}
    return {DIVERGENT: worst(slots, "divergent_ttft_max_s"), "slots": slots}


def append_metrics(summary):
    """Q3 append-only estimands, refusing incomplete timing/event coverage instead of omitting it."""
    require(summary.get("workload_variant") is None, "append-only workload required")
    slots = {}
    for name, agent in _agents(summary).items():
        turns = _turns(agent)
        repair = agent.get("repair_turns")
        limit = max_tools(0 if repair is None else repair) + 1
        tools = agent.get("tools")
        require(isinstance(tools, list), f"{name}: executed tool receipts required")
        require(len(turns) == len(tools) + 1, f"{name}: incomplete tool/turn coverage")
        require(set(turns) == set(range(len(turns))) and
                len(EXPECTED) + 1 <= len(turns) <= limit and len(turns) % 2 == 1,
                f"{name}: incomplete turn coverage")
        streamed = 0
        for turn in turns.values():
            timing = _timing(turn)
            require(turn["stream"] == (turn["turn"] % 2 == 1),
                    f"{name}: retained transport matrix mismatch")
            if not timing["stream"]:
                continue
            streamed += 1
            events = turn.get("semantic_event_monotonic_ns")
            require(isinstance(events, list) and len(events) == timing["semantic_events"] and
                    all(type(x) is int and x >= 0 for x in events) and
                    all(a <= b for a, b in zip(events, events[1:])),
                    f"{name}: complete ordered semantic events required")
        require(streamed > 0, f"{name}: streamed turn coverage required")
        slots[name] = slot_metrics(agent)
        require(slots[name]["sse_gap_max_s"] is not None,
                f"{name}: adjacent semantic SSE gap coverage required")
    return {"makespan_s": positive(summary.get("own_output_four_task_makespan_seconds"), "makespan"),
            "worst_slot_ttft_p95_s": worst(slots, "ttft_p95_s"),
            "worst_slot_sse_gap_max_s": positive(worst(slots, "sse_gap_max_s"), "worst SSE gap"),
            "slots": slots}
