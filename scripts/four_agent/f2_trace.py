"""Closed F2 metadata trace validation and observed-population qualification.

No HTTP overlap, scheduler phase, temporal verifier row or inflight heartbeat is
promoted into physical batching or runnable-forward evidence.
"""
import json

SCHEMA = "hf2q.f2.phase-trace.v1"
TOTAL_LIMIT = 256 * 1024 * 1024
NATIVE_TOTAL_LIMIT = 2 * 1024 * 1024 * 1024
EVENT_LIMIT = 64 * 1024
SLOT_KEYS = {"request_id", "slot_id", "generation", "phase", "active",
             "runnable_forward", "runnable_emit", "inflight", "blocked"}
ROW_KEYS = {"request_id", "slot_id", "generation", "token_position", "token_count", "phase"}
NATIVE_KINDS = {"native_baseline", "native_arm_ack", "scope_definition", "forward_input",
                "forward_witness", "forward_witness_unmapped", "native_event", "native_final"}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def natural(x):
    return type(x) is int and 0 <= x <= 2**64 - 1


def lineage(slot):
    return slot["slot_id"], slot["generation"]


def metadata_total_limit(header):
    """Only the exact declared diagnostic profile can select a larger file quota."""
    if "metadata_budget_profile" not in header:
        return TOTAL_LIMIT
    require(type(header["metadata_budget_profile"]) is str and
            header["metadata_budget_profile"] == "native_command_capture_v2",
            "unsupported metadata budget profile")
    native = header.get("native_observer")
    require(type(native) is dict and native.get("diagnostic_case") is True,
            "native metadata profile requires explicit diagnostic observer")
    return NATIVE_TOTAL_LIMIT


def validate_trace(records, expected_identity, scenario):
    require(type(records) is list and len(records) >= 2, "header and final bound required")
    require(scenario.get("schema") == "hf2q.f2.trace-expectation.v1", "frozen trace expectation")
    submissions, drained, latest, generations, progress_positions, owners = {}, {}, {}, {}, {}, {}
    terminated = set()
    previous = 0
    size = 0
    header = None
    native = None
    for seq, record in enumerate(records):
        require(set(record) == {"schema", "seq", "at_ns", "kind", "payload"} and
                record["schema"] == SCHEMA and type(record["seq"]) is int and record["seq"] == seq and
                natural(record["at_ns"]) and record["at_ns"] >= previous,
                "trace schema, sequence gap or monotonic clock failure")
        previous = record["at_ns"]
        encoded = len(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode())
        require(encoded <= EVENT_LIMIT, "event quota exceeded")
        size += encoded + 1
        payload, kind = record["payload"], record["kind"]
        if seq == 0:
            base_keys = {"identity", "run_id", "server_budget_bytes", "client_budget_bytes", "clock",
                         "clock_id", "origin_monotonic_ns", "config_sha256"}
            native_keys = base_keys | {"native_observer", "server_jsonl_budget_bytes"}
            require(kind == "header" and set(payload) in (base_keys, native_keys,
                    native_keys | {"metadata_budget_profile"}) and
                    payload["identity"] == expected_identity and payload["run_id"] == scenario["run_id"] and
                    payload["clock"] == "CLOCK_MONOTONIC nanoseconds since origin" and
                    payload["clock_id"] == "CLOCK_MONOTONIC" and
                    natural(payload["origin_monotonic_ns"]),
                    "bound trace header required")
            require(natural(payload["server_budget_bytes"]) and payload["server_budget_bytes"] > 0 and
                    natural(payload["client_budget_bytes"]) and payload["client_budget_bytes"] >= 128 * 1024 * 1024 and
                    payload["server_budget_bytes"] + payload["client_budget_bytes"] <= metadata_total_limit(payload),
                    "combined server/client quota")
            require(payload.get("metadata_budget_profile") == scenario.get("metadata_budget_profile"),
                    "trace metadata budget profile drift")
            require(payload["config_sha256"] == scenario["config_sha256"], "trace configuration drift")
            header = payload
            if "native_observer" in payload:
                from f2_native_trace import NativeTrace
                native = NativeTrace(header, scenario)
            continue
        require(size <= header.get("server_jsonl_budget_bytes", header["server_budget_bytes"]),
                "server trace quota exceeded")
        if kind == "slots":
            require(set(payload) == {"cause", "slots"} and type(payload["slots"]) is list and
                    len(payload["slots"]) <= 256, "complete bounded scheduler snapshot")
            current = {}
            for s in payload["slots"]:
                require(set(s) in (SLOT_KEYS, SLOT_KEYS | {"forward_eligibility"}) and
                        (s.get("forward_eligibility") is None or type(s["forward_eligibility"]) is bool) and
                        all(natural(s[k]) for k in
                        ("request_id", "slot_id", "generation")) and s["active"] is True and
                        s["phase"] in ("prefill", "decode"), "slot lineage/phase")
                key = lineage(s)
                require(key not in current and s["slot_id"] not in {k[0] for k in current},
                        "ambiguous simultaneous slot lineage")
                require(s["generation"] >= generations.get(s["slot_id"], 0), "generation regressed")
                if key in owners:
                    require(s["request_id"] == owners[key], "request changed in recycled generation")
                terminal_reason = s["blocked"] in ("ClientClosed", "RequestTerminal", "Cancelled", "WorkerError")
                require(not terminal_reason or s.get("forward_eligibility") is not True,
                        "terminal/cancelled owner cannot remain eligible")
                require(key not in terminated or s.get("forward_eligibility") is not True,
                        "terminated generation cannot regain forward eligibility")
                if terminal_reason:
                    terminated.add(key)
                owners[key] = s["request_id"]
                for k in ("runnable_forward", "runnable_emit", "inflight"):
                    require(s[k] is None or type(s[k]) is bool, "typed readiness/Unknown required")
                require(s["blocked"] is None or type(s["blocked"]) is str, "typed blocked reason")
                current[key] = s
                generations[s["slot_id"]] = s["generation"]
            latest = current
        elif kind == "submission":
            require(set(payload) == {"submission_id", "membership", "rows"} and
                    natural(payload["submission_id"]) and payload["submission_id"] not in submissions and
                    payload["membership"] == "Known" and type(payload["rows"]) is list and
                    0 < len(payload["rows"]) <= 256, "actual known physical row membership required")
            row_ranges = {}
            for row in payload["rows"]:
                require(set(row) == ROW_KEYS and all(natural(row[k]) for k in ROW_KEYS - {"phase"}) and
                        row["phase"] in ("ordinary", "draft", "verify", "prefill"), "physical row schema")
                key = lineage(row)
                require(key in latest and row["request_id"] == latest[key]["request_id"],
                        "physical row has unknown/stale scheduler owner")
                begin = row["token_position"]
                count = row["token_count"]
                require(0 < count <= 2**32 - 1 and begin + count <= 2**64 - 1,
                        "bounded contiguous physical token-position span")
                identity = (*key, row["phase"])
                intervals = row_ranges.setdefault(identity, [])
                require(all(begin + count <= a or begin >= b for a, b in intervals),
                        "duplicate/overlapping physical rows")
                intervals.append((begin, begin + count))
            submissions[payload["submission_id"]] = record
        elif kind == "drain":
            require(set(payload) == {"submission_id", "success"} and
                    payload["submission_id"] in submissions and payload["submission_id"] not in drained and
                    payload["success"] is True, "missing/duplicate/failed checked drain")
            drained[payload["submission_id"]] = record
        elif kind == "progress":
            require(set(payload) == {"slot_id", "generation", "source", "tokens", "committed_position"} and
                    natural(payload["slot_id"]) and natural(payload["generation"]) and
                    natural(payload["tokens"]) and payload["tokens"] > 0 and
                    natural(payload["committed_position"]) and lineage(payload) in latest and
                    payload["source"] in ("committed_prefill", "generated_publication", "semantic_emit"),
                    "validated progress source required")
            key = (*lineage(payload), payload["source"])
            require(payload["committed_position"] > progress_positions.get(key, -1),
                    "duplicate or regressed committed progress")
            previous_position = progress_positions.get(key)
            require(payload["committed_position"] >= payload["tokens"] and
                    (previous_position is None or payload["committed_position"] - previous_position == payload["tokens"]),
                    "progress token/cursor mismatch")
            progress_positions[key] = payload["committed_position"]
        elif kind == "transaction":
            # This is valuable source evidence, but NOT a native submit timestamp.
            require(set(payload) == {"family", "operation", "logical_lanes", "physical_membership", "reason"}
                    and payload["physical_membership"] == "Unknown" and payload["logical_lanes"],
                    "transaction cannot invent rows")
            for lane in payload["logical_lanes"]:
                require(set(lane) == {"request_id", "slot_id", "generation"} and
                        lineage(lane) in latest and lane["request_id"] == latest[lineage(lane)]["request_id"],
                        "transaction has unknown logical owner")
        elif kind in NATIVE_KINDS:
            require(native is not None, "native record without explicit diagnostic configuration")
            native.consume(record, latest)
        elif kind == "end":
            require(seq == len(records) - 1 and set(payload) == {"error", "dropped_events"} and
                    payload["error"] is None and payload["dropped_events"] == 0,
                    "finalized nonoverflowing trace required")
        else:
            raise ValueError("unknown trace event")
    require(records[-1]["kind"] == "end", "missing final trace: no truncation qualification")
    if native is not None:
        require(not submissions and not drained, "legacy CB-union rows cannot replace native operator witnesses")
        report = native.finish()
        return {"schema": SCHEMA, "status": "Qualified", "qualified": True, "events": len(records),
                "bytes": size, "diagnostic_only": True, "primary_native_trace": "NotCaptured",
                "shared_batch4_claim": False, **report}
    require(submissions and submissions.keys() == drained.keys(), "missing actual submit/drain join")
    return {"schema": SCHEMA, "status": "Qualified", "qualified": True, "events": len(records), "bytes": size,
            "submissions": len(submissions), "physical_membership": "Known",
            "shared_batch4_claim": any(len({lineage(r) for r in s["payload"]["rows"]}) == 4
                                       for s in submissions.values())}


def qualify_window(records, composition, arrival_ns, window_ns=2_000_000_000,
                   deadline_ns=30_000_000_000, population_policy="ready_now_v1"):
    """Called only after validate_trace; no two-second GPU completion deadline."""
    require(composition in ((1, 0), (0, 1), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)) and
            natural(arrival_ns) and window_ns == 2_000_000_000 and deadline_ns == 30_000_000_000,
            "frozen population/window required")
    require(population_policy in ("ready_now_v1", "forward_demand_v2"),
            "externally frozen population policy required")
    native_report = None
    if "native_observer" in records[0]["payload"]:
        # Public callers must first validate against the independently frozen
        # expected identity/config/witness manifest; Q2 does this explicitly.
        from f2_native_trace import NativeTrace
        native_header = records[0]["payload"]
        parser = NativeTrace(native_header, {"forward_witness_manifest":
            native_header["native_observer"]["witness_manifest"]})
        latest = {}
        for record in records[1:]:
            if record["kind"] == "slots":
                latest = {lineage(slot): slot for slot in record["payload"]["slots"]}
            elif record["kind"] in NATIVE_KINDS:
                parser.consume(record, latest)
        native_report = parser.finish()
    require(population_policy != "forward_demand_v2" or native_report is not None,
            "forward demand requires actual native operator witnesses")
    snapshots = [r for r in records if r["kind"] == "slots"]
    def population(record, ready_only=False):
        slots = record["payload"]["slots"]
        if len(slots) != sum(composition):
            return None
        counts = [0,0]
        for slot in slots:
            ready = slot["runnable_forward"]
            if population_policy == "forward_demand_v2" and slot.get("forward_eligibility") is not True:
                # Explicit closed/terminal/cancel/error or Unknown ends credit;
                # ordinary transaction-busy false-readiness does not.
                return None
            outstanding = False
            if population_policy == "forward_demand_v2" and not ready_only:
                outstanding = any(item["end_ns"] <= record["at_ns"] < item["checked_return_ns"] and
                    any(lineage(row) == lineage(slot) and row["request_id"] == slot["request_id"] and
                        ((row["phase"] == "prefill") == (slot["phase"] == "prefill"))
                        for row in item["rows"]) for item in native_report["submissions"])
            if ready is None and not outstanding:
                return None
            if ready is True or outstanding:
                counts[0 if slot["phase"] == "prefill" else 1] += 1
        return tuple(counts)
    if population_policy == "forward_demand_v2":
        # Integration boundaries, not fabricated producer snapshots: readiness
        # remains the last actual owner snapshot; service comes only from the
        # validated exact commit/checked-return interval for that current owner.
        boundaries = {r["at_ns"] for r in snapshots}
        boundaries.update(t for item in native_report["submissions"]
                          for t in (item["end_ns"], item["checked_return_ns"]))
        original = snapshots
        snapshots = []
        cursor = 0
        current = None
        for at in sorted(boundaries):
            while cursor < len(original) and original[cursor]["at_ns"] <= at:
                current = original[cursor]
                cursor += 1
            if current is not None:
                snapshots.append({"at_ns":at,"payload":current["payload"]})
    candidates = [r for r in snapshots if arrival_ns <= r["at_ns"] and
                  r["at_ns"] + window_ns <= arrival_ns + deadline_ns and population(r) == composition]
    if not candidates:
        return {"status": "Unqualified", "reason": "no source-qualified population within deadline"}
    first = candidates[0]
    start, end = first["at_ns"], first["at_ns"] + window_ns
    require(records[-1]["at_ns"] >= end, "trace ends before entire frozen window")
    coverage = 0
    ready_coverage = 0
    intervals = [r for r in snapshots if start <= r["at_ns"] < end]
    for a, b in zip(intervals, intervals[1:] + [{"at_ns": end}]):
        if population(a) == composition:
            coverage += b["at_ns"] - a["at_ns"]
        if population(a, ready_only=True) == composition:
            ready_coverage += b["at_ns"] - a["at_ns"]
    admitted = {lineage(s) for s in first["payload"]["slots"]}
    progress = set()
    drained = {r["payload"]["submission_id"] for r in records
               if r["kind"] == "drain" and r["payload"]["success"] is True}
    for r in records:
        if not start <= r["at_ns"] <= end:
            continue
        if r["kind"] == "progress":
            progress.add(lineage(r["payload"]))
        elif r["kind"] == "submission" and r["payload"]["submission_id"] in drained:
            progress.update(lineage(row) for row in r["payload"]["rows"])
    witness_progress = set()
    if native_report is not None:
        for item in native_report["submissions"]:
            if start <= item["begin_ns"] <= item["end_ns"] <= end:
                witness_progress.update(lineage(row) for row in item["rows"])
        progress.update(witness_progress)
    qualified = coverage * 5 >= window_ns * 4 and admitted <= progress and (
        native_report is None or admitted <= witness_progress)
    return {"status": "Qualified" if qualified else "Unqualified", "start_ns": start,
            "end_ns": end, "target_coverage_ns": coverage, "target_fraction": coverage / window_ns,
            "population_policy":population_policy,"ready_now_coverage_ns":ready_coverage,
            "ready_now_fraction":ready_coverage / window_ns,
            "service_semantics":"host-known SubmittedOutstanding, never Running/utilization"
                if population_policy == "forward_demand_v2" else "ready-now only",
            "slots_without_progress": [list(k) for k in sorted(admitted - progress)],
            "slots_without_required_forward_witness": [list(k) for k in sorted(
                admitted - witness_progress)] if native_report is not None else None,
            "physical_batching_scope": "operator_only_full_body_unknown" if native_report is not None else "legacy",
            "window_selection": "first qualifying target population; fixed, no favorable search"}
