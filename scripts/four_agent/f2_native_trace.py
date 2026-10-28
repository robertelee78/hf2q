"""Actual native command joins and required arithmetic-witness coverage.

Opaque auxiliary work is retained in lifecycle/counts, never promoted into
logical rows. This validates a diagnostic, not an observer-OFF primary cohort.
"""

ROW = {"request_id", "slot_id", "generation", "token_position", "token_count", "phase"}
EVENT = {"clock_id", "command_id", "invocation_epoch", "queue_identity", "device_registry_id",
         "kind", "begin_monotonic_ns", "end_monotonic_ns", "dispatch_ordinal", "noncompute_operations",
         "scope", "captured_scope", "capture_command_id", "capture_generation", "replay_binding",
         "tainted", "geometry", "submitted", "checked_status"}
STATUS = {"installed", "invalid", "events", "unknown_scopes", "event_limit", "metadata_reserved_bytes",
          "metadata_limit_bytes", "dropped_events", "unobserved_commits", "capture_valid",
          "comparison_qualified", "pending_commands", "pending_command_limit"}
SOURCES = {"qwen35.greedy.actual_tokens", "qwen35.forward.actual_tokens", "gemma4.scalar_forward",
           "gemma4.decode.actual_token_slot_position_arrays", "gemma4.prefill.actual_concat_descriptor",
           "gemma4.prefill.actual_single_input", "deepseek4.scalar_verifier",
           "deepseek4.prefill.actual_cache_span", "deepseek4.sorted_cache_lane",
           "deepseek4.prefill.actual_cooperative_lane", "deepseek4.prefill.actual_cooperative_pack"}


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def number(value):
    return type(value) is int and 0 <= value <= 2**64 - 1


def key(scope):
    need(type(scope) is dict and set(scope) == {"epoch", "id"} and
         all(number(value) and value > 0 for value in scope.values()), "typed current native scope")
    return scope["epoch"], scope["id"]


class NativeTrace:
    def __init__(self, header, scenario):
        self.header = header
        config = header["native_observer"]
        need(type(config) is dict and set(config) == {"event_limit", "metadata_limit_bytes",
             "diagnostic_case", "control_nonce", "witness_manifest"} and config["diagnostic_case"] is True,
             "closed separate native diagnostic configuration")
        need(number(config["event_limit"]) and 0 < config["event_limit"] <= 1_000_000 and
             number(config["metadata_limit_bytes"]) and 0 < config["metadata_limit_bytes"] <= 16 * 1024**2 and
             type(config["control_nonce"]) is str and 32 <= len(config["control_nonce"]) <= 128 and
             all(c.isascii() and (c.isalnum() or c in "_-") for c in config["control_nonce"]) and
             header["server_jsonl_budget_bytes"] == header["server_budget_bytes"] -
                 config["metadata_limit_bytes"] - 1024**2 and header["server_jsonl_budget_bytes"] >= 64 * 1024,
             "prospective native/server quota carve-out or nonce")
        self.policy = config["witness_manifest"]
        need(self.policy == scenario.get("forward_witness_manifest") and set(self.policy) == {
             "schema", "operator", "input_sources", "coverage", "batching_scope"} and
             self.policy["schema"] == "hf2q.f2.forward-witness-policy.v1" and
             self.policy["operator"] == "target_input_rms_norm" and
             self.policy["coverage"] == "each_admitted_lane_in_first_fixed_window" and
             self.policy["batching_scope"] == "operator_only_full_body_unknown" and
             type(self.policy["input_sources"]) is list and 0 < len(self.policy["input_sources"]) <= 16 and
             all(type(source) is str and source in SOURCES for source in self.policy["input_sources"]) and
             len(set(self.policy["input_sources"])) == len(self.policy["input_sources"]),
             "external predeclared required-witness policy")
        self.origin = header["origin_monotonic_ns"]
        self.scopes, self.inputs, self.witnesses, self.commands = {}, {}, {}, {}
        self.baseline = self.final = self.ack = None
        self.events = self.auxiliary = self.unmapped = 0

    def rows(self, rows, latest):
        need(type(rows) is list and 0 < len(rows) <= 256, "bounded actual logical rows")
        ranges = {}
        for row in rows:
            need(type(row) is dict and set(row) == ROW and
                 all(number(row[name]) for name in ROW - {"phase"}) and
                 row["phase"] in ("ordinary", "verify", "draft", "prefill"), "physical row schema")
            owner = latest.get((row["slot_id"], row["generation"]))
            need(owner and owner["request_id"] == row["request_id"], "witness stale/missing owner")
            begin, count = row["token_position"], row["token_count"]
            need(0 < count <= 2**32 - 1 and begin + count <= 2**64 - 1, "row extent overflow")
            intervals = ranges.setdefault((row["slot_id"], row["generation"]), [])
            need(all(begin + count <= a or begin >= b for a, b in intervals), "overlapping witness rows")
            intervals.append((begin, begin + count))

    def status(self, value):
        need(type(value) is dict and set(value) == STATUS and value["installed"] is True and
             value["invalid"] is False and value["capture_valid"] is True and
             value["comparison_qualified"] is False and value["dropped_events"] == 0 and
             value["unknown_scopes"] == 0, "invalid/dropped/unknown native observation")
        for name in STATUS - {"installed", "invalid", "capture_valid", "comparison_qualified"}:
            need(number(value[name]), "native counter type")
        need(value["pending_commands"] == 0 and value["pending_command_limit"] == 1024,
             "native pending command ownership not retired or limit changed")
        config = self.header["native_observer"]
        need(value["event_limit"] == config["event_limit"] and
             value["metadata_limit_bytes"] == config["metadata_limit_bytes"] and
             value["events"] <= value["event_limit"] and
             value["metadata_reserved_bytes"] <= value["metadata_limit_bytes"], "native quota identity")

    def consume(self, record, latest):
        kind, p = record["kind"], record["payload"]
        need(self.final is None, "native data after final status")
        if kind == "native_baseline":
            need(self.baseline is None and not self.commands, "one pre-arrival native baseline")
            self.status(p)
            self.baseline = p
        elif kind == "native_arm_ack":
            need(self.baseline is not None and self.ack is None and set(p) == {
                 "nonce", "clock_id", "worker_queue_checked", "quiescent", "baseline", "scope"} and
                 p["nonce"] == self.header["native_observer"]["control_nonce"] and
                 p["clock_id"] == "CLOCK_MONOTONIC" and p["worker_queue_checked"] is True and
                 p["quiescent"] is True and p["baseline"] == self.baseline and
                 p["scope"] == "one_fresh_phase_diagnostic", "actual owning-worker arm acknowledgement")
            self.status(p["baseline"])
            self.ack = record
        elif kind == "scope_definition":
            need(self.ack is not None and set(p) == {"scope", "operation", "membership", "rows"},
                 "scope before arm or malformed definition")
            identity = key(p["scope"])
            need(identity not in self.scopes, "scope identity redefined")
            if p["membership"] == "Known":
                need(p["operation"] == "target.input_rms_norm", "undeclared known operator")
                self.rows(p["rows"], latest)
            else:
                need(p["membership"] == "OpaqueAuxiliary" and p["rows"] == [] and
                     p["operation"] == "opaque_auxiliary", "unmapped work cannot claim rows")
            self.scopes[identity] = record
        elif kind == "forward_input":
            need(self.ack is not None and set(p) == {"forward_id", "epoch", "source", "role",
                 "matrix_rows", "packed_rows", "operator_site", "execution_proven"} and
                 p["role"] in ("target", "target_verify") and
                 p["execution_proven"] is False and p["source"] in self.policy["input_sources"] and
                 number(p["forward_id"]) and p["forward_id"] > 0 and
                 number(p["epoch"]) and p["epoch"] > 0 and p["forward_id"] not in self.inputs,
                 "actual source-bound forward input required")
            need(p["operator_site"] == ("ffn_hc_input" if p["source"] ==
                 "deepseek4.prefill.actual_cooperative_pack" else "pre_attention_input"),
                 "actual target operator site")
            need(type(p["packed_rows"]) is list and 0 < len(p["packed_rows"]) <= 256,
                 "bounded packed input")
            offset, rows = 0, []
            for span in p["packed_rows"]:
                need(set(span) == {"matrix_row_start", "matrix_row_end", "logical"} and
                     number(span["matrix_row_start"]) and number(span["matrix_row_end"]) and
                     span["matrix_row_start"] == offset, "packed tensor row offset mismatch")
                row = span["logical"]
                self.rows([row], latest)
                offset += row["token_count"]
                need(number(offset) and span["matrix_row_end"] == offset, "packed tensor extent mismatch")
                rows.append(row)
            self.rows(rows, latest)
            if p["role"] == "target_verify":
                need(p["source"] == "qwen35.forward.actual_tokens" and
                     all(row["phase"] == "verify" for row in rows), "actual verifier source/role required")
            else:
                need(all(row["phase"] in ("ordinary", "prefill") for row in rows),
                     "ordinary/prefill source cannot invent a verifier role")
            need(number(p["matrix_rows"]) and p["matrix_rows"] == offset,
                 "input rows disagree with actual concatenation")
            source = p["source"]
            scalar = {"qwen35.greedy.actual_tokens", "gemma4.scalar_forward",
                      "deepseek4.scalar_verifier", "deepseek4.sorted_cache_lane"}
            if source in scalar:
                need(len(rows) == 1 and offset == 1, "scalar source cannot claim a physical quad")
            elif source in {"qwen35.forward.actual_tokens", "gemma4.prefill.actual_single_input",
                            "deepseek4.prefill.actual_cache_span", "deepseek4.prefill.actual_cooperative_lane"}:
                need(len(rows) == 1, "single-sequence source cannot claim independent lanes")
            else:
                if source == "deepseek4.prefill.actual_cooperative_pack":
                    need(2 <= len(rows) <= 4, "actual cooperative pack cardinality")
                need(len({(row["slot_id"], row["generation"]) for row in rows}) == len(rows),
                     "packed independent input repeats one logical lane")
                if source == "gemma4.decode.actual_token_slot_position_arrays":
                    need(all(row["token_count"] == 1 for row in rows), "decode array source has temporal rows")
            self.inputs[p["forward_id"]] = record
        elif kind == "forward_witness":
            self.witness(record)
        elif kind == "forward_witness_unmapped":
            need(set(p) == {"operator", "reason", "progress_eligible"} and
                 p["operator"] == "target_input_rms_norm" and p["reason"] == "NoActualInputManifest" and
                 p["progress_eligible"] is False, "unmapped operator cannot be progress")
            self.unmapped += 1
        elif kind == "native_event":
            self.event(record)
        elif kind == "native_final":
            self.status(p)
            need(self.baseline is not None and self.ack is not None and
                 p["events"] - self.baseline["events"] == self.events and
                 p["unobserved_commits"] == self.baseline["unobserved_commits"] and
                 p["metadata_reserved_bytes"] == 0, "native count/retirement/completeness delta")
            self.final = p
        else:
            raise ValueError("unknown native record kind")

    def witness(self, record):
        p = record["payload"]
        need(set(p) == {"forward_id", "scope", "source", "operator", "layer", "role", "rows", "columns",
             "dtype", "input_parent_identity", "input_byte_offset", "input_read_bytes", "input_data_bytes",
             "input_parent_bytes", "operator_site", "dispatches_expected", "execution_proven",
             "full_body_batching_proven"},
             "closed arithmetic-witness declaration")
        identity = key(p["scope"])
        need(number(p["forward_id"]), "typed forward identity")
        scope = self.scopes.get(identity)
        source = self.inputs.get(p["forward_id"])
        need(scope and source and number(p["forward_id"]) and identity not in self.witnesses and
             scope["payload"]["membership"] == "Known" and
             scope["payload"]["rows"] == [r["logical"] for r in source["payload"]["packed_rows"]] and
             identity[0] == source["payload"]["epoch"] and p["source"] == source["payload"]["source"] and
             p["rows"] == source["payload"]["matrix_rows"] and p["role"] == source["payload"]["role"] and
             p["operator_site"] == source["payload"]["operator_site"] and
             p["operator"] == self.policy["operator"] and type(p["dispatches_expected"]) is int and
             p["dispatches_expected"] == 1 and (p["layer"] is None or number(p["layer"])) and
             p["execution_proven"] is False and p["full_body_batching_proven"] is False,
             "required witness replaced/stale/missing input")
        for name in ("rows", "columns", "input_parent_identity", "input_byte_offset", "input_read_bytes",
                     "input_data_bytes", "input_parent_bytes"):
            need(number(p[name]), "typed matrix/view extent")
        width = {"F32": 4, "F16": 2, "BF16": 2}.get(p["dtype"])
        expected = p["rows"] * p["columns"] * (width or 0)
        need(width and p["rows"] > 0 and p["columns"] > 0 and p["input_parent_identity"] > 0 and
             number(expected) and expected == p["input_read_bytes"] <= p["input_data_bytes"] and
             p["input_byte_offset"] + expected <= p["input_parent_bytes"], "matrix footprint mismatch")
        self.witnesses[identity] = {"declaration": record, "encodes": []}

    def event(self, record):
        p = record["payload"]
        need(self.ack is not None and type(p) is dict and set(p) == EVENT and
             p["clock_id"] == "CLOCK_MONOTONIC" and p["tainted"] is False and
             type(p["submitted"]) is bool and
             all(number(p[name]) for name in ("command_id", "invocation_epoch", "queue_identity",
                 "device_registry_id", "begin_monotonic_ns", "end_monotonic_ns", "dispatch_ordinal",
             "noncompute_operations")) and p["command_id"] > 0 and p["invocation_epoch"] > 0 and
             p["queue_identity"] > 0 and p["device_registry_id"] > 0 and
             self.origin <= p["begin_monotonic_ns"] <= p["end_monotonic_ns"] <= self.origin + record["at_ns"],
             "native event source/clock/taint")
        self.events += 1
        identity, kind = p["command_id"], p["kind"]
        if kind != "DispatchEncoded":
            need(p["scope"] is None and p["geometry"] is None and p["captured_scope"] is None and
                 p["capture_command_id"] is None and p["capture_generation"] is None,
                 "non-dispatch event cannot claim physical rows")
        if kind == "Created":
            need(identity not in self.commands and p["dispatch_ordinal"] == 0 and not p["submitted"] and
                 p["noncompute_operations"] == 0 and p["checked_status"] is None and p["scope"] is None,
                 "duplicate/late command creation")
            self.commands[identity] = {"created": p, "dispatches": [], "submit": None, "drain": None,
                                       "dropped": False, "noncompute": 0, "end": p["end_monotonic_ns"]}
            return
        c = self.commands.get(identity)
        need(c and not c["dropped"] and all(p[name] == c["created"][name] for name in
             ("invocation_epoch", "queue_identity", "device_registry_id")) and
             p["begin_monotonic_ns"] >= c["end"], "command identity/order drift")
        c["end"] = p["end_monotonic_ns"]
        if kind == "DispatchEncoded":
            need(c["submit"] is None and p["dispatch_ordinal"] == len(c["dispatches"]) + 1 and
                 p["geometry"] is not None and p["replay_binding"] in
                 ("Direct", "ImmediateCapture", "ExplicitRemap"), "actual dispatch ordinal/geometry/replay")
            geometry = p["geometry"]
            need(type(geometry) is dict and set(geometry) == {"pipeline_identity", "dispatch_kind", "grid",
                 "threads_per_threadgroup"} and number(geometry["pipeline_identity"]) and
                 geometry["pipeline_identity"] > 0 and geometry["dispatch_kind"] in ("threads", "thread_groups") and
                 all(type(geometry[name]) is list and len(geometry[name]) == 3 and
                     all(number(value) for value in geometry[name]) for name in
                     ("grid", "threads_per_threadgroup")), "actual fixed dispatch geometry")
            if p["replay_binding"] == "Direct":
                need(p["captured_scope"] is None and p["capture_command_id"] is None and
                     p["capture_generation"] is None, "direct dispatch has replay metadata")
            else:
                need(number(p["capture_command_id"]) and p["capture_command_id"] > 0 and
                     number(p["capture_generation"]) and p["capture_generation"] > 0,
                     "actual capture lineage required")
                old = key(p["captured_scope"])
                need(old in self.scopes, "replay from an unbound capture scope")
                if p["replay_binding"] == "ImmediateCapture":
                    need(old == key(p["scope"]) and p["capture_command_id"] == identity,
                         "automatic replay must be original command/current scope")
            scope = key(p["scope"])
            need(scope in self.scopes and scope[0] == p["invocation_epoch"], "dispatch missing current scope")
            if self.scopes[scope]["payload"]["membership"] == "Known":
                need(scope in self.witnesses, "known scope without declared arithmetic witness")
                need(p["begin_monotonic_ns"] >= self.origin +
                     self.witnesses[scope]["declaration"]["at_ns"],
                     "witness declared after actual encode")
                need(p["replay_binding"] != "ExplicitRemap",
                     "current-input replay-map producer proof not implemented for required witness")
                need(all(value > 0 for name in ("grid", "threads_per_threadgroup") for value in geometry[name]),
                     "required witness has empty dispatch geometry")
                self.witnesses[scope]["encodes"].append((identity, record))
            else:
                self.auxiliary += 1
            c["dispatches"].append(record)
        elif kind == "NonComputeEncoded":
            need(c["submit"] is None and p["checked_status"] is None and
                 p["noncompute_operations"] == c["noncompute"] + 1,
                 "noncompute ordering/count")
            c["noncompute"] += 1
        elif kind == "Submitted":
            need(c["submit"] is None and p["submitted"] is True and
                 p["dispatch_ordinal"] == len(c["dispatches"]) and
                 p["noncompute_operations"] == c["noncompute"] and p["checked_status"] is None,
                 "one actual commit bracket")
            c["submit"] = record
        elif kind == "CheckedDrain":
            need(c["submit"] and c["drain"] is None and p["checked_status"] == 4,
                 "actual successful checked drain")
            c["drain"] = record
        elif kind == "Dropped":
            need(not c["submit"] or not (c["dispatches"] or c["noncompute"]) or c["drain"],
                 "observed submitted work dropped without checked completion")
            need(p["submitted"] == bool(c["submit"]) and
                 p["checked_status"] == (4 if c["drain"] else None), "drop status mismatch")
            c["dropped"] = True
        else:
            raise ValueError("unknown native command event")
        if kind != "DispatchEncoded":
            need(p["scope"] is None and p["geometry"] is None and p["captured_scope"] is None and
                 p["capture_command_id"] is None and p["capture_generation"] is None,
                 "non-dispatch event cannot claim physical rows")

    def finish(self):
        need(self.final is not None and all(c["dropped"] for c in self.commands.values()),
             "final native owner status missing")
        submissions = []
        forwards = [item["declaration"]["payload"]["forward_id"] for item in self.witnesses.values()]
        need(len(forwards) == len(set(forwards)) and set(forwards) == self.inputs.keys(),
             "required forward input replaced by auxiliary/missing/duplicate witness")
        for identity, witness in self.witnesses.items():
            need(len(witness["encodes"]) == 1, "required one-dispatch witness missing/duplicated")
            command, encoded = witness["encodes"][0]
            c = self.commands[command]
            need(c["submit"] and c["drain"], "arithmetic witness was never submitted/completed")
            p = c["submit"]["payload"]
            submissions.append({"command_id": command, "dispatch_ordinal": encoded["payload"]["dispatch_ordinal"],
                "begin_ns": p["begin_monotonic_ns"] - self.origin,
                "end_ns": p["end_monotonic_ns"] - self.origin,
                "checked_return_ns": c["drain"]["payload"]["end_monotonic_ns"] - self.origin,
                "rows": self.scopes[identity]["payload"]["rows"], "operator": self.policy["operator"]})
        need(submissions, "auxiliary-only trace cannot qualify required forward witnesses")
        return {"submissions": submissions, "native_events": self.events,
                "auxiliary_dispatches": self.auxiliary, "unmapped_operator_attempts": self.unmapped,
                "input_sources_seen": sorted({r["payload"]["source"] for r in self.inputs.values()}),
                "physical_membership": "RequiredOperatorWitnessesOnly", "full_body_batching": "Unknown",
                "operator_shared_batch4_observed": any(len({(r["slot_id"], r["generation"])
                    for r in item["rows"]}) == 4 for item in submissions)}
