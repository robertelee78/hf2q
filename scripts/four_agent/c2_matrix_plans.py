"""Independent C2 pre-call geometry and required-ticket reconciliation."""
from c2_matrix_io import bound, coordinate, extents, input_binding, input_digest, integer, require

KINDS = ["Case", "Request", "Step", "NativeCall", "Projection", "Selection", "State", "Checkpoint",
         "SavedCheckpointConsumption", "Settlement"]
CHILDREN = {"Case": {"Request"}, "Request": {"Step", "Settlement"},
            "Step": {"NativeCall", "Selection", "State", "Checkpoint", "SavedCheckpointConsumption"},
            "NativeCall": {"Projection", "Selection"}}
MINIMUM = {"Case": {"Request"}, "Request": {"Settlement"}, "Step": {"State"}, "NativeCall": {"Projection"}}
WINDOWS = {"cancel-beforecapture": "BeforeCapture", "cancel-pending": "Pending",
           "cancel-resumed-decode": "ResumedDecode", "cancel-fresh-reset": "FreshReset"}


def expected(stage, agent, step, row, scalar, shape, segment=0):
    return {"at": {"logical_step": step, "layer": 0, "stage": stage, "agent": agent,
                   "logical_row_start": row, "segment": segment}, "scalar": scalar, "shape": shape}


def projection(request, agent):
    require(set(request) == {"sequence", "slot", "start", "input_rows", "projected_rows", "vocab", "hidden_width", "origin"},
            "closed native request geometry required")
    for key in ("sequence", "start"):
        integer(request[key], key)
    require(request["slot"] == 0, "projection belongs to another physical slot")
    integer(request["slot"], "projection physical slot", 0, 0)
    rows = integer(request["projected_rows"], "projected rows", 1, 32768)
    inputs = integer(request["input_rows"], "native input rows", rows)
    vocab = integer(request["vocab"], "requested vocabulary", 1)
    hidden = integer(request["hidden_width"], "requested hidden width", 1)
    first = integer(request["start"] + inputs - rows, "requested first physical row")
    integer(first + rows - 1, "requested last physical row")
    if request["origin"] != "Native":
        origin = request["origin"]
        require(type(origin) is dict and set(origin) == {"SavedCheckpoint"}, "unknown projection origin")
        saved = origin["SavedCheckpoint"]
        require(set(saved) == {"witness", "hidden"} and integer(saved["witness"], "checkpoint witness", 1)
                and saved["hidden"] in ("ConsumedSpeculative", "OriginalNormalizedHead") and rows == inputs == 1,
                "saved checkpoint invented rows or hidden mode")
    return [expected(stage, agent, request["sequence"], first + row, "F32", [1, width])
            for stage, width in [("Logits", vocab), ("LayerOutput", hidden)] for row in range(rows)]


def decision(request, agent, origin):
    require(set(request) == {"semantic_index", "physical_position", "vocab", "hidden_width"}, "closed semantic request required")
    index = integer(request["semantic_index"], "semantic index")
    position = integer(request["physical_position"], "semantic physical position")
    require(position == origin + index, "semantic selector moved from its source-requested physical row")
    vocab = integer(request["vocab"], "semantic vocabulary", 1)
    hidden = integer(request["hidden_width"], "semantic hidden width", 1)
    raw = expected("SpecCanonicalIds", agent, index, index, "U32", [1])
    canonical_agent = agent.removesuffix("/slot0") + "-single-slot-parity/slot0"
    canonical = [expected(stage, canonical_agent, index, position, scalar, shape) for stage, scalar, shape in
                 [("Logits", "F32", [1, vocab]), ("LayerOutput", "F32", [1, hidden]), ("SpecCanonicalIds", "U32", [1])]]
    return raw, canonical


def state(layout, agent):
    require(type(layout) is dict and set(layout) == {"cursors", "parity", "regions", "ledger"}, "closed allocator state layout required")
    cursors = integer(layout["cursors"], "target/MTP cursor count", 1)
    parity = integer(layout["parity"], "DeltaNet parity count", 1)
    ledger = integer(layout["ledger"], "retained ledger length")
    require(type(layout["regions"]) is list, "explicit empty/nonempty state regions required")
    result = [expected("CacheState", agent, 0, 0, "U32", [cursors], 0),
              expected("CacheState", agent, 0, 0, "U8", [parity], 1)]
    for index, length in enumerate(layout["regions"]):
        integer(length, "semantic region bytes")
        if length:
            result.append(expected("CacheState", agent, 0, 0, "U8", [length], index + 2))
    if ledger:
        result.append(expected("CacheState", agent, 0, 0, "U32", [ledger], (1 << 32) - 1))
    return result


def purpose(case, arm):
    if case not in WINDOWS:
        require(not case.startswith("cancel-"), "unknown cancellation case")
        return "Generation"
    require(arm in ("actual", "twin"), "cancellation has no such arm")
    return {"CancelledRequest" if arm == "actual" else "RecoveryReference": WINDOWS[case]}


def terminal(value, wanted):
    require(type(value) is dict and len(value) == 1, "typed actual request outcome required")
    if wanted == "Generation":
        require(set(value) == {"Completed"} and set(value["Completed"]) == {"emitted_tokens"}, "generation terminal differs")
        integer(value["Completed"]["emitted_tokens"], "actual settled token count")
        return
    if "RecoveryReference" in wanted:
        require(set(value) == {"Reference"} and set(value["Reference"]) == {"window", "recovered_tokens"}, "reference terminal differs")
        require(value["Reference"]["window"] == wanted["RecoveryReference"], "reference window differs")
        integer(value["Reference"]["recovered_tokens"], "reference depth")
        return
    require(set(value) == {"Cancelled"}, "cancellation relabelled as generation")
    row = value["Cancelled"]
    require(set(row) == {"window", "request", "recovery", "recovered_tokens"}
            and row["window"] == wanted["CancelledRequest"], "cancellation request/window differs")
    integer(row["request"], "actual cancelled admission")
    depth = integer(row["recovered_tokens"], "recovered depth")
    recovery = {"BeforeCapture": "Preserved", "Pending": {"Restored": {"rank": 0}},
                "ResumedDecode": {"Restored": {"rank": 0}}, "FreshReset": "Reset"}[row["window"]]
    require(row["recovery"] == recovery and (row["window"] != "FreshReset" or depth == 0), "wrong actual cancellation recovery branch")


def ledger(entries):
    require(type(entries) is list and 0 < len(entries) <= 65536, "finite required execution tree required")
    child_masks = [0] * len(entries)
    for index, entry in enumerate(entries):
        require(type(entry) is dict and set(entry) == {"parent", "kind", "phase", "remaining", "children", "required_children"},
                "closed required obligation record required")
        integer(entry["remaining"], "unfinished required work", 0, 0)
        require(entry["kind"] != "Checkpoint", "checkpoint obligation lacks a supported source proof")
        require(entry["kind"] in KINDS and entry["phase"] == "Complete" and entry["remaining"] == 0,
                "required execution work is unfinished")
        parent = entry["parent"]
        if parent is None:
            require(index == 0 and entry["kind"] == "Case", "execution has another root")
        else:
            integer(parent, "parent ticket", 0, index - 1)
            require(entry["kind"] in CHILDREN.get(entries[parent]["kind"], set()), "incompatible required parent/child")
            child_masks[parent] |= 1 << KINDS.index(entry["kind"])
    for index, entry in enumerate(entries):
        minimum = sum(1 << KINDS.index(kind) for kind in MINIMUM.get(entry["kind"], set()))
        declared = integer(entry["required_children"], "required child-kind mask", 0, (1 << len(KINDS)) - 1)
        require(entry["children"] == child_masks[index] and declared & minimum == minimum
                and child_masks[index] & declared == declared, "required child kind was omitted or weakened")
    return entries


def execution(scope, case, arm, binding, raw_required):
    document, _ = bound(scope.root, binding)
    require(document["schema"] == "hf2q.c2.execution-index.v1" and document["all_required_execution_obligations_closed"] is True
            and document["unforced"] is True and document["whole_matrix_or_release_qualified"] is False, "execution unfinished or overclaims acceptance")
    scope.header(document["context"], case, arm)
    require(type(document["obligations"]) is list and len(document["obligations"]) <= scope.registration["grants"]["events"],
            "execution exceeds its preregistered event bound")
    entries = ledger(document["obligations"])
    wanted = purpose(case, arm)
    parents, calls, required_projections, required_gpu = set(), set(), {}, {}
    request = origin = outcome = None

    def ticket(index, kind, parent=None):
        integer(index, "required ticket", 0, len(entries) - 1)
        require(entries[index]["kind"] == kind and (parent is None or entries[index]["parent"] == parent), "ticket kind/parent differs")

    for reference in document["parents"]:
        value, _ = bound(scope.root, reference)
        scope.header(value["context"], case, arm, frozen=document["context"]["acquisition"])
        schema = value["schema"]
        if schema == "hf2q.c2.required-request.v1":
            require(request is None and value["purpose"] == wanted, "required request duplicated or relabelled")
            request, origin = value["request"], integer(value["prediction_origin"], "source prediction origin")
            ticket(request, "Request", 0); ticket(value["required_settlement"], "Settlement", request)
            parents.add(request)
        elif schema == "hf2q.c2.required-step.v1":
            require(request is not None, "work preceded required request")
            index = value["ticket"]; ticket(index, "Step", request); ticket(value["required_state"], "State", index)
            require(index not in parents, "required work repeated"); parents.add(index)
            kinds = {entry["kind"] for entry in entries if entry["parent"] == index}
            require(value["work"] in ("Native", "StateOnly", "SavedCheckpoint"), "unknown selected work")
            if value["work"] == "Native": require("NativeCall" in kinds, "whole selected native API omitted")
            if value["work"] == "SavedCheckpoint": require("SavedCheckpointConsumption" in kinds and "NativeCall" not in kinds,
                                                         "equality omitted saved consumption or invented a native forward")
        elif schema == "hf2q.c2.required-native-call.v1":
            integer(value["call"], "required native call identity")
            require(value["call"] not in calls, "native API contract repeated"); calls.add(value["call"])
            parent = value["parent"]
            if parent is None:
                require(value["other_slots"] and not value["required_projections"] and not value["required_gpu_selectors"], "native input contract omitted")
            else:
                ticket(parent, "NativeCall"); require(parent not in parents and value["required_projections"], "native head contract absent/repeated")
                parents.add(parent)
                for index, request_geometry, source in value["required_projections"]:
                    ticket(index, "Projection", parent)
                    require(index not in required_projections and request_geometry["origin"] == "Native", "required head repeated/relabelled")
                    projection(request_geometry, f"{case}/slot0")
                    required_projections[index] = (request_geometry, input_binding(source, request_geometry["input_rows"]))
                for index, position in value["required_gpu_selectors"]:
                    ticket(index, "Selection", parent)
                    require(index not in required_gpu, "required GPU selector repeated")
                    required_gpu[index] = integer(position, "required GPU position")
        elif schema == "hf2q.c2.required-settlement.v1":
            require(request is not None, "terminal preceded its owning required request")
            index = value["ticket"]; ticket(index, "Settlement", request)
            require(outcome is None and index not in parents, "terminal outcome repeated")
            terminal(value["outcome"], wanted); outcome = value["outcome"]; parents.add(index)
        else:
            raise ValueError("unknown required parent schema")
    require(request is not None and outcome is not None and any(entry["kind"] == "Step" for entry in entries), "whole request/work/terminal proof omitted")
    require(all(index in parents for index, entry in enumerate(entries)
                if entry["kind"] in ("Request", "Step", "NativeCall", "Settlement")), "whole required parent proof omitted")
    planned, forwards, decisions, gpu, raw, canonical, witnesses, inputs = set(), {}, {}, [], [], [], [], {}
    agent = f"{case}/slot0"
    for reference in document["plans"]:
        index = reference["ticket"]
        require(index not in planned, "lower required plan repeated"); planned.add(index)
        value, _ = bound(scope.root, reference["file"])
        scope.header(value["context"], case, arm, frozen=document["context"]["acquisition"])
        require(value["ticket"] == index, "lower plan changed its required ticket")
        if reference["kind"] == "Forward":
            require(value["schema"] == "hf2q.c2.forward-plan.v1", "forward plan schema differs")
            geometry = value["request"]
            if geometry["origin"] == "Native":
                ticket(index, "Projection")
                require(required_projections.pop(index, None) == (geometry, value["input"])
                        and value["original_witness"] is None, "projection changed source API input/geometry")
            else:
                ticket(index, "SavedCheckpointConsumption")
                require(type(value["input"]) is dict and set(value["input"]) == {"Host"}, "saved logical input is not pre-bound")
                require(value["original_witness"] is not None, "saved consumption lost its original receipt")
                witnesses.append((geometry, value["original_witness"]))
            registered = projection(geometry, agent)
            require(value["expected"] == registered and geometry["sequence"] not in forwards, "full forward registration differs/repeats")
            forwards[geometry["sequence"]] = geometry; raw.extend(registered)
            inputs[geometry["sequence"]] = input_binding(value["input"], geometry["input_rows"])
        elif reference["kind"] == "Decision":
            require(value["schema"] == "hf2q.c2.semantic-selector-plan.v1", "semantic plan schema differs")
            ticket(index, "Selection")
            require(entries[entries[index]["parent"]]["kind"] == "Step", "semantic selector has no required work parent")
            registered, canonical_rows = decision(value["request"], agent, origin)
            require(value["raw_expected"] == registered and value["canonical_expected"] == canonical_rows, "semantic registration differs")
            semantic = value["request"]["semantic_index"]
            require(semantic not in decisions, "semantic decision repeated")
            decisions[semantic] = value["request"]; raw.append(registered); canonical.extend(canonical_rows)
        elif reference["kind"] == "GpuArgmax":
            require(value["schema"] == "hf2q.c2.gpu-selector-plan.v1" and required_gpu.pop(index, None) == value["position"], "GPU selection changed source obligation")
            gpu.append(value["position"])
        else:
            raise ValueError("unknown lower evidence plan")
    require(not required_projections and not required_gpu, "whole planner/observer pair omitted")
    require(all(index in planned for index, entry in enumerate(entries)
                if entry["kind"] in ("Projection", "Selection", "SavedCheckpointConsumption")), "required lower plan omitted")
    require(sorted(forwards) == list(range(len(forwards))) and sorted(decisions) == list(range(len(decisions))), "source rows/selectors omit intermediate identities")
    require(bool(forwards) == bool(decisions) == raw_required, "raw target work differs from source request purpose")
    if "Completed" in outcome:
        count = outcome["Completed"]["emitted_tokens"]
        require(count <= len(decisions) <= count + 1, "settled output omitted semantic/terminal work")
    return {"binding": binding, "entries": entries, "raw": raw, "canonical": canonical, "forwards": forwards,
            "decisions": decisions, "gpu_positions": gpu, "witnesses": witnesses, "outcome": outcome,
            "inputs": inputs, "context": document["context"], "origin": origin}


def native_inputs(scope, case, arm, binding, bundle):
    document, _ = bound(scope.root, binding)
    require(document["schema"] == "hf2q.c2.native-input-records.v1"
            and document["execution"] == bundle["binding"], "native inputs lost their required execution binding")
    scope.header(document["context"], case, arm, frozen=bundle["context"]["acquisition"])
    records = document["records"]
    require(type(records) is list and len(records) == len(bundle["inputs"]), "native input records omit required forwards")
    seen, validated = set(), {}
    for record in records:
        require(type(record) is dict and set(record) == {"sequence", "slot", "start", "source", "actual_tokens", "actual_sha256"},
                "closed actual native input record required")
        sequence = integer(record["sequence"], "native input source sequence")
        require(sequence in bundle["inputs"] and sequence not in seen, "unrequested or duplicated native input record")
        seen.add(sequence)
        geometry, source = bundle["forwards"][sequence], bundle["inputs"][sequence]
        require(integer(record["slot"], "actual input physical slot", 0, 0) == geometry["slot"]
                and integer(record["start"], "actual native input start") == geometry["start"]
                and record["source"] == source, "actual input changed its source descriptor")
        tokens = record["actual_tokens"]
        require(type(tokens) is list and len(tokens) == geometry["input_rows"], "actual native input geometry differs")
        actual = input_digest(tokens)
        require(record["actual_sha256"] == actual, "actual input values changed after checked observation")
        if "Host" in source:
            require(source["Host"]["sha256"] == actual, "consumed tokens differ from pre-call host input")
        validated[sequence] = record
    require(seen == set(bundle["inputs"]), "whole native input record omitted")
    return validated
