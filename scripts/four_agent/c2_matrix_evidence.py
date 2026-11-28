"""Reconcile each C2 arm against its required source plans, not its peer's rows."""
import hashlib
import math
from pathlib import Path

from c2_matrix_io import SCALARS, acquisition, bound, coordinate, integer, require, resolve
from c2_matrix_plans import expected, native_inputs, state
from q1_decision_policy import captured_records, exact, verified_payload


def scoped(scope, reference):
    path = resolve(scope.root, reference, absolute=True)
    return {"path": str(path.relative_to(scope.root)), "sha256": reference["sha256"]}


def rows(trace):
    return {coordinate(row["at"]): row for row in trace["coverage"]["captured"]}


def row(trace, stage, agent, sequence, position, segment=0):
    key = (stage, agent, position, 0, segment, sequence)
    require(key in trace, "required source row is absent")
    return trace[key]


def equal(left, right):
    verdict = exact(left, right)
    require(verdict["classification"] == "ExactEqual" and verdict["common_records"] > 0,
            "complete source-bound pair is not ExactEqual")
    return verdict


def semantic_state_pair(left_trace, right_trace):
    """Compare complete active state, retaining physical-selector evidence.

    Callers first authenticate each allocator-derived state plan/store. C2
    NamedResetParity restores into named buffers; a cold twin may select the
    other buffer. The recorded regions already use the parity-current views.
    Intra-arm settlement and no-mutation checks still use literal `exact`.
    """
    left, right = captured_records(left_trace), captured_records(right_trace)
    for records in (left, right):
        require(all(key[0] == "CacheState" and key[2] == key[3] == key[5] == 0 for key in records),
                "semantic state comparison requires a complete typed state store")
    selectors = [[key for key in records if key[4] == 1] for records in (left, right)]
    require(len(selectors[0]) == len(selectors[1]) == 1 and selectors[0] == selectors[1],
            "both states require the same physical-selector extent")
    selector = selectors[0][0]
    cursor = (*selector[:4], 0, selector[5])
    require(cursor in left and cursor in right, "semantic state omitted target/MTP cursors")
    parity = []
    for records in (left, right):
        record = records[selector]
        raw = verified_payload(record)
        require(record["scalar"] == "U8" and len(record["shape"]) == 1
                and integer(record["shape"][0], "physical parity count", 1) == len(raw)
                and all(value in (0, 1) for value in raw), "invalid physical buffer-selector metadata")
        parity.append(raw)
    common = sorted(set(left) & set(right))
    left_only, right_only = len(set(left) - set(right)), len(set(right) - set(left))
    differing = []
    for key in common:
        l, r = left[key], right[key]
        same_layout = (l["scalar"], l["shape"]) == (r["scalar"], r["shape"])
        if not same_layout or (key != selector and verified_payload(l) != verified_payload(r)):
            differing.append(key)
    classification = ("DivergedExactInvariant" if differing else
                      "ExactOnCommon" if left_only + right_only else "ExactEqual")
    return {"schema": "hf2q.c2.semantic-state-comparison.v1", "classification": classification,
            "common_records": len(common), "left_only": left_only, "right_only": right_only,
            "differing_records": len(differing), "first_differing": list(differing[0]) if differing else None,
            "semantic_records": len(common) - 1, "representation_records": 1,
            "representation_parity_equal": parity[0] == parity[1]}


def equal_semantic_state(left, right):
    verdict = semantic_state_pair(left, right)
    require(verdict["classification"] == "ExactEqual" and verdict["semantic_records"] > 0,
            "complete active-state pair is not ExactEqual")
    return verdict


def state_plan(scope, case, arm, binding, trace_reference, schema, event=None, frozen=None):
    plan, _ = bound(scope.root, binding)
    require(plan["schema"] == schema and plan["derived_before_state_read"] is True,
            "allocator geometry was not registered before state readback")
    scope.header(plan, case, arm, name="scope_acquisition", frozen=frozen)
    if event is not None:
        require(plan["event"] == event, "state child changed its required parent event")
    wanted = state(plan["layout"], f"{case}/slot0")
    require(plan["expected"] == wanted, "state extents differ from independent allocator geometry")
    path, trace = scope.trace(trace_reference, wanted)
    require(plan["store_root"] == str(path.parent), "state plan points at another store")
    return path, trace


def final_state(scope, case, arm, binding):
    value, _ = bound(scope.root, binding)
    require(value["schema"] == "hf2q.c2.final-state-binding.v1", "final state binding schema differs")
    trace_path = resolve(scope.root, value["trace"])
    scope.header(value, case, arm, output=trace_path.parent)
    return state_plan(scope, case, arm, value["plan"], value["trace"], "hf2q.c2.final-state-plan.v1")


def transactions(scope, case, arm, binding, bundle):
    value, path = bound(scope.root, binding)
    require(value["schema"] == "hf2q.c2.transaction-index.v1"
            and value["all_declared_state_events_completed"] is True, "unfinished state event stream")
    frozen = bundle["context"]["acquisition"]
    scope.header(value, case, arm, name="scope_acquisition", frozen=frozen)
    declared, children = value["declared_events"], value["children"]
    wanted = {index for index, entry in enumerate(bundle["entries"]) if entry["kind"] in ("State", "Settlement")}
    require(type(declared) is list and type(children) is list and len(declared) == len(children) == len(wanted) > 0,
            "whole required state event or settlement omitted")
    seen, result = set(), []
    for ordinal, (event, child) in enumerate(zip(declared, children)):
        require(type(event) is dict and set(event) == {"ordinal", "ticket", "kind"}, "closed state event required")
        require(integer(event["ordinal"], "state event ordinal") == ordinal and child["event"] == event,
                "state events reordered or replaced")
        ticket = integer(event["ticket"], "state event ticket", 0, len(bundle["entries"]) - 1)
        require(ticket in wanted and ticket not in seen, "state event duplicated or unrequested")
        seen.add(ticket)
        kind = "Settlement" if bundle["entries"][ticket]["kind"] == "Settlement" else "Transaction"
        require(event["kind"] == kind and (kind != "Settlement" or ordinal == len(declared) - 1),
                "typed settlement is not the final state event")
        plan_reference = {"path": child["plan"], "sha256": child["plan_sha256"]}
        plan_path = resolve(path.parent, plan_reference)
        root = path.parent / child["root"]
        require(root.is_relative_to(path.parent) and root.resolve(strict=True) == root and root.is_dir(),
                "state child root escapes or aliases its owner")
        trace_reference = {"path": str((root / "trace.json").relative_to(scope.root)), "sha256": child["trace_sha256"]}
        result.append(state_plan(scope, case, arm,
            {"path": str(plan_path.relative_to(scope.root)), "sha256": child["plan_sha256"]}, trace_reference,
            "hf2q.c2.state-plan.v1", event, frozen))
        logical_bytes = sum(math.prod(item["shape"]) * SCALARS[item["scalar"]]
                            for item in result[-1][1]["coverage"]["expected"])
        require(integer(child["logical_bytes"], "state event bytes") == logical_bytes, "state event byte count differs")
    require(seen == wanted, "required state event stream is incomplete")
    return result


def raw_arm(scope, case, arm, binding, bundle):
    value, _ = bound(scope.root, binding)
    require(value["schema"] == "hf2q.c2.raw-arm-binding.v1" and value["identity"] == scope.identity
            and value["case"] == case and value["arm"] == arm and value["execution"] == bundle["binding"],
            "raw arm lost its owning source execution")
    acquisition(value["scope_acquisition"], scope.actual, scope.root)
    inputs = native_inputs(scope, case, arm, value["native_inputs"], bundle)
    path, trace = scope.trace(value["trace"], bundle["raw"])
    route, _ = bound(scope.root, value["route_observation"])
    require(route["schema"] == "hf2q.c2.route-observation.v1" and route["unforced"] is True,
            "raw arm has no unforced route observation")
    require(type(route["routes"]) is list and len(route["routes"]) == len(bundle["forwards"])
            and all(type(item) is str and item for item in route["routes"]), "raw forward route omitted")
    selected = route["actual_gpu_argmax"]
    require(type(selected) is list and len(selected) == len(bundle["gpu_positions"]), "GPU selector omitted")
    positions = []
    for selection in selected:
        require(type(selection) is list and len(selection) == 2, "closed GPU selector observation required")
        positions.append(integer(selection[0], "actual GPU selection position"))
        integer(selection[1], "actual GPU-selected token", 0, (1 << 32) - 1)
    require(sorted(positions) == sorted(bundle["gpu_positions"]), "GPU selections differ from required native heads")
    require(type(route["completed_route_events"]) is list
            and all(type(item) is str for item in route["completed_route_events"]), "completed route events absent")
    return {"path": path, "trace": trace, "rows": rows(trace), "route": route,
            "inputs": inputs, "forwards": bundle["forwards"]}


def canonical_arm(scope, case, arm, source_arm, binding, bundle, raw, prompt, tokens, graft):
    value, _ = bound(scope.root, binding)
    require(value["schema"] == "hf2q.c2.canonical-arm-binding.v1" and value["identity"] == scope.identity
            and value["case"] == case and value["arm"] == arm and value["source_arm"] == source_arm
            and value["execution"] == bundle["binding"], "canonical rows changed their owning raw arm")
    acquisition(value["scope_acquisition"], scope.actual, scope.root)
    path, trace = scope.trace(value["trace"], bundle["canonical"])
    routes, _ = bound(scope.root, value["route_observation"])
    require(type(routes) is list and len(routes) == len(bundle["decisions"]), "canonical route observation incomplete")
    canonical_rows, source_rows = rows(trace), raw["rows"]
    for index, selection in bundle["decisions"].items():
        position = selection["physical_position"]
        require(type(routes[index]) is list and len(routes[index]) == 3 and routes[index][:2] == [index, position]
                and type(routes[index][2]) is str, "canonical source route coordinate differs")
        agent = f"{case}-single-slot-parity/slot0"
        canonical_id = row(canonical_rows, "SpecCanonicalIds", agent, index, position)
        source_id = row(source_rows, "SpecCanonicalIds", f"{case}/slot0", index, index)
        require(canonical_id["sha256"] == source_id["sha256"], "canonical token differs from its required actual selector")
        copied = tuple(row(canonical_rows, stage, agent, index, position)["sha256"]
                       for stage in ("Logits", "LayerOutput"))
        candidates = []
        for sequence, request in sorted(bundle["forwards"].items()):
            first = request["start"] + request["input_rows"] - request["projected_rows"]
            if not first <= position < first + request["projected_rows"]:
                continue
            accepted = True
            for offset, token in enumerate(raw["inputs"][sequence]["actual_tokens"]):
                physical = request["start"] + offset
                if physical > position or physical < graft:
                    continue
                logical = physical - graft
                prefix = prompt if logical < len(prompt) else tokens
                logical_index = logical if logical < len(prompt) else logical - len(prompt)
                if logical_index >= len(prefix) or prefix[logical_index] != token:
                    accepted = False
                    break
            if accepted:
                values = tuple(row(source_rows, stage, f"{case}/slot0", sequence, position)["sha256"]
                               for stage in ("Logits", "LayerOutput"))
                candidates.append((sequence, values))
        require(candidates and all(values == candidates[0][1] for _, values in candidates),
                "accepted-prefix canonical row is absent or repeated source rows diverge")
        require(copied == candidates[-1][1] and routes[index][2] == raw["route"]["routes"][candidates[-1][0]],
                "canonical full rows/route were not captured on the actual accepted input prefix")
    return path, trace


def prefix_digest(tokens):
    require(type(tokens) is list and tokens, "original checkpoint prefix tokens absent")
    value = hashlib.sha256(b"hf2q.c2.checkpoint-prefix.le-u32.v1\0")
    value.update(len(tokens).to_bytes(8, "little"))
    for token in tokens:
        value.update(integer(token, "original prefix token", 0, (1 << 32) - 1).to_bytes(4, "little"))
    return value.hexdigest()


def saved_witnesses(scope, case, bundle, raw, authority):
    for request, reference in bundle["witnesses"]:
        receipt, _ = bound(scope.root, reference, absolute=True)
        saved = request["origin"]["SavedCheckpoint"]
        require(receipt["schema"] == "hf2q.c2.checkpoint-head-witness.v1" and receipt["identity"] == scope.identity
                and receipt["id"] == saved["witness"] and receipt["position"] == request["start"]
                and receipt["vocab"] == request["vocab"] and receipt["hidden_width"] == request["hidden_width"],
                "saved row receipt belongs to another original checkpoint")
        acquisition(receipt["acquisition"], scope.actual, scope.root)
        require(type(authority) is list and any(type(turn) is dict and "prompt" in turn
                and len(turn["prompt"]) == receipt["tokens"] and prefix_digest(turn["prompt"]) == receipt["prefix_sha256"]
                for turn in authority), "saved row original prefix is absent from the independent input authority")
        agent, position = f"checkpoint-{receipt['id']}/slot0", receipt["position"]
        normal = [expected(stage, agent, 0, position, "F32", [1, width])
                  for stage, width in (("Logits", receipt["vocab"]), ("LayerOutput", receipt["hidden_width"]))]

        def original(plan_reference, trace_reference, wanted, schema, native):
            plan, _ = bound(scope.root, plan_reference, absolute=True)
            require(plan["schema"] == schema and plan["id"] == receipt["id"] and plan["identity"] == scope.identity
                    and plan["acquisition"] == receipt["acquisition"] and plan["prefix_sha256"] == receipt["prefix_sha256"]
                    and plan["tokens"] == receipt["tokens"] and plan["expected"] == wanted
                    and plan["native_prefill_required"] is native, "original checkpoint plan differs from requested geometry")
            return scope.trace(scoped(scope, trace_reference), wanted)[1]

        normal_trace = original(receipt["plan"], receipt["normal"], normal, "hf2q.c2.checkpoint-head-plan.v1", True)
        chosen = row(rows(normal_trace), "Logits", agent, 0, position)
        require(row(raw["rows"], "Logits", f"{case}/slot0", request["sequence"], position)["sha256"] == chosen["sha256"],
                "saved logits differ from the original native checkpoint row")
        segment, hidden_trace = 0, normal_trace
        if saved["hidden"] == "ConsumedSpeculative":
            spec = receipt["spec_hidden"]
            require(type(spec) is dict, "consumed speculative hidden has no original boundary witness")
            wanted = [expected("LayerOutput", agent, 0, position, "F32", [1, receipt["hidden_width"]], 1)]
            hidden_trace = original(spec["plan"], spec["trace"], wanted, "hf2q.c2.checkpoint-spec-plan.v1", False)
            segment = 1
        require(row(raw["rows"], "LayerOutput", f"{case}/slot0", request["sequence"], position)["sha256"]
                == row(rows(hidden_trace), "LayerOutput", agent, 0, position, segment)["sha256"],
                "saved hidden differs from its prepared original witness mode")


def logical_prompt(scope, case, authority):
    parts = case.split("-")
    require(len(parts) == 3 and parts[0] in ("rank0", "rank2") and parts[1] in ("eqfalse", "eqtrue")
            and parts[2] in ("peersfalse", "peerstrue"), "unknown original canonical case recipe")
    selected = 3 - int(parts[0][4:])
    require(type(authority) is list and len(authority) == 4, "original K4 input authority is incomplete")
    prompt = authority[selected]["prompt"]
    require(type(prompt) is list and prompt, "original checkpoint prompt is absent")
    for token in prompt:
        integer(token, "independent prompt token", 0, (1 << 32) - 1)
    if parts[1] == "eqfalse":
        lane = scope.packet["workload"]["lanes"][0]
        require(type(lane) is list and len(lane) >= 144, "original divergence input suffix is absent")
        suffix = lane[128:144]
        for token in suffix:
            integer(token, "original divergent suffix token", 0, (1 << 32) - 1)
        prompt = prompt + suffix
    return prompt


def same_raw_twins(left, right):
    require(set(left["inputs"]) == set(right["inputs"]) == set(left["forwards"]) == set(right["forwards"]),
            "same-concurrency twin omitted a consumed native input")
    geometry = ("sequence", "slot", "start", "input_rows", "projected_rows", "vocab", "hidden_width")
    for sequence, consumed in left["inputs"].items():
        other = right["inputs"][sequence]
        require(all(left["forwards"][sequence][key] == right["forwards"][sequence][key] for key in geometry)
                and all(consumed[key] == other[key] for key in ("sequence", "slot", "start", "actual_tokens")),
                "equal projection bytes hide different same-concurrency consumed inputs or geometry")
    for key in ("routes", "actual_gpu_argmax", "completed_route_events", "initial_readiness"):
        require(left["route"][key] == right["route"][key],
                "same-concurrency twin changed its actual native route or GPU selection schedule")


def settlement_proof(final, stream, outcome):
    require(stream, "required typed Settlement state is absent")
    proof = equal(final[0], stream[-1][0])
    recovery = outcome.get("Cancelled", outcome.get("Reference"))
    if recovery is not None:
        ledger = [item for item in final[1]["coverage"]["captured"]
                  if item["at"]["stage"] == "CacheState" and item["at"]["segment"] == (1 << 32) - 1]
        require(len(ledger) <= 1, "recovered ledger is ambiguous")
        depth = ledger[0]["shape"][0] if ledger else 0
        require(integer(recovery["recovered_tokens"], "typed recovered token count") == depth,
                "typed recovery count differs from its exact Settlement/final ledger extent")
    return proof


def capability_markers(scope, value, raw):
    route, graft = scope.packet["route"], scope.packet["graft"] is not None
    equality = value["equality"]
    reasons = [value[f"{arm}_lane0_admission_decision"] for arm in ("actual", "twin", "single")]
    allowed = {"disabled", "unsupported_semantics", "prompt_cache_hit", "unprofitable", "runtime_unavailable", "eligible"}
    require(all(reason in allowed and reason == reasons[0] for reason in reasons),
            "restored/cold/single admission capability markers differ")
    if route == "MtpAuto":
        reason = "eligible" if not graft else "prompt_cache_hit" if equality else "runtime_unavailable"
        literal = ("ObservedActualK3" if not graft else "PromptCacheHit" if equality else
                   "ExpectedCapabilityNegativeControl-ADR059-no-grafted-MTP")
        resolved = ("C2-SlotAware4-AUTO-K3" if not graft else "PromptCacheHit/ordinary-grafted" if equality else
                    "RuntimeUnavailable/ordinary-grafted")
        require(reasons[0] == reason, "AUTO admission marker changed the actual graft/equality capability branch")
    else:
        require(route in ("Off", "CurrentDefault"), "unknown actual route")
        literal = "NotDeclaredK3"
        resolved = "C2-SlotAware4-OFF" if route == "Off" else "C2-SlotAware4-current-default"
        if route == "Off":
            require(reasons[0] == "disabled", "OFF invented an eligible speculation admission")
    require(value["literal_k3_capability"] == literal and value["resolved_speculation"] == resolved,
            "declared capability/resolution markers differ from the actual route branch")
    for arm in ("actual", "twin", "single"):
        source = raw[arm] if arm in raw else raw["twin"]
        events = source["route"]["completed_route_events"]
        require(value[f"{arm}_route_events"] == events, "classification route events changed their bound observations")
        k3, joint = events.count("K3-actual-solo-completed"), events.count("JointMTP-actual-window-completed")
        require(integer(value[f"{arm}_lane0_completed_k3"], "actual completed K3 count") == k3
                and integer(value[f"{arm}_lane0_completed_joint_mtp"], "actual completed joint-MTP count") == joint,
                "declared K3 participation differs from actual completed route events")
        if route == "MtpAuto":
            require((k3 + joint > 0) is (not graft), "AUTO skipped eligible K3 or invented grafted MTP")
        elif route == "Off":
            require(k3 + joint == 0 and "DFlash2-actual-runner-completed" not in events, "OFF completed speculation")
        else:
            require(("DFlash2-actual-runner-completed" in events) is (not graft),
                    "current default completed the wrong capability route")
