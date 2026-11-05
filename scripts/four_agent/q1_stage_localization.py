"""ADR-063 Q1 stage localization record: execution-order first divergence per lane.

  python3 -B q1_stage_localization.py REFERENCE_STAGES_TRACE CANDIDATE_STAGES_TRACE WORKLOAD_JSON > record.json
  python3 -B q1_stage_localization.py ngram-map NGRAM_T5_EXECUTION ORDINARY_T5_EXECUTION PLAN > map.json
  python3 -B q1_stage_localization.py pass-binding MAP CANDIDATE_STAGES REFERENCE_STAGES CANDIDATE_LOGITS \\
      REFERENCE_LOGITS > pass-binding.json
  python3 -B q1_stage_localization.py mapped MAP PASS_BINDING WORKLOAD_JSON > record.json
  python3 -B q1_stage_localization.py all128-precondition ENDPOINT_TRACE WORKLOAD_JSON > precondition.json

Both traces are complete stage stores written under one frozen manifest (the
Gemma cohort localizer's All128 reference and packed route share it). For each
lane the record names the first (phase, decode position, layer, stage) group in
that lane's execution order whose rows are not byte-identical, the differing
rows and their magnitude. Router-id changes are listed with the reference's raw
router-logit kth/next gap; the selection-domain (softmax) margins belong to the
routing-witness `routing` decision, not to this record. The repeated-token
prefix of the prompt (rows 29-127) is disclosed with its repetition and the
divergence magnitude inside it. No fix, policy or whole-Q1 claim is made.

The Gemma T5 verifier-route localizer (#665) compares two different manifests
through a frozen key map instead. `ngram-map` derives it mechanically from the
two T5 execution receipts and the frozen localizer plan file ({branch, pass,
stage_localizer[, t5_receipts]}): branch N pairs the n-gram arm (candidate)
with the ordinary-global arm (reference) per phase (logits, verify-prefix,
verify-committed, continuation) plus the routing-witness pairs that
`q1_decision_policy.py routing-mapped` reads; branch P is the identity map of
the ordinary-global arm (candidate) against the All128 stage localizer
(reference) on prefill rows 0..127 and Logits row 127 (the candidate's decode
rows 128..143 are listed unmapped). `pass-binding` binds the map and the four
traces of one completed pass; `mapped` localizes from them.
`all128-precondition` checks branch P's precondition: the All128 endpoint
store's row-127 Logits argmax equals frozen[128] on every lane.
"""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

from q1_decision_policy import captured_records, logit_records, read_row, require, verified_payload

SCHEMA = "hf2q.q1.stage-localization.v1"
PROMPT = 128
TEMPLATE_ROWS = 29  # rows 0-28: real template prefix; rows 29-127: the repeated-token tail
FORMATS = {"F32": "f", "U32": "I", "I32": "i"}


def key(at):
    return (at["stage"], at["agent"], at["logical_row_start"], at["layer"], at["segment"], at["logical_step"])


def values(record):
    raw = verified_payload(record)
    fmt = FORMATS.get(record["scalar"])
    require(fmt is not None and len(raw) % 4 == 0, f"{record['path']}: unsupported stage scalar {record['scalar']}")
    return struct.unpack(f"<{len(raw) // 4}{fmt}", raw)


def group_of(at):
    decode = at["logical_row_start"] >= PROMPT
    return ("decode" if decode else "prefill", at["logical_row_start"] if decode else None, at["layer"], at["stage"])


def frozen_order(left_path, right_path):
    left, right = (json.loads(Path(p).read_text())["coverage"]["expected"] for p in (left_path, right_path))
    require([e["at"] for e in left] == [e["at"] for e in right], "stores are not one frozen manifest in one order")
    return [e["at"] for e in left]


def compare_group(rows, left, right):
    differing, max_abs, diff_sq, ref_sq = [], 0.0, 0.0, 0.0
    for row, at in rows:
        l, r = left[key(at)], right[key(at)]
        require((l["scalar"], l["shape"]) == (r["scalar"], r["shape"]), f"layout differs at {key(at)}")
        if verified_payload(l) == verified_payload(r):
            continue
        differing.append(row)
        if l["scalar"] == "F32":
            for a, b in zip(values(l), values(r)):
                max_abs = max(max_abs, abs(a - b)); diff_sq += (a - b) ** 2; ref_sq += a * a
    return differing, max_abs, (math.sqrt(diff_sq / ref_sq) if ref_sq else None)


def segment_magnitude(rows, left, right):
    """Differing rows and max |delta| inside the template prefix and the repeated tail of one prefill group."""
    out = {}
    for name, lo, hi in (("rows_0_28", 0, TEMPLATE_ROWS), ("rows_29_127", TEMPLATE_ROWS, PROMPT)):
        part = [(row, at) for row, at in rows if lo <= row < hi]
        differing, max_abs, rel = compare_group(part, left, right)
        out[name] = dict(rows=len(part), differing_rows=len(differing), max_abs=max_abs, relative_rms=rel)
    return out


def repetition(tokens):
    tail = tokens[TEMPLATE_ROWS:PROMPT]
    counts = {}
    for t in tail:
        counts[t] = counts.get(t, 0) + 1
    modal = max(counts, key=lambda t: (counts[t], -t))
    run = best = 1
    for a, b in zip(tail, tail[1:]):
        run = run + 1 if a == b else 1; best = max(best, run)
    return dict(rows="29-127", distinct_tokens=len(counts), modal_token=modal, modal_count=counts[modal], longest_run=best)


def routing_changes(lane_order, left, right):
    """Rows whose selected expert ids differ, with the reference raw router-logit kth/next gap."""
    changes = []
    for at in lane_order:
        if at["stage"] != "RouterIds":
            continue
        l, r = left[key(at)], right[key(at)]
        a, b = values(l), values(r)
        if a == b:
            continue
        logits_at = dict(at, stage="RouterLogits"); margin = None
        if key(logits_at) in left:
            ordered = sorted(values(left[key(logits_at)]), reverse=True); k = len(a)
            margin = ordered[k - 1] - ordered[k] if len(ordered) > k else None
        changes.append(dict(layer=at["layer"], row=at["logical_row_start"], set_changed=set(a) != set(b),
                            reference_ids=list(a), candidate_ids=list(b), reference_raw_logit_gap=margin))
    return changes


def localize(left_path, right_path, workload_path):
    order = frozen_order(left_path, right_path)
    left, right = captured_records(left_path), captured_records(right_path)
    require(set(left) == set(right) == {key(at) for at in order}, "stores do not cover the frozen manifest")
    lanes = workload_lanes(workload_path)
    record = {"schema": SCHEMA, "reference": str(left_path), "candidate": str(right_path), "lanes": [],
              "route_class": ROUTE_CLASS, "whole_q1_qualified": False}
    for lane in range(4):
        agent = "agent-" + str(lane)
        lane_order = [at for at in order if at["agent"] == agent]
        require(lane_order, f"no records on {agent}")
        groups = []
        for at in lane_order:
            g = group_of(at)
            if not groups or groups[-1][0] != g:
                groups.append((g, []))
            groups[-1][1].append((at["logical_row_start"], at))
        first, equal_before = None, 0
        for (phase, position, layer, stage), rows in groups:
            differing, max_abs, rel = compare_group(rows, left, right)
            if not differing:
                equal_before += 1
                continue
            first = dict(phase=phase, decode_position=position, layer=layer, stage=stage, rows_in_group=len(rows),
                         differing_rows=differing, max_abs=max_abs, relative_rms=rel)
            if phase == "prefill":
                first["prompt_segments"] = segment_magnitude(rows, left, right)
            break
        record["lanes"].append(dict(lane=lane, groups=len(groups), equal_groups_before_first=equal_before,
                                    first_divergence=first, repeated_prefix=repetition(lanes[lane]),
                                    router_id_changes=routing_changes(lane_order, left, right)))
    return record


def workload_lanes(workload_path):
    lanes = json.loads(Path(workload_path).read_text())["lanes"]
    require(len(lanes) == 4 and all(len(l) == TOKENS for l in lanes), "four 144-token workload lanes")
    return lanes


# --- Gemma T5 verifier-route key map (#665, design §2.5/§2.6) ---------------

MAP_SCHEMA = "hf2q.q1.key-map.v1"
BINDING_SCHEMA = "hf2q.q1.localizer-pass-binding.v1"
MAPPED_SCHEMA = "hf2q.q1.mapped-stage-localization.v1"
T5_SCHEMA = "hf2q.q1.gemma-t5-phase.v1"
ROUTE_CLASS = "ClassB: route-sensitive; a localized stage implies no fix"
CLASS_B_REASON = "RouteSensitive: global-route initial decision differs from frozen at row 127"  # gemma_ngram::t5::CLASS_B
TOKENS = 144
VERIFY_ROWS = 132
LAYERS = 30  # Gemma 4 hidden layers; the routing witness covers all of them
LANES = [0, 1, 2, 3]
LOGIT_LAYER = 0xFFFFFFFF  # u32::MAX
# gemma_localizer::PREFILL_STAGES and gemma_cohort_localizer::DECODE_STAGES, in source order.
PREFILL_STAGES = ("Query", "Key", "Value", "RopeQuery", "RopeKey", "Attention", "Residual", "FfnNorm", "RouterLogits",
                  "RouterIds", "RouterWeights", "DenseOutput", "MoeOutput", "LayerOutput")
DECODE_STAGES = ("Norm", "Query", "Key", "RopeQuery", "RopeKey", "Value", "Attention", "FfnNorm", "RouterLogits",
                 "RouterIds", "RouterWeights", "LayerOutput")
PASS1_STAGES = ("RouterLogits", "RouterIds", "LayerOutput")
PASS2_MAX_LAYERS = 8
PLAN_KEYS = {"rerun_id", "layers", "prefill_stages", "decode_stages", "host_payload_limit", "accepted"}
STAGE_PHASES = {"N": ("verify-prefix", "verify-committed", "continuation"), "P": ("prefill",)}


def sha256_file(path):
    raw = Path(path).read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def k(stage, lane, row, layer, segment=0, step=0):
    return (stage, "agent-%d" % lane, row, layer, segment, step)


def at_of(key_):
    return dict(zip(("stage", "agent", "logical_row_start", "layer", "segment", "logical_step"), key_))


def map_key(item):
    require(isinstance(item, list) and len(item) == 6 and all(isinstance(v, str) for v in item[:2])
            and all(type(v) is int for v in item[2:]), f"malformed coordinate key {item!r}")
    return tuple(item)


def lane_of(key_):
    return int(key_[1].split("-", 1)[1])


def execution_order(key_):
    """Candidate per-lane execution order: prefill (segment 1 verify re-prefill or rows < 128) layer -> stage
    -> row, then decode position -> layer -> stage."""
    stage, _, row, layer, segment, _ = key_
    if segment == 1 or row < PROMPT:
        return (lane_of(key_), 0, layer, PREFILL_STAGES.index(stage), row)
    return (lane_of(key_), 1, row, layer, DECODE_STAGES.index(stage))


def t5_receipt(path, arm):
    raw, digest = sha256_file(path)
    doc = json.loads(raw)
    require(doc.get("schema") == T5_SCHEMA and doc.get("arm") == arm, f"{path}: not a {arm} {T5_SCHEMA} receipt")
    require(doc.get("lanes") == LANES and doc.get("whole_q1_qualified") is False, f"{path}: lanes/qualification")
    return digest, doc


def complete_decisions(doc, path):
    require(doc["status"] == "CompleteScopedAcquisition" and doc["error"] is None and doc["class_b_refusal"] is None,
            f"{path}: receipt is not complete, carries an error or carries a Class B refusal")
    require([d["lane"] for d in doc["decisions"]] == LANES, f"{path}: one decision per lane 0..3 in order")
    return doc["decisions"]


def accepted_counts(doc, path):
    accepted = []
    for decision in complete_decisions(doc, path):
        t = decision["transaction"]
        require(t is not None, f"{path}: lane {decision['lane']} has no transaction")
        a = t["accepted"]
        require(type(a) is int and a in (0, 1, 2) and t["rejected"] == 3 - a and t["rollback_cursor"] == 129 + a,
                f"{path}: lane {decision['lane']} transaction is not a in {{0,1,2}}, rejected 3-a, cursor 129+a")
        accepted.append(a)
    return accepted


def frozen_plan(plan, pass_):
    """The localizer packet's `stage_localizer` Plan under the frozen two-pass rule (Rust `require_pass`; both
    branches: the branch-P candidate is the ordinary-arm localizer, which captures decode rows too)."""
    require(isinstance(plan, dict) and set(plan) == PLAN_KEYS and type(plan["host_payload_limit"]) is int,
            "stage_localizer must carry exactly the Plan keys")
    layers, prefill, decode = plan["layers"], plan["prefill_stages"], plan["decode_stages"]
    require(layers and all(type(l) is int and 0 <= l < LAYERS for l in layers)
            and all(a < b for a, b in zip(layers, layers[1:])), "plan layers must be ascending distinct Gemma layers")
    if pass_ == 1:
        require(layers == list(range(LAYERS)) and prefill == list(PASS1_STAGES) and decode == list(PASS1_STAGES),
                "pass 1 plan: all 30 layers, roles {RouterLogits, RouterIds, LayerOutput}")
    else:
        require(pass_ == 2, "pass must be 1 or 2")
        require(len(layers) <= PASS2_MAX_LAYERS and prefill == list(PREFILL_STAGES) and decode == list(DECODE_STAGES),
                "pass 2 plan: at most 8 layers, all 14 prefill and 12 decode roles")
    return plan


def build_map(branch, pass_, plan, receipts, accepted):
    """Pairs per phase in candidate execution order; keys are T2Loc's key(at)."""
    layers, prefill, decode = plan["layers"], plan["prefill_stages"], plan["decode_stages"]
    phases = {name: [] for name in ("logits", *STAGE_PHASES[branch])}
    unmapped = {"candidate": [], "reference": []}
    routing = None
    for lane in LANES:
        if branch == "P":
            # Identity map on the ordinary-global prefill rows 0..127 and Logits row 127; the reference (All128,
            # decode_stages []) has no decode rows, so the candidate's decode rows 128..143 are listed unmapped.
            phases["logits"].append((k("Logits", lane, PROMPT - 1, LOGIT_LAYER),) * 2)
            for layer in layers:
                for stage in prefill:
                    phases["prefill"] += [(k(stage, lane, row, layer),) * 2 for row in range(PROMPT)]
            unmapped["candidate"] += [k(stage, lane, position, layer) for position in range(PROMPT, TOKENS)
                                      for layer in layers for stage in decode]
            continue
        a = accepted[lane]
        phases["logits"] += [(k("Logits", lane, row, LOGIT_LAYER),) * 2 for row in range(PROMPT, TOKENS)]
        for layer in layers:
            for stage in prefill:
                for row in range(VERIFY_ROWS):
                    candidate = k(stage, lane, row, layer, 1)
                    if row < PROMPT:
                        phases["verify-prefix"].append((candidate, k(stage, lane, row, layer)))
                    elif row <= PROMPT + a and stage in decode:
                        phases["verify-committed"].append((candidate, k(stage, lane, row, layer)))
                    else:
                        unmapped["candidate"].append(candidate)
        for position in range(PROMPT + 1 + a, TOKENS):
            for layer in layers:
                phases["continuation"] += [(k(stage, lane, position, layer),) * 2 for stage in decode]
        for position in range(PROMPT, PROMPT + 1 + a):
            for layer in layers:
                unmapped["reference"] += [k(stage, lane, position, layer) for stage in decode if stage not in prefill]
    for name in phases:
        phases[name].sort(key=lambda pair: execution_order(pair[0]) if name != "logits" else pair[0][1:3])
    if branch == "N":
        routing = {"phases": [], "unmapped": {"candidate": [], "reference": []}}
        spans = {"verify-prefix": ("T5Verify", lambda a: range(PROMPT)),
                 "verify-committed": ("T5Verify", lambda a: range(PROMPT, PROMPT + 1 + a)),
                 "continuation": ("T5Continuation", lambda a: range(PROMPT + 1 + a, TOKENS))}
        for name, (namespace, span) in spans.items():
            routing["phases"].append({"phase": name, "pairs": [
                [[namespace, lane, p, layer], ["Canonical", lane, p, layer]]
                for lane in LANES for p in span(accepted[lane]) for layer in range(LAYERS)]})
        routing["unmapped"]["candidate"] = (
            [["T5Initial", lane, p, layer] for lane in LANES for p in range(PROMPT) for layer in range(LAYERS)]
            + [["T5Verify", lane, p, layer] for lane in LANES for p in range(PROMPT + 1 + accepted[lane], VERIFY_ROWS)
               for layer in range(LAYERS)])
    doc = {"schema": MAP_SCHEMA, "branch": branch, "pass": pass_, "plan": plan, "receipts": receipts,
           "accepted": accepted, "lanes": LANES,
           "phases": [{"phase": name, "stores": "logits" if name == "logits" else "stages",
                       "pairs": [[list(c), list(r)] for c, r in pairs]} for name, pairs in phases.items()],
           "unmapped": {side: [list(x) for x in keys] for side, keys in unmapped.items()},
           "routing": routing, "whole_q1_qualified": False}
    validated_map(doc)
    return doc


def validated_map(doc):
    """Closed structure; one-to-one per phase, no stage key mapped in two phases or both mapped and unmapped.
    Returns {phase: [(candidate, reference)]} and the unmapped key sets."""
    require(doc.get("schema") == MAP_SCHEMA and doc.get("whole_q1_qualified") is False, "unexpected key-map schema")
    branch = doc["branch"]
    require(branch in STAGE_PHASES, "branch must be N or P")
    names = [p["phase"] for p in doc["phases"]]
    require(names == ["logits", *STAGE_PHASES[branch]], f"branch {branch} phases must be {STAGE_PHASES[branch]}")
    phases, seen = {}, {"candidate": set(), "reference": set()}
    for phase in doc["phases"]:
        pairs = [(map_key(c), map_key(r)) for c, r in phase["pairs"]]
        require(pairs, f"phase {phase['phase']} maps nothing")
        for side, keys in (("candidate", [c for c, _ in pairs]), ("reference", [r for _, r in pairs])):
            require(len(set(keys)) == len(keys), f"duplicate {side} key within phase {phase['phase']}")
            if phase["phase"] != "logits":
                require(not seen[side] & set(keys), f"{side} key mapped in two phases")
                seen[side] |= set(keys)
        phases[phase["phase"]] = pairs
    unmapped = {side: [map_key(x) for x in doc["unmapped"][side]] for side in ("candidate", "reference")}
    for side, keys in unmapped.items():
        require(len(set(keys)) == len(keys) and not seen[side] & set(keys), f"{side} unmapped key repeated or mapped")
    return phases, unmapped


def ngram_map(ngram_path, ordinary_path, plan_path):
    ngram_sha, ngram = t5_receipt(ngram_path, "NgramTransaction")
    ordinary_sha, ordinary = t5_receipt(ordinary_path, "OrdinaryGlobal")
    complete_decisions(ordinary, ordinary_path)
    packet = json.loads(Path(plan_path).read_text())
    branch, pass_ = packet["branch"], packet["pass"]
    require(branch in STAGE_PHASES and type(pass_) is int and pass_ in (1, 2), "plan branch must be N or P and pass 1 or 2")
    plan = frozen_plan(packet["stage_localizer"], pass_)
    if "t5_receipts" in packet:
        require(packet["t5_receipts"]["ngram"]["sha256"] == ngram_sha
                and packet["t5_receipts"]["ordinary"]["sha256"] == ordinary_sha,
                "plan t5_receipts do not bind these T5 receipts")
    if branch == "N":
        accepted = accepted_counts(ngram, ngram_path)
        require(plan["accepted"] == accepted, f"frozen accepted {plan['accepted']} differs from the receipt's {accepted}")
    else:
        refusal = ngram["class_b_refusal"]
        require(plan["accepted"] is None and isinstance(refusal, dict) and refusal.get("row") == PROMPT - 1
                and refusal.get("reason") == CLASS_B_REASON,
                "branch P needs accepted null and the n-gram receipt's row-127 Class B refusal")
        accepted = None
    receipts = {"ngram": {"path": str(ngram_path), "sha256": ngram_sha},
                "ordinary": {"path": str(ordinary_path), "sha256": ordinary_sha}}
    return build_map(branch, pass_, plan, receipts, accepted)


BINDING_TRACES = ("candidate_stages", "reference_stages", "candidate_logits", "reference_logits")


def pass_binding(map_path, *traces):
    require(len(traces) == len(BINDING_TRACES), "four traces: candidate/reference stages, candidate/reference logits")
    raw, map_sha = sha256_file(map_path)
    doc = json.loads(raw)
    validated_map(doc)
    return {"schema": BINDING_SCHEMA, "map_sha256": map_sha, "pass": doc["pass"],
            "traces": {name: {"path": str(Path(p).resolve()), "sha256": sha256_file(p)[1]}
                       for name, p in zip(BINDING_TRACES, traces)}}


def rekeyed(pairs, candidate, reference):
    """Both sides keyed by the candidate key, reference first (compare_group's left)."""
    left, right = {}, {}
    for c, r in pairs:
        require(c in candidate and r in reference, f"mapped pair {c}->{r} not captured")
        cr, rr = candidate[c], reference[r]
        require((cr["scalar"], cr["shape"]) == (rr["scalar"], rr["shape"]), f"layout differs for pair {c}->{r}")
        left[c], right[c] = rr, cr
    return left, right


def mapped_group(phase, key_):
    return (phase, key_[2] if phase == "continuation" else None, key_[3], key_[0])


def mapped(map_path, binding_path, workload_path):
    map_raw, map_sha = sha256_file(map_path)
    binding_raw, binding_sha = sha256_file(binding_path)
    doc, binding = json.loads(map_raw), json.loads(binding_raw)
    require(binding.get("schema") == BINDING_SCHEMA and set(binding) == {"schema", "map_sha256", "pass", "traces"},
            "unexpected pass-binding schema")
    require(binding["map_sha256"] == map_sha, "PASS_BINDING.map_sha256 differs from the map's sha256")
    require(binding["pass"] == doc["pass"], "PASS_BINDING pass differs from the map's pass")
    require(set(binding["traces"]) == set(BINDING_TRACES), "PASS_BINDING must bind exactly the four traces")
    for name, bound in binding["traces"].items():
        require(sha256_file(bound["path"])[1] == bound["sha256"], f"{name} trace sha256 differs from PASS_BINDING")
    phases, unmapped = validated_map(doc)
    branch, accepted = doc["branch"], doc["accepted"]
    stores = {name: captured_records(binding["traces"][name]["path"]) for name in BINDING_TRACES}
    stage_pairs = [(name, c, r) for name in STAGE_PHASES[branch] for c, r in phases[name]]
    for side in ("candidate", "reference"):
        mapped_keys = {c if side == "candidate" else r for _, c, r in stage_pairs}
        extra = sorted(set(stores[side + "_stages"]) - mapped_keys - set(unmapped[side]))
        absent = sorted((mapped_keys | set(unmapped[side])) - set(stores[side + "_stages"]))
        require(not extra and not absent, f"{side} stage store is not the mapped+unmapped extent: "
                f"{len(extra)} uncovered, {len(absent)} missing, first {(extra or absent)[:1]}")
    logit_left, logit_right = rekeyed(phases["logits"], stores["candidate_logits"], stores["reference_logits"])
    stage_left, stage_right = rekeyed([(c, r) for _, c, r in stage_pairs], stores["candidate_stages"],
                                      stores["reference_stages"])
    phase_of = {c: name for name, c, _ in stage_pairs}
    lanes = workload_lanes(workload_path)
    record = {"schema": MAPPED_SCHEMA, "branch": branch, "pass": doc["pass"], "map_sha256": map_sha,
              "pass_binding": binding, "pass_binding_sha256": binding_sha, "accepted": accepted, "lanes": [],
              "route_class": ROUTE_CLASS, "whole_q1_qualified": False}
    for lane in LANES:
        agent = "agent-%d" % lane
        lane_order = sorted((c for c in phase_of if c[1] == agent), key=execution_order)
        require(lane_order, f"no mapped stage records on {agent}")
        groups = []
        for c in lane_order:
            g = mapped_group(phase_of[c], c)
            if not groups or groups[-1][0] != g:
                groups.append((g, []))
            groups[-1][1].append((c[2], at_of(c)))
        first, equal_before = None, 0
        for (phase, position, layer, stage), rows in groups:
            differing, max_abs, rel = compare_group(rows, stage_left, stage_right)
            if not differing:
                equal_before += 1
                continue
            first = dict(phase=phase, decode_position=position, layer=layer, stage=stage, rows_in_group=len(rows),
                         differing_rows=differing, max_abs=max_abs, relative_rms=rel)
            if phase in ("verify-prefix", "prefill"):
                first["prompt_segments"] = segment_magnitude(rows, stage_left, stage_right)
            elif phase == "verify-committed":
                first["prompt_segments"] = {"rows_128_%d" % (PROMPT + accepted[lane]): dict(
                    rows=len(rows), differing_rows=len(differing), max_abs=max_abs, relative_rms=rel)}
            break
        logit_rows = [(c[2], at_of(c)) for c, _ in phases["logits"] if c[1] == agent]
        logit_differing, logit_max, logit_rel = compare_group(logit_rows, logit_left, logit_right)
        head = first is None and bool(logit_differing)
        if head:
            first = dict(phase="logits", decode_position=None, layer=None, stage="Head", rows_in_group=len(logit_rows),
                         differing_rows=logit_differing, max_abs=logit_max, relative_rms=logit_rel)
        record["lanes"].append(dict(
            lane=lane, accepted=None if accepted is None else accepted[lane], groups=len(groups),
            equal_groups_before_first=equal_before, first_divergence=first,
            logits=dict(rows=len(logit_rows), differing_rows=logit_differing, max_abs=logit_max,
                        relative_rms=logit_rel),
            head_rule=dict(fires=head, pass_2_skipped=head,
                           rule="Head only when no mapped stage group differs but committed Logits differ"),
            repeated_prefix=repetition(lanes[lane]),
            router_id_changes=routing_changes([at_of(c) for c in lane_order], stage_left, stage_right),
            unmapped=[dict(side=side, key=list(x), scalar=stores[side + "_stages"][x]["scalar"],
                           shape=stores[side + "_stages"][x]["shape"], sha256=stores[side + "_stages"][x]["sha256"])
                      for side in ("candidate", "reference") for x in unmapped[side] if x[1] == agent]))
    return record


PRECONDITION_SCHEMA = "hf2q.q1.gemma-all128-precondition.v1"
GEMMA_VOCAB = 262144


def all128_precondition(endpoint_trace, workload_path):
    """Branch P precondition (design §2.6, condition 7): the All128 reference endpoint store's own row-127 Logits
    argmax equals frozen[128] of the greedy144 workload on every lane."""
    _, trace_sha = sha256_file(endpoint_trace)
    workload_raw, workload_sha = sha256_file(workload_path)
    workload = json.loads(workload_raw)
    require(workload.get("family") == "Gemma4" and workload.get("vocab_size") == GEMMA_VOCAB,
            f"{workload_path}: not the Gemma4 greedy144 workload")
    frozen = workload_lanes(workload_path)
    rows = logit_records(endpoint_trace)
    lanes = []
    for lane in LANES:
        entry = rows.get(("agent-%d" % lane, PROMPT - 1, LOGIT_LAYER, 0, 0))
        require(entry is not None, f"{endpoint_trace}: no Logits row 127 on agent-{lane}")
        path, sha256, vocab = entry
        require(vocab == GEMMA_VOCAB, f"{endpoint_trace}: agent-{lane} row 127 is not a {GEMMA_VOCAB}-wide row")
        _, row = read_row(path, sha256, vocab)
        argmax = max(range(len(row)), key=row.__getitem__)  # first maximum
        lanes.append({"lane": lane, "argmax": argmax, "frozen_128": frozen[lane][PROMPT],
                      "equal": argmax == frozen[lane][PROMPT]})
    return {"schema": PRECONDITION_SCHEMA,
            "endpoint_trace": {"path": str(Path(endpoint_trace).resolve()), "sha256": trace_sha},
            "workload": {"path": str(Path(workload_path).resolve()), "sha256": workload_sha},
            "lanes": lanes, "holds": all(l["equal"] for l in lanes), "whole_q1_qualified": False}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    commands = {"ngram-map": (ngram_map, 3), "pass-binding": (pass_binding, 5), "mapped": (mapped, 3),
                "all128-precondition": (all128_precondition, 2)}
    if args and args[0] in commands:
        fn, arity = commands[args[0]]
        require(len(args) == arity + 1, f"usage: see q1_stage_localization.py {args[0]} docstring")
        result = fn(*args[1:])
        print(json.dumps(result, allow_nan=False) if args[0] == "ngram-map" else json.dumps(result, indent=2, allow_nan=False))
        return
    require(len(args) == 3, "usage: q1_stage_localization.py REFERENCE_TRACE CANDIDATE_TRACE WORKLOAD_JSON")
    print(json.dumps(localize(*args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
