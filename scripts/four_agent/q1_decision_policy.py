#!/usr/bin/env python3
"""ADR-063 Q1 decision-level numeric policy. Offline: never loads a model.

  exact LEFT RIGHT [--scope FILE]
                       Two hf2q.adr063.invariance.v2 trace.json stores. Every
                       common coordinate (all stages) must have identical
                       layout and payload bytes: ExactEqual, ExactOnCommon
                       (unpaired counts reported) or DivergedExactInvariant.
                       --scope: a closed JSON list of [stage, agent,
                       logical_row_start, layer, segment, logical_step] keys;
                       each must be captured in both stores and only those are
                       compared (adds scope_sha256, scoped_records).
  envelope LEFT RIGHT  Two hf2q.adr063.invariance.v2 trace.json stores over the
                       same frozen inputs. Every Logits record is joined,
                       hash-checked and classified as ExactEqual,
                       WithinDecisionEnvelope or RouteSensitive.
  routing LEFT RIGHT   Two endpoint routing-witness.jsonl files. Every
                       routing_row is joined on (namespace, lane, position,
                       layer); selected_ids are compared in order. Each changed
                       row reports both sides' k-th minus (k+1)-th score gap,
                       boundary_ids and domain: ExactRouting or RoutingChanged.
  routing-mapped LEFT RIGHT MAP
                       Candidate (n-gram) and reference (Canonical) witnesses
                       joined through a q1_stage_localization.py ngram-map
                       key map, one-to-one per phase, each phase reported
                       separately; every row is mapped or listed unmapped.
  fork HF2Q PEER       Two non-streamed responses with logprobs/top_logprobs
                       (chat `choices[0].logprobs.content` or the peer's
                       `/completion` `completion_probabilities`). Classifies
                       the first generated-token divergence as NoFork,
                       NearTie, NumericSuspect or Inconclusive.
  paired FILE          Paired per-instance task outcomes (subject vs
                       reference). Exact one-sided McNemar plus fixed floors.
  transcripts RUN...   Four-agent client run directories. Every turn whose
                       request history is identical across runs must have an
                       identical assistant message (tool-call ids excluded).

Thresholds live in POLICY and are frozen before candidate data; see
docs/research/adr063-q1-numeric-policy.md for their evidence.
"""
import argparse
import array
import hashlib
import heapq
import json
import math
import struct
import sys
from pathlib import Path

from qwen38_n1_verdict import Invalid, require

POLICY = {
    "schema": "hf2q.q1.numeric-policy.v1",
    # Largest decision error (nats) tolerated between two hf2q routes that
    # reorder arithmetic over identical inputs (chunking, verifier rows, batch).
    "route_decision_error_nats": 0.25,
    # hf2q-vs-peer first-fork rule.
    "fork_neartie_floor_nats": 0.10,
    "fork_suspect_floor_nats": 0.5,
    "fork_noise_ceiling_nats": 0.25,
    "fork_top_k": 5,
    "fork_noise_quantile": 0.95,
    # Paired task-quality rule.
    "paired_min_instances": 24,
    "paired_alpha": 0.05,
    "paired_max_excess_failure_fraction": 0.125,
    "paired_min_pass_fraction": 0.75,
}
TRACE_SCHEMA = "hf2q.adr063.invariance.v2"


def nearest_rank(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


# --- trace stores ---------------------------------------------------------

def captured_records(trace_path):
    trace_path = Path(trace_path)
    doc = json.loads(trace_path.read_text())
    require(doc.get("schema") == TRACE_SCHEMA, f"{trace_path}: unexpected trace schema")
    coverage = doc["coverage"]
    require(not coverage.get("errors"), f"{trace_path}: trace records capture errors")
    require(len(coverage["captured"]) == len(coverage["expected"]),
            f"{trace_path}: captured extent differs from expected extent")
    records = {}
    for entry in coverage["captured"]:
        at = entry["at"]
        key = (at["stage"], at["agent"], at["logical_row_start"], at["layer"], at["segment"], at["logical_step"])
        require(key not in records, f"{trace_path}: duplicate coordinate {key}")
        records[key] = {**entry, "path": trace_path.parent / entry["file"]}
    require(records, f"{trace_path}: no captured records")
    return records


def verified_payload(record):
    raw = record["path"].read_bytes()
    require(hashlib.sha256(raw).hexdigest() == record["sha256"], f"{record['path']}: payload SHA-256 mismatch")
    return raw


def scope_keys(scope_path):
    """A closed JSON list of [stage, agent, logical_row_start, layer, segment, logical_step] keys."""
    raw = Path(scope_path).read_bytes()
    doc = json.loads(raw)
    require(isinstance(doc, list) and doc, f"{scope_path}: scope must be a non-empty JSON list of coordinate keys")
    keys = []
    for item in doc:
        require(isinstance(item, list) and len(item) == 6 and all(isinstance(v, str) for v in item[:2])
                and all(type(v) is int for v in item[2:]), f"{scope_path}: malformed coordinate key {item!r}")
        keys.append(tuple(item))
    require(len(set(keys)) == len(keys), f"{scope_path}: duplicate coordinate key in scope")
    return hashlib.sha256(raw).hexdigest(), keys


def exact(left_trace, right_trace, scope=None):
    left, right = captured_records(left_trace), captured_records(right_trace)
    if scope is None:
        common = sorted(set(left) & set(right))
        left_only, right_only = len(set(left) - set(right)), len(set(right) - set(left))
    else:
        scope_sha256, common = scope_keys(scope)
        missing = [key for key in common if key not in left or key not in right]
        require(not missing, f"{len(missing)} scoped keys not captured in both stores, first {missing[:1]}")
        common, left_only, right_only = sorted(common), 0, 0
    differing = []
    for key in common:
        l, r = left[key], right[key]
        same_layout = (l["scalar"], l["shape"]) == (r["scalar"], r["shape"])
        if not same_layout or verified_payload(l) != verified_payload(r):
            differing.append(key)
    if differing:
        classification = "DivergedExactInvariant"
    else:
        classification = "ExactEqual" if not left_only + right_only else "ExactOnCommon"
    result = {"schema": "hf2q.q1.exact-store-comparison.v1", "classification": classification,
              "common_records": len(common), "left_only": left_only,
              "right_only": right_only, "differing_records": len(differing),
              "first_differing": list(differing[0]) if differing else None}
    if scope is not None:
        result.update(scope_sha256=scope_sha256, scoped_records=len(common))
    return result


# --- envelope -------------------------------------------------------------

def logit_records(trace_path):
    records = {}
    for key, entry in captured_records(trace_path).items():
        if key[0] != "Logits":
            continue
        require(entry["scalar"] == "F32" and len(entry["shape"]) == 2 and entry["shape"][0] == 1,
                f"{trace_path}: Logits record {key} is not one F32 row")
        records[key[1:]] = (entry["path"], entry["sha256"], entry["shape"][1])
    require(records, f"{trace_path}: no Logits records")
    return records


def read_row(path, sha256, vocab):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha256, f"{path}: payload SHA-256 mismatch")
    require(len(raw) == 4 * vocab, f"{path}: payload is not {vocab} F32 values")
    row = array.array("f")
    row.frombytes(raw)
    if sys.byteorder != "little":
        row.byteswap()
    require(all(math.isfinite(x) for x in row), f"{path}: non-finite logit")
    return raw, row


def log_partition(row):
    peak = max(row)
    return peak + math.log(math.fsum(math.exp(x - peak) for x in row))


def top_two(row):
    return heapq.nlargest(2, range(len(row)), key=row.__getitem__)


def compare_row(left, right, identical):
    a, a2 = top_two(left)
    b = a if identical else top_two(right)[0]
    other = b if b != a else a2
    decision = abs((left[a] - left[other]) - (right[a] - right[other]))
    if identical:
        kl = 0.0
    else:
        zl, zr = log_partition(left), log_partition(right)
        kl = math.fsum(math.exp(l - zl) * ((l - zl) - (r - zr)) for l, r in zip(left, right))
    return {"left_top1": a, "right_top1": b, "agree": a == b,
            "left_margin": left[a] - left[a2], "decision_error": decision, "kl": max(kl, 0.0)}


def envelope(left_trace, right_trace, policy=POLICY):
    left, right = logit_records(left_trace), logit_records(right_trace)
    missing = sorted(set(left) ^ set(right))
    if missing:
        return {"schema": "hf2q.q1.decision-envelope.v1", "policy": policy["schema"],
                "classification": "Incomplete", "unpaired_coordinates": len(missing),
                "first_unpaired": list(missing[0])}
    rows = []
    for key in sorted(left):
        require(left[key][2] == right[key][2], f"vocabulary differs at {key}")
        lraw, lrow = read_row(*left[key])
        rraw, rrow = read_row(*right[key])
        row = compare_row(lrow, rrow, lraw == rraw)
        row.update(agent=key[0], row=key[1], exact=lraw == rraw)
        rows.append(row)
    tau = policy["route_decision_error_nats"]
    over = [r for r in rows if r["decision_error"] > tau]
    if all(r["exact"] for r in rows):
        classification = "ExactEqual"
    elif over:
        classification = "RouteSensitive"
    else:
        classification = "WithinDecisionEnvelope"
    detail = ("agent", "row", "left_top1", "right_top1", "left_margin", "decision_error", "kl")
    decisions = [r["decision_error"] for r in rows]
    kls = [r["kl"] for r in rows]
    return {
        "schema": "hf2q.q1.decision-envelope.v1", "policy": policy["schema"],
        "route_decision_error_nats": tau, "classification": classification,
        "rows": len(rows), "exact_rows": sum(r["exact"] for r in rows),
        "top1_agree": sum(r["agree"] for r in rows),
        "decision_error": {"p50": nearest_rank(decisions, 0.5), "p95": nearest_rank(decisions, 0.95),
                           "max": max(decisions)},
        "kl_nats": {"p50": nearest_rank(kls, 0.5), "p95": nearest_rank(kls, 0.95), "max": max(kls)},
        "rows_left_margin_below_tau": sum(r["left_margin"] < tau for r in rows),
        "flips": [{k: r[k] for k in detail} for r in rows if not r["agree"]],
        "rows_over_tau": [{k: r[k] for k in detail} for r in over],
    }


# --- routing --------------------------------------------------------------

ROUTING_IDENTITY = ("family", "model_sha256", "token_vectors_sha256")


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def routing_rows(path):
    path = Path(path)
    header = end = None
    rows = {}
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            event = json.loads(line)
            kind = event.get("kind")
            if kind == "header":
                require(header is None and not rows, f"{path}:{number}: header must come first, once")
                header = event
            elif kind == "routing_row":
                require(header is not None and end is None, f"{path}:{number}: row outside header/end")
                require(event["native_boundary_valid"] is True and event["selected_copy_valid"] is True
                        and event["native_decode_error"] is None, f"{path}:{number}: invalid witness row")
                at = event["at"]
                key = (at["namespace"], at["row"]["lane"], at["row"]["position"], at["layer"])
                require(key not in rows, f"{path}:{number}: duplicate routing key {key}")
                rows[key] = event
            elif kind == "end":
                require(end is None, f"{path}:{number}: duplicate end")
                end = event
            else:
                raise Invalid(f"{path}:{number}: unknown event kind {kind!r}")
    require(header is not None and end is not None, f"{path}: missing header or end")
    require(end["all_scored_rows_captured"] is True and end["records"] == len(rows),
            f"{path}: end record count/coverage differs from rows")
    require(rows, f"{path}: no routing rows")
    return header, rows


def boundary_gap(event):
    scores = event["scores_f32_bits"]
    return f32(scores[0]) - f32(scores[1]) if len(scores) == 2 else None


def routing_side(event):
    return {"selected_ids": event["selected_ids"], "gap": boundary_gap(event),
            "boundary_ids": event["boundary_ids"], "domain": event["at"]["domain"]}


def routing_witnesses(left_path, right_path):
    left_header, left = routing_rows(left_path)
    right_header, right = routing_rows(right_path)
    for field in ROUTING_IDENTITY:
        require(left_header["identity"][field] == right_header["identity"][field],
                f"routing witnesses differ in identity.{field}; not the same frozen inputs")
    return left, right


def routing_pairs(pairs, left, right):
    """Selected-id comparison of joined (left key, right key) witness rows."""
    changed = []
    for lkey, rkey in pairs:
        l, r = left[lkey], right[rkey]
        if l["selected_ids"] != r["selected_ids"]:
            row = {"namespace": lkey[0], "lane": lkey[1], "position": lkey[2], "layer": lkey[3],
                   "set_changed": set(l["selected_ids"]) != set(r["selected_ids"]),
                   "left": routing_side(l), "right": routing_side(r)}
            if rkey != lkey:
                row["right_key"] = list(rkey)
            changed.append(row)
    margins = [abs(side["gap"]) for row in changed for side in (row["left"], row["right"])
               if side["gap"] is not None]
    return {"rows": len(pairs), "changed_rows": len(changed),
            "set_changed_rows": sum(row["set_changed"] for row in changed),
            "order_only_rows": sum(not row["set_changed"] for row in changed),
            "weight_bits_differing_rows": sum(left[l]["selected_weights_f32_bits"]
                                              != right[r]["selected_weights_f32_bits"] for l, r in pairs),
            "changed_gap_max": max(margins) if margins else None,
            "changed": changed}


def routing(left_path, right_path):
    left, right = routing_witnesses(left_path, right_path)
    unpaired = sorted(set(left) ^ set(right))
    require(not unpaired, f"{len(unpaired)} unpaired routing keys, first {unpaired[:1]}")
    result = routing_pairs([(key, key) for key in sorted(left)], left, right)
    return {"schema": "hf2q.q1.routing-comparison.v1",
            "classification": "RoutingChanged" if result["changed_rows"] else "ExactRouting", **result}


KEY_MAP_SCHEMA = "hf2q.q1.key-map.v1"
# routing-mapped phase -> the candidate (left, n-gram arm) namespace; the reference side is always Canonical.
ROUTING_PHASES = {"verify-prefix": "T5Verify", "verify-committed": "T5Verify", "continuation": "T5Continuation"}


def routing_span(phase, a):
    """Absolute positions of one routing phase on a lane with accepted count a (design §2.5)."""
    return {"verify-prefix": range(128), "verify-committed": range(128, 129 + a),
            "continuation": range(129 + a, 144)}[phase]


def routing_key(item):
    require(isinstance(item, list) and len(item) == 4 and isinstance(item[0], str)
            and all(type(v) is int for v in item[1:]), f"malformed routing key {item!r}")
    return tuple(item)


def routing_mapped(left_path, right_path, map_path):
    """LEFT is the candidate (n-gram arm) witness, RIGHT the reference (ordinary-global) witness."""
    raw = Path(map_path).read_bytes()
    key_map = json.loads(raw)
    require(key_map.get("schema") == KEY_MAP_SCHEMA, f"{map_path}: unexpected key-map schema")
    section, accepted = key_map.get("routing"), key_map.get("accepted")
    require(isinstance(section, dict), f"{map_path}: key map carries no routing pairs")
    require(isinstance(accepted, list) and len(accepted) == 4 and all(type(a) is int and a in (0, 1, 2) for a in accepted),
            f"{map_path}: routing pairs need the per-lane accepted counts a in {{0,1,2}}")
    left, right = routing_witnesses(left_path, right_path)
    seen_left, seen_right, phases = set(), set(), {}
    for phase in section["phases"]:
        name = phase["phase"]
        require(name in ROUTING_PHASES and name not in phases, f"unknown or repeated routing phase {name!r}")
        pairs = [(routing_key(l), routing_key(r)) for l, r in phase["pairs"]]
        for lkey, rkey in pairs:
            require(lkey[0] == ROUTING_PHASES[name] and rkey[0] == "Canonical",
                    f"{name} pair {lkey}->{rkey} outside its namespaces")
            require(lkey[1:] == rkey[1:], f"{name} pair {lkey}->{rkey} joins a different lane/position/layer")
            require(0 <= lkey[1] < 4 and lkey[2] in routing_span(name, accepted[lkey[1]]),
                    f"{name} pair {lkey}->{rkey} outside the phase's position span")
            require(lkey not in seen_left and rkey not in seen_right, f"routing key mapped twice: {lkey}->{rkey}")
            seen_left.add(lkey); seen_right.add(rkey)
            require(lkey in left and rkey in right, f"mapped routing pair {lkey}->{rkey} not captured")
        phases[name] = routing_pairs(pairs, left, right)
    require(set(phases) == set(ROUTING_PHASES), "key map must carry every routing phase")
    unmapped = {side: [routing_key(k) for k in section["unmapped"][side]] for side in ("candidate", "reference")}
    for side, rows, seen in (("candidate", left, seen_left), ("reference", right, seen_right)):
        listed = set(unmapped[side])
        require(len(listed) == len(unmapped[side]) and not listed & seen, f"{side} unmapped list repeats a key")
        unpaired = sorted(set(rows) ^ (seen | listed))
        require(not unpaired, f"{len(unpaired)} unpaired routing keys after mapping ({side}), first {unpaired[:1]}")
    return {"schema": "hf2q.q1.routing-comparison.v1",
            "classification": "RoutingChanged" if any(p["changed_rows"] for p in phases.values()) else "ExactRouting",
            "map_sha256": hashlib.sha256(raw).hexdigest(),
            "phase_pairs": {name: p["rows"] for name, p in phases.items()}, "phases": phases,
            "unmapped": [{"side": side, "namespace": k[0], "lane": k[1], "position": k[2], "layer": k[3],
                          **routing_side((left if side == "candidate" else right)[k])}
                         for side in ("candidate", "reference") for k in unmapped[side]]}


# --- fork -----------------------------------------------------------------

def token_bytes(entry):
    raw = entry.get("bytes")
    return bytes(raw) if raw is not None else entry["token"].encode()


def positions(response):
    if "completion_probabilities" in response:
        entries = response["completion_probabilities"]
    else:
        entries = response["choices"][0]["logprobs"]["content"]
    require(isinstance(entries, list) and entries, "response has no per-token logprobs")
    out = []
    for entry in entries:
        token = token_bytes(entry)
        top = {token_bytes(t): t["logprob"] for t in entry["top_logprobs"]}
        top.setdefault(token, entry["logprob"])
        out.append((token, top))
    return out


def ranked(top, k):
    return sorted(top, key=top.get, reverse=True)[:k]


def fork(hf2q_response, peer_response, policy=POLICY):
    hf, peer = positions(hf2q_response), positions(peer_response)
    k = next((i for i, (h, p) in enumerate(zip(hf, peer)) if h[0] != p[0]), None)
    base = {"schema": "hf2q.q1.fork-verdict.v1", "policy": policy["schema"]}
    if k is None:
        return {**base, "classification": "NoFork", "shared_tokens": min(len(hf), len(peer))}
    top_k = policy["fork_top_k"]
    noise = []
    for (_, lh), (_, lp) in zip(hf[:k], peer[:k]):
        shared = sorted(set(ranked(lh, top_k)) & set(ranked(lp, top_k)))
        noise += [abs((lh[x] - lh[y]) - (lp[x] - lp[y]))
                  for i, x in enumerate(shared) for y in shared[i + 1:]]
    e = nearest_rank(noise, policy["fork_noise_quantile"])
    a, lh = hf[k]
    b, lp = peer[k]
    hf_top, peer_top = ranked(lh, top_k), ranked(lp, top_k)
    result = {**base, "fork_index": k, "hf2q_token": a.decode(errors="replace"),
              "peer_token": b.decode(errors="replace"), "noise_E": e, "noise_pairs": len(noise),
              "hf2q_top": [[t.decode(errors="replace"), lh[t]] for t in hf_top],
              "peer_top": [[t.decode(errors="replace"), lp[t]] for t in peer_top],
              "hf2q_choice_is_raw_top1": bool(hf_top) and hf_top[0] == a,
              "peer_choice_is_raw_top1": bool(peer_top) and peer_top[0] == b}
    if len(lh) < top_k or len(lp) < top_k:
        return {**result, "classification": "Inconclusive", "reason": "fewer than top_k alternatives"}
    if b not in hf_top or a not in peer_top:
        return {**result, "classification": "NumericSuspect", "reason": "fork token outside top_k"}
    d = abs((lh[a] - lh[b]) - (lp[a] - lp[b]))
    result["decision_error_D"] = d
    scale = e if e is not None else 0.0
    if d > max(4 * scale, policy["fork_suspect_floor_nats"]) or scale > policy["fork_noise_ceiling_nats"]:
        return {**result, "classification": "NumericSuspect", "reason": "D or E above suspect bound"}
    if set(hf_top[:2]) == {a, b} == set(peer_top[:2]) and d <= max(2 * scale, policy["fork_neartie_floor_nats"]):
        return {**result, "classification": "NearTie"}
    return {**result, "classification": "Inconclusive", "reason": "between NearTie and NumericSuspect"}


# --- paired ---------------------------------------------------------------

def one_sided_sign_p(b, c):
    n = b + c
    return math.fsum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n if n else 1.0


def paired(document, policy=POLICY):
    require(document.get("schema") == "hf2q.q1.paired-task-outcomes.v1", "unexpected outcomes schema")
    instances = document["instances"]
    ids = [i["id"] for i in instances]
    require(len(ids) == len(set(ids)), "duplicate instance id")
    require(all(type(i["subject"]) is bool and (i["reference"] is None or type(i["reference"]) is bool)
                for i in instances), "outcomes must be booleans (reference may be null)")
    with_reference = {i["reference"] is not None for i in instances}
    require(len(with_reference) <= 1, "mixed referenced and unreferenced instances")
    n = len(instances)
    subject = sum(i["subject"] for i in instances)
    out = {"schema": "hf2q.q1.paired-task-verdict.v1", "policy": policy["schema"],
           "reference_kind": document.get("reference_kind"), "instances": n, "subject_pass": subject}
    if n < policy["paired_min_instances"]:
        return {**out, "verdict": "Insufficient"}
    floor_ok = subject >= policy["paired_min_pass_fraction"] * n
    if with_reference == {False}:
        return {**out, "verdict": "Pass" if floor_ok else "BelowFloor"}
    b = sum(not i["subject"] and i["reference"] for i in instances)
    c = sum(i["subject"] and not i["reference"] for i in instances)
    p = one_sided_sign_p(b, c)
    out.update(reference_pass=sum(i["reference"] for i in instances),
               subject_only_fail=b, reference_only_fail=c, one_sided_p=p)
    if p < policy["paired_alpha"]:
        return {**out, "verdict": "Loss"}
    if not floor_ok:
        return {**out, "verdict": "BelowFloor"}
    if b - c > math.floor(policy["paired_max_excess_failure_fraction"] * n):
        return {**out, "verdict": "Inconclusive"}
    return {**out, "verdict": "Pass"}


# --- transcripts ----------------------------------------------------------


def normalise_call_ids(messages):
    """Replace every tool-call id (assistant `tool_calls[].id` and tool `tool_call_id`) by its ordinal of first
    appearance, so random per-run ids (hf2q's `call_hf2q_…`, the peer's 32-character ids) never make two otherwise
    identical histories differ, while the call/result pairing stays part of the comparison."""
    ordinal = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            if "id" in call:
                call["id"] = ordinal.setdefault(call["id"], f"CALL{len(ordinal)}")
        if "tool_call_id" in message:
            message["tool_call_id"] = ordinal.setdefault(message["tool_call_id"], f"CALL{len(ordinal)}")
    return messages


def turn_digests(run):
    turns = {}
    for answer in sorted(Path(run).glob("client/agent-*/turn-*/assembled-assistant.json")):
        message = json.loads(answer.read_text())
        for call in message.get("tool_calls") or []:
            call.pop("id", None)
        request = json.loads((answer.parent / "request.json").read_text())
        history = json.dumps(normalise_call_ids(request["messages"]), sort_keys=True)
        key = (answer.parent.parent.name, answer.parent.name)
        turns[key] = (hashlib.sha256(history.encode()).hexdigest(),
                      hashlib.sha256(json.dumps(message, sort_keys=True).encode()).hexdigest())
    require(turns, f"{run}: no client turns")
    return turns


def transcripts(runs):
    require(len(runs) >= 2, "need at least two runs")
    digests = {str(run): turn_digests(run) for run in runs}
    keys = set().union(*digests.values())
    groups = violations = 0
    divergent = []
    for key in sorted(keys):
        by_request = {}
        for run, turns in digests.items():
            if key in turns:
                by_request.setdefault(turns[key][0], {}).setdefault(turns[key][1], []).append(run)
        for answers in by_request.values():
            groups += 1
            if len(answers) > 1:
                violations += 1
                divergent.append({"agent": key[0], "turn": key[1], "outputs": list(answers.values())})
    return {"schema": "hf2q.q1.transcript-equality.v1", "runs": len(runs), "turn_positions": len(keys),
            "same_request_groups": groups, "same_request_different_output": violations,
            "classification": "ExactTranscripts" if not violations else "DivergedTranscripts",
            "divergent": divergent}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    x = sub.add_parser("exact")
    x.add_argument("left_trace")
    x.add_argument("right_trace")
    x.add_argument("--scope")
    e = sub.add_parser("envelope")
    e.add_argument("left_trace")
    e.add_argument("right_trace")
    r = sub.add_parser("routing")
    r.add_argument("left_witness")
    r.add_argument("right_witness")
    m = sub.add_parser("routing-mapped")
    m.add_argument("left_witness")
    m.add_argument("right_witness")
    m.add_argument("key_map")
    f = sub.add_parser("fork")
    f.add_argument("hf2q_response")
    f.add_argument("peer_response")
    p = sub.add_parser("paired")
    p.add_argument("outcomes")
    t = sub.add_parser("transcripts")
    t.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.command == "exact":
            result = exact(args.left_trace, args.right_trace, args.scope)
        elif args.command == "envelope":
            result = envelope(args.left_trace, args.right_trace)
        elif args.command == "routing":
            result = routing(args.left_witness, args.right_witness)
        elif args.command == "routing-mapped":
            result = routing_mapped(args.left_witness, args.right_witness, args.key_map)
        elif args.command == "fork":
            result = fork(json.loads(Path(args.hf2q_response).read_text()),
                          json.loads(Path(args.peer_response).read_text()))
        elif args.command == "paired":
            result = paired(json.loads(Path(args.outcomes).read_text()))
        else:
            result = transcripts(args.runs)
    except (Invalid, KeyError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
