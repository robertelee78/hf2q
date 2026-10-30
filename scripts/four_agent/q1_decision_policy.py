#!/usr/bin/env python3
"""ADR-063 Q1 decision-level numeric policy. Offline: never loads a model.

  exact LEFT RIGHT     Two hf2q.adr063.invariance.v2 trace.json stores. Every
                       common coordinate (all stages) must have identical
                       layout and payload bytes: ExactEqual, ExactOnCommon
                       (unpaired counts reported) or DivergedExactInvariant.
  envelope LEFT RIGHT  Two hf2q.adr063.invariance.v2 trace.json stores over the
                       same frozen inputs. Every Logits record is joined,
                       hash-checked and classified as ExactEqual,
                       WithinDecisionEnvelope or RouteSensitive.
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
import re
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


def exact(left_trace, right_trace):
    left, right = captured_records(left_trace), captured_records(right_trace)
    common = sorted(set(left) & set(right))
    differing = []
    for key in common:
        l, r = left[key], right[key]
        same_layout = (l["scalar"], l["shape"]) == (r["scalar"], r["shape"])
        if not same_layout or verified_payload(l) != verified_payload(r):
            differing.append(key)
    unpaired = len(left) + len(right) - 2 * len(common)
    if differing:
        classification = "DivergedExactInvariant"
    else:
        classification = "ExactEqual" if not unpaired else "ExactOnCommon"
    return {"schema": "hf2q.q1.exact-store-comparison.v1", "classification": classification,
            "common_records": len(common), "left_only": len(set(left) - set(right)),
            "right_only": len(set(right) - set(left)), "differing_records": len(differing),
            "first_differing": list(differing[0]) if differing else None}


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

CALL_ID = re.compile(r"call_hf2q_[0-9a-f]+")


def turn_digests(run):
    turns = {}
    for answer in sorted(Path(run).glob("client/agent-*/turn-*/assembled-assistant.json")):
        message = json.loads(answer.read_text())
        for call in message.get("tool_calls") or []:
            call.pop("id", None)
        request = json.loads((answer.parent / "request.json").read_text())
        history = CALL_ID.sub("CALL", json.dumps(request["messages"], sort_keys=True))
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
    e = sub.add_parser("envelope")
    e.add_argument("left_trace")
    e.add_argument("right_trace")
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
            result = exact(args.left_trace, args.right_trace)
        elif args.command == "envelope":
            result = envelope(args.left_trace, args.right_trace)
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
