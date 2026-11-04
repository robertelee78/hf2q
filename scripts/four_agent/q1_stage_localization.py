"""ADR-063 Q1 stage localization record: execution-order first divergence per lane.

  python3 -B q1_stage_localization.py REFERENCE_STAGES_TRACE CANDIDATE_STAGES_TRACE WORKLOAD_JSON > record.json

Both traces are complete stage stores written under one frozen manifest (the
Gemma cohort localizer's All128 reference and packed route share it). For each
lane the record names the first (phase, decode position, layer, stage) group in
that lane's execution order whose rows are not byte-identical, the differing
rows and their magnitude. Router-id changes are listed with the reference's raw
router-logit kth/next gap; the selection-domain (softmax) margins belong to the
routing-witness `routing` decision, not to this record. The repeated-token
prefix of the prompt (rows 29-127) is disclosed with its repetition and the
divergence magnitude inside it. No fix, policy or whole-Q1 claim is made.
"""
import json
import math
import struct
import sys
from pathlib import Path

from q1_decision_policy import captured_records, require, verified_payload

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
    lanes = json.loads(Path(workload_path).read_text())["lanes"]
    require(len(lanes) == 4 and all(len(l) == 144 for l in lanes), "four 144-token workload lanes")
    record = {"schema": SCHEMA, "reference": str(left_path), "candidate": str(right_path), "lanes": [],
              "route_class": "ClassB: route-sensitive; a localized stage implies no fix", "whole_q1_qualified": False}
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


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    require(len(args) == 3, "usage: q1_stage_localization.py REFERENCE_TRACE CANDIDATE_TRACE WORKLOAD_JSON")
    print(json.dumps(localize(*args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
