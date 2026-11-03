#!/usr/bin/env python3
"""ADR-063 Q4-GEMMA (#660): per-stage profile of four-agent runs, hf2q and the peer, from run receipts.

Read-only; launches nothing. Inputs are settled Q5 run directories (client/summary.json plus each turn's
raw response.bin), optionally a sanitized Metal System Trace export per run and hf2q's worker-exit
category dumps from profiled runs.

Layer 1, request level, both engines, unperturbed (every Q3/D2-style run already has it). Per turn:
  span      client send -> validated terminal (CLOCK_MONOTONIC, the client's own clock)
  prefill   the engine's prompt-processing time: hf2q x_hf2q_timing.prefill_time_secs (forward only; a
            joint warm cohort's whole forward is charged to every lane), peer timings.prompt_ms
  decode    the engine's generation time: hf2q decode_time_secs (first token -> finish, shared batch
            steps and Mixed-step prefill quanta included), peer predicted_ms
  residual  span - prefill - decode: queue, admission, HTTP and SSE delivery (same arithmetic for both)
  queue     hf2q only: total_time_secs - prefill - decode (server-side wait, the part of residual the
            server sees)
  client    previous terminal (or makespan start) -> this send: tool execution and client work
Per agent the stages sum to its completion time; the per-run figure is the mean over agents, and the
tail is makespan - mean agent completion.

Layer 2, GPU timeline (optional, --mst DIR): a sanitized `metal-gpu-execution-points` export (and the
`metal-gpu-submission-to-command-buffer-id` mapping to keep only the server process) is aligned to the
client clock by the offset that minimizes GPU-busy time outside request spans (searched around
--mst-offset-ns, the sidecar's anchor). Classes inside the makespan: GPU busy (union), GPU idle while a
request is in flight (host dispatch/encode/scheduling/sampling/delivery) and idle with no request.

Layer 3, hf2q kernel classes (optional, profiled runs only): `[CATSPLIT]` (decode body) and
`[PREFILL_CATSPLIT_JSON]` (warm prefill) lines in server.stderr, as shares of their GPU-busy sums.
Both serialize command buffers; the distortion is the profiled/unprofiled makespan ratio, which
`compare` reports when given both kinds of run.
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

SCHEMA = "hf2q.adr063.q4-gemma.stage-profile.v1"
SUCCESS = "FourAgentProtocolCompleteAwaitingIndependentReview"
STAGES = ("prefill", "decode", "residual", "client", "tail")
BLOCK = re.compile(r"-(w\d+|b\d+)-")
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 6306


def load_json(path):
    return json.loads(Path(path).read_text())


def _last_object(raw, key):
    """The JSON object after the last `"key":` in raw bytes, or None."""
    at = raw.rfind(b'"' + key + b'":')
    if at < 0:
        return None
    text = raw[at + len(key) + 3:].decode("utf-8", "replace").lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def server_timing(raw):
    """Engine-reported timing of one response (stream or unary), seconds; None when absent."""
    t = _last_object(raw, b"x_hf2q_timing")
    if t is not None:
        prefill, decode = float(t["prefill_time_secs"]), float(t["decode_time_secs"])
        total = t.get("total_time_secs")
        return {"engine": "hf2q", "prefill_s": prefill, "decode_s": decode,
                "queue_s": None if total is None else max(0.0, float(total) - prefill - decode),
                "gpu_syncs": t.get("gpu_sync_count"), "gpu_dispatches": t.get("gpu_dispatch_count")}
    t = _last_object(raw, b"timings")
    if t is not None and "prompt_ms" in t:
        return {"engine": "peer", "prefill_s": float(t["prompt_ms"]) / 1e3,
                "decode_s": float(t["predicted_ms"]) / 1e3, "queue_s": None,
                "gpu_syncs": None, "gpu_dispatches": None}
    return None


def turn_rows(run_dir):
    """One row per (agent, turn) of a run, ordered by agent then turn."""
    run_dir = Path(run_dir)
    summary = load_json(run_dir / "client" / "summary.json")
    start = summary["makespan_started_ns"]
    rows = []
    for agent, record in sorted(summary["agents"].items()):
        previous = start
        for turn in record["turns"]:
            raw = (run_dir / "client" / agent / f"turn-{turn['turn']}" / "response.bin").read_bytes()
            timing = server_timing(raw)
            if timing is None:
                raise ValueError(f"{run_dir}: {agent} turn {turn['turn']} has no engine timing")
            usage = turn.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            send, terminal = turn["send_monotonic_ns"], turn["validated_terminal_monotonic_ns"]
            span = (terminal - send) / 1e9
            events = turn.get("semantic_event_monotonic_ns") or []
            rows.append({
                "agent": agent, "turn": turn["turn"], "stream": bool(turn.get("stream")),
                "send_ns": send, "terminal_ns": terminal, "span_s": span,
                "client_s": (send - previous) / 1e9,
                "prefill_s": timing["prefill_s"], "decode_s": timing["decode_s"],
                "residual_s": span - timing["prefill_s"] - timing["decode_s"],
                "queue_s": timing["queue_s"], "engine": timing["engine"],
                "gpu_syncs": timing["gpu_syncs"], "gpu_dispatches": timing["gpu_dispatches"],
                "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
                "new_tokens": None if usage.get("prompt_tokens") is None else usage["prompt_tokens"] - cached,
                "completion_tokens": usage.get("completion_tokens"),
                "semantic_events_ns": events, "semantic_events": len(events) if turn.get("stream") else None,
                "semantic_ttft_s": (events[0] - send) / 1e9 if turn.get("stream") and events else None,
            })
            previous = terminal
        rows.append({"agent": agent, "turn": None, "client_s": (record["task_completed_monotonic_ns"] - previous) / 1e9,
                     "completed_ns": record["task_completed_monotonic_ns"]})
    return summary, rows


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def nearest_rank(values, q):
    values = sorted(values)
    return values[math.ceil(q * len(values)) - 1] if values else None


def union_length(intervals):
    total, end = 0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def overlap_with_union(a, b, intervals):
    """Length of [a, b) covered by the union of intervals."""
    return union_length([(max(a, s), min(b, e)) for s, e in intervals if s < b and e > a])


def sse_attribution(rows):
    """Per-slot worst semantic gap and whether it coincides with another agent's prefill window.

    A request's prefill window is [send, terminal - decode]: queue, admission and prompt processing as
    the client clock sees them. Gaps above twice the run's median adjacent gap are 'excess';
    the share of excess time covered by other agents' prefill windows is the SSE cadence cost attributable
    to co-scheduled prefill (the rest is the slot's own decode or emission cadence). Event counts matter:
    an engine that coalesces a tool call into few events has few, long gaps.
    """
    turns = [r for r in rows if r.get("turn") is not None]
    windows = {}
    for r in turns:
        windows.setdefault(r["agent"], []).append((r["send_ns"], r["terminal_ns"] - int(r["decode_s"] * 1e9)))
    gaps = []
    for r in turns:
        ev = r["semantic_events_ns"] if r["stream"] else []
        gaps += [(r["agent"], r["turn"], a, b) for a, b in zip(ev, ev[1:])]
    if not gaps:
        return {"gaps": 0}
    sizes = [b - a for _, _, a, b in gaps]
    threshold = 2 * median(sizes)
    slots, excess, covered = {}, 0, 0
    for agent, turn, a, b in gaps:
        foreign = [w for other, ws in windows.items() if other != agent for w in ws]
        cover = overlap_with_union(a, b, foreign)
        if b - a > threshold:
            excess += b - a
            covered += cover
        best = slots.get(agent)
        if best is None or b - a > best["gap_s"] * 1e9:
            slots[agent] = {"gap_s": (b - a) / 1e9, "turn": turn,
                            "foreign_prefill_overlap": cover / (b - a) if b > a else 0.0}
    worst = max(slots.items(), key=lambda kv: kv[1]["gap_s"])
    return {"gaps": len(gaps), "p95_gap_s": nearest_rank(sizes, 0.95) / 1e9, "excess_threshold_s": threshold / 1e9,
            "excess_gap_s": excess / 1e9,
            "excess_foreign_prefill_share": covered / excess if excess else 0.0,
            "worst_slot": worst[0], "worst_slot_gap_s": worst[1]["gap_s"],
            "worst_slot_foreign_prefill_overlap": worst[1]["foreign_prefill_overlap"], "slots": slots}


def run_profile(run_dir):
    run_dir = Path(run_dir)
    summary, rows = turn_rows(run_dir)
    turns = [r for r in rows if r["turn"] is not None]
    agents = sorted({r["agent"] for r in rows})
    start = summary["makespan_started_ns"]
    makespan = summary["own_output_four_task_makespan_seconds"]
    per_agent = {}
    for agent in agents:
        mine = [r for r in rows if r["agent"] == agent]
        done = next(r["completed_ns"] for r in mine if r["turn"] is None)
        sums = {k: sum(r[f"{k}_s"] for r in mine if r["turn"] is not None) for k in ("prefill", "decode", "residual")}
        sums["client"] = sum(r["client_s"] for r in mine)
        sums["completion"] = (done - start) / 1e9
        q = [r["queue_s"] for r in mine if r["turn"] is not None]
        sums["queue"] = None if any(v is None for v in q) else sum(q)
        per_agent[agent] = sums
    mean = {k: sum(a[k] for a in per_agent.values()) / len(per_agent) for k in ("prefill", "decode", "residual", "client", "completion")}
    mean["tail"] = makespan - mean["completion"]
    queues = [a["queue"] for a in per_agent.values()]
    mean["queue"] = None if any(v is None for v in queues) else sum(queues) / len(queues)
    by_turn = {}
    for index in sorted({r["turn"] for r in turns}):
        sel = [r for r in turns if r["turn"] == index]
        by_turn[str(index)] = {k: median([r[k] for r in sel]) for k in
                               ("span_s", "prefill_s", "decode_s", "residual_s", "queue_s", "client_s",
                                "new_tokens", "cached_tokens", "completion_tokens", "semantic_events",
                                "semantic_ttft_s")}
    return {"schema": SCHEMA, "run": run_dir.name, "path": str(run_dir), "status": summary.get("status"),
            "engine": turns[0]["engine"], "makespan_s": makespan, "agents": len(agents),
            "stages": mean, "per_agent": per_agent, "by_turn": by_turn, "sse": sse_attribution(rows),
            "rows": [{k: v for k, v in r.items() if k != "semantic_events_ns"} for r in turns]}


def warm_prefill_model(rows):
    """OLS prefill_s ~ b0 + b1*new + b2*new*ctx over warm turns (turn >= 1, new >= 1), ctx = cached + new/2.

    The b2 term scales with attention work (new rows x attended context); b1 with per-row work
    (projections, MoE). Model-based and confounded by co-scheduled work: a fit, not a measurement.
    """
    pts = [(1.0, r["new_tokens"] / 1e3, r["new_tokens"] * (r["cached_tokens"] + r["new_tokens"] / 2) / 1e6,
            r["prefill_s"]) for r in rows if r.get("turn") and (r.get("new_tokens") or 0) >= 1]
    if len(pts) < 6:
        return None
    n = 3
    a = [[sum(p[i] * p[j] for p in pts) for j in range(n)] + [sum(p[i] * p[3] for p in pts)] for i in range(n)]
    for c in range(n):
        pivot = max(range(c, n), key=lambda r: abs(a[r][c]))
        if abs(a[pivot][c]) < 1e-12:
            return None
        a[c], a[pivot] = a[pivot], a[c]
        for r in range(n):
            if r != c:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    b = [a[i][n] / a[i][i] for i in range(n)]
    fit = [sum(bi * xi for bi, xi in zip(b, p[:3])) for p in pts]
    y = [p[3] for p in pts]
    ybar = sum(y) / len(y)
    ss_res = sum((yi - fi) ** 2 for yi, fi in zip(y, fit))
    ss_tot = sum((yi - ybar) ** 2 for yi in y)
    attention = sum(b[2] * p[2] for p in pts)
    return {"points": len(pts), "intercept_s": b[0], "per_1k_new_rows_s": b[1], "per_1m_row_ctx_s": b[2],
            "r2": 1 - ss_res / ss_tot if ss_tot else None,
            "attention_term_share": attention / sum(fit) if sum(fit) else None}


def _bootstrap(a, b, paired_blocks):
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    if paired_blocks:
        diffs = list(paired_blocks.values())
        for _ in range(BOOTSTRAP_SAMPLES):
            draws.append(median([rng.choice(diffs) for _ in diffs]))
    else:
        for _ in range(BOOTSTRAP_SAMPLES):
            draws.append(median([rng.choice(a) for _ in a]) - median([rng.choice(b) for _ in b]))
    draws.sort()
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]]


def _metric(p, stage):
    if stage == "makespan":
        return p["makespan_s"]
    if stage == "worst_slot_sse_gap_max_s":
        return p["sse"].get("worst_slot_gap_s")
    return p["stages"].get(stage)


def compare(arm, base):
    """Stage medians of arm and base, arm - base deltas with 95% bootstrap CIs and shares of the makespan delta.

    Blocks (`-wNN-`/`-bNN-` in run names) shared by both arms are resampled as paired differences;
    otherwise runs are resampled independently per arm.
    """
    out = {"arm_runs": len(arm), "base_runs": len(base), "stages": {}}
    blocks_a = {m.group(1): p for p in arm if (m := BLOCK.search(p["run"]))}
    blocks_b = {m.group(1): p for p in base if (m := BLOCK.search(p["run"]))}
    common = sorted(set(blocks_a) & set(blocks_b))
    out["pairing"] = "block" if len(common) >= 3 else "unpaired"
    d_makespan = None
    for stage in ("makespan",) + STAGES + ("queue", "worst_slot_sse_gap_max_s"):
        xa = [v for p in arm if (v := _metric(p, stage)) is not None]
        xb = [v for p in base if (v := _metric(p, stage)) is not None]
        if not xa or not xb:
            out["stages"][stage] = {"arm_median": median(xa), "base_median": median(xb)}
            continue
        paired = None
        if out["pairing"] == "block":
            paired = {k: _metric(blocks_a[k], stage) - _metric(blocks_b[k], stage) for k in common
                      if _metric(blocks_a[k], stage) is not None and _metric(blocks_b[k], stage) is not None}
        delta = median(list(paired.values())) if paired else median(xa) - median(xb)
        if stage == "makespan":
            d_makespan = delta
        row = {"arm_median": median(xa), "base_median": median(xb), "delta": delta,
               "delta_ci95": _bootstrap(xa, xb, paired)}
        if stage in STAGES and d_makespan:
            row["share_of_makespan_delta"] = delta / d_makespan
        out["stages"][stage] = row
    turns = sorted(set().union(*(p["by_turn"] for p in arm + base)), key=int)
    out["by_turn"] = {t: {k: {"arm": median([p["by_turn"].get(t, {}).get(k) for p in arm]),
                              "base": median([p["by_turn"].get(t, {}).get(k) for p in base])}
                          for k in ("span_s", "prefill_s", "decode_s", "residual_s", "queue_s", "new_tokens",
                                    "cached_tokens", "semantic_events", "semantic_ttft_s")}
                      for t in turns}
    return out


# ---- Layer 2: Metal System Trace ----------------------------------------------------------------------

def mst_busy(exec_xml, mapping_xml=None, process=None):
    """GPU-busy union intervals (trace ns) from sanitized xctrace exports."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import aggregate_decode_mst as mst
    paired, _, _ = mst.pair_dispatches(mst.parse_dispatches(Path(exec_xml).read_text()))
    if mapping_xml and process:
        paired, _, _ = mst.filter_process(paired, mst.parse_submission_metadata(Path(mapping_xml).read_text()), process)
    return [tuple(x) for x in mst.merge_intervals(paired)]


def merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        elif b > a:
            out.append([a, b])
    return out


def intersection_length(xs, ys, shift=0):
    """Overlap of two merged interval lists, xs shifted by `shift`; linear two-pointer walk."""
    i = j = total = 0
    while i < len(xs) and j < len(ys):
        a, b = xs[i][0] + shift, xs[i][1] + shift
        c, d = ys[j]
        total += max(0, min(b, d) - max(a, c))
        if b < d:
            i += 1
        else:
            j += 1
    return total


def align_offset(busy, spans, center, radius_ns=2_000_000_000, step_ns=1_000_000):
    """Offset (client ns - trace ns) minimizing GPU-busy time outside request spans; ties -> nearest center.

    A 10-step coarse grid over +-radius, then step_ns resolution around the coarse optimum.
    """
    xs, ys = merged(busy), merged(spans)
    busy_total = sum(b - a for a, b in xs)

    def search(mid, radius, step):
        best = None
        for k in range(-(radius // step), radius // step + 1):
            off = mid + k * step
            score = (busy_total - intersection_length(xs, ys, off), abs(off - center))
            if best is None or score < best[0]:
                best = (score, off)
        return best

    coarse = search(center, radius_ns, 10 * step_ns)
    (outside, _), offset = search(coarse[1], 10 * step_ns, step_ns)
    return offset, outside


def gpu_classes(profile, busy, offset):
    """Classes of the request window [first send, last terminal] on the client clock, seconds."""
    rows = profile["rows"]
    lo, hi = min(r["send_ns"] for r in rows), max(r["terminal_ns"] for r in rows)
    spans = merged([(r["send_ns"], r["terminal_ns"]) for r in rows])
    prefill = merged([(r["send_ns"], r["terminal_ns"] - int(r["decode_s"] * 1e9)) for r in rows])
    gpu = merged([(max(lo, a + offset), min(hi, b + offset)) for a, b in busy])
    in_flight = sum(b - a for a, b in spans)
    prefill_union = sum(b - a for a, b in prefill)
    busy_total = sum(b - a for a, b in gpu)
    busy_in_flight = intersection_length(gpu, spans)
    busy_in_prefill = intersection_length(gpu, prefill)
    return {"window_s": (hi - lo) / 1e9, "gpu_busy_s": busy_total / 1e9,
            "gpu_busy_outside_requests_s": (busy_total - busy_in_flight) / 1e9,
            "gpu_idle_in_flight_s": (in_flight - busy_in_flight) / 1e9,
            "no_request_s": (hi - lo - in_flight) / 1e9,
            "prefill_window_s": prefill_union / 1e9, "gpu_busy_in_prefill_window_s": busy_in_prefill / 1e9,
            "decode_only_window_s": (in_flight - prefill_union) / 1e9,
            "gpu_busy_decode_only_s": (busy_in_flight - busy_in_prefill) / 1e9}


# ---- Layer 3: hf2q category dumps ---------------------------------------------------------------------

DECODE_CLASSES = {"attention": ("attn_pre", "attention("), "moe": ("moe_",), "dense": ("dense_",),
                  "norm_embed": ("rms_norm", "embed")}
CATSPLIT_ROW = re.compile(r"^\[CATSPLIT\]\s+(\S.*?)\s+([0-9.]+)\s+[0-9.]+%\s+(\d+)\s+(\d+)\s*$")


def parse_catsplit(text):
    """hf2q worker-exit dumps -> {'decode': {class: ms}, 'prefill': {category: ms}} (absent parts omitted)."""
    out = {}
    decode = {}
    for line in text.splitlines():
        m = CATSPLIT_ROW.match(line.strip())
        if m and m.group(1) != "category":
            name, ms = m.group(1), float(m.group(2))
            cls = next((c for c, keys in DECODE_CLASSES.items() if any(name.startswith(k) for k in keys)), "other")
            decode[cls] = decode.get(cls, 0.0) + ms
        if line.startswith("[PREFILL_CATSPLIT_JSON] "):
            record = json.loads(line.split(" ", 1)[1])
            out["prefill"] = {k: v["gpu_ns"] / 1e6 for k, v in record["categories"].items()}
            out["prefill_forwards"] = record["forwards"]
    if decode:
        out["decode"] = decode
    for part in ("decode", "prefill"):
        if part in out:
            total = sum(out[part].values()) or 1.0
            out[f"{part}_share"] = {k: v / total for k, v in out[part].items()}
    return out


# ---- CLI ----------------------------------------------------------------------------------------------

def _runs(root, pattern):
    """Complete runs matching a glob relative to root (it may name subdirectories, e.g. 'l1/*-hf2q-*')."""
    runs = []
    for d in sorted(Path(root).glob(pattern)):
        if d.is_dir() and (d / "client" / "summary.json").is_file():
            if load_json(d / "client" / "summary.json").get("status") == SUCCESS:
                runs.append(d)
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="per-run profiles of RUN_DIR...")
    p.add_argument("runs", nargs="+")
    p.add_argument("--json")
    c = sub.add_parser("compare", help="arm vs base over complete runs under ROOT")
    c.add_argument("root")
    c.add_argument("--arm", required=True, help="NAME=GLOB of run directories relative to ROOT")
    c.add_argument("--base", required=True, help="NAME=GLOB relative to ROOT")
    c.add_argument("--json")
    m = sub.add_parser("mst", help="GPU classes of one run from sanitized xctrace exports")
    m.add_argument("run")
    m.add_argument("--exec-xml", required=True)
    m.add_argument("--mapping-xml")
    m.add_argument("--process")
    m.add_argument("--offset-ns", type=int, required=True, help="anchor: client CLOCK_MONOTONIC ns at trace t=0")
    m.add_argument("--radius-ms", type=int, default=2000)
    m.add_argument("--json")
    k = sub.add_parser("catsplit", help="hf2q category dumps from a profiled run's server.stderr")
    k.add_argument("stderr")
    k.add_argument("--json")
    args = ap.parse_args(argv)

    if args.cmd == "profile":
        result = [run_profile(r) for r in args.runs]
        for r in result:
            s = r["stages"]
            print(f"{r['run']}\t{r['engine']}\tmakespan {r['makespan_s']:.2f}\t" +
                  "\t".join(f"{k} {s[k]:.2f}" for k in STAGES))
    elif args.cmd == "compare":
        (an, ag), (bn, bg) = (x.split("=", 1) for x in (args.arm, args.base))
        arm, base = [run_profile(r) for r in _runs(args.root, ag)], [run_profile(r) for r in _runs(args.root, bg)]
        if not arm or not base:
            ap.error("no complete runs matched")
        result = {"schema": SCHEMA, "root": args.root, "arm": an, "arm_glob": ag, "base": bn, "base_glob": bg,
                  "comparison": compare(arm, base),
                  "warm_prefill_model": {an: warm_prefill_model([r for p in arm for r in p["rows"]]),
                                         bn: warm_prefill_model([r for p in base for r in p["rows"]])},
                  "sse": {an: [p["sse"] for p in arm], bn: [p["sse"] for p in base]}}
        cmp = result["comparison"]
        print(f"{an} ({cmp['arm_runs']} runs) vs {bn} ({cmp['base_runs']} runs), pairing {cmp['pairing']}")
        for stage, row in cmp["stages"].items():
            if "delta" in row:
                share = row.get("share_of_makespan_delta")
                print(f"  {stage:<26} {row['arm_median']:8.3f} {row['base_median']:8.3f}  delta {row['delta']:+.3f} "
                      f"[{row['delta_ci95'][0]:+.3f}, {row['delta_ci95'][1]:+.3f}]"
                      + ("" if share is None else f"  share {share:+.2f}"))
    elif args.cmd == "mst":
        profile = run_profile(args.run)
        radius = args.radius_ms * 1_000_000
        spans = [(r["send_ns"], r["terminal_ns"]) for r in profile["rows"]]
        lo, hi = min(s for s, _ in spans) - radius, max(e for _, e in spans) + radius
        # Model load and readiness traffic precede the run; align only on GPU work near the request window.
        busy = [(a, b) for a, b in mst_busy(args.exec_xml, args.mapping_xml, args.process)
                if b + args.offset_ns > lo and a + args.offset_ns < hi]
        offset, outside = align_offset(busy, spans, args.offset_ns, radius)
        result = {"schema": SCHEMA, "run": profile["run"], "offset_ns": offset, "anchor_ns": args.offset_ns,
                  "busy_intervals": len(busy), "busy_outside_spans_s": outside / 1e9,
                  "classes": gpu_classes(profile, busy, offset)}
        print(json.dumps(result["classes"], indent=1))
    else:
        result = parse_catsplit(Path(args.stderr).read_text(errors="replace"))
        print(json.dumps(result, indent=1))
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
