#!/usr/bin/env python3
"""tail.py FILE [--lanes N] [--split MARK] [--skip K]

Per-cycle quantities of an HF2Q_DFLASH2_CYCLE_TIMELINE file that stay meaningful when a cycle's
draft command is committed inside the previous cycle (prefetch):

  period      begin of this cycle -> begin of the next cycle (both N lanes, consecutive: the next
              begins within 2 ms of this one's end)
  tail        output head's GPU end -> GPU start of the draft command the next cycle uses
  gpu classes GPU ms of the draft the cycle used, the layer commands, the head command, the
              context commit and the replay
  busy/idle   union of GPU execution inside [this begin, next begin) and its complement

Draft commands are assigned to the cycle that follows their commit when they were committed after
the committing cycle's head (a prefetch), else to the cycle that committed them.
--split MARK: report cycles that carry host mark MARK and cycles that do not, and their difference
(the within-run block method); the first two cycles after each change of arm are dropped.
"""
import argparse, json, statistics, sys, math

def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.median(v) if v else float("nan")

def load(path):
    cycles = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        c = json.loads(line)
        if not c.get("finished", True):
            cycles.append(None); continue
        marks = c["marks"]
        m = {}
        for name, t in marks:
            m.setdefault(name, t)
        c["m"] = m
        c["names"] = {name for name, _ in marks}
        cycles.append(c)
    return cycles

def classify(c):
    """commands of a cycle record by class"""
    out = {"draft": [], "layers": [], "head": [], "replay": [], "ctx": [], "embed": [], "other": []}
    begin = c["m"]["begin"]
    for cmd in c["commands"]:
        if cmd["commit_begin"] < begin or not cmd["completed"]:
            continue
        label = cmd["label"] or ""
        if "dflash2_draft" in label: out["draft"].append(cmd)
        elif "joint_decode.layers" in label: out["layers"].append(cmd)
        elif "output_head" in label or label == "adr065.head": out["head"].append(cmd)
        elif "joint_verify.replay" in label: out["replay"].append(cmd)
        elif "embedding" in label: out["embed"].append(cmd)
        elif label == "" or "context" in label: out["ctx"].append(cmd)
        else: out["other"].append(cmd)
    return out

def reduce(cycles, lanes, skip):
    rows = []
    carried_draft = None  # a draft committed after the previous cycle's head
    seen = 0
    for i, c in enumerate(cycles):
        if c is None:
            carried_draft = None; continue
        k = classify(c)
        head = k["head"][-1] if k["head"] else None
        own = None; prefetched = None
        for d in k["draft"]:
            if head is not None and d["commit_begin"] > head["commit_begin"]:
                prefetched = d
            else:
                own = d
        used = own if own is not None else carried_draft
        used_prefetch = own is None and carried_draft is not None
        carried_draft = prefetched
        seen += 1
        if seen <= skip or head is None or used is None:
            continue
        nxt = cycles[i + 1] if i + 1 < len(cycles) else None
        begin, end = c["m"]["begin"], c["m"]["end"]
        row = {"lanes": c["active_lanes"], "names": c["names"], "wall": (end - begin) * 1e3,
               "draft_gpu": (used["gpu_end"] - used["gpu_start"]) * 1e3,
               "draft_prefetched": used_prefetch,
               "layers_gpu": sum(x["gpu_end"] - x["gpu_start"] for x in k["layers"]) * 1e3,
               "layers_span": (k["layers"][-1]["gpu_end"] - k["layers"][0]["gpu_start"]) * 1e3 if k["layers"] else None,
               "head_gpu": (head["gpu_end"] - head["gpu_start"]) * 1e3,
               "ctx_gpu": sum(x["gpu_end"] - x["gpu_start"] for x in k["ctx"]) * 1e3,
               "replay_gpu": sum(x["gpu_end"] - x["gpu_start"] for x in k["replay"]) * 1e3,
               "head_end_to_end": (end - head["gpu_end"]) * 1e3,
               "accepted": c["accepted_rows"], "gpu_acceptance": c["gpu_acceptance"],
               "draft_to_layers_idle": (k["layers"][0]["gpu_start"] - used["gpu_end"]) * 1e3 if k["layers"] else None,
               "period": None, "tail": None, "idle": None}
        if nxt is not None and nxt["active_lanes"] == c["active_lanes"] and 0 <= nxt["m"]["begin"] - end < 2e-3:
            nk = classify(nxt)
            nhead = nk["head"][-1] if nk["head"] else None
            ndraft = prefetched
            if ndraft is None:
                for d in nk["draft"]:
                    if nhead is None or d["commit_begin"] < nhead["commit_begin"]:
                        ndraft = d
            row["period"] = (nxt["m"]["begin"] - begin) * 1e3
            if ndraft is not None:
                row["tail"] = (ndraft["gpu_start"] - head["gpu_end"]) * 1e3
            # GPU busy union over [begin, next begin)
            lo, hi = begin, nxt["m"]["begin"]
            iv = []
            for cmd in c["commands"] + nxt["commands"] + ([used] if used_prefetch else []):
                if cmd["completed"]:
                    a, b = max(cmd["gpu_start"], lo), min(cmd["gpu_end"], hi)
                    if b > a: iv.append((a, b))
            iv.sort(); busy = 0.0; cur = lo
            for a, b in iv:
                a = max(a, cur)
                if b > a: busy += b - a; cur = b
            row["idle"] = (hi - lo - busy) * 1e3
        row["replay_dispatches"] = sum(x.get("own_dispatches", 0) for x in k["replay"])
        row["index"] = i
        rows.append(row)
    return [r for r in rows if r["lanes"] == lanes]

def regress(xs, ys):
    n = len(xs)
    if n < 3: return float("nan"), float("nan"), n
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0: return float("nan"), my, n
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return slope, my - slope * mx, n

KEYS = ["period", "wall", "tail", "head_end_to_end", "idle", "draft_gpu", "draft_to_layers_idle", "layers_gpu",
        "layers_span", "head_gpu", "ctx_gpu", "replay_gpu", "accepted"]

def table(rows):
    return {k: med([r[k] for r in rows]) for k in KEYS}

def se(v):
    v = [x for x in v if x is not None]
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("file"); p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--split"); p.add_argument("--skip", type=int, default=3)
    p.add_argument("--json"); p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    rows = reduce(load(a.file), a.lanes, a.skip)
    if a.replay:
        # What a replay costs the draft it runs beside: cycle N's replay GPU ms against the GPU ms
        # of the prefetched draft cycle N+1 uses and against cycle N's period, consecutive cycles.
        by_index = {r["index"]: r for r in rows}
        pairs = [(r, by_index[r["index"] + 1]) for r in rows if r["index"] + 1 in by_index]
        pairs = [(r, n) for r, n in pairs if r["period"] is not None and r["tail"] is not None
                 and (a.split is None or a.split in r["names"])]
        print(f"{a.file}: {a.lanes} lanes, {len(pairs)} consecutive cycle pairs"
              f"{' carrying mark ' + a.split if a.split else ''}, {sum(1 for _, n in pairs if n['draft_prefetched'])} with a prefetched draft")
        xs = [r["replay_gpu"] for r, _ in pairs]
        for name, ys in (("next draft GPU ms", [n["draft_gpu"] for _, n in pairs]),
                         ("period ms", [r["period"] for r, _ in pairs]),
                         ("tail ms", [r["tail"] for r, _ in pairs if r["tail"] is not None])):
            if len(ys) != len(xs): continue
            slope, intercept, n = regress(xs, ys)
            print(f"  {name:20s} = {intercept:7.3f} + {slope:6.3f} x replay GPU ms   ({n} cycle pairs, replay median {med(xs):.3f}, "
                  f"without replay: {sum(1 for x in xs if x == 0)})")
        none = [n["draft_gpu"] for r, n in pairs if r["replay_gpu"] == 0]
        some = [n["draft_gpu"] for r, n in pairs if r["replay_gpu"] > 0]
        if none and some:
            print(f"  next draft GPU ms: {med(none):.3f} behind no replay ({len(none)}), {med(some):.3f} behind a replay ({len(some)})")
        return
    if not a.split:
        t = table(rows)
        print(f"{a.file}: {len(rows)} cycles of {a.lanes} lanes, {sum(1 for r in rows if r['period'] is not None)} with a successor, "
              f"{sum(1 for r in rows if r['draft_prefetched'])} prefetched drafts, {sum(1 for r in rows if r['gpu_acceptance'])} gpu-acceptance")
        for k in KEYS: print(f"  {k:22s} {t[k]:8.3f}")
        if a.json: json.dump(t, open(a.json, "w"), indent=1)
        return
    # block method: drop two cycles after each arm change
    arms = {True: [], False: []}
    last = None; since = 0
    for r in rows:
        on = a.split in r["names"]
        if on != last: since = 0; last = on
        since += 1
        if since > 2: arms[on].append(r)
    print(f"{a.file}: {a.lanes} lanes, split on mark {a.split!r}: on {len(arms[True])} cycles, off {len(arms[False])} cycles")
    print(f"  {'':22s} {'on':>8s} {'off':>8s} {'on-off (median)':>16s} {'mean diff':>10s} {'se':>7s}")
    for k in KEYS:
        on = [r[k] for r in arms[True] if r[k] is not None]; off = [r[k] for r in arms[False] if r[k] is not None]
        if not on or not off: continue
        d = statistics.mean(on) - statistics.mean(off)
        s = math.sqrt(se(on) ** 2 + se(off) ** 2) if len(on) > 1 and len(off) > 1 else float("nan")
        print(f"  {k:22s} {med(on):8.3f} {med(off):8.3f} {med(on) - med(off):16.3f} {d:10.3f} {s:7.3f}")

if __name__ == "__main__":
    main()
