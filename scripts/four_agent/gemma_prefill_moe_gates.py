#!/usr/bin/env python3
"""ADR-063 Q4-GEMMA-PREFILL-MOE (#690): speed and fairness gates of the same-binary A/B from run receipts.

Read-only; launches nothing. Implements the gates frozen in
docs/research/adr063-q4-gemma-prefill-moe-design.md §7.1 and §7.3 over settled Q5 run directories
(server.stderr, client/summary.json) given in execution order:

  gemma_prefill_moe_gates.py gates --run A=DIR --run B=DIR --run B=DIR --run A=DIR ... [--s-meas S]

Consecutive runs form the ABBA blocks (one A and one B run each).

S2 (primary): per-run sum of the unprofiled warm-forward walls (`Batched prefill complete` of every
  forward whose header carries `live=`), over structure-matched blocks. A run's structure is the
  ordered list of its warm transactions, each (lanes, per-lane advanced rows). Runs off the campaign's
  modal structure are excluded and listed. Pass: >= 6 matched blocks, paired median delta (B - A)
  <= -0.05 s and 95% bootstrap CI upper bound < 0. With fewer than 6 matched blocks the per-transaction
  substitution applies: every modal transaction is matched by (lanes, rows) occurrence across the two
  runs of a block and the sum is taken over the matched transactions only (recorded as substituted).
S4 (makespan): client `own_output_four_task_makespan_seconds` over every block, split or not.
  Pass: median delta <= 0 and 95% CI upper bound <= +0.05 s.
S3 (category, optional `--catsplit-run A=DIR ...`): `[PREFILL_CATSPLIT_JSON]` medians per arm. Pass:
  MoE + layer tail falls by >= 50% of --s-meas, the other categories change by < 3%.
Fairness (§7.3): median worst-slot TTFT p95 and SSE-gap max, ratio B / A <= 1.025 (NEUTRAL_RATIO).
"""
import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from q3_metrics import slot_metrics, worst
from q3_stats import NEUTRAL_RATIO

SCHEMA = "hf2q.adr063.q4-gemma-prefill-moe.gates.v1"
S2_MIN_BLOCKS = 6
S2_MAX_EXTRA_BLOCKS = 6
S2_SAVING_S = 0.05
S4_UPPER_S = 0.05
S3_OTHER_BAND = 0.03
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 6900

HEADER = re.compile(r"^Batched prefill: .*seq_len=(\d+)")
COMPLETE = re.compile(r"^Batched prefill complete: (\d+) tokens in ([0-9.]+) ms")
JOINT = re.compile(r'transaction="joint warm" requests=(\d+) advanced_tokens=(\d+)')
LANE = re.compile(r"bounded prefill transaction complete slot=(\d+) advanced_tokens=(\d+)")


def warm_transactions(text):
    """Ordered warm transactions of one server log: dicts with lanes, rows (per lane, sorted), wall_ms."""
    out, current, pending_lanes = [], None, 0
    for line in text.splitlines():
        header = HEADER.match(line)
        if header:
            current = {"warm": "live=" in line, "seq_len": int(header.group(1))}
            pending_lanes = 0
            continue
        complete = COMPLETE.match(line)
        if complete and current is not None:
            if current["warm"]:
                rows = int(complete.group(1))
                if rows != current["seq_len"]:
                    raise ValueError(f"forward header seq_len {current['seq_len']} != completed rows {rows}")
                out.append({"lanes": 1, "rows": [rows], "wall_ms": float(complete.group(2)), "joint": False})
            current = None
            continue
        joint = JOINT.search(line)
        if joint and out:
            last = out[-1]
            if last["joint"] or last["rows"] != [int(joint.group(2))]:
                raise ValueError("joint warm transaction without its own warm forward")
            last.update(lanes=int(joint.group(1)), joint=True, rows=[], total=int(joint.group(2)))
            pending_lanes = last["lanes"]
            continue
        lane = LANE.search(line)
        if lane and pending_lanes and out:
            out[-1]["rows"].append(int(lane.group(2)))
            pending_lanes -= 1
            if not pending_lanes:
                out[-1]["rows"].sort()
                if sum(out[-1]["rows"]) != out[-1]["total"]:
                    raise ValueError("lane rows do not sum to the joint transaction rows")
    for transaction in out:
        if transaction["joint"] and len(transaction["rows"]) != transaction["lanes"]:
            raise ValueError(f"joint warm transaction lists {len(transaction['rows'])} of "
                             f"{transaction['lanes']} lane completions")
    return out


def structure(transactions):
    return tuple((t["lanes"], tuple(t["rows"])) for t in transactions)


def load_run(arm, run_dir):
    run_dir = Path(run_dir)
    transactions = warm_transactions((run_dir / "server.stderr").read_text(errors="replace"))
    summary = json.loads((run_dir / "client" / "summary.json").read_text())
    agents = summary.get("agents") or {}
    slots = {name: slot_metrics(agent) for name, agent in sorted(agents.items())}
    return {
        "arm": arm, "dir": str(run_dir), "transactions": transactions,
        "structure": structure(transactions),
        "warm_wall_s": sum(t["wall_ms"] for t in transactions) / 1000,
        "makespan_s": summary.get("own_output_four_task_makespan_seconds"),
        "worst_ttft_p95_s": worst(slots, "ttft_p95_s"),
        "worst_sse_gap_max_s": worst(slots, "sse_gap_max_s"),
    }


def blocks_of(runs):
    if len(runs) % 2:
        raise ValueError("runs must form complete blocks (pairs in execution order)")
    blocks = []
    for index in range(0, len(runs), 2):
        pair = {run["arm"]: run for run in runs[index:index + 2]}
        if set(pair) != {"A", "B"}:
            raise ValueError(f"block {index // 2 + 1} must hold one A and one B run")
        blocks.append(pair)
    return blocks


def median_ci(deltas, seed=BOOTSTRAP_SEED):
    """Median of paired deltas with a 95% percentile-bootstrap interval of the median."""
    if not deltas:
        return None, None
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(deltas, k=len(deltas))) for _ in range(BOOTSTRAP_SAMPLES))
    return statistics.median(deltas), [medians[int(0.025 * BOOTSTRAP_SAMPLES)],
                                       medians[int(0.975 * BOOTSTRAP_SAMPLES) - 1]]


def matched_wall_s(run, modal):
    """Per-transaction substitution: walls of the run's transactions that match a modal (lanes, rows)
    occurrence, keyed by (element, occurrence)."""
    seen, walls = Counter(), {}
    for t in run["transactions"]:
        element = (t["lanes"], tuple(t["rows"]))
        key = (element, seen[element])
        seen[element] += 1
        walls[key] = t["wall_ms"] / 1000
    wanted, keys = Counter(), []
    for element in modal:
        keys.append((element, wanted[element]))
        wanted[element] += 1
    return {key: walls[key] for key in keys if key in walls}


def s2(blocks):
    modal = Counter(run["structure"] for block in blocks for run in block.values()).most_common(1)[0][0]
    excluded = [{"arm": run["arm"], "dir": run["dir"], "structure": run["structure"]}
                for block in blocks for run in block.values() if run["structure"] != modal]
    matched = [b for b in blocks if b["A"]["structure"] == modal and b["B"]["structure"] == modal]
    if len(matched) >= S2_MIN_BLOCKS:
        deltas = [b["B"]["warm_wall_s"] - b["A"]["warm_wall_s"] for b in matched]
        substituted = False
    else:
        deltas = []
        for b in blocks:
            a, bb = matched_wall_s(b["A"], modal), matched_wall_s(b["B"], modal)
            common = sorted(set(a) & set(bb))
            if common:
                deltas.append(sum(bb[k] for k in common) - sum(a[k] for k in common))
        substituted = True
    median, ci = median_ci(deltas)
    passed = (median is not None and ci is not None and median <= -S2_SAVING_S and ci[1] < 0
              and (substituted or len(matched) >= S2_MIN_BLOCKS))
    return {"modal_structure": modal, "matched_blocks": len(matched), "excluded_runs": excluded,
            "per_transaction_substitution": substituted, "deltas_s": deltas, "median_delta_s": median,
            "ci95_s": ci, "pass": passed,
            "note": None if len(blocks) - len(matched) <= S2_MAX_EXTRA_BLOCKS else
            "more replacement blocks than the frozen cap of 6"}


def s4(blocks):
    deltas = [b["B"]["makespan_s"] - b["A"]["makespan_s"] for b in blocks]
    median, ci = median_ci(deltas)
    return {"blocks": len(deltas), "deltas_s": deltas, "median_delta_s": median, "ci95_s": ci,
            "pass": median is not None and median <= 0 and ci[1] <= S4_UPPER_S}


def fairness(blocks):
    out = {}
    for key in ("worst_ttft_p95_s", "worst_sse_gap_max_s"):
        a = [b["A"][key] for b in blocks if b["A"][key] is not None]
        bb = [b["B"][key] for b in blocks if b["B"][key] is not None]
        ratio = statistics.median(bb) / statistics.median(a) if a and bb else None
        out[key] = {"ratio_b_over_a": ratio, "pass": ratio is not None and ratio <= NEUTRAL_RATIO}
    return out


def parse_catsplit(text):
    last = None
    for line in text.splitlines():
        if line.startswith("[PREFILL_CATSPLIT_JSON] "):
            last = json.loads(line.split(" ", 1)[1])
    if last is None:
        raise ValueError("no [PREFILL_CATSPLIT_JSON] line")
    return {name: value["gpu_ns"] / 1e9 for name, value in last["categories"].items()}


def s3(runs, s_meas):
    by_arm = {"A": [], "B": []}
    for arm, run_dir in runs:
        by_arm[arm].append(parse_catsplit((Path(run_dir) / "server.stderr").read_text(errors="replace")))
    if not by_arm["A"] or not by_arm["B"]:
        raise ValueError("S3 needs catsplit runs of both arms")
    med = {arm: {c: statistics.median(r[c] for r in rs) for c in rs[0]} for arm, rs in by_arm.items()}
    reduction = med["A"]["moe"] - med["B"]["moe"]
    others = {c: (med["B"][c] - med["A"][c]) / med["A"][c] for c in med["A"] if c != "moe"}
    return {"median_gpu_s": med, "moe_reduction_s": reduction, "s_meas_s": s_meas,
            "other_relative_change": others,
            "pass": s_meas is not None and reduction >= 0.5 * s_meas
            and all(abs(v) < S3_OTHER_BAND for v in others.values())}


def arm_dir(value):
    arm, _, path = value.partition("=")
    if arm not in ("A", "B") or not path:
        raise argparse.ArgumentTypeError("expected A=DIR or B=DIR")
    return arm, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    gates = sub.add_parser("gates")
    gates.add_argument("--run", type=arm_dir, action="append", required=True,
                       help="A=DIR or B=DIR, in execution order")
    gates.add_argument("--catsplit-run", type=arm_dir, action="append", default=[])
    gates.add_argument("--s-meas", type=float, help="M0c S_meas (or S_pred(P1) for M3), seconds per run")
    args = parser.parse_args(argv)
    runs = [load_run(arm, path) for arm, path in args.run]
    blocks = blocks_of(runs)
    report = {"schema": SCHEMA, "runs": [{k: r[k] for k in ("arm", "dir", "warm_wall_s", "makespan_s")}
                                          for r in runs],
              "S2": s2(blocks), "S4": s4(blocks), "fairness": fairness(blocks)}
    if args.catsplit_run:
        report["S3"] = s3(args.catsplit_run, args.s_meas)
    json.dump(report, sys.stdout, indent=1, default=list)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
