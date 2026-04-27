#!/usr/bin/env python3
# scripts/aggregate_decode.py
#
# ADR-015 Wave 2b hard gate #3 + #4 + #5: corrected aggregation harness
# for xctrace TimeProfiler exports across N cold-SoC decode trials.
#
# Replaces the working-copy script at
#   /tmp/cfa-adr015-wave2a-p3a-prime/aggregate_decode.py
# (volatile; not committable from /tmp; gone from disk after reboot).  The
# original had two bugs documented in ADR-015 Codex review (2026-04-27):
#
#   AF3 — `aggregate_decode.py:129-132` summed inclusive ms across
#         multiple needle patterns per hypothesis.  Because xctrace
#         inclusive frames nest (a `build_dense_ffn` frame contains an
#         `alloc proj dst` child frame), summing them double-counted
#         the inner.  This rewrite reports ONE canonical frame per
#         hypothesis and lists named subcomponents *separately*.
#
#   AF4 — `aggregate.py:160-168` rank-stability section stored frames
#         as tuples then called `.split()` on the tuples.  This rewrite
#         keeps frame names as strings end-to-end and uses a typed
#         dataclass for trial state.
#
# Q1 — the original used 3-trial *mean*; trial-3 was a 1.8× outlier on
#      the Wave 2a run, inflating the rank-1 figure by 21 %.  This
#      rewrite defaults to 5-trial *median* (configurable).  Without
#      discarding outlier trials silently — discarding is selection
#      bias — the median absorbs them.
#
# Usage
# -----
#   scripts/aggregate_decode.py \
#       --trials /path/to/topcalls-1-*.txt /path/to/topcalls-2-*.txt ... \
#       --tokens-per-trial 64 \
#       --hypothesis-config scripts/aggregate_hypotheses.json \
#       --output-md /tmp/wave2b-decode-aggregate.md
#
#   # Generic top-N frame report (no hypothesis config):
#   scripts/aggregate_decode.py --trials topcalls-*.txt --top-n 20
#
# The hypothesis-config JSON lists canonical frame regexes per
# hypothesis ID.  Each hypothesis declares ONE primary frame; named
# subcomponents are reported in their own column without inflating the
# primary's number.  See scripts/aggregate_hypotheses.json for the
# ADR-015 H1..H5 register.
#
# xctrace export format
# ---------------------
# The script accepts the XML produced by:
#   xctrace export --input <trace> --xpath \
#     '/trace-toc/run[1]/data/table[@schema="time-profile"]'
#
# The schema yields one <row> per sampled stack.  Each row carries the
# stack frame names plus an inclusive sample weight.  We aggregate
# inclusive-ms-per-frame by walking all <row> elements and summing the
# `sample-time` weight for every frame named in any backtrace step
# whose immediate-self frame matches the canonical regex.
#
# Robustness: the parser treats unknown attribute names defensively.  If
# the xctrace XML format drifts, the script logs the unmatched schema
# and exits non-zero rather than producing wrong numbers.
#
# Note on backtrace double-count: this script accumulates *immediate-
# leaf* time (the most-specific frame in each sample's backtrace) per
# frame name.  Inclusive time can be derived by walking the parent
# chain — but the canonical-frame design avoids needing that, since
# ONE primary frame's leaf-time is the whole hypothesis's number.

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrialFrames:
    """Per-trial leaf-time accumulator for one xctrace export.

    `leaf_ms_by_frame[frame_name]` = total ms accumulated as the
    immediate-leaf frame across all samples in this trial.
    """

    trial_id: int
    source: Path
    samples: int = 0
    total_ms: float = 0.0
    leaf_ms_by_frame: dict[str, float] = field(default_factory=lambda: defaultdict(float))


@dataclass
class HypothesisSpec:
    """One hypothesis under test (ADR-015 H1..H5 + Wave 2b additions).

    `canonical_frame_regex` is the SINGLE frame whose leaf-ms is the
    hypothesis's primary number.  `subcomponent_regexes` are reported
    side-by-side without being summed into the primary.
    """

    id: str
    description: str
    canonical_frame_regex: str
    subcomponent_regexes: dict[str, str] = field(default_factory=dict)
    static_estimate_us_per_token: float | None = None


def parse_xctrace_topcalls(
    path: Path,
    trial_id: int,
    decode_filter_re: re.Pattern[str] | None = None,
) -> TrialFrames:
    """Walk the xctrace TimeProfiler XML export and accumulate
    INCLUSIVE ms per frame name across all sampled stacks.

    xctrace XML uses an id/ref dictionary pattern: any element with
    `id="N"` defines a value (a weight, a frame name, etc.); subsequent
    occurrences appear as `<tag ref="N"/>` and inherit that value.
    Refs only point backwards within the document, so a single forward
    pass that records definitions as they are encountered resolves all
    refs lazily.

    Inclusive aggregation: for each row's backtrace, the row's weight
    is added to every frame in the backtrace (the frame plus all its
    callers).  This is the standard "inclusive time" aggregation
    Instruments displays — the canonical-frame design relies on it
    because the canonical frame for a hypothesis is typically a
    high-level Rust function that contains many leaf samples.

    Weight is in nanoseconds (xctrace text content; the `fmt`
    attribute is a display string like "1.00 ms").  Converted to ms
    (÷ 1_000_000) at accumulation time.

    Leaf frame note: in xctrace's <tagged-backtrace>, frames appear in
    LEAF→ROOT order — the FIRST `<frame>` is the most-specific (the
    deepest active call site at sample time), the LAST is `start` /
    `dyld_start`.  We do not single out the leaf for canonical
    aggregation — every frame in the backtrace receives the weight.

    Decode-filter: if `decode_filter_re` is given, only samples whose
    backtrace contains a frame matching the regex are counted.  This
    isolates the decode loop from model-load / tokenizer / cleanup
    samples that share the trace.  Wave 2a §P3a' used this pattern with
    `forward_gpu_greedy` for qwen35; gemma uses `forward_decode`.
    """
    frames = TrialFrames(trial_id=trial_id, source=path)
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        sys.stderr.write(f"WARN: trial {trial_id} ({path.name}): XML parse error: {e}\n")
        return frames

    root = tree.getroot()

    # ID→name dictionary for frames; ID→ns dictionary for weights.
    # xctrace defines each value once (with id=N) then references it
    # via ref=N in subsequent occurrences.
    frame_name_by_id: dict[str, str] = {}
    weight_ns_by_id: dict[str, int] = {}

    def resolve_weight(elem: ET.Element) -> int | None:
        rid = elem.get("ref")
        if rid is not None:
            return weight_ns_by_id.get(rid)
        eid = elem.get("id")
        text = (elem.text or "").strip()
        if not text:
            return None
        try:
            ns = int(text)
        except ValueError:
            return None
        if eid is not None:
            weight_ns_by_id[eid] = ns
        return ns

    def resolve_frame(elem: ET.Element) -> str | None:
        rid = elem.get("ref")
        if rid is not None:
            return frame_name_by_id.get(rid)
        eid = elem.get("id")
        name = elem.get("name")
        if name and eid is not None:
            frame_name_by_id[eid] = name
        return name

    for row in root.iter("row"):
        weight_elem = row.find("weight")
        if weight_elem is None:
            continue
        ns = resolve_weight(weight_elem)
        if ns is None:
            continue
        ms = ns / 1_000_000.0

        backtrace = row.find("tagged-backtrace")
        if backtrace is None:
            backtrace = row.find("backtrace")
        if backtrace is None:
            continue
        # `backtrace` is now a non-None Element; safe to iter().

        # First pass: resolve all frame names so the decode-filter can
        # match on any of them.  The aggregation pass below dedupes
        # within the row.
        names: list[str] = []
        for frame_elem in backtrace.iter("frame"):
            name = resolve_frame(frame_elem)
            if name:
                names.append(name)

        if decode_filter_re is not None:
            if not any(decode_filter_re.search(n) for n in names):
                # Sample is from outside the decode loop (model load,
                # tokenizer init, cleanup).  Skip — not part of the
                # per-token aggregate.
                continue

        frames.samples += 1
        frames.total_ms += ms

        seen_in_row: set[str] = set()
        for name in names:
            if name in seen_in_row:
                continue
            seen_in_row.add(name)
            frames.leaf_ms_by_frame[name] += ms

    return frames


def aggregate_canonical(
    trials: list[TrialFrames],
    hypothesis: HypothesisSpec,
) -> dict[str, object]:
    """For one hypothesis, find its canonical frame in each trial,
    record the per-trial leaf-ms, then summarize via median + min/max.

    No summing across needles — that was AF3.  Subcomponents are
    reported in their own dict alongside the primary."""
    canon_re = re.compile(hypothesis.canonical_frame_regex)
    sub_res = {
        name: re.compile(pat) for name, pat in hypothesis.subcomponent_regexes.items()
    }

    per_trial_primary_ms: list[float] = []
    per_trial_primary_frame: list[str | None] = []
    per_trial_subs: dict[str, list[float]] = {n: [] for n in sub_res}

    for trial in trials:
        primary_match = _largest_match(trial.leaf_ms_by_frame, canon_re)
        if primary_match is None:
            per_trial_primary_ms.append(0.0)
            per_trial_primary_frame.append(None)
        else:
            frame_name, ms = primary_match
            per_trial_primary_ms.append(ms)
            per_trial_primary_frame.append(frame_name)

        for sub_name, sub_re in sub_res.items():
            sub_match = _largest_match(trial.leaf_ms_by_frame, sub_re)
            per_trial_subs[sub_name].append(0.0 if sub_match is None else sub_match[1])

    out = {
        "id": hypothesis.id,
        "description": hypothesis.description,
        "canonical_frame_regex": hypothesis.canonical_frame_regex,
        "static_estimate_us_per_token": hypothesis.static_estimate_us_per_token,
        "per_trial_primary_ms": per_trial_primary_ms,
        "per_trial_primary_frame": per_trial_primary_frame,
        "primary_median_ms": _median(per_trial_primary_ms),
        "primary_min_ms": min(per_trial_primary_ms) if per_trial_primary_ms else 0.0,
        "primary_max_ms": max(per_trial_primary_ms) if per_trial_primary_ms else 0.0,
        "subcomponents": {
            name: {
                "per_trial_ms": vals,
                "median_ms": _median(vals),
            }
            for name, vals in per_trial_subs.items()
        },
    }
    return out


def _largest_match(
    frames: dict[str, float], regex: re.Pattern[str]
) -> tuple[str, float] | None:
    """Find the matching frame with the largest leaf-ms.  Returns
    (frame_name, ms) or None."""
    best: tuple[str, float] | None = None
    for frame, ms in frames.items():
        if regex.search(frame):
            if best is None or ms > best[1]:
                best = (frame, ms)
    return best


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return statistics.median(xs)


def rank_stability(
    trials: list[TrialFrames],
    top_n: int = 10,
) -> dict[str, object]:
    """Compute rank stability across trials.

    For each trial, take the top-N frames by leaf-ms.  Report:
      - The set of frames in the top-N of EVERY trial (rank-stable set).
      - For each rank-stable frame, its rank in each trial.

    Frame names are kept as strings throughout — fixes AF4 (the original
    stored tuples and called `.split()` on them).
    """
    per_trial_top: list[list[tuple[str, float]]] = []
    for trial in trials:
        sorted_frames = sorted(trial.leaf_ms_by_frame.items(), key=lambda kv: -kv[1])
        per_trial_top.append(sorted_frames[:top_n])

    # Frames present in EVERY trial's top-N.
    common: set[str] = set()
    if per_trial_top:
        common = {name for name, _ in per_trial_top[0]}
        for top in per_trial_top[1:]:
            common &= {name for name, _ in top}

    rank_per_trial: dict[str, list[int]] = {}
    for frame in common:
        ranks: list[int] = []
        for top in per_trial_top:
            for i, (name, _) in enumerate(top):
                if name == frame:
                    ranks.append(i + 1)
                    break
            else:
                ranks.append(-1)  # not present
        rank_per_trial[frame] = ranks

    return {
        "top_n": top_n,
        "n_trials": len(trials),
        "common_frames": sorted(common),
        "rank_per_trial": rank_per_trial,
    }


def render_markdown(
    trials: list[TrialFrames],
    hypotheses: list[dict[str, object]],
    rank_stability_data: dict[str, object],
    tokens_per_trial: int,
) -> str:
    lines: list[str] = []
    lines.append("# decode-trace aggregation\n")
    lines.append(f"- trials: {len(trials)}")
    lines.append(f"- tokens per trial: {tokens_per_trial}")
    lines.append(f"- aggregator: 5-trial median (configurable)\n")

    lines.append("## per-trial totals\n")
    lines.append("| trial | source | samples | total_ms |")
    lines.append("|---:|---|---:|---:|")
    for t in trials:
        lines.append(
            f"| {t.trial_id} | {t.source.name} | {t.samples} | {t.total_ms:.1f} |"
        )
    lines.append("")

    if hypotheses:
        lines.append("## hypothesis register — per-hypothesis canonical frame\n")
        lines.append(
            "| ID | canonical frame | static est µs/token | "
            "median µs/token | min | max | per-trial µs/token |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for h in hypotheses:
            ms_to_us_per_token = 1000.0 / max(tokens_per_trial, 1)
            median_us = h["primary_median_ms"] * ms_to_us_per_token  # type: ignore[operator]
            min_us = h["primary_min_ms"] * ms_to_us_per_token  # type: ignore[operator]
            max_us = h["primary_max_ms"] * ms_to_us_per_token  # type: ignore[operator]
            per_trial = ", ".join(
                f"{v * ms_to_us_per_token:.1f}"
                for v in h["per_trial_primary_ms"]  # type: ignore[index]
            )
            static = h.get("static_estimate_us_per_token")
            static_str = f"{static}" if static is not None else "—"
            lines.append(
                f"| {h['id']} | `{h['canonical_frame_regex']}` | {static_str} | "
                f"{median_us:.1f} | {min_us:.1f} | {max_us:.1f} | {per_trial} |"
            )
        lines.append("")

        lines.append("### subcomponents (named, side-by-side; not summed into primary)\n")
        any_subs = False
        for h in hypotheses:
            sub = h.get("subcomponents", {}) or {}
            if not sub:
                continue
            any_subs = True
            lines.append(f"#### {h['id']}\n")
            lines.append("| subcomponent | median ms | per-trial ms |")
            lines.append("|---|---:|---|")
            for name, data in sub.items():  # type: ignore[union-attr]
                per_trial = ", ".join(f"{v:.1f}" for v in data["per_trial_ms"])
                lines.append(f"| {name} | {data['median_ms']:.1f} | {per_trial} |")
            lines.append("")
        if not any_subs:
            lines.append("(no subcomponents declared)\n")

    lines.append("## rank stability — frames in top-N of EVERY trial\n")
    lines.append(f"top_n = {rank_stability_data['top_n']}, "
                 f"n_trials = {rank_stability_data['n_trials']}\n")
    common = rank_stability_data["common_frames"]
    if not common:
        lines.append("(no frame ranked in top-N of every trial — investigate noise)\n")
    else:
        lines.append("| frame | per-trial ranks |")
        lines.append("|---|---|")
        for frame in common:  # type: ignore[union-attr]
            ranks = rank_stability_data["rank_per_trial"][frame]  # type: ignore[index]
            ranks_str = ", ".join(str(r) for r in ranks)
            lines.append(f"| `{frame}` | {ranks_str} |")
        lines.append("")

    return "\n".join(lines)


def load_hypothesis_config(path: Path | None) -> list[HypothesisSpec]:
    if path is None:
        return []
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        sys.stderr.write(f"ERROR: hypothesis config not found: {path}\n")
        sys.exit(2)
    out: list[HypothesisSpec] = []
    for entry in data.get("hypotheses", []):
        out.append(
            HypothesisSpec(
                id=entry["id"],
                description=entry.get("description", ""),
                canonical_frame_regex=entry["canonical_frame_regex"],
                subcomponent_regexes=entry.get("subcomponent_regexes", {}),
                static_estimate_us_per_token=entry.get("static_estimate_us_per_token"),
            )
        )
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate xctrace TimeProfiler exports across N "
        "cold-SoC trials with corrected canonical-frame "
        "methodology (ADR-015 Wave 2b)."
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        type=Path,
        required=True,
        help="One or more xctrace export files (XML).  5 trials recommended "
        "for median methodology.",
    )
    parser.add_argument(
        "--tokens-per-trial",
        type=int,
        default=64,
        help="Decode tokens per trial.  Used to convert ms→µs/token.",
    )
    parser.add_argument(
        "--hypothesis-config",
        type=Path,
        default=None,
        help="JSON file with hypothesis register (canonical frame regex "
        "per hypothesis ID).  See scripts/aggregate_hypotheses.json.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N frames to consider for rank-stability analysis.",
    )
    parser.add_argument(
        "--decode-filter",
        type=str,
        default=None,
        help="Regex; only samples whose backtrace contains a matching "
        "frame are counted.  Use to isolate decode-loop samples from "
        "model-load / tokenizer init / cleanup samples that share the "
        "trace.  qwen35: 'forward_gpu_greedy'.  gemma: 'forward_decode' "
        "(but NOT forward_decode_kernel_profile).  Recommended: "
        "'qwen35::forward_gpu::.*forward_gpu_greedy|forward_mlx::.*forward_decode(?!_kernel_profile)'.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Output markdown path.  Defaults to stdout.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional structured output path (raw aggregate dict).",
    )
    args = parser.parse_args(argv)

    if len(args.trials) < 1:
        sys.stderr.write("ERROR: at least one trial file is required\n")
        return 2

    if len(args.trials) < 5:
        sys.stderr.write(
            f"WARN: {len(args.trials)} trials given; 5 recommended for "
            f"median methodology (ADR-015 Wave 2b Q1 outlier-bias fix)\n"
        )

    decode_filter_re: re.Pattern[str] | None = None
    if args.decode_filter is not None:
        try:
            decode_filter_re = re.compile(args.decode_filter)
        except re.error as e:
            sys.stderr.write(f"ERROR: invalid --decode-filter regex: {e}\n")
            return 2

    trials: list[TrialFrames] = []
    for i, path in enumerate(args.trials, start=1):
        if not path.exists():
            sys.stderr.write(f"ERROR: trial {i} file not found: {path}\n")
            return 2
        trials.append(
            parse_xctrace_topcalls(
                path, trial_id=i, decode_filter_re=decode_filter_re
            )
        )

    hypotheses_specs = load_hypothesis_config(args.hypothesis_config)
    hypotheses_results: list[dict[str, object]] = []
    for spec in hypotheses_specs:
        hypotheses_results.append(aggregate_canonical(trials, spec))

    rank_stability_data = rank_stability(trials, top_n=args.top_n)

    md = render_markdown(
        trials, hypotheses_results, rank_stability_data, args.tokens_per_trial
    )

    if args.output_md is None:
        sys.stdout.write(md)
    else:
        args.output_md.write_text(md)
        sys.stderr.write(f"wrote markdown report → {args.output_md}\n")

    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(
                {
                    "trials": [
                        {
                            "trial_id": t.trial_id,
                            "source": str(t.source),
                            "samples": t.samples,
                            "total_ms": t.total_ms,
                        }
                        for t in trials
                    ],
                    "hypotheses": hypotheses_results,
                    "rank_stability": rank_stability_data,
                },
                indent=2,
            )
        )
        sys.stderr.write(f"wrote json report     → {args.output_json}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
