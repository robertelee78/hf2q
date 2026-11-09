#!/usr/bin/env python3
"""Reduce paired ADR-065 E1 wall-control/profile runs with transcript controls."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q1_decision_policy import transcripts

PREFIX = "[ADR065_STAGE_PROFILE_JSON] "
SCHEMA = "hf2q.adr065.stage-profile-round.v1"
GPU_CLASSES = {
    "projections", "delta_conv", "delta_gdn", "kv_write", "attention",
    "ffn_moe", "head", "mtp_draft", "mtp_catchup", "unknown",
}
CPU_CLASSES = {
    "logit_readback", "host_selection", "rollback", "per_row_arena_setup", "host_gap",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_files(run):
    run = Path(run)
    receipt = load_json(run / "receipt.json")
    summary_path = Path(receipt["client"]["summary_path"])
    if not summary_path.exists():
        summary_path = run / "client" / "summary.json"
    return run, receipt, load_json(summary_path), summary_path


def validate_receipt(run, receipt):
    verdict = receipt.get("verdict", {}).get("verdict")
    if verdict not in {"CalibrationPass", "AdmittedRunPass"}:
        raise ValueError(f"{run}: non-passing receipt verdict {verdict!r}")
    if not receipt.get("cleanup_proven", receipt.get("cleanup", {}).get("proven", False)):
        raise ValueError(f"{run}: cleanup is not proven")
    return receipt["artifact"]["sha256"], receipt["binary"]["sha256"]


def parse_records(path):
    records = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.startswith(PREFIX):
            continue
        record = json.loads(line[len(PREFIX):])
        required = {
            "schema", "mode", "family", "route", "width", "wall_ns", "gpu", "cpu",
            "unknown_labels", "deepseek_decode_scratch",
        }
        if set(record) != required:
            raise ValueError(f"{path}: stage record keys {sorted(record)} != {sorted(required)}")
        if record["schema"] != SCHEMA or record["mode"] not in {"wall", "profile"}:
            raise ValueError(f"{path}: invalid stage schema or mode")
        if not isinstance(record["width"], int) or record["width"] < 1 or record["wall_ns"] <= 0:
            raise ValueError(f"{path}: invalid width/wall")
        if not set(record["gpu"]).issubset(GPU_CLASSES):
            raise ValueError(f"{path}: unknown GPU class {set(record['gpu']) - GPU_CLASSES}")
        if not set(record["cpu"]).issubset(CPU_CLASSES):
            raise ValueError(f"{path}: unknown CPU class {set(record['cpu']) - CPU_CLASSES}")
        for entry in record["gpu"].values():
            if set(entry) != {"count", "gpu_ns"} or min(entry.values()) < 0:
                raise ValueError(f"{path}: invalid GPU bucket")
        for entry in record["cpu"].values():
            if set(entry) != {"count", "cpu_ns"} or min(entry.values()) < 0:
                raise ValueError(f"{path}: invalid CPU bucket")
        if record["mode"] == "profile" and "host_gap" not in record["cpu"]:
            raise ValueError(f"{path}: profile record has no host_gap bucket")
        unknown = record["unknown_labels"]
        if not isinstance(unknown, dict):
            raise ValueError(f"{path}: unknown_labels is not an object")
        for label, entry in unknown.items():
            if (
                not isinstance(label, str)
                or set(entry) != {"count", "gpu_ns"}
                or min(entry.values()) < 0
            ):
                raise ValueError(f"{path}: invalid Unknown label bucket")
        scratch = record["deepseek_decode_scratch"]
        scratch_keys = {
            "release_events", "reallocation_events",
            "release_events_total", "reallocation_events_total",
        }
        if set(scratch) != scratch_keys or min(scratch.values()) < 0:
            raise ValueError(f"{path}: invalid DeepSeek scratch counters")
        if record["mode"] == "wall" and (record["gpu"] or record["cpu"]):
            raise ValueError(f"{path}: wall control contains stage attribution")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no {PREFIX.strip()} records")
    return records




def groups(records):
    grouped = {}
    for record in records:
        grouped.setdefault((record["route"], record["width"]), []).append(record)
    return grouped


def median(values):
    return statistics.median(values) if values else None


def reduce_group(records):
    gpu = {name: sum(r["gpu"].get(name, {}).get("gpu_ns", 0) for r in records) for name in GPU_CLASSES}
    cpu = {name: sum(r["cpu"].get(name, {}).get("cpu_ns", 0) for r in records) for name in CPU_CLASSES}
    wall = sum(r["wall_ns"] for r in records)
    unknown = gpu["unknown"]
    known = sum(value for name, value in gpu.items() if name != "unknown") + sum(cpu.values())
    return {
        "rounds": len(records),
        "wall_ns_total": wall,
        "wall_ns_median": median([r["wall_ns"] for r in records]),
        "gpu_ns": {name: value for name, value in sorted(gpu.items()) if value},
        "cpu_ns": {name: value for name, value in sorted(cpu.items()) if value},
        "known_share": known / wall if wall else None,
        "unknown_gpu_share": unknown / wall if wall else None,
        "scratch_release_events": sum(r["deepseek_decode_scratch"]["release_events"] for r in records),
        "scratch_reallocation_events": sum(r["deepseek_decode_scratch"]["reallocation_events"] for r in records),
    }


def analyze_pair(family, control_dir, profile_dir, reference_dir):
    c_run, c_receipt, c_summary, c_summary_path = run_files(control_dir)
    p_run, p_receipt, p_summary, p_summary_path = run_files(profile_dir)
    r_run, r_receipt, r_summary, r_summary_path = run_files(reference_dir)
    c_artifact, c_binary = validate_receipt(c_run, c_receipt)
    p_artifact, p_binary = validate_receipt(p_run, p_receipt)
    r_artifact, _ = validate_receipt(r_run, r_receipt)
    if {c_receipt.get("family"), p_receipt.get("family"), r_receipt.get("family")} != {family}:
        raise ValueError(f"{family}: receipt families differ")
    if len({c_artifact, p_artifact, r_artifact}) != 1:
        raise ValueError(f"{family}: artifact identities differ")
    if c_binary != p_binary:
        raise ValueError(f"{family}: control/profile binaries differ")
    exactness = transcripts([c_run, r_run])
    if exactness["classification"] != "ExactTranscripts":
        raise ValueError(f"{family}: wall-control transcript differs from previous default")
    c_env = c_receipt.get("serve_env", {})
    p_env = p_receipt.get("serve_env", {})
    if c_env.get("HF2Q_ADR065_STAGE_PROFILE") != "wall":
        raise ValueError(f"{family}: control is not wall mode")
    if p_env.get("HF2Q_ADR065_STAGE_PROFILE") != "1" or p_env.get("MLX_PROFILE_CB") != "1":
        raise ValueError(f"{family}: profile controls are incomplete")
    control = parse_records(c_run / "server.stderr")
    profile = parse_records(p_run / "server.stderr")
    record_family = "qwen35" if family == "qwen38" else family
    if any(r["mode"] != "wall" or r["family"] != record_family for r in control):
        raise ValueError(f"{family}: invalid control records")
    if any(r["mode"] != "profile" or r["family"] != record_family for r in profile):
        raise ValueError(f"{family}: invalid profile records")
    cg, pg = groups(control), groups(profile)
    comparisons = {}
    for key in sorted(cg.keys() & pg.keys()):
        control_median = median([r["wall_ns"] for r in cg[key]])
        profile_median = median([r["wall_ns"] for r in pg[key]])
        comparisons[f"{key[0]}:w{key[1]}"] = {
            "control_rounds": len(cg[key]),
            "profile_rounds": len(pg[key]),
            "control_wall_ns_median": control_median,
            "profile_wall_ns_median": profile_median,
            "profile_over_control": profile_median / control_median,
            "profile_attribution": reduce_group(pg[key]),
        }
    if not comparisons:
        raise ValueError(f"{family}: no common route/width round groups")
    return {
        "artifact_sha256": c_artifact,
        "binary_sha256": c_binary,
        "exact_transcripts_vs_previous_default": exactness,
        "control_makespan_seconds": c_summary["own_output_four_task_makespan_seconds"],
        "profile_makespan_seconds": p_summary["own_output_four_task_makespan_seconds"],
        "profile_makespan_over_control": (
            p_summary["own_output_four_task_makespan_seconds"]
            / c_summary["own_output_four_task_makespan_seconds"]
        ),
        "comparisons": comparisons,
        "evidence": {
            "control_receipt_sha256": sha256(c_run / "receipt.json"),
            "profile_receipt_sha256": sha256(p_run / "receipt.json"),
            "reference_receipt_sha256": sha256(r_run / "receipt.json"),
            "control_server_log_sha256": sha256(c_run / "server.stderr"),
            "profile_server_log_sha256": sha256(p_run / "server.stderr"),
            "control_summary_sha256": sha256(c_summary_path),
            "profile_summary_sha256": sha256(p_summary_path),
            "reference_summary_sha256": sha256(r_summary_path),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", action="append", nargs=4, metavar=("FAMILY", "CONTROL", "PROFILE", "REFERENCE"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = {
        "schema": "hf2q.adr065.stage-profile-report.v1",
        "families": {
            family: analyze_pair(family, control, profile, reference)
            for family, control, profile, reference in args.pair
        },
        "decision_rule": "Unknown GPU time is reported and excluded from all stage decisions.",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
