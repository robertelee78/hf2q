#!/usr/bin/env python3
"""Verify and analyze the immutable ADR-049 B.2 prefill-width receipt."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

SCHEMA = 1
WIDTHS = [128, 256, 512, 1024, 1792]
WARMUPS = 2
ALLOWED_TRIALS = {7, 21}
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 49_002


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"ADR-049 B.2 receipt rejected: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_one_json(path: Path) -> dict:
    try:
        decoder = json.JSONDecoder()
        text = path.read_text(encoding="utf-8")
        value, end = decoder.raw_decode(text)
        if text[end:].strip():
            fail(f"{path.name} contains more than one JSON value")
        if not isinstance(value, dict):
            fail(f"{path.name} is not a JSON object")
        return value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read {path.name}: {error}")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                fail(f"{path.name}:{line_number} is blank")
            value = json.loads(line)
            if not isinstance(value, dict):
                fail(f"{path.name}:{line_number} is not an object")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read {path.name}: {error}")
    return rows


def finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        fail(f"{label} is outside its finite positive domain")
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def fit(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator <= 0:
        fail("row-width fit has no x-axis variance")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in points]
    total = sum((y - mean_y) ** 2 for _, y in points)
    r_squared = 1 - sum(value * value for value in residuals) / total if total > 0 else 0.0
    median_relative_residual = statistics.median(
        abs(residual) / (intercept + slope * x)
        for (x, _), residual in zip(points, residuals)
    )
    return intercept, slope, r_squared, median_relative_residual


def parse_telemetry(path: Path, expected_phases: set[str]) -> list[tuple[int, str, str]]:
    parsed: list[tuple[int, str, str]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split("\t")
            if len(fields) != 3 or not fields[0].isdigit() or fields[2] not in expected_phases:
                fail(f"malformed telemetry row {path.name}:{number}")
            parsed.append((int(fields[0]), fields[1], fields[2]))
    except (OSError, UnicodeError) as error:
        fail(f"cannot read {path.name}: {error}")
    if not parsed:
        fail(f"{path.name} is empty")
    return parsed


def validate_telemetry(receipt_dir: Path, manifest: dict) -> None:
    settle_path = receipt_dir / manifest["files"]["thermal_settle"]["path"]
    measurement_path = receipt_dir / manifest["files"]["thermal_measurement"]["path"]
    contention_settle = receipt_dir / manifest["files"]["contention_settle"]["path"]
    contention_measurement = receipt_dir / manifest["files"]["contention_measurement"]["path"]

    settle = parse_telemetry(settle_path, {"adr049-b2-settle"})
    measurement = parse_telemetry(
        measurement_path,
        {"adr049-b2-measurement-start", "adr049-b2-measurement", "adr049-b2-measurement-end"},
    )
    if settle[-1][0] - settle[0][0] < 60 or any(state != "nominal" for _, state, _ in settle):
        fail("thermal settle was not nominal for at least 60 seconds")
    if any(b[0] - a[0] > 8 or b[0] < a[0] for a, b in zip(settle, settle[1:])):
        fail("thermal settle telemetry has a gap")
    if len(measurement) < 2 or measurement[0][2] != "adr049-b2-measurement-start":
        fail("measurement telemetry lacks its start sentinel")
    if measurement[-1][2] != "adr049-b2-measurement-end" or measurement[0][1] != "nominal":
        fail("measurement telemetry lacks a calibrated boundary")
    if any(state not in {"nominal", "fair"} for _, state, _ in measurement):
        fail("measurement exceeded fair thermal state")
    if any(b[0] - a[0] > 5 or b[0] < a[0] for a, b in zip(measurement, measurement[1:])):
        fail("measurement thermal telemetry has a gap")

    def contention(path: Path, thermal: list[tuple[int, str, str]]) -> None:
        try:
            rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
        except (OSError, UnicodeError) as error:
            fail(f"cannot read {path.name}: {error}")
        if len(rows) != len(thermal):
            fail(f"{path.name} is not aligned with thermal telemetry")
        for fields, thermal_row in zip(rows, thermal):
            if len(fields) != 5 or fields[0] != str(thermal_row[0]) or fields[1] != "quiet":
                fail(f"{path.name} reports contention or timestamp drift")
            if fields[2] != thermal_row[2] or fields[3] == "" or fields[4] != "-":
                fail(f"{path.name} has malformed contention evidence")

    contention(contention_settle, settle)
    contention(contention_measurement, measurement)


def validate_manifest(receipt_dir: Path, manifest: dict) -> int:
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "measured":
        fail("manifest schema or status is invalid")
    if manifest.get("family") not in {"qwen35_moe", "gemma4_moe"}:
        fail("family is not a B.2 MoE family")
    expected_trace = "qwen35_chunk" if manifest["family"] == "qwen35_moe" else "gemma4_transaction"
    if manifest.get("trace_kind") != expected_trace:
        fail("family trace hook is mismatched")
    if manifest.get("width_targets") != WIDTHS or manifest.get("warmups") != WARMUPS:
        fail("pre-registered width/warmup contract drifted")
    trials = manifest.get("trials")
    if not isinstance(trials, int) or isinstance(trials, bool) \
            or trials not in ALLOWED_TRIALS \
            or manifest.get("order") != "ascending-descending-alternating":
        fail("pre-registered trial/order contract drifted")
    if manifest.get("max_slots") != 4:
        fail("measurement did not use the four-slot scheduler configuration")
    if manifest.get("request_settings") != {
        "max_tokens": 1,
        "repetition_penalty": 1,
        "seed": 42,
        "stream": False,
        "temperature": 0,
        "thinking": False,
    }:
        fail("request settings drifted")
    identity = manifest.get("identity", {})
    for key, pattern_length in (("source_sha", 40), ("binary_sha256", 64), ("model_sha256", 64)):
        value = identity.get(key, "")
        if len(value) != pattern_length or any(character not in "0123456789abcdef" for character in value):
            fail(f"identity.{key} is invalid")
    if identity.get("source_dirty") is not False or not isinstance(identity.get("model_bytes"), int):
        fail("source cleanliness or model byte identity is missing")
    if identity["model_bytes"] <= 0 or identity.get("binary_path", "")[0:1] != "/":
        fail("binary/model identity is incomplete")
    if identity.get("model_path", "")[0:1] != "/" or not identity.get("server_command"):
        fail("model/server identity is incomplete")
    files = manifest.get("files", {})
    required = {"samples", "models", "server_log", "thermal_settle", "thermal_measurement", "contention_settle", "contention_measurement"}
    if set(files) != required:
        fail("manifest file set is not exact")
    for label, record in files.items():
        relative = record.get("path", "")
        if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            fail(f"unsafe receipt path for {label}")
        path = receipt_dir / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != record.get("sha256"):
            fail(f"receipt file identity failed for {label}")
    models = read_one_json(receipt_dir / files["models"]["path"])
    model_rows = models.get("data")
    if not isinstance(model_rows, list) or len(model_rows) != 1 \
            or not isinstance(model_rows[0], dict) \
            or model_rows[0].get("id") != identity.get("server_model_id"):
        fail("server model-list response does not bind the recorded model ID")
    validate_telemetry(receipt_dir, manifest)
    return trials


def validate_samples(
    receipt_dir: Path,
    manifest: dict,
    rows: list[dict],
    trials: int,
) -> dict[int, list[tuple[int, float]]]:
    expected_count = (WARMUPS + trials) * len(WIDTHS)
    if len(rows) != expected_count:
        fail(f"expected {expected_count} samples, found {len(rows)}")
    grouped: dict[int, list[tuple[int, float]]] = {width: [] for width in WIDTHS}
    cursor = 0
    for phase, sweeps in (("warmup", WARMUPS), ("measure", trials)):
        for sweep in range(sweeps):
            order = WIDTHS if sweep % 2 == 0 else list(reversed(WIDTHS))
            for position, target in enumerate(order):
                row = rows[cursor]
                cursor += 1
                required = {
                    "schema_version", "sample_id", "phase", "sweep", "position", "target_rows",
                    "prompt_tokens", "cached_tokens", "work_rows", "prefill_ms", "ttft_ms", "wall_ms",
                    "trace_event_count", "trace_advanced_rows", "request_path", "request_sha256",
                    "response_path", "response_sha256", "wall_path", "wall_sha256",
                    "trace_path", "trace_sha256",
                }
                if set(row) != required or row["schema_version"] != SCHEMA:
                    fail(f"sample {cursor} schema is not exact")
                if (row["phase"], row["sweep"], row["position"], row["target_rows"]) != (phase, sweep, position, target):
                    fail(f"sample {cursor} order drifted")
                prompt = row["prompt_tokens"]
                cached = row["cached_tokens"]
                work = row["work_rows"]
                if not all(isinstance(value, int) and not isinstance(value, bool) for value in (prompt, cached, work)):
                    fail(f"sample {cursor} token counts are not integers")
                if cached != 0 or prompt <= cached or work != prompt - cached or work > 2048:
                    fail(f"sample {cursor} has invalid uncached work rows")
                if work < target * 0.75 or work > target * 1.25:
                    fail(f"sample {cursor} missed its target row bin")
                for label in ("prefill_ms", "ttft_ms", "wall_ms"):
                    finite_number(row[label], f"sample {cursor} {label}", positive=True)
                if row["trace_event_count"] != 1 or row["trace_advanced_rows"] != work:
                    fail(f"sample {cursor} is not one production prefill transaction")
                for kind in ("request", "response", "wall", "trace"):
                    relative = row[f"{kind}_path"]
                    path = receipt_dir / relative
                    if Path(relative).is_absolute() or ".." in Path(relative).parts:
                        fail(f"sample {cursor} has unsafe {kind} path")
                    if not path.is_file() or path.is_symlink() or sha256(path) != row[f"{kind}_sha256"]:
                        fail(f"sample {cursor} {kind} identity failed")
                request = read_one_json(receipt_dir / row["request_path"])
                if request.get("model") != manifest["identity"]["server_model_id"]:
                    fail(f"sample {cursor} request model identity drifted")
                if request.get("max_tokens") != 1 or request.get("seed") != 42 \
                        or request.get("temperature") != 0 or request.get("repetition_penalty") != 1 \
                        or request.get("stream") is not False:
                    fail(f"sample {cursor} request settings drifted")
                if request.get("hf2q_enable_thinking") is not False \
                        or request.get("chat_template_kwargs") != {"enable_thinking": False}:
                    fail(f"sample {cursor} request thinking policy drifted")
                messages = request.get("messages")
                if not isinstance(messages, list) or len(messages) != 1 \
                        or messages[0].get("role") != "user" \
                        or not isinstance(messages[0].get("content"), str) \
                        or not messages[0]["content"].startswith(f"adr049-b2-{row['sample_id']} "):
                    fail(f"sample {cursor} prompt identity drifted")
                response = read_one_json(receipt_dir / row["response_path"])
                try:
                    response_choices = response["choices"]
                    response_prompt = response["usage"]["prompt_tokens"]
                    response_cached = response["usage"]["prompt_tokens_details"]["cached_tokens"]
                    response_prefill = response["x_hf2q_timing"]["prefill_time_secs"] * 1000
                    response_ttft = response["x_hf2q_timing"]["time_to_first_token_ms"]
                except (KeyError, TypeError):
                    fail(f"sample {cursor} response timing schema is invalid")
                if not isinstance(response_choices, list) or len(response_choices) != 1 \
                        or not isinstance(response_choices[0].get("message", {}).get("content"), str) \
                        or not response_choices[0]["message"]["content"] \
                        or not isinstance(response_choices[0].get("finish_reason"), str):
                    fail(f"sample {cursor} response completion is invalid")
                if response_prompt != prompt or response_cached != cached \
                        or abs(float(response_prefill) - float(row["prefill_ms"])) > 1e-9 \
                        or abs(float(response_ttft) - float(row["ttft_ms"])) > 1e-9:
                    fail(f"sample {cursor} raw timing does not match its response")
                try:
                    wall_from_file = float((receipt_dir / row["wall_path"]).read_text(encoding="utf-8").strip()) * 1000
                except (OSError, UnicodeError, ValueError):
                    fail(f"sample {cursor} wall timing file is invalid")
                if not math.isfinite(wall_from_file) \
                        or abs(wall_from_file - float(row["wall_ms"])) > 1e-6:
                    fail(f"sample {cursor} raw wall timing does not match its wall file")
                trace_text = (receipt_dir / row["trace_path"]).read_text(encoding="utf-8")
                if manifest["trace_kind"] == "qwen35_chunk":
                    event_name = "Qwen35 bounded prefill chunk complete"
                    pattern = r"chunk_tokens[=: ]+([0-9]+)"
                else:
                    event_name = "Gemma4 bounded prefill transaction complete"
                    pattern = r"advanced_tokens[=: ]+([0-9]+)"
                trace_values = []
                for trace_line in trace_text.splitlines():
                    if event_name in trace_line:
                        trace_values.extend(int(value) for value in re.findall(pattern, trace_line))
                if len(trace_values) != row["trace_event_count"] \
                        or trace_values != [row["trace_advanced_rows"]]:
                    fail(f"sample {cursor} raw trace fields do not match its trace slice")
                if phase == "measure":
                    grouped[target].append((work, float(row["prefill_ms"])))
    if any(len(values) != trials for values in grouped.values()):
        fail("measured width groups are incomplete")
    return grouped


def analyze(grouped: dict[int, list[tuple[int, float]]], trials: int) -> dict:
    median_points = [
        (statistics.median(row for row, _ in grouped[target]), statistics.median(ms for _, ms in grouped[target]))
        for target in WIDTHS
    ]
    intercept, slope, r_squared, residual = fit(median_points)
    fit_valid = intercept > 0 and slope > 0 and r_squared >= 0.95 and residual <= 0.05
    bootstrap_fixed: list[float] = []
    bootstrap_gain: list[float] = []
    rng = random.Random(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAPS):
        points = []
        for target in WIDTHS:
            values = grouped[target]
            sampled = [values[rng.randrange(trials)] for _ in range(trials)]
            points.append((statistics.median(x for x, _ in sampled), statistics.median(y for _, y in sampled)))
        candidate_intercept, candidate_slope, _, _ = fit(points)
        if candidate_intercept > 0 and candidate_slope > 0:
            bootstrap_fixed.append(candidate_intercept / (candidate_intercept + candidate_slope * 128))
            bootstrap_gain.append(4 * (candidate_intercept + candidate_slope * 128) / (candidate_intercept + candidate_slope * 512))
    if len(bootstrap_fixed) < BOOTSTRAPS * 0.95:
        fit_valid = False
    fixed_share = intercept / (intercept + slope * 128) if intercept > 0 and slope > 0 else float("nan")
    gain = 4 * (intercept + slope * 128) / (intercept + slope * 512) if intercept > 0 and slope > 0 else float("nan")
    fixed_ci = [percentile(bootstrap_fixed, 0.025), percentile(bootstrap_fixed, 0.975)] if bootstrap_fixed else [None, None]
    gain_ci = [percentile(bootstrap_gain, 0.025), percentile(bootstrap_gain, 0.975)] if bootstrap_gain else [None, None]
    decision = "invalid"
    if fit_valid:
        if fixed_ci[0] > 0.50 and gain_ci[0] > 1.10:
            decision = "confirmed"
        elif fixed_ci[1] <= 0.25 or gain_ci[1] <= 1.05:
            decision = "falsified"
        else:
            decision = "inconclusive"
    return {
        "fit_valid": fit_valid,
        "decision": decision,
        "intercept_ms": intercept,
        "slope_ms_per_row": slope,
        "r_squared": r_squared,
        "median_relative_residual": residual,
        "fixed_share_at_128": fixed_share,
        "fixed_share_95pct_ci": fixed_ci,
        "projected_four_by_128_gain": gain,
        "projected_gain_95pct_ci": gain_ci,
        "bootstrap_samples": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "trials_per_width": trials,
        "median_points": [{"actual_rows": x, "prefill_ms": y} for x, y in median_points],
        "decision_rule": {
            "confirmed": "lower fixed-share CI > 0.50 and lower projected-gain CI > 1.10",
            "falsified": "upper fixed-share CI <= 0.25 or upper projected-gain CI <= 1.05",
            "inconclusive": "valid fit satisfying neither terminal rule",
            "valid_fit": "F > 0, c > 0, R^2 >= 0.95, median relative residual <= 0.05",
        },
    }


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        fail("usage: verify_adr049_b2_prefill_curve.py RECEIPT_DIR [SUMMARY_PATH]")
    receipt_dir = Path(sys.argv[1]).resolve()
    manifest_path = receipt_dir / "manifest.json"
    manifest = read_one_json(manifest_path)
    trials = validate_manifest(receipt_dir, manifest)
    samples_path = receipt_dir / manifest["files"]["samples"]["path"]
    grouped = validate_samples(receipt_dir, manifest, read_jsonl(samples_path), trials)
    analysis = analyze(grouped, trials)
    output = {
        "schema_version": SCHEMA,
        "status": "valid" if analysis["fit_valid"] else "invalid",
        "manifest_sha256": sha256(manifest_path),
        "family": manifest["family"],
        "identity": manifest["identity"],
        "analysis": analysis,
    }
    rendered = json.dumps(output, sort_keys=True, indent=2) + "\n"
    if len(sys.argv) == 3:
        destination = Path(sys.argv[2])
        if destination.exists():
            fail(f"refusing to overwrite summary: {destination}")
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(destination)
    else:
        sys.stdout.write(rendered)
    if not analysis["fit_valid"]:
        fail("linear cost model did not pass its pre-registered validity gate")


if __name__ == "__main__":
    main()
