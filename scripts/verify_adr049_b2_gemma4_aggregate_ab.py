#!/usr/bin/env python3
"""Independently verify an ADR-049 B.2 Gemma4 aggregation receipt."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

SCHEMA = 1
PAIRS = 8
WIDTHS = [128, 256, 512]
LANES = 4
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 49_004
MIN_LOWER_CI = 1.05
TRACE_RE = re.compile(
    r"\[PREFILL_TIMING\] BATCHED ([0-9]+) seqs in "
    r"([0-9]+(?:\.[0-9]+)?) ms \(one multi-seq forward, iter-G\(a\)\)"
)
FATAL_RE = re.compile(
    r"GPU Timeout|SubmissionsIgnored|Command buffer error|Generation error|"
    r"engine_unhealthy|panicked at|worker-fatal",
    re.IGNORECASE,
)


def fail(message: str) -> NoReturn:
    raise SystemExit(f"ADR-049 B.2 Gemma A/B receipt rejected: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        fail(f"cannot hash {path}: {error}")
    return digest.hexdigest()


def command(*args: str) -> str:
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"identity command failed ({' '.join(args)}): {error}")


def read_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
        value, end = json.JSONDecoder().raw_decode(text)
        if text[end:].strip() or not isinstance(value, dict):
            fail(f"{path.name} is not exactly one JSON object")
        return value
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")


def read_jsonl(path: Path) -> list[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            fail(f"{path.name} is empty or contains a blank row")
        values = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"cannot read {path}: {error}")
    if any(not isinstance(value, dict) for value in values):
        fail(f"{path.name} contains a non-object row")
    return values


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        fail(f"{label} is not finite and positive")
    return result


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bind_file(root: Path, relative: object, expected_sha: object, label: str) -> Path:
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected_sha, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
    ):
        fail(f"unsafe or malformed {label} binding")
    path = root / relative
    if not path.is_file() or path.is_symlink() or sha256(path) != expected_sha:
        fail(f"{label} file identity failed")
    return path


def binding_args(binding: object, label: str) -> dict:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        fail(f"{label} binding schema is invalid")
    return {"relative": binding["path"], "expected_sha": binding["sha256"], "label": label}


def file_snapshot(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{int(stat.st_mtime)}:{int(stat.st_ctime)}"


def validate_live_identity(identity: dict) -> None:
    expected_keys = {
        "source_root", "source_sha", "source_dirty", "binary_path",
        "binary_sha256", "model_path", "model_sha256", "model_bytes",
        "model_snapshot", "operator_launcher_path", "operator_launcher_sha256",
    }
    if set(identity) != expected_keys or identity.get("source_dirty") is not False:
        fail("identity schema is not exact")
    for key, length in (("source_sha", 40), ("binary_sha256", 64),
                        ("model_sha256", 64), ("operator_launcher_sha256", 64)):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", str(identity.get(key, ""))):
            fail(f"identity.{key} is malformed")
    for key in ("source_root", "binary_path", "model_path", "operator_launcher_path"):
        value = identity.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            fail(f"identity.{key} is not absolute")
    source = Path(identity["source_root"])
    binary = Path(identity["binary_path"])
    model = Path(identity["model_path"])
    launcher = Path(identity["operator_launcher_path"])
    if any(path.is_symlink() for path in (source, binary, model, launcher)):
        fail("source/binary/model/launcher cannot be symlinks")
    if not source.is_dir() or not binary.is_file() or not model.is_file() or not launcher.is_file():
        fail("source/binary/model/launcher is unavailable")
    if binary != source / "target/release/hf2q" or launcher != source / "scripts/serve_gemma4_opencode.sh":
        fail("binary or launcher is not the canonical source-tree object")
    if command("git", "-C", str(source), "rev-parse", "HEAD") != identity["source_sha"]:
        fail("live source SHA drifted")
    if command("git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"):
        fail("live source is dirty")
    if sha256(binary) != identity["binary_sha256"]:
        fail("live binary hash drifted")
    try:
        if identity["source_sha"].encode() not in binary.read_bytes():
            fail("binary does not embed source SHA")
    except OSError as error:
        fail(f"cannot inspect binary provenance: {error}")
    if not isinstance(identity.get("model_bytes"), int) or identity["model_bytes"] <= 0:
        fail("model byte binding is invalid")
    if model.stat().st_size != identity["model_bytes"] or sha256(model) != identity["model_sha256"]:
        fail("live model content drifted")
    if file_snapshot(model) != identity["model_snapshot"]:
        fail("live model file snapshot drifted")
    if sha256(launcher) != identity["operator_launcher_sha256"]:
        fail("live operator launcher drifted")


def telemetry_rows(path: Path, phases: set[str], fields: int) -> list[list[str]]:
    try:
        rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError) as error:
        fail(f"cannot read telemetry {path.name}: {error}")
    if not rows:
        fail(f"empty telemetry {path.name}")
    for number, row in enumerate(rows, 1):
        if len(row) != fields or not row[0].isdigit() or row[2] not in phases:
            fail(f"malformed telemetry {path.name}:{number}")
    return rows


def validate_environment(root: Path, manifest: dict) -> None:
    environment = manifest.get("environment")
    if not isinstance(environment, dict) or set(environment) != {
        "power", "power_mode", "power_mode_code", "thermal",
        "host_contention", "clean_process_environment",
    }:
        fail("environment contract schema drifted")
    if environment["power"] != "ac" or environment["power_mode"] == "low" \
            or not environment["power_mode"] or not environment["power_mode_code"] \
            or environment["thermal"] != "nominal-settle-and-fair-or-better-measurement" \
            or environment["host_contention"] != "quiet" \
            or environment["clean_process_environment"] is not True:
        fail("environment contract is not acceptable")
    files = manifest["files"]
    settle = telemetry_rows(
        bind_file(root, **binding_args(files["thermal_settle"], "thermal settle")),
        {"adr049-b2-gemma-ab-settle"}, 3,
    )
    measurement = telemetry_rows(
        bind_file(root, **binding_args(files["thermal_measurement"], "thermal measurement")),
        {"adr049-b2-gemma-ab-start", "adr049-b2-gemma-ab-measurement", "adr049-b2-gemma-ab-end"}, 3,
    )
    contention_settle = telemetry_rows(
        bind_file(root, **binding_args(files["contention_settle"], "contention settle")),
        {"adr049-b2-gemma-ab-settle"}, 5,
    )
    contention_measurement = telemetry_rows(
        bind_file(root, **binding_args(files["contention_measurement"], "contention measurement")),
        {"adr049-b2-gemma-ab-start", "adr049-b2-gemma-ab-measurement", "adr049-b2-gemma-ab-end"}, 5,
    )
    if int(settle[-1][0]) - int(settle[0][0]) < 60 or any(row[1] != "nominal" for row in settle):
        fail("settle was not nominal for at least 60 seconds")
    if any(int(right[0]) < int(left[0]) or int(right[0]) - int(left[0]) > 8
           for left, right in zip(settle, settle[1:])):
        fail("settle telemetry has a gap")
    if len(measurement) < 2 or measurement[0][2] != "adr049-b2-gemma-ab-start" \
            or measurement[-1][2] != "adr049-b2-gemma-ab-end" \
            or measurement[0][1] != "nominal" \
            or any(row[1] not in {"nominal", "fair"} for row in measurement):
        fail("measurement thermal sentinels are invalid")
    if any(int(right[0]) < int(left[0]) or int(right[0]) - int(left[0]) > 5
           for left, right in zip(measurement, measurement[1:])):
        fail("measurement thermal telemetry has a gap")
    for thermal, contention, label in ((settle, contention_settle, "settle"),
                                       (measurement, contention_measurement, "measurement")):
        if len(thermal) != len(contention):
            fail(f"{label} telemetry is not aligned")
        for thermal_row, contention_row in zip(thermal, contention):
            if contention_row[0] != thermal_row[0] or contention_row[2] != thermal_row[2] \
                    or contention_row[1] != "quiet" or not contention_row[3] \
                    or contention_row[4] != "-":
                fail(f"{label} telemetry reports contention or alignment drift")
    guard = files["power_guard"]
    if set(guard) != {"caffeinate_log", "assertions", "events_baseline", "events_final", "events_new"}:
        fail("power guard receipt schema drifted")
    guard_paths = {key: bind_file(root, **binding_args(value, f"power guard {key}"))
                   for key, value in guard.items()}
    if "caffeinate" not in guard_paths["assertions"].read_text(encoding="utf-8"):
        fail("power guard assertion was not captured")
    if guard_paths["events_new"].read_text(encoding="utf-8").strip():
        fail("power guard observed new sleep/wake/power events")


def validate_power(path: Path, pair: int, arm: str, environment: dict) -> None:
    try:
        rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError) as error:
        fail(f"cannot read process power telemetry: {error}")
    expected_phases = [f"pair-{pair}-{arm}-{phase}" for phase in (
        "before-launch", "loaded-warm", "measurement-start", "measurement-end", "after-shutdown"
    )]
    if len(rows) != len(expected_phases):
        fail(f"pair {pair} {arm} power sample count drifted")
    previous = -1
    for row, phase in zip(rows, expected_phases):
        if len(row) != 5 or not row[0].isdigit() or int(row[0]) < previous \
                or row[1:] != ["ac", environment["power_mode"], environment["power_mode_code"], phase]:
            fail(f"pair {pair} {arm} power contract drifted")
        previous = int(row[0])


def expected_configuration() -> dict:
    return {
        "pairs": PAIRS, "width_targets": WIDTHS, "lanes": LANES,
        "pair_order": "off-on-even_on-off-odd", "warmup_waves_per_process": 2,
        "measured_waves_per_process": 3,
        "off_env": {"HF2Q_CROSS_SLOT_ADMIT": "0", "HF2Q_ADMIT_COALESCE_US": "0"},
        "on_env": {"HF2Q_CROSS_SLOT_ADMIT": "1", "HF2Q_ADMIT_COALESCE_US": "25000"},
        "request": {"max_tokens": 1, "seed": 42, "temperature": 0,
                    "repetition_penalty": 1, "stream": False, "thinking": False},
        "analysis": {"statistic": "median paired OFF/ON wave speedup",
                     "order_stratified_bootstrap_samples": BOOTSTRAPS,
                     "bootstrap_seed": BOOTSTRAP_SEED,
                     "lower_confidence_percentile": 2.5,
                     "minimum_lower_95_speedup_exclusive": MIN_LOWER_CI},
    }


def validate_manifest(root: Path, manifest: dict) -> tuple[list[dict], dict[tuple[int, str], dict]]:
    if set(manifest) != {"schema_version", "status", "configuration", "identity", "environment", "processes", "files"} \
            or manifest.get("schema_version") != SCHEMA or manifest.get("status") != "measured":
        fail("manifest schema/status is invalid")
    if manifest.get("configuration") != expected_configuration():
        fail("pre-registered A/B configuration drifted")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        fail("identity record is missing")
    validate_live_identity(identity)
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {
        "samples", "process_bindings", "thermal_settle", "thermal_measurement",
        "contention_settle", "contention_measurement", "power_guard",
        "model_verification",
    }:
        fail("top-level receipt file set is not exact")
    validate_environment(root, manifest)
    model_verification_path = bind_file(
        root, **binding_args(files["model_verification"], "model verification")
    )
    model_verification = read_json(model_verification_path)
    if set(model_verification) != {
        "schema_version", "path", "sha256", "file_snapshot", "file_stamp",
        "content_hash_verified",
    } or model_verification["schema_version"] != 2 \
            or model_verification["path"] != identity["model_path"] \
            or model_verification["sha256"] != identity["model_sha256"] \
            or model_verification["file_snapshot"] != identity["model_snapshot"] \
            or model_verification["content_hash_verified"] is not True \
            or not isinstance(model_verification["file_stamp"], dict):
        fail("sealed model verification receipt is invalid")
    samples_path = bind_file(root, **binding_args(files["samples"], "samples"))
    bindings_path = bind_file(root, **binding_args(files["process_bindings"], "process bindings"))
    processes = manifest.get("processes")
    if not isinstance(processes, list) or len(processes) != PAIRS * 2 \
            or read_jsonl(bindings_path) != processes:
        fail("fresh process bindings are invalid")
    process_map: dict[tuple[int, str], dict] = {}
    seen_pids: set[int] = set()
    for index, binding in enumerate(processes):
        pair, position = index // 2, index % 2
        arm = (["off", "on"] if pair % 2 == 0 else ["on", "off"])[position]
        if binding != {"pair": pair, "position": position, "arm": arm,
                       "path": binding.get("path"), "sha256": binding.get("sha256")}:
            fail(f"process binding {index} violates alternating order or schema")
        record_path = bind_file(root, binding["path"], binding["sha256"], f"process {index}")
        record = read_json(record_path)
        expected_record_keys = {
            "schema_version", "status", "pair", "position", "arm", "pid", "command",
            "model_id", "max_slots", "runtime", "lever_env", "source_sha",
            "binary_sha256", "model_sha256", "wait_status", "power_path", "power_sha256",
            "command_path", "command_sha256", "models_path", "models_sha256",
            "stdout_path", "stdout_sha256", "stderr_path", "stderr_sha256",
        }
        if set(record) != expected_record_keys or record.get("schema_version") != SCHEMA \
                or (record.get("pair"), record.get("position"), record.get("arm"), record.get("status")) \
                != (pair, position, arm, "stopped") or record.get("wait_status") not in {0, 143}:
            fail(f"process record {index} schema/status drifted")
        pid = record.get("pid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or pid in seen_pids:
            fail("fresh-process proof reused or omitted a PID")
        seen_pids.add(pid)
        if record["lever_env"] != expected_configuration()[f"{arm}_env"] or record["max_slots"] != LANES:
            fail(f"process {index} lever environment drifted")
        if (record["source_sha"], record["binary_sha256"], record["model_sha256"]) != (
                identity["source_sha"], identity["binary_sha256"], identity["model_sha256"]):
            fail(f"process {index} exact identity drifted")
        runtime = record["runtime"]
        expected_runtime = {"clean_environment": True, "home": runtime.get("home"),
                            "path": "/usr/bin:/bin:/usr/sbin:/sbin", "tmpdir": "/var/tmp",
                            "locale": {"LANG": "C", "LC_ALL": "C"}, "rust_backtrace": "1",
                            "operator_launcher": identity["operator_launcher_path"],
                            "operator_launcher_sha256": identity["operator_launcher_sha256"],
                            "model_verification_receipt": str(model_verification_path),
                            "model_verification_receipt_sha256": files["model_verification"]["sha256"]}
        if runtime != expected_runtime or not isinstance(runtime.get("home"), str) \
                or not Path(runtime["home"]).is_absolute():
            fail(f"process {index} runtime environment drifted")
        process_command = record.get("command", "")
        if identity["binary_path"] not in process_command or identity["model_path"] not in process_command \
                or "--scheduler inflight-batched" not in process_command or "--max-slots 4" not in process_command:
            fail(f"process {index} command is not the production four-slot route")
        command_path = bind_file(root, record["command_path"], record["command_sha256"], f"process {index} command")
        if command_path.read_text(encoding="utf-8").strip() != process_command:
            fail(f"process {index} command is not raw-bound")
        models_path = bind_file(root, record["models_path"], record["models_sha256"], f"process {index} models")
        for artifact in ("stdout", "stderr"):
            artifact_path = bind_file(root, record[f"{artifact}_path"], record[f"{artifact}_sha256"], f"process {index} {artifact}")
            if artifact == "stderr" and FATAL_RE.search(artifact_path.read_text(encoding="utf-8")):
                fail(f"process {index} contains a fatal server signature")
        models = read_json(models_path)
        loaded = [row for row in models.get("data", []) if isinstance(row, dict) and row.get("loaded") is True]
        if len(loaded) != 1 or loaded[0].get("id") != record.get("model_id") or loaded[0].get("arch") != "gemma4":
            fail(f"process {index} model endpoint identity is invalid")
        power_path = bind_file(root, record["power_path"], record["power_sha256"], f"process {index} power")
        validate_power(power_path, pair, arm, manifest["environment"])
        process_map[(pair, arm)] = record
    return read_jsonl(samples_path), process_map


def normalize_response(response: dict) -> dict:
    try:
        return {"message": response["choices"][0]["message"],
                "finish_reason": response["choices"][0]["finish_reason"],
                "usage": {key: response["usage"].get(key) for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens",
                    "prompt_tokens_details",
                )}}
    except (KeyError, IndexError, TypeError):
        fail("response lacks normalized semantic fields")


def validate_samples(root: Path, rows: list[dict], processes: dict[tuple[int, str], dict]) -> dict[tuple[int, int, str], dict]:
    if len(rows) != PAIRS * 2 * len(WIDTHS):
        fail("wave sample count is invalid")
    samples: dict[tuple[int, int, str], dict] = {}
    cursor = 0
    for pair in range(PAIRS):
        arm_order = ["off", "on"] if pair % 2 == 0 else ["on", "off"]
        for position, arm in enumerate(arm_order):
            process = processes[(pair, arm)]
            for width_position, target in enumerate(WIDTHS):
                row = rows[cursor]
                cursor += 1
                required = {"schema_version", "pair", "process_position", "arm", "width_position",
                            "target_rows", "wave_ms", "wave_wall_path", "wave_wall_sha256",
                            "trace_path", "trace_sha256", "trace_event_count", "trace_requests",
                            "trace_elapsed_ms", "aggregate_work_rows", "launch_skew_seconds",
                            "latest_start", "earliest_finish", "actual_overlap",
                            "lanes_path", "lanes_sha256", "lanes"}
                if set(row) != required or row["schema_version"] != SCHEMA \
                        or (row["pair"], row["process_position"], row["arm"], row["width_position"], row["target_rows"]) \
                        != (pair, position, arm, width_position, target):
                    fail(f"wave {cursor} schema or execution order drifted")
                wave_ms = finite(row["wave_ms"], f"wave {cursor} wall")
                wall_path = bind_file(root, row["wave_wall_path"], row["wave_wall_sha256"], f"wave {cursor} wall")
                try:
                    raw_wall_ms = float(wall_path.read_text(encoding="utf-8").strip()) * 1000
                except (OSError, UnicodeError, ValueError):
                    fail(f"wave {cursor} wall file is invalid")
                if abs(raw_wall_ms - wave_ms) > 1e-6:
                    fail(f"wave {cursor} wall is not raw-bound")
                trace_path = bind_file(root, row["trace_path"], row["trace_sha256"], f"wave {cursor} trace")
                trace_values = [(int(a), float(b)) for a, b in TRACE_RE.findall(trace_path.read_text(encoding="utf-8"))]
                expected_events = 1 if arm == "on" else 0
                if row["trace_event_count"] != expected_events or len(trace_values) != expected_events:
                    fail(f"wave {cursor} aggregate reachability count is invalid")
                lanes_path = bind_file(root, row["lanes_path"], row["lanes_sha256"], f"wave {cursor} lanes")
                lanes = row["lanes"]
                if not isinstance(lanes, list) or len(lanes) != LANES or read_jsonl(lanes_path) != lanes:
                    fail(f"wave {cursor} does not contain four raw-bound lanes")
                starts, finishes, aggregate_rows = [], [], 0
                for lane_index, lane in enumerate(lanes):
                    lane_required = {"lane", "prompt_tokens", "cached_tokens", "work_rows", "prefill_ms",
                                     "ttft_ms", "wall_ms", "request_path", "request_sha256", "response_path",
                                     "response_sha256", "wall_path", "wall_sha256", "timing_path", "timing_sha256",
                                     "normalized_path", "normalized_sha256"}
                    if set(lane) != lane_required or lane["lane"] != lane_index:
                        fail(f"wave {cursor} lane {lane_index} schema is invalid")
                    prompt, cached, work = lane["prompt_tokens"], lane["cached_tokens"], lane["work_rows"]
                    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (prompt, cached, work)) \
                            or cached != 0 or work != prompt or not target * 0.75 <= work <= target * 1.25:
                        fail(f"wave {cursor} lane {lane_index} is not cold in its target bin")
                    aggregate_rows += work
                    for name in ("prefill_ms", "ttft_ms", "wall_ms"):
                        finite(lane[name], f"wave {cursor} lane {lane_index} {name}")
                    request_path = bind_file(root, lane["request_path"], lane["request_sha256"], "lane request")
                    response_path = bind_file(root, lane["response_path"], lane["response_sha256"], "lane response")
                    lane_wall_path = bind_file(root, lane["wall_path"], lane["wall_sha256"], "lane wall")
                    timing_path = bind_file(root, lane["timing_path"], lane["timing_sha256"], "lane timing")
                    normalized_path = bind_file(root, lane["normalized_path"], lane["normalized_sha256"], "lane normalized")
                    request = read_json(request_path)
                    messages = request.get("messages")
                    expected_prefix = f"adr049-b2-gemma-p{pair:02d}-w{target:03d}-l{lane_index} "
                    if request.get("model") != process["model_id"] or request.get("max_tokens") != 1 \
                            or request.get("seed") != 42 or request.get("temperature") != 0 \
                            or request.get("repetition_penalty") != 1 or request.get("stream") is not False \
                            or request.get("hf2q_enable_thinking") is not False \
                            or request.get("chat_template_kwargs") != {"enable_thinking": False} \
                            or not isinstance(messages, list) or len(messages) != 1 \
                            or not isinstance(messages[0], dict) or messages[0].get("role") != "user" \
                            or not messages[0].get("content", "").startswith(expected_prefix):
                        fail(f"wave {cursor} lane {lane_index} request drifted")
                    response = read_json(response_path)
                    try:
                        response_prompt = response["usage"]["prompt_tokens"]
                        response_cached = response["usage"]["prompt_tokens_details"]["cached_tokens"]
                        response_completion = response["usage"]["completion_tokens"]
                        response_prefill = response["x_hf2q_timing"]["prefill_time_secs"] * 1000
                        response_ttft = response["x_hf2q_timing"]["time_to_first_token_ms"]
                    except (KeyError, TypeError):
                        fail(f"wave {cursor} lane {lane_index} response timing is malformed")
                    if response_prompt != prompt or response_cached != cached or response_completion != 1 \
                            or abs(float(response_prefill) - float(lane["prefill_ms"])) > 1e-9 \
                            or abs(float(response_ttft) - float(lane["ttft_ms"])) > 1e-9:
                        fail(f"wave {cursor} lane {lane_index} response is not raw-bound")
                    try:
                        lane_wall = float(lane_wall_path.read_text(encoding="utf-8").strip()) * 1000
                        start_text, finish_text = timing_path.read_text(encoding="utf-8").strip().split("\t")
                        start, finish = float(start_text), float(finish_text)
                    except (OSError, UnicodeError, ValueError):
                        fail(f"wave {cursor} lane {lane_index} timing file is malformed")
                    if abs(lane_wall - float(lane["wall_ms"])) > 1e-6 or not math.isfinite(start) \
                            or not math.isfinite(finish) or start <= 0 or finish <= start:
                        fail(f"wave {cursor} lane {lane_index} timing is not raw-bound")
                    starts.append(start)
                    finishes.append(finish)
                    if read_json(normalized_path) != normalize_response(response):
                        fail(f"wave {cursor} lane {lane_index} normalized result is not derived")
                skew, latest, earliest = max(starts) - min(starts), max(starts), min(finishes)
                if row["actual_overlap"] is not True or skew > 0.100 or latest >= earliest \
                        or abs(float(row["launch_skew_seconds"]) - skew) > 1e-6 \
                        or abs(float(row["latest_start"]) - latest) > 1e-6 \
                        or abs(float(row["earliest_finish"]) - earliest) > 1e-6:
                    fail(f"wave {cursor} did not prove a simultaneous four-lane wave")
                if row["aggregate_work_rows"] != aggregate_rows:
                    fail(f"wave {cursor} aggregate work is not lane-bound")
                if arm == "on":
                    requests, elapsed = trace_values[0]
                    if requests != LANES or elapsed <= 0 or row["trace_requests"] != requests \
                            or abs(float(row["trace_elapsed_ms"]) - elapsed) > 1e-9:
                        fail(f"wave {cursor} did not reach one four-lane multi-seq forward")
                elif row["trace_requests"] is not None or row["trace_elapsed_ms"] is not None:
                    fail(f"wave {cursor} OFF arm fabricated reachability fields")
                samples[(pair, target, arm)] = row
    for pair in range(PAIRS):
        for target in WIDTHS:
            off, on = samples[(pair, target, "off")], samples[(pair, target, "on")]
            if [lane["request_sha256"] for lane in off["lanes"]] != [lane["request_sha256"] for lane in on["lanes"]]:
                fail(f"pair {pair} width {target} request bytes differ")
            if [lane["normalized_sha256"] for lane in off["lanes"]] != [lane["normalized_sha256"] for lane in on["lanes"]]:
                fail(f"pair {pair} width {target} canonical results differ")
    return samples


def analyze(samples: dict[tuple[int, int, str], dict]) -> dict:
    rng = random.Random(BOOTSTRAP_SEED)
    widths, all_pass = [], True
    for target in WIDTHS:
        strata = {"off_first": [], "on_first": []}
        ratios = []
        for pair in range(PAIRS):
            ratio = samples[(pair, target, "off")]["wave_ms"] / samples[(pair, target, "on")]["wave_ms"]
            ratios.append(ratio)
            strata["off_first" if pair % 2 == 0 else "on_first"].append(ratio)
        bootstrapped = []
        for _ in range(BOOTSTRAPS):
            draw = []
            for values in strata.values():
                draw.extend(values[rng.randrange(len(values))] for _ in values)
            bootstrapped.append(statistics.median(draw))
        interval = [percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)]
        accepted = interval[0] > MIN_LOWER_CI
        all_pass = all_pass and accepted
        widths.append({"target_rows": target, "paired_speedups": ratios,
                       "median_speedup": statistics.median(ratios),
                       "order_stratified_95pct_ci": interval, "accepted": accepted})
    return {"decision": "confirmed" if all_pass else "not_confirmed", "accepted": all_pass,
            "minimum_lower_ci_exclusive": MIN_LOWER_CI, "bootstrap_samples": BOOTSTRAPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "statistic": "median paired OFF/ON wave speedup; resampled independently within process-order strata",
            "widths": widths}


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        fail("usage: verify_adr049_b2_gemma4_aggregate_ab.py RECEIPT_DIR [SUMMARY_PATH]")
    root = Path(sys.argv[1]).resolve()
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    rows, processes = validate_manifest(root, manifest)
    analysis = analyze(validate_samples(root, rows, processes))
    summary = {"schema_version": SCHEMA, "status": "pass" if analysis["accepted"] else "fail",
               "manifest_sha256": sha256(manifest_path), "identity": manifest["identity"],
               "environment": manifest["environment"], "analysis": analysis}
    rendered = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    if len(sys.argv) == 3:
        destination = Path(sys.argv[2])
        if destination.exists():
            fail(f"refusing to overwrite summary: {destination}")
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(destination)
    else:
        sys.stdout.write(rendered)
    if not analysis["accepted"]:
        fail("one or more widths did not clear the immutable lower-95% speedup gate")


if __name__ == "__main__":
    main()
