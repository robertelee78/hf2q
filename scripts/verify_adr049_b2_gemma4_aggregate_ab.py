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

SCHEMA = 2
PAIRS = 8
WIDTHS = [128, 192, 256]
LANES = 4
PRIME_HISTORY_WORDS = 1200
MIN_PRIME_AGGREGATE_TOKENS = 4097
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 49_004
MIN_LOWER_CI = 1.05
TOOL_TURN_FIXED_TOKENS = 103
PAYLOAD_WORD_TOKENS = 2
MAX_TARGET_ROW_DRIFT = 4
TRACE_RE = re.compile(
    r"\[PREFILL_TIMING\] STABLE BATCHED ([0-9]+) seqs x "
    r"([0-9]+) boundary rows in ([0-9]+(?:\.[0-9]+)?) ms count=([0-9]+)"
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
        "prime_turns_per_wave": 4, "prime_history_words": PRIME_HISTORY_WORDS,
        "minimum_prime_aggregate_tokens": MIN_PRIME_AGGREGATE_TOKENS,
        "continuation_protocols": ["unary", "unary", "sse", "sse"],
        "tool_turn_fixed_tokens": TOOL_TURN_FIXED_TOKENS,
        "payload_word_tokens": PAYLOAD_WORD_TOKENS,
        "maximum_target_row_drift": MAX_TARGET_ROW_DRIFT,
        "off_env": {"HF2Q_CROSS_SLOT_ADMIT": "0", "HF2Q_ADMIT_COALESCE_US": "0"},
        "on_env": {"HF2Q_CROSS_SLOT_ADMIT": "1", "HF2Q_ADMIT_COALESCE_US": "25000"},
        "prime_request": {"max_tokens": 96, "seed": 42, "temperature": 0,
                          "repetition_penalty": 1, "stream": False,
                          "thinking": False, "tool_choice": "required"},
        "continuation_request": {"max_tokens": 32, "seed": 42, "temperature": 0,
                                 "repetition_penalty": 1, "tool_choice": "auto",
                                 "thinking": False},
        "semantic_normalization": "generated-call-ids-only",
        "wire_validation": "exact-envelope-single-choice-no-reasoning-logprobs-or-continuation-tools",
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


def normalized_call_ids(value: dict) -> dict:
    normalized = json.loads(json.dumps(value))
    for message in normalized.get("messages", []):
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict) and "id" in call:
                call["id"] = "<generated-call-id>"
        if message.get("role") == "tool" and "tool_call_id" in message:
            message["tool_call_id"] = "<generated-call-id>"
    for choice in normalized.get("choices", []):
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            continue
        for call in choice["message"].get("tool_calls") or []:
            if isinstance(call, dict) and "id" in call:
                call["id"] = "<generated-call-id>"
    return normalized


def prime_content(pair: int, target: int, lane: int) -> str:
    path = f"/tmp/adr049-p{pair:02d}-w{target:03d}-l{lane}.txt"
    return (
        f"Long agent history for pair {pair:02d} width {target:03d} lane {lane}. "
        + "history " * PRIME_HISTORY_WORDS
        + f"Call read_note exactly once with path {path}. "
        + "After the tool result, reply exactly ADR049_GEMMA_STABLE_OK."
    )


def expected_prime_request(model_id: str, pair: int, target: int, lane: int) -> dict:
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prime_content(pair, target, lane)}],
        "tools": [{"type": "function", "function": {
            "name": "read_note", "description": "Read one exact local note",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"], "additionalProperties": False},
        }}],
        "tool_choice": "required",
        "max_tokens": 96, "seed": 42, "temperature": 0, "repetition_penalty": 1,
        "stream": False, "hf2q_enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def validate_envelope(
    response: dict, *, required: set[str], optional: set[str], object_name: str,
    model_id: str, label: str,
) -> None:
    keys = set(response) if isinstance(response, dict) else set()
    if not required <= keys or keys - required - optional:
        fail(f"{label} envelope drifted")
    response_id, created = response.get("id"), response.get("created")
    if not isinstance(response_id, str) or not response_id \
            or not isinstance(created, int) or isinstance(created, bool) or created < 0 \
            or response.get("object") != object_name or response.get("model") != model_id \
            or ("system_fingerprint" in response
                and not isinstance(response["system_fingerprint"], str)):
        fail(f"{label} envelope drifted")


def validate_prime_response(
    response: dict, model_id: str, pair: int, target: int, lane: int
) -> tuple[int, int]:
    expected_path = f"/tmp/adr049-p{pair:02d}-w{target:03d}-l{lane}.txt"
    validate_envelope(
        response,
        required={"id", "object", "created", "model", "choices", "usage"},
        optional={"system_fingerprint", "x_hf2q_timing"},
        object_name="chat.completion", model_id=model_id,
        label=f"prime pair {pair} width {target} lane {lane}",
    )
    try:
        choices = response["choices"]
        choice = choices[0]
        message = choice["message"]
        calls = message["tool_calls"]
        call = calls[0]
        arguments = json.loads(call["function"]["arguments"])
        usage = response["usage"]
        prompt = usage["prompt_tokens"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        completion = usage["completion_tokens"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        fail(f"prime pair {pair} width {target} lane {lane} response is malformed")
    message_keys = set(message) if isinstance(message, dict) else set()
    if len(choices) != 1 or set(choice) != {"index", "message", "finish_reason"} \
            or choice.get("index") != 0 or choice.get("finish_reason") != "tool_calls" \
            or message_keys not in ({"role", "tool_calls"}, {"role", "content", "tool_calls"}) \
            or message.get("role") != "assistant" \
            or ("content" in message and not isinstance(message["content"], str)) \
            or len(calls) != 1 \
            or not isinstance(call.get("id"), str) or not call["id"] \
            or call.get("type") != "function" or call["function"].get("name") != "read_note" \
            or arguments != {"path": expected_path} \
            or not isinstance(prompt, int) or isinstance(prompt, bool) or prompt <= 0 \
            or cached != 0 or not isinstance(completion, int) or completion <= 0:
        fail(f"prime pair {pair} width {target} lane {lane} is not one exact cold tool call")
    return prompt, cached


def expected_prime_normalized(response: dict) -> dict:
    wrapped = normalized_call_ids({"choices": [response["choices"][0]]})
    return {
        "choice": wrapped["choices"][0],
        "completion_tokens": response["usage"]["completion_tokens"],
    }


def tool_result_content(target: int) -> str:
    words = (target - TOOL_TURN_FIXED_TOKENS) // PAYLOAD_WORD_TOKENS
    if words <= 0:
        fail(f"target {target} exhausts tool-turn calibration")
    return (
        "read_note succeeded. "
        + "measurement " * words
        + "Now reply exactly ADR049_GEMMA_STABLE_OK."
    )


def expected_continuation_request(
    prime_request: dict, prime_response: dict, target: int, stream: bool
) -> dict:
    expected = json.loads(json.dumps(prime_request))
    prior = json.loads(json.dumps(prime_response["choices"][0]["message"]))
    expected["messages"].extend([
        prior,
        {"role": "tool", "tool_call_id": prior["tool_calls"][0]["id"],
         "content": tool_result_content(target)},
    ])
    expected["tool_choice"] = "auto"
    expected["max_tokens"] = 32
    expected["stream"] = stream
    if stream:
        expected["stream_options"] = {"include_usage": True}
    else:
        expected.pop("stream_options", None)
    return expected


def canonical_unary(wire: dict, model_id: str) -> dict:
    validate_envelope(
        wire,
        required={"id", "object", "created", "model", "choices", "usage", "x_hf2q_timing"},
        optional={"system_fingerprint"}, object_name="chat.completion",
        model_id=model_id, label="unary wire response",
    )
    try:
        choices = wire["choices"]
        choice = choices[0]
        message = choice["message"]
        if len(choices) != 1 or set(choice) != {"index", "message", "finish_reason"} \
                or choice["index"] != 0 or choice["finish_reason"] != "stop" \
                or set(message) != {"role", "content"} \
                or message["role"] != "assistant" \
                or not isinstance(message["content"], str) \
                or not isinstance(wire["usage"], dict) \
                or not isinstance(wire["x_hf2q_timing"], dict):
            fail("unary wire response choice semantics drifted")
        return {
            "choices": [{"index": choice["index"], "message": message,
                         "finish_reason": choice["finish_reason"]}],
            "usage": wire["usage"],
            "x_hf2q_timing": wire["x_hf2q_timing"],
        }
    except (KeyError, IndexError, TypeError):
        fail("unary wire response lacks canonical fields")


def canonical_sse(wire_path: Path, events_path: Path, model_id: str) -> dict:
    try:
        lines = wire_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        fail(f"cannot read SSE wire response: {error}")
    data_lines = [line[len("data: "):] for line in lines if line.startswith("data: ")]
    if any(line and not line.startswith("data: ") for line in lines) \
            or data_lines.count("[DONE]") != 1 or not data_lines \
            or data_lines[-1] != "[DONE]":
        fail("SSE wire did not end with exactly one [DONE]")
    try:
        events = [json.loads(payload) for payload in data_lines[:-1]]
    except json.JSONDecodeError:
        fail("SSE wire contains malformed JSON")
    if not events or any(not isinstance(event, dict) for event in events) \
            or read_jsonl(events_path) != events:
        fail("SSE events are not raw-derived")
    if len(events) < 3:
        fail("SSE event sequence is incomplete")
    first = events[0]
    required = {"id", "object", "created", "model", "choices"}
    optional = {"system_fingerprint", "usage", "x_hf2q_timing"}
    for event_index, event in enumerate(events):
        validate_envelope(
            event, required=required, optional=optional,
            object_name="chat.completion.chunk", model_id=model_id,
            label=f"SSE event {event_index}",
        )
        if event["id"] != first["id"] or event["created"] != first["created"] \
                or event.get("system_fingerprint") != first.get("system_fingerprint"):
            fail("SSE envelope identity drifted within stream")
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            fail("SSE event choice cardinality drifted")
        choice = choices[0]
        if not isinstance(choice, dict) \
                or set(choice) != {"index", "delta", "finish_reason"} \
                or choice.get("index") != 0 or not isinstance(choice.get("delta"), dict):
            fail("SSE event choice/index/logprobs semantics drifted")
    first_choice = events[0]["choices"][0]
    final_choice = events[-1]["choices"][0]
    if set(first_choice["delta"]) != {"role"} \
            or first_choice["delta"]["role"] != "assistant" \
            or first_choice["finish_reason"] is not None \
            or "usage" in events[0] or "x_hf2q_timing" in events[0]:
        fail("SSE role event semantics drifted")
    for event in events[1:-1]:
        choice = event["choices"][0]
        if set(choice["delta"]) != {"content"} \
                or not isinstance(choice["delta"]["content"], str) \
                or choice["finish_reason"] is not None \
                or "usage" in event or "x_hf2q_timing" in event:
            fail("SSE content event semantics drifted")
    if final_choice["delta"] != {} or final_choice["finish_reason"] != "stop" \
            or set(events[-1]) - required - optional \
            or not isinstance(events[-1].get("usage"), dict) \
            or not isinstance(events[-1].get("x_hf2q_timing"), dict) \
            or sum("usage" in event for event in events) != 1 \
            or sum("x_hf2q_timing" in event for event in events) != 1:
        fail("SSE finish/usage/timing contract drifted")
    contents = [event["choices"][0]["delta"]["content"] for event in events[1:-1]]
    return {
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": "".join(contents)},
                     "finish_reason": "stop"}],
        "usage": events[-1]["usage"],
        "x_hf2q_timing": events[-1]["x_hf2q_timing"],
    }


def semantic_response(response: dict) -> dict:
    try:
        return {
            "choice": response["choices"][0],
            "usage": response["usage"],
        }
    except (KeyError, IndexError, TypeError):
        fail("response lacks normalized semantic fields")


def validate_samples(
    root: Path, rows: list[dict], processes: dict[tuple[int, str], dict]
) -> dict[tuple[int, int, str], dict]:
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
                required = {
                    "schema_version", "pair", "process_position", "arm", "width_position",
                    "target_rows", "wave_ms", "prime_aggregate_prompt_tokens",
                    "wave_wall_path", "wave_wall_sha256", "trace_path", "trace_sha256",
                    "trace_event_count", "trace_requests", "trace_boundary_rows",
                    "trace_elapsed_ms", "trace_batch_count", "aggregate_work_rows",
                    "launch_skew_seconds", "earliest_start", "latest_start",
                    "earliest_finish", "latest_finish",
                    "actual_overlap", "lanes_path", "lanes_sha256", "lanes",
                }
                if set(row) != required or row["schema_version"] != SCHEMA \
                        or (row["pair"], row["process_position"], row["arm"],
                            row["width_position"], row["target_rows"]) \
                        != (pair, position, arm, width_position, target):
                    fail(f"wave {cursor} schema or execution order drifted")
                wave_ms = finite(row["wave_ms"], f"wave {cursor} wall")
                wall_path = bind_file(
                    root, row["wave_wall_path"], row["wave_wall_sha256"], f"wave {cursor} wall"
                )
                try:
                    raw_wall_ms = float(wall_path.read_text(encoding="utf-8").strip()) * 1000
                except (OSError, UnicodeError, ValueError):
                    fail(f"wave {cursor} wall file is invalid")
                if abs(raw_wall_ms - wave_ms) > 1e-6:
                    fail(f"wave {cursor} wall is not raw-bound")

                trace_path = bind_file(
                    root, row["trace_path"], row["trace_sha256"], f"wave {cursor} trace"
                )
                trace_values = [
                    (int(requests), int(boundary), float(elapsed), int(count))
                    for requests, boundary, elapsed, count
                    in TRACE_RE.findall(trace_path.read_text(encoding="utf-8"))
                ]
                expected_events = 1 if arm == "on" else 0
                if row["trace_event_count"] != expected_events \
                        or len(trace_values) != expected_events:
                    fail(f"wave {cursor} stable-route reachability count is invalid")

                lanes_path = bind_file(
                    root, row["lanes_path"], row["lanes_sha256"], f"wave {cursor} lanes"
                )
                lanes = row["lanes"]
                if not isinstance(lanes, list) or len(lanes) != LANES \
                        or read_jsonl(lanes_path) != lanes:
                    fail(f"wave {cursor} does not contain four raw-bound lanes")
                starts: list[float] = []
                finishes: list[float] = []
                aggregate_rows = 0
                prime_prompts: list[int] = []
                cached_shapes: list[int] = []
                work_shapes: list[int] = []
                prime_request_hashes: set[str] = set()

                for lane_index, lane in enumerate(lanes):
                    protocol = "unary" if lane_index < 2 else "sse"
                    lane_required = {
                        "lane", "protocol", "prime_prompt_tokens", "prime_cached_tokens",
                        "prompt_tokens", "cached_tokens", "work_rows", "prefill_ms",
                        "ttft_ms", "wall_ms", "prime_request_path", "prime_request_sha256",
                        "prime_response_path", "prime_response_sha256",
                        "prime_normalized_path", "prime_normalized_sha256",
                        "request_path", "request_sha256", "request_normalized_path",
                        "request_normalized_sha256", "wire_response_path",
                        "wire_response_sha256", "sse_events_path", "sse_events_sha256",
                        "canonical_response_path", "canonical_response_sha256",
                        "wall_path", "wall_sha256", "timing_path", "timing_sha256",
                        "normalized_path", "normalized_sha256",
                    }
                    if set(lane) != lane_required or lane["lane"] != lane_index \
                            or lane["protocol"] != protocol:
                        fail(f"wave {cursor} lane {lane_index} schema/protocol is invalid")

                    prime_request_path = bind_file(
                        root, lane["prime_request_path"], lane["prime_request_sha256"],
                        "prime request",
                    )
                    prime_response_path = bind_file(
                        root, lane["prime_response_path"], lane["prime_response_sha256"],
                        "prime response",
                    )
                    prime_normalized_path = bind_file(
                        root, lane["prime_normalized_path"], lane["prime_normalized_sha256"],
                        "prime normalized",
                    )
                    prime_request = read_json(prime_request_path)
                    prime_response = read_json(prime_response_path)
                    if prime_request != expected_prime_request(
                        process["model_id"], pair, target, lane_index
                    ):
                        fail(f"wave {cursor} lane {lane_index} prime request drifted")
                    prime_prompt, prime_cached = validate_prime_response(
                        prime_response, process["model_id"], pair, target, lane_index
                    )
                    if (lane["prime_prompt_tokens"], lane["prime_cached_tokens"]) \
                            != (prime_prompt, prime_cached):
                        fail(f"wave {cursor} lane {lane_index} prime usage is not raw-bound")
                    if read_json(prime_normalized_path) != expected_prime_normalized(prime_response):
                        fail(f"wave {cursor} lane {lane_index} prime normalization drifted")
                    prime_prompts.append(prime_prompt)
                    prime_request_hashes.add(lane["prime_request_sha256"])

                    request_path = bind_file(
                        root, lane["request_path"], lane["request_sha256"], "continuation request"
                    )
                    request_normalized_path = bind_file(
                        root, lane["request_normalized_path"],
                        lane["request_normalized_sha256"], "continuation normalized request",
                    )
                    request = read_json(request_path)
                    stream = protocol == "sse"
                    if request != expected_continuation_request(
                        prime_request, prime_response, target, stream
                    ):
                        fail(f"wave {cursor} lane {lane_index} continuation does not carry "
                             "the exact prior assistant and matching tool result")
                    if read_json(request_normalized_path) != normalized_call_ids(request):
                        fail(f"wave {cursor} lane {lane_index} request normalization drifted")

                    wire_path = bind_file(
                        root, lane["wire_response_path"], lane["wire_response_sha256"],
                        "wire response",
                    )
                    events_path = bind_file(
                        root, lane["sse_events_path"], lane["sse_events_sha256"], "SSE events"
                    )
                    canonical_path = bind_file(
                        root, lane["canonical_response_path"],
                        lane["canonical_response_sha256"], "canonical response",
                    )
                    canonical = read_json(canonical_path)
                    if protocol == "unary":
                        if events_path.read_text(encoding="utf-8"):
                            fail(f"wave {cursor} unary lane carries SSE events")
                        derived = canonical_unary(read_json(wire_path), process["model_id"])
                    else:
                        derived = canonical_sse(wire_path, events_path, process["model_id"])
                    if canonical != derived:
                        fail(f"wave {cursor} lane {lane_index} canonical response is not wire-derived")
                    try:
                        choice = canonical["choices"][0]
                        message = choice["message"]
                        usage = canonical["usage"]
                        prompt = usage["prompt_tokens"]
                        cached = usage["prompt_tokens_details"]["cached_tokens"]
                        completion = usage["completion_tokens"]
                        timing = canonical["x_hf2q_timing"]
                        response_prefill = timing["prefill_time_secs"] * 1000
                        response_ttft = timing["time_to_first_token_ms"]
                    except (KeyError, IndexError, TypeError):
                        fail(f"wave {cursor} lane {lane_index} response timing is malformed")
                    work = prompt - cached
                    if choice.get("finish_reason") != "stop" \
                            or message.get("role") != "assistant" \
                            or message.get("content") != "ADR049_GEMMA_STABLE_OK" \
                            or message.get("tool_calls") not in (None, []) \
                            or not all(isinstance(value, int) and not isinstance(value, bool)
                                       for value in (prompt, cached, completion)) \
                            or cached <= 0 or completion <= 0 \
                            or work < 32 or work > 256 \
                            or abs(work - target) > MAX_TARGET_ROW_DRIFT \
                            or (lane["prompt_tokens"], lane["cached_tokens"], lane["work_rows"]) \
                            != (prompt, cached, work):
                        fail(f"wave {cursor} lane {lane_index} is not a cached stable "
                             "tool-result continuation in its target bin")
                    cached_shapes.append(cached)
                    work_shapes.append(work)
                    aggregate_rows += work
                    for name in ("prefill_ms", "ttft_ms", "wall_ms"):
                        finite(lane[name], f"wave {cursor} lane {lane_index} {name}")
                    if abs(float(response_prefill) - float(lane["prefill_ms"])) > 1e-9 \
                            or abs(float(response_ttft) - float(lane["ttft_ms"])) > 1e-9:
                        fail(f"wave {cursor} lane {lane_index} response timing is not raw-bound")

                    lane_wall_path = bind_file(
                        root, lane["wall_path"], lane["wall_sha256"], "lane wall"
                    )
                    timing_path = bind_file(
                        root, lane["timing_path"], lane["timing_sha256"], "lane timing"
                    )
                    normalized_path = bind_file(
                        root, lane["normalized_path"], lane["normalized_sha256"],
                        "lane normalized",
                    )
                    try:
                        lane_wall = float(lane_wall_path.read_text(encoding="utf-8").strip()) * 1000
                        start_text, finish_text = timing_path.read_text(
                            encoding="utf-8"
                        ).strip().split("\t")
                        start, finish = float(start_text), float(finish_text)
                    except (OSError, UnicodeError, ValueError):
                        fail(f"wave {cursor} lane {lane_index} timing file is malformed")
                    if abs(lane_wall - float(lane["wall_ms"])) > 1e-6 \
                            or not math.isfinite(start) or not math.isfinite(finish) \
                            or start <= 0 or finish <= start:
                        fail(f"wave {cursor} lane {lane_index} timing is not raw-bound")
                    starts.append(start)
                    finishes.append(finish)
                    if read_json(normalized_path) != semantic_response(canonical):
                        fail(f"wave {cursor} lane {lane_index} normalized result is not derived")

                if len(prime_request_hashes) != LANES:
                    fail(f"wave {cursor} did not prime four distinct conversations")
                if len(set(prime_prompts)) != 1:
                    fail(f"wave {cursor} prime turns were not equal-token-width")
                prime_aggregate = sum(prime_prompts)
                if prime_aggregate < MIN_PRIME_AGGREGATE_TOKENS \
                        or row["prime_aggregate_prompt_tokens"] != prime_aggregate:
                    fail(f"wave {cursor} did not prove a >4096-token aggregate prime history")
                if len(set(cached_shapes)) != 1 or len(set(work_shapes)) != 1:
                    fail(f"wave {cursor} stable continuations were not equal-shaped")

                earliest_start, latest_start = min(starts), max(starts)
                earliest_finish, latest_finish = min(finishes), max(finishes)
                skew = latest_start - earliest_start
                measured_wave_ms = (latest_finish - earliest_start) * 1000
                if row["actual_overlap"] is not True or skew > 0.100 \
                        or latest_start >= earliest_finish \
                        or abs(float(row["launch_skew_seconds"]) - skew) > 1e-6 \
                        or abs(float(row["earliest_start"]) - earliest_start) > 1e-6 \
                        or abs(float(row["latest_start"]) - latest_start) > 1e-6 \
                        or abs(float(row["earliest_finish"]) - earliest_finish) > 1e-6 \
                        or abs(float(row["latest_finish"]) - latest_finish) > 1e-6:
                    fail(f"wave {cursor} did not prove a simultaneous four-lane wave")
                if abs(wave_ms - measured_wave_ms) > 1e-6:
                    fail(f"wave {cursor} wall is not derived from concurrent lane timestamps")
                if row["aggregate_work_rows"] != aggregate_rows:
                    fail(f"wave {cursor} aggregate work is not lane-bound")
                if arm == "on":
                    requests, boundary, elapsed, count = trace_values[0]
                    if requests != LANES:
                        fail(f"wave {cursor} did not reach exactly one B4 stable rectangle")
                    if boundary < 32 or boundary > 256:
                        fail(f"wave {cursor} stable boundary is outside proven 32..256 range")
                    if elapsed <= 0 or count <= 0 or row["trace_requests"] != requests \
                            or row["trace_boundary_rows"] != boundary \
                            or abs(float(row["trace_elapsed_ms"]) - elapsed) > 1e-9 \
                            or row["trace_batch_count"] != count:
                        fail(f"wave {cursor} did not reach exactly one B4 stable rectangle")
                elif any(row[key] is not None for key in (
                    "trace_requests", "trace_boundary_rows", "trace_elapsed_ms",
                    "trace_batch_count",
                )):
                    fail(f"wave {cursor} OFF arm fabricated stable-route fields")
                samples[(pair, target, arm)] = row

    for pair in range(PAIRS):
        for target in WIDTHS:
            off, on = samples[(pair, target, "off")], samples[(pair, target, "on")]
            for lane_index, (off_lane, on_lane) in enumerate(zip(off["lanes"], on["lanes"])):
                if off_lane["prime_request_sha256"] != on_lane["prime_request_sha256"]:
                    fail(f"pair {pair} width {target} lane {lane_index} prime bytes differ")
                if off_lane["prime_normalized_sha256"] != on_lane["prime_normalized_sha256"]:
                    fail(f"pair {pair} width {target} lane {lane_index} prime results differ")
                if off_lane["request_normalized_sha256"] \
                        != on_lane["request_normalized_sha256"]:
                    fail(f"pair {pair} width {target} lane {lane_index} normalized continuation requests differ")
                if off_lane["normalized_sha256"] != on_lane["normalized_sha256"]:
                    fail(f"pair {pair} width {target} lane {lane_index} canonical results differ")
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
