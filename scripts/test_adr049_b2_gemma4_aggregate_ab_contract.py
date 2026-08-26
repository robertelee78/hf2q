#!/usr/bin/env python3
"""Model-free positive and mutation contract for the Gemma4 B.2 gate."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "bench_adr049_b2_gemma4_aggregate_ab.sh"
VERIFY = SCRIPT_DIR / "verify_adr049_b2_gemma4_aggregate_ab.py"
WIDTHS = [64, 128, 256]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{int(stat.st_mtime)}:{int(stat.st_ctime)}"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8")


def relative(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def single_quoted_line_continuations(source: str) -> list[int]:
    """Return lines where a literal backslash ends an open shell single quote."""
    violations: list[int] = []
    in_single = False
    in_double = False
    for line_number, line in enumerate(source.splitlines(), start=1):
        escaped = False
        for index, char in enumerate(line):
            if in_single:
                if char == "'":
                    in_single = False
                continue
            if in_double:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_double = False
                continue
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                in_single = True
            elif char == '"':
                in_double = True
            elif char == "#" and (index == 0 or line[index - 1] in " \t;|&("):
                break
        if in_single and line.endswith("\\"):
            violations.append(line_number)
    return violations


def assert_runner_shell_quoting_contract() -> None:
    runner_source = RUNNER.read_text(encoding="utf-8")
    violations = single_quoted_line_continuations(runner_source)
    assert not violations, (
        "runner embeds shell line continuations in single-quoted programs at lines "
        f"{violations}"
    )
    assert single_quoted_line_continuations("jq '.a \\\nand .b' file\n") == [1]
    assert single_quoted_line_continuations("jq '.a\nand .b' \\\nfile\n") == []
    assert "readonly PRIME_HISTORY_WORDS=1200" in runner_source
    assert "readonly MIN_PRIME_AGGREGATE_TOKENS=4097" in runner_source
    assert "readonly TOOL_RESULT_WORD_ADJUSTMENT=40" in runner_source
    assert "payload_words=$((target - TOOL_RESULT_WORD_ADJUSTMENT))" in runner_source
    assert "readonly MAX_TARGET_ROW_DRIFT=4" in runner_source
    assert "STABLE BATCHED 4 seqs x" in runner_source
    assert 'for (i = 1; i <= words; i++) printf "history "' in runner_source


def runner_function_source(name: str) -> str:
    lines = RUNNER.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{name}() {{")
    for end in range(start + 1, len(lines)):
        if lines[end] == "}":
            return "\n".join(lines[start:end + 1]) + "\n"
    raise AssertionError(f"runner function {name} has no closing brace")


def assert_runner_wire_helpers(valid: Path, scratch: Path) -> None:
    wave = valid / "processes" / "pair-0-off" / "wave-64"
    unary_output = scratch / "runner-unary.canonical.json"
    unary_events = scratch / "runner-unary.events.jsonl"
    sse_output = scratch / "runner-sse.canonical.json"
    sse_events = scratch / "runner-sse.events.jsonl"
    script = (
        "set -euo pipefail\n"
        + runner_function_source("validate_prime_response_wire")
        + runner_function_source("canonicalize_response")
        + 'validate_prime_response_wire "$1" /tmp/adr049-p00-w064-l0.txt fixture-gemma4\n'
        + 'validate_prime_response_wire "$2" /tmp/adr049-p00-w064-l1.txt fixture-gemma4\n'
        + 'canonicalize_response unary "$3" "$4" "$5" fixture-gemma4\n'
        + 'canonicalize_response sse "$6" "$7" "$8" fixture-gemma4\n'
    )
    subprocess.run(
        ["bash", "-s", "--",
         str(wave / "prime-lane-0.response.json"),
         str(wave / "prime-lane-1.response.json"),
         str(wave / "lane-0.response.wire"), str(unary_events), str(unary_output),
         str(wave / "lane-2.response.wire"), str(sse_events), str(sse_output)],
        input=script, text=True, check=True,
    )
    assert json.loads(unary_output.read_text()) == json.loads(
        (wave / "lane-0.response.canonical.json").read_text()
    )
    assert json.loads(sse_output.read_text()) == json.loads(
        (wave / "lane-2.response.canonical.json").read_text()
    )

    invalid = scratch / "runner-invalid-unary.json"
    wire = json.loads((wave / "lane-0.response.wire").read_text())
    wire["choices"][0]["message"]["reasoning_content"] = "must fail closed"
    write_json(invalid, wire)
    negative_script = (
        "set -euo pipefail\n" + runner_function_source("canonicalize_response")
        + 'canonicalize_response unary "$1" "$2" "$3" fixture-gemma4\n'
    )
    rejected = subprocess.run(
        ["bash", "-s", "--", str(invalid), str(scratch / "invalid.events"),
         str(scratch / "invalid.canonical")],
        input=negative_script, text=True, capture_output=True, check=False,
    )
    assert rejected.returncode != 0, "runner accepted unary reasoning_content drift"


def make_identity(root: Path) -> dict:
    source = root / "identity" / "source"
    scripts = source / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / "serve_gemma4_opencode.sh"
    launcher.write_text("#!/usr/bin/env bash\nexec \"$HF2Q_BIN\" \"$@\"\n", encoding="utf-8")
    launcher.chmod(0o755)
    (source / ".gitignore").write_text("/target/\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", ".gitignore", "scripts/serve_gemma4_opencode.sh"], check=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=hf2q-fixture",
        "-c", "user.email=fixture@hf2q.invalid", "commit", "-qm", "fixture",
    ], check=True)
    source_sha = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    binary = source / "target" / "release" / "hf2q"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixture sealed hf2q binary " + source_sha.encode() + b"\n")
    binary.chmod(0o755)
    model = root / "identity" / "fixture-gemma4.gguf"
    model.write_bytes(b"fixture-gemma4-model-content")
    return {
        "source_root": str(source), "source_sha": source_sha, "source_dirty": False,
        "binary_path": str(binary), "binary_sha256": digest(binary),
        "model_path": str(model), "model_sha256": digest(model),
        "model_bytes": model.stat().st_size, "model_snapshot": snapshot(model),
        "operator_launcher_path": str(launcher),
        "operator_launcher_sha256": digest(launcher),
    }


def prime_request(model_id: str, pair: int, target: int, lane: int) -> dict:
    expected_path = f"/tmp/adr049-p{pair:02d}-w{target:03d}-l{lane}.txt"
    content = (
        f"Long agent history for pair {pair:02d} width {target:03d} lane {lane}. "
        + "history " * 1200
        + f"Call read_note exactly once with path {expected_path}. "
        + "After the tool result, reply exactly ADR049_GEMMA_STABLE_OK."
    )
    return {
        "model": model_id, "messages": [{"role": "user", "content": content}],
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


def prime_response(pair: int, arm: str, target: int, lane: int) -> dict:
    expected_path = f"/tmp/adr049-p{pair:02d}-w{target:03d}-l{lane}.txt"
    message = {
        "role": "assistant",
        "tool_calls": [{"id": f"call-{pair}-{arm}-{target}-{lane}",
                        "type": "function",
                        "function": {"name": "read_note",
                                     "arguments": json.dumps({"path": expected_path})}}],
    }
    if lane % 2 == 0:
        message["content"] = ""
    return {
        "id": f"prime-{pair}-{arm}-{target}-{lane}",
        "object": "chat.completion", "created": 1700000000 + pair,
        "model": "fixture-gemma4",
        "choices": [{"index": 0,
                     "message": message,
                     "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1300, "completion_tokens": 12, "total_tokens": 1312,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }


def normalize_call_ids(value: dict) -> dict:
    value = json.loads(json.dumps(value))
    for message in value.get("messages", []):
        for call in message.get("tool_calls") or []:
            call["id"] = "<generated-call-id>"
        if message.get("role") == "tool":
            message["tool_call_id"] = "<generated-call-id>"
    for choice in value.get("choices", []):
        for call in choice.get("message", {}).get("tool_calls") or []:
            call["id"] = "<generated-call-id>"
    return value


def continuation_request(prime: dict, response: dict, target: int, stream: bool) -> dict:
    value = json.loads(json.dumps(prime))
    prior = json.loads(json.dumps(response["choices"][0]["message"]))
    tool_result = (
        "read_note succeeded. " + "measurement " * (target - 40)
        + "Now reply exactly ADR049_GEMMA_STABLE_OK."
    )
    value["messages"].extend([
        prior,
        {"role": "tool", "tool_call_id": prior["tool_calls"][0]["id"],
         "content": tool_result},
    ])
    value["tool_choice"] = "auto"
    value["max_tokens"] = 32
    value["stream"] = stream
    if stream:
        value["stream_options"] = {"include_usage": True}
    return value


def make_fixture(root: Path, speedup: float = 2.0) -> None:
    root.mkdir()
    root = root.resolve()
    (root / "processes").mkdir()
    identity = make_identity(root)
    model_verification = root / "model-verification.json"
    write_json(model_verification, {
        "schema_version": 2, "path": identity["model_path"],
        "sha256": identity["model_sha256"], "file_snapshot": identity["model_snapshot"],
        "file_stamp": {"fixture": True}, "content_hash_verified": True,
    })
    samples: list[dict] = []
    bindings: list[dict] = []
    sequence = 0
    for pair in range(8):
        arms = ["off", "on"] if pair % 2 == 0 else ["on", "off"]
        for position, arm in enumerate(arms):
            process_dir = root / "processes" / f"pair-{pair}-{arm}"
            process_dir.mkdir()
            model_id = "fixture-gemma4"
            models = process_dir / "models.json"
            stdout = process_dir / "server.stdout"
            stderr = process_dir / "server.stderr"
            command_file = process_dir / "server-command.txt"
            power = process_dir / "power.tsv"
            write_json(models, {"data": [{"id": model_id, "loaded": True, "arch": "gemma4"}]})
            stdout.write_text("fixture stdout\n", encoding="utf-8")
            command_text = (
                f"{identity['binary_path']} -v serve --model {identity['model_path']} "
                "--scheduler inflight-batched --max-slots 4"
            )
            command_file.write_text(command_text + "\n", encoding="utf-8")
            power.write_text("".join(
                f"{1500 + pair * 20 + position * 10 + offset}\tac\tautomatic\t0\t"
                f"pair-{pair}-{arm}-{phase}\n"
                for offset, phase in enumerate((
                    "before-launch", "loaded-warm", "measurement-start",
                    "measurement-end", "after-shutdown",
                ))
            ), encoding="utf-8")
            stderr_lines: list[str] = []
            for width_position, target in enumerate(WIDTHS):
                wave_dir = process_dir / f"wave-{target}"
                wave_dir.mkdir()
                lanes: list[dict] = []
                starts: list[float] = []
                finishes: list[float] = []
                aggregate_rows = 0
                for lane_index in range(4):
                    protocol = "unary" if lane_index < 2 else "sse"
                    prime_req = prime_request(model_id, pair, target, lane_index)
                    prime_resp = prime_response(pair, arm, target, lane_index)
                    prime_request_path = wave_dir / f"prime-lane-{lane_index}.request.json"
                    prime_response_path = wave_dir / f"prime-lane-{lane_index}.response.json"
                    prime_normalized_path = wave_dir / f"prime-lane-{lane_index}.normalized.json"
                    write_json(prime_request_path, prime_req)
                    write_json(prime_response_path, prime_resp)
                    normalized_prime = {
                        "choice": normalize_call_ids(
                            {"choices": [prime_resp["choices"][0]]}
                        )["choices"][0],
                        "completion_tokens": prime_resp["usage"]["completion_tokens"],
                    }
                    write_json(prime_normalized_path, normalized_prime)

                    request_value = continuation_request(
                        prime_req, prime_resp, target, protocol == "sse"
                    )
                    request = wave_dir / f"lane-{lane_index}.request.json"
                    request_normalized = wave_dir / f"lane-{lane_index}.request.normalized.json"
                    write_json(request, request_value)
                    write_json(request_normalized, normalize_call_ids(request_value))

                    cached = 1300
                    prompt = cached + target
                    prefill_ms = 20 + target * 0.05
                    ttft_ms = prefill_ms + 1
                    lane_wall_ms = prefill_ms + 2
                    usage = {
                        "prompt_tokens": prompt, "completion_tokens": 5,
                        "total_tokens": prompt + 5,
                        "prompt_tokens_details": {"cached_tokens": cached},
                    }
                    timing_value = {
                        "prefill_time_secs": prefill_ms / 1000,
                        "time_to_first_token_ms": ttft_ms,
                    }
                    canonical_value = {
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant",
                                                 "content": "ADR049_GEMMA_STABLE_OK"},
                                     "finish_reason": "stop"}],
                        "usage": usage, "x_hf2q_timing": timing_value,
                    }
                    wire = wave_dir / f"lane-{lane_index}.response.wire"
                    events = wave_dir / f"lane-{lane_index}.response.events.jsonl"
                    canonical = wave_dir / f"lane-{lane_index}.response.canonical.json"
                    response_id = f"continuation-{pair}-{arm}-{target}-{lane_index}"
                    created = 1800000000 + sequence
                    if protocol == "unary":
                        write_json(wire, {
                            "id": response_id, "object": "chat.completion",
                            "created": created, "model": model_id,
                            **canonical_value,
                        })
                        events.write_text("", encoding="utf-8")
                    else:
                        event_values = [
                            {"id": response_id, "object": "chat.completion.chunk",
                             "created": created, "model": model_id,
                             "choices": [{"index": 0, "delta": {"role": "assistant"},
                                           "finish_reason": None}]},
                            {"id": response_id, "object": "chat.completion.chunk",
                             "created": created, "model": model_id,
                             "choices": [{"index": 0,
                                           "delta": {"content": "ADR049_GEMMA_STABLE_OK"},
                                           "finish_reason": None}]},
                            {"id": response_id, "object": "chat.completion.chunk",
                             "created": created, "model": model_id,
                             "choices": [{"index": 0, "delta": {},
                                           "finish_reason": "stop"}],
                             "usage": usage, "x_hf2q_timing": timing_value},
                        ]
                        wire.write_text(
                            "".join(
                                f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
                                for event in event_values
                            ) + "data: [DONE]\n\n",
                            encoding="utf-8",
                        )
                        write_jsonl(events, event_values)
                    write_json(canonical, canonical_value)
                    normalized = wave_dir / f"lane-{lane_index}.normalized.json"
                    write_json(normalized, {
                        "choice": canonical_value["choices"][0], "usage": usage,
                    })
                    wall = wave_dir / f"lane-{lane_index}.wall"
                    timing = wave_dir / f"lane-{lane_index}.timing"
                    wall.write_text(f"{lane_wall_ms / 1000:.12f}\n", encoding="utf-8")
                    base_start = 3000 + sequence * 6
                    start = base_start + lane_index * 0.01
                    desired_wave_ms = (1000 + target) * (speedup if arm == "off" else 1.0)
                    finish = base_start + desired_wave_ms / 1000
                    timing.write_text(f"{start:.9f}\t{finish:.9f}\n", encoding="utf-8")
                    starts.append(start)
                    finishes.append(finish)
                    lanes.append({
                        "lane": lane_index, "protocol": protocol,
                        "prime_prompt_tokens": 1300, "prime_cached_tokens": 0,
                        "prompt_tokens": prompt, "cached_tokens": cached,
                        "work_rows": target, "prefill_ms": prefill_ms,
                        "ttft_ms": ttft_ms, "wall_ms": lane_wall_ms,
                        "prime_request_path": relative(root, prime_request_path),
                        "prime_request_sha256": digest(prime_request_path),
                        "prime_response_path": relative(root, prime_response_path),
                        "prime_response_sha256": digest(prime_response_path),
                        "prime_normalized_path": relative(root, prime_normalized_path),
                        "prime_normalized_sha256": digest(prime_normalized_path),
                        "request_path": relative(root, request),
                        "request_sha256": digest(request),
                        "request_normalized_path": relative(root, request_normalized),
                        "request_normalized_sha256": digest(request_normalized),
                        "wire_response_path": relative(root, wire),
                        "wire_response_sha256": digest(wire),
                        "sse_events_path": relative(root, events),
                        "sse_events_sha256": digest(events),
                        "canonical_response_path": relative(root, canonical),
                        "canonical_response_sha256": digest(canonical),
                        "wall_path": relative(root, wall), "wall_sha256": digest(wall),
                        "timing_path": relative(root, timing), "timing_sha256": digest(timing),
                        "normalized_path": relative(root, normalized),
                        "normalized_sha256": digest(normalized),
                    })
                    aggregate_rows += target

                lanes_file = wave_dir / "lanes.jsonl"
                write_jsonl(lanes_file, lanes)
                wave_ms = (max(finishes) - min(starts)) * 1000
                wave_wall = wave_dir / "wave.wall"
                wave_wall.write_text(f"{wave_ms / 1000:.12f}\n", encoding="utf-8")
                trace = wave_dir / "server.trace.log"
                if arm == "on":
                    trace_count = sequence + 1
                    boundary_rows = target - 5
                    line = (
                        f"[PREFILL_TIMING] STABLE BATCHED 4 seqs x {boundary_rows} "
                        f"boundary rows in 10.5 ms count={trace_count}\n"
                    )
                    trace.write_text(line, encoding="utf-8")
                    stderr_lines.append(line)
                    event_count, trace_requests, trace_rows = 1, 4, boundary_rows
                    trace_elapsed, trace_batch_count = 10.5, trace_count
                else:
                    trace.write_text("OFF stable scalar continuation\n", encoding="utf-8")
                    event_count, trace_requests, trace_rows = 0, None, None
                    trace_elapsed, trace_batch_count = None, None
                samples.append({
                    "schema_version": 2, "pair": pair, "process_position": position,
                    "arm": arm, "width_position": width_position, "target_rows": target,
                    "wave_ms": wave_ms, "prime_aggregate_prompt_tokens": 5200,
                    "wave_wall_path": relative(root, wave_wall),
                    "wave_wall_sha256": digest(wave_wall),
                    "trace_path": relative(root, trace), "trace_sha256": digest(trace),
                    "trace_event_count": event_count, "trace_requests": trace_requests,
                    "trace_boundary_rows": trace_rows, "trace_elapsed_ms": trace_elapsed,
                    "trace_batch_count": trace_batch_count,
                    "aggregate_work_rows": aggregate_rows,
                    "launch_skew_seconds": max(starts) - min(starts),
                    "earliest_start": min(starts), "latest_start": max(starts),
                    "earliest_finish": min(finishes), "latest_finish": max(finishes),
                    "actual_overlap": True, "lanes_path": relative(root, lanes_file),
                    "lanes_sha256": digest(lanes_file), "lanes": lanes,
                })
                sequence += 1
            stderr.write_text("".join(stderr_lines) or "OFF server\n", encoding="utf-8")
            runtime_home = process_dir / "runtime-home"
            runtime_home.mkdir()
            process_record = {
                "schema_version": 2, "status": "stopped", "pair": pair,
                "position": position, "arm": arm, "pid": 1000 + pair * 2 + position,
                "command": command_text, "model_id": model_id, "max_slots": 4,
                "runtime": {"clean_environment": True, "home": str(runtime_home),
                            "path": "/usr/bin:/bin:/usr/sbin:/sbin", "tmpdir": "/var/tmp",
                            "locale": {"LANG": "C", "LC_ALL": "C"}, "rust_backtrace": "1",
                            "operator_launcher": identity["operator_launcher_path"],
                            "operator_launcher_sha256": identity["operator_launcher_sha256"],
                            "model_verification_receipt": str(model_verification),
                            "model_verification_receipt_sha256": digest(model_verification)},
                "lever_env": {"HF2Q_CROSS_SLOT_ADMIT": "1" if arm == "on" else "0",
                              "HF2Q_ADMIT_COALESCE_US": "25000" if arm == "on" else "0"},
                "source_sha": identity["source_sha"],
                "binary_sha256": identity["binary_sha256"],
                "model_sha256": identity["model_sha256"], "wait_status": 143,
                "power_path": relative(root, power), "power_sha256": digest(power),
                "command_path": relative(root, command_file),
                "command_sha256": digest(command_file),
                "models_path": relative(root, models), "models_sha256": digest(models),
                "stdout_path": relative(root, stdout), "stdout_sha256": digest(stdout),
                "stderr_path": relative(root, stderr), "stderr_sha256": digest(stderr),
            }
            record_path = process_dir / "process.json"
            write_json(record_path, process_record)
            bindings.append({"pair": pair, "position": position, "arm": arm,
                             "path": relative(root, record_path), "sha256": digest(record_path)})

    samples_path, bindings_path = root / "samples.jsonl", root / "processes.jsonl"
    write_jsonl(samples_path, samples)
    write_jsonl(bindings_path, bindings)
    settle = root / "thermal-settle.log"
    contention_settle = root / "contention-settle.log"
    settle.write_text("".join(
        f"{1000 + offset}\tnominal\tadr049-b2-gemma-ab-settle\n"
        for offset in range(0, 61, 5)
    ), encoding="utf-8")
    contention_settle.write_text("".join(
        f"{1000 + offset}\tquiet\tadr049-b2-gemma-ab-settle\t100\t-\n"
        for offset in range(0, 61, 5)
    ), encoding="utf-8")
    measurement = root / "thermal-measurement.log"
    contention_measurement = root / "contention-measurement.log"
    measurement.write_text(
        "2000\tnominal\tadr049-b2-gemma-ab-start\n"
        "2002\tfair\tadr049-b2-gemma-ab-measurement\n"
        "2004\tfair\tadr049-b2-gemma-ab-end\n", encoding="utf-8")
    contention_measurement.write_text(
        "2000\tquiet\tadr049-b2-gemma-ab-start\t100\t-\n"
        "2002\tquiet\tadr049-b2-gemma-ab-measurement\t100\t-\n"
        "2004\tquiet\tadr049-b2-gemma-ab-end\t100\t-\n", encoding="utf-8")
    guard_files = {}
    for key, name, content in (
        ("caffeinate_log", "caffeinate.log", ""),
        ("assertions", "caffeinate.log.assertions",
         "pid 999(caffeinate) PreventUserIdleSystemSleep\n"),
        ("events_baseline", "caffeinate.log.power-events.baseline", "baseline event\n"),
        ("events_final", "caffeinate.log.power-events.final", "baseline event\n"),
        ("events_new", "caffeinate.log.power-events.new", ""),
    ):
        path = root / name
        path.write_text(content, encoding="utf-8")
        guard_files[key] = {"path": name, "sha256": digest(path)}
    files = {
        "samples": {"path": "samples.jsonl", "sha256": digest(samples_path)},
        "model_verification": {"path": "model-verification.json",
                               "sha256": digest(model_verification)},
        "process_bindings": {"path": "processes.jsonl", "sha256": digest(bindings_path)},
        "thermal_settle": {"path": settle.name, "sha256": digest(settle)},
        "thermal_measurement": {"path": measurement.name, "sha256": digest(measurement)},
        "contention_settle": {"path": contention_settle.name,
                              "sha256": digest(contention_settle)},
        "contention_measurement": {"path": contention_measurement.name,
                                   "sha256": digest(contention_measurement)},
        "power_guard": guard_files,
    }
    write_json(root / "manifest.json", {
        "schema_version": 2, "status": "measured",
        "configuration": {
            "pairs": 8, "width_targets": WIDTHS, "lanes": 4,
            "pair_order": "off-on-even_on-off-odd", "warmup_waves_per_process": 2,
            "measured_waves_per_process": 3, "prime_turns_per_wave": 4,
            "prime_history_words": 1200, "minimum_prime_aggregate_tokens": 4097,
            "continuation_protocols": ["unary", "unary", "sse", "sse"],
            "tool_result_word_adjustment": 40, "maximum_target_row_drift": 4,
            "off_env": {"HF2Q_CROSS_SLOT_ADMIT": "0", "HF2Q_ADMIT_COALESCE_US": "0"},
            "on_env": {"HF2Q_CROSS_SLOT_ADMIT": "1",
                       "HF2Q_ADMIT_COALESCE_US": "25000"},
            "prime_request": {"max_tokens": 96, "seed": 42, "temperature": 0,
                              "repetition_penalty": 1, "stream": False,
                              "thinking": False, "tool_choice": "required"},
            "continuation_request": {"max_tokens": 32, "seed": 42, "temperature": 0,
                                     "repetition_penalty": 1, "tool_choice": "auto",
                                     "thinking": False},
            "semantic_normalization": "generated-call-ids-only",
            "wire_validation": "exact-envelope-single-choice-no-reasoning-logprobs-or-continuation-tools",
            "analysis": {"statistic": "median paired OFF/ON wave speedup",
                         "order_stratified_bootstrap_samples": 10000,
                         "bootstrap_seed": 49004, "lower_confidence_percentile": 2.5,
                         "minimum_lower_95_speedup_exclusive": 1.05},
        },
        "identity": identity,
        "environment": {"power": "ac", "power_mode": "automatic",
                        "power_mode_code": "0",
                        "thermal": "nominal-settle-and-fair-or-better-measurement",
                        "host_contention": "quiet", "clean_process_environment": True},
        "processes": bindings, "files": files,
    })

def run_verify(root: Path, *, success: bool, reason: str | None = None) -> None:
    result = subprocess.run([str(VERIFY), str(root)], text=True, capture_output=True, check=False)
    if (result.returncode == 0) != success:
        raise AssertionError(
            f"verifier {'rejected valid' if success else 'accepted invalid'} fixture:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    if reason is not None and reason not in result.stderr:
        raise AssertionError(
            f"verifier rejected for the wrong reason; expected {reason!r}:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )


def reseal_top(root: Path, key: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    binding = manifest["files"][key]
    binding["sha256"] = digest(root / binding["path"])
    write_json(root / "manifest.json", manifest)


def reseal_samples(root: Path, rows: list[dict]) -> None:
    write_jsonl(root / "samples.jsonl", rows)
    reseal_top(root, "samples")


def reseal_row(root: Path, rows: list[dict], row: dict) -> None:
    lanes_path = root / row["lanes_path"]
    write_jsonl(lanes_path, row["lanes"])
    row["lanes_sha256"] = digest(lanes_path)
    reseal_samples(root, rows)


def rewrite_sse_events(
    root: Path, rows: list[dict], row: dict, lane: dict, events: list[dict]
) -> None:
    events_path = root / lane["sse_events_path"]
    wire_path = root / lane["wire_response_path"]
    write_jsonl(events_path, events)
    wire_path.write_text(
        "".join(
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            for event in events
        ) + "data: [DONE]\n\n",
        encoding="utf-8",
    )
    lane["sse_events_sha256"] = digest(events_path)
    lane["wire_response_sha256"] = digest(wire_path)
    reseal_row(root, rows, row)


def reseal_process(root: Path, index: int, record: dict) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest["processes"][index]
    path = root / binding["path"]
    write_json(path, record)
    binding["sha256"] = digest(path)
    write_jsonl(root / "processes.jsonl", manifest["processes"])
    manifest["files"]["process_bindings"]["sha256"] = digest(root / "processes.jsonl")
    write_json(manifest_path, manifest)


def clone(source: Path, parent: Path, name: str) -> Path:
    destination = parent / name
    shutil.copytree(source, destination)
    summary = destination / "summary.json"
    if summary.exists():
        summary.unlink()
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = str((destination / "model-verification.json").resolve())
    for binding in manifest["processes"]:
        record_path = destination / binding["path"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["runtime"]["model_verification_receipt"] = receipt
        record["runtime"]["home"] = str(
            (destination / "processes" / f"pair-{record['pair']}-{record['arm']}" / "runtime-home").resolve()
        )
        write_json(record_path, record)
        binding["sha256"] = digest(record_path)
    write_jsonl(destination / "processes.jsonl", manifest["processes"])
    manifest["files"]["process_bindings"]["sha256"] = digest(destination / "processes.jsonl")
    write_json(manifest_path, manifest)
    return destination


def main() -> None:
    assert_runner_shell_quoting_contract()
    rejected = 0
    with tempfile.TemporaryDirectory(prefix="hf2q-adr049-gemma-ab-") as temporary:
        parent = Path(temporary)
        valid = parent / "valid"
        make_fixture(valid)
        assert_runner_wire_helpers(valid, parent)
        valid_rows = [json.loads(line) for line in
                      (valid / "samples.jsonl").read_text().splitlines()]
        assert all(
            row["trace_boundary_rows"] != row["lanes"][0]["work_rows"]
            for row in valid_rows if row["arm"] == "on"
        ), "valid fixture must distinguish stable-boundary rows from usage work"
        summary = valid / "summary.json"
        subprocess.run([str(VERIFY), str(valid), str(summary)], check=True)
        result = json.loads(summary.read_text(encoding="utf-8"))
        assert result["status"] == "pass" and result["analysis"]["decision"] == "confirmed"
        run_verify(valid, success=True)

        slow = parent / "slow"
        make_fixture(slow, speedup=1.02)
        run_verify(slow, success=False, reason="immutable lower-95% speedup gate")
        rejected += 1

        bad_hash = clone(valid, parent, "bad-hash")
        with (bad_hash / "samples.jsonl").open("a", encoding="utf-8") as target_file:
            target_file.write("\n")
        run_verify(bad_hash, success=False, reason="samples file identity failed")
        rejected += 1

        bad_trace = clone(valid, parent, "bad-trace")
        rows = [json.loads(line) for line in
                (bad_trace / "samples.jsonl").read_text().splitlines()]
        on_row = next(row for row in rows if row["arm"] == "on")
        trace_path = bad_trace / on_row["trace_path"]
        trace_path.write_text(
            trace_path.read_text().replace("STABLE BATCHED 4", "STABLE BATCHED 3"),
            encoding="utf-8",
        )
        on_row["trace_sha256"] = digest(trace_path)
        reseal_samples(bad_trace, rows)
        run_verify(bad_trace, success=False, reason="exactly one B4 stable rectangle")
        rejected += 1

        boundary_range = clone(valid, parent, "boundary-outside-range")
        rows = [json.loads(line) for line in
                (boundary_range / "samples.jsonl").read_text().splitlines()]
        row = next(row for row in rows if row["arm"] == "on" and row["target_rows"] == 64)
        trace_path = boundary_range / row["trace_path"]
        trace_path.write_text(
            trace_path.read_text().replace("x 59 boundary rows", "x 31 boundary rows"),
            encoding="utf-8",
        )
        row["trace_boundary_rows"] = 31
        row["trace_sha256"] = digest(trace_path)
        reseal_samples(boundary_range, rows)
        run_verify(
            boundary_range, success=False,
            reason="stable boundary is outside proven 32..256 range",
        )
        rejected += 1

        wave_scope = clone(valid, parent, "wave-scope-drift")
        rows = [json.loads(line) for line in
                (wave_scope / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        row["wave_ms"] += 250
        wall_path = wave_scope / row["wave_wall_path"]
        wall_path.write_text(f"{row['wave_ms'] / 1000:.12f}\n", encoding="utf-8")
        row["wave_wall_sha256"] = digest(wall_path)
        reseal_samples(wave_scope, rows)
        run_verify(
            wave_scope, success=False,
            reason="wall is not derived from concurrent lane timestamps",
        )
        rejected += 1

        extra_choice = clone(valid, parent, "unary-extra-choice")
        rows = [json.loads(line) for line in
                (extra_choice / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][0]
        wire_path = extra_choice / lane["wire_response_path"]
        wire = json.loads(wire_path.read_text())
        wire["choices"].append(json.loads(json.dumps(wire["choices"][0])))
        write_json(wire_path, wire)
        lane["wire_response_sha256"] = digest(wire_path)
        reseal_row(extra_choice, rows, row)
        run_verify(extra_choice, success=False, reason="choice semantics drifted")
        rejected += 1

        unary_reasoning = clone(valid, parent, "unary-reasoning")
        rows = [json.loads(line) for line in
                (unary_reasoning / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][0]
        wire_path = unary_reasoning / lane["wire_response_path"]
        wire = json.loads(wire_path.read_text())
        wire["choices"][0]["message"]["reasoning_content"] = "hidden drift"
        write_json(wire_path, wire)
        lane["wire_response_sha256"] = digest(wire_path)
        reseal_row(unary_reasoning, rows, row)
        run_verify(unary_reasoning, success=False, reason="choice semantics drifted")
        rejected += 1

        unary_logprobs = clone(valid, parent, "unary-logprobs")
        rows = [json.loads(line) for line in
                (unary_logprobs / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][0]
        wire_path = unary_logprobs / lane["wire_response_path"]
        wire = json.loads(wire_path.read_text())
        wire["choices"][0]["logprobs"] = {"content": []}
        write_json(wire_path, wire)
        lane["wire_response_sha256"] = digest(wire_path)
        reseal_row(unary_logprobs, rows, row)
        run_verify(unary_logprobs, success=False, reason="choice semantics drifted")
        rejected += 1

        unary_index = clone(valid, parent, "unary-index")
        rows = [json.loads(line) for line in
                (unary_index / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][0]
        wire_path = unary_index / lane["wire_response_path"]
        wire = json.loads(wire_path.read_text())
        wire["choices"][0]["index"] = 1
        write_json(wire_path, wire)
        lane["wire_response_sha256"] = digest(wire_path)
        reseal_row(unary_index, rows, row)
        run_verify(unary_index, success=False, reason="choice semantics drifted")
        rejected += 1

        unary_envelope = clone(valid, parent, "unary-envelope")
        rows = [json.loads(line) for line in
                (unary_envelope / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][0]
        wire_path = unary_envelope / lane["wire_response_path"]
        wire = json.loads(wire_path.read_text())
        wire["unsealed_transport_field"] = True
        write_json(wire_path, wire)
        lane["wire_response_sha256"] = digest(wire_path)
        reseal_row(unary_envelope, rows, row)
        run_verify(unary_envelope, success=False, reason="envelope drifted")
        rejected += 1

        prime_reasoning = clone(valid, parent, "prime-reasoning")
        rows = [json.loads(line) for line in
                (prime_reasoning / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][0]
        prime_path = prime_reasoning / lane["prime_response_path"]
        prime = json.loads(prime_path.read_text())
        prime["choices"][0]["message"]["reasoning_content"] = "replay would discard me"
        write_json(prime_path, prime)
        lane["prime_response_sha256"] = digest(prime_path)
        reseal_row(prime_reasoning, rows, row)
        run_verify(prime_reasoning, success=False, reason="one exact cold tool call")
        rejected += 1

        sse_reasoning = clone(valid, parent, "sse-reasoning")
        rows = [json.loads(line) for line in
                (sse_reasoning / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][2]
        events = [json.loads(line) for line in
                  (sse_reasoning / lane["sse_events_path"]).read_text().splitlines()]
        events[1]["choices"][0]["delta"]["reasoning_content"] = "hidden drift"
        rewrite_sse_events(sse_reasoning, rows, row, lane, events)
        run_verify(sse_reasoning, success=False, reason="content event semantics drifted")
        rejected += 1

        sse_logprobs = clone(valid, parent, "sse-logprobs")
        rows = [json.loads(line) for line in
                (sse_logprobs / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][2]
        events = [json.loads(line) for line in
                  (sse_logprobs / lane["sse_events_path"]).read_text().splitlines()]
        events[1]["choices"][0]["logprobs"] = {"content": []}
        rewrite_sse_events(sse_logprobs, rows, row, lane, events)
        run_verify(sse_logprobs, success=False, reason="choice/index/logprobs semantics drifted")
        rejected += 1

        sse_index = clone(valid, parent, "sse-index")
        rows = [json.loads(line) for line in
                (sse_index / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][2]
        events = [json.loads(line) for line in
                  (sse_index / lane["sse_events_path"]).read_text().splitlines()]
        events[1]["choices"][0]["index"] = 1
        rewrite_sse_events(sse_index, rows, row, lane, events)
        run_verify(sse_index, success=False, reason="choice/index/logprobs semantics drifted")
        rejected += 1

        sse_envelope = clone(valid, parent, "sse-envelope")
        rows = [json.loads(line) for line in
                (sse_envelope / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][2]
        events = [json.loads(line) for line in
                  (sse_envelope / lane["sse_events_path"]).read_text().splitlines()]
        events[1]["id"] = "different-stream-id"
        rewrite_sse_events(sse_envelope, rows, row, lane, events)
        run_verify(sse_envelope, success=False, reason="identity drifted within stream")
        rejected += 1

        duplicate = clone(valid, parent, "duplicate-process")
        manifest = json.loads((duplicate / "manifest.json").read_text())
        first = json.loads((duplicate / manifest["processes"][0]["path"]).read_text())
        second = json.loads((duplicate / manifest["processes"][1]["path"]).read_text())
        second["pid"] = first["pid"]
        reseal_process(duplicate, 1, second)
        run_verify(duplicate, success=False, reason="reused or omitted a PID")
        rejected += 1

        output_drift = clone(valid, parent, "output-drift")
        rows = [json.loads(line) for line in
                (output_drift / "samples.jsonl").read_text().splitlines()]
        row = next(row for row in rows if row["arm"] == "on")
        lane = row["lanes"][0]
        for key in ("wire_response_path", "canonical_response_path"):
            path = output_drift / lane[key]
            value = json.loads(path.read_text())
            value["usage"]["completion_tokens"] += 1
            value["usage"]["total_tokens"] += 1
            write_json(path, value)
            lane[key.replace("_path", "_sha256")] = digest(path)
        normalized_path = output_drift / lane["normalized_path"]
        normalized = json.loads(normalized_path.read_text())
        normalized["usage"]["completion_tokens"] += 1
        normalized["usage"]["total_tokens"] += 1
        write_json(normalized_path, normalized)
        lane["normalized_sha256"] = digest(normalized_path)
        reseal_row(output_drift, rows, row)
        run_verify(output_drift, success=False, reason="canonical results differ")
        rejected += 1

        bad_thermal = clone(valid, parent, "bad-thermal")
        thermal = bad_thermal / "thermal-measurement.log"
        thermal.write_text(
            thermal.read_text().replace("2000\tnominal", "2000\tfair", 1),
            encoding="utf-8",
        )
        reseal_top(bad_thermal, "thermal_measurement")
        run_verify(bad_thermal, success=False, reason="measurement thermal sentinels")
        rejected += 1

        wrong_link = clone(valid, parent, "wrong-tool-link")
        rows = [json.loads(line) for line in
                (wrong_link / "samples.jsonl").read_text().splitlines()]
        row = next(row for row in rows if row["arm"] == "on")
        lane = row["lanes"][0]
        request_path = wrong_link / lane["request_path"]
        request = json.loads(request_path.read_text())
        request["messages"][-1]["tool_call_id"] = "wrong-generated-id"
        write_json(request_path, request)
        lane["request_sha256"] = digest(request_path)
        reseal_row(wrong_link, rows, row)
        run_verify(
            wrong_link, success=False,
            reason="exact prior assistant and matching tool result",
        )
        rejected += 1

        no_overlap = clone(valid, parent, "no-overlap")
        rows = [json.loads(line) for line in
                (no_overlap / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][3]
        timing_path = no_overlap / lane["timing_path"]
        timing_path.write_text("4000.000000000\t4000.500000000\n", encoding="utf-8")
        lane["timing_sha256"] = digest(timing_path)
        row["launch_skew_seconds"] = 1000.0
        row["latest_start"] = 4000.0
        row["actual_overlap"] = False
        reseal_row(no_overlap, rows, row)
        run_verify(no_overlap, success=False, reason="simultaneous four-lane wave")
        rejected += 1

        zero_cached = clone(valid, parent, "zero-cached")
        rows = [json.loads(line) for line in
                (zero_cached / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][0]
        for key in ("wire_response_path", "canonical_response_path"):
            path = zero_cached / lane[key]
            value = json.loads(path.read_text())
            value["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
            write_json(path, value)
            lane[key.replace("_path", "_sha256")] = digest(path)
        normalized_path = zero_cached / lane["normalized_path"]
        normalized = json.loads(normalized_path.read_text())
        normalized["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
        write_json(normalized_path, normalized)
        lane["normalized_sha256"] = digest(normalized_path)
        lane["cached_tokens"] = 0
        lane["work_rows"] = lane["prompt_tokens"]
        row["aggregate_work_rows"] += lane["prompt_tokens"] - 1300 - 64
        reseal_row(zero_cached, rows, row)
        run_verify(zero_cached, success=False, reason="cached stable tool-result continuation")
        rejected += 1

        short_prime = clone(valid, parent, "short-prime")
        rows = [json.loads(line) for line in
                (short_prime / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        for lane in row["lanes"]:
            response_path = short_prime / lane["prime_response_path"]
            response = json.loads(response_path.read_text())
            response["usage"]["prompt_tokens"] = 1000
            response["usage"]["total_tokens"] = 1012
            write_json(response_path, response)
            lane["prime_response_sha256"] = digest(response_path)
            lane["prime_prompt_tokens"] = 1000
        row["prime_aggregate_prompt_tokens"] = 4000
        reseal_row(short_prime, rows, row)
        run_verify(short_prime, success=False, reason=">4096-token aggregate prime history")
        rejected += 1

        unequal_prime = clone(valid, parent, "unequal-prime")
        rows = [json.loads(line) for line in
                (unequal_prime / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][3]
        response_path = unequal_prime / lane["prime_response_path"]
        response = json.loads(response_path.read_text())
        response["usage"]["prompt_tokens"] += 1
        response["usage"]["total_tokens"] += 1
        write_json(response_path, response)
        lane["prime_response_sha256"] = digest(response_path)
        lane["prime_prompt_tokens"] += 1
        row["prime_aggregate_prompt_tokens"] += 1
        reseal_row(unequal_prime, rows, row)
        run_verify(unequal_prime, success=False, reason="equal-token-width")
        rejected += 1

        bad_sse = clone(valid, parent, "bad-sse")
        rows = [json.loads(line) for line in
                (bad_sse / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][2]
        wire_path = bad_sse / lane["wire_response_path"]
        wire_path.write_text(
            wire_path.read_text().replace("data: [DONE]\n\n", ""),
            encoding="utf-8",
        )
        lane["wire_response_sha256"] = digest(wire_path)
        reseal_row(bad_sse, rows, row)
        run_verify(bad_sse, success=False, reason="SSE wire did not end")
        rejected += 1

        broad_normalization = clone(valid, parent, "broad-normalization")
        rows = [json.loads(line) for line in
                (broad_normalization / "samples.jsonl").read_text().splitlines()]
        row = next(row for row in rows if row["arm"] == "on")
        lane = row["lanes"][0]
        normalized_path = broad_normalization / lane["request_normalized_path"]
        normalized = json.loads(normalized_path.read_text())
        normalized["messages"][0]["content"] = "<normalized-content>"
        write_json(normalized_path, normalized)
        lane["request_normalized_sha256"] = digest(normalized_path)
        reseal_row(broad_normalization, rows, row)
        run_verify(broad_normalization, success=False, reason="request normalization drifted")
        rejected += 1

        off_trace = clone(valid, parent, "off-trace")
        rows = [json.loads(line) for line in
                (off_trace / "samples.jsonl").read_text().splitlines()]
        row = next(row for row in rows if row["arm"] == "off")
        trace_path = off_trace / row["trace_path"]
        trace_path.write_text(
            "[PREFILL_TIMING] STABLE BATCHED 4 seqs x 64 "
            "boundary rows in 10.5 ms count=1\n",
            encoding="utf-8",
        )
        row["trace_sha256"] = digest(trace_path)
        reseal_samples(off_trace, rows)
        run_verify(off_trace, success=False, reason="stable-route reachability count")
        rejected += 1

        outside_range = clone(valid, parent, "outside-proven-range")
        rows = [json.loads(line) for line in
                (outside_range / "samples.jsonl").read_text().splitlines()]
        row = next(row for row in rows if row["target_rows"] == 256)
        lane = row["lanes"][0]
        for key in ("wire_response_path", "canonical_response_path"):
            path = outside_range / lane[key]
            value = json.loads(path.read_text())
            value["usage"]["prompt_tokens"] += 1
            value["usage"]["total_tokens"] += 1
            write_json(path, value)
            lane[key.replace("_path", "_sha256")] = digest(path)
        normalized_path = outside_range / lane["normalized_path"]
        normalized = json.loads(normalized_path.read_text())
        normalized["usage"]["prompt_tokens"] += 1
        normalized["usage"]["total_tokens"] += 1
        write_json(normalized_path, normalized)
        lane["normalized_sha256"] = digest(normalized_path)
        lane["prompt_tokens"] += 1
        lane["work_rows"] += 1
        row["aggregate_work_rows"] += 1
        reseal_row(outside_range, rows, row)
        run_verify(
            outside_range, success=False,
            reason="cached stable tool-result continuation",
        )
        rejected += 1

        power_drift = clone(valid, parent, "power-drift")
        manifest = json.loads((power_drift / "manifest.json").read_text())
        record = json.loads((power_drift / manifest["processes"][0]["path"]).read_text())
        power_path = power_drift / record["power_path"]
        power_path.write_text(
            power_path.read_text().replace("\tac\t", "\tbattery\t", 1),
            encoding="utf-8",
        )
        record["power_sha256"] = digest(power_path)
        reseal_process(power_drift, 0, record)
        run_verify(power_drift, success=False, reason="power contract drifted")
        rejected += 1

        identity_drift = clone(valid, parent, "identity-drift")
        manifest_path = identity_drift / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["operator_launcher_sha256"] = "0" * 64
        for binding in manifest["processes"]:
            record_path = identity_drift / binding["path"]
            record = json.loads(record_path.read_text())
            record["runtime"]["operator_launcher_sha256"] = "0" * 64
            write_json(record_path, record)
            binding["sha256"] = digest(record_path)
        write_jsonl(identity_drift / "processes.jsonl", manifest["processes"])
        manifest["files"]["process_bindings"]["sha256"] = digest(
            identity_drift / "processes.jsonl"
        )
        write_json(manifest_path, manifest)
        run_verify(identity_drift, success=False, reason="live operator launcher drifted")
        rejected += 1

    assert rejected == 29
    print(
        "ADR-049 B.2 Gemma stable cached A/B contract passed; "
        f"mutation battery {rejected}/{rejected} rejected"
    )

if __name__ == "__main__":
    main()
