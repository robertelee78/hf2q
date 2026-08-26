#!/usr/bin/env python3
"""Model-free positive and mutation contract for the Gemma4 B.2 gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RUNNER = SCRIPT_DIR / "bench_adr049_b2_gemma4_aggregate_ab.sh"
VERIFY = SCRIPT_DIR / "verify_adr049_b2_gemma4_aggregate_ab.py"
WIDTHS = [128, 256, 512]


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
    assert "readonly PAYLOAD_WORD_ADJUSTMENT=40" in runner_source
    assert "payload_words=$((target - PAYLOAD_WORD_ADJUSTMENT))" in runner_source
    assert "readonly MAX_TARGET_ROW_DRIFT=4" in runner_source
    assert 'printf "adr049-b2-gemma-p%02d-w%03d-l%d ", pair, nominal, lane' in runner_source
    assert 'for (i = 1; i <= words; i++) printf "measurement "' in runner_source


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


def make_fixture(root: Path, speedup: float = 2.0) -> None:
    root.mkdir()
    root = root.resolve()
    (root / "processes").mkdir()
    identity = make_identity(root)
    model_verification = root / "model-verification.json"
    write_json(model_verification, {
        "schema_version": 2, "path": identity["model_path"],
        "sha256": identity["model_sha256"],
        "file_snapshot": identity["model_snapshot"],
        "file_stamp": {"fixture": True}, "content_hash_verified": True,
    })
    samples, bindings = [], []
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
            stderr_lines = []
            for width_position, target in enumerate(WIDTHS):
                wave_dir = process_dir / f"wave-{target}"
                wave_dir.mkdir()
                lanes, aggregate_rows = [], 0
                starts, finishes = [], []
                for lane_index in range(4):
                    prompt = target
                    prefill_ms = 20 + target * 0.05
                    ttft_ms, lane_wall_ms = prefill_ms + 1, prefill_ms + 2
                    request = wave_dir / f"lane-{lane_index}.request.json"
                    response = wave_dir / f"lane-{lane_index}.response.json"
                    wall = wave_dir / f"lane-{lane_index}.wall"
                    timing = wave_dir / f"lane-{lane_index}.timing"
                    normalized = wave_dir / f"lane-{lane_index}.normalized.json"
                    content = (
                        f"adr049-b2-gemma-p{pair:02d}-w{target:03d}-l{lane_index} "
                        + "measurement " * (target - 40)
                        + "Reply with one word."
                    )
                    write_json(request, {
                        "model": model_id, "messages": [{"role": "user", "content": content}],
                        "max_tokens": 1, "seed": 42, "temperature": 0,
                        "repetition_penalty": 1, "stream": False,
                        "hf2q_enable_thinking": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                    })
                    response_value = {
                        "choices": [{"message": {"role": "assistant", "content": "OK"},
                                     "finish_reason": "length"}],
                        "usage": {"prompt_tokens": prompt, "completion_tokens": 1,
                                  "total_tokens": prompt + 1,
                                  "prompt_tokens_details": {"cached_tokens": 0}},
                        "x_hf2q_timing": {"prefill_time_secs": prefill_ms / 1000,
                                           "time_to_first_token_ms": ttft_ms},
                    }
                    write_json(response, response_value)
                    wall.write_text(f"{lane_wall_ms / 1000:.12f}\n", encoding="utf-8")
                    start = 3000 + sequence * 2 + lane_index * 0.01
                    finish = start + 0.5
                    timing.write_text(f"{start:.9f}\t{finish:.9f}\n", encoding="utf-8")
                    starts.append(start)
                    finishes.append(finish)
                    write_json(normalized, {
                        "message": response_value["choices"][0]["message"],
                        "finish_reason": "length", "usage": response_value["usage"],
                    })
                    lanes.append({
                        "lane": lane_index, "prompt_tokens": prompt, "cached_tokens": 0,
                        "work_rows": prompt, "prefill_ms": prefill_ms,
                        "ttft_ms": ttft_ms, "wall_ms": lane_wall_ms,
                        "request_path": relative(root, request), "request_sha256": digest(request),
                        "response_path": relative(root, response), "response_sha256": digest(response),
                        "wall_path": relative(root, wall), "wall_sha256": digest(wall),
                        "timing_path": relative(root, timing), "timing_sha256": digest(timing),
                        "normalized_path": relative(root, normalized),
                        "normalized_sha256": digest(normalized),
                    })
                    aggregate_rows += prompt
                lanes_file = wave_dir / "lanes.jsonl"
                write_jsonl(lanes_file, lanes)
                wave_ms = (1000 + target) * (speedup if arm == "off" else 1.0)
                wave_wall = wave_dir / "wave.wall"
                wave_wall.write_text(f"{wave_ms / 1000:.12f}\n", encoding="utf-8")
                trace = wave_dir / "server.trace.log"
                if arm == "on":
                    line = "[PREFILL_TIMING] BATCHED 4 seqs in 10.5 ms (one multi-seq forward, iter-G(a))\n"
                    trace.write_text(line, encoding="utf-8")
                    stderr_lines.append(line)
                    trace_count, trace_requests, trace_elapsed = 1, 4, 10.5
                else:
                    trace.write_text("OFF cold prefill\n", encoding="utf-8")
                    trace_count, trace_requests, trace_elapsed = 0, None, None
                samples.append({
                    "schema_version": 1, "pair": pair, "process_position": position,
                    "arm": arm, "width_position": width_position, "target_rows": target,
                    "wave_ms": wave_ms, "wave_wall_path": relative(root, wave_wall),
                    "wave_wall_sha256": digest(wave_wall), "trace_path": relative(root, trace),
                    "trace_sha256": digest(trace), "trace_event_count": trace_count,
                    "trace_requests": trace_requests, "trace_elapsed_ms": trace_elapsed,
                    "aggregate_work_rows": aggregate_rows,
                    "launch_skew_seconds": max(starts) - min(starts),
                    "latest_start": max(starts), "earliest_finish": min(finishes),
                    "actual_overlap": True, "lanes_path": relative(root, lanes_file),
                    "lanes_sha256": digest(lanes_file), "lanes": lanes,
                })
                sequence += 1
            stderr.write_text("".join(stderr_lines) or "OFF server\n", encoding="utf-8")
            runtime_home = process_dir / "runtime-home"
            runtime_home.mkdir()
            process_record = {
                "schema_version": 1, "status": "stopped", "pair": pair,
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
                "command_path": relative(root, command_file), "command_sha256": digest(command_file),
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
    settle, contention_settle = root / "thermal-settle.log", root / "contention-settle.log"
    settle.write_text("".join(f"{1000 + offset}\tnominal\tadr049-b2-gemma-ab-settle\n"
                              for offset in range(0, 61, 5)), encoding="utf-8")
    contention_settle.write_text("".join(
        f"{1000 + offset}\tquiet\tadr049-b2-gemma-ab-settle\t100\t-\n"
        for offset in range(0, 61, 5)), encoding="utf-8")
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
        ("assertions", "caffeinate.log.assertions", "pid 999(caffeinate) PreventUserIdleSystemSleep\n"),
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
        "contention_settle": {"path": contention_settle.name, "sha256": digest(contention_settle)},
        "contention_measurement": {"path": contention_measurement.name,
                                   "sha256": digest(contention_measurement)},
        "power_guard": guard_files,
    }
    write_json(root / "manifest.json", {
        "schema_version": 1, "status": "measured",
        "configuration": {
            "pairs": 8, "width_targets": WIDTHS, "lanes": 4,
            "pair_order": "off-on-even_on-off-odd", "warmup_waves_per_process": 2,
            "measured_waves_per_process": 3,
            "payload_word_adjustment": 40, "maximum_target_row_drift": 4,
            "off_env": {"HF2Q_CROSS_SLOT_ADMIT": "0", "HF2Q_ADMIT_COALESCE_US": "0"},
            "on_env": {"HF2Q_CROSS_SLOT_ADMIT": "1", "HF2Q_ADMIT_COALESCE_US": "25000"},
            "request": {"max_tokens": 1, "seed": 42, "temperature": 0,
                        "repetition_penalty": 1, "stream": False, "thinking": False},
            "analysis": {"statistic": "median paired OFF/ON wave speedup",
                         "order_stratified_bootstrap_samples": 10000,
                         "bootstrap_seed": 49004, "lower_confidence_percentile": 2.5,
                         "minimum_lower_95_speedup_exclusive": 1.05},
        },
        "identity": identity,
        "environment": {"power": "ac", "power_mode": "automatic", "power_mode_code": "0",
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
    engine_source = (ROOT / "src/serve/api/engine.rs").read_text(encoding="utf-8")
    function = engine_source.split("fn admit_gemma4_slots_batched(", 1)[1].split(
        "fn admit_gemma4_slots_stable_batched(", 1
    )[0]
    forward = function.index('supervised_gemma4_gpu_call(supervisor, "gemma4_batched_prefill"')
    success = function.index("let output = match forward")
    trace = function.index("[PREFILL_TIMING] BATCHED")
    commit = function.index("commit_gemma4_slot_cursors")
    assert forward <= success < trace < commit
    assert 'std::env::var("HF2Q_PREFILL_TIMING").is_ok()' in function[trace - 300:trace]

    rejected = 0
    with tempfile.TemporaryDirectory(prefix="hf2q-adr049-gemma-ab-") as temporary:
        parent = Path(temporary)
        valid = parent / "valid"
        make_fixture(valid)
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
        with (bad_hash / "samples.jsonl").open("a", encoding="utf-8") as target:
            target.write("\n")
        run_verify(bad_hash, success=False, reason="samples file identity failed")
        rejected += 1

        bad_trace = clone(valid, parent, "bad-trace")
        rows = [json.loads(line) for line in (bad_trace / "samples.jsonl").read_text().splitlines()]
        on_row = next(row for row in rows if row["arm"] == "on")
        trace_path = bad_trace / on_row["trace_path"]
        trace_path.write_text(trace_path.read_text().replace("BATCHED 4", "BATCHED 3"), encoding="utf-8")
        on_row["trace_sha256"] = digest(trace_path)
        reseal_samples(bad_trace, rows)
        run_verify(bad_trace, success=False, reason="four-lane multi-seq forward")
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
        rows = [json.loads(line) for line in (output_drift / "samples.jsonl").read_text().splitlines()]
        on_row = next(row for row in rows if row["arm"] == "on")
        lane = on_row["lanes"][0]
        response_path = output_drift / lane["response_path"]
        response = json.loads(response_path.read_text())
        response["choices"][0]["message"]["content"] = "DIFFERENT"
        write_json(response_path, response)
        normalized_path = output_drift / lane["normalized_path"]
        normalized = json.loads(normalized_path.read_text())
        normalized["message"]["content"] = "DIFFERENT"
        write_json(normalized_path, normalized)
        lane["response_sha256"], lane["normalized_sha256"] = digest(response_path), digest(normalized_path)
        lanes_path = output_drift / on_row["lanes_path"]
        write_jsonl(lanes_path, on_row["lanes"])
        on_row["lanes_sha256"] = digest(lanes_path)
        reseal_samples(output_drift, rows)
        run_verify(output_drift, success=False, reason="canonical results differ")
        rejected += 1

        bad_thermal = clone(valid, parent, "bad-thermal")
        thermal = bad_thermal / "thermal-measurement.log"
        thermal.write_text(thermal.read_text().replace("2000\tnominal", "2000\tfair", 1), encoding="utf-8")
        reseal_top(bad_thermal, "thermal_measurement")
        run_verify(bad_thermal, success=False, reason="measurement thermal sentinels")
        rejected += 1

        request_drift = clone(valid, parent, "request-drift")
        rows = [json.loads(line) for line in (request_drift / "samples.jsonl").read_text().splitlines()]
        on_row = next(row for row in rows if row["arm"] == "on")
        lane = on_row["lanes"][0]
        request_path = request_drift / lane["request_path"]
        request = json.loads(request_path.read_text())
        request["messages"][0]["content"] += " drift"
        write_json(request_path, request)
        lane["request_sha256"] = digest(request_path)
        lanes_path = request_drift / on_row["lanes_path"]
        write_jsonl(lanes_path, on_row["lanes"])
        on_row["lanes_sha256"] = digest(lanes_path)
        reseal_samples(request_drift, rows)
        run_verify(request_drift, success=False, reason="request drifted")
        rejected += 1

        no_overlap = clone(valid, parent, "no-overlap")
        rows = [json.loads(line) for line in (no_overlap / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][3]
        timing_path = no_overlap / lane["timing_path"]
        timing_path.write_text("4000.000000000\t4000.500000000\n", encoding="utf-8")
        lane["timing_sha256"] = digest(timing_path)
        lanes_path = no_overlap / row["lanes_path"]
        write_jsonl(lanes_path, row["lanes"])
        row["lanes_sha256"] = digest(lanes_path)
        row["launch_skew_seconds"] = 1000.0
        row["latest_start"] = 4000.0
        row["actual_overlap"] = False
        reseal_samples(no_overlap, rows)
        run_verify(no_overlap, success=False, reason="simultaneous four-lane wave")
        rejected += 1

        target_drift = clone(valid, parent, "target-drift")
        rows = [json.loads(line) for line in (target_drift / "samples.jsonl").read_text().splitlines()]
        row = rows[0]
        lane = row["lanes"][0]
        response_path = target_drift / lane["response_path"]
        response = json.loads(response_path.read_text())
        response["usage"]["prompt_tokens"] += 5
        response["usage"]["total_tokens"] += 5
        write_json(response_path, response)
        normalized_path = target_drift / lane["normalized_path"]
        normalized = json.loads(normalized_path.read_text())
        normalized["usage"]["prompt_tokens"] += 5
        normalized["usage"]["total_tokens"] += 5
        write_json(normalized_path, normalized)
        lane["prompt_tokens"] += 5
        lane["work_rows"] += 5
        lane["response_sha256"] = digest(response_path)
        lane["normalized_sha256"] = digest(normalized_path)
        lanes_path = target_drift / row["lanes_path"]
        write_jsonl(lanes_path, row["lanes"])
        row["lanes_sha256"] = digest(lanes_path)
        row["aggregate_work_rows"] += 5
        reseal_samples(target_drift, rows)
        run_verify(target_drift, success=False, reason="not cold in its target bin")
        rejected += 1

        power_drift = clone(valid, parent, "power-drift")
        manifest = json.loads((power_drift / "manifest.json").read_text())
        record = json.loads((power_drift / manifest["processes"][0]["path"]).read_text())
        power_path = power_drift / record["power_path"]
        power_path.write_text(power_path.read_text().replace("\tac\t", "\tbattery\t", 1), encoding="utf-8")
        record["power_sha256"] = digest(power_path)
        reseal_process(power_drift, 0, record)
        run_verify(power_drift, success=False, reason="power contract drifted")
        rejected += 1

        identity_drift = clone(valid, parent, "identity-drift")
        manifest_path = identity_drift / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["identity"]["operator_launcher_sha256"] = "0" * 64
        for index, binding in enumerate(manifest["processes"]):
            record_path = identity_drift / binding["path"]
            record = json.loads(record_path.read_text())
            record["runtime"]["operator_launcher_sha256"] = "0" * 64
            write_json(record_path, record)
            binding["sha256"] = digest(record_path)
        write_jsonl(identity_drift / "processes.jsonl", manifest["processes"])
        manifest["files"]["process_bindings"]["sha256"] = digest(identity_drift / "processes.jsonl")
        write_json(manifest_path, manifest)
        run_verify(identity_drift, success=False, reason="live operator launcher drifted")
        rejected += 1

    assert rejected == 11
    print(f"ADR-049 B.2 Gemma aggregate A/B contract passed; mutation battery {rejected}/{rejected} rejected")


if __name__ == "__main__":
    main()
