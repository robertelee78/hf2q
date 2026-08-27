#!/usr/bin/env python3
"""Independent ADR-049 B.1 real-HTTP receipt verifier.

The producer deliberately does not import this module.  This verifier reopens
the raw request, timed-SSE, server-log, telemetry, RSS, and manifest evidence
and derives the result again instead of trusting producer summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROCESS_ORDER = ("off-a", "on-a", "on-b", "off-b")
TRIALS = 5
MAX_SLOTS = 8
LIVE_DECODERS = 4
PREFILLERS = 4
MIXED_ROWS = 128
DECODER_PRIME_MAX_TOKENS = 1
SCHEDULER_GAP_MS = 15_000.0
SEMANTIC_SSE_GAP_MS = 15_000.0
MAX_PREFILL_WALL_SECONDS = 60.0
MAX_PEAK_RSS_BYTES = 116 * 1024**3
MIN_WAVE_SPEEDUP = 1.0
MAX_SAMPLE_GAP_SECONDS = 5
MODEL_BYTES = 107_431_343_168
MODEL_SHA256 = "936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d"
SHARED_EVIDENCE_FILES = (
    "model-verification.json",
    f"model-verification-cache/{MODEL_SHA256}.json",
    "caffeinate.log",
    "caffeinate.log.assertions",
    "caffeinate.log.power-events.baseline",
    "caffeinate.log.power-events.final",
    "caffeinate.log.power-events.new",
    "off-wave-samples-seconds",
    "on-wave-samples-seconds",
)


class ReceiptError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_snapshot(path: Path) -> str:
    stat = path.stat()
    return (f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:"
            f"{int(stat.st_mtime)}:{int(stat.st_ctime)}")


def file_stamp(path: Path) -> dict[str, int]:
    stat = path.stat()
    modified_seconds, modified_nanoseconds = divmod(stat.st_mtime_ns, 1_000_000_000)
    changed_seconds, changed_nanoseconds = divmod(stat.st_ctime_ns, 1_000_000_000)
    return {
        "device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size,
        "modified_seconds": modified_seconds,
        "modified_nanoseconds": modified_nanoseconds,
        "changed_seconds": changed_seconds,
        "changed_nanoseconds": changed_nanoseconds,
    }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
        require(stream.read().strip() == "", f"multiple JSON documents: {path}")
        return value


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def exact_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    require(type(value) in (int, float), f"{name} is not numeric")
    number = float(value)
    require(math.isfinite(number), f"{name} is not finite")
    if minimum is not None:
        require(number >= minimum, f"{name} is below {minimum}")
    return number


def close(actual: float, expected: float, name: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
            f"{name} drifted: {actual} != {expected}")


def parse_timed_sse(path: Path) -> dict[str, Any]:
    role_events = 0
    content: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[Any] = []
    finish_reasons: list[str] = []
    semantic_at: list[float] = []
    usage: dict[str, Any] | None = None
    done = 0
    event_count = 0
    previous_at = -math.inf
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            row = raw.rstrip("\r\n")
            if not row:
                continue
            fields = row.split("\t", 1)
            require(len(fields) == 2, f"malformed timed SSE row {path}:{line_number}")
            at = exact_number(float(fields[0]), f"SSE timestamp {path}:{line_number}", minimum=0)
            require(at >= previous_at, f"timed SSE moved backwards: {path}:{line_number}")
            previous_at = at
            payload_line = fields[1]
            if not payload_line.startswith("data: "):
                continue
            payload = payload_line[6:]
            if payload == "[DONE]":
                done += 1
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError as error:
                raise ReceiptError(f"malformed SSE JSON {path}:{line_number}: {error}") from error
            event_count += 1
            choices = event.get("choices")
            if isinstance(choices, list) and len(choices) == 1:
                choice = choices[0]
                delta = choice.get("delta") or {}
                if delta.get("role") is not None:
                    require(delta["role"] == "assistant", f"non-assistant SSE role in {path}")
                    role_events += 1
                semantic = False
                for field, output in (("content", content), ("reasoning_content", reasoning)):
                    value = delta.get(field)
                    if value is not None:
                        require(isinstance(value, str), f"non-string {field} in {path}")
                        if value:
                            output.append(value)
                            semantic = True
                calls = delta.get("tool_calls")
                if calls:
                    require(isinstance(calls, list), f"invalid tool_calls in {path}")
                    tool_calls.extend(calls)
                    semantic = True
                finish = choice.get("finish_reason")
                if finish is not None:
                    require(isinstance(finish, str) and finish, f"invalid finish_reason in {path}")
                    finish_reasons.append(finish)
                if semantic:
                    semantic_at.append(at)
            if event.get("usage") is not None:
                require(isinstance(event["usage"], dict), f"invalid SSE usage in {path}")
                usage = event["usage"]
    require(done == 1, f"{path} did not emit exactly one [DONE]")
    require(event_count > 0 and semantic_at, f"{path} emitted no semantic SSE event")
    require(role_events == 1, f"{path} did not emit exactly one assistant role event")
    require(len(finish_reasons) == 1, f"{path} did not emit one finish reason")
    require(not tool_calls, f"plain-text B.1 fixture unexpectedly emitted a tool call: {path}")
    require(isinstance(usage, dict), f"{path} omitted final usage")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(type(usage.get(field)) is int and usage[field] > 0,
                f"{path} has invalid usage.{field}")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            f"{path} usage token conservation failed")
    gaps = [later - earlier for earlier, later in zip(semantic_at, semantic_at[1:])]
    return {
        "schema": 1,
        "role_events": role_events,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "tool_calls": [],
        "finish_reason": finish_reasons[0],
        "usage": usage,
        "done_count": done,
        "event_count": event_count,
        "semantic_events": len(semantic_at),
        "semantic_max_gap_ms": (max(gaps, default=0.0) * 1000.0),
    }


def semantic_projection(canonical: dict[str, Any]) -> dict[str, Any]:
    return {key: canonical[key] for key in (
        "role_events", "content", "reasoning_content", "tool_calls",
        "finish_reason", "usage", "done_count",
    )}


def usage_cached_tokens(canonical: dict[str, Any], name: str) -> int:
    details = canonical["usage"].get("prompt_tokens_details")
    require(isinstance(details, dict), f"{name} omitted usage.prompt_tokens_details")
    cached = details.get("cached_tokens")
    require(type(cached) is int and cached >= 0,
            f"{name} has invalid usage.prompt_tokens_details.cached_tokens")
    return cached


def verify_manifest(process_dir: Path) -> str:
    manifest = process_dir / "evidence.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), f"missing manifest: {manifest}")
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00]+)", line)
        require(match is not None, f"malformed manifest row: {line!r}")
        digest, relative = match.groups()
        require(relative not in listed, f"duplicate manifest entry: {relative}")
        path = process_dir / relative
        require(path.is_file() and not path.is_symlink(), f"manifest path is not a regular file: {path}")
        require(sha256_file(path) == digest, f"manifest digest mismatch: {path}")
        listed.add(relative)
    require(listed, f"empty evidence manifest: {manifest}")
    actual = {
        str(path.relative_to(process_dir))
        for path in process_dir.rglob("*")
        if path.is_file() and path.name not in ("evidence.sha256", "summary.json")
    }
    require(listed == actual, f"manifest is not exhaustive: {process_dir}")
    return sha256_file(manifest)


def verify_shared_manifest(root: Path, expected_sha256: str) -> None:
    manifest = root / "shared-evidence.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "missing shared evidence manifest")
    require(sha256_file(manifest) == expected_sha256, "shared evidence manifest hash drift")
    expected = set(SHARED_EVIDENCE_FILES)
    listed: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00]+)", line)
        require(match is not None, f"malformed shared manifest row: {line!r}")
        digest, relative = match.groups()
        require(relative not in listed and relative in expected,
                f"unexpected/duplicate shared evidence: {relative}")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"missing shared evidence: {relative}")
        require(sha256_file(path) == digest, f"shared evidence hash drift: {relative}")
        listed.add(relative)
    require(listed == expected, "shared evidence inventory drift")


def parse_tsv(path: Path, fields: int) -> list[list[str]]:
    require(path.is_file() and path.stat().st_size > 0, f"missing telemetry: {path}")
    rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
    require(all(len(row) == fields for row in rows), f"malformed telemetry shape: {path}")
    timestamps = [int(row[0]) for row in rows]
    require(all(value >= 0 for value in timestamps), f"invalid timestamp: {path}")
    require(all(0 <= later - earlier <= MAX_SAMPLE_GAP_SECONDS
                for earlier, later in zip(timestamps, timestamps[1:])),
            f"telemetry gap/backwards timestamp: {path}")
    return rows


def verify_environment(
    process_dir: Path, summary: dict[str, Any], owner_pgid: int
) -> tuple[str, str]:
    settle = parse_tsv(process_dir / "thermal-settle.tsv", 3)
    require(all(row[1] == "nominal" for row in settle), "settle was not continuously nominal")
    require(int(settle[-1][0]) - int(settle[0][0]) >= 60, "settle was shorter than 60 seconds")
    measurement = parse_tsv(process_dir / "thermal-measurement.tsv", 3)
    require(measurement[0][1] == "nominal", "measurement did not start nominal")
    require(all(row[1] in ("nominal", "fair") for row in measurement),
            "measurement exceeded Fair")
    for thermal_name, contention_name in (("thermal-settle.tsv", "contention-settle.tsv"),
                                           ("thermal-measurement.tsv", "contention-measurement.tsv")):
        thermal = parse_tsv(process_dir / thermal_name, 3)
        contention = parse_tsv(process_dir / contention_name, 6)
        require([row[0] for row in thermal] == [row[0] for row in contention],
                f"thermal/contention timestamps differ: {process_dir}")
        require(all(row[1] == "quiet"
                    and row[3] == str(owner_pgid)
                    and math.isfinite(float(row[4])) and 0 <= float(row[4]) < 100
                    and row[5] == "-" for row in contention),
                f"host contention or owner drift observed: {process_dir}")
    power_settle = parse_tsv(process_dir / "power-settle.tsv", 5)
    require(len(power_settle) >= 2
            and int(power_settle[-1][0]) - int(power_settle[0][0]) >= 60,
            "power settle did not continuously cover 60 seconds")
    power = parse_tsv(process_dir / "power-measurement.tsv", 5)
    all_power = power_settle + power
    require(len(power) >= 2 and all(row[1] == "ac" for row in all_power),
            "power was not continuously AC")
    require(len({row[2] for row in all_power}) == 1
            and len({row[3] for row in all_power}) == 1,
            "power mode drifted")
    memory = parse_tsv(process_dir / "memory-measurement.tsv", 18)
    require(all(row[3] in ("1", "2") and row[13] == "0" for row in memory),
            "memory pressure exceeded Warning or throttled pages appeared")
    rss = [int(row) for row in (process_dir / "rss-kib").read_text().splitlines() if row]
    require(len(rss) >= 2 and all(value > 0 for value in rss), "RSS receipt is incomplete")
    peak = max(rss) * 1024
    require(peak <= MAX_PEAK_RSS_BYTES, "sampled peak RSS exceeds 116 GiB")
    require(summary["sampled_peak_rss_bytes"] == peak, "summary peak RSS is not raw-derived")
    return power[0][2], power[0][3]


FIELD_RE = re.compile(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)")


def fields(line: str) -> dict[str, str]:
    return {key: value.strip('"') for key, value in FIELD_RE.findall(line)}


def verify_request(path: Path, kind: str, model_id: str) -> None:
    request = load_json(path)
    require(set(request) == {
        "model", "messages", "temperature", "seed", "repetition_penalty",
        "max_tokens", "stream", "stream_options",
    }, f"request schema drift: {path}")
    require(request.get("model") == model_id, f"request model binding drift: {path}")
    require(request.get("stream") is True and request.get("temperature") == 0,
            f"non-deterministic request: {path}")
    require(request.get("seed") == 42 and request.get("repetition_penalty") == 1,
            f"request sampling drift: {path}")
    expected_max_tokens = {
        "decoder-prime": DECODER_PRIME_MAX_TOKENS,
        "decoder": 256,
        "prefill": 8,
    }.get(kind)
    require(expected_max_tokens is not None and request.get("max_tokens") == expected_max_tokens,
            f"request role/max_tokens drift: {path}")
    require(isinstance(request.get("messages"), list) and len(request["messages"]) == 1,
            f"request message shape drift: {path}")
    require(set(request["messages"][0]) == {"role", "content"}
            and request["messages"][0]["role"] == "user"
            and isinstance(request["messages"][0]["content"], str)
            and request["messages"][0]["content"],
            f"request message content drift: {path}")
    require(request.get("stream_options") == {"include_usage": True},
            f"request SSE options drift: {path}")


def request_timing(path: Path) -> tuple[float, float]:
    rows = path.read_text(encoding="utf-8").splitlines()
    require(len(rows) == 1, f"request timing is not singular: {path}")
    fields = rows[0].split("\t")
    require(len(fields) == 2, f"request timing shape drift: {path}")
    started, finished = map(float, fields)
    require(math.isfinite(started) and math.isfinite(finished) and 0 < started < finished,
            f"invalid request timing: {path}")
    return started, finished


def verify_decoder_prime(process_dir: Path, model_id: str) -> list[dict[str, Any]]:
    prime_dir = process_dir / "decoder-prime"
    prime = load_json(prime_dir / "prime.json")
    require(set(prime) == {"schema", "max_tokens", "request_ids"}
            and prime["schema"] == 1
            and prime["max_tokens"] == DECODER_PRIME_MAX_TOKENS,
            f"decoder-prime identity drift: {process_dir.name}")
    request_ids = prime.get("request_ids")
    require(isinstance(request_ids, list) and len(request_ids) == LIVE_DECODERS
            and len(set(request_ids)) == LIVE_DECODERS
            and all(type(value) is int and value > 0 for value in request_ids),
            f"decoder-prime request IDs are invalid: {process_dir.name}")
    prime_requests: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    for lane in range(1, LIVE_DECODERS + 1):
        stem = f"decoder-{lane}"
        prime_request_path = prime_dir / f"{stem}.request.json"
        verify_request(prime_request_path, "decoder-prime", model_id)
        prime_request = load_json(prime_request_path)
        prime_requests.append(prime_request)
        for trial in range(1, TRIALS + 1):
            measured_request = load_json(
                process_dir / "waves" / str(trial) / f"{stem}.request.json"
            )
            measured_request["max_tokens"] = DECODER_PRIME_MAX_TOKENS
            require(measured_request == prime_request,
                    f"measured decoder does not reuse its primed prompt: {stem}/{trial}")
        request_timing(prime_dir / f"{stem}.timing.tsv")
        raw = parse_timed_sse(prime_dir / f"{stem}.timed-sse")
        require(load_json(prime_dir / f"{stem}.canonical.json") == raw,
                f"decoder-prime canonical SSE drift: {stem}")
        projections.append(semantic_projection(raw))
    contents = [request["messages"][0]["content"] for request in prime_requests]
    require(len(set(contents)) == LIVE_DECODERS,
            f"decoder-prime lane prompts are not distinct: {process_dir.name}")
    log = (prime_dir / "server.delta.log").read_text(encoding="utf-8")
    starts = [fields(line) for line in log.splitlines()
              if "DeepSeek-V4 request started" in line]
    observed_ids = [int(row["request_id"]) for row in starts
                    if row.get("max_tokens") == str(DECODER_PRIME_MAX_TOKENS)]
    require(observed_ids == request_ids,
            f"decoder-prime server identities drifted: {process_dir.name}")
    completions = [fields(line) for line in log.splitlines()
                   if "DeepSeek-V4 request complete" in line]
    require(all(sum(row.get("request_id") == str(request_id) for row in completions) == 1
                for request_id in request_ids),
            f"decoder-prime completion receipt is incomplete: {process_dir.name}")
    return projections


def verify_wave(process_dir: Path, arm: str, trial: int, model_id: str) -> dict[str, Any]:
    wave_dir = process_dir / "waves" / str(trial)
    wave = load_json(wave_dir / "wave.json")
    require(set(wave) == {
        "schema", "trial", "arm", "live_decoders", "prefillers",
        "decoder_request_ids", "prefill_request_ids", "wall_seconds",
        "decoder_launch_skew_seconds", "prefill_launch_skew_seconds",
        "cooperative_transactions",
    } and wave["schema"] == 1 and wave["trial"] == trial and wave["arm"] == arm,
            f"wave identity drift: {wave_dir}")
    require(wave.get("live_decoders") == 4 and wave.get("prefillers") == 4,
            f"wave is not exact four-plus-four: {wave_dir}")
    log_path = wave_dir / "server.delta.log"
    log = log_path.read_text(encoding="utf-8")
    decoder_ids = wave.get("decoder_request_ids")
    prefill_ids = wave.get("prefill_request_ids")
    require(isinstance(decoder_ids, list) and len(decoder_ids) == 4,
            f"decoder id receipt is incomplete: {wave_dir}")
    require(isinstance(prefill_ids, list) and len(prefill_ids) == 4,
            f"prefill id receipt is incomplete: {wave_dir}")
    require(len(set(decoder_ids + prefill_ids)) == 8, f"request IDs are not unique: {wave_dir}")

    decoder_timings: list[tuple[float, float]] = []
    prefill_timings: list[tuple[float, float]] = []
    projections: list[dict[str, Any]] = []
    client_decoder_gaps: list[float] = []
    client_cached_tokens: dict[str, list[int]] = {"decoder": [], "prefill": []}
    for kind, count, timings in (("decoder", 4, decoder_timings), ("prefill", 4, prefill_timings)):
        for lane in range(1, count + 1):
            stem = f"{kind}-{lane}"
            verify_request(wave_dir / f"{stem}.request.json", kind, model_id)
            started, finished = request_timing(wave_dir / f"{stem}.timing.tsv")
            timings.append((started, finished))
            raw = parse_timed_sse(wave_dir / f"{stem}.timed-sse")
            sealed = load_json(wave_dir / f"{stem}.canonical.json")
            require(sealed == raw, f"producer canonical SSE drift: {stem}")
            projections.append(semantic_projection(raw))
            client_cached_tokens[kind].append(usage_cached_tokens(raw, stem))
            if kind == "decoder":
                require(raw["semantic_events"] >= 2, f"live decoder emitted fewer than two semantic events: {stem}")
                client_decoder_gaps.append(float(raw["semantic_max_gap_ms"]))

    latest_prefill_start = max(start for start, _ in prefill_timings)
    earliest_decoder_finish = min(finish for _, finish in decoder_timings)
    require(latest_prefill_start < earliest_decoder_finish,
            f"four prefills did not overlap four live decoders: {wave_dir}")
    request_starts = [start for start, _ in decoder_timings + prefill_timings]
    request_finishes = [finish for _, finish in decoder_timings + prefill_timings]
    wall = max(request_finishes) - min(request_starts)
    close(wall, exact_number(wave.get("wall_seconds"), "wave wall", minimum=0), "wave wall")
    decoder_skew = max(start for start, _ in decoder_timings) - min(start for start, _ in decoder_timings)
    prefill_skew = max(start for start, _ in prefill_timings) - min(start for start, _ in prefill_timings)
    require(decoder_skew <= 0.100 and prefill_skew <= 0.100,
            f"four-lane launch skew exceeded 100 ms: {wave_dir}")
    close(decoder_skew, float(wave["decoder_launch_skew_seconds"]), "decoder launch skew")
    close(prefill_skew, float(wave["prefill_launch_skew_seconds"]), "prefill launch skew")

    start_lines = [line for line in log.splitlines() if "DeepSeek-V4 request started" in line]
    starts_by_budget: dict[int, list[int]] = {8: [], 256: []}
    for line in start_lines:
        row = fields(line)
        if row.get("max_tokens") in ("8", "256"):
            starts_by_budget[int(row["max_tokens"])].append(int(row["request_id"]))
    require(starts_by_budget[256] == decoder_ids and starts_by_budget[8] == prefill_ids,
            f"server request identities disagree with wave receipt: {wave_dir}")
    log_lines = log.splitlines()
    prefill_start_positions = [index for index, line in enumerate(log_lines)
                               if "DeepSeek-V4 request started" in line
                               and fields(line).get("request_id") in {str(value) for value in prefill_ids}]
    decoder_complete_positions = [index for index, line in enumerate(log_lines)
                                  if "DeepSeek-V4 request complete" in line
                                  and fields(line).get("request_id") in {str(value) for value in decoder_ids}]
    require(len(prefill_start_positions) == PREFILLERS
            and len(decoder_complete_positions) == LIVE_DECODERS
            and max(prefill_start_positions) < min(decoder_complete_positions),
            f"prefills were not admitted before the first live decoder completed: {wave_dir}")

    scheduler_gaps: list[float] = []
    server_sse_gaps: list[float] = []
    receipt_lines = [line for line in log.splitlines() if "DeepSeek-V4 slot latency receipt" in line]
    prefill_plan_lines = [line for line in log.splitlines() if "DeepSeek-V4 prefill planned" in line]
    decoder_plan_cached_tokens: list[int] = []
    for request_id in decoder_ids:
        matches = [fields(line) for line in receipt_lines
                   if fields(line).get("request_id") == str(request_id)]
        require(len(matches) == 1, f"decoder latency receipt cardinality drift: {request_id}")
        row = matches[0]
        require(int(row["scheduler_decode_visits"]) >= 2, f"decoder was not live across Mixed turns: {request_id}")
        require(int(row["semantic_sse_events"]) >= 2, f"server semantic SSE receipt incomplete: {request_id}")
        scheduler_gaps.append(exact_number(float(row["scheduler_decode_max_gap_ms"]),
                                           "scheduler gap", minimum=0))
        server_sse_gaps.append(exact_number(float(row["semantic_sse_max_gap_ms"]),
                                            "server SSE gap", minimum=0))
        plans = [fields(line) for line in prefill_plan_lines
                 if fields(line).get("request_id") == str(request_id)]
        require(len(plans) == 1, f"decoder cache-plan cardinality drift: {request_id}")
        plan = plans[0]
        cached_tokens = int(plan.get("cached_tokens", "0"))
        prompt_tokens = int(plan.get("prompt_tokens", "0"))
        work_tokens = int(plan.get("work_tokens", "-1"))
        require(0 < cached_tokens <= prompt_tokens
                and work_tokens == prompt_tokens - cached_tokens
                and plan.get("cache") in {
                    "live", "grow-live", "recovery-anchor", "grow-recovery-anchor",
                },
                f"decoder did not use its primed cache: {request_id}")
        decoder_plan_cached_tokens.append(cached_tokens)

    require(all(value > 0 for value in client_cached_tokens["decoder"])
            and sorted(client_cached_tokens["decoder"]) == sorted(decoder_plan_cached_tokens),
            f"decoder client usage disagrees with warm-cache plans: {wave_dir}")
    for request_id in prefill_ids:
        plans = [fields(line) for line in prefill_plan_lines
                 if fields(line).get("request_id") == str(request_id)]
        require(len(plans) == 1, f"prefill cache-plan cardinality drift: {request_id}")
        plan = plans[0]
        cached_tokens = int(plan.get("cached_tokens", "-1"))
        prompt_tokens = int(plan.get("prompt_tokens", "0"))
        work_tokens = int(plan.get("work_tokens", "-1"))
        suffix_tokens = int(plan.get("suffix_tokens", "-1"))
        require(cached_tokens == 0
                and suffix_tokens == prompt_tokens
                and prompt_tokens <= work_tokens < 2 * prompt_tokens
                and plan.get("cache") in {"reset", "grow-reset"},
                f"declared cold prefill reused cache state: {request_id}")
    require(client_cached_tokens["prefill"] == [0] * PREFILLERS,
            f"prefill client usage is not cold: {wave_dir}")

    cooperative: list[dict[str, str]] = [fields(line) for line in log.splitlines()
                                              if "DeepSeek-V4 cooperative prefill complete" in line]
    require(all(row.get("bounded_mixed") in ("true", "false")
                and row.get("rows_per_lane_cap") is not None for row in cooperative),
            f"cooperative route discriminator is missing: {wave_dir}")
    mixed_cooperative = [row for row in cooperative if row["bounded_mixed"] == "true"]
    require(all(row.get("rows_per_lane_cap") == "128" for row in mixed_cooperative),
            f"bounded Mixed cap drifted: {wave_dir}")
    require(all(row.get("rows_per_lane_cap") == "0" for row in cooperative
                if row["bounded_mixed"] == "false"),
            f"pure-prefill cooperative cap drifted: {wave_dir}")
    if arm == "on":
        require(mixed_cooperative, f"ON wave published no Mixed cohort: {wave_dir}")
        require(all(row.get("lanes") == "4" and row.get("rows_per_lane") == "128"
                    and row.get("aggregate_rows") == "512" for row in mixed_cooperative),
                f"ON wave published a non-4x128 cohort: {wave_dir}")
    else:
        require(not mixed_cooperative, f"OFF wave published a Mixed cohort: {wave_dir}")
    require(wave["cooperative_transactions"] == len(mixed_cooperative),
            f"wave cooperative count is not raw-derived: {wave_dir}")

    prefill_walls = [finish - start for start, finish in prefill_timings]
    if arm == "on":
        require(max(prefill_walls) <= MAX_PREFILL_WALL_SECONDS,
                f"ON prefill exceeded the 60-second product ceiling: {wave_dir}")
        require(max(scheduler_gaps) <= SCHEDULER_GAP_MS,
                f"ON scheduler decode gap exceeded 15 seconds: {wave_dir}")
        require(max(server_sse_gaps + client_decoder_gaps) <= SEMANTIC_SSE_GAP_MS,
                f"ON semantic SSE gap exceeded 15 seconds: {wave_dir}")
    return {
        "wall": wall,
        "prefill_walls": prefill_walls,
        "scheduler_gaps": scheduler_gaps,
        "server_sse_gaps": server_sse_gaps,
        "client_sse_gaps": client_decoder_gaps,
        "cooperative": len(mixed_cooperative),
        "projections": projections,
    }


def verify_process(root: Path, label: str, evidence: dict[str, Any]) -> dict[str, Any]:
    process_dir = root / label
    summary_path = process_dir / "summary.json"
    summary = load_json(summary_path)
    require(sha256_file(summary_path) == evidence["summary_sha256"], f"summary hash drift: {label}")
    manifest_sha = verify_manifest(process_dir)
    require(manifest_sha == evidence["manifest_sha256"], f"manifest hash drift: {label}")
    arm = "on" if label.startswith("on-") else "off"
    require(summary.get("schema") == 1 and summary.get("label") == label and summary.get("arm") == arm,
            f"process summary identity drift: {label}")
    require(summary.get("evidence_manifest_sha256") == manifest_sha,
            f"summary does not bind its process manifest: {label}")
    server = summary.get("server", {})
    require(type(server.get("pid")) is int and server["pid"] > 1,
            f"process PID receipt drift: {label}")
    require(server.get("port") == evidence["port"] and server.get("max_slots") == 8,
            f"process did not use the fixed loopback port/eight slots: {label}")
    require(server.get("binary_sha256") == evidence["binary_sha256"]
            and server.get("model_sha256") == evidence["model_sha256"],
            f"process artifact identity drift: {label}")
    require(summary.get("policy") == {"mixed_cohort": arm == "on", "rows_per_lane": 128},
            f"process policy receipt drift: {label}")
    raw_pid = (process_dir / "server.pid").read_text(encoding="utf-8").strip()
    require(raw_pid == str(server["pid"]), f"raw process PID drift: {label}")
    command = shlex.split((process_dir / "server-command.txt").read_text(encoding="utf-8"))
    require(command and command[0] == evidence["binary_path"],
            f"process command does not execute the sealed binary: {label}")
    required_options = {
        "--model": evidence["model_path"], "--host": evidence["host"],
        "--port": str(evidence["port"]), "--scheduler": "inflight-batched", "--max-slots": "8",
        "--kv-cache-budget": "8589934592", "--overflow-policy": "reject",
    }
    for option, expected in required_options.items():
        require(option in command and command.index(option) + 1 < len(command)
                and command[command.index(option) + 1] == expected,
                f"process command {option} binding drift: {label}")
    models = load_json(process_dir / "models.json")
    loaded = [model for model in models.get("data", []) if model.get("loaded") is True]
    require(len(loaded) == 1 and loaded[0].get("arch") == "deepseek4"
            and isinstance(loaded[0].get("id"), str) and loaded[0]["id"],
            f"loaded model architecture receipt drift: {label}")
    model_id = loaded[0]["id"]
    prime_projections = verify_decoder_prime(process_dir, model_id)
    startup_log = (process_dir / "server.stderr").read_text(encoding="utf-8")
    require(not re.search(
        r"GPU Timeout|SubmissionsIgnored|Command buffer error|Generation error|"
        r"engine_unhealthy|panicked at|worker-fatal",
        startup_log,
        re.IGNORECASE,
    ), f"fatal runtime log signature: {label}")
    startup = [fields(line) for line in startup_log.splitlines()
               if "DeepSeek-V4 full-context session worker started" in line]
    require(len(startup) == 1, f"startup policy event cardinality drift: {label}")
    require(startup[0].get("slots") == "8"
            and startup[0].get("mixed_cohort") == str(arm == "on").lower()
            and startup[0].get("mixed_cohort_selection") == f"explicit-{arm}"
            and startup[0].get("mixed_cohort_rows_per_lane") == "128",
            f"startup policy event does not prove the selected arm: {label}")
    power_mode = verify_environment(process_dir, summary, evidence["owner_pgid"])
    waves = [verify_wave(process_dir, arm, trial, model_id)
             for trial in range(1, TRIALS + 1)]
    walls = [wave["wall"] for wave in waves]
    reported_walls = summary.get("wave_samples_seconds")
    require(isinstance(reported_walls, list) and len(reported_walls) == TRIALS,
            f"process wave sample shape drift: {label}")
    for trial, (actual, expected) in enumerate(zip(reported_walls, walls, strict=True), 1):
        close(float(actual), expected, f"process wave {label}/{trial}")
    close(statistics.median(walls), float(summary["wave_median_seconds"]), f"process median {label}")
    require(summary.get("cooperative_transactions") == sum(wave["cooperative"] for wave in waves),
            f"cooperative count is not raw-derived: {label}")
    return {"arm": arm, "walls": walls, "waves": waves,
            "prime_projections": prime_projections,
            "power_mode": power_mode, "model_id": model_id}


def git_output(source_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(source_root), *args], text=True).strip()


def verify(
    receipt_path: Path,
    source_root: Path,
    *,
    expected_model_bytes: int = MODEL_BYTES,
    expected_model_sha256: str = MODEL_SHA256,
) -> None:
    receipt = load_json(receipt_path)
    root = receipt_path.parent
    require(receipt.get("schema") == 1 and receipt.get("verdict") == "pass"
            and receipt.get("gate") == "deepseek4-mixed-policy-abba-http",
            "top-level receipt identity drift")
    require(receipt.get("workload") == {
        "process_order": list(PROCESS_ORDER), "same_binary": True,
        "trials_per_process": TRIALS, "max_slots": MAX_SLOTS,
        "live_decoders": LIVE_DECODERS, "prefillers": PREFILLERS,
        "mixed_rows_per_lane": MIXED_ROWS, "temperature": 0, "seed": 42,
        "decoder_prime": {"lanes": LIVE_DECODERS,
                          "max_tokens": DECODER_PRIME_MAX_TOKENS,
                          "stable_prompt_required": True,
                          "cache_reuse_required": True},
    }, "workload contract drift")
    environment = receipt.get("environment")
    require(isinstance(environment, dict) and set(environment) == {"host_contention"},
            "environment contract schema drift")
    host_contention = environment["host_contention"]
    require(isinstance(host_contention, dict) and host_contention == {
        "policy": "process-group-cpu-v2",
        "maximum_foreign_cpu_percent": 100,
        "owner_scope": "release-gate-process-group",
        "owner_pgid": host_contention.get("owner_pgid"),
        "continuous": True,
    }, "environment host contention policy drift")
    owner_pgid = host_contention.get("owner_pgid")
    require(type(owner_pgid) is int and owner_pgid > 0,
            "environment host contention owner drift")
    require(receipt.get("thresholds") == {
        "scheduler_decode_gap_ms": SCHEDULER_GAP_MS,
        "semantic_sse_gap_ms": SEMANTIC_SSE_GAP_MS,
        "max_prefill_wall_seconds": MAX_PREFILL_WALL_SECONDS,
        "max_peak_rss_bytes": MAX_PEAK_RSS_BYTES,
        "min_wave_speedup": MIN_WAVE_SPEEDUP,
    }, "immutable threshold contract drift")
    endpoint = receipt.get("endpoint", {})
    require(endpoint.get("host") == "127.0.0.1" and type(endpoint.get("port")) is int
            and 0 < endpoint["port"] <= 65535,
            "loopback HTTP endpoint binding drift")
    source = receipt.get("source", {})
    require(source_root.resolve() == Path(source.get("root", "")).resolve(), "source root binding drift")
    require(git_output(source_root, "rev-parse", "HEAD") == source.get("commit"), "source commit drift")
    require(git_output(source_root, "status", "--porcelain", "--untracked-files=all") == "",
            "source tree is not clean")
    binary = Path(source.get("binary", ""))
    require(binary == source_root / "target/release/hf2q" and binary.is_file()
            and not binary.is_symlink() and os.access(binary, os.X_OK),
            "binary path drift")
    require(sha256_file(binary) == source.get("sha256"), "binary hash drift")
    require(source["commit"].encode() in binary.read_bytes(), "binary does not embed source commit")
    model = receipt.get("model", {})
    model_path = Path(model.get("path", ""))
    require(model_path.is_absolute() and model_path.is_file() and not model_path.is_symlink(),
            "DeepSeek artifact path drift")
    require(model.get("bytes") == expected_model_bytes
            and model_path.stat().st_size == expected_model_bytes,
            "DeepSeek artifact size drift")
    require(model.get("sha256") == expected_model_sha256, "DeepSeek artifact identity drift")
    require(model.get("snapshot") == file_snapshot(model_path),
            "DeepSeek artifact file snapshot drift")
    verify_shared_manifest(root, receipt.get("evidence", {}).get("shared_manifest_sha256", ""))
    model_verification = load_json(root / "model-verification.json")
    require(model_verification.get("schema_version") == 2
            and model_verification.get("path") == str(model_path)
            and model_verification.get("sha256") == expected_model_sha256
            and model_verification.get("file_snapshot") == model["snapshot"]
            and model_verification.get("file_stamp") == file_stamp(model_path)
            and model_verification.get("content_hash_verified") is True,
            "binary-authored model verification receipt drift")
    cached_verification = load_json(root / f"model-verification-cache/{MODEL_SHA256}.json")
    identity_fields = (
        "schema_version", "path", "sha256", "file_snapshot", "file_stamp",
        "content_hash_verified",
    )
    require({field: cached_verification.get(field) for field in identity_fields}
            == {field: model_verification.get(field) for field in identity_fields},
            "cached and per-run model verification identities differ")
    require(model_verification.get("run_verification") in (
        "content_hash", "cached_unchanged_file", "upgraded_legacy_receipt",
    ), "model verification mode drift")
    require((root / "caffeinate.log.power-events.new").stat().st_size == 0,
            "macOS power-state transition occurred during the gate")
    require("caffeinate" in (root / "caffeinate.log.assertions").read_text(encoding="utf-8"),
            "caffeinate assertion evidence is absent")

    process_evidence = receipt.get("evidence", {}).get("processes", {})
    require(set(process_evidence) == set(PROCESS_ORDER), "process evidence inventory drift")
    bound_evidence = {
        label: {
            **process_evidence[label],
            "binary_path": str(binary), "binary_sha256": source["sha256"],
            "model_path": str(model_path), "model_sha256": expected_model_sha256,
            "host": endpoint["host"], "port": endpoint["port"],
            "owner_pgid": owner_pgid,
        }
        for label in PROCESS_ORDER
    }
    processes = {label: verify_process(root, label, bound_evidence[label])
                 for label in PROCESS_ORDER}
    require(len({process["power_mode"] for process in processes.values()}) == 1,
            "power mode differs across order-balanced processes")
    require(len({process["model_id"] for process in processes.values()}) == 1,
            "loaded model ID differs across order-balanced processes")
    for replica in ("a", "b"):
        off = processes[f"off-{replica}"]["waves"]
        on = processes[f"on-{replica}"]["waves"]
        require(processes[f"off-{replica}"]["prime_projections"]
                == processes[f"on-{replica}"]["prime_projections"],
                f"OFF/ON decoder-prime semantic parity failed for replica {replica}")
        for lane in range(1, LIVE_DECODERS + 1):
            for suffix in ("request.json", "canonical.json"):
                relative = Path("decoder-prime") / f"decoder-{lane}.{suffix}"
                require((root / f"off-{replica}" / relative).read_bytes()
                        == (root / f"on-{replica}" / relative).read_bytes(),
                        f"OFF/ON decoder-prime bytes drifted: {replica}/{relative}")
        require([wave["projections"] for wave in off] == [wave["projections"] for wave in on],
                f"OFF/ON semantic or token parity failed for replica {replica}")
        for trial in range(1, TRIALS + 1):
            for kind in ("decoder", "prefill"):
                for lane in range(1, 5):
                    relative = Path("waves") / str(trial) / f"{kind}-{lane}.request.json"
                    require((root / f"off-{replica}" / relative).read_bytes()
                            == (root / f"on-{replica}" / relative).read_bytes(),
                            f"OFF/ON request bytes drifted: {replica}/{relative}")
    projections = []
    for replica in ("a", "b"):
        projections.extend(processes[f"on-{replica}"]["prime_projections"])
        for wave in processes[f"on-{replica}"]["waves"]:
            projections.extend(wave["projections"])
    semantic_sha = hashlib.sha256(canonical_bytes(projections)).hexdigest()
    require(receipt.get("equality", {}).get("semantic_and_token_sha256") == semantic_sha,
            "semantic aggregate hash is not raw-derived")

    off_walls = processes["off-a"]["walls"] + processes["off-b"]["walls"]
    on_walls = processes["on-a"]["walls"] + processes["on-b"]["walls"]
    speedup = statistics.median(off_walls) / statistics.median(on_walls)
    neighbors = [statistics.median(processes[f"off-{replica}"]["walls"])
                 / statistics.median(processes[f"on-{replica}"]["walls"])
                 for replica in ("a", "b")]
    require(speedup > MIN_WAVE_SPEEDUP and all(value > 1.0 for value in neighbors),
            "ON did not beat both order-balanced OFF neighbors")
    result = receipt.get("result", {})
    close(float(result.get("wave_speedup")), speedup, "pooled speedup")
    reported_neighbors = result.get("neighboring_process_speedups")
    require(isinstance(reported_neighbors, list) and len(reported_neighbors) == 2,
            "neighbor speedup receipt shape drifted")
    for index, (actual, expected) in enumerate(zip(reported_neighbors, neighbors, strict=True)):
        close(float(actual), expected, f"neighbor speedup {index}")
    for name, reported, expected in (
        ("OFF", result.get("off_wave_samples_seconds"), off_walls),
        ("ON", result.get("on_wave_samples_seconds"), on_walls),
    ):
        require(isinstance(reported, list) and len(reported) == len(expected),
                f"{name} sample receipt shape drifted")
        for index, (actual, raw) in enumerate(zip(reported, expected, strict=True)):
            close(float(actual), raw, f"{name} sample {index}")


def canonicalize(input_path: Path, output_path: Path) -> None:
    value = parse_timed_sse(input_path)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", nargs="?", type=Path)
    parser.add_argument("source_root", nargs="?", type=Path)
    parser.add_argument("--canonicalize", nargs=2, metavar=("TIMED_SSE", "OUTPUT"), type=Path)
    args = parser.parse_args()
    try:
        if args.canonicalize:
            canonicalize(*args.canonicalize)
        else:
            require(args.receipt is not None and args.source_root is not None,
                    "receipt and source_root are required")
            verify(args.receipt, args.source_root)
            print("DeepSeek-V4 Mixed policy receipt verified", file=sys.stderr)
    except (ReceiptError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"DeepSeek-V4 Mixed policy receipt rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
