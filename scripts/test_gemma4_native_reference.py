#!/usr/bin/env python3
"""Compare Gemma semantics using frozen OpenAI requests on one running engine.

Run engines sequentially on the same artifact. This harness does not launch an
engine or claim identical rendered tokens/KV arithmetic. It retains requests,
responses and externally measured SSE latency for source-bound comparison.
"""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import pathlib
import time
import threading
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--engine", choices=["hf2q", "peer"], required=True)
    parser.add_argument("--request", type=pathlib.Path, required=True)
    parser.add_argument("--tool-result", type=pathlib.Path, required=True)
    parser.add_argument("--sentinel", default="HF2Q_GEMMA4_AGENTIC_OK")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--concurrency", action="store_true", help="also compare four concurrent HTTP requests against serial references")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    request_bytes = args.request.read_bytes()
    request = json.loads(request_bytes)
    expected_path = str(args.tool_result.resolve())
    tool_result = args.tool_result.read_text()

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")

    def post(label, payload, raw=None):
        body = raw if raw is not None else json.dumps(payload).encode()
        (args.output / f"{label}.request.json").write_bytes(body)
        started = time.monotonic()
        req = urllib.request.Request(
            args.base_url.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
        elapsed = (time.monotonic() - started) * 1000
        save(f"{label}.response.json", result)
        return result, elapsed

    def check_tool(result):
        assert len(result["choices"]) == 1, result
        choice = result["choices"][0]
        assert choice["finish_reason"] == "tool_calls", choice
        calls = choice["message"].get("tool_calls", [])
        assert len(calls) == 1, calls
        assert calls[0]["type"] == "function", calls
        function = calls[0]["function"]
        assert function["name"] == "read_file", function
        assert json.loads(function["arguments"]) == {"path": expected_path}, function
        return choice["message"]

    def metrics(result, elapsed):
        usage = result.get("usage", {})
        native = result.get("x_hf2q_timing", {})
        peer = result.get("timings", {})
        return {
            "wall_ms": elapsed,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": usage.get("prompt_tokens_details", {}).get(
                "cached_tokens", peer.get("cache_n")
            ),
            "prefill_tokens_per_sec": native.get(
                "prefill_tokens_per_sec", peer.get("prompt_per_second")
            ),
            "decode_tokens_per_sec": native.get(
                "decode_tokens_per_sec", peer.get("predicted_per_second")
            ),
        }

    cold, cold_ms = post("cold-tool", request, request_bytes)
    check_tool(cold)
    cached, cached_ms = post("cached-tool", request, request_bytes)
    previous = check_tool(cached)
    continuation = copy.deepcopy(request)
    continuation["messages"] += [
        {
            "role": "assistant",
            "content": previous.get("content"),
            "tool_calls": previous["tool_calls"],
        },
        {
            "role": "tool",
            "tool_call_id": previous["tool_calls"][0]["id"],
            "content": "Successful read_file result. File follows:\n" + tool_result,
        },
    ]
    continuation["tool_choice"] = "auto"
    continued, continued_ms = post("tool-result", continuation)
    choice = continued["choices"][0]
    assert choice["finish_reason"] == "stop", choice
    assert choice["message"].get("content") == args.sentinel, choice
    assert not choice["message"].get("tool_calls"), choice

    # Empty role events do not count as semantic progress. Retain raw SSE and
    # require a complete valid tool call, in addition to first-semantic timing.
    streamed = copy.deepcopy(request)
    streamed["stream"] = True
    streamed["stream_options"] = {"include_usage": True}
    save("stream.request.json", streamed)
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(streamed).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_semantic = None
    calls = {}
    finish = None
    done = False
    with urllib.request.urlopen(req, timeout=120) as response, (
        args.output / "stream.response.sse"
    ).open("wb") as raw:
        for line in response:
            raw.write(line)
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            for item in event.get("choices", []):
                delta = item.get("delta", {})
                semantic = bool(delta.get("content"))
                for call in delta.get("tool_calls", []):
                    entry = calls.setdefault(
                        call["index"], {"name": "", "arguments": ""}
                    )
                    function = call.get("function", {})
                    for field in ("name", "arguments"):
                        entry[field] += function.get(field, "")
                        semantic |= bool(function.get(field))
                if semantic and first_semantic is None:
                    first_semantic = (time.monotonic() - started) * 1000
                if item.get("finish_reason"):
                    finish = item["finish_reason"]
    assert done and finish == "tool_calls" and first_semantic is not None
    assert list(calls) == [0], calls
    assert calls[0]["name"] == "read_file", calls
    assert json.loads(calls[0]["arguments"]) == {"path": expected_path}, calls
    stream_ms = (time.monotonic() - started) * 1000

    # A longer literal-copy task measures sustained decode while checking
    # every output character, not merely the existence of a completion.
    expected = "\n".join(f"row_{i:02d},value_{i:02d}" for i in range(24))
    transcription = {
        "model": request["model"],
        "messages": [{
            "role": "user",
            "content": "Copy these CSV rows exactly. Output only the rows, with no markdown or explanation:\n\n" + expected,
        }],
        "temperature": 0,
        "max_tokens": 512,
        "stream": False,
    }
    for setting in ("chat_template_kwargs", "hf2q_enable_thinking", "reasoning_effort"):
        if setting in request:
            transcription[setting] = copy.deepcopy(request[setting])
    copied, copied_ms = post("transcription", transcription)
    choice = copied["choices"][0]
    assert choice["finish_reason"] == "stop", choice
    assert choice["message"]["content"].strip() == expected, choice
    concurrency_pass = None
    if args.concurrency:
        jobs = []
        references = []
        for slot in range(4):
            text = "\n".join(f"slot_{slot}_row_{i:02d},value_{i:02d}" for i in range(8))
            payload = copy.deepcopy(transcription)
            payload["messages"][0]["content"] = (
                f"Slot {slot}: copy these CSV rows exactly, without markdown or explanation:\n\n" + text
            )
            result, _ = post(f"serial-slot-{slot}", payload)
            choice = result["choices"][0]
            assert choice["finish_reason"] == "stop", choice
            assert choice["message"]["content"].strip() == text, choice
            jobs.append(payload)
            references.append(choice["message"]["content"])
        assert len(set(references)) == 4, "vacuous slot-isolation fixture"
        barrier = threading.Barrier(4)

        def concurrent_request(slot):
            barrier.wait(timeout=10)
            return post(f"concurrent-slot-{slot}", jobs[slot])[0]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(concurrent_request, range(4)))
        for slot, result in enumerate(results):
            choice = result["choices"][0]
            assert choice["finish_reason"] == "stop", choice
            assert choice["message"]["content"] == references[slot], choice
        concurrency_pass = True
    report = {
        "status": "pass",
        "engine": args.engine,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "tool_result_sha256": hashlib.sha256(tool_result.encode()).hexdigest(),
        "cold_tool": metrics(cold, cold_ms),
        "cached_tool": metrics(cached, cached_ms),
        "tool_result": metrics(continued, continued_ms),
        "stream_first_semantic_ms": first_semantic,
        "stream_complete_tool_ms": stream_ms,
        "transcription": metrics(copied, copied_ms),
        "concurrent_four_slot_parity": concurrency_pass,
    }
    save("report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
