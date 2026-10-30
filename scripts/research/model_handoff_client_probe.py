#!/usr/bin/env python3
"""Capture installed OMP/OpenCode A-B-A requests against a loopback SSE fixture.

Protocol research only: no inference, model load, native KV reuse or latency
qualification. Uses temporary HOME/XDG/TMPDIR, no inherited credentials, and a
macOS sandbox denying outbound traffic except loopback. OpenCode must use its
private --standalone server, never an existing background session. Raw receipts
contain only this probe's synthetic conversation and dummy API key.
"""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time

MODELS = ["Qwen3.8-27B", "DeepSeek-V4-Flash-0731", "Qwen3.8-27B"]
SANDBOX = '(version 1) (allow default) (deny network-outbound) (allow network-outbound (remote ip "localhost:*"))'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def isolated(root):
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    for key, directory in {"HOME": "home", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
            "XDG_CACHE_HOME": "cache", "XDG_STATE_HOME": "state", "TMPDIR": "tmp",
            "PI_CODING_AGENT_DIR": "agent", "OPENCODE_TEST_HOME": "home",
            "OPENCODE_TEST_MANAGED_CONFIG_DIR": "managed"}.items():
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for flag in ("AUTOUPDATE", "MODELS_FETCH", "DEFAULT_PLUGINS", "CLAUDE_CODE", "EXTERNAL_SKILLS",
            "PROJECT_CONFIG", "AUTOCOMPACT", "PRUNE"):
        env["OPENCODE_DISABLE_" + flag] = "1"
    (root / "work").mkdir()
    (root / "sessions").mkdir()
    return env


class Fixture:
    def __init__(self, client, marker):
        self.client, self.marker = client, marker
        self.requests, self.errors = [], []

    def reply(self, handler):
        try:
            assert handler.path == "/v1/chat/completions", handler.path
            body = handler.rfile.read(int(handler.headers["Content-Length"]))
            value = json.loads(body)
            index = len(self.requests)
            self.requests.append({"path": handler.path, "headers": dict(handler.headers),
                "body_utf8": body.decode(), "json": value})
            assert index < 6, "unexpected extra model request"
            turn, continuation = divmod(index, 2)
            assert value["model"] == MODELS[turn], value["model"]
            assert value["stream"] is True
            assert handler.headers.get("Authorization") == "Bearer h0-fixture-only"
            messages = value["messages"]
            flattened = json.dumps(messages, ensure_ascii=False)
            assert f"H0_USER_{turn + 1}" in flattened
            for previous in range(1, turn + 1):
                assert f"H0_DONE_{previous}" in flattened, "earlier visible assistant content lost"
                assert f"H0_USER_{previous}" in flattened, "earlier user content lost"
                assert f"call_h0_{previous}" in flattened, "earlier tool ID lost"
            user_text = [json.dumps(m, ensure_ascii=False) for m in messages if m["role"] == "user"]
            positions = [next(i for i, text in enumerate(user_text) if f"H0_USER_{n}" in text) for n in range(1, turn + 2)]
            assert positions == sorted(set(positions)), "user chronology changed"
            calls = {call["id"]: call for m in messages for call in m.get("tool_calls", [])}
            results = {m["tool_call_id"]: m for m in messages if m["role"] == "tool"}
            for n in range(1, turn + 1 + continuation):
                call_id = f"call_h0_{n}"
                assert call_id in calls and call_id in results, "tool pairing lost"
                assert "H0_TOOL_OK" in json.dumps(results[call_id]), "tool content lost"
            if continuation:
                delta = {"content": f"H0_DONE_{turn + 1}: keep id_007 and λ exactly."}
                chunks = [(delta, None), ({}, "stop")]
            else:
                name = "echo_host" if self.client == "omp" else "read"
                definitions = {t["function"]["name"]: t["function"] for t in value["tools"]}
                assert name in definitions, f"required fixture tool {name} not advertised"
                if self.client == "omp":
                    arguments = {"message": "H0_TOOL_OK"}
                else:
                    properties = definitions[name]["parameters"]["properties"]
                    path_key = "path" if "path" in properties else "filePath"
                    assert path_key in properties, "unsupported advertised read schema"
                    arguments = {path_key: str(self.marker)}
                chunks = [({"reasoning_content": f"H0_REASON_{turn + 1}"}, None),
                    ({"tool_calls": [{"index": 0, "id": f"call_h0_{turn + 1}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)}}]}, None),
                    ({}, "tool_calls")]
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Connection", "close")
            handler.end_headers()
            for delta, finish in [({"role": "assistant"}, None), *chunks]:
                chunk = {"id": f"chatcmpl-h0-{index}", "object": "chat.completion.chunk", "created": 1,
                    "model": value["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                handler.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
            handler.close_connection = True
        except Exception as error:
            self.errors.append(repr(error))
            handler.send_error(400, "H0 fixture contract failed")


def launch(command, root, env):
    return subprocess.Popen(["/usr/bin/sandbox-exec", "-p", SANDBOX, *command], cwd=root / "work",
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1, start_new_session=True)


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def omp_capture(binary, root, env, url, output):
    provider = {"baseUrl": url, "api": "openai-completions", "apiKey": "h0-fixture-only", "models": [
        {"id": name, "name": name, "reasoning": True, "input": ["text"],
         "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
         "contextWindow": 32768, "maxTokens": 128} for name in dict.fromkeys(MODELS)]}
    # JSON is valid YAML; avoid adding a YAML dependency to this research probe.
    save(root / "agent/models.yml", {"providers": {"h0": provider}})
    process = launch([str(binary), "--mode", "rpc", "--provider", "h0", "--model", MODELS[0],
        "--session-dir", str(root / "sessions"), "--no-tools", "--no-extensions", "--no-skills",
        "--no-rules", "--no-context-files", "--no-lsp", "--no-title", "--no-prewalk", "--no-pty",
        "--system-prompt", "H0 synthetic protocol fixture. Keep exact code, reasoning and tool history."], root, env)
    events, frames, errors = queue.Queue(), [], []
    def consume(stream, destination):
        for line in stream:
            destination.append(line)
            if stream is process.stdout:
                events.put(line)
    threading.Thread(target=consume, args=(process.stdout, frames), daemon=True).start()
    threading.Thread(target=consume, args=(process.stderr, errors), daemon=True).start()
    def send(frame):
        process.stdin.write(json.dumps(frame) + "\n")
        process.stdin.flush()
    def until(predicate):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                line = events.get(timeout=1)
            except queue.Empty:
                assert process.poll() is None, "OMP exited: " + "".join(errors)[-2000:]
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if frame.get("type") == "host_tool_call":
                assert frame["toolName"] == "echo_host" and frame["arguments"] == {"message": "H0_TOOL_OK"}
                send({"type": "host_tool_result", "id": frame["id"], "result": {"content": [{"type": "text", "text": "H0_TOOL_OK"}]}})
            if frame.get("type") == "response" and frame.get("success") is False:
                raise AssertionError(frame)
            if predicate(frame):
                return frame
        raise TimeoutError("OMP RPC did not reach expected frame")
    try:
        send({"id": "tools", "type": "set_host_tools", "tools": [{"name": "echo_host", "label": "Echo Host",
            "description": "Return the synthetic marker", "parameters": {"type": "object", "properties": {
                "message": {"type": "string"}}, "required": ["message"], "additionalProperties": False}}]})
        until(lambda f: f.get("id") == "tools" and f.get("type") == "response")
        for turn, model in enumerate(MODELS, 1):
            if turn > 1:
                send({"id": f"model{turn}", "type": "set_model", "provider": "h0", "modelId": model})
                until(lambda f: f.get("id") == f"model{turn}" and f.get("type") == "response")
            send({"id": f"prompt{turn}", "type": "prompt", "message": f"H0_USER_{turn}: preserve src/λ.rs and id_007; run the marker tool."})
            result = until(lambda f: f.get("id") == f"prompt{turn}" and f.get("type") == "prompt_result")
            assert "H0_DONE_" in "".join(frames), result
        send({"id": "state", "type": "get_state"})
        state = until(lambda f: f.get("id") == "state" and f.get("type") == "response")
        return state
    finally:
        stop(process)
        (output / "omp-rpc.jsonl").write_text("".join(frames))
        (output / "omp-stderr.txt").write_text("".join(errors))


def opencode_capture(binary, root, env, url, output):
    config = {"$schema": "https://opencode.ai/config.json", "enabled_providers": ["h0"], "autoupdate": False,
        "share": "disabled", "snapshot": False, "plugin": [], "mcp": {}, "lsp": False, "formatter": False,
        "permission": {"*": "deny", "read": "allow", "external_directory": {str(root / "work") + "/**": "allow"}},
        "provider": {"h0": {"npm": "@ai-sdk/openai-compatible",
            "options": {"baseURL": url, "apiKey": "h0-fixture-only"}, "models": {name: {
                "name": name, "reasoning": True, "interleaved": {"field": "reasoning_content"},
                "limit": {"context": 32768, "output": 128}} for name in dict.fromkeys(MODELS)}}}}
    (root / "config/opencode").mkdir(exist_ok=True)
    save(root / "config/opencode/opencode.json", config)
    session = None
    for turn, model in enumerate(MODELS, 1):
        command = [str(binary), "run", "--standalone", "--format", "json", "--title", "H0 fixture", "--model", "h0/" + model]
        if session:
            command += ["--session", session]
        command += [f"H0_USER_{turn}: preserve src/λ.rs and id_007; read only the marker file."]
        process = launch(command, root, env)
        try:
            stdout, stderr = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            stop(process)
            stdout, stderr = process.communicate()
            raise TimeoutError("OpenCode standalone startup/run exceeded 60 seconds: " + stderr[-2000:])
        finally:
            stop(process)
        (output / f"opencode-{turn}.jsonl").write_text(stdout)
        (output / f"opencode-{turn}-stderr.txt").write_text(stderr)
        assert process.returncode == 0, stderr[-3000:] + stdout[-3000:]
        for line in stdout.splitlines():
            event = json.loads(line)
            session = event.get("sessionID", session)
        assert session, "OpenCode did not report an isolated session ID"
    return {"sessionID": session}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=("omp", "opencode"), required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New local receipt directory outside the repository")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="hf2q-handoff-client-") as directory:
        root = Path(directory).resolve()
        env = isolated(root)
        marker = root / "work/h0-marker.txt"
        marker.write_text("H0_TOOL_OK: let id_007 = 42; λ\n")
        fixture = Fixture(args.client, marker)
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                fixture.reply(self)
            def log_message(self, *unused):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/v1"
        receipt = {"schema": "hf2q.model-handoff-client-probe.v1", "client": args.client, "scope": __doc__,
            "binary_sha256": hashlib.file_digest(args.binary.open("rb"), "sha256").hexdigest(),
            "environment": env, "sandbox": SANDBOX, "models": MODELS, "passed": False}
        try:
            receipt["state"] = (omp_capture if args.client == "omp" else opencode_capture)(args.binary, root, env, url, output)
            assert len(fixture.requests) == 6 and not fixture.errors, fixture.errors
            receipt["passed"] = True
        except Exception as error:
            receipt["error"] = repr(error)
        finally:
            server.shutdown()
            server.server_close()
            save(output / "requests.json", fixture.requests)
            receipt["fixture_errors"] = fixture.errors
            receipt["request_count"] = len(fixture.requests)
            save(output / "receipt.json", receipt)
        print(json.dumps(receipt, indent=2))
        if not receipt["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
