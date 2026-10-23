#!/usr/bin/env python3
"""Capture OpenCode wire settings against a synthetic localhost model stub.

Copies only the selected provider/model into an isolated temporary profile.
Never edits the source configuration or runs an inference model.
"""
import argparse
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time


def capture(client, source, output, model=None, timeout=90):
    selected = model or source["model"]
    provider_name, model_name = selected.split("/", 1)
    provider = copy.deepcopy(source["providers"][provider_name])
    definition = provider["models"][model_name]
    output.mkdir(parents=True, exist_ok=False)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": [
                {"id": definition.get("modelID", model_name), "object": "model", "owned_by": "stub"}
            ]}).encode())

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError("invalid body length")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("request is not an object")
            except (ValueError, TypeError):
                self.send_error(400)
                return
            requests.append({"path": self.path, "request": body})
            (output / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta, finish in [({"role": "assistant", "content": "CAPTURED"}, None), ({}, "stop")]:
                    chunk = {"id": "capture", "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": body.get("model"), "choices": [
                                 {"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                result = {"id": "capture", "object": "chat.completion", "created": int(time.time()),
                          "model": body.get("model"), "choices": [{"index": 0, "message": {
                              "role": "assistant", "content": "CAPTURED"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # No source credentials, other providers, plugins or project configuration.
    provider["settings"] = {"baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                            "apiKey": "synthetic-local-capture"}
    provider["models"] = {model_name: definition}
    profile = {"model": selected, "providers": {provider_name: provider}}
    config = output / "config/opencode"
    config.mkdir(parents=True)
    (config / "opencode.json").write_text(json.dumps(profile, indent=2) + "\n")
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "LANG", "SHELL") if key in os.environ}
    # These are child-process isolation roots, never changes to the user's shell.
    env.update(HOME=str(output / "home"), XDG_CONFIG_HOME=str(output / "config"),
               XDG_DATA_HOME=str(output / "data"), XDG_STATE_HOME=str(output / "state"),
               XDG_CACHE_HOME=str(output / "cache"))
    Path(env["HOME"]).mkdir()
    process = None
    try:
        version = subprocess.check_output([client, "--version"], cwd=output, env=env, text=True, timeout=15).strip()
        command = [client, "run", "--standalone", "--format", "json", "--model", selected,
                   "Reply with CAPTURED. Do not call tools."]
        with (output / "client.log").open("w") as log:
            process = subprocess.Popen(command, cwd=output, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=timeout)
    finally:
        if process is not None:
            # The private server and any children share our owned process group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        server.shutdown()
        server.server_close()
    if code != 0 or not requests:
        raise ValueError(f"client did not complete a capture (exit {code}); inspect client.log")
    fields = ("model", "stream", "temperature", "top_p", "top_k", "seed", "repetition_penalty",
              "presence_penalty", "frequency_penalty", "max_tokens", "max_completion_tokens")
    summary = {"client_version": version, "source_config_sha256": hashlib.sha256(
                   json.dumps(source, sort_keys=True).encode()).hexdigest(),
               "requests_sha256": hashlib.sha256((output / "requests.json").read_bytes()).hexdigest(),
               "profile": "selected provider/model only; provider settings replaced by localhost stub",
               "model_inference_performed": False, "requests": [
                   {"path": row["path"], "keys": sorted(row["request"]),
                    "wire_settings": {k: row["request"][k] for k in fields if k in row["request"]},
                    "tool_count": len(row["request"].get("tools", []))} for row in requests]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default="opencode")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory outside the product checkout")
    parser.add_argument("--model", help="provider/model key present in the supplied configuration")
    args = parser.parse_args()
    repository = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).parent, text=True).strip())
    if args.output.resolve().is_relative_to(repository):
        parser.error("capture output must be outside the product checkout")
    os.umask(0o077)
    client = shutil.which(args.client)
    if not client:
        parser.error("OpenCode executable is unavailable")
    try:
        summary = capture(client, json.loads(args.config.read_text()), args.output.resolve(), args.model)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"client capture failed: {exc}\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
