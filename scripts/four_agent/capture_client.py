#!/usr/bin/env python3
"""Capture supported OpenCode wire settings through a localhost model stub.

This redirects the selected model endpoint; it is not a network sandbox. Only
known provider/model fields are copied into a private, isolated profile.
"""
import copy
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time

SUPPORTED_PACKAGE = "aisdk:@ai-sdk/openai-compatible"
PROJECT_MARKERS = (".git", "opencode.json", "opencode.jsonc", ".opencode", "AGENTS.md", "CLAUDE.md", ".claude")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def parse_config(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate configuration key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def object_keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(label + ": unsupported object keys")


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + ": nonempty text required")


def selected_profile(source, model):
    object_keys(source, {"$schema", "model", "providers", "plugins"}, "configuration")
    selected = model or source.get("model")
    text(selected, "selected model")
    if "/" not in selected:
        raise ValueError("selected model requires provider/model")
    provider_name, model_name = selected.split("/", 1)
    providers = source.get("providers")
    if not isinstance(providers, dict):
        raise ValueError("providers: object required")
    provider = providers[provider_name]
    object_keys(provider, {"name", "package", "settings", "models"}, "provider")
    if provider.get("package") != SUPPORTED_PACKAGE:
        raise ValueError("unsupported provider package")
    if "name" in provider:
        text(provider["name"], "provider name")
    settings = provider.get("settings", {})
    object_keys(settings, {"baseURL", "apiKey"}, "provider settings")
    text(settings.get("baseURL"), "source baseURL")
    if "apiKey" in settings:
        text(settings["apiKey"], "source apiKey")
    models = provider.get("models")
    if not isinstance(models, dict):
        raise ValueError("models: object required")
    definition = models[model_name]
    object_keys(definition, {"modelID", "name", "capabilities", "compatibility", "cost", "limit"}, "model")
    text(definition.get("modelID"), "modelID")
    if "name" in definition:
        text(definition["name"], "model name")
    caps = definition.get("capabilities", {})
    object_keys(caps, {"tools", "input", "output"}, "model capabilities")
    if "tools" in caps and type(caps["tools"]) is not bool:
        raise ValueError("model tools capability must be boolean")
    for key, supported in (("input", {"text", "image"}), ("output", {"text"})):
        if key in caps and (not isinstance(caps[key], list) or not caps[key]
                            or any(not isinstance(v, str) or v not in supported for v in caps[key])):
            raise ValueError("unsupported model capability " + key)
    compatibility = definition.get("compatibility", {})
    object_keys(compatibility, {"reasoningField"}, "model compatibility")
    if compatibility and compatibility["reasoningField"] != "reasoning_content":
        raise ValueError("unsupported reasoning field")
    for key, allowed in (("cost", {"input", "output"}), ("limit", {"context", "output"})):
        values = definition.get(key, {})
        object_keys(values, allowed, "model " + key)
        for value in values.values():
            if key == "limit":
                if type(value) is not int or value <= 0:
                    raise ValueError("model limits must be positive integers")
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("model costs must be finite nonnegative numbers")
    # Construct rather than copying arbitrary provider data into executable config.
    safe = {"package": SUPPORTED_PACKAGE, "models": {model_name: copy.deepcopy(definition)}}
    if "name" in provider:
        safe["name"] = provider["name"]
    return selected, provider_name, model_name, safe


def outside_checkout(path):
    resolved = path.resolve()
    for ancestor in (resolved, *resolved.parents):
        marker = ancestor / ".git"
        if marker.exists() or marker.is_symlink():
            raise ValueError("capture output must be outside all Git checkouts")
    return resolved


def clean_scratch(path):
    for ancestor in (path, *path.parents):
        if any((ancestor / name).exists() or (ancestor / name).is_symlink() for name in PROJECT_MARKERS):
            raise ValueError("scratch cwd has ancestor project configuration: " + str(ancestor))


def write_private(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)


def live_group_members(group):
    result = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True, check=True)
    members = []
    for row in result.stdout.splitlines():
        fields = row.split()
        if len(fields) == 3 and int(fields[1]) == group and not fields[2].startswith("Z"):
            members.append(int(fields[0]))
    return members


def stop_group(process, grace=1.0):
    group = process.pid  # start_new_session=True makes this our owned group.
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return False
    deadline = time.monotonic() + grace
    while live_group_members(group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.025)
    escalated = bool(live_group_members(group))
    if escalated:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while live_group_members(group) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.025)
    process.wait(timeout=5)
    if live_group_members(group):
        raise RuntimeError("owned client process group did not terminate")
    return escalated


def request_kind(body):
    systems = [m.get("content", "") for m in body.get("messages", [])
               if isinstance(m, dict) and m.get("role") == "system"]
    if any(isinstance(s, str) and s.startswith("You are a title generator.") for s in systems):
        return "title", "recognized-title-system-marker"
    if body.get("tools"):
        return "agent", "tools-present"
    if any(isinstance(s, str) and s.startswith("You are an AI agent running in OpenCode") for s in systems):
        return "agent", "recognized-agent-system-marker"
    return "other", "unclassified"


def capture(client, source, output, model=None, timeout=90, *, source_bytes=None, cleanup_grace=1.0):
    selected, provider_name, model_name, provider = selected_profile(source, model)
    if source_bytes is not None and canonical(parse_config(source_bytes)) != canonical(source):
        raise ValueError("source bytes do not match parsed configuration")
    output = outside_checkout(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    output.chmod(0o700)
    requests, lock = [], threading.Lock()
    server = thread = process = None
    server_started = False
    escalated = False

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": [
                {"id": provider["models"][model_name]["modelID"], "object": "model", "owned_by": "stub"}
            ]}).encode())

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError("invalid body length")
                body = parse_config(self.rfile.read(size))
                if (not isinstance(body, dict) or not isinstance(body.get("model"), str)
                        or not isinstance(body.get("messages"), list) or not body["messages"]
                        or body["model"] != provider["models"][model_name]["modelID"]
                        or ("stream" in body and type(body["stream"]) is not bool)
                        or ("tools" in body and not isinstance(body["tools"], list))):
                    raise ValueError("invalid chat request shape")
            except (ValueError, TypeError, OSError):
                self.send_error(400)
                return
            with lock:
                requests.append({"path": self.path, "request": body})
                write_private(output / "requests.json", json.dumps(requests, indent=2, allow_nan=False) + "\n")
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta, finish in [({"role": "assistant", "content": "CAPTURED"}, None), ({}, "stop")]:
                    chunk = {"id": "capture", "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": body["model"], "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                result = {"id": "capture", "object": "chat.completion", "created": int(time.time()),
                          "model": body["model"], "choices": [{"index": 0, "message": {
                              "role": "assistant", "content": "CAPTURED"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

    class Server(ThreadingHTTPServer):
        daemon_threads = False  # server_close joins bounded handler threads.

    with tempfile.TemporaryDirectory(prefix="hf2q-wire-cwd-") as scratch:
        cwd = Path(scratch).resolve()
        clean_scratch(cwd)
        try:
            server = Server(("127.0.0.1", 0), Handler)
            provider["settings"] = {"baseURL": f"http://127.0.0.1:{server.server_port}/v1",
                                    "apiKey": "synthetic-local-capture"}
            profile = {"model": selected, "providers": {provider_name: provider}}
            config = output / "config/opencode"
            config.mkdir(parents=True)
            write_private(config / "opencode.json", json.dumps(profile, indent=2) + "\n")
            env = {key: os.environ[key] for key in ("PATH", "LANG", "SHELL") if key in os.environ}
            env.update(HOME=str(output / "home"), XDG_CONFIG_HOME=str(output / "config"),
                       XDG_DATA_HOME=str(output / "data"), XDG_STATE_HOME=str(output / "state"),
                       XDG_CACHE_HOME=str(output / "cache"), TMPDIR=str(cwd))
            for key in ("HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
                Path(env[key]).mkdir(mode=0o700)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            server_started = True
            version_process = subprocess.Popen([client, "--version"], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, umask=0o077)
            try:
                version, _ = version_process.communicate(timeout=15)
                if version_process.returncode != 0:
                    raise subprocess.CalledProcessError(version_process.returncode, [client, "--version"])
            finally:
                stop_group(version_process, cleanup_grace)
            command = [client, "run", "--standalone", "--format", "json", "--model", selected,
                       "Reply with CAPTURED. Do not call tools."]
            descriptor = os.open(output / "client.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as log:
                process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, umask=0o077)
                code = process.wait(timeout=timeout)
        finally:
            try:
                if process is not None:
                    escalated = stop_group(process, cleanup_grace)
            finally:
                if server is not None:
                    if server_started:
                        server.shutdown()
                        thread.join(timeout=5)
                    server.server_close()
    if code != 0 or not requests:
        raise ValueError(f"client did not complete a capture (exit {code}); inspect client.log")
    fields = ("model", "stream", "temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
              "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty", "max_tokens",
              "max_completion_tokens", "reasoning", "reasoning_effort", "thinking", "thinking_token_budget",
              "tool_choice", "parallel_tool_calls", "chat_template_kwargs", "response_format", "stop", "logit_bias")
    with lock:
        captured = copy.deepcopy(requests)
    normalized_profile = copy.deepcopy(profile)
    normalized_profile["providers"][provider_name]["settings"]["baseURL"] = "http://127.0.0.1:STUB/v1"
    summary = {"client_version": version.strip(),
               "source_config_file_sha256": hashlib.sha256(source_bytes).hexdigest() if source_bytes is not None else None,
               "source_config_semantic_sha256": hashlib.sha256(canonical(source)).hexdigest(),
               "effective_profile_semantic_sha256": hashlib.sha256(canonical(normalized_profile)).hexdigest(),
               "requests_sha256": hashlib.sha256((output / "requests.json").read_bytes()).hexdigest(),
               "profile": "allowlisted provider/model; source endpoint redirected to localhost stub",
               "network_isolation": "configuration-redirection-only; not an outbound-network sandbox",
               "owned_process_group_terminated": True, "cleanup_escalated": escalated,
               "model_inference_performed": False, "requests": []}
    for row in captured:
        body = row["request"]
        kind, basis = request_kind(body)
        summary["requests"].append({"path": row["path"], "kind": kind, "classification_basis": basis,
            "keys": sorted(body), "wire_settings": {k: body[k] for k in fields if k in body},
            "tool_count": len(body.get("tools", []))})
    write_private(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", default="opencode")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new directory outside all Git checkout ancestors")
    parser.add_argument("--model", help="provider/model key present in the supplied configuration")
    args = parser.parse_args()
    client = shutil.which(args.client)
    if not client:
        parser.error("OpenCode executable is unavailable")
    try:
        source_bytes = args.config.read_bytes()  # One read binds parsed semantics and raw-file digest.
        summary = capture(client, parse_config(source_bytes), args.output, args.model, source_bytes=source_bytes)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"client capture failed: {exc}\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
