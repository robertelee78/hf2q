"""ROOT-only, approved model-free OpenCode capture. Import never runs capture.
The stub is scripted measurement infrastructure, not a model/quality result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from q2_certify import BoundReader, digest
from q2_capture_packet import validate_capture_approval
from q2_capture_preflight import quiet_inventory, default_resolution, standalone_capture_argv
from q2_client_profile import (FAMILIES, capture_profile, encoded, projected_config,
                               validate_capture_request)
from qwen38_n1_verdict import loads, require
import qwen38_n1_process as owned

CLOCK = "CLOCK_MONOTONIC"
BODY_LIMIT = 2 * 1024 * 1024
MAX_HTTP = 32
CLIENT_SECONDS = 120
INTERRUPTED = False


def clock():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def interrupt(_signum, _frame):
    global INTERRUPTED
    INTERRUPTED = True


def save(path, value, raw=False):
    data = value if raw else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def quiet(service, binary):
    result = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,lstart=,comm=,args="], capture_output=True, timeout=2, check=True)
    require(not result.stderr, "quiet inventory stderr")
    return quiet_inventory(result.stdout, service, binary)


def sandbox_profile(root, binary, port):
    # Empty HOME/config/cwd removes repository/user hooks. File reads cannot
    # reach old user configuration, credentials, plugins or Ruflo packages.
    for value in (str(root), str(binary)):
        require(not any(c in value for c in ('"', '\\', '\n')), "sandbox path quoting")
    return '\n'.join([
        '(version 1)', '(deny default)', '(allow process*)', '(allow sysctl-read)',
        '(allow mach-lookup)', '(allow file-read* (literal "/"))',
        '(allow file-read* (subpath "/System") (subpath "/usr") (subpath "/bin") (subpath "/Library/Apple") (subpath "/dev"))',
        '(allow file-map-executable)',
        f'(allow file-read* (literal "{binary}") (subpath "{root}"))',
        f'(allow file-write* (subpath "{root}") (literal "/dev/null"))',
        f'(allow network-outbound (remote tcp "127.0.0.1:{port}"))',
        '(allow file-read* (literal "/private/etc/localtime") (literal "/private/etc/ssl/openssl.cnf"))',
    ]) + '\n'


def child(command, root, env, output, maximum_seconds):
    require(not INTERRUPTED, "interrupted before actual owned Popen")
    process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    known = {}; raw = {"stdout": bytearray(), "stderr": bytearray()}
    receipt = {"argv": command, "f2_clock_id": CLOCK, "started_ns": clock(), "cleanup": None}
    selector = selectors.DefaultSelector(); started = time.monotonic()
    try:
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False); selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            require(not INTERRUPTED and time.monotonic() - started < maximum_seconds, "owned client deadline/interruption")
            rows = owned.process_snapshot(); pids = owned.tree(rows, process.pid, known)
            known.update({p: rows[p][3] for p in pids if p in rows})
            require(sum(rows[p][2] for p in pids if p in rows) <= 1024 * 1024 * 1024, "capture client observed RSS1GiB")
            owned.directory_bytes(root, 256 * 1024 * 1024)
            for key, _ in selector.select(.05):
                data = os.read(key.fileobj.fileno(), 8192)
                if data: raw[key.data].extend(data)
                else: selector.unregister(key.fileobj)
                require(sum(map(len, raw.values())) <= 8 * 1024 * 1024, "capture stdout/stderr quota")
        receipt["exit"] = process.wait(timeout=1)
    except BaseException as error:
        receipt["failure"] = repr(error)
    finally:
        # Metrics failures never gate signaling the freshly owned PG/reaping.
        try: receipt["cleanup"] = owned.stop_group(process, known)
        except BaseException as error: receipt["cleanup_error"] = repr(error)
        for name, data in raw.items(): save(output / (name + ".bin"), bytes(data), True)
        selector.close()
        for stream in (process.stdout, process.stderr): stream.close()
        receipt["finished_ns"] = clock(); save(output / "receipt.json", receipt)
    require(receipt.get("exit") == 0 and not receipt.get("failure") and not receipt.get("cleanup_error")
            and receipt.get("cleanup", {}).get("verified") is True, "owned client failed/unreaped")
    return bytes(raw["stdout"]), receipt


class Stub(HTTPServer):
    # One bounded callback at a time: no unbounded handler-thread join.
    allow_reuse_address = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args): pass

    def setup(self):
        super().setup(); self.connection.settimeout(2)

    def do_GET(self):
        self.send_error(405)  # Proven client needs only chat completions; fail rather than fake discovery.

    def do_POST(self):
        state = self.server.state
        with state["lock"]:
            try:
                index = state["http_count"]; state["http_count"] += 1
                require(index < MAX_HTTP, "bounded auxiliary and primary calls")
                headers = [[k, "REDACTED_BEFORE_PERSISTENCE" if k.lower() in {"authorization", "proxy-authorization", "cookie", "x-api-key", "api-key"} else v]
                           for k, v in self.headers.items()]
                save(state["root"] / f"http-{index}-headers.json", {"request_line": self.requestline, "headers": headers})
                require(not INTERRUPTED and self.path == "/v1/chat/completions", "unexpected endpoint")
                require(not self.headers.get("Transfer-Encoding") and self.headers.get("Content-Type", "").startswith("application/json"), "HTTP framing/type")
                require(len(self.headers.get_all("Content-Length", [])) == 1 and
                        len(self.headers.get_all("Content-Type", [])) == 1, "duplicate HTTP framing headers")
                length = self.headers.get("Content-Length", "")
                require(re.fullmatch(r"[0-9]+", length) and 0 < int(length) <= BODY_LIMIT, "bounded explicit body")
                raw = self.rfile.read(int(length))
                body_ref = save(state["root"] / f"http-{index}-request.json", raw, True)
                state["all"].append(body_ref)
                require(len(raw) == int(length), "truncated actual body")
                request = loads(raw); require(request.get("model") == state["model"], "unbound auxiliary/model fallback")
                tool_request = bool(request.get("tools"))
                if tool_request:
                    phase = len(state["phases"]); require(phase < 7, "extra actual tool phase")
                    controls = validate_capture_request(request, state["model"],
                        None if phase == 0 else f"q2_capture_{phase-1}",
                        None if phase == 0 else str(state["fixture"] / f"phase-{phase-1}.txt"))
                    record = {"phase": phase, "request": body_ref, "tools": request["tools"],
                              "controls": controls, "f2_clock_id": CLOCK, "received_ns": clock()}
                    state["phases"].append(record)
                    if phase < 6:
                        delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": f"q2_capture_{phase}",
                            "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": str(state["fixture"] / f"phase-{phase}.txt")})}}]}
                        finish = "tool_calls"
                    else: delta = {"role": "assistant", "content": "Q2_CAPTURE_COMPLETE"}; finish = "stop"
                else:
                    # Title/metadata requests remain raw auxiliary captures; not
                    # recast as one of the seven agent tool phases.
                    delta = {"role": "assistant", "content": "Q2 metadata"}; finish = "stop"
                require(request.get("stream") is True, "unexpected nonstream actual client mode")
                frames = []
                for chunk, stop in ((delta, None), ({}, finish)):
                    frames.append("data: " + json.dumps({"id": f"q2_stub_{index}", "object": "chat.completion.chunk",
                        "created": int(time.time()), "model": request["model"], "choices": [{"index": 0, "delta": chunk, "finish_reason": stop}]}))
                response = ("\n\n".join(frames) + "\n\ndata: [DONE]\n\n").encode()
                save(state["root"] / f"http-{index}-stub-response.bin", response, True)
                self.send_response(200); self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(response))); self.send_header("Connection", "close")
                self.end_headers(); self.wfile.write(response); self.wfile.flush(); self.close_connection = True
            except BaseException as error:
                if len(state["errors"]) < MAX_HTTP: state["errors"].append(repr(error))
                self.send_error(500); self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", required=True); parser.add_argument("--approval-sha256", required=True)
    parser.add_argument("--output", required=True); args = parser.parse_args()
    require(digest(args.approval) == args.approval_sha256, "external capture approval hash")
    reader = BoundReader(); approval = validate_capture_approval(loads(Path(args.approval).read_bytes()))
    # The actual shared validator above owns the closed schema, including its
    # single exact-service exception. Do not reapply an older duplicate schema.
    for name, sha in approval["source_files"].items():
        require(Path(name).name == name, "capture source basename")
        reader.path({"path": str(Path(__file__).resolve().with_name(name)), "sha256": sha})
    require({"q2_capture_client.py", "q2_client_profile.py", "q2_certify.py", "qwen38_n1_process.py", "qwen38_n1_verdict.py"}
            <= set(approval["source_files"]), "capture source closure")
    binary = reader.path(approval["client_binary"], 512 * 1024 * 1024)
    sandbox = reader.path(approval["sandbox_binary"])
    require(sandbox == Path("/usr/bin/sandbox-exec"), "reviewed macOS sandbox")
    source = reader.raw(approval["source_config"], 1048576)  # Never persisted: may contain a credential.
    require(type(approval["port"]) is int and 1024 <= approval["port"] <= 65535, "owned loopback port")
    output = Path(args.output)
    require(output.is_absolute() and output.parent == output.parent.resolve() and not output.exists()
            and not any((p / ".git").exists() for p in (output.parent, *output.parent.parents)), "new private packet outside Git")
    output.mkdir(mode=0o700)
    for signum in (signal.SIGINT, signal.SIGTERM): signal.signal(signum, interrupt)
    save(output / "approval.json", approval)
    result = {"schema": "hf2q.q2.client-capture-result.v1", "status": "Pending", "model_inference": False,
              "campaign_primary_eligible": False, "effective_sampling_qualified": False, "profiles": {}, "client_binary": approval["client_binary"], "f2_clock_id": CLOCK}
    try:
        for family in sorted(FAMILIES):
            save(output / ("quiet-" + family + ".json"), quiet(approval.get("existing_client_service"), approval["client_binary"]))
            root = output / family; root.mkdir(mode=0o700)
            fixture = root / "fixture"; fixture.mkdir(mode=0o700)
            for i in range(6): save(fixture / f"phase-{i}.txt", f"Q2_OWNED_CAPTURE_{i}\n".encode(), True)
            cfg = approval["profiles"][family]
            require(set(cfg) == {"model_id", "artifact_sha256", "effective_hf_defaults"}, "explicit derived profile bindings")
            defaults = reader.obj(cfg["effective_hf_defaults"])
            require(isinstance(cfg["artifact_sha256"], str) and re.fullmatch("[a-f0-9]{64}", cfg["artifact_sha256"]), "actual artifact identity")
            default_disposition = default_resolution(defaults, family, reader)
            effective, projection = projected_config(source, family, cfg["model_id"], f"http://127.0.0.1:{approval['port']}/v1")
            for name in ("home", "config/opencode", "data", "state", "cache", "version", "client"):
                (root / name).mkdir(parents=True, mode=0o700)
            save(root / "config/opencode/opencode.json", effective); save(root / "config/opencode/cli.json", {"plugins": []})
            sb = root / "capture.sb"; save(sb, sandbox_profile(root, binary, approval["port"]).encode(), True)
            env = {"PATH": "/usr/bin:/bin", "HOME": str(root / "home"), "XDG_CONFIG_HOME": str(root / "config"),
                   "XDG_DATA_HOME": str(root / "data"), "XDG_STATE_HOME": str(root / "state"), "XDG_CACHE_HOME": str(root / "cache"),
                   "TMPDIR": str(root), "LANG": "C", "LC_ALL": "C", "SHELL": "/bin/sh"}
            state = {"root": root, "fixture": fixture, "model": cfg["model_id"], "all": [], "http_count": 0, "phases": [], "errors": [], "lock": threading.Lock()}
            server = Stub(("127.0.0.1", approval["port"]), Handler); server.state = state
            thread = threading.Thread(target=server.serve_forever); thread.start()
            try:
                version, _ = child([str(sandbox), "-f", str(sb), str(binary), "--version"], root, env, root / "version", 10)
                require(version.decode().strip() == approval["expected_version"] and not state["all"] and not state["errors"],
                        "actual client binary/version or unexpected version HTTP")
                require(not INTERRUPTED, "interrupted immediately before capture child")
                capture_argv = [str(sandbox), "-f", str(sb), str(binary), "run", "--standalone", "--format", "json",
                    "--model", "hf2q/" + cfg["model_id"], "Diagnostic only: follow the provided read tool calls and then acknowledge completion. Do not call other tools."]
                _, receipt = child(standalone_capture_argv(capture_argv, sandbox, sb, binary), root, env, root / "client", CLIENT_SECONDS)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=3)
                save(root / "capture-state.json", {k: v for k, v in state.items() if k not in {"lock", "root", "fixture"}})
            require(not thread.is_alive() and not state["errors"], "stub error or thread cleanup unknown")
            profile = capture_profile(state["phases"], cfg["model_id"])
            profile.update(projection=projection, client_version=version.decode().strip(), client_binary=approval["client_binary"],
                           artifact_sha256=cfg["artifact_sha256"], effective_hf_defaults=cfg["effective_hf_defaults"],
                           effective_hf_defaults_value=defaults, default_resolution_disposition=default_disposition,
                           effective_sampling_qualified=False, cleanup=receipt["cleanup"])
            result["profiles"][family] = save(root / "profile.json", profile)
        reader.verify(); save(output / "quiet-final.json", quiet(approval.get("existing_client_service"), approval["client_binary"])); result["status"] = "CapturedControlsAwaitingIndependentReview"
    except BaseException as error: result.update(status="Failed", failure=repr(error))
    save(output / "receipt.json", result)
    hashes = {}
    for path in output.rglob("*"):
        if path.is_file():
            require(not path.is_symlink(), "capture artifact symlink")
            hashes[str(path.relative_to(output))] = digest(path)
    save(output / "artifact-hashes.json", hashes)
    return 0 if result["status"] == "CapturedControlsAwaitingIndependentReview" else 1


if __name__ == "__main__":
    raise SystemExit(main())
