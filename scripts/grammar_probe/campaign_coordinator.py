#!/usr/bin/env python3
"""Coordinate pinned legacy generation and owned managed judging from local JSON.
Paths/process IDs belong in configuration. Old servers are verified by PID,
start time and command; new children are owned, never terminated by port.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def hash_file(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def capture(argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()

def process_identity(pid):
    try:
        started = capture(["ps", "-p", str(pid), "-o", "lstart="])
        command = capture(["ps", "-p", str(pid), "-o", "command="])
    except subprocess.CalledProcessError:
        return None
    return {"pid": pid, "lstart": started, "command": command}

def require_process(expected, allow_exited=False):
    actual = process_identity(expected["pid"])
    if actual is None and allow_exited:
        return False
    if actual != {k: expected[k] for k in ("pid", "lstart", "command")}:
        raise ValueError("recorded process identity changed or disappeared")
    return True

def descendants_include(pid, ancestor):
    while pid > 1:
        if pid == ancestor:
            return True
        try:
            pid = int(capture(["ps", "-p", str(pid), "-o", "ppid="]))
        except (subprocess.CalledProcessError, ValueError):
            return False
    return False

# Guarded process names. "hf2q" is ALWAYS guarded (one model at a time is a
# memory-safety rule, not a measurement-preference). "cargo"/"rustc" guard
# latency-sensitive and binary-identity-sensitive campaigns; a SEMANTIC
# campaign (deterministic temperature-0 outcomes — refusal/fulfillment/
# degeneration verdicts and finish reasons — invariant to host CPU load)
# may narrow the guard to hf2q only via "guarded_processes" in its config,
# allowing parallel build lanes on the host. Latency fields recorded during
# such a campaign must not be quoted as performance evidence.
DEFAULT_GUARDED = ("hf2q", "cargo", "rustc")


def guard_host(allowed_pids=(), owned_roots=(), names=DEFAULT_GUARDED):
    for name in names:
        found = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True)
        for value in found.stdout.split():
            pid = int(value)
            if pid not in allowed_pids and not any(descendants_include(pid, p) for p in owned_roots):
                raise ValueError(f"host contention: unexpected {name} process {pid}")

def listener_pids(port):
    found = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
                           capture_output=True, text=True)
    return {int(pid) for pid in found.stdout.split()}

def verify_pins(pins):
    for pin in pins:
        if hash_file(pin["path"]) != pin["sha256"]:
            raise ValueError(f"pinned artifact changed: {pin['path']}")

def verify_results(path, expected_rows, config_sha256):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    keys = {(r["arm"], r["prompt_id"], r["rep"], r.get("budget")) for r in rows}
    if len(rows) != expected_rows or len(keys) != expected_rows:
        raise ValueError("legacy generation inventory is incomplete or contains duplicate cells")
    if any(r.get("config_sha256") != config_sha256 or "error" in r or "content" not in r for r in rows):
        raise ValueError("legacy generation contains a changed configuration or failed response")
    return hash_file(path)

def stop_owned(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=70)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)

def stop_recorded_server(expected, timeout=60, require_listener=True):
    require_process(expected)
    if require_listener and listener_pids(expected["port"]) != {expected["pid"]}:
        raise ValueError("old server listener identity changed")
    os.kill(expected["pid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while require_process(expected, allow_exited=True):
        if time.monotonic() > deadline:
            raise ValueError("old server did not stop gracefully; manual review required")
        time.sleep(1)
    if require_listener and listener_pids(expected["port"]):
        raise ValueError("another listener replaced the old server")

def source_identity(build):
    root = build["worktree"]
    if capture(["git", "rev-parse", "HEAD"], root) != build["commit"]:
        raise ValueError("build worktree is not at the configured commit")
    if capture(["git", "status", "--porcelain", "--untracked-files=no"], root):
        raise ValueError("build worktree has tracked modifications")


class Coordinator:
    def __init__(self, config):
        self.config = config
        guarded = tuple(config.get("guarded_processes", DEFAULT_GUARDED))
        if "hf2q" not in guarded:
            raise ValueError("guarded_processes must always include hf2q (one model at a time)")
        self.guarded = guarded
        self.output = Path(config["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = {"schema_version": 1, "phase": "starting", "completed_commands": []}
        self.judge = None
        self.judge_server = None
        self.judge_log = None
        self.active_child = None

    def remember_judge_server(self, manifest):
        """Retain verified ownership before a crashed launcher can orphan it."""
        pid = manifest.get("process_pid")
        if self.judge_server:
            if pid != self.judge_server["pid"]:
                raise ValueError("managed judge process identity changed")
            return
        if pid and self.judge and descendants_include(pid, self.judge.pid):
            self.judge_server = process_identity(pid)

    def status(self, phase, **fields):
        self.state.update(phase=phase, updated=time.time(), **fields)
        path = self.output / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        os.replace(temporary, path)

    def run_command(self, step, allowed=(), extra_env=None, cwd=None):
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "TMPDIR", "LANG", "USER", "LOGNAME")}
        env.update(step.get("env", {})); env.update(extra_env or {})
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        guard_host(allowed, [self.judge.pid] if self.judge else [], names=self.guarded)
        self.status(step["name"], command=step["argv"], cwd=cwd)
        with (self.output / (step["name"] + ".log")).open("ab") as log:
            child = subprocess.Popen(step["argv"], env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
            self.active_child = child
            try:
                while child.poll() is None:
                    if self.judge is not None and self.judge.poll() is not None:
                        raise ValueError("managed judge exited during a command")
                    if allowed and listener_pids(self.config["server_process"]["port"]) != set(allowed):
                        raise ValueError("old server listener changed during generation")
                    guard_host(allowed, [child.pid] + ([self.judge.pid] if self.judge else []), names=self.guarded)
                    time.sleep(1)
                if child.returncode:
                    raise ValueError(f"{step['name']} exited {child.returncode}")
            finally:
                stop_owned(child); self.active_child = None
        self.state["completed_commands"].append(step["name"])

    def run(self):
        c = self.config; legacy = c["legacy"]; server = c["server_process"]
        if c.get("retired_process") and require_process(c["retired_process"], allow_exited=True):
            raise ValueError("the superseded coordinator is still active")
        verify_pins(c["pinned_files"])
        runner = next((a for a in legacy["argv"] if a.endswith(".py")), None)
        if runner not in {p["path"] for p in c["pinned_files"]}:
            raise ValueError("legacy command does not execute a pinned runner")
        self.status("waiting_initial", configuration=c)
        while require_process(c["initial_process"], allow_exited=True):
            require_process(server); guard_host([server["pid"]], names=self.guarded)
            if listener_pids(server["port"]) != {server["pid"]}:
                raise ValueError("old server listener changed while waiting")
            time.sleep(5)
        require_process(server); verify_pins(c["pinned_files"])
        self.run_command({"name": "legacy_generation", **legacy}, [server["pid"]])
        verify_pins(c["pinned_files"])
        self.status("generation_complete", legacy_results_sha256=verify_results(
            legacy["results_path"], legacy["expected_rows"], legacy["config_sha256"]))
        stop_recorded_server(server)
        build = c["build"]; source_identity(build); guard_host(names=self.guarded)
        if not {"build", "--release", "--locked"}.issubset(build["argv"]):
            raise ValueError("build command must use build --release --locked")
        self.run_command({"name": "build", **build}, cwd=build["worktree"])
        source_identity(build)
        binary = Path(build["binary"]).resolve()
        if not binary.is_relative_to(Path(build["worktree"]).resolve()):
            raise ValueError("built binary must reside in the verified worktree")
        self.status("build_complete", source_commit=build["commit"], binary_sha256=hash_file(binary))
        judge = c["judge"]
        if listener_pids(judge["port"]):
            raise ValueError("judge port already has a listener")
        guard_host(names=self.guarded); self.judge_log = (self.output / "managed_judge.log").open("ab")
        argv = [sys.executable, "-B", judge["managed_runner"], "--binary", str(binary),
                "--model", judge["model"], "--manifest", judge["manifest"], "--port", str(judge["port"])]
        self.judge = subprocess.Popen(argv, stdout=self.judge_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + judge.get("startup_timeout", 900)
        manifest = None
        while self.judge.poll() is None and time.monotonic() < deadline:
            guard_host(owned_roots=[self.judge.pid], names=self.guarded)
            if Path(judge["manifest"]).exists():
                manifest = json.loads(Path(judge["manifest"]).read_text())
                self.remember_judge_server(manifest)
                if manifest.get("state") == "running":
                    break
                if manifest.get("state") in ("invalid", "stopped"):
                    raise ValueError("managed judge failed during startup")
            time.sleep(1)
        if not manifest or manifest.get("state") != "running" or self.judge.poll() is not None:
            raise ValueError("managed judge did not become ready")
        if listener_pids(judge["port"]) != {manifest["process_pid"]}:
            raise ValueError("judge listener is not the managed child")
        if self.judge_server is None:
            raise ValueError("managed judge ancestry was not verified")
        require_process(self.judge_server)
        self.status("judging", judge_identity_sha256=manifest["runtime_identity"]["identity_sha256"])
        env = {"BASE_URL": f"http://127.0.0.1:{judge['port']}",
               "JUDGE_RUNTIME_MANIFEST": judge["manifest"], "JUDGE_MODEL": manifest["runtime_identity"]["model_id"]}
        for command in c["commands"]:
            source_identity(build)
            self.run_command(command, extra_env=env, cwd=build["worktree"])
        if self.judge.poll() is not None:
            raise ValueError("managed judge exited before cleanup")
        self.status("commands_complete")

    def cleanup(self):
        errors = []
        def attempt(action):
            try:
                action()
            except Exception as error:
                errors.append(error)
        attempt(lambda: stop_owned(self.active_child))
        if self.judge is not None and self.judge_server is None:
            try:
                manifest = json.loads(Path(self.config["judge"]["manifest"]).read_text())
                self.remember_judge_server(manifest)
            except (OSError, ValueError, TypeError, AttributeError):
                pass  # Damaged diagnostics must not prevent owned-child cleanup.
        attempt(lambda: stop_owned(self.judge))
        def stop_remembered_server():
            if self.judge_server and require_process(self.judge_server, allow_exited=True):
                stop_recorded_server(self.judge_server, require_listener=False)
        attempt(stop_remembered_server)
        if self.judge_log:
            attempt(self.judge_log.close)
        if errors:
            raise RuntimeError("owned-process cleanup failed: " + "; ".join(map(str, errors))) from errors[0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    args = parser.parse_args()
    coordinator = Coordinator(json.loads(args.configuration.read_text()))
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("coordinator interrupted")
    signal.signal(signal.SIGINT, interrupted); signal.signal(signal.SIGTERM, interrupted)
    try:
        coordinator.run()
    except BaseException as exc:
        coordinator.status("failed", error=str(exc))
        raise
    finally:
        try:
            coordinator.cleanup()
        except BaseException as exc:
            coordinator.status("failed", cleanup_error=str(exc))
            raise
    coordinator.status("complete")


if __name__ == "__main__":
    main()
