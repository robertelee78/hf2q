#!/usr/bin/env python3
"""Run one local hf2q measurement server with verified artifact identity.

Hash immutable inputs before launching an owned child, bind its live runtime
snapshot, and keep the manifest valid only while that child and those files
remain unchanged. SIGINT/SIGTERM stop only this child. Existing listeners are
never stopped. Files and logs are local evidence, not publication artifacts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def stamp(path):
    s = Path(path).stat()
    return {"device": s.st_dev, "inode": s.st_ino, "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


def seal(path, role):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{role} must be a regular file")
    before = stamp(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if stamp(path) != before:
        raise ValueError(f"{role} changed while hashing")
    return {"role": role, "path": str(path), "sha256": digest,
            "bytes": before["bytes"], "stat": before}


def verify_seals(artifacts):
    for artifact in artifacts:
        if stamp(artifact["path"]) != artifact["stat"]:
            raise ValueError(f"{artifact['role']} changed after verification")


def fetch_snapshot(base_url):
    with urllib.request.urlopen(base_url + "/hf2q/v1/runtime", timeout=15) as r:
        return json.load(r).get("measurement_snapshot")


def validate_snapshot(snapshot, pid):
    if not isinstance(snapshot, dict):
        raise ValueError("runtime must expose exactly one measurement snapshot")
    if snapshot.get("schema_version") != "hf2q.measurement-snapshot.v1":
        raise ValueError("unsupported runtime snapshot schema")
    if snapshot.get("process_pid") != pid:
        raise ValueError("listener is not the owned server process")
    # These families use the GGUF tokenizer/config or the native encoder.
    # Sidecar-dependent families need explicit file binding before inclusion.
    if snapshot.get("architecture") not in (
            "deepseek4", "qwen35", "qwen35moe", "qwen3_5", "qwen3_5_moe"):
        raise ValueError("managed measurements currently support DeepSeek-V4 and Qwen3.5-family GGUFs")
    for key in ("tokenizer_sha256", "template_sha256"):
        value = snapshot.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
                c not in "0123456789abcdef" for c in value):
            raise ValueError(f"missing exact {key}")
    for control in ("glp", "grammar"):
        active = snapshot.get("active_controls", {}).get(control, {}).get("active")
        if not isinstance(active, bool):
            raise ValueError(f"unknown {control} state")


def write_manifest(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def build_manifest(snapshot, artifacts, environment, command):
    binary = next(a for a in artifacts if a["role"] == "binary")
    identity = {
        "binary_sha256": binary["sha256"],
        # A running executable does not prove the source checkout that built it.
        "source_commit": None,
        "model_artifacts": [{k: a[k] for k in ("role", "sha256", "bytes")}
                            for a in artifacts if a["role"] != "binary"],
        "tokenizer_sha256": snapshot["tokenizer_sha256"],
        "template_sha256": snapshot["template_sha256"],
        "model_id": snapshot["model_id"],
        "sampling_defaults": snapshot["sampling_defaults"],
        "active_controls": snapshot["active_controls"],
        "engine_config": snapshot["engine_config"],
        "environment_sha256": canonical_hash(environment),
    }
    identity["identity_sha256"] = canonical_hash(identity)
    return {
        "schema_version": "hf2q.measurement-runtime.v1", "state": "running",
        "process_pid": snapshot["process_pid"],
        "runtime_snapshot": snapshot,
        "runtime_snapshot_sha256": canonical_hash(snapshot),
        "runtime_identity": identity, "artifacts": artifacts,
        "environment": environment, "command": command,
    }


def child_environment(overrides):
    # Do not inherit hidden HF2Q experiment switches, setup, auth or dynamic
    # loader overrides. Record the exact narrow environment used for this run.
    names = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "USER", "LOGNAME")
    environment = {name: os.environ[name] for name in names if name in os.environ}
    for entry in overrides:
        key, separator, value = entry.partition("=")
        if not separator or not key.startswith("HF2Q_"):
            raise ValueError("--env accepts an explicit HF2Q_NAME=value only")
        if any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            raise ValueError("credential variables cannot be recorded in measurement evidence")
        environment[key] = value
    return environment


def require_free_port(port):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))


def stop_owned_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--gcd", action="store_true")
    parser.add_argument("--gcd-schema", type=Path)
    parser.add_argument("--gcd-schema-locked", action="store_true")
    parser.add_argument("--glp", type=Path)
    parser.add_argument("--glp-alpha", type=float)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        parser.error("use a fresh manifest path; old runtime evidence is immutable")
    require_free_port(args.port)
    if args.gcd_schema_locked and not args.gcd_schema:
        parser.error("--gcd-schema-locked requires --gcd-schema")
    if args.glp_alpha is not None and not args.glp:
        parser.error("--glp-alpha requires --glp")
    if args.gcd and args.gcd_schema:
        parser.error("choose one server default grammar")
    if args.model.suffix.lower() != ".gguf":
        parser.error("--model must be an existing local GGUF")
    artifacts = [seal(args.binary, "binary"), seal(args.model, "model")]
    command = [artifacts[0]["path"], "serve", "--model", artifacts[1]["path"],
               "--host", "127.0.0.1", "--port", str(args.port)]
    for flag, path, role in (("--glp", args.glp, "glp"),
                             ("--gcd-schema", args.gcd_schema, "grammar_schema")):
        if path:
            artifact = seal(path, role)
            artifacts.append(artifact)
            command.extend([flag, artifact["path"]])
    if args.gcd:
        command.append("--gcd")
    if args.gcd_schema_locked:
        command.append("--gcd-schema-locked")
    if args.glp_alpha is not None:
        command.extend(["--glp-alpha", str(args.glp_alpha)])
    environment = child_environment(args.env)
    verify_seals(artifacts)
    require_free_port(args.port)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{args.port}"
    manifest = {"schema_version": "hf2q.measurement-runtime.v1", "state": "starting"}
    stop = False

    def requested_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, requested_stop)
    signal.signal(signal.SIGINT, requested_stop)
    child = None
    try:
        with manifest_path.with_suffix(".server.log").open("x") as log:
            child = subprocess.Popen(command, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            manifest["process_pid"] = child.pid
            write_manifest(manifest_path, manifest)
            deadline = time.monotonic() + args.startup_timeout
            while not stop:
                if child.poll() is not None:
                    raise ValueError(f"owned server exited with status {child.returncode}")
                verify_seals(artifacts)
                try:
                    snapshot = fetch_snapshot(base_url)
                except (OSError, ValueError):
                    if manifest["state"] == "running" or time.monotonic() > deadline:
                        raise ValueError("runtime identity endpoint became unavailable")
                    time.sleep(1)
                    continue
                validate_snapshot(snapshot, child.pid)
                if manifest["state"] != "running":
                    manifest = build_manifest(snapshot, artifacts, environment, command)
                    write_manifest(manifest_path, manifest)
                    print(f"Verified runtime ready: {base_url}; manifest: {manifest_path}", flush=True)
                elif canonical_hash(snapshot) != manifest["runtime_snapshot_sha256"]:
                    raise ValueError("active runtime changed during measurement")
                time.sleep(1)
    except Exception as error:
        manifest["state"] = "invalid"
        manifest["error"] = str(error)
        write_manifest(manifest_path, manifest)
        raise
    finally:
        if child is not None:
            stop_owned_child(child)
        if manifest.get("state") != "invalid":
            manifest["state"] = "stopped"
        write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
