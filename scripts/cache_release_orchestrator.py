#!/usr/bin/env python3
"""Closed, single-attempt cache release driver and portable evidence verifier.

Only the canonical shell produces workload receipts and the release manifest.
This module owns admission, frozen inputs, hostq tickets, deadlines and evidence.
The downloaded-evidence verifier never opens local binaries, models or sources.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
from typing import Any

SCHEMA = 3
STATE = ".cache-release"
GLOBAL_LOCK = Path("/var/tmp/hf2q-cache-release-driver.lock")
WALL_SECONDS = 540
TERM_SECONDS = 520
PHASES = (
    "deepseek-cooperative", "deepseek-decode", "deepseek-wave1", "deepseek-wave2",
    "deepseek-process-a", "gemma-process-a", "gemma-process-b", "gemma-parity-n4",
    "gemma-parity-n8", "gemma-parity-n8-seed-budget", "gemma-parity-n8-tiny-hybrid",
    "gemma-parity-n8-tiny-full-tq", "gemma-parity-boundary-tail",
    "gemma-parity-long-resume", "qwen-process-a", "qwen-process-b",
    "qwen38-long-decode", "qwen38-structured",
)
BUILDS = ("build-probe", "build-debug", "build-release", "build-lcp", "build-fixtures")
TEST_BINARIES = {"debug-bin": "build/hf2q-debug-tests", "release-bin": "build/hf2q-release-tests",
                 "release-integration": "build/lcp-partial-prefill-byte-identity-tests"}
COVERAGE = {
    "deepseek-cooperative": ["deepseek/cooperative-prefill"],
    "deepseek-decode": ["deepseek/decode-cohort"],
    "deepseek-wave1": ["deepseek/full-context-1", "deepseek/process-wave-1"],
    "deepseek-wave2": ["deepseek/full-context-2", "deepseek/process-wave-2"],
    "deepseek-process-a": ["deepseek/" + p for p in
        ("process-a", "interactive", "cached-suffix", "lifecycle", "r2c-structured")],
    "gemma-process-a": ["gemma/" + p for p in
        ("process-a", "wave1", "wave2", "overlap", "lifecycle", "r2c-structured",
         "heap-summary.json", "transactions-four-slots.json", "transactions-four-slots.rows")]
        + [f"gemma/heap-{p}.{ext}" for p in
           ("baseline", "post-wave1", "post-wave2", "post-overlap", "post-lifecycle")
           for ext in ("txt", "json")],
    "gemma-process-b": ["gemma/" + p for p in
        ("process-b", "eight-slots", "transactions-eight-slots.json", "transactions-eight-slots.rows")],
    **{f"gemma-parity-{p}": [f"gemma/parity/{p}.log"] for p in
       ("n4", "n8", "n8-seed-budget", "n8-tiny-hybrid", "n8-tiny-full-tq", "boundary-tail", "long-resume")},
    "qwen-process-a": ["qwen/" + p for p in ("process-a", "cumulative", "lifecycle", "r2c-structured")],
    "qwen-process-b": ["qwen/process-b", "qwen/cancellation"],
    "qwen38-long-decode": ["qwen38/long-decode"],
    "qwen38-structured": ["qwen38/structured-outputs", "qwen38/r2c-structured"],
}
FAMILIES = {"deepseek": ("DEEPSEEK", "deepseek4"), "gemma": ("GEMMA", "gemma4"),
            "qwen": ("QWEN", "qwen35moe"), "qwen38": ("QWEN38", "qwen35")}
IDENTITY_ENV = ("EXPECTED_SHA", "EXPECTED_VERSION", "STANDALONE_CANDIDATE_RUN_ID",
                "CRATE_SHA256", "EXPECTED_BINARY_SHA256")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class Refusal(RuntimeError):
    """Invalid evidence or inadmissible operation; never a resumable attempt."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise Refusal(message)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def regular(path: Path) -> None:
    require(not path.is_symlink() and path.is_file(), f"missing or linked file: {path}")


def sha256(path: Path) -> str:
    regular(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def descriptor(path: Path) -> dict:
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def unique_pairs(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path) -> Any:
    regular(path)
    try:
        return json.loads(path.read_bytes(), object_pairs_hook=unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(Refusal(f"invalid number: {value}")))
    except (ValueError, UnicodeError) as error:
        raise Refusal(f"invalid JSON: {path}: {error}") from error


def exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def below(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    require(isinstance(relative, str) and relative and not p.is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            f"unsafe evidence path: {relative}")
    result = root.joinpath(*p.parts)
    current = root
    require(not root.is_symlink(), f"linked root: {root}")
    for part in p.parts:
        current = current / part
        require(not current.is_symlink(), f"linked evidence: {current}")
    return result


def inventory(root: Path, exclude_state: bool = True) -> dict:
    result = {}
    if not root.exists():
        return result
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        if Path(directory) == root and exclude_state:
            dirs[:] = [d for d in dirs if d != STATE]
        for name in dirs + files:
            path = Path(directory) / name
            require(not path.is_symlink(), f"linked artifact: {path}")
        for name in files:
            path = Path(directory) / name
            result[path.relative_to(root).as_posix()] = descriptor(path)
    return result


def copy_exclusive(source: Path, destination: Path) -> dict:
    expected = descriptor(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    os.chmod(destination, 0o444)
    require(descriptor(source) == expected and descriptor(destination) == expected,
            f"input changed while copying: {source}")
    return expected


def snapshot_artifacts(root: Path, work: str, before: dict) -> dict:
    after = inventory(root)
    require(set(before) <= set(after), f"{work}: raw artifacts were deleted")
    changed = {}
    for name, desc in after.items():
        if before.get(name) != desc:
            target = root / STATE / "raw" / work / name
            require(copy_exclusive(below(root, name), target) == desc, "snapshot mismatch")
            changed[name] = {**desc, "snapshot": target.relative_to(root).as_posix()}
    return changed


def file_snapshot(path: Path) -> dict:
    regular(path)
    st = path.stat()
    return {"device": st.st_dev, "inode": st.st_ino, "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def source_files(repo: Path) -> dict:
    result = {}
    for name in ("scripts", "src", "tests", ".cargo", "docs"):
        base = repo / name
        if not base.exists():
            require(name in (".cargo", "docs"), f"source tree missing: {name}")
            continue
        for directory, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for entry in dirs + files:
                require(not (Path(directory) / entry).is_symlink(), f"linked source: {entry}")
            for filename in sorted(files):
                if filename.endswith((".pyc", ".pyo")):
                    continue
                path = Path(directory) / filename
                result[path.relative_to(repo).as_posix()] = descriptor(path)
    for path in sorted(repo.iterdir()):
        if path.is_file() and (path.name.startswith("Cargo.") or path.name in
                               ("build.rs", "README.md", ".cargo_vcs_info.json", "rust-toolchain.toml")):
            result[path.name] = descriptor(path)
    require("Cargo.toml" in result and "Cargo.lock" in result, "Cargo source identity is missing")
    return result


def source_sha(repo: Path) -> str:
    packed = repo / ".cargo_vcs_info.json"
    if packed.exists():
        git = load(packed)["git"]
        require(git.get("dirty", False) is False, "packed source is dirty")
        return git["sha1"]
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def env_identity() -> dict:
    values = {name: os.environ[name] for name in IDENTITY_ENV}
    require(re.fullmatch(r"[0-9a-f]{40}", values["EXPECTED_SHA"]), "EXPECTED_SHA must be full Git SHA")
    require(re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", values["EXPECTED_VERSION"]), "invalid release version")
    require(re.fullmatch(r"[1-9][0-9]*", values["STANDALONE_CANDIDATE_RUN_ID"]), "invalid candidate run ID")
    for key in ("CRATE_SHA256", "EXPECTED_BINARY_SHA256"):
        require(DIGEST.fullmatch(values[key]), f"invalid {key}")
    return values


def configuration() -> dict:
    # Freeze all model/runtime, cargo and gate knobs, including their absence.
    prefixes = ("HF2Q_", "MLX_", "CACHE_RELEASE_", "CARGO_", "RUST", "DEEPSEEK_", "GEMMA_", "QWEN_", "QWEN38_")
    omitted = {"CACHE_RELEASE_INTERNAL", "CACHE_RELEASE_INTERNAL_RECEIPT", "CACHE_RELEASE_WORK",
               "CACHE_RELEASE_HOSTQ", "CACHE_RELEASE_FIT_RECORD", "HF2Q_BIN"}
    values = {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes) and k not in omitted}
    credentials = re.compile(r"(?:^|_)(?:TOKEN|SECRET|PASSWORD|CREDENTIALS?|API_KEY|ACCESS_KEY|PRIVATE_KEY)(?:_|$)")
    require(not any(credentials.search(name) for name in values),
            "credential-bearing environment is forbidden in release evidence")
    return values


def candidate_inputs(root: Path, check_signature: bool = False) -> dict:
    identity = env_identity()
    proof_path = Path(os.environ["STANDALONE_CANDIDATE_PROOF"])
    crate = Path(os.environ["STANDALONE_CANDIDATE_CRATE"])
    binary = Path(os.environ["HF2Q_BIN"])
    proof = load(proof_path)
    require(proof.get("kind") == "hf2q.standalone-apple-release-proof" and proof.get("schema_version") == 1,
            "not a standalone signed candidate proof")
    require(proof.get("source_sha") == identity["EXPECTED_SHA"] and proof.get("version") == identity["EXPECTED_VERSION"]
            and proof.get("package") == "hf2q" and proof.get("target") == "aarch64-apple-darwin", "signed candidate identity mismatch")
    require(proof["asset"]["sha256"] == identity["EXPECTED_BINARY_SHA256"] == sha256(binary)
            and proof["asset"]["size"] == binary.stat().st_size
            and proof["asset"]["name"] == "hf2q-aarch64-apple-darwin", "signed candidate binary mismatch")
    require(proof["signing"]["identifier"] == "us.hf2q.cli"
            and re.fullmatch(r"[A-Z0-9]{10}", proof["signing"]["team_id"])
            and proof["signing"]["authority"] == "Developer ID Application"
            and proof["signing"]["hardened_runtime"] is True and proof["signing"]["secure_timestamp"] is True
            and proof["notarization"]["status"] == "Accepted"
            and proof["verification"]["codesign"] == "accepted"
            and proof["verification"]["online_notarization"] == "accepted"
            and proof["verification"]["notary_ticket_cdhash_matches"] is True, "candidate signing proof rejected")
    require(sha256(crate) == identity["CRATE_SHA256"], "candidate crate digest mismatch")
    with tarfile.open(crate, "r:gz") as archive:
        matches = [m for m in archive.getmembers() if m.name.endswith("/.cargo_vcs_info.json")]
        require(len(matches) == 1 and matches[0].isfile(), "crate source identity missing or ambiguous")
        stream = archive.extractfile(matches[0])
        require(stream is not None, "crate source identity missing")
        info = json.load(stream, object_pairs_hook=unique_pairs)
        require(info["git"]["sha1"] == identity["EXPECTED_SHA"] and info["git"].get("dirty", False) is False,
                "candidate crate source mismatch")
    files = {"signed-proof.json": proof_path, "candidate.crate": crate}
    for filename, section, key in (
        ("notary-submission.json", "notarization", "submission_sha256"),
        ("notary-wait.json", "notarization", "wait_sha256"),
        ("notary-log.json", "notarization", "log_sha256"),
        ("codesign.txt", "verification", "codesign_log_sha256"),
        ("notarization-check.txt", "verification", "online_notarization_log_sha256"),
    ):
        path = proof_path.parent / filename
        require(sha256(path) == proof[section][key], f"signed candidate evidence mismatch: {filename}")
        files[filename] = path
    notary = load(files["notary-log.json"])
    require(notary.get("status") == "Accepted" and not notary.get("issues")
            and any(row.get("digestAlgorithm") == "SHA-256" and row.get("cdhash") == proof["signing"]["cdhash"]
                    for row in notary.get("ticketContents", [])), "notary ticket does not match candidate")
    if check_signature:
        require(sys.platform == "darwin", "signed release execution requires macOS")
        log = root / "build" / "candidate-signature-check.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("xb") as output:
            for argv in (["/usr/bin/codesign", "--verify", "--strict", "--all-architectures", str(binary)],
                         ["/usr/bin/codesign", "--verify", "--strict", "--all-architectures", "--check-notarization",
                          "--test-requirement", "=notarized", str(binary)],
                         ["/usr/bin/codesign", "--display", "--verbose=4", str(binary)]):
                output.write(canonical({"argv": argv}))
                output.flush()
                subprocess.run(argv, check=True, stdout=output, stderr=output, timeout=60)
            version = subprocess.check_output([str(binary), "--version"], text=True, timeout=15).strip()
            require(version == f"hf2q {identity['EXPECTED_VERSION']}", "signed binary version mismatch")
            output.write(canonical({"binary_version": version}))
        text = log.read_text()
        require(f"CDHash={proof['signing']['cdhash']}" in text
                and f"TeamIdentifier={proof['signing']['team_id']}" in text
                and "Identifier=us.hf2q.cli" in text and "Authority=Developer ID Application:" in text,
                "live signature identity mismatch")
    return {"identity": identity, "files": {k: {"path": str(v.absolute()), **descriptor(v)} for k, v in files.items()},
            "proof": proof}


def verify_crate_sources(crate: Path, sources: dict, identity: dict) -> None:
    prefix = f"hf2q-{identity['EXPECTED_VERSION']}/"
    with tarfile.open(crate, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            require(member.name.startswith(prefix) and not member.issym() and not member.islnk(),
                    "candidate crate has unexpected root or linked source")
            name = member.name[len(prefix):]
            require(name not in members, "duplicate candidate crate source member")
            members[name] = member
        require(".cargo_vcs_info.json" in members, "candidate crate lacks source commit identity")
        stream = archive.extractfile(members[".cargo_vcs_info.json"])
        require(stream is not None, "candidate crate source identity is not a file")
        vcs = json.load(stream, object_pairs_hook=unique_pairs)
        require(vcs["git"]["sha1"] == identity["EXPECTED_SHA"] and vcs["git"].get("dirty", False) is False,
                "candidate crate source SHA mismatch")
        for name, desc in sources.items():
            # Cargo normalizes Cargo.toml in a checkout; its original is shipped
            # alongside it. Excluded repository-only tests/docs remain commit-bound.
            member = members.get(name)
            if name == "Cargo.toml" and "Cargo.toml.orig" in members:
                original = archive.extractfile(members["Cargo.toml.orig"])
                if original is not None and digest_bytes(original.read()) == desc["sha256"]:
                    continue
            if member is None:
                require(not name.startswith(("src/", "scripts/fixtures/"))
                        and name not in ("scripts/cache_release_orchestrator.py", "scripts/run_agentic_cache_release_gate.sh"),
                        f"required gate source absent from candidate crate: {name}")
                continue
            stream = archive.extractfile(member)
            require(stream is not None and digest_bytes(stream.read()) == desc["sha256"]
                    and member.size == desc["bytes"], f"candidate crate/current source mismatch: {name}")


def model_inputs(root: Path) -> dict:
    result = {}
    for family, (prefix, architecture) in FAMILIES.items():
        receipt_path = root / family / "model-verification.json"
        receipt = load(receipt_path)
        path = Path(os.environ[f"{prefix}_MODEL"])
        digest = os.environ[f"{prefix}_MODEL_SHA256"]
        require(DIGEST.fullmatch(digest) and receipt.get("schema_version") == 1
                and receipt.get("content_hash_verified") is True and receipt.get("path") == str(path)
                and receipt.get("sha256") == digest, f"{family}: model verification mismatch")
        snap = file_snapshot(path)
        st = path.stat()
        expected = f"{st.st_dev}:{st.st_ino}:{st.st_size}:{int(st.st_mtime)}:{int(st.st_ctime)}"
        require(receipt.get("file_snapshot") == expected, f"{family}: model stat changed since hashing")
        info = root / family / "model-info.txt"
        rows = [line for line in info.read_text().splitlines() if line.startswith("Architecture: ")]
        require(rows == [f"Architecture: {architecture}"], f"{family}: model architecture mismatch")
        result[family] = {"path": str(path), "sha256": digest, "bytes": snap["bytes"],
                          "architecture": architecture, "file_snapshot": snap,
                          "verification": descriptor(receipt_path), "info": descriptor(info),
                          "hash_record": descriptor(root / family / "model.sha256")}
    return result


def build_inventory(root: Path) -> dict:
    value = load(root / "build" / "inventory.json")
    require(value.get("schema_version") == 1, "invalid prebuilt inventory schema")
    rows = value.get("test_binaries", [])
    require(len(rows) == 3 and {r["role"] for r in rows} == {"debug-bin", "release-bin", "release-integration"},
            "prebuilt test inventory is incomplete or duplicated")
    for row in rows:
        require(Path(row["path"]) == root / TEST_BINARIES[row["role"]]
                and Path(row["build_log"]) == root / "build" / f"{row['role']}.jsonl",
                "prebuilt executable/log is not in its canonical sealed role")
        for path_key, hash_key in (("path", "sha256"), ("build_log", "build_log_sha256")):
            path = Path(row[path_key])
            require(path.is_absolute() and path.is_relative_to(root) and sha256(path) == row[hash_key],
                    f"prebuilt test inventory mismatch: {path}")
    thermal = value["thermal_probe"]
    require(Path(thermal["path"]) == root / "build/macos-thermal-probe"
            and Path(thermal["source"]) == canonical_script(None).parent / "macos_thermal_probe.swift"
            and thermal["compiler"] == "/usr/bin/swiftc", "noncanonical thermal probe/toolchain")
    require(sha256(Path(thermal["path"])) == thermal["sha256"], "sealed thermal probe mismatch")
    require(sha256(Path(thermal["source"])) == thermal["source_sha256"], "thermal probe source mismatch")
    require(sha256(Path(thermal["compiler"])) == thermal["compiler_sha256"] and thermal["compiler_version"],
            "thermal compiler identity mismatch")
    require(value.get("fixtures"), "fixture inventory is empty")
    for row in value["fixtures"]:
        path = Path(row["path"])
        require(path.is_relative_to(root / "fixtures") and sha256(path) == row["sha256"], "fixture inventory mismatch")
    binary = value["release_binary"]
    require(Path(binary["path"]).is_relative_to(root) and sha256(Path(binary["path"])) == binary["sha256"]
            == os.environ["EXPECTED_BINARY_SHA256"], "sealed release binary mismatch")
    return value


def fit_records() -> dict:
    name = os.environ.get("CACHE_RELEASE_FIT_RECORD")
    if not name:
        return {phase: {"status": "unknown", "note": "No measured fit record; bounded execution is not proof of fit."}
                for phase in PHASES}
    record = load(Path(name))
    require(record.get("schema_version") == 1 and isinstance(record.get("phases"), dict), "invalid measured fit record")
    require(set(record["phases"]) <= set(PHASES), "fit record contains an unknown phase")
    result = {}
    for phase in PHASES:
        rows = record["phases"].get(phase, [])
        require(isinstance(rows, list), "fit measurements must be a list of indivisible units")
        for row in rows:
            seconds = row["wall_seconds"]
            require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
                    and row.get("unit") and row.get("evidence"), "invalid measured duration")
            require(seconds <= WALL_SECONDS, f"{phase}: known indivisible unit exceeds 540 seconds: {row['unit']}")
            require(sha256(Path(row["evidence"])) == row["evidence_sha256"], "fit evidence digest mismatch")
        result[phase] = {"status": "measured" if rows else "unknown", "units": rows}
    return result


_BSD_PROCESS_INFO = None


def kernel_process(pid: int) -> dict | None:
    """Use kernel start identity, not second-resolution ps dates, for signals."""
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return None
        return {"ppid": int(fields[1]), "pgid": int(fields[2]), "birth": "linux-ticks:" + fields[19]}
    if sys.platform == "darwin":
        global _BSD_PROCESS_INFO
        if _BSD_PROCESS_INFO is None:
            import ctypes

            class BSDInfo(ctypes.Structure):
                _fields_ = [("ids", ctypes.c_uint32 * 12), ("comm", ctypes.c_char * 16),
                            ("name", ctypes.c_char * 32), ("group", ctypes.c_uint32 * 6),
                            ("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]

            proc_info = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pidinfo
            proc_info.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
            proc_info.restype = ctypes.c_int
            _BSD_PROCESS_INFO = ctypes, BSDInfo, proc_info
        ctypes, info_type, proc_info = _BSD_PROCESS_INFO
        info = info_type()
        if proc_info(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
            return None
        return {"ppid": int(info.ids[4]), "pgid": int(info.group[1]),
                "birth": f"darwin-timeval:{info.start_sec}:{info.start_usec}"}
    raise Refusal("owned process supervision requires macOS or Linux kernel identity")


def process_table() -> dict:
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid=,lstart=,comm="], text=True, timeout=2)
    result = {}
    for line in output.splitlines():
        parts = line.split(None, 8)
        if len(parts) != 9:
            continue
        pid = int(parts[0])
        identity = kernel_process(pid)
        if identity is not None:
            result[pid] = {"pid": pid, **identity, "command": parts[8]}
    return result


def cleanup_table() -> dict:
    # An unavailable inspection tool cannot justify dropping the hostq lease.
    while True:
        try:
            return process_table()
        except (Refusal, OSError, ValueError, subprocess.SubprocessError):
            time.sleep(0.05)


def signal_owned(identity: dict, sig: int) -> bool:
    while True:
        try:
            current = kernel_process(identity["pid"])
            if current is None or current["birth"] != identity["birth"]:
                return False
            os.kill(identity["pid"], sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Fail closed, retaining the lease until owned cleanup is possible.
            time.sleep(0.05)


def process_identity(pid: int, table: dict | None = None) -> dict:
    row = (table if table is not None else process_table()).get(pid)
    require(row is not None, f"process disappeared: {pid}")
    return {"pid": pid, "birth": row["birth"]}


def same_process(identity: dict, table: dict) -> bool:
    row = table.get(identity["pid"])
    return row is not None and row["birth"] == identity["birth"]


def listeners(pids: set[int]) -> list:
    if not pids:
        return []
    command = ["lsof", "-nP", "-a", "-p", ",".join(map(str, sorted(pids))), "-iTCP", "-sTCP:LISTEN", "-Fpcn"]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
    require(result.returncode in (0, 1) and not result.stderr.strip(), "cannot inspect owned TCP listeners")
    return result.stdout.splitlines()


def conflicts(work: str, table: dict, owned_runtime: set[int] | None = None) -> list:
    rows = []
    for row in table.values():
        name = Path(row["command"]).name
        compiler = name in {"cargo", "rustc", "swiftc", "swift-frontend", "clang", "clang++", "cc1", "metal", "metallib"}
        runtime = name in {"llama-server", "llama-cli", "llama-bench", "mlx_lm.server",
                           "lcp-partial-prefill-byte-identity-tests"} or name.startswith("hf2q-")
        if name.startswith("hf2q") or name == "lcp-partial-prefill-byte-identity-tests":
            cmd = subprocess.run(["ps", "-ww", "-p", str(row["pid"]), "-o", "command="],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2).stdout
            runtime = runtime or bool(re.search(
                r"(?:^|\s)(?:serve|slot_aware_\S+|gemma_n8_\S+|gemma_fresh_\S+|gemma_hybrid_long_resume_byte_identity|"
                r"official_artifact_\S+|deepseek4_cooperative_\S+|deepseek4_decode_\S+)(?:\s|$)", cmd))
        if (runtime and row["pid"] not in (owned_runtime or set())) or (work in PHASES and compiler):
            rows.append(row)
    return rows


@contextlib.contextmanager
def driver_lock():
    fd = os.open(GLOBAL_LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o666)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), "driver lock is not regular")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Refusal("another cache release driver is active on this machine (including another OUT_ROOT)") from error
        yield
    finally:
        os.close(fd)


def attempt_path(root: Path, work: str, kind: str) -> Path:
    require(work in (*BUILDS, "finalize", *PHASES), "unknown work ID")
    return root / STATE / "attempts" / f"{work}.{kind}.json"


def authorize(root: Path, work: str, launching_ticket: bool = False) -> dict:
    token = os.environ.get("CACHE_RELEASE_INTERNAL", "")
    path = Path(os.environ.get("CACHE_RELEASE_INTERNAL_RECEIPT", "/nonexistent"))
    require(path == attempt_path(root, work, "start") and len(token) == 64, "internal work requires a driver capability")
    start = load(path)
    require(start["work"] == work and start["out_root"] == str(root)
            and digest_bytes(token.encode()) == start["capability_sha256"], "internal capability mismatch")
    table = process_table()
    require(same_process(start["driver"], table), "internal driver is no longer alive")
    # A copied receipt/approval bit is not a live driver capability.
    fd = os.open(GLOBAL_LOCK, os.O_RDWR | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise Refusal("internal work has no locked live driver")
    finally:
        os.close(fd)
    ancestor = start["driver"]
    if not launching_ticket:
        ticket = load(attempt_path(root, work, "ticket"))
        require(ticket["start_sha256"] == sha256(path), "ticket/start identity mismatch")
        ancestor = ticket["ticket_process"]
        require(same_process(ancestor, table), "owned ticket supervisor is no longer alive")
    pid = os.getpid()
    seen = set()
    while pid in table and pid not in seen and pid != ancestor["pid"]:
        seen.add(pid)
        pid = table[pid]["ppid"]
    require(pid == ancestor["pid"], "internal work is not a descendant of its owned ticket/driver")
    require(not attempt_path(root, work, "end").exists(), "completed work cannot be authorized again")
    if work not in BUILDS:
        verify_current(root, canonical_script(None), check_configuration=False)
        require(start["plan_sha256"] == sha256(root / STATE / "plan.json"), "authorized phase plan mismatch")
    elif work != BUILDS[0]:
        verify_builds(root, {}, BUILDS.index(work))
    return start


def coverage_check(work: str, artifacts: dict) -> None:
    if work not in PHASES:
        return
    required = COVERAGE[work] + [f"phases/{work}/{name}" for name in
                                ("power.log", "thermal.log", "power-assertions.before.txt", "power-assertions.after.txt")]
    for prefix in required:
        require(any((name == prefix or name.startswith(prefix + "/")) and desc["bytes"] > 0
                    for name, desc in artifacts.items()), f"{work}: missing child-produced output coverage: {prefix}")


def require_fresh_outputs(root: Path, work: str) -> None:
    if work in PHASES:
        for name in COVERAGE[work] + [f"phases/{work}"]:
            path = below(root, name)
            require(not path.exists(), f"{work}: pre-existing phase output forbids stale evidence/replay: {name}")



def telemetry_sample(root: Path, work: str, output) -> dict:
    probe = load(root / "build" / "inventory.json")["thermal_probe"]
    thermal = subprocess.check_output([probe["path"]], text=True, timeout=2).strip()
    battery = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=2)
    live = subprocess.check_output(["pmset", "-g", "live"], text=True, timeout=2)
    source = "ac" if "Now drawing from 'AC Power'" in battery else (
        "battery" if "Now drawing from 'Battery Power'" in battery else "unknown")
    modes = re.findall(r"^\s*powermode\s+([012])\s*$", live, re.MULTILINE)
    sample = {"at_ns": time.time_ns(), "thermal": thermal, "source": source,
              "mode_code": modes[0] if len(modes) == 1 else "unknown",
              "pmset_battery": battery, "pmset_live": live}
    output.write(canonical(sample))
    output.flush()
    require(thermal in ("nominal", "fair"), f"{work}: independent thermal guard rejected {thermal}")
    policy = load(root / "build" / "power-policy.json")
    require(sample["mode_code"] == str(policy["mode_code"])
            and (source, sample["mode_code"]) in (("ac", "0"), ("ac", "2"), ("battery", "2")),
            f"{work}: independent power guard rejected policy change")
    return sample


def verify_telemetry(root: Path, work: str, end: dict) -> None:
    if work not in PHASES:
        return
    receipt = end["telemetry"]
    path = below(root, receipt["path"])
    require(receipt["path"] == f"{STATE}/logs/{work}.telemetry.jsonl"
            and descriptor(path) == {"sha256": receipt["sha256"], "bytes": receipt["bytes"]},
            f"{work}: independent telemetry hash mismatch")
    rows = [json.loads(line, object_pairs_hook=unique_pairs) for line in path.read_text().splitlines()]
    require(len(rows) >= 2 and rows[0]["at_ns"] <= end["child_started_ns"]
            and rows[-1]["at_ns"] >= end["child_ended_ns"], f"{work}: telemetry does not span owned work")
    policy = load(root / "build" / "power-policy.json")
    for index, row in enumerate(rows):
        require(row["thermal"] in ("nominal", "fair")
                and row["mode_code"] == str(policy["mode_code"])
                and (row["source"], row["mode_code"]) in (("ac", "0"), ("ac", "2"), ("battery", "2"))
                and row["pmset_battery"] and row["pmset_live"], f"{work}: invalid power/thermal sample")
        if index:
            require(0 <= row["at_ns"] - rows[index - 1]["at_ns"] <= 5 * 10**9, f"{work}: telemetry gap")


def track_owned(table: dict, child_pid: int, owned: dict, adopt_orphans: bool = False) -> set[int]:
    changed = True
    while changed:
        changed = False
        for pid, row in table.items():
            if pid in owned:
                continue
            try:
                session_owned = os.getsid(pid) == child_pid
            except ProcessLookupError:
                continue
            if session_owned or row["pgid"] == child_pid or row["ppid"] == child_pid or (
                    row["ppid"] in owned and same_process(owned[row["ppid"]], table)) or (
                    adopt_orphans and row["ppid"] == os.getpid()):
                owned[pid] = {"pid": pid, "birth": row["birth"]}
                changed = True
    return {pid for pid, identity in owned.items() if same_process(identity, table)}


def reap_owned(owned: dict, direct_pid: int) -> None:
    for pid in owned:
        if pid != direct_pid:
            with contextlib.suppress(ChildProcessError, ProcessLookupError):
                os.waitpid(pid, os.WNOHANG)



def supervise(root: Path, work: str, script: Path, start: dict) -> int:
    """Run only canonical work; never signal hostq or a foreign process."""
    ticket = attempt_path(root, work, "ticket")
    began = time.time_ns()
    mono = time.monotonic()
    deadline = WALL_SECONDS if work in PHASES else 3600
    term_at = TERM_SECONDS if work in PHASES else 3580
    exclusive(ticket, {"schema_version": SCHEMA, "work": work, "start_sha256": sha256(attempt_path(root, work, "start")),
                       "ticket_process": process_identity(os.getpid()), "began_ns": began,
                       "deadline_ns": began + deadline * 10**9, "term_after_seconds": term_at,
                       "wall_limit_seconds": deadline, "hostq_class": "build" if work in BUILDS else "measure"})
    before = inventory(root)
    child = None
    owned = {}
    observed_listeners = []
    status = "failed"
    rc = 1
    reason = ""
    sent_term = False
    sent_kill = False
    interrupted = False
    conflict_rows = []
    clean = False
    remaining = []
    remaining_listeners = []
    old_handlers = {}
    telemetry = None
    child_started_ns = None
    child_ended_ns = None
    adopted = False
    if sys.platform.startswith("linux"):
        import ctypes
        adopted = ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[sig] = signal.signal(sig, interrupt)
    try:
        table = process_table()
        conflict_rows = conflicts(work, table) if work != "finalize" else []
        require(not conflict_rows, "runtime/compiler admission conflict; no foreign process was signaled")
        require_fresh_outputs(root, work)
        if work == BUILDS[0]:
            # Signature checks and preflight remain inside the single build ticket.
            candidate_inputs(root, check_signature=True)
        env = dict(os.environ)
        env["CACHE_RELEASE_WORK"] = work
        log_path = root / STATE / "logs" / f"{work}.child.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if work in PHASES:
            telemetry = (root / STATE / "logs" / f"{work}.telemetry.jsonl").open("xb")
            telemetry_sample(root, work, telemetry)
        next_sample = time.monotonic() + 2
        with log_path.open("xb") as log:
            child_started_ns = time.time_ns()
            child = subprocess.Popen(["bash", str(script), "_work", work], env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=script.parent.parent)
            cleanup_started = None
            while True:
                table = process_table()
                if adopted:
                    reap_owned(owned, child.pid)
                    table = process_table()
                alive = track_owned(table, child.pid, owned, adopted)
                # Collect listeners at admission, while running, and after cleanup.
                observed_listeners = sorted(set(observed_listeners + listeners(alive)))
                if telemetry is not None and time.monotonic() >= next_sample:
                    telemetry_sample(root, work, telemetry)
                    next_sample = time.monotonic() + 2
                elapsed = time.monotonic() - mono
                canceled = interrupted or (root / STATE / "cancel.json").exists() or not same_process(start["driver"], table)
                foreign = conflicts(work, table, alive) if work != "finalize" else []
                if foreign:
                    conflict_rows = foreign
                exit_code = child.poll()
                stopping = canceled or elapsed >= term_at or bool(conflict_rows) or exit_code is not None
                if stopping and cleanup_started is None:
                    cleanup_started = time.monotonic()
                    if canceled:
                        reason = "driver interrupted or disappeared"
                    elif elapsed >= term_at:
                        reason = "TERM deadline reached"
                    elif conflict_rows:
                        reason = "compiler/runtime conflict appeared during model work"
                    # Signal only individually tracked identities; no broad pgrep/killall.
                    for pid in alive:
                        if pid != os.getpid():
                            sent_term = signal_owned(owned[pid], signal.SIGTERM) or sent_term
                if cleanup_started is not None and (time.monotonic() - cleanup_started >= 5 or elapsed >= deadline - 5):
                    for pid in alive:
                        sent_kill = signal_owned(owned[pid], signal.SIGKILL) or sent_kill
                if exit_code is not None and not alive:
                    rc = exit_code
                    break
                if elapsed >= deadline - 1:
                    reason = reason or "owned cleanup exceeded deadline"
                    rc = child.poll() if child.poll() is not None else 124
                    break
                time.sleep(0.2)
            # KILL/reap only our direct child; descendants are also checked below.
            if child.poll() is None:
                child.kill()
            child.wait()
            # Do not release the hostq lease while an owned descendant remains,
            # even on a failed/deadline attempt. Only tracked identities are killed.
            while True:
                table = cleanup_table()
                live_owned = track_owned(table, child.pid, owned, adopted)
                if adopted:
                    reap_owned(owned, child.pid)
                    table = cleanup_table()
                    live_owned = {pid for pid in live_owned if same_process(owned[pid], table)}
                if not live_owned:
                    break
                for pid in live_owned:
                    sent_kill = signal_owned(owned[pid], signal.SIGKILL) or sent_kill
                time.sleep(0.05)
            child_ended_ns = time.time_ns()
            if telemetry is not None:
                telemetry_sample(root, work, telemetry)
            table = process_table()
            remaining = [ident for ident in owned.values() if same_process(ident, table)]
            remaining_listeners = listeners({ident["pid"] for ident in remaining})
            clean = not remaining and not remaining_listeners and child.returncode is not None
            elapsed = time.monotonic() - mono
            if rc == 0 and clean and not reason and not conflict_rows and elapsed < deadline:
                status = "pass"
            elif canceled:
                status = "interrupted"
            elif elapsed >= term_at:
                status = "deadline"
    except (Refusal, OSError, subprocess.SubprocessError, KeyError, ValueError) as error:
        reason = str(error)
        if child is not None:
            table = cleanup_table()
            track_owned(table, child.pid, owned, adopted)
            for identity in owned.values():
                signal_owned(identity, signal.SIGKILL)
            child.wait()
            while True:
                table = cleanup_table()
                alive = track_owned(table, child.pid, owned, adopted)
                if adopted:
                    reap_owned(owned, child.pid)
                    table = cleanup_table()
                    alive = {pid for pid in alive if same_process(owned[pid], table)}
                if not alive:
                    break
                for pid in alive:
                    signal_owned(owned[pid], signal.SIGKILL)
                time.sleep(0.05)
            child_ended_ns = time.time_ns()
        table = cleanup_table()
        remaining = [ident for ident in owned.values() if same_process(ident, table)]
        remaining_listeners = listeners({ident["pid"] for ident in remaining})
        clean = not remaining and not remaining_listeners
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if telemetry is not None:
            telemetry.close()
    try:
        artifacts = snapshot_artifacts(root, work, before)
        coverage_check(work, artifacts)
    except (Refusal, OSError) as error:
        artifacts = {}
        status, reason = "failed", str(error)
    ended = time.time_ns()
    if ended > began + deadline * 10**9:
        status, reason = "deadline", "evidence sealing exceeded ticket deadline"
    log_path = root / STATE / "logs" / f"{work}.child.log"
    log_desc = descriptor(log_path) if log_path.exists() else None
    end = {"schema_version": SCHEMA, "work": work, "status": status, "exit_code": rc,
           "reason": reason, "plan_sha256": start["plan_sha256"],
           "start_sha256": sha256(attempt_path(root, work, "start")), "ticket_sha256": sha256(ticket),
           "began_ns": began, "ended_ns": ended, "elapsed_seconds": time.monotonic() - mono,
           "deadline_ns": began + deadline * 10**9, "artifacts": artifacts, "child_log": log_desc,
           "child_started_ns": child_started_ns, "child_ended_ns": child_ended_ns,
           "telemetry": {"path": f"{STATE}/logs/{work}.telemetry.jsonl",
                         **descriptor(root / STATE / "logs" / f"{work}.telemetry.jsonl")}
                         if telemetry is not None else None,
           "cleanup": {"clean": clean, "term_sent": sent_term, "kill_sent": sent_kill,
                       "owned_processes": sorted(owned.values(), key=lambda row: row["pid"]),
                       "observed_listeners": observed_listeners, "remaining_processes": remaining,
                       "remaining_listeners": remaining_listeners, "direct_child_reaped": child is not None and child.returncode is not None},
           "conflicts": conflict_rows}
    exclusive(attempt_path(root, work, "end"), end)
    return 0 if status == "pass" else 1


def verify_record(root: Path, work: str, plan_sha: str | None, latest: dict) -> dict:
    start_path = attempt_path(root, work, "start")
    ticket_path = attempt_path(root, work, "ticket")
    end_path = attempt_path(root, work, "end")
    dispatch_path = attempt_path(root, work, "dispatch")
    dispatch_receipt = load(dispatch_path)
    require(dispatch_receipt["schema_version"] == SCHEMA and dispatch_receipt["work"] == work
            and dispatch_receipt["exit_code"] == 0 and dispatch_receipt["interrupted"] is False
            and dispatch_receipt["end_sha256"] == sha256(end_path)
            and dispatch_receipt["completed_ns"] >= load(end_path)["ended_ns"],
            f"{work}: hostq/driver dispatch failed or interrupted")
    require(descriptor(root / STATE / "logs" / f"{work}.hostq.log") == dispatch_receipt["hostq_log"],
            f"{work}: hostq dispatch log tampered")
    start, ticket, end = load(start_path), load(ticket_path), load(end_path)
    for record in (start, ticket, end):
        require(record.get("schema_version") == SCHEMA and record.get("work") == work, f"{work}: invalid record identity")
    require(start["plan_sha256"] == end["plan_sha256"] == plan_sha, f"{work}: plan mismatch")
    require(end["start_sha256"] == ticket["start_sha256"] == sha256(start_path)
            and end["ticket_sha256"] == sha256(ticket_path), f"{work}: start/ticket binding mismatch")
    require(end["status"] == "pass" and end["exit_code"] == 0 and not end["reason"] and not end["conflicts"],
            f"{work}: failed, interrupted or conflicted attempt")
    if work == BUILDS[0]:
        for name, desc in start["initial_artifacts"].items():
            require(name.startswith("standalone/"), "pre-existing non-candidate evidence before build")
            latest[name] = desc
    cleanup = end["cleanup"]
    require(cleanup["clean"] is True and cleanup["direct_child_reaped"] is True
            and cleanup["remaining_processes"] == [] and cleanup["remaining_listeners"] == [], f"{work}: unclean attempt")
    require(start["began_ns"] <= ticket["began_ns"] == end["began_ns"] <= end["ended_ns"]
            and ticket["deadline_ns"] == end["deadline_ns"] and end["ended_ns"] <= end["deadline_ns"]
            and 0 <= end["elapsed_seconds"] <= ticket["wall_limit_seconds"], f"{work}: invalid deadline/timing evidence")
    require(ticket["deadline_ns"] - ticket["began_ns"] == ticket["wall_limit_seconds"] * 10**9
            and abs((end["ended_ns"] - end["began_ns"]) / 10**9 - end["elapsed_seconds"]) < 1,
            f"{work}: inconsistent elapsed/deadline evidence")
    if work in PHASES:
        require(ticket["hostq_class"] == "measure" and ticket["wall_limit_seconds"] == WALL_SECONDS
                and ticket["term_after_seconds"] == TERM_SECONDS
                and end["elapsed_seconds"] < WALL_SECONDS and not cleanup["kill_sent"], f"{work}: deadline or forced cleanup")
    coverage_check(work, end["artifacts"])
    verify_telemetry(root, work, end)
    for name, desc in end["artifacts"].items():
        path = below(root, desc["snapshot"])
        require(desc["snapshot"] == f"{STATE}/raw/{work}/{name}", f"{work}: raw snapshot path mismatch")
        require(descriptor(path) == {"sha256": desc["sha256"], "bytes": desc["bytes"]}, f"{work}: raw artifact tampered: {name}")
        latest[name] = {"sha256": desc["sha256"], "bytes": desc["bytes"]}
    if end["child_log"] is not None:
        require(descriptor(root / STATE / "logs" / f"{work}.child.log") == end["child_log"], f"{work}: child log tampered")
    return {"phase": work, "start_sha256": sha256(start_path), "ticket_sha256": sha256(ticket_path),
            "end_sha256": sha256(end_path), "dispatch_sha256": sha256(dispatch_path)}


def verify_builds(root: Path, latest: dict, count: int | None = None) -> list:
    bindings = []
    previous = None
    previous_time = 0
    for work in BUILDS[:count]:
        start = load(attempt_path(root, work, "start"))
        require(start["previous_end_sha256"] == previous and start["began_ns"] >= previous_time,
                f"{work}: stale, replayed or overlapping build")
        binding = verify_record(root, work, None, latest)
        previous = binding["end_sha256"]
        previous_time = load(attempt_path(root, work, "end"))["ended_ns"]
        bindings.append(binding)
    return bindings

def verify_frozen_identities(root: Path, plan: dict) -> None:
    """Cross-link inventories, rather than merely trusting a bag of digests."""
    candidate = plan["candidate"]
    identity = candidate["identity"]
    require(re.fullmatch(r"[0-9a-f]{40}", identity["EXPECTED_SHA"])
            and plan["source_sha"] == identity["EXPECTED_SHA"], "frozen source identity mismatch")
    for key in ("CRATE_SHA256", "EXPECTED_BINARY_SHA256"):
        require(DIGEST.fullmatch(identity[key]), f"invalid frozen {key}")
    proof = load(root / STATE / "inputs" / "candidate" / "signed-proof.json")
    require(proof == candidate["proof"] and proof["kind"] == "hf2q.standalone-apple-release-proof"
            and proof["schema_version"] == 1 and proof["source_sha"] == identity["EXPECTED_SHA"]
            and proof["version"] == identity["EXPECTED_VERSION"]
            and proof["asset"]["sha256"] == identity["EXPECTED_BINARY_SHA256"]
            and proof["notarization"]["status"] == "Accepted"
            and proof["verification"]["codesign"] == "accepted"
            and proof["verification"]["online_notarization"] == "accepted"
            and proof["verification"]["notary_ticket_cdhash_matches"] is True,
            "frozen signed proof identity mismatch")
    required_candidate_files = {"signed-proof.json", "candidate.crate", "notary-submission.json",
                                "notary-wait.json", "notary-log.json", "codesign.txt", "notarization-check.txt"}
    require(set(candidate["files"]) == required_candidate_files, "incomplete/unknown signed candidate evidence inventory")
    require(proof["package"] == "hf2q" and proof["target"] == "aarch64-apple-darwin"
            and proof["asset"]["name"] == "hf2q-aarch64-apple-darwin"
            and proof["signing"]["identifier"] == "us.hf2q.cli"
            and re.fullmatch(r"[A-Z0-9]{10}", proof["signing"]["team_id"])
            and proof["signing"]["authority"] == "Developer ID Application"
            and proof["signing"]["hardened_runtime"] is True and proof["signing"]["secure_timestamp"] is True,
            "invalid frozen signing identity")
    for filename, section, key in (
        ("notary-submission.json", "notarization", "submission_sha256"),
        ("notary-wait.json", "notarization", "wait_sha256"),
        ("notary-log.json", "notarization", "log_sha256"),
        ("codesign.txt", "verification", "codesign_log_sha256"),
        ("notarization-check.txt", "verification", "online_notarization_log_sha256"),
    ):
        require(candidate["files"][filename]["sha256"] == proof[section][key],
                f"frozen signature/notary evidence is not bound: {filename}")
    notary = load(root / STATE / "inputs" / "candidate" / "notary-log.json")
    require(notary["status"] == "Accepted" and not notary.get("issues")
            and any(row.get("digestAlgorithm") == "SHA-256" and row.get("cdhash") == proof["signing"]["cdhash"]
                    for row in notary.get("ticketContents", [])), "frozen notary ticket CDHash mismatch")
    for name, row in candidate["files"].items():
        desc = {"sha256": row["sha256"], "bytes": row["bytes"]}
        require(plan["frozen_files"].get(f"{STATE}/inputs/candidate/{name}") == desc,
                f"unsealed candidate input: {name}")
    require(candidate["files"]["candidate.crate"]["sha256"] == identity["CRATE_SHA256"], "unbound candidate crate")
    verify_crate_sources(root / STATE / "inputs" / "candidate" / "candidate.crate", plan["source_files"], identity)
    require(load(root / "build" / "inventory.json") == plan["build_inventory"], "prebuilt inventory differs from plan")
    prebuilt = plan["build_inventory"]
    rows = prebuilt["test_binaries"]
    require(len(rows) == 3 and {row["role"] for row in rows} ==
            {"debug-bin", "release-bin", "release-integration"}, "missing/duplicate prebuilt test roles")
    original_root = Path(plan["out_root"])

    def artifact(path: str, digest: str) -> Path:
        absolute = Path(path)
        require(absolute.is_absolute() and absolute.is_relative_to(original_root), "prebuilt artifact outside original output root")
        relocated = below(root, absolute.relative_to(original_root).as_posix())
        require(sha256(relocated) == digest, f"prebuilt artifact digest mismatch: {relocated}")
        return relocated

    for row in rows:
        require(Path(row["path"]) == original_root / TEST_BINARIES[row["role"]]
                and Path(row["build_log"]) == original_root / "build" / f"{row['role']}.jsonl",
                "prebuilt role path mismatch")
        artifact(row["path"], row["sha256"])
        artifact(row["build_log"], row["build_log_sha256"])
    probe = prebuilt["thermal_probe"]
    require(Path(probe["path"]) == original_root / "build/macos-thermal-probe",
            "thermal probe is not the canonical sealed build output")
    artifact(probe["path"], probe["sha256"])
    source = Path(probe["source"])
    source_root = Path(plan["source_root"])
    require(source.is_relative_to(source_root) and
            plan["source_files"][source.relative_to(source_root).as_posix()]["sha256"] == probe["source_sha256"],
            "thermal probe source is not sealed source")
    require(probe["compiler"] == "/usr/bin/swiftc" and probe["compiler_version"]
            and DIGEST.fullmatch(probe["compiler_sha256"]), "thermal compiler identity missing")
    binary = prebuilt["release_binary"]
    binary_path = artifact(binary["path"], binary["sha256"])
    require(binary["sha256"] == identity["EXPECTED_BINARY_SHA256"]
            and binary_path.stat().st_size == proof["asset"]["size"], "sealed binary does not match signed candidate")
    require(prebuilt["fixtures"], "empty fixture inventory")
    seen = set()
    for row in prebuilt["fixtures"]:
        require(row["path"] not in seen, "duplicate fixture identity")
        seen.add(row["path"])
        path = artifact(row["path"], row["sha256"])
        require(path.is_relative_to(root / "fixtures"), "fixture outside canonical fixture root")
    require(set(plan["models"]) == set(FAMILIES), "incomplete model identity set")
    for family, (_, architecture) in FAMILIES.items():
        model = plan["models"][family]
        require(DIGEST.fullmatch(model["sha256"]) and type(model["bytes"]) is int and model["bytes"] > 0
                and model["architecture"] == architecture, f"{family}: invalid model identity")
        for filename, key in (("model-info.txt", "info"), ("model-verification.json", "verification"), ("model.sha256", "hash_record")):
            require(descriptor(root / family / filename) == model[key], f"{family}: unbound model metadata")
        receipt = load(root / family / "model-verification.json")
        require(receipt["content_hash_verified"] is True and receipt["sha256"] == model["sha256"]
                and receipt["path"] == model["path"] and receipt["file_snapshot"], f"{family}: verification identity mismatch")
        require((root / family / "model.sha256").read_text().strip() == f"{model['sha256']}  {model['path']}",
                f"{family}: model hash record mismatch")
    for name, desc in plan["dependency"].items():
        require(plan["frozen_files"].get(f"{STATE}/inputs/dependency/{name}") == desc
                and descriptor(root / "dependency-provenance" / name) == desc, "dependency provenance is not frozen")
    for phase, row in plan["fit"].items():
        for index, unit in enumerate(row.get("units", [])):
            require(plan["frozen_files"][f"{STATE}/inputs/fit/{phase}/{index}"]["sha256"] == unit["evidence_sha256"],
                    "measured fit evidence is not frozen")


def verify_evidence(root: Path, manifest: dict | None = None) -> dict:
    """Strict reusable offline verifier. No live local model/source requirement."""
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink(), "evidence root missing or linked")
    require(not (root / STATE / "cancel.json").exists(), "interrupted driver cannot pass verification")
    plan_path = root / STATE / "plan.json"
    plan = load(plan_path)
    plan_sha = sha256(plan_path)
    require(plan.get("schema_version") == SCHEMA and plan.get("kind") == "hf2q.cache-release-plan", "invalid frozen plan schema")
    require(plan.get("required_phases") == list(PHASES) and plan.get("dag") == phase_dag()
            and plan.get("coverage") == COVERAGE, "selected, reordered or altered required phase set")
    require(plan.get("wall_limit_seconds") == WALL_SECONDS and plan.get("term_after_seconds") == TERM_SECONDS,
            "altered ticket deadline")
    require(set(plan["fit"]) == set(PHASES), "incomplete fit uncertainty records")
    for row in plan["fit"].values():
        require(row["status"] in ("unknown", "measured"), "invented fit claim")
        for unit in row.get("units", []):
            require(0 < unit["wall_seconds"] <= WALL_SECONDS, "known oversized indivisible unit")
    for name, desc in plan["frozen_files"].items():
        require(descriptor(below(root, name)) == desc, f"frozen input tampered: {name}")
    require(plan["frozen_files"] and plan["source_files"] and plan["models"] and plan["build_inventory"], "empty frozen inputs")
    for name, desc in plan["source_files"].items():
        require(plan["frozen_files"].get(f"{STATE}/inputs/source/{name}") == desc, "unsealed source inventory")
    for prefix in ("scripts/", "src/", "tests/"):
        require(any(name.startswith(prefix) for name in plan["source_files"]), f"source inventory missing {prefix}")
    require(all(name in plan["source_files"] for name in ("Cargo.toml", "Cargo.lock")), "Cargo inputs missing")
    verify_frozen_identities(root, plan)
    admission = load(attempt_path(root, BUILDS[0], "start"))["build_inputs"]
    require(admission["source_files"] == plan["source_files"] and admission["source_sha"] == plan["source_sha"]
            and admission["identity"] == plan["candidate"]["identity"]
            and admission["configuration"] == plan["configuration"], "frozen plan differs from prebuild admission")
    expected_argv = ["bash", str(Path(plan["source_root"]) / "scripts" / "run_agentic_cache_release_gate.sh"), "_work"]
    require(plan["source_argv"] == expected_argv, "plan has alternate workload argv")
    for work in (*BUILDS, *PHASES, *(("finalize",) if manifest is not None else ())):
        start = load(attempt_path(root, work, "start"))
        require(start["out_root"] == plan["out_root"] and start["argv"] == [*expected_argv, work],
                f"{work}: different output root or workload argv")
    latest = {}
    build_bindings = verify_builds(root, latest)
    require(plan["build_end_sha256"] == build_bindings[-1]["end_sha256"]
            and plan["build_attempts"] == build_bindings, "plan does not bind genuine build attempts")
    require(load(attempt_path(root, BUILDS[-1], "end"))["ended_ns"] <= plan["created_ns"], "plan predates build completion")
    bindings = []
    previous = plan["created_ns"]
    previous_end_sha = plan["build_end_sha256"]
    for work in PHASES:
        start = load(attempt_path(root, work, "start"))
        require(start["began_ns"] >= previous and start["previous_end_sha256"] == previous_end_sha, f"{work}: stale/replayed or overlapping attempt")
        binding = verify_record(root, work, plan_sha, latest)
        end = load(attempt_path(root, work, "end"))
        previous, previous_end_sha = end["ended_ns"], binding["end_sha256"]
        bindings.append(binding)
    allowed = {f"{work}.{kind}.json" for work in (*BUILDS, *PHASES) for kind in ("start", "ticket", "end", "dispatch")}
    actual = {p.name for p in (root / STATE / "attempts").iterdir()}
    finalize_names = {f"finalize.{kind}.json" for kind in ("start", "ticket", "end", "dispatch")}
    require(actual - finalize_names == allowed and actual <= allowed | finalize_names, "duplicate, missing or unknown ledger records")
    binding = {"plan_sha256": plan_sha, "ledger_sha256": digest_bytes(canonical([*build_bindings, *bindings])),
               "required_phases": list(PHASES), "attempts": bindings}
    if manifest is not None:
        require(manifest == load(root / "manifest.json"), "consumer manifest differs from finalized raw manifest")
        require(manifest.get("schema_version") == SCHEMA and manifest.get("kind") == "hf2q.cache-lifecycle-release-manifest"
                and manifest.get("status") == "pass" and manifest.get("phase_ledger") == binding,
                "manifest is not schema3 bound to this complete phase ledger")
        verify_manifest_identity(plan, manifest)
        final = verify_record(root, "finalize", plan_sha, latest)
        final_start = load(attempt_path(root, "finalize", "start"))
        require(final_start["previous_end_sha256"] == previous_end_sha and final_start["began_ns"] >= previous,
                "stale finalization attempt")
        require(final["phase"] == "finalize", "invalid finalization")
    for name, desc in latest.items():
        require(descriptor(below(root, name)) == desc, f"current raw evidence mismatch: {name}")
    if manifest is not None:
        require(inventory(root) == latest, "unrecorded, removed or changed raw artifacts in finalized evidence")
        expected_state = set(plan["frozen_files"]) | {f"{STATE}/plan.json"}
        for work in (*BUILDS, *PHASES, "finalize"):
            expected_state.update(f"{STATE}/attempts/{work}.{kind}.json" for kind in ("start", "ticket", "end", "dispatch"))
            expected_state.update(f"{STATE}/logs/{work}.{kind}.log" for kind in ("child", "hostq"))
            if work in PHASES:
                expected_state.add(f"{STATE}/logs/{work}.telemetry.jsonl")
            end = load(attempt_path(root, work, "end"))
            expected_state.update(row["snapshot"] for row in end["artifacts"].values())
        actual_state = {f"{STATE}/{name}" for name in inventory(root / STATE, exclude_state=False)}
        require(actual_state == expected_state, "unknown, unrecorded or missing immutable evidence files")
    return binding


def verify_manifest_identity(plan: dict, manifest: dict) -> None:
    values = plan["candidate"]["identity"]
    for key, env in (("source_sha", "EXPECTED_SHA"), ("version", "EXPECTED_VERSION"),
                     ("crate_sha256", "CRATE_SHA256"), ("binary_sha256", "EXPECTED_BINARY_SHA256"),
                     ("standalone_candidate_run_id", "STANDALONE_CANDIDATE_RUN_ID")):
        require(manifest.get(key) == values[env], f"manifest {key} differs from frozen candidate")
    for family, expected in plan["models"].items():
        actual = manifest["models"][family]
        require(all(actual.get(key) == expected[key] for key in ("path", "bytes", "sha256", "architecture"))
                and actual.get("model_info_sha256") == expected["info"]["sha256"], f"manifest {family} differs from frozen model")


def phase_dag() -> list:
    return [{"id": phase, "requires": [PHASES[index - 1]] if index else [], "required": True}
            for index, phase in enumerate(PHASES)]


def make_plan(root: Path, script: Path) -> None:
    require(not (root / STATE / "plan.json").exists(), "plan already exists; no replay")
    require(not any(attempt_path(root, phase, "start").exists() for phase in PHASES), "model work already began")
    latest = {}
    builds = verify_builds(root, latest)
    for name, desc in latest.items():
        require(descriptor(below(root, name)) == desc, f"build output changed: {name}")
    candidate = candidate_inputs(root)
    repo = script.parent.parent
    require(source_sha(repo) == candidate["identity"]["EXPECTED_SHA"], "current source differs from expected source SHA")
    sources = source_files(repo)
    verify_crate_sources(Path(os.environ["STANDALONE_CANDIDATE_CRATE"]), sources, candidate["identity"])
    admission = load(attempt_path(root, BUILDS[0], "start"))["build_inputs"]
    require(admission["source_files"] == sources and admission["source_sha"] == source_sha(repo)
            and admission["identity"] == candidate["identity"] and admission["configuration"] == configuration(),
            "source/candidate/configuration changed after build admission")
    models = model_inputs(root)
    prebuilt = build_inventory(root)
    fit = fit_records()
    frozen = {}
    for name, desc in sources.items():
        path = root / STATE / "inputs" / "source" / name
        require(copy_exclusive(repo / name, path) == desc, "source changed while freezing")
        frozen[path.relative_to(root).as_posix()] = desc
    for name, row in candidate["files"].items():
        path = root / STATE / "inputs" / "candidate" / name
        frozen[path.relative_to(root).as_posix()] = copy_exclusive(Path(row["path"]), path)
    dependency_dir = Path(os.environ["DEPENDENCY_PROVENANCE_DIR"])
    dependency = {}
    for name in ("Cargo.lock", "cargo-metadata.json", "receipt.json"):
        path = root / STATE / "inputs" / "dependency" / name
        dependency[name] = copy_exclusive(dependency_dir / name, path)
        require(dependency[name] == descriptor(root / "dependency-provenance" / name), "dependency receipt changed in build")
        frozen[path.relative_to(root).as_posix()] = dependency[name]
    for phase, row in fit.items():
        for index, unit in enumerate(row.get("units", [])):
            path = root / STATE / "inputs" / "fit" / phase / str(index)
            frozen[path.relative_to(root).as_posix()] = copy_exclusive(Path(unit["evidence"]), path)
    exclusive(root / STATE / "plan.json", {
        "schema_version": SCHEMA, "kind": "hf2q.cache-release-plan", "created_ns": time.time_ns(),
        "out_root": str(root), "source_root": str(repo), "source_sha": source_sha(repo),
        "source_files": sources, "candidate": candidate, "models": models, "dependency": dependency,
        "build_inventory": prebuilt, "configuration": configuration(),
        "source_argv": ["bash", str(script), "_work"], "hostq": {"path": os.environ["CACHE_RELEASE_HOSTQ"],
        **descriptor(Path(os.environ["CACHE_RELEASE_HOSTQ"]))}, "fit": fit,
        "required_phases": list(PHASES), "dag": phase_dag(), "coverage": COVERAGE,
        "wall_limit_seconds": WALL_SECONDS, "term_after_seconds": TERM_SECONDS,
        "build_end_sha256": builds[-1]["end_sha256"], "build_attempts": builds, "frozen_files": frozen})
    unknown = [phase for phase in PHASES if fit[phase]["status"] == "unknown"]
    if unknown:
        print("Fit uncertainty (bounded execution, not a fit claim): " + ", ".join(unknown), file=sys.stderr)


def verify_current(root: Path, script: Path, check_configuration: bool = True) -> dict:
    plan = load(root / STATE / "plan.json")
    require(plan["required_phases"] == list(PHASES) and plan["dag"] == phase_dag(), "plan phase set changed")
    require(plan["out_root"] == str(root) and plan["source_root"] == str(script.parent.parent), "plan belongs to another checkout/output root")
    require(plan["source_files"] == source_files(script.parent.parent) and source_sha(script.parent.parent) == plan["source_sha"], "source changed after plan freeze")
    require(plan["candidate"] == candidate_inputs(root) and plan["models"] == model_inputs(root), "candidate/model identity changed after freeze")
    require((not check_configuration or plan["configuration"] == configuration())
            and plan["build_inventory"] == build_inventory(root), "runtime configuration or prebuilt inventory changed")
    require(plan["hostq"] == {"path": os.environ["CACHE_RELEASE_HOSTQ"], **descriptor(Path(os.environ["CACHE_RELEASE_HOSTQ"]))}, "hostq executable changed")
    for name, desc in plan["frozen_files"].items():
        require(descriptor(below(root, name)) == desc, f"frozen input changed: {name}")
    return plan


def dispatch(root: Path, work: str, script: Path) -> None:
    require(not attempt_path(root, work, "start").exists(), f"{work}: attempt already began; retry/replay forbidden")
    plan_sha = None
    previous_end = None
    latest = {}
    if work not in BUILDS:
        verify_current(root, script)
        plan_sha = sha256(root / STATE / "plan.json")
        expected_index = len(PHASES) if work == "finalize" else PHASES.index(work)
        builds = verify_builds(root, latest)
        previous_end = builds[-1]["end_sha256"]
        for phase in PHASES[:expected_index]:
            previous_end = verify_record(root, phase, plan_sha, latest)["end_sha256"]
        require(not any(attempt_path(root, phase, "start").exists() for phase in PHASES[expected_index:]),
                "out-of-order or replayed phase attempt")
        if work == "finalize":
            verify_evidence(root)
    else:
        require(not (root / STATE / "plan.json").exists(), "build cannot follow a frozen plan")
        require(not any(attempt_path(root, phase, "start").exists() for phase in PHASES), "build cannot follow model work")
        position = BUILDS.index(work)
        if position:
            previous_end = verify_builds(root, latest, position)[-1]["end_sha256"]
        else:
            require(not any((root / family).exists() for family in FAMILIES), "build refuses pre-existing model results")
        require(not any(attempt_path(root, step, "start").exists() for step in BUILDS[position:]),
                "out-of-order or replayed build attempt")
    for name, desc in latest.items():
        require(descriptor(below(root, name)) == desc, f"prior evidence changed: {name}")
    require_fresh_outputs(root, work)
    token = secrets.token_hex(32)
    start_path = attempt_path(root, work, "start")
    exclusive(start_path, {"schema_version": SCHEMA, "work": work, "out_root": str(root),
                           "began_ns": time.time_ns(), "driver": process_identity(os.getpid()),
                           "capability_sha256": digest_bytes(token.encode()), "plan_sha256": plan_sha,
                           "previous_end_sha256": previous_end,
                           "initial_artifacts": inventory(root) if work == BUILDS[0] else {},
                           "build_inputs": {"source_files": source_files(script.parent.parent),
                                            "source_sha": source_sha(script.parent.parent),
                                            "identity": env_identity(), "configuration": configuration()}
                                            if work == BUILDS[0] else None,
                           "argv": ["bash", str(script), "_work", work]})
    env = dict(os.environ, CACHE_RELEASE_INTERNAL=token, CACHE_RELEASE_INTERNAL_RECEIPT=str(start_path))
    command = [sys.executable, str(Path(__file__).absolute()), "--script", str(script), "_ticket", work]
    if work != "finalize":
        command = [os.environ["CACHE_RELEASE_HOSTQ"], f"cache-release-{work}",
                   "build" if work in BUILDS else "measure", "--", *command]
    log_path = root / STATE / "logs" / f"{work}.hostq.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    interrupted = False

    def cancel(signum, _frame):
        nonlocal interrupted
        interrupted = True
        path = root / STATE / "cancel.json"
        if not path.exists():
            exclusive(path, {"signal": signum, "at_ns": time.time_ns(), "work": work})
        # Never signal hostq: its lock must outlive owned cleanup.

    handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    rc = 1
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            rc = process.wait()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        exclusive(attempt_path(root, work, "dispatch"), {
            "schema_version": SCHEMA, "work": work, "exit_code": rc, "interrupted": interrupted,
            "completed_ns": time.time_ns(), "hostq_log": descriptor(log_path),
            "end_sha256": sha256(attempt_path(root, work, "end"))
                if attempt_path(root, work, "end").exists() else None})
    require(not interrupted and rc == 0, f"{work}: ticket failed or driver interrupted; see {log_path}")
    verify_record(root, work, plan_sha, {})


def canonical_script(value: str | None) -> Path:
    expected = Path(__file__).resolve().parent / "run_agentic_cache_release_gate.sh"
    supplied = Path(value) if value else expected
    actual = supplied.resolve()
    require(actual == expected and not supplied.is_symlink(), "only the canonical workload script is accepted")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Execution requires OUT_ROOT (stable absolute), CACHE_RELEASE_HOSTQ (executable), "
        "EXPECTED_SHA, EXPECTED_VERSION, STANDALONE_CANDIDATE_RUN_ID, CRATE_SHA256, "
        "EXPECTED_BINARY_SHA256, STANDALONE_CANDIDATE_PROOF, STANDALONE_CANDIDATE_CRATE, "
        "DEPENDENCY_PROVENANCE_DIR, HF2Q_BIN and all four *_MODEL / *_MODEL_SHA256 pairs. "
        "run continues completed passing positions without replay; failed/interrupted starts block it. "
        "CACHE_RELEASE_FIT_RECORD optionally supplies measured indivisible durations; absent fit remains unknown. "
        "verify and consumer-verify are model-free; consumer-verify also requires finalized manifest evidence."))
    parser.add_argument("--script", help="Canonical run_agentic_cache_release_gate.sh path only (no alternate commands)")
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("run", "build", "plan", "finalize"):
        sub.add_parser(name)
    phase = sub.add_parser("phase", help="Run exactly the next immutable required phase")
    phase.add_argument("phase_id", choices=PHASES)
    for name in ("verify", "consumer-verify"):
        verify = sub.add_parser(name, help="Verify portable evidence without opening live models/sources")
        verify.add_argument("root", type=Path)
        if name == "consumer-verify":
            verify.add_argument("manifest", type=Path)
    auth = sub.add_parser("authorize", help="Internal canonical shell capability check; not a work launcher")
    auth.add_argument("root", type=Path)
    auth.add_argument("work", choices=(*BUILDS, "finalize", *PHASES))
    ticket = sub.add_parser("_ticket", help=argparse.SUPPRESS)
    ticket.add_argument("work", choices=(*BUILDS, "finalize", *PHASES))
    args = parser.parse_args(argv)
    try:
        if args.mode == "authorize":
            authorize(args.root.absolute(), args.work)
            return 0
        if args.mode in ("verify", "consumer-verify"):
            # Canonical finalize calls verify while holding the same driver lock.
            nested = bool(os.environ.get("CACHE_RELEASE_INTERNAL"))
            with contextlib.nullcontext() if nested else driver_lock():
                if nested:
                    authorize(args.root.absolute(), os.environ.get("CACHE_RELEASE_WORK", "finalize"))
                result = verify_evidence(args.root, load(args.manifest) if args.mode == "consumer-verify" else None)
            print(canonical(result).decode(), end="")
            return 0
        root_text = os.environ["OUT_ROOT"]
        require(Path(root_text).is_absolute(), "OUT_ROOT must be a stable absolute non-linked path")
        root = Path(os.path.abspath(root_text))
        require(not root.is_symlink(), "OUT_ROOT must be a stable absolute non-linked path")
        os.environ["OUT_ROOT"] = str(root)
        script = canonical_script(args.script)
        if args.mode == "_ticket":
            start = authorize(root, args.work, launching_ticket=True)
            return supervise(root, args.work, script, start)
        configured_hostq = os.environ["CACHE_RELEASE_HOSTQ"]
        hostq = Path(shutil.which(configured_hostq) or configured_hostq).resolve(strict=True)
        require(hostq.is_file() and os.access(hostq, os.X_OK), "CACHE_RELEASE_HOSTQ must name an executable")
        os.environ["CACHE_RELEASE_HOSTQ"] = str(hostq)
        env_identity()
        root.mkdir(parents=True, exist_ok=True)
        with driver_lock():
            require(not (root / STATE / "cancel.json").exists(), "interrupted driver evidence is immutable; replay forbidden")
            if args.mode in ("run", "build"):
                for work in BUILDS:
                    if args.mode == "run" and attempt_path(root, work, "start").exists():
                        verify_record(root, work, None, {})
                    else:
                        dispatch(root, work, script)
            if args.mode == "plan" or (args.mode == "run" and not (root / STATE / "plan.json").exists()):
                make_plan(root, script)
            if args.mode == "run":
                for work in PHASES:
                    if attempt_path(root, work, "start").exists():
                        verify_current(root, script)
                        verify_record(root, work, sha256(root / STATE / "plan.json"), {})
                    else:
                        dispatch(root, work, script)
                if attempt_path(root, "finalize", "start").exists():
                    verify_evidence(root, load(root / "manifest.json"))
                else:
                    dispatch(root, "finalize", script)
            elif args.mode == "phase":
                dispatch(root, args.phase_id, script)
            elif args.mode == "finalize":
                dispatch(root, "finalize", script)
        return 0
    except (Refusal, OSError, KeyError, ValueError, TypeError, subprocess.SubprocessError, tarfile.TarError) as error:
        print(f"cache release refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
