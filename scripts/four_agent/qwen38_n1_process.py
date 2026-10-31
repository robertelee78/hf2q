"""Bounded owned-tool process supervision. No execution on import."""
import errno
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import time

from qwen38_n1_verdict import Invalid, require

REVISION = "n1-sandbox-profiles-v8"
RSS_LIMIT = 512 * 1024 * 1024
TARGET_LIMIT = 128 * 1024 * 1024
TOOL_LIMIT = 64 * 1024
WRITABLE_LIMIT = TARGET_LIMIT + 1024 * 1024
FILE_COUNT_LIMIT = 4096
PROOF_CASES = {"compile_allowed", "compile_source_write_denied", "compile_outside_write_denied",
               "compile_network_denied", "compile_outside_read_denied", "runtime_allowed", "runtime_source_write_denied",
               "runtime_test_write_denied", "runtime_outside_write_denied",
               "runtime_network_denied", "runtime_outside_read_denied", "runtime_fork_denied"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_proof(path, expected_hash):
    require(digest(path) == expected_hash, "sandbox proof hash mismatch")
    from qwen38_n1_verdict import loads
    proof = loads(Path(path).read_bytes())
    require(proof.get("schema") == "hf2q.n1-sandbox-proof.v2" and
            proof.get("profile_revision") == REVISION and
            proof.get("process_source_sha256") == digest(__file__), "sandbox proof identity")
    require(set(proof.get("cases", {})) == PROOF_CASES and
            all(v is True for v in proof["cases"].values()), "sandbox confinement unproven")
    require(proof.get("cleanup") is True and proof.get("model_loaded") is False,
            "sandbox proof cleanup/context")
    return proof


def developer_root(sdk):
    sdk = Path(sdk).resolve()
    return next((p for p in sdk.parents if p.name == "Developer" and
                 p.parent.name == "Contents"), sdk.parent.parent)


def compiler_environment(cargo, sdk):
    developer = developer_root(sdk)
    clang = (developer / "Toolchains/XcodeDefault.xctoolchain/usr/bin/clang"
             if developer.parent.name == "Contents" else developer / "usr/bin/clang")
    require(clang.is_file(), "selected trusted clang missing")
    return {"RUSTC": str(Path(cargo).resolve().with_name("rustc")),
            "SDKROOT": str(Path(sdk).resolve()),
            "CARGO_TARGET_AARCH64_APPLE_DARWIN_LINKER": str(clang)}


def profile(kind, fixture, target, temporary, log, executable, toolchain, sdk, cwd):
    """Read allowlist, exercised by the separate model-free confinement proof."""
    require(kind in {"compile", "runtime"}, "sandbox kind")
    paths = [str(Path(p).resolve()) for p in
             (fixture, target, temporary, log, executable, toolchain, sdk, cwd)]
    require(all('"' not in p and "\\" not in p and "\n" not in p for p in paths), "sandbox path")
    fixture, target, temporary, log, executable, toolchain, sdk, cwd = paths
    require(toolchain != "/" and sdk != "/", "unbounded sandbox roots")
    developer = Path(sdk).parent.parent
    roots = [str(developer / "usr/bin"), str(developer / "usr/lib"), "/System/Library", "/usr/lib", "/usr/share", "/usr/bin", "/bin",
             fixture, target, temporary, log, toolchain, sdk, cwd]
    text = ['(version 1)', '(deny default)', '(allow sysctl-read)',
            '(allow mach-lookup)', '(allow ipc-posix*)', '(allow signal (target self))',
            '(deny network*)', '(allow file-read-metadata)',
            '(allow file-read* file-test-existence (literal "/"))',
            '(allow file-read* ' + ' '.join(f'(subpath "{p}")' for p in roots) + ')',
            '(allow file-read* (literal "/dev/null") (literal "/dev/urandom") '
            '(literal "/dev/random") (literal "/private/etc/localtime"))']
    if kind == "compile":
        xcode = developer_root(sdk)
        compiler_roots = [xcode.parent if xcode.parent.name == "Contents" else xcode]  # Trusted compiler installation only.
        text += ['(allow file-read* ' + ' '.join(f'(subpath "{p}")' for p in compiler_roots) + ')']
        text += ['(allow file-read* (literal "/private/etc/ssl/openssl.cnf"))', '(allow process*)', f'(allow file-write* (subpath "{target}") '
                 f'(subpath "{temporary}") (subpath "{log}"))']
    else:
        text += [f'(allow process-exec (literal "{executable}"))', '(deny process-fork)']
    return "\n".join(text) + "\n"


def directory_bytes(path, limit=TARGET_LIMIT):
    total = 0
    entries = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            entries += 1
            require(entries <= FILE_COUNT_LIMIT, "writable tree file-count quota")
            p = Path(root) / name
            try:
                st = p.lstat()
            except FileNotFoundError:
                continue  # removed mid-walk by the running tool (e.g. rustc temp files); it holds no bytes
            require(not stat.S_ISLNK(st.st_mode), "symlink in tool target")
            if stat.S_ISREG(st.st_mode):
                total += st.st_size
                require(total <= limit, "writable tree byte quota")
    return total


def process_snapshot():
    # No model/device queries; process RSS is an observed limit, not a hard kernel reservation.
    result = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,lstart="],
                            capture_output=True, timeout=2, check=True)
    require(len(result.stdout) < 4 * 1024 * 1024, "process inventory envelope")
    rows = {}
    for line in result.stdout.decode("ascii", "strict").splitlines():
        fields = line.split()
        require(len(fields) == 9 and all(x.isdigit() for x in fields[:4]), "process metrics unknown")
        pid, ppid, pgid, rss = map(int, fields[:4])
        rows[pid] = (ppid, pgid, rss * 1024, " ".join(fields[4:]))
    return rows


def tree(rows, leader, known):
    require(all(p not in rows or rows[p][3] == stamp for p, stamp in known.items()),
            "tool PID identity reused")
    owned = {leader} | (set(rows) & set(known))
    changed = True
    while changed:
        new = {pid for pid, row in rows.items() if row[0] in owned or row[1] == leader}
        changed = not new <= owned
        owned |= new
    require(all(rows[p][1] == leader for p in owned if p in rows), "tool descendant escaped group")
    return owned


def stop_group(process, known):
    """Metrics never gate killing the owned PG or bounded reap; uncertainty stays failure."""
    result = {"actions": [], "errors": [], "warnings": [], "reaped": False, "verified": False}

    def observe():
        rows = None
        try:
            rows = process_snapshot()
            owned = tree(rows, process.pid, known)
            return rows, {p for p in owned if p in rows}
        except BaseException as exc:
            result["errors"].append(f"observe: {type(exc).__name__}: {exc}")
            return rows, None

    for sig, seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 5)):
        rows, live = observe()
        if live == set():
            result["verified"] = not result["errors"]
            break
        # Popen(start_new_session=True) created this PG. Failed metrics MUST NOT skip it.
        try:
            os.killpg(process.pid, sig)
            result["actions"].append({"pgid": process.pid, "signal": sig.name})
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            if exc.errno == errno.EPERM:
                # A completed child can disappear between inventory and signaling.
                # This diagnostic is resolved ONLY by successful reap + final empty inventory.
                result["warnings"].append({"operation": "killpg", "errno": exc.errno,
                                           "pgid": process.pid, "signal": sig.name,
                                           "error": str(exc), "resolved": False})
            else:
                result["errors"].append(f"killpg: {exc}")
        except BaseException as exc:
            result["errors"].append(f"killpg: {exc}")
        # Escaped PIDs may be signaled only with the earlier observed identity intact.
        if rows is not None:
            for pid, stamp in known.items():
                if pid in rows and rows[pid][1] != process.pid:
                    if rows[pid][3] != stamp:
                        result["errors"].append(f"escaped PID identity unknown: {pid}")
                        continue
                    try:
                        os.kill(pid, sig)
                        result["actions"].append({"pid": pid, "signal": sig.name})
                    except ProcessLookupError:
                        pass
                    except BaseException as exc:
                        result["errors"].append(f"escaped kill: {exc}")
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            try:
                process.poll()
            except BaseException as exc:
                result["errors"].append(f"poll: {exc}")
            _, live = observe()
            if live == set():
                result["verified"] = not result["errors"]
                break
            time.sleep(0.02)
        if live == set():
            break
    try:
        process.wait(timeout=1)
        result["reaped"] = True
    except BaseException as exc:
        result["errors"].append(f"bounded reap: {exc}")
    _, live = observe()
    result["final_observation"] = {"available": live is not None,
                                   "live_owned_pids": sorted(live) if live is not None else None}
    result["verified"] = live == set() and result["reaped"] and not result["errors"]
    for warning in result["warnings"]:
        warning["resolved"] = result["verified"]
    return result


def run(argv, cwd, env, sandbox, sbpl, output, target, byte_limit=TOOL_LIMIT, deadline=45, writable_root=None):
    """One fresh group,45s overall deadline,20ms RSS observations, always bounded cleanup."""
    require(0 < byte_limit <= TOOL_LIMIT, "tool output budget")
    require(0 < deadline <= 45, "tool deadline budget")
    output = Path(output)
    output.mkdir(mode=0o700)
    sb = output / "profile.sb"
    sb.write_text(sbpl)
    command = [str(sandbox), "-f", str(sb), *map(str, argv)]
    receipt = {"argv": command, "cwd": str(cwd), "profile_sha256": digest(sb),
               "cleanup": False, "rss_limit_bytes": RSS_LIMIT, "target_limit_bytes": TARGET_LIMIT,
               "deadline_seconds": deadline, "poll_requested_ms": 20, "observations": []}
    process = None
    known = {}
    raw = {"stdout": bytearray(), "stderr": bytearray()}
    error = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        receipt["pid"] = process.pid
        sel = selectors.DefaultSelector()
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            sel.register(stream, selectors.EVENT_READ, name)
        previous = time.monotonic()
        while sel.get_map() or process.poll() is None:
            now = time.monotonic()
            require(now - started <= deadline, "tool deadline")
            require(now - previous <= 0.25, "tool memory observation gap")
            rows = process_snapshot()
            owned = tree(rows, process.pid, known)
            known.update({p: rows[p][3] for p in owned if p in rows})
            rss = sum(rows[p][2] for p in owned if p in rows)
            target_bytes = directory_bytes(target)
            writable_bytes = directory_bytes(writable_root or target, WRITABLE_LIMIT)
            receipt["observations"].append({"elapsed": now - started, "rss_bytes": rss,
                                              "target_bytes": target_bytes, "writable_bytes": writable_bytes, "pids": sorted(owned)})
            require(rss <= RSS_LIMIT, "tool RSS512MiB exceeded; observed overshoot retained")
            previous = now
            for key, _ in sel.select(0.02):
                chunk = os.read(key.fileobj.fileno(), 8192)
                if not chunk:
                    sel.unregister(key.fileobj)
                else:
                    raw[key.data].extend(chunk)
                    require(sum(map(len, raw.values())) <= byte_limit,
                            "full tool output exceeds64KiB; no clipped continuation")
        receipt["exit_code"] = process.wait(timeout=1)
        # Group membership can outlive leader; cleanup also checks/reaps descendants.
    except BaseException as exc:
        error = exc
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None:
            try:
                receipt["cleanup_receipt"] = stop_group(process, known)
                receipt["cleanup"] = receipt["cleanup_receipt"]["verified"]
                if not receipt["cleanup"]:
                    error = error or Invalid("tool cleanup unverifiable; group signaled/reap attempted")
            except BaseException as exc:
                receipt["cleanup_error"] = str(exc)
                error = error or exc
        for name, data in raw.items():
            (output / f"{name}.bin").write_bytes(data)
        receipt["elapsed_seconds"] = time.monotonic() - started
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if error:
        raise Invalid(receipt.get("error", receipt.get("cleanup_error", str(error))))
    require(receipt["cleanup"], "tool cleanup incomplete")
    return receipt, bytes(raw["stdout"]), bytes(raw["stderr"])
