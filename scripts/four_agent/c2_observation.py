"""C2 dedicated-cohort observations; no model launch or qualification verdict."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

SCHEMA = "hf2q.adr063.c2-observation.v1"
SCOPE = "decode_ready_on_turns_with_at_least_two_decode_lanes"
TARGET = "hf2q::serve::api::engine_qwen35::progress"
EVENT = "Qwen C2 decode lane steps"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
COUNTERS = {"hf2q_qwen_c2_joint_lane_steps_total": "joint_lane_steps",
            "hf2q_qwen_c2_serial_lane_steps_total": "serial_lane_steps"}
REASONS = "hf2q_qwen_joint_not_eligible_turns_total"
CACHE_CONTROL_SCHEMA = "hf2q.adr063.c2-cache-control.v1"
STARTUP_LIMIT = 64 * 1024
ANCHOR_EVENT = re.compile(
    r"^(?:\d{4}-\d\d-\d\dT\S+\s+)?INFO\s+hf2q::serve::api::engine:\s+"
    r"Qwen35 multi-depth anchor store configured(?:\s|$)")
ANCHOR_CONFIG = re.compile(
    r"^(?:\d{4}-\d\d-\d\dT\S+\s+)?INFO\s+hf2q::serve::api::engine:\s+"
    r"Qwen35 multi-depth anchor store configured anchor_depth=([14]) "
    r"anchor_bytes_per_slot_max=Some\(([1-9][0-9]*)\)$")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def activation(packet, server_log, family, engine_kind, phase_plan):
    """The collector is a separately source-bound coordinator capability."""
    granted = (packet.get("c2_server_log"), packet.get("c2_observation_sources"))
    if server_log is None:
        require(granted == (None, None), "C2 observation approval requires explicit CLI activation")
        return
    require(family == "qwen" and engine_kind == "hf2q" and phase_plan is None,
            "C2 observations require one dedicated four-slot Qwen hf2q cohort")
    path = Path(server_log)
    require(path.is_absolute() and granted == (server_log, {"c2_observation.py": sha(__file__)}),
            "C2 log/source approval binding mismatch")


def regular_evidence(path):
    """Refuse special files before reading; FIFO opens must not wait for a writer."""
    stream = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK),
                       "rb", buffering=0)
    try:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "regular C2 evidence required")
        return stream
    except BaseException:
        stream.close()
        raise


def anchor_declaration(raw):
    line = ANSI.sub("", raw.decode("utf-8")).strip()
    if ANCHOR_EVENT.match(line) is None:
        return None
    match = ANCHOR_CONFIG.fullmatch(line)
    require(match is not None, "malformed C2 anchor configuration")
    depth, limit = map(int, match.groups())
    require(limit <= 0xffffffffffffffff, "C2 anchor limit exceeds u64")
    return depth, limit


def anchor_config(raw):
    require(0 < len(raw) <= STARTUP_LIMIT, "bounded C2 startup evidence")
    declarations = [row for line in raw.splitlines() if (row := anchor_declaration(line)) is not None]
    require(len(declarations) == 1, "one explicit C2 anchor configuration required")
    return declarations[0]


def scan_anchor_config(stream):
    """Scan the whole original log with fresh descriptor reads and bounded lines."""
    info = os.fstat(stream.fileno())
    require(stat.S_ISREG(info.st_mode), "regular C2 evidence required")
    offset, pending, declaration = 0, b"", None
    while offset < info.st_size:
        raw = os.pread(stream.fileno(), min(STARTUP_LIMIT, info.st_size - offset), offset)
        require(raw, "C2 configuration log truncated during scan")
        offset += len(raw)
        lines = (pending + raw).split(b"\n")
        pending = lines.pop()
        require(all(len(line) <= STARTUP_LIMIT for line in lines) and len(pending) <= STARTUP_LIMIT,
                "C2 configuration log line exceeds bounded scan")
        for line in lines:
            row = anchor_declaration(line)
            if row is not None:
                require(declaration is None, "one explicit C2 anchor configuration required")
                declaration = row
    if pending:
        row = anchor_declaration(pending)
        if row is not None:
            require(declaration is None, "one explicit C2 anchor configuration required")
            declaration = row
    require(declaration is not None, "one explicit C2 anchor configuration required")
    return declaration


def checked_cache_control(control, server_pid, run_id):
    require(type(control) is dict and set(control) == {
        "schema", "server_pid", "run_id", "server_log", "log_dev", "log_ino",
        "anchor_depth", "anchor_bytes_per_slot_max", "startup_log"},
        "closed observed C2 cache control")
    require(control["schema"] == CACHE_CONTROL_SCHEMA and
            type(control["server_pid"]) is int and control["server_pid"] == server_pid > 1 and
            type(control["run_id"]) is str and control["run_id"] == run_id and
            re.fullmatch(r"[a-zA-Z0-9_-]{16,64}", run_id) is not None and
            type(control["server_log"]) is str and Path(control["server_log"]).is_absolute() and
            all(type(control[key]) is int and control[key] >= 0 for key in ("log_dev", "log_ino")),
            "C2 cache control server/log identity")
    binding = control["startup_log"]
    require(type(binding) is dict and set(binding) == {"path", "sha256"} and
            type(binding["path"]) is str and Path(binding["path"]).is_absolute(),
            "bound C2 startup evidence")
    with regular_evidence(binding["path"]) as stream:
        raw = stream.read(STARTUP_LIMIT + 1)
    require(hashlib.sha256(raw).hexdigest() == binding["sha256"], "C2 startup evidence changed")
    depth, limit = anchor_config(raw)
    require(type(control["anchor_depth"]) is int and control["anchor_depth"] == depth and
            type(control["anchor_bytes_per_slot_max"]) is int and
            control["anchor_bytes_per_slot_max"] == limit, "C2 cache control contradicts observed configuration")
    return control


def cache_control(server_log, output, server_pid, run_id):
    """Called only after activation authenticates this cohort's owned stderr."""
    path = Path(server_log)
    require(path.is_absolute(), "absolute C2 server log")
    with regular_evidence(path) as stream:
        info = os.fstat(stream.fileno())
        raw = stream.read(STARTUP_LIMIT)
        depth, limit = anchor_config(raw)
        require(scan_anchor_config(stream) == (depth, limit), "C2 startup configuration changed")
    startup = Path(output) / "c2-startup.log"
    with startup.open("xb") as stream:
        stream.write(raw)
    return checked_cache_control({
        "schema": CACHE_CONTROL_SCHEMA, "server_pid": server_pid, "run_id": run_id, "server_log": str(path),
        "log_dev": info.st_dev, "log_ino": info.st_ino, "anchor_depth": depth,
        "anchor_bytes_per_slot_max": limit,
        "startup_log": {"path": str(startup), "sha256": hashlib.sha256(raw).hexdigest()}},
        server_pid, run_id)


def check_startup_identity(control, stream):
    """Join the retained startup bytes to the same append-only owned log."""
    info = os.fstat(stream.fileno())
    require(stat.S_ISREG(info.st_mode) and
            (info.st_dev, info.st_ino) == (control["log_dev"], control["log_ino"]),
            "C2 startup/observation log identity changed")
    with regular_evidence(control["startup_log"]["path"]) as retained:
        raw = retained.read(STARTUP_LIMIT + 1)
    require(0 < len(raw) <= STARTUP_LIMIT and
            hashlib.sha256(raw).hexdigest() == control["startup_log"]["sha256"],
            "C2 retained startup evidence changed")
    require(os.pread(stream.fileno(), len(raw), 0) == raw, "C2 startup log replaced or truncated")
    require(scan_anchor_config(stream) ==
            (control["anchor_depth"], control["anchor_bytes_per_slot_max"]),
            "C2 startup configuration changed")


def metrics(text):
    """Read explicit u64 families; absent, duplicate or malformed data is not zero."""
    result = {"not_eligible": {}}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\s{]", line, maxsplit=1)[0]
        if name not in COUNTERS and name != REASONS:
            continue
        match = re.fullmatch(re.escape(name) + r'(?:\{reason="([a-z][a-z0-9_]*)"\})?\s+([0-9]+)', line)
        require(match is not None, "malformed C2 metric: " + name)
        reason, value = match.groups()
        value = int(value)
        require(value <= (1 << 64) - 1, "C2 metric exceeds u64")
        if name in COUNTERS:
            key = COUNTERS[name]
            require(reason is None and key not in result, "duplicate/labeled C2 lane counter")
            result[key] = value
        else:
            require(reason is not None, "joint refusal metric requires an explicit reason")
            if reason.startswith("anchor_"):
                require(reason not in result["not_eligible"], "duplicate C2 anchor reason")
                result["not_eligible"][reason] = value
    require(set(COUNTERS.values()) <= result.keys() and
            {"anchor_capture", "anchor_restore"} <= result["not_eligible"].keys(),
            "explicit C2 lane counters and anchor reasons required")
    return result


def trace_row(raw):
    line = ANSI.sub("", raw)
    if EVENT not in line:
        return None
    require(TARGET + ": " + EVENT in line, "unbound C2 lane-step event target")
    fields = {}
    for name in ("decode_ready_lanes", "joint_lane_steps", "serial_lane_steps"):
        found = re.findall(r"\b" + name + r"=([^\s]+)", line)
        require(len(found) == 1 and re.fullmatch(r"[0-9]+", found[0]) is not None,
                "missing/duplicate/malformed C2 event field: " + name)
        fields[name] = int(found[0])
    for name in ("route", "path"):
        found = re.findall(r"\b" + name + r'=(?:"([a-z0-9_-]+)"|([a-z0-9_-]+))(?=\s|$)', line)
        require(len(found) == 1, "missing/duplicate/malformed C2 event field: " + name)
        fields[name] = next(value for value in found[0] if value)
    ready, joint, serial = (fields[k] for k in ("decode_ready_lanes", "joint_lane_steps", "serial_lane_steps"))
    require(2 <= ready <= 4 and 0 < joint + serial <= ready and
            ((fields["path"] == "joint" and joint > 0 and serial == 0) or
             (fields["path"] == "serial" and serial > 0 and joint == 0)),
            "C2 event is not a scoped actual dispatch")
    return fields


def eligibility(before, after, counts):
    require(before["not_eligible"].keys() == after["not_eligible"].keys(),
            "C2 refusal reason coverage changed during cohort")
    result = {"scope": SCOPE, "not_eligible": {}}
    for key in COUNTERS.values():
        delta = after[key] - before[key]
        require(delta >= 0 and delta == counts[key], "C2 trace/counter delta mismatch: " + key)
        result[key] = delta
    for reason, value in before["not_eligible"].items():
        delta = after["not_eligible"][reason] - value
        require(delta >= 0, "C2 refusal counter decreased")
        result["not_eligible"][reason] = delta
    total = result["joint_lane_steps"] + result["serial_lane_steps"]
    require(total > 0, "C2 scoped eligibility denominator is zero")
    result["rate"] = result["joint_lane_steps"] / total
    return result


class Observation:
    """Start immediately before the acknowledged client barrier release; finish after owned clients stop."""
    def __init__(self, base_url, server_log, output, run_id, server_pid, cache_control=None):
        url = urlsplit(base_url)
        require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port is not None and
                url.username is None and url.password is None and url.path.rstrip("/") == "/v1" and
                not url.query and not url.fragment, "C2 metrics must use the owned local API endpoint")
        self.endpoint = f"http://127.0.0.1:{url.port}/metrics"
        self.output = Path(output)
        self.log_path = Path(server_log)
        self.run_id, self.server_pid = run_id, server_pid
        self.cache_control = cache_control
        self.log = regular_evidence(self.log_path)
        try:
            info = os.fstat(self.log.fileno())
            require(stat.S_ISREG(info.st_mode), "C2 log must be the owned regular stderr file")
            self.identity = (info.st_dev, info.st_ino)
            if cache_control is not None:
                checked_cache_control(cache_control, server_pid, run_id)
                require(cache_control["server_log"] == str(self.log_path), "C2 startup log path changed")
                check_startup_identity(cache_control, self.log)
            self.start_offset = info.st_size
            self.started_ns = time.monotonic_ns()
            self.before = self.snapshot("before")
        except BaseException:
            self.log.close()
            raise

    def snapshot(self, position):
        request = Request(self.endpoint, headers={"Accept": "text/plain"})
        with build_opener(ProxyHandler({})).open(request, timeout=10) as response:
            require(response.status == 200, "C2 metrics request failed")
            body = response.read()
        with (self.output / f"c2-{position}.prom").open("xb") as stream:
            stream.write(body)
        return metrics(body.decode("utf-8"))

    def finish(self):
        record = {"schema": SCHEMA, "status": "Invalid", "run_id": self.run_id,
                  "server_pid": self.server_pid, "server_log": str(self.log_path),
                  "start_offset": self.start_offset, "started_monotonic_ns": self.started_ns,
                  "before": self.before, "collector_sha256": sha(__file__),
                  "scope": SCOPE, "model_or_runtime_qualification": False}
        if self.cache_control is not None:
            record["c2_cache_control"] = self.cache_control
        try:
            after = self.snapshot("after")
            info, named = os.fstat(self.log.fileno()), self.log_path.stat()
            require((info.st_dev, info.st_ino) == (named.st_dev, named.st_ino) == self.identity and
                    info.st_size >= self.start_offset, "C2 server log replaced or truncated")
            if self.cache_control is not None:
                check_startup_identity(self.cache_control, self.log)
            record.update(after=after, end_offset=info.st_size)
            counts = {key: 0 for key in COUNTERS.values()}
            routes = {}
            self.log.seek(self.start_offset)
            with (self.output / "c2-trace.log").open("xb") as retained:
                while self.log.tell() < info.st_size:
                    raw = self.log.readline(info.st_size - self.log.tell())
                    require(raw.endswith(b"\n"), "C2 trace ends inside an unacknowledged log line")
                    retained.write(raw)
                    row = trace_row(raw.decode("utf-8"))
                    if row is not None:
                        for key in counts:
                            counts[key] += row[key]
                        route = row["route"] + ":" + row["path"]
                        routes[route] = routes.get(route, 0) + row["joint_lane_steps"] + row["serial_lane_steps"]
            record.update(eligibility=eligibility(self.before, after, counts), route_lane_steps=routes,
                          trace_sha256=sha(self.output / "c2-trace.log"), status="Observed")
            return {"path": str(self.output / "c2-observation.json"), "eligibility": record["eligibility"]}
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["finished_monotonic_ns"] = time.monotonic_ns()
            self.log.close()
            with (self.output / "c2-observation.json").open("x") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
