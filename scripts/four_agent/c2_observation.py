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
    def __init__(self, base_url, server_log, output, run_id, server_pid):
        url = urlsplit(base_url)
        require(url.scheme == "http" and url.hostname == "127.0.0.1" and url.port is not None and
                url.username is None and url.password is None and url.path.rstrip("/") == "/v1" and
                not url.query and not url.fragment, "C2 metrics must use the owned local API endpoint")
        self.endpoint = f"http://127.0.0.1:{url.port}/metrics"
        self.output = Path(output)
        self.log_path = Path(server_log)
        self.run_id, self.server_pid = run_id, server_pid
        self.log = os.fdopen(os.open(self.log_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC), "rb")
        try:
            info = os.fstat(self.log.fileno())
            require(stat.S_ISREG(info.st_mode), "C2 log must be the owned regular stderr file")
            self.identity = (info.st_dev, info.st_ino)
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
        try:
            after = self.snapshot("after")
            info, named = os.fstat(self.log.fileno()), self.log_path.stat()
            require((info.st_dev, info.st_ino) == (named.st_dev, named.st_ino) == self.identity and
                    info.st_size >= self.start_offset, "C2 server log replaced or truncated")
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
