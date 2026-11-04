#!/usr/bin/env python3
"""ADR-063 Q4-GEMMA (#660): one supervised Q5 run under a Metal System Trace, retaining sanitized GPU tables only.

  gemma_mst_capture.py --output DIR -- python3 -B scripts/four_agent/q5_calibration.py CASE ... --output RUN

Starts `xctrace record --template 'Metal System Trace' --all-processes` before the Q5 command (so the
model load and the first turn are inside the trace), records the client clock (CLOCK_MONOTONIC) when
xctrace reports that recording started, runs the command, stops xctrace with SIGINT, exports
EXPORT_SCHEMAS as XML, scans them for credential patterns and deletes the raw bundle (ADR-042 rule: raw
traces can carry process environments; only scanned GPU-schema exports are retained). xctrace runs with a
minimal environment. DIR/capture.json records the anchor, the exports' sha256 and the Q5 exit code.

gemma_stage_profile.py mst RUN --exec-xml DIR/metal-gpu-execution-points.xml
  --mapping-xml DIR/metal-gpu-submission-to-command-buffer-id.xml --process <server process name>
  --offset-ns <capture.json anchor_monotonic_ns> aligns the GPU timeline to the run's request spans.
The trace perturbs the run; compare traced against untraced makespans of the same arm.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

XCTRACE = ["/usr/bin/xcrun", "xctrace"]
TEMPLATE = "Metal System Trace"
EXPORT_SCHEMAS = ("metal-gpu-execution-points", "metal-gpu-submission-to-command-buffer-id",
                  "metal-application-encoders-list")
STARTED = re.compile(r"Starting recording|Ctrl-C to stop", re.I)
# Credential shapes refused in retained exports (the ADR-042 scan, kept deliberately broad).
CREDENTIALS = re.compile(rb"(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|"
                         rb"sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,}|"
                         rb"(?i:(?:token|secret|password|passwd|api[_-]?key)\s*[=:]\s*\S{8,}))")
MINIMAL_ENV = ("PATH", "HOME", "TMPDIR", "USER", "LOGNAME")


def monotonic_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def record_argv(trace, time_limit_s):
    return XCTRACE + ["record", "--template", TEMPLATE, "--all-processes", "--no-prompt",
                      "--output", str(trace), "--time-limit", f"{time_limit_s}s"]


def export_argv(trace, schema):
    return XCTRACE + ["export", "--input", str(trace), "--xpath",
                      f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]']


def credential_hits(data):
    """Distinct credential-shaped matches (redacted to their first 6 bytes) in bytes."""
    return sorted({m.group(0)[:6].decode("latin-1") + "..." for m in CREDENTIALS.finditer(data)})


def minimal_env(environ):
    return {k: environ[k] for k in MINIMAL_ENV if k in environ}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--time-limit-s", type=int, default=1800)
    ap.add_argument("--start-timeout-s", type=int, default=120)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("a Q5 command is required after --")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    trace = out / "raw.trace"
    record = {"schema": "hf2q.adr063.q4-gemma.mst-capture.v1", "template": TEMPLATE, "command": command,
              "record_argv": record_argv(trace, args.time_limit_s)}
    xc = subprocess.Popen(record["record_argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env=minimal_env(os.environ), text=True)
    started = threading.Event()
    log = []

    def pump():
        for line in xc.stdout:
            log.append(line)
            if not started.is_set() and STARTED.search(line):
                record["anchor_monotonic_ns"] = monotonic_ns()
                started.set()
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    record["spawn_monotonic_ns"] = monotonic_ns()
    if not started.wait(args.start_timeout_s):
        xc.send_signal(signal.SIGINT)
        xc.wait(60)
        record["error"] = "xctrace did not report a started recording"
        (out / "capture.json").write_text(json.dumps({**record, "xctrace_log": log}, indent=1) + "\n")
        return 2
    rc = subprocess.call(command)
    record["q5_exit_code"] = rc
    record["q5_finished_monotonic_ns"] = monotonic_ns()
    xc.send_signal(signal.SIGINT)
    record["xctrace_exit_code"] = xc.wait()
    reader.join(10)
    record["xctrace_log"] = log
    exports = {}
    for schema in EXPORT_SCHEMAS:
        result = subprocess.run(export_argv(trace, schema), capture_output=True, env=minimal_env(os.environ))
        hits = credential_hits(result.stdout)
        if result.returncode != 0 or hits:
            exports[schema] = {"retained": False, "exit_code": result.returncode, "credential_hits": hits}
            continue
        path = out / f"{schema}.xml"
        path.write_bytes(result.stdout)
        exports[schema] = {"retained": True, "path": str(path), "bytes": len(result.stdout),
                           "sha256": hashlib.sha256(result.stdout).hexdigest()}
    record["exports"] = exports
    subprocess.run(["/bin/rm", "-rf", str(trace)], check=True)
    record["raw_trace_deleted"] = not trace.exists()
    (out / "capture.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("anchor_monotonic_ns", "q5_exit_code", "xctrace_exit_code")}))
    return rc if all(e["retained"] for e in exports.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
