#!/usr/bin/env python3
"""Read-only actual shared census join; exit3 means no qualifying live cohort."""
import argparse
import json
import os
from pathlib import Path
import re
import time

MARKER = "Qwen shared ordinary target completed; partial operators, not whole-body B4"
ROW = re.compile(r"Row \{ request: (\d+), handle: SlotHandle \{ slot_id: SlotId\((\d+)\), generation: (\d+) \}, position: (\d+), token: (\d+) \}")
LIMIT = 4 * 1024 * 1024

def census(line):
    if MARKER not in line:
        return None
    width = re.search(r"\bwidth=(\d+)\b", line)
    if not width or int(width[1]) != 2:
        raise ValueError("exclusive two-stream case requires actual width2")
    owners = [dict(zip(("request", "slot", "generation", "position", "token"), map(int, r)))
              for r in ROW.findall(line)]
    if len(owners) != 2 or len({r["request"] for r in owners}) != 2 or len({r["slot"] for r in owners}) != 2:
        raise ValueError("missing, duplicate or non-independent actual owners")
    if any(r["slot"] >= 4 or r["position"] >= 262144 or r["token"] >= 248320 for r in owners):
        raise ValueError("actual owner geometry outside selected Qwen38 profile")
    if (not re.search(r"\bsingleton_attention_maps=128\b", line) or
        not re.search(r"\bshared_operator_maps=66\b", line)):
        raise ValueError("partial operator census mismatch")
    return owners

def live(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--log", type=Path, required=True)
    a.add_argument("--offset", type=int, required=True)
    a.add_argument("--active-pid", type=int, required=True)
    a.add_argument("--survivor-pid", type=int, required=True)
    a.add_argument("--seconds", type=int, required=True)
    a.add_argument("--output", type=Path, required=True)
    args = a.parse_args()
    if (args.offset < 0 or not 1 <= args.seconds <= 180 or
        not all(1 < pid < 2**31 for pid in (args.active_pid, args.survivor_pid)) or
        args.active_pid == args.survivor_pid):
        raise ValueError("bounded distinct stream configuration")
    started = time.monotonic_ns()
    deadline = time.monotonic() + args.seconds
    status = "InconclusiveNoLiveSharedCohort"
    found = None
    consumed = 0
    pending = b""
    with args.log.open("rb") as src:
        initial = os.fstat(src.fileno())
        if args.offset > initial.st_size:
            raise ValueError("census log truncated before boundary")
        src.seek(args.offset)
        while time.monotonic() < deadline and live(args.active_pid) and live(args.survivor_pid):
            current = args.log.stat()
            if (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino):
                raise ValueError("census log replaced during case")
            if os.fstat(src.fileno()).st_size < src.tell():
                raise ValueError("census log truncated during case")
            chunk = src.read(65536)
            consumed += len(chunk)
            if consumed > LIMIT:
                raise ValueError("bounded census log quota exceeded")
            pending += chunk
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                owners = census(raw.decode("utf8", "strict"))
                if owners is not None:
                    if not live(args.active_pid) or not live(args.survivor_pid):
                        break
                    found = {"owners": owners, "raw_line": raw.decode(),
                             "file_end_offset": src.tell() - len(pending),
                             "observed_monotonic_ns": time.monotonic_ns()}
                    status = "WitnessedTwoDistinctLiveOwners"
                    break
            if found:
                break
            time.sleep(0.02)
    result = {"schema": "hf2q.s3.shared-cancel-witness.v1", "status": status,
              "start_offset": args.offset, "log_dev": initial.st_dev, "log_ino": initial.st_ino,
              "started_monotonic_ns": started, "active_curl_pid": args.active_pid,
              "survivor_curl_pid": args.survivor_pid, "bytes_read": consumed,
              "witness": found, "requires_exclusive_case": True, "whole_body_b4": False}
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    return 0 if found else 3

if __name__ == "__main__":
    raise SystemExit(main())
