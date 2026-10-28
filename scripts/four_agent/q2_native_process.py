"""Capture-only detached native-client ownership. Never weakens tool ownership.
Only Popen's fresh root PG and observed fresh PPID descendants can be owned.
Metrics/identity uncertainty invalidates cleanup, never authorizes a new group.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from qwen38_n1_verdict import require

MAX_NODES = 64
MAX_GROUPS = 8
MAX_QUERIES = 4096
MAX_RAW_BYTES = 64 * 1024 * 1024
RSS_LIMIT = 1024 * 1024 * 1024


def parse_snapshot(raw):
    require(type(raw) is bytes and len(raw) <= 4 * 1024 * 1024, "bounded native process inventory")
    rows = {}
    for line in raw.decode("ascii", "strict").splitlines():
        parts = line.split()
        require(len(parts) == 9 and all(x.isdecimal() for x in parts[:4]), "typed native process inventory")
        pid, ppid, pgid, rss = map(int, parts[:4])
        require(pid not in rows and pid > 0 and pgid > 0, "unique process identities")
        rows[pid] = (ppid, pgid, rss * 1024, " ".join(parts[4:]))
    return rows


class Inventory:
    def __init__(self, output):
        self.output = Path(output); self.count = self.bytes = 0
    def query(self):
        require(self.count < MAX_QUERIES, "native inventory query quota")
        index = self.count; self.count += 1; begin = time.monotonic_ns()
        record = {"begin_ns": begin}
        try:
            result = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,pgid=,rss=,lstart="],
                                    capture_output=True, timeout=.5)
            raw, err = result.stdout, result.stderr
            require(len(raw) + len(err) <= 4 * 1024 * 1024, "native query raw quota")
            self.bytes += len(raw) + len(err); require(self.bytes <= MAX_RAW_BYTES, "native inventory total raw quota")
            record.update(exit=result.returncode, stdout=raw.decode("ascii", "replace"), stderr=err.decode("ascii", "replace"))
            require(result.returncode == 0 and not err, "native process query failed")
            rows = parse_snapshot(raw)
            require(time.monotonic_ns() - begin <= 500000000, "native inventory query late")
            return rows
        except BaseException as error:
            record["error"] = repr(error); raise
        finally:
            record["end_ns"] = time.monotonic_ns()
            (self.output / ("native-ps-" + str(index) + ".json")).write_text(json.dumps(record) + "\n")


class Ownership:
    def __init__(self, baseline, leader):
        require(type(leader) is int and leader > 1 and leader not in baseline, "fresh native root PID")
        self.baseline = dict(baseline); self.leader = leader
        self.known = {}; self.depth = {leader: 0}
        # Popen(start_new_session=True), not a group guessed from process name.
        self.groups = {leader: None}; self.group_depth = {leader: 0}
    def observe(self, rows):
        require(all(pid not in rows or rows[pid][3] == stamp for pid, stamp in self.known.items()), "native owned PID reused")
        owned = {self.leader} | (self.known.keys() & rows.keys())
        changed = True
        while changed:
            new = {pid for pid, row in rows.items() if row[0] in owned}
            changed = not new <= owned; owned |= new
        require(len(owned | self.known.keys()) <= MAX_NODES, "native owned node quota")
        # Register provenance BEFORE inspecting a detached child's PG. A later
        # refusal still retains identity evidence for safe individual cleanup.
        for pid in owned & rows.keys():
            require(pid not in self.baseline, "native descendant was preexisting")
            self.known.setdefault(pid, rows[pid][3])
        for _ in range(MAX_NODES):
            for pid in owned & rows.keys():
                parent = rows[pid][0]
                if parent in self.depth: self.depth.setdefault(pid, self.depth[parent] + 1)
        for pid in owned & rows.keys():
            pgid = rows[pid][1]
            if pgid not in self.groups:
                require(pgid not in self.baseline and not any(r[1] == pgid for r in self.baseline.values()), "native child joined old group")
                require(pgid in self.known and pgid in rows and rows[pgid][1] == pgid,
                        "detached group leader must be observed fresh owned descendant")
                require(len(self.groups) < MAX_GROUPS, "native owned group quota")
                self.groups[pgid] = self.known[pgid]; self.group_depth[pgid] = self.depth.get(pgid, MAX_NODES)
            if self.groups[pgid] is None and pgid in self.known: self.groups[pgid] = self.known[pgid]
            require(all(p in self.known and self.known[p] == row[3] for p, row in rows.items() if row[1] == pgid),
                    "unowned member joined native group")
        require(sum(rows[p][2] for p in self.known.keys() & rows.keys()) <= RSS_LIMIT, "native capture observed aggregate RSS1GiB")
        return self.known.keys() & rows.keys()
    def safe_targets(self, rows, root_running=False):
        groups = []; individuals = []
        for pgid in sorted(self.groups, key=lambda p: self.group_depth[p], reverse=True):
            if rows is None:
                # The live Popen root is still ours. A detached group needs a
                # fresh identity check; unavailable metrics cannot authorize
                # signaling a possibly reused group ID.
                if pgid == self.leader and root_running: groups.append(pgid)
            elif (pgid not in rows or self.groups[pgid] is None or rows[pgid][3] == self.groups[pgid]) and all(
                    p in self.known and self.known[p] == r[3] for p, r in rows.items() if r[1] == pgid):
                groups.append(pgid)
        if rows is not None:
            individuals = [p for p in self.known if p in rows and rows[p][3] == self.known[p] and rows[p][1] not in groups]
        return groups, individuals


def stop_owned(process, ownership, inventory):
    result = {"actions": [], "errors": [], "reaped": False, "verified": False}
    def observe():
        rows = None
        try:
            rows = inventory.query(); live = ownership.observe(rows)
            return rows, set(live)
        except BaseException as error:
            result["errors"].append(repr(error)); return rows, None
    for sig, seconds in ((signal.SIGTERM, 3), (signal.SIGKILL, 5)):
        rows, live = observe()
        groups, individuals = ownership.safe_targets(rows, process.poll() is None)
        for pgid in groups: # registered detached groups first; fresh root last
            try: os.killpg(pgid, sig); result["actions"].append({"pgid": pgid, "signal": sig.name})
            except ProcessLookupError: pass
            except BaseException as error: result["errors"].append(repr(error))
        for pid in individuals:
            try: os.kill(pid, sig); result["actions"].append({"pid": pid, "signal": sig.name})
            except ProcessLookupError: pass
            except BaseException as error: result["errors"].append(repr(error))
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            process.poll(); _, live = observe()
            if live == set() and process.poll() is not None: break
            time.sleep(.02)
        if live == set() and process.poll() is not None: break
    try: process.wait(timeout=1); result["reaped"] = True
    except BaseException as error: result["errors"].append(repr(error))
    rows, live = observe()
    # Groups cannot be called empty merely because their old leader is gone.
    groups_empty = rows is not None and not any(r[1] in ownership.groups for r in rows.values())
    result["final_observation"] = {"available": live is not None, "live_owned_pids": sorted(live) if live is not None else None,
                                   "all_registered_groups_empty": groups_empty}
    result["verified"] = result["reaped"] and live == set() and groups_empty and not result["errors"]
    result["ownership"] = {"known": ownership.known, "groups": ownership.groups}
    return result
