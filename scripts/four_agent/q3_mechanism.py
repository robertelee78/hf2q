"""ADR-063 Qwen3.8 mechanism gate, reduced from the campaign's own passing n4 runs.

Only new plans opt in. Retained input rows count the matched proposals plus the
bonus decision (including terminal decisions), not HTTP completion usage.
"""
import hashlib
import json
from pathlib import Path
import re

CONTRACT = {"version": 1, "minimum": 5.304,
            "metric": "accepted_plus_bonus_per_lane_cycle"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counter(line, name):
    values = re.findall(r"\b" + name + r"=([^\s]+)", line)
    if len(values) != 1 or re.fullmatch(r"[0-9]+", values[0]) is None:
        raise ValueError("missing, duplicate or malformed DFlash2 " + name)
    return int(values[0])


def counts(text):
    """One verified window is one lane-cycle; optional stage totals must agree."""
    accepted = lanes = stage_rows = stage_lanes = 0
    for line in ANSI.sub("", text).splitlines():
        if "Qwen DFlash2 verified window" in line:
            value = counter(line, "accepted")
            if not 1 <= value <= 8:
                raise ValueError("invalid DFlash2 verified-window count")
            accepted += value
            lanes += 1
        if "Qwen DFlash2 stage timing" in line:
            n, rows = counter(line, "active_lanes"), counter(line, "accepted_rows")
            if not 1 <= n <= 4 or not n <= rows <= 8 * n:
                raise ValueError("invalid DFlash2 stage counts")
            stage_rows += rows
            stage_lanes += n
    if not lanes:
        raise ValueError("missing DFlash2 verified-window telemetry")
    if stage_lanes and (stage_rows, stage_lanes) != (accepted, lanes):
        raise ValueError("DFlash2 stage/window totals disagree")
    return accepted, lanes


def evaluate(manifest):
    """Return None for old plans, otherwise a fail-closed source-bound proof."""
    plan = manifest["plan"]
    if "qwen38_mechanism" not in plan:
        return None
    result = {"contract": plan["qwen38_mechanism"], "status": "unresolved", "runs": [],
              "accepted_rows": 0, "lane_cycles": 0, "ratio": None}
    try:
        if plan["family"] != "qwen38" or plan["qwen38_mechanism"] != CONTRACT:
            raise ValueError("invalid Qwen3.8 mechanism contract")
        ids = manifest["identities"]
        source = plan["declared"]["experiment"]["commit"]
        if source != ids["harness"]["git_commit"]:
            raise ValueError("mechanism source differs from frozen candidate source")
        frozen = ids["arms"]["experiment"]
        if (re.fullmatch(r"[0-9a-f]{40}", source) is None
                or re.fullmatch(r"[0-9a-f]{64}", frozen["binary"]["sha256"]) is None
                or re.fullmatch(r"[0-9a-f]{64}", frozen["artifact"]["sha256"]) is None
                or type(frozen["artifact"]["bytes"]) is not int or frozen["artifact"]["bytes"] <= 0):
            raise ValueError("missing canonical mechanism source/binary/model binding")
        result["binding"] = {"source_commit": source, "binary_sha256": frozen["binary"]["sha256"],
                             "artifact": frozen["artifact"], "workload": "n4",
                             "case_sha256": plan["arms"]["experiment"]["case_sha256"]}
        base_case = Path(plan["arms"]["experiment"]["case"])
        if digest(base_case) != plan["arms"]["experiment"]["case_sha256"]:
            raise ValueError("mechanism candidate case changed")
        expected_case = json.loads(base_case.read_text())
        stage = next(s for s in manifest["stages"] if s["workload"] == "n4")
        for block in stage["blocks"]:
            for entry in block["entries"]:
                if entry["arm"] != "experiment" or entry["outcome"] != "Pass":
                    continue
                attempt = entry["attempts"][-1]
                root = Path(attempt["output"])
                receipt_path, log = root / "receipt.json", root / "server.stderr"
                if digest(receipt_path) != attempt["receipt_sha256"]:
                    raise ValueError("mechanism Q5 receipt changed")
                if digest(log) != attempt["server_stderr_sha256"]:
                    raise ValueError("mechanism server log changed")
                receipt = json.loads(receipt_path.read_text())
                if receipt["schema"] != "hf2q.q5.calibration-receipt.v1" or receipt["family"] != "qwen38":
                    raise ValueError("mechanism run is not a Qwen3.8 Q5 receipt")
                if receipt["binary"]["sha256"] != frozen["binary"]["sha256"]:
                    raise ValueError("mechanism binary differs from frozen candidate")
                if any(receipt["artifact"].get(k) != frozen["artifact"].get(k) for k in ("sha256", "bytes")):
                    raise ValueError("mechanism model differs from frozen candidate")
                if receipt["verdict"]["verdict"] != "CalibrationPass" or receipt.get("cleanup_proven") is not True:
                    raise ValueError("mechanism run lacks fresh Q5 CalibrationPass/cleanup")
                case = json.loads(Path(attempt["case"]).read_text())
                if (receipt["workload"] != case["workload"] or case != expected_case
                        or receipt["case"]["sha256"] != digest(attempt["case"])):
                    raise ValueError("mechanism workload differs from measured case")
                rows, lanes = counts(log.read_text())
                result["runs"].append({"entry": entry["name"], "receipt": str(receipt_path),
                                       "receipt_sha256": attempt["receipt_sha256"], "log": str(log),
                                       "log_sha256": digest(log), "accepted_rows": rows,
                                       "lane_cycles": lanes, "ratio": rows / lanes})
                result["accepted_rows"] += rows
                result["lane_cycles"] += lanes
        if not result["runs"]:
            raise ValueError("no passing measured n4 candidate mechanism runs")
        result["ratio"] = result["accepted_rows"] / result["lane_cycles"]
        result["status"] = "pass" if result["ratio"] >= CONTRACT["minimum"] else "loss"
        result["reason"] = None if result["status"] == "pass" else "Qwen3.8 mechanism below 5.304"
    except (OSError, ValueError, KeyError, IndexError, StopIteration, TypeError) as error:
        result["reason"] = str(error) or type(error).__name__
    return result
