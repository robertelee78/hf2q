"""ADR-063 Q3 frozen campaign identities: arm binaries, model, peer build, harness, Python and host.

Recorded once at campaign start and recomputed before every run; any difference halts the campaign.
Models are never hashed inline (q5_artifacts: conversion-receipt digest or size + mtime).
"""
import os
from pathlib import Path
import subprocess
import sys

import q5_artifacts
import q5_case
import q5_host
import quality_campaign as qc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def git(*argv):
    try:
        return subprocess.run(["git", "-C", str(REPO), *argv], capture_output=True, text=True, timeout=30,
                              check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def harness():
    files = (sorted(HERE.glob("*.py")) + sorted((HERE / "q5_cases").glob("*.json"))
             + sorted(p for p in (HERE / "fixtures").iterdir() if p.is_file())
             + sorted((REPO / "scripts").glob("serve_*.sh")))
    status = git("status", "--porcelain", "--", "scripts")
    return {"git_commit": (git("rev-parse", "HEAD") or "").strip() or None,
            "git_dirty": None if status is None else sorted(status.splitlines()),
            "sources": {str(p.relative_to(REPO)): q5_artifacts.sha256(p) for p in files}}


def binary(path, siblings):
    if not os.path.isfile(path):
        return {"path": path, "sha256": None, "exists": False}
    identity = {"path": path, "sha256": q5_artifacts.sha256(path), "bytes": os.path.getsize(path), "exists": True}
    if siblings:  # the peer server is a launcher over sibling dylibs
        identity["sibling_dylibs"] = {p.name: q5_artifacts.sha256(p) for p in sorted(Path(path).parent.glob("*.dylib"))
                                      if p.is_file() and not p.is_symlink()}
    return identity


def arm(name, case, overrides, environ):
    import q5_calibration
    plan = q5_case.build_plan(case, q5_calibration.builtins(Path("/q3-identity") / name), qc.arm_environ(environ, overrides))
    artifact = plan["artifact"]
    exists = not q5_case.PLACEHOLDER.search(artifact["path"]) and os.path.isfile(artifact["path"])
    return {"binary": binary(plan["binary"], name.startswith("peer")),
            "artifact": q5_artifacts.artifact_identity(artifact["path"], artifact.get("sha256") or None) if exists else None,
            "peer_build_info": plan["variables"].get("PEER_BUILD_INFO"),
            "missing_variables": plan["missing_variables"],
            "file_checks": q5_artifacts.file_checks(plan) if plan["complete"] else {},
            "inputs": q5_artifacts.input_hashes(plan)}


def identities(plan, cases, environ):
    arms = {name: arm(name, cases[name], spec["overrides"], environ) for name, spec in plan["arms"].items()}
    python = Path(sys.executable).resolve()
    return {"arms": arms, "harness": harness(), "python": {"path": str(python), "sha256": q5_artifacts.sha256(python)},
            "host": q5_host.host_identity(), "negative_control_receipt": plan.get("negative_control_receipt")}


def same_model(ids):
    keys = set()
    for identity in ids["arms"].values():
        artifact = identity["artifact"]
        if artifact is None:
            return False
        keys.add(artifact["sha256"] or (artifact["path"], artifact["bytes"]))
    return len(keys) == 1


def launch_problems(plan, ids):
    """Everything that keeps a campaign from starting; empty when launchable."""
    problems = [f"--{name}-bin is required" for name in ("stock", "experiment") if not plan["declared"][name]["binary"]]
    problems += [f"{name}: missing variables {','.join(i['missing_variables'])}" for name, i in ids["arms"].items()
                 if i["missing_variables"]]
    problems += [f"{name}: {what} not found" for name, i in ids["arms"].items() for what, ok in i["file_checks"].items() if not ok]
    problems += [f"{name}: binary {i['binary']['path']} not found" for name, i in ids["arms"].items() if not i["binary"]["exists"]]
    if not same_model(ids):
        problems.append("every arm must serve the same model artifact (sha256, or path and size)")
    stock, experiment = ids["arms"]["stock"]["binary"]["sha256"], ids["arms"]["experiment"]["binary"]["sha256"]
    if stock is not None and stock == experiment:
        problems.append("stock and experiment binaries are identical")
    for name in ("stock", "experiment"):
        commit = plan["declared"][name]["commit"]
        if not (isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
            problems.append(f"--{name}-commit must be the 40-hex source commit of the {name} binary")
    if ids["harness"]["git_commit"] is None or ids["harness"]["git_dirty"]:
        problems.append("the harness checkout must be a clean Git commit (scripts/ has uncommitted changes)")
    if not plan["declared"]["qe_identity"]:
        problems.append("--qe-identity (the named QE session/host executing Q3) is required")
    if "negative_control_receipt" not in plan:
        problems.append("--negative-control-receipt is required")
    return problems


def diff(frozen, current, prefix=""):
    """Dotted paths where two identity documents differ."""
    if isinstance(frozen, dict) and isinstance(current, dict):
        return [d for k in sorted(set(frozen) | set(current))
                for d in diff(frozen.get(k), current.get(k), f"{prefix}.{k}" if prefix else k)]
    return [] if frozen == current else [prefix or "<root>"]
