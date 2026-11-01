"""ADR-063 Q3 frozen campaign identities: arm binaries, model, peer build, harness, Python and host.

Recorded once at campaign start and recomputed before every run; any difference halts the campaign.
Models are never hashed inline (q5_artifacts: conversion-receipt digest or size + mtime).
"""
import os
from pathlib import Path
import subprocess
import sys

import native_peers
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
    identity = {"binary": binary(plan["binary"], name.startswith("peer")),
                "artifact": q5_artifacts.artifact_identity(artifact["path"], artifact.get("sha256") or None) if exists else None,
                "peer_build_info": plan["variables"].get("PEER_BUILD_INFO"),
                "missing_variables": plan["missing_variables"],
                "file_checks": q5_artifacts.file_checks(plan) if plan["complete"] else {},
                "inputs": q5_artifacts.input_hashes(plan)}
    if native_peers.is_native(qc.engine_kind(case)):  # the arm's build identity and frozen native runtime
        identity.update(native_peers.adapter(qc.engine_kind(case)).campaign_identity(plan))
    return identity


def identities(plan, cases, environ):
    arms = {name: arm(name, cases[name], spec["overrides"], environ) for name, spec in plan["arms"].items()}
    python = Path(sys.executable).resolve()
    return {"arms": arms, "harness": harness(), "python": {"path": str(python), "sha256": q5_artifacts.sha256(python)},
            "host": q5_host.host_identity(), "negative_control_receipt": plan.get("negative_control_receipt")}


def native_arms(plan):
    """Arms that serve their own native artifact (native_peers kinds); excluded from the same-model rule."""
    return [name for name, spec in plan["arms"].items() if native_peers.is_native(spec["engine_kind"])]


def same_model(ids, native=()):
    keys = set()
    for name, identity in ids["arms"].items():
        if name in native:
            continue
        artifact = identity["artifact"]
        if artifact is None:
            return False
        keys.add(artifact["sha256"] or (artifact["path"], artifact["bytes"]))
    return len(keys) == 1


# Arm overrides outside the controls identity: the client approval template must be regenerated whenever
# the frozen client sources change (a controls campaign may run newer harness sources), and the hf2q
# binaries are compared by content (binary_sha256).
UNBOUND_OVERRIDES = ("CLIENT_APPROVAL_TEMPLATE", "HF2Q_BIN")


def override_identity(spec, identity):
    """An arm's overrides by content: the sha256 of a file its argv names (identity `inputs`), else the
    literal value. Other absolute paths (model, binaries, directories) are bound by the model, binary and
    native runtime identities instead."""
    inputs = identity.get("inputs") or {}
    bound = {}
    for name, value in spec["overrides"].items():
        if name in UNBOUND_OVERRIDES:
            continue
        if isinstance(value, str) and value in inputs:
            bound[name] = {"sha256": inputs[value]}
        elif not (isinstance(value, str) and os.path.isabs(value)):
            bound[name] = value
    return bound


def controls_identity(plan, ids):
    """What a controls-only campaign must share with its primary of record (#369 amendment): family,
    measured arms, peer-unavailable evidence, host, Python, and per arm the case, binary, declared commit,
    model artifact, override contents, peer build, sibling dylibs and native runtime. Harness sources and
    approval templates may differ."""
    arms = {}
    for name, spec in plan["arms"].items():
        identity = ids["arms"][name]
        artifact = identity["artifact"] or {}
        arms[name] = {"engine_kind": spec["engine_kind"], "case_sha256": spec["case_sha256"],
                      "binary_sha256": identity["binary"]["sha256"],
                      "model": {"sha256": artifact.get("sha256"), "bytes": artifact.get("bytes")},
                      "overrides": override_identity(spec, identity)}
        if name in ("stock", "experiment"):
            arms[name]["commit"] = plan["declared"][name]["commit"]
        else:
            arms[name].update(peer_build_info=identity["peer_build_info"],
                              sibling_dylibs=identity["binary"].get("sibling_dylibs"),
                              native_runtime=identity.get("native_runtime"))
    return {"family": plan["family"], "measured_arms": plan["measured_arms"],
            "peer_unavailable_evidence": sorted(e["sha256"] for e in plan["peer_unavailable_evidence"]),
            "host": ids["host"], "python_sha256": ids["python"]["sha256"], "arms": arms}


def launch_problems(plan, ids):
    """Everything that keeps a campaign from starting; empty when launchable."""
    problems = [f"--{name}-bin is required" for name in ("stock", "experiment") if not plan["declared"][name]["binary"]]
    problems += [f"{name}: missing variables {','.join(i['missing_variables'])}" for name, i in ids["arms"].items()
                 if i["missing_variables"]]
    problems += [f"{name}: {what} not found" for name, i in ids["arms"].items() for what, ok in i["file_checks"].items() if not ok]
    problems += [f"{name}: binary {i['binary']['path']} not found" for name, i in ids["arms"].items() if not i["binary"]["exists"]]
    native = native_arms(plan)
    if not same_model(ids, native):
        problems.append("every arm must serve the same model artifact (sha256, or path and size)")
    problems += [f"{name}: native artifact with its sha256 is required" for name in native
                 if not (ids["arms"][name]["artifact"] or {}).get("sha256")]
    hf2q_sha = (ids["arms"]["experiment"]["artifact"] or {}).get("sha256")
    for name in native:  # an ambient hf2q artifact must never stand in for the native artifact
        problems += native_peers.adapter(plan["arms"][name]["engine_kind"]).launch_problems(
            name, ids["arms"][name]["artifact"] or {}, hf2q_sha)
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
    record = plan.get("controls_of")
    if record is not None:
        drift = diff(record["identity"], controls_identity(plan, ids))
        if drift:
            problems.append(f"differs from the primary of record {record['verdict']['path']}: " + ", ".join(drift))
    return problems


def diff(frozen, current, prefix=""):
    """Dotted paths where two identity documents differ."""
    if isinstance(frozen, dict) and isinstance(current, dict):
        return [d for k in sorted(set(frozen) | set(current))
                for d in diff(frozen.get(k), current.get(k), f"{prefix}.{k}" if prefix else k)]
    return [] if frozen == current else [prefix or "<root>"]
