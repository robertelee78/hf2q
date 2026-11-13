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
import q3_splash_evidence

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
    if native_peers.has_adapter(qc.engine_kind(case)):  # the arm's build identity and frozen native runtime
        identity.update(native_peers.adapter(qc.engine_kind(case)).campaign_identity(plan))
    return identity


def stock_record_files(record):
    """Current sha256 of the bound stock record's verdict.json and campaign.json (None when a file is gone)."""
    return {name: q5_artifacts.sha256(record[name]["path"]) if os.path.isfile(record[name]["path"]) else None
            for name in ("verdict", "campaign")}


def identities(plan, cases, environ):
    """Frozen identities. A stock-record campaign (--stock-record) also freezes the bound record files' sha256, so a
    record changed after binding halts the campaign before the next run."""
    arms = {name: arm(name, cases[name], spec["overrides"], environ) for name, spec in plan["arms"].items()}
    python = Path(sys.executable).resolve()
    return {"arms": arms, "harness": harness(), "python": {"path": str(python), "sha256": q5_artifacts.sha256(python)},
            "host": q5_host.host_identity(), "negative_control_receipt": plan.get("negative_control_receipt"),
            **({"splash_substitution_inputs": q3_splash_evidence.current_inputs(plan["qwen38_splash_substitution"])}
               if "qwen38_splash_substitution" in plan else {}),
            **({"stock_record": stock_record_files(plan["stock_record"])} if "stock_record" in plan else {})}


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


def arm_problems(ids):
    """Per-arm launch problems shared with the ADR-065 driver: unset variables, missing files and binaries."""
    problems = [f"{name}: missing variables {','.join(i['missing_variables'])}" for name, i in ids["arms"].items()
                if i["missing_variables"]]
    problems += [f"{name}: {what} not found" for name, i in ids["arms"].items() for what, ok in i["file_checks"].items() if not ok]
    return problems + [f"{name}: binary {i['binary']['path']} not found" for name, i in ids["arms"].items()
                       if not i["binary"]["exists"]]


def harness_clean(ids):
    return ids["harness"]["git_commit"] is not None and not ids["harness"]["git_dirty"]


def launch_problems(plan, ids, output=None):
    """Everything that keeps a campaign from starting; empty when launchable. output: the campaign directory, which
    same-artifact arms check (Splash run directories must be on APFS). Without a stock arm (--no-stock-arm) only the
    experiment binary and commit are required; a bound stock record must still be the frozen files, measured on this
    campaign's experiment binary."""
    hf2q = [name for name in ("stock", "experiment") if name in plan["arms"]]
    problems = [f"--{name}-bin is required" for name in hf2q if not plan["declared"][name]["binary"]]
    problems += arm_problems(ids)
    native = native_arms(plan)
    if not same_model(ids, native):
        problems.append("every arm must serve the same model artifact (sha256, or path and size)")
    problems += [f"{name}: native artifact with its sha256 is required" for name in native
                 if not (ids["arms"][name]["artifact"] or {}).get("sha256")]
    hf2q_sha = (ids["arms"]["experiment"]["artifact"] or {}).get("sha256")
    same_artifact = [name for name, spec in plan["arms"].items() if native_peers.is_same_artifact(spec["engine_kind"])]
    # Native arms: an ambient hf2q artifact must never stand in for the native artifact. Same-artifact arms
    # (Splash): the artifact must be the experiment's GGUF itself, the build found must be the declared one, and the
    # output volume must suit its run directories.
    for name in native:
        spec = plan["arms"][name]
        problems += native_peers.adapter(spec["engine_kind"]).launch_problems(
            name, ids["arms"][name]["artifact"] or {}, hf2q_sha, q5_case.loads(Path(spec["case"]).read_bytes()))
    for name in same_artifact:
        problems += native_peers.adapter(plan["arms"][name]["engine_kind"]).launch_problems(
            name, ids["arms"][name]["artifact"] or {}, hf2q_sha, ids["arms"][name], output)
    experiment = ids["arms"]["experiment"]["binary"]["sha256"]
    stock = ids["arms"]["stock"]["binary"]["sha256"] if "stock" in ids["arms"] else None
    if stock is not None and stock == experiment:
        problems.append("stock and experiment binaries are identical")
    for name in hf2q:
        commit = plan["declared"][name]["commit"]
        if not (isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)):
            problems.append(f"--{name}-commit must be the 40-hex source commit of the {name} binary")
    bound = plan.get("stock_record")
    if bound is not None:
        if bound["experiment_binary_sha256"] != experiment:
            problems.append(f"the stock record {bound['verdict']['path']} was measured on experiment binary "
                            f"{bound['experiment_binary_sha256']}, not this campaign's --experiment-bin ({experiment})")
        files = ids.get("stock_record") or {}
        changed = [name for name in ("verdict", "campaign") if files.get(name) != bound[name]["sha256"]]
        if changed:
            problems.append(f"the stock record {' and '.join(bound[n]['path'] for n in changed)} changed since the "
                            "campaign bound it (freeze the record read-only before binding)")
    if not harness_clean(ids):
        problems.append("the harness checkout must be a clean Git commit (scripts/ has uncommitted changes)")
    if not plan["declared"]["qe_identity"]:
        problems.append("--qe-identity (the named QE session/host executing Q3) is required")
    if "negative_control_receipt" not in plan:
        problems.append("--negative-control-receipt is required")
    if plan.get("qwen38_mechanism") and plan.get("splash_required") is False:
        try:
            q3_splash_evidence.check(plan, q5_case.loads(Path(plan["arms"]["experiment"]["case"]).read_bytes()),
                                    ids["arms"]["experiment"]["artifact"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append(f"invalid-Splash substitution: {error}")
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
