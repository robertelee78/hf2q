"""Validate a managed measurement runtime attestation against its live process."""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from measurement import digest

SCHEMA = "hf2q.measurement-runtime.v1"


def validate_identity(manifest, snapshot):
    if manifest.get("state") != "running":
        raise ValueError("managed runtime attestation is not running")
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("missing supported runtime attestation schema")
    if (not isinstance(snapshot, dict) or not snapshot.get("process_pid")
            or snapshot.get("schema_version") != "hf2q.measurement-snapshot.v1"):
        raise ValueError("live runtime does not expose a measurement snapshot")
    if manifest.get("runtime_snapshot", snapshot) != snapshot:
        raise ValueError("runtime snapshot differs from recorded snapshot")
    if manifest.get("process_pid") != snapshot["process_pid"]:
        raise ValueError("runtime process differs from attested process")
    if manifest.get("runtime_snapshot_sha256") != digest(snapshot):
        raise ValueError("live runtime snapshot differs from attestation")
    ident = manifest.get("runtime_identity", {})
    expected = digest({k: v for k, v in ident.items() if k != "identity_sha256"})
    if ident.get("identity_sha256") != expected:
        raise ValueError("runtime identity digest is invalid")
    for field in ("binary_sha256", "tokenizer_sha256", "template_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(ident.get(field, ""))):
            raise ValueError(f"runtime identity lacks {field}")
    for field in ("model_id", "engine_config", "admission", "tokenizer_sha256", "template_sha256", "sampling_defaults", "active_controls"):
        if field not in ident or ident[field] != snapshot.get(field):
            raise ValueError(f"runtime identity disagrees with live {field}")
    artifacts = ident.get("model_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("runtime identity lacks model artifact hashes")
    if sum(artifact.get("role") == "model" for artifact in artifacts) != 1:
        raise ValueError("runtime identity requires exactly one model artifact")
    for artifact in artifacts:
        if (not re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256", "")))
                or type(artifact.get("bytes")) is not int or artifact["bytes"] <= 0
                or not artifact.get("role")):
            raise ValueError("invalid model artifact binding")
    if "source_commit" not in ident:
        raise ValueError("source_commit must be recorded, including explicit null when unknown")
    controls = ident.get("active_controls", {})
    for name in ("glp", "grammar"):
        if type(controls.get(name, {}).get("active")) is not bool:
            raise ValueError(f"runtime identity lacks known {name} activation state")
    return ident


def fetch_runtime_identity(base_url, manifest_path=None):
    if not manifest_path:
        raise ValueError("measurement requires a managed RUNTIME_MANIFEST (or JUDGE_RUNTIME_MANIFEST)")
    manifest = json.loads(Path(manifest_path).read_text())
    with urllib.request.urlopen(base_url.rstrip("/") + "/hf2q/v1/runtime", timeout=15) as stream:
        payload = json.load(stream)
    snapshot = payload.get("measurement_snapshot")
    return validate_identity(manifest, snapshot)


def require_unconstrained(identity):
    if identity.get("active_controls", {}).get("vision_projector") is not False:
        raise ValueError("paired baseline requires verified inactive vision projector")
    if identity.get("active_controls", {}).get("dwq_overlay") is not False:
        raise ValueError("paired baseline requires verified inactive DWQ overlay")
    for name in ("glp", "grammar"):
        control = identity.get("active_controls", {}).get(name, {})
        if control.get("active") is not False:
            raise ValueError(f"paired baseline requires verified inactive {name}")


def verify_unchanged(base_url, manifest_path, expected):
    current = fetch_runtime_identity(base_url, manifest_path)
    if current != expected:
        raise ValueError("measurement runtime changed during the pass")
    return current


def attested_model(identity, requested=None):
    model = identity.get("model_id")
    if not model or (requested and requested != model):
        raise ValueError("requested model alias differs from the attested resident model")
    return model
