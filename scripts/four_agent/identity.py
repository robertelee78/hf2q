#!/usr/bin/env python3
"""Offline identity checks for ADR-063. Never launches a model or a build."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

SCHEMA = "hf2q.four-agent-identities.v1"
MODELS = {"qwen38", "deepseek4", "qwen36", "gemma4"}
ARMS = {"stock", "experiment", "latest_peer"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value, label, length=64):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value),
            f"{label}: expected lowercase {length}-character digest")
    require(len(set(value)) > 1, f"{label}: placeholder digest")


def nonempty(value, label):
    require(isinstance(value, str) and bool(value.strip()), f"{label}: missing text")


def file_identity(value, label, root):
    require(isinstance(value, dict), f"{label}: missing file identity")
    sha(value.get("sha256"), label)
    nonempty(value.get("path"), label + " path")
    size = value.get("bytes")
    require(type(size) is int and size > 0, f"{label}: invalid byte count")
    path = Path(value["path"])
    if not path.is_absolute():
        path = root / path
    require(path.is_file(), f"{label}: file unavailable: {path}")
    # Bound reads; the contract intentionally checks actual bytes, not names.
    with path.open("rb") as stream:
        descriptor = stream.fileno()
        initial = os.fstat(descriptor)
        digest = hashlib.sha256()
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
        final = os.fstat(descriptor)
    require((initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
            == (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns),
            f"{label}: file changed while hashing")
    require(final.st_size == size and digest.hexdigest() == value["sha256"],
            f"{label}: file bytes differ from frozen identity")
    return value["sha256"]


def exact_keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == expected,
            f"{label}: requires exactly {sorted(expected)}")


def validate(manifest, root):
    require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA,
            "unsupported identity schema")
    for field in ("campaign_id", "frozen_at", "coordination_ref"):
        nonempty(manifest.get(field), field)
    from datetime import datetime
    try:
        frozen = datetime.fromisoformat(manifest["frozen_at"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("frozen_at: invalid timestamp") from exc
    require(frozen.tzinfo is not None, "frozen_at: timezone required")
    host = manifest.get("hardware", {})
    require(isinstance(host, dict), "hardware: missing object")
    require(host.get("chip") == "Apple M5 Max" and type(host.get("physical_bytes")) is int
            and host.get("physical_bytes") == 128 * 2**30,
            "campaign requires M5 Max / 128 GiB")
    for field in ("os_version", "os_build", "metal_version", "compiler_version"):
        nonempty(host.get(field), "hardware " + field)
    engines = manifest.get("engines")
    exact_keys(engines, ARMS, "engines")
    for name, engine in engines.items():
        require(isinstance(engine, dict), f"{name}: missing engine")
        sha(engine.get("source_commit"), name + " source", 40)
        nonempty(engine.get("version_output"), name + " executable version output")
        file_identity(engine.get("binary"), name + " binary", root)
        file_identity(engine.get("build_receipt"), name + " build receipt", root)
        if name != "latest_peer":
            pin = engine.get("native", {})
            require(isinstance(pin, dict), name + ": missing native identity")
            sha(pin.get("source_commit"), name + " native source", 40)
            require(bool(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", str(pin.get("version", "")))),
                    name + ": missing published native version")
            require(pin.get("source") == "registry+https://github.com/rust-lang/crates.io-index",
                    name + ": native dependency is not the registry package")
            sha(pin.get("checksum"), name + " native checksum")
            require(file_identity(pin.get("crate"), name + " native crate", root) == pin["checksum"],
                    name + ": native archive disagrees with locked checksum")
            file_identity(engine.get("dependency_provenance_receipt"), name + " dependency proof", root)
    require(engines["experiment"].get("baseline_commit") == engines["stock"]["source_commit"],
            "experiment does not declare the exact stock baseline")
    peer = engines["latest_peer"]
    require(peer.get("resolved_upstream_commit") == peer["source_commit"],
            "latest-peer upstream resolution differs from built source")
    file_identity(peer.get("upstream_resolution_receipt"), "latest-peer resolution receipt", root)
    models = manifest.get("models")
    exact_keys(models, MODELS, "models")
    for family, model in models.items():
        require(isinstance(model, dict), family + ": missing model")
        for field in ("hf_repository", "quantization"):
            nonempty(model.get(field), family + " " + field)
        require(model.get("variant") == "vanilla", family + ": vanilla artifact required")
        sha(model.get("hf_revision"), family + " HF revision", 40)
        model_sha = file_identity(model.get("gguf"), family + " GGUF", root)
        file_identity(model.get("conversion_receipt"), family + " conversion receipt", root)
        file_identity(model.get("source_integrity_receipt"), family + " source integrity receipt", root)
        for field in ("tokenizer", "template", "rendered_tokens", "production_sampling"):
            file_identity(model.get(field), family + " " + field, root)
        bindings = model.get("arms")
        exact_keys(bindings, ARMS, family + " arms")
        for arm, binding in bindings.items():
            require(isinstance(binding, dict) and binding.get("model_sha256") == model_sha,
                    family + " " + arm + ": same-GGUF identity mismatch")
            for field in ("matched_settings", "capability_receipt"):
                file_identity(binding.get(field), family + " " + arm + " " + field, root)
            drafts = binding.get("drafts")
            require(isinstance(drafts, list), family + " " + arm + ": explicit draft list required")
            for draft in drafts:
                file_identity(draft, family + " " + arm + " draft", root)
        file_identity(bindings["latest_peer"].get("peer_best_settings"), family + " peer-best", root)
    # This is intentionally not a scientific certification or a live admission.
    return {"schema": SCHEMA, "campaign_id": manifest["campaign_id"],
            "offline_file_identities_verified": True, "campaign_qualified": False,
            "remaining_gates": ["source/build/provenance receipt review", "latest-peer resolution review",
                                "live runtime attestation", "memory/thermal/coordination admission",
                                "capability and four-slot QE", "paired four-model speed verdicts"]}


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        result = validate(load(args.manifest), args.manifest.resolve().parent)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print("identity validation failed: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
