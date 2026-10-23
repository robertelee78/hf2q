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
import tarfile
import tomllib

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


def _file_identity(value, label, root):
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



def native_archive(pin, label, root):
    path = Path(pin["crate"]["path"])
    if not path.is_absolute():
        path = root / path
    prefix = "mlx-native-" + pin["version"] + "/"
    with tarfile.open(path, "r:gz") as archive:
        def member(name):
            entries = [entry for entry in archive if entry.name == prefix + name]
            require(len(entries) == 1 and entries[0].isfile() and entries[0].size <= 1024 * 1024,
                    label + ": missing, duplicate or invalid native archive metadata")
            with archive.extractfile(entries[0]) as stream:
                return stream.read().decode("utf-8")
        vcs = json.loads(member(".cargo_vcs_info.json"))
        git = vcs.get("git", {})
        require(git.get("sha1") == pin["source_commit"] and not git.get("dirty", False),
                label + ": native archive source differs or is dirty")
        package = tomllib.loads(member("Cargo.toml")).get("package", {})
        require(package.get("name") == "mlx-native" and package.get("version") == pin["version"],
                label + ": native archive package/version differs")


def exact_keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == expected,
            f"{label}: requires exactly {sorted(expected)}")


def validate(manifest, root):
    observations = []

    def file_identity(value, label, checked_root):
        actual = _file_identity(value, label, checked_root)
        path = Path(value["path"])
        if not path.is_absolute():
            path = checked_root / path
        observations.append({"label": label, "path": str(path.resolve()),
                             "bytes": value["bytes"], "sha256": actual})
        return actual

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
        dependencies = engine.get("runtime_dependencies")
        require(isinstance(dependencies, list), name + ": explicit runtime dependency list required")
        for index, dependency in enumerate(dependencies):
            file_identity(dependency, name + " runtime dependency " + str(index), root)
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
            native_archive(pin, name, root)
            file_identity(engine.get("dependency_provenance_receipt"), name + " dependency proof", root)
    require(engines["experiment"].get("baseline_commit") == engines["stock"]["source_commit"],
            "experiment does not declare the exact stock baseline")
    peer = engines["latest_peer"]
    require(peer.get("resolved_upstream_commit") == peer["source_commit"],
            "latest-peer upstream resolution differs from built source")
    file_identity(peer.get("upstream_resolution_receipt"), "latest-peer resolution receipt", root)
    binary_hashes = {arm: engine["binary"]["sha256"] for arm, engine in engines.items()}
    require(binary_hashes["latest_peer"] not in (binary_hashes["stock"], binary_hashes["experiment"]),
            "latest-peer executable must differ from hf2q")
    require(binary_hashes["stock"] != binary_hashes["experiment"]
            or engines["experiment"].get("identity_mode") == "stock_control",
            "identical hf2q binaries require explicit stock_control identity_mode")
    evidence_files = manifest.get("evidence_files")
    require(isinstance(evidence_files, list), "explicit supporting evidence list required")
    for index, evidence in enumerate(evidence_files):
        file_identity(evidence, "supporting evidence " + str(index), root)
    comparators = manifest.get("comparators")
    exact_keys(comparators, {"historical", "basert", "kapitsa", "mlx_release", "mlx_main", "mlx_lm"},
               "supplemental comparators")
    for name, comparator in comparators.items():
        require(isinstance(comparator, dict), name + ": missing comparator disposition")
        for field in ("identity", "status", "format_route", "qualification_limit"):
            nonempty(comparator.get(field), name + " " + field)
        receipts = comparator.get("evidence_files")
        require(isinstance(receipts, list) and receipts, name + ": comparator evidence required")
        for index, receipt in enumerate(receipts):
            file_identity(receipt, name + " comparator evidence " + str(index), root)
    models = manifest.get("models")
    exact_keys(models, MODELS, "models")
    model_hashes = set()
    for family, model in models.items():
        require(isinstance(model, dict), family + ": missing model")
        for field in ("hf_repository", "quantization"):
            nonempty(model.get(field), family + " " + field)
        require(model.get("variant") == "vanilla", family + ": vanilla artifact required")
        sha(model.get("hf_revision"), family + " HF revision", 40)
        model_sha = file_identity(model.get("gguf"), family + " GGUF", root)
        require(model_sha not in model_hashes, "different models must have distinct GGUF identities")
        model_hashes.add(model_sha)
        file_identity(model.get("conversion_receipt"), family + " conversion receipt", root)
        file_identity(model.get("source_integrity_receipt"), family + " source integrity receipt", root)
        for field in ("tokenizer", "template", "rendered_tokens", "production_sampling"):
            file_identity(model.get(field), family + " " + field, root)
        role_hashes = [model[field]["sha256"] for field in ("gguf", "tokenizer", "template", "rendered_tokens")]
        require(len(set(role_hashes)) == len(role_hashes), family + ": GGUF/tokenizer/template/render receipt must be distinct files")
        bindings = model.get("arms")
        exact_keys(bindings, ARMS, family + " arms")
        for arm, binding in bindings.items():
            required_binding = {"model_sha256", "matched_settings", "capability_receipt", "drafts", "resolved_production_sampling"}
            if arm == "latest_peer":
                required_binding.add("peer_best_settings")
            exact_keys(binding, required_binding, family + " " + arm + " binding")
            require(isinstance(binding, dict) and binding.get("model_sha256") == model_sha,
                    family + " " + arm + ": same-GGUF identity mismatch")
            for field in ("matched_settings", "capability_receipt", "resolved_production_sampling"):
                file_identity(binding.get(field), family + " " + arm + " " + field, root)
            drafts = binding.get("drafts")
            require(isinstance(drafts, list), family + " " + arm + ": explicit draft list required")
            for draft in drafts:
                file_identity(draft, family + " " + arm + " draft", root)
        file_identity(bindings["latest_peer"].get("peer_best_settings"), family + " peer-best", root)
    require(len({m["rendered_tokens"]["sha256"] for m in models.values()}) == 1,
            "all four models must bind one coherent rendered-input receipt")
    _validate_render_receipts(models, root, file_identity, {f["sha256"] for f in evidence_files})
    # This is intentionally not a scientific certification or a live admission.
    return {"schema": SCHEMA, "campaign_id": manifest["campaign_id"],
            "offline_file_identities_verified": True, "campaign_qualified": False,
            "validator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "manifest_canonical_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
            "file_observations": observations,
            "remaining_gates": ["source/build/provenance receipt review", "latest-peer resolution review",
                                "live runtime attestation", "memory/thermal/coordination admission",
                                "capability and four-slot QE", "paired four-model speed verdicts"]}



def _validate_render_receipts(models, root, check_file, request_hashes):
    for family, model in models.items():
        path = Path(model["rendered_tokens"]["path"])
        if not path.is_absolute():
            path = root / path
        receipt = load(path)
        require(isinstance(receipt, dict) and receipt.get("schema") == "hf2q.offline-rendered-inputs.v1",
                family + ": unsupported rendered-input receipt")
        require(receipt.get("model_inference_performed") is False and receipt.get("runtime_qualified") is False,
                family + ": invalid offline render qualification flags")
        require(receipt.get("gcd_enabled") is False and receipt.get("overflow_policy_applied") is False,
                family + ": unsupported render policy")
        sha(receipt.get("source_commit"), family + " render source commit", 40)
        require(receipt.get("source_worktree_clean") is True, family + ": render source tree was not clean")
        for field in ("capture_config_sha256", "capture_recipe_source_sha256", "http_handler_source_sha256",
                      "renderer_source_sha256", "deepseek_loader_source_sha256", "capture_binary_sha256"):
            sha(receipt.get(field), family + " " + field)
        rows = receipt.get("observations")
        require(isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows),
                family + ": missing rendered observations")
        require({row.get("model") for row in rows} == MODELS,
                family + ": rendered receipt requires all four models")
        seen = set()
        for row in rows:
            if row.get("model") != family:
                continue
            case = row.get("case")
            require(isinstance(case, str) and re.fullmatch(r"[A-Za-z0-9_-]+", case)
                    and case not in seen, family + ": invalid/duplicate rendered case")
            seen.add(case)
            sha(row.get("request_file_sha256"), family + " request file")
            require(row["request_file_sha256"] in request_hashes,
                    family + ": rendered request file not bound by supporting evidence")
            require(row.get("declared_model_sha256") == model["gguf"]["sha256"],
                    family + ": rendered receipt model differs from verified GGUF")
            for field in ("template", "tokenizer"):
                require(row.get(field + "_sha256") == model[field]["sha256"],
                        family + ": rendered " + field + " differs from frozen identity")
            expected_source = "native_encoding" if family == "deepseek4" else "gguf_metadata"
            require(row.get("chat_template_source") == expected_source,
                    family + ": wrong serving template selection")
            for suffix, field in (("tokens.json", "tokens_sha256"), ("prompt.txt", "rendered_sha256")):
                sidecar = path.parent / (family + "-" + case + "." + suffix)
                sha(row.get(field), family + " rendered sidecar")
                require(sidecar.is_file(), family + ": rendered sidecar unavailable")
                check_file({"path": str(sidecar), "bytes": sidecar.stat().st_size,
                            "sha256": row[field]}, family + " " + case + " " + suffix, root)
            tokens = load(path.parent / (family + "-" + case + ".tokens.json"))
            require(isinstance(tokens, list) and tokens and all(type(t) is int and 0 <= t < 2**32 for t in tokens),
                    family + ": invalid rendered token IDs")
            require(type(row.get("token_count")) is int and row["token_count"] == len(tokens) and type(row.get("context_tokens")) is int
                    and 0 < len(tokens) < row["context_tokens"], family + ": rendered context/count mismatch")
        required_cases = {"cold"} | {"tool_round_" + str(i) for i in range(1, 7)}
        require(required_cases <= seen, family + ": cold plus six tool rounds required")


def parse_manifest(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)


def load(path):
    return parse_manifest(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        raw = args.manifest.read_bytes()
        result = validate(parse_manifest(raw), args.manifest.resolve().parent)
        result["manifest_file_sha256"] = hashlib.sha256(raw).hexdigest()
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError, KeyError, OSError, tarfile.TarError) as exc:
        print("identity validation failed: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
