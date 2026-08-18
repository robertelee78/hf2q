#!/usr/bin/env python3
"""Generate hf2q's independent Python-TUF interoperability corpus.

The private keys below are deterministic, public, test-only fixture material.
They MUST NOT be used for an hf2q release repository or any other production
purpose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from securesystemslib.signer import CryptoSigner
from tuf.api.metadata import (
    MetaFile,
    Metadata,
    Role,
    Root,
    Snapshot,
    TargetFile,
    Targets,
    Timestamp,
)
from tuf.api.serialization.json import JSONSerializer

EXPECTED_TOOLS = {
    "cffi": "2.1.1",
    "cryptography": "50.0.0",
    "pycparser": "3.0",
    "securesystemslib": "1.4.0",
    "tuf": "7.0.0",
    "urllib3": "2.7.0",
}
EXPECTED_PYTHON = "3.14.6"
EXPIRES = datetime(2999, 1, 1, tzinfo=timezone.utc)
SPEC_VERSION = "1.0.0"
TARGET_PATH = "channels/stable/aarch64-apple-darwin.json"
TARGET_BYTES = b'{"fixture":"python-tuf-v1","schema_version":1}\n'
SERIALIZER = JSONSerializer(compact=True, validate=True)
GENERATED_NAMES = (
    "1.root.json",
    "2.root.json",
    "timestamp.json",
    "2.snapshot.json",
    "2.targets.json",
    "PROVENANCE.json",
    "SHA256SUMS",
)


EXPECTED_KEY_IDS = {
    "old-a": "364eff02cceee862c2e7e7a8afeb3036d8f09802023c99277f3c9d799a9fad44",
    "old-b": "46f2d0a47d2ec6c0254b0454ea3c5395de4837392788527fd782d585d923e267",
    "new-a": "9d3d4e5375cacc11125397e0eab4fc3b14f23f7cfa0edcb82c2a3363e4a8aa7f",
    "new-b": "a8007901c02a96e7414dcb1e7694c7462913ff6e2526c12f9875dc3f6a508075",
}


def signer(seed_byte: int) -> CryptoSigner:
    private_key = Ed25519PrivateKey.from_private_bytes(bytes([seed_byte]) * 32)
    return CryptoSigner(private_key)


def sign(metadata: Metadata, signers: list[CryptoSigner]) -> bytes:
    for index, role_signer in enumerate(signers):
        metadata.sign(role_signer, append=index != 0)
    return metadata.to_bytes(SERIALIZER)


def role_bindings(signers: list[CryptoSigner]) -> dict[str, Role]:
    keyids = [role_signer.public_key.keyid for role_signer in signers]
    return {
        name: Role(keyids.copy(), threshold=2)
        for name in ("root", "snapshot", "targets", "timestamp")
    }


def root_metadata(version_number: int, signers: list[CryptoSigner]) -> Metadata:
    keys = {
        role_signer.public_key.keyid: role_signer.public_key
        for role_signer in signers
    }
    return Metadata(
        Root(
            version=version_number,
            spec_version=SPEC_VERSION,
            expires=EXPIRES,
            keys=keys,
            roles=role_bindings(signers),
            consistent_snapshot=True,
        )
    )


def metadata_bytes() -> tuple[dict[str, bytes], dict[str, CryptoSigner]]:
    named_signers = {
        "old-a": signer(0x31),
        "old-b": signer(0x32),
        "new-a": signer(0x41),
        "new-b": signer(0x42),
    }
    assert {
        name: role_signer.public_key.keyid
        for name, role_signer in named_signers.items()
    } == EXPECTED_KEY_IDS
    old_signers = [named_signers["old-a"], named_signers["old-b"]]
    new_signers = [named_signers["new-a"], named_signers["new-b"]]

    root_v1 = root_metadata(1, old_signers)
    root_v1_bytes = sign(root_v1, old_signers)

    root_v2 = root_metadata(2, new_signers)
    # TUF root rotation requires the new root to satisfy both the prior and
    # replacement root thresholds. Keep both sets of signatures in one
    # independently serialized envelope.
    root_v2_bytes = sign(root_v2, old_signers + new_signers)

    targets = Metadata(
        Targets(
            version=2,
            spec_version=SPEC_VERSION,
            expires=EXPIRES,
            targets={
                TARGET_PATH: TargetFile.from_data(
                    TARGET_PATH, TARGET_BYTES, ["sha256"]
                )
            },
        )
    )
    targets_bytes = sign(targets, new_signers)

    targets_pin = MetaFile.from_data(2, targets_bytes, ["sha256"])
    snapshot = Metadata(
        Snapshot(
            version=2,
            spec_version=SPEC_VERSION,
            expires=EXPIRES,
            meta={"targets.json": targets_pin},
        )
    )
    snapshot_bytes = sign(snapshot, new_signers)

    snapshot_pin = MetaFile.from_data(2, snapshot_bytes, ["sha256"])
    timestamp = Metadata(
        Timestamp(
            version=2,
            spec_version=SPEC_VERSION,
            expires=EXPIRES,
            snapshot_meta=snapshot_pin,
        )
    )
    timestamp_bytes = sign(timestamp, new_signers)

    assert root_v1.signed.get_root_verification_result(
        None, root_v1.signed_bytes, root_v1.signatures
    ).verified
    assert root_v2.signed.get_root_verification_result(
        root_v1.signed, root_v2.signed_bytes, root_v2.signatures
    ).verified
    root_v2.signed.verify_delegate(
        "timestamp", timestamp.signed_bytes, timestamp.signatures
    )
    root_v2.signed.verify_delegate(
        "snapshot", snapshot.signed_bytes, snapshot.signatures
    )
    root_v2.signed.verify_delegate(
        "targets", targets.signed_bytes, targets.signatures
    )
    targets_pin.verify_length_and_hashes(targets_bytes)
    snapshot_pin.verify_length_and_hashes(snapshot_bytes)

    return (
        {
            "1.root.json": root_v1_bytes,
            "2.root.json": root_v2_bytes,
            "timestamp.json": timestamp_bytes,
            "2.snapshot.json": snapshot_bytes,
            "2.targets.json": targets_bytes,
        },
        named_signers,
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output: Path) -> None:
    actual_tools = {name: version(name) for name in EXPECTED_TOOLS}
    if actual_tools != EXPECTED_TOOLS:
        raise SystemExit(
            f"exact fixture tools required: {EXPECTED_TOOLS}; found: {actual_tools}"
        )
    if platform.python_implementation() != "CPython" or platform.python_version() != EXPECTED_PYTHON:
        raise SystemExit(
            f"CPython {EXPECTED_PYTHON} required; found "
            f"{platform.python_implementation()} {platform.python_version()}"
        )

    output.mkdir(parents=True, exist_ok=True)
    files, named_signers = metadata_bytes()
    for name, data in files.items():
        (output / name).write_bytes(data)

    fixture_directory = Path(__file__).resolve().parent
    generator_bytes = Path(__file__).read_bytes()
    lock_bytes = (fixture_directory / "requirements.lock").read_bytes()
    provenance = {
        "consistent_snapshot": True,
        "expires": "2999-01-01T00:00:00Z",
        "fixture": "hf2q-python-tuf-v1",
        "generator_sha256": sha256(generator_bytes),
        "keys": {
            name: {
                "keyid": role_signer.public_key.keyid,
                "key": role_signer.public_key.to_dict(),
            }
            for name, role_signer in sorted(named_signers.items())
        },
        "metadata": {
            name: {"length": len(data), "sha256": sha256(data)}
            for name, data in sorted(files.items())
        },
        "packages": actual_tools,
        "private_keys": "deterministic public test fixtures; never production keys",
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "requirements_lock_sha256": sha256(lock_bytes),
        "schema_version": 1,
        "spec_version": SPEC_VERSION,
        "target_path": TARGET_PATH,
        "target_payload": {
            "length": len(TARGET_BYTES),
            "sha256": sha256(TARGET_BYTES),
        },
        "threshold": 2,
    }
    provenance_bytes = (
        json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    (output / "PROVENANCE.json").write_bytes(provenance_bytes)

    checksummed = files | {
        "PROVENANCE.json": provenance_bytes,
        "generate.py": generator_bytes,
        "requirements.lock": lock_bytes,
    }
    checksums = "".join(
        f"{sha256(data)}  {name}\n" for name, data in sorted(checksummed.items())
    ).encode()
    (output / "SHA256SUMS").write_bytes(checksums)


def check(expected: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="hf2q-python-tuf-") as temporary:
        generated = Path(temporary)
        generate(generated)
        expected_names = set(GENERATED_NAMES)
        actual_names = {path.name for path in generated.iterdir()}
        if actual_names != expected_names:
            raise SystemExit(
                f"generated inventory mismatch: expected {sorted(expected_names)}, "
                f"found {sorted(actual_names)}"
            )
        for name in GENERATED_NAMES:
            if (generated / name).read_bytes() != (expected / name).read_bytes():
                raise SystemExit(f"retained fixture drifted: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="directory for generated fixture files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in a temporary directory and compare exact bytes",
    )
    arguments = parser.parse_args()
    if arguments.check:
        check(arguments.output)
    else:
        generate(arguments.output)


if __name__ == "__main__":
    main()
