"""Adversarial file and cross-arm identity controls; no GPU or model fixture."""
import copy
import hashlib
import json
import io
import tarfile
from pathlib import Path
import tempfile
import unittest

from identity import ARMS, MODELS, SCHEMA, load, validate


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class IdentityContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "fixture").write_bytes(b"synthetic identity fixture, not model weights")
        data = (self.root / "fixture").read_bytes()
        f = {"path": "fixture", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.file = f
        native_commit = digest("native")[:40]
        archive = self.root / "native.crate"
        with tarfile.open(archive, "w:gz") as tar:
            contents = {".cargo_vcs_info.json": json.dumps({"git": {"sha1": native_commit}}),
                        "Cargo.toml": '[package]\nname="mlx-native"\nversion="0.15.1"\n'}
            for name, content in contents.items():
                data = content.encode()
                info = tarfile.TarInfo("mlx-native-0.15.1/" + name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        crate = {"path": "native.crate", "bytes": archive.stat().st_size,
                 "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        native = {"source_commit": native_commit, "version": "0.15.1",
                  "source": "registry+https://github.com/rust-lang/crates.io-index",
                  "checksum": crate["sha256"], "crate": crate}
        stock = digest("stock")[:40]
        engines = {arm: {"source_commit": digest(arm)[:40], "binary": f,
                         "version_output": arm + " fixture version", "build_receipt": f,
                         "dependency_provenance_receipt": f, "native": native} for arm in ARMS}
        engines["stock"]["source_commit"] = stock
        engines["experiment"]["baseline_commit"] = stock
        engines["latest_peer"]["resolved_upstream_commit"] = engines["latest_peer"]["source_commit"]
        engines["latest_peer"]["upstream_resolution_receipt"] = f
        models = {}
        for family in MODELS:
            models[family] = {"hf_repository": "fixture/" + family, "hf_revision": digest(family)[:40],
                              "variant": "vanilla", "quantization": "synthetic",
                              **{key: f for key in ("gguf", "conversion_receipt", "source_integrity_receipt",
                                                   "tokenizer", "template", "rendered_tokens", "production_sampling")},
                              "arms": {arm: {"model_sha256": f["sha256"], "drafts": [],
                                               "matched_settings": f, "capability_receipt": f,
                                               "peer_best_settings": f} for arm in ARMS}}
        self.manifest = copy.deepcopy({"schema": SCHEMA, "campaign_id": "synthetic-control",
                                     "frozen_at": "2026-10-23T14:00:00Z", "coordination_ref": "fixture",
                                     "hardware": {"chip": "Apple M5 Max", "physical_bytes": 128 * 2**30,
                                                  "os_version": "fixture", "os_build": "fixture",
                                                  "metal_version": "fixture", "compiler_version": "fixture"},
                                     "engines": engines, "models": models})

    def reject(self, message):
        with self.assertRaisesRegex(ValueError, message):
            validate(self.manifest, self.root)

    def test_hashes_are_not_a_campaign_verdict(self):
        result = validate(self.manifest, self.root)
        self.assertTrue(result["offline_file_identities_verified"])
        self.assertFalse(result["campaign_qualified"])
        self.assertIn("live runtime attestation", result["remaining_gates"])

    def test_changed_binary_bytes_reject_even_with_same_name_and_size(self):
        data = (self.root / "fixture").read_bytes()
        (self.root / "fixture").write_bytes(b"X" + data[1:])
        self.reject("bytes differ")

    def test_unavailable_binary_rejects(self):
        self.manifest["engines"]["latest_peer"]["binary"]["path"] = "missing"
        self.reject("file unavailable")

    def test_wrong_baseline_rejects(self):
        self.manifest["engines"]["experiment"]["baseline_commit"] = digest("wrong")[:40]
        self.reject("exact stock baseline")

    def test_peer_resolution_must_bind_built_source(self):
        self.manifest["engines"]["latest_peer"]["resolved_upstream_commit"] = digest("old")[:40]
        self.reject("resolution differs")

    def test_native_archive_must_match_registry_checksum(self):
        self.manifest["engines"]["experiment"]["native"]["checksum"] = digest("wrong crate")
        self.reject("archive disagrees")

    def test_local_native_patch_is_not_release_identity(self):
        self.manifest["engines"]["experiment"]["native"]["source"] = "path"
        self.reject("not the registry")

    def test_native_source_must_match_downloaded_package(self):
        self.manifest["engines"]["experiment"]["native"]["source_commit"] = digest("other source")[:40]
        self.reject("archive source differs")

    def test_native_version_must_match_downloaded_package(self):
        self.manifest["engines"]["experiment"]["native"]["version"] = "0.15.2"
        self.reject("invalid native archive metadata")

    def test_missing_fourth_model_rejects(self):
        del self.manifest["models"]["gemma4"]
        self.reject("models: requires exactly")

    def test_same_gguf_must_reach_all_arms(self):
        self.manifest["models"]["qwen38"]["arms"]["latest_peer"]["model_sha256"] = digest("other model")
        self.reject("same-GGUF identity mismatch")

    def test_abliterated_target_cannot_replace_vanilla(self):
        self.manifest["models"]["qwen38"]["variant"] = "abliterated"
        self.reject("vanilla artifact required")

    def test_peer_best_settings_are_mandatory(self):
        del self.manifest["models"]["deepseek4"]["arms"]["latest_peer"]["peer_best_settings"]
        self.reject("peer-best: missing file")

    def test_version_is_executable_identity_not_source_only(self):
        del self.manifest["engines"]["stock"]["version_output"]
        self.reject("executable version output")

    def test_duplicate_json_keys_reject(self):
        path = self.root / "manifest.json"
        path.write_text('{"schema":"first","schema":"second"}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            load(path)

    def test_malformed_shapes_fail_closed(self):
        for field in ("hardware", "engines", "models"):
            with self.subTest(field=field):
                original = self.manifest[field]
                self.manifest[field] = "invalid"
                self.reject("requires exactly|missing object")
                self.manifest[field] = original


if __name__ == "__main__":
    unittest.main()
