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
        def make_file(name, payload):
            data = payload.encode() if isinstance(payload, str) else payload
            path = self.root / name
            path.write_bytes(data)
            return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        self.make_file = make_file
        stock = digest("stock")[:40]
        engines = {arm: {"source_commit": digest(arm)[:40], "binary": make_file(arm + ".bin", arm + " executable fixture"),
                         "version_output": arm + " fixture version", "build_receipt": f, "runtime_dependencies": [],
                         "dependency_provenance_receipt": f, "native": native} for arm in ARMS}
        engines["stock"]["source_commit"] = stock
        engines["experiment"]["baseline_commit"] = stock
        engines["latest_peer"]["resolved_upstream_commit"] = engines["latest_peer"]["source_commit"]
        engines["latest_peer"]["upstream_resolution_receipt"] = f
        models = {}
        rendered = {"schema": "hf2q.offline-rendered-inputs.v1", "model_inference_performed": False,
                    "runtime_qualified": False, "observations": [], "gcd_enabled": False, "overflow_policy_applied": False,
                    "source_commit": digest("render source")[:40], "source_worktree_clean": True}
        for field in ("capture_config_sha256", "capture_recipe_source_sha256", "http_handler_source_sha256", "renderer_source_sha256", "capture_binary_sha256"):
            rendered[field] = digest(field)
        for family in MODELS:
            model_file = make_file(family + ".gguf", family + " synthetic weights")
            tokenizer = make_file(family + ".tokenizer", family + " synthetic tokenizer")
            template = make_file(family + ".template", family + " synthetic template")
            token_file = make_file(family + "-cold.tokens.json", "[1,2,3]")
            prompt_file = make_file(family + "-cold.prompt.txt", family + " synthetic prompt")
            rendered["observations"].append({"model": family, "case": "cold", "declared_model_sha256": model_file["sha256"],
                "template_sha256": template["sha256"], "tokenizer_sha256": tokenizer["sha256"],
                "tokens_sha256": token_file["sha256"], "rendered_sha256": prompt_file["sha256"],
                "token_count": 3, "context_tokens": 100,
                "chat_template_source": "native_encoding" if family == "deepseek4" else "gguf_metadata"})
            bindings = {arm: {"model_sha256": model_file["sha256"], "drafts": [], "matched_settings": f,
                "capability_receipt": f, "resolved_production_sampling": f} for arm in ARMS}
            bindings["latest_peer"]["peer_best_settings"] = f
            models[family] = {"hf_repository": "fixture/" + family, "hf_revision": digest(family)[:40],
                "variant": "vanilla", "quantization": "synthetic", "gguf": model_file,
                "tokenizer": tokenizer, "template": template, "production_sampling": f,
                "conversion_receipt": f, "source_integrity_receipt": f, "arms": bindings}
        render_file = make_file("render-receipt.json", json.dumps(rendered))
        for model in models.values():
            model["rendered_tokens"] = render_file
        self.manifest = copy.deepcopy({"schema": SCHEMA, "campaign_id": "synthetic-control",
                                     "frozen_at": "2026-10-23T14:00:00Z", "coordination_ref": "fixture",
                                     "hardware": {"chip": "Apple M5 Max", "physical_bytes": 128 * 2**30,
                                                  "os_version": "fixture", "os_build": "fixture",
                                                  "metal_version": "fixture", "compiler_version": "fixture"},
                                     "engines": engines, "models": models})
        self.manifest["evidence_files"] = [f]
        self.manifest["comparators"] = {name: {"identity": name + " fixture",
            "status": "source_inspected", "format_route": "fixture only",
            "qualification_limit": "no runtime proof", "evidence_files": [f]}
            for name in ("historical", "basert", "kapitsa", "mlx_release", "mlx_main", "mlx_lm")}

    def reject(self, message):
        with self.assertRaisesRegex(ValueError, message):
            validate(self.manifest, self.root)

    def test_runtime_dependency_bytes_are_checked(self):
        dependency = self.make_file("runtime.dylib", "runtime fixture")
        self.manifest["engines"]["latest_peer"]["runtime_dependencies"] = [dependency]
        (self.root / "runtime.dylib").write_text("changed runtime fixture")
        self.reject("runtime dependency")

    def test_missing_runtime_dependency_inventory_rejects(self):
        del self.manifest["engines"]["latest_peer"]["runtime_dependencies"]
        self.reject("explicit runtime dependency")

    def test_supporting_evidence_bytes_are_checked(self):
        proof = self.make_file("help.txt", "capabilities fixture")
        self.manifest["evidence_files"] = [proof]
        (self.root / "help.txt").write_text("changed capabilities")
        self.reject("supporting evidence")

    def test_supplemental_comparator_disposition_is_required(self):
        del self.manifest["comparators"]["kapitsa"]
        self.reject("supplemental comparators")

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
        self.reject("requires exactly")

    def test_version_is_executable_identity_not_source_only(self):
        del self.manifest["engines"]["stock"]["version_output"]
        self.reject("executable version output")

    def mutate_render_receipt(self, mutate):
        path = self.root / "render-receipt.json"
        data = json.loads(path.read_text())
        mutate(data)
        pin = self.make_file(path.name, json.dumps(data))
        for model in self.manifest["models"].values():
            model["rendered_tokens"] = copy.deepcopy(pin)

    def test_receipt_binds_manifest_and_actual_file_observations(self):
        first = validate(self.manifest, self.root)
        self.manifest["coordination_ref"] = "different observation"
        second = validate(self.manifest, self.root)
        self.assertNotEqual(first["manifest_canonical_sha256"], second["manifest_canonical_sha256"])
        self.assertTrue(first["file_observations"])
        self.assertTrue(all("sha256" in row and "bytes" in row for row in first["file_observations"]))

    def test_duplicate_model_identity_rejects(self):
        self.manifest["models"]["qwen36"]["gguf"] = copy.deepcopy(self.manifest["models"]["qwen38"]["gguf"])
        for binding in self.manifest["models"]["qwen36"]["arms"].values():
            binding["model_sha256"] = self.manifest["models"]["qwen38"]["gguf"]["sha256"]
        self.reject("distinct GGUF")

    def test_wrong_render_model_binding_rejects_even_with_fresh_receipt_hash(self):
        self.mutate_render_receipt(lambda r: r["observations"][0].update(declared_model_sha256=digest("other model")))
        self.reject("model differs")

    def test_wrong_render_template_binding_rejects(self):
        self.mutate_render_receipt(lambda r: r["observations"][0].update(template_sha256=digest("other template")))
        self.reject("template differs")

    def test_changed_token_sidecar_rejects(self):
        (self.root / "qwen38-cold.tokens.json").write_text("[3,2,1]")
        self.reject("bytes differ")

    def test_undeclared_identical_binary_rejects(self):
        self.manifest["engines"]["experiment"]["binary"] = copy.deepcopy(self.manifest["engines"]["stock"]["binary"])
        self.reject("explicit stock_control")
        self.manifest["engines"]["experiment"]["identity_mode"] = "stock_control"
        self.assertFalse(validate(self.manifest, self.root)["campaign_qualified"])

    def test_unknown_arm_field_rejects(self):
        self.manifest["models"]["qwen38"]["arms"]["stock"]["peer_best_settings"] = self.file
        self.reject("requires exactly")

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
