"""Actual capture preflight service/recipe boundaries; no process or client."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from q2_capture_packet import validate_capture_approval
from q2_capture_preflight import quiet_inventory, standalone_capture_argv, default_resolution
from q2_certify import BoundReader
from qwen38_n1_verdict import Invalid
from test_q2_capture_packet_controls import positive

BINARY = {"path": "/private/tmp/MOCK-opencode", "sha256": "a" * 64}
SERVICE = {"pid": 36057, "ppid": 1, "start": "Mon Oct 20 10:00:00 2026",
           "argv": [BINARY["path"], "serve", "--service"], "binary": BINARY}
RAW = b"36057 1 Mon Oct 20 10:00:00 2026 /private/tmp/MOCK-opencode /private/tmp/MOCK-opencode serve --service\n"


class CapturePreflightControls(unittest.TestCase):
    def test_exact_same_binary_service_is_logged_not_killed_or_general_allowlist(self):
        value = positive(); value["client_binary"] = BINARY; value["existing_client_service"] = SERVICE
        self.assertIs(validate_capture_approval(value), value)
        result = quiet_inventory(RAW, SERVICE, BINARY)
        self.assertEqual(result["allowed_preexisting_service"]["pid"], 36057)
        self.assertEqual(result["raw_ps"], RAW.decode())
    def test_pid_reuse_start_argv_parent_or_same_client_binary_drift_refuses(self):
        for raw in (RAW.replace(b"10:00:00", b"10:00:01"), RAW.replace(b"36057 1", b"36057 2"),
                    RAW.replace(b" --service", b" --other"), RAW.replace(b"MOCK-opencode", b"other-client", 1)):
            with self.subTest(raw=raw), self.assertRaises(Invalid): quiet_inventory(raw, SERVICE, BINARY)
        changed = dict(SERVICE, binary={"path": BINARY["path"], "sha256": "b" * 64})
        with self.assertRaises(Invalid): quiet_inventory(RAW, changed, BINARY)
    def test_other_opencode_helper_or_disappeared_bound_service_refuses(self):
        for raw in (RAW + b"36058 1 Mon Oct 20 10:00:00 2026 /usr/bin/opencode /usr/bin/opencode run\n",
                    RAW + b"36059 1 Mon Oct 20 10:00:00 2026 /tmp/helper-binary /tmp/helper-binary\n", b""):
            with self.assertRaises(Invalid): quiet_inventory(raw, SERVICE, BINARY)
        with self.assertRaises(Invalid): quiet_inventory(RAW.replace(b"MOCK-opencode", b"opencode"), None, BINARY)
    def test_actual_launch_prefix_requires_standalone_not_service(self):
        argv = ["/usr/bin/sandbox-exec", "-f", "/tmp/MOCK.sb", BINARY["path"], "run", "--standalone", "--format", "json"]
        self.assertEqual(standalone_capture_argv(argv, argv[0], argv[2], argv[3]), argv)
        for bad in (argv[:5] + argv[6:], argv[:4] + ["serve", "--service"]):
            with self.assertRaises(Invalid): standalone_capture_argv(bad, argv[0], argv[2], argv[3])
    def recipe(self, source):
        return {"schema": "hf2q.q2.hf-default-resolution-recipe.v1", "family": "qwen38",
                "server_source": "a" * 40, "source_files": {str(source): hashlib.sha256(source.read_bytes()).hexdigest()},
                "resolution_scope": "source_rules_only_no_actual_request_or_server_resolution",
                "effective_sampling_qualified": False, "status": "PendingActualServerJoin"}
    def test_source_recipe_is_hashed_rules_only_not_effective_sampling(self):
        with tempfile.TemporaryDirectory(prefix="q2-recipe-MOCK-") as d:
            source = Path(d).resolve() / "MOCK-source.rs"; source.write_text("// source fixture, not a runtime\n")
            recipe = self.recipe(source); reader = BoundReader()
            disposition = default_resolution(recipe, "qwen38", reader); reader.verify()
            self.assertIs(disposition["effective_sampling_qualified"], False)
            self.assertEqual(disposition["status"], "PendingActualServerJoin")
            source.write_text("// changed source\n")
            with self.assertRaises(Invalid): default_resolution(recipe, "qwen38", BoundReader())
    def test_no_fabricated_resolved_values_and_legacy_receipt_stays_supported(self):
        with tempfile.TemporaryDirectory(prefix="q2-recipe-MOCK-") as d:
            source = Path(d).resolve() / "MOCK-source.rs"; source.write_text("// source\n"); recipe = self.recipe(source)
            for key, value in (("effective_sampling_qualified", True), ("status", "Qualified"),
                               ("per_phase", [{"phase": i, "temperature": 0} for i in range(7)])):
                bad = copy.deepcopy(recipe); bad[key] = value
                with self.assertRaises(Invalid): default_resolution(bad, "qwen38", BoundReader())
            legacy = {"schema": "hf2q.q2.effective-hf-defaults.v1", "family": "qwen38", "server_source": "a" * 40,
                      "source_files": recipe["source_files"], "per_phase": [{"phase": i} for i in range(7)]}
            self.assertIs(default_resolution(legacy, "qwen38", BoundReader())["effective_sampling_qualified"], False)


if __name__ == "__main__": unittest.main()
