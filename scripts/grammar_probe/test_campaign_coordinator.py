"""Coordinator tests use temporary records and only explicitly owned tiny children."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import campaign_coordinator as cc


class CampaignCoordinator(unittest.TestCase):
    def test_legacy_pin_and_inventory_validation_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "baseline_run.py"; runner.write_text("# pinned old source\n")
            path = root / "results.jsonl"
            rows = [{"arm": arm, "prompt_id": "p1", "rep": 1, "budget": budget,
                     "config_sha256": "recorded-config", "content": "answer"}
                    for arm in ("BASE", "W1") for budget in (800, 1600)]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            before = runner.read_bytes(), path.read_bytes()
            cc.verify_pins([{"path": str(runner), "sha256": cc.hash_file(runner)}])
            self.assertEqual(cc.verify_results(path, 4, "recorded-config"), cc.hash_file(path))
            for count, cfg in ((5, "recorded-config"), (4, "different")):
                with self.assertRaises(ValueError): cc.verify_results(path, count, cfg)
            with self.assertRaisesRegex(ValueError, "pinned artifact changed"):
                cc.verify_pins([{"path": str(runner), "sha256": "0" * 64}])
            self.assertEqual((runner.read_bytes(), path.read_bytes()), before)

    def test_recycled_server_identity_cannot_be_signaled(self):
        expected = {"pid": 42, "lstart": "original", "command": "hf2q serve", "port": 8081}
        with patch.object(cc, "process_identity", return_value={"pid": 42, "lstart": "different", "command": "hf2q serve"}), patch.object(cc.os, "kill") as kill:
            with self.assertRaisesRegex(ValueError, "identity changed"):
                cc.stop_recorded_server(expected)
            kill.assert_not_called()

    def test_cleanup_terminates_only_the_owned_child(self):
        owned = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"])
        unrelated = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"])
        try:
            cc.stop_owned(owned)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            cc.stop_owned(owned); cc.stop_owned(unrelated)

    def test_child_failure_prevents_completion_and_records_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = cc.Coordinator({"output_dir": tmp})
            command = {"name": "failing_stage", "argv": [sys.executable, "-B", "-c", "print('diagnostic'); raise SystemExit(7)"]}
            with patch.object(cc, "guard_host"):
                with self.assertRaisesRegex(ValueError, "exited 7"):
                    coordinator.run_command(command)
            self.assertEqual(coordinator.state["completed_commands"], [])
            self.assertIsNone(coordinator.active_child)
            self.assertIn("diagnostic", (Path(tmp) / "failing_stage.log").read_text())

    def test_no_running_foreign_work_is_admitted(self):
        found = subprocess.CompletedProcess([], 0, "789\n", "")
        with patch.object(cc.subprocess, "run", return_value=found), patch.object(cc, "descendants_include", return_value=False):
            with self.assertRaisesRegex(ValueError, "host contention"):
                cc.guard_host(allowed_pids=[123])


if __name__ == "__main__":
    unittest.main()
