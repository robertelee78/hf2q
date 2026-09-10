#!/usr/bin/env python3
"""Offline managed-runtime regression tests; never launch a model or server.

Run with ``python3 scripts/grammar_probe/test_managed_runtime.py``. An explicit
HF2Q_MANAGED_RUNTIME_SOURCE path can select an implementation in another
worktree during integration. All artifacts are temporary synthetic files;
Popen and HTTP requests are mocked. The port-ownership test only binds an
ephemeral loopback socket and never contacts or stops an existing listener.
"""

from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


sys.dont_write_bytecode = True
SOURCE = Path(os.environ.get(
    "HF2Q_MANAGED_RUNTIME_SOURCE",
    Path(__file__).with_name("managed_runtime.py"),
)).resolve(strict=True)
SPEC = importlib.util.spec_from_file_location("managed_runtime_under_test", SOURCE)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


def snapshot(pid=12345):
    return {
        "schema_version": "hf2q.measurement-snapshot.v1",
        "process_pid": pid,
        "model_id": "synthetic-model",
        "architecture": "deepseek4",
        "engine_generation": 1,
        "tokenizer_sha256": "a" * 64,
        "template_sha256": "b" * 64,
        "admission": {"queue_capacity": 32, "max_concurrent_requests": 0,
                      "request_timeout_seconds": 0},
        "sampling_defaults": {
            "repetition_penalty": 1.0,
            "thinking_token_budget": 1024,
            "tool_thinking_token_budget": None,
            "overflow_policy": "Error",
        },
        "active_controls": {
            "glp": {"active": False, "alpha_override_bits": None},
            "grammar": {"active": False, "kind": None,
                        "sha256": None, "locked": False},
            "dwq_overlay": False,
            "vision_projector": False,
        },
        "engine_config": {
            "queue_capacity": 32,
            "scheduler": {"mode": "serial_fifo"},
            "requested_context_tokens": None,
            "kv_persist_enabled": False,
            "glp_active": False,
            "glp_alpha_bits": None,
        },
    }


class ManagedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hf2q-managed-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.binary = self.directory / "synthetic-hf2q"
        self.model = self.directory / "synthetic.gguf"
        self.binary.write_bytes(b"synthetic binary; never executed")
        self.model.write_bytes(b"synthetic model; never loaded")
        self.manifest = self.directory / "run.json"
        self.child = Mock(pid=12345, returncode=None)
        self.child.poll.return_value = None
        self.child.wait.return_value = 0
        self.argv = ["managed_runtime.py", "--binary", str(self.binary),
                     "--model", str(self.model), "--manifest", str(self.manifest)]

    def snapshot(self, pid=12345):
        value = snapshot(pid)
        value["model_locator_sha256"] = runtime.hashlib.sha256(
            str(self.model.resolve()).encode()).hexdigest()
        return value

    def invoke(self, responses, *, after_sleep=None, free_port=True):
        """Execute the real lifecycle with an owned mock child and mock HTTP."""
        handlers = {}

        def register(signum, handler):
            handlers[signum] = handler

        def sleep(_seconds):
            if after_sleep:
                after_sleep(handlers)
            else:
                handlers[signal.SIGTERM](signal.SIGTERM, None)

        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", self.argv))
            stack.enter_context(patch.object(runtime.signal, "signal", register))
            stack.enter_context(patch.object(runtime.time, "sleep", sleep))
            stack.enter_context(patch.object(runtime.time, "monotonic", return_value=0))
            stack.enter_context(patch.object(runtime, "child_environment", return_value={}))
            stack.enter_context(patch.object(sys, "stdout", io.StringIO()))
            if free_port:
                stack.enter_context(patch.object(runtime, "require_free_port"))
            self.spawn = stack.enter_context(patch.object(
                runtime.subprocess, "Popen", return_value=self.child))
            self.fetch = stack.enter_context(patch.object(
                runtime, "fetch_snapshot", side_effect=responses))
            runtime.main()

    def read_manifest(self):
        return json.loads(self.manifest.read_text())

    def assert_owned_child_stopped(self):
        self.child.terminate.assert_called_once_with()
        self.child.wait.assert_called_once_with(timeout=30)
        self.child.kill.assert_not_called()

    def test_occupied_port_does_not_spawn_or_stop_any_process(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            self.argv += ["--port", str(port)]
            with self.assertRaises(OSError):
                self.invoke([], free_port=False)
            self.assertEqual(listener.getsockname()[1], port)
            self.assertGreaterEqual(listener.fileno(), 0)
        self.spawn.assert_not_called()
        self.child.terminate.assert_not_called()
        self.child.kill.assert_not_called()
        self.assertFalse(self.manifest.exists())

    def test_changed_artifact_invalidates_before_next_snapshot(self):
        def change_artifact(_handlers):
            self.model.write_bytes(b"changed synthetic model")

        with self.assertRaisesRegex(ValueError, "model changed after verification"):
            self.invoke([self.snapshot()], after_sleep=change_artifact)
        self.assertEqual(self.fetch.call_count, 1)
        self.assertEqual(self.read_manifest()["state"], "invalid")
        self.assert_owned_child_stopped()

    def test_wrong_listener_pid_invalidates_and_stops_only_owned_child(self):
        with patch.object(os, "kill") as unrelated_kill:
            with self.assertRaisesRegex(ValueError, "not the owned server process"):
                self.invoke([self.snapshot(pid=54321)])
            unrelated_kill.assert_not_called()
        self.assertEqual(self.read_manifest()["state"], "invalid")
        self.assert_owned_child_stopped()

    def test_full_snapshot_drift_invalidates_including_unprojected_fields(self):
        mutations = [
            (("engine_generation",), 2),
            (("model_id",), "another-model"),
            (("architecture",), "qwen35"),
            (("tokenizer_sha256",), "c" * 64),
            (("template_sha256",), "d" * 64),
            (("sampling_defaults", "thinking_token_budget"), 2048),
            (("active_controls", "glp", "active"), True),
            (("active_controls", "glp", "alpha_override_bits"), 1065353216),
            (("active_controls", "grammar", "locked"), True),
            (("active_controls", "grammar", "sha256"), "e" * 64),
            (("active_controls", "dwq_overlay"), True),
            (("active_controls", "vision_projector"), True),
            (("engine_config", "queue_capacity"), 16),
            (("engine_config", "scheduler"), {"mode": "slot_aware", "max_slots": 2}),
            (("engine_config", "kv_persist_enabled"), True),
            (("admission", "max_concurrent_requests"), 2),
            (("admission", "request_timeout_seconds"), 60),
            (("future_snapshot_field",), {"value": "changed"}),
        ]
        for index, (keys, value) in enumerate(mutations):
            with self.subTest(field=".".join(keys)):
                self.manifest = self.directory / f"drift-{index}.json"
                self.argv[self.argv.index("--manifest") + 1] = str(self.manifest)
                self.child.reset_mock()
                changed = deepcopy(self.snapshot())
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, "changed|differs|unbound"):
                    self.invoke([self.snapshot(), changed], after_sleep=lambda _: None)
                self.assertEqual(self.read_manifest()["state"], "invalid")
                self.assert_owned_child_stopped()

    def test_endpoint_loss_after_binding_invalidates(self):
        with self.assertRaisesRegex(ValueError, "endpoint became unavailable"):
            self.invoke([self.snapshot(), OSError("synthetic disconnect")],
                        after_sleep=lambda _: None)
        self.assertEqual(self.read_manifest()["state"], "invalid")
        self.assert_owned_child_stopped()

    def test_startup_network_retry_then_orderly_stop(self):
        def stop_when_bound(handlers):
            if self.read_manifest()["state"] == "running":
                handlers[signal.SIGINT](signal.SIGINT, None)

        self.invoke([OSError("not ready"), self.snapshot()], after_sleep=stop_when_bound)
        self.assertEqual(self.fetch.call_count, 2)
        manifest = self.read_manifest()
        self.assertEqual(manifest["state"], "stopped")
        self.assertIsNone(manifest["runtime_identity"]["source_commit"])
        self.assertEqual(manifest["process_pid"], self.child.pid)
        self.assertEqual(manifest["runtime_snapshot"], self.snapshot())
        self.assert_owned_child_stopped()

    def test_orderly_stop_records_exact_artifact_hashes(self):
        self.invoke([self.snapshot()])
        manifest = self.read_manifest()
        self.assertEqual(manifest["state"], "stopped")
        for artifact in manifest["artifacts"]:
            sealed = runtime.seal(artifact["path"], artifact["role"])
            self.assertEqual(artifact, sealed)
        self.assertEqual(manifest["runtime_snapshot_sha256"],
                         runtime.canonical_hash(self.snapshot()))
        self.assert_owned_child_stopped()

    def test_launch_uses_fresh_state_root_instead_of_inherited_setup(self):
        self.invoke([self.snapshot()])
        command = self.spawn.call_args.args[0]
        self.assertIn("--state-root", command)
        state_root = Path(command[command.index("--state-root") + 1])
        self.assertTrue(state_root.is_absolute())
        self.assertTrue(state_root.is_dir())
        self.assertEqual(state_root.parent, self.directory.resolve())
        self.assertFalse((state_root / "config.toml").exists())
        self.assertEqual(self.read_manifest()["command"], command)

    def test_initial_configuration_mismatch_is_rejected_before_binding(self):
        mutations = [
            (("model_locator_sha256",), "f" * 64, "loaded model does not match"),
            (("active_controls", "glp", "active"), True, "GLP state differs"),
            (("active_controls", "glp", "alpha_override_bits"), 0, "GLP dose differs"),
            (("active_controls", "grammar", "active"), True, "grammar differs"),
            (("active_controls", "grammar", "kind"), "gbnf", "grammar kind differs"),
            (("active_controls", "grammar", "locked"), True, "schema lock differs"),
            (("active_controls", "dwq_overlay"), True, "unbound weight overlay"),
            (("active_controls", "vision_projector"), True, "unbound vision projector"),
        ]
        for index, (keys, value, error) in enumerate(mutations):
            with self.subTest(field=".".join(keys)):
                self.manifest = self.directory / f"mismatch-{index}.json"
                self.argv[self.argv.index("--manifest") + 1] = str(self.manifest)
                self.child.reset_mock()
                changed = self.snapshot()
                target = changed
                for key in keys[:-1]:
                    target = target[key]
                target[keys[-1]] = value
                with self.assertRaisesRegex(ValueError, error):
                    self.invoke([changed])
                manifest = self.read_manifest()
                self.assertEqual(manifest["state"], "invalid")
                self.assertNotIn("runtime_identity", manifest)
                self.assert_owned_child_stopped()

    def test_requested_controls_are_recorded_and_bound(self):
        vector = self.directory / "synthetic-glp.gguf"
        vector.write_bytes(b"synthetic vector; never loaded")
        schema = self.directory / "synthetic-schema.json"
        schema.write_text('{"const": "synthetic"}')
        cases = [
            (["--gcd"], "gbnf", False, False, None),
            (["--glp", str(vector)], None, False, True, None),
            (["--gcd", "--glp", str(vector), "--glp-alpha", "0"],
             "gbnf", False, True, 0),
            (["--gcd-schema", str(schema), "--gcd-schema-locked"],
             "compiled_json_schema", True, False, None),
            (["--gcd-schema", str(schema), "--glp", str(vector), "--glp-alpha", "1"],
             "compiled_json_schema", False, True, 1065353216),
        ]
        base_argv = self.argv[:]
        for index, (flags, kind, locked, glp, alpha_bits) in enumerate(cases):
            with self.subTest(flags=flags):
                self.manifest = self.directory / f"controls-{index}.json"
                self.argv = base_argv + flags
                self.argv[self.argv.index("--manifest") + 1] = str(self.manifest)
                self.child.reset_mock()
                active = self.snapshot()
                active["active_controls"]["glp"] = {
                    "active": glp, "alpha_override_bits": alpha_bits}
                active["active_controls"]["grammar"] = {
                    "active": kind is not None, "kind": kind,
                    "locked": locked, "sha256": "e" * 64 if kind else None}
                self.invoke([active])
                manifest = self.read_manifest()
                self.assertEqual(manifest["state"], "stopped")
                self.assertEqual(manifest["runtime_identity"]["active_controls"],
                                 active["active_controls"])
                roles = {artifact["role"] for artifact in manifest["artifacts"]}
                expected_roles = {"binary", "model"}
                if glp:
                    expected_roles.add("glp")
                if kind == "compiled_json_schema":
                    expected_roles.add("grammar_schema")
                self.assertEqual(roles, expected_roles)
                self.assert_owned_child_stopped()

    def test_admission_changes_measurement_identity(self):
        artifacts = [runtime.seal(self.binary, "binary"), runtime.seal(self.model, "model")]
        original = self.snapshot()
        changed = deepcopy(original)
        changed["admission"]["request_timeout_seconds"] = 60
        first = runtime.build_manifest(original, artifacts, {}, self.argv)
        second = runtime.build_manifest(changed, artifacts, {}, self.argv)
        self.assertNotEqual(first["runtime_identity"]["identity_sha256"],
                            second["runtime_identity"]["identity_sha256"])

    def test_existing_state_root_prevents_spawn(self):
        state_root = self.manifest.with_name(self.manifest.name + ".state")
        state_root.mkdir()
        config = state_root / "config.toml"
        config.write_text("synthetic existing setup")
        with self.assertRaises(FileExistsError):
            self.invoke([])
        self.spawn.assert_not_called()
        self.child.terminate.assert_not_called()
        self.assertEqual(config.read_text(), "synthetic existing setup")

    def test_child_exit_invalidates_without_signalling_finished_process(self):
        self.child.poll.return_value = 7
        self.child.returncode = 7
        with self.assertRaisesRegex(ValueError, "owned server exited with status 7"):
            self.invoke([])
        self.assertEqual(self.read_manifest()["state"], "invalid")
        self.fetch.assert_not_called()
        self.child.terminate.assert_not_called()
        self.child.kill.assert_not_called()

    def test_shutdown_timeout_escalates_only_owned_child(self):
        self.child.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 30), 0]
        with patch.object(os, "kill") as unrelated_kill:
            runtime.stop_owned_child(self.child)
            unrelated_kill.assert_not_called()
        self.child.terminate.assert_called_once_with()
        self.child.kill.assert_called_once_with()
        self.assertEqual([call.kwargs for call in self.child.wait.call_args_list],
                         [{"timeout": 30}, {"timeout": 10}])

    def test_seal_rejects_change_during_hashing(self):
        before = runtime.stamp(self.model)
        after = {**before, "mtime_ns": before["mtime_ns"] + 1}
        with patch.object(runtime, "stamp", side_effect=[before, after]):
            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                runtime.seal(self.model, "model")

    def test_environment_drops_inherited_experiments_and_loader_overrides(self):
        with patch.dict(os.environ, {"HOME": str(self.directory), "PATH": "/synthetic",
                                    "HF2Q_EXPERIMENT": "hidden",
                                    "DYLD_INSERT_LIBRARIES": "hidden"}, clear=True):
            environment = runtime.child_environment(["HF2Q_EXPERIMENT=explicit"])
        self.assertEqual(environment, {"HOME": str(self.directory), "PATH": "/synthetic",
                                       "HF2Q_EXPERIMENT": "explicit"})
        with self.assertRaisesRegex(ValueError, "credential variables"):
            runtime.child_environment(["HF2Q_API_KEY=synthetic-secret"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
