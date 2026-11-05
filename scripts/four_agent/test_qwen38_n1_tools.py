"""Model-free controls for the run_tests workload normalization (ADR-063, operator 2026-11-05, from B1).

The real `Tools.invoke("run_tests")` path runs against a scripted cargo/libtest double (`child.run`): no cargo,
sandbox or model. Two owned roots in different run directories, with different agent ids and wall-clock
durations, must give byte-identical model-visible results while each receipt keeps its own raw output, and
error codes and test-pass parsing must read the same as before.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import qwen38_n1_tools as tools_module
from qwen38_n1_tools import CARGO_CWD, OWNED_ROOT, Tools, model_visible
from qwen38_n1_verdict import test_harness
import q3_report

LIBTEST = ("\nrunning 2 tests\ntest accepts_u16_ports ... ok\ntest rejects_invalid_ports ... ok\n\n"
           "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in {test}\n\n")
MISMATCH = ("error[E0308]: mismatched types\n --> src/ports.rs:1:37\n  |\n"
            "1 | pub fn parse_port(s: &str) -> u16 { s.parse::<u32>().unwrap() }\n"
            "  |                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^ expected `u16`, found `u32`\n\n")


class FakeCargo:
    """Scripted `child.run`: one cargo compile, then one run of each test artifact it reported."""

    def __init__(self, tools, finished, test, compile_error=False):
        self.tools, self.finished, self.test, self.compile_error = tools, finished, test, compile_error

    def __call__(self, argv, cwd, env, sandbox, sbpl, output, target, **_):
        receipt = {"exit_code": 0, "cleanup": True}
        fixture = self.tools.fixture
        if argv[0] != str(self.tools.cargo):
            return receipt, LIBTEST.format(test=self.test).encode(), b""
        if self.compile_error:
            stderr = (f"   Compiling qwen38_n1_fixture v0.0.0 ({fixture})\n"
                      "error: could not compile `qwen38_n1_fixture` (lib) due to 1 previous error\n")
            message = {"reason": "compiler-message", "message": {"level": "error", "rendered": MISMATCH}}
            return dict(receipt, exit_code=101), (json.dumps(message) + "\n").encode(), stderr.encode()
        executable = Path(target) / "debug" / "deps" / "ports-0123456789abcdef"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"test binary")
        warning = {"reason": "compiler-message", "message": {
            "level": "warning", "rendered": f"warning: unused import\n --> {fixture}/src/lib.rs:1:5\n"
                                            f"  = note: cargo ran from {cwd}\n\n"}}
        artifact = {"reason": "compiler-artifact", "executable": str(executable),
                    "profile": {"test": True}, "target": {"name": "ports"}}
        stdout = "".join(json.dumps(item) + "\n" for item in (warning, artifact,
                                                              {"reason": "build-finished", "success": True}))
        stderr = (f"   Compiling qwen38_n1_fixture v0.0.0 ({fixture})\n"
                  f"    Finished `test` profile [unoptimized + debuginfo] target(s) in {self.finished}\n")
        return receipt, stdout.encode(), stderr.encode()


class RunTestsNormalization(unittest.TestCase):
    def owned(self, run, agent):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / run / "client" / agent / "tools"
        root.parent.mkdir(parents=True)
        tools = Tools(root, "/not/executed/bin/cargo", "/not/executed/sandbox", "/Library/Developer/SDKs/Mock.sdk")
        self.addCleanup(tools.close)
        self.addCleanup(tools.cwd.rmdir)
        return tools

    def run_tests(self, tools, target="ports", **cargo):
        with mock.patch.object(tools_module.child, "run", FakeCargo(tools, **cargo)), \
                mock.patch.object(tools_module.child, "profile", return_value="(version 1)"), \
                mock.patch.object(tools_module.child, "compiler_environment", return_value={}):
            result = tools.invoke("run_tests", {"target": target})
        receipt = tools.receipts[-1]
        self.assertEqual(receipt["result"], result, "the receipt records exactly what the model saw")
        return result, receipt

    def test_two_run_dirs_and_timings_give_byte_identical_model_visible_output(self):
        a = self.owned("011-b06-on-20261105T033737Z", "agent-0")
        b = self.owned("010-b05-on-20261105T033714Z", "agent-2")
        result_a, receipt_a = self.run_tests(a, finished="0.15s", test="0.00s")
        result_b, receipt_b = self.run_tests(b, finished="1m 05s", test="0.37s")
        self.assertEqual(result_a.encode(), result_b.encode())
        visible = json.loads(result_a)["output"]
        self.assertIn(f"({OWNED_ROOT}/fixture)", visible)
        self.assertIn(f"{OWNED_ROOT}/fixture/src/lib.rs:1:5", visible)
        self.assertIn(f"cargo ran from {CARGO_CWD}", visible)
        self.assertIn("target(s) in 0.00s\n", visible)
        self.assertIn("filtered out; finished in 0.00s\n", visible)
        for tools, receipt in ((a, receipt_a), (b, receipt_b)):
            self.assertNotIn(str(tools.root), visible)
            self.assertNotIn(str(tools.cwd), visible)
            # Raw output stays verbatim in the receipt, run directory, agent id and timings included.
            raw = receipt["raw_output"]
            self.assertIn(f"({tools.fixture})", raw)
            self.assertIn(f"cargo ran from {tools.cwd}", raw)
            self.assertEqual(model_visible(raw, tools.run_roots()), receipt["output"])
        self.assertIn("in 0.15s\n", receipt_a["raw_output"])
        self.assertIn("in 1m 05s\n", receipt_b["raw_output"])
        self.assertIn("finished in 0.37s\n", receipt_b["raw_output"])
        self.assertNotEqual(receipt_a["raw_output"], receipt_b["raw_output"])

    def test_test_pass_parsing_is_unchanged(self):
        tools = self.owned("001-b01-off-20261105T033349Z", "agent-1")
        result, receipt = self.run_tests(tools, finished="0.21s", test="0.01s")
        detail = json.loads(result)
        self.assertIs(detail["passed"], True)
        self.assertEqual([p.get("actual_tests_passed") for p in receipt["processes"]], [None, 2])
        # The libtest result marker still parses on the model view.
        self.assertEqual(test_harness(detail["output"].encode(), 2), 2)

    def test_compiler_error_code_is_unchanged(self):
        tools = self.owned("002-b01-on-20261105T033411Z", "agent-3")
        result, receipt = self.run_tests(tools, finished="0.15s", test="0.00s", compile_error=True)
        detail = json.loads(result)
        self.assertIs(detail["passed"], False)
        self.assertIn(MISMATCH, detail["output"])
        self.assertIn(f"({OWNED_ROOT}/fixture)", detail["output"])
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "client" / "summary.json"
            summary.parent.mkdir()
            summary.write_text(json.dumps({"agents": {"agent-3": {
                "status": "Failed", "error": "Invalid: tool arguments/workflow drift", "tools": [receipt]}}}))
            self.assertEqual(q3_report.failure_detail({"attempts": [{"output": directory}]}),
                             "agent-3: run_tests ports failed with E0308, then Invalid: tool arguments/workflow drift")

    def test_longest_prefix_first_and_durations(self):
        roots = [("/private/tmp/run/tools", OWNED_ROOT), ("/tmp/run/tools", OWNED_ROOT)]
        raw = ("x (/private/tmp/run/tools/fixture) (/tmp/run/tools/fixture)\n"
               "    Finished `test` profile [unoptimized + debuginfo] target(s) in 12.34s\n"
               "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s\n"
               "user text in 1.5s stays\n")
        self.assertEqual(model_visible(raw, roots),
                         f"x ({OWNED_ROOT}/fixture) ({OWNED_ROOT}/fixture)\n"
                         "    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.00s\n"
                         "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
                         "user text in 1.5s stays\n")


if __name__ == "__main__":
    unittest.main()
