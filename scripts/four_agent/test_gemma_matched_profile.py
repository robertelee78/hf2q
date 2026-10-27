"""Pure semantic/opt-in controls; synthetic raw fixtures NEVER execute a process."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import gemma_matched_profile as m
from qwen38_n1_verdict import Invalid
from qwen_n4_live_smoke import identity
from qwen38_n1_smoke import build_request


def fixture(target="all", prefix="/owned/a"):
    names = list(m.TESTS) if target == "all" else ["ports"]
    compiler, runs = [], []
    processes = [{"exit_code": 0, "cleanup": True, "cleanup_receipt": {"verified": True}}]
    for name in names:
        executable = prefix + "/" + name
        compiler.append({"reason": "compiler-artifact", "target": {"name": name},
                         "profile": {"test": True}, "executable": executable})
        text = "running 2 tests\n" + "".join("test " + t + " ... ok\n" for t in m.TESTS[name])
        text += "test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s\n"
        runs.append({"target": name, "executable": executable, "stdout": text.encode(), "stderr": b""})
        processes.append(copy.deepcopy(processes[0]))
    compiler.append({"reason": "build-finished", "success": True})
    detail = {"target": target, "passed": True, "cleanup": True,
              "source_sha256": "a" * 64, "processes": processes}
    return detail, compiler, runs


def encode(compiler):
    return b"\n".join(json.dumps(item).encode() for item in compiler)


class Semantic(unittest.TestCase):
    def test_real_named_targets_and_counts_in_semantic_wire(self):
        for target, count in (("ports", 1), ("all", 2)):
            d, c, runs = fixture(target)
            result = json.loads(m.semantic_tests(d, encode(c), runs))
            self.assertEqual(len(result["artifacts"]), count)
            self.assertEqual(sum(len(a["tests"]) for a in result["artifacts"]), 2 * count)
            self.assertTrue(result["passed"])

    def test_paths_wall_time_and_artifact_order_do_not_enter_model_wire(self):
        d, c, runs = fixture()
        original = m.semantic_tests(d, encode(c), runs)
        d2, c2, runs2 = fixture(prefix="/different/evidence/path")
        for run in runs2:
            run["stdout"] = run["stdout"].replace(b"0.13s", b"9.87s")
        self.assertEqual(original, m.semantic_tests(d2, encode(c2), list(reversed(runs2))))
        self.assertNotIn("/owned", original)
        self.assertNotIn("finished", original)

    def test_exit_zero_without_complete_build_or_actual_tests_refused(self):
        d, c, runs = fixture()
        for bad in (c[:-1], c + [c[-1]], c[:-1] + [{"reason": "build-finished", "success": False}]):
            with self.assertRaises(Invalid): m.semantic_tests(d, encode(bad), runs)
        for badstdout in (b"", b"test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0s\n"):
            bad = copy.deepcopy(runs);bad[0]["stdout"] = badstdout
            with self.assertRaises(Invalid): m.semantic_tests(d, encode(c), bad)

    def test_duplicate_unknown_nontest_artifacts_refused(self):
        d, c, runs = fixture()
        variants = [c + [c[0]]]
        unknown = copy.deepcopy(c);unknown[0]["target"]["name"] = "foreign";variants.append(unknown)
        non_test = copy.deepcopy(c);non_test[0]["profile"]["test"] = False;variants.append(non_test)
        for bad in variants:
            with self.assertRaises(Invalid): m.semantic_tests(d, encode(bad), runs)

    def test_compiler_error_not_hidden_by_synthetic_pass_flags(self):
        d, c, runs = fixture()
        c.append({"reason": "compiler-message", "message": {"level": "error", "rendered": "failure"}})
        with self.assertRaises(Invalid): m.semantic_tests(d, encode(c), runs)

    def test_nonzero_cleanup_failure_and_bad_digest_refused(self):
        d, c, runs = fixture()
        variants = []
        for field, value in (("passed", False), ("cleanup", False), ("source_sha256", "bad")):
            bad = copy.deepcopy(d);bad[field] = value;variants.append(bad)
        for field, value in (("exit_code", 1), ("cleanup", False), ("cleanup_receipt", {"verified": False})):
            bad = copy.deepcopy(d);bad["processes"][1][field] = value;variants.append(bad)
        for bad in variants:
            with self.assertRaises(Invalid): m.semantic_tests(bad, encode(c), runs)

    def test_names_duplicates_stderr_and_foreign_executable_refused(self):
        d, c, runs = fixture()
        variants = []
        for field, value in (("stderr", b"failure"), ("executable", "/foreign"), ("target", "bad")):
            bad = copy.deepcopy(runs);bad[0][field] = value;variants.append(bad)
        for replacement in (b"wrong_name", m.TESTS[runs[0]["target"]][1].encode()):
            bad = copy.deepcopy(runs)
            bad[0]["stdout"] = bad[0]["stdout"].replace(m.TESTS[runs[0]["target"]][0].encode(), replacement)
            variants.append(bad)
        for bad in variants:
            with self.assertRaises(Invalid): m.semantic_tests(d, encode(c), bad)

    def test_explicit_profile_never_inherits_other_family_or_peer(self):
        m.validate_activation(m.PROFILE, {"benchmark_profile": m.PROFILE}, "gemma4", "hf2q", 4)
        for profile, packet, family, engine, slots in (
            (None, {}, "gemma4", "hf2q", 4), (m.PROFILE, {}, "gemma4", "hf2q", 4),
            (m.PROFILE, {"benchmark_profile": m.PROFILE}, "qwen", "hf2q", 4),
            (m.PROFILE, {"benchmark_profile": m.PROFILE}, "gemma4", "peer", 4),
            (m.PROFILE, {"benchmark_profile": m.PROFILE}, "gemma4", "hf2q", 1)):
            with self.assertRaises(Invalid): m.validate_activation(profile, packet, family, engine, slots)

    def test_existing_model_visible_identity_and_request_are_run_independent(self):
        self.assertEqual(identity("agent-0", "run_123456789012345"), identity("agent-0", "run_987654321098765"))
        self.assertNotEqual(identity("agent-0", "run_123456789012345"), identity("agent-1", "run_123456789012345"))
        request = build_request("Gemma", [], [], 1, "gemma4")
        self.assertNotIn("benchmark_profile", request)
        self.assertEqual(request["max_completion_tokens"], 8192)

    def test_actual_used_invoke_wrapper_runs_real_seam_before_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);source = root / "fixture/src/ports.rs";source.parent.mkdir(parents=True)
            source.write_text("real fixture bytes");log = root / "tool-01";log.mkdir()
            tools = SimpleNamespace(root=root, fixture=root / "fixture", receipts=[], calls=0)
            d, c, runs = fixture("ports");d.update(name="run_tests", arguments={"target": "ports"},
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), status="Complete", result="raw compile stderr/time")
            events = []
            def real(name, args):
                events.append((name, args));tools.calls += 1;tools.receipts.append(d);return d["result"]
            with patch.object(m, "load_real_proof", return_value=(encode(c), runs)) as proof:
                result = m.invoke_semantic(tools, "run_tests", {"target": "ports"}, real)
            self.assertEqual(events, [("run_tests", {"target": "ports"})]);proof.assert_called_once_with(d, log)
            self.assertEqual(d["raw_model_result"], "raw compile stderr/time")
            self.assertEqual(d["result"], result)
            self.assertEqual(json.loads((log / "result.json").read_text())["benchmark_profile"], m.PROFILE)

    def test_wrapper_failure_retains_raw_and_marks_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"fixture/src/ports.rs";source.parent.mkdir(parents=True);source.write_text("x")
            (root/"tool-01").mkdir();d={"name":"run_tests","arguments":{"target":"ports"},"source_sha256":hashlib.sha256(b"x").hexdigest(),"status":"Complete"}
            tools=SimpleNamespace(root=root,fixture=root/"fixture",receipts=[d],calls=1)
            with patch.object(m,"load_real_proof",side_effect=Invalid("missing actual proof")):
                with self.assertRaises(Invalid):m.invoke_semantic(tools,"run_tests",{"target":"ports"},lambda *_:"retained raw failure")
            self.assertEqual(d["status"],"Failed");self.assertEqual(d["raw_model_result"],"retained raw failure")

    def test_actual_raw_loader_binds_artifacts_and_refuses_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            log=Path(directory);(log/"target").mkdir();(log/"logs/compile").mkdir(parents=True)
            (log/"logs/test-0").mkdir();executable=log/"target/ports";executable.write_bytes(b"synthetic artifact, NEVER executed")
            d,c,runs=fixture("ports",str(log/"target"));d["artifact_sha256"]=[hashlib.sha256(executable.read_bytes()).hexdigest()]
            d["processes"][0].update(pid=7,argv=["compiler"])
            d["processes"][1].update(pid=8,argv=["sandbox","-f","profile",str(executable),"--test-threads=1"],actual_tests_passed=2)
            (log/"logs/compile/stdout.bin").write_bytes(encode(c))
            for index,p in enumerate(d["processes"]):
                place=log/"logs"/("compile" if index==0 else "test-0")
                (place/"receipt.json").write_text(json.dumps(p))
            (log/"logs/test-0/stdout.bin").write_bytes(runs[0]["stdout"])
            (log/"logs/test-0/stderr.bin").write_bytes(b"")
            actual_compiler,actual_runs=m.load_real_proof(d,log)
            self.assertTrue(json.loads(m.semantic_tests(d,actual_compiler,actual_runs))["passed"])
            executable.write_bytes(b"changed")
            with self.assertRaises(Invalid):m.load_real_proof(d,log)
            executable.write_bytes(b"synthetic artifact, NEVER executed")
            receipt=log/"logs/test-0/receipt.json";bad=json.loads(receipt.read_text());bad["pid"]=999;receipt.write_text(json.dumps(bad))
            with self.assertRaises(Invalid):m.load_real_proof(d,log)

    def test_non_test_tool_result_unchanged(self):
        self.assertEqual(m.invoke_semantic(None,"read_file",{"path":"src/ports.rs"},lambda *_:"exact bytes"),"exact bytes")


if __name__ == "__main__": unittest.main()
