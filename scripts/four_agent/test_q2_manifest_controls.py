"""Mock file-layout controls exercise actual post-unload assembler, not QE proof."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import q2_manifest as assembler


class ManifestControls(unittest.TestCase):
    def test_actual_bound_file_hash_refuses_symlink_and_large_file(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root).resolve() / "raw"
            path.write_bytes(b"raw")
            self.assertEqual(assembler.bound(path)["sha256"], hashlib.sha256(b"raw").hexdigest())
            with self.assertRaises(ValueError): assembler.bound(path, 2)
            alias = path.with_name("alias"); alias.symlink_to(path)
            with self.assertRaises(ValueError): assembler.bound(alias)

    def test_actual_compiler_layout_requires_count_and_retains_binary_hash(self):
        with tempfile.TemporaryDirectory() as root:
            agent = Path(root).resolve(); logs = agent / "tools/tool-04/logs/compile"
            logs.mkdir(parents=True)
            binary = agent / "mock-test-binary"; binary.write_bytes(b"NOT EXECUTED")
            (logs / "receipt.json").write_text(json.dumps({"exit_code": 0}))
            artifact = {"reason": "compiler-artifact", "executable": str(binary)}
            (logs / "stdout.bin").write_text(json.dumps(artifact) + "\n")
            (logs / "stderr.bin").write_bytes(b"")
            execution = logs.parent / "test-0"; execution.mkdir()
            for name in ("receipt.json", "stdout.bin", "stderr.bin"):
                (execution / name).write_bytes(b"{}" if name.endswith("json") else b"")
            result = assembler.test_group(agent, 4)
            self.assertEqual(result["executions"][0]["binary"]["sha256"], hashlib.sha256(b"NOT EXECUTED").hexdigest())
            (logs / "stdout.bin").write_text("")
            with self.assertRaises(ValueError): assembler.test_group(agent, 4)
            (logs / "receipt.json").write_text(json.dumps({"exit_code": 1}))
            with self.assertRaises(ValueError): assembler.test_group(agent, 4)

    def test_assembler_closed_input_never_defaults_missing_proof(self):
        for spec in ({}, {"schema": "hf2q.q2.manifest-input.v1"}, {"schema": "other", "extra": True}):
            with self.assertRaises(ValueError): assembler.assemble(spec)


if __name__ == "__main__":
    unittest.main()
