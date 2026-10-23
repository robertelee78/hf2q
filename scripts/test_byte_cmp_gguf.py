#!/usr/bin/env python3
"""Real GGUF boundary controls; run in the checker's pinned gguf/numpy environment."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import gguf
import numpy as np


class TensorComparisonBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="hf2q-gguf-cmp-")
        cls.root = Path(cls.temp.name)
        cls.script = Path(__file__).with_name("byte_cmp_gguf.py")
        cls.base = cls.fixture("base", "weight", np.float32, (32,))
        cls.same = cls.fixture("same", "weight", np.float32, (32,))
        cls.missing = cls.fixture("missing", "other", np.float32, (32,))
        cls.dtype = cls.fixture("dtype", "weight", np.float16, (32,))
        cls.shape = cls.fixture("shape", "weight", np.float32, (4, 8))
        cls.payload = cls.fixture("payload", "weight", np.float32, (32,), 1)
        cls.empty = cls.fixture("empty", None, np.float32, (32,))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def fixture(cls, name, tensor, dtype, shape, value=0):
        path = cls.root / (name + ".gguf")
        writer = gguf.GGUFWriter(str(path), "llama")
        if tensor is not None:
            writer.add_tensor(tensor, np.full(shape, value, dtype=dtype))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path

    def compare(self, candidate, code, message, *, stderr=False):
        result = subprocess.run([sys.executable, str(self.script), str(self.base), str(candidate)],
                                text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        self.assertIn(message, result.stderr if stderr else result.stdout)

    def test_identical_tensors_pass(self):
        self.compare(self.same, 0, "All 1 tensors byte-identical")

    def test_missing_or_extra_tensor_rejects(self):
        self.compare(self.missing, 1, "tensors missing")

    def test_tensor_type_mismatch_rejects(self):
        self.compare(self.dtype, 1, "TYPE-MISMATCH")

    def test_tensor_shape_mismatch_rejects(self):
        self.compare(self.shape, 1, "SHAPE-MISMATCH")

    def test_tensor_payload_mismatch_rejects(self):
        self.compare(self.payload, 1, "1 MISMATCHED TENSORS")

    def test_empty_input_rejects(self):
        self.compare(self.empty, 1, "must contain tensor data", stderr=True)


if __name__ == "__main__":
    unittest.main()
