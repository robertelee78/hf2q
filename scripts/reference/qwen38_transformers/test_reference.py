import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("run_reference.py")
SPEC = importlib.util.spec_from_file_location("qwen38_reference", SCRIPT)
REFERENCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REFERENCE)


class ReferenceArtifactTests(unittest.TestCase):
    def test_target_writer_uses_exact_framing_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.bin"
            writer = REFERENCE.TargetWriter(output, 4, 1)
            point = {
                "point_ordinal": 0,
                "prefix_token_ids_sha256": "12" * 32,
            }
            self.assertEqual(writer.write(point, np.array([0.0, 3.0, 2.0, 1.0])), 1)
            length, digest = writer.finish()
            payload = output.read_bytes()
            self.assertEqual(length, len(REFERENCE.TARGET_MAGIC) + 60 + 16)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
            self.assertEqual(payload[: len(REFERENCE.TARGET_MAGIC)], REFERENCE.TARGET_MAGIC)
            self.assertEqual(payload[len(REFERENCE.TARGET_MAGIC) :][:4], b"ROW1")
            with self.assertRaises(FileExistsError):
                REFERENCE.TargetWriter(output, 4, 1)

    def test_canonical_json_rejects_nonfinite_values(self) -> None:
        with self.assertRaises(ValueError):
            REFERENCE.canonical_json({"value": float("nan")})
        value = {"schema_version": 1, "enabled": False}
        self.assertEqual(
            REFERENCE.canonical_json(value),
            json.dumps(value, separators=(",", ":")).encode(),
        )


if __name__ == "__main__":
    unittest.main()
