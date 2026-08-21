import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_nested_generation_kind_executes_exact_trajectory(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.rows = 0

            def write(self, point: dict, logits: np.ndarray) -> int:
                self.rows += 1
                self.assert_point = point
                return int(np.argmax(logits))

        point = {
            "point_ordinal": 0,
            "stable_id": "generation-1",
            "kind": {"kind": "generation_next"},
            "prefix_token_count": 2,
            "prefix_token_ids_sha256": "34" * 32,
        }
        prompt = {
            "stable_id": "generation-1",
            "prefix_token_count": 2,
            "prefix_token_ids_sha256": "34" * 32,
        }
        reference_input = {
            "prediction_plan": {
                "prediction_points": [point],
                "greedy_prompts": [prompt],
            },
            "examples": [{"stable_id": "generation-1", "token_ids": [4, 5]}],
        }
        calls: list[list[int]] = []

        def fake_last_logits(model, torch, device, tokens, cache):
            calls.append(tokens)
            return np.array([0.0, 2.0, 1.0], dtype=np.float32), object()

        writer = Writer()
        with mock.patch.object(REFERENCE, "last_logits", fake_last_logits):
            trajectories = REFERENCE.execute_plan(
                reference_input, writer, object(), object(), object()
            )

        self.assertEqual(writer.rows, 1)
        self.assertEqual(len(calls), REFERENCE.GREEDY_TOKEN_COUNT)
        self.assertEqual(calls[0], [4, 5])
        self.assertTrue(all(tokens == [1] for tokens in calls[1:]))
        self.assertEqual(
            trajectories[0]["token_ids"], [1] * REFERENCE.GREEDY_TOKEN_COUNT
        )

    def test_nested_teacher_forced_kind_validates_target(self) -> None:
        reference_input = {
            "schema_version": 1,
            "profile": "exact_teacher_reference_input_v1",
            "prediction_plan": {
                "examples": [
                    {
                        "stable_id": "teacher-1",
                        "render_mode": "completed_transcript",
                        "token_count": 3,
                        "token_ids_sha256": REFERENCE.rendered_tokens_sha256(
                            "teacher-1", [4, 5, 6]
                        ),
                    }
                ],
                "prediction_points": [
                    {
                        "stable_id": "teacher-1",
                        "kind": {
                            "kind": "teacher_forced",
                            "target_token_index": 2,
                            "target_token_id": 6,
                        },
                        "prefix_token_count": 2,
                        "prefix_token_ids_sha256": REFERENCE.prefix_sha256([4, 5]),
                    }
                ],
            },
            "vocabulary_size": 8,
            "target_limits": {},
            "examples": [
                {
                    "stable_id": "teacher-1",
                    "render_mode": "completed_transcript",
                    "token_ids": [4, 5, 6],
                }
            ],
            "source_teacher_authority": False,
            "sensitivity_authority": False,
            "allocator_authority": False,
            "selector_authority": False,
            "autoquant_authority": False,
            "runtime_dependency": False,
        }
        reference_input["reference_input_sha256"] = REFERENCE.sha256_bytes(
            REFERENCE.canonical_json(REFERENCE.reference_input_hash_view(reference_input))
        )
        REFERENCE.validate_reference_input(reference_input)

        reference_input["prediction_plan"]["prediction_points"][0]["kind"][
            "target_token_id"
        ] = 7
        reference_input["reference_input_sha256"] = REFERENCE.sha256_bytes(
            REFERENCE.canonical_json(REFERENCE.reference_input_hash_view(reference_input))
        )
        with self.assertRaisesRegex(ValueError, "teacher-forced target"):
            REFERENCE.validate_reference_input(reference_input)

    def test_policy_validation_executes_rows_without_a_trajectory(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.rows = 0

            def write(self, point: dict, logits: np.ndarray) -> int:
                self.rows += 1
                self.assert_point = point
                return int(np.argmax(logits))

        point = {
            "point_ordinal": 0,
            "stable_id": "policy-1",
            "kind": {
                "kind": "teacher_forced",
                "target_token_index": 2,
                "target_token_id": 6,
            },
            "prefix_token_count": 2,
            "prefix_token_ids_sha256": REFERENCE.prefix_sha256([4, 5]),
        }
        reference_input = {
            "prediction_plan": {
                "prediction_points": [point],
                "greedy_prompts": [],
            },
            "examples": [{"stable_id": "policy-1", "token_ids": [4, 5, 6]}],
        }
        calls: list[list[int]] = []

        def fake_last_logits(model, torch, device, tokens, cache):
            calls.append(tokens)
            return np.array([0.0, 2.0, 1.0], dtype=np.float32), object()

        writer = Writer()
        with mock.patch.object(REFERENCE, "last_logits", fake_last_logits):
            trajectories = REFERENCE.execute_plan(
                reference_input, writer, object(), object(), object()
            )

        self.assertEqual(writer.rows, 1)
        self.assertEqual(calls, [[4, 5]])
        self.assertEqual(trajectories, [])


if __name__ == "__main__":
    unittest.main()
