#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("stage", HERE / "adr065_stage_profile.py")
stage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage)


def summary(seconds=10.0, response="r"):
    return {
        "own_output_four_task_makespan_seconds": seconds,
        "agents": {
            "agent-0": {
                "turns": [{
                    "turn": 0,
                    "request_sha256": "q",
                    "response_sha256": response,
                }],
                "final_source_sha256": "s",
                "observed_final_source_sha256": "s",
            }
        },
    }


def record(mode, wall, gpu=None, cpu=None, width=4, route="joint"):
    return {
        "schema": stage.SCHEMA,
        "mode": mode,
        "family": "qwen35",
        "route": route,
        "width": width,
        "wall_ns": wall,
        "gpu": gpu or {},
        "cpu": cpu or {},
        "unknown_labels": {},
        "deepseek_decode_scratch": {
            "release_events": 0,
            "reallocation_events": 0,
            "release_events_total": 0,
            "reallocation_events_total": 0,
        },
    }


class StageProfileTests(unittest.TestCase):
    def make_run(self, root, name, mode, records, seconds=10.0, response="r"):
        run = root / name
        (run / "client").mkdir(parents=True)
        client = run / "client" / "summary.json"
        client.write_text(json.dumps(summary(seconds, response)))
        env = {"HF2Q_ADR065_STAGE_PROFILE": mode}
        if mode == "1":
            env["MLX_PROFILE_CB"] = "1"
        receipt = {
            "verdict": {"verdict": "CalibrationPass"},
            "family": "qwen38",
            "cleanup": {"proven": True},
            "artifact": {"sha256": "artifact"},
            "binary": {"sha256": "binary"},
            "client": {"summary_path": str(client)},
            "serve_env": env,
        }
        (run / "receipt.json").write_text(json.dumps(receipt))
        (run / "server.stderr").write_text("".join(
            stage.PREFIX + json.dumps(value) + "\n" for value in records
        ))
        return run

    def test_pair_reports_distortion_and_excludes_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = self.make_run(root, "control", "wall", [
                record("wall", 100), record("wall", 120), record("wall", 110),
            ], 10.0)
            profile = self.make_run(root, "profile", "1", [
                record("profile", 200,
                       {"projections": {"count": 2, "gpu_ns": 50},
                        "unknown": {"count": 1, "gpu_ns": 20}},
                       {"host_selection": {"count": 1, "cpu_ns": 10},
                        "host_gap": {"count": 1, "cpu_ns": 120}}),
                record("profile", 220,
                       {"projections": {"count": 2, "gpu_ns": 60}},
                       {"host_gap": {"count": 1, "cpu_ns": 160}}),
            ], 20.0)
            reference = self.make_run(root, "reference", "off", [], 9.0)
            result = stage.analyze_pair("qwen38", control, profile, reference)
            compared = result["comparisons"]["joint:w4"]
            self.assertEqual(compared["control_wall_ns_median"], 110)
            self.assertEqual(compared["profile_wall_ns_median"], 210.0)
            self.assertAlmostEqual(compared["profile_over_control"], 210 / 110)
            attribution = compared["profile_attribution"]
            self.assertEqual(attribution["gpu_ns"]["unknown"], 20)
            self.assertAlmostEqual(attribution["unknown_gpu_share"], 20 / 420)
            self.assertAlmostEqual(attribution["known_share"], 400 / 420)
            self.assertTrue(result["exact_transcripts_vs_previous_default"])

    def test_transcript_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = self.make_run(root, "control", "wall", [record("wall", 100)])
            profile = self.make_run(root, "profile", "1", [record("profile", 200)])
            reference = self.make_run(root, "reference", "off", [], response="changed")
            with self.assertRaisesRegex(ValueError, "transcript differs"):
                stage.analyze_pair("qwen38", control, profile, reference)

    def test_wall_record_rejects_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.stderr"
            value = record("wall", 100, {"head": {"count": 1, "gpu_ns": 5}})
            path.write_text(stage.PREFIX + json.dumps(value) + "\n")
            with self.assertRaisesRegex(ValueError, "wall control contains"):
                stage.parse_records(path)


if __name__ == "__main__":
    unittest.main()
