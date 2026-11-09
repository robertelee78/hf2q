#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import adr065_critical_path as subject


def prefill(slot, prompt, begin, end):
    return {
        "schema": subject.INPUT_SCHEMA,
        "clock_id": subject.CLOCK_ID,
        "kind": "prefill",
        "begin_ns": begin,
        "end_ns": end,
        "slot": slot,
        "prompt_tokens": prompt,
        "chunk_start": 0,
        "chunk_end": prompt,
    }


def decode(mode, lanes, begin, end):
    return {
        "schema": subject.INPUT_SCHEMA,
        "clock_id": subject.CLOCK_ID,
        "kind": "decode",
        "mode": mode,
        "begin_ns": begin,
        "end_ns": end,
        "width": len(lanes),
        "lanes": [
            {"slot": slot, "verify_eligibility": eligibility}
            for slot, eligibility in lanes
        ],
    }


class CriticalPathTests(unittest.TestCase):
    def make_run(self, name="run-a", records=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name) / name
        (run / "client").mkdir(parents=True)
        (run / "receipt.json").write_text(json.dumps({
            "verdict": {"verdict": "CalibrationPass"},
            "cleanup": {"proven": True},
        }))
        agents = {}
        for slot in range(4):
            send = 110 + slot * 5
            terminal = 500 + slot * 10
            agents[f"agent-{slot}"] = {
                "f2_clock_id": subject.CLOCK_ID,
                "task_completed_monotonic_ns": 600 + slot * 10,
                "turns": [{
                    "turn": 0,
                    "f2_clock_id": subject.CLOCK_ID,
                    "send_monotonic_ns": send,
                    "validated_terminal_monotonic_ns": terminal,
                    "usage": {
                        "prompt_tokens": 100 + slot,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                }],
            }
        summary = {
            "schema": "hf2q.qwen-n4-coordinator-result.v1",
            "status": "FourAgentProtocolCompleteAwaitingIndependentReview",
            "f2_clock_id": subject.CLOCK_ID,
            "makespan_started_ns": 100,
            "makespan_finished_ns": 1000,
            "agents": agents,
        }
        (run / "client" / "summary.json").write_text(json.dumps(summary))
        if records is None:
            records = [
                prefill(0, 100, 150, 170),
                prefill(1, 101, 170, 190),
                prefill(2, 102, 190, 210),
                prefill(3, 103, 210, 230),
                decode("joint", [(0, "eligible"), (1, "eligible"),
                                 (2, "blocked_transform"), (3, "no_draft")], 250, 350),
                decode("solo", [(0, "eligible")], 350, 400),
            ]
        body = "noise\n" + "\n".join(
            f"prefix {subject.PREFIX}{json.dumps(record)}" for record in records
        ) + "\n"
        (run / "server.stderr").write_text(body)
        return run

    def test_partitions_makespan_and_time_weights_joint_eligibility(self):
        run = subject.summarize_run(self.make_run())
        self.assertEqual(sum(run["critical_path_ns"].values()), 900)
        self.assertEqual(run["critical_path_ns"]["prefill"], 80)
        self.assertEqual(run["critical_path_ns"]["joint"], 100)
        self.assertEqual(run["critical_path_ns"]["solo"], 50)
        self.assertEqual(run["joint_width_distribution"], {
            "4": {"turns": 1, "duration_ns": 100},
        })
        eligibility = run["verify_eligibility"]
        self.assertEqual(eligibility["eligible_lane_weighted_ns"], 50)
        self.assertEqual(eligibility["eligible_lane_ns"], 200)
        self.assertEqual(eligibility["total_lane_ns"], 400)
        self.assertAlmostEqual(eligibility["f"], 1 / 18)

    def test_run_cluster_interval_uses_independent_run_values(self):
        report = subject.summarize([self.make_run("run-a"), self.make_run("run-b")])
        clustered = report["run_clustered_f"]
        self.assertEqual(clustered["runs"], 2)
        self.assertAlmostEqual(clustered["mean"], 1 / 18)
        self.assertEqual(clustered["ci95"], [1 / 18, 1 / 18])

    def test_rejects_overlapping_engine_intervals(self):
        records = [
            prefill(0, 100, 150, 200),
            decode("solo", [(0, "eligible")], 190, 210),
        ]
        run = self.make_run(records=records)
        with self.assertRaisesRegex(ValueError, "engine intervals overlap"):
            subject.read_intervals(run / "server.stderr")


if __name__ == "__main__":
    unittest.main()
