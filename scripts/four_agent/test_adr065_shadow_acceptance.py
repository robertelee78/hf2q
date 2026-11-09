#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import adr065_shadow_acceptance as subject


def record(**changes):
    value = {
        "schema": subject.INPUT_SCHEMA,
        "slot": 0,
        "proposer": "mtp_k3",
        "turn_type": "warm",
        "context_bucket": "near_16k",
        "start_position": 16384,
        "draft_count": 3,
        "accepted_prefix": 2,
        "proposal_cost_ns": 120,
        "censored": False,
        "no_grammar": True,
        "no_repetition_penalty": True,
        "no_forced_thinking": True,
        "no_tool_trigger": True,
        "no_transforms": True,
    }
    value.update(changes)
    return value


class ShadowAcceptanceTests(unittest.TestCase):
    def write_log(self, rows):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "server.log"
        path.write_text("noise\n" + "\n".join(
            f"prefix {subject.PREFIX}{json.dumps(row)}" for row in rows) + "\n")
        return path

    def test_reports_histograms_groups_costs_and_transform_share(self):
        path = self.write_log([
            record(slot=0, accepted_prefix=2, proposal_cost_ns=100),
            record(slot=1, accepted_prefix=0, proposal_cost_ns=300,
                   no_grammar=False, no_transforms=False),
            record(proposer="history", slot=1, turn_type="cold",
                   context_bucket="near_2k", draft_count=1,
                   accepted_prefix=1, proposal_cost_ns=20),
            record(censored=True, accepted_prefix=1),
        ])
        report = subject.summarize(subject.read_records([path]), [path])
        self.assertEqual(report["records"], 4)
        self.assertEqual(report["complete_records"], 3)
        self.assertEqual(report["censored_records"], 1)
        mtp = report["proposers"]["mtp_k3"]
        self.assertEqual(mtp["accepted_prefix_histogram"], {"0": 1, "1": 0, "2": 1, "3": 0})
        self.assertEqual(mtp["by_slot"]["1"]["0"], 1)
        self.assertEqual(mtp["proposal_cost_ns"]["mean"], 200)
        self.assertEqual(mtp["no_transform_share"], 0.5)
        history = report["proposers"]["history"]
        self.assertEqual(history["by_turn_type"]["cold"]["1"], 1)

    def test_rejects_inconsistent_transform_summary(self):
        path = self.write_log([record(no_grammar=False, no_transforms=True)])
        with self.assertRaisesRegex(ValueError, "no_transforms disagrees"):
            subject.read_records([path])

    def test_rejects_prefix_longer_than_draft(self):
        path = self.write_log([record(draft_count=1, accepted_prefix=2)])
        with self.assertRaisesRegex(ValueError, "invalid accepted_prefix"):
            subject.read_records([path])


if __name__ == "__main__":
    unittest.main()
