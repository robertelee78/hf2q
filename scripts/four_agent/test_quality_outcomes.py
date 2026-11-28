"""Consumer-visible quality errors: rejected/unreviewed patches and forged legacy Pass claims."""
import unittest

import f2_q3_tier
import quality_campaign as campaign
import quality_outcomes
from qwen_n4_live_smoke import AGENTS

SUBJECT, REFERENCE = "a" * 64, "b" * 64


def captured_tasks():
    runs = []
    for label in range(6):
        for arm, source in (("subject", SUBJECT), ("reference", REFERENCE)):
            agents = {agent: {"outcome": "Pass", "final_source_sha256": source} for agent in AGENTS}
            runs.append({"instance": label, "arm": arm, "outcome": "Pass", "attempts": [{"agents": agents}]})
    return {"plan": {"instances": 6, "reference_kind": "stock-hf2q"}, "runs": runs}


def review(verdicts):
    return {"schema": quality_outcomes.PATCH_REVIEW_SCHEMA, "verdicts": verdicts}


class ReviewedTasks(unittest.TestCase):
    def test_reference_patches_also_need_review(self):
        document, pending = quality_outcomes.adjudicate(captured_tasks(), review({SUBJECT: "Pass"}))
        self.assertIsNone(document)
        self.assertEqual(pending, [REFERENCE])

    def test_protocol_success_with_rejected_patch_is_a_real_task_failure(self):
        document, pending = quality_outcomes.adjudicate(
            captured_tasks(), review({SUBJECT: "Fail", REFERENCE: "Pass"}))
        self.assertEqual(pending, [])
        verdict = campaign.paired(document)
        self.assertEqual((verdict["verdict"], verdict["subject_pass"], verdict["reference_pass"],
                          verdict["subject_only_fail"]), ("Loss", 0, 24, 24))

    def test_duplicate_cohort_cannot_replace_missing_paired_tasks(self):
        manifest = captured_tasks()
        manifest["runs"][1] = manifest["runs"][0]
        with self.assertRaises(ValueError):
            quality_outcomes.adjudicate(manifest, review({SUBJECT: "Pass", REFERENCE: "Pass"}))

    def test_release_gate_rejects_a_legacy_aggregate_pass(self):
        legacy = {"schema": "hf2q.adr063.quality-campaign.v1", "status": "Complete",
                  "plan": {"reference_kind": "stock-hf2q", "arms": {"subject": {"family": "qwen38"}}},
                  "verdict": {"verdict": "Pass"}}
        with self.assertRaises(ValueError):
            f2_q3_tier.gate_status(legacy, "stock-hf2q", "qwen38", {})


if __name__ == "__main__":
    unittest.main()
