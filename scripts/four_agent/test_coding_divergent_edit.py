#!/usr/bin/env python3
import json
from pathlib import Path
import unittest

import coding_divergent_edit as divergent
import q5_case
import quality_campaign


class DivergentEditTests(unittest.TestCase):
    def test_turns_three_and_five_edit_the_t_minus_two_tool_result_once(self):
        variant = divergent.select()
        messages = []
        originals = []
        for turn in range(6):
            edit = variant.before_request(messages, turn)
            if turn in divergent.TURNS:
                self.assertEqual(edit["target_turn"], turn - 2)
                target = [message for message in messages if message["role"] == "tool"][turn - 2]
                self.assertEqual(target["content"].splitlines()[-1], variant.marker)
                self.assertEqual(len(target["content"].splitlines()), divergent.KEEP_LINES + 1)
            else:
                self.assertIsNone(edit)
            content = "\n".join(f"turn-{turn}-line-{line}" for line in range(24))
            originals.append(content)
            messages.append({"role": "assistant", "content": ""})
            messages.append({"role": "tool", "content": content})
        tools = [message for message in messages if message["role"] == "tool"]
        self.assertEqual(tools[0]["content"], originals[0])
        self.assertNotEqual(tools[1]["content"], originals[1])
        self.assertEqual(tools[2]["content"], originals[2])
        self.assertNotEqual(tools[3]["content"], originals[3])

    def test_estimate_records_bounded_wire_rows_and_both_edits(self):
        estimate = divergent.estimate()
        rows = estimate[divergent.ID]
        self.assertEqual([row["turn"] for row in rows], list(range(7)))
        self.assertEqual([row["turn"] for row in rows if row["edit"]], [3, 5])
        self.assertTrue(all(row["request_bytes"] <= estimate["wire_limit_bytes"] for row in rows))

    def test_derived_case_binds_variant_fixture_and_sources(self):
        path = Path(__file__).with_name("q5_cases") / "qwen38.json"
        case = q5_case.loads(path.read_bytes())
        derived = divergent.derive_case(case)
        self.assertIn("--workload", derived["client"]["argv"])
        self.assertEqual(derived["client"]["approval"]["set"]["workload_variant"], divergent.ID)
        sources = derived["client"]["approval"]["set"]["sources"]["sha256_of"]
        self.assertTrue(any(source.endswith("coding_divergent_edit.py") for source in sources))
        q5_case.validate(json.loads(json.dumps(derived)))


        composed = quality_campaign.derive_case(derived, 0)
        self.assertEqual(composed["client"]["argv"][-4:],
                         ["--workload", divergent.ID, "--instance", "0"])
        self.assertEqual(composed["client"]["approval"]["set"]["instance"], 0)
if __name__ == "__main__":
    unittest.main()
