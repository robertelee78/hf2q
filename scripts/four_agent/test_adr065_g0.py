import unittest

import adr065_g0 as g0


class Adr065G0Test(unittest.TestCase):
    def e2_receipt(self):
        samples = []
        for context in g0.REQUIRED_CONTEXTS:
            for shape in g0.REQUIRED_SHAPES:
                for repetition in range(8):
                    samples.append(
                        {
                            "context_tokens": context,
                            "shape": shape,
                            "warmup": repetition == 0,
                            "repetition": repetition,
                            "prepare_ns": 1,
                            "draft_ns": 2,
                            "verify_readback_ns": 3,
                            "selection_ns": 4,
                            "capture_recovery_ns": 5,
                            "settle_catch_up_ns": 6,
                            "total_ns": 100_000_000,
                        }
                    )
        return {
            "schema": "hf2q.adr065.joint-verify-timing.v1",
            "grid": "full",
            "warmups_per_configuration": 1,
            "timed_runs_per_configuration": 7,
            "samples": samples,
        }

    def test_full_grid_validation_requires_every_repetition(self):
        grouped, warmups = g0.validate_e2(self.e2_receipt())
        self.assertEqual(len(grouped), 27)
        self.assertEqual(len(warmups), 27)
        receipt = self.e2_receipt()
        receipt["samples"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            g0.validate_e2(receipt)

    def test_zero_eligible_fraction_forces_S_to_one(self):
        grouped, _ = g0.validate_e2(self.e2_receipt())
        clusters = [
            {"f": 0.0, "width_ns": {1: 1, 2: 2, 3: 3, 4: 4}},
            {"f": 0.0, "width_ns": {1: 2, 2: 3, 3: 4, 4: 5}},
            {"f": 0.0, "width_ns": {1: 3, 2: 4, 3: 5, 4: 6}},
        ]
        old_replicates = g0.BOOTSTRAP_REPLICATES
        g0.BOOTSTRAP_REPLICATES = 200
        try:
            s_values, f_values, S_values = g0.bootstrap(grouped, 3.5, [0.05] * 20, clusters)
        finally:
            g0.BOOTSTRAP_REPLICATES = old_replicates
        self.assertTrue(all(value > 0 for value in s_values))
        self.assertEqual(set(f_values), {0.0})
        self.assertEqual(set(S_values), {1.0})

    def test_expected_committed_includes_bonus_token(self):
        e3 = {
            "proposers": {
                "mtp_k3": {
                    "accepted_prefix_histogram": {"0": 1, "1": 1, "2": 1, "3": 1}
                }
            }
        }
        self.assertEqual(g0.expected_committed_tokens(e3), 2.5)


if __name__ == "__main__":
    unittest.main()
