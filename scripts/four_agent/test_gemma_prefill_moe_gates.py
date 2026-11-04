import json
import tempfile
import unittest
from pathlib import Path

import gemma_prefill_moe_gates as gates


def log(turns):
    """Server log of warm turns: each turn is (lanes, rows_per_lane, wall_ms) or a list of per-slot
    forwards [(rows, wall_ms), ...] (a split cohort)."""
    lines = ["Batched prefill: KV=F32, seq_len=458, path=flash-attn, globals=tensor-mm",
             "Batched prefill complete: 458 tokens in 190.0 ms (1.0 tok/s), first decode token = 1"]
    for turn in turns:
        if isinstance(turn, list):
            for rows, wall in turn:
                lines += [f"Batched prefill: KV=F32, seq_len={rows}, path=flash-attn, live=vec-hybrid",
                          f"Batched prefill complete: {rows} tokens in {wall} ms (1.0 tok/s), first decode token = 1"]
            continue
        lanes, rows, wall = turn
        lines += [f"Batched prefill: KV=F32, seq_len={lanes * rows}, path=flash-attn, live=vec-hybrid",
                  f"Batched prefill complete: {lanes * rows} tokens in {wall} ms (1.0 tok/s), first decode token = 1",
                  f' INFO x: transaction="joint warm" requests={lanes} advanced_tokens={lanes * rows} elapsed_ms=1']
        lines += [f" INFO x: Gemma4 bounded prefill transaction complete slot={s} advanced_tokens={rows} prompt_tokens=9"
                  for s in range(lanes)]
    return "\n".join(lines) + "\n"


def summary(makespan, ttft=1.0):
    turn = {"semantic_timing_v3": {"stream": True, "semantic_ttft_seconds": ttft},
            "semantic_event_monotonic_ns": [0, 10_000_000]}
    return {"own_output_four_task_makespan_seconds": makespan,
            "agents": {f"agent-{i}": {"turns": [turn]} for i in range(4)}}


class Gates(unittest.TestCase):
    def run_dir(self, root, name, turns, makespan):
        d = Path(root) / name
        (d / "client").mkdir(parents=True)
        (d / "server.stderr").write_text(log(turns))
        (d / "client" / "summary.json").write_text(json.dumps(summary(makespan)))
        return str(d)

    def test_parses_joint_and_split_transactions(self):
        joint = gates.warm_transactions(log([(4, 58, 137.0), [(355, 280.0), (355, 281.0)]]))
        self.assertEqual(gates.structure(joint), ((4, (58, 58, 58, 58)), (1, (355,)), (1, (355,))))
        self.assertEqual([t["wall_ms"] for t in joint], [137.0, 280.0, 281.0])

    def test_structure_matched_s2_pass_and_exclusion(self):
        with tempfile.TemporaryDirectory() as root:
            runs, order = [], "ABBA" * 4
            for i, arm in enumerate(order):
                wall = 1000.0 if arm == "A" else 900.0
                turns = [(4, 58, 137.0), (4, 355, wall)]
                if i == 0:
                    turns = [(4, 58, 137.0), [(355, 300.0)] * 4]
                runs.append(gates.load_run(arm, self.run_dir(root, f"r{i}", turns, 100.0 - (arm == "B"))))
            blocks = gates.blocks_of(runs)
            s2 = gates.s2(blocks)
            self.assertEqual(s2["matched_blocks"], 7)
            self.assertEqual(len(s2["excluded_runs"]), 1)
            self.assertFalse(s2["per_transaction_substitution"])
            self.assertAlmostEqual(s2["median_delta_s"], -0.1)
            self.assertTrue(s2["pass"])
            self.assertTrue(gates.s4(blocks)["pass"])
            self.assertTrue(all(v["pass"] for v in gates.fairness(blocks).values()))

    def campaign(self, root, blocks, split_blocks):
        """ABBA campaign of `blocks` blocks; the first run of each block in `split_blocks` splits."""
        runs = []
        for i, arm in enumerate(("ABBA" * blocks)[:2 * blocks]):
            wall = 1000.0 if arm == "A" else 920.0
            turns = [(4, 58, 137.0), (4, 355, wall)]
            if i % 2 == 0 and i // 2 in split_blocks:
                turns = [(4, 58, 137.0), [(355, 300.0)] * 4]
            runs.append(gates.load_run(arm, self.run_dir(root, f"r{i}", turns, 100.0)))
        return gates.blocks_of(runs)

    def test_fewer_than_six_matched_blocks_needs_more_pairs_before_the_cap(self):
        with tempfile.TemporaryDirectory() as root:
            s2 = gates.s2(self.campaign(root, 3, {0, 1}))
            self.assertFalse(s2["per_transaction_substitution"])
            self.assertFalse(s2["pass"])
            self.assertEqual(s2["deltas_s"], [])
            self.assertIn("need more pairs", s2["note"])

    def test_per_transaction_substitution_once_the_cap_is_used(self):
        with tempfile.TemporaryDirectory() as root:
            s2 = gates.s2(self.campaign(root, 12, set(range(7))))
            self.assertEqual(s2["matched_blocks"], 5)
            self.assertTrue(s2["per_transaction_substitution"])
            self.assertEqual(len(s2["deltas_s"]), 12)
            self.assertEqual(s2["deltas_s"][0], 0.0, "split block: only the matched 4x58 turn is summed")
            self.assertAlmostEqual(s2["deltas_s"][-1], -0.08)
            # 7 of 12 blocks only match on the unchanged 4x58 turn: median 0, gate fails.
            self.assertEqual(s2["median_delta_s"], 0.0)
            self.assertFalse(s2["pass"])

    def test_blocks_require_one_run_per_arm(self):
        run = {"arm": "A"}
        with self.assertRaises(ValueError):
            gates.blocks_of([run, dict(run)])


if __name__ == "__main__":
    unittest.main()
