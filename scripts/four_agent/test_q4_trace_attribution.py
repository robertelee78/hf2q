"""Q4-TRACE attribution extractor on the synthetic joint fixtures (P1 shape); not hardware evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import q4_trace_attribution as attribution
import q4_trace_verdict as verdict
from test_f2_native_trace import record
from test_f2_native_trace_joint import report, widths

JOINT = "joint_ordinary_decode"


class Attribution(unittest.TestCase):
    def test_phases_split_and_kernels_follow_the_reducer(self):
        records, identity, scenario = widths()
        reduced = report(records, identity, scenario)[JOINT]
        result = attribution.attribute(records)
        self.assertEqual(set(result["phases"]), {f"{JOINT}/logical{n}" for n in (1, 2, 3, 4)})
        four = result["joint_forward_settle"][f"{JOINT}/logical4"]
        first = reduced["forwards"][0]
        self.assertEqual(four["forward"]["commands"]["sum"], first["forward"]["commands"])
        self.assertEqual(four["forward"]["dispatches"]["sum"], first["forward"]["dispatches"])
        self.assertEqual(four["settle"]["commands"]["sum"], first["settle"]["commands"])
        self.assertEqual(four["boundary_ambiguous"]["commands"]["sum"], 1)
        begin, end = first["epoch_span_ns"]
        self.assertEqual(result["phases"][f"{JOINT}/logical4"]["span_ns"]["sum"], end - begin)
        # The four serial lanes of the fixture run without a transaction: kept, but unattributed.
        self.assertEqual(result["unattributed"], {"commands": 4, "dispatches": 4})
        (kernel,) = result["kernels"]
        self.assertEqual(kernel["dispatches"], result["dispatches_total"])  # attributed epochs only
        self.assertEqual(kernel["dispatches"], sum(m["dispatches"]["sum"] for m in result["phases"].values()))
        self.assertEqual(kernel["by_phase"][f"{JOINT}/logical4:forward"], 3)
        self.assertEqual(kernel["first_joint_site"], {"command_index": 0, "dispatch_ordinal": 1})
        self.assertIn("pipeline names", result["not_observed"])

    def test_parent_shape_attributes_phases_without_a_split(self):
        records, _, _ = widths()
        hooked = attribution.attribute(records)
        plain = attribution.attribute(list(verdict.PreHook(records)))
        self.assertEqual(plain["joint_forward_settle"], {})
        self.assertEqual(plain["phases"], hooked["phases"])

    def test_cli_cross_checks_the_reducer_forward_count(self):
        records, identity, scenario = widths()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            (root / "w.json").write_text(json.dumps({"validate_trace": report(records, identity, scenario)}))
            import contextlib, io
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(attribution.main([str(root / "t.jsonl"), "--windows", str(root / "w.json")]), 0)
            self.assertTrue(json.loads(out.getvalue())["reducer_cross_check"]["agree"])


class GpuAttribution(unittest.TestCase):
    def test_gpu_busy_chains_refusal_and_bounded_launch(self):
        from test_f2_gpu_timing import gpu_trace
        records, _, _ = gpu_trace(widths)
        plan = {"model": "qwen35", "n_layers": 2, "chain_n": 2, "layer_kinds": ["linear", "full"],
                "commands": ["layers[0,2)", "output_head"]}
        records.insert(5, dict(records[5], kind="joint_chain_plan", payload=plan))
        result = attribution.attribute(records)
        four = result["phases"][f"{JOINT}/logical4"]
        self.assertEqual(four["gpu_busy_ns"]["n"], 1)
        self.assertEqual(four["command_gpu_ns"]["n"], 2)  # two drained commands; empty ones have no GPU time
        self.assertGreaterEqual(four["gpu_bubble_ns"]["sum"], 0)
        # Epoch 100 has five commands (not the plan's two): refused, never mapped.
        self.assertEqual(result["chain_map_refused"], 1)
        three = result["joint_chains"][f"{JOINT}/logical3"]
        self.assertEqual([c["command"] for c in three], ["layers[0,2)", "output_head"])
        self.assertEqual(three[0]["layers"], {"range": [0, 2], "kinds": ["linear", "full"]})
        self.assertIsNone(three[1]["layers"])
        self.assertEqual(three[0]["gpu_ns"]["n"], 1)
        self.assertNotIn(f"{JOINT}/logical4", result["joint_chains"])
        launch = result["launch"]
        self.assertTrue(launch["reported"])
        self.assertEqual(launch["bound_ns"], 250 + 8_000)  # read span + monotonic-mach offset drift
        self.assertIn("pipeline names", result["not_observed"])

    def test_launch_needs_both_clock_pairs(self):
        from test_f2_gpu_timing import gpu_trace
        records, _, _ = gpu_trace(widths)
        record(records, "native_final")["payload"].pop("clock_pair")
        self.assertFalse(attribution.attribute(records)["launch"]["reported"])


if __name__ == "__main__":
    unittest.main()
