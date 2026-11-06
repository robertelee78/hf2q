"""Q4-TRACE attribution extractor on the synthetic joint fixtures (P1 shape); not hardware evidence."""
import json
import tempfile
import unittest
from pathlib import Path

import q4_trace_attribution as attribution
import q4_trace_verdict as verdict
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


if __name__ == "__main__":
    unittest.main()
