"""Q4-TRACE overhead extractor: one joint-epoch rule for the hook (T1) and parent (T0) traces."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import q4_trace_overhead as overhead
import q4_trace_verdict as verdict
from test_f2_native_trace_joint import report, widths


class OverheadExtractor(unittest.TestCase):
    def test_parent_shape_finds_the_same_joint_epochs_and_spans_by_transaction_order(self):
        records, identity, scenario = widths(pre_aux=True)
        reduced = report(records, identity, scenario)
        hook = overhead.scan(records)
        parent = overhead.scan(list(verdict.PreHook(records)))
        self.assertEqual((hook["joint_epochs"], hook["incomplete_epochs"]), (4, 0))
        self.assertEqual(hook["spans_ns"], parent["spans_ns"])
        expected = {}
        for forward in reduced["joint_ordinary_decode"]["forwards"]:
            begin, end = forward["epoch_span_ns"]
            expected[forward["logical_rows"]] = [end - begin]
        self.assertEqual(hook["spans_ns"], expected)  # the reducer's independent epoch span
        result = overhead.compare(hook, parent)
        self.assertEqual(result["records_by_kind_delta"]["forward_input"], 4)
        self.assertEqual(result["records_by_kind_delta"]["forward_witness"], 4)
        self.assertEqual(result["records_by_kind_delta"]["forward_input_end"], 4)
        self.assertEqual(result["records_delta_per_joint_epoch"], 3.0)
        for row in result["by_logical_width"].values():
            self.assertEqual((row["median_delta_ns"], row["median_delta_ci95_ns"]), (0, [0, 0]))

    def test_bootstrap_ci_brackets_a_known_shift_and_is_seeded(self):
        right = {"joint_epochs": 50, "incomplete_epochs": 0, "records_by_kind": {},
                 "spans_ns": {4: [1000 + 7 * i for i in range(50)]}}
        left = dict(right, spans_ns={4: [value + 30 for value in right["spans_ns"][4]]})
        first = overhead.compare(left, right)
        row = first["by_logical_width"]["logical4"]
        self.assertEqual((row["median_delta_ns"], row["p95_delta_ns"]), (30, 30))
        low, high = row["median_delta_ci95_ns"]
        self.assertLessEqual(low, 30)
        self.assertGreaterEqual(high, 30)
        self.assertEqual(first, overhead.compare(left, right))
        self.assertNotIn("median_delta_ns", overhead.compare(left, dict(right, spans_ns={}))
                         ["by_logical_width"]["logical4"])

    def test_cli_reports_both_arms_split_and_makespan(self):
        records, identity, scenario = widths()
        reduced = report(records, identity, scenario)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = {}
            for name, items in (("t1", records), ("t0", list(verdict.PreHook(records)))):
                paths[name] = root / f"{name}.jsonl"
                paths[name].write_text("".join(json.dumps(r) + "\n" for r in items))
            (root / "windows.json").write_text(json.dumps({"validate_trace": reduced}))
            for name, seconds in (("s1", 101.5), ("s0", 100.0)):
                (root / f"{name}.json").write_text(json.dumps({"own_output_four_task_makespan_seconds": seconds}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = overhead.main([str(paths["t1"]), str(paths["t0"]), "--left-windows", str(root / "windows.json"),
                                      "--left-summary", str(root / "s1.json"), "--right-summary", str(root / "s0.json")])
            self.assertEqual(code, 0)
            result = json.loads(out.getvalue())
            self.assertGreater(result["trace_bytes"]["hook"], result["trace_bytes"]["parent"])
            self.assertEqual(result["makespan_s"]["delta"], 1.5)
            self.assertEqual(result["hook_forward_settle"]["logical4"]["forward"]["n"], 1)
            self.assertEqual(set(result["by_logical_width"]), {"logical1", "logical2", "logical3", "logical4"})
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(overhead.main([str(paths["t1"]), str(paths["t0"]),
                                                "--left-summary", str(root / "s1.json")]), 2)


if __name__ == "__main__":
    unittest.main()
