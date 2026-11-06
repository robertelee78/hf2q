"""Model-free checks for the Qwen cohort replay (#717): stall detection, k, validity."""
import json
import os
import tempfile
import unittest

from qwen_cohort_replay import (MANIFEST_SCHEMA, analyze, assemble, cached_map, cached_mismatches,
                                event_kind, joint_turn_lines, joint_window, load_manifest,
                                output_digest)

MS = 1_000_000


def write_json(path, value):
    with open(path, "w") as f:
        json.dump(value, f)


def tokens(start_ms, count, step_ms):
    return [((start_ms + i * step_ms) * MS, "reasoning") for i in range(count)]


class JointWindowTest(unittest.TestCase):
    def test_counts_companion_tokens_after_the_prefill_stall_until_the_target_ends(self):
        # Companion streams 5 tokens at 30 ms, stalls 800 ms for the target's prefill, then shares
        # 7 joint steps at 70 ms before its token budget ends.
        events = tokens(0, 5, 30) + tokens(920, 7, 70) + [(1500 * MS, "other")]
        window = joint_window(events, target_send_ns=125 * MS, target_done_ns=5000 * MS)
        self.assertEqual(window["k"], 7)
        self.assertEqual(window["pre"], 5)
        self.assertEqual(window["stall_ns"], 800 * MS)

    def test_joint_steps_end_when_the_target_finishes(self):
        events = tokens(0, 4, 30) + tokens(890, 20, 70)
        window = joint_window(events, target_send_ns=95 * MS, target_done_ns=(890 + 9 * 70) * MS)
        self.assertEqual(window["k"], 10)

    def test_no_gap_after_the_target_means_no_joint_steps(self):
        self.assertEqual(joint_window(tokens(0, 4, 30), 500 * MS, 900 * MS)["k"], 0)
        self.assertEqual(joint_window([], 0, 1)["k"], 0)


class ValidityTest(unittest.TestCase):
    def test_equal_cached_tokens_across_arms_is_valid(self):
        solo = {"warm/agent-0/turn-1": 725, "test/agent-1/turn-4": 1961}
        joint = dict(solo, **{"test/companion": 1836})
        self.assertEqual(cached_mismatches([("solo", solo), ("joint", joint)]), [])

    def test_a_differing_resume_point_is_reported(self):
        a = {"test/agent-1/turn-4": 1961}
        b = {"test/agent-1/turn-4": 1956}
        self.assertEqual(cached_mismatches([("a", a), ("b", b)]),
                         [("b", "test/agent-1/turn-4", (1961, 1956))])

    def test_missing_usage_is_reported(self):
        self.assertEqual(cached_mismatches([("j", {"test/companion": None})]),
                         [("j", "test/companion", None)])

    def test_cached_map_reads_usage_details(self):
        records = {"a": {"usage": {"prompt_tokens_details": {"cached_tokens": 720}}}, "b": {"usage": None}}
        self.assertEqual(cached_map(records), {"a": 720, "b": None})


class ParsingTest(unittest.TestCase):
    def test_sse_event_kinds_and_assembly(self):
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                  {"choices": [{"index": 0, "delta": {"reasoning_content": "Think"}}]},
                  {"choices": [{"index": 0, "delta": {"content": "Done."}}]},
                  {"choices": [{"index": 0, "delta": {"tool_calls": [
                      {"index": 0, "function": {"name": "run_tests", "arguments": "{}"}}]}}]},
                  {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                   "usage": {"prompt_tokens_details": {"cached_tokens": 1836}}}]
        lines = [f"data: {json.dumps(c)}".encode() for c in chunks] + [b"data: [DONE]"]
        self.assertEqual([event_kind(line) for line in lines],
                         ["other", "reasoning", "content", "tool", "other", None])
        got = assemble(b"\n\n".join(lines))
        self.assertEqual((got["reasoning"], got["content"], got["tool_calls"], got["finish"]),
                         ("Think", "Done.", [["run_tests", "{}"]], "tool_calls"))
        self.assertEqual(got["usage"]["prompt_tokens_details"]["cached_tokens"], 1836)

    def test_digest_ignores_everything_but_the_assistant_output(self):
        a = {"reasoning": "r", "content": "c", "tool_calls": [["f", "{}"]], "usage": {"x": 1}}
        b = dict(a, usage={"x": 2}, timing={"t": 3})
        self.assertEqual(output_digest(a), output_digest(b))
        self.assertNotEqual(output_digest(a), output_digest(dict(a, reasoning="r2")))

    def test_joint_turn_lines(self):
        text = ("INFO x\nDEBUG hf2q: Qwen joint decode turn lanes=1 rows=1 serial=0\n"
                "DEBUG hf2q: Qwen joint decode turn lanes=2 rows=2 serial=0\n"
                "DEBUG hf2q: Qwen joint decode turn lanes=1 rows=1 serial=0\n")
        self.assertEqual(joint_turn_lines(text), {"lanes=1 rows=1 serial=0": 2, "lanes=2 rows=2 serial=0": 1})


class AnalyzeTest(unittest.TestCase):
    def arm(self, root, name, reasoning, cached, companion_events=None, logprobs=False):
        arm_dir = os.path.join(root, name)
        os.makedirs(os.path.join(arm_dir, "client"))
        target = {"reasoning": reasoning, "content": "c", "tool_calls": [], "send_ns": 100 * MS,
                  "done_ns": 9000 * MS, "events": [], "timing": {"decode_tokens_per_sec": 20.0},
                  "usage": {"prompt_tokens_details": {"cached_tokens": cached}}}
        records = {"test/agent-1/turn-4": target}
        if companion_events is not None:
            records["test/companion"] = {"events": companion_events, "reasoning": "", "content": "",
                                         "tool_calls": [], "usage": {"prompt_tokens_details": {"cached_tokens": 1836}}}
        summary = {"manifest": self.manifest, "arm": name, "logprobs": logprobs, "error": None, "records": records}
        write_json(os.path.join(arm_dir, "client", "summary.json"), summary)
        return arm_dir

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        ref = os.path.join(self.tmp.name, "ref.json")
        write_json(ref, {"role": "assistant", "reasoning_content": "to check", "content": "c"})
        self.manifest = {"schema": MANIFEST_SCHEMA, "target": {"agent": 1, "turn": 4},
                         "references": {"TO": ref}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_arms_are_byte_identical_and_classified(self):
        root = self.tmp.name
        dirs = [self.arm(root, "solo", "to check", 1961),
                self.arm(root, "joint13", "to check", 1961, tokens(0, 5, 30) + tokens(900, 13, 70))]
        report = analyze(dirs)
        self.assertTrue(report["valid"])
        self.assertTrue(report["byte_identical"])
        self.assertEqual([row["class"] for row in report["arms"]], ["TO", "TO"])
        self.assertEqual(report["arms"][1]["joint_window"]["k"], 13)

    def test_route_dependent_arms_are_reported(self):
        root = self.tmp.name
        dirs = [self.arm(root, "solo", "and check", 1961), self.arm(root, "all", "to check", 1961, [])]
        report = analyze(dirs)
        self.assertTrue(report["valid"])
        self.assertFalse(report["byte_identical"])
        self.assertEqual(report["distinct_outputs"], 2)
        self.assertEqual([row["class"] for row in report["arms"]], ["other", "TO"])

    def test_differing_resume_points_invalidate_the_comparison(self):
        root = self.tmp.name
        report = analyze([self.arm(root, "a", "to check", 1961), self.arm(root, "b", "to check", 1956)])
        self.assertFalse(report["valid"])

    def test_manifest_schema_is_checked(self):
        path = os.path.join(self.tmp.name, "m.json")
        write_json(path, {"schema": "other"})
        with self.assertRaises(SystemExit):
            load_manifest(path)


if __name__ == "__main__":
    unittest.main()
