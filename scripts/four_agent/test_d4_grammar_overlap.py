"""Isolated synthetic edge cases; these fixtures are not actual T2 evidence."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d4_grammar_overlap as overlap


def attempt(outcome="planning_consumed", *, awaiting=False, eos=False):
    return {"awaiting_trigger": awaiting, "eos": eos, "outcome": outcome}


def lane(entries, *, actual=None, awaiting=False):
    return {"actual_proposals": len(entries) if actual is None else actual,
            "awaiting_trigger_at_start": awaiting,
            "attempts": entries + [None] * (7 - len(entries))}


def command(label, commit, gpu):
    return {"label": label, "encode_begin": commit - 0.1,
            "commit_begin": commit, "commit_end": commit + 0.05,
            "wait_begin": 0, "wait_end": 0,
            "kernel_start": commit, "kernel_end": commit + 0.01,
            "gpu_start": gpu[0], "gpu_end": gpu[1],
            "dispatches_at_commit": 0, "barriers_at_commit": 0,
            "completed": True, "waited_in_drain": False}


def cycle(work=None, *, legacy=False, no_callback=False, sync=False, head=(6.25, 6.75), body=(4.5, 5)):
    if work is None:
        work = [lane([attempt()]), None, None, None]
    marks = [["begin", 1], ["verify_begin", 2]]
    if not no_callback:
        marks += [["joint.head_profile_sync_no_overlap" if sync else "joint.head_submitted", 3.5],
                  ["joint.host_work_begin", 4], ["grammar.simulate_begin", 4.1]]
        cursor = 5
        for slot, value in enumerate(work):
            if value is None:
                continue
            marks.append([f"grammar.slot{slot}.simulate_begin", cursor])
            cursor += 1
            for entry in value["attempts"]:
                if entry is not None:
                    marks += [["grammar.proposal_begin", cursor], ["grammar.proposal_end", cursor + 1]]
                    cursor += 2
            marks.append([f"grammar.slot{slot}.simulate_end", cursor])
            cursor += 1
        marks += [["grammar.simulate_end", cursor], ["joint.host_work_end", cursor + 1]]
    marks += [["joint.head_completed", 80], ["verify_end", 81], ["end", 82]]
    if legacy:
        marks = [mark for mark in marks if not mark[0].startswith("grammar.slot")
                 and not mark[0].startswith("grammar.proposal")]
    result = {"finished": True, "active_lanes": 4, "selected_rows": 32, "accepted_rows": 4,
              "gpu_acceptance": False, "dropped_commands": 0, "dropped_marks": 0,
              "marks": marks,
              "commands": [command(overlap.LAYERS, 2.5, body), command(overlap.HEAD, 3.2, head)]}
    if not legacy:
        result.update(grammar_work_schema=overlap.WORK_SCHEMA,
                      grammar_work=[None] * 4 if no_callback else work)
    return result


class GrammarOverlapTests(unittest.TestCase):
    @contextlib.contextmanager
    def input_file(self, records=None, raw=None):
        if raw is None:
            raw = b"".join((json.dumps(record) + "\n").encode() for record in records)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.jsonl"
            path.write_bytes(raw)
            yield path, hashlib.sha256(raw).hexdigest()

    def analyze(self, record, scope="head"):
        with self.input_file([record]) as (path, digest):
            return overlap.analyze(path, expected_sha256=digest, scope=scope)

    def rejected(self, record):
        with self.assertRaises(ValueError):
            self.analyze(record)

    def work(self, record, scope="head"):
        return self.analyze(record, scope)["cycles"][0]["grammar_work"]

    def test_active_consumed_uses_only_its_proposal(self):
        work = self.work(cycle())
        self.assertTrue(work["useful_work_eligible"])
        self.assertTrue(work["active_consumed_overlap_observed"])
        self.assertEqual(work["active_consumed_overlap_seconds"], 0.5)
        self.assertEqual(work["by_state_and_outcome"]["active"]["planning_consumed"]["attempts"], 1)

    def test_actual_proposals_fit_selected_rows_after_anchors(self):
        record = cycle()
        record.update(active_lanes=1, selected_rows=1, accepted_rows=1)
        self.rejected(record)
        record.update(selected_rows=2, accepted_rows=2)
        self.assertTrue(self.work(record)["active_consumed_overlap_observed"])

    def test_observed_grammar_lanes_do_not_exceed_active_lanes(self):
        record = cycle([lane([attempt()]), lane([attempt()]), None, None])
        record.update(active_lanes=1, selected_rows=8, accepted_rows=2)
        self.rejected(record)

    def test_selected_rows_do_not_exceed_per_lane_capacity(self):
        record = cycle()
        record.update(active_lanes=1, selected_rows=9, accepted_rows=1)
        self.rejected(record)

    def test_global_callback_and_envelope_gap_are_not_consumed_overlap(self):
        record = cycle(head=(5.25, 5.75))
        record["commands"].append(command(overlap.HEAD, 3.3, (7.25, 7.75)))
        row = self.analyze(record)["cycles"][0]
        self.assertEqual(row["grammar_callback_overlap_seconds"], 1)
        self.assertEqual(row["gpu_union_seconds"], 1)
        self.assertEqual(row["grammar_work"]["active_consumed_overlap_seconds"], 0)
        self.assertTrue(row["grammar_work"]["useful_work_eligible"])
        self.assertFalse(row["grammar_work"]["active_consumed_overlap_observed"])

    def test_overlapping_commands_are_unioned_not_added(self):
        record = cycle(head=(6, 6.75))
        record["commands"].append(command(overlap.HEAD, 3.3, (6.25, 7)))
        row = self.analyze(record)["cycles"][0]
        self.assertEqual(row["gpu_union_seconds"], 1)
        self.assertEqual(row["grammar_work"]["active_consumed_overlap_seconds"], 1)

    def test_awaiting_prepared_unused_invalid_and_discarded_are_not_active_consumed(self):
        variants = [lane([attempt(awaiting=True)], awaiting=True),
                    lane([attempt("prepared")]), lane([attempt("invalid")]),
                    lane([attempt("divergence_discarded")])]
        for value in variants:
            with self.subTest(value=value):
                work = self.work(cycle([value, None, None, None]))
                self.assertFalse(work["useful_work_eligible"])
                self.assertFalse(work["active_consumed_overlap_observed"])
                self.assertEqual(work["active_consumed_overlap_seconds"], 0)
                state = "awaiting_trigger" if value["awaiting_trigger_at_start"] else "active"
                outcome = value["attempts"][0]["outcome"]
                measured = work["by_state_and_outcome"][state][outcome]
                self.assertEqual(measured["attempts"], 1)
                self.assertEqual(measured["gpu_overlap_seconds"], 0.5)

    def test_awaiting_trigger_on_same_token_is_conservative(self):
        entries = [attempt(awaiting=True), attempt()]
        record = cycle([lane(entries, awaiting=True), None, None, None])
        work = self.work(record)
        self.assertTrue(work["useful_work_eligible"])
        self.assertEqual(work["active_consumed_overlap_seconds"], 0)
        self.assertEqual(work["by_state_and_outcome"]["awaiting_trigger"]["planning_consumed"]["gpu_overlap_seconds"], 0.5)
        record["commands"][-1].update(gpu_start=8.25, gpu_end=8.75)
        self.assertEqual(self.work(record)["active_consumed_overlap_seconds"], 0.5)

    def test_other_proposal_and_physical_lane_cannot_supply_overlap(self):
        variants = [
            [lane([attempt(), attempt("prepared")]), None, None, None],
            [lane([attempt(), attempt("divergence_discarded")]), None, None, None],
            [lane([attempt()]), None, None, lane([attempt("prepared")])],
        ]
        for work in variants:
            with self.subTest(work=work):
                record = cycle(work, head=(10.25, 10.75) if work[3] else (8.25, 8.75))
                measured = self.work(record)
                self.assertTrue(measured["useful_work_eligible"])
                self.assertFalse(measured["active_consumed_overlap_observed"])
                self.assertEqual(measured["active_consumed_overlap_seconds"], 0)

    def test_active_consumed_spans_union_excludes_unused_gap(self):
        record = cycle([lane([attempt(), attempt()]), None, None, None], head=(6, 9))
        work = self.work(record)
        self.assertEqual(work["active_consumed_overlap_seconds"], 2)
        self.assertEqual(work["by_state_and_outcome"]["active"]["planning_consumed"]["span_seconds"], 2)

    def test_invalid_and_eos_allow_unattempted_tail(self):
        for terminal in (attempt("invalid"), attempt(eos=True), attempt("prepared", eos=True)):
            with self.subTest(terminal=terminal):
                work = self.work(cycle([lane([attempt(), terminal], actual=7), None, None, None]))
                self.assertEqual(work["actual_proposals"], 7)
                self.assertEqual(work["attempts"], 2)

    def test_empty_actual_vector_and_grammar_free_cycle_are_no_work(self):
        for record in (cycle([lane([]), None, None, None]), cycle(no_callback=True)):
            with self.subTest(record=record):
                work = self.work(record)
                self.assertEqual(work["actual_proposals"], 0)
                self.assertEqual(work["attempts"], 0)
                self.assertFalse(work["useful_work_eligible"])
                self.assertFalse(work["active_consumed_overlap_observed"])

    def test_legacy_intervals_are_valid_but_useful_evidence_is_missing(self):
        report = self.analyze(cycle(legacy=True))
        row = report["cycles"][0]
        self.assertTrue(report["interval_capture_valid"])
        self.assertEqual(row["grammar_callback_overlap_seconds"], 0.5)
        self.assertEqual(row["grammar_work"]["metadata_status"], "missing")
        self.assertFalse(row["grammar_work"]["useful_work_eligible"])
        self.assertEqual(report["summary"]["legacy_missing_metadata_cycles"], 1)
        self.assertFalse(any(report["claims"].values()))

    def test_async_head_zero_and_body_positive_are_distinct_scopes(self):
        record = cycle(head=(20, 21), body=(6.25, 6.75))
        head, body = self.analyze(record, "head"), self.analyze(record, "target-verifier-body")
        self.assertEqual(head["summary"]["active_consumed_overlap_cycles"], 0)
        self.assertEqual(body["summary"]["active_consumed_overlap_cycles"], 1)
        self.assertEqual(head["cycles"][0]["selected_gpu_commands"], 1)
        self.assertEqual(body["cycles"][0]["selected_gpu_commands"], 2)

    def test_sync_is_explicit_ineligibility_not_async_pass(self):
        record = cycle(sync=True, head=(3.25, 3.4), body=(2.6, 3))
        row = self.analyze(record)["cycles"][0]
        self.assertTrue(row["interval_capture_valid"])
        self.assertTrue(row["synchronous_head_profile"])
        self.assertFalse(row["grammar_work"]["useful_work_eligible"])
        self.assertIn("explicit_synchronous_head_profile", row["grammar_work"]["ineligibility_reasons"])
        self.assertEqual(row["grammar_work"]["active_consumed_overlap_seconds"], 0)

    def test_draft_other_cycle_and_encode_outside_verify_are_excluded(self):
        record = cycle(head=(20, 21), body=(22, 23))
        record["commands"].insert(0, command(overlap.HEAD, 1.5, (6.25, 6.75)))
        record["commands"] += [command("qwen35.draft.output_head", 3.3, (6.25, 6.75)),
                               command("qwen35.joint_decode.layers.replay", 3.4, (6.25, 6.75)),
                               command(overlap.LAYERS, 82.5, (83, 84))]
        outside_encode = command(overlap.LAYERS, 2.6, (6.25, 6.75))
        outside_encode["encode_begin"] = 1.9
        record["commands"].insert(2, outside_encode)
        for scope in overlap.SCOPES:
            with self.subTest(scope=scope):
                row = self.analyze(record, scope)["cycles"][0]
                self.assertEqual(row["selected_gpu_commands"], 1 if scope == "head" else 2)
                self.assertEqual(row["grammar_work"]["active_consumed_overlap_seconds"], 0)

    def test_missing_body_layers_cannot_be_called_body_scope(self):
        record = cycle()
        record["commands"] = record["commands"][1:]
        with self.assertRaises(ValueError):
            self.analyze(record, "target-verifier-body")
        self.assertTrue(self.analyze(record, "head")["interval_capture_valid"])

    def test_proposal_markers_must_match_attempts_and_lanes(self):
        mutations = [
            lambda r: r["marks"].pop(next(i for i, m in enumerate(r["marks"]) if m[0] == "grammar.proposal_end")),
            lambda r: r["marks"].insert(7, ["grammar.proposal_begin", 6]),
            lambda r: r["marks"].__setitem__(8, ["grammar.slot1.simulate_end", 8]),
            lambda r: r["marks"].__setitem__(5, ["grammar.slot4.simulate_begin", 5]),
            lambda r: r["grammar_work"].__setitem__(1, r["grammar_work"][0]),
            lambda r: r["grammar_work"].__setitem__(0, None),
            lambda r: r.pop("grammar_work"),
            lambda r: r.pop("grammar_work_schema"),
            lambda r: r.__setitem__("grammar_work_schema", "other"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                record = cycle()
                mutate(record)
                self.rejected(record)

    def test_metadata_prefix_stop_state_and_disposition_rules(self):
        invalid = [
            lane([attempt(), attempt()], actual=1),
            lane([attempt("prepared")], actual=2),
            lane([attempt("invalid"), attempt()]),
            lane([attempt(eos=True), attempt()]),
            lane([attempt("prepared"), attempt()]),
            lane([attempt("divergence_discarded"), attempt()]),
            lane([attempt("prepared"), attempt("divergence_discarded")]),
            lane([attempt("divergence_discarded"), attempt("prepared")]),
            lane([attempt(awaiting=True)]),
            lane([attempt(), attempt(awaiting=True)]),
            lane([], actual=1),
        ]
        for value in invalid:
            with self.subTest(value=value):
                self.rejected(cycle([value, None, None, None]))
        record = cycle()
        record["grammar_work"][0]["attempts"] = [None, attempt()] + [None] * 5
        self.rejected(record)

    def test_strict_metadata_types_and_array_shapes(self):
        paths = [
            ("grammar_work", 0, "actual_proposals", True),
            ("grammar_work", 0, "actual_proposals", 1.0),
            ("grammar_work", 0, "actual_proposals", 8),
            ("grammar_work", 0, "awaiting_trigger_at_start", 0),
            ("grammar_work", 0, "attempts", []),
            ("grammar_work", 0, "attempts", [attempt()] * 8),
        ]
        for keys in paths:
            with self.subTest(keys=keys):
                record = cycle()
                node = record
                for key in keys[:-2]:
                    node = node[key]
                node[keys[-2]] = keys[-1]
                self.rejected(record)
        for key, value in (("awaiting_trigger", 0), ("eos", "false"), ("outcome", "accepted"), ("extra", 1)):
            record = cycle()
            record["grammar_work"][0]["attempts"][0][key] = value
            self.rejected(record)
        for value in ([], [None] * 3, {}, True):
            record = cycle()
            record["grammar_work"] = value
            self.rejected(record)

    def test_malformed_finished_summary_drops_and_command_types(self):
        for key, values in {
            "finished": [False, 1, None], "active_lanes": [True, 0, 5, 1.0],
            "selected_rows": [True, -1, 33], "accepted_rows": [True, -1, 33],
            "gpu_acceptance": [0, "false"], "dropped_commands": [True, -1, 1, 0.0],
            "dropped_marks": [False, -1, 1, 0.0],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    record = cycle()
                    record[key] = value
                    self.rejected(record)
        for key, values in {
            "completed": [False, 1], "waited_in_drain": [0], "label": [True, []],
            "gpu_start": [True, 0, -1, float("nan"), float("inf"), "6"],
            "gpu_end": [False, 6.25, float("inf")],
            "commit_begin": [True, 4], "commit_end": [3.1], "encode_begin": [4],
            "kernel_start": [float("nan"), True, 9], "wait_begin": [True, 4],
            "dispatches_at_commit": [True, -1, 1.0, 2**64], "barriers_at_commit": [False, -1],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    record = cycle()
                    record["commands"][-1][key] = value
                    self.rejected(record)

    def test_marker_clock_order_pairs_and_head_completion_are_required(self):
        records = []
        for value in (True, float("inf"), -1, 0, "6"):
            record = cycle()
            record["marks"][6][1] = value
            records.append(record)
        for name in ("begin", "verify_begin", "verify_end", "end", "joint.head_completed",
                     "joint.host_work_end", "grammar.simulate_end", "joint.head_submitted"):
            record = cycle()
            record["marks"] = [mark for mark in record["marks"] if mark[0] != name]
            records.append(record)
        record = cycle()
        record["marks"][6], record["marks"][7] = record["marks"][7], record["marks"][6]
        records.append(record)
        record = cycle()
        record["marks"].insert(2, ["verify_begin", 2.1])
        records.append(record)
        record = cycle(head=(79, 80.5))
        records.append(record)
        for record in records:
            with self.subTest(record=record):
                self.rejected(record)

    def test_cross_record_cycles_and_cumulative_counters_cannot_regress(self):
        first = cycle()
        second = copy.deepcopy(first)
        for mark in second["marks"]:
            mark[1] += 100
        for cmd in second["commands"]:
            for key in ("encode_begin", "commit_begin", "commit_end", "kernel_start", "kernel_end", "gpu_start", "gpu_end"):
                cmd[key] += 100
        first["commands"][-1]["dispatches_at_commit"] = 10
        for cmd in second["commands"]:
            cmd["dispatches_at_commit"] = 10
        with self.input_file([first, second]) as (path, digest):
            report = overlap.analyze(path, expected_sha256=digest, scope="head")
            self.assertEqual(report["summary"]["active_consumed_overlap_cycles"], 2)
        second["commands"][0]["dispatches_at_commit"] = 9
        with self.input_file([first, second]) as (path, digest):
            with self.assertRaises(ValueError):
                overlap.analyze(path, expected_sha256=digest, scope="head")
        with self.input_file([cycle(), cycle()]) as (path, digest):
            with self.assertRaises(ValueError):
                overlap.analyze(path, expected_sha256=digest, scope="head")

    def test_hash_binds_original_bytes_not_reencoded_json(self):
        raw = json.dumps(cycle(), indent=None).encode() + b"\r\n"
        with self.input_file(raw=raw) as (path, digest):
            report = overlap.analyze(path, expected_sha256=digest.upper(), scope="head")
            self.assertEqual(report["input"]["sha256"], digest)
            self.assertEqual(report["input"]["original_bytes"], len(raw))
            for wrong in ("0" * 64, "", True, "g" * 64):
                with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                    overlap.analyze(path, expected_sha256=wrong, scope="head")

    def test_no_malformed_record_skipping_or_duplicate_json_keys(self):
        valid = json.dumps(cycle()).encode()
        for raw in (b"", b"\n", valid + b"\n\n", valid + b"\nnot-json\n", b"[]\n", b"\xff\n",
                    valid.replace(b'"finished": true', b'"finished": true, "finished": true')):
            with self.subTest(raw=raw[:30]), self.input_file(raw=raw) as (path, digest):
                with self.assertRaises(ValueError):
                    overlap.analyze(path, expected_sha256=digest, scope="head")

    def test_original_byte_and_record_command_mark_limits(self):
        record = cycle()
        limits = {"MAX_FILE_BYTES": 10, "MAX_LINE_BYTES": 10, "MAX_RECORDS": 0,
                  "MAX_COMMANDS_PER_RECORD": 1, "MAX_MARKS_PER_RECORD": 2,
                  "MAX_COMMANDS": 1, "MAX_MARKS": 2}
        for name, value in limits.items():
            with self.subTest(name=name), patch.object(overlap, name, value):
                self.rejected(record)

    def test_cli_requires_scope_and_hash_and_emits_json_only_on_success(self):
        with self.input_file([cycle()]) as (path, digest):
            with contextlib.redirect_stderr(io.StringIO()):
                for args in ([str(path)], [str(path), "--expected-sha256", digest],
                             [str(path), "--scope", "head"]):
                    with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                        overlap.main(args)
                    self.assertNotEqual(error.exception.code, 0)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = overlap.main([str(path), "--expected-sha256", digest, "--scope", "head"])
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["scope"], "head")
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = overlap.main([str(path), "--expected-sha256", "0" * 64, "--scope", "head"])
            self.assertNotEqual(result, 0)
            self.assertEqual(stdout.getvalue(), "")
            self.assertTrue(stderr.getvalue())

    def test_driver_scheduling_times_are_not_gpu_intervals(self):
        record = cycle(head=(20, 21))
        record["commands"][-1].update(kernel_start=6, kernel_end=7)
        self.assertEqual(self.work(record)["active_consumed_overlap_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
