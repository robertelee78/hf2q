"""Deterministic retained-client metric controls, not model qualification evidence."""
import copy
import math
import unittest

import c2_metrics as c2
from coding_divergent_edit import select
from f2_semantic import semantic_timing_v3
from q3_metrics import slot_metrics


def turn(identity, ttft=1, gap=0.1):
    stream = identity % 2 == 1
    send = 10_000_000_000
    events = [send + round(ttft * 1e9), send + round((ttft + gap) * 1e9)] if stream else []
    terminal = send + round((ttft + gap + 1) * 1e9)
    return {"turn": identity, "stream": stream, "semantic_event_monotonic_ns": events,
            "semantic_timing_v3": semantic_timing_v3(send, events, terminal, stream, True)}


def summary(divergent=True):
    return {"status": c2.SUCCESS, "own_output_four_task_makespan_seconds": 100,
            "workload_variant": select().record() if divergent else None,
            "agents": {name: {"agent_id": name, "status": c2.AGENT_SUCCESS,
                               "tools": [{} for _ in range(6)],
                               "turns": [turn(i) for i in range(7)]} for name in c2.AGENTS}}


class DivergentMetrics(unittest.TestCase):
    def test_all_streamed_pool_masks_the_actual_divergent_regression(self):
        reference, subject = summary(), summary()
        for agent in reference["agents"].values():
            agent["turns"][1] = turn(1, 100)
        for agent in subject["agents"].values():
            agent["turns"][1] = turn(1, 80)
            agent["turns"][3], agent["turns"][5] = turn(3, 2), turn(5, 1.5)
        self.assertLess(slot_metrics(subject["agents"]["agent-0"])["ttft_p95_s"],
                        slot_metrics(reference["agents"]["agent-0"])["ttft_p95_s"])
        extracted = c2.divergent_metrics(subject)
        self.assertEqual(extracted[c2.DIVERGENT] / c2.divergent_metrics(reference)[c2.DIVERGENT], 2)
        self.assertEqual(extracted["slots"]["agent-0"], {
            "turn_ttft_s": {"3": 2, "5": 1.5}, "divergent_turns": [3, 5], "divergent_ttft_max_s": 2})

    def test_turn_identity_not_position_and_worst_slot_not_pooled_mean(self):
        record = summary()
        record["agents"]["agent-2"]["turns"][5] = turn(5, 9)
        for agent in record["agents"].values():
            agent["turns"].reverse()
        value = c2.divergent_metrics(record)
        self.assertEqual(value[c2.DIVERGENT], 9)
        self.assertEqual(len(value["slots"]), 4)
        self.assertEqual(value["slots"]["agent-2"]["turn_ttft_s"], {"3": 1, "5": 9})

    def test_missing_and_duplicate_actual_divergent_turns_are_refused(self):
        for mode in ("missing", "duplicate", "indexless"):
            record = summary()
            turns = record["agents"]["agent-0"]["turns"]
            if mode == "missing":
                turns.pop(3)
            elif mode == "duplicate":
                turns.append(copy.deepcopy(turns[3]))
            else:
                del turns[3]["turn"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                c2.divergent_metrics(record)

    def test_no_unary_empty_role_or_nonfinite_ttft_substitutes(self):
        for field, value in (("stream", False), ("semantic_events", 0), ("schema", "old"),
                             ("semantic_ttft_seconds", None), ("semantic_ttft_seconds", 0),
                             ("semantic_ttft_seconds", math.inf), ("semantic_ttft_seconds", math.nan)):
            record = summary()
            record["agents"]["agent-0"]["turns"][3]["semantic_timing_v3"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                c2.divergent_metrics(record)

    def test_incomplete_cohort_failed_run_and_wrong_fixture_are_refused(self):
        for mode in ("cohort", "failed", "fixture"):
            record = summary()
            if mode == "cohort":
                del record["agents"]["agent-3"]
            elif mode == "failed":
                record["status"] = "Failed"
            else:
                record["workload_variant"]["fixture_sha256"] = "0" * 64
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                c2.divergent_metrics(record)

    def test_changed_edit_schedule_or_extent_is_refused(self):
        for field, value in (("edit_turns", [1, 3]), ("keep_lines", 8)):
            record = summary()
            record["workload_variant"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                c2.divergent_metrics(record)


class AppendMetrics(unittest.TestCase):
    def test_three_actual_append_estimands_and_slot_audit(self):
        record = summary(False)
        record["agents"]["agent-3"]["turns"][3] = turn(3, 4, 0.75)
        extracted = c2.append_metrics(record)
        self.assertEqual([extracted[name] for name in c2.APPEND], [100, 4, 0.75])
        self.assertEqual(extracted["slots"]["agent-3"]["streamed_turns"], 3)

    def test_missing_turn_or_events_cannot_improve_an_append_limit(self):
        for mode in ("turn", "tail", "events", "slot"):
            record = summary(False)
            agent = record["agents"]["agent-0"]
            if mode == "turn":
                agent["turns"].pop(3)
            elif mode == "tail":
                agent["turns"].pop()
            elif mode == "events":
                del agent["turns"][3]["semantic_event_monotonic_ns"]
            else:
                del record["agents"]["agent-0"]
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                c2.append_metrics(record)

    def test_declared_repair_turns_preserve_extra_streamed_tail(self):
        record = summary(False)
        agent = record["agents"]["agent-0"]
        agent["repair_turns"] = 1
        agent["tools"].extend([{}, {}])
        agent["turns"].extend([turn(7, 10, 0.5), turn(8)])
        result = c2.append_metrics(record)
        self.assertEqual(result["worst_slot_ttft_p95_s"], 10)
        self.assertEqual(result["slots"]["agent-0"]["streamed_turns"], 4)

    def test_repair_budget_cannot_hide_an_executed_slow_tail(self):
        record = summary(False)
        agent = record["agents"]["agent-0"]
        agent["repair_turns"] = 1
        agent["tools"].extend([{}, {}])
        with self.assertRaisesRegex(ValueError, "incomplete tool/turn coverage"):
            c2.append_metrics(record)

    def test_unused_repair_budget_does_not_require_unexecuted_turns(self):
        record = summary(False)
        record["agents"]["agent-0"]["repair_turns"] = 1
        result = c2.append_metrics(record)
        self.assertEqual(result["worst_slot_ttft_p95_s"], 1)
        self.assertEqual(result["slots"]["agent-0"]["streamed_turns"], 3)

    def test_missing_executed_tool_receipts_are_refused(self):
        record = summary(False)
        del record["agents"]["agent-0"]["tools"]
        with self.assertRaisesRegex(ValueError, "executed tool receipts required"):
            c2.append_metrics(record)


if __name__ == "__main__":
    unittest.main()
