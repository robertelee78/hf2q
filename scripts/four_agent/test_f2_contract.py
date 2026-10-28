"""Small actual-entrypoint controls. Hardware/source binding remains separate."""
import copy
import json
import unittest

import f2_scoreboard as score
import f2_semantic as timing
import f2_trace as trace
from qwen38_n1_verdict import SSE, Invalid


class SemanticControls(unittest.TestCase):
    def test_client_quota_is_sticky_across_turn_vectors(self):
        old = timing._events_reserved
        try:
            timing._events_reserved = timing.CLIENT_TRACE_BUDGET // 128
            first, second = [], []
            for events in (first, second):
                with self.assertRaises(ValueError):
                    timing.append_semantic_event(events, 1)
                self.assertEqual(events, [])
        finally:
            timing._events_reserved = old

    def test_actual_parser_name_counts_but_role_reasoning_do_not(self):
        parser = SSE("model", 0, set())
        def feed(delta):
            return parser.feed(json.dumps({"id": "one", "model": "model", "choices": [
                {"index": 0, "delta": delta, "finish_reason": None}]}))
        self.assertFalse(feed({"role": "assistant"}))
        self.assertFalse(feed({"reasoning_content": "thinking"}))
        self.assertTrue(feed({"tool_calls": [{"index": 0, "id": "tool", "type": "function",
                                              "function": {"name": "read_files"}}]}))
        with self.assertRaises(Invalid):
            parser.result()  # A name timestamp cannot qualify an incomplete tool.

    def test_end_to_end_gap_includes_terminal_and_send(self):
        result = timing.semantic_timing_v3(0, [3_000_000_000, 4_000_000_000],
                                          10_000_000_000, True, True)
        self.assertEqual(result["semantic_gap_end_to_end_seconds"], 6)
        self.assertEqual(result["semantic_gap_inter_events_seconds"], 1)
        self.assertEqual(result["semantic_ttft_seconds"], 3)

    def test_one_event_unary_and_invalid_terminal(self):
        one = timing.semantic_timing_v3(0, [1], 2, True, True)
        self.assertIsNone(one["semantic_gap_inter_events_seconds"])
        unary = timing.semantic_timing_v3(0, [], 2, False, True)
        self.assertIsNone(unary["semantic_ttft_seconds"])
        with self.assertRaises(ValueError):
            timing.semantic_timing_v3(0, [1], 2, True, False)
        with self.assertRaises(ValueError):
            timing.semantic_timing_v3(0, [], 2, True, True)


def trace_fixture(temporal=False):
    identity = {"fixture": "MOCK, no device or inferred runtime evidence"}
    header = {"identity": identity, "run_id":"MOCK", "server_budget_bytes": 128*1024*1024,
              "client_budget_bytes": 128*1024*1024,
              "clock": "CLOCK_MONOTONIC nanoseconds since origin", "clock_id":"CLOCK_MONOTONIC", "origin_monotonic_ns": 100,
              "config_sha256": "f"*64}
    slots = [dict(request_id=i, slot_id=i, generation=0, phase="decode", active=True,
                  runnable_forward=True, runnable_emit=False, inflight=False, blocked=None)
             for i in range(4)]
    rows = [dict(request_id=0 if temporal else i, slot_id=0 if temporal else i,
                 generation=0, token_position=i, token_count=1, phase="verify" if temporal else "ordinary")
            for i in range(4)]
    events = [(0, "header", header), (0, "slots", {"cause": "MOCK", "slots": slots}),
              (1, "submission", {"submission_id": 7, "membership": "Known", "rows": rows}),
              (3_000_000_000, "drain", {"submission_id": 7, "success": True}),
              (4_000_000_000, "end", {"error": None, "dropped_events": 0})]
    records = [dict(schema=trace.SCHEMA, seq=i, at_ns=ns, kind=k, payload=v)
               for i, (ns, k, v) in enumerate(events)]
    return records, identity, {"schema": "hf2q.f2.trace-expectation.v1", "config_sha256": "f"*64, "run_id":"MOCK"}


class TraceControls(unittest.TestCase):
    def test_quad_and_temporal_singleton_are_not_confused(self):
        records, identity, scenario = trace_fixture()
        self.assertTrue(trace.validate_trace(records, identity, scenario)["shared_batch4_claim"])
        records, identity, scenario = trace_fixture(True)
        self.assertFalse(trace.validate_trace(records, identity, scenario)["shared_batch4_claim"])
        # Four runnable logical slots do not require a shared four-row graph.
        records[2]["payload"]["rows"] = records[2]["payload"]["rows"][:1]
        self.assertTrue(trace.validate_trace(records, identity, scenario)["qualified"])

    def test_missing_failed_unknown_stale_overflow_refuse(self):
        for mutation in ("missing", "failed", "unknown", "stale", "overflow", "gap"):
            r, identity, scenario = trace_fixture()
            if mutation == "missing": r.pop(3)
            if mutation == "failed": r[3]["payload"]["success"] = False
            if mutation == "unknown": r[2]["payload"]["membership"] = "Unknown"
            if mutation == "stale": r[2]["payload"]["rows"][0]["generation"] = 1
            if mutation == "overflow": r[-1]["payload"]["error"] = "quota exceeded"
            if mutation == "gap": r[2]["seq"] += 1
            with self.assertRaises(ValueError, msg=mutation):
                trace.validate_trace(r, identity, scenario)

    def test_submitted_in_window_drain_later_counts_not_heartbeat(self):
        r, identity, scenario = trace_fixture()
        trace.validate_trace(r, identity, scenario)
        self.assertEqual(trace.qualify_window(r, (0, 4), 0)["status"], "Qualified")
        r[2]["at_ns"] = 2_000_000_001  # Existing inflight, no new work inside window.
        self.assertEqual(trace.qualify_window(r, (0, 4), 0)["status"], "Unqualified")

    def test_unknown_readiness_and_end_after_deadline_refuse(self):
        r, _, _ = trace_fixture()
        r[1]["payload"]["slots"][0]["runnable_forward"] = None
        self.assertEqual(trace.qualify_window(r, (0, 4), 0)["status"], "Unqualified")
        r, _, _ = trace_fixture()
        r[1]["at_ns"] = 29_000_000_000
        self.assertEqual(trace.qualify_window(r, (0, 4), 0)["status"], "Unqualified")


def scoreboard_fixture():
    plan = dict(schema=score.SCHEMA, fixed_blocks=6, orders=list(score.ORDERS),
                optional_extension=False, primary="own_output_four_successful_task_makespan",
                precision_halfwidth_pp=2.5, sampling_lane="captured_production_client",
                production_client_profile={"path":"/MOCK-not-real-evidence","sha256":"0"*64},
                arms={a: {"arm_fixture": a,"profile_sha256":"f"*64} for a in "ABC"})
    cohorts = []
    for block, order in enumerate(score.ORDERS):
        for position, arm in enumerate(order):
            cid = f"MOCK-{block}-{arm}"
            cohorts.append(dict(cohort_id=cid, block_index=block, position=position, arm=arm,
                identity=plan["arms"][arm], started_ns=0,
                finished_ns={"A":100_000_000_000,"B":90_000_000_000,"C":110_000_000_000}[arm],
                trace_status="Qualified", certification=dict(schema="hf2q.q2.certification.v1",
                    cohort_id=cid, identity=plan["arms"][arm], required_cardinality=4,
                    verified_refs={"MOCK": "pure calculation only, not accepted raw evidence"},
                    sampling_lane="captured_production_client", profile_sha256="f"*64,
                    makespan_started_ns=0,
                    makespan_finished_ns={"A":100_000_000_000,"B":90_000_000_000,"C":110_000_000_000}[arm],
                    status="Qualified")))
    return plan, cohorts


class ScoreControls(unittest.TestCase):
    def test_constant_paired_effect_has_exact_ten_percent_reduction(self):
        result = score.paired_scoreboard(*scoreboard_fixture())
        self.assertAlmostEqual(result["primary"]["reduction_percent"], 10)
        self.assertLess(result["primary"]["halfwidth_percentage_points"], 1e-10)

    def test_failed_task_fast_time_never_wins_and_bad_trace_invalidates(self):
        plan, cohorts = scoreboard_fixture()
        cohorts[0]["finished_ns"] = 1
        cohorts[0]["certification"]["makespan_finished_ns"] = 1
        cohorts[0]["certification"].update(status="Rejected", failure_class="quality")
        result = score.paired_scoreboard(plan, cohorts)
        self.assertEqual(result["status"], "QualityIneligible")
        self.assertIsNone(result["primary"])
        cohorts[1]["trace_status"] = "Unqualified"
        self.assertEqual(score.paired_scoreboard(plan, cohorts)["status"], "InvalidCampaign")

    def test_n1_order_replacement_and_optional_stop_refuse(self):
        for mutation in ("N1", "order", "optional"):
            plan, cohorts = scoreboard_fixture()
            if mutation == "N1": cohorts[0]["certification"]["required_cardinality"] = 1
            if mutation == "order": cohorts[0], cohorts[1] = cohorts[1], cohorts[0]
            if mutation == "optional": plan["optional_extension"] = True
            with self.assertRaises(ValueError):
                score.paired_scoreboard(plan, cohorts)

    def test_peer_imprecision_and_forged_clock_do_not_qualify(self):
        plan, cohorts = scoreboard_fixture()
        for c in cohorts:
            if c["arm"] == "C":
                c["finished_ns"] *= c["block_index"] + 1
                c["certification"]["makespan_finished_ns"] = c["finished_ns"]
        result = score.paired_scoreboard(plan, cohorts)
        self.assertTrue(result["primary_precision_met"])
        self.assertFalse(result["peer_precision_met"])
        self.assertEqual(result["status"], "InconclusivePrecision")
        cohorts[0]["started_ns"] = 1
        with self.assertRaises(ValueError):
            score.paired_scoreboard(plan, cohorts)


if __name__ == "__main__":
    unittest.main()
