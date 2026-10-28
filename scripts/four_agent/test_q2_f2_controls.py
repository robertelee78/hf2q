"""Independent handwritten finite controls call F1's actual production entry points.
Requires integration of F2 source; no skip or mock substitute when it is missing.
"""
import copy
import importlib
import unittest


def trace_fixture(composition=(2, 2)):
    slots = [dict(request_id=i+1, slot_id=i, generation=7, phase="prefill" if i < composition[0]
                  else "decode", active=True, runnable_forward=True, runnable_emit=False,
                  inflight=False, blocked=None) for i in range(sum(composition))]
    events = [(0, "header", dict(identity={"test_identity": "independent-fixture"},
               server_budget_bytes=1048576, client_budget_bytes=128*1024*1024,
               clock="CLOCK_MONOTONIC nanoseconds since origin", clock_id="CLOCK_MONOTONIC", origin_monotonic_ns=1000,
               config_sha256="f"*64,run_id="q2_control")), (0, "slots", dict(cause="admitted", slots=slots))]
    for i, slot in enumerate(slots):
        events.append((100000000+i, "submission", dict(submission_id=i, membership="Known",
                       rows=[dict(request_id=slot["request_id"], slot_id=i, generation=7,
                                  token_position=1, token_count=1, phase="prefill" if slot["phase"]=="prefill" else "ordinary")])))
    # Actual checked drain may occur AFTER the 2s measurement window.
    for i in range(len(slots)):
        events.append((3000000000+i, "drain", dict(submission_id=i, success=True)))
    events.append((4000000000, "end", dict(error=None, dropped_events=0)))
    return [dict(schema="hf2q.f2.phase-trace.v1", seq=i, at_ns=t, kind=k, payload=p)
            for i, (t, k, p) in enumerate(events)]


def scoreboard_fixture():
    orders = ["ABC", "BCA", "CAB", "CBA", "BAC", "ACB"]
    arms = {a: dict(model_sha256="a"*64, profile_sha256="f"*64, arm=a) for a in "ABC"}
    plan = dict(schema="hf2q.f2.fixed-six-scoreboard.v1", fixed_blocks=6, orders=orders,
                optional_extension=False, primary="own_output_four_successful_task_makespan",
                precision_halfwidth_pp=2.5, arms=arms, sampling_lane="captured_production_client",
                production_client_profile=dict(path="MOCK-profile", sha256="f"*64))
    cohorts=[]
    for block, order in enumerate(orders):
        for position, arm in enumerate(order):
            identity=arms[arm]
            cohorts.append(dict(cohort_id=f"block{block}-{arm}", block_index=block,
                position=position, arm=arm, identity=identity, started_ns=0,
                finished_ns={"A":10000000000,"B":9000000000,"C":10000000000}[arm],
                certification=dict(schema="hf2q.q2.certification.v1", status="Qualified",
                    cohort_id=f"block{block}-{arm}", identity=identity,
                    required_cardinality=4, sampling_lane="captured_production_client", profile_sha256="f"*64, makespan_started_ns=0,
                    makespan_finished_ns={"A":10000000000,"B":9000000000,"C":10000000000}[arm], verified_refs=[dict(path="MOCK",sha256="a"*64)]),
                trace_status="Qualified"))
    return plan,cohorts


class ProductionContractControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace=importlib.import_module("f2_trace")
        cls.stats=importlib.import_module("f2_scoreboard")
        cls.semantic=importlib.import_module("f2_semantic")

    def validate(self, records):
        return self.trace.validate_trace(records, {"test_identity":"independent-fixture"},
                dict(schema="hf2q.f2.trace-expectation.v1", config_sha256="f"*64,run_id="q2_control"))

    def test_all_populations_and_known_singletons_eventual_drain(self):
        for population in ((1,0),(0,1),(4,0),(3,1),(2,2),(1,3),(0,4)):
            with self.subTest(population=population):
                r=trace_fixture(population)
                self.assertTrue(self.validate(r)["qualified"])
                self.assertFalse(self.validate(r)["shared_batch4_claim"])
                self.assertEqual(self.trace.qualify_window(r,population,0)["status"],"Qualified")

    def test_serialized_four_http_labels_never_make_four_runnable_slots(self):
        r=trace_fixture((1,0));self.validate(r)
        self.assertNotEqual(self.trace.qualify_window(r,(4,0),0)["status"],"Qualified")

    def test_unknown_duplicate_and_stale_physical_membership(self):
        for corrupt in ("unknown","duplicate","generation"):
            r=trace_fixture();row=r[2]["payload"]
            if corrupt=="unknown":row["membership"]="Unknown"
            elif corrupt=="duplicate":row["rows"].append(copy.deepcopy(row["rows"][0]))
            else:row["rows"][0]["generation"]=6
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):self.validate(r)

    def test_missing_failed_drains_sequence_overflow(self):
        for corrupt in ("drain","failed","sequence","overflow"):
            r=trace_fixture()
            if corrupt=="drain":r.pop(-2)
            elif corrupt=="failed":r[-2]["payload"]["success"]=False
            elif corrupt=="sequence":r[2]["seq"]=999
            else:r[-1]["payload"]["dropped_events"]=1
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):self.validate(r)

    def test_stale_progress_cannot_replace_actual_owner(self):
        r=trace_fixture();r.insert(2,dict(schema=r[0]["schema"],seq=2,at_ns=1,kind="progress",
            payload=dict(slot_id=99,generation=999,source="generated_publication",tokens=1,committed_position=1)))
        for i,e in enumerate(r):e["seq"]=i
        with self.assertRaises(ValueError):self.validate(r)

    def test_metadata_quota_256m_plus_one_fails_without_allocating_payload(self):
        r=trace_fixture();r[0]["payload"]["server_budget_bytes"]=128*1024*1024+1
        r[0]["payload"]["client_budget_bytes"]=128*1024*1024
        with self.assertRaises(ValueError):self.validate(r)

    def test_semantic_units_first_inter_terminal_and_unary(self):
        value=self.semantic.semantic_timing_v3(1000000000,[1100000000,1400000000],2000000000,True,True)
        self.assertAlmostEqual(value["semantic_ttft_seconds"],.1)
        self.assertAlmostEqual(value["semantic_gap_inter_events_seconds"],.3)
        self.assertAlmostEqual(value["semantic_terminal_gap_seconds"],.6)
        self.assertAlmostEqual(value["semantic_gap_end_to_end_seconds"],.6)
        single=self.semantic.semantic_timing_v3(0,[100000000],200000000,True,True)
        self.assertIsNone(single["semantic_gap_inter_events_seconds"])
        self.assertIsNone(self.semantic.semantic_timing_v3(0,[],200000000,False,True)["semantic_ttft_seconds"])
        with self.assertRaises(ValueError):self.semantic.semantic_timing_v3(0,[],1,True,True)

    def test_fixed_six_known_ten_percent_gain_null_loss(self):
        for ratio,reduction in ((.9,10.),(1.,0.),(1.1,-10.)):
            plan,cohorts=scoreboard_fixture()
            for c in cohorts:
                if c["arm"]=="B":
                    c["finished_ns"]=round(10000000000*ratio)
                    c["certification"]["makespan_finished_ns"]=c["finished_ns"]
            result=self.stats.paired_scoreboard(plan,cohorts)
            self.assertAlmostEqual(result["primary"]["reduction_percent"],reduction)
            self.assertAlmostEqual(result["primary"]["halfwidth_percentage_points"],0.)
            self.assertEqual(len(result["primary"]["block_log_ratios"]),6)

    def test_all_failed_or_missing_peer_blocks_retained_ineligible(self):
        for status,failure_class in (("Rejected","quality"),("PendingIndependentQE","infrastructure")):
            plan,cohorts=scoreboard_fixture();cohorts[0]["certification"].update(status=status,failure_class=failure_class)
            result=self.stats.paired_scoreboard(plan,cohorts)
            self.assertIsNone(result["primary"]);self.assertEqual(len(result["cohorts"]),18)

    def test_n1_identity_changed_units_or_optional_extension_cannot_win(self):
        for corrupt in ("n1","identity","order","zero","duplicate","extra","extension","units"):
            plan,cohorts=scoreboard_fixture()
            if corrupt=="n1":cohorts[0]["certification"]["required_cardinality"]=1
            elif corrupt=="identity":cohorts[0]["identity"]={"other":"source"}
            elif corrupt=="order":cohorts.reverse()
            elif corrupt=="zero":cohorts[0]["finished_ns"]=0
            elif corrupt=="duplicate":cohorts[1]["cohort_id"]=cohorts[0]["cohort_id"]
            elif corrupt=="extra":cohorts.append(copy.deepcopy(cohorts[-1]))
            elif corrupt=="units":cohorts[0]["finished_ns"]*=1000000
            else:plan["optional_extension"]=True
            with self.subTest(corrupt=corrupt),self.assertRaises(ValueError):self.stats.paired_scoreboard(plan,cohorts)

    def test_noisy_peer_precision_not_hidden_by_constant_stock_contrast(self):
        plan,cohorts=scoreboard_fixture()
        for c in cohorts:
            if c["arm"]=="C":
                c["finished_ns"]=(1 if c["block_index"]%2 else 100)*1000000000
                c["certification"]["makespan_finished_ns"]=c["finished_ns"]
        self.assertNotEqual(self.stats.paired_scoreboard(plan,cohorts)["status"],"Qualified")


if __name__=="__main__":unittest.main()
