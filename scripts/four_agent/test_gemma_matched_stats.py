"""Pure fixed-study controls, never a model/tokenizer or measured benchmark."""
import copy
import math
import unittest
from qwen38_n1_verdict import Invalid
import gemma_matched_stats as m


def agents():
    return {a: {"benchmark_profile":"gemma-matched-v1", "status":"ProtocolCompleteAwaitingIndependentPatchReview",
                "cleanup":True, "turns":[{"turn":t,"stream":bool(t%2),"semantic_ttft_seconds":2.0 if t%2 else None} for t in range(7)],
                "tools":[{"benchmark_profile":"gemma-matched-v1","passed":True,"status":"Complete"} for _ in range(6)]}
            for a in m.AGENTS}


def study():
    runs=[{**p,"quality_verified":True,"cleanup_verified":True,
           "primary_sse_geomean_seconds":1.0 if p["arm"]=="ON" else 2.0,
           "coordinator_wall_seconds":20.0} for p in m.PLAN]
    vectors=[{a:[[1,t+2] for t in range(7)] for a in m.AGENTS} for _ in runs]
    return runs,vectors


class Statistics(unittest.TestCase):
    def test_fixed_abba_no_optional_extra_runs(self):
        self.assertEqual([p["arm"] for p in m.PLAN], ["OFF","ON","ON","OFF"]*5)
        runs,vectors=study()
        for bad in (runs[:-1],runs+[runs[-1]]):
            with self.assertRaises(Invalid):m.analyze(bad,vectors)

    def test_actual_cohort_composite_uses_only_three_sse_turns(self):
        a=agents();a["agent-0"]["turns"][0]["semantic_ttft_seconds"]=999999
        result=m.cohort_score(a)
        self.assertAlmostEqual(result["primary_sse_geomean_seconds"],2.0)
        self.assertEqual(result["sse_observations"],12)
        self.assertFalse(result["unary_semantic_ttft_claimed"])

    def test_missing_or_nonfinite_sse_values_fail(self):
        for value in (None,0,-1,float('nan'),float('inf'),True):
            a=agents();a["agent-0"]["turns"][1]["semantic_ttft_seconds"]=value
            with self.assertRaises(Invalid):m.cohort_score(a)

    def test_incomplete_conversation_or_actual_tools_fail(self):
        for change in ("agent","turn","tool","cleanup","profile"):
            a=agents()
            if change=="agent":del a["agent-3"]
            elif change=="turn":a["agent-0"]["turns"].pop()
            elif change=="tool":a["agent-0"]["tools"][3]["passed"]=False
            elif change=="cleanup":a["agent-0"]["cleanup"]=False
            else:a["agent-0"]["benchmark_profile"]=None
            with self.assertRaises(Invalid):m.cohort_score(a)

    def test_exact_tokens_not_length_hash_or_subset(self):
        _,v=study();m.exact_vectors(v[0],v[1])
        for patch in ("token","turn","agent","hash","bool"):
            right=copy.deepcopy(v[1])
            if patch=="token":right["agent-1"][4][1]=99
            elif patch=="turn":right["agent-0"].pop()
            elif patch=="agent":del right["agent-3"]
            elif patch=="hash":right["agent-0"][0]="a"*64
            else:right["agent-0"][0]=[True,2]
            with self.assertRaises(Invalid):m.exact_vectors(v[0],right)

    def test_one_late_pair_mismatch_rejects_whole_study(self):
        runs,vectors=study();vectors[18]["agent-3"][6][1]=999
        with self.assertRaises(Invalid):m.analyze(runs,vectors)

    def test_plan_order_quality_cleanup_or_boolean_index_refused(self):
        for field,value in (("index",False),("arm","ON"),("quality_verified",False),("cleanup_verified",False)):
            runs,vectors=study();runs[0][field]=value
            with self.assertRaises(Invalid):m.analyze(runs,vectors)

    def test_fixed_bootstrap_reports_win_and_loss_without_qualification(self):
        runs,vectors=study();first=m.analyze(runs,vectors);second=m.analyze(runs,vectors)
        self.assertEqual(first,second)
        self.assertEqual(first["primary_ratio_on_over_off"],.5)
        self.assertEqual(first["paired_block_ci95"],[.5,.5])
        self.assertTrue(first["component_latency_improvement_observed"])
        self.assertFalse(first["performance_qualified"])
        self.assertEqual(first["secondary_wall_ratio_on_over_off"],1.0)
        for r in runs:r["primary_sse_geomean_seconds"] = 3 if r["arm"]=="ON" else 2
        self.assertFalse(m.analyze(runs,vectors)["component_latency_improvement_observed"])

    def test_component_point_and_precision_flags_never_qualify_campaign(self):
        runs,vectors=study()
        for r in runs:r["primary_sse_geomean_seconds"]=1.92 if r["arm"]=="ON" else 2.0
        result=m.analyze(runs,vectors)
        self.assertAlmostEqual(result["point_improvement_percent"],4.0)
        self.assertFalse(result["stock_point_at_least_5percent_met"])
        self.assertTrue(result["precision_target_met"])
        self.assertEqual(result["ci95_halfwidth_percentage_points"],0)
        self.assertTrue(result["component_latency_improvement_observed"])
        self.assertTrue(result["peer_comparison_missing"])
        self.assertFalse(result["campaign_primary_acceptance"])

    def test_precision_miss_stops_fixed_plan_inconclusive_without_extension(self):
        runs,vectors=study();values=[.1,1.,1.5,2.,3.]
        for r in runs:r["primary_sse_geomean_seconds"]=values[r["block"]] if r["arm"]=="ON" else 2.
        result=m.analyze(runs,vectors)
        self.assertGreater(result["ci95_halfwidth_percentage_points"],2.5)
        self.assertFalse(result["precision_target_met"])
        self.assertFalse(result["component_latency_improvement_observed"])
        self.assertEqual(result["component_study_status"],"InconclusiveFixedPlanPrecisionMiss")
        self.assertFalse(result["campaign_primary_acceptance"])
        self.assertEqual(result["cohorts"],20)

if __name__=="__main__":unittest.main()
