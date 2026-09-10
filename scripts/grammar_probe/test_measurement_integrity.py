import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from harness_test_support import Fixture, legacy_v, result, verdict
from measurement import (checked_join, digest, generation_inventory, judgment_input,
                         read_jsonl, text_hash)
import report
import rejudge
from runtime_identity import validate_identity, require_unconstrained


class IdentityAndReporting(unittest.TestCase):
    def build(self, rs, vs, control="BASE", selected=None):
        with patch.object(report, "CONTROL", control), patch.dict(os.environ, {"PASS": selected} if selected else {}, clear=True):
            return report.build_report(rs, vs)

    def test_budget_cells_and_run_cells_are_distinct(self):
        rows = [result(a, budget=b, run=run, content=f"{a}-{b}-{run}")
                for run in ("r1", "r2") for b in (400, 800) for a in ("BASE", "W1")]
        vs = [legacy_v(r, generation_config_sha256=r["config_sha256"],
                       response_hash=text_hash(r["content"])) for r in rows]
        data = self.build(rows, vs)
        self.assertEqual(len(data["arms"]), 8)
        self.assertEqual(sum(v["n_responses"] for v in data["arms"].values()), 8)
        self.assertEqual(len(data["paired"]), 4)
        self.assertTrue(all(v["n_pairs"] == 1 for v in data["paired"].values()))

    def test_supplied_hash_mismatch_and_ambiguous_legacy_fail(self):
        r = result(content="actual")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build([r], [legacy_v(r, response_hash=text_hash("other"))])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.build([result(budget=400), result(budget=800)], [legacy_v(r)])
        with self.assertRaisesRegex(ValueError, "conflicting generation"):
            self.build([r, result(content="changed")], [])

    def test_explicit_pass_and_ambiguous_default(self):
        r = result()
        vs = [legacy_v(r, scoring_config_hash="more", ts=i) for i in (1, 2)]
        vs.append(legacy_v(r, scoring_config_hash="wanted", response_state="maintained_refusal"))
        self.assertEqual(self.build([r], vs, selected="wanted")["arms"]["W1"]["metrics"]["refusal"]["count"], 1)
        with self.assertRaisesRegex(ValueError, "multiple scoring passes"):
            self.build([r], vs)
        with self.assertRaisesRegex(ValueError, "absent"):
            self.build([r], vs, selected="missing")

    def test_termination_and_generation_errors_do_not_depend_on_judge(self):
        rows = [result(pid="p1", finish="length"), result(pid="p2", finish="stop"),
                {"arm": "W1", "prompt_id": "p3", "rep": 1, "error": "HTTP 500"}]
        data = self.build(rows, [legacy_v(rows[1], judge_error="failed")])["arms"]["W1"]
        self.assertEqual((data["n_observations"], data["n_responses"], data["n_generation_error"]), (3, 2, 1))
        self.assertEqual((data["n_judge_error"], data["n_unjudged"]), (1, 1))
        self.assertEqual(data["metrics"]["budget_limited"]["per_response"], .5)
        self.assertEqual(data["finish_x_state"]["length|unjudged"], 1)
        self.assertEqual(data["finish_x_state"]["generation_error|not_judged"], 1)

    def test_known_pairs_missing_judgment_and_clustered_interval(self):
        rows = [result(a, pid=p, rep=rep) for a in ("BASE", "W1") for p in ("a", "b") for rep in (1, 2)]
        vs = [legacy_v(r, response_state="maintained_refusal" if r["arm"] == "BASE" else "valid_fulfillment") for r in rows]
        data = self.build(rows, vs)["paired"]["W1"]
        ci = data["metrics"]["refusal"]
        self.assertEqual((ci["delta_point"], ci["ci_low"], ci["ci_high"]), (-1, -1, -1))
        self.assertEqual(ci["n_clusters"], 2)
        data = self.build(rows, vs[:-1])["paired"]["W1"]
        self.assertEqual(data["n_pairs"], 3)
        self.assertEqual(data["pairs_missing_judgment_one_side"], 1)
        self.assertEqual(data["metrics"]["clean_stop"]["n_pairs"], 4)

    def test_legacy_inconsistency_and_orphan_are_visible(self):
        r = result()
        vs = [legacy_v(r, output_validity="invalid"), legacy_v(result(pid="orphan"))]
        data = self.build([r], vs)
        self.assertTrue(data["legacy_binding_warning"])
        self.assertEqual(len(data["orphan_verdicts"]), 1)
        self.assertEqual(data["arms"]["W1"]["metrics"]["material_fulfill"]["count"], 0)
        self.assertTrue(data["arms"]["W1"]["inconsistent_fields"])

    def test_full_judgment_input_changes_with_every_observed_dimension(self):
        r = result(budget=400, run="r", content="same")
        original = digest(judgment_input(r, "prompt"))
        self.assertNotEqual(original, digest(judgment_input(r, "changed prompt")))
        for key, value in (("finish", "length"), ("completion_tokens", 999), ("budget", 800),
                           ("config_sha256", "other"), ("generation_run_id", "other")):
            self.assertNotEqual(original, digest(judgment_input({**r, key: value}, "prompt")), key)

    def test_human_samples_are_full_and_machine_blind(self):
        rows = [result(pid=f"p{i}", content="A" * 3100 + f"TAIL{i}") for i in range(10)]
        old = [legacy_v(r) for r in rows]
        new = [legacy_v(r, response_state="maintained_refusal" if i < 2 else "valid_fulfillment") for i, r in enumerate(rows)]
        prompts = {r["prompt_id"]: "Full prompt " * 100 for r in rows}
        sample, key = rejudge.build_blind_sample(old, new, rows, prompts, 10, 7)
        self.assertEqual(sample["population_size"], 10)
        for entry in sample["entries"]:
            self.assertGreater(len(entry["response"]), 3100)
            self.assertGreater(len(entry["prompt"]), 300)
            self.assertNotIn("judgments", entry)
            self.assertNotIn("arm", entry)
            self.assertEqual(text_hash(entry["response"]), entry["response_sha256"])
        self.assertIn("historical", next(iter(key.values())))
        diagnostic, _ = rejudge.build_blind_sample(old, new, rows, prompts, 10, 7, diagnostic=True)
        self.assertEqual(diagnostic["population_size"], 2)
        self.assertEqual(diagnostic["sample_kind"], "disagreement_diagnostic")
        self.assertEqual(sample, rejudge.build_blind_sample(old, new, rows, prompts, 10, 7)[0])

    def test_transition_identity_preserves_both_arms_and_budgets(self):
        rows = [result(a, budget=b, content=f"{a}-{b}") for a in ("BASE", "W1") for b in (400, 800)]
        old = [legacy_v(r) for r in rows]
        new = [legacy_v(r, response_state="maintained_refusal" if r["arm"] == "BASE" else "valid_fulfillment") for r in rows]
        table = rejudge.compute_transitions(old, new, rows)
        self.assertEqual(table["n_results"], 4)
        self.assertEqual(table["transitions"]["valid_fulfillment->maintained_refusal"], 2)


class RuntimeAttestations(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
    def tearDown(self):
        self.f.close()
    def test_verified_identity_changes_and_unsteered_baseline(self):
        f = self.f
        ident = validate_identity(f.manifest, f.snapshot)
        require_unconstrained(ident)
        changed = copy.deepcopy(f.snapshot); changed["engine_generation"] += 1
        with self.assertRaisesRegex(ValueError, "snapshot"):
            validate_identity(f.manifest, changed)
        for control in ("glp", "grammar", "dwq_overlay"):
            active = copy.deepcopy(ident)
            if control == "dwq_overlay": active["active_controls"][control] = True
            else: active["active_controls"][control]["active"] = True
            with self.assertRaises(ValueError): require_unconstrained(active)
    def test_missing_hash_and_not_running_fail(self):
        for field in ("binary_sha256", "tokenizer_sha256", "template_sha256", "model_artifacts"):
            m = copy.deepcopy(self.f.manifest)
            m["runtime_identity"].pop(field)
            m["runtime_identity"]["identity_sha256"] = digest({k:v for k,v in m["runtime_identity"].items() if k != "identity_sha256"})
            with self.assertRaises(ValueError): validate_identity(m, self.f.snapshot)
        m = copy.deepcopy(self.f.manifest);m["state"]="exited"
        with self.assertRaisesRegex(ValueError, "not running"): validate_identity(m, self.f.snapshot)


if __name__ == "__main__":
    unittest.main()
