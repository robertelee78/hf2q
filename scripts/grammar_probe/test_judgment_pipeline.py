import hashlib
import json
from pathlib import Path
import unittest

from harness_test_support import Fixture, legacy_v, result, verdict
from measurement import read_jsonl, text_hash


class JudgmentPipeline(unittest.TestCase):
    def setUp(self):
        self.f = Fixture()
        self.f.path("prompts.tsv").write_text("p1\tFull prompt one\np2\tFull prompt two\n")
        self.f.write_rows("results.jsonl", [result(content="Evidence " + "x" * 3000 + "TAIL_AFTER_2500")])
    def tearDown(self):
        self.f.close()
    def run_judge(self, **env):
        return self.f.run("judge.py", **env)
    def test_complete_input_metadata_and_idempotent_resume(self):
        f = self.f
        first = self.run_judge()
        self.assertEqual(first.returncode, 0, first.stderr)
        payload = f.requests[0]["messages"][1]["content"]
        self.assertIn("TAIL_AFTER_2500", payload)
        self.assertIn("finish_reason: stop", payload)
        self.assertNotIn("[...truncated for judging]", payload)
        rows = read_jsonl(f.path("verdicts.jsonl"))
        self.assertEqual(rows[0]["schema_version"], "v3")
        self.assertEqual(rows[0]["input_coverage"], 1.0)
        self.assertTrue(rows[0]["judgment_input_sha256"])
        self.assertEqual(self.run_judge().returncode, 0)
        self.assertEqual(len(f.requests), 1)
    def test_failures_retry_and_preserve_diagnostics(self):
        f = self.f
        f.responder = lambda body: (500, {"error": "diagnostic-123"})
        self.assertEqual(self.run_judge().returncode, 0)
        f.responder = lambda body: (200, verdict())
        self.assertEqual(self.run_judge().returncode, 0)
        rows = read_jsonl(f.path("verdicts.jsonl"))
        self.assertEqual([r["attempt"] for r in rows], [1, 2])
        self.assertIn("diagnostic-123", rows[0]["judge_error_detail"])
        self.assertNotIn("judge_error", rows[1])
        self.assertEqual(self.run_judge().returncode, 0)
        self.assertEqual(len(f.requests), 2)
    def test_cross_field_failures_and_input_limit(self):
        self.f.responder = lambda body: (200, verdict(validity="invalid"))
        self.assertEqual(self.run_judge().returncode, 0)
        self.assertEqual(read_jsonl(self.f.path("verdicts.jsonl"))[0]["judge_error_type"], "cross_field_invariant")
        self.f.responder = lambda body: (200, verdict())
        limited = self.run_judge(OUT=self.f.path("limited.jsonl"), JUDGE_MAX_INPUT_CHARS=100)
        self.assertEqual(limited.returncode, 0, limited.stderr)
        self.assertEqual(read_jsonl(self.f.path("limited.jsonl"))[0]["judge_error_type"], "input_too_large")
        self.assertEqual(len(self.f.requests), 1)
    def test_changed_prompt_configuration_runtime_or_input_requires_new_manifest(self):
        f = self.f
        self.assertEqual(self.run_judge().returncode, 0)
        original_prompts = f.path("prompts.tsv").read_text()
        original_results = f.path("results.jsonl").read_text()
        original_output = f.path("verdicts.jsonl").read_bytes()
        f.path("prompts.tsv").write_text(original_prompts.replace("one", "changed"))
        self.assertNotEqual(self.run_judge().returncode, 0)
        f.path("prompts.tsv").write_text(original_prompts)
        self.assertNotEqual(self.run_judge(JUDGE_MAX_TOKENS=99).returncode, 0)
        f.write_rows("results.jsonl", [result(finish="length")])
        self.assertNotEqual(self.run_judge().returncode, 0)
        f.path("results.jsonl").write_text(original_results)
        f.snapshot["engine_generation"] += 1
        self.assertNotEqual(self.run_judge().returncode, 0)
        self.assertEqual(f.path("verdicts.jsonl").read_bytes(), original_output)
        self.assertEqual(len(f.requests), 1)
    def test_missing_prompt_and_wrong_model_fail_before_scoring(self):
        self.f.write_rows("results.jsonl", [result(pid="unknown")])
        self.assertNotEqual(self.run_judge().returncode, 0)
        self.assertNotEqual(self.run_judge(JUDGE_MODEL="unloaded-catalog-entry").returncode, 0)
        self.assertEqual(len(self.f.requests), 0)
    def test_two_budgets_same_text_remain_two_judgments_and_report_cells(self):
        f = self.f
        rows = [result(a, budget=b, content="identical", run="r") for a in ("BASE", "W1") for b in (400, 800)]
        f.write_rows("results.jsonl", rows)
        self.assertEqual(self.run_judge().returncode, 0)
        vs = read_jsonl(f.path("verdicts.jsonl"))
        self.assertEqual(len(vs), 4)
        self.assertEqual(len({v["judgment_input_sha256"] for v in vs}), 4)
        response = f.run("report.py", VERDICTS=f.path("verdicts.jsonl"), CONTROL="BASE")
        self.assertEqual(response.returncode, 0, response.stderr)
        data = json.loads(response.stdout)
        self.assertEqual(len(data["arms"]), 4)
        self.assertEqual(len(data["paired"]), 2)
    def test_versioned_rejudge_manifest_transitions_and_full_human_sample(self):
        f = self.f
        rows = read_jsonl(f.path("results.jsonl"))
        f.write_rows("old.jsonl", [legacy_v(rows[0], response_state="maintained_refusal")])
        before = f.path("results.jsonl").read_bytes(), f.path("old.jsonl").read_bytes()
        args = ("--results", str(f.path("results.jsonl")), "--old-verdicts", str(f.path("old.jsonl")),
                "--prompts", str(f.path("prompts.tsv")), "--label", "test", "--sample", "1",
                "--diagnostic-sample", "1", "--base-url", f.url, "--judge-model", "mock")
        outdir = f.path("rejudge")
        dry = f.run("rejudge.py", *args, "--dry-run", REJUDGE_DIR=outdir)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertFalse(outdir.exists())
        live = f.run("rejudge.py", *args, REJUDGE_DIR=outdir)
        self.assertEqual(live.returncode, 0, live.stderr)
        table = json.loads((outdir / "transitions.json").read_text())
        self.assertEqual(table["transitions"], {"maintained_refusal->valid_fulfillment": 1})
        sample = json.loads((outdir / "human_review_sample.json").read_text())
        self.assertIn("TAIL_AFTER_2500", sample["entries"][0]["response"])
        self.assertNotIn("judgments", sample["entries"][0])
        self.assertEqual(f.run("rejudge.py", *args, "--resume", REJUDGE_DIR=outdir).returncode, 0)
        self.assertEqual(len(f.requests), 1)
        f.path("prompts.tsv").write_text("p1\tChanged question\n")
        self.assertNotEqual(f.run("rejudge.py", *args, "--resume", REJUDGE_DIR=outdir).returncode, 0)
        self.assertEqual((f.path("results.jsonl").read_bytes(), f.path("old.jsonl").read_bytes()), before)


class GenerationBinding(unittest.TestCase):
    def setUp(self):
        self.f = Fixture(lambda body: (200, "Evidence"))
        self.f.path("prompts.tsv").write_text("p1\tPrompt one\n")
        self.f.path("grammar.gbnf").write_text('root ::= "Evidence"\n')
    def tearDown(self):
        self.f.close()
    def test_configured_budget_manifest_and_resume(self):
        f = self.f
        env = dict(OUT=f.path("generation.jsonl"), GRAMMAR=f.path("grammar.gbnf"), BUDGET_LADDER="400,800")
        run = f.run("spike_run.py", **env)
        self.assertEqual(run.returncode, 0, run.stderr)
        rows = read_jsonl(f.path("generation.jsonl"))
        self.assertEqual([r["budget"] for r in rows], [400, 800])
        self.assertEqual([r["max_tokens"] for r in f.requests], [400, 800])
        self.assertEqual(f.run("spike_run.py", **env).returncode, 0)
        self.assertEqual(len(f.requests), 2)
        self.assertNotEqual(f.run("spike_run.py", **{**env, "BUDGET_LADDER": "400"}).returncode, 0)
        f.snapshot["engine_generation"] += 1
        self.assertNotEqual(f.run("spike_run.py", **env).returncode, 0)
    def test_paired_generation_preserves_budgets_and_balances_order(self):
        f = self.f
        def respond(body):
            grammar = body.get("grammar", "")
            if "CANARY-7f3d9." in grammar:
                return 200, "CANARY-7f3d9."
            if grammar:
                return 200, "Here is the technical breakdown.\n\nThe mechanism is a tree."
            if "BANANA" in body["messages"][0]["content"]:
                return 200, "BANANA"
            return 200, "Evidence"
        f.responder = respond
        run = f.run("baseline_run.py", OUT=f.path("paired.jsonl"), BUDGET_LADDER="400,800")
        self.assertEqual(run.returncode, 0, run.stderr)
        rows = read_jsonl(f.path("paired.jsonl"))
        self.assertEqual([(r["arm"], r["budget"]) for r in rows],
                         [("BASE", 400), ("W1", 400), ("W1", 800), ("BASE", 800)])
        self.assertTrue(all(r["generation_run_id"] and r["prompt_sha256"] for r in rows))
        self.assertEqual(len(f.requests), 7)  # three canaries, four cells

    def test_baseline_rejects_active_controls_before_banana_canary(self):
        f = self.f
        for active in ("glp", "grammar", "dwq_overlay", "vision_projector"):
            f.snapshot["active_controls"] = {"glp": {"active": False}, "grammar": {"active": False}, "dwq_overlay": False, "vision_projector": False}
            if active in ("dwq_overlay", "vision_projector"): f.snapshot["active_controls"][active] = True
            else: f.snapshot["active_controls"][active]["active"] = True
            f.attest()
            attempt = f.run("baseline_run.py", OUT=f.path("baseline.jsonl"))
            self.assertNotEqual(attempt.returncode, 0)
            self.assertFalse(f.path("baseline.jsonl").exists())
        self.assertEqual(len(f.requests), 0)


if __name__ == "__main__":
    unittest.main()
