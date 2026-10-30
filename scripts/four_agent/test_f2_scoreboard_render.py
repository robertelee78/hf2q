"""Pure renderer controls: synthetic rows only, never model, GPU or receipt evidence."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import f2_scoreboard_render as r

SHA = "a" * 64
COMMIT = "b" * 40


def row(rid, metric, arm, value, status="Descriptive", variant=None, group="g", **over):
    out = {"id": rid, "family": "fam", "group": group, "metric": metric, "kind": "measurement", "arm": arm,
           "variant": variant, "value": value, "decimals": 3, "unit": r.METRICS[metric][1], "n_slots": 4,
           "context": "262144", "date": "2026-10-28", "source": {"commit": COMMIT, "binary_sha256": None},
           "artifact_sha256": SHA, "receipt": f"$E/{rid}/client/summary.json", "status": status,
           "verification": {"state": "Verified", "key": "elapsed_seconds",
                            "extract": {"value": {"kind": "json", "file": f"$E/{rid}/client/summary.json",
                                                  "path": "elapsed_seconds", "reduce": "one"}}},
           "note": ""}
    out.update(over)
    return out


def effect(rid, value, ci95, status="Descriptive"):
    out = row(rid, "semantic_ttft_s", "candidate", value, status=status, kind="effect", unit=r.EFFECT_UNIT,
              ci95=ci95, baseline_arm="hf2q_current")
    if ci95 is not None:
        out["verification"]["extract"]["ci95"] = {"kind": "json", "file": out["receipt"], "path": "ci.*", "reduce": "list"}
    return out


def manifest(rows):
    return r.validate_manifest({
        "schema": r.ROWS_SCHEMA, "evidence_root_symbol": "$E", "neutral_band_ratio": 1.025,
        "families": {"fam": {"title": "Family", "artifact_sha256": SHA, "note": ""}},
        "groups": {"g": {"family": "fam", "title": "G", "workload": "w"},
                   "h": {"family": "fam", "title": "H", "workload": "w"}},
        "rows": rows})


def runs(prefix, metric, arm, values, **kw):
    return [row(f"{prefix}{i}", metric, arm, v, **kw) for i, v in enumerate(values)]


def section(text, title):
    start = text.index(title)
    end = text.find("\n## ", start + len(title))
    return text[start:end if end >= 0 else None]


def scoreboard(status="Qualified", lo=4.0, hi=6.0, precision=True):
    e = {"geometric_ratio": 0.95, "reduction_percent": 5.0, "nominal_pointwise_ci95_percent": [lo, hi],
         "precision_met": precision}
    return {"schema": r.SCOREBOARD_SCHEMA, "status": status, "primary": e,
            "secondary": {"B_over_C": e, "A_over_C": e},
            "primary_precision_met": precision, "peer_precision_met": precision}


class Direction(unittest.TestCase):
    def verdicts(self, metric, candidate, baseline):
        m = manifest(runs("c", metric, "candidate", candidate) + runs("b", metric, "hf2q_current", baseline))
        (c,) = r.comparisons(m)
        return c

    def test_tok_per_second_higher_is_better(self):
        c = self.verdicts("decode_tps_per_agent", [12.0, 12.2], [10.0, 10.1])
        self.assertGreater(c["ratio"], 1)
        self.assertEqual(c["verdict"], "Win")
        c = self.verdicts("decode_tps_per_agent", [8.0, 8.1], [10.0, 10.1])
        self.assertLess(c["ratio"], 1)
        self.assertEqual(c["verdict"], "Regression")

    def test_seconds_lower_is_better(self):
        c = self.verdicts("makespan_s", [100.0, 101.0], [120.0, 121.0])
        self.assertAlmostEqual(c["ratio"], 120.5 / 100.5)
        self.assertEqual(c["verdict"], "Win")
        c = self.verdicts("client_elapsed_s", [130.0, 131.0], [120.0, 121.0])
        self.assertAlmostEqual(c["ratio"], 120.5 / 130.5)
        self.assertEqual(c["verdict"], "Regression")

    def test_neutral_band_and_unseparated_runs_are_not_wins(self):
        self.assertEqual(self.verdicts("makespan_s", [99.0, 99.5], [100.0, 100.5])["verdict"], "Neutral")
        # Large median gain from single runs, or overlapping ranges, is never a win.
        self.assertEqual(self.verdicts("makespan_s", [100.0], [120.0])["verdict"], "Unseparated")
        self.assertEqual(self.verdicts("makespan_s", [100.0, 125.0, 101.0], [120.0, 121.0, 122.0])["verdict"], "Unseparated")

    def test_peer_and_stock_are_baselines_for_hf2q_current(self):
        m = manifest(runs("h", "prefill_tps", "hf2q_current", [700.0, 701.0]) + runs("p", "prefill_tps", "peer", [600.0, 601.0])
                     + runs("s", "prefill_tps", "v0127_stock", [800.0, 801.0]))
        got = {c["baseline"][0]: (c["arm"][0], c["verdict"]) for c in r.comparisons(m)}
        self.assertEqual(got, {"peer": ("hf2q_current", "Win"), "v0127_stock": ("hf2q_current", "Regression")})

    def test_comparisons_stay_within_group(self):
        m = manifest(runs("c", "makespan_s", "candidate", [100.0, 101.0], group="g") +
                     runs("b", "makespan_s", "hf2q_current", [120.0, 121.0], group="h"))
        self.assertEqual(r.comparisons(m), [])


class Labels(unittest.TestCase):
    def test_descriptive_win_never_rendered_as_qualified(self):
        m = manifest(runs("c", "makespan_s", "candidate", [100.0, 101.0]) + runs("b", "makespan_s", "hf2q_current", [120.0, 121.0]))
        text = r.render(m, [], "0" * 64)
        self.assertIn("None yet: no F2 primary Qualified win exists.", section(text, "### Qualified wins"))
        self.assertIn("Win [D]", section(text, "### Descriptive wins"))

    def test_one_descriptive_row_makes_comparison_descriptive(self):
        rows = runs("c", "makespan_s", "candidate", [100.0, 101.0], status="Qualified")
        rows += runs("b", "makespan_s", "hf2q_current", [120.0], status="Qualified")
        rows += runs("d", "makespan_s", "hf2q_current", [121.0])
        (c,) = r.comparisons(manifest(rows))
        self.assertEqual(c["label"], "Descriptive")

    def test_negative_result_and_failure_rows_are_never_wins_or_baselines(self):
        rows = runs("c", "makespan_s", "candidate", [100.0, 101.0], status="NegativeResult")
        rows += runs("b", "makespan_s", "hf2q_current", [120.0, 121.0])
        rows += runs("p", "client_elapsed_s", "peer", [30.0, 31.0])
        rows += runs("f", "client_elapsed_s", "hf2q_current", [28.0, 28.5], status="Failure", note="agent failed")
        m = manifest(rows)
        comps = r.comparisons(m)
        self.assertEqual([(c["metric"], c["verdict"]) for c in comps], [("makespan_s", "NegativeResult")])
        text = r.render(m, [], "0" * 64)
        self.assertIn("None.", section(text, "### Descriptive wins"))
        self.assertIn("28.000 s [FAIL]: agent failed", section(text, "## Negative results"))

    def test_unverified_rows_are_excluded_and_reported(self):
        rows = runs("c", "makespan_s", "candidate", [100.0, 101.0]) + runs("b", "makespan_s", "hf2q_current", [120.0, 121.0])
        rows[0]["verification"] = {"state": "Unverified", "reason": "receipt missing"}
        m = manifest(rows)
        (c,) = r.comparisons(m)
        self.assertEqual(c["verdict"], "Unseparated")
        text = r.render(m, [], "0" * 64)
        self.assertIn("Unverified row `c0`: receipt missing", section(text, "## Open gaps"))
        self.assertIn("(unverified)", text)

    def test_effect_win_needs_positive_interval(self):
        m = manifest([effect("e1", 39.871, [39.518, 40.345]), effect("e2", 1.794, [-0.5, 3.5]),
                      effect("e3", 22.0, None), effect("e4", 5.0, [1.0, 9.0], status="NegativeResult")])
        text = r.render(m, [], "0" * 64)
        wins, losses = section(text, "### Descriptive wins"), section(text, "## Negative results")
        self.assertIn("`e1`", wins)
        for rid in ("e2", "e3", "e4"):
            self.assertNotIn(f"`{rid}`", wins)
            self.assertIn(f"`{rid}`", losses)


class Gaps(unittest.TestCase):
    def test_missing_and_failed_only_arms(self):
        rows = runs("p", "client_elapsed_s", "peer", [30.0])
        rows += runs("f", "task_pass", "hf2q_current", [0], status="Failure")
        text = section(r.render(manifest(rows), [], "0" * 64), "## Open gaps")
        self.assertIn("| Family | makespan | candidate, v0.1.27 stock | hf2q current |", text)
        self.assertIn("| Family | prefill_tps | candidate, v0.1.27 stock, peer | hf2q current |", text)
        self.assertIn("Family: no F2 primary Qualified fixed-six result.", text)


class Scoreboards(unittest.TestCase):
    def render(self, data):
        sb = r.load_scoreboard("fam", "$E/f2/score.json", data, {"fam": {}})
        return r.render(manifest([]), [sb], "0" * 64)

    def test_qualified_win_requires_interval_above_zero(self):
        self.assertIn("candidate vs hf2q current (B vs A)", section(self.render(scoreboard()), "### Qualified wins"))
        text = self.render(scoreboard(lo=-1.0, hi=2.0))
        self.assertIn("None yet", section(text, "### Qualified wins"))

    def test_inconclusive_is_shown_without_win(self):
        text = self.render(scoreboard(status="InconclusivePrecision", precision=False))
        self.assertIn("| Family | InconclusivePrecision | — | — | — |", text)
        self.assertIn("None yet", section(text, "### Qualified wins"))

    def test_cli_cites_scoreboard_symbolically_and_refuses_outside_root(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as other:
            rows = Path(root, "rows.json")
            data = manifest([])
            rows.write_text(json.dumps(data))
            Path(root, "f2").mkdir()
            Path(root, "f2", "score.json").write_text(json.dumps(scoreboard()))
            Path(other, "score.json").write_text(json.dumps(scoreboard()))
            out = Path(other, "out.md")
            base = ["--rows", str(rows), "--evidence-root", root, "--output", str(out)]
            self.assertEqual(r.main(base + ["--scoreboard", f"fam={root}/f2/score.json"]), 0)
            text = out.read_text()
            self.assertIn("`$E/f2/score.json`", text)
            self.assertNotIn(root, text)
            with self.assertRaises(ValueError):
                r.main(base + ["--scoreboard", f"fam={other}/score.json"])

    def test_inconsistent_qualified_and_foreign_schema_rejected(self):
        with self.assertRaises(ValueError):
            r.load_scoreboard("fam", "$E/x.json", scoreboard(precision=False), {"fam": {}})
        bad = scoreboard()
        bad["schema"] = "other"
        with self.assertRaises(ValueError):
            r.load_scoreboard("fam", "$E/x.json", bad, {"fam": {}})


class Validation(unittest.TestCase):
    def test_rejects_absolute_paths_wrong_units_and_bad_task_counts(self):
        good = runs("b", "makespan_s", "hf2q_current", [1.0])
        for mutate in (lambda x: x.update(receipt="/Users/someone/run/summary.json"),
                       lambda x: x.update(note="see /home/user/run"),
                       lambda x: x.update(unit="ms"),
                       lambda x: x.update(metric="task_pass", unit="tasks", value=5, decimals=0),
                       lambda x: x.update(status="Qualifed"),
                       lambda x: x.update(value=float("nan"))):
            rows = copy.deepcopy(good)
            mutate(rows[0])
            with self.assertRaises(ValueError):
                manifest(rows)

    def test_strict_json_rejects_duplicates_and_nonfinite(self):
        for raw in ('{"a": 1, "a": 2}', '{"a": NaN}'):
            with self.assertRaises(ValueError):
                r.strict_json(raw)


class Verification(unittest.TestCase):
    def test_verify_detects_receipt_drift(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root, "b0", "client")
            run.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({"elapsed_seconds": 115.5444}))
            m = manifest(runs("b", "client_elapsed_s", "hf2q_current", [115.544]))
            self.assertEqual(r.verify_rows(m, root), ["b0"])
            m["rows"][0]["value"] = 115.545
            with self.assertRaises(ValueError):
                r.verify_rows(m, root)

    def test_extract_ratio_wildcards_and_skip_null(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "s.json").write_text(json.dumps({"agents": {
                "b": {"turns": [{"t": None, "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}}}]},
                "a": {"turns": [{"t": 2.0, "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 60}}}]}}}))
            share = {"kind": "ratio", "scale": 100,
                     "numerator": {"kind": "json", "file": "$E/s.json", "reduce": "sum",
                                   "path": "agents.*.turns.*.usage.prompt_tokens_details.cached_tokens"},
                     "denominator": {"kind": "json", "file": "$E/s.json", "path": "agents.*.turns.*.usage.prompt_tokens", "reduce": "sum"}}
            self.assertEqual(r.extract(share, root, {}), 70.0)
            ttft = {"kind": "json", "file": "$E/s.json", "path": "agents.*.turns.*.t", "reduce": "median", "skip_null": True}
            self.assertEqual(r.extract(ttft, root, {}), 2.0)
            with self.assertRaises(ValueError):
                r.extract({"kind": "json", "file": "$E/../s.json", "path": "x", "reduce": "one"}, root + "/sub", {})


class Determinism(unittest.TestCase):
    def test_render_is_byte_stable_and_wins_ignore_row_order(self):
        rows = (runs("c", "makespan_s", "candidate", [100.0, 101.0]) + runs("b", "makespan_s", "hf2q_current", [120.0, 121.0])
                + runs("p", "prefill_tps", "peer", [600.0, 601.0]) + runs("h", "prefill_tps", "hf2q_current", [700.0, 701.0]))
        m = manifest(rows)
        self.assertEqual(r.render(m, [], "0" * 64), r.render(copy.deepcopy(m), [], "0" * 64))
        flipped = manifest(list(reversed(copy.deepcopy(rows))))
        for title in ("## Wins vs baseline", "## Negative results", "## Open gaps"):
            self.assertEqual(section(r.render(m, [], "0" * 64), title), section(r.render(flipped, [], "0" * 64), title))

    def test_checked_in_scoreboard_matches_checked_in_rows(self):
        rows_path = r.REPO / "docs/research/adr063-f2-scoreboard-rows.json"
        raw = rows_path.read_bytes()
        expected = r.render(r.validate_manifest(r.strict_json(raw)), [], hashlib.sha256(raw).hexdigest())
        self.assertEqual((r.REPO / "docs/research/adr063-f2-scoreboard.md").read_text(), expected)


if __name__ == "__main__":
    unittest.main()
