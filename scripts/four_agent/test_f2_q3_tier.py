"""Controls for the F2 'Q3 verdicts of record' tier: synthetic records and rows, plus a fake-runner campaign
through q3_report.rows(of_record=True). Never model, GPU or real receipt evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import f2_q3_tier as tier
import f2_scoreboard_render as r
import q3_report
import quality_campaign
import test_q3_campaign as base

SHA, COMMIT, EXP = "a" * 64, "b" * 40, "e" * 64
FAMILIES = {"qwen38": "Qwen3.8 27B", "deepseek4": "DeepSeek-V4 Flash", "qwen36": "Qwen3.6 35B-A3B", "gemma4": "Gemma4 26B-A4B"}
WIDTH_HALT = {"entry": "p3p1-047-w00-peer", "stage": "3+1",
              "reason": "p3p1-047-w00-peer is Invalid (run audit: observed co-progress width 3 < 4)"}


def gate(reference, status, q1=None):
    return {"reference": reference, "manifest": f"$E/qg/{reference}/manifest.json", "status": status,
            "q1_verdict": None if status == "Pending" else q1 or "Pass"}


def record(fam, verdict, gates, **over):
    out = {"verdict": verdict, "verdict_ref": f"$E/q3/{fam}/verdict.json", "campaign_ref": f"$E/q3/{fam}/campaign.json",
           "campaign_status": "Invalid", "halt": WIDTH_HALT, "primary_stop": {"reason": "precision", "blocks": 6},
           "reasons": [], "peer_unavailable": None, "thermal_valid": True, "memory_valid": True,
           "quality_gate": gates, "review": f"$E/reviews/{fam}.md"}
    out.update(over)
    return out


def row(fam, rid, metric, value, group=None, **over):
    ref = f"$E/q3/{fam}/verdict.json"
    out = {"id": rid, "family": fam, "group": group or f"{fam}-q3-record", "metric": metric, "kind": "measurement",
           "arm": "candidate", "variant": "Q3 experiment main HEAD", "value": value, "decimals": 3, "unit": "tokens",
           "n_slots": 4, "context": "262144", "date": "2026-11-01", "source": {"commit": COMMIT, "binary_sha256": EXP},
           "artifact_sha256": SHA, "receipt": ref, "status": "Descriptive", "q3_of_record": True,
           "verification": {"state": "Verified", "key": "k", "extract": {"value": {"kind": "json", "file": ref, "path": "x", "reduce": "one"}}},
           "note": ""}
    out.update(over)
    return out


def effect(fam, rid, metric, baseline, value, ci, hw=1.0, **over):
    out = row(fam, rid, metric, value, kind="effect", unit=r.EFFECT_UNIT, ci95=ci, baseline_arm=baseline, half_width_pp=hw, **over)
    out["verification"]["extract"].update(ci95={"kind": "json", "file": out["receipt"], "path": "c.*", "reduce": "list"},
                                          half_width_pp={"kind": "json", "file": out["receipt"], "path": "h", "reduce": "one"})
    return out


def fairness(fam, stock=((10.0, [5.0, 15.0]), (2.0, [-1.0, 5.0])), peer=None):
    out = []
    for baseline, values in (("v0128_stock", stock), ("peer", peer)):
        for metric, (value, ci) in zip(tier.FAIRNESS, values or ()):
            out.append(effect(fam, f"{fam}-{metric}-{baseline}", metric, baseline, value, ci))
    return out


def board(records, rows):
    groups = {f"{fam}-q3-record": {"family": fam, "title": f"{fam} Q3 verdict of record", "workload": "w", "q3_record": rec}
              for fam, rec in records.items()}
    groups["deepseek4-q3"] = {"family": "deepseek4", "title": "deepseek4 Q3 three-arm campaign", "workload": "w"}
    return r.validate_manifest({
        "schema": r.ROWS_SCHEMA, "evidence_root_symbol": "$E", "neutral_band_ratio": 1.025,
        "families": {k: {"title": v, "artifact_sha256": SHA, "note": ""} for k, v in FAMILIES.items()},
        "groups": groups, "rows": rows})


def tier_line(text, title):
    section = text[text.index("## Q3 verdicts of record"):text.index("Token volume and per-output-token rate")]
    [line] = [l for l in section.splitlines() if l.startswith(f"| {title} |")]
    return line.strip("| ").split(" | ")


def qwen38(gates):
    rows = [effect("qwen38", "q38-ms-stock", "makespan_s", "v0128_stock", 26.4, [24.7, 28.1], hw=2.22),
            effect("qwen38", "q38-ms-peer", "makespan_s", "peer", 31.5, [31.1, 31.9], hw=0.62),
            row("qwen38", "q38-ct-hf2q", "completion_tokens", 3531.0, decimals=1, receipt="$E/q3/qwen38/campaign.json"),
            row("qwen38", "q38-ct-peer", "completion_tokens", 6716.0, arm="peer", decimals=1, receipt="$E/q3/qwen38/campaign.json"),
            row("qwen38", "q38-rate-peer", "output_tokens_per_s", 52.7, arm="peer", unit="tok/s", receipt="$E/q3/qwen38/campaign.json")]
    rows += fairness("qwen38", stock=((30.4, [25.9, 34.7]), (6.2, [-26.8, 30.6])), peer=((-69.3, [-80.5, -58.8]), (20.0, [10.0, 30.0])))
    return board({"qwen38": record("qwen38", "Win", gates)}, rows)


class Tier(unittest.TestCase):
    def test_win_vs_stock_needs_the_quality_gate_against_both_references(self):
        text = r.render(qwen38([gate("stock-hf2q", "Pass"), gate("peer", "Pass")]), [], "0" * 64)
        cells = tier_line(text, "Qwen3.8 27B")
        self.assertEqual(cells[1], "Win")
        self.assertEqual(cells[2], "hf2q finishes 26.4% sooner than v0.1.28 stock (95% CI 24.7 to 28.1%, half-width 2.22 points); **Win**")
        self.assertEqual(cells[3], "hf2q finishes 31.5% sooner than the peer (95% CI 31.1 to 31.9%, half-width 0.62 points)")
        self.assertEqual(cells[4], "vs stock: TTFT p95 0.696 [0.653, 0.741], SSE gap max 0.938 [0.694, 1.268]; "
                                   "vs peer: TTFT p95 1.693 [1.588, 1.805], SSE gap max 0.800 [0.700, 0.900]")
        self.assertEqual(cells[6], "Invalid: halted at 3+1 entry `p3p1-047-w00-peer` (p3p1-047-w00-peer is Invalid (run audit: "
                                   "observed co-progress width 3 < 4)); the primary stopped on precision after 6 blocks")
        self.assertEqual(cells[7], "`$E/q3/qwen38/verdict.json`; review `$E/reviews/qwen38.md`")
        self.assertIn("| Qwen3.6 35B-A3B | No Q3 verdict of record |", text)
        self.assertIn("| Qwen3.8 27B | peer | not recorded | 6716.0 | 52.700 |", text)
        for title in ("### Descriptive wins", "## Negative results", "## Open gaps"):  # rows of record render only in the tier
            start = text.index(title)
            self.assertNotIn("`q38-ms-stock`", text[start:text.find("\n## ", start + len(title))])
        self.assertIn("- Qwen3.8 27B: `q38-ct-hf2q`, `q38-ct-peer`, `q38-ms-peer`, `q38-ms-stock`", text)

    def test_quality_gate_pending_blocks_win(self):
        text = r.render(qwen38([gate("stock-hf2q", "Pass"), gate("peer", "Pending")]), [], "0" * 64)
        cells = tier_line(text, "Qwen3.8 27B")
        self.assertNotIn("**Win**", " | ".join(cells))
        self.assertEqual(cells[1], "Withheld: quality gate Pending vs peer")
        self.assertTrue(cells[2].endswith("half-width 2.22 points); not a win: quality gate Pending vs peer"))
        self.assertEqual(cells[5], "v0.1.28 stock Pass `$E/qg/stock-hf2q/manifest.json`; peer Pending `$E/qg/peer/manifest.json`")
        undeclared = r.render(qwen38([gate("stock-hf2q", "Pass")]), [], "0" * 64)
        self.assertEqual(tier_line(undeclared, "Qwen3.8 27B")[1], "Withheld: quality gate vs the peer not declared")

    def test_loss_states_the_peer_sooner_on_the_slower_arms_base(self):
        rows = [effect("gemma4", "g-ms-stock", "makespan_s", "v0128_stock", 41.6, [40.8, 42.3], hw=1.24),
                effect("gemma4", "g-ms-peer", "makespan_s", "peer", -80.0, [-100.0, -60.0], hw=2.53)]
        rows += fairness("gemma4", stock=((20.0, [10.0, 30.0]), (-30.0, [-45.0, -15.0])))
        gates = [gate("stock-hf2q", "Pass"), gate("peer", "Pass")]
        cells = tier_line(r.render(board({"gemma4": record("gemma4", "Loss", gates)}, rows), [], "0" * 64), "Gemma4 26B-A4B")
        self.assertEqual(cells[1], "Loss")
        # hf2q/peer = 1.8: the peer finishes 1 - 1/1.8 = 44.4% sooner, never 80% (hf2q's base)
        self.assertEqual(cells[3], "the peer finishes 44.4% sooner than hf2q (95% CI 37.5 to 50.0%, half-width 2.53 points)")
        self.assertEqual(cells[2], "hf2q finishes 41.6% sooner than v0.1.28 stock (95% CI 40.8 to 42.3%, half-width 1.24 points); "
                                   "not a win: worst-slot SSE gap max vs stock regresses beyond the ×1.025 band (ratio CI lower 1.150)")

    def test_peer_sentence_names_the_faster_arm_and_flags_a_tie(self):
        self.assertEqual(tier.sooner({"value": 2.0, "ci95": [-1.0, 5.0]}, "the peer"),
                         "hf2q finishes 2.0% sooner than the peer (95% CI -1.0 to 5.0%; the interval includes a tie)")
        self.assertEqual(tier.sooner({"value": -25.0, "ci95": None}, "the peer"), "the peer finishes 20.0% sooner than hf2q (no CI)")

    def test_unavailable_shows_the_native_ds4_statement(self):
        rows = [effect("deepseek4", "ds-ms-stock", "makespan_s", "v0128_stock", 24.7, [23.9, 25.6], hw=1.17)] + fairness("deepseek4")
        statement = ("main HEAD finishes the same coding task 5.2% sooner than ds4 c97964b (native Flash-0731 q2 artifact) "
                     "(95% CI 3.0..7.4%, 15 blocks; different artifacts, task makespan only)")
        native = effect("deepseek4", "ds-native", "makespan_s", "ds4_native", 5.218, [3.006, 7.380], group="deepseek4-q3", note=statement)
        for key in ("q3_of_record", "half_width_pp"):
            native.pop(key)
        native["verification"]["extract"].pop("half_width_pp")
        rec = record("deepseek4", "Unavailable", [gate("stock-hf2q", "Pass"), gate("ds4-native", "Pass")],
                     peer_unavailable="declared from prior evidence", reasons=["no runnable peer arm: declared from prior evidence"])
        text = r.render(board({"deepseek4": rec}, rows + [native]), [], "0" * 64)
        cells = tier_line(text, "DeepSeek-V4 Flash")
        self.assertEqual(cells[1], "Unavailable")
        self.assertTrue(cells[2].endswith("; **Win**"))  # Win vs stock: every stock condition and both quality gates pass
        self.assertEqual(cells[3], f"Unavailable: declared from prior evidence. Native {r.BASELINE_ARMS['ds4_native']} "
                                   f"(different artifact, task makespan only): {statement} (`ds-native`)")
        self.assertEqual(cells[4], "vs stock: TTFT p95 0.900 [0.850, 0.950], SSE gap max 0.980 [0.950, 1.010]; vs peer: —")
        self.assertIn("`ds-native`", text[text.index("## Native-engine comparisons"):])  # keeps its own section

    def test_failed_family_shows_its_reasons_and_never_wins(self):
        reason = "experiment failed its task: n4-002-w00-experiment: client_failed: agent-0 Failed"
        rec = record("qwen36", "Failed", [], campaign_status="Complete", halt=None, reasons=[reason],
                     primary_stop={"reason": "experiment_failed", "blocks": 0})
        cells = tier_line(r.render(board({"qwen36": rec}, []), [], "0" * 64), "Qwen3.6 35B-A3B")
        self.assertEqual(cells[1], f"Failed: {reason}")
        self.assertTrue(cells[2].startswith("—; not a win: the family verdict is Failed (hf2q task failure); no makespan effect vs stock"))
        self.assertEqual((cells[3], cells[5], cells[6]), ("—", "not declared", "Complete; the primary stopped on experiment_failed after 0 blocks"))


class Validation(unittest.TestCase):
    def test_refuses_inconsistent_records_and_rows(self):
        good = qwen38([gate("stock-hf2q", "Pass"), gate("peer", "Pending")])
        mutations = (lambda m: m["groups"]["qwen38-q3-record"]["q3_record"]["quality_gate"][0].update(q1_verdict=None),
                     lambda m: m["groups"]["qwen38-q3-record"]["q3_record"]["quality_gate"][1].update(reference="peer-x"),
                     lambda m: m["groups"]["qwen38-q3-record"]["q3_record"].update(verdict_ref="$E/q3/other/verdict.json"),
                     lambda m: m["groups"].update(other={**m["groups"]["qwen38-q3-record"]}),
                     lambda m: m["rows"][0].update(group="deepseek4-q3", family="deepseek4"),
                     lambda m: m["rows"][0].update(baseline_arm="hf2q_current"),
                     lambda m: m["rows"][0]["verification"]["extract"].pop("half_width_pp"),
                     lambda m: m["groups"]["qwen38-q3-record"]["q3_record"]["halt"].update(reason="/Users/x/campaign.json"))
        for mutate in mutations:
            m = copy.deepcopy(good)
            mutate(m)
            with self.assertRaises(ValueError):
                r.validate_manifest(m)

    def test_vocabularies_bind_quality_campaign_and_the_q3_rule(self):
        self.assertEqual(set(tier.REFERENCES), set(quality_campaign.REFERENCE_KINDS))
        self.assertEqual(tuple(tier.FAIRNESS), q3_report.FAIRNESS)


def quality_manifest(path, run, reference, binary, status="Complete", q1="Pass"):
    run.mkdir(parents=True, exist_ok=True)
    (run / "receipt.json").write_text(json.dumps({"binary": {"sha256": binary}}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"plan": {"reference_kind": reference, "arms": {"subject": {"family": "qwen38"}}},
                                "status": status, "verdict": {"verdict": q1} if q1 else None,
                                "runs": [{"arm": "subject", "attempts": [{"output": str(run)}]}]}))


class Verify(unittest.TestCase):
    def test_record_rederives_from_receipts_and_refuses_gate_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "q3" / "qwen38").mkdir(parents=True)
            verdict = {"family": "qwen38", "verdict": "Win", "campaign_status": "Invalid", "reasons": [],
                       "peer": {"unavailable": {}}, "thermal_valid": True, "memory_valid": True,
                       "primary": {"stop": {"reason": "precision", "blocks": 6}}}
            campaign = {"identities": {"arms": {"experiment": {"binary": {"sha256": EXP}}}},
                        "halted": {"entry": "p3p1-047-w00-peer", "reason": f"width 3 < 4 in {root}/q3/qwen38/runs/x"},
                        "stages": [{"workload": "3+1", "warmups": [{"name": "p3p1-047-w00-peer"}], "blocks": []}]}
            (root / "q3/qwen38/verdict.json").write_text(json.dumps(verdict))
            (root / "q3/qwen38/campaign.json").write_text(json.dumps(campaign))
            peer = root / "qg/peer/manifest.json"
            quality_manifest(peer, root / "qg/peer/run", "peer", EXP)
            stock = root / "qg/stock-hf2q/manifest.json"
            gates = [("stock-hf2q", "$E/qg/stock-hf2q/manifest.json", None),
                     ("peer", "$E/qg/peer/manifest.json", json.loads(peer.read_text()))]
            rec = tier.build_record(verdict, campaign, root, ("$E/q3/qwen38/verdict.json", "$E/q3/qwen38/campaign.json"), gates, None)
            self.assertEqual(rec["halt"], {"entry": "p3p1-047-w00-peer", "stage": "3+1", "reason": "width 3 < 4 in $E/q3/qwen38/runs/x"})
            self.assertEqual([(g["status"], g["q1_verdict"]) for g in rec["quality_gate"]], [("Pending", None), ("Pass", "Pass")])
            m = board({"qwen38": rec}, [])
            self.assertEqual(r.verify_rows(m, root), [])
            quality_manifest(stock, root / "qg/stock-hf2q/run", "stock-hf2q", EXP, q1="BelowFloor")  # the gate settled
            with self.assertRaisesRegex(ValueError, "quality_gate drift"):
                r.verify_rows(m, root)
            quality_manifest(peer, root / "qg/peer/run", "peer", "0" * 64)
            with self.assertRaisesRegex(ValueError, "not run on the experiment binary"):
                tier.build_record(verdict, campaign, root, ("$E/a/verdict.json", "$E/a/campaign.json"),
                                  [("peer", "$E/qg/peer/manifest.json", json.loads(peer.read_text()))], None)


class RowsOfRecord(base.Harness):
    win = base.Driver.win

    def test_rows_of_record_validate_verify_and_render_the_tier(self):
        driver, _ = self.driver(self.win)
        base.q3.write_verdict(self.out, driver.run(), [], None)
        stock = self.root / "qg" / "stock" / "manifest.json"
        quality_manifest(stock, self.root / "qg" / "stock" / "run", "stock-hf2q", base.EXP_SHA)
        review = self.root / "reviews" / "q3-qwen38.md"
        review.parent.mkdir()
        review.write_text("ACCEPT")
        gates = [("stock-hf2q", stock), ("peer", self.root / "qg" / "peer" / "manifest.json")]
        rows = q3_report.rows(self.out, self.root, True, gates, review)
        rec = rows["groups"]["qwen38-q3-record"]["q3_record"]
        self.assertEqual((rec["verdict"], rec["review"]), ("Win", "$E/reviews/q3-qwen38.md"))
        self.assertEqual([(g["reference"], g["status"]) for g in rec["quality_gate"]], [("stock-hf2q", "Pass"), ("peer", "Pending")])
        verdict = json.loads((self.out / "verdict.json").read_text())
        fair = {(m, {"vs_stock": "v0128_stock", "vs_peer": "peer"}[n]) for m in tier.FAIRNESS for n in ("vs_stock", "vs_peer")
                if verdict["primary"]["secondary"][m][n]["improvement_percent"] is not None}
        self.assertLessEqual({(m, "v0128_stock") for m in tier.FAIRNESS}, fair)  # the stock fairness gate is decidable
        got = {(x["metric"], x.get("baseline_arm") or x["arm"]) for x in rows["rows"]}
        self.assertEqual(got, {("makespan_s", "v0128_stock"), ("makespan_s", "peer"), *fair,
                               *((m, a) for m in tier.TOKENS for a in ("candidate", "v0128_stock", "peer"))})
        self.assertTrue(all(x["q3_of_record"] is True for x in rows["rows"]))
        document = {"schema": r.ROWS_SCHEMA, "evidence_root_symbol": "$E", "neutral_band_ratio": 1.025,
                    "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": base.MODEL_SHA, "note": ""}}, **rows}
        r.validate_manifest(document)
        self.assertEqual(r.verify_rows(document, self.root), [x["id"] for x in rows["rows"]])
        self.assertEqual(tier_line(r.render(document, [], "0" * 64), "Qwen3.8 27B")[1], "Withheld: quality gate Pending vs peer")
        with self.assertRaisesRegex(ValueError, "unknown quality gate reference"):
            q3_report.rows(self.out, self.root, True, [("bogus", stock)])

    def test_cli_options_need_of_record(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            base.q3.parse_args(["--rows", str(self.out), "--evidence-root", str(self.root), "--quality-gate", "peer=x"])
        args = base.q3.parse_args(["--rows", str(self.out), "--evidence-root", str(self.root), "--of-record",
                                   "--quality-gate", "stock-hf2q=/q/m.json"])
        self.assertEqual((args.of_record, args.quality_gate), (True, ["stock-hf2q=/q/m.json"]))


if __name__ == "__main__":
    unittest.main()
