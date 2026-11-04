"""Controls for the rendered F2 'Q3 verdicts of record' tier and its manifest validation: synthetic records and
rows only. Receipt re-derivation, quality-gate binding and q3_report.rows(of_record=True) are in test_f2_q3_record.py."""
import copy
import unittest

import f2_q3_tier as tier
import f2_q3_view as view
import f2_scoreboard_render as r
import q3_report
import quality_campaign

SHA, COMMIT, EXP = "a" * 64, "b" * 40, "e" * 64
FAMILIES = {"qwen38": "Qwen3.8 27B", "deepseek4": "DeepSeek-V4 Flash", "qwen36": "Qwen3.6 35B-A3B", "gemma4": "Gemma4 26B-A4B"}
WIDTH_HALT = {"entry": "p3p1-047-w00-peer", "stage": "3+1",
              "reason": "p3p1-047-w00-peer is Invalid (run audit: observed co-progress width 3 < 4)"}
V7_STAGES = [["n4", "Complete"], ["n1", "Complete"], ["3+1", "Running"], ["2+2", "Pending"], ["1+3", "Pending"]]
V7_CONTROLS = "controls measured: n1; unmeasured: 3+1 (halted); not run: 2+2, 1+3"
PENDING_SOURCE = "ab" * 32


def gate(reference, status, q1=None):
    settled = status in ("Pass", "Fail")
    return {"reference": reference, "manifest": f"$E/qg/{reference}/manifest.json", "status": status,
            "q1_verdict": (q1 or "Pass") if settled else None, "reference_sha256": "c" * 64 if settled else None}


def record(fam, verdict, gates, **over):
    out = {"verdict": verdict, "verdict_ref": f"$E/q3/{fam}/verdict.json", "campaign_ref": f"$E/q3/{fam}/campaign.json",
           "campaign_status": "Invalid", "halt": WIDTH_HALT, "primary_stop": {"reason": "precision", "blocks": 6},
           "stages": V7_STAGES, "reasons": [], "experiment": "hf2q 38485a23", "peer_unavailable": None,
           "unmeasurable": [], "thermal_valid": True, "memory_valid": True, "patch_review_pending": [],
           "quality_gate": gates, "review": f"$E/reviews/{fam}.md"}
    out.update(over)
    return out


def row(fam, rid, metric, value, group=None, **over):
    ref = f"$E/q3/{fam}/verdict.json"
    out = {"id": rid, "family": fam, "group": group or f"{fam}-q3-record", "metric": metric, "kind": "measurement",
           "arm": "candidate", "variant": "Q3 experiment hf2q 38485a23", "value": value, "decimals": 3, "unit": "tokens",
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


def plain(out):
    """The same row outside the record: no q3_of_record flag, no half-width."""
    for key in ("q3_of_record", "half_width_pp"):
        out.pop(key, None)
    out["verification"]["extract"].pop("half_width_pp", None)
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


def qwen38(gates, **over):
    rows = [effect("qwen38", "q38-ms-stock", "makespan_s", "v0128_stock", 26.4, [24.7, 28.1], hw=2.221),
            effect("qwen38", "q38-ms-peer", "makespan_s", "peer", 31.5, [31.1, 31.9], hw=0.6155),
            row("qwen38", "q38-ct-hf2q", "completion_tokens", 3531.0, decimals=1, receipt="$E/q3/qwen38/campaign.json"),
            row("qwen38", "q38-ct-peer", "completion_tokens", 6716.0, arm="peer", decimals=1, receipt="$E/q3/qwen38/campaign.json"),
            row("qwen38", "q38-rate-peer", "output_tokens_per_s", 52.7, arm="peer", unit="tok/s", receipt="$E/q3/qwen38/campaign.json")]
    rows += fairness("qwen38", stock=((30.4, [25.9, 34.7]), (6.2, [-26.8, 30.6])), peer=((-69.3, [-80.5, -58.8]), (20.0, [10.0, 30.0])))
    return board({"qwen38": record("qwen38", "Win", gates, **over)}, rows)


BOTH = [gate("stock-hf2q", "Pass"), gate("peer", "Pass")]


class Tier(unittest.TestCase):
    def test_win_vs_stock_needs_the_quality_gate_against_both_references(self):
        text = r.render(qwen38(BOTH), [], "0" * 64)
        cells = tier_line(text, "Qwen3.8 27B")
        self.assertEqual(cells[1], "Win")
        self.assertEqual(cells[2], "hf2q 38485a23 finishes 26.4% sooner than v0.1.28 stock (95% CI 24.7 to 28.1%, "
                                   "half-width 2.22 points); **Win**")
        # the stored full-precision half-width 0.6155 renders as 0.62 (the review's figure), never 0.61
        self.assertEqual(cells[3], "hf2q 38485a23 finishes 31.5% sooner than the peer (95% CI 31.1 to 31.9%, half-width 0.62 points)")
        self.assertEqual(cells[4], "vs stock: TTFT p95 0.696 [0.653, 0.741], SSE gap max 0.938 [0.694, 1.268] (no regression "
                                   "demonstrated); vs peer: TTFT p95 1.693 [1.588, 1.805], SSE gap max not comparable "
                                   "(different event granularity)")
        self.assertEqual(cells[6], "Invalid: halted at 3+1 entry `p3p1-047-w00-peer` (p3p1-047-w00-peer is Invalid (run audit: "
                                   f"observed co-progress width 3 < 4)); the primary stopped on precision after 6 blocks; {V7_CONTROLS}")
        self.assertEqual(cells[7], "`$E/q3/qwen38/verdict.json`; review `$E/reviews/qwen38.md`")
        self.assertIn("| Qwen3.6 35B-A3B | No Q3 verdict of record |", text)
        self.assertIn("| Qwen3.8 27B | hf2q 38485a23 | not recorded | 3531.0 | not recorded |", text)
        self.assertIn("| Qwen3.8 27B | peer | not recorded | 6716.0 | 52.700 |", text)
        for title in ("### Descriptive wins", "## Negative results", "## Open gaps"):  # rows of record render only in the tier
            start = text.index(title)
            self.assertNotIn("`q38-ms-stock`", text[start:text.find("\n## ", start + len(title))])
        self.assertIn("- Qwen3.8 27B: `q38-ct-hf2q`, `q38-ct-peer`, `q38-ms-peer`, `q38-ms-stock`", text)
        self.assertIn("None yet: no fixed-six F2 primary Qualified win exists (successor scope). Under option B, F2's "
                      "qualified display is the Q3 verdicts of record tier above.", text)

    def test_quality_gate_pending_halted_undeclared_or_wrong_kind_blocks_win(self):
        cases = {"Withheld: quality gate Pending vs peer": [gate("stock-hf2q", "Pass"), gate("peer", "Pending")],
                 "Withheld: quality gate Halted vs peer": [gate("stock-hf2q", "Pass"), gate("peer", "Halted")],
                 "Withheld: quality gate Fail vs v0.1.28 stock (BelowFloor)": [gate("stock-hf2q", "Fail", "BelowFloor"), gate("peer", "Pass")],
                 "Withheld: quality gate vs peer not declared": [gate("stock-hf2q", "Pass")],
                 # a native gate never stands in for the runnable same-GGUF peer
                 "Withheld: quality gate vs peer not declared ": [gate("stock-hf2q", "Pass"), gate("tensorfold-native", "Pass")]}
        for want, gates in cases.items():
            cells = tier_line(r.render(qwen38(gates), [], "0" * 64), "Qwen3.8 27B")
            self.assertNotIn("**Win**", " | ".join(cells))
            self.assertEqual(cells[1], want.strip())
            self.assertTrue(cells[2].endswith("half-width 2.22 points); not a Q3 Win: " + want.strip()[len("Withheld: "):]))
        cells = tier_line(r.render(qwen38([gate("stock-hf2q", "Pass"), gate("peer", "Halted")]), [], "0" * 64), "Qwen3.8 27B")
        self.assertEqual(cells[5], "v0.1.28 stock Pass `$E/qg/stock-hf2q/manifest.json`; peer Halted `$E/qg/peer/manifest.json`")

    def test_pending_patch_review_blocks_win(self):
        cells = tier_line(r.render(qwen38(BOTH, patch_review_pending=[PENDING_SOURCE]), [], "0" * 64), "Qwen3.8 27B")
        self.assertEqual(cells[1], "Withheld: patch review pending for 1 final source(s)")
        self.assertTrue(cells[5].endswith("; patch review pending: `abababababab`"))

    def test_loss_states_the_peer_sooner_on_the_slower_arms_base_and_discloses_imprecision(self):
        rows = [effect("gemma4", "g-ms-stock", "makespan_s", "v0128_stock", 41.6, [40.8, 42.3], hw=1.24),
                effect("gemma4", "g-ms-peer", "makespan_s", "peer", -80.0, [-100.0, -60.0], hw=3.31)]
        rows += fairness("gemma4", stock=((20.0, [10.0, 30.0]), (-30.0, [-45.0, -15.0])))
        rec = record("gemma4", "Loss", BOTH, experiment="hf2q 5d421a2c", primary_stop={"reason": "max_blocks", "blocks": 20})
        cells = tier_line(r.render(board({"gemma4": rec}, rows), [], "0" * 64), "Gemma4 26B-A4B")
        self.assertEqual(cells[1], "Loss")
        # hf2q/peer = 1.8: the peer finishes 1 - 1/1.8 = 44.4% sooner, never 80% (hf2q's base)
        self.assertEqual(cells[3], "the peer finishes 44.4% sooner than hf2q 5d421a2c (95% CI 37.5 to 50.0%, half-width 3.31 points)")
        self.assertEqual(cells[2], "hf2q 5d421a2c finishes 41.6% sooner than v0.1.28 stock (95% CI 40.8 to 42.3%, half-width "
                                   "1.24 points); not a Q3 Win: the family verdict is Loss vs the peer; worst-slot SSE gap max vs stock regresses beyond the ×1.025 band "
                                   "(ratio CI lower 1.150)")
        self.assertIn("; the primary stopped on max_blocks after 20 blocks, precision not met (vs-peer half-width 3.31 points); "
                      + V7_CONTROLS, cells[6])

    def test_peer_sentence_names_the_faster_arm_and_flags_a_tie(self):
        self.assertEqual(view.sooner({"value": 2.0, "ci95": [-1.0, 5.0]}, "the peer"),
                         "hf2q finishes 2.0% sooner than the peer (95% CI -1.0 to 5.0%; the interval includes a tie)")
        self.assertEqual(view.sooner({"value": -25.0, "ci95": None}, "the peer", "hf2q 5d421a2c"),
                         "the peer finishes 20.0% sooner than hf2q 5d421a2c (no CI)")

    def test_unavailable_shows_the_native_ds4_statement_and_an_unmeasurable_sse_gap_does_not_gate(self):
        rows = [effect("deepseek4", "ds-ms-stock", "makespan_s", "v0128_stock", 24.7, [23.9, 25.6], hw=1.17)]
        rows += fairness("deepseek4", stock=((10.0, [5.0, 15.0]),))  # the verdict's SSE effect is null: no row
        statement = ("hf2q 38485a23 finishes the same coding task 5.2% sooner than ds4 c97964b (native Flash-0731 q2 artifact) "
                     "(95% CI 3.0..7.4%, 15 blocks; different artifacts, task makespan only)")
        native = plain(effect("deepseek4", "ds-native", "makespan_s", "ds4_native", 5.218, [3.006, 7.380],
                              group="deepseek4-q3", note=statement, receipt="$E/q3/deepseek4-native-ds4/verdict.json"))
        rec = record("deepseek4", "Unavailable", [gate("stock-hf2q", "Pass"), gate("ds4-native", "Pass")],
                     peer_unavailable="peer out of memory (prior evidence)", unmeasurable=["worst_slot_sse_gap_max_s.vs_stock"],
                     reasons=["no runnable peer arm: declared from prior evidence"])
        text = r.render(board({"deepseek4": rec}, rows + [native]), [], "0" * 64)
        cells = tier_line(text, "DeepSeek-V4 Flash")
        self.assertEqual(cells[1], "Unavailable")
        self.assertTrue(cells[2].endswith("; **Win**"))  # the unestimable SSE gap passes, as in Q3 rule 6
        self.assertEqual(cells[3], f"Unavailable: peer out of memory (prior evidence). Native {r.BASELINE_ARMS['ds4_native']} "
                                   f"(different artifact, task makespan only): {statement} (`ds-native`)")
        self.assertEqual(cells[4], "vs stock: TTFT p95 0.900 [0.850, 0.950], SSE gap max not measurable (one semantic event "
                                   "per streamed turn); vs peer: —")
        self.assertTrue(cells[6].endswith("not run: 2+2, 1+3"))
        self.assertIn("`ds-native`", text[text.index("## Native-engine comparisons"):])  # keeps its own section
        # the same absence without the verdict's null effect is "not recorded" and blocks; a peer gate is the wrong reference
        for over, reason in (({"unmeasurable": []}, "worst-slot SSE gap max vs stock not recorded"),
                             ({"quality_gate": [gate("stock-hf2q", "Pass"), gate("peer", "Pass")]}, "quality gate vs ds4 native not declared")):
            cells = tier_line(r.render(board({"deepseek4": {**rec, **over}}, rows), [], "0" * 64), "DeepSeek-V4 Flash")
            self.assertTrue(cells[2].endswith(f"; not a Q3 Win: {reason}"), cells[2])

    def test_failed_family_shows_its_failure_row_and_never_wins(self):
        reason = "experiment failed its task: n4-002-w00-experiment: client_failed: agent-0 Failed"
        rec = record("qwen36", "Failed", [], campaign_status="Complete", halt=None, reasons=[reason],
                     primary_stop={"reason": "experiment_failed", "blocks": 0},
                     stages=[["n4", "Stopped"], ["n1", "NotRun"], ["3+1", "NotRun"], ["2+2", "NotRun"], ["1+3", "NotRun"]])
        failure = row("qwen36", "q36-fail", "task_pass", 1, unit="tasks", decimals=0, status="Failure",
                      receipt="$E/q3/qwen36/campaign.json", note="agent-2: run_tests ports failed with E0282")
        cells = tier_line(r.render(board({"qwen36": rec}, [failure]), [], "0" * 64), "Qwen3.6 35B-A3B")
        self.assertEqual(cells[1], f"Failed: {reason}")
        self.assertEqual(cells[2], "hf2q 38485a23: no speed effect; not a Q3 Win: the family verdict is Failed (hf2q task "
                                   "failure). 1/4 tasks [FAIL]: agent-2: run_tests ports failed with E0282 (`q36-fail`)")
        self.assertEqual((cells[3], cells[4], cells[5], cells[6]),
                         ("—", "— (no paired block)", "not declared",
                          "Complete; the primary stopped on experiment_failed after 0 blocks; not run: n1, 3+1, 2+2, 1+3"))
        # a task_pass row of record is only the Failure of a Failed verdict, read from its campaign.json
        for over in ({"status": "Descriptive"}, {"receipt": "$E/q3/qwen36/verdict.json"}):
            with self.subTest(over=over), self.assertRaises(ValueError):
                board({"qwen36": rec}, [{**failure, **over}])
        with self.assertRaises(ValueError):
            board({"qwen38": record("qwen38", "Win", BOTH)}, [row("qwen38", "q38-fail", "task_pass", 1, unit="tasks",
                  decimals=0, status="Failure", receipt="$E/q3/qwen38/campaign.json")])

    def test_a_failed_primary_has_no_unmeasurable_effect(self):
        null = {"blocks": 0, "improvement_percent": None}
        verdict = {"verdict": "Failed", "family": "qwen36", "campaign_status": "Complete", "reasons": [],
                   "primary": {"stop": {"reason": "experiment_failed", "blocks": 0},
                               "secondary": {m: {"vs_stock": null, "vs_peer": null} for m in tier.FAIRNESS}},
                   "peer": {"unavailable": {}}, "thermal_valid": True, "memory_valid": True,
                   "quality": {"patch_review": {"pending": []}}}
        campaign = {"identities": {"arms": {}}, "stages": [],
                    "plan": {"declared": {"experiment": {"label": "main HEAD", "commit": COMMIT}}}}
        build = lambda: tier.build_record(verdict, campaign, "/nonexistent", ("$E/v.json", "$E/c.json"), [], None)
        self.assertEqual((build()["unmeasurable"], build()["experiment"]), ([], f"hf2q {COMMIT[:8]}"))
        verdict["primary"]["stop"] = {"reason": "precision", "blocks": 5}  # paired blocks, but the metric was empty
        self.assertEqual(build()["unmeasurable"], [f"{m}.{c}" for m in tier.FAIRNESS for c in ("vs_stock", "vs_peer")])

    def test_family_notes_render_under_the_tier_and_validate(self):
        document = qwen38(BOTH)
        document["q3_family_notes"] = {"qwen38": ["**Vs peer:** sentence of record."]}
        text = r.render(r.validate_manifest(document), [], "0" * 64)
        notes = text[text.index("### Family notes"):text.index("## Wins vs baseline")]
        self.assertEqual(notes, "### Family notes\n\n**Qwen3.8 27B**\n\n- **Vs peer:** sentence of record.\n\n")
        for bad in ({"nope": ["x"]}, {"qwen38": []}, {"qwen38": [""]}, {"qwen38": "x"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                r.validate_manifest({**document, "q3_family_notes": bad})


class Validation(unittest.TestCase):
    def test_refuses_inconsistent_records_and_rows(self):
        good = qwen38([gate("stock-hf2q", "Pass"), gate("peer", "Pending")])
        rec = lambda m: m["groups"]["qwen38-q3-record"]["q3_record"]
        mutations = (lambda m: rec(m)["quality_gate"][0].update(q1_verdict=None),
                     lambda m: rec(m)["quality_gate"][0].update(reference_sha256=None),
                     lambda m: rec(m)["quality_gate"][1].update(reference="peer-x"),
                     lambda m: rec(m).update(verdict_ref="$E/q3/other/verdict.json"),
                     lambda m: rec(m).update(stages=[["n4"]]),
                     lambda m: m["groups"].update(other={**m["groups"]["qwen38-q3-record"]}),
                     lambda m: m["rows"][0].update(group="deepseek4-q3", family="deepseek4"),
                     lambda m: m["rows"][0].update(baseline_arm="hf2q_current"),
                     lambda m: m["rows"][0]["verification"]["extract"].pop("half_width_pp"),
                     lambda m: rec(m)["halt"].update(reason="/Users/x/campaign.json"))
        for mutate in mutations:
            m = copy.deepcopy(good)
            mutate(m)
            with self.assertRaises(ValueError):
                r.validate_manifest(m)

    def test_refuses_a_duplicate_merge_but_keeps_controls_of_the_record(self):
        good = qwen38(BOTH)
        duplicate = plain(effect("deepseek4", "dup-ms-stock", "makespan_s", "v0128_stock", 26.4, [24.7, 28.1],
                                 group="deepseek4-q3", receipt="$E/q3/qwen38/verdict.json"))
        m = copy.deepcopy(good)
        m["rows"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "merge its --of-record fragment"):
            r.validate_manifest(m)
        m["rows"][-1].update(control_of="$E/q3/qwen38/verdict.json", id="n1-ms-stock")
        r.validate_manifest(m)
        text = r.render(m, [], "0" * 64)
        self.assertIn("`n1-ms-stock`", text[text.index("## Q3 controls of a primary verdict of record"):])
        self.assertIn("None.", text[text.index("### Descriptive wins"):text.index("## Negative results")])

    def test_vocabularies_bind_quality_campaign_and_the_q3_rule(self):
        self.assertEqual(set(tier.REFERENCES), set(quality_campaign.REFERENCE_KINDS))
        self.assertEqual(tuple(tier.FAIRNESS), q3_report.FAIRNESS)
        self.assertEqual(tier.required_references("deepseek4", "peer out of memory (prior evidence)"), ["stock-hf2q", "ds4-native"])
        self.assertEqual(tier.required_references("qwen38", None), ["stock-hf2q", "peer"])


class StockRecordTier(unittest.TestCase):
    """A --stock-record campaign's record (ADR-063 Q4-DS4-RACE decision D1): its vs-stock terms come from the bound
    same-artifact record on hf2q's artifact, and the tier says so."""
    BOUND = {"verdict_ref": "$E/q3/deepseek4-hf2q-artifact/verdict.json", "artifact_sha256": "f" * 64,
             "stock_label": "v0.1.28 release"}

    def test_the_cells_name_the_hf2q_artifact_record(self):
        text = r.render(qwen38(BOTH, stock_record=self.BOUND), [], "0" * 64)
        cells = tier_line(text, "Qwen3.8 27B")
        self.assertEqual(cells[2], "hf2q 38485a23 finishes 26.4% sooner than v0.1.28 stock (hf2q artifact record) (95% CI "
                                   "24.7 to 28.1%, half-width 2.22 points); **Win**")
        self.assertTrue(cells[4].startswith("vs stock (hf2q artifact record): TTFT p95 0.696 [0.653, 0.741]"))
        plain_cells = tier_line(r.render(qwen38(BOTH), [], "0" * 64), "Qwen3.8 27B")  # records before it: unchanged
        self.assertTrue(plain_cells[4].startswith("vs stock: TTFT p95"))
        # the gate references stay [stock-hf2q, peer]; the stock-hf2q gate binds the record (test_f2_q3_record)
        self.assertEqual(tier.required_references("deepseek4", None), ["stock-hf2q", "peer"])
        self.assertIn("quality gate Pending vs peer",
                      tier_line(r.render(qwen38([gate("stock-hf2q", "Pass"), gate("peer", "Pending")],
                                                stock_record=self.BOUND), [], "0" * 64), "Qwen3.8 27B")[1])

    def test_the_record_field_is_closed(self):
        for bad in ({**self.BOUND, "extra": 1}, {**self.BOUND, "verdict_ref": "q3/verdict.json"},
                    {**self.BOUND, "verdict_ref": "$E/q3/campaign.json"}, {**self.BOUND, "artifact_sha256": None}, None):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "q3_record stock_record"):
                qwen38(BOTH, stock_record=bad)


if __name__ == "__main__":
    unittest.main()
