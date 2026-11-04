"""Controls for F2 'Q3 verdicts of record' records: quality-gate binding to the campaign's arm identities, receipt
re-derivation under --verify, and q3_report.rows(of_record=True) over a fake-runner campaign. Never model, GPU or
real receipt evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import f2_q3_tier as tier
import f2_scoreboard_render as r
import q3_report
import test_q3_campaign as base
from test_f2_q3_tier import EXP, board, tier_line

STOCK, PEER, NATIVE = "5" * 64, "9" * 64, "4" * 64
ARMS = {"stock": {"binary": {"sha256": STOCK}}, "experiment": {"binary": {"sha256": EXP}}, "peer": {"binary": {"sha256": PEER}}}


def write_run(run, binary, readiness=None):
    run.mkdir(parents=True, exist_ok=True)
    (run / "receipt.json").write_text(json.dumps({"binary": {"sha256": binary}}))
    if readiness is not None:
        (run / "client").mkdir(exist_ok=True)
        (run / "client" / "summary.json").write_text(json.dumps({"native_readiness": readiness}))


def quality_manifest(path, reference, subject_sha, reference_runs, status="Complete", q1="Pass", engine="hf2q"):
    """A quality_campaign.py manifest: one subject run on subject_sha and the given reference run directories."""
    subject = path.parent / "subject-run"
    write_run(subject, subject_sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "plan": {"reference_kind": reference, "arms": {"subject": {"family": "qwen38"}, "reference": {"engine_kind": engine}}},
        "status": status, "verdict": {"verdict": q1} if q1 else None,
        "runs": [{"arm": "subject", "attempts": [{"output": str(subject)}]},
                 {"arm": "reference", "attempts": [{"output": str(run)} for run in reference_runs]}]}))
    return json.loads(path.read_text())


class Gates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, name, reference, binaries, readiness=None, subject=EXP, **kw):
        runs = []
        for i, sha in enumerate(binaries):
            runs.append(self.root / name / f"ref-{i}")
            write_run(runs[-1], sha, None if readiness is None else readiness[i])
        return quality_manifest(self.root / name / "manifest.json", reference, subject, runs, **kw)

    def test_reference_runs_must_be_the_campaigns_arm(self):
        self.assertEqual(tier.gate_status(self.manifest("s", "stock-hf2q", [STOCK]), "stock-hf2q", "qwen38", ARMS),
                         ("Pass", "Pass", STOCK))
        self.assertEqual(tier.gate_status(self.manifest("p", "peer", [PEER], q1="BelowFloor"), "peer", "qwen38", ARMS),
                         ("Fail", "BelowFloor", PEER))
        # a "stock" reference that is the experiment binary (an env-toggle A/B) is not the v0.1.28 gate
        with self.assertRaisesRegex(ValueError, "reference runs are not on the campaign's v0.1.28 stock binary"):
            tier.gate_status(self.manifest("w", "stock-hf2q", [EXP]), "stock-hf2q", "qwen38", ARMS)
        with self.assertRaisesRegex(ValueError, "reference runs are not on the campaign's peer binary"):
            tier.gate_status(self.manifest("x", "peer", [STOCK]), "peer", "qwen38", ARMS)
        with self.assertRaisesRegex(ValueError, "the campaign has no peer arm"):
            tier.gate_status(self.manifest("y", "peer", [PEER]), "peer", "qwen38", {k: ARMS[k] for k in ("stock", "experiment")})
        with self.assertRaisesRegex(ValueError, "not run on the experiment binary"):
            tier.gate_status(self.manifest("z", "stock-hf2q", [STOCK], subject=STOCK), "stock-hf2q", "qwen38", ARMS)

    def test_native_reference_needs_one_readiness_identity_of_its_kind(self):
        ds4 = {"engine_kind": "ds4", "build_info": {"commit": "c97964b", "binary_sha256": NATIVE}}
        ok = self.manifest("n", "ds4-native", [NATIVE, NATIVE], readiness=[ds4, ds4], engine="ds4")
        self.assertEqual(tier.gate_status(ok, "ds4-native", "qwen38", ARMS), ("Pass", "Pass", NATIVE))
        other = {**ds4, "build_info": {"commit": "0000000", "binary_sha256": NATIVE}}
        mixed = self.manifest("m", "ds4-native", [NATIVE, NATIVE], readiness=[ds4, other], engine="ds4")
        with self.assertRaisesRegex(ValueError, "do not share one ds4 native readiness identity"):
            tier.gate_status(mixed, "ds4-native", "qwen38", ARMS)
        with self.assertRaisesRegex(ValueError, "reference arm is not engine kind ds4"):
            tier.gate_status(self.manifest("k", "ds4-native", [NATIVE], readiness=[ds4]), "ds4-native", "qwen38", ARMS)

    def test_unsettled_gates(self):
        self.assertEqual(tier.gate_status(None, "peer", "qwen38", ARMS), ("Pending", None, None))
        running = self.manifest("r", "peer", [PEER], status="Running", q1=None)
        self.assertEqual(tier.gate_status(running, "peer", "qwen38", ARMS), ("Pending", None, None))
        halted = self.manifest("h", "peer", [PEER], status="Halted", q1=None)
        self.assertEqual(tier.gate_status(halted, "peer", "qwen38", ARMS), ("Halted", None, None))

    def test_declared_peer_evidence_with_a_metal_oom_reads_as_out_of_memory(self):
        log = self.root / "peer" / "server.stderr"
        log.parent.mkdir()
        log.write_bytes(b"x" * 10 + b"kIOGPUCommandBufferCallbackErrorOutOfMemory\n")
        peer = {"reason": "declared from prior evidence", "evidence": [{"path": str(log), "sha256": "0" * 64}]}
        self.assertEqual(tier.unavailable(peer, self.root), "peer out of memory (prior evidence)")
        log.write_bytes(b"loaded\n")
        self.assertEqual(tier.unavailable(peer, self.root), "declared from prior evidence")


class Verify(unittest.TestCase):
    def test_record_rederives_from_receipts_and_refuses_gate_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "q3" / "qwen38").mkdir(parents=True)
            verdict = {"family": "qwen38", "verdict": "Win", "campaign_status": "Invalid", "reasons": [],
                       "peer": {"unavailable": {}}, "thermal_valid": True, "memory_valid": True,
                       "quality": {"patch_review": {"pending": []}},
                       "primary": {"stop": {"reason": "precision", "blocks": 6}, "secondary": {
                           "worst_slot_ttft_p95_s": {"vs_stock": {"improvement_percent": 30.0}},
                           "worst_slot_sse_gap_max_s": {"vs_stock": {"improvement_percent": None}}}}}
            campaign = {"identities": {"arms": ARMS}, "plan": {"declared": {"experiment": {"label": "main HEAD", "commit": "3848" * 10}}},
                        "halted": {"entry": "p3p1-047-w00-peer", "reason": f"width 3 < 4 in {root}/q3/qwen38/runs/x"},
                        "stages": [{"workload": "n4", "status": "Complete", "warmups": [], "blocks": []},
                                   {"workload": "3+1", "status": "Running", "warmups": [{"name": "p3p1-047-w00-peer"}], "blocks": []}]}
            (root / "q3/qwen38/verdict.json").write_text(json.dumps(verdict))
            (root / "q3/qwen38/campaign.json").write_text(json.dumps(campaign))
            write_run(root / "qg/peer-run", PEER)
            peer = quality_manifest(root / "qg/peer/manifest.json", "peer", EXP, [root / "qg/peer-run"])
            gates = [("stock-hf2q", "$E/qg/stock-hf2q/manifest.json", None), ("peer", "$E/qg/peer/manifest.json", peer)]
            refs = ("$E/q3/qwen38/verdict.json", "$E/q3/qwen38/campaign.json")
            rec = tier.build_record(verdict, campaign, root, refs, gates, None)
            self.assertEqual(rec["halt"], {"entry": "p3p1-047-w00-peer", "stage": "3+1", "reason": "width 3 < 4 in $E/q3/qwen38/runs/x"})
            self.assertEqual((rec["experiment"], rec["stages"], rec["unmeasurable"]),
                             ("hf2q 38483848", [["n4", "Complete"], ["3+1", "Running"]], ["worst_slot_sse_gap_max_s.vs_stock"]))
            self.assertEqual([(g["status"], g["q1_verdict"], g["reference_sha256"]) for g in rec["quality_gate"]],
                             [("Pending", None, None), ("Pass", "Pass", PEER)])
            m = board({"qwen38": rec}, [])
            self.assertEqual(r.verify_rows(m, root), [])
            write_run(root / "qg/stock-run", STOCK)
            quality_manifest(root / "qg/stock-hf2q/manifest.json", "stock-hf2q", EXP, [root / "qg/stock-run"], q1="BelowFloor")
            with self.assertRaisesRegex(ValueError, "quality_gate drift"):  # the gate settled after the merge
                r.verify_rows(m, root)


class RowsOfRecord(base.Harness):
    win = base.Driver.win

    def test_rows_of_record_validate_verify_and_render_the_tier_and_its_controls(self):
        driver, _ = self.driver(self.win, workloads="n4,n1")
        base.q3.write_verdict(self.out, driver.run(), [], None)
        stock = self.root / "qg" / "stock" / "manifest.json"
        write_run(self.root / "qg" / "stock-run", base.STOCK_SHA)
        quality_manifest(stock, "stock-hf2q", base.EXP_SHA, [self.root / "qg" / "stock-run"])
        review = self.root / "reviews" / "q3-qwen38.md"
        review.parent.mkdir()
        review.write_text("ACCEPT")
        gates = [("stock-hf2q", stock), ("peer", self.root / "qg" / "peer" / "manifest.json")]
        rows = q3_report.rows(self.out, self.root, True, gates, review)
        group = rows["groups"]["qwen38-q3-record"]
        rec, ref = group["q3_record"], "$E/evidence/q3-qwen38/verdict.json"
        self.assertEqual((rec["verdict"], rec["review"], rec["experiment"]), ("Win", "$E/reviews/q3-qwen38.md", "hf2q 22222222"))
        self.assertEqual([(g["reference"], g["status"]) for g in rec["quality_gate"]], [("stock-hf2q", "Pass"), ("peer", "Pending")])
        self.assertIn("vs experiment hf2q 22222222 vs the peer, Williams blocks.", group["workload"])
        verdict = json.loads((self.out / "verdict.json").read_text())
        fair = {(m, {"vs_stock": "v0128_stock", "vs_peer": "peer"}[n]) for m in tier.FAIRNESS for n in ("vs_stock", "vs_peer")
                if verdict["primary"]["secondary"][m][n]["improvement_percent"] is not None}
        self.assertLessEqual({(m, "v0128_stock") for m in tier.FAIRNESS}, fair)  # the stock fairness gate is decidable
        record_rows = [x for x in rows["rows"] if "q3_of_record" in x]
        self.assertEqual({(x["metric"], x.get("baseline_arm") or x["arm"]) for x in record_rows},
                         {("makespan_s", "v0128_stock"), ("makespan_s", "peer"), *fair,
                          *((m, a) for m in tier.TOKENS for a in ("candidate", "v0128_stock", "peer"))})
        self.assertTrue(all(x["variant"].startswith(("Q3 experiment hf2q 22222222", "Q3 stock", "Q3 peer")) for x in record_rows))
        # the record's completed n1 control: Descriptive makespan effects, controls of this verdict
        controls = [x for x in rows["rows"] if "q3_of_record" not in x]
        self.assertEqual({(x["id"], x["status"], x["control_of"], x["baseline_arm"]) for x in controls},
                         {("qwen38-q3-record-n1-makespan-vs-stock", "Descriptive", ref, "v0128_stock"),
                          ("qwen38-q3-record-n1-makespan-vs-peer", "Descriptive", ref, "peer")})
        document = {"schema": r.ROWS_SCHEMA, "evidence_root_symbol": "$E", "neutral_band_ratio": 1.025,
                    "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": base.MODEL_SHA, "note": ""}}, **rows}
        r.validate_manifest(document)
        self.assertEqual(r.verify_rows(document, self.root), [x["id"] for x in rows["rows"]])
        text = r.render(document, [], "0" * 64)
        self.assertEqual(tier_line(text, "Qwen3.8 27B")[1],  # the fake runs' final source still awaits patch review
                         "Withheld: quality gate Pending vs peer; patch review pending for 1 final source(s)")
        self.assertIn("`qwen38-q3-record-n1-makespan-vs-stock`", text[text.index("## Q3 controls of a primary verdict of record"):])
        # the same campaign's default fragment merged beside its record is refused
        default = q3_report.rows(self.out, self.root)
        both = {**document, "groups": {**document["groups"], **default["groups"]}, "rows": document["rows"] + default["rows"]}
        with self.assertRaisesRegex(ValueError, "merge its --of-record fragment"):
            r.validate_manifest(both)
        with self.assertRaisesRegex(ValueError, "unknown quality gate reference"):
            q3_report.rows(self.out, self.root, True, [("bogus", stock)])

    def test_cli_options_need_of_record(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            base.q3.parse_args(["--rows", str(self.out), "--evidence-root", str(self.root), "--quality-gate", "peer=x"])
        args = base.q3.parse_args(["--rows", str(self.out), "--evidence-root", str(self.root), "--of-record",
                                   "--quality-gate", "stock-hf2q=/q/m.json"])
        self.assertEqual((args.of_record, args.quality_gate), (True, ["stock-hf2q=/q/m.json"]))


def write_receipt(run, binary, artifact):
    run.mkdir(parents=True, exist_ok=True)
    (run / "receipt.json").write_text(json.dumps({"binary": {"sha256": binary}, "artifact": {"sha256": artifact}}))


class StockRecordGates(unittest.TestCase):
    """ADR-063 Q4-DS4-RACE decision D1: in a stock-record campaign the stock-hf2q gate is bound to the record's stock
    binary and hf2q artifact, and the peer gate to this campaign's peer binary and artifact (ds4's file)."""
    HF2Q_ART, DS4_ART = "a" * 64, "d" * 64
    BOUND = {"stock_binary_sha256": STOCK, "experiment_binary_sha256": EXP, "artifact_sha256": HF2Q_ART}
    CAMPAIGN = {"experiment": {"binary": {"sha256": EXP}}, "peer": {"binary": {"sha256": PEER}, "artifact": {"sha256": DS4_ART}}}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, name, reference, binary, artifact, subject=(EXP, None)):
        subject_run, reference_run = self.root / name / "subject-run", self.root / name / "ref-0"
        write_receipt(subject_run, subject[0], subject[1] or artifact)
        write_receipt(reference_run, binary, artifact)
        path = self.root / name / "manifest.json"
        path.write_text(json.dumps({
            "plan": {"reference_kind": reference, "arms": {"subject": {"family": "deepseek4"}, "reference": {"engine_kind": "hf2q"}}},
            "status": "Complete", "verdict": {"verdict": "Pass"},
            "runs": [{"arm": "subject", "attempts": [{"output": str(subject_run)}]},
                     {"arm": "reference", "attempts": [{"output": str(reference_run)}]}]}))
        return json.loads(path.read_text())

    def status(self, document, reference, bound=BOUND, arms=CAMPAIGN):
        return tier.gate_status(document, reference, "deepseek4", arms, bound)

    def test_the_stock_hf2q_gate_runs_the_records_stock_binary_on_hf2qs_artifact(self):
        self.assertEqual(self.status(self.manifest("s", "stock-hf2q", STOCK, self.HF2Q_ART), "stock-hf2q"),
                         ("Pass", "Pass", STOCK))
        with self.assertRaisesRegex(ValueError, "not run on the stock record's hf2q artifact"):
            self.status(self.manifest("s-ds4", "stock-hf2q", STOCK, self.DS4_ART), "stock-hf2q")
        with self.assertRaisesRegex(ValueError, "not run on the stock record's hf2q artifact"):  # subject on the other file
            self.status(self.manifest("s-mixed", "stock-hf2q", STOCK, self.HF2Q_ART, subject=(EXP, self.DS4_ART)), "stock-hf2q")
        with self.assertRaisesRegex(ValueError, "reference runs are not on the campaign's v0.1.28 stock binary"):
            self.status(self.manifest("s-other", "stock-hf2q", "6" * 64, self.HF2Q_ART), "stock-hf2q")
        with self.assertRaisesRegex(ValueError, "not run on the experiment binary"):
            self.status(self.manifest("s-subject", "stock-hf2q", STOCK, self.HF2Q_ART, subject=(STOCK, None)), "stock-hf2q")
        with self.assertRaisesRegex(ValueError, "the stock record's experiment binary is not this campaign's"):
            self.status(self.manifest("s-rebuilt", "stock-hf2q", STOCK, self.HF2Q_ART), "stock-hf2q",
                        bound={**self.BOUND, "experiment_binary_sha256": "7" * 64})
        # without a binding (a campaign with a stock arm) there is still no stock arm to bind here
        with self.assertRaisesRegex(ValueError, "the campaign has no stock arm"):
            self.status(self.manifest("s-unbound", "stock-hf2q", STOCK, self.HF2Q_ART), "stock-hf2q", bound=None)

    def test_the_peer_gate_runs_on_the_peer_arms_artifact(self):
        self.assertEqual(self.status(self.manifest("p", "peer", PEER, self.DS4_ART), "peer"), ("Pass", "Pass", PEER))
        with self.assertRaisesRegex(ValueError, "not run on the peer arm's artifact"):
            self.status(self.manifest("p-hf2q", "peer", PEER, self.HF2Q_ART), "peer")
        with self.assertRaisesRegex(ValueError, "reference runs are not on the campaign's peer binary"):
            self.status(self.manifest("p-other", "peer", STOCK, self.DS4_ART), "peer")


class StockRecordRows(base.StockRecordHarness):
    """The record of a stock-record campaign: --of-record rows, gates bound through build_record, --verify."""

    def test_rows_of_record_bind_the_gates_and_cite_the_record(self):
        record = self.record()
        driver, _ = self.campaign(record)
        base.q3.write_verdict(self.out, driver.run(), [], None)
        stock, peer = self.root / "qg" / "stock" / "manifest.json", self.root / "qg" / "peer" / "manifest.json"
        for path, binary, artifact in ((stock, base.STOCK_SHA, base.MODEL_SHA), (peer, base.PEER_SHA, base.DS4_ART)):
            write_receipt(path.parent / "ref-run", binary, artifact)
            write_receipt(path.parent / "subject-run", base.EXP_SHA, artifact)
            path.write_text(json.dumps({
                "plan": {"reference_kind": path.parent.name.replace("stock", "stock-hf2q"),
                         "arms": {"subject": {"family": "qwen38"}, "reference": {"engine_kind": "hf2q"}}},
                "status": "Complete", "verdict": {"verdict": "Pass"},
                "runs": [{"arm": "subject", "attempts": [{"output": str(path.parent / "subject-run")}]},
                         {"arm": "reference", "attempts": [{"output": str(path.parent / "ref-run")}]}]}))
        rows = q3_report.rows(self.out, self.root, True, [("stock-hf2q", stock), ("peer", peer)])
        rec = rows["groups"]["qwen38-q3-record"]["q3_record"]
        self.assertEqual([(g["reference"], g["status"], g["reference_sha256"]) for g in rec["quality_gate"]],
                         [("stock-hf2q", "Pass", base.STOCK_SHA), ("peer", "Pass", base.PEER_SHA)])
        self.assertEqual(rec["stock_record"]["verdict_ref"], "$E/evidence/q3-qwen38/verdict.json")
        stock_rows = [x for x in rows["rows"] if x.get("baseline_arm") == "v0128_stock"]
        self.assertTrue(stock_rows and all(x["verification"]["key"].startswith("verdict.json stock_record.")
                                           and x["artifact_sha256"] == base.MODEL_SHA for x in stock_rows))
        document = {"schema": r.ROWS_SCHEMA, "evidence_root_symbol": "$E", "neutral_band_ratio": 1.025,
                    "families": {"qwen38": {"title": "Qwen3.8 27B", "artifact_sha256": base.DS4_ART, "note": ""}}, **rows}
        r.validate_manifest(document)
        self.assertEqual(r.verify_rows(document, self.root), [x["id"] for x in rows["rows"]])
        # a stock gate run on ds4's file is refused when the record is built
        write_receipt(stock.parent / "ref-run", base.STOCK_SHA, base.DS4_ART)
        with self.assertRaisesRegex(ValueError, "not run on the stock record's hf2q artifact"):
            q3_report.rows(self.out, self.root, True, [("stock-hf2q", stock), ("peer", peer)])


if __name__ == "__main__":
    unittest.main()
