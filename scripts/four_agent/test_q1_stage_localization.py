"""Host-only controls for the Q1 stage localization record; synthetic stores only."""
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import q1_stage_localization as loc

STAGES = ("Key", "RouterLogits", "RouterIds", "LayerOutput")


def at(lane, layer, stage, row):
    return {"logical_step": 0, "layer": layer, "stage": stage, "agent": "agent-%d" % lane, "logical_row_start": row, "segment": 0}


def manifest(rows=(0, 1, 30, 128, 129)):
    """Per-lane execution order: prefill layer -> stage -> row, then decode position -> layer -> stage."""
    order = []
    for layer in (0, 1):
        for stage in STAGES:
            for lane in range(4):
                order += [at(lane, layer, stage, r) for r in rows if r < 128]
    for position in (r for r in rows if r >= 128):
        for layer in (0, 1):
            for stage in STAGES:
                order += [at(lane, layer, stage, position) for lane in range(4)]
    return order


def payload(a, perturb):
    lane, layer, stage, row = int(a["agent"][-1]), a["layer"], a["stage"], a["logical_row_start"]
    if stage == "RouterIds":
        ids = [3, 1] if (lane, layer, row) in perturb.get("ids", ()) else [1, 2]
        return "U32", struct.pack("<2I", *ids)
    if stage == "RouterLogits":
        return "F32", struct.pack("<4f", 0.1, 0.9, 0.85, 0.2)
    base = [float(lane * 100 + layer * 10 + row), 1.0]
    if (lane, layer, stage, row) in perturb.get("values", ()):
        base[1] += 0.25
    return "F32", struct.pack("<2f", *base)


def write_store(root, order, perturb):
    root.mkdir()
    captured = []
    for i, a in enumerate(order):
        scalar, raw = payload(a, perturb)
        name = "tensor-%06d.bin" % i
        (root / name).write_bytes(raw)
        captured.append({"at": a, "scalar": scalar, "shape": [1, len(raw) // 4], "file": name,
                         "sha256": hashlib.sha256(raw).hexdigest(), "logical_bytes": len(raw)})
    expected = [{"at": a, "scalar": c["scalar"], "shape": c["shape"]} for a, c in zip(order, captured)]
    trace = {"schema": "hf2q.adr063.invariance.v2", "coverage": {"captured": captured, "expected": expected, "errors": []}}
    (root / "trace.json").write_text(json.dumps(trace))
    return root / "trace.json"


class Localization(unittest.TestCase):
    def run_pair(self, perturb, workload=None, order=None):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); order = order or manifest()
            left = write_store(tmp / "all128", order, {})
            right = write_store(tmp / "packed", order, perturb)
            lanes = workload or [[2] * 29 + [496] * 99 + list(range(16)) for _ in range(4)]
            (tmp / "workload.json").write_text(json.dumps({"lanes": lanes}))
            return loc.localize(left, right, tmp / "workload.json")

    def test_identical_stores_have_no_divergence(self):
        record = self.run_pair({})
        self.assertEqual([l["first_divergence"] for l in record["lanes"]], [None] * 4)
        self.assertFalse(record["whole_q1_qualified"])
        self.assertEqual(record["lanes"][0]["repeated_prefix"],
                         dict(rows="29-127", distinct_tokens=1, modal_token=496, modal_count=99, longest_run=99))

    def test_first_divergence_is_per_lane_execution_order(self):
        # Lane 1 diverges first at layer-0 LayerOutput rows 1 and 30; a later layer-1 Key also differs.
        # Lane 2 diverges only in decode at position 129. Lanes 0 and 3 stay exact.
        record = self.run_pair({"values": {(1, 0, "LayerOutput", 1), (1, 0, "LayerOutput", 30), (1, 1, "Key", 0),
                                           (2, 1, "Key", 129)}})
        lane1 = record["lanes"][1]["first_divergence"]
        self.assertEqual((lane1["phase"], lane1["layer"], lane1["stage"], lane1["differing_rows"]), ("prefill", 0, "LayerOutput", [1, 30]))
        self.assertEqual(record["lanes"][1]["equal_groups_before_first"], 3)
        self.assertAlmostEqual(lane1["max_abs"], 0.25)
        self.assertEqual(lane1["prompt_segments"]["rows_0_28"]["differing_rows"], 1)
        self.assertEqual(lane1["prompt_segments"]["rows_29_127"]["differing_rows"], 1)
        lane2 = record["lanes"][2]["first_divergence"]
        self.assertEqual((lane2["phase"], lane2["decode_position"], lane2["layer"], lane2["stage"]), ("decode", 129, 1, "Key"))
        self.assertNotIn("prompt_segments", lane2)
        self.assertIsNone(record["lanes"][0]["first_divergence"])
        self.assertIsNone(record["lanes"][3]["first_divergence"])

    def test_router_id_changes_carry_the_raw_logit_gap(self):
        record = self.run_pair({"ids": {(3, 1, 30)}})
        first = record["lanes"][3]["first_divergence"]
        self.assertEqual((first["layer"], first["stage"], first["differing_rows"]), (1, "RouterIds", [30]))
        self.assertEqual(first["max_abs"], 0.0)  # integer ids carry no float magnitude
        change, = record["lanes"][3]["router_id_changes"]
        self.assertEqual((change["layer"], change["row"], change["set_changed"]), (1, 30, True))
        self.assertAlmostEqual(change["reference_raw_logit_gap"], 0.85 - 0.2, places=6)

    def test_refuses_unpaired_or_reordered_manifests(self):
        order = manifest()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            left = write_store(tmp / "a", order, {})
            right = write_store(tmp / "b", list(reversed(order)), {})
            (tmp / "w.json").write_text(json.dumps({"lanes": [[0] * 144] * 4}))
            with self.assertRaises(ValueError):
                loc.localize(left, right, tmp / "w.json")
            short = write_store(tmp / "c", order[:-1], {})
            with self.assertRaises(ValueError):
                loc.localize(left, short, tmp / "w.json")
            # A payload that no longer matches its recorded sha is refused.
            (tmp / "a" / "tensor-000000.bin").write_bytes(b"\0" * 8)
            with self.assertRaises(ValueError):
                loc.localize(left, write_store(tmp / "d", order, {}), tmp / "w.json")


def t5_receipt(arm, accepted=None, refusal=None):
    decisions = [{"lane": lane, "initial_argmax": 7, "frozen_128": 7, "initial_equals_frozen": True,
                  "decode_inputs": [], "decode_argmax": [],
                  "transaction": None if accepted is None else {
                      "proposer_ran": True, "proposed": [], "injected": [1, 2, 3],
                      "injected_origin": "InjectedDiagnosticFixtureNotProductionProposalQuality",
                      "verifier_argmax": [1, 2, 4, 5], "accepted": accepted[lane], "rejected": 3 - accepted[lane],
                      "fallback": 4, "rollback_cursor": 129 + accepted[lane]}} for lane in range(4)]
    return {"schema": loc.T5_SCHEMA, "packet_sha256": "0" * 64, "arm": arm,
            "status": "CompleteScopedAcquisition" if refusal is None else "FailedOrIncomplete", "error": None,
            "lanes": [0, 1, 2, 3], "decisions": decisions if refusal is None else decisions[:refusal["lane"]],
            "class_b_refusal": refusal, "stores": {}, "routing_witness": "routing-witness.jsonl",
            "whole_q1_qualified": False}


def plan(pass_, branch="N", accepted=None, layers=None):
    full = pass_ == 2
    return {"rerun_id": "r", "layers": layers or (list(range(30)) if pass_ == 1 else [3, 4]),
            "prefill_stages": list(loc.PREFILL_STAGES if full else loc.PASS1_STAGES),
            "decode_stages": [] if branch == "P" else list(loc.DECODE_STAGES if full else loc.PASS1_STAGES),
            "host_payload_limit": 1 << 30, "accepted": accepted}


REFUSAL = {"row": 127, "reason": "RouteSensitive: global-route initial decision differs from frozen at row 127", "lane": 1}


class KeyMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, pass_, accepted, branch="N", plan_accepted="same", refusal=None):
        ngram, ordinary, packet = self.root / "ngram.json", self.root / "ordinary.json", self.root / "plan.json"
        ngram.write_text(json.dumps(t5_receipt("NgramTransaction", accepted, refusal)))
        ordinary.write_text(json.dumps(t5_receipt("OrdinaryGlobal")))
        frozen = accepted if plan_accepted == "same" else plan_accepted
        packet.write_text(json.dumps({"branch": branch, "pass": pass_,
                                      "stage_localizer": plan(pass_, branch, frozen)}))
        return loc.ngram_map(ngram, ordinary, packet)

    def test_map_is_one_to_one_per_phase_and_complete_for_each_a(self):
        for a in (0, 1, 2):
            doc = self.make(2, [a] * 4)
            phases, unmapped = loc.validated_map(doc)
            P, D, L = loc.PREFILL_STAGES, loc.DECODE_STAGES, [3, 4]
            both = [s for s in P if s in D]
            self.assertEqual(len(both), 11)
            counts = {name: len(pairs) for name, pairs in phases.items()}
            self.assertEqual(counts, {"logits": 64, "verify-prefix": 4 * 2 * 14 * 128,
                                      "verify-committed": 4 * 2 * 11 * (a + 1), "continuation": 4 * 2 * 12 * (15 - a)})
            candidate = {loc.k(s, lane, r, l, 1) for lane in range(4) for l in L for s in P for r in range(132)}
            candidate |= {loc.k(s, lane, r, l) for lane in range(4) for l in L for s in D for r in range(129 + a, 144)}
            reference = {loc.k(s, lane, r, l) for lane in range(4) for l in L for s in P for r in range(128)}
            reference |= {loc.k(s, lane, r, l) for lane in range(4) for l in L for s in D for r in range(128, 144)}
            stage = [p for name, pairs in phases.items() if name != "logits" for p in pairs]
            self.assertEqual({c for c, _ in stage} | set(unmapped["candidate"]), candidate)
            self.assertEqual({r for _, r in stage} | set(unmapped["reference"]), reference)
            self.assertEqual(len(stage) + len(unmapped["candidate"]), len(candidate))
            self.assertEqual(len(stage) + len(unmapped["reference"]), len(reference))
            # Norm (decode-only) at decode rows 128..128+a is the only unmapped reference role.
            self.assertEqual(set(unmapped["reference"]),
                             {loc.k("Norm", lane, r, l) for lane in range(4) for l in L for r in range(128, 129 + a)})
            # Segment-1 rows 129+a..131 (all roles) and prefill-only roles at rows 128..128+a are unmapped candidates.
            expected = {loc.k(s, lane, r, l, 1) for lane in range(4) for l in L for s in P for r in range(129 + a, 132)}
            expected |= {loc.k(s, lane, r, l, 1) for lane in range(4) for l in L
                         for s in ("Residual", "DenseOutput", "MoeOutput") for r in range(128, 129 + a)}
            self.assertEqual(set(unmapped["candidate"]), expected)
            self.assertEqual((doc["accepted"], doc["pass"], doc["branch"]), ([a] * 4, 2, "N"))
            self.assertEqual(len(doc["receipts"]["ngram"]["sha256"]), 64)

    def test_pass1_map_and_routing_pairs(self):
        doc = self.make(1, [0, 1, 2, 1])
        phases, unmapped = loc.validated_map(doc)
        self.assertEqual(len(phases["verify-prefix"]), 4 * 30 * 3 * 128)
        self.assertEqual(unmapped["reference"], [])  # pass 1 has no decode-only role
        routing = {p["phase"]: p["pairs"] for p in doc["routing"]["phases"]}
        rights = [tuple(r) for pairs in routing.values() for _, r in pairs]
        self.assertEqual(len(rights), len(set(rights)))
        self.assertEqual(set(rights), {("Canonical", lane, p, l) for lane in range(4) for p in range(144) for l in range(30)})
        self.assertFalse(any(l[0] == "T5Initial" for pairs in routing.values() for l, _ in pairs))

    def test_duplicate_key_within_a_phase_refuses(self):
        doc = self.make(1, [1] * 4)
        phase = next(p for p in doc["phases"] if p["phase"] == "verify-committed")
        phase["pairs"].append(list(phase["pairs"][0]))
        with self.assertRaises(ValueError):
            loc.validated_map(doc)
        doc = self.make(1, [1] * 4)
        phase = next(p for p in doc["phases"] if p["phase"] == "continuation")
        phase["pairs"][1] = [phase["pairs"][1][0], phase["pairs"][0][1]]  # reference side repeated
        with self.assertRaises(ValueError):
            loc.validated_map(doc)

    def test_plan_and_receipt_binding_refusals(self):
        with self.assertRaises(ValueError):
            self.make(1, [1] * 4, plan_accepted=[2, 1, 1, 1])  # frozen accepted differs from the receipt
        with self.assertRaises(ValueError):
            self.make(1, [3] * 4)  # a outside {0,1,2}
        with self.assertRaises(ValueError):
            self.make(1, [1] * 4, branch="P")  # branch P needs accepted null and the row-127 refusal

    def test_branch_p_identity_map(self):
        doc = self.make(1, [1] * 4, branch="P", plan_accepted=None, refusal=REFUSAL)
        phases, unmapped = loc.validated_map(doc)
        self.assertEqual(sorted(phases), ["logits", "prefill"])
        self.assertTrue(all(c == r for pairs in phases.values() for c, r in pairs))
        self.assertEqual({c[2] for c, _ in phases["logits"]}, {127})
        self.assertEqual({c[2] for c, _ in phases["prefill"]}, set(range(128)))
        self.assertEqual(len(phases["prefill"]), 4 * 30 * 3 * 128)
        self.assertEqual((unmapped, doc["routing"], doc["accepted"]), ({"candidate": [], "reference": []}, None, None))


SMALL = {"rerun_id": "r", "layers": [2, 3], "prefill_stages": ["Attention", "Residual", "RouterLogits", "RouterIds",
                                                                "LayerOutput"],
         "decode_stages": ["Norm", "Attention", "RouterLogits", "RouterIds", "LayerOutput"],
         "host_payload_limit": 1 << 30, "accepted": [0, 1, 2, 1]}


def small_payload(key_, perturb):
    stage, lane, row, layer = key_[0], int(key_[1][-1]), key_[2], key_[3]
    if stage == "RouterIds":
        return "U32", struct.pack("<2I", *perturb.get(key_, (1, 2)))
    if stage == "RouterLogits":
        return "F32", struct.pack("<4f", 0.1, 0.9, 0.85, 0.2)
    if stage == "Logits":
        return "F32", struct.pack("<8f", *[float(lane + row), 3.0, 2.0, 1.0, 0.0, 0.5, perturb.get(key_, 0.25), 0.1])
    return "F32", struct.pack("<2f", float(lane * 1000 + layer * 200 + row), perturb.get(key_, 1.0))


def keyed_store(root, keys, perturb=(), drop=(), shape_of=()):
    """A trace.json store over `keys`; payload files are shared by content."""
    perturb, shape_of = dict(perturb), dict(shape_of)
    root.mkdir()
    captured = []
    for key_ in keys:
        scalar, raw = small_payload(key_, perturb)
        digest = hashlib.sha256(raw).hexdigest()
        (root / digest).write_bytes(raw)
        if key_ in drop:
            continue
        captured.append({"at": loc.at_of(key_), "scalar": scalar, "shape": shape_of.get(key_, [1, len(raw) // 4]),
                         "file": digest, "sha256": digest})
    expected = [{"at": c["at"], "scalar": c["scalar"], "shape": c["shape"]} for c in captured]
    trace = {"schema": "hf2q.adr063.invariance.v2", "coverage": {"captured": captured, "expected": expected, "errors": []}}
    (root / "trace.json").write_text(json.dumps(trace))
    return root / "trace.json"


class Mapped(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        receipts = {"ngram": {"path": "n", "sha256": "1" * 64}, "ordinary": {"path": "o", "sha256": "2" * 64}}
        self.doc = loc.build_map("N", 2, SMALL, receipts, SMALL["accepted"])
        self.map = self.root / "map.json"
        self.map.write_text(json.dumps(self.doc))
        phases, self.unmapped = loc.validated_map(self.doc)
        self.stage = [p for name, pairs in phases.items() if name != "logits" for p in pairs]
        self.logits = phases["logits"]
        (self.root / "workload.json").write_text(json.dumps({"lanes": [[2] * 29 + [496] * 99 + list(range(16))] * 4}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_mapped(self, candidate=(), reference=(), logits=(), drop=(), shape_of=(), tamper=None):
        n = len(list(self.root.iterdir()))
        stores = [keyed_store(self.root / f"cs{n}", [c for c, _ in self.stage] + self.unmapped["candidate"],
                              candidate, drop, shape_of),
                  keyed_store(self.root / f"rs{n}", [r for _, r in self.stage] + self.unmapped["reference"], reference),
                  keyed_store(self.root / f"cl{n}", [c for c, _ in self.logits], logits),
                  keyed_store(self.root / f"rl{n}", [r for _, r in self.logits])]
        binding = loc.pass_binding(self.map, *stores)
        if tamper:
            tamper(binding, stores)
        path = self.root / f"binding{n}.json"
        path.write_text(json.dumps(binding))
        return binding, path, loc.mapped(self.map, path, self.root / "workload.json")

    def test_identical_mapped_stores_have_no_divergence_and_embed_the_binding(self):
        binding, path, record = self.run_mapped()
        self.assertEqual(record["schema"], loc.MAPPED_SCHEMA)
        self.assertEqual([l["first_divergence"] for l in record["lanes"]], [None] * 4)
        self.assertEqual(record["pass_binding"], binding)
        self.assertEqual(record["pass_binding_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(record["map_sha256"], hashlib.sha256(self.map.read_bytes()).hexdigest())
        self.assertEqual([l["accepted"] for l in record["lanes"]], [0, 1, 2, 1])
        self.assertFalse(record["whole_q1_qualified"])
        self.assertEqual(record["route_class"], loc.ROUTE_CLASS)
        self.assertFalse(any(l["head_rule"]["fires"] for l in record["lanes"]))
        # Norm at decode rows 128..128+a is listed unmapped on the reference side.
        lane2 = [u for u in record["lanes"][2]["unmapped"] if u["side"] == "reference"]
        self.assertEqual({(u["key"][0], u["key"][2]) for u in lane2}, {("Norm", r) for r in (128, 129, 130)})

    def test_verify_prefix_layer3_attention_is_localized_there(self):
        _, _, record = self.run_mapped(candidate={loc.k("Attention", 1, 40, 3, 1): 1.5})
        first = record["lanes"][1]["first_divergence"]
        self.assertEqual((first["phase"], first["decode_position"], first["layer"], first["stage"], first["differing_rows"]),
                         ("verify-prefix", None, 3, "Attention", [40]))
        self.assertEqual(first["rows_in_group"], 128)
        self.assertAlmostEqual(first["max_abs"], 0.5)
        self.assertEqual(first["prompt_segments"]["rows_29_127"]["differing_rows"], 1)
        self.assertEqual(first["prompt_segments"]["rows_0_28"]["differing_rows"], 0)
        self.assertEqual([record["lanes"][l]["first_divergence"] for l in (0, 2, 3)], [None] * 3)

    def test_verify_committed_and_continuation_groups(self):
        _, _, record = self.run_mapped(candidate={loc.k("LayerOutput", 2, 130, 2, 1): 2.0,
                                                  loc.k("Attention", 0, 140, 3): 2.0})
        committed = record["lanes"][2]["first_divergence"]
        self.assertEqual((committed["phase"], committed["layer"], committed["stage"], committed["differing_rows"]),
                         ("verify-committed", 2, "LayerOutput", [130]))
        self.assertEqual(committed["prompt_segments"]["rows_128_130"]["rows"], 3)
        continuation = record["lanes"][0]["first_divergence"]
        self.assertEqual((continuation["phase"], continuation["decode_position"], continuation["layer"]),
                         ("continuation", 140, 3))

    def test_unmapped_difference_is_reported_not_localized(self):
        key_ = loc.k("Attention", 0, 131, 3, 1)  # a=0: segment-1 row 131 is unmapped
        _, _, record = self.run_mapped(candidate={key_: 9.0})
        self.assertEqual([l["first_divergence"] for l in record["lanes"]], [None] * 4)
        listed = next(u for u in record["lanes"][0]["unmapped"] if tuple(u["key"]) == key_)
        self.assertEqual(listed["sha256"], hashlib.sha256(small_payload(key_, {key_: 9.0})[1]).hexdigest())

    def test_missing_mapped_key_or_layout_mismatch_refuses(self):
        with self.assertRaises(ValueError):
            self.run_mapped(drop={self.stage[5][0]})
        with self.assertRaises(ValueError):
            self.run_mapped(shape_of={self.stage[5][0]: [2, 1]})

    def test_binding_sha_and_map_sha_refusals(self):
        def stale_trace(binding, stores):
            stores[0].write_text(stores[0].read_text() + " ")
        with self.assertRaises(ValueError):
            self.run_mapped(tamper=stale_trace)

        def other_map(binding, stores):
            binding["map_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.run_mapped(tamper=other_map)

    def test_head_rule_fires_only_without_an_earlier_stage_difference(self):
        logit = loc.k("Logits", 3, 135, loc.LOGIT_LAYER)
        _, _, record = self.run_mapped(logits={logit: 7.0})
        lane = record["lanes"][3]
        self.assertTrue(lane["head_rule"]["fires"])
        self.assertEqual((lane["first_divergence"]["stage"], lane["first_divergence"]["differing_rows"]), ("Head", [135]))
        # An order-only RouterIds change in verify-prefix layer 2 comes first: no head rule.
        _, _, record = self.run_mapped(logits={logit: 7.0}, candidate={loc.k("RouterIds", 3, 5, 2, 1): (2, 1)})
        lane = record["lanes"][3]
        self.assertFalse(lane["head_rule"]["fires"])
        self.assertEqual((lane["first_divergence"]["phase"], lane["first_divergence"]["stage"]),
                         ("verify-prefix", "RouterIds"))
        change, = lane["router_id_changes"]
        self.assertFalse(change["set_changed"])
        self.assertEqual(lane["logits"]["differing_rows"], [135])


if __name__ == "__main__":
    unittest.main()
