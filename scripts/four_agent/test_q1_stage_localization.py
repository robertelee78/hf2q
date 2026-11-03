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


if __name__ == "__main__":
    unittest.main()
