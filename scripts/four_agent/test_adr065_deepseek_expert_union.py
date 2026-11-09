#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

import adr065_deepseek_expert_union as union


def layer(index, routing, unique, width=2, top_k=2, expert_bytes=100):
    naive = expert_bytes * width * top_k
    unique_bytes = expert_bytes * unique
    return {
        "layer": index,
        "routing": routing,
        "unique_experts": unique,
        "routed_expert_rows": width * top_k,
        "expert_bytes": expert_bytes,
        "routed_bytes_naive": naive,
        "routed_bytes_unique": unique_bytes,
        "reduction_share": union.reduction(naive, unique_bytes),
    }


def record(width=2):
    layers = [layer(0, "hash", 3, width), layer(1, "score", 2, width)]
    naive = sum(value["routed_bytes_naive"] for value in layers)
    unique = sum(value["routed_bytes_unique"] for value in layers)
    return {
        "schema": union.SCHEMA,
        "width": width,
        "top_k": 2,
        "layers": layers,
        "routed_bytes_naive": naive,
        "routed_bytes_unique": unique,
        "reduction_share": union.reduction(naive, unique),
    }


class ExpertUnion(unittest.TestCase):
    def test_load_and_summarize_separates_route_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.stderr"
            path.write_text("noise\n" + union.PREFIX + json.dumps(record()) + "\n")
            report = union.summarize(union.load([path]), [2])
        self.assertEqual(report["steps_by_width"], {"2": 1})
        self.assertEqual(report["routes"]["hash"]["routed_bytes_unique"], 300)
        self.assertEqual(report["routes"]["score"]["routed_bytes_unique"], 200)
        self.assertEqual(report["reduction_share"], 0.375)

    def test_unknown_field_is_rejected(self):
        value = record()
        value["extra"] = 1
        with self.assertRaisesRegex(ValueError, "record keys differ"):
            union.parse_record(value, "x", 1)

    def test_missing_required_width_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required widths: \\[3\\]"):
            union.summarize([record()], [2, 3])


if __name__ == "__main__":
    unittest.main()
