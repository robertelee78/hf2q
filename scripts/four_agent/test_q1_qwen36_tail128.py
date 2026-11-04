import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import q1_qwen36_tail128 as tail128

HERE = Path(__file__).resolve().parent


def suffix144(vocab=248320):
    lanes = [[(lane * 1000 + i) % vocab for i in range(144)] for lane in range(4)]
    return (json.dumps({"family": "Qwen36", "lanes": lanes, "vocab_size": vocab}, indent=2) + "\n").encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Tail128Tests(unittest.TestCase):
    def setUp(self):
        self.source = suffix144()
        self.sha = sha(self.source)

    def test_derive_takes_lane_window_16_to_144_in_prefixes_form(self):
        out = json.loads(tail128.derive(self.source, self.sha))
        lanes = json.loads(self.source)["lanes"]
        self.assertEqual(list(out), ["family", "vocab_size", "prompts"])
        self.assertEqual(out["family"], "Qwen36")
        self.assertEqual(out["vocab_size"], 248320)
        self.assertEqual(len(out["prompts"]), 4)
        for lane in range(4):
            self.assertEqual(out["prompts"][lane], lanes[lane][16:144])
            self.assertEqual(len(out["prompts"][lane]), 128)
        self.assertEqual(tail128.derive(self.source, self.sha), tail128.derive(self.source, self.sha))

    def test_default_binding_is_the_accepted_suffix144_sha(self):
        self.assertEqual(tail128.INPUT_SHA256, "db413896a984f8a7559df70739738c85a8dcc059fecf307c00e3051846ae9880")
        with self.assertRaises(SystemExit):
            tail128.derive(self.source)

    def test_derive_refuses_malformed_workloads(self):
        def bad(mutate):
            work = json.loads(self.source)
            mutate(work)
            data = json.dumps(work).encode()
            with self.assertRaises(SystemExit):
                tail128.derive(data, sha(data))
        bad(lambda w: w.update(family="Qwen38"))
        bad(lambda w: w.update(extra=1))
        bad(lambda w: w["lanes"].pop())
        bad(lambda w: w["lanes"][2].pop())
        bad(lambda w: w["lanes"][1].__setitem__(20, 248320))
        bad(lambda w: w["lanes"].__setitem__(3, list(w["lanes"][0])))
        bad(lambda w: w.update(vocab_size=0))

    def test_record_binds_input_and_output_sha_and_check_matches(self):
        tail = tail128.derive(self.source, self.sha)
        rec = tail128.record(Path("/x/workload.json"), self.source, tail, self.sha)
        r = json.loads(rec)
        self.assertEqual(r["schema"], tail128.SCHEMA)
        self.assertEqual(r["input"], {"path": "/x/workload.json", "sha256": self.sha, "bytes": len(self.source)})
        self.assertEqual(r["output"], {"sha256": sha(tail), "bytes": len(tail)})
        self.assertEqual(r["window"], {"start": 16, "end": 144, "tokens_per_lane": 128})
        self.assertFalse(r["lanes_reselected"])
        result = json.loads(tail128.check(Path("/x/workload.json"), self.source, tail, rec, self.sha))
        self.assertEqual(result["verdict"], "Match")
        self.assertEqual(result["output_sha256"], sha(tail))

    def test_check_refuses_any_byte_difference(self):
        tail = tail128.derive(self.source, self.sha)
        rec = tail128.record(Path("/x/workload.json"), self.source, tail, self.sha)
        with self.assertRaises(SystemExit):
            tail128.check(Path("/x/workload.json"), self.source, tail + b" ", rec, self.sha)
        reselected = json.loads(tail)
        reselected["prompts"][0] = json.loads(self.source)["lanes"][0][:128]
        with self.assertRaises(SystemExit):
            tail128.check(Path("/x/workload.json"), self.source, tail128.encode(reselected), rec, self.sha)
        with self.assertRaises(SystemExit):
            tail128.check(Path("/x/workload.json"), self.source, tail, rec.replace(b"false", b"true", 1), self.sha)
        with self.assertRaises(SystemExit):
            tail128.check(Path("/x/other.json"), self.source, tail, rec, self.sha)
        with self.assertRaises(SystemExit):
            tail128.record(Path("/x/workload.json"), self.source, tail[:-1], self.sha)

    def test_cli_refuses_synthetic_input_not_bound_to_accepted_sha(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "workload.json"
            path.write_bytes(self.source)
            run = subprocess.run([sys.executable, "-B", str(HERE / "q1_qwen36_tail128.py"), "derive", str(path)],
                                 capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn(b"accepted Suffix144", run.stderr)


if __name__ == "__main__":
    unittest.main()
