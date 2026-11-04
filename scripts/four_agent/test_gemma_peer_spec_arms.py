"""Controls for the Q4-GEMMA peer speculation arm cases (gemma_peer_spec_arms.py)."""
import json
from pathlib import Path
import tempfile
import unittest

import gemma_peer_spec_arms as arms
import q5_case

BASE = Path(__file__).resolve().parent / "q5_cases" / "peer-gemma4.json"
DRAFTER = {"path": "/tmp/drafter.gguf", "sha256": "a" * 64, "source": "test"}


def base():
    return q5_case.loads(BASE.read_text())


class Derive(unittest.TestCase):
    def test_only_the_spec_pair_changes(self):
        b = base()
        for arm in arms.ARMS:
            case = arms.derive(b, arm, DRAFTER if arm in arms.DRAFT_ARMS else None)
            flags = arms.ARMS[arm][1]
            expected = b["serve"]["argv"][:-2] + flags if flags else b["serve"]["argv"]
            self.assertEqual(case["serve"]["argv"], expected, arm)
            for key in ("readiness", "client", "artifact", "binary", "port", "family"):
                self.assertEqual(case[key], b[key], (arm, key))
            self.assertEqual(arms.EXPECTED_TYPE[arm], None if arm == "P0" else flags[1])

    def test_draft_arms_bind_path_and_hash(self):
        case = arms.derive(base(), "MT", DRAFTER)
        self.assertIn("{DRAFT_MODEL}", case["serve"]["argv"])
        self.assertEqual(case["variables"]["DRAFT_MODEL"], DRAFTER["path"])
        self.assertEqual(case["variables"]["DRAFT_SHA256"], DRAFTER["sha256"])
        values, missing = q5_case.resolve_variables(
            case, {"REPO": "/r", "OUT": "/o", "PYTHON": "p", "PYTHON_SHA256": "x", "RUN_ID": "i", "NONCE32": "n"},
            {"HF2Q_Q5_CARGO": "c", "HF2Q_Q5_TOOLCHAIN": "t", "HF2Q_Q5_SANDBOX_PROOF": "s",
             "HF2Q_Q5_SANDBOX_PROOF_SHA256": "h", "HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE": "a"})
        self.assertEqual(missing, [])
        self.assertIn(DRAFTER["path"], q5_case.expand(case["serve"]["argv"], values))

    def test_refusals(self):
        b = base()
        with self.assertRaises(arms.ArmError):
            arms.derive(b, "MT")
        with self.assertRaises(arms.ArmError):
            arms.derive(b, "NM", DRAFTER)
        with self.assertRaises(arms.ArmError):
            arms.derive(b, "XX")
        with self.assertRaises(arms.ArmError):
            arms.derive(b, "MT", {"path": "/d", "sha256": "short"})
        bad = base()
        bad["serve"]["argv"] = bad["serve"]["argv"][:-2]
        with self.assertRaises(arms.ArmError):
            arms.derive(bad, "NM")


class Cli(unittest.TestCase):
    def test_writes_cases_and_skips_missing_drafters(self):
        with tempfile.TemporaryDirectory() as tmp:
            drafters = Path(tmp) / "drafters.json"
            drafters.write_text(json.dumps({"E3": DRAFTER}))
            out = Path(tmp) / "cases"
            self.assertEqual(arms.main([str(BASE), str(out), "--drafters", str(drafters)]), 0)
            receipt = json.loads((out / "cases-receipt.json").read_text())
            self.assertEqual(sorted(receipt["skipped"]), ["DF", "DS", "MT"])
            self.assertIn("E3", receipt["cases"])
            self.assertTrue((out / "G-NM-ngram-mod.json").is_file())
            self.assertEqual(receipt["cases"]["E3"]["drafter_sha256"], DRAFTER["sha256"])
            # Idempotent; a changed existing file is refused.
            self.assertEqual(arms.main([str(BASE), str(out), "--drafters", str(drafters)]), 0)
            (out / "G-NM-ngram-mod.json").write_text("{}")
            with self.assertRaises(arms.ArmError):
                arms.main([str(BASE), str(out)])


if __name__ == "__main__":
    unittest.main()
