"""Actual capture-runner preflight boundary controls, no process/socket/client."""
import copy
import unittest
from q2_capture_packet import SOURCE_CLOSURE, validate_capture_approval, approval_from_packet
from qwen38_n1_verdict import Invalid


def positive():
    ref = {"path": "/private/tmp/MOCK-bound", "sha256": "a" * 64}
    return {"schema": "hf2q.q2.client-capture-approval.v1", "approved_capture_only": True,
            "source_files": {name: "b" * 64 for name in SOURCE_CLOSURE},
            "client_binary": ref, "source_config": dict(ref),
            "sandbox_binary": {"path": "/usr/bin/sandbox-exec", "sha256": "c" * 64},
            "expected_version": "2.1.1", "port": 18321,
            "profiles": {family: {"model_id": "MOCK-" + family, "artifact_sha256": "d" * 64,
                         "effective_hf_defaults": dict(ref)} for family in ("qwen38", "qwen36", "deepseek4", "gemma4")}}


class NativeCapturePacketControls(unittest.TestCase):
    def test_actual_preflight_positive_keeps_all_four_explicit_profiles(self):
        value = positive(); self.assertIs(validate_capture_approval(value), value)
        self.assertEqual(value["expected_version"], "2.1.1")
    def test_disabled_packet_refuses_before_any_capture(self):
        for packet in ({"schema": "hf2q.q2.native-capture-packet.v1", "review": None, "approval": None},
                       {"schema": "hf2q.q2.native-capture-packet.v1", "review": None, "approval": positive()}):
            with self.assertRaises(Invalid): approval_from_packet(packet)
    def test_other_version_approval_boolean_or_extra_claim_refuses(self):
        for key, value in (("expected_version", "2.1"), ("expected_version", "MOCK-2.1.1"),
                           ("approved_capture_only", 1), ("campaign_primary_eligible", True)):
            bad = positive(); bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(Invalid): validate_capture_approval(bad)
    def test_missing_transitive_source_or_hash_does_not_count_as_bound(self):
        for name in SOURCE_CLOSURE:
            bad = positive(); del bad["source_files"][name]
            with self.subTest(name=name), self.assertRaises(Invalid): validate_capture_approval(bad)
        bad = positive(); bad["source_files"]["../escape.py"] = "f" * 64
        with self.assertRaises(Invalid): validate_capture_approval(bad)
    def test_family_fallback_incomplete_identity_and_bool_port_refuse(self):
        variants = []
        bad = positive(); del bad["profiles"]["gemma4"]; variants.append(bad)
        bad = positive(); bad["profiles"]["qwen38"]["effective_hf_defaults"] = None; variants.append(bad)
        bad = positive(); bad["client_binary"]["sha256"] = None; variants.append(bad)
        bad = positive(); bad["port"] = True; variants.append(bad)
        bad = positive(); bad["sandbox_binary"]["path"] = "/private/tmp/other"; variants.append(bad)
        for bad in variants:
            with self.assertRaises(Invalid): validate_capture_approval(bad)


if __name__ == "__main__": unittest.main()
