"""Prospective quota controls through the real strict artifact validator.

Small synthetic records exercise declared byte ceilings; never allocate GiBs.
No fixture is runtime/model evidence.
"""
import unittest

import f2_trace as trace
from test_f2_native_trace import fixture

PROFILE = "native_command_capture_v2"


def candidate(total=2 * 1024**3):
    records, identity, scenario = fixture()
    header = records[0]["payload"]
    header["metadata_budget_profile"] = PROFILE
    scenario["metadata_budget_profile"] = PROFILE
    header["server_budget_bytes"] = total - header["client_budget_bytes"]
    header["server_jsonl_budget_bytes"] = (header["server_budget_bytes"] -
        header["native_observer"]["metadata_limit_bytes"] - 1024**2)
    return records, identity, scenario


class MetadataBudgetControls(unittest.TestCase):
    def test_actual_validator_accepts_explicit_native_2g_and_legacy_256m(self):
        for args in (candidate(), fixture()):
            self.assertTrue(trace.validate_trace(*args)["qualified"])

    def test_actual_validator_refuses_legacy256_plus1_and_native2g_plus1(self):
        records, identity, scenario = fixture()
        records[0]["payload"]["server_budget_bytes"] += 1
        records[0]["payload"]["server_jsonl_budget_bytes"] += 1
        with self.assertRaises(ValueError):
            trace.validate_trace(records, identity, scenario)
        with self.assertRaises(ValueError):
            trace.validate_trace(*candidate(2 * 1024**3 + 1))

    def test_actual_validator_requires_closed_profile_native_presence_and_diagnostic(self):
        for value in ("other", "native_command_capture_v1", None, 1, True, {PROFILE: None}):
            records, identity, scenario = candidate()
            records[0]["payload"]["metadata_budget_profile"] = value
            scenario["metadata_budget_profile"] = value
            with self.subTest(profile=value), self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)
        for change in ("absent", "null", "not_diagnostic"):
            records, identity, scenario = candidate()
            header = records[0]["payload"]
            if change == "absent":
                del header["native_observer"]
                del header["server_jsonl_budget_bytes"]
            elif change == "null":
                header["native_observer"] = None
            else:
                header["native_observer"]["diagnostic_case"] = False
            with self.subTest(native=change), self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)

    def test_actual_validator_binds_external_profile_and_128m_client_minimum(self):
        records, identity, scenario = candidate()
        del scenario["metadata_budget_profile"]
        with self.assertRaises(ValueError):
            trace.validate_trace(records, identity, scenario)
        records, identity, scenario = candidate()
        records[0]["payload"]["client_budget_bytes"] = 128 * 1024**2 - 1
        with self.assertRaises(ValueError):
            trace.validate_trace(records, identity, scenario)

    def test_actual_native_event_and_metadata_limits_are_not_enlarged(self):
        for field, value in (("event_limit", 1_000_001),
                             ("metadata_limit_bytes", 16 * 1024**2 + 1)):
            records, identity, scenario = candidate()
            header = records[0]["payload"]
            header["native_observer"][field] = value
            header["server_jsonl_budget_bytes"] = (header["server_budget_bytes"] -
                header["native_observer"]["metadata_limit_bytes"] - 1024**2)
            with self.subTest(field=field), self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)

    def test_v3_exact_8g_and_16million_are_separate_from_legacy_and_v2(self):
        for profile, total, ceiling in ((None, 256 * 1024**2, 1_000_000),
                (PROFILE, 2 * 1024**3, 1_000_000),
                ("native_command_capture_v3", 8 * 1024**3, 16_000_000)):
            records, identity, scenario = candidate(total)
            header = records[0]["payload"]
            if profile is None:
                del header["metadata_budget_profile"]
                del scenario["metadata_budget_profile"]
            else:
                header["metadata_budget_profile"] = profile
                scenario["metadata_budget_profile"] = profile
            header["native_observer"]["event_limit"] = ceiling
            for record in records:
                if record["kind"] in ("native_baseline", "native_final"):
                    record["payload"]["event_limit"] = ceiling
                elif record["kind"] == "native_arm_ack":
                    record["payload"]["baseline"]["event_limit"] = ceiling
            self.assertTrue(trace.validate_trace(records, identity, scenario)["qualified"])
            header["native_observer"]["event_limit"] += 1
            with self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)
            header["native_observer"]["event_limit"] = ceiling
            header["server_budget_bytes"] += 1
            header["server_jsonl_budget_bytes"] += 1
            with self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)

    def test_v3_requires_external_profile_and_preserves_live_cap(self):
        for change in ("external", "diagnostic", "live", "event_bool", "event_overflow"):
            records, identity, scenario = candidate(8 * 1024**3)
            header = records[0]["payload"]
            header["metadata_budget_profile"] = "native_command_capture_v3"
            scenario["metadata_budget_profile"] = "native_command_capture_v3"
            if change == "external":
                scenario["metadata_budget_profile"] = PROFILE
            elif change == "diagnostic":
                header["native_observer"]["diagnostic_case"] = False
            elif change == "live":
                header["native_observer"]["metadata_limit_bytes"] = 16 * 1024**2 + 1
            elif change == "event_bool":
                header["native_observer"]["event_limit"] = True
            else:
                header["native_observer"]["event_limit"] = 2**64
            with self.subTest(change=change), self.assertRaises(ValueError):
                trace.validate_trace(records, identity, scenario)


if __name__ == "__main__":
    unittest.main()
