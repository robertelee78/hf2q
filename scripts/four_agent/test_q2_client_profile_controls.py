"""Actual-used profile projection/omission validation; no client or socket launch."""
import copy
import json
import unittest
from q2_client_profile import (FAMILIES, SAMPLING, capture_profile, phase_controls,
                               projected_config, validate_capture_request)


def config():
    return {"model": "hf2q/original", "providers": {"hf2q": {
        "name": "Local", "package": "aisdk:@ai-sdk/openai-compatible",
        "settings": {"baseURL": "http://127.0.0.1:8081/v1", "apiKey": "SECRET_MUST_NOT_PERSIST"},
        "models": {"original": {"modelID": "original", "name": "Original",
            "capabilities": {"tools": True, "input": ["text", "image"], "output": ["text"]},
            "compatibility": {"reasoningField": "reasoning_content"},
            "limit": {"context": 262144, "output": 8192}, "options": {"temperature": .7}}}}}}


def request():
    tools = [{"type": "function", "function": {"name": name, "strict": False,
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
             for name in ("read", *[f"other{i}" for i in range(11)])]
    return {"model": "derived", "stream": True, "messages": [{"role": "user", "content": "owned"}], "tools": tools}


class ActualProfileControls(unittest.TestCase):
    def test_actual_projection_replaces_credentials_and_endpoint_only_with_explicit_model_substitution(self):
        original = config(); raw = json.dumps(original).encode()
        for family in FAMILIES:
            projected, receipt = projected_config(raw, family, "derived", "http://127.0.0.1:18000/v1")
            self.assertNotIn("SECRET_MUST_NOT_PERSIST", json.dumps(projected))
            self.assertNotIn("SECRET_MUST_NOT_PERSIST", json.dumps(receipt))
            model = projected["providers"]["hf2q"]["models"]["derived"]
            self.assertEqual(model["options"], {"temperature": .7})
            self.assertEqual(model["compatibility"], original["providers"]["hf2q"]["models"]["original"]["compatibility"])
            self.assertTrue(receipt["artifact_substitution_explicit"])
            self.assertFalse(receipt["configured_all_four_in_original"])
        self.assertEqual(original, config())

    def test_actual_projection_refuses_plugins_unknown_provider_or_unreviewed_settings(self):
        for defect in ("plugins", "provider", "setting", "context", "model_secret"):
            c = config()
            if defect == "plugins": c["plugins"] = ["forbidden"]
            elif defect == "provider": c["providers"]["external"] = {}
            elif defect == "setting": c["providers"]["hf2q"]["settings"]["headers"] = {"X": "secret"}
            elif defect == "context": c["providers"]["hf2q"]["models"]["original"]["limit"]["context"] = 8192
            else: c["providers"]["hf2q"]["models"]["original"]["options"]["apiKey"] = "never persist"
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                projected_config(json.dumps(c).encode(), "qwen38", "derived", "http://127.0.0.1:18000/v1")

    def test_absent_null_zero_false_and_explicit_sampling_remain_distinct(self):
        r = request(); base = phase_controls(r)
        self.assertEqual(base["present"], {}); self.assertEqual(base["absent"], list(SAMPLING))
        for value in (None, 0, False, .7):
            r["temperature"] = value; p = phase_controls(r)
            self.assertIn("temperature", p["present"]); self.assertNotIn("temperature", p["absent"])
            self.assertIs(type(p["present"]["temperature"]), type(value))
        self.assertEqual(phase_controls(request())["tools_sha256"], base["tools_sha256"])

    def test_actual_native_tools_and_real_owned_read_result_join(self):
        r = request(); validate_capture_request(r, "derived")
        r["messages"] += [{"role": "assistant", "tool_calls": [{"id": "q2_capture_0", "type": "function",
            "function": {"name": "read", "arguments": '{"path":"/owned/phase-0.txt"}'}}]},
            {"role": "tool", "tool_call_id": "q2_capture_0", "content": "Q2_OWNED_CAPTURE_0"}]
        validate_capture_request(r, "derived", "q2_capture_0", "/owned/phase-0.txt")
        for defect in ("wrong_id", "missing_result", "wrong_file", "three_tools"):
            bad = copy.deepcopy(r)
            if defect == "wrong_id": bad["messages"][-1]["tool_call_id"] = "other"
            elif defect == "missing_result": bad["messages"].pop()
            elif defect == "wrong_file": bad["messages"][-2]["tool_calls"][0]["function"]["arguments"] = '{"path":"/other"}'
            else: bad["tools"] = bad["tools"][:3]
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                validate_capture_request(bad, "derived", "q2_capture_0", "/owned/phase-0.txt")

    def test_whole_seven_phases_never_qualify_primary_or_coding_and_schema_drift_rejects(self):
        r = request(); records = [{"phase": i, "controls": phase_controls(r), "tools": r["tools"],
                   "request": {"path": f"MOCK-{i}", "sha256": "a"*64}} for i in range(7)]
        result = capture_profile(records, "derived")
        self.assertFalse(result["campaign_primary_eligible"]); self.assertFalse(result["coding_quality_qualified"])
        self.assertEqual(len(result["actual_tools"]), 12)
        with self.assertRaises(ValueError): capture_profile(records[:6], "derived")
        records[-1]["controls"]["tools_sha256"] = "b"*64
        with self.assertRaises(ValueError): capture_profile(records, "derived")


if __name__ == "__main__": unittest.main()
