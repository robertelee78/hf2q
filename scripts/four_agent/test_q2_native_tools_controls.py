"""MOCK native profile, actual shared parser/policy controls; no tool/client process."""
import copy
import hashlib
import tempfile
from pathlib import Path
import json
import unittest
from q2_client_profile import phase_controls
from q2_native_tools import (SCHEMA, WORKLOAD, TOOLS, ordered_digest, validate_profile,
                             validate_request, native_assistant, extend_actual_history)
from q2_certify import replay_response
from qwen38_n1_verdict import Invalid, SSE, unary


def fixtures():
    tools = [{"type": "function", "function": {"name": n, "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}}, "additionalProperties": False}}}
        for n in sorted(TOOLS)]
    r = {"model": "Qwen3.6 35B A3B", "stream": True, "max_completion_tokens": 8192,
         "tools": tools, "messages": [{"role": "user", "content": "MOCK task"}]}
    p = {"schema": SCHEMA, "workload_id": WORKLOAD, "task_sha256": "a" * 64,
         "family": "qwen36", "model_sha256": "d" * 64, "model": r["model"], "client_version": "MOCK-2.1.1",
         "client_binary_sha256": "b" * 64, "profile_sha256": "c" * 64,
         "tools_order_sha256": ordered_digest(tools), "fixture_root": "/private/tmp/MOCK-owned-fixture",
         "shell_commands": ["MOCK locked ports command", "MOCK locked all command"]}
    profile = {"schema": "hf2q.q2.production-client-wire-profile.v1", "model": r["model"],
               "artifact_sha256": p["model_sha256"], "projection": {"family": "qwen36"},
               "client_version": p["client_version"], "client_binary": {"sha256": "b" * 64},
               "model_inference": False, "scripted_stub_responses": True, "campaign_primary_eligible": False,
               "actual_tools": tools, "phase_controls": [phase_controls(r) for _ in range(7)],
               "raw_request_refs": [{} for _ in range(7)]}
    return p, profile, r


def call(name="read", args=None, cid="MOCK-call"):
    return {"role": "assistant", "reasoning_content": "MOCK real-wire reasoning",
            "tool_calls": [{"id": cid, "type": "function", "function": {
                "name": name, "arguments": json.dumps(args or {"path": "/private/tmp/MOCK-owned-fixture/src/ports.rs"})}}]}


def usage():
    return {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101,
            "prompt_tokens_details": {"cached_tokens": 0}}


def unary_wire(message):
    return json.dumps({"id": "MOCK-response", "model": "Qwen3.6 35B A3B",
                       "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
                       "usage": usage()}).encode()


def chunk(delta, finish=None, stats=None):
    d = {"id": "MOCK-response", "model": "Qwen3.6 35B A3B",
         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if stats is not None: d["usage"] = stats
    return json.dumps(d)


class NativeToolsControls(unittest.TestCase):
    def test_order_digest_distinguishes_prompt_order_from_semantics(self):
        a = {"name": "x", "parameters": {"z": 1, "a": 2}}
        b = {"parameters": {"a": 2, "z": 1}, "name": "x"}
        self.assertEqual(a, b); self.assertNotEqual(ordered_digest(a), ordered_digest(b))
    def test_exact_native_catalog_and_profile_version(self):
        p, profile, r = fixtures(); self.assertEqual(validate_profile(profile, p), r["tools"])
        for key, value in [("client_version", "other"), ("model", "other"), ("model_inference", True), ("campaign_primary_eligible", True)]:
            bad = copy.deepcopy(profile); bad[key] = value
            with self.assertRaises(Invalid): validate_profile(bad, p)
    def test_schema_order_changed_with_same_semantic_tools_refuses(self):
        p, profile, r = fixtures(); r["tools"][0] = dict(reversed(list(r["tools"][0].items())))
        with self.assertRaises(Invalid): validate_request(r, profile, p, 0)
    def test_omission_is_not_null_zero_or_invented_seed(self):
        p, profile, r = fixtures()
        for value in (None, 0, False, 0.7):
            bad = copy.deepcopy(r); bad["temperature"] = value
            with self.assertRaises(Invalid): validate_request(bad, profile, p, 0)
    def test_native_primary_all_sse_no_manufactured_unary(self):
        p, profile, r = fixtures(); self.assertEqual(validate_request(r, profile, p, 0), r)
        r["stream"] = False
        with self.assertRaises(Invalid): validate_request(r, profile, p, 0)
    def test_native_callback_reuses_unary_codec_but_default_remains_three_tools(self):
        p, _, _ = fixtures(); raw = unary_wire(call())
        with self.assertRaises(Invalid): unary(raw, p["model"], 0, set())
        m, c, u = unary(raw, p["model"], 0, set(), assistant_policy=native_assistant(p))
        self.assertEqual(c[1], "read"); self.assertEqual(u["prompt_tokens"], 100)
        legacy = call("read_file", {"path": "src/ports.rs"})
        self.assertEqual(unary(unary_wire(legacy), p["model"], 0, set())[1][1], "read_file")
    def test_actual_shared_sse_feed_semantic_and_native_final_validation(self):
        p, _, _ = fixtures(); parser = SSE(p["model"], 0, set(), assistant_policy=native_assistant(p))
        self.assertFalse(parser.feed(chunk({"role": "assistant"})))
        self.assertTrue(parser.feed(chunk({"tool_calls": [{"index": 0, "id": "MOCK-call", "function": {"name": "read"}}]})))
        self.assertTrue(parser.feed(chunk({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps({"path": p["fixture_root"] + "/src/ports.rs"})}}]})))
        parser.feed(chunk({}, "tool_calls", usage())); parser.feed("[DONE]")
        self.assertEqual(parser.result()[1][1], "read")
    def test_name_only_timestamp_never_certifies_missing_arguments(self):
        p, _, _ = fixtures(); parser = SSE(p["model"], 0, set(), assistant_policy=native_assistant(p))
        self.assertTrue(parser.feed(chunk({"tool_calls": [{"index": 0, "id": "MOCK-call", "function": {"name": "read"}}]})))
        parser.feed(chunk({}, "tool_calls", usage())); parser.feed("[DONE]")
        with self.assertRaises((Invalid, json.JSONDecodeError)): parser.result()
    def test_reused_framing_consumer_missing_done_and_duplicate_identity_refuse(self):
        p, _, r = fixtures(); delta = copy.deepcopy(call()["tool_calls"][0]); delta["index"] = 0
        frames = [chunk({"tool_calls": [delta]}), chunk({}, "tool_calls", usage()), "[DONE]"]
        raw = ("\n\n".join("data: " + x for x in frames) + "\n\n").encode()
        self.assertEqual(replay_response(raw, r, 0, set(), "hf2q", assistant_policy=native_assistant(p))[1][1], "read")
        with self.assertRaises(Invalid): replay_response(raw.replace(b"data: [DONE]\n\n", b""), r, 0, set(), "hf2q", assistant_policy=native_assistant(p))
        changed = json.loads(frames[1]); changed["id"] = "MOCK-other-response"
        bad = ("\n\n".join("data: " + x for x in [frames[0], json.dumps(changed), "[DONE]"]) + "\n\n").encode()
        with self.assertRaises(Invalid): replay_response(bad, r, 0, set(), "hf2q", assistant_policy=native_assistant(p))
    def test_native_call_policy_denies_external_tools_wrong_path_parallel_and_duplicate(self):
        p, _, _ = fixtures(); v = native_assistant(p)
        for message in [call("webfetch", {"url": "https://example.invalid"}), call(args={"path": "/etc/passwd"}), call(args={"path": p["fixture_root"] + "/../secret"})]:
            with self.assertRaises(Invalid): v(message, "tool_calls", 0, set())
        message = call(); message["tool_calls"] *= 2
        with self.assertRaises(Invalid): v(message, "tool_calls", 0, set())
        with self.assertRaises(Invalid): v(call(), "tool_calls", 0, {"MOCK-call"})
    def test_exact_shell_request_only_no_background_or_other_command(self):
        p, _, _ = fixtures(); v = native_assistant(p)
        args = {"command": p["shell_commands"][0], "workdir": p["fixture_root"]}
        self.assertEqual(v(call("shell", args), "tool_calls", 3, set())[1], "shell")
        for bad in [{**args, "background": True}, {**args, "command": "rm -rf /"}, {**args, "workdir": "/"}]:
            with self.assertRaises(Invalid): v(call("shell", bad), "tool_calls", 3, set())
    def test_actual_result_history_preserves_reasoning_ids_and_wrapper_without_attesting_execution(self):
        p, _, _ = fixtures(); m = call(); c = native_assistant(p)(m, "tool_calls", 0, set()); before = [{"role": "user", "content": "MOCK"}]
        result = {"role": "tool", "tool_call_id": c[0], "content": "MOCK native result wrapper"}
        after = before + [m, result]; self.assertEqual(extend_actual_history(before, m, c, after)[0], after)
        for bad in [before + [dict(m, reasoning_content="changed"), result], before + [m, dict(result, tool_call_id="wrong")]]:
            with self.assertRaises(Invalid): extend_actual_history(before, m, c, bad)
    def test_complete_mock_raw_chain_stays_execution_unverified_and_primary_false(self):
        from f2_semantic import semantic_timing_v3
        from q2_native_audit import audit_native_agent
        with tempfile.TemporaryDirectory(prefix="q2-native-MOCK-") as directory:
            root = Path(directory).resolve()
            def put(name, value, raw=False):
                data = value if raw else json.dumps(value, ensure_ascii=False).encode()
                path = root / name; path.write_bytes(data)
                return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
            p, profile, request = fixtures()
            p["fixture_root"] = str(root / "MOCK-fixture")
            p["task_sha256"] = hashlib.sha256(b"MOCK task").hexdigest()
            profile["client_binary"] = put("MOCK-client.bin", b"not an actual client", raw=True)
            p["client_binary_sha256"] = profile["client_binary"]["sha256"]
            messages = request["messages"]
            turns = []
            for phase in range(7):
                request = dict(request, messages=copy.deepcopy(messages))
                ref = put(f"request-{phase}.json", request)
                if phase < 6:
                    name = ("read", "read", "write", "shell", "read", "shell")[phase]
                    args = {"path": p["fixture_root"] + ("/tests/ports.rs" if phase == 1 else "/src/ports.rs")}
                    if name == "write": args["content"] = "MOCK patch"
                    if name == "shell": args = {"command": p["shell_commands"][0 if phase == 3 else 1], "workdir": p["fixture_root"]}
                    message = call(name, args, f"MOCK-call-{phase}")
                    delta = copy.deepcopy(message); delta["tool_calls"][0]["index"] = 0
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": "MOCK final", "reasoning_content": ""}
                    delta, finish = message, "stop"
                stats = usage(); stats["prompt_tokens_details"]["cached_tokens"] = 0 if phase == 0 else 50
                raw = ("data: " + chunk(delta) + "\n\ndata: " + chunk({}, finish, stats) + "\n\ndata: [DONE]\n\n").encode()
                # Shared SSE assembler preserves empty content and reasoning fields.
                message.setdefault("content", ""); message.setdefault("reasoning_content", "")
                response = put(f"response-{phase}.bin", raw, True)
                send = 1000000000 + phase * 10000000
                receipt = {"f2_clock_id": "CLOCK_MONOTONIC", "turn": phase, "stream": True,
                           "http_status": 200, "request_bytes": len(Path(ref["path"]).read_bytes()),
                           "request_sha256": ref["sha256"], "response_sha256": response["sha256"],
                           "response_bytes": len(raw), "usage": stats, "send_monotonic_ns": send,
                           "semantic_event_monotonic_ns": [send + 1000],
                           "validated_terminal_monotonic_ns": send + 2000,
                           "semantic_timing_v3": semantic_timing_v3(send, [send + 1000], send + 2000, True, True)}
                turns.append({"request": ref, "response": response,
                              "receipt": put(f"receipt-{phase}.json", receipt),
                              "assembled": put(f"assembled-{phase}.json", message)})
                messages = messages + [message]
                if phase < 6: messages += [{"role": "tool", "tool_call_id": f"MOCK-call-{phase}", "content": "MOCK unverified native result"}]
            profile["raw_request_refs"] = [t["request"] for t in turns]
            profile["phase_controls"] = [phase_controls(json.loads(Path(t["request"]["path"]).read_text())) for t in turns]
            profile_ref = put("profile.json", profile); p["profile_sha256"] = profile_ref["sha256"]
            manifest = {"schema": "hf2q.q2.native-agent-chain.v1", "profile": profile_ref,
                        "initial_request": turns[0]["request"], "turns": turns}
            manifest_path = Path(put("manifest.json", manifest)["path"])
            result = audit_native_agent(manifest_path, p)
            self.assertEqual(len(result["observations"]), 7)
            self.assertFalse(result["campaign_primary_eligible"])
            self.assertFalse(result["coding_quality_qualified"])
            self.assertFalse(result["native_tool_execution_qualified"])
            self.assertFalse(result["whole_cohort_trace_qualified"])
            # Changing raw evidence after its frozen reference must refuse.
            Path(turns[3]["response"]["path"]).write_bytes(b"MOCK corruption")
            with self.assertRaises(Invalid): audit_native_agent(manifest_path, p)
    def test_final_summary_cannot_contain_seventh_tool(self):
        p, _, _ = fixtures(); v = native_assistant(p)
        self.assertIsNone(v({"role": "assistant", "content": "MOCK complete"}, "stop", 6, set()))
        with self.assertRaises(Invalid): v(call(), "tool_calls", 6, set())

if __name__ == "__main__": unittest.main()
