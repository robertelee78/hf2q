"""Re-sealed adversarial raw artifacts call the actual whole-chain auditor.
MOCK fixture bytes cannot certify native execution, sampling or model quality.
"""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from f2_semantic import semantic_timing_v3
from q2_client_profile import phase_controls
from q2_native_audit import audit_native_agent
from qwen38_n1_verdict import Invalid
from test_q2_native_tools_controls import fixtures,call,usage,chunk


def raw_fixture(root):
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
    return p, manifest_path, manifest, turns, put


class NativeWholeAuditControls(unittest.TestCase):
    def test_full_native_chain_remains_protocol_only_despite_all_seven_valid_turns(self):
        with tempfile.TemporaryDirectory(prefix="q2-audit-MOCK-") as d:
            policy, path, _, _, _ = raw_fixture(Path(d).resolve())
            result = audit_native_agent(path, policy)
            self.assertEqual(len(result["observations"]), 7)
            for key in ("native_tool_execution_qualified", "coding_quality_qualified", "campaign_primary_eligible",
                        "whole_cohort_trace_qualified", "server_identity_qualified", "effective_sampling_qualified"):
                self.assertIs(result[key], False)
    def test_resealed_wrong_usage_status_timing_or_assembled_reasoning_refuse(self):
        # Re-hash deliberately changed files/manifest: refusal must come from
        # semantic joins, not merely the first stale file digest.
        for fault in ("usage", "http_status", "clock", "timing", "assembled"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix="q2-audit-MOCK-") as d:
                policy, path, manifest, turns, put = raw_fixture(Path(d).resolve())
                key = "assembled" if fault == "assembled" else "receipt"
                value = json.loads(Path(turns[2][key]["path"]).read_text())
                if fault == "usage": value["usage"]["completion_tokens"] += 1
                elif fault == "http_status": value["http_status"] = 500
                elif fault == "clock": value["f2_clock_id"] = "CLOCK_REALTIME"
                elif fault == "timing": value["semantic_timing_v3"]["semantic_ttft_seconds"] = 0
                else: value["reasoning_content"] = "MOCK altered reasoning"
                manifest["turns"][2][key] = put("corrupt-" + key + ".json", value)
                put(path.name, manifest)
                with self.assertRaises(Invalid): audit_native_agent(path, policy)
    def test_resealed_missing_turn_or_initial_request_substitution_refuses(self):
        for fault in ("missing", "initial", "duplicate"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(prefix="q2-audit-MOCK-") as d:
                policy, path, manifest, turns, put = raw_fixture(Path(d).resolve())
                if fault == "missing": manifest["turns"].pop()
                elif fault == "duplicate": manifest["turns"][4] = copy.deepcopy(turns[1])
                else: manifest["initial_request"] = turns[1]["request"]
                put(path.name, manifest)
                with self.assertRaises(Invalid): audit_native_agent(path, policy)
    def test_resealed_wrong_external_profile_or_task_identity_refuses(self):
        for key in ("profile_sha256", "task_sha256", "client_binary_sha256", "model_sha256"):
            with self.subTest(key=key), tempfile.TemporaryDirectory(prefix="q2-audit-MOCK-") as d:
                policy, path, _, _, _ = raw_fixture(Path(d).resolve()); policy[key] = "e" * 64
                with self.assertRaises(Invalid): audit_native_agent(path, policy)


if __name__ == "__main__": unittest.main()
