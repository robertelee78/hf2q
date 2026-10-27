"""Pure adversarial controls invoke actual raw audit/client boundaries, no child."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from q2_certify import BoundReader, checked_cleanup, process_result, replay_response
from qwen38_n1_smoke import build_request
from qwen38_n1_verdict import Invalid, loads, wire
from test_qwen38_n1_verdict import response, request, chunk, call_message, stats


def encoded_sse(parts):
    return ("\n\n".join("data: " + p for p in parts) + "\n\n").encode()


class ActualRawControls(unittest.TestCase):
    def test_actual_unary_read_call(self):
        model = "Qwen3.8"
        _, call, usage = replay_response(json.dumps(response()).encode(),
                                        dict(model=model, stream=False), 0, set(), "hf2q")
        self.assertEqual(call[1:], ("read_file", {"path": "src/ports.rs"}))
        self.assertEqual(usage, stats())

    def test_wrong_model_and_tool_arguments(self):
        for kind in ("model", "arguments", "id"):
            raw = response()
            if kind == "model":
                raw["model"] = "other artifact"
            elif kind == "arguments":
                raw["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"path":"../secret"}'
            else:
                raw["choices"][0]["message"]["tool_calls"][0]["id"] = "duplicate"
            with self.subTest(kind=kind), self.assertRaises(Invalid):
                replay_response(json.dumps(raw).encode(), dict(model="Qwen3.8", stream=False),
                                0, {"duplicate"}, "hf2q")

    def test_name_only_is_semantic_but_missing_arguments_is_quality_failure(self):
        from qwen38_n1_verdict import SSE
        parser = SSE("Qwen3.8", 0, set())
        name = chunk({"tool_calls": [{"index": 0, "id": "id", "type": "function",
                                     "function": {"name": "read_file"}}]})
        self.assertTrue(parser.feed(name))  # Existing actual NAME policy preserved.
        parser.feed(chunk({}, "tool_calls", stats()))
        parser.feed("[DONE]")
        with self.assertRaises((Invalid, json.JSONDecodeError)):
            parser.result()

    def test_role_reasoning_only_do_not_count_semantic(self):
        from qwen38_n1_verdict import SSE
        parser = SSE("Qwen3.8", 6, set())
        self.assertFalse(parser.feed(chunk({"role": "assistant"})))
        self.assertFalse(parser.feed(chunk({"reasoning_content": "thinking"})))
        parser.feed(chunk({}, "stop", stats(6)))
        parser.feed("[DONE]")
        with self.assertRaises((Invalid, json.JSONDecodeError)):
            parser.result()

    def test_actual_sse_complete_and_truncated_frames(self):
        message = call_message()
        call = dict(message["tool_calls"][0], index=0)
        parts = [chunk({"role": "assistant"}), chunk({"tool_calls": [call]}),
                 chunk({}, "tool_calls", stats()), "[DONE]"]
        raw = encoded_sse(parts)
        self.assertEqual(replay_response(raw, dict(model="Qwen3.8", stream=True), 0,
                                         set(), "hf2q")[1][1], "read_file")
        for corrupt in (raw[:-1], encoded_sse(parts[:-1]), raw + b"data: {}\n\n"):
            with self.subTest(corrupt=corrupt[-20:]), self.assertRaises(Invalid):
                replay_response(corrupt, dict(model="Qwen3.8", stream=True), 0, set(), "hf2q")

    def test_actual_builder_changed_output_and_silent_truncation_rejected(self):
        value = request()
        for limit in (1, 8191, 8193):
            with self.subTest(limit=limit), self.assertRaises(Invalid):
                wire({**value, "max_completion_tokens": limit})
        value["messages"][0]["content"] = "x" * 262144
        with self.assertRaises(Invalid):
            wire(value)

    def test_legitimate_ds_cached_above_first_prompt_is_not_rejected(self):
        from qwen38_n1_verdict import usage
        actual = dict(prompt_tokens=846, completion_tokens=25, total_tokens=871,
                      prompt_tokens_details=dict(cached_tokens=814))
        self.assertEqual(usage(actual, 1)["prompt_tokens_details"]["cached_tokens"], 814)

    def test_raw_hash_and_symlink_corruption_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp).resolve() / "raw.json"
            p.write_text('{"value":1}')
            ref = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(BoundReader().obj(ref), {"value": 1})
            p.write_text('{"value":2}')
            with self.assertRaises(Invalid):
                BoundReader().obj(ref)
            link = p.with_name("alias"); link.symlink_to(p)
            with self.assertRaises(Invalid):
                BoundReader().obj(dict(path=str(link), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))

    def test_actual_cleanup_requires_reaped_empty_observation(self):
        good = dict(verified=True, reaped=True, errors=[],
                    final_observation=dict(available=True, live_owned_pids=[]))
        checked_cleanup(good)
        for key, value in (("verified", False), ("reaped", False), ("errors", ["ps failure"]),
                           ("final_observation", dict(available=False, live_owned_pids=None)),
                           ("final_observation", dict(available=True, live_owned_pids=[123]))):
            with self.subTest(key=key), self.assertRaises(Invalid):
                checked_cleanup({**good, key: value})

    def test_exit_zero_forged_pass_text_and_zero_tests_fail_actual_process_audit(self):
        # MOCK files test the raw validator only, never claim a compiler executed.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            def bound(name, content):
                p = root / name; p.write_bytes(content)
                return dict(path=str(p), sha256=hashlib.sha256(content).hexdigest())
            binary = bound("mock-not-executable", b"MOCK")
            receipt = dict(exit_code=0, cleanup=True, argv=[str(root / "mock-not-executable"),
                           "--test-threads=1"], cleanup_receipt=dict(verified=True, reaped=True,
                           errors=[], final_observation=dict(available=True, live_owned_pids=[])))
            refs = dict(receipt=bound("receipt.json", json.dumps(receipt).encode()),
                        stderr=bound("stderr.bin", b""), binary=binary)
            for output in (b"passed=true", b"test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"):
                refs["stdout"] = bound("stdout.bin", output)
                with self.assertRaises(Invalid):
                    process_result(BoundReader(), refs, binary["sha256"])

    def test_actual_shared_clock_validator_rejects_absent_or_mixed_domains(self):
        from q2_certify import require_clock
        require_clock({"f2_clock_id": "CLOCK_MONOTONIC"})
        for value in ({}, {"f2_clock_id": "mach_absolute_time"}, {"f2_clock_id": "CLOCK_REALTIME"},
                      {"f2_clock_id": None}, {"f2_clock_id": True}):
            with self.subTest(value=value), self.assertRaises(Invalid): require_clock(value)


if __name__ == "__main__":
    unittest.main()
