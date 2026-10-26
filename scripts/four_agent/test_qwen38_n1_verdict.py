"""Model-free handwritten controls for the same boundaries the actual N1 client uses."""
import json
import unittest

from qwen38_n1_verdict import (EXPECTED, Invalid, SSE, WORKFLOW, assistant, loads,
                              qualify, test_harness, unary, usage, wire)


def stats(turn=0):
    return {"prompt_tokens": 1000, "completion_tokens": 25, "total_tokens": 1025,
            "prompt_tokens_details": {"cached_tokens": 0 if turn == 0 else 900}}


def call_message(turn=0):
    name, args = EXPECTED[turn]
    args = args or {"path": "src/ports.rs", "content": "pub fn parse_port() {}"}
    return {"role": "assistant", "content": "", "reasoning_content": "Check first.",
            "tool_calls": [{"id": f"call_{turn}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def response(turn=0):
    return {"id": "response1", "model": "Qwen3.8", "choices": [
        {"index": 0, "message": call_message(turn), "finish_reason": "tool_calls"}],
        "usage": stats(turn)}


def request():
    return {"model": "Qwen3.8", "messages": [{"role": "user", "content": WORKFLOW}],
            "tools": [{"type": "function", "function": {"name": n, "strict": False}}
                      for n in ("read_file", "write_file", "run_tests")],
            "max_completion_tokens": 8192, "tool_choice": "auto", "logprobs": False,
            "hf2q_overflow_policy": "reject"}


def chunk(delta, finish=None, usage_value=None):
    item = {"id": "response1", "model": "Qwen3.8", "choices": [
        {"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage_value is not None:
        item["usage"] = usage_value
    return json.dumps(item)


def transcript():
    turns = [{"turn": i, "stream": i % 2 == 1} for i in range(7)]
    tools = [{"name": e[0], "cleanup": True, "passed": True, "source_sha256": "hash"}
             for e in EXPECTED]
    return turns, tools


class Protocol(unittest.TestCase):
    def test_valid_request_preserves_omissions(self):
        self.assertNotIn("seed", loads(wire(request())))

    def test_wire_guard_no_truncation(self):
        value = request()
        value["messages"][0]["content"] = "x" * 262144
        with self.assertRaises(Invalid):
            wire(value)

    def test_forced_tool_choice_rejected(self):
        value = request()
        value["tool_choice"] = "required"
        with self.assertRaises(Invalid):
            wire(value)

    def test_changed_sampling_rejected(self):
        for key in ("temperature", "seed", "top_p", "repetition_penalty"):
            with self.subTest(key=key), self.assertRaises(Invalid):
                wire({**request(), key: 0})

    def test_strict_false_is_not_assumed(self):
        value = request()
        value["tools"][0]["function"]["strict"] = True
        with self.assertRaises(Invalid):
            wire(value)

    def test_response_format_not_allowed(self):
        with self.assertRaises(Invalid):
            wire({**request(), "response_format": {"type": "json_object"}})

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(Invalid):
            loads('{"x":1,"x":2}')

    def test_nonfinite_json_rejected(self):
        with self.assertRaises(Invalid):
            loads('{"x":NaN}')

    def test_unary_real_call(self):
        _, call, _ = unary(json.dumps(response()), "Qwen3.8", 0, set())
        self.assertEqual(call[1:], ("read_file", {"path": "src/ports.rs"}))

    def test_wrong_model_rejected(self):
        with self.assertRaises(Invalid):
            unary(json.dumps(response()), "Other", 0, set())

    def test_context_plus_full_output_guard(self):
        value = stats()
        value["prompt_tokens"] = 262144 - 8191
        with self.assertRaises(Invalid):
            usage(value, 0)

    def test_missing_cache_field_not_zero(self):
        value = stats()
        del value["prompt_tokens_details"]
        with self.assertRaises(Invalid):
            usage(value, 0)

    def test_continuation_must_reuse_prefix(self):
        with self.assertRaises(Invalid):
            usage(stats(0), 1)

    def test_cold_cannot_claim_reuse(self):
        with self.assertRaises(Invalid):
            usage(stats(1), 0)

    def test_usage_bool_not_integer(self):
        value = stats()
        value["completion_tokens"] = True
        with self.assertRaises(Invalid):
            usage(value, 0)

    def test_output_allowance_not_exceeded(self):
        value = stats()
        value["completion_tokens"] = 8193
        with self.assertRaises(Invalid):
            usage(value, 0)

    def test_duplicate_tool_id(self):
        with self.assertRaises(Invalid):
            assistant(call_message(), "tool_calls", 0, {"call_0"})

    def test_missing_tool_call(self):
        with self.assertRaises(Invalid):
            assistant({"role": "assistant", "content": "Done"}, "stop", 0, set())

    def test_parallel_calls_not_six_step_packet(self):
        value = call_message()
        value["tool_calls"] *= 2
        with self.assertRaises(Invalid):
            assistant(value, "tool_calls", 0, set())

    def test_path_escape_rejected_at_actual_argument_boundary(self):
        value = call_message()
        value["tool_calls"][0]["function"]["arguments"] = '{"path":"../secret"}'
        with self.assertRaises(Invalid):
            assistant(value, "tool_calls", 0, set())

    def test_shell_target_rejected(self):
        value = call_message(3)
        value["tool_calls"][0]["function"]["arguments"] = '{"target":"ports; whoami"}'
        with self.assertRaises(Invalid):
            assistant(value, "tool_calls", 3, set())

    def test_all_target_valid(self):
        self.assertEqual(assistant(call_message(5), "tool_calls", 5, set())[2], {"target": "all"})

    def test_source_write_not_clipped(self):
        value = call_message(2)
        value["tool_calls"][0]["function"]["arguments"] = json.dumps(
            {"path": "src/ports.rs", "content": "x" * 65537})
        with self.assertRaises(Invalid):
            assistant(value, "tool_calls", 2, set())

    def test_final_summary_only(self):
        self.assertIsNone(assistant({"role": "assistant", "content": "Tests passed."}, "stop", 6, set()))
        with self.assertRaises(Invalid):
            assistant(call_message(), "tool_calls", 6, set())

    def test_length_finish_not_success(self):
        with self.assertRaises(Invalid):
            assistant(call_message(), "length", 0, set())

    def test_empty_role_not_semantic_ttft(self):
        self.assertFalse(SSE("Qwen3.8", 0, set()).feed(chunk({"role": "assistant"})))

    def test_reasoning_separate_from_semantic_ttft(self):
        self.assertFalse(SSE("Qwen3.8", 0, set()).feed(chunk({"reasoning_content": "Think"})))

    def test_sse_valid_split_tool_and_terminal(self):
        value = SSE("Qwen3.8", 0, set())
        self.assertTrue(value.feed(chunk({"tool_calls": [{"index": 0, "id": "live", "type": "function",
                                                         "function": {"name": "read_file", "arguments": '{"path":'}}]})))
        value.feed(chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"src/ports.rs"}'}}]}))
        value.feed(chunk({}, "tool_calls", stats()))
        value.feed("[DONE]")
        self.assertEqual(value.result()[1][0], "live")

    def test_sse_missing_done(self):
        with self.assertRaises(Invalid):
            SSE("Qwen3.8", 0, set()).result()

    def test_sse_done_before_usage(self):
        with self.assertRaises(Invalid):
            SSE("Qwen3.8", 0, set()).feed("[DONE]")

    def test_sse_changed_identity(self):
        value = SSE("Qwen3.8", 0, set())
        value.feed(chunk({"role": "assistant"}))
        with self.assertRaises(Invalid):
            value.feed(chunk({"content": "x"}).replace("response1", "response2"))

    def test_sse_usage_before_finish(self):
        with self.assertRaises(Invalid):
            SSE("Qwen3.8", 0, set()).feed(chunk({}, usage_value=stats()))

    def test_sse_post_terminal_rejected(self):
        value = SSE("Qwen3.8", 6, set())
        value.feed(chunk({"content": "Done"}, "stop", stats(6)))
        value.feed("[DONE]")
        with self.assertRaises(Invalid):
            value.feed(chunk({}))

    def test_sse_error_chunk_rejected(self):
        with self.assertRaises(Invalid):
            SSE("Qwen3.8", 0, set()).feed('{"error":{"message":"failed"}}')

    def test_pending_cannot_qualify(self):
        with self.assertRaises(Invalid):
            qualify([], [], True, "hash", "hash")

    def test_cleanup_and_stale_tests_not_success(self):
        turns, tools = transcript()
        for cleanup, final_hash in ((False, "hash"), (True, "changed")):
            with self.subTest(cleanup=cleanup), self.assertRaises(Invalid):
                qualify(turns, tools, cleanup, "hash", final_hash)

    def test_actual_tool_failure_not_summary_success(self):
        turns, tools = transcript()
        tools[-1]["passed"] = False
        with self.assertRaises(Invalid):
            qualify(turns, tools, True, "hash", "hash")

    def test_protocol_success_still_requires_independent_patch_review(self):
        turns, tools = transcript()
        self.assertEqual(qualify(turns, tools, True, "hash", "hash"),
                         "ProtocolCompleteAwaitingIndependentPatchReview")


class OwnedBoundary(unittest.TestCase):
    def fixture(self):
        import tempfile
        from pathlib import Path
        from qwen38_n1_tools import Tools
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tools = Tools(Path(temporary.name) / "owned", "/not/executed/bin/cargo",
                      "/not/executed/sandbox", "/Library/Developer/SDKs/Mock.sdk")
        self.addCleanup(tools.close)
        self.addCleanup(tools.cwd.rmdir)
        return tools

    def test_actual_owned_read_write_boundary(self):
        tools = self.fixture()
        self.assertIn("unwrap", tools.read("src/ports.rs"))
        tools.write("src/ports.rs", "replacement")
        self.assertEqual(tools.read("src/ports.rs"), "replacement")

    def test_actual_symlink_read_rejected(self):
        tools = self.fixture()
        path = tools.fixture / "src/ports.rs"
        path.unlink()
        path.symlink_to(tools.fixture / "tests/ports.rs")
        with self.assertRaises((Invalid, OSError)):
            tools.read("src/ports.rs")

    def test_actual_test_mutation_rejected(self):
        tools = self.fixture()
        path = tools.fixture / "tests/ports.rs"
        path.chmod(0o600)
        path.write_text("modified")
        with self.assertRaises(Invalid):
            tools.invariant()

    def test_actual_oversize_write_preserves_original(self):
        tools = self.fixture()
        original = tools.read("src/ports.rs")
        with self.assertRaises(Invalid):
            tools.write("src/ports.rs", "x" * 65537)
        self.assertEqual(tools.read("src/ports.rs"), original)

    def test_same_runtime_profile_denies_fork_and_source_write(self):
        from qwen38_n1_process import profile
        text = profile("runtime", "/tmp/fixture", "/tmp/target", "/tmp/tmp", "/tmp/log", "/tmp/test",
                       "/trusted/toolchain", "/Library/Developer/SDKs/Mock.sdk", "/private/tmp/cargo-cwd")
        self.assertIn("(deny process-fork)", text)
        writes = "\n".join(line for line in text.splitlines() if "allow file-write" in line)
        self.assertNotIn('(subpath "/tmp/fixture")', writes)
        self.assertNotIn('(subpath "/tmp/target")', writes)
        # This is a source grammar control ONLY, not sandbox confinement evidence.


class CleanupAndHarness(unittest.TestCase):
    def test_metrics_failure_still_terms_kills_and_reaps_owned_group(self):
        import itertools
        import signal
        from unittest.mock import Mock, patch, call
        import qwen38_n1_process as process
        child = Mock(pid=123)
        clock = itertools.count(0, 10)
        with patch.object(process, "process_snapshot", side_effect=Invalid("metrics denied")), \
             patch.object(process.os, "killpg") as kill, \
             patch.object(process.time, "monotonic", side_effect=lambda: next(clock)), \
             patch.object(process.time, "sleep"):
            result = process.stop_group(child, {})
        kill.assert_has_calls([call(123, signal.SIGTERM), call(123, signal.SIGKILL)])
        child.wait.assert_called_once_with(timeout=1)
        self.assertTrue(result["reaped"])
        self.assertFalse(result["verified"])
        self.assertTrue(result["errors"])

    def test_escaped_identity_checked_but_group_killed_on_tree_failure(self):
        import itertools
        import signal
        from unittest.mock import Mock, patch
        import qwen38_n1_process as process
        rows = {123: (1, 123, 10, "leader"), 124: (123, 999, 10, "known")}
        clock = itertools.count(0, 10)
        with patch.object(process, "process_snapshot", return_value=rows), \
             patch.object(process.os, "killpg") as group, \
             patch.object(process.os, "kill") as individual, \
             patch.object(process.time, "monotonic", side_effect=lambda: next(clock)), \
             patch.object(process.time, "sleep"):
            result = process.stop_group(Mock(pid=123), {124: "known"})
        group.assert_any_call(123, signal.SIGTERM)
        group.assert_any_call(123, signal.SIGKILL)
        individual.assert_any_call(124, signal.SIGTERM)
        self.assertFalse(result["verified"])

    def test_reused_escaped_pid_never_individually_killed(self):
        import itertools
        from unittest.mock import Mock, patch
        import qwen38_n1_process as process
        rows = {123: (1, 123, 10, "leader"), 124: (1, 999, 10, "different")}
        clock = itertools.count(0, 10)
        with patch.object(process, "process_snapshot", return_value=rows), \
             patch.object(process.os, "killpg") as group, \
             patch.object(process.os, "kill") as individual, \
             patch.object(process.time, "monotonic", side_effect=lambda: next(clock)), \
             patch.object(process.time, "sleep"):
            result = process.stop_group(Mock(pid=123), {124: "old"})
        self.assertEqual(group.call_count, 2)
        individual.assert_not_called()
        self.assertFalse(result["verified"])

    def test_clean_empty_group_reaped(self):
        from unittest.mock import Mock, patch
        import qwen38_n1_process as process
        child = Mock(pid=123)
        with patch.object(process, "process_snapshot", return_value={}):
            result = process.stop_group(child, {})
        self.assertTrue(result["verified"])
        child.wait.assert_called_once_with(timeout=1)

    def test_actual_positive_marker_count(self):
        raw = b"running 2 tests\ntest result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
        self.assertEqual(test_harness(raw), 2)

    def test_exit_zero_without_actual_tests_not_success(self):
        for raw in (b"", b"test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;\n",
                    b"test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out;\n"):
            with self.subTest(raw=raw), self.assertRaises(Invalid):
                test_harness(raw)

    def test_duplicate_result_marker_not_success(self):
        raw = b"test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;\n"
        with self.assertRaises(Invalid):
            test_harness(raw + raw)

    def test_sandbox_has_no_global_read_allow(self):
        from qwen38_n1_process import profile, PROOF_CASES
        text = profile("runtime", "/tmp/fixture", "/tmp/target", "/tmp/tmp", "/tmp/log", "/tmp/test",
                       "/trusted/toolchain", "/Library/Developer/SDKs/Mock.sdk", "/private/tmp/cargo-cwd")
        self.assertNotIn("(allow file-read*)", text)
        self.assertIn('(subpath "/trusted/toolchain")', text)
        self.assertIn("compile_outside_read_denied", PROOF_CASES)
        self.assertIn("runtime_outside_read_denied", PROOF_CASES)


class AbsoluteDeadline(unittest.TestCase):
    def test_actual_client_postread_deadline_even_without_connection_socket(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch
        import qwen38_n1_smoke as client
        now = [0.0]
        response_value = Mock(status=200)
        response_value.getheaders.return_value = [("Content-Type", "application/json")]
        response_value.getheader.return_value = "application/json"
        def late_read(_size):
            now[0] = 901.0
            return json.dumps(response()).encode()
        response_value.read1.side_effect = late_read
        connection = Mock(sock=None)
        connection.getresponse.return_value = response_value
        value = request()
        value["stream"] = False
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(client.http.client, "HTTPConnection", return_value=connection), \
             patch.object(client.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(client.signal, "signal"), \
             patch.object(client.signal, "setitimer", return_value=(0.0, 0.0)) as alarm:
            with self.assertRaises(Invalid):
                client.exchange("http://127.0.0.1:8081/v1", value, Path(temporary), 0, set())
            alarm.assert_any_call(client.signal.ITIMER_REAL, 900)
            connection.close.assert_called_once()

    def test_actual_alarm_raises_instead_of_extending_trickle_deadline(self):
        from qwen38_n1_smoke import request_timeout
        with self.assertRaises(TimeoutError):
            request_timeout(None, None)


if __name__ == "__main__":
    unittest.main()
