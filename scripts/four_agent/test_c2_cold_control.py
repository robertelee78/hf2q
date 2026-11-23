"""CPU client/replay regressions for the designed K1 cold edit; no model claims."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import c2_observation as observation
import c2_evidence
import coding_long_context
import q2_fake_service as fake
from q2_fake_c2 import context
from q2_run_audit import check_run
from q2_certify import replay_response
from qwen38_n1_verdict import Invalid

VARIANT = "divergent-edit-v1"


def responses(inline=False, stream=True, **changes):
    usage = {"prompt_tokens": 2048, "completion_tokens": 32, "total_tokens": 2080,
             "prompt_tokens_details": {"cached_tokens": 0}}
    usage.update(changes)
    call = {"id": "call-cold-control", "type": "function",
            "function": {"name": "run_tests", "arguments": '{"target":"ports"}'}}
    if not stream:
        return json.dumps({"id": "response-cold-control", "model": "model", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [call]},
             "finish_reason": "tool_calls"}], "usage": usage}).encode()
    def frame(choices, **extra):
        return ("data: " + json.dumps({"id": "response-cold-control", "model": "model",
                                       "choices": choices, **extra}) + "\n\n").encode()
    delta = {"index": 0, "delta": {"tool_calls": [{"index": 0, **call}]}, "finish_reason": None}
    finish = {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
    return (frame([delta]) + (frame([finish], usage=usage) if inline else
            frame([finish]) + frame([], usage=usage)) + b"data: [DONE]\n\n")


class FullResponseControls(unittest.TestCase):
    def test_cold_control_accepts_unary_and_both_complete_sse_usage_shapes(self):
        for stream, inline in ((False, False), (True, False), (True, True)):
            with self.subTest(stream=stream, inline=inline):
                request = {"model": "model", "stream": stream}
                raw = responses(inline=inline, stream=stream)
                message, call, usage = replay_response(raw, request, 3, set(), "hf2q", cache_miss_allowed=True)
                self.assertEqual(call[1:], ("run_tests", {"target": "ports"}))
                self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 0)
                self.assertEqual(message["tool_calls"][0]["id"], "call-cold-control")
                with self.assertRaisesRegex(Invalid, "prefix reuse not proven"):
                    replay_response(raw, request, 3, set(), "hf2q")

    def test_cold_permission_does_not_bypass_usage_or_terminal_validation(self):
        request = {"model": "model", "stream": True}
        for bad in (responses(total_tokens=1), responses(completion_tokens=9000),
                    responses(prompt_tokens_details={"cached_tokens": 2049}),
                    responses().replace(b"data: [DONE]\n\n", b"")):
            with self.subTest(raw=bad[-100:]), self.assertRaises(Invalid):
                replay_response(bad, request, 3, set(), "hf2q", cache_miss_allowed=True)


class StartupControls(unittest.TestCase):
    def test_observed_depth_cannot_be_relabelled_or_taken_from_another_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = context(root / "client", 1, fake.RUN, fake.SERVER_PID)
            self.assertEqual(observation.checked_cache_control(control, fake.SERVER_PID, fake.RUN)["anchor_depth"], 1)
            for depth in (4, True):
                bad = copy.deepcopy(control)
                bad["anchor_depth"] = depth
                with self.assertRaises(ValueError):
                    observation.checked_cache_control(bad, fake.SERVER_PID, fake.RUN)
            raw = Path(control["startup_log"]["path"]).read_bytes()
            for bad in (raw + raw, raw.replace(b"engine:", b"unrelated:"),
                        raw.replace(b"anchor_depth=1", b"anchor_depth=2"), b""):
                with self.assertRaises(ValueError):
                    observation.anchor_config(bad)

    def test_replaced_or_truncated_owned_log_cannot_authorize_cold_edit(self):
        for replacement in (False, True):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                control = context(Path(directory) / "client", 1, fake.RUN, fake.SERVER_PID)
                log = Path(control["server_log"])
                raw = log.read_bytes()
                if replacement:
                    log.unlink()
                    log.write_bytes(raw)
                else:
                    log.write_bytes(raw[:10])
                with log.open("rb") as stream, self.assertRaises(ValueError):
                    observation.check_startup_identity(control, stream)

    def test_malformed_competing_declaration_cannot_disappear_from_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            control = context(Path(directory) / "client", 1, fake.RUN, fake.SERVER_PID)
            raw = Path(control["startup_log"]["path"]).read_bytes()
            quota = f'anchor_bytes_per_slot_max=Some({control["anchor_bytes_per_slot_max"]})'.encode()
            malformed = raw.replace(b"anchor_depth=1", b"anchor_depth=4").replace(
                quota, b"anchor_bytes_per_slot_max=None")
            for bad in (raw + malformed, raw + raw.replace(b"anchor_depth=1", b"anchor_depth=2")):
                with self.subTest(raw=bad), self.assertRaises(ValueError):
                    observation.anchor_config(bad)

    def test_buffered_original_bytes_cannot_hide_in_place_log_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            control = context(Path(directory) / "client", 1, fake.RUN, fake.SERVER_PID)
            log = Path(control["server_log"])
            raw = log.read_bytes()
            with log.open("rb") as stream:
                stream.read(len(raw))
                stream.seek(0)
                observation.check_startup_identity(control, stream)
                log.write_bytes(raw.replace(b"anchor_depth=1", b"anchor_depth=4") + b"more log\n")
                with self.assertRaises(ValueError):
                    observation.check_startup_identity(control, stream)

    def test_configuration_after_startup_retention_bound_still_invalidates_arm(self):
        for malformed in (False, True):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                control = context(Path(directory) / "client", 1, fake.RUN, fake.SERVER_PID)
                log = Path(control["server_log"])
                other = log.read_bytes().replace(b"anchor_depth=1", b"anchor_depth=4")
                if malformed:
                    quota = f'anchor_bytes_per_slot_max=Some({control["anchor_bytes_per_slot_max"]})'.encode()
                    other = other.replace(quota, b"anchor_bytes_per_slot_max=None")
                with log.open("ab") as stream:
                    stream.write(b"ordinary log\n" * observation.STARTUP_LIMIT + other)
                with log.open("rb") as stream, self.assertRaises(ValueError):
                    observation.check_startup_identity(control, stream)
                with self.assertRaises(ValueError):
                    c2_evidence.configured_depth(log.parent)

    def test_fifo_evidence_is_rejected_without_waiting_for_a_writer(self):
        script = """
import json, sys
from pathlib import Path
import c2_observation as o
c, case = json.loads(sys.argv[1]), sys.argv[2]
try:
    if case == "startup":
        o.checked_cache_control(c, c["server_pid"], c["run_id"])
    elif case == "cache":
        o.cache_control(c["server_log"], Path(c["startup_log"]["path"]).parent,
                        c["server_pid"], c["run_id"])
    elif case == "identity":
        with Path(c["server_log"]).open("rb") as stream:
            o.check_startup_identity(c, stream)
    elif case == "observation":
        o.Observation("http://127.0.0.1:1/v1", c["server_log"],
                      Path(c["startup_log"]["path"]).parent, c["run_id"], c["server_pid"], c)
    else:
        raise AssertionError("unknown fixture case")
except ValueError as error:
    if "regular C2 evidence" not in str(error):
        raise
else:
    raise AssertionError("FIFO evidence accepted")
"""
        for case in ("startup", "cache", "identity", "observation"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                control = context(Path(directory) / "client", 1, fake.RUN, fake.SERVER_PID)
                path = Path(control["startup_log"]["path"] if case in ("startup", "identity")
                            else control["server_log"])
                path.unlink()
                os.mkfifo(path)
                subprocess.run([sys.executable, "-E", "-s", "-B", "-c", script,
                                json.dumps(control), case], cwd=Path(__file__).parent,
                               check=True, timeout=5, capture_output=True)


class LiveControls(unittest.TestCase):
    def test_actual_live_k1_edit_completes_tools_but_next_unchanged_turn_stays_warm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control = context(root / "client", 1, fake.RUN, fake.SERVER_PID)
            with fake.FakeService(c2_depth=1) as service:
                result = fake.drive_agent(service.url, root / "client/agent-0", "agent-0",
                                          variant=VARIANT, cache_control=control)
            self.assertEqual(result["status"], "ProtocolCompleteAwaitingIndependentPatchReview")
            self.assertEqual(len(result["tools"]), 6)
            self.assertEqual([result["turns"][i]["usage"]["prompt_tokens_details"]["cached_tokens"]
                              for i in (3, 5)], [0, 0])
            self.assertGreater(result["turns"][4]["usage"]["prompt_tokens_details"]["cached_tokens"], 0)

    def test_k4_missing_context_and_unchanged_or_repair_tail_zero_cache_are_rejected(self):
        scenarios = ((4, {}, 0), (None, {}, 0), (1, {(None, 4): "no_reuse"}, 0),
                     (1, {(None, 7): "no_reuse"}, 1))
        for depth, faults, repair in scenarios:
            with self.subTest(depth=depth, faults=faults, repair=repair), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                control = None if depth is None else context(root / "client", depth, fake.RUN, fake.SERVER_PID)
                with fake.FakeService(c2_depth=1, faults=faults, repair=bool(repair)) as service:
                    with self.assertRaisesRegex(Invalid, "prefix reuse not proven"):
                        fake.drive_agent(service.url, root / "client/agent-0", "agent-0", variant=VARIANT,
                                         cache_control=control, repair_turns=repair)

    def test_an_actual_no_op_transform_does_not_grant_cold_permission(self):
        selected = coding_long_context.select(VARIANT)
        unchanged = "\n".join([f"line-{i}" for i in range(16)] + [selected.marker])
        messages = [{"role": "tool", "content": "first"}, {"role": "tool", "content": unchanged}]
        edit = selected.before_request(messages, 3)
        self.assertEqual(edit["original_sha256"], edit["edited_sha256"])
        self.assertFalse(fake.coordinator.edit_cache_miss_allowed({"anchor_depth": 1}, selected, 3, edit))
        with self.assertRaisesRegex(Invalid, "prefix reuse not proven"):
            replay_response(responses(), {"model": "model", "stream": True}, 3, set(), "hf2q",
                            cache_miss_allowed=False)


class SettledAuditControls(unittest.TestCase):
    def run_fixture(self, root, depth=1, repair=0):
        client = root / "client"
        summary = fake.honest_run(client, variant=VARIANT, c2_depth=depth, repair_turns=repair)
        receipt = json.loads((root / "c2-fixture-receipt.json").read_bytes())
        return client, summary, receipt

    def test_all_raw_exchanges_replay_with_independently_derived_k1_edit_permission(self):
        for depth, repair, count in ((1, 0, 28), (4, 0, 28), (1, 1, 36)):
            with self.subTest(depth=depth, repair=repair), tempfile.TemporaryDirectory() as directory:
                client, summary, receipt = self.run_fixture(Path(directory), depth, repair)
                report, failure = check_run(receipt, client, summary, None, "hf2q", require_width=True)
                self.assertIsNone(failure)
                self.assertEqual(report["response_ids"], count)
                self.assertEqual(report["progress"]["width"], 4)

    def test_forged_depth_child_context_and_missing_outer_approval_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            client, summary, receipt = self.run_fixture(Path(directory))
            variants = []
            bad = copy.deepcopy(summary)
            bad["c2_cache_control"]["anchor_depth"] = 4
            variants.append((receipt, bad))
            bad = copy.deepcopy(summary)
            bad["agents"]["agent-0"]["c2_cache_control"]["anchor_depth"] = 4
            variants.append((receipt, bad))
            variants.append(({}, summary))
            for outer, value in variants:
                with self.subTest(outer=bool(outer)):
                    report, failure = check_run(outer, client, value, None, "hf2q", require_width=True)
                    self.assertIsNone(report)
                    self.assertIsNotNone(failure)

    def test_resealed_zero_cache_on_unchanged_turn_cannot_use_an_earlier_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            client, summary, receipt = self.run_fixture(Path(directory))
            result = summary["agents"]["agent-0"]
            path = client / "agent-0/turn-4/response.bin"
            raw = json.loads(path.read_bytes())
            raw["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
            encoded = json.dumps(raw).encode()
            path.write_bytes(encoded)
            result["turns"][4].update(usage=raw["usage"], response_bytes=len(encoded),
                                       response_sha256=hashlib.sha256(encoded).hexdigest(), cache_miss_allowed=True)
            report, failure = check_run(receipt, client, summary, None, "hf2q", require_width=True)
            self.assertIsNone(report)
            self.assertIn("prefix reuse not proven", failure)

    def test_resealed_skipped_history_edit_is_still_a_lineage_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            client, summary, receipt = self.run_fixture(Path(directory))
            result = summary["agents"]["agent-0"]
            path = client / "agent-0/turn-3/request.json"
            request = json.loads(path.read_bytes())
            request["messages"][5]["content"] = "not the pinned transform"
            encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
            path.write_bytes(encoded)
            result["turns"][3].update(request_bytes=len(encoded), request_sha256=hashlib.sha256(encoded).hexdigest())
            report, failure = check_run(receipt, client, summary, None, "hf2q", require_width=True)
            self.assertIsNone(report)
            self.assertIn("stale lineage", failure)


if __name__ == "__main__":
    unittest.main()
