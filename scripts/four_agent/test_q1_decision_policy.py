"""Synthetic controls for the Q1 decision policy; never a model or measured result."""
import array
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import q1_decision_policy as q
from qwen38_n1_verdict import Invalid

BASE = [0.0, 3.0, 2.95, 1.0, -1.0, 0.5, -2.0, 0.25]


def store(root, name, rows, drop=None, corrupt=None):
    directory = Path(root) / name
    directory.mkdir()
    captured, expected = [], []
    for index, (agent, row, values) in enumerate(rows):
        at = {"logical_step": 0, "layer": 64, "stage": "Logits", "agent": agent,
              "logical_row_start": row, "segment": 0}
        payload = array.array("f", values).tobytes()
        file = f"tensor-{index:06d}.bin"
        (directory / file).write_bytes(payload if index != corrupt else payload[::-1])
        expected.append({"at": at, "scalar": "F32", "shape": [1, len(values)]})
        if index != drop:
            captured.append({**expected[-1], "file": file, "sha256": hashlib.sha256(payload).hexdigest()})
    if drop is not None:
        expected.pop(drop)
    trace = {"schema": q.TRACE_SCHEMA, "coverage": {"expected": expected, "captured": captured, "errors": []}}
    path = directory / "trace.json"
    path.write_text(json.dumps(trace))
    return path


def shifted(index, delta):
    values = list(BASE)
    values[index] += delta
    return values


class Envelope(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.left = store(self.root, "left", [("agent-0", 0, BASE), ("agent-1", 0, BASE)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_bytes_are_exact(self):
        right = store(self.root, "right", [("agent-0", 0, BASE), ("agent-1", 0, BASE)])
        result = q.envelope(self.left, right)
        self.assertEqual(result["classification"], "ExactEqual")
        self.assertEqual(result["decision_error"]["max"], 0.0)

    def test_near_tie_flip_inside_tau_is_within_envelope(self):
        # Token 2 overtakes token 1 by 0.07 nats: a flip on a 0.05-nat margin.
        right = store(self.root, "right", [("agent-0", 0, shifted(2, 0.12)), ("agent-1", 0, BASE)])
        result = q.envelope(self.left, right)
        self.assertEqual(result["classification"], "WithinDecisionEnvelope")
        self.assertEqual(len(result["flips"]), 1)
        self.assertAlmostEqual(result["flips"][0]["decision_error"], 0.12, places=5)
        self.assertGreater(result["kl_nats"]["max"], 0.0)

    def test_large_shift_without_flip_is_route_sensitive(self):
        right = store(self.root, "right", [("agent-0", 0, BASE), ("agent-1", 0, shifted(1, 0.6))])
        result = q.envelope(self.left, right)
        self.assertEqual(result["classification"], "RouteSensitive")
        self.assertEqual(result["flips"], [])
        self.assertEqual(result["rows_over_tau"][0]["agent"], "agent-1")

    def test_missing_logit_row_is_incomplete_not_subset_compared(self):
        right = store(self.root, "right", [("agent-0", 0, BASE), ("agent-1", 0, BASE)], drop=1)
        self.assertEqual(q.envelope(self.left, right)["classification"], "Incomplete")

    def test_payload_hash_mismatch_refuses(self):
        right = store(self.root, "right", [("agent-0", 0, BASE), ("agent-1", 0, BASE)], corrupt=0)
        with self.assertRaises(Invalid):
            q.envelope(self.left, right)

    def test_exact_store_comparison_reports_unpaired_and_first_difference(self):
        same = store(self.root, "same", [("agent-0", 0, BASE), ("agent-1", 0, BASE)])
        self.assertEqual(q.exact(self.left, same)["classification"], "ExactEqual")
        extra = store(self.root, "extra", [("agent-0", 0, BASE), ("agent-1", 0, BASE), ("agent-2", 0, BASE)])
        result = q.exact(self.left, extra)
        self.assertEqual((result["classification"], result["right_only"]), ("ExactOnCommon", 1))
        one_ulp = list(BASE)
        one_ulp[5] = 0.5000001
        changed = store(self.root, "changed", [("agent-0", 0, BASE), ("agent-1", 0, one_ulp)])
        result = q.exact(self.left, changed)
        self.assertEqual(result["classification"], "DivergedExactInvariant")
        self.assertEqual(result["first_differing"][:2], ["Logits", "agent-1"])


def response(tokens):
    content = [{"token": t, "logprob": dict(top)[t], "bytes": list(t.encode()),
                "top_logprobs": [{"token": x, "logprob": lp, "bytes": list(x.encode())} for x, lp in top]}
               for t, top in tokens]
    return {"choices": [{"logprobs": {"content": content}}]}


SHARED = [("a", -0.1), ("b", -2.5), ("c", -3.0), ("d", -4.0), ("e", -5.0)]


class Fork(unittest.TestCase):
    def prefix(self, wobble=0.0):
        top = [(t, lp + (wobble if t == "b" else 0.0)) for t, lp in SHARED]
        return [("a", top)] * 4

    def test_identical_streams_have_no_fork(self):
        stream = response(self.prefix())
        self.assertEqual(q.fork(stream, stream)["classification"], "NoFork")

    def test_top_two_swap_inside_floor_is_near_tie(self):
        hf = response(self.prefix() + [("x", [("x", -0.70), ("y", -0.72), ("c", -3), ("d", -4), ("e", -5)])])
        peer = response(self.prefix(0.01) + [("y", [("y", -0.70), ("x", -0.73), ("c", -3), ("d", -4), ("e", -5)])])
        result = q.fork(hf, peer)
        self.assertEqual(result["fork_index"], 4)
        self.assertEqual(result["classification"], "NearTie")
        self.assertAlmostEqual(result["decision_error_D"], 0.05, places=6)

    def test_fork_token_outside_top_k_is_numeric_suspect(self):
        hf = response(self.prefix() + [("x", [("x", -0.1), ("c", -3), ("d", -4), ("e", -5), ("f", -6)])])
        peer = response(self.prefix() + [("y", [("y", -0.1), ("x", -2), ("c", -3), ("d", -4), ("e", -5)])])
        self.assertEqual(q.fork(hf, peer)["classification"], "NumericSuspect")

    def test_large_disagreement_inside_top_two_is_numeric_suspect(self):
        hf = response(self.prefix() + [("x", [("x", -0.1), ("y", -0.4), ("c", -3), ("d", -4), ("e", -5)])])
        peer = response(self.prefix() + [("y", [("y", -0.1), ("x", -0.9), ("c", -3), ("d", -4), ("e", -5)])])
        result = q.fork(hf, peer)
        self.assertEqual(result["classification"], "NumericSuspect")
        self.assertAlmostEqual(result["decision_error_D"], 1.1, places=6)

    def test_peer_completion_probabilities_shape_is_accepted(self):
        stream = response(self.prefix())
        peer = {"completion_probabilities": stream["choices"][0]["logprobs"]["content"]}
        self.assertEqual(q.fork(stream, peer)["classification"], "NoFork")


def outcomes(pairs, reference_kind="peer"):
    return {"schema": "hf2q.q1.paired-task-outcomes.v1", "reference_kind": reference_kind,
            "instances": [{"id": f"run-{i}", "subject": s, "reference": r} for i, (s, r) in enumerate(pairs)]}


class Paired(unittest.TestCase):
    def test_exact_one_sided_sign_probability(self):
        self.assertAlmostEqual(q.one_sided_sign_p(5, 0), 1 / 32)
        self.assertAlmostEqual(q.one_sided_sign_p(4, 0), 1 / 16)
        self.assertEqual(q.one_sided_sign_p(0, 0), 1.0)

    def test_five_unmatched_failures_are_a_loss(self):
        result = q.paired(outcomes([(False, True)] * 5 + [(True, True)] * 19))
        self.assertEqual(result["verdict"], "Loss")

    def test_four_unmatched_failures_are_inconclusive_not_pass(self):
        result = q.paired(outcomes([(False, True)] * 4 + [(True, True)] * 20))
        self.assertEqual(result["verdict"], "Inconclusive")

    def test_three_unmatched_failures_pass(self):
        self.assertEqual(q.paired(outcomes([(False, True)] * 3 + [(True, True)] * 21))["verdict"], "Pass")

    def test_shared_failures_below_floor_fail_even_without_a_loss(self):
        result = q.paired(outcomes([(False, False)] * 7 + [(True, True)] * 17))
        self.assertEqual(result["verdict"], "BelowFloor")

    def test_too_few_instances_are_insufficient(self):
        self.assertEqual(q.paired(outcomes([(True, True)] * 23))["verdict"], "Insufficient")

    def test_unreferenced_family_uses_the_floor_only(self):
        self.assertEqual(q.paired(outcomes([(True, None)] * 18 + [(False, None)] * 6, None))["verdict"], "Pass")
        self.assertEqual(q.paired(outcomes([(True, None)] * 17 + [(False, None)] * 7, None))["verdict"], "BelowFloor")

    def test_mixed_reference_and_duplicates_refuse(self):
        with self.assertRaises(Invalid):
            q.paired(outcomes([(True, True)] * 23 + [(True, None)]))
        doc = outcomes([(True, True)] * 24)
        doc["instances"][1]["id"] = "run-0"
        with self.assertRaises(Invalid):
            q.paired(doc)


def client_run(root, name, turns):
    for (agent, turn), (history, answer) in turns.items():
        directory = Path(root) / name / "client" / agent / turn
        directory.mkdir(parents=True)
        (directory / "request.json").write_text(json.dumps({"messages": history}))
        (directory / "assembled-assistant.json").write_text(json.dumps(answer))
    return Path(root) / name


def call(arguments, call_id):
    return {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
            "function": {"name": "read_file", "arguments": arguments}}]}


class Transcripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_call_ids_are_ignored_and_divergent_histories_are_not_compared(self):
        user = [{"role": "user", "content": "fix it"}]
        a = client_run(self.tmp.name, "a", {
            ("agent-0", "turn-0"): (user, call('{"path": "a"}', "call_hf2q_01")),
            ("agent-0", "turn-1"): (user + [{"tool_call_id": "call_hf2q_01", "content": "t=1"}], {"content": "x"})})
        b = client_run(self.tmp.name, "b", {
            ("agent-0", "turn-0"): (user, call('{"path": "a"}', "call_hf2q_ff")),
            ("agent-0", "turn-1"): (user + [{"tool_call_id": "call_hf2q_ff", "content": "t=2"}], {"content": "y"})})
        result = q.transcripts([a, b])
        self.assertEqual(result["classification"], "ExactTranscripts")
        self.assertEqual(result["same_request_groups"], 3)

    def test_same_history_different_answer_is_divergent(self):
        user = [{"role": "user", "content": "fix it"}]
        a = client_run(self.tmp.name, "a", {("agent-1", "turn-0"): (user, call('{"path": "a"}', "call_hf2q_01"))})
        b = client_run(self.tmp.name, "b", {("agent-1", "turn-0"): (user, call('{"path": "b"}', "call_hf2q_01"))})
        result = q.transcripts([a, b])
        self.assertEqual(result["classification"], "DivergedTranscripts")
        self.assertEqual(result["divergent"][0]["agent"], "agent-1")


if __name__ == "__main__":
    unittest.main()
