"""Synthetic controls for the Q1 decision policy; never a model or measured result."""
import array
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import q1_decision_policy as q
import q1_stage_localization as loc
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

    def scope(self, keys):
        path = Path(self.root) / f"scope-{len(list(Path(self.root).iterdir()))}.json"
        path.write_text(json.dumps(keys))
        return path

    def test_scoped_exact_compares_only_the_closed_key_list(self):
        one_ulp = list(BASE)
        one_ulp[5] = 0.5000001
        changed = store(self.root, "changed", [("agent-0", 0, BASE), ("agent-1", 0, one_ulp), ("agent-2", 0, BASE)])
        agent0, agent1 = ["Logits", "agent-0", 0, 64, 0, 0], ["Logits", "agent-1", 0, 64, 0, 0]
        path = self.scope([agent0])
        result = q.exact(self.left, changed, path)  # agent-1 differs and agent-2 is unpaired, both out of scope
        self.assertEqual((result["classification"], result["common_records"], result["left_only"],
                          result["right_only"], result["differing_records"], result["scoped_records"]),
                         ("ExactEqual", 1, 0, 0, 0, 1))
        self.assertEqual(result["scope_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        result = q.exact(self.left, changed, self.scope([agent0, agent1]))
        self.assertEqual((result["classification"], result["first_differing"]), ("DivergedExactInvariant", agent1))
        self.assertNotIn("scope_sha256", q.exact(self.left, changed))

    def test_scoped_exact_refuses_missing_or_malformed_keys(self):
        right = store(self.root, "right", [("agent-0", 0, BASE), ("agent-1", 0, BASE), ("agent-2", 0, BASE)])
        for keys in ([["Logits", "agent-2", 0, 64, 0, 0]], [], [["Logits", "agent-0", 0, 64, 0]],
                     [["Logits", "agent-0", 0, 64, 0, True]], [["Logits", "agent-0", 0, 64, 0, 0]] * 2):
            with self.assertRaises(Invalid):
                q.exact(self.left, right, self.scope(keys))


def bits(value):
    return array.array("I", array.array("f", [value]).tobytes())[0]


IDENTITY = {"family": "Gemma4", "model_sha256": "m" * 64, "token_vectors_sha256": "t" * 64}


def routing_row(lane, position, layer, selected, kth=0.30, nxt=0.29, namespace="Canonical"):
    return {"kind": "routing_row",
            "at": {"namespace": namespace, "layer": layer, "row": {"lane": lane, "position": position},
                   "domain": "SoftmaxBeforeTopKRenorm"},
            "scores_f32_bits": [bits(kth), bits(nxt)], "boundary_ids": [selected[-1], 99],
            "selected_ids": selected, "selected_weights_f32_bits": [bits(0.5)] * len(selected),
            "native_boundary_valid": True, "selected_copy_valid": True, "native_decode_error": None}


def witness(root, name, rows, identity=IDENTITY, records=None):
    path = Path(root) / f"{name}.jsonl"
    events = [{"kind": "header", "identity": identity}, *rows,
              {"kind": "end", "records": len(rows) if records is None else records,
               "all_scored_rows_captured": True}]
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    return path


class Routing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.rows = [routing_row(lane, 0, layer, [1, 2, 3]) for lane in range(2) for layer in range(2)]
        self.left = witness(self.root, "left", self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identical_selection_is_exact_routing(self):
        result = q.routing(self.left, witness(self.root, "right", list(reversed(self.rows))))
        self.assertEqual((result["classification"], result["rows"], result["changed_rows"]),
                         ("ExactRouting", 4, 0))

    def test_flip_and_order_change_report_both_margins(self):
        rows = list(self.rows)
        rows[1] = routing_row(0, 0, 1, [1, 2, 99], kth=0.295, nxt=0.30)
        rows[2] = routing_row(1, 0, 0, [2, 1, 3])
        result = q.routing(self.left, witness(self.root, "right", rows))
        self.assertEqual(result["classification"], "RoutingChanged")
        self.assertEqual((result["changed_rows"], result["set_changed_rows"], result["order_only_rows"]),
                         (2, 1, 1))
        flip = next(row for row in result["changed"] if row["set_changed"])
        self.assertEqual((flip["lane"], flip["layer"], flip["right"]["selected_ids"]), (0, 1, [1, 2, 99]))
        self.assertAlmostEqual(flip["left"]["gap"], 0.01, places=6)
        self.assertAlmostEqual(flip["right"]["gap"], -0.005, places=6)
        self.assertEqual(flip["left"]["domain"], "SoftmaxBeforeTopKRenorm")

    def test_unpaired_duplicate_or_incomplete_witness_refuses(self):
        with self.assertRaises(Invalid):
            q.routing(self.left, witness(self.root, "short", self.rows[:3]))
        with self.assertRaises(Invalid):
            q.routing(self.left, witness(self.root, "dup", self.rows + self.rows[:1]))
        with self.assertRaises(Invalid):
            q.routing(self.left, witness(self.root, "count", self.rows, records=5))
        other = {**IDENTITY, "token_vectors_sha256": "u" * 64}
        with self.assertRaises(Invalid):
            q.routing(self.left, witness(self.root, "inputs", self.rows, identity=other))
        bad = dict(self.rows[0], selected_copy_valid=False)
        with self.assertRaises(Invalid):
            q.routing(self.left, witness(self.root, "invalid", [bad, *self.rows[1:]]))

    def test_namespace_is_part_of_the_join_key(self):
        rows = self.rows + [routing_row(0, 0, 0, [1, 2, 3], namespace="UnpublishedCancelled")]
        with self.assertRaises(Invalid):
            q.routing(self.left, witness(self.root, "extra", rows))


ACCEPTED = [0, 1, 2, 1]


class RoutingMapped(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        plan = {"rerun_id": "r", "layers": [0], "prefill_stages": ["LayerOutput"], "decode_stages": ["LayerOutput"],
                "host_payload_limit": 1, "accepted": ACCEPTED}
        receipts = {"ngram": {"path": "n", "sha256": "1" * 64}, "ordinary": {"path": "o", "sha256": "2" * 64}}
        self.doc = loc.build_map("N", 1, plan, receipts, ACCEPTED)
        self.map = Path(self.root) / "map.json"
        self.map.write_text(json.dumps(self.doc))
        self.reference = {(lane, p, l): routing_row(lane, p, l, [1, 2, 3])
                          for lane in range(4) for p in range(144) for l in range(30)}
        self.candidate = {}
        for lane in range(4):
            for namespace, span in (("T5Initial", range(128)), ("T5Verify", range(132)),
                                    ("T5Continuation", range(129 + ACCEPTED[lane], 144))):
                for p in span:
                    for l in range(30):
                        self.candidate[(namespace, lane, p, l)] = routing_row(lane, p, l, [1, 2, 3], namespace=namespace)

    def tearDown(self):
        self.tmp.cleanup()

    def run_mapped(self, candidate=None, reference=None, identity=IDENTITY, key_map=None):
        left = witness(self.root, "candidate", list((candidate or self.candidate).values()), identity=identity)
        right = witness(self.root, "reference", list((reference or self.reference).values()))
        return q.routing_mapped(left, right, key_map or self.map)

    def test_phases_pair_each_canonical_row_once_without_t5_initial(self):
        result = self.run_mapped()
        self.assertEqual((result["schema"], result["classification"]), ("hf2q.q1.routing-comparison.v1", "ExactRouting"))
        self.assertEqual(result["phase_pairs"], {"verify-prefix": 4 * 128 * 30,
                                                 "verify-committed": 30 * sum(a + 1 for a in ACCEPTED),
                                                 "continuation": 30 * sum(15 - a for a in ACCEPTED)})
        self.assertEqual(sum(result["phase_pairs"].values()), 4 * 144 * 30)  # Canonical 0..127 not double counted
        self.assertNotIn("changed_rows", result)  # no cross-phase total
        for pairs in (p["pairs"] for p in self.doc["routing"]["phases"]):
            self.assertFalse(any(l[0] == "T5Initial" for l, _ in pairs))
        namespaces = [(u["side"], u["namespace"]) for u in result["unmapped"]]
        self.assertEqual(namespaces.count(("candidate", "T5Initial")), 4 * 128 * 30)
        self.assertEqual(namespaces.count(("candidate", "T5Verify")), 30 * sum(3 - a for a in ACCEPTED))
        self.assertEqual(result["map_sha256"], hashlib.sha256(self.map.read_bytes()).hexdigest())

    def test_changes_are_reported_per_phase_with_gaps(self):
        candidate = dict(self.candidate)
        candidate[("T5Verify", 2, 129, 5)] = routing_row(2, 129, 5, [1, 2, 99], kth=0.295, nxt=0.30, namespace="T5Verify")
        candidate[("T5Continuation", 0, 140, 0)] = routing_row(0, 140, 0, [2, 1, 3], namespace="T5Continuation")
        candidate[("T5Initial", 1, 3, 3)] = routing_row(1, 3, 3, [7, 8, 9], namespace="T5Initial")  # unmapped
        result = self.run_mapped(candidate)
        self.assertEqual(result["classification"], "RoutingChanged")
        phases = result["phases"]
        self.assertEqual([phases[p]["changed_rows"] for p in ("verify-prefix", "verify-committed", "continuation")],
                         [0, 1, 1])
        committed = phases["verify-committed"]
        self.assertEqual((committed["set_changed_rows"], committed["order_only_rows"]), (1, 0))
        self.assertAlmostEqual(committed["changed_gap_max"], 0.01, places=6)
        flip, = committed["changed"]
        self.assertEqual((flip["namespace"], flip["lane"], flip["position"], flip["right_key"]),
                         ("T5Verify", 2, 129, ["Canonical", 2, 129, 5]))
        self.assertAlmostEqual(flip["left"]["gap"], -0.005, places=6)
        self.assertEqual((phases["continuation"]["order_only_rows"], phases["continuation"]["set_changed_rows"]), (1, 0))
        listed = next(u for u in result["unmapped"] if (u["namespace"], u["lane"], u["position"], u["layer"])
                      == ("T5Initial", 1, 3, 3))
        self.assertEqual(listed["selected_ids"], [7, 8, 9])
        self.assertAlmostEqual(listed["gap"], 0.01, places=6)

    def test_unpaired_identity_or_out_of_namespace_refuses(self):
        candidate = dict(self.candidate)
        del candidate[("T5Continuation", 3, 143, 29)]
        with self.assertRaises(Invalid):
            self.run_mapped(candidate)
        reference = dict(self.reference)
        del reference[(0, 0, 0)]
        with self.assertRaises(Invalid):
            self.run_mapped(reference=reference)
        with self.assertRaises(Invalid):
            self.run_mapped(identity={**IDENTITY, "model_sha256": "x" * 64})
        doc = json.loads(self.map.read_text())
        doc["routing"]["phases"][0]["pairs"][0][0][0] = "T5Initial"  # a T5Initial <-> Canonical pair
        bad = Path(self.root) / "bad-map.json"
        bad.write_text(json.dumps(doc))
        with self.assertRaises(Invalid):
            self.run_mapped(key_map=bad)
        doc = json.loads(self.map.read_text())
        continuation = doc["routing"]["phases"][2]["pairs"]
        continuation[0][1] = doc["routing"]["phases"][0]["pairs"][0][1]  # Canonical 0..127 row mapped twice
        bad.write_text(json.dumps(doc))
        with self.assertRaises(Invalid):
            self.run_mapped(key_map=bad)


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

    def test_random_tool_call_ids_compare_exact_but_content_differences_do_not(self):
        user = [{"role": "user", "content": "fix it"}]

        def run(name, call_id, next_path):
            history = user + [call('{"path": "a"}', call_id), {"role": "tool", "tool_call_id": call_id, "content": "pub fn f()"}]
            return client_run(self.tmp.name, name, {
                ("agent-0", "turn-0"): (user, call('{"path": "a"}', call_id)),
                ("agent-0", "turn-1"): (history, call(json.dumps({"path": next_path}), call_id[::-1]))})
        # the peer's random 32-character ids differ in every run; here only the ids differ
        a = run("a", "WOVpsbvDqk3nC8xYr2LmT7aZp0eUw9Hf", "b")
        b = run("b", "Jb4tQ0zNcR6vYp1sKx8mWd3hLa5uFe2G", "b")
        result = q.transcripts([a, b])
        self.assertEqual((result["classification"], result["same_request_groups"]), ("ExactTranscripts", 2))
        # the same history with a different turn-1 answer still diverges once the ids are normalised
        c = run("c", "Zq9wXe8rTy7uIo6pAs5dFg4hJk3lMn2B", "c")
        result = q.transcripts([a, c])
        self.assertEqual(result["classification"], "DivergedTranscripts")
        self.assertEqual([(d["agent"], d["turn"]) for d in result["divergent"]], [("agent-0", "turn-1")])
        self.assertEqual(q.normalise_call_ids([{"tool_calls": [{"id": "x"}]}, {"tool_call_id": "x"}, {"tool_call_id": "y"}]),
                         [{"tool_calls": [{"id": "CALL0"}]}, {"tool_call_id": "CALL0"}, {"tool_call_id": "CALL1"}])


if __name__ == "__main__":
    unittest.main()
