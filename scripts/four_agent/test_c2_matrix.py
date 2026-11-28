"""Untrusted metadata fixtures only: these tests never attest native acquisition."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from c2_matrix import classification
from c2_matrix_io import Scope, acquisition, coordinate, input_digest
from c2_matrix_plans import decision, expected, native_inputs, projection
from c2_matrix_evidence import canonical_arm, capability_markers, same_raw_twins, settlement_proof
from q1_decision_policy import exact


class C2ReaderFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hf2q-c2-reader-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.identity = {"family": "Qwen38", "binary_sha256": "1" * 64}
        self.actual = dict(packet_sha256="2" * 64, nonce="3" * 64, output_root=str(self.root),
                           actual_binary_sha256="1" * 64, composition_sha256="4" * 64,
                           independent_review_sha256="5" * 64,
                           opened_model=dict(device=1, inode=2, bytes=3, mtime=4, mtime_ns=5, ctime=6, ctime_ns=7),
                           observed_model_opens=2, required_model_opens=1, device_registry_id=8,
                           queue_identity=9, model_instance=10, provenance="IndependentRootReviewedComposition")
        packet = dict(identity=self.identity, output=str(self.root), host_payload_limit=4096)
        registration = dict(grants=dict(payload_bytes=4096, events=64))
        self.scope = Scope(packet, self.actual["packet_sha256"], registration, "6" * 64, self.actual)

    def write(self, path, value):
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return {"path": str(path.relative_to(self.root)), "sha256": hashlib.sha256(raw).hexdigest()}

    def request(self):
        return dict(sequence=0, slot=0, start=10, input_rows=3, projected_rows=2,
                    vocab=4, hidden_width=3, origin="Native")


    def trace_store(self, name, descriptors, payloads):
        root = self.root / name
        root.mkdir()
        captured = []
        for index, (descriptor, data) in enumerate(zip(descriptors, payloads)):
            filename = f"row-{index}.bin"
            (root / filename).write_bytes(data)
            captured.append({**descriptor, "file": filename, "sha256": hashlib.sha256(data).hexdigest(),
                             "logical_bytes": len(data), "checked_drain": True})
        trace = dict(schema="hf2q.adr063.invariance.v2", identity=self.identity, comparison_key=self.identity,
                     acquisition={**self.actual, "output_root": str(root)},
                     coverage=dict(expected=descriptors, captured=captured, errors=[], missing_native_capabilities=[]))
        path = root / "trace.json"
        return (path, trace), self.write(path, trace)


class CompleteShardControls(C2ReaderFixture):
    def test_same_acquisition_header_can_precede_a_legitimate_graft_header_open_but_not_its_lifetime(self):
        earlier = dict(self.actual, observed_model_opens=1)
        acquisition(earlier, self.actual, self.root)
        with self.assertRaises(ValueError):
            acquisition(dict(earlier, observed_model_opens=3), self.actual, self.root)
        with self.assertRaises(ValueError):
            acquisition(dict(earlier, nonce="a" * 64), self.actual, self.root)
        with self.assertRaises(ValueError):
            acquisition(dict(earlier, provenance="Synthetic"), self.actual, self.root)

    def test_source_receipt_cannot_omit_its_opened_descriptor_even_symmetrically(self):
        incomplete = copy.deepcopy(self.actual)
        del incomplete["opened_model"]["ctime_ns"]
        with self.assertRaises(ValueError):
            acquisition(incomplete, incomplete, self.root)
        incomplete = copy.deepcopy(self.actual)
        del incomplete["model_instance"]
        with self.assertRaises(ValueError):
            acquisition(incomplete, incomplete, self.root)

    def test_bitwise_equal_twins_cannot_erase_the_same_verifier_and_hidden_tail(self):
        required = projection(self.request(), "case/slot0")
        truncated = [item for item in required if item["at"]["logical_row_start"] == 11]
        bindings, paths = [], []
        for arm in ("actual", "twin"):
            root = self.root / arm
            root.mkdir()
            captured = []
            for index, descriptor in enumerate(truncated):
                data = b"\0" * (math.prod(descriptor["shape"]) * 4)
                name = f"row-{index}.bin"
                (root / name).write_bytes(data)
                captured.append({**descriptor, "file": name, "sha256": hashlib.sha256(data).hexdigest(),
                                 "logical_bytes": len(data), "checked_drain": True})
            trace = dict(schema="hf2q.adr063.invariance.v2", identity=self.identity, comparison_key=self.identity,
                         acquisition={**self.actual, "output_root": str(root)},
                         coverage=dict(expected=truncated, captured=captured, errors=[], missing_native_capabilities=[]))
            paths.append(root / "trace.json")
            bindings.append(self.write(paths[-1], trace))
        self.assertEqual(exact(*paths)["classification"], "ExactEqual")
        for binding in bindings:
            with self.assertRaises(ValueError):
                self.scope.trace(binding, required)

    def test_canonical_selector_cannot_relabel_a_grafted_physical_position_as_a_decision_index(self):
        request = dict(semantic_index=0, physical_position=0, vocab=4, hidden_width=3)
        with self.assertRaises(ValueError):
            decision(request, "case/slot0", 100)

    def test_input_custody_rejects_coherently_rehashed_substitution_and_whole_record_omission(self):
        requested = self.request()
        source = {"Host": {"sha256": input_digest([1, 2, 3])}}
        header = dict(identity=self.identity, acquisition=self.actual, registration_sha256="6" * 64,
                      case="case", arm="actual", slot=0)
        execution = {"path": "execution.json", "sha256": "7" * 64}
        bundle = dict(binding=execution, context=header, inputs={0: source}, forwards={0: requested})
        record = dict(sequence=0, slot=0, start=10, source=source, actual_tokens=[1, 9, 3],
                      actual_sha256=input_digest([1, 9, 3]))
        document = dict(schema="hf2q.c2.native-input-records.v1", context=header, execution=execution, records=[record])
        binding = self.write(self.root / "substituted.json", document)
        with self.assertRaises(ValueError):
            native_inputs(self.scope, "case", "actual", binding, bundle)
        document["records"] = []
        binding = self.write(self.root / "omitted.json", document)
        with self.assertRaises(ValueError):
            native_inputs(self.scope, "case", "actual", binding, bundle)


class O1SensitivityControls(C2ReaderFixture):
    def setUp(self):
        self.case = "o1-shallowertrue-pruningdisabledtrue"
        super().setUp()
        self.scope.registration["mode"] = "o1"
        self.value = dict(schema="hf2q.c2.o1.v1", shallower=True, pruning_disabled=True,
                          b_completed_high_cursor=True, second_fault=False, twin_image_restores=0,
                          tombstoned=1, restore_refused=1, classification="DivergedExactInvariant")
        self.equal = [{"classification": "ExactEqual", "common_records": 1}]

    def test_pruning_only_negative_cannot_relabel_itself_as_a_positive_case(self):
        changed = dict(self.value, shallower=False, pruning_disabled=False, classification="ExactEqual")
        with self.assertRaises(ValueError):
            classification(self.scope, self.case, changed, {"sha256": "1" * 64}, {}, {}, self.equal)

    def test_declared_divergence_cannot_substitute_for_actual_raw_or_state_sensitivity(self):
        with self.assertRaises(ValueError):
            classification(self.scope, self.case, self.value, {"sha256": "1" * 64}, {}, {}, self.equal)


class RequiredJoinControls(C2ReaderFixture):
    def test_equal_outputs_do_not_hide_different_consumed_verifier_inputs(self):
        descriptor = expected("Logits", "case/slot0", 0, 11, "F32", [1, 4])
        left, _ = self.trace_store("left-output", [descriptor], [b"\0" * 16])
        right, _ = self.trace_store("right-output", [descriptor], [b"\0" * 16])
        self.assertEqual(exact(left[0], right[0])["classification"], "ExactEqual")
        request = self.request()
        route = dict(routes=["fixture"], actual_gpu_argmax=[], completed_route_events=[], initial_readiness="fixture")
        left = dict(forwards={0: request}, inputs={0: dict(sequence=0, slot=0, start=10, actual_tokens=[1, 2, 3])}, route=route)
        right = copy.deepcopy(left)
        right["inputs"][0]["actual_tokens"][1] = 9
        with self.assertRaises(ValueError):
            same_raw_twins(left, right)

    def test_equal_final_snapshots_do_not_replace_divergent_cancelled_settlement_states(self):
        ledger = expected("CacheState", "case/slot0", 0, 0, "U32", [1], (1 << 32) - 1)
        actual, _ = self.trace_store("actual-final", [ledger], [(1).to_bytes(4, "little")])
        twin, _ = self.trace_store("twin-final", [ledger], [(1).to_bytes(4, "little")])
        settled, _ = self.trace_store("actual-settled", [ledger], [(9).to_bytes(4, "little")])
        self.assertEqual(exact(actual[0], twin[0])["classification"], "ExactEqual")
        with self.assertRaises(ValueError):
            settlement_proof(actual, [settled], {"Cancelled": {"recovered_tokens": 1}})

    def test_typed_recovery_count_must_match_the_exact_recovered_ledger(self):
        ledger = expected("CacheState", "case/slot0", 0, 0, "U32", [2], (1 << 32) - 1)
        final, _ = self.trace_store("recovered-two", [ledger], [(1).to_bytes(4, "little") * 2])
        with self.assertRaises(ValueError):
            settlement_proof(final, [final], {"Reference": {"recovered_tokens": 1}})

    def test_canonical_rows_reject_rejected_prefix_repeated_disagreement_and_wrong_source_route(self):
        request = dict(semantic_index=0, physical_position=1, vocab=2, hidden_width=1)
        _, canonical = decision(request, "case/slot0", 1)
        token = (1).to_bytes(4, "little")
        captured, trace_binding = self.trace_store("canonical", canonical, [b"\0" * 8, b"\0" * 4, token])
        route_binding = self.write(self.root / "canonical-routes.json", [[0, 1, "fixture"]])
        execution = {"path": "execution.json", "sha256": "7" * 64}
        binding = self.write(self.root / "canonical-binding.json",
            dict(schema="hf2q.c2.canonical-arm-binding.v1", identity=self.identity, scope_acquisition=self.actual,
                 case="case", arm="actual", source_arm="actual", execution=execution,
                 trace=trace_binding, route_observation=route_binding))
        forwards = {sequence: dict(sequence=sequence, slot=0, start=0, input_rows=2, projected_rows=1,
                                  vocab=2, hidden_width=1, origin="Native") for sequence in (0, 1)}
        bundle = dict(binding=execution, decisions={0: request}, canonical=canonical, forwards=forwards)
        source_rows = {}
        for sequence in (0, 1):
            for stage, width in (("Logits", 2), ("LayerOutput", 1)):
                descriptor = expected(stage, "case/slot0", sequence, 1, "F32", [1, width])
                data = (b"\1" if sequence == 0 else b"\0") * (width * 4)
                source_rows[coordinate(descriptor["at"])] = {"sha256": hashlib.sha256(data).hexdigest()}
        source_rows[("SpecCanonicalIds", "case/slot0", 0, 0, 0, 0)] = {"sha256": hashlib.sha256(token).hexdigest()}
        raw = dict(rows=source_rows, inputs={0: {"actual_tokens": [1, 2]}, 1: {"actual_tokens": [1, 9]}},
                   route={"routes": ["fixture", "fixture"]})
        with self.assertRaises(ValueError):
            canonical_arm(self.scope, "case", "actual", "actual", binding, bundle, raw, [1, 2], [], 0)
        raw["inputs"][1]["actual_tokens"] = [1, 2]
        with self.assertRaises(ValueError):
            canonical_arm(self.scope, "case", "actual", "actual", binding, bundle, raw, [1, 2], [], 0)
        for stage, width in (("Logits", 2), ("LayerOutput", 1)):
            descriptor = expected(stage, "case/slot0", 0, 1, "F32", [1, width])
            source_rows[coordinate(descriptor["at"])] = {"sha256": hashlib.sha256(b"\0" * (width * 4)).hexdigest()}
        raw["route"]["routes"] = ["actual-source", "actual-source"]
        with self.assertRaises(ValueError):
            canonical_arm(self.scope, "case", "actual", "actual", binding, bundle, raw, [1, 2], [], 0)

    def test_marker_only_mutations_cannot_invent_observed_k3_or_eligible_grafted_admission(self):
        self.scope.packet.update(route="Off", graft=None)
        value = dict(equality=False, literal_k3_capability="NotDeclaredK3", resolved_speculation="C2-SlotAware4-OFF")
        for arm in ("actual", "twin", "single"):
            value.update({f"{arm}_lane0_admission_decision": "disabled", f"{arm}_route_events": [],
                          f"{arm}_lane0_completed_k3": 0, f"{arm}_lane0_completed_joint_mtp": 0})
        raw = {arm: {"route": {"completed_route_events": []}} for arm in ("actual", "twin")}
        capability_markers(self.scope, value, raw)
        with self.assertRaises(ValueError):
            capability_markers(self.scope, dict(value, literal_k3_capability="ObservedActualK3"), raw)
        self.scope.packet.update(route="MtpAuto", graft="declared-fixture-bank")
        value.update(equality=True, literal_k3_capability="PromptCacheHit", resolved_speculation="PromptCacheHit/ordinary-grafted")
        for arm in ("actual", "twin", "single"):
            value[f"{arm}_lane0_admission_decision"] = "eligible"
        with self.assertRaises(ValueError):
            capability_markers(self.scope, value, raw)


if __name__ == "__main__":
    unittest.main()
