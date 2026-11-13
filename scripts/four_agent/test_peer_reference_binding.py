"""Model-free consumer proofs for explicit frozen latest-peer identity and MTP exercise."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import family_peer as fp
import q5_calibration
from q5_case import CaseError
from qwen38_n1_verdict import Invalid
from test_family_peer import MODEL, TEMPLATE, ready

COMMIT = '5676dfbe01e7ebe0387913680eee0ed39bf4a363'
BUILD = 'b12352-5676dfbe0'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FrozenReference(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / 'server'
        self.binary.write_bytes(b'model-free executable fixture')
        self.library = self.root / 'sibling.dylib'
        self.library.write_bytes(b'model-free sibling fixture')
        self.receipt = self.root / 'identity.json'
        identity = {'source_commit': COMMIT, 'peer_build_info': BUILD, 'binary': str(self.binary),
                    'binary_sha256': digest(self.binary.read_bytes()),
                    'libraries_and_binary_sha256': {str(path): digest(path.read_bytes())
                                                   for path in (self.binary, self.library)}}
        self.receipt.write_text(json.dumps(identity))
        self.reference = {'identity_receipt': str(self.receipt), 'identity_sha256': digest(self.receipt.read_bytes()),
                          'source_commit': COMMIT, 'build_info': BUILD,
                          'serve_argv': [str(self.binary), '--host', '127.0.0.1', '--port', '18245',
                                         '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3'],
                          'selected_speculation': 'draft-mtp'}
        self.issuances = 0
        self.packet = self.issue(self.reference)
        self.reference = self.packet['peer_reference']
        patcher = patch.dict(fp.TEMPLATE_SHA256, {'gemma4': digest(TEMPLATE['gemma4'].encode())})
        patcher.start()
        self.addCleanup(patcher.stop)

    def issue(self, reference, actual_argv=None, template_extra=None, set_extra=None):
        self.issuances += 1
        out = self.root / f'issuance-{self.issuances}'
        out.mkdir()
        template = out / 'template.json'
        packet = {'peer_reference': reference, 'server_pid': 12345,
                  'base_url': 'http://127.0.0.1:18245/v1'}
        packet.update(template_extra or {})
        template.write_text(json.dumps(packet))
        argv = list(reference['serve_argv'] if actual_argv is None else actual_argv)
        server = SimpleNamespace(args=argv, pid=12345)
        actual = q5_calibration.supervised_peer_launch(server, {'binary': argv[0], 'port': '18245'})
        q5_calibration.render_approval({'template': str(template), 'set': set_extra or {}}, {}, out, actual)
        return json.loads((out / 'client-approval.json').read_text())

    def metadata(self, build=BUILD):
        health, props, slots = ready('gemma4', build_info=build)
        props['chat_template'] = TEMPLATE['gemma4']
        return health, props, slots

    def consume(self, packet=None, build=BUILD):
        with patch.object(fp.qwen_n4_peer, 'fetch_readiness', return_value=self.metadata(build)):
            return fp.readiness('http://127.0.0.1:18245/v1', MODEL['gemma4'], self.root, 'gemma4', packet=packet)

    def response(self, drafted=9, accepted=6, stream=False, missing_timings=False):
        turn = self.root / 'agent-0' / ('turn-1' if stream else 'turn-0')
        turn.mkdir(parents=True, exist_ok=True)
        value = {} if missing_timings else {'timings': {'draft_n': drafted, 'draft_n_accepted': accepted}}
        raw = json.dumps(value).encode()
        if stream:
            raw = b'data: {"choices": []}\n\n' + b'data: ' + raw + b'\n\ndata: [DONE]\n\n'
        (turn / 'response.bin').write_bytes(raw)
        (turn / 'receipt.json').write_text(json.dumps({'stream': stream, 'response_sha256': digest(raw)}))
        return turn

    def test_actual_readiness_consumer_accepts_frozen_latest_mtp_binding(self):
        result = self.consume(self.packet)
        self.assertEqual(result['source_revision'], COMMIT)
        self.assertEqual(result['build_info'], BUILD)
        self.assertEqual(result['selected_speculation'], 'draft-mtp')
        self.assertEqual(result['identity_receipt'], str(self.receipt))
        self.assertIsInstance(result['identity_receipt'], str)
        json.dumps({'peer_readiness': result})  # actual coordinator summary publication boundary
        # The actual sampler metadata is deliberately still 'none', not an engine launch verdict.
        self.assertEqual(self.metadata()[1]['default_generation_settings']['params']['speculative.types'], 'none')

    def test_wrong_or_old_actual_build_refuses_new_binding(self):
        for build in ('b128-e86f5fd', 'b12352-deadbee', None):
            with self.subTest(build=build), self.assertRaises(Invalid):
                self.consume(self.packet, build)

    def test_full_commit_and_build_receipt_are_both_bound(self):
        for commit in (COMMIT[:9], fp.PEER_COMMIT, 'f' * 40):
            packet = copy.deepcopy(self.packet)
            packet['peer_reference']['source_commit'] = commit
            with self.subTest(commit=commit), self.assertRaises(Invalid):
                self.consume(packet)
        self.receipt.write_text(self.receipt.read_text() + '\n')
        with self.assertRaisesRegex(Invalid, 'receipt hash'):
            self.consume(self.packet)

    def test_receipt_cannot_claim_a_different_source_even_with_updated_file_hash(self):
        identity = json.loads(self.receipt.read_text())
        identity['source_commit'] = 'f' * 40
        self.receipt.write_text(json.dumps(identity))
        self.reference['identity_sha256'] = digest(self.receipt.read_bytes())
        with self.assertRaisesRegex(Invalid, 'receipt source/build'):
            self.consume(self.packet)

    def test_changed_executable_or_sibling_refuses(self):
        for path in (self.binary, self.library):
            old = path.read_bytes()
            path.write_bytes(old + b'changed')
            with self.subTest(path=path.name), self.assertRaises(Invalid):
                self.consume(self.packet)
            path.write_bytes(old)

    def test_omitted_or_new_actual_sibling_is_not_accepted_as_a_subset(self):
        identity = json.loads(self.receipt.read_text())
        del identity['libraries_and_binary_sha256'][str(self.library)]
        self.receipt.write_text(json.dumps(identity))
        self.reference['identity_sha256'] = digest(self.receipt.read_bytes())
        with self.assertRaisesRegex(Invalid, 'omits actual sibling'):
            self.consume(self.packet)

    def test_new_unlisted_sibling_refuses_even_when_all_listed_hashes_match(self):
        (self.root / 'unlisted.dylib').write_bytes(b'not in the frozen receipt')
        with self.assertRaisesRegex(Invalid, 'omits actual sibling'):
            self.consume(self.packet)

    def test_supervisor_rejects_a_different_actual_launch_speculation(self):
        actual = list(self.reference['serve_argv'])
        actual[actual.index('--spec-type') + 1] = 'none'
        with self.assertRaisesRegex(CaseError, 'actual supervised'):
            self.issue(self.reference, actual_argv=actual)
        forged = copy.deepcopy(self.packet)
        forged['peer_supervised_launch']['serve_argv'] = actual
        with self.assertRaisesRegex(Invalid, 'actual supervised'):
            self.consume(forged)

    def test_supervisor_rejects_caller_supplied_authoritative_launch_fields(self):
        for template, replacement in (({'peer_supervised_launch': {}}, None),
                                      (None, {'peer_supervised_launch': {}})):
            with self.subTest(template=template), self.assertRaisesRegex(CaseError, 'supervisor-owned'):
                self.issue(self.reference, template_extra=template, set_extra=replacement)

    def test_supervisor_and_consumer_bind_actual_pid_and_endpoint(self):
        for changes in ({'server_pid': 99}, {'base_url': 'http://127.0.0.1:9/v1'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(CaseError, 'actual supervised'):
                self.issue(self.reference, template_extra=changes)
            packet = copy.deepcopy(self.packet)
            packet.update(changes)
            with self.assertRaisesRegex(Invalid, 'PID/endpoint'):
                self.consume(packet)

    def test_opted_in_consumer_refuses_a_missing_supervisor_attestation(self):
        packet = copy.deepcopy(self.packet)
        del packet['peer_supervised_launch']
        with self.assertRaisesRegex(Invalid, 'supervisor-issued'):
            self.consume(packet)

    def test_selected_speculation_must_match_the_frozen_launch(self):
        for selected in ('none', 'draft-simple', 'draft-dflash', True):
            packet = copy.deepcopy(self.packet)
            packet['peer_reference']['selected_speculation'] = selected
            with self.subTest(selected=selected), self.assertRaises(Invalid):
                self.consume(packet)
        self.reference['serve_argv'] += ['--spec-type', 'none']
        with self.assertRaises(Invalid):
            self.consume(self.packet)

    def test_ambient_overrides_cannot_bypass_legacy_or_explicit_identity(self):
        with patch.dict(os.environ, {'HF2Q_PEER_COMMIT': COMMIT, 'PEER_BUILD_INFO': BUILD,
                                     'HF2Q_Q5_PEER_BUILD_INFO': BUILD}):
            with self.assertRaises(Invalid):
                self.consume()
            with self.assertRaises(Invalid):
                self.consume(self.packet, 'b128-e86f5fd')

    def test_legacy_no_binding_retains_historical_identity_and_no_proof_requirement(self):
        result = self.consume(build='b128-e86f5fd')
        self.assertEqual(result['source_revision'], fp.PEER_COMMIT)
        self.assertIsNone(fp.validate_speculation_proof(self.root, {}))
        with self.assertRaises(Invalid):
            self.consume()

    def test_actual_unary_and_sse_response_counters_prove_mtp_exercise(self):
        self.response(stream=False)
        self.response(stream=True)
        proof = fp.validate_speculation_proof(self.root, self.packet)
        self.assertEqual((proof['drafted_tokens'], proof['accepted_tokens']), (18, 12))
        for mode in ('unary', 'sse'):
            self.assertEqual(proof['transports'][mode],
                             {'responses': 1, 'drafted_tokens': 9, 'accepted_tokens': 6})
        self.assertFalse(proof['sampler_metadata_is_launch_selection'])

    def test_loaded_but_unexercised_assistant_is_not_a_valid_mtp_proof(self):
        self.consume(self.packet)
        for stream in (False, True):
            self.response(stream=not stream)
            for drafted, accepted in ((0, 0), (9, 0)):
                self.response(drafted, accepted, stream=stream)
                with self.subTest(stream=stream, drafted=drafted, accepted=accepted), self.assertRaisesRegex(Invalid, 'not exercised'):
                    fp.validate_speculation_proof(self.root, self.packet)

    def test_missing_counters_in_either_transport_cannot_be_hidden_by_the_other(self):
        for stream in (False, True):
            self.response(stream=not stream)
            self.response(stream=stream, missing_timings=True)
            with self.subTest(stream=stream), self.assertRaisesRegex(Invalid, 'lacks draft timing counters'):
                fp.validate_speculation_proof(self.root, self.packet)

    def test_changed_response_bytes_and_impossible_counters_refuse(self):
        turn = self.response()
        (turn / 'response.bin').write_bytes(b'{}')
        with self.assertRaisesRegex(Invalid, 'response proof hash'):
            fp.validate_speculation_proof(self.root, self.packet)
        self.response(3, 4)
        with self.assertRaisesRegex(Invalid, 'counters malformed'):
            fp.validate_speculation_proof(self.root, self.packet)

    def test_declared_no_speculation_rejects_actual_draft_counters(self):
        reference = copy.deepcopy(self.reference)
        reference['selected_speculation'] = 'none'
        reference['serve_argv'][reference['serve_argv'].index('--spec-type') + 1] = 'none'
        self.packet = self.issue(reference)
        self.response()
        with self.assertRaisesRegex(Invalid, 'contradict no speculation'):
            fp.validate_speculation_proof(self.root, self.packet)


if __name__ == '__main__':
    unittest.main()
