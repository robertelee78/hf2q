"""Offline actual-request parity inputs do not turn an interrupted client into a valid run."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import native_peer_splash as sp
import native_peer_splash_audit as audit
import native_peer_splash_ready as ready
import splash_sweep
from qwen38_n1_smoke import build_request
from qwen38_n1_verdict import Invalid


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


class SplashSummarylessParity(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run = self.root / 'run'
        self.client = self.run / 'client'
        self.inputs_path = self.client / ready.CLOSING_INPUTS
        self.launch_path = self.run / 'splash' / 'launch.json'
        self.template = self.root / 'reviewed.jinja'
        self.template.write_bytes(b'reviewed test template\n')

    def fixture(self, q3='qwen36'):
        artifact = sp.ARTIFACTS[q3]
        patch = mock.patch.dict(artifact, {'template_sha256': sha(self.template.read_bytes())})
        patch.start()
        self.addCleanup(patch.stop)
        versions = {key: 'archived ' + key for key in ready.TOOLCHAIN}
        spec = {
            'q3_family': q3, 'splash_dir': '/absent/offline-splash', 'commit': sp.SPLASH_COMMIT,
            'build_id': sp.BUILD_ID, 'binary_sha256': 'a' * 64, 'metallib_sha256': 'b' * 64,
            'host_step_sha256': 'c' * 64, 'toolchain_sha256': sha(json.dumps(versions, sort_keys=True).encode()),
            'venv_freeze_sha256': 'e' * 64, 'model': '/absent/target.gguf', 'model_sha256': 'f' * 64,
            'draft_dir': '/absent/hub/models--' + artifact['draft_repo'].replace('/', '--')
                         + '/snapshots/' + artifact['draft_revision'],
            'config': sp.preset('s0', q3), 'run_dir': str(self.run / 'splash'), 'port': 18257,
            'server_stdout': str(self.run / 'server.stdout'), 'server_stderr': str(self.run / 'server.stderr'),
        }
        state = {'ane_dir': '/absent/ane', 'ane_remembered': {}, 'prefix_cache_namespaces': [],
                 'draft_pin_dir': str(ready.pin_dir(spec['draft_dir'])), 'draft_pins': []}
        identity = {
            'source': {'commit': sp.SPLASH_COMMIT, 'describe': sp.SPLASH_DESCRIBE, 'build_id': sp.BUILD_ID,
                       'binary_sha256': spec['binary_sha256'], 'metallib_sha256': spec['metallib_sha256'],
                       'host_step_sha256': spec['host_step_sha256'], 'requirements_sha256': sp.REQUIREMENTS_SHA256,
                       'toolchain': {'versions': versions, 'sha256': spec['toolchain_sha256']},
                       'tracked_changes': 'none (git status --untracked-files=no)'},
            'venv': {'venv': spec['splash_dir'] + '/.venv', 'packages': 12, 'pins': 10,
                     'freeze_sha256': spec['venv_freeze_sha256']},
            'draft': {'repo': artifact['draft_repo'], 'revision': artifact['draft_revision'],
                      'path': spec['draft_dir'], 'config_sha256': artifact['draft_config_sha256'],
                      'weights_sha256': artifact['draft_weights_sha256'], 'weights_bytes': 100},
            'model': {'path': spec['model'], 'bytes': 200, 'sha256': spec['model_sha256'],
                      'sha256_source': 'Splash model.json content digest (verify_assembly)'},
        }
        assembly = {'path': str(self.run / 'splash' / 'models' / '.resolved' / 'target'),
                    'model_json_sha256': '1' * 64, 'family': artifact['splash_family'],
                    'target_digest': spec['model_sha256'], 'draft_digest': artifact['draft_weights_sha256'],
                    'template_digest': artifact['template_sha256'],
                    'linked_files': ['target/target.gguf', 'draft/model.safetensors', 'tokenizer/chat_template.jinja']}
        # Use the production launch-record convention without consulting host state or loading a model.
        with mock.patch.object(ready, 'cross_run_state', return_value=state), mock.patch.object(ready.os, 'getpid', return_value=321):
            self.launch = ready.launch_record({k: spec[k] for k in ready.LAUNCH_KEYS}, identity, assembly, ready.SERVE_ENV)
        self.inputs = {'spec': spec, 'server_pid': 321, 'state_before': state}
        save(self.inputs_path, self.inputs)
        save(self.launch_path, self.launch)
        self.q3 = q3
        self.messages = [{'role': 'system', 'content': 'unchanged'}, {'role': 'user', 'content': 'actual task ☃'}]
        self.schemas = [{'type': 'function', 'function': {'name': 'read_file', 'strict': False,
                         'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}}}}]
        self.directories = []
        self.receipts = []
        for agent, turn in (('agent-0', 0), ('agent-0', 1), ('agent-1', 0)):
            directory = self.client / agent / f'turn-{turn}'
            request = build_request(sp.MODEL_IDS[q3], self.messages, self.schemas, turn, 'qwen', 'splash')
            save(directory / 'request.json', request)
            raw = (directory / 'request.json').read_bytes()
            (directory / 'splash-apply-template.raw').write_bytes(json.dumps({'prompt': f'{agent}/{turn} ☃'}).encode())
            receipt = {'turn': turn, 'request_sha256': sha(raw), 'request_bytes': len(raw)}
            if agent == 'agent-0':
                response = b'archived response ' + str(turn).encode()
                (directory / 'response.bin').write_bytes(response)
                save(directory / 'response-headers.json', receipt)
                receipt = {**receipt, 'response_sha256': sha(response), 'response_bytes': len(response)}
                save(directory / 'receipt.json', receipt)
            self.directories.append(directory)
            self.receipts.append(receipt)
        # Only the first turn reached the agent's validated-receipt list.
        save(self.client / 'agent-0' / 'summary.json', {'turns': [self.receipts[0]]})

    def generate(self, name='parity'):
        return splash_sweep.parity_config(self.run, self.template, self.root / name)

    def present_summary(self, **changes):
        summary = {'engine_kind': sp.ENGINE, 'native_readiness': {'q3_family': self.q3},
                   'status': 'Failed', 'agents': {'agent-0': {'turns': [self.receipts[0]]}}}
        save(self.client / 'summary.json', {**summary, **changes})

    def test_summaryless_covers_validated_failed_and_interrupted_actual_requests(self):
        for q3, architecture in (('qwen36', 'qwen35moe'), ('qwen38', 'qwen35')):
            with self.subTest(q3=q3):
                self.fixture(q3)
                forbidden = AssertionError('offline generation must not inspect runtime identity')
                with mock.patch.object(ready, 'verify_identity', side_effect=forbidden), \
                        mock.patch.object(ready, 'verify_launch', side_effect=forbidden), \
                        mock.patch.object(ready.subprocess, 'run', side_effect=forbidden), \
                        mock.patch.object(ready, 'cross_run_state', side_effect=forbidden), \
                        mock.patch.object(ready, 'fetch', side_effect=forbidden):
                    config = json.loads(self.generate(q3).read_bytes())
                self.assertEqual(set(config), {'architecture', 'template', 'turns', 'output'})
                self.assertEqual(config['architecture'], architecture)
                self.assertEqual(config['template'], {'path': str(self.template), 'sha256': sha(self.template.read_bytes())})
                self.assertEqual([(r['agent'], r['turn']) for r in config['turns']],
                                 [('agent-0', 0), ('agent-0', 1), ('agent-1', 0)])
                for row, directory in zip(config['turns'], self.directories):
                    path = Path(row['canonical_request']['path'])
                    self.assertEqual(json.loads(path.read_bytes()), build_request(
                        sp.MODEL_IDS[q3], self.messages, self.schemas, row['turn'], 'qwen'))
                    self.assertEqual(row['canonical_request']['sha256'], sha(path.read_bytes()))
                    self.assertEqual(row['splash_apply_template'], {
                        'path': str(directory / 'splash-apply-template.raw'),
                        'sha256': sha((directory / 'splash-apply-template.raw').read_bytes())})
                self.assertFalse((self.client / 'summary.json').exists())
                record = audit.run_record(self.run)
                self.assertNotEqual(record['audit'], 'Pass')
                self.assertIsNone(splash_sweep.select([record])['choice'])

    def test_missing_or_malformed_archived_provenance_is_fatal(self):
        self.fixture()
        for path in (self.inputs_path, self.launch_path):
            original = path.read_bytes()
            for raw in (None, b'{', b'[]', b'{}', b'{"schema":NaN}', b'{"spec":{},"spec":{}}'):
                with self.subTest(path=path.name, raw=raw):
                    if raw is None:
                        path.unlink()
                    else:
                        path.write_bytes(raw)
                    with self.assertRaises((Invalid, FileNotFoundError, KeyError, TypeError, ValueError)):
                        self.generate()
                    path.write_bytes(original)

    def test_conflicting_launch_readiness_identity_and_paths_are_fatal(self):
        self.fixture()
        cases = [
            ('inputs', ('server_pid',), 0), ('inputs', ('server_pid',), True),
            ('inputs', ('spec', 'q3_family'), 'unknown'), ('inputs', ('spec', 'commit'), '0' * 40),
            ('inputs', ('spec', 'extra'), 'not approved'),
            ('inputs', ('spec', 'run_dir'), str(self.root / 'another' / 'splash')),
            ('inputs', ('spec', 'server_stdout'), '/elsewhere/server.stdout'),
            ('inputs', ('state_before',), {}), ('inputs', ('state_before', 'ane_remembered'), {'x': 'bad'}),
            ('launch', ('schema',), 'unknown'), ('launch', ('pid',), 322), ('launch', ('pid',), True),
            ('launch', ('spec', 'model'), '/another.gguf'), ('launch', ('spec', 'q3_family'), 'qwen38'),
            ('launch', ('spec', 'port'), '18257'), ('launch', ('state_before', 'draft_pins'), ['unexpected']),
            ('launch', ('identity',), None), ('launch', ('identity', 'source', 'commit'), '0' * 40),
            ('launch', ('identity', 'source', 'binary_sha256'), '0' * 64),
            ('launch', ('identity', 'source', 'toolchain', 'sha256'), '0' * 64),
            ('launch', ('identity', 'source', 'toolchain', 'versions', 'metal'), 'changed'),
            ('launch', ('identity', 'venv', 'freeze_sha256'), '0' * 64),
            ('launch', ('identity', 'venv', 'venv'), '/another/.venv'),
            ('launch', ('identity', 'draft', 'revision'), '0' * 40),
            ('launch', ('identity', 'draft', 'path'), '/another/snapshot'),
            ('launch', ('identity', 'model', 'path'), '/another.gguf'),
            ('launch', ('identity', 'model', 'sha256'), '0' * 64),
            ('launch', ('assembly', 'path'), str(self.root / 'models' / '.resolved' / 'target')),
            ('launch', ('assembly', 'family'), 'Qwen3.8-27B'),
            ('launch', ('assembly', 'target_digest'), '0' * 64),
            ('launch', ('assembly', 'draft_digest'), '0' * 64),
            ('launch', ('assembly', 'template_digest'), '0' * 64),
            ('launch', ('assembly', 'model_json_sha256'), 'bad'),
            ('launch', ('assembly', 'linked_files'), []), ('launch', ('argv',), []),
            ('launch', ('env', 'PYTHONPATH'), '/another'), ('launch', ('env', 'HF_HUB_OFFLINE'), '0'),
        ]
        for source, keys, value in cases:
            with self.subTest(source=source, keys=keys):
                document = copy.deepcopy(self.inputs if source == 'inputs' else self.launch)
                node = document
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                path = self.inputs_path if source == 'inputs' else self.launch_path
                save(path, document)
                with self.assertRaises((Invalid, KeyError, TypeError, ValueError)):
                    self.generate()
                save(path, self.inputs if source == 'inputs' else self.launch)

    def test_present_corrupt_client_summary_never_uses_archived_fallback(self):
        self.fixture()
        path = self.client / 'summary.json'
        for raw in (b'{', b'[]', b'null', b'{}', b'{"engine_kind":"splash","engine_kind":"splash"}'):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaises((Invalid, ValueError)):
                    self.generate()
        for index, changes in enumerate(({'engine_kind': 'hf2q'}, {'native_readiness': None},
                                        {'native_readiness': {'q3_family': 'unknown'}}, {'agents': []}, {'agents': None})):
            with self.subTest(changes=changes):
                self.present_summary(**changes)
                with self.assertRaises(Invalid):
                    self.generate(f'invalid-summary-{index}')

    def test_pinned_family_model_and_closed_wire_controls_are_required(self):
        self.fixture()
        # Change only an interrupted request, so its lack of receipts cannot conceal control drift.
        path = self.directories[-1] / 'request.json'
        original = json.loads(path.read_bytes())
        for key, value in (('model', sp.MODEL_IDS['qwen38']), ('model', 'unknown'),
                           ('temperature', 1), ('preserve_thinking', False), ('thinking_token_budget', 512),
                           ('max_completion_tokens', 10), ('max_completion_tokens', True),
                           ('tool_choice', 'none'), ('logprobs', True), ('stream', 0),
                           ('messages', [{'role': 'unknown', 'content': 'invalid'}]), ('extra', 'unsupported')):
            with self.subTest(key=key, value=value):
                save(path, {**original, key: value})
                with self.assertRaises(Invalid):
                    self.generate('invalid-' + key + '-' + str(len(list(self.root.iterdir()))))
        for raw in (b'{', b'[]', b'{"model":NaN}', b'{"model":"x","model":"y"}'):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaises((Invalid, ValueError)):
                    self.generate('invalid-json-' + str(len(list(self.root.iterdir()))))

    def test_reviewed_template_and_all_request_response_digest_records_are_required(self):
        self.fixture()
        self.template.write_bytes(b'not reviewed')
        with self.assertRaises(Invalid):
            self.generate()
        self.template.write_bytes(b'reviewed test template\n')
        index = 0
        for directory in self.directories[:2]:
            for name in ('response-headers.json', 'receipt.json'):
                path = directory / name
                original = json.loads(path.read_bytes())
                for change in ([], {'request_sha256': 'bad'}, {**original, 'turn': 99},
                               {**original, 'request_bytes': 0}, {**original, 'response_sha256': '0' * 64},
                               {**original, 'response_bytes': 0}):
                    with self.subTest(directory=directory.name, name=name, change=change):
                        save(path, change)
                        with self.assertRaises(Invalid):
                            self.generate(f'invalid-record-{index}')
                        index += 1
                        save(path, original)
        for directory in self.directories[:2]:
            for name in ('request.json', 'response.bin'):
                path = directory / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                with self.subTest(directory=directory.name, name=name), self.assertRaises(Invalid):
                    self.generate(f'changed-archive-{index}')
                index += 1
                path.write_bytes(original)

    def test_receipt_sources_cannot_hide_missing_duplicate_or_conflicting_validated_turns(self):
        self.fixture()
        path = self.client / 'agent-0' / 'summary.json'
        original = {'turns': [self.receipts[0]]}
        self.present_summary()
        cases = [None, [], {}, {'turns': None}, {'turns': [None]}, {'turns': [self.receipts[0]] * 2},
                 {'turns': [{**self.receipts[0], 'turn': True}]},
                 {'turns': [{**self.receipts[0], 'request_sha256': '0' * 64}]},
                 {'turns': [{**self.receipts[0], 'turn': 7}]}]
        for index, value in enumerate(cases):
            with self.subTest(value=value):
                save(path, value)
                with self.assertRaises(Invalid):
                    self.generate(f'bad-agent-{index}')
                save(path, original)
        path.unlink()
        config = json.loads(self.generate('embedded').read_bytes())
        self.assertEqual(len(config['turns']), 3)
        self.present_summary(agents={'agent-0': {'turns': [{**self.receipts[0], 'turn': 7}]}})
        with self.assertRaises(Invalid):
            self.generate('missing-validated-archive')

    def test_standalone_response_records_require_their_actual_request(self):
        self.fixture()
        directory = self.directories[1]
        request = directory / 'request.json'
        raw = request.read_bytes()
        records = {name: (directory / name).read_bytes()
                   for name in ('receipt.json', 'response-headers.json')}
        for retained in records:
            with self.subTest(retained=retained):
                request.unlink()
                for name in records:
                    if name != retained:
                        (directory / name).unlink()
                with self.assertRaises(Invalid):
                    self.generate('orphan-' + retained)
                request.write_bytes(raw)
                for name, data in records.items():
                    (directory / name).write_bytes(data)

    def test_genuine_lexically_normalized_draft_identity_accepts_trailing_slash(self):
        self.fixture()
        self.inputs['spec']['draft_dir'] += '/'
        self.launch['spec']['draft_dir'] += '/'
        save(self.inputs_path, self.inputs)
        save(self.launch_path, self.launch)
        config = json.loads(self.generate('normalized-draft').read_bytes())
        self.assertEqual([(row['agent'], row['turn']) for row in config['turns']],
                         [('agent-0', 0), ('agent-0', 1), ('agent-1', 0)])

    def test_missing_rendering_and_noncanonical_archives_are_fatal(self):
        self.fixture()
        path = self.directories[-1] / 'splash-apply-template.raw'
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.generate('missing-render')
        path.write_bytes(b'{"prompt":"actual"}')
        directory = self.directories[-1]
        directory.rename(directory.with_name('turn-00'))
        with self.assertRaises(Invalid):
            self.generate('noncanonical-turn')
        directory.with_name('turn-00').rename(directory)
        agent = directory.parent
        agent.rename(agent.with_name('agent-01'))
        with self.assertRaises(Invalid):
            self.generate('noncanonical-agent')


if __name__ == '__main__':
    unittest.main()
