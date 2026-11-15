"""Oracle-record consumer boundaries; synthetic archives, no renderer or model execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import native_peer_splash as sp
import native_peer_splash_audit as audit
import q3_splash_evidence
from qwen38_n1_smoke import build_request
from qwen38_n1_verdict import Invalid


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def bound(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write_oracle_fixture(root, family='qwen38', keys=(('agent-0', 0), ('agent-0', 1))):
    """A complete Rust-result-shaped fixture, not a claim that an oracle ran."""
    run = Path(root).resolve() / 'run'
    parity = run / audit.PARITY_DIR
    parity.mkdir(parents=True)
    template = run / 'reviewed-template.jinja'
    template.write_bytes(b'reviewed synthetic template\n')
    turns, observations = [], []
    for agent, turn in keys:
        directory = run / 'client' / agent / f'turn-{turn}'
        request = build_request(sp.MODEL_IDS[family], [{'role': 'user', 'content': f'actual {agent} {turn}: λ'}],
                                [{'type': 'function', 'function': {'name': 'read_file', 'strict': False,
                                  'parameters': {'type': 'object', 'properties': {}}}}],
                                turn, 'qwen', 'splash')
        save(directory / 'request.json', request)
        save(directory / 'response-headers.json', {'request_sha256': bound(directory / 'request.json')['sha256']})
        canonical = parity / f'{agent}-turn-{turn}.canonical-request.json'
        save(canonical, sp.canonical(request, 'qwen'))
        rendered = directory / 'splash-apply-template.raw'
        prompt = f'archived native prompt {agent} {turn}: λ'
        save(rendered, {'prompt': prompt})
        turns.append({'agent': agent, 'turn': turn,
                      'canonical_request': bound(canonical), 'splash_apply_template': bound(rendered)})
        observations.append({'agent': agent, 'turn': turn,
                             'canonical_request_sha256': bound(canonical)['sha256'],
                             'splash_apply_template_sha256': bound(rendered)['sha256'],
                             'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                             'prompt_bytes': len(prompt.encode()), 'equal': True})
    result_path = parity / 'splash-prompt-parity-post-oracle.json'
    architecture = 'qwen35' if family == 'qwen38' else 'qwen35moe'
    config = {'architecture': architecture, 'template': bound(template), 'turns': turns, 'output': str(result_path)}
    config_path = parity / 'config.json'
    save(config_path, config)
    sources = {name + '_source_sha256': bound(Path(__file__).resolve().parents[2] / 'src/serve/api' / filename)['sha256']
               for name, filename in (('oracle', 'handlers_splash_prompt_parity_post_oracle_tests.rs'),
                                      ('handler', 'handlers.rs'), ('renderer', 'engine.rs'))}
    result = {'schema': 'hf2q.splash-prompt-parity-post-oracle.v1',
              'hf2q_prompt_parity': 'PassedByteIdentical', 'model_inference_performed': False,
              'architecture': architecture, 'config_sha256': bound(config_path)['sha256'],
              'template_file_sha256': bound(template)['sha256'], 'observations': observations, **sources}
    save(result_path, result)
    return {'run': run, 'config_path': config_path, 'result_path': result_path, 'template': template,
            'family': family, 'config': config, 'result': result, 'sources': sources}


class SplashParityCoverage(unittest.TestCase):
    def test_complete_partial_archives_bind_every_input_but_not_client_validity(self):
        for family in ('qwen36', 'qwen38'):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                fixture = write_oracle_fixture(root, family)
                with mock.patch.dict(sp.ARTIFACTS[family], {'template_sha256': bound(fixture['template'])['sha256']}):
                    self.assertEqual(audit.prompt_parity(fixture['run']), 'PassedByteIdentical')
                    inputs = audit.parity_inputs(fixture['run'])
                    for turn in (0, 1):
                        directory = fixture['run'] / 'client' / 'agent-0' / f'turn-{turn}'
                        self.assertTrue({directory / 'request.json', directory / 'response-headers.json',
                                         directory / 'splash-apply-template.raw'} <= inputs)
                    self.assertFalse((fixture['run'] / 'client' / 'summary.json').exists())

    def test_config_and_result_cannot_omit_failed_or_later_added_requests(self):
        for failure in ('omitted', 'added', 'duplicate'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                fixture = write_oracle_fixture(root)
                config, result = fixture['config'], fixture['result']
                if failure == 'omitted':
                    config['turns'].pop()
                    result['observations'].pop()
                elif failure == 'duplicate':
                    config['turns'][1] = copy.deepcopy(config['turns'][0])
                    result['observations'][1] = copy.deepcopy(result['observations'][0])
                else:
                    extra = fixture['run'] / 'client' / 'agent-1' / 'turn-0' / 'request.json'
                    save(extra, json.loads((fixture['run'] / 'client' / 'agent-0' / 'turn-0' / 'request.json').read_bytes()))
                save(fixture['config_path'], config)
                result['config_sha256'] = bound(fixture['config_path'])['sha256']
                save(fixture['result_path'], result)
                with mock.patch.dict(sp.ARTIFACTS['qwen38'], {'template_sha256': bound(fixture['template'])['sha256']}):
                    self.assertNotEqual(audit.prompt_parity(fixture['run']), 'PassedByteIdentical')

    def test_changed_actual_request_or_digest_record_invalidates_complete_oracle(self):
        for failure in ('request', 'headers', 'receipt'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                fixture = write_oracle_fixture(root)
                directory = fixture['run'] / 'client' / 'agent-0' / 'turn-1'
                if failure == 'request':
                    path = directory / 'request.json'
                    request = json.loads(path.read_bytes())
                    request['messages'][0]['content'] = 'changed actual coding request'
                    save(path, request)
                else:
                    save(directory / ('response-headers.json' if failure == 'headers' else 'receipt.json'),
                         {'request_sha256': '0' * 64})
                with mock.patch.dict(sp.ARTIFACTS['qwen38'], {'template_sha256': bound(fixture['template'])['sha256']}):
                    self.assertNotEqual(audit.prompt_parity(fixture['run']), 'PassedByteIdentical')

    def test_unbound_rendering_template_observation_or_oracle_source_is_rejected(self):
        for failure in ('render', 'template', 'observation', 'observation-count', 'source', 'inference', 'config'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                fixture = write_oracle_fixture(root)
                reviewed = bound(fixture['template'])['sha256']
                result = fixture['result']
                if failure == 'render':
                    save(fixture['run'] / 'client' / 'agent-0' / 'turn-1' / 'splash-apply-template.raw', {'prompt': 'changed'})
                elif failure == 'template':
                    fixture['template'].write_text('changed template')
                elif failure == 'observation':
                    result['observations'][1]['prompt_bytes'] += 1
                elif failure == 'observation-count':
                    result['observations'].pop()
                elif failure == 'source':
                    result['renderer_source_sha256'] = '0' * 64
                elif failure == 'inference':
                    result['model_inference_performed'] = True
                else:
                    result['config_sha256'] = '0' * 64
                save(fixture['result_path'], result)
                with mock.patch.dict(sp.ARTIFACTS['qwen38'], {'template_sha256': reviewed}):
                    self.assertNotEqual(audit.prompt_parity(fixture['run']), 'PassedByteIdentical')

    def test_qwen38_exemption_still_requires_all_four_initial_requests(self):
        for count in (4, 3):
            with self.subTest(agents=count), tempfile.TemporaryDirectory() as root:
                fixture = write_oracle_fixture(root, keys=tuple((f'agent-{n}', 0) for n in range(count)))
                with mock.patch.dict(sp.ARTIFACTS['qwen38'], {'template_sha256': bound(fixture['template'])['sha256']}):
                    self.assertEqual(audit.prompt_parity(fixture['run']), 'PassedByteIdentical')
                    if count == 4:
                        inputs = q3_splash_evidence.parity_inputs(fixture['run'], fixture['sources'])
                        self.assertIn(fixture['config_path'], inputs)
                    else:
                        with self.assertRaises(Invalid):
                            q3_splash_evidence.parity_inputs(fixture['run'], fixture['sources'])

    def test_orphan_receipt_or_response_headers_cannot_hide_a_missing_actual_request(self):
        for retained in ('receipt.json', 'response-headers.json'):
            with self.subTest(retained=retained), tempfile.TemporaryDirectory() as root:
                fixture = write_oracle_fixture(root)
                directory = fixture['run'] / 'client' / 'agent-0' / 'turn-1'
                raw_sha = bound(directory / 'request.json')['sha256']
                (directory / 'request.json').unlink()
                (directory / 'response-headers.json').unlink()
                save(directory / retained, {'request_sha256': raw_sha})
                fixture['config']['turns'].pop()
                fixture['result']['observations'].pop()
                save(fixture['config_path'], fixture['config'])
                fixture['result']['config_sha256'] = bound(fixture['config_path'])['sha256']
                save(fixture['result_path'], fixture['result'])
                with mock.patch.dict(sp.ARTIFACTS['qwen38'], {'template_sha256': bound(fixture['template'])['sha256']}):
                    self.assertNotEqual(audit.prompt_parity(fixture['run']), 'PassedByteIdentical')


if __name__ == '__main__':
    unittest.main()
