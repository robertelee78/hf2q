"""Synthetic failed-Q5 evidence for model-free consumer tests; never campaign inputs."""
import json
from pathlib import Path
from unittest.mock import patch

import q3_campaign
import q3_mechanism as mechanism
import q3_splash_evidence as splash_evidence
import q5_calibration
import native_peer_splash as sp
import native_peer_splash_audit as splash_audit
import splash_sweep

HERE = Path(__file__).resolve().parent
TEMPLATE = b"synthetic reviewed template for model-free oracle-binding tests\n"


def reviewed_template_fixture(test):
    """No real model/template asset is needed; exercise the same pinned-digest contract on synthetic bytes."""
    context = patch.dict(sp.ARTIFACTS["qwen38"], template_sha256=splash_evidence.hashlib.sha256(TEMPLATE).hexdigest())
    context.start()
    test.addCleanup(context.stop)


def rebind_approval(run):
    """Only fixture construction: describe an original launch on a different reviewed source variant."""
    path = run / "receipt.json"
    receipt, approval = json.loads(path.read_text()), json.loads((run / "client-approval.json").read_text())
    argv = receipt["client_argv"]
    argv[argv.index("--approval-sha256") + 1] = mechanism.digest(run / "client-approval.json")
    for name in receipt["inputs"]:
        if Path(name).name in approval["sources"]:
            receipt["inputs"][name] = approval["sources"][Path(name).name]
    path.write_text(json.dumps(receipt))



def invalid_sweep(root):
    """Synthetic real-schema failed Q5 runs, reduced by the actual sweep/audit consumers."""
    root = root / 'invalid-sweep'
    root.mkdir()
    template = root / 'reviewed-template.jinja'
    template.write_bytes(TEMPLATE)
    base = json.loads((HERE / 'q5_cases/splash-qwen38.json').read_text())
    sources = q5_calibration.resolve_set(base['client']['approval']['set']['sources'],
                                       {'REPO': str(HERE.parent.parent)})
    records = []
    for point in sp.PRESETS:
        if point == 'verify-all':
            continue
        for repeat in (1, 2):
            run = root / f'{point}-{repeat}'
            (run / 'client').mkdir(parents=True)
            (run / 'splash').mkdir()
            case = splash_sweep.derive_case(base, point)
            for name in case['variables']:
                if name.endswith('_SHA256'):
                    case['variables'][name] = 'b' * 64
            case['variables'].update(MODEL='/model.gguf', MODEL_SHA256=splash_evidence.ARTIFACT['sha256'],
                                     CARGO='/usr/bin/true', TOOLCHAIN='test',
                                     SANDBOX_PROOF='/synthetic-proof.json')
            path = run / 'case.json'
            path.write_text(json.dumps(case))
            values, _ = q3_campaign.q5_case.resolve_variables(case, q5_calibration.builtins(run), {})
            spec = q3_campaign.q5_case.expand(case['client']['approval']['set']['native_splash'], values)
            spec = splash_evidence.ready.native_spec(spec)
            controls = sp.controls('qwen38', spec['config'])
            approval = {'schema': 'hf2q.qwen-n4-coordinator-approval.v1', 'approved': True,
                        'native_splash': spec, 'family_controls': controls, 'sources': sources,
                        'agent_template': {'fixture_sha256': splash_evidence.FIXTURE_SHA,
                                           'workflow_sha256': splash_evidence.hashlib.sha256(
                                               splash_evidence.WORKFLOW.encode()).hexdigest()}}
            ap = run / 'client-approval.json'
            ap.write_text(json.dumps(approval))
            values.update(SERVER_PID='123', CLIENT_APPROVAL=str(ap), CLIENT_APPROVAL_SHA256=mechanism.digest(ap))
            receipt = {'schema': 'hf2q.q5.calibration-receipt.v1', 'family': 'qwen38',
                       'case': {'path': str(path), 'sha256': mechanism.digest(path)},
                       'artifact': splash_evidence.ARTIFACT, 'binary': {'sha256': spec['binary_sha256']},
                       'workload': case['workload'], 'cleanup_proven': True,
                       'inputs': {str(HERE / 'qwen_n4_live_smoke.py'): sources['qwen_n4_live_smoke.py']},
                       'verdict': {'verdict': 'ClientFailed', 'reason': 'synthetic failed client'},
                       'client_argv': q3_campaign.q5_case.expand(case['client']['argv'], values)}
            (run / 'receipt.json').write_text(json.dumps(receipt))
            source = {k: spec[k] for k in ('binary_sha256', 'metallib_sha256', 'host_step_sha256')}
            source.update(commit=sp.SPLASH_COMMIT, build_id=sp.BUILD_ID,
                          toolchain={'sha256': spec['toolchain_sha256']})
            identity = {'source': source, 'model': splash_evidence.ARTIFACT,
                        'venv': {'freeze_sha256': spec['venv_freeze_sha256']},
                        'draft': {'revision': sp.ARTIFACTS['qwen38']['draft_revision'],
                                  'config_sha256': sp.ARTIFACTS['qwen38']['draft_config_sha256'],
                                  'weights_sha256': sp.ARTIFACTS['qwen38']['draft_weights_sha256']}}
            summary = {'engine_kind': 'splash', 'status': 'Failed', 'agents': {},
                       'run_id': f'synthetic-splash-{point.replace(".", "_")}-{repeat}',
                       'native_readiness': {'q3_family': 'qwen38', 'source_revision': sp.SPLASH_COMMIT,
                                            'build_id': sp.BUILD_ID, 'config': spec['config'],
                                            'controls': controls, 'identity': identity,
                                            'chat_template': {'hf2q_reviewed_sha256': sp.ARTIFACTS['qwen38']['template_sha256']}}}
            fixture = json.loads((HERE / 'fixtures/canonical-coding-requests.json').read_text())[0]['request']
            for tool in fixture['tools']:
                tool['function']['strict'] = False
            for agent in splash_evidence.n4.AGENTS:
                directory = run / 'client' / agent
                (directory / 'turn-0').mkdir(parents=True)
                prompt_identity = splash_evidence.n4.identity(agent, summary['run_id'])
                messages = splash_evidence.n1.initial_messages(fixture['messages'], prompt_identity)
                request = splash_evidence.n1.build_request(sp.MODEL_IDS['qwen38'], messages, fixture['tools'],
                                                          0, 'qwen', 'splash')
                (directory / 'turn-0/request.json').write_text(json.dumps(request))
                (directory / 'summary.json').write_text(json.dumps({
                    'initial_identity_sha256': splash_evidence.hashlib.sha256(prompt_identity.encode()).hexdigest(),
                    'initial_messages_sha256': splash_evidence.hashlib.sha256(
                        json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}))
            (run / 'client/summary.json').write_text(json.dumps(summary))
            (run / 'client/response.raw').write_text('synthetic failed response')
            for name in ('server.stdout', 'server.stderr'):
                (run / name).write_text('synthetic failed run')
            (run / 'splash/launch.json').write_text(json.dumps({'spec': spec}))
            parity = run / splash_audit.PARITY_DIR
            parity.mkdir()
            config, result_path = parity / 'config.json', parity / 'splash-prompt-parity-post-oracle.json'
            turns, observations = [], []
            for agent in splash_evidence.n4.AGENTS:
                raw_directory = run / 'client' / agent / 'turn-0'
                request = json.loads((raw_directory / 'request.json').read_text())
                canonical = parity / f'{agent}-turn-0.canonical-request.json'
                canonical.write_text(json.dumps(sp.canonical(request, 'qwen')))
                rendered = raw_directory / 'splash-apply-template.raw'
                prompt = 'synthetic bound rendered prompt'
                rendered.write_text(json.dumps({'prompt': prompt}))
                turns.append({'agent': agent, 'turn': 0,
                              'canonical_request': {'path': str(canonical), 'sha256': mechanism.digest(canonical)},
                              'splash_apply_template': {'path': str(rendered), 'sha256': mechanism.digest(rendered)}})
                observations.append({'agent': agent, 'turn': 0, 'canonical_request_sha256': mechanism.digest(canonical),
                                     'splash_apply_template_sha256': mechanism.digest(rendered),
                                     'prompt_sha256': splash_evidence.hashlib.sha256(prompt.encode()).hexdigest(),
                                     'prompt_bytes': len(prompt.encode()), 'equal': True})
            config.write_text(json.dumps({'architecture': 'qwen35', 'template': {'path': str(template),
                                          'sha256': mechanism.digest(template)}, 'turns': turns, 'output': str(result_path)}))
            result_path.write_text(json.dumps({
                'schema': 'hf2q.splash-prompt-parity-post-oracle.v1', 'hf2q_prompt_parity': 'PassedByteIdentical',
                'model_inference_performed': False, 'architecture': 'qwen35', 'config_sha256': mechanism.digest(config),
                'template_file_sha256': mechanism.digest(template), 'observations': observations,
                **{key: mechanism.digest(path) for key, path in splash_evidence.ORACLE_SOURCES.items()}}))
            records.append(splash_audit.run_record(run))
    path = root / 'selection.json'
    path.write_text(json.dumps(splash_sweep.select(records)))
    return path


def mechanism_campaign(root, rows=663, lanes=125):
    from test_native_peer_splash import splash_manifest
    m = splash_manifest(arms=('stock', 'experiment', 'peer'))
    artifact = m['identities']['arms']['experiment']['artifact']
    m['identities']['arms'].update({
        'stock': {'binary': {'sha256': '5' * 64}, 'artifact': dict(artifact)},
        'peer': {'binary': {'sha256': '9' * 64}, 'artifact': dict(artifact)}})
    p = m['plan']
    p['splash_required'] = False
    selection = invalid_sweep(root)
    p['splash_unavailable_evidence'] = [{'path': str(selection), 'sha256': mechanism.digest(selection)}]
    p['qwen38_mechanism'] = dict(mechanism.CONTRACT)
    p['design'].update(stock_margin_ratio=.95, peer_margin_ratio=1.)
    m['identities']['harness'] = {'git_commit': p['declared']['experiment']['commit']}
    case = json.loads((HERE / 'q5_cases/qwen38.json').read_text())
    p['qwen38_splash_substitution'] = splash_evidence.freeze(p['splash_unavailable_evidence'], case)
    for arm in m['identities']['arms'].values():
        arm['artifact'].update(splash_evidence.ARTIFACT)
    base = root / 'case.json'
    base.write_text(json.dumps(case))
    p['arms']['experiment'].update(case=str(base), case_sha256=mechanism.digest(base))
    for block in m['stages'][0]['blocks']:
        for entry in block['entries']:
            if entry['arm'] != 'experiment':
                continue
            output = root / entry['name']
            output.mkdir()
            values = [rows // lanes + (i < rows % lanes) for i in range(lanes)]
            text = ''.join(f'DEBUG progress: Qwen DFlash2 verified window slot=0 drafted=7 accepted={n}\n'
                           for n in values)
            (output / 'server.stderr').write_text(text)
            receipt = {'schema': 'hf2q.q5.calibration-receipt.v1', 'family': 'qwen38',
                       'binary': {'sha256': 'e' * 64}, 'artifact': m['identities']['arms']['experiment']['artifact'],
                       'workload': case['workload'], 'case': {'path': str(base), 'sha256': mechanism.digest(base)},
                       'verdict': {'verdict': 'CalibrationPass', 'reason': None}, 'cleanup_proven': True}
            (output / 'receipt.json').write_text(json.dumps(receipt))
            entry['attempts'] = [{'output': str(output), 'case': str(base), 'outcome': 'Pass',
                                 'receipt_sha256': mechanism.digest(output / 'receipt.json'),
                                 'server_stderr_sha256': mechanism.digest(output / 'server.stderr')}]
    return m
