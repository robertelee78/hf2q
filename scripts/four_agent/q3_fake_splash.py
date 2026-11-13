"""Synthetic failed-Q5 evidence for model-free consumer tests; never campaign inputs."""
import json
from pathlib import Path

import q3_campaign
import q3_mechanism as mechanism
import q3_splash_evidence as splash_evidence
import q5_calibration
import native_peer_splash as sp
import native_peer_splash_audit as splash_audit
import splash_sweep

HERE = Path(__file__).resolve().parent


def invalid_sweep(root):
    """Synthetic real-schema failed Q5 runs, reduced by the actual sweep/audit consumers."""
    root = root / 'invalid-sweep'
    base = json.loads((HERE / 'q5_cases/splash-qwen38.json').read_text())
    expected = json.loads((HERE / 'q5_cases/qwen38.json').read_text())
    sources = {str(HERE / Path(p).name): mechanism.digest(HERE / Path(p).name)
               for p in expected['client']['approval']['set']['sources']['sha256_of']
               if not Path(p).name.startswith('test_')}
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
            config = parity / 'config.json'
            config.write_text(json.dumps({'turns': [{'splash_apply_template': {
                'path': str(run / 'client/agent-0/turn-0/splash-apply-template.raw')}}]}))
            (parity / 'splash-prompt-parity-post-oracle.json').write_text(json.dumps({
                'schema': 'hf2q.splash-prompt-parity-post-oracle.v1',
                'hf2q_prompt_parity': 'PassedByteIdentical', 'config_sha256': mechanism.digest(config)}))
            records.append(splash_audit.run_record(run))
    path = root / 'selection.json'
    path.write_text(json.dumps(splash_sweep.select(records)))
    return path
