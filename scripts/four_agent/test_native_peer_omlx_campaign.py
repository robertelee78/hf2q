"""Pure controls for the oMLX Q5 cases, the Q3 native-comparison campaign, F2 rows and the post-run audit.

No launch, model or oMLX process: cases are planned with q5_case.build_plan, the Q3 verdict/rows come from a
hand-built campaign manifest, and the audit reads a hand-built run directory whose server.stderr carries oMLX's
`Chat completion:` lines (server.py:5337-5343 unary, 7227-7236 stream) and tool-parser warnings.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import f2_scoreboard_render as f2
import native_peer_omlx as om
import native_peer_omlx_audit as audit
import native_peer_omlx_ready as ready
import q3_campaign as q3
import q3_identity
import q3_report
import q5_case
from qwen38_n1_verdict import Invalid
from qwen_n4_live_smoke import MODULES
from test_native_peer_omlx import CALL, MODEL, call_message, completion_body, stream, use
from test_native_peer_tensorfold_campaign import metrics

HERE = Path(__file__).resolve().parent
CASES = {q3f: HERE / 'q5_cases' / f'native-omlx-{q3f}.json' for q3f in om.Q3_FAMILIES}
PORTS = {'qwen38': '18254', 'qwen36': '18255', 'gemma4': '18256'}
TITLES = {'qwen38': 'Qwen3.8 27B', 'qwen36': 'Qwen3.6 35B-A3B', 'gemma4': 'Gemma 4 26B-A4B'}


class Case(unittest.TestCase):
    BUILTINS = {'REPO': str(HERE.parent.parent), 'OUT': '/out/run', 'PYTHON': '/usr/bin/python3',
                'PYTHON_SHA256': 'a' * 64, 'RUN_ID': 'q5-run-0123456789', 'NONCE32': 'b' * 32}
    REQUIRED = {'HF2Q_Q5_CARGO': '/c/cargo', 'HF2Q_Q5_TOOLCHAIN': '/c/tc', 'HF2Q_Q5_SANDBOX_PROOF': '/c/proof.json',
                'HF2Q_Q5_SANDBOX_PROOF_SHA256': 'c' * 64, 'HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE': '/c/approval.json'}
    DIGESTS = {'HF2Q_Q5_MODEL_CONFIG_SHA256': 'e' * 64, 'HF2Q_Q5_TEMPLATE_SHA256': 'f' * 64}

    def plan(self, path, **env):
        return q5_case.build_plan(q5_case.loads(path.read_bytes()), self.BUILTINS, {**self.REQUIRED, **env})

    def test_cases_validate_and_require_config_and_template_digests(self):
        for q3f, path in CASES.items():
            plan = self.plan(path)
            self.assertFalse(plan['complete'])
            self.assertEqual(set(plan['missing_variables']), {'MODEL_CONFIG_SHA256', 'TEMPLATE_SHA256'})
            self.assertEqual(q5_case.loads(path.read_bytes())['family'], q3f)
            self.assertIn(q3f, om.Q3_FAMILIES)

    def test_serve_argv_is_the_launcher_over_the_approved_spec(self):
        for q3f, path in CASES.items():
            case = q5_case.loads(path.read_bytes())
            self.assertEqual(om.label(case), om.LABELS[q3f])
            plan = self.plan(path, **self.DIGESTS)
            self.assertTrue(plan['complete'])
            approval = plan['client']['approval']['set']
            spec = ready.native_spec(approval['native_omlx'])
            argv = plan['serve']['argv']
            self.assertEqual(argv[:5], ['/usr/bin/python3', '-E', '-s', '-B',
                                        str(HERE / 'native_peer_omlx_launch.py')])
            flags = dict(zip(argv[5::2], argv[6::2]))
            self.assertEqual({k[2:].replace('-', '_'): v for k, v in flags.items()},
                             {k: str(spec[k]) for k in ready.LAUNCH_KEYS})
            self.assertEqual((spec['q3_family'], spec['port'], spec['run_dir']), (q3f, int(PORTS[q3f]), '/out/run/omlx'))
            self.assertEqual(spec['model_dir'], '/Users/robert/.cache/huggingface/hub/models--'
                             + om.ARTIFACTS[q3f]['repo'].replace('/', '--') + '/snapshots/' + om.ARTIFACTS[q3f]['revision'])
            self.assertEqual(spec['omlx_py'], '/Users/robert/.local/share/hf2q/peers/omlx-6b413c9/.venv/bin/python')
            self.assertEqual(plan['binary'], spec['omlx_py'])
            self.assertEqual(plan['serve']['env'], ready.SERVE_ENV)
            self.assertEqual(plan['artifact'], {'path': spec['model_dir'] + '/config.json', 'sha256': 'e' * 64})
            self.assertEqual((approval['native_omlx_settings_sha256'], approval['native_omlx_launch_sha256']),
                             ({'sha256_of': '/out/run/omlx/base/model_settings.json'},
                              {'sha256_of': '/out/run/omlx/launch.json'}))
            client = plan['client']['argv']
            self.assertEqual((client[client.index('--family') + 1], client[client.index('--engine-kind') + 1],
                              client[client.index('--model') + 1]), (om.ARTIFACTS[q3f]['family'], 'omlx', MODEL[q3f]))
            sources = approval['sources']['sha256_of']
            self.assertEqual({Path(s).name for s in sources}, set(MODULES) | set(om.CLIENT_MODULES))
            self.assertTrue(all(Path(s).is_file() for s in sources))
            checks = {(c['url'].split('/', 3)[3], c['field']): c['equals'] for c in plan['readiness']['checks']}
            self.assertEqual(checks, {('health', 'default_model'): MODEL[q3f], ('api/status', 'version'): om.OMLX_VERSION,
                                      ('api/status', 'models_loaded'): 1, ('api/status', 'total_requests'): 0})
            self.assertEqual((plan['readiness']['field'], plan['readiness']['equals']), ('status', 'healthy'))

    def test_ports_are_unique_among_checked_in_cases(self):
        ports = [q5_case.loads(p.read_bytes())['variables']['PORT'] for p in sorted((HERE / 'q5_cases').glob('*.json'))
                 if not p.name.startswith('cancel-')]
        self.assertEqual([ports.count(p) for p in PORTS.values()], [1, 1, 1])


def manifest(family='qwen36', warnings=0, comparison=True):
    trace = {'tool_parser_warnings': warnings}
    blocks = [{'block': b, 'order': ['stock', 'experiment', 'peer'], 'entries': [
        {'name': f'n4-{b}-{arm}', 'arm': arm, 'kind': 'measured', 'block': b, 'attempts': [], 'outcome': 'Pass',
         'failure_kind': None, 'reason': None,
         'metrics': {**metrics(*spec), 'audit': {'native_trace': dict(trace)} if arm == 'peer' else None}}
        for arm, spec in (('stock', (100.0,)), ('experiment', (80.0, 2400, 1.5)), ('peer', (100.0, 1600, 3.0)))]}
        for b in range(1, 6)]
    plan = {'family': family, 'primary': 'n4', 'measured_arms': ['stock', 'experiment', 'peer'],
            'arms': {'stock': {'engine_kind': 'hf2q'}, 'experiment': {'engine_kind': 'hf2q'},
                     'peer': {'engine_kind': 'omlx', 'label': om.LABELS[family]}},
            'peer_unavailable_evidence': [], 'design': {},
            'declared': {'stock': {'label': 'v0.1.28 release', 'commit': '1' * 40},
                         'experiment': {'label': 'main HEAD', 'commit': '2' * 40}, 'qe_identity': 'qe'}}
    if comparison:
        plan['native_comparison'] = om.NATIVE_COMPARISON
    artifact = {'path': '/m.gguf', 'bytes': 10, 'sha256': 'a' * 64}
    return {'plan': plan, 'unavailable': {}, 'identities_sha256': 'x', 'status': 'Complete',
            'created_utc': '2026-10-31T00:00:00+00:00',
            'identities': {'arms': {'experiment': {'binary': {'sha256': 'e' * 64}, 'artifact': artifact}}},
            'stages': [{'workload': 'n4', 'phase_plan': None, 'status': 'Complete',
                        'stop': {'reason': 'precision', 'blocks': 5}, 'warmups': [], 'blocks': blocks}]}


class Campaign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.nc, self.evidence = self.root / 'nc.json', self.root / 'evidence.json'
        self.nc.write_text('{}')
        self.evidence.write_text('{}')

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, family, *extra):
        return q3.parse_args(['--family', family, '--stock-bin', '/bin/stock', '--stock-commit', '1' * 40,
                              '--experiment-bin', '/bin/experiment', '--experiment-commit', '2' * 40,
                              '--qe-identity', 'qe', '--negative-control-receipt', str(self.nc), '--workloads', 'n4',
                              '--output', str(self.root / 'q3'), *extra])

    def test_native_comparison_campaign_labels_the_omlx_peer_arm(self):
        for q3f, path in CASES.items():
            plan, cases = q3.build_plan(self.args(q3f, '--peer-case', str(path), '--native-comparison'), {}, q3f)
            peer = plan['arms']['peer']
            self.assertEqual((peer['engine_kind'], peer['label'], peer['role']),
                             ('omlx', om.LABELS[q3f], om.q3_role(cases['peer'])))
            self.assertEqual((plan['native_comparison'], plan['peer_unavailable_evidence'], plan['measured_arms']),
                             (om.NATIVE_COMPARISON, [], ['stock', 'experiment', 'peer']))
            self.assertEqual(q3_identity.native_arms(plan), ['peer'])

    def test_native_comparison_refusals(self):
        qwen36 = str(CASES['qwen36'])
        bad = [('qwen36', '--peer-case', qwen36), ('qwen36', '--peer-case', qwen36, '--peer-unavailable-evidence',
                                                   str(self.evidence)),
               ('qwen36', '--peer-case', qwen36, '--native-comparison', '--peer-unavailable-evidence', str(self.evidence)),
               ('qwen38', '--peer-case', qwen36, '--native-comparison'),
               ('gemma4', '--peer-case', str(CASES['qwen38']), '--native-comparison'),
               ('deepseek4', '--peer-case', str(CASES['gemma4']), '--native-comparison')]
        for extra in bad:
            with self.subTest(extra=extra[1:]), self.assertRaises(q3.qc.CampaignError):
                q3.build_plan(self.args(*extra), {}, extra[0])

    def test_native_artifact_is_a_pinned_snapshot_config(self):
        plan, _ = q3.build_plan(self.args('gemma4', '--peer-case', str(CASES['gemma4']), '--native-comparison'), {}, 'gemma4')
        ids = {'arms': {arm: {'binary': {'path': f'/bin/{arm}', 'sha256': str(i) * 64, 'exists': True},
                              'artifact': {'path': '/m.gguf', 'bytes': 10, 'sha256': 'a' * 64}, 'peer_build_info': None,
                              'missing_variables': [], 'file_checks': {}, 'inputs': {}}
                        for i, arm in enumerate(plan['arms'], 1)},
               'harness': {'git_commit': 'c' * 40, 'git_dirty': [], 'sources': {}}, 'python': {}, 'host': {},
               'negative_control_receipt': plan['negative_control_receipt']}
        artifact = om.ARTIFACTS['gemma4']
        config = (f"/hub/models--{artifact['repo'].replace('/', '--')}/snapshots/{artifact['revision']}/config.json")
        ids['arms']['peer']['artifact'] = {'path': config, 'bytes': 9, 'sha256': 'c' * 64}
        self.assertEqual(q3_identity.launch_problems(plan, ids), [])
        ids['arms']['peer']['artifact'] = {'path': '/m.gguf', 'bytes': 10, 'sha256': 'a' * 64}  # ambient hf2q artifact
        self.assertEqual(len([p for p in q3_identity.launch_problems(plan, ids) if p.startswith('peer:')]), 2)

    def test_verdict_is_labeled_not_of_record_and_reports_the_native_comparison(self):
        verdict = q3_report.compute_verdict(manifest())
        self.assertEqual((verdict['verdict'], verdict['reasons']), ('Unavailable', [om.NATIVE_COMPARISON]))
        self.assertEqual((verdict['verdict_of_record'], verdict['campaign_label']),
                         (False, 'native-comparison campaign; not the verdict of record'))
        native = verdict['native_peer']
        self.assertTrue(native['statement'].startswith(
            f"main HEAD finishes the same coding task 20.0% sooner than {om.LABELS['qwen36']}"))
        self.assertEqual(native['ttft_p95_ratio_experiment_over_omlx'], 0.5)
        self.assertEqual(native['tool_parser_warnings_per_run']['n4-1-peer'], 0)

    def test_failures_missing_audits_or_tool_parser_warnings_withhold_the_statement(self):
        native = q3_report.compute_verdict(manifest(warnings=2))['native_peer']
        self.assertEqual(native['statement'], None)
        self.assertTrue(native['statement_withheld'].startswith('oMLX tool-parser warnings in runs: n4-1-peer'))
        document = manifest()
        document['stages'][0]['blocks'][0]['entries'][2].update(outcome='Fail', failure_kind='task', reason='x')
        self.assertEqual(q3_report.compute_verdict(document)['native_peer']['statement_withheld'], 'failed task runs')
        document = manifest()
        document['stages'][0]['blocks'][1]['entries'][2]['metrics']['audit'] = None
        self.assertTrue(q3_report.compute_verdict(document)['native_peer']['statement_withheld']
                        .startswith('oMLX audit missing for runs: n4-2-peer'))

    def test_rows_are_labeled_and_render_in_the_native_section(self):
        for q3f in om.Q3_FAMILIES:
            out = self.root / 'evidence' / f'q3-{q3f}-omlx'
            out.mkdir(parents=True)
            document = manifest(q3f)
            (out / 'campaign.json').write_text(json.dumps(document))
            (out / 'verdict.json').write_text(json.dumps(q3_report.compute_verdict(document)))
            rows = q3_report.rows(out, self.root)
            group = f'{q3f}-q3-omlx-{q3f}-native'
            self.assertEqual(list(rows['groups']), [group])
            self.assertEqual([r['baseline_arm'] for r in rows['rows']], [f'omlx_{q3f}_native'])
            self.assertIn('TTFT ratio experiment/oMLX 0.5', rows['rows'][0]['note'])
            board = {'schema': f2.ROWS_SCHEMA, 'evidence_root_symbol': '$E', 'evidence_root_description': 'test',
                     'neutral_band_ratio': 1.025, 'neutral_band_basis': 'test',
                     'families': {q3f: {'title': TITLES[q3f], 'artifact_sha256': 'a' * 64, 'note': ''}}, **rows}
            f2.validate_manifest(board)
            self.assertEqual(f2.verify_rows(board, self.root), [r['id'] for r in rows['rows']])
            text = f2.render(f2.validate_manifest(board), [], '0' * 64)
            title = '## Native-engine comparisons (different artifact, task makespan only)'
            section = text[text.index(title):text.find('\n## ', text.index(title) + len(title))]
            self.assertIn(f"`{rows['rows'][0]['id']}`", section)
            self.assertIn(om.LABELS[q3f], section)
            self.assertFalse(f2.effect_is_win(rows['rows'][0]))


def line(model, usage, finish, stream):
    tail = ', stream_model_ttft=0.41s, stream_visible_ttft=0.52s' if stream else ''
    return (f"2026-10-31 10:00:02,000 - omlx.server - INFO - Chat completion: model={model}, "
            f"{usage['completion_tokens']} tokens in 2.10s (23.7 tok/s), prompt: {usage['prompt_tokens']}, "
            f"finish_reason={finish}, max_tokens=8192, request_max_tokens=8192{tail}\n")


class Audit(unittest.TestCase):
    def run_dir(self, root, lines=None, extra='', proof=True, tamper=False):
        client, model = root / 'client', MODEL['qwen38']
        responses = [(completion_body(call_message(), 'tool_calls'), False, use()),
                     (stream(), True, use(cached=768, time_to_first_visible_token=0.5))]
        receipts, rows = [], []
        for turn, (raw, streamed, usage) in enumerate(responses):
            directory = client / 'agent-0' / f'turn-{turn}'
            directory.mkdir(parents=True)
            (directory / 'response.bin').write_bytes(raw)
            tokens = json.dumps({'count': 1000, 'max_model_len': 262144, 'tokens': list(range(1000)),
                                 'token_strs': None}).encode()
            (directory / 'omlx-tokenize.raw').write_bytes(tokens + (b' ' if tamper else b''))
            receipt = {'turn': turn, 'stream': streamed, 'request_sha256': str(turn) * 64, 'usage': om.usage(usage, turn)}
            receipts.append(receipt)
            rows.append({'turn': turn, 'request_sha256': receipt['request_sha256'], 'count': 1000,
                         'tokens_sha256': ready.token_sha256(range(1000)),
                         'response_sha256': hashlib.sha256(tokens).hexdigest()})
        default = line(model, use(), 'stop', False) + line(model, use(cached=768), 'tool_calls', True)
        (root / 'server.stderr').write_text((default if lines is None else lines) + extra)
        summary = {'native_readiness': {'engine_kind': 'omlx', 'source_revision': om.OMLX_COMMIT, 'q3_family': 'qwen38',
                                        'think_mode': 'thinking'},
                   'native_closing': {'render': {'agent-0': {'status': 'Observed', 'turns': rows}} if proof else {}}}
        return summary, client, {'agent-0': receipts}

    def test_audit_binds_completion_lines_and_tokenize_proofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit.audit_run(*self.run_dir(Path(tmp)))
        self.assertEqual((report['requests'], report['totals']['completion_tokens'], report['totals']['cached_tokens'],
                          report['tool_parser_warnings']), (2, 80, 768, 0))
        for warning in ('2026-10-31 10:00:03,000 - omlx.api.tool_calling - WARNING - Native tool parser failed '
                        'for qwen, recovered via XML fallback\n',
                        '2026-10-31 10:00:03,000 - omlx.api.tool_calling - INFO - Remapped namespaced tool call name '
                        "'functions.read_file' -> 'read_file'\n"):
            with tempfile.TemporaryDirectory() as tmp:
                warned = audit.audit_run(*self.run_dir(Path(tmp), extra=warning))
            self.assertEqual(warned['tool_parser_warnings'], 1, warning)
            self.assertTrue(warned['recovery_attribution'].startswith('Unavailable:'))
        self.assertEqual(audit.final_finish(stream(), True), 'tool_calls')
        self.assertEqual(audit.final_finish(completion_body(call_message(), 'tool_calls'), False), 'tool_calls')
        self.assertEqual(CALL, call_message()['tool_calls'][0]['id'])

    def test_audit_refuses_contradicting_server_records(self):
        model = MODEL['qwen38']
        stream_line = line(model, use(cached=768), 'tool_calls', True)
        cases = [{'lines': line(model, use(), 'length', False) + stream_line},
                 {'lines': line(model, use(completion=41), 'stop', False) + stream_line},
                 {'lines': line(model, use(), 'stop', True) + stream_line},  # unary served as a stream
                 {'lines': line(MODEL['qwen36'], use(), 'stop', False) + stream_line},
                 {'lines': stream_line},  # a client response the server never recorded
                 {'extra': line(model, use(), 'stop', False)},  # a chat request the client never sent
                 {'extra': '2026-10-31 - omlx.api.tool_calling - WARNING - Dropping tool call \'x\': bad\n'},
                 {'extra': '2026-10-31 - omlx.api.tool_calling - WARNING - Dropping bracket tool call: bad\n'},
                 {'extra': '2026-10-31 - omlx.api.tool_calling - WARNING - Dropped 1 tool call(s) from reasoning\n'},
                 {'proof': False}, {'tamper': True}]
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp, self.assertRaises(Invalid):
                audit.audit_run(*self.run_dir(Path(tmp), **changes))


if __name__ == '__main__':
    unittest.main()
