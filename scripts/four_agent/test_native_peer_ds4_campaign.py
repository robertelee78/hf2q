"""Pure controls for the native ds4 arm's Q5 launch case and its Q3 campaign integration.

No launch, model or ds4 process: the Q5 case is planned with q5_case.build_plan, and the Q3
verdict/rows are computed from a hand-built campaign manifest.
"""
import json
from pathlib import Path
import tempfile
import unittest

import f2_scoreboard_render as f2
import native_peer_ds4 as nd
import q3_campaign as q3
import q3_identity
import q3_metrics
import q3_report
import q5_case
from qwen38_n1_verdict import Invalid
from native_peer_ds4 import CLIENT_MODULES as NATIVE_MODULES
from qwen_n4_live_smoke import MODULES
from test_native_peer_ds4 import CASE, HERE, M, use


class Case(unittest.TestCase):
    BUILTINS = {'REPO': str(HERE.parent.parent), 'OUT': '/out/run', 'PYTHON': '/usr/bin/python3',
                'PYTHON_SHA256': 'a' * 64, 'RUN_ID': 'q5-run-0123456789', 'NONCE32': 'b' * 32}
    REQUIRED = {'HF2Q_Q5_CARGO': '/c/cargo', 'HF2Q_Q5_TOOLCHAIN': '/c/tc', 'HF2Q_Q5_SANDBOX_PROOF': '/c/proof.json',
                'HF2Q_Q5_SANDBOX_PROOF_SHA256': 'c' * 64, 'HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE': '/c/approval.json'}
    # Every flag parse_options accepts for this argv (ds4_server.c:15652-15800).
    DS4_FLAGS = {'--chdir', '-m', '--ctx', '--batched-session', '--host', '--port', '--tokens', '--power', '--metal',
                 '--trace'}

    def plan(self, **env):
        return q5_case.build_plan(q5_case.loads(CASE.read_bytes()), self.BUILTINS, {**self.REQUIRED, **env})

    def test_case_validates_and_requires_binary_and_model_digests(self):
        plan = self.plan()
        self.assertFalse(plan['complete'])
        self.assertEqual(plan['missing_variables'], ['DS4_BIN_SHA256', 'MODEL_SHA256'])

    def test_launch_argv_ties_to_the_declared_controls(self):
        plan = self.plan(HF2Q_Q5_DS4_BIN_SHA256='d' * 64, HF2Q_Q5_MODEL_SHA256='e' * 64)
        self.assertTrue(plan['complete'])
        argv = plan['serve']['argv']
        self.assertEqual(argv, ['/opt/ds4/ds4-server', '--chdir', '/opt/ds4', '-m', f'/opt/ds4/gguf/{nd.MODEL_FILE}',
                                '--ctx', '393216', '--batched-session', '4', '--host', '127.0.0.1', '--port', '18251',
                                '--tokens', '8192', '--power', '100', '--metal', '--trace', '/out/run/ds4-trace.txt'])
        self.assertTrue({a for a in argv if a.startswith('-')} <= self.DS4_FLAGS)
        flag = lambda name: argv[argv.index(name) + 1]
        controls = nd.CONTROLS
        self.assertEqual((int(flag('--ctx')), int(flag('--batched-session')), int(flag('--tokens')), int(flag('--power'))),
                         (controls['native_ctx_per_session'], controls['slots'], controls['default_tokens'],
                          controls['power_percent']))
        self.assertFalse({'--mtp', '--dspark', '--kv-disk-dir', '--disable-exact-dsml-tool-replay', '--ssd-streaming',
                          '--quality', '--vision'} & set(argv))
        self.assertEqual(plan['artifact'], {'path': f'/opt/ds4/gguf/{nd.MODEL_FILE}', 'sha256': 'e' * 64})
        self.assertEqual((plan['readiness']['url'], plan['readiness']['field'], plan['readiness']['equals']),
                         ('http://127.0.0.1:18251/v1/models', 'object', 'list'))
        client = plan['client']['argv']
        self.assertEqual((client[client.index('--family') + 1], client[client.index('--engine-kind') + 1],
                          client[client.index('--model') + 1]), ('deepseek4', 'ds4', M))
        approval = plan['client']['approval']['set']
        sources = approval['sources']['sha256_of']
        self.assertEqual({Path(s).name for s in sources}, set(MODULES) | set(NATIVE_MODULES))
        self.assertTrue(all(Path(s).is_file() for s in sources))
        spec = nd.native_spec(approval['native_ds4'])
        self.assertEqual((spec['ctx'], spec['port'], spec['trace'], spec['server_log']),
                         (393216, 18251, '/out/run/ds4-trace.txt', '/out/run/server.stderr'))
        self.assertEqual(spec['binary_sha256'], 'd' * 64)

    def test_lowering_the_context_is_refused_by_the_adapter(self):
        plan = self.plan(HF2Q_Q5_DS4_BIN_SHA256='d' * 64, HF2Q_Q5_MODEL_SHA256='e' * 64, HF2Q_Q5_NATIVE_DS4_CTX='262144')
        with self.assertRaises(Invalid):
            nd.native_spec(plan['client']['approval']['set']['native_ds4'])

    def test_port_is_unique_among_checked_in_cases(self):
        ports = [q5_case.loads(p.read_bytes())['variables']['PORT'] for p in sorted((HERE / 'q5_cases').glob('*.json'))
                 if not p.name.startswith('cancel-')]
        self.assertEqual(ports.count('18251'), 1)


def metrics(makespan, completion=2000, ttft=1.0):
    return {'makespan_s': makespan, 'client_elapsed_s': makespan, 'goodput_tasks_per_hour': 4 * 3600 / makespan,
            'worst_slot_ttft_p95_s': ttft, 'worst_slot_ttft_max_s': ttft, 'worst_slot_sse_gap_p95_s': 0.1,
            'worst_slot_sse_gap_max_s': 0.2, 'final_source_sha256': {'agent-0': 'f' * 64},
            'memory': {'q5_verdict': 'CalibrationPass', 'q5_pass': True, 'min_free_plus_cached_bytes': 1,
                       'swap_growth_bytes': 0, 'pressure_counts': {}},
            'thermal': {'preflight_state': 0, 'post_run_state': 0, 'valid': True},
            'tokens': {'prompt': 100000, 'completion': completion, 'cached': 0}}


def manifest(peer_kind):
    blocks = [{'block': b, 'order': ['stock', 'experiment', 'peer'], 'entries': [
        {'name': f'n4-{b}-{arm}', 'arm': arm, 'kind': 'measured', 'block': b, 'attempts': [], 'outcome': 'Pass',
         'failure_kind': None, 'reason': None,
         'metrics': {**metrics(*spec), 'audit': {'native_trace': {'recovery_total': 0}} if arm == 'peer' else None}}
        for arm, spec in (('stock', (100.0,)), ('experiment', (80.0, 2400, 1.5)), ('peer', (100.0, 1600, 3.0)))]}
        for b in range(1, 6)]
    peer = {'engine_kind': peer_kind, **({'label': nd.LABEL} if peer_kind == 'ds4' else {})}
    plan = {'family': 'deepseek4', 'primary': 'n4', 'measured_arms': ['stock', 'experiment', 'peer'],
            'arms': {'stock': {'engine_kind': 'hf2q'}, 'experiment': {'engine_kind': 'hf2q'}, 'peer': peer},
            'peer_unavailable_evidence': [{'path': '/e/peer-arms-v1/peer-deepseek4-oom.json', 'sha256': '0' * 64}]
            if peer_kind == 'ds4' else [], 'design': {},
            'declared': {'stock': {'label': 'v0.1.28 release', 'commit': '1' * 40},
                         'experiment': {'label': 'main HEAD', 'commit': '2' * 40}, 'qe_identity': 'qe'}}
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
        self.nc, self.evidence = self.root / 'nc.json', self.root / 'peer-deepseek4-oom.json'
        self.nc.write_text('{}')
        self.evidence.write_text('{}')

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return q3.parse_args(['--family', 'deepseek4', '--stock-bin', '/bin/stock', '--stock-commit', '1' * 40,
                              '--experiment-bin', '/bin/experiment', '--experiment-commit', '2' * 40,
                              '--qe-identity', 'qe', '--negative-control-receipt', str(self.nc), '--workloads', 'n4',
                              '--output', str(self.root / 'q3'), *extra])

    def test_native_peer_arm_is_labeled_and_needs_same_gguf_unavailable_evidence(self):
        plan, _ = q3.build_plan(self.args('--peer-case', str(CASE), '--peer-unavailable-evidence', str(self.evidence)),
                                {}, 'deepseek4')
        self.assertEqual(plan['measured_arms'], ['stock', 'experiment', 'peer'])
        self.assertEqual((plan['arms']['peer']['engine_kind'], plan['arms']['peer']['role'], plan['arms']['peer']['label']),
                         ('ds4', nd.Q3_ROLE, nd.LABEL))
        for extra in (('--peer-case', str(CASE)),
                      ('--peer-case', str(HERE / 'q5_cases' / 'peer-deepseek4.json'),
                       '--peer-unavailable-evidence', str(self.evidence)),
                      ('--peer-matched-case', str(CASE))):
            with self.subTest(extra=extra), self.assertRaises(q3.qc.CampaignError):
                q3.build_plan(self.args(*extra), {}, 'deepseek4')
        default, _ = q3.build_plan(self.args(), {}, 'deepseek4')
        self.assertEqual((default['arms']['peer']['engine_kind'], default['arms']['peer']['role']), ('peer', q3.ROLES['peer']))
        self.assertNotIn('label', default['arms']['peer'])

    def test_native_artifact_is_exempt_from_the_same_model_rule_but_must_be_bound(self):
        plan, _ = q3.build_plan(self.args('--peer-case', str(CASE), '--peer-unavailable-evidence', str(self.evidence)),
                                {}, 'deepseek4')
        ids = {'arms': {arm: {'binary': {'path': f'/bin/{arm}', 'sha256': str(i) * 64, 'exists': True},
                              'artifact': {'path': '/m.gguf', 'bytes': 10, 'sha256': 'a' * 64}, 'peer_build_info': None,
                              'missing_variables': [], 'file_checks': {}, 'inputs': {}}
                        for i, arm in enumerate(plan['arms'], 1)},
               'harness': {'git_commit': 'c' * 40, 'git_dirty': [], 'sources': {}}, 'python': {}, 'host': {},
               'negative_control_receipt': plan['negative_control_receipt']}
        ids['arms']['peer']['artifact'] = {'path': f'/opt/ds4/gguf/{nd.MODEL_FILE}', 'bytes': 9, 'sha256': 'c' * 64}
        self.assertEqual(q3_identity.native_arms(plan), ['peer'])
        self.assertEqual(q3_identity.launch_problems(plan, ids), [])
        self.assertFalse(q3_identity.same_model(ids))  # the default rule still sees two artifacts
        ids['arms']['peer']['artifact']['sha256'] = None
        self.assertIn('peer: native artifact with its sha256 is required', q3_identity.launch_problems(plan, ids))
        ids['arms']['peer']['artifact'] = {'path': '/m.gguf', 'bytes': 10, 'sha256': 'a' * 64}  # ambient hf2q MODEL leaked
        problems = q3_identity.launch_problems(plan, ids)
        self.assertIn(f'peer: native artifact must be {nd.MODEL_FILE} (set --peer-var MODEL)', problems)
        self.assertIn("peer: native artifact sha256 equals the hf2q artifact's (set --peer-var MODEL_SHA256)", problems)
        ids['arms']['peer']['artifact'] = {'path': f'/opt/ds4/gguf/{nd.MODEL_FILE}', 'bytes': 9, 'sha256': 'a' * 64}
        self.assertEqual([p for p in q3_identity.launch_problems(plan, ids) if p.startswith('peer:')],
                         ["peer: native artifact sha256 equals the hf2q artifact's (set --peer-var MODEL_SHA256)"])

    def test_verdict_reports_task_makespan_separately_from_the_same_gguf_verdict(self):
        verdict = q3_report.compute_verdict(manifest('ds4'))
        self.assertEqual(verdict['verdict'], 'Unavailable')
        self.assertIn('same-GGUF peer unavailable', verdict['reasons'][0])
        native = verdict['native_peer']
        self.assertTrue(native['statement'].startswith(
            'hf2q 22222222 finishes the same coding task 20.0% sooner than ds4 c97964b (native Flash-0731 q2 artifact)'))
        self.assertIn('task makespan only', native['statement'])
        self.assertEqual(native['token_volume_median'], {'experiment': {'prompt': 100000, 'completion': 2400},
                                                         'peer': {'prompt': 100000, 'completion': 1600}})
        self.assertEqual(native['output_tokens_per_s_median'], {'experiment': 30.0, 'peer': 16.0})
        self.assertEqual(native['ttft_p95_ratio_experiment_over_ds4'], 0.5)
        same_gguf = q3_report.compute_verdict(manifest('peer'))
        self.assertNotIn('native_peer', same_gguf)
        self.assertNotEqual(same_gguf['verdict'], 'Unavailable')

    def test_a_failed_ds4_task_withholds_the_makespan_statement(self):
        document = manifest('ds4')
        failed = document['stages'][0]['blocks'][0]['entries'][2]
        self.assertEqual(failed['arm'], 'peer')
        failed.update(outcome='Fail', failure_kind='task', reason='exactly one live call required')
        native = q3_report.compute_verdict(document)['native_peer']
        self.assertIsNone(native['statement'])
        self.assertEqual((native['statement_withheld'], native['failures']['peer']), ('failed task runs', ['n4-1-peer']))
        self.assertEqual(native['makespan']['blocks'], 4)

    def test_server_side_tool_call_recovery_withholds_the_statement(self):
        document = manifest('ds4')
        peer = document['stages'][0]['blocks'][0]['entries'][2]
        peer['metrics']['audit']['native_trace']['recovery_total'] = 2
        native = q3_report.compute_verdict(document)['native_peer']
        self.assertIsNone(native['statement'])
        self.assertTrue(native['statement_withheld'].startswith('ds4 server-side tool-call recovery'))
        self.assertIn('n4-1-peer', native['statement_withheld'])
        self.assertEqual(native['tool_call_recovery_per_run']['n4-1-peer'], 2)
        self.assertEqual(q3_report.compute_verdict(document)['verdict'], 'Unavailable')  # recorded, never Invalid
        peer['metrics']['audit'] = None
        self.assertTrue(q3_report.compute_verdict(document)['native_peer']['statement_withheld']
                        .startswith('ds4 trace audit missing'))

    def test_rows_label_the_native_baseline_and_validate(self):
        out = self.root / 'evidence' / 'q3-deepseek4'
        out.mkdir(parents=True)
        document = manifest('ds4')
        (out / 'campaign.json').write_text(json.dumps(document))
        (out / 'verdict.json').write_text(json.dumps(q3_report.compute_verdict(document)))
        rows = q3_report.rows(out, self.root)
        # a native-peer campaign is never the verdict of record: only the ds4 row, no experiment-vs-stock row
        self.assertEqual([r['baseline_arm'] for r in rows['rows']], ['ds4_native'])
        self.assertIn(nd.LABEL, rows['groups']['deepseek4-q3']['workload'])
        self.assertIn('not the same-GGUF peer verdict', rows['rows'][0]['note'])
        board = {'schema': f2.ROWS_SCHEMA, 'evidence_root_symbol': '$E', 'evidence_root_description': 'test',
                 'neutral_band_ratio': 1.025, 'neutral_band_basis': 'test',
                 'families': {'deepseek4': {'title': 'DeepSeek-V4 Flash', 'artifact_sha256': 'a' * 64, 'note': ''}},
                 **rows}
        f2.validate_manifest(board)
        self.assertEqual(f2.verify_rows(board, self.root), [r['id'] for r in rows['rows']])
        native_row = rows['rows'][0]
        for figure in ('finishes the same coding task 20.0% sooner', 'Median tokens per run', 'output tokens/s',
                       'TTFT ratio experiment/ds4 0.5'):
            self.assertIn(figure, native_row['note'])
        text = f2.render(f2.validate_manifest(board), [], '0' * 64)
        title = '## Native-engine comparisons (different artifact, task makespan only)'
        section = lambda start: text[text.index(start):text.find('\n## ', text.index(start) + len(start))]
        self.assertIn(f"`{native_row['id']}`", section(title))
        self.assertIn('Median tokens per run', section(title))
        self.assertIn('None.', section('### Descriptive wins'))  # nothing of this campaign lands among the wins
        for other in ('## Wins vs baseline', '## Negative results', '## Open gaps', '## DeepSeek-V4 Flash'):
            self.assertNotIn(f"`{native_row['id']}`", section(other))
        self.assertFalse(f2.effect_is_win(native_row))

    def test_metrics_carry_native_build_identity_and_token_volume(self):
        summary = {'native_readiness': {'build_info': 'ds4 x'},
                   'agents': {'agent-0': {'turns': [{'usage': use(cached=0)}, {'usage': use(cached=900)}]}}}
        self.assertEqual(q3_metrics.peer_build_info(self.root, {'readiness': {'documents': {}}}, summary), 'ds4 x')
        self.assertIsNone(q3_metrics.peer_build_info(self.root, {}, {}))
        self.assertEqual(q3_metrics.token_volume(summary['agents']), {'prompt': 2000, 'completion': 80, 'cached': 900})
        self.assertIsNone(q3_metrics.token_volume({'agent-0': {'turns': [{'usage': None}]}}))


if __name__ == '__main__':
    unittest.main()
