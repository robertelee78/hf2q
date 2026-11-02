"""Pure controls for the TensorFold Q5 cases, the Q3 native-comparison campaign, F2 rows and the post-run audit.

No launch, model or TensorFold process: cases are planned with q5_case.build_plan, the Q3 verdict/rows come
from a hand-built campaign manifest, and the audit reads a hand-built run directory (server logs follow
server/chat.zig:590 done lines and native/qwen27_served.zig:199 slot waits).
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import f2_scoreboard_render as f2
import native_peer_tensorfold as tf
import native_peer_tensorfold_audit as audit
import native_peer_tensorfold_ready as ready
import q3_campaign as q3
import q3_identity
import q3_report
import q5_case
from qwen38_n1_verdict import Invalid
from qwen_n4_live_smoke import MODULES
from test_native_peer_ds4 import CASE as DS4_CASE
from test_native_peer_tensorfold import CALL, SHA, call_message, completion_body, speculative, stream, use

HERE = Path(__file__).resolve().parent
PLAIN = HERE / 'q5_cases' / 'native-tensorfold-qwen38.json'
DFLASH2 = HERE / 'q5_cases' / 'native-tensorfold-qwen38-dflash2.json'
# Every flag serve accepts that these cases use (server/cli.zig:41-155).
TF_FLAGS = {'--name', '--host', '--port', '--context', '--parallel', '--temperature', '--max-tokens', '--keep-warm',
            '--no-update-check', '--drafter', '--drafter-bits'}


class Case(unittest.TestCase):
    BUILTINS = {'REPO': str(HERE.parent.parent), 'OUT': '/out/run', 'PYTHON': '/usr/bin/python3',
                'PYTHON_SHA256': 'a' * 64, 'RUN_ID': 'q5-run-0123456789', 'NONCE32': 'b' * 32}
    REQUIRED = {'HF2Q_Q5_CARGO': '/c/cargo', 'HF2Q_Q5_TOOLCHAIN': '/c/tc', 'HF2Q_Q5_SANDBOX_PROOF': '/c/proof.json',
                'HF2Q_Q5_SANDBOX_PROOF_SHA256': 'c' * 64, 'HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE': '/c/approval.json'}
    DIGESTS = {'HF2Q_Q5_TF_BIN_SHA256': 'd' * 64, 'HF2Q_Q5_TARGET_CONFIG_SHA256': 'e' * 64}

    def plan(self, path, **env):
        return q5_case.build_plan(q5_case.loads(path.read_bytes()), self.BUILTINS, {**self.REQUIRED, **env})

    def test_cases_validate_and_require_binary_and_config_digests(self):
        for path in (PLAIN, DFLASH2):
            plan = self.plan(path)
            self.assertFalse(plan['complete'])
            self.assertEqual(set(plan['missing_variables']), {'TF_BIN_SHA256', 'TARGET_CONFIG_SHA256'})
            self.assertEqual((q5_case.loads(path.read_bytes())['family'],), tf.Q3_FAMILIES)

    def test_launch_argv_is_the_adapter_launch_and_differs_only_by_the_drafter(self):
        argv = {}
        for path, variant in ((PLAIN, 'none'), (DFLASH2, 'dflash2')):
            case = q5_case.loads(path.read_bytes())
            self.assertEqual((tf.variant(case), tf.label(case)), (variant, tf.LABELS[variant]))
            plan = self.plan(path, **self.DIGESTS)
            self.assertTrue(plan['complete'])
            approval = plan['client']['approval']['set']
            spec = ready.native_spec(approval['native_tensorfold'])
            argv[variant] = plan['serve']['argv']
            self.assertEqual(argv[variant], ready.server_argv(spec))
            self.assertTrue({a for a in argv[variant] if a.startswith('-')} <= TF_FLAGS)
            self.assertEqual(plan['serve']['env'], {'HF_HUB_CACHE': '/Users/robert/.cache/huggingface/hub',
                                                    'HF_HUB_OFFLINE': '1', 'TENSORFOLD_NO_UPDATE_CHECK': '1',
                                                    'TENSORFOLD_NO_LIVE': '1', 'TF_CHECKPOINT_REVISION': tf.TARGET_REVISION})
            self.assertEqual(plan['artifact'], {'path': spec['target_dir'] + '/config.json', 'sha256': 'e' * 64})
            self.assertEqual(spec['binary_sha256'], 'd' * 64)
            client = plan['client']['argv']
            self.assertEqual((client[client.index('--family') + 1], client[client.index('--engine-kind') + 1],
                              client[client.index('--model') + 1]), ('qwen', 'tensorfold', tf.MODEL_ID))
            sources = approval['sources']['sha256_of']
            self.assertEqual({Path(s).name for s in sources}, set(MODULES) | set(tf.CLIENT_MODULES))
            self.assertTrue(all(Path(s).is_file() for s in sources))
            checks = {(c['url'].split('/', 3)[3], c['field']): c['equals'] for c in plan['readiness']['checks']}
            self.assertEqual(checks, {('health', 'model'): tf.MODEL_ID, ('health', 'max_batch_size'): 4,
                                      ('health', 'version'): tf.TF_VERSION,
                                      ('v1/benchmark/manifest', 'model.revision'): tf.TARGET_REVISION,
                                      ('v1/benchmark/manifest', 'model.config_sha256'): 'e' * 64})
        port = argv['none'].index('--port') + 1
        self.assertEqual((argv['none'][port], argv['dflash2'][port]), ('18252', '18253'))
        plain = argv['none'][:port] + ['PORT'] + argv['none'][port + 1:]
        drafted = argv['dflash2'][:port] + ['PORT'] + argv['dflash2'][port + 1:]
        self.assertEqual(drafted[:len(plain)], plain)
        self.assertEqual(drafted[len(plain):][::2], ['--drafter', '--drafter-bits'])

    def test_ports_are_unique_among_checked_in_cases(self):
        ports = [q5_case.loads(p.read_bytes())['variables']['PORT'] for p in sorted((HERE / 'q5_cases').glob('*.json'))
                 if not p.name.startswith('cancel-')]
        self.assertEqual((ports.count('18252'), ports.count('18253')), (1, 1))


def metrics(makespan, completion=2000, ttft=1.0):
    return {'makespan_s': makespan, 'client_elapsed_s': makespan, 'goodput_tasks_per_hour': 4 * 3600 / makespan,
            'worst_slot_ttft_p95_s': ttft, 'worst_slot_ttft_max_s': ttft, 'worst_slot_sse_gap_p95_s': 0.1,
            'worst_slot_sse_gap_max_s': 0.2, 'final_source_sha256': {'agent-0': 'f' * 64},
            'memory': {'q5_verdict': 'CalibrationPass', 'q5_pass': True, 'min_free_plus_cached_bytes': 1,
                       'swap_growth_bytes': 0, 'pressure_counts': {}},
            'thermal': {'preflight_state': 0, 'post_run_state': 0, 'valid': True},
            'tokens': {'prompt': 100000, 'completion': completion, 'cached': 0}}


def manifest(comparison=True):
    trace = {'accepted_per_drafted': 0.6, 'tokens_per_round': 1.4, 'slot_memory_waits': 0}
    blocks = [{'block': b, 'order': ['stock', 'experiment', 'peer'], 'entries': [
        {'name': f'n4-{b}-{arm}', 'arm': arm, 'kind': 'measured', 'block': b, 'attempts': [], 'outcome': 'Pass',
         'failure_kind': None, 'reason': None,
         'metrics': {**metrics(*spec), 'audit': {'native_trace': dict(trace)} if arm == 'peer' else None}}
        for arm, spec in (('stock', (100.0,)), ('experiment', (80.0, 2400, 1.5)), ('peer', (100.0, 1600, 3.0)))]}
        for b in range(1, 6)]
    plan = {'family': 'qwen38', 'primary': 'n4', 'measured_arms': ['stock', 'experiment', 'peer'],
            'arms': {'stock': {'engine_kind': 'hf2q'}, 'experiment': {'engine_kind': 'hf2q'},
                     'peer': {'engine_kind': 'tensorfold', 'label': tf.LABELS['dflash2']}},
            'peer_unavailable_evidence': [], 'design': {},
            'declared': {'stock': {'label': 'v0.1.28 release', 'commit': '1' * 40},
                         'experiment': {'label': 'main HEAD', 'commit': '2' * 40}, 'qe_identity': 'qe'}}
    if comparison:
        plan['native_comparison'] = tf.NATIVE_COMPARISON
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

    def test_native_comparison_campaign_labels_the_tensorfold_peer_arm(self):
        for path, variant in ((PLAIN, 'none'), (DFLASH2, 'dflash2')):
            plan, cases = q3.build_plan(self.args('qwen38', '--peer-case', str(path), '--native-comparison'), {}, 'qwen38')
            peer = plan['arms']['peer']
            self.assertEqual((peer['engine_kind'], peer['label'], peer['role']),
                             ('tensorfold', tf.LABELS[variant], tf.q3_role(cases['peer'])))
            self.assertEqual((plan['native_comparison'], plan['peer_unavailable_evidence'], plan['measured_arms']),
                             (tf.NATIVE_COMPARISON, [], ['stock', 'experiment', 'peer']))
            self.assertEqual(q3_identity.native_arms(plan), ['peer'])
        default, _ = q3.build_plan(self.args('qwen38'), {}, 'qwen38')
        self.assertNotIn('native_comparison', default)

    def test_native_comparison_refusals(self):
        bad = [('qwen38', '--peer-case', str(DFLASH2)),
               ('qwen38', '--peer-case', str(DFLASH2), '--peer-unavailable-evidence', str(self.evidence)),
               ('qwen38', '--peer-case', str(DFLASH2), '--native-comparison', '--peer-unavailable-evidence',
                str(self.evidence)),
               ('qwen38', '--native-comparison'),
               ('qwen38', '--peer-case', str(HERE / 'q5_cases' / 'peer-qwen38.json'), '--native-comparison'),
               ('deepseek4', '--peer-case', str(DS4_CASE), '--native-comparison'),
               ('deepseek4', '--peer-case', str(DFLASH2), '--native-comparison')]
        for extra in bad:
            with self.subTest(extra=extra[1:]), self.assertRaises(q3.qc.CampaignError):
                q3.build_plan(self.args(*extra), {}, extra[0])
        with self.assertRaises(SystemExit):
            self.args('qwen38', '--peer-case', str(DFLASH2), '--native-comparison',
                      '--peer-matched-case', str(HERE / 'q5_cases' / 'peer-qwen38.json'))

    def test_native_artifact_is_the_pinned_snapshot_config(self):
        plan, _ = q3.build_plan(self.args('qwen38', '--peer-case', str(PLAIN), '--native-comparison'), {}, 'qwen38')
        ids = {'arms': {arm: {'binary': {'path': f'/bin/{arm}', 'sha256': str(i) * 64, 'exists': True},
                              'artifact': {'path': '/m.gguf', 'bytes': 10, 'sha256': 'a' * 64}, 'peer_build_info': None,
                              'missing_variables': [], 'file_checks': {}, 'inputs': {}}
                        for i, arm in enumerate(plan['arms'], 1)},
               'harness': {'git_commit': 'c' * 40, 'git_dirty': [], 'sources': {}}, 'python': {}, 'host': {},
               'negative_control_receipt': plan['negative_control_receipt']}
        config = f'/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/{tf.TARGET_REVISION}/config.json'
        ids['arms']['peer']['artifact'] = {'path': config, 'bytes': 9, 'sha256': 'c' * 64}
        self.assertEqual(q3_identity.launch_problems(plan, ids), [])
        ids['arms']['peer']['artifact'] = {'path': '/m.gguf', 'bytes': 10, 'sha256': 'a' * 64}  # ambient hf2q artifact
        self.assertEqual(len([p for p in q3_identity.launch_problems(plan, ids) if p.startswith('peer:')]), 2)

    def test_verdict_is_labeled_not_of_record_and_reports_the_native_comparison(self):
        verdict = q3_report.compute_verdict(manifest())
        self.assertEqual((verdict['verdict'], verdict['reasons']), ('Unavailable', [tf.NATIVE_COMPARISON]))
        self.assertEqual((verdict['verdict_of_record'], verdict['campaign_label']),
                         (False, 'native-comparison campaign; not the verdict of record'))
        native = verdict['native_peer']
        self.assertTrue(native['statement'].startswith(
            f"main HEAD finishes the same coding task 20.0% sooner than {tf.LABELS['dflash2']}"))
        self.assertIn('task makespan only', native['statement'])
        self.assertEqual(native['campaign'], tf.NATIVE_COMPARISON)
        self.assertEqual(native['ttft_p95_ratio_experiment_over_tensorfold'], 0.5)
        self.assertEqual(native['speculative_per_run']['n4-1-peer'], {'accepted_per_drafted': 0.6, 'tokens_per_round': 1.4})

    def test_failures_or_missing_audits_withhold_the_statement(self):
        document = manifest()
        failed = document['stages'][0]['blocks'][0]['entries'][2]
        failed.update(outcome='Fail', failure_kind='task', reason='exactly one live call required')
        native = q3_report.compute_verdict(document)['native_peer']
        self.assertEqual((native['statement'], native['statement_withheld']), (None, 'failed task runs'))
        document = manifest()
        document['stages'][0]['blocks'][1]['entries'][2]['metrics']['audit'] = None
        native = q3_report.compute_verdict(document)['native_peer']
        self.assertIsNone(native['statement'])
        self.assertTrue(native['statement_withheld'].startswith('TensorFold audit missing for runs: n4-2-peer'))

    def test_rows_are_labeled_and_render_in_the_native_section(self):
        out = self.root / 'evidence' / 'q3-qwen38-tensorfold'
        out.mkdir(parents=True)
        document = manifest()
        (out / 'campaign.json').write_text(json.dumps(document))
        (out / 'verdict.json').write_text(json.dumps(q3_report.compute_verdict(document)))
        rows = q3_report.rows(out, self.root)
        group = 'qwen38-q3-tensorfold-dflash2-native'
        self.assertEqual(list(rows['groups']), [group])
        self.assertIn('not the verdict of record', rows['groups'][group]['title'])
        # not the verdict of record: no experiment-vs-stock row, only the TensorFold comparison
        self.assertEqual([r['baseline_arm'] for r in rows['rows']], ['tensorfold_dflash2_native'])
        for row in rows['rows']:
            self.assertTrue(row['note'].startswith('native-comparison campaign; not the verdict of record. Q3 verdict'))
            self.assertTrue(row['variant'].endswith('(native-comparison campaign; not the verdict of record)'))
        self.assertIn('TTFT ratio experiment/TensorFold 0.5', rows['rows'][0]['note'])
        board = {'schema': f2.ROWS_SCHEMA, 'evidence_root_symbol': '$E', 'evidence_root_description': 'test',
                 'neutral_band_ratio': 1.025, 'neutral_band_basis': 'test',
                 'families': {'qwen38': {'title': 'Qwen3.8 27B', 'artifact_sha256': 'a' * 64, 'note': ''}}, **rows}
        f2.validate_manifest(board)
        self.assertEqual(f2.verify_rows(board, self.root), [r['id'] for r in rows['rows']])
        text = f2.render(f2.validate_manifest(board), [], '0' * 64)
        native_row = rows['rows'][0]
        title = '## Native-engine comparisons (different artifact, task makespan only)'
        section = lambda start: text[text.index(start):text.find('\n## ', text.index(start) + len(start))]
        self.assertIn(f"`{native_row['id']}`", section(title))
        self.assertIn(tf.LABELS['dflash2'], section(title))
        self.assertFalse(f2.effect_is_win(native_row))
        for other in ('## Wins vs baseline', '## Negative results', '## Open gaps'):
            self.assertNotIn(f"`{native_row['id']}`", section(other))
        wins = section('## Wins vs baseline')
        self.assertNotIn('qwen38-q3-tensorfold-dflash2-native-makespan-vs-stock', text)
        self.assertNotIn('Q3 experiment main HEAD', wins)


def done(rid, usage, finish='stop', thinking='True', effort='xhigh', spec=None):
    spec = speculative() if spec is None else spec
    return (f"[tensorfold] done {rid} prompt={usage['prompt_tokens']} cached={usage['prompt_tokens_details']['cached_tokens']} "
            f"thinking={thinking} effort={effort} tokens={usage['completion_tokens']} sha={SHA[:12]} finish={finish} "
            f"rounds={spec['rounds']} accepted={spec['accepted']}/{spec['drafted']} ttft=0.600 decode=1.900 "
            'elapsed=2.600 queued=0.000 tok/s=21.0\n')


class Audit(unittest.TestCase):
    def run_dir(self, root, variant='dflash2', spec=None, done_changes=None, proof=True, tamper=False, copies='off',
                extra_done=''):
        spec = speculative() if spec is None else spec
        client = root / 'client'
        unary_usage, stream_usage = use(), use(cached=900)
        responses = [(completion_body(call_message(), 'tool_calls', speculative=spec), False, unary_usage,
                      'chatcmpl-' + '1' * 32),
                     (stream(final=None), True, stream_usage, 'chatcmpl-' + '2' * 32)]
        if spec != speculative():  # rebuild the stream's final chunk with the same counters
            from test_native_peer_tensorfold import chunk, extras
            responses[1] = (stream(final=chunk({}, 'tool_calls', **extras(speculative=spec))), True, stream_usage,
                            responses[1][3])
        receipts, rows, lines = [], [], ''
        for turn, (raw, streamed, usage, rid) in enumerate(responses):
            directory = client / 'agent-0' / f'turn-{turn}'
            directory.mkdir(parents=True)
            (directory / 'response.bin').write_bytes(raw)
            tokens = json.dumps({'count': 1000, 'max_model_len': 262144, 'tokens': list(range(1000))}).encode()
            (directory / 'tensorfold-tokenize.raw').write_bytes(tokens + (b' ' if tamper else b''))
            receipt = {'turn': turn, 'stream': streamed, 'request_sha256': str(turn) * 64, 'usage': tf.usage(usage, turn)}
            receipts.append(receipt)
            rows.append({'turn': turn, 'request_sha256': receipt['request_sha256'], 'count': 1000,
                         'tokens_sha256': ready.token_sha256(range(1000)),
                         'response_sha256': hashlib.sha256(tokens).hexdigest()})
            lines += done(rid, usage, **{'spec': spec, **((done_changes or {}) if turn == 0 else {})})
        (root / 'server.stdout').write_text('[tensorfold] serving qwen3.8-27b ...\n' + lines + extra_done)
        (root / 'server.stderr').write_text('info: Qwen3.8-27B: a slot needs 16.1 GiB and 9.0 GiB is free beside the '
                                            'weights and an 8 GiB margin: the request waits for a stream to end\n')
        summary = {'native_readiness': {'engine_kind': 'tensorfold', 'source_revision': tf.TF_COMMIT, 'variant': variant,
                                        'think_mode': 'thinking, reasoning_effort xhigh',
                                        'startup': {'lane_copies': copies}},
                   'native_closing': {'render': {'agent-0': {'status': 'Observed', 'turns': rows}} if proof else {}}}
        return summary, client, {'agent-0': receipts}

    def test_audit_binds_done_lines_runtime_and_tokenize_proofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = audit.audit_run(*self.run_dir(Path(tmp)))
        totals = report['totals']
        self.assertEqual((report['requests'], totals['drafted'], totals['accepted'], totals['rounds'],
                          totals['completion_tokens'], totals['cached_tokens'], totals['drafts_enabled']),
                         (2, 128, 80, 60, 80, 900, 2))
        self.assertEqual((report['accepted_per_drafted'], report['slot_memory_waits'], report['lane_copies']),
                         (0.625, 1, 'off'))
        self.assertAlmostEqual(report['tokens_per_round'], 80 / 60)
        self.assertTrue(report['recovery'].startswith('Unavailable:'))
        self.assertEqual(audit.final_object(stream(), True)[1], ['{"path":"tests/ports.rs"}'])
        self.assertEqual(audit.final_object(completion_body(call_message(), 'tool_calls'), False)[0]['choices'][0]
                         ['message']['tool_calls'][0]['id'], CALL)

    def test_audit_refuses_contradicting_server_records(self):
        cases = [{'done_changes': {'effort': 'high'}}, {'done_changes': {'thinking': 'False'}},
                 {'done_changes': {'finish': 'length'}}, {'done_changes': {'spec': speculative(accepted=39)}},
                 {'proof': False}, {'tamper': True},
                 {'spec': speculative(drafted=0, accepted=0, acceptance_rate=0.0)},
                 {'extra_done': done('chatcmpl-' + '3' * 32, use())},  # a chat request the client never sent
                 {'variant': 'none'}]  # lone loop=plain with lane copies off, yet drafted
        for changes in cases:
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as tmp, self.assertRaises(Invalid):
                audit.audit_run(*self.run_dir(Path(tmp), **changes))
        plain = speculative(drafted=0, accepted=0, acceptance_rate=0.0)
        with tempfile.TemporaryDirectory() as tmp:  # the plain arm drafts nothing
            report = audit.audit_run(*self.run_dir(Path(tmp), 'none', spec=plain))
            self.assertIsNone(report['accepted_per_drafted'])
        with tempfile.TemporaryDirectory() as tmp:  # with lane copies on, copy drafts are recorded, not refused
            self.assertEqual(audit.audit_run(*self.run_dir(Path(tmp), 'none', copies='on'))['totals']['drafted'], 128)
        with tempfile.TemporaryDirectory() as tmp:
            summary, client, turns = self.run_dir(Path(tmp))
            (Path(tmp) / 'server.stdout').write_text('')
            with self.assertRaises(Invalid):
                audit.audit_run(summary, client, turns)


if __name__ == '__main__':
    unittest.main()
