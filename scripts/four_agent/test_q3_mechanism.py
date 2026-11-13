"""Model-free full-verdict regressions for the ADR-063 Qwen3.8 substitution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import q3_campaign
import q3_mechanism as mechanism
import q3_report
import q3_splash_evidence as splash_evidence
import native_peer_splash as sp
import native_peer_splash_audit as splash_audit
import splash_sweep
import q5_calibration
from test_native_peer_splash import splash_manifest
from test_q3_campaign import FakeRunner, fake_identities
from q3_fake_splash import invalid_sweep

HERE = Path(__file__).resolve().parent



def canonical_identities(plan):
    ids = fake_identities(plan)
    for arm in ids['arms'].values():
        arm['artifact'].update(splash_evidence.ARTIFACT)
    if 'qwen38_splash_substitution' in plan:
        ids['splash_substitution_inputs'] = splash_evidence.current_inputs(plan['qwen38_splash_substitution'])
    return ids


class MechanismVerdict(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self, rows=663, lanes=125):
        m = splash_manifest(arms=('stock', 'experiment', 'peer'))
        p = m['plan']
        p['splash_required'] = False
        selection = invalid_sweep(self.root)
        p['splash_unavailable_evidence'] = [{'path': str(selection), 'sha256': mechanism.digest(selection)}]
        p['qwen38_mechanism'] = dict(mechanism.CONTRACT)
        p['design'].update(stock_margin_ratio=.95, peer_margin_ratio=1.)
        m['identities']['harness'] = {'git_commit': p['declared']['experiment']['commit']}
        case = json.loads((HERE / 'q5_cases/qwen38.json').read_text())
        p['qwen38_splash_substitution'] = splash_evidence.freeze(p['splash_unavailable_evidence'], case)
        for arm in m['identities']['arms'].values():
            arm['artifact'].update(splash_evidence.ARTIFACT)
        base = self.root / 'case.json'
        base.write_text(json.dumps(case))
        p['arms']['experiment'].update(case=str(base), case_sha256=mechanism.digest(base))
        for block in m['stages'][0]['blocks']:
            for e in block['entries']:
                if e['arm'] != 'experiment':
                    continue
                root = self.root / e['name']
                root.mkdir()
                values = [rows // lanes + (i < rows % lanes) for i in range(lanes)]
                text = ''.join(f'DEBUG progress: Qwen DFlash2 verified window slot=0 drafted=7 accepted={n}\n'
                               for n in values)
                (root / 'server.stderr').write_text(text)
                receipt = {'schema': 'hf2q.q5.calibration-receipt.v1', 'family': 'qwen38',
                           'binary': {'sha256': 'e' * 64}, 'artifact': m['identities']['arms']['experiment']['artifact'],
                           'workload': case['workload'], 'case': {'path': str(base), 'sha256': mechanism.digest(base)},
                           'verdict': {'verdict': 'CalibrationPass', 'reason': None}, 'cleanup_proven': True}
                (root / 'receipt.json').write_text(json.dumps(receipt))
                e['attempts'] = [{'output': str(root), 'case': str(base), 'outcome': 'Pass',
                                  'receipt_sha256': mechanism.digest(root / 'receipt.json'),
                                  'server_stderr_sha256': mechanism.digest(root / 'server.stderr')}]
        return m

    def verdict(self, manifest):
        return q3_report.compute_verdict(manifest)

    def test_exact_boundary_and_above_pass_full_report(self):
        for rows in (663, 664):
            with self.subTest(rows=rows):
                # A separate fixture root for every full campaign.
                self.root = self.root / str(rows)
                self.root.mkdir()
                v = self.verdict(self.manifest(rows))
                self.assertEqual(v['verdict'], 'Win', v['reasons'])
                self.assertEqual(v['mechanism']['lane_cycles'], 625)
                self.assertEqual(v['mechanism']['accepted_rows'], rows * 5)
                self.assertEqual(len(v['mechanism']['runs']), 5)

    def test_exact_reviewed_peer_only_variants_preserve_full_report_win(self):
        m = self.manifest()
        selection = json.loads(Path(m['plan']['splash_unavailable_evidence'][0]['path']).read_text())
        run = Path(next(iter(selection['configurations'].values()))['runs'][0])
        path = run / 'client-approval.json'
        approval = json.loads(path.read_text())
        for name, group in splash_evidence.SOURCE_CONTINUITY_WHITELIST.items():
            approval['sources'][str(HERE / name)] = next(value for value in group if value != mechanism.digest(HERE / name))
        path.write_text(json.dumps(approval))
        m['plan']['qwen38_splash_substitution'] = splash_evidence.freeze(
            m['plan']['splash_unavailable_evidence'], json.loads(Path(m['plan']['arms']['experiment']['case']).read_text()))
        self.assertEqual(self.verdict(m)['verdict'], 'Win')

    def test_below_floor_blocks_otherwise_winning_campaign(self):
        v = self.verdict(self.manifest(662))
        self.assertEqual(v['verdict'], 'Inconclusive')
        self.assertIn('Qwen3.8 mechanism below 5.304', ' '.join(v['reasons']))
        self.assertEqual(v['mechanism']['status'], 'loss')

    def test_every_position_required_and_source_case_binary_model_bound(self):
        for fault in ('missing-log', 'changed-log', 'missing-attempt', 'source', 'missing-source', 'missing-artifact',
                      'case', 'receipt', 'binary', 'artifact', 'workload', 'q5', 'zero', 'negative', 'nan',
                      'fractional', 'duplicate', 'inconsistent', 'missing-splash', 'changed-splash', 'valid-splash',
                      'changed-splash-raw', 'missing-splash-raw', 'splash-source', 'splash-artifact', 'splash-workload',
                      'splash-parity', 'splash-config', 'splash-request'):
            with self.subTest(fault=fault):
                self.root = Path(self.tmp.name) / fault
                self.root.mkdir()
                m = self.manifest()
                e = m['stages'][0]['blocks'][-1]['entries'][1]
                a = e['attempts'][-1]
                log, receipt = Path(a['output']) / 'server.stderr', Path(a['output']) / 'receipt.json'
                if fault == 'missing-log':
                    log.unlink()
                elif fault == 'changed-log':
                    log.write_text(log.read_text() + 'changed')
                elif fault == 'missing-attempt':
                    e['attempts'] = []
                elif fault == 'source':
                    m['identities']['harness']['git_commit'] = 'f' * 40
                elif fault == 'missing-source':
                    m['plan']['declared']['experiment']['commit'] = None
                    m['identities']['harness']['git_commit'] = None
                elif fault == 'missing-artifact':
                    m['identities']['arms']['experiment']['artifact']['sha256'] = None
                elif fault == 'missing-splash':
                    m['plan'].pop('splash_unavailable_evidence')
                elif fault == 'changed-splash':
                    Path(m['plan']['splash_unavailable_evidence'][0]['path']).write_text('{}')
                elif fault == 'valid-splash':
                    bound = m['plan']['splash_unavailable_evidence'][0]
                    path = Path(bound['path'])
                    selection = json.loads(path.read_text())
                    config = next(iter(selection['configurations']))
                    selection['choice'] = {'config': config}
                    selection['configurations'][config]['valid'] = True
                    path.write_text(json.dumps(selection))
                    bound['sha256'] = mechanism.digest(path)
                elif fault in ('changed-splash-raw', 'missing-splash-raw', 'splash-source',
                               'splash-artifact', 'splash-workload'):
                    selection = json.loads(Path(m['plan']['splash_unavailable_evidence'][0]['path']).read_text())
                    run = Path(next(iter(selection['configurations'].values()))['runs'][0])
                    if fault.endswith('-raw'):
                        raw = run / 'client/response.raw'
                        if fault.startswith('changed'):
                            raw.write_text('changed raw bytes despite the same invalid selection')
                        else:
                            raw.unlink()
                    elif fault == 'splash-source':
                        path = run / 'client-approval.json'
                        value = json.loads(path.read_text())
                        value['sources'][str(HERE / 'qwen_n4_live_smoke.py')] = 'f' * 64
                        path.write_text(json.dumps(value))
                    else:
                        path = run / 'receipt.json'
                        value = json.loads(path.read_text())
                        if fault == 'splash-artifact':
                            value['artifact']['sha256'] = 'f' * 64
                        else:
                            value['workload']['context_per_slot'] = 1
                        path.write_text(json.dumps(value))
                elif fault in ('splash-parity', 'splash-config', 'splash-request'):
                    selection = json.loads(Path(m['plan']['splash_unavailable_evidence'][0]['path']).read_text())
                    run = Path(next(iter(selection['configurations'].values()))['runs'][0])
                    if fault == 'splash-parity':
                        (run / splash_audit.PARITY_DIR / 'splash-prompt-parity-post-oracle.json').unlink()
                    else:
                        path = run / ('client/summary.json' if fault == 'splash-config'
                                      else 'client/agent-0/turn-0/request.json')
                        value = json.loads(path.read_text())
                        if fault == 'splash-config':
                            value['native_readiness']['config'] = sp.preset('s0', 'qwen38')
                        else:
                            value['max_completion_tokens'] = 1
                        path.write_text(json.dumps(value))
                elif fault == 'case':
                    Path(a['case']).write_text('{}')
                elif fault == 'receipt':
                    receipt.write_text('{}')
                elif fault in ('binary', 'artifact', 'workload', 'q5'):
                    r = json.loads(receipt.read_text())
                    if fault in ('binary', 'artifact'):
                        r[fault]['sha256'] = 'f' * 64
                    elif fault == 'workload':
                        r['workload']['context_per_slot'] = 1
                    else:
                        r['verdict']['verdict'] = 'ClientFailed'
                    receipt.write_text(json.dumps(r))
                    a['receipt_sha256'] = mechanism.digest(receipt)
                else:
                    value = {'zero': '0', 'negative': '-1', 'nan': 'NaN',
                             'fractional': '6.0', 'duplicate': '6 accepted=7'}.get(fault)
                    if value is not None:
                        log.write_text(f'Qwen DFlash2 verified window accepted={value}\n')
                    else:
                        log.write_text(log.read_text() + 'Qwen DFlash2 stage timing active_lanes=4 accepted_rows=24\n')
                    a['server_stderr_sha256'] = mechanism.digest(log)
                v = self.verdict(m)
                self.assertNotEqual(v['verdict'], 'Win', fault)
                self.assertEqual(v['mechanism']['status'], 'unresolved', fault)
                self.assertTrue(any('mechanism' in r for r in v['reasons']))

    def test_candidate_failure_and_other_gates_are_not_waived(self):
        m = self.manifest()
        for arm in ('experiment', 'peer'):
            bad = copy.deepcopy(m)
            e = next(e for e in bad['stages'][0]['blocks'][0]['entries'] if e['arm'] == arm)
            e.update(outcome='Fail', reason='task', failure_kind='task')
            self.assertEqual(self.verdict(bad)['verdict'], 'Failed')
        bad = copy.deepcopy(m)
        bad['plan']['design']['stock_margin_ratio'] = .5
        self.assertEqual(self.verdict(bad)['verdict'], 'Inconclusive')

    def test_legacy_and_qwen36_splash_requirement_unchanged(self):
        legacy = splash_manifest(arms=('stock', 'experiment', 'peer'))
        self.assertEqual(self.verdict(legacy)['verdict'], 'Unavailable')
        legacy['plan']['family'] = 'qwen36'
        self.assertEqual(self.verdict(legacy)['verdict'], 'Unavailable')
        self.assertNotIn('mechanism', self.verdict(legacy))

    def test_new_plan_rejects_unbound_evidence_and_keeps_qwen36_requirement(self):
        evidence = self.root / 'invalid-splash.json'
        evidence.write_text('{"finding":"no valid Qwen3.8 Splash arm"}')
        args = q3_campaign.parse_args(['--dry-run', '--family', 'qwen38'])
        plan, _ = q3_campaign.build_plan(args, {}, 'qwen38')
        self.assertTrue(plan['splash_required'])
        args.splash_unavailable_evidence = [str(evidence)]
        with self.assertRaisesRegex(ValueError, 'invalid sweep'):
            q3_campaign.build_plan(args, {}, 'qwen38')
        args.family = ['qwen36']
        plan, _ = q3_campaign.build_plan(args, {}, 'qwen36')
        self.assertTrue(plan['splash_required'])
        self.assertNotIn('qwen38_mechanism', plan)

    def test_current_deepseek_bound_stock_full_verdict_has_no_qwen_gate(self):
        # Real deepseek4 family/case schema, without widening the family whitelist.
        source = splash_manifest(arms=('stock', 'experiment', 'peer'))
        source['schema'] = q3_campaign.SCHEMA
        source['plan']['family'] = 'deepseek4'
        source['plan'].pop('splash_required')
        source['identities']['arms']['stock'] = {'binary': {'sha256': 's' * 64}}
        source['identities_sha256'] = q3_campaign.qc.plan_sha256(source['identities'])
        campaign_path = self.root / 'campaign.json'
        campaign_path.write_text(json.dumps(source))
        record = q3_report.compute_verdict(source, campaign_sha256=mechanism.digest(campaign_path))
        record_path = self.root / 'verdict.json'
        record_path.write_text(json.dumps(record))
        args = q3_campaign.parse_args(['--dry-run', '--family', 'deepseek4', '--no-stock-arm',
                                      '--stock-record', str(record_path), '--experiment-bin', '/bin/experiment',
                                      '--experiment-commit', '2' * 40, '--workloads', 'n4'])
        plan, _ = q3_campaign.build_plan(args, {}, 'deepseek4')
        self.assertNotIn('qwen38_mechanism', plan)
        measured = splash_manifest(arms=('experiment', 'peer'))
        measured['plan'] = plan
        verdict = self.verdict(measured)
        self.assertEqual(verdict['verdict'], 'Win', verdict['reasons'])
        self.assertAlmostEqual(verdict['stock_record']['effects']['vs_stock']['ratio'], .8)
        self.assertNotIn('mechanism', verdict)

class CanonicalDriverTelemetry(unittest.TestCase):
    """Actual canonical case -> frozen override -> effective child env -> settle -> full report."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.selection = invalid_sweep(self.root)
        self.nc = self.root / 'nc.json'
        self.nc.write_text('{}')

    def plan(self, *extra):
        args = q3_campaign.parse_args(['--family', 'qwen38', '--stock-bin', '/bin/stock',
                                      '--stock-commit', '1' * 40, '--experiment-bin', '/bin/experiment',
                                      '--experiment-commit', 'c' * 40, '--qe-identity', 'qe-test',
                                      '--negative-control-receipt', str(self.nc), '--workloads', 'n4',
                                      '--output', str(self.root / 'campaign'),
                                      '--splash-unavailable-evidence', str(self.selection), *extra])
        return q3_campaign.build_plan(args, {}, 'qwen38')

    def test_default_canonical_driver_emits_and_freezes_proof_then_detects_changed_log(self):
        plan, cases = self.plan()
        self.assertEqual(plan['arms']['experiment']['case'], str(HERE / 'q5_cases/qwen38.json'))
        self.assertEqual(plan['arms']['experiment']['overrides']['LOG_FILTER'], mechanism.DEFAULT_LOG_FILTER)
        self.assertNotIn('LOG_FILTER', plan['arms']['stock']['overrides'])
        runner = FakeRunner(lambda arm, workload, call: {
            'makespan': {'stock': 100., 'experiment': 80., 'peer': 90.}[arm]})

        def emit(argv, environ, stdout, stderr):
            result = runner(argv, environ, stdout, stderr)
            case_path, output = Path(argv[3]), Path(argv[5])
            receipt_path = output / 'receipt.json'
            receipt = json.loads(receipt_path.read_text())
            receipt['artifact'].update(splash_evidence.ARTIFACT)
            receipt_path.write_text(json.dumps(receipt))
            if case_path.stem.endswith('-experiment'):
                case = json.loads(case_path.read_text())
                effective = q3_campaign.q5_case.build_plan(case, q5_calibration.builtins(output), environ)
                env = q3_campaign.q5_case.child_env(environ, effective['serve']['env'])
                self.assertEqual(env['LOG_FILTER'], mechanism.DEFAULT_LOG_FILTER)
                self.assertNotIn('MLX_PROFILE_CB', env)
                if mechanism.progress_enabled(env['LOG_FILTER']):
                    (output / 'server.stderr').write_text(
                        'DEBUG hf2q::serve::api::engine_qwen35::progress: Qwen DFlash2 stage timing '
                        'active_lanes=1 selected_rows=8 accepted_rows=6\n'
                        'DEBUG hf2q::serve::api::engine_qwen35::progress: Qwen DFlash2 verified window '
                        'slot=0 drafted=7 accepted=6\n')
                receipt_path = output / 'receipt.json'
                receipt = json.loads(receipt_path.read_text())
                receipt.update(schema='hf2q.q5.calibration-receipt.v1', family='qwen38',
                               workload=case['workload'], case={'path': str(case_path),
                                                              'sha256': mechanism.digest(case_path)})
                receipt_path.write_text(json.dumps(receipt))
            return result

        driver = q3_campaign.Driver(plan, cases, self.root / 'campaign', {}, emit, lambda: 0,
                                    lambda: canonical_identities(plan))
        manifest = driver.run()
        verdict = q3_report.compute_verdict(manifest)
        self.assertEqual(verdict['verdict'], 'Win', verdict['reasons'])
        self.assertEqual((verdict['mechanism']['accepted_rows'], verdict['mechanism']['lane_cycles']), (30, 5))
        q3_campaign.write_verdict(driver.out, manifest, [], None)
        controls_args = q3_campaign.parse_args([
            '--family', 'qwen38', '--stock-bin', '/bin/stock', '--stock-commit', '1' * 40,
            '--experiment-bin', '/bin/experiment', '--experiment-commit', 'c' * 40,
            '--qe-identity', 'qe-test', '--negative-control-receipt', str(self.nc),
            '--controls-of', str(driver.out / 'verdict.json'), '--workloads', 'n1',
            '--output', str(self.root / 'controls')])
        controls, control_cases = q3_campaign.build_plan(controls_args, {}, 'qwen38')
        self.assertEqual(controls['arms']['experiment']['overrides']['LOG_FILTER'], mechanism.DEFAULT_LOG_FILTER)
        self.assertEqual(q3_campaign.q3_identity.launch_problems(controls, canonical_identities(controls)), [])
        control_driver = q3_campaign.Driver(controls, control_cases, self.root / 'controls', {}, emit, lambda: 0,
                                            lambda: canonical_identities(controls))
        self.assertEqual(q3_report.compute_verdict(control_driver.run())['verdict'], 'ControlsOnly')
        entry = next(e for e in manifest['stages'][0]['blocks'][0]['entries'] if e['arm'] == 'experiment')
        attempt = entry['attempts'][-1]
        log = Path(attempt['output']) / 'server.stderr'
        self.assertEqual(attempt['server_stderr_sha256'], mechanism.digest(log))  # Driver.settle supplied it
        log.write_text(log.read_text() + 'changed after settlement\n')
        changed = q3_report.compute_verdict(manifest)
        self.assertEqual(changed['verdict'], 'Inconclusive')
        self.assertIn('mechanism server log changed', ' '.join(changed['reasons']))

    def test_explicit_valid_filters_are_preserved_and_disabling_filters_refuse_before_driver(self):
        for value in ('debug', 'DEBUG', 'hf2q', 'hf2q=DEBUG', 'debug,unrelated=warn,',
                      'info,' + mechanism.PROGRESS_TARGET + '=trace',
                      mechanism.PROGRESS_TARGET + '=debug,hf2q=error'):
            with self.subTest(value=value):
                plan, _ = self.plan('--experiment-var', 'LOG_FILTER=' + value)
                self.assertEqual(plan['arms']['experiment']['overrides']['LOG_FILTER'], value)
        for value in ('', 'info', 'hf2q=debug,' + mechanism.PROGRESS_TARGET + '=info'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'mechanism telemetry'):
                self.plan('--experiment-var', 'LOG_FILTER=' + value)
        with self.assertRaisesRegex(ValueError, 'MLX_PROFILE_CB unset'):
            self.plan('--experiment-var', 'NATIVE_PROFILE_CB=1')

    def test_custom_case_filter_default_is_preserved_and_missing_forwarding_refuses(self):
        path = self.root / 'custom-case.json'
        case = json.loads((HERE / 'q5_cases/qwen38.json').read_text())
        case['variables']['LOG_FILTER'] = 'debug,unrelated=error'
        path.write_text(json.dumps(case))
        plan, cases = self.plan('--hf2q-case', str(path))
        self.assertNotIn('LOG_FILTER', plan['arms']['experiment']['overrides'])
        self.assertEqual(cases['experiment']['variables']['LOG_FILTER'], 'debug,unrelated=error')
        del case['serve']['env']['LOG_FILTER']
        path.write_text(json.dumps(case))
        with self.assertRaisesRegex(ValueError, 'mechanism telemetry'):
            self.plan('--hf2q-case', str(path))

    def test_junk_empty_and_valid_sweep_selections_refuse_during_planning(self):
        original = json.loads(self.selection.read_text())
        records = []
        for row in original['configurations'].values():
            for run in row['runs']:
                record = splash_audit.run_record(run)
                record.update(q5_verdict='CalibrationPass', audit='Pass', tasks_passed=4,
                              prompt_parity='PassedByteIdentical', makespan_s=10.)
                records.append(record)
        valid_selection = splash_sweep.select(records)
        self.assertIsNotNone(valid_selection['choice'])
        for value in ('not-json', '', '{}', json.dumps(valid_selection)):
            with self.subTest(value=value[:30]), self.assertRaises(ValueError):
                self.selection.write_text(value)
                self.plan()

    def test_other_workload_and_artifact_cannot_inherit_the_exemption_before_launch(self):
        plan, _ = self.plan()
        ids = canonical_identities(plan)
        ids['arms']['experiment']['artifact']['sha256'] = 'f' * 64
        problems = q3_campaign.q3_identity.launch_problems(plan, ids)
        self.assertTrue(any('substitution candidate artifact differs' in p for p in problems), problems)
        case = json.loads((HERE / 'q5_cases/qwen38.json').read_text())
        case['workload']['id'] += '-different'
        path = self.root / 'other-workload.json'
        path.write_text(json.dumps(case))
        with self.assertRaisesRegex(ValueError, 'canonical Qwen3.8 n4 workload'):
            self.plan('--hf2q-case', str(path))

    def test_changed_raw_binding_is_rejected_before_launch_even_if_selection_stays_invalid(self):
        plan, _ = self.plan()
        selection = json.loads(self.selection.read_text())
        run = Path(next(iter(selection['configurations'].values()))['runs'][0])
        (run / 'client/response.raw').write_text('changed but still invalid')
        problems = q3_campaign.q3_identity.launch_problems(plan, canonical_identities(plan))
        self.assertTrue(any('raw evidence changed after freezing' in p for p in problems), problems)

    def test_wrong_actual_request_cannot_plan_even_when_the_selection_is_still_invalid(self):
        selection = json.loads(self.selection.read_text())
        run = Path(next(iter(selection['configurations'].values()))['runs'][0])
        path = run / 'client/agent-0/turn-0/request.json'
        request = json.loads(path.read_text())
        request['max_completion_tokens'] = 1
        path.write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError, 'raw initial request'):
            self.plan()

    def test_raw_mutation_stops_driver_before_the_next_position(self):
        plan, cases = self.plan()
        selected = json.loads(self.selection.read_text())
        run = Path(next(iter(selected['configurations'].values()))['runs'][0])
        runner = FakeRunner(lambda arm, workload, call: {'makespan': 100.})

        def mutate(argv, environ, stdout, stderr):
            result = runner(argv, environ, stdout, stderr)
            receipt_path = Path(argv[5]) / 'receipt.json'
            receipt = json.loads(receipt_path.read_text())
            receipt['artifact'].update(splash_evidence.ARTIFACT)
            receipt_path.write_text(json.dumps(receipt))
            (run / 'client/response.raw').write_text('changed after the first position')
            return result

        driver = q3_campaign.Driver(plan, cases, self.root / 'campaign', {}, mutate, lambda: 0,
                                    lambda: canonical_identities(plan))
        manifest = driver.run()
        self.assertEqual(manifest['status'], 'IdentityDrift')
        self.assertEqual(len(runner.calls), 1)
        self.assertTrue(any('splash_substitution_inputs' in p for p in manifest['identity_drift']))



if __name__ == '__main__':
    unittest.main()
