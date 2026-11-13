"""Model-free full-verdict regressions for the ADR-063 Qwen3.8 substitution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import q3_campaign
import q3_mechanism as mechanism
import q3_report
from test_native_peer_splash import splash_manifest

HERE = Path(__file__).resolve().parent


class MechanismVerdict(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self, rows=663, lanes=125):
        m = splash_manifest(arms=('stock', 'experiment', 'peer'))
        p = m['plan']
        p['splash_required'] = False
        p['splash_unavailable_evidence'] = [{'path': '/recorded/invalid-splash.json', 'sha256': 'a' * 64}]
        p['qwen38_mechanism'] = dict(mechanism.CONTRACT)
        p['design'].update(stock_margin_ratio=.95, peer_margin_ratio=1.)
        m['identities']['harness'] = {'git_commit': p['declared']['experiment']['commit']}
        case = json.loads((HERE / 'q5_cases/qwen38.json').read_text())
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

    def test_below_floor_blocks_otherwise_winning_campaign(self):
        v = self.verdict(self.manifest(662))
        self.assertEqual(v['verdict'], 'Inconclusive')
        self.assertIn('Qwen3.8 mechanism below 5.304', ' '.join(v['reasons']))
        self.assertEqual(v['mechanism']['status'], 'loss')

    def test_every_position_required_and_source_case_binary_model_bound(self):
        for fault in ('missing-log', 'changed-log', 'missing-attempt', 'source', 'missing-source', 'missing-artifact',
                      'case', 'receipt', 'binary', 'artifact', 'workload', 'q5', 'zero', 'negative', 'nan',
                      'fractional', 'duplicate', 'inconsistent'):
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

    def test_new_plan_requires_explicit_evidence_to_substitute(self):
        evidence = self.root / 'invalid-splash.json'
        evidence.write_text('{"finding":"no valid Qwen3.8 Splash arm"}')
        args = q3_campaign.parse_args(['--dry-run', '--family', 'qwen38'])
        plan, _ = q3_campaign.build_plan(args, {}, 'qwen38')
        self.assertTrue(plan['splash_required'])
        args.splash_unavailable_evidence = [str(evidence)]
        plan, _ = q3_campaign.build_plan(args, {}, 'qwen38')
        self.assertFalse(plan['splash_required'])
        self.assertEqual(plan['qwen38_mechanism'], mechanism.CONTRACT)
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


if __name__ == '__main__':
    unittest.main()
