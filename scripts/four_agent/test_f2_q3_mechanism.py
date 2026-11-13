"""Measured synthetic Q3 logs -> full verdict -> F2 record/schema/render/verification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import f2_q3_tier as tier
import f2_scoreboard_render as render
import q3_report
from q3_fake_splash import mechanism_campaign, reviewed_template_fixture
from test_f2_q3_record import quality_manifest, write_run
from test_f2_q3_tier import BOTH, qwen38, tier_line
from test_native_peer_splash import splash_manifest


class MechanismScoreboard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        reviewed_template_fixture(self)

    def fixture(self, name, rows=663, missing_log=False):
        root = self.root / name
        root.mkdir()
        campaign = mechanism_campaign(root, rows)
        if missing_log:
            entry = next(e for e in campaign['stages'][0]['blocks'][-1]['entries'] if e['arm'] == 'experiment')
            (Path(entry['attempts'][-1]['output']) / 'server.stderr').unlink()
        gates, paths = [], []
        for reference, arm in (('stock-hf2q', 'stock'), ('peer', 'peer')):
            run = root / 'qg' / (reference + '-run')
            write_run(run, campaign['identities']['arms'][arm]['binary']['sha256'])
            path = root / 'qg' / reference / 'manifest.json'
            document = quality_manifest(path, reference, campaign['identities']['arms']['experiment']['binary']['sha256'], [run])
            gates.append((reference, '$E/' + str(path.relative_to(root)), document))
            paths.append(path)
        initial = q3_report.compute_verdict(campaign)
        review = root / 'review.json'
        review.write_text(json.dumps({'schema': q3_report.PATCH_REVIEW_SCHEMA,
                                      'verdicts': {h: 'Pass' for h in initial['quality']['patch_review']['pending']}}))
        verdict = q3_report.compute_verdict(campaign, paths, review)
        directory = root / 'q3/qwen38'
        directory.mkdir(parents=True)
        (directory / 'campaign.json').write_text(json.dumps(campaign))
        (directory / 'verdict.json').write_text(json.dumps(verdict))
        refs = ('$E/q3/qwen38/verdict.json', '$E/q3/qwen38/campaign.json')
        record = tier.build_record(verdict, campaign, root, refs, gates, '$E/review.json')
        board = qwen38(BOTH)
        board['groups']['qwen38-q3-record']['q3_record'] = record
        render.validate_manifest(board)
        return root, campaign, verdict, record, board

    def cells(self, board):
        return tier_line(render.render(board, [], '0' * 64), 'Qwen3.8 27B')

    def test_full_report_floor_and_missing_log_withhold_stock_win_but_exact_boundary_and_above_pass(self):
        for name, rows, missing in (('below', 662, False), ('boundary', 663, False),
                                    ('above', 664, False), ('missing', 663, True)):
            with self.subTest(name=name):
                root, campaign, verdict, record, board = self.fixture(name, rows, missing)
                cells = self.cells(board)
                self.assertEqual([g['status'] for g in record['quality_gate']], ['Pass', 'Pass'])
                self.assertEqual(record['patch_review_pending'], [])
                tier.verify(board, root, render.resolve)
                proof = record[tier.QWEN38_MECHANISM]
                self.assertEqual(proof['contract'], campaign['plan']['qwen38_mechanism'])
                if name in ('boundary', 'above'):
                    self.assertEqual(verdict['verdict'], 'Win')
                    self.assertIn('**Win**', cells[2])
                    self.assertIn('Qwen3.8 mechanism pass', cells[6])
                    self.assertEqual(proof['ratio'], rows / 125)
                else:
                    self.assertEqual(verdict['verdict'], 'Inconclusive')
                    self.assertNotIn('**Win**', cells[2])
                    self.assertIn('Qwen3.8 mechanism', cells[2])
                    self.assertIn('Qwen3.8 mechanism ' + proof['status'], cells[6])

    def test_schema_rejects_forged_or_malformed_mechanism_results(self):
        _, _, _, record, board = self.fixture('schema')
        mutations = (lambda m: m.update(ratio=5.0), lambda m: m.update(ratio=float('nan')),
                     lambda m: m.update(ratio=float('inf')), lambda m: m.update(accepted_rows=True),
                     lambda m: m.update(runs=0), lambda m: m.update(status='unknown'),
                     lambda m: m.update(ratio=None), lambda m: m.update(reason='fabricated pass'),
                     lambda m: m['contract'].update(minimum=0.),
                     lambda m: m['binding'].update(source_commit='bad'))
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                bad = copy.deepcopy(board)
                mutate(bad['groups']['qwen38-q3-record']['q3_record'][tier.QWEN38_MECHANISM])
                with self.assertRaisesRegex(ValueError, 'mechanism'):
                    render.validate_manifest(bad)
        self.assertEqual(record[tier.QWEN38_MECHANISM]['status'], 'pass')

    def test_verify_rejects_missing_or_forged_record_fields_and_missing_verdict_result(self):
        root, campaign, verdict, _, board = self.fixture('verify')
        for fault in ('missing', 'forged'):
            bad = copy.deepcopy(board)
            record = bad['groups']['qwen38-q3-record']['q3_record']
            if fault == 'missing':
                record.pop(tier.QWEN38_MECHANISM)
            else:
                record[tier.QWEN38_MECHANISM]['binding']['source_commit'] = 'f' * 40
            with self.assertRaisesRegex(ValueError, 'qwen38_mechanism drift'):
                tier.verify(bad, root, render.resolve)
        bad_verdict = copy.deepcopy(verdict)
        bad_verdict.pop('mechanism')
        with self.assertRaisesRegex(ValueError, 'mechanism result drift'):
            tier.build_record(bad_verdict, campaign, root,
                              ('$E/q3/qwen38/verdict.json', '$E/q3/qwen38/campaign.json'), [], None)

    def test_verify_reduces_raw_evidence_instead_of_trusting_a_stale_passing_verdict(self):
        root, campaign, _, _, board = self.fixture('stale')
        entry = next(e for e in campaign['stages'][0]['blocks'][0]['entries'] if e['arm'] == 'experiment')
        path = Path(entry['attempts'][-1]['output']) / 'server.stderr'
        path.write_text(path.read_text() + 'changed after verdict\n')
        with self.assertRaisesRegex(ValueError, 'mechanism result drift'):
            tier.verify(board, root, render.resolve)

    def test_historical_plans_and_stock_cell_semantics_do_not_acquire_the_new_contract(self):
        campaign = splash_manifest(arms=('stock', 'experiment', 'peer'))
        campaign['plan']['splash_required'] = False
        verdict = q3_report.compute_verdict(campaign)
        record = tier.build_record(verdict, campaign, self.root,
                                   ('$E/q3/qwen38/verdict.json', '$E/q3/qwen38/campaign.json'), [], None)
        self.assertNotIn(tier.QWEN38_MECHANISM, record)
        self.assertIn('**Win**', self.cells(qwen38(BOTH))[2])


if __name__ == '__main__':
    unittest.main()
