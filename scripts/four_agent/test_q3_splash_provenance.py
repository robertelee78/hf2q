"""Full-report/planning regressions for original Q5 approvals and complete raw parity evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import q3_campaign
import q3_mechanism
import q3_report
import q3_splash_evidence as evidence
from q3_fake_splash import mechanism_campaign, rebind_approval, reviewed_template_fixture

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value))


class OriginalProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        reviewed_template_fixture(self)

    def fixture(self, name):
        root = self.root / name
        root.mkdir()
        manifest = mechanism_campaign(root)
        selection = Path(manifest['plan']['splash_unavailable_evidence'][0]['path'])
        run = Path(next(iter(read(selection)['configurations'].values()))['runs'][0])
        return manifest, selection, run

    def refused(self, manifest, selection, reason=None):
        verdict = q3_report.compute_verdict(manifest)
        self.assertEqual(verdict['verdict'], 'Inconclusive', verdict['reasons'])
        self.assertEqual(verdict['mechanism']['status'], 'unresolved')
        if reason:
            self.assertIn(reason, verdict['mechanism']['reason'])
        args = q3_campaign.parse_args(['--dry-run', '--family', 'qwen38',
                                      '--splash-unavailable-evidence', str(selection)])
        with self.assertRaises((ValueError, OSError)):
            q3_campaign.build_plan(args, {}, 'qwen38')

    def test_rewriting_obsolete_approval_sources_cannot_launder_original_launch(self):
        manifest, selection, run = self.fixture('rewrite')
        path = run / 'client-approval.json'
        packet = read(path)
        current = copy.deepcopy(packet)
        packet['sources'][str(HERE / 'qwen38_n1_smoke.py')] = '2bb07db9bd67a1ea5feeb4892cf08485f005a1d8127497d4971ac9bf1db31e0c'
        packet['sources'][str(HERE / 'qwen_n4_live_smoke.py')] = 'cf137e169e087fcee2f303ccddbbe887c8f9a0a3b7b1463c0ab36969f992a33e'
        write(path, packet)
        rebind_approval(run)  # Construct the original historical-source synthetic launch.
        self.refused(manifest, selection, 'workload source drift')
        write(path, current)  # Attack: rewrite only the packet, keeping the original receipt immutable.
        self.refused(manifest, selection, 'original launch approval path/hash mismatch')

    def test_original_receipt_input_and_approval_path_are_required(self):
        for fault in ('source-input', 'approval-path', 'approval-hash', 'missing-flag'):
            with self.subTest(fault=fault):
                manifest, selection, run = self.fixture(fault)
                path = run / 'receipt.json'
                receipt = read(path)
                if fault == 'source-input':
                    receipt['inputs'][receipt['client_argv'][4]] = 'f' * 64
                elif fault == 'approval-path':
                    clone = run / 'other-approval.json'
                    clone.write_bytes((run / 'client-approval.json').read_bytes())
                    receipt['client_argv'][receipt['client_argv'].index('--approval') + 1] = str(clone)
                elif fault == 'approval-hash':
                    receipt['client_argv'][receipt['client_argv'].index('--approval-sha256') + 1] = 'f' * 64
                else:
                    index = receipt['client_argv'].index('--approval-sha256')
                    del receipt['client_argv'][index:index+2]
                write(path, receipt)
                self.refused(manifest, selection, 'original')

    def test_raw_parity_inputs_observations_sources_and_failed_turn_coverage_fail_closed(self):
        faults = ('missing-render', 'changed-render', 'missing-canonical', 'canonical-not-wire',
                  'missing-template', 'unreviewed-template', 'marker-only', 'observation-hash',
                  'observation-count', 'oracle-source', 'oracle-inference', 'duplicate-turn',
                  'omitted-turn', 'uncovered-failed-request')
        for fault in faults:
            with self.subTest(fault=fault):
                manifest, selection, run = self.fixture(fault)
                directory = run / 'splash-prompt-parity'
                config_path = directory / 'config.json'
                result_path = directory / 'splash-prompt-parity-post-oracle.json'
                config, result = read(config_path), read(result_path)
                first = config['turns'][0]
                rendered = Path(first['splash_apply_template']['path'])
                canonical = Path(first['canonical_request']['path'])
                if fault == 'missing-render':
                    rendered.unlink()
                elif fault == 'changed-render':
                    write(rendered, {'prompt': 'different bytes'})
                elif fault == 'missing-canonical':
                    canonical.unlink()
                elif fault == 'canonical-not-wire':
                    request = read(canonical)
                    request['messages'][-1]['content'] = 'not the archived request'
                    write(canonical, request)
                    first['canonical_request']['sha256'] = q3_mechanism.digest(canonical)
                    result['observations'][0]['canonical_request_sha256'] = q3_mechanism.digest(canonical)
                elif fault == 'missing-template':
                    Path(config['template']['path']).unlink()
                elif fault == 'unreviewed-template':
                    template = Path(config['template']['path'])
                    template.write_text('different reviewed template claim')
                    config['template']['sha256'] = q3_mechanism.digest(template)
                    result['template_file_sha256'] = q3_mechanism.digest(template)
                elif fault == 'marker-only':
                    result = {key: result[key] for key in ('schema', 'hf2q_prompt_parity', 'config_sha256')}
                elif fault == 'observation-hash':
                    result['observations'][0]['prompt_sha256'] = 'f' * 64
                elif fault == 'observation-count':
                    result['observations'].pop()
                elif fault == 'oracle-source':
                    result['renderer_source_sha256'] = 'f' * 64
                elif fault == 'oracle-inference':
                    result['model_inference_performed'] = True
                elif fault == 'duplicate-turn':
                    config['turns'].append(copy.deepcopy(first))
                    result['observations'].append(copy.deepcopy(result['observations'][0]))
                elif fault == 'omitted-turn':
                    config['turns'].pop()
                    result['observations'].pop()
                else:
                    raw = run / 'client/agent-0/turn-1/request.json'
                    raw.parent.mkdir()
                    write(raw, read(run / 'client/agent-0/turn-0/request.json'))
                write(config_path, config)
                result['config_sha256'] = q3_mechanism.digest(config_path)
                write(result_path, result)
                # Rehashed config/claims must not make an altered original input or missing turn legitimate.
                self.refused(manifest, selection)


if __name__ == '__main__':
    unittest.main()
