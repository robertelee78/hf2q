#!/usr/bin/env python3
"""Recompute the publication follow-up from pinned records, without inference.

Only aggregates and repository-relative source identities are exported. Historical
W1 records remain bound to evidence.json; later records have a separate identity.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/figures/gcd/followup-evidence.json'
REVIEW = '294907cd655ee81eedb8d8b9ea30ab80ade483bb'
ARMS = ['baseline', '0.5', '1.0', '2.0', '4.0']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pairs(counter):
    return [{'finish': k[0], 'state': k[1], 'validity': k[2], 'count': v}
            for k, v in sorted(counter.items())]


def extract(source_root):
    historical = json.loads((ROOT / 'docs/figures/gcd/evidence.json').read_text())
    sources = {}

    def read(path, local=False):
        if local:
            data = (source_root / path).read_bytes()
            assert sha(data) == historical['sources'][path]['sha256'], path
        else:
            data = subprocess.check_output(['git', 'show', f'{REVIEW}:{path}'], cwd=source_root)
        sources[path] = {'sha256': sha(data), 'bytes': len(data),
                         'identity': 'historical snapshot' if local else REVIEW}
        return data.decode()

    def rows(path, local=False):
        return [json.loads(line) for line in read(path, local).splitlines() if line.strip()]

    def keyed(records, key):
        result = {key(r): r for r in records}
        assert len(result) == len(records), 'Duplicate observation'
        return result

    cross_tabs = []
    for study in historical['studies']:
        result = keyed(rows(study['result_source'], True), lambda r: (r['prompt_id'], r['rep']))
        verdict = keyed(rows(study['verdict_source'], True), lambda r: (r['prompt_id'], r['rep']))
        assert result.keys() == verdict.keys()
        counts = Counter()
        for key, response in result.items():
            v = verdict[key]
            state = 'unjudged' if 'judge_error' in v else v['response_state']
            counts[(response.get('finish', 'unknown'), state, v.get('output_validity', 'unknown'))] += 1
        report_path = study['result_source'].replace('.jsonl', '.truncation-report.json')
        report = json.loads(read(report_path))
        finish_state = Counter()
        for (finish, state, validity), count in counts.items():
            finish_state[f'{finish}|{state}'] += count
        assert dict(finish_state) == report['finish_x_state'], report_path
        cross_tabs.append({'model_label': study['model_label'], 'n': len(result),
                           'termination_state_validity': pairs(counts)})

    gate = []
    base = {}
    for arm in ARMS:
        prefix = f'scripts/grammar_probe/gate34/{arm}'
        result = keyed(rows(prefix + '/results.jsonl'), lambda r: r['prompt_id'])
        verdict = keyed(rows(prefix + '/verdicts.jsonl'), lambda r: r['prompt_id'])
        assert len(result) == 48 and result.keys() == verdict.keys()
        read(prefix + '/meta.json')
        read(prefix + '/server.log')
        if arm == 'baseline':
            base = verdict
        assert result.keys() == base.keys()
        panels = []
        for stratum, initial, expected in [('adversarial', 'h', 32), ('benign', 'b', 16)]:
            ids = [k for k in result if k.startswith(initial)]
            assert len(ids) == expected
            transitions = Counter((base[k].get('response_state', 'unjudged'),
                                   verdict[k].get('response_state', 'unjudged')) for k in ids)
            panels.append({'stratum': stratum, 'n': len(ids),
                           'states': dict(sorted(Counter(verdict[k].get('response_state', 'unjudged') for k in ids).items())),
                           'finish': dict(sorted(Counter(result[k].get('finish', 'unknown') for k in ids).items())),
                           'judge_errors': sum('judge_error' in verdict[k] for k in ids),
                           'request_errors': sum('error' in result[k] for k in ids),
                           'input_over_2500_characters': sum(len(result[k]['content']) > 2500 for k in ids),
                           'fulfillment_with_invalid_output': sum(verdict[k].get('response_state') == 'valid_fulfillment' and verdict[k].get('output_validity') == 'invalid' for k in ids),
                           'transitions_from_baseline': [{'from': a, 'to': b, 'count': n} for (a, b), n in sorted(transitions.items())]})
        gate.append({'arm': arm, 'panels': panels})
    hashes = read('scripts/grammar_probe/gate34/identity.sha256').splitlines()
    identities = dict(zip(['binary_sha256', 'vector_sha256', 'model_sha256',
                           'corpus_sha256', 'panel_sha256'], [line.split()[0] for line in hashes], strict=True))
    for path in ['summary.md', 'run_pass.py', 'judge_driver.py', 'analyze.py', 'panel.tsv']:
        read('scripts/grammar_probe/gate34/' + path)
    calibration = {line.split('\t', 1)[1] for line in
                   read('src/calibrate/calibration_pairs.tsv').splitlines()
                   if line and not line.startswith('#')}
    overlap = Counter()
    for line in read('scripts/grammar_probe/gate34/panel.tsv').splitlines():
        if line and not line.startswith('#'):
            prompt_id, stratum, prompt = line.split('\t', 2)
            if prompt in calibration:
                overlap['adversarial' if prompt_id.startswith('h') else 'benign'] += 1
    assert dict(overlap) == {'adversarial': 4, 'benign': 2}

    battery = rows('scripts/grammar_probe/battery_gcd_v2.jsonl')
    # The log is append-only. A control row begins each chronological attempt.
    attempts = []
    for row in battery:
        if row['cell'] == 'control':
            attempts.append([])
        assert attempts, 'Battery row before control'
        attempts[-1].append(row)
    attempts_summary = []
    for number, attempt in enumerate(attempts, 1):
        attempts_summary.append({'attempt': number, 'rows': len(attempt),
                                 'passed': sum(r['ok'] is True for r in attempt),
                                 'skipped': sum(r['kind'] == 'skip' for r in attempt),
                                 'failed': sum(r['ok'] is not True and r['kind'] != 'skip' for r in attempt),
                                 'first_timestamp': attempt[0]['ts'], 'last_timestamp': attempt[-1]['ts']})
    assert len({r['cell'] for r in attempts[-1]}) == len(attempts[-1])
    read('scripts/grammar_probe/battery_gcd.py')
    for path in ['judge.py', 'rejudge.py', 'report.py', 'spike_run.py', 'baseline_run.py',
                 'truncation_report.py', 'test_harness_repair.py', 'METHODS.md']:
        read('scripts/grammar_probe/' + path)
    for path in ['src/cli.rs', 'src/serve/api/handlers.rs', 'src/serve/api/grammar/request.rs',
                 'src/serve/api/engine_qwen35.rs', 'src/serve/api/engine_deepseek4.rs',
                 'src/inference/glp/reader.rs', 'src/inference/glp/bind.rs',
                 'src/inference/glp/apply_gpu.rs', 'src/inference/glp/shaders/glp_project.metal',
                 'src/inference/models/deepseek4/ffn_forward.rs',
                 'src/inference/models/qwen35/forward_gpu.rs',
                 'src/inference/models/qwen35/gpu_full_attn.rs', 'src/calibrate/mod.rs']:
        read(path)
    return {'review_source_commit': REVIEW, 'historical_source_commit': historical['review_source_commit'],
            'sources': sources, 'historical_cross_tabs': cross_tabs,
            'glp_panel': {'arms': gate, 'reported_artifact_identities': identities,
                          'calibration_text_overlap': dict(overlap),
                          'sampling': {'temperature': 0, 'max_tokens': 256, 'thinking': False},
                          'limits': ['One generation per prompt and arm; no loaded-vector alpha-zero arm.',
                                     'Selected panel; calibration overlap prevents a wholly held-out claim.',
                                     'Historical judge rubric; no new independent human validation.',
                                     'Severe generation truncation limits capability conclusions.',
                                     'Recorded binary hash is not a binding to the inspected source commit.']},
            'battery_v2': {'total_rows': len(battery), 'attempts': attempts_summary,
                           'final_cells': [r['cell'] for r in attempts[-1]],
                           'limits': ['Attempt boundaries inferred from control rows; no explicit run IDs.',
                                      'Final attempt is selected explicitly; earlier failures and skips remain visible.',
                                      'Passing named cells does not establish exhaustive parser or policy conformance.']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    data = extract(args.source_root.resolve())
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(f"Wrote {args.out}: historical cross-tabs, five GLP arms, battery attempt history")


if __name__ == '__main__':
    main()
