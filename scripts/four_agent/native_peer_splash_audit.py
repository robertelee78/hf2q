#!/usr/bin/env python3
"""Post-run audit of the Splash arm.

Used by q2_run_audit (a Pass audit) and the sweep tool; never by the four-agent client. Splash logs one status line
per request on stdout (server/diagnostics.py:30-114), each `HH:MM:SS <message>`:
- `Started · input <P> · cached <C> · <effort> · [enable_thinking true ·] thinking on|off` when the engine admits it;
- `Done · input <P> · cached <C> · [deduplicated <D> ·] output <O> · [tools ...] · [TTFT ...] · [... tok/s] ·
  [<x> tok/cycle]` when it completes, or `Cancelled · ...`.
The lines carry no request id, so each client response is bound by its counts: the multiset of (prompt, cached,
completion) over every client turn must equal the multiset of Done lines, so a request the client did not send, or
one it sent that the server does not record, refuses. Every Started line must think (thinking on). A Cancelled line,
an `Error · ` line or an engine restart (`Engine restarted`, `Engine stopped`) refuses the run: a relaunched engine
lost the run's caches. The /apply-template + /tokenize render proof (native_peer_splash_ready.render_proof) is
re-checked from its archive.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

from native_peer_splash import ARTIFACTS, ENGINE, MODEL_IDS, SLOTS, SPLASH_COMMIT, canonical, request_controls
from native_peer_splash_ready import sha256, token_sha256
from qwen38_n1_verdict import Invalid, loads, require

STAMP = r'^\d\d:\d\d:\d\d '
DONE = re.compile(STAMP + r'Done · input ([\d,]+) · cached ([\d,]+)(?: · deduplicated ([\d,]+))? · output ([\d,]+)(.*)$',
                  re.M)
STARTED = re.compile(STAMP + r'Started · input ([\d,]+) · cached ([\d,]+) · (.*)$', re.M)
CANCELLED = re.compile(STAMP + r'Cancelled · ', re.M)
REFUSED = re.compile(STAMP + r'(Error · |Engine restarted|Engine stopped)', re.M)
CYCLE = re.compile(r' · ([0-9.]+) tok/cycle')


def number(text):
    return int(text.replace(',', ''))


def audit_run(summary, client_dir, turns):
    """q2_run_audit hook: Done/Started lines and /apply-template + /tokenize proofs of every client request."""
    record = summary.get('native_readiness')
    require(isinstance(record, dict) and record.get('engine_kind') == ENGINE and record.get('source_revision') == SPLASH_COMMIT
            and record.get('q3_family') in MODEL_IDS, 'Splash readiness record missing from the client summary')
    run = Path(client_dir).resolve().parent
    stdout = (run / 'server.stdout').read_bytes().decode('utf-8', 'replace')
    stderr = (run / 'server.stderr').read_bytes().decode('utf-8', 'replace')
    require(not CANCELLED.search(stdout), 'Splash cancelled a request during the run')
    refused = REFUSED.findall(stdout + '\n' + stderr)
    require(not refused, 'Splash logged an error or engine restart: ' + ', '.join(sorted(set(refused))))
    started = STARTED.findall(stdout)
    require(all(' thinking on' in ' ' + rest for _, _, rest in started), 'a Splash request did not think')
    done = Counter((number(m.group(1)), number(m.group(2)), number(m.group(4))) for m in DONE.finditer(stdout))
    cycles = [float(c) for m in DONE.finditer(stdout) for c in CYCLE.findall(m.group(5))]
    totals = dict.fromkeys(('requests', 'prompt_tokens', 'cached_tokens', 'completion_tokens', 'reasoning_tokens'), 0)
    renders = (summary.get('native_closing') or {}).get('render') or {}
    for agent, receipts in sorted(turns.items()):
        proof = renders.get(agent) or {}
        rows = proof.get('turns') or []
        require(proof.get('status') == 'Observed' and [r.get('turn') for r in rows] == [r['turn'] for r in receipts],
                f'Splash render proof missing for {agent}')
        for receipt, row in zip(receipts, rows):
            directory = Path(client_dir) / agent / f"turn-{receipt['turn']}"
            rendered = (directory / 'splash-apply-template.raw').read_bytes()
            encoded = (directory / 'splash-tokenize.raw').read_bytes()
            usage, tokens = receipt['usage'], loads(encoded)['tokens']
            require(hashlib.sha256(rendered).hexdigest() == row['apply_template_sha256']
                    and hashlib.sha256(encoded).hexdigest() == row['tokenize_sha256']
                    and row['request_sha256'] == receipt['request_sha256'],
                    'Splash render archive differs from its proof')
            require(len(tokens) == row['count'] == usage['prompt_tokens'] and token_sha256(tokens) == row['tokens_sha256'],
                    'Splash rendered prompt ids differ from the replayed usage')
            key = (usage['prompt_tokens'], usage['prompt_tokens_details']['cached_tokens'], usage['completion_tokens'])
            require(done[key] > 0, 'no Splash Done line for a client response (input/cached/output counts)')
            done[key] -= 1
            totals['requests'] += 1
            for name in ('prompt_tokens', 'completion_tokens'):
                totals[name] += usage[name]
            totals['cached_tokens'] += usage['prompt_tokens_details']['cached_tokens']
            totals['reasoning_tokens'] += usage['completion_tokens_details']['reasoning_tokens']
    require(not +done, 'Splash completed requests the client did not send')
    require(len(started) == totals['requests'], 'Splash Started lines differ from the client requests')
    return {'engine_kind': ENGINE, 'q3_family': record['q3_family'], 'config': record.get('config'),
            'requests': totals['requests'], 'totals': totals,
            'tokens_per_cycle': {'values': cycles, 'median': sorted(cycles)[len(cycles) // 2] if cycles else None},
            'render': '/apply-template + /tokenize count == usage.prompt_tokens on every turn',
            'binding': 'Done lines matched by (input, cached, output); Splash logs no request id',
            'thinking': record.get('think_mode'), 'sampling': 'request-explicit greedy (closed request envelope)',
            'repetition_history': record.get('repetition_history')}


PARITY_DIR = 'splash-prompt-parity'   # {run}/splash-prompt-parity: splash_sweep.py parity-config's default output


def parity_inputs(run, source_hashes=None):
    """Return the complete raw input set after checking the Rust oracle chain."""
    from splash_sweep import ARCHITECTURES
    directory = Path(run).resolve() / PARITY_DIR
    config_path = directory / 'config.json'
    result_path = directory / 'splash-prompt-parity-post-oracle.json'
    config_bytes = config_path.read_bytes()
    config, result = loads(config_bytes), loads(result_path.read_bytes())
    require(isinstance(config, dict) and set(config) == {'architecture', 'template', 'turns', 'output'},
            'closed prompt-parity configuration required')
    families = [family for family, architecture in ARCHITECTURES.items()
                if architecture == config['architecture']]
    require(len(families) == 1 and Path(config['output']).resolve() == result_path,
            'prompt-parity architecture/output mismatch')
    family = families[0]
    sources = {name + '_source_sha256': Path(__file__).resolve().parents[2] / 'src/serve/api' / filename
               for name, filename in (('oracle', 'handlers_splash_prompt_parity_post_oracle_tests.rs'),
                                      ('handler', 'handlers.rs'), ('renderer', 'engine.rs'))}
    if source_hashes is None:
        source_hashes = {key: sha256(path) for key, path in sources.items()}
    require(isinstance(source_hashes, dict) and set(source_hashes) == set(sources)
            and all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)
                    for value in source_hashes.values()), 'invalid prompt-parity source binding')
    require(isinstance(result, dict) and set(result) == {
        'schema', 'hf2q_prompt_parity', 'model_inference_performed', 'architecture',
        'config_sha256', 'template_file_sha256', 'observations', *sources},
        'closed prompt-parity oracle result required')
    require(result['schema'] == 'hf2q.splash-prompt-parity-post-oracle.v1'
            and result['hf2q_prompt_parity'] == 'PassedByteIdentical'
            and result['model_inference_performed'] is False
            and result['architecture'] == config['architecture']
            and result['config_sha256'] == hashlib.sha256(config_bytes).hexdigest()
            and all(result[key] == digest for key, digest in source_hashes.items()),
            'prompt-parity oracle result/config/source mismatch')
    files = {config_path, result_path}

    def bound(spec, expected=None):
        require(isinstance(spec, dict) and set(spec) == {'path', 'sha256'}
                and isinstance(spec['path'], str) and Path(spec['path']).is_absolute()
                and isinstance(spec['sha256'], str)
                and re.fullmatch('[0-9a-f]{64}', spec['sha256']) is not None,
                'malformed prompt-parity bound input')
        path = Path(spec['path']).resolve()
        require((expected is None or
                 (path == expected.resolve() and path.parent == expected.parent.resolve()))
                and sha256(path) == spec['sha256'], 'prompt-parity bound input path/hash mismatch')
        files.add(path)
        return path

    bound(config['template'])
    require(config['template']['sha256'] == result['template_file_sha256']
            == ARTIFACTS[family]['template_sha256'], 'prompt-parity reviewed template mismatch')
    root = directory.parent / 'client'
    agents = {f'agent-{index}' for index in range(SLOTS)}
    archived = set()
    for pattern in ('agent-*/turn-*/receipt.json', 'agent-*/turn-*/response-headers.json'):
        for path in root.glob(pattern):
            require((path.parent / 'request.json').is_file(),
                    'prompt-parity standalone response record has no actual request archive')
    for path in root.glob('agent-*/turn-*/request.json'):
        agent, turn_dir = path.parent.parent.name, path.parent.name
        turn = int(turn_dir.removeprefix('turn-'))
        require(agent in agents and turn >= 0 and turn_dir == f'turn-{turn}',
                'noncanonical archived request directory')
        archived.add((agent, turn))
    turns, observations = config['turns'], result['observations']
    require(isinstance(turns, list) and turns and isinstance(observations, list)
            and len(turns) == len(observations), 'prompt-parity observation coverage mismatch')
    seen = set()
    for turn, observation in zip(turns, observations):
        require(isinstance(turn, dict) and set(turn) == {
            'agent', 'turn', 'canonical_request', 'splash_apply_template'}
            and turn['agent'] in agents and type(turn['turn']) is int and turn['turn'] >= 0,
            'malformed prompt-parity turn')
        key = turn['agent'], turn['turn']
        require(key not in seen, 'duplicate prompt-parity turn')
        seen.add(key)
        raw_directory = root / key[0] / f'turn-{key[1]}'
        canonical_path = bound(turn['canonical_request'], directory / f'{key[0]}-turn-{key[1]}.canonical-request.json')
        rendered_path = bound(turn['splash_apply_template'], raw_directory / 'splash-apply-template.raw')
        request_path = raw_directory / 'request.json'
        raw = request_path.read_bytes()
        files.add(request_path)
        request = loads(raw)
        require(isinstance(request, dict) and request.get('model') == MODEL_IDS[family],
                'prompt-parity actual request family mismatch')
        canonical_request = canonical(request, 'qwen')
        request_controls({key: value for key, value in canonical_request.items() if key != 'stream_options'}, 'qwen')
        require(loads(canonical_path.read_bytes()) == canonical_request,
                'prompt-parity canonical request differs from actual request')
        raw_sha = hashlib.sha256(raw).hexdigest()
        for name in ('response-headers.json', 'receipt.json'):
            path = raw_directory / name
            if path.exists():
                record = loads(path.read_bytes())
                require(isinstance(record, dict) and record.get('request_sha256') == raw_sha,
                        'prompt-parity raw request digest record mismatch')
                files.add(path)
        rendered = loads(rendered_path.read_bytes())
        require(isinstance(rendered, dict) and set(rendered) == {'prompt'}
                and isinstance(rendered['prompt'], str) and rendered['prompt'],
                'malformed native prompt rendering')
        prompt = rendered['prompt'].encode()
        expected = {'agent': key[0], 'turn': key[1],
                    'canonical_request_sha256': sha256(canonical_path),
                    'splash_apply_template_sha256': sha256(rendered_path),
                    'prompt_sha256': hashlib.sha256(prompt).hexdigest(),
                    'prompt_bytes': len(prompt), 'equal': True}
        require(isinstance(observation, dict) and observation == expected
                and observation['equal'] is True and type(observation['prompt_bytes']) is int,
                'prompt-parity oracle observation mismatch')
    require(seen == archived, 'prompt-parity oracle does not cover every actual archived request')
    return files


def prompt_parity(run):
    """Parity does not make a failed/summaryless client run valid."""
    directory = Path(run).resolve() / PARITY_DIR
    if not all((directory / name).is_file()
               for name in ('config.json', 'splash-prompt-parity-post-oracle.json')):
        return 'PendingPostOracle'
    try:
        parity_inputs(run)
    except (OSError, Invalid, KeyError, TypeError, ValueError, AttributeError) as exc:
        return f'invalid prompt-parity evidence: {type(exc).__name__}: {exc}'
    return 'PassedByteIdentical'


def run_record(run_dir):
    """One settled q5 run of a Splash case: audit, task outcome, memory, makespan and prompt parity (CLI below; the
    sweep reads it). A run that stopped before the client (no summary) still yields a record, which is invalid."""
    import q2_run_audit
    import q3_metrics
    run = Path(run_dir).resolve()
    receipt = q3_metrics.load_json(run / 'receipt.json')
    summary = q3_metrics.load_json(run / 'client' / 'summary.json')
    launch = q3_metrics.load_json(run / 'splash' / 'launch.json')
    metrics = q3_metrics.run_metrics(run, receipt, summary, 4, None)
    if receipt is None:
        report, failure = None, 'Q5 receipt missing'
    else:
        report, failure = q2_run_audit.check_run(receipt, run / 'client', summary, None, ENGINE, True)
    ready = (summary or {}).get('native_readiness') or {}
    return {'run': str(run), 'label': ready.get('label'),
            'config': ready.get('config') or ((launch or {}).get('spec') or {}).get('config'),
            'q5_verdict': ((receipt or {}).get('verdict') or {}).get('verdict'),
            'audit': 'Pass' if failure is None else failure, 'tasks_passed': metrics['tasks_passed'],
            'final_source_sha256': metrics['final_source_sha256'], 'makespan_s': metrics['makespan_s'],
            'q5_memory': metrics['memory'],
            'splash_memory': {'ready': (ready.get('status') or {}).get('memory_actual'),
                              'final': (((summary or {}).get('native_closing') or {}).get('status') or {}).get(
                                  'memory_actual')},
            'trace': (report or {}).get('native_trace'), 'tokens': metrics['tokens'],
            'worst_slot_ttft_p95_s': metrics['worst_slot_ttft_p95_s'], 'prompt_parity': prompt_parity(run)}


if __name__ == '__main__':
    print(json.dumps([run_record(path) for path in sys.argv[1:]], indent=2))
