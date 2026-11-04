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

from native_peer_splash import ENGINE, MODEL_IDS, SPLASH_COMMIT
from native_peer_splash_ready import token_sha256
from qwen38_n1_verdict import loads, require

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


def run_record(run_dir):
    """One settled q5 run of a Splash case: audit, task outcome, memory and makespan (CLI below; the sweep reads it)."""
    import q2_run_audit
    import q3_metrics
    run = Path(run_dir).resolve()
    receipt = json.loads((run / 'receipt.json').read_text())
    summary = json.loads((run / 'client' / 'summary.json').read_text())
    metrics = q3_metrics.run_metrics(run, receipt, summary, 4, None)
    report, failure = q2_run_audit.check_run(receipt, run / 'client', summary, None, ENGINE, True)
    ready = summary.get('native_readiness') or {}
    return {'run': str(run), 'label': ready.get('label'), 'config': ready.get('config'),
            'q5_verdict': (receipt.get('verdict') or {}).get('verdict'),
            'audit': 'Pass' if failure is None else failure, 'tasks_passed': metrics['tasks_passed'],
            'final_source_sha256': metrics['final_source_sha256'], 'makespan_s': metrics['makespan_s'],
            'q5_memory': metrics['memory'],
            'splash_memory': {'ready': (ready.get('status') or {}).get('memory_actual'),
                              'final': ((summary.get('native_closing') or {}).get('status') or {}).get('memory_actual')},
            'trace': (report or {}).get('native_trace'), 'tokens': metrics['tokens'],
            'worst_slot_ttft_p95_s': metrics['worst_slot_ttft_p95_s']}


if __name__ == '__main__':
    print(json.dumps([run_record(path) for path in sys.argv[1:]], indent=2))
