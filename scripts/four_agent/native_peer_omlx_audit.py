#!/usr/bin/env python3
"""Post-run audit of the oMLX arm and its Q3 native_peer section.

Used by q2_run_audit (a Pass audit) and q3_report (verdict, F2 note); never by the four-agent client. oMLX
logs no response id, so each client response is bound to the server's own completion line by its counts:
`Chat completion: model=<id>, <C> tokens in <s>s (<speed>), prompt: <P>, finish_reason=<F>, max_tokens=<M>,
request_max_tokens=<R>` (server.py:5337-5343 unary, logged with the engine's finish (stop, length or tool_calls)
before a parsed call turns it into tool_calls; 7227-7236 stream, with `, stream_model_ttft=…,
stream_visible_ttft=…` and the final finish). The multiset of (transport, prompt, completion, finish, 8192, 8192)
over every client turn must equal the multiset of those lines, so a request the client did not send, or one it
sent that the server does not record, refuses. The /v1/tokenize render proof (native_peer_omlx_ready.render_proof)
is re-checked from its archive. WARNING and INFO lines of the tool-call parser (logger omlx.api.tool_calling:
"Native tool parser failed …, recovered via XML fallback", "Remapped namespaced tool call name …") are counted
per run; any count withholds the Q3 statement, and a dropped call ("Dropping …", "Dropped …") refuses.
Two server-side repairs log nothing and so cannot be counted or attributed (RECOVERY_ATTRIBUTION, ds4 precedent):
promotion of a tool call found in the reasoning (api/tool_calling.py:2509-2556) and the Qwen missing-close
recovery (api/tool_calling.py:2347-2464).
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

from native_peer_omlx import ENGINE, MODEL_IDS, OMLX_COMMIT
from native_peer_omlx_ready import token_sha256
from qwen38_n1_verdict import loads, require

DONE = re.compile(r'Chat completion: model=(\S+), (\d+) tokens in [0-9.]+s \(.*?\), prompt: (\d+), '
                  r'finish_reason=(\w+), max_tokens=(\d+), request_max_tokens=(\w+)'
                  r'(, stream_model_ttft=\S+, stream_visible_ttft=\S+)?$', re.M)
TOOL_WARNING = re.compile(r' - omlx\.api\.tool_calling - (?:WARNING|INFO) - (.*)$', re.M)
DROPPED = re.compile(r'\bDropp(ing|ed)\b')
# A unary call is logged with the engine finish before parsing: a complete envelope parses after `length` too.
FINISHES = ('stop', 'length', 'tool_calls')
RECOVERY_ATTRIBUTION = ('Unavailable:oMLX repairs some tool calls server-side without a log line: a call found in '
                        'the reasoning is promoted to a tool call (api/tool_calling.py:2509-2556) and a Qwen call '
                        'missing its closing markup is recovered (api/tool_calling.py:2347-2464); such repairs are '
                        'neither counted nor attributable to a request')


def final_finish(raw, stream):
    """The client-visible finish_reason of an archived response."""
    if not stream:
        return loads(raw)['choices'][0]['finish_reason']
    found = [choice['finish_reason'] for line in raw.split(b'\n') if line.startswith(b'data: {')
             for choice in loads(line[6:]).get('choices') or [] if 'finish_reason' in choice]
    require(len(found) == 1, 'oMLX stream without exactly one final chunk')
    return found[0]


def audit_run(summary, client_dir, turns):
    """q2_run_audit hook: completion lines, tool-parser warnings and /v1/tokenize proofs of every client request."""
    record = summary.get('native_readiness')
    require(isinstance(record, dict) and record.get('engine_kind') == ENGINE and record.get('source_revision') == OMLX_COMMIT
            and record.get('q3_family') in MODEL_IDS, 'oMLX readiness record missing from the client summary')
    model = MODEL_IDS[record['q3_family']]
    stderr = (Path(client_dir).resolve().parent / 'server.stderr').read_bytes().decode('utf-8', 'replace')
    served = Counter((bool(m.group(7)), int(m.group(3)), int(m.group(2)), m.group(4), m.group(5), m.group(6))
                     for m in DONE.finditer(stderr) if m.group(1) == model)
    require(all(m.group(1) == model for m in DONE.finditer(stderr)), 'oMLX served another model id')
    warnings = TOOL_WARNING.findall(stderr)
    require(not any(DROPPED.search(w) for w in warnings), 'oMLX silently dropped a tool call')
    totals = dict.fromkeys(('requests', 'prompt_tokens', 'cached_tokens', 'completion_tokens', 'reasoning_tokens'), 0)
    renders = (summary.get('native_closing') or {}).get('render') or {}
    for agent, receipts in sorted(turns.items()):
        proof = renders.get(agent) or {}
        rows = proof.get('turns') or []
        require(proof.get('status') == 'Observed' and [r.get('turn') for r in rows] == [r['turn'] for r in receipts],
                f'oMLX render proof missing for {agent}')
        for receipt, row in zip(receipts, rows):
            directory = Path(client_dir) / agent / f"turn-{receipt['turn']}"
            reply = (directory / 'omlx-tokenize.raw').read_bytes()
            usage, tokens = receipt['usage'], loads(reply)['tokens']
            require(hashlib.sha256(reply).hexdigest() == row['response_sha256']
                    and row['request_sha256'] == receipt['request_sha256'], 'oMLX /v1/tokenize archive differs from its proof')
            require(len(tokens) == row['count'] == usage['prompt_tokens'] and token_sha256(tokens) == row['tokens_sha256'],
                    'oMLX /v1/tokenize ids differ from the replayed usage')
            finish = final_finish((directory / 'response.bin').read_bytes(), receipt['stream'])
            logged = (finish,) if receipt['stream'] else (FINISHES if finish == 'tool_calls' else (finish,))
            key = next((k for k in ((receipt['stream'], usage['prompt_tokens'], usage['completion_tokens'], f, '8192',
                                     '8192') for f in logged) if served[k] > 0), None)
            require(key is not None, 'no oMLX completion line for a client response (counts/finish/max_tokens)')
            served[key] -= 1
            totals['requests'] += 1
            for name in ('prompt_tokens', 'completion_tokens'):
                totals[name] += usage[name]
            totals['cached_tokens'] += usage['prompt_tokens_details']['cached_tokens']
            totals['reasoning_tokens'] += usage['completion_tokens_details']['reasoning_tokens']
    require(not +served, 'oMLX served chat requests the client did not send')
    return {'engine_kind': ENGINE, 'q3_family': record['q3_family'], 'requests': totals['requests'], 'totals': totals,
            'tool_parser_warnings': len(warnings), 'tool_parser_warning_lines': warnings[:20],
            'recovery_attribution': RECOVERY_ATTRIBUTION,
            'render': '/v1/tokenize count == usage.prompt_tokens on every turn',
            'binding': 'completion lines matched by (transport, prompt, completion, finish, max_tokens); oMLX logs '
                       'no response id',
            'thinking': record.get('think_mode'), 'sampling': 'request-explicit greedy (closed request envelope)'}


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def q3_section(stage, manifest, experiment_label):
    """Task-makespan-only native comparison for verdict.json (never the same-GGUF peer verdict).

    Withheld while oMLX is unavailable, either arm has a failed task, an oMLX Pass lacks its audit, or any oMLX
    run logged tool-parser WARNING/INFO lines. Repairs oMLX does not log are disclosed as RECOVERY_ATTRIBUTION."""
    from q3_report import effect
    label = manifest['plan']['arms']['peer']['label']
    makespan = effect(stage, 'experiment', 'peer', 'makespan_s')
    entries = [e for b in stage['blocks'] for e in b['entries']]
    runs = {arm: [e['metrics'] for e in entries if e['arm'] == arm and e['outcome'] == 'Pass' and e.get('metrics')]
            for arm in ('experiment', 'peer')}
    volume = {arm: {k: _median([(m.get('tokens') or {}).get(k) for m in ms]) for k in ('prompt', 'completion')}
              for arm, ms in runs.items()}
    rate = {arm: _median([m['tokens']['completion'] / m['makespan_s'] for m in ms
                          if m.get('makespan_s') and (m.get('tokens') or {}).get('completion') is not None])
            for arm, ms in runs.items()}
    ttft = {arm: _median([m.get('worst_slot_ttft_p95_s') for m in ms]) for arm, ms in runs.items()}
    failures = {arm: [e['name'] for e in entries if e['arm'] == arm and e['outcome'] == 'Fail']
                for arm in ('experiment', 'peer')}
    audits = {e['name']: ((e.get('metrics') or {}).get('audit') or {}).get('native_trace')
              for e in entries if e['arm'] == 'peer' and e['outcome'] == 'Pass'}
    warned = sorted(n for n, a in audits.items() if a is not None and a['tool_parser_warnings'])
    unavailable = manifest['unavailable'].get('peer')
    withheld = ('oMLX arm unavailable' if unavailable else 'failed task runs' if any(failures.values())
                else 'oMLX audit missing for runs: ' + ', '.join(n for n, a in audits.items() if a is None)
                if None in audits.values()
                else 'oMLX tool-parser warnings in runs: ' + ', '.join(warned) if warned
                else 'fewer than two complete paired blocks' if makespan['improvement_percent'] is None else None)
    x = makespan['improvement_percent']
    statement = None if withheld else (
        f"{experiment_label} finishes the same coding task {abs(x):.1f}% {'sooner' if x >= 0 else 'later'} than "
        f"{label} (95% CI {makespan['improvement_ci95'][0]:.1f}..{makespan['improvement_ci95'][1]:.1f}%, "
        f"{makespan['blocks']} blocks; different artifacts, task makespan only)")
    return {'engine_kind': ENGINE, 'label': label, 'comparison': 'different-artifact native engine; task makespan only',
            'campaign': manifest['plan'].get('native_comparison'), 'statement': statement,
            'statement_withheld': withheld, 'makespan': makespan, 'token_volume_median': volume,
            'output_tokens_per_s_median': rate,
            'ttft_p95_ratio_experiment_over_omlx': None if not (ttft['experiment'] and ttft['peer'])
            else ttft['experiment'] / ttft['peer'],
            'tool_parser_warnings_per_run': {n: None if a is None else a['tool_parser_warnings'] for n, a in audits.items()},
            'recovery_attribution': RECOVERY_ATTRIBUTION, 'unavailable': unavailable, 'failures': failures}


def note(native):
    """F2 row note: the statement with its companion figures (token volume, output rate, TTFT ratio)."""
    v, rate = native['token_volume_median'], native['output_tokens_per_s_median']
    ratio = native['ttft_p95_ratio_experiment_over_omlx']
    figure = lambda x, spec='.3g': 'n/a' if x is None else format(x, spec)
    return (f"{native['statement']}. Median tokens per run (prompt/completion): experiment "
            f"{figure(v['experiment']['prompt'], '.0f')}/{figure(v['experiment']['completion'], '.0f')}, oMLX "
            f"{figure(v['peer']['prompt'], '.0f')}/{figure(v['peer']['completion'], '.0f')}; median output tokens/s: "
            f"experiment {figure(rate['experiment'])}, oMLX {figure(rate['peer'])}; worst-slot p95 TTFT ratio "
            f"experiment/oMLX {figure(ratio)}. Silent oMLX tool-call repairs: {native['recovery_attribution']}")


def run_record(run_dir):
    """One settled q5 run of an oMLX case: audit, task outcome, memory and makespan (CLI below)."""
    import q2_run_audit
    import q3_metrics
    run = Path(run_dir).resolve()
    receipt = json.loads((run / 'receipt.json').read_text())
    summary = json.loads((run / 'client' / 'summary.json').read_text())
    metrics = q3_metrics.run_metrics(run, receipt, summary, 4, None)
    report, failure = q2_run_audit.check_run(receipt, run / 'client', summary, None, ENGINE, True)
    ready = summary.get('native_readiness') or {}
    return {'run': str(run), 'label': ready.get('label'), 'audit': 'Pass' if failure is None else failure,
            'tasks_passed': metrics['tasks_passed'], 'final_source_sha256': metrics['final_source_sha256'],
            'makespan_s': metrics['makespan_s'], 'q5_memory': metrics['memory'],
            'omlx_memory': {'ready': (ready.get('status') or {}).get('model_memory_used'),
                            'final': ((summary.get('native_closing') or {}).get('status') or {}).get('model_memory_used')},
            'trace': (report or {}).get('native_trace'), 'tokens': metrics['tokens'],
            'worst_slot_ttft_p95_s': metrics['worst_slot_ttft_p95_s']}


if __name__ == '__main__':
    print(json.dumps([run_record(path) for path in sys.argv[1:]], indent=2))
