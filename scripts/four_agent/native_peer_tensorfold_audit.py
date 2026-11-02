#!/usr/bin/env python3
"""Post-run audit of the TensorFold arm, its Q3 native_peer section, and the D2 per-run record.

Used by q2_run_audit (a Pass audit), q3_report (verdict, F2 note) and the D2 comparison (CLI below);
never by the four-agent client. Per request it binds three server-side records to the client receipt:
- the reply's runtime extras (thinking on, effort xhigh, greedy) and draft counters, already checked by
  the client's parser (native_peer_tensorfold.check_unary/check_frame) and re-read here from response.bin;
- the server's own `[tensorfold] done <reply id> prompt= cached= thinking= effort= tokens= sha= finish=
  rounds= accepted=A/D` line (server/chat.zig:582-590, printed before openai.zig:333-350 turns a parsed
  call's `stop` into `tool_calls`), keyed by the response id the client received (openai.zig:235-246);
- the /tokenize render proof (native_peer_tensorfold_ready.render_proof): count equals usage.prompt_tokens.
The DFlash2 arm must have drafted in the run. 'a slot needs X GiB ... the request waits' lines
(native/qwen27_served.zig:199) are counted: a slot that waited for memory was not resident.
"""
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

from native_peer_tensorfold import CONTROLS, EFFORT, ENGINE, TF_COMMIT, compact_json
from native_peer_tensorfold_ready import token_sha256
from qwen38_n1_verdict import loads, require

DONE = re.compile(r'^\[tensorfold\] done (chatcmpl-[0-9a-f]{32}) prompt=(\d+) cached=(\d+) thinking=(True|False) '
                  r'effort=(\S+) tokens=(\d+) sha=([0-9a-f]{12}) finish=(\S+?)(?: loop=period:\d+)? rounds=(\d+) '
                  r'accepted=(\d+)/(\d+) ', re.M)
SLOT_WAIT = re.compile(r'Qwen3\.8-27B: a slot needs ([0-9.]+) GiB and ([0-9.]+) GiB is free', re.M)
COUNTERS = ('prompt_tokens', 'cached_tokens', 'completion_tokens', 'reasoning_tokens', 'rounds', 'drafted', 'accepted')


def final_object(raw, stream):
    """(the reply object carrying the extras, streamed tool-call argument strings)."""
    if not stream:
        obj = loads(raw)
        return obj, [c['function']['arguments'] for c in obj['choices'][0]['message'].get('tool_calls') or []]
    final, args = None, {}
    for line in raw.split(b'\n'):
        if not line.startswith(b'data: {'):
            continue
        obj = loads(line[6:])
        for choice in obj.get('choices') or []:
            for part in choice['delta'].get('tool_calls') or []:
                args[part['index']] = args.get(part['index'], '') + part['function'].get('arguments', '')
            if choice['finish_reason'] is not None:
                final = obj
    require(final is not None, 'TensorFold stream without a final chunk')
    return final, list(args.values())


def audit_run(summary, client_dir, turns):
    """q2_run_audit hook: done lines, runtime extras and /tokenize proofs of every client request."""
    record = summary.get('native_readiness')
    require(isinstance(record, dict) and record.get('engine_kind') == ENGINE and record.get('source_revision') == TF_COMMIT
            and record.get('variant') in CONTROLS and record.get('think_mode') == f'thinking, reasoning_effort {EFFORT}',
            'TensorFold readiness record missing from the client summary')
    run = Path(client_dir).resolve().parent
    stdout, stderr = ((run / name).read_bytes().decode('utf-8', 'replace') for name in ('server.stdout', 'server.stderr'))
    done = {}
    for match in DONE.finditer(stdout):
        require(match.group(1) not in done, 'repeated TensorFold done line')
        done[match.group(1)] = match.groups()[1:]
    totals = dict.fromkeys(COUNTERS, 0)
    totals.update(requests=0, drafts_enabled=0)
    renders = (summary.get('native_closing') or {}).get('render') or {}
    for agent, receipts in sorted(turns.items()):
        proof = renders.get(agent) or {}
        rows = proof.get('turns') or []
        require(proof.get('status') == 'Observed' and [r.get('turn') for r in rows] == [r['turn'] for r in receipts],
                f'TensorFold render proof missing for {agent}')
        for receipt, row in zip(receipts, rows):
            directory = Path(client_dir) / agent / f"turn-{receipt['turn']}"
            reply = (directory / 'tensorfold-tokenize.raw').read_bytes()
            require(hashlib.sha256(reply).hexdigest() == row['response_sha256']
                    and row['request_sha256'] == receipt['request_sha256'], 'TensorFold /tokenize archive differs from its proof')
            usage, tokens = receipt['usage'], loads(reply)['tokens']
            require(len(tokens) == row['count'] == usage['prompt_tokens'] and token_sha256(tokens) == row['tokens_sha256'],
                    'TensorFold /tokenize ids differ from the replayed usage')
            obj, args = final_object((directory / 'response.bin').read_bytes(), receipt['stream'])
            runtime, speculative = obj['tensorfold'], obj['speculative']
            line = done.pop(obj['id'], None)
            require(line is not None, 'no TensorFold done line for a client response')
            prompt, cached, thinking, effort, count, sha, finish, rounds, accepted, drafted = line
            client_finish = obj['choices'][0]['finish_reason']
            require(int(prompt) == usage['prompt_tokens'] and int(cached) == usage['prompt_tokens_details']['cached_tokens']
                    and int(count) == usage['completion_tokens'] and thinking == 'True' and effort == EFFORT
                    and sha == runtime['token_sha'] and (finish == client_finish or (client_finish == 'tool_calls'
                                                                                     and finish == 'stop'))
                    and (int(rounds), int(accepted), int(drafted)) == (speculative['rounds'], speculative['accepted'],
                                                                       speculative['drafted']),
                    'TensorFold done line contradicts the client response')
            require(all(compact_json(a) for a in args), 'TensorFold tool-call arguments are not compact JSON')
            for key, value in (('prompt_tokens', usage['prompt_tokens']), ('cached_tokens', int(cached)),
                               ('completion_tokens', usage['completion_tokens']),
                               ('reasoning_tokens', usage['completion_tokens_details']['reasoning_tokens']),
                               ('rounds', speculative['rounds']), ('drafted', speculative['drafted']),
                               ('accepted', speculative['accepted'])):
                totals[key] += value
            totals['requests'] += 1
            totals['drafts_enabled'] += runtime['drafts'] is True
    require(not done, 'TensorFold served chat requests the client did not send')
    variant, copies = record['variant'], (record.get('startup') or {}).get('lane_copies')
    require(variant == 'none' or totals['drafted'] > 0, 'the DFlash2 arm drafted nothing in this run')
    require(variant != 'none' or copies != 'off' or totals['drafted'] == 0,
            'the no-drafter arm (lone loop=plain, lane copies=off) drafted')
    waits = SLOT_WAIT.findall(stderr)
    return {'engine_kind': ENGINE, 'variant': variant, 'requests': totals['requests'], 'totals': totals,
            'accepted_per_drafted': totals['accepted'] / totals['drafted'] if totals['drafted'] else None,
            'tokens_per_round': totals['completion_tokens'] / totals['rounds'] if totals['rounds'] else None,
            'thinking': f'every request thinking=True effort={EFFORT} (done line and runtime extras)',
            'sampling': 'greedy (runtime extras)', 'render': '/tokenize count == usage.prompt_tokens on every turn',
            'slot_memory_waits': len(waits), 'lane_copies': copies,
            'startup': record.get('startup'),
            'recovery': 'Unavailable:TensorFold closes a call that an end token cut before its closing markup '
                        '(chat.zig:313-317, docs/api.md:290-293) without a log line; malformed calls come back as '
                        'content and there is no hidden retry (docs/api.md:285-289)'}


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def q3_section(stage, manifest, experiment_label):
    """Task-makespan-only native comparison for verdict.json (never the same-GGUF peer verdict).

    Withheld while TensorFold is unavailable, either arm has a failed task, or a TensorFold Pass lacks its audit."""
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
    unavailable = manifest['unavailable'].get('peer')
    withheld = ('TensorFold arm unavailable' if unavailable else 'failed task runs' if any(failures.values())
                else 'TensorFold audit missing for runs: ' + ', '.join(n for n, a in audits.items() if a is None)
                if None in audits.values()
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
            'ttft_p95_ratio_experiment_over_tensorfold': None if not (ttft['experiment'] and ttft['peer'])
            else ttft['experiment'] / ttft['peer'],
            'speculative_per_run': {n: None if a is None else {'accepted_per_drafted': a['accepted_per_drafted'],
                                                               'tokens_per_round': a['tokens_per_round']}
                                    for n, a in audits.items()},
            'slot_memory_waits_per_run': {n: None if a is None else a['slot_memory_waits'] for n, a in audits.items()},
            'unavailable': unavailable, 'failures': failures}


def note(native):
    """F2 row note: the statement with its companion figures (token volume, output rate, TTFT ratio)."""
    v, rate = native['token_volume_median'], native['output_tokens_per_s_median']
    ratio = native['ttft_p95_ratio_experiment_over_tensorfold']
    figure = lambda x, spec='.3g': 'n/a' if x is None else format(x, spec)
    return (f"{native['statement']}. Median tokens per run (prompt/completion): experiment "
            f"{figure(v['experiment']['prompt'], '.0f')}/{figure(v['experiment']['completion'], '.0f')}, TensorFold "
            f"{figure(v['peer']['prompt'], '.0f')}/{figure(v['peer']['completion'], '.0f')}; median output tokens/s: "
            f"experiment {figure(rate['experiment'])}, TensorFold {figure(rate['peer'])}; worst-slot p95 TTFT ratio "
            f"experiment/TensorFold {figure(ratio)}")


def d2_record(run_dir):
    """ADR-063 D2 facts of one settled q5 run of either TensorFold case (T0 no drafter, T2 DFlash2)."""
    import q2_run_audit
    import q3_metrics
    run = Path(run_dir).resolve()
    receipt = json.loads((run / 'receipt.json').read_text())
    summary = json.loads((run / 'client' / 'summary.json').read_text())
    metrics = q3_metrics.run_metrics(run, receipt, summary, 4, None)
    report, failure = q2_run_audit.check_run(receipt, run / 'client', summary, None, ENGINE, True)
    ready = summary.get('native_readiness') or {}
    trace = (report or {}).get('native_trace') or {}
    return {'run': str(run), 'label': ready.get('label'), 'variant': ready.get('variant'),
            # Compare T0 and T2 only on the same workload: the q5 workload id and the client's
            # workload variant (None for the canonical files, else {id, fixture_sha256}).
            'workload': {'id': (receipt.get('workload') or {}).get('id'),
                         'variant': summary.get('workload_variant')},
            'audit': 'Pass' if failure is None else failure,
            'tasks_passed': metrics['tasks_passed'], 'final_source_sha256': metrics['final_source_sha256'],
            'makespan_s': metrics['makespan_s'], 'q5_memory': metrics['memory'],
            'tensorfold_memory': {'startup_reserve_bytes': (ready.get('startup') or {}).get('startup_reserve_bytes'),
                                  'prompt_cache': (ready.get('startup') or {}).get('prompt_cache'),
                                  'ready_bytes': (ready.get('health') or {}).get('memory_bytes'),
                                  'final_bytes': ((summary.get('native_closing') or {}).get('health') or {}).get('memory_bytes'),
                                  'slot_memory_waits': trace.get('slot_memory_waits')},
            'drafting': {'totals': trace.get('totals'), 'accepted_per_drafted': trace.get('accepted_per_drafted'),
                         'tokens_per_round': trace.get('tokens_per_round')},
            'tokens': metrics['tokens'], 'worst_slot_ttft_p95_s': metrics['worst_slot_ttft_p95_s']}


if __name__ == '__main__':
    print(json.dumps([d2_record(path) for path in sys.argv[1:]], indent=2))
