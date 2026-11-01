"""Post-run audit of the native ds4 arm: its --trace file, and the Q3 native_peer section.

Used by q2_run_audit (Pass audit) and q3_report (verdict, F2 note); never by the four-agent client.
Trace format (ds4_server.c c97964b):
- trace_begin 11983-12036 writes, under one lock, a request header, the cache decision, the raw HTTP
  body (one '\\n' appended when it lacks one) and the rendered prompt. trace_piece/trace_event
  (12039-12058) interleave later, across the four concurrent requests.
- Header token counts: effective_prompt_tokens and cached_tokens are the values usage reports
  (prompt_for_sync->len and cache_read_tokens, 13647-13656; final_response 14679).
- Server-side tool-call recovery events (12627-12639, 14170-14220, 14313-14385, 14448-14520): ds4 repairs
  an unterminated call, or appends a model-visible tool error and decodes again inside the same
  request (completion counts the hidden retry, 14562). trace_event carries no request id, so these
  are counted per run; they never invalidate a run, but they withhold the makespan statement.
"""
import hashlib
from pathlib import Path
import re
import statistics

from native_peer_ds4 import BOS, DS4_COMMIT, ENGINE, LABEL, MAX_TOKENS, MODEL_ID, THINK_MAX_PREFIX, TRACE_NAME
from qwen38_n1_verdict import require

HEADER = re.compile(r'\n===== request (\d+) \d{4}-\d\d-\d\d \d\d:\d\d:\d\d =====\nkind: (\w+)\nmodel: ([^\n]*)\n'
                    r'stream: ([01])\ntools: ([01])\nthink_mode: (\S+)\nprompt_tokens: (\d+)\n'
                    r'effective_prompt_tokens: (\d+)\ncached_tokens: (\d+)\nmax_tokens: (\d+)\n'
                    r'temperature: ([0-9.]+)\ntop_k: (-?\d+)\ntop_p: ([0-9.]+)\nmin_p: ([0-9.]+)\n'
                    r'ignore_eos: ([01])\nseed: (\d+)\nstream_include_usage: ([01])\n')
RAW_MARK, PROMPT_MARK, GEN_MARK = '\n--- raw request json ---\n', '\n--- rendered prompt ---\n', '\n--- generated text ---\n'
RECOVERY_KINDS = ('recovered complete tool call from unclosed reasoning', 'ignored orphan tool-call end marker',
                  'repaired unterminated tool call', 'unterminated tool call; continuing with model-visible tool error',
                  'invalid tool call; continuing with model-visible tool error', 'tool-error continuation appended',
                  'invalid tool call returned as assistant text')
RECOVERY = re.compile(r'\n\n--- trace: (' + '|'.join(map(re.escape, RECOVERY_KINDS)) + r')[^\n]* ---\n')


def parse_trace(text):
    """Per-request header facts, request-body sha256, and per-run recovery event counts."""
    records = []
    for match in HEADER.finditer(text):
        (rid, kind, model, stream, tools, think, prompt, effective, cached, max_tokens,
         temperature, top_k, top_p, min_p, ignore_eos, _seed, include_usage) = match.groups()
        raw = text.find(RAW_MARK, match.end())
        rendered = text.find(PROMPT_MARK, raw)
        generated = text.find(GEN_MARK, rendered)
        nxt = HEADER.search(text, match.end())
        require(0 < raw < rendered < generated and (nxt is None or nxt.start() > generated), 'ds4 trace request block')
        body = text[raw + len(RAW_MARK):rendered]
        require(body.endswith('\n'), 'ds4 trace raw request body')
        records.append({'id': int(rid), 'kind': kind, 'model': model, 'stream': stream == '1', 'tools': tools == '1',
                        'think_mode': think, 'prompt_tokens': int(prompt), 'effective_prompt_tokens': int(effective),
                        'cached_tokens': int(cached), 'max_tokens': int(max_tokens),
                        'sampling': (temperature, int(top_k), top_p, min_p), 'ignore_eos': ignore_eos == '1',
                        'include_usage': include_usage == '1',
                        'body_sha256': hashlib.sha256(body[:-1].encode('utf-8', 'surrogateescape')).hexdigest(),
                        'think_max_prefix': text.startswith(BOS + THINK_MAX_PREFIX, rendered + len(PROMPT_MARK)),
                        'ended': f'\n===== end request {rid} =====\n' in text})
    recovery = {kind: 0 for kind in RECOVERY_KINDS}
    for match in RECOVERY.finditer(text):
        recovery[match.group(1)] += 1
    return records, recovery


def audit_trace(path, model, receipts):
    """Every client request (by body sha256) ran Think Max, greedy, with the ds4 prefix and the usage it reported."""
    records, recovery = parse_trace(Path(path).read_bytes().decode('utf-8', 'surrogateescape'))
    require([r['id'] for r in records] == list(range(1, len(receipts) + 1)), 'ds4 trace request count/order')
    pending = {}
    for receipt in receipts:
        pending.setdefault(receipt.get('request_sha256'), []).append(receipt)
    for r in records:
        require(r['kind'] == 'chat' and r['model'] == model and r['tools'] and r['ended'], 'ds4 trace request identity')
        require(r['think_mode'] == 'max' and r['think_max_prefix'], 'ds4 request did not run Think Max')
        require(r['sampling'] == ('0.000', 0, '1.000', '0.000') and r['max_tokens'] == MAX_TOKENS
                and not r['ignore_eos'] and r['include_usage'] == r['stream'], 'ds4 trace sampling/output controls')
        require(pending.get(r['body_sha256']), 'ds4 trace request body is not a client request')
        receipt = pending[r['body_sha256']].pop()
        usage = receipt.get('usage')
        require(receipt.get('stream') is r['stream'], 'ds4 trace transport differs from the client request')
        require(isinstance(usage, dict) and usage['prompt_tokens'] == r['effective_prompt_tokens']
                and usage['prompt_tokens_details']['cached_tokens'] == r['cached_tokens'],
                'ds4 trace token counts differ from the replayed usage')
    require(not any(pending.values()), 'client request missing from the ds4 trace')
    return {'path': str(path), 'requests': len(records), 'think_mode': 'max', 'rendered_prefix': 'ds4 Think Max',
            'sampling': 'greedy (temperature 0.000, top_k 0, top_p 1.000, min_p 0.000)', 'max_tokens': MAX_TOKENS,
            'request_bodies': 'sha256 multiset equals the client receipts', 'usage': 'effective prompt/cached = usage',
            'recovery_events': recovery, 'recovery_total': sum(recovery.values()),
            'recovery_attribution': 'Unavailable:trace events carry no request id (per run only)'}


def audit_run(summary, client_dir, turns):
    """q2_run_audit hook: bind the client's readiness record and audit the run's --trace."""
    record = summary.get('native_readiness')
    require(isinstance(record, dict) and record.get('engine_kind') == ENGINE and record.get('source_revision') == DS4_COMMIT
            and record.get('think_mode') == 'max', 'ds4 readiness record missing from the client summary')
    path = Path(record['startup']['trace_path']).resolve()
    require(path == Path(client_dir).resolve().parent / TRACE_NAME and path.is_file(),
            'ds4 trace missing or not in this run directory')
    return audit_trace(path, MODEL_ID, [receipt for t in turns.values() for receipt in t])


def _median(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return statistics.median(values) if values else None


def q3_section(stage, manifest, experiment_label):
    """Task-makespan-only native comparison for verdict.json (never the same-GGUF peer verdict).

    The statement is withheld while ds4 is unavailable, either arm has a failed task, or any ds4 run
    shows server-side tool-call recovery (or lacks its trace audit): no makespan claim then (ADR-063 Q3)."""
    from q3_report import effect
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
    recovery = {e['name']: (((e.get('metrics') or {}).get('audit') or {}).get('native_trace') or {}).get('recovery_total')
                for e in entries if e['arm'] == 'peer' and e['outcome'] == 'Pass'}
    unavailable = manifest['unavailable'].get('peer')
    withheld = ('ds4 arm unavailable' if unavailable else 'failed task runs' if any(failures.values())
                else 'ds4 trace audit missing for runs: ' + ', '.join(n for n, c in recovery.items() if c is None)
                if None in recovery.values()
                else 'ds4 server-side tool-call recovery (repair or hidden tool-error retry) in runs: '
                     + ', '.join(n for n, c in recovery.items() if c) if any(recovery.values())
                else 'fewer than two complete paired blocks' if makespan['improvement_percent'] is None else None)
    x = makespan['improvement_percent']
    statement = None if withheld else (
        f"{experiment_label} finishes the same coding task {abs(x):.1f}% {'sooner' if x >= 0 else 'later'} than "
        f"{LABEL} (95% CI {makespan['improvement_ci95'][0]:.1f}..{makespan['improvement_ci95'][1]:.1f}%, "
        f"{makespan['blocks']} blocks; different artifacts, task makespan only)")
    return {'engine_kind': ENGINE, 'label': LABEL, 'comparison': 'different-artifact native engine; task makespan only',
            'statement': statement, 'statement_withheld': withheld, 'makespan': makespan,
            'token_volume_median': volume, 'output_tokens_per_s_median': rate,
            'ttft_p95_ratio_experiment_over_ds4': None if not (ttft['experiment'] and ttft['peer'])
            else ttft['experiment'] / ttft['peer'],
            'tool_call_recovery_per_run': recovery, 'unavailable': unavailable, 'failures': failures}


def note(native):
    """F2 row note: the statement with its companion figures (token volume, output rate, TTFT ratio)."""
    v, rate, ratio = native['token_volume_median'], native['output_tokens_per_s_median'], native['ttft_p95_ratio_experiment_over_ds4']
    figure = lambda x, spec='.3g': 'n/a' if x is None else format(x, spec)
    return (f"{native['statement']}. Median tokens per run (prompt/completion): experiment "
            f"{figure(v['experiment']['prompt'], '.0f')}/{figure(v['experiment']['completion'], '.0f')}, ds4 "
            f"{figure(v['peer']['prompt'], '.0f')}/{figure(v['peer']['completion'], '.0f')}; median output tokens/s: "
            f"experiment "
            f"{figure(rate['experiment'])}, ds4 {figure(rate['peer'])}; worst-slot p95 TTFT ratio experiment/ds4 "
            f"{figure(ratio)}")
