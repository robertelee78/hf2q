"""Pure controls for the native ds4 --trace audit (native_peer_ds4_audit). No ds4 process.

Trace fixtures follow ds4_server.c c97964b: trace_begin 11983-12036 (header, cache decision, raw
body plus one appended newline, rendered prompt), trace_event 12047-12058, trace_finish 12060-12105,
and the recovery events at 14173, 14202, 14345, 14358, 14374, 14460, 14476 and 14517.
"""
import hashlib
from pathlib import Path
import tempfile
import unittest

import native_peer_ds4 as nd
import native_peer_ds4_audit as audit
from qwen38_n1_verdict import Invalid
from test_native_peer_ds4 import M, use

# One event of each recovery kind, formatted as trace_event writes them.
RECOVERY_EVENTS = ('recovered complete tool call from unclosed reasoning after 812 generated tokens',
                   'ignored orphan tool-call end marker after 90 generated tokens',
                   'repaired unterminated tool call (1 calls recovered)',
                   'unterminated tool call; continuing with model-visible tool error',
                   'invalid tool call; continuing with model-visible tool error',
                   'tool-error continuation appended 37 tokens',
                   'invalid tool call returned as assistant text finish=stop')


def body(rid):
    return f'{{"model":"{M}","messages":[{{"role":"user","content":"task {rid}"}}]}}'


def block(rid, stream_on, think='max', temp='0.000', min_p='0.000', max_tokens=8192, prefix=True, include=None,
          effective=1000, cached=0, raw=None):
    include = stream_on if include is None else include
    return (f'\n===== request {rid} 2026-10-31 12:00:0{rid} =====\nkind: chat\nmodel: {M}\nstream: {int(stream_on)}\n'
            f'tools: 1\nthink_mode: {think}\nprompt_tokens: 1000\neffective_prompt_tokens: {effective}\n'
            f'cached_tokens: {cached}\nmax_tokens: {max_tokens}\ntemperature: {temp}\ntop_k: 0\ntop_p: 1.000\n'
            f'min_p: {min_p}\nignore_eos: 0\nseed: 0\nstream_include_usage: {int(include)}\n'
            '\n--- cache decision ---\nlive_tokens_before: 0\ncache_source: none\ncached_tokens: 0\n'
            f'\n--- raw request json ---\n{body(rid) if raw is None else raw}\n'
            '\n--- rendered prompt ---\n' + nd.BOS + (nd.THINK_MAX_PREFIX if prefix else '') +
            'unchanged<｜User｜>task<｜Assistant｜><think>\n\n--- generated text ---\n')


def text(first=None, second=None, ends=(1, 2), events=()):
    end = ''.join(f'\n\n--- parsed message ---\nfinish: tool_calls\n\n===== end request {r} =====\n' for r in ends)
    middle = ''.join(f'\n\n--- trace: {event} ---\n\n' for event in events)
    return ((first or block(1, False)) + 'piece of 1 ' + (second or block(2, True)) + middle +
            'more of 1 piece of 2' + end)


def receipts(**changes):
    out = [{'request_sha256': hashlib.sha256(body(rid).encode()).hexdigest(), 'stream': rid == 2,
            'usage': nd.usage(use(prompt=1000, cached=0), rid)} for rid in (1, 2)]
    for index, change in changes.items():
        out[int(index[1:])].update(change)
    return out


class Trace(unittest.TestCase):
    def audit(self, trace, records=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / nd.TRACE_NAME
            path.write_text(trace)
            return audit.audit_trace(path, M, receipts() if records is None else records)

    def test_every_client_request_ran_think_max_greedy_with_the_prefix_and_its_usage(self):
        result = self.audit(text())
        self.assertEqual((result['requests'], result['recovery_total']), (2, 0))
        self.assertTrue(result['recovery_attribution'].startswith('Unavailable:'))

    def test_trace_drift_refuses(self):
        for trace in (text(block(1, False, think='high')), text(block(1, False, temp='1.000')),
                      text(block(1, False, min_p='0.050')), text(block(1, False, prefix=False)),
                      text(block(1, False, max_tokens=393216)), text(block(1, False, include=True)),
                      text(ends=(1,)), text(second=block(3, True))):
            with self.subTest(trace=trace[:90]), self.assertRaises(Invalid):
                self.audit(trace)

    def test_trace_is_bound_to_the_client_request_bodies_and_usage(self):
        for trace, records in ((text(block(1, False, raw='{"model":"other"}')), None),
                               (text(block(1, False, effective=999)), None),
                               (text(block(1, False, cached=5)), None),
                               (text(), receipts(r0={'stream': True})),
                               (text(), receipts()[:1]), (text(), receipts() + receipts()[:1]),
                               (text(), receipts(r1={'request_sha256': '0' * 64}))):
            with self.subTest(trace=trace[:60], records=records), self.assertRaises(Invalid):
                self.audit(trace, records)

    def test_each_recovery_event_kind_is_counted_without_invalidating_the_run(self):
        self.assertEqual(len(RECOVERY_EVENTS), len(audit.RECOVERY_KINDS))
        for event, kind in zip(RECOVERY_EVENTS, audit.RECOVERY_KINDS):
            with self.subTest(kind=kind):
                result = self.audit(text(events=(event,)))
                self.assertEqual((result['recovery_total'], result['recovery_events'][kind]), (1, 1))
        everything = self.audit(text(events=RECOVERY_EVENTS + ('prefill done; decode_max=8192 ctx_room=1000',)))
        self.assertEqual(everything['recovery_total'], len(RECOVERY_EVENTS))

    def test_run_audit_binds_the_trace_of_this_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp).resolve()
            (run / 'client').mkdir()
            (run / nd.TRACE_NAME).write_text(text(events=RECOVERY_EVENTS[2:3]))
            summary = {'native_readiness': {'engine_kind': 'ds4', 'source_revision': nd.DS4_COMMIT, 'think_mode': 'max',
                                            'startup': {'trace_path': str(run / nd.TRACE_NAME)}}}
            turns = {'agent-0': receipts()}
            self.assertEqual(audit.audit_run(summary, run / 'client', turns)['recovery_total'], 1)
            summary['native_readiness']['startup']['trace_path'] = str(run / 'client' / nd.TRACE_NAME)
            with self.assertRaises(Invalid):
                audit.audit_run(summary, run / 'client', turns)
            with self.assertRaises(Invalid):
                audit.audit_run({}, run / 'client', turns)


if __name__ == '__main__':
    unittest.main()
