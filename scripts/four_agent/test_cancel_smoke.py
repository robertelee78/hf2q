"""S3 cancel smoke controls: a loopback fake server streams SSE and writes joint-turn log lines; no model."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

import cancel_smoke as cs
import q5_case

HERE = Path(__file__).resolve().parent

MODEL = 'Fake Model'
QWEN_TURN = '\x1b[2m DEBUG\x1b[0m hf2q::serve::api::engine_qwen35::progress: Qwen joint decode turn lanes={n} rows={n} serial=0\n'


class Fake(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, log, plans, *, count_cancels=True, joint_log=True, step=0.01):
        super().__init__(('127.0.0.1', 0), Handler)
        self.log, self.plans, self.step = log, plans, step
        self.count_cancels, self.joint_log = count_cancels, joint_log
        self.lock = threading.Lock()
        self.active = 0
        self.cancellations = 0
        self.requests = []

    def joint_turn(self):
        with self.lock:
            if self.joint_log and self.active:
                with open(self.log, 'a') as handle:
                    handle.write(QWEN_TURN.format(n=self.active))


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == '/readyz':
            body = b'{"ready":true}'
        elif self.path == '/metrics':
            body = f'# TYPE hf2q_sse_cancellations counter\nhf2q_sse_cancellations {self.server.cancellations}\n'.encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def chunk(self, obj):
        data = obj if isinstance(obj, str) else json.dumps(obj)
        payload = f'data: {data}\n\n'.encode()
        self.wfile.write(b'%x\r\n%s\r\n' % (len(payload), payload))

    def frame(self, delta=None, finish=None, usage=None):
        obj = {'id': 'chatcmpl-1', 'model': MODEL,
               'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}]}
        if usage is not None:
            obj['usage'] = usage
        self.chunk(obj)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        server = self.server
        agent = request['messages'][0]['content']
        server.requests.append(request)
        plan = server.plans[agent]
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Transfer-Encoding', 'chunked')
        self.send_header('Connection', 'close')
        self.end_headers()
        with server.lock:
            server.active += 1
        try:
            self.frame({'role': 'assistant'})
            for index in range(plan['tokens']):
                time.sleep(server.step)
                server.joint_turn()
                key = 'reasoning_content' if index < plan['tokens'] // 2 else 'content'
                self.frame({key: f'{agent}:{index};'})
            if plan.get('tool'):
                self.frame({'tool_calls': [{'index': 0, 'id': 'call_1', 'type': 'function',
                                            'function': {'name': 'write_file', 'arguments': ''}}]})
                self.frame({'tool_calls': [{'index': 0, 'function': {'arguments': plan['tool']}}]})
            finish = plan.get('finish', 'tool_calls' if plan.get('tool') else 'stop')
            self.frame(finish=finish, usage={'prompt_tokens': 10, 'completion_tokens': plan['tokens'],
                                             'total_tokens': 10 + plan['tokens']})
            self.chunk('[DONE]')
            self.wfile.write(b'0\r\n\r\n')
        except (BrokenPipeError, ConnectionResetError):
            if server.count_cancels:
                with server.lock:
                    server.cancellations += 1
        finally:
            with server.lock:
                server.active -= 1


class Smoke(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = self.tmp / 'server.stderr'
        self.log.write_text(' INFO hf2q::serve: listening\n')
        self.recorded = self.tmp / 'client'
        for agent in range(4):
            turn = self.recorded / f'agent-{agent}' / 'turn-2'
            turn.mkdir(parents=True)
            (turn / 'request.json').write_text(json.dumps(
                {'model': MODEL, 'messages': [{'role': 'user', 'content': f'agent-{agent}'}],
                 'stream': False, 'tool_choice': 'auto', 'max_completion_tokens': 8192}))

    def plans(self, **changes):
        plans = {'agent-0': {'tokens': 200}, 'agent-1': {'tokens': 60, 'tool': '{"path":"src/ports.rs"}'},
                 'agent-2': {'tokens': 60}, 'agent-3': {'tokens': 60}}
        for agent, change in changes.items():
            plans[agent.replace('_', '-')] = {**plans[agent.replace('_', '-')], **change}
        return plans

    def smoke(self, fake, *extra, after=8):
        thread = threading.Thread(target=fake.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(fake.server_close)
        self.addCleanup(fake.shutdown)
        out = self.tmp / f'out-{len(list(self.tmp.glob("out-*")))}'
        with contextlib.redirect_stdout(io.StringIO()):
            code = cs.main(['run', '--base-url', f'http://127.0.0.1:{fake.server_address[1]}/v1', '--model', MODEL,
                            '--family', 'qwen38', '--requests', str(self.recorded), '--server-log', str(self.log),
                            '--cancel-after-events', str(after), '--counter-wait-seconds', '3',
                            '--output', str(out), *extra])
        return code, json.loads((out / 'summary.json').read_text())

    def test_mid_decode_close_passes_and_survivors_match_control(self):
        fake = Fake(self.log, self.plans())
        code, summary = self.smoke(fake)
        self.assertEqual((code, summary['status']), (0, cs.PASS), summary['reasons'])
        cancel = summary['passes']['cancel']
        victim = cancel['agents']['agent-0']
        self.assertTrue(victim['cancelled_by_client'])
        self.assertGreaterEqual(victim['events_at_close'], 8)
        self.assertLess(victim['semantic_events'], 200, 'the victim stream was really cut short')
        for agent, events in cancel['close']['survivor_events'].items():
            self.assertTrue(1 <= events < 60, (agent, events))
        self.assertEqual(cancel['cancel_counter']['delta'], 1)
        self.assertEqual(summary['passes']['control']['cancel_counter']['delta'], 0)
        self.assertEqual(summary['joint_before_close']['width'], 4)
        self.assertIsNotNone(summary['joint_after_close_first'])
        self.assertIn('4', summary['widths']['cancel_pass']['turns_by_width'])
        # The server notices the close at its next write, then the step narrows to the three survivors.
        self.assertIn('3', summary['widths']['cancel_pass_after_close']['turns_by_width'])
        self.assertGreater(summary['widths']['control_pass']['turns_by_width'].get('4', 0), 0)
        comparison = summary['control_comparison']
        self.assertTrue(comparison['survivors_exact'])
        self.assertTrue(comparison['victim_partial_is_prefix_of_control'])
        self.assertEqual(summary['passes']['control']['agents']['agent-1']['output']['tool_calls'],
                         [{'name': 'write_file', 'arguments': '{"path":"src/ports.rs"}'}])
        self.assertTrue(all(r['stream'] is True and r['stream_options'] == {'include_usage': True}
                            for r in fake.requests), 'recorded unary turns are replayed as streams')

    def test_survivor_error_finish_fails(self):
        code, summary = self.smoke(Fake(self.log, self.plans(agent_2={'finish': 'error'})))
        self.assertEqual((code, summary['status']), (1, cs.FAIL))
        self.assertTrue(any('agent-2' in r and 'finish_reason error' in r for r in summary['reasons']), summary['reasons'])

    def test_malformed_tool_arguments_fail(self):
        code, summary = self.smoke(Fake(self.log, self.plans(agent_1={'tool': '{"path": '})))
        self.assertEqual(summary['status'], cs.FAIL)
        self.assertTrue(any('agent-1' in r and 'not JSON' in r for r in summary['reasons']), summary['reasons'])

    def test_survivors_finishing_before_k_is_invalid_not_pass(self):
        plans = self.plans(agent_1={'tokens': 3}, agent_2={'tokens': 3}, agent_3={'tokens': 3})
        code, summary = self.smoke(Fake(self.log, plans), '--no-control', after=50)
        self.assertEqual((code, summary['status']), (1, cs.INVALID), summary['reasons'])
        self.assertIsNone(summary['passes']['cancel']['close'])
        self.assertEqual(summary['passes']['cancel']['cancel_counter']['delta'], 0)

    def test_uncounted_cancellation_fails(self):
        code, summary = self.smoke(Fake(self.log, self.plans(), count_cancels=False), '--no-control')
        self.assertEqual(summary['status'], cs.FAIL)
        self.assertIn('hf2q_sse_cancellations delta 0 != 1', summary['reasons'])

    def test_log_without_joint_turns_fails(self):
        code, summary = self.smoke(Fake(self.log, self.plans(), joint_log=False), '--no-control')
        self.assertEqual(summary['status'], cs.FAIL)
        self.assertIn('no joint decode turn of width >= 2 before the close', summary['reasons'])

    def test_divergent_control_is_reported_not_gated(self):
        fake = Fake(self.log, self.plans())
        original = Handler.frame

        def frame(handler, delta=None, finish=None, usage=None):
            if delta and 'content' in delta and len(handler.server.requests) > 4:
                delta = {'content': delta['content'] + 'x'}
            original(handler, delta, finish, usage)
        Handler.frame = frame
        self.addCleanup(setattr, Handler, 'frame', original)
        code, summary = self.smoke(fake)
        self.assertEqual(summary['status'], cs.PASS, summary['reasons'])
        self.assertFalse(summary['control_comparison']['survivors_exact'])
        self.assertIn('content', summary['control_comparison']['survivors']['agent-2']['first_difference'])

    def test_model_mismatch_refuses_before_any_request(self):
        fake = Fake(self.log, self.plans())
        thread = threading.Thread(target=fake.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(fake.server_close)
        self.addCleanup(fake.shutdown)
        with contextlib.redirect_stderr(io.StringIO()):
            code = cs.main(['run', '--base-url', f'http://127.0.0.1:{fake.server_address[1]}/v1', '--model', 'Other',
                            '--family', 'qwen38', '--requests', str(self.recorded), '--server-log', str(self.log),
                            '--output', str(self.tmp / 'refused')])
        self.assertEqual(code, 2)
        self.assertEqual(fake.requests, [])
        self.assertFalse((self.tmp / 'refused').exists())


class Cases(unittest.TestCase):
    BUILTINS = {'REPO': str(HERE.parent.parent), 'OUT': '/out/run', 'PYTHON': '/usr/bin/python3',
                'PYTHON_SHA256': 'a' * 64, 'RUN_ID': 'q5-run-0123456789', 'NONCE32': 'b' * 32}

    def plan(self, name, environ):
        return q5_case.build_plan(q5_case.loads((HERE / 'q5_cases' / name).read_bytes()), self.BUILTINS, environ)

    def test_cancel_cases_drive_this_client_with_the_trace_enabled(self):
        for family, engine in (('deepseek4', 'engine_deepseek4'), ('qwen38', 'engine_qwen35')):
            self.assertFalse(self.plan(f'cancel-{family}.json', {})['complete'], 'RECORDED_CLIENT is required')
            plan = self.plan(f'cancel-{family}.json', {'HF2Q_Q5_RECORDED_CLIENT': '/rec/client'})
            canonical = self.plan(f'{family}.json', {f'HF2Q_Q5_{k}': '/x' for k in (
                'CARGO', 'TOOLCHAIN', 'SANDBOX_PROOF', 'SANDBOX_PROOF_SHA256', 'CLIENT_APPROVAL_TEMPLATE')})
            self.assertTrue(plan['complete'], family)
            self.assertEqual((plan['serve']['argv'], plan['readiness'], plan['port'], plan['artifact']),
                             (canonical['serve']['argv'], canonical['readiness'], canonical['port'],
                              canonical['artifact']), 'same server as the canonical case')
            self.assertIn(f'hf2q::serve::api::{engine}::progress=debug', plan['serve']['env']['LOG_FILTER'])
            self.assertEqual(canonical['serve']['env']['LOG_FILTER'], '', 'canonical runs keep -v')
            argv = q5_case.expand(plan['client']['argv'], {'SERVER_PID': 42})
            self.assertEqual(Path(argv[4]), HERE / 'cancel_smoke.py')
            args = cs.parse_args(argv[5:])  # the case's argv must be what this CLI accepts
            self.assertEqual((args.family, args.requests, args.server_log, args.output, args.server_pid),
                             (family, Path('/rec/client'), Path('/out/run/server.stderr'),
                              Path('/out/run/client'), 42))
            self.assertEqual(plan['client']['summary'], str(args.output / 'summary.json'))
            self.assertEqual(plan['client']['success_status'], cs.PASS)
            self.assertNotIn('approval', plan['client'])

    def test_launchers_replace_verbose_flag_only_when_log_filter_is_set(self):
        # hf2q ignores RUST_LOG under -v; the filter must replace -v, not sit beside it.
        for name, args in (('serve_deepseek4_opencode.sh', 'VERBOSITY'), ('serve_qwen36_opencode.sh', 'HF2Q_SERVE_ARGS')):
            text = (HERE.parent / name).read_text()
            self.assertIn('if [[ -n "${LOG_FILTER:-}" ]]; then\n    export RUST_LOG="$LOG_FILTER"\n'
                          f'    {args}=(serve)\nelse\n    {args}=(-v serve)\nfi\n', text)
            self.assertNotIn('"$HF2Q_BIN" -v serve', text)
        self.assertIn('exec "$script_dir/serve_qwen36_opencode.sh"', (HERE.parent / 'serve_qwen38_opencode.sh').read_text())


class Parsing(unittest.TestCase):
    def lines(self, text):
        return [(index, line) for index, line in enumerate(text.splitlines(), 1)]

    def test_deepseek_turns_weight_budget_and_ignore_other_engines(self):
        lines = self.lines(
            ' DEBUG hf2q::serve::api::engine_deepseek4::progress: DeepSeek-V4 joint decode turn lanes=4 serial=0 budget=8\n'
            ' DEBUG hf2q::serve::api::engine_deepseek4::progress: DeepSeek-V4 joint decode turn lanes=3 serial=1 budget=8\n'
            ' INFO hf2q::serve::api::engine_deepseek4::progress: DeepSeek-V4 joint decode engaged lanes=4\n'
            ' DEBUG hf2q::serve::api::engine_qwen35::progress: Qwen joint decode turn lanes=4 rows=4 serial=0\n')
        turns = cs.joint_turns(lines, 'deepseek4')
        self.assertEqual([t['width'] for t in turns], [4, 3])
        result = cs.histogram(turns, 'deepseek4')
        self.assertEqual(result['turns_by_width'], {'3': 1, '4': 1})
        self.assertEqual(result['budget_steps_by_width_upper_bound'], {'3': 8, '4': 8})
        self.assertEqual((result['turns_with_serial_lanes'], result['serial_lanes_total']), (1, 1))

    def test_qwen_width_is_rows_not_lanes(self):
        lines = self.lines(' DEBUG hf2q::serve::api::engine_qwen35::progress: Qwen joint decode turn lanes=4 rows=2 serial=1\n')
        self.assertEqual(cs.joint_turns(lines, 'qwen38')[0]['width'], 2)
        with self.assertRaises(cs.Invalid):
            cs.joint_turns(self.lines(' DEBUG x: Qwen joint decode turn lanes=4 serial=0\n'), 'qwen38')

    def test_log_lines_keep_only_complete_lines_in_range(self):
        with tempfile.NamedTemporaryFile('wb', delete=False) as handle:
            handle.write(b'a\nb\npartial')
        self.addCleanup(Path(handle.name).unlink)
        self.assertEqual(cs.log_lines(handle.name, 2, 11), [(4, 'b')])

    def test_problem_lines(self):
        self.assertEqual(cs.problem_lines(self.lines(' INFO ok\n ERROR boom\n WARN engine poisoned\n')),
                         [' ERROR boom', ' WARN engine poisoned'])
        # A healthy progress line reports `cache_poisoned=false`; that is not a poison event.
        quiet = ' INFO prefix reuse rejected cache_poisoned=false matrix_prefill_forced=false\n'
        self.assertEqual(cs.problem_lines(self.lines(quiet + ' WARN cache_poisoned=true\n')),
                         [' WARN cache_poisoned=true'])


class Accumulating(unittest.TestCase):
    def frame(self, **choice):
        return json.dumps({'id': 'x', 'model': MODEL, 'choices': [{'index': 0, 'delta': {}, **choice}]})

    def test_usage_contradiction_and_data_after_done(self):
        acc = cs.Accumulator(MODEL)
        acc.feed(self.frame(delta={'content': 'hi'}))
        with self.assertRaises(cs.Invalid):
            acc.feed(json.dumps({'id': 'x', 'model': MODEL, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                                 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 3}}))
        acc = cs.Accumulator(MODEL)
        acc.feed(self.frame(finish_reason='stop'))
        acc.feed(json.dumps({'id': 'x', 'model': MODEL, 'choices': [],
                             'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}))
        acc.feed('[DONE]')
        acc.validate()
        with self.assertRaises(cs.Invalid):
            acc.feed('[DONE]')

    def test_tool_finish_requires_calls(self):
        acc = cs.Accumulator(MODEL)
        acc.feed(self.frame(finish_reason='tool_calls'))
        acc.feed(json.dumps({'id': 'x', 'model': MODEL, 'choices': [],
                             'usage': {'prompt_tokens': 1, 'completion_tokens': 0, 'total_tokens': 1}}))
        acc.feed('[DONE]')
        with self.assertRaises(cs.Invalid):
            acc.validate()


if __name__ == '__main__':
    unittest.main()
