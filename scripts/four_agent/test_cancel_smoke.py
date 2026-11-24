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

    def __init__(self, log, plans, *, count_cancels=True, joint_log=True, step=0.01, recovery=None,
                 recovery_after_counter_read=False):
        super().__init__(('127.0.0.1', 0), Handler)
        self.log, self.plans, self.step = log, plans, step
        self.count_cancels, self.joint_log = count_cancels, joint_log
        self.lock = threading.Lock()
        self.active = 0
        self.recovery = recovery
        self.cancellations = 0
        self.requests = []
        self.recovery_after_counter_read = recovery_after_counter_read
        self.counter_reads_after_close = 0
        self.recovery_ready = threading.Event()
        self.recovery_done = threading.Event()

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
            if self.server.cancellations:
                with self.server.lock:
                    self.server.counter_reads_after_close += 1
                    if self.server.counter_reads_after_close >= 2:
                        self.server.recovery_ready.set()
            body = f'# TYPE hf2q_sse_cancellations counter\nhf2q_sse_cancellations {self.server.cancellations}\n'.encode()
            for slot in range(4):
                body += f'hf2q_qwen_anchor_committed{{slot="{slot}"}} 1\n'.encode()
                for kind in ('payload_image', 'logits', 'hidden', 'tokens', 'metadata'):
                    body += (f'hf2q_qwen_anchor_owned_bytes{{slot="{slot}",class="{kind}",state="pending"}} 0\n').encode()
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
                if server.recovery is not None:
                    if server.recovery_after_counter_read:
                        server.recovery_ready.wait(timeout=5)
                    with open(server.log, 'a') as handle:
                        for event in server.recovery:
                            handle.write(' INFO x: Qwen anchor cancellation recovery ' + event + '\n')
                    server.recovery_done.set()
        finally:
            with server.lock:
                server.active -= 1


class Smoke(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
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

    def test_c2_owned_restore_and_semantic_retry_are_recorded(self):
        self.log.write_text(self.log.read_text() +
                            ' INFO x: Qwen35 multi-depth anchor store configured anchor_depth=4\n')
        event = ('request_id=99 slot=0 disposition="restored" recovered_tokens=32 '
                 'restored_rank=2 performed_restore_count=1')
        code, summary = self.smoke(Fake(self.log, self.plans(), recovery=[event]),
                                   '--c2-anchor-depth', '4', '--warmup-turn', '2')
        self.assertEqual((code, summary['status']), (0, cs.PASS), summary['reasons'])
        recovery = summary['passes']['cancel']['recovery']
        self.assertEqual((recovery['disposition'], recovery['restored_rank'], recovery['restored_tokens']),
                         ('restored', 2, 32))
        self.assertFalse(recovery['preserved_live_prefix'])
        retry = summary['passes']['retry']['agents']['agent-0']
        self.assertTrue(retry['valid'])
        self.assertGreater(retry['first_semantic_token_seconds'], 0)
        self.assertEqual(set(summary['passes']['retry']['agents']), {'agent-0'})
        self.assertEqual(summary['passes']['retry']['anchor_state_after']['pending_owned_bytes'], 0)

    def test_c2_missing_recovery_is_not_inferred_from_counter_delta(self):
        self.log.write_text(self.log.read_text() +
                            ' INFO x: Qwen35 multi-depth anchor store configured anchor_depth=1\n')
        code, summary = self.smoke(Fake(self.log, self.plans()), '--c2-anchor-depth', '1',
                                   '--warmup-turn', '2')
        self.assertEqual((code, summary['status']), (1, cs.FAIL))
        self.assertTrue(any('observed 0' in reason for reason in summary['reasons']))
        self.assertNotIn('retry', summary['passes'])

    def test_qwen36_source_is_bound_to_completed_declared_repair_receipts(self):
        selected = cs.load_requests(self.recorded, 2, range(4), MODEL)
        for row in selected:
            path = self.recorded / f'agent-{row["agent"]}' / 'summary.json'
            path.write_text(json.dumps({'repair_turns': 1,
                                       'status': 'ProtocolCompleteAwaitingIndependentPatchReview',
                                       'turns': [{'turn': 2, 'request_sha256': row['recorded_sha256']}]}))
        sources = cs.repair_workload_sources(self.recorded, range(4), selected)
        self.assertEqual(sources['agent-0']['selected_request_sha256'], {'2': selected[0]['recorded_sha256']})
        (self.recorded / 'agent-0' / 'summary.json').write_text(json.dumps(
            {'repair_turns': 0, 'status': 'ProtocolCompleteAwaitingIndependentPatchReview'}))
        with self.assertRaises(cs.Invalid):
            cs.repair_workload_sources(self.recorded, range(4), selected)

    def test_changed_cancel_or_warmup_request_cannot_borrow_a_repair_receipt(self):
        for agent in range(4):
            warmup = self.recorded / f'agent-{agent}' / 'turn-1'
            warmup.mkdir()
            (warmup / 'request.json').write_bytes(
                (self.recorded / f'agent-{agent}' / 'turn-2/request.json').read_bytes())
        selected = [row for turn in (1, 2) for row in cs.load_requests(self.recorded, turn, range(4), MODEL)]
        for agent in range(4):
            (self.recorded / f'agent-{agent}/summary.json').write_text(json.dumps({
                'repair_turns': 1, 'status': 'ProtocolCompleteAwaitingIndependentPatchReview',
                'turns': [{'turn': row['turn'], 'request_sha256': row['recorded_sha256']}
                          for row in selected if row['agent'] == agent]}))
        for turn in (1, 2):
            path = self.recorded / f'agent-0/turn-{turn}/request.json'
            original = path.read_bytes()
            changed = json.loads(original)
            changed['messages'][0]['content'] += '-changed'
            path.write_text(json.dumps(changed))
            current = [row for item in (1, 2) for row in cs.load_requests(self.recorded, item, range(4), MODEL)]
            with self.subTest(turn=turn), self.assertRaises(cs.Invalid):
                cs.repair_workload_sources(self.recorded, range(4), current)
            path.write_bytes(original)

    def test_joint_warmup_cannot_rescue_an_entirely_serial_cancellation_pass(self):
        class WarmupJointOnly(Fake):
            def joint_turn(self):
                if len(self.requests) <= 4:
                    super().joint_turn()

        self.log.write_text(self.log.read_text() +
                            ' INFO x: Qwen35 multi-depth anchor store configured anchor_depth=4\n')
        recovery = 'request_id=99 slot=0 disposition=\"preserved\" recovered_tokens=32 restored_rank=-1 performed_restore_count=0'
        code, summary = self.smoke(WarmupJointOnly(self.log, self.plans(), recovery=[recovery]),
                                   '--c2-anchor-depth', '4', '--warmup-turn', '2', '--no-control')
        self.assertEqual((code, summary['status']), (1, cs.FAIL))
        self.assertIsNone(summary['joint_before_close'])
        self.assertGreater(summary['widths']['whole_run']['turns'], 0)
        self.assertEqual(summary['widths']['cancel_pass']['turns'], 0)

    def test_recovery_after_counter_ack_is_awaited_before_interval_freezes(self):
        self.log.write_text(self.log.read_text() +
                            ' INFO x: Qwen35 multi-depth anchor store configured anchor_depth=4\n')
        recovery = 'request_id=99 slot=0 disposition=\"restored\" recovered_tokens=32 restored_rank=2 performed_restore_count=1'
        fake = Fake(self.log, self.plans(), recovery=[recovery], recovery_after_counter_read=True)
        try:
            code, summary = self.smoke(fake, '--c2-anchor-depth', '4', '--warmup-turn', '2', '--no-control')
            self.assertEqual((code, summary['status']), (0, cs.PASS), summary['reasons'])
            cancel = summary['passes']['cancel']
            self.assertLessEqual(cancel['recovery']['end_offset'], cancel['log_offsets']['end'])
            self.assertEqual(cancel['recovery']['restored_tokens'], 32)
        finally:
            fake.recovery_ready.set()
            self.assertTrue(fake.recovery_done.wait(timeout=5), 'fixture recovery writer did not settle')

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

    def test_qwen_mtp_width_is_lanes(self):
        lines = self.lines(' DEBUG hf2q::serve::api::engine_qwen35::progress: Qwen joint MTP turn lanes=4 serial=0 turn_lanes=4\n')
        turn = cs.joint_turns(lines, 'qwen38')[0]
        self.assertEqual((turn['kind'], turn['width']), ('MTP', 4))

    def test_qwen_dflash_width_is_active_lanes(self):
        lines = self.lines(
            ' DEBUG hf2q::serve::api::engine_qwen35::progress: Qwen DFlash2 stage timing '
            'active_lanes=3 verify_rows=17 draft_us=20106 target_verify_us=117961 '
            'host_plan_us=321 target_commit_us=2175 draft_commit_us=2251 total_us=142823\n'
        )
        turn = cs.joint_turns(lines, 'qwen38')[0]
        self.assertEqual((turn['kind'], turn['width'], turn['serial']), ('DFlash2', 3, 0))
        self.assertEqual(turn['verify_rows'], 17)

    def test_actual_dflash_turn_scope_supersedes_stage_width_sample(self):
        turns = cs.joint_turns(self.lines(
            ' DEBUG x: Qwen DFlash2 stage timing active_lanes=3 verify_rows=17\n'
            ' DEBUG x: Qwen DFlash2 decode turn lanes=3 serial=1 turn_lanes=4 path="joint"\n'
            ' DEBUG x: Qwen joint decode turn reason="too_few_ready"\n'), 'qwen38')
        self.assertEqual(len(turns), 1)
        self.assertEqual((turns[0]['width'], turns[0]['serial'], turns[0]['turn_lanes']), (3, 1, 4))

    def test_recovery_preserve_restore_and_reset_are_distinct(self):
        prefix = ' INFO x: Qwen anchor cancellation recovery request_id=1 slot=0 '
        for disposition, tokens, rank, count in [('preserved', 17, -1, 0), ('restored', 19, 2, 1), ('reset', 0, -1, 0)]:
            line = prefix + (f'disposition="{disposition}" recovered_tokens={tokens} '
                             f'restored_rank={rank} performed_restore_count={count}')
            event = cs.cancellation_recovery([(2, line)], 1)
            self.assertEqual(event['disposition'], disposition)
            self.assertEqual(event['preserved_tokens'], tokens if disposition == 'preserved' else 0)
            self.assertEqual(event['restored_tokens'], tokens if disposition == 'restored' else 0)
            with self.assertRaises(cs.Invalid):
                cs.cancellation_recovery([(2, line), (3, line)], 1)
        with self.assertRaises(cs.Invalid):
            cs.cancellation_recovery([(2, prefix + 'disposition=reset')], 1)

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

    def test_empty_role_does_not_count_as_semantic_token(self):
        acc = cs.Accumulator(MODEL)
        self.assertFalse(acc.feed(self.frame(delta={'role': 'assistant'})))
        self.assertEqual(acc.events, 0)
        self.assertTrue(acc.feed(self.frame(delta={'reasoning_content': 'thinking'})))
        self.assertEqual(acc.events, 1)

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
