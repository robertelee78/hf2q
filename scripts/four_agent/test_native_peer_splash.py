"""Pure controls for the Splash arm (engine_kind 'splash'): request translation, refusals, reply envelopes, binding,
launch inputs, the model.json assembly check, the Done-line audit, the Splash verdict wiring and the sweep.

No HTTP server, model or Splash process. Reply fixtures follow incoai/splash 031a25a6 byte formats: chat.completion
(server/api_shapes.py:1147-1163), chunks (server/server.py:1455-1515, server/api_shapes.py:1186-1223, tool deltas
server/output.py:265-305), usage (server/metrics.py:235-242), call ids call_<public id>_<n> (output.py:294).
"""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import native_peer_splash as sp
import native_peer_splash_audit as audit
import native_peer_splash_ready as ready
import native_peers
import q3_report
import q3_stats
import splash_sweep
from coding_family import ADAPTER_KINDS, NATIVE_KINDS, SAME_ARTIFACT_KINDS, preflight as family_preflight, \
    validate_binding, validate_selection
from q2_certify import replay_response
from qwen38_n1_smoke import PREFLIGHT, build_request, exchange
from qwen38_n1_verdict import SSE, Invalid, unary, wire
from qwen_n4_live_smoke import preflight as n4_preflight, validate_preflight

HERE = Path(__file__).resolve().parent
CREATED = 1767225600
RID = '0123456789abcdef0123456789abcdef'
CALL, CALL2 = f'call_{RID}_0', 'call_fedcba9876543210fedcba9876543210_0'
M = sp.MODEL_IDS['qwen38']
S0 = sp.preset('s0', 'qwen38')


def schemas():
    return [{'type': 'function', 'function': {'name': name, 'strict': False,
             'parameters': {'type': 'object', 'properties': {}}}} for name in ('read_file', 'write_file', 'run_tests')]


def chain():
    return [{'role': 'system', 'content': 'unchanged'}, {'role': 'user', 'content': 'same coding task'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
             'tool_calls': [{'id': CALL, 'type': 'function',
                             'function': {'name': 'read_file', 'arguments': '{"path":"src/ports.rs"}'}}]},
            {'role': 'tool', 'tool_call_id': CALL, 'content': 'fn main() {}'}]


def native(turn, model=M, messages=None):
    return build_request(model, chain() if messages is None else messages, schemas(), turn, 'qwen', 'splash')


def use(prompt=1000, completion=40, cached=0, reasoning=12):
    return {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion,
            'prompt_tokens_details': {'cached_tokens': cached}, 'completion_tokens_details': {'reasoning_tokens': reasoning}}


METRICS = {'prefill': {'tokens': 1000}, 'decode': {'tokens': 30},
           'request_latency': {'start_to_first_token_ms': 120.0, 'ttft_ms': 130.0}, 'cache': {
               'status': 'miss', 'matched_tokens': 0, 'deduplicated_tokens': 0, 'lane': 0},
           'speculation': {'drafted_tokens': 21, 'accepted_draft_tokens': 14, 'cycles': 3}}
TIMINGS = {'prompt_n': 1000, 'prompt_ms': 120.0, 'prompt_per_second': 8333.3, 'predicted_n': 40, 'predicted_ms': 600.0,
           'predicted_per_second': 50.0, 'cache_n': 0, 'draft_n': 21, 'draft_n_accepted': 14}


def completion_body(message, finish, model=M, **changes):
    body = {'id': f'chatcmpl-{RID}', 'object': 'chat.completion', 'created': CREATED, 'model': model,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': use(),
            'metrics': copy.deepcopy(METRICS), 'timings': dict(TIMINGS), **changes}
    return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()


def call_message(name='read_file', args='{"path": "src/ports.rs"}', call_id=CALL, **extra):
    return {'role': 'assistant', 'content': None, 'reasoning_content': 'plan',
            'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': args}}], **extra}


def frame(obj):
    return b'data: ' + json.dumps(obj, separators=(',', ':')).encode() + b'\n\n'


def chunk(delta, finish=None, model=M, **extra):
    return frame({'id': 'chatcmpl-fedcba9876543210fedcba9876543210', 'object': 'chat.completion.chunk',
                  'created': CREATED, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}],
                  **extra})


def usage_chunk(value=None, model=M):
    return frame({'id': 'chatcmpl-fedcba9876543210fedcba9876543210', 'object': 'chat.completion.chunk',
                  'created': CREATED, 'model': model, 'choices': [],
                  'usage': use(cached=768) if value is None else value, 'metrics': copy.deepcopy(METRICS)})


def stream(extra=b'', final=None, usage_frame=None, model=M):
    opening = {'index': 0, 'id': CALL2, 'type': 'function', 'function': {'name': 'read_file'}}
    return b''.join([chunk({'role': 'assistant', 'content': ''}, model=model),
                     chunk({'reasoning_content': 'Read the tests.'}, model=model), chunk({}, model=model), extra,
                     chunk({'tool_calls': [opening]}, model=model),
                     chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '{"path": "tests'}}]}, model=model),
                     chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '/ports.rs"}'}}]}, model=model),
                     chunk({}, 'tool_calls', model=model, timings=dict(TIMINGS)) if final is None else final,
                     usage_chunk(model=model) if usage_frame is None else usage_frame, b'data: [DONE]\n\n'])


class Translation(unittest.TestCase):
    def test_canonical_request_maps_to_exact_splash_fields(self):
        for model in sp.MODEL_IDS.values():
            for messages, budget in ((chain(), 512), (chain()[:2], 2048)):
                for turn in range(7):
                    with self.subTest(model=model, turn=turn, budget=budget):
                        hf = build_request(model, messages, schemas(), turn, 'qwen')
                        request = native(turn, model, messages)
                        expected = {'model': model, 'messages': messages, 'tools': schemas(), 'tool_choice': 'auto',
                                    'max_completion_tokens': 8192, 'logprobs': False, 'stream': turn % 2 == 1,
                                    'temperature': 0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'presence_penalty': 0.0,
                                    'frequency_penalty': 0.0, 'chat_template_kwargs': {'enable_thinking': True},
                                    'preserve_thinking': True, 'parallel_tool_calls': False,
                                    'thinking_token_budget': budget}
                        if turn % 2:
                            expected['stream_options'] = {'include_usage': True}
                        self.assertEqual(request, expected)
                        self.assertFalse({'repetition_penalty', 'reasoning_effort', 'hf2q_overflow_policy'} & set(request))
                        self.assertIn('hf2q_overflow_policy', hf)

    def test_wire_boundary_round_trips_and_rejects_drift(self):
        for turn in (0, 1):
            canonical = sp.canonical(native(turn), 'qwen')
            self.assertEqual(canonical, build_request(M, chain(), schemas(), turn, 'qwen'))
            wire(canonical, 'qwen')
        request = native(1)
        drift = [('temperature', 0.6), ('top_p', 0.95), ('top_k', 20), ('min_p', 0.05), ('repetition_penalty', 1.05),
                 ('presence_penalty', 0.5), ('chat_template_kwargs', {'enable_thinking': False}),
                 ('preserve_thinking', False), ('parallel_tool_calls', True), ('reasoning_effort', 'xhigh'),
                 ('thinking_token_budget', 2048), ('seed', 1), ('hf2q_overflow_policy', 'reject'),
                 ('stream_options', {'include_usage': False}), ('model', 'Qwen3.8 27B'), ('cache_salt', 'x')]
        for key, value in drift:
            with self.subTest(key=key), self.assertRaises(Invalid):
                sp.canonical({**request, key: value}, 'qwen')
        for key in list(sp.NATIVE) + ['thinking_token_budget', 'stream_options']:
            with self.subTest(missing=key), self.assertRaises(Invalid):
                sp.canonical({k: v for k, v in request.items() if k != key}, 'qwen')

    def test_exchange_refuses_drifted_request_before_any_socket(self):
        with self.assertRaises(Invalid):
            exchange('http://127.0.0.1:9/v1', {**native(0), 'repetition_penalty': 1.05}, None, 0, set(),
                     engine_kind='splash', family='qwen')

    def test_unknown_fields_and_other_families_refuse(self):
        base = build_request(M, chain(), schemas(), 0, 'qwen')
        for key, value in [('parallel_tool_calls', False), ('temperature', 0), ('thinking_token_budget', 512),
                           ('max_tokens', 8192), ('cache_salt', 'x')]:
            with self.subTest(key=key), self.assertRaises(Invalid):
                sp.request_controls({**base, key: value}, 'qwen')
        with self.assertRaises(Invalid):
            sp.request_controls(base, 'gemma4')
        with self.assertRaises(Invalid):
            sp.request_controls({**base, 'model': 'Qwen3.8 27B'}, 'qwen')


class Responses(unittest.TestCase):
    def test_unary_tool_call_and_final_summary(self):
        message, call, stats = unary(completion_body(call_message(), 'tool_calls'), M, 0, set(), 'splash')
        self.assertEqual(call, (CALL, 'read_file', {'path': 'src/ports.rs'}))
        self.assertEqual((stats['engine_kind'], stats['reasoning_observation']['reasoning_tokens']), ('splash', 12))
        final = {'role': 'assistant', 'content': 'Fixed parse_port; all tests passed.', 'reasoning_content': 'done'}
        self.assertIsNone(unary(completion_body(final, 'stop'), M, 6, set(), 'splash')[1])

    def test_unknown_unary_shapes_refuse(self):
        message = call_message()
        body = json.loads(completion_body(message, 'tool_calls'))
        no_timings = {k: v for k, v in body.items() if k != 'timings'}
        cases = [completion_body(message, 'tool_calls', system_fingerprint='x'), completion_body(message, 'error'),
                 completion_body(message, 'stop'), completion_body({'role': 'assistant', 'content': 'x'}, 'tool_calls'),
                 completion_body(call_message(content=''), 'tool_calls'),
                 completion_body(call_message(reasoning_content=''), 'tool_calls'),
                 completion_body(call_message(call_id='call_0a1b2c3d'), 'tool_calls'),
                 completion_body(call_message(args='[1]'), 'tool_calls'),
                 completion_body(message, 'tool_calls', id='chatcmpl-1a2b3c4d'),
                 completion_body(message, 'tool_calls', usage={**use(), 'input_tokens': 1000}),
                 completion_body(message, 'tool_calls', metrics={'cache': {}}),
                 completion_body(message, 'tool_calls', timings={**TIMINGS, 'extra': 1}),
                 json.dumps(no_timings).encode()]
        for raw in cases:
            with self.subTest(raw=raw[-90:]), self.assertRaises(Invalid):
                unary(raw, M, 0, set(), 'splash')

    def test_sse_role_reasoning_streamed_call_and_usage(self):
        semantic = []
        message, call, stats = replay_response(stream(), native(1), 1, set(), 'splash', semantic)
        self.assertEqual(call, (CALL2, 'read_file', {'path': 'tests/ports.rs'}))
        self.assertEqual(message['reasoning_content'], 'Read the tests.')
        self.assertEqual(stats['cache_observation']['cached_tokens'], 768)
        self.assertTrue(semantic)

    def test_unknown_sse_shapes_refuse(self):
        request = native(1)
        bad_frames = [chunk({'role': 'assistant'}), chunk({'reasoning': 'x'}), chunk({'content': 'x'}, timings=TIMINGS),
                      chunk({'tool_calls': [{'index': 0, 'id': CALL, 'type': 'function',
                                             'function': {'name': 'run_tests', 'arguments': '{}'}}]}),
                      chunk({'tool_calls': [{'index': 0, 'id': 'call_0a1b2c3d', 'type': 'function',
                                             'function': {'name': 'run_tests'}}]}),
                      chunk({'content': 'x'}, prompt_progress={'total': 1}),
                      b'data: {"error":{"message":"x","type":"server_error"}}\n\n']
        for bad in bad_frames:
            with self.subTest(frame=bad[:90]), self.assertRaises(Invalid):
                replay_response(stream(bad), request, 1, set(), 'splash')
        for final in (chunk({}, 'tool_calls'), chunk({}, 'error', timings=TIMINGS),
                      chunk({'content': 'x'}, 'tool_calls', timings=TIMINGS)):
            with self.subTest(final=final[:90]), self.assertRaises(Invalid):
                replay_response(stream(final=final), request, 1, set(), 'splash')
        with self.assertRaises(Invalid):
            replay_response(stream(usage_frame=b''), request, 1, set(), 'splash')

    def test_usage_refuses_contradictions(self):
        self.assertEqual(sp.usage(use(cached=0), 0)['cache_observation']['cached_tokens'], 0)
        for value in (use(cached=1001), use(prompt=260000), {**use(), 'total_tokens': 1}, {**use(), 'extra': 0},
                      {**use(), 'prompt_tokens_details': {}}, use(completion=8193), use(reasoning=41), None):
            with self.subTest(value=value), self.assertRaises(Invalid):
                sp.usage(value, 1)


class Configuration(unittest.TestCase):
    def test_presets_flags_and_family_rules(self):
        self.assertEqual(sp.server_flags(sp.parse_config(S0, 'qwen38')), ['--decode-share', '0.5'])
        self.assertEqual(sp.server_flags(sp.parse_config(sp.preset('matched', 'qwen38'), 'qwen38')),
                         ['--decode-share', '0.5', '--kv-format', 'bf16', '--disable-ane', '--max-cache-disk', '0'])
        self.assertEqual(sp.server_flags(sp.parse_config(sp.preset('matched', 'qwen36'), 'qwen36')),
                         ['--decode-share', '0.5', '--kv-format', 'bf16', '--verify-rows', 'all', '--max-cache-disk', '0'])
        with self.assertRaises(Invalid):
            sp.preset('verify-all', 'qwen38')
        with self.assertRaises(Invalid):
            sp.preset('ane-off', 'qwen36')
        for text in ('kv=int8', S0.replace('kv=int8', 'kv=fp8'), S0.replace('max_memory=auto', 'max_memory=96GB'),
                     ','.join(reversed(S0.split(',')))):
            with self.subTest(text=text), self.assertRaises(Invalid):
                sp.parse_config(text, 'qwen38')
        capped = sp.parse_config(S0.replace('max_memory=auto', 'max_memory=96G'), 'qwen38')
        self.assertEqual(sp.server_flags(capped)[-2:], ['--max-memory', '96G'])

    def test_controls_bind_one_family_and_configuration(self):
        for q3 in sp.ARTIFACTS:
            controls = sp.controls(q3, sp.preset('s0', q3))
            validate_binding({'family': 'qwen', 'engine_kind': 'splash', 'family_controls': controls}, 'qwen', 4)
            self.assertEqual(sp.controls_family(copy.deepcopy(controls)), q3)
        base = sp.controls('qwen36', sp.preset('s0', 'qwen36'))
        for changed in ({**base, 'slots': 2}, {**base, 'request_repetition_penalty': 1.05},
                        {**base, 'request_parallel_tool_calls': True}, {**base, 'model_id': M},
                        {**base, 'config': sp.preset('kv-bf16', 'qwen36')}, {**base, 'extra': 1}, None):
            with self.subTest(changed=str(changed)[:60]), self.assertRaises(Invalid):
                validate_binding({'family': 'qwen', 'engine_kind': 'splash', 'family_controls': changed}, 'qwen', 4)

    def test_preflight_replaces_hf_keys(self):
        keys = sp.preflight(family_preflight(n4_preflight(PREFLIGHT), 'qwen'), 'qwen')
        validate_preflight({k: True for k in keys}, PREFLIGHT, 'splash', 'qwen')
        self.assertTrue(sp.NATIVE_KEYS <= keys)
        with self.assertRaises(Invalid):
            validate_selection('gemma4', 'splash', 4)


class Launch(unittest.TestCase):
    def spec(self, **changes):
        values = {'q3_family': 'qwen38', 'splash_dir': '/opt/splash-x', 'commit': sp.SPLASH_COMMIT,
                  'build_id': sp.BUILD_ID, 'binary_sha256': 'a' * 64, 'metallib_sha256': 'b' * 64,
                  'host_step_sha256': 'c' * 64, 'toolchain_sha256': 'd' * 64, 'venv_freeze_sha256': 'e' * 64,
                  'model': '/m/Qwen3.8-27B-vanilla.gguf', 'model_sha256': 'f' * 64,
                  'draft_dir': '/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/' + sp.ARTIFACTS['qwen38']['draft_revision'],
                  'config': S0, 'run_dir': '/out/splash', 'port': '18257'}
        return {**values, **changes}

    def test_launch_spec_is_closed_and_pinned(self):
        self.assertEqual(ready.launch_spec(self.spec())['port'], 18257)
        for change in ({'commit': '0' * 40}, {'build_id': 'src-x'}, {'binary_sha256': 'A' * 64}, {'model': '/m/x.bin'},
                       {'draft_dir': '/hub/models--incoai--Qwen3.6-35B-A3B-DFlash2/snapshots/x'}, {'model': 'm.gguf'},
                       {'config': 'kv=int8'}, {'port': '80'}, {'q3_family': 'gemma4'}, {'extra': 'x'}):
            with self.subTest(change=change), self.assertRaises((Invalid, KeyError)):
                ready.launch_spec(self.spec(**change))

    def test_server_argv_and_environment(self):
        spec = ready.launch_spec(self.spec())
        argv = ready.server_argv(spec, '/out/splash/models/.resolved/abc')
        self.assertEqual(argv[:6], ['/opt/splash-x/.venv/bin/python', '-P', '-u', '-m', 'server.server',
                                    '/out/splash/models/.resolved/abc'])
        for flag, value in (('--model', M), ('--max-context', '256K'), ('--idle-release', 'off'),
                            ('--thinking-end-text', '\nI need to answer now.</think>'), ('--port', '18257')):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn('--no-webui', argv)
        self.assertNotIn('--persistent-cache', argv)
        env = {'PATH': '/usr/bin', 'HOME': '/h', **ready.SERVE_ENV}
        self.assertEqual(ready.serve_environ(spec, env)['PYTHONPATH'], '/opt/splash-x')
        for bad in ({**env, 'SPLASH_PORT': '1'}, {**env, 'PYTHONPATH': '/x'}, {'PATH': '/usr/bin'},
                    {**env, 'HF_HUB_CACHE': '/c'}):
            with self.subTest(bad=bad), self.assertRaises(Invalid):
                ready.check_environ(bad)

    def test_argv_template_freezes_no_assembly_path(self):
        variables = {'SPLASH_DIR': '/opt/splash-x', 'SPLASH_CONFIG': S0, 'PORT': '18257'}
        template = ready.argv_template(variables, 'qwen38')
        self.assertIn(ready.ASSEMBLY_PLACEHOLDER, template)
        self.assertIsNone(ready.argv_template({**variables, 'SPLASH_CONFIG': 'kv=int8'}, 'qwen38'))

    def test_assembly_binds_the_gguf_digest_and_the_draft_blob(self):
        with tempfile.TemporaryDirectory() as root:
            spec = ready.launch_spec(self.spec(run_dir=f'{root}/splash'))
            assembly = Path(root, 'splash', 'models', '.resolved', 'abc')
            assembly.mkdir(parents=True)
            weights = sp.ARTIFACTS['qwen38']['draft_weights_sha256']
            record = {'version': 1, 'model': 'x', 'family': 'Qwen3.8-27B', 'target_format': 'gguf',
                      'vision_format': 'none', 'sources': {}, 'files': {
                          'target/Qwen3.8-27B-vanilla.gguf': {'path': spec['model'], 'bytes': 1, 'mtime_ns': 1,
                                                              'ctime_ns': 1, 'digest': 'f' * 64},
                          'draft/model.safetensors': {'path': '/hub/x', 'bytes': 1, 'mtime_ns': 1, 'ctime_ns': 1,
                                                      'digest': weights}}}
            (assembly / 'model.json').write_text(json.dumps(record))
            self.assertEqual(ready.verify_assembly(spec, assembly)['target_digest'], 'f' * 64)
            for change in ({'family': 'Qwen3.6-35B-A3B'}, {'vision_format': 'gguf'},
                           {'files': {**record['files'], 'target/Qwen3.8-27B-vanilla.gguf': {
                               **record['files']['target/Qwen3.8-27B-vanilla.gguf'], 'digest': '0' * 64}}},
                           {'files': {k: v for k, v in record['files'].items() if k.startswith('target')}}):
                (assembly / 'model.json').write_text(json.dumps({**record, **change}))
                with self.subTest(change=str(change)[:60]), self.assertRaises(Invalid):
                    ready.verify_assembly(spec, assembly)
            with self.assertRaises(Invalid):
                ready.verify_assembly(spec, Path(root, 'elsewhere', '.resolved', 'abc'))

    def test_launch_problems_require_the_hf2q_gguf(self):
        self.assertEqual(ready.launch_problems('splash', {'path': '/m/a.gguf', 'sha256': 'f' * 64}, 'f' * 64), [])
        self.assertTrue(ready.launch_problems('splash', {'path': '/m/a.gguf', 'sha256': 'e' * 64}, 'f' * 64))
        self.assertTrue(ready.launch_problems('splash', {'path': '/m/a.gguf', 'sha256': None}, 'f' * 64))
        self.assertTrue(ready.launch_problems('splash', {'path': '/m/config.json', 'sha256': 'f' * 64}, 'f' * 64))

    def test_build_info_needs_every_identity_hash(self):
        hashes = {name: '1' * 64 for name in ready.BUILD_INFO_HASHES}
        text = ready.build_info(sp.SPLASH_COMMIT, sp.BUILD_ID, hashes)
        for name in ('binary-sha256', 'metallib-sha256', 'host-step-sha256', 'toolchain-sha256', 'venv-freeze-sha256'):
            self.assertIn(name, text)
        self.assertIsNone(ready.build_info(sp.SPLASH_COMMIT, sp.BUILD_ID, {**hashes, 'metallib_sha256': None}))


class Audit(unittest.TestCase):
    STDOUT = ('19:00:00 Loading · hf2q/Qwen3.8-27B\n'
              '19:00:05 Started · input 1,000 · cached 0 · effort absent → xhigh · enable_thinking true · thinking on\n'
              '19:00:07 Done · input 1,000 · cached 0 · output 40 · TTFT 0.3s · 50.0 tok/s · 5.7 tok/cycle\n'
              '19:00:08 Started · input 1,200 · cached 1,000 · effort absent → xhigh · enable_thinking true · thinking on\n'
              '19:00:09 Done · input 1,200 · cached 1,000 · deduplicated 64 · output 30 · TTFT 0.1s · 48.0 tok/s\n')

    def test_done_lines_parse_counts_and_cycles(self):
        done = [(audit.number(m.group(1)), audit.number(m.group(2)), audit.number(m.group(4)))
                for m in audit.DONE.finditer(self.STDOUT)]
        self.assertEqual(done, [(1000, 0, 40), (1200, 1000, 30)])
        self.assertEqual([c for m in audit.DONE.finditer(self.STDOUT) for c in audit.CYCLE.findall(m.group(5))], ['5.7'])
        self.assertEqual(len(audit.STARTED.findall(self.STDOUT)), 2)
        self.assertTrue(audit.REFUSED.search('19:00:10 Engine restarted\n'))
        self.assertTrue(audit.CANCELLED.search('19:00:10 Cancelled · input 1 · cached 0 · output 0\n'))


class Verdict(unittest.TestCase):
    def effect(self, low, high):
        return {**q3_stats.paired_effect([]), 'blocks': 6, 'half_width_pp': 2.0, 'ratio_ci95': [low, high]}

    def state(self, **changes):
        base = {'experiment_failures': [], 'peer_unavailable': None, 'peer_failures': [], 'quality': {'status': 'pass'},
                'complete': True, 'vs_stock': self.effect(0.7, 0.8), 'vs_peer': self.effect(0.85, 0.9),
                'fairness': {}, 'thermal_valid': True, 'memory_valid': True, 'splash_required': True,
                'splash_unavailable': None, 'splash_failures': [], 'vs_splash': self.effect(0.88, 0.93)}
        return {**base, **changes}

    def test_win_needs_both_peers(self):
        self.assertEqual(q3_stats.family_verdict(self.state())[0], 'Win')
        verdict, reasons = q3_stats.family_verdict(self.state(vs_splash=self.effect(0.9, 0.97)))
        self.assertEqual(verdict, 'Inconclusive')
        self.assertIn('Splash margin not demonstrated (95% CI upper bound of the ratio above 0.95)', reasons)
        self.assertEqual(q3_stats.family_verdict(self.state(vs_splash=self.effect(1.02, 1.1)))[0], 'Loss')
        self.assertEqual(q3_stats.family_verdict(self.state(splash_failures=['x']))[0], 'Failed')
        unavailable = q3_stats.family_verdict(self.state(splash_unavailable={'reason': 'no Splash arm'}))
        self.assertEqual(unavailable, ('Unavailable', ['no runnable Splash arm: no Splash arm']))
        # a required Splash never lets a peer-only result win; a family without Splash keeps the two-arm rule
        self.assertEqual(q3_stats.family_verdict(self.state(splash_required=False, vs_splash=None))[0], 'Win')

    def test_check_order(self):
        order = q3_stats.family_verdict(self.state(peer_failures=['p'], splash_unavailable={'reason': 's'}))
        self.assertEqual(order[0], 'Unavailable')
        order = q3_stats.family_verdict(self.state(peer_failures=['p'], splash_failures=['s']))
        self.assertTrue(order[1][0].startswith('peer arm failed'))
        order = q3_stats.family_verdict(self.state(vs_peer=self.effect(1.01, 1.1), vs_splash=self.effect(1.01, 1.1)))
        self.assertIn("peer's", order[1][0])

    def test_splash_runs_only_in_n4_and_n1(self):
        plan = {'family': 'qwen38', 'measured_arms': ['stock', 'experiment', 'peer', 'splash'], 'arms': {}}
        self.assertEqual(q3_report.stage_arms(plan, 'n4'), plan['measured_arms'])
        self.assertEqual(q3_report.stage_arms(plan, 'n1'), plan['measured_arms'])
        for control in ('3+1', '2+2', '1+3'):
            self.assertNotIn('splash', q3_report.stage_arms(plan, control))
        self.assertTrue(q3_report.splash_required(plan))
        self.assertFalse(q3_report.splash_required({**plan, 'family': 'gemma4'}))
        self.assertFalse(q3_report.splash_required({**plan, 'native_comparison': 'x'}))


class Sweep(unittest.TestCase):
    def record(self, config, makespan, passed=4, q5='CalibrationPass', audited='Pass', run='r'):
        return {'run': run, 'config': config, 'makespan_s': makespan, 'tasks_passed': passed, 'q5_verdict': q5,
                'audit': audited}

    def test_select_uses_splash_outcomes_and_marks_close_calls_provisional(self):
        s0, kv = sp.preset('s0', 'qwen38'), sp.preset('kv-bf16', 'qwen38')
        ane = sp.preset('ane-off', 'qwen38')
        result = splash_sweep.select([self.record(s0, 60), self.record(s0, 62), self.record(kv, 55), self.record(kv, 57),
                                      self.record(ane, 40, passed=2), self.record(ane, 41, passed=2)])
        self.assertFalse(result['configurations'][ane]['valid'])
        self.assertEqual(result['choice']['config'], kv)
        self.assertFalse(result['choice']['provisional'])
        close = splash_sweep.select([self.record(s0, 60), self.record(kv, 61)])
        self.assertTrue(close['choice']['provisional'])
        invalid = splash_sweep.select([self.record(s0, 60, q5='MemoryStop'), self.record(kv, 70, audited='bad')])
        self.assertIsNone(invalid['choice'])

    def test_derive_case_changes_only_the_configuration(self):
        import q5_case
        case = q5_case.validate(q5_case.loads((HERE / 'q5_cases' / 'splash-qwen36.json').read_bytes()))
        derived = splash_sweep.derive_case(case, 'verify-all', '96G')
        self.assertEqual(derived['variables']['SPLASH_CONFIG'],
                         sp.preset('verify-all', 'qwen36').replace('max_memory=auto', 'max_memory=96G'))
        self.assertEqual({k: v for k, v in derived['variables'].items() if k != 'SPLASH_CONFIG'},
                         {k: v for k, v in case['variables'].items() if k != 'SPLASH_CONFIG'})
        self.assertEqual(derived['serve'], case['serve'])
        with self.assertRaises(Invalid):
            splash_sweep.derive_case(case, 'ane-off')

    def test_parity_config_binds_canonical_requests_and_splash_renderings(self):
        import hashlib
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            run = root / 'run'
            agent = run / 'client' / 'agent-0'
            (agent / 'turn-0').mkdir(parents=True)
            raw = json.dumps(native(0), separators=(',', ':')).encode()
            (agent / 'turn-0' / 'request.json').write_bytes(raw)
            (agent / 'turn-0' / 'splash-apply-template.raw').write_bytes(b'{"prompt":"<|im_start|>system"}')
            (agent / 'summary.json').write_text(json.dumps(
                {'turns': [{'turn': 0, 'request_sha256': hashlib.sha256(raw).hexdigest()}]}))
            (run / 'client' / 'summary.json').write_text(json.dumps(
                {'engine_kind': 'splash', 'native_readiness': {'q3_family': 'qwen38'}}))
            template = root / 'qwen38.template.jinja'
            template.write_text('not the reviewed template')
            with self.assertRaises(Invalid):
                splash_sweep.parity_config(run, template, root / 'parity')
            with mock.patch.dict(sp.ARTIFACTS['qwen38'], {'template_sha256': hashlib.sha256(template.read_bytes()).hexdigest()}):
                config = json.loads(splash_sweep.parity_config(run, template, root / 'parity').read_text())
            self.assertEqual(config['architecture'], 'qwen35')
            (turn,) = config['turns']
            canonical = json.loads(Path(turn['canonical_request']['path']).read_bytes())
            self.assertEqual(canonical, build_request(M, chain(), schemas(), 0, 'qwen'))
            self.assertEqual(turn['splash_apply_template']['sha256'],
                             hashlib.sha256(b'{"prompt":"<|im_start|>system"}').hexdigest())


class Registry(unittest.TestCase):
    def test_splash_is_a_same_artifact_adapter_kind(self):
        self.assertEqual(SAME_ARTIFACT_KINDS, ('splash',))
        self.assertEqual(ADAPTER_KINDS, NATIVE_KINDS + ('splash',))
        self.assertIs(native_peers.adapter('splash'), sp)
        self.assertFalse(native_peers.is_native('splash'))
        self.assertTrue(native_peers.is_same_artifact('splash'))
        self.assertEqual(native_peers.audit('splash').__name__, 'native_peer_splash_audit')
        self.assertIs(SSE(M, 0, set(), 'splash').native_frame, sp.check_frame)
        self.assertIsNone(sp.SAME_GGUF_UNAVAILABLE)
        self.assertIsNone(sp.NATIVE_COMPARISON)
        self.assertEqual(sp.BASELINES, {})
        self.assertNotIn('splash', native_peers.baselines())
        self.assertIn('splash', native_peers.process_names())
        self.assertEqual(sp.CLIENT_MODULES, ('native_peers.py', 'native_peer_splash.py', 'native_peer_splash_ready.py',
                                             'test_native_peer_splash.py'))
        self.assertIn('same-artifact peer of record', sp.q3_role({'family': 'qwen36'}))

    def test_cases_validate_and_name_the_launch_inputs(self):
        import q5_case
        for q3 in sp.ARTIFACTS:
            case = q5_case.validate(q5_case.loads((HERE / 'q5_cases' / f'splash-{q3}.json').read_bytes()))
            block = case['client']['approval']['set']['native_splash']
            self.assertEqual(set(block), ready.SPEC_KEYS)
            argv = case['serve']['argv']
            launched = {argv[i][2:].replace('-', '_') for i in range(len(argv)) if argv[i].startswith('--')}
            self.assertEqual(launched, set(ready.LAUNCH_KEYS))
            self.assertEqual(case['variables']['MODEL_ID'], sp.MODEL_IDS[q3])
            self.assertEqual(case['client']['argv'][-1], 'splash')
            hf2q = q5_case.loads((HERE / 'q5_cases' / f'{q3}.json').read_bytes())
            self.assertEqual(case['variables']['MODEL'], hf2q['variables']['MODEL'])
            self.assertIsNone(case['variables']['MODEL_SHA256'])


def splash_manifest(arms=('stock', 'experiment', 'peer', 'splash'), spans=None, unavailable=None, outcomes=None):
    from test_native_peer_tensorfold_campaign import metrics
    spans = spans or {'stock': 100.0, 'experiment': 80.0, 'peer': 90.0, 'splash': 88.0}
    outcomes = outcomes or {}
    blocks = [{'block': b, 'order': list(arms), 'entries': [
        {'name': f'n4-{b}-{arm}', 'arm': arm, 'kind': 'measured', 'block': b, 'attempts': [],
         'outcome': outcomes.get(arm, 'Pass'), 'failure_kind': 'task' if outcomes.get(arm) == 'Fail' else None,
         'reason': 'x' if outcomes.get(arm) == 'Fail' else None, 'metrics': {**metrics(spans[arm]), 'audit': None}}
        for arm in arms]} for b in range(1, 6)]
    plan = {'family': 'qwen38', 'primary': 'n4', 'measured_arms': list(arms),
            'arms': {arm: {'engine_kind': {'splash': 'splash', 'peer': 'peer'}.get(arm, 'hf2q'),
                           **({'label': sp.LABELS['qwen38']} if arm == 'splash' else {})} for arm in arms},
            'peer_unavailable_evidence': [], 'design': {}, 'splash_required': True,
            'declared': {'stock': {'label': 'v0.1.28 release', 'commit': '1' * 40},
                         'experiment': {'label': 'main HEAD', 'commit': '2' * 40}, 'qe_identity': 'qe'}}
    artifact = {'path': '/m.gguf', 'bytes': 10, 'sha256': 'a' * 64}
    return {'plan': plan, 'unavailable': unavailable or {}, 'identities_sha256': 'x', 'status': 'Complete',
            'created_utc': '2026-11-04T00:00:00+00:00',
            'identities': {'arms': {'experiment': {'binary': {'sha256': 'e' * 64}, 'artifact': artifact}}},
            'stages': [{'workload': 'n4', 'phase_plan': None, 'status': 'Complete',
                        'stop': {'reason': 'precision', 'blocks': 5}, 'warmups': [], 'blocks': blocks}]}


class Campaign(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.nc, self.evidence = self.root / 'nc.json', self.root / 'splash-oom.json'
        self.nc.write_text('{}')
        self.evidence.write_text('{}')

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, family='qwen38', *extra):
        import q3_campaign as q3
        return q3.parse_args(['--family', family, '--stock-bin', '/bin/stock', '--stock-commit', '1' * 40,
                              '--experiment-bin', '/bin/experiment', '--experiment-commit', '2' * 40,
                              '--qe-identity', 'qe', '--negative-control-receipt', str(self.nc),
                              '--workloads', 'n4,n1,3+1', '--output', str(self.root / 'q3'), *extra])

    def test_qwen_plans_run_splash_in_n4_and_n1_only(self):
        import q3_campaign as q3
        for family in sp.ARTIFACTS:
            plan, cases = q3.build_plan(self.args(family), {}, family)
            self.assertEqual(plan['measured_arms'], ['stock', 'experiment', 'peer', 'splash'])
            self.assertEqual(plan['arms']['splash']['engine_kind'], 'splash')
            self.assertIs(plan['splash_required'], True)
            self.assertEqual(plan['arms']['splash']['label'], sp.LABELS[family])
            schedule = q3.schedule_preview(plan)
            self.assertIn('splash', schedule['n1']['warmup_order'])
            self.assertNotIn('splash', schedule['3+1']['warmup_order'])
        plan, _ = q3.build_plan(self.args('gemma4'), {}, 'gemma4')
        self.assertNotIn('splash', plan['measured_arms'])
        self.assertNotIn('splash_required', plan)
        historical = q3_report.compute_verdict({**splash_manifest(arms=('stock', 'experiment', 'peer')),
                                                'plan': {**splash_manifest()['plan'], 'splash_required': False,
                                                         'measured_arms': ['stock', 'experiment', 'peer']}})
        self.assertNotEqual(historical['verdict'], 'Unavailable')  # a campaign planned before Splash keeps its rule
        with self.assertRaises(ValueError):  # quality_campaign.CampaignError
            q3.build_plan(self.args('gemma4', '--splash-case', str(HERE / 'q5_cases' / 'splash-qwen38.json')), {}, 'gemma4')
        plan, _ = q3.build_plan(self.args('qwen38', '--splash-unavailable-evidence', str(self.evidence)), {}, 'qwen38')
        self.assertNotIn('splash', plan['measured_arms'])
        self.assertEqual(plan['splash_unavailable_evidence'][0]['path'], str(self.evidence))
        with self.assertRaises(ValueError):
            q3.build_plan(self.args('qwen38', '--splash-case', str(HERE / 'q5_cases' / 'peer-qwen38.json')), {}, 'qwen38')

    def test_verdict_requires_splash_for_qwen(self):
        verdict = q3_report.compute_verdict(splash_manifest())
        self.assertEqual(verdict['verdict'], 'Win')
        self.assertEqual(verdict['splash'], {'required': True, 'unavailable': None, 'failures': [],
                                             'label': sp.LABELS['qwen38']})
        self.assertIn('vs_splash', verdict['primary']['effects'])
        slower = q3_report.compute_verdict(splash_manifest(spans={'stock': 100.0, 'experiment': 80.0, 'peer': 90.0,
                                                                  'splash': 70.0}))
        self.assertEqual(slower['verdict'], 'Loss')
        close = q3_report.compute_verdict(splash_manifest(spans={'stock': 100.0, 'experiment': 80.0, 'peer': 90.0,
                                                                 'splash': 82.0}))
        self.assertEqual(close['verdict'], 'Inconclusive')
        missing = q3_report.compute_verdict(splash_manifest(arms=('stock', 'experiment', 'peer')))
        self.assertEqual(missing['verdict'], 'Unavailable')
        self.assertIn('no runnable Splash arm', missing['reasons'][0])
        failed = q3_report.compute_verdict(splash_manifest(outcomes={'splash': 'Fail'}))
        self.assertEqual(failed['verdict'], 'Failed')
        gone = q3_report.compute_verdict(splash_manifest(unavailable={'splash': {'reason': 'memory_stop: x',
                                                                                 'evidence': []}}))
        self.assertEqual(gone['verdict'], 'Unavailable')

    def test_tier_requires_a_splash_quality_gate(self):
        import f2_q3_tier as tier
        self.assertEqual(tier.required_references('qwen38', None, {'required': True}),
                         ['stock-hf2q', 'peer', 'splash'])
        self.assertEqual(tier.required_references('qwen38', None), ['stock-hf2q', 'peer'])
        self.assertEqual(tier.REFERENCES['splash'], 'Splash')
        import quality_campaign
        self.assertEqual(quality_campaign.REFERENCE_KINDS['splash'], 'splash')


if __name__ == '__main__':
    unittest.main()
