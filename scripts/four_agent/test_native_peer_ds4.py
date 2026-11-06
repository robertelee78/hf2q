"""Pure controls for the native ds4 arm (engine_kind 'ds4'). No HTTP server, model or ds4 process.

Response fixtures follow ds4_server.c c97964b byte formats: final_response (9016-9051), chunk
builders (7038-7160, 7270-7329, 8149-8175), usage (7075-7119), keepalive (12712-12715).
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import f2_scoreboard_render as f2
import native_peer_ds4 as nd
import q3_campaign as q3
from coding_family import preflight as family_preflight, validate_binding, validate_selection
from q2_certify import replay_response
from qwen38_n1_smoke import PREFLIGHT, build_request, exchange
from qwen38_n1_verdict import SSE, Invalid, unary, usage as verdict_usage, wire
from native_peer_ds4 import CLIENT_MODULES as NATIVE_MODULES
from qwen_n4_live_smoke import MODULES, preflight as n4_preflight, validate_preflight

HERE = Path(__file__).resolve().parent
CASE = HERE / 'q5_cases' / 'native-ds4-deepseek4.json'
M = nd.MODEL_ID
CREATED = 1767225600


def schemas():
    return [{'type': 'function', 'function': {'name': name, 'strict': False,
             'parameters': {'type': 'object', 'properties': {}}}} for name in ('read_file', 'write_file', 'run_tests')]


def chain():
    return [{'role': 'system', 'content': 'unchanged'}, {'role': 'user', 'content': 'same coding task'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
             'tool_calls': [{'id': 'call_0', 'type': 'function',
                             'function': {'name': 'read_file', 'arguments': '{"path":"src/ports.rs"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_0', 'content': 'fn main() {}'}]


def ds4(turn):
    return build_request(M, chain(), schemas(), turn, 'deepseek4', 'ds4')


def use(prompt=1000, completion=40, cached=0, written=None):
    written = prompt - cached if written is None else written
    return {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion,
            'prompt_tokens_details': {'cached_tokens': cached, 'cache_write_tokens': written}}


def completion_body(message, finish, **changes):
    body = {'id': 'chatcmpl-1', 'object': 'chat.completion', 'created': CREATED, 'model': M,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': use()}
    body.update(changes)
    return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()


def call_message(name='read_file', args='{"path":"src/ports.rs"}', **extra):
    return {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
            'tool_calls': [{'id': 'call_abc', 'type': 'function', 'function': {'name': name, 'arguments': args}}], **extra}


def chunk(delta, finish=None):
    obj = {'id': 'chatcmpl-2', 'object': 'chat.completion.chunk', 'created': CREATED, 'model': M,
           'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
    return b'data: ' + json.dumps(obj, separators=(',', ':')).encode() + b'\n\n'


def usage_chunk(value=None):
    obj = {'id': 'chatcmpl-2', 'object': 'chat.completion.chunk', 'created': CREATED, 'model': M,
           'choices': [], 'usage': use(cached=900) if value is None else value}
    return b'data: ' + json.dumps(obj, separators=(',', ':')).encode() + b'\n\n'


def stream(extra=b'', usage_frame=None):
    tool = {'index': 0, 'id': 'call_t1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': ''}}
    return b''.join([chunk({'role': 'assistant'}), b': prefill\n\n', chunk({'reasoning_content': 'Read the tests.'}),
                     chunk({'tool_calls': [tool]})] +
                    [chunk({'tool_calls': [{'index': 0, 'function': {'arguments': part}}]})
                     for part in ('{', '"path":"tests/ports.rs"', '}')] +
                    [extra, chunk({}, 'tool_calls'), usage_chunk() if usage_frame is None else usage_frame,
                     b'data: [DONE]\n\n'])


def models(ctx=393216, **changes):
    def entry(model_id):
        return {'id': model_id, 'object': 'model', 'created': CREATED, 'owned_by': 'ds4.c', 'name': nd.SHAPE_NAME,
                'context_length': ctx, 'top_provider': {'context_length': ctx, 'max_completion_tokens': 8192,
                                                        'is_moderated': False},
                'supported_parameters': list(nd.SUPPORTED), **changes}
    return {'object': 'list', 'data': [entry('deepseek-v4-flash'), entry('deepseek-v4-pro')]}


def log(ctx=393216, port=18251, trace='/run/ds4-trace.txt', sessions=4, backend='metal', extra=(), memory=True):
    stamp = '1031 12:00:01 ds4-server: '
    lines = ['ds4: Metal backend initialized']
    if memory:
        lines += ['ds4: memory: KV 3.37 GiB (raw 0.36 + compressed 3.01) + buffers 3.00 GiB + resident model 80.71 GiB'
                  ' = 87.08 GiB planned',
                  f'ds4: memory detail: ctx={ctx} prefill_cap=4096 raw_kv_rows=4352 compressed_kv_rows=98306 backend={backend}']
    lines += [stamp + f'context buffers 6520.00 MiB (ctx={ctx}, backend={backend}, prefill_chunk=4096, raw_kv_rows=4352, '
                      'compressed_kv_rows=98306)',
              stamp + f'{sessions} resident sessions request at least 25.47 GiB of context buffers, or 16.48 GiB when '
                      'the model shares one prefill workspace',
              stamp + f'batched mode enabled resident_sessions={sessions} prefill_quantum=2048 mixed_prefill_quantum=128 '
                      'decode_coalesce_us=0',
              stamp + f'tracing session to {trace}', *extra, stamp + f'listening on http://127.0.0.1:{port}']
    return '\n'.join(lines) + '\n'


class Translation(unittest.TestCase):
    def test_canonical_request_maps_to_exact_ds4_fields(self):
        for turn in range(7):
            hf = build_request(M, chain(), schemas(), turn, 'deepseek4')
            native = ds4(turn)
            expected = {'model': M, 'messages': chain(), 'tools': schemas(), 'tool_choice': 'auto',
                        'max_completion_tokens': 8192, 'stream': turn % 2 == 1, 'temperature': 0,
                        'reasoning_effort': 'max', 'think': True, 'top_p': 1.0, 'min_p': 0.0, 'top_k': 0}
            if turn % 2:
                expected['stream_options'] = {'include_usage': True}
            self.assertEqual(native, expected)
            self.assertIs(type(native['temperature']), int)
            delta = {k for k in native if k not in hf} | {k for k in hf if k not in native}
            self.assertEqual(delta, {'hf2q_overflow_policy', 'logprobs', 'think', 'top_p', 'min_p', 'top_k'})

    def test_wire_boundary_round_trips_and_rejects_native_drift(self):
        native = ds4(1)
        canonical = nd.canonical(native, 'deepseek4')
        self.assertEqual(canonical, build_request(M, chain(), schemas(), 1, 'deepseek4'))
        wire(canonical, 'deepseek4')
        for key, value in [('think', False), ('think', 1), ('top_p', 0.95), ('min_p', 0.05), ('top_k', 1),
                           ('top_p', True), ('logprobs', False), ('hf2q_overflow_policy', 'reject')]:
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                nd.canonical({**native, key: value}, 'deepseek4')
        for key in nd.NATIVE:
            with self.subTest(missing=key), self.assertRaises(Invalid):
                nd.canonical({k: v for k, v in native.items() if k != key}, 'deepseek4')

    def test_exchange_refuses_drifted_native_request_before_any_socket(self):
        with self.assertRaises(Invalid):
            exchange('http://127.0.0.1:9/v1', {**ds4(0), 'min_p': 0.05}, None, 0, set(),
                     engine_kind='ds4', family='deepseek4')

    def test_translation_does_not_mutate_its_input(self):
        base = build_request(M, chain(), schemas(), 0, 'deepseek4')
        before = copy.deepcopy(base)
        nd.request_controls(base, 'deepseek4')['think'] = False
        self.assertEqual(base, before)
        self.assertIs(nd.request_controls(base, 'deepseek4')['think'], True)


class Refusal(unittest.TestCase):
    def base(self):
        return build_request(M, chain(), schemas(), 0, 'deepseek4')

    def test_unknown_or_native_fields_in_the_canonical_request_refuse(self):
        for key, value in [('parallel_tool_calls', False), ('chat_template_kwargs', {'enable_thinking': True}),
                           ('think', True), ('thinking', {'type': 'enabled'}), ('top_p', 1.0), ('min_p', 0.0),
                           ('top_k', 0), ('seed', 1), ('stop', ['x']), ('ignore_eos', True), ('stream_options', {}),
                           ('max_tokens', 8192), ('response_format', {'type': 'text'})]:
            with self.subTest(key=key), self.assertRaises(Invalid):
                nd.request_controls({**self.base(), key: value}, 'deepseek4')

    def test_canonical_controls_must_be_exact(self):
        for key, value in [('reasoning_effort', 'high'), ('reasoning_effort', 'xhigh'), ('temperature', 0.6),
                           ('temperature', False), ('max_completion_tokens', 1024), ('max_completion_tokens', 8192.0),
                           ('hf2q_overflow_policy', 'truncate'), ('tool_choice', 'required'), ('logprobs', True),
                           ('stream', 1), ('model', 'DeepSeek V4 Flash 0731'), ('model', 'deepseek-chat'),
                           ('messages', []), ('messages', [{'role': 'developer'}])]:
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                nd.request_controls({**self.base(), key: value}, 'deepseek4')
        for key in ('reasoning_effort', 'temperature', 'hf2q_overflow_policy', 'logprobs'):
            with self.subTest(missing=key), self.assertRaises(Invalid):
                nd.request_controls({k: v for k, v in self.base().items() if k != key}, 'deepseek4')

    def test_other_families_and_slot_counts_refuse(self):
        for family in ('qwen', 'gemma4'):
            with self.subTest(family=family), self.assertRaises(Invalid):
                validate_selection(family, 'ds4', 4)
        with self.assertRaises(Invalid):
            validate_selection('deepseek4', 'ds4', 1)
        validate_selection('deepseek4', 'ds4', 4)
        with self.assertRaises(Invalid):
            nd.request_controls(self.base(), 'gemma4')


class Responses(unittest.TestCase):
    def test_unary_tool_call_reasoning_and_observed_cache(self):
        message, call, stats = unary(completion_body(call_message(), 'tool_calls'), M, 0, set(), 'ds4')
        self.assertEqual(call, ('call_abc', 'read_file', {'path': 'src/ports.rs'}))
        self.assertEqual(message['reasoning_content'], 'plan')
        self.assertEqual(stats['engine_kind'], 'ds4')
        self.assertEqual(stats['cache_observation'], {
            'status': 'Observed', 'cached_tokens': 0, 'source_field': 'usage.prompt_tokens_details.cached_tokens',
            'cache_write_tokens': 1000,
            'cache_write_source_field': 'usage.prompt_tokens_details.cache_write_tokens (ds4 extension)',
            'full_prefix_reuse': 'PendingPostOracle'})

    def test_unary_final_summary(self):
        final = {'role': 'assistant', 'content': 'Fixed parse_port; all tests passed.', 'reasoning_content': 'done'}
        _, call, _ = unary(completion_body(final, 'stop', usage=use(cached=900)), M, 6, set(), 'ds4')
        self.assertIsNone(call)

    def test_sse_with_keepalive_reasoning_and_streamed_tool_call(self):
        request = ds4(1)
        semantic = []
        message, call, stats = replay_response(stream(), request, 1, set(), 'ds4', semantic)
        self.assertEqual(call, ('call_t1', 'read_file', {'path': 'tests/ports.rs'}))
        self.assertEqual(message['reasoning_content'], 'Read the tests.')
        self.assertEqual(stats['cache_observation']['cached_tokens'], 900)
        self.assertTrue(semantic)

    def test_unknown_unary_shapes_refuse(self):
        cases = [completion_body(call_message(), 'tool_calls', system_fingerprint='x'),
                 completion_body(call_message(refusal=None), 'tool_calls'),
                 completion_body(call_message(), 'error'), completion_body(call_message(), 'content_filter'),
                 completion_body({**call_message(), 'content': None}, 'tool_calls'),
                 completion_body({**call_message(), 'tool_calls': [{**call_message()['tool_calls'][0], 'index': 0}]},
                                 'tool_calls'),
                 completion_body(call_message(), 'tool_calls', object='text_completion'),
                 completion_body(call_message(), 'tool_calls', created='now')]
        for raw in cases:
            with self.subTest(raw=raw[-60:]), self.assertRaises(Invalid):
                unary(raw, M, 0, set(), 'ds4')

    def test_unknown_sse_shapes_and_parallel_calls_refuse(self):
        request = ds4(1)
        bad_frames = [chunk({'reasoning': 'x'}), chunk({'content': 'x', 'refusal': None}),
                      chunk({'tool_calls': [{'index': 1, 'id': 'call_2', 'type': 'function',
                                             'function': {'name': 'run_tests', 'arguments': ''}}]}),
                      b'event: error\ndata: {"error":{"message":"x","type":"server_error"}}\n\n',
                      b'data: {"id":"chatcmpl-2","object":"chat.completion.chunk","model":"deepseek-v4-flash",'
                      b'"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\n\n']
        for frame in bad_frames:
            with self.subTest(frame=frame[:70]), self.assertRaises(Invalid):
                replay_response(stream(frame), request, 1, set(), 'ds4')
        with self.assertRaises(Invalid):
            replay_response(stream(usage_frame=usage_chunk({k: v for k, v in use().items()
                                                            if k != 'prompt_tokens_details'})), request, 1, set(), 'ds4')

    def test_usage_mapping_refuses_contradictions(self):
        self.assertEqual(nd.usage(use(cached=0), 0)['cache_observation']['cached_tokens'], 0)
        for value in (use(cached=600, written=500), use(prompt=260000), {**use(), 'total_tokens': 1},
                      {**use(), 'extra': 0}, {**use(), 'prompt_tokens_details': {'cached_tokens': 0}},
                      {**use(), 'completion_tokens': True}, use(completion=8193), None):
            with self.subTest(value=value), self.assertRaises(Invalid):
                nd.usage(value, 1)


class Readiness(unittest.TestCase):
    def test_ready_server_reports_think_max_context_sessions_and_memory(self):
        record = nd.validate_readiness(models(), log(), M, 393216, 18251, '/run/ds4-trace.txt')
        self.assertEqual((record['engine_kind'], record['native_ctx_per_session'], record['context_per_slot'],
                          record['logical_slots'], record['think_mode']), ('ds4', 393216, 262144, 4, 'max'))
        self.assertEqual(record['startup']['sessions_context_gib_shared_prefill'], 16.48)
        self.assertEqual(record['startup']['engine_memory_plan_gib']['status'], 'Observed')
        self.assertTrue(record['chat_template'].startswith('Unavailable:'))
        bare = nd.validate_readiness(models(), log(memory=False), M)
        self.assertTrue(bare['startup']['engine_memory_plan_gib']['status'].startswith('Unavailable'))
        self.assertEqual(nd.validate_readiness(models(524288), log(524288), M)['native_ctx_per_session'], 524288)

    def test_model_document_mismatches_refuse(self):
        doc = models()
        for bad in (models(262144), models(name='DeepSeek V4 Pro'), models(name='DeepSeek V4 Flash Vision Experimental'),
                    models(owned_by='other'), models(supported_parameters=['tools']),
                    {**doc, 'data': doc['data'][::-1]}, {**doc, 'data': doc['data'][:1]}, {**doc, 'extra': 1},
                    models(top_provider={'context_length': 393216, 'max_completion_tokens': 393216,
                                         'is_moderated': False})):
            with self.subTest(bad=str(bad)[:80]), self.assertRaises(Invalid):
                nd.validate_readiness(bad, log(), M)
        with self.assertRaises(Invalid):
            nd.validate_readiness(doc, log(), 'DeepSeek V4 Flash 0731')

    def test_startup_log_mismatches_refuse(self):
        stamp = '1031 12:00:01 ds4-server: '
        for bad in (log(ctx=262144), log(sessions=2), log(backend='cpu'), log(port=9),
                    log(trace='/elsewhere/ds4-trace.txt'),
                    log(extra=[stamp + 'MTP speculative decoding is disabled while native session batching is active']),
                    log(extra=[stamp + 'exact DSML tool replay disabled; tool history uses canonical JSON rendering']),
                    log(extra=[stamp + 'listening on http://127.0.0.1:18251']),
                    log().replace('tracing session to', 'tracing to'), log().replace('127.0.0.1:18251', '0.0.0.0:18251'),
                    log().replace('detail: ctx=393216', 'detail: ctx=262144')):
            with self.subTest(bad=bad[-120:]), self.assertRaises(Invalid):
                nd.validate_readiness(models(), bad, M, 393216, 18251, '/run/ds4-trace.txt')

    def spec(self, root, **changes):
        spec = {'binary': str(root / 'ds4' / 'ds4-server'), 'binary_sha256': hashlib.sha256(b'bin').hexdigest(),
                'commit': nd.DS4_COMMIT, 'chdir': str(root / 'ds4'), 'model': str(root / 'ds4' / 'gguf' / nd.MODEL_FILE),
                'model_sha256': 'b' * 64, 'ctx': '393216', 'port': '18251',
                'server_log': str(root / 'run' / 'server.stderr'), 'trace': str(root / 'run' / nd.TRACE_NAME)}
        spec.update(changes)
        return spec

    def tree(self, root):
        (root / 'ds4' / 'metal').mkdir(parents=True)
        (root / 'ds4' / 'gguf').mkdir()
        (root / 'run').mkdir()
        (root / 'ds4' / 'ds4-server').write_bytes(b'bin')
        (root / 'ds4' / 'metal' / 'moe.metal').write_text('kernel')
        (root / 'ds4' / 'gguf' / nd.MODEL_FILE).write_bytes(b'gguf')
        return {'moe.metal': hashlib.sha256(b'kernel').hexdigest()}

    def test_native_spec_refuses_identity_and_think_max_drift(self):
        root = Path('/x')
        nd.native_spec(self.spec(root))
        for key, value in [('commit', '0' * 40), ('binary', '/x/ds4/ds4-cli'), ('chdir', '/other'),
                           ('model', '/x/ds4/gguf/DeepSeek-V4-Flash-hf2q-q2.gguf'), ('ctx', '262144'), ('ctx', '393216.0'),
                           ('binary_sha256', 'A' * 64), ('model_sha256', ''), ('server_log', '/x/run/server.log'),
                           ('trace', '/y/ds4-trace.txt'), ('port', 'x'), ('binary', 'ds4/ds4-server')]:
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                nd.native_spec(self.spec(root, **{key: value}))
        with self.assertRaises(Invalid):
            nd.native_spec({**self.spec(root), 'extra': ''})

    def test_identity_binds_binary_and_runtime_kernels(self):
        self.assertEqual(len(nd.METAL_SHA256), 26)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernels = self.tree(root)
            with patch.dict(nd.METAL_SHA256, kernels, clear=True):
                identity = nd.verify_identity(nd.native_spec(self.spec(root)))
                self.assertEqual((identity['artifact']['bytes'], identity['source_revision']), (4, nd.DS4_COMMIT))
                with self.assertRaises(Invalid):
                    nd.verify_identity(nd.native_spec(self.spec(root, binary_sha256='c' * 64)))
                (root / 'ds4' / 'metal' / 'extra.metal').write_text('x')
                with self.assertRaises(Invalid):
                    nd.verify_identity(nd.native_spec(self.spec(root)))

    def test_readiness_archives_and_binds_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kernels = self.tree(root)
            spec = self.spec(root)
            Path(spec['server_log']).write_text(log(trace=spec['trace']))
            out = root / 'client'
            out.mkdir()
            packet = {'native_ds4': spec, 'family_controls': dict(nd.CONTROLS)}
            with patch.dict(nd.METAL_SHA256, kernels, clear=True), patch.object(nd, 'fetch_models', return_value=models()):
                record = nd.readiness('http://127.0.0.1:18251/v1', M, out, packet)
                self.assertEqual(record['build_info'], nd.build_info(nd.DS4_COMMIT, spec['binary_sha256']))
                self.assertTrue((out / 'ds4-startup.log').is_file())
                for base, bad in (('http://127.0.0.1:18252/v1', packet), ('http://127.0.0.1:18251/v1', None),
                                  ('http://127.0.0.1:18251/v1', {'native_ds4': spec}),
                                  ('http://127.0.0.1:18251/v1', {**packet, 'family_controls': {
                                      **nd.CONTROLS, 'native_ctx_per_session': 524288}})):
                    with self.subTest(base=base, bad=str(bad)[:40]), self.assertRaises(Invalid):
                        nd.readiness(base, M, out, bad)


class Binding(unittest.TestCase):
    def test_preflight_replaces_hf_family_keys_with_native_bindings(self):
        keys = nd.preflight(family_preflight(n4_preflight(PREFLIGHT), 'deepseek4'), 'deepseek4')
        validate_preflight({k: True for k in keys}, PREFLIGHT, 'ds4', 'deepseek4')
        self.assertTrue(nd.NATIVE_KEYS <= keys)
        self.assertFalse(family_preflight(set(), 'deepseek4') & keys)
        self.assertFalse({'auto_tq8_text_only_verified', 'pool_profile_selection_verified',
                          'peer_deepseek4_template_bound'} & keys)
        with self.assertRaises(Invalid):
            validate_preflight({k: True for k in family_preflight(n4_preflight(PREFLIGHT), 'deepseek4')},
                               PREFLIGHT, 'ds4', 'deepseek4')

    def test_controls_bind_think_max_context_and_refuse_drift(self):
        packet = {'family': 'deepseek4', 'engine_kind': 'ds4', 'family_controls': copy.deepcopy(nd.CONTROLS)}
        validate_binding(packet, 'deepseek4', 4)
        validate_binding({**packet, 'family_controls': {**nd.CONTROLS, 'native_ctx_per_session': 524288}}, 'deepseek4', 4)
        import family_peer
        for controls in ({**nd.CONTROLS, 'native_ctx_per_session': 262144}, {**nd.CONTROLS, 'trace': False},
                         {**nd.CONTROLS, 'speculation': 'mtp'}, {**nd.CONTROLS, 'request_think': 1},
                         {**nd.CONTROLS, 'slots': True}, {**nd.CONTROLS, 'disk_kv_cache': True},
                         {**nd.CONTROLS, 'extra': 1}, family_peer.CONTROLS['deepseek4'], None):
            with self.subTest(controls=str(controls)[:60]), self.assertRaises(Invalid):
                validate_binding({**packet, 'family_controls': controls}, 'deepseek4', 4)
        with self.assertRaises(Invalid):
            validate_binding({**packet, 'engine_kind': 'peer'}, 'deepseek4', 4)


class ExistingKindsUnchanged(unittest.TestCase):
    def test_hf2q_and_peer_requests_and_usage_keep_their_contracts(self):
        hf = build_request('DeepSeek V4 Flash 0731', chain(), schemas(), 0, 'deepseek4')
        peer = build_request('DeepSeek V4 Flash 0731', chain(), schemas(), 0, 'deepseek4', 'peer')
        self.assertFalse({'think', 'top_p', 'min_p', 'top_k'} & (set(hf) | set(peer)))
        self.assertEqual(peer['chat_template_kwargs'], {'enable_thinking': True})
        self.assertEqual(verdict_usage(use(cached=0), 0, 'peer')['engine_kind'], 'peer')
        with self.assertRaises(Invalid):  # hf2q still demands proven prefix reuse after turn 0
            verdict_usage({**use(cached=0), 'prompt_tokens_details': {'cached_tokens': 0}}, 1)
        self.assertIsNone(SSE('m', 0, set()).native_frame)
        with self.assertRaises(Invalid):
            build_request(M, chain(), schemas(), 0, 'deepseek4', 'tensorfold')

    def test_source_freeze_and_vocabularies_are_additive(self):
        self.assertEqual(set(MODULES) & set(NATIVE_MODULES), set())
        self.assertNotIn('ds4_native', f2.ARMS)
        self.assertEqual(f2.BASELINE_ARMS['ds4_native'], nd.LABEL)
        self.assertEqual(q3.ARM_KINDS['peer_matched'], ('peer',))
        for family in ('qwen', 'gemma4', 'deepseek4'):
            validate_selection(family, 'peer', 4)


def related(relation=None):
    """A minimal q5 case whose approval native_ds4 block declares (or omits) artifact_relation."""
    block = {} if relation is None else {'artifact_relation': relation}
    return {'client': {'approval': {'set': {'native_ds4': block}}}}


class SameFile(unittest.TestCase):
    """ADR-063 Q4-DS4-RACE decision D2: ds4 on the ds4 file the hf2q arm loads (artifact_relation same_as_hf2q)."""
    SHA = 'c' * 64

    def test_the_declaration_is_closed_and_defaults_to_different(self):
        spec = {'binary': '/x/ds4/ds4-server', 'binary_sha256': 'a' * 64, 'commit': nd.DS4_COMMIT, 'chdir': '/x/ds4',
                'model': f'/x/ds4/gguf/{nd.MODEL_FILE}', 'model_sha256': 'b' * 64, 'ctx': '393216', 'port': '18251',
                'server_log': '/x/run/server.stderr', 'trace': f'/x/run/{nd.TRACE_NAME}'}
        self.assertEqual(nd.native_spec(spec)['artifact_relation'], 'different')
        for relation in ('different', 'same_as_hf2q'):
            self.assertEqual(nd.native_spec({**spec, 'artifact_relation': relation})['artifact_relation'], relation)
        for bad in ('same', '', 'SAME_AS_HF2Q'):
            with self.subTest(bad=bad), self.assertRaises(Invalid):
                nd.native_spec({**spec, 'artifact_relation': bad})
        with self.assertRaises(Invalid):
            nd.relation(related('mixed'))
        self.assertEqual((nd.relation(related()), nd.relation({})), ('different', 'different'))

    def test_labels_roles_and_baselines_follow_the_relation(self):
        self.assertEqual((nd.label(related()), nd.q3_role(related())), (nd.LABEL, nd.Q3_ROLE))
        same = related('same_as_hf2q')
        self.assertEqual(nd.label(same), 'ds4 c97964b (same ds4 file)')
        self.assertIn('same-file native comparison', nd.q3_role(same))
        self.assertIn('task makespan only', nd.q3_role(same))
        self.assertEqual(nd.BASELINES['ds4_samefile'], nd.label(same))
        self.assertEqual(f2.BASELINE_ARMS['ds4_samefile'], nd.SAMEFILE_LABEL)
        record = nd.validate_readiness(models(), log(), M, 393216, 18251, '/run/ds4-trace.txt', 'same_as_hf2q')
        self.assertEqual(record['label'], nd.SAMEFILE_LABEL)

    def test_campaign_modes_follow_the_relation(self):
        same, different = related('same_as_hf2q'), related()
        self.assertEqual(nd.mode_problems(same, False, True), [])
        self.assertEqual(nd.mode_problems(different, True, False), [])
        self.assertIn('--peer-unavailable-evidence is refused', nd.mode_problems(same, True, False)[0])
        self.assertIn('--native-comparison needs artifact_relation same_as_hf2q', nd.mode_problems(different, False, True)[0])
        self.assertIn('--native-comparison when it is same_as_hf2q', nd.PEER_CASE_REFUSAL)

    def test_same_file_launch_needs_both_sha256_present_and_equal(self):
        same = related('same_as_hf2q')
        artifact = {'path': f'/m/gguf/{nd.MODEL_FILE}', 'sha256': self.SHA}
        self.assertEqual(nd.launch_problems('peer', artifact, self.SHA, same), [])
        for native, hf2q in ((self.SHA, 'd' * 64), (None, self.SHA), (self.SHA, None), (None, None)):
            with self.subTest(native=native, hf2q=hf2q):
                self.assertEqual(nd.launch_problems('peer', {**artifact, 'sha256': native}, hf2q, same),
                                 ['peer: a same_as_hf2q ds4 case needs the native and the hf2q artifact sha256 both '
                                  'present and equal (set MODEL_SHA256 in both cases)'])
        self.assertIn(f'peer: native artifact must be {nd.MODEL_FILE} (set --peer-var MODEL)',
                      nd.launch_problems('peer', {**artifact, 'path': '/m/other.gguf'}, self.SHA, same))
        # the different-artifact case keeps today's refusal, with or without the case argument
        for case in (related(), None):
            self.assertEqual(nd.launch_problems('peer', artifact, self.SHA, case),
                             ["peer: native artifact sha256 equals the hf2q artifact's (set --peer-var MODEL_SHA256)"])

    def test_campaign_identity_binds_a_same_file_declaration_only(self):
        variables = {'variables': {'DS4_COMMIT': nd.DS4_COMMIT, 'DS4_BIN_SHA256': 'a' * 64, 'DS4_DIR': '/nowhere'}}
        self.assertNotIn('artifact_relation', nd.campaign_identity(variables))  # existing campaigns keep their identity
        self.assertEqual(nd.campaign_identity({**variables, **related('same_as_hf2q')})['artifact_relation'], 'same_as_hf2q')


if __name__ == '__main__':
    unittest.main()
