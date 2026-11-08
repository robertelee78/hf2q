"""Pure controls for the TensorFold arm (engine_kind 'tensorfold') and the native-peer registry.

No HTTP server, model or TensorFold process. Response fixtures follow /opt/TensorFold 1157ea5b byte formats:
chat.completion (server/openai.zig:408-438), chunks (275-301, 440-595), extras (322-330), usage (307-320),
runtime facts (server/chat.zig:515-560), tool-call deltas (server/tool_stream.zig:73-80, 214-221).
"""
import copy
import json
from pathlib import Path
import unittest

import f2_scoreboard_render as f2
import native_peer_ds4 as nd
import native_peer_tensorfold as tf
import native_peers
import q3_campaign as q3
import quality_campaign
from coding_family import ENGINE_KINDS, NATIVE_KINDS, preflight as family_preflight, validate_binding, validate_selection
from q2_certify import replay_response
from qwen38_n1_smoke import PREFLIGHT, build_request, exchange
from qwen38_n1_verdict import SSE, Invalid, unary, usage as verdict_usage, wire
from qwen_n4_live_smoke import MODULES, preflight as n4_preflight, validate_preflight

HERE = Path(__file__).resolve().parent
M = tf.MODEL_ID
CREATED = 1767225600
SHA = 'ab' * 32
CALL = 'call_0123456789abcdef01234567'
CALL2 = 'call_89abcdef0123456789abcdef'


def schemas():
    return [{'type': 'function', 'function': {'name': name, 'strict': False,
             'parameters': {'type': 'object', 'properties': {}}}} for name in ('read_file', 'write_file', 'run_tests')]


def chain():
    return [{'role': 'system', 'content': 'unchanged'}, {'role': 'user', 'content': 'same coding task'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
             'tool_calls': [{'id': CALL, 'type': 'function',
                             'function': {'name': 'read_file', 'arguments': '{"path":"src/ports.rs"}'}}]},
            {'role': 'tool', 'tool_call_id': CALL, 'content': 'fn main() {}'}]


def native(turn, messages=None):
    return build_request(M, chain() if messages is None else messages, schemas(), turn, 'qwen', 'tensorfold')


def runtime(**changes):
    return {'enable_thinking': True, 'reasoning_effort': 'xhigh', 'engine': 'qwen27', 'tokens_per_second': 21.5,
            'seconds': 1.9, 'prefill_seconds': 0.4, 'prefill_widths': [512], 'prefill_raised': [False],
            'time_to_first_token': 0.6, 'sampling': 'greedy', 'drafts': True, 'token_sha': SHA[:12],
            'token_sha256': SHA, 'min_rows': 16, **changes}


def speculative(**changes):
    return {'rounds': 30, 'drafted': 64, 'accepted': 40, 'acceptance_rate': 0.625, 'tokens_per_round': 1.33, **changes}


def extras(**changes):
    return {'exact_mode': 'exact', 'tensorfold': runtime(), 'speculative': speculative(), **changes}


def use(prompt=1000, completion=40, cached=0, reasoning=12):
    return {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion,
            'prompt_tokens_details': {'cached_tokens': cached},
            'completion_tokens_details': {'reasoning_tokens': reasoning}}


def completion_body(message, finish, **changes):
    body = {'id': 'chatcmpl-' + '1' * 32, 'object': 'chat.completion', 'created': CREATED, 'model': M,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': use(), **extras()}
    body.update(changes)
    return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()


def call_message(name='read_file', args='{"path":"src/ports.rs"}', call_id=CALL, **extra):
    return {'role': 'assistant', 'content': None, 'reasoning_content': 'plan',
            'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': args}}], **extra}


def chunk(delta, finish=None, **extra):
    obj = {'id': 'chatcmpl-' + '2' * 32, 'object': 'chat.completion.chunk', 'created': CREATED, 'model': M,
           'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}], **extra}
    return b'data: ' + json.dumps(obj, separators=(',', ':')).encode() + b'\n\n'


def usage_chunk(value=None):
    obj = {'id': 'chatcmpl-' + '2' * 32, 'object': 'chat.completion.chunk', 'created': CREATED, 'model': M,
           'choices': [], 'usage': use(cached=900) if value is None else value}
    return b'data: ' + json.dumps(obj, separators=(',', ':')).encode() + b'\n\n'


def stream(extra=b'', final=None, usage_frame=None):
    head = {'index': 0, 'id': CALL2, 'type': 'function', 'function': {'name': 'read_file', 'arguments': ''}}
    return b''.join([b': keepalive\n\n', chunk({'reasoning_content': 'Read the tests.'}), chunk({'tool_calls': [head]})] +
                    [chunk({'tool_calls': [{'index': 0, 'function': {'arguments': part}}]})
                     for part in ('{"path":', '"tests/ports.rs"', '}')] +
                    [extra, chunk({}, 'tool_calls', **extras()) if final is None else final,
                     usage_chunk() if usage_frame is None else usage_frame, b'data: [DONE]\n\n'])


class Translation(unittest.TestCase):
    def test_canonical_request_maps_to_exact_tensorfold_fields(self):
        for messages, budget in ((chain(), 512), (chain()[:2], 2048)):
            for turn in range(7):
                hf = build_request(M, messages, schemas(), turn, 'qwen')
                request = native(turn, messages)
                expected = {'model': M, 'messages': messages, 'tools': tf.native_tools(schemas()), 'tool_choice': 'auto',
                            'max_completion_tokens': 8192, 'logprobs': False, 'stream': turn % 2 == 1,
                            'temperature': 0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'reasoning_effort': 'xhigh',
                            'chat_template_kwargs': {'enable_thinking': True}, 'thinking_budget': budget}
                if turn % 2:
                    expected['stream_options'] = {'include_usage': True}
                self.assertEqual(request, expected)
                self.assertIs(type(request['temperature']), int)
                # hf2q's render: tool schemas without `strict`, every object key sorted (qwen38-cold.prompt.txt:9-11)
                self.assertEqual(json.dumps(request['tools']), json.dumps(request['tools'], sort_keys=True))
                self.assertEqual(json.dumps(request['tools'][0]), '{"function": {"name": "read_file", "parameters": '
                                 '{"properties": {}, "type": "object"}}, "type": "function"}')
                self.assertNotIn('parallel_tool_calls', request)  # omitted: TensorFold returns every call
                delta = {k for k in request if k not in hf} | {k for k in hf if k not in request}
                self.assertEqual(delta, {'hf2q_overflow_policy', 'temperature', 'top_p', 'top_k', 'min_p',
                                         'reasoning_effort', 'chat_template_kwargs', 'thinking_budget'})

    def test_budget_follows_the_hf2q_qwen_ceiling_rule(self):
        tail = chain() + [{'role': 'system', 'content': 'late system'}]
        self.assertEqual(native(0, tail)['thinking_budget'], 512)  # system messages never change the rule
        self.assertEqual(native(0, chain()[:3] + [{'role': 'user', 'content': 'next'}])['thinking_budget'], 2048)

    def test_wire_boundary_round_trips_and_rejects_native_drift(self):
        for turn in (0, 1):
            request = native(turn)
            canonical = tf.canonical(request, 'qwen')
            self.assertEqual(canonical, build_request(M, chain(), schemas(), turn, 'qwen'))
            wire(canonical, 'qwen')
        request = native(1)
        for key, value in [('temperature', 0.6), ('top_p', 0.95), ('top_k', 20), ('min_p', 0.05), ('top_p', True),
                           ('reasoning_effort', 'high'), ('reasoning_effort', 'max'),
                           ('chat_template_kwargs', {'enable_thinking': True, 'preserve_thinking': True}),
                           ('chat_template_kwargs', {'enable_thinking': False}), ('thinking_budget', 2048),
                           ('thinking_budget', 512.0), ('parallel_tool_calls', False), ('repetition_penalty', 1.05),
                           ('draft', False), ('return_token_ids', True), ('hf2q_overflow_policy', 'reject'),
                           ('stream_options', {'include_usage': False}), ('tools', schemas()),
                           ('tools', [{'type': 'function', 'function': t['function']} for t in tf.native_tools(schemas())])]:
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                tf.canonical({**request, key: value}, 'qwen')
        for key in list(tf.NATIVE) + ['thinking_budget', 'stream_options']:
            with self.subTest(missing=key), self.assertRaises(Invalid):
                tf.canonical({k: v for k, v in request.items() if k != key}, 'qwen')

    def test_exchange_refuses_drifted_native_request_before_any_socket(self):
        with self.assertRaises(Invalid):
            exchange('http://127.0.0.1:9/v1', {**native(0), 'reasoning_effort': 'low'}, None, 0, set(),
                     engine_kind='tensorfold', family='qwen')

    def test_translation_does_not_mutate_its_input(self):
        base = build_request(M, chain(), schemas(), 0, 'qwen')
        before = copy.deepcopy(base)
        tf.request_controls(base, 'qwen')['chat_template_kwargs']['enable_thinking'] = False
        self.assertEqual(base, before)
        self.assertEqual(tf.NATIVE['chat_template_kwargs'], {'enable_thinking': True})


class Refusal(unittest.TestCase):
    def base(self):
        return build_request(M, chain(), schemas(), 0, 'qwen')

    def test_unknown_or_native_fields_in_the_canonical_request_refuse(self):
        for key, value in [('parallel_tool_calls', False), ('chat_template_kwargs', {'enable_thinking': True}),
                           ('reasoning_effort', 'xhigh'), ('temperature', 0), ('top_p', 1.0), ('top_k', 0),
                           ('min_p', 0.0), ('thinking_budget', 512), ('seed', 1), ('stop', ['x']), ('draft', False),
                           ('stream_options', {}), ('max_tokens', 8192), ('response_format', {'type': 'text'}),
                           ('presence_penalty', 0.0), ('repetition_penalty', 1.05)]:
            with self.subTest(key=key), self.assertRaises(Invalid):
                tf.request_controls({**self.base(), key: value}, 'qwen')

    def test_canonical_controls_must_be_exact(self):
        for key, value in [('max_completion_tokens', 1024), ('max_completion_tokens', 8192.0),
                           ('hf2q_overflow_policy', 'truncate'), ('tool_choice', 'required'), ('logprobs', True),
                           ('stream', 1), ('model', 'Qwen3.8 27B'), ('messages', []),
                           ('messages', [{'role': 'developer', 'content': 'x'}])]:
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                tf.request_controls({**self.base(), key: value}, 'qwen')
        for key in ('hf2q_overflow_policy', 'logprobs', 'tools'):
            with self.subTest(missing=key), self.assertRaises(Invalid):
                tf.request_controls({k: v for k, v in self.base().items() if k != key}, 'qwen')

    def test_other_families_and_slot_counts_refuse(self):
        for family in ('deepseek4', 'gemma4'):
            with self.subTest(family=family), self.assertRaises(Invalid):
                validate_selection(family, 'tensorfold', 4)
        with self.assertRaises(Invalid):
            validate_selection('qwen', 'tensorfold', 1)
        validate_selection('qwen', 'tensorfold', 4)
        with self.assertRaises(Invalid):
            tf.request_controls(self.base(), 'deepseek4')


class Responses(unittest.TestCase):
    def test_unary_tool_call_reasoning_usage_and_extras(self):
        message, call, stats = unary(completion_body(call_message(), 'tool_calls'), M, 0, set(), 'tensorfold')
        self.assertEqual(call, (CALL, 'read_file', {'path': 'src/ports.rs'}))
        self.assertEqual(message['reasoning_content'], 'plan')
        self.assertEqual((stats['engine_kind'], stats['cache_observation']['status'],
                          stats['cache_observation']['cached_tokens']), ('tensorfold', 'Observed', 0))
        self.assertEqual(stats['reasoning_observation']['reasoning_tokens'], 12)
        unary(completion_body(call_message(), 'tool_calls', stop_sequence='</tool_call>'), M, 0, set(), 'tensorfold')

    def test_unary_final_summary(self):
        final = {'role': 'assistant', 'content': 'Fixed parse_port; all tests passed.', 'reasoning_content': 'done'}
        _, call, _ = unary(completion_body(final, 'stop', usage=use(cached=900)), M, 6, set(), 'tensorfold')
        self.assertIsNone(call)

    def test_unknown_unary_shapes_and_runtime_drift_refuse(self):
        message = call_message()
        cases = [completion_body(message, 'tool_calls', system_fingerprint='x'),
                 completion_body(call_message(refusal=None), 'tool_calls'),
                 completion_body(message, 'cancelled'), completion_body(message, 'content_filter'),
                 completion_body(message, 'error'), completion_body(call_message(reasoning_content=''), 'tool_calls'),
                 completion_body({'role': 'assistant', 'content': None}, 'stop'),
                 completion_body({**message, 'tool_calls': [{**message['tool_calls'][0], 'index': 0}]}, 'tool_calls'),
                 completion_body(call_message(call_id='call_abc'), 'tool_calls'),
                 completion_body(call_message(call_id='call_0123456789ABCDEF01234567'), 'tool_calls'),
                 completion_body(call_message(args='{"path": "src/ports.rs"}'), 'tool_calls'),
                 completion_body(message, 'tool_calls', object='text_completion'),
                 completion_body(message, 'tool_calls', created='now'),
                 completion_body(message, 'tool_calls', exact_mode='approximate'),
                 completion_body(message, 'tool_calls', tensorfold=runtime(reasoning_effort='high')),
                 completion_body(message, 'tool_calls', tensorfold=runtime(enable_thinking=False)),
                 completion_body(message, 'tool_calls', tensorfold=runtime(sampling='exact')),
                 completion_body(message, 'tool_calls', tensorfold=runtime(engine='lanes')),
                 completion_body(message, 'tool_calls', tensorfold=runtime(drafts=1)),
                 completion_body(message, 'tool_calls', tensorfold=runtime(token_ids=[1, 2])),
                 completion_body(message, 'tool_calls', tensorfold=runtime(token_sha='0' * 12)),
                 completion_body(message, 'tool_calls', speculative=speculative(accepted=70)),
                 completion_body(message, 'tool_calls', speculative=speculative(extra=1))]
        body = json.loads(completion_body(message, 'tool_calls'))
        del body['tensorfold']
        cases.append(json.dumps(body).encode())
        for raw in cases:
            with self.subTest(raw=raw[-90:]), self.assertRaises(Invalid):
                unary(raw, M, 0, set(), 'tensorfold')

    def test_sse_with_keepalive_reasoning_and_streamed_tool_call(self):
        semantic = []
        message, call, stats = replay_response(stream(), native(1), 1, set(), 'tensorfold', semantic)
        self.assertEqual(call, (CALL2, 'read_file', {'path': 'tests/ports.rs'}))
        self.assertEqual(message['reasoning_content'], 'Read the tests.')
        self.assertEqual(stats['cache_observation']['cached_tokens'], 900)
        self.assertTrue(semantic)

    def test_unknown_sse_shapes_and_parallel_calls_refuse(self):
        request = native(1)
        second = {'index': 1, 'id': CALL, 'type': 'function', 'function': {'name': 'run_tests', 'arguments': ''}}
        bad_frames = [chunk({'reasoning': 'x'}), chunk({'content': 'x', 'refusal': None}), chunk({'tool_calls': [second]}),
                      chunk({'tool_calls': [{**second, 'index': 0, 'id': 'call_x'}]}),
                      chunk({'content': 'x'}, None, exact_mode='exact'),
                      b'event: error\ndata: {"error":{"message":"x","type":"server_error"}}\n\n']
        for frame in bad_frames:
            with self.subTest(frame=frame[:80]), self.assertRaises(Invalid):
                replay_response(stream(frame), request, 1, set(), 'tensorfold')
        finals = [chunk({}, 'tool_calls'), chunk({}, 'cancelled', **extras()), chunk({}, 'content_filter', **extras()),
                  chunk({'content': 'x'}, 'tool_calls', **extras()), chunk({}, 'tool_calls', **extras(), usage=use()),
                  chunk({}, 'tool_calls', **extras(tensorfold=runtime(reasoning_effort='low')))]
        for final in finals:
            with self.subTest(final=final[:80]), self.assertRaises(Invalid):
                replay_response(stream(final=final), request, 1, set(), 'tensorfold')
        missing = {k: v for k, v in use().items() if k != 'completion_tokens_details'}
        with self.assertRaises(Invalid):
            replay_response(stream(usage_frame=usage_chunk(missing)), request, 1, set(), 'tensorfold')

    def test_usage_mapping_refuses_contradictions(self):
        self.assertEqual(tf.usage(use(cached=0), 0)['cache_observation']['cached_tokens'], 0)
        for value in (use(cached=1001), use(reasoning=41), use(prompt=260000), {**use(), 'total_tokens': 1},
                      {**use(), 'extra': 0}, {k: v for k, v in use().items() if k != 'completion_tokens_details'},
                      {**use(), 'prompt_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0}},
                      {**use(), 'completion_tokens': True}, use(completion=8193), None):
            with self.subTest(value=value), self.assertRaises(Invalid):
                tf.usage(value, 1)

    def test_compact_json_is_whitespace_free_outside_strings(self):
        self.assertTrue(tf.compact_json('{"content":"fn a() {\\n  x \\"y\\"\\n}","path":"src/ports.rs"}'))
        for text in ('{"a": 1}', '{ "a":1}', '{"a":1}\n', '{"a":"unterminated}'):
            with self.subTest(text=text):
                self.assertFalse(tf.compact_json(text))


class Binding(unittest.TestCase):
    def test_preflight_replaces_hf_keys_with_native_bindings(self):
        keys = tf.preflight(family_preflight(n4_preflight(PREFLIGHT), 'qwen'), 'qwen')
        validate_preflight({k: True for k in keys}, PREFLIGHT, 'tensorfold', 'qwen')
        self.assertTrue(tf.NATIVE_KEYS <= keys)
        self.assertFalse({'auto_tq8_text_only_verified', 'greedy_omissions_rep105_thinking2048_512',
                          'auto_lazy_tool_grammar_charged', 'pool_profile_selection_verified'} & keys)
        with self.assertRaises(Invalid):
            validate_preflight({k: True for k in n4_preflight(PREFLIGHT)}, PREFLIGHT, 'tensorfold', 'qwen')

    def test_controls_bind_one_declared_variant_and_refuse_drift(self):
        for variant in ('none', 'dflash2'):
            packet = {'family': 'qwen', 'engine_kind': 'tensorfold', 'family_controls': copy.deepcopy(tf.CONTROLS[variant])}
            validate_binding(packet, 'qwen', 4)
            self.assertEqual(tf.controls_variant(packet['family_controls']), variant)
        base = tf.CONTROLS['dflash2']
        for controls in ({**base, 'drafter_bits': 0}, {**base, 'drafter': None}, {**base, 'prompt_cache': 0},
                         {**base, 'slots': 2}, {**base, 'context_per_slot': 131072}, {**base, 'slot_resume': False},
                         {**base, 'request_reasoning_effort': 'high'}, {**base, 'keep_warm_seconds': 0},
                         {**base, 'request_parallel_tool_calls': False}, {**base, 'extra': 1},
                         {**base, 'slots': True}, nd.CONTROLS, None):
            with self.subTest(controls=str(controls)[:60]), self.assertRaises(Invalid):
                validate_binding({'family': 'qwen', 'engine_kind': 'tensorfold', 'family_controls': controls}, 'qwen', 4)


class Registry(unittest.TestCase):
    def test_every_native_arm_declares_the_interface(self):
        self.assertEqual(native_peers.KINDS, NATIVE_KINDS)
        self.assertEqual(ENGINE_KINDS, ('hf2q', 'peer', 'ds4', 'tensorfold', 'omlx', 'splash'))
        for kind in NATIVE_KINDS:
            arm = native_peers.adapter(kind)
            self.assertEqual(arm.ENGINE, kind)
            self.assertTrue(all(hasattr(arm, name) for name in native_peers.INTERFACE))
            self.assertEqual(set(MODULES) & set(arm.CLIENT_MODULES), set())
            self.assertEqual(arm.CLIENT_MODULES[0], 'native_peers.py')
            self.assertTrue(all((HERE / name).is_file() for name in arm.CLIENT_MODULES))
            self.assertIsNotNone(native_peers.audit(kind).q3_section)
        for kind in ('hf2q', 'peer', 'vllm', None):
            with self.subTest(kind=kind), self.assertRaises(Invalid):
                native_peers.adapter(kind)

    def test_ds4_resolves_to_its_unchanged_arm(self):
        self.assertIs(native_peers.adapter('ds4'), nd)
        self.assertEqual(nd.BASELINES, {'ds4_native': nd.LABEL, 'ds4_samefile': nd.SAMEFILE_LABEL})
        self.assertEqual(native_peers.same_file_baselines(), {'ds4_samefile'})
        self.assertIsNone(nd.closing)
        self.assertTrue(nd.PEER_CASE_REFUSAL.startswith('a native ds4 --peer-case needs --peer-unavailable-evidence'))
        # Q4-DS4-RACE decision D2: the same-file ds4 case runs as a --native-comparison campaign
        self.assertIsNotNone(nd.NATIVE_COMPARISON)
        self.assertTrue(nd.NATIVE_COMPARISON.startswith("same-file native comparison on ds4's GGUF; not the verdict of record"))
        self.assertEqual(nd.label({}), nd.LABEL)
        self.assertEqual(nd.q3_role({}), nd.Q3_ROLE)
        self.assertEqual(native_peers.baseline_of('ds4', nd.LABEL), 'ds4_native')
        self.assertEqual(native_peers.baseline_of('ds4', nd.SAMEFILE_LABEL), 'ds4_samefile')
        self.assertIsNone(SSE('m', 0, set()).native_frame)
        self.assertIs(SSE('m', 0, set(), 'ds4').native_frame, nd.check_frame)
        self.assertIs(SSE('m', 0, set(), 'tensorfold').native_frame, tf.check_frame)
        self.assertEqual(verdict_usage(use(), 0, 'tensorfold')['engine_kind'], 'tensorfold')
        with self.assertRaises(Invalid):
            verdict_usage(use(), 0, 'vllm')

    def test_consumer_vocabularies_come_from_the_registry(self):
        import native_peer_omlx as om
        self.assertEqual(f2.NATIVE_BASELINES, {'ds4_native': nd.LABEL, 'ds4_samefile': nd.SAMEFILE_LABEL,
                                               'tensorfold_native': tf.LABELS['none'],
                                               'tensorfold_dflash2_native': tf.LABELS['dflash2'],
                                               **{f'omlx_{q3}_native': om.LABELS[q3] for q3 in om.Q3_FAMILIES}})
        self.assertFalse(set(f2.NATIVE_BASELINES) & set(f2.ARMS))
        self.assertEqual(q3.ARM_KINDS['peer'], ('peer', 'ds4', 'tensorfold', 'omlx'))
        self.assertEqual(q3.ARM_KINDS['peer_matched'], ('peer',))
        self.assertEqual({k: v for k, v in quality_campaign.REFERENCE_KINDS.items() if k.endswith('-native')},
                         {'ds4-native': 'ds4', 'tensorfold-native': 'tensorfold', 'omlx-native': 'omlx'})
        names = native_peers.process_names()
        self.assertTrue({'ds4', 'ds4-server', 'tensorfold', 'tensorfold-native', 'omlx', 'omlx-server'} <= names)
        import q5_preflight
        self.assertEqual(q5_preflight.classify_argv(['/opt/TensorFold/zig-out/native/bin/tensorfold-native', 'serve']),
                         'runtime')


if __name__ == '__main__':
    unittest.main()
