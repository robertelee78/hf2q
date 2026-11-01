"""Pure controls for the oMLX arm (engine_kind 'omlx'): request translation, refusals, reply envelopes, binding.

No HTTP server, model or oMLX process. Reply fixtures follow /opt/omlx 6b413c90 byte formats (exclude_none
pydantic dumps): chat.completion (server.py:5440-5484), chunks (6632-6642, 6720-6866, 7114-7166), the usage
chunk (7238-7283), Usage (api/shared_models.py:55-79, api/openai_models.py:469-503), call ids
(api/tool_calling.py:1310-1330: call_<8 hex>).
"""
import copy
import json
import unittest

import native_peer_omlx as om
import native_peers
from coding_family import preflight as family_preflight, validate_binding, validate_selection
from q2_certify import replay_response
from qwen38_n1_smoke import PREFLIGHT, build_request, exchange
from qwen38_n1_verdict import SSE, Invalid, unary, wire
from qwen_n4_live_smoke import preflight as n4_preflight, validate_preflight

CREATED = 1767225600
CALL, CALL2 = 'call_0a1b2c3d', 'call_9f8e7d6c'
MODEL = {'qwen38': om.MODEL_IDS['qwen38'], 'qwen36': om.MODEL_IDS['qwen36'], 'gemma4': om.MODEL_IDS['gemma4']}
FAMILY = {q3: a['family'] for q3, a in om.ARTIFACTS.items()}
M = MODEL['qwen38']


def schemas():
    return [{'type': 'function', 'function': {'name': name, 'strict': False,
             'parameters': {'type': 'object', 'properties': {}}}} for name in ('read_file', 'write_file', 'run_tests')]


def chain():
    return [{'role': 'system', 'content': 'unchanged'}, {'role': 'user', 'content': 'same coding task'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
             'tool_calls': [{'id': CALL, 'type': 'function',
                             'function': {'name': 'read_file', 'arguments': '{"path":"src/ports.rs"}'}}]},
            {'role': 'tool', 'tool_call_id': CALL, 'content': 'fn main() {}'}]


def native(turn, q3='qwen38', messages=None):
    return build_request(MODEL[q3], chain() if messages is None else messages, schemas(), turn, FAMILY[q3], 'omlx')


def use(prompt=1000, completion=40, cached=0, reasoning=12, **changes):
    return {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion,
            'input_tokens': prompt, 'output_tokens': completion, 'prompt_tokens_details': {'cached_tokens': cached},
            'completion_tokens_details': {'reasoning_tokens': reasoning}, 'time_to_first_token': 0.41,
            'total_time': 2.1, 'prompt_eval_duration': 0.41, 'generation_duration': 1.69,
            'prompt_tokens_per_second': 2439.02, 'generation_tokens_per_second': 23.67, **changes}


def completion_body(message, finish, model=M, **changes):
    body = {'id': 'chatcmpl-1a2b3c4d', 'object': 'chat.completion', 'created': CREATED, 'model': model,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': use(), **changes}
    return json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()


def call_message(name='read_file', args='{"path": "src/ports.rs"}', call_id=CALL, **extra):
    return {'role': 'assistant', 'reasoning_content': 'plan',
            'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': args}}], **extra}


def frame(obj):
    return b'data: ' + json.dumps(obj, separators=(',', ':')).encode() + b'\n\n'


def chunk(delta, finish=None, model=M, **extra):
    choice = {'index': 0, 'delta': delta, **({} if finish is None else {'finish_reason': finish})}
    return frame({'id': 'chatcmpl-9f8e7d6c', 'object': 'chat.completion.chunk', 'created': CREATED, 'model': model,
                  'choices': [choice], **extra})


def usage_chunk(value=None, model=M):
    return frame({'id': 'chatcmpl-9f8e7d6c', 'object': 'chat.completion.chunk', 'created': CREATED, 'model': model,
                  'choices': [], 'usage': use(cached=768, time_to_first_visible_token=0.5) if value is None else value})


def stream(extra=b'', final=None, usage_frame=None, model=M):
    call = {'index': 0, 'id': CALL2, 'type': 'function',
            'function': {'name': 'read_file', 'arguments': '{"path": "tests/ports.rs"}'}}
    return b''.join([chunk({'role': 'assistant'}, model=model), chunk({'reasoning_content': 'Read the tests.'}, model=model),
                     extra, chunk({'tool_calls': [call]}, model=model),
                     chunk({}, 'tool_calls', model=model) if final is None else final,
                     usage_chunk(model=model) if usage_frame is None else usage_frame, b'data: [DONE]\n\n'])


class Translation(unittest.TestCase):
    def test_canonical_request_maps_to_exact_omlx_fields(self):
        for q3 in om.Q3_FAMILIES:
            qwen = FAMILY[q3] == 'qwen'
            for messages, budget in ((chain(), 512), (chain()[:2], 2048)):
                for turn in range(7):
                    with self.subTest(q3=q3, turn=turn, budget=budget):
                        hf = build_request(MODEL[q3], messages, schemas(), turn, FAMILY[q3])
                        request = native(turn, q3, messages)
                        expected = {'model': MODEL[q3], 'messages': messages, 'tools': schemas(), 'tool_choice': 'auto',
                                    'max_completion_tokens': 8192, 'logprobs': False, 'stream': turn % 2 == 1,
                                    'temperature': 0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'presence_penalty': 0.0,
                                    'frequency_penalty': 0.0, 'repetition_penalty': 1.05, 'repetition_context_size': 64,
                                    'chat_template_kwargs': ({'enable_thinking': True, 'preserve_thinking': True} if qwen
                                                             else {'enable_thinking': False}),
                                    **({'thinking_budget': budget} if qwen else {})}
                        if turn % 2:
                            expected['stream_options'] = {'include_usage': True}
                        self.assertEqual(request, expected)
                        self.assertIs(type(request['temperature']), int)
                        self.assertFalse({'parallel_tool_calls', 'reasoning_effort'} & set(request))
                        delta = {k for k in request if k not in hf} | {k for k in hf if k not in request}
                        self.assertEqual(delta, {'hf2q_overflow_policy', 'chat_template_kwargs', *om.SAMPLING_FIELDS}
                                         | ({'thinking_budget'} if qwen else set()))

    def test_budget_follows_the_hf2q_qwen_ceiling_rule(self):
        tail = chain() + [{'role': 'system', 'content': 'late system'}]
        for q3 in ('qwen38', 'qwen36'):
            self.assertEqual(native(0, q3, tail)['thinking_budget'], 512)
            self.assertEqual(native(0, q3, chain()[:3] + [{'role': 'user', 'content': 'next'}])['thinking_budget'], 2048)

    def test_wire_boundary_round_trips_and_rejects_native_drift(self):
        for q3 in om.Q3_FAMILIES:
            family = FAMILY[q3]
            for turn in (0, 1):
                canonical = om.canonical(native(turn, q3), family)
                self.assertEqual(canonical, build_request(MODEL[q3], chain(), schemas(), turn, family))
                wire(canonical, family)
            request = native(1, q3)
            drift = [('temperature', 0.6), ('top_p', 0.95), ('top_k', 20), ('min_p', 0.05), ('top_p', True),
                     ('repetition_penalty', 1.0), ('repetition_context_size', 20), ('presence_penalty', 0.5),
                     ('frequency_penalty', 0.1), ('chat_template_kwargs', {'enable_thinking': True}),
                     ('chat_template_kwargs', {'enable_thinking': family != 'qwen', 'preserve_thinking': True}),
                     ('reasoning_effort', 'xhigh'), ('parallel_tool_calls', False), ('seed', 1), ('stop', ['x']),
                     ('hf2q_overflow_policy', 'reject'), ('stream_options', {'include_usage': False}),
                     ('thinking_budget', 2048 if family == 'qwen' else 512), ('thinking_budget', 512.0),
                     ('model', MODEL['gemma4' if family == 'qwen' else 'qwen38']), ('enable_thinking', True)]
            for key, value in drift:
                with self.subTest(q3=q3, key=key, value=value), self.assertRaises(Invalid):
                    om.canonical({**request, key: value}, family)
            for key in list(om.NATIVE[family]) + ['stream_options'] + (['thinking_budget'] if family == 'qwen' else []):
                with self.subTest(q3=q3, missing=key), self.assertRaises(Invalid):
                    om.canonical({k: v for k, v in request.items() if k != key}, family)

    def test_exchange_refuses_drifted_native_request_before_any_socket(self):
        with self.assertRaises(Invalid):
            exchange('http://127.0.0.1:9/v1', {**native(0), 'repetition_penalty': 1.0}, None, 0, set(),
                     engine_kind='omlx', family='qwen')

    def test_translation_does_not_mutate_its_input_or_the_constants(self):
        base = build_request(M, chain(), schemas(), 0, 'qwen')
        before = copy.deepcopy(base)
        om.request_controls(base, 'qwen')['chat_template_kwargs']['enable_thinking'] = False
        self.assertEqual(base, before)
        self.assertEqual(om.NATIVE['qwen']['chat_template_kwargs'], {'enable_thinking': True, 'preserve_thinking': True})


class Refusal(unittest.TestCase):
    def test_unknown_or_native_fields_in_the_canonical_request_refuse(self):
        for q3 in om.Q3_FAMILIES:
            base = build_request(MODEL[q3], chain(), schemas(), 0, FAMILY[q3])
            for key, value in [('parallel_tool_calls', False), ('chat_template_kwargs', {'enable_thinking': True}),
                               ('reasoning_effort', 'xhigh'), ('temperature', 0), ('top_p', 1.0), ('thinking_budget', 512),
                               ('repetition_penalty', 1.05), ('seed', 1), ('stream_options', {}), ('max_tokens', 8192),
                               ('response_format', {'type': 'text'}), ('cache_prompt', True), ('id_slot', 0)]:
                with self.subTest(q3=q3, key=key), self.assertRaises(Invalid):
                    om.request_controls({**base, key: value}, FAMILY[q3])

    def test_canonical_controls_and_model_ids_must_be_exact(self):
        base = build_request(M, chain(), schemas(), 0, 'qwen')
        for key, value in [('max_completion_tokens', 1024), ('max_completion_tokens', 8192.0),
                           ('hf2q_overflow_policy', 'truncate'), ('tool_choice', 'required'), ('logprobs', True),
                           ('stream', 1), ('model', 'Qwen3.8 27B'), ('model', MODEL['gemma4']), ('messages', []),
                           ('messages', [{'role': 'developer', 'content': 'x'}])]:
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                om.request_controls({**base, key: value}, 'qwen')
        gemma = build_request(MODEL['gemma4'], chain(), schemas(), 0, 'gemma4')
        for value in (M, MODEL['qwen36'], 'gemma-4-26b-a4b-it-4bit'):
            with self.subTest(model=value), self.assertRaises(Invalid):
                om.request_controls({**gemma, 'model': value}, 'gemma4')
        for key in ('hf2q_overflow_policy', 'logprobs', 'tools'):
            with self.subTest(missing=key), self.assertRaises(Invalid):
                om.request_controls({k: v for k, v in base.items() if k != key}, 'qwen')

    def test_other_families_and_slot_counts_refuse(self):
        for family in ('qwen', 'gemma4'):
            validate_selection(family, 'omlx', 4)
            with self.assertRaises(Invalid):
                validate_selection(family, 'omlx', 1)
        with self.assertRaises(Invalid):
            validate_selection('deepseek4', 'omlx', 4)
        with self.assertRaises(Invalid):
            om.request_controls(build_request(M, chain(), schemas(), 0, 'qwen'), 'deepseek4')


class Responses(unittest.TestCase):
    def test_unary_tool_call_reasoning_usage_and_keepalive_prefix(self):
        for q3 in om.Q3_FAMILIES:
            raw = completion_body(call_message(), 'tool_calls', MODEL[q3])
            for body in (raw, b'  ' + raw):  # past 2 s the body is led by keepalive spaces (server.py:3203-3230)
                message, call, stats = unary(body, MODEL[q3], 0, set(), 'omlx')
                self.assertEqual(call, (CALL, 'read_file', {'path': 'src/ports.rs'}))
                self.assertEqual(message['reasoning_content'], 'plan')
                self.assertEqual((stats['engine_kind'], stats['cache_observation']['status'],
                                  stats['reasoning_observation']['reasoning_tokens']), ('omlx', 'Observed', 12))

    def test_unary_final_summary(self):
        final = {'role': 'assistant', 'content': 'Fixed parse_port; all tests passed.', 'reasoning_content': 'done'}
        _, call, _ = unary(completion_body(final, 'stop', usage=use(cached=768)), M, 6, set(), 'omlx')
        self.assertIsNone(call)

    def test_unknown_unary_shapes_refuse(self):
        message = call_message()
        error = b' ' + json.dumps({'error': {'message': 'x', 'type': 'server_error', 'code': 409}}).encode()
        cases = [completion_body(message, 'tool_calls', system_fingerprint='x'),
                 completion_body(call_message(refusal='x'), 'tool_calls'), completion_body(message, 'error'),
                 completion_body(message, 'abort'), completion_body(message, 'content_filter'),
                 completion_body(message, 'stop'), completion_body({'role': 'assistant', 'content': 'x'}, 'tool_calls'),
                 completion_body(call_message(reasoning_content=''), 'tool_calls'),
                 completion_body({**message, 'tool_calls': [{**message['tool_calls'][0], 'index': 0}]}, 'tool_calls'),
                 completion_body(call_message(call_id='call_0123456789abcdef01234567'), 'tool_calls'),
                 completion_body(call_message(call_id='call_0A1B2C3D'), 'tool_calls'),
                 completion_body(call_message(args='[1]'), 'tool_calls'),
                 completion_body(call_message(args='{"path":'), 'tool_calls'),
                 completion_body(message, 'tool_calls', object='text_completion'),
                 completion_body(message, 'tool_calls', id='chatcmpl-keepalive'),
                 completion_body(message, 'tool_calls', created='now'),
                 completion_body(message, 'tool_calls', usage={k: v for k, v in use().items() if k != 'input_tokens'}),
                 error]
        body = json.loads(completion_body(message, 'tool_calls'))
        body['choices'][0]['logprobs'] = None
        cases.append(json.dumps(body).encode())
        for raw in cases:
            with self.subTest(raw=raw[-90:]), self.assertRaises(Invalid):
                unary(raw, M, 0, set(), 'omlx')

    def test_sse_role_reasoning_complete_tool_call_and_usage_chunk(self):
        for q3 in om.Q3_FAMILIES:
            semantic = []
            message, call, stats = replay_response(stream(model=MODEL[q3]), native(1, q3), 1, set(), 'omlx', semantic)
            self.assertEqual(call, (CALL2, 'read_file', {'path': 'tests/ports.rs'}))
            self.assertEqual(message['reasoning_content'], 'Read the tests.')
            self.assertEqual(stats['cache_observation']['cached_tokens'], 768)
            self.assertTrue(semantic)

    def test_unknown_sse_shapes_and_parallel_calls_refuse(self):
        request = native(1)
        second = {'index': 1, 'id': CALL, 'type': 'function', 'function': {'name': 'run_tests', 'arguments': '{}'}}
        keepalive = frame({'id': 'chatcmpl-9f8e7d6c', 'object': 'chat.completion.chunk', 'created': CREATED, 'model': M,
                           'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''},
                                        'finish_reason': None}]})  # SSE keepalive mode 'chunk' (server.py:2873-2902)
        bad_frames = [keepalive, chunk({'reasoning': 'x'}), chunk({'content': 'x', 'refusal': None}),
                      chunk({'role': 'assistant', 'content': 'x'}), chunk({'tool_calls': [second]}),
                      chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '{}'}}]}),
                      chunk({'content': 'x'}, usage=use()), chunk({'content': 'x'}, system_fingerprint='x'),
                      b'data: {"error":{"message":"x","type":"server_error"}}\n\n']
        for bad in bad_frames:
            with self.subTest(frame=bad[:90]), self.assertRaises(Invalid):
                replay_response(stream(bad), request, 1, set(), 'omlx')
        for final in (chunk({}, 'error'), chunk({}, 'abort'), chunk({'content': 'x'}, 'tool_calls'),
                      chunk({}, 'tool_calls', usage=use())):
            with self.subTest(final=final[:90]), self.assertRaises(Invalid):
                replay_response(stream(final=final), request, 1, set(), 'omlx')
        with self.assertRaises(Invalid):  # no usage chunk: oMLX emits it only for a finished output (7237-7283)
            replay_response(stream(usage_frame=b''), request, 1, set(), 'omlx')

    def test_usage_mapping_refuses_contradictions(self):
        self.assertEqual(om.usage(use(cached=0), 0)['cache_observation']['cached_tokens'], 0)
        self.assertEqual(om.usage({k: v for k, v in use().items() if k not in om.TIMING}, 1)['engine_kind'], 'omlx')
        for value in (use(cached=1001), use(prompt=260000), {**use(), 'total_tokens': 1}, {**use(), 'input_tokens': 3},
                      {**use(), 'output_tokens': 0}, {**use(), 'extra': 0}, use(total_time=-1.0), use(total_time=True),
                      {k: v for k, v in use().items() if k != 'completion_tokens_details'},
                      {**use(), 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}},
                      {**use(), 'prompt_tokens_details': {}}, {**use(), 'completion_tokens': True},
                      use(completion=8193), None):
            with self.subTest(value=value), self.assertRaises(Invalid):
                om.usage(value, 1)


class Binding(unittest.TestCase):
    def test_preflight_replaces_hf_keys_with_native_bindings(self):
        for family in om.FAMILIES:
            keys = om.preflight(family_preflight(n4_preflight(PREFLIGHT), family), family)
            validate_preflight({k: True for k in keys}, PREFLIGHT, 'omlx', family)
            self.assertTrue(om.NATIVE_KEYS <= keys)
            self.assertFalse({'auto_tq8_text_only_verified', 'greedy_omissions_rep105_thinking2048_512',
                              'gemma4_controls_bound', 'gemma4_hybrid_cache_bound'} & keys)
            with self.assertRaises(Invalid):
                validate_preflight({k: True for k in n4_preflight(PREFLIGHT)}, PREFLIGHT, 'omlx', family)

    def test_controls_bind_one_family_and_refuse_drift(self):
        for q3 in om.Q3_FAMILIES:
            packet = {'family': FAMILY[q3], 'engine_kind': 'omlx', 'family_controls': copy.deepcopy(om.CONTROLS[q3])}
            validate_binding(packet, FAMILY[q3], 4)
            self.assertEqual(om.controls_family(packet['family_controls']), q3)
        with self.assertRaises(Invalid):
            validate_binding({'family': 'qwen', 'engine_kind': 'omlx', 'family_controls': om.CONTROLS['gemma4']}, 'qwen', 4)
        base = om.CONTROLS['qwen36']
        for controls in ({**base, 'slots': 2}, {**base, 'context_per_slot': 131072}, {**base, 'prefix_cache': 'ssd'},
                         {**base, 'ssd_cache_tier': True}, {**base, 'speculation': 'mtp'},
                         {**base, 'request_repetition_penalty': 1.0}, {**base, 'request_parallel_tool_calls': False},
                         {**base, 'model_id': MODEL['qwen38']}, {**base, 'extra': 1}, {**base, 'slots': True}, None):
            with self.subTest(controls=str(controls)[:60]), self.assertRaises(Invalid):
                validate_binding({'family': 'qwen', 'engine_kind': 'omlx', 'family_controls': controls}, 'qwen', 4)


class Registry(unittest.TestCase):
    def test_omlx_resolves_through_the_registry(self):
        self.assertIs(native_peers.adapter('omlx'), om)
        self.assertEqual(native_peers.audit('omlx').__name__, 'native_peer_omlx_audit')
        self.assertIs(SSE(M, 0, set(), 'omlx').native_frame, om.check_frame)
        self.assertIsNone(om.SAME_GGUF_UNAVAILABLE)
        self.assertTrue(om.NATIVE_COMPARISON.startswith('native-comparison campaign; not the verdict of record'))
        self.assertTrue(om.PEER_CASE_REFUSAL.startswith('a native oMLX --peer-case needs --native-comparison'))
        self.assertEqual(om.CLIENT_MODULES, ('native_peers.py', 'native_peer_omlx.py', 'native_peer_omlx_ready.py',
                                             'test_native_peer_omlx.py'))
        for q3 in om.Q3_FAMILIES:
            case = {'family': q3}
            self.assertEqual(native_peers.baseline_of('omlx', om.label(case)), f'omlx_{q3}_native')
            self.assertIn('task makespan only', om.q3_role(case))
        self.assertEqual(len(set(om.LABELS.values())), 3)
        self.assertTrue(all('gemma' not in m.lower() for m in om.MODEL_IDS.values()))  # server.py:4961 rename gate
        with self.assertRaises(Invalid):
            om.label({'family': 'deepseek4'})
        import q5_preflight
        self.assertEqual(q5_preflight.classify_argv(['omlx-server']), 'runtime')
        self.assertEqual(q5_preflight.classify_argv(['/x/.venv/bin/omlx', 'serve']), 'runtime')


if __name__ == '__main__':
    unittest.main()
