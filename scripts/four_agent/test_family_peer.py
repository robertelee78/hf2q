"""Pure controls for the DeepSeek-V4 and Gemma 4 peer adapters. No HTTP/server/model."""
import hashlib
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import family_peer as fp
from coding_family import preflight as family_preflight, validate_binding, validate_selection
from qwen38_n1_smoke import PREFLIGHT, build_request, exchange
from qwen38_n1_verdict import Invalid, wire
from qwen_n4_live_smoke import preflight as n4_preflight, validate_preflight

MODEL = {'deepseek4': 'DeepSeek V4 Flash 0731', 'gemma4': 'Gemma 4 26B A4B It'}

TEMPLATE = {'deepseek4': (Path(__file__).resolve().parents[2] / 'src' / 'core' / 'chat_templates'
                          / 'deepseek-v4-flash-0731.jinja').read_bytes().decode('utf-8'),
            'gemma4': "{{ '<|tool_call>call:' }} gemma4 stand-in"}


def schemas():
    return [{'type': 'function', 'function': {'name': name, 'strict': False,
             'parameters': {'type': 'object', 'properties': {}}}}
            for name in ('read_file', 'write_file', 'run_tests')]


def chain():
    return [{'role': 'system', 'content': 'unchanged'}, {'role': 'user', 'content': 'same coding task'},
            {'role': 'assistant', 'content': '', 'reasoning_content': 'plan',
             'tool_calls': [{'id': 'c0', 'type': 'function',
                             'function': {'name': 'read_file', 'arguments': '{"path":"src/lib.rs"}'}}]},
            {'role': 'tool', 'tool_call_id': 'c0', 'content': 'fn main() {}'}]


def ready(family, **props_changes):
    props = {'total_slots': 4, 'model_alias': MODEL[family], 'endpoint_slots': True,
             'build_info': 'b128-e86f5fd', 'modalities': {'vision': False, 'video': False, 'audio': False},
             'chat_template': None,
             'default_generation_settings': {'n_ctx': 262144, 'params': {
                 'temperature': 0.0, 'top_k': 0, 'top_p': 1.0, 'min_p': 0.0, 'repeat_last_n': 64,
                 # float32 as the peer serializes it
                 'repeat_penalty': 1.0499999523162842 if family == 'gemma4' else 1.0,
                 'speculative.types': 'none'}}}
    props.update(props_changes)
    return {'status': 'ok'}, props, [{'id': i, 'n_ctx': 262144} for i in range(4)]


class Translation(unittest.TestCase):
    def test_deepseek_keeps_native_max_reasoning_and_adds_matched_native_fields(self):
        for turn in range(7):
            hf = build_request(MODEL['deepseek4'], chain(), schemas(), turn, 'deepseek4')
            peer = build_request(MODEL['deepseek4'], chain(), schemas(), turn, 'deepseek4', 'peer')
            self.assertNotIn('hf2q_overflow_policy', peer)
            self.assertEqual((peer['temperature'], peer['reasoning_effort']), (0, 'max'))
            self.assertEqual(peer['chat_template_kwargs'], {'enable_thinking': True})
            self.assertIs(peer['parallel_tool_calls'], False)
            # No thinking ceiling: hf2q max effort is unbounded, so no budget is sent.
            self.assertFalse({'reasoning_budget_tokens', 'thinking_budget_tokens'} & set(peer))
            delta = {k for k in peer if k not in hf} | {k for k in hf if k not in peer}
            self.assertEqual(delta, {'hf2q_overflow_policy', 'chat_template_kwargs', 'parallel_tool_calls'})
            self.assertEqual(peer['messages'], hf['messages'])
            self.assertEqual(peer.get('stream_options'), {'include_usage': True} if turn % 2 else None)

    def test_gemma_disables_thinking_natively_like_hf2q(self):
        hf = build_request(MODEL['gemma4'], chain(), schemas(), 1, 'gemma4')
        peer = build_request(MODEL['gemma4'], chain(), schemas(), 1, 'gemma4', 'peer')
        self.assertEqual(peer['chat_template_kwargs'], {'enable_thinking': False})
        self.assertIs(peer['parallel_tool_calls'], False)
        self.assertFalse({'temperature', 'reasoning_effort', 'reasoning_budget_tokens'} & set(peer))
        self.assertEqual({k: v for k, v in peer.items() if k not in ('chat_template_kwargs', 'parallel_tool_calls')},
                         {k: v for k, v in hf.items() if k != 'hf2q_overflow_policy'})

    def test_translation_does_not_mutate_or_share_native_values(self):
        base = build_request(MODEL['gemma4'], chain(), schemas(), 0, 'gemma4')
        before = copy.deepcopy(base)
        first = fp.request_controls(base, 'gemma4')
        first['chat_template_kwargs']['enable_thinking'] = True
        self.assertEqual(base, before)
        self.assertEqual(fp.request_controls(base, 'gemma4')['chat_template_kwargs'], {'enable_thinking': False})

    def test_wire_boundary_round_trips_and_rejects_native_drift(self):
        for family in fp.FAMILIES:
            peer = build_request(MODEL[family], chain(), schemas(), 1, family, 'peer')
            canonical = fp.canonical(peer, family)
            hf = build_request(MODEL[family], chain(), schemas(), 1, family)
            self.assertEqual(canonical, hf)
            wire(canonical, family)
            for key, value in [('chat_template_kwargs', {'enable_thinking': family == 'gemma4'}),
                               ('chat_template_kwargs', {'enable_thinking': 1 if family == 'deepseek4' else 0}),
                               ('parallel_tool_calls', True), ('parallel_tool_calls', 0)]:
                with self.subTest(family=family, key=key, value=value), self.assertRaises(Invalid):
                    fp.canonical({**peer, key: value}, family)
            missing = {k: v for k, v in peer.items() if k != 'parallel_tool_calls'}
            with self.assertRaises(Invalid): fp.canonical(missing, family)
            with self.assertRaises(Invalid):
                fp.canonical({**peer, 'hf2q_overflow_policy': 'reject'}, family)

    def test_exchange_refuses_drifted_native_request_before_any_socket(self):
        peer = build_request(MODEL['gemma4'], chain(), schemas(), 0, 'gemma4', 'peer')
        bad = {**peer, 'chat_template_kwargs': {'enable_thinking': True}}
        with self.assertRaises(Invalid):
            exchange('http://127.0.0.1:9/v1', bad, None, 0, set(), engine_kind='peer', family='gemma4')


class Refusal(unittest.TestCase):
    def base(self, family):
        return build_request(MODEL[family], chain(), schemas(), 0, family)

    def test_explicit_reasoning_and_sampling_overrides_refuse_not_overwrite(self):
        overrides = [('reasoning_budget_tokens', 512), ('thinking_budget_tokens', 512),
                     ('chat_template_kwargs', {'enable_thinking': False}), ('reasoning', {'effort': 'low'}),
                     ('hf2q_enable_thinking', True), ('reasoning_format', 'none'), ('parallel_tool_calls', True),
                     ('repeat_penalty', 1.0), ('n_predict', 8192), ('top_k', 1), ('seed', 1),
                     ('cache_prompt', False), ('id_slot', 0), ('grammar', 'root ::= "x"')]
        for family in fp.FAMILIES:
            base = self.base(family)
            for key, value in overrides:
                with self.subTest(family=family, key=key), self.assertRaises(Invalid):
                    fp.request_controls({**base, key: value}, family)
        gemma = self.base('gemma4')
        for key, value in [('reasoning_effort', 'none'), ('temperature', 0)]:
            with self.subTest(key=key), self.assertRaises(Invalid):
                fp.request_controls({**gemma, key: value}, 'gemma4')

    def test_deepseek_canonical_controls_must_be_exact(self):
        base = self.base('deepseek4')
        for key, value in [('reasoning_effort', 'high'), ('reasoning_effort', 'xhigh'), ('temperature', 0.6),
                           ('temperature', False), ('max_completion_tokens', 1024),
                           ('max_completion_tokens', 8192.0), ('hf2q_overflow_policy', 'truncate'),
                           ('tool_choice', 'required'), ('logprobs', True)]:
            with self.subTest(key=key, value=value), self.assertRaises(Invalid):
                fp.request_controls({**base, key: value}, 'deepseek4')
        for key in ('reasoning_effort', 'temperature', 'hf2q_overflow_policy'):
            with self.subTest(missing=key), self.assertRaises(Invalid):
                fp.request_controls({k: v for k, v in base.items() if k != key}, 'deepseek4')

    def test_invalid_message_chain_refuses(self):
        base = self.base('gemma4')
        for messages in (None, [], [None], [{'role': 'developer'}], [{'role': 'assistant', 'tool_calls': {}}]):
            with self.subTest(messages=messages), self.assertRaises(Invalid):
                fp.request_controls({**base, 'messages': messages}, 'gemma4')

    def test_unknown_family_refuses(self):
        with self.assertRaises(Invalid):
            fp.request_controls(self.base('gemma4'), 'llama')
        with self.assertRaises(Invalid):
            fp.canonical({}, 'llama')


class Readiness(unittest.TestCase):
    def setUp(self):
        # The Gemma 4 template ships only inside the GGUF; bind a stand-in for this test.
        patcher = patch.dict(fp.TEMPLATE_SHA256, {'gemma4': hashlib.sha256(TEMPLATE['gemma4'].encode()).hexdigest()})
        patcher.start()
        self.addCleanup(patcher.stop)

    def ok(self, family, **changes):
        health, props, slots = ready(family, **changes)
        if props['chat_template'] is None:
            props['chat_template'] = TEMPLATE[family]
        return fp.validate_readiness(health, props, slots, MODEL[family], family)

    def test_deepseek_pin_is_the_template_hf2q_renders(self):
        self.assertEqual(hashlib.sha256(TEMPLATE['deepseek4'].encode()).hexdigest(),
                         fp.TEMPLATE_SHA256['deepseek4'])

    def test_matched_peer_ready_reports_inspected_source(self):
        for family in fp.FAMILIES:
            result = self.ok(family)
            self.assertEqual((result['family'], result['source_revision'], result['context_per_slot']),
                             (family, fp.PEER_COMMIT, 262144))
            self.assertEqual(result['full_prefix_reuse'], 'PendingPostOracle')

    def test_wrong_template_build_projector_or_sampling_refuse(self):
        params = ready('gemma4')[1]['default_generation_settings']['params']
        for family in fp.FAMILIES:
            other = 'gemma4' if family == 'deepseek4' else 'deepseek4'
            for changes in ({'chat_template': TEMPLATE[other]}, {'chat_template': TEMPLATE[family] + ' '},
                            {'build_info': 'b129-d854d98'}, {'build_info': 'b128-e8'}, {'build_info': None},
                            {'modalities': {'vision': True}}, {'modalities': None}):
                with self.subTest(family=family, changes=changes), self.assertRaises(Invalid):
                    self.ok(family, **changes)
            for key, value in [('temperature', 0.8), ('top_k', 40), ('repeat_penalty', 1.1),
                               ('repeat_last_n', 0), ('speculative.types', 'ngram-simple'), ('top_p', True)]:
                settings = {'n_ctx': 262144, 'params': {**params, key: value}}
                if family == 'deepseek4' and key != 'repeat_penalty':
                    settings['params']['repeat_penalty'] = 1.0
                with self.subTest(family=family, key=key), self.assertRaises(Invalid):
                    self.ok(family, default_generation_settings=settings)
        # Gemma's 1.05 is not DeepSeek's 1.0 and vice versa.
        with self.assertRaises(Invalid):
            self.ok('deepseek4', default_generation_settings={'n_ctx': 262144, 'params': params})

    def test_shared_slot_context_and_alias_contradictions_refuse(self):
        for changes in ({'total_slots': 2}, {'model_alias': 'other'}, {'endpoint_slots': False},
                        {'default_generation_settings': {'n_ctx': 65536, 'params': {}}}):
            with self.subTest(changes=changes), self.assertRaises(Invalid):
                self.ok('deepseek4', **changes)


class Binding(unittest.TestCase):
    def test_family_selection_preflight_and_controls(self):
        for family in fp.FAMILIES:
            validate_selection(family, 'peer', 4)
            keys = fp.preflight(family_preflight(n4_preflight(PREFLIGHT), family), family)
            validate_preflight({k: True for k in keys}, PREFLIGHT, 'peer', family)
            self.assertTrue(fp.PEER_FAMILY_KEYS[family] <= keys)
            self.assertFalse(family_preflight(set(), family) & keys)  # no HF family claim forged
            self.assertFalse({'auto_tq8_text_only_verified', 'pool_profile_selection_verified'} & keys)
            hf_keys = family_preflight(n4_preflight(PREFLIGHT), family)
            with self.assertRaises(Invalid):
                validate_preflight({k: True for k in hf_keys}, PREFLIGHT, 'peer', family)
            packet = {'family': family, 'engine_kind': 'peer', 'family_controls': copy.deepcopy(fp.CONTROLS[family])}
            validate_binding(packet, family, 4)

    def test_hf_controls_cannot_approve_peer_and_peer_controls_cannot_approve_hf(self):
        hf_ds = dict(cache_family='deepseek4', context_per_slot=262144, slots=4, repetition_penalty=1.0,
                     required_tool_thinking_budget=512, prefill_windows='adaptive', graft=False,
                     disk_persistence=False, glp=False, request_temperature=0, request_reasoning_effort='max')
        with self.assertRaises(Invalid):
            validate_binding({'family': 'deepseek4', 'engine_kind': 'peer', 'family_controls': hf_ds}, 'deepseek4', 4)
        with self.assertRaises(Invalid):
            validate_binding({'family': 'deepseek4', 'engine_kind': 'hf2q',
                              'family_controls': fp.CONTROLS['deepseek4']}, 'deepseek4', 4)

    def test_control_drift_and_type_confusion_refuse(self):
        for family, key, value in [('deepseek4', 'kv_unified', True), ('deepseek4', 'reasoning_budget', 2048),
                                   ('deepseek4', 'request_enable_thinking', 1), ('gemma4', 'reasoning', 'on'),
                                   ('gemma4', 'repetition_penalty', 1.0), ('gemma4', 'slots', True),
                                   ('gemma4', 'context_per_slot', 65536), ('gemma4', 'cache_type_k', 'q8_0')]:
            controls = {**fp.CONTROLS[family], key: value}
            with self.subTest(family=family, key=key), self.assertRaises(Invalid):
                fp.validate_controls(controls, family)
        with self.assertRaises(Invalid):
            fp.validate_controls({**fp.CONTROLS['gemma4'], 'extra': 1}, 'gemma4')


if __name__ == '__main__':
    unittest.main()
