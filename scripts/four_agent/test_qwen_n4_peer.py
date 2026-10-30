"""Pure controls for actual peer adapter functions. No HTTP/server/tool execution."""
import copy
import json
from unittest.mock import patch
import unittest
from qwen38_n1_verdict import Invalid, usage as hf_usage, wire
from qwen38_n1_smoke import PREFLIGHT, build_request
from qwen_n4_live_smoke import validate_preflight, preflight as hf_preflight
from qwen_n4_peer import preflight, usage, validate_readiness, request_controls


def ready():
    return ({'status': 'ok'}, {'total_slots': 4, 'model_alias': 'bound-model',
            'endpoint_slots': True, 'default_generation_settings': {'n_ctx': 262144}},
            [{'id': i, 'n_ctx': 262144} for i in range(4)])


class PeerAdapter(unittest.TestCase):
    def test_actual_context_is_per_sequence_not_divided_total(self):
        h, p, slots = ready()
        self.assertEqual(validate_readiness(h, p, slots, 'bound-model')['context_per_slot'], 262144)
        p['default_generation_settings']['n_ctx'] = 65536
        with self.assertRaises(Invalid): validate_readiness(h, p, slots, 'bound-model')
    def test_slot_alias_health_and_endpoint_contradictions_reject(self):
        for mutation in ('health', 'alias', 'slots', 'ctx', 'endpoint'):
            h, p, slots = ready()
            if mutation == 'health': h['status'] = 'loading model'
            if mutation == 'alias': p['model_alias'] = 'foreign'
            if mutation == 'slots': slots[3]['id'] = 2
            if mutation == 'ctx': slots[2]['n_ctx'] = 65536
            if mutation == 'endpoint': p['endpoint_slots'] = False
            with self.assertRaises(Invalid): validate_readiness(h, p, slots, 'bound-model')
    def test_missing_cache_is_unknown_never_zero_or_hf_claim(self):
        raw = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}
        got = usage(raw, 1)
        self.assertEqual(got['cache_observation']['status'], 'Unavailable')
        self.assertIsNone(got['cache_observation']['cached_tokens'])
        self.assertNotIn('prompt_tokens_details', raw)
        with self.assertRaises(Invalid): hf_usage(raw, 1)
    def test_peer_cache_field_preserved_and_invalid_counts_reject(self):
        raw = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12,
               'prompt_tokens_details': {'cached_tokens': 0}}
        self.assertEqual(usage(raw, 1)['cache_observation']['cached_tokens'], 0)
        self.assertEqual(usage(raw, 1)['cache_observation']['full_prefix_reuse'], 'PendingPostOracle')
        for bad in (-1, 11, True):
            raw['prompt_tokens_details']['cached_tokens'] = bad
            with self.assertRaises(Invalid): usage(raw, 1)
    def test_peer_preflight_cannot_forge_hf_profile_or_tq_flags(self):
        keys = preflight(hf_preflight(PREFLIGHT))
        self.assertNotIn('pool_profile_selection_verified', keys)
        self.assertNotIn('auto_tq8_text_only_verified', keys)
        validate_preflight({k: True for k in keys}, PREFLIGHT, 'peer')
        with self.assertRaises(Invalid):
            validate_preflight({k: True for k in hf_preflight(PREFLIGHT)}, PREFLIGHT, 'peer')
    def test_hf_default_usage_contract_remains_strict(self):
        raw = {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12,
               'prompt_tokens_details': {'cached_tokens': 0}}
        self.assertIs(hf_usage(raw, 0), raw)
        with self.assertRaises(Invalid): hf_usage(raw, 1)
        raw['prompt_tokens_details']['cached_tokens'] = 5
        self.assertIs(hf_usage(raw, 1), raw)

class PeerThinkingParity(unittest.TestCase):
    def setUp(self):
        self.messages = [{'role': 'system', 'content': 'unchanged'},
                         {'role': 'user', 'content': 'same coding task'}]
        self.schemas = [{'type': 'function', 'function': {'name': name,
                        'strict': False, 'parameters': {'type': 'object', 'properties': {}}}}
                        for name in ('read_file', 'write_file', 'run_tests')]

    def pair(self, messages, turn):
        original = copy.deepcopy((messages, self.schemas))
        hf = build_request('Qwen3.6 35B A3B', messages, self.schemas, turn)
        peer = build_request('Qwen3.6 35B A3B', messages, self.schemas, turn,
                             'qwen', 'peer')
        budget = peer['reasoning_budget_tokens']
        self.assertEqual({k: v for k, v in peer.items() if k != 'reasoning_budget_tokens'},
                         {k: v for k, v in hf.items() if k != 'hf2q_overflow_policy'})
        self.assertEqual(hf['hf2q_overflow_policy'], 'reject')
        self.assertNotIn('reasoning_budget_tokens', hf)
        self.assertEqual((messages, self.schemas), original)
        self.assertEqual(peer['max_completion_tokens'], 8192)
        self.assertEqual(peer['stream'], turn % 2 == 1)
        self.assertEqual(json.loads(wire(hf))['messages'], peer['messages'])
        wire({**peer, 'hf2q_overflow_policy': 'reject'})
        return budget

    def test_actual_builder_all_seven_turns_native_only_delta(self):
        budgets = []
        for turn in range(7):
            budgets.append(self.pair(self.messages, turn))
            self.messages.extend([{'role': 'assistant', 'tool_calls': [{'id': str(turn)}]},
                                  {'role': 'tool', 'content': 'actual result placeholder'}])
        self.assertEqual(budgets, [2048, 512, 256, 256, 256, 256, 256])

    def test_latest_user_resets_cycles_and_system_does_not_hide_tool(self):
        for turn in range(4):
            self.messages.extend([{'role': 'assistant', 'tool_calls': [{'id': str(turn)}]},
                                  {'role': 'tool', 'content': 'result'}])
        self.messages.append({'role': 'system', 'content': 'unchanged metadata'})
        self.assertEqual(self.pair(self.messages, 4), 256)
        self.messages.append({'role': 'user', 'content': 'new user'})
        self.assertEqual(self.pair(self.messages, 0), 2048)
        self.messages.extend([{'role': 'assistant', 'tool_calls': [{'id': 'new'}]},
                              {'role': 'tool', 'content': 'new result'}])
        self.assertEqual(self.pair(self.messages, 1), 512)

    def test_one_assistant_cycle_not_parallel_result_or_plain_assistant_count(self):
        self.messages.extend([{'role': 'assistant', 'content': 'plain'},
                              {'role': 'assistant', 'tool_calls': []},
                              {'role': 'assistant', 'tool_calls': [{'id': 'a'}, {'id': 'b'}]},
                              {'role': 'tool', 'content': 'a'}, {'role': 'tool', 'content': 'b'}])
        self.assertEqual(self.pair(self.messages, 1), 512)
        self.messages.append({'role': 'assistant', 'content': 'not tool continuation'})
        self.assertEqual(self.pair(self.messages, 2), 2048)

    def test_explicit_reasoning_or_disabled_thinking_refuses_not_overwrites(self):
        base = build_request('model', self.messages, self.schemas, 0)
        for key, value in [('reasoning_budget_tokens', 2048), ('thinking_budget_tokens', 0),
                ('thinking_token_budget', None), ('tool_thinking_token_budget', 512),
                ('thinking_budget', 0), ('reasoning_effort', 'none'), ('reasoning', {}),
                ('reasoning_control', 'none'), ('reasoning_budget_message', 'override'),
                ('chat_template_kwargs', {'enable_thinking': False}),
                ('hf2q_enable_thinking', False), ('enable_thinking', False)]:
            with self.subTest(key=key), self.assertRaises(Invalid):
                request_controls({**base, key: value})

    def test_actual_builder_rejects_injected_override_before_wire(self):
        for key, value in [('chat_template_kwargs', {'enable_thinking': False}),
                           ('reasoning_control', 'none'),
                           ('reasoning_budget_message', 'override')]:
            def inject(request, family):
                request[key] = value
            with self.subTest(key=key), patch(
                    'qwen38_n1_smoke.family_contract.request_controls', side_effect=inject):
                with self.assertRaises(Invalid):
                    build_request('model', self.messages, self.schemas, 0, 'qwen', 'peer')

    def test_invalid_chain_and_output_envelope_refuse(self):
        base = build_request('model', self.messages, self.schemas, 0)
        for messages in (None, [], [None], [{'role': 'developer'}],
                         [{'role': 'assistant', 'tool_calls': {}}]):
            with self.subTest(messages=messages), self.assertRaises(Invalid):
                request_controls({**base, 'messages': messages})
        for patch in ({'max_completion_tokens': 1024}, {'hf2q_overflow_policy': 'truncate'}):
            with self.assertRaises(Invalid): request_controls({**base, **patch})

    def test_nonqwen_and_hf_builder_routes_remain_explicit(self):
        for family in ('deepseek4', 'gemma4'):
            hf = build_request('model', self.messages, self.schemas, 0, family)
            self.assertNotIn('reasoning_budget_tokens', hf)
            self.assertEqual(hf['hf2q_overflow_policy'], 'reject')
            peer = build_request('model', self.messages, self.schemas, 0, family, 'peer')
            self.assertNotIn('reasoning_budget_tokens', peer)  # never the Qwen adaptive budget
            self.assertNotIn('hf2q_overflow_policy', peer)
        with self.assertRaises(Invalid):
            build_request('model', self.messages, self.schemas, 0, 'qwen', 'unknown')


if __name__ == '__main__': unittest.main()
