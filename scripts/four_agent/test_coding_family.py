"""Pure controls of the actual-used family binding and request seams."""
import copy
import unittest
from coding_family import preflight, profile, schema, validate_binding, request_controls
from qwen38_n1_verdict import Invalid
from qwen38_n1_smoke import PREFLIGHT
from qwen_n4_live_smoke import preflight as n4_preflight, validate_preflight


def controls(slots):
    return dict(cache_family='deepseek4', context_per_slot=262144, slots=slots,
                repetition_penalty=1.0, required_tool_thinking_budget=512,
                prefill_windows='adaptive', graft=False, disk_persistence=False, glp=False,
                request_temperature=0, request_reasoning_effort='max')


class FamilyControls(unittest.TestCase):
    def test_qwen_default_contract_and_request_are_unchanged(self):
        self.assertEqual(profile('qwen', 1), 'adr063-qwen38-n1-live-v2')
        self.assertEqual(profile('qwen', 4), 'adr063-qwen-n4-live-v1')
        self.assertEqual(schema('qwen', 'approval', 1), 'hf2q.qwen38-n1-client-approval.v2')
        request = {'model': 'same', 'messages': [{'role': 'user', 'content': 'same'}],
                   'tools': [{'function': {'strict': False}}], 'max_completion_tokens': 8192}
        before = copy.deepcopy(request)
        self.assertIs(request_controls(request, 'qwen'), request)
        self.assertEqual(request, before)
        self.assertEqual(preflight(PREFLIGHT, 'qwen'), PREFLIGHT)
    def test_ds_separates_preflight_from_qwen_tq_and_pool_flags(self):
        keys = preflight(n4_preflight(PREFLIGHT), 'deepseek4')
        self.assertNotIn('auto_tq8_text_only_verified', keys)
        self.assertNotIn('pool_profile_selection_verified', keys)
        self.assertIn('deepseek4_native_cache_bound', keys)
        validate_preflight({k: True for k in keys}, PREFLIGHT, 'deepseek4')
        with self.assertRaises(Invalid):
            validate_preflight({k: True for k in n4_preflight(PREFLIGHT)}, PREFLIGHT, 'deepseek4')
    def test_actual_ds_binding_rejects_missing_mixed_or_divided_family(self):
        p = {'family': 'deepseek4', 'family_controls': controls(4)}
        validate_binding(p, 'deepseek4', 4)
        for bad in ({}, dict(p, family='qwen'), dict(p, family_controls=controls(1))):
            with self.assertRaises(Invalid): validate_binding(bad, 'deepseek4', 4)
        p['family_controls']['context_per_slot'] = 65536
        with self.assertRaises(Invalid): validate_binding(p, 'deepseek4', 4)
    def test_native_cache_and_control_mismatches_reject(self):
        for key, value in [('cache_family', 'qwen35'), ('required_tool_thinking_budget', 2048),
                           ('repetition_penalty', 1.05), ('graft', True),
                           ('prefill_windows', 1), ('repetition_penalty', True), ('slots', True)]:
            p = {'family': 'deepseek4', 'family_controls': controls(1)}
            p['family_controls'][key] = value
            with self.assertRaises(Invalid): validate_binding(p, 'deepseek4', 1)
    def test_ds_request_only_adds_native_supported_controls(self):
        request = {'messages': ['actual'], 'tools': ['same'], 'max_completion_tokens': 8192,
                   'tool_choice': 'auto', 'hf2q_overflow_policy': 'reject'}
        before = copy.deepcopy(request)
        request_controls(request, 'deepseek4')
        self.assertEqual({k: request[k] for k in before}, before)
        self.assertEqual(request['temperature'], 0)
        self.assertEqual(request['reasoning_effort'], 'max')
        self.assertNotIn('chat_template_kwargs', request)
        self.assertNotIn('thinking_budget', request)
    def test_family_schema_distinct_and_unsupported_family_refuses(self):
        self.assertNotEqual(schema('qwen', 'agent-result', 4), schema('deepseek4', 'agent-result', 4))
        with self.assertRaises(Invalid): profile('approximately-deepseek', 4)
        with self.assertRaises(Invalid): request_controls({}, 'gemma4')

if __name__ == '__main__': unittest.main()
