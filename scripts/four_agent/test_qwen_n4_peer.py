"""Pure controls for actual peer adapter functions. No HTTP/server/tool execution."""
import unittest
from qwen38_n1_verdict import Invalid, usage as hf_usage
from qwen38_n1_smoke import PREFLIGHT
from qwen_n4_live_smoke import validate_preflight, preflight as hf_preflight
from qwen_n4_peer import preflight, usage, validate_readiness


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

if __name__ == '__main__': unittest.main()
