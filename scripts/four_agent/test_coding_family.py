"""Pure controls of the actual-used family binding and request seams."""
import copy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from coding_family import preflight, profile, schema, validate_binding, request_controls, validate_selection
from qwen38_n1_verdict import Invalid
from qwen38_n1_smoke import PREFLIGHT, build_request, exchange
from qwen38_n1_verdict import wire
from qwen_n4_live_smoke import preflight as n4_preflight, validate_preflight


def controls(slots):
    return dict(cache_family='deepseek4', context_per_slot=262144, slots=slots,
                repetition_penalty=1.0, required_tool_thinking_budget=512,
                prefill_windows='adaptive', graft=False, disk_persistence=False, glp=False,
                request_temperature=0, request_reasoning_effort='max')


class FamilyControls(unittest.TestCase):
    def test_only_supported_engine_family_selections_can_reach_work(self):
        for family, engine, slots in [('qwen', 'hf2q', 1), ('qwen', 'hf2q', 4),
                                      ('qwen', 'peer', 4), ('deepseek4', 'hf2q', 1),
                                      ('deepseek4', 'hf2q', 4), ('deepseek4', 'peer', 4),
                                      ('gemma4', 'peer', 4)]:
            validate_selection(family, engine, slots)
        for family, engine, slots in [('deepseek4', 'peer', 1), ('gemma4', 'peer', 1),
                                      ('qwen', 'peer', 1), ('llama', 'peer', 4)]:
            with self.assertRaises(Invalid): validate_selection(family, engine, slots)
    def test_actual_preflight_composes_family_and_engine_without_forging(self):
        from qwen_n4_peer import preflight as peer_preflight
        qwen_peer = peer_preflight(n4_preflight(PREFLIGHT))
        validate_preflight({k: True for k in qwen_peer}, PREFLIGHT, 'peer', 'qwen')
        ds_hf = preflight(n4_preflight(PREFLIGHT), 'deepseek4')
        validate_preflight({k: True for k in ds_hf}, PREFLIGHT, 'hf2q', 'deepseek4')
        with self.assertRaises(Invalid):
            validate_preflight({k: True for k in qwen_peer}, PREFLIGHT, 'peer', 'deepseek4')
        with self.assertRaises(Invalid):
            validate_preflight({k: True for k in qwen_peer}, PREFLIGHT, 'hf2q', 'deepseek4')
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
        validate_preflight({k: True for k in keys}, PREFLIGHT, family='deepseek4')
        with self.assertRaises(Invalid):
            validate_preflight({k: True for k in n4_preflight(PREFLIGHT)}, PREFLIGHT, family='deepseek4')
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
        with self.assertRaises(Invalid): request_controls({}, 'unsupported-gemma')


    def test_actual_builder_then_exchange_wire_accepts_both_families_all_turns(self):
        schemas = [{"type": "function", "function": {"name": name, "strict": False}}
                   for name in ("read_file", "write_file", "run_tests")]
        messages = [{"role": "user", "content": "actual coding task"}]
        for family in ("qwen", "deepseek4", "gemma4"):
            for turn in range(7):
                with self.subTest(family=family, turn=turn), tempfile.TemporaryDirectory() as tmp:
                    request = build_request("actual-model", messages, schemas, turn, family)
                    # Reach the real exchange's wire validation/archive boundary.
                    # Constructor sentinel prevents any socket/HTTP execution.
                    with patch("qwen38_n1_smoke.http.client.HTTPConnection",
                               side_effect=RuntimeError("validated-before-HTTP")):
                        with self.assertRaisesRegex(RuntimeError, "validated-before-HTTP"):
                            exchange("http://127.0.0.1:18240/v1", request, Path(tmp),
                                     turn, set(), family=family)
                    observed = json.loads((Path(tmp) / "request.json").read_bytes())
                    self.assertEqual(observed, request)
                    self.assertEqual(observed["messages"], messages)
                    self.assertEqual(observed["tools"], schemas)
                    self.assertEqual(observed["max_completion_tokens"], 8192)
                    self.assertEqual(observed["stream"], turn % 2 == 1)
                    if family == "deepseek4":
                        qwen = build_request("actual-model", messages, schemas, turn, "qwen")
                        self.assertEqual({k: v for k, v in observed.items()
                                          if k not in ("temperature", "reasoning_effort")}, qwen)
                        self.assertEqual(observed["temperature"], 0)
                        self.assertEqual(observed["reasoning_effort"], "max")
                    else:
                        self.assertNotIn("temperature", observed)
                        self.assertNotIn("reasoning_effort", observed)

    def test_actual_ds_builder_rejects_wrong_family_missing_or_changed_controls(self):
        schemas = [{"function": {"name": n, "strict": False}}
                   for n in ("read_file", "write_file", "run_tests")]
        request = build_request("DS", [], schemas, 0, "deepseek4")
        with self.assertRaises(Invalid): wire(request)  # Default Qwen stays strict.
        for key in ("temperature", "reasoning_effort"):
            bad = dict(request)
            del bad[key]
            with self.subTest(missing=key), self.assertRaises(Invalid):
                wire(bad, "deepseek4")
        for key, value in (("temperature", True), ("temperature", 0.5),
                           ("reasoning_effort", "none"), ("seed", 0),
                           ("repetition_penalty", 1.05), ("chat_template_kwargs", {}),
                           ("thinking_budget", 512), ("thinking_token_budget", 512),
                           ("hf2q_enable_thinking", False), ("max_completion_tokens", 128)):
            with self.subTest(changed=key), self.assertRaises(Invalid):
                wire({**request, key: value}, "deepseek4")
        with self.assertRaises(Invalid): wire(request, "approximately-deepseek")

    def test_gemma_actual_binding_and_wire_do_not_borrow_other_family_contracts(self):
        selected = dict(cache_family='gemma4', context_per_slot=262144, slots=4,
                        hybrid_kv=True, codebook_bits=8, lcp_capacity_bytes=8589934592,
                        repetition_penalty=1.05, prefill_route='auto', graft=False,
                        disk_persistence=False, glp=False, external_drafter=False)
        packet = dict(family='gemma4', family_controls=selected)
        validate_selection('gemma4', 'hf2q', 4)
        validate_binding(packet, 'gemma4', 4)
        keys = preflight(n4_preflight(PREFLIGHT), 'gemma4')
        self.assertIn('gemma4_hybrid_cache_bound', keys)
        self.assertNotIn('pool_profile_selection_verified', keys)
        self.assertNotIn('deepseek4_native_cache_bound', keys)
        validate_preflight({k: True for k in keys}, PREFLIGHT, family='gemma4')
        for key, value in [('hybrid_kv', False), ('cache_family', 'qwen35'),
                           ('slots', True), ('codebook_bits', 4), ('external_drafter', True)]:
            with self.subTest(key=key), self.assertRaises(Invalid):
                validate_binding(dict(packet, family_controls={**selected, key:value}), 'gemma4', 4)
        schemas = [{"function":{"name":n,"strict":False}}
                   for n in ('read_file','write_file','run_tests')]
        req = build_request('Gemma', [], schemas, 0, 'gemma4')
        wire(req, 'gemma4')
        for key, value in [('reasoning_effort','max'), ('temperature',0),
                           ('chat_template_kwargs',{}), ('hf2q_enable_thinking',False)]:
            with self.subTest(wire_key=key), self.assertRaises(Invalid):
                wire({**req,key:value}, 'gemma4')
        with self.assertRaises(Invalid): validate_selection('gemma4','peer',1)

if __name__ == '__main__': unittest.main()
