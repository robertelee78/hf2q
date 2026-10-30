"""DeepSeek-V4 and Gemma 4 peer adapters plus the per-family peer dispatch.

Harness-only: no server launch, model load or inference fallback. Each adapter
translates the canonical hf2q coding request into the peer's native request
fields, refuses every field outside the closed canonical envelope, and checks
the peer's readiness against the settings matched to hf2q for that family.

Peer source inspected: e86f5fd1 (the build in peer-e86-release-build-v1).
Field sources (tools/server/server-common.cpp at e86f5fd1):
- chat_template_kwargs.enable_thinking selects the template thinking mode (1479-1493).
- reasoning_effort is passed to the template as a kwarg (1495-1504); the DeepSeek
  V4 template renders the `max` tier itself (only V4.1 remaps effort names,
  common/parsers/deepseek.cpp:111-131).
- parallel_tool_calls (1445) bounds the native parsers to one call
  (common/parsers/deepseek.cpp:247, common/parsers/gemma4.cpp:272).
- max_completion_tokens is the n_predict alias (server-schema.cpp:48).
hf2q_overflow_policy=reject has no request field: the launch binds
--no-context-shift, and an over-long prompt is rejected at
tools/server/server-context.cpp:3975-3991.
"""
import hashlib
import math

import qwen_n4_peer
from qwen38_n1_verdict import require

PEER_COMMIT = 'e86f5fd1e9f584b0fcec71d7da9a92de44e8bcbf'
CONTEXT_PER_SLOT = 262144
SLOTS = 4
FAMILIES = ('deepseek4', 'gemma4')
SHARED_KEYS = frozenset({'model', 'messages', 'tools', 'tool_choice', 'max_completion_tokens',
                         'logprobs', 'hf2q_overflow_policy', 'stream'})
# The exact canonical envelope qwen38_n1_smoke.build_request produces per family.
CANONICAL_KEYS = {'deepseek4': SHARED_KEYS | {'temperature', 'reasoning_effort'},
                  'gemma4': SHARED_KEYS}
# hf2q renders thinking for DeepSeek-V4 (reasoning_effort=max, no ceiling) and
# none for Gemma 4 (its template has no thinking channel for hf2q); hf2q's
# parallel_tool_calls default is false (serve/api/handlers.rs:1762-1764).
NATIVE = {'deepseek4': {'chat_template_kwargs': {'enable_thinking': True}, 'parallel_tool_calls': False},
          'gemma4': {'chat_template_kwargs': {'enable_thinking': False}, 'parallel_tool_calls': False}}
# Family-native templates hf2q renders: DeepSeek-V4 is
# src/core/chat_templates/deepseek-v4-flash-0731.jinja (byte-identical to the
# GGUF template); Gemma 4 is the GGUF-embedded template. The peer must serve
# exactly these so DSML and Gemma tool-call encodings are the hf2q ones.
TEMPLATE_SHA256 = {'deepseek4': 'aec3d906bce80d43a9e6e89922c04e8e752c05f71f10040ba3f61b746ebacd6d',
                   'gemma4': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4'}
# Greedy sampling with the hf2q family repetition default; sampling fields are
# server defaults because the canonical request omits them.
SAMPLING = {'deepseek4': {'temperature': 0.0, 'top_k': 0, 'top_p': 1.0, 'min_p': 0.0,
                          'repeat_penalty': 1.0, 'repeat_last_n': 64},
            'gemma4': {'temperature': 0.0, 'top_k': 0, 'top_p': 1.0, 'min_p': 0.0,
                       'repeat_penalty': 1.05, 'repeat_last_n': 64}}
# Launch settings the operator approval attests (and the Q5 peer case launches).
CONTROLS = {
    'deepseek4': {'cache_family': 'deepseek4', 'context_per_slot': CONTEXT_PER_SLOT, 'slots': SLOTS,
                  'kv_unified': False, 'cache_type_k': 'f16', 'cache_type_v': 'f16', 'flash_attn': 'on',
                  'context_shift': False, 'repetition_penalty': 1.0, 'reasoning': 'on',
                  'reasoning_budget': -1, 'reasoning_preserve': True, 'speculation': 'none',
                  'request_temperature': 0, 'request_reasoning_effort': 'max',
                  'request_enable_thinking': True, 'request_parallel_tool_calls': False},
    'gemma4': {'cache_family': 'gemma4', 'context_per_slot': CONTEXT_PER_SLOT, 'slots': SLOTS,
               'kv_unified': True, 'cache_type_k': 'f16', 'cache_type_v': 'f16', 'flash_attn': 'on',
               'context_shift': False, 'repetition_penalty': 1.05, 'reasoning': 'off',
               'reasoning_preserve': False, 'speculation': 'none', 'vision_projector': False,
               'request_enable_thinking': False, 'request_parallel_tool_calls': False}}
PEER_FAMILY_KEYS = {family: {f'peer_{family}_template_bound', f'peer_{family}_controls_bound'}
                    for family in FAMILIES}


def same(left, right):
    """Equality that never lets bool stand in for int or int for str."""
    if isinstance(right, dict):
        return isinstance(left, dict) and set(left) == set(right) and all(
            same(left[k], right[k]) for k in right)
    if type(right) is bool or type(left) is bool:
        return type(left) is type(right) and left == right
    if type(right) in (int, float):
        return type(left) in (int, float) and left == right
    return type(left) is type(right) and left == right


def _family(family):
    require(family in FAMILIES, 'unsupported peer family')


def preflight(keys, family):
    """Peer preflight: HF family/runtime keys are replaced, never forged."""
    if family == 'qwen':
        return qwen_n4_peer.preflight(keys)
    _family(family)
    from coding_family import preflight as family_preflight
    hf_family_keys = family_preflight(set(), family)
    return qwen_n4_peer.preflight(set(keys) - hf_family_keys) | PEER_FAMILY_KEYS[family]


def validate_controls(controls, family):
    _family(family)
    require(same(controls, CONTROLS[family]), f'{family} peer launch controls differ from the matched settings')


def _messages(messages):
    require(isinstance(messages, list) and bool(messages), 'peer message chain required')
    for message in messages:
        require(isinstance(message, dict) and message.get('role') in
                ('system', 'user', 'assistant', 'tool'), 'invalid peer message role')
        if message['role'] == 'assistant':
            calls = message.get('tool_calls')
            require(calls is None or isinstance(calls, list), 'invalid assistant tool-call cycles')


def request_controls(request, family):
    """Canonical hf2q request -> native peer request; any other field refuses."""
    if family == 'qwen':
        return qwen_n4_peer.request_controls(request)
    _family(family)
    extra = sorted(set(request) - CANONICAL_KEYS[family])
    require(not extra, 'unsupported peer override: ' + ','.join(extra))
    require(set(request) == CANONICAL_KEYS[family], 'incomplete canonical peer request')
    require(request['hf2q_overflow_policy'] == 'reject', 'canonical overflow policy required')
    require(type(request['max_completion_tokens']) is int and request['max_completion_tokens'] == 8192,
            'canonical output allowance required')
    require(request['tool_choice'] == 'auto' and request['logprobs'] is False
            and type(request['stream']) is bool, 'canonical tool/transport controls required')
    if family == 'deepseek4':
        require(type(request['temperature']) in (int, float) and request['temperature'] == 0,
                'DeepSeek greedy temperature required')
        require(request['reasoning_effort'] == 'max', 'DeepSeek max reasoning required')
    _messages(request['messages'])
    translated = {k: v for k, v in request.items() if k != 'hf2q_overflow_policy'}
    for key, value in NATIVE[family].items():
        translated[key] = dict(value) if isinstance(value, dict) else value
    return translated


def canonical(request, family):
    """Inverse of request_controls at the wire boundary: native fields must be exact."""
    require('hf2q_overflow_policy' not in request, 'peer request must use native transport metadata')
    if family == 'qwen':
        return {**request, 'hf2q_overflow_policy': 'reject'}
    _family(family)
    for key, value in NATIVE[family].items():
        require(key in request and same(request[key], value), 'peer native control drift: ' + key)
    stripped = {k: v for k, v in request.items() if k not in NATIVE[family]}
    return {**stripped, 'hf2q_overflow_policy': 'reject'}


def _close(value, expected):
    # /props serializes float32 sampling values (1.05 -> 1.0499999523162842).
    return type(value) in (int, float) and type(value) is not bool and math.isclose(
        value, expected, rel_tol=0, abs_tol=1e-6)


def validate_readiness(health, props, slots, model, family):
    if family == 'qwen':
        return qwen_n4_peer.validate_readiness(health, props, slots, model)
    _family(family)
    qwen_n4_peer.validate_readiness(health, props, slots, model)  # shared slot/context/alias checks
    build = props.get('build_info')
    commit = build.split('-', 1)[1] if isinstance(build, str) and '-' in build else ''
    require(len(commit) >= 7 and PEER_COMMIT.startswith(commit), 'peer build is not the inspected source')
    template = props.get('chat_template')
    require(isinstance(template, str) and hashlib.sha256(template.encode()).hexdigest()
            == TEMPLATE_SHA256[family], f'{family} peer template is not the hf2q family template')
    modalities = props.get('modalities')
    require(isinstance(modalities, dict) and modalities.get('vision') is False,
            'peer must serve the text-only workload without a projector')
    params = props.get('default_generation_settings', {}).get('params')
    require(isinstance(params, dict) and all(_close(params.get(k), v) for k, v in SAMPLING[family].items())
            and params.get('speculative.types') == 'none', f'{family} peer sampling settings differ')
    return {'engine_kind': 'peer', 'family': family, 'source_revision': PEER_COMMIT, 'build_info': build,
            'chat_template_sha256': TEMPLATE_SHA256[family], 'logical_slots': SLOTS,
            'context_per_slot': CONTEXT_PER_SLOT, 'sampling': dict(SAMPLING[family]),
            'physical_cohort': 'Unavailable:not proved by readiness or HTTP overlap',
            'pool_profile': 'NotApplicable:HF-only', 'full_prefix_reuse': 'PendingPostOracle'}


def readiness(base, model, output, family):
    return validate_readiness(*qwen_n4_peer.fetch_readiness(base, output), model, family)
