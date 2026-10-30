"""Explicit d854d98b peer metadata adapter; no server launch or inference fallback."""
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen
from qwen38_n1_verdict import integer, require

PEER_COMMIT = 'd854d98bae598388ecea1ed84396587e5edc326d'
PEER_ONLY = {'peer_settings_provenance_bound', 'peer_context_slots_readiness_verified',
             'peer_image_free_workload_verified', 'peer_cache_usage_semantics_bound',
             'peer_generation_settings_bound', 'peer_tool_template_settings_bound'}
HF_ONLY = {'auto_tq8_text_only_verified', 'pool_profile_selection_verified',
           'profile24g_ack_verified', 'no_glp_graft_disk_persistence',
           'greedy_omissions_rep105_thinking2048_512', 'auto_lazy_tool_grammar_charged'}


def preflight(hf_keys):
    return (set(hf_keys) - HF_ONLY) | PEER_ONLY


def request_controls(request):
    """Translate only canonical HF Qwen defaults into native peer metadata.

    Mirrors qwen_thinking_policy::{qwen_tool_chain_state,
    adaptive_qwen_default_thinking_budget}: a fresh user turn gets 2048 and
    every tool-result continuation gets 512; system messages do not change
    whether the latest conversation role is tool.
    """
    overrides = {'reasoning_budget_tokens', 'thinking_budget_tokens',
                 'thinking_token_budget', 'tool_thinking_token_budget',
                 'thinking_budget', 'reasoning_effort', 'reasoning',
                 'reasoning_control', 'reasoning_budget_message',
                 'chat_template_kwargs', 'hf2q_enable_thinking', 'enable_thinking'}
    require(not overrides.intersection(request), 'unsupported peer explicit reasoning override')
    require(request.get('hf2q_overflow_policy') == 'reject', 'canonical overflow policy required')
    require(request.get('max_completion_tokens') == 8192, 'canonical output allowance required')
    messages = request.get('messages')
    require(isinstance(messages, list) and bool(messages), 'peer message chain required')
    for message in messages:
        require(isinstance(message, dict) and message.get('role') in
                ('system', 'user', 'assistant', 'tool'), 'invalid peer message role')
        if message['role'] == 'assistant':
            calls = message.get('tool_calls')
            require(calls is None or isinstance(calls, list), 'invalid assistant tool-call cycles')
    latest = next((m for m in reversed(messages) if m['role'] != 'system'), None)
    budget = 2048
    if latest is not None and latest['role'] == 'tool':
        budget = 512
    translated = dict(request)
    del translated['hf2q_overflow_policy']
    translated['reasoning_budget_tokens'] = budget
    return translated


def usage(value, turn):
    require(isinstance(value, dict), 'missing peer usage')
    prompt = integer(value.get('prompt_tokens'), 'peer prompt tokens', 262144)
    completion = integer(value.get('completion_tokens'), 'peer completion tokens', 8192)
    require(integer(value.get('total_tokens'), 'peer total tokens') == prompt + completion,
            'peer usage total contradiction')
    require(prompt + 8192 <= 262144, 'peer context plus full output exceeds context')
    details = value.get('prompt_tokens_details')
    require(details is None or isinstance(details, dict), 'malformed peer cache details')
    cached = details.get('cached_tokens') if isinstance(details, dict) else None
    if cached is not None:
        integer(cached, 'peer cached tokens', prompt)
    # Preserve absent/zero honestly; protocol completion is not a cache verdict.
    return {**value, 'engine_kind': 'peer', 'cache_observation': {
        'status': 'Unavailable' if cached is None else 'Observed',
        'cached_tokens': cached, 'source_field': 'usage.prompt_tokens_details.cached_tokens',
        'full_prefix_reuse': 'PendingPostOracle'}}


def validate_readiness(health, props, slots, model):
    require(isinstance(health, dict) and health.get('status') == 'ok', 'peer not healthy')
    require(isinstance(props, dict) and props.get('total_slots') == 4 and
            props.get('default_generation_settings', {}).get('n_ctx') == 262144,
            'peer requires four FULL per-sequence contexts')
    require(props.get('model_alias') == model, 'peer model alias mismatch')
    require(props.get('endpoint_slots') is True and isinstance(slots, list) and len(slots) == 4,
            'peer --slots endpoint required')
    require({s.get('id') for s in slots} == set(range(4)) and
            all(type(s.get('id')) is int and s.get('n_ctx') == 262144 for s in slots), 'peer slot/context contradiction')
    return {'engine_kind': 'peer', 'source_revision': PEER_COMMIT,
            'logical_slots': 4, 'context_per_slot': 262144,
            'physical_cohort': 'Unavailable:not proved by readiness or HTTP overlap',
            'pool_profile': 'NotApplicable:HF-only', 'full_prefix_reuse': 'PendingPostOracle'}


def fetch_readiness(base, output):
    """Raw /health, /props and /slots replies, archived before any validation."""
    u = urlsplit(base)
    require(u.scheme == 'http' and u.hostname == '127.0.0.1' and u.port and u.path == '/v1'
            and not u.query and not u.fragment and u.username is None, 'peer loopback base')
    values = []
    for name in ('health', 'props', 'slots'):
        with urlopen(f'http://127.0.0.1:{u.port}/{name}', timeout=2) as response:
            raw = response.read(1024 * 1024 + 1)
            (Path(output) / f'peer-{name}.raw').write_bytes(raw)
            require(response.status == 200 and len(raw) <= 1024 * 1024, 'peer readiness status/size')
        values.append(json.loads(raw))
    return values


def readiness(base, model, output):
    return validate_readiness(*fetch_readiness(base, output), model)
