"""Explicit HF diagnostic family metadata; never inference routing or fallback."""
from qwen38_n1_verdict import require
FAMILIES = ('qwen', 'deepseek4')


def validate_selection(family, engine_kind, slots):
    require(family in FAMILIES and engine_kind in ('hf2q', 'peer') and slots in (1, 4),
            'unsupported diagnostic selection')
    require(engine_kind != 'peer' or (family == 'qwen' and slots == 4),
            'only Qwen N4 peer is supported; DeepSeek peer has no qualified adapter')


def profile(family, slots):
    require(family in FAMILIES and slots in (1, 4), 'unsupported diagnostic family/slots')
    if family == 'qwen':
        return 'adr063-qwen38-n1-live-v2' if slots == 1 else 'adr063-qwen-n4-live-v1'
    return f'adr063-deepseek4-n{slots}-live-v1'


def schema(family, role, slots):
    profile(family, slots)
    if family == 'qwen':
        if slots == 1: return f'hf2q.qwen38-n1-client-{role}.v2'
        return f'hf2q.qwen-n4-{role}.v1'
    if slots == 1: return f'hf2q.deepseek4-n1-client-{role}.v1'
    return f'hf2q.deepseek4-n4-{role}.v1'


def preflight(keys, family):
    require(family in FAMILIES, 'unsupported diagnostic family')
    if family == 'qwen': return set(keys)
    return (set(keys) - {'auto_tq8_text_only_verified', 'profile24g_ack_verified',
            'pool_profile_selection_verified', 'no_glp_graft_disk_persistence',
            'greedy_omissions_rep105_thinking2048_512', 'auto_lazy_tool_grammar_charged',
            'cpu_owned_io_3g_tool2g_client128m'}) | {
        'deepseek4_native_cache_bound', 'deepseek4_controls_bound',
        'deepseek4_no_graft_disk_persist_glp_bound', 'deepseek4_resource_packet_bound'}


def validate_binding(packet, family, slots):
    profile(family, slots)
    require(packet.get('family', 'qwen') == family, 'mixed diagnostic family')
    if family == 'deepseek4':
        controls = packet.get('family_controls')
        require(isinstance(controls, dict) and
                all(type(controls.get(k)) is int for k in ('context_per_slot', 'slots', 'required_tool_thinking_budget', 'request_temperature')) and
                type(controls.get('repetition_penalty')) in (int, float) and
                all(type(controls.get(k)) is bool for k in ('graft', 'disk_persistence', 'glp')),
                'DeepSeek control types')
        require(controls == {
            'cache_family': 'deepseek4', 'context_per_slot': 262144, 'slots': slots,
            'repetition_penalty': 1.0, 'required_tool_thinking_budget': 512,
            'prefill_windows': 'adaptive', 'graft': False, 'disk_persistence': False,
            'glp': False, 'request_temperature': 0, 'request_reasoning_effort': 'max'},
            'DeepSeek actual selected controls/cache provenance unknown')


def request_controls(request, family):
    require(family in FAMILIES, 'unsupported diagnostic family')
    if family == 'deepseek4':
        # Existing DS agentic gate uses greedy sampling; canonical DS coding
        # controls support native max reasoning. No Qwen thinking kwargs/budget.
        request.update(temperature=0, reasoning_effort='max')
    return request
