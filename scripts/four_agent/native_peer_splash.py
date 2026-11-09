"""Splash (Apple-silicon engine with DFlash2 speculation) arm: engine_kind 'splash'.

Harness-only: no server launch, model load or inference fallback. Splash is a same-artifact peer of record for the
Qwen families (ADR-063 "Splash peer of record for the Qwen families", design
docs/research/adr063-splash-peer-design.md): it serves the hf2q GGUF of the family's hf2q arm (same path and sha256)
through its local-file mode, with its family's pinned DFlash2 draft. It is never a native comparison and never
stands in for the peer: Q3 runs it as its own measured arm 'splash' beside 'peer'.

Source inspected: incoai/splash at 031a25a6 (git describe 1.5.0-29-g031a25a6), built from a clean detached clone
(build_id below = /status identity.cache.build_id = build/engine/build-identity.json). Paths are in that checkout.
Request (server/frontend.py, server/api_shapes.py; docs/api.md "Request fields", "Sampling", "Tool calls"):
- Chat fields are read with body.get and unknown fields are ignored, so this adapter's closed envelope is the only
  guard. logprobs true and n != 1 return 400; logprobs false is accepted.
- temperature 0 is argmax (server/protocol.py:203-207, 717-718); top_p 1, top_k 0 (keep all), min_p 0 and zero
  presence/frequency penalties change nothing and are sent explicitly. No repetition_penalty is sent (operator
  decision D3): Splash's penalty counts every prompt and output token (docs/api.md Sampling), unlike hf2q's 1.05 over
  a 64-token window seeded from the prompt tail; the difference is recorded (REPETITION_PARITY).
- Thinking as hf2q renders Qwen: chat_template_kwargs {enable_thinking: true} and preserve_thinking true (a top-level
  Chat field Splash passes to the template), no reasoning_effort (the template default: xhigh on Qwen3.8, no tiers on
  Qwen3.6). hf2q's ceilings 2048 (fresh user turn) / 512 (tool continuation) are sent as thinking_token_budget, and
  the server's --thinking-end-text is hf2q's forced close "\\nI need to answer now.</think>" (decision D4;
  src/serve/api/qwen_thinking_policy.rs:238), which Splash validates against the tokenizer at start.
- parallel_tool_calls false (hf2q's default; decision D5) turns on Splash's tool grammar (docs/api.md Tool calls).
- max_completion_tokens 8192; Splash lowers a limit past the context instead of refusing, so hf2q_overflow_policy
  has no field: the client refuses a turn whose prompt plus 8192 exceeds 262144.
Reply (server/api_shapes.py:1147-1223, server/server.py:1455-1515, server/output.py:265-305):
- unary {id chatcmpl-<32 hex>, object, created, model, choices[1], usage, metrics, timings}; message content is
  null when empty, reasoning_content present only when nonempty; a call is {id call_<32 hex>_<n>, type, function}.
- SSE: a {role: assistant, content: ""} frame, text/reasoning deltas, tool-call deltas (first {index, id, type,
  function{name}}, then {index, function{arguments}}), keepalive empty deltas, a finish frame (empty delta, final
  finish_reason, top-level timings), a choices:[] usage frame with metrics, then [DONE]. Every choice carries
  finish_reason (null until the finish frame).
- usage {prompt_tokens, completion_tokens, total_tokens, prompt_tokens_details{cached_tokens},
  completion_tokens_details{reasoning_tokens}} (server/metrics.py:235-242): cached_tokens is the exact-prefix match.
Identity, launch, readiness and the /apply-template + /tokenize render proof: native_peer_splash_ready.py.
"""
import copy
import re

import qwen_n4_peer
from family_peer import SHARED_KEYS, same
from qwen38_n1_verdict import Invalid, integer, loads, require

ENGINE = 'splash'
FAMILIES = ('qwen',)
Q3_FAMILIES = ('qwen38', 'qwen36')
TITLE = 'Qwen3.8 27B or Qwen3.6 35B-A3B'
SLOTS = 4                                # SPLASH_MAXIMUM_BATCH_WIDTH (runtime/metal/abi/ExecutionGeometry.h:16)
CONTEXT_PER_SLOT = 262144                # --max-context 256K, the native window (no YaRN)
MAX_CONTEXT_FLAG = '256K'
MAX_TOKENS = 8192
SPLASH_COMMIT = '031a25a6125af2d7acf7096580afe56eb33d65bc'
SPLASH_DESCRIBE = '1.5.0-29-g031a25a6'
# Deterministic digest of the production sources and flags (dev/tools/build_identity.py), equal on every clean build
# of SPLASH_COMMIT; /status identity.cache.build_id reports it.
BUILD_ID = 'src-f7663ee765e357e6ecff82e45483f19866d459f0204afaf94f71737b0584f919'
REQUIREMENTS_SHA256 = '3f6fd07c57837a9a0664f65cf61a0e851f40ccec20697008696e54a1c469067b'
BUDGET_USER, BUDGET_TOOL = 2048, 512
THINKING_END_TEXT = '\nI need to answer now.</think>'
# Per Q3 family: the served API id (OWNER/REPO form, --model), Splash's family name (model.json), the reviewed hf2q
# template the GGUF embeds, and the pinned DFlash2 draft (HF cache snapshot, its config.json sha256 and the LFS
# sha256 of model.safetensors). The Qwen3.8 draft is z-lab/...@50307d4c, byte-identical to
# incoai/Qwen3.8-27B-DFlash2@015e7956 (same config bytes and LFS sha256).
ARTIFACTS = {
    'qwen38': {'family': 'qwen', 'title': 'Qwen3.8-27B', 'splash_family': 'Qwen3.8-27B',
               'model_id': 'hf2q/Qwen3.8-27B', 'dense': True,
               'template_sha256': 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041',
               'draft_repo': 'z-lab/Qwen3.8-27B-DFlash2', 'draft_revision': '50307d4c4cde6860d4eee73e2547cd786fe8e8a4',
               'draft_config_sha256': '873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980',
               'draft_weights_sha256': '67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c'},
    'qwen36': {'family': 'qwen', 'title': 'Qwen3.6-35B-A3B', 'splash_family': 'Qwen3.6-35B-A3B',
               'model_id': 'hf2q/Qwen3.6-35B-A3B', 'dense': False,
               'template_sha256': 'e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259',
               'draft_repo': 'incoai/Qwen3.6-35B-A3B-DFlash2', 'draft_revision': '51ef7b6923ad6c14cb1bb41c37a9041446496ab6',
               'draft_config_sha256': '86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59',
               'draft_weights_sha256': 'd1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d'},
}
MODEL_IDS = {q3: a['model_id'] for q3, a in ARTIFACTS.items()}
LABELS = {q3: f"Splash {SPLASH_COMMIT[:7]} (same hf2q {a['title']} GGUF, DFlash2)" for q3, a in ARTIFACTS.items()}

# The server knobs a configuration may set (docs/server.md "Server options"); one value per knob, written as
# `knob=value` pairs in KNOB order (SPLASH_CONFIG). Speculation has no off switch: every model decodes with its draft.
KNOBS = {'kv': ('int8', 'bf16'), 'state': ('fp32', 'bf16'), 'ane': ('auto', 'off'), 'verify': ('auto', 'all'),
         'decode_share': ('0.25', '0.5', '1.0'), 'cache_disk': ('auto', '0'), 'max_memory': None}
DEFAULT_KNOBS = {'kv': 'int8', 'state': 'fp32', 'ane': 'auto', 'verify': 'auto', 'decode_share': '0.5',
                 'cache_disk': 'auto', 'max_memory': 'auto'}
MAX_MEMORY = re.compile(r'auto|[1-9][0-9]{0,3}G')
# One-factor sweep points and the matched-settings diagnostic (design section 7.2). A point that does not apply to a
# family (the Neural Engine split on the MoE target, verify rows on the dense target) is refused for it.
PRESETS = {'s0': {}, 'kv-bf16': {'kv': 'bf16'}, 'state-bf16': {'state': 'bf16'}, 'ane-off': {'ane': 'off'},
           'verify-all': {'verify': 'all'}, 'decode-share-0.25': {'decode_share': '0.25'},
           'decode-share-1.0': {'decode_share': '1.0'}, 'cache-disk-off': {'cache_disk': '0'},
           'matched': {'kv': 'bf16', 'ane': 'off', 'verify': 'all', 'cache_disk': '0'}}
SAMPLING = {'temperature': 0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'presence_penalty': 0.0,
            'frequency_penalty': 0.0}
NATIVE = {**SAMPLING, 'chat_template_kwargs': {'enable_thinking': True}, 'preserve_thinking': True,
          'parallel_tool_calls': False}
REPETITION_PARITY = ('Differs (operator decision D3): no repetition penalty is sent to Splash (1.0); hf2q applies '
                     '1.05 over a 64-token window seeded from the prompt tail (engine_qwen35.rs:4508-4519)')
NATIVE_KEYS = frozenset({'splash_source_build_venv_bound', 'splash_same_artifact_bound', 'splash_draft_bound',
                         'splash_controls_bound', 'splash_render_audit_bound',
                         'splash_slots_context_readiness_verified'})
FINISH = ('stop', 'tool_calls', 'length')
RESPONSE_ID = re.compile(r'chatcmpl-[0-9a-f]{32}')
CALL_ID = re.compile(r'call_[0-9a-f]{32}_[0-9]+')
TOP = frozenset({'id', 'object', 'created', 'model', 'choices', 'usage', 'metrics', 'timings'})
CHUNK = frozenset({'id', 'object', 'created', 'model', 'choices'})
USAGE = frozenset({'prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_tokens_details',
                   'completion_tokens_details'})
METRICS = frozenset({'prefill', 'decode', 'request_latency', 'cache', 'speculation'})
TIMINGS = frozenset({'prompt_n', 'prompt_ms', 'prompt_per_second', 'predicted_n', 'predicted_ms',
                     'predicted_per_second', 'cache_n', 'draft_n', 'draft_n_accepted'})
# native_peers.py interface (the registry resolves kind 'splash' to this module). Splash is a same-artifact peer of
# record: never a native comparison, never the stand-in for an unavailable same-GGUF peer, never an F2 native baseline.
SAME_GGUF_UNAVAILABLE = None
NATIVE_COMPARISON = None
PEER_CASE_REFUSAL = 'a Splash case is the Q3 arm --splash-case, never --peer-case (ADR-063 Splash peer of record)'
CLIENT_MODULES = ('native_peers.py', 'native_peer_splash.py', 'native_peer_splash_ready.py',
                  'test_native_peer_splash.py')
PROCESSES = ('splash',)                  # build/splash serve-native (the engine child of the Python server)
PYTHON_MODULES = ('server.server',)     # the Python server under the pinned .venv (server/server.py), engine or not
BASELINES = {}
SAME_FILE_BASELINES = ()
AUDIT = 'native_peer_splash_audit'


def mode_problems(case, evidence, comparison):
    """Splash is never a --peer-case (PEER_CASE_REFUSAL), so no campaign mode depends on its case."""
    return []


def _family(family):
    require(family in FAMILIES, 'the Splash arm serves Qwen3.8 and Qwen3.6 only')


def q3_family(case):
    require(isinstance(case, dict) and case.get('family') in ARTIFACTS, 'Splash case family is not qwen38/qwen36')
    return case['family']


def label(case):
    return LABELS[q3_family(case)]


def q3_role(case):
    return f'{label(case)}: same-artifact peer of record, best valid configuration (ADR-063 Splash peer of record)'


def parse_config(text, q3):
    """SPLASH_CONFIG `knob=value,...` (every knob, KNOB order) -> knob dict; refuses a value that does not apply."""
    require(q3 in ARTIFACTS, 'Splash config family')
    require(isinstance(text, str) and bool(text), 'SPLASH_CONFIG must be knob=value pairs')
    pairs = [part.split('=', 1) for part in text.split(',')]
    require(all(len(p) == 2 for p in pairs) and [p[0] for p in pairs] == list(KNOBS),
            'SPLASH_CONFIG must name every knob once, in order: ' + ','.join(KNOBS))
    knobs = dict(pairs)
    for name, allowed in KNOBS.items():
        require(knobs[name] in allowed if allowed else bool(MAX_MEMORY.fullmatch(knobs[name])),
                f'SPLASH_CONFIG {name}={knobs[name]} is not an allowed value')
    dense = ARTIFACTS[q3]['dense']
    require(not dense or knobs['verify'] == 'auto', 'verify rows are every row on the dense Qwen3.8 (no knob)')
    require(dense or knobs['ane'] == 'auto', 'the Neural Engine split applies to the dense Qwen3.8 only')
    return knobs


def config_text(knobs):
    return ','.join(f'{name}={knobs[name]}' for name in KNOBS)


def preset(name, q3):
    """A named sweep point as SPLASH_CONFIG text; the matched diagnostic keeps only the knobs the family has."""
    require(name in PRESETS, 'unknown Splash sweep point ' + repr(name))
    knobs = {**DEFAULT_KNOBS, **PRESETS[name]}
    if name == 'matched':
        knobs['verify' if ARTIFACTS[q3]['dense'] else 'ane'] = 'auto'
    return config_text(parse_config(config_text(knobs), q3))


def server_flags(knobs):
    """The `splash serve` options of a configuration, beyond the fixed ones (native_peer_splash_ready.server_argv)."""
    flags = ['--decode-share', knobs['decode_share']]
    if knobs['kv'] != 'int8':
        flags += ['--kv-format', knobs['kv']]
    if knobs['state'] != 'fp32':
        flags += ['--state-format', knobs['state']]
    if knobs['ane'] == 'off':
        flags += ['--disable-ane']
    if knobs['verify'] != 'auto':
        flags += ['--verify-rows', knobs['verify']]
    if knobs['cache_disk'] != 'auto':
        flags += ['--max-cache-disk', knobs['cache_disk']]
    if knobs['max_memory'] != 'auto':
        flags += ['--max-memory', knobs['max_memory']]
    return flags


def controls(q3, config):
    """The approval's family_controls for one family and configuration."""
    knobs = parse_config(config, q3)
    return {'engine': ENGINE, 'source_commit': SPLASH_COMMIT, 'build_id': BUILD_ID, 'q3_family': q3,
            'cache_family': ARTIFACTS[q3]['family'], 'model_id': MODEL_IDS[q3], 'slots': SLOTS,
            'context_per_slot': CONTEXT_PER_SLOT, 'max_context': MAX_CONTEXT_FLAG, 'default_max_tokens': MAX_TOKENS,
            'same_artifact': True, 'draft': f"{ARTIFACTS[q3]['draft_repo']}@{ARTIFACTS[q3]['draft_revision']}",
            'speculation': 'dflash2 (always on)', 'config': config_text(knobs), 'knobs': knobs,
            'idle_release': 'off', 'persistent_cache': False, 'vision': False, 'webui': False,
            'thinking_end_text': THINKING_END_TEXT, **{'request_' + k: v for k, v in SAMPLING.items()},
            'request_repetition_penalty': None, 'request_enable_thinking': True, 'request_preserve_thinking': True,
            'request_parallel_tool_calls': False, 'request_thinking_budget_user': BUDGET_USER,
            'request_thinking_budget_tool': BUDGET_TOOL}


def controls_family(value):
    """The Q3 family of approved family_controls, which must equal controls(q3, their config) exactly."""
    require(isinstance(value, dict) and value.get('q3_family') in ARTIFACTS and isinstance(value.get('config'), str),
            'Splash launch controls lack a family and configuration')
    q3 = value['q3_family']
    require(same(value, controls(q3, value['config'])), 'Splash launch controls differ from the declared settings')
    return q3


def preflight(keys, family):
    """HF family/runtime keys are replaced by the Splash bindings, never forged."""
    _family(family)
    from coding_family import preflight as family_preflight
    return (set(keys) - family_preflight(set(), family) - qwen_n4_peer.HF_ONLY) | NATIVE_KEYS


def validate_controls(value, family):
    _family(family)
    require(ARTIFACTS[controls_family(value)]['family'] == family, 'Splash controls are for another family')


def thinking_budget(messages):
    """hf2q's Qwen default ceiling: 512 when the latest non-system message is a tool result, else 2048."""
    require(isinstance(messages, list) and bool(messages), 'Splash message chain required')
    latest = next((m for m in reversed(messages) if not (isinstance(m, dict) and m.get('role') == 'system')), None)
    return BUDGET_TOOL if isinstance(latest, dict) and latest.get('role') == 'tool' else BUDGET_USER


def _model(model):
    require(model in MODEL_IDS.values(), 'Splash model id must be a pinned served id (' +
            ', '.join(sorted(MODEL_IDS.values())) + ')')


def _wire_tools(tools):
    """Omit the semantic `strict:false` default Splash renders but hf2q does not."""
    require(isinstance(tools, list) and tools, 'canonical Splash tools required')
    result = copy.deepcopy(tools)
    for tool in result:
        function = tool.get('function') if isinstance(tool, dict) else None
        require(isinstance(function, dict) and function.get('strict') is False,
                'canonical Splash function strict:false required')
        function.pop('strict')
    return result


def _canonical_tools(tools):
    require(isinstance(tools, list) and tools, 'Splash wire tools required')
    result = copy.deepcopy(tools)
    for tool in result:
        function = tool.get('function') if isinstance(tool, dict) else None
        require(isinstance(function, dict) and 'strict' not in function,
                'Splash wire function must omit strict:false')
        function['strict'] = False
    return result


def request_controls(request, family):
    """Canonical hf2q request -> exact Splash request; any other field refuses."""
    _family(family)
    require(isinstance(request, dict), 'canonical Splash request')
    extra = sorted(set(request) - SHARED_KEYS)
    require(not extra, 'unsupported Splash override: ' + ','.join(extra))
    require(set(request) == SHARED_KEYS, 'incomplete canonical Splash request')
    _model(request['model'])
    require(type(request['max_completion_tokens']) is int and request['max_completion_tokens'] == MAX_TOKENS,
            'canonical output allowance required')
    require(request['tool_choice'] == 'auto' and request['logprobs'] is False
            and type(request['stream']) is bool, 'canonical tool/transport controls required')
    budget = qwen_n4_peer.request_controls(request)['reasoning_budget_tokens']  # also checks roles/overflow
    require(budget == thinking_budget(request['messages']), 'Qwen thinking ceiling rule drift')
    native = {k: copy.deepcopy(v) for k, v in request.items() if k != 'hf2q_overflow_policy'}
    native['tools'] = _wire_tools(native['tools'])
    return {**native, **copy.deepcopy(NATIVE), 'thinking_token_budget': budget}


def canonical(request, family):
    """Inverse of request_controls at the wire boundary: native fields must be exact, nothing else added."""
    _family(family)
    require(isinstance(request, dict) and 'hf2q_overflow_policy' not in request,
            'Splash request must use native transport fields')
    for key, value in NATIVE.items():
        require(key in request and same(request[key], value), 'Splash native control drift: ' + key)
    require(type(request.get('thinking_token_budget')) is int
            and request['thinking_token_budget'] == thinking_budget(request.get('messages')),
            'Splash thinking budget drift')
    stripped = {k: copy.deepcopy(v) for k, v in request.items()
                if k not in NATIVE and k != 'thinking_token_budget'}
    stream = stripped.get('stream') is True
    require(set(stripped) - {'stream_options'} == SHARED_KEYS - {'hf2q_overflow_policy'}
            and ('stream_options' in stripped) == stream
            and (not stream or same(stripped['stream_options'], {'include_usage': True})),
            'Splash request carries a field outside the closed envelope')
    _model(stripped['model'])
    stripped['tools'] = _canonical_tools(stripped['tools'])
    return {**stripped, 'hf2q_overflow_policy': 'reject'}


def _arguments(text):
    try:
        value = loads(text)
    except (ValueError, TypeError) as exc:
        raise Invalid('Splash tool-call arguments are not JSON') from exc
    require(isinstance(value, dict), 'Splash tool-call arguments are not a JSON object')


def _identity(obj, kind):
    require(isinstance(obj, dict) and obj.get('object') == kind and type(obj.get('created')) is int
            and isinstance(obj.get('id'), str) and RESPONSE_ID.fullmatch(obj['id']) is not None
            and isinstance(obj.get('model'), str), 'Splash response identity (chatcmpl-<32 hex>)')


def _metrics(value):
    require(isinstance(value, dict) and {'prefill', 'decode', 'request_latency', 'cache'} <= set(value) <= METRICS
            and isinstance(value['cache'], dict) and isinstance(value['request_latency'], dict),
            'Splash metrics envelope')


def _timings(value):
    require(isinstance(value, dict) and TIMINGS - {'draft_n', 'draft_n_accepted'} <= set(value) <= TIMINGS
            and all(type(v) in (int, float) and v >= 0 for v in value.values()), 'Splash timings envelope')


def check_unary(obj):
    """Closed Splash chat.completion envelope (server/api_shapes.py:1147-1163)."""
    _identity(obj, 'chat.completion')
    require(set(obj) == TOP and isinstance(obj['usage'], dict), 'Splash unary envelope')
    _metrics(obj['metrics'])
    _timings(obj['timings'])
    choices = obj['choices']
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and set(choices[0]) == {'index', 'message', 'finish_reason'} and choices[0]['index'] == 0,
            'Splash unary choice (logprobs are never requested)')
    message, finish = choices[0]['message'], choices[0]['finish_reason']
    require(isinstance(message, dict) and message.get('role') == 'assistant' and 'content' in message
            and set(message) <= {'role', 'content', 'reasoning_content', 'tool_calls'}, 'Splash unary message fields')
    require(message['content'] is None or (isinstance(message['content'], str) and bool(message['content'])),
            'Splash content (null when empty)')
    require('reasoning_content' not in message or (isinstance(message['reasoning_content'], str)
                                                   and bool(message['reasoning_content'])), 'Splash reasoning_content')
    require(finish in FINISH, f'Splash finish_reason {finish!r}')
    require(finish != 'tool_calls' or 'tool_calls' in message, 'Splash finish tool_calls without calls')
    require('tool_calls' not in message or finish in ('tool_calls', 'length'),
            'Splash calls are tool_calls unless the output limit cut them')
    for call in message.get('tool_calls') or []:
        function = call.get('function') if isinstance(call, dict) else None
        require(isinstance(function, dict) and set(call) == {'id', 'type', 'function'}
                and set(function) == {'name', 'arguments'} and call['type'] == 'function'
                and isinstance(call['id'], str) and CALL_ID.fullmatch(call['id']) is not None
                and all(isinstance(v, str) for v in function.values()) and bool(function['name']),
                'Splash tool call (id call_<32 hex>_<n>, function name and arguments)')
        if finish == 'tool_calls':
            _arguments(function['arguments'])


def check_frame(obj):
    """Closed Splash chat.completion.chunk envelope (server/server.py:1455-1515, server/api_shapes.py:1186-1223)."""
    _identity(obj, 'chat.completion.chunk')
    if obj.get('choices') == []:
        require(set(obj) == CHUNK | {'usage', 'metrics'} and isinstance(obj['usage'], dict), 'Splash SSE usage frame')
        _metrics(obj['metrics'])
        return
    final = isinstance(obj.get('choices'), list) and len(obj['choices']) == 1 and isinstance(obj['choices'][0], dict) \
        and obj['choices'][0].get('finish_reason') is not None
    require(set(obj) == (CHUNK | {'timings'} if final else CHUNK), 'Splash SSE chunk fields (timings on the finish frame)')
    choices = obj['choices']
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and set(choices[0]) == {'index', 'delta', 'finish_reason'} and choices[0]['index'] == 0, 'Splash SSE choice')
    delta = choices[0]['delta']
    require(isinstance(delta, dict) and set(delta) <= {'role', 'content', 'reasoning_content', 'tool_calls'},
            'unknown Splash delta field')
    require('role' not in delta or delta == {'role': 'assistant', 'content': ''}, 'Splash role frame')
    require(all(isinstance(delta[k], str) for k in ('content', 'reasoning_content') if k in delta),
            'Splash text deltas must be strings')
    if final:
        require(choices[0]['finish_reason'] in FINISH and delta == {},
                f"Splash final chunk (finish_reason {choices[0]['finish_reason']!r})")
        _timings(obj['timings'])
    if 'tool_calls' in delta:
        parts = delta['tool_calls']
        require(isinstance(parts, list) and len(parts) == 1 and isinstance(parts[0], dict), 'Splash tool delta')
        part = parts[0]
        function = part.get('function')
        require(type(part.get('index')) is int and isinstance(function, dict)
                and all(isinstance(v, str) for v in function.values()), 'Splash tool delta fields')
        if 'id' in part:
            require(set(part) == {'index', 'id', 'type', 'function'} and set(function) == {'name'}
                    and part['type'] == 'function' and isinstance(part['id'], str)
                    and CALL_ID.fullmatch(part['id']) is not None and bool(function['name']),
                    'Splash tool-call opening (index, id call_<32 hex>_<n>, type, name)')
        else:
            require(set(part) == {'index', 'function'} and set(function) == {'arguments'},
                    'Splash tool-call argument delta')


def usage(value, turn):
    """Splash usage (server/metrics.py:235-242); cached tokens are the exact-prefix match."""
    require(isinstance(value, dict) and set(value) == USAGE, 'Splash usage envelope')
    prompt = integer(value['prompt_tokens'], 'Splash prompt tokens', CONTEXT_PER_SLOT)
    completion = integer(value['completion_tokens'], 'Splash completion tokens', MAX_TOKENS)
    require(integer(value['total_tokens'], 'Splash total tokens') == prompt + completion,
            'Splash usage total contradiction')
    require(prompt + MAX_TOKENS <= CONTEXT_PER_SLOT, 'Splash prompt plus full output exceeds the 262144 workload context')
    details, thought = value['prompt_tokens_details'], value['completion_tokens_details']
    require(isinstance(details, dict) and set(details) == {'cached_tokens'}, 'Splash cache details envelope')
    require(isinstance(thought, dict) and set(thought) == {'reasoning_tokens'}, 'Splash reasoning details envelope')
    cached = integer(details['cached_tokens'], 'Splash cached tokens', prompt)
    reasoning = integer(thought['reasoning_tokens'], 'Splash reasoning tokens', completion)
    return {**value, 'engine_kind': ENGINE, 'cache_observation': {
        'status': 'Observed', 'cached_tokens': cached, 'source_field': 'usage.prompt_tokens_details.cached_tokens',
        'semantics': 'exact-prefix KV and state reuse (docs/api.md Output limits and streaming)',
        'full_prefix_reuse': 'PendingPostOracle'},
        'reasoning_observation': {'status': 'Observed', 'reasoning_tokens': reasoning,
                                  'source_field': 'usage.completion_tokens_details.reasoning_tokens',
                                  'semantics': 'engine count of output tokens inside the thinking block'}}


def readiness(base, model, output, packet):
    from native_peer_splash_ready import readiness as ready
    return ready(base, model, output, packet)


def closing(base, output):
    """Every agent's /apply-template + /tokenize proofs and the final server state, after all agents finished."""
    from native_peer_splash_ready import closing as close
    return close(base, output)


def campaign_identity(plan):
    from native_peer_splash_ready import campaign_identity as identity
    return identity(plan)


def launch_problems(name, artifact, hf2q_sha256, identity=None, output=None):
    from native_peer_splash_ready import launch_problems as problems
    return problems(name, artifact, hf2q_sha256, identity, output)
