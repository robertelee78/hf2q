"""oMLX (MLX/mlx-lm continuous-batching server) benchmark arm: engine_kind 'omlx'.

Harness-only: no server launch, model load or inference fallback. oMLX serves mlx-community 4-bit MLX
checkpoints, not the hf2q GGUFs, so every case is a separately labeled different-artifact comparison,
never the same-GGUF peer. Three cases share the arm (q5_cases/native-omlx-{qwen38,qwen36,gemma4}.json);
the Q3 family selects the pinned snapshot, model id and per-family controls (ARTIFACTS, CONTROLS).

Source inspected: /opt/omlx at 6b413c90 (clean tree; __version__ 0.7.3rc1, omlx/_version.py:1). Paths are
under omlx/ unless marked.
Request (api/openai_models.py:333-428, server.py:4994-5240):
- ChatCompletionRequest has no `extra` config, so pydantic silently drops unknown fields (parallel_tool_calls,
  reasoning_budget_tokens, cache_prompt, ...): this adapter's closed envelope is the only guard.
- temperature 0 is plain argmax (utils/sampling.py:214-220). Request values win over model settings and the
  global defaults (temperature 1.0, top_p 0.95; settings.py:806-808) unless force_sampling is set
  (server.py:1882-2047); the model settings pin force_sampling false and every request sends top_p 1, top_k 0,
  min_p 0, presence/frequency 0 explicitly.
- repetition_penalty 1.05 (hf2q's Qwen and Gemma value) with a per-family repetition_context_size; mlx-lm's
  default window is 20 (request.py:64-67; server.py:5198-5203).
  Qwen 64: hf2q seeds a 64-token sliding window with the prompt tail and slides it over generated tokens
  (src/serve/api/engine_qwen35.rs:4508-4519).
  Gemma 8192 (= MAX_TOKENS): hf2q penalizes every distinct token generated in the request, with no window and an
  empty history at the first token (src/serve/api/engine.rs:7785-7788, src/serve/sampler_pure.rs:557-574).
  Whether oMLX's mlx-lm processor history includes prompt tokens is not established from source: for Qwen the
  window may hold different prompt tokens than hf2q's tail; for Gemma a prompt-seeded history would penalize
  prompt tokens hf2q never penalizes. Recorded as a parity risk (Unavailable), never assumed.
- chat_template_kwargs reach the template (server.py:4817-4842; model_settings.py:1919-1949). Qwen: hf2q
  renders thinking on with preserve_thinking true (handlers.rs:1733-1737) and no explicit effort, so the
  request sends {enable_thinking: true, preserve_thinking: true} and no reasoning_effort (the template's own
  default applies, as in hf2q). Gemma 4: hf2q renders no thinking channel, so {enable_thinking: false}.
- thinking_budget carries hf2q's Qwen ceilings, 2048 on a fresh user turn and 512 on a tool-result continuation
  (qwen_n4_peer.request_controls). oMLX enforces it with a logits processor that forces the close-think
  sequence (api/thinking.py:481-520: leading + </think> + trailing); hf2q forces "\\nI need to answer
  now.</think>". Gemma sends none.
- max_completion_tokens is accepted as max_tokens (openai_models.py:345-349). oMLX's context check counts prompt
  tokens only (server.py:2436-2452), so hf2q_overflow_policy=reject has no server field; the client refuses a
  turn whose usage leaves no room for 8192 output tokens.
- parallel_tool_calls is NOT sent (oMLX would drop it). Every parsed call is returned (server.py:5433,
  7114-7142) and the client's exactly-one-call rule refuses a second one. tool_choice "auto" only differs from
  "none" (server.py:4938). `strict` is dropped by convert_tools_for_template (api/tool_calling.py:3741-3801):
  no tool grammar runs; hf2q charges a lazy grammar.
Prompt mutations (recorded, proven per turn by /v1/tokenize at closing): tool schemas are re-serialized with
sorted keys and `"description": ""` injected into every property schema (api/tool_calling.py:48-97); a model id
containing "gemma" also renames `description` params and adds labels (server.py:4961-4962, tool_calling.py:
3812-3853), so the Gemma model id avoids that substring; tool messages keep only tool_call_id and content
(api/utils.py:1066-1088).
Reply (server.py:5330-5490 unary, 6573-7290 SSE; every frame model_dump_json(exclude_none=True)):
- unary chat.completion {id chatcmpl-<8 hex>, object, created, model, choices, usage}; message content is
  stripped (5445) and omitted when empty, reasoning_content omitted when empty; finish_reason is tool_calls
  whenever calls parsed (5433). Bodies past 2 s are led by keepalive spaces under a committed 200; a late
  failure is a JSON error body (server.py:3203-3345), which the client refuses.
- SSE: {role: assistant} first (6632-6642), content/reasoning_content deltas, one complete call per chunk
  {index, id call_<8 hex>, type, function{name, arguments}} (7114-7142), an empty-delta finish chunk
  (7149-7166), a choices:[] usage chunk when include_usage (7238-7283), then [DONE]. Errors are a data:{error}
  frame then [DONE] with no finish chunk (6867-6870, 7144-7147). finish_reason is absent (not null) until the
  final chunk.
- usage adds input/output_tokens (api/shared_models.py:55-79) and timing extensions (openai_models.py:482-503);
  cached_tokens is block-granular prefix reuse; reasoning_tokens re-tokenizes the cleaned reasoning text
  (5434-5438, 7243-7247), never an engine count.
Identity, settings files, readiness and the /v1/tokenize proof: native_peer_omlx_ready.py.
"""
import copy
import re

import family_peer
import qwen_n4_peer
from family_peer import SHARED_KEYS, same
from qwen38_n1_verdict import Invalid, integer, loads, require

ENGINE = 'omlx'
FAMILIES = ('qwen', 'gemma4')
Q3_FAMILIES = ('qwen38', 'qwen36', 'gemma4')
TITLE = 'Qwen3.8 27B, Qwen3.6 35B-A3B or Gemma 4 26B-A4B'
SLOTS = 4
CONTEXT_PER_SLOT = 262144
MAX_TOKENS = 8192
OMLX_COMMIT = '6b413c900352db1382466cf4b7f0825f61a550a9'
OMLX_VERSION = '0.7.3rc1'                # omlx/_version.py:1; cli.py --version and /api/status report it
BUDGET_USER, BUDGET_TOOL = 2048, 512
# Pinned mlx-community snapshots (recipe adr063-omlx-recipe-v1 section 2). model_id is the served directory name
# (cli.py serve help; model_discovery.py:1972-2096); hf2q_template_sha256 is the reviewed hf2q template
# (rendered-inputs-reviewed/<family>.template.jinja) the readiness compares the served one against.
ARTIFACTS = {
    'qwen38': {'family': 'qwen', 'title': 'Qwen3.8-27B', 'repo': 'mlx-community/Qwen3.8-27B-4bit',
               'revision': '10c35caafbb80f7dc6a7a432cdd11af10a6d4818', 'model_id': 'Qwen3.8-27B-4bit',
               'config_model_type': 'qwen3_5',
               'hf2q_template_sha256': 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'},
    'qwen36': {'family': 'qwen', 'title': 'Qwen3.6-35B-A3B', 'repo': 'mlx-community/Qwen3.6-35B-A3B-4bit',
               'revision': '38740b847e4cb78f352aba30aa41c76e08e6eb46', 'model_id': 'Qwen3.6-35B-A3B-4bit',
               'config_model_type': 'qwen3_5_moe',
               'hf2q_template_sha256': 'e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259'},
    'gemma4': {'family': 'gemma4', 'title': 'Gemma 4 26B-A4B', 'repo': 'mlx-community/gemma-4-26b-a4b-it-4bit',
               'revision': '0d77464eeb233a2da68ebf9d7dc4edaac7db956d', 'model_id': 'g4-26b-a4b-it-4bit',
               'config_model_type': 'gemma4',
               'hf2q_template_sha256': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4'},
}
MODEL_IDS = {q3: a['model_id'] for q3, a in ARTIFACTS.items()}
LABELS = {q3: f"oMLX {OMLX_COMMIT[:7]} (native {a['title']} MLX 4-bit)" for q3, a in ARTIFACTS.items()}
SAMPLING_FIELDS = {'temperature': 0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'presence_penalty': 0.0,
                   'frequency_penalty': 0.0, 'repetition_penalty': 1.05}
# hf2q's repetition history: Qwen a 64-token sliding window seeded from the prompt tail (engine_qwen35.rs:4508-4519);
# Gemma every generated token of the request, no window (engine.rs:7785-7788, sampler_pure.rs:557-574).
REPETITION_WINDOW = {'qwen': 64, 'gemma4': MAX_TOKENS}
SAMPLING = {family: {**SAMPLING_FIELDS, 'repetition_context_size': window}
            for family, window in REPETITION_WINDOW.items()}
REPETITION_PARITY = ('Unavailable:whether oMLX/mlx-lm seeds the repetition history with prompt tokens is not '
                     'established from source (hf2q: Qwen seeds 64 prompt-tail tokens, Gemma seeds none)')
NATIVE = {'qwen': {**SAMPLING['qwen'], 'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True}},
          'gemma4': {**SAMPLING['gemma4'], 'chat_template_kwargs': {'enable_thinking': False}}}
_CONTROLS = {'engine': ENGINE, 'source_commit': OMLX_COMMIT, 'version': OMLX_VERSION, 'slots': SLOTS,
             'context_per_slot': CONTEXT_PER_SLOT, 'max_concurrent_requests': SLOTS,
             'max_context_window': CONTEXT_PER_SLOT, 'default_max_tokens': MAX_TOKENS, 'model_type_override': 'llm',
             'prefix_cache': 'hot-only', 'hot_cache_max_size': '16GB', 'ssd_cache_tier': False,
             'memory_guard': 'aggressive', 'sse_keepalive': 'off', 'hf_cache_discovery': False,
             'speculation': 'none', 'kv_compression': 'none', 'force_sampling': False,
             **{'request_' + k: v for k, v in SAMPLING_FIELDS.items()}, 'request_parallel_tool_calls': 'omitted'}
CONTROLS = {q3: {**_CONTROLS, 'q3_family': q3, 'cache_family': a['family'], 'model_id': a['model_id'],
                 'target': f"{a['repo']}@{a['revision']}",
                 'request_repetition_context_size': REPETITION_WINDOW[a['family']],
                 **({'request_enable_thinking': True, 'request_preserve_thinking': True,
                     'request_thinking_budget_user': BUDGET_USER, 'request_thinking_budget_tool': BUDGET_TOOL}
                    if a['family'] == 'qwen' else {'request_enable_thinking': False, 'request_thinking_budget': None})}
            for q3, a in ARTIFACTS.items()}
NATIVE_KEYS = frozenset({'omlx_source_venv_kernels_bound', 'omlx_model_snapshot_bound', 'omlx_settings_files_bound',
                         'omlx_controls_bound', 'omlx_tokenize_render_audit_bound',
                         'omlx_slots_context_readiness_verified'})
FINISH = ('stop', 'tool_calls', 'length')
RESPONSE_ID = re.compile(r'chatcmpl-[0-9a-f]{8}')
CALL_ID = re.compile(r'call_[0-9a-f]{8}')
TOP = frozenset({'id', 'object', 'created', 'model', 'choices', 'usage'})
CHUNK = frozenset({'id', 'object', 'created', 'model', 'choices'})
USAGE = frozenset({'prompt_tokens', 'completion_tokens', 'total_tokens', 'input_tokens', 'output_tokens',
                   'prompt_tokens_details', 'completion_tokens_details'})
TIMING = frozenset({'model_load_duration', 'time_to_first_token', 'time_to_first_visible_token', 'total_time',
                    'prompt_eval_duration', 'generation_duration', 'prompt_tokens_per_second',
                    'generation_tokens_per_second'})
SAME_GGUF_UNAVAILABLE = None             # every oMLX family has a runnable same-GGUF peer
NATIVE_COMPARISON = ('native-comparison campaign; not the verdict of record: the same-GGUF peer verdict comes '
                     'from the campaign of record, and the oMLX arm is reported separately as native_peer '
                     '(ADR-063 Q4)')
# native_peers.py interface (the registry resolves kind 'omlx' to this module).
CLIENT_MODULES = ('native_peers.py', 'native_peer_omlx.py', 'native_peer_omlx_ready.py', 'test_native_peer_omlx.py')
PROCESSES = ('omlx', 'omlx-server')     # pyproject.toml:233-234 console script; process_title.py:8 setproctitle
PYTHON_MODULES = ()                     # the server runs as the omlx console script (PROCESSES)
BASELINES = {f'omlx_{q3}_native': label for q3, label in LABELS.items()}
AUDIT = 'native_peer_omlx_audit'
PEER_CASE_REFUSAL = ('a native oMLX --peer-case needs --native-comparison (Qwen3.8, Qwen3.6 and Gemma 4 have a '
                     'runnable same-GGUF peer, ADR-063 Q4)')


def _family(family):
    require(family in FAMILIES, 'the native oMLX arm serves Qwen3.8, Qwen3.6 and Gemma 4 only')


def q3_family(case):
    require(isinstance(case, dict) and case.get('family') in ARTIFACTS, 'oMLX case family is not qwen38/qwen36/gemma4')
    return case['family']


def label(case):
    return LABELS[q3_family(case)]


def q3_role(case):
    return (f'{label(case)}: separately labeled different-artifact native comparison, task makespan only; '
            'not the same-GGUF peer verdict (ADR-063 Q4)')


def preflight(keys, family):
    """HF family/runtime keys are replaced by the native oMLX bindings, never forged."""
    _family(family)
    from coding_family import preflight as family_preflight
    return (set(keys) - family_preflight(set(), family) - qwen_n4_peer.HF_ONLY) | NATIVE_KEYS


def controls_family(controls):
    """The Q3 family whose declared launch controls the approval carries."""
    found = [q3 for q3, expected in CONTROLS.items() if same(controls, expected)]
    require(len(found) == 1, 'oMLX launch controls differ from the declared native settings')
    return found[0]


def validate_controls(controls, family):
    _family(family)
    require(ARTIFACTS[controls_family(controls)]['family'] == family, 'oMLX controls are for another family')


def thinking_budget(messages):
    """hf2q's Qwen default ceiling: 512 when the latest non-system message is a tool result, else 2048."""
    require(isinstance(messages, list) and bool(messages), 'oMLX message chain required')
    latest = next((m for m in reversed(messages) if not (isinstance(m, dict) and m.get('role') == 'system')), None)
    return BUDGET_TOOL if isinstance(latest, dict) and latest.get('role') == 'tool' else BUDGET_USER


def _model(model, family):
    require(model in {a['model_id'] for a in ARTIFACTS.values() if a['family'] == family},
            'oMLX model id must be the served directory name of a pinned ' + family + ' snapshot')


def request_controls(request, family):
    """Canonical hf2q request -> exact oMLX request; any other field refuses."""
    _family(family)
    require(isinstance(request, dict), 'canonical oMLX request')
    extra = sorted(set(request) - SHARED_KEYS)
    require(not extra, 'unsupported oMLX override: ' + ','.join(extra))
    require(set(request) == SHARED_KEYS, 'incomplete canonical oMLX request')
    _model(request['model'], family)
    require(type(request['max_completion_tokens']) is int and request['max_completion_tokens'] == MAX_TOKENS,
            'canonical output allowance required')
    require(request['tool_choice'] == 'auto' and request['logprobs'] is False
            and type(request['stream']) is bool, 'canonical tool/transport controls required')
    native = {k: v for k, v in request.items() if k != 'hf2q_overflow_policy'}
    if family == 'qwen':
        budget = qwen_n4_peer.request_controls(request)['reasoning_budget_tokens']  # also checks roles/overflow
        require(budget == thinking_budget(request['messages']), 'Qwen thinking ceiling rule drift')
        return {**native, **copy.deepcopy(NATIVE[family]), 'thinking_budget': budget}
    family_peer.request_controls(request, family)  # the Gemma canonical envelope, roles and overflow policy
    return {**native, **copy.deepcopy(NATIVE[family])}


def canonical(request, family):
    """Inverse of request_controls at the wire boundary: native fields must be exact, nothing else added."""
    _family(family)
    require(isinstance(request, dict) and 'hf2q_overflow_policy' not in request,
            'oMLX request must use native transport fields')
    for key, value in NATIVE[family].items():
        require(key in request and same(request[key], value), 'oMLX native control drift: ' + key)
    budgeted = family == 'qwen'
    require(('thinking_budget' in request) == budgeted, 'oMLX thinking budget presence drift')
    require(not budgeted or (type(request['thinking_budget']) is int and request['thinking_budget'] == thinking_budget(
        request.get('messages'))), 'oMLX thinking budget drift')
    stripped = {k: v for k, v in request.items() if k not in NATIVE[family] and k != 'thinking_budget'}
    stream = stripped.get('stream') is True
    require(set(stripped) - {'stream_options'} == SHARED_KEYS - {'hf2q_overflow_policy'}
            and ('stream_options' in stripped) == stream
            and (not stream or same(stripped['stream_options'], {'include_usage': True})),
            'oMLX request carries a field outside the closed envelope')
    _model(stripped['model'], family)
    return {**stripped, 'hf2q_overflow_policy': 'reject'}


def _arguments(text):
    try:
        value = loads(text)
    except (ValueError, TypeError) as exc:
        raise Invalid('oMLX tool-call arguments are not JSON') from exc
    require(isinstance(value, dict), 'oMLX tool-call arguments are not a JSON object (openai_models.py:145)')


def _tool_calls(calls, delta):
    require(isinstance(calls, list) and bool(calls), 'oMLX tool_calls must be a nonempty array')
    for call in calls:
        function = call.get('function') if isinstance(call, dict) else None
        keys = {'index', 'id', 'type', 'function'} if delta else {'id', 'type', 'function'}
        require(isinstance(function, dict) and set(call) == keys and set(function) == {'name', 'arguments'},
                'unknown oMLX tool-call field (each call is complete in one object)')
        require(not delta or type(call['index']) is int, 'oMLX tool-call index')
        require(call['type'] == 'function' and isinstance(call['id'], str) and CALL_ID.fullmatch(call['id']),
                'oMLX tool-call id is not call_<8 hex>')
        require(all(isinstance(v, str) for v in function.values()) and bool(function['name']),
                'oMLX tool-call fields must be strings')
        _arguments(function['arguments'])


def _identity(obj, kind):
    require(isinstance(obj, dict) and obj.get('object') == kind and type(obj.get('created')) is int
            and isinstance(obj.get('id'), str) and RESPONSE_ID.fullmatch(obj['id']) is not None
            and isinstance(obj.get('model'), str), 'oMLX response identity (chatcmpl-<8 hex>)')


def check_unary(obj):
    """Closed oMLX chat.completion envelope (server.py:5440-5484, exclude_none)."""
    _identity(obj, 'chat.completion')
    require(set(obj) == TOP and isinstance(obj['usage'], dict), 'oMLX unary envelope')
    choices = obj['choices']
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and set(choices[0]) == {'index', 'message', 'finish_reason'} and choices[0]['index'] == 0,
            'oMLX unary choice (logprobs are never requested)')
    message, finish = choices[0]['message'], choices[0]['finish_reason']
    require(isinstance(message, dict) and message.get('role') == 'assistant'
            and set(message) <= {'role', 'content', 'reasoning_content', 'tool_calls'}, 'oMLX unary message fields')
    require('content' not in message or isinstance(message['content'], str), 'oMLX content')
    require('reasoning_content' not in message or (isinstance(message['reasoning_content'], str)
                                                   and bool(message['reasoning_content'])), 'oMLX reasoning_content')
    require(finish in FINISH, f'oMLX finish_reason {finish!r} (error/abort/unknown refuse)')
    require(('tool_calls' in message) == (finish == 'tool_calls'), 'oMLX tool_calls iff finish tool_calls')
    if 'tool_calls' in message:
        _tool_calls(message['tool_calls'], False)


def check_frame(obj):
    """Closed oMLX chat.completion.chunk envelope (server.py:6632-6642, 6720-6866, 7114-7166, 7238-7283)."""
    _identity(obj, 'chat.completion.chunk')
    if obj.get('choices') == []:
        require(set(obj) == CHUNK | {'usage'} and isinstance(obj['usage'], dict), 'oMLX SSE usage frame')
        return
    require(set(obj) == CHUNK, 'oMLX SSE chunk fields')
    choices = obj['choices']
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and set(choices[0]) in ({'index', 'delta'}, {'index', 'delta', 'finish_reason'})
            and choices[0]['index'] == 0, 'oMLX SSE choice')
    delta = choices[0]['delta']
    require(isinstance(delta, dict) and set(delta) <= {'role', 'content', 'reasoning_content', 'tool_calls'},
            'unknown oMLX delta field')
    require('role' not in delta or delta == {'role': 'assistant'}, 'oMLX role frame carries only the role')
    require(all(isinstance(delta[k], str) for k in ('content', 'reasoning_content') if k in delta),
            'oMLX text deltas must be strings')
    if 'finish_reason' in choices[0]:
        require(choices[0]['finish_reason'] in FINISH and delta == {},
                f"oMLX final chunk (finish_reason {choices[0]['finish_reason']!r}; error/abort/unknown refuse)")
    if 'tool_calls' in delta:
        _tool_calls(delta['tool_calls'], True)


def usage(value, turn):
    """oMLX usage (api/shared_models.py:55-79, openai_models.py:469-503); cached and reasoning tokens reported."""
    require(isinstance(value, dict) and USAGE <= set(value) <= USAGE | TIMING, 'oMLX usage envelope')
    prompt = integer(value['prompt_tokens'], 'oMLX prompt tokens', CONTEXT_PER_SLOT)
    completion = integer(value['completion_tokens'], 'oMLX completion tokens', MAX_TOKENS)
    require(integer(value['total_tokens'], 'oMLX total tokens') == prompt + completion
            and integer(value['input_tokens'], 'oMLX input tokens') == prompt
            and integer(value['output_tokens'], 'oMLX output tokens') == completion, 'oMLX usage total contradiction')
    require(prompt + MAX_TOKENS <= CONTEXT_PER_SLOT, 'oMLX prompt plus full output exceeds the 262144 workload context')
    require(all(type(value[k]) in (int, float) and value[k] >= 0 for k in set(value) & TIMING), 'oMLX usage timing')
    details, thought = value['prompt_tokens_details'], value['completion_tokens_details']
    require(isinstance(details, dict) and set(details) == {'cached_tokens'}, 'oMLX cache details envelope')
    require(isinstance(thought, dict) and set(thought) == {'reasoning_tokens'}, 'oMLX reasoning details envelope')
    cached = integer(details['cached_tokens'], 'oMLX cached tokens', prompt)
    reasoning = integer(thought['reasoning_tokens'], 'oMLX reasoning tokens')
    return {**value, 'engine_kind': ENGINE, 'cache_observation': {
        'status': 'Observed', 'cached_tokens': cached, 'source_field': 'usage.prompt_tokens_details.cached_tokens',
        'semantics': 'paged prefix-cache reuse minus resumed prefill, block-granular (scheduler.py '
                     'prefill_resumed_tokens)', 'full_prefix_reuse': 'PendingPostOracle'},
        'reasoning_observation': {'status': 'Observed', 'reasoning_tokens': reasoning,
                                  'source_field': 'usage.completion_tokens_details.reasoning_tokens',
                                  'semantics': 're-tokenization of the cleaned reasoning text (server.py:5434-5438, '
                                               '7243-7247), not an engine count'}}


def readiness(base, model, output, packet):
    from native_peer_omlx_ready import readiness as ready
    return ready(base, model, output, packet)


def closing(base, output):
    """Every agent's /v1/tokenize proofs and the final server state, after all agents finished (outside the makespan)."""
    from native_peer_omlx_ready import closing as close
    return close(base, output)


def campaign_identity(plan):
    from native_peer_omlx_ready import campaign_identity as identity
    return identity(plan)


def launch_problems(name, artifact, hf2q_sha256):
    from native_peer_omlx_ready import launch_problems as problems
    return problems(name, artifact, hf2q_sha256)
