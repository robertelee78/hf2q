"""TensorFold (native Zig/Metal engine) Qwen3.8-27B benchmark arm: engine_kind 'tensorfold'.

Harness-only: no server launch, model load or inference fallback. TensorFold serves its own MLX 4-bit
checkpoint (TensorFold/Qwen3.8-27B-MLX-4bit@22d8d538), not the hf2q GGUF, so the arm is a separately
labeled different-artifact comparison, never the same-GGUF peer. Two cases share it:
q5_cases/native-tensorfold-qwen38.json (no drafter) and native-tensorfold-qwen38-dflash2.json (DFlash2,
z-lab/Qwen3.8-27B-DFlash2@50307d4c); their argv differ only in `--drafter DIR --drafter-bits 4`.

Source inspected: /opt/TensorFold at 1157ea5b (clean tree; version 1.2.0, build.zig.zon:3). Paths are
under zig/src/server/ unless marked.
Request (openai.zig:127-187, fields.zig:200-234, chat.zig:124-203):
- the chat route reads known keys only and ignores the rest, so this adapter's envelope is the closed one.
- temperature 0 means no sampler, i.e. argmax (server.zig:252-264). top_p/top_k/min_p are sent at their
  off values.
- reasoning_effort 'xhigh' is what hf2q renders. hf2q's Qwen3.8 template is the same chat_template.jinja
  bytes (git blob c0c686f9), which defaults the effort to xhigh, and hf2q passes it only preserve_thinking
  (src/serve/api/handlers.rs:1733-1737). Sending xhigh pins that system instruction instead of relying on
  TensorFold's default (no --reasoning-effort, server.zig:246-249).
- chat_template_kwargs.enable_thinking true. TensorFold forwards only enable_thinking, thinking,
  reasoning_effort and clear_thinking to the template (fields.zig:200-234, hf_text.zig:367-371), so
  preserve_thinking is not sent: the template's default (undefined: keep earlier reasoning) equals hf2q's
  explicit true.
- thinking_budget carries hf2q's Qwen ceilings: 2048 on a fresh user turn, 512 on a tool-result
  continuation (qwen_n4_peer.request_controls; src/serve/api/qwen_thinking_policy.rs:60-71). TensorFold
  forces "\\n</think>\\n\\n" at the budget (server.zig:213-217, chat.zig:185-196); hf2q forces
  "\\nI need to answer now.</think>" (qwen_thinking_policy.rs:238).
- max_completion_tokens 8192 bounds the reply (openai.zig:143-150). A prompt that leaves no room for it
  refuses with context_length_exceeded (chat.zig:146-153), which is hf2q_overflow_policy=reject.
- parallel_tool_calls is NOT sent. With false, TensorFold keeps the first call and silently drops the
  rest (openai.zig:333-350, tool_stream.zig:449-456). Omitted, every call is returned and the client's
  exactly-one-call rule refuses a second one: no silent drop.
- logprobs false is honoured as sent (fields.zig:288-301). Tool schemas keep strict false, so no tool
  grammar runs (server.zig:302-314); hf2q charges a lazy tool grammar instead.
- TensorFold has no repetition penalty (server.zig:283-294 resolves presence/frequency only, and this
  engine ignores them), so hf2q's Qwen repetition penalty 1.05 has no counterpart: Unavailable.
Reply: chat.completion (openai.zig:408-438) and chunks (openai.zig:275-301, 440-595) carry the extras
exact_mode, tensorfold (runtime facts, chat.zig:515-548), speculative (draft counters, chat.zig:549-560)
and an optional stop_sequence (openai.zig:322-330). Usage has prompt_tokens_details.cached_tokens and
completion_tokens_details.reasoning_tokens (openai.zig:307-320); with include_usage it rides alone in a
choices:[] chunk (openai.zig:580-591). Streams interleave ': keepalive' comments and end with data: [DONE]
(sse.zig:26-32). Streamed calls are a head {index, id: call_<24 hex>, type, function{name, arguments:""}}
then {index, function{arguments}} pieces (tool_stream.zig:73-80, 214-221; ids.zig:7-15). Finish reasons
are stop/length/tool_calls; cancelled (chat.zig:468-474) and content_filter (--loop-guard-stop) refuse.
Identity, readiness and the /tokenize rendered-prompt proof: native_peer_tensorfold_ready.py.
"""
import copy
import json
import re

import qwen_n4_peer
from family_peer import SHARED_KEYS, same
from qwen38_n1_verdict import integer, require

ENGINE = 'tensorfold'
FAMILY = 'qwen'
FAMILIES = (FAMILY,)
TITLE = 'Qwen3.8'
Q3_FAMILIES = ('qwen38',)
SLOTS = 4
CONTEXT_PER_SLOT = 262144
MAX_TOKENS = 8192
TF_COMMIT = '1157ea5b0f636469aa550aab8d8da7b22083dcd9'
TF_VERSION = '1.2.0'                     # build.zig:74-79 embeds -Dversion or build.zig.zon's, never a commit
TARGET_REPO = 'TensorFold/Qwen3.8-27B-MLX-4bit'
TARGET_REVISION = '22d8d538154e0e5b6f8dcb1fcc74a60a0798104c'
DRAFTER_REPO = 'z-lab/Qwen3.8-27B-DFlash2'
DRAFTER_REVISION = '50307d4c4cde6860d4eee73e2547cd786fe8e8a4'
DRAFTER_BITS = 4                         # --drafter-bits default (server/cli.zig:222), sent explicitly
MODEL_ID = 'qwen3.8-27b'                 # --name; the reply's model echoes a known request id (openai.zig:107-111)
# chat_template.jinja in the target snapshot: the hf2q Qwen3.8 template's bytes (git blob c0c686f9, 8952 B,
# rendered-inputs-reviewed/qwen38.template.jinja). Readiness recomputes it from the local snapshot.
TEMPLATE_SHA256 = 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
EFFORT = 'xhigh'
BUDGET_USER, BUDGET_TOOL = 2048, 512
LABELS = {'none': f'TensorFold {TF_COMMIT[:7]} (native Qwen3.8-27B MLX 4-bit, no drafter)',
          'dflash2': f'TensorFold {TF_COMMIT[:7]} + DFlash2 (native Qwen3.8-27B MLX 4-bit)'}
CANONICAL_KEYS = SHARED_KEYS             # qwen38_n1_smoke.build_request's Qwen envelope
NATIVE = {'temperature': 0, 'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'reasoning_effort': EFFORT,
          'chat_template_kwargs': {'enable_thinking': True}}
SAMPLING = {'temperature': 0, 'top_k': 0, 'top_p': 1.0, 'min_p': 0.0,
            'repetition_penalty': 'Unavailable:TensorFold has no repetition penalty (hf2q Qwen: 1.05 over 64)'}
_CONTROLS = {'engine': ENGINE, 'source_commit': TF_COMMIT, 'version': TF_VERSION,
             'target': f'{TARGET_REPO}@{TARGET_REVISION}', 'chat_template_sha256': TEMPLATE_SHA256,
             'cache_family': FAMILY, 'context_per_slot': CONTEXT_PER_SLOT, 'slots': SLOTS, 'parallel': 'fixed',
             'kv_cache': 'bf16', 'prompt_cache': 'auto', 'slot_resume': True, 'keep_warm_seconds': 900,
             'idle_unload_seconds': 0, 'learn': False, 'recall': False, 'loop_guard': False,
             'default_max_tokens': MAX_TOKENS, 'server_temperature': 0, 'repetition_penalty': 'Unavailable',
             'request_temperature': 0, 'request_top_p': 1.0, 'request_top_k': 0, 'request_min_p': 0.0,
             'request_reasoning_effort': EFFORT, 'request_enable_thinking': True,
             'request_thinking_budget_user': BUDGET_USER, 'request_thinking_budget_tool': BUDGET_TOOL,
             'request_parallel_tool_calls': 'omitted'}
CONTROLS = {'none': {**_CONTROLS, 'drafter': None, 'drafter_bits': None},
            'dflash2': {**_CONTROLS, 'drafter': f'{DRAFTER_REPO}@{DRAFTER_REVISION}', 'drafter_bits': DRAFTER_BITS}}
NATIVE_KEYS = frozenset({'tensorfold_source_binary_bound', 'tensorfold_target_snapshot_bound',
                         'tensorfold_drafter_state_bound', 'tensorfold_controls_bound',
                         'tensorfold_tokenize_render_audit_bound', 'tensorfold_slots_context_readiness_verified'})
FINISH = ('stop', 'tool_calls', 'length')
CALL_ID = re.compile(r'call_[0-9a-f]{24}')
TOP = frozenset({'id', 'object', 'created', 'model', 'choices', 'usage', 'exact_mode', 'tensorfold', 'speculative'})
CHUNK = frozenset({'id', 'object', 'created', 'model', 'choices'})
EXTRAS = frozenset({'exact_mode', 'tensorfold', 'speculative'})
RUNTIME = frozenset({'enable_thinking', 'reasoning_effort', 'engine', 'tokens_per_second', 'seconds',
                     'prefill_seconds', 'prefill_widths', 'prefill_raised', 'time_to_first_token', 'sampling',
                     'drafts', 'token_sha', 'token_sha256', 'min_rows'})
RUNTIME_OPTIONAL = frozenset({'gpu_seconds', 'loop'})  # TF_Q27_PROFILE only; a detected loop period
SPECULATIVE = frozenset({'rounds', 'drafted', 'accepted', 'acceptance_rate', 'tokens_per_round'})
SAME_GGUF_UNAVAILABLE = None             # Qwen3.8's same-GGUF peer is runnable: never --peer-unavailable-evidence
NATIVE_COMPARISON = ('native-comparison campaign; not the verdict of record: the Qwen3.8 same-GGUF peer '
                     'verdict comes from the campaign of record, and the TensorFold arm is reported '
                     'separately as native_peer (ADR-063 Q4)')
# native_peers.py interface (the registry resolves kind 'tensorfold' to this module).
CLIENT_MODULES = ('native_peers.py', 'native_peer_tensorfold.py', 'native_peer_tensorfold_ready.py',
                  'test_native_peer_tensorfold.py')
PROCESSES = ('tensorfold', 'tensorfold-native')  # packaging/homebrew/tensorfold.rb.in:18-23, README.md:375
BASELINES = {'tensorfold_native': LABELS['none'], 'tensorfold_dflash2_native': LABELS['dflash2']}
AUDIT = 'native_peer_tensorfold_audit'
PEER_CASE_REFUSAL = ('a native TensorFold --peer-case needs --native-comparison (Qwen3.8 has a runnable same-GGUF '
                     'peer, ADR-063 Q4)')


def _family(family):
    require(family == FAMILY, 'the native TensorFold arm serves Qwen3.8 only')


def variant(case):
    """'dflash2' when the q5 case launches a drafter, else 'none'."""
    return 'dflash2' if '--drafter' in case['serve']['argv'] else 'none'


def label(case):
    return LABELS[variant(case)]


def q3_role(case):
    return (f'{label(case)}: separately labeled different-artifact native comparison, task makespan only; '
            'not the same-GGUF peer verdict (ADR-063 Q4)')


def preflight(keys, family):
    """HF family/runtime keys are replaced by the native TensorFold bindings, never forged."""
    _family(family)
    from coding_family import preflight as family_preflight
    return (set(keys) - family_preflight(set(), family) - qwen_n4_peer.HF_ONLY) | NATIVE_KEYS


def controls_variant(controls):
    found = [name for name, expected in CONTROLS.items() if same(controls, expected)]
    require(len(found) == 1, 'TensorFold launch controls differ from the declared native settings')
    return found[0]


def validate_controls(controls, family):
    _family(family)
    controls_variant(controls)


def thinking_budget(messages):
    """hf2q's Qwen default ceiling: 512 when the latest non-system message is a tool result, else 2048."""
    require(isinstance(messages, list) and bool(messages), 'TensorFold message chain required')
    latest = next((m for m in reversed(messages) if not (isinstance(m, dict) and m.get('role') == 'system')), None)
    return BUDGET_TOOL if isinstance(latest, dict) and latest.get('role') == 'tool' else BUDGET_USER


def request_controls(request, family):
    """Canonical hf2q Qwen3.8 request -> exact TensorFold request; any other field refuses."""
    _family(family)
    extra = sorted(set(request) - CANONICAL_KEYS)
    require(not extra, 'unsupported TensorFold override: ' + ','.join(extra))
    require(set(request) == CANONICAL_KEYS, 'incomplete canonical TensorFold request')
    require(request['model'] == MODEL_ID, 'TensorFold model id must be ' + MODEL_ID)
    require(type(request['max_completion_tokens']) is int and request['max_completion_tokens'] == MAX_TOKENS,
            'canonical output allowance required')
    require(request['tool_choice'] == 'auto' and request['logprobs'] is False
            and type(request['stream']) is bool, 'canonical tool/transport controls required')
    budget = qwen_n4_peer.request_controls(request)['reasoning_budget_tokens']  # also checks roles/overflow
    require(budget == thinking_budget(request['messages']), 'Qwen thinking ceiling rule drift')
    native = {k: v for k, v in request.items() if k != 'hf2q_overflow_policy'}
    return {**native, 'tools': native_tools(request['tools']), **copy.deepcopy(NATIVE), 'thinking_budget': budget}


def _sorted(value):
    if isinstance(value, dict):
        return {key: _sorted(value[key]) for key in sorted(value)}
    return [_sorted(item) for item in value] if isinstance(value, list) else value


def native_tools(tools):
    """hf2q's rendered tool schemas: no `strict` key and every object key sorted (rendered-inputs-reviewed/
    qwen38-cold.prompt.txt:9-11); TensorFold's template sees the request's own order, and strict false is
    TensorFold's default (no tool grammar, server.zig:302-314)."""
    require(isinstance(tools, list) and all(isinstance(t, dict) and isinstance(t.get('function'), dict)
                                            and t['function'].get('strict') is False for t in tools),
            'canonical tool schemas carry strict false')
    return [_sorted({**t, 'function': {k: v for k, v in t['function'].items() if k != 'strict'}}) for t in tools]


def canonical_tools(tools):
    """Inverse of native_tools: sorted, strict-free schemas get strict false back; anything else refuses."""
    require(isinstance(tools, list) and all(isinstance(t, dict) and isinstance(t.get('function'), dict)
                                            and 'strict' not in t['function']
                                            and json.dumps(t) == json.dumps(_sorted(t)) for t in tools),
            'TensorFold tool schemas must be the sorted, strict-free hf2q render')
    return [{**t, 'function': {**t['function'], 'strict': False}} for t in tools]


def canonical(request, family):
    """Inverse of request_controls at the wire boundary: native fields must be exact, nothing else added."""
    _family(family)
    require('hf2q_overflow_policy' not in request, 'TensorFold request must use native transport fields')
    for key, value in NATIVE.items():
        require(key in request and same(request[key], value), 'TensorFold native control drift: ' + key)
    require(type(request.get('thinking_budget')) is int and request['thinking_budget'] == thinking_budget(
        request.get('messages')), 'TensorFold thinking budget drift')
    stripped = {k: v for k, v in request.items() if k not in NATIVE and k != 'thinking_budget'}
    stream = stripped.get('stream') is True
    require(set(stripped) - {'stream_options'} == CANONICAL_KEYS - {'hf2q_overflow_policy'}
            and ('stream_options' in stripped) == stream
            and (not stream or same(stripped['stream_options'], {'include_usage': True})),
            'TensorFold request carries a field outside the closed envelope')
    return {**stripped, 'tools': canonical_tools(stripped['tools']), 'hf2q_overflow_policy': 'reject'}


def compact_json(text):
    """No whitespace outside strings: TensorFold's compact argument JSON (tool_parse.zig:202-211)."""
    inside = escaped = False
    for ch in text:
        if inside:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                inside = False
        elif ch == '"':
            inside = True
        elif ch in ' \t\r\n':
            return False
    return not inside


def _tool_calls(calls, delta):
    require(isinstance(calls, list) and bool(calls), 'TensorFold tool_calls must be a nonempty array')
    for call in calls:
        function = call.get('function') if isinstance(call, dict) else None
        require(isinstance(function, dict), 'TensorFold tool-call shape')
        if not delta:
            require(set(call) == {'id', 'type', 'function'} and set(function) == {'name', 'arguments'},
                    'unknown TensorFold tool-call field')
        elif 'id' in call:
            require(set(call) == {'index', 'id', 'type', 'function'} and set(function) == {'name', 'arguments'}
                    and function['arguments'] == '', 'TensorFold streamed tool-call head')
        else:
            require(set(call) == {'index', 'function'} and set(function) == {'arguments'},
                    'TensorFold streamed tool-call arguments')
        require(not delta or type(call['index']) is int, 'TensorFold tool-call index')
        require(all(isinstance(v, str) for v in function.values()), 'TensorFold tool-call fields must be strings')
        require('id' not in call or (call['type'] == 'function' and CALL_ID.fullmatch(call['id']) is not None),
                'TensorFold tool-call id is not call_<24 hex>')
        require(delta or compact_json(function['arguments']), 'TensorFold tool-call arguments are not compact JSON')


def _finish(value, allow_none):
    require((allow_none and value is None) or value in FINISH,
            f'TensorFold finish_reason {value!r} (cancelled/content_filter/unknown refuse)')


def _extras(obj):
    """exact_mode, the per-request runtime facts (thinking, effort, greedy) and the draft counters."""
    require(obj.get('exact_mode') == 'exact' and isinstance(obj.get('stop_sequence', ''), str),
            'TensorFold exact_mode/stop_sequence extras')
    runtime = obj.get('tensorfold')
    require(isinstance(runtime, dict) and RUNTIME <= set(runtime) <= RUNTIME | RUNTIME_OPTIONAL,
            'TensorFold runtime extras envelope')
    require(runtime['enable_thinking'] is True and runtime['reasoning_effort'] == EFFORT,
            'TensorFold request did not render thinking at effort xhigh')
    require(runtime['engine'] == 'qwen27' and runtime['sampling'] == 'greedy' and type(runtime['drafts']) is bool,
            'TensorFold engine/greedy/draft runtime facts')
    require(isinstance(runtime['token_sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', runtime['token_sha256'])
            and runtime['token_sha'] == runtime['token_sha256'][:12], 'TensorFold reply token sha')
    speculative = obj.get('speculative')
    require(isinstance(speculative, dict) and SPECULATIVE <= set(speculative) <= SPECULATIVE | {'proposer'},
            'TensorFold speculative extras envelope')
    for key in ('rounds', 'drafted', 'accepted'):
        integer(speculative[key], 'TensorFold speculative ' + key)
    require(speculative['accepted'] <= speculative['drafted'], 'TensorFold accepted more drafts than it drafted')


def check_unary(obj):
    """Closed TensorFold chat.completion envelope (openai.zig:408-438, extras 322-330)."""
    require(isinstance(obj, dict) and TOP <= set(obj) <= TOP | {'stop_sequence'}
            and obj['object'] == 'chat.completion' and type(obj['created']) is int, 'TensorFold unary envelope')
    choices = obj['choices']
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and set(choices[0]) == {'index', 'message', 'finish_reason'}, 'TensorFold unary choice')
    message = choices[0]['message']
    require(isinstance(message, dict) and {'role', 'content'} <= set(message)
            <= {'role', 'content', 'reasoning_content', 'tool_calls'}, 'TensorFold unary message fields')
    calls = message.get('tool_calls')
    require(isinstance(message['content'], str) or (message['content'] is None and calls is not None),
            'TensorFold content is null only beside tool calls')
    require('reasoning_content' not in message or (isinstance(message['reasoning_content'], str)
                                                   and bool(message['reasoning_content'])),
            'TensorFold reasoning_content')
    if calls is not None:
        _tool_calls(calls, False)
    _finish(choices[0]['finish_reason'], False)
    _extras(obj)


def check_frame(obj):
    """Closed TensorFold chat.completion.chunk envelope (openai.zig:275-301, 580-591)."""
    require(isinstance(obj, dict) and obj.get('object') == 'chat.completion.chunk'
            and type(obj.get('created')) is int, 'TensorFold SSE envelope')
    if obj.get('choices') == []:
        require(set(obj) == CHUNK | {'usage'}, 'TensorFold SSE usage frame')
        return
    choices = obj.get('choices')
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and set(choices[0]) == {'index', 'delta', 'finish_reason'}, 'TensorFold SSE choice')
    delta, finish = choices[0]['delta'], choices[0]['finish_reason']
    require(isinstance(delta, dict) and set(delta) <= {'role', 'content', 'reasoning_content', 'tool_calls'},
            'unknown TensorFold delta field')
    _finish(finish, True)
    if finish is None:
        require(set(obj) == CHUNK, 'TensorFold SSE chunk fields')
    else:  # include_usage is always sent, so usage never rides on the final chunk
        require(delta == {} and CHUNK | EXTRAS <= set(obj) <= CHUNK | EXTRAS | {'stop_sequence'},
                'TensorFold SSE final chunk')
        _extras(obj)
    if 'tool_calls' in delta:
        _tool_calls(delta['tool_calls'], True)


def usage(value, turn):
    """TensorFold usage (openai.zig:307-320); cached and reasoning tokens are reported, never inferred."""
    require(isinstance(value, dict) and set(value) == {'prompt_tokens', 'completion_tokens', 'total_tokens',
                                                       'prompt_tokens_details', 'completion_tokens_details'},
            'TensorFold usage envelope')
    prompt = integer(value['prompt_tokens'], 'TensorFold prompt tokens', CONTEXT_PER_SLOT)
    completion = integer(value['completion_tokens'], 'TensorFold completion tokens', MAX_TOKENS)
    require(integer(value['total_tokens'], 'TensorFold total tokens') == prompt + completion,
            'TensorFold usage total contradiction')
    require(prompt + MAX_TOKENS <= CONTEXT_PER_SLOT,
            'TensorFold prompt plus full output exceeds the 262144 workload context')
    details, thought = value['prompt_tokens_details'], value['completion_tokens_details']
    require(isinstance(details, dict) and set(details) == {'cached_tokens'}, 'TensorFold cache details envelope')
    require(isinstance(thought, dict) and set(thought) == {'reasoning_tokens'}, 'TensorFold reasoning details envelope')
    cached = integer(details['cached_tokens'], 'TensorFold cached tokens', prompt)
    reasoning = integer(thought['reasoning_tokens'], 'TensorFold reasoning tokens', completion)
    return {**value, 'engine_kind': ENGINE, 'cache_observation': {
        'status': 'Observed', 'cached_tokens': cached, 'source_field': 'usage.prompt_tokens_details.cached_tokens',
        'semantics': 'prefix-restored tokens (chat.zig:569); a background warm prefill of the next turn '
                     '(warm.zig) can contribute', 'full_prefix_reuse': 'PendingPostOracle'},
        'reasoning_observation': {'status': 'Observed', 'reasoning_tokens': reasoning,
                                  'source_field': 'usage.completion_tokens_details.reasoning_tokens'}}


def readiness(base, model, output, packet):
    from native_peer_tensorfold_ready import readiness as ready
    return ready(base, model, output, packet)



def closing(base, output):
    """Every agent's /tokenize proofs and the final /health, after all agents finished (outside the makespan)."""
    from native_peer_tensorfold_ready import closing as close
    return close(base, output)


def campaign_identity(plan):
    from native_peer_tensorfold_ready import campaign_identity as identity
    return identity(plan)


def launch_problems(name, artifact, hf2q_sha256):
    from native_peer_tensorfold_ready import launch_problems as problems
    return problems(name, artifact, hf2q_sha256)
