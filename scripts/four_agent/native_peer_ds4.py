"""Native ds4 (DeepSeek-V4 Flash, antirez ds4 c97964b) benchmark arm: engine_kind 'ds4'.

Harness-only: no server launch, model load or inference fallback. ds4 serves its own
native artifact (the ds4f-q2 Flash 0731 GGUF, download_model.sh:16), not the hf2q GGUF, so
this arm is a separately labeled different-artifact comparison, never the same-GGUF peer.

Source inspected: /opt/ds4 at c97964b8aa77034a8e17afb10ca7a686c40ab511 (clean tree).
Request (ds4_server.c parse_chat_request 4115-4310):
- unknown keys are skipped (json_skip_value, 4255): logprobs and parallel_tool_calls are not
  ds4 fields, so they are not sent; one call per response is enforced on the reply instead.
- max_completion_tokens is the max_tokens alias (4175); tool_choice only distinguishes "none"
  (4155-4170); "think" is an explicit bool (4248-4253); reasoning_effort "max" -> Think Max
  (1037-1062, 4283-4286) but only when --ctx >= 393216, else silently normal thinking
  (ds4.c:424-427, 61096-61100).
- sampling defaults are temperature 1, top_p 1, min_p 0.05 (ds4.h:59-61; request_init 984-996);
  with thinking on an explicit request value wins (13966-13975, generate_job_inner, also the
  batched slot workers); temperature <= 0 is argmax (ds4.c:45544). ds4 has no repetition
  penalty (ds4_session_sample, ds4.c:78060-78064), i.e. hf2q's DeepSeek 1.0.
Reply: chat.completion (final_response 9016-9051), chunks (sse_chunk 7038-7068, 7121-7160,
7270-7329, openai_sse_finish_live 8149-8175), usage with prompt_tokens_details.cached_tokens +
cache_write_tokens (7075-7090; the usage chunk only with stream_options.include_usage,
7092-7119), finish stop/length/tool_calls/error (13926-14541), ": prefill" keepalive comments
(12712-12715), "event: error" frames (7021-7036), streamed tool ids reused (10774-10793).
Readiness: GET /v1/models (15315-15318, 15108-15189), served only after the model and all
sessions exist (listen at 16071-16088 follows session creation at 15993-16003). Identity: the
startup log (log_context_memory 15496-15528, batched line 16014-16020, trace 16035, listen
16088; ds4.c:43328-43445) and the --trace request headers (trace_begin 11983-12036).
"""
import hashlib
from pathlib import Path
import re
from urllib.parse import urlsplit
from urllib.request import urlopen

import qwen_n4_peer
from family_peer import CANONICAL_KEYS as PEER_CANONICAL_KEYS, same
from qwen38_n1_verdict import integer, loads, require

ENGINE = 'ds4'
FAMILY = 'deepseek4'
DS4_COMMIT = 'c97964b8aa77034a8e17afb10ca7a686c40ab511'
LABEL = 'ds4 c97964b (native Flash-0731 q2 artifact)'
MODEL_FILE = 'DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf'
MODEL_ID = 'deepseek-v4-flash'          # /v1/models id for a Flash GGUF (ds4_server.c:15183-15187)
SHAPE_NAME = 'DeepSeek V4 Flash'        # ds4.c:603; Vision-Exp and Pro report other names
THINK_MAX_MIN_CONTEXT = 393216          # DS4_THINK_MAX_MIN_CONTEXT, ds4.c:424-427
CONTEXT_PER_SLOT = 262144               # the canonical workload bound, enforced on usage
SLOTS = 4
MAX_TOKENS = 8192
TRACE_NAME = 'ds4-trace.txt'
CANONICAL_KEYS = PEER_CANONICAL_KEYS[FAMILY]
# Explicit ds4 fields beside the canonical temperature 0 / reasoning_effort max.
NATIVE = {'think': True, 'top_p': 1.0, 'min_p': 0.0, 'top_k': 0}
SAMPLING = {'temperature': 0, 'top_k': 0, 'top_p': 1.0, 'min_p': 0.0, 'repetition_penalty': 1.0}
CONTROLS = {'engine': ENGINE, 'source_commit': DS4_COMMIT, 'artifact_file': MODEL_FILE,
            'cache_family': FAMILY, 'context_per_slot': CONTEXT_PER_SLOT,
            'native_ctx_per_session': THINK_MAX_MIN_CONTEXT, 'slots': SLOTS, 'think_mode': 'max',
            'repetition_penalty': 1.0, 'speculation': 'none', 'disk_kv_cache': False, 'power_percent': 100,
            'default_tokens': MAX_TOKENS, 'trace': True, 'exact_dsml_tool_replay': True,
            'request_temperature': 0, 'request_top_p': 1.0, 'request_min_p': 0.0, 'request_top_k': 0,
            'request_reasoning_effort': 'max', 'request_think': True}
NATIVE_KEYS = frozenset({'ds4_source_binary_kernels_bound', 'ds4_native_artifact_bound', 'ds4_controls_bound',
                         'ds4_think_max_context_bound', 'ds4_trace_render_audit_bound',
                         'ds4_sessions_context_readiness_verified'})
FINISH = ('stop', 'tool_calls', 'length')
# The 26 runtime Metal sources ds4_metal.m:4754-4823 concatenates from the --chdir tree, at c97964b.
METAL_SHA256 = {
    'argsort.metal': '591a2d8472abefcfc3b38dec1fb04699e6e27a1af05ee7040570b0544a7702f1',
    'bin.metal': '4b537679fe874fa4dea079bbb10634b3feeb2cce9b737bdb602fe6926fbb175a',
    'concat.metal': '04fb0fc2a3b51015444e330290b979a56e4cac82aee523afa2b42e361ab23da7',
    'cpy.metal': 'df8cabe88c33076c16a54cfb88a00133b68ea5cbfc1742ac79608c84d4456cd4',
    'deepseek4_vision.metal': 'e0908e53ab5b33903408e5723434a59b28fb90a998123a32a794570fdf5a27ea',
    'dense.metal': '4f536cb6e2b7ac6c4faa45d708ad78b2298877d31bb0ce66d13e5f5b3b787bf5',
    'dsv41.metal': '35292e858dc606221377bd1b620895006e3687efd55e21b707a7923d666c90e2',
    'dsv4_hc.metal': '5d1988269671c9cfa9ad09024dd1c04b77559ea3894628e9bfddab44005b80d3',
    'dsv4_kv.metal': '65e6e12ac3266fdf72b1e9c69bfbd07ea7324e930a8117b63b651b5016b1b5f6',
    'dsv4_misc.metal': '033ebd3d93ce7f132c02fdc43eca83ccf9ba8cc2e5ba90c02551c173cc78de3d',
    'dsv4_rope.metal': 'f56cb24a381c27435953d120448bc22c1b2314b151280445246bd28920b4d518',
    'flash_attn.metal': 'c9ec79daf2063cd9e8c4e8874c123fbf2642c3fb6247b16aa21e1e0e6db92fcc',
    'get_rows.metal': 'bd3b352a6ffb138ae5e391db97b211f8386d417365eac2caa3730f55a075aa3a',
    'glm53_bf16.metal': '27d93dcfe81b09f93ed65cd986aaa0b9c1ca1df7fc43e58c15da533284d50db5',
    'glm53_kda.metal': 'a695c313e36b76fa260164c6f5a2f3203748cbcff9becc6707e71aa703ed0818',
    'glm53_vision.metal': 'd20bdf039f9bd9d7b0cab8c6a4fc10bcb3097571a81246dbde1c278e0e9bec7b',
    'glu.metal': '59281e30a27cc2eb618800b6e53a57e234f65f1a1e680d6267d9139e1a06b627',
    'moe.metal': 'b3ae62713a47e54c47959023a0d15c8589198ec6092307dfcf4014f180bc7f42',
    'norm.metal': '700bf92792e70a969c1f564c3c0ea7c8c9fe707d51a3537b766a9d336acfddd6',
    'qwen4.metal': 'dff0b9e48e5d55b68a818d4ccc28ea3468ecd464b8eff83cf426dcb9a102b4cf',
    'qwen4_vision.metal': 'b37c7fc2c66131bf5817621920a6b5b30271af21160e7da33edc71e2d5c7fb4b',
    'repeat.metal': '2c6142b47a2007d246f5eb6e7395047627bb6dfa02c649311d16620da35007cd',
    'set_rows.metal': '25f597bf5f5fb5a3791efc68e3182539727609f09c45750ef9376ed430d7a029',
    'softmax.metal': '6eaa6ce78c4b221b3c7fe45647d6eb42f165201992a243d77d04cf2668fb7664',
    'sum_rows.metal': '7e1f3ad3f945b1f6f59aa2d8ee5bf464052d989a632c33ea95ce50d3c4fbf0fc',
    'unary.metal': '4607dbea21501230a193f8a333b1bd543d566f405568a094d6faf2c17cb0c5cf'}
# ds4's Think Max prompt prefix (ds4.c:419-422), rendered after BOS (ds4_server.c:3132-3134).
# It is the hf2q 0731 template's `high` tier text; that template's `max` tier ("Beyond
# maximum ...") has no ds4 counterpart, so prompt-byte parity with hf2q is Unavailable.
BOS = '<｜begin▁of▁sentence｜>'
THINK_MAX_PREFIX = ('Reasoning Effort: Absolute maximum with no shortcuts permitted.\n'
                    'You MUST be very thorough in your thinking and comprehensively decompose the problem to '
                    'resolve the root cause, rigorously stress-testing your logic against all potential paths, '
                    'edge cases, and adversarial scenarios.\n'
                    'Explicitly write out your entire deliberation process, documenting every intermediate step, '
                    'considered alternative, and rejected hypothesis to ensure absolutely no assumption is left '
                    'unchecked.\n\n')
# Known prompt differences from hf2q's 0731 template (deepseek-v4-flash-0731.jinja:42-46, 54-60):
# ds4 puts tool schemas BEFORE the client system text (ds4_server.c:3114-3133), and its Think Max
# prefix is the template's weaker `high` wording, which likely shortens ds4 reasoning (favours ds4).
TEMPLATE = ('Unavailable:ds4 renders DSML/think prompts in C (no template file); differs from the hf2q 0731 '
            'template in tools-before-system ordering and in using the template `high` (not `max`) effort text, '
            'which likely shortens ds4 reasoning and favours ds4 on makespan; per-request think_mode and the ds4 '
            'Think Max rendered-prompt prefix are proven by the --trace audit')
SUPPORTED = ['tools', 'tool_choice', 'max_tokens', 'temperature', 'top_p', 'top_k', 'min_p', 'ignore_eos',
             'stop', 'seed', 'stream', 'reasoning_effort']  # ds4_server.c:15124-15136
Q3_ROLE = (f'{LABEL}: separately labeled different-artifact comparison, task makespan only; '
           'not the same-GGUF peer verdict (ADR-063 Q4)')
SAME_GGUF_UNAVAILABLE = ('same-GGUF peer unavailable; the peer arm is the native ds4 different-artifact '
                         'comparison, reported separately as native_peer (ADR-063 Q4)')
NATIVE_COMPARISON = None  # DeepSeek-V4 runs only with the same-GGUF --peer-unavailable-evidence
# native_peers.py interface (the registry resolves kind 'ds4' to this module).
TITLE = 'DeepSeek-V4'
Q3_FAMILY = FAMILY
CLIENT_MODULES = ('native_peers.py', 'native_peer_ds4.py', 'test_native_peer_ds4.py')
PROCESSES = ('ds4', 'ds4-server', 'ds4-agent', 'ds4-bench', 'ds4-eval')  # native ds4 binaries (/opt/ds4 Makefile:78)
BASELINES = {'ds4_native': LABEL}
AUDIT = 'native_peer_ds4_audit'
closing = None
PEER_CASE_REFUSAL = ('a native ds4 --peer-case needs --peer-unavailable-evidence for the same-GGUF peer (ADR-063 Q4), '
                     'and that evidence admits no same-GGUF peer case')


def label(case):
    return LABEL


def q3_role(case):
    return Q3_ROLE


def _family(family):
    require(family == FAMILY, 'the native ds4 arm serves DeepSeek-V4 only')


def preflight(keys, family):
    """HF family/runtime keys are replaced by the native ds4 bindings, never forged."""
    _family(family)
    from coding_family import preflight as family_preflight
    return (set(keys) - family_preflight(set(), family) - qwen_n4_peer.HF_ONLY) | NATIVE_KEYS


def validate_controls(controls, family):
    _family(family)
    require(isinstance(controls, dict) and type(controls.get('native_ctx_per_session')) is int
            and controls['native_ctx_per_session'] >= THINK_MAX_MIN_CONTEXT,
            'ds4 Think Max needs native_ctx_per_session >= 393216 (no silent normal-thinking downgrade)')
    expected = {**CONTROLS, 'native_ctx_per_session': controls['native_ctx_per_session']}
    require(same(controls, expected), 'ds4 launch controls differ from the declared native settings')


def request_controls(request, family):
    """Canonical hf2q DeepSeek-V4 request -> exact ds4 request; any other field refuses."""
    _family(family)
    extra = sorted(set(request) - CANONICAL_KEYS)
    require(not extra, 'unsupported ds4 override: ' + ','.join(extra))
    require(set(request) == CANONICAL_KEYS, 'incomplete canonical ds4 request')
    require(request['model'] == MODEL_ID, 'ds4 model id must be ' + MODEL_ID)
    require(request['hf2q_overflow_policy'] == 'reject', 'canonical overflow policy required')
    require(type(request['max_completion_tokens']) is int and request['max_completion_tokens'] == MAX_TOKENS,
            'canonical output allowance required')
    require(request['tool_choice'] == 'auto' and request['logprobs'] is False
            and type(request['stream']) is bool, 'canonical tool/transport controls required')
    require(type(request['temperature']) in (int, float) and request['temperature'] == 0,
            'DeepSeek greedy temperature required')
    require(request['reasoning_effort'] == 'max', 'DeepSeek max reasoning required')
    messages = request['messages']
    require(isinstance(messages, list) and bool(messages), 'ds4 message chain required')
    for message in messages:
        require(isinstance(message, dict) and message.get('role') in ('system', 'user', 'assistant', 'tool'),
                'invalid ds4 message role')
        calls = message.get('tool_calls') if message['role'] == 'assistant' else None
        require(calls is None or isinstance(calls, list), 'invalid assistant tool-call cycles')
    # logprobs=false is ds4's only behavior; overflow is refused by the harness's 262144 usage bound.
    return {**{k: v for k, v in request.items() if k not in ('hf2q_overflow_policy', 'logprobs')}, **NATIVE}


def canonical(request, family):
    """Inverse of request_controls at the wire boundary: native fields must be exact."""
    _family(family)
    require('hf2q_overflow_policy' not in request and 'logprobs' not in request,
            'ds4 request must use native transport fields')
    for key, value in NATIVE.items():
        require(key in request and same(request[key], value), 'ds4 native control drift: ' + key)
    stripped = {k: v for k, v in request.items() if k not in NATIVE}
    return {**stripped, 'logprobs': False, 'hf2q_overflow_policy': 'reject'}


def _tool_calls(calls, delta):
    require(isinstance(calls, list) and calls, 'ds4 tool_calls must be a nonempty array')
    for call in calls:
        keys = {'id', 'type', 'function'}
        require(isinstance(call, dict) and (set(call) <= keys | {'index'} if delta else set(call) == keys),
                'unknown ds4 tool-call field')
        function = call.get('function', {})
        require(isinstance(function, dict) and (set(function) <= {'name', 'arguments'} if delta
                                                else set(function) == {'name', 'arguments'}),
                'unknown ds4 tool-call function field')
        require(all(isinstance(v, str) for v in function.values()), 'ds4 tool-call fields must be strings')


def _finish(value, allow_none):
    require(value != 'error', 'ds4 finish_reason error (generation or tool-parse failure)')
    require((allow_none and value is None) or value in FINISH, 'unknown ds4 finish_reason')


def check_unary(obj):
    """Closed ds4 chat.completion envelope (final_response, ds4_server.c:9016-9051)."""
    require(isinstance(obj, dict) and set(obj) == {'id', 'object', 'created', 'model', 'choices', 'usage'}
            and obj['object'] == 'chat.completion' and type(obj['created']) is int, 'ds4 unary envelope')
    choices = obj['choices']
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and set(choices[0]) == {'index', 'message', 'finish_reason'}, 'ds4 unary choice')
    message = choices[0]['message']
    require(isinstance(message, dict) and {'role', 'content'} <= set(message)
            <= {'role', 'content', 'reasoning_content', 'tool_calls'} and isinstance(message['content'], str),
            'ds4 unary message fields')
    if 'tool_calls' in message:
        _tool_calls(message['tool_calls'], False)
    _finish(choices[0]['finish_reason'], False)


def check_frame(obj):
    """Closed ds4 chat.completion.chunk envelope (ds4_server.c:7038-7160, 7270-7329, 8149-8175)."""
    require(isinstance(obj, dict) and obj.get('object') == 'chat.completion.chunk'
            and type(obj.get('created')) is int, 'ds4 SSE envelope')
    if obj.get('choices') == []:
        require(set(obj) == {'id', 'object', 'created', 'model', 'choices', 'usage'}, 'ds4 SSE usage frame')
        return
    require(set(obj) == {'id', 'object', 'created', 'model', 'choices'} and isinstance(obj['choices'], list)
            and len(obj['choices']) == 1 and isinstance(obj['choices'][0], dict)
            and set(obj['choices'][0]) == {'index', 'delta', 'finish_reason'}, 'ds4 SSE choice')
    delta = obj['choices'][0]['delta']
    require(isinstance(delta, dict) and set(delta) <= {'role', 'content', 'reasoning_content', 'tool_calls'},
            'unknown ds4 delta field')
    if 'tool_calls' in delta:
        _tool_calls(delta['tool_calls'], True)
    _finish(obj['choices'][0]['finish_reason'], True)


def usage(value, turn):
    """ds4 usage (ds4_server.c:7075-7090); cached tokens are always reported, never inferred."""
    require(isinstance(value, dict) and set(value) == {'prompt_tokens', 'completion_tokens', 'total_tokens',
                                                       'prompt_tokens_details'}, 'ds4 usage envelope')
    prompt = integer(value['prompt_tokens'], 'ds4 prompt tokens', CONTEXT_PER_SLOT)
    completion = integer(value['completion_tokens'], 'ds4 completion tokens', MAX_TOKENS)
    require(integer(value['total_tokens'], 'ds4 total tokens') == prompt + completion, 'ds4 usage total contradiction')
    require(prompt + MAX_TOKENS <= CONTEXT_PER_SLOT, 'ds4 prompt plus full output exceeds the 262144 workload context')
    details = value['prompt_tokens_details']
    require(isinstance(details, dict) and set(details) == {'cached_tokens', 'cache_write_tokens'},
            'ds4 cache details envelope')
    cached = integer(details['cached_tokens'], 'ds4 cached tokens', prompt)
    written = integer(details['cache_write_tokens'], 'ds4 cache write tokens', prompt - cached)
    return {**value, 'engine_kind': ENGINE, 'cache_observation': {
        'status': 'Observed', 'cached_tokens': cached, 'source_field': 'usage.prompt_tokens_details.cached_tokens',
        'cache_write_tokens': written,
        'cache_write_source_field': 'usage.prompt_tokens_details.cache_write_tokens (ds4 extension)',
        'full_prefix_reuse': 'PendingPostOracle'}}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build_info(commit, binary_sha256):
    """Campaign identity string (q3 drift check); None while either input is unresolved."""
    if not (isinstance(commit, str) and isinstance(binary_sha256, str)
            and re.fullmatch(r'[0-9a-f]{40}', commit) and re.fullmatch(r'[0-9a-f]{64}', binary_sha256)):
        return None
    return f'ds4 {commit} binary-sha256 {binary_sha256}'


def runtime_identity(ds4_dir):
    """sha256 of every runtime Metal source under DS4_DIR/metal (None when the tree is absent)."""
    metal = Path(ds4_dir) / 'metal' if isinstance(ds4_dir, str) else None
    if metal is None or not metal.is_dir():
        return None
    return {p.name: sha256(p) for p in sorted(metal.glob('*.metal')) if p.is_file()}


def campaign_identity(plan):
    variables = plan['variables']
    return {'peer_build_info': build_info(variables.get('DS4_COMMIT'), variables.get('DS4_BIN_SHA256')),
            'native_runtime': runtime_identity(variables.get('DS4_DIR'))}


def launch_problems(name, artifact, hf2q_sha256):
    """Q3 launch refusals: an ambient hf2q MODEL/MODEL_SHA256 must never stand in for the native artifact."""
    problems = []
    if artifact and Path(artifact['path']).name != MODEL_FILE:
        problems.append(f'{name}: native artifact must be {MODEL_FILE} (set --peer-var MODEL)')
    if artifact.get('sha256') and artifact['sha256'] == hf2q_sha256:
        problems.append(f"{name}: native artifact sha256 equals the hf2q artifact's (set --peer-var MODEL_SHA256)")
    return problems


def native_spec(spec):
    """The approval's `native_ds4` block (q5 case client.approval.set)."""
    keys = {'binary', 'binary_sha256', 'commit', 'chdir', 'model', 'model_sha256', 'ctx', 'port', 'server_log', 'trace'}
    require(isinstance(spec, dict) and set(spec) == keys and all(isinstance(spec[k], str) for k in keys),
            'closed native_ds4 approval block')
    paths = {k: Path(spec[k]) for k in ('binary', 'chdir', 'model', 'server_log', 'trace')}
    require(all(p.is_absolute() for p in paths.values()), 'native_ds4 paths must be absolute')
    require(paths['binary'].name == 'ds4-server' and paths['binary'].parent == paths['chdir'],
            'ds4-server must run from its own source tree (--chdir; Metal sources are read at runtime)')
    require(spec['commit'] == DS4_COMMIT, 'ds4 source commit is not the inspected c97964b')
    require(re.fullmatch(r'[0-9a-f]{64}', spec['binary_sha256']) and re.fullmatch(r'[0-9a-f]{64}', spec['model_sha256']),
            'DS4_BIN_SHA256 and MODEL_SHA256 must be 64 lowercase hex')
    require(paths['model'].name == MODEL_FILE, 'ds4 must serve the ds4f-q2 Flash 0731 artifact')
    require(spec['ctx'].isdecimal() and int(spec['ctx']) >= THINK_MAX_MIN_CONTEXT,
            'NATIVE_DS4_CTX below 393216 silently downgrades Think Max to normal thinking')
    require(spec['port'].isdecimal(), 'native_ds4 port')
    require(paths['server_log'].name == 'server.stderr' and paths['trace'].name == TRACE_NAME
            and paths['trace'].parent == paths['server_log'].parent, 'ds4 server log/trace must be in the run directory')
    return {**spec, 'ctx': int(spec['ctx']), 'port': int(spec['port'])}


def verify_identity(spec):
    require(sha256(spec['binary']) == spec['binary_sha256'], 'ds4-server sha256 differs from DS4_BIN_SHA256')
    require(runtime_identity(spec['chdir']) == METAL_SHA256, 'ds4 runtime Metal sources are not c97964b')
    stat = Path(spec['model']).stat()
    return {'source_revision': DS4_COMMIT, 'binary': spec['binary'], 'binary_sha256': spec['binary_sha256'],
            'binary_provenance': 'operator-declared DS4_BIN_SHA256 of the c97964b build; not reproducible-build verified',
            'metal_sources': 'c97964b (26 files, sha256 pinned)',
            'artifact': {'path': spec['model'], 'bytes': stat.st_size, 'sha256': spec['model_sha256'],
                         'sha256_source': 'operator-provided, not hashed inline (q3_identity policy)'}}


def validate_models(models, model, ctx):
    require(model == MODEL_ID, 'ds4 model id must be ' + MODEL_ID)
    require(isinstance(models, dict) and set(models) == {'object', 'data'} and models['object'] == 'list'
            and isinstance(models['data'], list), 'ds4 /v1/models envelope')
    require([m.get('id') if isinstance(m, dict) else None for m in models['data']] == [MODEL_ID, 'deepseek-v4-pro'],
            'ds4 did not load a DeepSeek V4 Flash/Pro-shape GGUF')
    for entry in models['data']:
        require(set(entry) == {'id', 'object', 'created', 'owned_by', 'name', 'context_length', 'top_provider',
                               'supported_parameters'} and entry['object'] == 'model' and entry['owned_by'] == 'ds4.c'
                and entry['name'] == SHAPE_NAME and entry['supported_parameters'] == SUPPORTED,
                'ds4 /v1/models entry is not the V4 Flash text model')
        found = entry['context_length']
        require(type(found) is int and found >= THINK_MAX_MIN_CONTEXT and (ctx is None or found == ctx),
                'ds4 per-session context is not the declared Think Max context')
        ctx = found
        require(same(entry['top_provider'], {'context_length': ctx, 'max_completion_tokens': MAX_TOKENS,
                                             'is_moderated': False}), 'ds4 context/output allowance (--tokens 8192)')
    return ctx


STAMP = r'^\d{4} \d\d:\d\d:\d\d ds4-server: '
LOG = {'context': re.compile(STAMP + r'context buffers ([0-9.]+) MiB \(ctx=(\d+), backend=(\w+), prefill_chunk=(\d+), '
                             r'raw_kv_rows=(\d+), compressed_kv_rows=(\d+)\)$', re.M),
       'sessions': re.compile(STAMP + r'(\d+) resident sessions request at least ([0-9.]+) GiB of context buffers, '
                              r'or ([0-9.]+) GiB when the model shares one prefill workspace$', re.M),
       'batched': re.compile(STAMP + r'batched mode enabled resident_sessions=(\d+) prefill_quantum=(\d+) '
                             r'mixed_prefill_quantum=(\d+) decode_coalesce_us=(\d+)$', re.M),
       'trace': re.compile(STAMP + r'tracing session to (.+)$', re.M),
       'listen': re.compile(STAMP + r'listening on http://([0-9.]+):(\d+)$', re.M)}
ENGINE_LOG = {'memory': re.compile(r'^ds4: memory: KV ([0-9.]+) GiB \(raw ([0-9.]+) \+ compressed ([0-9.]+)\) '
                                   r'\+ buffers ([0-9.]+) GiB \+ resident model ([0-9.]+) GiB = ([0-9.]+) GiB planned$', re.M),
              'detail': re.compile(r'^ds4: memory detail: ctx=(\d+) prefill_cap=(\d+) raw_kv_rows=(\d+) '
                                   r'compressed_kv_rows=(\d+) backend=(\w+)$', re.M)}
FORBIDDEN_LOG = ('MTP speculative decoding is disabled', 'exact DSML tool replay disabled')


def startup(log, ctx, port=None, trace=None):
    """Context/sessions/memory facts of the startup log; a missing or repeated line refuses."""
    require(isinstance(log, str), 'ds4 startup log')
    require(not any(text in log for text in FORBIDDEN_LOG), 'ds4 started with MTP or without exact DSML replay')
    found = {name: pattern.findall(log) for name, pattern in LOG.items()}
    require(all(len(rows) == 1 for rows in found.values()), 'ds4 startup log lines missing or repeated')
    context, sessions, batched = found['context'][0], found['sessions'][0], found['batched'][0]
    host, listen_port = found['listen'][0]
    require(int(context[1]) == ctx and context[2] == 'metal', 'ds4 startup context/backend')
    require(int(sessions[0]) == SLOTS and int(batched[0]) == SLOTS, 'ds4 must hold four resident sessions')
    require(host == '127.0.0.1' and (port is None or int(listen_port) == port), 'ds4 listener')
    require(trace is None or found['trace'][0] == trace, 'ds4 trace path')
    engine = {}
    for name, pattern in ENGINE_LOG.items():
        rows = pattern.findall(log)
        require(len(rows) <= 1, 'repeated ds4 engine memory line')
        engine[name] = rows[0] if rows else None
    require(engine['detail'] is None or (int(engine['detail'][0]) == ctx and engine['detail'][4] == 'metal'),
            'ds4 engine memory detail contradicts the context')
    memory = engine['memory']
    return {'native_ctx_per_session': ctx, 'backend': 'metal', 'prefill_chunk': int(context[3]),
            'context_buffers_mib_per_session': float(context[0]), 'resident_sessions': SLOTS,
            'sessions_context_gib': float(sessions[1]), 'sessions_context_gib_shared_prefill': float(sessions[2]),
            'prefill_quantum': int(batched[1]), 'trace_path': found['trace'][0], 'listen_port': int(listen_port),
            'engine_memory_plan_gib': ({'status': 'Unavailable:engine memory line absent'} if memory is None else
                                       {'status': 'Observed', 'kv': float(memory[0]), 'buffers': float(memory[3]),
                                        'resident_model': float(memory[4]), 'planned_one_session': float(memory[5])})}


def validate_readiness(models, log, model, ctx=None, port=None, trace=None):
    ctx = validate_models(models, model, ctx)
    facts = startup(log, ctx, port, trace)
    return {'engine_kind': ENGINE, 'family': FAMILY, 'label': LABEL, 'source_revision': DS4_COMMIT,
            'logical_slots': SLOTS, 'context_per_slot': CONTEXT_PER_SLOT, 'native_ctx_per_session': ctx,
            'think_mode': 'max', 'sampling': dict(SAMPLING), 'startup': facts, 'chat_template': TEMPLATE,
            'physical_cohort': 'Unavailable:not proved by readiness or HTTP overlap',
            'pool_profile': 'NotApplicable:HF-only', 'full_prefix_reuse': 'PendingPostOracle'}


def fetch_models(base, output):
    """Raw /v1/models reply, archived before any validation."""
    u = urlsplit(base)
    require(u.scheme == 'http' and u.hostname == '127.0.0.1' and u.port and u.path == '/v1'
            and not u.query and not u.fragment and u.username is None, 'ds4 loopback base')
    with urlopen(f'http://127.0.0.1:{u.port}/v1/models', timeout=2) as response:
        raw = response.read(1024 * 1024 + 1)
        (Path(output) / 'ds4-models.raw').write_bytes(raw)
        require(response.status == 200 and len(raw) <= 1024 * 1024, 'ds4 readiness status/size')
    return loads(raw)


def readiness(base, model, output, packet):
    """Coordinator readiness: approval `native_ds4` + `family_controls`, binary/kernels, /v1/models, startup log."""
    require(isinstance(packet, dict), 'ds4 approval packet')
    spec = native_spec(packet.get('native_ds4'))
    validate_controls(packet.get('family_controls'), FAMILY)
    require(packet['family_controls']['native_ctx_per_session'] == spec['ctx'],
            'approved native_ctx_per_session differs from the launch NATIVE_DS4_CTX')
    require(urlsplit(base).port == spec['port'], 'ds4 base URL port differs from the launch port')
    identity = verify_identity(spec)
    models = fetch_models(base, output)
    log = Path(spec['server_log']).read_bytes()
    (Path(output) / 'ds4-startup.log').write_bytes(log)
    record = validate_readiness(models, log.decode('utf-8', 'replace'), model, spec['ctx'], spec['port'], spec['trace'])
    return {**record, 'identity': identity, 'build_info': build_info(spec['commit'], spec['binary_sha256'])}
