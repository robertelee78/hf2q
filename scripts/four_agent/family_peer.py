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
import json
import math
from pathlib import Path
import re

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


def request_controls(request, family, setup=None):
    """Canonical hf2q request -> native peer request; any other field refuses."""
    if family == 'qwen':
        return qwen_n4_peer.request_controls(request, setup)
    require(setup is None, 'request setups are Qwen only')
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


def canonical(request, family, setup=None):
    """Inverse of request_controls at the wire boundary: native fields must be exact."""
    require('hf2q_overflow_policy' not in request, 'peer request must use native transport metadata')
    if family == 'qwen':
        if setup is not None:
            import request_setup
            require('reasoning_budget_tokens' not in request, 'request setup peer budget drift')
            request = request_setup.from_peer(request, setup)
        return {**request, 'hf2q_overflow_policy': 'reject'}
    require(setup is None, 'request setups are Qwen only')
    _family(family)
    for key, value in NATIVE[family].items():
        require(key in request and same(request[key], value), 'peer native control drift: ' + key)
    stripped = {k: v for k, v in request.items() if k not in NATIVE[family]}
    return {**stripped, 'hf2q_overflow_policy': 'reject'}


def _close(value, expected):
    # /props serializes float32 sampling values (1.05 -> 1.0499999523162842).
    return type(value) in (int, float) and type(value) is not bool and math.isclose(
        value, expected, rel_tol=0, abs_tol=1e-6)


def served_template_matches(served, family):
    """/props reports the peer lexer's source, not the file bytes (chat.cpp:1344).

    The lexer turns CR/CRLF into LF and strips one trailing newline
    (common/jinja/lexer.cpp:44-60 at e86f5fd1), as Jinja's default
    keep_trailing_newline=false does when rendering. Both pinned templates are
    CR-free and end in exactly one newline, so the served source must be the
    pinned bytes or the pinned bytes minus that newline; nothing else matches.
    """
    if not isinstance(served, str) or '\r' in served:
        return False
    pin = TEMPLATE_SHA256[family]
    return any(hashlib.sha256(text.encode()).hexdigest() == pin for text in (served, served + '\n'))


def _selected_speculation(reference):
    require(isinstance(reference, dict), 'peer reference binding required')
    argv = reference.get('serve_argv')
    require(isinstance(argv, list) and bool(argv) and all(isinstance(arg, str) for arg in argv),
            'peer reference launch argv required')
    require(argv.count('--spec-type') == 1 and argv.index('--spec-type') + 1 < len(argv),
            'peer reference requires one explicit speculation setting')
    selected = argv[argv.index('--spec-type') + 1]
    require(selected in ('none', 'draft-mtp', 'ngram-simple', 'ngram-mod') and
            reference.get('selected_speculation') == selected,
            'peer reference speculation differs from the frozen launch selection')
    return selected


def _supervised_launch(packet, base_url=None):
    launch = packet.get('peer_supervised_launch')
    require(isinstance(launch, dict) and set(launch) == {
        'schema', 'server_pid', 'base_url', 'binary', 'binary_sha256', 'serve_argv'} and
        launch.get('schema') == 'hf2q.peer.supervised-launch.v1',
        'peer reference requires supervisor-issued launch identity')
    require(type(launch.get('server_pid')) is int and launch['server_pid'] > 0 and
            launch['server_pid'] == packet.get('server_pid') and
            launch.get('base_url') == packet.get('base_url') and
            (base_url is None or launch['base_url'] == base_url),
            'peer supervised PID/endpoint binding differs')
    return launch


def _reference_identity(reference, build, supervised):
    require(isinstance(reference, dict) and set(reference) == {
        'identity_receipt', 'identity_sha256', 'source_commit', 'build_info',
        'serve_argv', 'selected_speculation'}, 'peer reference binding fields')
    commit = reference.get('source_commit')
    require(isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit),
            'peer reference requires a full source commit')
    expected = reference.get('build_info')
    short = expected.split('-', 1)[1] if isinstance(expected, str) and '-' in expected else ''
    require(isinstance(expected, str) and re.fullmatch(r'b[0-9]+-[0-9a-f]{7,40}', expected) and
            commit.startswith(short) and build == expected,
            'peer build differs from the frozen source/build identity')
    receipt_path, digest = reference.get('identity_receipt'), reference.get('identity_sha256')
    require(isinstance(receipt_path, str) and Path(receipt_path).is_absolute() and
            isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest),
            'peer identity receipt path/hash required')
    raw = Path(receipt_path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'peer identity receipt hash mismatch')
    identity = json.loads(raw)
    require(isinstance(identity, dict) and identity.get('source_commit') == commit and
            identity.get('peer_build_info') == expected, 'peer identity receipt source/build mismatch')
    selected = _selected_speculation(reference)
    require(identity.get('binary') == reference['serve_argv'][0],
            'peer reference executable differs from the frozen launch')
    require(isinstance(supervised, dict) and supervised.get('serve_argv') == reference['serve_argv'] and
            supervised.get('binary') == identity['binary'] and
            supervised.get('binary_sha256') == identity.get('binary_sha256'),
            'peer reference differs from actual supervised executable/argv')
    binary_hash = identity.get('binary_sha256')
    require(isinstance(binary_hash, str) and re.fullmatch(r'[0-9a-f]{64}', binary_hash) and
            hashlib.sha256(Path(identity['binary']).read_bytes()).hexdigest() == binary_hash,
            'peer reference executable hash mismatch')
    libraries = identity.get('libraries_and_binary_sha256')
    require(isinstance(libraries, dict) and libraries.get(identity['binary']) == binary_hash,
            'peer identity receipt must bind the executable and sibling libraries')
    siblings = {str(path) for path in Path(identity['binary']).parent.glob('*.dylib')
                if path.is_file() and not path.is_symlink()}
    require(siblings <= set(libraries), 'peer identity receipt omits actual sibling libraries')
    external = identity.get('external_dylibs_sha256', {})
    require(isinstance(external, dict), 'peer external library identity fields')
    hashes = {Path(identity['binary']).resolve(): binary_hash}
    for name, expected_hash in {**libraries, **external}.items():
        require(isinstance(name, str) and Path(name).is_absolute() and
                isinstance(expected_hash, str) and re.fullmatch(r'[0-9a-f]{64}', expected_hash),
                'peer library identity path/hash required')
        path = Path(name).resolve()
        if path not in hashes:
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        require(hashes[path] == expected_hash, 'peer sibling/external library hash mismatch')
    return {'source_revision': commit, 'build_info': expected,
            'identity_receipt': receipt_path, 'identity_sha256': digest,
            'binary_sha256': binary_hash, 'selected_speculation': selected,
            'serve_argv_sha256': hashlib.sha256(json.dumps(
                reference['serve_argv'], separators=(',', ':')).encode()).hexdigest()}


def validate_speculation_proof(output, packet):
    """Check actual retained replies, never /props sampler metadata, for MTP exercise."""
    reference = packet.get('peer_reference')
    if reference is None:
        return None
    binding = _reference_identity(reference, reference.get('build_info') if isinstance(reference, dict) else None,
                                  _supervised_launch(packet))
    selected = binding['selected_speculation']
    transports = {mode: {'responses': 0, 'drafted_tokens': 0, 'accepted_tokens': 0}
                  for mode in ('unary', 'sse')}
    for turn in sorted(Path(output).glob('agent-*/turn-*')):
        receipt = json.loads((turn / 'receipt.json').read_bytes())
        raw = (turn / 'response.bin').read_bytes()
        require(hashlib.sha256(raw).hexdigest() == receipt.get('response_sha256'),
                'peer response proof hash mismatch')
        require(type(receipt.get('stream')) is bool, 'peer response proof transport missing')
        mode = 'sse' if receipt['stream'] else 'unary'
        if receipt['stream']:
            frames = [json.loads(line[5:].strip()) for line in raw.splitlines()
                      if line.startswith(b'data:') and line[5:].strip() != b'[DONE]']
        else:
            frames = [json.loads(raw)]
        timings = [frame['timings'] for frame in frames if isinstance(frame, dict) and
                   isinstance(frame.get('timings'), dict)]
        if selected == 'draft-mtp':
            require(bool(timings) and {'draft_n', 'draft_n_accepted'} <= set(timings[-1]),
                    f'peer MTP {mode} response lacks draft timing counters')
        if not timings:
            continue
        latest = timings[-1]
        n, a = latest.get('draft_n', 0), latest.get('draft_n_accepted', 0)
        require(type(n) is int and type(a) is int and 0 <= a <= n,
                'peer draft proof counters malformed')
        transports[mode]['responses'] += 1
        transports[mode]['drafted_tokens'] += n
        transports[mode]['accepted_tokens'] += a
    if selected == 'draft-mtp':
        for mode, proof in transports.items():
            require(all(value > 0 for value in proof.values()),
                    f'declared peer MTP was not exercised with nonzero drafted/accepted tokens on {mode}')
    drafted = sum(proof['drafted_tokens'] for proof in transports.values())
    accepted = sum(proof['accepted_tokens'] for proof in transports.values())
    if selected == 'none':
        require(drafted == 0 and accepted == 0, 'peer draft counters contradict no speculation')
    return {'selected_speculation': selected, 'transports': transports,
            'responses_with_timings': sum(proof['responses'] for proof in transports.values()),
            'drafted_tokens': drafted, 'accepted_tokens': accepted,
            'source': 'retained actual request response bytes and their receipt hashes',
            'sampler_metadata_is_launch_selection': False}


def validate_readiness(health, props, slots, model, family, reference=None, supervised=None):
    if family == 'qwen':
        result = qwen_n4_peer.validate_readiness(health, props, slots, model)
        return {**result, **_reference_identity(reference, props.get('build_info'), supervised)} if reference is not None else result
    _family(family)
    qwen_n4_peer.validate_readiness(health, props, slots, model)  # shared slot/context/alias checks
    build = props.get('build_info')
    if reference is None:
        commit = build.split('-', 1)[1] if isinstance(build, str) and '-' in build else ''
        require(len(commit) >= 7 and PEER_COMMIT.startswith(commit), 'peer build is not the inspected source')
        binding = {'source_revision': PEER_COMMIT, 'build_info': build}
    else:
        binding = _reference_identity(reference, build, supervised)
    template = props.get('chat_template')
    require(served_template_matches(template, family), f'{family} peer template is not the hf2q family template')
    modalities = props.get('modalities')
    require(isinstance(modalities, dict) and modalities.get('vision') is False,
            'peer must serve the text-only workload without a projector')
    params = props.get('default_generation_settings', {}).get('params')
    require(isinstance(params, dict) and all(_close(params.get(k), v) for k, v in SAMPLING[family].items())
            and params.get('speculative.types') == 'none', f'{family} peer sampling settings differ')
    return {'engine_kind': 'peer', 'family': family, **binding,
            'chat_template_sha256': TEMPLATE_SHA256[family], 'logical_slots': SLOTS,
            'context_per_slot': CONTEXT_PER_SLOT, 'sampling': dict(SAMPLING[family]),
            'physical_cohort': 'Unavailable:not proved by readiness or HTTP overlap',
            'pool_profile': 'NotApplicable:HF-only', 'full_prefix_reuse': 'PendingPostOracle'}


def readiness(base, model, output, family, packet=None):
    reference = (packet or {}).get('peer_reference')
    supervised = _supervised_launch(packet, base) if reference is not None else None
    require(reference is not None or 'peer_supervised_launch' not in (packet or {}),
            'peer supervised identity requires a frozen reference')
    return validate_readiness(*qwen_n4_peer.fetch_readiness(base, output), model, family, reference, supervised)
