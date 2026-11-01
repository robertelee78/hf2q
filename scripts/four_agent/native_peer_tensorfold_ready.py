"""TensorFold arm identity and readiness, frozen with the client (native_peer_tensorfold.CLIENT_MODULES).

Binds the binary sha256, the pinned source commit and version, the local HF snapshots, the server argv,
/health, /v1/models, /v1/benchmark/manifest and the startup log. After each agent's task it proves every
turn's rendered prompt with /tokenize. Source (/opt/TensorFold 1157ea5b, zig/src/):
- /health (server/status_routes.zig:14-65): status, model, model_ids, max_batch_size (the fixed --parallel
  lanes, native/qwen27_served.zig:336-339,376), policy, warming, memory{active,cache,peak,limit}, live,
  version (build.zig:74-79: -Dversion or build.zig.zon; no commit is embedded), backend, device.
- /v1/models (status_routes.zig:75-125): one entry per served id; context_length = max_model_len = --context
  (qwen27_served.zig:302,376); reasoning {default_enabled, default_effort only with --reasoning-effort,
  supported_efforts = the template's quoted levels plus none (server/fields.zig:270-285, server.zig:202-206)}.
- /v1/benchmark/manifest (status_routes.zig:68-73, server/benchmark_manifest.zig:190-244): model.revision is
  the snapshots/<sha> directory name (TF_CHECKPOINT_REVISION must agree or it is dropped), config_sha256 is
  sha256(config.json), tokenizer_sha256 hashes the tokenizer files present, each framed "name\\0size\\0" then
  its bytes, chat_template.jinja included (server/benchmark_model.zig:14,160-189).
- /tokenize (server/tokens.zig:26-68): the ids the chat route renders (prompt.zig:32-61,109-128) for the same
  messages, tools and thinking fields; `count` equals the turn's usage.prompt_tokens (chat.zig:568).
- startup: '[tensorfold] ' lines on stdout (server/log.zig:38-49) and std.log lines on stderr. The serving line
  (server/serve.zig:242-250; drafts text server/draft_line.zig:5-29); 'Qwen3.8-27B: N streams of C tokens'
  with ', DFlash2 drafts for one of them' iff a drafter loaded (qwen27_served.zig:407; the wording is wrong,
  every live stream drafts, qwen27_drafts.zig:26-80); the prompt rows and startup reserve
  (qwen27_served.zig:345); the loop policy (:383); the prompt-cache budget (native/cache_fit.zig:99,113).
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import native_peer_tensorfold as tf
from family_peer import same
from qwen38_n1_verdict import loads, require

HEX64 = re.compile(r'[0-9a-f]{64}')
DOC_LIMIT = 1024 * 1024
TOKENIZE_LIMIT = 16 * 1024 * 1024
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json',
                   'vocab.json', 'merges.txt', 'chat_template.jinja')  # benchmark_model.zig:14
TOKENIZE_FIELDS = ('messages', 'tools', 'tool_choice', 'reasoning_effort', 'chat_template_kwargs')
EFFORTS = ['xhigh', 'medium', 'low', 'none']   # the Qwen3.8 template's quoted levels, highest first, plus none
LOG = r'^(?:\w+(?:\(\w+\))?: )?'                # std.log's "info: " prefix
SERVING = re.compile(r'^\[tensorfold\] serving (\S+) at http://([0-9.]+):(\d+)/v1 \(sampling: ([^;]*); '
                     r'drafts: ([^;]*); context: (\d+|unlimited); loaded in ([0-9.]+)s\)$', re.M)
STREAMS = re.compile(LOG + r'Qwen3\.8-27B: (\d+) streams of (\d+) tokens(, DFlash2 drafts for one of them)?'
                     r'(, images read)?$', re.M)
CAPACITY = re.compile(LOG + r'qwen27: prompt capacity=(\d+) rows, startup stream and scratch reserve=(\d+) bytes$',
                      re.M)
POLICY = re.compile(LOG + r'qwen27: (?!prompt capacity=)(.+)$', re.M)
CACHE = re.compile(LOG + r'prompt cache: ([0-9.]+) GiB (.*)$', re.M)
FORBIDDEN_LOG = ('idle unload on', 'cortheon on', '--thinking-reserve is not applied', 'has no API key',
                 'past the cap by --prompt-cache-over-cap', 'is not supported by this native engine')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_sha256(directory):
    """TensorFold's tokenizer_sha256 recomputed (benchmark_model.zig:160-189); None when no file or past 256 MiB."""
    digest, total, found = hashlib.sha256(), 0, 0
    for name in TOKENIZER_FILES:
        path = Path(directory) / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        total += len(data)
        if total > 256 << 20:
            return None
        digest.update(f'{name}\0{len(data)}\0'.encode())
        digest.update(data)
        found += 1
    return digest.hexdigest() if found else None


def snapshot_layout(directory, repo, revision):
    """<hub>/models--OWNER--NAME/snapshots/<revision>: the layout whose name the manifest attests."""
    path = Path(directory)
    return (path.name == revision and path.parent.name == 'snapshots'
            and path.parent.parent.name == 'models--' + repo.replace('/', '--'))


def checkpoint(directory):
    """Local snapshot facts: config/tokenizer/template digests and every file's size and cache blob name."""
    path = Path(directory)
    require(path.is_dir(), f'snapshot {directory} missing')
    files = {}
    for entry in sorted(path.iterdir()):
        blob = Path(os.readlink(entry)).name if entry.is_symlink() else None
        files[entry.name] = {'dir': True} if entry.is_dir() else {'bytes': entry.stat().st_size, 'blob': blob}
    weights = sorted(name for name in files if name.endswith('.safetensors'))
    require(weights, f'snapshot {directory} has no safetensors weights')
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        mapped = loads(index.read_bytes()).get('weight_map')
        require(isinstance(mapped, dict) and sorted(set(mapped.values())) == weights,
                'snapshot shards differ from model.safetensors.index.json')
    config, template = path / 'config.json', path / 'chat_template.jinja'
    require(config.is_file() and config.stat().st_size <= 16 << 20, 'snapshot config.json (16 MiB limit)')
    return {'path': str(path), 'revision': path.name, 'config_sha256': sha256(config),
            'tokenizer_sha256': tokenizer_sha256(path),
            'chat_template_sha256': sha256(template) if template.is_file() else None, 'weights': weights,
            'files': files, 'files_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'weights_sha256': 'Unavailable:shard bytes are not re-hashed; files[*].blob records each HF cache blob '
                              'name (the LFS sha256)'}


def native_spec(spec):
    """The approval's closed `native_tensorfold` block (q5 case client.approval.set)."""
    keys = {'binary', 'binary_sha256', 'commit', 'version', 'target_dir', 'target_revision', 'target_config_sha256',
            'drafter', 'port', 'server_stdout', 'server_stderr'}
    require(isinstance(spec, dict) and spec.get('drafter') in tf.CONTROLS
            and set(spec) == keys | ({'drafter_dir', 'drafter_revision'} if spec['drafter'] == 'dflash2' else set())
            and all(isinstance(v, str) for v in spec.values()), 'closed native_tensorfold approval block')
    dflash2 = spec['drafter'] == 'dflash2'
    names = ('binary', 'target_dir', 'server_stdout', 'server_stderr') + (('drafter_dir',) if dflash2 else ())
    paths = {k: Path(spec[k]) for k in names}
    require(all(p.is_absolute() for p in paths.values()), 'native_tensorfold paths must be absolute')
    require(paths['binary'].name in tf.PROCESSES, 'TF_BIN must be the tensorfold or tensorfold-native executable')
    require(spec['commit'] == tf.TF_COMMIT, 'TensorFold source commit is not the inspected 1157ea5b')
    require(spec['version'] == tf.TF_VERSION, 'TensorFold version is not the inspected 1.2.0')
    require(HEX64.fullmatch(spec['binary_sha256']) and HEX64.fullmatch(spec['target_config_sha256']),
            'TF_BIN_SHA256 and TARGET_CONFIG_SHA256 must be 64 lowercase hex')
    require(spec['target_revision'] == tf.TARGET_REVISION
            and snapshot_layout(paths['target_dir'], tf.TARGET_REPO, tf.TARGET_REVISION),
            f'TARGET_DIR must be the HF cache snapshot of {tf.TARGET_REPO}@{tf.TARGET_REVISION}')
    require(not dflash2 or (spec['drafter_revision'] == tf.DRAFTER_REVISION and snapshot_layout(
        paths['drafter_dir'], tf.DRAFTER_REPO, tf.DRAFTER_REVISION)),
            f'DRAFTER_DIR must be the HF cache snapshot of {tf.DRAFTER_REPO}@{tf.DRAFTER_REVISION}')
    require(spec['port'].isdecimal(), 'native_tensorfold port')
    require(paths['server_stdout'].name == 'server.stdout' and paths['server_stderr'].name == 'server.stderr'
            and paths['server_stdout'].parent == paths['server_stderr'].parent, 'TensorFold server logs must be the run\'s')
    return {**spec, 'port': int(spec['port'])}


def server_argv(spec):
    """The one approved launch (both q5 cases); the drafter case adds only --drafter DIR --drafter-bits 4."""
    argv = [spec['binary'], 'serve', spec['target_dir'], '--name', tf.MODEL_ID, '--host', '127.0.0.1',
            '--port', str(spec['port']), '--context', str(tf.CONTEXT_PER_SLOT), '--parallel', str(tf.SLOTS),
            '--temperature', '0', '--max-tokens', str(tf.MAX_TOKENS), '--keep-warm', '900', '--no-update-check']
    if spec['drafter'] == 'dflash2':
        argv += ['--drafter', spec['drafter_dir'], '--drafter-bits', str(tf.DRAFTER_BITS)]
    return argv


def observed_argv(pid):
    result = subprocess.run(['/bin/ps', '-ww', '-o', 'args=', '-p', str(pid)], capture_output=True, timeout=2, check=True)
    return result.stdout.decode('utf-8', 'strict').strip()


def verify_identity(spec):
    require(sha256(spec['binary']) == spec['binary_sha256'], 'TensorFold binary sha256 differs from TF_BIN_SHA256')
    target = checkpoint(spec['target_dir'])
    require(target['config_sha256'] == spec['target_config_sha256'],
            'target config.json sha256 differs from TARGET_CONFIG_SHA256')
    require(target['chat_template_sha256'] == tf.TEMPLATE_SHA256,
            'target chat_template.jinja is not the hf2q Qwen3.8 template bytes')
    require(target['tokenizer_sha256'] is not None, 'target snapshot has no tokenizer files')
    return {'source_revision': tf.TF_COMMIT, 'version': tf.TF_VERSION, 'binary': spec['binary'],
            'binary_sha256': spec['binary_sha256'],
            'binary_provenance': 'operator-declared TF_BIN_SHA256 of a 1157ea5b build (the binary embeds only its '
                                 'version, build.zig:74-79); not reproducible-build verified. Its Metal kernels are '
                                 'embedded metallibs (zig/build/metal_targets.zig:60-65), so the sha256 covers them',
            'target': target, 'drafter': checkpoint(spec['drafter_dir']) if spec['drafter'] == 'dflash2' else None}


def validate_health(doc, ready=True):
    keys = {'status', 'model', 'model_ids', 'max_batch_size', 'warming', 'memory', 'live', 'version', 'backend'}
    require(isinstance(doc, dict) and keys <= set(doc) <= keys | {'policy', 'device'},
            'TensorFold /health envelope (an API key would hide it)')
    require(doc['status'] == 'ok' and doc['model'] == tf.MODEL_ID and doc['model_ids'] == [tf.MODEL_ID],
            'TensorFold served model identity')
    require(type(doc['max_batch_size']) is int and doc['max_batch_size'] == tf.SLOTS,
            'TensorFold must serve four lanes (--parallel 4)')
    require(doc['version'] == tf.TF_VERSION and doc['backend'] == 'metal', 'TensorFold version/backend')
    require(not ready or doc['warming'] is False, 'TensorFold still warming at readiness')
    memory = doc['memory']
    require(isinstance(memory, dict) and set(memory) <= {'active', 'cache', 'peak', 'limit'}
            and all(type(v) is int and v >= 0 for v in memory.values()), 'TensorFold /health memory')
    return {'memory_bytes': memory, 'policy': doc.get('policy'), 'device': doc.get('device'), 'live': doc['live']}


def validate_models(doc):
    require(isinstance(doc, dict) and set(doc) == {'object', 'data'} and doc['object'] == 'list'
            and isinstance(doc['data'], list) and len(doc['data']) == 1, 'TensorFold /v1/models envelope')
    entry = doc['data'][0]
    require(isinstance(entry, dict) and set(entry) == {'id', 'object', 'created', 'owned_by', 'context_length',
                                                       'max_model_len', 'reasoning'}
            and entry['id'] == tf.MODEL_ID and entry['object'] == 'model' and entry['owned_by'] == 'tensorfold'
            and type(entry['created']) is int, 'TensorFold /v1/models entry')
    require(all(type(entry[k]) is int and entry[k] == tf.CONTEXT_PER_SLOT for k in ('context_length', 'max_model_len')),
            'TensorFold context is not 262144 per request')
    require(same(entry['reasoning'], {'default_enabled': True, 'supported_efforts': EFFORTS}),
            'TensorFold reasoning defaults (thinking on, no server effort, the Qwen3.8 template levels)')
    return entry


def validate_manifest(doc, target):
    require(isinstance(doc, dict) and {'schema_version', 'runtime', 'model', 'hardware'} == set(doc)
            and doc['schema_version'] == 1, 'TensorFold benchmark manifest envelope')
    runtime = doc['runtime']
    require(isinstance(runtime, dict) and {'tensorfold_version', 'backend', 'python_version', 'platform'}
            <= set(runtime) <= {'tensorfold_version', 'backend', 'python_version', 'platform', 'driver'}
            and runtime['tensorfold_version'] == tf.TF_VERSION and runtime['backend'] == 'mlx'
            and runtime['platform'] == 'macos', 'TensorFold manifest runtime')
    require(same(doc['model'], {'revision': tf.TARGET_REVISION, 'config_sha256': target['config_sha256'],
                                'tokenizer_sha256': target['tokenizer_sha256']}),
            'TensorFold manifest revision/config/tokenizer (chat_template.jinja included) differ from the local snapshot')
    return doc


def startup(stdout, stderr, spec):
    """Serving line, streams/context/drafter line, prompt rows and reserve, loop policy and prompt cache."""
    require(not any(text in stdout + stderr for text in FORBIDDEN_LOG), 'TensorFold started with an unapproved mode')
    serving = SERVING.findall(stdout)
    require(len(serving) == 1, 'TensorFold serving line missing or repeated')
    name, host, port, sampling, drafts, context, loaded = serving[0]
    require(name == tf.MODEL_ID and host == '127.0.0.1' and int(port) == spec['port'] and sampling == 'greedy'
            and context == str(tf.CONTEXT_PER_SLOT), 'TensorFold serving line (name, listener, greedy, context)')
    dflash2 = spec['drafter'] == 'dflash2'
    require(drafts.startswith(f"DFlash2 ({spec['drafter_dir']})") if dflash2 else 'DFlash2' not in drafts,
            'TensorFold drafter state differs from the case (serving line)')
    found = {name: pattern.findall(stderr) for name, pattern in (('streams', STREAMS), ('capacity', CAPACITY),
                                                                  ('cache', CACHE))}
    require(all(len(rows) == 1 for rows in found.values()), 'TensorFold startup lines missing or repeated')
    count, tokens, drafted, images = found['streams'][0]
    require(int(count) == tf.SLOTS and int(tokens) == tf.CONTEXT_PER_SLOT and bool(drafted) == dflash2 and not images,
            'TensorFold streams/context/drafter startup line')
    rows, reserve = found['capacity'][0]
    budget, detail = found['cache'][0]
    return {'drafts': drafts, 'loaded_seconds': float(loaded), 'streams': tf.SLOTS, 'stream_tokens': tf.CONTEXT_PER_SLOT,
            'dflash2_loaded': dflash2, 'prompt_rows': int(rows), 'startup_reserve_bytes': int(reserve),
            'loop_policy': POLICY.findall(stderr),
            'prompt_cache': {'status': 'Observed', 'budget_gib': float(budget), 'detail': detail,
                             'enabled': float(budget) > 0, 'flags': 'default auto budget (no --prompt-cache-gib)'}}


def _port(base):
    u = urlsplit(base)
    require(u.scheme == 'http' and u.hostname == '127.0.0.1' and u.port and u.path == '/v1'
            and not u.query and not u.fragment and u.username is None, 'TensorFold loopback base')
    return u.port


def fetch(port, path, output, name, body=None, limit=DOC_LIMIT, timeout=5):
    """GET (or POST `body`) a loopback document; the raw reply is archived before any validation."""
    data = None if body is None else json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
    request = Request(f'http://127.0.0.1:{port}{path}', data=data, method='GET' if body is None else 'POST',
                      headers={} if body is None else {'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(limit + 1)
        Path(output, name).write_bytes(raw)
        require(response.status == 200 and len(raw) <= limit, f'TensorFold {path} status/size')
    return raw


def build_info(commit, binary_sha256, version):
    """Campaign identity string (q3 drift check); None while an input is unresolved."""
    if not all(isinstance(x, str) for x in (commit, binary_sha256, version)) or not (
            re.fullmatch(r'[0-9a-f]{40}', commit) and HEX64.fullmatch(binary_sha256)):
        return None
    return f'tensorfold {version} {commit} binary-sha256 {binary_sha256}'


def readiness(base, model, output, packet):
    """Coordinator readiness: approval binding, binary and snapshots, argv, documents and startup log."""
    require(isinstance(packet, dict), 'TensorFold approval packet')
    spec = native_spec(packet.get('native_tensorfold'))
    variant = tf.controls_variant(packet.get('family_controls'))
    require(variant == spec['drafter'], 'approved family_controls drafter state differs from the launch')
    require(model == tf.MODEL_ID, 'TensorFold model id must be ' + tf.MODEL_ID)
    port = _port(base)
    require(port == spec['port'], 'TensorFold base URL port differs from the launch port')
    identity = verify_identity(spec)
    argv = server_argv(spec)
    require(observed_argv(packet['server_pid']) == ' '.join(argv),
            'TensorFold server argv differs from the approved launch (flags, drafter, prompt cache)')
    health = validate_health(loads(fetch(port, '/health', output, 'tensorfold-health.raw')))
    validate_models(loads(fetch(port, '/v1/models', output, 'tensorfold-models.raw')))
    manifest = validate_manifest(loads(fetch(port, '/v1/benchmark/manifest', output, 'tensorfold-manifest.raw')),
                                 identity['target'])
    logs = {}
    for stream in ('stdout', 'stderr'):
        raw = Path(spec['server_' + stream]).read_bytes()
        Path(output, 'tensorfold-startup.' + stream).write_bytes(raw)
        logs[stream] = raw.decode('utf-8', 'replace')
    facts = startup(logs['stdout'], logs['stderr'], spec)
    return {'engine_kind': tf.ENGINE, 'family': tf.FAMILY, 'label': tf.LABELS[variant], 'variant': variant,
            'source_revision': tf.TF_COMMIT, 'version': tf.TF_VERSION, 'logical_slots': tf.SLOTS,
            'context_per_slot': tf.CONTEXT_PER_SLOT, 'think_mode': f'thinking, reasoning_effort {tf.EFFORT}',
            'sampling': dict(tf.SAMPLING), 'controls': packet['family_controls'], 'identity': identity,
            'server_argv': argv, 'health': health, 'manifest': {'hardware': manifest['hardware'],
                                                                'runtime': manifest['runtime']},
            'startup': facts, 'build_info': build_info(tf.TF_COMMIT, spec['binary_sha256'], tf.TF_VERSION),
            'chat_template': f'Observed: chat_template.jinja sha256 {tf.TEMPLATE_SHA256} is the hf2q Qwen3.8 '
                             'template; every turn\'s rendered ids are proven by /tokenize (native_render)',
            'tokenizer': 'Unavailable:the target tokenizer.json differs from the base Qwen/Qwen3.8-27B one hf2q '
                         'serves; per-turn TensorFold ids are archived for the post-run oracle',
            'physical_cohort': 'Unavailable:not proved by readiness or HTTP overlap',
            'pool_profile': 'NotApplicable:HF-only', 'full_prefix_reuse': 'PendingPostOracle'}


def token_sha256(tokens):
    """TensorFold's token sha (decimal ids joined by commas, server/tokens.zig:105-115)."""
    return hashlib.sha256(','.join(str(t) for t in tokens).encode()).hexdigest()


def render_proof(base, output, turns):
    """Every turn's archived request re-rendered by /tokenize; its count must equal usage.prompt_tokens."""
    port, rows = _port(base), []
    for receipt in turns:
        directory = Path(output) / f"turn-{receipt['turn']}"
        raw = (directory / 'request.json').read_bytes()
        require(hashlib.sha256(raw).hexdigest() == receipt['request_sha256'],
                'TensorFold render proof: request archive differs from its receipt')
        request = loads(raw)
        reply = fetch(port, '/tokenize', directory, 'tensorfold-tokenize.raw',
                      {k: request[k] for k in TOKENIZE_FIELDS}, TOKENIZE_LIMIT, 60)
        doc = loads(reply)
        require(isinstance(doc, dict) and set(doc) == {'count', 'max_model_len', 'tokens'}
                and doc['max_model_len'] == tf.CONTEXT_PER_SLOT and isinstance(doc['tokens'], list)
                and all(type(t) is int and t >= 0 for t in doc['tokens']) and doc['count'] == len(doc['tokens']),
                'TensorFold /tokenize envelope')
        require(doc['count'] == receipt['usage']['prompt_tokens'],
                'TensorFold /tokenize count differs from the turn usage.prompt_tokens (rendered prompt not proven)')
        rows.append({'turn': receipt['turn'], 'request_sha256': receipt['request_sha256'], 'count': doc['count'],
                     'tokens_sha256': token_sha256(doc['tokens']),
                     'response_sha256': hashlib.sha256(reply).hexdigest()})
    return {'status': 'Observed', 'endpoint': '/tokenize', 'fields': list(TOKENIZE_FIELDS), 'turns': rows,
            'timing': 'after task_completed_monotonic_ns: outside the makespan clock',
            'hf2q_token_parity': 'PendingPostOracle: hf2q renders the same template bytes; its GGUF tokenizer '
                                 'is the base one, so ids are compared offline from tensorfold-tokenize.raw'}


def closing(base, output):
    """/health once every agent finished: the D2 memory peak, archived raw."""
    health = validate_health(loads(fetch(_port(base), '/health', output, 'tensorfold-health-final.raw')), ready=False)
    return {'health': health, 'source': '/health memory (status_routes.zig:48-55), peak since start'}


def campaign_identity(plan):
    """Frozen per campaign and re-checked before every run (q3_identity)."""
    v = plan['variables']
    def listed(directory):
        return checkpoint(directory) if isinstance(directory, str) and Path(directory).is_dir() else None
    return {'peer_build_info': build_info(v.get('TF_COMMIT'), v.get('TF_BIN_SHA256'), v.get('TF_VERSION')),
            'native_runtime': {'target': listed(v.get('TARGET_DIR')), 'drafter': listed(v.get('DRAFTER_DIR'))}}


def launch_problems(name, artifact, hf2q_sha256):
    """Q3 refusals: the artifact is the pinned target snapshot's config.json, never the hf2q GGUF."""
    problems = []
    if artifact and not (Path(artifact['path']).name == 'config.json' and snapshot_layout(
            Path(artifact['path']).parent, tf.TARGET_REPO, tf.TARGET_REVISION)):
        problems.append(f'{name}: native artifact must be the {tf.TARGET_REPO}@{tf.TARGET_REVISION[:8]} snapshot '
                        'config.json (set --peer-var TARGET_DIR)')
    if artifact.get('sha256') and artifact['sha256'] == hf2q_sha256:
        problems.append(f"{name}: native artifact sha256 equals the hf2q artifact's (set --peer-var TARGET_CONFIG_SHA256)")
    return problems
