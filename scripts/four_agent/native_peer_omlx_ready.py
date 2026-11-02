"""oMLX arm launch files, identity and readiness, frozen with the client (native_peer_omlx.CLIENT_MODULES).

native_peer_omlx_launch.py (the q5 serve entry) calls prepare_run: it writes the run's own oMLX base directory
(model_settings.json and settings.json), links the pinned HF snapshot as the one served model, records
launch.json and execs `omlx serve` (server_argv). Readiness re-derives all of it and binds the running server.
Source (/opt/omlx 6b413c90, omlx/):
- model settings: {"version":1,"models":{id:{...}}} (model_settings.py:35, 797-863); unknown keys are dropped
  silently (from_dict, 631-648) and a record that fails validation is dropped with "Failed to load settings for
  model" (853-856), so the file is generated here from known fields only and that line refuses. A model with
  saved settings gets no initial/MTP defaults (engine_pool.py:1624-1660).
- settings.json: CLI > env > file (settings.py:1096-1134); serve re-saves file + CLI (cli.py:289-292,
  settings.py:1613-1620). cache.hot_cache_only keeps the paged prefix cache in RAM with no SSD I/O
  (cache/paged_ssd_cache.py:1795-1799, 1948-1949); the hot tier exists only with a paged cache dir
  (cli.py:388-396). --no-hf-cache stops HF-cache discovery (settings.py:1517-1542), so the run's model
  directory (one symlink named by the model id) is the only source; the id is the directory name
  (model_discovery.py:1972-2096).
- `omlx serve` sets the process title to omlx-server (cli.py:185, process_title.py:8-27), so ps shows the title,
  not the argv: launch.json (written by the same pid before exec) carries the argv.
- /health (server.py:3348-3395) is 503 "loading" until the pinned preload ends; /api/status (3398-3479):
  version, custom_kernels (custom_kernels/__init__.py:16-41), session counters (server_metrics.py:265-285);
  /v1/models/status (3937-3984, engine_pool.py:4713-4775); POST /v1/tokenize (8380-8434) renders messages,
  tools and chat_template_kwargs through the chat path (8327-8377): the ids generation sees.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import native_peer_omlx as om
from family_peer import same
from qwen38_n1_verdict import loads, require

HEX64, HEX40 = re.compile(r'[0-9a-f]{64}'), re.compile(r'[0-9a-f]{40}')
DOC_LIMIT, TOKENIZE_LIMIT = 1024 * 1024, 16 * 1024 * 1024
TOKENIZE_FIELDS = ('model', 'messages', 'tools', 'chat_template_kwargs')  # the chat request /v1/tokenize builds
HOT_CACHE = '16GB'
TITLE = 'omlx-server'
# Venv pins (pyproject.toml:45-47, 114): name -> (version or None, VCS commit or None).
PINS = {'mlx': ('0.32.4', None), 'mlx-lm': (None, '0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d'),
        'mlx-vlm': (None, '138d7617d39653b39a5cbc379b1b5907a4481d67')}
LAUNCH_KEYS = ('q3_family', 'omlx_dir', 'omlx_py', 'commit', 'version', 'model_dir', 'model_revision',
               'model_config_sha256', 'template_sha256', 'run_dir', 'port')
SPEC_KEYS = frozenset(LAUNCH_KEYS) | {'server_stdout', 'server_stderr'}
SERVE_ENV = {'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
GLOBAL_SETTINGS = {'version': '1.0', 'cache': {'enabled': True, 'hot_cache_only': True}}
STATUS = frozenset({'status', 'version', 'uptime_seconds', 'models_discovered', 'models_loaded', 'models_loading',
                    'default_model', 'loaded_models', 'total_requests', 'active_requests', 'waiting_requests',
                    'total_prompt_tokens', 'total_completion_tokens', 'total_cached_tokens', 'cache_efficiency',
                    'avg_prefill_tps', 'avg_generation_tps', 'model_memory_used', 'model_memory_max',
                    'model_memory_used_formatted', 'model_memory_max_formatted', 'custom_kernels', 'ane_prefill'})
REQUIRED_LOG = ('Loaded settings for 1 models',)
FORBIDDEN_LOG = ('Failed to load settings for model', 'Applied initial settings for', 'Settings file version',
                 'Skipped profile reference repair', 'SSD cache directory unavailable', 'Configuration error',
                 'Failed to save settings', 'Dropping tool call', 'Traceback (most recent call last)')
CLOSING_INPUTS = 'omlx-closing-inputs.json'   # readiness -> closing, in the coordinator output


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def document_bytes(doc):
    return (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode()


def model_settings(q3):
    """The run's model_settings.json: hf2q-matched sampling and thinking, everything speculative or lossy off."""
    artifact = om.ARTIFACTS[q3]
    qwen = artifact['family'] == 'qwen'
    entry = {'max_context_window': om.CONTEXT_PER_SLOT, 'max_tokens': om.MAX_TOKENS, 'temperature': 0.0,
             'top_p': 1.0, 'top_k': 0, 'min_p': 0.0, 'repetition_penalty': 1.05, 'presence_penalty': 0.0,
             'force_sampling': False, 'enable_thinking': qwen, **({'preserve_thinking': True} if qwen else {}),
             'thinking_budget_enabled': False, 'model_type_override': 'llm', 'turboquant_kv_enabled': False,
             'mtp_enabled': False, 'dflash_enabled': False, 'vlm_mtp_enabled': False, 'specprefill_enabled': False,
             'a8_prefill_enabled': False, 'qwen35_ane_prefill_enabled': False, 'moe_expert_offload_enabled': False,
             'activation_fp16_enabled': False, 'guided_grammar_enabled': False, 'trust_remote_code': False,
             'is_pinned': True, 'is_default': True}
    return {'version': 1, 'models': {artifact['model_id']: entry}}


def layout(spec):
    run = Path(spec['run_dir'])
    return {'run': run, 'base': run / 'base', 'models': run / 'models', 'cache': run / 'ssd-cache',
            'link': run / 'models' / om.MODEL_IDS[spec['q3_family']], 'launch': run / 'launch.json'}


def snapshot_layout(directory, repo, revision):
    path = Path(directory)
    return (path.name == revision and path.parent.name == 'snapshots'
            and path.parent.parent.name == 'models--' + repo.replace('/', '--'))


def launch_spec(values):
    """The launch inputs (q5 serve argv): closed, absolute, pinned."""
    require(isinstance(values, dict) and set(values) == set(LAUNCH_KEYS)
            and all(isinstance(v, str) for v in values.values()), 'closed oMLX launch inputs')
    q3 = values['q3_family']
    require(q3 in om.ARTIFACTS, 'oMLX q3_family must be qwen38, qwen36 or gemma4')
    artifact = om.ARTIFACTS[q3]
    require(all(Path(values[k]).is_absolute() for k in ('omlx_dir', 'omlx_py', 'model_dir', 'run_dir')),
            'oMLX paths must be absolute')
    require(values['commit'] == om.OMLX_COMMIT and values['version'] == om.OMLX_VERSION,
            'oMLX source must be the inspected 6b413c90 (0.7.3rc1)')
    require(values['model_revision'] == artifact['revision']
            and snapshot_layout(values['model_dir'], artifact['repo'], artifact['revision']),
            f"MODEL_DIR must be the HF cache snapshot of {artifact['repo']}@{artifact['revision']}")
    require(HEX64.fullmatch(values['model_config_sha256']) and HEX64.fullmatch(values['template_sha256']),
            'MODEL_CONFIG_SHA256 and TEMPLATE_SHA256 must be 64 lowercase hex')
    require(values['port'].isdecimal() and 1024 <= int(values['port']) <= 65535, 'oMLX port')
    return {**values, 'port': int(values['port'])}


def native_spec(spec):
    """The approval's closed `native_omlx` block (q5 case client.approval.set)."""
    require(isinstance(spec, dict) and set(spec) == SPEC_KEYS, 'closed native_omlx approval block')
    launch = launch_spec({k: spec[k] for k in LAUNCH_KEYS})
    out, err = Path(spec['server_stdout']), Path(spec['server_stderr'])
    require(out.name == 'server.stdout' and err.name == 'server.stderr' and out.parent == err.parent
            == Path(launch['run_dir']).parent, "oMLX server logs must be the run's")
    return {**launch, 'server_stdout': str(out), 'server_stderr': str(err)}


def server_argv(spec):
    """The one approved `omlx serve` (cli.py:1169-1366); -I ignores PYTHON* env, -u keeps the startup prints."""
    paths = layout(spec)
    return [spec['omlx_py'], '-I', '-u', '-m', 'omlx.cli', 'serve', '--model-dir', str(paths['models']),
            '--base-path', str(paths['base']), '--host', '127.0.0.1', '--port', str(spec['port']), '--headless',
            '--no-hf-cache', '--max-concurrent-requests', str(om.SLOTS), '--paged-ssd-cache-dir', str(paths['cache']),
            '--hot-cache-max-size', HOT_CACHE, '--sse-keepalive-mode', 'off', '--memory-guard', 'aggressive',
            '--log-level', 'info']


def source_identity(spec):
    """Detached HEAD at the pinned commit, no tracked-file edits, the pinned version file."""
    root = Path(spec['omlx_dir'])
    head = (root / '.git' / 'HEAD').read_text().strip() if (root / '.git' / 'HEAD').is_file() else None
    require(head == om.OMLX_COMMIT, 'OMLX_DIR is not a detached checkout of 6b413c90')
    dirty = subprocess.run(['/usr/bin/git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, timeout=30, check=True).stdout.decode()
    require(dirty == '', 'OMLX_DIR has tracked-file changes')
    version = (root / 'omlx' / '_version.py').read_text()
    require(version.strip() == f'__version__ = "{om.OMLX_VERSION}"', 'omlx/_version.py is not 0.7.3rc1')
    return {'commit': head, 'version': om.OMLX_VERSION, 'tracked_changes': 'none (git status --untracked-files=no)'}


def venv_inventory(python, omlx_dir):
    """pip-freeze-equivalent inventory from the venv's dist-info (what `pip freeze` reads), and its sha256."""
    venv = Path(python).parent.parent
    require((venv / 'pyvenv.cfg').is_file(), 'OMLX_PY must be a venv interpreter')
    rows = {}
    for info in sorted(venv.glob('lib/python3.*/site-packages/*.dist-info')):
        meta = dict(line.split(': ', 1) for line in (info / 'METADATA').read_text(errors='replace').splitlines()
                    if line.startswith(('Name: ', 'Version: ')))
        name = meta.get('Name', '').lower().replace('_', '-')
        url = json.loads((info / 'direct_url.json').read_text()) if (info / 'direct_url.json').is_file() else {}
        commit = (url.get('vcs_info') or {}).get('commit_id')
        editable = (url.get('dir_info') or {}).get('editable') is True
        require(name and name not in rows, f'venv dist-info {info.name} unreadable or duplicated')
        rows[name] = {'version': meta.get('Version'), 'commit': commit, 'url': url.get('url'), 'editable': editable}
    for name, (version, commit) in PINS.items():
        row = rows.get(name) or {}
        require((version is None or row.get('version') == version) and (commit is None or row.get('commit') == commit),
                f'venv {name} is not the oMLX pin')
    own = rows.get('omlx') or {}
    require(own.get('editable') is True and own.get('url') == Path(omlx_dir).resolve().as_uri(),
            'venv omlx must be the editable install of OMLX_DIR')
    text = ''.join(f"{n}=={r['version']}" + (f" @{r['commit']}" if r['commit'] else '')
                   + (f" editable {r['url']}" if r['editable'] else '') + '\n' for n, r in sorted(rows.items()))
    return {'venv': str(venv), 'packages': len(rows), 'freeze_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'pins': {n: rows[n] for n in sorted(set(PINS) | {'omlx', 'transformers'}) if n in rows}}


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
    digests = {name: sha256(path / name) if (path / name).is_file() else None
               for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                            'generation_config.json', 'special_tokens_map.json')}
    require(digests['config.json'] and digests['tokenizer.json'] and digests['chat_template.jinja'],
            'snapshot lacks config.json, tokenizer.json or chat_template.jinja')
    return {'path': str(path), 'revision': path.name, 'sha256': digests, 'weights': weights, 'files': files,
            'files_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            'weights_sha256': 'Unavailable:shard bytes are not re-hashed; files[*].blob records each HF cache blob '
                              'name (the LFS sha256)'}


def verify_identity(spec):
    target = checkpoint(spec['model_dir'])
    require(target['sha256']['config.json'] == spec['model_config_sha256'],
            'snapshot config.json sha256 differs from MODEL_CONFIG_SHA256')
    require(target['sha256']['chat_template.jinja'] == spec['template_sha256'],
            'snapshot chat_template.jinja sha256 differs from TEMPLATE_SHA256')
    hf2q = om.ARTIFACTS[spec['q3_family']]['hf2q_template_sha256']
    return {'source': source_identity(spec), 'venv': venv_inventory(spec['omlx_py'], spec['omlx_dir']), 'model': target,
            'chat_template': {'sha256': spec['template_sha256'], 'hf2q_reviewed_sha256': hf2q,
                              'identical_to_hf2q': spec['template_sha256'] == hf2q}}


def prepare_run(spec):
    """Launcher side: the run's base directory, settings files and model link; returns the exec argv."""
    paths = layout(spec)
    paths['run'].mkdir()
    for name in ('base', 'models', 'cache'):
        paths[name].mkdir()
    (paths['base'] / 'model_settings.json').write_bytes(document_bytes(model_settings(spec['q3_family'])))
    (paths['base'] / 'settings.json').write_bytes(document_bytes(GLOBAL_SETTINGS))
    os.symlink(spec['model_dir'], paths['link'])
    return server_argv(spec)


def launch_record(spec, identity, environ):
    return {'schema': 'hf2q.omlx-launch.v1', 'pid': os.getpid(), 'argv': server_argv(spec),
            'env': {k: environ[k] for k in sorted(environ)}, 'spec': spec, 'identity': identity,
            'model_settings_sha256': hashlib.sha256(document_bytes(model_settings(spec['q3_family']))).hexdigest(),
            'initial_settings_sha256': hashlib.sha256(document_bytes(GLOBAL_SETTINGS)).hexdigest()}


def check_environ(environ):
    require(not any(k.startswith(('OMLX_', 'PYTHON', 'MLX_')) for k in environ),
            'no OMLX_/PYTHON/MLX_ variable may reach the oMLX server (settings.py:1270-1340 env overrides)')
    require(all(environ.get(k) == v for k, v in SERVE_ENV.items()), 'oMLX serve env must be offline (HF_HUB_OFFLINE=1)')


def verify_launch(spec, packet):
    """launch.json bound by the approval, its pid the server's, the settings files exactly as generated."""
    paths = layout(spec)
    require(sha256(paths['launch']) == packet.get('native_omlx_launch_sha256'), 'oMLX launch.json differs from the approval')
    record = loads(paths['launch'].read_bytes())
    require(isinstance(record, dict) and record.get('pid') == packet.get('server_pid')
            and record.get('argv') == server_argv(spec) and record.get('spec') == {k: spec[k] for k in LAUNCH_KEYS},
            'oMLX launch record pid/argv/spec differs from the approved launch')
    check_environ(record.get('env') or {})
    expected = hashlib.sha256(document_bytes(model_settings(spec['q3_family']))).hexdigest()
    require(sha256(paths['base'] / 'model_settings.json') == expected == packet.get('native_omlx_settings_sha256'),
            'oMLX model_settings.json differs from the generated settings')
    require(os.readlink(paths['link']) == spec['model_dir'] and sorted(os.listdir(paths['models']))
            == [paths['link'].name], 'oMLX model directory must hold exactly the pinned snapshot link')
    return record


def process_title(pid):
    result = subprocess.run(['/bin/ps', '-ww', '-o', 'args=', '-p', str(pid)], capture_output=True, timeout=2, check=True)
    return result.stdout.decode('utf-8', 'strict').strip()


def validate_health(doc, model_id):
    require(isinstance(doc, dict) and set(doc) == {'status', 'default_model', 'engine_pool', 'mcp'}
            and doc['status'] == 'healthy' and doc['default_model'] == model_id and doc['mcp'] is None,
            'oMLX /health (healthy, default model, no MCP)')
    pool = doc['engine_pool']
    require(isinstance(pool, dict) and set(pool) == {'model_count', 'loaded_count', 'final_ceiling',
                                                     'current_model_memory'}
            and pool['model_count'] == pool['loaded_count'] == 1, 'oMLX /health engine pool (one loaded model)')
    return pool


def validate_status(doc, model_id, requests=0):
    require(isinstance(doc, dict) and set(doc) == STATUS and doc['status'] == 'ok'
            and doc['version'] == om.OMLX_VERSION, 'oMLX /api/status envelope/version')
    require(doc['models_discovered'] == doc['models_loaded'] == 1 and doc['models_loading'] == 0
            and doc['loaded_models'] == [model_id] and doc['default_model'] == model_id, 'oMLX /api/status models')
    kernels = doc['custom_kernels']
    require(isinstance(kernels, dict) and kernels and all(v == {'available': True, 'import_error': None}
                                                          for v in kernels.values()),
            'oMLX native custom kernels unavailable (generic fallback paths)')
    require(doc['active_requests'] == doc['waiting_requests'] == 0 and doc['total_requests'] == requests,
            f'oMLX served {doc["total_requests"]} chat requests, expected {requests}')
    return {k: doc[k] for k in ('custom_kernels', 'ane_prefill', 'model_memory_used', 'model_memory_max',
                                'total_requests', 'total_prompt_tokens', 'total_completion_tokens',
                                'total_cached_tokens', 'uptime_seconds')}


def validate_models_status(doc, spec):
    require(isinstance(doc, dict) and isinstance(doc.get('models'), list), 'oMLX /v1/models/status envelope')
    model_id = om.MODEL_IDS[spec['q3_family']]
    served = [m for m in doc['models'] if not (isinstance(m, dict) and m.get('engine_type') == 'markitdown'
                                               and m.get('model_path') == 'builtin://markitdown')]
    require(len(served) == 1 and isinstance(served[0], dict) and served[0].get('id') == model_id,
            'oMLX must serve exactly the pinned model')
    entry = served[0]
    expected = {'loaded': True, 'is_loading': False, 'pinned': True, 'engine_type': 'batched', 'model_type': 'llm',
                'config_model_type': om.ARTIFACTS[spec['q3_family']]['config_model_type'], 'external_engine': None,
                'distributed': False, 'is_helper': False, 'max_context_window': om.CONTEXT_PER_SLOT,
                'max_tokens': om.MAX_TOKENS}
    require(all(k in entry and same(entry[k], v) for k, v in expected.items())
            and isinstance(entry.get('model_path'), str)
            and Path(entry['model_path']).resolve() == Path(spec['model_dir']).resolve(),
            'oMLX served model state (path, text engine, pinned, context, max tokens)')
    return {k: entry.get(k) for k in ('model_path', 'model_context_length', 'thinking_default',
                                      'preserve_thinking_default', 'reasoning_efforts', 'source_type',
                                      'source_repo_id', 'actual_size', 'estimated_size')}


def validate_settings(spec):
    """settings.json as serve re-saved it (file + CLI): the global controls in force."""
    doc = loads((layout(spec)['base'] / 'settings.json').read_bytes())
    want = {('server', 'host'): '127.0.0.1', ('server', 'port'): spec['port'], ('server', 'sse_keepalive_mode'): 'off',
            ('server', 'log_level'): 'info', ('scheduler', 'max_concurrent_requests'): om.SLOTS,
            ('cache', 'enabled'): True, ('cache', 'hot_cache_only'): True,
            ('cache', 'ssd_cache_dir'): str(layout(spec)['cache']), ('cache', 'hot_cache_max_size'): HOT_CACHE,
            ('cache', 'hot_cache_write_through'): False, ('memory', 'memory_guard_tier'): 'aggressive',
            ('memory', 'prefill_memory_guard'): True, ('huggingface', 'hf_cache_enabled'): False,
            ('idle_timeout', 'idle_timeout_seconds'): None}
    bad = [f'{a}.{b}' for (a, b), v in want.items() if not (isinstance(doc.get(a), dict) and b in doc[a]
                                                           and same(doc[a][b], v))]
    require(not bad, 'oMLX settings.json controls differ: ' + ','.join(bad))
    return {'sha256': sha256(layout(spec)['base'] / 'settings.json'), 'scheduler': doc['scheduler'],
            'burst_decode_mode': doc['server'].get('burst_decode_mode'), 'sampling_defaults': doc.get('sampling')}


def startup(stdout, stderr, spec):
    text = stdout + stderr
    require(not any(line in text for line in FORBIDDEN_LOG), 'oMLX started with a dropped/unapproved setting')
    model_id = om.MODEL_IDS[spec['q3_family']]
    lines = (f'Version: {om.OMLX_VERSION}', f"Base path: {Path(layout(spec)['base']).resolve()}",
             'Memory guard: on (tier: aggressive)', f'Hot cache: {HOT_CACHE[:-2]}.0GB (in-memory)',
             'Saved CLI arguments to settings.json',
             f'Applied model_type override for {model_id}: type=llm, engine=batched') + REQUIRED_LOG
    missing = [line for line in lines if line not in text]
    require(not missing, 'oMLX startup lines missing: ' + ' | '.join(missing))
    return {'required_lines': list(lines), 'prefix_cache': {'status': 'Observed', 'tier': 'hot-only (RAM)',
                                                            'hot_cache_max_size': HOT_CACHE, 'ssd_io': False}}


def _port(base):
    u = urlsplit(base)
    require(u.scheme == 'http' and u.hostname == '127.0.0.1' and u.port and u.path == '/v1'
            and not u.query and not u.fragment and u.username is None, 'oMLX loopback base')
    return u.port


def fetch(port, path, output, name, body=None, limit=DOC_LIMIT, timeout=5):
    """GET (or POST `body`) a loopback document; the raw reply is archived before any validation."""
    data = None if body is None else json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
    request = Request(f'http://127.0.0.1:{port}{path}', data=data, method='GET' if body is None else 'POST',
                      headers={} if body is None else {'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(limit + 1)
        Path(output, name).write_bytes(raw)
        require(response.status == 200 and len(raw) <= limit, f'oMLX {path} status/size')
    return raw


def build_info(commit, version, freeze_sha256):
    """Campaign identity string (q3 drift check); None while an input is unresolved."""
    if not all(isinstance(x, str) for x in (commit, version, freeze_sha256)) or not (
            HEX40.fullmatch(commit) and HEX64.fullmatch(freeze_sha256)):
        return None
    return f'omlx {version} {commit} venv-freeze-sha256 {freeze_sha256}'


def readiness(base, model, output, packet):
    """Coordinator readiness: approval binding, source/venv/snapshot, launch record, documents and startup log."""
    require(isinstance(packet, dict), 'oMLX approval packet')
    spec = native_spec(packet.get('native_omlx'))
    q3 = om.controls_family(packet.get('family_controls'))
    require(q3 == spec['q3_family'] and packet.get('family', 'qwen') == om.ARTIFACTS[q3]['family'],
            'approved family_controls are for another oMLX family')
    model_id = om.MODEL_IDS[q3]
    require(model == model_id, 'oMLX model id must be ' + model_id)
    port = _port(base)
    require(port == spec['port'], 'oMLX base URL port differs from the launch port')
    identity = verify_identity(spec)
    launch = verify_launch(spec, packet)
    title = process_title(packet['server_pid'])
    require(title == TITLE, f'oMLX server process title {title!r} is not {TITLE} (setproctitle)')
    health = validate_health(loads(fetch(port, '/health', output, 'omlx-health.raw')), model_id)
    status = validate_status(loads(fetch(port, '/api/status', output, 'omlx-status.raw')), model_id)
    served = validate_models_status(loads(fetch(port, '/v1/models/status', output, 'omlx-models-status.raw')), spec)
    settings = validate_settings(spec)
    logs = {}
    for stream in ('stdout', 'stderr'):
        raw = Path(spec['server_' + stream]).read_bytes()
        Path(output, 'omlx-startup.' + stream).write_bytes(raw)
        logs[stream] = raw.decode('utf-8', 'replace')
    facts = startup(logs['stdout'], logs['stderr'], spec)
    Path(output, CLOSING_INPUTS).write_bytes(document_bytes(
        {'spec': spec, 'model_settings_sha256': packet['native_omlx_settings_sha256'], 'settings': settings}))
    return {'engine_kind': om.ENGINE, 'family': om.ARTIFACTS[q3]['family'], 'q3_family': q3, 'label': om.LABELS[q3],
            'source_revision': om.OMLX_COMMIT, 'version': om.OMLX_VERSION, 'logical_slots': om.SLOTS,
            'context_per_slot': om.CONTEXT_PER_SLOT,
            'think_mode': 'thinking (preserve_thinking, budget 2048/512)' if om.ARTIFACTS[q3]['family'] == 'qwen'
            else 'no thinking', 'sampling': dict(om.SAMPLING[om.ARTIFACTS[q3]['family']]),
            'repetition_history': om.REPETITION_PARITY, 'controls': packet['family_controls'],
            'identity': identity, 'server_argv': launch['argv'], 'process_title': title, 'health': health,
            'status': status, 'served_model': served, 'settings': {**settings, 'model_settings_sha256':
                                                                    packet['native_omlx_settings_sha256']},
            'startup': facts, 'build_info': build_info(om.OMLX_COMMIT, om.OMLX_VERSION, identity['venv']['freeze_sha256']),
            'chat_template': identity['chat_template'],
            'tokenizer': 'Unavailable:the mlx-community tokenizer is not the GGUF-embedded one hf2q serves; per-turn '
                         'oMLX ids are archived for the post-run oracle',
            'physical_cohort': 'Unavailable:not proved by readiness or HTTP overlap',
            'pool_profile': 'NotApplicable:HF-only', 'full_prefix_reuse': 'PendingPostOracle'}


def token_sha256(tokens):
    return hashlib.sha256(','.join(str(t) for t in tokens).encode()).hexdigest()


def render_proof(base, output, turns):
    """Every turn's archived request re-rendered by /v1/tokenize; its count must equal usage.prompt_tokens."""
    port, rows = _port(base), []
    for receipt in turns:
        directory = Path(output) / f"turn-{receipt['turn']}"
        raw = (directory / 'request.json').read_bytes()
        require(hashlib.sha256(raw).hexdigest() == receipt['request_sha256'],
                'oMLX render proof: request archive differs from its receipt')
        request = loads(raw)
        reply = fetch(port, '/v1/tokenize', directory, 'omlx-tokenize.raw',
                      {k: request[k] for k in TOKENIZE_FIELDS}, TOKENIZE_LIMIT, 60)
        doc = loads(reply)
        require(isinstance(doc, dict) and set(doc) == {'count', 'max_model_len', 'tokens', 'token_strs'}
                and doc['token_strs'] is None and doc['max_model_len'] == om.CONTEXT_PER_SLOT
                and isinstance(doc['tokens'], list) and all(type(t) is int and t >= 0 for t in doc['tokens'])
                and doc['count'] == len(doc['tokens']), 'oMLX /v1/tokenize envelope')
        require(doc['count'] == receipt['usage']['prompt_tokens'],
                'oMLX /v1/tokenize count differs from the turn usage.prompt_tokens (rendered prompt not proven)')
        rows.append({'turn': receipt['turn'], 'request_sha256': receipt['request_sha256'], 'count': doc['count'],
                     'tokens_sha256': token_sha256(doc['tokens']), 'response_sha256': hashlib.sha256(reply).hexdigest()})
    return {'status': 'Observed', 'endpoint': '/v1/tokenize', 'fields': list(TOKENIZE_FIELDS), 'turns': rows,
            'timing': 'coordinator closing, after every agent finished: outside the makespan clock',
            'hf2q_token_parity': 'PendingPostOracle: ids are compared offline from omlx-tokenize.raw'}


def closing(base, output):
    """Once every agent finished: /v1/tokenize proofs, then /api/status (no hidden request) and the settings files."""
    render, turns = {}, []
    for summary in sorted(Path(output).glob('agent-*/summary.json')):
        receipts = loads(summary.read_bytes())['turns']
        render[summary.parent.name] = render_proof(base, summary.parent, receipts)
        turns += receipts
    inputs = loads(Path(output, CLOSING_INPUTS).read_bytes())
    spec = native_spec({**inputs['spec'], 'port': str(inputs['spec']['port'])})
    model_id = om.MODEL_IDS[spec['q3_family']]
    status = validate_status(loads(fetch(_port(base), '/api/status', output, 'omlx-status-final.raw')), model_id,
                             len(turns))
    require(status['total_prompt_tokens'] == sum(t['usage']['prompt_tokens'] for t in turns)
            and status['total_completion_tokens'] == sum(t['usage']['completion_tokens'] for t in turns),
            'oMLX session token counters differ from the client receipts')
    require(sha256(layout(spec)['base'] / 'model_settings.json') == inputs['model_settings_sha256'],
            'oMLX model_settings.json changed during the run')
    final = validate_settings(spec)
    require(final == inputs['settings'], 'oMLX settings.json changed during the run')
    return {'render': render, 'status': status, 'settings': final,
            'source': '/api/status session counters (server_metrics.py:265-285) equal the client receipts'}


def campaign_identity(plan):
    """Frozen per campaign and re-checked before every run (q3_identity)."""
    v = plan['variables']
    venv = model = None
    if isinstance(v.get('OMLX_PY'), str) and Path(v['OMLX_PY']).parent.parent.joinpath('pyvenv.cfg').is_file():
        venv = venv_inventory(v['OMLX_PY'], v.get('OMLX_DIR') or '/')
    if isinstance(v.get('MODEL_DIR'), str) and Path(v['MODEL_DIR']).is_dir():
        model = checkpoint(v['MODEL_DIR'])
    return {'peer_build_info': build_info(v.get('OMLX_COMMIT'), v.get('OMLX_VERSION'), venv and venv['freeze_sha256']),
            'native_runtime': {'venv': venv, 'model': model}}


def launch_problems(name, artifact, hf2q_sha256):
    """Q3 refusals: the artifact is a pinned snapshot's config.json, never the hf2q GGUF."""
    problems = []
    path = Path((artifact or {}).get('path') or '/')
    if artifact and not (path.name == 'config.json' and any(snapshot_layout(path.parent, a['repo'], a['revision'])
                                                            for a in om.ARTIFACTS.values())):
        problems.append(f'{name}: native artifact must be a pinned mlx-community snapshot config.json '
                        '(set --peer-var MODEL_DIR)')
    if (artifact or {}).get('sha256') and artifact['sha256'] == hf2q_sha256:
        problems.append(f"{name}: native artifact sha256 equals the hf2q artifact's (set --peer-var MODEL_CONFIG_SHA256)")
    return problems
