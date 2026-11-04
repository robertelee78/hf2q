"""Splash arm launch, identity, readiness and closing, frozen with the client (native_peer_splash.CLIENT_MODULES).

native_peer_splash_launch.py (the q5 serve entry) calls prepare_run: it verifies the pinned checkout, build and
venv, the pinned DFlash2 draft and the hf2q GGUF, prepares the run's own Splash model directory with Splash's one
public installer interface (install.local.prepare_local_model; docs/models.md "Local model files"), records
launch.json and execs the Splash server in the same process, so q5's SERVER_PID is the server. Readiness
re-derives all of it and binds the running server. Source (incoai/splash 031a25a6):
- prepare_local_model(target, draft, models_dir) links the GGUF and the draft into
  <models_dir>/.resolved/<sha256 of model.json> (install/assembly.py:64-78) and records in model.json each linked
  file's path, size, times and content digest: the SHA-256 of a file outside the HF cache, the blob name (LFS
  sha256) of one inside it (install/assembly.py:113-141). It reads no network (install/local.py). The draft's
  config is checked against the target by the engine's model-check before it installs.
- `python -P -m server.server ASSEMBLY --model ID --binary build/splash ...` (server/server.py:1676-1872; the
  launcher's `splash serve` runs the same module) binds, loads the tokenizer and template from the GGUF, starts
  `build/splash serve-native` as a child (server/runtime.py:520, same process group) and only then listens: it
  prints `Ready · <id> · context 256K · text only · http://127.0.0.1:<port>`.
- GET /ready is 200 {"status": "ready"} while the engine is ready (server/server.py:563-568). GET /status
  (runtime/engine/Status.cpp, server/frontend.py:337-354, server/server.py:1583-1599): instance, ready,
  maximum_context_tokens, identity.cache.build_id, identity.kv, verify_rows, ane_ffn, streamed_experts, requests,
  transport, vision. POST /apply-template renders Chat messages, tools and reasoning options with the serving template
  and POST /tokenize encodes text with the loaded tokenizer (docs/api.md "Request fields").
- Cross-run state Splash keeps outside the run: the Neural Engine split calibration
  ($(getconf DARWIN_USER_CACHE_DIR)/splash-ane/remembered-*; runtime/ane/Program.mm:369-380), recorded before and
  after every run, and ~/Library/Caches/Splash/prefix-cache (written only with --persistent-cache, which stays off;
  every start prunes namespaces unused for 14 days), listed before and after.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import native_peer_splash as sp
from family_peer import same
from qwen38_n1_verdict import loads, require

HEX64, HEX40 = re.compile(r'[0-9a-f]{64}'), re.compile(r'[0-9a-f]{40}')
DOC_LIMIT, RENDER_LIMIT = 4 * 1024 * 1024, 32 * 1024 * 1024
LAUNCH_KEYS = ('q3_family', 'splash_dir', 'commit', 'build_id', 'binary_sha256', 'metallib_sha256', 'host_step_sha256',
               'toolchain_sha256', 'venv_freeze_sha256', 'model', 'model_sha256', 'draft_dir', 'config', 'run_dir',
               'port')
# The build toolchain (design 6.2): each command's full output, hashed as one canonical JSON document.
TOOLCHAIN = {'xcode': ['/usr/bin/xcodebuild', '-version'], 'metal': ['/usr/bin/xcrun', '-sdk', 'macosx', 'metal', '--version'],
             'clang': ['/usr/bin/xcrun', '-sdk', 'macosx', 'clang++', '--version'],
             'sdk': ['/usr/bin/xcrun', '--show-sdk-version'], 'macos': ['/usr/bin/sw_vers']}
SPEC_KEYS = frozenset(LAUNCH_KEYS) | {'server_stdout', 'server_stderr'}
SERVE_ENV = {'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
# kv::storageFormatName (runtime/ops/PagedKv.hpp:32-41) per kv knob; the value layout suffix is the engine's choice.
KV_DTYPE_PREFIX = {'int8': 'q8s8_f32_scale_per_token_head_k_token_major_v_', 'bf16': 'bf16_k_token_major_v_'}
# What a run must never inherit: Splash's own environment overrides (port, API key, defaults, crash traces) and
# Python variables that would change the interpreter or module search path; HF_HOME/HF_HUB_CACHE would move nothing
# here (the draft is local) but are refused so no ambient cache is consulted.
FORBIDDEN_ENV = ('SPLASH_', 'PYTHON', 'HF_HOME', 'HF_HUB_CACHE', 'HF_TOKEN')
TEMPLATE_FIELDS = ('messages', 'tools', 'chat_template_kwargs', 'preserve_thinking')
FORBIDDEN_LOG = ('Engine restarted', 'Engine stopped', 'Error · ', 'Traceback (most recent call last)')
CLOSING_INPUTS = 'splash-closing-inputs.json'   # readiness -> closing, in the coordinator output
PREPARE = ('import json, sys\n'
           'from install.local import prepare_local_model\n'
           'print(json.dumps(str(prepare_local_model(sys.argv[1], draft=sys.argv[2], models_dir=sys.argv[3]))))\n')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def document_bytes(doc):
    return (json.dumps(doc, indent=2, sort_keys=True) + '\n').encode()


def layout(spec):
    run = Path(spec['run_dir'])
    root = Path(spec['splash_dir'])
    return {'run': run, 'models': run / 'models', 'launch': run / 'launch.json', 'root': root,
            'python': root / '.venv' / 'bin' / 'python', 'binary': root / 'build' / 'splash',
            'metallib': root / 'build' / 'splash.metallib', 'host_step': root / 'build' / 'host-step.metallib',
            'build_identity': root / 'build' / 'engine' / 'build-identity.json',
            'requirements': root / 'install' / 'requirements.txt'}


def snapshot_layout(directory, repo, revision):
    path = Path(directory)
    return (path.name == revision and path.parent.name == 'snapshots'
            and path.parent.parent.name == 'models--' + repo.replace('/', '--'))


def launch_spec(values):
    """The launch inputs (q5 serve argv): closed, absolute, pinned."""
    require(isinstance(values, dict) and set(values) == set(LAUNCH_KEYS)
            and all(isinstance(v, str) for v in values.values()), 'closed Splash launch inputs')
    q3 = values['q3_family']
    require(q3 in sp.ARTIFACTS, 'Splash q3_family must be qwen38 or qwen36')
    artifact = sp.ARTIFACTS[q3]
    require(all(Path(values[k]).is_absolute() for k in ('splash_dir', 'model', 'draft_dir', 'run_dir')),
            'Splash paths must be absolute')
    require(values['commit'] == sp.SPLASH_COMMIT and values['build_id'] == sp.BUILD_ID,
            'Splash source must be the inspected 031a25a6 build (build_id ' + sp.BUILD_ID + ')')
    hashes = ('binary_sha256', 'metallib_sha256', 'host_step_sha256', 'toolchain_sha256', 'venv_freeze_sha256',
              'model_sha256')
    require(all(HEX64.fullmatch(values[k]) for k in hashes),
            'SPLASH_BIN/METALLIB/HOST_STEP/TOOLCHAIN/VENV_FREEZE_SHA256 and MODEL_SHA256 must be 64 lowercase hex')
    require(values['model'].endswith('.gguf'), 'the Splash target is the hf2q GGUF file')
    require(snapshot_layout(values['draft_dir'], artifact['draft_repo'], artifact['draft_revision']),
            f"DRAFT_DIR must be the HF cache snapshot of {artifact['draft_repo']}@{artifact['draft_revision']}")
    sp.parse_config(values['config'], q3)
    require(values['port'].isdecimal() and 1024 <= int(values['port']) <= 65535, 'Splash port')
    return {**values, 'port': int(values['port'])}


def native_spec(spec):
    """The approval's closed `native_splash` block (q5 case client.approval.set)."""
    require(isinstance(spec, dict) and set(spec) == SPEC_KEYS, 'closed native_splash approval block')
    launch = launch_spec({k: spec[k] if k != 'port' else str(spec[k]) for k in LAUNCH_KEYS})
    out, err = Path(spec['server_stdout']), Path(spec['server_stderr'])
    require(out.name == 'server.stdout' and err.name == 'server.stderr' and out.parent == err.parent
            == Path(launch['run_dir']).parent, "Splash server logs must be the run's")
    return {**launch, 'server_stdout': str(out), 'server_stderr': str(err)}


def server_argv(spec, assembly):
    """The one approved Splash server (server/server.py parse_args + serve_options); -P keeps the working directory
    off sys.path, -u keeps the status lines unbuffered. PYTHONPATH (serve_environ) names the checkout."""
    paths = layout(spec)
    knobs = sp.parse_config(spec['config'], spec['q3_family'])
    return [str(paths['python']), '-P', '-u', '-m', 'server.server', str(assembly),
            '--model', sp.MODEL_IDS[spec['q3_family']], '--binary', str(paths['binary']),
            '--host', '127.0.0.1', '--port', str(spec['port']), '--max-context', sp.MAX_CONTEXT_FLAG,
            '--idle-release', 'off', '--no-webui', '--thinking-end-text', sp.THINKING_END_TEXT] + sp.server_flags(knobs)


def check_environ(environ):
    require(not any(k.startswith(FORBIDDEN_ENV) for k in environ),
            'no SPLASH_/PYTHON/HF_HOME/HF_HUB_CACHE/HF_TOKEN variable may reach the Splash server')
    require(all(environ.get(k) == v for k, v in SERVE_ENV.items()),
            'Splash serve env must be offline (HF_HUB_OFFLINE=1, HF_HUB_DISABLE_TELEMETRY=1)')


def serve_environ(spec, environ):
    check_environ(environ)
    return {**environ, 'PYTHONPATH': str(layout(spec)['root'])}


def git(root, *args):
    return subprocess.run(['/usr/bin/git', '-C', str(root), *args], capture_output=True, timeout=30,
                          check=True).stdout.decode().strip()


def source_identity(spec):
    """Detached HEAD at the pinned commit, no tracked-file edits, the pinned build identity and binary."""
    paths = layout(spec)
    require(git(paths['root'], 'rev-parse', 'HEAD') == sp.SPLASH_COMMIT, 'SPLASH_DIR is not a checkout of 031a25a6')
    require(git(paths['root'], 'status', '--porcelain', '--untracked-files=no') == '', 'SPLASH_DIR has tracked-file changes')
    build = loads(paths['build_identity'].read_bytes())
    require(isinstance(build, dict) and build.get('build_id') == sp.BUILD_ID,
            'build/engine/build-identity.json is not the pinned build_id')
    require(sha256(paths['binary']) == spec['binary_sha256'], 'build/splash differs from SPLASH_BIN_SHA256')
    require(sha256(paths['metallib']) == spec['metallib_sha256'], 'build/splash.metallib differs from SPLASH_METALLIB_SHA256')
    require(paths['host_step'].is_file() and sha256(paths['host_step']) == spec['host_step_sha256'],
            'build/host-step.metallib is missing or differs from SPLASH_HOST_STEP_SHA256 (awake host steps)')
    require(sha256(paths['requirements']) == sp.REQUIREMENTS_SHA256, 'install/requirements.txt differs from the pin')
    tools = toolchain()
    require(tools['sha256'] == spec['toolchain_sha256'], 'the host toolchain differs from SPLASH_TOOLCHAIN_SHA256')
    return {'commit': sp.SPLASH_COMMIT, 'describe': sp.SPLASH_DESCRIBE, 'build_id': sp.BUILD_ID,
            'binary_sha256': spec['binary_sha256'], 'metallib_sha256': spec['metallib_sha256'],
            'host_step_sha256': spec['host_step_sha256'], 'requirements_sha256': sp.REQUIREMENTS_SHA256,
            'toolchain': tools, 'tracked_changes': 'none (git status --untracked-files=no)'}


def toolchain():
    """The host's Xcode, Metal, clang, SDK and macOS versions and the sha256 of their canonical JSON."""
    versions = {name: subprocess.run(argv, capture_output=True, timeout=30, check=True).stdout.decode().strip()
                for name, argv in TOOLCHAIN.items()}
    # clang/metal print their install directory, which a Metal toolchain update relocates; keep the version lines.
    versions = {k: '\n'.join(line for line in v.splitlines() if not line.startswith('InstalledDir:'))
                for k, v in versions.items()}
    return {'versions': versions, 'sha256': hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()}


def venv_inventory(splash_dir):
    """pip-freeze-equivalent inventory of the checkout's .venv from dist-info, every requirements.txt pin present."""
    root = Path(splash_dir)
    venv = root / '.venv'
    require((venv / 'pyvenv.cfg').is_file(), 'SPLASH_DIR/.venv is not a venv (make install-environment)')
    rows = {}
    for info in sorted(venv.glob('lib/python3.*/site-packages/*.dist-info')):
        meta = dict(line.split(': ', 1) for line in (info / 'METADATA').read_text(errors='replace').splitlines()
                    if line.startswith(('Name: ', 'Version: ')))
        name = meta.get('Name', '').lower().replace('_', '-')
        require(name and name not in rows, f'venv dist-info {info.name} unreadable or duplicated')
        rows[name] = meta.get('Version')
    pins = {}
    for line in (root / 'install' / 'requirements.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            name, _, version = line.partition('==')
            pins[name.strip().lower().replace('_', '-')] = version.strip()
    wrong = sorted(n for n, v in pins.items() if rows.get(n) != v)
    require(not wrong, 'Splash venv differs from install/requirements.txt: ' + ','.join(wrong))
    text = ''.join(f'{n}=={v}\n' for n, v in sorted(rows.items()))
    return {'venv': str(venv), 'packages': len(rows), 'freeze_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'pins': len(pins)}


def draft_identity(spec):
    """The pinned DFlash2 draft snapshot: config.json bytes and the model.safetensors LFS blob."""
    artifact = sp.ARTIFACTS[spec['q3_family']]
    path = Path(spec['draft_dir'])
    require((path / 'config.json').is_file() and (path / 'model.safetensors').is_symlink(),
            'DRAFT_DIR lacks config.json or the model.safetensors cache link')
    require(sha256(path / 'config.json') == artifact['draft_config_sha256'], 'draft config.json differs from the pin')
    blob = Path(os.readlink(path / 'model.safetensors')).name
    require(blob == artifact['draft_weights_sha256'], 'draft model.safetensors is not the pinned LFS blob')
    return {'repo': artifact['draft_repo'], 'revision': artifact['draft_revision'], 'path': str(path),
            'config_sha256': artifact['draft_config_sha256'], 'weights_sha256': blob,
            'weights_bytes': (path / 'model.safetensors').stat().st_size}


def verify_identity(spec):
    model = Path(spec['model'])
    require(model.is_file() and not model.is_symlink(), 'MODEL must be the hf2q GGUF file itself')
    return {'source': source_identity(spec), 'venv': venv_inventory(spec['splash_dir']), 'draft': draft_identity(spec),
            'model': {'path': str(model), 'bytes': model.stat().st_size, 'sha256': spec['model_sha256'],
                      'sha256_source': 'Splash model.json content digest (verify_assembly)'}}


def verify_assembly(spec, assembly):
    """Splash's model.json: the hf2q GGUF by content digest, the pinned draft by blob digest, its family, text only."""
    assembly = Path(assembly)
    require(assembly.parent.name == '.resolved' and assembly.parent.parent == layout(spec)['models'],
            "the Splash assembly must be the run's own")
    raw = (assembly / 'model.json').read_bytes()
    record = loads(raw)
    artifact = sp.ARTIFACTS[spec['q3_family']]
    require(isinstance(record, dict) and record.get('family') == artifact['splash_family']
            and record.get('target_format') == 'gguf' and record.get('vision_format') == 'none',
            'Splash model.json family/formats (gguf target, no vision)')
    files = record.get('files')
    require(isinstance(files, dict) and all(isinstance(f, dict) for f in files.values()), 'Splash model.json files')
    by_path = {f.get('path'): f for f in files.values()}
    target = by_path.get(spec['model'])
    require(target is not None and target.get('digest') == spec['model_sha256'],
            'Splash model.json does not record the hf2q GGUF with MODEL_SHA256')
    require(sum(1 for f in files.values() if str(f.get('path', '')).endswith('.gguf')) == 1,
            'Splash links exactly one GGUF (no projector, no other part)')
    weights = [f for f in files.values() if f.get('digest') == artifact['draft_weights_sha256']]
    require(len(weights) == 1, 'Splash model.json does not link the pinned draft weights')
    template = files.get('tokenizer/chat_template.jinja') or {}
    require(template.get('digest') == artifact['template_sha256'],
            'the template Splash derived from the GGUF is not the reviewed hf2q template')
    return {'path': str(assembly), 'model_json_sha256': hashlib.sha256(raw).hexdigest(),
            'family': record['family'], 'target_digest': target['digest'], 'draft_digest': weights[0]['digest'],
            'template_digest': template['digest'], 'linked_files': sorted(files)}


def filesystem(path):
    """The file system type of the volume holding path (`df -P` mount point, then `mount`)."""
    df = subprocess.run(['/bin/df', '-P', str(path)], capture_output=True, timeout=10, check=True).stdout.decode()
    point = df.splitlines()[-1].split(None, 5)[-1]
    for line in subprocess.run(['/sbin/mount'], capture_output=True, timeout=10, check=True).stdout.decode().splitlines():
        head, _, options = line.rpartition(' (')
        if head.endswith(' on ' + point):
            return options.split(',')[0]
    return None


def ane_cache_dir():
    out = subprocess.run(['/usr/bin/getconf', 'DARWIN_USER_CACHE_DIR'], capture_output=True, timeout=5,
                         check=True).stdout.decode().strip()
    require(out.startswith('/'), 'macOS names no per-user cache directory')
    return Path(out) / 'splash-ane'


def pin_dir(draft_dir):
    """refs/splash of the draft's HF cache repository, where Splash pins the snapshots an installation links
    (install/hub.py pin, one <sha256 of the installation path>/<commit> per installation). It pins only sources
    with a revision (install/assembly.py pins); the local paths prepare_run passes have none (model.json
    `sources.*.revision` null, observed in the dry run), so a run must add no pin."""
    return Path(draft_dir).parent.parent / 'refs' / 'splash'


def draft_pins(draft_dir):
    folder = pin_dir(draft_dir)
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def cross_run_state(draft_dir):
    """What Splash keeps between runs on this Mac: the Neural Engine split calibration, the persistent-cache
    namespaces and any draft snapshot pins in the shared HF cache."""
    ane = ane_cache_dir()
    remembered = {p.name: sha256(p) for p in sorted(ane.glob('remembered-*')) if p.is_file()} if ane.is_dir() else {}
    prefix = Path.home() / 'Library' / 'Caches' / 'Splash' / 'prefix-cache'
    return {'ane_dir': str(ane), 'ane_remembered': remembered,
            'prefix_cache_namespaces': sorted(p.name for p in prefix.iterdir()) if prefix.is_dir() else [],
            'draft_pin_dir': str(pin_dir(draft_dir)), 'draft_pins': draft_pins(draft_dir)}


def prepare_run(spec, environ):
    """Launcher side: the run's own model directory through prepare_local_model; returns (assembly, record).
    Refuses an assembly that pinned a snapshot in the shared HF cache (state outside the run directory)."""
    paths = layout(spec)
    # Splash's installer collects garbage in models_dir and stops on the AppleDouble `._*` files exFAT volumes write
    # (observed on the external drive), so the run directory must be on APFS (the internal disk).
    require(filesystem(paths['run'].parent) == 'apfs', 'the Splash run directory must be on an APFS volume')
    paths['run'].mkdir()
    paths['models'].mkdir()
    before = draft_pins(spec['draft_dir'])
    result = subprocess.run([str(paths['python']), '-P', '-c', PREPARE, spec['model'], spec['draft_dir'],
                             str(paths['models'])], capture_output=True, timeout=1800,
                            env=serve_environ(spec, environ), cwd=str(paths['run']))
    (paths['run'] / 'prepare.stdout').write_bytes(result.stdout)
    (paths['run'] / 'prepare.stderr').write_bytes(result.stderr)
    require(result.returncode == 0, 'Splash prepare_local_model refused: ' + result.stderr.decode('utf-8', 'replace')[-2000:])
    assembly = loads(result.stdout.decode().strip().splitlines()[-1])
    require(draft_pins(spec['draft_dir']) == before,
            f"Splash pinned the draft snapshot in the shared HF cache ({pin_dir(spec['draft_dir'])})")
    return assembly, verify_assembly(spec, assembly)


def launch_record(spec, identity, assembly, environ):
    return {'schema': 'hf2q.splash-launch.v1', 'pid': os.getpid(), 'argv': server_argv(spec, assembly['path']),
            'env': {k: v for k, v in sorted(serve_environ(spec, environ).items())}, 'spec': spec,
            'identity': identity, 'assembly': assembly, 'state_before': cross_run_state(spec['draft_dir'])}


def verify_launch(spec, packet):
    """launch.json bound by the approval, its pid the server's, the argv the approved one, the assembly intact."""
    paths = layout(spec)
    require(sha256(paths['launch']) == packet.get('native_splash_launch_sha256'),
            'Splash launch.json differs from the approval')
    record = loads(paths['launch'].read_bytes())
    require(isinstance(record, dict) and record.get('schema') == 'hf2q.splash-launch.v1'
            and record.get('pid') == packet.get('server_pid')
            and record.get('spec') == {k: spec[k] for k in LAUNCH_KEYS}, 'Splash launch record pid/spec differs')
    assembly = verify_assembly(spec, record['assembly']['path'])
    require(assembly == record['assembly'], 'Splash assembly changed since launch')
    require(record.get('argv') == server_argv(spec, assembly['path']), 'Splash launch argv differs from the approved one')
    env = record.get('env') or {}
    check_environ({k: v for k, v in env.items() if k != 'PYTHONPATH'})
    require(env.get('PYTHONPATH') == spec['splash_dir'], 'Splash PYTHONPATH must be the checkout alone')
    require(record['identity']['venv']['freeze_sha256'] == spec['venv_freeze_sha256'],
            'Splash venv freeze differs from SPLASH_VENV_FREEZE_SHA256')
    return record


def _port(base):
    u = urlsplit(base)
    require(u.scheme == 'http' and u.hostname == '127.0.0.1' and u.port and u.path == '/v1'
            and not u.query and not u.fragment and u.username is None, 'Splash loopback base')
    return u.port


def fetch(port, path, output, name, body=None, limit=DOC_LIMIT, timeout=5):
    """GET (or POST `body`) a loopback document; the raw reply is archived before any validation."""
    data = None if body is None else json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
    request = Request(f'http://127.0.0.1:{port}{path}', data=data, method='GET' if body is None else 'POST',
                      headers={} if body is None else {'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(limit + 1)
        Path(output, name).write_bytes(raw)
        require(response.status == 200 and len(raw) <= limit, f'Splash {path} status/size')
    return raw


def _get(doc, dotted):
    for key in dotted.split('.'):
        require(isinstance(doc, dict) and key in doc, f'Splash /status lacks {dotted}')
        doc = doc[key]
    return doc


def validate_status(doc, spec, pid, requests=0):
    """/status of the approved server: identity, context, text only, verify rows, no streaming, no restart."""
    q3 = spec['q3_family']
    knobs = sp.parse_config(spec['config'], q3)
    require(isinstance(doc, dict), 'Splash /status envelope')
    instance = _get(doc, 'instance')
    require(instance.get('model') == sp.MODEL_IDS[q3] and instance.get('port') == spec['port']
            and instance.get('pid') == pid, 'Splash /status instance (model, port, pid)')
    require(_get(doc, 'ready') is True and _get(doc, 'maximum_context_tokens') == sp.CONTEXT_PER_SLOT
            and _get(doc, 'vision') is False, 'Splash /status ready, 262144 context, text only')
    require(_get(doc, 'identity.cache.build_id') == sp.BUILD_ID, 'Splash /status build_id is not the pinned build')
    # runtime/engine/Status.cpp: identity.kv.format is kv::formatName, identity.cache.dtype kv::storageFormatName
    # (runtime/ops/PagedKv.hpp), both of the configured --kv-format.
    require(_get(doc, 'identity.kv.format') == knobs['kv'], f"Splash /status KV format is not {knobs['kv']}")
    require(str(_get(doc, 'identity.cache.dtype')).startswith(KV_DTYPE_PREFIX[knobs['kv']]),
            f"Splash /status cache dtype is not a {knobs['kv']} layout")
    verify = 'all' if sp.ARTIFACTS[q3]['dense'] else knobs['verify']
    require(_get(doc, 'verify_rows') == verify, f'Splash /status verify_rows is not {verify}')
    require(_get(doc, 'streamed_experts') is None, 'Splash streams routed experts (memory plan below the model)')
    ane = _get(doc, 'ane_ffn')
    require(knobs['ane'] == 'auto' or (isinstance(ane, dict) and ane.get('state') == 'off'),
            'Splash Neural Engine split is not off under ane=off')
    transport = _get(doc, 'transport')
    require(isinstance(transport, dict) and transport.get('recovering') is False and not transport.get('stopped'),
            'Splash engine is restarting or stopped')
    require(transport.get('restarts') == 0, f"Splash engine restarted {transport.get('restarts')} time(s)")
    counts = _get(doc, 'requests')
    require(isinstance(counts, dict) and counts.get('submitted') == counts.get('completed') == requests
            and counts.get('failed') == counts.get('cancelled') == 0,
            f'Splash served {counts.get("submitted")} requests, expected {requests} completed and none failed')
    return {'instance': instance, 'build_id': sp.BUILD_ID, 'identity': _get(doc, 'identity'), 'verify_rows': verify,
            'ane_ffn': ane, 'requests': counts, 'memory_actual': doc.get('memory_actual'),
            'memory_plan': doc.get('memory_plan'), 'thermal_state': doc.get('thermal_state'),
            'chat_template': doc.get('chat_template'), 'maximum_context_tokens': sp.CONTEXT_PER_SLOT}


def validate_models(doc, q3):
    require(isinstance(doc, dict) and isinstance(doc.get('data'), list), 'Splash /v1/models envelope')
    served = [m for m in doc['data'] if isinstance(m, dict) and m.get('id') == sp.MODEL_IDS[q3]]
    require(len(doc['data']) == 1 and len(served) == 1, 'Splash must serve exactly the pinned model id')
    entry = served[0]
    require(entry.get('max_model_len') == sp.CONTEXT_PER_SLOT and entry.get('vision') is False,
            'Splash /v1/models context and text-only state')
    return {k: entry.get(k) for k in ('id', 'max_model_len', 'context_length', 'vision', 'input_modalities', 'reasoning')}


def startup(stdout, stderr, spec):
    text = stdout + stderr
    bad = [line for line in FORBIDDEN_LOG if line in text]
    require(not bad, 'Splash startup logged: ' + ' | '.join(bad))
    ready = f"Ready · {sp.MODEL_IDS[spec['q3_family']]} · context 256K · text only · http://127.0.0.1:{spec['port']}"
    require(ready in stdout, 'Splash Ready line missing: ' + ready)
    template = [line for line in stdout.splitlines() if ' Chat template · ' in line]
    require(len(template) == 1 and 'patched to render later system messages in place' in template[0],
            'Splash chat-template probe line (patched in place, as probed on the hf2q GGUF template)')
    status = [line for line in text.splitlines() if re.match(r'\d\d:\d\d:\d\d ', line)]
    return {'ready_line': ready, 'chat_template_line': template[0], 'status_lines': status[:200]}


BUILD_INFO_HASHES = ('binary_sha256', 'metallib_sha256', 'host_step_sha256', 'toolchain_sha256', 'venv_freeze_sha256')


def build_info(commit, build_id, hashes):
    """Campaign identity string (q3 drift check); None while an input is unresolved. `hashes` maps every
    BUILD_INFO_HASHES name to its 64-hex value."""
    values = [hashes.get(name) for name in BUILD_INFO_HASHES]
    if not (isinstance(commit, str) and HEX40.fullmatch(commit) and isinstance(build_id, str)
            and all(isinstance(v, str) and HEX64.fullmatch(v) for v in values)):
        return None
    return (f'splash {sp.SPLASH_DESCRIBE} {commit} {build_id} '
            + ' '.join(f"{name.removesuffix('_sha256').replace('_', '-')}-sha256 {value}"
                       for name, value in zip(BUILD_INFO_HASHES, values)))


def readiness(base, model, output, packet):
    """Coordinator readiness: approval binding, source/build/venv/draft, assembly, documents and startup log."""
    require(isinstance(packet, dict), 'Splash approval packet')
    spec = native_spec(packet.get('native_splash'))
    q3 = sp.controls_family(packet.get('family_controls'))
    require(q3 == spec['q3_family'] and packet['family_controls']['config'] == spec['config']
            and packet.get('family', 'qwen') == sp.ARTIFACTS[q3]['family'],
            'approved family_controls are for another Splash family or configuration')
    require(model == sp.MODEL_IDS[q3], 'Splash model id must be ' + sp.MODEL_IDS[q3])
    port = _port(base)
    require(port == spec['port'], 'Splash base URL port differs from the launch port')
    identity = verify_identity(spec)
    launch = verify_launch(spec, packet)
    require(identity['venv']['freeze_sha256'] == spec['venv_freeze_sha256'], 'Splash venv changed since launch')
    ready = loads(fetch(port, '/ready', output, 'splash-ready.raw'))
    require(ready == {'status': 'ready'}, 'Splash /ready is not ready')
    status = validate_status(loads(fetch(port, '/status', output, 'splash-status.raw')), spec, packet['server_pid'])
    served = validate_models(loads(fetch(port, '/v1/models', output, 'splash-models.raw')), q3)
    fetch(port, '/metrics', output, 'splash-metrics.raw')
    logs = {}
    for stream in ('stdout', 'stderr'):
        raw = Path(spec['server_' + stream]).read_bytes()
        Path(output, 'splash-startup.' + stream).write_bytes(raw)
        logs[stream] = raw.decode('utf-8', 'replace')
    facts = startup(logs['stdout'], logs['stderr'], spec)
    Path(output, CLOSING_INPUTS).write_bytes(document_bytes({'spec': spec, 'server_pid': packet['server_pid'],
                                                             'state_before': launch['state_before']}))
    knobs = sp.parse_config(spec['config'], q3)
    return {'engine_kind': sp.ENGINE, 'family': sp.ARTIFACTS[q3]['family'], 'q3_family': q3, 'label': sp.LABELS[q3],
            'role': 'same-artifact peer of record', 'source_revision': sp.SPLASH_COMMIT, 'build_id': sp.BUILD_ID,
            'logical_slots': sp.SLOTS, 'context_per_slot': sp.CONTEXT_PER_SLOT, 'config': spec['config'],
            'knobs': knobs, 'speculation': 'DFlash2 always on (no off switch)',
            'think_mode': 'thinking (enable_thinking, preserve_thinking, budget 2048/512, hf2q forced close)',
            'sampling': dict(sp.SAMPLING), 'repetition_history': sp.REPETITION_PARITY,
            'controls': packet['family_controls'], 'identity': identity, 'assembly': launch['assembly'],
            'state_before': launch['state_before'], 'server_argv': launch['argv'], 'status': status,
            'served_model': served, 'startup': facts,
            'build_info': build_info(sp.SPLASH_COMMIT, sp.BUILD_ID, {k: spec[k] for k in BUILD_INFO_HASHES}),
            'same_artifact': {'model_sha256': spec['model_sha256'], 'bound_by': 'Splash model.json content digest'},
            'chat_template': {'source': 'the GGUF-embedded template (no cross-repository fallback)',
                              'hf2q_reviewed_sha256': sp.ARTIFACTS[q3]['template_sha256'],
                              'later_system': status['chat_template']},
            'tokenizer': 'derived by Splash from the GGUF metadata; per-turn ids archived by the closing render proof',
            'physical_cohort': 'Unavailable:not proved by readiness or HTTP overlap',
            'pool_profile': 'NotApplicable:HF-only', 'full_prefix_reuse': 'PendingPostOracle'}


def token_sha256(tokens):
    return hashlib.sha256(','.join(str(t) for t in tokens).encode()).hexdigest()


def render_proof(base, output, turns):
    """Every turn's archived request re-rendered by /apply-template and encoded by /tokenize; the count must equal
    usage.prompt_tokens. The rendered prompt is archived for the offline byte comparison with hf2q's rendering."""
    port, rows = _port(base), []
    for receipt in turns:
        directory = Path(output) / f"turn-{receipt['turn']}"
        raw = (directory / 'request.json').read_bytes()
        require(hashlib.sha256(raw).hexdigest() == receipt['request_sha256'],
                'Splash render proof: request archive differs from its receipt')
        request = loads(raw)
        rendered = fetch(port, '/apply-template', directory, 'splash-apply-template.raw',
                         {k: request[k] for k in TEMPLATE_FIELDS}, RENDER_LIMIT, 60)
        prompt = loads(rendered)
        require(isinstance(prompt, dict) and set(prompt) == {'prompt'} and isinstance(prompt['prompt'], str),
                'Splash /apply-template envelope')
        encoded = fetch(port, '/tokenize', directory, 'splash-tokenize.raw',
                        {'content': prompt['prompt'], 'add_special': False}, RENDER_LIMIT, 60)
        doc = loads(encoded)
        require(isinstance(doc, dict) and set(doc) == {'tokens'} and isinstance(doc['tokens'], list)
                and all(type(t) is int and t >= 0 for t in doc['tokens']), 'Splash /tokenize envelope')
        require(len(doc['tokens']) == receipt['usage']['prompt_tokens'],
                'Splash rendered prompt token count differs from the turn usage.prompt_tokens')
        rows.append({'turn': receipt['turn'], 'request_sha256': receipt['request_sha256'],
                     'prompt_sha256': hashlib.sha256(prompt['prompt'].encode()).hexdigest(),
                     'count': len(doc['tokens']), 'tokens_sha256': token_sha256(doc['tokens']),
                     'apply_template_sha256': hashlib.sha256(rendered).hexdigest(),
                     'tokenize_sha256': hashlib.sha256(encoded).hexdigest()})
    return {'status': 'Observed', 'endpoints': ['/apply-template', '/tokenize'], 'fields': list(TEMPLATE_FIELDS),
            'turns': rows, 'timing': 'coordinator closing, after every agent finished: outside the makespan clock',
            'hf2q_prompt_parity': 'PendingPostOracle: splash-apply-template.raw is compared byte for byte with hf2q\'s '
                                  'rendering of the canonical request by the hf2q post-oracle '
                                  '(splash_sweep.py parity-config, then splash_prompt_parity_post_oracle); a mismatch '
                                  'makes the run Invalid'}


def closing(base, output):
    """Once every agent finished: render proofs, then /status (every request completed, none hidden) and the
    cross-run state after the run."""
    render, turns = {}, []
    for summary in sorted(Path(output).glob('agent-*/summary.json')):
        receipts = loads(summary.read_bytes())['turns']
        render[summary.parent.name] = render_proof(base, summary.parent, receipts)
        turns += receipts
    inputs = loads(Path(output, CLOSING_INPUTS).read_bytes())
    spec = native_spec({**inputs['spec'], 'port': str(inputs['spec']['port'])})
    status = validate_status(loads(fetch(_port(base), '/status', output, 'splash-status-final.raw')), spec,
                             inputs['server_pid'], len(turns))
    after = cross_run_state(spec['draft_dir'])
    before = inputs['state_before']
    return {'render': render, 'status': status, 'state_before': before, 'state_after': after,
            'ane_calibration_changed': before['ane_remembered'] != after['ane_remembered'],
            'source': '/status requests counters (Status.cpp) equal the client turn count; no request failed'}


ASSEMBLY_PLACEHOLDER = '<per-run assembly>'


def argv_template(variables, q3):
    """The server argv with the assembly as a per-run value (the frozen campaign form); None while unresolved."""
    try:
        spec = {'q3_family': q3, 'splash_dir': variables['SPLASH_DIR'], 'run_dir': '/', 'config': variables['SPLASH_CONFIG'],
                'port': int(variables['PORT'])}
        return server_argv(spec, ASSEMBLY_PLACEHOLDER)
    except (KeyError, TypeError, ValueError):  # an unresolved or invalid variable (Invalid is a ValueError)
        return None


def campaign_identity(plan):
    """Frozen per campaign and re-checked before every run (q3_identity): build, venv, draft, configuration and the
    argv template. Never the per-run assembly path, model.json bytes, launch.json or the Neural Engine calibration
    state, which readiness binds per run (an hf2q run that hard-links the GGUF changes its ctime and so Splash's
    assembly path; the first start of a configuration calibrates the split)."""
    v = plan['variables']
    venv = draft = tools = None
    hashes = {}
    root = v.get('SPLASH_DIR')
    if isinstance(root, str) and Path(root, '.venv', 'pyvenv.cfg').is_file():
        venv = venv_inventory(root)
        hashes['venv_freeze_sha256'] = venv['freeze_sha256']
    if isinstance(root, str):
        paths = layout({'splash_dir': root, 'run_dir': '/'})
        for name, key in (('binary_sha256', 'binary'), ('metallib_sha256', 'metallib'), ('host_step_sha256', 'host_step')):
            if paths[key].is_file():
                hashes[name] = sha256(paths[key])
        try:
            tools = toolchain()
            hashes['toolchain_sha256'] = tools['sha256']
        except (OSError, subprocess.SubprocessError):
            tools = None
    q3 = plan.get('family')
    if q3 in sp.ARTIFACTS and isinstance(v.get('DRAFT_DIR'), str) and Path(v['DRAFT_DIR']).is_dir():
        draft = draft_identity({'q3_family': q3, 'draft_dir': v['DRAFT_DIR']})
    # The build string is of the files and toolchain as found; launch refuses values that differ from the case's.
    return {'peer_build_info': build_info(v.get('SPLASH_COMMIT'), v.get('SPLASH_BUILD_ID'), hashes),
            'native_runtime': {'venv': venv, 'draft': draft, 'config': v.get('SPLASH_CONFIG'), 'toolchain': tools,
                               'actual': hashes,
                               'declared': {name: v.get(var) for name, var in (
                                   ('binary_sha256', 'SPLASH_BIN_SHA256'), ('metallib_sha256', 'SPLASH_METALLIB_SHA256'),
                                   ('host_step_sha256', 'SPLASH_HOST_STEP_SHA256'),
                                   ('toolchain_sha256', 'SPLASH_TOOLCHAIN_SHA256'),
                                   ('venv_freeze_sha256', 'SPLASH_VENV_FREEZE_SHA256'))},
                               'argv_template': argv_template(v, q3) if q3 in sp.ARTIFACTS else None}}


def launch_problems(name, artifact, hf2q_sha256, identity=None, output=None):
    """Q3 refusals before any run: Splash serves the hf2q GGUF itself, so its artifact sha256 must equal the hf2q
    arm's; the build, toolchain and venv found must be the case's declared ones; and the campaign output must be on
    APFS (prepare_run's rule). Each would otherwise refuse every launch and leave Splash Unavailable mid-campaign."""
    problems = []
    if not (artifact or {}).get('sha256'):
        problems.append(f'{name}: the Splash artifact needs MODEL_SHA256 (the hf2q GGUF sha256)')
    elif artifact['sha256'] != hf2q_sha256:
        problems.append(f"{name}: Splash must serve the hf2q arm's GGUF (sha256 {artifact['sha256'][:12]}... is not "
                        f"the experiment's {str(hf2q_sha256)[:12]}...)")
    if artifact and not str(artifact.get('path', '')).endswith('.gguf'):
        problems.append(f'{name}: the Splash artifact must be the hf2q GGUF file')
    runtime = (identity or {}).get('native_runtime') or {}
    actual = runtime.get('actual') or {}
    for key, declared in sorted((runtime.get('declared') or {}).items()):
        if actual.get(key) != declared:
            problems.append(f'{name}: {key} found {actual.get(key)} but the case declares {declared} (rebuild, '
                            'toolchain or venv change: re-run splash_sweep.py identity and re-confirm the case)')
    if output is not None:
        where = next((p for p in (Path(output), *Path(output).parents) if p.exists()), None)
        if where is None or filesystem(where) != 'apfs':
            problems.append(f'{name}: the campaign output {output} must be on APFS (Splash run directories)')
    return problems
