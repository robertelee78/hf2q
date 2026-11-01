"""Pure controls for the oMLX arm's launch files, identity, readiness, /v1/tokenize proof and closing record.

No oMLX process: the checkout, venv and HF snapshot are temporary trees, git/ps, HTTP documents and exec are
patched. Document shapes follow /opt/omlx 6b413c90: /health (server.py:3348-3395), /api/status (3398-3479),
/v1/models/status (3937-3984, engine_pool.py:4713-4775), /v1/tokenize (8380-8434, TokenizeResponse
api/openai_models.py:632-638), settings.json (settings.py:1548-1571), the serve prints (cli.py:185-420) and the
settings log lines (model_settings.py:856, engine_pool.py:1640-1643).
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import native_peer_omlx as om
import native_peer_omlx_launch as launcher
import native_peer_omlx_ready as ready
from qwen38_n1_verdict import Invalid
from test_native_peer_omlx import FAMILY, MODEL, native

TEMPLATE = b'{%- if enable_thinking %}<think>{% endif %}\n'
CONFIG = b'{"model_type":"qwen3_5"}'
ENV = {'PATH': '/usr/bin:/bin', 'HOME': '/Users/x', 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
# ModelSettings fields at 6b413c90 (model_settings.py:363-523); the generated file may use no other key.
KNOWN_FIELDS = {'max_context_window', 'max_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'min_p',
                'presence_penalty', 'force_sampling', 'enable_thinking', 'preserve_thinking', 'thinking_budget_enabled',
                'model_type_override', 'turboquant_kv_enabled', 'mtp_enabled', 'dflash_enabled', 'vlm_mtp_enabled',
                'specprefill_enabled', 'a8_prefill_enabled', 'qwen35_ane_prefill_enabled', 'moe_expert_offload_enabled',
                'activation_fp16_enabled', 'guided_grammar_enabled', 'trust_remote_code', 'is_pinned', 'is_default'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(hub, q3, template=TEMPLATE):
    artifact = om.ARTIFACTS[q3]
    base = Path(hub) / ('models--' + artifact['repo'].replace('/', '--'))
    path = base / 'snapshots' / artifact['revision']
    path.mkdir(parents=True)
    (base / 'blobs').mkdir()
    index = json.dumps({'weight_map': {'a': 'model-00001-of-00002.safetensors', 'b': 'model-00002-of-00002.safetensors'}})
    for name, data in {'config.json': CONFIG, 'tokenizer.json': b'{"vocab":{}}', 'tokenizer_config.json': b'{}',
                       'chat_template.jinja': template, 'model.safetensors.index.json': index.encode()}.items():
        (path / name).write_bytes(data)
    for name, data in {'model-00001-of-00002.safetensors': b'w1', 'model-00002-of-00002.safetensors': b'w2'}.items():
        (base / 'blobs' / sha(data)).write_bytes(data)
        os.symlink(base / 'blobs' / sha(data), path / name)
    return path


def checkout(root, packages=None):
    """A detached 6b413c90 checkout with its editable venv (.venv) and dist-info inventory."""
    src = Path(root) / 'omlx-6b413c9'
    (src / '.git').mkdir(parents=True)
    (src / '.git' / 'HEAD').write_text(om.OMLX_COMMIT + '\n')
    (src / 'omlx').mkdir()
    (src / 'omlx' / '_version.py').write_text('__version__ = "0.7.3rc1"\n')
    venv = src / '.venv'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'bin' / 'python').write_bytes(b'python')
    (venv / 'pyvenv.cfg').write_text('home = /usr/bin\n')
    site = venv / 'lib' / 'python3.12' / 'site-packages'
    git = lambda url, commit: {'url': url, 'vcs_info': {'vcs': 'git', 'commit_id': commit}}
    rows = {'mlx': ('0.32.4', None), 'mlx_lm': ('0.32.3.dev5', git('https://github.com/ml-explore/mlx-lm', ready.PINS['mlx-lm'][1])),
            'mlx_vlm': ('0.7.11', git('https://github.com/Blaizzy/mlx-vlm', ready.PINS['mlx-vlm'][1])),
            'omlx': ('0.7.3rc1', {'url': src.resolve().as_uri(), 'dir_info': {'editable': True}}),
            'transformers': ('5.20.0', None), **(packages or {})}
    for name, (version, url) in rows.items():
        info = site / f'{name}-{version}.dist-info'
        info.mkdir(parents=True)
        (info / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
        if url is not None:
            (info / 'direct_url.json').write_text(json.dumps(url))
    return src


def spec_for(root, q3='qwen38', **changes):
    artifact = om.ARTIFACTS[q3]
    src = Path(root) / 'omlx-6b413c9'
    model = Path(root) / 'hub' / ('models--' + artifact['repo'].replace('/', '--')) / 'snapshots' / artifact['revision']
    spec = {'q3_family': q3, 'omlx_dir': str(src), 'omlx_py': str(src / '.venv' / 'bin' / 'python'),
            'commit': om.OMLX_COMMIT, 'version': om.OMLX_VERSION, 'model_dir': str(model),
            'model_revision': artifact['revision'], 'model_config_sha256': sha(CONFIG), 'template_sha256': sha(TEMPLATE),
            'run_dir': str(Path(root) / 'run' / 'omlx'), 'port': '18254',
            'server_stdout': str(Path(root) / 'run' / 'server.stdout'),
            'server_stderr': str(Path(root) / 'run' / 'server.stderr')}
    spec.update(changes)
    return spec


def tools(title='omlx-server', dirty=b''):
    """subprocess.run stand-in for `git status` (clean) and `ps` (the process title)."""
    def run(argv, **kwargs):
        out = dirty if argv[0] == '/usr/bin/git' else (title + '\n').encode()
        return subprocess.CompletedProcess(argv, 0, out, b'')
    return patch.object(ready.subprocess, 'run', side_effect=run)


def saved_settings(spec, **changes):
    paths = ready.layout(ready.native_spec(spec))
    doc = {'version': '1.0', 'server': {'host': '127.0.0.1', 'port': int(spec['port']), 'sse_keepalive_mode': 'off',
                                        'log_level': 'info', 'burst_decode_mode': 'balanced'},
           'scheduler': {'max_concurrent_requests': 4, 'chunked_prefill': False, 'decode_fairness': True},
           'cache': {'enabled': True, 'hot_cache_only': True, 'ssd_cache_dir': str(paths['cache']),
                     'hot_cache_max_size': '16GB', 'hot_cache_write_through': False},
           'memory': {'memory_guard_tier': 'aggressive', 'prefill_memory_guard': True},
           'huggingface': {'endpoint': None, 'hf_cache_enabled': False}, 'idle_timeout': {'idle_timeout_seconds': None},
           'sampling': {'temperature': 1.0, 'top_p': 0.95, 'max_tokens': 32768}}
    for (section, key), value in changes.items():
        doc[section][key] = value
    (paths['base'] / 'settings.json').write_text(json.dumps(doc))


def logs(spec, extra_err=''):
    s = ready.native_spec(spec)
    model_id = om.MODEL_IDS[s['q3_family']]
    out = (f"\x1b[33m└─ Version: {om.OMLX_VERSION}\x1b[0m\nSaved CLI arguments to settings.json\n"
           f"Base path: {Path(ready.layout(s)['base']).resolve()}\nMemory guard: on (tier: aggressive)\n"
           'Mode: Multi-model serving (continuous batching + paged SSD cache)\nHot cache: 16.0GB (in-memory)\n')
    err = ('2026-10-31 10:00:00,000 - omlx.model_settings - INFO - Loaded settings for 1 models\n'
           f'2026-10-31 10:00:01,000 - omlx.engine_pool - INFO - Applied model_type override for {model_id}: '
           f'type=llm, engine=batched\n' + extra_err)
    Path(s['server_stdout']).write_text(out)
    Path(s['server_stderr']).write_text(err)


def health(model_id, **changes):
    return {'status': 'healthy', 'default_model': model_id, 'engine_pool': {'model_count': 1, 'loaded_count': 1,
            'final_ceiling': 120 << 30, 'current_model_memory': 16 << 30}, 'mcp': None, **changes}


def status(model_id, requests=0, prompt=0, completion=0, **changes):
    doc = {k: 0 for k in ready.STATUS}
    doc.update(status='ok', version=om.OMLX_VERSION, models_discovered=1, models_loaded=1, default_model=model_id,
               loaded_models=[model_id], total_requests=requests, total_prompt_tokens=prompt,
               total_completion_tokens=completion, model_memory_used_formatted='16GB', model_memory_max_formatted='120GB',
               custom_kernels={'gdn': {'available': True, 'import_error': None},
                               'moe': {'available': True, 'import_error': None}},
               ane_prefill={'enabled': False})
    doc.update(changes)
    return doc


def models_status(spec, **changes):
    s = ready.native_spec(spec)
    entry = {'id': om.MODEL_IDS[s['q3_family']], 'model_path': str(ready.layout(s)['link']), 'loaded': True,
             'is_loading': False, 'pinned': True, 'engine_type': 'batched', 'model_type': 'llm',
             'config_model_type': om.ARTIFACTS[s['q3_family']]['config_model_type'], 'external_engine': None,
             'distributed': False, 'is_helper': False, 'max_context_window': 262144, 'max_tokens': 8192,
             'model_context_length': 262144, 'thinking_default': True, 'preserve_thinking_default': True,
             'source_type': 'local', **changes}
    markitdown = {'id': 'MarkItDown', 'model_path': 'builtin://markitdown', 'engine_type': 'markitdown'}
    return {'model_count': 2, 'loaded_count': 2, 'models': [entry, markitdown]}


class Inputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        checkout(self.root)
        for q3 in om.Q3_FAMILIES:
            snapshot(self.root / 'hub', q3)

    def tearDown(self):
        self.tmp.cleanup()

    def test_model_settings_use_known_fields_and_hf2q_controls(self):
        for q3 in om.Q3_FAMILIES:
            doc = ready.model_settings(q3)
            entry = doc['models'][om.MODEL_IDS[q3]]
            self.assertEqual((doc['version'], list(doc['models'])), (1, [om.MODEL_IDS[q3]]))
            self.assertTrue(set(entry) <= KNOWN_FIELDS)
            self.assertEqual((entry['temperature'], entry['top_k'], entry['repetition_penalty'], entry['max_tokens'],
                              entry['max_context_window'], entry['model_type_override']), (0.0, 0, 1.05, 8192, 262144, 'llm'))
            self.assertFalse(any(entry[k] for k in ('mtp_enabled', 'dflash_enabled', 'vlm_mtp_enabled',
                                                    'turboquant_kv_enabled', 'force_sampling', 'trust_remote_code')))
            self.assertEqual(entry['enable_thinking'], FAMILY[q3] == 'qwen')
            self.assertEqual(entry.get('preserve_thinking'), True if FAMILY[q3] == 'qwen' else None)
            self.assertEqual(ready.document_bytes(doc), ready.document_bytes(ready.model_settings(q3)))

    def test_launch_spec_refuses_unpinned_inputs(self):
        good = {k: spec_for(self.root)[k] for k in ready.LAUNCH_KEYS}
        self.assertEqual(ready.launch_spec(good)['port'], 18254)
        for changes in ({'commit': '0' * 40}, {'version': '0.7.2'}, {'model_revision': '0' * 40},
                        {'model_dir': spec_for(self.root, 'qwen36')['model_dir']}, {'template_sha256': 'x'},
                        {'model_config_sha256': 'A' * 64}, {'run_dir': 'run/omlx'}, {'port': '80'}, {'port': 'x'},
                        {'q3_family': 'deepseek4'}, {'extra': '1'}):
            with self.subTest(changes=changes), self.assertRaises(Invalid):
                ready.launch_spec({**good, **changes})
        with self.assertRaises(Invalid):
            ready.native_spec(spec_for(self.root, server_stdout=str(self.root / 'elsewhere' / 'server.stdout')))

    def test_identity_binds_checkout_venv_and_snapshot(self):
        spec = ready.launch_spec({k: spec_for(self.root)[k] for k in ready.LAUNCH_KEYS})
        with tools():
            identity = ready.verify_identity(spec)
        self.assertEqual(identity['source']['commit'], om.OMLX_COMMIT)
        self.assertEqual(identity['venv']['pins']['mlx']['version'], '0.32.4')
        self.assertFalse(identity['chat_template']['identical_to_hf2q'])
        self.assertEqual(identity['model']['sha256']['config.json'], sha(CONFIG))
        cases = [({}, lambda: (Path(spec['model_dir']) / 'chat_template.jinja').write_bytes(TEMPLATE + b' ')),
                 ({'dirty': b' M omlx/server.py\n'}, None),
                 ({}, lambda: (Path(spec['omlx_dir']) / '.git' / 'HEAD').write_text('ref: refs/heads/main\n'))]
        for kwargs, mutate in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                checkout(root)
                snapshot(root / 'hub', 'qwen38')
                spec = ready.launch_spec({k: spec_for(root)[k] for k in ready.LAUNCH_KEYS})
                if mutate:
                    mutate()
                with tools(**kwargs), self.assertRaises(Invalid):
                    ready.verify_identity(spec)

    def test_venv_inventory_refuses_unpinned_packages(self):
        for packages in ({'mlx': ('0.32.3', None)}, {'mlx_lm': ('0.32.3', {'url': 'x', 'vcs_info': {'commit_id': '0' * 40}})},
                         {'omlx': ('0.7.3rc1', None)}):
            with self.subTest(packages=list(packages)), tempfile.TemporaryDirectory() as tmp:
                src = checkout(Path(tmp).resolve(), packages)
                with self.assertRaises(Invalid):
                    ready.venv_inventory(str(src / '.venv' / 'bin' / 'python'), str(src))
        with self.assertRaises(Invalid):
            ready.venv_inventory('/usr/bin/python3', str(self.root))

    def test_environment_must_be_offline_and_free_of_overrides(self):
        ready.check_environ(ENV)
        for env in ({**ENV, 'OMLX_HOT_CACHE_ONLY': '0'}, {**ENV, 'PYTHONPATH': '/x'}, {**ENV, 'MLX_METAL_DEBUG': '1'},
                    {k: v for k, v in ENV.items() if k != 'HF_HUB_OFFLINE'}):
            with self.subTest(env=sorted(env)), self.assertRaises(Invalid):
                ready.check_environ(env)


class Readiness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        checkout(self.root)
        for q3 in om.Q3_FAMILIES:
            snapshot(self.root / 'hub', q3)

    def tearDown(self):
        self.tmp.cleanup()

    def launch(self, q3):
        """Run the launcher (exec patched), then stand in for the server's own startup writes."""
        spec = spec_for(self.root, q3, run_dir=str(self.root / q3 / 'omlx'),
                        server_stdout=str(self.root / q3 / 'server.stdout'),
                        server_stderr=str(self.root / q3 / 'server.stderr'))
        Path(spec['run_dir']).parent.mkdir()
        argv = [a for k in ready.LAUNCH_KEYS for a in ('--' + k.replace('_', '-'), spec[k])]
        with tools(), patch.dict(os.environ, ENV, clear=True), patch.object(launcher.os, 'execve') as execve:
            launcher.main(argv)
        program, command, env = execve.call_args[0]
        self.assertEqual((program, command, env), (spec['omlx_py'], ready.server_argv(ready.native_spec(spec)), ENV))
        saved_settings(spec)
        logs(spec)
        paths = ready.layout(ready.native_spec(spec))
        packet = {'family': FAMILY[q3], 'native_omlx': spec, 'family_controls': om.CONTROLS[q3], 'server_pid': os.getpid(),
                  'native_omlx_settings_sha256': ready.sha256(paths['base'] / 'model_settings.json'),
                  'native_omlx_launch_sha256': ready.sha256(paths['launch'])}
        return spec, packet

    def serve(self, spec, **docs):
        model_id = om.MODEL_IDS[spec['q3_family']]
        documents = {'/health': health(model_id), '/api/status': status(model_id), '/v1/models/status': models_status(spec),
                     **docs}

        def fetch(port, path, output, name, body=None, limit=None, timeout=None):
            raw = json.dumps(documents[path]).encode()
            Path(output, name).write_bytes(raw)
            return raw
        return patch.object(ready, 'fetch', side_effect=fetch)

    def test_launcher_writes_bound_files_and_readiness_binds_the_run(self):
        for q3 in om.Q3_FAMILIES:
            spec, packet = self.launch(q3)
            out = self.root / q3 / 'client'
            out.mkdir()
            with self.serve(spec), tools():
                record = ready.readiness(f'http://127.0.0.1:{spec["port"]}/v1', MODEL[q3], out, packet)
            self.assertEqual((record['engine_kind'], record['q3_family'], record['label']), ('omlx', q3, om.LABELS[q3]))
            self.assertEqual(record['server_argv'][:6], [spec['omlx_py'], '-I', '-u', '-m', 'omlx.cli', 'serve'])
            self.assertTrue(record['build_info'].startswith(f'omlx {om.OMLX_VERSION} {om.OMLX_COMMIT} venv-freeze-sha256 '))
            self.assertEqual(record['served_model']['model_context_length'], 262144)
            self.assertTrue(record['tokenizer'].startswith('Unavailable:'))
            self.assertEqual(os.readlink(ready.layout(ready.native_spec(spec))['link']), spec['model_dir'])
            for name in ('omlx-health.raw', 'omlx-status.raw', 'omlx-models-status.raw', 'omlx-startup.stdout',
                         'omlx-startup.stderr', ready.CLOSING_INPUTS):
                self.assertTrue((out / name).is_file())

    def test_readiness_refuses_launch_document_and_settings_drift(self):
        spec, good = self.launch('qwen36')
        model_id, base = MODEL['qwen36'], f'http://127.0.0.1:{spec["port"]}/v1'
        cases = [({'base': base.replace('18254', '18255')}, {}), ({'packet': {'family_controls': om.CONTROLS['qwen38']}}, {}),
                 ({'packet': {'family_controls': om.CONTROLS['gemma4']}}, {}), ({'model': MODEL['qwen38']}, {}),
                 ({'packet': {'native_omlx_settings_sha256': 'e' * 64}}, {}),
                 ({'packet': {'native_omlx_launch_sha256': 'e' * 64}}, {}), ({'packet': {'server_pid': 1}}, {}),
                 ({'title': '/x/.venv/bin/python -m omlx.cli serve'}, {}),
                 ({}, {'/health': health(model_id, status='loading')}), ({}, {'/health': health(model_id, mcp={})}),
                 ({}, {'/api/status': status(model_id, version='0.7.2')}),
                 ({}, {'/api/status': status(model_id, requests=1)}),
                 ({}, {'/api/status': status(model_id, custom_kernels={'gdn': {'available': False, 'import_error': 'x'}})}),
                 ({}, {'/v1/models/status': models_status(spec, engine_type='vlm', model_type='vlm')}),
                 ({}, {'/v1/models/status': models_status(spec, model_path='/elsewhere')}),
                 ({}, {'/v1/models/status': models_status(spec, max_context_window=32768)}),
                 ({'settings': {('cache', 'hot_cache_only'): False}}, {}),
                 ({'settings': {('server', 'sse_keepalive_mode'): 'chunk'}}, {}),
                 ({'settings': {('huggingface', 'hf_cache_enabled'): True}}, {}),
                 ({'log': "WARNING - Failed to load settings for model 'x': bad\n"}, {})]
        for change, docs in cases:
            with self.subTest(change=change, docs=list(docs)):
                packet = {**good, **change.get('packet', {})}
                saved_settings(spec, **change.get('settings', {}))
                logs(spec, change.get('log', ''))
                with self.serve(spec, **docs), tools(change.get('title', 'omlx-server')), self.assertRaises(Invalid):
                    ready.readiness(change.get('base', base), change.get('model', model_id), self.root, packet)
        saved_settings(spec)
        logs(spec)
        (ready.layout(ready.native_spec(spec))['models'] / 'other').mkdir()
        with self.serve(spec), tools(), self.assertRaises(Invalid):
            ready.readiness(base, model_id, self.root, good)

    def test_render_proof_and_closing_bind_every_turn_and_the_session_counters(self):
        spec, packet = self.launch('qwen38')
        out = self.root / 'coordinator'
        out.mkdir()
        with self.serve(spec), tools():
            ready.readiness(f'http://127.0.0.1:{spec["port"]}/v1', MODEL['qwen38'], out, packet)
        for agent in ('agent-0', 'agent-1'):
            receipts = []
            for turn in (0, 1):
                (out / agent / f'turn-{turn}').mkdir(parents=True)
                raw = json.dumps(native(turn), ensure_ascii=False, separators=(',', ':')).encode()
                (out / agent / f'turn-{turn}' / 'request.json').write_bytes(raw)
                receipts.append({'turn': turn, 'request_sha256': sha(raw),
                                 'usage': {'prompt_tokens': 3, 'completion_tokens': 5}})
            (out / agent / 'summary.json').write_text(json.dumps({'turns': receipts}))
        bodies = []

        def fetch(port, path, output, name, body=None, limit=None, timeout=None):
            bodies.append((path, body))
            doc = (status(MODEL['qwen38'], 4, 12, 20) if path == '/api/status'
                   else {'count': 3, 'max_model_len': 262144, 'tokens': [1, 2, 3], 'token_strs': None})
            raw = json.dumps(doc).encode()
            Path(output, name).write_bytes(raw)
            return raw
        with patch.object(ready, 'fetch', side_effect=fetch):
            closing = ready.closing('http://127.0.0.1:18254/v1', out)
        self.assertEqual(sorted(closing['render']), ['agent-0', 'agent-1'])
        self.assertEqual(closing['render']['agent-1']['turns'][1]['tokens_sha256'], sha(b'1,2,3'))
        self.assertEqual(bodies[0][0], '/v1/tokenize')
        self.assertEqual(set(bodies[0][1]), set(ready.TOKENIZE_FIELDS))
        self.assertEqual(bodies[1][1]['chat_template_kwargs'], {'enable_thinking': True, 'preserve_thinking': True})
        self.assertTrue((out / 'agent-1' / 'turn-0' / 'omlx-tokenize.raw').is_file())
        self.assertEqual(closing['status']['total_requests'], 4)
        bad = [lambda p, b: status(MODEL['qwen38'], 5, 12, 20) if p == '/api/status' else None,  # a hidden request
               lambda p, b: status(MODEL['qwen38'], 4, 13, 20) if p == '/api/status' else None,
               lambda p, b: None if p == '/api/status' else {'count': 4, 'max_model_len': 262144,
                                                              'tokens': [1, 2, 3, 4], 'token_strs': None},
               lambda p, b: None if p == '/api/status' else {'count': 3, 'max_model_len': 262144, 'tokens': [1, 2, 3]}]
        for override in bad:
            def fetch_bad(port, path, output, name, body=None, limit=None, timeout=None, override=override):
                raw = json.dumps(override(path, body) or json.loads(fetch(port, path, output, name, body))).encode()
                Path(output, name).write_bytes(raw)
                return raw
            with self.subTest(override=override), patch.object(ready, 'fetch', side_effect=fetch_bad), \
                    self.assertRaises(Invalid):
                ready.closing('http://127.0.0.1:18254/v1', out)
        (ready.layout(ready.native_spec(spec))['base'] / 'model_settings.json').write_text('{}')
        with patch.object(ready, 'fetch', side_effect=fetch), self.assertRaises(Invalid):
            ready.closing('http://127.0.0.1:18254/v1', out)

    def test_campaign_identity_and_launch_problems(self):
        spec = spec_for(self.root)
        plan = {'variables': {'OMLX_COMMIT': om.OMLX_COMMIT, 'OMLX_VERSION': om.OMLX_VERSION, 'OMLX_DIR': spec['omlx_dir'],
                              'OMLX_PY': spec['omlx_py'], 'MODEL_DIR': spec['model_dir']}}
        identity = ready.campaign_identity(plan)
        self.assertTrue(identity['peer_build_info'].startswith(f'omlx {om.OMLX_VERSION} {om.OMLX_COMMIT} venv-freeze-sha256 '))
        self.assertEqual(identity['native_runtime']['model']['weights'],
                         ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'])
        missing = ready.campaign_identity({'variables': {'OMLX_PY': '/absent/bin/python', 'MODEL_DIR': '/absent'}})
        self.assertEqual(missing, {'peer_build_info': None, 'native_runtime': {'venv': None, 'model': None}})
        for q3 in om.Q3_FAMILIES:
            artifact = {'path': spec_for(self.root, q3)['model_dir'] + '/config.json', 'sha256': 'c' * 64}
            self.assertEqual(ready.launch_problems('peer', artifact, 'd' * 64), [])
        artifact = {'path': spec['model_dir'] + '/config.json', 'sha256': 'c' * 64}
        self.assertEqual(len(ready.launch_problems('peer', {**artifact, 'path': '/m/Qwen3.8-27B.gguf'}, 'd' * 64)), 1)
        self.assertEqual(len(ready.launch_problems('peer', artifact, 'c' * 64)), 1)


if __name__ == '__main__':
    unittest.main()
