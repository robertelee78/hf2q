"""Pure controls for the TensorFold arm's identity, readiness, /tokenize proof and closing record.

No TensorFold process: snapshots are temporary HF-cache layouts, HTTP documents and the server argv are
patched. Document shapes follow /opt/TensorFold 1157ea5b: /health and /v1/models (server/status_routes.zig:
14-125), the benchmark manifest (server/benchmark_manifest.zig:206-244), /tokenize (server/tokens.zig:26-68),
the serving line (server/serve.zig:247-250) and the qwen27 startup lines (native/qwen27_served.zig:345,383,407;
native/cache_fit.zig:113).
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import native_peer_tensorfold as tf
import native_peer_tensorfold_ready as ready
from qwen38_n1_verdict import Invalid
from test_native_peer_tensorfold import CREATED, M, native

TEMPLATE = b'{%- set reasoning_effort = reasoning_effort|default("xhigh") %}\n'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(hub, repo, revision, files, weights):
    """An HF cache snapshot: regular files plus weights linked to blobs named by their sha256."""
    base = Path(hub) / ('models--' + repo.replace('/', '--'))
    path = base / 'snapshots' / revision
    path.mkdir(parents=True)
    (base / 'blobs').mkdir()
    for name, data in files.items():
        (path / name).write_bytes(data)
    for name, data in weights.items():
        blob = base / 'blobs' / sha(data)
        blob.write_bytes(data)
        os.symlink(blob, path / name)
    return path


def target(hub):
    index = json.dumps({'weight_map': {'a': 'model-00001-of-00002.safetensors',
                                       'b': 'model-00002-of-00002.safetensors'}}).encode()
    return snapshot(hub, tf.TARGET_REPO, tf.TARGET_REVISION,
                    {'config.json': b'{"model_type":"qwen3_5"}', 'tokenizer.json': b'{"vocab":{}}',
                     'tokenizer_config.json': b'{}', 'chat_template.jinja': TEMPLATE, 'README.md': b'card',
                     'model.safetensors.index.json': index},
                    {'model-00001-of-00002.safetensors': b'w1', 'model-00002-of-00002.safetensors': b'w2'})


def drafter(hub):
    return snapshot(hub, tf.DRAFTER_REPO, tf.DRAFTER_REVISION, {'config.json': b'{"model_type":"dflash"}'},
                    {'model.safetensors': b'd1'})


def spec_for(root, variant='none', **changes):
    spec = {'binary': str(root / 'bin' / 'tensorfold-native'), 'binary_sha256': sha(b'bin'), 'commit': tf.TF_COMMIT,
            'version': tf.TF_VERSION, 'target_dir': str(root / 'hub' / 'models--TensorFold--Qwen3.8-27B-MLX-4bit'
                                                        / 'snapshots' / tf.TARGET_REVISION),
            'target_revision': tf.TARGET_REVISION, 'target_config_sha256': sha(b'{"model_type":"qwen3_5"}'),
            'drafter': variant, 'port': '18252', 'server_stdout': str(root / 'run' / 'server.stdout'),
            'server_stderr': str(root / 'run' / 'server.stderr')}
    if variant == 'dflash2':
        spec.update(drafter_dir=str(root / 'hub' / 'models--z-lab--Qwen3.8-27B-DFlash2' / 'snapshots'
                                    / tf.DRAFTER_REVISION), drafter_revision=tf.DRAFTER_REVISION)
    spec.update(changes)
    return spec


def health(**changes):
    return {'status': 'ok', 'model': M, 'model_ids': [M], 'max_batch_size': 4, 'policy': 'lanes', 'warming': False,
            'memory': {'active': 1, 'cache': 2, 'peak': 3, 'limit': 4}, 'live': {'running': 0}, 'version': tf.TF_VERSION,
            'backend': 'metal', 'device': 'Apple M5 Max', **changes}


def models(**changes):
    entry = {'id': M, 'object': 'model', 'created': CREATED, 'owned_by': 'tensorfold', 'context_length': 262144,
             'max_model_len': 262144, 'reasoning': {'default_enabled': True,
                                                    'supported_efforts': ['xhigh', 'medium', 'low', 'none']}, **changes}
    return {'object': 'list', 'data': [entry]}


def manifest(facts, **model):
    return {'schema_version': 1, 'runtime': {'tensorfold_version': tf.TF_VERSION, 'backend': 'mlx',
                                             'python_version': 'none', 'platform': 'macos'},
            'model': {'revision': tf.TARGET_REVISION, 'config_sha256': facts['config_sha256'],
                      'tokenizer_sha256': facts['tokenizer_sha256'], **model},
            'hardware': {'memory_type': 'unified', 'gpus': [], 'gpu_count': None, 'source': 'partial'}}


def logs(spec, drafts=None, streams=4, tokens=262144, suffix=None, extra_out='', extra_err=''):
    dflash2 = spec['drafter'] == 'dflash2'
    drafts = (f"DFlash2 ({spec['drafter_dir']}) + copies" if dflash2 else 'copies only') if drafts is None else drafts
    suffix = (', DFlash2 drafts for one of them' if dflash2 else '') if suffix is None else suffix
    out = ('[tensorfold] prompt chunks of up to 512 tokens, cut at replies 256+ tokens apart\n'
           f"[tensorfold] serving {M} at http://127.0.0.1:{spec['port']}/v1 (sampling: greedy; drafts: {drafts}; "
           'context: 262144; loaded in 41.2s)\n' + extra_out)
    err = ('info: qwen27: prompt capacity=512 rows, startup stream and scratch reserve=21474836480 bytes\n'
           'info: qwen27: lane rounds, 16-row window\n'
           'info: prompt cache: 0.0 GiB from 0.0 GiB free under the 70% cap (95.2 GiB in use once loaded, 128 GiB '
           'of RAM, 8 GiB kept for prompt buffers)\n'
           f'info: Qwen3.8-27B: {streams} streams of {tokens} tokens{suffix}\n' + extra_err)
    return out, err


class Snapshot(unittest.TestCase):
    def test_tokenizer_hash_uses_tensorfold_framing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tokenizer.json').write_bytes(b'tok')
            (root / 'chat_template.jinja').write_bytes(TEMPLATE)
            expected = hashlib.sha256(b'tokenizer.json\x003\x00tok' + b'chat_template.jinja\x00'
                                      + str(len(TEMPLATE)).encode() + b'\x00' + TEMPLATE).hexdigest()
            self.assertEqual(ready.tokenizer_sha256(root), expected)
            self.assertIsNone(ready.tokenizer_sha256(root / 'absent'))

    def test_checkpoint_records_blobs_and_refuses_shard_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = target(Path(tmp) / 'hub')
            facts = ready.checkpoint(path)
            self.assertEqual((facts['revision'], facts['chat_template_sha256']), (tf.TARGET_REVISION, sha(TEMPLATE)))
            self.assertEqual(facts['files']['model-00001-of-00002.safetensors']['blob'], sha(b'w1'))
            self.assertIsNone(facts['files']['config.json']['blob'])
            self.assertTrue(facts['weights_sha256'].startswith('Unavailable:'))
            self.assertTrue(ready.snapshot_layout(path, tf.TARGET_REPO, tf.TARGET_REVISION))
            self.assertFalse(ready.snapshot_layout(path, tf.DRAFTER_REPO, tf.TARGET_REVISION))
            (path / 'model.safetensors.index.json').write_text('{"weight_map":{"a":"model-00001-of-00002.safetensors"}}')
            with self.assertRaises(Invalid):
                ready.checkpoint(path)


class Spec(unittest.TestCase):
    def test_native_spec_refuses_identity_and_drafter_drift(self):
        root = Path('/x')
        self.assertEqual(ready.native_spec(spec_for(root))['port'], 18252)
        ready.native_spec(spec_for(root, 'dflash2'))
        bad = [spec_for(root, commit='0' * 40), spec_for(root, version='1.2.1'), spec_for(root, binary='/x/bin/llama'),
               spec_for(root, binary='bin/tensorfold'), spec_for(root, binary_sha256='A' * 64),
               spec_for(root, target_config_sha256=''), spec_for(root, target_revision='0' * 40),
               spec_for(root, target_dir='/x/hub/models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/main'),
               spec_for(root, target_dir='/x/hub/models--Qwen--Qwen3.8-27B/snapshots/' + tf.TARGET_REVISION),
               spec_for(root, drafter='dflash'), spec_for(root, drafter='dflash2'),
               {**spec_for(root), 'drafter_dir': '/x/d', 'drafter_revision': tf.DRAFTER_REVISION},
               spec_for(root, 'dflash2', drafter_revision='0' * 40), spec_for(root, 'dflash2', drafter_dir='/x/d'),
               spec_for(root, port='x'), spec_for(root, server_stdout='/x/run/server.log'),
               spec_for(root, server_stderr='/y/server.stderr'), {**spec_for(root), 'extra': ''},
               {**spec_for(root), 'port': 18252}]
        for spec in bad:
            with self.subTest(spec={k: v for k, v in spec.items() if v != spec_for(root).get(k)}), self.assertRaises(Invalid):
                ready.native_spec(spec)

    def test_server_argv_differs_only_by_the_drafter(self):
        root = Path('/x')
        plain, drafted = (ready.server_argv(ready.native_spec(spec_for(root, v))) for v in ('none', 'dflash2'))
        self.assertEqual(drafted[:len(plain)], plain)
        self.assertEqual(drafted[len(plain):], ['--drafter', spec_for(root, 'dflash2')['drafter_dir'], '--drafter-bits', '4'])
        self.assertFalse({'--prompt-cache-gib', '--no-drafts', '--learn', '--recall', '--idle-unload', '--loop-guard',
                          '--reasoning-effort', '--thinking-budget', '--no-thinking', '--kv-cache'} & set(drafted))


class Documents(unittest.TestCase):
    def test_health_models_and_manifest_bind_the_served_identity(self):
        self.assertEqual(ready.validate_health(health())['memory_bytes']['peak'], 3)
        ready.validate_health(health(warming=True), ready=False)
        ready.validate_models(models())
        facts = {'config_sha256': 'c' * 64, 'tokenizer_sha256': 'd' * 64}
        ready.validate_manifest(manifest(facts), facts)
        for bad in (health(max_batch_size=1), health(model='other'), health(model_ids=[M, 'alias']), health(version='1.2.1'),
                    health(backend='cuda'), health(warming=True), health(memory={'active': -1}), health(extra=1),
                    {'status': 'ok'}):
            with self.subTest(health=str(bad)[:60]), self.assertRaises(Invalid):
                ready.validate_health(bad)
        doc = models()
        for bad in (models(context_length=131072), models(max_model_len=262145), models(owned_by='x'),
                    models(reasoning={'default_enabled': True, 'default_effort': 'xhigh',
                                      'supported_efforts': ['xhigh', 'medium', 'low', 'none']}),
                    models(reasoning={'default_enabled': False, 'supported_efforts': ['xhigh', 'medium', 'low', 'none']}),
                    models(id='Qwen3.8 27B'), {**doc, 'data': doc['data'] * 2}, {**doc, 'extra': 1}):
            with self.subTest(models=str(bad)[:60]), self.assertRaises(Invalid):
                ready.validate_models(bad)
        good = manifest(facts)
        for bad in (manifest(facts, revision='0' * 40), manifest(facts, tokenizer_sha256='e' * 64),
                    manifest(facts, config_sha256='e' * 64), {**good, 'rank_count': 2},
                    {**good, 'runtime': {**good['runtime'], 'tensorfold_version': '1.2.1'}},
                    {k: v for k, v in good.items() if k != 'model'},
                    {**good, 'model': {k: v for k, v in good['model'].items() if k != 'revision'}}):
            with self.subTest(manifest=str(bad)[:60]), self.assertRaises(Invalid):
                ready.validate_manifest(bad, facts)

    def test_startup_log_binds_listener_context_streams_drafter_and_cache(self):
        for variant in ('none', 'dflash2'):
            spec = ready.native_spec(spec_for(Path('/x'), variant))
            facts = ready.startup(*logs(spec), spec)
            self.assertEqual((facts['streams'], facts['stream_tokens'], facts['dflash2_loaded'], facts['prompt_rows']),
                             (4, 262144, variant == 'dflash2', 512))
            self.assertEqual((facts['prompt_cache']['budget_gib'], facts['prompt_cache']['enabled']), (0.0, False))
            self.assertEqual(facts['loop_policy'], ['lane rounds, 16-row window'])
        plain = ready.native_spec(spec_for(Path('/x')))
        drafted = ready.native_spec(spec_for(Path('/x'), 'dflash2'))
        bad = [(plain, logs(plain, drafts='DFlash2 (/x/d)')), (plain, logs(plain, suffix=', DFlash2 drafts for one of them')),
               (drafted, logs(drafted, drafts='copies only')), (drafted, logs(drafted, suffix='')),
               (plain, logs(plain, streams=3)), (plain, logs(plain, tokens=131072)),
               (plain, logs(plain, extra_out='[tensorfold] idle unload on: the engine closes after 60 s idle\n')),
               (plain, logs(plain, extra_err='info: prompt cache: 4.0 GiB from 4.0 GiB free, past the cap by '
                                             '--prompt-cache-over-cap\n')),
               (plain, (logs(plain)[0].replace('sampling: greedy', 'sampling: temperature 0.6'), logs(plain)[1])),
               (plain, (logs(plain)[0].replace('context: 262144', 'context: 32768'), logs(plain)[1])),
               (plain, (logs(plain)[0].replace('127.0.0.1:18252', '127.0.0.1:9'), logs(plain)[1])),
               (plain, (logs(plain)[0] * 2, logs(plain)[1])),
               (plain, (logs(plain)[0], logs(plain)[1].replace('info: prompt cache', 'info: kept states')))]
        for spec, (out, err) in bad:
            with self.subTest(out=out[-120:], err=err[-80:]), self.assertRaises(Invalid):
                ready.startup(out, err, spec)


class Readiness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'bin').mkdir()
        (self.root / 'bin' / 'tensorfold-native').write_bytes(b'bin')
        (self.root / 'run').mkdir()
        (self.root / 'client').mkdir()
        self.target, self.drafter = target(self.root / 'hub'), drafter(self.root / 'hub')
        self.template = patch.object(tf, 'TEMPLATE_SHA256', sha(TEMPLATE))
        self.template.start()

    def tearDown(self):
        self.template.stop()
        self.tmp.cleanup()

    def serve(self, spec, documents, argv=None):
        out, err = logs(ready.native_spec(spec))
        Path(spec['server_stdout']).write_text(out)
        Path(spec['server_stderr']).write_text(err)

        def fetch(port, path, output, name, body=None, limit=None, timeout=None):
            raw = json.dumps(documents[path]).encode()
            Path(output, name).write_bytes(raw)
            return raw
        expected = ' '.join(ready.server_argv(ready.native_spec(spec))) if argv is None else argv
        return patch.object(ready, 'fetch', side_effect=fetch), patch.object(ready, 'observed_argv', return_value=expected)

    def packet(self, variant):
        return {'native_tensorfold': spec_for(self.root, variant), 'family_controls': copy.deepcopy(tf.CONTROLS[variant]),
                'server_pid': 4242}

    def documents(self, **changes):
        facts = ready.checkpoint(self.target)
        return {'/health': health(), '/v1/models': models(), '/v1/benchmark/manifest': manifest(facts), **changes}

    def test_readiness_archives_and_binds_the_run(self):
        for variant in ('none', 'dflash2'):
            packet, out = self.packet(variant), self.root / 'client' / variant
            out.mkdir()
            fetching, argv = self.serve(packet['native_tensorfold'], self.documents())
            with fetching, argv:
                record = ready.readiness('http://127.0.0.1:18252/v1', M, out, packet)
            self.assertEqual((record['engine_kind'], record['variant'], record['label']), ('tensorfold', variant, tf.LABELS[variant]))
            self.assertEqual(record['build_info'], f"tensorfold {tf.TF_VERSION} {tf.TF_COMMIT} binary-sha256 {sha(b'bin')}")
            self.assertEqual(record['identity']['drafter'] is None, variant == 'none')
            self.assertTrue(record['tokenizer'].startswith('Unavailable:'))
            for name in ('tensorfold-health.raw', 'tensorfold-models.raw', 'tensorfold-manifest.raw',
                         'tensorfold-startup.stdout', 'tensorfold-startup.stderr'):
                self.assertTrue((out / name).is_file())

    def test_readiness_refuses_launch_document_and_snapshot_drift(self):
        good = self.packet('none')
        cases = [('http://127.0.0.1:18253/v1', good, {}, None, None),
                 ('http://127.0.0.1:18252/v1', {**good, 'family_controls': tf.CONTROLS['dflash2']}, {}, None, None),
                 ('http://127.0.0.1:18252/v1', {**good, 'native_tensorfold': spec_for(self.root, binary_sha256='e' * 64)},
                  {}, None, None),
                 ('http://127.0.0.1:18252/v1', {**good, 'native_tensorfold': spec_for(self.root, target_config_sha256='e' * 64)},
                  {}, None, None),
                 ('http://127.0.0.1:18252/v1', good, {}, 'tensorfold serve elsewhere', None),
                 ('http://127.0.0.1:18252/v1', good, {'/health': health(max_batch_size=2)}, None, None),
                 ('http://127.0.0.1:18252/v1', good, {'/v1/models': models(context_length=32768)}, None, None),
                 ('http://127.0.0.1:18252/v1', good, {'/v1/benchmark/manifest': manifest(
                     {'config_sha256': 'c' * 64, 'tokenizer_sha256': 'd' * 64})}, None, None),
                 ('http://127.0.0.1:18252/v1', good, {}, None, 'template')]
        for base, packet, changes, argv, mutate in cases:
            with self.subTest(base=base, changes=list(changes), argv=argv, mutate=mutate):
                if mutate == 'template':
                    (self.target / 'chat_template.jinja').write_bytes(TEMPLATE + b'x')
                fetching, observed = self.serve(packet['native_tensorfold'], self.documents(**changes), argv)
                with fetching, observed, self.assertRaises(Invalid):
                    ready.readiness(base, M, self.root / 'client', packet)
        with self.assertRaises(Invalid):
            ready.readiness('http://127.0.0.1:18252/v1', 'Qwen3.8 27B', self.root / 'client', good)

    def test_render_proof_binds_every_turn_to_its_usage(self):
        out = self.root / 'agent'
        receipts = []
        for turn in (0, 1):
            (out / f'turn-{turn}').mkdir(parents=True)
            raw = json.dumps(native(turn), ensure_ascii=False, separators=(',', ':')).encode()
            (out / f'turn-{turn}' / 'request.json').write_bytes(raw)
            receipts.append({'turn': turn, 'request_sha256': sha(raw), 'usage': {'prompt_tokens': 3}})
        bodies = []

        def fetch(port, path, output, name, body=None, limit=None, timeout=None):
            bodies.append((path, body))
            raw = json.dumps({'count': 3, 'max_model_len': 262144, 'tokens': [1, 2, 3]}).encode()
            Path(output, name).write_bytes(raw)
            return raw
        with patch.object(ready, 'fetch', side_effect=fetch):
            proof = ready.render_proof('http://127.0.0.1:18252/v1', out, receipts)
        self.assertEqual([r['count'] for r in proof['turns']], [3, 3])
        self.assertEqual(proof['turns'][0]['tokens_sha256'], sha(b'1,2,3'))
        self.assertEqual(bodies[0][0], '/tokenize')
        self.assertEqual(set(bodies[0][1]), set(ready.TOKENIZE_FIELDS))
        self.assertEqual(bodies[1][1]['reasoning_effort'], 'xhigh')
        self.assertTrue((out / 'turn-1' / 'tensorfold-tokenize.raw').is_file())
        for bad in ([{**receipts[0], 'usage': {'prompt_tokens': 4}}], [{**receipts[0], 'request_sha256': '0' * 64}]):
            with self.subTest(bad=bad), patch.object(ready, 'fetch', side_effect=fetch), self.assertRaises(Invalid):
                ready.render_proof('http://127.0.0.1:18252/v1', out, bad)

    def test_closing_and_campaign_identity(self):
        def fetch(port, path, output, name, body=None, limit=None, timeout=None):
            raw = json.dumps(health(warming=True)).encode()
            Path(output, name).write_bytes(raw)
            return raw
        with patch.object(ready, 'fetch', side_effect=fetch):
            closing = ready.closing('http://127.0.0.1:18252/v1', self.root / 'client')
        self.assertEqual(closing['health']['memory_bytes']['peak'], 3)
        plan = {'variables': {'TF_COMMIT': tf.TF_COMMIT, 'TF_BIN_SHA256': 'a' * 64, 'TF_VERSION': tf.TF_VERSION,
                              'TARGET_DIR': str(self.target), 'DRAFTER_DIR': str(self.drafter)}}
        identity = ready.campaign_identity(plan)
        self.assertEqual(identity['peer_build_info'], f'tensorfold {tf.TF_VERSION} {tf.TF_COMMIT} binary-sha256 ' + 'a' * 64)
        self.assertEqual(identity['native_runtime']['drafter']['weights'], ['model.safetensors'])
        missing = ready.campaign_identity({'variables': {'TF_BIN_SHA256': None, 'TARGET_DIR': '/absent'}})
        self.assertEqual(missing, {'peer_build_info': None, 'native_runtime': {'target': None, 'drafter': None}})
        artifact = {'path': str(self.target / 'config.json'), 'sha256': 'c' * 64}
        self.assertEqual(ready.launch_problems('peer', artifact, 'd' * 64), [])
        self.assertEqual(len(ready.launch_problems('peer', {**artifact, 'path': '/m/Qwen3.8-27B.gguf'}, 'd' * 64)), 1)
        self.assertEqual(len(ready.launch_problems('peer', artifact, 'c' * 64)), 1)


if __name__ == '__main__':
    unittest.main()
