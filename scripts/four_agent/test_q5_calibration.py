"""Q5 calibration supervisor controls: synthetic samples/rows/cases only; no host commands or launches."""
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unittest

import q5_admission
import q5_artifacts
import q5_calibration
import q5_case
import q5_host
import q5_negative_control as nc
import q5_policy as p
import q5_preflight as pre
import q5_runtime

MIB, GIB = 1 << 20, 1 << 30
NC_LATENCY = 146_000_000  # accepted negative control: sample->signal->exit about 146 ms
HERE = Path(__file__).resolve().parent
VM_STAT = '''Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  5264938.
Pages active:                                1138994.
Pages inactive:                              1096022.
Pages speculative:                             45030.
Pages wired down:                             389925.
"Translation faults":                     8769540764.
File-backed pages:                            889466.
Anonymous pages:                             1390580.
Pages stored in compressor:                   788580.
Pages occupied by compressor:                 392967.
Pages tagged:                                 311550.
'''


def host(free_cached=60 * GIB, swap=20 * MIB, level=1, compressor=GIB):
    return {'pressure_level': level, 'pressure': q5_host.PRESSURE_NAMES.get(level, 'unknown'), 'memorystatus_level': 50,
            'swap': {'used_bytes': swap}, 'vm': {'free_plus_cached_bytes': free_cached, 'compressor_bytes': compressor,
                                                 'wired_bytes': 5 * GIB, 'anonymous_bytes': 9 * GIB}}


def sample(index, h='default', footprint=None, source='footprint'):
    group = None if footprint is None else {'members': [100], 'footprint_bytes': footprint,
                                            'footprint_source': source, 'rss_bytes': footprint // 2}
    return {'index': index, 'start_ns': index * 500_000_000, 'end_ns': index * 500_000_000 + 1,
            'host': host() if h == 'default' else h, 'group': group}


def policy(**overrides):
    return p.effective_policy(overrides)


BASELINE, _ = p.baseline_from([sample(i) for i in range(6)], policy())


class StopRules(unittest.TestCase):
    def first_stop(self, samples, **overrides):
        monitor = p.Monitor(BASELINE, policy(**overrides))
        for item in samples:
            decision = monitor.feed(item)
            if decision is not None:
                return decision
        return None

    def test_quiet_run_never_stops(self):
        self.assertIsNone(self.first_stop([sample(i, footprint=30 * GIB) for i in range(20)]))

    def test_only_critical_pressure_stops_and_warning_is_recorded(self):
        monitor = p.Monitor(BASELINE, policy())
        for i in range(40):  # sustained warning while a large artifact faults in
            self.assertIsNone(monitor.feed(sample(i, host(level=2))))
        self.assertEqual(monitor.summary()['pressure_counts'], {'warning': 40})
        stop = self.first_stop([sample(0), sample(1, host(level=2)), sample(2, host(level=4))])
        self.assertEqual((stop['reason'], stop['sample_index']), ('pressure_critical', 2))

    def test_swap_tolerance_is_512_mib_exclusive(self):
        at = BASELINE['swap_used_bytes'] + 512 * MIB
        self.assertIsNone(self.first_stop([sample(0, host(swap=at))]))
        self.assertEqual(self.first_stop([sample(0, host(swap=at + 1))])['reason'], 'swap_growth')

    def test_free_cached_floor_is_two_gib(self):
        self.assertIsNone(self.first_stop([sample(0, host(free_cached=2 * GIB))]))
        self.assertEqual(self.first_stop([sample(0, host(free_cached=2 * GIB - 1))])['reason'], 'free_cached_floor')

    def test_footprint_cap_only_when_configured(self):
        self.assertIsNone(self.first_stop([sample(0, footprint=120 * GIB)]))
        stop = self.first_stop([sample(0, footprint=40 * GIB), sample(1, footprint=40 * GIB + 1)], footprint_cap_bytes=40 * GIB)
        self.assertEqual((stop['reason'], stop['footprint_at_stop_bytes']), ('footprint_cap', 40 * GIB + 1))

    def test_priority_names_first_rule_but_records_all(self):
        stop = self.first_stop([sample(0, host(level=4, free_cached=GIB, swap=GIB), footprint=99 * GIB)], footprint_cap_bytes=GIB)
        self.assertEqual(stop['reason'], 'pressure_critical')
        self.assertEqual([v[0] for v in stop['all_violations']], list(p.MEMORY_STOPS))

    def test_consecutive_blind_samples_fail_safe(self):
        blind = lambda i: sample(i, None)
        self.assertIsNone(self.first_stop([blind(0), blind(1), sample(2), blind(3), blind(4)]))
        self.assertEqual(self.first_stop([sample(0), blind(1), blind(2), blind(3)])['reason'], 'sampler_blind')

    def test_enforced_cap_without_real_footprint_is_blind(self):
        degraded = [sample(i, footprint=GIB, source='ps-rss-fallback') for i in range(3)]
        self.assertEqual(self.first_stop(degraded, footprint_cap_bytes=40 * GIB)['reason'], 'sampler_blind')
        self.assertIsNone(self.first_stop(degraded))

    def test_one_stop_then_post_stop_peak_tracked(self):
        monitor = p.Monitor(BASELINE, policy(footprint_cap_bytes=4 * GIB))
        self.assertIsNotNone(monitor.feed(sample(0, footprint=5 * GIB)))
        self.assertIsNone(monitor.feed(sample(1, host(level=4), footprint=6 * GIB)))
        summary = monitor.summary()
        self.assertEqual((summary['peak_footprint_bytes'], summary['post_stop_peak_footprint_bytes']), (6 * GIB, 6 * GIB))
        self.assertEqual(summary['pressure_counts'], {'normal': 1, 'critical': 1})

    def test_summary_extrema_and_deltas(self):
        monitor = p.Monitor(BASELINE, policy())
        for i, (free, swap, comp) in enumerate(((50 * GIB, 20 * MIB, GIB), (30 * GIB, 21 * MIB, 3 * GIB), (40 * GIB, 20 * MIB, 2 * GIB))):
            monitor.feed(sample(i, host(free_cached=free, swap=swap, compressor=comp), footprint=(i + 1) * GIB))
        s = monitor.summary()
        self.assertEqual((s['min_free_plus_cached_bytes'], s['max_swap_used_bytes'], s['max_compressor_bytes']),
                         (30 * GIB, 21 * MIB, 3 * GIB))
        self.assertEqual(s['deltas']['swap_used_bytes'], MIB)
        self.assertEqual(s['deltas']['free_plus_cached_bytes'], -30 * GIB)
        self.assertEqual(s['peak_footprint_bytes'], 3 * GIB)

    def test_baseline_refusals(self):
        pol = policy()
        self.assertIsNone(p.baseline_from([sample(i) for i in range(5)], pol)[0])
        self.assertIsNone(p.baseline_from([sample(i) for i in range(5)] + [sample(5, None)], pol)[0])
        self.assertIn('critical', p.baseline_from([sample(i, host(level=4)) for i in range(6)], pol)[1])
        self.assertIsNotNone(p.baseline_from([sample(i, host(level=2)) for i in range(6)], pol)[0])
        self.assertIn('floor', p.baseline_from([sample(i, host(free_cached=2 * GIB)) for i in range(6)], pol)[1])
        mixed = [sample(i, host(swap=(20 + i) * MIB)) for i in range(6)]
        self.assertEqual(p.baseline_from(mixed, pol)[0]['swap_used_bytes'], 25 * MIB)

    def test_policy_rejects_unknown_and_invalid(self):
        for bad in ({'footprint_cap': 1}, {'free_cached_floor_bytes': 0}, {'interval_seconds': 5},
                    {'footprint_cap_bytes': 1.5}, {'max_blind_samples': 2.0}):
            with self.assertRaises(p.PolicyError):
                p.effective_policy(bad)


class Verdicts(unittest.TestCase):
    def summary(self, **changes):
        monitor = p.Monitor(BASELINE, policy(), NC_LATENCY, 'nc')
        for i in range(4):
            monitor.feed(sample(i, footprint=30 * GIB))
        return {**monitor.summary(), **changes}

    ok_client = {'exit_code': 0, 'status_ok': True}

    def classify(self, stop, client, summary, proven=True):
        return p.classify(stop, client, summary, proven, policy())

    def test_pass_needs_min_above_the_raised_floor_and_a_measured_latency(self):
        raised = self.summary()
        raised['effective_floor'] = {**raised['effective_floor'], 'effective_floor_bytes': 70 * GIB}
        verdict = self.classify(None, self.ok_client, raised)  # min free+cached is 60 GiB
        self.assertEqual(verdict['verdict'], 'CalibrationFailed')
        self.assertIn('effective floor', verdict['reason'])
        unmeasured = self.summary()
        unmeasured['effective_floor'] = {**unmeasured['effective_floor'], 'latency_ns': None}
        self.assertIn('latency', self.classify(None, self.ok_client, unmeasured)['reason'])

    def test_clean_completed_run_passes_even_with_warning_pressure(self):
        self.assertEqual(self.classify(None, self.ok_client, self.summary())['verdict'], 'CalibrationPass')
        warned = self.summary(pressure_counts={'normal': 3, 'warning': 9})
        self.assertEqual(self.classify(None, self.ok_client, warned)['verdict'], 'CalibrationPass')

    def test_stop_wins_over_client_failure(self):
        stop = {'reason': 'free_cached_floor', 'detail': 'x'}
        verdict = self.classify(stop, {'exit_code': -15}, self.summary(), False)
        self.assertEqual((verdict['verdict'], verdict['reason']), ('Stopped', 'free_cached_floor'))

    def test_client_failures(self):
        for client in (None, {'exit_code': 1, 'failure': 'client exit 1'}, {'exit_code': 0, 'status_ok': False}):
            self.assertEqual(self.classify(None, client, self.summary())['verdict'], 'ClientFailed')

    def test_swap_growth_pass_boundary(self):
        under = self.summary(deltas={'swap_used_bytes': 512 * MIB - 1})
        self.assertEqual(self.classify(None, self.ok_client, under)['verdict'], 'CalibrationPass')
        verdict = self.classify(None, self.ok_client, self.summary(deltas={'swap_used_bytes': 512 * MIB}))
        self.assertEqual(verdict['verdict'], 'CalibrationFailed')
        self.assertIn('swap', verdict['reason'])

    def test_free_cached_pass_boundary(self):
        at = self.summary(min_free_plus_cached_bytes=2 * GIB)
        self.assertEqual(self.classify(None, self.ok_client, at)['verdict'], 'CalibrationPass')
        below = self.summary(min_free_plus_cached_bytes=2 * GIB - 1)
        self.assertIn('floor', self.classify(None, self.ok_client, below)['reason'])

    def test_unproven_cleanup_blind_or_critical_fail(self):
        cases = ((self.summary(), False, 'cleanup'), (self.summary(max_blind_run=1), True, 'blind'),
                 (self.summary(pressure_counts={'normal': 3, 'critical': 1}), True, 'critical'),
                 (self.summary(observed_samples=0), True, 'no observed'))
        for summary, proven, word in cases:
            verdict = self.classify(None, self.ok_client, summary, proven)
            self.assertEqual(verdict['verdict'], 'CalibrationFailed')
            self.assertIn(word, verdict['reason'])

    def test_negative_control(self):
        fired = {'verdict': 'Stopped', 'reason': 'footprint_cap'}
        summary = {'peak_footprint_bytes': 5 * GIB}
        self.assertEqual(p.classify_negative_control(fired, summary, 8 * GIB, True, -15)['verdict'], 'NegativeControlPass')
        failures = ((fired, summary, False, -15), (fired, summary, True, 4),
                    ({'verdict': 'Stopped', 'reason': 'free_cached_floor'}, summary, True, -15),
                    ({'verdict': 'ClientFailed'}, summary, True, 4), (fired, {'peak_footprint_bytes': 8 * GIB}, True, -15),
                    (fired, {'peak_footprint_bytes': None}, True, -15))
        for verdict, summ, proven, code in failures:
            self.assertEqual(p.classify_negative_control(verdict, summ, 8 * GIB, proven, code)['verdict'], 'NegativeControlFailed')

    def test_latency(self):
        stop = {'sample_start_ns': 100}
        self.assertEqual(p.latency(stop, 150, 400), {'sample_to_signal_ns': 50, 'signal_to_exit_ns': 250, 'sample_to_exit_ns': 300})
        self.assertIsNone(p.latency(None, 1, 2))
        self.assertIsNone(p.latency(stop, None, None)['sample_to_signal_ns'])


class CompetitorMatching(unittest.TestCase):
    def test_runtimes_are_found(self):
        cases = {
            ('/opt/hf2q/target/release/hf2q', '-v', 'serve', '--model', 'm.gguf'): 'runtime',
            ('/opt/hf2q/target/debug/deps/hf2q-0f3a9c1b2d4e5f60',): 'runtime',
            ('/usr/local/bin/llama-server', '-m', 'x.gguf'): 'runtime',
            ('/usr/bin/env', 'A=1', '-i', '/opt/llama/llama-server'): 'runtime',
            ('/opt/homebrew/bin/python3.12', '-m', 'mlx_lm.server', '--port', '8080'): 'python-hosted-runtime',
            ('/opt/homebrew/bin/python3', '-u', '-m', 'mlx_lm', 'server'): 'python-hosted-runtime',
            ('/v/bin/python3.12', '-X', 'utf8', '/v/bin/mlx_lm.server', '--model', 'q'): 'python-hosted-runtime',
            ('/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python',
             '-m', 'vllm.entrypoints.openai.api_server'): 'python-hosted-runtime',
            ('/bin/bash', '/opt/hf2q/scripts/serve_qwen38_opencode.sh'): 'launcher',
            ('/Users/u/.cargo/bin/cargo', 'test', '--release'): 'build',
        }
        for argv, category in cases.items():
            self.assertEqual(pre.classify_argv(list(argv)), category, argv)

    def test_mentions_are_not_runtimes(self):
        for argv in (['/Users/u/.local/bin/omp', '--append-system-prompt', 'hf2q', 'parallel', 'policy'],
                     ['/usr/bin/grep', 'hf2q'], ['/usr/bin/python3', '-c', 'import mlx_lm'],
                     ['/usr/bin/python3', 'tool.py', '--model', 'hf2q', 'mlx_lm.server'],
                     ['/bin/bash', '-c', 'hf2q serve'], ['/bin/zsh', '/tmp/notes_about_hf2q.sh'],
                     ['/usr/bin/vim', '/opt/hf2q/scripts/serve_qwen38_opencode.sh'], []):
            self.assertIsNone(pre.classify_argv(argv), argv)

    def test_own_ancestry_is_ignored_but_others_are_reported(self):
        rows = [dict(pid=1, ppid=0, pgid=1, rss_bytes=0, args='/sbin/launchd', argv=['/sbin/launchd']),
                dict(pid=10, ppid=1, pgid=10, rss_bytes=0, args='/x/hf2q-wrapper', argv=['/x/hf2q-wrapper']),
                dict(pid=11, ppid=10, pgid=10, rss_bytes=0, args='python3 q5_calibration.py', argv=['/usr/bin/python3', 'q5_calibration.py']),
                dict(pid=20, ppid=1, pgid=20, rss_bytes=GIB, args='/o/hf2q serve', argv=['/o/hf2q', 'serve'])]
        ignore = pre.ancestors(rows, 11)
        self.assertEqual(ignore, {1, 10, 11})
        self.assertEqual([c['pid'] for c in pre.competitors(rows, ignore)], [20])

    def test_port_state_needs_both_probes_clear(self):
        original = pre.port_connectable
        try:
            pre.port_connectable = lambda port: False
            self.assertTrue(pre.port_state(18000, [])['free'])
            self.assertFalse(pre.port_state(18000, [123])['free'])
            pre.port_connectable = lambda port: True
            self.assertFalse(pre.port_state(18000, [])['free'])
        finally:
            pre.port_connectable = original


class ArgvAndCases(unittest.TestCase):
    def test_argv_rules(self):
        self.assertEqual(q5_case.validate_argv(['/bin/bash', '/r/scripts/serve_x.sh'], 's'), ['/bin/bash', '/r/scripts/serve_x.sh'])
        for argv in (['hf2q', 'serve'], ['/bin/bash', '-c', 'hf2q serve'], ['/bin/zsh', '-lc', 'x'],
                     ['/bin/bash', 'relative.sh'], ['/usr/bin/env', 'A=1', '/x/hf2q'], ['/x/hf2q', '--port', '{PORT}'],
                     ['/x/hf2q', ''], ['/x/hf2q', 'a\0b'], [], 'hf2q serve'):
            with self.assertRaises(q5_case.CaseError, msg=repr(argv)):
                q5_case.validate_argv(argv, 's')

    def case(self, **changes):
        case = q5_case.loads((HERE / 'q5_cases' / 'qwen38.json').read_bytes())
        case.update(changes)
        return case

    def builtins(self):
        return {'REPO': '/repo', 'OUT': '/out/run', 'PYTHON': '/usr/bin/python3', 'PYTHON_SHA256': 'a' * 64,
                'RUN_ID': 'q5-run-0123456789', 'NONCE32': 'b' * 32}

    def required(self):
        return {'HF2Q_Q5_CARGO': '/c/cargo', 'HF2Q_Q5_TOOLCHAIN': '/c/tc', 'HF2Q_Q5_SANDBOX_PROOF': '/c/proof.json',
                'HF2Q_Q5_SANDBOX_PROOF_SHA256': 'c' * 64, 'HF2Q_Q5_CLIENT_APPROVAL_TEMPLATE': '/c/approval.json'}

    def test_missing_required_variables_leave_plan_incomplete(self):
        plan = q5_case.build_plan(self.case(), self.builtins(), {})
        self.assertFalse(plan['complete'])
        self.assertIn('CARGO', plan['missing_variables'])
        self.assertIn('{CARGO}', plan['client']['argv'])

    def test_overrides_apply_and_unknown_overrides_refuse(self):
        env = {**self.required(), 'HF2Q_Q5_PORT': '19000', 'HF2Q_Q5_MODEL': '/m/q.gguf'}
        plan = q5_case.build_plan(self.case(), self.builtins(), env)
        self.assertTrue(plan['complete'])
        self.assertEqual((plan['port'], plan['serve']['env']['PORT'], plan['artifact']['path']), (19000, '19000', '/m/q.gguf'))
        self.assertIn('http://127.0.0.1:19000/v1', plan['client']['argv'])
        with self.assertRaises(q5_case.CaseError):
            q5_case.build_plan(self.case(), self.builtins(), {**env, 'HF2Q_Q5_MODLE': '/typo'})

    def test_launch_time_values_only_in_client(self):
        case = self.case()
        case['serve']['argv'].append('{SERVER_PID}')
        with self.assertRaises(q5_case.CaseError):
            q5_case.validate(case)
        case = self.case()
        case['serve']['env']['X'] = '{UNDECLARED}'
        with self.assertRaises(q5_case.CaseError):
            q5_case.validate(case)

    def test_bad_values_refuse(self):
        for env in ({'HF2Q_Q5_PORT': '80'}, {'HF2Q_Q5_MODEL_SHA256': 'ABC'}, {'HF2Q_Q5_HF2Q_BIN': 'target/hf2q'}):
            with self.assertRaises(q5_case.CaseError, msg=env):
                q5_case.build_plan(self.case(), self.builtins(), {**self.required(), **env})
        for change in ({'workload': {'id': 'x', 'slots': 2, 'context_per_slot': 1}}, {'family': 'llama'}, {'schema': 'v0'},
                       {'extra': 1}, {'policy': {'footprint_cap_bytes': GIB}},
                       {'artifact': {'path': '{MODEL}', 'text_only_locator': 'yes'}}):
            with self.assertRaises(q5_case.CaseError, msg=change):
                q5_case.validate(self.case(**change))

    def test_duplicate_keys_and_cycles_refuse(self):
        with self.assertRaises(q5_case.CaseError):
            q5_case.loads(b'{"a": 1, "a": 2}')
        case = self.case(variables={'A': '{B}', 'B': '{A}'})
        with self.assertRaises(q5_case.CaseError):
            q5_case.resolve_variables(case, self.builtins(), {})

    def test_runtime_values_keep_types(self):
        values = {'SERVER_PID': 4242, 'PORT': '18238'}
        self.assertEqual(q5_case.runtime_value('{SERVER_PID}', values), 4242)
        self.assertEqual(q5_case.runtime_value('http://127.0.0.1:{PORT}/v1', values), 'http://127.0.0.1:18238/v1')

    def test_checked_in_family_cases(self):
        ports, families = set(), {'qwen38': 'qwen', 'qwen36': 'qwen', 'gemma4': 'gemma4', 'deepseek4': 'deepseek4'}
        repo = HERE.parent.parent
        for family, client_family in families.items():
            case = q5_case.loads((HERE / 'q5_cases' / f'{family}.json').read_bytes())
            self.assertEqual(case['family'], family)
            builtins = {**self.builtins(), 'REPO': str(repo)}
            plan = q5_case.build_plan(case, builtins, self.required())
            self.assertTrue(plan['complete'], family)
            ports.add(plan['port'])
            launcher = Path(plan['serve']['argv'][1])
            self.assertTrue(launcher.is_file() and launcher.name == f'serve_{family}_opencode.sh', family)
            argv = plan['client']['argv']
            self.assertEqual(argv[argv.index('--family') + 1], client_family)
            self.assertTrue(Path(argv[4]).is_file())
            self.assertEqual(plan['serve']['env']['MAX_SLOTS'], '4')
            self.assertEqual(plan['serve']['env']['PORT'], str(plan['port']))
            sources = plan['client']['approval']['set']['sources']['sha256_of']
            self.assertTrue(all(Path(s).is_file() for s in sources), family)
            approval = plan['client']['approval']['set']
            self.assertEqual(approval['server_pid'], '{SERVER_PID}')
            self.assertEqual(plan['policy']['footprint_cap_bytes'], None)
            self.assertEqual(plan['serve']['env']['MODEL'], plan['served_model'])
            checks = {(c['field'], json.dumps(c['equals'])) for c in plan['readiness']['checks']}
            if family in ('qwen38', 'gemma4'):
                # The real sibling projector auto-loads unless the model is served from a private directory.
                self.assertTrue(plan['artifact']['text_only_locator'])
                self.assertEqual(plan['served_model'], f"/out/run/text-model/{Path(plan['artifact']['path']).name}")
            else:
                self.assertEqual(plan['served_model'], plan['artifact']['path'])
            if family != 'deepseek4':
                self.assertIn(('measurement_snapshot.active_controls.vision_projector', 'false'), checks)
        self.assertEqual(len(ports), 4)

    def test_checked_in_peer_cases_launch_the_matched_peer_arm(self):
        import family_peer
        from qwen_n4_live_smoke import MODULES
        repo = HERE.parent.parent
        hf_ports = {q5_case.build_plan(q5_case.loads((HERE / 'q5_cases' / f'{f}.json').read_bytes()),
                                       {**self.builtins(), 'REPO': str(repo)}, self.required())['port']
                    for f in ('qwen38', 'qwen36', 'gemma4', 'deepseek4')}
        ports = set()
        for family, client_family in (('deepseek4', 'deepseek4'), ('gemma4', 'gemma4'), ('qwen38', 'qwen'),
                                      ('qwen36', 'qwen')):
            hf_case = q5_case.loads((HERE / 'q5_cases' / f'{family}.json').read_bytes())
            case = q5_case.loads((HERE / 'q5_cases' / f'peer-{family}.json').read_bytes())
            self.assertEqual(case['family'], family)
            self.assertEqual(case['variables']['MODEL'], hf_case['variables']['MODEL'], 'same GGUF as hf2q')
            self.assertEqual(case['variables']['MODEL_ID'], hf_case['variables']['MODEL_ID'])
            plan = q5_case.build_plan(case, {**self.builtins(), 'REPO': str(repo)}, self.required())
            self.assertTrue(plan['complete'], family)
            ports.add(plan['port'])
            argv = plan['serve']['argv']
            self.assertEqual(argv[0], plan['binary'])
            self.assertEqual(Path(argv[0]).name, 'llama-server')

            def flag(name):
                return argv[argv.index(name) + 1]
            self.assertEqual((flag('--parallel'), flag('--kv-unified-per-slot'), flag('--ctx-size')),
                             ('4', '262144', '1048576'))
            self.assertEqual((flag('--flash-attn'), flag('--cache-type-k'), flag('--cache-type-v')), ('on', 'f16', 'f16'))
            self.assertEqual((flag('--model'), flag('--alias'), flag('--port')),
                             (plan['artifact']['path'], plan['variables']['MODEL_ID'], str(plan['port'])))
            self.assertIn('--no-context-shift', argv)
            self.assertEqual(flag('--spec-type'), 'none')
            self.assertEqual(Path(flag('--chat-template-file')).name, f'{family}.template.jinja')
            if family in family_peer.CONTROLS:
                controls = family_peer.CONTROLS[family]
                self.assertEqual('--kv-unified' in argv, controls['kv_unified'], family)
                self.assertEqual(float(flag('--repeat-penalty')), controls['repetition_penalty'])
                self.assertEqual(flag('--reasoning'), controls['reasoning'])
                self.assertEqual('--reasoning-preserve' in argv, controls['reasoning_preserve'])
                self.assertEqual(argv.count('--reasoning-budget') and flag('--reasoning-budget'),
                                 str(controls['reasoning_budget']) if 'reasoning_budget' in controls else 0)
            else:
                self.assertIn('--kv-unified', argv)
                self.assertEqual((flag('--reasoning'), flag('--reasoning-budget'), flag('--repeat-penalty')),
                                 ('on', '2048', '1.05'))
            client = plan['client']['argv']
            self.assertEqual((client[client.index('--family') + 1], client[client.index('--engine-kind') + 1]),
                             (client_family, 'peer'))
            sources = plan['client']['approval']['set']['sources']['sha256_of']
            self.assertEqual({Path(s).name for s in sources}, set(MODULES))
            checks = {(c['field'], json.dumps(c['equals'])) for c in plan['readiness']['checks']}
            self.assertTrue({('total_slots', '4'), ('default_generation_settings.n_ctx', '262144'),
                             ('modalities.vision', 'false'), ('build_info', '"b128-e86f5fd"')} <= checks)
        self.assertEqual(len(ports), 4)
        self.assertFalse(ports & hf_ports)

    def test_hf2q_case_sources_match_the_client_source_freeze(self):
        from qwen_n4_live_smoke import MODULES
        for family in ('qwen38', 'qwen36', 'gemma4', 'deepseek4'):
            case = q5_case.loads((HERE / 'q5_cases' / f'{family}.json').read_bytes())
            sources = case['client']['approval']['set']['sources']['sha256_of']
            self.assertEqual({Path(s).name for s in sources}, set(MODULES), family)

    def test_deepseek_case_uses_scoped_compressor_override(self):
        case = q5_case.loads((HERE / 'q5_cases' / 'deepseek4.json').read_bytes())
        env = case['serve']['env']
        self.assertNotIn('HF2Q_DEEPSEEK_UNSAFE_MEMORY_OVERRIDE', env)
        self.assertEqual(env['MAX_COMPRESSOR_USED_GIB'], '16')
        launcher = (HERE.parent / 'serve_deepseek4_opencode.sh').read_text()
        self.assertIn('MAX_COMPRESSOR_USED_GIB="${MAX_COMPRESSOR_USED_GIB:-8}"', launcher)
        self.assertEqual(case['workload']['id'], 'adr063-q5-deepseek4-n4-coding-ctx262144-final-default')

    def test_gemma_case_forwards_only_an_explicit_full_input_checkpoint_arm(self):
        case = q5_case.loads((HERE / 'q5_cases' / 'gemma4.json').read_bytes())
        builtins = {**self.builtins(), 'REPO': str(HERE.parent.parent)}
        stock = q5_case.build_plan(case, builtins, self.required())
        self.assertEqual(stock['serve']['env']['FULL_INPUT_CHECKPOINT'], '')
        self.assertNotIn('HF2Q_GEMMA_FULL_INPUT_CHECKPOINT', stock['serve']['env'])
        off = q5_case.build_plan(case, builtins, {**self.required(), 'HF2Q_Q5_GEMMA_FULL_INPUT_CHECKPOINT': '0'})
        self.assertEqual(off['serve']['env']['FULL_INPUT_CHECKPOINT'], '0')
        launcher = (HERE.parent / 'serve_gemma4_opencode.sh').read_text()
        self.assertEqual(launcher.count('${FULL_INPUT_ENV:+HF2Q_GEMMA_FULL_INPUT_CHECKPOINT="$FULL_INPUT_ENV"}'), 2)
        self.assertIn('0|1) FULL_INPUT_ENV="$FULL_INPUT_CHECKPOINT" ;;', launcher)

    def test_gemma_case_forwards_only_an_explicit_joint_warm_prefill_arm(self):
        case = q5_case.loads((HERE / 'q5_cases' / 'gemma4.json').read_bytes())
        builtins = {**self.builtins(), 'REPO': str(HERE.parent.parent)}
        stock = q5_case.build_plan(case, builtins, self.required())
        self.assertEqual(stock['serve']['env']['JOINT_WARM_PREFILL'], '')
        self.assertNotIn('HF2Q_GEMMA_JOINT_WARM_PREFILL', stock['serve']['env'])
        off = q5_case.build_plan(case, builtins, {**self.required(), 'HF2Q_Q5_GEMMA_JOINT_WARM_PREFILL': '0'})
        self.assertEqual(off['serve']['env']['JOINT_WARM_PREFILL'], '0')
        self.assertEqual(off['serve']['env']['FULL_INPUT_CHECKPOINT'], '', 'arms are independent')
        launcher = (HERE.parent / 'serve_gemma4_opencode.sh').read_text()
        self.assertEqual(launcher.count('${JOINT_WARM_ENV:+HF2Q_GEMMA_JOINT_WARM_PREFILL="$JOINT_WARM_ENV"}'), 2)
        self.assertIn('0|1) JOINT_WARM_ENV="$JOINT_WARM_PREFILL" ;;', launcher)

    def test_gemma_case_forwards_only_an_explicit_warm_tiled_attention_arm(self):
        case = q5_case.loads((HERE / 'q5_cases' / 'gemma4.json').read_bytes())
        builtins = {**self.builtins(), 'REPO': str(HERE.parent.parent)}
        stock = q5_case.build_plan(case, builtins, self.required())
        self.assertEqual(stock['serve']['env']['WARM_TILED_ATTENTION'], '')
        self.assertNotIn('HF2Q_GEMMA_WARM_TILED_ATTENTION', stock['serve']['env'])
        off = q5_case.build_plan(case, builtins, {**self.required(), 'HF2Q_Q5_GEMMA_WARM_TILED_ATTENTION': '0'})
        self.assertEqual(off['serve']['env']['WARM_TILED_ATTENTION'], '0')
        self.assertEqual(off['serve']['env']['JOINT_WARM_PREFILL'], '', 'arms are independent')
        self.assertEqual(off['serve']['env']['FULL_INPUT_CHECKPOINT'], '', 'arms are independent')
        launcher = (HERE.parent / 'serve_gemma4_opencode.sh').read_text()
        self.assertEqual(launcher.count('${WARM_TILED_ENV:+HF2Q_GEMMA_WARM_TILED_ATTENTION="$WARM_TILED_ENV"}'), 2)
        self.assertIn('0|1) WARM_TILED_ENV="$WARM_TILED_ATTENTION" ;;', launcher)

    def test_run_id_matches_client_identity_rule(self):
        for name in ('x', 'q5-qwen38-20261030T120000Z', 'a.b c' * 30):
            self.assertRegex(q5_calibration.run_id(name, 'f' * 32), r'^[A-Za-z0-9_-]{16,64}$')


class Parsers(unittest.TestCase):
    def test_vm_stat(self):
        vm = q5_host.parse_vm_stat(VM_STAT)
        self.assertEqual(vm['free_plus_cached_bytes'], (5264938 + 45030 + 889466) * 16384)
        self.assertEqual(vm['compressor_bytes'], 392967 * 16384)
        with self.assertRaises(q5_host.ObservationError):
            q5_host.parse_vm_stat(VM_STAT.replace('File-backed pages', 'File pages'))
        with self.assertRaises(q5_host.ObservationError):
            q5_host.parse_vm_stat(VM_STAT + 'Pages free:  1.\n')

    def test_swap_sysctl_footprint_ps(self):
        swap = q5_host.parse_swapusage('total = 1024.00M  used = 22.25M  free = 1001.75M  (encrypted)')
        self.assertEqual((swap['total_bytes'], swap['used_bytes']), (GIB, round(22.25 * MIB)))
        with self.assertRaises(q5_host.ObservationError):
            q5_host.parse_swapusage('total = 1024.00M  free = 1001.75M')
        text = 'hf2q [4242]: 64-bit    Footprint: 9 B\nAuxiliary data:\n    phys_footprint: 26016 B\n    phys_footprint_peak: 57149 B\n'
        self.assertEqual(q5_host.parse_footprint(text, 4242), 26016)
        with self.assertRaises(q5_host.ObservationError):
            q5_host.parse_footprint(text, 4243)
        rows = q5_host.parse_ps('  42  1  42  1024 Thu Oct 29 17:30:35 2026 /o/hf2q -v serve --model a b\n')
        self.assertEqual((rows[0]['pid'], rows[0]['rss_bytes'], rows[0]['start'], rows[0]['argv'][:2]),
                         (42, 1024 * 1024, 'Thu Oct 29 17:30:35 2026', ['/o/hf2q', '-v']))
        self.assertEqual(q5_host.parse_sysctl('a: 1\nb: x y\n', ('a', 'b')), {'a': '1', 'b': 'x y'})


class OwnershipAndControl(unittest.TestCase):
    def test_descendants_follow_ancestry_and_drop_reused_pids(self):
        rows = [dict(pid=10, ppid=1, start='s10'), dict(pid=11, ppid=10, start='s11'),
                dict(pid=12, ppid=11, start='s12'), dict(pid=30, ppid=1, start='s30')]
        known = {}
        self.assertEqual(q5_runtime.descendants(rows, (10,), known), {10, 11, 12})
        reused = [dict(pid=10, ppid=1, start='s10'), dict(pid=12, ppid=1, start='other')]
        self.assertEqual(q5_runtime.descendants(reused, (10,), known), {10})
        self.assertNotIn(12, known)

    def test_group_stats_footprint_and_fallback(self):
        rows = [dict(pid=5, pgid=5, rss_bytes=100), dict(pid=6, pgid=5, rss_bytes=50), dict(pid=7, pgid=7, rss_bytes=1)]

        def footprint(pid):
            if pid == 6:
                raise q5_host.ObservationError('gone')
            return 1000
        stats = q5_runtime.group_stats(rows, 5, footprint)
        self.assertEqual((stats['footprint_bytes'], stats['footprint_source'], stats['rss_bytes']), (1050, 'ps-rss-fallback', 150))
        self.assertIsNone(q5_runtime.group_stats(rows, 9, footprint)['footprint_bytes'])

    def test_ceiling_respects_headroom_cap_and_hard_max(self):
        info = nc.plan_ceiling(60 * GIB, 2 * GIB, 8 * GIB, 4 * GIB)
        self.assertEqual((info['ceiling_bytes'], info['cap_bytes']), (8 * GIB, 4 * GIB))
        for args in ((13 * GIB, 2 * GIB, 8 * GIB, 4 * GIB), (60 * GIB, 2 * GIB, 8 * GIB, 8 * GIB),
                     (200 * GIB, 2 * GIB, 40 * GIB, 4 * GIB), (60 * GIB, 2 * GIB, 8 * GIB, 0)):
            with self.assertRaises(nc.ControlError, msg=args):
                nc.plan_ceiling(*args)

    def test_steps_never_exceed_ceiling(self):
        for ceiling, step in ((8 * GIB, 256 * MIB), (GIB + 5, 256 * MIB), (100, 1000)):
            sizes = nc.steps(ceiling, step)
            self.assertEqual(sum(sizes), ceiling)
            self.assertTrue(all(0 < s <= step for s in sizes))

    def test_child_refuses_out_of_bounds_ceiling(self):
        self.assertEqual(nc.main(['--ceiling-bytes', str(nc.HARD_MAX_BYTES + 1), '--parent-pid', str(os.getppid())]), 2)


PHYSICAL = 128 * GIB


def calibration_receipt(**changes):
    receipt = {'schema': q5_admission.RECEIPT_SCHEMA, 'mode': 'calibration', 'family': 'deepseek4',
               'verdict': {'verdict': 'CalibrationPass', 'reason': None},
               'workload': {'id': 'w', 'slots': 4, 'context_per_slot': 262144},
               'binary': {'sha256': 'b' * 64}, 'host': {'physical_bytes': PHYSICAL},
               'artifact': {'path': '/m.gguf', 'bytes': 100, 'mtime_ns': 7, 'sha256': None},
               'baseline': {'free_plus_cached_bytes': 110 * GIB},
               'samples': {'min_free_plus_cached_bytes': 4 * GIB, 'peak_footprint_bytes': 4 * GIB,
                           'deltas': {'swap_used_bytes': 0, 'compressor_bytes': 0}}}
    receipt.update(changes)
    return receipt


def current(**changes):
    value = {'family': 'deepseek4', 'workload': {'id': 'w', 'slots': 4, 'context_per_slot': 262144, 'client': 'x'},
             'binary_sha256': 'b' * 64, 'physical_bytes': PHYSICAL,
             'artifact': {'path': '/m.gguf', 'bytes': 100, 'mtime_ns': 7, 'sha256': None}}
    value.update(changes)
    return value


class Admission(unittest.TestCase):
    def test_charge_is_host_level_drop_not_process_footprint(self):
        admission = q5_admission.admission_from(calibration_receipt(), current())
        self.assertEqual(admission['peak_host_used_bytes'], PHYSICAL - 4 * GIB)
        self.assertEqual(admission['baseline_host_used_bytes'], PHYSICAL - 110 * GIB)
        self.assertEqual(admission['charged_bytes'], 106 * GIB)  # not the 4 GiB phys_footprint
        self.assertEqual(admission['footprint_cap_bytes'], 5 * GIB)  # 4 GiB x 1.25

    def test_admission_boundary_uses_charge_plus_initial_effective_floor(self):
        admission = q5_admission.admission_from(calibration_receipt(), current())
        self.assertEqual(admission['seed_rate_bytes_per_s'], 0)  # pre-record receipts seed nothing
        need = 106 * GIB + 2 * GIB
        self.assertTrue(q5_admission.admit(admission, need, 2 * GIB)['admitted'])
        decision = q5_admission.admit(admission, need - 1, 2 * GIB)
        self.assertFalse(decision['admitted'])
        self.assertEqual(decision['required_free_plus_cached_bytes'], need)

    def test_measured_rate_seeds_the_admitted_floor(self):
        samples = {**calibration_receipt()['samples'],
                   'effective_floor': {'max_post_readiness_drop_rate_bytes_per_s': 20 * GIB}}
        admission = q5_admission.admission_from(calibration_receipt(samples=samples), current())
        self.assertEqual(admission['seed_rate_bytes_per_s'], 20 * GIB)
        initial = p.effective_floor(2 * GIB, admission['seed_rate_bytes_per_s'], 0.5, NC_LATENCY)
        self.assertEqual(initial, math.ceil(20 * GIB * 0.646))
        self.assertFalse(q5_admission.admit(admission, 106 * GIB + 2 * GIB, initial)['admitted'])

    def test_any_identity_mismatch_requires_recalibration(self):
        bad = [(calibration_receipt(family='qwen38'), current()),
               (calibration_receipt(verdict={'verdict': 'Stopped'}), current()),
               (calibration_receipt(run_kind='admitted'), current()),
               (calibration_receipt(mode='negative-control'), current()),
               (calibration_receipt(schema='other'), current()),
               (calibration_receipt(), current(binary_sha256='c' * 64)),
               (calibration_receipt(), current(workload={'id': 'w2', 'slots': 4, 'context_per_slot': 262144})),
               (calibration_receipt(), current(workload={'id': 'w', 'slots': 4, 'context_per_slot': 131072})),
               (calibration_receipt(), current(physical_bytes=64 * GIB)),
               (calibration_receipt(), current(artifact={'bytes': 100, 'mtime_ns': 8, 'sha256': None})),
               (calibration_receipt(artifact={'bytes': 100, 'mtime_ns': 7, 'sha256': 'a' * 64}),
                current(artifact={'bytes': 100, 'mtime_ns': 7, 'sha256': 'e' * 64})),
               (calibration_receipt(samples={'min_free_plus_cached_bytes': GIB, 'peak_footprint_bytes': None}), current())]
        for receipt, now in bad:
            with self.assertRaises(q5_admission.AdmissionError, msg=(receipt, now)):
                q5_admission.admission_from(receipt, now)

    def test_sha_match_wins_over_touched_mtime(self):
        receipt = calibration_receipt(artifact={'bytes': 100, 'mtime_ns': 7, 'sha256': 'a' * 64})
        now = current(artifact={'bytes': 100, 'mtime_ns': 9, 'sha256': 'a' * 64})
        self.assertEqual(q5_admission.admission_from(receipt, now)['charged_bytes'], 106 * GIB)

    def test_growth_cap_is_never_absent(self):
        admission = q5_admission.admission_from(calibration_receipt(), current())
        self.assertEqual(q5_admission.growth_cap(admission, PHYSICAL, 2 * GIB)['kind'], 'footprint')
        floor_cap = q5_admission.growth_cap(None, PHYSICAL, 2 * GIB)
        self.assertEqual((floor_cap['kind'], floor_cap['bytes']), ('host-floor', PHYSICAL - 2 * GIB))

    def test_negative_control_latency(self):
        good = {'schema': q5_admission.RECEIPT_SCHEMA, 'mode': 'negative-control',
                'verdict': {'verdict': 'NegativeControlPass'}, 'stop_latency': {'sample_to_exit_ns': 145871166}}
        self.assertEqual(q5_admission.negative_control_latency(good), 145871166)
        for bad in ({**good, 'verdict': {'verdict': 'NegativeControlFailed'}}, {**good, 'stop_latency': None},
                    {**good, 'mode': 'calibration'}):
            with self.assertRaises(q5_admission.AdmissionError):
                q5_admission.negative_control_latency(bad)


class DynamicFloor(unittest.TestCase):
    def feed(self, monitor, values, ready_index):
        for index, free in enumerate(values):
            if index == ready_index:
                monitor.ready_ns = index * 500_000_000
            decision = monitor.feed(sample(index, host(free_cached=free)))
            if decision is not None:
                return decision
        return None

    def test_load_phase_drop_does_not_raise_the_floor(self):
        # 80 GiB drop while weights become wired, then 1 GiB per 0.5 s after readiness (DeepSeek-like).
        monitor = p.Monitor(BASELINE, policy(), NC_LATENCY, 'nc')
        self.assertIsNone(self.feed(monitor, [110 * GIB, 30 * GIB, 29 * GIB, 28 * GIB], 2))
        record = monitor.summary()['effective_floor']
        self.assertEqual(record['max_load_phase_drop_bytes_per_sample'], 80 * GIB)
        self.assertEqual(record['max_post_readiness_drop_rate_bytes_per_s'], 2 * GIB)
        self.assertEqual(record['effective_floor_bytes'], 2 * GIB)  # 2 GiB/s x 0.646 s < 2 GiB minimum
        self.assertEqual((record['latency_ns'], record['min_free_plus_cached_bytes']), (NC_LATENCY, 28 * GIB))

    def test_fast_allocation_raises_the_live_floor(self):
        # Qwen3.8-like: 10 GiB in one post-readiness sample -> floor = 20 GiB/s x 0.646 s ~ 12.9 GiB.
        monitor = p.Monitor(BASELINE, policy(), NC_LATENCY, 'nc')
        descent = [100 * GIB, 90 * GIB, 80 * GIB] + [(75 - i) * GIB for i in range(64)]
        stop = self.feed(monitor, descent, 1)
        self.assertEqual(monitor.effective_floor(), math.ceil(20 * GIB * 0.646))
        self.assertEqual(stop['reason'], 'free_cached_floor')
        # The first sample below the raised floor stops, far above the 2 GiB minimum.
        self.assertEqual(monitor.summary()['effective_floor']['min_free_plus_cached_bytes'], 12 * GIB)

    def test_seeded_rate_applies_before_any_drop_and_missing_latency_uses_interval(self):
        seeded = p.Monitor(BASELINE, policy(), NC_LATENCY, 'nc', seed_rate=20 * GIB)
        self.assertEqual(seeded.feed(sample(0, host(free_cached=12 * GIB)))['reason'], 'free_cached_floor')
        unmeasured = p.Monitor(BASELINE, policy(), None, 'unavailable', seed_rate=20 * GIB)
        self.assertEqual(unmeasured.effective_floor(), 10 * GIB)


class ArtifactIdentity(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name
        self.model = os.path.join(self.root, 'm.gguf')
        with open(self.model, 'wb') as handle:
            handle.write(b'x' * 100)

    def tearDown(self):
        self.dir.cleanup()

    def receipt(self, **output):
        with open(self.model + '.receipt.json', 'w') as handle:
            json.dump({'schema_version': 3, 'output': {'path': self.model, 'size': 100, 'sha256': 'a' * 64, **output}}, handle)

    def test_sha_comes_from_receipt_output_sha256(self):
        self.receipt()
        identity = q5_artifacts.artifact_identity(self.model)
        self.assertEqual(identity['sha256'], 'a' * 64)
        self.assertEqual(identity['sha256_source']['field'], 'output.sha256')
        self.assertFalse(identity['sha256_hashed_by_supervisor'])

    def test_size_mismatch_or_missing_receipt_falls_back_to_size_mtime(self):
        self.assertIsNone(q5_artifacts.artifact_identity(self.model)['sha256'])
        self.receipt(size=99)
        identity = q5_artifacts.artifact_identity(self.model)
        self.assertIsNone(identity['sha256'])
        self.assertIn('size', identity['receipt_rejected'])
        self.assertEqual(identity['bytes'], 100)

    def test_provided_digest_must_agree(self):
        self.receipt()
        with self.assertRaises(q5_artifacts.ArtifactError):
            q5_artifacts.artifact_identity(self.model, 'f' * 64)
        os.unlink(self.model + '.receipt.json')
        identity = q5_artifacts.artifact_identity(self.model, 'f' * 64)
        self.assertEqual((identity['sha256'], identity['sha256_source']), ('f' * 64, 'operator-provided, not verified by the supervisor'))

    def test_projector_candidates_and_private_locator(self):
        for name in ('m-mmproj.gguf', 'mmproj-x.gguf', 'other.gguf', 'm-mmproj.gguf.receipt.json'):
            with open(os.path.join(self.root, name), 'wb') as handle:
                handle.write(b'p')
        names = [os.path.basename(p['path']) for p in q5_artifacts.projector_candidates(self.model)]
        self.assertEqual(names, ['m-mmproj.gguf', 'mmproj-x.gguf'])
        out = os.path.join(self.root, 'out')
        os.mkdir(out)
        locator = q5_artifacts.create_locator(self.model, out)
        self.assertEqual(os.stat(locator['served_path']).st_ino, os.stat(self.model).st_ino)
        self.assertEqual(q5_artifacts.projector_candidates(locator['served_path']), [])
        open(locator['served_path'] + '.pair.lock', 'w').close()
        removal = q5_artifacts.remove_locator(locator)
        self.assertTrue(removal['directory_removed'])
        self.assertTrue(os.path.isfile(self.model))


if __name__ == '__main__':
    unittest.main()
