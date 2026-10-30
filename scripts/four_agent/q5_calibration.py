#!/usr/bin/env python3
"""ADR-063 Q5 per-family memory calibration supervisor.

Runs ONE supervised four-agent workload from a JSON case file (or, with
--negative-control, a bounded synthetic memory-growth child) on a host where it
must be the only runtime. It records a receipt; it never retries, never selects
a favorable sample and never claims speed. See README "Q5 per-family calibration".
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

import q5_case
import q5_host
import q5_negative_control as nc
import q5_policy
import q5_preflight
import q5_runtime

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RECEIPT_SCHEMA = 'hf2q.q5.calibration-receipt.v1'
SOURCES = ('q5_calibration.py', 'q5_case.py', 'q5_host.py', 'q5_policy.py', 'q5_preflight.py',
           'q5_runtime.py', 'q5_negative_control.py')
INPUT_FILE_LIMIT = 64 << 20
INTERRUPTED = []


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inside_git(path):
    return any((parent / '.git').exists() for parent in [path, *path.parents])


def run_id(name, nonce):
    """[A-Za-z0-9_-]{16,64} derived from the output directory name (clients reject other run ids)."""
    value = 'q5-' + re.sub(r'[^A-Za-z0-9_-]', '-', name)
    if len(value) < 16:
        value += '-' + nonce
    return value[:64]


def builtins(out):
    python = Path(sys.executable).resolve()
    nonce = secrets.token_hex(16)
    return {'REPO': str(REPO), 'OUT': str(out), 'PYTHON': sys.executable, 'PYTHON_SHA256': sha256(python),
            'RUN_ID': run_id(out.name, nonce), 'NONCE32': nonce}


def negative_plan(args, out):
    ceiling, cap = args.nc_ceiling_bytes, args.nc_cap_bytes or args.nc_ceiling_bytes // 2
    argv = [sys.executable, '-I', '-B', str(HERE / 'q5_negative_control.py'), '--ceiling-bytes', str(ceiling),
            '--step-bytes', str(args.nc_step_bytes), '--step-seconds', str(args.nc_step_seconds),
            '--hold-seconds', '10', '--max-seconds', '180', '--parent-pid', str(os.getpid())]
    return {'family': 'negative-control', 'complete': True, 'missing_variables': [],
            'workload': {'id': 'q5-negative-control-synthetic-growth', 'ceiling_bytes': ceiling, 'cap_bytes': cap,
                         'step_bytes': args.nc_step_bytes, 'step_seconds': args.nc_step_seconds},
            'binary': str(Path(sys.executable).resolve()), 'artifact': None, 'port': None,
            'serve': {'argv': argv, 'env': {}}, 'readiness': None, 'client': None,
            'policy': q5_policy.effective_policy({'footprint_cap_bytes': cap})}


def file_checks(plan):
    paths = {'binary': plan['binary'], 'serve.argv[0]': plan['serve']['argv'][0]}
    if plan.get('artifact'):
        paths['artifact'] = plan['artifact']['path']
    if plan.get('client'):
        paths['client.argv[0]'] = plan['client']['argv'][0]
        if plan['client'].get('approval'):
            paths['client.approval.template'] = plan['client']['approval']['template']
    return {name: os.path.isfile(path) for name, path in paths.items()}


def input_hashes(plan):
    argv = plan['serve']['argv'] + (plan['client']['argv'] if plan.get('client') else [])
    if plan.get('client') and plan['client'].get('approval'):
        argv = argv + [plan['client']['approval']['template']]
    found = {}
    for item in argv:
        if os.path.isabs(item) and os.path.isfile(item) and os.path.getsize(item) <= INPUT_FILE_LIMIT:
            found[item] = sha256(item)
    return found


LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route loopback through a proxy


def get_json(url):
    with LOOPBACK.open(url, timeout=2) as response:
        return json.loads(response.read())


def field(doc, dotted):
    for part in dotted.split('.'):
        if not isinstance(doc, dict) or part not in doc:
            return None
        doc = doc[part]
    return doc


def same(actual, expected):
    """JSON equality without Python's True == 1 coercion."""
    return type(actual) is type(expected) and actual == expected


def resolve_set(spec, values):
    if isinstance(spec, dict) and set(spec) == {'sha256_of'}:
        paths = q5_case.expand(spec['sha256_of'], values)
        return sha256(paths) if isinstance(paths, str) else {os.path.basename(p): sha256(p) for p in paths}
    return q5_case.runtime_value(spec, values)


def render_approval(approval, values, out):
    packet = q5_case.loads(Path(approval['template']).read_bytes())
    for key, spec in approval['set'].items():
        packet[key] = resolve_set(spec, values)
    path = out / 'client-approval.json'
    save(path, packet)
    return {'CLIENT_APPROVAL': str(path), 'CLIENT_APPROVAL_SHA256': sha256(path)}


def preflight(plan, policy, receipt):
    rows = q5_host.process_table()
    ignore = q5_preflight.ancestors(rows, os.getpid())
    found = q5_preflight.competitors(rows, ignore)
    port = None if plan['port'] is None else q5_preflight.port_state(plan['port'], q5_host.listeners(plan['port']))
    samples = []
    for index in range(policy['baseline_samples']):
        samples.append(q5_runtime.take_sample(index))
        time.sleep(policy['interval_seconds'])
    baseline, why = q5_policy.baseline_from(samples, policy)
    receipt['preflight'] = {'competitors': found, 'ignored_own_ancestry_pids': sorted(ignore), 'port': port,
                            'thermal_state': q5_host.thermal_state(),
                            'baseline_samples': [dict(index=s['index'], start_ns=s['start_ns'], errors=s['errors'],
                                                      host=q5_runtime.compact(s['host'])) for s in samples]}
    receipt['baseline'] = baseline
    if found:
        return 'competing runtime(s): ' + ', '.join(f"{c['pid']}:{c['category']}" for c in found)
    if port is not None and not port['free']:
        return f'port {plan["port"]} is not free'
    return why


def wait_ready(plan, server, latch, receipt):
    spec, start = plan['readiness'], time.monotonic()
    while True:
        if latch.event.is_set():
            return False
        if INTERRUPTED:
            return 'interrupted'
        if server.poll() is not None:
            return 'server_exited'
        if time.monotonic() - start > spec['timeout_seconds']:
            return 'readiness_timeout'
        try:
            if same(field(get_json(spec['url']), spec['field']), spec['equals']):
                break
        except (OSError, ValueError, urllib.error.URLError):
            pass
        latch.event.wait(0.5)
    receipt['readiness'] = {'seconds': time.monotonic() - start, 'documents': {}, 'checks': []}
    for index, url in enumerate(spec.get('record', [])):
        try:
            doc = get_json(url)
            name = f'readiness-{index}.json'
            save(receipt['_out'] / name, doc)
            receipt['readiness']['documents'][url] = name
        except (OSError, ValueError) as error:
            receipt['readiness']['documents'][url] = 'unavailable: ' + repr(error)
    for check in spec.get('checks', []):
        try:
            actual = field(get_json(check['url']), check['field'])
        except (OSError, ValueError) as error:
            actual = 'unavailable: ' + repr(error)
        receipt['readiness']['checks'].append({**check, 'actual': actual, 'ok': same(actual, check['equals'])})
    if not all(c['ok'] for c in receipt['readiness']['checks']):
        return 'readiness_mismatch'
    return True


def client_outcome(plan, client, started):
    spec = plan['client']
    result = {'pid': client.pid, 'exit_code': client.returncode, 'elapsed_seconds': time.monotonic() - started,
              'summary_path': spec.get('summary'), 'status_ok': None}
    if client.returncode == 0 and spec.get('summary'):
        try:
            status = json.loads(Path(spec['summary']).read_text()).get('status')
            result['summary_status'] = status
            result['status_ok'] = spec.get('success_status') in (None, status)
        except (OSError, ValueError) as error:
            result.update(status_ok=False, failure='client summary unreadable: ' + repr(error))
    if client.returncode != 0:
        result['failure'] = f'client exit {client.returncode}'
    elif result['status_ok'] is False:
        result.setdefault('failure', f"client status {result.get('summary_status')!r}")
    return result


def launch(argv, env, out, label):
    files = [open(out / f'{label}.stdout', 'wb'), open(out / f'{label}.stderr', 'wb')]
    try:
        proc = subprocess.Popen(argv, cwd=out, env=q5_case.child_env(os.environ, env), stdin=subprocess.DEVNULL,
                                stdout=files[0], stderr=files[1], start_new_session=True)
    finally:
        for handle in files:
            handle.close()
    return proc


def supervise(plan, values, out, receipt, negative, state):
    """Preflight, launch, watch and clean up; `state` carries the monitor out even when this raises."""
    policy = plan['policy']
    reason = preflight(plan, policy, receipt)
    if reason is not None:
        receipt['verdict'] = {'verdict': 'PreflightRefused', 'reason': reason}
        return
    if negative:
        try:
            receipt['negative_control'] = nc.plan_ceiling(
                receipt['baseline']['free_plus_cached_bytes'], policy['free_cached_floor_bytes'],
                plan['workload']['ceiling_bytes'], plan['workload']['cap_bytes'])
        except nc.ControlError as error:
            receipt['verdict'] = {'verdict': 'PreflightRefused', 'reason': str(error)}
            return
    monitor, latch = q5_policy.Monitor(receipt['baseline'], policy), q5_runtime.StopLatch()
    sampler = q5_runtime.Sampler(monitor, policy['interval_seconds'], out / 'samples.jsonl', latch)
    state['monitor'] = monitor
    server = client = None
    popens = []

    def stop(name, detail):
        if latch.trip({'reason': name, 'detail': detail, 'sample_index': None, 'sample_start_ns': None, 'sample_end_ns': None}):
            monitor.stop = latch.stop

    sampler.start()
    try:
        server = launch(plan['serve']['argv'], plan['serve']['env'], out, 'server')
        popens.append(server)
        latch.groups.append(server.pid)
        sampler.pgid, sampler.roots = server.pid, {server.pid}
        values['SERVER_PID'] = server.pid
        receipt['server'] = {'pid': server.pid}
        if plan['readiness'] is not None:
            ready = wait_ready(plan, server, latch, receipt)
            if isinstance(ready, str):
                stop(ready, 'before client launch')
        if plan['client'] is not None and not latch.event.is_set():
            spec = plan['client']
            if spec.get('approval'):
                values.update(render_approval(spec['approval'], values, out))
            argv = q5_case.validate_argv(q5_case.expand(spec['argv'], values), 'client')
            receipt['client_argv'] = argv
            started = time.monotonic()
            client = launch(argv, spec['env'], out, 'client')
            popens.append(client)
            latch.groups.append(client.pid)
            sampler.roots = {server.pid, client.pid}
            while client.poll() is None and not latch.event.is_set():
                if INTERRUPTED:
                    stop('interrupted', 'operator signal')
                elif server.poll() is not None:
                    stop('server_exited', f'server exit {server.returncode} during client run')
                elif time.monotonic() - started > spec['timeout_seconds']:
                    stop('client_timeout', f"client exceeded {spec['timeout_seconds']} s")
                latch.event.wait(0.25)
            if latch.stop is None:
                receipt['client'] = client_outcome(plan, client, started)
        elif plan['client'] is None:
            limit = time.monotonic() + 240
            while server.poll() is None and not latch.event.is_set():
                if INTERRUPTED or time.monotonic() > limit:
                    stop('interrupted', 'operator signal or negative-control lifetime')
                latch.event.wait(0.25)
    finally:
        # A completed run is not stopped retroactively; later violations still fail the summary.
        latch.armed = False
        cleanup = q5_runtime.terminate([p.pid for p in popens], sampler.known, popens, policy['term_grace_seconds'],
                                       policy['kill_grace_seconds'], latch.signal_ns)
        time.sleep(policy['interval_seconds'] * 2)  # observe memory release after exit
        sampler.halt.set()
        sampler.join(10)
        if server is not None:
            receipt['server']['exit_code'] = server.returncode
        if client is not None and 'client' not in receipt:
            receipt['client'] = {'pid': client.pid, 'exit_code': client.returncode, 'stopped': latch.stop is not None}
        receipt['stop'] = latch.stop
        receipt['stop_latency'] = q5_policy.latency(latch.stop, latch.signal_ns, cleanup['empty_ns'])
        if latch.stop is None and monitor.stop is not None:
            receipt['post_completion_violation'] = monitor.stop
        receipt['sampler_error'] = sampler.error
        receipt['cleanup'] = cleanup


def finish(plan, receipt, monitor, negative):
    if plan['port'] is not None:
        try:
            pids = q5_host.listeners(plan['port'])
            receipt['listener_after_cleanup'] = {'pids': pids, 'clear': not pids}
        except q5_host.ObservationError as error:
            receipt['listener_after_cleanup'] = {'error': repr(error), 'clear': False}
    after = q5_runtime.take_sample(-1)
    receipt['post_cleanup_sample'] = q5_runtime.compact(after['host'])
    if monitor is None:
        return
    summary = monitor.summary()
    receipt['samples'] = {'file': 'samples.jsonl', **summary}
    if after['host'] is not None:
        receipt['recovery_free_plus_cached_delta_bytes'] = (after['host']['vm']['free_plus_cached_bytes']
                                                            - receipt['baseline']['free_plus_cached_bytes'])
    if negative and summary['peak_footprint_bytes'] is not None:
        receipt['negative_control']['peak_over_cap_bytes'] = summary['peak_footprint_bytes'] - plan['policy']['footprint_cap_bytes']
    proven = receipt['cleanup']['proven'] and receipt.get('listener_after_cleanup', {}).get('clear', True)
    receipt['cleanup_proven'] = proven
    if receipt.get('supervisor_error'):
        receipt['verdict'] = {'verdict': 'SupervisorError', 'reason': receipt['supervisor_error']}
        return
    verdict = q5_policy.classify(receipt['stop'], receipt.get('client'), summary, proven, plan['policy'])
    if negative:
        receipt['calibration_verdict'] = verdict
        verdict = q5_policy.classify_negative_control(verdict, summary, plan['workload']['ceiling_bytes'], proven,
                                                      receipt['server'].get('exit_code'))
    receipt['verdict'] = verdict


def identities(plan, case_path, receipt):
    receipt['supervisor_sources'] = {name: sha256(HERE / name) for name in SOURCES}
    receipt['case'] = None if case_path is None else {'path': str(case_path), 'sha256': sha256(case_path)}
    receipt['inputs'] = input_hashes(plan)
    binary = plan['binary']
    receipt['binary'] = {'path': binary, 'sha256': sha256(binary), 'bytes': os.path.getsize(binary)}
    if plan.get('artifact'):
        stat = os.stat(plan['artifact']['path'])
        receipt['artifact'] = {'path': plan['artifact']['path'], 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                               'provided_sha256': plan['artifact']['sha256'], 'sha256_verified_by_supervisor': False}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case', nargs='?', type=Path, help='calibration case JSON (omit with --negative-control)')
    parser.add_argument('--output', type=Path, help='new receipt directory outside any Git tree')
    parser.add_argument('--dry-run', action='store_true', help='validate and print the plan; launch nothing')
    parser.add_argument('--footprint-cap-bytes', type=int, help='calibrated hf2q footprint cap (overrides the case)')
    parser.add_argument('--negative-control', action='store_true', help='run the synthetic memory-growth control')
    parser.add_argument('--nc-ceiling-bytes', type=int, default=8 << 30)
    parser.add_argument('--nc-cap-bytes', type=int, help='footprint stop for the control (default ceiling/2)')
    parser.add_argument('--nc-step-bytes', type=int, default=256 << 20)
    parser.add_argument('--nc-step-seconds', type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.negative_control == (args.case is not None):
        parser.error('give exactly one of CASE or --negative-control')
    if not args.dry_run and args.output is None:
        parser.error('--output is required unless --dry-run')
    return args


def main(argv=None):
    args = parse_args(argv)
    out = (args.output or Path('/OUT-not-given-in-dry-run')).absolute()
    try:
        base = builtins(out)
        if args.negative_control:
            plan = negative_plan(args, out)
            nc.plan_ceiling(1 << 62, plan['policy']['free_cached_floor_bytes'], args.nc_ceiling_bytes, plan['workload']['cap_bytes'])
        else:
            case = q5_case.loads(args.case.read_bytes())
            plan = q5_case.build_plan(case, base, os.environ)
        if args.footprint_cap_bytes is not None:
            plan['policy'] = q5_policy.effective_policy({**plan['policy'], 'footprint_cap_bytes': args.footprint_cap_bytes})
    except (OSError, q5_case.CaseError, q5_policy.PolicyError, nc.ControlError) as error:
        print(json.dumps({'valid': False, 'error': str(error)}, indent=2))
        return 2
    checks = file_checks(plan) if plan['complete'] else {}
    if args.dry_run:
        print(json.dumps({'valid': True, 'dry_run': True, 'launchable': plan['complete'] and all(checks.values()),
                          'file_checks': checks, 'plan': plan}, indent=2))
        return 0 if plan['complete'] and all(checks.values()) else 2
    if not plan['complete'] or not all(checks.values()):
        print(json.dumps({'valid': False, 'missing_variables': plan['missing_variables'], 'file_checks': checks}, indent=2))
        return 2
    if out.exists() or inside_git(out.parent.resolve()):
        print(json.dumps({'valid': False, 'error': 'output must be a new directory outside any Git tree'}))
        return 2
    os.umask(0o077)
    out.mkdir(mode=0o700)
    values = {**base, **plan.get('variables', {})}
    receipt = {'schema': RECEIPT_SCHEMA, 'mode': 'negative-control' if args.negative_control else 'calibration',
               'family': plan['family'], 'workload': plan['workload'], 'policy': plan['policy'],
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'host': q5_host.host_identity(), 'serve_argv': plan['serve']['argv'], 'serve_env': plan['serve']['env'],
               'independent_review_required': True, 'performance_claim': None, '_out': out}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, _frame: INTERRUPTED.append(signum))
    state = {}
    try:
        identities(plan, args.case.resolve() if args.case else None, receipt)
        receipt['plan_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        supervise(plan, values, out, receipt, args.negative_control, state)
    except Exception as error:
        receipt['supervisor_error'] = repr(error)
    finally:
        try:
            finish(plan, receipt, state.get('monitor'), args.negative_control)
        except Exception as error:
            receipt['finish_error'] = repr(error)
        receipt.setdefault('verdict', {'verdict': 'SupervisorError', 'reason': receipt.get('supervisor_error')})
        receipt['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt.pop('_out')
        save(out / 'receipt.json', receipt)
    print(json.dumps({'receipt': str(out / 'receipt.json'), 'verdict': receipt['verdict'],
                      'stop': receipt.get('stop'), 'cleanup_proven': receipt.get('cleanup_proven')}, indent=2))
    verdict = receipt['verdict']['verdict']
    return 0 if verdict in ('CalibrationPass', 'NegativeControlPass') else 2 if verdict == 'PreflightRefused' else 1


if __name__ == '__main__':
    raise SystemExit(main())
