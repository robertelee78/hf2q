#!/usr/bin/env python3
"""Splash best-valid-configuration sweep helpers (ADR-063 "Splash peer of record for the Qwen families";
docs/research/adr063-splash-peer-design.md section 7). No launches: the root runs each derived case with
q5_calibration.py on a granted GPU lane, then `select` reads the settled runs.

  identity     print the case variables the Splash build binds (SPLASH_BIN/METALLIB/HOST_STEP/TOOLCHAIN/
               VENV_FREEZE_SHA256) from a built checkout. Reads the binaries and runs the toolchain version commands;
               no model file is read. --build-toolchain is the toolchain.txt the build recorded: every host toolchain
               version line must appear in it, so SPLASH_TOOLCHAIN_SHA256 names the toolchain that built the binaries
               (a later Xcode or Metal update is refused here and at every launch).
  derive-case  write a copy of q5_cases/splash-<family>.json for one sweep point (native_peer_splash.PRESETS), with
               an optional --max-memory cap (decision D6); only SPLASH_CONFIG, the workload id and the notes change.
  select       validity and choice over settled runs: a configuration is valid when every one of its runs is Q5
               CalibrationPass, passes the run audits (q2_run_audit with the Splash audit module) and has a
               PassedByteIdentical prompt-parity post-oracle in RUN/splash-prompt-parity, and Splash's
               task pass fraction over its runs is at least the Q1 numeric-policy floor (0.75). Splash's own outcomes
               decide; no hf2q arm is involved (decision D10). The best is the lowest median makespan among valid
               configurations; it is provisional when the runner-up's median is within the Q3 half-width target
               (2.5 points) of it, and both are then re-run before the campaign of record.
  parity-config write the frozen config of the hf2q prompt-parity post-oracle for one settled Splash run: each
               archived request's canonical hf2q form (native_peer_splash.canonical) beside Splash's archived
               `/apply-template` rendering, bound by sha256, with the reviewed GGUF template. Then run
               `HF2Q_SPLASH_PARITY_POST_ORACLE_CONFIG=<config> cargo test --locked --bin hf2q
               splash_prompt_parity_post_oracle_tests::splash_prompt_parity_post_oracle -- --ignored` (host lock
               absent); a byte mismatch fails it and makes the run Invalid.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import native_peer_splash as sp
import q3_stats
import quality_campaign
from qwen38_n1_verdict import require

HERE = Path(__file__).resolve().parent
PASS_FLOOR = quality_campaign.POLICY["paired_min_pass_fraction"]


def build_toolchain_problems(versions, recorded):
    """Host toolchain version lines missing from the build's toolchain.txt (clang/metal install directories, which
    a Metal toolchain update relocates, are not compared)."""
    lines = {line.strip() for line in recorded.splitlines()}
    return sorted(f'{name}: {line.strip()}' for name, text in versions.items() for line in text.splitlines()
                  if line.strip() and not line.startswith('InstalledDir:') and line.strip() not in lines
                  and not (name == 'clang' and line.startswith(('Target:', 'Thread model:'))))


def identity(splash_dir, build_toolchain):
    import native_peer_splash_ready as ready
    paths = ready.layout({'splash_dir': str(Path(splash_dir).resolve()), 'run_dir': '/'})
    tools = ready.toolchain()
    missing = build_toolchain_problems(tools['versions'], Path(build_toolchain).read_text())
    require(not missing, 'the host toolchain is not the one the Splash build recorded: ' + '; '.join(missing))
    venv = ready.venv_inventory(paths['root'])
    return {'SPLASH_BIN_SHA256': ready.sha256(paths['binary']), 'SPLASH_METALLIB_SHA256': ready.sha256(paths['metallib']),
            'SPLASH_HOST_STEP_SHA256': ready.sha256(paths['host_step']), 'SPLASH_TOOLCHAIN_SHA256': tools['sha256'],
            'SPLASH_VENV_FREEZE_SHA256': venv['freeze_sha256'], 'toolchain': tools['versions'],
            'build_toolchain': {'path': str(Path(build_toolchain).resolve()), 'sha256': ready.sha256(Path(build_toolchain))}}


def derive_case(case, point, max_memory=None):
    """The Splash case of one sweep point; refuses a point that does not apply to the case's family."""
    q3 = sp.q3_family(case)
    knobs = sp.parse_config(sp.preset(point, q3), q3)
    if max_memory is not None:
        knobs['max_memory'] = max_memory
    config = sp.config_text(sp.parse_config(sp.config_text(knobs), q3))
    derived = json.loads(json.dumps(case))
    derived['variables']['SPLASH_CONFIG'] = config
    slug = point + ('' if max_memory is None else f'-cap{max_memory}')
    derived['workload']['id'] = f"{case['workload']['id']}-sweep-{slug}"
    derived['notes'] = list(case.get('notes', [])) + [
        f'Derived by splash_sweep.py: sweep point {point} (SPLASH_CONFIG {config})'
        + ('' if max_memory is None else f'; --max-memory {max_memory} recorded per decision D6') + '.']
    return derived


def select(records):
    """records: native_peer_splash_audit.run_record outputs. Returns the validity table and the choice. A run that
    names no configuration (it stopped before Splash launched) is its own invalid row."""
    by_config = {}
    for record in records:
        config = record.get('config') if isinstance(record.get('config'), str) else f"unconfigured run {record['run']}"
        by_config.setdefault(config, []).append(record)
    table = {}
    for config, runs in sorted(by_config.items()):
        reasons = []
        if any(not isinstance(r.get('config'), str) for r in runs):
            reasons.append('the run records no Splash configuration (no readiness and no launch.json)')
        if any(r['q5_verdict'] != 'CalibrationPass' for r in runs):
            reasons.append('a run is not Q5 CalibrationPass')
        if any(r['audit'] != 'Pass' for r in runs):
            reasons.append('a run failed its audit: ' + '; '.join(str(r['audit']) for r in runs if r['audit'] != 'Pass'))
        if any(r.get('prompt_parity') != 'PassedByteIdentical' for r in runs):
            # Design section 7.1: request parity on every run; section 6.4: a mismatch makes the run Invalid.
            reasons.append('a run lacks a PassedByteIdentical prompt-parity post-oracle: ' + '; '.join(
                f"{r['run']}: {r.get('prompt_parity')}" for r in runs if r.get('prompt_parity') != 'PassedByteIdentical'))
        passed = sum(r['tasks_passed'] or 0 for r in runs)
        agents = 4 * len(runs)
        fraction = passed / agents if agents else 0.0
        if fraction < PASS_FLOOR:
            reasons.append(f'task pass fraction {fraction:.3f} below the Q1 floor {PASS_FLOOR}')
        spans = [r['makespan_s'] for r in runs if isinstance(r.get('makespan_s'), (int, float))]
        table[config] = {'runs': [r['run'] for r in runs], 'valid': not reasons, 'reasons': reasons,
                         'task_pass_fraction': fraction, 'makespan_median_s': statistics.median(spans) if spans else None,
                         'makespans_s': spans}
    ranked = sorted((row['makespan_median_s'], config) for config, row in table.items()
                    if row['valid'] and row['makespan_median_s'] is not None)
    choice = None
    if ranked:
        best, config = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        provisional = runner is not None and runner[0] <= best * (1 + q3_stats.HALF_WIDTH_PP / 100)
        choice = {'config': config, 'makespan_median_s': best, 'provisional': provisional,
                  'runner_up': None if runner is None else {'config': runner[1], 'makespan_median_s': runner[0]},
                  'rule': 'lowest median makespan among valid configurations; provisional within '
                          f'{q3_stats.HALF_WIDTH_PP} points of the runner-up (re-run both, 3 runs each)'}
    return {'schema': 'hf2q.splash-sweep-selection.v1', 'pass_floor': PASS_FLOOR, 'configurations': table,
            'choice': choice}


ARCHITECTURES = {'qwen38': 'qwen35', 'qwen36': 'qwen35moe'}


def parity_config(run_dir, template, output_dir=None):
    """Oracle config (and the canonical request files it binds) for one settled Splash run. The default directory,
    {run}/splash-prompt-parity, is where run_record (and so `select`) reads the oracle's result."""
    import hashlib
    from native_peer_splash_audit import PARITY_DIR
    run = Path(run_dir).resolve()
    out = Path(output_dir).resolve() if output_dir is not None else run / PARITY_DIR
    summary = json.loads((run / 'client' / 'summary.json').read_text())
    ready = summary.get('native_readiness') or {}
    q3 = ready.get('q3_family')
    require(summary.get('engine_kind') == sp.ENGINE and q3 in sp.ARTIFACTS, f'{run} is not a settled Splash run')
    template_bytes = Path(template).read_bytes()
    digest = lambda data: hashlib.sha256(data).hexdigest()
    require(digest(template_bytes) == sp.ARTIFACTS[q3]['template_sha256'],
            f"{template} is not the reviewed {q3} template the GGUF embeds")
    require(not out.exists(), f'{out} exists')
    require(not any((p / '.git').exists() for p in (out.parent, *out.parent.parents)), 'output must be outside Git')
    out.mkdir(mode=0o700)
    turns = []
    for agent_summary in sorted((run / 'client').glob('agent-*/summary.json')):
        agent = agent_summary.parent.name
        for receipt in json.loads(agent_summary.read_text())['turns']:
            directory = agent_summary.parent / f"turn-{receipt['turn']}"
            raw = (directory / 'request.json').read_bytes()
            require(digest(raw) == receipt['request_sha256'], f'{agent} turn {receipt["turn"]}: request archive changed')
            canonical = sp.canonical(json.loads(raw), 'qwen')
            path = out / f"{agent}-turn-{receipt['turn']}.canonical-request.json"
            body = json.dumps(canonical, ensure_ascii=False, separators=(',', ':')).encode()
            path.write_bytes(body)
            rendered = directory / 'splash-apply-template.raw'
            turns.append({'agent': agent, 'turn': receipt['turn'],
                          'canonical_request': {'path': str(path), 'sha256': digest(body)},
                          'splash_apply_template': {'path': str(rendered), 'sha256': digest(rendered.read_bytes())}})
    require(turns, f'{run} has no archived turns')
    config = {'architecture': ARCHITECTURES[q3],
              'template': {'path': str(Path(template).resolve()), 'sha256': digest(template_bytes)},
              'turns': turns, 'output': str(out / 'splash-prompt-parity-post-oracle.json')}
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    return out / 'config.json'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    ident = sub.add_parser('identity', help='print the build identity variables of a Splash checkout')
    ident.add_argument('--splash-dir', required=True)
    ident.add_argument('--build-toolchain', type=Path, required=True,
                       help="the build's toolchain.txt (xcodebuild/metal/clang/sdk/sw_vers output at build time)")
    derive = sub.add_parser('derive-case', help='write the Splash case of one sweep point')
    derive.add_argument('--family', choices=sorted(sp.ARTIFACTS), required=True)
    derive.add_argument('--point', choices=sorted(sp.PRESETS), required=True)
    derive.add_argument('--max-memory', help='decision D6 cap, e.g. 96G (only after a Q5 floor trip)')
    derive.add_argument('--case', type=Path, help='base case (default q5_cases/splash-<family>.json)')
    derive.add_argument('--out', type=Path, required=True)
    pick = sub.add_parser('select', help='validity and best configuration over settled sweep runs')
    pick.add_argument('runs', nargs='+', type=Path, help='q5 run directories of Splash sweep cases')
    parity = sub.add_parser('parity-config', help='prompt-parity post-oracle config for one settled Splash run')
    parity.add_argument('run', type=Path, help='q5 run directory of a Splash case')
    parity.add_argument('--template', type=Path, required=True,
                        help='reviewed GGUF-embedded template (rendered-inputs-reviewed/<family>.template.jinja)')
    parity.add_argument('--output-dir', type=Path,
                        help='new directory outside Git (default RUN/splash-prompt-parity, which select reads)')
    args = parser.parse_args(argv)
    if args.command == 'identity':
        print(json.dumps(identity(args.splash_dir, args.build_toolchain), indent=2))
    elif args.command == 'derive-case':
        import q5_case
        base = args.case or HERE / 'q5_cases' / f'splash-{args.family}.json'
        case = q5_case.validate(q5_case.loads(Path(base).read_bytes()))
        require(case['family'] == args.family, f'{base} is not family {args.family}')
        require(not args.out.exists(), f'{args.out} exists')
        args.out.write_text(json.dumps(q5_case.validate(derive_case(case, args.point, args.max_memory)), indent=2) + '\n')
    elif args.command == 'parity-config':
        print(parity_config(args.run, args.template, args.output_dir))
    else:
        from native_peer_splash_audit import run_record
        print(json.dumps(select([run_record(path) for path in args.runs]), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
