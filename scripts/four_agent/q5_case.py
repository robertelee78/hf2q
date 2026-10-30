"""ADR-063 Q5 calibration case files: strict loading, placeholder expansion, argv rules. Pure.

Placeholders are `{NAME}`. Values come from, in order: built-ins (REPO, OUT,
PYTHON, PYTHON_SHA256, RUN_ID, NONCE32), the case's `variables` (a null value
is required input), and `HF2Q_Q5_<NAME>` environment overrides of declared
variables. SERVER_PID, CLIENT_APPROVAL and CLIENT_APPROVAL_SHA256 exist only
after launch and are allowed only in the client section.
"""
import json
import os
import re

SCHEMA = 'hf2q.q5.calibration-case.v1'
FAMILIES = ('qwen38', 'qwen36', 'gemma4', 'deepseek4')
BUILTINS = ('REPO', 'OUT', 'PYTHON', 'PYTHON_SHA256', 'RUN_ID', 'NONCE32')
RUNTIME = ('SERVER_PID', 'CLIENT_APPROVAL', 'CLIENT_APPROVAL_SHA256')
# SERVED_MODEL: the path the server loads; the artifact itself, or its hard link in
# {OUT}/text-model/ when artifact.text_only_locator is true (q5_artifacts).
DERIVED = ('SERVED_MODEL',)
OVERRIDE_PREFIX = 'HF2Q_Q5_'
PLACEHOLDER = re.compile(r'\{([A-Z][A-Z0-9_]*)\}')
NAME = re.compile(r'[A-Z][A-Z0-9_]*')
SHELLS = {'sh', 'bash', 'zsh', 'dash', 'ksh', 'fish', 'csh', 'tcsh'}
# Children receive only these ambient variables plus the case's explicit env.
PASSTHROUGH = ('PATH', 'HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL', 'LC_CTYPE', 'SHELL', 'TERM')
TOP_KEYS = {'schema', 'family', 'workload', 'variables', 'binary', 'artifact', 'port', 'serve',
            'readiness', 'client', 'policy', 'notes'}


class CaseError(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise CaseError(why)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def loads(raw):
    try:
        return json.loads(raw, object_pairs_hook=_unique,
                          parse_constant=lambda c: (_ for _ in ()).throw(CaseError('nonfinite JSON ' + c)))
    except json.JSONDecodeError as error:
        raise CaseError(f'invalid JSON: {error}') from error


def validate(case):
    """Structural checks independent of variable values."""
    require(isinstance(case, dict) and set(case) <= TOP_KEYS, 'unknown or missing top-level keys')
    require(case.get('schema') == SCHEMA, 'schema must be ' + SCHEMA)
    require(case.get('family') in FAMILIES, 'family must be one of ' + ','.join(FAMILIES))
    workload = case.get('workload')
    require(isinstance(workload, dict) and isinstance(workload.get('id'), str) and workload['id'], 'workload.id')
    require(workload.get('slots') == 4, 'Q5 calibrates the four-slot configuration (workload.slots == 4)')
    require(type(workload.get('context_per_slot')) is int and workload['context_per_slot'] > 0,
            'workload.context_per_slot must be a positive integer')
    variables = case.get('variables', {})
    require(isinstance(variables, dict), 'variables must be an object')
    for name, value in variables.items():
        require(NAME.fullmatch(name) is not None, 'variable names are UPPER_SNAKE: ' + name)
        require(name not in BUILTINS + RUNTIME + DERIVED, 'variable shadows a built-in: ' + name)
        require(value is None or isinstance(value, str), f'variable {name} must be a string or null')
    for key in ('binary', 'port'):
        require(isinstance(case.get(key), str), key + ' must be a string')
    artifact = case.get('artifact')
    require(isinstance(artifact, dict) and set(artifact) <= {'path', 'sha256', 'text_only_locator'}
            and isinstance(artifact.get('path'), str) and type(artifact.get('text_only_locator', False)) is bool,
            'artifact needs path, optional sha256 and optional boolean text_only_locator')
    require('SERVED_MODEL' not in PLACEHOLDER.findall(artifact['path']), 'artifact.path cannot use {SERVED_MODEL}')
    serve = case.get('serve')
    require(isinstance(serve, dict) and set(serve) <= {'argv', 'env'}, 'serve needs argv and optional env')
    _check_section_env(serve, 'serve')
    readiness = case.get('readiness')
    require(isinstance(readiness, dict) and set(readiness) <= {'url', 'field', 'equals', 'timeout_seconds', 'record', 'checks'},
            'readiness keys')
    require(isinstance(readiness.get('url'), str) and isinstance(readiness.get('field'), str) and 'equals' in readiness,
            'readiness needs url, field and equals')
    require(_positive(readiness.get('timeout_seconds')), 'readiness.timeout_seconds')
    require(isinstance(readiness.get('record', []), list) and all(isinstance(u, str) for u in readiness.get('record', [])),
            'readiness.record must be a list of URLs')
    require(isinstance(readiness.get('checks', []), list), 'readiness.checks must be a list')
    for check in readiness.get('checks', []):
        require(isinstance(check, dict) and set(check) == {'url', 'field', 'equals'}, 'readiness check needs url/field/equals')
    client = case.get('client')
    require(isinstance(client, dict) and set(client) <= {'argv', 'env', 'approval', 'summary', 'success_status', 'timeout_seconds'},
            'client keys')
    _check_section_env(client, 'client')
    require(_positive(client.get('timeout_seconds')), 'client.timeout_seconds')
    require(all(isinstance(client.get(k, ''), str) for k in ('summary', 'success_status')),
            'client.summary and client.success_status must be strings')
    approval = client.get('approval')
    if approval is not None:
        require(isinstance(approval, dict) and set(approval) == {'template', 'set'} and isinstance(approval['template'], str)
                and isinstance(approval['set'], dict), 'client.approval needs a template path and a set object')
        for name in referenced({'template': approval['template']}):
            require(name not in RUNTIME, 'client.approval.template cannot depend on launch-time values')
    require(isinstance(case.get('policy', {}), dict), 'policy must be an object')
    require('footprint_cap_bytes' not in case.get('policy', {}),
            'the footprint cap comes from --admission-receipt (calibrated peak), never from the case')
    for name in referenced(case, sections=('binary', 'artifact', 'port', 'serve', 'readiness')):
        require(name not in RUNTIME, f'{{{name}}} is known only after launch; use it in the client section')
    for name in referenced(case):
        require(name in BUILTINS + RUNTIME + DERIVED or name in variables, f'undeclared placeholder {{{name}}}')
    return case


def _positive(value):
    return type(value) in (int, float) and value > 0


def _check_section_env(section, label):
    argv = section.get('argv')
    require(isinstance(argv, list) and argv and all(isinstance(a, str) for a in argv), label + '.argv must be a nonempty string list')
    env = section.get('env', {})
    require(isinstance(env, dict) and all(NAME.fullmatch(k) and isinstance(v, str) for k, v in env.items()),
            label + '.env must map UPPER_SNAKE names to strings')


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def referenced(case, sections=None):
    keys = sections or [k for k in case if k not in ('notes', 'schema', 'variables')]
    names = set()
    for key in keys:
        for text in _strings(case.get(key)):
            names.update(PLACEHOLDER.findall(text))
    for text in _strings(case.get('variables', {}) if sections is None else {}):
        names.update(PLACEHOLDER.findall(text))
    return names


def resolve_variables(case, builtins, environ):
    """Final variable values; returns (values, missing names). Overrides must name declared variables."""
    declared = case.get('variables', {})
    unknown = sorted(k[len(OVERRIDE_PREFIX):] for k in environ
                     if k.startswith(OVERRIDE_PREFIX) and k[len(OVERRIDE_PREFIX):] not in declared)
    require(not unknown, 'override names no declared variable: ' + ','.join(unknown))
    values = dict(builtins)
    raw = {name: environ.get(OVERRIDE_PREFIX + name, value) for name, value in declared.items()}
    missing = sorted(name for name, value in raw.items() if value is None)
    pending = {name: value for name, value in raw.items() if value is not None}
    for _ in range(len(pending) + 1):
        progressed = False
        for name, value in list(pending.items()):
            needed = set(PLACEHOLDER.findall(value))
            require(not needed & set(RUNTIME), f'variable {name} uses a launch-time placeholder')
            if needed & set(missing):
                missing.append(name)
                del pending[name]
                progressed = True
            elif needed <= set(values):
                values[name] = expand(value, values)
                del pending[name]
                progressed = True
        if not progressed:
            break
    require(not pending, 'cyclic variables: ' + ','.join(sorted(pending)))
    return values, sorted(set(missing))


def expand(value, values, keep=()):
    """Substitute placeholders in strings (recursively); names in `keep` stay verbatim."""
    if isinstance(value, list):
        return [expand(v, values, keep) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, values, keep) for k, v in value.items()}
    if not isinstance(value, str):
        return value

    def substitute(match):
        name = match.group(1)
        if name in keep:
            return match.group(0)
        require(name in values, f'unresolved placeholder {{{name}}}')
        return str(values[name])
    return PLACEHOLDER.sub(substitute, value)


def validate_argv(argv, label):
    """Exec-ready argv: absolute program, no shell command strings, nothing unresolved."""
    require(isinstance(argv, list) and argv and all(isinstance(a, str) and a for a in argv),
            label + ': argv must be nonempty strings')
    require(all('\0' not in a for a in argv), label + ': NUL in argv')
    require(not any(PLACEHOLDER.search(a) for a in argv), label + ': unresolved placeholder in argv')
    require(os.path.isabs(argv[0]), label + ': argv[0] must be an absolute path (no PATH lookup)')
    program = os.path.basename(argv[0])
    require(program != 'env', label + ': use the env object, not env(1)')
    if program in SHELLS:
        require(not any(a == '-c' or (a.startswith('-') and not a.startswith('--') and 'c' in a[1:]) for a in argv[1:]),
                label + ': shell command strings (-c) are not allowed; give the script path')
        require(len(argv) >= 2 and os.path.isabs(argv[1]), label + ': shell argv[1] must be an absolute script path')
    return argv


def child_env(environ, explicit):
    env = {k: environ[k] for k in PASSTHROUGH if k in environ}
    env.update(explicit)
    return env


def runtime_value(template, values):
    """Expand an approval `set` value; a value that is exactly one placeholder keeps the value's type."""
    if isinstance(template, str):
        whole = PLACEHOLDER.fullmatch(template)
        if whole is not None:
            require(whole.group(1) in values, f'unresolved placeholder {template}')
            return values[whole.group(1)]
    return expand(template, values)


def build_plan(case, builtins, environ):
    """Expanded, exec-ready plan. Missing required variables stay verbatim and mark it incomplete."""
    from q5_policy import effective_policy, PolicyError
    validate(case)
    values, missing = resolve_variables(case, builtins, environ)
    artifact_path = expand(case['artifact']['path'], values, tuple(missing))
    keep_served = ()
    if PLACEHOLDER.search(artifact_path):
        keep_served = ('SERVED_MODEL',)  # a missing variable leaves the served path unknown too
    else:
        values['SERVED_MODEL'] = (os.path.join(values['OUT'], 'text-model', os.path.basename(artifact_path))
                                  if case['artifact'].get('text_only_locator') else artifact_path)
    keep = RUNTIME + tuple(missing) + keep_served
    sub = lambda value: expand(value, values, keep)
    plan = {'family': case['family'], 'workload': case['workload'], 'missing_variables': missing,
            'complete': not missing, 'variables': {k: values[k] for k in case.get('variables', {}) if k in values},
            'binary': sub(case['binary']), 'artifact': sub(case['artifact']), 'port': sub(case['port']),
            'serve': {'argv': sub(case['serve']['argv']), 'env': sub(case['serve'].get('env', {}))},
            'readiness': sub(case['readiness']),
            'client': {**sub(case['client']), 'env': sub(case['client'].get('env', {}))},
            'served_model': values.get('SERVED_MODEL')}
    try:
        plan['policy'] = effective_policy(case.get('policy', {}))
    except PolicyError as error:
        raise CaseError(str(error)) from error
    if missing:
        return plan
    require(plan['port'].isdecimal() and 1024 <= int(plan['port']) <= 65535, 'port must be 1024..65535')
    plan['port'] = int(plan['port'])
    require(os.path.isabs(plan['binary']) and os.path.isabs(plan['artifact']['path']), 'binary/artifact paths must be absolute')
    digest = plan['artifact'].get('sha256') or None
    require(digest is None or re.fullmatch(r'[0-9a-f]{64}', digest) is not None, 'artifact.sha256 must be 64 lowercase hex')
    plan['artifact']['sha256'] = digest
    validate_argv(plan['serve']['argv'], 'serve')
    probe = {**values, 'SERVER_PID': 1, 'CLIENT_APPROVAL': '/approval.json', 'CLIENT_APPROVAL_SHA256': '0' * 64}
    validate_argv(expand(plan['client']['argv'], probe), 'client')
    for url in [plan['readiness']['url']] + [c['url'] for c in plan['readiness'].get('checks', [])] + plan['readiness'].get('record', []):
        require(re.fullmatch(r'http://127\.0\.0\.1:[0-9]+/\S*', url) is not None, 'readiness URLs must be loopback http: ' + url)
    return plan
