"""ADR-063 Q5 single-runtime preflight: competing runtime detection and port checks.

Matching reads the executable (argv[0]) and, for interpreters/`env`, the program
they run. It never searches free text in the command line: an agent or editor
whose prompt merely mentions "hf2q" is not a runtime.
"""
import errno
import os
import re
import socket

RUNTIME_EXECUTABLES = {
    'hf2q', 'llama-server', 'llama-cli', 'llama-bench', 'llama-mtmd-cli', 'llama-run',
    'llama-simple', 'llama-speculative', 'ollama', 'ollama_llama_server', 'lms',
    'mlx_lm.server', 'mlx_lm.generate', 'mlx_lm.chat', 'mlx_vlm.server', 'mlx_vlm.generate',
    'vllm', 'sglang', 'koboldcpp',
}
# Hosted peers started as `python -m <module>` or via the module's console script.
PYTHON_MODULE_ROOTS = ('mlx_lm', 'mlx_vlm', 'vllm', 'sglang')
BUILD_EXECUTABLES = {'cargo', 'rustc'}
LAUNCHER = re.compile(r'serve_[a-z0-9_]+_opencode\.sh')
SHELLS = {'sh', 'bash', 'zsh', 'dash', 'ksh', 'fish'}
PYTHON = re.compile(r'(?i)python(?:[0-9](?:\.[0-9]+)?)?t?')
PYTHON_VALUE_OPTIONS = {'-W', '-X', '--check-hash-based-pycs'}


def _base(path):
    return os.path.basename(path).lstrip('-')


def _runtime_name(name):
    if name in RUNTIME_EXECUTABLES or name.startswith('hf2q-'):
        return 'runtime'
    if name.split('.')[0] in PYTHON_MODULE_ROOTS:
        return 'python-hosted-runtime'
    return None


def classify_argv(argv):
    """Category of a competing process or None; argv is the process's argument vector."""
    if not argv:
        return None
    name = _base(argv[0])
    if name == 'env':
        rest = list(argv[1:])
        while rest and (rest[0].startswith('-') or '=' in rest[0]):
            rest.pop(0)
        return classify_argv(rest)
    found = _runtime_name(name)
    if found is not None:
        return found
    if name in BUILD_EXECUTABLES:
        return 'build'
    if PYTHON.fullmatch(name):
        args = argv[1:]
        index = 0
        while index < len(args):
            arg = args[index]
            if arg == '-m' and index + 1 < len(args):
                return 'python-hosted-runtime' if args[index + 1].split('.')[0] in PYTHON_MODULE_ROOTS else None
            if arg == '-c':
                return None
            if arg in PYTHON_VALUE_OPTIONS:
                index += 2
                continue
            if arg.startswith('-'):
                index += 1
                continue
            script = _base(arg)
            return 'python-hosted-runtime' if _runtime_name(script) else None
        return None
    if name in SHELLS:
        for arg in argv[1:]:
            if arg == '-c':
                return None
            if arg.startswith('-'):
                continue
            return 'launcher' if LAUNCHER.fullmatch(_base(arg)) else None
    return None


def ancestors(rows, pid):
    """PIDs of `pid` and every ancestor present in the process table."""
    parent = {row['pid']: row['ppid'] for row in rows}
    chain, current = set(), pid
    while current in parent and current not in chain:
        chain.add(current)
        current = parent[current]
    chain.add(pid)
    return chain


def competitors(rows, ignore_pids):
    """Every process that would make this run not the single runtime on the host."""
    found = []
    for row in rows:
        if row['pid'] in ignore_pids:
            continue
        category = classify_argv(row['argv'])
        if category is not None:
            found.append({'pid': row['pid'], 'pgid': row['pgid'], 'category': category,
                          'rss_bytes': row['rss_bytes'], 'args': row['args'][:300]})
    return found


def port_connectable(port, host='127.0.0.1', timeout=0.5):
    """True unless a TCP connect to host:port is refused (timeouts and errors fail closed)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        code = probe.connect_ex((host, port))
    return code != errno.ECONNREFUSED


def port_state(port, listener_pids):
    """Free only when no listener is visible to lsof and a connect probe is refused."""
    connectable = port_connectable(port)
    return {'port': port, 'listener_pids': listener_pids, 'connectable': connectable,
            'free': not listener_pids and not connectable}
