"""ADR-063 Q5 host memory observations: pure parsers plus thin macOS command wrappers.

Every wrapper runs a fixed system binary with an argv list (never a shell) and a
short timeout. Parsers reject output they do not fully understand instead of
returning partial numbers; a failed observation is a blind sample, never zero.
"""
import re
import subprocess
import time

MIB = 1 << 20
GIB = 1 << 30
VM_STAT = '/usr/bin/vm_stat'
SYSCTL = '/usr/sbin/sysctl'
PS = '/bin/ps'
FOOTPRINT = '/usr/bin/footprint'
LSOF = '/usr/sbin/lsof'
OSASCRIPT = '/usr/bin/osascript'
# kern.memorystatus_vm_pressure_level: the kernel's pressure state machine.
PRESSURE_NAMES = {1: 'normal', 2: 'warning', 4: 'critical'}
THERMAL_NAMES = {0: 'nominal', 1: 'fair', 2: 'serious', 3: 'critical'}
VM_KEYS = {
    'Pages free': 'free', 'Pages speculative': 'speculative', 'File-backed pages': 'file_backed',
    'Anonymous pages': 'anonymous', 'Pages occupied by compressor': 'compressor_occupied',
    'Pages stored in compressor': 'compressor_stored', 'Pages wired down': 'wired',
    'Pages active': 'active', 'Pages inactive': 'inactive', 'Pages purgeable': 'purgeable',
}
REQUIRED_VM = ('free', 'speculative', 'file_backed', 'anonymous', 'compressor_occupied', 'wired')
UNITS = {'K': 1 << 10, 'M': MIB, 'G': GIB, 'T': 1 << 40}
PS_COLUMNS = 'pid=,ppid=,pgid=,rss=,lstart=,args='


class ObservationError(RuntimeError):
    pass


def parse_vm_stat(text):
    """Page counts and byte totals; Free+cached = free + speculative + file-backed pages."""
    header = re.search(r'page size of ([0-9]+) bytes', text)
    if header is None:
        raise ObservationError('vm_stat page size missing')
    page = int(header.group(1))
    pages = {}
    for line in text.splitlines()[1:]:
        match = re.fullmatch(r'"?([^":]+)"?:\s+([0-9]+)\.', line.strip())
        if match is None:
            continue  # Unknown/new rows are ignored; every required row is still enforced below.
        key = VM_KEYS.get(match.group(1))
        if key is not None:
            if key in pages:
                raise ObservationError('duplicate vm_stat key: ' + match.group(1))
            pages[key] = int(match.group(2))
    missing = [key for key in REQUIRED_VM if key not in pages]
    if missing:
        raise ObservationError('vm_stat missing ' + ','.join(missing))
    result = {'page_size': page, 'pages': pages}
    result.update({key + '_bytes': value * page for key, value in pages.items()})
    result['free_plus_cached_bytes'] = (pages['free'] + pages['speculative'] + pages['file_backed']) * page
    result['compressor_bytes'] = result['compressor_occupied_bytes']
    return result


def parse_size(text):
    match = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)([KMGT])', text)
    if match is None:
        raise ObservationError('unrecognized size: ' + text)
    return round(float(match.group(1)) * UNITS[match.group(2)])


def parse_swapusage(text):
    found = dict(re.findall(r'\b(total|used|free) = ([0-9.]+[KMGT])', text))
    if set(found) != {'total', 'used', 'free'}:
        raise ObservationError('vm.swapusage incomplete: ' + text.strip()[:120])
    return {key + '_bytes': parse_size(value) for key, value in found.items()}


def parse_sysctl(text, names):
    values = {}
    for line in text.splitlines():
        name, sep, value = line.partition(':')
        if sep and name.strip() in names:
            values[name.strip()] = value.strip()
    missing = [name for name in names if name not in values]
    if missing:
        raise ObservationError('sysctl missing ' + ','.join(missing))
    return values


def parse_footprint(text, pid):
    """`footprint -p PID -f bytes --noCategories`: the target's current phys_footprint."""
    if re.search(r'\[' + str(pid) + r'\]', text) is None:
        raise ObservationError('footprint output names a different process')
    values = re.findall(r'^\s*phys_footprint:\s+([0-9]+) B\s*$', text, re.MULTILINE)
    if len(values) != 1:
        raise ObservationError('footprint phys_footprint missing or ambiguous')
    return int(values[0])


def parse_ps(text):
    """`ps -axww -o pid=,ppid=,pgid=,rss=,lstart=,args=` rows; argv is whitespace-split args."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(None, 9)
        if len(parts) < 9 or not all(p.isdecimal() for p in parts[:4]):
            raise ObservationError('unrecognized ps row: ' + line[:80])
        args = parts[9] if len(parts) == 10 else ''
        rows.append({'pid': int(parts[0]), 'ppid': int(parts[1]), 'pgid': int(parts[2]),
                     'rss_bytes': int(parts[3]) * 1024, 'start': ' '.join(parts[4:9]),
                     'args': args, 'argv': args.split()})
    return rows


def run(argv, timeout=2.0):
    try:
        proc = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ObservationError(f'{argv[0]}: {error!r}') from error
    if proc.returncode != 0:
        raise ObservationError(f'{argv[0]} exit {proc.returncode}: {proc.stderr.strip()[:200]}')
    return proc.stdout


def host_memory():
    """One host observation: vm_stat, swap, pressure level and memorystatus level."""
    start = time.monotonic_ns()
    vm = parse_vm_stat(run([VM_STAT]))
    names = ('vm.swapusage', 'kern.memorystatus_vm_pressure_level', 'kern.memorystatus_level')
    raw = parse_sysctl(run([SYSCTL, *names]), names)
    level = int(raw['kern.memorystatus_vm_pressure_level'])
    return {'start_ns': start, 'end_ns': time.monotonic_ns(), 'vm': vm,
            'swap': parse_swapusage(raw['vm.swapusage']),
            'pressure_level': level, 'pressure': PRESSURE_NAMES.get(level, 'unknown'),
            'memorystatus_level': int(raw['kern.memorystatus_level'])}


def process_table():
    return parse_ps(run([PS, '-axww', '-o', PS_COLUMNS]))


def footprint(pid, timeout=1.5):
    """phys_footprint from /usr/bin/footprint; raises ObservationError when unavailable."""
    return parse_footprint(run([FOOTPRINT, '-p', str(pid), '-f', 'bytes', '--noCategories'], timeout), pid)


def thermal_state():
    """NSProcessInfo.thermalState (0 nominal .. 3 critical) or None when unavailable."""
    script = 'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState'
    try:
        raw = run([OSASCRIPT, '-l', 'JavaScript', '-e', script], timeout=3.0).strip()
    except ObservationError:
        return None
    return int(raw) if raw in ('0', '1', '2', '3') else None


def host_identity():
    names = ('hw.memsize', 'hw.pagesize', 'machdep.cpu.brand_string', 'kern.osproductversion')
    raw = parse_sysctl(run([SYSCTL, *names]), names)
    return {'physical_bytes': int(raw['hw.memsize']), 'page_size': int(raw['hw.pagesize']),
            'chip': raw['machdep.cpu.brand_string'], 'os_version': raw['kern.osproductversion']}


def listeners(port):
    """PIDs listening on TCP port (lsof exits 1 with no output when none)."""
    try:
        proc = subprocess.run([LSOF, '-nP', f'-iTCP:{port}', '-sTCP:LISTEN', '-t'], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=3.0)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ObservationError(f'lsof: {error!r}') from error
    if proc.returncode not in (0, 1):
        raise ObservationError(f'lsof exit {proc.returncode}')
    return sorted(int(x) for x in proc.stdout.split() if x.isdecimal())
