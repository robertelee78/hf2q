#!/usr/bin/env python3
"""Bounded, optional CPU observation. Import is pure; execution requires --run.

No admission, source-age, reclaimability, or whole-process bound is inferred.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import meminfo_interface
from meminfo_interface import Invalid, parse_interface

MIB = 1024 * 1024
BINARY = Path('/usr/bin/meminfo')
BINARY_SHA = 'a7bf5894154086eb8d48f4232a18053d666369132bb221b7cd1d5b2abcfa7cdf'
PHASES = (('I0', 12), ('A16', 8), ('A32', 8), ('F16', 8),
          ('Fhold', 8), ('Aoff', 8), ('I1', 12))
SPACING_NS = 250_000_000
LATENCY_NS = LATENESS_NS = 100_000_000
NOISE = 8 * MIB


def validate_bracket(sample):
    start, end = sample['mono_start_ns'], sample['mono_end_ns']
    if end < start or sample['wall_end_ns'] < sample['wall_start_ns']:
        raise Invalid('clock bracket reversed')
    offset_change = abs((sample['wall_end_ns'] - end) - (sample['wall_start_ns'] - start))
    if offset_change > 50_000_000:
        raise Invalid('wall/monotonic offset changed')
    stamp = sample['physical']['csv_wall_ns']
    if not sample['wall_start_ns'] - 100_000_000 <= stamp <= sample['wall_end_ns'] + 100_000_000:
        raise Invalid('CSV timestamp outside invocation bracket')
    return end - start <= LATENCY_NS and sample['lateness_ns'] <= LATENESS_NS


def evaluate(samples, boundaries):
    """Pure, fixed thresholds. No per-process subtraction or file-cache credit."""
    if len(samples) != 64 or [s['phase'] for s in samples] != [p for p, n in PHASES for _ in range(n)]:
        raise Invalid('incomplete or reordered schedule')
    if not all([validate_bracket(s) for s in samples]):
        return {'outcome': 'inconclusive', 'reason': 'latency_or_spacing'}
    offsets = [s['wall_start_ns'] - s['mono_start_ns'] for s in samples]
    if max(offsets) - min(offsets) > 50_000_000:
        return {'outcome': 'inconclusive', 'reason': 'clock_offset_drift'}
    grouped = {p: [s for s in samples if s['phase'] == p] for p, _ in PHASES}
    for phase in ('I0', 'Fhold', 'I1'):
        values = [s['physical']['occupied'] for s in grouped[phase]]
        drift = abs(statistics.median(values[:4]) - statistics.median(values[-4:]))
        if max(values) - min(values) > NOISE or drift > NOISE:
            return {'outcome': 'inconclusive', 'reason': 'idle_noise_' + phase}
    transitions = []
    for previous, current in (('I0', 'A16'), ('A16', 'A32')):
        baseline = statistics.median(s['physical']['occupied'] for s in grouped[previous][-4:])
        final = statistics.median(s['physical']['occupied'] for s in grouped[current][-4:])
        crossing = next((s for s in grouped[current] if s['physical']['occupied'] >= baseline + NOISE), None)
        lag = None if crossing is None else crossing['mono_start_ns'] - boundaries[current]
        passed = final - baseline >= NOISE and lag is not None and 0 <= lag <= 1_000_000_000
        transitions.append({'phase': current, 'median_delta_bytes': final - baseline,
                            'first_crossing_invocation_lag_ns': lag, 'passed': passed})
    return {'outcome': 'observed' if all(t['passed'] for t in transitions) else 'threshold_failed',
            'anonymous_transitions': transitions,
            'file_accounting_responsiveness': 'diagnostic_unresolved',
            'source_age_proved': False}


class Maps:
    """Own libc mappings only. No pointers escape after close; no payload copies.

    ABI: installed SDK sys/mman.h: mmap(void*,size_t,int,int,int,off_t),
    munmap(void*,size_t), mincore(const void*,size_t,char*). Darwin off_t is int64.
    MAP_SHARED=1, PRIVATE=2, ANON=0x1000, FAILED=(void*)-1; INCORE=1.
    """
    def __init__(self, directory):
        self.lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        self.lib.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        self.lib.mmap.restype = ctypes.c_void_p
        self.lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.munmap.restype = ctypes.c_int
        self.lib.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char)]
        self.lib.mincore.restype = ctypes.c_int
        self.page = os.sysconf('SC_PAGE_SIZE')
        if not 0 < self.page <= 65536 or 16 * MIB % self.page:
            raise Invalid('unsupported page size')
        self.directory = directory
        self.anon = self.filemap = None
        self.fd = None
        self.filepath = directory / 'payload.bin'
        self.anon_touched = 0
        self.cleanup_errors = []

    def map(self, length, prot, flags, fd):
        address = self.lib.mmap(None, length, prot, flags, fd, 0)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), 'mmap failed')
        # Zero is not dereferenced; an unexpected null result is still unmapped.
        if not address or address % self.page:
            self.lib.munmap(address, length)
            raise Invalid('unexpected mapping address')
        return address

    def unmap(self, name, length):
        address = getattr(self, name)
        if address is not None:
            if self.lib.munmap(address, length) != 0:
                raise OSError(ctypes.get_errno(), 'munmap failed: ' + name)
            setattr(self, name, None)

    def touch_anon(self, end):
        if self.anon is None:
            self.anon = self.map(32 * MIB, 1 | 2, 2 | 0x1000, -1)
        for offset in range(self.anon_touched, end, self.page):
            # One small nonconstant page, repeated; no large temporary image.
            data = self.pattern(1)
            ctypes.memmove(self.anon + offset, data, self.page)
        self.anon_touched = end

    def create_file(self):
        self.fd = os.open(self.filepath, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        for offset in range(0, 16 * MIB, self.page):
            data = self.pattern(71)
            view = memoryview(data)
            while view:
                written = os.write(self.fd, view)
                if written <= 0:
                    raise OSError('short file write')
                view = view[written:]
        self.filemap = self.map(16 * MIB, 1, 1, self.fd)
        # Explicit read materialization precedes the independent residency proof.
        self.check_bytes(self.filemap, 16 * MIB, 71)

    def pattern(self, seed):
        unit = bytes((index * 17 + seed) % 251 for index in range(251))
        return (unit * ((self.page + 250) // 251))[:self.page]

    def check_bytes(self, address, length, seed):
        expected = self.pattern(seed)
        checksum = hashlib.sha256()
        for offset in range(0, length, self.page):
            # A single page-sized copy, never a whole-map copy.
            page = ctypes.string_at(address + offset, self.page)
            if page != expected:
                raise Invalid('own-map readback mismatch')
            checksum.update(page)
        return checksum.hexdigest()

    def proof(self):
        proof = {'mono_start_ns': time.monotonic_ns(), 'maps': []}
        for address, length, seed, name in ((self.anon, self.anon_touched, 1, 'anonymous'),
                                            (self.filemap, 16 * MIB, 71, 'file')):
            if address is None:
                continue
            pages = length // self.page
            vector = (ctypes.c_char * pages)()
            if self.lib.mincore(address, length, vector) != 0:
                raise OSError(ctypes.get_errno(), 'mincore failed')
            resident = sum(bool(byte[0] & 1) for byte in vector)
            # Readback follows mincore so it cannot manufacture the preceding proof.
            if resident != pages:
                raise Invalid('own-map residency incomplete')
            checksum = self.check_bytes(address, length, seed)
            proof['maps'].append({'kind': name, 'bytes': length, 'pages': pages,
                                  'resident_pages': resident, 'checksum': checksum})
        proof['mono_end_ns'] = time.monotonic_ns()
        return proof

    def close_file(self):
        self.unmap('filemap', 16 * MIB)
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.filepath.unlink(missing_ok=True)

    def close(self):
        for operation in (lambda: self.unmap('anon', 32 * MIB), self.close_file):
            try:
                operation()
            except Exception as exc:
                self.cleanup_errors.append(str(exc))


def capacity_bytes():
    lib = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
    lib.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                ctypes.c_void_p, ctypes.c_size_t]
    lib.sysctlbyname.restype = ctypes.c_int
    value = ctypes.c_uint64()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if lib.sysctlbyname(b'hw.memsize', ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), 'hw.memsize')
    if size.value != ctypes.sizeof(value):
        raise Invalid('hw.memsize size')
    return value.value


def sample(directory, index, phase, capacity, page_bytes, scheduled_ns):
    stem = directory / ('sample-%02d' % index)
    csv_path = stem.with_suffix('.csv')
    record = {'phase': phase, 'index': index}
    command = [str(BINARY), '--summary', '--csv', str(csv_path)]
    record['argv'] = command
    record['wall_start_ns'] = time.time_ns()
    record['mono_start_ns'] = time.monotonic_ns()
    record['lateness_ns'] = max(0, record['mono_start_ns'] - scheduled_ns)
    try:
        with stem.with_suffix('.stdout').open('wb') as out, stem.with_suffix('.stderr').open('wb') as err:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise Invalid('meminfo command timeout')
    finally:
        record['mono_end_ns'] = time.monotonic_ns()
        record['wall_end_ns'] = time.time_ns()
        record['exit_code'] = proc.returncode if 'proc' in locals() else None
        stem.with_suffix('.invocation.json').write_text(json.dumps(record, indent=2) + '\n')
    if proc.returncode != 0 or stem.with_suffix('.stderr').stat().st_size:
        raise Invalid('meminfo failure or stderr')
    stdout_path = stem.with_suffix('.stdout')
    if stdout_path.stat().st_size > MIB:
        raise Invalid('oversized stdout')
    if re.search(r'\b(?:warning|error)\b', stdout_path.read_text(), re.IGNORECASE):
        raise Invalid('meminfo stdout warning or error')
    if not csv_path.exists() or csv_path.stat().st_size > MIB:
        raise Invalid('missing or oversized CSV')
    csv_bytes = csv_path.read_bytes()
    stdout_bytes = stdout_path.read_bytes()
    record['input_binding'] = {'stem': stem.name, 'csv_sha256': hashlib.sha256(csv_bytes).hexdigest(),
                               'stdout_sha256': hashlib.sha256(stdout_bytes).hexdigest()}
    stem.with_suffix('.invocation.json').write_text(json.dumps(record, indent=2) + '\n')
    record['physical'] = parse_interface(csv_bytes.decode('utf-8'), stdout_bytes.decode('utf-8'), capacity, page_bytes)
    stem.with_suffix('.invocation.json').write_text(json.dumps(record, indent=2) + '\n')
    validate_bracket(record)
    return record


def run(directory, quiet_window):
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipt = {'schema': 'hf2q-q5-meminfo-calibration-v2', 'runtime_admission': False,
               'source_age_proved': False, 'whole_memory_bound_known': False,
               'quiet_local_tool_window': quiet_window,
               'quiet_window_is_controller_attestation_not_os_quiescence': True,
               'payload_bytes_max': 48 * MIB, 'samples': [], 'proofs': [], 'boundaries': {}}
    maps = None
    start = time.monotonic_ns()
    try:
        if Path(meminfo_interface.__file__).resolve() != Path(__file__).with_name('meminfo_interface.py').resolve():
            raise Invalid('unexpected interface module path')
        if sys.platform != 'darwin' or hashlib.sha256(BINARY.read_bytes()).hexdigest() != BINARY_SHA:
            raise Invalid('platform or binary provenance mismatch')
        receipt['binary_sha256'] = BINARY_SHA
        receipt['host'] = dict(zip(('system', 'node', 'release', 'version', 'machine'), os.uname()))
        receipt['capacity'] = capacity_bytes()
        maps = Maps(directory)
        receipt['page_bytes'] = maps.page
        for phase, count in PHASES:
            if phase == 'A16':
                maps.touch_anon(16 * MIB)
            elif phase == 'A32':
                maps.touch_anon(32 * MIB)
            elif phase == 'F16':
                maps.create_file()
            elif phase == 'Aoff':
                maps.unmap('anon', 32 * MIB)
            elif phase == 'I1':
                maps.close_file()
            receipt['boundaries'][phase] = time.monotonic_ns()
            proof = maps.proof()
            receipt['proofs'].append({'phase': phase, 'when': 'before', **proof})
            scheduled = time.monotonic_ns()
            for _ in range(count):
                if time.monotonic_ns() - start > 60_000_000_000:
                    raise Invalid('whole run deadline')
                delay = scheduled - time.monotonic_ns()
                if delay > 0:
                    time.sleep(delay / 1e9)
                receipt['samples'].append(sample(directory, len(receipt['samples']), phase,
                                                 receipt['capacity'], maps.page, scheduled))
                scheduled += SPACING_NS
            receipt['proofs'].append({'phase': phase, 'when': 'after', **maps.proof()})
        if time.monotonic_ns() - start > 60_000_000_000:
            raise Invalid('whole run deadline')
        receipt.update(evaluate(receipt['samples'], receipt['boundaries']))
    except Exception as exc:
        receipt.update(outcome='inconclusive', reason=type(exc).__name__ + ': ' + str(exc))
    finally:
        if maps is not None:
            maps.close()
            receipt['cleanup_errors'] = maps.cleanup_errors
            if maps.cleanup_errors:
                receipt.update(outcome='inconclusive', reason='cleanup_failed')
        receipt['elapsed_ns'] = time.monotonic_ns() - start
        if sys.platform == 'darwin':
            receipt['binary_sha256_after'] = hashlib.sha256(BINARY.read_bytes()).hexdigest()
            if receipt['binary_sha256_after'] != BINARY_SHA:
                receipt.update(outcome='inconclusive', reason='binary_changed')
        receipt['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        receipt['interface_source_sha256'] = hashlib.sha256(Path(meminfo_interface.__file__).read_bytes()).hexdigest()
        receipt['artifact_hashes'] = {}
        for path in directory.iterdir():
            if path.is_file() and path.name not in ('receipt.json', 'payload.bin'):
                with path.open('rb') as source:
                    receipt['artifact_hashes'][path.name] = hashlib.file_digest(source, 'sha256').hexdigest()
        (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0 if receipt.get('outcome') == 'observed' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True,
                        help='explicit bounded CPU allocation/observation opt-in')
    parser.add_argument('--quiet-local-tool-window', required=True,
                        help='record agreed local-tool quiet window and unexpected activity policy')
    parser.add_argument('--output', type=Path, required=True, help='new private evidence directory')
    args = parser.parse_args()
    if not args.quiet_local_tool_window.strip():
        parser.error('a nonempty quiet-local-tool-window attestation is required')
    raise SystemExit(run(args.output, args.quiet_local_tool_window))
