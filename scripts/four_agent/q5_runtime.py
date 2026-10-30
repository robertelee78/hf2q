"""ADR-063 Q5 live supervision: 500 ms sampler thread, stop latch and owned process-group cleanup."""
import json
import os
import signal
import threading
import time

import q5_host


def compact(host):
    """The per-sample fields kept on disk (raw vm_stat page counts are summarized)."""
    if host is None:
        return None
    vm = host['vm']
    return {'free_plus_cached_bytes': vm['free_plus_cached_bytes'], 'compressor_bytes': vm['compressor_bytes'],
            'wired_bytes': vm['wired_bytes'], 'anonymous_bytes': vm['anonymous_bytes'],
            'file_backed_bytes': vm['file_backed_bytes'], 'free_bytes': vm['free_bytes'],
            'speculative_bytes': vm['speculative_bytes'], 'swap_used_bytes': host['swap']['used_bytes'],
            'pressure_level': host['pressure_level'], 'pressure': host['pressure'],
            'memorystatus_level': host['memorystatus_level']}


def descendants(rows, roots, known):
    """Update `known` (pid -> start) with every live process descending from a root.

    Ownership comes only from observed ancestry. A known PID whose start time changed
    was reused by an unrelated process and is dropped, never inherited.
    """
    current = {row['pid']: row for row in rows}
    for pid in [p for p, start in known.items() if p in current and current[p]['start'] != start]:
        del known[pid]
    owned = {pid for pid in roots if pid in current} | {pid for pid in known if pid in current}
    changed = True
    while changed:
        extra = {row['pid'] for row in rows if row['ppid'] in owned} - owned
        changed = bool(extra)
        owned |= extra
    for pid in owned:
        known.setdefault(pid, current[pid]['start'])
    return owned


def group_stats(rows, pgid, footprint=q5_host.footprint):
    """phys_footprint (from `footprint`) summed over the group; ps RSS stands in per failed member."""
    members = [row for row in rows if row['pgid'] == pgid]
    total, sources = 0, set()
    for row in members:
        try:
            total += footprint(row['pid'])
            sources.add('footprint')
        except q5_host.ObservationError:
            total += row['rss_bytes']
            sources.add('ps-rss-fallback')
    source = None if not members else ('footprint' if sources == {'footprint'} else 'ps-rss-fallback')
    return {'pgid': pgid, 'members': [row['pid'] for row in members],
            'footprint_bytes': total if members else None, 'footprint_source': source,
            'rss_bytes': sum(row['rss_bytes'] for row in members) if members else None}


def take_sample(index, pgid=None, roots=(), known=None):
    start = time.monotonic_ns()
    sample = {'index': index, 'start_ns': start, 'host': None, 'group': None, 'errors': []}
    try:
        sample['host'] = q5_host.host_memory()
    except q5_host.ObservationError as error:
        sample['errors'].append(repr(error))
    if pgid is not None or roots:
        try:
            rows = q5_host.process_table()
            if known is not None:
                descendants(rows, roots, known)
            if pgid is not None:
                sample['group'] = group_stats(rows, pgid)
        except q5_host.ObservationError as error:
            sample['errors'].append(repr(error))
    sample['end_ns'] = time.monotonic_ns()
    return sample


class StopLatch:
    """First stop wins; signals the supervised groups immediately from the deciding thread."""

    def __init__(self):
        self.lock, self.event = threading.Lock(), threading.Event()
        self.stop, self.signal_ns, self.groups, self.armed = None, None, [], True

    def trip(self, stop):
        with self.lock:
            if self.stop is not None or not self.armed:
                return False
            stop = dict(stop, decided_ns=time.monotonic_ns())
            self.stop = stop
            for pgid in self.groups:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            self.signal_ns = time.monotonic_ns() if self.groups else None
            self.event.set()
            return True


class Sampler(threading.Thread):
    def __init__(self, monitor, interval, path, latch):
        super().__init__(name='q5-sampler', daemon=True)
        self.monitor, self.interval, self.latch = monitor, interval, latch
        self.path, self.halt = path, threading.Event()
        self.pgid, self.roots, self.known = None, set(), {}
        self.error = None

    def run(self):
        index, target = 0, time.monotonic()
        try:
            with open(self.path, 'w', encoding='utf-8') as out:
                while not self.halt.is_set():
                    sample = take_sample(index, self.pgid, tuple(self.roots), self.known)
                    decision = self.monitor.feed(sample)
                    row = {**{k: sample[k] for k in ('index', 'start_ns', 'end_ns', 'group', 'errors')},
                           'host': compact(sample['host'])}
                    out.write(json.dumps(row, separators=(',', ':')) + '\n')
                    out.flush()
                    if decision is not None:
                        self.latch.trip(decision)
                    index += 1
                    target += self.interval
                    delay = target - time.monotonic()
                    if delay < 0:
                        target = time.monotonic()
                    elif self.halt.wait(delay):
                        break
        except BaseException as error:  # a dead sampler must never look like a quiet host
            self.error = repr(error)
            self.latch.trip({'reason': 'sampler_blind', 'detail': 'sampler thread failed: ' + self.error,
                             'sample_index': index, 'sample_start_ns': time.monotonic_ns(), 'sample_end_ns': None})


def _alive(pgids, known, popens):
    for proc in popens:
        proc.poll()
    rows = q5_host.process_table()
    members = [r for r in rows if r['pgid'] in pgids]
    strays = [r for r in rows if known.get(r['pid']) == r['start'] and r['pgid'] not in pgids]
    return rows, members, strays


def _signal(pgids, sig, errors):
    for pgid in sorted(pgids):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except OSError as error:
            errors.append(f'killpg({pgid},{sig.name}): {error!r}')


def terminate(pgids, known, popens, term_grace, kill_grace, already_signaled_ns=None):
    """TERM every owned group, wait `term_grace`, KILL survivors, then prove the groups are empty.

    Descendants that moved to their own process groups are signaled by group only
    when every member of that group is a known descendant; otherwise they are listed.
    """
    pgids, errors, refused = set(pgids), [], []
    result = {'pgids': sorted(pgids), 'term_ns': already_signaled_ns, 'kill_ns': None, 'empty_ns': None}
    try:
        rows, _, strays = _alive(pgids, known, popens)
        for pgid in {r['pgid'] for r in strays}:
            members = [r for r in rows if r['pgid'] == pgid]
            if all(known.get(r['pid']) == r['start'] for r in members):
                pgids.add(pgid)
            else:
                refused.append(pgid)
    except q5_host.ObservationError as error:
        errors.append(repr(error))
    _signal(pgids, signal.SIGTERM, errors)
    result['term_ns'] = result['term_ns'] or time.monotonic_ns()
    for sig, grace in ((None, term_grace), (signal.SIGKILL, kill_grace)):
        if sig is not None:
            _signal(pgids, sig, errors)
            result['kill_ns'] = time.monotonic_ns()
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            try:
                _, members, strays = _alive(pgids, known, popens)
            except q5_host.ObservationError as error:
                errors.append(repr(error))
                time.sleep(0.1)
                continue
            if not members and not strays and all(p.returncode is not None for p in popens):
                result['empty_ns'] = time.monotonic_ns()
                break
            time.sleep(0.05)
        if result['empty_ns'] is not None:
            break
    try:
        _, members, strays = _alive(pgids, known, popens)
        result['remaining'] = [{'pid': r['pid'], 'pgid': r['pgid'], 'args': r['args'][:200]} for r in members + strays]
    except q5_host.ObservationError as error:
        errors.append(repr(error))
        result['remaining'] = None
    result.update(pgids=sorted(pgids), refused_pgids=sorted(refused), errors=errors,
                  exit_codes=[p.returncode for p in popens],
                  proven=result['remaining'] == [] and not refused and all(p.returncode is not None for p in popens))
    return result
