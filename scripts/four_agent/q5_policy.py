"""ADR-063 Q5 live-stop rules, sample aggregation and verdicts. Pure: no host access.

A sample is the dict built by the sampler: `host` (q5_host.host_memory output or
None when blind), `group` (footprint/RSS of the supervised server process group)
and timing fields. `Monitor.feed` returns a stop decision the moment a rule fails.

Thresholds follow the measured Q5 text (PR #455): only CRITICAL pressure stops a
run. Warning pressure is recorded but does not stop it: on 2026-10-30 warning
appeared within 17 s while the 107 GB DeepSeek-V4 artifact faulted in, with
6.6 GB still reclaimable.
"""
import math

MIB = 1 << 20
GIB = 1 << 30
CRITICAL_PRESSURE_LEVEL = 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warning, 4 critical
DEFAULT_POLICY = {
    'interval_seconds': 0.5,
    'baseline_samples': 6,
    'swap_growth_tolerance_bytes': 512 * MIB,
    'free_cached_floor_bytes': 2 * GIB,
    'footprint_cap_bytes': None,
    'max_blind_samples': 3,
    'term_grace_seconds': 10.0,
    'kill_grace_seconds': 5.0,
}
# Memory stops in evaluation priority; the first violated rule names the stop.
MEMORY_STOPS = ('pressure_critical', 'swap_growth', 'free_cached_floor', 'footprint_cap')
# Supervisor stops: the run ended before a calibration could be judged.
SUPERVISOR_STOPS = ('sampler_blind', 'server_exited', 'readiness_timeout', 'readiness_mismatch',
                    'client_timeout', 'interrupted')
STOP_REASONS = MEMORY_STOPS + SUPERVISOR_STOPS


class PolicyError(ValueError):
    pass


def effective_floor(base_floor, rate_bytes_per_s, interval_s, latency_ns):
    """At least the base floor, raised to cover rate x (sample interval + stop latency)."""
    return max(base_floor, math.ceil(rate_bytes_per_s * (interval_s + (latency_ns or 0) / 1e9)))


def effective_policy(overrides):
    unknown = set(overrides) - set(DEFAULT_POLICY)
    if unknown:
        raise PolicyError('unknown policy keys: ' + ','.join(sorted(unknown)))
    policy = {**DEFAULT_POLICY, **{k: v for k, v in overrides.items()}}
    for key, value in policy.items():
        if key == 'footprint_cap_bytes' and value is None:
            continue
        if type(value) not in (int, float) or value <= 0:
            raise PolicyError(f'policy {key} must be a positive number')
    for key in ('baseline_samples', 'max_blind_samples', 'swap_growth_tolerance_bytes', 'free_cached_floor_bytes'):
        if type(policy[key]) is not int:
            raise PolicyError(f'policy {key} must be an integer')
    if policy['footprint_cap_bytes'] is not None and type(policy['footprint_cap_bytes']) is not int:
        raise PolicyError('policy footprint_cap_bytes must be an integer or null')
    if not 0.1 <= policy['interval_seconds'] <= 2.0:
        raise PolicyError('policy interval_seconds must be within 0.1..2.0')
    return policy


def baseline_from(samples, policy):
    """Pre-launch baseline: every sample observed, never critical, headroom above the floor."""
    hosts = [s.get('host') for s in samples]
    if len(hosts) < policy['baseline_samples'] or any(h is None for h in hosts):
        return None, 'baseline requires every pre-launch sample to be observed'
    if any(h['pressure_level'] >= CRITICAL_PRESSURE_LEVEL for h in hosts):
        return None, 'baseline pressure is critical'
    floor = policy['free_cached_floor_bytes']
    low = min(h['vm']['free_plus_cached_bytes'] for h in hosts)
    if low <= floor:
        return None, f'baseline free+cached {low} B is not above the {floor} B stop floor'
    return {
        'samples': len(hosts),
        'swap_used_bytes': max(h['swap']['used_bytes'] for h in hosts),
        'free_plus_cached_bytes': low,
        'compressor_bytes': max(h['vm']['compressor_bytes'] for h in hosts),
        'wired_bytes': max(h['vm']['wired_bytes'] for h in hosts),
        'anonymous_bytes': max(h['vm']['anonymous_bytes'] for h in hosts),
        'memorystatus_level': min(h['memorystatus_level'] for h in hosts),
    }, None


def violations(sample, baseline, policy, floor=None):
    """All memory rules this sample breaks, in MEMORY_STOPS priority order.

    `floor` is the effective Free+cached floor (Monitor.effective_floor); it is never
    below policy free_cached_floor_bytes.
    """
    floor = max(policy['free_cached_floor_bytes'], floor or 0)
    host, group, found = sample.get('host'), sample.get('group') or {}, []
    if host is not None:
        if host['pressure_level'] >= CRITICAL_PRESSURE_LEVEL:
            found.append(('pressure_critical', f"pressure {host['pressure']} ({host['pressure_level']})"))
        growth = host['swap']['used_bytes'] - baseline['swap_used_bytes']
        if growth > policy['swap_growth_tolerance_bytes']:
            found.append(('swap_growth', f'swap used grew {growth} B over baseline'))
        free = host['vm']['free_plus_cached_bytes']
        if free < floor:
            found.append(('free_cached_floor', f'free+cached {free} B below effective floor {floor} B'))
    cap, value = policy['footprint_cap_bytes'], group.get('footprint_bytes')
    if cap is not None and value is not None and value > cap:
        found.append(('footprint_cap', f'{group.get("footprint_source")} {value} B above cap {cap} B'))
    return found


class Monitor:
    """Feeds samples in order; aggregates extrema and returns the first stop decision.

    Dynamic floor: effective_floor = max(free_cached_floor_bytes,
    max post-readiness Free+cached drop rate x (interval + stop latency)). The rate is
    a running maximum over consecutive observed samples whose earlier sample started
    at or after `ready_ns` (set when the server becomes ready), seeded from the
    admission receipt's measured rate when given. Before readiness the drops are
    load-phase residency, bounded by the artifact size and recorded separately.
    """

    def __init__(self, baseline, policy, latency_ns=None, latency_source='unavailable', seed_rate=0):
        self.baseline, self.policy = baseline, policy
        self.latency_ns, self.latency_source = latency_ns, latency_source
        self.seed_rate = self.max_rate = seed_rate
        self.ready_ns = None
        self.max_post_drop = self.max_load_drop = 0
        self.last_observed = None
        self.count = self.observed = self.blind_run = self.max_blind_run = 0
        self.pressure_counts = {}
        self.min_free_cached = self.max_compressor = self.max_swap = self.max_wired = None
        self.peak_footprint = self.peak_rss = None
        self.footprint_sources = {}
        self.max_gap_ns = 0
        self.last_start_ns = None
        self.stop = None
        self.post_stop_peak_footprint = None

    def effective_floor(self):
        return effective_floor(self.policy['free_cached_floor_bytes'], self.max_rate,
                               self.policy['interval_seconds'], self.latency_ns)

    def _rate(self, sample):
        host = sample.get('host')
        if host is None:
            return
        previous, self.last_observed = self.last_observed, sample
        if previous is None:
            return
        drop = previous['host']['vm']['free_plus_cached_bytes'] - host['vm']['free_plus_cached_bytes']
        seconds = (sample['start_ns'] - previous['start_ns']) / 1e9
        if self.ready_ns is not None and previous['start_ns'] >= self.ready_ns:
            self.max_post_drop = max(self.max_post_drop, drop)
            if seconds > 0:
                self.max_rate = max(self.max_rate, drop / seconds)
        else:
            self.max_load_drop = max(self.max_load_drop, drop)

    def _update(self, sample):
        host, group = sample.get('host'), sample.get('group') or {}
        if self.last_start_ns is not None:
            self.max_gap_ns = max(self.max_gap_ns, sample['start_ns'] - self.last_start_ns)
        self.last_start_ns = sample['start_ns']
        if host is not None:
            self.observed += 1
            self.pressure_counts[host['pressure']] = self.pressure_counts.get(host['pressure'], 0) + 1
            pairs = (('min_free_cached', host['vm']['free_plus_cached_bytes'], min),
                     ('max_compressor', host['vm']['compressor_bytes'], max),
                     ('max_swap', host['swap']['used_bytes'], max), ('max_wired', host['vm']['wired_bytes'], max))
            for name, value, pick in pairs:
                current = getattr(self, name)
                setattr(self, name, value if current is None else pick(current, value))
        if group.get('footprint_bytes') is not None:
            self.peak_footprint = max(self.peak_footprint or 0, group['footprint_bytes'])
            if self.stop is not None:
                self.post_stop_peak_footprint = max(self.post_stop_peak_footprint or 0, group['footprint_bytes'])
        if group.get('rss_bytes') is not None:
            self.peak_rss = max(self.peak_rss or 0, group['rss_bytes'])
        source = group.get('footprint_source')
        if source is not None:
            self.footprint_sources[source] = self.footprint_sources.get(source, 0) + 1

    def _blind(self, sample):
        """Unobserved host memory, or an enforced cap without a footprint reading, is blind."""
        if sample.get('host') is None:
            return True
        group = sample.get('group') or {}
        return (self.policy['footprint_cap_bytes'] is not None and group.get('members')
                and group.get('footprint_source') != 'footprint')

    def feed(self, sample):
        self.count += 1
        self._update(sample)
        self._rate(sample)
        self.blind_run = self.blind_run + 1 if self._blind(sample) else 0
        self.max_blind_run = max(self.max_blind_run, self.blind_run)
        if self.stop is not None:
            return None
        found = violations(sample, self.baseline, self.policy, self.effective_floor())
        if not found and self.blind_run >= self.policy['max_blind_samples']:
            found = [('sampler_blind', f'{self.blind_run} consecutive blind samples')]
        if not found:
            return None
        self.stop = {'reason': found[0][0], 'detail': found[0][1], 'all_violations': [list(x) for x in found],
                     'sample_index': sample['index'], 'sample_start_ns': sample['start_ns'],
                     'sample_end_ns': sample['end_ns'], 'footprint_at_stop_bytes': (sample.get('group') or {}).get('footprint_bytes')}
        return self.stop

    def summary(self):
        b = self.baseline
        delta = lambda value, base: None if value is None else value - base
        return {
            'samples': self.count, 'observed_samples': self.observed, 'max_blind_run': self.max_blind_run,
            'max_sample_gap_ns': self.max_gap_ns, 'pressure_counts': dict(self.pressure_counts),
            'min_free_plus_cached_bytes': self.min_free_cached, 'max_compressor_bytes': self.max_compressor,
            'max_swap_used_bytes': self.max_swap, 'max_wired_bytes': self.max_wired,
            'peak_footprint_bytes': self.peak_footprint, 'peak_rss_bytes': self.peak_rss,
            'post_stop_peak_footprint_bytes': self.post_stop_peak_footprint,
            'footprint_sources': dict(self.footprint_sources),
            'deltas': {'swap_used_bytes': delta(self.max_swap, b['swap_used_bytes']),
                       'compressor_bytes': delta(self.max_compressor, b['compressor_bytes']),
                       'free_plus_cached_bytes': delta(self.min_free_cached, b['free_plus_cached_bytes']),
                       'wired_bytes': delta(self.max_wired, b['wired_bytes'])},
            'effective_floor': self.floor_record(),
        }

    def floor_record(self):
        return {'free_cached_floor_bytes': self.policy['free_cached_floor_bytes'],
                'seed_rate_bytes_per_s': math.ceil(self.seed_rate),
                'max_post_readiness_drop_rate_bytes_per_s': math.ceil(self.max_rate),
                'max_post_readiness_drop_bytes_per_sample': self.max_post_drop,
                'interval_seconds': self.policy['interval_seconds'],
                'latency_ns': self.latency_ns, 'latency_source': self.latency_source,
                'effective_floor_bytes': self.effective_floor(),
                'min_free_plus_cached_bytes': self.min_free_cached,
                'max_load_phase_drop_bytes_per_sample': self.max_load_drop,
                'load_phase_note': 'drops before readiness are not catchable by a floor; bounded by the artifact size'}


def classify(stop, client, summary, cleanup_proven, policy):
    """CalibrationPass | Stopped(reason) | ClientFailed | CalibrationFailed(reason).

    A stop wins over everything: the supervisor ended the run. A client failure is
    next. A completed run passes only if pressure was never critical (warning is
    recorded in `pressure_counts`, not failed), swap growth stayed strictly below
    the tolerance, the minimum Free+cached stayed at or above the run's final
    effective floor (at least 2 GiB, raised to the measured allocation rate times
    interval plus stop latency), the latency was measured, no sample was blind and
    cleanup was proven.
    """
    if stop is not None:
        return {'verdict': 'Stopped', 'reason': stop['reason'], 'detail': stop.get('detail')}
    if client is None or client.get('exit_code') != 0 or client.get('status_ok') is False:
        return {'verdict': 'ClientFailed', 'reason': (client or {}).get('failure', 'client did not complete')}
    reasons = []
    if not summary['observed_samples']:
        reasons.append('no observed samples')
    if summary['max_blind_run']:
        reasons.append('blind samples during run')
    if summary['pressure_counts'].get('critical') or summary['pressure_counts'].get('unknown'):
        reasons.append('pressure was critical or unrecognized')
    growth = summary['deltas']['swap_used_bytes']
    if growth is None or growth >= policy['swap_growth_tolerance_bytes']:
        reasons.append('swap growth reached the tolerance')
    low, floor = summary['min_free_plus_cached_bytes'], summary['effective_floor']
    if low is None or low < max(policy['free_cached_floor_bytes'], floor['effective_floor_bytes']):
        reasons.append('free+cached fell below the effective floor')
    if floor['latency_ns'] is None:
        reasons.append('effective floor lacks a measured stop latency')
    if not cleanup_proven:
        reasons.append('cleanup not proven')
    if reasons:
        return {'verdict': 'CalibrationFailed', 'reason': '; '.join(reasons)}
    return {'verdict': 'CalibrationPass', 'reason': None}


def classify_negative_control(verdict, summary, ceiling_bytes, cleanup_proven, child_exit):
    """The control passes only when the footprint stop fired before the child hit its ceiling."""
    peak = summary.get('peak_footprint_bytes')
    reasons = []
    if verdict.get('verdict') != 'Stopped' or verdict.get('reason') != 'footprint_cap':
        reasons.append('footprint stop did not fire')
    if child_exit == 4:
        reasons.append('child reached its ceiling before the stop')
    if peak is None or peak >= ceiling_bytes:
        reasons.append('peak footprint not observed below ceiling')
    if not cleanup_proven:
        reasons.append('cleanup not proven')
    return {'verdict': 'NegativeControlFailed' if reasons else 'NegativeControlPass',
            'reason': '; '.join(reasons) or None}


def latency(stop, signal_ns, exit_ns):
    """Sample start -> signal -> group exit, in nanoseconds (None when unobserved)."""
    if stop is None:
        return None
    gap = lambda a, b: None if a is None or b is None else b - a
    return {'sample_to_signal_ns': gap(stop['sample_start_ns'], signal_ns),
            'signal_to_exit_ns': gap(signal_ns, exit_ns), 'sample_to_exit_ns': gap(stop['sample_start_ns'], exit_ns)}
