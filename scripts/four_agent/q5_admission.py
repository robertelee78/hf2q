"""ADR-063 Q5 admission charge, mandatory growth cap and floor seeding. Pure.

Admission (ADR bullet 1) charges the measured host baseline at admission time plus
the calibrated *host-level* peak: hf2q's phys_footprint leaves out file-backed
weights (a DeepSeek-V4 receipt shows a 5.3 GiB footprint against a ~102 GiB
Free+cached drop), so the charge is the calibrated drop in Free+cached:

    charged = (physical - min Free+cached during calibration)
              - (physical - baseline Free+cached at calibration)

A run is admitted only when the current baseline Free+cached >= charged + the
stop margin. The stop margin is the run's initial effective floor
(q5_policy.effective_floor): at least 2 GiB, raised to the calibration's measured
post-readiness drop rate x (sample interval + measured stop latency). The same
rate seeds the admitted run's live floor (q5_policy.Monitor).
"""
import math

MIB = 1 << 20
GIB = 1 << 30
RECEIPT_SCHEMA = 'hf2q.q5.calibration-receipt.v1'
# Post-calibration process-growth stop: calibrated peak phys_footprint x 1.25. The
# footprint excludes file-backed weights, so this cap bounds anonymous growth
# (caches, scratch, projector) with 25% headroom for run-to-run variation; host-level
# memory stays guarded by the floor/swap/critical-pressure stops on every run.
FOOTPRINT_CAP_FACTOR = 1.25


class AdmissionError(ValueError):
    pass


def require(ok, why):
    if not ok:
        raise AdmissionError(why)


def _int(value, name):
    require(type(value) is int and value >= 0, name + ' must be a nonnegative integer')
    return value


def host_charge(receipt):
    """Host-level peak and charge recorded by a calibration receipt (works for v1 receipts)."""
    physical = _int(receipt['host']['physical_bytes'], 'host.physical_bytes')
    base = _int(receipt['baseline']['free_plus_cached_bytes'], 'baseline.free_plus_cached_bytes')
    low = _int(receipt['samples']['min_free_plus_cached_bytes'], 'samples.min_free_plus_cached_bytes')
    peak_used, base_used = physical - low, physical - base
    return {'physical_bytes': physical, 'baseline_host_used_bytes': base_used, 'peak_host_used_bytes': peak_used,
            'charged_bytes': max(0, peak_used - base_used),
            'peak_footprint_bytes': receipt['samples'].get('peak_footprint_bytes'),
            'swap_delta_bytes': receipt['samples'].get('deltas', {}).get('swap_used_bytes'),
            'compressor_delta_bytes': receipt['samples'].get('deltas', {}).get('compressor_bytes')}


def same_artifact(calibrated, current):
    """Same bytes: equal sha256 when both are bound, else equal size and mtime."""
    if calibrated.get('sha256') and current.get('sha256'):
        return calibrated['sha256'] == current['sha256']
    return (calibrated.get('bytes'), calibrated.get('mtime_ns')) == (current.get('bytes'), current.get('mtime_ns'))


def admission_from(receipt, current):
    """Validate a prior CalibrationPass for this exact family/workload/source/artifact/host.

    `current` holds family, workload, binary_sha256, artifact (identity) and physical_bytes.
    The ADR recalibrates whenever artifact, source, context or slot count changes.
    """
    require(receipt.get('schema') == RECEIPT_SCHEMA, 'admission receipt schema')
    require(receipt.get('mode') == 'calibration', 'admission receipt must come from a calibration run')
    require(receipt.get('run_kind', 'calibration') == 'calibration', 'admission receipt must be a calibration run, not an admitted run')
    require((receipt.get('verdict') or {}).get('verdict') == 'CalibrationPass', 'admission receipt is not a CalibrationPass')
    require(receipt.get('family') == current['family'], 'admission receipt family differs')
    calibrated, workload = receipt.get('workload') or {}, current['workload']
    for key in ('id', 'slots', 'context_per_slot'):
        require(calibrated.get(key) == workload.get(key), f'admission receipt workload.{key} differs')
    require((receipt.get('binary') or {}).get('sha256') == current['binary_sha256'],
            'admission receipt binary (source) differs; recalibrate')
    require(receipt.get('artifact') is not None and current.get('artifact') is not None
            and same_artifact(receipt['artifact'], current['artifact']), 'admission receipt artifact differs; recalibrate')
    charge = host_charge(receipt)
    require(charge['physical_bytes'] == current['physical_bytes'], 'admission receipt host memory size differs')
    peak = charge['peak_footprint_bytes']
    require(type(peak) is int and peak > 0, 'admission receipt has no calibrated peak footprint')
    return {**charge, 'footprint_cap_factor': FOOTPRINT_CAP_FACTOR,
            'footprint_cap_bytes': math.ceil(peak * FOOTPRINT_CAP_FACTOR),
            'seed_rate_bytes_per_s': seed_rate(receipt)}


def admit(admission, baseline_free_cached, initial_floor):
    """Admit when the baseline covers the calibrated charge plus the run's initial effective floor."""
    required = admission['charged_bytes'] + initial_floor
    return {'baseline_free_plus_cached_bytes': baseline_free_cached, 'charged_bytes': admission['charged_bytes'],
            'initial_effective_floor_bytes': initial_floor, 'required_free_plus_cached_bytes': required,
            'admitted': baseline_free_cached >= required}


def seed_rate(receipt):
    """Post-readiness drop rate measured by a calibration receipt (0 when it predates the record)."""
    record = (receipt.get('samples') or {}).get('effective_floor') or {}
    rate = record.get('max_post_readiness_drop_rate_bytes_per_s')
    return rate if type(rate) is int and rate >= 0 else 0


def growth_cap(admission, physical, floor):
    """The process-growth stop every run carries."""
    if admission is not None:
        return {'kind': 'footprint', 'bytes': admission['footprint_cap_bytes'],
                'basis': f'calibrated peak_footprint_bytes x {FOOTPRINT_CAP_FACTOR}'}
    return {'kind': 'host-floor', 'bytes': physical - floor,
            'basis': 'calibration run: host used may not exceed physical - effective Free+cached floor; '
                     'this cap is the free_cached_floor stop itself (bytes shown at the 2 GiB minimum floor; '
                     'the floor rises with the measured drop rate)'}


def negative_control_latency(receipt):
    """sample->signal->exit latency measured by an accepted negative control."""
    require(receipt.get('schema') == RECEIPT_SCHEMA and receipt.get('mode') == 'negative-control',
            'not a negative-control receipt')
    require((receipt.get('verdict') or {}).get('verdict') == 'NegativeControlPass', 'negative control did not pass')
    latency = (receipt.get('stop_latency') or {}).get('sample_to_exit_ns')
    require(type(latency) is int and latency > 0, 'negative control has no measured sample_to_exit_ns')
    return latency
