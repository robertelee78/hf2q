"""Synthetic parser/receipt controls only: no commands, mappings, libc or hardware."""
import unittest
from unittest.mock import patch

import meminfo_calibration as m


def fixture(**changes):
    values = dict(anonymous='10', cached_files='20', compressor='3', free='50',
                  wired='15', static_carveouts='2', total='100',
                  datetime='2026-10-24T00:00:00+00:00', uncompressed_data='9999')
    values.update(changes)
    return ','.join(values) + '\n' + ','.join(values.values()) + '\n'


def receipt():
    samples, boundaries = [], {}
    wall = 1_792_800_000_000_000_000
    mono = 10_000_000_000
    for phase, count in m.PHASES:
        boundaries[phase] = mono
        occupied = {'I0': 100, 'A16': 116, 'A32': 132, 'F16': 148,
                    'Fhold': 148, 'Aoff': 116, 'I1': 116}[phase] * m.MIB
        for _ in range(count):
            samples.append({'phase': phase, 'mono_start_ns': mono,
                            'mono_end_ns': mono + 10_000_000,
                            'wall_start_ns': wall + mono,
                            'wall_end_ns': wall + mono + 10_000_000,
                            'lateness_ns': 0,
                            'physical': {'occupied': occupied, 'csv_wall_ns': wall + mono}})
            mono += m.SPACING_NS
    return samples, boundaries


class ParserControls(unittest.TestCase):
    def test_overlap_is_ignored_not_summed(self):
        parsed = m.parse_csv(fixture(), 100)
        self.assertEqual(parsed['occupied'], 50)
        self.assertNotIn('uncompressed_data', parsed)

    def test_duplicate_missing_width_and_extra_row_rejected(self):
        examples = [fixture().replace('anonymous,', 'free,', 1),
                    fixture().replace('anonymous,', 'other,', 1),
                    fixture().replace('10,20,', '10,', 1), fixture() + '\n1,2\n']
        for text in examples:
            with self.subTest(text=text), self.assertRaises(m.Invalid):
                m.parse_csv(text, 100)

    def test_noninteger_negative_and_capacity_rejected(self):
        for value in ('-1', '1.0', '+1', ' 1', 'NaN', ''):
            with self.subTest(value=value), self.assertRaises(m.Invalid):
                m.parse_csv(fixture(anonymous=value), 100)
        with self.assertRaises(m.Invalid):
            m.parse_csv(fixture(), 101)
        with self.assertRaises(m.Invalid):
            m.parse_csv(fixture(total='101'), 101)

    def test_timestamp_requires_zone(self):
        with self.assertRaises(m.Invalid):
            m.parse_csv(fixture(datetime='2026-10-24T00:00:00'), 100)


class ReceiptControls(unittest.TestCase):
    def test_fixed_schedule_and_two_transitions(self):
        samples, bounds = receipt()
        result = m.evaluate(samples, bounds)
        self.assertEqual(len(samples), 64)
        self.assertEqual(result['outcome'], 'observed')
        self.assertFalse(result['source_age_proved'])
        self.assertEqual(result['file_accounting_responsiveness'], 'diagnostic_unresolved')

    def test_zero_or_missing_filter_equivalent_never_passes(self):
        samples, bounds = receipt()
        for invalid in ([], samples[:-1], list(reversed(samples))):
            with self.assertRaises(m.Invalid):
                m.evaluate(invalid, bounds)

    def test_quiet_no_response_and_late_response_fail(self):
        samples, bounds = receipt()
        for s in samples:
            if s['phase'] == 'A16':
                s['physical']['occupied'] = 100 * m.MIB
        self.assertEqual(m.evaluate(samples, bounds)['outcome'], 'threshold_failed')
        samples, bounds = receipt()
        bounds['A16'] -= 1_000_000_001
        self.assertEqual(m.evaluate(samples, bounds)['outcome'], 'threshold_failed')

    def test_idle_noise_and_latency_are_inconclusive(self):
        samples, bounds = receipt()
        samples[0]['physical']['occupied'] += 9 * m.MIB
        self.assertEqual(m.evaluate(samples, bounds)['reason'], 'idle_noise_I0')
        samples, bounds = receipt()
        samples[0]['lateness_ns'] = m.LATENESS_NS + 1
        self.assertEqual(m.evaluate(samples, bounds)['reason'], 'latency_or_spacing')
        samples, bounds = receipt()
        samples[0]['mono_end_ns'] += m.LATENCY_NS
        samples[0]['wall_end_ns'] += m.LATENCY_NS
        self.assertEqual(m.evaluate(samples, bounds)['reason'], 'latency_or_spacing')

    def test_timestamp_and_clock_faults_reject(self):
        samples, bounds = receipt()
        samples[-1]['physical']['csv_wall_ns'] += 200_000_000
        with self.assertRaises(m.Invalid):
            m.evaluate(samples, bounds)
        samples, bounds = receipt()
        for s in samples[12:]:
            for key in ('wall_start_ns', 'wall_end_ns'):
                s[key] += 60_000_000
            s['physical']['csv_wall_ns'] += 60_000_000
        self.assertEqual(m.evaluate(samples, bounds)['reason'], 'clock_offset_drift')

    def test_pure_functions_never_load_libc_or_run_child(self):
        with patch.object(m.ctypes, 'CDLL', side_effect=AssertionError('hardware access')), \
                patch.object(m.subprocess, 'Popen', side_effect=AssertionError('child execution')):
            m.parse_csv(fixture(), 100)
            samples, bounds = receipt()
            self.assertEqual(m.evaluate(samples, bounds)['outcome'], 'observed')


if __name__ == '__main__':
    unittest.main()
