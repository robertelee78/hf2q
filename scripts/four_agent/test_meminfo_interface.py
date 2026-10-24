"""Synthetic V2 interface fixtures only; no private capture or host observation."""
import csv
import io
from pathlib import Path
import unittest

import meminfo_interface as m

FIXTURES = Path(__file__).parent / 'fixtures' / 'meminfo_v2'
PAGE = 16384
CAPACITY = 128 * PAGE


def fixture():
    return ((FIXTURES / 'residual21.csv').read_text(),
            (FIXTURES / 'residual21.stdout').read_text())


def csv_change(text, **changes):
    rows = list(csv.reader(io.StringIO(text)))
    values = dict(zip(*rows))
    values.update({k: str(v) for k, v in changes.items()})
    return ','.join(values) + '\n' + ','.join(values.values()) + '\n'


def without_residual(text):
    return '\n'.join(line.replace('├─ Static carveouts', '└─ Static carveouts')
                     for line in text.splitlines() if 'Unaccounted' not in line) + '\n'


class InterfaceControls(unittest.TestCase):
    def reject(self, csv_text, text, capacity=CAPACITY, page=PAGE):
        with self.assertRaises(m.Invalid):
            m.parse_interface(csv_text, text, capacity, page)

    def test_explicit21page_residual_is_fully_occupied(self):
        result = m.parse_interface(*fixture(), CAPACITY, PAGE)
        self.assertEqual(result['unknown_residual_bytes'], 21 * PAGE)
        self.assertEqual(result['csv_accounted_subtotal'], 107 * PAGE)
        self.assertEqual(result['physical_capacity'], 128 * PAGE)
        self.assertEqual(result['occupied'], 78 * PAGE)
        self.assertTrue(result['unknown_residual_fully_charged'])
        self.assertNotIn('total', result)  # CSV subtotal never relabeled physical capacity.
        self.assertNotIn('uncompressed_data', result)

    def test_zero_residual_absent_or_explicit_zero(self):
        c, t = fixture()
        c = csv_change(c, total=128 * PAGE, anonymous=37 * PAGE)
        t = t.replace('Anonymous: 16 pages', 'Anonymous: 37 pages')
        zero = t.replace('Unaccounted: 21 pages', 'Unaccounted: 0 pages')
        for text in (zero, without_residual(zero)):
            result = m.parse_interface(c, text, CAPACITY, PAGE)
            self.assertEqual(result['unknown_residual_bytes'], 0)

    def test_category_order_may_vary(self):
        c, t = fixture()
        lines = t.splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        self.assertEqual(m.parse_interface(c, '\n'.join(lines), CAPACITY, PAGE)['occupied'], 78 * PAGE)

    def test_missing_duplicate_mismatched_residual_rejected(self):
        c, t = fixture()
        u = next(line for line in t.splitlines() if 'Unaccounted' in line)
        for text in (without_residual(t), t.replace(u, u + '\n' + u),
                     t.replace('Unaccounted: 21 pages', 'Unaccounted: 20 pages')):
            with self.subTest(text=text):
                self.reject(c, text)

    def test_duplicate_zero_residual_rejected(self):
        c, t = fixture()
        c = csv_change(c, total=128 * PAGE, anonymous=37 * PAGE)
        t = t.replace('Anonymous: 16 pages', 'Anonymous: 37 pages')
        t = t.replace('Unaccounted: 21 pages', 'Unaccounted: 0 pages')
        u = next(line for line in t.splitlines() if 'Unaccounted' in line)
        self.reject(c, t.replace(u, u + '\n' + u))

    def test_nested_and_swap_residual_cannot_satisfy_physical_root(self):
        c, t = fixture()
        self.reject(c, t.replace('└─ Unaccounted:', '│  └─ Unaccounted:'))
        u = next(line for line in t.splitlines() if 'Unaccounted' in line)
        fake = without_residual(t).replace('└─ Swapped data:', u + '\n└─ Swapped data:')
        self.reject(c, fake)
        self.reject(c, t.replace('└─ Swapped data:', '└─ Free:'))

    def test_duplicate_missing_unknown_roots_and_header_rejected(self):
        c, t = fixture()
        for text in (t.replace('Cached files:', 'Anonymous:'),
                     '\n'.join(line for line in t.splitlines() if 'Anonymous:' not in line),
                     t.replace('Anonymous:', 'Other anonymous:'),
                     t.splitlines()[0] + '\n' + t,
                     t.replace('└─ Unaccounted:', '├─ Unaccounted:')):
            with self.subTest(text=text):
                self.reject(c, text)

    def test_csv_text_onepage_and_rounded_mb_contradictions_rejected(self):
        c, t = fixture()
        self.reject(csv_change(c, total=108 * PAGE, anonymous=17 * PAGE), t)
        # Rounded MB strings remain identical; integer pages determine agreement.
        self.reject(c, t.replace('Anonymous: 16 pages', 'Anonymous: 17 pages'))
        self.reject(c, t.replace('Physical Memory: 128 pages', 'Physical Memory: 127 pages'))

    def test_negative_or_nonpage_residual_rejected(self):
        c, t = fixture()
        self.reject(c, t, capacity=100 * PAGE)
        self.reject(c, t, capacity=CAPACITY + 1)
        self.reject(c, t, page=0)

    def test_malformed_grouping_negative_numbers_and_diagnostics_rejected(self):
        c, t = fixture()
        for number in ('1,28', '-128', '0128', '128.0', '+128'):
            with self.subTest(number=number):
                self.reject(c, t.replace('128 pages', number + ' pages', 1))
        self.reject(c, t + '\nwarning: ignored residual\n')
        self.reject(c, t.replace('21 pages', '-21 pages'))

    def test_truncation_extra_tree_and_fake_prefix_rejected(self):
        c, t = fixture()
        for text in (t.split('Swap:')[0], 'diagnostic ' + t,
                     t.replace('└─ Unaccounted:', '├─ Unaccounted:')
                     .replace('\n\nSwap:', '\n└─ Alien: 0 pages 0.0 MB  ' + '░' * 20 + ' 0.00%\n\nSwap:'),
                     t + '\nPhysical Memory: 128 pages 2.0 MB\n'):
            with self.subTest(text=text):
                self.reject(c, text)

    def test_csv_duplicate_missing_noninteger_sum_and_timezone_rejected(self):
        c, t = fixture()
        cases = [c.replace('anonymous,', 'free,', 1),
                 c.replace('anonymous,', 'unknown,', 1),
                 c + '\n1,2\n', csv_change(c, anonymous=-1),
                 csv_change(c, anonymous='1.0'), csv_change(c, total=1),
                 csv_change(c, datetime='2026-01-01T00:00:00')]
        for text in cases:
            with self.subTest(csv=text):
                self.reject(text, t)


if __name__ == '__main__':
    unittest.main()
