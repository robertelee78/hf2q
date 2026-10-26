"""Synthetic one-page rejected-observation classifier controls; no host queries."""
import unittest

import meminfo_interface as m
from test_meminfo_interface import CAPACITY, PAGE, csv_change, fixture


def overcount():
    csv_text, text = fixture()
    csv_text = csv_change(csv_text, anonymous=38 * PAGE, total=129 * PAGE)
    text = text.replace('Anonymous: 16 pages', 'Anonymous: 38 pages')
    text = text.replace('Unaccounted: 21 pages 0.0 MB', 'Unaccounted: -1 pages -0.0 MB')
    lines = text.splitlines()
    lines = [line[:-5] + '-0.00%' if 'Unaccounted:' in line else line for line in lines]
    return csv_text, '\n'.join(lines) + '\n'


class RetryControls(unittest.TestCase):
    def retry(self, csv_text, text, capacity=CAPACITY, page=PAGE):
        return m.retryable_single_page_overcount(csv_text, text, capacity, page)

    def test_exact_witness_retryable_but_never_admissible(self):
        c, t = overcount()
        self.assertIs(self.retry(c, t), True)
        with self.assertRaises(m.Invalid):
            m.parse_interface(c, t, CAPACITY, PAGE)
        # Whitespace padding from the summary printer does not change witness.
        self.assertTrue(self.retry(c, t.replace(': -1 pages', ':      -1 pages')))

    def test_valid_readings_are_not_retried_to_seek_better_memory(self):
        self.assertFalse(self.retry(*fixture()))
        c, t = overcount()
        c = csv_change(c, anonymous=37 * PAGE, total=128 * PAGE)
        t = t.replace('Anonymous: 38 pages', 'Anonymous: 37 pages')
        self.assertFalse(self.retry(c, t.replace('-1 pages -0.0 MB', '0 pages 0.0 MB').replace('-0.00%', '0.00%')))

    def test_larger_or_uncorroborated_overcounts_reject(self):
        c, t = overcount()
        for bad in [csv_change(c, total=130 * PAGE, anonymous=39 * PAGE),
                    csv_change(c, total=128 * PAGE, anonymous=37 * PAGE),
                    csv_change(c, total=129 * PAGE + 1, anonymous=38 * PAGE + 1),
                    csv_change(c, total=128 * PAGE), c + c.splitlines()[-1] + '\n']:
            self.assertFalse(self.retry(bad, t))
        for bad in [t.replace('-1 pages', '-2 pages'), t.replace('-0.0 MB', '-0.1 MB'),
                    t.replace('-0.00%', '-0.01%'), t.replace('128 pages', '129 pages'),
                    t.replace('Anonymous: 38 pages', 'Anonymous: 37 pages')]:
            self.assertFalse(self.retry(c, bad))

    def test_incomplete_duplicate_nested_or_foreign_root_reject(self):
        c, t = overcount()
        residual = next(line for line in t.splitlines() if 'Unaccounted' in line)
        for bad in [t.replace(residual, ''), t.replace(residual, residual + '\n' + residual),
                    t.replace('└─ Unaccounted:', '│  └─ Unaccounted:'),
                    t.replace('Unaccounted:', 'Other:'),
                    t.replace('Cached files:', 'Anonymous:'),
                    t[:t.index('Swap:')], t + '\nWarning: incomplete\n',
                    t.replace('└─ Swapped data:', '└─ Unaccounted:')]:
            self.assertFalse(self.retry(c, bad))
        self.assertFalse(self.retry(c.replace('anonymous,', 'missing,'), t))
        self.assertFalse(self.retry('"unterminated', t))

    def test_independent_capacity_page_identity_required(self):
        c, t = overcount()
        for cap, page in [(True, PAGE), (CAPACITY, True), (0, PAGE),
                          (CAPACITY + PAGE, PAGE), (CAPACITY, 0), (CAPACITY + 1, PAGE)]:
            self.assertFalse(self.retry(c, t, cap, page))
