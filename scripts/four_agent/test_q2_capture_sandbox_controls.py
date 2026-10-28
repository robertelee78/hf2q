"""Actual Seatbelt builder boundaries only, not confinement/loader proof.
Root's separate native client capture decides whether standalone bind works.
"""
import unittest
from pathlib import Path
from q2_capture_client import sandbox_profile
from qwen38_n1_verdict import Invalid


class CaptureSandboxControls(unittest.TestCase):
    def test_actual_builder_admits_only_local_tcp_for_ephemeral_standalone_helper(self):
        text = sandbox_profile(Path('/private/tmp/MOCK-capture'), Path('/private/tmp/MOCK-opencode'), 18321)
        rules = [line for line in text.splitlines() if 'network-' in line]
        self.assertEqual(rules, ['(allow network-bind (local tcp "localhost:*"))',
                                 '(allow network-inbound (local tcp "localhost:*"))',
                                 '(allow network-outbound (remote tcp "localhost:*"))'])
        self.assertIn('(deny default)', text)
        self.assertNotIn('(allow network*)', text)
        self.assertNotIn('(remote tcp)', text)
        self.assertNotIn('(local ip)', text)
        self.assertNotIn(' udp', text)
    def test_invalid_stub_port_cannot_generate_even_a_loopback_profile(self):
        for port in (True, 0, 65536, '18321', None):
            with self.subTest(port=port), self.assertRaises(Invalid):
                sandbox_profile(Path('/private/tmp/MOCK-capture'), Path('/private/tmp/MOCK-opencode'), port)
    def test_path_injection_refuses_without_widening_existing_filesystem_rules(self):
        for bad in ('/private/tmp/MOCK"path', '/private/tmp/MOCK\npath', '/private/tmp/MOCK\\path'):
            with self.assertRaises(Invalid): sandbox_profile(Path(bad), Path('/private/tmp/MOCK-opencode'), 18321)
        text = sandbox_profile(Path('/private/tmp/MOCK-capture'), Path('/private/tmp/MOCK-opencode'), 18321)
        self.assertIn('(allow file-read* (literal "/private/tmp/MOCK-opencode") (subpath "/private/tmp/MOCK-capture"))', text)
        self.assertIn('(allow file-write* (subpath "/private/tmp/MOCK-capture") (literal "/dev/null"))', text)


if __name__ == '__main__': unittest.main()
