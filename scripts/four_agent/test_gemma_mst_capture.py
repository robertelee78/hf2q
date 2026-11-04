"""Controls for the Q4-GEMMA Metal System Trace wrapper's pure parts (gemma_mst_capture.py)."""
import unittest

import gemma_mst_capture as cap


class Sanitize(unittest.TestCase):
    def test_credential_shapes_are_refused_and_redacted(self):
        samples = [b"<row>ghp_" + b"a" * 36 + b"</row>", b"<row>hf_" + b"b" * 34 + b"</row>",
                   b"<row>HF_TOKEN=abcdefgh12345</row>", b"<row>api_key: abcdefgh12345</row>",
                   b"<row>AKIA" + b"C" * 16 + b"</row>", b"sk-" + b"d" * 24]
        for data in samples:
            hits = cap.credential_hits(data)
            self.assertEqual(len(hits), 1, data)
            self.assertTrue(hits[0].endswith("...") and len(hits[0]) <= 9)

    def test_gpu_tables_pass(self):
        xml = (b'<trace-query-result><node><row><event-time>123</event-time>'
               b'<metal-command-buffer-id fmt="Command Buffer 0x1">1</metal-command-buffer-id>'
               b'<process fmt="hf2q (4242)">4242</process></row></node></trace-query-result>')
        self.assertEqual(cap.credential_hits(xml), [])

    def test_minimal_env_drops_everything_else(self):
        env = {"PATH": "/bin", "HOME": "/h", "GITHUB_TOKEN": "x", "HF_TOKEN": "y", "TMPDIR": "/t"}
        self.assertEqual(cap.minimal_env(env), {"PATH": "/bin", "HOME": "/h", "TMPDIR": "/t"})


class Argv(unittest.TestCase):
    def test_record_and_export(self):
        r = cap.record_argv("/o/raw.trace", 900)
        self.assertEqual(r[:3], ["/usr/bin/xcrun", "xctrace", "record"])
        for flag in ("--all-processes", "--no-prompt"):
            self.assertIn(flag, r)
        self.assertEqual(r[r.index("--template") + 1], "Metal System Trace")
        self.assertEqual(r[r.index("--time-limit") + 1], "900s")
        e = cap.export_argv("/o/raw.trace", "metal-gpu-execution-points")
        self.assertEqual(e[-1], '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-execution-points"]')

    def test_started_line(self):
        self.assertTrue(cap.STARTED.search("Starting recording with the Metal System Trace template. Ctrl-C to stop"))
        self.assertFalse(cap.STARTED.search("Recording failed"))


if __name__ == "__main__":
    unittest.main()
