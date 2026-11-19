"""Real kernel RSS/lineage observations; no model, timing claims or subprocess fallback."""
import os
import subprocess
import sys
import unittest

import qwen38_n1_process as process


@unittest.skipUnless(sys.platform == "darwin", "Darwin native process metrics")
class NativeToolMetrics(unittest.TestCase):
    def test_live_allocation_lineage_and_lifetime_identity(self):
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import sys; data=bytearray(8<<20); data[::4096]=b'x'*2048; "
             "print('ready',flush=True); sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True)
        try:
            self.assertEqual(child.stdout.readline(), b"ready\n")
            first = process.process_snapshot()
            self.assertIn(child.pid, first)
            ppid, pgid, rss, birth = first[child.pid]
            self.assertEqual(ppid, os.getpid())
            self.assertEqual(pgid, child.pid)
            self.assertGreaterEqual(rss, 8 << 20)
            self.assertLess(rss, 512 << 20, "resident bytes must not be virtual size or another field")
            self.assertIsInstance(birth, str)
            self.assertTrue(birth)
            self.assertEqual(process.tree(first, child.pid, {}), {child.pid})
            second = process.process_snapshot()
            self.assertEqual(second[child.pid][3], birth)
            child.communicate(b"x", timeout=2)
            self.assertEqual(child.returncode, 0)
            final = process.process_snapshot()
            self.assertTrue(child.pid not in final or final[child.pid][3] != birth,
                            "an exited process must not retain its live identity")
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
            child.stdin.close()
            child.stdout.close()

    @unittest.skipIf(os.geteuid() == 0, "root can inspect foreign tasks")
    def test_protected_discovery_is_retained_but_not_faked_as_zero(self):
        rows = process.process_snapshot()
        self.assertIn(1, rows)
        self.assertIsNone(rows[1][2])
        self.assertIsNone(rows[1][3])
        self.assertNotIn(1, process.tree(rows, os.getpid(), {}))


if __name__ == "__main__":
    unittest.main()
