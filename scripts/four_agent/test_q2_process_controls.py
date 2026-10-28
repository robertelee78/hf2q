"""Bounded actual-owned-child cleanup controls, no compiler/model/sandbox claim.
ROOT executes only after the model window closes. Calls the production stop_group.
"""
import os
import selectors
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch
import qwen38_n1_process as production


class ActualOwnedProcessControls(unittest.TestCase):
    def spawn_owned(self):
        p = subprocess.Popen([sys.executable, "-I", "-S", "-c",
            "import signal; print('Q2_OWNED_READY', flush=True); signal.pause()"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(p.stdout, selectors.EVENT_READ)
                self.assertTrue(ready.select(timeout=2), "owned child startup timeout")
                self.assertEqual(p.stdout.readline(), b"Q2_OWNED_READY\n")
            rows = production.process_snapshot()
            self.assertIn(p.pid, rows)
            return p, {p.pid: rows[p.pid][3]}
        except BaseException:
            self.emergency(p)
            raise

    def emergency(self, p):
        # Safety cleanup ONLY our Popen-created group, even after test assertion fails.
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        p.wait(timeout=3)
        p.stdout.close(); p.stderr.close()

    def test_marker_then_hang_production_stop_reaps_owned_group(self):
        p, known = self.spawn_owned()
        try:
            result = production.stop_group(p, known)
            self.assertTrue(result["reaped"])
            self.assertTrue(result["verified"])
            self.assertEqual(result["final_observation"]["live_owned_pids"], [])
        finally:
            self.emergency(p)

    def test_unavailable_metrics_still_kills_and_reaps_but_cannot_qualify(self):
        p, known = self.spawn_owned()
        try:
            with patch.object(production, "process_snapshot", side_effect=RuntimeError("Q2 deliberate ps failure")):
                result = production.stop_group(p, known)
            self.assertTrue(result["reaped"])
            self.assertFalse(result["verified"])
            self.assertFalse(result["final_observation"]["available"])
            with self.assertRaises(ProcessLookupError):
                os.killpg(p.pid, 0)
        finally:
            self.emergency(p)


if __name__ == "__main__": unittest.main()
