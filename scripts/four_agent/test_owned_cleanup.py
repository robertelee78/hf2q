"""Actual cleanup seam with simulated process races; never launches a child."""
import errno
import subprocess
import unittest
from unittest.mock import Mock, patch

import qwen38_n1_process as process


class CleanupTests(unittest.TestCase):
    def cleanup(self, final=None, reap_error=None, observe_error=None, signal_error=None):
        leader = 12345
        live = {leader: (1, leader, 1, "same-start")}
        snapshots = iter([live, final if final is not None else {},
                          final if final is not None else {}])
        child = Mock(pid=leader)
        child.wait.side_effect = reap_error
        clock = iter(range(0, 1000, 10))
        def inventory():
            if observe_error:
                raise observe_error
            return next(snapshots, final if final is not None else {})
        with patch.object(process, "process_snapshot", side_effect=inventory), \
             patch.object(process.os, "killpg", side_effect=signal_error), \
             patch.object(process.time, "monotonic", side_effect=lambda: next(clock)), \
             patch.object(process.time, "sleep"):
            return process.stop_group(child, {leader: "same-start"})

    def test_eperm_race_reaped_empty_is_resolved_and_retained(self):
        result = self.cleanup(signal_error=PermissionError(errno.EPERM, "denied"))
        self.assertTrue(result["verified"])
        self.assertTrue(result["reaped"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["warnings"][0]["errno"], errno.EPERM)
        self.assertTrue(result["warnings"][0]["resolved"])
        self.assertEqual(result["final_observation"]["live_owned_pids"], [])

    def test_eperm_live_members_not_resolved(self):
        result = self.cleanup(final={12345: (1, 12345, 1, "same-start")},
                              signal_error=PermissionError(errno.EPERM, "denied"))
        self.assertFalse(result["verified"])
        self.assertFalse(result["warnings"][0]["resolved"])

    def test_unknown_inventory_fatal(self):
        self.assertFalse(self.cleanup(observe_error=RuntimeError("unknown"),
                         signal_error=PermissionError(errno.EPERM, "denied"))["verified"])

    def test_reused_identity_fatal(self):
        self.assertFalse(self.cleanup(final={12345: (1, 12345, 1, "other-start")})["verified"])

    def test_escaped_identity_fatal(self):
        self.assertFalse(self.cleanup(final={12345: (1, 999, 1, "same-start")})["verified"])

    def test_live_descendant_fatal(self):
        self.assertFalse(self.cleanup(final={12346: (12345, 12345, 1, "descendant")})["verified"])

    def test_reap_failure_fatal(self):
        self.assertFalse(self.cleanup(reap_error=subprocess.TimeoutExpired("child", 1))["verified"])

    def test_other_signal_failure_fatal(self):
        self.assertFalse(self.cleanup(signal_error=OSError(errno.EIO, "io"))["verified"])

    def test_normal_cleanup_unchanged(self):
        result = self.cleanup()
        self.assertTrue(result["verified"])
        self.assertEqual(result["warnings"], [])


if __name__ == "__main__":
    unittest.main()
