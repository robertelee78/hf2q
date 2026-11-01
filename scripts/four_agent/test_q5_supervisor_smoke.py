"""Model-free smoke controls of the actual Q5 supervisor pieces with stub server/client processes.

Stub children are short Python scripts launched through q5_calibration.launch, exactly as the
supervisor launches its server and client (own session, scrubbed environment). Host observation
(ps, lsof, vm_stat) is real and read-only. No model, server binary or coding client runs here.
`supervise` itself is not driven: its preflight refuses while any other runtime is on the host, so
it cannot be deterministic on a shared machine.
"""
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest

import q5_calibration
import q5_host
import q5_policy as p
import q5_runtime
from test_q5_calibration import BASELINE, GIB, NC_LATENCY, policy, sample

SUCCESS = "FourAgentProtocolCompleteAwaitingIndependentReview"
IGNORE_TERM = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Supervisor(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name).resolve()
        self.children = []
        self.addCleanup(self.reap)

    def reap(self):
        live = [proc for proc in self.children if proc.poll() is None]
        if live:
            q5_runtime.terminate([proc.pid for proc in live], {}, live, 0.5, 5)

    def stub(self, code, label):
        proc = q5_calibration.launch([sys.executable, "-I", "-c", code], {}, self.out, label)
        self.children.append(proc)
        return proc

    def wait_for(self, name, timeout=10):
        path, until = self.out / name, time.monotonic() + timeout
        while not (path.exists() and path.read_text()):
            self.assertLess(time.monotonic(), until, f"stub never wrote {name}")
            time.sleep(0.01)
        return path.read_text()

    @staticmethod
    def monitor():
        monitor = p.Monitor(BASELINE, policy(), NC_LATENCY, "nc")
        for index in range(4):
            monitor.feed(sample(index, footprint=30 * GIB))
        return monitor

    def summary(self):
        return self.monitor().summary()

    def client(self, code, label):
        proc = self.stub(code, label)
        proc.wait(10)
        plan = {"client": {"summary": str(self.out / f"{label}.json"), "success_status": SUCCESS}}
        return q5_calibration.client_outcome(plan, proc, time.monotonic())

    def test_client_status_mismatch_exit_and_unreadable_summary_fail_the_client(self):
        write = "import json, pathlib; pathlib.Path('{label}.json').write_text(json.dumps({{'status': {status!r}}}))"
        passed = self.client(write.format(label="passed", status=SUCCESS), "passed")
        self.assertEqual((passed["exit_code"], passed["status_ok"], passed["summary_status"]), (0, True, SUCCESS))
        self.assertNotIn("failure", passed)
        self.assertEqual(p.classify(None, passed, self.summary(), True, policy())["verdict"], "CalibrationPass")
        mismatch = self.client(write.format(label="mismatch", status="Failed"), "mismatch")
        self.assertEqual((mismatch["status_ok"], mismatch["failure"]), (False, "client status 'Failed'"))
        exited = self.client("import sys; sys.exit(3)", "exited")
        self.assertEqual((exited["exit_code"], exited["status_ok"], exited["failure"]), (3, None, "client exit 3"))
        unreadable = self.client("pass", "unreadable")
        self.assertIs(unreadable["status_ok"], False)
        self.assertTrue(unreadable["failure"].startswith("client summary unreadable: FileNotFoundError"))
        for outcome in (mismatch, exited, unreadable):
            self.assertEqual(p.classify(None, outcome, self.summary(), True, policy())["verdict"], "ClientFailed")

    def test_readiness_reports_server_exit_timeout_and_a_tripped_latch(self):
        spec = {"url": f"http://127.0.0.1:{free_port()}/health", "field": "status", "equals": "ok"}
        exited = self.stub("import sys; sys.exit(7)", "server-exited")
        exited.wait(10)
        plan = {"readiness": {**spec, "timeout_seconds": 30}}
        latch = q5_runtime.StopLatch()
        self.assertEqual(q5_calibration.wait_ready(plan, exited, latch, {"_out": self.out}), "server_exited")
        silent = self.stub("import time; time.sleep(60)", "server-silent")
        plan = {"readiness": {**spec, "timeout_seconds": 0.3}}
        receipt = {"_out": self.out}
        self.assertEqual(q5_calibration.wait_ready(plan, silent, latch, receipt), "readiness_timeout")
        self.assertNotIn("readiness", receipt)
        latch.trip({"reason": "free_cached_floor", "detail": "synthetic"})
        self.assertIs(q5_calibration.wait_ready(plan, silent, latch, receipt), False)

    def test_stop_latch_signals_once_and_only_while_armed(self):
        server = self.stub("import time; time.sleep(60)", "server")
        latch = q5_runtime.StopLatch()
        latch.groups.append(server.pid)
        self.assertIs(latch.trip({"reason": "client_timeout", "detail": "synthetic"}), True)
        self.assertEqual(server.wait(10), -15)
        self.assertIsNotNone(latch.signal_ns)
        self.assertIs(latch.trip({"reason": "server_exited", "detail": "later"}), False)
        self.assertEqual(latch.stop["reason"], "client_timeout")
        disarmed = q5_runtime.StopLatch()
        disarmed.armed = False
        self.assertIs(disarmed.trip({"reason": "client_timeout"}), False)
        self.assertFalse(disarmed.event.is_set())

    def test_terminate_kills_a_term_ignoring_group_and_proves_it_empty(self):
        polite = self.stub("import time; time.sleep(60)", "polite")
        cleanup = q5_runtime.terminate([polite.pid], {}, [polite], 2, 5)
        self.assertEqual((cleanup["proven"], cleanup["kill_ns"], cleanup["remaining"], cleanup["exit_codes"]),
                         (True, None, [], [-15]))
        stubborn = self.stub(IGNORE_TERM + "import subprocess, sys, time; "
                             "child = subprocess.Popen([sys.executable, '-I', '-c', " + repr(IGNORE_TERM + "import time; time.sleep(60)")
                             + "]); open('stubborn.pid', 'w').write(str(child.pid)); time.sleep(60)", "stubborn")
        grandchild = int(self.wait_for("stubborn.pid"))
        cleanup = q5_runtime.terminate([stubborn.pid], {}, [stubborn], 0.5, 5)
        self.assertIs(cleanup["proven"], True)
        self.assertIsNotNone(cleanup["kill_ns"])
        self.assertEqual((cleanup["remaining"], cleanup["exit_codes"], cleanup["errors"]), ([], [-9], []))
        rows = q5_host.process_table()
        self.assertNotIn(grandchild, [row["pid"] for row in rows if row["pgid"] == stubborn.pid])

    def finish(self, port):
        receipt = {"cleanup": {"proven": True}, "host": {"physical_bytes": 128 * GIB}, "baseline": BASELINE,
                   "stop": None, "client": {"exit_code": 0, "status_ok": True}, "run_kind": "calibration"}
        q5_calibration.finish({"port": port, "policy": policy()}, receipt, self.monitor(), False)
        return receipt

    def test_listener_left_behind_means_cleanup_is_not_proven(self):
        listener = self.stub("import socket, time; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
                             "open('listener.port', 'w').write(str(s.getsockname()[1])); time.sleep(60)", "listener")
        port = int(self.wait_for("listener.port"))
        left = self.finish(port)
        self.assertEqual(left["listener_after_cleanup"], {"pids": [listener.pid], "clear": False})
        self.assertIs(left["cleanup_proven"], False)
        self.assertEqual(left["verdict"], {"verdict": "CalibrationFailed", "reason": "cleanup not proven"})
        self.assertTrue(q5_runtime.terminate([listener.pid], {}, [listener], 2, 5)["proven"])
        clear = self.finish(port)
        self.assertEqual(clear["listener_after_cleanup"], {"pids": [], "clear": True})
        self.assertIs(clear["cleanup_proven"], True)
        self.assertEqual(clear["verdict"], {"verdict": "CalibrationPass", "reason": None})


if __name__ == "__main__":
    unittest.main()
