"""Real owned-process regressions; no model or passing Q5 receipt is fabricated."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import q5_host


OBSERVER = r"""
import json, os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
(root / 'observer.json').write_text(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp(),
    'term_blocked': signal.SIGTERM in mask}))
(root / 'observer-ready').touch()
until = time.monotonic() + 5
while not (root / 'release').exists():
    if time.monotonic() >= until:
        raise TimeoutError('test observer not released')
    time.sleep(.01)
print('complete-observation')
"""

LEADER = r"""
import json, os, pathlib, signal, sys, threading, time
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
import q5_host

def stop(signum, frame):
    (root / 'stop-seen').touch()
signal.signal(signal.SIGTERM, stop)
result = {}
def observe():
    before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    try:
        result['stdout'] = q5_host.run([sys.executable, '-I', '-c', sys.argv[3], str(root)], timeout=5)
    except BaseException as error:
        result['error'] = repr(error)
    result['thread_mask_restored'] = signal.pthread_sigmask(signal.SIG_BLOCK, set()) == before
worker = threading.Thread(target=observe)
worker.start()
worker.join(timeout=6)
result['worker_stopped'] = not worker.is_alive()
(root / 'result.json').write_text(json.dumps(result))
"""

INTERRUPTED_RESTORE = r"""
import json, os, signal, sys
sys.path.insert(0, sys.argv[1])
import q5_host

class StopRequested(BaseException):
    pass

def stop(signum, frame):
    raise StopRequested()

signal.signal(signal.SIGTERM, stop)
before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
original = q5_host.subprocess.Popen
children = []
result = {}
def spawn_then_signal(*args, **kwargs):
    child = original(*args, **kwargs)
    children.append(child)
    os.kill(os.getpid(), signal.SIGTERM)
    result['term_pending_before_restore'] = signal.SIGTERM in signal.sigpending()
    return child

q5_host.subprocess.Popen = spawn_then_signal
try:
    q5_host.run(['/bin/sleep', '30'], timeout=.5)
    raise AssertionError('pending signal handler did not run')
except StopRequested:
    child = children[0]
    result['killed_and_reaped_before_exception_escaped'] = child.returncode == -signal.SIGKILL
    result['pipes_closed_before_exception_escaped'] = child.stdout.closed and child.stderr.closed
    result['thread_mask_restored'] = signal.pthread_sigmask(signal.SIG_BLOCK, set()) == before
finally:
    q5_host.subprocess.Popen = original
    for child in children:
        child.kill()
        child.wait()
        child.stdout.close()
        child.stderr.close()
print(json.dumps(result))
"""


@unittest.skipUnless(hasattr(signal, 'pthread_sigmask') and hasattr(os, 'killpg'),
                     'requires POSIX signal masks and owned process groups')
class ObservationLifecycle(unittest.TestCase):
    def test_pending_signal_exception_reaps_the_spawned_observer(self):
        process = subprocess.Popen(
            [sys.executable, '-I', '-c', INTERRUPTED_RESTORE, str(Path(q5_host.__file__).parent)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 0, stderr)
            result = json.loads(stdout)
            self.assertTrue(result['term_pending_before_restore'])
            self.assertTrue(result['killed_and_reaped_before_exception_escaped'])
            self.assertTrue(result['pipes_closed_before_exception_escaped'])
            self.assertTrue(result['thread_mask_restored'])
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def wait_file(self, path, timeout=5):
        until = time.monotonic() + timeout
        while not path.exists():
            if time.monotonic() >= until:
                self.fail('controlled observation boundary not reached: ' + path.name)
            time.sleep(.01)

    def test_group_term_does_not_destroy_an_inflight_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / 'stdout').open('wb') as stdout, (root / 'stderr').open('wb') as stderr:
                process = subprocess.Popen([sys.executable, '-I', '-c', LEADER, directory,
                                            str(Path(q5_host.__file__).parent), OBSERVER],
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    self.wait_file(root / 'observer-ready')
                    observer = json.loads((root / 'observer.json').read_text())
                    self.assertEqual(observer['pgid'], process.pid)
                    os.killpg(process.pid, signal.SIGTERM)
                    self.wait_file(root / 'stop-seen')
                    (root / 'release').touch()
                    self.assertEqual(process.wait(timeout=5), 0)
                    result = json.loads((root / 'result.json').read_text())
                    self.assertNotIn('error', result)
                    self.assertTrue(observer['term_blocked'])
                    self.assertTrue(result['worker_stopped'])
                    self.assertTrue(result['thread_mask_restored'])
                    self.assertEqual(result['stdout'], 'complete-observation\n')
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()

    def test_failed_spawn_restores_the_calling_thread_mask(self):
        before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(q5_host.ObservationError):
                q5_host.run([str(Path(directory) / 'missing-observer')])
        self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), before)

    def test_timeout_reaps_the_term_protected_observer(self):
        before = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = "import os,pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
            with self.assertRaises(q5_host.ObservationError):
                q5_host.run([sys.executable, '-I', '-c', code, str(root / 'pid')], timeout=2)
            self.assertTrue((root / 'pid').is_file(), 'test observer never entered its timed region')
            pid = int((root / 'pid').read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), before)


if __name__ == '__main__':
    unittest.main()
