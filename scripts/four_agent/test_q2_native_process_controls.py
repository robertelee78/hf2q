"""Pure ownership negatives plus one explicit bounded actual detached-child case.
Root runs after source review; no native client/model/compiler/device involved.
"""
import json,os,selectors,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from q2_native_process import Ownership,Inventory,stop_owned
from qwen38_n1_verdict import Invalid


def row(parent, group, stamp, rss=1): return parent, group, rss, stamp


class NativeProcessOwnershipControls(unittest.TestCase):
    def test_register_fresh_descendants_before_accepting_their_detached_group(self):
        baseline = {9: row(1, 9, "old")}; owner = Ownership(baseline, 100)
        rows = {**baseline, 100: row(1, 100, "root"), 101: row(100, 101, "detached"), 102: row(101, 101, "grandchild")}
        self.assertEqual(owner.observe(rows), {100, 101, 102})
        self.assertEqual(set(owner.groups), {100, 101})
        self.assertEqual(owner.safe_targets(rows)[0], [101, 100])
    def test_baseline_pid_group_join_unowned_leader_and_reuse_refuse(self):
        baseline = {9: row(1, 9, "old")}
        for child in (row(100, 9, "new"), row(100, 777, "new")):
            owner = Ownership(baseline, 100)
            with self.assertRaises(Invalid): owner.observe({**baseline,100:row(1,100,"root"),101:child})
            self.assertIn(101, owner.known) # Registered identity remains safe-cleanup evidence.
            self.assertNotIn(9, owner.groups); self.assertNotIn(777, owner.groups)
        owner = Ownership({}, 100); owner.observe({100:row(1,100,"root"),101:row(100,101,"child")})
        with self.assertRaises(Invalid): owner.observe({100:row(1,100,"root"),101:row(1,101,"reused")})
        self.assertNotIn(101, owner.safe_targets({101:row(1,101,"reused")})[0])
    def test_unowned_group_member_never_authorizes_signaling_its_group(self):
        owner = Ownership({},100); rows={100:row(1,100,"root"),101:row(100,101,"child"),777:row(1,101,"unowned")}
        with self.assertRaises(Invalid): owner.observe(rows)
        groups,pids=owner.safe_targets(rows);self.assertNotIn(101,groups);self.assertIn(101,pids);self.assertNotIn(777,pids)
    def test_metrics_failure_signals_live_root_not_unverified_detached_group_and_cannot_qualify(self):
        owner=Ownership({},100);owner.observe({100:row(1,100,"root"),101:row(100,101,"child")})
        class Process:
            pid=100
            done=False
            def poll(self): return 0 if self.done else None
            def wait(self,timeout): self.done=True; return 0
        class Broken:
            def query(self): raise RuntimeError("MOCK ps failure")
        with patch('q2_native_process.os.killpg') as kill, patch('q2_native_process.time.monotonic',side_effect=range(100)):
            result=stop_owned(Process(),owner,Broken())
        self.assertFalse(result['verified']);self.assertTrue(result['reaped'])
        self.assertTrue(kill.call_args_list)
        self.assertTrue(all(call.args[0]==100 for call in kill.call_args_list))
        self.assertEqual(owner.safe_targets(None, root_running=False)[0], [])
    def test_actual_detached_child_registered_then_all_groups_reaped_empty(self):
        # Parent stays alive until the supervisor registers the detached child;
        # no orphan/name-based inference is needed. Emergency cleanup targets
        # only this test's Popen group and the child PID received from its pipe.
        with tempfile.TemporaryDirectory(prefix='q2-native-owned-') as d:
            inventory=Inventory(Path(d));baseline=inventory.query()
            child_code="import signal; signal.pause()"
            parent_code=("import subprocess,sys,signal; p=subprocess.Popen([sys.executable,'-I','-S','-c',"+repr(child_code)+"],start_new_session=True); signal.signal(signal.SIGTERM,lambda *_:(p.wait(timeout=2),sys.exit(0))); print('CHILD_PID='+str(p.pid),flush=True); signal.pause()")
            process=subprocess.Popen([sys.executable,'-I','-S','-c',parent_code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
            detached=None;owner=None
            try:
                with selectors.DefaultSelector() as ready:
                    ready.register(process.stdout,selectors.EVENT_READ)
                    require_line=ready.select(timeout=2);self.assertTrue(require_line)
                    line=process.stdout.readline().decode().strip();self.assertTrue(line.startswith('CHILD_PID='));detached=int(line.split('=')[1])
                owner=Ownership(baseline,process.pid)
                deadline=time.monotonic()+2
                while time.monotonic()<deadline:
                    owner.observe(inventory.query())
                    if detached in owner.known:break
                    time.sleep(.02)
                self.assertIn(detached,owner.known);self.assertIn(detached,owner.groups)
                result=stop_owned(process,owner,inventory)
                self.assertTrue(result['reaped']);self.assertTrue(result['verified'],result)
                self.assertTrue(result['final_observation']['all_registered_groups_empty'])
            finally:
                for pg in (detached,process.pid):
                    if pg is not None:
                        try:os.killpg(pg,signal.SIGKILL)
                        except ProcessLookupError:pass
                process.wait(timeout=3);process.stdout.close();process.stderr.close()


if __name__=='__main__':unittest.main()
