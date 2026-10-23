"""Capture isolation and cleanup at real subprocess/HTTP boundaries (no model)."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from capture_client import capture, live_group_members, parse_config, SUPPORTED_PACKAGE


WIRE_CLIENT = '''
profile=json.loads((Path(os.environ['XDG_CONFIG_HOME'])/'opencode/opencode.json').read_text())
assert set(profile)=={'model','providers'}
assert set(profile['providers'])=={'hf2q'}
settings=profile['providers']['hf2q']['settings']
assert settings['baseURL'].startswith('http://127.0.0.1:')
assert settings['apiKey']=='synthetic-local-capture'
assert 'HF2Q_TEST_SECRET' not in os.environ
state=Path(os.environ['XDG_STATE_HOME'])
state.joinpath('cwd.txt').write_text(os.getcwd())
assert Path.cwd()!=state.parent
assert not any((p/'opencode.json').exists() for p in (Path.cwd(),*Path.cwd().parents))
payload={'model':'fixture','stream':False,'max_completion_tokens':37,'tools':[{'type':'function'}],
         'messages':[{'role':'user','content':'synthetic capture'}], 'min_p':0.02,
         'reasoning':{'effort':'low'},'reasoning_effort':'low','tool_choice':'auto','parallel_tool_calls':False}
request=urllib.request.Request(settings['baseURL']+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=3) as response:
 assert json.load(response)['choices'][0]['message']['content']=='CAPTURED'
'''


class CaptureBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {"model": "hf2q/fixture", "plugins": ["must-not-load"], "providers": {
            "hf2q": {"name": "fixture provider", "package": SUPPORTED_PACKAGE,
                     "settings": {"baseURL": "http://must-not-contact.invalid", "apiKey": "source-secret"},
                     "models": {"fixture": {"modelID": "fixture", "name": "fixture model",
                         "capabilities": {"tools": True, "input": ["text", "image"], "output": ["text"]},
                         "compatibility": {"reasoningField": "reasoning_content"},
                         "cost": {"input": 0, "output": 0}, "limit": {"context": 262144, "output": 8192}}}},
            "unrelated": {"settings": {"apiKey": "other-secret"}}}}

    def client(self, body, name="client"):
        script = self.root / name
        script.write_text("#!" + sys.executable + "\n" + '''
import json,os,sys,time,urllib.request,subprocess,signal,concurrent.futures
from pathlib import Path
if '--version' in sys.argv:
 print('synthetic-boundary-client');sys.exit(0)
state=Path(os.environ['XDG_STATE_HOME'])
state.joinpath('client.pid').write_text(str(os.getpid()))
''' + body)
        script.chmod(0o700)
        return str(script)

    def test_only_allowlisted_profile_reaches_child_and_stub(self):
        client = self.client(WIRE_CLIENT)
        original = copy.deepcopy(self.source)
        # An output ancestor config is intentionally present: it must not be cwd.
        (self.root / "opencode.json").write_text('{"plugins":["must-not-load"]}')
        output = self.root / "capture"
        source_bytes = json.dumps(self.source, indent=2).encode()
        with patch.dict(os.environ, {"HF2Q_TEST_SECRET": "must-not-leak"}):
            result = capture(client, self.source, output, timeout=5, source_bytes=source_bytes)
        self.assertEqual(self.source, original)
        self.assertFalse(result["model_inference_performed"])
        request = result["requests"][0]
        self.assertEqual(request["kind"], "agent")
        self.assertEqual(request["wire_settings"]["max_completion_tokens"], 37)
        self.assertEqual(request["wire_settings"]["reasoning"], {"effort": "low"})
        self.assertEqual(request["wire_settings"]["min_p"], 0.02)
        self.assertEqual(request["wire_settings"]["tool_choice"], "auto")
        self.assertIs(request["wire_settings"]["parallel_tool_calls"], False)
        self.assertEqual(request["tool_count"], 1)
        self.assertEqual(result["source_config_file_sha256"], hashlib.sha256(source_bytes).hexdigest())
        self.assertNotEqual(result["source_config_file_sha256"], result["source_config_semantic_sha256"])
        self.assertIn("not an outbound-network sandbox", result["network_isolation"])
        profile_file = output / "config/opencode/opencode.json"
        self.assertNotIn("source-secret", profile_file.read_text())
        self.assertEqual(profile_file.stat().st_mode & 0o777, 0o600)
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        self.assertFalse(Path((output / "state/cwd.txt").read_text()).exists(), "scratch cwd must be removed")
        with self.assertRaises(FileExistsError):
            capture(client, self.source, output)

    def test_unknown_provider_model_and_nested_overrides_fail_before_launch(self):
        mutations = [
            lambda p, m: p.update({"headers": {"Authorization": "secret"}}),
            lambda p, m: p["settings"].update({"headers": {"Authorization": "secret"}}),
            lambda p, m: p["settings"].update({"temperature": 0.5}),
            lambda p, m: p.update({"package": "untrusted-adapter"}),
            lambda p, m: m.update({"options": {"baseURL": "https://must-not-contact.invalid"}}),
            lambda p, m: m.update({"settings": {"apiKey": "secret"}}),
            lambda p, m: m["compatibility"].update({"headers": {"Authorization": "secret"}}),
            lambda p, m: m["capabilities"].update({"baseURL": "https://must-not-contact.invalid"}),
            lambda p, m: m["limit"].update({"headers": {"Authorization": "secret"}}),
            lambda p, m: m["cost"].update({"options": {"apiKey": "secret"}}),
            lambda p, m: m["capabilities"].update({"tools": {"apiKey": "secret"}}),
            lambda p, m: m["limit"].update({"output": True}),
            lambda p, m: m.update({"modelID": {"apiKey": "secret"}}),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                source = copy.deepcopy(self.source)
                provider = source["providers"]["hf2q"]
                mutate(provider, provider["models"]["fixture"])
                output = self.root / f"reject-{index}"
                with self.assertRaises(ValueError):
                    capture("must-not-execute", source, output)
                self.assertFalse(output.exists())

    def test_cli_does_not_forward_driver_stdin_to_client(self):
        client = self.client("state.joinpath('stdin.txt').write_text(sys.stdin.read())\n" + WIRE_CLIENT)
        config = self.root / "source.json"
        config.write_text(json.dumps(self.source))
        output = self.root / "closed-stdin"
        subprocess.run([sys.executable, str(Path(__file__).with_name("capture_client.py")),
                        "--client", str(client), "--config", str(config), "--output", str(output)],
                       input="driver context must not enter the captured prompt", text=True,
                       # The outer CLI includes version probing (itself bounded at 15s),
                       # two interpreter starts and owned-process/server cleanup. Allow
                       # hosted runners time for those stages; stdin isolation is
                       # asserted from the child's actual read, not elapsed time.
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True)
        self.assertEqual((output / "state/stdin.txt").read_text(), "")

    def test_programmatic_config_has_no_fabricated_raw_file_digest(self):
        result = capture(self.client(WIRE_CLIENT), self.source, self.root / "programmatic", timeout=5)
        self.assertIsNone(result["source_config_file_sha256"])
        self.assertEqual(len(result["source_config_semantic_sha256"]), 64)

    def test_changed_raw_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source bytes"):
            capture("must-not-execute", self.source, self.root / "bad-bytes", source_bytes=b'{}')
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_config(b'{"model":"first","model":"second"}')
        # Python equality conflates distinct JSON number/bool representations.
        for index, changed in enumerate(("number", "boolean")):
            source = copy.deepcopy(self.source)
            raw = copy.deepcopy(source)
            if changed == "number":
                raw["providers"]["hf2q"]["models"]["fixture"]["cost"]["input"] = 0.0
            else:
                source["plugins"], raw["plugins"] = [1], [True]
            self.assertEqual(source, raw)
            output = self.root / f"different-json-{index}"
            with self.assertRaisesRegex(ValueError, "source bytes"):
                capture("must-not-execute", source, output,
                        source_bytes=json.dumps(raw).encode())
            self.assertFalse(output.exists())

    def test_output_inside_any_checkout_is_rejected(self):
        for marker_directory in (True, False):
            root = self.root / ("repo-dir" if marker_directory else "repo-worktree")
            root.mkdir()
            marker = root / ".git"
            if marker_directory:
                marker.mkdir()
            else:
                marker.write_text("gitdir: external\n")
            with self.assertRaisesRegex(ValueError, "all Git checkouts"):
                capture("must-not-execute", self.source, root / "nested/capture")
            self.assertFalse((root / "nested").exists())

    def test_timed_out_parent_is_reaped_without_success_receipt(self):
        output = self.root / "timeout"
        with self.assertRaises(subprocess.TimeoutExpired):
            capture(self.client("time.sleep(60)\n"), self.source, output, timeout=0.5, cleanup_grace=0.1)
        pid = int((output / "state/client.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertFalse((output / "summary.json").exists())

    def test_stubborn_descendant_is_killed_after_parent_exits(self):
        body = WIRE_CLIENT + '''
child=subprocess.Popen([sys.executable,'-c',"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)"])
state.joinpath('descendant.pid').write_text(str(child.pid))
time.sleep(0.15)
'''
        output = self.root / "descendant"
        sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
        try:
            result = capture(self.client(body), self.source, output, timeout=5, cleanup_grace=0.1)
            self.assertIsNone(sentinel.poll(), "cleanup must not touch an unrelated process group")
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=5)
        parent = int((output / "state/client.pid").read_text())
        self.assertTrue(result["cleanup_escalated"])
        self.assertTrue(result["owned_process_group_terminated"])
        self.assertEqual(live_group_members(parent), [])

    def test_concurrent_request_log_and_title_classification(self):
        body = WIRE_CLIENT + '''
def send(i):
 p={'model':'fixture','stream':False,'messages':[{'role':'system','content':'You are a title generator.'},{'role':'user','content':str(i)}]}
 req=urllib.request.Request(settings['baseURL']+'/chat/completions',data=json.dumps(p).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=5) as response: response.read()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(send,range(8)))
'''
        output = self.root / "concurrent"
        result = capture(self.client(body), self.source, output, timeout=10)
        recorded = json.loads((output / "requests.json").read_text())
        self.assertEqual(len(recorded), 9)
        title_inputs = {row["request"]["messages"][-1]["content"] for row in recorded
                        if row["request"]["messages"][0].get("role") == "system"}
        self.assertEqual(title_inputs, {str(i) for i in range(8)})
        self.assertEqual(len(result["requests"]), 9)
        self.assertEqual(sum(row["kind"] == "title" for row in result["requests"]), 8)
        self.assertEqual(result["requests_sha256"], hashlib.sha256((output / "requests.json").read_bytes()).hexdigest())

    def test_scratch_with_ancestor_project_configuration_is_refused(self):
        (self.root / "opencode.json").write_text('{"plugins":["must-not-load"]}')
        temporary_directory = tempfile.TemporaryDirectory
        def under_project(**kwargs):
            return temporary_directory(dir=self.root, **kwargs)
        with patch("capture_client.tempfile.TemporaryDirectory", side_effect=under_project), \
                patch("capture_client.subprocess.Popen") as launch:
            with self.assertRaisesRegex(ValueError, "ancestor project configuration"):
                capture("must-not-execute", self.source, self.root / "ancestor-fail")
        launch.assert_not_called()
        self.assertFalse((self.root / "ancestor-fail/summary.json").exists())

    def test_thread_start_failure_closes_stub_listener_without_shutdown_deadlock(self):
        bound = []
        from http.server import ThreadingHTTPServer
        original = ThreadingHTTPServer.__init__
        def remember(server, *args, **kwargs):
            original(server, *args, **kwargs)
            bound.append(server)
        with patch.object(ThreadingHTTPServer, "__init__", remember), \
                patch("capture_client.threading.Thread.start", side_effect=RuntimeError("synthetic thread start failure")):
            with self.assertRaisesRegex(RuntimeError, "thread start failure"):
                capture("must-not-execute", self.source, self.root / "thread-fail")
        self.assertEqual(bound[0].fileno(), -1)

    def test_setup_failure_closes_stub_listener(self):
        bound = []
        from http.server import ThreadingHTTPServer
        original = ThreadingHTTPServer.__init__
        def remember(server, *args, **kwargs):
            original(server, *args, **kwargs)
            bound.append(server)
        with patch.object(ThreadingHTTPServer, "__init__", remember), \
                patch("capture_client.write_private", side_effect=OSError("synthetic setup failure")):
            with self.assertRaisesRegex(OSError, "setup failure"):
                capture("must-not-execute", self.source, self.root / "setup-fail")
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0].fileno(), -1)


if __name__ == "__main__":
    unittest.main()
