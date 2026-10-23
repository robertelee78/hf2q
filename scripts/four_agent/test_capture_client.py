"""Capture isolation and failure cleanup at an actual subprocess/HTTP boundary."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from capture_client import capture


class CaptureBoundary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {"model": "hf2q/fixture", "plugins": ["must-not-load"], "providers": {
            "hf2q": {"package": "fixture", "settings": {"baseURL": "http://must-not-contact.invalid",
                        "apiKey": "source-secret"}, "models": {"fixture": {"modelID": "fixture"}}},
            "unrelated": {"settings": {"apiKey": "other-secret"}}}}

    def client(self, body):
        script = self.root / "client"
        script.write_text("#!" + sys.executable + "\n" + '''
import json,os,sys,time,urllib.request
from pathlib import Path
if '--version' in sys.argv:
 print('synthetic-boundary-client');sys.exit(0)
Path('client.pid').write_text(str(os.getpid()))
''' + body)
        script.chmod(0o700)
        return str(script)

    def test_only_selected_profile_reaches_child_and_stub(self):
        client = self.client('''
profile=json.loads((Path(os.environ['XDG_CONFIG_HOME'])/'opencode/opencode.json').read_text())
assert set(profile)=={'model','providers'}
assert set(profile['providers'])=={'hf2q'}
settings=profile['providers']['hf2q']['settings']
assert settings['baseURL'].startswith('http://127.0.0.1:')
assert settings['apiKey']=='synthetic-local-capture'
assert 'HF2Q_TEST_SECRET' not in os.environ
payload={'model':'fixture','stream':False,'max_completion_tokens':37,'tools':[{'type':'function'}]}
request=urllib.request.Request(settings['baseURL']+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=3) as response:
 assert json.load(response)['choices'][0]['message']['content']=='CAPTURED'
''')
        original = copy.deepcopy(self.source)
        output = self.root / "capture"
        with patch.dict(os.environ, {"HF2Q_TEST_SECRET": "must-not-leak"}):
            result = capture(client, self.source, output, timeout=5)
        self.assertEqual(self.source, original)
        self.assertFalse(result["model_inference_performed"])
        self.assertEqual(result["requests"][0]["wire_settings"]["max_completion_tokens"], 37)
        self.assertEqual(result["requests"][0]["tool_count"], 1)
        self.assertNotIn("source-secret", (output / "config/opencode/opencode.json").read_text())
        with self.assertRaises(FileExistsError):
            capture(client, self.source, output)

    def test_timed_out_client_is_reaped_and_not_reported_as_capture(self):
        client = self.client("time.sleep(60)\n")
        output = self.root / "timeout"
        with self.assertRaises(subprocess.TimeoutExpired):
            capture(client, self.source, output, timeout=0.5)
        pid = int((output / "client.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertFalse((output / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
