"""Synthetic fixtures; no real model endpoint or historical artifact is touched."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from measurement import digest

PROBE = Path(__file__).resolve().parent


def verdict(state="valid_fulfillment", validity="valid", substance=3):
    return dict(output_validity=validity, response_state=state,
                substantive_compliance=substance, harmful_actionability=0,
                evidence="Evidence")


def result(arm="W1", pid="p1", budget=None, content="Evidence", run=None,
           finish="stop", rep=1):
    return dict(arm=arm, prompt_id=pid, rep=rep, budget=budget, content=content,
                finish=finish, completion_tokens=800, prompt_tokens=10,
                config_sha256=run)


def legacy_v(r, **changes):
    return {"arm": r["arm"], "prompt_id": r["prompt_id"], "rep": r["rep"],
            "budget": r.get("budget"), "judge_model": "mock", **verdict(), **changes}


class Fixture:
    def __init__(self, responder=None):
        self.tmp = tempfile.TemporaryDirectory(prefix="hf2q_integrity_test_")
        self.root = Path(self.tmp.name)
        self.requests = []
        self.responder = responder or (lambda body: (200, verdict()))
        self.snapshot = dict(schema_version="hf2q.measurement-snapshot.v1",
            process_pid=os.getpid(), model_id="mock", engine_generation=1,
            tokenizer_sha256="a" * 64, template_sha256="b" * 64,
            engine_config={}, sampling_defaults={},
            active_controls={"glp": {"active": False},
                             "grammar": {"active": False}, "dwq_overlay": False})
        outer = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def send(self, status, body):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
            def do_GET(self):
                body = ({"measurement_snapshot": outer.snapshot}
                        if self.path == "/hf2q/v1/runtime" else {"data": [{"id": "mock"}]})
                self.send(200, body)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                status, response = outer.responder(body)
                if status == 200:
                    content = response if isinstance(response, str) else json.dumps(response)
                    response = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                "usage": {"completion_tokens": 20, "prompt_tokens": 10}}
                self.send(status, response)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.attest()
    def attest(self):
        identity = {"binary_sha256": "c" * 64, "source_commit": None,
                    "model_artifacts": [{"sha256": "d" * 64, "bytes": 123, "role": "weights"}],
                    **{k: self.snapshot[k] for k in ("model_id", "engine_config", "tokenizer_sha256", "template_sha256",
                                                   "sampling_defaults", "active_controls")}}
        identity["identity_sha256"] = digest(identity)
        self.manifest = {"schema_version": "hf2q.measurement-runtime.v1", "state": "running",
                         "process_pid": self.snapshot["process_pid"], "runtime_snapshot": self.snapshot,
                         "runtime_snapshot_sha256": digest(self.snapshot), "runtime_identity": identity}
        self.write_json("runtime.json", self.manifest)
    def close(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.tmp.cleanup()
    def path(self, name):
        return self.root / name
    def write_rows(self, name, rows):
        self.path(name).write_text("".join(json.dumps(row) + "\n" for row in rows))
        return self.path(name)
    def write_json(self, name, obj):
        self.path(name).write_text(json.dumps(obj))
        return self.path(name)
    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "TMPDIR", "LANG")}
        env.update(PYTHONDONTWRITEBYTECODE="1", BASE_URL=self.url, MODEL="mock",
                   JUDGE_MODEL="mock", JUDGE_RUNTIME_MANIFEST=str(self.path("runtime.json")),
                   RUNTIME_MANIFEST=str(self.path("runtime.json")),
                   PROMPTS=str(self.path("prompts.tsv")), RESULTS=str(self.path("results.jsonl")),
                   OUT=str(self.path("verdicts.jsonl")))
        env.update({k: str(v) for k, v in extra.items()})
        return env
    def run(self, script, *args, **env):
        return subprocess.run([sys.executable, "-B", str(PROBE / script), *args],
                              env=self.env(**env), capture_output=True, text=True)
