"""Explicit bounded loopback integration using ACTUAL exchange, never inference.
ROOT runs only in a serialized model-free window. These fake frames prove timer/
parser rejection, not model quality, real GPU progress or independent QE.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from qwen38_n1_smoke import exchange
from qwen38_n1_verdict import Invalid
from test_qwen38_n1_verdict import request, response, chunk, stats


class FakeSocket:
    def __init__(self, frames, status=200):
        self.frames, self.events, self.status = frames, [], status
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                self.connection.settimeout(2)
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                value = json.loads(raw)
                owner.events.append(dict(kind="received", at_ns=time.monotonic_ns(), request=value))
                self.send_response(owner.status)
                self.send_header("Content-Type", "text/event-stream" if value["stream"] else "application/json")
                self.end_headers()
                try:
                    for delay, frame in owner.frames:
                        time.sleep(delay)
                        owner.events.append(dict(kind="send_begin", at_ns=time.monotonic_ns(), bytes=len(frame)))
                        self.wfile.write(frame)
                        self.wfile.flush()
                        owner.events.append(dict(kind="send_end", at_ns=time.monotonic_ns()))
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Client refused corrupt packet; no hidden server retry.
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("fake-server cleanup not verified")

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"


def frame(value):
    return ("data: " + value + "\n\n").encode()


class ActualSocketControls(unittest.TestCase):
    def run_exchange(self, server, value, output, turn=0, deadline=2):
        output.mkdir()
        return exchange(server.url, value, output, turn, set(), deadline=deadline,
                        engine_kind="hf2q", family="qwen")

    def test_known_injected_delay_not_role_or_reasoning_progress(self):
        value = request(); value["stream"] = True
        tool_name = chunk({"tool_calls": [{"index": 0, "id": "live", "type": "function",
                                           "function": {"name": "read_file"}}]})
        parts = [(0, frame(chunk({"role": "assistant"}))),
                 (.02, frame(chunk({"reasoning_content": "not semantic"}))),
                 (.10, frame(tool_name)),
                 (.02, frame(chunk({"tool_calls": [{"index": 0, "function": {
                         "arguments": '{"path":"src/ports.rs"}'}}]}))),
                 (.01, frame(chunk({}, "tool_calls", stats()))), (0, frame("[DONE]"))]
        with tempfile.TemporaryDirectory() as tmp, FakeSocket(parts) as server:
            _, call, receipt = self.run_exchange(server, value, Path(tmp).resolve()/"turn")
            self.assertEqual(call[1:], ("read_file", {"path": "src/ports.rs"}))
            self.assertGreaterEqual(receipt["semantic_ttft_seconds"], .10)
            self.assertLess(receipt["first_frame_seconds"], receipt["semantic_ttft_seconds"])
            self.assertLessEqual(receipt["semantic_ttft_seconds"], receipt["elapsed_seconds"])
            timing = receipt["semantic_timing_v3"]
            self.assertEqual(timing["schema"], "hf2q.f2.semantic-timing.v3")
            self.assertAlmostEqual(timing["semantic_ttft_seconds"],
                                   receipt["semantic_ttft_seconds"], delta=.02)
            self.assertGreaterEqual(len(server.events), 5)

    def test_name_only_time_never_certifies_missing_tool_arguments(self):
        value = request(); value["stream"] = True
        name = chunk({"tool_calls": [{"index": 0, "id": "live", "type": "function",
                                     "function": {"name": "read_file"}}]})
        with tempfile.TemporaryDirectory() as tmp, FakeSocket([
                (0, frame(name)), (0, frame(chunk({}, "tool_calls", stats()))),
                (0, frame("[DONE]"))]) as server:
            out = Path(tmp).resolve()/"turn"
            with self.assertRaises((Invalid, json.JSONDecodeError)):
                self.run_exchange(server, value, out)
            self.assertGreater((out/"response.bin").stat().st_size, 0)

    def test_role_only_and_missing_done_fail_with_raw_retention(self):
        value = request(); value["stream"] = True
        for parts in ([(0, frame(chunk({"role":"assistant"})))],
                      [(0, frame(chunk({}, "stop", stats()))) ]):
            with tempfile.TemporaryDirectory() as tmp, FakeSocket(parts) as server:
                out=Path(tmp).resolve()/"turn"
                with self.assertRaises(Invalid):self.run_exchange(server,value,out)
                self.assertTrue((out/"response.bin").exists())

    def test_fragmented_utf8_sse_uses_actual_exchange_accumulator(self):
        value=request();value["stream"]=True
        name=chunk({"tool_calls":[{"index":0,"id":"live","type":"function",
                                  "function":{"name":"read_file","arguments":'{"path":"src/ports.rs"}'}}]})
        raw=(frame(chunk({"reasoning_content":"é"}))+frame(name)+
             frame(chunk({},"tool_calls",stats()))+frame("[DONE]"))
        # Split EACH byte; this is a deliberate transport fragmentation control.
        with tempfile.TemporaryDirectory() as tmp, FakeSocket([(0,bytes([b]))for b in raw]) as server:
            message,call,_=self.run_exchange(server,value,Path(tmp).resolve()/"turn")
            self.assertEqual(message["reasoning_content"],"é")
            self.assertEqual(call[1],"read_file")

    def test_actual_absolute_deadline_preserves_failure_instead_of_fast_pass(self):
        value=request();value["stream"]=False
        with tempfile.TemporaryDirectory() as tmp, FakeSocket([(.2,json.dumps(response()).encode())]) as server:
            out=Path(tmp).resolve()/"turn";started=time.monotonic()
            with self.assertRaises((Invalid,TimeoutError,OSError)):
                self.run_exchange(server,value,out,deadline=.05)
            self.assertLess(time.monotonic()-started,2)
            self.assertTrue((out/"request.json").exists())


if __name__=="__main__":unittest.main()
