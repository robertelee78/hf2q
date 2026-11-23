"""CPU telemetry/HTTP/file controls; never model qualification evidence."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

import c2_observation as c2


def text(joint=10, serial=3, capture=0, restore=0, extra=""):
    return (f"hf2q_qwen_c2_joint_lane_steps_total {joint}\n"
            f"hf2q_qwen_c2_serial_lane_steps_total {serial}\n"
            f'hf2q_qwen_joint_not_eligible_turns_total{{reason="anchor_capture"}} {capture}\n'
            f'hf2q_qwen_joint_not_eligible_turns_total{{reason="anchor_restore"}} {restore}\n' + extra)


def event(ready=4, joint=3, serial=0, route="mtp", path="joint"):
    return (f"2026-11-19T00:00:00Z DEBUG {c2.TARGET}: {c2.EVENT} "
            f"decode_ready_lanes={ready} joint_lane_steps={joint} serial_lane_steps={serial} "
            f'route="{route}" path="{path}"\n')


class Metrics(unittest.TestCase):
    def test_nonzero_lifetime_baseline_and_additional_anchor_reason_are_retained(self):
        before = c2.metrics(text(extra='hf2q_qwen_joint_not_eligible_turns_total{reason="anchor_encode"} 1\n'))
        after = c2.metrics(text(16, 5, extra='hf2q_qwen_joint_not_eligible_turns_total{reason="anchor_encode"} 3\n'))
        result = c2.eligibility(before, after, {"joint_lane_steps": 6, "serial_lane_steps": 2})
        self.assertEqual(result["rate"], 0.75)
        self.assertEqual(result["not_eligible"], {"anchor_capture": 0, "anchor_restore": 0, "anchor_encode": 2})

    def test_missing_duplicate_or_noninteger_counters_are_not_zero(self):
        for bad in (text().replace("hf2q_qwen_c2_serial_lane_steps_total 3\n", ""),
                    text() + "hf2q_qwen_c2_joint_lane_steps_total 10\n",
                    text().replace("total 10", "total NaN"),
                    text().replace('{reason="anchor_capture"} 0\n', '{reason="capture"} 0\n')):
            with self.subTest(record=bad), self.assertRaises(ValueError):
                c2.metrics(bad)

    def test_counter_reset_and_dropped_trace_event_fail_closed(self):
        before, after = c2.metrics(text()), c2.metrics(text(16, 5))
        with self.assertRaises(ValueError):
            c2.eligibility(before, after, {"joint_lane_steps": 3, "serial_lane_steps": 2})
        with self.assertRaises(ValueError):
            c2.eligibility(after, before, {"joint_lane_steps": 0, "serial_lane_steps": 0})

    def test_serial_only_is_real_zero_eligibility_not_missing_data(self):
        result = c2.eligibility(c2.metrics(text()), c2.metrics(text(10, 7)),
                                {"joint_lane_steps": 0, "serial_lane_steps": 4})
        self.assertEqual(result["rate"], 0)
        self.assertEqual(result["serial_lane_steps"], 4)

    def test_zero_denominator_and_new_unbaselined_anchor_reason_are_refused(self):
        baseline = c2.metrics(text())
        with self.assertRaises(ValueError):
            c2.eligibility(baseline, baseline, {"joint_lane_steps": 0, "serial_lane_steps": 0})
        after = c2.metrics(text(16, 5, extra='hf2q_qwen_joint_not_eligible_turns_total{reason="anchor_encode"} 1\n'))
        with self.assertRaises(ValueError):
            c2.eligibility(baseline, after, {"joint_lane_steps": 6, "serial_lane_steps": 2})


class TaggedDispatch(unittest.TestCase):
    def test_actual_mtp_lane_steps_are_not_accepted_token_or_stage_width_counts(self):
        line = event().rstrip() + " accepted=12 active_lanes=4\n"
        value = c2.trace_row(line.replace("joint_lane_steps=3", "\x1b[3mjoint_lane_steps\x1b[0m=3"))
        self.assertEqual((value["joint_lane_steps"], value["serial_lane_steps"], value["route"]), (3, 0, "mtp"))
        self.assertIsNone(c2.trace_row("Qwen DFlash2 stage timing active_lanes=4 accepted_rows=12\n"))

    def test_single_ready_lane_mixed_paths_and_duplicate_counts_are_refused(self):
        for bad in (event(ready=1, joint=1), event(joint=3, serial=1),
                    event().rstrip() + " joint_lane_steps=3\n",
                    event(path="serial"), event().replace(c2.TARGET, "other::target")):
            with self.subTest(line=bad), self.assertRaises(ValueError):
                c2.trace_row(bad)


class OwnedObservation(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.log = self.root / "server.stderr"
        self.log.write_text("earlier completed workload\n")
        self.responses = [text(), text(12, 3)]
        responses = self.responses

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/metrics" or not responses:
                    self.send_error(404)
                    return
                body = responses.pop(0).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def test_retains_exact_http_snapshots_and_only_acknowledged_cohort_trace(self):
        observer = c2.Observation(self.url, str(self.log), self.root, "run-control", 123)
        with self.log.open("a") as stream:
            stream.write(event(joint=2))
        result = observer.finish()
        record = json.loads(Path(result["path"]).read_text())
        self.assertEqual(record["status"], "Observed")
        self.assertEqual(result["eligibility"]["joint_lane_steps"], 2)
        self.assertEqual(result["eligibility"]["rate"], 1)
        self.assertEqual((self.root / "c2-trace.log").read_text(), event(joint=2))
        self.assertEqual((self.root / "c2-before.prom").read_text(), text())
        self.assertEqual((self.root / "c2-after.prom").read_text(), text(12, 3))
        self.assertEqual(record["route_lane_steps"], {"mtp:joint": 2})
        self.assertTrue(observer.log.closed)

    def test_truncated_backing_log_leaves_invalid_raw_evidence_not_a_rate(self):
        observer = c2.Observation(self.url, str(self.log), self.root, "run-control", 123)
        self.log.write_text("")
        with self.assertRaises(ValueError):
            observer.finish()
        record = json.loads((self.root / "c2-observation.json").read_text())
        self.assertEqual(record["status"], "Invalid")
        self.assertNotIn("eligibility", record)
        self.assertTrue(observer.log.closed)

    def test_coordinator_source_approval_is_required_and_bound_to_activation(self):
        path = str(self.log)
        packet = {"c2_server_log": path, "c2_observation_sources": {"c2_observation.py": "0" * 64}}
        with self.assertRaises(ValueError):
            c2.activation(packet, path, "qwen", "hf2q", None)
        with self.assertRaises(ValueError):
            c2.activation(packet, None, "qwen", "hf2q", None)


if __name__ == "__main__":
    unittest.main()
