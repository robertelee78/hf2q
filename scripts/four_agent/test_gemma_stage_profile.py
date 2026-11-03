"""Controls for the ADR-063 Q4-GEMMA stage profile (gemma_stage_profile.py) on synthetic receipts."""
import json
from pathlib import Path
import tempfile
import unittest

import gemma_stage_profile as gsp

MS = 1_000_000


def hf2q_stream(prefill, decode, total):
    t = {"prefill_time_secs": prefill, "decode_time_secs": decode, "total_time_secs": total,
         "time_to_first_token_ms": prefill * 1e3}
    return (b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5},'
            b'"x_hf2q_timing":' + json.dumps(t).encode() + b'}\n\ndata: [DONE]\n\n')


def peer_unary(prefill, decode):
    t = {"cache_n": 4, "prompt_n": 1, "prompt_ms": prefill * 1e3, "predicted_n": 2, "predicted_ms": decode * 1e3}
    return json.dumps({"choices": [{"message": {"content": "{\"timings\": 1}"}}], "timings": t}).encode()


def write_run(root, name, turns_by_agent, start=0, makespan=None, engine="hf2q"):
    """turns_by_agent: {agent: [(send_ms, terminal_ms, prefill_s, decode_s, new, cached, events_ms or None)], ...}"""
    run = Path(root) / name
    agents = {}
    finish = 0
    for agent, turns in turns_by_agent.items():
        records = []
        for i, (send, term, pre, dec, new, cached, events) in enumerate(turns):
            d = run / "client" / agent / f"turn-{i}"
            d.mkdir(parents=True)
            raw = (hf2q_stream(pre, dec, pre + dec + 0.01) if engine == "hf2q" else peer_unary(pre, dec))
            (d / "response.bin").write_bytes(raw)
            records.append({"turn": i, "stream": events is not None, "send_monotonic_ns": start + send * MS,
                            "validated_terminal_monotonic_ns": start + term * MS,
                            "semantic_event_monotonic_ns": [start + e * MS for e in events or []],
                            "usage": {"prompt_tokens": new + cached, "completion_tokens": 3,
                                      "prompt_tokens_details": {"cached_tokens": cached}}})
        done = turns[-1][1] + 5
        finish = max(finish, done)
        agents[agent] = {"turns": records, "task_completed_monotonic_ns": start + done * MS}
    summary = {"status": gsp.SUCCESS, "makespan_started_ns": start, "agents": agents,
               "own_output_four_task_makespan_seconds": (makespan if makespan is not None else finish / 1e3)}
    (run / "client" / "summary.json").write_text(json.dumps(summary))
    return run


class TimingParse(unittest.TestCase):
    def test_hf2q_stream_last_chunk(self):
        t = gsp.server_timing(hf2q_stream(0.25, 0.5, 0.8))
        self.assertEqual((t["engine"], t["prefill_s"], t["decode_s"]), ("hf2q", 0.25, 0.5))
        self.assertAlmostEqual(t["queue_s"], 0.05)

    def test_peer_unary_ignores_content_lookalike(self):
        t = gsp.server_timing(peer_unary(0.1, 0.4))
        self.assertEqual((t["engine"], t["prefill_s"], t["decode_s"], t["queue_s"]), ("peer", 0.1, 0.4, None))

    def test_absent_or_malformed(self):
        self.assertIsNone(gsp.server_timing(b'{"choices":[]}'))
        self.assertIsNone(gsp.server_timing(b'"x_hf2q_timing":[1]'))


class RunProfile(unittest.TestCase):
    def test_stages_sum_to_agent_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, "n4-001-b01-experiment-a0", {
                "agent-0": [(10, 510, 0.2, 0.25, 100, 0, None), (600, 900, 0.1, 0.15, 20, 100, [700, 720, 800])],
                "agent-1": [(10, 410, 0.1, 0.2, 100, 0, None), (500, 1000, 0.2, 0.2, 30, 100, [800, 820])]})
            p = gsp.run_profile(run)
        a0 = p["per_agent"]["agent-0"]
        self.assertAlmostEqual(a0["prefill"], 0.3)
        self.assertAlmostEqual(a0["decode"], 0.4)
        self.assertAlmostEqual(a0["residual"], 0.8 - 0.7)
        self.assertAlmostEqual(a0["client"], 0.01 + 0.09 + 0.005)
        self.assertAlmostEqual(a0["prefill"] + a0["decode"] + a0["residual"] + a0["client"], a0["completion"])
        self.assertAlmostEqual(p["stages"]["tail"], p["makespan_s"] - p["stages"]["completion"])
        self.assertAlmostEqual(a0["queue"], 0.02)
        self.assertEqual(p["engine"], "hf2q")
        self.assertEqual(p["by_turn"]["1"]["semantic_events"], 2.5)
        self.assertAlmostEqual(p["by_turn"]["1"]["semantic_ttft_s"], 0.2)

    def test_missing_engine_timing_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, "r", {"agent-0": [(0, 10, 0.001, 0.001, 1, 0, None)]})
            (run / "client/agent-0/turn-0/response.bin").write_bytes(b"{}")
            with self.assertRaises(ValueError):
                gsp.run_profile(run)


class SseAttribution(unittest.TestCase):
    def test_worst_gap_overlapping_foreign_prefill(self):
        rows = [
            {"agent": "a", "turn": 1, "stream": True, "send_ns": 0, "terminal_ns": 1000 * MS, "decode_s": 0.9,
             "semantic_events_ns": [100 * MS, 120 * MS, 140 * MS, 160 * MS, 460 * MS, 480 * MS]},
            # b's window [200, 500) ms covers 260 of a's 300 ms gap [160, 460).
            {"agent": "b", "turn": 1, "stream": False, "send_ns": 200 * MS, "terminal_ns": 600 * MS,
             "decode_s": 0.1, "semantic_events_ns": []},
        ]
        s = gsp.sse_attribution(rows)
        self.assertEqual(s["worst_slot"], "a")
        self.assertAlmostEqual(s["worst_slot_gap_s"], 0.3)
        self.assertAlmostEqual(s["worst_slot_foreign_prefill_overlap"], 260 / 300)
        self.assertAlmostEqual(s["excess_foreign_prefill_share"], 260 / 300)


class Compare(unittest.TestCase):
    def profile(self, run, makespan, prefill):
        return {"run": run, "makespan_s": makespan, "sse": {"worst_slot_gap_s": 0.05},
                "stages": {"prefill": prefill, "decode": 1.0, "residual": 0.1, "client": 1.0,
                           "tail": makespan - prefill - 2.1, "queue": None}, "by_turn": {"0": {"span_s": 1.0}}}

    def test_block_pairing_and_share(self):
        arm = [self.profile(f"n4-{i:03}-b{i:02}-experiment-a0", 10.0 + i * 0.1 + 1.0, 3.0) for i in range(5)]
        base = [self.profile(f"n4-{i:03}-b{i:02}-peer-a0", 10.0 + i * 0.1, 2.0) for i in range(5)]
        c = gsp.compare(arm, base)
        self.assertEqual(c["pairing"], "block")
        self.assertAlmostEqual(c["stages"]["makespan"]["delta"], 1.0)
        self.assertAlmostEqual(c["stages"]["prefill"]["share_of_makespan_delta"], 1.0)
        lo, hi = c["stages"]["makespan"]["delta_ci95"]
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 1.0)
        self.assertNotIn("delta", c["stages"]["queue"])

    def test_unpaired_without_common_blocks(self):
        arm = [self.profile(f"a{i}", 11.0, 3.0) for i in range(3)]
        base = [self.profile(f"b{i}", 10.0, 2.0) for i in range(3)]
        self.assertEqual(gsp.compare(arm, base)["pairing"], "unpaired")


class WarmModel(unittest.TestCase):
    def test_recovers_known_coefficients(self):
        rows = []
        for new in (50, 120, 300, 500):
            for cached in (400, 900, 1500):
                ctx = cached + new / 2
                rows.append({"turn": 2, "new_tokens": new, "cached_tokens": cached,
                             "prefill_s": 0.05 + 1.5 * new / 1e3 + 0.2 * new * ctx / 1e6})
        rows.append({"turn": 0, "new_tokens": 465, "cached_tokens": 0, "prefill_s": 99.0})  # cold: excluded
        m = gsp.warm_prefill_model(rows)
        self.assertEqual(m["points"], 12)
        self.assertAlmostEqual(m["intercept_s"], 0.05, places=6)
        self.assertAlmostEqual(m["per_1k_new_rows_s"], 1.5, places=6)
        self.assertAlmostEqual(m["per_1m_row_ctx_s"], 0.2, places=6)
        self.assertAlmostEqual(m["r2"], 1.0, places=9)

    def test_too_few_points(self):
        self.assertIsNone(gsp.warm_prefill_model([{"turn": 1, "new_tokens": 5, "cached_tokens": 1, "prefill_s": 1}]))


class Intervals(unittest.TestCase):
    def test_merge_and_intersection(self):
        xs = gsp.merged([(5, 7), (0, 2), (1, 3)])
        self.assertEqual(xs, [[0, 3], [5, 7]])
        self.assertEqual(gsp.intersection_length(xs, [[2, 6]]), 2)
        self.assertEqual(gsp.intersection_length(xs, [[2, 6]], shift=3), 3)

    def test_align_recovers_offset(self):
        spans = [(1_000 * MS, 1_500 * MS), (2_000 * MS, 2_300 * MS), (3_000 * MS, 3_050 * MS)]
        true = 1_234 * MS
        busy = [(a - true + 10 * MS, b - true - 10 * MS) for a, b in spans]
        off, outside = gsp.align_offset(busy, spans, center=1_000 * MS, radius_ns=1_000 * MS, step_ns=MS)
        self.assertEqual(outside, 0)
        self.assertLessEqual(abs(off - true), 10 * MS)

    def test_gpu_classes(self):
        profile = {"rows": [{"send_ns": 0, "terminal_ns": 1000 * MS, "decode_s": 0.6},
                            {"send_ns": 2000 * MS, "terminal_ns": 2500 * MS, "decode_s": 0.25}]}
        busy = [(100 * MS, 900 * MS), (2100 * MS, 2400 * MS)]
        c = gsp.gpu_classes(profile, busy, 0)
        self.assertAlmostEqual(c["window_s"], 2.5)
        self.assertAlmostEqual(c["gpu_busy_s"], 1.1)
        self.assertAlmostEqual(c["gpu_idle_in_flight_s"], 0.4)
        self.assertAlmostEqual(c["no_request_s"], 1.0)
        self.assertAlmostEqual(c["prefill_window_s"], 0.65)
        self.assertAlmostEqual(c["gpu_busy_in_prefill_window_s"], 0.45)
        self.assertAlmostEqual(c["gpu_busy_decode_only_s"], 0.65)
        self.assertAlmostEqual(c["gpu_busy_outside_requests_s"], 0.0)


class MstExport(unittest.TestCase):
    def test_busy_union_from_execution_points(self):
        def row(t, fn, sub):
            return (f'<row><event-time>{t}</event-time><metal-command-buffer-id fmt="c0">0</metal-command-buffer-id>'
                    f'<uint32 fmt="{fn}">{fn}</uint32><uint32 fmt="0">0</uint32>'
                    f'<metal-command-buffer-id fmt="{sub}">{sub}</metal-command-buffer-id></row>')
        xml = ('<trace-query-result><node>' + row(100, 1, "s1") + row(300, 2, "s1") + row(250, 1, "s2")
               + row(400, 2, "s2") + row(600, 1, "s3") + row(700, 2, "s3") + '</node></trace-query-result>')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exec.xml"
            path.write_text(xml)
            self.assertEqual(gsp.mst_busy(path), [(100, 400), (600, 700)])


class Catsplit(unittest.TestCase):
    def test_decode_table_and_prefill_json(self):
        text = "\n".join([
            "[CATSPLIT] worker-exit lifetime dump: 3 categories, 30 total dispatches:",
            "[CATSPLIT]   category                                     total_ms   % step   total_disp  total_cbs",
            "[CATSPLIT]   moe_gate_up(mv_id)                              400.000    40.0%           10          5",
            "[CATSPLIT]   attention(kv_enc+flash)                         300.000    30.0%           10          5",
            "[CATSPLIT]   attn_pre(qkv_norm_rope+vnorm+fwht)              100.000    10.0%            5          5",
            "[CATSPLIT]   dense_q_proj                                    200.000    20.0%            5          5",
            "[CATSPLIT]   TOTAL                                          1000.000   100.0% (category-sum; CB-serialized, > real overlapped step)",
            '[PREFILL_CATSPLIT_JSON] {"schema":"hf2q.gemma4.prefill-catsplit.v1","forwards":2,"categories":'
            '{"pre_attn":{"gpu_ns":1000000},"attention":{"gpu_ns":3000000},"dense":{"gpu_ns":0},"moe":{"gpu_ns":0}}}',
        ])
        r = gsp.parse_catsplit(text)
        self.assertEqual(r["decode"], {"moe": 400.0, "attention": 400.0, "dense": 200.0})
        self.assertAlmostEqual(r["decode_share"]["attention"], 0.4)
        self.assertEqual(r["prefill"], {"pre_attn": 1.0, "attention": 3.0, "dense": 0.0, "moe": 0.0})
        self.assertAlmostEqual(r["prefill_share"]["attention"], 0.75)
        self.assertEqual(r["prefill_forwards"], 2)

    def test_absent(self):
        self.assertEqual(gsp.parse_catsplit("nothing"), {})


class Cli(unittest.TestCase):
    def test_compare_over_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                write_run(tmp, f"n4-{2 * i:03}-b{i:02}-experiment-a0",
                          {"agent-0": [(0, 1000, 0.4, 0.5, 100, 0, None)]}, makespan=1.2)
                write_run(tmp, f"n4-{2 * i + 1:03}-b{i:02}-peer-a0",
                          {"agent-0": [(0, 800, 0.2, 0.5, 100, 0, None)]}, makespan=1.0, engine="peer")
            out = Path(tmp) / "out.json"
            self.assertEqual(gsp.main(["compare", tmp, "--arm", "hf2q=*-experiment-*", "--base", "peer=*-peer-*",
                                       "--json", str(out)]), 0)
            result = json.loads(out.read_text())
        stages = result["comparison"]["stages"]
        self.assertEqual(result["comparison"]["pairing"], "block")
        self.assertAlmostEqual(stages["makespan"]["delta"], 0.2)
        self.assertAlmostEqual(stages["prefill"]["delta"], 0.2)
        self.assertAlmostEqual(stages["prefill"]["share_of_makespan_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
