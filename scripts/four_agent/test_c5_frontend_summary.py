"""C5 frontend summary controls: synthetic log lines only, never model or hardware evidence."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import c5_frontend_summary as s


def text_line(**fields):
    body = " ".join(f'{k}="{v}"' if isinstance(v, str) and k == "outcome" else
                    f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in fields.items())
    return f"\x1b[32m INFO\x1b[0m hf2q::serve::frontend_timing: frontend timing {body}\n"


def record(**over):
    base = {"seq": 0, "family": "Qwen35", "stream": True, "outcome": "first_reasoning", "messages": 6,
            "tools": 3, "rendered_bytes": 9000, "prompt_tokens": 2300, "admission_us": 900,
            "vision_us": 1, "constraint_us": 10, "render_us": 400, "tokenize_us": 1200,
            "overflow_rewrite_us": 0, "stable_render_us": 390, "stable_tokenize_us": 1100,
            "prefix_match_us": 5, "probe_render_us": 80, "budget_us": 15, "enqueue_us": 30,
            "prepare_us": 4200, "first_output_us": 9_000_000}
    base.update(over)
    return base


class ParseTests(unittest.TestCase):
    def test_text_line_round_trips_types_and_ignores_other_lines(self):
        fields = s.parse_line(text_line(**record()))
        self.assertEqual(fields["family"], "Qwen35")
        self.assertIs(fields["stream"], True)
        self.assertEqual(fields["outcome"], "first_reasoning")
        self.assertEqual(fields["first_output_us"], 9_000_000)
        self.assertIsNone(s.parse_line(" INFO hf2q::serve::api::handlers: chat completion prepared\n"))

    def test_json_line_is_accepted(self):
        line = json.dumps({"level": "INFO", "fields": {"message": "frontend timing", **record()}})
        self.assertEqual(s.parse_line(line)["render_us"], 400)


class SummaryTests(unittest.TestCase):
    def test_frontend_excludes_admission_vision_enqueue(self):
        self.assertEqual(s.frontend_us(record()), 10 + 400 + 1200 + 390 + 1100 + 5 + 80 + 15)

    def test_cold_and_unary_requests_do_not_enter_the_warm_share(self):
        rows = [record(), record(messages=2, first_output_us=10),
                {**record(stream=False, completion_us=20), "first_output_us": 5}]
        fam = s.summarize(rows)["families"]["Qwen35"]
        self.assertEqual(fam["requests"], 3)
        self.assertEqual(fam["warm_streamed"], 1)
        self.assertEqual(fam["verdict"], "not_material")

    def test_share_above_ten_percent_is_material(self):
        rows = [record(first_output_us=20_000), record(first_output_us=25_000)]
        self.assertEqual(s.summarize(rows)["families"]["Qwen35"]["verdict"], "material")

    def test_no_warm_stream_is_unmeasured_not_a_pass(self):
        rows = [record(messages=2)]
        self.assertEqual(s.summarize(rows)["families"]["Qwen35"]["verdict"], "unmeasured")

    def test_cli_refuses_logs_without_timing_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.stderr"
            path.write_text(" INFO hf2q: model prepared\n")
            self.assertEqual(s.main([str(path)]), 2)
            path.write_text(text_line(**record()) + text_line(**record(family="Deepseek4")))
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(s.main([str(path)]), 0)
            self.assertEqual(sorted(json.loads(out.getvalue())["families"]), ["Deepseek4", "Qwen35"])


if __name__ == "__main__":
    unittest.main()
