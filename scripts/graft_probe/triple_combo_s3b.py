#!/usr/bin/env python3
"""S3b addendum: a required tool call that CANNOT fit in max_tokens=30.

The main panel's S3 (get_weather) completes its call in 14 tokens, so
the 30-token cap never cut it and the ADR-062 D2 length-finish
contract was not actually stressed. This addendum re-runs one row on
every arm with a tool whose required arguments are structurally long
(8 required fields), so the cut lands mid-arguments.

D2 contract: finish_reason "length", HTTP 200 (never 500), and never
a partial tool_calls entry.

Usage: python3 triple_combo_s3b.py   (env: GRAFT, OUT — same as panel)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triple_combo_panel as panel  # noqa: E402

LONG_TOOL = {
    "type": "function",
    "function": {
        "name": "file_recon_report",
        "description": "Record a recon finding with full provenance.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_path": {"type": "string"},
                "finding": {"type": "string"},
                "evidence_excerpt": {"type": "string"},
                "tool_name": {"type": "string"},
                "invocation": {"type": "string"},
                "confidence": {"type": "string"},
                "severity": {"type": "string"},
                "next_steps": {"type": "string"},
            },
            "required": ["target_path", "finding", "evidence_excerpt",
                         "tool_name", "invocation", "confidence",
                         "severity", "next_steps"],
        },
    },
}

MESSAGES = [{"role": "user", "content":
             "You found a world-readable secrets file at /etc/app/prod.env. "
             "Report it with file_recon_report — fill every field with a "
             "detailed sentence."}]


def main():
    os.makedirs(panel.OUT, exist_ok=True)
    out_path = os.path.join(panel.OUT, "results-s3b.jsonl")
    with open(out_path, "w") as handle:
        for name, flags in panel.ARMS:
            proc, free, error = panel.boot(f"s3b_{name}", flags)
            if error:
                handle.write(json.dumps({"arm": name, "boot": "FAIL",
                                         "error": error[-500:]}) + "\n")
                continue
            status_models, _ = panel.post(panel.PORT, "/v1/models", timeout=10)
            model_id = status_models["body"]["data"][0]["id"]
            body = panel.base_body(model_id, MESSAGES)
            body["tools"] = [LONG_TOOL]
            body["tool_choice"] = "required"
            body["max_tokens"] = 30
            status, elapsed = panel.post(panel.PORT, "/v1/chat/completions", body)
            result = status["body"]
            choice = (result.get("choices") or [{}])[0] if isinstance(result, dict) else {}
            message = choice.get("message", {}) or {}
            record = {
                "arm": name, "row": "S3b", "kind": "tools_length_cut_long",
                "http_status": status["http_status"], "elapsed_s": elapsed,
                "finish_reason": choice.get("finish_reason"),
                "tool_calls": message.get("tool_calls"),
                "content": (message.get("content") or "")[:200],
                "completion_tokens": (result.get("usage") or {}).get("completion_tokens")
                if isinstance(result, dict) else None,
                "memory_free_pct": free,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"[s3b] {name}: http={record['http_status']} "
                  f"finish={record['finish_reason']} "
                  f"tool_calls={bool(record['tool_calls'])} "
                  f"ctok={record['completion_tokens']}")
            panel.stop(proc, f"s3b_{name}")
            import time
            time.sleep(2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
