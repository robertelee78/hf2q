#!/usr/bin/env python3
"""#310 requalification generation runner (ADR-059 gate-7 paired arms).

Serves one arm (baseline or graft-bound) of a family's IDENTICAL model
artifact via the current hf2q release binary, sends the 256-prompt
harmful suite (probes_harmful_all.jsonl) and the capability spot-check
(probes_capability.jsonl), and writes one results row per prompt:

  {"prompt_id", "prompt", "content", "finish_reason",
   "completion_tokens", "elapsed_s"}

Configuration is IDENTICAL across arms per the measurement doctrine
(temperature 0, seed 42, max_tokens 512, thinking off per request;
HF2Q_TQ_KV=1, speculation off, thinking budget 0 server-side; the ONLY
difference between arms is --kv-graft on the server boot line). The
default (family) scheduler is the canonical product config — gemma4 +
fifo-serial hits a resumable-prefill capacity fault (#326 note in
product_run.sh), so no scheduler override is applied here.

Resumable: rows already present (by prompt_id) in the output file are
skipped, so a killed run costs its progress, not the arm.

The server is a child process: it is ALWAYS killed before exit (one
model runtime at a time on the host).

Usage:
  python3 requal_generate.py --model <gguf> --arm baseline \
      --out-dir <dir> [--graft <gguf>] [--port 8397] [--bin <hf2q>]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES = ("probes_harmful_all.jsonl", "probes_capability.jsonl")
MAX_TOKENS = 512


def wait_ready(port: str, pid: int, log: str, timeout_s: int = 2400) -> str:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    return json.load(resp)["data"][0]["id"]
        except Exception:
            pass
        try:
            os.kill(pid, 0)
        except OSError:
            print("FAIL: server exited during startup", file=sys.stderr)
            with open(log) as f:
                sys.stderr.write(f.read()[-2000:])
            raise SystemExit(1)
        time.sleep(2)
    raise SystemExit("FAIL: server did not become ready")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, choices=("baseline", "grafted"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--graft", default=None)
    ap.add_argument("--port", default="8397")
    ap.add_argument("--bin", default=os.path.join(
        HERE, "..", "..", "target", "release", "hf2q"))
    args = ap.parse_args()
    if args.arm == "grafted" and not args.graft:
        raise SystemExit("--arm grafted requires --graft")

    os.makedirs(args.out_dir, exist_ok=True)
    log = os.path.join(args.out_dir, f"serve-{args.arm}.log")
    env = dict(os.environ, HF2Q_TQ_KV="1", HF2Q_QWEN_SPECULATION="off")
    cmd = [os.path.abspath(args.bin), "serve", args.model,
           "--port", args.port, "--quiet",
           "--default-thinking-token-budget", "0"]
    if args.graft:
        cmd += ["--kv-graft", args.graft]
    print(f"[requal] arm {args.arm}: booting {' '.join(cmd)}", flush=True)
    with open(log, "w") as lf:
        srv = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env,
                               start_new_session=True)
    model_id = None
    try:
        model_id = wait_ready(args.port, srv.pid, log)
        print(f"[requal] arm {args.arm}: ready (model id {model_id})", flush=True)

        for suite in SUITES:
            probes = [json.loads(l) for l in open(os.path.join(HERE, suite))
                      if l.strip()]
            out_path = os.path.join(
                args.out_dir,
                f"results-{args.arm}-{os.path.splitext(suite)[0]}.jsonl")
            done = set()
            if os.path.exists(out_path):
                for line in open(out_path):
                    try:
                        done.add(json.loads(line)["prompt_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            with open(out_path, "a") as out:
                for probe in probes:
                    if probe["id"] in done:
                        continue
                    body = json.dumps({
                        "model": model_id,
                        "messages": [{"role": "user",
                                      "content": probe["prompt"]}],
                        "temperature": 0,
                        "max_tokens": MAX_TOKENS,
                        "seed": 42,
                        "stream": False,
                        "hf2q_enable_thinking": False,
                    }).encode()
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{args.port}/v1/chat/completions",
                        data=body, headers={"Content-Type": "application/json"})
                    started = time.time()
                    record = {"arm": args.arm, "suite": suite,
                              "prompt_id": probe["id"], "prompt": probe["prompt"]}
                    try:
                        with urllib.request.urlopen(req, timeout=900) as resp:
                            result = json.load(resp)
                        choice = result["choices"][0]
                        record["content"] = choice["message"].get("content") or ""
                        record["finish_reason"] = choice.get("finish_reason")
                        usage = result.get("usage") or {}
                        record["completion_tokens"] = usage.get("completion_tokens")
                    except urllib.error.HTTPError as err:
                        record["content"] = ""
                        record["http_error"] = err.code
                        record["error_body"] = err.read().decode("utf-8", "replace")[:300]
                    except Exception as exc:  # fail-closed, never counts as compliance
                        record["content"] = ""
                        record["http_error"] = "exception"
                        record["error_body"] = str(exc)[:300]
                    record["elapsed_s"] = round(time.time() - started, 2)
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    print(f"[requal] {args.arm} {probe['id']}: "
                          f"finish={record.get('finish_reason')} "
                          f"tok={record.get('completion_tokens')}", flush=True)
            print(f"[requal] arm {args.arm}: {suite} complete -> {out_path}",
                  flush=True)
    finally:
        if srv.poll() is None:
            os.killpg(os.getpgid(srv.pid), signal.SIGTERM)
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(srv.pid), signal.SIGKILL)
                srv.wait()
        print(f"[requal] arm {args.arm}: server stopped", flush=True)


if __name__ == "__main__":
    main()
