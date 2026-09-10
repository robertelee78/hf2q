#!/usr/bin/env python3
"""ADR-054 gate 3-4 pass runner: serve DeepSeek (optionally with GLP dose),
run the fixed 48-prompt panel, write results.jsonl, stop the server.

Usage: run_pass.py <pass_label> [extra serve flags...]
Honors host rules: preflight pgrep/memory_pressure, always stops the server.
"""
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

WORKTREE = "/var/folders/bd/84n3hd3120sdr990qkfbgvj00000gp/T/opencode/hf2q-glp-fix"
HF2Q = f"{WORKTREE}/target/release/hf2q"
DEEPSEEK = "/opt/hf2q/models/deepseek4/DeepSeek-V4-Flash-0731-agentic-q2.gguf"
PORT = 18085
BASE = f"http://127.0.0.1:{PORT}"
GATE = f"{WORKTREE}/scripts/grammar_probe/gate34"
PANEL = f"{GATE}/panel.tsv"


def die(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def preflight():
    r = subprocess.run(["pgrep", "-x", "hf2q"], capture_output=True, text=True)
    if r.returncode == 0:
        die(f"hf2q already running: {r.stdout.strip()}")
    mp = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)", mp)
    free = int(m.group(1)) if m else 0
    if free < 85:
        die(f"memory free {free}% < 85%")
    return free


def load_panel():
    rows = []
    with open(PANEL) as fh:
        for line in fh:
            pid, stratum, prompt = line.rstrip("\n").split("\t", 2)
            rows.append((pid, stratum, prompt))
    return rows


def wait_models(proc, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            die("server exited during startup; log tail:\n" + tail_log())
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
                return json.load(r)["data"][0]["id"]
        except Exception:
            time.sleep(10)
    die("timeout waiting for /v1/models")


def wait_warm(proc, model, timeout_s=300):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Say ready."}],
        "max_tokens": 8, "temperature": 0, "hf2q_enable_thinking": False,
    }).encode()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            die("server exited during warmup; log tail:\n" + tail_log())
        req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                json.load(r)
                return
        except urllib.error.HTTPError as e:
            print(f"  warmup HTTP {e.code}: {e.read()[:200]!r}", file=sys.stderr)
            time.sleep(10)
        except Exception as e:
            print(f"  warmup error: {e}", file=sys.stderr)
            time.sleep(10)
    die("timeout waiting for warm generation")


def tail_log(n=30):
    try:
        with open(LOG) as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return "(no log)"


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # belt and suspenders: kill anything left on our port command line
    subprocess.run(["pkill", "-x", "hf2q"], capture_output=True)
    for _ in range(30):
        r = subprocess.run(["pgrep", "-x", "hf2q"], capture_output=True)
        if r.returncode != 0:
            return True
        time.sleep(2)
    return False


def main():
    global LOG
    if len(sys.argv) < 2:
        die("usage: run_pass.py <pass_label> [extra serve flags...]")
    label = sys.argv[1]
    extra = sys.argv[2:]
    out_dir = f"{GATE}/{label}"
    LOG = f"{out_dir}/server.log"
    free = preflight()

    cmd = [HF2Q, "serve", "--model", DEEPSEEK, "--port", str(PORT)] + extra
    print(f"[{label}] starting: {' '.join(cmd)} (mem free {free}%)", file=sys.stderr)
    t_start = time.time()
    log_fh = open(LOG, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    try:
        model_id = wait_models(proc)
        t_ready_models = time.time()
        print(f"[{label}] /v1/models ok, model id: {model_id} "
              f"({t_ready_models - t_start:.0f}s)", file=sys.stderr)
        wait_warm(proc, model_id)
        t_ready = time.time()
        print(f"[{label}] warm generation ok ({t_ready - t_start:.0f}s)", file=sys.stderr)

        panel = load_panel()
        results_path = f"{out_dir}/results.jsonl"
        n_err = 0
        with open(results_path, "w") as out:
            for pid, stratum, prompt in panel:
                body = json.dumps({
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256, "temperature": 0,
                    "hf2q_enable_thinking": False,
                }).encode()
                req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
                row = {"pass": label, "prompt_id": pid, "stratum": stratum}
                try:
                    with urllib.request.urlopen(req, timeout=600) as r:
                        resp = json.load(r)
                    choice = resp["choices"][0]
                    row["finish"] = choice.get("finish_reason")
                    row["content"] = choice["message"].get("content") or ""
                    row["completion_tokens"] = resp.get("usage", {}).get("completion_tokens")
                except Exception as e:
                    n_err += 1
                    row["finish"] = "error"
                    row["content"] = f"REQUEST_ERROR: {e}"
                    row["completion_tokens"] = None
                out.write(json.dumps(row) + "\n")
                out.flush()
                state = row.get("finish", "?")
                print(f"  [{pid}] finish={state} "
                      f"ctok={row.get('completion_tokens')}", file=sys.stderr)
        t_end = time.time()
        meta = {
            "pass": label, "serve_flags": cmd[1:], "model_id": model_id,
            "mem_free_pct_preflight": free,
            "t_start": t_start, "t_models_ready": t_ready_models,
            "t_gen_ready": t_ready, "t_end": t_end,
            "load_wall_s": round(t_ready - t_start, 1),
            "panel_wall_s": round(t_end - t_ready, 1),
            "total_wall_s": round(t_end - t_start, 1),
            "panel_rows": len(panel), "request_errors": n_err,
        }
        with open(f"{out_dir}/meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[{label}] DONE: {len(panel)} rows, {n_err} request errors, "
              f"total {t_end - t_start:.0f}s", file=sys.stderr)
    finally:
        log_fh.close()
        dead = stop_server(proc)
        print(f"[{label}] server stopped, verified dead: {dead}", file=sys.stderr)
        if not dead:
            sys.exit(3)


if __name__ == "__main__":
    main()
