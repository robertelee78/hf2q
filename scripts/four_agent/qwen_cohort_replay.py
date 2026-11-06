"""Qwen cohort replay (#717): one recorded request decoded under a controlled joint window.

Replays recorded four-agent request bodies against an already running hf2q server, then decodes
the target request either alone (`solo`) or next to a companion that leaves the cohort after a
measured number of joint steps (`joint:M`). It tests whether the target's output depends on how
many of its decode steps ran jointly, the defect #717 fixes. CPU-only client; it never starts a
server or loads a model.

  run      --manifest M --arm solo|joint:M [--logprobs] --base-url URL --out DIR
  analyze  ARM_DIR...   (each holds client/summary.json; server.stderr is read when present)

Manifest (JSON, schema `hf2q.qwen-cohort-replay.manifest.v1`):
  recording       run directory with client/agent-N/turn-K/request.json
  agents          agents warmed per turn (int)
  warm_turns      turns 0..warm_turns-1 replayed before the test (int)
  target          {"agent": int, "turn": int}           request under test, sent verbatim
  companion       {"agent": int, "turn": int, "question": str}
                  that agent's recorded body plus one appended user message, streamed with usage
  pre_events      companion token events before the target is sent (int, default 4)
  references      optional {"label": path to assembled-assistant.json} for output classes

Warm-up turns are sent verbatim except `max_completion_tokens=1` and `stream=false`. Slot-local
prompt-boundary anchors are captured during prefill, so each slot's anchor chain is rebuilt
without decoding. The resume point then differs from a live run, which resumes at the
prompt-5 boundary after a full turn: compare arms with each other, never with the recording.
`analyze` requires every arm's per-request cached_tokens to match the first arm's.
"""
import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.parse

MANIFEST_SCHEMA = "hf2q.qwen-cohort-replay.manifest.v1"
REPORT_SCHEMA = "hf2q.qwen-cohort-replay.report.v1"
TOKEN_KINDS = ("reasoning", "content", "tool")


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_bytes(path, data, mode="wb"):
    with open(path, mode) as f:
        f.write(data)


def load_manifest(path):
    manifest = read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit(f"{path}: schema must be {MANIFEST_SCHEMA}")
    for key in ("recording", "agents", "warm_turns", "target", "companion"):
        if key not in manifest:
            raise SystemExit(f"{path}: missing {key}")
    manifest.setdefault("pre_events", 4)
    manifest.setdefault("references", {})
    return manifest


def assemble(raw):
    """Assistant output of a non-stream JSON body or an SSE stream."""
    text = raw.decode()
    if text.lstrip().startswith("{"):
        body = json.loads(text)
        choice = body["choices"][0]
        msg = choice["message"]
        calls = [[c["function"]["name"], c["function"]["arguments"]] for c in msg.get("tool_calls") or []]
        return {"tool_calls": calls, "content": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content") or "", "usage": body.get("usage"),
                "finish": choice.get("finish_reason"), "logprobs": choice.get("logprobs"),
                "timing": body.get("x_hf2q_timing")}
    calls, content, reasoning, usage, finish, timing = {}, [], [], None, None, None
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        usage = chunk.get("usage") or usage
        timing = chunk.get("x_hf2q_timing") or timing
        for choice in chunk.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {})
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc["index"], ["", ""])
                fn = tc.get("function", {})
                slot[0] += fn.get("name") or ""
                slot[1] += fn.get("arguments") or ""
    return {"tool_calls": [calls[i] for i in sorted(calls)], "content": "".join(content),
            "reasoning": "".join(reasoning), "usage": usage, "finish": finish, "logprobs": None,
            "timing": timing}


def event_kind(line):
    """Kind of one SSE line: a token kind, `other`, or None for non-data lines."""
    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
        return None
    chunk = json.loads(line[6:])
    for choice in chunk.get("choices", []):
        delta = choice.get("delta", {})
        if delta.get("reasoning_content"):
            return "reasoning"
        if delta.get("content"):
            return "content"
        if delta.get("tool_calls"):
            return "tool"
    return "other"


def joint_window(events, target_send_ns, target_done_ns):
    """Joint steps the target shared with the companion.

    The target's prefill stalls the companion's stream: the longest gap between companion token
    events after the target was sent. Every companion token after that gap and before the
    target finished came from a joint step that also produced one target token.
    """
    times = [t for t, kind in events if kind in TOKEN_KINDS]
    gaps = [(times[i + 1] - times[i], i) for i in range(len(times) - 1) if times[i + 1] > target_send_ns]
    if not gaps:
        return {"k": 0, "pre": len(times), "stall_ns": None}
    gap, index = max(gaps)
    k = sum(1 for t in times[index + 1:] if t <= target_done_ns)
    return {"k": k, "pre": index + 1, "stall_ns": gap}


def output_key(record):
    return {"reasoning": record["reasoning"], "content": record["content"],
            "tool_calls": [list(call) for call in record["tool_calls"]]}


def output_digest(record):
    return hashlib.sha256(json.dumps(output_key(record), sort_keys=True).encode()).hexdigest()


def cached_map(records):
    return {label: ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            for label, r in records.items()}


def cached_mismatches(arms):
    """Validity across arms. Every request must report cached_tokens, and a request present in
    two arms must report the same value in both. A solo arm has no companion, so only shared
    labels are compared."""
    issues = [(name, label, None) for name, cached in arms for label, value in sorted(cached.items())
              if value is None]
    seen = {}
    for name, cached in arms:
        for label, value in sorted(cached.items()):
            if value is None:
                continue
            if label in seen and seen[label][1] != value:
                issues.append((name, label, (seen[label][1], value)))
            seen.setdefault(label, (name, value))
    return issues


def joint_turn_lines(stderr_text):
    """Per-turn joint decode debug lines, keyed by lanes/rows/serial."""
    counts = {}
    for line in stderr_text.splitlines():
        if "Qwen joint decode turn" in line:
            match = re.search(r"lanes=(\d+) rows=(\d+) serial=(\d+)", line)
            key = match.group(0) if match else "unparsed"
            counts[key] = counts.get(key, 0) + 1
    return counts


class Client:
    def __init__(self, base_url, out):
        self.base = urllib.parse.urlparse(base_url)
        self.out = out
        self.records = {}
        self.lock = threading.Lock()

    def post(self, label, body, on_token=None):
        raw = json.dumps(body).encode()
        conn = http.client.HTTPConnection(self.base.hostname, self.base.port, timeout=900)
        send_ns = time.monotonic_ns()
        conn.request("POST", "/v1/chat/completions", raw, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        events, data = [], bytearray()
        if body.get("stream"):
            tokens = 0
            while True:
                line = resp.readline()
                if not line:
                    break
                data += line
                kind = event_kind(line)
                if kind is not None:
                    events.append((time.monotonic_ns(), kind))
                    if kind in TOKEN_KINDS:
                        tokens += 1
                        if on_token is not None:
                            on_token(tokens)
        else:
            data += resp.read()
        done_ns = time.monotonic_ns()
        status = resp.status
        conn.close()
        dst = os.path.join(self.out, label)
        os.makedirs(dst)
        write_bytes(os.path.join(dst, "request.json"), raw)
        write_bytes(os.path.join(dst, "response.bin"), bytes(data))
        if status != 200:
            raise RuntimeError(f"{label}: HTTP {status}: {bytes(data)[:400]!r}")
        got = assemble(bytes(data))
        with self.lock:
            self.records[label] = {"request_sha256": hashlib.sha256(raw).hexdigest(),
                                   "send_ns": send_ns, "done_ns": done_ns, "events": events, **got}
        return got


def recorded(manifest, agent, turn):
    path = os.path.join(manifest["recording"], "client", f"agent-{agent}", f"turn-{turn}", "request.json")
    return read_json(path)


def concurrent(jobs):
    errors = []

    def run(fn):
        try:
            fn()
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(fn,)) for fn in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def warm(client, manifest):
    for turn in range(manifest["warm_turns"]):
        jobs = []
        for agent in range(manifest["agents"]):
            body = recorded(manifest, agent, turn)
            body["max_completion_tokens"] = 1
            body["stream"] = False
            body.pop("stream_options", None)
            jobs.append(lambda b=body, a=agent, t=turn: client.post(f"warm/agent-{a}/turn-{t}", b))
        concurrent(jobs)


def run_arm(client, manifest, arm, logprobs):
    target_spec = manifest["target"]
    target = recorded(manifest, target_spec["agent"], target_spec["turn"])
    if logprobs:
        target["logprobs"], target["top_logprobs"] = True, 5
    target_label = f"test/agent-{target_spec['agent']}/turn-{target_spec['turn']}"
    if arm == "solo":
        client.post(target_label, target)
        return
    match = re.fullmatch(r"joint:(\d+)", arm)
    if not match:
        raise SystemExit(f"unknown arm {arm!r}; expected solo or joint:M")
    spec = manifest["companion"]
    companion = recorded(manifest, spec["agent"], spec["turn"])
    companion["messages"].append({"role": "user", "content": spec["question"]})
    companion["stream"] = True
    companion["stream_options"] = {"include_usage": True}
    companion["max_completion_tokens"] = int(match.group(1))
    ready = threading.Event()
    pre_events = manifest["pre_events"]

    def send_target():
        if not ready.wait(120):
            raise RuntimeError("companion never streamed")
        client.post(target_label, target)

    concurrent([lambda: client.post("test/companion", companion,
                                    on_token=lambda n: n >= pre_events and ready.set()),
                send_target])


def cmd_run(args):
    manifest = load_manifest(args.manifest)
    os.makedirs(args.out)
    client = Client(args.base_url, args.out)
    error = None
    try:
        warm(client, manifest)
        run_arm(client, manifest, args.arm, args.logprobs)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    summary = {"manifest": manifest, "arm": args.arm, "logprobs": args.logprobs, "error": error,
               "records": client.records}
    write_bytes(os.path.join(args.out, "summary.json"), json.dumps(summary, indent=1), "x")
    print(json.dumps({"arm": args.arm, "error": error, "requests": len(client.records)}))
    return 1 if error else 0


def analyze(arm_dirs):
    arms, cached = [], []
    for arm_dir in arm_dirs:
        summary = read_json(os.path.join(arm_dir, "client", "summary.json"))
        manifest = summary["manifest"]
        records = summary["records"]
        spec = manifest["target"]
        target = records.get(f"test/agent-{spec['agent']}/turn-{spec['turn']}")
        row = {"dir": arm_dir, "arm": summary["arm"], "logprobs": summary["logprobs"],
               "error": summary["error"], "digest": None, "class": None, "joint_window": None}
        if target is not None:
            row["digest"] = output_digest(target)
            classes = [label for label, path in sorted(manifest["references"].items())
                       if output_key(target) == output_key(reference_record(path))]
            row["class"] = classes[0] if classes else "other"
            companion = records.get("test/companion")
            if companion is not None:
                row["joint_window"] = joint_window(companion["events"], target["send_ns"], target["done_ns"])
            row["decode_tokens_per_sec"] = (target.get("timing") or {}).get("decode_tokens_per_sec")
        stderr = os.path.join(arm_dir, "server.stderr")
        if os.path.exists(stderr):
            with open(stderr, errors="replace") as f:
                row["joint_turn_lines"] = joint_turn_lines(f.read())
        arms.append(row)
        cached.append((arm_dir, cached_map(records)))
    mismatches = cached_mismatches(cached)
    plain = [row["digest"] for row in arms if not row["logprobs"] and row["digest"]]
    return {"schema": REPORT_SCHEMA, "arms": arms,
            "valid": not mismatches and all(row["error"] is None for row in arms),
            "cached_mismatches": mismatches,
            "distinct_outputs": len(set(plain)),
            "byte_identical": len(set(plain)) == 1 and len(plain) == sum(1 for row in arms if not row["logprobs"])}


def reference_record(path):
    msg = read_json(path)
    calls = [[c["function"]["name"], c["function"]["arguments"]] for c in msg.get("tool_calls") or []]
    return {"reasoning": msg.get("reasoning_content") or "", "content": msg.get("content") or "",
            "tool_calls": calls}


def cmd_analyze(args):
    report = analyze(args.arm_dirs)
    print(json.dumps(report, indent=1))
    return 0 if report["valid"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--arm", required=True)
    run.add_argument("--logprobs", action="store_true")
    run.add_argument("--base-url", required=True)
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)
    ana = sub.add_parser("analyze")
    ana.add_argument("arm_dirs", nargs="+")
    ana.set_defaults(func=cmd_analyze)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
