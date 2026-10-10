#!/usr/bin/env bash
# ADR-059 gate 6 — throughput report: prefill/decode tok/s, grafted vs
# ungrafted, median of multiple runs (measurement doctrine: identical
# model artifact, prompt, sampling, budget; the ONLY difference between
# arms is the graft).
#
# Arms (fresh serve process per arm), A-B-A order to de-confound arm
# order from machine state (thermal drift on sustained GPU load moves
# decode tok/s by more than the graft's mechanical cost, so a single
# A-B pass is not a valid comparison):
#   baseline  — no graft (measured FIRST)
#   grafted   — --kv-graft live bank (64 slots, every full-attn layer)
#   baseline2 — no graft again (measured LAST; if baseline2 ~= baseline
#               the machine state was stable and the graft delta is
#               real; if baseline2 ~= grafted the A-B delta was thermal
#               drift and the honest graft delta is ~0)
#
# The graft's expected cost is attention over N extra slots at covered
# full-attention layers only (≈ a prompt N tokens longer at those
# layers). Acceptance per ADR-059: within the measured noise band, or
# the delta is documented and accepted — never silently shipped.
#
# Configuration (production substrate, matched across arms):
#   HF2Q_TQ_KV=1              — the production KV substrate
#   --scheduler fifo-serial   — the production default scheduler
#   HF2Q_QWEN_SPECULATION=off — a bound graft skips MTP; matched
#   --default-thinking-token-budget 0 — thinking on, unbudgeted
#
# Per arm: 2 warmup streaming requests, then K measured streaming
# requests (temperature 0, max_tokens 128, fixed ~400-token prompt).
# Client-side timing: ttft = first SSE content chunk; decode window =
# last chunk - first. Prefill tok/s = prompt_tokens / ttft; decode
# tok/s = (completion_tokens - 1) / decode window. Medians across K.
#
# Usage:
#   bash scripts/graft_probe/throughput.sh [MODEL_GGUF] [PORT]
#   K (measured runs per arm) defaults to 5; override with
#   HF2Q_THROUGHPUT_RUNS.
set -euo pipefail

MODEL="${1:-/opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf}"
PORT="${2:-8392}"
RUNS="${HF2Q_THROUGHPUT_RUNS:-5}"
BIN="$(cd "$(dirname "$0")/../.." && pwd)/target/release/hf2q"
WORK="$(mktemp -d /tmp/hf2q-graft-throughput.XXXXXX)"
CANARY_PID=""

stop_server() {
    if [[ -n "$CANARY_PID" ]] && kill -0 "$CANARY_PID" 2>/dev/null; then
        kill "$CANARY_PID" 2>/dev/null || true
        wait "$CANARY_PID" 2>/dev/null || true
    fi
    CANARY_PID=""
}
trap stop_server EXIT

boot() {
    local log="$WORK/serve-$1.log"
    shift
    HF2Q_TQ_KV=1 HF2Q_QWEN_SPECULATION=off "$BIN" serve "$MODEL" \
        --port "$PORT" --quiet --scheduler fifo-serial \
        --default-thinking-token-budget 0 "$@" >"$log" 2>&1 &
    CANARY_PID=$!
    local i
    for i in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$CANARY_PID" 2>/dev/null; then
            echo "FAIL: serve exited during startup; log tail:" >&2
            tail -5 "$log" >&2 || true
            exit 1
        fi
        sleep 2
    done
    echo "FAIL: serve did not become ready in 480s" >&2
    exit 1
}

# A deterministic ~400-token prompt (tokenized length reported by the
# server in usage.prompt_tokens; the exact count is recorded per run).
PROMPT="$(python3 - <<'PY'
sentence = ("The quick analysis of distributed cache coherence covers "
            "invalidation protocols, write-back policies, and the trade-offs "
            "between eager and lazy propagation. ")
print("Summarize the key trade-offs in one paragraph. " + sentence * 12)
PY
)"

run_stream() {
    # One streaming request; prints "ttft_s total_s prompt_tokens
    # completion_tokens" on stdout.
    python3 - "$PORT" "$MODEL_ID" "$PROMPT" <<'PY'
import json, sys, time, urllib.request

port, model_id, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": 128,
    "seed": 42,
    "stream": True,
    "stream_options": {"include_usage": True},
}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:%s/v1/chat/completions" % port,
    data=body, headers={"Content-Type": "application/json"})
start = time.perf_counter()
ttft = None
prompt_tokens = completion_tokens = None
with urllib.request.urlopen(req, timeout=600) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        event = json.loads(payload)
        if event.get("usage"):
            prompt_tokens = event["usage"].get("prompt_tokens")
            completion_tokens = event["usage"].get("completion_tokens")
        delta = event.get("choices", [{}])[0].get("delta", {})
        if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
            ttft = time.perf_counter() - start
total = time.perf_counter() - start
if ttft is None or prompt_tokens is None or completion_tokens is None:
    sys.exit("throughput probe: stream ended without usage or first token")
print("%.4f %.4f %d %d" % (ttft, total, prompt_tokens, completion_tokens))
PY
}

measure_arm() {
    local name="$1"
    shift
    echo "── arm $name: booting serve on :$PORT"
    boot "$name" "$@"
    MODEL_ID="$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
    echo "   arm $name: model id $MODEL_ID"
    local i
    for i in $(seq 1 2); do
        run_stream >/dev/null
        echo "   arm $name: warmup $i done"
    done
    : >"$WORK/runs-$name.txt"
    for i in $(seq 1 "$RUNS"); do
        local line
        line="$(run_stream)"
        echo "   arm $name: run $i  ttft/total/tokens: $line"
        echo "$line" >>"$WORK/runs-$name.txt"
    done
    stop_server
    sleep 2
}

echo "throughput workdir: $WORK"
python3 "$(dirname "$0")/build_canary_graft.py" \
    --model "$MODEL" \
    --out-zero /dev/null \
    --out-live "$WORK/graft-live.gguf" \
    --n-slots 64 --scale 8.0

# Pre-warm: the host's GPU warm-up ramp moves decode tok/s by >10% over
# the first ~10 minutes of sustained load (measured: an un-warmed A-B-A
# showed +13.2% decode / +31.1% prefill A-vs-A drift, swamping any graft
# effect). Drive the ramp with ~3 minutes of load BEFORE the first
# measured arm so all arms measure on the plateau.
echo "── pre-warm: driving the GPU warm-up ramp (~3 min)"
boot prewarm
MODEL_ID="$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
PREWARM_END=$(( $(date +%s) + 180 ))
while [[ $(date +%s) -lt $PREWARM_END ]]; do
    run_stream >/dev/null || true
done
stop_server
sleep 2

measure_arm baseline
measure_arm grafted --kv-graft "$WORK/graft-live.gguf"
measure_arm baseline2

python3 - "$WORK" "$RUNS" "$(git -C "$(dirname "$0")/../.." rev-parse HEAD)" "$MODEL" <<'PY'
import json, os, statistics, sys

work, runs, git_rev, model = sys.argv[1:5]

def arm(name):
    rows = []
    with open(os.path.join(work, "runs-%s.txt" % name)) as f:
        for line in f:
            ttft, total, prompt, completion = line.split()
            ttft, total = float(ttft), float(total)
            prompt, completion = int(prompt), int(completion)
            decode_window = max(total - ttft, 1e-9)
            rows.append({
                "ttft_s": ttft,
                "total_s": total,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "prefill_tok_s": prompt / ttft,
                "decode_tok_s": max(completion - 1, 1) / decode_window,
            })
    return rows

def medians(rows):
    return {
        "ttft_s_median": statistics.median(r["ttft_s"] for r in rows),
        "prefill_tok_s_median": statistics.median(r["prefill_tok_s"] for r in rows),
        "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in rows),
        "completion_tokens": rows[0]["completion_tokens"],
        "prompt_tokens": rows[0]["prompt_tokens"],
    }

baseline, grafted, baseline2 = arm("baseline"), arm("grafted"), arm("baseline2")
mb, mg, mb2 = medians(baseline), medians(grafted), medians(baseline2)
# Machine-state drift across the campaign (baseline2 vs baseline): the
# graft's honest delta is grafted vs the LOCAL baseline state, and the
# drift band is the A-vs-A spread.
drift = {
    "prefill_tok_s_pct": 100.0 * (mb2["prefill_tok_s_median"] - mb["prefill_tok_s_median"]) / mb["prefill_tok_s_median"],
    "decode_tok_s_pct": 100.0 * (mb2["decode_tok_s_median"] - mb["decode_tok_s_median"]) / mb["decode_tok_s_median"],
}
delta_ab = {
    "prefill_tok_s_pct": 100.0 * (mg["prefill_tok_s_median"] - mb["prefill_tok_s_median"]) / mb["prefill_tok_s_median"],
    "decode_tok_s_pct": 100.0 * (mg["decode_tok_s_median"] - mb["decode_tok_s_median"]) / mb["decode_tok_s_median"],
}
delta_ba = {
    "prefill_tok_s_pct": 100.0 * (mg["prefill_tok_s_median"] - mb2["prefill_tok_s_median"]) / mb2["prefill_tok_s_median"],
    "decode_tok_s_pct": 100.0 * (mg["decode_tok_s_median"] - mb2["decode_tok_s_median"]) / mb2["decode_tok_s_median"],
}
manifest = {
    "report": "adr-059-gate-6-throughput",
    "design": "A-B-A (fresh process per arm; de-confounds arm order from thermal/machine drift)",
    "git_rev": git_rev,
    "model": model,
    "substrate": "tq_kv=1 scheduler=fifo-serial speculation=off",
    "runs_per_arm": int(runs),
    "graft": {"n_slots": 64, "layers": "every full-attention layer", "scale": 8.0},
    "baseline": mb,
    "grafted": mg,
    "baseline2": mb2,
    "machine_drift_baseline2_vs_baseline": drift,
    "delta_grafted_vs_baseline": delta_ab,
    "delta_grafted_vs_baseline2": delta_ba,
    "raw": {"baseline": baseline, "grafted": grafted, "baseline2": baseline2},
}
path = os.path.join(work, "throughput-manifest.json")
with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
print("manifest: %s" % path)
for label, m in (("baseline ", mb), ("grafted  ", mg), ("baseline2", mb2)):
    print("%s: ttft %.3fs  prefill %.1f tok/s  decode %.1f tok/s" % (
        label, m["ttft_s_median"], m["prefill_tok_s_median"], m["decode_tok_s_median"]))
print("machine drift (A vs A): prefill %+.1f%%  decode %+.1f%%" % (
    drift["prefill_tok_s_pct"], drift["decode_tok_s_pct"]))
print("graft delta vs baseline : prefill %+.1f%%  decode %+.1f%%" % (
    delta_ab["prefill_tok_s_pct"], delta_ab["decode_tok_s_pct"]))
print("graft delta vs baseline2: prefill %+.1f%%  decode %+.1f%%" % (
    delta_ba["prefill_tok_s_pct"], delta_ba["decode_tok_s_pct"]))
PY
