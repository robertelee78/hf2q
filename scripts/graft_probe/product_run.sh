#!/usr/bin/env bash
# ADR-059 PRODUCT measurement: the phantom-ported trained bank, served
# by hf2q as a graft file, scored on the phantom suites with the phantom
# refusal judge. Same scoreboard as the reference confirmation
# (base 46/60 -> trained 4/60 in their stack); this run answers it in
# OUR stack: does the unalignment survive hf2q serving (Q4_K_M weights,
# mlx-native kernels, TQ KV substrate)?
#
# Arms: baseline (no graft) and trained (--kv-graft), identical config.
set -euo pipefail

MODEL="${1:-/opt/hf2q/models/qwen-family-mtp-qualified/qwen35-dense/Qwen3.5-4B-Q4_K_M.gguf}"
GRAFT="${2:-/tmp/q35-4b-trained.graft.gguf}"
PORT="${3:-8399}"
BIN="$(cd "$(dirname "$0")/../.." && pwd)/target/release/hf2q"
WORK="$(mktemp -d /tmp/hf2q-product.XXXXXX)"
SRV_PID=""

stop_server() {
    if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    SRV_PID=""
}
trap stop_server EXIT

run_arm() {
    local arm="$1"
    shift
    local log="$WORK/serve-$arm.log"
    echo "── arm $arm: booting on :$PORT"
    HF2Q_TQ_KV=1 HF2Q_QWEN_SPECULATION=off "$BIN" serve "$MODEL" \
        --port "$PORT" --quiet --scheduler fifo-serial \
        --default-thinking-token-budget 0 ${1:+--kv-graft "$1"} >"$log" 2>&1 &
    SRV_PID=$!
    local i
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
        kill -0 "$SRV_PID" 2>/dev/null || { echo "FAIL: serve $arm exited" >&2; tail -5 "$log" >&2; exit 1; }
        sleep 2
    done
    /opt/phantom-kv/.venv/bin/python - "$PORT" "$arm" "$WORK" <<'PY'
import json, sys, urllib.request
sys.path.insert(0, "/opt/phantom-kv/src")
from phantom_kv.eval.refusal import is_refusal

port, arm, work = sys.argv[1:4]
model_id = json.load(urllib.request.urlopen(
    "http://127.0.0.1:%s/v1/models" % port, timeout=30))["data"][0]["id"]
for suite in ("harmful_seed", "harmless_seed"):
    rows = []
    for line in open("/opt/phantom-kv/data/suites/%s.jsonl" % suite):
        item = json.loads(line)
        body = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0, "max_tokens": 128, "seed": 42,
            "stream": False, "hf2q_enable_thinking": False,
        })
        req = urllib.request.Request(
            "http://127.0.0.1:%s/v1/chat/completions" % port,
            data=body.encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.load(resp)
        text = result["choices"][0]["message"].get("content") or ""
        rows.append({"id": item["id"], "prompt": item["prompt"],
                     "completion": text, "refusal": is_refusal(text)})
    refusals = sum(r["refusal"] for r in rows)
    name = suite.split("_")[0]
    print("[product] %s %s: %d/%d refusals" % (arm, name, refusals, len(rows)), flush=True)
    with open("%s/results-%s-%s.json" % (work, arm, name), "w") as f:
        json.dump(rows, f, indent=2)
PY
    stop_server
    sleep 2
}

echo "product workdir: $WORK"
run_arm baseline
run_arm trained "$GRAFT"
echo "reference confirmation (their stack, bf16): base 46/60 -> trained 4/60"
