#!/usr/bin/env bash
# ADR-059 hardware canary — gates 4 and 5 on the serial unary path.
#
# Paired arms, identical model artifact / prompt / sampling / budget,
# one full-model runtime at a time (fresh serve process per arm):
#
#   A baseline      — no graft
#   B zero-slot     — --kv-graft zero-slot canary (n_slots=0, no tensors)
#   C live          — --kv-graft live bank (64 slots, every full-attn layer)
#   D disable-fresh — no graft, fresh process (baseline restore)
#
# Verdicts:
#   gate 4a (plumbing no-op):   B output == A output
#   gate 4b (splice live):      C output != A output
#   gate 5  (disable restores): D output == A output
#
# Configuration matching across arms (measurement doctrine: the ONLY
# difference between arms is the graft):
#
#   HF2Q_TQ_KV=$TQ_KV               — one campaign per KV substrate: the
#                                     F32 control path (0) and the
#                                     production TQ path (1, default
#                                     serving substrate; the graft splice
#                                     encodes bank rows through the same
#                                     hadamard kernel prefill uses).
#                                     HF2Q_CANARY_TQ_KV selects (default
#                                     0). Arms never mix substrates.
#   --scheduler $SCHEDULER          — identical engine path per campaign
#                                     (SlotAware vs Serial batching
#                                     changes greedy numerics, so arms
#                                     never mix schedulers). Both engine
#                                     paths are graft-wired: run one
#                                     campaign per scheduler.
#                                     HF2Q_CANARY_SCHEDULER selects:
#                                     fifo-serial (default) or
#                                     inflight-batched (SlotAware).
#   HF2Q_QWEN_SPECULATION=off       — a bound graft skips native MTP;
#                                     disabling it everywhere keeps the
#                                     decode path identical.
#   --default-thinking-token-budget 0 — zero disables the default budget
#                                     (thinking stays on, unbudgeted)
#                                     identically across arms.
#
# Usage:
#   bash scripts/graft_probe/canary.sh [MODEL_GGUF] [PORT]
#   HF2Q_CANARY_SCHEDULER=inflight-batched bash scripts/graft_probe/canary.sh
set -euo pipefail

MODEL="${1:-/opt/hf2q/models/qwen3.6/APEX-Q5_K_M.gguf}"
PORT="${2:-8391}"
SCHEDULER="${HF2Q_CANARY_SCHEDULER:-fifo-serial}"
TQ_KV="${HF2Q_CANARY_TQ_KV:-0}"
BIN="$(cd "$(dirname "$0")/../.." && pwd)/target/release/hf2q"
WORK="$(mktemp -d /tmp/hf2q-graft-canary.XXXXXX)"
PROMPT='In one sentence: what is the capital of France?'
MAX_TOKENS=48
SEED=42
CANARY_PID=""

stop_server() {
    if [[ -n "$CANARY_PID" ]] && kill -0 "$CANARY_PID" 2>/dev/null; then
        kill "$CANARY_PID" 2>/dev/null || true
        wait "$CANARY_PID" 2>/dev/null || true
    fi
    CANARY_PID=""
}
trap stop_server EXIT

echo "canary workdir: $WORK"
# Dose rationale: the canary proves PARTICIPATION (the splice is read by
# attention and shifts the forward pass), not behavioral quality — that
# is the paired-arm panel's job per the measurement doctrine. 64 slots
# at scale 8.0 is the measured guaranteed-shift dose on Qwen3.6-35B
# (8/0.5 left greedy argmax unchanged; 64/8.0 degenerates the stream —
# an unambiguous, deterministic participation signal).
python3 "$(dirname "$0")/build_canary_graft.py" \
    --model "$MODEL" \
    --out-zero "$WORK/graft-zero.gguf" \
    --out-live "$WORK/graft-live.gguf" \
    --n-slots 64 --scale 8.0

sha256sum "$WORK/graft-zero.gguf" "$WORK/graft-live.gguf" | tee "$WORK/artifact-hashes.txt"

request_body() {
    python3 - "$MODEL_ID" "$PROMPT" "$MAX_TOKENS" "$SEED" <<'PY'
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": sys.argv[2]}],
    "temperature": 0,
    "max_tokens": int(sys.argv[3]),
    "seed": int(sys.argv[4]),
    "stream": False,
}))
PY
}

run_arm() {
    local name="$1"
    shift
    local log="$WORK/serve-$name.log"
    echo "── arm $name: booting serve on :$PORT"
    HF2Q_TQ_KV="$TQ_KV" HF2Q_QWEN_SPECULATION=off "$BIN" serve "$MODEL" \
        --port "$PORT" --quiet --scheduler "$SCHEDULER" \
        --default-thinking-token-budget 0 "$@" >"$log" 2>&1 &
    CANARY_PID=$!

    local ready=0
    local i
    for i in $(seq 1 240); do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$CANARY_PID" 2>/dev/null; then
            echo "FAIL: serve ($name) exited during startup"
            tail -20 "$log"
            exit 1
        fi
        sleep 2
    done
    if [[ "$ready" != "1" ]]; then
        echo "FAIL: serve ($name) never became ready"
        tail -20 "$log"
        exit 1
    fi

    # Resolve the served model id from /v1/models (an unknown `model`
    # name in the request is a 400; the canary must address the model
    # the server actually loaded).
    MODEL_ID="$(curl -sf --max-time 5 "http://127.0.0.1:$PORT/v1/models" | \
        python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
    echo "   arm $name: model id $MODEL_ID"

    curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(request_body)" \
        > "$WORK/response-$name.json"

    stop_server

    python3 - "$WORK/response-$name.json" "$WORK/content-$name.txt" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    body = json.load(f)
message = body["choices"][0]["message"]
# The full deterministic observable under greedy decoding: content AND
# reasoning (unbudgeted thinking can spend the whole budget in
# reasoning_content, leaving content empty — comparing content alone
# would be vacuous).
content = (message.get("content") or "") + "\n--reasoning--\n" + \
    (message.get("reasoning_content") or "")
finish = body["choices"][0].get("finish_reason")
usage = body.get("usage", {})
with open(sys.argv[2], "w") as f:
    f.write(content)
print("   arm content (%s, completion=%s): %.80r" % (
    finish, usage.get("completion_tokens"), content))
PY
}

run_arm baseline
run_arm zero --kv-graft "$WORK/graft-zero.gguf"
run_arm live --kv-graft "$WORK/graft-live.gguf"
run_arm disable

A=$(cat "$WORK/content-baseline.txt")
B=$(cat "$WORK/content-zero.txt")
C=$(cat "$WORK/content-live.txt")
D=$(cat "$WORK/content-disable.txt")

verdict_zero=$([[ "$A" == "$B" ]] && echo PASS || echo FAIL)
verdict_live=$([[ "$A" != "$C" ]] && echo PASS || echo FAIL)
verdict_disable=$([[ "$A" == "$D" ]] && echo PASS || echo FAIL)

git_rev="$(git -C "$(dirname "$0")/../.." rev-parse HEAD 2>/dev/null || echo unknown)"
python3 - "$WORK" "$git_rev" "$MODEL" "$SCHEDULER" "$TQ_KV" "$verdict_zero" "$verdict_live" "$verdict_disable" <<'PY'
import hashlib, json, os, sys
work, git_rev, model, scheduler, tq_kv, v_zero, v_live, v_disable = sys.argv[1:9]
def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
manifest = {
    "canary": "adr-059-gates-4-5",
    "git_rev": git_rev,
    "scheduler": scheduler,
    "tq_kv": tq_kv,
    "model": model,
    "model_sha256": sha(model) if os.path.exists(model) else None,
    "artifacts": {
        "zero_slot": sha(os.path.join(work, "graft-zero.gguf")),
        "live": sha(os.path.join(work, "graft-live.gguf")),
    },
    "arms": {
        name: open(os.path.join(work, "content-%s.txt" % name)).read()
        for name in ("baseline", "zero", "live", "disable")
    },
    "verdicts": {
        "gate4a_zero_slot_no_op": v_zero,
        "gate4b_live_graft_shifts_output": v_live,
        "gate5_disable_restores_baseline": v_disable,
    },
}
path = os.path.join(work, "canary-manifest.json")
with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
print("manifest: %s" % path)
PY

echo
echo "scheduler: $SCHEDULER  tq_kv: $TQ_KV"
echo "gate 4a (zero-slot no-op):     $verdict_zero"
echo "gate 4b (live graft shifts):   $verdict_live"
echo "gate 5  (disable restores):    $verdict_disable"
if [[ "$verdict_zero" == "PASS" && "$verdict_live" == "PASS" && "$verdict_disable" == "PASS" ]]; then
    echo "CANARY: ALL PASS"
    exit 0
fi
echo "CANARY: FAILURE"
exit 1
