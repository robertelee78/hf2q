#!/bin/bash
# #310 requalification judge driver: boot ONE judge server, judge every
# supplied results file (4-way concurrent pool — independent processes,
# distinct outputs, merged after; the grammar_probe campaign precedent),
# stop the server. One model runtime at a time.
#
# Usage: requal_judge.sh <judge-model-ref> <judge-label> <port> \
#            <results.jsonl> <verdicts.jsonl> [more results/verdicts pairs]
# Env:   JUDGE_POOL (default 4), JUDGE_MAX_TOKENS (default 600)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
JUDGE_REF="${1:?judge model ref}"
JUDGE_LABEL="${2:?judge label}"
PORT="${3:?port}"
shift 3
[[ $# -ge 2 && $(( $# % 2 )) -eq 0 ]] || { echo "need results/verdicts pairs" >&2; exit 2; }

POOL="${JUDGE_POOL:-4}"
# The current release binary: the primary checkout's build (a worktree
# run has no target build of its own; HF2Q_BIN overrides).
BIN="${HF2Q_BIN:-$(cd "$HERE/../.." && pwd)/target/release/hf2q}"
WORK="$(mktemp -d /tmp/requal-judge.XXXXXX)"
SRV_PID=""

stop_server() {
    if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    SRV_PID=""
}
trap stop_server EXIT

HF2Q_TQ_KV=1 HF2Q_QWEN_SPECULATION=off \
    "$BIN" serve "$JUDGE_REF" \
    --port "$PORT" --quiet --default-thinking-token-budget 0 \
    >"$WORK/judge-serve.log" 2>&1 &
SRV_PID=$!

for i in $(seq 1 2400); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SRV_PID" 2>/dev/null || {
        echo "FAIL: judge server exited" >&2; tail -5 "$WORK/judge-serve.log" >&2; exit 1; }
    sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
    echo "FAIL: judge server never became ready" >&2; exit 1; }
MODEL_ID="$(curl -sf "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "[judge] $JUDGE_LABEL ready on :$PORT (model id: $MODEL_ID)"

judge_one() {
    local results="$1" verdicts="$2"
    rm -rf "$WORK/chunks"; mkdir -p "$WORK/chunks"
    # Skip already-judged prompt ids (resume support): split only the
    # rows this judge model has not scored yet.
    python3 - "$results" "$verdicts" "$WORK/chunks" "$POOL" "$MODEL_ID" <<'PY'
import json, os, sys
results, verdicts, chunks, pool, model_id = sys.argv[1:6]
done = set()
if os.path.exists(verdicts):
    for line in open(verdicts):
        try:
            row = json.loads(line)
            if row["judge_model"] == model_id:
                done.add(row["prompt_id"])
        except (json.JSONDecodeError, KeyError):
            continue
rows = []
for line in open(results):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "content" not in row or not row.get("content"):
        continue  # empty rows are never judged and never count as compliance
    rows.append(row)
todo = [r for r in rows if r["prompt_id"] not in done]
n = int(pool)
files = [open(os.path.join(chunks, "chunk-%d.jsonl" % i), "w") for i in range(n)]
for i, row in enumerate(todo):
    files[i % n].write(json.dumps(row) + "\n")
for f in files:
    f.close()
print("[judge] %d todo / %d rows (%d already judged)" % (len(todo), len(rows), len(rows) - len(todo)))
PY
    pids=()
    for chunk in "$WORK"/chunks/chunk-*.jsonl; do
        [[ -s "$chunk" ]] || continue
        RESULTS="$chunk" \
        OUT="$chunk.verdicts" \
        SUITE="$HERE/probes_harmful_all.jsonl" \
        BASE_URL="http://127.0.0.1:$PORT" \
        JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-600}" \
        python3 "$HERE/judge_strongreject.py" >"$chunk.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    [[ $fail -eq 0 ]] || { echo "FAIL: a judge worker failed" >&2; exit 1; }
    touch "$verdicts"
    cat "$WORK"/chunks/chunk-*.jsonl.verdicts >>"$verdicts"
    n=$(grep -c . "$verdicts")
    echo "[judge] $JUDGE_LABEL: $n verdict rows -> $verdicts"
}

while [[ $# -ge 2 ]]; do
    judge_one "$1" "$2"
    shift 2
done

stop_server
echo "[judge] $JUDGE_LABEL COMPLETE (server stopped)"
