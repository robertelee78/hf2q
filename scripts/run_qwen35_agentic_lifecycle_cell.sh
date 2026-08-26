#!/usr/bin/env bash
set -euo pipefail

# Fresh-process exact-artifact lifecycle authority for one qwen35-family
# architecture shape. This is deliberately separate from the timed ABBA arm:
# cancellation and OpenAI/tool semantics are correctness gates, not timing
# samples that may perturb the performance process.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=scripts/agentic_cache_lifecycle_contract.sh
source "$script_dir/agentic_cache_lifecycle_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

SOURCE_ROOT=${SOURCE_ROOT:-$root_dir}
HF2Q_BIN=${HF2Q_BIN:-$SOURCE_ROOT/target/release/hf2q}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
MODEL_BYTES=${MODEL_BYTES:?MODEL_BYTES is required}
MODEL_SHAPE=${MODEL_SHAPE:?MODEL_SHAPE must be qwen38-dense or qwen36-moe}
Q5K_CANONICAL_Q4X4=${HF2Q_Q5K_CANONICAL_Q4X4:-1}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
PORT=${PORT:-52851}
readonly MAX_SLOTS=4
readonly CONTEXT_LINES=2800
readonly KV_CACHE_BUDGET_BYTES=51539607552
readonly RUNTIME_HOME=/var/empty
readonly RUNTIME_PATH=/usr/bin:/bin:/usr/sbin:/sbin
readonly RUNTIME_TMPDIR=/var/tmp

for command in awk curl env find git jq lsof mkdir mv pgrep ps rg shasum stat; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ "$SOURCE_ROOT" == /* && "$HF2Q_BIN" == /* && "$MODEL_PATH" == /* \
    && "$OUT_DIR" == /* ]] || {
    echo "source, binary, model, and output paths must be absolute" >&2
    exit 2
}
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd -P)
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] \
    || { echo "lifecycle cell requires clean source" >&2; exit 2; }
source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
[[ "$HF2Q_BIN" == "$SOURCE_ROOT/target/release/hf2q" && -x "$HF2Q_BIN" \
    && "$source_commit" =~ ^[0-9a-f]{40}$ ]] || {
    echo "lifecycle cell requires the exact worktree release binary" >&2
    exit 2
}
grep -aFq "$source_commit" "$HF2Q_BIN" || {
    echo "release binary does not embed source commit $source_commit" >&2
    exit 2
}
binary_sha256=$(shasum -a 256 "$HF2Q_BIN" | awk '{print $1}')
[[ -f "$MODEL_PATH" && -r "$MODEL_PATH" && ! -L "$MODEL_PATH" \
    && "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$MODEL_BYTES" =~ ^[1-9][0-9]*$ \
    && "$(stat -f '%z' "$MODEL_PATH" 2>/dev/null || stat -c '%s' "$MODEL_PATH")" == "$MODEL_BYTES" ]] \
    || { echo "invalid exact model contract" >&2; exit 2; }
case "$MODEL_SHAPE" in
    qwen38-dense) expected_arch=qwen35 ;;
    qwen36-moe) expected_arch=qwen35moe ;;
    *) echo "invalid MODEL_SHAPE: $MODEL_SHAPE" >&2; exit 2 ;;
esac
[[ "$Q5K_CANONICAL_Q4X4" == 0 || "$Q5K_CANONICAL_Q4X4" == 1 ]] \
    || { echo "HF2Q_Q5K_CANONICAL_Q4X4 must be 0 or 1" >&2; exit 2; }
if ! [[ "$PORT" =~ ^[1-9][0-9]*$ ]] || ((PORT > 65535)); then
    echo "PORT must be in 1..65535" >&2
    exit 2
fi
[[ ! -e "$OUT_DIR" \
    || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] \
    || { echo "OUT_DIR must be fresh" >&2; exit 2; }
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd -P)
case "$OUT_DIR/" in
    "$SOURCE_ROOT"/*)
        echo "OUT_DIR must be outside SOURCE_ROOT" >&2
        exit 2
        ;;
esac
mkdir -p "$OUT_DIR/runtime-cache"
[[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" \
    && -z "$(pgrep -x hf2q 2>/dev/null || true)" ]] || {
    echo "lifecycle cell requires no existing hf2q runtime/listener" >&2
    exit 2
}

server_pid=''
power_guard_started=false
manifest_tmp=''
cleanup() {
    local cleanup_rc=0
    if [[ -n "$server_pid" ]]; then
        kill -INT "$server_pid" 2>/dev/null || true
        for ((waited = 0; waited < 60; waited++)); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -TERM "$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
        server_pid=''
    fi
    if [[ "$power_guard_started" == true ]]; then
        qwen36_stop_power_guard || cleanup_rc=1
        power_guard_started=false
    fi
    [[ -z "$manifest_tmp" ]] || rm -f "$manifest_tmp"
    return "$cleanup_rc"
}
on_exit() {
    local original_rc=$? cleanup_rc=0
    trap - EXIT
    cleanup || cleanup_rc=1
    if ((original_rc == 0 && cleanup_rc != 0)); then exit "$cleanup_rc"; fi
    exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

qwen36_start_power_guard "$$" "$OUT_DIR/caffeinate.log"
power_guard_started=true
export HF2Q_MODEL_VERIFICATION_BINARY="$HF2Q_BIN"
hf2q_release_record_model_verification \
    "$MODEL_PATH" "$MODEL_SHA256" "$OUT_DIR/model-verification.json"
hf2q_release_verify_model \
    "$MODEL_PATH" "$MODEL_SHA256" "$OUT_DIR/model-verification.json"

env -i HOME="$RUNTIME_HOME" PATH="$RUNTIME_PATH" TMPDIR="$RUNTIME_TMPDIR" \
    LANG=C LC_ALL=C USER=hf2q-gate LOGNAME=hf2q-gate RUST_BACKTRACE=1 \
    HF2Q_MODEL_VERIFICATION_RECEIPT="$OUT_DIR/model-verification.json" \
    HF2Q_CROSS_SLOT_ADMIT=1 HF2Q_ADMIT_COALESCE_US=25000 \
    HF2Q_QWEN_SPECULATION=auto HF2Q_TQ_KV=1 HF2Q_ENCODER_SESSION=1 \
    HF2Q_FFN_TERMINAL_K_BATCH=8 \
    HF2Q_Q5K_CANONICAL_Q4X4="$Q5K_CANONICAL_Q4X4" \
    "$HF2Q_BIN" -v serve --model "$MODEL_PATH" \
    --cache-dir "$OUT_DIR/runtime-cache" \
    --host 127.0.0.1 --port "$PORT" --scheduler inflight-batched \
    --max-slots "$MAX_SLOTS" --overflow-policy reject \
    --kv-cache-budget "$KV_CACHE_BUDGET_BYTES" \
    --default-repetition-penalty 1 --default-thinking-token-budget 0 \
    --default-tool-thinking-token-budget 0 \
    >"$OUT_DIR/server.stdout" 2>"$OUT_DIR/server.stderr" &
server_pid=$!
for ((second = 0; second < 360; second++)); do
    curl --fail --silent "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 && break
    kill -0 "$server_pid" 2>/dev/null || {
        tail -n 120 "$OUT_DIR/server.stderr" >&2
        exit 1
    }
    sleep 1
done
curl --fail --silent "http://127.0.0.1:$PORT/readyz" >/dev/null \
    || { echo "lifecycle server did not become ready" >&2; exit 1; }
qwen36_assert_power_guard
qwen36_bind_server_process "http://127.0.0.1:$PORT" "$server_pid" \
    "$HF2Q_BIN" "$MODEL_PATH" "$MAX_SLOTS"
ps -ww -p "$server_pid" -o command= >"$OUT_DIR/server-command.txt"
curl --fail --silent "http://127.0.0.1:$PORT/v1/models" >"$OUT_DIR/models.json"
model_id=$(jq -er --arg arch "$expected_arch" '
  [.data[] | select(.loaded == true and .arch == $arch)]
  | if length == 1 then .[0].id else error("exact loaded architecture absent") end
' "$OUT_DIR/models.json")
run_id="rectangular-${source_commit:0:12}-${MODEL_SHAPE}"
env -i HOME="$RUNTIME_HOME" PATH="$RUNTIME_PATH" TMPDIR="$RUNTIME_TMPDIR" \
LANG=C LC_ALL=C USER=hf2q-gate LOGNAME=hf2q-gate \
BASE_URL="http://127.0.0.1:$PORT" MODEL="$model_id" \
OUT_DIR="$OUT_DIR/lifecycle" RUN_ID="$run_id" CONTEXT_LINES="$CONTEXT_LINES" \
ACTIVE_MAX_TOKENS=2048 CURL_CONNECT_TIMEOUT_SECONDS=5 \
CONTINUATION_THINKING_TOKEN_BUDGET=16 \
ISOLATION_THINKING_DISABLED=true \
CURL_MAX_TIME_SECONDS=1800 SEMANTIC_WAIT_SECONDS=300 SIBLING_SETTLE_SECONDS=1 \
EXPECTED_EXECUTION_ARTIFACT_SHA256="$MODEL_SHA256" \
EXPECTED_EXECUTION_ARCH_FAMILY=qwen35 \
EXPECTED_EXECUTION_ARCHITECTURE="$expected_arch" \
    "$script_dir/test_agentic_cache_lifecycle.sh" \
    >"$OUT_DIR/lifecycle.stdout" 2>"$OUT_DIR/lifecycle.stderr"
agentic_lifecycle_validate_summary "$OUT_DIR/lifecycle/summary.json" \
    "$run_id" "$CONTEXT_LINES" "$MODEL_SHA256" qwen35 "$expected_arch" 16 false
curl --fail --silent "http://127.0.0.1:$PORT/readyz" >"$OUT_DIR/readyz.json"
qwen36_assert_power_guard
qwen36_reject_fatal_log "$OUT_DIR/server.stderr"
expected_q5k_policy=false
[[ "$Q5K_CANONICAL_Q4X4" == 1 ]] && expected_q5k_policy=true
EXPECTED_Q5K_POLICY="$expected_q5k_policy" perl -ne '
  if (/frozen Qwen GGML routing policy/) {
    $seen++;
    $q5=$1 if /dense_q5k_canonical_q4x4=(true|false)/;
  }
  END {exit 1 unless $seen == 1 && $q5 eq $ENV{EXPECTED_Q5K_POLICY}}
' "$OUT_DIR/server.stderr" || {
    echo "lifecycle server did not freeze the requested Q5_K route" >&2
    exit 1
}
cleanup
[[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]] \
    || { echo "lifecycle server listener survived shutdown" >&2; exit 1; }
qwen36_reject_fatal_log "$OUT_DIR/server.stderr"
hf2q_release_verify_model \
    "$MODEL_PATH" "$MODEL_SHA256" "$OUT_DIR/model-verification.json"

manifest_tmp=$(mktemp "${TMPDIR:-/var/tmp}/qwen-lifecycle-manifest.XXXXXX")
(
    cd "$OUT_DIR"
    find . -type f ! -name receipt.json ! -name evidence.sha256 \
        ! -name evidence.sha256.tmp -print \
      | sed 's#^./##' | sort | while IFS= read -r relative; do
          printf '%s  %s\n' "$(shasum -a 256 "$relative" | awk '{print $1}')" \
              "$relative"
        done >"$manifest_tmp"
    mv "$manifest_tmp" evidence.sha256
    shasum -a 256 -c evidence.sha256 >/dev/null
)
manifest_tmp=''
manifest_sha=$(shasum -a 256 "$OUT_DIR/evidence.sha256" | awk '{print $1}')
summary_sha=$(shasum -a 256 "$OUT_DIR/lifecycle/summary.json" | awk '{print $1}')
jq -n --arg source_commit "$source_commit" --arg binary "$HF2Q_BIN" \
    --arg binary_sha256 "$binary_sha256" --arg model_shape "$MODEL_SHAPE" \
    --arg model_path "$MODEL_PATH" --arg model_sha256 "$MODEL_SHA256" \
    --argjson model_bytes "$MODEL_BYTES" --arg architecture "$expected_arch" \
    --arg run_id "$run_id" --arg manifest_sha "$manifest_sha" \
    --arg summary_sha "$summary_sha" \
    --argjson q5k_canonical_q4x4 "$Q5K_CANONICAL_Q4X4" '{
      schema:2,verdict:"pass",gate:"qwen35-agentic-lifecycle-cell",
      source:{commit:$source_commit,binary:$binary,sha256:$binary_sha256},
      model:{shape:$model_shape,path:$model_path,sha256:$model_sha256,
        bytes:$model_bytes,arch_family:"qwen35",architecture:$architecture},
      runtime:{clean_environment:true,max_slots:4,scheduler:"inflight-batched",
        speculation:"auto",kv_cache_budget_bytes:51539607552,kv_persist:false,
        cache_dir:"evidence-local",
        routing:{dense_q5k_canonical_q4x4:$q5k_canonical_q4x4}},
      lifecycle:{run_id:$run_id,context_lines:2800,
        continuation_thinking_token_budget:16,
        unrelated_conversation_thinking_enabled:false,
        summary_sha256:$summary_sha},
      evidence:{manifest_sha256:$manifest_sha}
    }' >"$OUT_DIR/receipt.json.tmp"
mv "$OUT_DIR/receipt.json.tmp" "$OUT_DIR/receipt.json"
"$script_dir/verify_qwen35_agentic_lifecycle_cell.sh" \
    "$OUT_DIR/receipt.json" "$SOURCE_ROOT"
jq . "$OUT_DIR/receipt.json"
