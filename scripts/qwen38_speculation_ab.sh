#!/usr/bin/env bash
set -euo pipefail

# Reproducible exact-output OFF/AUTO gate for the canonical Qwen3.8
# SlotAware server. This is a focused speculation receipt, not the calibrated
# long-context or release-artifact gate.

BINARY_PATH=${BINARY_PATH:-/opt/hf2q/target/release/hf2q}
MODEL_PATH=${MODEL_PATH:-/opt/hf2q/models/qwen3.8/Qwen3.8-27B-Abliterated-SFT-Q4_K_M.gguf}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
PORT=${PORT:-18084}
MIN_CODE_IMPROVEMENT_PERCENT=${MIN_CODE_IMPROVEMENT_PERCENT:-5}
MIN_REPEAT_IMPROVEMENT_PERCENT=${MIN_REPEAT_IMPROVEMENT_PERCENT:-5}
DECODE_MVN=${HF2Q_DECODE_MVN:-0}
DECODE_MV_EXT=${HF2Q_DECODE_MV_EXT:-1}

MODEL_ID=${MODEL_ID:-}
readonly MAX_TOKENS=128
readonly CASES='code-a code-b code-c repeat-a repeat-b repeat-c'
readonly TRIAL_ORDER='off auto auto off'

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in awk cmp curl find jq sed shasum sort stat; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ -x "$BINARY_PATH" ]] || {
    echo "hf2q binary is missing or non-executable: $BINARY_PATH" >&2
    exit 2
}
[[ -f "$MODEL_PATH" ]] || {
    echo "Qwen3.8 model is missing: $MODEL_PATH" >&2
    exit 2
}
[[ "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "MODEL_SHA256 must be a lowercase 64-character digest" >&2
    exit 2
}
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
    echo "PORT must be an integer from 1 through 65535" >&2
    exit 2
fi
for threshold in "$MIN_CODE_IMPROVEMENT_PERCENT" "$MIN_REPEAT_IMPROVEMENT_PERCENT"; do
    awk -v value="$threshold" 'BEGIN { exit !(value >= 0) }' || {
        echo "improvement thresholds must be non-negative numbers" >&2
        exit 2
    }
done
case "$DECODE_MVN:$DECODE_MV_EXT" in
    0:1|0:0|1:0|1:1) ;;
    *)
        echo "HF2Q_DECODE_MVN and HF2Q_DECODE_MV_EXT must each be 0 or 1" >&2
        exit 2
        ;;
esac
if [[ -e "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Qwen3.8 speculation receipt directory must be fresh: $OUT_DIR" >&2
    exit 2
fi
mkdir -p "$OUT_DIR/requests"

model_verification_receipt=${HF2Q_MODEL_VERIFICATION_RECEIPT:-}
if [[ -z "$model_verification_receipt" ]]; then
    if [[ -n ${HF2Q_MODEL_VERIFICATION_CACHE_DIR:-} ]]; then
        model_verification_cache_dir=$HF2Q_MODEL_VERIFICATION_CACHE_DIR
    elif [[ -n ${XDG_CACHE_HOME:-} ]]; then
        model_verification_cache_dir="$XDG_CACHE_HOME/hf2q/model-verification"
    else
        model_verification_cache_dir="${HOME:?HOME is required when XDG_CACHE_HOME is unset}/.cache/hf2q/model-verification"
    fi
    model_verification_receipt="$OUT_DIR/model-verification.json"
    hf2q_release_prepare_model_verification \
        "$MODEL_PATH" "$MODEL_SHA256" "$model_verification_receipt" \
        "$model_verification_cache_dir"
    model_verification_mode=$(jq -er .run_verification \
        "$model_verification_receipt")
else
    model_verification_mode=provided_receipt
fi
hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
    "$model_verification_receipt"
model_file_snapshot=$(jq -er .file_snapshot "$model_verification_receipt")

sha256_file() {
    shasum -a 256 "$1" | awk '{print $1}'
}

file_bytes() {
    stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1" 2>/dev/null
}

write_request() {
    local name=$1
    local system_prompt=$2
    local user_prompt=$3
    jq -n \
        --arg model "$MODEL_ID" \
        --arg system "$system_prompt" \
        --arg user "$user_prompt" \
        --argjson max_tokens "$MAX_TOKENS" \
        '{model:$model,messages:[
            {role:"system",content:$system},
            {role:"user",content:$user}
          ],max_tokens:$max_tokens,temperature:0,stream:false,
          hf2q_enable_thinking:false}' \
        >"$OUT_DIR/requests/$name.json.tmp"
    mv "$OUT_DIR/requests/$name.json.tmp" "$OUT_DIR/requests/$name.json"
}

write_request code-a \
    'You are a precise coding assistant. Return only the requested answer.' \
    'Write a complete Rust function fn fibonacci(n: u64) -> u64 using an iterative algorithm. Include a short explanation and one unit test. Benchmark marker A; do not mention the marker.'
write_request code-b \
    'You are a precise coding assistant. Return only the requested answer.' \
    'Write a complete Rust function fn binary_search(xs: &[i32], needle: i32) -> Option<usize> using an iterative algorithm. Include a short explanation and one unit test. Benchmark marker B; do not mention the marker.'
write_request code-c \
    'You are a precise coding assistant. Return only the requested answer.' \
    'Write a complete Rust function fn gcd(mut a: u64, mut b: u64) -> u64 using the iterative Euclidean algorithm. Include a short explanation and one unit test. Benchmark marker C; do not mention the marker.'
write_request repeat-a \
    'You are a transcription engine. Follow the request exactly.' \
    $'Repeat the following text exactly, with no introduction or quotation marks:\n\nThe copper observatory stood above the harbor while seven quiet instruments recorded wind, tide, temperature, pressure, cloud cover, rainfall, and the slow vibration of the old bridge. Each evening the keeper copied those readings into a blue ledger, checked every column twice, and left the completed page beneath a brass lamp for the morning crew.'
write_request repeat-b \
    'You are a transcription engine. Follow the request exactly.' \
    $'Repeat the following text exactly, with no introduction or quotation marks:\n\nA patient compiler reads the module, resolves every import, expands each macro, verifies every lifetime, checks every trait bound, lowers the typed program into an intermediate form, applies conservative optimizations, and finally emits a deterministic object file. The build report records each phase so that a later engineer can reproduce the result.'
write_request repeat-c \
    'You are a transcription engine. Follow the request exactly.' \
    $'Repeat the following text exactly, with no introduction or quotation marks:\n\nAt sunrise the research vessel crossed the calm channel, passed three red buoys, and turned north toward the ice station. The crew calibrated the sonar array, sealed the sample containers, reviewed the emergency checklist, and logged the exact coordinates before lowering the first instrument through the open deck.'

jq -n \
    --arg model "$MODEL_ID" \
    '{model:$model,messages:[
        {role:"system",content:"This request is benchmark warmup only."},
        {role:"user",content:"Return exactly WARMUP."}
      ],max_tokens:8,temperature:0,stream:false,hf2q_enable_thinking:false}' \
    >"$OUT_DIR/requests/warmup.json.tmp"
mv "$OUT_DIR/requests/warmup.json.tmp" "$OUT_DIR/requests/warmup.json"

server_pid=''
stop_server() {
    local waited=0
    [[ -n "$server_pid" ]] || return 0
    if kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid" 2>/dev/null || true
        while kill -0 "$server_pid" 2>/dev/null && ((waited < 30)); do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    if kill -0 "$server_pid" 2>/dev/null; then
        echo "Qwen3.8 speculation trial ignored bounded shutdown" >&2
        kill -TERM "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
    server_pid=''
}

on_exit() {
    local original_rc=$?
    trap - EXIT
    stop_server || true
    exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

wait_ready() {
    local log_path=$1
    local deadline=$((SECONDS + 600))
    while ((SECONDS < deadline)); do
        if curl --fail --silent --show-error --max-time 2 \
            "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Qwen3.8 speculation trial exited before readiness" >&2
            sed -n '1,240p' "$log_path" >&2
            return 1
        fi
        sleep 1
    done
    echo "Qwen3.8 speculation trial did not become ready" >&2
    sed -n '1,240p' "$log_path" >&2
    return 1
}

resolve_loaded_model_id() {
    local loaded_model_id
    local request
    loaded_model_id=$(curl --fail --silent --show-error \
        "http://127.0.0.1:$PORT/v1/models" | jq -er '
          [.data[] | select(.loaded == true)]
          | if length == 1 then .[0].id
            else error("expected exactly one loaded model") end
        ')
    if [[ -z "$MODEL_ID" ]]; then
        MODEL_ID=$loaded_model_id
        for request in "$OUT_DIR"/requests/*.json; do
            jq --arg model "$MODEL_ID" '.model = $model' "$request" \
                >"$request.tmp"
            mv "$request.tmp" "$request"
        done
    elif [[ "$loaded_model_id" != "$MODEL_ID" ]]; then
        echo "Qwen3.8 loaded model identity drifted: expected=$MODEL_ID actual=$loaded_model_id" >&2
        return 1
    fi
}

run_trial() {
    local trial_index=$1
    local mode=$2
    local mode_dir="$OUT_DIR/trial-$trial_index-$mode"
    local log_path="$mode_dir/server.log"
    mkdir -p "$mode_dir"

    env \
        HF2Q_BIN="$BINARY_PATH" \
        MODEL="$MODEL_PATH" \
        PORT="$PORT" \
        MAX_SLOTS=1 \
        KV_CACHE_BUDGET_BYTES=51539607552 \
        QWEN38_VISION=off \
        QWEN38_SPECULATION="$mode" \
        THINKING_TOKEN_BUDGET=0 \
        TOOL_THINKING_TOKEN_BUDGET=0 \
        REP_PENALTY=1.05 \
        HF2Q_DECODE_MVN="$DECODE_MVN" \
        HF2Q_DECODE_MV_EXT="$DECODE_MV_EXT" \
        "$script_dir/serve_qwen38_opencode.sh" >"$log_path" 2>&1 &
    server_pid=$!
    wait_ready "$log_path"
    resolve_loaded_model_id

    curl --fail --silent --show-error \
        --header 'Content-Type: application/json' \
        --data-binary "@$OUT_DIR/requests/warmup.json" \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        >"$mode_dir/warmup.json"

    local name
    for name in $CASES; do
        curl --fail --silent --show-error \
            --header 'Content-Type: application/json' \
            --data-binary "@$OUT_DIR/requests/$name.json" \
            --output "$mode_dir/$name.json" \
            --write-out '%{time_total}\n' \
            "http://127.0.0.1:$PORT/v1/chat/completions" \
            >"$mode_dir/$name.seconds"
        jq -e '.choices | length == 1' "$mode_dir/$name.json" >/dev/null
        jq -S -c '.choices[0]' "$mode_dir/$name.json" >"$mode_dir/$name.choice.json"
    done

    curl --fail --silent --show-error "http://127.0.0.1:$PORT/metrics" \
        >"$mode_dir/metrics.txt"
    stop_server
}

trial_index=0
for mode in $TRIAL_ORDER; do
    trial_index=$((trial_index + 1))
    run_trial "$trial_index" "$mode"
done

for name in $CASES; do
    reference="$OUT_DIR/trial-1-off/$name.choice.json"
    for trial_dir in "$OUT_DIR"/trial-*-*; do
        cmp "$reference" "$trial_dir/$name.choice.json"
    done
done

median_group() {
    local mode=$1
    local prefix=$2
    sort -n "$OUT_DIR"/trial-*-"$mode"/"$prefix"-*.seconds | awk '
        { value[NR] = $1 }
        END {
            if (NR == 0) exit 2
            if (NR % 2 == 1) printf "%.6f", value[(NR + 1) / 2]
            else printf "%.6f", (value[NR / 2] + value[NR / 2 + 1]) / 2.0
        }
    '
}

percent_improvement() {
    local baseline=$1
    local candidate=$2
    awk -v baseline="$baseline" -v candidate="$candidate" \
        'BEGIN { printf "%.6f", ((baseline / candidate) - 1.0) * 100.0 }'
}

off_code_median=$(median_group off code)
auto_code_median=$(median_group auto code)
off_repeat_median=$(median_group off repeat)
auto_repeat_median=$(median_group auto repeat)
code_improvement=$(percent_improvement "$off_code_median" "$auto_code_median")
repeat_improvement=$(percent_improvement "$off_repeat_median" "$auto_repeat_median")

awk -v actual="$code_improvement" -v minimum="$MIN_CODE_IMPROVEMENT_PERCENT" \
    'BEGIN { exit !(actual >= minimum) }' || {
    echo "Qwen3.8 AUTO code improvement ${code_improvement}% is below ${MIN_CODE_IMPROVEMENT_PERCENT}%" >&2
    exit 1
}
awk -v actual="$repeat_improvement" -v minimum="$MIN_REPEAT_IMPROVEMENT_PERCENT" \
    'BEGIN { exit !(actual >= minimum) }' || {
    echo "Qwen3.8 AUTO repeat improvement ${repeat_improvement}% is below ${MIN_REPEAT_IMPROVEMENT_PERCENT}%" >&2
    exit 1
}

auto_proposals=$(awk '
    /^hf2q_qwen_speculation_proposals_total\{proposer="(history_lookup|mtp)"\}/ { sum += $2 }
    END { print sum + 0 }
' "$OUT_DIR"/trial-*-auto/metrics.txt)
auto_accepted=$(awk '
    /^hf2q_qwen_speculation_accepted_tokens_total\{proposer="(history_lookup|mtp)"\}/ { sum += $2 }
    END { print sum + 0 }
' "$OUT_DIR"/trial-*-auto/metrics.txt)
if ((auto_proposals < 1 || auto_accepted < 1)); then
    echo "Qwen3.8 AUTO did not prove active accepted speculation" >&2
    exit 1
fi

binary_sha256=$(sha256_file "$BINARY_PATH")
hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
    "$model_verification_receipt"
model_bytes=$(file_bytes "$MODEL_PATH")
jq -n \
    --arg binary_path "$BINARY_PATH" \
    --arg binary_sha256 "$binary_sha256" \
    --arg model_id "$MODEL_ID" \
    --arg model_path "$MODEL_PATH" \
    --arg model_sha256 "$MODEL_SHA256" \
    --arg model_verification "$model_verification_mode" \
    --arg model_file_snapshot "$model_file_snapshot" \
    --argjson model_bytes "$model_bytes" \
    --argjson max_tokens "$MAX_TOKENS" \
    --argjson decode_mvn "$DECODE_MVN" \
    --argjson decode_mv_ext "$DECODE_MV_EXT" \
    --argjson off_code_median_seconds "$off_code_median" \
    --argjson auto_code_median_seconds "$auto_code_median" \
    --argjson code_throughput_improvement_percent "$code_improvement" \
    --argjson off_repeat_median_seconds "$off_repeat_median" \
    --argjson auto_repeat_median_seconds "$auto_repeat_median" \
    --argjson repeat_throughput_improvement_percent "$repeat_improvement" \
    --argjson auto_proposals "$auto_proposals" \
    --argjson auto_accepted_tokens "$auto_accepted" \
    '{schema:1,verdict:"pass",exact_choices_parity:true,
      binary:{path:$binary_path,sha256:$binary_sha256},
      model:{id:$model_id,path:$model_path,sha256:$model_sha256,bytes:$model_bytes,
             verification:$model_verification,file_snapshot:$model_file_snapshot},
      routing:{dense_decode_mvn:$decode_mvn,dense_decode_mv_ext:$decode_mv_ext},
      workload:{trial_order:"off auto auto off",cases_per_group_per_trial:3,
                repetitions_per_mode:2,max_tokens:$max_tokens,temperature:0},
      code:{off_median_seconds:$off_code_median_seconds,
            auto_median_seconds:$auto_code_median_seconds,
            throughput_improvement_percent:$code_throughput_improvement_percent},
      repeat:{off_median_seconds:$off_repeat_median_seconds,
              auto_median_seconds:$auto_repeat_median_seconds,
              throughput_improvement_percent:$repeat_throughput_improvement_percent},
      speculation:{proposals:$auto_proposals,accepted_tokens:$auto_accepted_tokens}}' \
    >"$OUT_DIR/summary.json.tmp"
mv "$OUT_DIR/summary.json.tmp" "$OUT_DIR/summary.json"
jq . "$OUT_DIR/summary.json"
