#!/usr/bin/env bash
set -euo pipefail

# Reproducible real-server proof that ordinary Qwen3.8 decode physically
# batches 1, 2, 4, 8, and 16 simultaneous slots. This gate measures hf2q only;
# matched peer/reference performance is a separate receipt.
#
# MODEL_PATH=/path/to/model.gguf MODEL_SHA256=<digest> \
# BINARY_PATH=/path/to/hf2q OUT_DIR=/fresh/receipt \
# scripts/qwen38_physical_multislot_gate.sh

BINARY_PATH=${BINARY_PATH:?BINARY_PATH is required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
MODEL_FORMAT=${MODEL_FORMAT:?MODEL_FORMAT is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
PORT=${PORT:-18092}
MAX_TOKENS=${MAX_TOKENS:-64}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-900}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-600}
KV_CACHE_BUDGET_BYTES=${KV_CACHE_BUDGET_BYTES:-51539607552}
DECODE_MVN=${HF2Q_DECODE_MVN:-0}
DECODE_MV_EXT=${HF2Q_DECODE_MV_EXT:-1}
readonly WIDTHS=(1 2 4 8 16)

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$script_dir/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/qwen38_physical_multislot_contract.sh
source "$script_dir/qwen38_physical_multislot_contract.sh"

for command in awk curl find grep jq lsof perl ps sed shasum sort stat; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ -x "$BINARY_PATH" ]] || {
    echo "hf2q binary is missing or non-executable: $BINARY_PATH" >&2
    exit 2
}
[[ -f "$MODEL_PATH" && -r "$MODEL_PATH" ]] || {
    echo "Qwen3.8 model is missing or unreadable: $MODEL_PATH" >&2
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
for value in "$MAX_TOKENS" "$REQUEST_TIMEOUT_SECONDS" \
    "$READY_TIMEOUT_SECONDS" "$KV_CACHE_BUDGET_BYTES"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
        echo "token, timeout, and cache-budget settings must be positive integers" >&2
        exit 2
    fi
done
case "$DECODE_MVN:$DECODE_MV_EXT" in
    0:1|0:0|1:0|1:1) ;;
    *)
        echo "HF2Q_DECODE_MVN and HF2Q_DECODE_MV_EXT must each be 0 or 1" >&2
        exit 2
        ;;
esac
qwen36_require_empty_receipt_dir "$OUT_DIR"

if [[ -n "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]]; then
    echo "127.0.0.1:$PORT is already in use; refusing before model load" >&2
    exit 2
fi

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
    hf2q_release_verify_model \
        "$MODEL_PATH" "$MODEL_SHA256" "$model_verification_receipt"
    model_verification_mode=provided_receipt
fi
model_file_snapshot=$(jq -er .file_snapshot "$model_verification_receipt")

binary_sha256=$(shasum -a 256 "$BINARY_PATH" | awk '{print $1}')
binary_snapshot=$(hf2q_release_model_snapshot "$BINARY_PATH")
model_bytes=$(stat -f '%z' "$MODEL_PATH" 2>/dev/null \
    || stat -c '%s' "$MODEL_PATH")
artifact_record=$(qwen38_artifact_record "$MODEL_FORMAT")
IFS=$'\t' read -r qualified_model_format qualified_model_file \
    _qualified_model_bytes _qualified_model_sha256 qualified_model_file_type \
    <<<"$artifact_record"
[[ "$qualified_model_format" == "$MODEL_FORMAT" ]]
qwen38_validate_artifact_identity "$MODEL_FORMAT" "$MODEL_SHA256" \
    "$model_bytes" "$qualified_model_file_type"
server_pid=''

monotonic_seconds() {
    perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
        -e 'printf "%.9f\n", clock_gettime(CLOCK_MONOTONIC)'
}

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
        kill -TERM "$server_pid" 2>/dev/null || true
        waited=0
        while kill -0 "$server_pid" 2>/dev/null && ((waited < 10)); do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    if kill -0 "$server_pid" 2>/dev/null; then
        echo "Qwen3.8 gate server ignored bounded shutdown; killing owned child $server_pid" >&2
        kill -KILL "$server_pid" 2>/dev/null || true
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
    local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
    while ((SECONDS < deadline)); do
        if curl --fail --silent --show-error --max-time 2 \
            "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Qwen3.8 physical-width server exited before readiness" >&2
            sed -n '1,240p' "$log_path" >&2
            return 1
        fi
        sleep 1
    done
    echo "Qwen3.8 physical-width server did not become ready" >&2
    sed -n '1,240p' "$log_path" >&2
    return 1
}

resolve_loaded_model_id() {
    local models_path=$1
    curl --fail --silent --show-error --max-time 10 \
        "http://127.0.0.1:$PORT/v1/models" >"$models_path"
    jq -er '
        [.data[] | select(.loaded == true)]
        | if length == 1 then .[0].id
          else error("expected exactly one loaded model") end
    ' "$models_path"
}

write_request() {
    local path=$1
    local model_id=$2
    local lane=$3
    jq -n --arg model "$model_id" --arg lane "$lane" \
        --argjson max_tokens "$MAX_TOKENS" '{
          model:$model,
          messages:[
            {role:"system",content:"You are a deterministic native-inference validation assistant."},
            {role:"user",content:("Physical batching lane " + $lane + ". Write a numbered sequence from 1 through 96. For each item emit the number, a colon, and the word cobalt. Do not stop early.")}
          ],
          max_tokens:$max_tokens,
          temperature:0,
          repetition_penalty:1.0,
          stream:false,
          hf2q_enable_thinking:false
        }' >"$path.tmp"
    mv "$path.tmp" "$path"
}

write_warmup_request() {
    local path=$1
    local model_id=$2
    jq -n --arg model "$model_id" '{
      model:$model,
      messages:[
        {role:"system",content:"You are a deterministic native-inference validation assistant."},
        {role:"user",content:"Return exactly WARMUP."}
      ],
      max_tokens:8,
      temperature:0,
      repetition_penalty:1.0,
      stream:false,
      hf2q_enable_thinking:false
    }' >"$path.tmp"
    mv "$path.tmp" "$path"
}

run_width() {
    local width=$1
    local width_dir="$OUT_DIR/width-$width"
    local log_path="$width_dir/server.log"
    local loaded_model_id wave_start wave_end wave_wall
    local request_start_file="$width_dir/start"
    local failed=0
    local index lane response request curl_metrics output scalar_response
    local pid
    local clients_json total_completion_tokens max_decode_seconds
    local summed_user_decode_tps aggregate_decode_tps aggregate_wave_tps
    local metrics_json
    local -a request_pids=()
    local -a responses=()

    mkdir -p "$width_dir/requests" "$width_dir/responses" \
        "$width_dir/scalar-responses"
    if [[ -n "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]]; then
        echo "127.0.0.1:$PORT remained occupied before width $width" >&2
        return 1
    fi

    env \
        HF2Q_BIN="$BINARY_PATH" \
        MODEL="$MODEL_PATH" \
        PORT="$PORT" \
        MAX_SLOTS="$width" \
        KV_CACHE_BUDGET_BYTES="$KV_CACHE_BUDGET_BYTES" \
        QWEN38_VISION=off \
        QWEN38_SPECULATION=off \
        THINKING_TOKEN_BUDGET=0 \
        TOOL_THINKING_TOKEN_BUDGET=0 \
        REP_PENALTY=1.0 \
        HF2Q_DECODE_MVN="$DECODE_MVN" \
        HF2Q_DECODE_MV_EXT="$DECODE_MV_EXT" \
        "$script_dir/serve_qwen38_opencode.sh" >"$log_path" 2>&1 &
    server_pid=$!
    wait_ready "$log_path"
    qwen36_bind_server_process \
        "http://127.0.0.1:$PORT" "$server_pid" \
        "$BINARY_PATH" "$MODEL_PATH" "$width"
    loaded_model_id=$(resolve_loaded_model_id "$width_dir/models.json")

    write_warmup_request "$width_dir/warmup-request.json" "$loaded_model_id"
    curl --fail --silent --show-error --max-time "$REQUEST_TIMEOUT_SECONDS" \
        --header 'Content-Type: application/json' \
        --data-binary "@$width_dir/warmup-request.json" \
        "http://127.0.0.1:$PORT/v1/chat/completions" \
        >"$width_dir/warmup-response.json"
    qwen38_physical_validate_response \
        "$width_dir/warmup-response.json" "$loaded_model_id"
    curl --fail --silent --show-error --max-time 10 \
        "http://127.0.0.1:$PORT/metrics" >"$width_dir/metrics-before.txt"

    for ((index = 1; index <= width; index++)); do
        printf -v lane '\\x%02x' "$((64 + index))"
        printf -v lane '%b' "$lane"
        request="$width_dir/requests/lane-$index.json"
        response="$width_dir/responses/lane-$index.json"
        curl_metrics="$width_dir/responses/lane-$index.curl"
        write_request "$request" "$loaded_model_id" "$lane"
        responses+=("$response")
        (
            while [[ ! -e "$request_start_file" ]]; do sleep 0.01; done
            curl --fail --silent --show-error \
                --max-time "$REQUEST_TIMEOUT_SECONDS" \
                --header 'Content-Type: application/json' \
                --data-binary "@$request" \
                --output "$response" \
                --write-out 'http_code=%{http_code}\ntotal_seconds=%{time_total}\n' \
                "http://127.0.0.1:$PORT/v1/chat/completions" \
                >"$curl_metrics"
        ) &
        request_pids+=("$!")
    done

    wave_start=$(monotonic_seconds)
    : >"$request_start_file"
    for pid in "${request_pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    wave_end=$(monotonic_seconds)
    wave_wall=$(awk -v start="$wave_start" -v end="$wave_end" \
        'BEGIN { printf "%.9f", end - start }')

    qwen36_bind_server_process \
        "http://127.0.0.1:$PORT" "$server_pid" \
        "$BINARY_PATH" "$MODEL_PATH" "$width"
    curl --fail --silent --show-error --max-time 10 \
        "http://127.0.0.1:$PORT/metrics" >"$width_dir/metrics-after.txt"
    (( failed == 0 )) || {
        echo "one or more width-$width requests failed" >&2
        sed -n '1,260p' "$log_path" >&2
        return 1
    }
    qwen36_reject_fatal_log "$log_path"
    hf2q_release_verify_model \
        "$MODEL_PATH" "$MODEL_SHA256" "$model_verification_receipt"
    [[ "$(hf2q_release_model_snapshot "$BINARY_PATH")" == "$binary_snapshot" ]] || {
        echo "hf2q binary changed during the physical-width gate" >&2
        return 1
    }

    for ((index = 1; index <= width; index++)); do
        response="$width_dir/responses/lane-$index.json"
        output="$width_dir/responses/lane-$index.txt"
        qwen38_physical_validate_response "$response" "$loaded_model_id"
        jq -er '.choices[0].message.content' "$response" >"$output"
        [[ "$(sed -n 's/^http_code=//p' "$width_dir/responses/lane-$index.curl")" == 200 ]] || {
            echo "width-$width lane-$index did not return HTTP 200" >&2
            return 1
        }
    done
    qwen38_physical_validate_equal_prompt_tokens "$width" "${responses[@]}"
    if [[ "$(for request in "$width_dir"/requests/*.json; do
                    shasum -a 256 "$request"
                done | awk '{print $1}' | sort -u | awk 'END {print NR}')" != "$width" ]]; then
        echo "width-$width request prompts were not all distinct" >&2
        return 1
    fi
    qwen38_physical_validate_metrics \
        "$width" "$width_dir/metrics-before.txt" "$width_dir/metrics-after.txt"

    # Replay each exact request alone after the concurrent wave. Width
    # telemetry plus non-empty JSON does not prove slot isolation; exact
    # greedy semantics against scalar execution does.
    for ((index = 1; index <= width; index++)); do
        request="$width_dir/requests/lane-$index.json"
        response="$width_dir/responses/lane-$index.json"
        scalar_response="$width_dir/scalar-responses/lane-$index.json"
        curl --fail --silent --show-error \
            --max-time "$REQUEST_TIMEOUT_SECONDS" \
            --header 'Content-Type: application/json' \
            --data-binary "@$request" \
            "http://127.0.0.1:$PORT/v1/chat/completions" \
            >"$scalar_response"
        qwen38_physical_validate_response "$scalar_response" "$loaded_model_id"
        qwen38_physical_validate_scalar_parity "$response" "$scalar_response"
    done
    qwen36_bind_server_process \
        "http://127.0.0.1:$PORT" "$server_pid" \
        "$BINARY_PATH" "$MODEL_PATH" "$width"
    stop_server
    qwen36_reject_fatal_log "$log_path"

    clients_json=$(for ((index = 1; index <= width; index++)); do
        response="$width_dir/responses/lane-$index.json"
        curl_metrics="$width_dir/responses/lane-$index.curl"
        output="$width_dir/responses/lane-$index.txt"
        jq -n \
            --argjson lane "$index" \
            --arg response_path "responses/lane-$index.json" \
            --arg scalar_response_path "scalar-responses/lane-$index.json" \
            --arg output_path "responses/lane-$index.txt" \
            --arg output_sha256 "$(shasum -a 256 "$output" | awk '{print $1}')" \
            --argjson wall_seconds "$(sed -n 's/^total_seconds=//p' "$curl_metrics")" \
            --argjson prompt_tokens "$(jq -er .usage.prompt_tokens "$response")" \
            --argjson completion_tokens "$(jq -er .usage.completion_tokens "$response")" \
            --argjson cached_tokens "$(jq -er .usage.prompt_tokens_details.cached_tokens "$response")" \
            --argjson decode_seconds "$(jq -er .x_hf2q_timing.decode_time_secs "$response")" \
            --argjson decode_tokens_per_second "$(jq -er .x_hf2q_timing.decode_tokens_per_sec "$response")" \
            --argjson time_to_first_token_ms "$(jq -er .x_hf2q_timing.time_to_first_token_ms "$response")" \
            '{lane:$lane,response_path:$response_path,
              scalar_response_path:$scalar_response_path,scalar_parity:true,
              output_path:$output_path,
              output_sha256:$output_sha256,wall_seconds:$wall_seconds,
              prompt_tokens:$prompt_tokens,completion_tokens:$completion_tokens,
              cached_tokens:$cached_tokens,decode_seconds:$decode_seconds,
              decode_tokens_per_second:$decode_tokens_per_second,
              time_to_first_token_ms:$time_to_first_token_ms}'
    done | jq -s .)

    total_completion_tokens=$(jq '[.[].completion_tokens] | add' <<<"$clients_json")
    max_decode_seconds=$(jq '[.[].decode_seconds] | max' <<<"$clients_json")
    summed_user_decode_tps=$(jq '[.[].decode_tokens_per_second] | add' <<<"$clients_json")
    aggregate_decode_tps=$(awk -v tokens="$total_completion_tokens" \
        -v seconds="$max_decode_seconds" 'BEGIN { printf "%.6f", tokens / seconds }')
    aggregate_wave_tps=$(awk -v tokens="$total_completion_tokens" \
        -v seconds="$wave_wall" 'BEGIN { printf "%.6f", tokens / seconds }')

    metrics_json=$(jq -n \
        --argjson scheduler_max_width "$(qwen38_physical_metric_u64 "$width_dir/metrics-after.txt" hf2q_qwen_decode_scheduler_max_width)" \
        --argjson body_max_width "$(qwen38_physical_metric_u64 "$width_dir/metrics-after.txt" hf2q_qwen_decode_ordinary_target_body_max_width)" \
        --argjson head_max_width "$(qwen38_physical_metric_u64 "$width_dir/metrics-after.txt" hf2q_qwen_decode_ordinary_target_head_max_width)" \
        --argjson forwards_before "$(qwen38_physical_metric_u64 "$width_dir/metrics-before.txt" hf2q_qwen_decode_ordinary_target_forwards_total)" \
        --argjson forwards_after "$(qwen38_physical_metric_u64 "$width_dir/metrics-after.txt" hf2q_qwen_decode_ordinary_target_forwards_total)" \
        --argjson body_rows_before "$(qwen38_physical_metric_u64 "$width_dir/metrics-before.txt" hf2q_qwen_decode_ordinary_target_body_rows_total)" \
        --argjson body_rows_after "$(qwen38_physical_metric_u64 "$width_dir/metrics-after.txt" hf2q_qwen_decode_ordinary_target_body_rows_total)" \
        --argjson head_rows_before "$(qwen38_physical_metric_u64 "$width_dir/metrics-before.txt" hf2q_qwen_decode_ordinary_target_head_rows_total)" \
        --argjson head_rows_after "$(qwen38_physical_metric_u64 "$width_dir/metrics-after.txt" hf2q_qwen_decode_ordinary_target_head_rows_total)" \
        --argjson buffers_created_before "$(qwen38_physical_metric_u64 "$width_dir/metrics-before.txt" hf2q_qwen_decode_ordinary_command_buffers_created_total)" \
        --argjson buffers_created_after "$(qwen38_physical_metric_u64 "$width_dir/metrics-after.txt" hf2q_qwen_decode_ordinary_command_buffers_created_total)" \
        --argjson submissions_before "$(qwen38_physical_metric_u64 "$width_dir/metrics-before.txt" hf2q_qwen_decode_ordinary_command_buffer_submissions_total)" \
        --argjson submissions_after "$(qwen38_physical_metric_u64 "$width_dir/metrics-after.txt" hf2q_qwen_decode_ordinary_command_buffer_submissions_total)" \
        '{scheduler_max_width:$scheduler_max_width,
          target_body_max_width:$body_max_width,
          target_head_max_width:$head_max_width,
          target_forwards_before:$forwards_before,
          target_forwards_after:$forwards_after,
          target_forwards_delta:($forwards_after-$forwards_before),
          target_body_rows_before:$body_rows_before,
          target_body_rows_after:$body_rows_after,
          target_body_rows_delta:($body_rows_after-$body_rows_before),
          target_head_rows_before:$head_rows_before,
          target_head_rows_after:$head_rows_after,
          target_head_rows_delta:($head_rows_after-$head_rows_before),
          command_buffers_created_before:$buffers_created_before,
          command_buffers_created_after:$buffers_created_after,
          command_buffers_created_delta:($buffers_created_after-$buffers_created_before),
          command_buffer_submissions_before:$submissions_before,
          command_buffer_submissions_after:$submissions_after,
          command_buffer_submissions_delta:($submissions_after-$submissions_before)}')

    jq -n \
        --argjson width "$width" \
        --arg model_id "$loaded_model_id" \
        --argjson max_tokens "$MAX_TOKENS" \
        --argjson wave_wall_seconds "$wave_wall" \
        --argjson total_completion_tokens "$total_completion_tokens" \
        --argjson aggregate_decode_tokens_per_second "$aggregate_decode_tps" \
        --argjson aggregate_wave_tokens_per_second "$aggregate_wave_tps" \
        --argjson summed_user_decode_tokens_per_second "$summed_user_decode_tps" \
        --argjson clients "$clients_json" \
        --argjson metrics "$metrics_json" '{
          schema:1,verdict:"pass",width:$width,model_id:$model_id,
          request:{temperature:0,thinking:false,speculation:"off",
            stream:false,max_tokens:$max_tokens,distinct_equal_token_prompts:true,
            exact_scalar_replay_per_lane:true},
          wave:{wall_seconds:$wave_wall_seconds,
            total_completion_tokens:$total_completion_tokens,
            aggregate_decode_tokens_per_second:$aggregate_decode_tokens_per_second,
            aggregate_wave_tokens_per_second:$aggregate_wave_tokens_per_second,
            summed_user_decode_tokens_per_second:$summed_user_decode_tokens_per_second},
          metrics:$metrics,clients:$clients
        }' >"$width_dir/summary.json.tmp"
    mv "$width_dir/summary.json.tmp" "$width_dir/summary.json"
}

for width in "${WIDTHS[@]}"; do
    run_width "$width"
done

summary_paths=()
for width in "${WIDTHS[@]}"; do
    summary_paths+=("$OUT_DIR/width-$width/summary.json")
done
width_summaries=$(jq -s . "${summary_paths[@]}")
jq -e '[.[].width] | sort == [1,2,4,8,16]' <<<"$width_summaries" >/dev/null
hf2q_release_verify_model \
    "$MODEL_PATH" "$MODEL_SHA256" "$model_verification_receipt"
qwen38_validate_artifact_identity "$MODEL_FORMAT" "$MODEL_SHA256" \
    "$model_bytes" "$qualified_model_file_type"
[[ "$(hf2q_release_model_snapshot "$BINARY_PATH")" == "$binary_snapshot" ]] || {
    echo "hf2q binary changed before the physical-width summary" >&2
    exit 1
}
jq -n \
    --arg binary_path "$BINARY_PATH" \
    --arg binary_sha256 "$binary_sha256" \
    --arg model_path "$MODEL_PATH" \
    --arg model_format "$MODEL_FORMAT" \
    --arg model_repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
    --arg model_revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
    --arg model_file "$qualified_model_file" \
    --arg model_sha256 "$MODEL_SHA256" \
    --arg model_file_snapshot "$model_file_snapshot" \
    --arg model_verification "$model_verification_mode" \
    --argjson model_bytes "$model_bytes" \
    --argjson model_file_type "$qualified_model_file_type" \
    --argjson max_tokens "$MAX_TOKENS" \
    --argjson kv_cache_budget_bytes "$KV_CACHE_BUDGET_BYTES" \
    --argjson decode_mvn "$DECODE_MVN" \
    --argjson decode_mv_ext "$DECODE_MV_EXT" \
    --argjson widths "$width_summaries" '{
      schema:1,verdict:"pass",gate:"qwen38-physical-multislot",
      binary:{path:$binary_path,sha256:$binary_sha256},
      model:{path:$model_path,format:$model_format,
        repository:$model_repository,revision:$model_revision,file:$model_file,
        gguf_file_type:$model_file_type,sha256:$model_sha256,bytes:$model_bytes,
        file_snapshot:$model_file_snapshot,verification:$model_verification},
      workload:{widths:[1,2,4,8,16],temperature:0,thinking:false,
        speculation:"off",stream:false,max_tokens:$max_tokens,
        exact_scalar_replay_per_lane:true,
        server_restart_per_width:true,
        kv_cache_budget_bytes:$kv_cache_budget_bytes,
        routing:{decode_mvn:$decode_mvn,decode_mv_ext:$decode_mv_ext}},
      results:$widths
    }' >"$OUT_DIR/summary.json.tmp"
jq -e '
  .schema == 1 and .verdict == "pass"
  and .workload.widths == [1,2,4,8,16]
  and .workload.exact_scalar_replay_per_lane == true
  and ([.results[].width] == [1,2,4,8,16])
  and all(.results[];
    .verdict == "pass"
    and .request.exact_scalar_replay_per_lane == true
    and (.clients | length) == .width
    and all(.clients[]; .scalar_parity == true))
  and ((.model.format == "BF16" and .model.gguf_file_type == 32)
    or (.model.format == "Q4_K_M" and .model.gguf_file_type == 15)
    or (.model.format == "Q5_K_M" and .model.gguf_file_type == 17)
    or (.model.format == "Q6_K" and .model.gguf_file_type == 18)
    or (.model.format == "Q8_0" and .model.gguf_file_type == 7))
' "$OUT_DIR/summary.json.tmp" >/dev/null
mv "$OUT_DIR/summary.json.tmp" "$OUT_DIR/summary.json"
jq . "$OUT_DIR/summary.json"
