#!/usr/bin/env bash
set -euo pipefail

# Production-route hardware spike for ADR-049 B.2. This script deliberately
# attaches to an already-running four-slot server: model loading is outside the
# measured interval, while every request is bound to one production prefill
# transaction and retained as immutable raw evidence.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"

readonly WIDTHS=(128 256 512 1024 1792)
readonly WARMUPS=2
readonly TRIALS=${TRIALS:-7}
readonly REQUEST_TIMEOUT_SECONDS=300
readonly SETTLE_SECONDS=60
readonly SETTLE_TIMEOUT_SECONDS=900
readonly SETTLE_SAMPLE_SECONDS=5
readonly MEASUREMENT_SAMPLE_SECONDS=2

FAMILY=${FAMILY:?FAMILY=qwen35_moe or gemma4_moe is required}
SOURCE_ROOT=${SOURCE_ROOT:?absolute clean source worktree is required}
EXPECTED_SOURCE_SHA=${EXPECTED_SOURCE_SHA:?exact 40-character source SHA is required}
HF2Q_BIN=${HF2Q_BIN:?exact server binary path is required}
EXPECTED_BINARY_SHA256=${EXPECTED_BINARY_SHA256:?exact binary SHA-256 is required}
MODEL_PATH=${MODEL_PATH:?exact model artifact path is required}
EXPECTED_MODEL_SHA256=${EXPECTED_MODEL_SHA256:?exact model SHA-256 is required}
EXPECTED_MODEL_BYTES=${EXPECTED_MODEL_BYTES:?exact model byte count is required}
SERVER_PID=${SERVER_PID:?live server PID is required}
SERVER_LOG=${SERVER_LOG:?server log path is required}
BASE_URL=${BASE_URL:?server base URL is required, for example http://127.0.0.1:8080}
OUT_DIR=${OUT_DIR:?new absolute receipt directory is required}

for command in awk curl git grep head jq mkdir mktemp mv perl ps python3 rm shasum stat tail; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
for path in "$SOURCE_ROOT" "$HF2Q_BIN" "$MODEL_PATH" "$SERVER_LOG" "$OUT_DIR"; do
    [[ "$path" == /* ]] || { echo "all filesystem inputs must be absolute: $path" >&2; exit 2; }
done
case "$FAMILY" in
    # The exact Qwen3.6 MoE chat template adds 36--37 rendered tokens around
    # this one-message payload. Rev-1 measured target+36/37 and correctly
    # failed the 128-row bin. Subtract the stable lower envelope; the recorded
    # response and trace remain the authority for the actual work rows.
    qwen35_moe) trace_kind=qwen35_chunk; payload_word_adjustment=36 ;;
    gemma4_moe) trace_kind=gemma4_transaction; payload_word_adjustment=0 ;;
    *) echo "unsupported B.2 family: $FAMILY" >&2; exit 2 ;;
esac
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ \
    && "$EXPECTED_BINARY_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$EXPECTED_MODEL_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$EXPECTED_MODEL_BYTES" =~ ^[1-9][0-9]*$ \
    && "$SERVER_PID" =~ ^[1-9][0-9]*$ ]] || {
    echo "malformed exact identity input" >&2
    exit 2
}
[[ "$TRIALS" == 7 || "$TRIALS" == 21 ]] || {
    echo "TRIALS must be the pre-registered initial 7 or inconclusive extension 21" >&2
    exit 2
}
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to reuse receipt directory: $OUT_DIR" >&2; exit 2; }
[[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" \
    && -x "$HF2Q_BIN" && ! -L "$HF2Q_BIN" \
    && -f "$MODEL_PATH" && ! -L "$MODEL_PATH" \
    && -f "$SERVER_LOG" && ! -L "$SERVER_LOG" ]] || {
    echo "source, binary, model, or server-log input is not a canonical local object" >&2
    exit 2
}
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd -P)
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$EXPECTED_SOURCE_SHA" \
    && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
    echo "source worktree is dirty or not at EXPECTED_SOURCE_SHA" >&2
    exit 2
}
grep -aFq "$EXPECTED_SOURCE_SHA" "$HF2Q_BIN" || {
    echo "server binary does not embed EXPECTED_SOURCE_SHA" >&2
    exit 2
}
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
file_bytes() { stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1"; }
[[ "$(sha256_file "$HF2Q_BIN")" == "$EXPECTED_BINARY_SHA256" ]] || {
    echo "binary SHA-256 mismatch" >&2
    exit 2
}
[[ "$(file_bytes "$MODEL_PATH")" == "$EXPECTED_MODEL_BYTES" \
    && "$(sha256_file "$MODEL_PATH")" == "$EXPECTED_MODEL_SHA256" ]] || {
    echo "model artifact identity mismatch" >&2
    exit 2
}
kill -0 "$SERVER_PID" 2>/dev/null || { echo "server PID is not live" >&2; exit 2; }
server_command=$(ps -ww -p "$SERVER_PID" -o command=)
scheduler_pattern='--scheduler(=|[[:space:]])inflight-batched'
slots_pattern='--max-slots(=|[[:space:]])4([[:space:]]|$)'
[[ -n "$server_command" && "$server_command" == *"$HF2Q_BIN"* \
    && "$server_command" == *"$MODEL_PATH"* \
    && "$server_command" =~ $scheduler_pattern \
    && "$server_command" =~ $slots_pattern ]] || {
    echo "server command is not the exact four-slot production route" >&2
    exit 2
}

mkdir -p "$OUT_DIR/samples"
OUT_DIR=$(cd "$OUT_DIR" && pwd -P)
samples="$OUT_DIR/samples.jsonl"
server_slice="$OUT_DIR/server-measurement.log"
settle_log="$OUT_DIR/thermal-settle.log"
measurement_log="$OUT_DIR/thermal-measurement.log"
contention_settle_log="$OUT_DIR/contention-settle.log"
contention_measurement_log="$OUT_DIR/contention-measurement.log"
: >"$samples"
: >"$server_slice"
: >"$measurement_log"
: >"$contention_measurement_log"

models_response="$OUT_DIR/models.json"
curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 30 \
    "$BASE_URL/v1/models" >"$models_response"
model_id=$(jq -er 'select((.data | length) == 1) | .data[0].id | select(type == "string" and length > 0)' "$models_response")
server_log_start=$(file_bytes "$SERVER_LOG")

cleanup() {
    thermal_cleanup_probe >/dev/null 2>&1 || true
}
trap cleanup EXIT
thermal_prepare_probe

build_request() {
    local target=$1 sample_id=$2 output=$3 content payload_words
    payload_words=$((target - payload_word_adjustment))
    ((payload_words > 0)) || {
        echo "rendered-prompt adjustment exhausts target row bin: $target" >&2
        return 1
    }
    content=$(awk -v target="$payload_words" -v sample="$sample_id" 'BEGIN {
        printf "adr049-b2-%s ", sample
        for (i = 1; i <= target; i++) printf "measurement "
        printf "Reply with one word."
    }')
    jq -n --arg model "$model_id" --arg content "$content" '{
      model:$model,
      messages:[{role:"user",content:$content}],
      max_tokens:1,seed:42,temperature:0,repetition_penalty:1,stream:false,
      hf2q_enable_thinking:false,
      chat_template_kwargs:{enable_thinking:false}
    }' >"$output"
}

trace_rows() {
    local trace=$1
    case "$trace_kind" in
        qwen35_chunk)
            perl -pe 's/\e\[[0-9;]*m//g' "$trace" \
                | perl -ne 'print "$1\n" if /Qwen35 bounded prefill chunk complete/ && /chunk_tokens[=: ]+([0-9]+)/'
            ;;
        gemma4_transaction)
            perl -pe 's/\e\[[0-9;]*m//g' "$trace" \
                | perl -ne 'print "$1\n" if /Gemma4 bounded prefill transaction complete/ && /advanced_tokens[=: ]+([0-9]+)/'
            ;;
    esac
}

preflight_live_trace() {
    local preflight_dir request response trace before after prompt cached event_count advanced
    preflight_dir=$(mktemp -d -t hf2q-adr049-b2-preflight.XXXXXX)
    request="$preflight_dir/request.json"
    response="$preflight_dir/response.json"
    trace="$preflight_dir/trace.log"
    if ! build_request 128 preflight-00-00-0128 "$request"; then
        rm -rf "$preflight_dir"
        return 1
    fi
    before=$(file_bytes "$SERVER_LOG")
    if ! curl --fail-with-body --silent --show-error --connect-timeout 5 \
        --max-time "$REQUEST_TIMEOUT_SECONDS" -H 'Content-Type: application/json' \
        --data-binary "@$request" -o "$response" \
        "$BASE_URL/v1/chat/completions"; then
        rm -rf "$preflight_dir"
        return 1
    fi
    after=$(file_bytes "$SERVER_LOG")
    ((after > before)) || {
        rm -rf "$preflight_dir"
        echo "server log is not live/unbuffered after the preflight request" >&2
        return 1
    }
    tail -c "+$((before + 1))" "$SERVER_LOG" | head -c "$((after - before))" >"$trace"
    prompt=$(jq -er '.usage.prompt_tokens' "$response")
    cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens' "$response")
    event_count=$(trace_rows "$trace" | awk 'END { print NR + 0 }')
    advanced=$(trace_rows "$trace" | awk 'NR == 1 { first = $0 } END { print first }')
    advanced=${advanced:-0}
    rm -rf "$preflight_dir"
    ((cached == 0 && prompt >= 96 && prompt <= 160)) || {
        echo "preflight request missed the cold 128-row bin: prompt=$prompt cached=$cached" >&2
        return 1
    }
    ((event_count == 1 && advanced == prompt)) || {
        echo "preflight trace is not one live production transaction: events=$event_count advanced=$advanced prompt=$prompt" >&2
        return 1
    }
}

run_sample() {
    local phase=$1 sweep=$2 position=$3 target=$4
    local sample_id request response wall trace before after event_count advanced
    local prompt cached work prefill_ms ttft_ms wall_ms
    sample_id=$(printf '%s-%02d-%02d-%04d' "$phase" "$sweep" "$position" "$target")
    request="$OUT_DIR/samples/$sample_id.request.json"
    response="$OUT_DIR/samples/$sample_id.response.json"
    wall="$OUT_DIR/samples/$sample_id.wall"
    trace="$OUT_DIR/samples/$sample_id.trace.log"
    build_request "$target" "$sample_id" "$request"
    before=$(file_bytes "$SERVER_LOG")
    curl --fail-with-body --silent --show-error --connect-timeout 5 \
        --max-time "$REQUEST_TIMEOUT_SECONDS" -H 'Content-Type: application/json' \
        --data-binary "@$request" -o "$response" -w '%{time_total}\n' \
        "$BASE_URL/v1/chat/completions" >"$wall"
    after=$(file_bytes "$SERVER_LOG")
    ((after >= before)) || { echo "server log shrank during $sample_id" >&2; return 1; }
    if ((after == before)); then
        : >"$trace"
    else
        tail -c "+$((before + 1))" "$SERVER_LOG" | head -c "$((after - before))" >"$trace"
    fi
    jq -e '
      (.choices | length) == 1
      and (.choices[0].message.content | type) == "string"
      and (.choices[0].message.content | length) > 0
      and (.choices[0].finish_reason | type) == "string"
      and (.usage.prompt_tokens | numbers) > 0
      and (.usage.prompt_tokens_details.cached_tokens == 0)
      and (.x_hf2q_timing.prefill_time_secs | numbers) > 0
      and (.x_hf2q_timing.time_to_first_token_ms | numbers) > 0
    ' "$response" >/dev/null
    prompt=$(jq -er '.usage.prompt_tokens' "$response")
    cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens' "$response")
    work=$((prompt - cached))
    prefill_ms=$(jq -er '.x_hf2q_timing.prefill_time_secs * 1000' "$response")
    ttft_ms=$(jq -er '.x_hf2q_timing.time_to_first_token_ms' "$response")
    wall_ms=$(awk '{printf "%.9f", $1 * 1000}' "$wall")
    event_count=$(trace_rows "$trace" | awk 'END { print NR + 0 }')
    advanced=$(trace_rows "$trace" | awk 'NR == 1 { first = $0 } END { print first }')
    advanced=${advanced:-0}
    jq -cn \
        --arg sample_id "$sample_id" --arg phase "$phase" \
        --argjson sweep "$sweep" --argjson position "$position" \
        --argjson target "$target" --argjson prompt "$prompt" \
        --argjson cached "$cached" --argjson work "$work" \
        --argjson prefill_ms "$prefill_ms" --argjson ttft_ms "$ttft_ms" \
        --argjson wall_ms "$wall_ms" --argjson events "$event_count" \
        --argjson advanced "$advanced" \
        --arg request_path "samples/$sample_id.request.json" \
        --arg request_sha "$(sha256_file "$request")" \
        --arg response_path "samples/$sample_id.response.json" \
        --arg response_sha "$(sha256_file "$response")" \
        --arg wall_path "samples/$sample_id.wall" \
        --arg wall_sha "$(sha256_file "$wall")" \
        --arg trace_path "samples/$sample_id.trace.log" \
        --arg trace_sha "$(sha256_file "$trace")" '{
          schema_version:1,sample_id:$sample_id,phase:$phase,sweep:$sweep,
          position:$position,target_rows:$target,prompt_tokens:$prompt,
          cached_tokens:$cached,work_rows:$work,prefill_ms:$prefill_ms,
          ttft_ms:$ttft_ms,wall_ms:$wall_ms,trace_event_count:$events,
          trace_advanced_rows:$advanced,request_path:$request_path,
          request_sha256:$request_sha,response_path:$response_path,
          response_sha256:$response_sha,wall_path:$wall_path,
          wall_sha256:$wall_sha,trace_path:$trace_path,
          trace_sha256:$trace_sha
        }' >>"$samples"
}

run_sweeps() {
    local phase=$1 sweeps=$2 sweep position target
    local -a order
    for ((sweep = 0; sweep < sweeps; sweep++)); do
        if ((sweep % 2 == 0)); then
            order=("${WIDTHS[@]}")
        else
            order=(1792 1024 512 256 128)
        fi
        for ((position = 0; position < ${#order[@]}; position++)); do
            target=${order[$position]}
            run_sample "$phase" "$sweep" "$position" "$target"
        done
    done
}

preflight_live_trace
thermal_wait_for_nominal "$settle_log" adr049-b2-settle "$SETTLE_SECONDS" \
    "$SETTLE_TIMEOUT_SECONDS" "$SETTLE_SAMPLE_SECONDS" \
    "$contention_settle_log" "$SERVER_PID"
thermal_sample "$measurement_log" adr049-b2-measurement-start
host_contention_sample "$contention_measurement_log" adr049-b2-measurement-start \
    "$SERVER_PID" "$THERMAL_SAMPLED_AT"
host_contention_require_quiet adr049-b2-measurement-start
(
    run_sweeps warmup "$WARMUPS"
    run_sweeps measure "$TRIALS"
) &
producer_pid=$!
monitor_status=0
thermal_monitor_fair_or_better_while_pid "$measurement_log" \
    adr049-b2-measurement "$producer_pid" "$MEASUREMENT_SAMPLE_SECONDS" \
    "$contention_measurement_log" "$SERVER_PID" || monitor_status=$?
wait "$producer_pid"
((monitor_status == 0)) || exit "$monitor_status"
thermal_sample "$measurement_log" adr049-b2-measurement-end
host_contention_sample "$contention_measurement_log" adr049-b2-measurement-end \
    "$SERVER_PID" "$THERMAL_SAMPLED_AT"
host_contention_require_quiet adr049-b2-measurement-end

server_log_end=$(file_bytes "$SERVER_LOG")
tail -c "+$((server_log_start + 1))" "$SERVER_LOG" \
    | head -c "$((server_log_end - server_log_start))" >"$server_slice"
kill -0 "$SERVER_PID" 2>/dev/null || { echo "server exited during measurement" >&2; exit 1; }
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$EXPECTED_SOURCE_SHA" \
    && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" \
    && "$(sha256_file "$HF2Q_BIN")" == "$EXPECTED_BINARY_SHA256" \
    && "$(file_bytes "$MODEL_PATH")" == "$EXPECTED_MODEL_BYTES" \
    && "$(sha256_file "$MODEL_PATH")" == "$EXPECTED_MODEL_SHA256" ]] || {
    echo "source, binary, or model identity changed during measurement" >&2
    exit 1
}

manifest="$OUT_DIR/manifest.json"
jq -n \
    --arg family "$FAMILY" --arg trace_kind "$trace_kind" \
    --arg source_sha "$EXPECTED_SOURCE_SHA" --arg binary_path "$HF2Q_BIN" \
    --arg binary_sha "$EXPECTED_BINARY_SHA256" --arg model_path "$MODEL_PATH" \
    --arg model_sha "$EXPECTED_MODEL_SHA256" --argjson model_bytes "$EXPECTED_MODEL_BYTES" \
    --arg server_command "$server_command" --arg server_model_id "$model_id" \
    --arg base_url "$BASE_URL" --argjson server_pid "$SERVER_PID" \
    --arg samples_sha "$(sha256_file "$samples")" \
    --arg server_sha "$(sha256_file "$server_slice")" \
    --arg settle_sha "$(sha256_file "$settle_log")" \
    --arg measurement_sha "$(sha256_file "$measurement_log")" \
    --arg contention_settle_sha "$(sha256_file "$contention_settle_log")" \
    --arg contention_measurement_sha "$(sha256_file "$contention_measurement_log")" \
    --arg models_sha "$(sha256_file "$models_response")" \
    --argjson trials "$TRIALS" '{
      schema_version:1,status:"measured",family:$family,trace_kind:$trace_kind,
      width_targets:[128,256,512,1024,1792],warmups:2,trials:$trials,
      order:"ascending-descending-alternating",max_slots:4,
      request_settings:{max_tokens:1,seed:42,repetition_penalty:1,stream:false,temperature:0,thinking:false},
      identity:{source_sha:$source_sha,source_dirty:false,binary_path:$binary_path,
        binary_sha256:$binary_sha,model_path:$model_path,model_sha256:$model_sha,
        model_bytes:$model_bytes,server_pid:$server_pid,server_command:$server_command,
        server_model_id:$server_model_id,base_url:$base_url},
      files:{
        samples:{path:"samples.jsonl",sha256:$samples_sha},
        models:{path:"models.json",sha256:$models_sha},
        server_log:{path:"server-measurement.log",sha256:$server_sha},
        thermal_settle:{path:"thermal-settle.log",sha256:$settle_sha},
        thermal_measurement:{path:"thermal-measurement.log",sha256:$measurement_sha},
        contention_settle:{path:"contention-settle.log",sha256:$contention_settle_sha},
        contention_measurement:{path:"contention-measurement.log",sha256:$contention_measurement_sha}
      }
    }' >"$manifest.tmp"
mv "$manifest.tmp" "$manifest"
python3 "$script_dir/verify_adr049_b2_prefill_curve.py" "$OUT_DIR" "$OUT_DIR/summary.json"
echo "ADR-049 B.2 prefill curve receipt: $OUT_DIR/summary.json" >&2
