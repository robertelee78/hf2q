#!/usr/bin/env bash
set -euo pipefail

# Exact-artifact Qwen3.8 short/long decode comparison. Each trial starts a
# fresh one-slot server from the same sealed binary, proves the scalar route
# below the crossover, then runs the same long greedy request bytes. The fixed
# ABBA order limits monotonic warmup/cooldown bias.

SOURCE_SHA=${SOURCE_SHA:?SOURCE_SHA is required}
CRATE_SHA256=${CRATE_SHA256:?CRATE_SHA256 is required}
BINARY_PATH=${BINARY_PATH:?BINARY_PATH is required}
BINARY_SHA256=${BINARY_SHA256:?BINARY_SHA256 is required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required}
PORT=${PORT:-18083}

readonly MODEL_ID='Qwen3.8 27B'
readonly MAX_TOKENS=512
readonly SHORT_MAX_TOKENS=512
readonly PROMPT_PADDING_TOKENS=105000
readonly SHORT_PROMPT_PADDING_TOKENS=4000
readonly MIN_PROMPT_TOKENS=100000
readonly MAX_PROMPT_TOKENS=120000
readonly MIN_SHORT_PROMPT_TOKENS=3000
readonly MAX_SHORT_PROMPT_TOKENS=6000
readonly MIN_IMPROVEMENT_PERCENT=15
readonly MAX_SHORT_REGRESSION_PERCENT=2
readonly MAX_WITHIN_ARM_SPREAD_PERCENT=5
readonly MAX_WALL_TIMING_DELTA_SECONDS=2
readonly TRIAL_SETTLE_SECONDS=30
readonly TRIAL_ORDER='off auto auto off'

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${HF2Q_THERMAL_SWIFTC_BIN+x} || ${HF2Q_THERMAL_PROBE_BIN+x} \
  || ${HF2Q_THERMAL_PROBE_SOURCE+x} ]]; then
  echo "thermal probe overrides are reserved for isolated contract tests" >&2
  exit 2
fi
readonly HF2Q_THERMAL_SWIFTC_BIN=/usr/bin/swiftc
[[ -x "$HF2Q_THERMAL_SWIFTC_BIN" ]] || {
  echo "required system Swift compiler is unavailable: $HF2Q_THERMAL_SWIFTC_BIN" >&2
  exit 2
}
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"

for command in awk basename cp curl date find grep jq lsof mv rg sed shasum \
  sort stat sw_vers sysctl uname wc; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || {
  echo "SOURCE_SHA must be a full lowercase Git SHA" >&2
  exit 2
}
for digest in "$CRATE_SHA256" "$BINARY_SHA256" "$MODEL_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "artifact identities must be lowercase SHA-256 digests" >&2
    exit 2
  }
done
[[ -x "$BINARY_PATH" ]] || {
  echo "sealed hf2q binary is missing or non-executable: $BINARY_PATH" >&2
  exit 2
}
[[ -f "$MODEL_PATH" ]] || {
  echo "Qwen3.8 GGUF is missing: $MODEL_PATH" >&2
  exit 2
}
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  echo "PORT must be an integer from 1 through 65535" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Qwen3.8 long-decode receipt directory must be fresh: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
file_identity() {
  stat -f '%d:%i' "$1" 2>/dev/null || stat -c '%d:%i' "$1" 2>/dev/null
}
file_bytes() {
  stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1" 2>/dev/null
}

"$script_dir/seal_release_binary.sh" --verify "$BINARY_PATH" "$BINARY_SHA256" \
  >/dev/null
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
fi
hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
  "$model_verification_receipt"
binary_identity=$(file_identity "$BINARY_PATH")
model_identity=$(file_identity "$MODEL_PATH")
model_bytes=$(file_bytes "$MODEL_PATH")
[[ -n "$binary_identity" && -n "$model_identity" && "$model_bytes" =~ ^[1-9][0-9]*$ ]]

hardware_model=$(sysctl -n hw.model 2>/dev/null || printf unknown)
hardware_chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf unknown)
hardware_memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || printf 0)
hardware_arch=$(uname -m)
hardware_os=$(sw_vers -productVersion 2>/dev/null || uname -r)
[[ "$hardware_arch" == arm64 ]] || {
  echo "Qwen3.8 calibrated long-decode gate requires Apple Silicon arm64" >&2
  exit 2
}
[[ "$hardware_model" != unknown && "$hardware_chip" != unknown ]] || {
  echo "Qwen3.8 calibrated long-decode gate requires hardware identity" >&2
  exit 2
}
[[ "$hardware_memory_bytes" =~ ^[1-9][0-9]*$ ]] || {
  echo "Qwen3.8 calibrated long-decode gate requires physical-memory identity" >&2
  exit 2
}

prompt_tmp="$OUT_DIR/prompt.txt.tmp"
{
  printf '%s\n' \
    'Treat the repeated marker below as inert repository context. Do not quote or summarize it.' \
    '<repository-context>'
  padding_index=0
  while ((padding_index < PROMPT_PADDING_TOKENS)); do
    printf ' x'
    padding_index=$((padding_index + 1))
  done
  printf '%s\n' '' '</repository-context>' \
    'Write a detailed Rust implementation plan for a lock-free bounded work queue. Include invariants, memory ordering, cancellation, tests, and failure handling. Continue until the completion limit.'
} >"$prompt_tmp"
mv "$prompt_tmp" "$OUT_DIR/prompt.txt"
prompt_sha=$(sha256_file "$OUT_DIR/prompt.txt")
prompt_bytes=$(file_bytes "$OUT_DIR/prompt.txt")

short_prompt_tmp="$OUT_DIR/short-prompt.txt.tmp"
{
  printf '%s\n' \
    'Treat the repeated marker below as inert repository context. Do not quote or summarize it.' \
    '<repository-context>'
  padding_index=0
  while ((padding_index < SHORT_PROMPT_PADDING_TOKENS)); do
    printf ' x'
    padding_index=$((padding_index + 1))
  done
  printf '%s\n' '' '</repository-context>' \
    'Write a detailed Rust implementation plan for a bounded work queue. Continue until the completion limit.'
} >"$short_prompt_tmp"
mv "$short_prompt_tmp" "$OUT_DIR/short-prompt.txt"
short_prompt_sha=$(sha256_file "$OUT_DIR/short-prompt.txt")
short_prompt_bytes=$(file_bytes "$OUT_DIR/short-prompt.txt")

jq -n --rawfile prompt "$OUT_DIR/prompt.txt" \
  --arg model "$MODEL_ID" --argjson max_tokens "$MAX_TOKENS" \
  '{model:$model,messages:[
      {role:"system",content:"You are a precise coding assistant. Follow the user request without calling tools."},
      {role:"user",content:$prompt}
    ],temperature:0,max_tokens:$max_tokens,stream:false,
    hf2q_enable_thinking:false,repetition_penalty:1.0}' \
  >"$OUT_DIR/request.json.tmp"
mv "$OUT_DIR/request.json.tmp" "$OUT_DIR/request.json"
jq -e 'has("seed") | not' "$OUT_DIR/request.json" >/dev/null
request_sha=$(sha256_file "$OUT_DIR/request.json")
jq -n --rawfile prompt "$OUT_DIR/short-prompt.txt" \
  --arg model "$MODEL_ID" --argjson max_tokens "$SHORT_MAX_TOKENS" \
  '{model:$model,messages:[
      {role:"system",content:"You are a precise coding assistant. Follow the user request without calling tools."},
      {role:"user",content:$prompt}
    ],temperature:0,max_tokens:$max_tokens,stream:false,
    hf2q_enable_thinking:false,repetition_penalty:1.0}' \
  >"$OUT_DIR/short-request.json.tmp"
mv "$OUT_DIR/short-request.json.tmp" "$OUT_DIR/short-request.json"
jq -e 'has("seed") | not' "$OUT_DIR/short-request.json" >/dev/null
short_request_sha=$(sha256_file "$OUT_DIR/short-request.json")
: >"$OUT_DIR/phase.log"

record_phase() {
  local trial_index=$1
  local mode=$2
  local event=$3
  printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$trial_index" "$mode" "$event" \
    >>"$OUT_DIR/phase.log"
}

server_pid=''
request_pid=''
stop_server() {
  local waited=0
  [[ -n "$server_pid" ]] || return 0
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid" 2>/dev/null || true
    while kill -0 "$server_pid" 2>/dev/null && ((waited < 30)); do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "$server_pid" 2>/dev/null; then
      kill -TERM "$server_pid" 2>/dev/null || true
      waited=0
      while kill -0 "$server_pid" 2>/dev/null && ((waited < 10)); do
        sleep 1
        waited=$((waited + 1))
      done
    fi
    if kill -0 "$server_pid" 2>/dev/null; then
      echo "Qwen3.8 trial server ignored bounded shutdown; killing exact child PID $server_pid" >&2
      kill -KILL "$server_pid" 2>/dev/null || true
    fi
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=''
}
cleanup() {
  local cleanup_rc=0
  if [[ -n "$request_pid" ]] && kill -0 "$request_pid" 2>/dev/null; then
    kill -TERM "$request_pid" 2>/dev/null || true
    wait "$request_pid" 2>/dev/null || true
  fi
  request_pid=''
  stop_server || cleanup_rc=1
  thermal_cleanup_probe || cleanup_rc=1
  return "$cleanup_rc"
}
on_exit() {
  local original_rc=$?
  trap - EXIT
  if ! cleanup && ((original_rc == 0)); then
    original_rc=1
  fi
  exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

thermal_prepare_probe
thermal_probe_binary_sha=$(sha256_file "$THERMAL_PROBE_BIN")
thermal_probe_source_sha=$(sha256_file "$THERMAL_PROBE_SOURCE")
thermal_probe_compiler_sha=$(sha256_file "$THERMAL_PROBE_COMPILER")
thermal_probe_compiler_version=$THERMAL_PROBE_COMPILER_VERSION

wait_ready() {
  local log_path=$1
  local deadline=$((SECONDS + 600))
  while ((SECONDS < deadline)); do
    if curl --fail --silent --show-error --max-time 2 \
      "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Qwen3.8 trial server exited before readiness" >&2
      sed -n '1,240p' "$log_path" >&2
      return 1
    fi
    sleep 1
  done
  echo "Qwen3.8 trial server did not become ready within 600 seconds" >&2
  sed -n '1,240p' "$log_path" >&2
  return 1
}

run_trial() {
  local trial_index=$1
  local mode=$2
  local expected_prewarm=$3
  local trial_dir="$OUT_DIR/trial-${trial_index}-${mode}"
  local response="$trial_dir/response.json"
  local server_log="$trial_dir/server.log"
  local short_response="$trial_dir/short-response.json"
  local short_server_log="$trial_dir/short-server.log"
  local decode_line log_generated log_elapsed_ms log_tps
  local prompt_tokens completion_tokens decode_seconds decode_tps finish_reason
  local response_total_seconds wall_total_seconds
  local short_decode_line short_log_generated short_log_elapsed_ms short_log_tps
  local short_prompt_tokens short_completion_tokens short_decode_seconds short_decode_tps
  local short_finish_reason short_response_total_seconds short_wall_total_seconds
  local short_semantic_sha
  local semantic_sha ready_code
  local artifacts_json

  mkdir -p "$trial_dir"
  cp "$OUT_DIR/request.json" "$trial_dir/request.json"
  cp "$OUT_DIR/short-request.json" "$trial_dir/short-request.json"
  printf 'HF2Q_QWEN_GQA_Q2=%s\nHF2Q_PIPELINE_PREWARM_LOG=1\nQWEN38_VISION=off\n' \
    "$mode" >"$trial_dir/environment.txt"
  thermal_wait_for_nominal "$trial_dir/settle.log" \
    "qwen38-trial-${trial_index}-${mode}-settle" \
    "$TRIAL_SETTLE_SECONDS" 900 5
  record_phase "$trial_index" "$mode" trial-start

  "$script_dir/seal_release_binary.sh" --verify "$BINARY_PATH" "$BINARY_SHA256" \
    >/dev/null
  test "$(file_identity "$BINARY_PATH")" = "$binary_identity"
  test "$(file_identity "$MODEL_PATH")" = "$model_identity"
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | rg -q .; then
    echo "Qwen3.8 benchmark port is already in use: $PORT" >&2
    return 1
  fi

  env MODEL="$MODEL_PATH" PORT="$PORT" HF2Q_BIN="$BINARY_PATH" MAX_SLOTS=1 \
    QWEN38_VISION=off THINKING_TOKEN_BUDGET=0 REP_PENALTY=1.0 \
    HF2Q_QWEN_GQA_Q2="$mode" HF2Q_PIPELINE_PREWARM_LOG=1 \
    "$script_dir/serve_qwen38_opencode.sh" >"$server_log" 2>&1 &
  server_pid=$!
  wait_ready "$server_log"

  curl --fail --silent --show-error --connect-timeout 5 --max-time 10 \
    "http://127.0.0.1:$PORT/v1/models" -o "$trial_dir/models.json"
  jq -e --arg model "$MODEL_ID" '
    (.object == "list")
    and ([.data[] | select(.loaded == true) | .id] == [$model])
  ' "$trial_dir/models.json" >/dev/null || {
    echo "Qwen3.8 trial server did not expose exactly the sealed loaded model: $MODEL_ID" >&2
    return 1
  }

  record_phase "$trial_index" "$mode" short-request-start
  curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 900 \
    -H 'Content-Type: application/json' \
    --data-binary "@$trial_dir/short-request.json" \
    -o "$short_response" \
    -w 'http_code=%{http_code}\ntotal_seconds=%{time_total}\n' \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    >"$trial_dir/short-curl.metrics" &
  request_pid=$!
  wait "$request_pid"
  request_pid=''
  record_phase "$trial_index" "$mode" short-request-end
  grep -qx 'http_code=200' "$trial_dir/short-curl.metrics"
  short_wall_total_seconds=$(sed -n 's/^total_seconds=//p' "$trial_dir/short-curl.metrics")
  [[ "$short_wall_total_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]
  jq -e --arg model "$MODEL_ID" --argjson max_tokens "$SHORT_MAX_TOKENS" \
    --argjson min_prompt "$MIN_SHORT_PROMPT_TOKENS" \
    --argjson max_prompt "$MAX_SHORT_PROMPT_TOKENS" '
      .model == $model
      and (.choices | type) == "array" and (.choices | length) == 1
      and .choices[0].finish_reason == "length"
      and .choices[0].message.role == "assistant"
      and (.choices[0].message.content | type) == "string"
      and (.choices[0].message.content | length) > 0
      and (.usage.prompt_tokens >= $min_prompt)
      and (.usage.prompt_tokens <= $max_prompt)
      and .usage.completion_tokens == $max_tokens
      and .usage.total_tokens == (.usage.prompt_tokens + .usage.completion_tokens)
      and (.x_hf2q_timing.decode_time_secs | type) == "number"
      and .x_hf2q_timing.decode_time_secs > 0
      and (.x_hf2q_timing.decode_tokens_per_sec | type) == "number"
      and .x_hf2q_timing.decode_tokens_per_sec > 0
      and (.x_hf2q_timing.total_time_secs | type) == "number"
      and .x_hf2q_timing.total_time_secs > 0
    ' "$short_response" >/dev/null
  jq -S '{model,choices,usage}' "$short_response" >"$trial_dir/short-semantic.json"
  short_prompt_tokens=$(jq -er '.usage.prompt_tokens' "$short_response")
  short_completion_tokens=$(jq -er '.usage.completion_tokens' "$short_response")
  short_decode_seconds=$(jq -er '.x_hf2q_timing.decode_time_secs' "$short_response")
  short_decode_tps=$(jq -er '.x_hf2q_timing.decode_tokens_per_sec' "$short_response")
  short_response_total_seconds=$(jq -er '.x_hf2q_timing.total_time_secs' "$short_response")
  short_finish_reason=$(jq -er '.choices[0].finish_reason' "$short_response")
  short_semantic_sha=$(sha256_file "$trial_dir/short-semantic.json")
  awk -v wall="$short_wall_total_seconds" -v response="$short_response_total_seconds" \
    -v tolerance="$MAX_WALL_TIMING_DELTA_SECONDS" '
      BEGIN {
        delta = wall - response
        if (delta < 0) delta = -delta
        exit !(wall > 0 && response > 0 && delta <= tolerance)
      }
    '
  cp "$server_log" "$short_server_log"
  [[ "$(rg -c 'Qwen35 decode complete' "$short_server_log")" == 1 ]]
  short_decode_line=$(rg 'Qwen35 decode complete' "$short_server_log")
  short_log_generated=$(sed -n 's/.*generated_tokens=\([^ ]*\).*/\1/p' <<<"$short_decode_line")
  short_log_elapsed_ms=$(sed -n 's/.*elapsed_ms=\([^ ]*\).*/\1/p' <<<"$short_decode_line")
  short_log_tps=$(sed -n 's/.*tokens_per_second=\([^ ]*\).*/\1/p' <<<"$short_decode_line")
  [[ "$short_log_generated" == "$short_completion_tokens" ]]
  [[ "$short_log_elapsed_ms" =~ ^[0-9]+([.][0-9]+)?$ ]]
  [[ "$short_log_tps" =~ ^[0-9]+([.][0-9]+)?$ ]]
  awk -v observed="$short_decode_tps" -v logged="$short_log_tps" \
    'BEGIN { delta=observed-logged; if (delta<0) delta=-delta; exit !(delta <= 0.2) }'
  if rg -Fq 'Qwen TQ-HB decode selected GQA-cooperative Q2 attention' \
    "$short_server_log"; then
    echo "Qwen3.8 short trial crossed into the GQA Q2 candidate" >&2
    return 1
  fi

  record_phase "$trial_index" "$mode" request-start
  curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 1800 \
    -H 'Content-Type: application/json' \
    --data-binary "@$trial_dir/request.json" \
    -o "$response" \
    -w 'http_code=%{http_code}\ntotal_seconds=%{time_total}\n' \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    >"$trial_dir/curl.metrics" &
  request_pid=$!
  wait "$request_pid"
  request_pid=''
  record_phase "$trial_index" "$mode" request-end
  grep -qx 'http_code=200' "$trial_dir/curl.metrics"
  wall_total_seconds=$(sed -n 's/^total_seconds=//p' "$trial_dir/curl.metrics")
  [[ "$wall_total_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]
  ready_code=$(curl --silent --show-error --max-time 3 \
    -o "$trial_dir/readyz.json" -w '%{http_code}' \
    "http://127.0.0.1:$PORT/readyz")
  [[ "$ready_code" == 200 ]]
  stop_server
  record_phase "$trial_index" "$mode" trial-end

  jq -e --arg model "$MODEL_ID" --argjson max_tokens "$MAX_TOKENS" \
    --argjson min_prompt "$MIN_PROMPT_TOKENS" \
    --argjson max_prompt "$MAX_PROMPT_TOKENS" '
      .model == $model
      and (.choices | type) == "array" and (.choices | length) == 1
      and .choices[0].finish_reason == "length"
      and .choices[0].message.role == "assistant"
      and (.choices[0].message.content | type) == "string"
      and (.choices[0].message.content | length) > 0
      and (.usage.prompt_tokens >= $min_prompt)
      and (.usage.prompt_tokens <= $max_prompt)
      and .usage.completion_tokens == $max_tokens
      and .usage.total_tokens == (.usage.prompt_tokens + .usage.completion_tokens)
      and (.x_hf2q_timing.decode_time_secs | type) == "number"
      and .x_hf2q_timing.decode_time_secs > 0
      and (.x_hf2q_timing.decode_tokens_per_sec | type) == "number"
      and .x_hf2q_timing.decode_tokens_per_sec > 0
      and (.x_hf2q_timing.total_time_secs | type) == "number"
      and .x_hf2q_timing.total_time_secs > 0
    ' "$response" >/dev/null
  jq -S '{model,choices,usage}' "$response" >"$trial_dir/semantic.json"

  prompt_tokens=$(jq -er '.usage.prompt_tokens' "$response")
  completion_tokens=$(jq -er '.usage.completion_tokens' "$response")
  decode_seconds=$(jq -er '.x_hf2q_timing.decode_time_secs' "$response")
  decode_tps=$(jq -er '.x_hf2q_timing.decode_tokens_per_sec' "$response")
  response_total_seconds=$(jq -er '.x_hf2q_timing.total_time_secs' "$response")
  finish_reason=$(jq -er '.choices[0].finish_reason' "$response")
  semantic_sha=$(sha256_file "$trial_dir/semantic.json")
  awk -v wall="$wall_total_seconds" -v response="$response_total_seconds" \
    -v tolerance="$MAX_WALL_TIMING_DELTA_SECONDS" '
      BEGIN {
        delta = wall - response
        if (delta < 0) delta = -delta
        exit !(wall > 0 && response > 0 && delta <= tolerance)
      }
    '

  [[ "$(rg -c 'Qwen35 decode complete' "$server_log")" == 2 ]]
  decode_line=$(rg 'Qwen35 decode complete' "$server_log" | tail -n 1)
  log_generated=$(sed -n 's/.*generated_tokens=\([^ ]*\).*/\1/p' <<<"$decode_line")
  log_elapsed_ms=$(sed -n 's/.*elapsed_ms=\([^ ]*\).*/\1/p' <<<"$decode_line")
  log_tps=$(sed -n 's/.*tokens_per_second=\([^ ]*\).*/\1/p' <<<"$decode_line")
  [[ "$log_generated" == "$completion_tokens" ]]
  [[ "$log_elapsed_ms" =~ ^[0-9]+([.][0-9]+)?$ ]]
  [[ "$log_tps" =~ ^[0-9]+([.][0-9]+)?$ ]]
  awk -v observed="$decode_tps" -v logged="$log_tps" \
    'BEGIN { delta=observed-logged; if (delta<0) delta=-delta; exit !(delta <= 0.2) }'
  rg -Fq "+ gqa_q2=$expected_prewarm " "$server_log"
  if [[ "$mode" == auto ]]; then
    rg -Fq 'Qwen TQ-HB decode selected GQA-cooperative Q2 attention' "$server_log"
  elif rg -Fq 'Qwen TQ-HB decode selected GQA-cooperative Q2 attention' \
    "$server_log"; then
    echo "Qwen3.8 off trial selected the GQA Q2 candidate" >&2
    return 1
  fi
  if grep -Eiq 'GPU Timeout|SubmissionsIgnored|Command buffer error|Generation error|engine_unhealthy|panicked at|worker-fatal' "$server_log"; then
    echo "Qwen3.8 long-decode trial observed a fatal runtime signature" >&2
    return 1
  fi
  if rg -Fq 'auto-pipeline: downloading from HF Hub' "$server_log"; then
    echo "Qwen3.8 long-decode trial escaped the sealed loaded-model path" >&2
    return 1
  fi

  artifacts_json=$(
    for name in request.json response.json semantic.json curl.metrics server.log \
      short-request.json short-response.json short-semantic.json short-curl.metrics \
      short-server.log readyz.json models.json environment.txt settle.log; do
      jq -n --arg name "$name" --arg sha256 "$(sha256_file "$trial_dir/$name")" \
        '{name:$name,sha256:$sha256}'
    done | jq -s .
  )
  jq -n --argjson index "$trial_index" --arg mode "$mode" \
    --arg status pass --arg binary_sha256 "$BINARY_SHA256" \
    --arg binary_file_identity "$binary_identity" \
    --arg model_sha256 "$MODEL_SHA256" --arg model_file_identity "$model_identity" \
    --arg request_sha256 "$request_sha" --arg semantic_sha256 "$semantic_sha" \
    --arg short_request_sha256 "$short_request_sha" \
    --arg short_semantic_sha256 "$short_semantic_sha" \
    --arg finish_reason "$finish_reason" --argjson prompt_tokens "$prompt_tokens" \
    --argjson completion_tokens "$completion_tokens" \
    --argjson decode_seconds "$decode_seconds" --argjson decode_tps "$decode_tps" \
    --argjson response_total_seconds "$response_total_seconds" \
    --argjson wall_total_seconds "$wall_total_seconds" \
    --arg short_finish_reason "$short_finish_reason" \
    --argjson short_prompt_tokens "$short_prompt_tokens" \
    --argjson short_completion_tokens "$short_completion_tokens" \
    --argjson short_decode_seconds "$short_decode_seconds" \
    --argjson short_decode_tps "$short_decode_tps" \
    --argjson short_response_total_seconds "$short_response_total_seconds" \
    --argjson short_wall_total_seconds "$short_wall_total_seconds" \
    --argjson artifacts "$artifacts_json" \
    '{index:$index,mode:$mode,status:$status,binary_sha256:$binary_sha256,
      binary_file_identity:$binary_file_identity,model_sha256:$model_sha256,
      model_file_identity:$model_file_identity,request_sha256:$request_sha256,
      semantic_sha256:$semantic_sha256,prompt_tokens:$prompt_tokens,
      completion_tokens:$completion_tokens,finish_reason:$finish_reason,
      decode_seconds:$decode_seconds,decode_tokens_per_second:$decode_tps,
      response_total_seconds:$response_total_seconds,
      wall_total_seconds:$wall_total_seconds,
      short:{request_sha256:$short_request_sha256,
        semantic_sha256:$short_semantic_sha256,prompt_tokens:$short_prompt_tokens,
        completion_tokens:$short_completion_tokens,finish_reason:$short_finish_reason,
        decode_seconds:$short_decode_seconds,
        decode_tokens_per_second:$short_decode_tps,
        response_total_seconds:$short_response_total_seconds,
        wall_total_seconds:$short_wall_total_seconds},
      artifacts:$artifacts}' >"$trial_dir/trial.json.tmp"
  mv "$trial_dir/trial.json.tmp" "$trial_dir/trial.json"
}

trial_index=0
for mode in $TRIAL_ORDER; do
  trial_index=$((trial_index + 1))
  if [[ "$mode" == auto ]]; then
    run_trial "$trial_index" "$mode" true
  else
    run_trial "$trial_index" "$mode" false
  fi
done

off_a=$(jq -er .decode_tokens_per_second "$OUT_DIR/trial-1-off/trial.json")
off_b=$(jq -er .decode_tokens_per_second "$OUT_DIR/trial-4-off/trial.json")
auto_a=$(jq -er .decode_tokens_per_second "$OUT_DIR/trial-2-auto/trial.json")
auto_b=$(jq -er .decode_tokens_per_second "$OUT_DIR/trial-3-auto/trial.json")
short_off_a=$(jq -er .short.decode_tokens_per_second "$OUT_DIR/trial-1-off/trial.json")
short_off_b=$(jq -er .short.decode_tokens_per_second "$OUT_DIR/trial-4-off/trial.json")
short_auto_a=$(jq -er .short.decode_tokens_per_second "$OUT_DIR/trial-2-auto/trial.json")
short_auto_b=$(jq -er .short.decode_tokens_per_second "$OUT_DIR/trial-3-auto/trial.json")
off_mean=$(awk -v a="$off_a" -v b="$off_b" \
  'BEGIN { printf "%.6f", (a+b)/2 }')
auto_mean=$(awk -v a="$auto_a" -v b="$auto_b" \
  'BEGIN { printf "%.6f", (a+b)/2 }')
short_off_mean=$(awk -v a="$short_off_a" -v b="$short_off_b" \
  'BEGIN { printf "%.6f", (a+b)/2 }')
short_auto_mean=$(awk -v a="$short_auto_a" -v b="$short_auto_b" \
  'BEGIN { printf "%.6f", (a+b)/2 }')
off_spread_percent=$(awk -v a="$off_a" -v b="$off_b" -v mean="$off_mean" '
  BEGIN { delta=a-b; if (delta<0) delta=-delta; printf "%.6f", delta/mean*100 }')
auto_spread_percent=$(awk -v a="$auto_a" -v b="$auto_b" -v mean="$auto_mean" '
  BEGIN { delta=a-b; if (delta<0) delta=-delta; printf "%.6f", delta/mean*100 }')
short_off_spread_percent=$(awk -v a="$short_off_a" -v b="$short_off_b" -v mean="$short_off_mean" '
  BEGIN { delta=a-b; if (delta<0) delta=-delta; printf "%.6f", delta/mean*100 }')
short_auto_spread_percent=$(awk -v a="$short_auto_a" -v b="$short_auto_b" -v mean="$short_auto_mean" '
  BEGIN { delta=a-b; if (delta<0) delta=-delta; printf "%.6f", delta/mean*100 }')
for spread in "$off_spread_percent" "$auto_spread_percent" \
  "$short_off_spread_percent" "$short_auto_spread_percent"; do
  awk -v observed="$spread" -v maximum="$MAX_WITHIN_ARM_SPREAD_PERCENT" \
    'BEGIN { exit !(observed <= maximum) }' || {
      echo "Qwen3.8 within-arm decode spread ${spread}% exceeds ${MAX_WITHIN_ARM_SPREAD_PERCENT}%" >&2
      exit 1
    }
done
improvement_percent=$(awk -v baseline="$off_mean" -v candidate="$auto_mean" \
  'BEGIN { if (baseline <= 0) exit 1; printf "%.6f", ((candidate / baseline) - 1) * 100 }')
short_regression_percent=$(awk -v baseline="$short_off_mean" -v candidate="$short_auto_mean" \
  'BEGIN { if (baseline <= 0) exit 1; printf "%.6f", (1 - (candidate / baseline)) * 100 }')
awk -v observed="$improvement_percent" -v required="$MIN_IMPROVEMENT_PERCENT" \
  'BEGIN { exit !(observed >= required) }' || {
    echo "Qwen3.8 GQA Q2 improvement ${improvement_percent}% is below ${MIN_IMPROVEMENT_PERCENT}%" >&2
    exit 1
  }
awk -v observed="$short_regression_percent" -v maximum="$MAX_SHORT_REGRESSION_PERCENT" \
  'BEGIN { exit !(observed <= maximum) }' || {
    echo "Qwen3.8 short-context regression ${short_regression_percent}% exceeds ${MAX_SHORT_REGRESSION_PERCENT}%" >&2
    exit 1
  }

off_wall_mean=$(awk \
  -v a="$(jq -er .wall_total_seconds "$OUT_DIR/trial-1-off/trial.json")" \
  -v b="$(jq -er .wall_total_seconds "$OUT_DIR/trial-4-off/trial.json")" \
  'BEGIN { printf "%.6f", (a+b)/2 }')
auto_wall_mean=$(awk \
  -v a="$(jq -er .wall_total_seconds "$OUT_DIR/trial-2-auto/trial.json")" \
  -v b="$(jq -er .wall_total_seconds "$OUT_DIR/trial-3-auto/trial.json")" \
  'BEGIN { printf "%.6f", (a+b)/2 }')
awk -v baseline="$off_wall_mean" -v candidate="$auto_wall_mean" \
  'BEGIN { exit !(candidate < baseline) }' || {
    echo "Qwen3.8 candidate did not reduce independently measured request wall time" >&2
    exit 1
  }

semantic_sha=$(jq -er .semantic_sha256 "$OUT_DIR/trial-1-off/trial.json")
short_semantic_sha=$(jq -er .short.semantic_sha256 "$OUT_DIR/trial-1-off/trial.json")
for trial in \
  "$OUT_DIR/trial-2-auto/trial.json" \
  "$OUT_DIR/trial-3-auto/trial.json" \
  "$OUT_DIR/trial-4-off/trial.json"; do
  test "$(jq -er .semantic_sha256 "$trial")" = "$semantic_sha"
  test "$(jq -er .short.semantic_sha256 "$trial")" = "$short_semantic_sha"
done

"$script_dir/seal_release_binary.sh" --verify "$BINARY_PATH" "$BINARY_SHA256" \
  >/dev/null
test "$(file_identity "$BINARY_PATH")" = "$binary_identity"
test "$(file_identity "$MODEL_PATH")" = "$model_identity"
hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
  "$model_verification_receipt"
phase_sha=$(sha256_file "$OUT_DIR/phase.log")
phase_bytes=$(file_bytes "$OUT_DIR/phase.log")

jq -n --arg status pass --arg benchmark qwen38-long-decode-gqa-q2 \
  --arg source_sha "$SOURCE_SHA" --arg crate_sha256 "$CRATE_SHA256" \
  --arg binary_path "$BINARY_PATH" --arg binary_sha256 "$BINARY_SHA256" \
  --arg binary_file_identity "$binary_identity" --arg model_id "$MODEL_ID" \
  --arg model_path "$MODEL_PATH" --arg model_sha256 "$MODEL_SHA256" \
  --arg model_file_identity "$model_identity" --argjson model_bytes "$model_bytes" \
  --arg prompt_sha256 "$prompt_sha" --argjson prompt_bytes "$prompt_bytes" \
  --arg short_prompt_sha256 "$short_prompt_sha" \
  --argjson short_prompt_bytes "$short_prompt_bytes" \
  --arg phase_sha256 "$phase_sha" --argjson phase_bytes "$phase_bytes" \
  --arg request_sha256 "$request_sha" --arg hardware_model "$hardware_model" \
  --arg short_request_sha256 "$short_request_sha" \
  --arg hardware_chip "$hardware_chip" --arg hardware_arch "$hardware_arch" \
  --arg hardware_os "$hardware_os" --argjson hardware_memory_bytes "$hardware_memory_bytes" \
  --arg thermal_probe_source_sha256 "$thermal_probe_source_sha" \
  --arg thermal_probe_compiler_path "$THERMAL_PROBE_COMPILER" \
  --arg thermal_probe_compiler_sha256 "$thermal_probe_compiler_sha" \
  --arg thermal_probe_compiler_version "$thermal_probe_compiler_version" \
  --arg thermal_probe_binary_sha256 "$thermal_probe_binary_sha" \
  --argjson max_tokens "$MAX_TOKENS" \
  --argjson short_max_tokens "$SHORT_MAX_TOKENS" \
  --argjson prompt_padding_tokens "$PROMPT_PADDING_TOKENS" \
  --argjson short_prompt_padding_tokens "$SHORT_PROMPT_PADDING_TOKENS" \
  --argjson min_prompt_tokens "$MIN_PROMPT_TOKENS" \
  --argjson max_prompt_tokens "$MAX_PROMPT_TOKENS" \
  --argjson min_short_prompt_tokens "$MIN_SHORT_PROMPT_TOKENS" \
  --argjson max_short_prompt_tokens "$MAX_SHORT_PROMPT_TOKENS" \
  --argjson trial_settle_seconds "$TRIAL_SETTLE_SECONDS" \
  --argjson minimum_improvement_percent "$MIN_IMPROVEMENT_PERCENT" \
  --argjson maximum_short_regression_percent "$MAX_SHORT_REGRESSION_PERCENT" \
  --argjson maximum_within_arm_spread_percent "$MAX_WITHIN_ARM_SPREAD_PERCENT" \
  --argjson maximum_wall_timing_delta_seconds "$MAX_WALL_TIMING_DELTA_SECONDS" \
  --argjson off_mean "$off_mean" --argjson auto_mean "$auto_mean" \
  --argjson off_spread_percent "$off_spread_percent" \
  --argjson auto_spread_percent "$auto_spread_percent" \
  --argjson off_wall_mean "$off_wall_mean" \
  --argjson auto_wall_mean "$auto_wall_mean" \
  --argjson improvement_percent "$improvement_percent" \
  --argjson short_off_mean "$short_off_mean" \
  --argjson short_auto_mean "$short_auto_mean" \
  --argjson short_off_spread_percent "$short_off_spread_percent" \
  --argjson short_auto_spread_percent "$short_auto_spread_percent" \
  --argjson short_regression_percent "$short_regression_percent" \
  --arg semantic_sha256 "$semantic_sha" \
  --arg short_semantic_sha256 "$short_semantic_sha" \
  --slurpfile trial1 "$OUT_DIR/trial-1-off/trial.json" \
  --slurpfile trial2 "$OUT_DIR/trial-2-auto/trial.json" \
  --slurpfile trial3 "$OUT_DIR/trial-3-auto/trial.json" \
  --slurpfile trial4 "$OUT_DIR/trial-4-off/trial.json" \
  '{schema_version:2,status:$status,benchmark:$benchmark,
    identity:{source_sha:$source_sha,crate_sha256:$crate_sha256,
      binary:{path:$binary_path,sha256:$binary_sha256,file_identity:$binary_file_identity},
      model:{id:$model_id,path:$model_path,sha256:$model_sha256,
        file_identity:$model_file_identity,bytes:$model_bytes},
      prompt:{path:"prompt.txt",sha256:$prompt_sha256,bytes:$prompt_bytes,
        padding_tokens:$prompt_padding_tokens},
      short_prompt:{path:"short-prompt.txt",sha256:$short_prompt_sha256,
        bytes:$short_prompt_bytes,padding_tokens:$short_prompt_padding_tokens},
      phase_log:{path:"phase.log",sha256:$phase_sha256,bytes:$phase_bytes},
      request:{path:"request.json",sha256:$request_sha256},
      short_request:{path:"short-request.json",sha256:$short_request_sha256},
      hardware:{model:$hardware_model,chip:$hardware_chip,arch:$hardware_arch,
        memory_bytes:$hardware_memory_bytes,os_version:$hardware_os,
        thermal_probe:{implementation:"compiled-foundation-helper",
          source_path:"scripts/macos_thermal_probe.swift",
          source_sha256:$thermal_probe_source_sha256,
          compiler_path:$thermal_probe_compiler_path,
          compiler_sha256:$thermal_probe_compiler_sha256,
          compiler_version:$thermal_probe_compiler_version,
          binary_sha256:$thermal_probe_binary_sha256}}},
    settings:{temperature:0,max_tokens:$max_tokens,short_max_tokens:$short_max_tokens,
      stream:false,
      thinking:false,repetition_penalty:1.0,min_prompt_tokens:$min_prompt_tokens,
      max_prompt_tokens:$max_prompt_tokens,
      min_short_prompt_tokens:$min_short_prompt_tokens,
      max_short_prompt_tokens:$max_short_prompt_tokens,
      trial_settle_seconds:$trial_settle_seconds,
      maximum_within_arm_spread_percent:$maximum_within_arm_spread_percent,
      maximum_wall_timing_delta_seconds:$maximum_wall_timing_delta_seconds},
    trial_order:["off","auto","auto","off"],
    trials:[$trial1[0],$trial2[0],$trial3[0],$trial4[0]],
    aggregate:{off_mean_decode_tokens_per_second:$off_mean,
      auto_mean_decode_tokens_per_second:$auto_mean,
      off_within_arm_spread_percent:$off_spread_percent,
      auto_within_arm_spread_percent:$auto_spread_percent,
      off_mean_wall_seconds:$off_wall_mean,
      auto_mean_wall_seconds:$auto_wall_mean,
      improvement_percent:$improvement_percent,
      minimum_improvement_percent:$minimum_improvement_percent,
      exact_output_sha256:$semantic_sha256,
      short_off_mean_decode_tokens_per_second:$short_off_mean,
      short_auto_mean_decode_tokens_per_second:$short_auto_mean,
      short_off_within_arm_spread_percent:$short_off_spread_percent,
      short_auto_within_arm_spread_percent:$short_auto_spread_percent,
      short_regression_percent:$short_regression_percent,
      maximum_short_regression_percent:$maximum_short_regression_percent,
      short_exact_output_sha256:$short_semantic_sha256}}' >"$OUT_DIR/summary.json.tmp"
mv "$OUT_DIR/summary.json.tmp" "$OUT_DIR/summary.json"
printf '%s  summary.json\n' "$(sha256_file "$OUT_DIR/summary.json")" \
  >"$OUT_DIR/summary.json.sha256"

bash "$script_dir/verify_qwen38_long_decode_receipt.sh" benchmark "$OUT_DIR" \
  "$SOURCE_SHA" "$CRATE_SHA256" "$BINARY_SHA256" "$MODEL_SHA256"

echo "Qwen3.8 long-decode ABBA benchmark evidence: $OUT_DIR/summary.json" >&2
echo "Release authority additionally requires the thermally supervised release envelope" >&2
