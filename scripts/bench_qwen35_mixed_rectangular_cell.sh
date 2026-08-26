#!/usr/bin/env bash
set -euo pipefail

# ADR-049 Qwen family mixed-workload gate.  One invocation exercises one
# artifact shape under fresh OFF-A/ON-A/ON-B/OFF-B processes.  A semantic SSE
# decoder is established before four cold compatible prefills are released;
# the ON arms must publish one width-four transaction while the decoder makes
# visible progress before, during, and after that transaction.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
# shellcheck source=scripts/agentic_cache_lifecycle_contract.sh
source "$script_dir/agentic_cache_lifecycle_contract.sh"
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"
# shellcheck source=scripts/qwen35_mixed_rectangular_contract.sh
source "$script_dir/qwen35_mixed_rectangular_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/qwen38_matched_reference_contract.sh
source "$script_dir/qwen38_matched_reference_contract.sh"

SOURCE_ROOT=${SOURCE_ROOT:-$root_dir}
HF2Q_BIN=${HF2Q_BIN:-$SOURCE_ROOT/target/release/hf2q}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
MODEL_BYTES=${MODEL_BYTES:?MODEL_BYTES is required}
MODEL_SHAPE=${MODEL_SHAPE:?MODEL_SHAPE must be qwen38-dense or qwen36-moe}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-52949}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-240}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-180}
THERMAL_SETTLE_TIMEOUT_SECONDS=${THERMAL_SETTLE_TIMEOUT_SECONDS:-600}
readonly TRIALS=5
readonly PREFILL_LANES=4
readonly MAX_SLOTS=8
readonly PREFILL_MAX_TOKENS=2
readonly DECODER_MAX_TOKENS=512
readonly COALESCE_US=25000
readonly KV_CACHE_BUDGET_BYTES=51539607552
readonly THERMAL_SETTLE_SECONDS=60
readonly THERMAL_SAMPLE_SECONDS=2
readonly POWER_PROBE_ATTEMPTS=3
readonly MIN_MIXED_SPEEDUP=1.01
readonly MIN_SEMANTIC_EVENTS=3
readonly MAX_DECODER_TTFT_MS=15000
readonly MAX_SEMANTIC_GAP_MS=15000
readonly MAX_PREFILL_TAIL_MS=60000
readonly MAX_LAUNCH_SKEW_MS=100
readonly RUNTIME_HOME=/var/empty
readonly RUNTIME_PATH=/usr/bin:/bin:/usr/sbin:/sbin
readonly RUNTIME_TMPDIR=/var/tmp

for command in awk caffeinate cat cmp curl date env find git grep head jq lsof \
  mkdir mv perl pgrep pmset ps rg rm sed seq shasum sleep sort stat \
  system_profiler tail tr wc; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ "$SOURCE_ROOT" == /* && "$HF2Q_BIN" == /* && "$MODEL_PATH" == /* \
  && "$OUT_DIR" == /* ]] || {
  echo "SOURCE_ROOT, HF2Q_BIN, MODEL_PATH, and OUT_DIR must be absolute" >&2
  exit 2
}
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd -P)
[[ -d "$SOURCE_ROOT/.git" || -f "$SOURCE_ROOT/.git" ]] || {
  echo "SOURCE_ROOT is not a Git worktree: $SOURCE_ROOT" >&2
  exit 2
}
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
  echo "mixed Qwen gate requires a clean exact source tree" >&2
  exit 2
}
source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$HF2Q_BIN" == "$SOURCE_ROOT/target/release/hf2q" && -x "$HF2Q_BIN" ]] || {
  echo "HF2Q_BIN must be the exact source tree's release output" >&2
  exit 2
}
grep -aFq "$source_commit" "$HF2Q_BIN" || {
  echo "release binary does not embed source commit $source_commit" >&2
  exit 2
}
binary_sha256=$(shasum -a 256 "$HF2Q_BIN" | awk '{print $1}')
[[ "$MODEL_PATH" != "$SOURCE_ROOT"/* && -f "$MODEL_PATH" \
  && -r "$MODEL_PATH" && ! -L "$MODEL_PATH" ]] || exit 2
[[ "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ && "$MODEL_BYTES" =~ ^[1-9][0-9]*$ \
  && "$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')" == "$MODEL_SHA256" \
  && "$(stat -f '%z' "$MODEL_PATH" 2>/dev/null || stat -c '%s' "$MODEL_PATH")" == "$MODEL_BYTES" ]] || {
  echo "model identity does not match MODEL_SHA256/MODEL_BYTES" >&2
  exit 2
}
model_snapshot=$(stat -f '%d:%i:%z:%m:%c' "$MODEL_PATH" 2>/dev/null \
  || stat -c '%d:%i:%s:%Y:%Z' "$MODEL_PATH")
case "$MODEL_SHAPE" in
  qwen38-dense)
    expected_arch=qwen35
    expected_mtp=succeeded
    expected_mtp_bool=true
    ;;
  qwen36-moe)
    expected_arch=qwen35moe
    expected_mtp=not-requested
    expected_mtp_bool=false
    ;;
  *) echo "MODEL_SHAPE must be qwen38-dense or qwen36-moe" >&2; exit 2 ;;
esac
for value_name in PORT READY_TIMEOUT_SECONDS REQUEST_TIMEOUT_SECONDS \
  THERMAL_SETTLE_TIMEOUT_SECONDS; do
  value=${!value_name}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2
done
((PORT <= 65535)) || exit 2
[[ ! -e "$OUT_DIR" \
  || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
  echo "OUT_DIR must be fresh: $OUT_DIR" >&2
  exit 2
}
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd -P)
case "$OUT_DIR/" in
  "$SOURCE_ROOT"/*) echo "OUT_DIR must be outside SOURCE_ROOT" >&2; exit 2 ;;
esac
[[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]] || {
  echo "$HOST:$PORT is already in use" >&2
  exit 2
}
if pgrep -x hf2q >/dev/null 2>&1; then
  echo "mixed Qwen gate requires no pre-existing hf2q runtime" >&2
  pgrep -flx hf2q >&2 || true
  exit 2
fi

resolve_ac_energy_mode() {
  local attempt observed
  for ((attempt = 1; attempt <= POWER_PROBE_ATTEMPTS; attempt++)); do
    if observed=$(LANG=C LC_ALL=C system_profiler SPPowerDataType \
      | matched_parse_ac_power_mode); then
      printf '%s\n' "$observed"
      return 0
    fi
    ((attempt == POWER_PROBE_ATTEMPTS)) || sleep 1
  done
  return 1
}

resolve_live_power_mode_code() {
  local attempt observed
  for ((attempt = 1; attempt <= POWER_PROBE_ATTEMPTS; attempt++)); do
    if observed=$(pmset -g live | matched_parse_live_power_mode_code); then
      printf '%s\n' "$observed"
      return 0
    fi
    ((attempt == POWER_PROBE_ATTEMPTS)) || sleep 1
  done
  return 1
}

resolve_live_power_source() {
  local attempt observed report
  for ((attempt = 1; attempt <= POWER_PROBE_ATTEMPTS; attempt++)); do
    if report=$(pmset -g batt 2>/dev/null) \
      && observed=$(matched_parse_live_power_source <<<"$report"); then
      printf '%s\n' "$observed"
      return 0
    fi
    ((attempt == POWER_PROBE_ATTEMPTS)) || sleep 1
  done
  return 1
}

power_source=$(resolve_live_power_source) || exit 2
[[ "$power_source" == ac ]] || {
  echo "mixed Qwen gate requires AC power" >&2
  exit 2
}
power_mode=$(resolve_ac_energy_mode) || exit 2
[[ "$power_mode" != low ]] || { echo "Low Power Mode is not accepted" >&2; exit 2; }
power_mode_code=$(resolve_live_power_mode_code) || exit 2

record_power_contract() {
  local output=$1 phase=$2 observed_source observed_mode observed_code sampled_at
  observed_source=$(resolve_live_power_source) || return 1
  [[ "$observed_source" == ac ]] || return 1
  observed_mode=$(resolve_ac_energy_mode) || return 1
  observed_code=$(resolve_live_power_mode_code) || return 1
  [[ "$observed_mode" == "$power_mode" && "$observed_code" == "$power_mode_code" ]] \
    || return 1
  sampled_at=$(date +%s)
  printf '%s\tac\t%s\t%s\t%s\n' "$sampled_at" "$observed_mode" \
    "$observed_code" "$phase" >>"$output"
}

server_pid=''
caffeinate_started=false
receipt_tmp="$OUT_DIR/.receipt.json.tmp.$$"
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill -INT "$server_pid" 2>/dev/null || true
    for ((waited = 0; waited < 30; waited++)); do
      kill -0 "$server_pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$server_pid" 2>/dev/null && kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=''
  fi
  rm -f "$receipt_tmp"
}
on_exit() {
  local original_rc=$? cleanup_rc=0
  trap - EXIT
  cleanup || cleanup_rc=1
  if [[ "$caffeinate_started" == true ]]; then
    qwen36_stop_power_guard || cleanup_rc=1
  fi
  thermal_cleanup_probe || cleanup_rc=1
  if ((original_rc == 0 && cleanup_rc != 0)); then exit "$cleanup_rc"; fi
  exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

assert_identity() {
  [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$source_commit" \
    && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" \
    && "$(shasum -a 256 "$HF2Q_BIN" | awk '{print $1}')" == "$binary_sha256" \
    && "$(shasum -a 256 "$MODEL_PATH" | awk '{print $1}')" == "$MODEL_SHA256" \
    && "$(stat -f '%d:%i:%z:%m:%c' "$MODEL_PATH" 2>/dev/null \
      || stat -c '%d:%i:%s:%Y:%Z' "$MODEL_PATH")" == "$model_snapshot" ]]
}

median_file() {
  local file=$1 count
  count=$(wc -l <"$file" | tr -d ' ')
  sort -n "$file" | awk -v count="$count" '
    NR == int((count + 1) / 2) {left=$1}
    NR == int((count + 2) / 2) {right=$1}
    END {if (count % 2) print left; else print (left + right) / 2}
  '
}

build_decoder_request() {
  local model_id=$1 trial=$2 output=$3
  jq -n --arg model "$model_id" --argjson trial "$trial" \
    --argjson max_tokens "$DECODER_MAX_TOKENS" '{
      model:$model,
      messages:[
        {role:"system",content:"You are a deterministic streaming scheduler probe. Do not call tools."},
        {role:"user",content:("Trial " + ($trial|tostring)
          + ". Begin with STREAM_BEGIN. Write a long numbered list of concise Rust scheduler invariants, one per line, and continue until the token limit. Do not stop early.")}
      ],
      temperature:0,seed:42,max_tokens:$max_tokens,repetition_penalty:1,
      stream:true,stream_options:{include_usage:true},
      hf2q_enable_thinking:false,chat_template_kwargs:{enable_thinking:false}
    }' >"$output"
}

build_prefill_request() {
  local model_id=$1 trial=$2 lane=$3 output=$4 context
  context=$(awk -v trial="$trial" -v lane="$lane" 'BEGIN {
    printf "mixed trial-%s lane-%s. ", trial, lane
    for (i = 1; i <= 64; i++) printf "cache "
    printf "Return exactly OK."
  }')
  jq -n --arg model "$model_id" --arg content "$context" \
    --argjson max_tokens "$PREFILL_MAX_TOKENS" '{
      model:$model,messages:[{role:"user",content:$content}],
      temperature:0,seed:42,max_tokens:$max_tokens,repetition_penalty:1,
      stream:false,hf2q_enable_thinking:false,
      chat_template_kwargs:{enable_thinking:false}
    }' >"$output"
}

post_unary_at_barrier() {
  local request=$1 response=$2 headers=$3 wall=$4 timing=$5 barrier=$6
  local started finished
  while [[ ! -e "$barrier" ]]; do sleep 0.001; done
  started=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
  curl --fail-with-body --silent --show-error --connect-timeout 5 \
    --max-time "$REQUEST_TIMEOUT_SECONDS" --dump-header "$headers" \
    -H 'Content-Type: application/json' --data-binary "@$request" \
    -o "$response" -w '%{time_total}\n' \
    "http://$HOST:$PORT/v1/chat/completions" >"$wall"
  finished=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
  printf '%s\t%s\n' "$started" "$finished" >"$timing"
  jq -e '
    (.choices | length) == 1
    and (.choices[0].message.content | type) == "string"
    and (.usage.prompt_tokens | numbers) > 0
    and (.usage.prompt_tokens_details.cached_tokens // 0) == 0
    and (.usage.completion_tokens | numbers) > 0
    and (.x_hf2q_timing.time_to_first_token_ms | numbers) > 0
  ' "$response" >/dev/null
}

stream_decoder() {
  local request=$1 sse=$2 frames=$3 headers=$4 started_file=$5 timing=$6
  local started finished
  started=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
  printf '%s\n' "$started" >"$started_file"
  curl --fail-with-body --silent --show-error --no-buffer --connect-timeout 5 \
    --max-time "$REQUEST_TIMEOUT_SECONDS" --dump-header "$headers" \
    -H 'Content-Type: application/json' --data-binary "@$request" \
    "http://$HOST:$PORT/v1/chat/completions" \
    | FRAME_LOG="$frames" perl -MTime::HiRes=time -MIO::Handle -ne '
        BEGIN {open(FRAMES, ">", $ENV{FRAME_LOG}) or die $!; FRAMES->autoflush(1)}
        print STDOUT $_;
        if (/^data: (.*)\r?\n$/) {printf FRAMES "%.9f\t%s\n", time, $1}
      ' >"$sse"
  finished=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
  printf '%s\t%s\n' "$started" "$finished" >"$timing"
}

wait_for_decoder_semantic() {
  local decoder_pid=$1 frames=$2 deadline=$((SECONDS + 30)) count
  while ((SECONDS < deadline)); do
    count=$(qwen35_mixed_semantic_frame_count "$frames")
    if ((count > 0)); then
      kill -0 "$decoder_pid" 2>/dev/null || return 1
      return 0
    fi
    kill -0 "$decoder_pid" 2>/dev/null || return 1
    sleep 0.05
  done
  return 1
}

derive_execution_receipts() {
  local engine_dir=$1 output=$2 relative phase stream receipt
  : >"$output.rows"
  while IFS= read -r relative; do
    phase=${relative#responses/}
    phase=${phase%.headers}
    stream=false
    [[ "$phase" == decoder-* ]] && stream=true
    receipt="$output.$$.json"
    agentic_lifecycle_execution_receipt_json "$engine_dir/$relative" \
      "$MODEL_SHA256" qwen35 "$expected_arch" \
      | jq --arg phase "$phase" --argjson stream "$stream" \
        '. + {phase:$phase,stream:$stream}' >"$receipt"
    jq -c . "$receipt" >>"$output.rows"
    rm -f "$receipt"
  done < <(cd "$engine_dir" && find responses -name '*.headers' -type f -print | sort)
  jq -s . "$output.rows" >"$output"
  rm -f "$output.rows"
}

run_warmups() {
  local model_id=$1 engine_dir=$2 warmup lane pid barrier
  for warmup in 1 2; do
    barrier="$engine_dir/waves/warmup-$warmup.start"
    rm -f "$barrier"
    pids=()
    for ((lane = 1; lane <= PREFILL_LANES; lane++)); do
      build_prefill_request "$model_id" "$((100 + warmup))" "$lane" \
        "$engine_dir/requests/warmup-$warmup-$lane.json"
      post_unary_at_barrier \
        "$engine_dir/requests/warmup-$warmup-$lane.json" \
        "$engine_dir/responses/warmup-$warmup-$lane.json" \
        "$engine_dir/responses/warmup-$warmup-$lane.headers" \
        "$engine_dir/responses/warmup-$warmup-$lane.wall" \
        "$engine_dir/responses/warmup-$warmup-$lane.timing" "$barrier" &
      pids+=("$!")
    done
    : >"$barrier"
    for pid in "${pids[@]}"; do wait "$pid"; done
  done
}

run_measurements() {
  local label=$1 arm=$2 model_id=$3 engine_dir=$4
  local trial lane pid decoder_pid barrier before after before_value after_value delta
  local log_start log_end publication earliest_start latest_start earliest_finish latest_finish
  local launch_skew_seconds prefill_tail_ms request_started semantic_count
  for ((trial = 1; trial <= TRIALS; trial++)); do
    decoder_request="$engine_dir/requests/decoder-$trial.json"
    decoder_sse="$engine_dir/responses/decoder-$trial.sse"
    decoder_frames="$engine_dir/responses/decoder-$trial.frames.tsv"
    decoder_events="$engine_dir/responses/decoder-$trial.events.jsonl"
    decoder_headers="$engine_dir/responses/decoder-$trial.headers"
    decoder_started="$engine_dir/responses/decoder-$trial.started"
    decoder_timing="$engine_dir/responses/decoder-$trial.timing"
    build_decoder_request "$model_id" "$trial" "$decoder_request"
    log_start=$(stat -f '%z' "$engine_dir/server.stderr" 2>/dev/null \
      || stat -c '%s' "$engine_dir/server.stderr")
    stream_decoder "$decoder_request" "$decoder_sse" "$decoder_frames" \
      "$decoder_headers" "$decoder_started" "$decoder_timing" &
    decoder_pid=$!
    wait_for_decoder_semantic "$decoder_pid" "$decoder_frames" || {
      echo "$label trial $trial decoder did not become semantically active" >&2
      return 1
    }

    before="$engine_dir/waves/$trial.metrics-before"
    after="$engine_dir/waves/$trial.metrics-after"
    curl --fail --silent --show-error "http://$HOST:$PORT/metrics" >"$before"
    before_value=$(qwen35_mixed_metric_value "$before" \
      hf2q_qwen_rectangular_prefill_cohorts_total)
    barrier="$engine_dir/waves/$trial.start"
    rm -f "$barrier"
    pids=()
    for ((lane = 1; lane <= PREFILL_LANES; lane++)); do
      request="$engine_dir/requests/prefill-$trial-$lane.json"
      response="$engine_dir/responses/prefill-$trial-$lane.json"
      headers="$engine_dir/responses/prefill-$trial-$lane.headers"
      wall="$engine_dir/responses/prefill-$trial-$lane.wall"
      timing="$engine_dir/responses/prefill-$trial-$lane.timing"
      build_prefill_request "$model_id" "$trial" "$lane" "$request"
      post_unary_at_barrier "$request" "$response" "$headers" "$wall" \
        "$timing" "$barrier" &
      pids+=("$!")
    done
    : >"$barrier"
    for pid in "${pids[@]}"; do wait "$pid"; done
    wait "$decoder_pid" || return 1
    decoder_pid=''
    qwen36_extract_and_validate_sse "mixed decoder" "$decoder_sse" "$decoder_events"
    semantic_count=$(qwen35_mixed_semantic_frame_count "$decoder_frames")
    ((semantic_count >= MIN_SEMANTIC_EVENTS)) || return 1

    timing_files=("$engine_dir"/responses/prefill-"$trial"-*.timing)
    [[ "${#timing_files[@]}" == "$PREFILL_LANES" ]] || return 1
    earliest_start=$(awk -F '\t' 'NR == 1 || $1 < value {value=$1} END {print value}' \
      "${timing_files[@]}")
    latest_start=$(awk -F '\t' 'NR == 1 || $1 > value {value=$1} END {print value}' \
      "${timing_files[@]}")
    earliest_finish=$(awk -F '\t' 'NR == 1 || $2 < value {value=$2} END {print value}' \
      "${timing_files[@]}")
    latest_finish=$(awk -F '\t' 'NR == 1 || $2 > value {value=$2} END {print value}' \
      "${timing_files[@]}")
    launch_skew_seconds=$(awk -v first="$earliest_start" -v last="$latest_start" \
      'BEGIN {printf "%.9f", last-first}')
    prefill_tail_ms=$(awk -v first="$earliest_start" -v last="$latest_finish" \
      'BEGIN {printf "%.6f", (last-first)*1000}')
    awk -v skew="$launch_skew_seconds" -v latest="$latest_start" \
      -v earliest="$earliest_finish" -v tail="$prefill_tail_ms" \
      -v max_skew="$MAX_LAUNCH_SKEW_MS" -v max_tail="$MAX_PREFILL_TAIL_MS" \
      'BEGIN {exit !(skew*1000 <= max_skew && latest < earliest && tail <= max_tail)}' \
      || return 1
    request_started=$(tr -d '[:space:]' <"$decoder_started")
    qwen35_mixed_semantic_trace_json "$decoder_frames" "$request_started" \
      "$earliest_start" "$latest_finish" >"$engine_dir/waves/$trial.semantic.json"
    qwen35_mixed_validate_semantic_trace "$engine_dir/waves/$trial.semantic.json" \
      "$MIN_SEMANTIC_EVENTS" "$MAX_DECODER_TTFT_MS" "$MAX_SEMANTIC_GAP_MS"
    jq -er '.first_semantic_ms' "$engine_dir/waves/$trial.semantic.json" \
      >>"$engine_dir/semantic-ttft-ms"
    jq -er '.max_semantic_gap_ms' "$engine_dir/waves/$trial.semantic.json" \
      >>"$engine_dir/semantic-gap-ms"
    printf '%s\n' "$prefill_tail_ms" >>"$engine_dir/prefill-tail-ms"

    curl --fail --silent --show-error "http://$HOST:$PORT/metrics" >"$after"
    after_value=$(qwen35_mixed_metric_value "$after" \
      hf2q_qwen_rectangular_prefill_cohorts_total)
    delta=$(awk -v before="$before_value" -v after="$after_value" \
      'BEGIN {printf "%.0f", after-before}')
    if [[ "$arm" == on ]]; then [[ "$delta" == 1 ]] || return 1
    else [[ "$delta" == 0 ]] || return 1
    fi
    log_end=$(stat -f '%z' "$engine_dir/server.stderr" 2>/dev/null \
      || stat -c '%s' "$engine_dir/server.stderr")
    tail -c "+$((log_start + 1))" "$engine_dir/server.stderr" \
      | head -c "$((log_end - log_start))" >"$engine_dir/waves/$trial.log"
    publication=$(rg 'Qwen rectangular stable-boundary prefill published' \
      "$engine_dir/waves/$trial.log" || true)
    if [[ "$arm" == on ]]; then
      [[ "$(printf '%s\n' "$publication" | awk 'NF {n++} END {print n+0}')" == 1 ]] \
        || return 1
      qwen35_mixed_validate_publication "$publication" "$expected_mtp"
    else
      [[ -z "$publication" ]] || return 1
    fi

    jq -n --argjson launch_skew_seconds "$launch_skew_seconds" \
      --argjson earliest_start "$earliest_start" --argjson latest_start "$latest_start" \
      --argjson earliest_finish "$earliest_finish" --argjson latest_finish "$latest_finish" \
      --argjson prefill_tail_ms "$prefill_tail_ms" --argjson cohort_delta "$delta" \
      --slurpfile semantic "$engine_dir/waves/$trial.semantic.json" '{
        launch_skew_seconds:$launch_skew_seconds,
        earliest_start:$earliest_start,latest_start:$latest_start,
        earliest_finish:$earliest_finish,latest_finish:$latest_finish,
        actual_prefill_overlap:($latest_start < $earliest_finish),
        prefill_tail_ms:$prefill_tail_ms,cohort_metric_delta:$cohort_delta,
        semantic:$semantic[0]
      }' >"$engine_dir/waves/$trial.json"
    ps -p "$server_pid" -o rss= | awk 'NF == 1 && $1 ~ /^[0-9]+$/ {print $1}' \
      >>"$engine_dir/rss-kib"
  done
}

sample_rss_while_pid() {
  local owner_pid=$1 producer_pid=$2 output=$3 state
  while :; do
    state=$(ps -p "$producer_pid" -o state= 2>/dev/null | tr -d '[:space:]' || true)
    [[ -n "$state" && "$state" != Z* ]] || break
    ps -p "$owner_pid" -o rss= | awk 'NF == 1 && $1 ~ /^[0-9]+$/ {print $1}' \
      >>"$output"
    sleep 0.1
  done
  ps -p "$owner_pid" -o rss= | awk 'NF == 1 && $1 ~ /^[0-9]+$/ {print $1}' \
    >>"$output"
}

run_process() {
  local label=$1 arm=$2 cross_slot=0 coalesce=0 expected_bool=false
  local engine_dir="$OUT_DIR/$label" ready_wait model_id actual_arch producer_pid
  local rss_monitor_pid producer_status=0 monitor_status=0 evidence_manifest_sha
  [[ "$arm" == on ]] && { cross_slot=1; coalesce=$COALESCE_US; expected_bool=true; }
  mkdir -p "$engine_dir"/{requests,responses,waves,runtime-cache}
  : >"$engine_dir/rss-kib"
  : >"$engine_dir/semantic-ttft-ms"
  : >"$engine_dir/semantic-gap-ms"
  : >"$engine_dir/prefill-tail-ms"
  : >"$engine_dir/power.tsv"
  record_power_contract "$engine_dir/power.tsv" "$label-before-launch"
  assert_identity
  env -i HOME="$RUNTIME_HOME" PATH="$RUNTIME_PATH" TMPDIR="$RUNTIME_TMPDIR" \
    LANG=C LC_ALL=C USER=hf2q-gate LOGNAME=hf2q-gate RUST_BACKTRACE=1 \
    HF2Q_CROSS_SLOT_ADMIT="$cross_slot" HF2Q_ADMIT_COALESCE_US="$coalesce" \
    HF2Q_QWEN_SPECULATION=auto HF2Q_TQ_KV=1 HF2Q_ENCODER_SESSION=1 \
    HF2Q_FFN_TERMINAL_K_BATCH=8 \
    "$HF2Q_BIN" -v serve --model "$MODEL_PATH" \
    --cache-dir "$engine_dir/runtime-cache" --host "$HOST" --port "$PORT" \
    --scheduler inflight-batched --max-slots "$MAX_SLOTS" --overflow-policy reject \
    --kv-cache-budget "$KV_CACHE_BUDGET_BYTES" --default-repetition-penalty 1 \
    --default-thinking-token-budget 0 --default-tool-thinking-token-budget 0 \
    >"$engine_dir/server.stdout" 2>"$engine_dir/server.stderr" &
  server_pid=$!
  for ((ready_wait = 0; ready_wait < READY_TIMEOUT_SECONDS; ready_wait++)); do
    curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null 2>&1 && break
    kill -0 "$server_pid" 2>/dev/null || { tail -n 100 "$engine_dir/server.stderr" >&2; return 1; }
    sleep 1
  done
  curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null || return 1
  curl --fail --silent "http://$HOST:$PORT/v1/models" >"$engine_dir/models.json"
  model_id=$(jq -er '[.data[] | select(.loaded == true)]
    | if length == 1 then .[0].id else error("one loaded model required") end' \
    "$engine_dir/models.json")
  actual_arch=$(jq -er --arg id "$model_id" '[.data[]
    | select(.loaded == true and .id == $id)]
    | if length == 1 then .[0].arch else error("one architecture required") end' \
    "$engine_dir/models.json")
  [[ "$actual_arch" == "$expected_arch" ]] || return 1
  EXPECTED_ADMIT="$expected_bool" EXPECTED_COALESCE="$coalesce" \
    EXPECTED_MTP="$expected_mtp_bool" perl -ne '
      if (/Qwen35 SlotAware prefill transaction ceiling selected/) {
        $seen++; $admit=$1 if /cross_slot_admit=(true|false)/;
        $coalesce=$1 if /cross_slot_coalesce_us=([0-9]+)/;
        $policy=$1 if /speculation_policy=(Auto|Off)/;
        $mtp=$1 if /mtp_capable=(true|false)/;
      }
      END {exit 1 unless $seen == 1 && $admit eq $ENV{EXPECTED_ADMIT}
        && $coalesce == $ENV{EXPECTED_COALESCE} && $policy eq "Auto"
        && $mtp eq $ENV{EXPECTED_MTP}}
    ' "$engine_dir/server.stderr" || return 1
  perl -ne '
    if (/resolved serving plan/) {
      $seen++; $persist=$1 if /kv_persist_enabled=(true|false)/;
      $budget=$1 if /kv_persist_budget_bytes=([0-9]+)/;
      $cache=$1 if /kv_cache_budget_bytes=([0-9]+)/;
    }
    END {exit 1 unless $seen == 1 && $persist eq "false" && $budget == 0
      && $cache == 51539607552}
  ' "$engine_dir/server.stderr" || return 1
  qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
    "$HF2Q_BIN" "$MODEL_PATH" "$MAX_SLOTS"
  printf '%s\n' "$server_pid" >"$engine_dir/server-pid.txt"
  ps -ww -p "$server_pid" -o command= >"$engine_dir/server-command.txt"
  ps -p "$server_pid" -o rss= | awk 'NF == 1 && $1 ~ /^[0-9]+$/ {print $1}' \
    >>"$engine_dir/rss-kib"
  run_warmups "$model_id" "$engine_dir"
  record_power_contract "$engine_dir/power.tsv" "$label-loaded-warm"
  thermal_wait_for_nominal "$engine_dir/thermal-settle.tsv" "$label-settle" \
    "$THERMAL_SETTLE_SECONDS" "$THERMAL_SETTLE_TIMEOUT_SECONDS" \
    "$THERMAL_SAMPLE_SECONDS" "$engine_dir/contention-settle.tsv" "$server_pid"
  record_power_contract "$engine_dir/power.tsv" "$label-measurement-start"
  thermal_sample "$engine_dir/thermal-measurement.tsv" "$label-measurement-start"
  host_contention_sample "$engine_dir/contention-measurement.tsv" \
    "$label-measurement-start" "$server_pid" "$THERMAL_SAMPLED_AT"
  host_contention_require_quiet "$label-measurement-start"
  (run_measurements "$label" "$arm" "$model_id" "$engine_dir") &
  producer_pid=$!
  sample_rss_while_pid "$server_pid" "$producer_pid" "$engine_dir/rss-kib" &
  rss_monitor_pid=$!
  thermal_monitor_fair_or_better_while_pid \
    "$engine_dir/thermal-measurement.tsv" "$label-measurement" "$producer_pid" \
    "$THERMAL_SAMPLE_SECONDS" "$engine_dir/contention-measurement.tsv" "$server_pid" \
    || monitor_status=$?
  wait "$producer_pid" || producer_status=$?
  wait "$rss_monitor_pid"
  ((producer_status == 0 && monitor_status == 0)) || return 1
  thermal_sample "$engine_dir/thermal-measurement.tsv" "$label-measurement-end"
  host_contention_sample "$engine_dir/contention-measurement.tsv" \
    "$label-measurement-end" "$server_pid" "$THERMAL_SAMPLED_AT"
  host_contention_require_quiet "$label-measurement-end"
  record_power_contract "$engine_dir/power.tsv" "$label-measurement-end"
  thermal_validate_settle_log "$engine_dir/thermal-settle.tsv" \
    "$THERMAL_SETTLE_SECONDS" "$((THERMAL_SAMPLE_SECONDS + 3))"
  host_contention_validate_settle_log "$engine_dir/contention-settle.tsv" \
    "$THERMAL_SETTLE_SECONDS" "$((THERMAL_SAMPLE_SECONDS + 3))"
  thermal_validate_fair_or_better_measurement_log \
    "$engine_dir/thermal-measurement.tsv" "$((THERMAL_SAMPLE_SECONDS + 3))"
  host_contention_validate_measurement_log "$engine_dir/contention-measurement.tsv" \
    "$((THERMAL_SAMPLE_SECONDS + 3))"
  host_contention_validate_thermal_alignment "$engine_dir/thermal-measurement.tsv" \
    "$engine_dir/contention-measurement.tsv"
  qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
    "$HF2Q_BIN" "$MODEL_PATH" "$MAX_SLOTS"
  qwen36_reject_fatal_log "$engine_dir/server.stderr"
  assert_identity
  derive_execution_receipts "$engine_dir" "$engine_dir/execution.json"
  : >"$engine_dir/canonical.jsonl"
  for ((trial = 1; trial <= TRIALS; trial++)); do
    qwen35_mixed_canonical_sse_json "$engine_dir/responses/decoder-$trial.events.jsonl" \
      | jq -c --argjson trial "$trial" '. + {kind:"decoder",trial:$trial}' \
      >>"$engine_dir/canonical.jsonl"
    for ((lane = 1; lane <= PREFILL_LANES; lane++)); do
      qwen35_mixed_canonical_unary_json \
        "$engine_dir/responses/prefill-$trial-$lane.json" \
        | jq -c --argjson trial "$trial" --argjson lane "$lane" \
          '. + {kind:"prefill",trial:$trial,lane:$lane}' \
        >>"$engine_dir/canonical.jsonl"
    done
  done
  cleanup
  [[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]] || return 1
  record_power_contract "$engine_dir/power.tsv" "$label-after-shutdown"
  assert_identity
  (
    cd "$engine_dir"
    {
      find requests responses waves runtime-cache -type f -print
      printf '%s\n' server.stdout server.stderr thermal-settle.tsv \
        thermal-measurement.tsv contention-settle.tsv contention-measurement.tsv \
        rss-kib semantic-ttft-ms semantic-gap-ms prefill-tail-ms power.tsv \
        models.json server-pid.txt server-command.txt execution.json canonical.jsonl
    } | sort | while IFS= read -r relative; do
      printf '%s  %s\n' "$(shasum -a 256 "$relative" | awk '{print $1}')" "$relative"
    done >evidence.sha256.tmp
    mv evidence.sha256.tmp evidence.sha256
    shasum -a 256 -c evidence.sha256 >/dev/null
  )
  evidence_manifest_sha=$(shasum -a 256 "$engine_dir/evidence.sha256" | awk '{print $1}')
  jq -n --arg label "$label" --arg arm "$arm" --arg model_id "$model_id" \
    --argjson server_pid "$(tr -d '[:space:]' <"$engine_dir/server-pid.txt")" \
    --argjson prefill_median_ms "$(median_file "$engine_dir/prefill-tail-ms")" \
    --argjson max_prefill_tail_ms "$(sort -n "$engine_dir/prefill-tail-ms" | tail -1)" \
    --argjson max_semantic_ttft_ms "$(sort -n "$engine_dir/semantic-ttft-ms" | tail -1)" \
    --argjson max_semantic_gap_ms "$(sort -n "$engine_dir/semantic-gap-ms" | tail -1)" \
    --argjson sampled_peak_rss_kib "$(sort -n "$engine_dir/rss-kib" | tail -1)" \
    --arg execution_sha "$(shasum -a 256 "$engine_dir/execution.json" | awk '{print $1}')" \
    --arg canonical_sha "$(shasum -a 256 "$engine_dir/canonical.jsonl" | awk '{print $1}')" \
    --arg thermal_settle_sha "$(shasum -a 256 "$engine_dir/thermal-settle.tsv" | awk '{print $1}')" \
    --arg thermal_measurement_sha "$(shasum -a 256 "$engine_dir/thermal-measurement.tsv" | awk '{print $1}')" \
    --arg contention_settle_sha "$(shasum -a 256 "$engine_dir/contention-settle.tsv" | awk '{print $1}')" \
    --arg contention_measurement_sha "$(shasum -a 256 "$engine_dir/contention-measurement.tsv" | awk '{print $1}')" \
    --arg power_sha "$(shasum -a 256 "$engine_dir/power.tsv" | awk '{print $1}')" \
    --arg manifest_sha "$evidence_manifest_sha" '{
      label:$label,arm:$arm,model_id:$model_id,server_pid:$server_pid,
      prefill_median_ms:$prefill_median_ms,
      max_prefill_tail_ms:$max_prefill_tail_ms,
      max_semantic_ttft_ms:$max_semantic_ttft_ms,
      max_semantic_gap_ms:$max_semantic_gap_ms,
      sampled_peak_rss_kib:$sampled_peak_rss_kib,
      runtime:{clean_environment:true,max_slots:8,scheduler:"inflight-batched",
        speculation:"auto",kv_cache_budget_bytes:51539607552,kv_persist:false,
        cache_dir:"evidence-local"},
      evidence:{execution_sha256:$execution_sha,canonical_sha256:$canonical_sha,
        thermal_settle_sha256:$thermal_settle_sha,
        thermal_measurement_sha256:$thermal_measurement_sha,
        contention_settle_sha256:$contention_settle_sha,
        contention_measurement_sha256:$contention_measurement_sha,
        power_sha256:$power_sha,manifest_sha256:$manifest_sha}
    }' >"$engine_dir/summary.json"
}

thermal_prepare_probe
qwen36_start_power_guard "$$" "$OUT_DIR/caffeinate.log"
caffeinate_started=true
run_process off-a off
run_process on-a on
run_process on-b on
run_process off-b off

for replica in a b; do
  while IFS= read -r relative; do
    cmp -s "$OUT_DIR/off-$replica/$relative" "$OUT_DIR/on-$replica/$relative" || {
      echo "OFF/ON mixed request bytes differ for $replica/$relative" >&2
      exit 1
    }
  done < <(cd "$OUT_DIR/off-$replica" && find requests -type f -print | sort)
  cmp -s "$OUT_DIR/off-$replica/canonical.jsonl" \
    "$OUT_DIR/on-$replica/canonical.jsonl" || {
    echo "OFF/ON canonical mixed responses differ for replica $replica" >&2
    exit 1
  }
done

off_prefill_samples="$OUT_DIR/off-prefill-tail-ms"
on_prefill_samples="$OUT_DIR/on-prefill-tail-ms"
cat "$OUT_DIR/off-a/prefill-tail-ms" "$OUT_DIR/off-b/prefill-tail-ms" >"$off_prefill_samples"
cat "$OUT_DIR/on-a/prefill-tail-ms" "$OUT_DIR/on-b/prefill-tail-ms" >"$on_prefill_samples"
off_median=$(median_file "$off_prefill_samples")
on_median=$(median_file "$on_prefill_samples")
mixed_speedup=$(awk -v off="$off_median" -v on="$on_median" 'BEGIN {print off/on}')
neighbor_a=$(awk -v off="$(median_file "$OUT_DIR/off-a/prefill-tail-ms")" \
  -v on="$(median_file "$OUT_DIR/on-a/prefill-tail-ms")" 'BEGIN {print off/on}')
neighbor_b=$(awk -v off="$(median_file "$OUT_DIR/off-b/prefill-tail-ms")" \
  -v on="$(median_file "$OUT_DIR/on-b/prefill-tail-ms")" 'BEGIN {print off/on}')
awk -v value="$mixed_speedup" -v minimum="$MIN_MIXED_SPEEDUP" \
  'BEGIN {exit !(value >= minimum)}' || {
  echo "mixed prefill speedup $mixed_speedup is below $MIN_MIXED_SPEEDUP" >&2
  exit 1
}
for neighbor in "$neighbor_a" "$neighbor_b"; do
  awk -v value="$neighbor" 'BEGIN {exit !(value > 1)}' || {
    echo "an ON mixed process did not beat its neighboring OFF process" >&2
    exit 1
  }
done
canonical_sha=$(shasum -a 256 "$OUT_DIR/on-a/canonical.jsonl" | awk '{print $1}')
assert_identity
qwen36_assert_power_guard
qwen36_stop_power_guard
caffeinate_started=false

jq -n --arg source_commit "$source_commit" --arg binary "$HF2Q_BIN" \
  --arg binary_sha "$binary_sha256" --arg model_shape "$MODEL_SHAPE" \
  --arg model_path "$MODEL_PATH" --arg model_sha "$MODEL_SHA256" \
  --arg model_snapshot "$model_snapshot" --argjson model_bytes "$MODEL_BYTES" \
  --arg canonical_sha "$canonical_sha" --arg power_mode "$power_mode" \
  --argjson mixed_speedup "$mixed_speedup" --argjson neighbor_a "$neighbor_a" \
  --argjson neighbor_b "$neighbor_b" --argjson off_median "$off_median" \
  --argjson on_median "$on_median" \
  --arg off_a_summary "$(shasum -a 256 "$OUT_DIR/off-a/summary.json" | awk '{print $1}')" \
  --arg on_a_summary "$(shasum -a 256 "$OUT_DIR/on-a/summary.json" | awk '{print $1}')" \
  --arg on_b_summary "$(shasum -a 256 "$OUT_DIR/on-b/summary.json" | awk '{print $1}')" \
  --arg off_b_summary "$(shasum -a 256 "$OUT_DIR/off-b/summary.json" | awk '{print $1}')" \
  --arg off_a_manifest "$(shasum -a 256 "$OUT_DIR/off-a/evidence.sha256" | awk '{print $1}')" \
  --arg on_a_manifest "$(shasum -a 256 "$OUT_DIR/on-a/evidence.sha256" | awk '{print $1}')" \
  --arg on_b_manifest "$(shasum -a 256 "$OUT_DIR/on-b/evidence.sha256" | awk '{print $1}')" \
  --arg off_b_manifest "$(shasum -a 256 "$OUT_DIR/off-b/evidence.sha256" | awk '{print $1}')" \
  --arg caffeinate_log "$(shasum -a 256 "$OUT_DIR/caffeinate.log" | awk '{print $1}')" \
  --arg caffeinate_assertions "$(shasum -a 256 "$OUT_DIR/caffeinate.log.assertions" | awk '{print $1}')" \
  --arg events_baseline "$(shasum -a 256 "$OUT_DIR/caffeinate.log.power-events.baseline" | awk '{print $1}')" \
  --arg events_final "$(shasum -a 256 "$OUT_DIR/caffeinate.log.power-events.final" | awk '{print $1}')" \
  --arg events_new "$(shasum -a 256 "$OUT_DIR/caffeinate.log.power-events.new" | awk '{print $1}')" '{
    schema:1,verdict:"pass",gate:"qwen35-mixed-rectangular-cell",
    source:{commit:$source_commit,binary:$binary,sha256:$binary_sha},
    model:{shape:$model_shape,path:$model_path,sha256:$model_sha,
      bytes:$model_bytes,snapshot:$model_snapshot},
    workload:{process_order:["off-a","on-a","on-b","off-b"],same_binary:true,
      trials_per_process:5,max_slots:8,live_decoders:1,cold_prefills:4,
      prefill_max_tokens:2,decoder_max_tokens:512,temperature:0,seed:42,
      speculation:"auto",coalesce_us:25000,kv_cache_budget_bytes:51539607552},
    environment:{power:"ac",power_mode:$power_mode,
      thermal:"nominal-settle-and-fair-or-better-measurement",
      host_contention:"quiet",clean_process_environment:true,serve_kv_persist:false},
    thresholds:{min_mixed_speedup:1.01,min_semantic_events:3,
      max_decoder_ttft_ms:15000,max_semantic_gap_ms:15000,
      max_prefill_tail_ms:60000,max_launch_skew_ms:100},
    equality:{canonical_sha256:$canonical_sha},
    evidence:{processes:{
      "off-a":{summary_sha256:$off_a_summary,manifest_sha256:$off_a_manifest},
      "on-a":{summary_sha256:$on_a_summary,manifest_sha256:$on_a_manifest},
      "on-b":{summary_sha256:$on_b_summary,manifest_sha256:$on_b_manifest},
      "off-b":{summary_sha256:$off_b_summary,manifest_sha256:$off_b_manifest}},
      power_guard:{caffeinate_log_sha256:$caffeinate_log,
        assertions_sha256:$caffeinate_assertions,
        events_baseline_sha256:$events_baseline,events_final_sha256:$events_final,
        events_new_sha256:$events_new}},
    result:{mixed_prefill_speedup:$mixed_speedup,
      neighboring_process_speedups:[$neighbor_a,$neighbor_b],
      off_median_prefill_tail_ms:$off_median,on_median_prefill_tail_ms:$on_median}
  }' >"$receipt_tmp"
mv "$receipt_tmp" "$OUT_DIR/receipt.json"
"$script_dir/verify_qwen35_mixed_rectangular_receipt.sh" \
  "$OUT_DIR/receipt.json" "$SOURCE_ROOT"
jq . "$OUT_DIR/receipt.json"
echo "mixed Qwen receipt: $OUT_DIR/receipt.json" >&2
