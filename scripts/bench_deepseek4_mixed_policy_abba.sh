#!/usr/bin/env bash
set -euo pipefail

# Exact same-binary ADR-049 B.1 control/candidate comparison. Each measured
# process first primes four distinct stable decoder prefixes. Every measured
# wave restores those caches, establishes four semantic SSE decoders, then
# launches four long prefills into the remaining slots. OFF disables only the
# bounded Mixed cohort attempt; pure-prefill cooperation and all model
# arithmetic stay identical.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
if [[ ${HF2Q_DEEPSEEK_B1_GATE_ISOLATED:-0} != 1 ]]; then
    exec "$script_dir/run_release_gate_process_group.sh" env \
      HF2Q_DEEPSEEK_B1_GATE_ISOLATED=1 "$0" "$@"
fi
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"
readonly HOST_CONTENTION_GATE_OWNER_PID=$$
host_contention_require_isolated_gate_owner \
  "$HOST_CONTENTION_GATE_OWNER_PID"
# shellcheck source=scripts/macos_memory_guard.sh
source "$script_dir/macos_memory_guard.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/qwen38_matched_reference_contract.sh
source "$script_dir/qwen38_matched_reference_contract.sh"

SOURCE_ROOT=${SOURCE_ROOT:-$root_dir}
HF2Q_BIN=${HF2Q_BIN:-$SOURCE_ROOT/target/release/hf2q}
MODEL_PATH=${MODEL_PATH:-/opt/hf2q/models/deepseek4/DeepSeek-V4-Flash-0731-agentic-q2.gguf}
MODEL_SHA256=${MODEL_SHA256:-936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d}
MODEL_BYTES=${MODEL_BYTES:-107431343168}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-52853}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-300}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-180}
THERMAL_SETTLE_TIMEOUT_SECONDS=${THERMAL_SETTLE_TIMEOUT_SECONDS:-900}
readonly TRIALS=5
readonly WARMUPS=1
readonly MAX_SLOTS=8
readonly LIVE_DECODERS=4
readonly PREFILLERS=4
readonly MIXED_ROWS=128
readonly DECODER_PRIME_MAX_TOKENS=1
readonly DECODER_MAX_TOKENS=256
readonly PREFILL_MAX_TOKENS=8
readonly PREFILL_REPETITIONS=3520
readonly KV_CACHE_BUDGET_BYTES=8589934592
readonly THERMAL_SETTLE_SECONDS=60
readonly SAMPLE_SECONDS=2
readonly MAX_SAMPLE_GAP_SECONDS=5
readonly MAX_SCHEDULER_GAP_MS=15000
readonly MAX_SEMANTIC_SSE_GAP_MS=15000
readonly MAX_PREFILL_WALL_SECONDS=60
readonly MAX_PEAK_RSS_BYTES=124554051584
readonly MIN_WAVE_SPEEDUP=1.0
readonly RUNTIME_HOME=/var/empty
readonly RUNTIME_PATH=/usr/bin:/bin:/usr/sbin:/sbin
readonly RUNTIME_TMPDIR=/var/tmp
readonly POWER_PROBE_ATTEMPTS=3
readonly EXPECTED_MODEL_SHA256=936a97e68fe1a04185df149fcb833c3e1462ca5923fbf4ef3e7296bd78c7ad0d
readonly EXPECTED_MODEL_BYTES=107431343168

for command in awk caffeinate curl date env find git head jq lsof mkdir mv \
    perl pgrep pmset ps python3 rg rm sed shasum sleep sort stat \
    system_profiler tail tr wc seq; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
for path in "$SOURCE_ROOT" "$HF2Q_BIN" "$MODEL_PATH" "$OUT_DIR"; do
    [[ "$path" == /* ]] || { echo "gate paths must be absolute: $path" >&2; exit 2; }
done
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd -P)
[[ -d "$SOURCE_ROOT/.git" || -f "$SOURCE_ROOT/.git" ]] || exit 2
[[ "$HOST" == 127.0.0.1 ]] || {
    echo "DeepSeek B.1 ABBA requires the 127.0.0.1 loopback endpoint" >&2
    exit 2
}
[[ -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" ]] || {
    echo "DeepSeek B.1 ABBA requires a clean exact source tree" >&2
    exit 2
}
source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
[[ "$HF2Q_BIN" == "$SOURCE_ROOT/target/release/hf2q" && -x "$HF2Q_BIN" ]] || {
    echo "HF2Q_BIN must be this source tree's release output" >&2
    exit 2
}
grep -aFq "$source_commit" "$HF2Q_BIN" || {
    echo "release binary does not embed source commit $source_commit" >&2
    exit 2
}
binary_sha256=$(shasum -a 256 "$HF2Q_BIN" | awk '{print $1}')
[[ -f "$MODEL_PATH" && -r "$MODEL_PATH" && ! -L "$MODEL_PATH" \
    && "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ && "$MODEL_BYTES" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$MODEL_SHA256" == "$EXPECTED_MODEL_SHA256" \
    && "$MODEL_BYTES" == "$EXPECTED_MODEL_BYTES" ]] || {
    echo "DeepSeek B.1 ABBA is pinned to the exact qualified artifact" >&2
    exit 2
}
MODEL_PATH="$(cd "${MODEL_PATH%/*}" && pwd -P)/${MODEL_PATH##*/}"
[[ "$(stat -f '%z' "$MODEL_PATH" 2>/dev/null || stat -c '%s' "$MODEL_PATH")" == "$MODEL_BYTES" ]] || {
    echo "DeepSeek model byte size drift" >&2
    exit 2
}
model_snapshot=$(hf2q_release_model_snapshot "$MODEL_PATH")
for setting in PORT READY_TIMEOUT_SECONDS REQUEST_TIMEOUT_SECONDS \
    THERMAL_SETTLE_TIMEOUT_SECONDS; do
    value=${!setting}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || exit 2
done
((PORT <= 65535)) || exit 2
[[ ! -e "$OUT_DIR" || -z "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]] || {
    echo "OUT_DIR must be fresh: $OUT_DIR" >&2
    exit 2
}
mkdir -p "$OUT_DIR"
OUT_DIR=$(cd "$OUT_DIR" && pwd -P)
case "$OUT_DIR/" in "$SOURCE_ROOT"/*) echo "OUT_DIR must be external" >&2; exit 2;; esac
[[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]] || exit 2
if pgrep -x hf2q >/dev/null 2>&1; then
    echo "DeepSeek B.1 ABBA requires no pre-existing hf2q runtime" >&2
    pgrep -flx hf2q >&2 || true
    exit 2
fi

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
median_file() {
    local file=$1 count
    count=$(wc -l <"$file" | tr -d ' ')
    sort -n "$file" | awk -v count="$count" '
      NR == int((count + 1) / 2) { left=$1 }
      NR == int((count + 2) / 2) { right=$1 }
      END {
        if (count % 2) value=left; else value=(left + right) / 2
        printf "%.15f\n", value
      }
    '
}

resolve_ac_energy_mode() {
    local attempt observed
    for ((attempt=1; attempt<=POWER_PROBE_ATTEMPTS; attempt++)); do
        if observed=$(LANG=C LC_ALL=C system_profiler SPPowerDataType | matched_parse_ac_power_mode); then
            printf '%s\n' "$observed"; return 0
        fi
        ((attempt == POWER_PROBE_ATTEMPTS)) || sleep 1
    done
    return 1
}
resolve_live_power_mode_code() {
    local attempt observed
    for ((attempt=1; attempt<=POWER_PROBE_ATTEMPTS; attempt++)); do
        if observed=$(pmset -g live | matched_parse_live_power_mode_code); then
            printf '%s\n' "$observed"; return 0
        fi
        ((attempt == POWER_PROBE_ATTEMPTS)) || sleep 1
    done
    return 1
}

resolve_live_power_source() {
    local attempt observed report
    for ((attempt=1; attempt<=POWER_PROBE_ATTEMPTS; attempt++)); do
        if report=$(pmset -g batt 2>/dev/null) \
            && observed=$(matched_parse_live_power_source <<<"$report"); then
            printf '%s\n' "$observed"; return 0
        fi
        ((attempt == POWER_PROBE_ATTEMPTS)) || sleep 1
    done
    return 1
}
power_source=$(resolve_live_power_source) || exit 2
[[ "$power_source" == ac ]] || {
    echo "DeepSeek B.1 gate requires AC power" >&2; exit 2
}
power_mode=$(resolve_ac_energy_mode) || exit 2
[[ "$power_mode" != low ]] || { echo "Low Power Mode is not accepted" >&2; exit 2; }
power_mode_code=$(resolve_live_power_mode_code) || exit 2

sample_fast_power() {
    local output=$1 phase=$2 sampled_at observed_source observed_code report
    report=$(pmset -g batt 2>/dev/null) || return 1
    observed_source=$(matched_parse_live_power_source <<<"$report") || return 1
    [[ "$observed_source" == ac ]] || return 1
    observed_code=$(pmset -g live | matched_parse_live_power_mode_code) || return 1
    [[ "$observed_code" == "$power_mode_code" ]] || return 1
    sampled_at=$(date +%s)
    printf '%s\tac\t%s\t%s\t%s\n' "$sampled_at" "$power_mode" \
        "$observed_code" "$phase" >>"$output"
}

server_pid=
caffeinate_started=false
cleanup_server() {
    if [[ -n "$server_pid" ]]; then
        kill -INT "$server_pid" 2>/dev/null || true
        for ((attempt=1; attempt<=30; attempt++)); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$server_pid" 2>/dev/null && kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}
on_exit() {
    local original_rc=$? cleanup_rc=0
    trap - EXIT
    cleanup_server || cleanup_rc=1
    if [[ "$caffeinate_started" == true ]]; then qwen36_stop_power_guard || cleanup_rc=1; fi
    thermal_cleanup_probe || cleanup_rc=1
    ((original_rc == 0 && cleanup_rc != 0)) && original_rc=$cleanup_rc
    exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

assert_identity() {
    [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$source_commit" \
        && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" \
        && "$(sha256_file "$HF2Q_BIN")" == "$binary_sha256" \
        && "$(hf2q_release_model_snapshot "$MODEL_PATH")" == "$model_snapshot" ]] || {
        echo "source, binary, or model identity changed during DeepSeek B.1 ABBA" >&2
        return 1
    }
}

build_request() {
    local model=$1 replica=$2 trial=$3 kind=$4 lane=$5 output=$6 body max_tokens
    if [[ "$kind" == decoder || "$kind" == decoder-prime ]]; then
        body=$(awk -v replica="$replica" -v lane="$lane" 'BEGIN {
          printf "replica-%s stable decoder-%s. ", replica, lane
          for (i=1; i<=128; i++) printf "sequence "
          printf "Write a long numbered sequence in words without commentary; continue until the response limit."
        }')
        if [[ "$kind" == decoder ]]; then
            max_tokens=$DECODER_MAX_TOKENS
        else
            max_tokens=$DECODER_PRIME_MAX_TOKENS
        fi
    else
        body=$(awk -v replica="$replica" -v trial="$trial" -v lane="$lane" \
          -v repetitions="$PREFILL_REPETITIONS" 'BEGIN {
          printf "replica-%s trial-%s prefill-%s. ", replica, trial, lane
          for (i=1; i<=repetitions; i++) printf "context "
          printf "Reply with exactly READY."
        }')
        max_tokens=$PREFILL_MAX_TOKENS
    fi
    jq -n --arg model "$model" --arg body "$body" --argjson max_tokens "$max_tokens" '{
      model:$model,messages:[{role:"user",content:$body}],temperature:0,seed:42,
      repetition_penalty:1,max_tokens:$max_tokens,stream:true,
      stream_options:{include_usage:true}
    }' >"$output"
}

post_timed_sse() {
    local request=$1 output=$2 timing=$3 stderr=$4 barrier=$5 started finished
    while [[ ! -e "$barrier" ]]; do sleep 0.001; done
    started=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
      -e 'printf "%.9f\n", clock_gettime(CLOCK_MONOTONIC)')
    curl --fail-with-body --silent --show-error --no-buffer --connect-timeout 5 \
      --max-time "$REQUEST_TIMEOUT_SECONDS" -H 'Content-Type: application/json' \
      --data-binary "@$request" "http://$HOST:$PORT/v1/chat/completions" 2>"$stderr" \
      | perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC -ne \
          'printf "%.9f\t%s", clock_gettime(CLOCK_MONOTONIC), $_' >"$output"
    finished=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
      -e 'printf "%.9f\n", clock_gettime(CLOCK_MONOTONIC)')
    printf '%s\t%s\n' "$started" "$finished" >"$timing"
}

wait_for_four_decoders() {
    local log=$1 start_byte=$2 deadline=$(( $(date +%s) + 90 )) count
    while (( $(date +%s) < deadline )); do
        count=$(tail -c "+$((start_byte + 1))" "$log" \
          | awk '/DeepSeek-V4 decode started/ && /max_tokens=256( |$)/ {n++}
            END {print n+0}')
        [[ "$count" == 4 ]] && return 0
        sleep 0.1
    done
    echo "timed out waiting for four live DeepSeek decoders" >&2
    return 1
}

extract_request_ids() {
    local log=$1 budget=$2
    perl -ne 'if (/DeepSeek-V4 request started/ && /max_tokens='"$budget"'(?: |$)/
      && /request_id=([0-9]+)/) { print "$1\n" }' "$log"
}

prime_decoder_sessions() {
    local model=$1 replica=$2 engine_dir=$3 log=$4 lane pid log_start log_end
    local prime_dir="$engine_dir/decoder-prime" barrier="$engine_dir/decoder-prime/start"
    local -a prime_pids=() prime_ids=()
    mkdir -p "$prime_dir"
    log_start=$(stat -f '%z' "$log" 2>/dev/null || stat -c '%s' "$log")
    for ((lane=1; lane<=LIVE_DECODERS; lane++)); do
        build_request "$model" "$replica" prime decoder-prime "$lane" \
          "$prime_dir/decoder-$lane.request.json"
        post_timed_sse "$prime_dir/decoder-$lane.request.json" \
          "$prime_dir/decoder-$lane.timed-sse" "$prime_dir/decoder-$lane.timing.tsv" \
          "$prime_dir/decoder-$lane.stderr" "$barrier" &
        prime_pids+=("$!")
    done
    : >"$barrier"
    for pid in "${prime_pids[@]}"; do wait "$pid"; done
    for ((lane=1; lane<=LIVE_DECODERS; lane++)); do
        python3 "$script_dir/verify_deepseek4_mixed_policy_receipt.py" \
          --canonicalize "$prime_dir/decoder-$lane.timed-sse" \
          "$prime_dir/decoder-$lane.canonical.json"
    done
    log_end=$(stat -f '%z' "$log" 2>/dev/null || stat -c '%s' "$log")
    perl -e '
      use strict; use warnings;
      my ($path, $offset, $length) = @ARGV;
      open my $stream, "<", $path or die "$path: $!";
      binmode $stream; seek($stream, $offset, 0) or die "seek: $!";
      my $read = read($stream, my $bytes, $length);
      die "short read" unless defined($read) && $read == $length;
      print $bytes;
    ' "$log" "$log_start" "$((log_end - log_start))" >"$prime_dir/server.delta.log"
    while IFS= read -r request_id; do
        [[ -n "$request_id" ]] && prime_ids+=("$request_id")
    done < <(extract_request_ids "$prime_dir/server.delta.log" "$DECODER_PRIME_MAX_TOKENS")
    [[ ${#prime_ids[@]} == LIVE_DECODERS ]] || {
        echo "decoder priming did not bind exactly four request IDs" >&2
        return 1
    }
    jq -n --argjson request_ids \
      "$(printf '%s\n' "${prime_ids[@]}" | jq -Rsc 'split("\n")|map(select(length>0)|tonumber)')" \
      '{schema:1,max_tokens:1,request_ids:$request_ids}' >"$prime_dir/prime.json"
}

run_wave() {
    local arm=$1 model=$2 replica=$3 trial=$4 wave_dir=$5 log=$6
    local lane pid log_start log_end started finished wall decoder_ids prefill_ids
    local decoder_skew prefill_skew
    local decoder_barrier="$wave_dir/decoder.start" prefill_barrier="$wave_dir/prefill.start"
    mkdir -p "$wave_dir"
    log_start=$(stat -f '%z' "$log" 2>/dev/null || stat -c '%s' "$log")
    wave_pids=()
    for ((lane=1; lane<=LIVE_DECODERS; lane++)); do
        build_request "$model" "$replica" "$trial" decoder "$lane" \
          "$wave_dir/decoder-$lane.request.json"
        post_timed_sse "$wave_dir/decoder-$lane.request.json" \
          "$wave_dir/decoder-$lane.timed-sse" "$wave_dir/decoder-$lane.timing.tsv" \
          "$wave_dir/decoder-$lane.stderr" "$decoder_barrier" &
        wave_pids+=("$!")
    done
    : >"$decoder_barrier"
    wait_for_four_decoders "$log" "$log_start"
    for pid in "${wave_pids[@]}"; do kill -0 "$pid" 2>/dev/null || {
        echo "a decoder exited before the four-prefill launch" >&2; return 1; }; done
    for ((lane=1; lane<=PREFILLERS; lane++)); do
        build_request "$model" "$replica" "$trial" prefill "$lane" \
          "$wave_dir/prefill-$lane.request.json"
        post_timed_sse "$wave_dir/prefill-$lane.request.json" \
          "$wave_dir/prefill-$lane.timed-sse" "$wave_dir/prefill-$lane.timing.tsv" \
          "$wave_dir/prefill-$lane.stderr" "$prefill_barrier" &
        wave_pids+=("$!")
    done
    : >"$prefill_barrier"
    for pid in "${wave_pids[@]}"; do wait "$pid"; done
    for kind in decoder prefill; do
        if [[ "$kind" == decoder ]]; then count=$LIVE_DECODERS; else count=$PREFILLERS; fi
        for ((lane=1; lane<=count; lane++)); do
            python3 "$script_dir/verify_deepseek4_mixed_policy_receipt.py" \
              --canonicalize "$wave_dir/$kind-$lane.timed-sse" \
              "$wave_dir/$kind-$lane.canonical.json"
        done
    done
    log_end=$(stat -f '%z' "$log" 2>/dev/null || stat -c '%s' "$log")
    perl -e '
      use strict; use warnings;
      my ($path, $offset, $length) = @ARGV;
      open my $stream, "<", $path or die "$path: $!";
      binmode $stream; seek($stream, $offset, 0) or die "seek: $!";
      my $read = read($stream, my $bytes, $length);
      die "short read" unless defined($read) && $read == $length;
      print $bytes;
    ' "$log" "$log_start" "$((log_end - log_start))" >"$wave_dir/server.delta.log"
    decoder_ids=()
    while IFS= read -r request_id; do
        [[ -n "$request_id" ]] && decoder_ids+=("$request_id")
    done < <(extract_request_ids "$wave_dir/server.delta.log" 256)
    prefill_ids=()
    while IFS= read -r request_id; do
        [[ -n "$request_id" ]] && prefill_ids+=("$request_id")
    done < <(extract_request_ids "$wave_dir/server.delta.log" 8)
    [[ ${#decoder_ids[@]} == 4 && ${#prefill_ids[@]} == 4 ]] || {
        echo "wave $trial did not bind exactly four decoder and four prefill request IDs" >&2
        return 1
    }
    started=$(awk -F '\t' 'NR==1 || $1<min {min=$1} END{print min}' \
      "$wave_dir"/*.timing.tsv)
    finished=$(awk -F '\t' 'NR==1 || $2>max {max=$2} END{print max}' \
      "$wave_dir"/*.timing.tsv)
    wall=$(awk -v start="$started" -v finish="$finished" 'BEGIN{printf "%.9f",finish-start}')
    decoder_skew=$(awk -F '\t' 'NR==1 {min=$1;max=$1} $1<min {min=$1} $1>max {max=$1}
      END {printf "%.9f",max-min}' "$wave_dir"/decoder-*.timing.tsv)
    prefill_skew=$(awk -F '\t' 'NR==1 {min=$1;max=$1} $1<min {min=$1} $1>max {max=$1}
      END {printf "%.9f",max-min}' "$wave_dir"/prefill-*.timing.tsv)
    awk -v value="$decoder_skew" 'BEGIN{exit !(value<=0.100)}' || return 1
    awk -v value="$prefill_skew" 'BEGIN{exit !(value<=0.100)}' || return 1
    cooperative=$(rg -c 'DeepSeek-V4 cooperative prefill complete.*bounded_mixed=true' \
      "$wave_dir/server.delta.log" || true)
    cooperative=${cooperative:-0}
    if [[ "$arm" == on ]]; then
        ((cooperative > 0)) || { echo "ON wave published no cooperative Mixed cohort" >&2; return 1; }
    else
        ((cooperative == 0)) || { echo "OFF wave published a cooperative Mixed cohort" >&2; return 1; }
    fi
    jq -n --arg arm "$arm" --argjson trial "$trial" --argjson wall "$wall" \
      --argjson decoder_skew "$decoder_skew" --argjson prefill_skew "$prefill_skew" \
      --argjson decoder_ids "$(printf '%s\n' "${decoder_ids[@]}" | jq -Rsc 'split("\n")|map(select(length>0)|tonumber)')" \
      --argjson prefill_ids "$(printf '%s\n' "${prefill_ids[@]}" | jq -Rsc 'split("\n")|map(select(length>0)|tonumber)')" \
      --argjson cooperative "$cooperative" '{schema:1,trial:$trial,arm:$arm,
        live_decoders:4,prefillers:4,decoder_request_ids:$decoder_ids,
        prefill_request_ids:$prefill_ids,wall_seconds:$wall,
        decoder_launch_skew_seconds:$decoder_skew,
        prefill_launch_skew_seconds:$prefill_skew,
        cooperative_transactions:$cooperative}' >"$wave_dir/wave.json"
}

power_monitor_while_pid() {
    local watched_pid=$1 output=$2 phase=$3
    while kill -0 "$watched_pid" 2>/dev/null; do sample_fast_power "$output" "$phase"; sleep "$SAMPLE_SECONDS"; done
    sample_fast_power "$output" "$phase-end"
}

resource_monitor_while_pid() {
    local watched_pid=$1 engine_dir=$2 phase=$3
    while kill -0 "$watched_pid" 2>/dev/null; do
        sample_fast_power "$engine_dir/power-measurement.tsv" "$phase"
        memory_sample "$engine_dir/memory-measurement.tsv" "$phase"
        ps -p "$server_pid" -o rss= | awk 'NF==1 && $1~/^[0-9]+$/ {print $1}' \
          >>"$engine_dir/rss-kib"
        sleep "$SAMPLE_SECONDS"
    done
    sample_fast_power "$engine_dir/power-measurement.tsv" "$phase-end"
    memory_sample "$engine_dir/memory-measurement.tsv" "$phase-end"
    ps -p "$server_pid" -o rss= | awk 'NF==1 && $1~/^[0-9]+$/ {print $1}' \
      >>"$engine_dir/rss-kib"
}

run_measurements() {
    local label=$1 arm=$2 model=$3 engine_dir=$4 trial
    local replica=${label##*-}
    for ((trial=1; trial<=TRIALS; trial++)); do
        run_wave "$arm" "$model" "$replica" "$trial" \
          "$engine_dir/waves/$trial" "$engine_dir/server.stderr"
    done
}

run_process() {
    local label=$1 arm=$2 mixed=0 expected=false
    local replica=${label##*-}
    local engine_dir="$OUT_DIR/$label"
    local expected_selection=explicit-off
    local model producer_pid settle_pid power_pid resource_pid monitor_status=0 producer_status=0
    local measured_server_pid
    mkdir -p "$engine_dir/waves" "$engine_dir/runtime-cache"
    : >"$engine_dir/rss-kib"; : >"$engine_dir/power-settle.tsv"
    : >"$engine_dir/power-measurement.tsv"; : >"$engine_dir/memory-measurement.tsv"
    if [[ "$arm" == on ]]; then
        mixed=1
        expected=true
        expected_selection=explicit-on
    fi
    env -i HOME="$RUNTIME_HOME" PATH="$RUNTIME_PATH" TMPDIR="$RUNTIME_TMPDIR" \
      LANG=C LC_ALL=C USER=hf2q-gate LOGNAME=hf2q-gate RUST_BACKTRACE=1 \
      HF2Q_DEEPSEEK_MIXED_COHORT="$mixed" \
      HF2Q_MODEL_VERIFICATION_RECEIPT="$OUT_DIR/model-verification.json" \
      "$HF2Q_BIN" -v serve --model "$MODEL_PATH" --cache-dir "$engine_dir/runtime-cache" \
      --host "$HOST" --port "$PORT" --ctx 262144 --overflow-policy reject \
      --scheduler inflight-batched --max-slots "$MAX_SLOTS" \
      --kv-cache-budget "$KV_CACHE_BUDGET_BYTES" --default-repetition-penalty 1 \
      --default-tool-thinking-token-budget 8 \
      >"$engine_dir/server.stdout" 2>"$engine_dir/server.stderr" &
    server_pid=$!
    for ((waited=0; waited<READY_TIMEOUT_SECONDS; waited++)); do
        curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null 2>&1 && break
        kill -0 "$server_pid" 2>/dev/null || { tail -n 100 "$engine_dir/server.stderr" >&2; return 1; }
        sleep 1
    done
    curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null || return 1
    curl --fail --silent "http://$HOST:$PORT/v1/models" >"$engine_dir/models.json"
    model=$(jq -er '[.data[]|select(.loaded==true)]|if length==1 then .[0].id else error("loaded model") end' \
      "$engine_dir/models.json")
    [[ "$(jq -er --arg id "$model" '.data[]|select(.id==$id and .loaded==true)|.arch' \
      "$engine_dir/models.json")" == deepseek4 ]] || return 1
    EXPECTED="$expected" EXPECTED_SELECTION="$expected_selection" perl -ne '
      if (/DeepSeek-V4 full-context session worker started/) {
        $seen++; $slots=$1 if /slots=([0-9]+)/; $mixed=$1 if /mixed_cohort=(true|false)/;
        $selection=$1 if /mixed_cohort_selection="?([a-z-]+)"?/;
        $rows=$1 if /mixed_cohort_rows_per_lane=([0-9]+)/;
      }
      END { exit 1 unless $seen==1 && $slots==8 && $mixed eq $ENV{EXPECTED}
        && $selection eq $ENV{EXPECTED_SELECTION} && $rows==128 }
    ' "$engine_dir/server.stderr" || { echo "$label startup policy was not proven" >&2; return 1; }
    qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" "$HF2Q_BIN" \
      "$MODEL_PATH" "$MAX_SLOTS"
    printf '%s\n' "$server_pid" >"$engine_dir/server.pid"
    ps -ww -p "$server_pid" -o command= >"$engine_dir/server-command.txt"
    assert_identity
    prime_decoder_sessions "$model" "$replica" "$engine_dir" "$engine_dir/server.stderr"
    for ((warmup=1; warmup<=WARMUPS; warmup++)); do
        run_wave "$arm" "$model" "$replica" "-$warmup" \
          "$engine_dir/waves/warmup-$warmup" "$engine_dir/server.stderr"
    done
    (
      thermal_wait_for_nominal "$engine_dir/thermal-settle.tsv" "$label-settle" \
        "$THERMAL_SETTLE_SECONDS" "$THERMAL_SETTLE_TIMEOUT_SECONDS" "$SAMPLE_SECONDS" \
        "$engine_dir/contention-settle.tsv" \
        "$HOST_CONTENTION_GATE_OWNER_PID" "$server_pid"
    ) &
    settle_pid=$!
    power_monitor_while_pid "$settle_pid" "$engine_dir/power-settle.tsv" "$label-settle" &
    power_pid=$!
    wait "$settle_pid"; wait "$power_pid"
    thermal_sample "$engine_dir/thermal-measurement.tsv" "$label-measurement-start"
    host_contention_sample "$engine_dir/contention-measurement.tsv" \
      "$label-measurement-start" "$HOST_CONTENTION_GATE_OWNER_PID" \
      "$THERMAL_SAMPLED_AT" "$server_pid"
    host_contention_require_quiet "$label-measurement-start"
    (run_measurements "$label" "$arm" "$model" "$engine_dir") &
    producer_pid=$!
    resource_monitor_while_pid "$producer_pid" "$engine_dir" "$label-measurement" &
    resource_pid=$!
    thermal_monitor_fair_or_better_while_pid "$engine_dir/thermal-measurement.tsv" \
      "$label-measurement" "$producer_pid" "$SAMPLE_SECONDS" \
      "$engine_dir/contention-measurement.tsv" \
      "$HOST_CONTENTION_GATE_OWNER_PID" "$server_pid" || monitor_status=$?
    wait "$producer_pid" || producer_status=$?
    wait "$resource_pid" || monitor_status=1
    ((producer_status == 0 && monitor_status == 0)) || return 1
    thermal_sample "$engine_dir/thermal-measurement.tsv" "$label-measurement-end"
    host_contention_sample "$engine_dir/contention-measurement.tsv" \
      "$label-measurement-end" "$HOST_CONTENTION_GATE_OWNER_PID" \
      "$THERMAL_SAMPLED_AT" "$server_pid"
    host_contention_require_quiet "$label-measurement-end"
    thermal_validate_settle_log "$engine_dir/thermal-settle.tsv" \
      "$THERMAL_SETTLE_SECONDS" "$MAX_SAMPLE_GAP_SECONDS"
    host_contention_validate_settle_log "$engine_dir/contention-settle.tsv" \
      "$THERMAL_SETTLE_SECONDS" "$MAX_SAMPLE_GAP_SECONDS"
    thermal_validate_fair_or_better_measurement_log \
      "$engine_dir/thermal-measurement.tsv" "$MAX_SAMPLE_GAP_SECONDS"
    host_contention_validate_measurement_log \
      "$engine_dir/contention-measurement.tsv" "$MAX_SAMPLE_GAP_SECONDS"
    host_contention_validate_thermal_alignment "$engine_dir/thermal-measurement.tsv" \
      "$engine_dir/contention-measurement.tsv"
    memory_validate_warning_log "$engine_dir/memory-measurement.tsv" \
      "$MAX_SAMPLE_GAP_SECONDS" 0
    qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" "$HF2Q_BIN" \
      "$MODEL_PATH" "$MAX_SLOTS"
    qwen36_reject_fatal_log "$engine_dir/server.stderr"
    assert_identity
    measured_server_pid=$server_pid
    cleanup_server
    [[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]] || return 1

    for ((trial=1; trial<=TRIALS; trial++)); do
        jq -er .wall_seconds "$engine_dir/waves/$trial/wave.json"
    done >"$engine_dir/wave-samples-seconds"
    peak_rss_bytes=$(sort -n "$engine_dir/rss-kib" | tail -1 | awk '{printf "%.0f",$1*1024}')
    ((peak_rss_bytes <= MAX_PEAK_RSS_BYTES)) || return 1
    cooperative=$(rg 'DeepSeek-V4 cooperative prefill complete.*bounded_mixed=true' \
      "$engine_dir"/waves/[1-5]/server.delta.log | wc -l | tr -d ' ' || true)
    cooperative=${cooperative:-0}
    jq -n --arg label "$label" --arg arm "$arm" --argjson pid "$measured_server_pid" \
      --argjson port "$PORT" --arg binary_sha "$binary_sha256" --arg model_sha "$MODEL_SHA256" \
      --argjson mixed "$expected" --argjson peak "$peak_rss_bytes" \
      --argjson samples "$(jq -Rsc 'split("\n")|map(select(length>0)|tonumber)' "$engine_dir/wave-samples-seconds")" \
      --argjson median "$(median_file "$engine_dir/wave-samples-seconds")" \
      --argjson cooperative "$cooperative" '{schema:1,label:$label,arm:$arm,
        server:{pid:$pid,port:$port,max_slots:8,binary_sha256:$binary_sha,model_sha256:$model_sha},
        policy:{mixed_cohort:$mixed,rows_per_lane:128},sampled_peak_rss_bytes:$peak,
        wave_samples_seconds:$samples,wave_median_seconds:$median,
        cooperative_transactions:$cooperative,evidence_manifest_sha256:"pending"}' \
      >"$engine_dir/summary.json"
    (
      cd "$engine_dir"
      find . -type f ! -name evidence.sha256 ! -name summary.json \
        | sed 's#^./##' | sort | while IFS= read -r relative; do
          printf '%s  %s\n' "$(sha256_file "$relative")" "$relative"
        done >evidence.sha256.tmp
      mv evidence.sha256.tmp evidence.sha256
    )
    manifest_sha=$(sha256_file "$engine_dir/evidence.sha256")
    jq --arg sha "$manifest_sha" '.evidence_manifest_sha256=$sha' \
      "$engine_dir/summary.json" >"$engine_dir/summary.json.tmp"
    mv "$engine_dir/summary.json.tmp" "$engine_dir/summary.json"
}

thermal_prepare_probe
qwen36_start_power_guard "$HOST_CONTENTION_GATE_OWNER_PID" \
  "$OUT_DIR/caffeinate.log"
caffeinate_started=true
HF2Q_MODEL_VERIFICATION_BINARY="$HF2Q_BIN" hf2q_release_prepare_model_verification \
  "$MODEL_PATH" "$MODEL_SHA256" "$OUT_DIR/model-verification.json" \
  "$OUT_DIR/model-verification-cache"
run_process off-a off
run_process on-a on
run_process on-b on
run_process off-b off

for replica in a b; do
    for relative in decoder-prime/decoder-{1,2,3,4}.request.json \
      decoder-prime/decoder-{1,2,3,4}.canonical.json; do
        cmp -s "$OUT_DIR/off-$replica/$relative" "$OUT_DIR/on-$replica/$relative" || {
            echo "OFF/ON decoder-prime bytes differ: replica=$replica file=$relative" >&2
            exit 1
        }
    done
    while IFS= read -r relative; do
        cmp -s "$OUT_DIR/off-$replica/$relative" "$OUT_DIR/on-$replica/$relative" || {
            echo "OFF/ON request bytes differ: replica=$replica file=$relative" >&2; exit 1; }
    done < <(cd "$OUT_DIR/off-$replica" && find waves/[1-5] -name '*.request.json' -print | sort)
done
off_samples="$OUT_DIR/off-wave-samples-seconds"; on_samples="$OUT_DIR/on-wave-samples-seconds"
cat "$OUT_DIR/off-a/wave-samples-seconds" "$OUT_DIR/off-b/wave-samples-seconds" >"$off_samples"
cat "$OUT_DIR/on-a/wave-samples-seconds" "$OUT_DIR/on-b/wave-samples-seconds" >"$on_samples"
speedup=$(awk -v off="$(median_file "$off_samples")" -v on="$(median_file "$on_samples")" \
  'BEGIN{printf "%.15f",off/on}')
neighbor_a=$(awk -v off="$(median_file "$OUT_DIR/off-a/wave-samples-seconds")" \
  -v on="$(median_file "$OUT_DIR/on-a/wave-samples-seconds")" \
  'BEGIN{printf "%.15f",off/on}')
neighbor_b=$(awk -v off="$(median_file "$OUT_DIR/off-b/wave-samples-seconds")" \
  -v on="$(median_file "$OUT_DIR/on-b/wave-samples-seconds")" \
  'BEGIN{printf "%.15f",off/on}')
awk -v value="$speedup" -v minimum="$MIN_WAVE_SPEEDUP" \
  'BEGIN{exit !(value>minimum)}' || exit 1
awk -v value="$neighbor_a" -v minimum="$MIN_WAVE_SPEEDUP" \
  'BEGIN{exit !(value>minimum)}' || exit 1
awk -v value="$neighbor_b" -v minimum="$MIN_WAVE_SPEEDUP" \
  'BEGIN{exit !(value>minimum)}' || exit 1

semantic_sha=$(python3 - "$OUT_DIR" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); values=[]
for replica in ("a","b"):
  for lane in range(1,5):
    value=json.loads((root/f"on-{replica}/decoder-prime/decoder-{lane}.canonical.json").read_text())
    values.append({k:value[k] for k in ("role_events","content","reasoning_content","tool_calls","finish_reason","usage","done_count")})
  for trial in range(1,6):
    for kind in ("decoder","prefill"):
      for lane in range(1,5):
        value=json.loads((root/f"on-{replica}/waves/{trial}/{kind}-{lane}.canonical.json").read_text())
        values.append({k:value[k] for k in ("role_events","content","reasoning_content","tool_calls","finish_reason","usage","done_count")})
payload=(json.dumps(values,sort_keys=True,separators=(",",":"))+"\n").encode()
print(hashlib.sha256(payload).hexdigest())
PY
)
assert_identity
qwen36_assert_power_guard
qwen36_stop_power_guard
caffeinate_started=false

(
  cd "$OUT_DIR"
  for relative in \
    model-verification.json \
    "model-verification-cache/$MODEL_SHA256.json" \
    caffeinate.log \
    caffeinate.log.assertions \
    caffeinate.log.power-events.baseline \
    caffeinate.log.power-events.final \
    caffeinate.log.power-events.new \
    off-wave-samples-seconds \
    on-wave-samples-seconds; do
      [[ -f "$relative" && ! -L "$relative" ]] || exit 1
      printf '%s  %s\n' "$(sha256_file "$relative")" "$relative"
  done >shared-evidence.sha256.tmp
  mv shared-evidence.sha256.tmp shared-evidence.sha256
)
shared_manifest_sha=$(sha256_file "$OUT_DIR/shared-evidence.sha256")

jq -n --arg source_root "$SOURCE_ROOT" --arg commit "$source_commit" \
  --arg binary "$HF2Q_BIN" --arg binary_sha "$binary_sha256" \
  --arg model "$MODEL_PATH" --arg model_sha "$MODEL_SHA256" \
  --arg model_snapshot "$model_snapshot" --argjson model_bytes "$MODEL_BYTES" \
  --arg semantic_sha "$semantic_sha" --argjson speedup "$speedup" \
  --arg host "$HOST" --argjson port "$PORT" --arg shared_manifest "$shared_manifest_sha" \
  --argjson mixed_rows "$MIXED_ROWS" \
  --argjson scheduler_gap "$MAX_SCHEDULER_GAP_MS" \
  --argjson semantic_gap "$MAX_SEMANTIC_SSE_GAP_MS" \
  --argjson prefill_wall "$MAX_PREFILL_WALL_SECONDS" \
  --argjson max_rss "$MAX_PEAK_RSS_BYTES" \
  --argjson min_speedup "$MIN_WAVE_SPEEDUP" \
  --arg contention_policy "$HOST_CONTENTION_POLICY" \
  --argjson contention_max_foreign_cpu_percent \
    "$HOST_CONTENTION_MAX_FOREIGN_CPU_PERCENT" \
  --argjson contention_owner_pgid "$HOST_CONTENTION_GATE_OWNER_PID" \
  --argjson neighbor_a "$neighbor_a" --argjson neighbor_b "$neighbor_b" \
  --argjson off_samples "$(jq -Rsc 'split("\n")|map(select(length>0)|tonumber)' "$off_samples")" \
  --argjson on_samples "$(jq -Rsc 'split("\n")|map(select(length>0)|tonumber)' "$on_samples")" \
  --arg off_a_summary "$(sha256_file "$OUT_DIR/off-a/summary.json")" \
  --arg on_a_summary "$(sha256_file "$OUT_DIR/on-a/summary.json")" \
  --arg on_b_summary "$(sha256_file "$OUT_DIR/on-b/summary.json")" \
  --arg off_b_summary "$(sha256_file "$OUT_DIR/off-b/summary.json")" \
  --arg off_a_manifest "$(sha256_file "$OUT_DIR/off-a/evidence.sha256")" \
  --arg on_a_manifest "$(sha256_file "$OUT_DIR/on-a/evidence.sha256")" \
  --arg on_b_manifest "$(sha256_file "$OUT_DIR/on-b/evidence.sha256")" \
  --arg off_b_manifest "$(sha256_file "$OUT_DIR/off-b/evidence.sha256")" '{
    schema:1,verdict:"pass",gate:"deepseek4-mixed-policy-abba-http",
    source:{root:$source_root,commit:$commit,binary:$binary,sha256:$binary_sha},
    model:{path:$model,sha256:$model_sha,bytes:$model_bytes,snapshot:$model_snapshot},
    endpoint:{host:$host,port:$port},
    workload:{process_order:["off-a","on-a","on-b","off-b"],same_binary:true,
      trials_per_process:5,max_slots:8,live_decoders:4,prefillers:4,
      mixed_rows_per_lane:$mixed_rows,temperature:0,seed:42,
      decoder_prime:{lanes:4,max_tokens:1,stable_prompt_required:true,
        cache_reuse_required:true}},
    environment:{host_contention:{policy:$contention_policy,
      maximum_foreign_cpu_percent:$contention_max_foreign_cpu_percent,
      owner_scope:"release-gate-process-group",
      owner_pgid:$contention_owner_pgid,continuous:true}},
    thresholds:{scheduler_decode_gap_ms:$scheduler_gap,semantic_sse_gap_ms:$semantic_gap,
      max_prefill_wall_seconds:$prefill_wall,max_peak_rss_bytes:$max_rss,
      min_wave_speedup:$min_speedup},
    equality:{semantic_and_token_sha256:$semantic_sha},
    evidence:{shared_manifest_sha256:$shared_manifest,processes:{
      "off-a":{summary_sha256:$off_a_summary,manifest_sha256:$off_a_manifest},
      "on-a":{summary_sha256:$on_a_summary,manifest_sha256:$on_a_manifest},
      "on-b":{summary_sha256:$on_b_summary,manifest_sha256:$on_b_manifest},
      "off-b":{summary_sha256:$off_b_summary,manifest_sha256:$off_b_manifest}}},
    result:{wave_speedup:$speedup,neighboring_process_speedups:[$neighbor_a,$neighbor_b],
      off_wave_samples_seconds:$off_samples,on_wave_samples_seconds:$on_samples}}
  ' >"$OUT_DIR/receipt.json.tmp"
mv "$OUT_DIR/receipt.json.tmp" "$OUT_DIR/receipt.json"
python3 "$script_dir/verify_deepseek4_mixed_policy_receipt.py" \
  "$OUT_DIR/receipt.json" "$SOURCE_ROOT"
"$script_dir/test_deepseek4_mixed_policy_receipt_mutations.sh" \
  "$OUT_DIR/receipt.json" "$SOURCE_ROOT"
jq . "$OUT_DIR/receipt.json"
echo "receipt: $OUT_DIR/receipt.json" >&2
