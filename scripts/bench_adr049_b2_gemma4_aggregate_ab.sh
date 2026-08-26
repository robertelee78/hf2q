#!/usr/bin/env bash
set -euo pipefail

# Direct production-route discriminator for Gemma4 cold cross-slot admission.
# Every OFF/ON arm gets a fresh process. Pair order alternates to distribute
# load/thermal drift, while each process serves the same three four-lane cold
# waves. Raw requests, responses, timings, trace slices, and process logs are
# sealed before the independent verifier computes any speedup.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$script_dir/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/qwen38_matched_reference_contract.sh
source "$script_dir/qwen38_matched_reference_contract.sh"

readonly PAIRS=8
readonly WIDTHS=(128 256 512)
readonly LANES=4
readonly HOST=127.0.0.1
readonly READY_TIMEOUT_SECONDS=300
readonly REQUEST_TIMEOUT_SECONDS=300
readonly KV_CACHE_BUDGET_BYTES=51539607552
readonly THERMAL_SETTLE_SECONDS=60
readonly THERMAL_SAMPLE_SECONDS=2
readonly POWER_PROBE_ATTEMPTS=3
readonly MIN_LOWER_95_SPEEDUP=1.05
readonly TRACE_NAME='[PREFILL_TIMING] BATCHED 4 seqs in '
readonly RUNTIME_PATH=/usr/bin:/bin:/usr/sbin:/sbin
readonly RUNTIME_TMPDIR=/var/tmp

SOURCE_ROOT=${SOURCE_ROOT:?absolute clean source worktree is required}
EXPECTED_SOURCE_SHA=${EXPECTED_SOURCE_SHA:?exact 40-character source SHA is required}
HF2Q_BIN=${HF2Q_BIN:?exact server binary path is required}
EXPECTED_BINARY_SHA256=${EXPECTED_BINARY_SHA256:?exact binary SHA-256 is required}
MODEL_PATH=${MODEL_PATH:?exact Gemma4 MoE GGUF path is required}
EXPECTED_MODEL_SHA256=${EXPECTED_MODEL_SHA256:?exact model SHA-256 is required}
EXPECTED_MODEL_BYTES=${EXPECTED_MODEL_BYTES:?exact model byte count is required}
PORT=${PORT:?dedicated local port is required}
OUT_DIR=${OUT_DIR:?new absolute receipt directory is required}
launcher="$SOURCE_ROOT/scripts/serve_gemma4_opencode.sh"

for command in awk caffeinate curl date dd env find git grep jq kill lsof \
    mkdir mv perl pgrep pmset ps python3 rg shasum stat system_profiler tail; do
    command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done
for path in "$SOURCE_ROOT" "$HF2Q_BIN" "$MODEL_PATH" "$OUT_DIR"; do
    [[ "$path" == /* ]] || { echo "filesystem inputs must be absolute: $path" >&2; exit 2; }
done
[[ "$EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ \
    && "$EXPECTED_BINARY_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$EXPECTED_MODEL_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$EXPECTED_MODEL_BYTES" =~ ^[1-9][0-9]*$ \
    && "$PORT" =~ ^[1-9][0-9]*$ ]] || { echo "malformed exact identity or port" >&2; exit 2; }
((PORT <= 65535)) || { echo "PORT exceeds 65535" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to reuse receipt directory: $OUT_DIR" >&2; exit 2; }
[[ -d "$SOURCE_ROOT" && ! -L "$SOURCE_ROOT" \
    && -x "$HF2Q_BIN" && ! -L "$HF2Q_BIN" \
    && -f "$MODEL_PATH" && ! -L "$MODEL_PATH" \
    && -x "$launcher" && ! -L "$launcher" ]] || {
    echo "source, binary, or model is not a canonical local object" >&2
    exit 2
}
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd -P)
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
file_bytes() { stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1"; }
[[ "$HF2Q_BIN" == "$SOURCE_ROOT/target/release/hf2q" \
    && "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$EXPECTED_SOURCE_SHA" \
    && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" \
    && "$(sha256_file "$HF2Q_BIN")" == "$EXPECTED_BINARY_SHA256" \
    && "$(file_bytes "$MODEL_PATH")" == "$EXPECTED_MODEL_BYTES" ]] || {
    echo "source, binary, or model identity mismatch" >&2
    exit 2
}
grep -aFq "$EXPECTED_SOURCE_SHA" "$HF2Q_BIN" || {
    echo "binary does not embed EXPECTED_SOURCE_SHA" >&2
    exit 2
}
launcher_sha256=$(sha256_file "$launcher")
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | grep -q .; then
    echo "$HOST:$PORT is already in use" >&2
    exit 2
fi
if pgrep -x hf2q >/dev/null 2>&1; then
    echo "Gemma B.2 A/B requires no pre-existing hf2q runtime" >&2
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

power_source=$(resolve_live_power_source) || {
    echo "could not resolve the live power source" >&2
    exit 2
}
[[ "$power_source" == ac ]] || {
    echo "Gemma B.2 A/B requires AC power" >&2
    exit 2
}
power_mode=$(resolve_ac_energy_mode) || {
    echo "could not resolve the active AC Energy Mode" >&2
    exit 2
}
[[ "$power_mode" != low ]] || {
    echo "Gemma B.2 A/B rejects Low Power Mode" >&2
    exit 2
}
power_mode_code=$(resolve_live_power_mode_code) || {
    echo "could not resolve the live AC power-mode canary" >&2
    exit 2
}

record_power_contract() {
    local output=$1 phase=$2 observed_source observed_mode observed_code sampled_at
    observed_source=$(resolve_live_power_source) || {
        echo "$phase could not resolve live power source" >&2
        return 1
    }
    [[ "$observed_source" == ac ]] || {
        echo "$phase observed non-AC power: $observed_source" >&2
        return 1
    }
    observed_mode=$(resolve_ac_energy_mode) || {
        echo "$phase could not resolve AC Energy Mode" >&2
        return 1
    }
    observed_code=$(resolve_live_power_mode_code) || {
        echo "$phase could not resolve live power-mode code" >&2
        return 1
    }
    [[ "$observed_mode" == "$power_mode" \
        && "$observed_code" == "$power_mode_code" ]] || {
        echo "$phase observed power-mode drift" >&2
        return 1
    }
    sampled_at=$(date +%s)
    printf '%s\tac\t%s\t%s\t%s\n' "$sampled_at" "$observed_mode" \
        "$observed_code" "$phase" >>"$output"
}

assert_identity() {
    [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$EXPECTED_SOURCE_SHA" \
        && -z "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=all)" \
        && "$(sha256_file "$HF2Q_BIN")" == "$EXPECTED_BINARY_SHA256" \
        && "$(sha256_file "$launcher")" == "$launcher_sha256" \
        && "$(file_bytes "$MODEL_PATH")" == "$EXPECTED_MODEL_BYTES" \
        && "$(stat -f '%d:%i:%z:%m:%c' "$MODEL_PATH" 2>/dev/null \
            || stat -c '%d:%i:%s:%Y:%Z' "$MODEL_PATH")" == "$model_snapshot" ]] || {
        echo "source, binary, launcher, or model identity changed during A/B" >&2
        return 1
    }
    if [[ "$(sha256_file "$model_verification_receipt")" != "$model_verification_sha256" ]] \
        || ! hf2q_release_verify_model \
            "$MODEL_PATH" "$EXPECTED_MODEL_SHA256" "$model_verification_receipt"; then
        echo "model verification receipt changed or no longer verifies" >&2
        return 1
    fi
}

mkdir -p "$OUT_DIR/processes"
OUT_DIR=$(cd "$OUT_DIR" && pwd -P)
model_verification_receipt="$OUT_DIR/model-verification.json"
export HF2Q_MODEL_VERIFICATION_BINARY="$HF2Q_BIN"
hf2q_release_record_model_verification \
    "$MODEL_PATH" "$EXPECTED_MODEL_SHA256" "$model_verification_receipt"
hf2q_release_verify_model \
    "$MODEL_PATH" "$EXPECTED_MODEL_SHA256" "$model_verification_receipt"
model_verification_sha256=$(sha256_file "$model_verification_receipt")
model_snapshot=$(jq -er .file_snapshot "$model_verification_receipt")
samples="$OUT_DIR/samples.jsonl"
process_bindings="$OUT_DIR/processes.jsonl"
settle_log="$OUT_DIR/thermal-settle.log"
measurement_log="$OUT_DIR/thermal-measurement.log"
contention_settle_log="$OUT_DIR/contention-settle.log"
contention_measurement_log="$OUT_DIR/contention-measurement.log"
: >"$samples"
: >"$process_bindings"
: >"$measurement_log"
: >"$contention_measurement_log"

server_pid=""
producer_pid=""
power_guard_started=false
STOP_WAIT_STATUS=0
cleanup() {
    if [[ -n "$producer_pid" ]]; then kill -TERM "$producer_pid" 2>/dev/null || true; fi
    if [[ -n "$server_pid" ]]; then kill -TERM "$server_pid" 2>/dev/null || true; fi
    if [[ "$power_guard_started" == true ]]; then
        qwen36_stop_power_guard >/dev/null 2>&1 || true
        power_guard_started=false
    fi
    thermal_cleanup_probe >/dev/null 2>&1 || true
}
trap cleanup EXIT

build_request() {
    local model_id=$1 pair=$2 target=$3 lane=$4 output=$5 content
    content=$(awk -v pair="$pair" -v target="$target" -v lane="$lane" 'BEGIN {
        printf "adr049-b2-gemma-p%02d-w%03d-l%d ", pair, target, lane
        for (i = 1; i <= target; i++) printf "measurement "
        printf "Reply with one word."
    }')
    jq -n --arg model "$model_id" --arg content "$content" '{
      model:$model,messages:[{role:"user",content:$content}],
      max_tokens:1,seed:42,temperature:0,repetition_penalty:1,stream:false,
      hf2q_enable_thinking:false,chat_template_kwargs:{enable_thinking:false}
    }' >"$output"
}

stop_server() {
    local wait_status=0
    if [[ -n "$server_pid" ]]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        set +e
        wait "$server_pid"
        wait_status=$?
        set -e
        server_pid=""
    fi
    STOP_WAIT_STATUS=$wait_status
}

run_wave() {
    local pair=$1 position=$2 arm=$3 target=$4 model_id=$5 process_dir=$6
    local width_position sample_dir log_before log_after wave_started wave_ended wave_seconds wave_ms
    local trace event_count trace_requests trace_elapsed lanes_file lane request response wall normalized timing
    local prompt cached work prefill_ms ttft_ms wall_ms aggregate_rows=0
    local barrier launch_skew latest_start earliest_finish actual_overlap
    local -a request_pids
    case "$target" in 128) width_position=0 ;; 256) width_position=1 ;; 512) width_position=2 ;; esac
    sample_dir="$process_dir/wave-$target"
    mkdir -p "$sample_dir"
    log_before=$(file_bytes "$process_dir/server.stderr")
    wave_started=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
    request_pids=()
    barrier="$sample_dir/start.barrier"
    rm -f "$barrier"
    for ((lane = 0; lane < LANES; lane++)); do
        request="$sample_dir/lane-$lane.request.json"
        response="$sample_dir/lane-$lane.response.json"
        wall="$sample_dir/lane-$lane.wall"
        timing="$sample_dir/lane-$lane.timing"
        build_request "$model_id" "$pair" "$target" "$lane" "$request"
        (
            while [[ ! -e "$barrier" ]]; do sleep 0.001; done
            lane_started=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
            curl --fail-with-body --silent --show-error --connect-timeout 5 \
                --max-time "$REQUEST_TIMEOUT_SECONDS" -H 'Content-Type: application/json' \
                --data-binary "@$request" -o "$response" -w '%{time_total}\n' \
                "http://$HOST:$PORT/v1/chat/completions" >"$wall"
            lane_finished=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
            printf '%s\t%s\n' "$lane_started" "$lane_finished" >"$timing"
        ) &
        request_pids+=("$!")
    done
    : >"$barrier"
    for request_pid in "${request_pids[@]}"; do wait "$request_pid"; done
    wave_ended=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
    wave_seconds=$(awk -v start="$wave_started" -v end="$wave_ended" 'BEGIN {printf "%.9f", end-start}')
    wave_ms=$(awk -v seconds="$wave_seconds" 'BEGIN {printf "%.9f", seconds*1000}')
    printf '%s\n' "$wave_seconds" >"$sample_dir/wave.wall"
    launch_skew=$(awk -F '\t' '
      NR == 1 {minimum=$1; maximum=$1}
      $1 < minimum {minimum=$1} $1 > maximum {maximum=$1}
      END {printf "%.9f", maximum-minimum}
    ' "$sample_dir"/lane-*.timing)
    latest_start=$(awk -F '\t' 'NR == 1 || $1 > value {value=$1} END {print value}' \
        "$sample_dir"/lane-*.timing)
    earliest_finish=$(awk -F '\t' 'NR == 1 || $2 < value {value=$2} END {print value}' \
        "$sample_dir"/lane-*.timing)
    actual_overlap=$(awk -v skew="$launch_skew" -v latest="$latest_start" \
        -v earliest="$earliest_finish" \
        'BEGIN {if (skew <= 0.100 && latest < earliest) print "true"; else print "false"}')
    [[ "$actual_overlap" == true ]] || {
        echo "pair $pair $arm width $target lacked a <=100ms overlapping launch" >&2
        return 1
    }
    log_after=$(file_bytes "$process_dir/server.stderr")
    ((log_after >= log_before)) || { echo "server log shrank during wave" >&2; return 1; }
    trace="$sample_dir/server.trace.log"
    if ((log_after == log_before)); then
        : >"$trace"
    else
        dd if="$process_dir/server.stderr" of="$trace" bs=1 skip="$log_before" \
            count="$((log_after - log_before))" 2>/dev/null
    fi
    lanes_file="$sample_dir/lanes.jsonl"
    : >"$lanes_file"
    for ((lane = 0; lane < LANES; lane++)); do
        request="$sample_dir/lane-$lane.request.json"
        response="$sample_dir/lane-$lane.response.json"
        wall="$sample_dir/lane-$lane.wall"
        timing="$sample_dir/lane-$lane.timing"
        normalized="$sample_dir/lane-$lane.normalized.json"
        jq -e '
          (.choices | length) == 1 and (.choices[0].message | type) == "object"
          and (.choices[0].finish_reason | type) == "string"
          and (.usage.prompt_tokens | numbers) > 0
          and .usage.prompt_tokens_details.cached_tokens == 0
          and (.usage.completion_tokens | numbers) == 1
          and (.x_hf2q_timing.prefill_time_secs | numbers) > 0
          and (.x_hf2q_timing.time_to_first_token_ms | numbers) > 0
        ' "$response" >/dev/null
        prompt=$(jq -er '.usage.prompt_tokens' "$response")
        cached=$(jq -er '.usage.prompt_tokens_details.cached_tokens' "$response")
        work=$((prompt - cached))
        ((work * 4 >= target * 3 && work * 4 <= target * 5)) || {
            echo "pair $pair $arm width $target lane $lane missed target bin: $work" >&2
            return 1
        }
        aggregate_rows=$((aggregate_rows + work))
        prefill_ms=$(jq -er '.x_hf2q_timing.prefill_time_secs * 1000' "$response")
        ttft_ms=$(jq -er '.x_hf2q_timing.time_to_first_token_ms' "$response")
        wall_ms=$(awk '{printf "%.9f", $1*1000}' "$wall")
        jq -S '{message:.choices[0].message,finish_reason:.choices[0].finish_reason,
          usage:(.usage|{prompt_tokens,completion_tokens,total_tokens,
            prompt_tokens_details})}' "$response" >"$normalized"
        jq -cn --argjson lane "$lane" --argjson prompt "$prompt" \
            --argjson cached "$cached" --argjson work "$work" \
            --argjson prefill "$prefill_ms" --argjson ttft "$ttft_ms" \
            --argjson wall_ms "$wall_ms" \
            --arg request "${request#"$OUT_DIR/"}" --arg request_sha "$(sha256_file "$request")" \
            --arg response "${response#"$OUT_DIR/"}" --arg response_sha "$(sha256_file "$response")" \
            --arg wall "${wall#"$OUT_DIR/"}" --arg wall_sha "$(sha256_file "$wall")" \
            --arg timing "${timing#"$OUT_DIR/"}" --arg timing_sha "$(sha256_file "$timing")" \
            --arg normalized "${normalized#"$OUT_DIR/"}" \
            --arg normalized_sha "$(sha256_file "$normalized")" '{
              lane:$lane,prompt_tokens:$prompt,cached_tokens:$cached,work_rows:$work,
              prefill_ms:$prefill,ttft_ms:$ttft,wall_ms:$wall_ms,
              request_path:$request,request_sha256:$request_sha,
              response_path:$response,response_sha256:$response_sha,
              wall_path:$wall,wall_sha256:$wall_sha,
              timing_path:$timing,timing_sha256:$timing_sha,
              normalized_path:$normalized,normalized_sha256:$normalized_sha
            }' >>"$lanes_file"
    done
    event_count=$(grep -cF "$TRACE_NAME" "$trace" || true)
    trace_requests=null
    trace_elapsed=null
    if ((event_count == 1)); then
        trace_requests=$(perl -ne 'print "$1\n" if /\[PREFILL_TIMING\] BATCHED ([0-9]+) seqs in/' "$trace")
        trace_elapsed=$(perl -ne 'print "$1\n" if /\[PREFILL_TIMING\] BATCHED [0-9]+ seqs in ([0-9]+(?:\.[0-9]+)?) ms/' "$trace")
    fi
    if [[ "$arm" == on ]]; then
        [[ "$event_count" == 1 && "$trace_requests" == 4 \
            && -n "$trace_elapsed" ]] || {
            echo "pair $pair ON width $target did not prove one four-lane aggregate" >&2
            return 1
        }
    else
        [[ "$event_count" == 0 ]] || { echo "OFF arm emitted aggregate trace" >&2; return 1; }
    fi
    jq -c -n --slurpfile lanes "$lanes_file" \
        --argjson pair "$pair" --argjson position "$position" --arg arm "$arm" \
        --argjson width_position "$width_position" --argjson target "$target" \
        --argjson wave_ms "$wave_ms" \
        --arg wall "${sample_dir#"$OUT_DIR/"}/wave.wall" \
        --arg wall_sha "$(sha256_file "$sample_dir/wave.wall")" \
        --arg trace "${trace#"$OUT_DIR/"}" --arg trace_sha "$(sha256_file "$trace")" \
        --arg lanes_path "${lanes_file#"$OUT_DIR/"}" --arg lanes_sha "$(sha256_file "$lanes_file")" \
        --argjson events "$event_count" --argjson trace_requests "$trace_requests" \
        --argjson trace_elapsed "$trace_elapsed" --argjson aggregate_rows "$aggregate_rows" \
        --argjson launch_skew "$launch_skew" --argjson latest_start "$latest_start" \
        --argjson earliest_finish "$earliest_finish" \
        --argjson actual_overlap "$actual_overlap" '{
          schema_version:1,pair:$pair,process_position:$position,arm:$arm,
          width_position:$width_position,target_rows:$target,wave_ms:$wave_ms,
          wave_wall_path:$wall,wave_wall_sha256:$wall_sha,
          trace_path:$trace,trace_sha256:$trace_sha,trace_event_count:$events,
          trace_requests:$trace_requests,trace_elapsed_ms:$trace_elapsed,
          aggregate_work_rows:$aggregate_rows,launch_skew_seconds:$launch_skew,
          latest_start:$latest_start,earliest_finish:$earliest_finish,
          actual_overlap:$actual_overlap,lanes_path:$lanes_path,
          lanes_sha256:$lanes_sha,lanes:$lanes
        }' >>"$samples"
}

run_warmup_waves() {
    local pair=$1 model_id=$2 warmup_dir warmup lane barrier pid request response wall
    local -a warmup_pids
    warmup_dir=$(mktemp -d "$RUNTIME_TMPDIR/gemma-b2-warmup.XXXXXX")
    for warmup in 1 2; do
        barrier="$warmup_dir/$warmup.start"
        warmup_pids=()
        for ((lane = 0; lane < LANES; lane++)); do
            request="$warmup_dir/$warmup-$lane.request.json"
            response="$warmup_dir/$warmup-$lane.response.json"
            wall="$warmup_dir/$warmup-$lane.wall"
            build_request "$model_id" "$((pair + 100 + warmup))" 256 "$lane" "$request"
            (
                while [[ ! -e "$barrier" ]]; do sleep 0.001; done
                curl --fail-with-body --silent --show-error --connect-timeout 5 \
                    --max-time "$REQUEST_TIMEOUT_SECONDS" \
                    -H 'Content-Type: application/json' --data-binary "@$request" \
                    -o "$response" -w '%{time_total}\n' \
                    "http://$HOST:$PORT/v1/chat/completions" >"$wall"
            ) &
            warmup_pids+=("$!")
        done
        : >"$barrier"
        for pid in "${warmup_pids[@]}"; do wait "$pid"; done
        for ((lane = 0; lane < LANES; lane++)); do
            jq -e '.usage.prompt_tokens_details.cached_tokens == 0 \
              and (.usage.completion_tokens | numbers) == 1' \
                "$warmup_dir/$warmup-$lane.response.json" >/dev/null
        done
    done
    rm -R "$warmup_dir"
}

run_arm() {
    local pair=$1 position=$2 arm=$3 cross_slot coalesce process_dir model_id command wait_status process_pid
    local runtime_home actual_arch
    if [[ "$arm" == on ]]; then cross_slot=1; coalesce=25000; else cross_slot=0; coalesce=0; fi
    process_dir="$OUT_DIR/processes/pair-$pair-$arm"
    runtime_home="$process_dir/runtime-home"
    mkdir -p "$process_dir" "$runtime_home"
    : >"$process_dir/power.tsv"
    record_power_contract "$process_dir/power.tsv" "pair-$pair-$arm-before-launch"
    env -i PATH="$RUNTIME_PATH" HOME="$runtime_home" TMPDIR="$RUNTIME_TMPDIR" \
        LANG=C LC_ALL=C USER=hf2q-gate LOGNAME=hf2q-gate RUST_BACKTRACE=1 \
        MODEL="$MODEL_PATH" MMPROJ="$process_dir/no-mmproj.gguf" \
        HOST="$HOST" PORT="$PORT" HF2Q_BIN="$HF2Q_BIN" MAX_SLOTS="$LANES" \
        KV_CACHE_BUDGET_BYTES="$KV_CACHE_BUDGET_BYTES" \
        HF2Q_CROSS_SLOT_ADMIT="$cross_slot" HF2Q_ADMIT_COALESCE_US="$coalesce" \
        HF2Q_PREFILL_TIMING=1 REP_PENALTY=1 \
        HF2Q_MODEL_VERIFICATION_RECEIPT="$model_verification_receipt" \
        "$launcher" \
        >"$process_dir/server.stdout" 2>"$process_dir/server.stderr" &
    server_pid=$!
    process_pid=$server_pid
    for ((attempt = 0; attempt < READY_TIMEOUT_SECONDS; attempt++)); do
        if curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null 2>&1; then break; fi
        kill -0 "$server_pid" 2>/dev/null || { tail -n 80 "$process_dir/server.stderr" >&2; return 1; }
        sleep 1
    done
    curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null || { echo "$arm server not ready" >&2; return 1; }
    curl --fail --silent "http://$HOST:$PORT/v1/models" >"$process_dir/models.json"
    model_id=$(jq -er '[.data[] | select(.loaded == true)] | if length == 1 then .[0].id else error("expected one loaded model") end' "$process_dir/models.json")
    actual_arch=$(jq -er --arg id "$model_id" \
        '[.data[] | select(.loaded == true and .id == $id)] \
         | if length == 1 then .[0].arch else error("expected one loaded architecture") end' \
        "$process_dir/models.json")
    [[ "$actual_arch" == gemma4 ]] || {
        echo "server loaded the wrong architecture: $actual_arch" >&2
        return 1
    }
    command=$(ps -ww -p "$server_pid" -o command=)
    [[ "$command" == *"$HF2Q_BIN"* && "$command" == *"$MODEL_PATH"* \
        && "$command" == *"--scheduler inflight-batched"* \
        && "$command" == *"--max-slots 4"* ]] || { echo "server process binding failed" >&2; return 1; }
    qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
        "$HF2Q_BIN" "$MODEL_PATH" "$LANES"
    ps -ww -p "$server_pid" -o command= >"$process_dir/server-command.txt"
    assert_identity
    run_warmup_waves "$pair" "$model_id"
    record_power_contract "$process_dir/power.tsv" "pair-$pair-$arm-loaded-warm"
    record_power_contract "$process_dir/power.tsv" "pair-$pair-$arm-measurement-start"
    for target in "${WIDTHS[@]}"; do run_wave "$pair" "$position" "$arm" "$target" "$model_id" "$process_dir"; done
    record_power_contract "$process_dir/power.tsv" "pair-$pair-$arm-measurement-end"
    qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
        "$HF2Q_BIN" "$MODEL_PATH" "$LANES"
    qwen36_reject_fatal_log "$process_dir/server.stderr"
    assert_identity
    stop_server
    wait_status=$STOP_WAIT_STATUS
    [[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]] || {
        echo "pair $pair $arm listener survived shutdown" >&2
        return 1
    }
    qwen36_reject_fatal_log "$process_dir/server.stderr"
    record_power_contract "$process_dir/power.tsv" "pair-$pair-$arm-after-shutdown"
    assert_identity
    jq -n --argjson pair "$pair" --argjson position "$position" --arg arm "$arm" \
        --argjson pid "$process_pid" --arg command "$command" --arg model_id "$model_id" \
        --arg cross_slot "$cross_slot" --arg coalesce "$coalesce" \
        --arg source_sha "$EXPECTED_SOURCE_SHA" --arg binary_sha "$EXPECTED_BINARY_SHA256" \
        --arg model_sha "$EXPECTED_MODEL_SHA256" --argjson wait_status "$wait_status" \
        --arg runtime_home "$runtime_home" \
        --arg launcher "$launcher" --arg launcher_sha "$launcher_sha256" \
        --arg model_verification "$model_verification_receipt" \
        --arg model_verification_sha "$model_verification_sha256" \
        --arg power "${process_dir#"$OUT_DIR/"}/power.tsv" --arg power_sha "$(sha256_file "$process_dir/power.tsv")" \
        --arg command_path "${process_dir#"$OUT_DIR/"}/server-command.txt" --arg command_sha "$(sha256_file "$process_dir/server-command.txt")" \
        --arg models "${process_dir#"$OUT_DIR/"}/models.json" --arg models_sha "$(sha256_file "$process_dir/models.json")" \
        --arg stdout "${process_dir#"$OUT_DIR/"}/server.stdout" --arg stdout_sha "$(sha256_file "$process_dir/server.stdout")" \
        --arg stderr "${process_dir#"$OUT_DIR/"}/server.stderr" --arg stderr_sha "$(sha256_file "$process_dir/server.stderr")" '{
          schema_version:1,status:"stopped",pair:$pair,position:$position,arm:$arm,pid:$pid,
          command:$command,model_id:$model_id,max_slots:4,
          runtime:{clean_environment:true,home:$runtime_home,
            path:"/usr/bin:/bin:/usr/sbin:/sbin",tmpdir:"/var/tmp",
            locale:{LANG:"C",LC_ALL:"C"},rust_backtrace:"1",
            operator_launcher:$launcher,operator_launcher_sha256:$launcher_sha,
            model_verification_receipt:$model_verification,
            model_verification_receipt_sha256:$model_verification_sha},
          lever_env:{HF2Q_CROSS_SLOT_ADMIT:$cross_slot,HF2Q_ADMIT_COALESCE_US:$coalesce},
          source_sha:$source_sha,binary_sha256:$binary_sha,model_sha256:$model_sha,
          wait_status:$wait_status,power_path:$power,power_sha256:$power_sha,
          command_path:$command_path,command_sha256:$command_sha,
          models_path:$models,models_sha256:$models_sha,
          stdout_path:$stdout,stdout_sha256:$stdout_sha,stderr_path:$stderr,stderr_sha256:$stderr_sha
        }' >"$process_dir/process.json"
    jq -cn --argjson pair "$pair" --argjson position "$position" --arg arm "$arm" \
        --arg path "${process_dir#"$OUT_DIR/"}/process.json" \
        --arg sha "$(sha256_file "$process_dir/process.json")" \
        '{pair:$pair,position:$position,arm:$arm,path:$path,sha256:$sha}' >>"$process_bindings"
}

run_all() {
    local pair position arm
    trap '[[ -z "$server_pid" ]] || kill -TERM "$server_pid" 2>/dev/null || true' EXIT
    for ((pair = 0; pair < PAIRS; pair++)); do
        if ((pair % 2 == 0)); then arms=(off on); else arms=(on off); fi
        for ((position = 0; position < 2; position++)); do
            arm=${arms[$position]}
            run_arm "$pair" "$position" "$arm"
        done
    done
}

thermal_prepare_probe
qwen36_start_power_guard "$$" "$OUT_DIR/caffeinate.log"
power_guard_started=true
thermal_wait_for_nominal "$settle_log" adr049-b2-gemma-ab-settle \
    "$THERMAL_SETTLE_SECONDS" 900 5 \
    "$contention_settle_log" "$$"
thermal_sample "$measurement_log" adr049-b2-gemma-ab-start
host_contention_sample "$contention_measurement_log" adr049-b2-gemma-ab-start "$$" "$THERMAL_SAMPLED_AT"
host_contention_require_quiet adr049-b2-gemma-ab-start
run_all &
producer_pid=$!
monitor_status=0
thermal_monitor_fair_or_better_while_pid "$measurement_log" \
    adr049-b2-gemma-ab-measurement "$producer_pid" "$THERMAL_SAMPLE_SECONDS" \
    "$contention_measurement_log" "$$" || monitor_status=$?
if ((monitor_status != 0)); then
    kill -TERM "$producer_pid" 2>/dev/null || true
    wait "$producer_pid" 2>/dev/null || true
    producer_pid=""
    exit "$monitor_status"
fi
producer_status=0
wait "$producer_pid" || producer_status=$?
producer_pid=""
((producer_status == 0)) || {
    echo "Gemma B.2 measurement producer failed with status $producer_status" >&2
    exit "$producer_status"
}
thermal_sample "$measurement_log" adr049-b2-gemma-ab-end
host_contention_sample "$contention_measurement_log" adr049-b2-gemma-ab-end "$$" "$THERMAL_SAMPLED_AT"
host_contention_require_quiet adr049-b2-gemma-ab-end
assert_identity
qwen36_assert_power_guard
qwen36_stop_power_guard
power_guard_started=false

jq -n --slurpfile processes "$process_bindings" \
    --arg source_root "$SOURCE_ROOT" --arg source_sha "$EXPECTED_SOURCE_SHA" \
    --arg binary_path "$HF2Q_BIN" --arg binary_sha "$EXPECTED_BINARY_SHA256" \
    --arg model_path "$MODEL_PATH" --arg model_sha "$EXPECTED_MODEL_SHA256" \
    --argjson model_bytes "$EXPECTED_MODEL_BYTES" --arg model_snapshot "$model_snapshot" \
    --arg launcher_path "$launcher" --arg launcher_sha "$launcher_sha256" \
    --arg power_mode "$power_mode" --arg power_mode_code "$power_mode_code" \
    --argjson min_lower_speedup "$MIN_LOWER_95_SPEEDUP" \
    --arg model_verification_sha "$model_verification_sha256" \
    --arg samples_sha "$(sha256_file "$samples")" \
    --arg processes_sha "$(sha256_file "$process_bindings")" \
    --arg settle_sha "$(sha256_file "$settle_log")" \
    --arg measurement_sha "$(sha256_file "$measurement_log")" \
    --arg contention_settle_sha "$(sha256_file "$contention_settle_log")" \
    --arg contention_measurement_sha "$(sha256_file "$contention_measurement_log")" \
    --arg caffeinate_log_sha "$(sha256_file "$OUT_DIR/caffeinate.log")" \
    --arg caffeinate_assertions_sha "$(sha256_file "$OUT_DIR/caffeinate.log.assertions")" \
    --arg power_events_baseline_sha "$(sha256_file "$OUT_DIR/caffeinate.log.power-events.baseline")" \
    --arg power_events_final_sha "$(sha256_file "$OUT_DIR/caffeinate.log.power-events.final")" \
    --arg power_events_new_sha "$(sha256_file "$OUT_DIR/caffeinate.log.power-events.new")" '{
      schema_version:1,status:"measured",
      configuration:{pairs:8,width_targets:[128,256,512],lanes:4,
        pair_order:"off-on-even_on-off-odd",
        warmup_waves_per_process:2,measured_waves_per_process:3,
        off_env:{HF2Q_CROSS_SLOT_ADMIT:"0",HF2Q_ADMIT_COALESCE_US:"0"},
        on_env:{HF2Q_CROSS_SLOT_ADMIT:"1",HF2Q_ADMIT_COALESCE_US:"25000"},
        request:{max_tokens:1,seed:42,temperature:0,repetition_penalty:1,stream:false,thinking:false},
        analysis:{statistic:"median paired OFF/ON wave speedup",
          order_stratified_bootstrap_samples:10000,bootstrap_seed:49004,
          lower_confidence_percentile:2.5,
          minimum_lower_95_speedup_exclusive:$min_lower_speedup}},
      identity:{source_root:$source_root,source_sha:$source_sha,source_dirty:false,
        binary_path:$binary_path,binary_sha256:$binary_sha,model_path:$model_path,
        model_sha256:$model_sha,model_bytes:$model_bytes,model_snapshot:$model_snapshot,
        operator_launcher_path:$launcher_path,operator_launcher_sha256:$launcher_sha},
      environment:{power:"ac",power_mode:$power_mode,power_mode_code:$power_mode_code,
        thermal:"nominal-settle-and-fair-or-better-measurement",
        host_contention:"quiet",clean_process_environment:true},
      processes:$processes,
      files:{samples:{path:"samples.jsonl",sha256:$samples_sha},
        model_verification:{path:"model-verification.json",sha256:$model_verification_sha},
        process_bindings:{path:"processes.jsonl",sha256:$processes_sha},
        thermal_settle:{path:"thermal-settle.log",sha256:$settle_sha},
        thermal_measurement:{path:"thermal-measurement.log",sha256:$measurement_sha},
        contention_settle:{path:"contention-settle.log",sha256:$contention_settle_sha},
        contention_measurement:{path:"contention-measurement.log",sha256:$contention_measurement_sha},
        power_guard:{caffeinate_log:{path:"caffeinate.log",sha256:$caffeinate_log_sha},
          assertions:{path:"caffeinate.log.assertions",sha256:$caffeinate_assertions_sha},
          events_baseline:{path:"caffeinate.log.power-events.baseline",sha256:$power_events_baseline_sha},
          events_final:{path:"caffeinate.log.power-events.final",sha256:$power_events_final_sha},
          events_new:{path:"caffeinate.log.power-events.new",sha256:$power_events_new_sha}}}
    }' >"$OUT_DIR/manifest.json.tmp"
mv "$OUT_DIR/manifest.json.tmp" "$OUT_DIR/manifest.json"
python3 "$script_dir/verify_adr049_b2_gemma4_aggregate_ab.py" "$OUT_DIR" "$OUT_DIR/summary.json"
echo "ADR-049 B.2 Gemma aggregate A/B receipt: $OUT_DIR/summary.json" >&2
