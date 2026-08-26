#!/usr/bin/env bash
set -euo pipefail

# Same-binary OFF/ON ABBA for the Qwen rectangular stable-boundary prefill
# policy. One invocation proves one artifact/architecture shape. Run it once
# for a dense Qwen3.8 artifact and once for a Qwen3.6 MoE artifact; the final
# ADR-049 evidence joins both receipts instead of treating either as a proxy.

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=$(cd "$script_dir/.." && pwd)
if [[ ${HF2Q_QWEN_RECTANGULAR_POLICY_GATE_ISOLATED:-0} != 1 ]]; then
    exec "$script_dir/run_release_gate_process_group.sh" env \
        HF2Q_QWEN_RECTANGULAR_POLICY_GATE_ISOLATED=1 "$0" "$@"
fi
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"
readonly HOST_CONTENTION_GATE_OWNER_PID=$$
host_contention_require_isolated_gate_owner \
    "$HOST_CONTENTION_GATE_OWNER_PID"
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
PORT=${PORT:-52849}
READY_TIMEOUT_SECONDS=${READY_TIMEOUT_SECONDS:-240}
REQUEST_TIMEOUT_SECONDS=${REQUEST_TIMEOUT_SECONDS:-120}
THERMAL_SETTLE_TIMEOUT_SECONDS=${THERMAL_SETTLE_TIMEOUT_SECONDS:-600}
readonly TRIALS=5
readonly MAX_TOKENS=2
readonly MAX_SLOTS=4
readonly COALESCE_US=25000
readonly KV_CACHE_BUDGET_BYTES=51539607552
readonly THERMAL_SETTLE_SECONDS=60
readonly THERMAL_SAMPLE_SECONDS=2
readonly POWER_PROBE_ATTEMPTS=3
readonly MIN_WAVE_SPEEDUP=1.01
readonly RUNTIME_HOME=/var/empty
readonly RUNTIME_PATH=/usr/bin:/bin:/usr/sbin:/sbin
readonly RUNTIME_TMPDIR=/var/tmp
# Two coalescing windows is a fail-closed product ceiling, not a tunable gate.
# The worker can wait at most one 25 ms window; the second window is the
# allowance for host scheduling and HTTP observation noise in a fresh process.
readonly MAX_SINGLE_OVERHEAD_MS=50

for command in awk caffeinate cmp curl date env find git head jq lsof mkdir mv \
    perl pgrep pmset ps rg rm sed shasum sleep sort stat system_profiler tail \
    tr wc; do
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
    echo "rectangular ABBA requires a clean exact source tree" >&2
    exit 2
}
source_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || {
    echo "could not resolve an exact source commit" >&2
    exit 2
}
[[ "$HF2Q_BIN" == "$SOURCE_ROOT/target/release/hf2q" \
    && -x "$HF2Q_BIN" ]] || {
    echo "HF2Q_BIN must be the exact source tree's executable release output" >&2
    exit 2
}
grep -aFq "$source_commit" "$HF2Q_BIN" || {
    echo "release binary does not embed source commit $source_commit" >&2
    exit 2
}
binary_sha256=$(shasum -a 256 "$HF2Q_BIN" | awk '{print $1}')
[[ "$MODEL_PATH" != "$SOURCE_ROOT"/* && -f "$MODEL_PATH" \
    && -r "$MODEL_PATH" && ! -L "$MODEL_PATH" ]] || {
    echo "MODEL_PATH must be a readable external regular non-symlink" >&2
    exit 2
}
[[ "$MODEL_SHA256" =~ ^[0-9a-f]{64}$ \
    && "$MODEL_BYTES" =~ ^[1-9][0-9]*$ \
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
    *)
        echo "MODEL_SHAPE must be qwen38-dense or qwen36-moe" >&2
        exit 2
        ;;
esac
for value_name in PORT KV_CACHE_BUDGET_BYTES READY_TIMEOUT_SECONDS \
    REQUEST_TIMEOUT_SECONDS THERMAL_SETTLE_TIMEOUT_SECONDS; do
    value=${!value_name}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "$value_name must be a positive integer" >&2
        exit 2
    }
done
((PORT <= 65535)) || { echo "PORT exceeds 65535" >&2; exit 2; }
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
    echo "rectangular ABBA requires no pre-existing hf2q runtime" >&2
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
    echo "rectangular ABBA requires AC power" >&2
    exit 2
}
power_mode=$(resolve_ac_energy_mode) || {
    echo "could not resolve the active AC Energy Mode" >&2
    exit 2
}
[[ "$power_mode" != low ]] || {
    echo "rectangular ABBA rejects Low Power Mode" >&2
    exit 2
}
power_mode_code=$(resolve_live_power_mode_code) || {
    echo "could not resolve the live AC power-mode canary" >&2
    exit 2
}

record_power_contract() {
    local output=$1 phase=$2 observed_source observed_mode observed_code sampled_at
    observed_source=$(resolve_live_power_source) || {
        echo "$phase could not resolve live power source after $POWER_PROBE_ATTEMPTS attempts" >&2
        return 1
    }
    [[ "$observed_source" == ac ]] || {
        echo "$phase observed non-AC power: $observed_source" >&2
        return 1
    }
    observed_mode=$(resolve_ac_energy_mode) || {
        echo "$phase could not resolve AC Energy Mode after $POWER_PROBE_ATTEMPTS attempts" >&2
        return 1
    }
    observed_code=$(resolve_live_power_mode_code) || {
        echo "$phase could not resolve live power-mode code after $POWER_PROBE_ATTEMPTS attempts" >&2
        return 1
    }
    [[ "$observed_mode" == "$power_mode" && "$observed_code" == "$power_mode_code" ]] || {
        echo "$phase observed power-mode drift: expected=$power_mode/$power_mode_code actual=$observed_mode/$observed_code" >&2
        return 1
    }
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
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -TERM "$server_pid" 2>/dev/null || true
        fi
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
    if ((original_rc == 0 && cleanup_rc != 0)); then
        exit "$cleanup_rc"
    fi
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
            || stat -c '%d:%i:%s:%Y:%Z' "$MODEL_PATH")" == "$model_snapshot" ]] || {
        echo "source, binary, or model identity changed during ABBA" >&2
        return 1
    }
}

median_file() {
    local file=$1 count
    count=$(wc -l <"$file" | tr -d ' ')
    sort -n "$file" | awk -v count="$count" '
      NR == int((count + 1) / 2) { left=$1 }
      NR == int((count + 2) / 2) { right=$1 }
      END { if (count % 2) print left; else print (left + right) / 2 }
    '
}

metric_value() {
    local file=$1 name=$2
    awk -v name="$name" '$1 == name { value=$2; found++ }
      END { if (found != 1 || value !~ /^[0-9]+([.][0-9]+)?$/) exit 1; print value }' "$file"
}

build_request() {
    local model_id=$1 sample=$2 lane=$3 output=$4 context
    context=$(awk -v sample="$sample" -v lane="$lane" 'BEGIN {
      printf "sample-%s lane-%s. ", sample, lane
      for (i = 1; i <= 64; i++) printf "cache "
      printf "Return exactly OK."
    }')
    jq -n --arg model "$model_id" --arg content "$context" \
        --argjson max_tokens "$MAX_TOKENS" '{
      model:$model,messages:[{role:"user",content:$content}],
      temperature:0,seed:42,max_tokens:$max_tokens,repetition_penalty:1,
      stream:false,hf2q_enable_thinking:false,
      chat_template_kwargs:{enable_thinking:false}
    }' >"$output"
}

post_request() {
    local request=$1 response=$2 wall=$3
    curl --fail-with-body --silent --show-error --connect-timeout 5 \
        --max-time "$REQUEST_TIMEOUT_SECONDS" -H 'Content-Type: application/json' \
        --data-binary "@$request" -o "$response" -w '%{time_total}\n' \
        "http://$HOST:$PORT/v1/chat/completions" >"$wall"
    jq -e '
      (.choices | length) == 1
      and (.choices[0].message.content | type) == "string"
      and (.usage.prompt_tokens | numbers) > 0
      and (.usage.prompt_tokens_details.cached_tokens == 0)
      and (.usage.completion_tokens | numbers) > 0
      and (.x_hf2q_timing.time_to_first_token_ms | numbers) > 0
    ' "$response" >/dev/null
}

post_request_at_barrier() {
    local request=$1 response=$2 wall=$3 timing=$4 barrier=$5 started finished
    while [[ ! -e "$barrier" ]]; do sleep 0.001; done
    started=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
    post_request "$request" "$response" "$wall"
    finished=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
    printf '%s\t%s\n' "$started" "$finished" >"$timing"
}

run_measurements() {
    local label=$1 arm=$2 model_id=$3 engine_dir=$4 trial lane pid replica
    local before after before_value after_value delta started finished log_start log_end
    local barrier launch_skew earliest_finish latest_start timing_files
    local single_prompt_tokens wave_prompt_tokens publication_line
    replica=${label##*-}
    for ((trial = 1; trial <= TRIALS; trial++)); do
        request="$engine_dir/requests/single-$trial.json"
        response="$engine_dir/responses/single-$trial.json"
        wall="$engine_dir/responses/single-$trial.wall"
        build_request "$model_id" "replica-$replica-single-$trial" 0 "$request"
        post_request "$request" "$response" "$wall"
        jq -er '.x_hf2q_timing.time_to_first_token_ms' "$response" \
            >>"$engine_dir/single-ttft-ms"
        single_prompt_tokens=$(jq -er '.usage.prompt_tokens' "$response")
        awk '{printf "%.6f\n", $1 * 1000}' "$wall" \
            >>"$engine_dir/single-wall-ms"

        before="$engine_dir/waves/$trial.metrics-before"
        after="$engine_dir/waves/$trial.metrics-after"
        curl --fail --silent --show-error "http://$HOST:$PORT/metrics" >"$before"
        before_value=$(metric_value "$before" hf2q_qwen_rectangular_prefill_cohorts_total)
        log_start=$(stat -f '%z' "$engine_dir/server.stderr" 2>/dev/null \
            || stat -c '%s' "$engine_dir/server.stderr")
        started=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
        wave_pids=()
        barrier="$engine_dir/waves/$trial.start"
        rm -f "$barrier"
        for ((lane = 1; lane <= MAX_SLOTS; lane++)); do
            request="$engine_dir/requests/wave-$trial-$lane.json"
            response="$engine_dir/responses/wave-$trial-$lane.json"
            wall="$engine_dir/responses/wave-$trial-$lane.wall"
            timing="$engine_dir/responses/wave-$trial-$lane.timing"
            build_request "$model_id" "replica-$replica-wave-$trial" "$lane" "$request"
            post_request_at_barrier "$request" "$response" "$wall" "$timing" "$barrier" &
            wave_pids+=("$!")
        done
        : >"$barrier"
        for pid in "${wave_pids[@]}"; do wait "$pid"; done
        for ((lane = 1; lane <= MAX_SLOTS; lane++)); do
            wave_prompt_tokens=$(jq -er '.usage.prompt_tokens' \
                "$engine_dir/responses/wave-$trial-$lane.json")
            [[ "$wave_prompt_tokens" == "$single_prompt_tokens" ]] || {
                echo "$label trial $trial single request was not the width-four cohort's eligible prompt shape" >&2
                return 1
            }
        done
        finished=$(perl -MTime::HiRes=time -e 'printf "%.9f\n", time')
        awk -v start="$started" -v finish="$finished" 'BEGIN {print finish-start}' \
            >>"$engine_dir/wave-wall-seconds"
        timing_files=("$engine_dir"/responses/wave-"$trial"-*.timing)
        launch_skew=$(awk -F '\t' '
          NR == 1 {minimum=$1; maximum=$1}
          $1 < minimum {minimum=$1} $1 > maximum {maximum=$1}
          END {printf "%.9f", maximum-minimum}
        ' "${timing_files[@]}")
        latest_start=$(awk -F '\t' 'NR == 1 || $1 > value {value=$1} END {print value}' \
            "${timing_files[@]}")
        earliest_finish=$(awk -F '\t' 'NR == 1 || $2 < value {value=$2} END {print value}' \
            "${timing_files[@]}")
        awk -v skew="$launch_skew" -v latest="$latest_start" \
            -v earliest="$earliest_finish" \
            'BEGIN {exit !(skew <= 0.100 && latest < earliest)}' || {
            echo "$label trial $trial lacked a <=100ms overlapping launch" >&2
            return 1
        }
        curl --fail --silent --show-error "http://$HOST:$PORT/metrics" >"$after"
        after_value=$(metric_value "$after" hf2q_qwen_rectangular_prefill_cohorts_total)
        delta=$(awk -v before="$before_value" -v after="$after_value" \
            'BEGIN {printf "%.0f", after-before}')
        if [[ "$arm" == on ]]; then
            [[ "$delta" == 1 ]] || {
                echo "$label trial $trial published $delta rectangular cohorts, expected 1" >&2
                return 1
            }
        else
            [[ "$delta" == 0 ]] || {
                echo "$label trial $trial published a cohort with policy OFF" >&2
                return 1
            }
        fi
        log_end=$(stat -f '%z' "$engine_dir/server.stderr" 2>/dev/null \
            || stat -c '%s' "$engine_dir/server.stderr")
        tail -c "+$((log_start + 1))" "$engine_dir/server.stderr" \
            | head -c "$((log_end - log_start))" >"$engine_dir/waves/$trial.log"
        publication_line=$(rg 'Qwen rectangular stable-boundary prefill published' \
            "$engine_dir/waves/$trial.log" || true)
        published=$(printf '%s\n' "$publication_line" | awk 'NF {count++} END {print count+0}')
        published=${published:-0}
        if [[ "$arm" == on ]]; then
            [[ "$published" == 1 ]] || {
                echo "$label trial $trial lacks one rectangular publication log" >&2
                return 1
            }
            perl -ne '
              if (/Qwen rectangular stable-boundary prefill published/ && /lanes=4/) {
                $rows = $1 if /rows_per_lane=([0-9]+)/;
                $aggregate = $1 if /aggregate_rows=([0-9]+)/;
                $found++;
                $good++ if defined($rows) && defined($aggregate)
                  && $rows >= 16 && $rows <= 128 && $aggregate == 4 * $rows;
              }
              END { exit(($found == 1 && $good == 1) ? 0 : 1) }
            ' <<<"$publication_line" || {
                echo "$label trial $trial published an invalid lane/row shape" >&2
                return 1
            }
            if [[ "$expected_mtp" == succeeded ]]; then
                [[ "$publication_line" == *"mtp_prefill=true mtp_outcome=Succeeded"* ]] || {
                    echo "$label trial $trial did not complete Qwen3.8 MTP catch-up" >&2
                    return 1
                }
            else
                [[ "$publication_line" == *"mtp_prefill=false mtp_outcome=NotRequested"* ]] || {
                    echo "$label trial $trial unexpectedly requested Qwen3.6 MTP" >&2
                    return 1
                }
            fi
        else
            [[ "$published" == 0 ]] || return 1
        fi
        jq -n --argjson launch_skew_seconds "$launch_skew" \
            --argjson earliest_finish "$earliest_finish" \
            --argjson latest_start "$latest_start" \
            --argjson cohort_metric_delta "$delta" \
            --argjson prompt_tokens "$single_prompt_tokens" '{
              launch_skew_seconds:$launch_skew_seconds,
              latest_start:$latest_start,
              earliest_finish:$earliest_finish,
              actual_overlap:($latest_start < $earliest_finish),
              cohort_metric_delta:$cohort_metric_delta,
              prompt_tokens:$prompt_tokens
            }' >"$engine_dir/waves/$trial.json"
        ps -p "$server_pid" -o rss= | awk 'NF == 1 && $1 ~ /^[0-9]+$/ {print $1}' \
            >>"$engine_dir/rss-kib"
    done
}

run_warmup_waves() {
    local label=$1 model_id=$2 engine_dir=$3 replica warmup lane pid barrier
    replica=${label##*-}
    for warmup in 1 2; do
        barrier="$engine_dir/waves/warmup-$warmup.start"
        rm -f "$barrier"
        wave_pids=()
        for ((lane = 1; lane <= MAX_SLOTS; lane++)); do
            request="$engine_dir/requests/warmup-$warmup-$lane.json"
            response="$engine_dir/responses/warmup-$warmup-$lane.json"
            wall="$engine_dir/responses/warmup-$warmup-$lane.wall"
            timing="$engine_dir/responses/warmup-$warmup-$lane.timing"
            build_request "$model_id" "replica-$replica-warmup-$warmup" "$lane" "$request"
            post_request_at_barrier "$request" "$response" "$wall" "$timing" "$barrier" &
            wave_pids+=("$!")
        done
        : >"$barrier"
        for pid in "${wave_pids[@]}"; do wait "$pid"; done
    done
}

sample_rss_while_pid() {
    local owner_pid=$1 producer_pid=$2 output=$3 state
    while :; do
        state=$(ps -p "$producer_pid" -o state= 2>/dev/null | tr -d '[:space:]' || true)
        [[ -n "$state" && "$state" != Z* ]] || break
        ps -p "$owner_pid" -o rss= \
            | awk 'NF == 1 && $1 ~ /^[0-9]+$/ {print $1}' >>"$output"
        sleep 0.1
    done
    ps -p "$owner_pid" -o rss= \
        | awk 'NF == 1 && $1 ~ /^[0-9]+$/ {print $1}' >>"$output" || true
}

run_process() {
    local label=$1 arm=$2 model_id producer_pid actual_arch expected_bool
    local engine_dir="$OUT_DIR/$label"
    local producer_status=0 monitor_status=0 evidence_manifest_sha rss_monitor_pid
    mkdir -p "$engine_dir/requests" "$engine_dir/responses" \
        "$engine_dir/waves" "$engine_dir/runtime-cache"
    : >"$engine_dir/single-ttft-ms"
    : >"$engine_dir/single-wall-ms"
    : >"$engine_dir/wave-wall-seconds"
    : >"$engine_dir/rss-kib"
    : >"$engine_dir/power.tsv"
    record_power_contract "$engine_dir/power.tsv" "$label-before-launch"
    cross_slot=0
    coalesce=0
    expected_bool=false
    if [[ "$arm" == on ]]; then
        cross_slot=1
        coalesce=$COALESCE_US
        expected_bool=true
    fi
    env -i HOME="$RUNTIME_HOME" PATH="$RUNTIME_PATH" TMPDIR="$RUNTIME_TMPDIR" \
        LANG=C LC_ALL=C USER=hf2q-gate LOGNAME=hf2q-gate RUST_BACKTRACE=1 \
        HF2Q_CROSS_SLOT_ADMIT="$cross_slot" \
        HF2Q_ADMIT_COALESCE_US="$coalesce" \
        HF2Q_QWEN_SPECULATION=auto HF2Q_TQ_KV=1 HF2Q_ENCODER_SESSION=1 \
        HF2Q_FFN_TERMINAL_K_BATCH=8 \
        "$HF2Q_BIN" -v serve --model "$MODEL_PATH" \
        --cache-dir "$engine_dir/runtime-cache" \
        --host "$HOST" --port "$PORT" --scheduler inflight-batched \
        --max-slots "$MAX_SLOTS" --overflow-policy reject \
        --kv-cache-budget "$KV_CACHE_BUDGET_BYTES" \
        --default-repetition-penalty 1 \
        --default-thinking-token-budget 0 --default-tool-thinking-token-budget 0 \
        >"$engine_dir/server.stdout" 2>"$engine_dir/server.stderr" &
    server_pid=$!
    for ((ready_wait = 0; ready_wait < READY_TIMEOUT_SECONDS; ready_wait++)); do
        curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null 2>&1 && break
        kill -0 "$server_pid" 2>/dev/null || {
            tail -n 100 "$engine_dir/server.stderr" >&2
            return 1
        }
        sleep 1
    done
    curl --fail --silent "http://$HOST:$PORT/readyz" >/dev/null || {
        echo "$label server did not become ready" >&2
        return 1
    }
    curl --fail --silent "http://$HOST:$PORT/v1/models" >"$engine_dir/models.json"
    model_id=$(jq -er '[.data[] | select(.loaded == true)] | if length == 1 then .[0].id else error("expected one loaded model") end' \
        "$engine_dir/models.json")
    actual_arch=$(jq -er --arg id "$model_id" \
        '[.data[] | select(.loaded == true and .id == $id)] | if length == 1 then .[0].arch else error("expected one loaded model architecture") end' \
        "$engine_dir/models.json")
    [[ "$actual_arch" == "$expected_arch" ]] || {
        echo "$label loaded the wrong architecture shape" >&2
        return 1
    }
    EXPECTED_ADMIT="$expected_bool" EXPECTED_COALESCE="$coalesce" \
    EXPECTED_MTP="$expected_mtp_bool" \
    perl -ne '
      if (/Qwen35 SlotAware prefill transaction ceiling selected/) {
        $seen++;
        $admit = $1 if /cross_slot_admit=(true|false)/;
        $coalesce = $1 if /cross_slot_coalesce_us=([0-9]+)/;
        $policy = $1 if /speculation_policy=(Auto|Off)/;
        $mtp = $1 if /mtp_capable=(true|false)/;
      }
      END {
        exit 1 unless $seen == 1
          && $admit eq $ENV{EXPECTED_ADMIT}
          && $coalesce == $ENV{EXPECTED_COALESCE}
          && $policy eq "Auto"
          && $mtp eq $ENV{EXPECTED_MTP};
      }
    ' "$engine_dir/server.stderr" || {
        echo "$label did not prove its exact immutable worker policy" >&2
        return 1
    }
    perl -ne '
      if (/resolved serving plan/) {
        $seen++;
        $persist = $1 if /kv_persist_enabled=(true|false)/;
        $budget = $1 if /kv_persist_budget_bytes=([0-9]+)/;
        $cache = $1 if /kv_cache_budget_bytes=([0-9]+)/;
      }
      END { exit 1 unless $seen == 1 && $persist eq "false" && $budget == 0
        && $cache == 51539607552 }
    ' "$engine_dir/server.stderr" || {
        echo "$label did not prove persistence-free serving" >&2
        return 1
    }
    qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
        "$HF2Q_BIN" "$MODEL_PATH" "$MAX_SLOTS"
    ps -ww -p "$server_pid" -o command= >"$engine_dir/server-command.txt"
    assert_identity

    ps -p "$server_pid" -o rss= | awk 'NF == 1 && $1 ~ /^[0-9]+$/ {print $1}' \
        >>"$engine_dir/rss-kib"
    run_warmup_waves "$label" "$model_id" "$engine_dir"
    record_power_contract "$engine_dir/power.tsv" "$label-loaded-warm"
    thermal_wait_for_nominal "$engine_dir/thermal-settle.tsv" "$label-settle" \
        "$THERMAL_SETTLE_SECONDS" "$THERMAL_SETTLE_TIMEOUT_SECONDS" \
        "$THERMAL_SAMPLE_SECONDS" "$engine_dir/contention-settle.tsv" \
        "$HOST_CONTENTION_GATE_OWNER_PID" "$server_pid"
    record_power_contract "$engine_dir/power.tsv" "$label-measurement-start"
    thermal_sample "$engine_dir/thermal-measurement.tsv" "$label-measurement-start"
    host_contention_sample "$engine_dir/contention-measurement.tsv" \
        "$label-measurement-start" "$HOST_CONTENTION_GATE_OWNER_PID" \
        "$THERMAL_SAMPLED_AT" "$server_pid"
    host_contention_require_quiet "$label-measurement-start"
    (
        run_measurements "$label" "$arm" "$model_id" "$engine_dir"
    ) &
    producer_pid=$!
    sample_rss_while_pid "$server_pid" "$producer_pid" "$engine_dir/rss-kib" &
    rss_monitor_pid=$!
    thermal_monitor_fair_or_better_while_pid \
        "$engine_dir/thermal-measurement.tsv" "$label-measurement" \
        "$producer_pid" "$THERMAL_SAMPLE_SECONDS" \
        "$engine_dir/contention-measurement.tsv" \
        "$HOST_CONTENTION_GATE_OWNER_PID" "$server_pid" \
        || monitor_status=$?
    wait "$producer_pid" || producer_status=$?
    wait "$rss_monitor_pid"
    ((producer_status == 0 && monitor_status == 0)) || return 1
    thermal_sample "$engine_dir/thermal-measurement.tsv" "$label-measurement-end"
    host_contention_sample "$engine_dir/contention-measurement.tsv" \
        "$label-measurement-end" "$HOST_CONTENTION_GATE_OWNER_PID" \
        "$THERMAL_SAMPLED_AT" "$server_pid"
    host_contention_require_quiet "$label-measurement-end"
    record_power_contract "$engine_dir/power.tsv" "$label-measurement-end"
    thermal_validate_settle_log "$engine_dir/thermal-settle.tsv" \
        "$THERMAL_SETTLE_SECONDS" "$((THERMAL_SAMPLE_SECONDS + 3))"
    host_contention_validate_settle_log "$engine_dir/contention-settle.tsv" \
        "$THERMAL_SETTLE_SECONDS" "$((THERMAL_SAMPLE_SECONDS + 3))"
    thermal_validate_fair_or_better_measurement_log \
        "$engine_dir/thermal-measurement.tsv" "$((THERMAL_SAMPLE_SECONDS + 3))"
    host_contention_validate_measurement_log \
        "$engine_dir/contention-measurement.tsv" "$((THERMAL_SAMPLE_SECONDS + 3))"
    host_contention_validate_thermal_alignment \
        "$engine_dir/thermal-measurement.tsv" "$engine_dir/contention-measurement.tsv"
    qwen36_bind_server_process "http://$HOST:$PORT" "$server_pid" \
        "$HF2Q_BIN" "$MODEL_PATH" "$MAX_SLOTS"
    qwen36_reject_fatal_log "$engine_dir/server.stderr"
    assert_identity
    kill -0 "$server_pid" 2>/dev/null || return 1
    rg -q 'panicked|fatal runtime error|Metal command buffer failed' \
        "$engine_dir/server.stderr" && {
        echo "$label server log contains a fatal marker" >&2
        return 1
    }
    cleanup
    [[ -z "$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)" ]] || {
        echo "$label listener survived cleanup" >&2
        return 1
    }
    qwen36_reject_fatal_log "$engine_dir/server.stderr"
    record_power_contract "$engine_dir/power.tsv" "$label-after-shutdown"
    assert_identity
    (
        cd "$engine_dir"
        {
            find requests responses waves runtime-cache -type f -print
            printf '%s\n' server.stdout server.stderr thermal-settle.tsv \
                thermal-measurement.tsv contention-settle.tsv \
                contention-measurement.tsv rss-kib single-ttft-ms \
                single-wall-ms wave-wall-seconds power.tsv models.json \
                server-command.txt
        } | sort | while IFS= read -r relative; do
            printf '%s  %s\n' "$(shasum -a 256 "$relative" | awk '{print $1}')" \
                "$relative"
        done >evidence.sha256.tmp
        mv evidence.sha256.tmp evidence.sha256
        shasum -a 256 -c evidence.sha256 >/dev/null
    )
    jq -n --arg label "$label" --arg arm "$arm" --arg model_id "$model_id" \
        --argjson single_median_ttft_ms "$(median_file "$engine_dir/single-ttft-ms")" \
        --argjson single_median_wall_ms "$(median_file "$engine_dir/single-wall-ms")" \
        --argjson wave_median_seconds "$(median_file "$engine_dir/wave-wall-seconds")" \
        --argjson sampled_peak_rss_kib "$(sort -n "$engine_dir/rss-kib" | tail -1)" \
        --argjson single_samples "$(jq -Rsc 'split("\n") | map(select(length>0)|tonumber)' "$engine_dir/single-ttft-ms")" \
        --argjson single_wall_samples "$(jq -Rsc 'split("\n") | map(select(length>0)|tonumber)' "$engine_dir/single-wall-ms")" \
        --argjson wave_samples "$(jq -Rsc 'split("\n") | map(select(length>0)|tonumber)' "$engine_dir/wave-wall-seconds")" \
        --arg thermal_settle_sha "$(shasum -a 256 "$engine_dir/thermal-settle.tsv" | awk '{print $1}')" \
        --arg thermal_measurement_sha "$(shasum -a 256 "$engine_dir/thermal-measurement.tsv" | awk '{print $1}')" \
        --arg contention_settle_sha "$(shasum -a 256 "$engine_dir/contention-settle.tsv" | awk '{print $1}')" \
        --arg contention_measurement_sha "$(shasum -a 256 "$engine_dir/contention-measurement.tsv" | awk '{print $1}')" '{
          label:$label,arm:$arm,model_id:$model_id,
          single_median_ttft_ms:$single_median_ttft_ms,
          single_median_wall_ms:$single_median_wall_ms,
          wave_median_seconds:$wave_median_seconds,
          sampled_peak_rss_kib:$sampled_peak_rss_kib,
          single_engine_ttft_samples_ms:$single_samples,
          single_wall_samples_ms:$single_wall_samples,
          wave_samples_seconds:$wave_samples,
          runtime:{clean_environment:true,home:"/var/empty",
            path:"/usr/bin:/bin:/usr/sbin:/sbin",tmpdir:"/var/tmp",
            locale:{LANG:"C",LC_ALL:"C"},rust_backtrace:"1",
            hf2q:{tq_kv:"1",encoder_session:"1",ffn_terminal_k_batch:"8",
              speculation:"auto"},serve:{kv_persist:false,
                kv_cache_budget_bytes:51539607552,cache_dir:"evidence-local"}},
          environment:{thermal_settle_sha256:$thermal_settle_sha,
            thermal_measurement_sha256:$thermal_measurement_sha,
            contention_settle_sha256:$contention_settle_sha,
            contention_measurement_sha256:$contention_measurement_sha,
            power_sha256:"__POWER_SHA__"},
          evidence_manifest_sha256:"__EVIDENCE_MANIFEST_SHA__"
        }' >"$engine_dir/summary.json"
    evidence_manifest_sha=$(shasum -a 256 "$engine_dir/evidence.sha256" | awk '{print $1}')
    jq --arg sha "$evidence_manifest_sha" \
        --arg power_sha "$(shasum -a 256 "$engine_dir/power.tsv" | awk '{print $1}')" \
        '.evidence_manifest_sha256 = $sha | .environment.power_sha256 = $power_sha' \
        "$engine_dir/summary.json" \
        >"$engine_dir/summary.json.tmp"
    mv "$engine_dir/summary.json.tmp" "$engine_dir/summary.json"
}

thermal_prepare_probe
qwen36_start_power_guard "$$" "$OUT_DIR/caffeinate.log"
caffeinate_started=true
run_process off-a off
run_process on-a on
run_process on-b on
run_process off-b off

off_summary="$OUT_DIR/off-summary.json"
on_summary="$OUT_DIR/on-summary.json"
jq -s '{arm:"off",
  single_wall_samples_ms:(map(.single_wall_samples_ms)|add),
  wave_samples_seconds:(map(.wave_samples_seconds)|add)}
  | def median: sort as $s | ($s|length) as $n
      | if ($n % 2) == 1 then $s[($n/2)|floor]
        else (($s[$n/2-1] + $s[$n/2]) / 2) end;
    .single_median_wall_ms=(.single_wall_samples_ms|median)
  | .wave_median_seconds=(.wave_samples_seconds|median)' \
    "$OUT_DIR/off-a/summary.json" "$OUT_DIR/off-b/summary.json" >"$off_summary"
jq -s '{arm:"on",
  single_wall_samples_ms:(map(.single_wall_samples_ms)|add),
  wave_samples_seconds:(map(.wave_samples_seconds)|add)}
  | def median: sort as $s | ($s|length) as $n
      | if ($n % 2) == 1 then $s[($n/2)|floor]
        else (($s[$n/2-1] + $s[$n/2]) / 2) end;
    .single_median_wall_ms=(.single_wall_samples_ms|median)
  | .wave_median_seconds=(.wave_samples_seconds|median)' \
    "$OUT_DIR/on-a/summary.json" "$OUT_DIR/on-b/summary.json" >"$on_summary"

for replica in a b; do
    while IFS= read -r relative; do
        cmp -s "$OUT_DIR/off-$replica/$relative" "$OUT_DIR/on-$replica/$relative" || {
            echo "OFF/ON request bytes differ for replica $replica: $relative" >&2
            exit 1
        }
    done < <(cd "$OUT_DIR/off-$replica" && find requests -type f -print | sort)
done

single_overhead_samples="$OUT_DIR/single-overhead-ms"
: >"$single_overhead_samples"
for replica in a b; do
    for ((trial = 1; trial <= TRIALS; trial++)); do
        off_wall=$(tr -d '[:space:]' \
            <"$OUT_DIR/off-$replica/responses/single-$trial.wall")
        on_wall=$(tr -d '[:space:]' \
            <"$OUT_DIR/on-$replica/responses/single-$trial.wall")
        awk -v off="$off_wall" -v on="$on_wall" \
            'BEGIN {printf "%.6f\n", (on-off) * 1000}' \
            >>"$single_overhead_samples"
    done
done

off_semantic_sha=$(jq -Ssc 'map({
  message:(.choices[0].message | {role,content,reasoning_content,tool_calls,refusal}),
  finish_reason:.choices[0].finish_reason,
  usage:(.usage | {prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details})})' \
  "$OUT_DIR"/off-*/responses/{single,wave}-*.json \
  | shasum -a 256 | awk '{print $1}')
on_semantic_sha=$(jq -Ssc 'map({
  message:(.choices[0].message | {role,content,reasoning_content,tool_calls,refusal}),
  finish_reason:.choices[0].finish_reason,
  usage:(.usage | {prompt_tokens,completion_tokens,total_tokens,prompt_tokens_details})})' \
  "$OUT_DIR"/on-*/responses/{single,wave}-*.json \
  | shasum -a 256 | awk '{print $1}')
[[ "$off_semantic_sha" == "$on_semantic_sha" ]] || {
    echo "OFF/ON semantic or token receipts differ" >&2
    exit 1
}
wave_speedup=$(jq -nr --slurpfile off "$off_summary" --slurpfile on "$on_summary" \
    '$off[0].wave_median_seconds / $on[0].wave_median_seconds')
single_median_overhead_ms=$(jq -nr --slurpfile off "$off_summary" --slurpfile on "$on_summary" \
    '$on[0].single_median_wall_ms - $off[0].single_median_wall_ms')
single_max_matched_overhead_ms=$(sort -n "$single_overhead_samples" | tail -1)
neighbor_a_speedup=$(jq -nr --slurpfile off "$OUT_DIR/off-a/summary.json" \
    --slurpfile on "$OUT_DIR/on-a/summary.json" \
    '$off[0].wave_median_seconds / $on[0].wave_median_seconds')
neighbor_b_speedup=$(jq -nr --slurpfile off "$OUT_DIR/off-b/summary.json" \
    --slurpfile on "$OUT_DIR/on-b/summary.json" \
    '$off[0].wave_median_seconds / $on[0].wave_median_seconds')
awk -v actual="$wave_speedup" -v minimum="$MIN_WAVE_SPEEDUP" \
    'BEGIN {exit !(actual >= minimum)}' || {
    echo "rectangular wave speedup $wave_speedup is below $MIN_WAVE_SPEEDUP" >&2
    exit 1
}
for neighbor in "$neighbor_a_speedup" "$neighbor_b_speedup"; do
    awk -v actual="$neighbor" 'BEGIN {exit !(actual > 1.0)}' || {
        echo "an ON process was not faster than its neighboring OFF process: $neighbor" >&2
        exit 1
    }
done
awk -v actual="$single_max_matched_overhead_ms" -v maximum="$MAX_SINGLE_OVERHEAD_MS" \
    'BEGIN {exit !(actual <= maximum)}' || {
    echo "worst matched single-user coalescing overhead ${single_max_matched_overhead_ms}ms exceeds ${MAX_SINGLE_OVERHEAD_MS}ms" >&2
    exit 1
}
assert_identity
qwen36_assert_power_guard
qwen36_stop_power_guard
caffeinate_started=false

jq -n --arg source_commit "$source_commit" --arg binary "$HF2Q_BIN" \
    --arg binary_sha256 "$binary_sha256" --arg model_shape "$MODEL_SHAPE" \
    --arg model_path "$MODEL_PATH" --arg model_sha256 "$MODEL_SHA256" \
    --arg model_snapshot "$model_snapshot" --argjson model_bytes "$MODEL_BYTES" \
    --arg semantic_sha256 "$on_semantic_sha" --arg power_mode "$power_mode" \
    --arg host_contention_policy "$HOST_CONTENTION_POLICY" \
    --argjson host_contention_max_foreign_cpu_percent \
        "$HOST_CONTENTION_MAX_FOREIGN_CPU_PERCENT" \
    --argjson host_contention_owner_pgid "$HOST_CONTENTION_OWNER_PGID" \
    --argjson trials_per_process "$TRIALS" --argjson max_tokens "$MAX_TOKENS" \
    --argjson coalesce_us "$COALESCE_US" \
    --argjson min_wave_speedup "$MIN_WAVE_SPEEDUP" \
    --argjson max_single_overhead_ms "$MAX_SINGLE_OVERHEAD_MS" \
    --argjson wave_speedup "$wave_speedup" \
    --argjson neighbor_a_speedup "$neighbor_a_speedup" \
    --argjson neighbor_b_speedup "$neighbor_b_speedup" \
    --argjson single_median_overhead_ms "$single_median_overhead_ms" \
    --argjson single_max_matched_overhead_ms "$single_max_matched_overhead_ms" \
    --argjson single_overhead_samples_ms "$(jq -Rsc 'split("\n") | map(select(length>0)|tonumber)' "$single_overhead_samples")" \
    --arg off_a_summary_sha "$(shasum -a 256 "$OUT_DIR/off-a/summary.json" | awk '{print $1}')" \
    --arg on_a_summary_sha "$(shasum -a 256 "$OUT_DIR/on-a/summary.json" | awk '{print $1}')" \
    --arg on_b_summary_sha "$(shasum -a 256 "$OUT_DIR/on-b/summary.json" | awk '{print $1}')" \
    --arg off_b_summary_sha "$(shasum -a 256 "$OUT_DIR/off-b/summary.json" | awk '{print $1}')" \
    --arg off_a_manifest_sha "$(shasum -a 256 "$OUT_DIR/off-a/evidence.sha256" | awk '{print $1}')" \
    --arg on_a_manifest_sha "$(shasum -a 256 "$OUT_DIR/on-a/evidence.sha256" | awk '{print $1}')" \
    --arg on_b_manifest_sha "$(shasum -a 256 "$OUT_DIR/on-b/evidence.sha256" | awk '{print $1}')" \
    --arg off_b_manifest_sha "$(shasum -a 256 "$OUT_DIR/off-b/evidence.sha256" | awk '{print $1}')" \
    --arg off_aggregate_sha "$(shasum -a 256 "$off_summary" | awk '{print $1}')" \
    --arg on_aggregate_sha "$(shasum -a 256 "$on_summary" | awk '{print $1}')" \
    --arg caffeinate_log_sha "$(shasum -a 256 "$OUT_DIR/caffeinate.log" | awk '{print $1}')" \
    --arg caffeinate_assertions_sha "$(shasum -a 256 "$OUT_DIR/caffeinate.log.assertions" | awk '{print $1}')" \
    --arg power_events_baseline_sha "$(shasum -a 256 "$OUT_DIR/caffeinate.log.power-events.baseline" | awk '{print $1}')" \
    --arg power_events_final_sha "$(shasum -a 256 "$OUT_DIR/caffeinate.log.power-events.final" | awk '{print $1}')" \
    --arg power_events_new_sha "$(shasum -a 256 "$OUT_DIR/caffeinate.log.power-events.new" | awk '{print $1}')" \
    --slurpfile off "$off_summary" --slurpfile on "$on_summary" '{
      schema:1,verdict:"pass",gate:"qwen35-rectangular-policy-abba",
      source:{commit:$source_commit,binary:$binary,sha256:$binary_sha256},
      model:{shape:$model_shape,path:$model_path,sha256:$model_sha256,
        bytes:$model_bytes,snapshot:$model_snapshot},
      workload:{process_order:["off-a","on-a","on-b","off-b"],
        same_binary:true,trials_per_process:$trials_per_process,lanes:4,
        stable_boundary_rows:{minimum:16,maximum:128},max_tokens:$max_tokens,
        temperature:0,seed:42,speculation:"auto",coalesce_us:$coalesce_us,
        kv_cache_budget_bytes:51539607552},
      environment:{power:"ac",power_mode:$power_mode,
        thermal:"nominal-settle-and-fair-or-better-measurement",
        host_contention:{policy:$host_contention_policy,
          maximum_foreign_cpu_percent:$host_contention_max_foreign_cpu_percent,
          owner_scope:"release-gate-process-group",
          owner_pgid:$host_contention_owner_pgid,continuous:true},
        clean_process_environment:true,
        serve_kv_persist:false},
      equality:{semantic_and_token_sha256:$semantic_sha256},
      evidence:{processes:{
        "off-a":{summary_sha256:$off_a_summary_sha,manifest_sha256:$off_a_manifest_sha},
        "on-a":{summary_sha256:$on_a_summary_sha,manifest_sha256:$on_a_manifest_sha},
        "on-b":{summary_sha256:$on_b_summary_sha,manifest_sha256:$on_b_manifest_sha},
        "off-b":{summary_sha256:$off_b_summary_sha,manifest_sha256:$off_b_manifest_sha}},
        aggregates:{off_sha256:$off_aggregate_sha,on_sha256:$on_aggregate_sha},
        power_guard:{caffeinate_log_sha256:$caffeinate_log_sha,
          assertions_sha256:$caffeinate_assertions_sha,
          events_baseline_sha256:$power_events_baseline_sha,
          events_final_sha256:$power_events_final_sha,
          events_new_sha256:$power_events_new_sha}},
      thresholds:{min_wave_speedup:$min_wave_speedup,
        max_single_overhead_ms:$max_single_overhead_ms},
      result:{wave_speedup:$wave_speedup,
        single_median_overhead_ms:$single_median_overhead_ms,
        single_max_matched_overhead_ms:$single_max_matched_overhead_ms,
        single_matched_overhead_samples_ms:$single_overhead_samples_ms,
        neighboring_process_speedups:[$neighbor_a_speedup,$neighbor_b_speedup],
        off:$off[0],on:$on[0]}
    }' >"$receipt_tmp"
mv "$receipt_tmp" "$OUT_DIR/receipt.json"
"$script_dir/verify_qwen35_rectangular_policy_receipt.sh" \
    "$OUT_DIR/receipt.json" "$SOURCE_ROOT"
jq . "$OUT_DIR/receipt.json"
echo "receipt: $OUT_DIR/receipt.json" >&2
