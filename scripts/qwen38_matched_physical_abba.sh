#!/usr/bin/env bash
set -euo pipefail

# One-artifact matched physical-performance authority. Every width runs fresh
# servers in fixed ABBA order. The timed arms use each engine's explicitly
# named shipping speculation policy. The separately sealed physical receipt
# proves ordinary target body/head width and real Metal submissions for the
# identical hf2q binary and artifact.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts"
if [[ ${HF2Q_MATCHED_GATE_ISOLATED:-0} != 1 ]]; then
    exec "$SCRIPT_DIR/run_release_gate_process_group.sh" env \
      HF2Q_MATCHED_GATE_ISOLATED=1 "$0" "$@"
fi
# shellcheck source=scripts/macos_thermal_guard.sh
source "$SCRIPT_DIR/macos_thermal_guard.sh"
readonly HOST_CONTENTION_GATE_OWNER_PID=$$
host_contention_require_isolated_gate_owner \
  "$HOST_CONTENTION_GATE_OWNER_PID"
# shellcheck source=scripts/qwen38_artifact_contract.sh
source "$SCRIPT_DIR/qwen38_artifact_contract.sh"
# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$SCRIPT_DIR/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/macos_runtime_identity.sh
source "$SCRIPT_DIR/macos_runtime_identity.sh"
# shellcheck source=scripts/qwen38_matched_reference_contract.sh
source "$SCRIPT_DIR/qwen38_matched_reference_contract.sh"
# shellcheck source=scripts/qwen38_physical_multislot_contract.sh
source "$SCRIPT_DIR/qwen38_physical_multislot_contract.sh"
# shellcheck source=scripts/qwen38_matched_physical_contract.sh
source "$SCRIPT_DIR/qwen38_matched_physical_contract.sh"

HF2Q_BIN=${HF2Q_BIN:?HF2Q_BIN is required}
HF2Q_SOURCE_DIR=${HF2Q_SOURCE_DIR:?HF2Q_SOURCE_DIR is required}
HF2Q_COMMIT=${HF2Q_COMMIT:?HF2Q_COMMIT is required}
HF2Q_SHA256=${HF2Q_SHA256:?HF2Q_SHA256 is required}
REFERENCE_BIN=${REFERENCE_BIN:?REFERENCE_BIN is required}
REFERENCE_SOURCE_DIR=${REFERENCE_SOURCE_DIR:?REFERENCE_SOURCE_DIR is required}
REFERENCE_COMMIT=${REFERENCE_COMMIT:?REFERENCE_COMMIT is required}
REFERENCE_SHA256=${REFERENCE_SHA256:?REFERENCE_SHA256 is required}
REFERENCE_RUNTIME_MANIFEST_SHA256=${REFERENCE_RUNTIME_MANIFEST_SHA256:?REFERENCE_RUNTIME_MANIFEST_SHA256 is required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
MODEL_FORMAT=${MODEL_FORMAT:?MODEL_FORMAT is required}
MODEL_BYTES=${MODEL_BYTES:?MODEL_BYTES is required}
PHYSICAL_MATRIX_RECEIPT=${PHYSICAL_MATRIX_RECEIPT:?PHYSICAL_MATRIX_RECEIPT is required}
PHYSICAL_MATRIX_SHA256=${PHYSICAL_MATRIX_SHA256:?PHYSICAL_MATRIX_SHA256 is required}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
PORT=${PORT:-18096}
MODEL_ID=${MODEL_ID:-}
MIN_HF2Q_RATIO=${MIN_HF2Q_RATIO:-1.0}
MAX_LAUNCH_SKEW_SECONDS=${MAX_LAUNCH_SKEW_SECONDS:-$QWEN38_MATCHED_MAX_LAUNCH_SKEW_SECONDS}
KV_CACHE_BUDGET_BYTES=${KV_CACHE_BUDGET_BYTES:-$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES}

readonly WIDTHS=(1 2 4 8 16)
readonly TRIAL_ENGINES=(hf2q reference reference hf2q)
readonly MAX_TOKENS=256
readonly THERMAL_SETTLE_SECONDS=120
readonly THERMAL_SETTLE_TIMEOUT_SECONDS=900
readonly THERMAL_SAMPLE_SECONDS=2
readonly MAX_WITHIN_ENGINE_GROUP_SPREAD_PERCENT=5
readonly MAX_WITHIN_ENGINE_CASE_SPREAD_PERCENT=10
readonly HF2Q_THERMAL_SWIFTC_BIN=/usr/bin/swiftc

for command in awk basename caffeinate cmp cp curl date dirname find git jq \
  lsof mkdir mv otool perl pmset ps realpath rg rustc sed shasum sort stat sw_vers \
  system_profiler sysctl uname; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
for executable in "$HF2Q_BIN" "$REFERENCE_BIN" \
  "$HF2Q_SOURCE_DIR/scripts/serve_qwen38_opencode.sh"; do
    [[ -x "$executable" ]] || {
        echo "server binary is missing or non-executable: $executable" >&2
        exit 2
    }
done
export HF2Q_MODEL_VERIFICATION_BINARY="$HF2Q_BIN"
[[ -f "$MODEL_PATH" && -r "$MODEL_PATH" ]] || {
    echo "model is missing or unreadable: $MODEL_PATH" >&2
    exit 2
}
[[ -f "$PHYSICAL_MATRIX_RECEIPT" && -r "$PHYSICAL_MATRIX_RECEIPT" ]] || {
    echo "physical matrix receipt is missing: $PHYSICAL_MATRIX_RECEIPT" >&2
    exit 2
}
for commit in "$HF2Q_COMMIT" "$REFERENCE_COMMIT"; do
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || {
        echo "source commits must be exact lowercase digests" >&2
        exit 2
    }
done
for digest in "$HF2Q_SHA256" "$REFERENCE_SHA256" \
  "$REFERENCE_RUNTIME_MANIFEST_SHA256" "$MODEL_SHA256" \
  "$PHYSICAL_MATRIX_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
        echo "artifact identities must be lowercase SHA-256 digests" >&2
        exit 2
    }
done
for value in "$MODEL_BYTES" "$PORT" "$KV_CACHE_BUDGET_BYTES"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "numeric settings must be non-negative integers" >&2
        exit 2
    }
done
((MODEL_BYTES > 0 && PORT >= 1 && PORT <= 65535 \
  && KV_CACHE_BUDGET_BYTES > 0)) || exit 2
[[ "$MIN_HF2Q_RATIO" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || exit 2
awk -v minimum="$MIN_HF2Q_RATIO" 'BEGIN { exit !(minimum >= 1.0) }'
awk -v maximum="$MAX_LAUNCH_SKEW_SECONDS" \
  'BEGIN { exit !(maximum > 0 && maximum <= 1) }'
[[ "$KV_CACHE_BUDGET_BYTES" == \
  "$QWEN38_CANONICAL_KV_CACHE_BUDGET_BYTES" ]] || {
    echo "matched physical gate requires the canonical KV-cache budget" >&2
    exit 2
}
awk -v actual="$MAX_LAUNCH_SKEW_SECONDS" \
  -v required="$QWEN38_MATCHED_MAX_LAUNCH_SKEW_SECONDS" \
  'BEGIN { exit !(actual == required) }' || {
    echo "matched physical gate requires the canonical launch-skew ceiling" >&2
    exit 2
}

artifact_record=$(qwen38_artifact_record "$MODEL_FORMAT")
IFS=$'\t' read -r _qualified_format qualified_file qualified_bytes \
  qualified_sha qualified_file_type <<<"$artifact_record"
qwen38_validate_artifact_identity "$MODEL_FORMAT" "$MODEL_SHA256" \
  "$MODEL_BYTES" "$qualified_file_type"
[[ "$MODEL_SHA256" == "$qualified_sha" && "$MODEL_BYTES" == "$qualified_bytes" ]]
qwen38_validate_pinned_peer_commit "$REFERENCE_COMMIT"
qwen38_validate_physical_matrix_seal "$PHYSICAL_MATRIX_RECEIPT"
[[ "$(shasum -a 256 "$PHYSICAL_MATRIX_RECEIPT" | awk '{print $1}')" \
  == "$PHYSICAL_MATRIX_SHA256" ]] || {
    echo "physical matrix SHA-256 mismatch" >&2
    exit 2
}
physical_binary_sha=$(jq -er '.results[0].binary.sha256' \
  "$PHYSICAL_MATRIX_RECEIPT")
[[ "$physical_binary_sha" == "$HF2Q_SHA256" ]] || {
    echo "physical matrix and matched hf2q binary differ" >&2
    exit 2
}

[[ "$HF2Q_SOURCE_DIR" == /* && "$REFERENCE_SOURCE_DIR" == /* \
  && "$OUT_DIR" == /* ]] || {
    echo "source directories and OUT_DIR must be absolute" >&2
    exit 2
}
if [[ -e "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "matched physical output directory must be fresh: $OUT_DIR" >&2
    exit 2
fi
case "$OUT_DIR" in
    "$ROOT_DIR"|"$ROOT_DIR"/*|"$HF2Q_SOURCE_DIR"|"$HF2Q_SOURCE_DIR"/*|\
    "$REFERENCE_SOURCE_DIR"|"$REFERENCE_SOURCE_DIR"/*)
        echo "evidence must live outside all source worktrees" >&2
        exit 2
        ;;
esac

sha256_file() {
    shasum -a 256 "$1" | awk '{print $1}'
}

file_snapshot() {
    stat -f '%d:%i:%z:%m:%c' "$1" 2>/dev/null \
      || stat -c '%d:%i:%s:%Y:%Z' "$1" 2>/dev/null
}

file_bytes() {
    stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1" 2>/dev/null
}

verify_executable_identity() {
    local label=$1 executable=$2 expected_sha=$3 expected_snapshot=$4
    [[ -x "$executable" \
      && "$(file_snapshot "$executable")" == "$expected_snapshot" \
      && "$(sha256_file "$executable")" == "$expected_sha" ]] || {
        echo "$label executable changed during the gate" >&2
        return 1
    }
}

harness_commit=$(git -C "$ROOT_DIR" rev-parse HEAD)
matched_physical_require_clean_exact_source "$ROOT_DIR" "$harness_commit" harness
matched_physical_require_clean_exact_source \
  "$HF2Q_SOURCE_DIR" "$HF2Q_COMMIT" hf2q
matched_physical_require_clean_exact_source \
  "$REFERENCE_SOURCE_DIR" "$REFERENCE_COMMIT" reference
[[ "$(sha256_file "$HF2Q_BIN")" == "$HF2Q_SHA256" ]]
grep -aFq "$HF2Q_COMMIT" "$HF2Q_BIN" || {
    echo "hf2q binary does not embed $HF2Q_COMMIT" >&2
    exit 2
}
reference_version=$("$REFERENCE_BIN" --version 2>&1)
[[ "$reference_version" == *"${REFERENCE_COMMIT:0:9}"* ]] || {
    echo "reference binary version is not bound to $REFERENCE_COMMIT" >&2
    exit 2
}
[[ "$(sha256_file "$REFERENCE_BIN")" == "$REFERENCE_SHA256" ]]
[[ "$(file_bytes "$MODEL_PATH")" == "$MODEL_BYTES" ]]
reference_pin_path=$(qwen38_pinned_peer_pin_path)
reference_pin_sha=$(sha256_file "$reference_pin_path")

hf2q_binary_snapshot=$(file_snapshot "$HF2Q_BIN")
reference_binary_snapshot=$(file_snapshot "$REFERENCE_BIN")
reference_runtime_manifest=$(hf2q_macos_runtime_manifest "$REFERENCE_BIN")
reference_runtime_manifest_sha=$(printf '%s\n' "$reference_runtime_manifest" \
  | shasum -a 256 | awk '{print $1}')
[[ "$reference_runtime_manifest_sha" == \
  "$REFERENCE_RUNTIME_MANIFEST_SHA256" ]] || {
    echo "reference runtime closure mismatch: expected=$REFERENCE_RUNTIME_MANIFEST_SHA256 actual=$reference_runtime_manifest_sha" >&2
    exit 2
}
hardware_model=$(sysctl -n hw.model)
hardware_chip=$(sysctl -n machdep.cpu.brand_string)
hardware_memory_bytes=$(sysctl -n hw.memsize)
hardware_arch=$(uname -m)
hardware_os=$(sw_vers -productVersion)
[[ "$hardware_arch" == arm64 && "$hardware_memory_bytes" =~ ^[1-9][0-9]*$ ]]

lock_identity=$(awk '
  $0 == "name = \"mlx-native\"" { found = 1; next }
  found && /^version = / { version = $3; gsub(/\"/, "", version); next }
  found && /^checksum = / { checksum = $3; gsub(/\"/, "", checksum); print version " " checksum; exit }
' "$HF2Q_SOURCE_DIR/Cargo.lock")
IFS=' ' read -r mlx_native_version mlx_native_checksum <<<"$lock_identity"
manifest_mlx_native_version=$(sed -nE \
  's/^mlx-native = "=([0-9]+\.[0-9]+\.[0-9]+)"$/\1/p' \
  "$HF2Q_SOURCE_DIR/Cargo.toml")
[[ "$mlx_native_version" == "$manifest_mlx_native_version" \
  && "$mlx_native_checksum" =~ ^[0-9a-f]{64}$ ]]

mkdir -p "$OUT_DIR/requests/code" "$OUT_DIR/requests/repeat" \
  "$OUT_DIR/requests/warmup" "$OUT_DIR/requests/cache-buster" \
  "$OUT_DIR/expected/repeat" "$OUT_DIR/expected/warmup" "$OUT_DIR/widths"
git -C "$ROOT_DIR" status --porcelain=v2 >"$OUT_DIR/harness-status.txt"
git -C "$HF2Q_SOURCE_DIR" status --porcelain=v2 \
  >"$OUT_DIR/hf2q-status.txt"
git -C "$REFERENCE_SOURCE_DIR" status --porcelain=v2 \
  >"$OUT_DIR/reference-status.txt"
printf '%s\n' "$reference_version" >"$OUT_DIR/reference-version.txt"
printf '%s\n' "$reference_runtime_manifest" \
  >"$OUT_DIR/reference-runtime.sha256"

model_verification_receipt=${HF2Q_MODEL_VERIFICATION_RECEIPT:-}
if [[ -z "$model_verification_receipt" ]]; then
    model_verification_receipt="$OUT_DIR/model-verification.json"
    if [[ -n ${HF2Q_MODEL_VERIFICATION_CACHE_DIR:-} ]]; then
        model_verification_cache_dir=$HF2Q_MODEL_VERIFICATION_CACHE_DIR
    elif [[ -n ${XDG_CACHE_HOME:-} ]]; then
        model_verification_cache_dir="$XDG_CACHE_HOME/hf2q/model-verification"
    else
        model_verification_cache_dir="${HOME:?HOME is required}/.cache/hf2q/model-verification"
    fi
    hf2q_release_prepare_model_verification "$MODEL_PATH" "$MODEL_SHA256" \
      "$model_verification_receipt" "$model_verification_cache_dir"
    model_verification_mode=$(jq -er .run_verification \
      "$model_verification_receipt")
else
    supplied_receipt=$model_verification_receipt
    model_verification_receipt="$OUT_DIR/model-verification.json"
    hf2q_release_materialize_model_verification "$MODEL_PATH" "$MODEL_SHA256" \
      "$supplied_receipt" "$model_verification_receipt"
    model_verification_mode=$(jq -er .run_verification \
      "$model_verification_receipt")
fi
model_file_snapshot=$(jq -er .file_snapshot "$model_verification_receipt")

requests_written=0
request_manifest="$OUT_DIR/requests.sha256"
repeat_text='The copper observatory stood above the harbor while seven quiet instruments recorded wind, tide, temperature, pressure, cloud cover, rainfall, and the slow vibration of the old bridge. Each evening the keeper copied those readings into a blue ledger, checked every column twice, and left the completed page beneath a brass lamp for the morning crew.'
warmup_segment='Amber lanterns marked the northern footpath while careful surveyors measured every stone, copied each coordinate into waterproof notebooks, checked the compass twice, and returned the polished instruments to numbered cedar cases before the evening rain reached the quiet valley.'
warmup_text="$warmup_segment $warmup_segment $warmup_segment $warmup_segment"

write_requests() {
    local index lane task
    ((requests_written == 0)) || return 0
    [[ -n "$MODEL_ID" ]] || return 1
    for ((index = 1; index <= 16; index++)); do
        printf -v lane '\\x%02x' "$((64 + index))"
        printf -v lane '%b' "$lane"
        case $(((index - 1) % 3)) in
            0)
                task='Implement fn fibonacci(n: u64) -> u64 iteratively. Include exactly one unit test containing exactly one assertion.'
                ;;
            1)
                task='Implement fn binary_search(xs: &[i32], needle: i32) -> Option<usize> iteratively for a sorted slice. Include exactly one unit test containing exactly one assertion.'
                ;;
            *)
                task='Implement fn gcd(mut a: u64, mut b: u64) -> u64 with the iterative Euclidean algorithm. Include exactly one unit test containing exactly one assertion.'
                ;;
        esac
        jq -n --arg model "$MODEL_ID" --arg lane "$lane" --arg task "$task" \
          --argjson max_tokens "$MAX_TOKENS" '{
            model:$model,messages:[
              {role:"system",content:("Validation lane " + $lane + ". Return only one complete compilable Rust source file. Do not use Markdown fences or prose.")},
              {role:"user",content:$task}],max_tokens:$max_tokens,temperature:0,
            repetition_penalty:1.05,stream:false,
            chat_template_kwargs:{enable_thinking:false}
          }' >"$OUT_DIR/requests/code/lane-$index.json"
        jq -n --arg model "$MODEL_ID" --arg lane "$lane" \
          --arg text "$repeat_text" --argjson max_tokens "$MAX_TOKENS" '{
            model:$model,messages:[
              {role:"system",content:("Validation lane " + $lane + ". You are a transcription engine.")},
              {role:"user",content:("Repeat the following text exactly, with no introduction or quotation marks:\n\n" + $text)}],
            max_tokens:$max_tokens,temperature:0,repetition_penalty:1.05,
            stream:true,stream_options:{include_usage:true},
            chat_template_kwargs:{enable_thinking:false}
          }' >"$OUT_DIR/requests/repeat/lane-$index.json"
        printf '%s' "$repeat_text" >"$OUT_DIR/expected/repeat/lane-$index.txt"
        jq -n --arg model "$MODEL_ID" --arg lane "$lane" \
          --arg text "$warmup_text" '{
            model:$model,messages:[
              {role:"system",content:("Warmup lane " + $lane + ". Transcribe exactly.")},
              {role:"user",content:("Repeat exactly: " + $text)}],
            max_tokens:256,temperature:0,repetition_penalty:1.05,stream:false,
            chat_template_kwargs:{enable_thinking:false}
          }' >"$OUT_DIR/requests/warmup/lane-$index.json"
        printf '%s' "$warmup_text" >"$OUT_DIR/expected/warmup/lane-$index.txt"
        jq -n --arg model "$MODEL_ID" --arg lane "$lane" '{
          model:$model,messages:[
            {role:"system",content:"This request replaces terminal cache state."},
            {role:"user",content:("Cache replacement lane " + $lane + ". Return exactly READY.")}],
          max_tokens:4,temperature:0,repetition_penalty:1.05,stream:false,
          chat_template_kwargs:{enable_thinking:false}
        }' >"$OUT_DIR/requests/cache-buster/lane-$index.json"
    done
    : >"$request_manifest.tmp"
    while IFS= read -r path; do
        printf '%s  %s\n' "$(sha256_file "$OUT_DIR/$path")" "$path" \
          >>"$request_manifest.tmp"
    done < <(cd "$OUT_DIR" && find requests expected -type f -print | sort)
    mv "$request_manifest.tmp" "$request_manifest"
    requests_written=1
}

verify_request_manifest() {
    [[ -s "$request_manifest" ]] \
      && (cd "$OUT_DIR" && shasum -a 256 -c "$(basename "$request_manifest")" \
        >/dev/null)
}

power_mode_name=''
power_mode_code=''
read_live_power_mode_code() {
    pmset -g live | matched_parse_live_power_mode_code
}
require_ac_power() {
    pmset -g batt | rg -q "Now drawing from 'AC Power'"
}
initialize_power_mode_contract() {
    require_ac_power
    power_mode_name=$(LANG=C LC_ALL=C system_profiler SPPowerDataType \
      | matched_parse_ac_power_mode)
    [[ "$power_mode_name" == automatic || "$power_mode_name" == high ]]
    power_mode_code=$(read_live_power_mode_code)
}
verify_power_mode_contract() {
    local name code
    require_ac_power
    name=$(LANG=C LC_ALL=C system_profiler SPPowerDataType \
      | matched_parse_ac_power_mode)
    code=$(read_live_power_mode_code)
    [[ "$name" == "$power_mode_name" && "$code" == "$power_mode_code" ]]
}

server_pid=''
monitor_pid=''
monitor_stop=''
sampler_pid=''
sampler_stop=''
caffeinate_pid=''
declare -a client_pids=()

stop_clients() {
    local pid
    for pid in "${client_pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
        wait "$pid" 2>/dev/null || true
    done
    client_pids=()
}

stop_sampler() {
    if [[ -n "$sampler_stop" ]]; then : >"$sampler_stop"; fi
    if [[ -n "$sampler_pid" ]]; then
        wait "$sampler_pid" 2>/dev/null || return 1
    fi
    sampler_pid=''
    sampler_stop=''
}

stop_server() {
    local cleanup_rc=0
    matched_physical_stop_owned_server "$server_pid" "$PORT" "$monitor_pid" \
      "$monitor_stop" || cleanup_rc=$?
    server_pid=''
    monitor_pid=''
    monitor_stop=''
    return "$cleanup_rc"
}

on_exit() {
    local original_rc=$? cleanup_rc=0
    trap - EXIT
    stop_clients || cleanup_rc=1
    stop_sampler || cleanup_rc=1
    stop_server || cleanup_rc=1
    if [[ -n "$caffeinate_pid" ]]; then
        kill -TERM "$caffeinate_pid" 2>/dev/null || true
        wait "$caffeinate_pid" 2>/dev/null || cleanup_rc=1
    fi
    thermal_cleanup_probe || cleanup_rc=1
    if ((original_rc == 0 && cleanup_rc != 0)); then exit "$cleanup_rc"; fi
    exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

require_no_model_runtime() {
    local reference_name offenders
    reference_name=$(basename "$REFERENCE_BIN")
    offenders=$(/bin/ps -axo pid=,comm= | awk -v reference="$reference_name" '
      { pid=$1; $1=""; sub(/^[[:space:]]+/, "", $0); name=$0; sub(/^.*\//, "", name);
        if (name == "hf2q" || name == reference) print pid ":" name }
    ')
    [[ -z "$offenders" ]] || {
        echo "existing model runtime detected: $offenders" >&2
        return 1
    }
    matched_require_port_available "$PORT" || return 1
    host_contention_sample "$OUT_DIR/contention-preflight.tsv" preflight \
      "$HOST_CONTENTION_GATE_OWNER_PID" || return 1
    host_contention_require_quiet preflight || return 1
}

record_calibration_observation() {
    matched_record_calibration_observation "$1" "$2" "$3" "$4" \
      "$HOST_CONTENTION_GATE_OWNER_PID" "$server_pid"
}

wait_loaded_idle_calibration() {
    local trial_dir=$1 deadline=$((SECONDS + THERMAL_SETTLE_TIMEOUT_SECONDS))
    local nominal_since=-1 thermal_log="$trial_dir/thermal-settle.tsv"
    local host_log="$trial_dir/host-settle.tsv"
    local contention_log="$trial_dir/contention-settle.tsv"
    : >"$thermal_log"; : >"$host_log"; : >"$contention_log"
    while :; do
        record_calibration_observation "$thermal_log" "$host_log" \
          "$contention_log" loaded-idle
        if [[ "$THERMAL_STATE" == nominal \
          && "$HOST_CONTENTION_STATE" == quiet ]]; then
            if ((nominal_since < 0)); then nominal_since=$SECONDS; fi
            if ((SECONDS - nominal_since >= THERMAL_SETTLE_SECONDS)); then
                thermal_validate_measurement_log "$thermal_log" \
                  "$((THERMAL_SAMPLE_SECONDS + 3))"
                matched_validate_host_observation_log "$host_log" 2 \
                  "$THERMAL_SETTLE_SECONDS" "$((THERMAL_SAMPLE_SECONDS + 3))"
                matched_validate_calibration_alignment "$thermal_log" "$host_log"
                host_contention_validate_settle_log "$contention_log" \
                  "$THERMAL_SETTLE_SECONDS" "$((THERMAL_SAMPLE_SECONDS + 3))"
                host_contention_validate_thermal_alignment "$thermal_log" \
                  "$contention_log"
                return 0
            fi
        else
            nominal_since=-1
            : >"$thermal_log"
            : >"$host_log"
            : >"$contention_log"
        fi
        ((SECONDS < deadline)) || return 1
        sleep "$THERMAL_SAMPLE_SECONDS"
    done
}

monitor_measurement() {
    local trial_dir=$1 parent_pid=$2 stop_file=$3
    local thermal_log="$trial_dir/thermal-measurement.tsv"
    local host_log="$trial_dir/host-measurement.tsv"
    local contention_log="$trial_dir/contention-measurement.tsv"
    while [[ ! -e "$stop_file" ]]; do
        if ! record_calibration_observation "$thermal_log" "$host_log" \
          "$contention_log" measurement \
          || ! host_contention_require_quiet measurement \
          || [[ "$THERMAL_STATE" != nominal ]]; then
            printf '%s\n' failed >"$trial_dir/calibration-failure.txt"
            kill -TERM "$parent_pid" 2>/dev/null || true
            return 1
        fi
        sleep "$THERMAL_SAMPLE_SECONDS"
    done
}

wait_ready() {
    local engine=$1 log=$2 endpoint=readyz deadline=$((SECONDS + 600))
    [[ "$engine" == reference ]] && endpoint=health
    while ((SECONDS < deadline)); do
        if curl --fail --silent --show-error --max-time 2 \
          "http://127.0.0.1:$PORT/$endpoint" >/dev/null 2>&1; then return 0; fi
        kill -0 "$server_pid" 2>/dev/null || {
            sed -n '1,240p' "$log" >&2
            return 1
        }
        sleep 1
    done
    return 1
}

resolve_loaded_model_id() {
    local engine=$1 trial_dir=$2 loaded
    curl --fail --silent --show-error --max-time 10 \
      "http://127.0.0.1:$PORT/v1/models" >"$trial_dir/models.json"
    if [[ "$engine" == hf2q ]]; then
        loaded=$(matched_resolve_hf2q_model_id "$trial_dir/models.json")
    else
        matched_validate_reference_model_alias "$trial_dir/models.json" "$MODEL_ID"
        loaded=$MODEL_ID
    fi
    if [[ -z "$MODEL_ID" ]]; then
        [[ "$engine" == hf2q ]]
        MODEL_ID=$loaded
        write_requests
    else
        [[ "$loaded" == "$MODEL_ID" ]]
    fi
    verify_request_manifest
}

launch_server() {
    local engine=$1 width=$2 log=$3
    local -a clean_env=(env -i "HOME=${HOME:?HOME is required}" "PATH=$PATH"
      "TMPDIR=${TMPDIR:-/tmp}" LANG=C LC_ALL=C "USER=${USER:-}" "LOGNAME=${LOGNAME:-}")
    verify_executable_identity hf2q "$HF2Q_BIN" "$HF2Q_SHA256" \
      "$hf2q_binary_snapshot"
    verify_executable_identity reference "$REFERENCE_BIN" "$REFERENCE_SHA256" \
      "$reference_binary_snapshot"
    hf2q_macos_verify_runtime_manifest "$REFERENCE_BIN" \
      "$reference_runtime_manifest"
    hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
      "$model_verification_receipt"
    if [[ "$engine" == hf2q ]]; then
        "${clean_env[@]}" HF2Q_BIN="$HF2Q_BIN" \
          HF2Q_MODEL_VERIFICATION_RECEIPT="$model_verification_receipt" \
          MODEL="$MODEL_PATH" PORT="$PORT" \
          MAX_SLOTS="$width" KV_CACHE_BUDGET_BYTES="$KV_CACHE_BUDGET_BYTES" \
          QWEN38_VISION=off QWEN38_SPECULATION=auto THINKING_TOKEN_BUDGET=0 \
          TOOL_THINKING_TOKEN_BUDGET=0 REP_PENALTY=1.05 \
          HF2Q_DECODE_MVN="$QWEN38_PHYSICAL_DECODE_MVN" \
          HF2Q_DECODE_MV_EXT="$QWEN38_PHYSICAL_DECODE_MV_EXT" \
          HF2Q_Q5K_CANONICAL_Q4X4="$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" \
          "$HF2Q_SOURCE_DIR/scripts/serve_qwen38_opencode.sh" \
          >"$log" 2>&1 &
    else
        "${clean_env[@]}" "$REFERENCE_BIN" --model "$MODEL_PATH" --alias "$MODEL_ID" \
          --host 127.0.0.1 --port "$PORT" --parallel "$width" \
          --ctx-size "$QWEN38_MATCHED_CONTEXT_TOKENS" \
          --batch-size 2048 --ubatch-size 512 --gpu-layers all --flash-attn on \
          --cache-type-k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K" \
          --cache-type-v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V" \
          --jinja --reasoning off --metrics \
          --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-min 0 \
          --spec-draft-p-min 0 --spec-draft-backend-sampling --gpu-layers-draft all \
          --temp 0 --repeat-penalty 1.05 >"$log" 2>&1 &
    fi
    server_pid=$!
}

run_warmup_wave() {
    local width=$1 trial_dir=$2 round index pid failed=0
    for round in 1 2 3; do
        local start_file="$trial_dir/warmup-$round.start"
        client_pids=()
        for ((index = 1; index <= width; index++)); do
            (
                while [[ ! -e "$start_file" ]]; do sleep 0.01; done
                curl --fail-with-body --silent --show-error --max-time 600 \
                  --header 'Content-Type: application/json' \
                  --data-binary "@$OUT_DIR/requests/warmup/lane-$index.json" \
                  "http://127.0.0.1:$PORT/v1/chat/completions" \
                  >"$trial_dir/warmup-$round-lane-$index.json"
            ) &
            client_pids+=("$!")
        done
        : >"$start_file"
        for pid in "${client_pids[@]}"; do wait "$pid" || failed=1; done
        client_pids=()
        ((failed == 0)) || return 1
        for ((index = 1; index <= width; index++)); do
            matched_validate_common_response \
              "$trial_dir/warmup-$round-lane-$index.json"
            cmp <(jq -j '.choices[0].message.content' \
              "$trial_dir/warmup-$round-lane-$index.json") \
              "$OUT_DIR/expected/warmup/lane-$index.txt"
        done
    done
}

sample_processing() {
    local engine=$1 stop_file=$2 output=$3 value now metrics
    metrics="$output.metrics"
    : >"$output"
    while [[ ! -e "$stop_file" ]]; do
        if curl --fail --silent --show-error --max-time 2 \
          "http://127.0.0.1:$PORT/metrics" >"$metrics"; then
            if [[ "$engine" == reference ]]; then
                value=$(awk '$1 == "llamacpp:requests_processing" && NF == 2 {print $2}' \
                  "$metrics")
            else
                value=$(awk '$1 == "hf2q_qwen_decode_scheduler_max_width" && NF == 2 {print $2}' \
                  "$metrics")
            fi
            if [[ "$value" =~ ^[0-9]+$ ]]; then
                now=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
                  -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
                printf '%s\t%s\n' "$now" "$value" >>"$output"
            fi
        fi
        sleep 0.02
    done
    rm -f -- "$metrics"
}

run_cache_replacement() {
    local width=$1 trial_dir=$2 group=$3 index pid failed=0
    local start_file="$trial_dir/$group-cache-buster.start"
    mkdir -p "$trial_dir/$group-cache-buster"
    client_pids=()
    for ((index = 1; index <= width; index++)); do
        (
            while [[ ! -e "$start_file" ]]; do sleep 0.01; done
            curl --fail-with-body --silent --show-error --max-time 600 \
              --header 'Content-Type: application/json' \
              --data-binary "@$OUT_DIR/requests/cache-buster/lane-$index.json" \
              "http://127.0.0.1:$PORT/v1/chat/completions" \
              >"$trial_dir/$group-cache-buster/lane-$index.json"
        ) &
        client_pids+=("$!")
    done
    : >"$start_file"
    for pid in "${client_pids[@]}"; do wait "$pid" || failed=1; done
    client_pids=()
    ((failed == 0)) || return 1
    for ((index = 1; index <= width; index++)); do
        matched_validate_common_response \
          "$trial_dir/$group-cache-buster/lane-$index.json"
    done
}

run_wave() {
    local engine=$1 trial=$2 width=$3 group=$4 trial_dir=$5
    local wave_dir="$trial_dir/$group" start_file="$trial_dir/$group.start"
    local sampler_file="$wave_dir/processing.tsv" index pid failed=0
    local started ended wave_start wave_end wave_wall clients api_total_tokens
    local comparison_work_units comparison_unit comparison_rate api_token_rate
    local request response expected scalar_request scalar_response
    local speculation_before="$wave_dir/metrics-speculation-before.txt"
    local speculation_after="$wave_dir/metrics-speculation-after.txt"
    local speculation_receipt="$wave_dir/speculation.json"
    mkdir -p "$wave_dir/responses" "$wave_dir/clients" "$wave_dir/scalar" \
      "$wave_dir/code-validation"
    curl --fail --silent --show-error --max-time 10 \
      "http://127.0.0.1:$PORT/metrics" >"$speculation_before"
    sampler_stop="$wave_dir/sampler.stop"
    sample_processing "$engine" "$sampler_stop" "$sampler_file" &
    sampler_pid=$!
    client_pids=()
    for ((index = 1; index <= width; index++)); do
        request="$OUT_DIR/requests/$group/lane-$index.json"
        if [[ "$group" == code ]]; then
            (
                while [[ ! -e "$start_file" ]]; do sleep 0.01; done
                started=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
                  -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
                curl --fail-with-body --silent --show-error --max-time 600 \
                  --header 'Content-Type: application/json' --data-binary "@$request" \
                  "http://127.0.0.1:$PORT/v1/chat/completions" \
                  >"$wave_dir/responses/lane-$index.json"
                ended=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
                  -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
                jq -n --argjson lane "$index" --argjson started_at "$started" \
                  --argjson ended_at "$ended" \
                  --slurpfile response "$wave_dir/responses/lane-$index.json" '{
                    lane:$lane,started_at:$started_at,ended_at:$ended_at,
                    wall_seconds:($ended_at-$started_at),
                    prompt_tokens:$response[0].usage.prompt_tokens,
                    completion_tokens:$response[0].usage.completion_tokens,
                    first_semantic_ms:(if $response[0].x_hf2q_timing?
                      then $response[0].x_hf2q_timing.time_to_first_token_ms else null end),
                    scalar_parity:false
                  }' >"$wave_dir/clients/lane-$index.json"
            ) &
        else
            (
                while [[ ! -e "$start_file" ]]; do sleep 0.01; done
                started=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
                  -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
                expected=$(<"$OUT_DIR/expected/repeat/lane-$index.txt")
                curl --fail-with-body --silent --show-error --no-buffer \
                  --max-time 600 --header 'Content-Type: application/json' \
                  --data-binary "@$request" \
                  "http://127.0.0.1:$PORT/v1/chat/completions" \
                  | matched_physical_parse_sse_stream "$started" \
                      "$wave_dir/responses/lane-$index.sse" \
                      "$wave_dir/responses/lane-$index.json" "$expected"
                ended=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
                  -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
                jq --argjson lane "$index" --argjson started_at "$started" \
                  --argjson ended_at "$ended" \
                  --argjson semantic_completion_tokens "$semantic_completion_tokens" \
                  --arg semantic_tokenization_sha256 "$semantic_tokenization_sha256" '
                    .stream_parse_wall_seconds=.wall_seconds
                    | . + {lane:$lane,started_at:$started_at,ended_at:$ended_at,
                      wall_seconds:($ended_at-$started_at),scalar_parity:false,
                      semantic_completion_tokens:$semantic_completion_tokens,
                      semantic_tokenization_sha256:$semantic_tokenization_sha256}' \
                  "$wave_dir/responses/lane-$index.json" \
                  >"$wave_dir/clients/lane-$index.json"
            ) &
        fi
        client_pids+=("$!")
    done
    wave_start=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
      -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
    : >"$start_file"
    for pid in "${client_pids[@]}"; do wait "$pid" || failed=1; done
    client_pids=()
    wave_end=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
      -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
    stop_sampler
    ((failed == 0)) || return 1
    matched_physical_validate_processing_peak "$width" "$sampler_file"
    curl --fail --silent --show-error --max-time 10 \
      "http://127.0.0.1:$PORT/metrics" >"$speculation_after"
    matched_physical_validate_wave_speculation "$engine" "$trial" "$group" \
      "$speculation_before" "$speculation_after" "$speculation_receipt"

    jq -s 'sort_by(.lane)' "$wave_dir"/clients/lane-*.json \
      >"$wave_dir/clients-before-scalar.json"
    matched_physical_validate_launch_skew "$wave_dir/clients-before-scalar.json" \
      "$MAX_LAUNCH_SKEW_SECONDS"
    matched_physical_validate_client_overlap \
      "$wave_dir/clients-before-scalar.json"

    for ((index = 1; index <= width; index++)); do
        response="$wave_dir/responses/lane-$index.json"
        if [[ "$group" == code ]]; then
            matched_validate_common_response "$response"
            if [[ "$engine" == hf2q ]]; then
                matched_validate_hf2q_telemetry "$response"
            else
                matched_validate_reference_telemetry "$response"
            fi
        fi
    done

    run_cache_replacement "$width" "$trial_dir" "$group"
    for ((index = 1; index <= width; index++)); do
        request="$OUT_DIR/requests/$group/lane-$index.json"
        scalar_request="$wave_dir/scalar/lane-$index-request.json"
        scalar_response="$wave_dir/scalar/lane-$index.json"
        if [[ "$group" == repeat ]]; then
            jq '.stream=false | del(.stream_options)' "$request" >"$scalar_request"
        else
            jq . "$request" >"$scalar_request"
        fi
        curl --fail-with-body --silent --show-error --max-time 600 \
          --header 'Content-Type: application/json' --data-binary "@$scalar_request" \
          "http://127.0.0.1:$PORT/v1/chat/completions" >"$scalar_response"
        matched_validate_common_response "$scalar_response"
        if [[ "$group" == code ]]; then
            qwen38_physical_validate_scalar_parity \
              "$wave_dir/responses/lane-$index.json" "$scalar_response"
        else
            matched_physical_validate_repeat_scalar \
              "$wave_dir/responses/lane-$index.json" "$scalar_response"
        fi
        jq '.scalar_parity=true' "$wave_dir/clients/lane-$index.json" \
          >"$wave_dir/clients/lane-$index.json.tmp"
        mv "$wave_dir/clients/lane-$index.json.tmp" \
          "$wave_dir/clients/lane-$index.json"
    done
    clients=$(jq -s 'sort_by(.lane)' "$wave_dir"/clients/lane-*.json)
    api_total_tokens=$(jq '[.[].completion_tokens] | add' <<<"$clients")
    if [[ "$group" == repeat ]]; then
        comparison_work_units=$(jq '[.[].semantic_completion_tokens] | add' \
          <<<"$clients")
        comparison_unit='canonical-semantic-output-token'
    else
        comparison_work_units=$width
        comparison_unit='evaluator-valid-code-request'
    fi
    wave_wall=$(awk -v start="$wave_start" -v end="$wave_end" \
      'BEGIN {printf "%.9f", end-start}')
    comparison_rate=$(awk -v units="$comparison_work_units" -v seconds="$wave_wall" \
      'BEGIN {printf "%.9f", units/seconds}')
    api_token_rate=$(awk -v tokens="$api_total_tokens" -v seconds="$wave_wall" \
      'BEGIN {printf "%.9f", tokens/seconds}')
    jq -n --arg engine "$engine" --arg group "$group" \
      --argjson trial "$trial" --argjson width "$width" \
      --argjson wave_started_at "$wave_start" \
      --argjson wave_ended_at "$wave_end" \
      --argjson wave_wall_seconds "$wave_wall" \
      --arg comparison_unit "$comparison_unit" \
      --argjson total_completion_tokens "$api_total_tokens" \
      --argjson comparison_work_units "$comparison_work_units" \
      --argjson comparison_units_per_second "$comparison_rate" \
      --argjson api_completion_tokens_per_second "$api_token_rate" \
      --argjson clients "$clients" '{schema:1,engine:$engine,trial:$trial,
        width:$width,group:$group,
        quality_pass:(if $group == "code" then false else true end),
        api_concurrency_proven:true,
        wave_started_at:$wave_started_at,wave_ended_at:$wave_ended_at,
        wave_wall_seconds:$wave_wall_seconds,comparison_unit:$comparison_unit,
        total_completion_tokens:$total_completion_tokens,
        total_semantic_completion_tokens:(if $group == "repeat"
          then $comparison_work_units else null end),
        comparison_work_units:$comparison_work_units,
        comparison_units_per_second:$comparison_units_per_second,
        diagnostics:{api_completion_tokens_per_second:$api_completion_tokens_per_second},
        clients:$clients}' >"$wave_dir/wave.json"
}

validate_trial_code_quality() {
    local width=$1 trial_dir=$2 index case_name response source_path wave_dir
    wave_dir="$trial_dir/code"
    for ((index = 1; index <= width; index++)); do
        response="$wave_dir/responses/lane-$index.json"
        case $(((index - 1) % 3)) in
            0) case_name='code-a' ;;
            1) case_name='code-b' ;;
            *) case_name='code-c' ;;
        esac
        source_path="$wave_dir/code-validation/lane-$index.rs"
        matched_extract_rust_source "$response" "$source_path"
        matched_validate_rust_case "$case_name" "$source_path" \
          "$wave_dir/code-validation/lane-$index"
        jq -S -c '{message:.choices[0].message,
          finish_reason:.choices[0].finish_reason,
          prompt_tokens:.usage.prompt_tokens,
          completion_tokens:.usage.completion_tokens}' "$response" \
          >"$wave_dir/responses/lane-$index.semantic.json"
    done
    jq '.quality_pass=true' "$wave_dir/wave.json" >"$wave_dir/wave.json.tmp"
    mv "$wave_dir/wave.json.tmp" "$wave_dir/wave.json"
}

run_trial() {
    local width=$1 trial=$2 engine=$3 width_dir=$4
    local trial_dir="$width_dir/trials/trial-$trial-$engine"
    local log="$trial_dir/server.log" thermal_log host_log contention_log
    mkdir -p "$trial_dir"
    require_no_model_runtime
    launch_server "$engine" "$width" "$log"
    wait_ready "$engine" "$log"
    resolve_loaded_model_id "$engine" "$trial_dir"
    if [[ "$engine" == hf2q ]]; then
        qwen36_bind_server_process "http://127.0.0.1:$PORT" "$server_pid" \
          "$HF2Q_BIN" "$MODEL_PATH" "$width"
    fi
    wait_loaded_idle_calibration "$trial_dir"
    thermal_log="$trial_dir/thermal-measurement.tsv"
    host_log="$trial_dir/host-measurement.tsv"
    contention_log="$trial_dir/contention-measurement.tsv"
    : >"$thermal_log"; : >"$host_log"; : >"$contention_log"
    record_calibration_observation "$thermal_log" "$host_log" \
      "$contention_log" measurement-start
    host_contention_require_quiet measurement-start
    [[ "$THERMAL_STATE" == nominal ]]
    monitor_stop="$trial_dir/monitor.stop"
    monitor_measurement "$trial_dir" "$$" "$monitor_stop" &
    monitor_pid=$!
    run_warmup_wave "$width" "$trial_dir"
    run_wave "$engine" "$trial" "$width" code "$trial_dir"
    run_wave "$engine" "$trial" "$width" repeat "$trial_dir"
    kill -0 "$server_pid"
    qwen36_reject_fatal_log "$log"
    : >"$monitor_stop"
    wait "$monitor_pid"
    monitor_pid=''; monitor_stop=''
    record_calibration_observation "$thermal_log" "$host_log" \
      "$contention_log" measurement-end
    host_contention_require_quiet measurement-end
    [[ "$THERMAL_STATE" == nominal ]]
    thermal_validate_measurement_log "$thermal_log" \
      "$((THERMAL_SAMPLE_SECONDS + 3))"
    matched_validate_host_observation_log "$host_log" 3 1 \
      "$((THERMAL_SAMPLE_SECONDS + 3))"
    matched_validate_calibration_alignment "$thermal_log" "$host_log"
    host_contention_validate_measurement_log "$contention_log" \
      "$((THERMAL_SAMPLE_SECONDS + 3))"
    host_contention_validate_thermal_alignment "$thermal_log" "$contention_log"
    [[ ! -e "$trial_dir/calibration-failure.txt" ]]
    hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
      "$model_verification_receipt"
    stop_server
    if [[ "$engine" == hf2q ]]; then
        matched_validate_qwen_frozen_routing_policy_log "$log" \
          "$QWEN38_PHYSICAL_DECODE_MVN" "$QWEN38_PHYSICAL_DECODE_MV_EXT" \
          "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4"
    fi
    validate_trial_code_quality "$width" "$trial_dir"
    verify_power_mode_contract
    qwen36_reject_fatal_log "$log"
}

thermal_prepare_probe
thermal_probe_source_sha=$(sha256_file "$THERMAL_PROBE_SOURCE")
thermal_probe_compiler_sha=$(sha256_file "$THERMAL_PROBE_COMPILER")
thermal_probe_binary_sha=$(sha256_file "$THERMAL_PROBE_BIN")
thermal_probe_compiler_version=$THERMAL_PROBE_COMPILER_VERSION
semantic_repeat_expected="$OUT_DIR/expected/repeat/canonical.txt"
semantic_repeat_receipt="$OUT_DIR/repeat-semantic-tokenization.json"
printf '%s' "$repeat_text" >"$semantic_repeat_expected"
require_no_model_runtime
matched_physical_record_semantic_repeat_tokens "$HF2Q_BIN" "$MODEL_PATH" \
  "$MODEL_SHA256" "$model_file_snapshot" "$HF2Q_SHA256" "$HF2Q_COMMIT" \
  "$semantic_repeat_expected" "$semantic_repeat_receipt"
matched_physical_validate_semantic_repeat_tokens "$semantic_repeat_receipt" \
  "$MODEL_PATH" "$MODEL_SHA256" "$model_file_snapshot" "$HF2Q_SHA256" \
  "$HF2Q_COMMIT" "$semantic_repeat_expected"
semantic_completion_tokens=$(jq -er .semantic_completion_tokens \
  "$semantic_repeat_receipt")
semantic_tokenization_sha256=$(sha256_file "$semantic_repeat_receipt")
[[ "$semantic_completion_tokens" =~ ^[1-9][0-9]*$ \
  && "$semantic_tokenization_sha256" =~ ^[0-9a-f]{64}$ ]]
initialize_power_mode_contract
require_no_model_runtime
caffeinate -dimsu -w "$$" &
caffeinate_pid=$!

width_summary_paths=()
for width in "${WIDTHS[@]}"; do
    width_dir="$OUT_DIR/widths/width-$width"
    rows="$width_dir/waves.jsonl"
    mkdir -p "$width_dir/trials"
    : >"$rows"
    trial=0
    for engine in "${TRIAL_ENGINES[@]}"; do
        trial=$((trial + 1))
        run_trial "$width" "$trial" "$engine" "$width_dir"
        jq -c . "$width_dir/trials/trial-$trial-$engine/code/wave.json" >>"$rows"
        jq -c . "$width_dir/trials/trial-$trial-$engine/repeat/wave.json" >>"$rows"
    done
    for group in code repeat; do
        for ((index = 1; index <= width; index++)); do
            if [[ "$group" == code ]]; then
                cmp "$width_dir/trials/trial-1-hf2q/$group/responses/lane-$index.semantic.json" \
                  "$width_dir/trials/trial-4-hf2q/$group/responses/lane-$index.semantic.json"
                cmp "$width_dir/trials/trial-2-reference/$group/responses/lane-$index.semantic.json" \
                  "$width_dir/trials/trial-3-reference/$group/responses/lane-$index.semantic.json"
            fi
        done
    done
    code_result=$(matched_physical_group_result_json "$rows" "$width" code \
      "$MAX_WITHIN_ENGINE_GROUP_SPREAD_PERCENT" \
      "$MAX_WITHIN_ENGINE_CASE_SPREAD_PERCENT")
    repeat_result=$(matched_physical_group_result_json "$rows" "$width" repeat \
      "$MAX_WITHIN_ENGINE_GROUP_SPREAD_PERCENT" \
      "$MAX_WITHIN_ENGINE_CASE_SPREAD_PERCENT")
    speculation_waves=$(jq -s . \
      "$width_dir/trials/trial-1-hf2q/code/speculation.json" \
      "$width_dir/trials/trial-1-hf2q/repeat/speculation.json" \
      "$width_dir/trials/trial-2-reference/code/speculation.json" \
      "$width_dir/trials/trial-2-reference/repeat/speculation.json" \
      "$width_dir/trials/trial-3-reference/code/speculation.json" \
      "$width_dir/trials/trial-3-reference/repeat/speculation.json" \
      "$width_dir/trials/trial-4-hf2q/code/speculation.json" \
      "$width_dir/trials/trial-4-hf2q/repeat/speculation.json")
    jq -e --argjson minimum "$MIN_HF2Q_RATIO" '
      .quality_pass == true and .stability.stable == true
      and .stability.observed_band_dominance == true
      and .hf2q_over_reference_comparison_rate >= $minimum
    ' <<<"$code_result" >/dev/null
    jq -e --argjson minimum "$MIN_HF2Q_RATIO" '
      .quality_pass == true and .stability.stable == true
      and .stability.observed_band_dominance == true
      and .hf2q_over_reference_comparison_rate >= $minimum
      and .reference_over_hf2q_p95_wall >= $minimum
      and .semantic_ttft.required == true
      and .semantic_ttft.stable == true
      and .semantic_ttft.observed_band_dominance == true
      and .semantic_ttft.reference_over_hf2q_p95 >= $minimum
    ' <<<"$repeat_result" >/dev/null
    physical_proof=$(matched_physical_extract_proof_json \
      "$PHYSICAL_MATRIX_RECEIPT" "$MODEL_FORMAT" "$width")
    jq -n --argjson width "$width" --argjson physical "$physical_proof" \
      --argjson code "$code_result" --argjson repeat "$repeat_result" \
      --arg hf2q_speculation_policy "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
      --arg reference_speculation_policy "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY" \
      --argjson dense_decode_mvn "$QWEN38_PHYSICAL_DECODE_MVN" \
      --argjson dense_decode_mv_ext "$QWEN38_PHYSICAL_DECODE_MV_EXT" \
      --argjson dense_q5k_canonical_q4x4 \
        "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" \
      --argjson speculation_waves "$speculation_waves" \
      --argjson minimum "$MIN_HF2Q_RATIO" '{schema:2,verdict:"pass",width:$width,
        samples:{hf2q:2,reference:2},
        scalar_replay:{hf2q:true,reference:true},
        hf2q_effective_routing_policy:{dense_decode_mvn:$dense_decode_mvn,
          dense_decode_mv_ext:$dense_decode_mv_ext,
          dense_q5k_canonical_q4x4:$dense_q5k_canonical_q4x4},
        physical_proof:$physical,
        speculation:{hf2q_policy:$hf2q_speculation_policy,
          reference_policy:$reference_speculation_policy,waves:$speculation_waves},
        acceptance:{minimum_hf2q_ratio:$minimum},code:$code,repeat:$repeat}' \
      >"$width_dir/summary.json"
    width_summary_paths+=("$width_dir/summary.json")
done

width_results=$(jq -s . "${width_summary_paths[@]}")
jq -e 'map(.width) == [1,2,4,8,16]' <<<"$width_results" >/dev/null
matched_physical_require_clean_exact_source "$ROOT_DIR" "$harness_commit" harness
matched_physical_require_clean_exact_source \
  "$HF2Q_SOURCE_DIR" "$HF2Q_COMMIT" hf2q
matched_physical_require_clean_exact_source \
  "$REFERENCE_SOURCE_DIR" "$REFERENCE_COMMIT" reference
qwen38_validate_pinned_peer_commit "$REFERENCE_COMMIT"
[[ "$(sha256_file "$reference_pin_path")" == "$reference_pin_sha" ]]
verify_executable_identity hf2q "$HF2Q_BIN" "$HF2Q_SHA256" \
  "$hf2q_binary_snapshot"
verify_executable_identity reference "$REFERENCE_BIN" "$REFERENCE_SHA256" \
  "$reference_binary_snapshot"
hf2q_macos_verify_runtime_manifest "$REFERENCE_BIN" \
  "$reference_runtime_manifest"
reference_runtime_manifest_final=$(hf2q_macos_runtime_manifest "$REFERENCE_BIN")
reference_runtime_manifest_final_sha=$(printf '%s\n' \
  "$reference_runtime_manifest_final" | shasum -a 256 | awk '{print $1}')
[[ "$reference_runtime_manifest_final_sha" == \
  "$REFERENCE_RUNTIME_MANIFEST_SHA256" ]] || {
    echo "reference runtime closure changed before sealing" >&2
    exit 2
}
hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
  "$model_verification_receipt"
matched_physical_validate_semantic_repeat_tokens "$semantic_repeat_receipt" \
  "$MODEL_PATH" "$MODEL_SHA256" "$model_file_snapshot" "$HF2Q_SHA256" \
  "$HF2Q_COMMIT" "$semantic_repeat_expected"
verify_request_manifest
qwen38_validate_physical_matrix_seal "$PHYSICAL_MATRIX_RECEIPT"
[[ "$(sha256_file "$PHYSICAL_MATRIX_RECEIPT")" == "$PHYSICAL_MATRIX_SHA256" ]]

script_sha=$(sha256_file "$SCRIPT_DIR/qwen38_matched_physical_abba.sh")
contract_sha=$(sha256_file "$SCRIPT_DIR/qwen38_matched_physical_contract.sh")
artifact_contract_sha=$(sha256_file "$SCRIPT_DIR/qwen38_artifact_contract.sh")
physical_contract_sha=$(sha256_file "$SCRIPT_DIR/qwen38_physical_multislot_contract.sh")
request_manifest_sha=$(sha256_file "$request_manifest")
evidence_manifest="$OUT_DIR/evidence.sha256"
: >"$evidence_manifest.tmp"
while IFS= read -r path; do
    printf '%s  %s\n' "$(sha256_file "$OUT_DIR/$path")" "$path" \
      >>"$evidence_manifest.tmp"
done < <(cd "$OUT_DIR" && find . -type f \
  ! -name evidence.sha256 ! -name evidence.sha256.tmp \
  ! -name summary.json ! -name summary.json.tmp ! -name result.sha256 \
  -print | sed 's#^./##' | sort)
mv "$evidence_manifest.tmp" "$evidence_manifest"
(cd "$OUT_DIR" && shasum -a 256 -c evidence.sha256 >/dev/null)
evidence_manifest_sha=$(sha256_file "$evidence_manifest")

jq -n --arg verdict pass --arg harness_commit "$harness_commit" \
  --arg hf2q_commit "$HF2Q_COMMIT" \
  --arg hf2q_sha "$HF2Q_SHA256" --arg hf2q_snapshot "$hf2q_binary_snapshot" \
  --arg reference_commit "$REFERENCE_COMMIT" \
  --arg reference_sha "$REFERENCE_SHA256" \
  --arg reference_snapshot "$reference_binary_snapshot" \
  --arg reference_runtime_manifest_sha "$reference_runtime_manifest_sha" \
  --arg expected_reference_runtime_manifest_sha \
    "$REFERENCE_RUNTIME_MANIFEST_SHA256" \
  --arg reference_pin_sha "$reference_pin_sha" \
  --arg semantic_tokenization_sha "$semantic_tokenization_sha256" \
  --arg physical_sha "$PHYSICAL_MATRIX_SHA256" \
  --arg model_id "$MODEL_ID" --arg model_path "$MODEL_PATH" \
  --arg model_format "$MODEL_FORMAT" --arg model_file "$qualified_file" \
  --arg model_sha "$MODEL_SHA256" --arg model_snapshot "$model_file_snapshot" \
  --arg model_verification "$model_verification_mode" \
  --arg repository "$QWEN38_QUALIFIED_MODEL_REPOSITORY" \
  --arg revision "$QWEN38_QUALIFIED_MODEL_REVISION" \
  --arg hardware_model "$hardware_model" --arg hardware_chip "$hardware_chip" \
  --arg hardware_arch "$hardware_arch" --arg hardware_os "$hardware_os" \
  --arg power_name "$power_mode_name" --argjson power_code "$power_mode_code" \
  --arg mlx_version "$mlx_native_version" --arg mlx_checksum "$mlx_native_checksum" \
  --arg script_sha "$script_sha" --arg contract_sha "$contract_sha" \
  --arg artifact_contract_sha "$artifact_contract_sha" \
  --arg physical_contract_sha "$physical_contract_sha" \
  --arg request_manifest_sha "$request_manifest_sha" \
  --arg evidence_manifest_sha "$evidence_manifest_sha" \
  --arg thermal_source_sha "$thermal_probe_source_sha" \
  --arg thermal_compiler_sha "$thermal_probe_compiler_sha" \
  --arg thermal_compiler_version "$thermal_probe_compiler_version" \
  --arg thermal_binary_sha "$thermal_probe_binary_sha" \
  --argjson model_bytes "$MODEL_BYTES" --argjson model_type "$qualified_file_type" \
  --argjson memory_bytes "$hardware_memory_bytes" \
  --argjson semantic_completion_tokens "$semantic_completion_tokens" \
  --argjson minimum "$MIN_HF2Q_RATIO" \
  --argjson kv_budget "$KV_CACHE_BUDGET_BYTES" \
  --argjson dense_decode_mvn "$QWEN38_PHYSICAL_DECODE_MVN" \
  --argjson dense_decode_mv_ext "$QWEN38_PHYSICAL_DECODE_MV_EXT" \
  --argjson dense_q5k_canonical_q4x4 \
    "$QWEN38_MATCHED_HF2Q_Q5K_CANONICAL_Q4X4" \
  --arg hf2q_speculation "$QWEN38_MATCHED_HF2Q_SPECULATION_POLICY" \
  --arg reference_speculation "$QWEN38_MATCHED_REFERENCE_SPECULATION_POLICY" \
  --arg hf2q_kv "$QWEN38_MATCHED_HF2Q_KV_CACHE" \
  --arg reference_k "$QWEN38_MATCHED_REFERENCE_KV_CACHE_K" \
  --arg reference_v "$QWEN38_MATCHED_REFERENCE_KV_CACHE_V" \
  --argjson context_tokens "$QWEN38_MATCHED_CONTEXT_TOKENS" \
  --argjson max_launch_skew "$MAX_LAUNCH_SKEW_SECONDS" \
  --arg host_contention_policy "$HOST_CONTENTION_POLICY" \
  --argjson host_contention_max_foreign_cpu_percent \
    "$HOST_CONTENTION_MAX_FOREIGN_CPU_PERCENT" \
  --argjson host_contention_owner_pgid \
    "$HOST_CONTENTION_GATE_OWNER_PID" \
  --argjson results "$width_results" '{
    schema:2,verdict:$verdict,gate:"qwen38-matched-physical-abba",
    harness:{commit:$harness_commit,
      source_binding:"clean exact harness worktree"},
    hf2q:{commit:$hf2q_commit,binary_sha256:$hf2q_sha,
      binary_file_snapshot:$hf2q_snapshot,
      source_binding:"embedded exact commit plus clean exact source worktree",
      mlx_native:{version:$mlx_version,checksum:$mlx_checksum},
      effective_routing_policy:{dense_decode_mvn:$dense_decode_mvn,
        dense_decode_mv_ext:$dense_decode_mv_ext,
        dense_q5k_canonical_q4x4:$dense_q5k_canonical_q4x4}},
    reference:{commit:$reference_commit,binary_sha256:$reference_sha,
      binary_file_snapshot:$reference_snapshot,
      runtime_manifest_sha256:$reference_runtime_manifest_sha,
      expected_runtime_manifest_sha256:
        $expected_reference_runtime_manifest_sha,
      pin_policy:"observed-current-then-frozen",frozen_for_run:true,
      pin_file_sha256:$reference_pin_sha},
    physical_matrix_sha256:$physical_sha,
    model:{id:$model_id,path:$model_path,format:$model_format,file:$model_file,
      repository:$repository,revision:$revision,sha256:$model_sha,
      bytes:$model_bytes,gguf_file_type:$model_type,
      verification:$model_verification,file_snapshot:$model_snapshot},
    workload:{widths:[1,2,4,8,16],
      trial_order:["hf2q","reference","reference","hf2q"],
      groups:["code","repeat"],temperature:0,repetition_penalty:1.05,
      speculation:{hf2q:$hf2q_speculation,reference:$reference_speculation},
      cache_settings:{
        hf2q:{format:$hf2q_kv,budget_bytes:$kv_budget,
          context_tokens_per_slot:$context_tokens},
        reference:{k_format:$reference_k,v_format:$reference_v,
          context_tokens_total:$context_tokens}},
      scalar_replay_per_lane:true,
      repeat_semantic_tokenization:{receipt_sha256:$semantic_tokenization_sha,
        completion_tokens:$semantic_completion_tokens,
        unit:"canonical-semantic-output-token"},
      reference_parallelism_matches_width:true},
    acceptance:{minimum_hf2q_ratio:$minimum,
      maximum_launch_skew_seconds:$max_launch_skew,
      maximum_group_spread_percent:5,maximum_case_spread_percent:10},
    host_contention:{policy:$host_contention_policy,
      maximum_foreign_cpu_percent:$host_contention_max_foreign_cpu_percent,
      owner_scope:"release-gate-process-group",
      owner_pgid:$host_contention_owner_pgid,continuous:true},
    hardware:{model:$hardware_model,chip:$hardware_chip,arch:$hardware_arch,
      os_version:$hardware_os,memory_bytes:$memory_bytes,
      power_mode:{name:$power_name,numeric_canary:$power_code},
      thermal_probe:{source_sha256:$thermal_source_sha,
        compiler_sha256:$thermal_compiler_sha,
        compiler_version:$thermal_compiler_version,
        binary_sha256:$thermal_binary_sha}},
    evidence:{script_sha256:$script_sha,contract_sha256:$contract_sha,
      artifact_contract_sha256:$artifact_contract_sha,
      physical_contract_sha256:$physical_contract_sha,
      request_manifest_sha256:$request_manifest_sha,
      reference_runtime_manifest_sha256:$reference_runtime_manifest_sha,
      expected_reference_runtime_manifest_sha256:
        $expected_reference_runtime_manifest_sha,
      reference_pin_file_sha256:$reference_pin_sha,
      evidence_manifest_sha256:$evidence_manifest_sha},
    results:$results
  }' >"$OUT_DIR/summary.json.tmp"
matched_physical_validate_inner_summary "$OUT_DIR/summary.json.tmp"
matched_physical_validate_expected_reference_closure \
  "$OUT_DIR/summary.json.tmp" "$REFERENCE_RUNTIME_MANIFEST_SHA256"
matched_publish_result "$OUT_DIR/summary.json.tmp" "$OUT_DIR/summary.json" \
  "$evidence_manifest" "$OUT_DIR/result.sha256"
if ! matched_physical_validate_reopened_child "$OUT_DIR"; then
    mv "$OUT_DIR/summary.json" "$OUT_DIR/summary.json.unsealed"
    exit 1
fi
printf 'matched physical ABBA result sealed at %s\n' "$OUT_DIR/summary.json"
