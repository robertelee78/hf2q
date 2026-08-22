#!/usr/bin/env bash
set -euo pipefail

# Matched single-user Qwen3.8 comparison on one exact GGUF. This developer
# harness runs hf2q and a caller-bound clean reference HEAD sequentially; the
# reference is never a product dependency or conversion pin. Fresh servers run in ABBA order,
# every response is retained, and speed is accepted only after complete-code,
# exact-transcription, and calibrated-host gates pass.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPT_DIR="$ROOT_DIR/scripts"
HF2Q_BIN=${HF2Q_BIN:?HF2Q_BIN is required}
HF2Q_COMMIT=${HF2Q_COMMIT:?HF2Q_COMMIT is required}
HF2Q_SHA256=${HF2Q_SHA256:?HF2Q_SHA256 is required}
REFERENCE_BIN=${REFERENCE_BIN:?REFERENCE_BIN is required}
REFERENCE_SOURCE_DIR=${REFERENCE_SOURCE_DIR:?REFERENCE_SOURCE_DIR is required}
REFERENCE_COMMIT=${REFERENCE_COMMIT:?REFERENCE_COMMIT is required}
REFERENCE_SHA256=${REFERENCE_SHA256:?REFERENCE_SHA256 is required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
MODEL_BYTES=${MODEL_BYTES:-19535701568}
OUT_DIR=${OUT_DIR:?OUT_DIR is required and must be fresh}
PORT=${PORT:-18086}
MODEL_ID=${MODEL_ID:-}
MIN_HF2Q_RATIO=${MIN_HF2Q_RATIO:-1.0}

readonly MAX_TOKENS=256
readonly TTFT_MAX_TOKENS=16
readonly CASES='code-a code-b code-c repeat-a repeat-b repeat-c'
readonly SUSTAINED_WARMUP_CASES='warmup-a warmup-b warmup-c'
readonly MIN_SUSTAINED_WARMUP_TOKENS=512
readonly MAX_WARMUP_TO_MEASUREMENT_SECONDS=2
readonly TRIAL_ORDER='hf2q reference reference hf2q'
readonly EXPECTED_MLX_NATIVE_VERSION='0.11.2'
readonly EXPECTED_MLX_NATIVE_CHECKSUM='22f4bd6661e77994c6f26a79fdd2c188f3d5252aa7e51616f5feb080b22da8e0'
readonly QUALIFIED_MODEL_REPOSITORY='jenerallee78/Qwen3.8-27B-Abliterated-SFT'
readonly QUALIFIED_MODEL_REVISION='0a72776892f98db49381fdf69f4b9982222ec9dc'
readonly QUALIFIED_MODEL_FILE='gguf/qwen38-abliterated-sft-q5_k_m.gguf'
readonly QUALIFIED_MODEL_SHA256='4b19f41c391d962882e459be3315d4e3c54079892db2848f66b78815b185156e'
readonly THERMAL_SETTLE_SECONDS=30
readonly THERMAL_SETTLE_TIMEOUT_SECONDS=900
readonly THERMAL_SAMPLE_SECONDS=2
readonly MAX_WITHIN_ENGINE_GROUP_SPREAD_PERCENT=5
readonly MAX_WITHIN_ENGINE_CASE_SPREAD_PERCENT=10

if [[ ${HF2Q_THERMAL_SWIFTC_BIN+x} || ${HF2Q_THERMAL_PROBE_BIN+x} \
  || ${HF2Q_THERMAL_PROBE_SOURCE+x} ]]; then
    echo "thermal probe overrides are reserved for isolated contract tests" >&2
    exit 2
fi
readonly HF2Q_THERMAL_SWIFTC_BIN=/usr/bin/swiftc

# shellcheck source=scripts/qwen36_watchdog_validate.sh
source "$SCRIPT_DIR/qwen36_watchdog_validate.sh"
# shellcheck source=scripts/macos_thermal_guard.sh
source "$SCRIPT_DIR/macos_thermal_guard.sh"
# shellcheck source=scripts/qwen38_matched_reference_contract.sh
source "$SCRIPT_DIR/qwen38_matched_reference_contract.sh"

for command in awk basename caffeinate cat cmp cp curl date dirname find git jq kill \
  lsof mkdir mv perl pmset ps rg rustc sed shasum sort stat sw_vers \
  system_profiler sysctl tr uname; do
    command -v "$command" >/dev/null || {
        echo "missing required command: $command" >&2
        exit 2
    }
done
[[ -x /usr/bin/swiftc ]] || {
    echo "the calibrated thermal-state compiler is unavailable" >&2
    exit 2
}
for path in "$HF2Q_BIN" "$REFERENCE_BIN"; do
    [[ -x "$path" ]] || {
        echo "server binary is missing or not executable: $path" >&2
        exit 2
    }
done
[[ -f "$MODEL_PATH" ]] || {
    echo "Qwen3.8 model is missing: $MODEL_PATH" >&2
    exit 2
}
[[ -d "$ROOT_DIR/.git" || -f "$ROOT_DIR/.git" ]] || {
    echo "hf2q git worktree is missing: $ROOT_DIR" >&2
    exit 2
}
[[ -d "$REFERENCE_SOURCE_DIR/.git" || -f "$REFERENCE_SOURCE_DIR/.git" ]] || {
    echo "reference git worktree is missing: $REFERENCE_SOURCE_DIR" >&2
    exit 2
}
for commit in "$HF2Q_COMMIT" "$REFERENCE_COMMIT"; do
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || {
        echo "source commits must be exact lowercase 40-character digests" >&2
        exit 2
    }
done
for digest in "$HF2Q_SHA256" "$MODEL_SHA256" "$REFERENCE_SHA256"; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
        echo "artifact digests must be lowercase 64-character SHA-256 values" >&2
        exit 2
    }
done
for setting in MODEL_BYTES PORT; do
    value=${!setting}
    [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "$setting must be a non-negative integer (got: $value)" >&2
        exit 2
    }
done
((MODEL_BYTES > 0 && PORT >= 1 && PORT <= 65535)) || {
    echo "invalid model byte size or port" >&2
    exit 2
}
[[ "$MIN_HF2Q_RATIO" =~ ^(0|[1-9][0-9]*)(\.[0-9]+)?$ ]] || {
    echo "MIN_HF2Q_RATIO must be a non-negative number" >&2
    exit 2
}
awk -v minimum="$MIN_HF2Q_RATIO" 'BEGIN { exit !(minimum >= 1.0) }' || {
    echo "MIN_HF2Q_RATIO must be >= 1.0" >&2
    exit 2
}
[[ "$MODEL_SHA256" == "$QUALIFIED_MODEL_SHA256" ]] || {
    echo "matched Q5_K_M gate requires the qualified model SHA-256" >&2
    exit 2
}
if [[ -e "$OUT_DIR" && -n "$(find "$OUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "matched-reference output directory must be fresh: $OUT_DIR" >&2
    exit 2
fi
[[ "$OUT_DIR" == /* ]] || {
    echo "matched-reference output directory must be an absolute path" >&2
    exit 2
}
case "$OUT_DIR" in
    "$ROOT_DIR"|"$ROOT_DIR"/*|"$REFERENCE_SOURCE_DIR"|"$REFERENCE_SOURCE_DIR"/*)
        echo "matched-reference evidence must live outside both source worktrees" >&2
        exit 2
        ;;
esac

require_clean_exact_source() {
    local source_dir=$1
    local expected_commit=$2
    local label=$3
    local actual_commit
    local status

    actual_commit=$(git -C "$source_dir" rev-parse HEAD)
    [[ "$actual_commit" == "$expected_commit" ]] || {
        echo "$label HEAD mismatch: expected=$expected_commit actual=$actual_commit" >&2
        return 1
    }
    status=$(git -C "$source_dir" status --porcelain)
    [[ -z "$status" ]] || {
        echo "$label worktree must be clean" >&2
        printf '%s\n' "$status" >&2
        return 1
    }
}

sha256_file() {
    shasum -a 256 "$1" | awk '{print $1}'
}

file_snapshot() {
    stat -f '%d:%i:%z:%m:%c' "$1" 2>/dev/null \
      || stat -c '%d:%i:%s:%Y:%Z' "$1" 2>/dev/null
}

verify_executable_identity() {
    local label=$1
    local executable=$2
    local expected_sha=$3
    local expected_snapshot=$4
    local actual_sha actual_snapshot

    [[ -x "$executable" ]] || {
        echo "$label executable disappeared: $executable" >&2
        return 1
    }
    actual_snapshot=$(file_snapshot "$executable") || return 1
    actual_sha=$(sha256_file "$executable") || return 1
    [[ "$actual_snapshot" == "$expected_snapshot" \
      && "$actual_sha" == "$expected_sha" ]] || {
        echo "$label executable changed during the matched run" >&2
        return 1
    }
}

file_bytes() {
    stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1" 2>/dev/null
}

require_clean_exact_source "$ROOT_DIR" "$HF2Q_COMMIT" hf2q
require_clean_exact_source "$REFERENCE_SOURCE_DIR" "$REFERENCE_COMMIT" reference
[[ "$(sha256_file "$HF2Q_BIN")" == "$HF2Q_SHA256" ]] || {
    echo "hf2q binary SHA-256 mismatch" >&2
    exit 2
}
grep -aFq "$HF2Q_COMMIT" "$HF2Q_BIN" || {
    echo "hf2q binary does not embed exact commit $HF2Q_COMMIT" >&2
    exit 2
}
reference_version=$("$REFERENCE_BIN" --version 2>&1)
[[ "$reference_version" == *"${REFERENCE_COMMIT:0:9}"* ]] || {
    echo "reference binary version does not match commit $REFERENCE_COMMIT" >&2
    printf '%s\n' "$reference_version" >&2
    exit 2
}
[[ "$(sha256_file "$REFERENCE_BIN")" == "$REFERENCE_SHA256" ]] || {
    echo "reference binary SHA-256 mismatch" >&2
    exit 2
}
hf2q_binary_snapshot=$(file_snapshot "$HF2Q_BIN")
reference_binary_snapshot=$(file_snapshot "$REFERENCE_BIN")
[[ -n "$hf2q_binary_snapshot" && -n "$reference_binary_snapshot" ]]
[[ "$(file_bytes "$MODEL_PATH")" == "$MODEL_BYTES" ]] || {
    echo "Qwen3.8 model byte size mismatch" >&2
    exit 2
}

hardware_model=$(sysctl -n hw.model 2>/dev/null || printf unknown)
hardware_chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf unknown)
hardware_memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || printf 0)
hardware_arch=$(uname -m)
hardware_os=$(sw_vers -productVersion 2>/dev/null || uname -r)
[[ "$hardware_arch" == arm64 ]] || {
    echo "matched Qwen3.8 comparison requires Apple Silicon arm64" >&2
    exit 2
}
[[ "$hardware_model" != unknown && "$hardware_chip" != unknown \
  && "$hardware_memory_bytes" =~ ^[1-9][0-9]*$ ]] || {
    echo "matched Qwen3.8 comparison requires complete hardware identity" >&2
    exit 2
}

lock_identity=$(awk '
  $0 == "name = \"mlx-native\"" { found = 1; next }
  found && /^version = / { version = $3; gsub(/\"/, "", version); next }
  found && /^checksum = / { checksum = $3; gsub(/\"/, "", checksum); print version " " checksum; exit }
' "$ROOT_DIR/Cargo.lock")
[[ "$lock_identity" == "$EXPECTED_MLX_NATIVE_VERSION $EXPECTED_MLX_NATIVE_CHECKSUM" ]] || {
    echo "Cargo.lock mlx-native identity is not the qualified release" >&2
    exit 2
}

mkdir -p "$OUT_DIR/requests" "$OUT_DIR/expected" "$OUT_DIR/trials"
git -C "$ROOT_DIR" status --porcelain=v2 >"$OUT_DIR/hf2q-status.txt"
git -C "$REFERENCE_SOURCE_DIR" status --porcelain=v2 >"$OUT_DIR/reference-status.txt"
printf '%s\n' "$reference_version" >"$OUT_DIR/reference-version.txt"

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
    hf2q_release_prepare_model_verification \
        "$MODEL_PATH" "$MODEL_SHA256" "$model_verification_receipt" \
        "$model_verification_cache_dir"
    model_verification_mode=$(jq -er .run_verification "$model_verification_receipt")
else
    provided_model_verification_receipt=$model_verification_receipt
    hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
      "$provided_model_verification_receipt"
    model_verification_receipt="$OUT_DIR/model-verification.json"
    jq . "$provided_model_verification_receipt" >"$model_verification_receipt.tmp"
    mv "$model_verification_receipt.tmp" "$model_verification_receipt"
    model_verification_mode=provided_receipt
fi
hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
    "$model_verification_receipt"
model_file_snapshot=$(jq -er .file_snapshot "$model_verification_receipt")

write_request() {
    local name=$1
    local system_prompt=$2
    local user_prompt=$3
    local expected=${4:-}

    jq -n --arg model "$MODEL_ID" --arg system "$system_prompt" \
      --arg user "$user_prompt" --argjson max_tokens "$MAX_TOKENS" '{
        model:$model,
        messages:[
          {role:"system",content:$system},
          {role:"user",content:$user}
        ],
        max_tokens:$max_tokens,
        temperature:0,
        stream:false,
        chat_template_kwargs:{enable_thinking:false}
      }' >"$OUT_DIR/requests/$name.json.tmp"
    mv "$OUT_DIR/requests/$name.json.tmp" "$OUT_DIR/requests/$name.json"
    if [[ -n "$expected" ]]; then
        printf '%s' "$expected" >"$OUT_DIR/expected/$name.txt"
    fi
}

write_request code-a \
  'Return only one complete compilable Rust source file. Do not use Markdown fences or prose.' \
  'Implement fn fibonacci(n: u64) -> u64 iteratively. Include exactly one unit test containing exactly one assertion. Do not add benchmarks.'
write_request code-b \
  'Return only one complete compilable Rust source file. Do not use Markdown fences or prose.' \
  'Implement fn binary_search(xs: &[i32], needle: i32) -> Option<usize> iteratively for a sorted slice. Include exactly one unit test containing exactly one assertion. Do not add benchmarks.'
write_request code-c \
  'Return only one complete compilable Rust source file. Do not use Markdown fences or prose.' \
  'Implement fn gcd(mut a: u64, mut b: u64) -> u64 with the iterative Euclidean algorithm. Include exactly one unit test containing exactly one assertion. Do not add benchmarks.'
write_request repeat-a \
  'You are a transcription engine. Follow the request exactly.' \
  $'Repeat the following text exactly, with no introduction or quotation marks:\n\nThe copper observatory stood above the harbor while seven quiet instruments recorded wind, tide, temperature, pressure, cloud cover, rainfall, and the slow vibration of the old bridge. Each evening the keeper copied those readings into a blue ledger, checked every column twice, and left the completed page beneath a brass lamp for the morning crew.' \
  'The copper observatory stood above the harbor while seven quiet instruments recorded wind, tide, temperature, pressure, cloud cover, rainfall, and the slow vibration of the old bridge. Each evening the keeper copied those readings into a blue ledger, checked every column twice, and left the completed page beneath a brass lamp for the morning crew.'
write_request repeat-b \
  'You are a transcription engine. Follow the request exactly.' \
  $'Repeat the following text exactly, with no introduction or quotation marks:\n\nA patient compiler reads the module, resolves every import, expands each macro, verifies every lifetime, checks every trait bound, lowers the typed program into an intermediate form, applies conservative optimizations, and finally emits a deterministic object file. The build report records each phase so that a later engineer can reproduce the result.' \
  'A patient compiler reads the module, resolves every import, expands each macro, verifies every lifetime, checks every trait bound, lowers the typed program into an intermediate form, applies conservative optimizations, and finally emits a deterministic object file. The build report records each phase so that a later engineer can reproduce the result.'
write_request repeat-c \
  'You are a transcription engine. Follow the request exactly.' \
  $'Repeat the following text exactly, with no introduction or quotation marks:\n\nAt sunrise the research vessel crossed the calm channel, passed three red buoys, and turned north toward the ice station. The crew calibrated the sonar array, sealed the sample containers, reviewed the emergency checklist, and logged the exact coordinates before lowering the first instrument through the open deck.' \
  'At sunrise the research vessel crossed the calm channel, passed three red buoys, and turned north toward the ice station. The crew calibrated the sonar array, sealed the sample containers, reviewed the emergency checklist, and logged the exact coordinates before lowering the first instrument through the open deck.'
write_request warmup \
  'This request is benchmark warmup only.' \
  'Return exactly WARMUP.' 'WARMUP'
warmup_a_segment='Amber lanterns marked the northern footpath while careful surveyors measured every stone, copied each coordinate into waterproof notebooks, checked the compass twice, and returned the polished instruments to numbered cedar cases before the evening rain reached the quiet valley.'
warmup_b_segment='Copper relays clicked inside the control room as patient technicians inspected every cable, compared the voltage readings with yesterday records, signed the maintenance sheet, and placed three calibrated meters beside the sealed cabinet for the incoming morning team.'
warmup_c_segment='Silver clouds crossed the observatory dome while four astronomers aligned the mirrors, verified the tracking clock, recorded the cooling pressure, archived the raw images, and carried the completed logbook downstairs before the first pale sunlight appeared above the ridge.'
write_request warmup-a \
  'You are a transcription engine. Return only the requested text exactly.' \
  "Repeat the following text exactly, with no introduction or quotation marks: $warmup_a_segment $warmup_a_segment $warmup_a_segment $warmup_a_segment" \
  "$warmup_a_segment $warmup_a_segment $warmup_a_segment $warmup_a_segment"
write_request warmup-b \
  'You are a transcription engine. Return only the requested text exactly.' \
  "Repeat the following text exactly, with no introduction or quotation marks: $warmup_b_segment $warmup_b_segment $warmup_b_segment $warmup_b_segment" \
  "$warmup_b_segment $warmup_b_segment $warmup_b_segment $warmup_b_segment"
write_request warmup-c \
  'You are a transcription engine. Return only the requested text exactly.' \
  "Repeat the following text exactly, with no introduction or quotation marks: $warmup_c_segment $warmup_c_segment $warmup_c_segment $warmup_c_segment" \
  "$warmup_c_segment $warmup_c_segment $warmup_c_segment $warmup_c_segment"
jq -n --arg model "$MODEL_ID" --argjson max_tokens "$TTFT_MAX_TOKENS" '{
  model:$model,
  messages:[
    {role:"system",content:"This is a streamed semantic-latency probe."},
    {role:"user",content:"Return exactly STREAM-TTFT."}
  ],
  max_tokens:$max_tokens,
  temperature:0,
  stream:true,
  chat_template_kwargs:{enable_thinking:false}
}' >"$OUT_DIR/requests/ttft.json.tmp"
mv "$OUT_DIR/requests/ttft.json.tmp" "$OUT_DIR/requests/ttft.json"
printf '%s' 'STREAM-TTFT' >"$OUT_DIR/expected/ttft.txt"

request_manifest="$OUT_DIR/requests.sha256"
write_request_manifest() {
    local manifest_tmp="$request_manifest.tmp"
    : >"$manifest_tmp"
    while IFS= read -r relative_path; do
        printf '%s  %s\n' "$(sha256_file "$OUT_DIR/$relative_path")" \
          "$relative_path" >>"$manifest_tmp"
    done < <(cd "$OUT_DIR" && find requests expected -type f -print | sort)
    mv "$manifest_tmp" "$request_manifest"
}

verify_request_manifest() {
    [[ -s "$request_manifest" ]] || {
        echo "matched request manifest is missing" >&2
        return 1
    }
    (cd "$OUT_DIR" && shasum -a 256 -c "$(basename "$request_manifest")" \
      >/dev/null)
}

launch_settings="$OUT_DIR/launch-settings.json"
launch_settings_sha=''
write_launch_settings() {
    jq -n --arg model_id "$MODEL_ID" --argjson port "$PORT" \
      --argjson max_tokens "$MAX_TOKENS" \
      --argjson ttft_max_tokens "$TTFT_MAX_TOKENS" '{
      schema:1,
      common:{model_id:$model_id,port:$port,temperature:0,
        repetition_penalty:1.05,thinking:false,seed:null,
        measured_max_tokens:$max_tokens,stream_ttft_max_tokens:$ttft_max_tokens},
      hf2q:{slots:1,context_tokens:262144,kv_cache:"tq-kv",
        kv_cache_budget_bytes:51539607552,scheduler:"inflight-batched",
        speculation:"shipping-auto-history-then-mtp",
        dense_decode_mvn:0,dense_decode_mv_ext:1,
        encoder_session:true,ffn_terminal_k_batch:8},
      reference:{slots:1,context_tokens:262144,batch_tokens:2048,
        microbatch_tokens:512,gpu_layers:"all",flash_attention:true,
        kv_cache_k:"q8_0",kv_cache_v:"q8_0",jinja:true,
        reasoning:false,speculation:"fixed-k3-mtp",
        draft_max:3,draft_min:0,draft_probability_minimum:0,
        draft_backend_sampling:true},
      host_calibration:{power:"ac",energy_mode:"automatic-or-high",
        thermal:"nominal",sample_seconds:2,
        settle_seconds:30,
        forbidden_processes:["hf2q","llama-server","llama-cli","llama-bench",
          "cargo","rustc","ollama","mlx-lm","mlx_lm","swift-frontend",
          "python model-generation/inference workloads"]}
    }' >"$launch_settings.tmp"
    mv "$launch_settings.tmp" "$launch_settings"
}

if [[ -n "$MODEL_ID" ]]; then
    write_request_manifest
    write_launch_settings
    launch_settings_sha=$(sha256_file "$launch_settings")
fi

case_group() {
    case "$1" in
        code-*) printf 'code\n' ;;
        repeat-*) printf 'repeat\n' ;;
        *) return 1 ;;
    esac
}

assert_no_model_runtime() {
    local name
    for name in hf2q llama-server llama-cli llama-bench; do
        if /usr/bin/pgrep -x "$name" >/dev/null 2>&1; then
            echo "matched run requires no existing $name process" >&2
            /usr/bin/pgrep -flx "$name" >&2 || true
            return 1
        fi
    done
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sed -n '2p' | rg -q .; then
        echo "matched run port already has a listener: $PORT" >&2
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2
        return 1
    fi
    require_no_foreign_heavy_work 0
}

require_ac_power() {
    pmset -g batt | rg -q "Now drawing from 'AC Power'" || {
        echo "matched run requires continuous AC power" >&2
        return 1
    }
}

power_mode_name=''
power_mode_code=''
read_live_power_mode_code() {
    pmset -g live | matched_parse_live_power_mode_code
}

initialize_power_mode_contract() {
    require_ac_power
    power_mode_name=$(LANG=C LC_ALL=C system_profiler SPPowerDataType \
      | matched_parse_ac_power_mode) || {
        echo "could not resolve the active AC Energy Mode" >&2
        return 1
    }
    [[ "$power_mode_name" != low ]] || {
        echo "matched performance gate rejects AC Low Power Mode; select Automatic or High Power Mode" >&2
        return 1
    }
    power_mode_code=$(read_live_power_mode_code) || {
        echo "could not read the live numeric power-mode canary" >&2
        return 1
    }
}

verify_power_mode_contract() {
    local explicit_mode live_code
    require_ac_power
    explicit_mode=$(LANG=C LC_ALL=C system_profiler SPPowerDataType \
      | matched_parse_ac_power_mode) || return 1
    live_code=$(read_live_power_mode_code) || return 1
    [[ "$explicit_mode" == "$power_mode_name" \
      && "$live_code" == "$power_mode_code" ]] || {
        echo "AC Energy Mode changed during the matched run: expected=$power_mode_name/$power_mode_code actual=$explicit_mode/$live_code" >&2
        return 1
    }
}

server_pid=''
monitor_pid=''
monitor_stop=''
caffeinate_pid=''
stop_server() {
    local waited=0
    if [[ -n "$monitor_stop" ]]; then
        : >"$monitor_stop"
    fi
    if [[ -n "$monitor_pid" ]]; then
        wait "$monitor_pid" 2>/dev/null || true
        monitor_pid=''
    fi
    monitor_stop=''
    [[ -n "$server_pid" ]] || return 0
    if kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid" 2>/dev/null || true
        while kill -0 "$server_pid" 2>/dev/null && ((waited < 60)); do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    if kill -0 "$server_pid" 2>/dev/null; then
        echo "server ignored bounded shutdown" >&2
        kill -TERM "$server_pid" 2>/dev/null || true
        waited=0
        while kill -0 "$server_pid" 2>/dev/null && ((waited < 30)); do
            sleep 1
            waited=$((waited + 1))
        done
    fi
    if kill -0 "$server_pid" 2>/dev/null; then
        echo "server ignored TERM; forcing bounded cleanup" >&2
        kill -KILL "$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
    server_pid=''
    if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | sed -n '2p' | rg -q .; then
        echo "server listener remained after cleanup: $PORT" >&2
        return 1
    fi
}

on_exit() {
    local original_rc=$?
    trap - EXIT
    stop_server || true
    if [[ -n "$caffeinate_pid" ]]; then
        kill -TERM "$caffeinate_pid" 2>/dev/null || true
        wait "$caffeinate_pid" 2>/dev/null || true
    fi
    thermal_cleanup_probe || true
    exit "$original_rc"
}
trap on_exit EXIT
trap 'exit 1' INT TERM

require_no_foreign_heavy_work() {
    local allowed_pid=$1
    local offenders
    local scripted_offenders
    offenders=$(/bin/ps -axo pid=,comm= | awk -v allowed="$allowed_pid" '
      {
        pid = $1
        $1 = ""
        sub(/^[[:space:]]+/, "", $0)
        name = tolower($0)
        sub(/^.*\//, "", name)
        if (pid != allowed && name ~ /^(hf2q|llama-server|llama-cli|llama-bench|cargo|rustc|ollama|mlx-lm|mlx_lm|swift-frontend)([0-9.-]|$)/) {
          print pid ":" name
        }
      }
    ')
    scripted_offenders=$(/bin/ps -axo pid=,command= \
      | matched_find_scripted_model_work "$allowed_pid")
    if [[ -n "$scripted_offenders" ]]; then
        offenders=${offenders:+$offenders$'\n'}$scripted_offenders
    fi
    [[ -z "$offenders" ]] || {
        echo "foreign calibrated-host work detected: $offenders" >&2
        return 1
    }
}

record_calibration_observation() {
    local thermal_log=$1
    local host_log=$2
    local phase=$3
    local sampled_at live_power_mode_code

    require_ac_power
    require_no_foreign_heavy_work "$server_pid"
    live_power_mode_code=$(read_live_power_mode_code)
    [[ "$live_power_mode_code" == "$power_mode_code" ]] || {
        echo "numeric power-mode canary changed during calibration" >&2
        return 1
    }
    thermal_read_state
    sampled_at=$(date +%s)
    [[ "$sampled_at" =~ ^[0-9]+$ ]]
    printf '%s\t%s\t%s\n' "$sampled_at" "$THERMAL_STATE" "$phase" \
      >>"$thermal_log"
    printf '%s\tac\tquiet\t%s\t%s\t%s\n' "$sampled_at" \
      "$power_mode_name" "$power_mode_code" "$phase" >>"$host_log"
}

wait_loaded_idle_calibration() {
    local trial_dir=$1
    local deadline=$((SECONDS + THERMAL_SETTLE_TIMEOUT_SECONDS))
    local nominal_since=-1
    local thermal_log="$trial_dir/thermal-settle.tsv"
    local host_log="$trial_dir/host-settle.tsv"
    : >"$thermal_log"
    : >"$host_log"
    while :; do
        record_calibration_observation "$thermal_log" "$host_log" loaded-idle
        if [[ "$THERMAL_STATE" == nominal ]]; then
            if ((nominal_since < 0)); then nominal_since=$SECONDS; fi
            if ((SECONDS - nominal_since >= THERMAL_SETTLE_SECONDS)); then
                thermal_validate_measurement_log "$thermal_log" \
                  "$((THERMAL_SAMPLE_SECONDS + 3))"
                matched_validate_host_observation_log "$host_log" 2 \
                  "$THERMAL_SETTLE_SECONDS" "$((THERMAL_SAMPLE_SECONDS + 3))"
                matched_validate_calibration_alignment "$thermal_log" "$host_log"
                return 0
            fi
        else
            nominal_since=-1
            : >"$thermal_log"
            : >"$host_log"
        fi
        if ((SECONDS >= deadline)); then
            echo "host did not remain nominal for ${THERMAL_SETTLE_SECONDS}s" >&2
            return 1
        fi
        sleep "$THERMAL_SAMPLE_SECONDS"
    done
}

monitor_measurement() {
    local trial_dir=$1
    local parent_pid=$2
    local stop_file=$3
    local thermal_log="$trial_dir/thermal-measurement.tsv"
    local host_log="$trial_dir/host-measurement.tsv"
    while [[ ! -e "$stop_file" ]]; do
        if ! record_calibration_observation \
          "$thermal_log" "$host_log" measurement \
          || [[ "$THERMAL_STATE" != nominal ]]; then
            echo "calibrated-host monitor failed" >"$trial_dir/calibration-failure.txt"
            kill -TERM "$parent_pid" 2>/dev/null || true
            return 1
        fi
        sleep "$THERMAL_SAMPLE_SECONDS"
    done
}

wait_ready() {
    local engine=$1
    local log=$2
    local endpoint=readyz
    local deadline=$((SECONDS + 600))
    [[ "$engine" == reference ]] && endpoint=health
    while ((SECONDS < deadline)); do
        if curl --fail --silent --show-error --max-time 2 \
          "http://127.0.0.1:$PORT/$endpoint" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "$engine exited before readiness" >&2
            sed -n '1,240p' "$log" >&2
            return 1
        fi
        sleep 1
    done
    echo "$engine did not become ready" >&2
    sed -n '1,240p' "$log" >&2
    return 1
}

resolve_loaded_model_id() {
    local engine=$1
    local trial_dir=$2
    local loaded_model_id
    local request
    curl --fail --silent --show-error \
      "http://127.0.0.1:$PORT/v1/models" >"$trial_dir/models.json"
    if [[ "$engine" == hf2q ]]; then
        loaded_model_id=$(matched_resolve_hf2q_model_id \
          "$trial_dir/models.json")
    else
        matched_validate_reference_model_alias \
          "$trial_dir/models.json" "$MODEL_ID" || {
            echo "reference did not advertise the exact loaded model alias: $MODEL_ID" >&2
            return 1
        }
        loaded_model_id=$MODEL_ID
    fi
    if [[ -z "$MODEL_ID" && "$engine" == hf2q ]]; then
        MODEL_ID=$loaded_model_id
        for request in "$OUT_DIR"/requests/*.json; do
            jq --arg model "$MODEL_ID" '.model = $model' "$request" \
              >"$request.tmp"
            mv "$request.tmp" "$request"
        done
        write_request_manifest
        write_launch_settings
        launch_settings_sha=$(sha256_file "$launch_settings")
    elif [[ "$loaded_model_id" != "$MODEL_ID" ]]; then
        echo "$engine loaded model identity drifted: expected=$MODEL_ID actual=$loaded_model_id" >&2
        return 1
    fi
    verify_request_manifest
    [[ -n "$launch_settings_sha" \
      && "$(sha256_file "$launch_settings")" == "$launch_settings_sha" ]] || {
        echo "matched launch settings changed during the run" >&2
        return 1
    }
}

launch_server() {
    local engine=$1
    local log=$2
    local -a clean_env=(
      env -i
      "HOME=${HOME:?HOME is required}"
      "PATH=$PATH"
      "TMPDIR=${TMPDIR:-/tmp}"
      "LANG=C"
      "LC_ALL=C"
      "USER=${USER:-}"
      "LOGNAME=${LOGNAME:-}"
    )
    verify_executable_identity hf2q "$HF2Q_BIN" "$HF2Q_SHA256" \
      "$hf2q_binary_snapshot"
    verify_executable_identity reference "$REFERENCE_BIN" "$REFERENCE_SHA256" \
      "$reference_binary_snapshot"
    hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
      "$model_verification_receipt"
    if [[ "$engine" == hf2q ]]; then
        "${clean_env[@]}" HF2Q_BIN="$HF2Q_BIN" MODEL="$MODEL_PATH" PORT="$PORT" \
          MAX_SLOTS=1 KV_CACHE_BUDGET_BYTES=51539607552 \
          QWEN38_VISION=off QWEN38_SPECULATION=auto \
          THINKING_TOKEN_BUDGET=0 TOOL_THINKING_TOKEN_BUDGET=0 \
          REP_PENALTY=1.05 HF2Q_DECODE_MVN=0 HF2Q_DECODE_MV_EXT=1 \
          "$SCRIPT_DIR/serve_qwen38_opencode.sh" >"$log" 2>&1 &
    else
        "${clean_env[@]}" "$REFERENCE_BIN" --model "$MODEL_PATH" --alias "$MODEL_ID" \
          --host 127.0.0.1 --port "$PORT" --parallel 1 --ctx-size 262144 \
          --batch-size 2048 --ubatch-size 512 --gpu-layers all \
          --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
          --jinja --reasoning off --metrics --spec-type draft-mtp \
          --spec-draft-n-max 3 --spec-draft-n-min 0 \
          --spec-draft-p-min 0 --spec-draft-backend-sampling \
          --gpu-layers-draft all --temp 0 --repeat-penalty 1.05 \
          >"$log" 2>&1 &
    fi
    server_pid=$!
}

rows_file="$OUT_DIR/measurements.jsonl"
: >"$rows_file"
ttft_rows_file="$OUT_DIR/ttft.jsonl"
: >"$ttft_rows_file"
warmup_rows_file="$OUT_DIR/sustained-warmup.jsonl"
: >"$warmup_rows_file"
thermal_prepare_probe
thermal_probe_source_sha=$(sha256_file "$THERMAL_PROBE_SOURCE")
thermal_probe_compiler_sha=$(sha256_file "$THERMAL_PROBE_COMPILER")
thermal_probe_binary_sha=$(sha256_file "$THERMAL_PROBE_BIN")
thermal_probe_compiler_version=$THERMAL_PROBE_COMPILER_VERSION
initialize_power_mode_contract
assert_no_model_runtime
caffeinate -dimsu -w "$$" &
caffeinate_pid=$!

run_stream_ttft() {
    local engine=$1
    local trial=$2
    local trial_dir=$3
    local stream_path="$trial_dir/ttft.sse"
    local receipt_path="$trial_dir/ttft.json"
    local started_at

    started_at=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
      -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
    curl --fail-with-body --silent --show-error --no-buffer \
      --connect-timeout 5 --max-time 600 \
      --header 'Content-Type: application/json' \
      --data-binary "@$OUT_DIR/requests/ttft.json" \
      "http://127.0.0.1:$PORT/v1/chat/completions" \
      | matched_parse_sse_stream "$started_at" "$stream_path" \
          "$receipt_path" "$(<"$OUT_DIR/expected/ttft.txt")"
    jq -e --arg expected "$(<"$OUT_DIR/expected/ttft.txt")" '
      .schema == 1
      and (.first_semantic_ms | type == "number" and . > 0)
      and .content == $expected
      and .done_count == 1
      and (.event_count | type == "number" and . > 0)
    ' "$receipt_path" >/dev/null
    jq -c --arg engine "$engine" --argjson trial "$trial" \
      '. + {engine:$engine,trial:$trial}' "$receipt_path" >>"$ttft_rows_file"
}

run_trial() {
    local index=$1
    local engine=$2
    local trial_dir="$OUT_DIR/trials/trial-$index-$engine"
    local log="$trial_dir/server.log"
    local measurement_log="$trial_dir/thermal-measurement.tsv"
    local host_measurement_log="$trial_dir/host-measurement.tsv"
    local name group response elapsed content source_path
    local warmup_finished_at measurement_started_at warmup_delay
    mkdir -p "$trial_dir"
    assert_no_model_runtime
    launch_server "$engine" "$log"
    wait_ready "$engine" "$log"
    resolve_loaded_model_id "$engine" "$trial_dir"
    wait_loaded_idle_calibration "$trial_dir"

    monitor_stop="$trial_dir/monitor.stop"
    : >"$measurement_log"
    : >"$host_measurement_log"
    record_calibration_observation "$measurement_log" "$host_measurement_log" \
      measurement-start
    [[ "$THERMAL_STATE" == nominal ]]
    monitor_measurement "$trial_dir" "$$" "$monitor_stop" &
    monitor_pid=$!

    curl --fail-with-body --silent --show-error --max-time 600 \
      --header 'Content-Type: application/json' \
      --data-binary "@$OUT_DIR/requests/warmup.json" \
      "http://127.0.0.1:$PORT/v1/chat/completions" \
      >"$trial_dir/warmup.json"
    cmp <(jq -j '.choices[0].message.content' "$trial_dir/warmup.json") \
      "$OUT_DIR/expected/warmup.txt"
    run_stream_ttft "$engine" "$index" "$trial_dir"
    for name in $SUSTAINED_WARMUP_CASES; do
        response="$trial_dir/$name.json"
        elapsed=$(curl --fail-with-body --silent --show-error --max-time 600 \
          --header 'Content-Type: application/json' \
          --data-binary "@$OUT_DIR/requests/$name.json" \
          --output "$response" --write-out '%{time_total}' \
          "http://127.0.0.1:$PORT/v1/chat/completions")
        matched_validate_common_response "$response"
        cmp <(jq -j '.choices[0].message.content' "$response") \
          "$OUT_DIR/expected/$name.txt"
        [[ "$(jq -er '.choices[0].finish_reason' "$response")" == stop ]]
        jq -cn --arg engine "$engine" --arg name "$name" \
          --argjson trial "$index" --argjson wall_seconds "$elapsed" \
          --arg content_sha256 "$(jq -j '.choices[0].message.content' \
            "$response" | shasum -a 256 | awk '{print $1}')" \
          --slurpfile response "$response" '{engine:$engine,trial:$trial,
            name:$name,wall_seconds:$wall_seconds,
            prompt_tokens:$response[0].usage.prompt_tokens,
            completion_tokens:$response[0].usage.completion_tokens,
            finish_reason:$response[0].choices[0].finish_reason,
            content_sha256:$content_sha256}' >>"$warmup_rows_file"
    done
    warmup_finished_at=$(perl -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
      -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
    curl --fail --silent --show-error --max-time 10 \
      "http://127.0.0.1:$PORT/metrics" >"$trial_dir/metrics-before.txt"

    for name in $CASES; do
        if [[ "$name" == code-a ]]; then
            measurement_started_at=$(perl \
              -MTime::HiRes=clock_gettime,CLOCK_MONOTONIC \
              -e 'printf "%.9f", clock_gettime(CLOCK_MONOTONIC)')
            warmup_delay=$(awk -v started="$measurement_started_at" \
              -v finished="$warmup_finished_at" \
              'BEGIN { printf "%.9f", started - finished }')
            awk -v delay="$warmup_delay" \
              -v maximum="$MAX_WARMUP_TO_MEASUREMENT_SECONDS" \
              'BEGIN { exit !(delay >= 0 && delay <= maximum) }'
            jq -s --arg engine "$engine" --argjson trial "$index" \
              --argjson delay "$warmup_delay" \
              --argjson minimum_tokens "$MIN_SUSTAINED_WARMUP_TOKENS" \
              --argjson maximum_delay "$MAX_WARMUP_TO_MEASUREMENT_SECONDS" '
              [.[] | select(.engine == $engine and .trial == $trial)] as $rows
              | {schema:1,engine:$engine,trial:$trial,samples:($rows | length),
                  completion_tokens:([$rows[].completion_tokens] | add),
                  exact_content:all($rows[];
                    (.content_sha256 | test("^[0-9a-f]{64}$"))),
                  natural_stop:all($rows[]; .finish_reason == "stop"),
                  warmup_to_measurement_seconds:$delay,
                  minimum_completion_tokens:$minimum_tokens,
                  maximum_warmup_to_measurement_seconds:$maximum_delay}
              | . + {pass:(.samples == 3
                  and .completion_tokens >= .minimum_completion_tokens
                  and .exact_content and .natural_stop
                  and .warmup_to_measurement_seconds
                    <= .maximum_warmup_to_measurement_seconds)}
            ' "$warmup_rows_file" >"$trial_dir/sustained-warmup.json"
            jq -e '.pass == true' "$trial_dir/sustained-warmup.json" >/dev/null
        fi
        response="$trial_dir/$name.json"
        elapsed=$(curl --fail-with-body --silent --show-error --max-time 600 \
          --header 'Content-Type: application/json' \
          --data-binary "@$OUT_DIR/requests/$name.json" \
          --output "$response" --write-out '%{time_total}' \
          "http://127.0.0.1:$PORT/v1/chat/completions")
        matched_validate_common_response "$response"
        if [[ "$engine" == hf2q ]]; then
            matched_validate_hf2q_telemetry "$response"
        else
            matched_validate_reference_telemetry "$response"
        fi
        if [[ "$name" == repeat-* ]]; then
            cmp <(jq -j '.choices[0].message.content' "$response") \
              "$OUT_DIR/expected/$name.txt"
            [[ "$(jq -er '.choices[0].finish_reason' "$response")" == stop ]]
            jq -n --arg case "$name" \
              --arg content_sha256 "$(jq -j '.choices[0].message.content' \
                "$response" | shasum -a 256 | awk '{print $1}')" \
              '{schema:1,case:$case,exact_expected_content:true,
                natural_stop:true,content_sha256:$content_sha256}' \
              >"$trial_dir/$name-quality.json"
        else
            content=$(jq -r '.choices[0].message.content' "$response")
            case "$name" in
                code-a) rg -q 'fibonacci' <<<"$content" ;;
                code-b) rg -q 'binary_search' <<<"$content" ;;
                code-c) rg -q 'fn gcd|gcd\(' <<<"$content" ;;
            esac
            mkdir -p "$trial_dir/code-validation"
            source_path="$trial_dir/code-validation/$name.rs"
            matched_extract_rust_source "$response" "$source_path"
        fi
        jq -S -c '{role:.choices[0].message.role,
          content:.choices[0].message.content,
          tool_calls:(.choices[0].message.tool_calls // null),
          finish_reason:.choices[0].finish_reason,
          prompt_tokens:.usage.prompt_tokens,
          completion_tokens:.usage.completion_tokens}' \
          "$response" >"$trial_dir/$name.semantic.json"
        group=$(case_group "$name")
        jq -n --argjson trial "$index" --arg engine "$engine" \
          --arg name "$name" --arg group "$group" \
          --argjson wall_seconds "$elapsed" --slurpfile response "$response" '{
            trial:$trial,engine:$engine,name:$name,group:$group,
            wall_seconds:$wall_seconds,
            prompt_tokens:($response[0].usage.prompt_tokens // -1),
            cached_tokens:(if $engine == "hf2q"
              then $response[0].usage.prompt_tokens_details.cached_tokens
              else $response[0].timings.cache_n end),
            processed_prompt_tokens:(if $engine == "hf2q"
              then ($response[0].usage.prompt_tokens
                - $response[0].usage.prompt_tokens_details.cached_tokens)
              else $response[0].timings.prompt_n end),
            completion_tokens:($response[0].usage.completion_tokens // -1),
            finish_reason:($response[0].choices[0].finish_reason // null),
            internal_prefill_seconds:(if $engine == "hf2q"
              then $response[0].x_hf2q_timing.prefill_time_secs
              else ($response[0].timings.prompt_ms / 1000) end),
            internal_decode_seconds:(if $engine == "hf2q"
              then $response[0].x_hf2q_timing.decode_time_secs
              else ($response[0].timings.predicted_ms / 1000) end),
            internal_prefill_tps:(if $engine == "hf2q"
              then $response[0].x_hf2q_timing.prefill_tokens_per_sec
              else $response[0].timings.prompt_per_second end),
            internal_decode_tps:(if $engine == "hf2q"
              then $response[0].x_hf2q_timing.decode_tokens_per_sec
              else $response[0].timings.predicted_per_second end),
            response_ttft_ms:(if $engine == "hf2q"
              then $response[0].x_hf2q_timing.time_to_first_token_ms
              else null end),
            drafted_tokens:($response[0].timings.draft_n // null),
            accepted_draft_tokens:($response[0].timings.draft_n_accepted // null)
          }' >>"$rows_file"
    done

    kill -0 "$server_pid" 2>/dev/null || {
        echo "$engine server exited before measurement completed" >&2
        return 1
    }
    curl --fail --silent --show-error --max-time 10 \
      "http://127.0.0.1:$PORT/metrics" >"$trial_dir/metrics-after.txt"
    qwen36_reject_fatal_log "$log"
    : >"$monitor_stop"
    wait "$monitor_pid"
    monitor_pid=''
    monitor_stop=''
    record_calibration_observation "$measurement_log" "$host_measurement_log" \
      measurement-end
    [[ "$THERMAL_STATE" == nominal ]]
    thermal_validate_measurement_log "$measurement_log" \
      "$((THERMAL_SAMPLE_SECONDS + 3))"
    matched_validate_host_observation_log "$host_measurement_log" 3 1 \
      "$((THERMAL_SAMPLE_SECONDS + 3))"
    matched_validate_calibration_alignment "$measurement_log" \
      "$host_measurement_log"
    [[ ! -e "$trial_dir/calibration-failure.txt" ]]
    stop_server
    verify_power_mode_contract
    qwen36_reject_fatal_log "$log"
    for name in code-a code-b code-c; do
        matched_validate_rust_case "$name" \
          "$trial_dir/code-validation/$name.rs" \
          "$trial_dir/code-validation"
    done
}

trial=0
for engine in $TRIAL_ORDER; do
    trial=$((trial + 1))
    run_trial "$trial" "$engine"
done

warmup_diagnostics=$(jq -s \
  --argjson minimum_tokens "$MIN_SUSTAINED_WARMUP_TOKENS" '
  sort_by(.engine, .name) as $rows
  | ($rows | group_by(.engine, .name)
    | map({engine:.[0].engine,name:.[0].name,samples:length,
        trials:(map(.trial) | sort),
        completion_token_variants:(map(.completion_tokens) | unique | length),
        content_sha256_variants:(map(.content_sha256) | unique | length)})) as $cases
  | ($rows | sort_by(.engine, .trial) | group_by(.engine, .trial)
    | map({engine:.[0].engine,trial:.[0].trial,samples:length,
        completion_tokens:(map(.completion_tokens) | add)})) as $trials
  | {schema:1,samples:($rows | length),cases:$cases,trials:$trials,
      minimum_completion_tokens_per_trial:$minimum_tokens,
      pass:(($rows | length) == 12
        and ($cases | length) == 6 and ($trials | length) == 4
        and all($cases[];
          .samples == 2
          and (.trials == (if .engine == "hf2q" then [1,4] else [2,3] end))
          and .completion_token_variants == 1
          and .content_sha256_variants == 1)
        and all($trials[];
          .samples == 3 and .completion_tokens >= $minimum_tokens))}
' "$warmup_rows_file")
jq -e '.pass == true' <<<"$warmup_diagnostics" >/dev/null

calibration_manifest="$OUT_DIR/calibration.sha256"
: >"$calibration_manifest"
while IFS= read -r calibration_log; do
    printf '%s  %s\n' "$(sha256_file "$calibration_log")" \
      "${calibration_log#"$OUT_DIR"/}" >>"$calibration_manifest"
done < <(find "$OUT_DIR/trials" -type f \
  \( -name 'thermal-settle.tsv' -o -name 'thermal-measurement.tsv' \
    -o -name 'host-settle.tsv' -o -name 'host-measurement.tsv' \) \
  -print | sort)
[[ "$(awk 'END { print NR }' "$calibration_manifest")" == 16 ]] || {
    echo "calibration manifest must bind four logs for each of four trials" >&2
    exit 1
}
(cd "$OUT_DIR" && shasum -a 256 -c "$(basename "$calibration_manifest")" \
  >/dev/null)
calibration_manifest_sha=$(sha256_file "$calibration_manifest")
verify_request_manifest

for name in $CASES; do
    quality_receipts=$(find "$OUT_DIR/trials" -path \
      "*/$name-quality.json" -type f | awk 'END { print NR }')
    [[ "$quality_receipts" == 4 ]] || {
        echo "expected four quality receipts for $name" >&2
        exit 1
    }
    cmp "$OUT_DIR/trials/trial-1-hf2q/$name.semantic.json" \
      "$OUT_DIR/trials/trial-4-hf2q/$name.semantic.json"
    cmp "$OUT_DIR/trials/trial-2-reference/$name.semantic.json" \
      "$OUT_DIR/trials/trial-3-reference/$name.semantic.json"
    if [[ "$name" == repeat-* ]]; then
        baseline="$OUT_DIR/trials/trial-1-hf2q/$name.semantic.json"
        for candidate in "$OUT_DIR"/trials/trial-*/"$name.semantic.json"; do
            cmp <(jq -S -c '{role,content,tool_calls,finish_reason,prompt_tokens}' \
              "$baseline") \
              <(jq -S -c '{role,content,tool_calls,finish_reason,prompt_tokens}' \
              "$candidate")
        done
    fi
    prompt_token_variants=$(jq -r --arg name "$name" \
      'select(.name == $name) | .prompt_tokens' "$rows_file" | sort -u | awk 'END { print NR }')
    [[ "$prompt_token_variants" == 1 ]] || {
        echo "prompt token count differs across engines for $name" >&2
        exit 1
    }
done
code_quality_receipts=$(find "$OUT_DIR/trials" -type f \
  -name 'code-*-quality.json' | awk 'END { print NR }')
repeat_quality_receipts=$(find "$OUT_DIR/trials" -type f \
  -name 'repeat-*-quality.json' | awk 'END { print NR }')
[[ "$code_quality_receipts" == 12 && "$repeat_quality_receipts" == 12 ]] || {
    echo "quality receipt cardinality mismatch: code=$code_quality_receipts repeat=$repeat_quality_receipts" >&2
    exit 1
}

metric_total() {
    local metric=$1
    local path=$2
    awk -v metric="$metric" '
      $1 ~ ("^" metric "\\{proposer=\\\"(history_lookup|mtp)\\\"\\}$") { sum += $2 }
      END { print sum + 0 }
    ' "$path"
}

hf2q_proposals=0
hf2q_accepted=0
for trial_dir in "$OUT_DIR"/trials/trial-*-hf2q; do
    proposals_before=$(metric_total hf2q_qwen_speculation_proposals_total \
      "$trial_dir/metrics-before.txt")
    proposals_after=$(metric_total hf2q_qwen_speculation_proposals_total \
      "$trial_dir/metrics-after.txt")
    accepted_before=$(metric_total hf2q_qwen_speculation_accepted_tokens_total \
      "$trial_dir/metrics-before.txt")
    accepted_after=$(metric_total hf2q_qwen_speculation_accepted_tokens_total \
      "$trial_dir/metrics-after.txt")
    proposals_delta=$((proposals_after - proposals_before))
    accepted_delta=$((accepted_after - accepted_before))
    ((proposals_delta > 0 && accepted_delta > 0)) || {
        echo "measured hf2q requests did not prove accepted speculation in $trial_dir" >&2
        exit 1
    }
    hf2q_proposals=$((hf2q_proposals + proposals_delta))
    hf2q_accepted=$((hf2q_accepted + accepted_delta))
done
reference_drafted=0
reference_accepted=0
for reference_trial in 2 3; do
    if ! IFS=$'\t' read -r trial_drafted trial_accepted \
      < <(matched_reference_speculation_totals "$rows_file" \
        "$reference_trial"); then
        echo "reference trial $reference_trial did not prove accepted speculation" >&2
        exit 1
    fi
    reference_drafted=$((reference_drafted + trial_drafted))
    reference_accepted=$((reference_accepted + trial_accepted))
done
((hf2q_proposals > 0 && hf2q_accepted > 0 \
  && reference_drafted > 0 && reference_accepted > 0)) || {
    echo "both engines must prove active, accepted speculation" >&2
    exit 1
}

stability_json="$OUT_DIR/stability.json"
matched_measurement_stability_json "$rows_file" \
  "$MAX_WITHIN_ENGINE_GROUP_SPREAD_PERCENT" \
  "$MAX_WITHIN_ENGINE_CASE_SPREAD_PERCENT" >"$stability_json"
jq -e '.stable == true' "$stability_json" >/dev/null || {
    echo "matched performance calibration is unstable; no speed verdict is valid" >&2
    jq --argjson maximum_group_spread_percent \
      "$MAX_WITHIN_ENGINE_GROUP_SPREAD_PERCENT" \
      --argjson maximum_case_spread_percent \
      "$MAX_WITHIN_ENGINE_CASE_SPREAD_PERCENT" \
      '{maximum_group_spread_percent,maximum_case_spread_percent,
      cases:[.cases[] | select(
        .wall_spread_percent > $maximum_case_spread_percent or
        .decode_tps_spread_percent > $maximum_case_spread_percent)],
      groups:[.groups[] | select(
        .wall_spread_percent > $maximum_group_spread_percent or
        .decode_tps_spread_percent > $maximum_group_spread_percent)]}' \
      "$stability_json" >&2
    exit 1
}
jq -e '.observed_band_dominance == true' "$stability_json" >/dev/null || {
    echo "matched speed bands overlap; result is inconclusive, not parity" >&2
    jq '.comparisons' "$stability_json" >&2
    exit 1
}

group_seconds() {
    local engine=$1
    local group=$2
    jq -er --arg engine "$engine" --arg group "$group" '
      .groups[] | select(.engine == $engine and .group == $group)
      | .median_trial_total_seconds
    ' "$stability_json"
}

hf2q_code=$(group_seconds hf2q code)
reference_code=$(group_seconds reference code)
hf2q_repeat=$(group_seconds hf2q repeat)
reference_repeat=$(group_seconds reference repeat)
code_ratio=$(awk -v reference="$reference_code" -v hf2q="$hf2q_code" \
  'BEGIN { printf "%.6f", reference / hf2q }')
repeat_ratio=$(awk -v reference="$reference_repeat" -v hf2q="$hf2q_repeat" \
  'BEGIN { printf "%.6f", reference / hf2q }')

for ratio in "$code_ratio" "$repeat_ratio"; do
    awk -v actual="$ratio" -v minimum="$MIN_HF2Q_RATIO" \
      'BEGIN { exit !(actual >= minimum) }' || {
        echo "hf2q ratio $ratio is below required $MIN_HF2Q_RATIO" >&2
        exit 1
      }
done

runtime_diagnostics=$(jq -s '
  def median:
    sort as $values
    | ($values | length) as $n
    | if $n == 0 then error("empty diagnostic sample")
      elif ($n % 2) == 1 then $values[($n / 2 | floor)]
      else (($values[$n / 2 - 1] + $values[$n / 2]) / 2)
      end;
  def runtime($engine):
    [.[] | select(.engine == $engine)] as $rows
    | {
        samples:($rows | length),
        prompt_tokens:[$rows[].prompt_tokens],
        cached_tokens:[$rows[].cached_tokens],
        processed_prompt_tokens:[$rows[].processed_prompt_tokens],
        completion_tokens:[$rows[].completion_tokens],
        internal_prefill_tps:[$rows[].internal_prefill_tps],
        internal_decode_tps:[$rows[].internal_decode_tps],
        median_internal_prefill_tps:([$rows[].internal_prefill_tps] | median),
        median_internal_decode_tps:([$rows[].internal_decode_tps] | median),
        wall_seconds:[$rows[].wall_seconds]
      }
    | if $engine == "hf2q" then . + {
        response_ttft_ms:[$rows[].response_ttft_ms],
        median_response_ttft_ms:([$rows[].response_ttft_ms] | median)
      } else . end;
  {hf2q:runtime("hf2q"),reference:runtime("reference")}
' "$rows_file")
jq -e '
  .hf2q.samples == 12 and .reference.samples == 12
  and all(.hf2q.cached_tokens[], .reference.cached_tokens[];
    type == "number" and floor == . and . >= 0)
  and all(.hf2q.internal_prefill_tps[], .reference.internal_prefill_tps[],
    .hf2q.internal_decode_tps[], .reference.internal_decode_tps[];
    type == "number" and . > 0)
' <<<"$runtime_diagnostics" >/dev/null

ttft_diagnostics=$(jq -s '
  def median:
    sort as $values
    | ($values | length) as $n
    | if $n == 0 then error("empty TTFT sample")
      elif ($n % 2) == 1 then $values[($n / 2 | floor)]
      else (($values[$n / 2 - 1] + $values[$n / 2]) / 2)
      end;
  def runtime($engine):
    [.[] | select(.engine == $engine)] as $rows
    | {samples:($rows | length),values_ms:[$rows[].first_semantic_ms],
       median_ms:([$rows[].first_semantic_ms] | median)};
  {hf2q:runtime("hf2q"),reference:runtime("reference")}
' "$ttft_rows_file")
jq -e '
  .hf2q.samples == 2 and .reference.samples == 2
  and all(.hf2q.values_ms[], .reference.values_ms[];
    type == "number" and . > 0)
' <<<"$ttft_diagnostics" >/dev/null

verify_executable_identity hf2q "$HF2Q_BIN" "$HF2Q_SHA256" \
  "$hf2q_binary_snapshot"
verify_executable_identity reference "$REFERENCE_BIN" "$REFERENCE_SHA256" \
  "$reference_binary_snapshot"
final_reference_version=$("$REFERENCE_BIN" --version 2>&1)
[[ "$final_reference_version" == "$reference_version" ]]
rustc_version=$(rustc --version)
hf2q_binary_sha=$(sha256_file "$HF2Q_BIN")
hf2q_lock_sha=$(sha256_file "$ROOT_DIR/Cargo.lock")
hf2q_release_verify_model "$MODEL_PATH" "$MODEL_SHA256" \
  "$model_verification_receipt"
require_clean_exact_source "$ROOT_DIR" "$HF2Q_COMMIT" hf2q
require_clean_exact_source "$REFERENCE_SOURCE_DIR" "$REFERENCE_COMMIT" reference
verify_request_manifest
[[ "$(sha256_file "$launch_settings")" == "$launch_settings_sha" ]]

script_sha=$(sha256_file "$ROOT_DIR/scripts/qwen38_matched_reference_abba.sh")
contract_sha=$(sha256_file \
  "$ROOT_DIR/scripts/qwen38_matched_reference_contract.sh")
request_manifest_sha=$(sha256_file "$request_manifest")
measurements_sha=$(sha256_file "$rows_file")
ttft_rows_sha=$(sha256_file "$ttft_rows_file")
evidence_manifest="$OUT_DIR/evidence.sha256"
: >"$evidence_manifest.tmp"
while IFS= read -r evidence_path; do
    printf '%s  %s\n' "$(sha256_file "$OUT_DIR/$evidence_path")" \
      "$evidence_path" >>"$evidence_manifest.tmp"
done < <(cd "$OUT_DIR" && find . -type f \
  ! -name 'evidence.sha256' ! -name 'evidence.sha256.tmp' \
  ! -name 'summary.json' ! -name 'summary.json.tmp' \
  ! -name 'result.sha256' -print | sed 's#^./##' | sort)
mv "$evidence_manifest.tmp" "$evidence_manifest"
(cd "$OUT_DIR" && shasum -a 256 -c "$(basename "$evidence_manifest")" \
  >/dev/null)
evidence_manifest_sha=$(sha256_file "$evidence_manifest")

jq -n --arg verdict pass --arg trial_order "$TRIAL_ORDER" \
  --arg hf2q_commit "$HF2Q_COMMIT" --arg hf2q_binary_sha "$hf2q_binary_sha" \
  --arg hf2q_binary_snapshot "$hf2q_binary_snapshot" \
  --arg hf2q_lock_sha "$hf2q_lock_sha" \
  --arg mlx_native_version "$EXPECTED_MLX_NATIVE_VERSION" \
  --arg mlx_native_checksum "$EXPECTED_MLX_NATIVE_CHECKSUM" \
  --arg reference_commit "$REFERENCE_COMMIT" \
  --arg reference_sha "$REFERENCE_SHA256" \
  --arg reference_binary_snapshot "$reference_binary_snapshot" \
  --arg rustc_version "$rustc_version" \
  --arg model_id "$MODEL_ID" --arg model_path "$MODEL_PATH" \
  --arg model_repository "$QUALIFIED_MODEL_REPOSITORY" \
  --arg model_revision "$QUALIFIED_MODEL_REVISION" \
  --arg model_file "$QUALIFIED_MODEL_FILE" \
  --arg model_sha "$MODEL_SHA256" \
  --arg model_verification "$model_verification_mode" \
  --arg model_file_snapshot "$model_file_snapshot" \
  --arg hardware_model "$hardware_model" --arg hardware_chip "$hardware_chip" \
  --arg hardware_arch "$hardware_arch" --arg hardware_os "$hardware_os" \
  --arg power_mode_name "$power_mode_name" \
  --argjson power_mode_code "$power_mode_code" \
  --arg thermal_probe_source_sha "$thermal_probe_source_sha" \
  --arg thermal_probe_compiler_sha "$thermal_probe_compiler_sha" \
  --arg thermal_probe_compiler_version "$thermal_probe_compiler_version" \
  --arg thermal_probe_binary_sha "$thermal_probe_binary_sha" \
  --arg script_sha "$script_sha" \
  --arg contract_sha "$contract_sha" \
  --arg request_manifest_sha "$request_manifest_sha" \
  --arg measurements_sha "$measurements_sha" \
  --arg ttft_rows_sha "$ttft_rows_sha" \
  --arg launch_settings_sha "$launch_settings_sha" \
  --arg calibration_manifest_sha "$calibration_manifest_sha" \
  --arg evidence_manifest_sha "$evidence_manifest_sha" \
  --argjson model_bytes "$MODEL_BYTES" --argjson max_tokens "$MAX_TOKENS" \
  --argjson hardware_memory_bytes "$hardware_memory_bytes" \
  --argjson maximum_group_spread "$MAX_WITHIN_ENGINE_GROUP_SPREAD_PERCENT" \
  --argjson maximum_case_spread "$MAX_WITHIN_ENGINE_CASE_SPREAD_PERCENT" \
  --argjson runtime_diagnostics "$runtime_diagnostics" \
  --argjson ttft_diagnostics "$ttft_diagnostics" \
  --argjson warmup_diagnostics "$warmup_diagnostics" \
  --argjson hf2q_code_seconds "$hf2q_code" \
  --argjson reference_code_seconds "$reference_code" \
  --argjson code_ratio "$code_ratio" \
  --argjson hf2q_repeat_seconds "$hf2q_repeat" \
  --argjson reference_repeat_seconds "$reference_repeat" \
  --argjson repeat_ratio "$repeat_ratio" \
  --argjson minimum_ratio "$MIN_HF2Q_RATIO" \
  --argjson hf2q_proposals "$hf2q_proposals" \
  --argjson hf2q_accepted "$hf2q_accepted" \
  --argjson reference_drafted "$reference_drafted" \
  --argjson reference_accepted "$reference_accepted" \
  --argjson code_quality_receipts "$code_quality_receipts" \
  --argjson repeat_quality_receipts "$repeat_quality_receipts" \
  --slurpfile launch_settings "$launch_settings" \
  --slurpfile stability "$stability_json" '{
    schema:4,verdict:$verdict,
    workload:{trial_order:$trial_order,cases_per_group_per_engine:6,
      max_tokens:$max_tokens,temperature:0,repetition_penalty:1.05,seed:null,
      chat_template_kwargs:{enable_thinking:false},
      quality_scope:"complete Rust compilation plus evaluator tests; exact repeat transcription"},
    quality:{
      code:{complete_rust:true,compiled:true,model_unit_test_present:true,
        evaluator_tests_passed:true,
        receipts:$code_quality_receipts,rustc_version:$rustc_version},
      repeat:{exact_expected_content:true,natural_stop:true,
        exact_across_engines:true,receipts:$repeat_quality_receipts}},
    launch_settings:$launch_settings[0],
    hf2q:{commit:$hf2q_commit,binary_sha256:$hf2q_binary_sha,
      binary_file_snapshot:$hf2q_binary_snapshot,
      source_binding:"embedded exact commit plus clean exact worktree",
      cargo_lock_sha256:$hf2q_lock_sha,
      mlx_native:{version:$mlx_native_version,checksum:$mlx_native_checksum},
      speculation:"shipping-auto-history-then-mtp",
      proposals:$hf2q_proposals,accepted_draft_tokens:$hf2q_accepted},
    reference:{commit:$reference_commit,binary_sha256:$reference_sha,
      binary_file_snapshot:$reference_binary_snapshot,
      source_policy:"caller-bound-clean-current-head",
      speculation:"fixed-k3-mtp",drafted_tokens:$reference_drafted,
      accepted_draft_tokens:$reference_accepted,
      kv_cache:"q8_0-k-and-v"},
    model:{id:$model_id,path:$model_path,repository:$model_repository,
      revision:$model_revision,file:$model_file,sha256:$model_sha,
      bytes:$model_bytes,verification:$model_verification,
      file_snapshot:$model_file_snapshot},
    hardware:{model:$hardware_model,chip:$hardware_chip,arch:$hardware_arch,
      memory_bytes:$hardware_memory_bytes,os_version:$hardware_os,
      power_mode:{name:$power_mode_name,numeric_canary:$power_mode_code},
      thermal_probe:{source_sha256:$thermal_probe_source_sha,
        compiler_sha256:$thermal_probe_compiler_sha,
        compiler_version:$thermal_probe_compiler_version,
        binary_sha256:$thermal_probe_binary_sha}},
    diagnostics:$runtime_diagnostics,
    streamed_semantic_ttft:$ttft_diagnostics,
    sustained_warmup:$warmup_diagnostics,
    code:{hf2q_median_trial_total_seconds:$hf2q_code_seconds,
      reference_median_trial_total_seconds:$reference_code_seconds,
      hf2q_over_reference:$code_ratio},
    repeat:{hf2q_median_trial_total_seconds:$hf2q_repeat_seconds,
      reference_median_trial_total_seconds:$reference_repeat_seconds,
      hf2q_over_reference:$repeat_ratio},
    acceptance:{minimum_hf2q_ratio:$minimum_ratio,
      maximum_within_engine_group_spread_percent:$maximum_group_spread,
      maximum_within_engine_case_spread_percent:$maximum_case_spread},
    stability:$stability[0],
    calibration:{required_state:"nominal",required_power:"ac",
      required_energy_mode:"automatic-or-high",
      required_process_state:"quiet",settle_seconds:30,sample_seconds:2,
      trial_logs:16,manifest_sha256:$calibration_manifest_sha},
    evidence:{script_sha256:$script_sha,contract_sha256:$contract_sha,
      request_manifest_sha256:$request_manifest_sha,
      launch_settings_sha256:$launch_settings_sha,
      measurements_sha256:$measurements_sha,
      ttft_rows_sha256:$ttft_rows_sha,
      evidence_manifest_sha256:$evidence_manifest_sha}
  }' >"$OUT_DIR/summary.json.tmp"
jq -e '
  .schema == 4 and .verdict == "pass"
  and .quality.code.complete_rust == true
  and .quality.code.compiled == true
  and .quality.code.model_unit_test_present == true
  and .quality.code.evaluator_tests_passed == true
  and .quality.code.receipts == 12
  and .quality.repeat.exact_expected_content == true
  and .quality.repeat.natural_stop == true
  and .quality.repeat.exact_across_engines == true
  and .quality.repeat.receipts == 12
  and .code.hf2q_over_reference >= .acceptance.minimum_hf2q_ratio
  and .repeat.hf2q_over_reference >= .acceptance.minimum_hf2q_ratio
  and .stability.stable == true
  and .stability.observed_band_dominance == true
  and .hf2q.proposals > 0 and .hf2q.accepted_draft_tokens > 0
  and .reference.drafted_tokens > 0 and .reference.accepted_draft_tokens > 0
  and .streamed_semantic_ttft.hf2q.samples == 2
  and .streamed_semantic_ttft.reference.samples == 2
  and .sustained_warmup.pass == true
  and .calibration.trial_logs == 16
  and (.evidence.evidence_manifest_sha256 | test("^[0-9a-f]{64}$"))
' "$OUT_DIR/summary.json.tmp" >/dev/null
matched_publish_result "$OUT_DIR/summary.json.tmp" "$OUT_DIR/summary.json" \
  "$evidence_manifest" "$OUT_DIR/result.sha256"
printf 'matched Qwen3.8 result sealed at %s\n' "$OUT_DIR/summary.json"
