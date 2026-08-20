#!/usr/bin/env bash
set -euo pipefail

# Reproducible same-host llama.cpp discriminator for the frozen DeepSeek
# four-agent cold workload. Each wave starts from a fresh model process, lets
# that loaded-but-idle peer return to a continuous Nominal state, and is then
# monitored fail-closed throughout measurement. This is matched-reference
# evidence, not the hf2q release gate's unloaded-host settle contract.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LLAMA_SERVER_BIN=${LLAMA_SERVER_BIN:-/opt/llama.cpp/build/bin/llama-server}
LLAMA_SERVER_SHA256=${LLAMA_SERVER_SHA256:?LLAMA_SERVER_SHA256 is required}
MODEL=${MODEL:-/opt/hf2q/artifacts/DeepSeek-V4-Flash-0731-agentic-q2-repro.gguf}
MODEL_SHA256=${MODEL_SHA256:?MODEL_SHA256 is required}
MODEL_ALIAS=${MODEL_ALIAS:-Deepseek v4 Flash 0731 Source}
WAVES=${WAVES:-2}
PORT=${PORT:-18080}
OUT_ROOT=${OUT_ROOT:-$(mktemp -d /var/tmp/hf2q-deepseek-peer.XXXXXX)}
THERMAL_SETTLE_SECONDS=${THERMAL_SETTLE_SECONDS:-180}
THERMAL_SETTLE_TIMEOUT_SECONDS=${THERMAL_SETTLE_TIMEOUT_SECONDS:-900}
PEER_EXPECTED_PROMPT_TOKENS=${PEER_EXPECTED_PROMPT_TOKENS:-6695}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/macos_thermal_guard.sh
source "$script_dir/macos_thermal_guard.sh"

if [[ ${HF2Q_THERMAL_SWIFTC_BIN+x} || ${HF2Q_THERMAL_PROBE_BIN+x} \
  || ${HF2Q_THERMAL_PROBE_SOURCE+x} ]]; then
  echo "thermal probe overrides are reserved for isolated contract tests" >&2
  exit 2
fi
readonly HF2Q_THERMAL_SWIFTC_BIN=/usr/bin/swiftc
for command in awk caffeinate curl jq lsof pmset rg shasum stat; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 2
  }
done
[[ -x /usr/bin/pgrep && -x "$HF2Q_THERMAL_SWIFTC_BIN" ]] || {
  echo "required system process probe or Swift compiler is unavailable" >&2
  exit 2
}
[[ -x "$LLAMA_SERVER_BIN" ]] || {
  echo "llama-server binary is not executable: $LLAMA_SERVER_BIN" >&2
  exit 2
}
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 2; }
for digest in "$LLAMA_SERVER_SHA256" "$MODEL_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || {
    echo "binary/model SHA-256 values must be lowercase 64-character digests" >&2
    exit 2
  }
done
for setting in WAVES PORT THERMAL_SETTLE_SECONDS THERMAL_SETTLE_TIMEOUT_SECONDS \
  PEER_EXPECTED_PROMPT_TOKENS; do
  value=${!setting}
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "$setting must be a positive integer (got: $value)" >&2
    exit 2
  }
done

mkdir -p "$OUT_ROOT"
parent_pid=$$
server_pid=""
power_pid=""
caffeinate_pid=""
thermal_pid=""
thermal_stop_file=""

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
require_ac() {
  local state
  state=$(pmset -g batt)
  printf 'sample_utc=%s\n%s\n' "$(date -u +%FT%TZ)" "$state" \
    >>"$OUT_ROOT/power.log"
  rg -q "Now drawing from 'AC Power'" <<<"$state" || {
    echo "matched peer gate requires continuous AC power" >&2
    return 1
  }
}
ensure_guard_health() {
  local assertions=""
  require_ac
  [[ ! -e "$OUT_ROOT/power-failure.txt" ]] || {
    cat "$OUT_ROOT/power-failure.txt" >&2
    return 1
  }
  if [[ -z "$power_pid" ]] || ! kill -0 "$power_pid" 2>/dev/null; then
    echo "AC power monitor is not running" >&2
    return 1
  fi
  if [[ -z "$caffeinate_pid" ]] || ! kill -0 "$caffeinate_pid" 2>/dev/null; then
    echo "caffeinate guard is not running" >&2
    return 1
  fi
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    assertions=$(pmset -g assertions)
    if rg -q "pid ${caffeinate_pid}\\(caffeinate\\):" <<<"$assertions"; then
      printf '%s\n' "$assertions" >"$OUT_ROOT/power-assertions.current.txt"
      return 0
    fi
    sleep 1
  done
  echo "caffeinate process has no visible power assertion" >&2
  return 1
}
require_no_model_runtime() {
  local name
  for name in hf2q llama-server llama-cli; do
    if /usr/bin/pgrep -x "$name" >/dev/null 2>&1; then
      echo "matched peer wave requires no existing $name runtime" >&2
      /usr/bin/pgrep -flx "$name" >&2 || true
      return 1
    fi
  done
}
stop_server() {
  local deadline
  if [[ -z "$server_pid" ]] || ! kill -0 "$server_pid" 2>/dev/null; then
    server_pid=""
    return 0
  fi
  kill -INT "$server_pid" 2>/dev/null || true
  deadline=$((SECONDS + 180))
  while kill -0 "$server_pid" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 1
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    echo "llama-server did not stop within 180 seconds" >&2
    kill -KILL "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
    return 1
  fi
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
cleanup() {
  local cleanup_rc=0
  if [[ -n "$thermal_stop_file" ]]; then : >"$thermal_stop_file"; fi
  if [[ -n "$thermal_pid" ]]; then
    kill -TERM "$thermal_pid" 2>/dev/null || true
    wait "$thermal_pid" 2>/dev/null || true
  fi
  stop_server || cleanup_rc=1
  if [[ -n "$power_pid" ]]; then
    kill -TERM "$power_pid" 2>/dev/null || true
    wait "$power_pid" 2>/dev/null || true
  fi
  if [[ -n "$caffeinate_pid" ]]; then
    kill -TERM "$caffeinate_pid" 2>/dev/null || true
    wait "$caffeinate_pid" 2>/dev/null || true
  fi
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

actual_binary_sha=$(sha256_file "$LLAMA_SERVER_BIN")
[[ "$actual_binary_sha" == "$LLAMA_SERVER_SHA256" ]] || {
  echo "llama-server SHA-256 mismatch" >&2
  exit 2
}
actual_model_sha=$(sha256_file "$MODEL")
[[ "$actual_model_sha" == "$MODEL_SHA256" ]] || {
  echo "model SHA-256 mismatch" >&2
  exit 2
}
"$LLAMA_SERVER_BIN" --version >"$OUT_ROOT/llama-version.txt"
printf '%s  %s\n' "$actual_binary_sha" "$LLAMA_SERVER_BIN" \
  >"$OUT_ROOT/llama-server.sha256"
printf '%s  %s\n' "$actual_model_sha" "$MODEL" >"$OUT_ROOT/model.sha256"
pmset -g assertions >"$OUT_ROOT/power-assertions.before.txt"
vm_stat >"$OUT_ROOT/vm-stat.before.txt"
sysctl vm.swapusage >"$OUT_ROOT/swap.before.txt"

require_ac
caffeinate -dimsu -w "$parent_pid" &
caffeinate_pid=$!
(
  while kill -0 "$parent_pid" 2>/dev/null; do
    if ! require_ac; then
      printf 'AC power lost at %s\n' "$(date -u +%FT%TZ)" \
        >"$OUT_ROOT/power-failure.txt"
      kill -TERM "$parent_pid" 2>/dev/null || true
      exit 1
    fi
    sleep 5
  done
) &
power_pid=$!
ensure_guard_health

wait_ready() {
  local log=$1
  local deadline=$((SECONDS + 360))
  local status
  while ((SECONDS < deadline)); do
    status=$(curl --silent --show-error --max-time 2 -o /dev/null \
      -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
    if [[ "$status" == 200 ]]; then return 0; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "llama-server exited before readiness" >&2
      sed -n '1,240p' "$log" >&2
      return 1
    fi
    sleep 1
  done
  echo "llama-server did not become ready within 360 seconds" >&2
  sed -n '1,240p' "$log" >&2
  return 1
}

wave_envelopes=()
for ((wave = 1; wave <= WAVES; wave++)); do
  out="$OUT_ROOT/wave-$wave"
  thermal_dir="$out/thermal"
  settle_log="$thermal_dir/settle.log"
  measurement_log="$thermal_dir/measurement.log"
  thermal_summary="$thermal_dir/summary.json"
  server_log="$out/server.log"
  mkdir -p "$thermal_dir"
  require_no_model_runtime
  ensure_guard_health
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | rg -q .; then
    echo "matched peer port is already in use: $PORT" >&2
    exit 1
  fi
  "$LLAMA_SERVER_BIN" --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --ctx-size 131072 --parallel 4 --kv-unified --gpu-layers all \
    --flash-attn on --jinja --alias "$MODEL_ALIAS" --no-cache-prompt \
    --no-context-shift >"$server_log" 2>&1 &
  server_pid=$!
  wait_ready "$server_log"
  thermal_wait_for_nominal "$settle_log" \
    "llama-wave-${wave}-loaded-idle-settle" \
    "$THERMAL_SETTLE_SECONDS" "$THERMAL_SETTLE_TIMEOUT_SECONDS" 5

  thermal_stop_file="$thermal_dir/stop"
  rm -f "$thermal_stop_file"
  : >"$measurement_log"
  thermal_sample "$measurement_log" "llama-wave-${wave}-measurement-start"
  [[ "$THERMAL_STATE" == nominal ]]
  thermal_monitor_nominal "$measurement_log" \
    "llama-wave-${wave}-measurement" "$thermal_stop_file" 2 &
  thermal_pid=$!

  set +e
  BASE_URL="http://127.0.0.1:$PORT" MODEL="$MODEL_ALIAS" AGENTS=4 \
  WAVE_ID="wave-$wave" EXPECTED_PROMPT_TOKENS="$PEER_EXPECTED_PROMPT_TOKENS" \
  OUT_DIR="$out/agents" CURL_MAX_TIME_SECONDS=120 \
    "$ROOT_DIR/scripts/test_deepseek4_peer_cold_wave.sh" \
      >"$out/summary.stdout" 2>"$out/harness.stderr"
  harness_rc=$?
  set -e

  : >"$thermal_stop_file"
  thermal_rc=0
  if ! wait "$thermal_pid"; then thermal_rc=1; fi
  if ! thermal_sample "$measurement_log" "llama-wave-${wave}-measurement-end" \
    || [[ "$THERMAL_STATE" != nominal ]]; then
    thermal_rc=1
  fi
  thermal_pid=""
  thermal_stop_file=""
  ((thermal_rc == 0)) || {
    echo "llama.cpp wave $wave thermal monitor failed" >&2
    exit 1
  }
  ((harness_rc == 0)) || {
    cat "$out/harness.stderr" >&2
    exit "$harness_rc"
  }
  thermal_validate_measurement_log "$measurement_log" 5
  measurement_samples=$THERMAL_LOG_SAMPLES
  measurement_duration=$THERMAL_LOG_DURATION_SECONDS
  thermal_validate_settle_log "$settle_log" "$THERMAL_SETTLE_SECONDS" 8
  settle_samples=$THERMAL_LOG_SAMPLES
  settle_duration=$THERMAL_LOG_DURATION_SECONDS
  jq -n --arg status pass --arg phase "llama-wave-$wave" \
    --arg settle_log_sha256 "$(sha256_file "$settle_log")" \
    --arg measurement_log_sha256 "$(sha256_file "$measurement_log")" \
    --argjson settle_seconds "$THERMAL_SETTLE_SECONDS" \
    --argjson settle_duration_seconds "$settle_duration" \
    --argjson settle_samples "$settle_samples" \
    --argjson measurement_duration_seconds "$measurement_duration" \
    --argjson measurement_samples "$measurement_samples" \
    '{status:$status,phase:$phase,required_state:"nominal",
      settle_runtime:"loaded-idle-no-requests",
      settle_seconds:$settle_seconds,
      settle_duration_seconds:$settle_duration_seconds,
      settle_samples:$settle_samples,
      measurement_duration_seconds:$measurement_duration_seconds,
      measurement_samples:$measurement_samples,
      settle_log_sha256:$settle_log_sha256,
      measurement_log_sha256:$measurement_log_sha256}' >"$thermal_summary.tmp"
  mv "$thermal_summary.tmp" "$thermal_summary"
  stop_server
  rg -n 'error:|fatal|Metal.*error|out of memory' "$server_log" \
    >"$out/fatal.log" || true
  [[ ! -s "$out/fatal.log" ]] || {
    cat "$out/fatal.log" >&2
    exit 1
  }
  jq -n --arg status pass --argjson wave "$wave" \
    --arg binary_sha256 "$actual_binary_sha" \
    --arg model_sha256 "$actual_model_sha" \
    --arg server_log_sha256 "$(sha256_file "$server_log")" \
    --slurpfile receipt "$out/agents/summary.json" \
    --slurpfile thermal "$thermal_summary" \
    '{status:$status,wave:$wave,binary_sha256:$binary_sha256,
      model_sha256:$model_sha256,server_log_sha256:$server_log_sha256,
      flags:["--ctx-size","131072","--parallel","4","--kv-unified",
        "--gpu-layers","all","--flash-attn","on","--jinja",
        "--no-cache-prompt","--no-context-shift"],
      receipt:$receipt[0],thermal:$thermal[0]}' >"$out/envelope.json.tmp"
  mv "$out/envelope.json.tmp" "$out/envelope.json"
  wave_envelopes+=("$out/envelope.json")
  ensure_guard_health
done

jq -s --arg status pass --arg runtime llama.cpp \
  --arg binary_sha256 "$actual_binary_sha" --arg model_sha256 "$actual_model_sha" \
  --argjson waves "$WAVES" '
  if length != $waves or any(.[]; .status != "pass") then
    error("one or more matched peer waves did not pass")
  else {
    status:$status,runtime:$runtime,binary_sha256:$binary_sha256,
    model_sha256:$model_sha256,waves:$waves,
    cohort_cold_wall_ms:map(.receipt.cohort_cold_wall_ms),
    median_cohort_cold_wall_ms:(map(.receipt.cohort_cold_wall_ms)|sort
      | if length % 2 == 1 then .[length/2|floor]
        else ((.[length/2-1] + .[length/2]) / 2) end),
    receipts:.
  } end
' "${wave_envelopes[@]}" >"$OUT_ROOT/manifest.json.tmp"
mv "$OUT_ROOT/manifest.json.tmp" "$OUT_ROOT/manifest.json"
printf '%s  %s\n' "$(sha256_file "$OUT_ROOT/manifest.json")" \
  "$OUT_ROOT/manifest.json" >"$OUT_ROOT/manifest.json.sha256"
cat "$OUT_ROOT/manifest.json"
echo "matched peer evidence: $OUT_ROOT" >&2
